// llama.cpp's llama-server: an HTTP server with llama.cpp's and OpenAI's
// APIs for one model. HTTP is goose-http's, through Plod: its worker
// threads take the connections and answer /health and CORS preflights; the
// main thread loads the model and serves the other requests, which the
// workers send it as Plod jobs. One slot: they are served one at a time, and the KV cache keeps
// the last prompt so the next one reuses what they share. A client that
// goes away stops its generation.
//   server -m model.gguf [--host 127.0.0.1] [--port 8080] [-c n_ctx] [-b n_batch] [-t threads]
//          [-n n_predict] [-a alias] [--api-key key] [--jinja | --no-jinja]
//          [--chat-template tmpl] [--chat-template-file file] [--reasoning-format fmt]
//          [-s seed] [--temp t] [--top-k k] [--top-p p] [--min-p p] [...the sampling flags of completion]
// Routes: /health, /props, /models, /completion(s), /v1/completions,
// /v1/chat/completions, /apply-template, /tokenize, /detokenize.
// port-of: tools/server/server.cpp tools/server/server-context.cpp tools/server/server-task.cpp
// port-of: tools/server/server-common.cpp tools/server/server-schema.cpp tools/server/server-chat.cpp
import std;
import os;
import exception;
import ggml from "../ggml";
import ggml_cpu from "../ggml/cpu";
import ggml_shim from "../ggml/shim";
import ggml.gguf;
import llama from "../llama";
import llama.vocab;
import llama.model;
import llama.context;
import llama.sampler;
import llama.grammar;
import common from "../common";
import common.jinja;
import common.console;
import common.jdoc;
import common.json_schema;
import common.chat_render;
import common.chat_peg;
import common.chat_apply;
import net from "./net";
import http from "github.com/kaitk/goose-http@v0.5.16";
import http.shared;
import plod from "github.com/kaitk/plod@067b57c804435ea9479d58ba55e3cc103788a151";

let BUILD_INFO = "b0-goose";

// task_response_type
let RES_NONE = 0;
let RES_OAI_CMPL = 1;
let RES_OAI_CHAT = 2;

// stop_type
let STOP_NONE = 0;
let STOP_EOS = 1;
let STOP_WORD = 2;
let STOP_LIMIT = 3;

// error_type
let E_INVALID_REQUEST = 0;
let E_AUTHENTICATION = 1;
let E_NOT_FOUND = 2;
let E_SERVER = 3;
let E_PERMISSION = 4;
let E_NOT_SUPPORTED = 5;
let E_UNAVAILABLE = 6;
let E_EXCEED_CONTEXT = 7;

fn die(msg: const u8[:]) {
    write_stderr(str("error: ", msg, "\n"));
    exit(1);
}

// The log, on stdout, flushed line by line.
fn log(msg: const u8[:]) { write_stdout(msg); write_stdout("\n"); flush_stdout(); }

//
// the model and the server's settings
//

var m = model::Model { .. };
var c = context::Context { .. };
var ch = sampler::Chain { .. };
var base_sp = sampler::Params { .. };   // the sampling defaults, the command line's
var base_n_predict = -1;
var base_n_keep = 0;
var base_stops: u8[>..] = [];           // --reverse-prompt, each ended by a NUL
var n_batch = 2048;
var model_path: u8[>..] = [];
var model_name: u8[>..] = [];
var api_key: u8[>..] = [];
var static_dir: u8[>..] = [];           // --path: the files served under /, such as the web UI
var use_jinja = true;
var reasoning_format = chat_render::REASONING_DEEPSEEK;
var enable_thinking = true;
var prefill_assistant = true;
var ts = chat_apply::Templates { .. };
var tmpl_src: u8[>..] = [];             // the chat templates' sources, for /props
var tmpl_tool_src: u8[>..] = [];
var bos_text: u8[>..] = [];
var eos_text: u8[>..] = [];
var ftype_name: u8[>..] = [];
var n_params = 0;
var model_size = 0;

// The tokens in the KV cache, cell by cell (the slot's prompt).
var cached: i32[>..] = [];

//
// errors: what a request fails with, thrown to the route's handler
//

struct Fail { kind: i64 }
var fail_msg: u8[>..] = [];

fn fail(kind: i64, msg: const u8[:]) {
    fail_msg.clear();
    fail_msg.append(msg);
    exception::throw(Fail { kind: kind });
}

fn error_code(kind: i64) -> i64 {
    return match kind {
        E_INVALID_REQUEST => 400, E_AUTHENTICATION => 401, E_NOT_FOUND => 404, E_SERVER => 500,
        E_PERMISSION => 403, E_NOT_SUPPORTED => 501, E_UNAVAILABLE => 503, E_EXCEED_CONTEXT => 400, _ => 500,
    };
}

fn error_type_name(kind: i64) -> const u8[:] {
    return match kind {
        E_INVALID_REQUEST => "invalid_request_error", E_AUTHENTICATION => "authentication_error",
        E_NOT_FOUND => "not_found_error", E_SERVER => "server_error", E_PERMISSION => "permission_error",
        E_NOT_SUPPORTED => "not_supported_error", E_UNAVAILABLE => "unavailable_error",
        E_EXCEED_CONTEXT => "exceed_context_size_error", _ => "server_error",
    };
}

// format_error_response
fn error_json(out: u8[>..]&, kind: i64, msg: const u8[:]) {
    out.append("{\"code\":");
    jdoc::put_int(out, error_code(kind));
    out.append(",\"message\":");
    jdoc::put_str(out, msg);
    out.append(",\"type\":");
    jdoc::put_str(out, error_type_name(kind));
    out.push('}');
}

//
// writing JSON: a member or an element is preceded by a comma unless it
// is the first in its container
//

fn sep(o: u8[>..]&) {
    if o.len > 0 && o[o.len - 1] != '{' && o[o.len - 1] != '[' && o[o.len - 1] != ':' { o.push(','); }
}
fn k(o: u8[>..]&, name: const u8[:]) { sep(o); jdoc::put_str(o, name); o.push(':'); }
fn ks(o: u8[>..]&, name: const u8[:], v: const u8[:]) { k(o, name); jdoc::put_str(o, v); }
fn ki(o: u8[>..]&, name: const u8[:], v: i64) { k(o, name); jdoc::put_int(o, v); }
fn kf(o: u8[>..]&, name: const u8[:], v: f32) { k(o, name); jdoc::put_f32(o, v); }
fn kd(o: u8[>..]&, name: const u8[:], v: f64) { k(o, name); jdoc::put_num(o, v); }
fn kb(o: u8[>..]&, name: const u8[:], v: bool) { k(o, name); jdoc::put_bool(o, v); }
fn kraw(o: u8[>..]&, name: const u8[:], v: const u8[:]) { k(o, name); o.append(v); }
fn es(o: u8[>..]&, v: const u8[:]) { sep(o); jdoc::put_str(o, v); }
fn ei(o: u8[>..]&, v: i64) { sep(o); jdoc::put_int(o, v); }

// A list of strings, each ended by a NUL, as a JSON array.
fn str_list(o: u8[>..]&, list: const u8[:]) {
    o.push('[');
    var start = 0;
    for b, i in list {
        if b == 0 { es(o, list[start..i]); start = i + 1; }
    }
    o.push(']');
}

fn list_count(list: const u8[:]) -> i64 {
    var n = 0;
    for b in list { if b == 0 { n++; } }
    return n;
}

//
// tokens
//

fn piece(t: i32, special: bool) -> u8[>..] {
    var out: u8[>..] = [];
    vocab::token_to_piece(t, out, 0, special);
    return out;
}

// common_detokenize(tokens, special)
fn detokenize(tokens: const i32[:], special: bool) -> u8[>..] {
    var out: u8[>..] = [];
    vocab::detokenize(tokens, false, special, out);
    return out;
}

// validate_utf8: the length of text without an incomplete character at its end.
fn utf8_complete(text: const u8[:]) -> i64 {
    let len = text.len;
    for i in 1..min(4, len) + 1 {
        let ch0 = text[len - i];
        if (ch0 & 0xE0) == 0xC0 { if i < 2 { return len - i; } }
        else if (ch0 & 0xF0) == 0xE0 { if i < 3 { return len - i; } }
        else if (ch0 & 0xF8) == 0xF0 { if i < 4 { return len - i; } }
    }
    return len;
}

// tokenize_mixed: a string, or an array of strings and token ids; the
// special tokens added before the first element only.
fn tokenize_mixed(v: i64, add_special: bool, parse_special: bool, out: i32[>..]&) {
    if jdoc::is_str(v) {
        vocab::tokenize(jdoc::str_of(v), add_special, parse_special, out);
        return;
    }
    guard jdoc::is_arr(v) else { fail(E_INVALID_REQUEST, "\"prompt\" elements must be a string, a list of tokens, a JSON object containing a prompt string, or a list of mixed strings & tokens."); }
    var first = true;
    for i in jdoc::len(v) {
        let e = jdoc::at(v, i);
        if jdoc::is_str(e) {
            var part: i32[>..] = [];
            vocab::tokenize(jdoc::str_of(e), first && add_special, parse_special, part);
            out.append(part);
        } else if jdoc::is_int(e) {
            out.push(jdoc::int_of(e) as! i32);
        } else {
            fail(E_INVALID_REQUEST, "\"prompt\" elements must be a string, a list of tokens, a JSON object containing a prompt string, or a list of mixed strings & tokens.");
        }
        first = false;
    }
}

//
// a task's parameters (task_params), from the request (server-schema.cpp)
//

struct LogitBias { token: i32, bias: f32 }

var t_res_type = RES_NONE;
var t_stream = false;
var t_include_usage = false;
var t_timings_per_token = false;
var t_return_tokens = false;
var t_cache_prompt = true;
var t_n_predict = -1;
var t_n_keep = 0;
var t_n_indent = 0;
var t_max_predict_ms = -1;
var t_sp = sampler::Params { .. };
var t_ignore_eos = false;
var t_stops: u8[>..] = [];              // each ended by a NUL
var t_logit_bias: LogitBias[>..] = [];
var t_grammar: u8[>..] = [];
var t_grammar_lazy = false;
var t_grammar_user = false;           // the request's own "grammar": not prefilled
var t_trig_types: i64[>..] = [];
var t_trig_values: u8[>..] = [];        // each ended by a NUL
var t_trig_tokens: i32[>..] = [];       // the token of a TOKEN trigger, else -1
var t_preserved: i32[>..] = [];         // sorted
var t_chat = false;                     // the output is a chat message, parsed
var t_chat_format = chat_render::FORMAT_CONTENT_ONLY;
var t_reasoning_format = chat_render::REASONING_NONE;
var t_gen_prompt: u8[>..] = [];
var t_cmpl_id: u8[>..] = [];

fn reset_task(res_type: i64) {
    t_res_type = res_type;
    t_stream = false;
    t_include_usage = false;
    t_timings_per_token = false;
    t_return_tokens = false;
    t_cache_prompt = true;
    t_n_predict = base_n_predict;
    t_n_keep = base_n_keep;
    t_n_indent = 0;
    t_max_predict_ms = -1;
    t_sp = base_sp;
    t_ignore_eos = false;
    t_stops.clear();
    t_stops.append(base_stops);
    t_logit_bias.clear();
    t_grammar.clear();
    t_grammar_lazy = false;
    t_grammar_user = false;
    t_trig_types.clear();
    t_trig_values.clear();
    t_trig_tokens.clear();
    t_preserved.clear();
    t_chat = false;
    t_chat_format = chat_render::FORMAT_CONTENT_ONLY;
    t_reasoning_format = reasoning_format;
    t_gen_prompt.clear();
    t_cmpl_id.clear();
}

// The value of the first of the names the request has (not null), or -1.
fn field(data: i64, a: const u8[:], b: const u8[:] = "", c2: const u8[:] = "") -> i64, const u8[:] {
    if jdoc::has(data, a) { return jdoc::get(data, a), a; }
    if b.len > 0 && jdoc::has(data, b) { return jdoc::get(data, b), b; }
    if c2.len > 0 && jdoc::has(data, c2) { return jdoc::get(data, c2), c2; }
    return -1, "";
}

fn field_fail(name: const u8[:], msg: const u8[:]) { fail(E_INVALID_REQUEST, str("Field '", name, "': ", msg)); }

fn want_num(v: i64, name: const u8[:]) {
    if !jdoc::is_num(v) { field_fail(name, str("[json.exception.type_error.302] type must be number, but is ", kind_name(v))); }
}

fn kind_name(v: i64) -> const u8[:] {
    return match jdoc::kind(v) {
        jdoc::NULL => "null", jdoc::BOOL => "boolean", jdoc::INT => "number", jdoc::FLOAT => "number",
        jdoc::STRING => "string", jdoc::ARRAY => "array", jdoc::OBJECT => "object", _ => "null",
    };
}

// field_num with soft limits: clamped
fn num_i(data: i64, cur: i64, lo: i64, hi: i64, a: const u8[:], b: const u8[:] = "", c2: const u8[:] = "") -> i64 {
    let v, name = field(data, a, b, c2);
    guard v >= 0 else { return cur; }
    want_num(v, name);
    return max(lo, min(hi, jdoc::int_of(v)));
}

// field_num with hard limits: out of them is an error
fn num_i_hard(data: i64, cur: i64, lo: i64, hi: i64, a: const u8[:], b: const u8[:] = "", c2: const u8[:] = "") -> i64 {
    let v, name = field(data, a, b, c2);
    guard v >= 0 else { return cur; }
    want_num(v, name);
    let x = jdoc::int_of(v);
    if x < lo || x > hi { field_fail(name, str("Value must be between ", lo, " <= value <= ", hi, ", but got ", x)); }
    return x;
}

fn num_f(data: i64, cur: f32, lo: f32, hi: f32, a: const u8[:]) -> f32 {
    let v, name = field(data, a);
    guard v >= 0 else { return cur; }
    want_num(v, name);
    let x: f32 = jdoc::num_of(v) as f32;
    return max(lo, min(hi, x));
}

fn flag(data: i64, cur: bool, a: const u8[:]) -> bool {
    let v, name = field(data, a);
    guard v >= 0 else { return cur; }
    if !jdoc::is_bool(v) { field_fail(name, str("[json.exception.type_error.302] type must be boolean, but is ", kind_name(v))); }
    return jdoc::bool_of(v);
}

let F_MAX: f32 = 3.4028234663852886e38;
let F_INF: f32 = 1.0 / 0.0;
let I32_MAX = 2147483647;

// eval_llama_cmpl_schema: the task's parameters from the request, on the
// server's defaults.
fn eval_params(data: i64) {
    t_stream = flag(data, t_stream, "stream");
    t_timings_per_token = flag(data, t_timings_per_token, "timings_per_token");
    let so = jdoc::get(data, "stream_options");
    if jdoc::is_obj(so) { t_include_usage = flag(so, t_include_usage, "include_usage"); }
    t_cache_prompt = flag(data, t_cache_prompt, "cache_prompt");
    t_return_tokens = flag(data, t_return_tokens, "return_tokens");
    t_n_predict = num_i_hard(data, t_n_predict, -1, I32_MAX, "n_predict", "max_completion_tokens", "max_tokens");
    t_n_indent = num_i_hard(data, t_n_indent, 0, I32_MAX, "n_indent");
    t_n_keep = num_i_hard(data, t_n_keep, -1, I32_MAX, "n_keep");
    t_max_predict_ms = num_i_hard(data, t_max_predict_ms, -1, 9223372036854775807, "t_max_predict_ms");
    {
        let v, name = field(data, "n_cmpl", "n");
        if v >= 0 {
            want_num(v, name);
            if jdoc::int_of(v) != 1 { field_fail(name, str("Value must be between 1 <= value <= 1, but got ", jdoc::int_of(v))); }
        }
    }

    t_sp.top_k = num_i(data, t_sp.top_k, 0, I32_MAX, "top_k");
    t_sp.top_p = num_f(data, t_sp.top_p, 0.0, 1.0, "top_p");
    t_sp.min_p = num_f(data, t_sp.min_p, 0.0, 1.0, "min_p");
    t_sp.top_n_sigma = num_f(data, t_sp.top_n_sigma, -F_MAX, F_MAX, "top_n_sigma");
    t_sp.xtc_probability = num_f(data, t_sp.xtc_probability, 0.0, 1.0, "xtc_probability");
    t_sp.xtc_threshold = num_f(data, t_sp.xtc_threshold, 0.0, 1.0, "xtc_threshold");
    t_sp.typ_p = num_f(data, t_sp.typ_p, -F_MAX, F_MAX, "typical_p");
    t_sp.temp = num_f(data, t_sp.temp, 0.0, F_INF, "temperature");
    t_sp.dynatemp_range = num_f(data, t_sp.dynatemp_range, -F_MAX, F_MAX, "dynatemp_range");
    t_sp.dynatemp_exponent = num_f(data, t_sp.dynatemp_exponent, -F_MAX, F_MAX, "dynatemp_exponent");
    t_sp.penalty_last_n = num_i_hard(data, t_sp.penalty_last_n, 0, I32_MAX, "repeat_last_n");
    t_sp.penalty_repeat = num_f(data, t_sp.penalty_repeat, -F_MAX, F_MAX, "repeat_penalty");
    t_sp.penalty_freq = num_f(data, t_sp.penalty_freq, -F_MAX, F_MAX, "frequency_penalty");
    t_sp.penalty_present = num_f(data, t_sp.penalty_present, -F_MAX, F_MAX, "presence_penalty");
    t_sp.dry_multiplier = num_f(data, t_sp.dry_multiplier, -F_MAX, F_MAX, "dry_multiplier");
    {
        let v, name = field(data, "dry_base");
        if v >= 0 {
            want_num(v, name);
            let x: f32 = jdoc::num_of(v) as f32;
            t_sp.dry_base = if x < 1.0 { base_sp.dry_base } else { x };
        }
    }
    t_sp.dry_allowed_length = num_i_hard(data, t_sp.dry_allowed_length, 0, I32_MAX, "dry_allowed_length");
    t_sp.dry_penalty_last_n = num_i_hard(data, t_sp.dry_penalty_last_n, 0, I32_MAX, "dry_penalty_last_n");
    t_sp.mirostat = num_i(data, t_sp.mirostat, 0, 2, "mirostat");
    t_sp.mirostat_tau = num_f(data, t_sp.mirostat_tau, -F_MAX, F_MAX, "mirostat_tau");
    t_sp.mirostat_eta = num_f(data, t_sp.mirostat_eta, -F_MAX, F_MAX, "mirostat_eta");
    {
        let v, name = field(data, "seed");
        if v >= 0 {
            want_num(v, name);
            t_sp.seed = (jdoc::int_of(v) & 0xFFFFFFFF) as! u32;
        }
    }
    t_sp.n_probs = num_i(data, t_sp.n_probs, -I32_MAX, I32_MAX, "n_probs", "logprobs");
    if t_sp.n_probs > 0 { fail(E_NOT_SUPPORTED, "n_probs: token probabilities are not supported yet"); }
    t_sp.min_keep = num_i_hard(data, t_sp.min_keep, 0, I32_MAX, "min_keep");

    {
        let v, name = field(data, "dry_sequence_breakers");
        if v >= 0 {
            guard jdoc::is_arr(v) && jdoc::len(v) > 0 else { field_fail(name, "Error: dry_sequence_breakers must be a non-empty array of strings"); }
            t_sp.dry_breakers.clear();
            var seen: u8[>..] = [];
            for i in jdoc::len(v) {
                let e = jdoc::at(v, i);
                guard jdoc::is_str(e) else { field_fail(name, "Error: dry_sequence_breakers must be a non-empty array of strings"); }
                let s = jdoc::str_of(e);
                if list_has(seen, s) { continue; }
                seen.append(s);
                seen.push(0);
            }
            if list_count(seen) > 128 { field_fail(name, "Error: dry_sequence_breakers must contain at most 128 distinct entries"); }
            if seen.len > t_sp.dry_breakers.cap { field_fail(name, "Error: the DRY sequence breakers are too long"); }
            t_sp.dry_breakers.append(seen);
        }
    }

    // json_schema, or grammar
    {
        let v, name = field(data, "json_schema", "grammar");
        if v >= 0 {
            if jdoc::contains(data, "json_schema") && !jdoc::contains(data, "grammar") {
                var schema: u8[>..] = [];
                if jdoc::is_obj(v) && jdoc::len(v) == 0 { schema.append("{\"type\":\"object\"}"); } else { jdoc::dump(schema, v); }
                t_grammar.clear();
                if !json_schema::to_grammar(schema, t_grammar) { field_fail(name, str("\"json_schema\": ", json_schema::err)); }
            } else {
                let g = jdoc::get_str(data, "grammar", "");
                if g.len > 0 { t_grammar.clear(); t_grammar.append(g); t_grammar_user = true; }
            }
        }
    }
    t_grammar_lazy = flag(data, t_grammar_lazy, "grammar_lazy");

    {
        let v, name = field(data, "reasoning_format");
        if v >= 0 { t_reasoning_format = reasoning_format_from_name(jdoc::str_of(v)); }
    }

    {
        let v, name = field(data, "preserved_tokens");
        if v >= 0 {
            for i in jdoc::len(v) { add_preserved(jdoc::str_of(jdoc::at(v, i))); }
        }
    }
    {
        let v, name = field(data, "grammar_triggers");
        if v >= 0 {
            for i in jdoc::len(v) {
                let t = jdoc::at(v, i);
                add_trigger(jdoc::get_int(t, "type", chat_render::TRIGGER_WORD), jdoc::get_str(t, "value", ""));
            }
        }
    }

    {
        let v, name = field(data, "logit_bias");
        if v >= 0 {
            t_logit_bias.clear();
            let n_vocab = vocab::n_tokens();
            fn bias_of(b: i64) -> f32, bool {
                if jdoc::is_num(b) { return jdoc::num_of(b) as f32, true; }
                if jdoc::is_bool(b) && !jdoc::bool_of(b) { return -F_INF, true; }
                return 0.0, false;
            }
            fn add_text(text: const u8[:], bias: f32) {
                var ids: i32[>..] = [];
                vocab::tokenize(text, false, false, ids);
                for id in ids { t_logit_bias.push(LogitBias { token: id, bias: bias }); }
            }
            if jdoc::is_arr(v) {
                for i in jdoc::len(v) {
                    let e = jdoc::at(v, i);
                    guard jdoc::is_arr(e) && jdoc::len(e) == 2 else { continue; }
                    let bias, ok = bias_of(jdoc::at(e, 1));
                    guard ok else { continue; }
                    let t = jdoc::at(e, 0);
                    if jdoc::is_int(t) {
                        let tok = jdoc::int_of(t);
                        if tok >= 0 && tok < n_vocab { t_logit_bias.push(LogitBias { token: tok as! i32, bias: bias }); }
                    } else if jdoc::is_str(t) {
                        add_text(jdoc::str_of(t), bias);
                    }
                }
            } else if jdoc::is_obj(v) {
                for i in jdoc::len(v) {
                    let bias, ok = bias_of(jdoc::at(v, i));
                    guard ok else { continue; }
                    let key = jdoc::key(v, i);
                    let tok, is_num = parse_int(key);
                    if is_num {
                        if tok >= 0 && tok < n_vocab { t_logit_bias.push(LogitBias { token: tok as! i32, bias: bias }); }
                    } else {
                        add_text(key, bias);
                    }
                }
            }
        }
    }

    t_ignore_eos = flag(data, t_ignore_eos, "ignore_eos");

    {
        let v, name = field(data, "stop");
        if v >= 0 {
            t_stops.clear();
            if jdoc::is_arr(v) {
                for i in jdoc::len(v) {
                    let w = jdoc::str_of(jdoc::at(v, i));
                    if w.len > 0 { t_stops.append(w); t_stops.push(0); }
                }
            } else if jdoc::is_str(v) {
                t_stops.append(jdoc::str_of(v));
                t_stops.push(0);
            }
            if t_stops.len == 0 { t_stops.append(base_stops); }
        }
    }

    {
        let v, name = field(data, "samplers");
        if v >= 0 {
            if jdoc::is_arr(v) {
                t_sp.order.clear();
                for i in jdoc::len(v) {
                    let kind = sampler::kind_from_name(jdoc::str_of(jdoc::at(v, i)));
                    if kind >= 0 && t_sp.order.len < t_sp.order.cap { t_sp.order.push(kind); }
                }
            } else if jdoc::is_str(v) {
                t_sp.order.clear();
                for ch0 in jdoc::str_of(v) {
                    let kind = kind_from_char(ch0);
                    if kind >= 0 && t_sp.order.len < t_sp.order.cap { t_sp.order.push(kind); }
                }
            }
        }
    }
}

fn list_has(list: const u8[:], s: const u8[:]) -> bool {
    var start = 0;
    for b, i in list {
        if b == 0 {
            if list[start..i] == s { return true; }
            start = i + 1;
        }
    }
    return false;
}

// common_sampler_types_from_chars
fn kind_from_char(c0: u8) -> i64 {
    return match c0 {
        'd' => sampler::KIND_DRY, 'k' => sampler::KIND_TOP_K, 'y' => sampler::KIND_TYPICAL, 'p' => sampler::KIND_TOP_P,
        's' => sampler::KIND_TOP_N_SIGMA, 'm' => sampler::KIND_MIN_P, 't' => sampler::KIND_TEMP_EXT, 'x' => sampler::KIND_XTC,
        'e' => sampler::KIND_PENALTIES, _ => -1,
    };
}

// common_sampler_type_to_str
fn kind_name_of(kind: i64) -> const u8[:] {
    return match kind {
        sampler::KIND_DRY => "dry", sampler::KIND_TOP_K => "top_k", sampler::KIND_TYPICAL => "typ_p",
        sampler::KIND_TOP_P => "top_p", sampler::KIND_TOP_N_SIGMA => "top_n_sigma", sampler::KIND_MIN_P => "min_p",
        sampler::KIND_XTC => "xtc", sampler::KIND_TEMP_EXT => "temperature", sampler::KIND_PENALTIES => "penalties",
        sampler::KIND_INFILL => "infill", sampler::KIND_ADAPTIVE_P => "adaptive_p", _ => "",
    };
}

fn reasoning_format_from_name(name: const u8[:]) -> i64 {
    if name == "none" { return chat_render::REASONING_NONE; }
    if name == "auto" { return chat_render::REASONING_AUTO; }
    if name == "deepseek" { return chat_render::REASONING_DEEPSEEK; }
    if name == "deepseek-legacy" { return chat_render::REASONING_DEEPSEEK_LEGACY; }
    fail(E_INVALID_REQUEST, str("Unknown reasoning format: ", name));
    return 0;
}

fn reasoning_format_name(f: i64) -> const u8[:] {
    return match f {
        chat_render::REASONING_NONE => "none", chat_render::REASONING_AUTO => "auto",
        chat_render::REASONING_DEEPSEEK => "deepseek", chat_render::REASONING_DEEPSEEK_LEGACY => "deepseek-legacy", _ => "unknown",
    };
}

// common_sampling_add_preserved_tokens: the text, when it is one token.
fn add_preserved(text: const u8[:]) {
    var ids: i32[>..] = [];
    vocab::tokenize(text, false, true, ids);
    guard ids.len == 1 else { return; }
    let at = lower_bound(t_preserved) { it < ids[0] };
    if at < t_preserved.len && t_preserved[at] == ids[0] { return; }
    t_preserved.push(ids[0]);
    var j = t_preserved.len - 1;
    while j > at { t_preserved[j] = t_preserved[j - 1]; j--; }
    t_preserved[at] = ids[0];
}

fn is_preserved(t: i32) -> bool {
    let at = lower_bound(t_preserved) { it < t };
    return at < t_preserved.len && t_preserved[at] == t;
}

// common_sampling_add_grammar_triggers: a word that is one token, a token trigger.
fn add_trigger(ty: i64, value: const u8[:]) {
    var tok: i32 = -1;
    var kind = ty;
    if ty == chat_render::TRIGGER_WORD {
        var ids: i32[>..] = [];
        vocab::tokenize(value, false, true, ids);
        if ids.len == 1 {
            if !is_preserved(ids[0]) { fail(E_INVALID_REQUEST, str("Grammar trigger word should be marked as preserved token: ", value)); }
            kind = chat_render::TRIGGER_TOKEN;
            tok = ids[0];
        }
    } else if ty == chat_render::TRIGGER_TOKEN {
        var ids: i32[>..] = [];
        vocab::tokenize(value, false, true, ids);
        if ids.len == 1 { tok = ids[0]; }
    }
    t_trig_types.push(kind);
    t_trig_values.append(value);
    t_trig_values.push(0);
    t_trig_tokens.push(tok);
}

// task_params::to_json: the generation settings; only_metrics for /props.
fn params_json(o: u8[>..]&, only_metrics: bool) {
    o.push('{');
    ki(o, "seed", t_sp.seed);
    kf(o, "temperature", t_sp.temp);
    kf(o, "dynatemp_range", t_sp.dynatemp_range);
    kf(o, "dynatemp_exponent", t_sp.dynatemp_exponent);
    ki(o, "top_k", t_sp.top_k);
    kf(o, "top_p", t_sp.top_p);
    kf(o, "min_p", t_sp.min_p);
    kf(o, "top_n_sigma", t_sp.top_n_sigma);
    kf(o, "xtc_probability", t_sp.xtc_probability);
    kf(o, "xtc_threshold", t_sp.xtc_threshold);
    kf(o, "typical_p", t_sp.typ_p);
    ki(o, "repeat_last_n", t_sp.penalty_last_n);
    kf(o, "repeat_penalty", t_sp.penalty_repeat);
    kf(o, "presence_penalty", t_sp.penalty_present);
    kf(o, "frequency_penalty", t_sp.penalty_freq);
    kf(o, "dry_multiplier", t_sp.dry_multiplier);
    kf(o, "dry_base", t_sp.dry_base);
    ki(o, "dry_allowed_length", t_sp.dry_allowed_length);
    ki(o, "dry_penalty_last_n", t_sp.dry_penalty_last_n);
    if !only_metrics { k(o, "dry_sequence_breakers"); str_list(o, t_sp.dry_breakers); }
    ki(o, "mirostat", t_sp.mirostat);
    kf(o, "mirostat_tau", t_sp.mirostat_tau);
    kf(o, "mirostat_eta", t_sp.mirostat_eta);
    kf(o, "adaptive_target", -1.0);
    kf(o, "adaptive_decay", 0.9);
    if !only_metrics { k(o, "stop"); str_list(o, t_stops); }
    ki(o, "max_tokens", t_n_predict);
    ki(o, "n_predict", t_n_predict);
    ki(o, "n_keep", t_n_keep);
    ki(o, "n_discard", 0);
    kb(o, "ignore_eos", t_ignore_eos);
    kb(o, "stream", t_stream);
    if !only_metrics {
        k(o, "logit_bias");
        o.push('[');
        for lb in t_logit_bias {
            sep(o);
            o.push('{');
            kf(o, "bias", lb.bias);
            ki(o, "token", lb.token);
            o.push('}');
        }
        o.push(']');
    }
    ki(o, "n_probs", t_sp.n_probs);
    ki(o, "min_keep", t_sp.min_keep);
    if !only_metrics {
        ks(o, "grammar", t_grammar);
        kb(o, "grammar_lazy", t_grammar_lazy);
        k(o, "grammar_triggers");
        o.push('[');
        var start = 0;
        var n = 0;
        for b, i in t_trig_values {
            guard b == 0 else { continue; }
            sep(o);
            o.push('{');
            ki(o, "type", t_trig_types[n]);
            ks(o, "value", t_trig_values[start..i]);
            if t_trig_types[n] == chat_render::TRIGGER_TOKEN { ki(o, "token", t_trig_tokens[n]); }
            o.push('}');
            start = i + 1;
            n++;
        }
        o.push(']');
        k(o, "preserved_tokens");
        o.push('[');
        for t in t_preserved { ei(o, t); }
        o.push(']');
    }
    ks(o, "chat_format", chat_apply::format_name(t_chat_format));
    ks(o, "reasoning_format", reasoning_format_name(t_reasoning_format));
    kb(o, "reasoning_in_content", t_stream && t_reasoning_format == chat_render::REASONING_DEEPSEEK_LEGACY);
    ks(o, "generation_prompt", t_gen_prompt);
    k(o, "samplers");
    o.push('[');
    for kind in t_sp.order { es(o, kind_name_of(kind)); }
    o.push(']');
    ks(o, "speculative.types", "none");
    kb(o, "timings_per_token", t_timings_per_token);
    kb(o, "post_sampling_probs", false);
    kb(o, "backend_sampling", false);
    kraw(o, "lora", "[]");
    o.push('}');
}

//
// chat requests (oaicompat_chat_params_parse): the messages through the
// template, and what the template's parser needs
//

// The text of the request's messages and tools, which the messages are views of.
var chat_text: u8[>..] = [];

struct Span { at: i64, len: i64 }
struct MsgRaw {
    role: Span, content: Span, reasoning: Span, tool_name: Span, tool_call_id: Span,
    call0: i64, ncalls: i64, part0: i64, nparts: i64,
}
struct CallRaw { name: Span, arguments: Span, id: Span }
struct PartRaw { kind: Span, text: Span, name: Span }
struct ToolRaw { name: Span, description: Span, parameters: Span }

fn keep(s: const u8[:]) -> Span {
    let at = chat_text.len;
    chat_text.append(s);
    return Span { at: at, len: s.len };
}

fn keep_dump(v: i64) -> Span {
    var d: u8[>..] = [];
    jdoc::dump(d, v);
    return keep(d);
}

fn txt(s: Span) -> const u8[:] { chat_text[s.at..s.at + s.len] }

fn need_str(v: i64, what: const u8[:]) -> const u8[:] {
    if !jdoc::is_str(v) {
        var d: u8[>..] = [];
        jdoc::dump(d, v);
        fail(E_INVALID_REQUEST, str(what, d));
    }
    return jdoc::str_of(v);
}

// The prompt of a chat request (into prompt), and the task's chat parameters.
fn chat_prompt(body: i64, prompt: u8[>..]&) {
    let tools_v = jdoc::get(body, "tools");
    let has_tools = jdoc::is_arr(tools_v) && jdoc::len(tools_v) > 0;
    let stream = jdoc::get_bool(body, "stream", false);
    let tool_choice_name = jdoc::get_str(body, "tool_choice", "auto");
    if !use_jinja {
        if has_tools { fail(E_INVALID_REQUEST, "tools param requires --jinja flag"); }
        if tool_choice_name != "auto" { fail(E_INVALID_REQUEST, "tool_choice param requires --jinja flag"); }
    }
    var json_schema_text: u8[>..] = [];
    var has_schema = false;
    {
        let s = jdoc::get(body, "json_schema");
        if !jdoc::is_null(s) { jdoc::dump(json_schema_text, s); has_schema = true; }
    }
    let grammar = jdoc::get_str(body, "grammar", "");
    if has_schema && grammar.len > 0 { fail(E_INVALID_REQUEST, "Cannot use both json_schema and grammar"); }
    if jdoc::contains(body, "response_format") {
        let rf = jdoc::get(body, "response_format");
        let ty = jdoc::get_str(rf, "type", "");
        if ty == "json_object" {
            if jdoc::contains(rf, "schema") || !has_schema {
                json_schema_text.clear();
                let s = jdoc::get(rf, "schema");
                if s >= 0 { jdoc::dump(json_schema_text, s); } else { json_schema_text.append("{}"); }
                has_schema = true;
            }
        } else if ty == "json_schema" {
            let wrapper = jdoc::get(rf, "json_schema");
            let s = jdoc::get(wrapper, "schema");
            json_schema_text.clear();
            if s >= 0 { jdoc::dump(json_schema_text, s); } else { json_schema_text.append("{}"); }
            has_schema = true;
        } else if ty.len > 0 && ty != "text" {
            fail(E_INVALID_REQUEST, str("response_format type must be one of \"text\" or \"json_object\", but got: ", ty));
        }
    }
    if json_schema_text == "{}" { json_schema_text.clear(); json_schema_text.append("{\"type\":\"object\"}"); }

    guard jdoc::contains(body, "messages") else { fail(E_INVALID_REQUEST, "'messages' is required"); }
    let msgs_v = jdoc::get(body, "messages");
    guard jdoc::is_arr(msgs_v) else { fail(E_INVALID_REQUEST, "Expected 'messages' to be an array"); }

    // the messages and tools as spans of chat_text
    chat_text.clear();
    var raws: MsgRaw[>..] = [];
    var calls: CallRaw[>..] = [];
    var parts: PartRaw[>..] = [];
    var tools: ToolRaw[>..] = [];
    for i in jdoc::len(msgs_v) {
        let msg = jdoc::at(msgs_v, i);
        let role = jdoc::get_str(msg, "role", "");
        let has_content = jdoc::contains(msg, "content");
        let has_calls = jdoc::contains(msg, "tool_calls");
        if role != "assistant" && !has_content { fail(E_INVALID_REQUEST, "All non-assistant messages must contain 'content'"); }
        if role == "assistant" && !has_content && !has_calls { fail(E_INVALID_REQUEST, "Assistant message must contain either 'content' or 'tool_calls'!"); }
        guard jdoc::is_obj(msg) else { fail(E_INVALID_REQUEST, "Expected 'message' to be an object"); }
        guard jdoc::contains(msg, "role") else { fail(E_INVALID_REQUEST, "Missing 'role' in message"); }
        var r = MsgRaw {
            role: keep(need_str(jdoc::get(msg, "role"), "Invalid role: ")),
            content: Span { at: 0, len: 0 }, reasoning: Span { at: 0, len: 0 },
            tool_name: Span { at: 0, len: 0 }, tool_call_id: Span { at: 0, len: 0 },
            call0: calls.len, ncalls: 0, part0: parts.len, nparts: 0,
        };
        if has_content {
            let content = jdoc::get(msg, "content");
            if jdoc::is_str(content) {
                r.content = keep(jdoc::str_of(content));
            } else if jdoc::is_arr(content) {
                for p in jdoc::len(content) {
                    let part = jdoc::at(content, p);
                    let ty = jdoc::get_str(part, "type", "");
                    guard jdoc::contains(part, "type") else { fail(E_INVALID_REQUEST, "Missing content part type"); }
                    if ty == "text" || ty == "media_marker" {
                        parts.push(PartRaw { kind: keep(ty), text: keep(jdoc::get_str(part, "text", "")), name: Span { at: 0, len: 0 } });
                    } else if ty == "tool_reference" {
                        parts.push(PartRaw { kind: keep(ty), text: Span { at: 0, len: 0 }, name: keep(jdoc::get_str(part, "name", "")) });
                    } else if ty == "image_url" || ty == "input_audio" || ty == "input_video" || ty == "video_url" {
                        fail(E_INVALID_REQUEST, "image input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
                    } else {
                        fail(E_INVALID_REQUEST, "unsupported content[].type");
                    }
                }
                r.nparts = parts.len - r.part0;
            } else if !jdoc::is_null(content) {
                fail(E_INVALID_REQUEST, "Expected 'content' to be a string or an array");
            }
        }
        if has_calls {
            let tcs = jdoc::get(msg, "tool_calls");
            for t in jdoc::len(tcs) {
                let tc = jdoc::at(tcs, t);
                guard jdoc::get_str(tc, "type", "") == "function" else { fail(E_INVALID_REQUEST, "Unsupported tool call type"); }
                let fc = jdoc::get(tc, "function");
                guard jdoc::contains(fc, "name") else { fail(E_INVALID_REQUEST, "Missing tool call name"); }
                let args = jdoc::get(fc, "arguments");
                let name = keep(jdoc::get_str(fc, "name", ""));
                let arguments = if jdoc::is_str(args) { keep(jdoc::str_of(args)) } else { keep_dump(args) };
                calls.push(CallRaw { name: name, arguments: arguments, id: keep(jdoc::get_str(tc, "id", "")) });
            }
            r.ncalls = calls.len - r.call0;
        }
        r.reasoning = keep(jdoc::get_str(msg, "reasoning_content", ""));
        r.tool_name = keep(jdoc::get_str(msg, "name", ""));
        r.tool_call_id = keep(jdoc::get_str(msg, "tool_call_id", ""));
        raws.push(r);
    }
    if has_tools {
        for i in jdoc::len(tools_v) {
            let tool = jdoc::at(tools_v, i);
            guard jdoc::get_str(tool, "type", "") == "function" else { fail(E_INVALID_REQUEST, "Unsupported tool type"); }
            let f = jdoc::get(tool, "function");
            let params = jdoc::get(f, "parameters");
            tools.push(ToolRaw {
                name: keep(jdoc::get_str(f, "name", "")),
                description: keep(jdoc::get_str(f, "description", "")),
                parameters: if params >= 0 { keep_dump(params) } else { keep("{}") },
            });
        }
    }

    // the views, now that chat_text is complete
    var call_views: chat_apply::ToolCall[>..] = [];
    for cr in calls { call_views.push(chat_apply::ToolCall { name: txt(cr.name), arguments: txt(cr.arguments), id: txt(cr.id) }); }
    var part_views: chat_apply::Part[>..] = [];
    for pr in parts { part_views.push(chat_apply::Part { kind: txt(pr.kind), text: txt(pr.text), name: txt(pr.name) }); }
    var msgs: chat_apply::Message[>..] = [];
    for r in raws {
        msgs.push(chat_apply::Message {
            role: txt(r.role), content: txt(r.content), reasoning: txt(r.reasoning),
            tool_calls: call_views[r.call0..r.call0 + r.ncalls], parts: part_views[r.part0..r.part0 + r.nparts],
            tool_name: txt(r.tool_name), tool_call_id: txt(r.tool_call_id),
        });
    }
    var tool_views: chat_apply::Tool[>..] = [];
    for tr in tools { tool_views.push(chat_apply::Tool { name: txt(tr.name), description: txt(tr.description), parameters: txt(tr.parameters) }); }

    let tool_choice = if tool_choice_name == "auto" { chat_render::TOOL_CHOICE_AUTO }
        else if tool_choice_name == "none" { chat_render::TOOL_CHOICE_NONE }
        else if tool_choice_name == "required" { chat_render::TOOL_CHOICE_REQUIRED }
        else { fail(E_INVALID_REQUEST, str("Invalid tool_choice: ", tool_choice_name)); chat_render::TOOL_CHOICE_AUTO };
    let caps = if ts.has_tool_use { ts.tool_use.caps } else { ts.main.caps };
    var add_gen = jdoc::get_bool(body, "add_generation_prompt", true);
    var cont = chat_render::CONTINUE_NONE;
    if jdoc::contains(body, "continue_final_message") {
        let cv = jdoc::get(body, "continue_final_message");
        if jdoc::is_bool(cv) { cont = if jdoc::bool_of(cv) { chat_render::CONTINUE_AUTO } else { chat_render::CONTINUE_NONE }; }
        else if jdoc::is_str(cv) {
            let s = jdoc::str_of(cv);
            cont = if s == "auto" { chat_render::CONTINUE_AUTO } else if s == "reasoning" { chat_render::CONTINUE_REASONING }
                else if s == "content" { chat_render::CONTINUE_CONTENT } else if s == "none" { chat_render::CONTINUE_NONE }
                else { fail(E_INVALID_REQUEST, str("Invalid continue_final_message: ", s)); 0 };
        }
    }
    let n = msgs.len;
    if cont == chat_render::CONTINUE_NONE && prefill_assistant && n > 0 && msgs[n - 1].role == "assistant" {
        if n >= 2 && msgs[n - 2].role == "assistant" { fail(E_INVALID_REQUEST, "Cannot have 2 or more assistant messages at the end of the list."); }
        cont = chat_render::CONTINUE_AUTO;
        add_gen = false;
    }
    if cont != chat_render::CONTINUE_NONE && add_gen { fail(E_INVALID_REQUEST, "Cannot set both add_generation_prompt and continue_final_message to true."); }
    if cont != chat_render::CONTINUE_NONE && n > 0 && msgs[n - 1].role == "assistant" && msgs[n - 1].tool_calls.len > 0 {
        fail(E_INVALID_REQUEST, "Cannot continue an assistant message that contains tool calls.");
    }
    var rfmt = reasoning_format;
    if jdoc::contains(body, "reasoning_format") { rfmt = reasoning_format_from_name(jdoc::get_str(body, "reasoning_format", "")); }
    var thinking = enable_thinking;
    if tool_views.len > 0 && tool_choice != chat_render::TOOL_CHOICE_NONE && jdoc::contains(body, "grammar") {
        fail(E_INVALID_REQUEST, "Cannot use custom grammar constraints with tools.");
    }
    // the template's kwargs: the request's, as JSON
    var kw_text: u8[>..] = [];
    var kw_ends: i64[>..] = [];
    {
        let kw = jdoc::get(body, "chat_template_kwargs");
        if jdoc::is_obj(kw) {
            for i in jdoc::len(kw) {
                kw_text.append(jdoc::key(kw, i));
                kw_ends.push(kw_text.len);
                jdoc::dump(kw_text, jdoc::at(kw, i));
                kw_ends.push(kw_text.len);
                let key = jdoc::key(kw, i);
                if key == "enable_thinking" {
                    let v = jdoc::at(kw, i);
                    if jdoc::is_bool(v) { thinking = jdoc::bool_of(v); }
                    else if jdoc::is_str(v) { fail(E_INVALID_REQUEST, "invalid type for \"enable_thinking\" (expected boolean, got string)"); }
                }
            }
        }
        if jdoc::contains(body, "reasoning_effort") {
            let effort = jdoc::get_str(body, "reasoning_effort", "");
            if effort == "none" {
                thinking = false;
            } else if effort.len > 0 {
                kw_text.append("reasoning_effort");
                kw_ends.push(kw_text.len);
                jdoc::put_str(kw_text, effort);
                kw_ends.push(kw_text.len);
            }
        }
    }
    var kws: chat_apply::KV[>..] = [];
    {
        var start = 0;
        var i = 0;
        while i < kw_ends.len {
            let key = kw_text[start..kw_ends[i]];
            let value = kw_text[kw_ends[i]..kw_ends[i + 1]];
            // a key given twice: the last one
            var dup = -1;
            for kv, j in kws { if kv.key == key { dup = j; } }
            if dup >= 0 { kws[dup] = chat_apply::KV { key: key, value: value }; }
            else { kws.push(chat_apply::KV { key: key, value: value }); }
            start = kw_ends[i + 1];
            i += 2;
        }
    }
    // reasoning_effort "none" also drops the kwarg
    if jdoc::get_str(body, "reasoning_effort", "") == "none" {
        var kept: chat_apply::KV[>..] = [];
        for kv in kws { if kv.key != "reasoning_effort" { kept.push(kv); } }
        kws.clear();
        kws.append(kept);
    }

    let inp = chat_apply::Inputs {
        messages: msgs,
        grammar: grammar,
        json_schema: json_schema_text,
        add_generation_prompt: add_gen,
        continue_final_message: cont,
        tools: tool_views,
        tool_choice: tool_choice,
        parallel_tool_calls: jdoc::get_bool(body, "parallel_tool_calls", caps.supports_parallel_tool_calls),
        reasoning_format: rfmt,
        enable_thinking: thinking,
        kwargs: kws,
    };
    if !chat_apply::apply(ts, inp) { fail(E_INVALID_REQUEST, chat_apply::err); }
    prompt.append(chat_render::prompt);
    if stream && jdoc::get_bool(body, "logprobs", false) && has_tools { fail(E_INVALID_REQUEST, "logprobs is not supported with tools + stream"); }
}

// apply_chat_session: the template's grammar, triggers, stops and parser.
fn apply_chat_session() {
    t_chat = true;
    if chat_render::grammar.len > 0 {
        t_grammar.clear();
        t_grammar.append(chat_render::grammar);
        t_grammar_user = false;
    }
    t_grammar_lazy = chat_render::grammar_lazy;
    t_gen_prompt.clear();
    t_gen_prompt.append(chat_render::gen_prompt);
    {
        var start = 0;
        for b, i in chat_render::preserved_tokens {
            if b == 0 { add_preserved(chat_render::preserved_tokens[start..i]); start = i + 1; }
        }
    }
    {
        var start = 0;
        var n = 0;
        for b, i in chat_render::trigger_values {
            if b == 0 {
                add_trigger(chat_render::trigger_types[n], chat_render::trigger_values[start..i]);
                start = i + 1;
                n++;
            }
        }
    }
    if t_grammar_lazy && t_trig_types.len == 0 { fail(E_INVALID_REQUEST, "Error: no triggers set for lazy grammar!"); }
    t_chat_format = chat_render::format;
    t_stops.append(chat_render::additional_stops);
}

//
// the generation (the slot)
//

struct Stats { cache_n: i64, prompt_n: i64, prompt_ms: f64, predicted_n: i64, predicted_ms: f64 }

var generated: u8[>..] = [];        // the text so far
var gen_tokens: i32[>..] = [];
var fed: u8[>..] = [];              // the text sent so far (the chat parser's input)
var n_sent = 0;
var stop_type = STOP_NONE;
var stopping_word: u8[>..] = [];
var has_new_line = false;
var truncated = false;
var n_gen = 0;
var n_prompt = 0;
var stats = Stats { cache_n: 0, prompt_n: 0, prompt_ms: 0.0, predicted_n: 0, predicted_ms: 0.0 };
var prompt_text: u8[>..] = [];      // the prompt's tokens as text, for the response
var streaming = false;              // the events' head is sent

fn now_ms() -> f64 { clock_ns() / 1000000.0 }

// find_stopping_strings: the earliest stop string in text, at its end only
// (with last_len, the last token's piece) when full, or a stop string's
// start at the end when partial; -1 if none. A full stop sets the stop.
fn find_stop(text: const u8[:], last_len: i64, full: bool) -> i64 {
    var stop_pos = -1;
    var start = 0;
    for b, i in t_stops {
        guard b == 0 else { continue; }
        let word = t_stops[start..i];
        start = i + 1;
        var pos = -1;
        if full {
            let tmp = word.len + last_len;
            let from = if text.len > tmp { text.len - tmp } else { 0 };
            let at = find(text[from..], word);
            pos = if at >= 0 { from + at } else { -1 };
        } else {
            pos = partial_stop(text, word);
        }
        if pos >= 0 && (stop_pos < 0 || pos < stop_pos) {
            if full {
                stop_type = STOP_WORD;
                stopping_word.clear();
                stopping_word.append(word);
            }
            stop_pos = pos;
        }
    }
    return stop_pos;
}

// string_find_partial_stop: where text ends with a start of stop, or -1.
fn partial_stop(text: const u8[:], stop: const u8[:]) -> i64 {
    guard text.len > 0 && stop.len > 0 else { return -1; }
    let last = text[text.len - 1];
    var len = min(text.len, stop.len);
    while len > 0 {
        if stop[len - 1] == last && ends_with(text, stop[..len]) { return text.len - len; }
        len--;
    }
    return -1;
}

// The previous chat message, for the stream's diffs.
var prev_content: u8[>..] = [];
var prev_reasoning: u8[>..] = [];
var prev_calls: u8[>..] = [];       // name, arguments, id: each ended by a NUL
var call_ids: u8[>..] = [];         // the ids given to the calls so far, each ended by a NUL
var n_prev_calls = 0;

fn reset_chat_state() {
    prev_content.clear();
    prev_reasoning.clear();
    prev_calls.clear();
    call_ids.clear();
    n_prev_calls = 0;
}

fn nth(list: const u8[:], n: i64) -> const u8[:] {
    var start = 0;
    var k0 = 0;
    for b, i in list {
        guard b == 0 else { continue; }
        if k0 == n { return list[start..i]; }
        k0++;
        start = i + 1;
    }
    return "";
}

fn random_string() -> u8[>..] {
    let chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    var out: u8[>..] = [];
    for i in 32 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        out.push(chars[(rng_state % 62) as! i64]);
    }
    return out;
}
var rng_state: u64 = 0x9E3779B97F4A7C15;

// The id of call i: the parser's, or a new random one (set_tool_call_ids).
fn call_id(i: i64, parsed: const u8[:]) -> u8[>..] {
    while list_count(call_ids) <= i {
        if parsed.len > 0 && list_count(call_ids) == i { call_ids.append(parsed); } else { call_ids.append(random_string()); }
        call_ids.push(0);
    }
    return str(nth(call_ids, i));
}

// The chat message of the output: parsed into chat_peg's content_text,
// reasoning_text and calls (an empty parse keeps the previous message).
fn parse_chat(text: const u8[:], partial: bool) -> bool {
    if !chat_apply::parse(text, partial) {
        if partial { return false; }
        log(str("parse_chat: unparsed ", chat_apply::format_name(chat_render::format), " output: ", chat_apply::input_text));
        fail(E_SERVER, chat_apply::err);
    }
    let empty = chat_peg::content_text.len == 0 && chat_peg::reasoning_text.len == 0 && chat_peg::calls.len == 0;
    return !empty;
}

// The message's diffs from the previous one, as chunks' deltas, into
// deltas (each ended by a NUL); then the message becomes the previous one.
fn chat_diffs(deltas: u8[>..]&) {
    fn string_diff(last: const u8[:], cur: const u8[:]) -> u8[>..] {
        if last.len == 0 { return str(cur); }
        if !starts_with(cur, last) {
            if starts_with(last, cur) { return str(""); }
            fail(E_SERVER, str("Invalid diff: '", last, "' not found at start of '", cur, "'"));
        }
        return str(cur[last.len..]);
    }
    fn delta_call(idx: i64, id: const u8[:], name: const u8[:], args: const u8[:]) {
        var o: u8[>..] = [];
        o.push('{');
        k(o, "tool_calls");
        o.append("[{");
        ki(o, "index", idx);
        if id.len > 0 { ks(o, "id", id); ks(o, "type", "function"); }
        if name.len > 0 || args.len > 0 {
            k(o, "function");
            o.push('{');
            if name.len > 0 { ks(o, "name", name); }
            if args.len > 0 { ks(o, "arguments", args); }
            o.push('}');
        }
        o.append("}]}");
        deltas.append(o);
        deltas.push(0);
    }
    var content: u8[>..] = [];
    content.append(chat_peg::content_text);
    var reasoning: u8[>..] = [];
    reasoning.append(chat_peg::reasoning_text);
    var calls: u8[>..] = [];
    let ncalls = chat_peg::calls.len;
    for cl, i in chat_peg::calls {
        calls.append(chat_peg::t(cl.name));
        calls.push(0);
        calls.append(chat_peg::t(cl.arguments));
        calls.push(0);
        calls.append(call_id(i, chat_peg::t(cl.id)));
        calls.push(0);
    }
    if prev_reasoning != reasoning {
        var o: u8[>..] = [];
        o.push('{');
        ks(o, "reasoning_content", string_diff(prev_reasoning, reasoning));
        o.push('}');
        deltas.append(o);
        deltas.push(0);
    }
    if prev_content != content {
        var o: u8[>..] = [];
        o.push('{');
        ks(o, "content", string_diff(prev_content, content));
        o.push('}');
        deltas.append(o);
        deltas.push(0);
    }
    if ncalls < n_prev_calls { fail(E_SERVER, "Invalid diff: now finding less tool calls!"); }
    if n_prev_calls > 0 {
        let idx = n_prev_calls - 1;
        let pname = nth(prev_calls, 3 * idx);
        let pargs = nth(prev_calls, 3 * idx + 1);
        let pid = nth(prev_calls, 3 * idx + 2);
        let nname = nth(calls, 3 * idx);
        let nargs = nth(calls, 3 * idx + 1);
        let nid = nth(calls, 3 * idx + 2);
        if pname != nname && pname.len > 0 && nname.len > 0 && !starts_with(nname, pname) { fail(E_SERVER, "Invalid diff: tool call mismatch!"); }
        let args_diff = string_diff(pargs, nargs);
        if args_diff.len > 0 || pid != nid || pname != nname {
            let changed = pid != nid || pname != nname;
            delta_call(idx, if changed { nid } else { "" }, if changed { nname } else { "" }, args_diff);
        }
    }
    for idx in n_prev_calls..ncalls {
        delta_call(idx, nth(calls, 3 * idx + 2), nth(calls, 3 * idx), nth(calls, 3 * idx + 1));
    }
    prev_content.clear();
    prev_content.append(content);
    prev_reasoning.clear();
    prev_reasoning.append(reasoning);
    prev_calls.clear();
    prev_calls.append(calls);
    n_prev_calls = ncalls;
}

// common_chat_msg::to_json_oaicompat of the previous message.
fn chat_msg_json(o: u8[>..]&) {
    o.push('{');
    ks(o, "role", "assistant");
    ks(o, "content", prev_content);
    if prev_reasoning.len > 0 { ks(o, "reasoning_content", prev_reasoning); }
    if n_prev_calls > 0 {
        k(o, "tool_calls");
        o.push('[');
        for i in n_prev_calls {
            sep(o);
            o.push('{');
            ks(o, "type", "function");
            k(o, "function");
            o.push('{');
            ks(o, "name", nth(prev_calls, 3 * i));
            ks(o, "arguments", nth(prev_calls, 3 * i + 1));
            o.push('}');
            let id = nth(prev_calls, 3 * i + 2);
            if id.len > 0 { ks(o, "id", id); }
            o.push('}');
        }
        o.push(']');
    }
    o.push('}');
}

fn stats_json(o: u8[>..]&) {
    o.push('{');
    ki(o, "cache_n", stats.cache_n);
    ki(o, "prompt_n", stats.prompt_n);
    kd(o, "prompt_ms", stats.prompt_ms);
    kd(o, "prompt_per_token_ms", stats.prompt_ms / stats.prompt_n);
    kd(o, "prompt_per_second", 1000.0 / stats.prompt_ms * stats.prompt_n);
    ki(o, "predicted_n", stats.predicted_n);
    kd(o, "predicted_ms", stats.predicted_ms);
    kd(o, "predicted_per_token_ms", stats.predicted_ms / stats.predicted_n);
    kd(o, "predicted_per_second", 1000.0 / stats.predicted_ms * stats.predicted_n);
    o.push('}');
}

fn usage_json(o: u8[>..]&) {
    o.push('{');
    ki(o, "completion_tokens", n_gen);
    ki(o, "prompt_tokens", n_prompt);
    ki(o, "total_tokens", n_gen + n_prompt);
    k(o, "prompt_tokens_details");
    o.push('{');
    ki(o, "cached_tokens", stats.cache_n);
    o.push('}');
    o.push('}');
}

fn unix_time() -> i64 { time_ns() / 1000000000 }

// The fields of an OpenAI chunk that follow its choices.
fn chunk_tail(o: u8[>..]&, object: const u8[:]) {
    ki(o, "created", unix_time());
    ks(o, "id", t_cmpl_id);
    ks(o, "model", model_name);
    ks(o, "system_fingerprint", BUILD_INFO);
    ks(o, "object", object);
}

// One event: "data: <text>\n\n".
fn send(data: const u8[:]) {
    var o: u8[>..] = [];
    format(o, "data: ", data, "\n\n");
    if !reply_part(o, false) { client_gone = true; }
}
var client_gone = false;

// A chat chunk with one delta (a JSON object), unfinished.
fn chat_chunk(o: u8[>..]&, delta: const u8[:], finish: const u8[:]) {
    o.push('{');
    k(o, "choices");
    o.append("[{");
    if finish.len > 0 { ks(o, "finish_reason", finish); } else { kraw(o, "finish_reason", "null"); }
    ki(o, "index", 0);
    kraw(o, "delta", delta);
    o.append("}]");
    ki(o, "created", unix_time());
    ks(o, "id", t_cmpl_id);
    ks(o, "model", model_name);
    ks(o, "system_fingerprint", BUILD_INFO);
    ks(o, "object", "chat.completion.chunk");
}

// send_partial_response with the token's text (text_to_send).
fn send_partial(tok: i32, text: const u8[:]) {
    let with_stats = stop_type != STOP_NONE || t_timings_per_token;
    if with_stats { finish_stats(); }
    if t_res_type == RES_NONE {
        var o: u8[>..] = [];
        o.push('{');
        ki(o, "index", 0);
        ks(o, "content", text);
        k(o, "tokens");
        o.push('[');
        ei(o, tok);
        o.push(']');
        kb(o, "stop", false);
        ki(o, "id_slot", -1);
        ki(o, "tokens_predicted", n_gen);
        ki(o, "tokens_evaluated", n_prompt);
        if with_stats { k(o, "timings"); stats_json(o); }
        o.push('}');
        send(o);
    } else if t_res_type == RES_OAI_CMPL {
        var o: u8[>..] = [];
        o.push('{');
        k(o, "choices");
        o.append("[{");
        ks(o, "text", text);
        ki(o, "index", 0);
        kraw(o, "logprobs", "null");
        kraw(o, "finish_reason", "null");
        o.append("}]");
        ki(o, "created", unix_time());
        ks(o, "model", model_name);
        ks(o, "system_fingerprint", BUILD_INFO);
        ks(o, "object", "text_completion");
        ks(o, "id", t_cmpl_id);
        if with_stats { k(o, "timings"); stats_json(o); }
        o.push('}');
        send(o);
    } else {
        var deltas: u8[>..] = [];
        if n_gen == 1 { deltas.append("{\"role\":\"assistant\",\"content\":null}"); deltas.push(0); }
        fed.append(text);
        if parse_chat(fed, true) { chat_diffs(deltas); }
        let n = list_count(deltas);
        for i in n {
            var o: u8[>..] = [];
            chat_chunk(o, nth(deltas, i), "");
            if i == n - 1 && with_stats { k(o, "timings"); stats_json(o); }
            o.push('}');
            send(o);
        }
    }
}

var t_prompt_start = 0.0;
var t_gen_start = 0.0;
var t_last = 0.0;

fn finish_stats() {
    stats.predicted_n = n_gen;
    stats.predicted_ms = t_last - t_gen_start;
}

// process_token: the token's text checked for stops; false when the generation ends.
fn process_token(tok: i32, piece0: const u8[:]) -> bool {
    generated.append(piece0);
    if t_return_tokens { gen_tokens.push(tok); }
    var has_next = true;
    let incomplete = utf8_complete(generated) < generated.len;
    if !incomplete {
        var pos = min(n_sent, generated.len);
        var test: u8[>..] = [];
        test.append(generated[pos..]);
        var send_text = true;
        var stop_pos = find_stop(test, piece0.len, true);
        if stop_pos >= 0 {
            has_next = false;
            generated.resize(pos + stop_pos);
            pos = min(n_sent, generated.len);
        } else if !vocab::is_eog(tok) {
            stop_pos = find_stop(test, piece0.len, false);
            send_text = stop_pos < 0;
        }
        var text_to_send: u8[>..] = [];
        if send_text {
            text_to_send.append(generated[pos..]);
            n_sent += text_to_send.len;
        }
        if t_stream { send_partial(tok, text_to_send); }
        if find(text_to_send, "\n") >= 0 { has_new_line_now = true; }
    }
    if cached.len + 1 >= c.n_ctx {
        truncated = true;
        stop_type = STOP_LIMIT;
        has_next = false;
    }
    if n_gen > 0 && has_next && t_n_predict != -1 && t_n_predict - n_gen <= 0 {
        stop_type = STOP_LIMIT;
        has_next = false;
    }
    if has_new_line && t_n_indent > 0 {
        if last_nl_pos > 0 {
            var pos = last_nl_pos;
            var n_indent = 0;
            while pos < generated.len && (generated[pos] == ' ' || generated[pos] == '\t') { n_indent++; pos++; }
            if pos < generated.len && n_indent < t_n_indent {
                stop_type = STOP_LIMIT;
                has_next = false;
                generated.resize(pos);
            }
        }
        let rest = find(generated[last_nl_pos..], "\n");
        if rest >= 0 { last_nl_pos = last_nl_pos + rest + 1; }
    }
    if has_new_line_now {
        has_new_line = true;
        has_new_line_now = false;
        if t_max_predict_ms > 0 && t_last - t_gen_start > t_max_predict_ms {
            stop_type = STOP_LIMIT;
            has_next = false;
        }
    }
    if vocab::is_eog(tok) {
        stop_type = STOP_EOS;
        has_next = false;
    }
    return has_next;
}
var has_new_line_now = false;
var last_nl_pos = 0;

// The tokens of the prompt run (with the cache's common prefix reused),
// then tokens generated until a stop; the results sent as they come.
fn generate(prompt: const i32[:]) {
    generated.clear();
    gen_tokens.clear();
    fed.clear();
    n_sent = 0;
    stop_type = STOP_NONE;
    stopping_word.clear();
    has_new_line = false;
    has_new_line_now = false;
    last_nl_pos = 0;
    truncated = false;
    n_gen = 0;
    n_prompt = prompt.len;
    reset_chat_state();
    if t_cmpl_id.len == 0 { t_cmpl_id.append("chatcmpl-"); t_cmpl_id.append(random_string()); }

    if prompt.len == 0 { fail(E_INVALID_REQUEST, "\"prompt\" must not be empty"); }
    if prompt.len >= c.n_ctx {
        fail(E_EXCEED_CONTEXT, str("request (", prompt.len, " tokens) exceeds the available context size (", c.n_ctx, " tokens), try increasing it"));
    }
    // the sampler (common_sampler_init) and the grammar
    sampler::init(ch, t_sp);
    ch.has_grammar = false;
    if t_grammar.len > 0 {
        if !sampler::set_grammar(ch, t_grammar, "root") {
            fail(E_INVALID_REQUEST, str("Failed to initialize samplers: failed to parse grammar: ", grammar::err));
        }
        if t_grammar_lazy {
            grammar::make_lazy();
            for ty, i in t_trig_types {
                if ty == chat_render::TRIGGER_TOKEN && t_trig_tokens[i] >= 0 { grammar::add_trigger_token(t_trig_tokens[i]); }
                else if ty == chat_render::TRIGGER_WORD { grammar::add_trigger_word(nth(t_trig_values, i)); }
                else { fail(E_NOT_SUPPORTED, "regex grammar triggers are not supported yet"); }
            }
        } else if !t_grammar_user && t_gen_prompt.len > 0 {
            // the generation prompt is in the prompt already: the grammar goes past it
            var ids: i32[>..] = [];
            vocab::tokenize(t_gen_prompt, false, true, ids);
            for id, j in ids {
                let p = piece(id, true);
                if j == 0 && p.len > 0 && is_space(p[0]) && !is_space(t_gen_prompt[0]) { continue; }
                grammar::accept_impl(id);
            }
        }
    }

    // the cache: the common prefix kept, at least one token to evaluate
    var n_past = 0;
    if t_cache_prompt {
        while n_past < cached.len && n_past < prompt.len && cached[n_past] == prompt[n_past] { n_past++; }
    }
    if n_past == prompt.len && n_past > 0 { n_past--; }
    n_past = context::keep_first(c, m, n_past);
    if c.n_past != n_past { context::clear(c); n_past = 0; }
    cached.resize(n_past);
    stats = Stats { cache_n: n_past, prompt_n: prompt.len - n_past, prompt_ms: 0.0, predicted_n: 0, predicted_ms: 0.0 };
    prompt_text.clear();
    prompt_text.append(detokenize(prompt, true));

    if t_stream {
        reply_head(200, "text/event-stream", -1);
        guard reply_part("", false) else { client_gone = true; return; }
        streaming = true;
    }

    t_prompt_start = now_ms();
    var logits: f32[>..] = [];
    var start = n_past;
    while start < prompt.len {
        if !http::mail_live(reply_to) { client_gone = true; return; }
        let end = min(start + n_batch, prompt.len);
        context::decode(c, m, prompt[start..end], logits);
        cached.append(prompt[start..end]);
        start = end;
    }
    // init_sampler: the prompt's tokens into the penalties' windows
    sampler::reset(ch);
    for t in cached { sampler::accept(ch, t); }

    var cands: sampler::Cand[>..] = [];
    loop {
        for lb in t_logit_bias { logits[lb.token] += lb.bias; }
        if t_ignore_eos {
            for id in vocab::n_tokens() { if vocab::is_eog(id as! i32) { logits[id] = -F_INF; } }
        }
        let tok = sampler::sample(ch, cands, logits);
        n_gen++;
        t_last = now_ms();
        if n_gen == 1 {
            t_gen_start = t_last;
            stats.prompt_ms = t_last - t_prompt_start;
        }
        let text = piece(tok, is_preserved(tok));
        if !process_token(tok, text) { break; }
        if !http::mail_live(reply_to) { client_gone = true; }
        if client_gone { return; }
        context::decode(c, m, [tok], logits);
        cached.push(tok);
    }
    finish_stats();
    send_final();
}

fn stop_type_name(s: i64) -> const u8[:] {
    return match s { STOP_EOS => "eos", STOP_WORD => "word", STOP_LIMIT => "limit", _ => "none" };
}

// to_json_non_oaicompat of the final result.
fn final_json(o: u8[>..]&) {
    o.push('{');
    ki(o, "index", 0);
    ks(o, "content", if t_stream { "" } else { generated[..] });
    k(o, "tokens");
    o.push('[');
    if !t_stream { for t in gen_tokens { ei(o, t); } }
    o.push(']');
    ki(o, "id_slot", 0);
    kb(o, "stop", true);
    ks(o, "model", model_name);
    ki(o, "tokens_predicted", n_gen);
    ki(o, "tokens_evaluated", n_prompt);
    k(o, "generation_settings");
    params_json(o, false);
    ks(o, "prompt", prompt_text);
    kb(o, "has_new_line", has_new_line);
    kb(o, "truncated", truncated);
    ks(o, "stop_type", stop_type_name(stop_type));
    ks(o, "stopping_word", stopping_word);
    ki(o, "tokens_cached", cached.len);
    k(o, "timings");
    stats_json(o);
    o.push('}');
}

// send_final_response, in the response's format.
fn send_final() {
    let finish = if stop_type == STOP_WORD || stop_type == STOP_EOS { "stop" } else { "length" };
    var o: u8[>..] = [];
    if t_res_type == RES_NONE {
        final_json(o);
        if t_stream { send(o); } else { respond_with(200, "application/json; charset=utf-8", o); }
        return;
    }
    if t_res_type == RES_OAI_CMPL {
        if t_stream {
            // the final chunk: its text is empty, it was streamed
            o.push('{');
            k(o, "choices");
            o.append("[{");
            ks(o, "text", "");
            ki(o, "index", 0);
            kraw(o, "logprobs", "null");
            ks(o, "finish_reason", finish);
            o.append("}]");
        } else {
            o.push('{');
            k(o, "choices");
            o.append("[{");
            ks(o, "text", generated);
            ki(o, "index", 0);
            kraw(o, "logprobs", "null");
            ks(o, "finish_reason", finish);
            o.append("}]");
        }
        ki(o, "created", unix_time());
        ks(o, "model", model_name);
        ks(o, "system_fingerprint", BUILD_INFO);
        ks(o, "object", "text_completion");
        k(o, "usage");
        usage_json(o);
        ks(o, "id", t_cmpl_id);
        k(o, "timings");
        stats_json(o);
        o.push('}');
        if t_stream { send(o); send("[DONE]"); } else { respond_with(200, "application/json; charset=utf-8", o); }
        return;
    }
    // chat: the whole output parsed, not partial
    var deltas: u8[>..] = [];
    if parse_chat(if t_stream { fed[..] } else { generated[..] }, false) { chat_diffs(deltas); }
    else if !t_stream && prev_content.len == 0 { prev_content.append(generated); }
    let chat_finish = if stop_type == STOP_WORD || stop_type == STOP_EOS { if n_prev_calls > 0 { "tool_calls" } else { "stop" } } else { "length" };
    if !t_stream {
        o.push('{');
        k(o, "choices");
        o.append("[{");
        ks(o, "finish_reason", chat_finish);
        ki(o, "index", 0);
        k(o, "message");
        chat_msg_json(o);
        o.append("}]");
        ki(o, "created", unix_time());
        ks(o, "model", model_name);
        ks(o, "system_fingerprint", BUILD_INFO);
        ks(o, "object", "chat.completion");
        k(o, "usage");
        usage_json(o);
        ks(o, "id", t_cmpl_id);
        k(o, "timings");
        stats_json(o);
        o.push('}');
        respond_with(200, "application/json; charset=utf-8", o);
        return;
    }
    for i in list_count(deltas) {
        var d: u8[>..] = [];
        chat_chunk(d, nth(deltas, i), "");
        d.push('}');
        send(d);
    }
    var last: u8[>..] = [];
    chat_chunk(last, "{}", chat_finish);
    if t_include_usage {
        last.push('}');
        send(last);
        last.clear();
        last.push('{');
        kraw(last, "choices", "[]");
        chunk_tail(last, "chat.completion.chunk");
        k(last, "usage");
        usage_json(last);
    }
    k(last, "timings");
    stats_json(last);
    last.push('}');
    send(last);
    send("[DONE]");
}

//
// routes
//

fn respond_json(data: const u8[:]) { respond_with(200, "application/json; charset=utf-8", data); }

fn respond_error(kind: i64, msg: const u8[:]) {
    var o: u8[>..] = [];
    o.append("{\"error\":");
    error_json(o, kind, msg);
    o.push('}');
    respond_with(error_code(kind), "application/json; charset=utf-8", o);
}

// The request's body as JSON, or a 400.
fn body_json() -> i64 {
    jdoc::reset();
    let v = jdoc::parse(net::body);
    if v < 0 { fail(E_INVALID_REQUEST, str("[json.exception.parse_error.101] parse error: ", jdoc::err)); }
    return v;
}

fn route_props() {
    reset_task(RES_NONE);
    t_reasoning_format = chat_render::REASONING_NONE;
    var o: u8[>..] = [];
    o.push('{');
    k(o, "default_generation_settings");
    o.push('{');
    k(o, "params");
    params_json(o, true);
    ki(o, "n_ctx", c.n_ctx);
    o.push('}');
    ki(o, "total_slots", 1);
    ks(o, "model_alias", model_name);
    ks(o, "model_ftype", ftype_name);
    ks(o, "model_path", model_path);
    kraw(o, "modalities", "{\"vision\":false,\"video\":false,\"audio\":false}");
    kb(o, "endpoint_slots", false);
    kb(o, "endpoint_props", false);
    kb(o, "endpoint_metrics", false);
    kb(o, "ui", static_dir.len > 0);
    kraw(o, "ui_settings", "{}");
    ks(o, "chat_template", tmpl_src);
    k(o, "chat_template_caps");
    caps_json(o);
    ks(o, "bos_token", bos_text);
    ks(o, "eos_token", eos_text);
    ks(o, "build_info", BUILD_INFO);
    kb(o, "is_sleeping", false);
    kb(o, "cors_proxy_enabled", false);
    if use_jinja && tmpl_tool_src.len > 0 { ks(o, "chat_template_tool_use", tmpl_tool_src); }
    o.push('}');
    respond_json(o);
}

// common_chat_templates_get_caps: the caps' map (sorted by name) and the effort levels.
fn caps_json(o: u8[>..]&) {
    let caps = if ts.has_tool_use { ts.tool_use.caps } else { ts.main.caps };
    o.push('{');
    kb(o, "supports_object_arguments", caps.supports_object_arguments);
    kb(o, "supports_parallel_tool_calls", caps.supports_parallel_tool_calls);
    kb(o, "supports_preserve_reasoning", caps.supports_preserve_reasoning);
    kb(o, "supports_reasoning_effort", caps.supports_reasoning_effort);
    kb(o, "supports_string_content", caps.supports_string_content);
    kb(o, "supports_system_role", caps.supports_system_role);
    kb(o, "supports_tool_calls", caps.supports_tool_calls);
    kb(o, "supports_tools", caps.supports_tools);
    kb(o, "supports_typed_content", caps.supports_typed_content);
    k(o, "reasoning_effort_levels");
    o.push('[');
    for lv in caps.reasoning_effort_levels { es(o, jinja::effort_name(lv)); }
    o.push(']');
    o.push('}');
}

fn route_models() {
    var o: u8[>..] = [];
    o.push('{');
    k(o, "models");
    o.append("[{");
    ks(o, "name", model_name);
    ks(o, "model", model_name);
    ks(o, "modified_at", "");
    ks(o, "size", "");
    ks(o, "digest", "");
    ks(o, "type", "model");
    ks(o, "description", "");
    kraw(o, "tags", "[\"\"]");
    kraw(o, "capabilities", "[\"completion\"]");
    ks(o, "parameters", "");
    kraw(o, "details", "{\"parent_model\":\"\",\"format\":\"gguf\",\"family\":\"\",\"families\":[\"\"],\"parameter_size\":\"\",\"quantization_level\":\"\"}");
    o.append("}]");
    ks(o, "object", "list");
    k(o, "data");
    o.append("[{");
    ks(o, "id", model_name);
    k(o, "aliases");
    o.push('[');
    es(o, model_name);
    o.push(']');
    kraw(o, "tags", "[]");
    ks(o, "object", "model");
    kraw(o, "architecture", "{\"input_modalities\":[\"text\"],\"output_modalities\":[\"text\"]}");
    ki(o, "created", unix_time());
    ks(o, "owned_by", "llamacpp");
    k(o, "meta");
    o.push('{');
    ki(o, "vocab_type", vocab::vtype);
    ki(o, "n_vocab", vocab::n_tokens());
    ki(o, "n_ctx", c.n_ctx);
    ki(o, "n_ctx_train", m.hp.n_ctx_train);
    ki(o, "n_embd", m.hp.n_embd);
    ki(o, "n_params", n_params);
    ki(o, "size", model_size);
    ks(o, "ftype", ftype_name);
    o.push('}');
    o.append("}]");
    o.push('}');
    respond_json(o);
}

fn route_tokenize() {
    let body = body_json();
    var o: u8[>..] = [];
    o.append("{\"tokens\":[");
    let content = jdoc::get(body, "content");
    if content >= 0 {
        var tokens: i32[>..] = [];
        tokenize_mixed(content, jdoc::get_bool(body, "add_special", false), jdoc::get_bool(body, "parse_special", true), tokens);
        let with_pieces = jdoc::get_bool(body, "with_pieces", false);
        for t in tokens {
            if !with_pieces { ei(o, t); continue; }
            sep(o);
            o.push('{');
            ki(o, "id", t);
            let p = piece(t, true);
            k(o, "piece");
            if utf8_valid(p) { jdoc::put_str(o, p); }
            else {
                o.push('[');
                for b in p { ei(o, b); }
                o.push(']');
            }
            o.push('}');
        }
    }
    o.append("]}");
    respond_json(o);
}

// is_valid_utf8
fn utf8_valid(s: const u8[:]) -> bool {
    var i = 0;
    while i < s.len {
        if s[i] < 0x80 { i++; continue; }
        let n = jdoc::utf8_len(s, i);
        if n <= 0 { return false; }
        i += n;
    }
    return true;
}

fn route_detokenize() {
    let body = body_json();
    var content: u8[>..] = [];
    let tv = jdoc::get(body, "tokens");
    if tv >= 0 {
        var tokens: i32[>..] = [];
        for i in jdoc::len(tv) { tokens.push(jdoc::int_of(jdoc::at(tv, i)) as! i32); }
        for t in tokens {
            // llama_vocab's token_get_attr, which checks with .at()
            if t < 0 || t >= vocab::n_tokens() {
                fail(E_SERVER, str("vector::_M_range_check: __n (which is ", t as! u64, ") >= this->size() (which is ", vocab::n_tokens(), ")"));
            }
            content.append(piece(t, true));
        }
    }
    var o: u8[>..] = [];
    o.push('{');
    ks(o, "content", content);
    o.push('}');
    respond_json(o);
}

fn route_apply_template() {
    let body = body_json();
    var prompt: u8[>..] = [];
    chat_prompt(body, prompt);
    var o: u8[>..] = [];
    o.push('{');
    ks(o, "prompt", prompt);
    o.push('}');
    respond_json(o);
}

fn route_completion(res_type: i64) {
    let body = body_json();
    reset_task(res_type);
    if res_type == RES_OAI_CMPL {
        guard jdoc::contains(body, "prompt") else { fail(E_INVALID_REQUEST, "\"prompt\" is required"); }
        if jdoc::get_bool(body, "echo", false) { fail(E_INVALID_REQUEST, "Only no echo is supported"); }
        if jdoc::contains(body, "best_of") { fail(E_INVALID_REQUEST, "Unsupported param: best_of"); }
        if jdoc::contains(body, "suffix") { fail(E_INVALID_REQUEST, "Unsupported param: suffix"); }
    }
    let pv = jdoc::get(body, "prompt");
    guard pv >= 0 else { fail(E_INVALID_REQUEST, "[json.exception.out_of_range.403] key 'prompt' not found"); }
    var tokens: i32[>..] = [];
    // an array of prompts: one, or an error
    if jdoc::is_arr(pv) && jdoc::len(pv) > 0 && !has_number(pv) {
        guard jdoc::len(pv) == 1 else { fail(E_NOT_SUPPORTED, "several prompts in one request are not supported yet"); }
        let p = jdoc::at(pv, 0);
        if jdoc::is_obj(p) { tokenize_mixed(jdoc::get(p, "prompt_string"), true, true, tokens); } else { tokenize_mixed(p, true, true, tokens); }
    } else if jdoc::is_obj(pv) {
        tokenize_mixed(jdoc::get(pv, "prompt_string"), true, true, tokens);
    } else {
        tokenize_mixed(pv, true, true, tokens);
    }
    for t in tokens { if t < 0 || t >= vocab::n_tokens() { fail(E_INVALID_REQUEST, "Prompt contains invalid tokens"); } }
    eval_params(body);
    generate(tokens);
}

fn has_number(v: i64) -> bool {
    for i in jdoc::len(v) { if jdoc::is_num(jdoc::at(v, i)) { return true; } }
    return false;
}

fn route_chat() {
    let body = body_json();
    reset_task(RES_OAI_CHAT);
    var prompt: u8[>..] = [];
    chat_prompt(body, prompt);
    // the request as a completion's: its own grammar fields give way to the template's
    eval_params(body);
    t_grammar.clear();
    t_grammar_lazy = false;
    t_trig_types.clear();
    t_trig_values.clear();
    t_trig_tokens.clear();
    t_preserved.clear();
    let g = jdoc::get_str(body, "grammar", "");
    if g.len > 0 { t_grammar.append(g); t_grammar_user = true; }
    {
        // response_format and json_schema went through the template; else a schema of the request's own
        let s = jdoc::get(body, "json_schema");
        if chat_render::grammar.len == 0 && !jdoc::is_null(s) && g.len == 0 {
            var schema: u8[>..] = [];
            jdoc::dump(schema, s);
            if !json_schema::to_grammar(schema, t_grammar) { fail(E_INVALID_REQUEST, str("Field 'json_schema': \"json_schema\": ", json_schema::err)); }
        }
    }
    if jdoc::contains(body, "reasoning_format") { t_reasoning_format = reasoning_format_from_name(jdoc::get_str(body, "reasoning_format", "")); }
    apply_chat_session();
    var tokens: i32[>..] = [];
    vocab::tokenize(prompt, true, true, tokens);
    generate(tokens);
}

// A file of --path, for a GET of path ("/" is index.html): false if there
// is none. Paths that climb out of the directory are refused.
fn serve_file(path: const u8[:]) -> bool {
    guard find(path, "..") < 0 && find(path, "\\") < 0 else { return false; }
    var file: u8[>..] = [];
    file.append(static_dir);
    file.append(if path == "/" { "/index.html" } else { path });
    var data: u8[>..] = [];
    guard read_file(file, data) else { return false; }
    respond_with(200, content_type_of(file), data);
    return true;
}

fn content_type_of(file: const u8[:]) -> const u8[:] {
    if ends_with(file, ".html") { return "text/html; charset=utf-8"; }
    if ends_with(file, ".js") || ends_with(file, ".mjs") { return "text/javascript; charset=utf-8"; }
    if ends_with(file, ".css") { return "text/css; charset=utf-8"; }
    if ends_with(file, ".json") { return "application/json; charset=utf-8"; }
    if ends_with(file, ".webmanifest") { return "application/manifest+json"; }
    if ends_with(file, ".svg") { return "image/svg+xml"; }
    if ends_with(file, ".png") { return "image/png"; }
    if ends_with(file, ".ico") { return "image/x-icon"; }
    if ends_with(file, ".woff2") { return "font/woff2"; }
    if ends_with(file, ".txt") { return "text/plain; charset=utf-8"; }
    return "application/octet-stream";
}

fn authorized() -> bool {
    guard api_key.len > 0 else { return true; }
    let p = net::path();
    if p == "/health" || p == "/v1/health" || p == "/models" || p == "/v1/models" { return true; }
    let auth = net::header("Authorization");
    if auth == str("Bearer ", api_key) { return true; }
    if net::header("X-Api-Key") == api_key { return true; }
    return false;
}

// Serves the request in raw, its head and body, on the main thread.
fn handle(raw: const u8[:]) {
    streaming = false;
    client_gone = false;
    let st = net::take_request(raw);
    if st > 0 { respond_error(E_INVALID_REQUEST, net::status_text(st)); return; }
    let method = net::method();
    let path = str(net::path());
    log(str("request: ", method, " ", path));
    if !authorized() { respond_error(E_AUTHENTICATION, "Invalid API Key"); return; }
    let get = method == "GET";
    let post = method == "POST";
    fn body() {
        if get && (path == "/health" || path == "/v1/health") { respond_json("{\"status\":\"ok\"}"); }
        else if get && path == "/props" { route_props(); }
        else if get && (path == "/models" || path == "/v1/models") { route_models(); }
        else if post && (path == "/completion" || path == "/completions") { route_completion(RES_NONE); }
        else if post && path == "/v1/completions" { route_completion(RES_OAI_CMPL); }
        else if post && (path == "/chat/completions" || path == "/v1/chat/completions") { route_chat(); }
        else if post && path == "/apply-template" { route_apply_template(); }
        else if post && path == "/tokenize" { route_tokenize(); }
        else if post && path == "/detokenize" { route_detokenize(); }
        else if get && static_dir.len > 0 && serve_file(path) {}
        else { respond_error(E_NOT_FOUND, "File Not Found"); }
    }
    exception::try<Fail>(body) { f =>
        if streaming {
            var o: u8[>..] = [];
            o.append("{\"error\":");
            error_json(o, f.kind, fail_msg);
            o.push('}');
            send(o);
        } else {
            respond_error(f.kind, fail_msg);
        }
    };
}

//
// the threads: HTTP workers and the engine
//

// The engine's one job: a request, its head and body, in a shared blob,
// since a job must fit one mail message.
enum Job { Request { req: http::Shared } }
// What a request waits on: the engine, or nothing yet.
enum Use { None, Engine }
struct St { use: Use = Use.None }
struct App { engine: plod::Lane<Job, Use> }

let JSON_TYPE = "application/json; charset=utf-8";
// The longest request body.
let MAX_BODY: i64 = 256 * 1024 * 1024;
// How long a worker waits for each piece of a response: a request may wait
// behind others, and a long prompt takes a while before its first token.
let WAIT_MS: i64 = 3600 * 1000;

// The request the engine serves now: where its response goes, in messages
// of at most reply_piece bytes, and its job, which the first one echoes.
// Then the head of the response, which goes in that first message.
var reply_to = http::MailAddress { .. };
var reply_piece: i64 = 0;
var reply_job = default<Job>();
var head_msg: u8[>..] = [];
var begun = false;

// Starts the response: status, Content-Type, and the body's length, -1 for
// one sent in pieces as it comes (events). The worker writes it.
fn reply_head(status: i64, content_type: const u8[:], length: i64) {
    head_msg = plod::echo_head(reply_job);
    format(head_msg, status, " ", length, "\n", content_type, "\n");
    begun = false;
}

// Sends data, the next part of the response's body, in messages of at most
// a mail message's size; last ends the response. False once the client has
// gone. A full mailbox is waited out.
fn reply_part(data: const u8[:], last: bool) -> bool {
    var at = 0;
    loop {
        var m: u8[>..] = [];
        var tag: i32 = if begun { plod::REPLY_CONT } else { 0 };
        if !begun { m.append(head_msg); }
        let n = min(data.len - at, max(reply_piece - m.len, 0));
        m.append(data[at..at + n]);
        at += n;
        if at < data.len || !last { tag |= plod::REPLY_MORE; }
        loop {
            let sent = http::mail_send(reply_to, tag, m);
            if sent == http::SENT { break; }
            guard sent == http::FULL else { return false; }
            sleep_ms(1);
        }
        begun = true;
        guard at < data.len else { return true; }
    }
}

// A whole response.
fn respond_with(status: i64, content_type: const u8[:], data: const u8[:]) {
    reply_head(status, content_type, data.len);
    reply_part(data, true);
}

fn run(u: Job.Request, a: plod::Asked<Job>&) {
    var raw: u8[>..] = [];
    http::shared_append(u.req, 0, http::shared_len(u.req), raw);
    http::shared_release(u.req);
    a.replied = true;
    guard plod::still_wanted(a) else { return; }
    reply_to = a.reply;
    reply_piece = a.piece;
    reply_job = a.job;
    handle(raw);
    if streaming { reply_part("", true); }
}

// A whole response from a worker, with the fields every response has.
fn answer(out: http::Reply, status: i32, content_type: const u8[:], data: const u8[:]) {
    http::begin(out, status, data.len);
    if content_type.len > 0 { http::field(out, "Content-Type", content_type); }
    http::field(out, "Access-Control-Allow-Origin", "*");
    http::send(out, data);
    http::end(out);
}

// A request's first call, on a worker: CORS preflights, /health, and every
// request while the model loads (503) answered here, so they don't wait on
// the engine; the rest sent to it.
fn route(cx) {
    let method = cx.req.method;
    let target = cx.req.target;
    let q = find(target, "?");
    let path = if q >= 0 { target[..q] } else { target };
    if method == "OPTIONS" {
        http::begin(cx.out, 200, 0);
        http::field(cx.out, "Access-Control-Allow-Origin", "*");
        http::field(cx.out, "Access-Control-Allow-Credentials", "true");
        http::field(cx.out, "Access-Control-Allow-Methods", "GET, POST");
        http::field(cx.out, "Access-Control-Allow-Headers", "*");
        http::end(cx.out);
        return;
    }
    if net::ready() == 0 {
        answer(cx.out, 503, JSON_TYPE, "{\"error\":{\"code\":503,\"message\":\"Loading model\",\"type\":\"unavailable_error\"}}");
        return;
    }
    if method == "GET" && (path == "/health" || path == "/v1/health") {
        answer(cx.out, 200, JSON_TYPE, "{\"status\":\"ok\"}");
        return;
    }
    let body = plod::body(cx, MAX_BODY);
    var raw: u8[>..] = [];
    format(raw, method, " ", target, " HTTP/1.1\r\n");
    raw.append(cx.req.fields);
    raw.append("\r\n");
    raw.append(body);
    let b, err = http::shared_put(raw, 1);
    guard err == 0 else { plod::fail(cx, 503, "unavailable"); }
    let sent = plod::try_ask(cx, cx.app.engine, Job.Request { req: b }, WAIT_MS, 0);
    if sent != http::SENT {
        http::shared_release(b);
        plod::fail(cx, 503, "unavailable");
    }
}

fn resume(u: Use.None, cx) { plod::fail(cx, 500, "not_waiting"); }

// The response's first message, from the engine: "status length\ntype\n"
// and the body's first bytes. The rest, if any, plod::serve sends on.
fn resume(u: Use.Engine, cx) {
    let n = plod::echo_len(cx.reply);
    guard n >= 0 else { plod::fail(cx, 502, "bad_reply"); }
    let m = cx.reply[plod::ECHO_HEAD + n..];
    let nl = find(m, "\n");
    guard nl > 0 else { plod::fail(cx, 502, "bad_reply"); }
    let line = m[..nl];
    let rest = m[nl + 1..];
    let nl2 = find(rest, "\n");
    guard nl2 >= 0 else { plod::fail(cx, 502, "bad_reply"); }
    let sp = find(line, " ");
    guard sp > 0 else { plod::fail(cx, 502, "bad_reply"); }
    let status, ok1 = parse_int(line[..sp]);
    let length, ok2 = parse_int(line[sp + 1..]);
    guard ok1 && ok2 else { plod::fail(cx, 502, "bad_reply"); }
    let content_type = rest[..nl2];
    http::begin(cx.out, status as! i32, length);
    if content_type.len > 0 { http::field(cx.out, "Content-Type", content_type); }
    http::field(cx.out, "Access-Control-Allow-Origin", "*");
    if length < 0 { http::field(cx.out, "Cache-Control", "no-cache"); }
    plod::relay_piece(cx.out, rest[nl2 + 1..], cx.more);
}

// The HTTP threads: the workers, until a signal stops them; then the
// engine's loop ends too.
thread_fn http_main(cfg: http::Config, app: App) {
    http::run(cfg, app, worker);
    plod::lane_stop(app.engine);
}

thread_fn worker(cfg: http::Config, lanes: App) {
    var app = lanes;
    let e = plod::serve(cfg, app, St { .. }) { cx =>
        if plod::resumed(cx) {
            plod::on_reply(cx) { resume(cx.state.use, cx); };
        } else {
            plod::route(cx) { route(cx); };
        }
    };
    if e.code != 0 { http::report(e); }
}

//
// startup
//

// llama_ftype_name
fn ftype_of(ft: i64) -> const u8[:] {
    return match ft {
        0 => "all F32", 1 => "F16", 32 => "BF16", 2 => "Q4_0", 3 => "Q4_1", 8 => "Q5_0", 9 => "Q5_1", 7 => "Q8_0",
        38 => "MXFP4 MoE", 10 => "Q2_K - Medium", 21 => "Q2_K - Small", 11 => "Q3_K - Small", 12 => "Q3_K - Medium",
        13 => "Q3_K - Large", 14 => "Q4_K - Small", 15 => "Q4_K - Medium", 16 => "Q5_K - Small", 17 => "Q5_K - Medium",
        18 => "Q6_K", 36 => "TQ1_0 - 1.69 bpw ternary", 37 => "TQ2_0 - 2.06 bpw ternary", 19 => "IQ2_XXS - 2.0625 bpw",
        20 => "IQ2_XS - 2.3125 bpw", 28 => "IQ2_S - 2.5 bpw", 29 => "IQ2_M - 2.7 bpw", 22 => "IQ3_XS - 3.3 bpw",
        23 => "IQ3_XXS - 3.0625 bpw", 24 => "IQ1_S - 1.5625 bpw", 31 => "IQ1_M - 1.75 bpw", 25 => "IQ4_NL - 4.5 bpw",
        30 => "IQ4_XS - 4.25 bpw", 26 => "IQ3_S - 3.4375 bpw", 27 => "IQ3_S mix - 3.66 bpw",
        _ => "unknown, may not work",
    };
}

// The engine's startup: the arguments, the model, its context and templates.
fn setup() {
    let args = console::utf8_args();
    var host: u8[>..] = [];
    host.append("127.0.0.1");
    var port = 8080;
    var n_ctx = 0;
    var n_threads: i32 = 0;
    var alias: u8[>..] = [];
    var chat_template: u8[>..] = [];
    var breakers_given = false;
    var dry_breakers: u8[>..] = [];
    var i = 1;
    fn value() -> const u8[:] {
        if i + 1 >= args.len { die(str("missing value for ", args[i])); }
        i += 1;
        return args[i];
    }
    fn int_value() -> i64 {
        let v, ok = parse_int(value());
        if !ok { die(str("bad number for ", args[i - 1])); }
        return v;
    }
    fn flt_value() -> f32 {
        let v, ok = parse_flt(value());
        if !ok { die(str("bad number for ", args[i - 1])); }
        return v as f32;
    }
    var sp = sampler::Params { .. };
    while i < args.len {
        let a = args[i];
        if a == "-m" || a == "--model" { model_path.clear(); model_path.append(value()); }
        else if a == "--host" { host.clear(); host.append(value()); }
        else if a == "--port" { port = int_value(); }
        else if a == "-c" || a == "--ctx-size" { n_ctx = int_value(); }
        else if a == "-b" || a == "--batch-size" { n_batch = int_value(); }
        else if a == "-ub" || a == "--ubatch-size" { int_value(); }
        else if a == "-t" || a == "--threads" { n_threads = int_value() as! i32; }
        else if a == "-n" || a == "--n-predict" || a == "--predict" { base_n_predict = int_value(); }
        else if a == "--keep" { base_n_keep = int_value(); }
        else if a == "-a" || a == "--alias" { alias.clear(); alias.append(value()); }
        else if a == "--api-key" { api_key.clear(); api_key.append(value()); }
        else if a == "--path" { static_dir.clear(); static_dir.append(value()); }
        else if a == "-np" || a == "--parallel" {
            if int_value() != 1 { write_stderr("warning: one slot only: --parallel is ignored\n"); }
        }
        else if a == "--jinja" { use_jinja = true; }
        else if a == "--no-jinja" { use_jinja = false; }
        else if a == "--chat-template" { chat_template.clear(); chat_template.append(value()); }
        else if a == "--chat-template-file" {
            let path = value();
            chat_template.clear();
            if !read_file(path, chat_template) { die(str("failed to open file '", path, "'")); }
        }
        else if a == "--reasoning-format" {
            let v = value();
            reasoning_format = if v == "none" { chat_render::REASONING_NONE } else if v == "auto" { chat_render::REASONING_DEEPSEEK }
                else if v == "deepseek" { chat_render::REASONING_DEEPSEEK } else if v == "deepseek-legacy" { chat_render::REASONING_DEEPSEEK_LEGACY }
                else { die(str("unknown reasoning format: ", v)); 0 };
        }
        else if a == "--reasoning" {
            let v = value();
            enable_thinking = !(v == "off" || v == "0" || v == "false");
        }
        else if a == "--no-prefill-assistant" { prefill_assistant = false; }
        else if a == "-r" || a == "--reverse-prompt" { base_stops.append(value()); base_stops.push(0); }
        else if a == "-s" || a == "--seed" { sp.seed = (int_value() & 0xFFFFFFFF) as! u32; }
        else if a == "--temp" { sp.temp = max(flt_value(), 0.0); sp.user_set |= sampler::USER_TEMP; }
        else if a == "--top-k" { sp.top_k = int_value(); sp.user_set |= sampler::USER_TOP_K; }
        else if a == "--top-p" { sp.top_p = flt_value(); sp.user_set |= sampler::USER_TOP_P; }
        else if a == "--min-p" { sp.min_p = flt_value(); sp.user_set |= sampler::USER_MIN_P; }
        else if a == "--typical" { sp.typ_p = flt_value(); }
        else if a == "--repeat-penalty" { sp.penalty_repeat = flt_value(); sp.user_set |= sampler::USER_PENALTY_REPEAT; }
        else if a == "--repeat-last-n" { sp.penalty_last_n = int_value(); sp.user_set |= sampler::USER_PENALTY_LAST_N; }
        else if a == "--presence-penalty" { sp.penalty_present = flt_value(); }
        else if a == "--frequency-penalty" { sp.penalty_freq = flt_value(); }
        else if a == "--xtc-probability" { sp.xtc_probability = flt_value(); sp.user_set |= sampler::USER_XTC_PROBABILITY; }
        else if a == "--xtc-threshold" { sp.xtc_threshold = flt_value(); sp.user_set |= sampler::USER_XTC_THRESHOLD; }
        else if a == "--top-nsigma" { sp.top_n_sigma = flt_value(); }
        else if a == "--dynatemp-range" { sp.dynatemp_range = flt_value(); }
        else if a == "--dynatemp-exp" { sp.dynatemp_exponent = flt_value(); }
        else if a == "--mirostat" { sp.mirostat = int_value(); sp.user_set |= sampler::USER_MIROSTAT; }
        else if a == "--mirostat-lr" { sp.mirostat_eta = flt_value(); sp.user_set |= sampler::USER_MIROSTAT_ETA; }
        else if a == "--mirostat-ent" { sp.mirostat_tau = flt_value(); sp.user_set |= sampler::USER_MIROSTAT_TAU; }
        else if a == "--dry-multiplier" { sp.dry_multiplier = flt_value(); }
        else if a == "--dry-base" { let v = flt_value(); if v >= 1.0 { sp.dry_base = v; } }
        else if a == "--dry-allowed-length" { sp.dry_allowed_length = int_value(); }
        else if a == "--dry-penalty-last-n" { sp.dry_penalty_last_n = int_value(); }
        else if a == "--dry-sequence-breaker" {
            if !breakers_given { sp.dry_breakers.clear(); breakers_given = true; }
            let v = value();
            if v != "none" { dry_breakers.append(v); dry_breakers.push(0); }
        }
        else if a == "--no-webui" || a == "--webui" || a == "-fa" || a == "--flash-attn" || a == "--metrics" || a == "--slots" || a == "--cont-batching" {
            if (a == "-fa" || a == "--flash-attn") && i + 1 < args.len && !starts_with(args[i + 1], "-") { i++; }
        }
        else { die(str("unknown argument: ", a)); }
        i += 1;
    }
    if model_path.len == 0 { die("--model is required"); }
    if breakers_given {
        if dry_breakers.len > sp.dry_breakers.cap { die("the DRY sequence breakers are too long"); }
        sp.dry_breakers.append(dry_breakers);
    }
    model_name.append(if alias.len > 0 { alias[..] } else { model_path[..] });
    rng_state ^= time_ns() as! u64;

    model::load(m, model_path);
    c.n_threads = n_threads;
    context::init(c, m, if n_ctx > 0 { n_ctx } else { m.hp.n_ctx_train as i64 });
    sampler::apply_model_defaults(sp, m.meta);
    base_sp = sp;

    // the model's numbers for /models
    {
        let ftv = gguf::find_key(m.meta, "general.file_type");
        ftype_name.append(ftype_of(if ftv >= 0 { gguf::get_val_u32(m.meta, ftv) as i64 } else { -1 }));
        for id in gguf::get_n_tensors(m.meta) {
            var name: u8[>..] = [];
            gguf::get_tensor_name(m.meta, id, name);
            model_size += gguf::get_tensor_size(m.meta, id);
            let t = ggml::get_tensor(m.ctx, name);
            if t.p != 0 { n_params += ggml::nelements(t); }
        }
    }

    // the chat templates
    var tool_use: u8[>..] = [];
    if chat_template.len > 0 { tmpl_src.append(chat_template); }
    else if !sampler::meta_str(m.meta, "tokenizer.chat_template", tmpl_src) { tmpl_src.clear(); }
    if !sampler::meta_str(m.meta, "tokenizer.chat_template.tool_use", tool_use) { tool_use.clear(); }
    if vocab::special_bos_id != vocab::TOKEN_NULL { bos_text.append(piece(vocab::special_bos_id, true)); }
    if vocab::special_eos_id != vocab::TOKEN_NULL { eos_text.append(piece(vocab::special_eos_id, true)); }
    let ts0, ok = chat_apply::init(tmpl_src, tool_use, bos_text, eos_text, vocab::add_bos, vocab::add_eos);
    ts = ts0;
    if !ok { die(str("failed to initialize the chat template: ", chat_apply::err)); }
    if tmpl_src.len == 0 { tmpl_src.append(chat_render::source(ts.main)); }
    tmpl_tool_src.append(tool_use);
    // the sources as the templates keep them: Jinja drops one newline at the end
    if ends_with(tmpl_src, "\n") { tmpl_src.pop(); }
    if ends_with(tmpl_tool_src, "\n") { tmpl_tool_src.pop(); }
}

// The HTTP threads, started first, so that /health answers while the model
// loads; then the engine, on this thread, since only the program's first
// thread may hold the model's, vocab's and templates' references (spec
// §11.2): it loads them and serves the requests the workers send its box.
// Only --host and --port matter to the workers: setup reads every argument.
fn main() {
    let args = console::utf8_args();
    var cfg = http::Config { workers: 2, mail_slots: 64, mail_size: 16384, max_waiting: 1024,
                             max_body: MAX_BODY, max_drain: MAX_BODY, .. };
    cfg.address.append("127.0.0.1");
    var i = 1;
    while i + 1 < args.len {
        if args[i] == "--host" {
            cfg.address.clear();
            cfg.address.append(args[i + 1]);
            i++;
        } else if args[i] == "--port" {
            let p, ok = parse_int(args[i + 1]);
            if ok { cfg.port = p as! i32; }
            i++;
        }
        i++;
    }
    http::exit_on_error(http::stop_on_signals());
    http::gather_budget(MAX_BODY);
    let box, e = http::mail_open(64, 256);
    http::exit_on_error(e);
    let app = App { engine: plod::lane_of<Job>(box, Use.Engine) };
    let http_thread = thread_spawn(http_main, cfg, app);
    log(str("main: server is listening on http://", cfg.address, ":", cfg.port));
    setup();
    net::set_ready();
    log("main: model loaded");
    plod::lane(box, plod::Jobs<Job> { .. }) { u, a => run(u, a); };
    thread_wait(http_thread);
}
