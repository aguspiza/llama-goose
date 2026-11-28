// llama.cpp's llama-server: an HTTP server with llama.cpp's and OpenAI's
// APIs for one model. HTTP is goose-http's, through Plod: its worker
// threads take the connections and answer /health and CORS preflights; the
// main thread loads the model and serves the other requests, which the
// workers send it as Plod jobs. Those that generate each take a slot, 4 by
// default: the main thread runs the slots at work together, their tokens in
// one batch after another, in one KV cache they share, and keeps each
// request's state while it works on another's (with -np, each slot has a
// stream of the cache to itself, as llama-server without -kvu). The cache keeps each slot's
// last prompt, so that a request that starts as one of them reuses it. A
// client that goes away stops its generation. A model whose cache holds one
// sequence (a recurrent one, T5) has one slot.
//   server -m model.gguf [--host 127.0.0.1] [--port 9931] [-c n_ctx] [-np slots] [-kvu] ...
// The arguments are llama-server's (common/arg; -h lists them).
// Images: --mmproj file (-hf finds one beside the model, unless --no-mmproj), [--media-path dir] for file:// URLs.
// No -c (or -c 0): the model's context, fitted to the memory with -fit on|off, -fitt MiB, -fitc n: common/fit.
// Routes: /health, /props, /models, /completion(s), /v1/completions,
// /v1/chat/completions, /apply-template, /tokenize, /detokenize.
// With an mmproj, the chat's image_url parts and /completion's multimodal_data
// are images in the prompt (data: and base64; not http URLs yet).
// port-of: tools/server/server.cpp tools/server/server-context.cpp tools/server/server-task.cpp
// port-of: tools/server/server-common.cpp tools/server/server-schema.cpp tools/server/server-chat.cpp
// port-of: tools/server/main.cpp tools/server/server-slot-sched.cpp (get_reuse, the slot's choice)
import std;
import os;
import exception;
import slice_pool;
import base64;
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
import common.peg;
import common.console;
import common.jdoc;
import common.json_schema;
import common.download;
import common.fit;
import common.arg;
import common.chat_render;
import common.chat_peg;
import common.chat_apply;
import mtmd from "../mtmd";
import net from "./net";
import stash from "./stash";
import http from "github.com/kaitk/goose-http@v0.5.16";
import http.shared;
import plod from "github.com/kaitk/plod@067b57c804435ea9479d58ba55e3cc103788a151";

let BUILD_INFO = "b0-goose";

// task_response_type
enum ResponseKind { None, OaiCmpl, OaiChat }

// stop_type
enum Stop { None, Eos, Word, Limit }

// error_type
enum ErrorKind {
    InvalidRequest, Authentication, NotFound, Server, Permission, NotSupported, Unavailable,
    ExceedContext,
}

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
var ctx_shift = false;                   // --context-shift
var base_stops: u8[>..] = [];           // --reverse-prompt, each ended by a NUL
var n_batch = 2048;
var model_path: u8[>..] = [];
var model_name: u8[>..] = [];
var api_key: u8[>..] = [];
var static_dir: u8[>..] = [];           // --path: the files served under /, such as the web UI
var use_jinja = true;
var reasoning_format = chat_render::ReasoningFormat.Deepseek;
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

// --mmproj: images in the prompts, through mtmd (llama-server's mctx)
var mmproj_path: u8[>..] = [];
var mm = false;
var clip = mtmd::Clip { .. };
var media_marker: u8[>..] = [];         // what stands for an image in a prompt's text (get_media_marker)
var media_path: u8[>..] = [];           // --media-path, ending with a separator: where file:// URLs are

// An image in a prompt (server_tokens' media chunk): n tokens from position
// at, TOKEN_NULL in the prompt's tokens; id, the SHA-256 of its file. Those
// of the request (t_media) are its chunk of mtmd's chunks; those in the
// slots' tokens (medias) have the embeddings not decoded yet in media_embd
// (embd_len 0 once they are).
struct Media { slot: i64, at: i64, n: i64, id: u8[..32], chunk: i64, embd_at: i64, embd_len: i64 }
var t_media: Media[>..] = [];
var medias: Media[>..] = [];
var media_embd = slice_pool<f32> {};

fn embd_of(md: Media) -> f32[:] { media_embd.items[md.embd_at..md.embd_at + md.embd_len] }

// The one of list (of slot s's) that starts at pos, or -1.
fn media_at(list: const Media[:], s: i64, pos: i64) -> i64 {
    for md, i in list { if md.slot == s && md.at == pos { return i; } }
    return -1;
}

// The prompt's tokens from mtmd's chunks: an image's n tokens TOKEN_NULL, and the image in t_media.
fn tokens_of_chunks(out: i32[>..]&) {
    for k, ci in mtmd::chunks {
        if k.kind == mtmd::ChunkKind.Text { out.append(mtmd::chunk_text(k)); continue; }
        var md = Media { slot: -1, at: out.len, n: k.n, id: "", chunk: ci, embd_at: 0, embd_len: 0 };
        md.id.append(mtmd::bitmap_id(k.media));
        t_media.push(md);
        for _ in k.n { out.push(vocab::TOKEN_NULL); }
    }
}

// keep_first's rule: the prompt is not cut inside an image, but before it.
fn media_floor(n: i64) -> i64 {
    for md in t_media { if md.at < n && n < md.at + md.n { return md.at; } }
    return n;
}

// The images of slot s from position n on forgotten, with their embeddings.
fn drop_media(s: i64, n: i64) {
    var j = 0;
    for i in medias.len {
        let md = medias[i];
        if md.slot == s && md.at >= n {
            if md.embd_len > 0 { media_embd.free_slice(embd_of(md)); }
            continue;
        }
        medias[j] = md;
        j++;
    }
    while medias.len > j { medias.pop(); }
}

// get_common_prefix: how many of the prompt's first tokens slot s has; an
// image is the same when its id and its number of tokens are.
fn common_prefix(s: i64, prompt: const i32[:]) -> i64 {
    let have = tokens_of(s);
    var i = 0;
    while i < have.len && i < prompt.len {
        if have[i] == vocab::TOKEN_NULL && prompt[i] == vocab::TOKEN_NULL {
            let a = media_at(medias, s, i);
            let b = media_at(t_media, -1, i);
            guard a >= 0 && b >= 0 && medias[a].n == t_media[b].n && medias[a].id == t_media[b].id else { return i; }
            i += medias[a].n;
            continue;
        }
        guard have[i] == prompt[i] else { return i; }
        i++;
    }
    return i;
}

// fs_validate_filename, allow_subdirs: a relative path that does not climb out.
fn media_file_ok(p: const u8[:]) -> bool {
    guard p.len > 0 && p[0] != '/' && p[0] != '\\' && find(p, ":") < 0 else { return false; }
    var start = 0;
    for i in p.len + 1 {
        guard i == p.len || p[i] == '/' || p[i] == '\\' else { continue; }
        let part = p[start..i];
        if part.len == 0 || part == "." || part == ".." { return false; }
        start = i + 1;
    }
    return true;
}

// handle_media: an image's file, from a URL of data: (base64), file:// (under
// --media-path) or bare base64, decoded into mtmd's next bitmap.
fn load_media(url: const u8[:]) {
    var data: u8[>..] = [];
    if starts_with(url, "http") {
        fail(ErrorKind.InvalidRequest, "downloading media from a URL is not supported yet");
    } else if starts_with(url, "file://") {
        if media_path.len == 0 { fail(ErrorKind.InvalidRequest, "file:// URLs are not allowed unless --media-path is specified"); }
        let file = url[7..];
        if !media_file_ok(file) { fail(ErrorKind.InvalidRequest, str("file path is not allowed: ", file)); }
        if !read_file(str(media_path, file), data) { fail(ErrorKind.InvalidRequest, str("file does not exist or cannot be opened: ", file)); }
    } else if starts_with(url, "data:") {
        let comma = find(url, ",");
        if comma < 0 || find(url[comma + 1..], ",") >= 0 { fail(ErrorKind.InvalidRequest, "Invalid uri-encoded base64 value"); }
        let head = url[..comma];
        if !starts_with(head, "data:image/") && !starts_with(head, "data:video/") && !starts_with(head, "data:audio/") {
            fail(ErrorKind.InvalidRequest, str("Invalid uri format: ", head));
        }
        if !ends_with(head, "base64") { fail(ErrorKind.InvalidRequest, "uri must be base64 encoded"); }
        base64::decode(data, url[comma + 1..]);
    } else {
        if !base64::decode(data, url) || data.len == 0 { fail(ErrorKind.InvalidRequest, "Invalid base64 value"); }
    }
    add_file(data);
}

// mtmd_helper_bitmap_init_from_buf: the file decoded, or the request fails.
fn add_file(data: const u8[:]) {
    if !mtmd::add_bitmap(data) { fail(ErrorKind.InvalidRequest, "Failed to load image or audio file"); }
}

// process_mtmd_prompt: the prompt's text, with its markers, into tokens and images.
fn mtmd_prompt(text: const u8[:], out: i32[>..]&) {
    if !mtmd::tokenize(clip, text, true, media_marker) { fail(ErrorKind.InvalidRequest, "Failed to tokenize prompt"); }
    tokens_of_chunks(out);
}

// A slot: one of the requests served together, and the tokens it leaves in
// the KV cache, as the sequence of the slot's number, for the next request
// that starts as they do.
struct Slot {
    busy: bool = false,             // serving a request
    generating: bool = false,       // its prompt is in the cache: it samples
    n: i64 = 0,                     // its tokens, in slot_tokens: the prompt, then those generated
    n_kv: i64 = 0,                  // how many of them the cache has
    used_at: i64 = -1,              // when its last request ended
    out: i64 = -1,                  // in a step: the batch's token whose logits it samples from
    reply: http::MailAddress = http::MailAddress { .. },    // its client, to see whether it is still there
    state_at: i64 = 0,              // its request's state in states, while another's is in the globals
    state_len: i64 = 0,
    task: i64 = -1,                 // its request's number
    last_user: i64 = -1,            // where the prompt's last user message starts, -1 for none
}
var slots: Slot[>..] = [];
var slot_tokens: i32[>..] = [];     // room for c.n_ctx_seq tokens for each slot

fn tokens_of(s: i64) -> const i32[:] { slot_tokens[s * c.n_ctx_seq..s * c.n_ctx_seq + slots[s].n] }

//
// errors: what a request fails with, thrown to the route's handler
//

struct Fail { kind: ErrorKind }
var fail_msg: u8[>..] = [];

fn fail(kind: ErrorKind, msg: const u8[:]) {
    fail_msg.clear();
    fail_msg.append(msg);
    exception::throw(Fail { kind: kind });
}

fn error_code(kind: ErrorKind) -> i64 {
    return match kind {
        InvalidRequest => 400, Authentication => 401, NotFound => 404, Server => 500,
        Permission => 403, NotSupported => 501, Unavailable => 503, ExceedContext => 400,
    };
}

fn error_type_name(kind: ErrorKind) -> const u8[:] {
    return match kind {
        InvalidRequest => "invalid_request_error", Authentication => "authentication_error",
        NotFound => "not_found_error", Server => "server_error", Permission => "permission_error",
        NotSupported => "not_supported_error", Unavailable => "unavailable_error",
        ExceedContext => "exceed_context_size_error",
    };
}

// format_error_response
fn error_json(out: u8[>..]&, kind: ErrorKind, msg: const u8[:]) {
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
    guard jdoc::is_arr(v) else { fail(ErrorKind.InvalidRequest, "\"prompt\" elements must be a string, a list of tokens, a JSON object containing a prompt string, or a list of mixed strings & tokens."); }
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
            fail(ErrorKind.InvalidRequest, "\"prompt\" elements must be a string, a list of tokens, a JSON object containing a prompt string, or a list of mixed strings & tokens.");
        }
        first = false;
    }
}

//
// a task's parameters (task_params), from the request (server-schema.cpp)
//

struct LogitBias { token: i32, bias: f32 }

var t_res_type = ResponseKind.None;
var t_stream = false;
var t_include_usage = false;
var t_timings_per_token = false;
var t_return_tokens = false;
var t_cache_prompt = true;
var t_n_predict = -1;
var t_n_keep = 0;
var t_n_discard = 0;
var t_n_indent = 0;
var t_max_predict_ms = -1;
var t_sp = sampler::Params { .. };
var t_ignore_eos = false;
var t_stops: u8[>..] = [];              // each ended by a NUL
var t_logit_bias: LogitBias[>..] = [];
var t_grammar: u8[>..] = [];
var t_grammar_lazy = false;
var t_grammar_user = false;           // the request's own "grammar": not prefilled
var t_trig_types: chat_render::Trigger[>..] = [];
var t_trig_values: u8[>..] = [];        // each ended by a NUL
var t_trig_tokens: i32[>..] = [];       // the token of a TOKEN trigger, else -1
var t_preserved: i32[>..] = [];         // sorted
var t_chat = false;                     // the output is a chat message, parsed
var t_chat_format = chat_render::Format.ContentOnly;
var t_reasoning_format = chat_render::ReasoningFormat.None;
var t_gen_prompt: u8[>..] = [];
var t_cmpl_id: u8[>..] = [];

fn reset_task(res_type: ResponseKind) {
    t_res_type = res_type;
    t_stream = false;
    t_include_usage = false;
    t_timings_per_token = false;
    t_return_tokens = false;
    t_cache_prompt = true;
    t_n_predict = base_n_predict;
    t_n_keep = base_n_keep;
    t_n_discard = 0;
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
    t_chat_format = chat_render::Format.ContentOnly;
    t_reasoning_format = reasoning_format;
    t_gen_prompt.clear();
    t_cmpl_id.clear();
    t_media.clear();
    mtmd::clear_bitmaps();
}

// The value of the first of the names the request has (not null), or -1.
fn field(data: i64, a: const u8[:], b: const u8[:] = "", c2: const u8[:] = "") -> i64, const u8[:] {
    if jdoc::has(data, a) { return jdoc::get(data, a), a; }
    if b.len > 0 && jdoc::has(data, b) { return jdoc::get(data, b), b; }
    if c2.len > 0 && jdoc::has(data, c2) { return jdoc::get(data, c2), c2; }
    return -1, "";
}

fn field_fail(name: const u8[:], msg: const u8[:]) { fail(ErrorKind.InvalidRequest, str("Field '", name, "': ", msg)); }

fn want_num(v: i64, name: const u8[:]) {
    if !jdoc::is_num(v) { field_fail(name, str("[json.exception.type_error.302] type must be number, but is ", kind_name(v))); }
}

fn kind_name(v: i64) -> const u8[:] {
    return match jdoc::kind(v) {
        Null => "null", Bool => "boolean", Int => "number", Float => "number",
        String => "string", Array => "array", Object => "object",
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
    t_n_discard = num_i_hard(data, t_n_discard, 0, I32_MAX, "n_discard");
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
    if t_sp.n_probs > 0 { fail(ErrorKind.NotSupported, "n_probs: token probabilities are not supported yet"); }
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
                add_trigger(chat_render::trigger_of(jdoc::get_int(t, "type", 1)), jdoc::get_str(t, "value", ""));
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
            } else if jdoc::is_str(v) && jdoc::str_of(v).len > 0 {
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
                    let kind, ok = sampler::kind_from_name(jdoc::str_of(jdoc::at(v, i)));
                    if ok && t_sp.order.len < t_sp.order.cap { t_sp.order.push(kind); }
                }
            } else if jdoc::is_str(v) {
                t_sp.order.clear();
                for ch0 in jdoc::str_of(v) {
                    let kind, ok = sampler::kind_from_char(ch0);
                    if ok && t_sp.order.len < t_sp.order.cap { t_sp.order.push(kind); }
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

fn reasoning_format_from_name(name: const u8[:]) -> chat_render::ReasoningFormat {
    if name == "none" { return chat_render::ReasoningFormat.None; }
    if name == "auto" { return chat_render::ReasoningFormat.Auto; }
    if name == "deepseek" { return chat_render::ReasoningFormat.Deepseek; }
    if name == "deepseek-legacy" { return chat_render::ReasoningFormat.DeepseekLegacy; }
    fail(ErrorKind.InvalidRequest, str("Unknown reasoning format: ", name));
    return chat_render::ReasoningFormat.None;
}

fn reasoning_format_name(f: chat_render::ReasoningFormat) -> const u8[:] {
    return match f {
        None => "none", Auto => "auto",
        Deepseek => "deepseek", DeepseekLegacy => "deepseek-legacy",
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
fn add_trigger(ty: chat_render::Trigger, value: const u8[:]) {
    var tok: i32 = -1;
    var kind = ty;
    if ty == chat_render::Trigger.Word {
        var ids: i32[>..] = [];
        vocab::tokenize(value, false, true, ids);
        if ids.len == 1 {
            if !is_preserved(ids[0]) { fail(ErrorKind.InvalidRequest, str("Grammar trigger word should be marked as preserved token: ", value)); }
            kind = chat_render::Trigger.Token;
            tok = ids[0];
        }
    } else if ty == chat_render::Trigger.Token {
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
    ki(o, "n_discard", t_n_discard);
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
            ki(o, "type", chat_render::trigger_number(t_trig_types[n]));
            ks(o, "value", t_trig_values[start..i]);
            if t_trig_types[n] == chat_render::Trigger.Token { ki(o, "token", t_trig_tokens[n]); }
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
    kb(o, "reasoning_in_content", t_stream && t_reasoning_format == chat_render::ReasoningFormat.DeepseekLegacy);
    ks(o, "generation_prompt", t_gen_prompt);
    k(o, "samplers");
    o.push('[');
    for kind in t_sp.order { es(o, sampler::type_to_str(kind)); }
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
        fail(ErrorKind.InvalidRequest, str(what, d));
    }
    return jdoc::str_of(v);
}

// The prompt of a chat request (into prompt), and the task's chat parameters.
// Whether the template renders reasoning_effort = none as a level of its own.
fn effort_none_known() -> bool {
    let caps = if ts.has_tool_use { ts.tool_use.caps } else { ts.main.caps };
    for lv in caps.reasoning_effort_levels { if lv == 0 { return true; } }
    return false;
}

fn chat_prompt(body: i64, prompt: u8[>..]&) {
    let tools_v = jdoc::get(body, "tools");
    let has_tools = jdoc::is_arr(tools_v) && jdoc::len(tools_v) > 0;
    let stream = jdoc::get_bool(body, "stream", false);
    let tool_choice_name = jdoc::get_str(body, "tool_choice", "auto");
    if !use_jinja {
        if has_tools { fail(ErrorKind.InvalidRequest, "tools param requires --jinja flag"); }
        if tool_choice_name != "auto" { fail(ErrorKind.InvalidRequest, "tool_choice param requires --jinja flag"); }
    }
    var json_schema_text: u8[>..] = [];
    var has_schema = false;
    {
        let s = jdoc::get(body, "json_schema");
        if !jdoc::is_null(s) { jdoc::dump(json_schema_text, s); has_schema = true; }
    }
    let grammar = jdoc::get_str(body, "grammar", "");
    if has_schema && grammar.len > 0 { fail(ErrorKind.InvalidRequest, "Cannot use both json_schema and grammar"); }
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
            let s = if jdoc::contains(rf, "json_schema") { jdoc::get(jdoc::get(rf, "json_schema"), "schema") } else { jdoc::get(rf, "schema") };
            json_schema_text.clear();
            if s >= 0 { jdoc::dump(json_schema_text, s); } else { json_schema_text.append("{}"); }
            has_schema = true;
        } else if ty.len > 0 && ty != "text" {
            fail(ErrorKind.InvalidRequest, str("response_format type must be one of \"text\", \"json_object\" or \"json_schema\", but got: ", ty));
        }
    }
    if json_schema_text == "{}" { json_schema_text.clear(); json_schema_text.append("{\"type\":\"object\"}"); }

    guard jdoc::contains(body, "messages") else { fail(ErrorKind.InvalidRequest, "'messages' is required"); }
    let msgs_v = jdoc::get(body, "messages");
    guard jdoc::is_arr(msgs_v) else { fail(ErrorKind.InvalidRequest, "Expected 'messages' to be an array"); }

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
        if role != "assistant" && !has_content { fail(ErrorKind.InvalidRequest, "All non-assistant messages must contain 'content'"); }
        if role == "assistant" && !has_content && !has_calls { fail(ErrorKind.InvalidRequest, "Assistant message must contain either 'content' or 'tool_calls'!"); }
        guard jdoc::is_obj(msg) else { fail(ErrorKind.InvalidRequest, "Expected 'message' to be an object"); }
        guard jdoc::contains(msg, "role") else { fail(ErrorKind.InvalidRequest, "Missing 'role' in message"); }
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
                    guard jdoc::contains(part, "type") else { fail(ErrorKind.InvalidRequest, "Missing content part type"); }
                    if ty == "text" || ty == "media_marker" {
                        parts.push(PartRaw { kind: keep(ty), text: keep(jdoc::get_str(part, "text", "")), name: Span { at: 0, len: 0 } });
                    } else if ty == "tool_reference" {
                        parts.push(PartRaw { kind: keep(ty), text: Span { at: 0, len: 0 }, name: keep(jdoc::get_str(part, "name", "")) });
                    } else if ty == "image_url" {
                        // oaicompat_content_load_media: the image loaded, a marker in its place
                        if !mm { fail(ErrorKind.InvalidRequest, "image input is not supported - hint: if this is unexpected, you may need to provide the mmproj"); }
                        let iu = jdoc::get(part, "image_url");
                        load_media(if jdoc::is_obj(iu) { jdoc::get_str(iu, "url", "") } else { "" });
                        parts.push(PartRaw { kind: keep("media_marker"), text: keep(media_marker), name: Span { at: 0, len: 0 } });
                    } else if ty == "input_audio" {
                        fail(ErrorKind.InvalidRequest, "audio input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
                    } else if ty == "input_video" || ty == "video_url" {
                        fail(ErrorKind.InvalidRequest, "video input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
                    } else {
                        fail(ErrorKind.InvalidRequest, "unsupported content[].type");
                    }
                }
                r.nparts = parts.len - r.part0;
            } else if !jdoc::is_null(content) {
                fail(ErrorKind.InvalidRequest, "Expected 'content' to be a string or an array");
            }
        }
        if has_calls {
            let tcs = jdoc::get(msg, "tool_calls");
            for t in jdoc::len(tcs) {
                let tc = jdoc::at(tcs, t);
                guard jdoc::get_str(tc, "type", "") == "function" else { fail(ErrorKind.InvalidRequest, "Unsupported tool call type"); }
                let fc = jdoc::get(tc, "function");
                guard jdoc::contains(fc, "name") else { fail(ErrorKind.InvalidRequest, "Missing tool call name"); }
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
            guard jdoc::get_str(tool, "type", "") == "function" else { fail(ErrorKind.InvalidRequest, "Unsupported tool type"); }
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

    let tool_choice = if tool_choice_name == "auto" { chat_render::ToolChoice.Auto }
        else if tool_choice_name == "none" { chat_render::ToolChoice.None }
        else if tool_choice_name == "required" { chat_render::ToolChoice.Required }
        else { fail(ErrorKind.InvalidRequest, str("Invalid tool_choice: ", tool_choice_name)); chat_render::ToolChoice.Auto };
    let caps = if ts.has_tool_use { ts.tool_use.caps } else { ts.main.caps };
    var add_gen = jdoc::get_bool(body, "add_generation_prompt", true);
    var cont = chat_render::Continue.None;
    if jdoc::contains(body, "continue_final_message") {
        let cv = jdoc::get(body, "continue_final_message");
        if jdoc::is_bool(cv) { cont = if jdoc::bool_of(cv) { chat_render::Continue.Auto } else { chat_render::Continue.None }; }
        else if jdoc::is_str(cv) {
            let s = jdoc::str_of(cv);
            cont = if s == "auto" { chat_render::Continue.Auto } else if s == "reasoning" { chat_render::Continue.Reasoning }
                else if s == "content" { chat_render::Continue.Content } else if s == "none" { chat_render::Continue.None }
                else { fail(ErrorKind.InvalidRequest, str("Invalid continue_final_message: ", s)); 0 };
        }
    }
    let n = msgs.len;
    if cont == chat_render::Continue.None && prefill_assistant && n > 0 && msgs[n - 1].role == "assistant" {
        if n >= 2 && msgs[n - 2].role == "assistant" { fail(ErrorKind.InvalidRequest, "Cannot have 2 or more assistant messages at the end of the list."); }
        cont = chat_render::Continue.Auto;
        add_gen = false;
    }
    if cont != chat_render::Continue.None && add_gen { fail(ErrorKind.InvalidRequest, "Cannot set both add_generation_prompt and continue_final_message to true."); }
    if cont != chat_render::Continue.None && n > 0 && msgs[n - 1].role == "assistant" && msgs[n - 1].tool_calls.len > 0 {
        fail(ErrorKind.InvalidRequest, "Cannot continue an assistant message that contains tool calls.");
    }
    var rfmt = reasoning_format;
    if jdoc::contains(body, "reasoning_format") { rfmt = reasoning_format_from_name(jdoc::get_str(body, "reasoning_format", "")); }
    var thinking = enable_thinking;
    if tool_views.len > 0 && tool_choice != chat_render::ToolChoice.None && jdoc::contains(body, "grammar") {
        fail(ErrorKind.InvalidRequest, "Cannot use custom grammar constraints with tools.");
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
                    else if jdoc::is_str(v) { fail(ErrorKind.InvalidRequest, "invalid type for \"enable_thinking\" (expected boolean, got string)"); }
                }
            }
        }
        if jdoc::contains(body, "reasoning_effort") {
            let effort = jdoc::get_str(body, "reasoning_effort", "");
            let caps = if ts.has_tool_use { ts.tool_use.caps } else { ts.main.caps };
            var levels: u8[>..] = [];
            var none_level = false;
            for lv, i in caps.reasoning_effort_levels {
                if i > 0 { levels.append(", "); }
                levels.append(jinja::effort_name(lv));
                if lv == 0 { none_level = true; }
            }
            if effort == "none" {
                thinking = false;
                // some templates know reasoning_effort = none, others raise on it
                if none_level {
                    kw_text.append("reasoning_effort");
                    kw_ends.push(kw_text.len);
                    jdoc::put_str(kw_text, effort);
                    kw_ends.push(kw_text.len);
                }
            } else if effort.len > 0 {
                for lv in caps.reasoning_effort_rejected {
                    if jinja::effort_name(lv) == effort {
                        fail(ErrorKind.InvalidRequest, str("reasoning_effort \"", effort, "\" is not supported by the chat template, supported levels: ", levels));
                    }
                }
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
    // reasoning_effort "none" also drops the kwarg, unless the template knows it
    if jdoc::get_str(body, "reasoning_effort", "") == "none" && !effort_none_known() {
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
    if !chat_apply::apply(ts, inp) { fail(ErrorKind.InvalidRequest, chat_apply::err); }
    prompt.append(chat_render::prompt);
    gen_prompt_tokens(chat_render::gen_prompt, chat_apply::gen_prompt_toks);
    if stream && jdoc::get_bool(body, "logprobs", false) && has_tools { fail(ErrorKind.InvalidRequest, "logprobs is not supported with tools + stream"); }
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
    if t_grammar_lazy && t_trig_types.len == 0 { fail(ErrorKind.InvalidRequest, "Error: no triggers set for lazy grammar!"); }
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
var gen_itok: i32[>..] = [];        // the token each byte of generated starts, -1 for the rest of its piece
var n_sent = 0;
var stop_type = Stop.None;
var stopping_word: u8[>..] = [];
var has_new_line = false;
var truncated = false;
var n_gen = 0;
var n_prompt = 0;
var stats = Stats { cache_n: 0, prompt_n: 0, prompt_ms: 0.0, predicted_n: 0, predicted_ms: 0.0 };
var prompt_text: u8[>..] = [];      // the prompt's tokens as text, for the response
var streaming = false;              // the response is events, and has begun

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
        if word.len == 0 { continue; }
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
                stop_type = Stop.Word;
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

// The token each byte of the generation prompt starts (common_chat_input_tokenize);
// none if the pieces do not give back the text.
fn gen_prompt_tokens(text: const u8[:], out: i32[>..]&) {
    out.clear();
    var ids: i32[>..] = [];
    vocab::tokenize(text, false, true, ids);
    var back: u8[>..] = [];
    for id, i in ids {
        let pc = piece(id, true);
        // some tokenizers put a space before the first special token
        if i == 0 && pc.len > 0 && is_space_byte(pc[0]) && !(text.len > 0 && is_space_byte(text[0])) { continue; }
        if pc.len == 0 { continue; }
        back.append(pc);
        out.push(id);
        for k in pc.len - 1 { out.push(-1); }
    }
    if back != text { out.clear(); }
}

fn is_space_byte(c: u8) -> bool { c == ' ' || (c >= 9 && c <= 13) }

// The chat message of the output: parsed into chat_peg's content_text,
// reasoning_text and calls (an empty parse keeps the previous message).
fn parse_chat(text: const u8[:], partial: bool) -> bool {
    // the tokens of the text, which is the generation so far or a start of it
    let known = text.len <= generated.len && gen_itok.len == generated.len && generated[..text.len] == text;
    let toks = if known { gen_itok[..text.len] } else { gen_itok[..0] };
    if !chat_apply::parse_tokens(text, toks, partial) {
        if partial { return false; }
        log(str("parse_chat: unparsed ", chat_apply::format_name(chat_render::format), " output: ", chat_apply::input_text));
        fail(ErrorKind.Server, chat_apply::err);
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
            fail(ErrorKind.Server, str("Invalid diff: '", last, "' not found at start of '", cur, "'"));
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
    if ncalls < n_prev_calls { fail(ErrorKind.Server, "Invalid diff: now finding less tool calls!"); }
    if n_prev_calls > 0 {
        let idx = n_prev_calls - 1;
        let pname = nth(prev_calls, 3 * idx);
        let pargs = nth(prev_calls, 3 * idx + 1);
        let pid = nth(prev_calls, 3 * idx + 2);
        let nname = nth(calls, 3 * idx);
        let nargs = nth(calls, 3 * idx + 1);
        let nid = nth(calls, 3 * idx + 2);
        if pname != nname && pname.len > 0 && nname.len > 0 && !starts_with(nname, pname) { fail(ErrorKind.Server, "Invalid diff: tool call mismatch!"); }
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
    if !streaming {
        reply_head(200, "text/event-stream", -1);
        streaming = true;
    }
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
    let with_stats = stop_type != Stop.None || t_timings_per_token;
    if with_stats { finish_stats(); }
    if t_res_type == ResponseKind.None {
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
    } else if t_res_type == ResponseKind.OaiCmpl {
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
    if piece0.len > 0 {
        gen_itok.push(tok);
        for i in piece0.len - 1 { gen_itok.push(-1); }
    }
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
    if !ctx_shift && slots[loaded].n + 1 >= c.n_ctx_seq {
        truncated = true;
        stop_type = Stop.Limit;
        has_next = false;
    }
    if n_gen > 0 && has_next && t_n_predict != -1 && t_n_predict - n_gen <= 0 {
        stop_type = Stop.Limit;
        has_next = false;
    }
    if has_new_line && t_n_indent > 0 {
        if last_nl_pos > 0 {
            var pos = last_nl_pos;
            var n_indent = 0;
            while pos < generated.len && (generated[pos] == ' ' || generated[pos] == '\t') { n_indent++; pos++; }
            if pos < generated.len && n_indent < t_n_indent {
                stop_type = Stop.Limit;
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
            stop_type = Stop.Limit;
            has_next = false;
        }
    }
    if vocab::is_eog(tok) {
        stop_type = Stop.Eos;
        has_next = false;
    }
    return has_next;
}
var has_new_line_now = false;
var last_nl_pos = 0;

//
// the slots: each request's state, kept while the globals hold another's
//

// What a slot's request keeps from one token to the next, here and in the
// modules its generation uses: all of it, in one list.
fn slot_state(st: stash::Stash&) {
    // the task
    stash::keep_value(st, t_res_type);
    stash::keep_value(st, t_stream);
    stash::keep_value(st, t_include_usage);
    stash::keep_value(st, t_timings_per_token);
    stash::keep_value(st, t_return_tokens);
    stash::keep_value(st, t_cache_prompt);
    stash::keep_value(st, t_n_predict);
    stash::keep_value(st, t_n_keep);
    stash::keep_value(st, t_n_discard);
    stash::keep_value(st, t_n_indent);
    stash::keep_value(st, t_max_predict_ms);
    stash::keep_value(st, t_sp);
    stash::keep_value(st, t_ignore_eos);
    stash::keep(st, t_stops);
    stash::keep(st, t_logit_bias);
    stash::keep(st, t_grammar);
    stash::keep_value(st, t_grammar_lazy);
    stash::keep_value(st, t_grammar_user);
    stash::keep(st, t_trig_types);
    stash::keep(st, t_trig_values);
    stash::keep(st, t_trig_tokens);
    stash::keep(st, t_preserved);
    stash::keep_value(st, t_chat);
    stash::keep_value(st, t_chat_format);
    stash::keep_value(st, t_reasoning_format);
    stash::keep(st, t_gen_prompt);
    stash::keep(st, t_cmpl_id);
    // the generation
    stash::keep(st, generated);
    stash::keep(st, gen_itok);
    stash::keep(st, chat_apply::gen_prompt_toks);
    stash::keep(st, gen_tokens);
    stash::keep(st, fed);
    stash::keep_value(st, n_sent);
    stash::keep_value(st, stop_type);
    stash::keep(st, stopping_word);
    stash::keep_value(st, has_new_line);
    stash::keep_value(st, has_new_line_now);
    stash::keep_value(st, last_nl_pos);
    stash::keep_value(st, truncated);
    stash::keep_value(st, n_gen);
    stash::keep_value(st, n_prompt);
    stash::keep_value(st, stats);
    stash::keep(st, prompt_text);
    stash::keep_value(st, t_prompt_start);
    stash::keep_value(st, t_gen_start);
    stash::keep_value(st, t_last);
    // the chat message so far
    stash::keep(st, prev_content);
    stash::keep(st, prev_reasoning);
    stash::keep(st, prev_calls);
    stash::keep(st, call_ids);
    stash::keep_value(st, n_prev_calls);
    // the response
    stash::keep_value(st, streaming);
    stash::keep_value(st, client_gone);
    stash::keep_value(st, reply_to);
    stash::keep_value(st, reply_piece);
    stash::keep_value(st, reply_job);
    stash::keep(st, head_msg);
    stash::keep_value(st, begun);
    // the sampler, and its grammar
    stash::keep_value(st, ch);
    stash::keep(st, sampler::dry_heads);
    stash::keep(st, sampler::dry_tails);
    stash::keep(st, grammar::elems);
    stash::keep(st, grammar::rule_at);
    stash::keep_value(st, grammar::partial);
    stash::keep(st, grammar::sdata);
    stash::keep(st, grammar::sets);
    stash::keep_value(st, grammar::cur_n);
    stash::keep_value(st, grammar::awaiting);
    stash::keep(st, grammar::tbuf);
    stash::keep(st, grammar::tpos);
    stash::keep(st, grammar::ttokens);
    stash::keep(st, grammar::twords);
    stash::keep(st, grammar::tword_at);
    // the chat format's parser
    stash::keep(st, peg::pt);
    stash::keep(st, peg::ps);
    stash::keep(st, peg::kids);
    stash::keep(st, peg::ranges);
    stash::keep(st, peg::rules);
    stash::keep(st, peg::arenas);
    stash::keep_value(st, peg::cur);
    stash::keep_value(st, chat_render::format);
    stash::keep(st, chat_render::gen_prompt);
    stash::keep_value(st, chat_render::parser);
    stash::keep_value(st, chat_apply::content_only);
}

// The slot whose request's state the globals hold, or -1; the other slots
// at work have theirs in states.
var loaded = -1;
var states = slice_pool<u8> {};

fn state_of(s: i64) -> u8[:] { states.items[slots[s].state_at..slots[s].state_at + slots[s].state_len] }

// Puts the globals' state away, for a request of another slot or of none.
fn unload() {
    guard loaded >= 0 else { return; }
    var st = stash::Stash { .. };
    slot_state(st);
    let run = states.alloc_copy(st.bytes);
    slots[loaded].state_at = states.items.range_index(run);
    slots[loaded].state_len = run.len;
    loaded = -1;
}

// Brings the state of slot s, which is at work, back to the globals.
fn load(s: i64) {
    guard loaded != s else { return; }
    unload();
    var st = stash::Stash { loading: true, .. };
    st.bytes.append(state_of(s));
    states.free_slice(state_of(s));
    slot_state(st);
    loaded = s;
}

// The slot's request is over (release): the slot is free for another.
fn release(s: i64) {
    if loaded == s { loaded = -1; } else { states.free_slice(state_of(s)); }
    slots[s].busy = false;
    slots[s].generating = false;
    slots[s].out = -1;
    slots[s].used_at = clock_ns();
}

// The end of the response of slot s, whose state is loaded.
fn finish(s: i64) {
    if streaming { reply_part("", true); }
    release(s);
}

// prompt_clear: the slot's tokens forgotten, their cells free.
fn forget(s: i64) {
    context::keep_first(c, m, s, 0);
    drop_checkpoints(s);
    drop_media(s, 0);
    slots[s].n = 0;
    slots[s].n_kv = 0;
}

fn any_busy() -> bool {
    for sl in slots { if sl.busy { return true; } }
    return false;
}

fn any_free() -> bool {
    for sl in slots { if !sl.busy { return true; } }
    return false;
}

// get_available_slot: of the slots free, the one whose tokens start most
// like the prompt, if more than a tenth of it; else one with no tokens, or
// the one used longest ago.
//
// context checkpoints (common_prompt_checkpoint): what a slot's cache cannot
// get back by forgetting its last positions, the cells of the sliding-window
// layers' cache, saved while a prompt is evaluated; a later prompt that starts
// as it did evaluates from one of them on.
//

struct Checkpoint {
    slot: i64,
    task: i64,                      // the request that made it
    n_tokens: i64,                  // the prompt's tokens in the cache then
    pos_min: i64,                   // the positions of the slot's sequence then
    pos_max: i64,
    pos_at: i64,                    // the cells' positions in ckpt_pos
    pos_len: i64,
    data_at: i64,                   // their keys and values in ckpt_data
    data_len: i64,
}
var ckpts: Checkpoint[>..] = [];    // oldest first
var ckpt_pos = slice_pool<i64> {};
var ckpt_data = slice_pool<u8> {};
var n_ctx_checkpoints = 32;         // a slot's most (--ctx-checkpoints)
let CHECKPOINT_MIN_STEP = 8192;     // the fewest tokens between two
var n_tasks = 0;
var user_starts: i64[>..] = [];     // slot * n_ctx + the position of a user message's start, in the slots' prompts

// A model whose cache a slot cannot cut back is checkpointed: one with
// sliding-window layers (the recurrent ones, also checkpointed by llama-server, are not yet).
fn checkpointing() -> bool { n_ctx_checkpoints > 0 && c.n_swa_cells > 0 }

fn drop_checkpoint(i: i64) {
    let ck = ckpts[i];
    ckpt_pos.free_slice(ckpt_pos.items[ck.pos_at..ck.pos_at + ck.pos_len]);
    ckpt_data.free_slice(ckpt_data.items[ck.data_at..ck.data_at + ck.data_len]);
    for j in i..ckpts.len - 1 { ckpts[j] = ckpts[j + 1]; }
    ckpts.pop();
}

fn drop_checkpoints(s: i64) {
    var i = ckpts.len - 1;
    while i >= 0 {
        if ckpts[i].slot == s { drop_checkpoint(i); }
        i--;
    }
}

fn n_checkpoints(s: i64) -> i64 {
    var n = 0;
    for ck in ckpts { if ck.slot == s { n++; } }
    return n;
}

// The newest checkpoint of slot s, or -1.
fn last_checkpoint(s: i64) -> i64 {
    var i = ckpts.len - 1;
    while i >= 0 && ckpts[i].slot != s { i--; }
    return i;
}

fn is_user_start(s: i64, pos: i64) -> bool {
    for x in user_starts { if x == s * c.n_ctx_seq + pos { return true; } }
    return false;
}

// create_checkpoint: before the batch, whose last n_cur tokens are slot s's, is decoded.
fn create_checkpoint(s: i64, n_cur: i64, pos_min: i64, pos_max: i64) {
    let task = slots[s].task;
    // once the list is full, those within the step of an earlier one go, but this request's
    var last = -1;
    var i = 0;
    while i < ckpts.len && n_checkpoints(s) + 1 >= n_ctx_checkpoints {
        if ckpts[i].slot != s { i++; continue; }
        if ckpts[i].task != task && last >= 0 && ckpts[i].n_tokens <= last + CHECKPOINT_MIN_STEP {
            drop_checkpoint(i);
            continue;
        }
        last = ckpts[i].n_tokens;
        i++;
    }
    while n_checkpoints(s) >= n_ctx_checkpoints {
        var j = 0;
        while ckpts[j].slot != s { j++; }
        drop_checkpoint(j);
    }
    // one at the same point replaced
    let n_tokens = slots[s].n_kv - n_cur;
    var k = ckpts.len - 1;
    while k >= 0 {
        if ckpts[k].slot == s && ckpts[k].n_tokens == n_tokens { drop_checkpoint(k); }
        k--;
    }
    var pos: i64[>..] = [];
    var data: u8[>..] = [];
    context::save_swa(c, m, s, pos, data);
    let pr = ckpt_pos.alloc_copy(pos);
    let pos_at = ckpt_pos.items.range_index(pr);
    let dr = ckpt_data.alloc_copy(data);
    let data_at = ckpt_data.items.range_index(dr);
    ckpts.push(Checkpoint { slot: s, task: task, n_tokens: n_tokens, pos_min: pos_min, pos_max: pos_max,
                            pos_at: pos_at, pos_len: pos.len, data_at: data_at, data_len: data.len });
}

// Where the user messages of a chat prompt start (common_chat_msg_spans), for slot s.
fn find_user_starts(s: i64, prompt: const i32[:]) {
    var i = user_starts.len - 1;
    while i >= 0 {
        if user_starts[i] / c.n_ctx_seq == s {
            for j in i..user_starts.len - 1 { user_starts[j] = user_starts[j + 1]; }
            user_starts.pop();
        }
        i--;
    }
    slots[s].last_user = -1;
    guard t_chat && checkpointing() && chat_render::user_start.len > 0 else { return; }
    var delim: i32[>..] = [];
    vocab::tokenize(chat_render::user_start, false, true, delim);
    guard delim.len > 0 else { return; }
    for p in prompt.len {
        if p + delim.len <= prompt.len && prompt[p..p + delim.len] == delim[..] {
            user_starts.push(s * c.n_ctx_seq + p);
            slots[s].last_user = p;
        }
    }
}

fn pick_slot(prompt: const i32[:]) -> i64 {
    var ret = -1;
    var sim_best: f32 = 0.0;
    let sim_min: f32 = 0.1;
    for sl, s in slots {
        guard !sl.busy && sl.n > 0 else { continue; }
        let lcp = common_prefix(s, prompt);
        let sim = lcp as f32 / prompt.len;
        if sim > sim_best && sim > sim_min { sim_best = sim; ret = s; }
    }
    if ret >= 0 { return ret; }
    for sl, s in slots {
        guard !sl.busy else { continue; }
        if ret < 0 || (sl.n == 0 && slots[ret].n > 0) ||
           ((sl.n == 0) == (slots[ret].n == 0) && sl.used_at < slots[ret].used_at) { ret = s; }
    }
    return ret;
}

// launch_slot_with_task: the request, its prompt's tokens, takes a slot,
// which the steps to come run: what the slot's cache lacks of the prompt,
// then a token at a time until a stop. Its state stays in the globals.
fn start(prompt: const i32[:]) {
    generated.clear();
    gen_itok.clear();
    gen_tokens.clear();
    fed.clear();
    n_sent = 0;
    stop_type = Stop.None;
    stopping_word.clear();
    has_new_line = false;
    has_new_line_now = false;
    last_nl_pos = 0;
    truncated = false;
    n_gen = 0;
    n_prompt = prompt.len;
    reset_chat_state();
    if t_cmpl_id.len == 0 { t_cmpl_id.append("chatcmpl-"); t_cmpl_id.append(random_string()); }

    if prompt.len == 0 { fail(ErrorKind.InvalidRequest, "\"prompt\" must not be empty"); }
    // the sampler (common_sampler_init) and the grammar
    sampler::init(ch, t_sp);
    ch.has_grammar = false;
    if t_grammar.len > 0 {
        // ignore_eos bans the end-of-generation tokens as llama.cpp's logit biases do
        var banned = t_ignore_eos;
        for lb in t_logit_bias { if lb.bias == -F_INF && vocab::is_eog(lb.token) { banned = true; } }
        if banned { fail(ErrorKind.InvalidRequest, "Failed to initialize samplers: EOG tokens cannot be banned when a grammar is set"); }
    }
    if t_grammar.len > 0 {
        if !sampler::set_grammar(ch, t_grammar, "root") {
            fail(ErrorKind.InvalidRequest, str("Failed to initialize samplers: failed to parse grammar: ", grammar::err));
        }
        if t_grammar_lazy {
            grammar::make_lazy();
            for ty, i in t_trig_types {
                if ty == chat_render::Trigger.Token && t_trig_tokens[i] >= 0 { grammar::add_trigger_token(t_trig_tokens[i]); }
                else if ty == chat_render::Trigger.Word { grammar::add_trigger_word(nth(t_trig_values, i)); }
                else { fail(ErrorKind.NotSupported, "regex grammar triggers are not supported yet"); }
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

    let s = pick_slot(prompt);
    if prompt.len >= c.n_ctx_seq {
        slots[s].used_at = clock_ns();      // the slot took the request, and released it
        fail(ErrorKind.ExceedContext, str("request (", prompt.len, " tokens) exceeds the available context size (", c.n_ctx_seq, " tokens), try increasing it"));
    }
    // the cache: the common prefix kept, at least one token to evaluate
    var n_past = 0;
    if t_cache_prompt { n_past = common_prefix(s, prompt); }
    // server_prompt::get_reuse: the cache is of use when it starts before the window of the
    // first token to evaluate (a SWA model), or before that token (others); else the newest
    // checkpoint that does, from which the prompt is evaluated
    let n_prefix = n_past;
    var pos_next = n_past;
    if n_past > 0 && !context::any_recurrent(m) {
        let n_swa = if m.hp.swa_type != model::Swa.None { m.hp.n_swa as i64 } else { 0 };
        let pos_eval = n_past - (if n_past < prompt.len { 0 } else { 1 });
        let thold = max(0, if n_swa > 0 { pos_eval - n_swa + 2 } else { pos_eval });
        if context::seq_pos_min(s) >= thold {
            var i = ckpts.len - 1;
            while i >= 0 {
                let ck = ckpts[i];
                if ck.slot == s && ck.pos_max <= pos_next && (ck.pos_min < thold || ck.pos_min == 0) { break; }
                i--;
            }
            if i < 0 {
                n_past = 0;
                pos_next = 0;
            } else {
                let ck = ckpts[i];
                pos_next = min(pos_next, max(ck.pos_min + 1, ck.pos_max));
                n_past = min(pos_next, ck.n_tokens);
                context::load_swa(c, m, s, ckpt_pos.items[ck.pos_at..ck.pos_at + ck.pos_len],
                                  ckpt_data.items[ck.data_at..ck.data_at + ck.data_len]);
            }
        }
    }
    // those past the prompt's start in common, or past where it is evaluated from, go
    var i = ckpts.len - 1;
    while i >= 0 {
        if ckpts[i].slot == s && (ckpts[i].pos_max > pos_next || ckpts[i].n_tokens > n_prefix) { drop_checkpoint(i); }
        i--;
    }
    if n_past == prompt.len && n_past > 0 { n_past--; }
    // not inside an image: from its start
    var n_keep = media_floor(n_past);
    while true {
        n_past = context::keep_first(c, m, s, n_keep);
        n_keep = media_floor(n_past);
        if n_keep == n_past { break; }
    }
    drop_media(s, n_past);
    for t, i in prompt { slot_tokens[s * c.n_ctx_seq + i] = t; }
    slots[s].n = prompt.len;
    slots[s].n_kv = n_past;
    slots[s].generating = false;
    slots[s].reply = reply_to;
    slots[s].busy = true;
    slots[s].task = n_tasks;
    n_tasks++;
    find_user_starts(s, prompt);
    loaded = s;
    stats = Stats { cache_n: n_past, prompt_n: prompt.len - n_past, prompt_ms: 0.0, predicted_n: 0, predicted_ms: 0.0 };
    prompt_text.clear();
    var text_tokens: i32[>..] = [];
    for t in prompt { if t != vocab::TOKEN_NULL { text_tokens.push(t); } }
    prompt_text.append(detokenize(text_tokens, true));
    t_prompt_start = now_ms();
    // the images the cache lacks, encoded (process_mtmd_chunk encodes them
    // as the slot reaches them), to be decoded in the steps to come
    for md in t_media {
        guard md.at >= n_past else { continue; }
        let k = mtmd::chunks[md.chunk];
        var img = mtmd::ImageF32 { nx: k.nx, ny: k.ny, buf: [] };
        img.buf.append(mtmd::chunk_image(k));
        var embd: f32[>..] = [];
        mtmd::encode_image(clip, img, embd);
        let run = media_embd.alloc_copy(embd);
        medias.push(Media { slot: s, at: md.at, n: md.n, id: md.id, chunk: -1, embd_at: media_embd.items.range_index(run), embd_len: run.len });
    }
}

// update_slots: one batch for the slots at work, decoded: the token each
// generating one sampled last, then the prompts' next tokens, up to n_batch
// in all; and a token sampled for each slot whose logits the batch gives. A
// slot whose client has gone stops first.
fn step() {
    for s in slots.len {
        if slots[s].busy && !http::mail_live(slots[s].reply) {
            slots[s].n = slots[s].n_kv;
            drop_media(s, slots[s].n);
            release(s);
        }
    }
    // context shift (pre_decode): a slot whose next token does not fit keeps
    // its n_keep first tokens and drops n_discard, half the rest by default, after them
    for s in slots.len {
        guard ctx_shift && slots[s].busy && slots[s].generating && slots[s].n_kv + 1 >= c.n_ctx_seq else { continue; }
        load(s);
        var n_keep = if t_n_keep < 0 { n_prompt } else { t_n_keep };
        if vocab::add_bos { n_keep += 1; }
        n_keep = min(c.n_ctx_seq - 4, n_keep);
        let n_left = slots[s].n_kv - n_keep;
        let n_discard = clamp(if t_n_discard != 0 { t_n_discard } else { n_left / 2 }, 0, max(0, n_left - 1));
        write_stderr(str("slot context shift, id ", s, ", n_keep = ", n_keep, ", n_left = ", n_left, ", n_discard = ", n_discard, "\n"));
        context::seq_rm(c, s, n_keep, n_keep + n_discard);
        context::seq_add(c, s, n_keep + n_discard, slots[s].n_kv, -n_discard);
        // the tokens after those dropped moved back, the one to decode next with them
        let at = s * c.n_ctx_seq;
        for i in n_keep + n_discard..slots[s].n { slot_tokens[at + i - n_discard] = slot_tokens[at + i]; }
        slots[s].n -= n_discard;
        slots[s].n_kv -= n_discard;
        truncated = true;
    }
    var tokens: i32[>..] = [];
    var pos: i64[>..] = [];
    var seqs: u64[>..] = [];
    var want: i64[>..] = [];        // the slots to sample for, in the batch's order
    for s in slots.len {
        guard slots[s].busy && slots[s].generating else { continue; }
        let bit: u64 = 1 << s;
        slots[s].out = tokens.len;
        want.push(s);
        tokens.push(slot_tokens[s * c.n_ctx_seq + slots[s].n_kv]);
        pos.push(slots[s].n_kv);
        seqs.push(bit);
        slots[s].n_kv++;
    }
    for s in slots.len {
        if slots[s].busy && !slots[s].generating {
            let bit: u64 = 1 << s;
            var do_checkpoint = checkpointing();
            // the images where the slot is, each decoded on its own first (process_mtmd_chunk)
            var has_mtmd = false;
            while slots[s].n_kv < slots[s].n && slot_tokens[s * c.n_ctx_seq + slots[s].n_kv] == vocab::TOKEN_NULL {
                let j = media_at(medias, s, slots[s].n_kv);
                let md = medias[j];
                var mpos: i64[>..] = [];
                var mseqs: u64[>..] = [];
                for k in md.n { mpos.push(md.at + k); mseqs.push(bit); }
                if !context::fits(mseqs) {
                    load(s);
                    send_error(ErrorKind.Server, "failed to process mtmd chunk");
                    finish(s);
                    break;
                }
                context::decode_embd(c, m, embd_of(md), mpos, s, c.n_ubatch, !mtmd::use_non_causal(clip));
                media_embd.free_slice(embd_of(md));
                medias[j].embd_len = 0;
                slots[s].n_kv += md.n;
                has_mtmd = true;
            }
            guard slots[s].busy else { continue; }
            let n_start = slots[s].n_kv;
            let n_prompt_s = slots[s].n;
            while slots[s].n_kv < n_prompt_s && tokens.len < n_batch {
                // the end of a text chunk: the image after it, in the next step
                guard slot_tokens[s * c.n_ctx_seq + slots[s].n_kv] != vocab::TOKEN_NULL else { break; }
                tokens.push(slot_tokens[s * c.n_ctx_seq + slots[s].n_kv]);
                pos.push(slots[s].n_kv);
                seqs.push(bit);
                slots[s].n_kv++;
                guard do_checkpoint else { continue; }
                let n_kv = slots[s].n_kv;
                // a batch ends at the last user message, or at one a step past the last checkpoint
                if is_user_start(s, n_kv) {
                    let li = last_checkpoint(s);
                    if n_kv == slots[s].last_user || li < 0 || n_kv > ckpts[li].n_tokens + CHECKPOINT_MIN_STEP { break; }
                }
                // and 4 + n_ubatch and 4 tokens before the prompt's end, for a checkpoint there
                if n_prompt_s == n_kv + min(n_batch, 4 + c.n_ubatch) || n_prompt_s == n_kv + min(n_batch, 4) { break; }
            }
            let n_cur = slots[s].n_kv - n_start;
            let near_end = n_prompt_s < slots[s].n_kv + c.n_ubatch;
            if slots[s].n_kv == n_prompt_s && slots[s].n_kv > n_start {
                slots[s].out = tokens.len - 1;
                want.push(s);
            } else if !is_user_start(s, n_start) && !near_end {
                do_checkpoint = false;
            }
            // a step that decoded an image cuts its batch as the others, but keeps no checkpoint
            do_checkpoint = do_checkpoint && !has_mtmd;
            if do_checkpoint {
                let pos_min = context::seq_pos_min(s);
                let li = last_checkpoint(s);
                if pos_min >= 0 && (li < 0 || n_start == slots[s].last_user || near_end ||
                                    n_start > ckpts[li].n_tokens + CHECKPOINT_MIN_STEP) {
                    create_checkpoint(s, n_cur, pos_min, context::seq_pos_max(s));
                }
            }
        }
        if tokens.len >= n_batch { break; }
    }

    let n_vocab = vocab::n_tokens();
    var logits: f32[>..] = [];
    var outs: i32[>..] = [];
    var whose: i64[>..] = [];
    var nb = n_batch;
    var i = 0;
    while i < tokens.len {
        let n = min(nb, tokens.len - i);
        if !context::fits(seqs[i..i + n]) {
            // llama_decode's 1: no room. A slot not at work gives up its cells; else a smaller part is tried
            if nb == 1 { fail_all("Context size has been exceeded."); return; }
            if !purge_idle_slot() { nb /= 2; }
            continue;
        }
        outs.clear();
        whose.clear();
        for s in want {
            if slots[s].out >= i && slots[s].out < i + n {
                outs.push((slots[s].out - i) as! i32);
                whose.push(s);
            }
        }
        context::decode_seqs(c, m, tokens[i..i + n], pos[i..i + n], seqs[i..i + n], logits, outs);
        for s, k in whose {
            slots[s].out = -1;
            sample_slot(s, logits[k * n_vocab..(k + 1) * n_vocab]);
        }
        i += n;
        nb = n_batch;
    }
}

// The next token of slot s, sampled from its logits, and what it brings:
// a part of the response, or its end.
fn sample_slot(s: i64, row: const f32[:]) {
    load(s);
    if !slots[s].generating {
        // init_sampler: the prompt's tokens into the penalties' windows
        sampler::reset(ch);
        for t in tokens_of(s) { sampler::accept(ch, t); }
        slots[s].generating = true;
    }
    var logits: f32[>..] = [];
    logits.append(row);
    for lb in t_logit_bias { logits[lb.token] += lb.bias; }
    if t_ignore_eos {
        for id in vocab::eog_ids { logits[id] = -F_INF; }
    }
    var cands: sampler::Cand[>..] = [];
    let tok = sampler::sample(ch, cands, logits);
    n_gen++;
    t_last = now_ms();
    if n_gen == 1 {
        t_gen_start = t_last;
        stats.prompt_ms = t_last - t_prompt_start;
    }
    let text = piece(tok, is_preserved(tok));
    var more = false;
    fn body() {
        more = process_token(tok, text);
        if !more { finish_stats(); send_final(); }
    }
    exception::try<Fail>(body) { f => send_error(f.kind, fail_msg); more = false; };
    if more && !client_gone {
        slot_tokens[s * c.n_ctx_seq + slots[s].n] = tok;
        slots[s].n++;
    } else {
        finish(s);
    }
}

// The batch cannot be decoded: every slot at work fails, and forgets its tokens.
fn fail_all(msg: const u8[:]) {
    for s in slots.len {
        guard slots[s].busy else { continue; }
        load(s);
        send_error(ErrorKind.Server, msg);
        finish(s);
        forget(s);
    }
}

// try_clear_idle_slots: the first slot not at work that has tokens forgets
// them, for room in the cache the slots share; false if there is none.
fn purge_idle_slot() -> bool {
    guard c.unified else { return false; }
    for s in slots.len {
        if !slots[s].busy && slots[s].n > 0 { forget(s); return true; }
    }
    return false;
}

fn stop_type_name(s: Stop) -> const u8[:] {
    return match s { None => "none", Eos => "eos", Word => "word", Limit => "limit" };
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
    ki(o, "id_slot", loaded);
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
    ki(o, "tokens_cached", slots[loaded].n);
    k(o, "timings");
    stats_json(o);
    o.push('}');
}

// send_final_response, in the response's format.
fn send_final() {
    let finish = if stop_type == Stop.Word || stop_type == Stop.Eos { "stop" } else { "length" };
    var o: u8[>..] = [];
    if t_res_type == ResponseKind.None {
        final_json(o);
        if t_stream { send(o); } else { respond_with(200, "application/json; charset=utf-8", o); }
        return;
    }
    if t_res_type == ResponseKind.OaiCmpl {
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
    let chat_finish = if stop_type == Stop.Word || stop_type == Stop.Eos { if n_prev_calls > 0 { "tool_calls" } else { "stop" } } else { "length" };
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

fn respond_error(kind: ErrorKind, msg: const u8[:]) {
    var o: u8[>..] = [];
    o.append("{\"error\":");
    error_json(o, kind, msg);
    o.push('}');
    respond_with(error_code(kind), "application/json; charset=utf-8", o);
}

// A request's failure: an event if its response is events begun already,
// else the whole response.
fn send_error(kind: ErrorKind, msg: const u8[:]) {
    guard streaming else { respond_error(kind, msg); return; }
    var o: u8[>..] = [];
    o.append("{\"error\":");
    error_json(o, kind, msg);
    o.push('}');
    send(o);
}

// The request's body as JSON, or a 400.
fn body_json() -> i64 {
    jdoc::reset();
    let v = jdoc::parse(net::body);
    if v < 0 { fail(ErrorKind.InvalidRequest, str("[json.exception.parse_error.101] parse error: ", jdoc::err)); }
    return v;
}

fn route_props() {
    reset_task(ResponseKind.None);
    t_reasoning_format = chat_render::ReasoningFormat.None;
    var o: u8[>..] = [];
    o.push('{');
    k(o, "default_generation_settings");
    o.push('{');
    k(o, "params");
    params_json(o, true);
    ki(o, "n_ctx", c.n_ctx_seq);
    o.push('}');
    ki(o, "total_slots", slots.len);
    ks(o, "model_alias", model_name);
    ks(o, "model_ftype", ftype_name);
    ks(o, "model_path", model_path);
    kraw(o, "modalities", if mm { "{\"vision\":true,\"video\":false,\"audio\":false}" } else { "{\"vision\":false,\"video\":false,\"audio\":false}" });
    ks(o, "media_marker", media_marker);
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
    k(o, "reasoning_effort_rejected");
    o.push('[');
    for lv in caps.reasoning_effort_rejected { es(o, jinja::effort_name(lv)); }
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
    kraw(o, "capabilities", if mm { "[\"completion\",\"multimodal\"]" } else { "[\"completion\"]" });
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
    kraw(o, "architecture", if mm { "{\"input_modalities\":[\"text\",\"image\"],\"output_modalities\":[\"text\"]}" }
                            else { "{\"input_modalities\":[\"text\"],\"output_modalities\":[\"text\"]}" });
    ki(o, "created", unix_time());
    ks(o, "owned_by", "llamacpp");
    k(o, "meta");
    o.push('{');
    ki(o, "vocab_type", vocab::type_number(vocab::vtype));
    ki(o, "n_vocab", vocab::n_tokens());
    ki(o, "n_ctx", c.n_ctx_seq);
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
                fail(ErrorKind.Server, str("vector::_M_range_check: __n (which is ", t as! u64, ") >= this->size() (which is ", vocab::n_tokens(), ")"));
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

fn route_completion(res_type: ResponseKind) {
    let body = body_json();
    reset_task(res_type);
    if res_type == ResponseKind.OaiCmpl {
        guard jdoc::contains(body, "prompt") else { fail(ErrorKind.InvalidRequest, "\"prompt\" is required"); }
        if jdoc::get_bool(body, "echo", false) { fail(ErrorKind.InvalidRequest, "Only no echo is supported"); }
        if jdoc::contains(body, "best_of") { fail(ErrorKind.InvalidRequest, "Unsupported param: best_of"); }
        if jdoc::contains(body, "suffix") { fail(ErrorKind.InvalidRequest, "Unsupported param: suffix"); }
    }
    let pv = jdoc::get(body, "prompt");
    guard pv >= 0 else { fail(ErrorKind.InvalidRequest, "[json.exception.out_of_range.403] key 'prompt' not found"); }
    var tokens: i32[>..] = [];
    // an array of prompts: one, or an error
    if jdoc::is_arr(pv) && jdoc::len(pv) > 0 && !has_number(pv) {
        guard jdoc::len(pv) == 1 else { fail(ErrorKind.NotSupported, "several prompts in one request are not supported yet"); }
        let p = jdoc::at(pv, 0);
        if jdoc::is_obj(p) { subprompt_object(p, tokens); } else { tokenize_mixed(p, true, true, tokens); }
    } else if jdoc::is_obj(pv) {
        subprompt_object(pv, tokens);
    } else {
        tokenize_mixed(pv, true, true, tokens);
    }
    for t in tokens {
        guard t_media.len == 0 || t != vocab::TOKEN_NULL else { continue; }
        if t < 0 || t >= vocab::n_tokens() { fail(ErrorKind.InvalidRequest, "Prompt contains invalid tokens"); }
    }
    eval_params(body);
    start(tokens);
}

// A prompt object: its prompt_string, and the images of its multimodal_data (base64).
fn subprompt_object(p: i64, out: i32[>..]&) {
    let ps = jdoc::get(p, "prompt_string");
    guard jdoc::contains(p, "multimodal_data") else { tokenize_mixed(ps, true, true, out); return; }
    if !mm { fail(ErrorKind.InvalidRequest, "Multimodal data provided, but model does not support multimodal requests."); }
    let md = jdoc::get(p, "multimodal_data");
    for i in jdoc::len(md) {
        var data: u8[>..] = [];
        base64::decode(data, jdoc::str_of(jdoc::at(md, i)));
        add_file(data);
    }
    mtmd_prompt(need_str(ps, "\"prompt_string\" must be a string: "), out);
}

fn has_number(v: i64) -> bool {
    for i in jdoc::len(v) { if jdoc::is_num(jdoc::at(v, i)) { return true; } }
    return false;
}

fn route_chat() {
    let body = body_json();
    reset_task(ResponseKind.OaiChat);
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
            if !json_schema::to_grammar(schema, t_grammar) { fail(ErrorKind.InvalidRequest, str("Field 'json_schema': \"json_schema\": ", json_schema::err)); }
        }
    }
    if jdoc::contains(body, "reasoning_format") { t_reasoning_format = reasoning_format_from_name(jdoc::get_str(body, "reasoning_format", "")); }
    apply_chat_session();
    var tokens: i32[>..] = [];
    if mm { mtmd_prompt(prompt, tokens); } else { vocab::tokenize(prompt, true, true, tokens); }
    start(tokens);
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

// Serves the request in raw, its head and body, on the main thread: all of
// it, or, for one that generates, as far as giving it a slot.
fn handle(raw: const u8[:]) {
    streaming = false;
    client_gone = false;
    let st = net::take_request(raw);
    if st > 0 { respond_error(ErrorKind.InvalidRequest, net::status_text(st)); return; }
    let method = net::method();
    let path = str(net::path());
    log(str("request: ", method, " ", path));
    if !authorized() { respond_error(ErrorKind.Authentication, "Invalid API Key"); return; }
    let get = method == "GET";
    let post = method == "POST";
    fn body() {
        if get && (path == "/health" || path == "/v1/health") { respond_json("{\"status\":\"ok\"}"); }
        else if get && path == "/props" { route_props(); }
        else if get && (path == "/models" || path == "/v1/models") { route_models(); }
        else if post && (path == "/completion" || path == "/completions") { route_completion(ResponseKind.None); }
        else if post && path == "/v1/completions" { route_completion(ResponseKind.OaiCmpl); }
        else if post && (path == "/chat/completions" || path == "/v1/chat/completions") { route_chat(); }
        else if post && path == "/apply-template" { route_apply_template(); }
        else if post && path == "/tokenize" { route_tokenize(); }
        else if post && path == "/detokenize" { route_detokenize(); }
        else if get && static_dir.len > 0 && serve_file(path) {}
        else { respond_error(ErrorKind.NotFound, "File Not Found"); }
    }
    exception::try<Fail>(body) { f => send_error(f.kind, fail_msg); };
}

//
// the threads: HTTP workers and the engine
//

// The engine's one job: a request, its head and body, in a shared blob,
// since a job must fit one mail message; and whether it generates, and so
// waits for a slot.
enum Job { Request { req: http::Shared, slot: bool } }
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

// A request that generates, until a slot is free: where its response goes,
// and its job.
struct Waiting { reply: http::MailAddress, piece: i64, job: Job }
var waiting: Waiting[>..] = [];

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

// A job from a worker: its request served at once, unless it generates;
// then it waits its turn for a slot.
fn run(u: Job.Request, a: plod::Asked<Job>&) {
    a.replied = true;
    let w = Waiting { reply: a.reply, piece: a.piece, job: a.job };
    if u.slot { waiting.push(w); } else { serve(w); }
}

fn blob_of(job: Job) -> http::Shared {
    return match job { Request r => r.req };
}

// Serves the request of w, if its client still waits, with no other
// request's state in the globals.
fn serve(w: Waiting) {
    let req = blob_of(w.job);
    var raw: u8[>..] = [];
    http::shared_append(req, 0, http::shared_len(req), raw);
    http::shared_release(req);
    guard http::mail_live(w.reply) else { return; }
    unload();
    reply_to = w.reply;
    reply_piece = w.piece;
    reply_job = w.job;
    handle(raw);
}

// The engine's loop, until the workers stop: the jobs they sent taken, the
// requests that wait given the slots free, in their order, and a step for
// the slots at work. With none at work it waits for a job.
fn engine(box: http::Box) {
    let jobs = plod::Jobs<Job> { .. };
    while !http::mail_stopped(box) {
        if !any_busy() { http::mail_wait(box, -1); }
        http::mail_take(box) { key, tag, bytes => plod::lane_job(jobs, bytes) { u, a => run(u, a); }; };
        var served = 0;
        while served < waiting.len && any_free() { serve(waiting[served]); served++; }
        for i in waiting.len - served { waiting[i] = waiting[i + served]; }
        waiting.resize(waiting.len - served);
        if any_busy() { step(); }
    }
    http::mail_close(box);
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
    let slot = method == "POST" && (path == "/completion" || path == "/completions" || path == "/v1/completions" ||
                                    path == "/chat/completions" || path == "/v1/chat/completions");
    let sent = plod::try_ask(cx, cx.app.engine, Job.Request { req: b, slot: slot }, WAIT_MS, 0);
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
    let p = arg::params;
    var n_ctx = p.n_ctx;
    let n_threads = if p.n_threads > 0 { p.n_threads as! i32 } else { ggml::default_threads() };   // postprocess_cpu_params
    var alias: u8[>..] = [];            // the first of the aliases
    for ch0 in p.model_alias { if ch0 == 0 { break; } alias.push(ch0); }
    let chat_template = p.chat_template;
    var n_slots = p.n_parallel;         // -1: llama-server's default
    var kv_unified = p.kv_unified;
    model_path.append(p.model_path);
    mmproj_path.append(p.mmproj_path);
    media_path.append(p.media_path);
    n_batch = p.n_batch;
    base_n_predict = p.n_predict;
    base_n_keep = p.n_keep;
    ctx_shift = p.ctx_shift;
    for ch0 in p.api_keys { if ch0 == 0 { break; } api_key.push(ch0); }    // the first key
    static_dir.append(p.public_path);
    n_ctx_checkpoints = p.n_ctx_checkpoints;
    use_jinja = p.use_jinja;
    reasoning_format = match p.reasoning_format {
        None => chat_render::ReasoningFormat.None, DeepseekLegacy => chat_render::ReasoningFormat.DeepseekLegacy,
        _ => chat_render::ReasoningFormat.Deepseek,
    };
    enable_thinking = p.enable_reasoning != 0;
    prefill_assistant = p.prefill_assistant;
    base_stops.append(p.antiprompt);
    var sp = p.sampling;
    // the server downloads the model (common_params_handle_models) itself
    download::resolve(model_path, mmproj_path, !p.no_mmproj);
    if model_path.len == 0 { die("--model is required"); }
    model_name.append(if alias.len > 0 { alias[..] } else { model_path[..] });
    rng_state ^= time_ns() as! u64;
    // get_media_marker: random, so that no text is taken for one, unless LLAMA_MEDIA_MARKER pins it
    if !env("LLAMA_MEDIA_MARKER", media_marker) || media_marker.len == 0 {
        media_marker.clear();
        media_marker.append(str("<__media_", random_string(), "__>"));
    }

    model::load(m, model_path);
    c.n_threads = n_threads;
    c.n_ubatch = min(n_batch, p.n_ubatch);
    // the slots: llama-server's 4 by default, in one cache unified for them; those
    // of -np, a stream of the cache each unless -kvu
    if context::any_recurrent(m) || m.hp.arch == model::Arch.T5 {
        if n_slots > 1 { write_stderr("warning: this model's cache holds one sequence: one slot\n"); }
        n_slots = 1;
    }
    if n_slots <= 0 {
        n_slots = 4;
        kv_unified = true;
    }
    if n_slots > 64 { die("at most 64 slots"); }
    c.unified = kv_unified && n_slots > 1;
    c.n_seq_max = n_slots;
    c.n_outputs_max = min(n_slots, n_batch);    // a token of each slot (server_output_limits)
    // no -c: the context the model was trained with, for each stream, fitted to the memory (common_fit_params)
    context::init(c, m, fit::n_ctx(c, m, n_ctx));
    if mmproj_path.len > 0 {
        mtmd::load_clip(clip, mmproj_path, c.n_threads);
        if mtmd::n_mmproj_embd(clip) != m.hp.n_embd {
            write_stderr(str("mismatch between text model (n_embd = ", m.hp.n_embd, ") and mmproj (n_embd = ", mtmd::n_mmproj_embd(clip), ")\n"));
            write_stderr("hint: you may be using wrong mmproj\n");
            die(str("failed to load multimodal model, '", mmproj_path, "'"));
        }
        log(str("loaded multimodal model, '", mmproj_path, "'"));
        mm = true;
        if ctx_shift {
            ctx_shift = false;
            write_stderr("warning: ctx_shift is not supported by multimodal, it will be disabled\n");
        }
    }
    if ctx_shift && !context::can_shift(c, m) {
        write_stderr("warning: ctx_shift is not supported by this context, it will be disabled\n");
        ctx_shift = false;
    }
    for s in n_slots { slots.push(Slot { .. }); }
    slot_tokens.resize(n_slots * c.n_ctx_seq, 0);
    sampler::apply_model_defaults(sp, m.meta);
    base_sp = sp;

    // the model's numbers for /models
    {
        let ftv = gguf::find_key(m.meta, "general.file_type");
        ftype_name.append(ftype_of(if ftv >= 0 { gguf::get_val_u32(m.meta, ftv) as i64 } else { -1 }));
        for id in gguf::get_n_tensors(m.meta) {
            var name: u8[>..] = [];
            gguf::get_tensor_name(m.meta, id, name);
            // llama_model_size and llama_model_n_params: the tensors loaded
            let t = ggml::get_tensor(m.ctx, name);
            guard t.p != 0 && model::is_used(t) else { continue; }
            model_size += gguf::get_tensor_size(m.meta, id);
            n_params += ggml::nelements(t);
        }
    }

    // the chat templates
    var tool_use: u8[>..] = [];
    if chat_template.len > 0 { tmpl_src.append(chat_template); }
    else if !sampler::meta_str(m.meta, "tokenizer.chat_template", tmpl_src) { tmpl_src.clear(); }
    if !sampler::meta_str(m.meta, "tokenizer.chat_template.tool_use", tool_use) { tool_use.clear(); }
    if vocab::special_bos_id != vocab::TOKEN_NULL { bos_text.append(piece(vocab::special_bos_id, true)); }
    if vocab::special_eos_id != vocab::TOKEN_NULL { eos_text.append(piece(vocab::special_eos_id, true)); }
    // the special tokens the chat parsers match by id (common_peg_tokens(vocab))
    {
        var pieces: const u8[:][>..] = [];
        var ids: i64[>..] = [];
        for id in vocab::n_tokens() {
            guard vocab::is_control(id) || vocab::is_user_defined(id) else { continue; }
            pieces.push(vocab::token_text(id));
            ids.push(id);
        }
        peg::set_tokens(pieces, ids);
    }
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
// The arguments (common/arg); only --host and --port matter to the workers.
fn main() {
    let args = console::utf8_args();
    var cfg = http::Config { workers: 2, mail_slots: 64, mail_size: 16384, max_waiting: 1024,
                             max_body: MAX_BODY, max_drain: MAX_BODY, .. };
    if !arg::parse(args, arg::Example.Server) { exit(1); }
    // the first address of --host
    for ch0 in arg::params.hostname { if ch0 == ',' { break; } cfg.address.push(ch0); }
    cfg.port = arg::params.port as! i32;
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
    engine(box);
    thread_wait(http_thread);
}
