// llama.cpp's chat templates applied with jinja the whole way, as
// common_chat_templates_apply does: the conversation and tools as JSON,
// the template's workarounds, and the parser of the model's output (a
// template's dedicated one, or the autoparser's) with its grammar and
// triggers; and that output parsed back into a message.
// port-of: common/chat.cpp (templates_apply_jinja, to_json_oaicompat, msgs_parse_oaicompat, common_chat_peg_parse, workarounds)
import std;
import exception;
import os;
import .jinja;
import .jinja_caps;
import .number;
import .json_schema;
import .peg;
import .chat_peg;
import .chat_render;
import .autoparser;
import .chat_parsers;

namespace chat_apply;

struct ToolCall { name: const u8[:], arguments: const u8[:], id: const u8[:] = "" }
struct Part { kind: const u8[:], text: const u8[:] = "", name: const u8[:] = "" }

// common_chat_msg
struct Message {
    role: const u8[:],
    content: const u8[:] = "",
    reasoning: const u8[:] = "",
    tool_calls: const ToolCall[:] = [],
    parts: const Part[:] = [],
    tool_name: const u8[:] = "",
    tool_call_id: const u8[:] = "",
    extra_json: const u8[:] = "",
}

// common_chat_tool: the parameters a JSON schema's text
struct Tool { name: const u8[:], description: const u8[:] = "", parameters: const u8[:] = "{}", defer_loading: bool = false }

struct KV { key: const u8[:], value: const u8[:] }

// common_chat_templates_inputs
struct Inputs {
    messages: const Message[:] = [],
    grammar: const u8[:] = "",
    json_schema: const u8[:] = "",
    add_generation_prompt: bool = true,
    continue_final_message: i64 = chat_render::CONTINUE_NONE,
    tools: const Tool[:] = [],
    tool_choice: i64 = chat_render::TOOL_CHOICE_AUTO,
    parallel_tool_calls: bool = false,
    reasoning_format: i64 = chat_render::REASONING_NONE,
    enable_thinking: bool = true,
    kwargs: const KV[:] = [],
    force_pure_content: bool = false,
}

// common_chat_templates: the default template and the one for tools.
struct Templates {
    main: chat_render::Template = chat_render::Template { .. },
    tool_use: chat_render::Template = chat_render::Template { .. },
    has_tool_use: bool = false,
    add_bos: bool = false,
    add_eos: bool = false,
}

let CHATML = "{%- for message in messages -%}\n  {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n  {{- '<|im_start|>assistant\n' -}}\n{%- endif -%}";

var err: u8[>..] = [];

fn make(src: const u8[:], bos: const u8[:], eos: const u8[:]) -> chat_render::Template {
    var t = chat_render::make_template(src, bos, eos);
    t.handler = chat_parsers::detect(chat_render::source(t));
    return t;
}

// common_chat_templates_init with a template's source (empty or "chatml"
// for ChatML, or the tool-use one when there is one); false, and the
// message in err, if it does not parse.
fn init(src: const u8[:], tool_use_src: const u8[:], bos: const u8[:], eos: const u8[:], add_bos: bool, add_eos: bool) -> Templates, bool {
    var ts = Templates { add_bos: add_bos, add_eos: add_eos, .. };
    var ok = false;
    var main_src: u8[>..] = [];
    if src.len == 0 || src == "chatml" {
        patch_source(if tool_use_src.len > 0 { tool_use_src } else { CHATML }, main_src);
    } else {
        patch_source(src, main_src);
    }
    var tsrc: u8[>..] = [];
    tsrc.append(tool_use_src);
    fn body() {
        ts.main = make(main_src, bos, eos);
        if tsrc.len > 0 {
            ts.tool_use = make(tsrc, bos, eos);
            ts.has_tool_use = true;
        }
        ok = true;
    }
    exception::try<jinja::Fail>(body) { f => err.clear(); err.append(jinja::err); };
    return ts, ok;
}

//
// JSON
//

fn js(out: u8[>..]&, s: const u8[:]) { json_schema::json_string(out, s); }

// common_chat_msg::to_json_oaicompat (typed parts kept, not joined)
fn msg_json(out: u8[>..]&, m: Message) {
    out.append("{\"role\":");
    js(out, m.role);
    out.append(",\"content\":");
    if m.content.len > 0 {
        js(out, m.content);
    } else if m.parts.len > 0 {
        var media = false;
        for p in m.parts { if p.kind == "media_marker" { media = true; } }
        if media {
            var text: u8[>..] = [];
            var last_media = false;
            for p in m.parts {
                var nl = true;
                if p.kind == "text" { nl = !last_media && text.len > 0; last_media = false; }
                else if p.kind == "media_marker" { nl = false; last_media = true; }
                else { continue; }
                if nl { text.push('\n'); }
                text.append(p.text);
            }
            js(out, text);
        } else {
            out.push('[');
            for p, k in m.parts {
                if k > 0 { out.push(','); }
                out.append("{\"type\":");
                js(out, p.kind);
                if p.kind == "tool_reference" { out.append(",\"name\":"); js(out, p.name); }
                else { out.append(",\"text\":"); js(out, p.text); }
                out.push('}');
            }
            out.push(']');
        }
    } else {
        out.append("\"\"");
    }
    if m.reasoning.len > 0 { out.append(",\"reasoning_content\":"); js(out, m.reasoning); }
    if m.tool_name.len > 0 { out.append(",\"name\":"); js(out, m.tool_name); }
    if m.tool_call_id.len > 0 { out.append(",\"tool_call_id\":"); js(out, m.tool_call_id); }
    if m.tool_calls.len > 0 {
        out.append(",\"tool_calls\":[");
        for tc, k in m.tool_calls {
            if k > 0 { out.push(','); }
            out.append("{\"type\":\"function\",\"function\":{\"name\":");
            js(out, tc.name);
            out.append(",\"arguments\":");
            js(out, tc.arguments);
            out.push('}');
            if tc.id.len > 0 { out.append(",\"id\":"); js(out, tc.id); }
            out.push('}');
        }
        out.push(']');
    }
    out.push('}');
}

// common_chat_tools_to_json_oaicompat ("" for none)
fn tools_json(out: u8[>..]&, tools: const Tool[:]) {
    guard tools.len > 0 else { return; }
    out.push('[');
    for t, k in tools {
        if k > 0 { out.push(','); }
        out.append("{\"type\":\"function\",\"function\":{\"name\":");
        js(out, t.name);
        out.append(",\"description\":");
        js(out, t.description);
        out.append(",\"parameters\":");
        // the parameters as they parse, written back compactly
        let v = json_schema::json_value(t.parameters);
        if v >= 0 { json_schema::json_dump(out, v); } else { out.append("null"); }
        if t.defer_loading { out.append(",\"defer_loading\":true"); }
        out.append("}}");
    }
    out.push(']');
}

fn trim_ws(s: const u8[:]) -> const u8[:] { autoparser::trim_whitespace(s) }

//
// templates_apply_jinja
//

// The messages' JSON with the workarounds the template needs (into out).
// It throws jinja::Fail.
fn fixed_messages(t: chat_render::Template, msgs_json: const u8[:], out: u8[>..]&) {
    jinja::reset();
    let arr = jinja::from_json(msgs_json, false);
    chat_render::normalize(t, arr, false);
    let src = chat_render::source(t);
    // developer is system, but for GPT-OSS
    if find(src, "<|channel|>") < 0 {
        for i in jinja::arr_len(arr) {
            let m = jinja::arr_at(arr, i);
            let r = jinja::obj_get_name(m, "role");
            if r >= 0 && jinja::kind(r) == jinja::K_STRING && jinja::sbytes(r) == "developer" {
                jinja::obj_set_name(m, "role", jinja::mk_str("system", false));
            }
        }
    }
    if !t.caps.supports_system_role { system_not_supported(arr); }
    if t.caps.supports_tool_calls {
        for i in jinja::arr_len(arr) {
            let m = jinja::arr_at(arr, i);
            if jinja::obj_get_name(m, "tool_calls") >= 0 && jinja::obj_get_name(m, "content") < 0 {
                jinja::obj_set_name(m, "content", jinja::mk_str("", false));
            }
        }
    }
    if t.caps.supports_object_arguments {
        for i in jinja::arr_len(arr) {
            let m = jinja::arr_at(arr, i);
            let tcs = jinja::obj_get_name(m, "tool_calls");
            guard tcs >= 0 && jinja::kind(tcs) == jinja::K_ARRAY else { continue; }
            for k in jinja::arr_len(tcs) {
                let tc = jinja::arr_at(tcs, k);
                let f = jinja::obj_get_name(tc, "function");
                guard f >= 0 && jinja::kind(f) == jinja::K_OBJECT else { continue; }
                let a = jinja::obj_get_name(f, "arguments");
                guard a >= 0 && jinja::kind(a) == jinja::K_STRING else { continue; }
                let text = str(jinja::sbytes(a));
                jinja::obj_set_name(f, "arguments", jinja::from_json(text, false));
            }
        }
    }
    chat_render::dump(out, arr);
}

// The local time as strftime writes it.
fn local_time(fmt: const u8[:]) -> u8[>..] {
    var buf: u8[100] = [0; 100];
    let n = number::c_strftime(buf[..], fmt, number::now());
    return str(buf[..n]);
}

// The two patches llama.cpp applies to template sources that would throw.
fn patch_source(src: const u8[:], out: u8[>..]&) {
    out.append(src);
    if find(out, "<|channel|>") >= 0 && find(out, "in message.content or") >= 0 {
        replace_all(out, "{%- if \"<|channel|>analysis<|message|>\" in message.content or \"<|channel|>final<|message|>\" in message.content %}", "{%- if false %}");
    }
    if find(out, "[TOOL_CALLS]") >= 0 && find(out, "if (message['content'] is none or") >= 0 {
        replace_all(out, "{%- if (message['content'] is none or message['content'] == '' or message['content']|length == 0) and (message['tool_calls'] is not defined or message['tool_calls'] is none or message['tool_calls']|length == 0) %}", "{%- if false %}");
    }
}

fn replace_all(s: u8[>..]&, from: const u8[:], to: const u8[:]) {
    var out: u8[>..] = [];
    var pos = 0;
    loop {
        let p = find(s[pos..], from);
        if p < 0 { break; }
        out.append(s[pos..pos + p]);
        out.append(to);
        pos += p + from.len;
    }
    out.append(s[pos..]);
    s.clear();
    s.append(out);
}

// system_message_not_supported: the first message, a system one, joined to
// the next (or dropped when it is the only one).
fn system_not_supported(arr: i64) {
    let n = jinja::arr_len(arr);
    if n == 0 { return; }
    let first = jinja::arr_at(arr, 0);
    let role = jinja::obj_get_name(first, "role");
    if jinja::sbytes(role) != "system" { return; }
    if n > 1 {
        let second = jinja::arr_at(arr, 1);
        let a = jinja::obj_get_name(first, "content");
        let b = jinja::obj_get_name(second, "content");
        if jinja::kind(a) != jinja::K_STRING || jinja::kind(b) != jinja::K_STRING {
            jinja::fail_plain("[json.exception.type_error.302] type must be string, but is array");
        }
        var joined: u8[>..] = [];
        joined.append(jinja::sbytes(a));
        joined.push(10);
        joined.append(jinja::sbytes(b));
        jinja::obj_set_name(second, "content", jinja::mk_str(joined, false));
    }
    let base = jinja::vals[arr].a;
    for k in n - 1 { jinja::items[base + k] = jinja::items[base + k + 1]; }
    jinja::vals[arr].n = n - 1;
}

var messages_text: u8[>..] = [];
var tools_text: u8[>..] = [];
var context_text: u8[>..] = [];
var continue_reasoning: u8[>..] = [];
var continue_content: u8[>..] = [];

// The inputs of the apply, in the module's buffers (apply copies them, as
// the parsers it resets cannot outlive them otherwise).
var cur_ts = Templates { .. };
var raw_messages: u8[>..] = [];
var schema_text: u8[>..] = [];
var grammar_text: u8[>..] = [];
var cur_tool_choice = 0;
var cur_parallel = false;
var cur_reasoning_format = 0;
var cur_enable_thinking = true;
var cur_add_gen = true;
var cur_cont = 0;
var cur_has_cont_msg = false;

// common_chat_templates_apply with jinja: the prompt in chat_render's
// prompt and prompt_parts, the rest in chat_render's params; false, and
// the message in err, if it fails. It resets the PEG parsers and the
// JSON schemas.
fn apply(ts: Templates, inp: Inputs) -> bool {
    cur_ts = ts;
    // made locally, while inp is in use
    var l_tools: u8[>..] = [];
    tools_json(l_tools, inp.tools);
    let t = if l_tools.len > 0 && ts.has_tool_use { ts.tool_use } else { ts.main };
    let src = chat_render::source(t);
    // StepFun: trimmed, or the model loops reasoning
    let trim_all = find(src, "You have access to the following functions in JSONSchema format") >= 0;
    var l_msgs: u8[>..] = [];
    l_msgs.push('[');
    var n_render = inp.messages.len;
    var cont = inp.continue_final_message;
    var add_gen = inp.add_generation_prompt;
    var l_cr: u8[>..] = [];
    var l_cc: u8[>..] = [];
    var has_cont_msg = false;
    if cont != chat_render::CONTINUE_NONE {
        add_gen = false;
        if inp.messages.len > 0 {
            let last = inp.messages[inp.messages.len - 1];
            n_render--;
            has_cont_msg = true;
            l_cr.append(last.reasoning);
            render_content(l_cc, last);
            if cont == chat_render::CONTINUE_AUTO {
                cont = chat_render::CONTINUE_CONTENT;
                if last.reasoning.len > 0 && last.content.len == 0 && last.parts.len == 0 { cont = chat_render::CONTINUE_REASONING; }
            }
        }
    }
    for k in n_render {
        if k > 0 { l_msgs.push(','); }
        var m = inp.messages[k];
        if trim_all {
            m.content = trim_ws(m.content);
            m.reasoning = trim_ws(m.reasoning);
        }
        msg_json(l_msgs, m);
    }
    l_msgs.push(']');
    // the extra context: the dates, and the template's kwargs
    var l_ctx: u8[>..] = [];
    l_ctx.append("{\"datetime\":");
    js(l_ctx, local_time("%b %d %Y"));
    l_ctx.append(",\"date_string\":");
    js(l_ctx, local_time("%d %b %Y"));
    for kv in inp.kwargs {
        l_ctx.push(',');
        js(l_ctx, kv.key);
        l_ctx.push(':');
        l_ctx.append(kv.value);
    }
    l_ctx.push('}');
    var l_schema: u8[>..] = [];
    l_schema.append(inp.json_schema);
    var l_grammar: u8[>..] = [];
    l_grammar.append(inp.grammar);
    cur_tool_choice = inp.tool_choice;
    cur_parallel = inp.parallel_tool_calls;
    cur_reasoning_format = inp.reasoning_format;
    cur_enable_thinking = inp.enable_thinking;
    cur_cont = cont;
    cur_add_gen = add_gen;
    cur_has_cont_msg = has_cont_msg;
    // inp no longer in use: into the module's buffers
    tools_text.clear();
    tools_text.append(l_tools);
    raw_messages.clear();
    raw_messages.append(l_msgs);
    continue_reasoning.clear();
    continue_reasoning.append(l_cr);
    continue_content.clear();
    continue_content.append(l_cc);
    context_text.clear();
    context_text.append(l_ctx);
    schema_text.clear();
    schema_text.append(l_schema);
    grammar_text.clear();
    grammar_text.append(l_grammar);
    return apply_current();
}

fn apply_current() -> bool {
    peg::reset();
    json_schema::reset();
    chat_render::clear_params();
    let t = if tools_text.len > 0 && cur_ts.has_tool_use { cur_ts.tool_use } else { cur_ts.main };
    var ok = false;
    fn fix() {
        messages_text.clear();
        fixed_messages(t, raw_messages, messages_text);
        ok = true;
    }
    exception::try<jinja::Fail>(fix) { f => err.clear(); err.append(jinja::err); };
    if !ok { return false; }
    if tools_text.len > 0 && cur_tool_choice != chat_render::TOOL_CHOICE_NONE && grammar_text.len > 0 {
        err.clear();
        err.append("Cannot specify grammar with tools");
        return false;
    }
    let g = chat_render::Gen {
        messages: messages_text,
        tools: tools_text,
        tool_choice: cur_tool_choice,
        json_schema: schema_text,
        parallel_tool_calls: cur_parallel,
        reasoning_format: cur_reasoning_format,
        grammar: grammar_text,
        add_generation_prompt: cur_add_gen,
        continue_final_message: cur_cont,
        has_continue_msg: cur_has_cont_msg,
        continue_reasoning: continue_reasoning,
        continue_content: continue_content,
        enable_thinking: cur_enable_thinking,
        extra_context: context_text,
        add_bos: cur_ts.add_bos,
        add_eos: cur_ts.add_eos,
        mark_input: true,
    };
    if t.handler >= 0 {
        if !chat_parsers::try_init(t, g) {
            err.clear();
            err.append(chat_parsers::err);
            return false;
        }
        return true;
    }
    if !analyzed(t) {
        err.clear();
        err.append("Unable to generate parser for this template. Automatic parser generation failed: ");
        err.append(analysis_error);
        return false;
    }
    if !autoparser::try_generate(t, g) {
        err.clear();
        err.append("Unable to generate parser for this template. Automatic parser generation failed: ");
        err.append(autoparser::err);
        return false;
    }
    return true;
}

// The autoparser's analysis of the template applied (made once a template).
var analyzed_src = -1;
var analysis_ok = false;
var analysis_error: u8[>..] = [];

fn analyzed(t: chat_render::Template) -> bool {
    if analyzed_src != t.prog.src {
        analyzed_src = t.prog.src;
        analysis_ok = autoparser::analyze_template(t);
        analysis_error.clear();
        if !analysis_ok { analysis_error.append(autoparser::err); }
    }
    return analysis_ok;
}

// common_chat_msg::render_content: the content, or the text parts joined.
fn render_content(out: u8[>..]&, m: Message) {
    if m.content.len > 0 { out.append(m.content); return; }
    var first = true;
    for p in m.parts {
        guard p.kind == "text" else { continue; }
        if !first { out.append("\n\n"); }
        first = false;
        out.append(p.text);
    }
}

//
// the output parsed (common_chat_peg_parse)
//

var content_only = -1;
var input_text: u8[>..] = [];

// The message the output of the last apply makes, in chat_peg's content_text,
// reasoning_text and calls; false, and the message in err, if the output
// does not match the format.
fn parse(output: const u8[:], is_partial: bool) -> bool {
    var a = chat_render::parser;
    if a < 0 {
        if content_only < 0 || content_only >= peg::arenas.len {
            peg::begin();
            let e = peg::end();
            let c = chat_peg::content(peg::rest());
            content_only = peg::build(peg::seq2(c, e));
        }
        a = content_only;
    }
    input_text.clear();
    input_text.append(chat_render::gen_prompt);
    input_text.append(output);
    var flags = peg::FLAG_LENIENT;
    if !is_partial { flags |= peg::FLAG_COMPLETE; }
    let r, ok = peg::parse(a, input_text, flags, 0);
    if !ok { err.clear(); err.append(peg::err); return false; }
    if r.kind == peg::FAIL {
        if is_partial && r.end > 0 {
            map_result();
            return true;
        }
        err.clear();
        err.append(str("The model produced output that does not match the expected ", format_name(chat_render::format), " format"));
        return false;
    }
    map_result();
    return true;
}

fn map_result() {
    if chat_render::format == chat_render::FORMAT_PEG_GEMMA4 { chat_parsers::map_gemma4(); }
    else if chat_render::format == chat_render::FORMAT_PEG_MINIMAX_M3 { chat_parsers::map_minimax_m3(); }
    else { chat_peg::map_parse(); }
}

fn format_name(f: i64) -> const u8[:] {
    return match f { 0 => "Content-only", 1 => "peg-simple", 2 => "peg-native", 3 => "peg-gemma4", 4 => "peg-minimax-m3", _ => "unknown" };
}
