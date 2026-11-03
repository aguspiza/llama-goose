// llama.cpp's common chat: a model's Jinja chat template applied to a
// conversation, the way common_chat_templates_apply does it with jinja on
// (the prompt; not yet the tool call and reasoning parsers it builds).
// port-of: common/chat.cpp (templates_init, templates_apply_jinja, direct_apply, format_single), common/chat.h
import std;
import exception;
import .jinja;
import .jinja_lexer;
import .jinja_parser;
import .jinja_runtime;
import .jinja_caps;

namespace common_chat;

// A message: its role and content, and the reasoning before it, if any.
struct Msg { role: const u8[:], content: const u8[:], reasoning: const u8[:] = "" }

// What llama.cpp's specialized handlers do to the messages before the
// template sees them (common_chat_template_detect_params_init); the rest
// render them as they are.
let HANDLER_NONE = 0;
let HANDLER_LFM2 = 1;       // LFM2 and LFM2.5: the reasoning is also the field thinking
let HANDLER_OTHER = 2;      // another specialized one, of those not ported

struct Template {
    prog: jinja::Program = jinja::Program { .. },
    caps: jinja::Caps = jinja::Caps { .. },
    handler: i64 = HANDLER_NONE,
}

// common_chat_templates: the default template and the one for tools, the
// BOS and EOS texts, and whether the vocabulary adds them itself.
struct Templates {
    main: Template = Template { .. },
    tool_use: Template = Template { .. },
    has_tool_use: bool = false,
    has_explicit_template: bool = false,
    add_bos: bool = false,
    add_eos: bool = false,
    bos: const u8[:] = "",
    eos: const u8[:] = "",
}

// The texts the templates keep: their BOS and EOS.
var keep: u8[>..] = [];

let CHATML = "{%- for message in messages -%}\n  {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n  {{- '<|im_start|>assistant\n' -}}\n{%- endif -%}";

// The handler llama.cpp picks for a template, as far as the prompt goes.
fn detect_handler(src: const u8[:]) -> i64 {
    fn has(s: const u8[:]) -> bool { find(src, s) >= 0 }
    if has("[SYSTEM_PROMPT]") && has("[TOOL_CALLS]") && has("[ARGS]") && !has("[CALL_ID]") { return HANDLER_OTHER; }  // Ministral 3
    if has("chat_format=llm-jp-harmony-v1") { return HANDLER_OTHER; }
    if has("<|channel|>") { return HANDLER_OTHER; }                                             // GPT-OSS, Harmony XML
    if has("<|ifm|im_start|>") && has("<ifm|tool_calls>") { return HANDLER_OTHER; }              // K2 Horizon
    if has("<|tool_list_start|>") && has("<|tool_list_end|>") { return HANDLER_LFM2; }
    if has("List of tools: [") && !has("<|tool_list_start|>") { return HANDLER_LFM2; }          // LFM2.5
    if (has("dsml_token") && has("DSML") && (has("function_calls") || has("tool_calls"))) || has("｜DSML｜ calls>") {
        return HANDLER_OTHER;                                                                     // DeepSeek V3.2/V4
    }
    if has("'<|tool_call>call:'") && !has("{#- OpenAI Chat Completions:") { return HANDLER_OTHER; }   // old Gemma 4
    if has("[source_lang_code]") && has("[target_lang_code]") { return HANDLER_OTHER; }          // TranslateGemma
    return HANDLER_NONE;
}

// A template parsed and probed (common_chat_template's constructor).
fn make_template(src: const u8[:]) -> Template {
    let off, len = jinja::tokenize(src);
    let prog = jinja::parse(off, len);
    let caps = jinja::caps_get(prog);
    return Template { prog: prog, caps: caps, handler: detect_handler(jinja::asrc[off..off + len]) };
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

// common_chat_templates_init: the template src (empty or "chatml" for
// ChatML, or the tool-use one when there is one) and the tool-use template
// (empty if none), with the BOS and EOS texts and whether the vocabulary
// adds them; false, and the message in jinja::err, if a template does not
// parse.
fn init(src: const u8[:], tool_use_src: const u8[:], bos: const u8[:], eos: const u8[:], add_bos: bool, add_eos: bool) -> Templates, bool {
    var t = Templates { .. };
    var ok = false;
    fn body() {
        t = make_templates(src, tool_use_src, bos, eos, add_bos, add_eos);
        ok = true;
    }
    exception::try<jinja::Fail>(body);
    return t, ok;
}

fn make_templates(src: const u8[:], tool_use_src: const u8[:], bos: const u8[:], eos: const u8[:], add_bos: bool, add_eos: bool) -> Templates {
    let explicit = src.len > 0 || tool_use_src.len > 0;
    var main_src: u8[>..] = [];
    if src.len == 0 || src == "chatml" {
        patch_source(if tool_use_src.len > 0 { tool_use_src } else { CHATML }, main_src);
    } else {
        patch_source(src, main_src);
    }
    let b0 = keep.len;
    keep.append(bos);
    let e0 = keep.len;
    keep.append(eos);
    let main = make_template(main_src);
    var t = Templates { main: main, tool_use: main, has_explicit_template: explicit, add_bos: add_bos, add_eos: add_eos,
                        bos: keep[b0..b0 + bos.len], eos: keep[e0..e0 + eos.len] };
    if tool_use_src.len > 0 {
        var tsrc: u8[>..] = [];
        tsrc.append(tool_use_src);
        t.tool_use = make_template(tsrc);
        t.has_tool_use = true;
    }
    return t;
}

// What common_chat_templates_apply takes, as far as it is ported.
struct Inputs {
    add_generation_prompt: bool = true,
    enable_thinking: bool = true,
}

// The prompt: its text, and where it came from the user's input (the parts
// whose special tokens must not be parsed), as (end offset, is_input).
struct PromptPart { end: i64, input: bool }
var prompt: u8[>..] = [];
var prompt_parts: PromptPart[>..] = [];

// "%b %d %Y" and the like, of the local time now.
fn local_time(fmt: const u8[:]) -> u8[>..] {
    var buf: u8[100] = [0; 100];
    let n = jinja::c_strftime(buf[..], fmt, jinja::c_time());
    return str(buf[..n]);
}

// The conversation as the template's `messages`, every string marked as
// input (render_message_to_json and the workarounds before rendering).
fn messages_value(t: Template, msgs: const Msg[:]) -> i64 {
    let src = jinja::asrc[t.prog.src..t.prog.src + t.prog.src_len];
    let only_typed = !t.caps.supports_string_content && t.caps.supports_typed_content;
    // StepFun: trimmed, or the model loops reasoning
    let trim_all = find(src, "You have access to the following functions in JSONSchema format") >= 0;
    let gpt_oss = find(src, "<|channel|>") >= 0;
    let arr = jinja::mk_array();
    for m in msgs {
        let o = jinja::mk_object();
        let role = if !gpt_oss && m.role == "developer" { "system" } else { m.role };
        jinja::obj_set_name(o, "role", jinja::mk_str(role, true));
        let content = if trim_all { trim(m.content) } else { m.content };
        let reasoning = if trim_all { trim(m.reasoning) } else { m.reasoning };
        if only_typed {
            let part = jinja::mk_object();
            jinja::obj_set_name(part, "type", jinja::mk_str("text", true));
            jinja::obj_set_name(part, "text", jinja::mk_str(content, true));
            let parts = jinja::mk_array();
            jinja::arr_push(parts, part);
            jinja::obj_set_name(o, "content", parts);
        } else {
            jinja::obj_set_name(o, "content", jinja::mk_str(content, true));
        }
        if reasoning.len > 0 {
            jinja::obj_set_name(o, "reasoning_content", jinja::mk_str(reasoning, true));
            if t.handler == HANDLER_LFM2 { jinja::obj_set_name(o, "thinking", jinja::mk_str(reasoning, true)); }
        }
        jinja::arr_push(arr, o);
    }
    if !t.caps.supports_system_role { system_not_supported(arr); }
    return arr;
}

// The first message, a system one, joined to the next (or dropped when it
// is the only one), for a template without a system role.
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
        joined.push('\n');
        joined.append(jinja::sbytes(b));
        jinja::obj_set_name(second, "content", jinja::mk_str(joined, true));
    }
    let base = jinja::vals[arr].a;
    for k in n - 1 { jinja::items[base + k] = jinja::items[base + k + 1]; }
    jinja::vals[arr].n = n - 1;
}

// Renders the conversation (common_chat_templates_apply with jinja) into
// prompt and prompt_parts; false, and the message in jinja::err, if the
// template fails.
fn apply(ts: Templates&, msgs: const Msg[:], inp: Inputs) -> bool {
    var ok = false;
    fn body() {
        render(ts, msgs, inp);
        ok = true;
    }
    exception::try<jinja::Fail>(body);
    return ok;
}

fn render(ts: Templates&, msgs: const Msg[:], inp: Inputs) {
    let t = ts.main;
    jinja::reset();
    let c = jinja::new_ctx();
    fn put(name: const u8[:], v: i64) { jinja::set_var(c, name, v); }
    put("messages", messages_value(t, msgs));
    put("bos_token", jinja::mk_str(ts.bos, true));
    put("eos_token", jinja::mk_str(ts.eos, true));
    put("enable_thinking", jinja::mk_bool(inp.enable_thinking));
    put("datetime", jinja::mk_str(local_time("%b %d %Y"), true));
    put("date_string", jinja::mk_str(local_time("%d %b %Y"), true));
    if inp.add_generation_prompt { put("add_generation_prompt", jinja::mk_bool(true)); }
    // bos and eos are the model's, not the user's
    put("bos_token", jinja::mk_str(ts.bos, false));
    put("eos_token", jinja::mk_str(ts.eos, false));
    let r = jinja::render(t.prog, c);

    prompt.clear();
    prompt_parts.clear();
    let v = jinja::vals[r];
    for k in v.n {
        let p = jinja::parts[v.a + k];
        var text = jinja::part_bytes(p);
        // the vocabulary adds them itself
        if k == 0 && ts.add_bos && !p.input && starts_with(text, ts.bos) { text = text[ts.bos.len..]; }
        if k == v.n - 1 && ts.add_eos && !p.input && ends_with(text, ts.eos) { text = text[..text.len - ts.eos.len]; }
        prompt.append(text);
        prompt_parts.push(PromptPart { end: prompt.len, input: p.input });
    }
}

// common_chat_format_single: what new_msg adds to the prompt of past, with
// the generation prompt when add_ass; false, and the message in jinja::err,
// if the template fails.
fn format_single(ts: Templates&, past: const Msg[:], new_msg: Msg, add_ass: bool, out: u8[>..]&) -> bool {
    var before: u8[>..] = [];
    if past.len > 0 {
        if !apply(ts, past, Inputs { add_generation_prompt: false, .. }) { return false; }
        before.append(prompt);
    }
    // a past prompt ending with a newline keeps it
    if add_ass && before.len > 0 && before[before.len - 1] == '\n' { out.push('\n'); }
    var all: Msg[>..] = [];
    all.append(past);
    all.push(new_msg);
    if !apply(ts, all, Inputs { add_generation_prompt: add_ass, .. }) { return false; }
    out.append(prompt[min(before.len, prompt.len)..]);
    return true;
}
