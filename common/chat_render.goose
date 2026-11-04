// A chat template rendered the way llama.cpp's chat code renders it: the
// conversation and tools as JSON, turned into the template's variables
// (strings marked as the user's input), and the output kept in parts, input
// and not. The JSON is kept as text and read into Jinja values for each
// render, since a render starts from fresh values.
// port-of: common/chat.cpp (common_chat_template, direct_apply_impl, generation_prompt_impl, messages_inp_normalizer)
import std;
import exception;
import .jinja;
import .number;
import .jinja_lexer;
import .jinja_parser;
import .jinja_runtime;
import .jinja_caps;

namespace chat_render;

// A parsed template (common_chat_template, without the parser it gets in
// chat): its program, capabilities, and BOS and EOS texts.
struct Template {
    prog: jinja::Program = jinja::Program { .. },
    caps: jinja::Caps = jinja::Caps { .. },
    bos: const u8[:] = "",
    eos: const u8[:] = "",
    // chat's: the dedicated handler, or the autoparser analysis
    handler: i64 = -1,
    analysis: i64 = -1,
}

fn source(t: Template) -> const u8[:] { jinja::asrc[t.prog.src..t.prog.src + t.prog.src_len] }

// The texts templates keep.
var keep: u8[>..] = [];

fn kept(s: const u8[:]) -> const u8[:] {
    let at0 = keep.len;
    keep.append(s);
    return keep[at0..keep.len];
}

// A template from its source; it throws jinja::Fail if it does not parse.
fn make_template(src: const u8[:], bos: const u8[:], eos: const u8[:]) -> Template {
    let off, len = jinja::tokenize(src);
    let prog = jinja::parse(off, len);
    let caps = jinja::caps_get(prog);
    return Template { prog: prog, caps: caps, bos: kept(bos), eos: kept(eos), .. };
}

let TOOL_CHOICE_AUTO = 0;
let TOOL_CHOICE_REQUIRED = 1;
let TOOL_CHOICE_NONE = 2;

let REASONING_NONE = 0;
let REASONING_AUTO = 1;
let REASONING_DEEPSEEK_LEGACY = 2;
let REASONING_DEEPSEEK = 3;

let CONTINUE_NONE = 0;
let CONTINUE_AUTO = 1;
let CONTINUE_REASONING = 2;
let CONTINUE_CONTENT = 3;

// autoparser::generation_params, its JSON as text ("" for none).
struct Gen {
    messages: const u8[:] = "[]",
    tools: const u8[:] = "",
    tool_choice: i64 = TOOL_CHOICE_AUTO,
    json_schema: const u8[:] = "",
    parallel_tool_calls: bool = true,
    reasoning_format: i64 = REASONING_AUTO,
    grammar: const u8[:] = "",
    add_generation_prompt: bool = false,
    continue_final_message: i64 = CONTINUE_NONE,
    // the message continued: whether there is one, its reasoning and content
    has_continue_msg: bool = false,
    continue_reasoning: const u8[:] = "",
    continue_content: const u8[:] = "",
    enable_thinking: bool = true,
    extra_context: const u8[:] = "",
    add_bos: bool = false,
    add_eos: bool = false,
    mark_input: bool = true,
}

fn has_continuation(g: Gen) -> bool { g.continue_final_message != CONTINUE_NONE && g.has_continue_msg }

// direct_apply_impl's overrides of the messages and tools, and more context.
struct Override {
    has_messages: bool = false,
    messages: const u8[:] = "",
    has_tools: bool = false,
    tools: const u8[:] = "",
    context: const u8[:] = "",
}

// The rendered prompt: its text, and its parts as (end offset, is_input).
struct PromptPart { end: i64, input: bool }
var prompt: u8[>..<] = [];
var prompt_parts: PromptPart[>..<] = [];

// messages_inp_normalizer: string or typed content, as the template takes it.
fn normalize(t: Template, messages: i64, mark: bool) {
    let only_string = t.caps.supports_string_content && !t.caps.supports_typed_content;
    let only_typed = !t.caps.supports_string_content && t.caps.supports_typed_content;
    guard (only_string || only_typed) && jinja::kind(messages) == jinja::K_ARRAY else { return; }
    for i in jinja::arr_len(messages) {
        let m = jinja::arr_at(messages, i);
        guard jinja::kind(m) == jinja::K_OBJECT else { continue; }
        let it = jinja::obj_get_name(m, "content");
        guard it >= 0 else { continue; }
        if only_typed && jinja::kind(it) == jinja::K_STRING {
            let part = jinja::mk_object();
            jinja::obj_set_name(part, "type", jinja::mk_str("text", mark));
            jinja::obj_set_name(part, "text", it);
            let parts = jinja::mk_array();
            jinja::arr_push(parts, part);
            jinja::obj_set_name(m, "content", parts);
        } else if only_string && jinja::kind(it) == jinja::K_ARRAY {
            jinja::obj_set_name(m, "content", jinja::mk_str(concat_content_parts(it), mark));
        }
    }
}

// The text parts joined with newlines, none next to a media marker.
fn concat_content_parts(parts: i64) -> u8[>..] {
    var text: u8[>..] = [];
    var last_was_media = false;
    for i in jinja::arr_len(parts) {
        let part = jinja::arr_at(parts, i);
        let ty = jinja::obj_get_name(part, "type");
        let type_name = if ty >= 0 && jinja::kind(ty) == jinja::K_STRING { jinja::sbytes(ty) } else { "" };
        var add_new_line = true;
        if type_name == "text" {
            add_new_line = !last_was_media && text.len > 0;
            last_was_media = false;
        } else if type_name == "media_marker" {
            add_new_line = false;
            last_was_media = true;
        } else {
            continue;
        }
        if add_new_line { text.push('\n'); }
        let tx = jinja::obj_get_name(part, "text");
        if tx >= 0 && jinja::kind(tx) == jinja::K_STRING { text.append(jinja::sbytes(tx)); }
    }
    return text;
}

// The keys of a JSON object's text into o, replacing those it has.
fn merge_into(o: i64, json_text: const u8[:], mark: bool) {
    guard json_text.len > 0 else { return; }
    let v = jinja::from_json(json_text, mark);
    guard jinja::kind(v) == jinja::K_OBJECT else { return; }
    for k in jinja::obj_len(v) {
        jinja::obj_set_name(o, jinja::sbytes(jinja::obj_key(v, k)), jinja::obj_value(v, k));
    }
}

// common_chat_template_direct_apply_impl, into prompt and prompt_parts. It
// throws jinja::Fail if the template, or the JSON, fails.
fn direct_apply(t: Template, g: Gen, ov: Override) {
    jinja::reset();
    let c = jinja::new_ctx();
    let mark = g.mark_input;
    // the variables, in llama.cpp's order: later ones replace earlier ones
    let inp = jinja::mk_object();
    var messages = -1;
    if ov.has_messages {
        messages = jinja::from_json(ov.messages, mark);
    } else {
        messages = jinja::from_json(g.messages, mark);
        normalize(t, messages, mark);
    }
    jinja::obj_set_name(inp, "messages", messages);
    jinja::obj_set_name(inp, "bos_token", jinja::mk_str(t.bos, mark));
    jinja::obj_set_name(inp, "eos_token", jinja::mk_str(t.eos, mark));
    jinja::obj_set_name(inp, "enable_thinking", jinja::mk_bool(g.enable_thinking));
    if ov.has_tools || (g.tools.len > 0 && !json_empty(g.tools)) {
        jinja::obj_set_name(inp, "tools", jinja::from_json(if ov.has_tools { ov.tools } else { g.tools }, mark));
    }
    merge_into(inp, g.extra_context, mark);
    merge_into(inp, ov.context, mark);
    if g.add_generation_prompt { jinja::obj_set_name(inp, "add_generation_prompt", jinja::mk_bool(true)); }
    let pr = jinja::obj_get_name(inp, "preserve_reasoning");
    if pr >= 0 && jinja::kind(pr) == jinja::K_BOOL { jinja::caps_apply_preserve_reasoning(c, jinja::vals[pr].i != 0); }
    let re = jinja::obj_get_name(inp, "reasoning_effort");
    if re >= 0 && jinja::kind(re) == jinja::K_STRING && jinja::sbytes(re).len > 0 {
        jinja::caps_apply_reasoning_effort(c, jinja::sbytes(re));
    }
    for k in jinja::obj_len(inp) {
        jinja::set_var(c, jinja::sbytes(jinja::obj_key(inp, k)), jinja::obj_value(inp, k));
    }
    // bos and eos are the model's, not the user's
    jinja::set_var(c, "bos_token", jinja::mk_str(t.bos, false));
    jinja::set_var(c, "eos_token", jinja::mk_str(t.eos, false));
    let r = jinja::render(t.prog, c);

    prompt.clear();
    prompt_parts.clear();
    let v = jinja::vals[r];
    for k in v.n {
        let p = jinja::parts[v.a + k];
        var text = jinja::part_bytes(p);
        // the vocabulary adds them itself
        if k == 0 && g.add_bos && !p.input && starts_with(text, t.bos) { text = text[t.bos.len..]; }
        if k == v.n - 1 && g.add_eos && !p.input && ends_with(text, t.eos) { text = text[..text.len - t.eos.len]; }
        prompt.append(text);
        prompt_parts.push(PromptPart { end: prompt.len, input: p.input });
    }
}

// Whether the JSON text is null, or an empty array or object (json::empty).
fn json_empty(s: const u8[:]) -> bool {
    var i = 0;
    while i < s.len && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r') { i++; }
    if i >= s.len { return true; }
    if starts_with(s[i..], "null") { return true; }
    if s[i] != '[' && s[i] != '{' { return false; }
    var j = i + 1;
    while j < s.len && (s[j] == ' ' || s[j] == '\n' || s[j] == '\t' || s[j] == '\r') { j++; }
    return j < s.len && (s[j] == ']' || s[j] == '}');
}

// The prompt's parts appended, in the shape common_chat_prompt::append keeps
// them: a part next to one of the same kind joins it.
fn append_part(text: const u8[:], input: bool) {
    guard text.len > 0 else { return; }
    prompt.append(text);
    if prompt_parts.len > 0 && prompt_parts[prompt_parts.len - 1].input == input {
        prompt_parts[prompt_parts.len - 1].end = prompt.len;
    } else {
        prompt_parts.push(PromptPart { end: prompt.len, input: input });
    }
}

// What the generation prompt adds: the render with it, past the common
// prefix with the render without it (generation_prompt_impl), into out.
// It throws jinja::Fail.
fn generation_prompt(t: Template, g: Gen, ov: Override, out) {
    var params = g;
    params.add_generation_prompt = false;
    params.continue_final_message = CONTINUE_NONE;
    direct_apply(t, params, ov);
    var without: u8[>..] = [];
    without.append(prompt);
    params.add_generation_prompt = true;
    direct_apply(t, params, ov);
    var n = 0;
    while n < without.len && n < prompt.len && without[n] == prompt[n] { n++; }
    out.append(prompt[n..]);
}

// A JSON value as compact JSON text (common_json::dump).
recursive fn dump(out: u8[>..]&, v: i64) {
    let x = jinja::vals[v];
    let k = x.kind;
    if k == jinja::K_BOOL { out.append(if x.i != 0 { "true" } else { "false" }); return; }
    if k == jinja::K_INT { format(out, x.i); return; }
    if k == jinja::K_FLOAT { number::float_json(out, x.f); return; }
    if k == jinja::K_STRING { jinja::json_string(out, jinja::sbytes(v)); return; }
    if k == jinja::K_ARRAY {
        out.push('[');
        for i in x.n {
            if i > 0 { out.push(','); }
            dump(out, jinja::arr_at(v, i));
        }
        out.push(']');
        return;
    }
    if k == jinja::K_OBJECT {
        out.push('{');
        for i in jinja::obj_len(v) {
            if i > 0 { out.push(','); }
            jinja::json_string(out, jinja::sbytes(jinja::obj_key(v, i)));
            out.push(':');
            dump(out, jinja::obj_value(v, i));
        }
        out.push('}');
        return;
    }
    out.append("null");
}

//
// what an apply makes (common_chat_params), besides the prompt
//

let FORMAT_CONTENT_ONLY = 0;
let FORMAT_PEG_SIMPLE = 1;
let FORMAT_PEG_NATIVE = 2;
let FORMAT_PEG_GEMMA4 = 3;
let FORMAT_PEG_MINIMAX_M3 = 4;

let TRIGGER_WORD = 0;
let TRIGGER_PATTERN = 1;
let TRIGGER_PATTERN_FULL = 2;
let TRIGGER_TOKEN = 3;

var format = FORMAT_CONTENT_ONLY;
var grammar: u8[>..] = [];
var grammar_lazy = false;
var gen_prompt: u8[>..] = [];
var supports_thinking = false;
var thinking_start: u8[>..] = [];
// lists: each item followed by a 0 byte
var thinking_ends: u8[>..] = [];
var preserved_tokens: u8[>..] = [];
var additional_stops: u8[>..] = [];
// triggers: their types, and their values as a list
var trigger_types: i64[>..] = [];
var trigger_values: u8[>..] = [];
// the parser of the output: a peg arena, -1 for content only
var parser = -1;
// the message delimiters: the assistant's and the user's start
var assistant_start: u8[>..] = [];
var user_start: u8[>..] = [];

fn clear_params() {
    format = FORMAT_CONTENT_ONLY;
    grammar.clear();
    grammar_lazy = false;
    gen_prompt.clear();
    supports_thinking = false;
    thinking_start.clear();
    thinking_ends.clear();
    preserved_tokens.clear();
    additional_stops.clear();
    trigger_types.clear();
    trigger_values.clear();
    parser = -1;
    assistant_start.clear();
    user_start.clear();
}

fn list_push(l: u8[>..]&, s: const u8[:]) { l.append(s); l.push(0); }

fn add_trigger(ty: i64, value: const u8[:]) { trigger_types.push(ty); list_push(trigger_values, value); }
