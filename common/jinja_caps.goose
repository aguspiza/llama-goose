// llama.cpp's Jinja engine: what a chat template supports, found out by
// rendering it with sample conversations and looking at what it read
// (string or typed content, a system role, tools, reasoning).
// caps_get resets the values: nothing made before it survives it.
// port-of: common/jinja/caps.cpp, common/jinja/caps.h
import std;
import exception;
import .jinja;
import .jinja_parser;
import .jinja_runtime;

namespace jinja;

struct Caps {
    supports_tools: bool = true,
    supports_tool_calls: bool = true,
    supports_system_role: bool = true,
    supports_parallel_tool_calls: bool = true,
    // reasoning kept in the whole history, not just the last assistant message
    supports_preserve_reasoning: bool = false,
    supports_reasoning_effort: bool = false,
    // the levels the template renders distinctly, lowest first (effort_name)
    reasoning_effort_levels: i64[..7] = [],
    // the levels the template raises on
    reasoning_effort_rejected: i64[..7] = [],
    // one of the two content kinds is supported
    supports_string_content: bool = true,
    supports_typed_content: bool = false,
    supports_object_arguments: bool = false,
}

// The reasoning effort levels caps_get tries, lowest first, and then the
// one no template knows.
let N_EFFORTS = 7;
fn effort_name(k: i64) -> const u8[:] {
    return match k { 0 => "none", 1 => "minimal", 2 => "low", 3 => "medium", 4 => "high", 5 => "xhigh", 6 => "max", _ => "REASONING_EFFORT_UNKNOWN" };
}

fn caps_apply_preserve_reasoning(c: Ctx, enabled: bool) {
    set_var(c, "preserve_thinking", mk_bool(enabled));
    set_var(c, "clear_thinking", mk_bool(!enabled));
    set_var(c, "truncate_history_thinking", mk_bool(!enabled));
    set_var(c, "drop_thinking", mk_bool(!enabled));
}

fn caps_apply_reasoning_effort(c: Ctx, effort: const u8[:]) {
    let v = mk_str(effort, false);   // both names the same value, for the stats
    set_var(c, "reasoning_effort", v);
    set_var(c, "reasoning_strength", v);
}

enum Setup {
    None,
    Preserve,  // thinking on, reasoning preserved
    Effort,    // thinking on, reasoning effort caps_effort
}

// A sample render's outcome: whether it worked, its output (in caps_out),
// the context, and the messages and tools as the template got them.
struct Probe { ok: bool, ctx: Ctx, messages: i64, tools: i64 }

var caps_out: u8[>..<] = [];
var caps_effort = 1;

// Renders p with the messages and tools (JSON) as caps_get does, with
// stats on; when that fails, once more with an empty reasoning_content on
// the assistant turns that lack one (templates such as K2 Horizon want one).
fn probe(p: Program, messages: const u8[:], tools: const u8[:], setup: Setup) -> Probe {
    var pr = Probe { ok: false, ctx: Ctx { env: -1, depth: 0 }, messages: -1, tools: -1 };
    var add_reasoning = false;
    for attempt in 2 {
        reset();
        let c = new_ctx();
        let msgs = from_json(messages, true);
        if add_reasoning { add_empty_reasoning(msgs); }
        set_var(c, "messages", msgs);
        set_var(c, "tools", from_json(if tools.len > 0 { tools } else { "[]" }, true));
        set_var(c, "bos_token", mk_str("", true));
        set_var(c, "eos_token", mk_str("", true));
        set_var(c, "add_generation_prompt", mk_bool(true));
        if setup == Setup.Preserve {
            set_var(c, "enable_thinking", mk_bool(true));
            caps_apply_preserve_reasoning(c, true);
        } else if setup == Setup.Effort {
            set_var(c, "enable_thinking", mk_bool(true));
            caps_apply_reasoning_effort(c, effort_name(caps_effort));
        }
        pr = Probe { ok: false, ctx: c, messages: get_var(c, "messages"), tools: get_var(c, "tools") };
        var ok = false;
        caps_out.clear();
        fn run() {
            stats = true;
            let r = render(p, c);
            caps_out.append(sbytes(r));
            ok = true;
        }
        exception::try<Fail>(run);
        stats = false;
        pr.ok = ok;
        if !ok { caps_out.clear(); }
        if ok || attempt == 1 { break; }
        add_reasoning = add_empty_reasoning(from_json(messages, true));
        if !add_reasoning { break; }
    }
    return pr;
}

// Gives the assistant turns of msgs that have none an empty
// reasoning_content: whether there were any.
fn add_empty_reasoning(msgs: i64) -> bool {
    var added = false;
    for i in arr_len(msgs) {
        let m = arr_at(msgs, i);
        if kind(m) != ValKind.Object { continue; }
        let role = obj_get_name(m, "role");
        if role < 0 || kind(role) != ValKind.String || sbytes(role) != "assistant" { continue; }
        if obj_find_name(m, "reasoning_content") >= 0 { continue; }
        obj_set_name(m, "reasoning_content", mk_str("", true));
        added = true;
    }
    return added;
}

// v[key] for an object, v[index] for an array, as caps_get's at() does it.
fn at_name(v: i64, key: const u8[:]) -> i64 {
    let r = if kind(v) == ValKind.Object { obj_get_name(v, key) } else { -1 };
    if r < 0 { fail(str("Key '", key, "' not found in value of type ", tname(v))); }
    return r;
}

fn at_index(v: i64, k: i64) -> i64 {
    if !is_array(v) { fail(str(tname(v), " is not an array value")); }
    let n = arr_len(v);
    let i = if k < 0 { k + n } else { k };
    if i < 0 || i >= n { fail(str("Index ", i, " out of bounds for array of size ", n)); }
    return arr_at(v, i);
}

fn has_op(v: i64, op: u8) -> bool { (vals[v].ops & op) != 0 }

let TOOLS_JSON = "[{\"name\": \"tool\", \"type\": \"function\", \"function\": {\"name\": \"tool1\", \"description\": \"Tool description\", \"parameters\": {\"type\": \"object\", \"properties\": {\"arg\": {\"type\": \"string\", \"description\": \"Arg description\"}}, \"required\": [\"arg\"]}}}]";

// A conversation with one tool call (two when parallel), its arguments as
// an object or as a JSON string.
fn tool_messages(out: u8[>..]&, object_args: bool, parallel: bool) {
    let args = if object_args { "{\"arg\": \"value\"}" } else { "\"{\\\"arg\\\": \\\"value\\\"}\"" };
    out.append("[{\"role\": \"user\", \"content\": \"User message\"}, {\"role\": \"assistant\", \"content\": \"\", \"tool_calls\": [");
    format(out, "{\"id\": \"call00001\", \"type\": \"function\", \"function\": {\"name\": \"tool1\", \"arguments\": ", args, "}}");
    if parallel {
        format(out, ", {\"id\": \"call00002\", \"type\": \"function\", \"function\": {\"name\": \"tool1\", \"arguments\": ", args, "}}");
    }
    out.append("]}, {\"role\": \"tool\", \"content\": \"Tool response\", \"tool_call_id\": \"call00001\"}, ");
    out.append("{\"role\": \"assistant\", \"content\": \"The tool response was 'tool response'\"}, {\"role\": \"user\", \"content\": \"User message\"}]");
}

let USER_ONLY = "[{\"role\": \"user\", \"content\": \"User message\"}]";

// What program p supports.
fn caps_get(p: Program) -> Caps {
    var r = Caps { .. };

    // typed content: the content read as an array
    var checks_for_string = false;
    var pr = probe(p, "[{\"role\": \"user\", \"content\": \"STRING_MARKER\"}]", "", Setup.None);
    var content = at_name(at_index(pr.messages, 0), "content");
    if has_op(content, OP_TEST_IS_STRING) { checks_for_string = true; }
    let used_as_array = has_op(content, OP_SELECTATTR) || has_op(content, OP_ARRAY_ACCESS);
    if used_as_array { r.supports_typed_content = true; }
    if !pr.ok {
        r.supports_string_content = false;
    } else if used_as_array && find(caps_out, "STRING_MARKER") < 0 {
        // read for a check, but not in the output
        r.supports_string_content = false;
    }
    if checks_for_string {
        pr = probe(p, "[{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"STRING_MARKER\"}]}]", "", Setup.None);
        content = at_name(at_index(pr.messages, 0), "content");
        if (has_op(content, OP_SELECTATTR) || has_op(content, OP_ARRAY_ACCESS)) && pr.ok { r.supports_typed_content = true; }
    }

    // a system prompt
    pr = probe(p, "[{\"role\": \"system\", \"content\": \"System message\"}, {\"role\": \"user\", \"content\": \"User message\"}]", "", Setup.None);
    content = at_name(at_index(pr.messages, 0), "content");
    if !vals[content].used { r.supports_system_role = false; }

    // one tool call, with object arguments
    var msgs: u8[>..] = [];
    tool_messages(msgs, true, false);
    pr = probe(p, msgs, TOOLS_JSON, Setup.None);
    if pr.ok {
        let tool_name = at_name(at_name(at_index(pr.tools, 0), "function"), "name");
        if !vals[tool_name].used { r.supports_tools = false; }
        let calls = at_name(at_index(pr.messages, 1), "tool_calls");
        if !vals[calls].used {
            r.supports_tool_calls = false;
        } else {
            let a = at_name(at_name(at_name(at_index(calls, 0), "function"), "arguments"), "arg");
            if vals[a].used { r.supports_object_arguments = true; }
        }
    }

    if !r.supports_object_arguments {
        // one tool call, with string arguments
        msgs.clear();
        tool_messages(msgs, false, false);
        pr = probe(p, msgs, TOOLS_JSON, Setup.None);
        if !pr.ok {
            r.supports_tool_calls = false;
            r.supports_tools = false;
        } else {
            let tool_name = at_name(at_name(at_index(pr.tools, 0), "function"), "name");
            if !vals[tool_name].used { r.supports_tools = false; }
            let calls = at_name(at_index(pr.messages, 1), "tool_calls");
            if !vals[calls].used { r.supports_tool_calls = false; }
        }
    }

    // parallel tool calls
    msgs.clear();
    tool_messages(msgs, r.supports_object_arguments, true);
    pr = probe(p, msgs, TOOLS_JSON, Setup.None);
    if !pr.ok {
        r.supports_parallel_tool_calls = false;
    } else {
        let calls = at_name(at_index(pr.messages, 1), "tool_calls");
        let second = at_name(at_index(calls, 1), "function");
        if !vals[second].used { r.supports_parallel_tool_calls = false; }
    }

    // reasoning kept in the history (stats would not do: the reasoning may
    // be read by an if, and not written)
    pr = probe(p, "[{\"role\": \"user\", \"content\": \"User message\"}, {\"role\": \"assistant\", \"content\": \"Assistant message\", \"reasoning_content\": \"<REASONING_CONTENT_PLACEHOLDER>\"}, {\"role\": \"user\", \"content\": \"User message\"}, {\"role\": \"assistant\", \"content\": \"Assistant message\", \"reasoning_content\": \"Reasoning content\"}, {\"role\": \"user\", \"content\": \"User message\"}]", "", Setup.Preserve);
    if find(caps_out, "<REASONING_CONTENT_PLACEHOLDER>") >= 0 { r.supports_preserve_reasoning = true; }

    // reasoning effort
    caps_effort = 1;
    pr = probe(p, USER_ONLY, "", Setup.Effort);
    r.supports_reasoning_effort = vals[get_var(pr.ctx, "reasoning_effort")].used;

    if r.supports_reasoning_effort {
        // an unknown level shows the template's fallback: a level that
        // renders the same is not honored; levels that render alike are
        // aliases, of which the highest is kept
        var unknown: u8[>..] = [];
        caps_effort = N_EFFORTS;
        let unknown_ok = probe(p, USER_ONLY, "", Setup.Effort).ok;
        unknown.append(caps_out);
        var renders: u8[][>..] = [];
        var rejected: i64[..7] = [];
        for li in N_EFFORTS {
            caps_effort = li;
            let ok = probe(p, USER_ONLY, "", Setup.Effort).ok;
            if !ok {
                // "none" also turns thinking off, so the server does not answer it with an error
                if li != 0 { rejected.push(li); }
                continue;
            }
            if unknown_ok && caps_out == unknown { continue; }
            var found = -1;
            for rd, k in renders { if found < 0 && rd == caps_out { found = k; } }
            if found >= 0 {
                r.reasoning_effort_levels[found] = li;
            } else {
                renders.push(copy(caps_out));
                r.reasoning_effort_levels.push(li);
            }
        }
        // a template that renders no level fails the probe for another reason: its failures say nothing of the levels
        if r.reasoning_effort_levels.len > 0 { r.reasoning_effort_rejected = rejected; }
    }
    return r;
}
