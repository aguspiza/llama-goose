// llama.cpp's differential autoparser: a chat template rendered with
// pairs of sample conversations that differ in one thing (reasoning or not,
// a tool call or not, one argument or two...), the differences telling
// where its reasoning, content and tool call markers are; then the PEG
// parser of the model's output built from those markers.
// port-of: common/chat-diff-analyzer.cpp, common/chat-auto-parser-helpers.cpp, common/chat-auto-parser-generator.cpp, common/chat-auto-parser.h
import std;
import exception;
import .jinja;
import .jinja_caps;
import .json_schema;
import .peg;
import .chat_peg;
import .chat_render;

namespace autoparser;

//
// text helpers
//

fn is_space(c: u8) -> bool { c == ' ' || (c >= 9 && c <= 13) }

fn trim_whitespace(s: const u8[:]) -> const u8[:] {
    var a = 0;
    while a < s.len && is_space(s[a]) { a++; }
    if a == s.len { return ""; }
    var e = s.len - 1;
    while e > a && is_space(s[e]) { e--; }
    return s[a..e + 1];
}

fn trim_leading_whitespace(s: const u8[:]) -> const u8[:] {
    var a = 0;
    while a < s.len && is_space(s[a]) { a++; }
    return s[a..];
}

fn find_from(s: const u8[:], needle: const u8[:], from: i64) -> i64 {
    if from > s.len { return -1; }
    let r = find(s[from..], needle);
    return if r < 0 { -1 } else { r + from };
}

fn rfind(s: const u8[:], needle: const u8[:]) -> i64 {
    if needle.len > s.len { return -1; }
    var i = s.len - needle.len;
    loop {
        if s[i..i + needle.len] == needle { return i; }
        if i == 0 { return -1; }
        i--;
    }
}

fn contains(s: const u8[:], needle: const u8[:]) -> bool { find(s, needle) >= 0 }

// s.substr(0, s.find(needle)): all of s when it is not there.
fn until_find(s: const u8[:], needle: const u8[:]) -> const u8[:] { let p = find(s, needle); return if p < 0 { s } else { s[..p] }; }

fn common_prefix_len(a: const u8[:], b: const u8[:]) -> i64 { var n = 0; while n < a.len && n < b.len && a[n] == b[n] { n++; } return n; }
fn common_suffix_len(a: const u8[:], b: const u8[:]) -> i64 { var n = 0; while n < a.len && n < b.len && a[a.len - 1 - n] == b[b.len - 1 - n] { n++; } return n; }

// A text of markers (<...>, [...]) and of what is between them.
let SEG_TEXT = 0;
let SEG_MARKER = 1;
struct Seg { kind: i64, at: i64, len: i64 }

fn segmentize(text: const u8[:], out: Seg[>..]&) {
    out.clear();
    var in_marker = false;
    var opener: u8 = 0;
    var last = 0;
    for i in text.len {
        let c = text[i];
        if !in_marker && (c == '<' || c == '[') {
            if last < i { out.push(Seg { kind: SEG_TEXT, at: last, len: i - last }); }
            last = i;
            in_marker = true;
            opener = c;
        } else if in_marker && ((opener == '<' && c == '>') || (opener == '[' && c == ']')) {
            out.push(Seg { kind: SEG_MARKER, at: last, len: i - last + 1 });
            last = i + 1;
            in_marker = false;
            opener = 0;
        }
    }
    if last < text.len { out.push(Seg { kind: SEG_TEXT, at: last, len: text.len - last }); }
}

fn seg_text(text: const u8[:], s: Seg) -> const u8[:] { text[s.at..s.at + s.len] }

// prune_whitespace_segments
fn prune_whitespace(text: const u8[:], segs: Seg[>..]&) {
    var w = 0;
    for k in segs.len {
        if trim_whitespace(seg_text(text, segs[k])).len > 0 { segs[w] = segs[k]; w++; }
    }
    segs.resize(w);
}

// A diff of two texts: the common prefix and suffix (ending on markers),
// and the parts of each between them (calculate_diff_split).
var d_prefix: u8[>..] = [];
var d_suffix: u8[>..] = [];
var d_left: u8[>..] = [];
var d_right: u8[>..] = [];

fn seg_eq(l: const u8[:], a: Seg, r: const u8[:], b: Seg) -> bool { a.kind == b.kind && seg_text(l, a) == seg_text(r, b) }

fn diff_split(left: const u8[:], right: const u8[:]) {
    d_prefix.clear();
    d_suffix.clear();
    d_left.clear();
    d_right.clear();
    var ls: Seg[>..] = [];
    var rs: Seg[>..] = [];
    segmentize(left, ls);
    segmentize(right, rs);
    if ls.len == 0 { d_right.append(right); return; }
    if rs.len == 0 { d_left.append(left); return; }
    var lstart = 0;
    var lend = ls.len - 1;
    var rstart = 0;
    var rend = rs.len - 1;
    var left_done = false;
    var right_done = false;
    // the suffix is built backwards: its segments, last first
    var suffix_rev: u8[>..] = [];
    fn prepend_suffix(b: const u8[:]) {
        var tmp: u8[>..] = [];
        tmp.append(b);
        tmp.append(d_suffix);
        d_suffix.clear();
        d_suffix.append(tmp);
    }
    while lstart != lend && rstart != rend {
        var advanced = false;
        if seg_eq(left, ls[lstart], right, rs[rstart]) {
            d_prefix.append(seg_text(left, ls[lstart]));
            lstart++;
            rstart++;
            advanced = true;
        }
        if seg_eq(left, ls[lend], right, rs[rend]) {
            prepend_suffix(seg_text(left, ls[lend]));
            if lstart != lend { lend--; } else { left_done = true; }
            if rstart != rend { rend--; } else { right_done = true; }
            advanced = true;
        }
        if !advanced { break; }
    }
    if lstart == lend && rstart != rend {
        if seg_eq(left, ls[lstart], right, rs[rend]) {
            prepend_suffix(seg_text(right, rs[rend]));
            rend--;
            left_done = true;
        } else if seg_eq(left, ls[lstart], right, rs[rstart]) {
            d_prefix.append(seg_text(right, rs[rstart]));
            rstart++;
            left_done = true;
        }
    } else if rstart == rend && lstart != lend {
        if seg_eq(left, ls[lend], right, rs[rstart]) {
            prepend_suffix(seg_text(left, ls[lend]));
            lend--;
            right_done = true;
        } else if seg_eq(left, ls[lstart], right, rs[rstart]) {
            d_prefix.append(seg_text(left, ls[lstart]));
            lstart++;
            right_done = true;
        }
    } else if lstart == lend && rstart == rend && seg_eq(left, ls[lstart], right, rs[rstart]) && ls[lstart].kind == SEG_MARKER {
        d_prefix.append(seg_text(right, rs[rstart]));
        left_done = true;
        right_done = true;
    }
    let can_text_suffix = ls[lend].kind == SEG_TEXT && rs[rend].kind == SEG_TEXT;
    let can_text_prefix = rs[rstart].kind == SEG_TEXT && ls[lstart].kind == SEG_TEXT;
    var rem_l: u8[>..] = [];
    var rem_r: u8[>..] = [];
    let l_last = if left_done { lend } else { lend + 1 };
    let r_last = if right_done { rend } else { rend + 1 };
    for k in lstart..l_last { rem_l.append(seg_text(left, ls[k])); }
    for k in rstart..r_last { rem_r.append(seg_text(right, rs[k])); }
    let suffix_len = if can_text_suffix { common_suffix_len(rem_l, rem_r) } else { 0 };
    let prefix_len = if can_text_prefix { common_prefix_len(rem_l[..rem_l.len - suffix_len], rem_r[..rem_r.len - suffix_len]) } else { 0 };
    d_prefix.append(rem_l[..prefix_len]);
    prepend_suffix(rem_l[rem_l.len - suffix_len..]);
    d_left.append(rem_l[prefix_len..rem_l.len - suffix_len]);
    d_right.append(rem_r[prefix_len..rem_r.len - suffix_len]);
    if d_left.len == 0 && d_right.len == 0 {
        // no diff
        d_prefix.clear();
        d_prefix.append(left);
        d_suffix.clear();
    }
    // left all shared with right: the prefix is left
    if d_left.len == 0 && d_right.len > 0 && left.len <= right.len && right[..left.len] == left {
        d_prefix.clear();
        d_prefix.append(left);
        d_suffix.clear();
        d_right.clear();
        d_right.append(right[left.len..]);
    }
}

// until_common_prefix: full up to where the common prefix of left and
// right starts in it ("" if none).
fn until_common_prefix(full: const u8[:], left: const u8[:], right: const u8[:]) -> const u8[:] {
    let n = common_prefix_len(left, right);
    if n == 0 { return ""; }
    let p = find(full, left[..n]);
    return if p < 0 { "" } else { full[..p] };
}

// after_common_suffix: full after the last place the common suffix of
// left and right is in it ("" if none).
fn after_common_suffix(full: const u8[:], left: const u8[:], right: const u8[:]) -> const u8[:] {
    let n = common_suffix_len(left, right);
    if n == 0 { return ""; }
    let p = rfind(full, left[left.len - n..]);
    return if p < 0 { "" } else { full[p + n..] };
}

//
// sample renders
//

let FUN_FIRST = "FFF_FIRST_FUN_F";
let FUN_SECOND = "SSS_SECOND_FUN_S";
let ARG_FIRST = "AA_ARG_FST_AA";
let ARG_SECOND = "BB_ARG_SND_BB";
let USER_MSG = "U_USER_MSG Hello END_U";
let USER_MSG_TWO = "V_USER_MSG Hello END_V";
let ASSISTANT_MSG = "A_ASST_MSG I can help END_A";
let THINKING_CONTENT = "REASON_PART I am thinking END_R";
let CALL_ID_001 = "call00001";
let CALL_ID_002 = "call00002";
let CALL_ID_999 = "call99999";

let PARAMS_SCHEMA = "{\"type\":\"object\",\"properties\":{\"AA_ARG_FST_AA\":{\"type\":\"string\",\"description\":\"First argument\"},\"BB_ARG_SND_BB\":{\"type\":\"string\",\"description\":\"Second argument\"}},\"required\":[]}";

fn tools_json() -> u8[>..] {
    return str("[{\"type\":\"function\",\"function\":{\"name\":\"", FUN_FIRST, "\",\"description\":\"Test function foo\",\"parameters\":", PARAMS_SCHEMA, "}},",
               "{\"type\":\"function\",\"function\":{\"name\":\"", FUN_SECOND, "\",\"description\":\"Test function bar\",\"parameters\":", PARAMS_SCHEMA, "}}]");
}

fn user_msg() -> u8[>..] { str("{\"role\":\"user\",\"content\":\"", USER_MSG, "\"}") }
fn user_msg_two() -> u8[>..] { str("{\"role\":\"user\",\"content\":\"", USER_MSG_TWO, "\"}") }
fn assistant(content: const u8[:]) -> u8[>..] { str("{\"role\":\"assistant\",\"content\":\"", content, "\"}") }
fn assistant_reasoning(content: const u8[:]) -> u8[>..] {
    return str("{\"role\":\"assistant\",\"content\":\"", content, "\",\"reasoning_content\":\"", THINKING_CONTENT, "\"}");
}

// build_tool_call: args a JSON object's text
fn tool_call(name: const u8[:], args: const u8[:], id: const u8[:]) -> u8[>..] {
    return str("{\"id\":\"", id, "\",\"type\":\"function\",\"function\":{\"name\":\"", name, "\",\"arguments\":", args, "}}");
}

fn two_args() -> u8[>..] { str("{\"", ARG_FIRST, "\":\"XXXX\",\"", ARG_SECOND, "\":\"YYYY\"}") }
fn first_tool_call() -> u8[>..] { tool_call(FUN_FIRST, two_args(), CALL_ID_001) }
fn second_tool_call() -> u8[>..] { tool_call(FUN_SECOND, two_args(), CALL_ID_002) }

// {"role": "assistant", "content": "", "tool_calls": [calls]}
fn assistant_calls(calls: const u8[:]) -> u8[>..] { str("{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[", calls, "]}") }

fn messages2(a: const u8[:], b: const u8[:]) -> u8[>..] { str("[", a, ",", b, "]") }

// template_params
struct TP {
    messages: const u8[:] = "[]",
    tools: const u8[:] = "",
    add_generation_prompt: bool = false,
    enable_thinking: bool = true,
}

var tmpl = chat_render::Template { .. };

// apply_template: the render, or false when the template fails.
fn apply_template(tp: TP, out: u8[>..]&) -> bool {
    var g = chat_render::Gen { messages: tp.messages, tools: tp.tools, add_generation_prompt: tp.add_generation_prompt,
                               enable_thinking: tp.enable_thinking, .. };
    let ctx = str("{\"enable_thinking\":", if tp.enable_thinking { "true" } else { "false" }, "}");
    g.extra_context = ctx;
    var ok = false;
    fn body() {
        chat_render::direct_apply(tmpl, g, chat_render::Override { .. });
        ok = true;
    }
    exception::try<jinja::Fail>(body);
    if ok { out.clear(); out.append(chat_render::prompt); }
    return ok;
}

// compare_variants: the two renders in out_a and out_b, and their diff;
// false if either fails.
var out_a: u8[>..] = [];
var out_b: u8[>..] = [];

fn compare(ma: const u8[:], mb: const u8[:], tools: const u8[:]) -> bool { compare_full(ma, mb, tools, false, true, true) }

fn compare_full(ma: const u8[:], mb: const u8[:], tools: const u8[:], add_gen: bool, think_a: bool, think_b: bool) -> bool {
    var la: u8[>..] = [];
    var lb: u8[>..] = [];
    if !apply_template(TP { messages: ma, tools: tools, add_generation_prompt: add_gen, enable_thinking: think_a }, la) { return false; }
    if !apply_template(TP { messages: mb, tools: tools, add_generation_prompt: add_gen, enable_thinking: think_b }, lb) { return false; }
    out_a.clear();
    out_a.append(la);
    out_b.clear();
    out_b.append(lb);
    diff_split(out_a, out_b);
    return true;
}

//
// tagged parsers
//

// Parses input with the parser the builder block makes, from the first
// place it can (parse_anywhere) or from the start; the tags in chat_peg::tags.
fn tagged<B>(anywhere: bool, input: const u8[:]) -> bool {
    peg::begin();
    let a = peg::build(B());
    let ok = if anywhere { chat_peg::parse_anywhere_and_extract(a, input) } else { chat_peg::parse_and_extract(a, input) };
    peg::release(a);
    return ok;
}

fn tag(name: const u8[:]) -> u8[>..] { str(chat_peg::tag_text(name)) }

// p.marker() + p.space(), and the like, in GCC's order
fn marker_space() -> i64 { let s = peg::space(); let m = peg::marker(); return peg::seq2(m, s); }
fn space_marker_space() -> i64 { let s2 = peg::space(); let m = peg::marker(); let s1 = peg::space(); return peg::chain([peg::ci(s1), peg::ci(m), peg::ci(s2)]); }
// last_marker = marker() + zero_or_more(negate(marker()) + any()) + end()
fn last_marker() -> i64 {
    let e = peg::end();
    let a = peg::any();
    let n = peg::negate(peg::marker());
    let z = peg::zero_or_more(peg::seq2(n, a));
    let m = peg::marker();
    return peg::chain([peg::ci(m), peg::ci(z), peg::ci(e)]);
}
// zero_or_more(negate(marker()) + any())
fn non_markers() -> i64 { let a = peg::any(); let n = peg::negate(peg::marker()); return peg::zero_or_more(peg::seq2(n, a)); }

//
// the analysis
//

let MODE_NONE = 0;
let MODE_TAG_BASED = 1;
let MODE_TOOLS_ONLY = 2;

let CONTENT_PLAIN = 0;
let CONTENT_ALWAYS_WRAPPED = 1;
let CONTENT_WRAPPED_WITH_REASONING = 2;

let ID_NONE = 0;
let ID_PRE_FUNC_NAME = 1;
let ID_BETWEEN_FUNC_AND_ARGS = 2;
let ID_POST_ARGS = 3;

let TOOLS_NONE = 0;
let TOOLS_JSON_NATIVE = 1;
let TOOLS_TAG_WITH_JSON = 2;
let TOOLS_TAG_WITH_TAGGED = 3;

// The analysis being made, as growable texts (one template at a time).
var r_mode = MODE_NONE;
var r_start: u8[>..] = [];
var r_end: u8[>..] = [];
var c_mode = CONTENT_PLAIN;
var c_start: u8[>..] = [];
var c_end: u8[>..] = [];
var f_mode = TOOLS_NONE;
var section_start: u8[>..] = [];
var section_end: u8[>..] = [];
var per_call_start: u8[>..] = [];
var per_call_end: u8[>..] = [];
var fun_name_is_key = false;
var tools_array_wrapped = false;
var openai_wrapper_trigger = false;
var function_field: u8[>..] = [];
var name_field: u8[>..] = [];
var args_field: u8[>..] = [];
var id_field: u8[>..] = [];
var gen_id_field: u8[>..] = [];
var parameter_order: u8[>..] = [];      // the names, each followed by a 0 byte
var fn_name_prefix: u8[>..] = [];
var fn_name_suffix: u8[>..] = [];
var fn_args_separator: u8[>..] = [];
var fn_close: u8[>..] = [];
var a_start: u8[>..] = [];
var a_end: u8[>..] = [];
var a_name_prefix: u8[>..] = [];
var a_name_suffix: u8[>..] = [];
var a_value_prefix: u8[>..] = [];
var a_value_suffix: u8[>..] = [];
var a_separator: u8[>..] = [];
var a_tolerate_ws = false;
var id_pos = ID_NONE;
var id_prefix: u8[>..] = [];
var id_suffix: u8[>..] = [];
var user_start: u8[>..] = [];
var assistant_start: u8[>..] = [];
var preserved: u8[>..] = [];            // the tokens, each followed by a 0 byte
var stops: u8[>..] = [];

fn set(dst: u8[>..]&, s: const u8[:]) {
    var tmp: u8[>..] = [];
    tmp.append(s);
    dst.clear();
    dst.append(tmp);
}

fn list_push(l: u8[>..]&, s: const u8[:]) { l.append(s); l.push(0); }

fn list_has(l: const u8[:], s: const u8[:]) -> bool {
    var start = 0;
    for i in l.len {
        if l[i] == 0 {
            if l[start..i] == s { return true; }
            start = i + 1;
        }
    }
    return false;
}

fn clear_analysis() {
    r_mode = MODE_NONE;
    r_start.clear();
    r_end.clear();
    c_mode = CONTENT_PLAIN;
    c_start.clear();
    c_end.clear();
    clear_tools();
    user_start.clear();
    assistant_start.clear();
    preserved.clear();
    stops.clear();
}

fn clear_tools() {
    f_mode = TOOLS_NONE;
    section_start.clear();
    section_end.clear();
    per_call_start.clear();
    per_call_end.clear();
    fun_name_is_key = false;
    tools_array_wrapped = false;
    openai_wrapper_trigger = false;
    set(function_field, "function");
    set(name_field, "name");
    set(args_field, "arguments");
    id_field.clear();
    gen_id_field.clear();
    parameter_order.clear();
    fn_name_prefix.clear();
    fn_name_suffix.clear();
    fn_args_separator.clear();
    fn_close.clear();
    a_start.clear();
    a_end.clear();
    a_name_prefix.clear();
    a_name_suffix.clear();
    a_value_prefix.clear();
    a_value_suffix.clear();
    a_separator.clear();
    a_tolerate_ws = false;
    id_pos = ID_NONE;
    id_prefix.clear();
    id_suffix.clear();
}

// analyze_reasoning::compare_reasoning_presence
fn reasoning_presence() {
    let um = user_msg();
    let m_a = messages2(um, assistant(ASSISTANT_MSG));
    let m_b = messages2(um, assistant_reasoning(ASSISTANT_MSG));
    guard compare(m_a, m_b, "") else { return; }
    guard d_right.len > 0 && contains(d_right, THINKING_CONTENT) else { return; }
    fn wrapped() -> i64 {
        // tag("pre", marker() + space()) + literal(R) + tag("post", space() + marker() + space()) + rest()
        let rest = peg::rest();
        let post = peg::tag("post", space_marker_space());
        let l = peg::literal(THINKING_CONTENT);
        let pre = peg::tag("pre", marker_space());
        return peg::chain([peg::ci(pre), peg::ci(l), peg::ci(post), peg::ci(rest)]);
    }
    fn delimiter() -> i64 {
        // literal(R) + space() + optional(tag("post", marker() + space()) + rest())
        let rest = peg::rest();
        let post = peg::tag("post", marker_space());
        let o = peg::optional(peg::seq2(post, rest));
        let s = peg::space();
        let l = peg::literal(THINKING_CONTENT);
        return peg::chain([peg::ci(l), peg::ci(s), peg::ci(o)]);
    }
    var ok = tagged(true, out_b, wrapped);
    if !ok { ok = tagged(true, out_b, delimiter); }
    guard ok else { return; }
    let pre = tag("pre");
    let post = tag("post");
    if pre.len > 0 && post.len > 0 {
        r_mode = MODE_TAG_BASED;
        set(r_start, pre);
        set(r_end, post);
    } else if post.len > 0 {
        r_mode = MODE_TAG_BASED;
        set(r_end, post);
    }
}

// analyze_reasoning::compare_thinking_enabled
fn thinking_enabled() {
    let m = str("[", user_msg(), "]");
    guard compare_full(m, m, "", true, false, true) else { return; }
    let left_trimmed = str(trim_whitespace(d_left));
    let right_trimmed = str(trim_whitespace(d_right));
    if left_trimmed.len == 0 && d_right.len > 0 {
        if right_trimmed.len > 0 && ends_with(out_b, right_trimmed) {
            if r_start.len == 0 {
                set(r_start, d_right);
                r_mode = MODE_TAG_BASED;
            }
        }
    } else if right_trimmed.len == 0 && d_left.len > 0 {
        if left_trimmed.len > 0 && ends_with(out_a, left_trimmed) && r_end.len == 0 {
            var seg: Seg[>..] = [];
            segmentize(out_a, seg);
            prune_whitespace(out_a, seg);
            if seg.len >= 2 && seg_text(out_a, seg[seg.len - 1]) == left_trimmed && seg[seg.len - 2].kind == SEG_MARKER {
                set(r_start, seg_text(out_a, seg[seg.len - 2]));
            }
            set(r_end, d_left);
            r_mode = MODE_TAG_BASED;
        }
    } else if left_trimmed.len > 0 && right_trimmed.len > 0 {
        // one output's generation prompt tail may be in the other with
        // reasoning markers after it
        let anchor_len = 64;
        for dir in 2 {
            let base = if dir == 0 { out_b[..] } else { out_a[..] };
            let extended = if dir == 0 { out_a[..] } else { out_b[..] };
            let len = min(base.len, anchor_len);
            let anchor = base[base.len - len..];
            let pos = rfind(extended, anchor);
            guard pos >= 0 && pos + len < extended.len else { continue; }
            let extra = trim_whitespace(extended[pos + len..]);
            guard extra.len > 0 else { continue; }
            var seg: Seg[>..] = [];
            segmentize(extra, seg);
            prune_whitespace(extra, seg);
            if seg.len == 2 && seg[0].kind == SEG_MARKER && seg[1].kind == SEG_MARKER {
                if r_start.len == 0 { set(r_start, seg_text(extra, seg[0])); }
                if r_end.len == 0 { set(r_end, seg_text(extra, seg[1])); }
                r_mode = MODE_TAG_BASED;
                break;
            }
        }
    }
    if r_mode == MODE_NONE && r_start.len == 0 && r_end.len > 0 { r_mode = MODE_TAG_BASED; }
}

// analyze_reasoning::compare_reasoning_scope
fn reasoning_scope() {
    let um = user_msg();
    let tools = tools_json();
    let call = tool_call(FUN_FIRST, str("{\"", ARG_FIRST, "\":\"VVVV\",\"", ARG_SECOND, "\":\"XXXX\"}"), CALL_ID_001);
    let with_tools = str("{\"role\":\"assistant\",\"content\":null,\"reasoning_content\":\"", THINKING_CONTENT, "\",\"tool_calls\":[", call, "]}");
    guard compare(messages2(um, assistant_reasoning(ASSISTANT_MSG)), messages2(um, with_tools), tools) else { return; }
    let in_a = contains(out_a, THINKING_CONTENT);
    let in_b = contains(out_b, THINKING_CONTENT);
    guard !in_a && in_b else { return; }
    r_mode = MODE_TOOLS_ONLY;
    fn wrapped() -> i64 {
        // tag("pre", marker() + space()) + literal(R) + space() + tag("post", marker() + space())
        let post = peg::tag("post", marker_space());
        let s = peg::space();
        let l = peg::literal(THINKING_CONTENT);
        let pre = peg::tag("pre", marker_space());
        return peg::chain([peg::ci(pre), peg::ci(l), peg::ci(s), peg::ci(post)]);
    }
    fn delimiter() -> i64 {
        // literal(R) + space() + optional(tag("post", marker() + space()))
        let o = peg::optional(peg::tag("post", marker_space()));
        let s = peg::space();
        let l = peg::literal(THINKING_CONTENT);
        return peg::chain([peg::ci(l), peg::ci(s), peg::ci(o)]);
    }
    if tagged(true, out_b, wrapped) {
        set(r_start, tag("pre"));
        set(r_end, tag("post"));
    } else if tagged(true, out_b, delimiter) {
        set(r_end, tag("post"));
    } else {
        r_mode = MODE_NONE;
    }
}

// analyze_content
fn analyze_content() {
    let um = user_msg();
    let tools = tools_json();
    let base = messages2(um, assistant(ASSISTANT_MSG));
    let with_tools = assistant_calls(tool_call("test_func", "{\"arg1\":\"value1\"}", CALL_ID_001));
    let with_reasoning = str("{\"role\":\"assistant\",\"content\":\"\",\"reasoning_content\":\"", THINKING_CONTENT, "\"}");
    guard compare(base, messages2(um, with_tools), tools) else { return; }
    let dt_left = str(d_left);
    guard compare(base, messages2(um, with_reasoning), tools) else { return; }
    let dr_left = str(d_left);
    var found_plain = false;
    if trim_whitespace(dt_left) == ASSISTANT_MSG {
        fn plain() -> i64 {
            // space() + diff_reasoning.left + space() + optional(marker()) + space() + end()
            let e = peg::end();
            let s3 = peg::space();
            let om = peg::optional(peg::marker());
            let s2 = peg::space();
            let s1 = peg::space();
            return peg::chain([peg::ci(s1), peg::cs(dr_left), peg::ci(s2), peg::ci(om), peg::ci(s3), peg::ci(e)]);
        }
        fn post_reasoning() -> i64 {
            let l2 = peg::literal(ASSISTANT_MSG);
            let s = peg::space();
            let l1 = peg::literal(r_end);
            return peg::chain([peg::ci(l1), peg::ci(s), peg::ci(l2)]);
        }
        if tagged(false, dr_left, plain) {
            c_mode = CONTENT_PLAIN;
            found_plain = true;
        } else if r_mode != MODE_NONE && r_end.len > 0 && tagged(true, dr_left, post_reasoning) {
            c_mode = CONTENT_PLAIN;
            found_plain = true;
        }
    }
    if !found_plain {
        var rdiff = str(dr_left);
        if r_end.len > 0 && contains(rdiff, r_end) { let cut = str(rdiff[find(rdiff, r_end) + r_end.len..]); rdiff = copy(cut); }
        let pure = if rdiff.len > dt_left.len { rdiff[..] } else { dt_left[..] };
        fn wrapped() -> i64 {
            let rest = peg::rest();
            let post = peg::tag("post", marker_space());
            let s = peg::space();
            let l = peg::literal(ASSISTANT_MSG);
            let pre = peg::tag("pre", marker_space());
            return peg::chain([peg::ci(pre), peg::ci(l), peg::ci(s), peg::ci(post), peg::ci(rest)]);
        }
        tagged(true, pure, wrapped);
        set(c_start, tag("pre"));
        set(c_end, tag("post"));
    }
    if c_start.len > 0 || c_end.len > 0 { c_mode = CONTENT_ALWAYS_WRAPPED; }
}

fn content_always_wrapped() -> bool { c_mode == CONTENT_ALWAYS_WRAPPED && c_start.len > 0 && c_end.len > 0 }

//
// tool calls
//

fn one_arg_call(name: const u8[:], arg: const u8[:], value: const u8[:], id: const u8[:]) -> u8[>..] {
    return tool_call(name, str("{\"", arg, "\":\"", value, "\"}"), id);
}

// analyze_tools::analyze_tool_calls
fn analyze_tool_calls(supports_parallel: bool) {
    let um = user_msg();
    let tools = tools_json();
    guard compare(messages2(um, assistant(ASSISTANT_MSG)), messages2(um, assistant_calls(first_tool_call())), tools) else { return; }
    let haystack = str(d_right);
    guard haystack.len > 0 else { return; }
    analyze_tool_call_format(haystack, supports_parallel);
}

fn in_json_haystack(haystack: const u8[:], needle: const u8[:]) -> bool {
    fn build() -> i64 {
        // choice({ "{", ":" }) << choice({ tag("dq", "\"" + needle + "\"") })
        let a = peg::literal("{");
        let b = peg::literal(":");
        let left = peg::choice([a, b]);
        let q2 = peg::literal("\"");
        let n = peg::literal(needle);
        let q1 = peg::literal("\"");
        let right = peg::choice([peg::tag("dq", peg::chain([peg::ci(q1), peg::ci(n), peg::ci(q2)]))]);
        return peg::spaced(left, right);
    }
    return tagged(true, haystack, build);
}

fn analyze_tool_call_format(haystack: const u8[:], supports_parallel: bool) {
    let fun_quote = in_json_haystack(haystack, FUN_FIRST);
    let arg_quote = in_json_haystack(haystack, ARG_FIRST);
    f_mode = if fun_quote { TOOLS_JSON_NATIVE } else if arg_quote { TOOLS_TAG_WITH_JSON } else { TOOLS_TAG_WITH_TAGGED };
    // the reasoning markers out
    var clean = str(haystack);
    if r_start.len > 0 {
        let p = find(haystack, r_start);
        if p >= 0 { clean = str(haystack[..p], haystack[p + r_start.len..]); }
    }
    if r_end.len > 0 {
        let p = find(clean, r_end);
        if p >= 0 { let cut = str(clean[..p], clean[p + r_end.len..]); clean = copy(cut); }
    }
    if f_mode == TOOLS_JSON_NATIVE {
        json_native_format(clean);
        if supports_parallel { json_native_parallel_calls(); }
    } else {
        non_json_format(clean);
    }
    // whitespace does not matter in end markers
    set(section_end, trim_whitespace(section_end));
    set(per_call_end, trim_whitespace(per_call_end));
}

fn json_native_parallel_calls() {
    let um = user_msg();
    let tools = tools_json();
    let one = first_tool_call();
    let two = str(first_tool_call(), ",", second_tool_call());
    guard compare(messages2(um, assistant_calls(one)), messages2(um, assistant_calls(two)), tools) else { return; }
    if section_start.len > 0 && contains(d_right, section_start) {
        set(per_call_start, section_start);
        set(per_call_end, section_end);
        section_start.clear();
        section_end.clear();
    }
}

fn register_field(prefix: const u8[:], key: const u8[:], value: i64) {
    let full = if prefix.len > 0 { str(prefix, ".", key) } else { str(key) };
    let is_str = json_schema::json_kind(value) == json_schema::J_STRING;
    var dumped: u8[>..] = [];
    json_schema::json_dump(dumped, value);
    if is_str && contains(json_schema::json_str(value), "call0000") { set(id_field, full); }
    else if is_str && json_schema::json_str(value) == FUN_FIRST { set(name_field, full); }
    else if contains(dumped, ARG_FIRST) { set(args_field, full); }
    else if contains(key, "id") { set(gen_id_field, full); }
}

struct Located { pos: i64, at: i64, len: i64 }

// analyze_tool_call_format_json_native
fn json_native_format(clean: const u8[:]) {
    var json_start = -1;
    for i in clean.len { if clean[i] == '{' { json_start = i; break; } }
    var json_end = -1;
    for i in clean.len { if clean[i] == '}' { json_end = i; } }
    if json_start < 0 { json_schema::fail("basic_string::substr: __pos out of range"); }
    let cut = str(clean[json_start..min(clean.len, max(json_start, json_end + 1))]);
    let call_struct = json_schema::json_value(cut);
    if call_struct < 0 { json_schema::fail(str("[json.exception.parse_error] ", json_schema::err)); }
    if json_schema::json_kind(call_struct) == json_schema::J_OBJECT {
        for k in json_schema::json_len(call_struct) {
            let key = str(json_schema::json_key(call_struct, k));
            let value = json_schema::json_at(call_struct, k);
            if key == FUN_FIRST {
                fun_name_is_key = true;
                name_field.clear();
                args_field.clear();
            } else {
                var dumped: u8[>..] = [];
                json_schema::json_dump(dumped, value);
                if json_schema::json_kind(value) == json_schema::J_OBJECT && !contains(dumped, ARG_FIRST) {
                    set(function_field, key);
                    for j in json_schema::json_len(value) {
                        register_field(key, str(json_schema::json_key(value, j)), json_schema::json_at(value, j));
                    }
                }
                register_field("", key, value);
            }
        }
    }
    fn array_parser() -> i64 {
        // tag("pre", "[" + space()) + literal(cut) + tag("post", space() + "]")
        let close = peg::literal("]");
        let s2 = peg::space();
        let post = peg::tag("post", peg::seq2(s2, close));
        let l = peg::literal(cut);
        let s1 = peg::space();
        let open = peg::literal("[");
        let pre = peg::tag("pre", peg::seq2(open, s1));
        return peg::chain([peg::ci(pre), peg::ci(l), peg::ci(post)]);
    }
    if tagged(true, clean, array_parser) {
        tools_array_wrapped = true;
        json_start -= chat_peg::tag_text("pre").len;
        json_end += chat_peg::tag_text("post").len;
    }
    json_end++;
    // the fields in the order they come in (pairs sorted by place, then name)
    var names: u8[>..] = [];
    var located: Located[>..] = [];
    fn locate(f: const u8[:]) {
        guard f.len > 0 else { return; }
        let p = find(clean, f);
        located.push(Located { pos: if p < 0 { 9223372036854775807 } else { p }, at: names.len, len: f.len });
        names.append(f);
    }
    locate(str(name_field));
    locate(str(args_field));
    locate(str(id_field));
    locate(str(gen_id_field));
    stable_sort(located) { a, b => a.pos < b.pos || (a.pos == b.pos && cmp(names[a.at..a.at + a.len], names[b.at..b.at + b.len]) < 0) };
    parameter_order.clear();
    for l in located { list_push(parameter_order, names[l.at..l.at + l.len]); }
    set(section_start, trim_leading_whitespace(clean[..max(0, json_start)]));
    set(section_end, trim_whitespace(clean[min(clean.len, json_end)..]));
    if tools_array_wrapped && section_end == "]" { section_end.clear(); }
}

fn cmp(a: const u8[:], b: const u8[:]) -> i64 {
    for i in min(a.len, b.len) { if a[i] != b[i] { return if a[i] < b[i] { -1 } else { 1 }; } }
    return a.len - b.len;
}

// analyze_tool_call_format_non_json
fn non_json_format(clean: const u8[:]) {
    fn fun_marker_parser() -> i64 {
        // A: tag("fun_pre", "<" + until_one_of({">", F})) + F + tag("fun_post", negate(space() + "<") + until(">") + ">") + space()
        let a_s = peg::space();
        let a_gt = peg::literal(">");
        let a_u = peg::until(">");
        let a_lt = peg::literal("<");
        let a_sp = peg::space();
        let a_neg = peg::negate(peg::seq2(a_sp, a_lt));
        let a_post = peg::tag("fun_post", peg::chain([peg::ci(a_neg), peg::ci(a_u), peg::ci(a_gt)]));
        let a_f = peg::literal(FUN_FIRST);
        let a_u1 = peg::until_one_of([">", FUN_FIRST]);
        let a_open = peg::literal("<");
        let a_pre = peg::tag("fun_pre", peg::seq2(a_open, a_u1));
        let a = peg::chain([peg::ci(a_pre), peg::ci(a_f), peg::ci(a_post), peg::ci(a_s)]);
        // B: tag("fun_pre", "[" + until_one_of({"]", F})) + F + tag("fun_post", negate(space() + "[" + until("]") + "]") + space())
        let b_s = peg::space();
        let b_cl = peg::literal("]");
        let b_u = peg::until("]");
        let b_op = peg::literal("[");
        let b_sp = peg::space();
        let b_neg = peg::negate(peg::chain([peg::ci(b_sp), peg::ci(b_op), peg::ci(b_u), peg::ci(b_cl)]));
        let b_post = peg::tag("fun_post", peg::seq2(b_neg, b_s));
        let b_f = peg::literal(FUN_FIRST);
        let b_u1 = peg::until_one_of(["]", FUN_FIRST]);
        let b_open = peg::literal("[");
        let b_pre = peg::tag("fun_pre", peg::seq2(b_open, b_u1));
        let b = peg::chain([peg::ci(b_pre), peg::ci(b_f), peg::ci(b_post)]);
        return peg::tag("fun_marker", peg::choice([a, b]));
    }
    var fun_marker = str(FUN_FIRST);
    if tagged(true, clean, fun_marker_parser) { fun_marker = tag("fun_marker"); }
    let fun_pre = tag("fun_pre");
    fn per_tool_parser() -> i64 {
        let rest = peg::tag("rest", peg::rest());
        let pre = peg::tag("fun_pre", peg::until(fun_marker));
        let cs = peg::tag("call_start", marker_space());
        let ss = peg::tag("sec_start", marker_space());
        return peg::chain([peg::ci(ss), peg::ci(cs), peg::ci(pre), peg::cs(fun_marker), peg::ci(rest)]);
    }
    fn section_parser() -> i64 {
        let rest = peg::tag("rest", peg::rest());
        let ss = peg::tag("sec_start", marker_space());
        return peg::chain([peg::ci(ss), peg::cs(fun_marker), peg::ci(rest)]);
    }
    fn double_closer() -> i64 {
        let e = peg::end();
        let se = peg::tag("sec_end", marker_space());
        let ce = peg::tag("call_end", marker_space());
        return peg::chain([peg::ci(ce), peg::ci(se), peg::ci(e)]);
    }
    fn single_closer() -> i64 {
        let e = peg::end();
        let se = peg::tag("sec_end", marker_space());
        return peg::seq2(se, e);
    }
    var call_start = str("");
    var sec_start = str("");
    if tagged(true, clean, per_tool_parser) {
        call_start = tag("call_start");
        sec_start = tag("sec_start");
        let rest = tag("rest");
        tagged(true, rest, double_closer);
        set(fn_name_prefix, str(fun_pre, fn_name_prefix));
    } else {
        tagged(true, clean, section_parser);
        call_start = tag("call_start");
        sec_start = tag("sec_start");
        let rest = tag("rest");
        tagged(true, rest, single_closer);
    }
    set(per_call_start, call_start);
    set(per_call_end, chat_peg::tag_text("call_end"));
    set(section_start, sec_start);
    set(section_end, chat_peg::tag_text("sec_end"));
}

fn count_occurrences(haystack: const u8[:], needle: const u8[:]) -> i64 {
    var n = 0;
    var p = find(haystack, needle);
    while p >= 0 { n++; p = find_from(haystack, needle, p + needle.len); }
    return n;
}

// analyze_tools::check_per_call_markers
fn check_per_call_markers() {
    let um = user_msg();
    let tools = tools_json();
    let one = first_tool_call();
    let two = str(first_tool_call(), ",", second_tool_call());
    guard compare(messages2(um, assistant_calls(one)), messages2(um, assistant_calls(two)), tools) else { return; }
    let a_out = str(out_a);
    let b_out = str(out_b);
    let suffix = str(d_suffix);
    let right = str(d_right);
    diff_split(suffix, right);
    let second = str(trim_leading_whitespace(d_right));
    if section_start.len > 0 && starts_with(second, section_start) {
        set(per_call_start, section_start);
        set(per_call_end, section_end);
        section_start.clear();
        section_end.clear();
    }
    if per_call_end.len > 0 {
        let calls_one = count_occurrences(a_out, per_call_end);
        let calls_two = count_occurrences(b_out, per_call_end);
        if calls_one > 0 && calls_one == calls_two {
            set(section_end, per_call_end);
            per_call_end.clear();
        }
    }
}

fn arg_start() -> i64 {
    let a = peg::literal(ARG_FIRST);
    let b = peg::literal(ARG_SECOND);
    let ch = peg::choice([a, b]);
    let s = peg::space();
    let m = peg::marker();
    return peg::chain([peg::ci(m), peg::ci(s), peg::ci(ch)]);
}

// analyze_tools::extract_function_markers
fn extract_function_markers() {
    let um = user_msg();
    let tools = tools_json();
    let foofoo = messages2(um, assistant_calls(first_tool_call()));
    guard compare(foofoo, messages2(um, assistant_calls(second_tool_call())), tools) else { return; }
    let prefix = str(d_prefix);
    let suffix = str(d_suffix);
    let left = str(d_left);
    let right = str(d_right);
    guard contains(left, FUN_FIRST) && contains(right, FUN_SECOND) else { return; }
    let prefix_marker = if per_call_start.len > 0 { str(per_call_start) } else { str(section_start) };
    if prefix_marker.len > 0 && rfind(prefix, prefix_marker) >= 0 {
        set(fn_name_prefix, prefix[rfind(prefix, prefix_marker) + prefix_marker.len..]);
    }
    fn name_parser() -> i64 {
        // tag("pre", until(F)) + literal(F) + tag("post", zero_or_more(negate(marker()) + any()))
        let post = peg::tag("post", non_markers());
        let f = peg::literal(FUN_FIRST);
        let pre = peg::tag("pre", peg::until(FUN_FIRST));
        return peg::chain([peg::ci(pre), peg::ci(f), peg::ci(post)]);
    }
    if tagged(false, left, name_parser) {
        set(fn_name_prefix, str(fn_name_prefix, chat_peg::tag_text("pre")));
        set(fn_name_suffix, chat_peg::tag_text("post"));
    }
    if f_mode == TOOLS_TAG_WITH_JSON {
        fn suffix_parser() -> i64 {
            // tag("ext", zero_or_more(marker() | (negate("{") + negate("[") + any())))
            //   + zero_or_more(negate(marker()) + any()) + marker()
            let a = peg::any();
            let nb = peg::negate(peg::literal("["));
            let na = peg::negate(peg::literal("{"));
            let rest = peg::chain([peg::ci(na), peg::ci(nb), peg::ci(a)]);
            let m = peg::marker();
            let non_json = peg::choice2(m, rest);
            let m2 = peg::marker();
            let nm = non_markers();
            let after_json = peg::seq2(nm, m2);
            let ext = peg::tag("ext", peg::zero_or_more(non_json));
            return peg::seq2(ext, after_json);
        }
        if tagged(false, suffix, suffix_parser) { set(fn_name_suffix, str(fn_name_suffix, chat_peg::tag_text("ext"))); }
    } else {
        fn ext_parser() -> i64 { peg::tag("ext", non_markers()) }
        if tagged(false, suffix, ext_parser) {
            let ext = tag("ext");
            set(fn_name_suffix, str(fn_name_suffix, ext));
            fn sep_parser() -> i64 {
                let tail = arg_start();
                let a = peg::any();
                let n = peg::negate(arg_start());
                let sep = peg::tag("sep", peg::zero_or_more(peg::seq2(n, a)));
                return peg::seq2(sep, tail);
            }
            if tagged(false, suffix[min(suffix.len, ext.len)..], sep_parser) { set(fn_args_separator, trim_whitespace(chat_peg::tag_text("sep"))); }
        }
    }
    // the closer, between the last argument and the call or section end
    let suffix_marker = if per_call_end.len > 0 { str(per_call_end) } else { str(section_end) };
    var closer = str("");
    if suffix_marker.len == 0 {
        if compare(foofoo, messages2(um, assistant(ASSISTANT_MSG)), tools) {
            let p = find(d_left, "YYYY");
            let at = if p < 0 { 3 } else { p + 4 };
            if at > d_left.len { json_schema::fail("basic_string::substr: __pos out of range"); }
            closer = str(d_left[at..]);
        }
    } else {
        closer = str(until_find(suffix, suffix_marker));
    }
    if closer.len > 0 {
        if f_mode == TOOLS_TAG_WITH_TAGGED {
            fn closer_tagged() -> i64 {
                // until("YYYY") + "YYYY" + space() + marker() + space() + tag("close", rest())
                let close = peg::tag("close", peg::rest());
                let s2 = peg::space();
                let m = peg::marker();
                let s1 = peg::space();
                let y = peg::literal("YYYY");
                let u = peg::until("YYYY");
                return peg::chain([peg::ci(u), peg::ci(y), peg::ci(s1), peg::ci(m), peg::ci(s2), peg::ci(close)]);
            }
            if tagged(false, closer, closer_tagged) { set(fn_close, chat_peg::tag_text("close")); }
        } else if f_mode == TOOLS_TAG_WITH_JSON {
            fn closer_json() -> i64 {
                let post = peg::tag("post_val", peg::rest());
                let y = peg::literal("YYYY");
                let u = peg::until("YYYY");
                return peg::chain([peg::ci(u), peg::ci(y), peg::ci(post)]);
            }
            if tagged(false, closer, closer_json) {
                let post = tag("post_val");
                var pos = -1;
                for i in post.len { if post[i] == '}' || post[i] == ']' { pos = i; } }
                if pos >= 0 && pos < post.len - 1 { set(fn_close, trim_leading_whitespace(post[pos + 1..])); }
            }
        }
    }
    set(fn_close, trim_leading_whitespace(fn_close));
}

// zero_or_more(negate(last_marker) + any()) + tag(name, last_marker), on all of s
fn last_marker_of(s: const u8[:], tag_name: const u8[:]) -> bool {
    fn build() -> i64 {
        let t2 = peg::tag(tag_name, last_marker());
        let a = peg::any();
        let n = peg::negate(last_marker());
        let z = peg::zero_or_more(peg::seq2(n, a));
        return peg::seq2(z, t2);
    }
    return tagged(false, s, build);
}

// analyze_tools::extract_argument_name_markers
fn extract_argument_name_markers() {
    let um = user_msg();
    let tools = tools_json();
    guard compare(messages2(um, assistant_calls(one_arg_call(FUN_FIRST, ARG_FIRST, "XXXX", CALL_ID_001))), messages2(um, assistant_calls(one_arg_call(FUN_FIRST, ARG_SECOND, "YYYY", CALL_ID_001))), tools) else { return; }
    let prefix = str(d_prefix);
    let suffix = str(d_suffix);
    let left = str(d_left);
    let right = str(d_right);
    guard left.len > 0 && right.len > 0 else { return; }
    fn left_parser() -> i64 {
        let sfx = peg::tag("suffix", peg::until_one_of(["\"", "X"]));
        let l = peg::literal(ARG_FIRST);
        let pre = peg::tag("pre", peg::until(ARG_FIRST));
        return peg::chain([peg::ci(pre), peg::ci(l), peg::ci(sfx)]);
    }
    fn right_parser() -> i64 {
        let sfx = peg::tag("suffix", peg::until_one_of(["\"", "Y"]));
        let l = peg::literal(ARG_SECOND);
        let pre = peg::tag("pre", peg::until(ARG_SECOND));
        return peg::chain([peg::ci(pre), peg::ci(l), peg::ci(sfx)]);
    }
    let lok = tagged(true, left, left_parser);
    let lpre = tag("pre");
    let lsuf = tag("suffix");
    let rok = tagged(true, right, right_parser);
    let rpre = tag("pre");
    let rsuf = tag("suffix");
    if lok && rok && lpre.len > 0 && lpre == rpre && lsuf == rsuf {
        set(a_name_prefix, lpre);
        set(a_name_suffix, lsuf);
    } else if left[..min(left.len, ARG_FIRST.len)] == ARG_FIRST && right[..min(right.len, ARG_SECOND.len)] == ARG_SECOND {
        if last_marker_of(prefix, "name_prefix") { set(a_name_prefix, chat_peg::tag_text("name_prefix")); } else { set(a_name_prefix, prefix); }
        let after_first = str(left[ARG_FIRST.len..], suffix);
        fn suffix_parser() -> i64 {
            let s = peg::space();
            let m = peg::marker();
            let nm = non_markers();
            return peg::tag("suffix", peg::chain([peg::ci(nm), peg::ci(m), peg::ci(s)]));
        }
        if tagged(true, after_first, suffix_parser) { set(a_name_suffix, chat_peg::tag_text("suffix")); }
    }
}

// analyze_tools::extract_argument_value_markers
fn extract_argument_value_markers() {
    let um = user_msg();
    let tools = tools_json();
    guard compare(messages2(um, assistant_calls(one_arg_call(FUN_FIRST, ARG_FIRST, "XXXX", CALL_ID_001))), messages2(um, assistant_calls(one_arg_call(FUN_FIRST, ARG_FIRST, "YYYY", CALL_ID_001))), tools) else { return; }
    guard d_left == "XXXX" && d_right == "YYYY" else { return; }
    let ending = str(ARG_FIRST, a_name_suffix);
    var prefix = str(d_prefix);
    let suffix = str(d_suffix);
    let rp = rfind(prefix, ending);
    if rp >= 0 { let cut = str(prefix[rp + ending.len..]); prefix = copy(cut); }
    if prefix.len > 0 {
        if last_marker_of(prefix, "val_prefix") { set(a_value_prefix, chat_peg::tag_text("val_prefix")); } else { set(a_value_prefix, prefix); }
    }
    var value_suffix = str(suffix);
    if fn_close.len > 0 {
        let p = find(value_suffix, fn_close);
        if p >= 0 { let cut = str(value_suffix[..p]); value_suffix = copy(cut); }
    } else if per_call_end.len > 0 || section_end.len > 0 {
        let end_marker = if per_call_end.len > 0 { str(per_call_end) } else { str(section_end) };
        let p = find(value_suffix, end_marker);
        if p >= 0 { let cut = str(value_suffix[..p]); value_suffix = copy(cut); }
    }
    if trim_whitespace(value_suffix).len > 0 { set(a_value_suffix, value_suffix); }
}

// analyze_tools::extract_argument_separator
fn extract_argument_separator() {
    let um = user_msg();
    let tools = tools_json();
    guard compare(messages2(um, assistant_calls(one_arg_call(FUN_FIRST, ARG_FIRST, "XXXX", CALL_ID_001))), messages2(um, assistant_calls(first_tool_call())), tools) else { return; }
    if d_right.len > 0 { set(a_separator, until_common_prefix(d_right, ARG_FIRST, ARG_SECOND)); }
}

fn first_marker_of(s: const u8[:]) -> u8[>..] {
    fn build() -> i64 { peg::tag("m", peg::marker()) }
    return if tagged(true, s, build) { tag("m") } else { str("") };
}

fn last_marker_anywhere(s: const u8[:]) -> u8[>..] {
    fn build() -> i64 {
        let m = peg::tag("m", peg::marker());
        let a = peg::any();
        let n = peg::negate(last_marker());
        let z = peg::zero_or_more(peg::seq2(n, a));
        return peg::seq2(z, m);
    }
    return if tagged(true, s, build) { tag("m") } else { str("") };
}

// analyze_tools::extract_call_id_markers
fn extract_call_id_markers() {
    let um = user_msg();
    let tools = tools_json();
    let alt = tool_call(FUN_FIRST, two_args(), CALL_ID_999);
    guard compare(messages2(um, assistant_calls(first_tool_call())), messages2(um, assistant_calls(alt)), tools) else { return; }
    guard d_left.len > 0 || d_right.len > 0 else { return; }
    let prefix = str(d_prefix);
    let suffix = str(d_suffix);
    let common_id = "call";
    let in_prefix = rfind(prefix, FUN_FIRST);
    let in_suffix = find(suffix, FUN_FIRST);
    if in_prefix >= 0 && in_suffix < 0 {
        let args_in_prefix = find_from(prefix, "{", in_prefix);
        let args_in_suffix = find(suffix, "{");
        if args_in_suffix >= 0 && args_in_prefix < 0 {
            id_pos = ID_BETWEEN_FUNC_AND_ARGS;
            let after_func = str(prefix[in_prefix + FUN_FIRST.len..]);
            fn id_prefix_parser() -> i64 {
                // tag("prefix", marker()) + zero_or_more(negate(marker()) + negate(literal(common)) + any()) + literal(common)
                let l = peg::literal(common_id);
                let a = peg::any();
                let nc = peg::negate(peg::literal(common_id));
                let nm = peg::negate(peg::marker());
                let z = peg::zero_or_more(peg::chain([peg::ci(nm), peg::ci(nc), peg::ci(a)]));
                let pfx = peg::tag("prefix", peg::marker());
                return peg::chain([peg::ci(pfx), peg::ci(z), peg::ci(l)]);
            }
            if tagged(true, after_func, id_prefix_parser) { set(id_prefix, chat_peg::tag_text("prefix")); }
            else { set(id_prefix, last_marker_anywhere(after_func)); }
            fn suffix_parser() -> i64 {
                let sfx = peg::tag("suffix", peg::marker());
                let a = peg::any();
                let nb = peg::negate(peg::literal("{"));
                let nm = peg::negate(peg::marker());
                let z = peg::zero_or_more(peg::chain([peg::ci(nm), peg::ci(nb), peg::ci(a)]));
                return peg::seq2(z, sfx);
            }
            if tagged(true, suffix, suffix_parser) { set(id_suffix, chat_peg::tag_text("suffix")); }
        } else if args_in_prefix >= 0 {
            id_pos = ID_POST_ARGS;
            let after_args = str(prefix[args_in_prefix..]);
            var closing = -1;
            for i in after_args.len { if after_args[i] == '}' { closing = i; } }
            if closing >= 0 { set(id_prefix, last_marker_anywhere(after_args[closing + 1..])); }
            set(id_suffix, first_marker_of(suffix));
        }
    } else if in_suffix >= 0 && in_prefix < 0 {
        id_pos = ID_PRE_FUNC_NAME;
        set(id_prefix, last_marker_anywhere(prefix));
        set(id_suffix, first_marker_of(suffix[..in_suffix]));
    }
    if id_prefix == a_end { id_prefix.clear(); }
    if id_suffix == a_start { id_suffix.clear(); }
    if id_pos != ID_NONE && id_suffix.len > 0 && starts_with(per_call_end, id_suffix) { per_call_end.clear(); }
}

fn analyze_tools(caps: jinja::Caps) {
    analyze_tool_calls(caps.supports_parallel_tool_calls);
    guard f_mode != TOOLS_NONE && f_mode != TOOLS_JSON_NATIVE else { return; }
    if caps.supports_parallel_tool_calls { check_per_call_markers(); }
    extract_function_markers();
    if f_mode == TOOLS_TAG_WITH_TAGGED {
        extract_argument_name_markers();
        extract_argument_value_markers();
    }
    extract_argument_separator();
    // extract_args_markers only changes JSON native analyses, which do not get here
    extract_call_id_markers();
}

//
// turn markers
//

fn detect_assistant_start() {
    let um = user_msg();
    guard compare(str("[", um, "]"), messages2(um, assistant(ASSISTANT_MSG)), "") else { return; }
    var ast_prefix = str(until_find(d_right, ASSISTANT_MSG));
    if r_start.len > 0 {
        let p = find(ast_prefix, trim_whitespace(r_start));
        if p >= 0 { let cut = str(ast_prefix[..p]); ast_prefix = copy(cut); }
    }
    if r_end.len > 0 {
        let p = find(ast_prefix, trim_whitespace(r_end));
        if p >= 0 { let cut = str(ast_prefix[..p]); ast_prefix = copy(cut); }
    }
    set(assistant_start, trim_whitespace(ast_prefix));
}

fn lower_has(s: const u8[:], needle: const u8[:]) -> bool {
    var lower: u8[>..] = [];
    for c in s { lower.push(if c >= 'A' && c <= 'Z' { c + 32 } else { c }); }
    return contains(lower, needle);
}

fn detect_user_start() {
    let um = user_msg();
    var ok = compare("[]", str("[", um, "]"), "");
    if !ok {
        let two = user_msg_two();
        let a = assistant(ASSISTANT_MSG);
        ok = compare(str("[", two, ",", a, "]"), str("[", two, ",", a, ",", um, "]"), "");
        guard ok else { return; }
    }
    var usermsg = str(d_right);
    let ap = find(usermsg, ASSISTANT_MSG);
    if ap >= 0 { let cut = str(usermsg[ap + ASSISTANT_MSG.len..]); usermsg = copy(cut); }
    let candidate = str(until_find(usermsg, USER_MSG));
    var segs: Seg[>..] = [];
    segmentize(candidate, segs);
    var result: u8[>..] = [];
    var encountered_marker = false;
    for s in segs {
        let v = seg_text(candidate, s);
        // what may be end markers, at the start, out
        if s.kind == SEG_MARKER && !encountered_marker && (lower_has(v, "end") || lower_has(v, "close")) { continue; }
        if s.kind == SEG_TEXT && !encountered_marker && trim_whitespace(v).len == 0 { continue; }
        if s.kind == SEG_MARKER { encountered_marker = true; }
        result.append(v);
    }
    set(user_start, trim_whitespace(result));
}

fn add_preserved(s: const u8[:]) {
    let tk = trim_whitespace(s);
    if tk.len > 0 && !list_has(preserved, tk) { list_push(preserved, tk); }
}

fn collect_preserved_tokens() {
    add_preserved(r_start);
    add_preserved(r_end);
    add_preserved(c_start);
    add_preserved(c_end);
    add_preserved(section_start);
    add_preserved(section_end);
    add_preserved(per_call_start);
    add_preserved(per_call_end);
    add_preserved(fn_name_prefix);
    add_preserved(fn_name_suffix);
    add_preserved(fn_args_separator);
    add_preserved(fn_close);
    add_preserved(a_start);
    add_preserved(a_end);
    add_preserved(a_name_prefix);
    add_preserved(a_name_suffix);
    add_preserved(a_separator);
    add_preserved(a_value_prefix);
    add_preserved(a_value_suffix);
    add_preserved(id_prefix);
    add_preserved(id_suffix);
}

// The fixes for templates the analysis gets wrong (workarounds).
fn workarounds(src: const u8[:]) {
    fn has(s: const u8[:]) -> bool { contains(src, s) }
    // old reasoning Qwen templates
    if has("content.split('</think>')") && !has("reasoning_content") && !has("<SPECIAL_12>") && r_mode == MODE_NONE {
        r_mode = MODE_TAG_BASED;
        set(r_start, "<think>");
        set(r_end, "</think>");
        list_push(preserved, "<think>");
        list_push(preserved, "</think>");
    }
    // Granite 3.3
    if has("Write your thoughts between <think></think> and write your response between <response></response>") {
        r_mode = MODE_TAG_BASED;
        set(r_start, "<think>");
        set(r_end, "</think>");
        list_push(preserved, "<think>");
        list_push(preserved, "</think>");
        c_mode = CONTENT_WRAPPED_WITH_REASONING;
        set(c_start, "<response>");
        set(c_end, "</response>");
        list_push(preserved, "<response>");
        list_push(preserved, "</response>");
    }
    // Cohere Command R+
    if has("<|CHATBOT_TOKEN|>") && has("<|END_OF_TURN_TOKEN|>") && c_start.len == 0 {
        c_mode = CONTENT_ALWAYS_WRAPPED;
        set(c_start, "<|CHATBOT_TOKEN|>");
        set(c_end, "<|END_OF_TURN_TOKEN|>");
        list_push(preserved, "<|CHATBOT_TOKEN|>");
        list_push(preserved, "<|END_OF_TURN_TOKEN|>");
        set(user_start, "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>");
    }
    // Functionary
    if has("set has_code_interpreter = tools | selectattr(\"type\", \"equalto\", \"code_interpreter\") | list | length > 0") {
        c_mode = CONTENT_PLAIN;
        c_end.clear();
        fn_name_prefix.clear();
        section_start.clear();
        section_end.clear();
        set(per_call_start, "<function=");
        set(per_call_end, "</function>");
        fn_close.clear();
        preserved.clear();
        list_push(preserved, "<|eot_id|>");
        list_push(preserved, "<|eom_id|>");
        list_push(preserved, "<function=");
        list_push(preserved, ">");
        list_push(preserved, "</function>");
    }
    // DeepSeek-R1-Distill-Qwen
    if has("{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>'") {
        set(section_start, "<｜tool▁calls▁begin｜>");
        set(section_end, "<｜tool▁calls▁end｜>");
        set(per_call_start, "<｜tool▁call▁begin｜>function");
        set(fn_name_prefix, "<｜tool▁sep｜>");
        set(per_call_end, "<｜tool▁call▁end｜>");
        set(fn_close, "```");
    }
    // Nemotron Nano v2
    if has("<SPECIAL_10>") && has("<SPECIAL_11>") && has("<SPECIAL_12>") && has("<TOOL_RESPONSE>") {
        f_mode = TOOLS_JSON_NATIVE;
        section_start.clear();
        section_end.clear();
        set(per_call_start, "<TOOLCALL>");
        set(per_call_end, "</TOOLCALL>");
        tools_array_wrapped = true;
        c_mode = CONTENT_PLAIN;
        c_start.clear();
        c_end.clear();
        r_mode = MODE_TAG_BASED;
        set(r_start, "<think>\n");
        set(r_end, "</think>");
        set(assistant_start, "<SPECIAL_11>Assistant");
        set(user_start, "<SPECIAL_11>User");
        preserved.clear();
        list_push(preserved, "<SPECIAL_11>");
        list_push(preserved, "</think>");
        list_push(preserved, "<TOOLCALL>");
        list_push(preserved, "</TOOLCALL>");
    }
    // Fireworks
    if has("{%- set system_prompt = '<|start_header_id|>' + 'system' + '<|end_header_id|>\\n\\n' + message['content'] | trim + '\\n' + system_prompt_suffix + '<|eot_id|>' -%}") {
        set(assistant_start, "<|start_header_id|>assistant<|end_header_id|>");
        set(user_start, "<|start_header_id|>user<|end_header_id|>");
    }
    // Solar Open
    if has("<|begin|>assistant<|think|><|end|>") { set(assistant_start, "<|begin|>assistant"); }
    // Apriel 1.6
    if has("if not loop.last and '[BEGIN FINAL RESPONSE]' in asst_text") {
        set(user_start, "<|begin_user|>");
        set(assistant_start, "<|begin_assistant|>");
    }
    // the JSON {name, parameters} tool instruction: the model writes the OpenAI wrapper
    if has("Respond in the format {\"name\": function name") && has("Do not use variables.") { openai_wrapper_trigger = true; }
    // Laguna
    if has("laguna_glm_thinking") {
        set(r_start, trim_whitespace(r_start));
        set(r_end, trim_whitespace(r_end));
        set(a_value_prefix, trim_whitespace(a_value_prefix));
        set(a_value_suffix, trim_whitespace(a_value_suffix));
        set(a_separator, trim_whitespace(a_separator));
        a_tolerate_ws = true;
        list_push(stops, "</assistant>");
    }
    // Bailing V3
    if has("Bailing V3 chat template") {
        set(a_value_suffix, trim_whitespace(a_value_suffix));
        a_tolerate_ws = true;
    }
}

// autoparser::analyze_template: the analysis of t into the module's state;
// false, and the error in err, if it fails.
var err: u8[>..] = [];

fn analyze_template(t: chat_render::Template) -> bool {
    tmpl = t;
    return analyze_tmpl();
}

fn analyze_tmpl() -> bool {
    clear_analysis();
    var ok = false;
    fn body() {
        reasoning_presence();
        thinking_enabled();
        if tmpl.caps.supports_tool_calls { reasoning_scope(); }
        analyze_content();
        if tmpl.caps.supports_tool_calls { analyze_tools(tmpl.caps); } else { clear_tools(); }
        detect_assistant_start();
        detect_user_start();
        collect_preserved_tokens();
        workarounds(chat_render::source(tmpl));
        ok = true;
    }
    exception::try<json_schema::Fail>(body) { f => err.clear(); err.append(json_schema::err); };
    return ok;
}

//
// the parser generator (chat-auto-parser-generator.cpp)
//
// The parsers are made in GCC's order: in a + b, b before a; a string
// operand's literal inside the operator, after both operands.

// generation_params, and its JSON read into json_schema values
var gen = chat_render::Gen { .. };
var tools_v = -1;
var schema_v = -1;
var extracting_reasoning = false;
var reasoning_parser = -1;

fn has_tools() -> bool { tools_v >= 0 && json_schema::json_kind(tools_v) == json_schema::J_ARRAY && json_schema::json_len(tools_v) > 0 }
fn has_response_format() -> bool { schema_v >= 0 && json_schema::json_kind(schema_v) == json_schema::J_OBJECT && json_schema::json_len(schema_v) > 0 }

// analyze_reasoning::build_parser
fn build_reasoning() -> i64 {
    if !extracting_reasoning { return peg::eps(); }
    if (r_mode == MODE_TAG_BASED || r_mode == MODE_TOOLS_ONLY) && r_end.len > 0 {
        if r_start.len > 0 {
            // optional(optspace(start) + reasoning(until(trim(end))) + optspace(end))
            let oe = chat_peg::optspace(r_end);
            let r = chat_peg::reasoning(peg::until(trim_whitespace(r_end)));
            let os = chat_peg::optspace(r_start);
            return peg::optional(peg::chain([peg::ci(os), peg::ci(r), peg::ci(oe)]));
        }
        let oe = chat_peg::optspace(r_end);
        let r = chat_peg::reasoning(peg::until(trim_whitespace(r_end)));
        return peg::optional(peg::seq2(r, oe));
    }
    return peg::eps();
}

// analyze_content::build_parser
fn build_content() -> i64 {
    if content_always_wrapped() {
        if extracting_reasoning {
            // R + start + content(until(end)) + end + p.end()
            let e = peg::end();
            let c = chat_peg::content(peg::until(c_end));
            return peg::chain([peg::ci(reasoning_parser), peg::cs(c_start), peg::ci(c), peg::cs(c_end), peg::ci(e)]);
        }
        // content(until(start)) + start + content(until(end)) + end + p.end()
        let e = peg::end();
        let c2 = chat_peg::content(peg::until(c_end));
        let c1 = chat_peg::content(peg::until(c_start));
        return peg::chain([peg::ci(c1), peg::cs(c_start), peg::ci(c2), peg::cs(c_end), peg::ci(e)]);
    }
    // R + content(rest()) + end()
    let e = peg::end();
    let c = chat_peg::content(peg::rest());
    return peg::chain([peg::ci(reasoning_parser), peg::ci(c), peg::ci(e)]);
}

// analyze_content::build_optional_wrapped
fn build_optional_wrapped() -> i64 {
    if content_always_wrapped() {
        // optional(start + content(until(end)) + end)
        let c = chat_peg::content(peg::until(c_end));
        return peg::optional(peg::chain([peg::cs(c_start), peg::ci(c), peg::cs(c_end)]));
    }
    return peg::eps();
}

fn order_list() -> const u8[:][>..] {
    var out: const u8[:][>..] = [];
    var start = 0;
    for i in parameter_order.len {
        if parameter_order[i] == 0 { out.push(parameter_order[start..i]); start = i + 1; }
    }
    return out;
}

// analyze_tools::build_tool_parser_json_native
fn build_tools_json_native() -> i64 {
    var nf = str(name_field);
    var af = str(args_field);
    if function_field.len > 0 && function_field != "function" && find(nf, ".") < 0 {
        let n2 = str(function_field, ".", nf);
        let a2 = str(function_field, ".", af);
        nf = copy(n2);
        af = copy(a2);
    }
    let order = order_list();
    let required = gen.tool_choice == chat_render::TOOL_CHOICE_REQUIRED;
    var tools_parser = peg::eps();
    if section_start.len == 0 && per_call_start.len > 0 {
        let single = chat_peg::standard_json_tools(per_call_start, per_call_end, tools_v, gen.parallel_tool_calls, required, nf, af,
            tools_array_wrapped, fun_name_is_key, id_field, gen_id_field, order, openai_wrapper_trigger);
        let s = peg::space();
        tools_parser = peg::trigger_rule("tool-calls", peg::one_or_more(peg::seq2(single, s)));
    } else {
        tools_parser = chat_peg::standard_json_tools(section_start, section_end, tools_v, gen.parallel_tool_calls, required, nf, af,
            tools_array_wrapped, fun_name_is_key, id_field, gen_id_field, order, openai_wrapper_trigger);
    }
    if content_always_wrapped() {
        let wrapped = build_optional_wrapped();
        let e = peg::end();
        return peg::chain([peg::ci(reasoning_parser), peg::ci(wrapped), peg::ci(tools_parser), peg::ci(e)]);
    }
    let tool_start = if section_start.len > 0 { str(section_start) } else if per_call_start.len > 0 { str(per_call_start) } else { str("{") };
    // R + optional(content(until(tool_start))) + tools_parser + end()
    let e = peg::end();
    let o = peg::optional(chat_peg::content(peg::until(tool_start)));
    return peg::chain([peg::ci(reasoning_parser), peg::ci(o), peg::ci(tools_parser), peg::ci(e)]);
}

// analyze_tools::build_func_parser (atomic_peek -1 for none)
fn build_func_parser(name: const u8[:], call_id_section: i64, have_call_id: bool, args: i64, atomic_peek: i64) -> i64 {
    // tool_open(name_prefix + tool_name(literal(name)) + name_suffix)
    let tn = chat_peg::tool_name(peg::literal(name));
    var open = chat_peg::tool_open(peg::chain([peg::cs(fn_name_prefix), peg::ci(tn), peg::cs(fn_name_suffix)]));
    var matched_atomic = false;
    var func = peg::eps();
    if fn_args_separator.len > 0 {
        let sep = peg::literal(fn_args_separator);
        let s = peg::space();
        open = peg::chain([peg::ci(open), peg::ci(s), peg::ci(sep)]);
    }
    if fn_name_suffix.len > 0 {
        let s = peg::space();
        func = peg::chain([peg::ci(open), peg::ci(call_id_section), peg::ci(s), peg::ci(args)]);
        matched_atomic = true;
    } else if have_call_id {
        // atomic(open + cid) + space() + args
        let s = peg::space();
        let a = peg::atomic(peg::seq2(open, call_id_section));
        func = peg::chain([peg::ci(a), peg::ci(s), peg::ci(args)]);
        matched_atomic = true;
    } else if atomic_peek >= 0 {
        // atomic(open + cid + space() + peek) + args
        let s = peg::space();
        let a = peg::atomic(peg::chain([peg::ci(open), peg::ci(call_id_section), peg::ci(s), peg::ci(atomic_peek)]));
        func = peg::seq2(a, args);
        matched_atomic = true;
    } else {
        let s = peg::space();
        func = peg::chain([peg::ci(open), peg::ci(call_id_section), peg::ci(s), peg::ci(args)]);
    }
    if fn_close.len > 0 {
        let tc = chat_peg::tool_close(peg::literal(fn_close));
        let s = peg::space();
        func = peg::chain([peg::ci(func), peg::ci(s), peg::ci(tc)]);
    } else if per_call_end.len > 0 {
        var close_peek = -1;
        if a_tolerate_ws {
            let l = peg::literal(per_call_end);
            let s = peg::space();
            close_peek = peg::peek(peg::seq2(s, l));
        } else {
            close_peek = peg::peek(peg::literal(per_call_end));
        }
        func = peg::seq2(func, chat_peg::tool_close(close_peek));
    } else {
        // forces the tool closing callbacks in the mapper
        func = peg::seq2(func, chat_peg::tool_close(peg::space()));
    }
    if !matched_atomic { func = peg::atomic(func); }
    return func;
}

// foreach_function's test: a function tool
fn is_function_tool(tool: i64) -> bool {
    let ty = chat_peg::jget(tool, "type");
    return ty >= 0 && chat_peg::jstr(ty) == "function" && json_schema::json_kind(ty) == json_schema::J_STRING && chat_peg::jget(tool, "function") >= 0;
}

// The call id parser between the function and its arguments; -1 for none
// (the eps it starts as is made by the caller).
fn call_id_parser(with_suffix_inside: bool) -> i64 {
    guard id_pos == ID_BETWEEN_FUNC_AND_ARGS && id_prefix.len > 0 && (id_suffix.len > 0 || a_start.len > 0) else { return -1; }
    if id_suffix.len > 0 {
        let ti = chat_peg::tool_id(peg::until(id_suffix));
        if with_suffix_inside {
            // optional(prefix + tool_id(until(suffix)) + suffix)
            return peg::optional(peg::chain([peg::cs(id_prefix), peg::ci(ti), peg::cs(id_suffix)]));
        }
        // optional(prefix + tool_id(until(suffix))) + suffix
        let o = peg::optional(peg::chain([peg::cs(id_prefix), peg::ci(ti)]));
        return peg::chain([peg::ci(o), peg::cs(id_suffix)]);
    }
    let ti = chat_peg::tool_id(peg::until(a_start));
    return peg::optional(peg::chain([peg::cs(id_prefix), peg::ci(ti)]));
}

// R + optional(content(until(trigger))) + tool_calls + end()
fn finish_tools(tool_calls: i64) -> i64 {
    let trigger = if section_start.len > 0 { str(section_start) } else { str(per_call_start) };
    let before = if trigger.len == 0 { peg::eps() } else { peg::until(trigger) };
    let e = peg::end();
    let o = peg::optional(chat_peg::content(before));
    return peg::chain([peg::ci(reasoning_parser), peg::ci(o), peg::ci(tool_calls), peg::ci(e)]);
}

// analyze_tools::build_tool_parser_tag_json
fn build_tools_tag_json() -> i64 {
    var tool_choice = peg::choice([]);
    for i in json_schema::json_len(tools_v) {
        let tool = json_schema::json_at(tools_v, i);
        guard is_function_tool(tool) else { continue; }
        let func = chat_peg::jget(tool, "function");
        let name = str(chat_peg::jstr(chat_peg::jget(func, "name")));
        var have_call_id = false;
        var call_id_section = peg::eps();
        let cid = call_id_parser(false);
        if cid >= 0 { call_id_section = cid; have_call_id = true; }
        let j = peg::json();
        let node = chat_peg::params_schema(func);
        var args_parser = chat_peg::tool_args(peg::schema(j, str("tool-", i, "-schema"), node, false));
        if a_start.len > 0 { let l = peg::literal(a_start); args_parser = peg::seq2(l, args_parser); }
        if a_end.len > 0 { let l = peg::literal(a_end); args_parser = peg::seq2(args_parser, l); }
        let atomic_peek = if a_start.len > 0 { peg::peek(peg::literal(a_start)) } else { -1 };
        let f = build_func_parser(name, call_id_section, have_call_id, args_parser, atomic_peek);
        let r = peg::rule(str("tool-", i), f, false);
        tool_choice = peg::choice2(tool_choice, r);
    }
    let require_calls = gen.tool_choice == chat_render::TOOL_CHOICE_REQUIRED;
    var tool_calls = peg::eps();
    if per_call_start.len > 0 {
        let wrapped = peg::chain([peg::cs(per_call_start), peg::ci(tool_choice), peg::cs(per_call_end)]);
        if gen.parallel_tool_calls {
            let s = peg::space();
            let z = peg::zero_or_more(peg::seq2(s, wrapped));
            tool_calls = peg::trigger_rule("tool-call", peg::seq2(wrapped, z));
        } else {
            tool_calls = peg::trigger_rule("tool-call", wrapped);
        }
        if section_start.len > 0 {
            let x = if section_end.len == 0 { peg::end() } else { peg::literal(section_end) };
            let s2 = peg::space();
            let s1 = peg::space();
            let l = peg::literal(section_start);
            tool_calls = peg::trigger_rule("tool-calls", peg::chain([peg::ci(l), peg::ci(s1), peg::ci(tool_calls), peg::ci(s2), peg::ci(x)]));
        }
    } else {
        let separator = ", ";
        if gen.parallel_tool_calls {
            let z = peg::zero_or_more(peg::chain([peg::cs(separator), peg::ci(tool_choice)]));
            tool_calls = peg::trigger_rule("tool-call", peg::chain([peg::cs(section_start), peg::ci(tool_choice), peg::ci(z), peg::cs(section_end)]));
        } else {
            tool_calls = peg::trigger_rule("tool-call", peg::chain([peg::cs(section_start), peg::ci(tool_choice), peg::cs(section_end)]));
        }
    }
    if !require_calls { tool_calls = peg::optional(tool_calls); }
    return finish_tools(tool_calls);
}

// analyze_tools::build_tool_parser_tag_tagged
fn build_tools_tag_tagged() -> i64 {
    let until_suffix = peg::rule("until-suffix", peg::until(a_value_suffix), false);
    var tool_choice = peg::choice([]);
    for i in json_schema::json_len(tools_v) {
        let tool = json_schema::json_at(tools_v, i);
        guard is_function_tool(tool) else { continue; }
        let func = chat_peg::jget(tool, "function");
        let name = str(chat_peg::jstr(chat_peg::jget(func, "name")));
        var required: i64[>..] = [];
        var optional: i64[>..] = [];
        let root = chat_peg::params_schema(func);
        for k in json_schema::prop_count(root) {
            let pname = str(json_schema::prop_name(root, k));
            let pnode = json_schema::prop_node(root, k);
            // the value, as a string or as JSON
            var x = -1;
            if json_schema::may_be_string(pnode) {
                let cl = chat_peg::tool_arg_close(peg::literal(a_value_suffix));
                let sv = chat_peg::tool_arg_string_value(until_suffix);
                x = peg::ac(peg::seq2(sv, cl), a_value_suffix);
            } else {
                let cl = chat_peg::tool_arg_close(peg::literal(a_value_suffix));
                let j = peg::json();
                let jvp = chat_peg::tool_arg_json_value(peg::schema(j, str("tool-", i, "-arg-", k, "-schema"), pnode, false));
                x = peg::seq2(jvp, cl);
            }
            // tool_arg_open(name_prefix + tool_arg_name(literal(name)) + name_suffix) + value_prefix + x
            let tan = chat_peg::tool_arg_name(peg::literal(pname));
            let open = chat_peg::tool_arg_open(peg::chain([peg::cs(a_name_prefix), peg::ci(tan), peg::cs(a_name_suffix)]));
            let arg = chat_peg::tool_arg(peg::chain([peg::ci(open), peg::cs(a_value_prefix), peg::ci(x)]));
            let named = peg::rule(str("tool-", i, "-arg-", k), arg, false);
            if json_schema::prop_required(root, k) { required.push(named); } else { optional.push(named); }
        }
        var args_seq = peg::eps();
        for r, k in required {
            if k > 0 { let s = peg::space(); args_seq = peg::seq2(args_seq, s); }
            args_seq = peg::seq2(args_seq, r);
        }
        if optional.len > 0 {
            var any_opt = peg::choice([]);
            for o in optional { any_opt = peg::choice2(any_opt, o); }
            let s = peg::space();
            let rep = peg::repeat(peg::seq2(s, any_opt), 0, -1);
            args_seq = peg::seq2(args_seq, rep);
        }
        if a_start.len > 0 { let l = peg::literal(a_start); args_seq = peg::seq2(l, args_seq); }
        if a_end.len > 0 { let l = peg::literal(a_end); args_seq = peg::seq2(args_seq, l); }
        var call_id_section = peg::eps();
        var have_call_id = false;
        let cid = call_id_parser(true);
        if cid >= 0 { call_id_section = cid; have_call_id = true; }
        let atomic_peek = if a_name_prefix.len > 0 && required.len > 0 { peg::peek(peg::literal(a_name_prefix)) } else { -1 };
        let f = build_func_parser(name, call_id_section, have_call_id, args_seq, atomic_peek);
        let r = peg::rule(str("tool-", i), f, false);
        tool_choice = peg::choice2(tool_choice, r);
    }
    let require_tools = gen.tool_choice == chat_render::TOOL_CHOICE_REQUIRED;
    var tool_calls = peg::eps();
    if per_call_start.len > 0 {
        // per_call_start + space() + tool_choice + space() + per_call_end
        let s2 = peg::space();
        let s1 = peg::space();
        let wrapped = peg::chain([peg::cs(per_call_start), peg::ci(s1), peg::ci(tool_choice), peg::ci(s2), peg::cs(per_call_end)]);
        if gen.parallel_tool_calls {
            let s = peg::space();
            let s3 = peg::space();
            let z = peg::zero_or_more(peg::seq2(s3, wrapped));
            tool_calls = peg::trigger_rule("tool-call", peg::chain([peg::ci(wrapped), peg::ci(z), peg::ci(s)]));
        } else {
            let s = peg::space();
            tool_calls = peg::trigger_rule("tool-call", peg::seq2(wrapped, s));
        }
        if section_start.len > 0 {
            var x = -1;
            if section_end.len == 0 {
                x = peg::end();
            } else {
                let s = peg::space();
                let l = peg::literal(section_end);
                x = peg::seq2(l, s);
            }
            let s4 = peg::space();
            let s3 = peg::space();
            let l = peg::literal(section_start);
            tool_calls = peg::trigger_rule("tool-calls", peg::chain([peg::ci(l), peg::ci(s3), peg::ci(tool_calls), peg::ci(s4), peg::ci(x)]));
        }
    } else {
        let separator = ", ";
        if gen.parallel_tool_calls {
            let s2 = peg::space();
            let z = peg::zero_or_more(peg::chain([peg::cs(separator), peg::ci(tool_choice)]));
            let s1 = peg::space();
            tool_calls = peg::trigger_rule("tool-call", peg::chain([peg::cs(section_start), peg::ci(s1), peg::ci(tool_choice), peg::ci(z), peg::ci(s2), peg::cs(section_end)]));
        } else {
            let s2 = peg::space();
            let s1 = peg::space();
            tool_calls = peg::trigger_rule("tool-call", peg::chain([peg::cs(section_start), peg::ci(s1), peg::ci(tool_choice), peg::ci(s2), peg::cs(section_end)]));
        }
    }
    if !require_tools { tool_calls = peg::optional(tool_calls); }
    return finish_tools(tool_calls);
}

// analyze_tools::build_parser
fn build_tools() -> i64 {
    if f_mode == TOOLS_JSON_NATIVE { return build_tools_json_native(); }
    if f_mode == TOOLS_TAG_WITH_JSON { return build_tools_tag_json(); }
    if f_mode == TOOLS_TAG_WITH_TAGGED { return build_tools_tag_tagged(); }
    write_stderr("[ERROR] Template seems to support tool calls, but failed to determine tool format. Tool calling will not work properly.\n");
    return peg::eps();
}

// autoparser::build_parser: the output parser, as an arena.
fn build_parser(generation_prompt: const u8[:]) -> i64 {
    peg::begin();
    let extract = gen.reasoning_format != chat_render::REASONING_NONE;
    extracting_reasoning = extract && r_mode != MODE_NONE;
    reasoning_parser = build_reasoning();
    var parser = peg::eps();
    var pure_content = r_mode == MODE_NONE;
    if has_response_format() {
        let j = peg::json();
        let node = json_schema::build_document(schema_v);
        let rf = peg::rule("response-format", chat_peg::content(peg::schema(j, "response-format-schema", node, false)), false);
        // R + space() + choice({ "```json" + space() + rf + space() + "```", space() + rf + space() }) + end()
        let e = peg::end();
        let a5 = peg::literal("```");
        let a4 = peg::space();
        let a2 = peg::space();
        let a1 = peg::literal("```json");
        let a = peg::chain([peg::ci(a1), peg::ci(a2), peg::ci(rf), peg::ci(a4), peg::ci(a5)]);
        let b3 = peg::space();
        let b1 = peg::space();
        let b = peg::chain([peg::ci(b1), peg::ci(rf), peg::ci(b3)]);
        let ch = peg::choice([a, b]);
        let s = peg::space();
        parser = peg::chain([peg::ci(reasoning_parser), peg::ci(s), peg::ci(ch), peg::ci(e)]);
        pure_content = false;
    } else if has_tools() && gen.tool_choice != chat_render::TOOL_CHOICE_NONE && tmpl.caps.supports_tool_calls {
        parser = build_tools();
        pure_content = false;
    } else {
        parser = build_content();
    }
    let rs = trim_whitespace(r_start);
    let root = if pure_content {
        let pfx = chat_peg::prefix(generation_prompt, rs);
        peg::seq2(pfx, parser)
    } else {
        let pfx = chat_peg::prefix(generation_prompt, rs);
        peg::spaced(pfx, parser)
    };
    return peg::build(root);
}

// peg_generator::generate_parser for template t, analyzed before (the
// analysis must be the module's state), with the inputs g: the prompt in
// chat_render's prompt, the rest in its params. It throws json_schema::Fail.
fn generate() {
    tools_v = if gen.tools.len > 0 { json_schema::json_value(gen.tools) } else { -1 };
    schema_v = if gen.json_schema.len > 0 { json_schema::json_value(gen.json_schema) } else { -1 };
    chat_render::clear_params();
    var gp: u8[>..] = [];
    fn render() {
        chat_render::generation_prompt(tmpl, gen, chat_render::Override { .. }, gp);
        chat_render::direct_apply(tmpl, gen, chat_render::Override { .. });
    }
    var rendered = false;
    fn try_render() { render(); rendered = true; }
    exception::try<jinja::Fail>(try_render);
    if !rendered { json_schema::fail(jinja::err); }
    chat_render::format = chat_render::FORMAT_PEG_NATIVE;
    chat_render::preserved_tokens.append(preserved);
    chat_render::additional_stops.append(stops);
    let parser_gp = str(gp);
    if chat_render::has_continuation(gen) {
        // the generation prompt made by hand
        if r_start.len > 0 {
            let cut = str(until_find(gp, r_start));
            gp = copy(cut);
            gp.append(r_start);
            gp.append(gen.continue_reasoning);
            if gen.continue_final_message == chat_render::CONTINUE_CONTENT { gp.append(r_end); }
        }
        if gen.continue_final_message == chat_render::CONTINUE_CONTENT { gp.append(gen.continue_content); }
        chat_render::append_part(gp, false);
    }
    chat_render::gen_prompt.append(gp);
    chat_render::parser = build_parser(parser_gp);
    let tools_ok = f_mode != TOOLS_NONE && has_tools();
    let trigger = if section_start.len > 0 { str(section_start) } else { str(per_call_start) };
    let rf = has_response_format();
    let include = rf || (tools_ok && ((gen.tool_choice == chat_render::TOOL_CHOICE_AUTO && trigger.len > 0) || gen.tool_choice == chat_render::TOOL_CHOICE_REQUIRED));
    if include {
        chat_render::grammar_lazy = !rf && gen.tool_choice == chat_render::TOOL_CHOICE_AUTO;
        json_schema::begin_grammar();
        peg::build_grammar(chat_render::parser, chat_render::grammar_lazy);
        json_schema::end_grammar(chat_render::grammar);
        if chat_render::grammar_lazy {
            chat_render::add_trigger(chat_render::TRIGGER_WORD, trigger);
            if openai_wrapper_trigger { chat_render::add_trigger(chat_render::TRIGGER_WORD, "{\"type\": \"function\","); }
        }
    }
    // the rest of templates_apply_jinja's autoparser branch
    chat_render::assistant_start.append(assistant_start);
    chat_render::user_start.append(user_start);
    chat_render::supports_thinking = r_mode != MODE_NONE;
    if chat_render::supports_thinking {
        chat_render::thinking_start.append(trim_whitespace(r_start));
        let end_tag = trim_whitespace(r_end);
        if end_tag.len > 0 { chat_render::list_push(chat_render::thinking_ends, end_tag); }
    }
}

// generate with its own handler: false, and the message in err, if it fails.
fn try_generate(t: chat_render::Template, g: chat_render::Gen) -> bool {
    tmpl = t;
    gen = g;
    return try_generate_current();
}

fn try_generate_current() -> bool {
    var ok = false;
    fn body() { generate(); ok = true; }
    exception::try<json_schema::Fail>(body) { f => err.clear(); err.append(json_schema::err); };
    return ok;
}
