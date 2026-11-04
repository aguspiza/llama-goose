// llama.cpp's chat PEG parsers: the builder's tags and the parsers of tool
// calls it makes from a tools array, the mapper that turns a parse's AST
// into a message (content, reasoning, tool calls), and the tagged parsers
// the template analysis uses.
// The parsers are made in the order GCC makes them in the C++ expressions
// (right operand first), which the GBNF names until-N and ac-N depend on.
// The tools are JSON values of json_schema's (json_schema::json_value).
// port-of: common/chat-peg-parser.cpp, common/chat-peg-parser.h
import std;
import exception;
import .json_schema;
import .peg;

namespace chat_peg;

let REASONING_BLOCK = "reasoning-block";
let REASONING = "reasoning";
let CONTENT = "content";
let TOOL = "tool";
let TOOL_OPEN = "tool-open";
let TOOL_CLOSE = "tool-close";
let TOOL_ID = "tool-id";
let TOOL_NAME = "tool-name";
let TOOL_ARGS = "tool-args";
let TOOL_ARG = "tool-arg";
let TOOL_ARG_OPEN = "tool-arg-open";
let TOOL_ARG_CLOSE = "tool-arg-close";
let TOOL_ARG_NAME = "tool-arg-name";
let TOOL_ARG_VALUE = "tool-arg-value";
let TOOL_ARG_STRING_VALUE = "tool-arg-string-value";

let MAX_PERMUTE = 6;

fn reasoning_block(p: i64) -> i64 { peg::tag(REASONING_BLOCK, p) }
fn reasoning(p: i64) -> i64 { peg::tag(REASONING, p) }
fn content(p: i64) -> i64 { peg::tag(CONTENT, p) }
fn tool(p: i64) -> i64 { peg::tag(TOOL, p) }
fn tool_open(p: i64) -> i64 { peg::atomic(peg::tag(TOOL_OPEN, p)) }
fn tool_close(p: i64) -> i64 { peg::atomic(peg::tag(TOOL_CLOSE, p)) }
fn tool_id(p: i64) -> i64 { peg::atomic(peg::tag(TOOL_ID, p)) }
fn tool_name(p: i64) -> i64 { peg::atomic(peg::tag(TOOL_NAME, p)) }
fn tool_args(p: i64) -> i64 { peg::tag(TOOL_ARGS, p) }
fn tool_arg(p: i64) -> i64 { peg::tag(TOOL_ARG, p) }
fn tool_arg_open(p: i64) -> i64 { peg::atomic(peg::tag(TOOL_ARG_OPEN, p)) }
fn tool_arg_close(p: i64) -> i64 { peg::atomic(peg::tag(TOOL_ARG_CLOSE, p)) }
fn tool_arg_name(p: i64) -> i64 { peg::atomic(peg::tag(TOOL_ARG_NAME, p)) }
fn tool_arg_value(p: i64) -> i64 { peg::tag(TOOL_ARG_VALUE, p) }
fn tool_arg_string_value(p: i64) -> i64 { peg::tag(TOOL_ARG_STRING_VALUE, p) }
fn tool_arg_json_value(p: i64) -> i64 { peg::tag(TOOL_ARG_VALUE, p) }

fn is_space(c: u8) -> bool { c == ' ' || (c >= 9 && c <= 13) }

// The parts of tag-wrapped content outside p: chunks that do not start
// with the marker.
fn tag_with_safe_content(tag_name: const u8[:], marker: const u8[:], p: i64) -> i64 {
    if marker.len == 0 {
        let chunk = peg::rule(tag_name, content(peg::any()), false);
        return peg::zero_or_more(peg::choice([p, chunk]));
    }
    // negate(literal(marker)) + any() + until(marker)
    let u = peg::until(marker);
    let a = peg::any();
    let n = peg::negate(peg::literal(marker));
    let chunk = peg::rule(tag_name, content(peg::seq2(peg::seq2(n, a), u)), false);
    return peg::zero_or_more(peg::choice([p, chunk]));
}

// Every parser once, in any order (permute).
fn permute(rule_prefix: const u8[:], parsers: const i64[:]) -> i64 {
    if parsers.len == 0 { return peg::eps(); }
    if parsers.len == 1 || parsers.len > MAX_PERMUTE { return peg::seq(parsers); }
    var cache: i64[>..] = [];
    for i in 1 << parsers.len { cache.push(-1); }
    return permute_rest(rule_prefix, parsers, (1 << parsers.len) - 1, cache);
}

recursive fn permute_rest(rule_prefix: const u8[:], parsers: const i64[:], remaining: i64, cache: i64[>..]&) -> i64 {
    if remaining == 0 { return peg::eps(); }
    if cache[remaining] >= 0 { return cache[remaining]; }
    var alternatives = peg::choice([]);
    for i in parsers.len {
        let bit = 1 << i;
        guard (remaining & bit) != 0 else { continue; }
        let rest = permute_rest(rule_prefix, parsers, remaining & ~bit, cache);
        let alt = peg::seq2(parsers[i], rest);
        alternatives = peg::choice2(alternatives, alt);
    }
    let r = peg::rule(str(rule_prefix, "-", remaining), alternatives, false);
    cache[remaining] = r;
    return r;
}

// The prefix of s up to the delimiter, as a literal (prefix).
fn prefix(s: const u8[:], delimiter: const u8[:]) -> i64 {
    if s.len == 0 { return peg::eps(); }
    if delimiter.len == 0 { return peg::literal(s); }
    let at = find(s, delimiter);
    return peg::literal(if at >= 0 { s[..at] } else { s });
}

// The tag, its leading and trailing spaces optional (optspace).
fn optspace(tag_text: const u8[:]) -> i64 {
    var parser = peg::eps();
    var end_of_prefix = tag_text.len;
    var start_of_suffix = tag_text.len;
    for i in tag_text.len { if !is_space(tag_text[i]) { end_of_prefix = i; break; } }
    var i = tag_text.len;
    while i > 0 { if !is_space(tag_text[i - 1]) { start_of_suffix = i; break; } i--; }
    for k in end_of_prefix {
        let o = peg::optional(peg::literal(tag_text[k..k + 1]));
        parser = peg::seq2(parser, o);
    }
    let mid = peg::literal(tag_text[end_of_prefix..max(end_of_prefix, start_of_suffix)]);
    parser = peg::seq2(parser, mid);
    for k in start_of_suffix..tag_text.len {
        let o = peg::optional(peg::literal(tag_text[k..k + 1]));
        parser = peg::seq2(parser, o);
    }
    return parser;
}

//
// tools
//

fn jget(v: i64, key: const u8[:]) -> i64 { if v >= 0 && json_schema::json_kind(v) == json_schema::J_OBJECT { json_schema::get_member(v, key) } else { -1 } }
fn jstr(v: i64) -> const u8[:] { if v >= 0 && json_schema::json_kind(v) == json_schema::J_STRING { json_schema::json_str(v) } else { "" } }

var default_params = -1;

// common_chat_tool_parameters: the parameters schema of a function tool,
// {"type": "object", "properties": {}} when it has none.
fn tool_parameters(function: i64) -> i64 {
    let params = jget(function, "parameters");
    if params >= 0 {
        let k = json_schema::json_kind(params);
        if k != json_schema::J_NULL && !(k == json_schema::J_OBJECT && json_schema::json_len(params) == 0) { return params; }
    }
    if default_params < 0 || default_params >= json_schema::jv.len {
        default_params = json_schema::json_value("{\"type\": \"object\", \"properties\": {}}");
    }
    return default_params;
}

// The schema document of a tool's parameters: its root node. It throws
// json_schema::Fail if the parameters are no schema.
fn params_schema(function) -> i64 { json_schema::build_document(tool_parameters(function)) }

// parse_key_spec: the parts of "a.b" (the prefix empty without a dot).
fn key_prefix(key: const u8[:]) -> const u8[:] { let d = find(key, "."); return if d < 0 { "" } else { key[..d] }; }
fn key_field(key: const u8[:]) -> const u8[:] { let d = find(key, "."); return if d < 0 { key } else { key[d + 1..] }; }

// literal("\"") + tool_id(string_content('"')) + literal("\"")
fn quoted_id() -> i64 {
    let q2 = peg::literal("\"");
    let id = tool_id(peg::string_content('"'));
    let q1 = peg::literal("\"");
    return peg::chain([peg::ci(q1), peg::ci(id), peg::ci(q2)]);
}

// atomic(literal("\"key\"") + space() + literal(":") + space() + literal("\"") + tool_id(string_content('"')) + literal("\""))
fn quoted_id_field(key: const u8[:]) -> i64 {
    let q2 = peg::literal("\"");
    let id = tool_id(peg::string_content('"'));
    let q1 = peg::literal("\"");
    let s2 = peg::space();
    let colon = peg::literal(":");
    let s1 = peg::space();
    let k = peg::literal(str("\"", key, "\""));
    return peg::atomic(peg::chain([peg::ci(k), peg::ci(s1), peg::ci(colon), peg::ci(s2), peg::ci(q1), peg::ci(id), peg::ci(q2)]));
}

// atomic(literal("\"key\"") + space() + ":" + space() + choice({ quoted id, tool_id(json_number()) }))
fn id_or_number_field(key: const u8[:]) -> i64 {
    let a = quoted_id();
    let b = tool_id(peg::json_number());
    let ch = peg::choice([a, b]);
    let s2 = peg::space();
    let colon = peg::literal(":");
    let s1 = peg::space();
    let k = peg::literal(str("\"", key, "\""));
    return peg::atomic(peg::chain([peg::ci(k), peg::ci(s1), peg::ci(colon), peg::ci(s2), peg::ci(ch)]));
}

// tool_args(schema(json(), "tool-<i>-schema", params))
fn tool_args_schema(i: i64, function: i64) -> i64 {
    let j = peg::json();
    let node = params_schema(function);
    return tool_args(peg::schema(j, str("tool-", i, "-schema"), node, false));
}

// A tool's name and whether it has a function (tools[i]).
fn tool_function(tools: i64, i: i64) -> i64 {
    let tool_def = json_schema::json_at(tools, i);
    return jget(tool_def, "function");
}

// Mode 1: function_is_key, {"function_name": {...}}.
fn json_tools_function_is_key(tools: i64, args_key: const u8[:], effective_args_key: const u8[:], call_id_key: const u8[:], gen_call_id_key: const u8[:]) -> i64 {
    var tool_choices = peg::choice([]);
    for i in json_schema::json_len(tools) {
        let function = tool_function(tools, i);
        guard function >= 0 else { continue; }
        let name = jstr(jget(function, "name"));
        var fields: i64[>..] = [];
        if call_id_key.len > 0 {
            let id_parser = quoted_id_field(call_id_key);
            fields.push(optional_field(id_parser));
        }
        if gen_call_id_key.len > 0 {
            let gen_id_parser = id_or_number_field(gen_call_id_key);
            fields.push(optional_field(gen_id_parser));
        }
        var args_parser = peg::eps();
        if args_key.len == 0 {
            args_parser = tool_args_schema(i, function);
        } else {
            let ta = tool_args_schema(i, function);
            let s2 = peg::space();
            let colon = peg::literal(":");
            let s1 = peg::space();
            let k = peg::literal(str("\"", effective_args_key, "\""));
            args_parser = peg::chain([peg::ci(k), peg::ci(s1), peg::ci(colon), peg::ci(s2), peg::ci(ta)]);
        }
        fields.push(args_parser);
        var inner = peg::eps();
        if args_key.len == 0 && fields.len == 1 {
            inner = fields[0];
        } else {
            let s = peg::space();
            let open = peg::literal("{");
            inner = peg::seq2(open, s);
            for f, k in fields {
                inner = peg::seq2(inner, f);
                if k < fields.len - 1 { let sp = peg::space(); inner = peg::seq2(inner, sp); }
            }
            let close = peg::literal("}");
            let s3 = peg::space();
            inner = peg::seq2(peg::seq2(inner, s3), close);
        }
        // tool_open(literal("{")) + space() + literal("\"") + tool_name(literal(name)) + literal("\"")
        //   + space() + literal(":") + space() + inner + space() + tool_close(literal("}"))
        let tc = tool_close(peg::literal("}"));
        let s4 = peg::space();
        let s3 = peg::space();
        let colon = peg::literal(":");
        let s2 = peg::space();
        let q2 = peg::literal("\"");
        let tn = tool_name(peg::literal(name));
        let q1 = peg::literal("\"");
        let s1 = peg::space();
        let to = tool_open(peg::literal("{"));
        let body = peg::chain([peg::ci(to), peg::ci(s1), peg::ci(q1), peg::ci(tn), peg::ci(q2), peg::ci(s2), peg::ci(colon), peg::ci(s3), peg::ci(inner), peg::ci(s4), peg::ci(tc)]);
        let r = peg::rule(str("tool-", i), tool(body), false);
        tool_choices = peg::choice2(tool_choices, r);
    }
    return tool_choices;
}

// optional(p + space() + optional(literal(",") + space()))
fn optional_field(p: i64) -> i64 {
    let s2 = peg::space();
    let comma = peg::literal(",");
    let oc = peg::optional(peg::seq2(comma, s2));
    let s1 = peg::space();
    return peg::optional(peg::chain([peg::ci(p), peg::ci(s1), peg::ci(oc)]));
}

// optional(p + space() + literal(",") + space())
fn optional_field_comma(p: i64) -> i64 {
    let s2 = peg::space();
    let comma = peg::literal(",");
    let s1 = peg::space();
    return peg::optional(peg::chain([peg::ci(p), peg::ci(s1), peg::ci(comma), peg::ci(s2)]));
}

// Mode 2: nested keys ("function.name").
fn json_tools_nested_keys(tools: i64, effective_name_key: const u8[:], effective_args_key: const u8[:], call_id_key: const u8[:], gen_call_id_key: const u8[:]) -> i64 {
    var tool_choices = peg::choice([]);
    let name_prefix = key_prefix(effective_name_key);
    let args_prefix = key_prefix(effective_args_key);
    let nested_prefix = if name_prefix.len > 0 { name_prefix } else { args_prefix };
    let nested_name_field = if name_prefix.len > 0 { key_field(effective_name_key) } else { effective_name_key };
    let nested_args_field = if args_prefix.len > 0 { key_field(effective_args_key) } else { effective_args_key };
    for i in json_schema::json_len(tools) {
        let function = tool_function(tools, i);
        guard function >= 0 else { continue; }
        let name = jstr(jget(function, "name"));
        // nested_name = "\"field\"" + space() + ":" + space() + atomic("\"" + tool_name(literal(name)) + "\"")
        var nested_name = -1;
        {
            let q2 = peg::literal("\"");
            let tn = tool_name(peg::literal(name));
            let q1 = peg::literal("\"");
            let at = peg::atomic(peg::chain([peg::ci(q1), peg::ci(tn), peg::ci(q2)]));
            let s2 = peg::space();
            let colon = peg::literal(":");
            let s1 = peg::space();
            let k = peg::literal(str("\"", nested_name_field, "\""));
            nested_name = peg::chain([peg::ci(k), peg::ci(s1), peg::ci(colon), peg::ci(s2), peg::ci(at)]);
        }
        var nested_args = -1;
        {
            let ta = tool_args_schema(i, function);
            let s2 = peg::space();
            let colon = peg::literal(":");
            let s1 = peg::space();
            let k = peg::literal(str("\"", nested_args_field, "\""));
            nested_args = peg::chain([peg::ci(k), peg::ci(s1), peg::ci(colon), peg::ci(s2), peg::ci(ta)]);
        }
        // literal("{") + space() + nested_name + space() + literal(",") + space() + nested_args + space() + literal("}")
        var nested_object = -1;
        {
            let close = peg::literal("}");
            let s4 = peg::space();
            let s3 = peg::space();
            let comma = peg::literal(",");
            let s2 = peg::space();
            let s1 = peg::space();
            let open = peg::literal("{");
            nested_object = peg::chain([peg::ci(open), peg::ci(s1), peg::ci(nested_name), peg::ci(s2), peg::ci(comma), peg::ci(s3), peg::ci(nested_args), peg::ci(s4), peg::ci(close)]);
        }
        let s0 = peg::space();
        let to = tool_open(peg::literal("{"));
        var body = peg::seq2(to, s0);
        if call_id_key.len > 0 && key_prefix(call_id_key).len == 0 {
            let id_parser = quoted_id_field(call_id_key);
            body = peg::seq2(body, optional_field_comma(id_parser));
        }
        if gen_call_id_key.len > 0 && key_prefix(gen_call_id_key).len == 0 {
            let gen_id_parser = id_or_number_field(gen_call_id_key);
            body = peg::seq2(body, optional_field_comma(gen_id_parser));
        }
        // "\"prefix\"" + space() + ":" + space() + nested_object
        var nested_field = -1;
        {
            let s2 = peg::space();
            let colon = peg::literal(":");
            let s1 = peg::space();
            let k = peg::literal(str("\"", nested_prefix, "\""));
            nested_field = peg::chain([peg::ci(k), peg::ci(s1), peg::ci(colon), peg::ci(s2), peg::ci(nested_object)]);
        }
        let tc = tool_close(peg::literal("}"));
        let s5 = peg::space();
        body = peg::chain([peg::ci(body), peg::ci(nested_field), peg::ci(s5), peg::ci(tc)]);
        let r = peg::rule(str("tool-", i), tool(body), false);
        tool_choices = peg::choice2(tool_choices, r);
    }
    return tool_choices;
}

// Mode 3: flat keys, the fields in parameters_order.
fn json_tools_flat_keys(tools: i64, effective_name_key: const u8[:], effective_args_key: const u8[:], call_id_key: const u8[:], gen_call_id_key: const u8[:], parameters_order: const u8[:][:], accept_openai_wrapper: bool) -> i64 {
    var tool_choices = peg::choice([]);
    let name_key_parser = peg::literal(str("\"", effective_name_key, "\""));
    let args_key_parser = peg::literal(str("\"", effective_args_key, "\""));
    for i in json_schema::json_len(tools) {
        let function = tool_function(tools, i);
        guard function >= 0 else { continue; }
        let name = jstr(jget(function, "name"));
        var tool_name_p = -1;
        {
            let q2 = peg::literal("\"");
            let tn = tool_name(peg::literal(name));
            let q1 = peg::literal("\"");
            let at = peg::atomic(peg::chain([peg::ci(q1), peg::ci(tn), peg::ci(q2)]));
            let s2 = peg::space();
            let colon = peg::literal(":");
            let s1 = peg::space();
            tool_name_p = peg::chain([peg::ci(name_key_parser), peg::ci(s1), peg::ci(colon), peg::ci(s2), peg::ci(at)]);
        }
        var tool_args_p = -1;
        {
            let ta = tool_args_schema(i, function);
            let s2 = peg::space();
            let colon = peg::literal(":");
            let s1 = peg::space();
            tool_args_p = peg::chain([peg::ci(args_key_parser), peg::ci(s1), peg::ci(colon), peg::ci(s2), peg::ci(ta)]);
        }
        var id_parser = peg::eps();
        if call_id_key.len > 0 { id_parser = id_or_number_field(call_id_key); }
        var gen_id_parser = peg::eps();
        if gen_call_id_key.len > 0 { gen_id_parser = id_or_number_field(gen_call_id_key); }
        // (parser, key) pairs, stably... std::sort, by the key's place in parameters_order
        var pp: i64[>..] = [];
        var pk: const u8[:][>..] = [];
        pp.push(tool_name_p);
        pk.push(effective_name_key);
        pp.push(tool_args_p);
        pk.push(effective_args_key);
        if call_id_key.len > 0 { pp.push(peg::optional(id_parser)); pk.push(call_id_key); }
        if gen_call_id_key.len > 0 { pp.push(peg::optional(gen_id_parser)); pk.push(gen_call_id_key); }
        fn order_of(key: const u8[:]) -> i64 {
            for o, k in parameters_order { if o == key { return k; } }
            return parameters_order.len;
        }
        // std::sort of at most four: insertion sort, as libstdc++'s for small ranges
        var idx: i64[>..] = [];
        for k in pp.len { idx.push(k); }
        for a in 1..idx.len {
            let x = idx[a];
            var b = a;
            while b > 0 && order_of(pk[x]) < order_of(pk[idx[b - 1]]) { idx[b] = idx[b - 1]; b--; }
            idx[b] = x;
        }
        var type_field = peg::eps();
        if accept_openai_wrapper {
            let s4 = peg::space();
            let comma = peg::literal(",");
            let s3 = peg::space();
            let fun = peg::literal("\"function\"");
            let s2 = peg::space();
            let colon = peg::literal(":");
            let s1 = peg::space();
            let ty = peg::literal("\"type\"");
            type_field = peg::optional(peg::chain([peg::ci(ty), peg::ci(s1), peg::ci(colon), peg::ci(s2), peg::ci(fun), peg::ci(s3), peg::ci(comma), peg::ci(s4)]));
        }
        let s0 = peg::space();
        let to = tool_open(peg::literal("{"));
        var body = peg::chain([peg::ci(to), peg::ci(s0), peg::ci(type_field)]);
        for x, k in idx {
            body = peg::seq2(body, pp[x]);
            if k < idx.len - 1 {
                let s2 = peg::space();
                let comma = peg::literal(",");
                let s1 = peg::space();
                body = peg::chain([peg::ci(body), peg::ci(s1), peg::ci(comma), peg::ci(s2)]);
            }
        }
        let tc = tool_close(peg::literal("}"));
        let s9 = peg::space();
        body = peg::chain([peg::ci(body), peg::ci(s9), peg::ci(tc)]);
        let r = peg::rule(str("tool-", i), tool(body), false);
        tool_choices = peg::choice2(tool_choices, r);
    }
    return tool_choices;
}

// standard_json_tools: the JSON tool calls between the section markers.
fn standard_json_tools(section_start: const u8[:], section_end: const u8[:], tools: i64, parallel_tool_calls: bool, force_tool_calls: bool,
                       name_key: const u8[:], args_key: const u8[:], array_wrapped: bool, function_is_key: bool,
                       call_id_key: const u8[:], gen_call_id_key: const u8[:], parameters_order: const u8[:][:], accept_openai_wrapper: bool) -> i64 {
    if tools < 0 || json_schema::json_kind(tools) != json_schema::J_ARRAY || json_schema::json_len(tools) == 0 { return peg::eps(); }
    let effective_name_key = if name_key.len == 0 { "name" } else { name_key };
    let effective_args_key = if args_key.len == 0 { "arguments" } else { args_key };
    var tool_choices = peg::eps();
    if function_is_key {
        tool_choices = json_tools_function_is_key(tools, args_key, effective_args_key, call_id_key, gen_call_id_key);
    } else if key_prefix(effective_name_key).len > 0 || key_prefix(effective_args_key).len > 0 {
        tool_choices = json_tools_nested_keys(tools, effective_name_key, effective_args_key, call_id_key, gen_call_id_key);
    } else {
        tool_choices = json_tools_flat_keys(tools, effective_name_key, effective_args_key, call_id_key, gen_call_id_key, parameters_order, accept_openai_wrapper);
    }
    var tool_calls = tool_choices;
    if parallel_tool_calls {
        let s2 = peg::space();
        let comma = peg::literal(",");
        let s1 = peg::space();
        let more = peg::zero_or_more(peg::chain([peg::ci(s1), peg::ci(comma), peg::ci(s2), peg::ci(tool_choices)]));
        tool_calls = peg::seq2(tool_calls, more);
    }
    if array_wrapped {
        let close = peg::literal("]");
        let s2 = peg::space();
        let s1 = peg::space();
        let open = peg::literal("[");
        tool_calls = peg::chain([peg::ci(open), peg::ci(s1), peg::ci(tool_calls), peg::ci(s2), peg::ci(close)]);
    }
    let le = peg::literal(section_end);
    let s2 = peg::space();
    let s1 = peg::space();
    let ls = peg::literal(section_start);
    let section = peg::trigger_rule("tool-call", peg::chain([peg::ci(ls), peg::ci(s1), peg::ci(tool_calls), peg::ci(s2), peg::ci(le)]));
    return if force_tool_calls { section } else { peg::optional(section) };
}

//
// the message a parse makes (common_chat_peg_mapper)
//

struct S { at: i64 = 0, len: i64 = 0 }
struct Call { name: S, arguments: S, id: S }

// the message: its texts in mt
var mt: u8[>..] = [];
var content_text: u8[>..] = [];
var reasoning_text: u8[>..] = [];
var calls: Call[>..] = [];

fn t(x: S) -> const u8[:] { mt[x.at..x.at + x.len] }
fn keep(b: const u8[:]) -> S { let at0 = mt.len; mt.append(b); return S { at: at0, len: b.len }; }

// the call in the making, before its name is known (pending), and the
// last call of the message, which may still grow
var pend_name: u8[>..] = [];
var pend_args: u8[>..] = [];
var pend_id: u8[>..] = [];
var has_pending = false;
var last_name: u8[>..] = [];
var last_args: u8[>..] = [];
var last_id: u8[>..] = [];
var has_last = false;
let CUR_NONE = 0;
let CUR_PENDING = 1;
let CUR_LAST = 2;
var cur = CUR_NONE;
var arg_count = 0;
var closing_quote_pending = false;
var args_buffer: u8[>..] = [];

fn mapper_reset() {
    mt.clear();
    content_text.clear();
    reasoning_text.clear();
    calls.clear();
    pend_name.clear();
    pend_args.clear();
    pend_id.clear();
    has_pending = false;
    last_name.clear();
    last_args.clear();
    last_id.clear();
    has_last = false;
    cur = CUR_NONE;
    arg_count = 0;
    closing_quote_pending = false;
    args_buffer.clear();
}

// the last call, its texts frozen into mt
fn freeze_last() {
    guard has_last else { return; }
    calls[calls.len - 1] = Call { name: keep(last_name), arguments: keep(last_args), id: keep(last_id) };
    has_last = false;
}

// result.tool_calls.push_back(pending)
fn push_pending() {
    freeze_last();
    calls.push(Call { name: S { .. }, arguments: S { .. }, id: S { .. } });
    last_name.clear();
    last_name.append(pend_name);
    last_args.clear();
    last_args.append(pend_args);
    last_id.clear();
    last_id.append(pend_id);
    has_last = true;
}

fn cur_name() -> const u8[:] { if cur == CUR_PENDING { pend_name[..] } else { last_name[..] } }

// args_target: the current tool's arguments once it has a name, else the buffer.
fn args_target_append(b: const u8[:]) {
    if cur != CUR_NONE && cur_name().len > 0 {
        if cur == CUR_PENDING { pend_args.append(b); } else { last_args.append(b); }
    } else {
        args_buffer.append(b);
    }
}

fn args_target_set(b: const u8[:]) {
    if cur != CUR_NONE && cur_name().len > 0 {
        if cur == CUR_PENDING { pend_args.clear(); pend_args.append(b); } else { last_args.clear(); last_args.append(b); }
    } else {
        args_buffer.clear();
        args_buffer.append(b);
    }
}

fn args_target_empty() -> bool {
    if cur != CUR_NONE && cur_name().len > 0 { return if cur == CUR_PENDING { pend_args.len == 0 } else { last_args.len == 0 }; }
    return args_buffer.len == 0;
}

fn trim_trailing(s: const u8[:]) -> const u8[:] { var e = s.len; while e > 0 && is_space(s[e - 1]) { e--; } return s[..e]; }
fn trim_one_leading(s: const u8[:]) -> const u8[:] { return if s.len > 0 && is_space(s[0]) { s[1..] } else { s }; }

// json_brace_depth: the '{' not closed, outside strings.
fn brace_depth(s: const u8[:]) -> i64 {
    var depth = 0;
    var in_string = false;
    var escaped = false;
    for c in s {
        if escaped { escaped = false; continue; }
        if c == '\\' && in_string { escaped = true; continue; }
        if c == '"' { in_string = !in_string; continue; }
        if !in_string { if c == '{' { depth++; } else if c == '}' { depth--; } }
    }
    return depth;
}

// escape_json_string_inner: s as a JSON string, without the quotes.
fn escape_inner(out: u8[>..]&, s: const u8[:]) {
    var tmp: u8[>..] = [];
    json_schema::json_string(tmp, s);
    out.append(tmp[1..tmp.len - 1]);
}

fn is_word_char(c: u8) -> bool { (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' }

// normalize_quotes_to_json: Python's quotes and literals as JSON's.
fn normalize_quotes(out: u8[>..]&, input: const u8[:]) {
    var in_single = false;
    var in_double = false;
    var i = 0;
    while i < input.len {
        let c = input[i];
        if c == '\\' && i + 1 < input.len {
            let next = input[i + 1];
            if in_single {
                if next == '\'' { out.push('\''); i += 2; continue; }
                if next == '"' { out.append("\\\""); i += 2; continue; }
                out.push(c);
                out.push(next);
                i += 2;
                continue;
            }
            if in_double { out.push(c); out.push(next); i += 2; continue; }
            out.push(c);
            i++;
            continue;
        }
        if c == '"' {
            if in_single { out.append("\\\""); } else { in_double = !in_double; out.push(c); }
        } else if c == '\'' {
            if in_double { out.push(c); }
            else if in_single { in_single = false; out.push('"'); }
            else { in_single = true; out.push('"'); }
        } else if !in_single && !in_double && (c == 'T' || c == 'F' || c == 'N') && (i == 0 || !is_word_char(input[i - 1])) {
            var n = 0;
            while i + n < input.len && is_word_char(input[i + n]) { n++; }
            let token = input[i..i + n];
            var matched = false;
            for k in 3 {
                let py = if k == 0 { "True" } else if k == 1 { "False" } else { "None" };
                let js = if k == 0 { "true" } else if k == 1 { "false" } else { "null" };
                guard !matched && n <= py.len && py[..n] == token else { continue; }
                out.append(js[..min(n, js.len)]);
                i += n - 1;
                matched = true;
            }
            if !matched { out.push(c); }
        } else {
            out.push(c);
        }
        i++;
    }
}

fn map_node(n: i64) {
    let tag_name = peg::node_tag(n);
    if tag_name == REASONING { peg::sanitized_text(n, reasoning_text); }
    if tag_name == CONTENT { peg::sanitized_text(n, content_text); }
    let text = peg::node_text(n);
    if tag_name == TOOL_OPEN {
        pend_name.clear();
        pend_args.clear();
        pend_id.clear();
        has_pending = true;
        cur = CUR_PENDING;
        arg_count = 0;
        args_buffer.clear();
        closing_quote_pending = false;
    }
    if tag_name == TOOL_ID && cur != CUR_NONE {
        var x = trim_trailing(text);
        if x.len >= 2 && x[0] == '"' && x[x.len - 1] == '"' { x = x[1..x.len - 1]; }
        let id = str(x);
        if cur == CUR_PENDING { pend_id.clear(); pend_id.append(id); } else { last_id.clear(); last_id.append(id); }
    }
    if tag_name == TOOL_NAME && cur != CUR_NONE {
        let nm = str(trim_trailing(text));
        if cur == CUR_PENDING { pend_name.clear(); pend_name.append(nm); } else { last_name.clear(); last_name.append(nm); }
        // the arguments so far are the tool's now
        if args_buffer.len > 0 {
            if cur == CUR_PENDING { pend_args.clear(); pend_args.append(args_buffer); } else { last_args.clear(); last_args.append(args_buffer); }
            args_buffer.clear();
        } else if (if cur == CUR_PENDING { pend_args.len } else { last_args.len }) == 0 {
            if cur == CUR_PENDING { pend_args.append("{"); } else { last_args.append("{"); }
        }
        if has_pending {
            push_pending();
            has_pending = false;
            cur = CUR_LAST;
        }
    }
    if tag_name == TOOL_ARGS && cur != CUR_NONE {
        let x = trim_trailing(text);
        if x.len > 0 && x[0] == '{' { args_target_set(x); }
    }
    if tag_name == TOOL_ARG_OPEN { closing_quote_pending = false; }
    if tag_name == TOOL_ARG_NAME && cur != CUR_NONE {
        var entry: u8[>..] = [];
        if arg_count > 0 { entry.push(','); }
        json_schema::json_string(entry, trim_trailing(trim_one_leading(text)));
        entry.push(':');
        arg_count++;
        if args_target_empty() { args_target_append("{"); }
        args_target_append(entry);
    }
    let is_value = tag_name == TOOL_ARG_VALUE;
    let is_string_value = tag_name == TOOL_ARG_STRING_VALUE;
    if (is_value || is_string_value) && cur != CUR_NONE {
        var add: u8[>..] = [];
        if text.len == 0 && is_string_value {
            add.push('"');
            closing_quote_pending = true;
        } else if text.len > 0 && is_string_value {
            if !closing_quote_pending { add.push('"'); closing_quote_pending = true; }
            escape_inner(add, text);
        } else if text.len > 0 {
            normalize_quotes(add, text);
        }
        args_target_append(add);
    }
    if tag_name == TOOL_ARG_CLOSE && cur != CUR_NONE {
        if closing_quote_pending { args_target_append("\""); closing_quote_pending = false; }
    }
    if tag_name == TOOL_CLOSE && cur != CUR_NONE {
        var name_empty = cur_name().len == 0;
        if name_empty && args_buffer.len > 0 {
            if cur == CUR_PENDING { pend_args.clear(); pend_args.append(args_buffer); } else { last_args.clear(); last_args.append(args_buffer); }
            args_buffer.clear();
        }
        if closing_quote_pending {
            if cur == CUR_PENDING { pend_args.push('"'); } else { last_args.push('"'); }
            closing_quote_pending = false;
        }
        var d = brace_depth(if cur == CUR_PENDING { pend_args[..] } else { last_args[..] });
        while d > 0 {
            if cur == CUR_PENDING { pend_args.push('}'); } else { last_args.push('}'); }
            d--;
        }
        if has_pending {
            if pend_name.len > 0 { push_pending(); }
            has_pending = false;
            cur = CUR_NONE;
        }
    }
}

// common_chat_peg_mapper::from_ast over the last parse.
fn map_parse() {
    mapper_reset();
    var order: i64[>..] = [];
    peg::visit_order(order);
    for n in order { map_node(n); }
    // a call started but never closed (a partial parse)
    if has_pending && pend_name.len > 0 {
        if args_buffer.len > 0 { pend_args.clear(); pend_args.append(args_buffer); }
        if closing_quote_pending && pend_args.len > 0 { pend_args.push('"'); }
        push_pending();
        has_pending = false;
    }
    freeze_last();
    // whitespace-only reasoning (an empty <think></think>) is none
    var all_space = true;
    for c in reasoning_text { if c != ' ' && c != '\n' && c != '\r' && c != '\t' { all_space = false; break; } }
    if all_space { reasoning_text.clear(); }
}

//
// tagged parsers (the template analysis)
//

// The tags of the last parse, the last node of each tag winning.
struct Tag { tag: S, text: S }
var tags: Tag[>..] = [];

fn collect_tags() {
    tags.clear();
    var order: i64[>..] = [];
    peg::visit_order(order);
    for n in order {
        let tg = peg::node_tag(n);
        guard tg.len > 0 else { continue; }
        let text = keep(peg::node_text(n));
        var found = false;
        for k in tags.len { if t(tags[k].tag) == tg { tags[k].text = text; found = true; } }
        if !found { tags.push(Tag { tag: keep(tg), text: text }); }
    }
}

fn tag_text(name: const u8[:]) -> const u8[:] {
    for g in tags { if t(g.tag) == name { return t(g.text); } }
    return "";
}

// parse_and_extract: whether arena a parses input; its tags in tags.
fn parse_and_extract(a: i64, input: const u8[:]) -> bool {
    mt.clear();
    let r, ok = peg::parse(a, input, 0, 0);
    collect_tags();
    return ok && r.kind == peg::SUCCESS;
}

// parse_anywhere_and_extract: the parse from the first place it succeeds
// (or the last place tried); its tags in tags.
fn parse_anywhere_and_extract(a: i64, input: const u8[:]) -> bool {
    if input.len == 0 { return parse_and_extract(a, input); }
    peg::input.clear();
    peg::input.append(input);
    for i in input.len {
        let r, ok = peg::parse_input(a, 0, i);
        if (ok && r.kind == peg::SUCCESS) || i == input.len - 1 {
            mt.clear();
            collect_tags();
            return ok && r.kind == peg::SUCCESS;
        }
    }
    return false;
}
