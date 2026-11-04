// llama.cpp's JSON schema to GBNF grammar: the schema parsed into nodes
// (the subset the converter handles), and the converter, which writes a
// grammar rule per node. The schema's JSON is read with the standard
// library's json, into a table of values that keeps the keys in order.
// Texts are ranges of one arena, `sb`, so that they can go through the
// recursion as plain values; a text is built at the arena's end, and
// nothing else may be added there until it is done.
// port-of: common/json-schema.cpp, common/json-schema-to-grammar.cpp, common/trie.cpp (the trie)
import std;
import exception;
import os;
import json;
import .number;

namespace json_schema;

let LIMIT = 128;     // COMMON_RECURSIVE_LIMIT

//
// texts
//

struct Str { at: i64 = 0, len: i64 = 0 }

var sb: u8[>..] = [];

fn t(x: Str) -> const u8[:] { sb[x.at..x.at + x.len] }

// A text of literal bytes (not from sb).
fn lit(b: const u8[:]) -> Str {
    let at0 = sb.len;
    sb.append(b);
    return Str { at: at0, len: b.len };
}

// The pieces of a text being built at the end of sb.
fn put(x: Str) { for i in x.len { sb.push(sb[x.at + i]); } }
fn put_b(b: const u8[:]) { sb.append(b); }
fn put_i(v: i64) { format(sb, v); }
fn fin(start: i64) -> Str { Str { at: start, len: sb.len - start } }

fn cat2(a: Str, b: const u8[:]) -> Str { let s0 = sb.len; put(a); put_b(b); return fin(s0); }
fn cat3(a: Str, b: const u8[:], c: Str) -> Str { let s0 = sb.len; put(a); put_b(b); put(c); return fin(s0); }
fn eq(a: Str, b: Str) -> bool { t(a) == t(b) }

fn cmp_bytes(x: const u8[:], y: const u8[:]) -> i64 {
    for i in min(x.len, y.len) { if x[i] != y[i] { return if x[i] < y[i] { -1 } else { 1 }; } }
    return if x.len < y.len { -1 } else if x.len > y.len { 1 } else { 0 };
}

//
// the schema's JSON
//

let J_NULL = 0;
let J_BOOL = 1;
let J_INT = 2;
let J_FLOAT = 3;
let J_STRING = 4;
let J_ARRAY = 5;
let J_OBJECT = 6;

// A JSON value: its kind, its number, its string (s), its key in its
// object, and its elements or members (jl[at..at + n]).
struct JV { kind: u8, i: i64 = 0, f: f64 = 0.0, s: Str = Str { .. }, key: Str = Str { .. }, at: i64 = 0, n: i64 = 0 }

var jv: JV[>..] = [];
var jl: i64[>..] = [];
var jstack: i64[>..<] = [];

recursive fn convert(n: json::Node&) -> i64 {
    var v = JV { kind: J_NULL, key: lit(n.key), .. };
    var container = false;
    match n.val {
        Null => {},
        Bool b => { v.kind = J_BOOL; v.i = if b.v { 1 } else { 0 }; },
        Int x => { v.kind = J_INT; v.i = x.v; v.f = x.v; },
        Num x => { v.kind = J_FLOAT; v.f = x.v; v.i = cast_i64(x.v); },
        Str x => { v.kind = J_STRING; v.s = lit(x.s); },
        Arr => { v.kind = J_ARRAY; container = true; },
        Obj => { v.kind = J_OBJECT; container = true; },
    }
    if container {
        let mark = jstack.len;
        var c = n.first;
        while c {
            let k = convert(c);
            jstack.push(k);
            c .= c.next;
        }
        v.at = jl.len;
        for k in mark..jstack.len { jl.push(jstack[k]); }
        v.n = jstack.len - mark;
        jstack.resize(mark);
    }
    jv.push(v);
    return jv.len - 1;
}

// The JSON text as values: the root, or -1 and the error in err.
fn parse_json(text: const u8[:]) -> i64 {
    jstack.clear();
    // the parse's own, as convert copies what it keeps
    var jtext: u8[>..] = [];
    var jdoc: json::Node[>..] = [];
    var jdecoded: u8[>..] = [];
    jtext.append(text);
    let root, e = json::parse(jtext, jdoc, jdecoded);
    guard root else {
        err.clear();
        format(err, "[json.exception.parse_error] ", e);
        return -1;
    }
    return convert(root);
}

fn kind(v: i64) -> u8 { jv[v].kind }
fn arr_len(v: i64) -> i64 { jv[v].n }
fn arr_at(v: i64, k: i64) -> i64 { jl[jv[v].at + k] }
fn obj_len(v: i64) -> i64 { jv[v].n }
fn obj_value(v: i64, k: i64) -> i64 { jl[jv[v].at + k] }
fn obj_key(v: i64, k: i64) -> Str { jv[jl[jv[v].at + k]].key }
fn sbytes(v: i64) -> const u8[:] { t(jv[v].s) }

// The member of object v named name (the first), or -1.
fn obj_get_name(v: i64, name: const u8[:]) -> i64 {
    for k in jv[v].n {
        let c = jl[jv[v].at + k];
        if t(jv[c].key) == name { return c; }
    }
    return -1;
}

// static_cast<int64_t>(v) as x86 does it.
fn cast_i64(v: f64) -> i64 {
    if !(v < 9223372036854775808.0 && v >= -9223372036854775808.0) { return -9223372036854775807 - 1; }
    return v as! i64;
}

// s as a JSON string, as nlohmann writes it.
fn json_string(out: u8[>..]&, s: const u8[:]) {
    out.push('"');
    for c in s {
        if c == '"' { out.append("\\\""); }
        else if c == '\\' { out.append("\\\\"); }
        else if c == 8 { out.append("\\b"); }
        else if c == 12 { out.append("\\f"); }
        else if c == '\n' { out.append("\\n"); }
        else if c == '\r' { out.append("\\r"); }
        else if c == '\t' { out.append("\\t"); }
        else if c < 0x20 {
            out.append("\\u00");
            out.push("0123456789abcdef"[c >> 4]);
            out.push("0123456789abcdef"[c & 15]);
        } else { out.push(c); }
    }
    out.push('"');
}

// The code point of the UTF-8 sequence at s[at], and its length; 0 if it
// is not a whole valid one.
fn utf8_decode(s: const u8[:], at: i64) -> i64, i64 {
    let c = s[at];
    if (c & 0x80) == 0 { return c, 1; }
    if (c & 0x40) == 0 { return 0, 0; }
    var n = 0;
    var cp = 0;
    if (c & 0x20) == 0 { n = 2; cp = c & 0x1F; }
    else if (c & 0x10) == 0 { n = 3; cp = c & 0x0F; }
    else if (c & 0x08) == 0 { n = 4; cp = c & 0x07; }
    else { return 0, 0; }
    for i in 1..n {
        if at + i >= s.len || (s[at + i] & 0xC0) != 0x80 { return 0, 0; }
        cp = (cp << 6) | (s[at + i] & 0x3F);
    }
    return cp, n;
}

//
// errors
//

let F_ERROR = 0;         // std::invalid_argument / std::runtime_error: the conversion fails
let F_UNSUPPORTED = 1;   // a pattern with no grammar equivalent
let F_INVALID = 2;       // a pattern that is not a valid regex

struct Fail { kind: i32 }
var err: u8[>..<] = [];

fn fail(msg: const u8[:]) {
    err.clear();
    err.append(msg);
    exception::throw(Fail { kind: F_ERROR });
}

fn fail_pattern(kind: i32, msg: const u8[:]) {
    err.clear();
    err.append(msg);
    exception::throw(Fail { kind: kind });
}

//
// the schema, parsed (common_chat_schema)
//

let K_ANY = 0;
let K_REF = 1;
let K_ANY_OF = 2;
let K_ALL_OF = 3;
let K_CONST = 4;
let K_ENUM = 5;
let K_NULL = 6;
let K_BOOLEAN = 7;
let K_NUMBER = 8;
let K_INTEGER = 9;
let K_STRING = 10;
let K_ARRAY = 11;
let K_TUPLE = 12;
let K_OBJECT = 13;

let FORMAT_NONE = 0;
let FORMAT_UUID = 1;
let FORMAT_DATE = 2;
let FORMAT_TIME = 3;
let FORMAT_DATE_TIME = 4;

let I64_MIN = -9223372036854775807 - 1;
let I64_MAX = 9223372036854775807;
let I32_MAX = 2147483647;

// A schema node. kids: children (any/allOf, tuple items: nodes; enum: JSON
// values), or properties (object). value: a const's JSON value. items: an
// array's item node. additional: an object's additional properties node, or
// -1. lo, hi: an integer's bounds, an array's or string's counts (hi -1:
// none). ref, target: a $ref and the node it names.
struct SNode {
    kind: u8,
    kids_at: i64 = 0,
    kids_n: i64 = 0,
    value: i64 = -1,
    items: i64 = -1,
    additional: i64 = -1,
    lo: i64 = 0,
    hi: i64 = -1,
    pattern: Str = Str { .. },
    has_pattern: bool = false,
    format: u8 = 0,
    ref: Str = Str { .. },
    target: i64 = -1,
}

struct Prop { name: Str, node: i64, required: bool }

var nodes: SNode[>..] = [];
var kids: i64[>..] = [];
var props: Prop[>..] = [];
// lists being built, each above its parent's
var kstack: i64[>..<] = [];
var pstack: Prop[>..<] = [];

// the $ref targets, by ref (node -1 while it is being built)
struct RefEntry { ref: Str, node: i64, doc: i64 }
var refs: RefEntry[>..] = [];
var root_json = -1;
var doc = 0;         // the document being built; its $refs are its own

fn new_node(n: SNode) -> i64 { nodes.push(n); return nodes.len - 1; }

fn take_kids(mark: i64) -> i64, i64 {
    let at0 = kids.len;
    for k in mark..kstack.len { kids.push(kstack[k]); }
    let n = kstack.len - mark;
    kstack.resize(mark);
    return at0, n;
}

fn sfail(path: Str, msg: const u8[:]) {
    fail(str("JSON schema error at ", t(path), ": ", msg));
}

fn get(v: i64, key: const u8[:]) -> i64 { if kind(v) == J_OBJECT { obj_get_name(v, key) } else { -1 } }
fn has(v: i64, key: const u8[:]) -> bool { get(v, key) >= 0 }
fn is_int(v: i64) -> bool { kind(v) == J_INT }
fn is_str(v: i64) -> bool { kind(v) == J_STRING }
fn is_bool(v: i64) -> bool { kind(v) == J_BOOL }
fn is_obj(v: i64) -> bool { kind(v) == J_OBJECT }
fn is_arr(v: i64) -> bool { kind(v) == J_ARRAY }

fn get_count(schema: i64, key: const u8[:], path: Str, def: i64) -> i64 {
    let v = get(schema, key);
    if v < 0 { return def; }
    if !is_int(v) || (jv[v].i as! i32) < 0 { sfail(path, str(key, " must be a non-negative integer")); }
    return jv[v].i as! i32;
}

// a fractional bound is rounded inwards, towards the integers it still admits
fn get_bound(schema: i64, key: const u8[:], path: Str, round_up: bool) -> i64 {
    let v = get(schema, key);
    if is_int(v) { return jv[v].i; }
    if kind(v) != J_FLOAT { sfail(path, str(key, " must be a number")); }
    let d = jv[v].f;
    let r = if round_up { ceil_f(d) } else { floor_f(d) };
    return cast_i64(r);
}

fn floor_f(d: f64) -> f64 { let i = cast_i64(d); let f: f64 = i; return if f > d { f - 1.0 } else { f }; }
fn ceil_f(d: f64) -> f64 { let i = cast_i64(d); let f: f64 = i; return if f < d { f + 1.0 } else { f }; }

fn get_format(schema: i64, path: Str) -> u8 {
    let v = get(schema, "format");
    if v < 0 { return FORMAT_NONE; }
    if !is_str(v) { sfail(path, "format must be a string"); }
    let f = sbytes(v);
    if f == "date" { return FORMAT_DATE; }
    if f == "time" { return FORMAT_TIME; }
    if f == "date-time" { return FORMAT_DATE_TIME; }
    if f == "uuid" || (f.len == 5 && starts_with(f, "uuid") && f[4] >= '1' && f[4] <= '5') { return FORMAT_UUID; }
    return FORMAT_NONE;
}

fn path_plus(path: Str, a: const u8[:], b: const u8[:]) -> Str { let s0 = sb.len; put(path); put_b(a); put_b(b); return fin(s0); }
fn path_index(path: Str, a: const u8[:], i: i64) -> Str { let s0 = sb.len; put(path); put_b(a); put_i(i); return fin(s0); }

// The JSON value the ref (#/a/b...) names in the document.
fn resolve_ref(ref: Str, path: Str) -> i64 {
    var target = root_json;
    let r = t(ref);
    // the selectors after "#/"
    var start = 2;
    var tokens_first = true;
    var i = 1;
    while i <= r.len {
        if i == r.len || r[i] == '/' {
            if tokens_first { tokens_first = false; i++; start = i; continue; }
            let sel = r[start..i];
            if is_obj(target) && obj_get_name(target, sel) >= 0 {
                target = obj_get_name(target, sel);
            } else if is_arr(target) {
                var idx: i64 = 0;
                var digits = 0;
                while digits < sel.len && sel[digits] >= '0' && sel[digits] <= '9' { idx = idx * 10 + (sel[digits] - '0'); digits++; }
                if digits == 0 { idx = arr_len(target); }
                if idx >= arr_len(target) { sfail(path, str("cannot resolve $ref ", r, ", ", sel, " is out of range")); }
                target = arr_at(target, idx);
            } else {
                sfail(path, str("cannot resolve $ref ", r, ", ", sel, " not found"));
            }
            start = i + 1;
        }
        i++;
    }
    return target;
}

fn find_ref(ref: const u8[:]) -> i64 {
    for e, k in refs { if e.doc == doc && t(e.ref) == ref { return k; } }
    return -1;
}

recursive fn build_ref(v: i64, path: Str) -> i64 {
    if !is_str(v) { sfail(path, "$ref must be a string"); }
    let ref = lit(sbytes(v));
    if !starts_with(t(ref), "#/") { sfail(path, str("unsupported $ref ", t(ref), ", only references into the same document are supported")); }
    if find_ref(t(ref)) < 0 {
        // the key first, so that a cycle back to this $ref stops here
        refs.push(RefEntry { ref: ref, node: -1, doc: doc });
        let k = refs.len - 1;
        let target = resolve_ref(ref, path);
        let n = build_node(target, ref);
        refs[k].node = n;
    }
    return new_node(SNode { kind: K_REF, ref: ref, .. });
}

recursive fn build_alternatives(kind: u8, alts: i64, path: Str) -> i64 {
    if !is_arr(alts) { sfail(path, "must be an array of schemas"); }
    if arr_len(alts) == 0 { sfail(path, "must not be empty"); }
    let mark = kstack.len;
    for i in arr_len(alts) {
        let child = build_node(arr_at(alts, i), path_index(path, "/", i));
        kstack.push(child);
    }
    let at0, n = take_kids(mark);
    return new_node(SNode { kind: kind, kids_at: at0, kids_n: n, .. });
}

fn in_required(schema: i64, name: const u8[:]) -> bool {
    let req = get(schema, "required");
    if req < 0 || !is_arr(req) { return false; }
    for i in arr_len(req) {
        let r = arr_at(req, i);
        if is_str(r) && sbytes(r) == name { return true; }
    }
    return false;
}

recursive fn build_object(schema: i64, path: Str) -> i64 {
    let pmark = pstack.len;
    let properties = get(schema, "properties");
    if properties >= 0 {
        if !is_obj(properties) { sfail(path, "properties must be an object"); }
        for k in obj_len(properties) {
            let name = obj_key(properties, k);
            let child = build_node(obj_value(properties, k), path_plus(path, "/properties/", t(name)));
            pstack.push(Prop { name: name, node: child, required: in_required(schema, t(name)) });
        }
    }
    var additional = -1;
    let ap = get(schema, "additionalProperties");
    if ap >= 0 {
        if is_bool(ap) {
            if jv[ap].i != 0 { additional = new_node(SNode { kind: K_ANY, .. }); }
        } else if is_obj(ap) {
            additional = build_node(ap, path_plus(path, "/additionalProperties", ""));
        } else {
            sfail(path, "additionalProperties must be a boolean or a schema");
        }
    } else if properties < 0 {
        // {"type": "object"} on its own accepts any object
        additional = new_node(SNode { kind: K_ANY, .. });
    }
    let at0 = props.len;
    for k in pmark..pstack.len { props.push(pstack[k]); }
    let n = pstack.len - pmark;
    pstack.resize(pmark);
    return new_node(SNode { kind: K_OBJECT, kids_at: at0, kids_n: n, additional: additional, .. });
}

recursive fn build_array(schema: i64, path: Str) -> i64 {
    var item = -1;
    if has(schema, "items") || has(schema, "prefixItems") {
        // "items" wins; a schema instead of an array is the item schema
        let key = if has(schema, "items") { "items" } else { "prefixItems" };
        let items = get(schema, key);
        if is_arr(items) {
            let mark = kstack.len;
            for i in arr_len(items) {
                let p = path_plus(path, "/", key);
                let child = build_node(arr_at(items, i), path_index(p, "/", i));
                kstack.push(child);
            }
            let at0, n = take_kids(mark);
            return new_node(SNode { kind: K_TUPLE, kids_at: at0, kids_n: n, .. });
        }
        item = build_node(items, path_plus(path, "/", key));
    } else {
        item = new_node(SNode { kind: K_ANY, .. });
    }
    let lo = get_count(schema, "minItems", path, 0);
    let hi = get_count(schema, "maxItems", path, -1);
    return new_node(SNode { kind: K_ARRAY, items: item, lo: lo, hi: hi, .. });
}

fn build_string(schema: i64, path: Str) -> i64 {
    var node = SNode { kind: K_STRING, .. };
    let p = get(schema, "pattern");
    if p >= 0 {
        if !is_str(p) { sfail(path, "pattern must be a string"); }
        node.pattern = lit(sbytes(p));
        node.has_pattern = true;
    }
    node.format = get_format(schema, path);
    node.lo = get_count(schema, "minLength", path, 0);
    node.hi = get_count(schema, "maxLength", path, -1);
    return new_node(node);
}

fn build_integer(schema: i64, path: Str) -> i64 {
    var lo = I64_MIN;
    var hi = I64_MAX;
    if has(schema, "minimum") { lo = get_bound(schema, "minimum", path, true); }
    else if has(schema, "exclusiveMinimum") { lo = get_bound(schema, "exclusiveMinimum", path, false) + 1; }
    if has(schema, "maximum") { hi = get_bound(schema, "maximum", path, false); }
    else if has(schema, "exclusiveMaximum") { hi = get_bound(schema, "exclusiveMaximum", path, true) - 1; }
    return new_node(SNode { kind: K_INTEGER, lo: lo, hi: hi, .. });
}

var depth = 0;

recursive fn build_node(schema: i64, path: Str) -> i64 { return build_node_typed(schema, path, -1); }

// build_node, with the schema's "type" replaced by type_v when that is >= 0
// (one alternative of a "type" array).
recursive fn build_node_typed(schema: i64, path: Str, type_v: i64) -> i64 {
    depth++;
    if depth > LIMIT { fail(str("JSON schema is nested more than ", LIMIT, " levels deep")); }
    let r = build_node_impl(schema, path, type_v);
    depth--;
    return r;
}

recursive fn build_node_impl(schema: i64, path: Str, type_override: i64) -> i64 {
    if !is_obj(schema) { sfail(path, "schema must be an object"); }
    if has(schema, "$ref") { return build_ref(get(schema, "$ref"), path); }
    if has(schema, "oneOf") || has(schema, "anyOf") {
        let key = if has(schema, "oneOf") { "oneOf" } else { "anyOf" };
        return build_alternatives(K_ANY_OF, get(schema, key), path_plus(path, "/", key));
    }
    let type_v = if type_override >= 0 { type_override } else { get(schema, "type") };
    if type_v >= 0 && is_arr(type_v) {
        // {"type": ["a", "b"], ...} is {"anyOf": [{"type": "a", ...}, {"type": "b", ...}]}
        if arr_len(type_v) == 0 { sfail(path, "type must not be empty"); }
        let mark = kstack.len;
        for i in arr_len(type_v) {
            let child = build_node_typed(schema, path_index(path, "/type/", i), arr_at(type_v, i));
            kstack.push(child);
        }
        let at0, n = take_kids(mark);
        return new_node(SNode { kind: K_ANY_OF, kids_at: at0, kids_n: n, .. });
    }
    if has(schema, "const") { return new_node(SNode { kind: K_CONST, value: get(schema, "const"), .. }); }
    if has(schema, "enum") {
        let values = get(schema, "enum");
        if !is_arr(values) || arr_len(values) == 0 { sfail(path, "enum must be a non-empty array"); }
        let mark = kstack.len;
        for i in arr_len(values) { kstack.push(arr_at(values, i)); }
        let at0, n = take_kids(mark);
        return new_node(SNode { kind: K_ENUM, kids_at: at0, kids_n: n, .. });
    }
    if type_v >= 0 && kind(type_v) != J_NULL && !is_str(type_v) {
        sfail(path, "type must be a string or an array of strings");
    }
    let tn = if type_v >= 0 && is_str(type_v) { lit(sbytes(type_v)) } else { lit("") };
    let ap = get(schema, "additionalProperties");
    let has_properties = has(schema, "properties") || (ap >= 0 && !(is_bool(ap) && jv[ap].i != 0));
    let type_name = t(tn);
    if type_name.len == 0 {
        // without a type the structural keywords decide
        if has_properties { return build_object(schema, path); }
        if has(schema, "allOf") { return build_alternatives(K_ALL_OF, get(schema, "allOf"), path_plus(path, "/allOf", "")); }
        if has(schema, "items") || has(schema, "prefixItems") { return build_array(schema, path); }
        if has(schema, "pattern") || has(schema, "minLength") || has(schema, "maxLength") || get_format(schema, path) != FORMAT_NONE {
            return build_string(schema, path);
        }
        return new_node(SNode { kind: K_ANY, .. });
    }
    if type_name == "object" {
        if !has_properties && has(schema, "allOf") { return build_alternatives(K_ALL_OF, get(schema, "allOf"), path_plus(path, "/allOf", "")); }
        return build_object(schema, path);
    }
    if type_name == "string" {
        if has(schema, "allOf") { return build_alternatives(K_ALL_OF, get(schema, "allOf"), path_plus(path, "/allOf", "")); }
        return build_string(schema, path);
    }
    if type_name == "array" { return build_array(schema, path); }
    if type_name == "integer" { return build_integer(schema, path); }
    if type_name == "number" { return new_node(SNode { kind: K_NUMBER, .. }); }
    if type_name == "boolean" { return new_node(SNode { kind: K_BOOLEAN, .. }); }
    if type_name == "null" { return new_node(SNode { kind: K_NULL, .. }); }
    sfail(path, str("unrecognized type ", type_name));
    return -1;
}

// common_chat_schema_from_json: the root node of the JSON schema v.
// The nodes are added to those of the documents before.
fn build_document(v: i64) -> i64 {
    kstack.clear();
    pstack.clear();
    depth = 0;
    doc++;
    root_json = v;
    let first = nodes.len;
    let root = build_node(v, lit("#"));
    // the $refs get their targets once all are built
    for k in first..nodes.len {
        guard nodes[k].kind == K_REF else { continue; }
        let e = find_ref(t(nodes[k].ref));
        nodes[k].target = refs[e].node;
    }
    return root;
}

//
// the converter
//

struct Rule { name: Str, rule: Str }

var rules: Rule[>..] = [];
var dotall = false;
var errors: Str[>..] = [];
var warnings: Str[>..] = [];
var being_resolved: Str[>..] = [];
var vdepth = 0;

fn rule_index(name: const u8[:]) -> i64 {
    for r, k in rules { if t(r.name) == name { return k; } }
    return -1;
}

fn set_rule(name: Str, rule: Str) {
    let k = rule_index(t(name));
    if k >= 0 { rules[k].rule = rule; } else { rules.push(Rule { name: name, rule: rule }); }
}

// gbnf_format_rule_name: what is not [a-zA-Z0-9-] becomes one '-' per run.
fn format_rule_name(name: Str) -> Str {
    let s0 = sb.len;
    var in_invalid = false;
    for i in name.len {
        let c = sb[name.at + i];
        if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' {
            sb.push(c);
            in_invalid = false;
        } else if !in_invalid {
            sb.push('-');
            in_invalid = true;
        }
    }
    return fin(s0);
}

// A rule named name (the name made valid), or name<i> when name has
// another rule: the name it got.
fn add_rule(name: Str, rule: Str) -> Str {
    let esc = format_rule_name(name);
    let k = rule_index(t(esc));
    if k < 0 || eq(rules[k].rule, rule) {
        set_rule(esc, rule);
        return esc;
    }
    var i = 0;
    loop {
        let s0 = sb.len;
        put(esc);
        put_i(i);
        let key = fin(s0);
        let j = rule_index(t(key));
        if j < 0 || eq(rules[j].rule, rule) {
            set_rule(key, rule);
            return key;
        }
        i++;
    }
}

fn add_rule_b(name: const u8[:], rule: Str) -> Str { add_rule(lit(name), rule) }

let SPACE_RULE = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";

// The primitive rules (and the string formats'), their text and the rules
// they need.
struct Prim { name: const u8[:], content: const u8[:], deps: const u8[:] }

let PRIMITIVES: Prim[18] = [
    Prim { "boolean", "(\"true\" | \"false\")", "" },
    Prim { "decimal-part", "[0-9]{1,16}", "" },
    Prim { "integral-part", "[0] | [1-9] [0-9]{0,15}", "" },
    Prim { "number", "(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)?", "integral-part decimal-part" },
    Prim { "integer", "(\"-\"? integral-part)", "integral-part" },
    Prim { "value", "object | array | string | number | boolean | null", "object array string number boolean null" },
    Prim { "object", "\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? space \"}\"", "string value" },
    Prim { "array", "\"[\" space ( value (\",\" space value)* )? space \"]\"", "value" },
    Prim { "uuid", "\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"\"", "" },
    Prim { "char", "[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})", "" },
    Prim { "string", "\"\\\"\" char* \"\\\"\"", "char" },
    Prim { "null", "\"null\"", "" },
    Prim { "date", "[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )", "" },
    Prim { "time", "([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" ) ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )", "" },
    Prim { "date-time", "date \"T\" time", "date time" },
    Prim { "date-string", "\"\\\"\" date \"\\\"\"", "date" },
    Prim { "time-string", "\"\\\"\" time \"\\\"\"", "time" },
    Prim { "date-time-string", "\"\\\"\" date-time \"\\\"\"", "date-time" },
];

fn primitive(name: const u8[:]) -> const u8[:], const u8[:] {
    for p in PRIMITIVES { if p.name == name { return p.content, p.deps; } }
    return "", "";
}

fn is_primitive(name: const u8[:]) -> bool {
    let content, deps = primitive(name);
    return content.len > 0;
}

fn is_reserved_name(name: const u8[:]) -> bool { name == "root" || is_primitive(name) }

// The primitive rule name as rule_name, and the rules it needs.
recursive fn add_primitive(rule_name: Str, name: const u8[:]) -> Str {
    let content, deps = primitive(name);
    let n = add_rule(rule_name, lit(content));
    var start = 0;
    for c, i in deps {
        guard c == ' ' || i == deps.len - 1 else { continue; }
        let end = if c == ' ' { i } else { i + 1 };
        let dep = deps[start..end];
        start = i + 1;
        if !is_primitive(dep) {
            errors.push(lit(str("Rule ", dep, " not known")));
            continue;
        }
        if rule_index(dep) < 0 { add_primitive(lit(dep), dep); }
    }
    return n;
}

fn add_primitive_b(name: const u8[:]) -> Str { add_primitive(lit(name), name) }

// format_literal: a GBNF string literal of the bytes.
fn format_literal(x: Str) -> Str {
    let s0 = sb.len;
    sb.push('"');
    for i in x.len {
        let c = sb[x.at + i];
        if c == '\r' { put_b("\\r"); }
        else if c == '\n' { put_b("\\n"); }
        else if c == '"' { put_b("\\\""); }
        else if c == '\\' { put_b("\\\\"); }
        else { sb.push(c); }
    }
    sb.push('"');
    return fin(s0);
}

// value.dump() as nlohmann writes it: compact.
fn dump(v: i64) -> Str {
    var out: u8[>..] = [];
    dump_into(out, v);
    return lit(out);
}

recursive fn dump_into(out: u8[>..]&, v: i64) {
    let x = jv[v];
    let k = x.kind;
    if k == J_NULL { out.append("null"); return; }
    if k == J_BOOL { out.append(if x.i != 0 { "true" } else { "false" }); return; }
    if k == J_INT { format(out, x.i); return; }
    if k == J_FLOAT { number::float_json(out, x.f); return; }
    if k == J_STRING { json_string(out, sbytes(v)); return; }
    if k == J_ARRAY {
        out.push('[');
        for i in x.n {
            if i > 0 { out.push(','); }
            dump_into(out, arr_at(v, i));
        }
        out.push(']');
        return;
    }
    out.push('{');
    for i in x.n {
        if i > 0 { out.push(','); }
        json_string(out, t(obj_key(v, i)));
        out.push(':');
        dump_into(out, obj_value(v, i));
    }
    out.push('}');
}

fn constant_rule(v: i64) -> Str { format_literal(dump(v)) }

// build_repetition: item repeated min..max times (max I32_MAX: unbounded),
// separated by sep when there is one.
recursive fn build_repetition(item: Str, lo: i64, hi: i64, sep: Str) -> Str {
    let has_max = hi != I32_MAX;
    if hi == 0 { return lit(""); }
    if lo == 0 && hi == 1 { return cat2(item, "?"); }
    if sep.len == 0 {
        if lo == 1 && !has_max { return cat2(item, "+"); }
        if lo == 0 && !has_max { return cat2(item, "*"); }
        let s0 = sb.len;
        put(item);
        put_b("{");
        put_i(lo);
        put_b(",");
        if has_max { put_i(hi); }
        put_b("}");
        return fin(s0);
    }
    let s0 = sb.len;
    put_b("(");
    put(sep);
    put_b(" ");
    put(item);
    put_b(")");
    let inner_item = fin(s0);
    let inner = build_repetition(inner_item, if lo == 0 { 0 } else { lo - 1 }, if has_max { hi - 1 } else { hi }, lit(""));
    let s1 = sb.len;
    if lo == 0 { put_b("("); }
    put(item);
    put_b(" ");
    put(inner);
    if lo == 0 { put_b(")?"); }
    return fin(s1);
}

// Number texts of build_min_max_int, outside sb (which holds the output).
var nb: u8[>..] = [];

fn nstr(v: i64) -> const u8[:] { let at0 = nb.len; format(nb, v); return nb[at0..nb.len]; }
fn nrepeat(c: u8, n: i64) -> const u8[:] { let at0 = nb.len; for i in n { nb.push(c); } return nb[at0..nb.len]; }

fn digit_range(from: u8, to: u8) {
    sb.push('[');
    if from == to { sb.push(from); } else { sb.push(from); sb.push('-'); sb.push(to); }
    sb.push(']');
}

fn more_digits(min_digits: i64, max_digits: i64) {
    put_b("[0-9]");
    if min_digits == max_digits && min_digits == 1 { return; }
    put_b("{");
    put_i(min_digits);
    if max_digits != min_digits {
        put_b(",");
        if max_digits != I32_MAX { put_i(max_digits); }
    }
    put_b("}");
}

fn all_of_char(s: const u8[:], c: u8) -> bool { for x in s { if x != c { return false; } } return true; }

recursive fn uniform_range(from: const u8[:], to: const u8[:]) {
    var i = 0;
    while i < from.len && i < to.len && from[i] == to[i] { i++; }
    if i > 0 { sb.push('"'); put_b(from[..i]); sb.push('"'); }
    if i < from.len && i < to.len {
        if i > 0 { sb.push(' '); }
        let sub_len = from.len - i - 1;
        if sub_len > 0 {
            let from_sub = from[i + 1..];
            let to_sub = to[i + 1..];
            var to_reached = false;
            sb.push('(');
            if all_of_char(from_sub, '0') {
                digit_range(from[i], to[i] - 1);
                sb.push(' ');
                more_digits(sub_len, sub_len);
            } else {
                put_b("[");
                sb.push(from[i]);
                put_b("] (");
                uniform_range(from_sub, nrepeat('9', sub_len));
                put_b(")");
                if from[i] < to[i] - 1 {
                    put_b(" | ");
                    if all_of_char(to_sub, '9') {
                        digit_range(from[i] + 1, to[i]);
                        to_reached = true;
                    } else {
                        digit_range(from[i] + 1, to[i] - 1);
                    }
                    sb.push(' ');
                    more_digits(sub_len, sub_len);
                }
            }
            if !to_reached {
                put_b(" | ");
                digit_range(to[i], to[i]);
                sb.push(' ');
                uniform_range(nrepeat('0', sub_len), to_sub);
            }
            sb.push(')');
        } else {
            sb.push('[');
            sb.push(from[i]);
            sb.push('-');
            sb.push(to[i]);
            sb.push(']');
        }
    }
}

// The integers min..max as a grammar, onto sb (build_min_max_int).
recursive fn min_max_int(lo: i64, hi: i64, decimals_left: i64, top_level: bool) {
    let has_min = lo != I64_MIN;
    let has_max = hi != I64_MAX;
    if has_min && has_max {
        if lo < 0 && hi < 0 {
            put_b("\"-\" (");
            min_max_int(-hi, -lo, decimals_left, true);
            put_b(")");
            return;
        }
        var min_v = lo;
        if lo < 0 {
            put_b("\"-\" (");
            min_max_int(0, -lo, decimals_left, true);
            put_b(") | ");
            min_v = 0;
        }
        var min_s = nstr(min_v);
        let max_s = nstr(hi);
        for digits in min_s.len..max_s.len {
            uniform_range(min_s, nrepeat('9', digits));
            let at0 = nb.len;
            nb.push('1');
            for k in digits { nb.push('0'); }
            min_s = nb[at0..nb.len];
            put_b(" | ");
        }
        uniform_range(min_s, max_s);
        return;
    }
    let less_decimals = max(decimals_left - 1, 1);
    if has_min {
        if lo < 0 {
            put_b("\"-\" (");
            min_max_int(I64_MIN, -lo, decimals_left, false);
            put_b(") | [0] | [1-9] ");
            more_digits(0, decimals_left - 1);
        } else if lo == 0 {
            if top_level {
                put_b("[0] | [1-9] ");
                more_digits(0, less_decimals);
            } else {
                more_digits(1, decimals_left);
            }
        } else if lo <= 9 {
            let c: u8 = ('0' + lo) as! u8;
            let range_start: u8 = if top_level { '1' } else { '0' };
            if c > range_start {
                digit_range(range_start, c - 1);
                sb.push(' ');
                more_digits(1, less_decimals);
                put_b(" | ");
            }
            digit_range(c, '9');
            sb.push(' ');
            more_digits(0, less_decimals);
        } else {
            let min_s = nstr(lo);
            let len = min_s.len;
            let c = min_s[0];
            if c > '1' {
                digit_range(if top_level { '1' } else { '0' }, c - 1);
                sb.push(' ');
                more_digits(len, less_decimals);
                put_b(" | ");
            }
            digit_range(c, c);
            put_b(" (");
            let rest, ok = parse_int(min_s[1..]);
            min_max_int(rest, I64_MAX, less_decimals, false);
            put_b(")");
            if c < '9' {
                put_b(" | ");
                digit_range(c + 1, '9');
                sb.push(' ');
                more_digits(len - 1, less_decimals);
            }
        }
        return;
    }
    if has_max {
        if hi >= 0 {
            if top_level {
                put_b("\"-\" [1-9] ");
                more_digits(0, less_decimals);
                put_b(" | ");
            }
            min_max_int(0, hi, decimals_left, true);
        } else {
            put_b("\"-\" (");
            min_max_int(-hi, I64_MAX, decimals_left, false);
            put_b(")");
        }
        return;
    }
    fail("At least one of min_value or max_value must be set");
}

//
// patterns
//

// gbnf_escape_length: the length of the GBNF escape at s[pos], 0 if none.
fn escape_length(s: const u8[:], pos: i64) -> i64 {
    if pos + 1 >= s.len || s[pos] != '\\' { return 0; }
    var n_hex = 0;
    let e = s[pos + 1];
    if e == 'x' { n_hex = 2; }
    else if e == 'u' { n_hex = 4; }
    else if e == 'U' { n_hex = 8; }
    else if e == 't' || e == 'r' || e == 'n' || e == '\\' || e == '"' || e == '[' || e == ']' || e == '-' { return 2; }
    else { return 0; }
    if pos + 2 + n_hex > s.len { return 0; }
    for i in pos + 2..pos + 2 + n_hex {
        let h = s[i];
        if !((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F')) { return 0; }
    }
    return 2 + n_hex;
}

fn non_literal(c: u8) -> bool {
    return c == '|' || c == '.' || c == '(' || c == ')' || c == '[' || c == ']' || c == '{' || c == '}' || c == '*' || c == '+' || c == '?' || c == '^' || c == '$';
}

fn escaped_in_regex_only(c: u8) -> bool {
    return c == '^' || c == '$' || c == '.' || c == '[' || c == ']' || c == '(' || c == ')' || c == '|' || c == '{' || c == '}' || c == '*' || c == '+' || c == '?';
}

// The pattern being converted, its position, and the open groups.
struct Item { s: Str, literal: bool }
var seq: Item[>..<] = [];
var pat = Str { .. };
var pi = 0;
var paren_depth = 0;
var pname = Str { .. };
// the repeated sub-rules of the pattern, by their text
struct SubRule { sub: Str, id: Str }
var sub_rules: SubRule[>..] = [];

fn pc(i: i64) -> u8 { sb[pat.at + i] }

fn to_rule(x: Item) -> Str {
    if !x.literal { return x.s; }
    let s0 = sb.len;
    sb.push('"');
    put(x.s);
    sb.push('"');
    return fin(s0);
}

fn get_dot() -> Str {
    return add_rule_b("dot", lit(if dotall { "[\\U00000000-\\U0010FFFF]" } else { "[^\\x0A\\x0D]" }));
}

// The sequence since mark joined, consecutive literals merged.
fn join_seq(mark: i64) -> Item {
    // the merged items, above the sequence
    let base = seq.len;
    var k = mark;
    while k < base {
        if seq[k].literal {
            let s0 = sb.len;
            while k < base && seq[k].literal { put(seq[k].s); k++; }
            seq.push(Item { s: fin(s0), literal: true });
        } else {
            seq.push(seq[k]);
            k++;
        }
    }
    // their rules
    let rmark = seq.len;
    for j in base..rmark { seq.push(Item { s: to_rule(seq[j]), literal: false }); }
    let s0 = sb.len;
    for j in rmark..seq.len {
        if j > rmark { sb.push(' '); }
        put(seq[j].s);
    }
    let r = fin(s0);
    seq.resize(mark);
    return Item { s: r, literal: false };
}

fn plen() -> i64 { pat.len }

recursive fn transform() -> Item {
    let mark = seq.len;
    let length = plen();
    while pi < length {
        let c = pc(pi);
        if c == '.' {
            seq.push(Item { s: get_dot(), literal: false });
            pi++;
        } else if c == '(' {
            pi++;
            if pi < length && pc(pi) == '?' {
                if pi + 1 < length && pc(pi + 1) == ':' { pi += 2; }
                else { fail_pattern(F_UNSUPPORTED, "unsupported group syntax"); }
            }
            paren_depth++;
            if paren_depth > LIMIT { fail_pattern(F_UNSUPPORTED, "pattern nesting too deep"); }
            let inner = transform();
            let r = to_rule(inner);
            let s0 = sb.len;
            sb.push('(');
            put(r);
            sb.push(')');
            seq.push(Item { s: fin(s0), literal: false });
        } else if c == ')' {
            pi++;
            if paren_depth == 0 { fail_pattern(F_INVALID, "unbalanced parentheses"); }
            paren_depth--;
            return join_seq(mark);
        } else if c == '^' || c == '$' {
            fail_pattern(F_UNSUPPORTED, "anchor inside the pattern");
        } else if c == '[' {
            let s0 = sb.len;
            sb.push('[');
            pi++;
            while pi < length && pc(pi) != ']' {
                if pc(pi) == '\\' {
                    let n = escape_length(t(pat), pi);
                    if n == 0 { fail_pattern(F_UNSUPPORTED, str("unsupported escape in character class: ", t(pat)[pi..min(pi + 2, length)])); }
                    for q in n { sb.push(pc(pi + q)); }
                    pi += n;
                } else {
                    sb.push(pc(pi));
                    pi++;
                }
            }
            if pi >= length { fail_pattern(F_INVALID, "unterminated character class"); }
            sb.push(']');
            pi++;
            seq.push(Item { s: fin(s0), literal: false });
        } else if c == '|' {
            seq.push(Item { s: lit("|"), literal: false });
            pi++;
        } else if c == '*' || c == '+' || c == '?' {
            if seq.len == mark { fail_pattern(F_INVALID, "nothing to repeat"); }
            let r = to_rule(seq[seq.len - 1]);
            let s0 = sb.len;
            put(r);
            sb.push(c);
            seq[seq.len - 1] = Item { s: fin(s0), literal: false };
            pi++;
        } else if c == '{' {
            let start = pi + 1;
            pi++;
            while pi < length && pc(pi) != '}' { pi++; }
            if pi >= length { fail_pattern(F_UNSUPPORTED, "unterminated curly brackets"); }
            let inside = t(pat)[start..pi];
            pi++;
            var lo = 0;
            var hi = I32_MAX;
            let comma = find(inside, ",");
            if comma >= 0 && find(inside[comma + 1..], ",") >= 0 { fail_pattern(F_UNSUPPORTED, "wrong number of values in curly brackets"); }
            if comma < 0 {
                let v = stoi(inside);
                lo = v;
                hi = v;
            } else {
                if comma > 0 { lo = stoi(inside[..comma]); }
                if comma + 1 < inside.len { hi = stoi(inside[comma + 1..]); }
            }
            if seq.len == mark { fail_pattern(F_INVALID, "nothing to repeat"); }
            let last = seq[seq.len - 1];
            var sub = last.s;
            if !last.literal {
                var id = Str { .. };
                var found = false;
                for e in sub_rules { if eq(e.sub, sub) { id = e.id; found = true; } }
                if !found {
                    sub_rules.push(SubRule { sub: sub, id: Str { .. } });
                    let s0 = sb.len;
                    put(pname);
                    sb.push('-');
                    put_i(sub_rules.len);
                    let nm = fin(s0);
                    id = add_rule(nm, sub);
                    sub_rules[sub_rules.len - 1].id = id;
                }
                sub = id;
            }
            let item = if last.literal { to_rule(last) } else { sub };
            let rep = build_repetition(item, lo, hi, lit(""));
            seq[seq.len - 1] = Item { s: rep, literal: false };
        } else {
            let s0 = sb.len;
            while pi < length {
                let ch = pc(pi);
                if ch == '\\' {
                    if pi == length - 1 { fail_pattern(F_INVALID, "trailing backslash"); }
                    let next = pc(pi + 1);
                    if escaped_in_regex_only(next) {
                        pi++;
                        sb.push(pc(pi));
                        pi++;
                    } else {
                        let n = escape_length(t(pat), pi);
                        if n == 0 { fail_pattern(F_UNSUPPORTED, str("unsupported escape: ", t(pat)[pi..min(pi + 2, length)])); }
                        for q in n { sb.push(pc(pi + q)); }
                        pi += n;
                    }
                } else if ch == '"' {
                    put_b("\\\"");
                    pi++;
                } else if !non_literal(ch) && (pi == length - 1 || sb.len == s0 || pc(pi + 1) == '.' || !non_literal(pc(pi + 1))) {
                    sb.push(ch);
                    pi++;
                } else {
                    break;
                }
            }
            if sb.len == s0 {
                var msg: u8[>..] = [];
                msg.append("unsupported character: ");
                msg.push(c);
                fail_pattern(F_UNSUPPORTED, msg);
            }
            seq.push(Item { s: fin(s0), literal: true });
        }
    }
    return join_seq(mark);
}

// std::stoi: leading space, a sign, digits; else the conversion fails.
fn stoi(s: const u8[:]) -> i64 {
    var i = 0;
    while i < s.len && (s[i] == ' ' || (s[i] >= 9 && s[i] <= 13)) { i++; }
    var neg = false;
    if i < s.len && (s[i] == '+' || s[i] == '-') { neg = s[i] == '-'; i++; }
    let d0 = i;
    var v = 0;
    while i < s.len && s[i] >= '0' && s[i] <= '9' {
        v = v * 10 + (s[i] - '0');
        if v > 2147483648 { fail_pattern(F_UNSUPPORTED, "invalid number in curly brackets"); }
        i++;
    }
    if i == d0 { fail_pattern(F_UNSUPPORTED, "invalid number in curly brackets"); }
    if neg { v = -v; }
    if v > 2147483647 { fail_pattern(F_UNSUPPORTED, "invalid number in curly brackets"); }
    return v;
}

fn pattern_to_rule(pattern: Str, name: Str) -> Str {
    let p = t(pattern);
    if p.len < 2 || p[0] != '^' || p[p.len - 1] != '$' { fail_pattern(F_UNSUPPORTED, "not anchored with '^' and '$'"); }
    pat = Str { at: pattern.at + 1, len: pattern.len - 2 };
    pi = 0;
    paren_depth = 0;
    pname = name;
    sub_rules.clear();
    seq.clear();
    let r = to_rule(transform());
    if paren_depth != 0 { fail_pattern(F_INVALID, "unbalanced parentheses"); }
    let s0 = sb.len;
    put_b("\"\\\"\" (");
    put(r);
    put_b(") \"\\\"\"");
    return add_rule(name, fin(s0));
}

fn visit_pattern(pattern: Str, name: Str) -> Str {
    // the rules as they were, for a pattern that cannot be converted
    var snapshot: Rule[>..] = [];
    snapshot.append(rules);
    var result = Str { .. };
    var kind = -1;
    fn body() { result = pattern_to_rule(pattern, name); }
    exception::try<Fail>(body) { f => kind = f.kind; };
    if kind < 0 { return result; }
    if kind == F_ERROR { exception::throw(Fail { kind: F_ERROR }); }
    rules.clear();
    rules.append(snapshot);
    if kind == F_UNSUPPORTED {
        warnings.push(lit(str("pattern ", t(pattern), " is not supported (", err, "), accepting any string")));
        return add_rule(name, add_primitive_b("string"));
    }
    errors.push(lit(str("Invalid pattern ", t(pattern), ": ", err)));
    return lit("");
}

//
// the trie of _not_strings
//

struct TNode { first: i64 = -1, next: i64 = -1, cpt: u32 = 0, pattern: i32 = -1 }
var tnodes: TNode[>..] = [];

// The child of node n for cpt, made if it has none (children kept by code point).
fn trie_child(n: i64, cpt: u32) -> i64 {
    var prev = -1;
    var c = tnodes[n].first;
    while c >= 0 && tnodes[c].cpt < cpt { prev = c; c = tnodes[c].next; }
    if c >= 0 && tnodes[c].cpt == cpt { return c; }
    tnodes.push(TNode { cpt: cpt, next: c, .. });
    let nc = tnodes.len - 1;
    if prev < 0 { tnodes[n].first = nc; } else { tnodes[prev].next = nc; }
    return nc;
}

fn trie_insert(word: const u8[:], index: i32) {
    var cur = 0;
    var pos = 0;
    while pos < word.len {
        let cp, n = utf8_decode(word, pos);
        if n == 0 { break; }
        cur = trie_child(cur, cp as! u32);
        pos += n;
    }
    if tnodes[cur].pattern < 0 { tnodes[cur].pattern = index; }
}

fn put_cpt(cp: u32) {
    var tmp: u8[>..] = [];
    push_utf8(tmp, cp);
    sb.append(tmp);
}

recursive fn not_strings_visit(n: i64, char_rule: Str, d: i64) {
    if d >= LIMIT { fail(str("JSON schema property name is longer than ", LIMIT, " chars")); }
    var c = tnodes[n].first;
    var first = true;
    while c >= 0 {
        if !first { put_b(" | "); }
        first = false;
        sb.push('[');
        put_cpt(tnodes[c].cpt);
        sb.push(']');
        if tnodes[c].first >= 0 {
            put_b(" (");
            not_strings_visit(c, char_rule, d + 1);
            sb.push(')');
        } else {
            sb.push(' ');
            put(char_rule);
            sb.push('+');
        }
        c = tnodes[c].next;
    }
    if tnodes[n].first >= 0 {
        put_b(" | [^\"");
        var k = tnodes[n].first;
        while k >= 0 { put_cpt(tnodes[k].cpt); k = tnodes[k].next; }
        put_b("] ");
        put(char_rule);
        sb.push('*');
    }
}

// A JSON string none of the names (props[at..at + n]) are (_not_strings).
fn not_strings(names: const Str[:]) -> Str {
    tnodes.clear();
    tnodes.push(TNode { .. });
    for nm, k in names { trie_insert(t(nm), k as i32); }
    let char_rule = add_primitive_b("char");
    let s0 = sb.len;
    put_b("[\"] ( ");
    not_strings_visit(0, char_rule, 0);
    put_b(" )");
    if tnodes[0].pattern < 0 { sb.push('?'); }
    put_b(" [\"]");
    return fin(s0);
}

//
// objects
//

// A property in the making: its name, node, whether required, its kv rule.
struct OProp { name: Str, node: i64, required: bool, kv: Str }
var oprops: OProp[>..<] = [];
var onames: Str[>..<] = [];


// The keys optional ones from k on, as get_recursive_refs writes them.
recursive fn recursive_refs(opt: const i64[:], k: i64, first_is_optional: bool, name: Str, first: i64, n: i64, star_kv: Str) -> Str {
    let key = if opt[k] < 0 { lit("*") } else { oprops[opt[k]].name };
    let kv = if opt[k] < 0 { star_kv } else { oprops[opt[k]].kv };
    let star = opt[k] < 0;
    let s0 = sb.len;
    put_b("( \",\" space ");
    put(kv);
    put_b(" )");
    let comma_ref = fin(s0);
    var res = Str { .. };
    if first_is_optional {
        res = cat2(comma_ref, if star { "*" } else { "?" });
    } else {
        let s1 = sb.len;
        put(kv);
        if star { sb.push(' '); put(comma_ref); sb.push('*'); }
        res = fin(s1);
    }
    if k + 1 < opt.len {
        let inner = recursive_refs(opt, k + 1, true, name, first, n, star_kv);
        let s2 = sb.len;
        put(name);
        if name.len > 0 { sb.push('-'); }
        put(key);
        put_b("-rest");
        let rest_name = add_rule(fin(s2), inner);
        res = cat3(res, " ", rest_name);
    }
    return res;
}

// _build_object_rule over the properties oprops[first..first + n].
recursive fn build_object_rule(first: i64, n: i64, name: Str, additional: i64) -> Str {
    for k in first..first + n {
        let s0 = sb.len;
        put(name);
        if name.len > 0 { sb.push('-'); }
        put(oprops[k].name);
        let sub = fin(s0);
        let prop_rule = visit(oprops[k].node, sub);
        let kv_name = cat2(sub, "-kv");
        var name_json: u8[>..] = [];
        json_string(name_json, t(oprops[k].name));
        let lit_name = format_literal(lit(name_json));
        let rule = cat3(lit_name, " space \":\" space ", prop_rule);
        oprops[k].kv = add_rule(kv_name, rule);
    }
    // a name given twice has the kv rule of the last (llama.cpp's map by name)
    for k in first..first + n {
        for q in k + 1..first + n { if eq(oprops[q].name, oprops[k].name) { oprops[k].kv = oprops[q].kv; } }
    }

    var star_kv = Str { .. };
    var has_star = false;
    if additional >= 0 {
        let s0 = sb.len;
        put(name);
        if name.len > 0 { sb.push('-'); }
        put_b("additional");
        let sub = fin(s0);
        let value_rule = if nodes[additional].kind != K_ANY { visit(additional, cat2(sub, "-value")) } else { add_primitive_b("value") };
        var key_rule = Str { .. };
        if n == 0 {
            key_rule = add_primitive_b("string");
        } else {
            let mark = onames.len;
            for k in first..first + n { onames.push(oprops[k].name); }
            var names: Str[>..] = [];
            for k in mark..onames.len { names.push(onames[k]); }
            onames.resize(mark);
            let ns = not_strings(names);
            key_rule = add_rule(cat2(sub, "-k"), ns);
        }
        star_kv = add_rule(cat2(sub, "-kv"), cat3(key_rule, " \":\" space ", value_rule));
        has_star = true;
    }
    var n_req = 0;
    var n_opt = 0;
    for k in first..first + n { if oprops[k].required { n_req++; } else { n_opt++; } }
    if has_star { n_opt++; }
    if n_req == 0 && n_opt == 0 { return lit("\"{\" space \"}\""); }

    // the optional ones, as indices (-1: the additional properties)
    var opt: i64[>..] = [];
    for k in first..first + n { if !oprops[k].required { opt.push(k); } }
    if has_star { opt.push(-1); }
    // the rest rules first, for each start among the optional ones
    var alts: Str[>..] = [];
    for i in opt.len { alts.push(recursive_refs(opt[..], i, false, name, first, n, star_kv)); }

    let s0 = sb.len;
    put_b("\"{\" space ");
    var i = 0;
    for k in first..first + n {
        guard oprops[k].required else { continue; }
        if i > 0 { put_b(" \",\" space "); }
        put(oprops[k].kv);
        i++;
    }
    if opt.len > 0 {
        put_b(" (");
        if n_req > 0 { put_b(" \",\" space ( "); }
        for a, j in alts {
            if j > 0 { put_b(" | "); }
            put(a);
        }
        if n_req > 0 { put_b(" )"); }
        put_b(" )?");
    }
    put_b(" space \"}\"");
    return fin(s0);
}

//
// visit
//

fn is_reserved(name: Str) -> bool { is_reserved_name(t(name)) }

recursive fn visit_all_of(node: i64, name: Str, rule_name: Str) -> Str {
    let first = oprops.len;
    let nd = nodes[node];
    // the rule of the enum values all the children have, if any
    let enum_rule = all_of_enums(node);
    if enum_rule.len > 0 {
        oprops.resize(first);
        return add_rule(rule_name, enum_rule);
    }
    // required is by name, as llama.cpp's set of names
    for k in first..oprops.len {
        for q in first..oprops.len { if oprops[q].required && eq(oprops[q].name, oprops[k].name) { oprops[k].required = true; } }
    }
    let r = build_object_rule(first, oprops.len - first, name, -1);
    oprops.resize(first);
    return add_rule(rule_name, r);
}

// The properties of allOf's children onto oprops, and the rule of the enum
// values every child has (empty if none).
fn all_of_enums(node: i64) -> Str {
    // the enum values, by their rule, and how many children have each
    var enum_rules: Str[>..] = [];
    var enum_counts: i64[>..] = [];
    let nd = nodes[node];
    for c in nd.kids_n {
        let child = kids[nd.kids_at + c];
        if nodes[child].kind == K_ANY_OF {
            for a in nodes[child].kids_n { add_component(kids[nodes[child].kids_at + a], false, 0, enum_rules, enum_counts); }
        } else {
            add_component(child, true, 0, enum_rules, enum_counts);
        }
    }
    // in the order of their rules
    var order: i64[>..] = [];
    for k in enum_rules.len { if enum_counts[k] == nd.kids_n { order.push(k); } }
    stable_sort(order) { a, b => cmp_bytes(t(enum_rules[a]), t(enum_rules[b])) < 0 };
    if order.len == 0 { return Str { .. }; }
    let s0 = sb.len;
    sb.push('(');
    for k, j in order {
        if j > 0 { put_b(" | "); }
        put(enum_rules[k]);
    }
    sb.push(')');
    return fin(s0);
}

recursive fn add_component(comp: i64, required: bool, d: i64, enum_rules: Str[>..]&, enum_counts: i64[>..]&) {
    if d >= LIMIT { fail(str("JSON schema is nested more than ", LIMIT, " levels deep")); }
    let c = nodes[comp];
    if c.kind == K_REF {
        if c.target >= 0 { add_component(c.target, required, d + 1, enum_rules, enum_counts); }
    } else if c.kind == K_OBJECT {
        for k in c.kids_n {
            let p = props[c.kids_at + k];
            oprops.push(OProp { name: p.name, node: p.node, required: required, kv: Str { .. } });
        }
    } else if c.kind == K_ENUM {
        for k in c.kids_n {
            let r = constant_rule(kids[c.kids_at + k]);
            var hit = false;
            for e, j in enum_rules { if !hit && eq(e, r) { enum_counts[j] += 1; hit = true; } }
            if !hit { enum_rules.push(r); enum_counts.push(1); }
        }
    }
}

recursive fn resolve_ref_rule(node: i64) -> Str {
    let nd = nodes[node];
    let r = t(nd.ref);
    let hash = find(r, "#");
    let frag = if hash >= 0 { Str { at: nd.ref.at + hash + 1, len: nd.ref.len - hash - 1 } } else { nd.ref };
    let esc = format_rule_name(frag);
    let s0 = sb.len;
    put_b("ref");
    put(esc);
    var ref_name = fin(s0);
    var resolving = false;
    for b in being_resolved { if eq(b, nd.ref) { resolving = true; } }
    if rule_index(t(ref_name)) < 0 && !resolving {
        if nd.target < 0 {
            errors.push(lit(str("Unresolved $ref ", r)));
            return lit("");
        }
        being_resolved.push(nd.ref);
        ref_name = visit(nd.target, ref_name);
        // erase it
        var k = 0;
        while k < being_resolved.len {
            if eq(being_resolved[k], nd.ref) {
                for q in k..being_resolved.len - 1 { being_resolved[q] = being_resolved[q + 1]; }
                being_resolved.pop();
                break;
            }
            k++;
        }
    }
    return ref_name;
}

fn primitive_rule(rule_name: Str, kind_name: const u8[:]) -> Str {
    return add_primitive(if t(rule_name) == "root" { lit("root") } else { lit(kind_name) }, kind_name);
}

// The rule of node, named after name: its name (common_chat_schema_converter::visit).
recursive fn visit(node: i64, name: Str) -> Str {
    vdepth++;
    if vdepth > LIMIT { fail(str("JSON schema is nested more than ", LIMIT, " levels deep")); }
    let r = visit_impl(node, name);
    vdepth--;
    return r;
}

recursive fn visit_impl(node: i64, name: Str) -> Str {
    let rule_name = if is_reserved(name) { cat2(name, "-") } else if name.len == 0 { lit("root") } else { name };
    let sub_name = if name.len == 0 { name } else { cat2(name, "-") };
    let nd = nodes[node];
    let k = nd.kind;
    if k == K_REF { return add_rule(rule_name, resolve_ref_rule(node)); }
    if k == K_ANY_OF {
        let mark = onames.len;
        for i in nd.kids_n {
            let s0 = sb.len;
            put(name);
            put_b(if name.len == 0 { "alternative-" } else { "-" });
            put_i(i);
            let alt_name = fin(s0);
            let r = visit(kids[nd.kids_at + i], alt_name);
            onames.push(r);
        }
        let s0 = sb.len;
        for i in mark..onames.len {
            if i > mark { put_b(" | "); }
            put(onames[i]);
        }
        let union = fin(s0);
        onames.resize(mark);
        return add_rule(rule_name, union);
    }
    if k == K_ALL_OF { return visit_all_of(node, name, rule_name); }
    if k == K_CONST { return add_rule(rule_name, constant_rule(nd.value)); }
    if k == K_ENUM {
        let mark = onames.len;
        for i in nd.kids_n { onames.push(constant_rule(kids[nd.kids_at + i])); }
        let s0 = sb.len;
        sb.push('(');
        for i in mark..onames.len {
            if i > mark { put_b(" | "); }
            put(onames[i]);
        }
        sb.push(')');
        onames.resize(mark);
        return add_rule(rule_name, fin(s0));
    }
    if k == K_OBJECT {
        if nd.kids_n == 0 && nd.additional >= 0 && nodes[nd.additional].kind == K_ANY {
            return add_rule(rule_name, add_primitive_b("object"));
        }
        let first = oprops.len;
        for i in nd.kids_n {
            let p = props[nd.kids_at + i];
            oprops.push(OProp { name: p.name, node: p.node, required: p.required, kv: Str { .. } });
        }
        let r = build_object_rule(first, nd.kids_n, name, nd.additional);
        oprops.resize(first);
        return add_rule(rule_name, r);
    }
    if k == K_TUPLE {
        let mark = onames.len;
        for i in nd.kids_n {
            let s0 = sb.len;
            put(sub_name);
            put_b("tuple-");
            put_i(i);
            let nm = fin(s0);
            let r = visit(kids[nd.kids_at + i], nm);
            onames.push(r);
        }
        let s0 = sb.len;
        put_b("\"[\" space ");
        for i in mark..onames.len {
            if i > mark { put_b(" \",\" space "); }
            put(onames[i]);
        }
        put_b(" space \"]\"");
        onames.resize(mark);
        return add_rule(rule_name, fin(s0));
    }
    if k == K_ARRAY {
        if nodes[nd.items].kind == K_ANY && nd.lo == 0 && nd.hi < 0 { return primitive_rule(rule_name, "array"); }
        let item = visit(nd.items, cat2(sub_name, "item"));
        let hi = if nd.hi < 0 { I32_MAX } else { nd.hi };
        let rep = build_repetition(item, nd.lo, hi, lit("\",\" space"));
        let s0 = sb.len;
        put_b("\"[\" space ");
        put(rep);
        put_b(" space \"]\"");
        return add_rule(rule_name, fin(s0));
    }
    if k == K_STRING {
        if nd.has_pattern { return visit_pattern(nd.pattern, rule_name); }
        if nd.format == FORMAT_UUID { return primitive_rule(rule_name, "uuid"); }
        if nd.format != FORMAT_NONE {
            let prim = if nd.format == FORMAT_DATE { "date-string" } else if nd.format == FORMAT_TIME { "time-string" } else { "date-time-string" };
            return add_rule(rule_name, add_primitive_b(prim));
        }
        if nd.lo > 0 || nd.hi >= 0 {
            let char_rule = add_primitive_b("char");
            let hi = if nd.hi < 0 { I32_MAX } else { nd.hi };
            let rep = build_repetition(char_rule, nd.lo, hi, lit(""));
            let s0 = sb.len;
            put_b("\"\\\"\" ");
            put(rep);
            put_b(" \"\\\"\"");
            return add_rule(rule_name, fin(s0));
        }
        return primitive_rule(rule_name, "string");
    }
    if k == K_INTEGER {
        if nd.lo == I64_MIN && nd.hi == I64_MAX { return primitive_rule(rule_name, "integer"); }
        nb.clear();
        let s0 = sb.len;
        sb.push('(');
        min_max_int(nd.lo, nd.hi, 16, true);
        sb.push(')');
        return add_rule(rule_name, fin(s0));
    }
    if k == K_NUMBER { return primitive_rule(rule_name, "number"); }
    if k == K_BOOLEAN { return primitive_rule(rule_name, "boolean"); }
    if k == K_NULL { return primitive_rule(rule_name, "null"); }
    return add_rule(rule_name, add_primitive_b("value"));
}

// The rules, by name, one a line (format_grammar).
fn format_grammar(out: u8[>..]&) {
    var order: i64[>..] = [];
    for k in rules.len { order.push(k); }
    stable_sort(order) { a, b => cmp_bytes(t(rules[a].name), t(rules[b].name)) < 0 };
    for k in order { format(out, t(rules[k].name), " ::= ", t(rules[k].rule), "\n"); }
}

fn start_converter() {
    rules.clear();
    errors.clear();
    warnings.clear();
    being_resolved.clear();
    oprops.clear();
    onames.clear();
    vdepth = 0;
    dotall = false;
    set_rule(lit("space"), lit(SPACE_RULE));
}

fn check_errors() {
    if errors.len > 0 {
        var msg: u8[>..] = [];
        msg.append("JSON schema conversion failed:\n");
        for e, k in errors {
            if k > 0 { msg.push('\n'); }
            msg.append(t(e));
        }
        fail(msg);
    }
    if warnings.len > 0 {
        var msg: u8[>..] = [];
        msg.append("WARNING: JSON schema conversion was incomplete: ");
        for w, k in warnings {
            if k > 0 { msg.append("; "); }
            msg.append(t(w));
        }
        msg.push('\n');
        write_stderr(msg);
    }
}

// json_schema_to_grammar: the GBNF grammar of the JSON schema `schema` (its
// text), into out; false, and the message in err, if the schema is not
// JSON or falls outside what the converter handles.
// Forgets the schemas, their JSON and the texts.
fn reset() {
    sb.clear();
    jv.clear();
    jl.clear();
    nodes.clear();
    kids.clear();
    props.clear();
    refs.clear();
}

// A JSON schema's text as schema nodes, added to those there are: its root
// node; -1, and the error in err, if it is not JSON or not a schema
// (common_chat_schema_from_json).
fn from_json(schema: const u8[:]) -> i64 {
    let v = parse_json(schema);
    if v < 0 { return -1; }
    var root = -1;
    fn body() { root = build_document(v); }
    exception::try<Fail>(body);
    return root;
}

// A JSON value's text, added to the values there are (common_json::parse):
// the value, or -1 and the error in err.
fn json_value(text: const u8[:]) -> i64 { parse_json(text) }

// The schema nodes of a JSON value read before: the root node, or -1 and
// the error in err.
fn from_value(v: i64) -> i64 {
    var root = -1;
    fn body() { root = build_document(v); }
    exception::try<Fail>(body);
    return root;
}

// A JSON value's members and elements, for the chat code.
fn get_member(v: i64, name: const u8[:]) -> i64 { get(v, name) }
fn json_kind(v: i64) -> u8 { kind(v) }
fn json_len(v: i64) -> i64 { jv[v].n }
fn json_at(v: i64, k: i64) -> i64 { jl[jv[v].at + k] }
fn json_key(v: i64, k: i64) -> const u8[:] { t(jv[jl[jv[v].at + k]].key) }
fn json_str(v: i64) -> const u8[:] { t(jv[v].s) }
fn json_dump(out: u8[>..]&, v: i64) { dump_into(out, v); }

// Whether a value the node admits may be a string (common_chat_schema::may_be_string).
fn may_be_string(node: i64) -> bool {
    var visited: i64[>..] = [];
    return may_be_string_impl(node, visited, 0);
}

recursive fn may_be_string_impl(node: i64, visited: i64[>..]&, d: i64) -> bool {
    let nd = nodes[node];
    let k = nd.kind;
    if k == K_STRING { return true; }
    if k == K_CONST { return kind(nd.value) == J_STRING; }
    if k == K_ENUM {
        for i in nd.kids_n { if kind(kids[nd.kids_at + i]) == J_STRING { return true; } }
        return false;
    }
    if k == K_REF {
        // a cycle is taken as not a string, to be safe
        if nd.target < 0 { return false; }
        for v in visited { if v == nd.target { return false; } }
        visited.push(nd.target);
        let r = may_be_string_impl(nd.target, visited, d + 1);
        visited.pop();
        return r;
    }
    if k == K_ANY_OF {
        for i in nd.kids_n { if may_be_string_impl(kids[nd.kids_at + i], visited, d + 1) { return true; } }
        return false;
    }
    if k == K_ALL_OF {
        // every child must allow a string, an any child constrains nothing
        var any_string = false;
        for i in nd.kids_n {
            let c = kids[nd.kids_at + i];
            guard nodes[c].kind != K_ANY else { continue; }
            if !may_be_string_impl(c, visited, d + 1) { return false; }
            any_string = true;
        }
        return any_string;
    }
    return false;
}

// build_grammar: begin_grammar, then rules and schemas added, then
// end_grammar for the text. add_rule and add_schema throw Fail, which
// the caller's handler takes.
fn begin_grammar() { start_converter(); }
fn add_schema(name, node: i64) -> Str { visit(node, if t(name) == "root" { lit("") } else { name }) }
fn end_grammar(out) {
    check_errors();
    format_grammar(out);
}

fn to_grammar(schema: const u8[:], out: u8[>..]&) -> bool {
    reset();
    let v = parse_json(schema);

    if v < 0 { return false; }
    var ok = false;
    fn body() {
        // the parse errors (runtime_error) say so
        let root = build_document(v);
        start_converter();
        visit(root, lit(""));
        check_errors();
        format_grammar(out);
        ok = true;
    }
    exception::try<Fail>(body);
    if !ok && starts_with(err, "JSON schema error at ") {
        var msg: u8[>..] = [];
        format(msg, "JSON schema conversion failed:\n", err);
        err.clear();
        err.append(msg);
    }
    return ok;
}

// A schema node's kind, and an object node's properties (in schema order).
fn node_kind(n: i64) -> u8 { nodes[n].kind }
fn prop_count(n: i64) -> i64 { if nodes[n].kind == K_OBJECT { nodes[n].kids_n } else { 0 } }
fn prop_name(n: i64, k: i64) -> const u8[:] { t(props[nodes[n].kids_at + k].name) }
fn prop_node(n: i64, k: i64) -> i64 { props[nodes[n].kids_at + k].node }
fn prop_required(n: i64, k: i64) -> bool { props[nodes[n].kids_at + k].required }
