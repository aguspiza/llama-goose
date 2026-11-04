// JSON documents for the server and its clients: text parsed (with the
// standard library's json) into a table of values that keeps the members
// in order, read by index, and JSON written as nlohmann's dump() writes
// it, with invalid UTF-8 replaced (error_handler_t::replace). The values
// of all the documents parsed since the last reset() stay valid.
// port-of: vendor/nlohmann/json.hpp (dump, the serializer's UTF-8 handling)
import std;
import json;
import .number;

namespace jdoc;

let NULL = 0;
let BOOL = 1;
let INT = 2;
let FLOAT = 3;
let STRING = 4;
let ARRAY = 5;
let OBJECT = 6;

// A value: its kind, its number, its string (text[s..s + sn]), its key in
// its object (text[k..k + kn]), and its elements or members (kids[at..at + n]).
struct V { kind: u8, i: i64 = 0, f: f64 = 0.0, s: i64 = 0, sn: i64 = 0, k: i64 = 0, kn: i64 = 0, at: i64 = 0, n: i64 = 0 }

var vals: V[>..] = [];
var kids: i64[>..] = [];
var text: u8[>..] = [];
var stack: i64[>..<] = [];
var err: u8[>..] = [];

fn reset() {
    vals.clear();
    kids.clear();
    text.clear();
}

fn keep(b: const u8[:]) -> i64 {
    let at = text.len;
    text.append(b);
    return at;
}

recursive fn convert(n: json::Node&) -> i64 {
    var v = V { kind: NULL, k: keep(n.key), kn: n.key.len, .. };
    var container = false;
    match n.val {
        Null => {},
        Bool b => { v.kind = BOOL; v.i = if b.v { 1 } else { 0 }; },
        Int x => { v.kind = INT; v.i = x.v; v.f = x.v; },
        Num x => { v.kind = FLOAT; v.f = x.v; v.i = to_i64(x.v); },
        Str x => { v.kind = STRING; v.s = keep(x.s); v.sn = x.s.len; },
        Arr => { v.kind = ARRAY; container = true; },
        Obj => { v.kind = OBJECT; container = true; },
    }
    if container {
        let mark = stack.len;
        var c = n.first;
        while c {
            let k = convert(c);
            stack.push(k);
            c .= c.next;
        }
        v.at = kids.len;
        for k in mark..stack.len { kids.push(stack[k]); }
        v.n = stack.len - mark;
        stack.resize(mark);
    }
    vals.push(v);
    return vals.len - 1;
}

// The text as a document: its root, or -1 and the message in err.
fn parse(src: const u8[:]) -> i64 {
    stack.clear();
    var copy: u8[>..] = [];
    var doc: json::Node[>..] = [];
    var decoded: u8[>..] = [];
    copy.append(src);
    let root, e = json::parse(copy, doc, decoded);
    guard root else {
        err.clear();
        err.append(e);
        return -1;
    }
    return convert(root);
}

// static_cast<int64_t>(v) as x86 does it.
fn to_i64(v: f64) -> i64 {
    if !(v < 9223372036854775808.0 && v >= -9223372036854775808.0) { return -9223372036854775807 - 1; }
    return v as! i64;
}

//
// reading; -1 stands for a value that is not there
//

fn kind(v: i64) -> i64 { if v < 0 { -1 } else { vals[v].kind } }
fn is_null(v: i64) -> bool { v < 0 || vals[v].kind == NULL }
fn is_str(v: i64) -> bool { kind(v) == STRING }
fn is_bool(v: i64) -> bool { kind(v) == BOOL }
fn is_num(v: i64) -> bool { kind(v) == INT || kind(v) == FLOAT }
fn is_int(v: i64) -> bool { kind(v) == INT }
fn is_arr(v: i64) -> bool { kind(v) == ARRAY }
fn is_obj(v: i64) -> bool { kind(v) == OBJECT }

// Elements or members.
fn len(v: i64) -> i64 { if is_arr(v) || is_obj(v) { vals[v].n } else { 0 } }
fn at(v: i64, k: i64) -> i64 { kids[vals[v].at + k] }
fn key(v: i64, k: i64) -> const u8[:] {
    let c = vals[kids[vals[v].at + k]];
    return text[c.k..c.k + c.kn];
}

// The member named name (the last, as nlohmann keeps it), or -1.
fn get(v: i64, name: const u8[:]) -> i64 {
    guard is_obj(v) else { return -1; }
    var found = -1;
    for k in vals[v].n {
        let c = vals[kids[vals[v].at + k]];
        if text[c.k..c.k + c.kn] == name { found = kids[vals[v].at + k]; }
    }
    return found;
}

// Whether the member is there and not null.
fn has(v: i64, name: const u8[:]) -> bool { !is_null(get(v, name)) }
fn contains(v: i64, name: const u8[:]) -> bool { get(v, name) >= 0 }

fn str_of(v: i64) -> const u8[:] {
    guard is_str(v) else { return ""; }
    return text[vals[v].s..vals[v].s + vals[v].sn];
}
fn num_of(v: i64) -> f64 { if is_num(v) { vals[v].f } else { 0.0 } }
fn int_of(v: i64) -> i64 { if is_num(v) { vals[v].i } else { 0 } }
fn bool_of(v: i64) -> bool { is_bool(v) && vals[v].i != 0 }

// json_value(obj, name, def) for each type: the default when the member is
// missing or null.
fn get_str(v: i64, name: const u8[:], def: const u8[:]) -> const u8[:] {
    let c = get(v, name);
    return if is_str(c) { str_of(c) } else { def };
}
fn get_int(v: i64, name: const u8[:], def: i64) -> i64 {
    let c = get(v, name);
    return if is_num(c) { int_of(c) } else { def };
}
fn get_num(v: i64, name: const u8[:], def: f64) -> f64 {
    let c = get(v, name);
    return if is_num(c) { num_of(c) } else { def };
}
fn get_bool(v: i64, name: const u8[:], def: bool) -> bool {
    let c = get(v, name);
    return if is_bool(c) { bool_of(c) } else { def };
}

//
// writing
//

// s as nlohmann writes a string: quoted, the control characters escaped,
// each invalid UTF-8 sequence replaced by U+FFFD.
fn put_str(out: u8[>..]&, s: const u8[:]) {
    out.push('"');
    var i = 0;
    while i < s.len {
        let c = s[i];
        if c < 0x80 {
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
            i++;
            continue;
        }
        let n = utf8_len(s, i);
        if n > 0 {
            out.append(s[i..i + n]);
            i += n;
        } else {
            out.append("\xEF\xBF\xBD");
            // the byte that ended a sequence too soon starts the next one
            i += max(1, -n);
        }
    }
    out.push('"');
}

// The length of the valid UTF-8 sequence at s[i] (not ASCII): n > 0, or
// -(the bytes the bad one takes up before the byte that breaks it); 0 for
// a byte that cannot start one. As nlohmann's decoder accepts them.
fn utf8_len(s: const u8[:], i: i64) -> i64 {
    let c = s[i];
    var n = 0;
    var lo: u8 = 0x80;
    var hi: u8 = 0xBF;
    if c >= 0xC2 && c <= 0xDF { n = 2; }
    else if c == 0xE0 { n = 3; lo = 0xA0; }
    else if c == 0xED { n = 3; hi = 0x9F; }
    else if c >= 0xE1 && c <= 0xEF { n = 3; }
    else if c == 0xF0 { n = 4; lo = 0x90; }
    else if c == 0xF4 { n = 4; hi = 0x8F; }
    else if c >= 0xF1 && c <= 0xF3 { n = 4; }
    else { return 0; }
    for k in 1..n {
        guard i + k < s.len else { return -k; }
        let b = s[i + k];
        let ok = if k == 1 { b >= lo && b <= hi } else { b >= 0x80 && b <= 0xBF };
        guard ok else { return -k; }
    }
    return n;
}

fn put_int(out: u8[>..]&, v: i64) { format(out, v); }
fn put_bool(out: u8[>..]&, b: bool) { out.append(if b { "true" } else { "false" }); }

// A double as nlohmann writes it: the fewest digits that read back,
// "1.0" for whole numbers, an exponent below 1e-4 and from 1e16; null for
// NaN and the infinities.
fn put_num(out: u8[>..]&, v: f64) {
    if v != v || v - v != 0.0 { out.append("null"); return; }
    number::float_repr(out, v);
}

// A float as nlohmann writes one: widened to a double.
fn put_f32(out: u8[>..]&, v: f32) { put_num(out, v); }

// Value v as dump() writes it, compact.
recursive fn dump(out: u8[>..]&, v: i64) {
    if v < 0 { out.append("null"); return; }
    let x = vals[v];
    if x.kind == NULL { out.append("null"); }
    else if x.kind == BOOL { put_bool(out, x.i != 0); }
    else if x.kind == INT { put_int(out, x.i); }
    else if x.kind == FLOAT { put_num(out, x.f); }
    else if x.kind == STRING { put_str(out, text[x.s..x.s + x.sn]); }
    else if x.kind == ARRAY {
        out.push('[');
        for k in x.n {
            if k > 0 { out.push(','); }
            dump(out, kids[x.at + k]);
        }
        out.push(']');
    } else {
        out.push('{');
        for k in x.n {
            if k > 0 { out.push(','); }
            let c = kids[x.at + k];
            put_str(out, text[vals[c].k..vals[c].k + vals[c].kn]);
            out.push(':');
            dump(out, c);
        }
        out.push('}');
    }
}

