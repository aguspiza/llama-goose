// llama.cpp's Jinja engine, the one its chat templates run on: its values.
// A value is an index into `vals`, shared the way llama.cpp shares a
// shared_ptr<value_t>: an array or an object changed through one index is
// changed for all. The contents live in arenas beside it: a string is a run
// of parts in `parts`, each a range of `text` that remembers whether it came
// from the user's input; an array is a range of `items`, an object a range
// of key and value pairs in it. Nothing is freed until reset().
// port-of: common/jinja/value.h, common/jinja/string.cpp, common/jinja/value.cpp (values, JSON)
import std;
import exception;
import .jinja_lexer;
import .jinja_parser;
import .jinja_runtime;
import .jinja_builtins;

namespace jinja;

extern "jinja_fmt_fixed" fn c_fmt_fixed(out: u8[:], v: f64) -> i64;
extern "jinja_fmt_exp" fn c_fmt_exp(out: u8[:], prec: i64, v: f64) -> i64;
extern "jinja_strtod" fn c_strtod(s: const u8[:], used: i64&) -> f64;
extern "jinja_stoi" fn c_stoi(s: const u8[:], base: i64, ok: i64&) -> i64;
extern "jinja_time" fn c_time() -> i64;
extern "jinja_strftime" fn c_strftime(out: u8[:], fmt: const u8[:], t: i64) -> i64;

// JINJA_RECURSIVE_LIMIT: how deep templates, values and calls may nest.
let LIMIT = 128;

let K_UNDEF = 0;
let K_NONE = 1;
let K_BOOL = 2;
let K_INT = 3;
let K_FLOAT = 4;
let K_STRING = 5;
let K_ARRAY = 6;
let K_TUPLE = 7;      // an array that cannot change
let K_OBJECT = 8;
let K_FUNC = 9;
let K_KWARG = 10;     // a keyword argument, name=value, on its way into a call

// What a template did with a value, which caps_get reads: the operations
// llama.cpp's stats record, the ones anything looks at.
let OP_TEST_IS_STRING: u8 = 1;
let OP_SELECTATTR: u8 = 2;
let OP_ARRAY_ACCESS: u8 = 4;

// A function value's `a`: a builtin's id, or one of these.
let FN_MACRO = -1;    // a {% macro %}: c is its node
let FN_CALLER = -2;   // the body of a {% call %}: c is its node, d the scope it captured

struct Val {
    kind: u8,
    used: bool = false,     // stats: the template read it
    ops: u8 = 0,            // stats: OP_* the template applied to it
    builtins: bool = true,  // an object with methods: not the context, a loop or a namespace
    i: i64 = 0,             // a bool or int; a string's length in bytes
    f: f64 = 0.0,           // a number as a float
    a: i64 = -1,            // string: first part; array, object: first item; func: builtin id or FN_*; undefined: hint (a string); kwarg: key (a string)
    n: i64 = 0,             // parts, items or key and value pairs
    cap: i64 = 0,           // array, object: room in items
    b: i64 = -1,            // func: the bound first argument; kwarg: the value
    c: i64 = -1,            // func: the macro's or call's node
    d: i64 = -1,            // func: the scope a call body captured
    name: i64 = -1,         // func: its name (a string)
    flat: i64 = -1,         // string: where all its bytes are, in a row, in text
}

struct Part { off: i64, len: i64, input: bool }

var vals: Val[>..] = [];
var text: u8[>..] = [];
var parts: Part[>..] = [];
var items: i64[>..] = [];

// The arguments of the calls in progress, each call's a range of it.
var argstack: i64[>..<] = [];
// String parts being gathered into a block's output.
var gather: Part[>..<] = [];

// Frees every value, string, array and object.
fn reset() {
    vals.clear();
    text.clear();
    parts.clear();
    items.clear();
    argstack.clear();
    gather.clear();
}

//
// errors
//

// What a failure throws; the message is in err. llama.cpp's not-implemented
// errors are told apart, as its tests skip them.
struct Fail { not_implemented: bool }

var err: u8[>..] = [];

// The statement being executed, which an error message names.
var cur_node = -1;

// Throws msg as a std::runtime_error would leave llama.cpp's statement::execute:
// with the statement being executed and where it is in the source.
fn fail(msg: const u8[:]) {
    err.clear();
    if cur_node < 0 {
        err.append(msg);
    } else {
        let src = run_source();
        let pos = nodes[cur_node].pos;
        if src.len == 0 {
            format(err, "\nError executing ", node_type(nodes[cur_node].kind), " at position ", pos, ": ", msg);
        } else {
            format(err, "\n------------\nWhile executing ", node_type(nodes[cur_node].kind), " at ");
            line_col(err, src, pos);
            err.append(" in source:\n");
            peak_source(err, src, pos);
            format(err, "\nError: ", msg);
        }
    }
    exception::throw(Fail { not_implemented: false });
}

// llama.cpp's not_implemented_exception: passed on as it is.
fn not_implemented(msg: const u8[:]) {
    err.clear();
    format(err, "NotImplemented: ", msg);
    exception::throw(Fail { not_implemented: true });
}

// An error that is not a statement's: the lexer's and the parser's.
fn fail_plain(msg: const u8[:]) {
    err.clear();
    err.append(msg);
    exception::throw(Fail { not_implemented: false });
}

// "line L, column C" of pos in source.
fn line_col(out: u8[>..]&, source: const u8[:], pos: i64) {
    var line = 1;
    var col = 1;
    var i = 0;
    while i < pos && i < source.len {
        if source[i] == '\n' { line++; col = 1; } else { col++; }
        i++;
    }
    format(out, "line ", line, ", column ", col);
}

// The source around pos with a caret under it (utils.h peak_source).
fn peak_source(out: u8[>..]&, source: const u8[:], pos: i64) {
    if source.len == 0 { out.append("(no source available)"); return; }
    let start = if pos >= 40 { pos - 40 } else { 0 };
    let end = min(pos + 40, source.len);
    out.append("...");
    for c in source[min(start, end)..end] {
        if c == '\n' { out.append("↵"); } else { out.push(c); }
    }
    out.append("...\n");
    for i in pos - start + 3 { out.push(' '); }
    out.push('^');
}

//
// making values
//

fn new_val(v: Val) -> i64 {
    vals.push(v);
    return vals.len - 1;
}

fn mk_undef() -> i64 { new_val(Val { kind: K_UNDEF, .. }) }

// An undefined value that says where it came from, in error messages.
fn mk_undef_hint(hint: const u8[:]) -> i64 {
    let h = mk_str(hint, false);
    return new_val(Val { kind: K_UNDEF, a: h, .. });
}

fn mk_none() -> i64 { new_val(Val { kind: K_NONE, .. }) }

fn mk_bool(b: bool) -> i64 {
    let i = if b { 1 } else { 0 };
    return new_val(Val { kind: K_BOOL, i: i, f: i, .. });
}

// value_int_t: its float is the same number, or an infinity when a double
// cannot hold it.
fn mk_int(v: i64) -> i64 {
    var f = v as f64;
    if f >= 9223372036854775808.0 || (f as! i64) != v { f = if v < 0 { -1.0 / 0.0 } else { 1.0 / 0.0 }; }
    return new_val(Val { kind: K_INT, i: v, f: f, .. });
}

fn mk_float(v: f64) -> i64 {
    return new_val(Val { kind: K_FLOAT, i: trunc_int(v), f: v, .. });
}

// static_cast<int64_t>(v) as x86 does it: 0 for NaN and infinities (which
// value_float_t checks for), INT64_MIN for what does not fit.
fn trunc_int(v: f64) -> i64 {
    if v != v || v - v != 0.0 { return 0; }
    if !(v < 9223372036854775808.0 && v >= -9223372036854775808.0) { return -9223372036854775807 - 1; }
    return v as! i64;
}

// A string of one part: bytes, copied into text.
fn mk_str(bytes: const u8[:], input: bool) -> i64 {
    let off = text.len;
    text.append(bytes);
    parts.push(Part { off: off, len: bytes.len, input: input });
    return new_val(Val { kind: K_STRING, i: bytes.len, a: parts.len - 1, n: 1, flat: off, .. });
}

// A string of no parts: what an empty block makes.
fn mk_empty() -> i64 {
    return new_val(Val { kind: K_STRING, a: parts.len, n: 0, flat: 0, .. });
}

// A string of the parts parts[first..]: the ones just pushed.
fn mk_parts(first: i64) -> i64 {
    var len = 0;
    for k in first..parts.len { len += parts[k].len; }
    let n = parts.len - first;
    let flat = if n == 1 { parts[first].off } else if n == 0 { 0 } else { -1 };
    return new_val(Val { kind: K_STRING, i: len, a: first, n: n, flat: flat, .. });
}

// Pushes the parts of string s onto parts.
fn push_parts(s: i64) {
    let v = vals[s];
    for k in v.n { parts.push(parts[v.a + k]); }
}

// A string of a's parts and then b's (string::append).
fn mk_concat(a: i64, b: i64) -> i64 {
    let first = parts.len;
    push_parts(a);
    push_parts(b);
    return mk_parts(first);
}

// A copy of string s whose parts are all input when all of s's are
// (string::mark_input_based_on).
fn mark_input_based_on(r: i64, s: i64) {
    if !all_input(s) { return; }
    let v = vals[r];
    for k in v.n { parts[v.a + k].input = true; }
}

fn all_input(s: i64) -> bool {
    let v = vals[s];
    for k in v.n { if !parts[v.a + k].input { return false; } }
    return true;
}

// A string's bytes, all of them in a row.
fn sbytes(s: i64) -> const u8[:] {
    let v = vals[s];
    if v.flat >= 0 { return text[v.flat..v.flat + v.i]; }
    let off = text.len;
    for k in v.n {
        let p = parts[v.a + k];
        for i in p.len { text.push(text[p.off + i]); }
    }
    vals[s].flat = off;
    return text[off..off + v.i];
}

fn part_bytes(p: Part) -> const u8[:] { text[p.off..p.off + p.len] }

fn mk_array() -> i64 { new_val(Val { kind: K_ARRAY, .. }) }

fn mk_tuple() -> i64 { new_val(Val { kind: K_TUPLE, .. }) }

fn mk_object() -> i64 { new_val(Val { kind: K_OBJECT, .. }) }

// Room for one more item in array or object v (a pair takes two).
fn grow(v: i64, width: i64) {
    let x = vals[v];
    if x.n < x.cap { return; }
    let cap = if x.cap == 0 { 4 } else { x.cap * 2 };
    let start = items.len;
    for k in x.n * width { items.push(items[x.a + k]); }
    for k in (cap - x.n) * width { items.push(-1); }
    vals[v].a = start;
    vals[v].cap = cap;
}

// Appends item to array a, a tuple too while it is being made.
fn arr_push(a: i64, item: i64) {
    grow(a, 1);
    let x = vals[a];
    items[x.a + x.n] = item;
    vals[a].n = x.n + 1;
}

fn arr_len(a: i64) -> i64 { vals[a].n }

fn arr_at(a: i64, k: i64) -> i64 { items[vals[a].a + k] }

fn is_array(v: i64) -> bool { vals[v].kind == K_ARRAY || vals[v].kind == K_TUPLE }

fn kind(v: i64) -> u8 { vals[v].kind }

// A new array (or tuple, after kind) of the values items[from..from + n].
fn mk_array_of(kind: u8, from: i64, n: i64) -> i64 {
    let a = new_val(Val { kind: kind, .. });
    for k in n { arr_push(a, items[from + k]); }
    return a;
}

fn obj_len(o: i64) -> i64 { vals[o].n }

fn obj_key(o: i64, k: i64) -> i64 { items[vals[o].a + 2 * k] }

fn obj_value(o: i64, k: i64) -> i64 { items[vals[o].a + 2 * k + 1] }

// A copy of object o, its pairs in the same order (the object's own copy
// constructor: it keeps whether o has methods).
fn obj_copy(o: i64) -> i64 {
    let c = new_val(Val { kind: K_OBJECT, used: vals[o].used, ops: vals[o].ops, builtins: vals[o].builtins, .. });
    for k in obj_len(o) { obj_append(c, obj_key(o, k), obj_value(o, k)); }
    return c;
}

fn obj_append(o: i64, key: i64, v: i64) {
    grow(o, 2);
    let x = vals[o];
    items[x.a + 2 * x.n] = key;
    items[x.a + 2 * x.n + 1] = v;
    vals[o].n = x.n + 1;
}

// The pair of o whose key is equivalent to key, or -1. A key that cannot be
// hashed is an error, as in llama.cpp's has_key.
fn obj_find(o: i64, key: i64) -> i64 {
    if !immutable(key) || !hashable(key) { fail(str("Object key of unhashable type: ", tname(key))); }
    for k in obj_len(o) {
        if equiv(obj_key(o, k), key) { return k; }
    }
    return -1;
}

// The pair of o whose key is the string name, or -1.
fn obj_find_name(o: i64, name: const u8[:]) -> i64 {
    for k in obj_len(o) {
        let key = obj_key(o, k);
        if vals[key].kind == K_STRING && vals[key].i == name.len && sbytes(key) == name { return k; }
    }
    return -1;
}

// o[key] = v, in the place key had if it had one (value_object_t::insert).
fn obj_set(o: i64, key: i64, v: i64) {
    let k = obj_find(o, key);
    if k >= 0 { items[vals[o].a + 2 * k + 1] = v; return; }
    obj_append(o, key, v);
}

fn obj_set_name(o: i64, name: const u8[:], v: i64) {
    let k = obj_find_name(o, name);
    if k >= 0 { items[vals[o].a + 2 * k + 1] = v; return; }
    obj_append(o, mk_str(name, false), v);
}

// o[key], or -1.
fn obj_get(o: i64, key: i64) -> i64 {
    let k = obj_find(o, key);
    return if k >= 0 { obj_value(o, k) } else { -1 };
}

fn obj_get_name(o: i64, name: const u8[:]) -> i64 {
    let k = obj_find_name(o, name);
    return if k >= 0 { obj_value(o, k) } else { -1 };
}

//
// what a value is
//

fn numeric(v: i64) -> bool {
    let k = vals[v].kind;
    return k == K_INT || k == K_FLOAT || k == K_BOOL;
}

fn immutable(v: i64) -> bool {
    let k = vals[v].kind;
    return k != K_ARRAY && k != K_OBJECT;
}

recursive fn hashable(v: i64) -> bool {
    let x = vals[v];
    if x.kind == K_ARRAY || x.kind == K_TUPLE {
        for k in x.n {
            let e = items[x.a + k];
            if !immutable(e) || !hashable(e) { return false; }
        }
        return true;
    }
    if x.kind == K_OBJECT {
        for k in x.n {
            let e = items[x.a + 2 * k + 1];
            if !immutable(e) || !hashable(e) { return false; }
        }
        return true;
    }
    return x.kind != K_UNDEF && x.kind != K_FUNC;
}

// `==` (value_t::operator==, its equivalent()): numbers by their int and
// float views, so 1 == 1.0 == true; strings by their text; arrays and
// objects item by item, if both can be hashed.
recursive fn equiv(a: i64, b: i64) -> bool {
    let x = vals[a];
    let y = vals[b];
    if x.kind == K_INT || x.kind == K_FLOAT || x.kind == K_BOOL {
        return numeric(b) && x.i == y.i && x.f == y.f;
    }
    if x.kind == K_STRING { return y.kind == K_STRING && x.i == y.i && sbytes(a) == sbytes(b); }
    if x.kind == K_ARRAY || x.kind == K_TUPLE {
        if y.kind != x.kind || !hashable(a) || !hashable(b) || x.n != y.n { return false; }
        for k in x.n {
            if !equiv(items[x.a + k], items[y.a + k]) { return false; }
        }
        return true;
    }
    if x.kind == K_OBJECT {
        if y.kind != K_OBJECT || !hashable(a) || !hashable(b) || x.n != y.n { return false; }
        for k in x.n {
            if !equiv(items[x.a + 2 * k], items[y.a + 2 * k]) { return false; }
            if !equiv(items[x.a + 2 * k + 1], items[y.a + 2 * k + 1]) { return false; }
        }
        return true;
    }
    if x.kind == K_KWARG { return y.kind == K_KWARG && equiv(x.a, y.a) && x.b == y.b; }
    return x.kind == y.kind;
}

// as_bool(): what `if` makes of a value. A function or keyword argument has none.
fn truthy(v: i64) -> bool {
    let x = vals[v];
    if x.kind == K_BOOL || x.kind == K_INT { return x.i != 0; }
    if x.kind == K_FLOAT { return x.f != 0.0; }
    if x.kind == K_STRING { return x.i > 0; }
    if x.kind == K_ARRAY || x.kind == K_TUPLE || x.kind == K_OBJECT { return x.n > 0; }
    if x.kind == K_NONE || x.kind == K_UNDEF { return false; }
    fail(str(tname(v), " is not a bool value"));
    return false;
}

// as_int(), for the values that have one.
fn as_int(v: i64) -> i64 {
    let k = vals[v].kind;
    if k == K_INT || k == K_FLOAT || k == K_BOOL { return vals[v].i; }
    fail(str(tname(v), " is not an int value"));
    return 0;
}

fn as_float(v: i64) -> f64 {
    let k = vals[v].kind;
    if k == K_INT || k == K_FLOAT { return vals[v].f; }
    fail(str(tname(v), " is not a float value"));
    return 0.0;
}

// value_t::type(): a value's type, as error messages name it.
fn tname(v: i64) -> u8[>..] {
    let x = vals[v];
    if x.kind == K_UNDEF {
        if x.a < 0 { return str("Undefined"); }
        return str("Undefined (hint: '", sbytes(x.a), "')");
    }
    return str(match x.kind {
        K_NONE => "None",
        K_BOOL => "Boolean",
        K_INT => "Integer",
        K_FLOAT => "Float",
        K_STRING => "String",
        K_ARRAY => "Array",
        K_TUPLE => "Tuple",
        K_OBJECT => "Object",
        K_FUNC => "Function",
        _ => "KwArg",
    });
}

//
// values as text
//

// std::to_string(v).
fn format_int(out: u8[>..]&, v: i64) { format(out, v); }

// value_float_t::as_string: std::to_string's six decimals, less the
// trailing zeros, but one.
fn format_float(out: u8[>..]&, v: f64) {
    var buf: u8[400] = [0; 400];
    let n = c_fmt_fixed(buf[..], v);
    var end = n;
    while end > 0 && buf[end - 1] == '0' { end--; }
    out.append(buf[..end]);
    if end > 0 && buf[end - 1] == '.' { out.push('0'); }
}

// A float as Python's repr() writes it: the fewest digits that read back as
// the same value (value.cpp float_repr).
fn float_repr(out: u8[>..]&, val: f64) {
    if val != val { out.append("nan"); return; }
    if val - val != 0.0 { out.append(if val < 0.0 { "-inf" } else { "inf" }); return; }
    let mag = abs(val);
    let is_pow2 = mag != 0.0 && frexp_half(mag);
    var digits: u8[..32] = [];
    var exp10 = 0;
    // DBL_DIG - 1 digits after the first to start with; 17 always read back
    var prec = if mag < 2.2250738585072014e-308 { 0 } else { 14 };
    while digits.len == 0 {
        var buf: u8[40] = [0; 40];
        let n = c_fmt_exp(buf[..], prec, mag);
        var d: u8[..32] = [];
        var e = 0;
        while e < n && buf[e] != 'e' {
            if buf[e] >= '0' && buf[e] <= '9' { d.push(buf[e]); }
            e++;
        }
        let ev, ok = parse_int(buf[e + 1..n]);
        exp10 = ev;
        if prec == 16 || reads_back(d[..], exp10 - prec, mag) {
            digits = d;
        } else if is_pow2 && d[d.len - 1] != '9' {
            d[d.len - 1] += 1;
            if reads_back(d[..], exp10 - prec, mag) { digits = d; }
        }
        prec++;
    }
    while digits.len > 1 && digits[digits.len - 1] == '0' { digits.pop(); }

    let n = digits.len;
    let point = exp10 + 1;
    if val < 0.0 || (val == 0.0 && 1.0 / val < 0.0) { out.push('-'); }
    if point > -4 && point <= 16 {
        if point <= 0 {
            out.append("0.");
            for i in -point { out.push('0'); }
            out.append(digits[..]);
        } else if point >= n {
            out.append(digits[..]);
            for i in point - n { out.push('0'); }
            out.append(".0");
        } else {
            out.append(digits[..point]);
            out.push('.');
            out.append(digits[point..]);
        }
    } else {
        out.push(digits[0]);
        if n > 1 { out.push('.'); out.append(digits[1..]); }
        out.push('e');
        out.push(if exp10 < 0 { '-' } else { '+' });
        let e = abs(exp10);
        if e < 10 { out.push('0'); }
        format(out, e);
    }
}

// Whether the digits times 10^e read back as mag.
fn reads_back(d: const u8[:], e: i64, mag: f64) -> bool {
    var s: u8[..64] = [];
    s.append(d);
    s.push('e');
    var t: u8[>..] = [];
    format(t, e);
    s.append(t);
    var used = 0;
    return c_strtod(s[..], used) == mag;
}

// frexp(mag) == 0.5: mag is a power of two.
fn frexp_half(mag: f64) -> bool {
    var m = mag;
    while m >= 1.0 { m *= 0.5; }
    while m < 0.5 { m *= 2.0; }
    return m == 0.5;
}

// A float as json.dumps writes it: NaN and Infinity, which JSON lacks.
fn float_json(out: u8[>..]&, v: f64) {
    if v != v { out.append("NaN"); return; }
    if v - v != 0.0 { out.append(if v < 0.0 { "-Infinity" } else { "Infinity" }); return; }
    float_repr(out, v);
}

// s as a JSON string, with value_to_json's escapes.
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

// value_to_json: v as JSON, its objects' keys in their order. indent < 0
// writes one line; otherwise each item on a line of its own.
recursive fn to_json(out: u8[>..]&, v: i64, lvl: i64, indent: i64, item_sep: const u8[:], key_sep: const u8[:]) {
    let x = vals[v];
    if x.kind == K_NONE || x.kind == K_UNDEF { out.append("null"); return; }
    if x.kind == K_BOOL { out.append(if x.i != 0 { "true" } else { "false" }); return; }
    if x.kind == K_INT { format(out, x.i); return; }
    if x.kind == K_FLOAT { float_json(out, x.f); return; }
    if x.kind == K_STRING { json_string(out, sbytes(v)); return; }
    let is_arr = x.kind == K_ARRAY || x.kind == K_TUPLE;
    if !is_arr && x.kind != K_OBJECT { out.append("null"); return; }
    if lvl >= LIMIT { fail(str("value is nested more than ", LIMIT, " levels deep")); }
    out.push(if is_arr { '[' } else { '{' });
    if x.n > 0 {
        if indent >= 0 { out.push('\n'); }
        for k in x.n {
            if indent > 0 { for i in (lvl + 1) * indent { out.push(' '); } }
            if is_arr {
                to_json(out, items[x.a + k], lvl + 1, indent, item_sep, key_sep);
            } else {
                let key = items[x.a + 2 * k];
                if vals[key].kind == K_FLOAT {
                    var t: u8[>..] = [];
                    float_json(t, vals[key].f);
                    json_string(out, t);
                } else {
                    json_key(out, key);
                }
                out.append(key_sep);
                to_json(out, items[x.a + 2 * k + 1], lvl + 1, indent, item_sep, key_sep);
            }
            if k < x.n - 1 { out.append(item_sep); }
            if indent >= 0 { out.push('\n'); }
        }
        if indent > 0 { for i in lvl * indent { out.push(' '); } }
    }
    out.push(if is_arr { ']' } else { '}' });
}

// An object key as JSON: its as_string().
fn json_key(out: u8[>..]&, key: i64) {
    let s = as_string(key);
    json_string(out, sbytes(s));
}

// value_to_string_repr: a string in quotes (as JSON if it holds a '),
// anything else as its repr.
recursive fn write_repr(out: u8[>..]&, v: i64) {
    let x = vals[v];
    if x.kind == K_STRING {
        let s = sbytes(v);
        if find(s, "'") >= 0 { json_string(out, s); } else { out.push('\''); out.append(s); out.push('\''); }
        return;
    }
    write_text(out, v);
}

// as_string() for what is not a string, as_repr() for what has no string.
recursive fn write_text(out: u8[>..]&, v: i64) {
    let x = vals[v];
    if x.kind == K_STRING { out.append(sbytes(v)); return; }
    if x.kind == K_INT { format(out, x.i); return; }
    if x.kind == K_FLOAT { format_float(out, x.f); return; }
    if x.kind == K_BOOL { out.append(if x.i != 0 { "True" } else { "False" }); return; }
    if x.kind == K_NONE { out.append("None"); return; }
    if x.kind == K_ARRAY || x.kind == K_TUPLE {
        let t = x.kind == K_TUPLE;
        out.push(if t { '(' } else { '[' });
        for k in x.n {
            if k > 0 { out.append(", "); }
            write_repr(out, items[x.a + k]);
        }
        if t && x.n == 1 { out.push(','); }
        out.push(if t { ')' } else { ']' });
        return;
    }
    if x.kind == K_OBJECT {
        out.push('{');
        for k in x.n {
            if k > 0 { out.append(", "); }
            write_repr(out, items[x.a + 2 * k]);
            out.append(": ");
            write_repr(out, items[x.a + 2 * k + 1]);
        }
        out.push('}');
        return;
    }
    if x.kind == K_FUNC {
        out.append("Function<");
        if x.name >= 0 { out.append(sbytes(x.name)); }
        out.append(">(");
        if x.b >= 0 { write_text(out, x.b); }
        out.push(')');
        return;
    }
    let t = tname(v);
    out.append(t);
}

// as_string(): the value itself for a string, a new string of its text for
// the rest; undefined, functions and keyword arguments have none.
fn as_string(v: i64) -> i64 {
    let k = vals[v].kind;
    if k == K_STRING { return v; }
    if k == K_UNDEF || k == K_FUNC || k == K_KWARG { fail(str(tname(v), " is not a string value")); }
    var out: u8[>..] = [];
    write_text(out, v);
    return mk_str(out, false);
}

//
// JSON in
//

// The JSON value in s as a value, its strings marked as input when mark is
// set (from_json); fails on malformed JSON.
fn from_json(s: const u8[:], mark: bool) -> i64 {
    var pos = skip_ws(s, 0);
    let v = parse_json(s, pos, mark, 0);
    pos = skip_ws(s, pos);
    if pos != s.len { fail_plain(str("[json.exception.parse_error] syntax error at byte ", pos + 1)); }
    return v;
}

fn skip_ws(s: const u8[:], at: i64) -> i64 {
    var p = at;
    while p < s.len && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t') { p++; }
    return p;
}

recursive fn parse_json(s: const u8[:], pos: i64&, mark: bool, depth: i64) -> i64 {
    guard pos < s.len else { fail_plain("[json.exception.parse_error] unexpected end of input"); return -1; };
    let c = s[pos];
    if c == '{' || c == '[' {
        if depth >= LIMIT { fail_plain(str("JSON is nested more than ", LIMIT, " levels deep")); }
        let is_obj = c == '{';
        let v = if is_obj { mk_object() } else { mk_array() };
        pos = skip_ws(s, pos + 1);
        let close: u8 = if is_obj { '}' } else { ']' };
        if pos < s.len && s[pos] == close { pos++; return v; }
        loop {
            if is_obj {
                guard pos < s.len && s[pos] == '"' else { fail_plain(str("[json.exception.parse_error] expected a key at byte ", pos + 1)); return -1; };
                let key = parse_json_string(s, pos, false);
                pos = skip_ws(s, pos);
                guard pos < s.len && s[pos] == ':' else { fail_plain(str("[json.exception.parse_error] expected ':' at byte ", pos + 1)); return -1; };
                pos = skip_ws(s, pos + 1);
                let item = parse_json(s, pos, mark, depth + 1);
                obj_set(v, key, item);
            } else {
                let item = parse_json(s, pos, mark, depth + 1);
                arr_push(v, item);
            }
            pos = skip_ws(s, pos);
            guard pos < s.len else { fail_plain("[json.exception.parse_error] unexpected end of input"); return -1; };
            if s[pos] == ',' { pos = skip_ws(s, pos + 1); continue; }
            guard s[pos] == close else { fail_plain(str("[json.exception.parse_error] syntax error at byte ", pos + 1)); return -1; };
            pos++;
            return v;
        }
    }
    if c == '"' { return parse_json_string(s, pos, mark); }
    if starts_with(s[pos..], "true") { pos += 4; return mk_bool(true); }
    if starts_with(s[pos..], "false") { pos += 5; return mk_bool(false); }
    if starts_with(s[pos..], "null") { pos += 4; return mk_none(); }
    return parse_json_number(s, pos);
}

// The string whose opening quote is at s[pos], its escapes decoded.
fn parse_json_string(s: const u8[:], pos: i64&, mark: bool) -> i64 {
    let off = text.len;
    var p = pos + 1;
    loop {
        guard p < s.len else { fail_plain("[json.exception.parse_error] unterminated string"); };
        let c = s[p];
        p++;
        if c == '"' { break; }
        if c != '\\' { text.push(c); continue; }
        guard p < s.len else { fail_plain("[json.exception.parse_error] unterminated string"); };
        let e = s[p];
        p++;
        if e == '"' || e == '\\' || e == '/' { text.push(e); }
        else if e == 'n' { text.push('\n'); }
        else if e == 't' { text.push('\t'); }
        else if e == 'r' { text.push('\r'); }
        else if e == 'b' { text.push(8); }
        else if e == 'f' { text.push(12); }
        else if e == 'u' {
            var cp = hex4(s, p);
            p += 4;
            if cp >= 0xD800 && cp < 0xDC00 && p + 6 <= s.len && s[p] == '\\' && s[p + 1] == 'u' {
                let lo = hex4(s, p + 2);
                if lo >= 0xDC00 && lo < 0xE000 {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    p += 6;
                }
            }
            push_utf8(text, cp);
        }
        else { fail_plain("[json.exception.parse_error] invalid escape"); }
    }
    pos = p;
    let len = text.len - off;
    parts.push(Part { off: off, len: len, input: mark });
    return new_val(Val { kind: K_STRING, i: len, a: parts.len - 1, n: 1, flat: off, .. });
}

fn hex4(s: const u8[:], at: i64) -> i64 {
    guard at + 4 <= s.len else { fail_plain("[json.exception.parse_error] invalid \\u escape"); return 0; };
    var cp = 0;
    for i in 4 {
        let c = s[at + i];
        var d = -1;
        if c >= '0' && c <= '9' { d = c - '0'; }
        else if c >= 'a' && c <= 'f' { d = c - 'a' + 10; }
        else if c >= 'A' && c <= 'F' { d = c - 'A' + 10; }
        if d < 0 { fail_plain("[json.exception.parse_error] invalid \\u escape"); }
        cp = cp * 16 + d;
    }
    return cp;
}

// A number: an int when it has no fraction or exponent and fits 64 bits
// (one above i64's range wraps, as nlohmann's unsigned get<int64_t> does),
// a float otherwise.
fn parse_json_number(s: const u8[:], pos: i64&) -> i64 {
    var p = pos;
    if p < s.len && s[p] == '-' { p++; }
    let digits = p;
    while p < s.len && s[p] >= '0' && s[p] <= '9' { p++; }
    if p == digits { fail_plain(str("[json.exception.parse_error] syntax error at byte ", pos + 1)); }
    var is_int = true;
    if p < s.len && s[p] == '.' {
        is_int = false;
        p++;
        while p < s.len && s[p] >= '0' && s[p] <= '9' { p++; }
    }
    if p < s.len && (s[p] == 'e' || s[p] == 'E') {
        is_int = false;
        p++;
        if p < s.len && (s[p] == '+' || s[p] == '-') { p++; }
        while p < s.len && s[p] >= '0' && s[p] <= '9' { p++; }
    }
    let num = s[pos..p];
    pos = p;
    if is_int {
        let v, ok = parse_int(num);
        if ok { return mk_int(v); }
        if num[0] != '-' {
            var u: u64 = 0;
            var fits = true;
            for c in num {
                let d = (c - '0') as u64;
                if u > (18446744073709551615 - d) / 10 { fits = false; break; }
                u = u * 10 + d;
            }
            if fits { return mk_int(u as! i64); }
        }
    }
    var used = 0;
    return mk_float(c_strtod(num, used));
}

//
// stats
//

// Marks v as read by the template, and with deep, everything in it.
recursive fn mark_used(v: i64, deep: bool, depth: i64) {
    vals[v].used = true;
    if !deep { return; }
    let x = vals[v];
    if (x.kind == K_ARRAY || x.kind == K_TUPLE || x.kind == K_OBJECT) && depth >= LIMIT {
        fail(str("value is nested more than ", LIMIT, " levels deep"));
    }
    if x.kind == K_ARRAY || x.kind == K_TUPLE {
        for k in x.n { mark_used(items[x.a + k], deep, depth + 1); }
    } else if x.kind == K_OBJECT {
        for k in 2 * x.n { mark_used(items[x.a + k], deep, depth + 1); }
    }
}
