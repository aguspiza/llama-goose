// llama.cpp's PEG parsers: a parser is a tree of combinators (literals,
// sequences, choices, repetitions, character classes, strings up to a
// delimiter...), built with the functions here, that parses a model's
// output into tagged nodes, also while it is still being generated
// (lenient parses say they need more input instead of failing), and that
// writes the GBNF grammar the output must follow.
// The parsers of every arena live in one pool, an arena being a run of it
// (ids are pool indices; GBNF names use the id within the arena). The
// builder adds to the arena begun last.
// port-of: common/peg-parser.cpp, common/peg-parser.h, common/trie.cpp
import std;
import exception;
import .json_schema;

namespace peg;

let RULE_RECURSIVE_LIMIT = 64;

//
// texts
//

struct S { at: i64 = 0, len: i64 = 0 }

var pt: u8[>..] = [];

fn t(x: S) -> const u8[:] { pt[x.at..x.at + x.len] }

fn lit(b: const u8[:]) -> S {
    let at0 = pt.len;
    pt.append(b);
    return S { at: at0, len: b.len };
}

//
// errors
//

// The errors are json_schema's (one handler takes both); err has the
// message once a public function returns false.
var err: u8[>..<] = [];

fn fail(msg: const u8[:]) { json_schema::fail(msg); }

//
// UTF-8 (common_parse_utf8_codepoint)
//

let UTF8_SUCCESS = 0;
let UTF8_INCOMPLETE = 1;
let UTF8_INVALID = 2;

// The status, code point and bytes of the sequence at s[at].
fn utf8_at(s: const u8[:], at: i64) -> i32, u32, i64 {
    if at >= s.len { return UTF8_INCOMPLETE, 0, 0; }
    let c = s[at];
    if (c & 0x80) == 0 { return UTF8_SUCCESS, c, 1; }
    if (c & 0x40) == 0 { return UTF8_INVALID, 0, 1; }
    var n = 0;
    var cp: u32 = 0;
    if (c & 0x20) == 0 { n = 2; cp = c & 0x1F; }
    else if (c & 0x10) == 0 { n = 3; cp = c & 0x0F; }
    else if (c & 0x08) == 0 { n = 4; cp = c & 0x07; }
    else { return UTF8_INVALID, 0, 1; }
    for i in 1..n {
        if at + i >= s.len { return UTF8_INCOMPLETE, 0, if n == 2 { 1 } else { i }; }
        if (s[at + i] & 0xC0) != 0x80 { return UTF8_INVALID, 0, if n == 2 { 1 } else { i }; }
        cp = (cp << 6) | (s[at + i] & 0x3F);
    }
    return UTF8_SUCCESS, cp, n;
}

fn put_utf8(out: u8[>..]&, cp: u32) {
    if cp < 0x80 { out.push(cp as! u8); }
    else if cp < 0x800 { out.push((0xC0 | (cp >> 6)) as! u8); out.push((0x80 | (cp & 0x3F)) as! u8); }
    else if cp < 0x10000 {
        out.push((0xE0 | (cp >> 12)) as! u8);
        out.push((0x80 | ((cp >> 6) & 0x3F)) as! u8);
        out.push((0x80 | (cp & 0x3F)) as! u8);
    } else {
        out.push((0xF0 | (cp >> 18)) as! u8);
        out.push((0x80 | ((cp >> 12) & 0x3F)) as! u8);
        out.push((0x80 | ((cp >> 6) & 0x3F)) as! u8);
        out.push((0x80 | (cp & 0x3F)) as! u8);
    }
}

//
// tries (common_trie, common_aho_corasick)
//

// A trie node: its first child and next sibling (children by code point),
// its code point, and the pattern ending at it.
struct TNode { first: i64 = -1, next: i64 = -1, cp: u32 = 0, pattern: i32 = -1 }

var tn: TNode[>..<] = [];
var n_patterns: i32 = 0;

fn trie_clear() {
    tn.clear();
    tn.push(TNode { .. });
    n_patterns = 0;
}

fn trie_find(n: i64, cp: u32) -> i64 {
    var c = tn[n].first;
    while c >= 0 && tn[c].cp < cp { c = tn[c].next; }
    if c >= 0 && tn[c].cp == cp { return c; }
    return -1;
}

fn trie_child(n: i64, cp: u32) -> i64 {
    var prev = -1;
    var c = tn[n].first;
    while c >= 0 && tn[c].cp < cp { prev = c; c = tn[c].next; }
    if c >= 0 && tn[c].cp == cp { return c; }
    tn.push(TNode { cp: cp, next: c, .. });
    let nc = tn.len - 1;
    if prev < 0 { tn[n].first = nc; } else { tn[prev].next = nc; }
    return nc;
}

// A word, as code points up to the first that does not decode.
fn trie_insert(word: const u8[:]) {
    var cur = 0;
    var pos = 0;
    while pos < word.len {
        let st, cp, n = utf8_at(word, pos);
        if st != UTF8_SUCCESS { break; }
        cur = trie_child(cur, cp);
        pos += n;
    }
    if tn[cur].pattern < 0 { tn[cur].pattern = n_patterns; n_patterns++; }
}

let NO_MATCH = 0;
let PARTIAL_MATCH = 1;
let COMPLETE_MATCH = 2;

// Whether a word starts at s[start] (check_at).
fn trie_check_at(s: const u8[:], start: i64) -> i32 {
    var cur = 0;
    var pos = start;
    while pos < s.len {
        let st, cp, n = utf8_at(s, pos);
        if st != UTF8_SUCCESS { break; }
        let c = trie_find(cur, cp);
        if c < 0 { return NO_MATCH; }
        cur = c;
        pos += n;
        if tn[cur].pattern >= 0 { return COMPLETE_MATCH; }
    }
    return if cur != 0 { PARTIAL_MATCH } else { NO_MATCH };
}

// The Aho-Corasick automaton of the trie: failure links, longest match per
// state, and the alphabet in order.
var ac_fail: i64[>..<] = [];
var ac_match: i32[>..<] = [];
var ac_alpha: u32[>..<] = [];

fn ac_build() {
    let n = tn.len;
    ac_fail.clear();
    ac_match.clear();
    for i in n { ac_fail.push(0); ac_match.push(-1); }
    var order: i64[>..] = [];
    order.push(0);
    var head = 0;
    while head < order.len {
        let u = order[head];
        head++;
        var v = tn[u].first;
        while v >= 0 {
            if u != 0 {
                var f = ac_fail[u];
                while f != 0 && trie_find(f, tn[v].cp) < 0 { f = ac_fail[f]; }
                let it = trie_find(f, tn[v].cp);
                ac_fail[v] = if it >= 0 && it != v { it } else { 0 };
            }
            order.push(v);
            v = tn[v].next;
        }
    }
    for u in order {
        ac_match[u] = if tn[u].pattern >= 0 { tn[u].pattern } else if u != 0 { ac_match[ac_fail[u]] } else { -1 };
    }
    ac_alpha.clear();
    for k in 1..n { ac_alpha.push(tn[k].cp); }
    stable_sort(ac_alpha) { a, b => a < b };
    var w = 0;
    for k in ac_alpha.len {
        if k == 0 || ac_alpha[k] != ac_alpha[w - 1] { ac_alpha[w] = ac_alpha[k]; w++; }
    }
    ac_alpha.resize(w);
}

fn ac_next(state: i64, cp: u32) -> i64 {
    var s = state;
    while s != 0 && trie_find(s, cp) < 0 { s = ac_fail[s]; }
    let it = trie_find(s, cp);
    return if it >= 0 { it } else { 0 };
}

//
// parsers
//

let P_EPSILON = 0;
let P_START = 1;
let P_END = 2;
let P_LITERAL = 3;
let P_SEQUENCE = 4;
let P_CHOICE = 5;
let P_REPETITION = 6;
let P_AND = 7;
let P_NOT = 8;
let P_ANY = 9;
let P_SPACE = 10;
let P_CHARS = 11;
let P_STRING = 12;
let P_UNTIL = 13;
let P_SCHEMA = 14;
let P_RULE = 15;
let P_REF = 16;
let P_ATOMIC = 17;
let P_TAG = 18;
let P_GBNF = 19;
let P_AC = 20;

// A parser. child: its one child. kids_at, kids_n: a sequence's or choice's
// children (in kids), an until's or ac's delimiters (in delims), a chars'
// ranges (in ranges). s: a literal, a chars' pattern, a rule's or ref's
// name, a tag, a gbnf's grammar, a schema's name.
struct P {
    kind: u8,
    child: i64 = -1,
    kids_at: i64 = 0,
    kids_n: i64 = 0,
    lo: i64 = 0,
    hi: i64 = -1,
    s: S = S { .. },
    negated: bool = false,
    delim: u8 = 0,
    trigger: bool = false,
    node: i64 = -1,
    raw: bool = false,
}

struct CharRange { lo: u32, hi: u32 }
struct Rule { arena: i64, name: S, id: i64 }
struct Arena { base: i64, root: i64 = -1 }

var ps: P[>..] = [];
var kids: i64[>..] = [];
var delims: S[>..] = [];
var ranges: CharRange[>..] = [];
var rules: Rule[>..] = [];
var arenas: Arena[>..] = [];
var cur = -1;                   // the arena being built
var scratch: i64[>..<] = [];

// Forgets every parser.
fn reset() {
    pt.clear();
    ps.clear();
    kids.clear();
    delims.clear();
    ranges.clear();
    rules.clear();
    arenas.clear();
    cur = -1;
}

// Starts an arena; the builder functions add to it.
fn begin() -> i64 {
    arenas.push(Arena { base: ps.len, .. });
    cur = arenas.len - 1;
    return cur;
}

fn add(p: P) -> i64 { ps.push(p); return ps.len - 1; }

fn find_rule(arena: i64, name: const u8[:]) -> i64 {
    for k in rules.len {
        let r = rules[rules.len - 1 - k];
        if r.arena == arena && t(r.name) == name { return rules.len - 1 - k; }
    }
    return -1;
}

fn set_rule(name: S, id: i64) {
    let k = find_rule(cur, t(name));
    if k >= 0 { rules[k].id = id; } else { rules.push(Rule { arena: cur, name: name, id: id }); }
}

fn has_rule(name: const u8[:]) -> bool { find_rule(cur, name) >= 0 }

fn eps() -> i64 { add(P { kind: P_EPSILON, .. }) }
fn start() -> i64 { add(P { kind: P_START, .. }) }
fn end() -> i64 { add(P { kind: P_END, .. }) }
fn literal(s: const u8[:]) -> i64 { let x = lit(s); return add(P { kind: P_LITERAL, s: x, .. }); }
fn any() -> i64 { add(P { kind: P_ANY, .. }) }
fn space() -> i64 { add(P { kind: P_SPACE, .. }) }
fn repeat(p: i64, lo: i64, hi: i64) -> i64 { add(P { kind: P_REPETITION, child: p, lo: lo, hi: hi, .. }) }
fn one_or_more(p: i64) -> i64 { repeat(p, 1, -1) }
fn zero_or_more(p: i64) -> i64 { repeat(p, 0, -1) }
fn optional(p: i64) -> i64 { repeat(p, 0, 1) }
fn peek(p: i64) -> i64 { add(P { kind: P_AND, child: p, .. }) }
fn negate(p: i64) -> i64 { add(P { kind: P_NOT, child: p, .. }) }
fn atomic(p: i64) -> i64 { add(P { kind: P_ATOMIC, child: p, .. }) }
fn tag(name: const u8[:], p: i64) -> i64 { let x = lit(name); return add(P { kind: P_TAG, child: p, s: x, .. }); }
fn gbnf(p: i64, grammar: const u8[:]) -> i64 { let x = lit(grammar); return add(P { kind: P_GBNF, child: p, s: x, .. }); }
fn string_content(delimiter: u8) -> i64 { add(P { kind: P_STRING, delim: delimiter, .. }) }
fn ref(name: const u8[:]) -> i64 { let x = lit(name); return add(P { kind: P_REF, s: x, .. }); }

// A sequence (choice) of the parsers, those that are sequences (choices)
// themselves flattened into it.
fn flat(kind: u8, items: const i64[:]) -> i64 {
    let at0 = kids.len;
    for id in items {
        if ps[id].kind == kind {
            let c = ps[id];
            for k in c.kids_n { kids.push(kids[c.kids_at + k]); }
        } else {
            kids.push(id);
        }
    }
    return add(P { kind: kind, kids_at: at0, kids_n: kids.len - at0, .. });
}

fn seq(items: const i64[:]) -> i64 { flat(P_SEQUENCE, items) }
fn choice(items: const i64[:]) -> i64 { flat(P_CHOICE, items) }
fn seq2(a: i64, b: i64) -> i64 { seq([a, b]) }
fn choice2(a: i64, b: i64) -> i64 { choice([a, b]) }
// a << b: a, spaces, b
fn spaced(a: i64, b: i64) -> i64 { let sp = space(); return seq([a, sp, b]); }

fn until_one_of(ds: const u8[:][:]) -> i64 {
    let at0 = delims.len;
    for d in ds { delims.push(lit(d)); }
    return add(P { kind: P_UNTIL, kids_at: at0, kids_n: ds.len, .. });
}
fn until(d: const u8[:]) -> i64 { until_one_of([d]) }
fn rest() -> i64 { add(P { kind: P_UNTIL, kids_at: delims.len, kids_n: 0, .. }) }

fn ac_one_of(p: i64, ds: const u8[:][:]) -> i64 {
    if ds.len == 0 { abort("ac parser requires at least one delimiter"); }
    let at0 = delims.len;
    for d in ds { delims.push(lit(d)); }
    return add(P { kind: P_AC, child: p, kids_at: at0, kids_n: ds.len, .. });
}
fn ac(p: i64, d: const u8[:]) -> i64 { ac_one_of(p, [d]) }

fn is_hex_digit(c: u8) -> bool { (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') }

fn parse_hex_escape(s: const u8[:], pos: i64, n: i64) -> u32, i64 {
    if pos + n > s.len { return 0, 0; }
    var v: u32 = 0;
    for i in n {
        let c = s[pos + i];
        if !is_hex_digit(c) { return 0, 0; }
        v <<= 4;
        if c >= 'a' && c <= 'f' { v += c - 'a' + 10; }
        else if c >= 'A' && c <= 'F' { v += c - 'A' + 10; }
        else { v += c - '0'; }
    }
    return v, n;
}

fn parse_class_char(s: const u8[:], pos: i64) -> u32, i64 {
    if s[pos] == '\\' && pos + 1 < s.len {
        let e = s[pos + 1];
        if e == 'x' || e == 'u' || e == 'U' {
            let n = if e == 'x' { 2 } else if e == 'u' { 4 } else { 8 };
            let v, used = parse_hex_escape(s, pos + 2, n);
            if used > 0 { return v, 2 + used; }
            return e, 2;
        }
        if e == 'n' { return '\n', 2; }
        if e == 't' { return '\t', 2; }
        if e == 'r' { return '\r', 2; }
        return e, 2;
    }
    return s[pos], 1;
}

// A character class [a-z...] repeated lo..hi times (hi -1: unbounded).
fn chars(classes: const u8[:], lo: i64, hi: i64) -> i64 {
    var a = 0;
    var b = classes.len;
    if b > 0 && classes[0] == '[' { a = 1; }
    if b > a && classes[b - 1] == ']' { b--; }
    let content = classes[a..b];
    var negated = false;
    var i = 0;
    if content.len > 0 && content[0] == '^' { negated = true; i = 1; }
    let at0 = ranges.len;
    while i < content.len {
        let lo_c, n = parse_class_char(content, i);
        i += n;
        if i + 1 < content.len && content[i] == '-' {
            let hi_c, m = parse_class_char(content, i + 1);
            ranges.push(CharRange { lo: lo_c, hi: hi_c });
            i += 1 + m;
        } else {
            ranges.push(CharRange { lo: lo_c, hi: lo_c });
        }
    }
    let x = lit(classes);
    return add(P { kind: P_CHARS, kids_at: at0, kids_n: ranges.len - at0, lo: lo, hi: hi, s: x, negated: negated, .. });
}

// gbnf_format_rule_name
fn clean_name(name: const u8[:]) -> S {
    let at0 = pt.len;
    var in_invalid = false;
    for c in name {
        if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' {
            pt.push(c);
            in_invalid = false;
        } else if !in_invalid {
            pt.push('-');
            in_invalid = true;
        }
    }
    return S { at: at0, len: pt.len - at0 };
}

// A named rule of p; a ref to it.
fn rule(name: const u8[:], p: i64, trigger: bool) -> i64 {
    let nm = clean_name(name);
    let id = add(P { kind: P_RULE, child: p, s: nm, trigger: trigger, .. });
    set_rule(nm, id);
    return add(P { kind: P_REF, s: nm, .. });
}

// The builder-function form of rule: rule_open, then the parser built, then
// rule_close. rule_open returns a ref if the rule is already there (and
// then there is nothing to build), else -1, after a placeholder that lets
// the parser refer to the rule.
fn rule_open(name: const u8[:], trigger: bool) -> i64 {
    let nm = clean_name(name);
    if has_rule(t(nm)) { return add(P { kind: P_REF, s: nm, .. }); }
    let placeholder = any();
    let id = add(P { kind: P_RULE, child: placeholder, s: nm, trigger: trigger, .. });
    set_rule(nm, id);
    return -1;
}

fn rule_close(name: const u8[:], p: i64, trigger: bool) -> i64 {
    let nm = clean_name(name);
    let id = add(P { kind: P_RULE, child: p, s: nm, trigger: trigger, .. });
    set_rule(nm, id);
    return add(P { kind: P_REF, s: nm, .. });
}

fn trigger_rule(name: const u8[:], p: i64) -> i64 { rule(name, p, true) }

// A parser with the schema node its GBNF comes from.
fn schema(p: i64, name: const u8[:], node: i64, raw: bool) -> i64 {
    let x = lit(name);
    return add(P { kind: P_SCHEMA, child: p, s: x, node: node, raw: raw, .. });
}

// Ends the arena: its refs resolved to their rules, and its root.
fn build(root: i64) -> i64 {
    let a = cur;
    let base = arenas[a].base;
    fn resolve(id: i64) -> i64 {
        guard id >= 0 && ps[id].kind == P_REF else { return id; }
        let k = find_rule(a, t(ps[id].s));
        if k < 0 { abort(str("Rule not found: ", t(ps[id].s))); }
        return rules[k].id;
    }
    for id in base..ps.len {
        let k = ps[id].kind;
        if k == P_SEQUENCE || k == P_CHOICE {
            for j in ps[id].kids_n { kids[ps[id].kids_at + j] = resolve(kids[ps[id].kids_at + j]); }
        } else if ps[id].child >= 0 {
            ps[id].child = resolve(ps[id].child);
        }
    }
    arenas[a].root = resolve(root);
    return a;
}

//
// the common parsers
//

fn double_quoted_string() -> i64 {
    let r = rule_open("double-quoted-string", false);
    if r >= 0 { return r; }
    let a = literal("\"");
    let b = string_content('"');
    let c = literal("\"");
    return rule_close("double-quoted-string", seq([a, b, c]), false);
}

fn single_quoted_string() -> i64 {
    let r = rule_open("single-quoted-string", false);
    if r >= 0 { return r; }
    let a = literal("'");
    let b = string_content('\'');
    let c = literal("'");
    return rule_close("single-quoted-string", seq([a, b, c]), false);
}

fn quoted_string() -> i64 {
    let r = rule_open("quoted-string", false);
    if r >= 0 { return r; }
    let a = double_quoted_string();
    let b = single_quoted_string();
    return rule_close("quoted-string", choice([a, b]), false);
}

fn json_number() -> i64 {
    let r = rule_open("json-number", false);
    if r >= 0 { return r; }
    let digit1_9 = chars("[1-9]", 1, 1);
    let digits = chars("[0-9]", 1, -1);
    let zero = literal("0");
    let more = chars("[0-9]", 0, -1);
    let int_part = choice([zero, seq([digit1_9, more])]);
    let dot = literal(".");
    let frac = seq([dot, digits]);
    let e1 = literal("e");
    let e2 = literal("E");
    let sign = optional(chars("[+-]", 1, 1));
    let exp = seq([choice([e1, e2]), sign, digits]);
    // only commit the number when the next character cannot extend it
    let not_cont = negate(chars("[0-9.eE+-]", 1, 1));
    let minus = optional(literal("-"));
    let of = optional(frac);
    let oe = optional(exp);
    return rule_close("json-number", seq([minus, int_part, of, oe, not_cont]), false);
}

fn json_string() -> i64 {
    let r = rule_open("json-string", false);
    if r >= 0 { return r; }
    let a = literal("\"");
    let b = string_content('"');
    let c = literal("\"");
    return rule_close("json-string", seq([a, b, c]), false);
}

fn json_bool() -> i64 {
    let r = rule_open("json-bool", false);
    if r >= 0 { return r; }
    let a = literal("true");
    let b = literal("false");
    return rule_close("json-bool", choice([a, b]), false);
}

fn json_null() -> i64 {
    let r = rule_open("json-null", false);
    if r >= 0 { return r; }
    return rule_close("json-null", literal("null"), false);
}

recursive fn json_object() -> i64 {
    let r = rule_open("json-object", false);
    if r >= 0 { return r; }
    let ws = space();
    let key = json_string();
    let colon = literal(":");
    let value = json();
    let member = seq([key, ws, colon, ws, value]);
    let comma = literal(",");
    let members = seq([member, zero_or_more(seq([ws, comma, ws, member]))]);
    let open = literal("{");
    let close1 = literal("}");
    let close2 = literal("}");
    return rule_close("json-object", seq([open, ws, choice([close1, seq([members, ws, close2])])]), false);
}

recursive fn json_array() -> i64 {
    let r = rule_open("json-array", false);
    if r >= 0 { return r; }
    let ws = space();
    let first = json();
    let comma = literal(",");
    let next = json();
    let elements = seq([first, zero_or_more(seq([ws, comma, ws, next]))]);
    let open = literal("[");
    let close1 = literal("]");
    let close2 = literal("]");
    return rule_close("json-array", seq([open, ws, choice([close1, seq([elements, ws, close2])])]), false);
}

recursive fn json() -> i64 {
    let r = rule_open("json-value", false);
    if r >= 0 { return r; }
    let a = json_object();
    let b = json_array();
    let c = json_string();
    let d = json_number();
    let e = json_bool();
    let f = json_null();
    return rule_close("json-value", choice([a, b, c, d, e, f]), false);
}

fn python_string() -> i64 {
    let r = rule_open("python-string", false);
    if r >= 0 { return r; }
    let a = double_quoted_string();
    let b = single_quoted_string();
    return rule_close("python-string", choice([a, b]), false);
}

fn python_number() -> i64 { json_number() }

fn python_bool() -> i64 {
    let r = rule_open("python-bool", false);
    if r >= 0 { return r; }
    let a = literal("True");
    let b = literal("False");
    return rule_close("python-bool", choice([a, b]), false);
}

fn python_null() -> i64 {
    let r = rule_open("python-none", false);
    if r >= 0 { return r; }
    return rule_close("python-none", literal("None"), false);
}

recursive fn python_dict() -> i64 {
    let r = rule_open("python-dict", false);
    if r >= 0 { return r; }
    let ws = space();
    let key = python_string();
    let colon = literal(":");
    let value = python_value();
    let member = seq([key, ws, colon, ws, value]);
    let comma = literal(",");
    let members = seq([member, zero_or_more(seq([ws, comma, ws, member]))]);
    let open = literal("{");
    let close1 = literal("}");
    let close2 = literal("}");
    return rule_close("python-dict", seq([open, ws, choice([close1, seq([members, ws, close2])]), ws]), false);
}

recursive fn python_array() -> i64 {
    let r = rule_open("python-array", false);
    if r >= 0 { return r; }
    let ws = space();
    let first = python_value();
    let comma = literal(",");
    let next = python_value();
    let elements = seq([first, zero_or_more(seq([comma, ws, next]))]);
    let open = literal("[");
    let close1 = literal("]");
    let close2 = literal("]");
    return rule_close("python-array", seq([open, ws, choice([close1, seq([elements, ws, close2])]), ws]), false);
}

recursive fn python_value() -> i64 {
    let r = rule_open("python-value", false);
    if r >= 0 { return r; }
    let a = python_dict();
    let b = python_array();
    let c = python_string();
    let d = python_number();
    let e = python_bool();
    let f = python_null();
    return rule_close("python-value", choice([a, b, c, d, e, f]), false);
}

// Text in <> or [].
fn marker() -> i64 {
    let a = literal("<");
    let b = until(">");
    let sharp = seq([seq([a, b]), literal(">")]);
    let c = literal("[");
    let d = until("]");
    let square = seq([seq([c, d]), literal("]")]);
    return choice([sharp, square]);
}

fn json_member(key: const u8[:], p: i64) -> i64 {
    let ws = space();
    let k = literal(str("\"", key, "\""));
    let colon = literal(":");
    return seq([k, ws, colon, ws, p]);
}

//
// parsing
//

let FAIL = 0;
let SUCCESS = 1;
let NEED_MORE_INPUT = 2;

let FLAG_LENIENT = 1;
let FLAG_DEBUG = 2;
let FLAG_COMPLETE = 4;

struct Res { kind: i32, start: i64, end: i64 }

struct Invalid { pos: i64, len: i64 }

// An AST node: its rule or tag, its text input[start..end], its children
// (in akids) and the invalid UTF-8 inside it (in ainv).
struct Node {
    rule: S,
    tag: S,
    start: i64,
    end: i64,
    kids_at: i64,
    kids_n: i64,
    partial: bool,
    inv_at: i64,
    inv_n: i64,
}

var input: u8[>..] = [];
var flags = 0;
var nodes: Node[>..] = [];
var akids: i64[>..] = [];
var ainv: Invalid[>..] = [];
var rule_depth: i32[>..] = [];
// a result's nodes and invalid runs, above the marks of its caller
var rn: i64[>..<] = [];
var ri: Invalid[>..<] = [];
var pbase = 0;

fn lenient() -> bool { (flags & FLAG_LENIENT) != 0 }
fn complete() -> bool { (flags & FLAG_COMPLETE) != 0 }

fn res(kind: i32, start: i64, end: i64) -> Res { Res { kind: kind, start: start, end: end } }

fn drop_to(nmark: i64, imark: i64) { rn.resize(nmark); ri.resize(imark); }

fn add_node(rule: S, tag: S, r: Res, nmark: i64, imark: i64) {
    let at0 = akids.len;
    for k in nmark..rn.len { akids.push(rn[k]); }
    let i0 = ainv.len;
    for k in imark..ri.len { ainv.push(ri[k]); }
    nodes.push(Node {
        rule: rule, tag: tag, start: r.start, end: r.end,
        kids_at: at0, kids_n: rn.len - nmark,
        partial: r.kind == NEED_MORE_INPUT,
        inv_at: i0, inv_n: ri.len - imark,
    });
    rn.resize(nmark);
    rn.push(nodes.len - 1);
}

fn escape_sequence(start: i64, pos0: i64, delimiter: u8) -> Res {
    var pos = pos0 + 1;
    if pos >= input.len {
        if !lenient() { return res(FAIL, start, start); }
        return res(NEED_MORE_INPUT, start, pos0);
    }
    let c = input[pos];
    if c == delimiter || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' || c == 'r' || c == 't' {
        return res(SUCCESS, start, pos + 1);
    }
    if c == 'u' {
        pos++;
        for i in 4 {
            if pos >= input.len {
                if !lenient() { return res(FAIL, start, start); }
                return res(NEED_MORE_INPUT, start, pos0);
            }
            if !is_hex_digit(input[pos]) { return res(FAIL, start, start); }
            pos++;
        }
        return res(SUCCESS, start, pos);
    }
    return res(FAIL, start, start);
}

fn exec_until(p: P, start: i64) -> Res {
    trie_clear();
    for k in p.kids_n { trie_insert(t(delims[p.kids_at + k])); }
    var pos = start;
    var last_valid = start;
    while pos < input.len {
        let st, cp, n = utf8_at(input, pos);
        if st == UTF8_INCOMPLETE && lenient() { return res(NEED_MORE_INPUT, start, last_valid); }
        if st != UTF8_SUCCESS {
            // a delimiter cannot start inside bytes that fail to decode
            ri.push(Invalid { pos: pos, len: n });
            pos += n;
            last_valid = pos;
            continue;
        }
        let m = trie_check_at(input, pos);
        if m == COMPLETE_MATCH { return res(SUCCESS, start, pos); }
        if m == PARTIAL_MATCH && !complete() { return res(SUCCESS, start, pos); }
        pos += n;
        last_valid = pos;
    }
    if last_valid == input.len && lenient() { return res(NEED_MORE_INPUT, start, last_valid); }
    return res(SUCCESS, start, last_valid);
}

fn exec_chars(p: P, start: i64) -> Res {
    var pos = start;
    var count = 0;
    while p.hi == -1 || count < p.hi {
        let st, cp, n = utf8_at(input, pos);
        if st == UTF8_INCOMPLETE {
            if count >= p.lo { return res(SUCCESS, start, pos); }
            if !lenient() { return res(FAIL, start, start); }
            return res(NEED_MORE_INPUT, start, pos);
        }
        if st == UTF8_INVALID {
            if count >= p.lo { return res(SUCCESS, start, pos); }
            return res(FAIL, start, start);
        }
        var matches = false;
        for k in p.kids_n {
            let r = ranges[p.kids_at + k];
            if cp >= r.lo && cp <= r.hi { matches = true; break; }
        }
        if p.negated { matches = !matches; }
        if !matches { break; }
        pos += n;
        count++;
    }
    if count < p.lo {
        if pos >= input.len && lenient() { return res(NEED_MORE_INPUT, start, pos); }
        return res(FAIL, start, pos);
    }
    return res(SUCCESS, start, pos);
}

fn exec_string(p: P, start: i64) -> Res {
    var pos = start;
    while pos < input.len {
        let c = input[pos];
        if c == p.delim { return res(SUCCESS, start, pos); }
        if c == '\\' {
            let r = escape_sequence(start, pos, p.delim);
            if r.kind != SUCCESS { return r; }
            pos = r.end;
        } else {
            let st, cp, n = utf8_at(input, pos);
            if st == UTF8_INCOMPLETE {
                if !lenient() { return res(FAIL, start, start); }
                return res(NEED_MORE_INPUT, start, pos);
            }
            if st == UTF8_INVALID { return res(FAIL, start, start); }
            pos += n;
        }
    }
    if !lenient() { return res(FAIL, start, pos); }
    return res(NEED_MORE_INPUT, start, pos);
}

// The parser id at pos: its result, its nodes and invalid runs pushed onto
// rn and ri (none when it fails).
recursive fn exec(id: i64, pos: i64) -> Res {
    let p = ps[id];
    let k = p.kind;
    let nmark = rn.len;
    let imark = ri.len;
    if k == P_EPSILON { return res(SUCCESS, pos, pos); }
    if k == P_START { return res(if pos == 0 { SUCCESS } else { FAIL }, pos, pos); }
    if k == P_END { return res(if pos >= input.len { SUCCESS } else { FAIL }, pos, pos); }
    if k == P_LITERAL {
        let s = t(p.s);
        var q = pos;
        for c in s {
            if q >= input.len {
                if !lenient() { return res(FAIL, pos, pos); }
                return res(NEED_MORE_INPUT, pos, q);
            }
            if input[q] != c { return res(FAIL, pos, pos); }
            q++;
        }
        return res(SUCCESS, pos, q);
    }
    if k == P_SEQUENCE {
        var q = pos;
        for j in p.kids_n {
            let r = exec(kids[p.kids_at + j], q);
            if r.kind == FAIL { drop_to(nmark, imark); return res(FAIL, pos, r.end); }
            if r.kind == NEED_MORE_INPUT { return res(NEED_MORE_INPUT, pos, r.end); }
            q = r.end;
        }
        return res(SUCCESS, pos, q);
    }
    if k == P_CHOICE {
        for j in p.kids_n {
            let r = exec(kids[p.kids_at + j], pos);
            if r.kind != FAIL { return r; }
        }
        return res(FAIL, pos, pos);
    }
    if k == P_REPETITION {
        var q = pos;
        var count = 0;
        while p.hi == -1 || count < p.hi {
            if q >= input.len { break; }
            let cm = rn.len;
            let ci = ri.len;
            let r = exec(p.child, q);
            if r.kind == SUCCESS {
                // an empty match ends it
                if r.end == q { drop_to(cm, ci); break; }
                q = r.end;
                count++;
                continue;
            }
            if r.kind == NEED_MORE_INPUT { return res(NEED_MORE_INPUT, pos, r.end); }
            break;
        }
        if p.lo > 0 && count < p.lo {
            if q >= input.len && lenient() { return res(NEED_MORE_INPUT, pos, q); }
            drop_to(nmark, imark);
            return res(FAIL, pos, q);
        }
        return res(SUCCESS, pos, q);
    }
    if k == P_AND {
        let r = exec(p.child, pos);
        drop_to(nmark, imark);
        return res(r.kind, pos, pos);
    }
    if k == P_NOT {
        let r = exec(p.child, pos);
        drop_to(nmark, imark);
        if r.kind == SUCCESS { return res(FAIL, pos, pos); }
        if r.kind == NEED_MORE_INPUT { return res(NEED_MORE_INPUT, pos, pos); }
        return res(SUCCESS, pos, pos);
    }
    if k == P_ANY {
        let st, cp, n = utf8_at(input, pos);
        if st == UTF8_INCOMPLETE {
            if !lenient() { return res(FAIL, pos, pos); }
            return res(NEED_MORE_INPUT, pos, pos);
        }
        if st == UTF8_INVALID { return res(FAIL, pos, pos); }
        return res(SUCCESS, pos, pos + n);
    }
    if k == P_SPACE {
        var q = pos;
        while q < input.len && is_space(input[q]) { q++; }
        return res(SUCCESS, pos, q);
    }
    if k == P_CHARS { return exec_chars(p, pos); }
    if k == P_STRING { return exec_string(p, pos); }
    if k == P_UNTIL { return exec_until(p, pos); }
    if k == P_RULE {
        // only rules recurse, a recursive rule is entered once per input level
        if rule_depth[p.child - pbase] >= RULE_RECURSIVE_LIMIT {
            fail(str("input is nested more than ", RULE_RECURSIVE_LIMIT, " levels deep"));
        }
        rule_depth[p.child - pbase] += 1;
        let r = exec(p.child, pos);
        rule_depth[p.child - pbase] -= 1;
        if r.kind != FAIL { add_node(p.s, S { .. }, r, nmark, imark); }
        return r;
    }
    if k == P_TAG {
        let r = exec(p.child, pos);
        if r.kind != FAIL { add_node(S { .. }, p.s, r, nmark, imark); }
        return r;
    }
    if k == P_REF {
        let rk = find_rule(cur_parse, t(p.s));
        if rk < 0 { fail(str("Rule not found: ", t(p.s))); }
        return exec(rules[rk].id, pos);
    }
    if k == P_ATOMIC {
        let r = exec(p.child, pos);
        if r.kind == NEED_MORE_INPUT { rn.resize(nmark); }
        return r;
    }
    // schema, gbnf, ac: their child parses
    return exec(p.child, pos);
}

// std::isspace in the C locale
fn is_space(c: u8) -> bool { c == ' ' || (c >= 9 && c <= 13) }

var cur_parse = -1;
var result_nodes: i64[>..] = [];
var result_invalid: Invalid[>..] = [];

// Parses text with arena a's root from start (common_peg_arena::parse):
// the result, its top nodes in result_nodes and the AST in nodes. The AST
// of a parse before is forgotten. false, and the message in err, if the
// input is nested too deep.
fn parse(a: i64, text: const u8[:], parse_flags: i64, start_pos: i64) -> Res, bool {
    input.clear();
    input.append(text);
    return parse_input(a, parse_flags, start_pos);
}

fn parse_input(a: i64, parse_flags: i64, start_pos: i64) -> Res, bool {
    flags = parse_flags;
    nodes.clear();
    akids.clear();
    ainv.clear();
    rn.clear();
    ri.clear();
    result_nodes.clear();
    result_invalid.clear();
    cur_parse = a;
    pbase = arenas[a].base;
    rule_depth.clear();
    let n = if a + 1 < arenas.len { arenas[a + 1].base } else { ps.len };
    for i in pbase..n { rule_depth.push(0); }
    var r = res(FAIL, start_pos, start_pos);
    var ok = false;
    fn body() {
        r = exec(arenas[a].root, start_pos);
        ok = true;
    }
    exception::try<json_schema::Fail>(body) { f => err.clear(); err.append(json_schema::err); };
    if ok {
        for x in rn { result_nodes.push(x); }
        for x in ri { result_invalid.push(x); }
    }
    return r, ok;
}

fn node_text(n: i64) -> const u8[:] {
    let x = nodes[n];
    if x.start >= input.len { return ""; }
    return input[x.start..x.end];
}

fn node_rule(n: i64) -> const u8[:] { t(nodes[n].rule) }
fn node_tag(n: i64) -> const u8[:] { t(nodes[n].tag) }
fn node_child(n: i64, k: i64) -> i64 { akids[nodes[n].kids_at + k] }

// The text with every invalid run replaced by U+FFFD (sanitized_text).
fn sanitized_text(n: i64, out: u8[>..]&) {
    let x = nodes[n];
    if x.inv_n == 0 { out.append(node_text(n)); return; }
    var seg = x.start;
    for k in x.inv_n {
        let inv = ainv[x.inv_at + k];
        out.append(input[seg..inv.pos]);
        out.append("\xEF\xBF\xBD");
        seg = inv.pos + inv.len;
    }
    out.append(input[seg..x.end]);
}

// The first child of n (depth first, up to max_depth levels) with the tag
// (rule), or -1.
recursive fn find_by_tag(n: i64, tag_name: const u8[:], max_depth: i64) -> i64 {
    let x = nodes[n];
    for k in x.kids_n {
        let c = akids[x.kids_at + k];
        if t(nodes[c].tag) == tag_name { return c; }
        if max_depth > 1 {
            let r = find_by_tag(c, tag_name, max_depth - 1);
            if r >= 0 { return r; }
        }
    }
    return -1;
}

recursive fn find_by_rule(n: i64, rule_name: const u8[:], max_depth: i64) -> i64 {
    let x = nodes[n];
    for k in x.kids_n {
        let c = akids[x.kids_at + k];
        if t(nodes[c].rule) == rule_name { return c; }
        if max_depth > 1 {
            let r = find_by_rule(c, rule_name, max_depth - 1);
            if r >= 0 { return r; }
        }
    }
    return -1;
}

// The nodes under the result's, depth first, in order (visit).
fn visit_order(out: i64[>..]&) {
    out.clear();
    var stack: i64[>..] = [];
    for top in result_nodes {
        stack.push(top);
        while stack.len > 0 {
            let c = stack[stack.len - 1];
            stack.pop();
            out.push(c);
            let x = nodes[c];
            for k in x.kids_n { stack.push(akids[x.kids_at + x.kids_n - 1 - k]); }
        }
    }
}

//
// GBNF
//

// Texts on the json_schema arena, as its converter wants them.
fn jlit(b: const u8[:]) -> json_schema::Str { json_schema::lit(b) }
fn jt(x: json_schema::Str) -> const u8[:] { json_schema::t(x) }

fn escape_class_char(out: u8[>..]&, c: u32) {
    if c == '-' || c == ']' || c == '[' || c == '\\' { out.push('\\'); out.push(c as! u8); return; }
    if c == '\n' { out.append("\\n"); return; }
    if c == '\t' { out.append("\\t"); return; }
    if c == '\r' { out.append("\\r"); return; }
    if c >= 0x20 && c <= 0x7E { out.push(c as! u8); return; }
    let hex = "0123456789ABCDEF";
    if c <= 0xFF {
        out.append("\\x");
        out.push(hex[(c >> 4) & 0xF]);
        out.push(hex[c & 0xF]);
    } else if c <= 0xFFFF {
        out.append("\\u");
        for i in 4 { out.push(hex[(c >> ((3 - i) * 4)) & 0xF]); }
    } else {
        out.append("\\U");
        for i in 8 { out.push(hex[(c >> ((7 - i) * 4)) & 0xF]); }
    }
}

fn char_class(out: u8[>..]&, cs: const u32[:], negate_it: bool) {
    out.append(if negate_it { "[^" } else { "[" });
    for c in cs { escape_class_char(out, c); }
    out.push(']');
}

fn state_name(out: u8[>..]&, prefix: const u8[:], s: i64) {
    out.append(prefix);
    if s == 0 { return; }
    out.push('-');
    if s < 10 { out.push('0'); }
    format(out, s);
}

// The rules of the Aho-Corasick automaton of the delimiters (gbnf_ac_grammar):
// including, everything up to and including the first delimiter; else
// everything without any delimiter.
fn ac_grammar(prefix: const u8[:], p: P, including: bool) -> json_schema::Str {
    trie_clear();
    for k in p.kids_n { trie_insert(t(delims[p.kids_at + k])); }
    ac_build();
    var rhs: u8[>..] = [];
    var nm: u8[>..] = [];
    var completing: u32[>..] = [];
    var specific: u32[>..] = [];
    var dests: i64[>..] = [];       // the bucket of each char of the alphabet (-1 none)
    for q in tn.len {
        if ac_match[q] >= 0 { continue; }
        completing.clear();
        specific.clear();
        dests.clear();
        for c in ac_alpha {
            let d = ac_next(q, c);
            if ac_match[d] >= 0 {
                completing.push(c);
                specific.push(c);
                dests.push(-1);
            } else if d != 0 {
                specific.push(c);
                dests.push(d);
            } else {
                dests.push(-1);
            }
        }
        // the buckets by destination state, in order
        var order: i64[>..] = [];
        for d in dests { if d >= 0 { var seen = false; for o in order { if o == d { seen = true; } } if !seen { order.push(d); } } }
        stable_sort(order) { a, b => a < b };
        rhs.clear();
        if !including { rhs.append("|"); }
        var first = true;
        if including && completing.len > 0 { char_class(rhs, completing, false); first = false; }
        var bucket: u32[>..] = [];
        for d in order {
            bucket.clear();
            for c, i in ac_alpha { if dests[i] == d { bucket.push(c); } }
            if including { if !first { rhs.append(" | "); } } else { rhs.push(' '); }
            first = false;
            char_class(rhs, bucket, false);
            rhs.push(' ');
            state_name(rhs, prefix, d);
            if !including { rhs.append(" |"); }
        }
        if including { if !first { rhs.append(" | "); } } else { rhs.push(' '); }
        char_class(rhs, specific, true);
        rhs.push(' ');
        state_name(rhs, prefix, 0);
        nm.clear();
        state_name(nm, prefix, q);
        json_schema::add_rule(jlit(nm), jlit(rhs));
    }
    // an empty delimiter makes the start state terminal
    if ac_match[0] >= 0 { json_schema::add_rule(jlit(prefix), jlit("|")); }
    return jlit(prefix);
}

fn schema_delegates(p: P) -> bool { p.node < 0 || (p.raw && json_schema::may_be_string(p.node)) }

// The parser itself, past tags, atomics and schemas that delegate.
fn effective(id: i64) -> u8 {
    var i = id;
    loop {
        let p = ps[i];
        if p.kind == P_TAG || p.kind == P_ATOMIC { i = p.child; continue; }
        if p.kind == P_SCHEMA && schema_delegates(p) { i = p.child; continue; }
        return p.kind;
    }
}

fn is_group(id: i64) -> bool { let k = effective(id); return k == P_CHOICE || k == P_SEQUENCE; }

fn repetition_suffix(out: u8[>..]&, lo: i64, hi: i64) {
    if lo == 0 && hi == 1 { out.push('?'); return; }
    if lo == 0 && hi == -1 { out.push('*'); return; }
    if lo == 1 && hi == -1 { out.push('+'); return; }
    if hi == -1 { format(out, "{", lo, ",}"); return; }
    if lo == hi {
        if lo == 1 { return; }
        format(out, "{", lo, "}");
        return;
    }
    format(out, "{", lo, ",", hi, "}");
}

var gstack: json_schema::Str[>..<] = [];
var gbase = 0;

// The GBNF of a parser (to_gbnf), on the json_schema arena.
recursive fn to_gbnf(id: i64) -> json_schema::Str {
    let p = ps[id];
    let k = p.kind;
    if k == P_EPSILON || k == P_START || k == P_END || k == P_AND || k == P_NOT { return jlit(""); }
    if k == P_LITERAL { return json_schema::format_literal(jlit(t(p.s))); }
    if k == P_SEQUENCE || k == P_CHOICE {
        let mark = gstack.len;
        for j in p.kids_n {
            let g = to_gbnf(kids[p.kids_at + j]);
            gstack.push(g);
        }
        var out: u8[>..] = [];
        for j in p.kids_n {
            let child = kids[p.kids_at + j];
            let g = jt(gstack[mark + j]);
            if k == P_SEQUENCE {
                if g.len == 0 { continue; }
                if out.len > 0 { out.push(' '); }
                if is_group(child) { out.push('('); out.append(g); out.push(')'); } else { out.append(g); }
            } else {
                if j > 0 { out.append(" | "); }
                if effective(child) == P_CHOICE { out.push('('); out.append(g); out.push(')'); } else { out.append(g); }
            }
        }
        gstack.resize(mark);
        return jlit(out);
    }
    if k == P_REPETITION {
        let g = to_gbnf(p.child);
        var out: u8[>..] = [];
        if is_group(p.child) { out.push('('); out.append(jt(g)); out.push(')'); } else { out.append(jt(g)); }
        repetition_suffix(out, p.lo, p.hi);
        return jlit(out);
    }
    if k == P_ANY { return jlit("."); }
    if k == P_SPACE { return jlit("space"); }
    if k == P_CHARS {
        var out: u8[>..] = [];
        out.append(t(p.s));
        repetition_suffix(out, p.lo, p.hi);
        return jlit(out);
    }
    if k == P_STRING {
        var out: u8[>..] = [];
        out.append("( [^");
        out.push(p.delim);
        out.append("\\\\] | \"\\\\\" ( [");
        out.push(p.delim);
        out.append("\\\\/ bfnrt] | \"u\" [0-9a-fA-F]{4} ) )*");
        return jlit(out);
    }
    if k == P_UNTIL {
        if p.kids_n == 0 { return jlit(".*"); }
        return ac_grammar(str("until-", id - gbase), p, false);
    }
    if k == P_SCHEMA {
        if schema_delegates(p) { return to_gbnf(p.child); }
        return json_schema::add_schema(jlit(t(p.s)), p.node);
    }
    if k == P_RULE || k == P_REF { return jlit(t(p.s)); }
    if k == P_GBNF { return jlit(t(p.s)); }
    if k == P_AC { return ac_grammar(str("ac-", id - gbase), p, true); }
    // tag, atomic
    return to_gbnf(p.child);
}

var reach: S[>..] = [];

fn reached(name: const u8[:]) -> bool {
    for r in reach { if t(r) == name { return true; } }
    return false;
}

// The rules reachable from id (collect_reachable_rules), onto reach.
recursive fn collect(a: i64, id: i64) {
    let p = ps[id];
    let k = p.kind;
    if k == P_SEQUENCE || k == P_CHOICE {
        for j in p.kids_n { collect(a, kids[p.kids_at + j]); }
    } else if k == P_RULE {
        if !reached(t(p.s)) {
            reach.push(p.s);
            collect(a, p.child);
        }
    } else if k == P_REF {
        let rk = find_rule(a, t(p.s));
        if rk < 0 { fail(str("Rule not found: ", t(p.s))); }
        collect(a, rules[rk].id);
    } else if p.child >= 0 {
        collect(a, p.child);
    }
}

// Arena a's rules into the json_schema grammar being built (build_grammar);
// lazy: only the trigger rules and what they reach, with root their choice.
// It throws json_schema's Fail.
fn build_grammar(a: i64, lazy: bool) {
    reach.clear();
    gbase = arenas[a].base;
    if lazy {
        for r in rules {
            guard r.arena == a && ps[r.id].kind == P_RULE && ps[r.id].trigger else { continue; }
            // marked, then what it reaches (its child: the rule itself is marked)
            if !reached(t(r.name)) { reach.push(r.name); }
            collect(a, ps[r.id].child);
        }
    } else {
        collect(a, arenas[a].root);
    }
    for r in rules {
        guard r.arena == a && reached(t(r.name)) else { continue; }
        let p = ps[r.id];
        guard p.kind == P_RULE else { continue; }
        let g = to_gbnf(p.child);
        json_schema::add_rule(jlit(t(p.s)), g);
    }
    if lazy {
        var names: S[>..] = [];
        for r in rules {
            guard r.arena == a && ps[r.id].kind == P_RULE && ps[r.id].trigger else { continue; }
            names.push(ps[r.id].s);
        }
        stable_sort(names) { x, y => json_schema::cmp_bytes(t(x), t(y)) < 0 };
        var out: u8[>..] = [];
        for nm, i in names {
            if i > 0 { out.append(" | "); }
            out.append(t(nm));
        }
        json_schema::add_rule(jlit("root"), jlit(out));
    } else if arenas[a].root >= 0 {
        let g = to_gbnf(arenas[a].root);
        json_schema::add_rule(jlit("root"), g);
    }
}

// The grammar of arena a alone (build_grammar with a builder that only
// adds a's rules): false, and the message in err, if it fails.
fn grammar(a: i64, lazy: bool, out: u8[>..]&) -> bool {
    var ok = false;
    fn body() {
        json_schema::begin_grammar();
        build_grammar(a, lazy);
        json_schema::end_grammar(out);
        ok = true;
    }
    exception::try<json_schema::Fail>(body) { f => err.clear(); err.append(json_schema::err); };
    return ok;
}
