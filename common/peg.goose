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

enum Utf8Status { Success, Incomplete, Invalid }

// The status, code point and bytes of the sequence at s[at].
fn utf8_at(s: const u8[:], at: i64) -> Utf8Status, u32, i64 {
    if at >= s.len { return Utf8Status.Incomplete, 0, 0; }
    let c = s[at];
    if (c & 0x80) == 0 { return Utf8Status.Success, c, 1; }
    if (c & 0x40) == 0 { return Utf8Status.Invalid, 0, 1; }
    var n = 0;
    var cp: u32 = 0;
    if (c & 0x20) == 0 { n = 2; cp = c & 0x1F; }
    else if (c & 0x10) == 0 { n = 3; cp = c & 0x0F; }
    else if (c & 0x08) == 0 { n = 4; cp = c & 0x07; }
    else { return Utf8Status.Invalid, 0, 1; }
    for i in 1..n {
        if at + i >= s.len { return Utf8Status.Incomplete, 0, if n == 2 { 1 } else { i }; }
        if (s[at + i] & 0xC0) != 0x80 { return Utf8Status.Invalid, 0, if n == 2 { 1 } else { i }; }
        cp = (cp << 6) | (s[at + i] & 0x3F);
    }
    return Utf8Status.Success, cp, n;
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

// A symbol: a code point, or SYM_TOKEN and a token id (common_trie::symbol);
// code points come first in their order.
let SYM_TOKEN: u64 = 4294967296;

fn sym_cp(cp: u32) -> u64 { cp }
fn sym_tok(id: i64) -> u64 { SYM_TOKEN + (id as! u32) }
fn sym_is_token(x: u64) -> bool { x >= SYM_TOKEN }
fn sym_value(x: u64) -> i64 { (x % SYM_TOKEN) as! i64 }

// A trie node: its first child and next sibling (children by symbol), its
// symbol, and the pattern (or the value marked) ending at it.
struct TNode { first: i64 = -1, next: i64 = -1, sym: u64 = 0, pattern: i32 = -1 }

var tn: TNode[>..<] = [];
var n_patterns: i32 = 0;
// the token ids the patterns use, sorted: at those the input is read a token at a time
var trie_tokens: i64[>..<] = [];

fn trie_clear() {
    tn.clear();
    tn.push(TNode { .. });
    n_patterns = 0;
    trie_tokens.clear();
}

fn trie_find(n: i64, x: u64) -> i64 {
    var c = tn[n].first;
    while c >= 0 && tn[c].sym < x { c = tn[c].next; }
    if c >= 0 && tn[c].sym == x { return c; }
    return -1;
}

// xs with v at index i, the rest moved up.
fn insert_at(xs: i64[>..<]&, i: i64, v: i64) {
    xs.push(v);
    var j = xs.len - 1;
    while j > i { xs[j] = xs[j - 1]; j--; }
    xs[i] = v;
}

fn has_trie_token(id: i64) -> bool {
    let i = lower_bound(trie_tokens) { it < id };
    return i < trie_tokens.len && trie_tokens[i] == id;
}

// The child of n on symbol x, made if missing (extend).
fn trie_child(n: i64, x: u64) -> i64 {
    if sym_is_token(x) {
        let id = sym_value(x);
        let i = lower_bound(trie_tokens) { it < id };
        if i == trie_tokens.len || trie_tokens[i] != id { insert_at(trie_tokens, i, id); }
    }
    var prev = -1;
    var c = tn[n].first;
    while c >= 0 && tn[c].sym < x { prev = c; c = tn[c].next; }
    if c >= 0 && tn[c].sym == x { return c; }
    tn.push(TNode { sym: x, next: c, .. });
    let nc = tn.len - 1;
    if prev < 0 { tn[n].first = nc; } else { tn[prev].next = nc; }
    return nc;
}

// Marks node as a pattern's end with value; the value already there, if any (mark).
fn trie_mark(node: i64, value: i32) -> i32 {
    if tn[node].pattern < 0 { tn[node].pattern = value; }
    return tn[node].pattern;
}

enum MatchKind { NoMatch, PartialMatch, CompleteMatch }

// The symbol at input[pos] (next): the token there if a pattern uses it,
// with its bytes up to the next token; else a code point. Its status,
// symbol and length.
fn trie_next(pos: i64) -> Utf8Status, u64, i64 {
    if trie_tokens.len > 0 && pos < itok.len {
        let id: i64 = itok[pos];
        if id >= 0 && has_trie_token(id) {
            var end = pos + 1;
            while end < itok.len && itok[end] < 0 { end++; }
            return Utf8Status.Success, sym_tok(id), end - pos;
        }
    }
    let st, cp, n = utf8_at(input, pos);
    return st, sym_cp(cp), n;
}

// Whether a pattern starts at input[start] (check_at).
fn trie_check_at(start: i64) -> MatchKind {
    var cur_node = 0;
    var pos = start;
    while pos < input.len {
        let st, x, n = trie_next(pos);
        if st != Utf8Status.Success { break; }
        let c = trie_find(cur_node, x);
        if c < 0 { return MatchKind.NoMatch; }
        cur_node = c;
        pos += n;
        if tn[cur_node].pattern >= 0 { return MatchKind.CompleteMatch; }
    }
    return if cur_node != 0 { MatchKind.PartialMatch } else { MatchKind.NoMatch };
}

// The Aho-Corasick automaton of the trie: failure links, longest match per
// state, and the alphabet in order.
var ac_fail: i64[>..<] = [];
var ac_match: i32[>..<] = [];
var ac_alpha: u64[>..<] = [];

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
                while f != 0 && trie_find(f, tn[v].sym) < 0 { f = ac_fail[f]; }
                let it = trie_find(f, tn[v].sym);
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
    for k in 1..n { ac_alpha.push(tn[k].sym); }
    stable_sort(ac_alpha) { a, b => a < b };
    var w = 0;
    for k in ac_alpha.len {
        if k == 0 || ac_alpha[k] != ac_alpha[w - 1] { ac_alpha[w] = ac_alpha[k]; w++; }
    }
    ac_alpha.resize(w);
}

fn ac_next(state: i64, x: u64) -> i64 {
    var st = state;
    while st != 0 && trie_find(st, x) < 0 { st = ac_fail[st]; }
    let it = trie_find(st, x);
    return if it >= 0 { it } else { 0 };
}

//
// parsers
//

enum Parser {
    Epsilon,
    Start,
    End,
    Literal,
    Sequence,
    Choice,
    Repetition,
    And,
    Not,
    Any,
    Space,
    Chars,
    String,
    Until,
    Schema,
    Rule,
    Ref,
    Atomic,
    Tag,
    Gbnf,
    Token,       // lo: the token id, s: its piece
    TokenNot,    // one token not among kids[kids_at..kids_at + kids_n] (ids)
}

// A parser. child: its one child (an until's content, with incl). kids_at,
// kids_n: a sequence's or choice's children (in kids), an until's branches
// (in kids, a delimiter and its rest parser, -1 for none, each), a chars'
// ranges (in ranges). s: a literal, a token's piece, a chars' pattern, a
// rule's or ref's name, a tag, a gbnf's grammar, a schema's name. incl: an
// until that consumes its first delimiter and parses that branch's rest
// (through); opt: then the input may end before a delimiter. start_p,
// rest_p: a trigger rule's start and rest.
struct P {
    kind: Parser,
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
    incl: bool = false,
    opt: bool = false,
    start_p: i64 = -1,
    rest_p: i64 = -1,
}

struct CharRange { lo: u32, hi: u32 }
struct Rule { arena: i64, name: S, id: i64 }
struct Arena { base: i64, root: i64 = -1 }

var ps: P[>..] = [];
var kids: i64[>..] = [];
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

fn eps() -> i64 { add(P { kind: Parser.Epsilon, .. }) }
fn start() -> i64 { add(P { kind: Parser.Start, .. }) }
fn end() -> i64 { add(P { kind: Parser.End, .. }) }
fn literal(s: const u8[:]) -> i64 { let x = lit(s); return add(P { kind: Parser.Literal, s: x, .. }); }
fn any() -> i64 { add(P { kind: Parser.Any, .. }) }
fn space() -> i64 { add(P { kind: Parser.Space, .. }) }
fn repeat(p: i64, lo: i64, hi: i64) -> i64 { add(P { kind: Parser.Repetition, child: p, lo: lo, hi: hi, .. }) }
fn one_or_more(p: i64) -> i64 { repeat(p, 1, -1) }
fn zero_or_more(p: i64) -> i64 { repeat(p, 0, -1) }
fn optional(p: i64) -> i64 { repeat(p, 0, 1) }
fn peek(p: i64) -> i64 { add(P { kind: Parser.And, child: p, .. }) }
fn negate(p: i64) -> i64 { add(P { kind: Parser.Not, child: p, .. }) }
fn atomic(p: i64) -> i64 { add(P { kind: Parser.Atomic, child: p, .. }) }
fn tag(name: const u8[:], p: i64) -> i64 { let x = lit(name); return add(P { kind: Parser.Tag, child: p, s: x, .. }); }
fn gbnf(p: i64, grammar: const u8[:]) -> i64 { let x = lit(grammar); return add(P { kind: Parser.Gbnf, child: p, s: x, .. }); }
fn string_content(delimiter: u8) -> i64 { add(P { kind: Parser.String, delim: delimiter, .. }) }
fn ref(name: const u8[:]) -> i64 { let x = lit(name); return add(P { kind: Parser.Ref, s: x, .. }); }

// A sequence (choice) of the parsers, those that are sequences (choices)
// themselves flattened into it.
fn flat(kind: Parser, items: const i64[:]) -> i64 {
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

fn seq(items: const i64[:]) -> i64 { flat(Parser.Sequence, items) }
fn choice(items: const i64[:]) -> i64 { flat(Parser.Choice, items) }
fn seq2(a: i64, b: i64) -> i64 { seq([a, b]) }
fn choice2(a: i64, b: i64) -> i64 { choice([a, b]) }
// a << b: a, spaces, b
fn spaced(a: i64, b: i64) -> i64 { let sp = space(); return seq([a, sp, b]); }

// Everything up to the first delimiter (not consumed): a literal each.
fn until_one_of(ds: const u8[:][:]) -> i64 {
    let at0 = kids.len;
    for d in ds { kids.push(-1); kids.push(-1); }
    for d, i in ds {
        let l = literal(d);
        kids[at0 + 2 * i] = l;
    }
    return add(P { kind: Parser.Until, kids_at: at0, kids_n: ds.len, .. });
}
fn until(d: const u8[:]) -> i64 { until_one_of([d]) }
// Everything.
fn rest() -> i64 { add(P { kind: Parser.Until, kids_at: kids.len, kids_n: 0, .. }) }

// Everything up to the delimiter, a sequence of literals and tokens, or of
// choices of them (until(parser)).
fn until_p(delimiter: i64) -> i64 {
    let at0 = kids.len;
    kids.push(delimiter);
    kids.push(-1);
    return add(P { kind: Parser.Until, kids_at: at0, kids_n: 1, .. });
}

// A delimiter of through, and the parser that follows it once it is consumed.
struct Branch { delim: i64, rest: i64 = -1 }

// content, then the first delimiter, then the rest of its branch:
// C (D1 R1 | D2 R2 | ...). optional: the input may end before a delimiter.
fn through_b(content: i64, branches: const Branch[:], optional: bool) -> i64 {
    if branches.len == 0 { abort("through requires at least one branch"); }
    let at0 = kids.len;
    for b in branches { kids.push(b.delim); kids.push(b.rest); }
    return add(P { kind: Parser.Until, child: content, kids_at: at0, kids_n: branches.len, incl: true, opt: optional, .. });
}
fn through(content: i64, delimiter: i64) -> i64 { through_b(content, [Branch { delim: delimiter }], false) }
fn through_s(content: i64, delimiter: const u8[:]) -> i64 { let l = literal(delimiter); return through(content, l); }

// The special tokens (common_peg_tokens): their pieces and ids, and whether
// the parsers built now know them (a builder made with the tokens).
var tok_text: u8[>..] = [];
var tok_at: i64[>..] = [];
var tok_ids: i64[>..] = [];
var use_tokens = false;

// The special tokens, known to the parsers built from now on.
fn set_tokens(pieces: const u8[:][:], ids: const i64[:]) {
    tok_text.clear();
    tok_at.clear();
    tok_ids.clear();
    for pc, i in pieces {
        tok_at.push(tok_text.len);
        tok_text.append(pc);
        tok_ids.push(ids[i]);
    }
    tok_at.push(tok_text.len);
    use_tokens = true;
}

fn clear_tokens() { tok_text.clear(); tok_at.clear(); tok_ids.clear(); use_tokens = false; }

fn n_tokens() -> i64 { tok_ids.len }
fn token_piece(i: i64) -> const u8[:] { tok_text[tok_at[i]..tok_at[i + 1]] }
fn token_id_at(i: i64) -> i64 { tok_ids[i] }

// The id of a special token's piece, or -1 (token_id); -1 too while the
// parsers being built do not know the tokens.
fn token_id(piece: const u8[:]) -> i64 {
    guard use_tokens else { return -1; }
    for id, i in tok_ids { if tok_text[tok_at[i]..tok_at[i + 1]] == piece { return id; } }
    return -1;
}

// A token; a literal if the piece is not a token the builder knows.
fn token(piece: const u8[:]) -> i64 {
    let id = token_id(piece);
    if id < 0 { return literal(piece); }
    let x = lit(piece);
    return add(P { kind: Parser.Token, lo: id, s: x, .. });
}

// One of the tokens; negate_it: one token that is none of them (each must be known).
fn tokens(pieces: const u8[:][:], negate_it: bool) -> i64 {
    if !negate_it {
        var alts: i64[>..] = [];
        for pc in pieces { alts.push(token(pc)); }
        return choice(alts);
    }
    let at0 = kids.len;
    for pc in pieces {
        let id = token_id(pc);
        if id < 0 { abort(str("token is not registered with the builder: ", pc)); }
        kids.push(id);
    }
    return add(P { kind: Parser.TokenNot, kids_at: at0, kids_n: pieces.len, .. });
}

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
    return add(P { kind: Parser.Chars, kids_at: at0, kids_n: ranges.len - at0, lo: lo, hi: hi, s: x, negated: negated, .. });
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
    let id = add(P { kind: Parser.Rule, child: p, s: nm, trigger: trigger, .. });
    set_rule(nm, id);
    return add(P { kind: Parser.Ref, s: nm, .. });
}

// The builder-function form of rule: rule_open, then the parser built, then
// rule_close. rule_open returns a ref if the rule is already there (and
// then there is nothing to build), else -1, after a placeholder that lets
// the parser refer to the rule.
fn rule_open(name: const u8[:], trigger: bool) -> i64 {
    let nm = clean_name(name);
    if has_rule(t(nm)) { return add(P { kind: Parser.Ref, s: nm, .. }); }
    let placeholder = any();
    let id = add(P { kind: Parser.Rule, child: placeholder, s: nm, trigger: trigger, .. });
    set_rule(nm, id);
    return -1;
}

fn rule_close(name: const u8[:], p: i64, trigger: bool) -> i64 {
    let nm = clean_name(name);
    let id = add(P { kind: Parser.Rule, child: p, s: nm, trigger: trigger, .. });
    set_rule(nm, id);
    return add(P { kind: Parser.Ref, s: nm, .. });
}

fn trigger_rule(name: const u8[:], p: i64) -> i64 { rule(name, p, true) }

// A trigger rule of start then rest: a lazy grammar scans for the start of
// any such rule, then allows only its rest. The start is limited as an
// until's delimiter.
fn trigger_rule_sr(name: const u8[:], start_p: i64, rest_p: i64) -> i64 {
    let nm = clean_name(name);
    let sq = seq2(start_p, rest_p);
    let id = add(P { kind: Parser.Rule, child: sq, s: nm, trigger: true, start_p: start_p, rest_p: rest_p, .. });
    set_rule(nm, id);
    return add(P { kind: Parser.Ref, s: nm, .. });
}

// A parser with the schema node its GBNF comes from.
fn schema(p: i64, name: const u8[:], node: i64, raw: bool) -> i64 {
    let x = lit(name);
    return add(P { kind: Parser.Schema, child: p, s: x, node: node, raw: raw, .. });
}

// Ends the arena: its refs resolved to their rules, and its root.
fn build(root: i64) -> i64 {
    let a = cur;
    let base = arenas[a].base;
    fn resolve(id: i64) -> i64 {
        guard id >= 0 && ps[id].kind == Parser.Ref else { return id; }
        let k = find_rule(a, t(ps[id].s));
        if k < 0 { abort(str("Rule not found: ", t(ps[id].s))); }
        return rules[k].id;
    }
    for id in base..ps.len {
        let k = ps[id].kind;
        if k == Parser.Sequence || k == Parser.Choice {
            for j in ps[id].kids_n { kids[ps[id].kids_at + j] = resolve(kids[ps[id].kids_at + j]); }
        } else if k == Parser.Until {
            for j in 2 * ps[id].kids_n { kids[ps[id].kids_at + j] = resolve(kids[ps[id].kids_at + j]); }
            ps[id].child = resolve(ps[id].child);
        } else if ps[id].child >= 0 {
            ps[id].child = resolve(ps[id].child);
        }
        if k == Parser.Rule && ps[id].start_p >= 0 {
            ps[id].start_p = resolve(ps[id].start_p);
            ps[id].rest_p = resolve(ps[id].rest_p);
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
    // the items of sequence({...}) are made in order
    let e1 = literal("e");
    let e2 = literal("E");
    let e = choice([e1, e2]);
    let sign = optional(chars("[+-]", 1, 1));
    let exp = seq([e, sign, digits]);
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

enum ResultKind { Fail, Success, NeedMoreInput }

let FLAG_LENIENT = 1;
let FLAG_DEBUG = 2;
let FLAG_COMPLETE = 4;

struct Res { kind: ResultKind, start: i64, end: i64 }

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
// the token each byte of input starts, -1 for the rest of a token's piece and for text
var itok: i32[>..] = [];
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

fn res(kind: ResultKind, start: i64, end: i64) -> Res { Res { kind: kind, start: start, end: end } }

fn drop_to(nmark: i64, imark: i64) { rn.resize(nmark); ri.resize(imark); }

fn add_node(rule: S, tag: S, r: Res, nmark: i64, imark: i64) {
    let at0 = akids.len;
    for k in nmark..rn.len { akids.push(rn[k]); }
    let i0 = ainv.len;
    for k in imark..ri.len { ainv.push(ri[k]); }
    nodes.push(Node {
        rule: rule, tag: tag, start: r.start, end: r.end,
        kids_at: at0, kids_n: rn.len - nmark,
        partial: r.kind == ResultKind.NeedMoreInput,
        inv_at: i0, inv_n: ri.len - imark,
    });
    rn.resize(nmark);
    rn.push(nodes.len - 1);
}

fn escape_sequence(start: i64, pos0: i64, delimiter: u8) -> Res {
    var pos = pos0 + 1;
    if pos >= input.len {
        if !lenient() { return res(ResultKind.Fail, start, start); }
        return res(ResultKind.NeedMoreInput, start, pos0);
    }
    let c = input[pos];
    if c == delimiter || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' || c == 'r' || c == 't' {
        return res(ResultKind.Success, start, pos + 1);
    }
    if c == 'u' {
        pos++;
        for i in 4 {
            if pos >= input.len {
                if !lenient() { return res(ResultKind.Fail, start, start); }
                return res(ResultKind.NeedMoreInput, start, pos0);
            }
            if !is_hex_digit(input[pos]) { return res(ResultKind.Fail, start, start); }
            pos++;
        }
        return res(ResultKind.Success, start, pos);
    }
    return res(ResultKind.Fail, start, start);
}

// Node sets of insert_delimiter, a run each, sorted.
var dset: i64[>..<] = [];
struct DRun { at: i64, n: i64 }
var druns: DRun[>..<] = [];

fn dset_add(run_at: i64, node: i64) {
    var i = run_at;
    while i < dset.len && dset[i] < node { i++; }
    if i < dset.len && dset[i] == node { return; }
    insert_at(dset, i, node);
}

// The delimiters parser id matches, into the trie from each node of the run
// dset[at..at + n]: the end nodes, a run at the top of dset (insert_delimiter).
// A choice adds each branch's; a sequence joins every combination.
recursive fn insert_delimiter(id: i64, at: i64, n: i64) -> i64, i64 {
    let p = ps[id];
    let k = p.kind;
    if k == Parser.Tag || k == Parser.Atomic { return insert_delimiter(p.child, at, n); }
    if k == Parser.Literal || k == Parser.Token {
        let out = dset.len;
        for i in n {
            var node = dset[at + i];
            if k == Parser.Token {
                node = trie_child(node, sym_tok(p.lo));
            } else {
                let lt = t(p.s);
                var pos = 0;
                while pos < lt.len {
                    let st, cp, m = utf8_at(lt, pos);
                    if st != Utf8Status.Success { break; }
                    node = trie_child(node, sym_cp(cp));
                    pos += m;
                }
            }
            dset_add(out, node);
        }
        return out, dset.len - out;
    }
    if k == Parser.Choice {
        let rmark = druns.len;
        for j in p.kids_n {
            let a2, n2 = insert_delimiter(kids[p.kids_at + j], at, n);
            druns.push(DRun { at: a2, n: n2 });
        }
        let out = dset.len;
        for r in rmark..druns.len {
            let dr = druns[r];
            for i in dr.n { dset_add(out, dset[dr.at + i]); }
        }
        druns.resize(rmark);
        return out, dset.len - out;
    }
    if k == Parser.Sequence {
        var fa = at;
        var fnn = n;
        for j in p.kids_n {
            let a2, n2 = insert_delimiter(kids[p.kids_at + j], fa, fnn);
            fa = a2;
            fnn = n2;
        }
        return fa, fnn;
    }
    fail("a delimiter may only be a sequence of literals, tokens, and choices of them, possibly wrapped in tags and atomics");
    return at, n;
}

// The trie of an until's delimiters, each end valued by its branch
// (build_until_matcher).
fn build_matcher(p: P) {
    trie_clear();
    for b in p.kids_n {
        let mark = dset.len;
        dset.push(0);
        let a, n = insert_delimiter(kids[p.kids_at + 2 * b], mark, 1);
        for i in n {
            let end = dset[a + i];
            if end == 0 { fail("delimiter must not match the empty string"); }
            // nothing follows a delimiter without incl, so the first branch to claim it wins
            let owner = trie_mark(end, b as! i32);
            if owner != b as! i32 && p.incl {
                fail(str("until branches ", owner, " and ", b, " must not share a delimiter"));
            }
        }
        dset.resize(mark);
    }
}

recursive fn exec_until(p: P, start: i64) -> Res {
    if p.incl { return exec_through(p, start); }
    build_matcher(p);
    var pos = start;
    var last_valid = start;
    while pos < input.len {
        let st, x, n = trie_next(pos);
        if st == Utf8Status.Incomplete && lenient() { return res(ResultKind.NeedMoreInput, start, last_valid); }
        if st != Utf8Status.Success {
            // a delimiter cannot start inside bytes that fail to decode
            ri.push(Invalid { pos: pos, len: n });
            pos += n;
            last_valid = pos;
            continue;
        }
        let m = trie_check_at(pos);
        if m == MatchKind.CompleteMatch { return res(ResultKind.Success, start, pos); }
        if m == MatchKind.PartialMatch && !complete() { return res(ResultKind.Success, start, pos); }
        pos += n;
        last_valid = pos;
    }
    if last_valid == input.len && lenient() { return res(ResultKind.NeedMoreInput, start, last_valid); }
    return res(ResultKind.Success, start, last_valid);
}

// The content, then the first branch whose delimiter and rest follow it,
// C (D1 R1 | D2 R2 | ...) (parse_through).
recursive fn exec_through(p: P, start: i64) -> Res {
    let nmark = rn.len;
    let imark = ri.len;
    let content = exec(p.child, start);
    if content.kind != ResultKind.Success { return content; }
    let pos = content.end;
    let cm = rn.len;
    let cim = ri.len;
    for b in p.kids_n {
        var end = pos;
        var matched = true;
        for which in 2 {
            let id = kids[p.kids_at + 2 * b + which];
            guard id >= 0 else { continue; }
            let r = exec(id, end);
            if r.kind == ResultKind.Fail { matched = false; break; }
            end = r.end;
            if r.kind == ResultKind.NeedMoreInput { return res(ResultKind.NeedMoreInput, start, end); }
        }
        if matched { return res(ResultKind.Success, start, end); }
        drop_to(cm, cim);
    }
    if p.opt && pos == input.len {
        return res(if lenient() { ResultKind.NeedMoreInput } else { ResultKind.Success }, start, pos);
    }
    drop_to(nmark, imark);
    return res(ResultKind.Fail, start, pos);
}

fn exec_chars(p: P, start: i64) -> Res {
    var pos = start;
    var count = 0;
    while p.hi == -1 || count < p.hi {
        let st, cp, n = utf8_at(input, pos);
        if st == Utf8Status.Incomplete {
            if count >= p.lo { return res(ResultKind.Success, start, pos); }
            if !lenient() { return res(ResultKind.Fail, start, start); }
            return res(ResultKind.NeedMoreInput, start, pos);
        }
        if st == Utf8Status.Invalid {
            if count >= p.lo { return res(ResultKind.Success, start, pos); }
            return res(ResultKind.Fail, start, start);
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
        if pos >= input.len && lenient() { return res(ResultKind.NeedMoreInput, start, pos); }
        return res(ResultKind.Fail, start, pos);
    }
    return res(ResultKind.Success, start, pos);
}

fn exec_string(p: P, start: i64) -> Res {
    var pos = start;
    while pos < input.len {
        let c = input[pos];
        if c == p.delim { return res(ResultKind.Success, start, pos); }
        if c == '\\' {
            let r = escape_sequence(start, pos, p.delim);
            if r.kind != ResultKind.Success { return r; }
            pos = r.end;
        } else {
            let st, cp, n = utf8_at(input, pos);
            if st == Utf8Status.Incomplete {
                if !lenient() { return res(ResultKind.Fail, start, start); }
                return res(ResultKind.NeedMoreInput, start, pos);
            }
            if st == Utf8Status.Invalid { return res(ResultKind.Fail, start, start); }
            pos += n;
        }
    }
    if !lenient() { return res(ResultKind.Fail, start, pos); }
    return res(ResultKind.NeedMoreInput, start, pos);
}

// The parser id at pos: its result, its nodes and invalid runs pushed onto
// rn and ri (none when it fails).
recursive fn exec(id: i64, pos: i64) -> Res {
    let p = ps[id];
    let k = p.kind;
    let nmark = rn.len;
    let imark = ri.len;
    if k == Parser.Epsilon { return res(ResultKind.Success, pos, pos); }
    if k == Parser.Start { return res(if pos == 0 { ResultKind.Success } else { ResultKind.Fail }, pos, pos); }
    if k == Parser.End { return res(if pos >= input.len { ResultKind.Success } else { ResultKind.Fail }, pos, pos); }
    if k == Parser.Literal {
        let s = t(p.s);
        var q = pos;
        for c in s {
            if q >= input.len {
                if !lenient() { return res(ResultKind.Fail, pos, pos); }
                return res(ResultKind.NeedMoreInput, pos, q);
            }
            if input[q] != c { return res(ResultKind.Fail, pos, pos); }
            q++;
        }
        return res(ResultKind.Success, pos, q);
    }
    if k == Parser.Token || k == Parser.TokenNot {
        if pos >= input.len {
            if !lenient() { return res(ResultKind.Fail, pos, pos); }
            return res(ResultKind.NeedMoreInput, pos, pos);
        }
        let id: i64 = if pos < itok.len { itok[pos] } else { -1 };
        var ok = false;
        if k == Parser.Token {
            ok = id == p.lo;
        } else {
            ok = id >= 0;
            for j in p.kids_n { if kids[p.kids_at + j] == id { ok = false; } }
        }
        if !ok { return res(ResultKind.Fail, pos, pos); }
        // the rest of the piece, up to the next token or the end
        var q = pos + 1;
        while q < input.len && itok[q] < 0 { q++; }
        return res(ResultKind.Success, pos, q);
    }
    if k == Parser.Sequence {
        var q = pos;
        for j in p.kids_n {
            let r = exec(kids[p.kids_at + j], q);
            if r.kind == ResultKind.Fail { drop_to(nmark, imark); return res(ResultKind.Fail, pos, r.end); }
            if r.kind == ResultKind.NeedMoreInput { return res(ResultKind.NeedMoreInput, pos, r.end); }
            q = r.end;
        }
        return res(ResultKind.Success, pos, q);
    }
    if k == Parser.Choice {
        for j in p.kids_n {
            let r = exec(kids[p.kids_at + j], pos);
            if r.kind != ResultKind.Fail { return r; }
        }
        return res(ResultKind.Fail, pos, pos);
    }
    if k == Parser.Repetition {
        var q = pos;
        var count = 0;
        while p.hi == -1 || count < p.hi {
            if q >= input.len { break; }
            let cm = rn.len;
            let ci = ri.len;
            let r = exec(p.child, q);
            if r.kind == ResultKind.Success {
                // an empty match ends it
                if r.end == q { drop_to(cm, ci); break; }
                q = r.end;
                count++;
                continue;
            }
            if r.kind == ResultKind.NeedMoreInput { return res(ResultKind.NeedMoreInput, pos, r.end); }
            break;
        }
        if p.lo > 0 && count < p.lo {
            if q >= input.len && lenient() { return res(ResultKind.NeedMoreInput, pos, q); }
            drop_to(nmark, imark);
            return res(ResultKind.Fail, pos, q);
        }
        return res(ResultKind.Success, pos, q);
    }
    if k == Parser.And {
        let r = exec(p.child, pos);
        drop_to(nmark, imark);
        return res(r.kind, pos, pos);
    }
    if k == Parser.Not {
        let r = exec(p.child, pos);
        drop_to(nmark, imark);
        if r.kind == ResultKind.Success { return res(ResultKind.Fail, pos, pos); }
        if r.kind == ResultKind.NeedMoreInput { return res(ResultKind.NeedMoreInput, pos, pos); }
        return res(ResultKind.Success, pos, pos);
    }
    if k == Parser.Any {
        let st, cp, n = utf8_at(input, pos);
        if st == Utf8Status.Incomplete {
            if !lenient() { return res(ResultKind.Fail, pos, pos); }
            return res(ResultKind.NeedMoreInput, pos, pos);
        }
        if st == Utf8Status.Invalid { return res(ResultKind.Fail, pos, pos); }
        return res(ResultKind.Success, pos, pos + n);
    }
    if k == Parser.Space {
        var q = pos;
        while q < input.len && is_space(input[q]) { q++; }
        return res(ResultKind.Success, pos, q);
    }
    if k == Parser.Chars { return exec_chars(p, pos); }
    if k == Parser.String { return exec_string(p, pos); }
    if k == Parser.Until { return exec_until(p, pos); }
    if k == Parser.Rule {
        // only rules recurse, a recursive rule is entered once per input level
        if rule_depth[p.child - pbase] >= RULE_RECURSIVE_LIMIT {
            fail(str("input is nested more than ", RULE_RECURSIVE_LIMIT, " levels deep"));
        }
        rule_depth[p.child - pbase] += 1;
        let r = exec(p.child, pos);
        rule_depth[p.child - pbase] -= 1;
        if r.kind != ResultKind.Fail { add_node(p.s, S { .. }, r, nmark, imark); }
        return r;
    }
    if k == Parser.Tag {
        let r = exec(p.child, pos);
        if r.kind != ResultKind.Fail { add_node(S { .. }, p.s, r, nmark, imark); }
        return r;
    }
    if k == Parser.Ref {
        let rk = find_rule(cur_parse, t(p.s));
        if rk < 0 { fail(str("Rule not found: ", t(p.s))); }
        return exec(rules[rk].id, pos);
    }
    if k == Parser.Atomic {
        let r = exec(p.child, pos);
        if r.kind == ResultKind.NeedMoreInput { rn.resize(nmark); }
        return r;
    }
    // schema, gbnf: their child parses
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
    itok.clear();
    itok.resize(input.len, -1);
    return parse_input(a, parse_flags, start_pos);
}

// parse, of text whose bytes start the tokens toks (-1 for none).
fn parse_tokens(a: i64, text: const u8[:], toks: const i32[:], parse_flags: i64, start_pos: i64) -> Res, bool {
    input.clear();
    input.append(text);
    itok.clear();
    itok.append(toks);
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
    var r = res(ResultKind.Fail, start_pos, start_pos);
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

// A token set, <[1,5-9]>, runs of consecutive ids as ranges (gbnf_token_set).
fn token_set(out: u8[>..]&, ids: const i64[:], negate_it: bool) {
    out.append(if negate_it { "!<[" } else { "<[" });
    var i = 0;
    while i < ids.len {
        var j = i;
        while j + 1 < ids.len && ids[j + 1] == ids[j] + 1 { j++; }
        if i > 0 { out.push(','); }
        format(out, ids[i]);
        if j > i { out.push('-'); format(out, ids[j]); }
        i = j + 1;
    }
    out.append("]>");
}

// The alternatives of a set of symbols (gbnf_ac_symbols::alternatives): one
// for the code points, one for the tokens, each followed by next if given;
// onto alts, each its own string in abuf.
struct Alt { at: i64, len: i64 }
var abuf: u8[>..<] = [];

fn push_alt(alts: Alt[>..]&, text: const u8[:]) {
    alts.push(Alt { at: abuf.len, len: text.len });
    abuf.append(text);
}

fn sym_alternatives(alts: Alt[>..]&, syms: const u64[:], next: const u8[:]) {
    var cps: u32[>..] = [];
    var ids: i64[>..] = [];
    for x in syms { if sym_is_token(x) { ids.push(sym_value(x)); } else { cps.push(sym_value(x) as! u32); } }
    var a: u8[>..] = [];
    if cps.len > 0 {
        char_class(a, cps, false);
        if next.len > 0 { a.push(' '); a.append(next); }
        push_alt(alts, a);
    }
    if ids.len > 0 {
        a.clear();
        token_set(a, ids, false);
        if next.len > 0 { a.push(' '); a.append(next); }
        push_alt(alts, a);
    }
}

var reach_tokens: i64[>..<] = [];

// The rules of the Aho-Corasick automaton of the trie (gbnf_ac_grammar).
// Excluding: everything without a pattern. Including: up to the first
// pattern, then the rest rule of that pattern's value (rests, empty for
// none); optional: or nothing more. Returns the start state's name.
fn ac_grammar(prefix: const u8[:], including: bool, optional: bool, rests: const json_schema::Str[:]) -> json_schema::Str {
    ac_build();
    // with only tokens in the alphabet, the grammar steps a whole token at a time
    var token_only = true;
    for x in ac_alpha { if !sym_is_token(x) { token_only = false; } }
    var rhs: u8[>..] = [];
    var nm: u8[>..] = [];
    var dests: i64[>..] = [];       // the destination of each symbol of the alphabet
    var specific: u64[>..] = [];
    var group: u64[>..] = [];
    var order: i64[>..] = [];
    var term_cps: u64[>..] = [];
    var term_ids: u64[>..] = [];
    var alts: Alt[>..] = [];
    var rest_names: Alt[>..] = [];
    var next: u8[>..] = [];
    var fallback: u8[>..] = [];
    for q in tn.len {
        if ac_match[q] >= 0 { continue; }
        dests.clear();
        specific.clear();
        for x in ac_alpha {
            let d = ac_next(q, x);
            dests.push(d);
            if ac_match[d] >= 0 || d != 0 { specific.push(x); }
        }
        // every other symbol goes back to the start
        fallback.clear();
        var spec_cps: u32[>..] = [];
        var spec_ids: i64[>..] = [];
        for x in specific { if sym_is_token(x) { spec_ids.push(sym_value(x)); } else { spec_cps.push(sym_value(x) as! u32); } }
        if token_only {
            token_set(fallback, spec_ids, true);
        } else {
            if spec_cps.len == 0 { fallback.push('.'); } else { char_class(fallback, spec_cps, true); }
            // char rules never match a token named by a token rule: the other named tokens by name
            var others: i64[>..] = [];
            for id in reach_tokens {
                var named = false;
                for sid in spec_ids { if sid == id { named = true; } }
                if !named { others.push(id); }
            }
            if others.len > 0 {
                var f2: u8[>..] = [];
                f2.push('(');
                f2.append(fallback);
                f2.append(" | ");
                token_set(f2, others, false);
                f2.push(')');
                fallback.clear();
                fallback.append(f2);
            }
        }
        fallback.push(' ');
        state_name(fallback, prefix, 0);
        abuf.clear();
        alts.clear();
        // the destinations, in order: those that complete a pattern, then the others
        order.clear();
        for d in dests { var seen = false; for o in order { if o == d { seen = true; } } if !seen { order.push(d); } }
        stable_sort(order) { a, b => a < b };
        if including {
            term_cps.clear();
            term_ids.clear();
            for d in order {
                guard ac_match[d] >= 0 else { continue; }
                group.clear();
                for x, i in ac_alpha { if dests[i] == d { group.push(x); } }
                // the rests of every pattern ending here, the longest and those by suffix links
                rest_names.clear();
                var terminal = false;
                var st = d;
                while st != 0 {
                    let pat = tn[st].pattern;
                    if pat >= 0 {
                        let rname = jt(rests[pat]);
                        if rname.len == 0 {
                            terminal = true;
                        } else {
                            var dup = false;
                            for r in rest_names { if abuf[r.at..r.at + r.len] == rname { dup = true; } }
                            if !dup { push_alt(rest_names, rname); }
                        }
                    }
                    st = ac_fail[st];
                }
                if rest_names.len > 0 {
                    next.clear();
                    if rest_names.len == 1 {
                        next.append(abuf[rest_names[0].at..rest_names[0].at + rest_names[0].len]);
                    } else {
                        next.push('(');
                        for r, i in rest_names {
                            if i > 0 { next.append(" | "); }
                            next.append(abuf[r.at..r.at + r.len]);
                        }
                        next.push(')');
                    }
                    sym_alternatives(alts, group, next);
                }
                if terminal {
                    for x in group { if sym_is_token(x) { term_ids.push(x); } else { term_cps.push(x); } }
                }
            }
            var term: u64[>..] = [];
            term.append(term_cps);
            term.append(term_ids);
            sym_alternatives(alts, term, "");
        }
        for d in order {
            guard ac_match[d] < 0 && d != 0 else { continue; }
            group.clear();
            for x, i in ac_alpha { if dests[i] == d { group.push(x); } }
            next.clear();
            state_name(next, prefix, d);
            sym_alternatives(alts, group, next);
        }
        rhs.clear();
        if including {
            if optional { rhs.append("| "); }
            for a, i in alts {
                rhs.append(abuf[a.at..a.at + a.len]);
                rhs.append(" | ");
            }
            rhs.append(fallback);
        } else {
            rhs.push('|');
            for a in alts {
                rhs.push(' ');
                rhs.append(abuf[a.at..a.at + a.len]);
                rhs.append(" |");
            }
            rhs.push(' ');
            rhs.append(fallback);
        }
        nm.clear();
        state_name(nm, prefix, q);
        json_schema::add_rule(jlit(nm), jlit(rhs));
    }
    return jlit(prefix);
}

fn schema_delegates(p: P) -> bool { p.node < 0 || (p.raw && json_schema::may_be_string(p.node)) }

// The parser itself, past tags, atomics and schemas that delegate.
fn effective(id: i64) -> Parser {
    var i = id;
    loop {
        let p = ps[i];
        if p.kind == Parser.Tag || p.kind == Parser.Atomic { i = p.child; continue; }
        if p.kind == Parser.Schema && schema_delegates(p) { i = p.child; continue; }
        return p.kind;
    }
}

fn is_group(id: i64) -> bool { let k = effective(id); return k == Parser.Choice || k == Parser.Sequence; }

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
var grests: json_schema::Str[>..<] = [];
var gbase = 0;

// The GBNF of a parser (to_gbnf), on the json_schema arena.
recursive fn to_gbnf(id: i64) -> json_schema::Str {
    let p = ps[id];
    let k = p.kind;
    if k == Parser.Epsilon || k == Parser.Start || k == Parser.End || k == Parser.And || k == Parser.Not { return jlit(""); }
    if k == Parser.Literal { return json_schema::format_literal(jlit(t(p.s))); }
    if k == Parser.Token { return jlit(str("<[", p.lo, "]>")); }
    if k == Parser.TokenNot {
        var out: u8[>..] = [];
        token_set(out, kids[p.kids_at..p.kids_at + p.kids_n], true);
        return jlit(out);
    }
    if k == Parser.Sequence || k == Parser.Choice {
        let mark = gstack.len;
        for j in p.kids_n {
            let g = to_gbnf(kids[p.kids_at + j]);
            gstack.push(g);
        }
        var out: u8[>..] = [];
        for j in p.kids_n {
            let child = kids[p.kids_at + j];
            let g = jt(gstack[mark + j]);
            if k == Parser.Sequence {
                if g.len == 0 { continue; }
                if out.len > 0 { out.push(' '); }
                if is_group(child) { out.push('('); out.append(g); out.push(')'); } else { out.append(g); }
            } else {
                if j > 0 { out.append(" | "); }
                if effective(child) == Parser.Choice { out.push('('); out.append(g); out.push(')'); } else { out.append(g); }
            }
        }
        gstack.resize(mark);
        return jlit(out);
    }
    if k == Parser.Repetition {
        let g = to_gbnf(p.child);
        var out: u8[>..] = [];
        if is_group(p.child) { out.push('('); out.append(jt(g)); out.push(')'); } else { out.append(jt(g)); }
        repetition_suffix(out, p.lo, p.hi);
        return jlit(out);
    }
    if k == Parser.Any { return jlit("."); }
    if k == Parser.Space { return jlit("space"); }
    if k == Parser.Chars {
        var out: u8[>..] = [];
        out.append(t(p.s));
        repetition_suffix(out, p.lo, p.hi);
        return jlit(out);
    }
    if k == Parser.String {
        var out: u8[>..] = [];
        out.append("( [^");
        out.push(p.delim);
        out.append("\\\\] | \"\\\\\" ( [");
        out.push(p.delim);
        out.append("\\\\/ bfnrt] | \"u\" [0-9a-fA-F]{4} ) )*");
        return jlit(out);
    }
    if k == Parser.Until {
        if !p.incl {
            if p.kids_n == 0 { return jlit(".*"); }
            build_matcher(p);
            var none: json_schema::Str[>..] = [];
            return ac_grammar(str("until-", id - gbase), false, false, none);
        }
        // the rests first: their grammar may build other matchers
        let mark = grests.len;
        for b in p.kids_n {
            let r = kids[p.kids_at + 2 * b + 1];
            if r < 0 {
                grests.push(jlit(""));
            } else {
                let g = to_gbnf(r);
                let rname = json_schema::add_rule(jlit(str("until-", id - gbase, "-rest-", b)), g);
                grests.push(rname);
            }
        }
        var rs: json_schema::Str[>..] = [];
        for k2 in mark..grests.len { rs.push(grests[k2]); }
        grests.resize(mark);
        build_matcher(p);
        return ac_grammar(str("until-", id - gbase), true, p.opt, rs);
    }
    if k == Parser.Schema {
        if schema_delegates(p) { return to_gbnf(p.child); }
        return json_schema::add_schema(jlit(t(p.s)), p.node);
    }
    if k == Parser.Rule || k == Parser.Ref { return jlit(t(p.s)); }
    if k == Parser.Gbnf { return jlit(t(p.s)); }
    // tag, atomic
    return to_gbnf(p.child);
}

var reach: S[>..] = [];

fn reached(name: const u8[:]) -> bool {
    for r in reach { if t(r) == name { return true; } }
    return false;
}

fn reach_token(id: i64) {
    let i = lower_bound(reach_tokens) { it < id };
    if i == reach_tokens.len || reach_tokens[i] != id { insert_at(reach_tokens, i, id); }
}

// The rules reachable from id, onto reach, and the tokens named by token
// rules, onto reach_tokens (collect_reachable).
recursive fn collect(a: i64, id: i64) {
    let p = ps[id];
    let k = p.kind;
    if k == Parser.Sequence || k == Parser.Choice {
        for j in p.kids_n { collect(a, kids[p.kids_at + j]); }
    } else if k == Parser.Token {
        reach_token(p.lo);
    } else if k == Parser.TokenNot {
        for j in p.kids_n { reach_token(kids[p.kids_at + j]); }
    } else if k == Parser.Until {
        // the delimiters name their tokens in the grammar
        if p.child >= 0 { collect(a, p.child); }
        for b in p.kids_n {
            collect(a, kids[p.kids_at + 2 * b]);
            if kids[p.kids_at + 2 * b + 1] >= 0 { collect(a, kids[p.kids_at + 2 * b + 1]); }
        }
    } else if k == Parser.Rule {
        if !reached(t(p.s)) {
            reach.push(p.s);
            collect(a, p.child);
        }
    } else if k == Parser.Ref {
        let rk = find_rule(a, t(p.s));
        if rk < 0 { fail(str("Rule not found: ", t(p.s))); }
        collect(a, rules[rk].id);
    } else if p.child >= 0 {
        collect(a, p.child);
    }
}

// Arena a's rules into the json_schema grammar being built (build_grammar);
// lazy: only the trigger rules and what they reach. The root is their
// choice, or, when they have a start and a rest, scans for any start and
// then allows only that rule's rest. It throws json_schema's Fail.
fn build_grammar(a: i64, lazy: bool) {
    reach.clear();
    reach_tokens.clear();
    gbase = arenas[a].base;
    // the trigger rules, by name
    var triggers: i64[>..] = [];
    for r in rules {
        guard r.arena == a && ps[r.id].kind == Parser.Rule && ps[r.id].trigger else { continue; }
        triggers.push(r.id);
    }
    stable_sort(triggers) { x, y => json_schema::cmp_bytes(t(ps[x].s), t(ps[y].s)) < 0 };
    var scans = false;
    for id in triggers { if ps[id].start_p >= 0 { scans = true; } }
    if scans {
        for id in triggers {
            if ps[id].start_p < 0 { fail(str("trigger rule without a start in a scanning grammar: ", t(ps[id].s))); }
        }
    }
    if lazy && scans {
        for id in triggers {
            collect(a, ps[id].start_p);
            collect(a, ps[id].rest_p);
        }
    } else if lazy {
        for id in triggers {
            // marked, then what it reaches
            if !reached(t(ps[id].s)) { reach.push(ps[id].s); }
            collect(a, ps[id].child);
        }
    } else {
        collect(a, arenas[a].root);
    }
    for r in rules {
        guard r.arena == a && reached(t(r.name)) else { continue; }
        let p = ps[r.id];
        guard p.kind == Parser.Rule else { continue; }
        let g = to_gbnf(p.child);
        json_schema::add_rule(jlit(t(p.s)), g);
    }
    if lazy && scans {
        // the rests' rules first: their grammar may build other matchers
        var rests: json_schema::Str[>..] = [];
        for id in triggers {
            let g = to_gbnf(ps[id].rest_p);
            let rname = json_schema::add_rule(jlit(str(t(ps[id].s), "-rest")), g);
            rests.push(rname);
        }
        trie_clear();
        for id, v in triggers {
            let mark = dset.len;
            dset.push(0);
            let da, dn = insert_delimiter(ps[id].start_p, mark, 1);
            for i in dn {
                let owner = trie_mark(dset[da + i], v as! i32);
                if owner != v as! i32 {
                    fail(str("trigger rules ", t(ps[triggers[owner]].s), " and ", t(ps[id].s), " must not share a start"));
                }
            }
            dset.resize(mark);
        }
        let g = ac_grammar("trigger", true, true, rests);
        json_schema::add_rule(jlit("root"), g);
    } else if lazy {
        var out: u8[>..] = [];
        for id, i in triggers {
            if i > 0 { out.append(" | "); }
            out.append(t(ps[id].s));
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

//
// C++ expressions, in the order GCC builds them
//

// An operand of a + chain: a parser, or a string (p < 0), whose literal the
// operator makes when it joins it.
struct CItem { p: i64 = -1, s: const u8[:] = "" }

fn ci(p: i64) -> CItem { CItem { p: p, .. } }
fn cs(s: const u8[:]) -> CItem { CItem { s: s, .. } }

// e1 + e2 + ... (left to right). GCC evaluates the operands right to left,
// so the caller makes the parser operands last to first; the sequences,
// and the literals of the string operands, are made here, left to right.
fn chain(items: const CItem[:]) -> i64 {
    var acc = items[0].p;
    var k = 1;
    if acc < 0 {
        // "s" + p
        let l = literal(items[0].s);
        acc = seq2(l, items[1].p);
        k = 2;
    }
    while k < items.len {
        let x = if items[k].p >= 0 { items[k].p } else { literal(items[k].s) };
        acc = seq2(acc, x);
        k++;
    }
    return acc;
}

// Ends the arena and forgets it if it is the last one (for the short-lived
// parsers of the template analysis).
fn release(a: i64) {
    guard a == arenas.len - 1 else { return; }
    let base = arenas[a].base;
    ps.resize(base);
    var w = 0;
    for r in rules.len { if rules[r].arena != a { rules[w] = rules[r]; w++; } }
    rules.resize(w);
    arenas.pop();
    cur = arenas.len - 1;
}
