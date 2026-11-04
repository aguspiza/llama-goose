// llama.cpp's GBNF grammars. The parser turns a grammar's text into rules,
// each a run of elements; the engine keeps the set of pushdown stacks the
// grammar can be in (each stack a list of positions in the rules, the top
// at a terminal), which each accepted character advances, and against
// which a token's text is checked before it may be sampled.
// One grammar is active at a time (the sampler's): its rules and stacks
// live in this module.
// port-of: src/llama-grammar.cpp, src/llama-grammar.h
import std;
import exception;
import .vocab;

namespace grammar;

let END = 0;             // the end of a rule
let ALT = 1;             // the start of another alternative
let RULE_REF = 2;        // a non-terminal: v is a rule id
let CHAR = 3;            // a terminal: v is a code point
let CHAR_NOT = 4;        // [^a], [^a-b]
let CHAR_RNG_UPPER = 5;  // makes the CHAR (or CHAR_ALT) before it a range [a-z]
let CHAR_ALT = 6;        // another character of a set [ab], [a-zA]
let CHAR_ANY = 7;        // .
let TOKEN = 8;           // <[id]>: v is a token id
let TOKEN_NOT = 9;       // !<[id]>

let MAX_REPETITION_THRESHOLD = 2000;
let RECURSIVE_LIMIT = 256;

struct Elem { t: u8, v: u32 }

// A UTF-8 sequence cut by the end of a token: its bits so far, and how many
// bytes are still to come (-1: invalid).
struct Partial { value: u32, n_remain: i32 }

struct Range { at: i64, len: i64 }

//
// errors
//

struct Fail { code: i32 }
var err: u8[>..<] = [];

fn fail(msg: const u8[:]) {
    err.clear();
    err.append(msg);
    exception::throw(Fail { code: 0 });
}

//
// the parser
//

// A grammar being parsed: the rules by id, each a range of `pelems` (empty
// until the rule is defined), and the names of the symbols.
var pelems: Elem[>..<] = [];
var prules: Range[>..<] = [];
var names: u8[>..<] = [];
var name_ranges: Range[>..<] = [];       // symbol id -> its name in names
// The rules being built, one above the other as groups nest.
var building: Elem[>..<] = [];

// The text being parsed, a copy of it.
var src: u8[>..<] = [];

// src[p], or 0 past its end, as a C string reads.
fn at(p: i64) -> u8 { if p < src.len { src[p] } else { 0 } }

// The text from p on, for error messages.
fn rest(p: i64) -> const u8[:] { src[min(p, src.len)..] }

fn symbol_name(id: i64) -> const u8[:] { names[name_ranges[id].at..name_ranges[id].at + name_ranges[id].len] }

// The id of the symbol name, a new one if it has none (get_symbol_id).
fn symbol_id(name: const u8[:]) -> u32 {
    for r, k in name_ranges {
        if names[r.at..r.at + r.len] == name { return k as u32; }
    }
    let at0 = names.len;
    names.append(name);
    name_ranges.push(Range { at: at0, len: name.len });
    return (name_ranges.len - 1) as u32;
}

// A new symbol, base_name_<id> (generate_symbol_id); a symbol already of
// that name now has that id.
fn generate_symbol_id(base_name: const u8[:]) -> u32 {
    let id = name_ranges.len;
    var name: u8[>..] = [];
    format(name, base_name, "_", id);
    for r, k in name_ranges {
        if names[r.at..r.at + r.len] == name { name_ranges[k] = Range { at: 0, len: 0 }; }
    }
    let at0 = names.len;
    names.append(name);
    name_ranges.push(Range { at: at0, len: name.len });
    return id as u32;
}

// rules[id] = building[from..], which is then dropped.
fn add_rule(id: u32, from: i64) {
    while prules.len <= id { prules.push(Range { at: 0, len: 0 }); }
    let at0 = pelems.len;
    for k in from..building.len { pelems.push(building[k]); }
    prules[id] = Range { at: at0, len: building.len - from };
    building.resize(from);
}

fn is_digit(c: u8) -> bool { c >= '0' && c <= '9' }
fn is_word_char(c: u8) -> bool { (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '-' || is_digit(c) }

fn parse_space(p: i64, newline_ok: bool) -> i64 {
    var pos = p;
    while at(pos) == ' ' || at(pos) == '\t' || at(pos) == '#' || (newline_ok && (at(pos) == '\r' || at(pos) == '\n')) {
        if at(pos) == '#' {
            while at(pos) != 0 && at(pos) != '\r' && at(pos) != '\n' { pos++; }
        } else {
            pos++;
        }
    }
    return pos;
}

fn parse_name(p: i64) -> i64 {
    var pos = p;
    while is_word_char(at(pos)) { pos++; }
    if pos == p { fail(str("expecting name at ", rest(p))); }
    return pos;
}

fn parse_int_end(p: i64) -> i64 {
    var pos = p;
    while is_digit(at(pos)) { pos++; }
    if pos == p { fail(str("expecting integer at ", rest(p))); }
    return pos;
}

fn parse_hex(p: i64, size: i64) -> u32, i64 {
    var pos = p;
    let end = p + size;
    var value: u32 = 0;
    while pos < end && at(pos) != 0 {
        let c = at(pos);
        var d: u32 = 0;
        if c >= 'a' && c <= 'f' { d = c - 'a' + 10; }
        else if c >= 'A' && c <= 'F' { d = c - 'A' + 10; }
        else if c >= '0' && c <= '9' { d = c - '0'; }
        else { break; }
        value = (value << 4) + d;
        pos++;
    }
    if pos != end { fail(str("expecting ", size, " hex chars at ", rest(p))); }
    return value, pos;
}

// One UTF-8 character of src (assumed valid, but not read past its end).
fn decode_char(p: i64) -> u32, i64 {
    let lookup: u8[16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4];
    let first = at(p);
    let len = lookup[first >> 4];
    let mask: u8 = ((1 << (8 - len)) - 1) as! u8;
    var value: u32 = first & mask;
    let end = p + len;
    var pos = p + 1;
    while pos < end && at(pos) != 0 {
        value = (value << 6) + (at(pos) & 0x3F);
        pos++;
    }
    return value, pos;
}

fn parse_char(p: i64) -> u32, i64 {
    if at(p) == '\\' {
        let e = at(p + 1);
        if e == 'x' { return parse_hex(p + 2, 2); }
        if e == 'u' { return parse_hex(p + 2, 4); }
        if e == 'U' { return parse_hex(p + 2, 8); }
        if e == 't' { return '\t', p + 2; }
        if e == 'r' { return '\r', p + 2; }
        if e == 'n' { return '\n', p + 2; }
        if e == '\\' || e == '"' || e == '[' || e == ']' || e == '-' { return e, p + 2; }
        fail(str("unknown escape at ", rest(p)));
    }
    if at(p) != 0 { return decode_char(p); }
    fail("unexpected end of input");
    return 0, p;
}

var parse_vocab = false;    // whether <token> can be tokenized

fn parse_token(p: i64) -> u32, i64 {
    var pos = p;
    if at(pos) != '<' { fail(str("expecting '<' at ", rest(pos))); }
    pos++;
    if at(pos) == '[' {
        pos++;
        let int_end = parse_int_end(pos);
        let v, ok = parse_int(src[pos..int_end]);
        if !ok || v > 4294967295 { fail(str("parsed token id is too big at ", rest(pos))); }
        pos = int_end;
        if at(pos) != ']' { fail(str("expecting ']' at ", rest(pos))); }
        pos++;
        if at(pos) != '>' { fail(str("expecting '>' at ", rest(pos))); }
        pos++;
        return v as! u32, pos;
    }
    if !parse_vocab { fail(str("no vocab to parse token at ", rest(p))); }
    while at(pos) != 0 && at(pos) != '>' { pos++; }
    if at(pos) != '>' { fail(str("expecting '>' at ", rest(pos))); }
    pos++;
    var ids: i32[>..] = [];
    vocab::tokenize(src[p..pos], false, true, ids);
    if ids.len != 1 { fail(str("invalid token '", src[p..pos], "'")); }
    return ids[0] as! u32, pos;
}

recursive fn parse_alternates(p: i64, rule_name: Range, rule_id: u32, is_nested: bool, depth: i64) -> i64 {
    let from = building.len;
    var pos = parse_sequence(p, rule_name, from, is_nested, depth);
    while at(pos) == '|' {
        building.push(Elem { t: ALT, v: 0 });
        pos = parse_space(pos + 1, true);
        pos = parse_sequence(pos, rule_name, from, is_nested, depth);
    }
    building.push(Elem { t: END, v: 0 });
    add_rule(rule_id, from);
    return pos;
}

// The rule name, for the symbols repetitions and groups generate.
fn rname(r: Range) -> const u8[:] { names[r.at..r.at + r.len] }

// One alternative onto building (the rule being built starts at rule_from).
recursive fn parse_sequence(p: i64, rule_name: Range, rule_from: i64, is_nested: bool, depth: i64) -> i64 {
    var last_sym_start = building.len;
    var pos = p;
    var n_prev_rules: u64 = 1;
    while at(pos) != 0 {
        let c = at(pos);
        if c == '"' {
            // a literal string
            pos++;
            last_sym_start = building.len;
            n_prev_rules = 1;
            while at(pos) != '"' {
                if at(pos) == 0 { fail("unexpected end of input"); }
                let ch, next = parse_char(pos);
                pos = next;
                building.push(Elem { t: CHAR, v: ch });
            }
            pos = parse_space(pos + 1, is_nested);
        } else if c == '[' {
            // character ranges
            pos++;
            var start_type: u8 = CHAR;
            if at(pos) == '^' { pos++; start_type = CHAR_NOT; }
            last_sym_start = building.len;
            n_prev_rules = 1;
            while at(pos) != ']' {
                if at(pos) == 0 { fail("unexpected end of input"); }
                let ch, next = parse_char(pos);
                pos = next;
                let t: u8 = if last_sym_start < building.len { CHAR_ALT } else { start_type };
                building.push(Elem { t: t, v: ch });
                if at(pos) == '-' && at(pos + 1) != ']' {
                    if at(pos + 1) == 0 { fail("unexpected end of input"); }
                    let endch, next2 = parse_char(pos + 1);
                    pos = next2;
                    building.push(Elem { t: CHAR_RNG_UPPER, v: endch });
                }
            }
            pos = parse_space(pos + 1, is_nested);
        } else if c == '<' || c == '!' {
            // a token
            var t: u8 = TOKEN;
            if c == '!' { t = TOKEN_NOT; pos++; }
            let id, token_end = parse_token(pos);
            last_sym_start = building.len;
            n_prev_rules = 1;
            building.push(Elem { t: t, v: id });
            pos = parse_space(token_end, is_nested);
        } else if is_word_char(c) {
            // a rule reference
            let name_end = parse_name(pos);
            let ref_id = symbol_id(src[pos..name_end]);
            pos = parse_space(name_end, is_nested);
            last_sym_start = building.len;
            n_prev_rules = 1;
            building.push(Elem { t: RULE_REF, v: ref_id });
        } else if c == '(' {
            // a group: its alternates become a rule of their own
            if depth >= RECURSIVE_LIMIT {
                fail(str("grammar recursion depth limit exceeded (max ", RECURSIVE_LIMIT, "), please reduce nesting depth"));
            }
            pos = parse_space(pos + 1, true);
            let n_before = name_ranges.len;
            let sub_id = generate_symbol_id(rname(rule_name));
            pos = parse_alternates(pos, rule_name, sub_id, true, depth + 1);
            n_prev_rules = max(1, (name_ranges.len - n_before) as u64);
            last_sym_start = building.len;
            building.push(Elem { t: RULE_REF, v: sub_id });
            if at(pos) != ')' { fail(str("expecting ')' at ", rest(pos))); }
            pos = parse_space(pos + 1, is_nested);
        } else if c == '.' {
            last_sym_start = building.len;
            n_prev_rules = 1;
            building.push(Elem { t: CHAR_ANY, v: 0 });
            pos = parse_space(pos + 1, is_nested);
        } else if c == '*' || c == '+' || c == '?' {
            pos = parse_space(pos + 1, is_nested);
            let none: u64 = 18446744073709551615;
            if c == '*' { repetitions(rule_name, last_sym_start, n_prev_rules, 0, none, pos); }
            else if c == '+' { repetitions(rule_name, last_sym_start, n_prev_rules, 1, none, pos); }
            else { repetitions(rule_name, last_sym_start, n_prev_rules, 0, 1, pos); }
            n_prev_rules = rep_n_prev;
        } else if c == '{' {
            pos = parse_space(pos + 1, is_nested);
            if !is_digit(at(pos)) { fail(str("expecting an int at ", rest(pos))); }
            let int_end = parse_int_end(pos);
            let min_times = parse_u64(src[pos..int_end]);
            pos = parse_space(int_end, is_nested);
            var max_times: u64 = 18446744073709551615;
            if at(pos) == '}' {
                max_times = min_times;
                pos = parse_space(pos + 1, is_nested);
            } else if at(pos) == ',' {
                pos = parse_space(pos + 1, is_nested);
                if is_digit(at(pos)) {
                    let int_end2 = parse_int_end(pos);
                    max_times = parse_u64(src[pos..int_end2]);
                    pos = parse_space(int_end2, is_nested);
                }
                if at(pos) != '}' { fail(str("expecting '}' at ", rest(pos))); }
                pos = parse_space(pos + 1, is_nested);
            } else {
                fail(str("expecting ',' at ", rest(pos)));
            }
            if min_times > MAX_REPETITION_THRESHOLD {
                fail("number of repetitions exceeds sane defaults, please reduce the number of repetitions");
            }
            if max_times != 18446744073709551615 && max_times > MAX_REPETITION_THRESHOLD { max_times = 18446744073709551615; }
            repetitions(rule_name, last_sym_start, n_prev_rules, min_times, max_times, pos);
            n_prev_rules = rep_n_prev;
        } else {
            break;
        }
    }
    return pos;
}

// std::stoull of digits; it fails above u64.
fn parse_u64(s: const u8[:]) -> u64 {
    var v: u64 = 0;
    for c in s {
        let d: u64 = c - '0';
        if v > (18446744073709551615 - d) / 10 { fail("stoull"); }
        v = v * 10 + d;
    }
    return v;
}

var rep_n_prev: u64 = 1;

// S{m,n} rewritten: S m times, then n - m optional ones, S'(x) ::= S S'(x-1) |;
// S{m,}: S m times then S' ::= S S' |. The symbol is building[last_sym_start..].
fn repetitions(rule_name: Range, last_sym_start: i64, n_prev_rules: u64, min_times: u64, max_times: u64, pos: i64) {
    let no_max = max_times == 18446744073709551615;
    if last_sym_start == building.len { fail(str("expecting preceding item to */+/?/{ at ", rest(pos))); }
    // the symbol, aside
    var prev: Elem[>..] = [];
    for k in last_sym_start..building.len { prev.push(building[k]); }
    var total_rules: u64 = 1;
    if !no_max && max_times > 0 { total_rules = max_times; } else if min_times > 0 { total_rules = min_times; }
    if n_prev_rules * total_rules > MAX_REPETITION_THRESHOLD {
        fail("number of rules that are going to be repeated multiplied by the new repetition exceeds sane defaults, please reduce the number of repetitions or rule complexity");
    }
    if min_times == 0 {
        building.resize(last_sym_start);
    } else {
        var i: u64 = 1;
        while i < min_times { building.append(prev); i++; }
    }
    var last_rec_id: u32 = 0;
    let n_opt: u64 = if no_max { 1 } else { max_times - min_times };
    var i: u64 = 0;
    while i < n_opt {
        let rec_id = generate_symbol_id(rname(rule_name));
        let from = building.len;
        building.append(prev);
        if i > 0 || no_max { building.push(Elem { t: RULE_REF, v: if no_max { rec_id } else { last_rec_id } }); }
        building.push(Elem { t: ALT, v: 0 });
        building.push(Elem { t: END, v: 0 });
        add_rule(rec_id, from);
        last_rec_id = rec_id;
        i++;
    }
    if n_opt > 0 { building.push(Elem { t: RULE_REF, v: last_rec_id }); }
    rep_n_prev = n_prev_rules * total_rules;
}

fn parse_rule(p: i64) -> i64 {
    let name_end = parse_name(p);
    var pos = parse_space(name_end, false);
    let rule_id = symbol_id(src[p..name_end]);
    let name = name_ranges[rule_id];
    if !(at(pos) == ':' && at(pos + 1) == ':' && at(pos + 2) == '=') { fail(str("expecting ::= at ", rest(pos))); }
    pos = parse_space(pos + 3, true);
    // the name, kept in its own copy: a generated symbol may take the id's place
    let nat = names.len;
    names.append(src[p..name_end]);
    pos = parse_alternates(pos, Range { at: nat, len: name.len }, rule_id, false, 0);
    if at(pos) == '\r' { pos += if at(pos + 1) == '\n' { 2 } else { 1 }; }
    else if at(pos) == '\n' { pos++; }
    else if at(pos) != 0 { fail(str("expecting newline or end at ", rest(pos))); }
    return parse_space(pos, true);
}

// Parses text into the rules (llama_grammar_parser::parse); false, and the
// error in err, on a malformed grammar or an undefined rule.
fn parse(text: const u8[:], with_vocab: bool) -> bool {
    pelems.clear();
    prules.clear();
    names.clear();
    name_ranges.clear();
    building.clear();
    src.clear();
    src.append(text);
    parse_vocab = with_vocab;
    var ok = false;
    fn body() {
        var pos = parse_space(0, true);
        while at(pos) != 0 { pos = parse_rule(pos); }
        for r in prules {
            if r.len == 0 { fail("Undefined rule"); }
            for e in pelems[r.at..r.at + r.len] {
                guard e.t == RULE_REF else { continue; }
                if e.v >= prules.len || prules[e.v].len == 0 {
                    fail(str("Undefined rule identifier '", symbol_name(e.v), "'"));
                }
            }
        }
        ok = true;
    }
    exception::try<Fail>(body);
    if !ok { prules.clear(); }
    return ok;
}

// The symbols, by name in byte order as llama.cpp's std::map keeps them.
fn sorted_symbols(out: i64[>..]&) {
    out.clear();
    for r, k in name_ranges { if r.len > 0 || k == 0 { out.push(k); } }
    stable_sort(out) { a, b => cmp_bytes(symbol_name(a), symbol_name(b)) < 0 };
}

fn cmp_bytes(x: const u8[:], y: const u8[:]) -> i64 {
    for i in min(x.len, y.len) { if x[i] != y[i] { return if x[i] < y[i] { -1 } else { 1 }; } }
    return if x.len < y.len { -1 } else if x.len > y.len { 1 } else { 0 };
}

//
// the engine
//

// The grammar in use: its rules (elements of `elems`, rule r starting at
// rule_at[r] and ending with END), its stacks and its partial UTF-8.
var elems: Elem[>..<] = [];
var rule_at: i64[>..<] = [];
var partial = Partial { value: 0, n_remain: 0 };

// Stacks: each a range of sdata (positions in elems, the top last); a set
// of stacks is a range of sets. The grammar's own set is at the bottom
// (cur_first, cur_n); the sets being worked on go above it.
var sdata: i32[>..<] = [];
var sets: Range[>..<] = [];
var cur_n = 0;
// Where new stacks are gathered while the current set is advanced.
var ndata: i32[>..<] = [];
var nsets: Range[>..<] = [];
// advance_stack's stacks to do and stacks seen.
var tdata: i32[>..<] = [];
var tsets: Range[>..<] = [];
var vdata: i32[>..<] = [];
var vsets: Range[>..<] = [];

fn is_end_of_sequence(pos: i64) -> bool { elems[pos].t == END || elems[pos].t == ALT }

// Whether stack (data[r.at..]) is in the stacks out_sets[first..].
fn set_has(odata: i32[>..<]&, osets: Range[>..<]&, first: i64, data: i32[>..<]&, r: Range) -> bool {
    for k in first..osets.len {
        let o = osets[k];
        guard o.len == r.len else { continue; }
        var same = true;
        for i in r.len { if odata[o.at + i] != data[r.at + i] { same = false; break; } }
        if same { return true; }
    }
    return false;
}

fn push_stack(odata: i32[>..<]&, osets: Range[>..<]&, data: i32[>..<]&, r: Range) {
    let at0 = odata.len;
    for i in r.len { odata.push(data[r.at + i]); }
    osets.push(Range { at: at0, len: r.len });
}

// The stack on top of tsets, advanced into the stacks that end at a
// terminal, added to odata/osets (from first on) unless there already
// (llama_grammar_advance_stack). The stack is taken off tsets.
fn advance(odata: i32[>..<]&, osets: Range[>..<]&, first: i64) {
    let tbase = tsets.len - 1;
    let vbase = vsets.len;
    let vdbase = vdata.len;
    while tsets.len > tbase {
        let top = tsets[tsets.len - 1];
        if set_has(vdata, vsets, vbase, tdata, top) {
            tsets.pop();
            tdata.resize(top.at);
            continue;
        }
        push_stack(vdata, vsets, tdata, top);
        let cur = vsets[vsets.len - 1];
        tsets.pop();
        tdata.resize(top.at);
        if cur.len == 0 {
            if !set_has(odata, osets, first, vdata, cur) { push_stack(odata, osets, vdata, cur); }
            continue;
        }
        let pos: i64 = vdata[cur.at + cur.len - 1];
        let t = elems[pos].t;
        if t == RULE_REF {
            var sub = rule_at[elems[pos].v];
            loop {
                // the stack without its top, then what follows the reference, then the alternative
                let at0 = tdata.len;
                for i in cur.len - 1 { tdata.push(vdata[cur.at + i]); }
                if !is_end_of_sequence(pos + 1) { tdata.push((pos + 1) as i32); }
                if !is_end_of_sequence(sub) { tdata.push(sub as i32); }
                tsets.push(Range { at: at0, len: tdata.len - at0 });
                while !is_end_of_sequence(sub) { sub++; }
                if elems[sub].t == ALT { sub++; } else { break; }
            }
        } else if t == CHAR || t == CHAR_NOT || t == CHAR_ANY || t == TOKEN || t == TOKEN_NOT {
            if !set_has(odata, osets, first, vdata, cur) { push_stack(odata, osets, vdata, cur); }
        } else {
            abort("grammar: a stack left at the middle of a character range");
        }
    }
    vsets.resize(vbase);
    vdata.resize(vdbase);
}

// Whether chr satisfies the character range at pos, and the position after
// the range (llama_grammar_match_char).
fn match_char(p: i64, chr: u32) -> bool, i64 {
    var pos = p;
    var found = false;
    let positive = elems[pos].t == CHAR || elems[pos].t == CHAR_ANY;
    loop {
        if elems[pos + 1].t == CHAR_RNG_UPPER {
            found = found || (elems[pos].v <= chr && chr <= elems[pos + 1].v);
            pos += 2;
        } else if elems[pos].t == CHAR_ANY {
            found = true;
            pos += 1;
        } else {
            found = found || elems[pos].v == chr;
            pos += 1;
        }
        if elems[pos].t != CHAR_ALT { break; }
    }
    return found == positive, pos;
}

// Whether some completion of a partial UTF-8 sequence could satisfy the
// range at pos (llama_grammar_match_partial_char).
fn match_partial_char(p: i64, pu: Partial) -> bool {
    var pos = p;
    let positive = elems[pos].t == CHAR || elems[pos].t == CHAR_ANY;
    let n = pu.n_remain;
    if n < 0 || (n == 1 && pu.value < 2) { return false; }
    var low: u32 = pu.value << (n * 6);
    let high: u32 = low | ((1 << (n * 6)) - 1);
    if low == 0 {
        if n == 2 { low = 1 << 11; } else if n == 3 { low = 1 << 16; }
    }
    loop {
        if elems[pos + 1].t == CHAR_RNG_UPPER {
            if elems[pos].v <= high && low <= elems[pos + 1].v { return positive; }
            pos += 2;
        } else if elems[pos].t == CHAR_ANY {
            return true;
        } else {
            if low <= elems[pos].v && elems[pos].v <= high { return positive; }
            pos += 1;
        }
        if elems[pos].t != CHAR_ALT { break; }
    }
    return !positive;
}

fn match_token(pos: i64, token: i32) -> bool {
    if elems[pos].t == TOKEN { return elems[pos].v == token as! u32; }
    return elems[pos].v != token as! u32;
}

// The code points of text after a partial sequence, then a 0, onto out;
// and the partial sequence the text ends in (decode_utf8).
fn decode(text: const u8[:], start: Partial, out: u32[>..<]&) -> Partial {
    let lookup: i32[16] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4];
    let first = out.len;
    var pos = 0;
    var value = start.value;
    var n_remain = start.n_remain;
    // the previous sequence first
    while pos < text.len && text[pos] != 0 && n_remain > 0 {
        let b = text[pos];
        if (b >> 6) != 2 {
            out.push(0);
            return Partial { value: 0, n_remain: -1 };
        }
        value = (value << 6) + (b & 0x3F);
        pos++;
        n_remain--;
    }
    if start.n_remain > 0 && n_remain == 0 { out.push(value); }
    while pos < text.len && text[pos] != 0 {
        let b = text[pos];
        n_remain = lookup[b >> 4] - 1;
        if n_remain < 0 {
            out.resize(first);
            out.push(0);
            return Partial { value: 0, n_remain: n_remain };
        }
        let mask: u8 = ((1 << (7 - n_remain)) - 1) as! u8;
        value = b & mask;
        pos++;
        while pos < text.len && text[pos] != 0 && n_remain > 0 {
            value = (value << 6) + (text[pos] & 0x3F);
            pos++;
            n_remain--;
        }
        if n_remain == 0 { out.push(value); }
    }
    out.push(0);
    return Partial { value: value, n_remain: n_remain };
}

// chr accepted by stack (data[r.at..]): the stacks that follow, added to
// odata/osets from first on (llama_grammar_accept_chr).
fn accept_chr(data: i32[>..<]&, r: Range, chr: u32, odata: i32[>..<]&, osets: Range[>..<]&, first: i64) {
    if r.len == 0 { return; }
    let pos: i64 = data[r.at + r.len - 1];
    let t = elems[pos].t;
    if t == TOKEN || t == TOKEN_NOT { return; }
    let ok, next = match_char(pos, chr);
    if !ok { return; }
    let at0 = tdata.len;
    for i in r.len - 1 { tdata.push(data[r.at + i]); }
    if !is_end_of_sequence(next) { tdata.push(next as i32); }
    tsets.push(Range { at: at0, len: tdata.len - at0 });
    advance(odata, osets, first);
}

// Rules in, each a run of elements ending with END: a grammar starting at
// rule `start` (llama_grammar_init_impl); false, and the error in err, if a
// rule is undefined or left-recursive.
fn init_rules(rules: const Elem[:], starts: const i64[:], start: i64) -> bool {
    elems.clear();
    rule_at.clear();
    for s in starts { rule_at.push(elems.len + s); }
    elems.append(rules);
    let n = starts.len;
    for r in n {
        var p = rule_at[r];
        loop {
            let e = elems[p];
            if e.t == RULE_REF && (e.v >= n) {
                err.clear();
                format(err, "invalid grammar: rule ", r, " references undefined rule ", e.v);
                return false;
            }
            if e.t == END { break; }
            p++;
        }
    }
    // left recursion
    var visited: bool[>..] = [];
    var in_progress: bool[>..] = [];
    var may_be_empty: bool[>..] = [];
    for r in n { visited.push(false); in_progress.push(false); may_be_empty.push(false); }
    var bad = -1;
    fn body() {
        for r in n {
            if visited[r] { continue; }
            if left_recursion(r, visited, in_progress, may_be_empty, 0) { bad = r; return; }
        }
    }
    var ok = true;
    exception::try<Fail>(body) { f => ok = false; };
    if !ok {
        var msg: u8[>..] = [];
        format(msg, "unsupported grammar: ", err);
        err.clear();
        err.append(msg);
        return false;
    }
    if bad >= 0 {
        err.clear();
        format(err, "unsupported grammar, left recursion detected for nonterminal at index ", bad);
        return false;
    }
    // the stacks of the start rule's alternatives
    sdata.clear();
    sets.clear();
    tdata.clear();
    tsets.clear();
    var p = rule_at[start];
    loop {
        let at0 = tdata.len;
        if !is_end_of_sequence(p) { tdata.push(p as i32); }
        tsets.push(Range { at: at0, len: tdata.len - at0 });
        advance(sdata, sets, 0);
        while !is_end_of_sequence(p) { p++; }
        if elems[p].t == ALT { p++; } else { break; }
    }
    cur_n = sets.len;
    partial = Partial { value: 0, n_remain: 0 };
    return true;
}

recursive fn left_recursion(r: i64, visited: bool[>..]&, in_progress: bool[>..]&, may_be_empty: bool[>..]&, depth: i64) -> bool {
    if in_progress[r] { return true; }
    if depth > RECURSIVE_LIMIT {
        fail(str("grammar rules are chained more than ", RECURSIVE_LIMIT, " levels deep"));
    }
    in_progress[r] = true;
    // whether the rule can make the empty string
    var p = rule_at[r];
    var at_rule_start = true;
    loop {
        if is_end_of_sequence(p) {
            if at_rule_start { may_be_empty[r] = true; break; }
            at_rule_start = true;
        } else {
            at_rule_start = false;
        }
        if elems[p].t == END { break; }
        p++;
    }
    // the leftmost non-terminals, and those after ones that can be empty
    p = rule_at[r];
    var recurse = true;
    loop {
        let e = elems[p];
        if e.t == RULE_REF && recurse {
            if left_recursion(e.v, visited, in_progress, may_be_empty, depth + 1) { return true; }
            if !may_be_empty[e.v] { recurse = false; }
        } else if is_end_of_sequence(p) {
            recurse = true;
        } else {
            recurse = false;
        }
        if e.t == END { break; }
        p++;
    }
    in_progress[r] = false;
    visited[r] = true;
    return false;
}

// The grammar from the parsed rules, starting at the symbol root.
fn init_parsed(root: const u8[:]) -> bool {
    var root_id = -1;
    for r, k in name_ranges { if r.len > 0 && names[r.at..r.at + r.len] == root { root_id = k; } }
    if root_id < 0 {
        err.clear();
        format(err, "grammar does not contain a '", root, "' symbol");
        return false;
    }
    var flat: Elem[>..] = [];
    var starts: i64[>..] = [];
    for r in prules {
        starts.push(flat.len);
        // up to the first END
        for e in pelems[r.at..r.at + r.len] {
            if e.t == END { break; }
            flat.push(e);
        }
        flat.push(Elem { t: END, v: 0 });
    }
    return init_rules(flat, starts, root_id);
}

// llama_grammar_init_impl from a grammar's text: false, and the error in
// err, if it does not parse or has no root symbol.
fn init(text: const u8[:], root: const u8[:], with_vocab: bool) -> bool {
    awaiting = false;
    tbuf.clear();
    tpos.clear();
    ttokens.clear();
    twords.clear();
    tword_at.clear();
    if !parse(text, with_vocab) || prules.len == 0 { return false; }
    return init_parsed(root);
}

//
// lazy grammars
//

// A lazy grammar (llama_sampler_init_grammar_lazy_patterns) constrains
// nothing until a trigger: one of its tokens, or one of its words in the
// text generated so far. Then the tokens from the word on are replayed
// into it. The words are what common_sampler makes of chat's word
// triggers: regex_escape(word), which std::regex_search finds where the
// word itself is.
var awaiting = false;
var tbuf: u8[>..<] = [];                 // the text since the start, while awaiting
struct TPos { token: i32, start: i64, end: i64 }
var tpos: TPos[>..<] = [];               // each token's piece in tbuf
var ttokens: i32[>..<] = [];
var twords: u8[>..<] = [];
var tword_at: Range[>..<] = [];

// Makes the grammar just initialized lazy, its triggers to be added.
fn make_lazy() { awaiting = true; }
fn add_trigger_token(token: i32) { ttokens.push(token); }
fn add_trigger_word(word: const u8[:]) {
    tword_at.push(Range { at: twords.len, len: word.len });
    twords.append(word);
}

fn awaiting_trigger() -> bool { awaiting }

fn find_bytes(hay: const u8[:], needle: const u8[:]) -> i64 {
    if needle.len == 0 { return 0; }
    var i = 0;
    while i + needle.len <= hay.len {
        if hay[i..i + needle.len] == needle { return i; }
        i++;
    }
    return -1;
}

// accept_impl while awaiting a trigger.
fn accept_awaiting(token: i32, piece: const u8[:]) {
    for t in ttokens {
        guard t == token else { continue; }
        awaiting = false;
        tbuf.clear();
        accept_checked(token, piece);
        return;
    }
    tpos.push(TPos { token: token, start: tbuf.len, end: tbuf.len + piece.len });
    tbuf.append(piece);
    for r in tword_at {
        let start = find_bytes(tbuf, twords[r.at..r.at + r.len]);
        guard start >= 0 else { continue; }
        awaiting = false;
        // the tokens from start on, the first maybe in part
        var replay: u8[>..] = [];
        for p in tpos {
            guard p.end > start else { continue; }
            let piece_start = max(p.start, start);
            replay.clear();
            replay.append(tbuf[piece_start..p.end]);
            accept_checked(p.token, replay);
        }
        tbuf.clear();
        tpos.clear();
        return;
    }
}

fn accept_checked(token: i32, piece: const u8[:]) {
    if !accept_token(token, piece) {
        write_stderr(str("Unexpected empty grammar stack after accepting piece: ", piece, " (", token, ")\n"));
        abort("grammar");
    }
}

// The grammar's stacks, one per entry: whether any is empty (the grammar
// may end here).
fn can_end() -> bool {
    for k in cur_n { if sets[k].len == 0 { return true; } }
    return false;
}

fn n_stacks() -> i64 { cur_n }

// Replaces the grammar's stacks with those gathered in ndata/nsets.
fn take_new_stacks() {
    sdata.clear();
    sets.clear();
    for r in nsets {
        let at0 = sdata.len;
        for i in r.len { sdata.push(ndata[r.at + i]); }
        sets.push(Range { at: at0, len: r.len });
    }
    cur_n = sets.len;
    ndata.clear();
    nsets.clear();
}

// One code point accepted by every stack (llama_grammar_accept).
fn accept(chr: u32) {
    ndata.clear();
    nsets.clear();
    for k in cur_n { accept_chr(sdata, sets[k], chr, ndata, nsets, 0); }
    take_new_stacks();
}

var cps: u32[>..<] = [];

// The token, of text piece, accepted (llama_grammar_accept_token); false
// if no stack accepts it, and then the grammar has no stacks.
fn accept_token(token: i32, piece: const u8[:]) -> bool {
    cps.clear();
    let after = decode(piece, partial, cps);
    ndata.clear();
    nsets.clear();
    for k in cur_n {
        let st = sets[k];
        guard st.len > 0 else { continue; }
        let pos: i64 = sdata[st.at + st.len - 1];
        if elems[pos].t == TOKEN || elems[pos].t == TOKEN_NOT {
            if match_token(pos, token) {
                let at0 = tdata.len;
                for i in st.len - 1 { tdata.push(sdata[st.at + i]); }
                if !is_end_of_sequence(pos + 1) { tdata.push((pos + 1) as i32); }
                tsets.push(Range { at: at0, len: tdata.len - at0 });
                advance(ndata, nsets, 0);
            }
            continue;
        }
        // the code points one by one, from this stack: the sets in turn above the grammar's
        let base_sets = sets.len;
        let base_data = sdata.len;
        push_stack(sdata, sets, sdata, st);
        var cur_first = base_sets;
        for i in cps.len - 1 {
            let next_first = sets.len;
            for j in cur_first..next_first { accept_chr(sdata, sets[j], cps[i], sdata, sets, next_first); }
            // drop the set just used: the next one moves down to where it was
            let n_next = sets.len - next_first;
            let at_data = if cur_first < sets.len { sets[cur_first].at } else { sdata.len };
            var moved_data = at_data;
            for j in n_next {
                let r = sets[next_first + j];
                let at0 = moved_data;
                for q in r.len { sdata[moved_data] = sdata[r.at + q]; moved_data++; }
                sets[cur_first + j] = Range { at: at0, len: r.len };
            }
            sets.resize(cur_first + n_next);
            sdata.resize(moved_data);
            if n_next == 0 { break; }
        }
        for j in cur_first..sets.len {
            if !set_has(ndata, nsets, 0, sdata, sets[j]) { push_stack(ndata, nsets, sdata, sets[j]); }
        }
        sets.resize(base_sets);
        sdata.resize(base_data);
    }
    take_new_stacks();
    partial = after;
    return cur_n > 0;
}

//
// candidates
//

// A token to check: where its code points are (in ccps, 0-ended), the
// partial sequence after them, its id, and its index among the candidates.
struct Cand { index: i64, cp_at: i64, partial: Partial, id: i32 }

var ccps: u32[>..<] = [];
var ridx: i64[>..<] = [];

struct Level { set_first: i64, set_n: i64, data_mark: i64, i: i64, cur_at: i64, cur_len: i64, rej_at: i64, rej_len: i64 }

// The candidates no stack of the set (sets[set_first..set_first + set_n])
// accepts (llama_grammar_reject_candidates), as indices into cands, in
// ridx[out..]. A level per code point; a level checks the candidates that
// reach it against its stacks in turn, each stack narrowing the rejects of
// the one before.
fn reject_candidates(set_first: i64, set_n: i64, cands: const Cand[:]) -> i64 {
    let base = ridx.len;
    for i in cands.len { ridx.push(i); }
    var levels: Level[>..<] = [];
    levels.push(Level { set_first: set_first, set_n: set_n, data_mark: sdata.len, i: 0, cur_at: base, cur_len: cands.len, rej_at: base + cands.len, rej_len: 0 });
    loop {
        let lv = levels[levels.len - 1];
        if lv.i == lv.set_n || lv.cur_len == 0 {
            if levels.len == 1 { break; }
            // the rejects at the next code point join those of stacks[i] of the parent
            levels.pop();
            let pi = levels.len - 1;
            var parent = levels[pi];
            // the child's cur follows the parent's rejects in ridx
            parent.rej_len += lv.cur_len;
            sets.resize(lv.set_first);
            sdata.resize(lv.data_mark);
            // the parent's rejects become its cur
            for k in parent.rej_len { ridx[parent.cur_at + k] = ridx[parent.rej_at + k]; }
            parent.cur_len = parent.rej_len;
            parent.rej_at = parent.cur_at + parent.cur_len;
            parent.rej_len = 0;
            ridx.resize(parent.rej_at);
            parent.i++;
            levels[pi] = parent;
            continue;
        }
        let depth = levels.len - 1;
        let st = sets[lv.set_first + lv.i];
        let li = levels.len - 1;
        if st.len == 0 || elems[sdata[st.at + st.len - 1]].t == TOKEN || elems[sdata[st.at + st.len - 1]].t == TOKEN_NOT {
            // rejects only: they become the level's cur
            let r0 = ridx.len;
            for k in lv.cur_len {
                let ci = ridx[lv.cur_at + k];
                let c = cands[ci];
                let cp = ccps[c.cp_at + depth];
                var reject = false;
                if st.len == 0 {
                    reject = cp != 0 || c.partial.n_remain != 0;
                } else if cp == 0 {
                    reject = c.partial.n_remain != 0;
                } else {
                    reject = !match_token(sdata[st.at + st.len - 1], c.id);
                }
                if reject { ridx.push(ci); }
            }
            let n_rej = ridx.len - r0;
            for k in n_rej { ridx[lv.cur_at + k] = ridx[r0 + k]; }
            ridx.resize(lv.cur_at + n_rej);
            levels[li] = Level { set_first: lv.set_first, set_n: lv.set_n, data_mark: lv.data_mark, i: lv.i + 1, cur_at: lv.cur_at, cur_len: n_rej, rej_at: lv.cur_at + n_rej, rej_len: 0 };
            continue;
        }
        // a character range: the rejects, then those that go on to the next code point
        let pos: i64 = sdata[st.at + st.len - 1];
        let r0 = ridx.len;
        for k in lv.cur_len {
            let ci = ridx[lv.cur_at + k];
            let c = cands[ci];
            let cp = ccps[c.cp_at + depth];
            if cp == 0 {
                if c.partial.n_remain != 0 && !match_partial_char(pos, c.partial) { ridx.push(ci); }
            } else {
                let ok, after = match_char(pos, cp);
                if !ok { ridx.push(ci); }
            }
        }
        let n_rej = ridx.len - r0;
        let n0 = ridx.len;
        for k in lv.cur_len {
            let ci = ridx[lv.cur_at + k];
            let c = cands[ci];
            let cp = ccps[c.cp_at + depth];
            if cp != 0 {
                let ok, after = match_char(pos, cp);
                if ok { ridx.push(ci); }
            }
        }
        let n_next = ridx.len - n0;
        // the stacks after this position
        let ok0, pos_after = match_char(pos, 0);
        let set0 = sets.len;
        let mark = sdata.len;
        let at0 = tdata.len;
        for i in st.len - 1 { tdata.push(sdata[st.at + i]); }
        if !is_end_of_sequence(pos_after) { tdata.push(pos_after as i32); }
        tsets.push(Range { at: at0, len: tdata.len - at0 });
        advance(sdata, sets, set0);
        assert(sets.len > set0);
        levels[li] = Level { set_first: lv.set_first, set_n: lv.set_n, data_mark: lv.data_mark, i: lv.i, cur_at: lv.cur_at, cur_len: lv.cur_len, rej_at: r0, rej_len: n_rej };
        levels.push(Level { set_first: set0, set_n: sets.len - set0, data_mark: mark, i: 0, cur_at: n0, cur_len: n_next, rej_at: n0 + n_next, rej_len: 0 });
    }
    let l0 = levels[0];
    // the rejected candidates, from base on
    for k in l0.cur_len { ridx[base + k] = ridx[l0.cur_at + k]; }
    ridx.resize(base + l0.cur_len);
    return base;
}

// llama_grammar_apply_impl: which of the tokens ids the grammar cannot take
// now (rejected[i]). An end of generation token is taken only where the
// grammar can end; a token with no text, never.
fn reject_tokens(ids: const i32[:], rejected: bool[>..]&) {
    rejected.clear();
    for id in ids { rejected.push(false); }
    if awaiting { return; }
    let allow_eog = can_end();
    var cands: Cand[>..] = [];
    ccps.clear();
    for id, i in ids {
        if vocab::is_eog(id) {
            if !allow_eog { rejected[i] = true; }
            continue;
        }
        let piece = vocab::token_piece(id);
        if piece.len == 0 || piece[0] == 0 { rejected[i] = true; continue; }
        let cp_at = ccps.len;
        let pu = decode(piece, partial, ccps);
        cands.push(Cand { index: i, cp_at: cp_at, partial: pu, id: id });
    }
    let base = reject_candidates(0, cur_n, cands);
    for k in base..ridx.len { rejected[cands[ridx[k]].index] = true; }
    ridx.resize(base);
}

// llama_grammar_accept_impl: the grammar after the token. It must be one
// the grammar takes.
fn accept_impl(token: i32) {
    let piece = vocab::token_piece(token);
    if awaiting { accept_awaiting(token, piece); return; }
    if vocab::is_eog(token) {
        if can_end() { return; }
        abort("grammar: an end of generation token where the grammar cannot end");
    }
    accept_checked(token, piece);
}

