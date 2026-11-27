// A backtracking regex matcher with the ECMAScript rules of std::regex, as
// llama.cpp's unicode_regex_split_stl uses it: whole matches only, no
// captures, over codepoints (or llama.cpp's one-byte "collapsed" codes).
// Supported: | ( ) (?: ) (?= ) (?! ) [ ] [^ ] . ^ $ \b \B * + ? {n} {n,}
// {n,m}, lazy quantifiers, \d \D \s \S \w \W \r \n \t \v \f \0 \xHH \uHHHH,
// and any other escaped character as itself. Classes are ASCII, as in the
// "C" locale. Unknown syntax aborts, as std::regex throws.
import std;
namespace ecma;

// A class member: a range, or one of the class escapes.
struct Item { lo: u32, hi: u32, kind: Escape }
enum Escape { Range, Digit, NotDigit, Space, NotSpace, Word, NotWord }

struct Class { first: i64, count: i64, negated: bool }

// The program. Op.Split tries x first, then y.
struct Inst { op: Op, x: i64, y: i64 }
enum Op {
    Char, Any, Class, Split, Jmp, Bol, Eol,
    Wordb,      // y = 1 for \B
    Look,       // the lookahead's program follows; x = where to go after it, y = 1 if negative
    Match,
}

struct Prog { insts: const Inst[:], items: const Item[:], classes: const Class[:] }

// The parse tree, children linked through `next`.
struct Node { kind: NodeKind, a: i64, b: i64, greedy: bool, child: i64, next: i64 }
enum NodeKind {
    Empty, Char, Any, Class, Seq, Alt,
    Repeat,     // a = min, b = max or -1
    Look,       // a = 1 if negative
    Bol, Eol,
    Wordb,      // a = 1 for \B
}

let END = 0xFFFFFFFF;

fn hex_value(c: u32) -> i64 {
    if c >= '0' && c <= '9' { return c - '0'; }
    if c >= 'a' && c <= 'f' { return c - 'a' + 10; }
    if c >= 'A' && c <= 'F' { return c - 'A' + 10; }
    return -1;
}

fn is_word_char(c: u32) -> bool {
    (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'
}

fn is_space_char(c: u32) -> bool { c == ' ' || (c >= 0x09 && c <= 0x0D) }

fn item_matches(it: Item, c: u32) -> bool {
    match it.kind {
        Range => it.lo <= c && c <= it.hi,
        Digit => c >= '0' && c <= '9',
        NotDigit => !(c >= '0' && c <= '9'),
        Space => is_space_char(c),
        NotSpace => !is_space_char(c),
        Word => is_word_char(c),
        _ => !is_word_char(c),
    }
}

fn class_matches(prog: Prog&, k: i64, c: u32) -> bool {
    let cl = prog.classes[k];
    var found = false;
    for i in cl.first..cl.first + cl.count {
        if item_matches(prog.items[i], c) { found = true; break; }
    }
    return found != cl.negated;
}

// Compiles pattern into insts, items and classes (which should start empty).
fn compile(pattern: const u32[:], insts: Inst[>..]&, items: Item[>..]&, classes: Class[>..]&) {
    var nodes: Node[>..] = [];
    var pos = 0;

    fn fail(why: const u8[:]) {
        var text: u8[>..] = [];
        for c in pattern { if c < 0x80 { text.push(c as! u8); } else { format(text, "\\u{", c, "}"); } }
        abort(str("regex: ", why, " at ", pos, " in ", text));
    }
    fn peek() -> u32 { if pos < pattern.len { pattern[pos] } else { END } }
    fn peek_at(k: i64) -> u32 { if pos + k < pattern.len { pattern[pos + k] } else { END } }
    fn node(kind: NodeKind, a: i64 = 0, b: i64 = 0) -> i64 {
        nodes.push(Node { kind: kind, a: a, b: b, greedy: true, child: -1, next: -1 });
        return nodes.len - 1;
    }
    fn class_of(kind: Escape) -> i64 {
        classes.push(Class { first: items.len, count: 1, negated: false });
        items.push(Item { lo: 0, hi: 0, kind: kind });
        return node(NodeKind.Class, classes.len - 1);
    }

    // A decimal number at pos; -1 if there is none.
    fn number() -> i64 {
        var n = -1;
        while peek() >= '0' && peek() <= '9' {
            n = (if n < 0 { 0 } else { n * 10 }) + (peek() - '0');
            pos += 1;
        }
        return n;
    }

    // The character an escape stands for, after the backslash; `kind` is
    // Escape.Range for a character, or the class escape it is.
    fn escape(in_class: bool) -> u32, Escape {
        let c = peek();
        guard c != END else { fail("trailing backslash"); }
        pos += 1;
        match c {
            'd' => { return 0, Escape.Digit; },
            'D' => { return 0, Escape.NotDigit; },
            's' => { return 0, Escape.Space; },
            'S' => { return 0, Escape.NotSpace; },
            'w' => { return 0, Escape.Word; },
            'W' => { return 0, Escape.NotWord; },
            'r' => { return '\r', Escape.Range; },
            'n' => { return '\n', Escape.Range; },
            't' => { return '\t', Escape.Range; },
            'v' => { return 0x0B, Escape.Range; },
            'f' => { return 0x0C, Escape.Range; },
            '0' => { return 0, Escape.Range; },
            'x', 'u' => {
                let digits = if c == 'x' { 2 } else { 4 };
                var v: u32 = 0;
                for i in digits {
                    let h = hex_value(peek());
                    guard h >= 0 else { fail("bad hex escape"); }
                    v = v * 16 + h as u32;
                    pos += 1;
                }
                return v, Escape.Range;
            },
            _ => {},
        }
        if in_class && c == 'b' { return 0x08, Escape.Range; }
        return c, Escape.Range;
    }

    // [...] after the '['.
    fn parse_class() -> i64 {
        var cl = Class { first: items.len, count: 0, negated: false };
        if peek() == '^' { cl.negated = true; pos += 1; }
        fn member() -> u32, Escape {
            let c = peek();
            pos += 1;
            if c == '\\' { return escape(true); }
            return c, Escape.Range;
        }
        while peek() != ']' {
            guard peek() != END else { fail("unterminated class"); }
            let lo, kind = member();
            if kind == Escape.Range && peek() == '-' && peek_at(1) != ']' && peek_at(1) != END {
                pos += 1;
                let hi, kind2 = member();
                if kind2 == Escape.Range {
                    guard lo <= hi else { fail("bad class range"); }
                    items.push(Item { lo: lo, hi: hi, kind: Escape.Range });
                } else {
                    items.push(Item { lo: lo, hi: lo, kind: Escape.Range });
                    items.push(Item { lo: '-', hi: '-', kind: Escape.Range });
                    items.push(Item { lo: 0, hi: 0, kind: kind2 });
                }
            } else {
                items.push(Item { lo: lo, hi: lo, kind: kind });
            }
        }
        pos += 1;
        cl.count = items.len - cl.first;
        classes.push(cl);
        return node(NodeKind.Class, classes.len - 1);
    }

    // alt := seq ('|' seq)*
    recursive fn parse_alt() -> i64 {
        let first = parse_seq();
        guard peek() == '|' else { return first; }
        let alt = node(NodeKind.Alt);
        nodes[alt].child = first;
        var last = first;
        while peek() == '|' {
            pos += 1;
            let s = parse_seq();
            nodes[last].next = s;
            last = s;
        }
        return alt;
    }

    // seq := term*
    recursive fn parse_seq() -> i64 {
        let seq = node(NodeKind.Seq);
        var last = -1;
        while peek() != END && peek() != '|' && peek() != ')' {
            let t = parse_term();
            if last < 0 { nodes[seq].child = t; } else { nodes[last].next = t; }
            last = t;
        }
        return seq;
    }

    // term := assertion | atom quantifier?
    recursive fn parse_term() -> i64 {
        let c = peek();
        if c == '^' { pos += 1; return node(NodeKind.Bol); }
        if c == '$' { pos += 1; return node(NodeKind.Eol); }
        if c == '\\' && (peek_at(1) == 'b' || peek_at(1) == 'B') {
            pos += 2;
            return node(NodeKind.Wordb, if pattern[pos - 1] == 'B' { 1 } else { 0 });
        }
        if c == '(' && peek_at(1) == '?' && (peek_at(2) == '=' || peek_at(2) == '!') {
            let negative = peek_at(2) == '!';
            pos += 3;
            let inner = parse_alt();
            guard peek() == ')' else { fail("missing )"); }
            pos += 1;
            let look = node(NodeKind.Look, if negative { 1 } else { 0 });
            nodes[look].child = inner;
            return look;
        }
        let atom = parse_atom();
        var min = -1;
        var max = -1;
        let q = peek();
        if q == '*' { min = 0; max = -1; pos += 1; }
        else if q == '+' { min = 1; max = -1; pos += 1; }
        else if q == '?' { min = 0; max = 1; pos += 1; }
        else if q == '{' {
            pos += 1;
            min = number();
            guard min >= 0 else { fail("bad quantifier"); }
            max = min;
            if peek() == ',' { pos += 1; max = number(); }
            guard peek() == '}' && (max < 0 || max >= min) else { fail("bad quantifier"); }
            pos += 1;
        }
        guard min >= 0 else { return atom; }
        let rep = node(NodeKind.Repeat, min, max);
        nodes[rep].child = atom;
        if peek() == '?' { nodes[rep].greedy = false; pos += 1; }
        return rep;
    }

    // atom := '.' | '(' alt ')' | '(?:' alt ')' | class | escape | character
    recursive fn parse_atom() -> i64 {
        let c = peek();
        if c == '(' {
            pos += 1;
            if peek() == '?' {
                guard peek_at(1) == ':' else { fail("unsupported group"); }
                pos += 2;
            }
            let inner = parse_alt();
            guard peek() == ')' else { fail("missing )"); }
            pos += 1;
            return inner;
        }
        if c == '[' { pos += 1; return parse_class(); }
        if c == '.' { pos += 1; return node(NodeKind.Any); }
        if c == '\\' {
            pos += 1;
            let ch, kind = escape(false);
            if kind != Escape.Range { return class_of(kind); }
            return node(NodeKind.Char, ch);
        }
        guard c != '*' && c != '+' && c != '?' && c != '{' && c != ')' else { fail("nothing to repeat"); }
        pos += 1;
        return node(NodeKind.Char, c);
    }

    fn push(op: Op, x: i64 = 0, y: i64 = 0) -> i64 {
        insts.push(Inst { op: op, x: x, y: y });
        return insts.len - 1;
    }

    // Emits the program for node n.
    recursive fn emit(n: i64) {
        let nd = nodes[n];
        match nd.kind {
            Empty => {},
            Char => { push(Op.Char, nd.a); },
            Any => { push(Op.Any); },
            Class => { push(Op.Class, nd.a); },
            Bol => { push(Op.Bol); },
            Eol => { push(Op.Eol); },
            Wordb => { push(Op.Wordb, 0, nd.a); },
            Seq => {
                var k = nd.child;
                while k >= 0 { emit(k); k = nodes[k].next; }
            },
            Alt => {
                // Each branch but the last: Op.Split this, next; branch; Op.Jmp end.
                // The JMPs to patch are chained through their x.
                var chain = -1;
                var k = nd.child;
                while nodes[k].next >= 0 {
                    let split = push(Op.Split);
                    insts[split].x = split + 1;
                    emit(k);
                    chain = push(Op.Jmp, chain);
                    insts[split].y = insts.len;
                    k = nodes[k].next;
                }
                emit(k);
                while chain >= 0 {
                    let prev = insts[chain].x;
                    insts[chain].x = insts.len;
                    chain = prev;
                }
            },
            Repeat => {
                for i in nd.a { emit(nd.child); }
                if nd.b < 0 {
                    let split = push(Op.Split);
                    emit(nd.child);
                    push(Op.Jmp, split);
                    if nd.greedy { insts[split].x = split + 1; insts[split].y = insts.len; }
                    else { insts[split].x = insts.len; insts[split].y = split + 1; }
                } else {
                    // Each optional copy skips to the end; the skips are chained through y.
                    var chain = -1;
                    for i in nd.b - nd.a {
                        let split = push(Op.Split, 0, chain);
                        chain = split;
                        emit(nd.child);
                    }
                    let end = insts.len;
                    while chain >= 0 {
                        let prev = insts[chain].y;
                        if nd.greedy { insts[chain].x = chain + 1; insts[chain].y = end; }
                        else { insts[chain].x = end; insts[chain].y = chain + 1; }
                        chain = prev;
                    }
                }
            },
            Look => {
                let look = push(Op.Look, 0, nd.a);
                emit(nd.child);
                push(Op.Match);
                insts[look].x = insts.len;
            },
        }
    }

    let root = parse_alt();
    guard pos == pattern.len else { fail("unmatched )"); }
    emit(root);
    push(Op.Match);
}

struct Frame { pc: i64, pos: i64 }

// Where a match of the program from pc ends, starting at pos; -1 for none.
// The target is text[lo..hi]: ^ and $ match only at its ends, and \b sees
// nothing outside it. A match ending at not_null_at does not count.
recursive fn run(prog: Prog&, text: const u32[:], lo: i64, hi: i64, pc0: i64, pos0: i64,
                 not_null_at: i64, stack: Frame[>..<]&) -> i64 {
    let mark = stack.len;
    var pc = pc0;
    var pos = pos0;
    loop {
        let ins = prog.insts[pc];
        var ok = true;
        match ins.op {
            Char => { ok = pos < hi && text[pos] == ins.x; if ok { pc += 1; pos += 1; } },
            Any => {
                ok = pos < hi && text[pos] != '\n' && text[pos] != '\r' && text[pos] != 0x2028 && text[pos] != 0x2029;
                if ok { pc += 1; pos += 1; }
            },
            Class => { ok = pos < hi && class_matches(prog, ins.x, text[pos]); if ok { pc += 1; pos += 1; } },
            Split => { stack.push(Frame { pc: ins.y, pos: pos }); pc = ins.x; },
            Jmp => { pc = ins.x; },
            Bol => { ok = pos == lo; pc += 1; },
            Eol => { ok = pos == hi; pc += 1; },
            Wordb => {
                let before = pos > lo && is_word_char(text[pos - 1]);
                let after = pos < hi && is_word_char(text[pos]);
                ok = (before != after) != (ins.y == 1);
                pc += 1;
            },
            Look => {
                let r = run(prog, text, lo, hi, pc + 1, pos, -1, stack);
                ok = (r >= 0) != (ins.y == 1);
                pc = ins.x;
            },
            Match => {
                if pos != not_null_at { stack.resize(mark); return pos; }
                ok = false;
            },
        }
        if !ok {
            if stack.len == mark { return -1; }
            let f = stack.pop();
            pc = f.pc;
            pos = f.pos;
        }
    }
}

// std::regex_search over text[from..hi] of the target text[lo..hi]: the
// first match's start and end, or -1, -1. `continuous` only tries at from;
// `not_null` refuses an empty match there.
fn search(prog: Prog&, text: const u32[:], lo: i64, hi: i64, from: i64, continuous: bool,
          not_null: bool, stack: Frame[>..<]&) -> i64, i64 {
    for s in from..hi + 1 {
        let e = run(prog, text, lo, hi, 0, s, if not_null { s } else { -1 }, stack);
        if e >= 0 { return s, e; }
        if continuous { break; }
    }
    return -1, -1;
}

// unicode_regex_split_stl: within each piece of text (offsets are the
// pieces' lengths), the matches of prog and the gaps between them become
// pieces of their own, in order. Empty matches give empty pieces.
fn split(prog: Prog&, text: const u32[:], offsets: const i64[:], out: i64[>..]&) {
    var stack: Frame[>..<] = [];
    var start = 0;
    for offset in offsets {
        let lo = start;
        let hi = start + offset;
        var start_idx = 0;
        var s, e = search(prog, text, lo, hi, lo, false, false, stack);
        while s >= 0 {
            let at = s - lo;
            if at > start_idx { out.push(at - start_idx); }
            out.push(e - s);
            start_idx = at + e - s;
            // std::regex_iterator: after an empty match, first try a non-empty
            // one at the same place, then search on from the next character.
            var from = e;
            if e == s {
                if e == hi { break; }
                let s2, e2 = search(prog, text, lo, hi, e, true, true, stack);
                if s2 >= 0 { s = s2; e = e2; continue; }
                from = e + 1;
            }
            s, e = search(prog, text, lo, hi, from, false, false, stack);
        }
        if start_idx < offset { out.push(offset - start_idx); }
        start = hi;
    }
}
