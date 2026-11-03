// llama.cpp's Jinja engine: the lexer, a template's source to tokens. As
// chat templates are rendered (lstrip_blocks and trim_blocks on), a block
// takes the indentation before it and the newline after it.
// port-of: common/jinja/lexer.h, common/jinja/lexer.cpp
import std;
import .jinja;

namespace jinja;

let T_EOF = 0;
let T_TEXT = 1;          // the text between statements and expressions
let T_NUMBER = 2;        // 123, 1.0
let T_STRING = 3;        // 'string'
let T_IDENT = 4;         // variables, functions, statement names, booleans
let T_EQUALS = 5;        // =
let T_OPEN_PAREN = 6;
let T_CLOSE_PAREN = 7;
let T_OPEN_STMT = 8;     // {%
let T_CLOSE_STMT = 9;    // %}
let T_OPEN_EXPR = 10;    // {{
let T_CLOSE_EXPR = 11;   // }}
let T_OPEN_SQUARE = 12;
let T_CLOSE_SQUARE = 13;
let T_OPEN_CURLY = 14;
let T_CLOSE_CURLY = 15;
let T_COMMA = 16;
let T_DOT = 17;
let T_COLON = 18;
let T_PIPE = 19;
let T_ADDITIVE = 21;     // + - ~
let T_MULTIPLICATIVE = 22;   // * / %
let T_COMPARISON = 23;   // < > <= >= == !=
let T_UNARY = 24;        // - +
let T_COMMENT = 25;      // {# ... #}

// A token: its type, its value (atext[s..s + n]) and where it starts.
struct Tok { t: u8, s: i64, n: i64, pos: i64 }

var toks: Tok[>..] = [];
// The normalized sources of the templates parsed so far.
var asrc: u8[>..] = [];
// The text of tokens and of the syntax tree: names, literals, operators.
var atext: u8[>..] = [];

fn tok_text(t: Tok) -> const u8[:] { atext[t.s..t.s + t.n] }

struct Seq { s: const u8[:], t: u8 }

// In the order they are tried.
let SEQUENCES: Seq[31] = [
    Seq { "{%-", T_OPEN_STMT }, Seq { "-%}", T_CLOSE_STMT },
    Seq { "{{-", T_OPEN_EXPR }, Seq { "-}}", T_CLOSE_EXPR },
    Seq { "{%", T_OPEN_STMT }, Seq { "%}", T_CLOSE_STMT },
    Seq { "{{", T_OPEN_EXPR }, Seq { "}}", T_CLOSE_EXPR },
    Seq { "(", T_OPEN_PAREN }, Seq { ")", T_CLOSE_PAREN },
    Seq { "{", T_OPEN_CURLY }, Seq { "}", T_CLOSE_CURLY },
    Seq { "[", T_OPEN_SQUARE }, Seq { "]", T_CLOSE_SQUARE },
    Seq { ",", T_COMMA }, Seq { ".", T_DOT }, Seq { ":", T_COLON }, Seq { "|", T_PIPE },
    Seq { "<=", T_COMPARISON }, Seq { ">=", T_COMPARISON }, Seq { "==", T_COMPARISON },
    Seq { "!=", T_COMPARISON }, Seq { "<", T_COMPARISON }, Seq { ">", T_COMPARISON },
    Seq { "+", T_ADDITIVE }, Seq { "-", T_ADDITIVE }, Seq { "~", T_ADDITIVE },
    Seq { "*", T_MULTIPLICATIVE }, Seq { "/", T_MULTIPLICATIVE }, Seq { "%", T_MULTIPLICATIVE },
    Seq { "=", T_EQUALS },
];

// C's isspace and isalnum, in the C locale.
fn c_space(c: u8) -> bool { c == ' ' || (c >= 9 && c <= 13) }
// What a -%} or {%- strips: " \t\r\n".
fn strip_space(c: u8) -> bool { c == ' ' || c == '\t' || c == '\r' || c == '\n' }
fn c_digit(c: u8) -> bool { c >= '0' && c <= '9' }
fn c_word(c: u8) -> bool { (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c_digit(c) || c == '_' }

// A lexer_exception: the message and where in the source.
fn lex_fail(msg: const u8[:], source: const u8[:], pos: i64) {
    var out: u8[>..] = [];
    format(out, "lexer: ", msg, "\n");
    peak_source(out, source, pos);
    fail_plain(out);
}

let W_DIGITS = 0;
let W_WORD = 1;
let W_SPACE = 2;
let W_NOT_QUOTE = 3;

struct Lexer { source: const u8[:], src: const u8[:], pos: i64 = 0 }

// src[p], or 0 past the end as std::string gives.
fn at(lx: Lexer&, p: i64) -> u8 { if p < lx.src.len { lx.src[p] } else { 0 } }

fn wanted(lx: Lexer&, what: i64, quote: u8) -> bool {
    let c = at(lx, lx.pos);
    if what == W_DIGITS { return c_digit(c); }
    if what == W_WORD { return c_word(c); }
    if what == W_SPACE { return c_space(c); }
    return c != quote;
}

// Consumes the characters `what` takes, unescaping backslash escapes, onto
// atext: where they start there and how many.
fn consume_while(lx: Lexer&, what: i64, quote: u8) -> i64, i64 {
    let start = atext.len;
    while wanted(lx, what, quote) {
        if at(lx, lx.pos) == '\\' {
            lx.pos++;
            if lx.pos >= lx.src.len { lex_fail("unexpected end of input after escape character", lx.source, lx.pos); }
            let e = lx.src[lx.pos];
            lx.pos++;
            if e == 'x' {
                if lx.pos + 1 >= lx.src.len { lex_fail("incomplete hexadecimal escape", lx.source, lx.pos); }
                let hi = hex_value(lx.src[lx.pos]);
                let lo = hex_value(lx.src[lx.pos + 1]);
                if hi < 0 || lo < 0 { lex_fail("invalid hexadecimal escape", lx.source, lx.pos); }
                push_utf8(atext, hi * 16 + lo);
                lx.pos += 2;
                continue;
            }
            let u: u8 = match e {
                'n' => '\n', 't' => '\t', 'r' => '\r', 'b' => 8, 'f' => 12, 'v' => 11,
                '\\' => '\\', '\'' => '\'', '"' => '"', _ => 0,
            };
            if u == 0 {
                var msg: u8[>..] = [];
                msg.append("unknown escape character \\");
                msg.push(e);
                lex_fail(msg, lx.source, lx.pos);
            }
            atext.push(u);
            continue;
        }
        atext.push(at(lx, lx.pos));
        lx.pos++;
        if lx.pos > lx.src.len { lex_fail("unexpected end of input during consume_while", lx.source, lx.pos); }
    }
    return start, atext.len - start;
}

fn hex_value(c: u8) -> i64 {
    if c >= '0' && c <= '9' { return c - '0'; }
    if c >= 'a' && c <= 'f' { return c - 'a' + 10; }
    if c >= 'A' && c <= 'F' { return c - 'A' + 10; }
    return -1;
}

// Digits, and a fraction if a digit follows the point.
fn consume_numeric(lx: Lexer&) -> i64, i64 {
    let start, digits = consume_while(lx, W_DIGITS, 0);
    if lx.pos < lx.src.len && lx.src[lx.pos] == '.' && lx.pos + 1 < lx.src.len && c_digit(lx.src[lx.pos + 1]) {
        lx.pos++;
        atext.push('.');
        consume_while(lx, W_DIGITS, 0);
    }
    return start, atext.len - start;
}

// Whether src[pos + n] is one of chars (and pos + n is not the last position).
fn next_is(lx: Lexer&, chars: const u8[:], n: i64) -> bool {
    if lx.pos + n >= lx.src.len { return false; }
    let c = lx.src[lx.pos + n];
    for x in chars { if c == x { return true; } }
    return false;
}

fn push_tok(t: u8, s: i64, n: i64, pos: i64) { toks.push(Tok { t: t, s: s, n: n, pos: pos }); }

fn last_type() -> u8 { if toks.len == 0 { T_EOF } else { toks[toks.len - 1].t } }

// Tokenizes source into toks. The source is kept, normalized (\r\n and \r
// to \n, one trailing newline dropped), in asrc: where it is.
fn tokenize(source: const u8[:]) -> i64, i64 {
    toks.clear();
    let off = asrc.len;
    if source.len == 0 { return off, 0; }
    var i = 0;
    while i < source.len {
        let c = source[i];
        if c == '\r' && i + 1 < source.len && source[i + 1] == '\n' { i++; continue; }
        asrc.push(if c == '\r' { '\n' } else { c });
        i++;
    }
    if source[source.len - 1] == '\n' { asrc.pop(); }
    let len = asrc.len - off;
    lex(source, asrc[off..off + len]);
    return off, len;
}

fn lex(source: const u8[:], src: const u8[:]) {
    var lx = Lexer { source: source, src: src };
    var curly_depth: u64 = 0;
    while lx.pos < src.len {
        var start_pos = lx.pos;
        // the text outside statements and expressions
        let last = if toks.len == 0 { T_CLOSE_STMT } else { last_type() };
        if last == T_CLOSE_STMT || last == T_CLOSE_EXPR || last == T_COMMENT {
            var can_rm_newline = false;
            var rstrip = false;
            if lx.pos > 3 {
                let c0 = src[lx.pos - 3];
                let c1 = src[lx.pos - 2];
                let c2 = src[lx.pos - 1];
                // -%} -}} -#}: strip the text's leading space
                rstrip = c0 == '-' && (c1 == '%' || c1 == '}' || c1 == '#') && c2 == '}';
                // #} %} -}: trim_blocks takes one newline (not after }})
                can_rm_newline = (c1 == '#' || c1 == '%' || c1 == '-') && c2 == '}';
            }
            let start = lx.pos;
            var end = start;
            while lx.pos < src.len && !(src[lx.pos] == '{' && next_is(lx, "%{#", 1)) {
                lx.pos++;
                end = lx.pos;
            }
            // lstrip_blocks: a block takes the space before it on its line
            if at(lx, lx.pos) == '{' && next_is(lx, "%#-", 1) {
                var current = end;
                while current > start {
                    let c = src[current - 1];
                    if current == 1 { end = 0; break; }
                    if c == '\n' { end = current; break; }
                    if !c_space(c) { break; }
                    current--;
                }
            }
            var t0 = start;
            var t1 = end;
            // trim_blocks: the newline after a block
            if can_rm_newline && t1 > t0 && src[t0] == '\n' { t0++; }
            if rstrip { while t0 < t1 && strip_space(src[t0]) { t0++; } }
            let lstrip = at(lx, lx.pos) == '{' && next_is(lx, "{%#", 1) && next_is(lx, "-", 2);
            if lstrip { while t1 > t0 && strip_space(src[t1 - 1]) { t1--; } }
            if t1 > t0 {
                let s = atext.len;
                atext.append(src[t0..t1]);
                push_tok(T_TEXT, s, t1 - t0, start_pos);
                continue;
            }
        }

        // a comment
        if at(lx, lx.pos) == '{' && next_is(lx, "#", 1) {
            start_pos = lx.pos;
            lx.pos += 2;
            let s = atext.len;
            while !(at(lx, lx.pos) == '#' && next_is(lx, "}", 1)) {
                if lx.pos + 2 >= src.len { lex_fail("missing end of comment tag", source, lx.pos); }
                atext.push(src[lx.pos]);
                lx.pos++;
            }
            push_tok(T_COMMENT, s, atext.len - s, start_pos);
            lx.pos += 2;
            continue;
        }

        // the - of {%- or {{-
        if at(lx, lx.pos) == '-' && (last == T_OPEN_EXPR || last == T_OPEN_STMT) {
            lx.pos++;
            if lx.pos >= src.len { break; }
        }

        // space inside a statement or expression
        while c_space(at(lx, lx.pos)) { lx.pos++; }
        if lx.pos >= src.len { break; }

        let ch = src[lx.pos];
        let closing = ch == '-' && next_is(lx, "%}", 1);

        // a unary operator, or the sign of a number
        if !closing && (ch == '-' || ch == '+') {
            start_pos = lx.pos;
            let lt = last_type();
            if lt == T_TEXT || lt == T_EOF {
                var msg: u8[>..] = [];
                msg.append("unexpected character: ");
                msg.push(ch);
                lex_fail(msg, source, lx.pos);
            }
            if lt != T_IDENT && lt != T_NUMBER && lt != T_STRING && lt != T_CLOSE_PAREN && lt != T_CLOSE_SQUARE {
                lx.pos++;
                let s = atext.len;
                atext.push(ch);
                let ns, nn = consume_numeric(lx);
                push_tok(if nn == 0 { T_UNARY } else { T_NUMBER }, s, atext.len - s, start_pos);
                continue;
            }
        }

        // the operators and brackets
        var matched = false;
        for seq in SEQUENCES {
            start_pos = lx.pos;
            // inside an object literal }} is two closing braces
            if seq.s == "}}" && curly_depth > 0 { continue; }
            if lx.pos + seq.s.len <= src.len && src[lx.pos..lx.pos + seq.s.len] == seq.s {
                let s = atext.len;
                atext.append(seq.s);
                push_tok(seq.t, s, seq.s.len, start_pos);
                if seq.t == T_OPEN_EXPR { curly_depth = 0; }
                else if seq.t == T_OPEN_CURLY { curly_depth++; }
                else if seq.t == T_CLOSE_CURLY { curly_depth--; }
                lx.pos += seq.s.len;
                matched = true;
                break;
            }
        }
        if matched { continue; }

        if ch == '\'' || ch == '"' {
            start_pos = lx.pos;
            lx.pos++;
            let s, n = consume_while(lx, W_NOT_QUOTE, ch);
            push_tok(T_STRING, s, n, start_pos);
            lx.pos++;
            continue;
        }
        if c_digit(ch) {
            start_pos = lx.pos;
            let s, n = consume_numeric(lx);
            push_tok(T_NUMBER, s, n, start_pos);
            continue;
        }
        if c_word(ch) {
            start_pos = lx.pos;
            let s, n = consume_while(lx, W_WORD, 0);
            push_tok(T_IDENT, s, n, start_pos);
            continue;
        }
        var msg: u8[>..] = [];
        msg.append("unexpected character: ");
        msg.push(ch);
        lex_fail(msg, source, lx.pos);
    }
}
