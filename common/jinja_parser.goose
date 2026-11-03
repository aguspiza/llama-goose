// llama.cpp's Jinja engine: the parser, tokens to a syntax tree. The nodes
// of every template parsed live in `nodes`; a node's lists of children
// (a body, arguments, items) are ranges of `alist`.
// port-of: common/jinja/parser.cpp, common/jinja/runtime.h (the node types)
import std;
import .jinja;
import .number;
import .jinja_lexer;

namespace jinja;

let N_PROGRAM = 0;
let N_IF = 1;            // x test, l0 body, l1 alternate
let N_FOR = 2;           // x loop variable(s), y iterable, l0 body, l1 the else block
let N_BREAK = 3;
let N_CONTINUE = 4;
let N_NOOP = 5;
let N_SET = 6;           // x assignee, y value (or -1 and the body l0)
let N_MACRO = 7;         // x name, l0 parameters, l1 body
let N_COMMENT = 8;
let N_BLANK = 9;         // the empty a[]
let N_MEMBER = 10;       // x object, y property; flag: computed, a[b] rather than a.b
let N_CALL = 11;         // x callee, l0 arguments
let N_IDENT = 12;        // s the name
let N_INT = 13;
let N_FLOAT = 14;
let N_STRING = 15;       // s the text
let N_ARRAY = 16;        // l0 items
let N_TUPLE = 17;        // l0 items
let N_OBJECT = 18;       // l0 keys and values, in turn
let N_BINARY = 19;       // s operator, x left, y right
let N_FILTER = 20;       // x operand, y filter (an identifier or a call)
let N_FILTER_STMT = 21;  // x filter, l0 body
let N_SELECT = 22;       // x lhs, y test: `lhs if test`
let N_TEST = 23;         // x operand, y test (an identifier or a call); flag: negated
let N_UNARY = 24;        // s operator, x argument
let N_SLICE = 25;        // x start, y stop, z step (-1 when left out)
let N_KWARG = 26;        // x key, y value
let N_SPREAD = 27;       // x argument
let N_CALL_STMT = 28;    // x the call, l0 the caller's parameters, l1 body
let N_TERNARY = 29;      // x condition, y if true, z if false

struct Node {
    kind: u8,
    flag: bool = false,
    pos: i64 = 0,
    x: i64 = -1,
    y: i64 = -1,
    z: i64 = -1,
    l0: i64 = 0,
    n0: i64 = 0,
    l1: i64 = 0,
    n1: i64 = 0,
    s: i64 = 0,
    sn: i64 = 0,
    i: i64 = 0,
    f: f64 = 0.0,
}

// A parsed template: its normalized source (in asrc) and its statements.
struct Program { src: i64 = 0, src_len: i64 = 0, body: i64 = 0, n: i64 = 0 }

var nodes: Node[>..] = [];
var alist: i64[>..] = [];
// Children being collected, each list above its parent's.
var pstack: i64[>..<] = [];

// The statement::type() of a node.
fn node_type(kind: u8) -> const u8[:] {
    return match kind {
        N_PROGRAM => "Program", N_IF => "If", N_FOR => "For", N_BREAK => "Break",
        N_CONTINUE => "Continue", N_NOOP => "Noop", N_SET => "Set", N_MACRO => "Macro",
        N_COMMENT => "Comment", N_BLANK => "BlankExpression", N_MEMBER => "MemberExpression",
        N_CALL => "CallExpression", N_IDENT => "Identifier", N_INT => "IntegerLiteral",
        N_FLOAT => "FloatLiteral", N_STRING => "StringLiteral", N_ARRAY => "ArrayLiteral",
        N_TUPLE => "TupleLiteral", N_OBJECT => "ObjectLiteral", N_BINARY => "BinaryExpression",
        N_FILTER => "FilterExpression", N_FILTER_STMT => "FilterStatement",
        N_SELECT => "SelectExpression", N_TEST => "TestExpression", N_UNARY => "UnaryExpression",
        N_SLICE => "SliceExpression", N_KWARG => "KeywordArgumentExpression",
        N_SPREAD => "SpreadExpression", N_CALL_STMT => "CallStatement", _ => "Ternary",
    };
}

fn node_text(n: i64) -> const u8[:] { atext[nodes[n].s..nodes[n].s + nodes[n].sn] }

fn list_item(l: i64, k: i64) -> i64 { alist[l + k] }

// The parser's state: the token it is at, how deep it is, the source.
var cur = 0;
var depth = 0;
var psrc = 0;
var psrc_len = 0;

fn psource() -> const u8[:] { asrc[psrc..psrc + psrc_len] }

// A parser_exception.
fn parse_fail(msg: const u8[:], pos: i64) {
    var out: u8[>..] = [];
    format(out, "parser: ", msg, "\n");
    peak_source(out, psource(), pos);
    fail_plain(out);
}

fn enter() {
    depth++;
    if depth > LIMIT { parse_fail(str("Parser Error: template is nested more than ", LIMIT, " levels deep"), peek(0).pos); }
}

fn leave() { depth--; }

fn peek(offset: i64) -> Tok {
    if cur + offset >= toks.len { return Tok { t: T_EOF, s: 0, n: 0, pos: 0 }; }
    return toks[cur + offset];
}

fn next() -> Tok {
    if cur >= toks.len { parse_fail("Parser Error: Unexpected EOF", if toks.len == 0 { 0 } else { toks[toks.len - 1].pos }); }
    cur++;
    return toks[cur - 1];
}

fn expect(t: u8, error: const u8[:]) -> Tok {
    let tk = peek(0);
    if tk.t != t { parse_fail(str("Parser Error: ", error, " (Got ", tok_text(tk), ")"), tk.pos); }
    cur++;
    return tk;
}

fn expect_identifier(name: const u8[:]) {
    let tk = peek(0);
    if tk.t != T_IDENT || tok_text(tk) != name { parse_fail(str("Expected identifier: ", name), tk.pos); }
    cur++;
}

fn is(t: u8) -> bool { peek(0).t == t }

fn is_identifier(name: const u8[:]) -> bool { peek(0).t == T_IDENT && tok_text(peek(0)) == name }

fn is_statement(a: const u8[:], b: const u8[:], c: const u8[:]) -> bool {
    if peek(0).t != T_OPEN_STMT || peek(1).t != T_IDENT { return false; }
    let v = tok_text(peek(1));
    return v == a || v == b || v == c;
}

// A node whose position is token start's.
fn mk_node(kind: u8, start: i64) -> i64 {
    let pos = if start < toks.len { toks[start].pos } else { 0 };
    nodes.push(Node { kind: kind, pos: pos, .. });
    return nodes.len - 1;
}

// A node with a token's text.
fn mk_text_node(kind: u8, start: i64, t: Tok) -> i64 {
    let n = mk_node(kind, start);
    let s = atext.len;
    for i in t.n { atext.push(atext[t.s + i]); }
    nodes[n].s = s;
    nodes[n].sn = t.n;
    return n;
}

// The children pushed on pstack since mark, as a list.
fn take_list(mark: i64) -> i64, i64 {
    let start = alist.len;
    for k in mark..pstack.len { alist.push(pstack[k]); }
    let n = pstack.len - mark;
    pstack.resize(mark);
    return start, n;
}

// Parses the tokens in toks of the source asrc[src..src + len].
fn parse(src: i64, len: i64) -> Program {
    cur = 0;
    depth = 0;
    psrc = src;
    psrc_len = len;
    pstack.clear();
    let mark = pstack.len;
    while cur < toks.len { pstack.push(parse_any()); }
    let body, n = take_list(mark);
    return Program { src: src, src_len: len, body: body, n: n };
}

// The statements up to {% a %}, {% b %} or {% c %}, as a list.
recursive fn parse_until(a: const u8[:], b: const u8[:], c: const u8[:]) -> i64, i64 {
    let mark = pstack.len;
    while !is_statement(a, b, c) { pstack.push(parse_any()); }
    return take_list(mark);
}

recursive fn parse_any() -> i64 {
    enter();
    let start = cur;
    let t = peek(0);
    var r = -1;
    if t.t == T_COMMENT { r = mk_text_node(N_COMMENT, start, next()); }
    else if t.t == T_TEXT { r = mk_text_node(N_STRING, start, next()); }
    else if t.t == T_OPEN_STMT { r = parse_statement(); }
    else if t.t == T_OPEN_EXPR {
        expect(T_OPEN_EXPR, "Expected {{");
        r = parse_expression();
        expect(T_CLOSE_EXPR, "Expected }}");
    }
    else { fail_plain("Unexpected token type"); }
    leave();
    return r;
}

recursive fn parse_statement() -> i64 {
    expect(T_OPEN_STMT, "Expected {%");
    if peek(0).t != T_IDENT { fail_plain("Unknown statement"); }
    let start = cur;
    let name_tok = next();
    let name = tok_text(name_tok);
    var r = -1;
    if name == "set" {
        r = parse_set(start);
    } else if name == "if" {
        r = parse_if(start);
        end_block("endif");
    } else if name == "macro" {
        r = mk_node(N_MACRO, start);
        let id = parse_primary();
        nodes[r].x = id;
        let args, nargs = parse_args();
        nodes[r].l0 = args;
        nodes[r].n0 = nargs;
        expect(T_CLOSE_STMT, "Expected %}");
        let body, nbody = parse_until("endmacro", "endmacro", "endmacro");
        nodes[r].l1 = body;
        nodes[r].n1 = nbody;
        end_block("endmacro");
    } else if name == "for" {
        r = parse_for(start);
        end_block("endfor");
    } else if name == "break" || name == "continue" {
        expect(T_CLOSE_STMT, "Expected %}");
        r = mk_node(if name == "break" { N_BREAK } else { N_CONTINUE }, start);
    } else if name == "call" {
        var cargs = 0;
        var ncargs = 0;
        if is(T_OPEN_PAREN) {
            let ca, nca = parse_args();
            cargs = ca;
            ncargs = nca;
        }
        let callee = parse_primary();
        if nodes[callee].kind != N_IDENT { fail_plain("Expected identifier"); }
        let args, nargs = parse_args();
        expect(T_CLOSE_STMT, "Expected %}");
        let body, nbody = parse_until("endcall", "endcall", "endcall");
        end_block("endcall");
        let call = mk_node(N_CALL, start);
        nodes[call].x = callee;
        nodes[call].l0 = args;
        nodes[call].n0 = nargs;
        r = mk_node(N_CALL_STMT, start);
        nodes[r].x = call;
        nodes[r].l0 = cargs;
        nodes[r].n0 = ncargs;
        nodes[r].l1 = body;
        nodes[r].n1 = nbody;
    } else if name == "filter" {
        var filter = parse_primary();
        if nodes[filter].kind == N_IDENT && is(T_OPEN_PAREN) { filter = parse_call(filter); }
        expect(T_CLOSE_STMT, "Expected %}");
        let body, nbody = parse_until("endfilter", "endfilter", "endfilter");
        end_block("endfilter");
        r = mk_node(N_FILTER_STMT, start);
        nodes[r].x = filter;
        nodes[r].l0 = body;
        nodes[r].n0 = nbody;
    } else if name == "generation" || name == "endgeneration" {
        // transformers' generation blocks, which mark what the assistant wrote
        r = mk_node(N_NOOP, start);
        cur++;
    } else {
        fail_plain(str("Unknown statement: ", name));
    }
    return r;
}

// {% name %}, the end of a block.
fn end_block(name: const u8[:]) {
    expect(T_OPEN_STMT, "Expected {%");
    expect_identifier(name);
    expect(T_CLOSE_STMT, "Expected %}");
}

recursive fn parse_set(start: i64) -> i64 {
    // set is both a declaration and an assignment
    let left = parse_sequence(false);
    var value = -1;
    var body = 0;
    var nbody = 0;
    if is(T_EQUALS) {
        cur++;
        value = parse_sequence(false);
    } else {
        // {% set x %}...{% endset %}
        expect(T_CLOSE_STMT, "Expected %}");
        let b, nb = parse_until("endset", "endset", "endset");
        body = b;
        nbody = nb;
        expect(T_OPEN_STMT, "Expected {%");
        expect_identifier("endset");
    }
    expect(T_CLOSE_STMT, "Expected %}");
    let r = mk_node(N_SET, start);
    nodes[r].x = left;
    nodes[r].y = value;
    nodes[r].l0 = body;
    nodes[r].n0 = nbody;
    return r;
}

recursive fn parse_if(start: i64) -> i64 {
    let test = parse_expression();
    expect(T_CLOSE_STMT, "Expected %}");
    let body, nbody = parse_until("elif", "else", "endif");
    var alt = 0;
    var nalt = 0;
    if is_statement("elif", "elif", "elif") {
        let pos0 = cur;
        cur += 2;
        enter();
        let nested = parse_if(pos0);
        leave();
        let mark = pstack.len;
        pstack.push(nested);
        let a, na = take_list(mark);
        alt = a;
        nalt = na;
    } else if is_statement("else", "else", "else") {
        cur += 2;
        expect(T_CLOSE_STMT, "Expected %}");
        let a, na = parse_until("endif", "endif", "endif");
        alt = a;
        nalt = na;
    }
    let r = mk_node(N_IF, start);
    nodes[r].x = test;
    nodes[r].l0 = body;
    nodes[r].n0 = nbody;
    nodes[r].l1 = alt;
    nodes[r].n1 = nalt;
    return r;
}

// One expression, or several separated by commas: a tuple.
recursive fn parse_sequence(primary: bool) -> i64 {
    let start = cur;
    let first = if primary { parse_primary() } else { parse_expression() };
    if !is(T_COMMA) { return first; }
    let mark = pstack.len;
    pstack.push(first);
    while is(T_COMMA) {
        cur++;
        let e = if primary { parse_primary() } else { parse_expression() };
        pstack.push(e);
    }
    let l, n = take_list(mark);
    let r = mk_node(N_TUPLE, start);
    nodes[r].l0 = l;
    nodes[r].n0 = n;
    return r;
}

recursive fn parse_for(start: i64) -> i64 {
    let loopvar = parse_sequence(true);
    if !is_identifier("in") { fail_plain("Expected 'in'"); }
    cur++;
    let iterable = parse_expression();
    expect(T_CLOSE_STMT, "Expected %}");
    let body, nbody = parse_until("endfor", "else", "endfor");
    var alt = 0;
    var nalt = 0;
    if is_statement("else", "else", "else") {
        cur += 2;
        expect(T_CLOSE_STMT, "Expected %}");
        let a, na = parse_until("endfor", "endfor", "endfor");
        alt = a;
        nalt = na;
    }
    let r = mk_node(N_FOR, start);
    nodes[r].x = loopvar;
    nodes[r].y = iterable;
    nodes[r].l0 = body;
    nodes[r].n0 = nbody;
    nodes[r].l1 = alt;
    nodes[r].n1 = nalt;
    return r;
}

recursive fn parse_expression() -> i64 {
    enter();
    let r = parse_if_expression();
    leave();
    return r;
}

// a if test else b; `a if test` alone selects.
recursive fn parse_if_expression() -> i64 {
    let a = parse_or();
    if !is_identifier("if") { return a; }
    let start = cur;
    cur++;
    let test = parse_or();
    if is_identifier("else") {
        let pos0 = cur;
        cur++;
        let b = parse_expression();
        let r = mk_node(N_TERNARY, pos0);
        nodes[r].x = test;
        nodes[r].y = a;
        nodes[r].z = b;
        return r;
    }
    let r = mk_node(N_SELECT, start);
    nodes[r].x = a;
    nodes[r].y = test;
    return r;
}

// A binary node for the operator token at start.
fn mk_binary(start: i64, op: Tok, left: i64, right: i64) -> i64 {
    let r = mk_text_node(N_BINARY, start, op);
    nodes[r].x = left;
    nodes[r].y = right;
    return r;
}

recursive fn parse_or() -> i64 {
    var left = parse_and();
    while is_identifier("or") {
        let start = cur;
        let op = next();
        let right = parse_and();
        left = mk_binary(start, op, left, right);
    }
    return left;
}

recursive fn parse_and() -> i64 {
    var left = parse_not();
    while is_identifier("and") {
        let start = cur;
        let op = next();
        let right = parse_not();
        left = mk_binary(start, op, left, right);
    }
    return left;
}

recursive fn parse_not() -> i64 {
    if !is_identifier("not") { return parse_comparison(); }
    enter();
    let start = cur;
    let op = next();
    let arg = parse_not();
    let r = mk_text_node(N_UNARY, start, op);
    nodes[r].x = arg;
    leave();
    return r;
}

// Comparisons and membership, one precedence:
// 'a' in 'apple' == 'b' in 'banana' is 'a' in ('apple' == ('b' in 'banana')).
recursive fn parse_comparison() -> i64 {
    var left = parse_additive();
    loop {
        let start = cur;
        var r = -1;
        if is_identifier("not") && peek(1).t == T_IDENT && tok_text(peek(1)) == "in" {
            cur += 2;
            r = mk_node(N_BINARY, start);
            nodes[r].s = atext.len;
            nodes[r].sn = 6;
            atext.append("not in");
        } else if is_identifier("in") || is(T_COMPARISON) {
            let op = next();
            r = mk_text_node(N_BINARY, start, op);
        } else {
            break;
        }
        let right = parse_additive();
        nodes[r].x = left;
        nodes[r].y = right;
        left = r;
    }
    return left;
}

recursive fn parse_additive() -> i64 {
    var left = parse_multiplicative();
    while is(T_ADDITIVE) {
        let start = cur;
        let op = next();
        let right = parse_multiplicative();
        left = mk_binary(start, op, left, right);
    }
    return left;
}

recursive fn parse_multiplicative() -> i64 {
    var left = parse_test();
    while is(T_MULTIPLICATIVE) {
        let start = cur;
        let op = next();
        let right = parse_test();
        left = mk_binary(start, op, left, right);
    }
    return left;
}

recursive fn parse_test() -> i64 {
    var operand = parse_filter();
    while is_identifier("is") {
        let start = cur;
        cur++;
        var negate = false;
        if is_identifier("not") { cur++; negate = true; }
        var test = parse_primary();
        if is(T_OPEN_PAREN) {
            test = parse_call(test);
        } else if is(T_NUMBER) || is(T_STRING) || is(T_OPEN_CURLY) || is(T_OPEN_SQUARE) ||
                  (is(T_IDENT) && !is_identifier("and") && !is_identifier("or") && !is_identifier("else")) {
            // `x is divisibleby 3`: one argument, without parentheses
            let call_pos = cur;
            let arg = parse_unary();
            let mark = pstack.len;
            pstack.push(arg);
            let l, n = take_list(mark);
            let call = mk_node(N_CALL, call_pos);
            nodes[call].x = test;
            nodes[call].l0 = l;
            nodes[call].n0 = n;
            test = call;
        }
        let r = mk_node(N_TEST, start);
        nodes[r].x = operand;
        nodes[r].y = test;
        nodes[r].flag = negate;
        operand = r;
    }
    return operand;
}

// Filters bind outside unary operators: -n|abs is (-n)|abs.
recursive fn parse_filter() -> i64 {
    var operand = parse_unary();
    while is(T_PIPE) {
        let start = cur;
        cur++;
        var filter = parse_primary();
        if is(T_OPEN_PAREN) { filter = parse_call(filter); }
        let r = mk_node(N_FILTER, start);
        nodes[r].x = operand;
        nodes[r].y = filter;
        operand = r;
    }
    return operand;
}

recursive fn parse_unary() -> i64 {
    if !is(T_UNARY) {
        let member = parse_member(parse_primary());
        return if is(T_OPEN_PAREN) { parse_call(member) } else { member };
    }
    enter();
    let start = cur;
    let op = next();
    let arg = parse_unary();
    let r = mk_text_node(N_UNARY, start, op);
    nodes[r].x = arg;
    leave();
    return r;
}

// callee(args), and any .member, [index] or (args) after it.
recursive fn parse_call(callee: i64) -> i64 {
    let start = cur;
    let args, nargs = parse_args();
    let call = mk_node(N_CALL, start);
    nodes[call].x = callee;
    nodes[call].l0 = args;
    nodes[call].n0 = nargs;
    let member = parse_member(call);
    if !is(T_OPEN_PAREN) { return member; }
    enter();
    let r = parse_call(member);
    leave();
    return r;
}

// (a, b, key=value, *spread)
recursive fn parse_args() -> i64, i64 {
    expect(T_OPEN_PAREN, "Expected (");
    let mark = pstack.len;
    while !is(T_CLOSE_PAREN) {
        var arg = -1;
        if peek(0).t == T_MULTIPLICATIVE && tok_text(peek(0)) == "*" {
            let start = cur;
            cur++;
            let e = parse_expression();
            arg = mk_node(N_SPREAD, start);
            nodes[arg].x = e;
        } else {
            arg = parse_expression();
            if is(T_EQUALS) {
                let start = cur;
                cur++;
                let v = parse_expression();
                let kw = mk_node(N_KWARG, start);
                nodes[kw].x = arg;
                nodes[kw].y = v;
                arg = kw;
            }
        }
        pstack.push(arg);
        if is(T_COMMA) { cur++; }
    }
    expect(T_CLOSE_PAREN, "Expected )");
    return take_list(mark);
}

recursive fn parse_member(object: i64) -> i64 {
    let start = cur;
    var obj = object;
    while is(T_DOT) || is(T_OPEN_SQUARE) {
        let op = next();
        let computed = op.t == T_OPEN_SQUARE;
        var prop = -1;
        if computed {
            prop = parse_member_arguments();
            expect(T_CLOSE_SQUARE, "Expected ]");
        } else {
            prop = parse_primary();
        }
        let r = mk_node(N_MEMBER, start);
        nodes[r].x = obj;
        nodes[r].y = prop;
        nodes[r].flag = computed;
        obj = r;
    }
    return obj;
}

// What is between [ and ]: an index, or a slice start:stop:step.
recursive fn parse_member_arguments() -> i64 {
    let start = cur;
    let mark = pstack.len;
    var is_slice = false;
    while !is(T_CLOSE_SQUARE) {
        if is(T_COLON) {
            pstack.push(-1);
            cur++;
            is_slice = true;
        } else {
            pstack.push(parse_expression());
            if is(T_COLON) { cur++; is_slice = true; }
        }
    }
    let n = pstack.len - mark;
    if is_slice {
        let r = mk_node(N_SLICE, start);
        nodes[r].x = if n > 0 { pstack[mark] } else { -1 };
        nodes[r].y = if n > 1 { pstack[mark + 1] } else { -1 };
        nodes[r].z = if n > 2 { pstack[mark + 2] } else { -1 };
        pstack.resize(mark);
        return r;
    }
    if n == 0 { return mk_node(N_BLANK, start); }
    let r = pstack[mark];
    pstack.resize(mark);
    return r;
}

recursive fn parse_primary() -> i64 {
    let start = cur;
    let t = next();
    if t.t == T_NUMBER {
        let s = tok_text(t);
        if find(s, ".") >= 0 {
            let r = mk_node(N_FLOAT, start);
            var used = 0;
            nodes[r].f = number::c_strtod(s, used);
            return r;
        }
        let v, ok = parse_int(s);
        if !ok { fail_plain("stoll"); }
        let r = mk_node(N_INT, start);
        nodes[r].i = v;
        return r;
    }
    if t.t == T_STRING {
        // adjacent literals are one: 'a' 'b' is 'ab'
        let r = mk_text_node(N_STRING, start, t);
        while is(T_STRING) {
            let more = next();
            for i in more.n { atext.push(atext[more.s + i]); }
            nodes[r].sn += more.n;
        }
        return r;
    }
    if t.t == T_IDENT { return mk_text_node(N_IDENT, start, t); }
    if t.t == T_OPEN_PAREN {
        let e = parse_sequence(false);
        expect(T_CLOSE_PAREN, "Expected )");
        return e;
    }
    if t.t == T_OPEN_SQUARE {
        let mark = pstack.len;
        while !is(T_CLOSE_SQUARE) {
            pstack.push(parse_expression());
            if is(T_COMMA) { cur++; }
        }
        cur++;
        let l, n = take_list(mark);
        let r = mk_node(N_ARRAY, start);
        nodes[r].l0 = l;
        nodes[r].n0 = n;
        return r;
    }
    if t.t == T_OPEN_CURLY {
        let mark = pstack.len;
        while !is(T_CLOSE_CURLY) {
            pstack.push(parse_expression());
            expect(T_COLON, "Expected :");
            pstack.push(parse_expression());
            if is(T_COMMA) { cur++; }
        }
        cur++;
        let l, n = take_list(mark);
        let r = mk_node(N_OBJECT, start);
        nodes[r].l0 = l;
        nodes[r].n0 = n / 2;
        return r;
    }
    fail_plain(str("Unexpected token: ", tok_text(t), " of type ", t.t));
    return -1;
}
