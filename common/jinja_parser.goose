// llama.cpp's Jinja engine: the parser, tokens to a syntax tree. The nodes
// of every template parsed live in `nodes`; a node's lists of children
// (a body, arguments, items) are ranges of `alist`.
// port-of: common/jinja/parser.cpp, common/jinja/runtime.h (the node types)
import std;
import .jinja;
import .number;
import .jinja_lexer;

namespace jinja;

enum NodeKind {
    Program,
    If,          // x test, l0 body, l1 alternate
    For,         // x loop variable(s), y iterable, l0 body, l1 the else block
    Break,
    Continue,
    Noop,
    Set,         // x assignee, y value (or -1 and the body l0)
    Macro,       // x name, l0 parameters, l1 body
    Comment,
    Blank,       // the empty a[]
    Member,      // x object, y property; flag: computed, a[b] rather than a.b
    Call,        // x callee, l0 arguments
    Ident,       // s the name
    Int,
    Float,
    String,      // s the text
    Array,       // l0 items
    Tuple,       // l0 items
    Object,      // l0 keys and values, in turn
    Binary,      // s operator, x left, y right
    Filter,      // x operand, y filter (an identifier or a call)
    FilterStmt,  // x filter, l0 body
    Select,      // x lhs, y test: `lhs if test`
    Test,        // x operand, y test (an identifier or a call); flag: negated
    Unary,       // s operator, x argument
    Slice,       // x start, y stop, z step (-1 when left out)
    Kwarg,       // x key, y value
    Spread,      // x argument
    CallStmt,    // x the call, l0 the caller's parameters, l1 body
    Ternary,     // x condition, y if true, z if false
    Chain,       // l0 operands, l1 operators (NodeKind.Binary nodes, only s is used): 1 <= x <= 100
}

struct Node {
    kind: NodeKind,
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
fn node_type(kind: NodeKind) -> const u8[:] {
    return match kind {
        Program => "Program", If => "If", For => "For", Break => "Break",
        Continue => "Continue", Noop => "Noop", Set => "Set", Macro => "Macro",
        Comment => "Comment", Blank => "BlankExpression", Member => "MemberExpression",
        Call => "CallExpression", Ident => "Identifier", Int => "IntegerLiteral",
        Float => "FloatLiteral", String => "StringLiteral", Array => "ArrayLiteral",
        Tuple => "TupleLiteral", Object => "ObjectLiteral", Binary => "BinaryExpression",
        Filter => "FilterExpression", FilterStmt => "FilterStatement",
        Select => "SelectExpression", Test => "TestExpression", Unary => "UnaryExpression",
        Slice => "SliceExpression", Kwarg => "KeywordArgumentExpression",
        Spread => "SpreadExpression", CallStmt => "CallStatement", Chain => "ComparisonChainExpression",
        _ => "Ternary",
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
    if cur + offset >= toks.len { return Tok { t: TokType.Eof, s: 0, n: 0, pos: 0 }; }
    return toks[cur + offset];
}

fn next() -> Tok {
    if cur >= toks.len { parse_fail("Parser Error: Unexpected EOF", if toks.len == 0 { 0 } else { toks[toks.len - 1].pos }); }
    cur++;
    return toks[cur - 1];
}

fn expect(t: TokType, error: const u8[:]) -> Tok {
    let tk = peek(0);
    if tk.t != t { parse_fail(str("Parser Error: ", error, " (Got ", tok_text(tk), ")"), tk.pos); }
    cur++;
    return tk;
}

fn expect_identifier(name: const u8[:]) {
    let tk = peek(0);
    if tk.t != TokType.Ident || tok_text(tk) != name { parse_fail(str("Expected identifier: ", name), tk.pos); }
    cur++;
}

fn is(t: TokType) -> bool { peek(0).t == t }

fn is_identifier(name: const u8[:]) -> bool { peek(0).t == TokType.Ident && tok_text(peek(0)) == name }

fn is_statement(a: const u8[:], b: const u8[:], c: const u8[:]) -> bool {
    if peek(0).t != TokType.OpenStmt || peek(1).t != TokType.Ident { return false; }
    let v = tok_text(peek(1));
    return v == a || v == b || v == c;
}

// A node whose position is token start's.
fn mk_node(kind: NodeKind, start: i64) -> i64 {
    let pos = if start < toks.len { toks[start].pos } else { 0 };
    nodes.push(Node { kind: kind, pos: pos, .. });
    return nodes.len - 1;
}

// A node with a token's text.
fn mk_text_node(kind: NodeKind, start: i64, t: Tok) -> i64 {
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
    if t.t == TokType.Comment { r = mk_text_node(NodeKind.Comment, start, next()); }
    else if t.t == TokType.Text { r = mk_text_node(NodeKind.String, start, next()); }
    else if t.t == TokType.OpenStmt { r = parse_statement(); }
    else if t.t == TokType.OpenExpr {
        expect(TokType.OpenExpr, "Expected {{");
        r = parse_expression();
        expect(TokType.CloseExpr, "Expected }}");
    }
    else { fail_plain("Unexpected token type"); }
    leave();
    return r;
}

recursive fn parse_statement() -> i64 {
    expect(TokType.OpenStmt, "Expected {%");
    if peek(0).t != TokType.Ident { fail_plain("Unknown statement"); }
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
        r = mk_node(NodeKind.Macro, start);
        let id = parse_primary();
        nodes[r].x = id;
        let args, nargs = parse_args(false);
        nodes[r].l0 = args;
        nodes[r].n0 = nargs;
        expect(TokType.CloseStmt, "Expected %}");
        let body, nbody = parse_until("endmacro", "endmacro", "endmacro");
        nodes[r].l1 = body;
        nodes[r].n1 = nbody;
        end_block("endmacro");
    } else if name == "for" {
        r = parse_for(start);
        end_block("endfor");
    } else if name == "break" || name == "continue" {
        expect(TokType.CloseStmt, "Expected %}");
        r = mk_node(if name == "break" { NodeKind.Break } else { NodeKind.Continue }, start);
    } else if name == "call" {
        var cargs = 0;
        var ncargs = 0;
        if is(TokType.OpenParen) {
            let ca, nca = parse_args(false);
            cargs = ca;
            ncargs = nca;
        }
        let callee = parse_primary();
        if nodes[callee].kind != NodeKind.Ident { fail_plain("Expected identifier"); }
        let args, nargs = parse_args();
        expect(TokType.CloseStmt, "Expected %}");
        let body, nbody = parse_until("endcall", "endcall", "endcall");
        end_block("endcall");
        let call = mk_node(NodeKind.Call, start);
        nodes[call].x = callee;
        nodes[call].l0 = args;
        nodes[call].n0 = nargs;
        r = mk_node(NodeKind.CallStmt, start);
        nodes[r].x = call;
        nodes[r].l0 = cargs;
        nodes[r].n0 = ncargs;
        nodes[r].l1 = body;
        nodes[r].n1 = nbody;
    } else if name == "filter" {
        var filter = parse_primary();
        if nodes[filter].kind == NodeKind.Ident && is(TokType.OpenParen) { filter = parse_call(filter); }
        expect(TokType.CloseStmt, "Expected %}");
        let body, nbody = parse_until("endfilter", "endfilter", "endfilter");
        end_block("endfilter");
        r = mk_node(NodeKind.FilterStmt, start);
        nodes[r].x = filter;
        nodes[r].l0 = body;
        nodes[r].n0 = nbody;
    } else if name == "generation" || name == "endgeneration" {
        // transformers' generation blocks, which mark what the assistant wrote
        r = mk_node(NodeKind.Noop, start);
        cur++;
    } else {
        fail_plain(str("Unknown statement: ", name));
    }
    return r;
}

// {% name %}, the end of a block.
fn end_block(name: const u8[:]) {
    expect(TokType.OpenStmt, "Expected {%");
    expect_identifier(name);
    expect(TokType.CloseStmt, "Expected %}");
}

recursive fn parse_set(start: i64) -> i64 {
    // set is both a declaration and an assignment
    let left = parse_sequence(false);
    var value = -1;
    var body = 0;
    var nbody = 0;
    if is(TokType.Equals) {
        cur++;
        value = parse_sequence(false);
    } else {
        // {% set x %}...{% endset %}
        expect(TokType.CloseStmt, "Expected %}");
        let b, nb = parse_until("endset", "endset", "endset");
        body = b;
        nbody = nb;
        expect(TokType.OpenStmt, "Expected {%");
        expect_identifier("endset");
    }
    expect(TokType.CloseStmt, "Expected %}");
    let r = mk_node(NodeKind.Set, start);
    nodes[r].x = left;
    nodes[r].y = value;
    nodes[r].l0 = body;
    nodes[r].n0 = nbody;
    return r;
}

recursive fn parse_if(start: i64) -> i64 {
    let test = parse_expression();
    expect(TokType.CloseStmt, "Expected %}");
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
        expect(TokType.CloseStmt, "Expected %}");
        let a, na = parse_until("endif", "endif", "endif");
        alt = a;
        nalt = na;
    }
    let r = mk_node(NodeKind.If, start);
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
    if !is(TokType.Comma) { return first; }
    let mark = pstack.len;
    pstack.push(first);
    while is(TokType.Comma) {
        cur++;
        let e = if primary { parse_primary() } else { parse_expression() };
        pstack.push(e);
    }
    let l, n = take_list(mark);
    let r = mk_node(NodeKind.Tuple, start);
    nodes[r].l0 = l;
    nodes[r].n0 = n;
    return r;
}

recursive fn parse_for(start: i64) -> i64 {
    let loopvar = parse_sequence(true);
    if !is_identifier("in") { fail_plain("Expected 'in'"); }
    cur++;
    let iterable = parse_expression();
    expect(TokType.CloseStmt, "Expected %}");
    let body, nbody = parse_until("endfor", "else", "endfor");
    var alt = 0;
    var nalt = 0;
    if is_statement("else", "else", "else") {
        cur += 2;
        expect(TokType.CloseStmt, "Expected %}");
        let a, na = parse_until("endfor", "endfor", "endfor");
        alt = a;
        nalt = na;
    }
    let r = mk_node(NodeKind.For, start);
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
        let r = mk_node(NodeKind.Ternary, pos0);
        nodes[r].x = test;
        nodes[r].y = a;
        nodes[r].z = b;
        return r;
    }
    let r = mk_node(NodeKind.Select, start);
    nodes[r].x = a;
    nodes[r].y = test;
    return r;
}

// A binary node for the operator token at start.
fn mk_binary(start: i64, op: Tok, left: i64, right: i64) -> i64 {
    let r = mk_text_node(NodeKind.Binary, start, op);
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
    let r = mk_text_node(NodeKind.Unary, start, op);
    nodes[r].x = arg;
    leave();
    return r;
}

// Comparisons and membership, one precedence, chained as in Python:
// 'a' in 'apple' == 'b' in 'banana' is 'a' in 'apple' and 'apple' == 'b' and 'b' in 'banana'.
recursive fn parse_comparison() -> i64 {
    let first = parse_additive();
    let pos = cur;
    // operator, operand, operator, operand, ... on pstack: no list may live across the recursion
    let mark = pstack.len;
    loop {
        var r = -1;
        if is_identifier("not") && peek(1).t == TokType.Ident && tok_text(peek(1)) == "in" {
            r = mk_node(NodeKind.Binary, cur);
            cur += 2;
            nodes[r].s = atext.len;
            nodes[r].sn = 6;
            atext.append("not in");
        } else if is_identifier("in") || is(TokType.Comparison) {
            let op_pos = cur;
            let op = next();
            r = mk_text_node(NodeKind.Binary, op_pos, op);
        } else {
            break;
        }
        pstack.push(r);
        let operand = parse_additive();
        pstack.push(operand);
    }
    let n_ops = (pstack.len - mark) / 2;
    if n_ops == 0 { return first; }
    if n_ops == 1 {
        let r = pstack[mark];
        nodes[r].pos = pos;
        nodes[r].x = first;
        nodes[r].y = pstack[mark + 1];
        pstack.resize(mark);
        return r;
    }
    var ops: i64[>..] = [];
    var operands: i64[>..] = [];
    operands.push(first);
    for k in n_ops { ops.push(pstack[mark + 2 * k]); operands.push(pstack[mark + 2 * k + 1]); }
    pstack.resize(mark);
    let r = mk_node(NodeKind.Chain, pos);
    for o in operands { pstack.push(o); }
    let l0, n0 = take_list(mark);
    for o in ops { pstack.push(o); }
    let l1, n1 = take_list(mark);
    nodes[r].l0 = l0;
    nodes[r].n0 = n0;
    nodes[r].l1 = l1;
    nodes[r].n1 = n1;
    return r;
}

recursive fn parse_additive() -> i64 {
    var left = parse_multiplicative();
    while is(TokType.Additive) {
        let start = cur;
        let op = next();
        let right = parse_multiplicative();
        left = mk_binary(start, op, left, right);
    }
    return left;
}

recursive fn parse_multiplicative() -> i64 {
    var left = parse_test();
    while is(TokType.Multiplicative) {
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
        if is(TokType.OpenParen) {
            test = parse_call(test);
        } else if is(TokType.Number) || is(TokType.String) || is(TokType.OpenCurly) || is(TokType.OpenSquare) ||
                  (is(TokType.Ident) && !is_identifier("and") && !is_identifier("or") && !is_identifier("else")) {
            // `x is divisibleby 3`: one argument, without parentheses
            let call_pos = cur;
            let arg = parse_unary();
            let mark = pstack.len;
            pstack.push(arg);
            let l, n = take_list(mark);
            let call = mk_node(NodeKind.Call, call_pos);
            nodes[call].x = test;
            nodes[call].l0 = l;
            nodes[call].n0 = n;
            test = call;
        }
        let r = mk_node(NodeKind.Test, start);
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
    while is(TokType.Pipe) {
        let start = cur;
        cur++;
        var filter = parse_primary();
        if is(TokType.OpenParen) { filter = parse_call(filter); }
        let r = mk_node(NodeKind.Filter, start);
        nodes[r].x = operand;
        nodes[r].y = filter;
        operand = r;
    }
    return operand;
}

// A sign where an operand is expected is unary: -x, 1 + -x, x if c else -1.
recursive fn parse_unary() -> i64 {
    if !(is(TokType.Additive) && tok_text(peek(0)) != "~") {
        let member = parse_member(parse_primary());
        return if is(TokType.OpenParen) { parse_call(member) } else { member };
    }
    enter();
    let start = cur;
    let op = next();
    let arg = parse_unary();
    let r = mk_text_node(NodeKind.Unary, start, op);
    nodes[r].x = arg;
    leave();
    return r;
}

// callee(args), and any .member, [index] or (args) after it.
recursive fn parse_call(callee: i64) -> i64 {
    let start = cur;
    let args, nargs = parse_args();
    let call = mk_node(NodeKind.Call, start);
    nodes[call].x = callee;
    nodes[call].l0 = args;
    nodes[call].n0 = nargs;
    let member = parse_member(call);
    if !is(TokType.OpenParen) { return member; }
    enter();
    let r = parse_call(member);
    leave();
    return r;
}

// (a, b, key=value, *array, **object); without spreads in a macro's or a
// caller's parameters. A spread's flag: ** (keyword arguments).
recursive fn parse_args(allow_spread: bool = true) -> i64, i64 {
    expect(TokType.OpenParen, "Expected (");
    let mark = pstack.len;
    while !is(TokType.CloseParen) {
        var arg = -1;
        let t = peek(0);
        if t.t == TokType.Power || (t.t == TokType.Multiplicative && tok_text(t) == "*") {
            if !allow_spread { fail_plain(str("Unexpected token: ", tok_text(t), " of type ", t.t)); }
            let start = cur;
            cur++;
            let e = parse_expression();
            arg = mk_node(NodeKind.Spread, start);
            nodes[arg].x = e;
            nodes[arg].flag = t.t == TokType.Power;
        } else {
            arg = parse_expression();
            if is(TokType.Equals) {
                let start = cur;
                cur++;
                let v = parse_expression();
                let kw = mk_node(NodeKind.Kwarg, start);
                nodes[kw].x = arg;
                nodes[kw].y = v;
                arg = kw;
            }
        }
        pstack.push(arg);
        if is(TokType.Comma) { cur++; }
    }
    expect(TokType.CloseParen, "Expected )");
    return take_list(mark);
}

recursive fn parse_member(object: i64) -> i64 {
    let start = cur;
    var obj = object;
    while is(TokType.Dot) || is(TokType.OpenSquare) {
        let op = next();
        let computed = op.t == TokType.OpenSquare;
        var prop = -1;
        if computed {
            prop = parse_member_arguments();
            expect(TokType.CloseSquare, "Expected ]");
        } else {
            prop = parse_primary();
        }
        let r = mk_node(NodeKind.Member, start);
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
    while !is(TokType.CloseSquare) {
        if is(TokType.Colon) {
            pstack.push(-1);
            cur++;
            is_slice = true;
        } else {
            pstack.push(parse_expression());
            if is(TokType.Colon) { cur++; is_slice = true; }
        }
    }
    let n = pstack.len - mark;
    if is_slice {
        let r = mk_node(NodeKind.Slice, start);
        nodes[r].x = if n > 0 { pstack[mark] } else { -1 };
        nodes[r].y = if n > 1 { pstack[mark + 1] } else { -1 };
        nodes[r].z = if n > 2 { pstack[mark + 2] } else { -1 };
        pstack.resize(mark);
        return r;
    }
    if n == 0 { return mk_node(NodeKind.Blank, start); }
    let r = pstack[mark];
    pstack.resize(mark);
    return r;
}

recursive fn parse_primary() -> i64 {
    let start = cur;
    let t = next();
    if t.t == TokType.Number {
        let s = tok_text(t);
        if find(s, ".") >= 0 {
            let r = mk_node(NodeKind.Float, start);
            var used = 0;
            nodes[r].f = number::c_strtod(s, used);
            return r;
        }
        let v, ok = parse_int(s);
        if !ok { fail_plain("stoll"); }
        let r = mk_node(NodeKind.Int, start);
        nodes[r].i = v;
        return r;
    }
    if t.t == TokType.String {
        // adjacent literals are one: 'a' 'b' is 'ab'
        let r = mk_text_node(NodeKind.String, start, t);
        while is(TokType.String) {
            let more = next();
            for i in more.n { atext.push(atext[more.s + i]); }
            nodes[r].sn += more.n;
        }
        return r;
    }
    if t.t == TokType.Ident { return mk_text_node(NodeKind.Ident, start, t); }
    if t.t == TokType.OpenParen {
        let e = parse_sequence(false);
        expect(TokType.CloseParen, "Expected )");
        return e;
    }
    if t.t == TokType.OpenSquare {
        let mark = pstack.len;
        while !is(TokType.CloseSquare) {
            pstack.push(parse_expression());
            if is(TokType.Comma) { cur++; }
        }
        cur++;
        let l, n = take_list(mark);
        let r = mk_node(NodeKind.Array, start);
        nodes[r].l0 = l;
        nodes[r].n0 = n;
        return r;
    }
    if t.t == TokType.OpenCurly {
        let mark = pstack.len;
        while !is(TokType.CloseCurly) {
            pstack.push(parse_expression());
            expect(TokType.Colon, "Expected :");
            pstack.push(parse_expression());
            if is(TokType.Comma) { cur++; }
        }
        cur++;
        let l, n = take_list(mark);
        let r = mk_node(NodeKind.Object, start);
        nodes[r].l0 = l;
        nodes[r].n0 = n / 2;
        return r;
    }
    fail_plain(str("Unexpected token: ", tok_text(t), " of type ", t.t));
    return -1;
}
