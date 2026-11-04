// llama.cpp's Jinja engine: the runtime, which executes a syntax tree. A
// scope is an object of variables, copied for each loop iteration, macro
// call and call block, so assignments inside one stay inside it.
// port-of: common/jinja/runtime.cpp, common/jinja/runtime.h (runtime, gather_string_parts)
import std;
import exception;
import .jinja;
import .number;
import .jinja_lexer;
import .jinja_parser;
import .jinja_builtins;

namespace jinja;

// A scope: its variables, and how deep in macro calls it is.
struct Ctx { env: i64, depth: i64 }

// A call's arguments: argstack[at..at + n]. name is the function's (a
// string, or -1), for error messages.
struct Args { ctx: Ctx, at: i64, n: i64, name: i64 = -1 }

fn arg(a: Args, k: i64) -> i64 { argstack[a.at + k] }

// The render in progress: the template's source, whether to record what it
// reads (caps_get), the time strftime_now gives.
var run_src = 0;
var run_src_len = 0;
var stats = false;
var now: i64 = 0;

// A {% break %} or {% continue %} on its way to its loop.
let SIG_NONE = 0;
let SIG_BREAK = 1;
let SIG_CONTINUE = 2;
var signal = SIG_NONE;

fn run_source() -> const u8[:] { asrc[run_src..run_src + run_src_len] }

// A new context: true, false and none defined (context::context).
fn new_ctx() -> Ctx {
    let env = mk_object();
    vals[env].builtins = false;
    obj_set_name(env, "true", mk_bool(true));
    obj_set_name(env, "True", mk_bool(true));
    obj_set_name(env, "false", mk_bool(false));
    obj_set_name(env, "False", mk_bool(false));
    obj_set_name(env, "none", mk_none());
    obj_set_name(env, "None", mk_none());
    now = number::now();
    return Ctx { env: env, depth: 0 };
}

// A scope inside c: a copy of its variables.
fn scope(c: Ctx) -> Ctx { Ctx { env: obj_copy(c.env), depth: c.depth } }

fn get_var(c: Ctx, name: const u8[:]) -> i64 {
    let v = obj_get_name(c.env, name);
    return if v >= 0 { v } else { mk_undef_hint(name) };
}

fn set_var(c: Ctx, name: const u8[:], v: i64) { obj_set_name(c.env, name, v); }

// Starts rendering program p.
fn begin(p: Program) {
    run_src = p.src;
    run_src_len = p.src_len;
    signal = SIG_NONE;
    cur_node = -1;
    argstack.clear();
    gather.clear();
}

// Runs program p in c (runtime::execute and gather_string_parts): its
// output, a string whose neighbouring parts of the same kind are joined.
fn render(p: Program, c: Ctx) -> i64 {
    begin(p);
    let results = mk_array();
    for k in p.n {
        let v = exec(list_item(p.body, k), c);
        if signal != SIG_NONE {
            fail_plain(if signal == SIG_BREAK { "Break statement executed" } else { "Continue statement executed" });
        }
        arr_push(results, v);
    }
    return gather_string_parts(results);
}

// The value's text, joined (gather_string_parts).
fn gather_string_parts(v: i64) -> i64 {
    let mark = gather.len;
    gather_value(v, 0);
    let first = parts.len;
    var k = mark;
    while k < gather.len {
        let p = gather[k];
        var e = k + 1;
        while e < gather.len && gather[e].input == p.input { e++; }
        if e == k + 1 {
            parts.push(p);
        } else {
            let off = text.len;
            for j in k..e {
                let q = gather[j];
                for i in q.len { text.push(text[q.off + i]); }
            }
            parts.push(Part { off: off, len: text.len - off, input: p.input });
        }
        k = e;
    }
    gather.resize(mark);
    return mk_parts(first);
}

// Pushes the text of v onto gather: strings and numbers as they are, an
// array's items in turn; nothing for the rest.
recursive fn gather_value(v: i64, depth: i64) {
    let x = vals[v];
    if x.kind == K_STRING {
        for k in x.n { gather.push(parts[x.a + k]); }
    } else if x.kind == K_INT || x.kind == K_FLOAT || x.kind == K_BOOL {
        let s = as_string(v);
        let y = vals[s];
        for k in y.n { gather.push(parts[y.a + k]); }
    } else if x.kind == K_ARRAY || x.kind == K_TUPLE {
        if depth >= LIMIT { fail(str("value is nested more than ", LIMIT, " levels deep")); }
        for k in x.n { gather_value(items[x.a + k], depth + 1); }
    }
}

// The parts gathered since mark, as a string.
fn take_gathered(mark: i64) -> i64 {
    let first = parts.len;
    for k in mark..gather.len { parts.push(gather[k]); }
    gather.resize(mark);
    return mk_parts(first);
}

// Executes the statements, their output one string (exec_statements). A
// break or continue stops them.
recursive fn exec_list(l: i64, n: i64, c: Ctx) -> i64 {
    let mark = gather.len;
    for k in n {
        let v = exec(list_item(l, k), c);
        if signal != SIG_NONE { break; }
        gather_value(v, 0);
    }
    return take_gathered(mark);
}

// statement::execute: an error names the innermost statement executing.
recursive fn exec(n: i64, c: Ctx) -> i64 {
    let saved = cur_node;
    cur_node = n;
    let r = exec_impl(n, c);
    cur_node = saved;
    return r;
}

recursive fn exec_impl(n: i64, c: Ctx) -> i64 {
    let nd = nodes[n];
    let k = nd.kind;
    if k == N_STRING { return mk_str(node_text(n), false); }
    if k == N_INT { return mk_int(nd.i); }
    if k == N_FLOAT { return mk_float(nd.f); }
    if k == N_IDENT { return exec_identifier(n, c); }
    if k == N_COMMENT || k == N_NOOP || k == N_BLANK { return mk_undef(); }
    if k == N_ARRAY || k == N_TUPLE {
        let arr = if k == N_ARRAY { mk_array() } else { mk_tuple() };
        for i in nd.n0 {
            let item = exec(list_item(nd.l0, i), c);
            arr_push(arr, item);
        }
        return arr;
    }
    if k == N_OBJECT {
        let obj = mk_object();
        for i in nd.n0 {
            let key = exec(list_item(nd.l0, 2 * i), c);
            let v = exec(list_item(nd.l0, 2 * i + 1), c);
            obj_set(obj, key, v);
        }
        return obj;
    }
    if k == N_BINARY { return exec_binary(n, c); }
    if k == N_UNARY { return exec_unary(n, c); }
    if k == N_FILTER {
        let input = exec(nd.x, c);
        return apply_filter(c, nd.y, input);
    }
    if k == N_FILTER_STMT {
        let body = exec_list(nd.l0, nd.n0, c);
        return apply_filter(c, nd.x, body);
    }
    if k == N_TEST { return exec_test(n, c); }
    if k == N_IF {
        let test = exec(nd.x, c);
        if truthy(test) { return exec_list(nd.l0, nd.n0, c); }
        return exec_list(nd.l1, nd.n1, c);
    }
    if k == N_FOR { return exec_for(n, c); }
    if k == N_SET { return exec_set(n, c); }
    if k == N_MACRO {
        if nodes[nd.x].kind != N_IDENT { fail("Macro name must be an identifier"); }
        let name = mk_str(node_text(nd.x), false);
        let f = new_val(Val { kind: K_FUNC, a: FN_MACRO, c: n, name: name, .. });
        set_var(c, node_text(nd.x), f);
        return mk_undef();
    }
    if k == N_CALL_STMT { return exec_call_statement(n, c); }
    if k == N_MEMBER { return exec_member(n, c); }
    if k == N_CALL { return exec_call(n, c); }
    if k == N_KWARG {
        if nodes[nd.x].kind != N_IDENT { fail("Keyword argument key must be identifiers"); }
        let v = exec(nd.y, c);
        let key = mk_str(node_text(nd.x), false);
        return new_val(Val { kind: K_KWARG, a: key, b: v, .. });
    }
    if k == N_SELECT {
        // the test and the value are executed in place, not as statements
        let pred = exec_impl(nd.y, c);
        if !truthy(pred) { return mk_undef(); }
        return exec_impl(nd.x, c);
    }
    if k == N_TERNARY {
        let cond = exec(nd.x, c);
        return if truthy(cond) { exec(nd.y, c) } else { exec(nd.z, c) };
    }
    if k == N_BREAK { signal = SIG_BREAK; return mk_undef(); }
    if k == N_CONTINUE { signal = SIG_CONTINUE; return mk_undef(); }
    if k == N_SLICE { fail("must be handled by MemberExpression"); }
    if k == N_PROGRAM { fail("Cannot execute program directly, use jinja::runtime instead"); }
    fail(str("cannot exec ", node_type(k)));
    return -1;
}

recursive fn exec_identifier(n: i64, c: Ctx) -> i64 {
    let name = node_text(n);
    let v = obj_get_name(c.env, name);
    if v >= 0 && kind(v) != K_UNDEF {
        if stats { mark_used(v, false, 0); }
        return v;
    }
    let b = global_builtin(name);
    if b >= 0 { return mk_func(name, b, -1); }
    return mk_undef_hint(name);
}

// A builtin function value, bound to a first argument when this >= 0.
fn mk_func(name: const u8[:], id: i64, this: i64) -> i64 {
    let s = mk_str(name, false);
    return new_val(Val { kind: K_FUNC, a: id, b: this, name: s, .. });
}

fn is_null(v: i64) -> bool { kind(v) == K_NONE || kind(v) == K_UNDEF }

recursive fn exec_binary(n: i64, c: Ctx) -> i64 {
    let nd = nodes[n];
    let op = node_text(n);
    let left = exec(nd.x, c);
    if op == "and" { return if truthy(left) { exec(nd.y, c) } else { left }; }
    if op == "or" { return if truthy(left) { left } else { exec(nd.y, c) }; }
    let right = exec(nd.y, c);
    return binary(c, op, left, right);
}

// left op right, for an operator that evaluates both.
fn binary(c: Ctx, op: const u8[:], left: i64, right: i64) -> i64 {
    if op == "==" { return mk_bool(equiv(left, right)); }
    if op == "!=" { return mk_bool(!equiv(left, right)); }
    let lk = kind(left);
    let rk = kind(right);
    let is_in = op == "in" || op == "not in";
    let concat = op == "+" || op == "~";

    if lk == K_UNDEF || rk == K_UNDEF {
        // anything in undefined is false
        if rk == K_UNDEF && is_in { return mk_bool(op == "not in"); }
        if concat {
            let r = concat_null_with_str(left, right);
            if r >= 0 { return r; }
        }
        fail(str("Cannot perform operation ", op, " on undefined values"));
    } else if lk == K_NONE || rk == K_NONE {
        // none in {'low': 1}: a none on the left is looked up like any value
        if rk != K_NONE && is_in {
            let member = test_is_in(left, right);
            return mk_bool(if op == "in" { member } else { !member });
        }
        if concat {
            let r = concat_null_with_str(left, right);
            if r >= 0 { return r; }
        }
        fail("Cannot perform operation on null values");
    }

    if (lk == K_INT || lk == K_FLOAT) && (rk == K_INT || rk == K_FLOAT) {
        let a = vals[left].f;
        let b = vals[right].f;
        let is_float = lk == K_FLOAT || rk == K_FLOAT;
        if op == "+" || op == "-" || op == "*" {
            let r = if op == "+" { a + b } else if op == "-" { a - b } else { a * b };
            return if is_float { mk_float(r) } else { mk_int(cast_i64(r)) };
        }
        if op == "/" { return mk_float(a / b); }
        if op == "%" {
            let r = fmod(a, b);
            return if is_float { mk_float(r) } else { mk_int(cast_i64(r)) };
        }
        if op == "<" { return mk_bool(a < b); }
        if op == ">" { return mk_bool(a > b); }
        if op == ">=" { return mk_bool(a >= b); }
        if op == "<=" { return mk_bool(a <= b); }
    }

    let la = lk == K_ARRAY || lk == K_TUPLE;
    let ra = rk == K_ARRAY || rk == K_TUPLE;
    if la && ra {
        if op == "+" {
            let r = mk_array();
            for i in arr_len(left) { arr_push(r, arr_at(left, i)); }
            for i in arr_len(right) { arr_push(r, arr_at(right, i)); }
            return r;
        }
    } else if ra && is_in {
        let member = test_is_in(left, right);
        return mk_bool(if op == "in" { member } else { !member });
    }

    if (lk == K_STRING || rk == K_STRING) && concat {
        return mk_concat(as_string(left), as_string(right));
    }

    // Python's string repetition
    if op == "*" && ((lk == K_STRING && rk == K_INT) || (lk == K_INT && rk == K_STRING)) {
        let s = if lk == K_STRING { left } else { right };
        let times = if rk == K_INT { vals[right].i } else { vals[left].i };
        let first = parts.len;
        for i in max(times, 0) { push_parts(s); }
        return if times <= 0 { mk_empty() } else { mk_parts(first) };
    }

    if ((lk == K_STRING && rk == K_STRING) || rk == K_OBJECT) && is_in {
        let member = test_is_in(left, right);
        return mk_bool(if op == "in" { member } else { !member });
    }

    fail(str("Unknown operator \"", op, "\" between ", tname(left), " and ", tname(right)));
    return -1;
}

// none + "x" and "x" + undefined: the null is an empty string. -1 if
// neither side is a string.
fn concat_null_with_str(left: i64, right: i64) -> i64 {
    let ln = is_null(left);
    let rn = is_null(right);
    let ls = kind(left) == K_STRING;
    let rs = kind(right) == K_STRING;
    if !((ln && rs) || (rn && ls)) { return -1; }
    let a = if ln { mk_empty() } else { as_string(left) };
    let b = if rn { mk_empty() } else { as_string(right) };
    return mk_concat(a, b);
}

// static_cast<int64_t>(v) as x86 does it: INT64_MIN for NaN, infinities
// and what does not fit.
fn cast_i64(v: f64) -> i64 {
    if !(v < 9223372036854775808.0 && v >= -9223372036854775808.0) { return -9223372036854775807 - 1; }
    return v as! i64;
}

fn fmod(a: f64, b: f64) -> f64 { a % b }

recursive fn exec_unary(n: i64, c: Ctx) -> i64 {
    let v = exec(nodes[n].x, c);
    let op = node_text(n);
    let k = kind(v);
    if op == "not" { return mk_bool(!truthy(v)); }
    if op == "-" {
        if k == K_INT { return mk_int(-vals[v].i); }
        if k == K_FLOAT { return mk_float(-vals[v].f); }
        fail("Unary - operator requires numeric operand");
    }
    if op == "+" {
        if k == K_INT || k == K_FLOAT { return v; }
        fail("Unary + operator requires numeric operand");
    }
    fail(str("Unknown unary operator '", op, "'"));
    return -1;
}

// input | filter, filter a name or a call.
recursive fn apply_filter(c: Ctx, filter: i64, input: i64) -> i64 {
    let fk = nodes[filter].kind;
    if fk == N_IDENT {
        return apply_filter_by_name(c, node_text(filter), input, Args { ctx: c, at: argstack.len, n: 0 });
    }
    if fk == N_CALL {
        let callee = nodes[filter].x;
        if nodes[callee].kind != N_IDENT { fail("Filter callee must be an identifier"); }
        let mark = argstack.len;
        for i in nodes[filter].n0 {
            let v = exec(list_item(nodes[filter].l0, i), c);
            argstack.push(v);
        }
        let r = apply_filter_by_name(c, node_text(callee), input, Args { ctx: c, at: mark, n: argstack.len - mark });
        argstack.resize(mark);
        return r;
    }
    fail("Invalid filter expression");
    return -1;
}

recursive fn exec_test(n: i64, c: Ctx) -> i64 {
    let nd = nodes[n];
    let input = exec(nd.x, c);
    let mark = argstack.len;
    argstack.push(input);
    let t = nodes[nd.y];
    var id = -1;
    if t.kind == N_IDENT {
        id = nd.y;
    } else if t.kind == N_CALL {
        if nodes[t.x].kind != N_IDENT { fail("Test callee must be an identifier"); }
        id = t.x;
        for i in t.n0 {
            let v = exec(list_item(t.l0, i), c);
            argstack.push(v);
        }
    } else {
        fail("Invalid test expression");
    }
    let b = test_builtin(node_text(id));
    if b < 0 { fail(str("Unknown test '", node_text(id), "'")); }
    if stats {
        mark_used(input, false, 0);
        if node_text(id) == "string" { vals[input].ops |= OP_TEST_IS_STRING; }
    }
    let r = call_builtin(b, Args { ctx: c, at: mark, n: argstack.len - mark });
    argstack.resize(mark);
    if nd.flag { return mk_bool(!truthy(r)); }
    return r;
}

recursive fn exec_for(n: i64, c: Ctx) -> i64 {
    let nd = nodes[n];
    let sc = scope(c);
    let it = nodes[nd.y];
    let is_select = it.kind == N_SELECT;
    let iter_expr = if is_select { it.x } else { nd.y };
    var iterable = exec(iter_expr, sc);
    if stats {
        mark_used(iterable, false, 0);
        vals[iterable].ops |= OP_ARRAY_ACCESS;
    }
    if kind(iterable) == K_UNDEF { iterable = mk_array(); }
    let ik = kind(iterable);
    let is_obj = ik == K_OBJECT;
    if !is_obj && ik != K_ARRAY && ik != K_TUPLE {
        fail(str("Expected iterable or object type in for loop: got ", tname(iterable)));
    }
    // the items, as they are now: a body that changes the array does not
    // change the loop
    let snap = mk_array();
    if is_obj {
        for i in obj_len(iterable) {
            let t = mk_tuple();
            arr_push(t, obj_key(iterable, i));
            arr_push(t, obj_value(iterable, i));
            arr_push(snap, t);
        }
    } else {
        for i in arr_len(iterable) { arr_push(snap, arr_at(iterable, i)); }
    }

    let lv = nodes[nd.x];
    // the items that pass the select test, and what the loop names each
    // (its key, over an object)
    let chosen = mk_array();
    let chosen_values = mk_array();
    for i in arr_len(snap) {
        let item = arr_at(snap, i);
        var current = item;
        if lv.kind == N_IDENT {
            if is_obj { current = arr_at(item, 0); }
        } else if lv.kind == N_TUPLE {
            if !is_array(current) { fail(str("Cannot unpack non-iterable type: ", tname(current))); }
            if lv.n0 != arr_len(current) {
                fail(str("Too ", if lv.n0 > arr_len(current) { "few" } else { "many" }, " items to unpack"));
            }
        } else {
            fail(str("Invalid loop variable(s): ", node_type(lv.kind)));
        }
        if is_select && it.y >= 0 {
            let ls = scope(sc);
            bind_loopvar(nd.x, item, is_obj, ls);
            let test = exec(it.y, ls);
            if !truthy(test) { continue; }
        }
        arr_push(chosen, item);
        arr_push(chosen_values, current);
    }
    let count = arr_len(chosen);
    let gmark = gather.len;
    var no_iteration = true;
    for i in count {
        let ls = scope(sc);
        let lo = mk_object();
        vals[lo].builtins = false;
        obj_set_name(lo, "index", mk_int(i + 1));
        obj_set_name(lo, "index0", mk_int(i));
        obj_set_name(lo, "revindex", mk_int(count - i));
        obj_set_name(lo, "revindex0", mk_int(count - i - 1));
        obj_set_name(lo, "first", mk_bool(i == 0));
        obj_set_name(lo, "last", mk_bool(i == count - 1));
        obj_set_name(lo, "length", mk_int(count));
        obj_set_name(lo, "previtem", if i > 0 { arr_at(chosen_values, i - 1) } else { mk_undef_hint("previtem") });
        obj_set_name(lo, "nextitem", if i < count - 1 { arr_at(chosen_values, i + 1) } else { mk_undef_hint("nextitem") });
        set_var(ls, "loop", lo);
        bind_loopvar(nd.x, arr_at(chosen, i), is_obj, ls);
        var stopped = false;
        for s in nd.n0 {
            let v = exec(list_item(nd.l0, s), ls);
            if signal != SIG_NONE { break; }
            gather_value(v, 0);
        }
        if signal == SIG_CONTINUE { signal = SIG_NONE; continue; }
        if signal == SIG_BREAK { signal = SIG_NONE; stopped = true; }
        if stopped { break; }
        no_iteration = false;
    }
    if no_iteration {
        for s in nd.n1 {
            let v = exec(list_item(nd.l1, s), c);
            if signal != SIG_NONE { break; }
            gather_value(v, 0);
        }
    }
    return take_gathered(gmark);
}

// Sets the loop variable(s) to item in c: a name to the item (a key, over
// an object), names to the item's members.
recursive fn bind_loopvar(lv: i64, item: i64, is_obj: bool, c: Ctx) {
    let nd = nodes[lv];
    if nd.kind == N_IDENT {
        set_var(c, node_text(lv), if is_obj { arr_at(item, 0) } else { item });
        return;
    }
    for j in nd.n0 {
        let e = list_item(nd.l0, j);
        if nodes[e].kind != N_IDENT { fail(str("Cannot unpack non-identifier type: ", node_type(nodes[e].kind))); }
        set_var(c, node_text(e), arr_at(item, j));
    }
}

recursive fn exec_set(n: i64, c: Ctx) -> i64 {
    let nd = nodes[n];
    let rhs = if nd.y >= 0 { exec(nd.y, c) } else { exec_list(nd.l0, nd.n0, c) };
    let a = nodes[nd.x];
    if a.kind == N_IDENT {
        set_var(c, node_text(nd.x), rhs);
    } else if a.kind == N_TUPLE {
        if !is_array(rhs) { fail(str("Cannot unpack non-iterable type in set: ", tname(rhs))); }
        if arr_len(rhs) != a.n0 {
            fail(str("Too ", if a.n0 > arr_len(rhs) { "few" } else { "many" }, " items to unpack in set"));
        }
        for i in a.n0 {
            let e = list_item(a.l0, i);
            if nodes[e].kind != N_IDENT { fail(str("Cannot unpack to non-identifier in set: ", node_type(nodes[e].kind))); }
            set_var(c, node_text(e), arr_at(rhs, i));
        }
    } else if a.kind == N_MEMBER {
        // {% set ns.x = value %}
        if a.flag { fail("Cannot assign to computed member"); }
        if nodes[a.y].kind != N_IDENT { fail("Cannot assign to member with non-identifier property"); }
        let obj = exec(a.x, c);
        if kind(obj) != K_OBJECT { fail("Cannot assign to member of non-object"); }
        obj_set_name(obj, node_text(a.y), rhs);
    } else {
        fail(str("Invalid LHS inside assignment expression: ", node_type(a.kind)));
    }
    return mk_undef();
}

// Binds a macro's (or call block's) parameters, list params, to the
// arguments in c. A parameter with a default takes it when there are
// fewer arguments than parameters, evaluated in the caller's scope.
recursive fn bind_parameters(name: const u8[:], l: i64, nparams: i64, a: Args, c: Ctx) {
    for i in nparams {
        let p = list_item(l, i);
        let pk = nodes[p].kind;
        if i < a.n {
            if pk == N_IDENT {
                set_var(c, node_text(p), get_kwarg_or_pos(a, node_text(p), i));
            } else if pk == N_KWARG {
                let key = nodes[p].x;
                if nodes[key].kind != N_IDENT { fail(str("Keyword argument key must be an identifier in '", name, "'")); }
                set_var(c, node_text(key), get_kwarg_or_pos(a, node_text(key), i));
            } else {
                fail(str("Invalid parameter type in '", name, "'"));
            }
        } else {
            if pk != N_KWARG { fail(str("Not enough arguments provided to '", name, "'")); }
            let key = nodes[p].x;
            if nodes[key].kind != N_IDENT { fail(str("Keyword argument key must be an identifier in '", name, "'")); }
            let v = exec(nodes[p].y, a.ctx);
            set_var(c, node_text(key), v);
        }
    }
}

recursive fn exec_call_statement(n: i64, c: Ctx) -> i64 {
    let nd = nodes[n];
    let call = nodes[nd.x];
    let callee = exec(call.x, c);
    if kind(callee) != K_FUNC { fail(str("Callee is not a function: got ", tname(callee))); }
    // the body, as the function `caller` in the macro
    let captured = scope(c);
    let caller_name = mk_str("caller", false);
    let f = new_val(Val { kind: K_FUNC, a: FN_CALLER, c: n, d: captured.env, name: caller_name, .. });
    let call_ctx = scope(c);
    set_var(call_ctx, "caller", f);
    let mark = argstack.len;
    for i in call.n0 {
        let v = exec(list_item(call.l0, i), c);
        argstack.push(v);
    }
    let r = invoke(callee, Args { ctx: call_ctx, at: mark, n: argstack.len - mark });
    argstack.resize(mark);
    return r;
}

recursive fn exec_call(n: i64, c: Ctx) -> i64 {
    let nd = nodes[n];
    let mark = argstack.len;
    for i in nd.n0 {
        let v = exec(list_item(nd.l0, i), c);
        argstack.push(v);
    }
    let callee = exec(nd.x, c);
    if kind(callee) != K_FUNC { fail(str("Callee is not a function: got ", tname(callee))); }
    let r = invoke(callee, Args { ctx: c, at: mark, n: argstack.len - mark });
    argstack.resize(mark);
    return r;
}

// Calls function f with arguments a, its bound first argument before them.
recursive fn invoke(f: i64, a: Args) -> i64 {
    let fv = vals[f];
    let mark = argstack.len;
    if fv.b >= 0 { argstack.push(fv.b); }
    for k in a.n { argstack.push(argstack[a.at + k]); }
    let args = Args { ctx: a.ctx, at: mark, n: argstack.len - mark, name: fv.name };
    var r = -1;
    if fv.a == FN_MACRO || fv.a == FN_CALLER {
        if a.ctx.depth >= LIMIT { fail(str("macro calls are nested more than ", LIMIT, " levels deep")); }
        let nd = nodes[fv.c];
        var mc = if fv.a == FN_MACRO { scope(a.ctx) } else { scope(Ctx { env: fv.d, depth: 0 }) };
        mc.depth = a.ctx.depth + 1;
        if fv.a == FN_MACRO {
            bind_parameters(node_text(nd.x), nd.l0, nd.n0, args, mc);
            r = exec_list(nd.l1, nd.n1, mc);
        } else {
            bind_parameters("caller", nd.l0, nd.n0, args, mc);
            r = exec_list(nd.l1, nd.n1, mc);
        }
    } else {
        r = call_builtin(fv.a, args);
    }
    argstack.resize(mark);
    return r;
}

recursive fn exec_member(n: i64, c: Ctx) -> i64 {
    let nd = nodes[n];
    let object = exec(nd.x, c);
    let pk = nodes[nd.y].kind;
    var property = -1;
    if nd.flag {
        if pk == N_SLICE {
            let ok = kind(object);
            let size = if ok == K_ARRAY || ok == K_TUPLE { arr_len(object) } else if ok == K_STRING { vals[object].i } else { 0 };
            let sl = nodes[nd.y];
            let step = if sl.z >= 0 { exec(sl.z, c) } else { mk_int(1) };
            let start = if sl.x >= 0 { exec(sl.x, c) } else if as_int(step) < 0 { mk_int(size - 1) } else { mk_int(0) };
            let stop = if sl.y >= 0 { exec(sl.y, c) } else if as_int(step) < 0 { mk_int(-1) } else { mk_int(size) };
            // object.slice(start, stop, step)
            let f = try_builtin_func(c, "slice", object, false);
            let mark = argstack.len;
            argstack.push(start);
            argstack.push(stop);
            argstack.push(step);
            let r = invoke(f, Args { ctx: c, at: mark, n: 3 });
            argstack.resize(mark);
            return r;
        }
        property = exec(nd.y, c);
    } else if pk == N_INT {
        property = mk_int(nodes[nd.y].i);
        if nodes[nd.y].i < 0 { fail("Static member property cannot be negative"); }
    } else {
        if pk != N_IDENT { fail("Static member property must be an identifier"); }
        property = mk_str(node_text(nd.y), false);
        // obj.items is the method even when obj has a key "items", as in
        // jinja2; obj['items'] is the value
        let m = try_builtin_func(c, node_text(nd.y), object, true);
        if kind(m) != K_UNDEF { return m; }
    }
    return member(c, object, property);
}

// object[property], after the slices and methods.
fn member(c: Ctx, object: i64, property: i64) -> i64 {
    var v = mk_undef_hint("object_property");
    if kind(property) == K_UNDEF { return v; }
    if !hashable(property) { fail(str("Type: ", tname(property), " is not allowed as object key")); }
    let ok = kind(object);
    let pk = kind(property);
    if ok == K_UNDEF { return v; }
    if ok == K_OBJECT {
        let key = as_string(property);
        let got = obj_get(object, property);
        if got >= 0 { v = got; }
        if kind(v) == K_UNDEF { v = try_builtin_func(c, sbytes(key), object, true); }
    } else if ok == K_ARRAY || ok == K_TUPLE || ok == K_STRING {
        if pk == K_INT || pk == K_BOOL {
            var index = vals[property].i;
            if ok == K_STRING {
                let s = sbytes(object);
                if index >= 0 && index < s.len { v = mk_str(s[index..index + 1], false); }
            } else {
                let len = arr_len(object);
                if index < 0 { index += len; }
                if index >= 0 && index < len { v = arr_at(object, index); }
            }
        } else if pk == K_STRING {
            v = try_builtin_func(c, sbytes(property), object, true);
        }
    } else {
        if pk != K_STRING { fail(str("Cannot access property with non-string: got ", tname(property))); }
        v = try_builtin_func(c, sbytes(property), object, true);
    }
    if stats {
        mark_used(v, false, 0);
        mark_used(object, false, 0);
        mark_used(property, false, 0);
        // an int index into anything but an object is an array access
        if ok != K_OBJECT && (pk == K_INT || pk == K_BOOL) { vals[object].ops |= OP_ARRAY_ACCESS; }
    }
    return v;
}

// render with its own handler: the output's text into out; false, and the
// message in err, if the template fails.
fn try_render(p: Program, c: Ctx, out: u8[>..]&) -> bool {
    var ok = false;
    fn body() {
        let r = render(p, c);
        out.append(sbytes(r));
        ok = true;
    }
    exception::try<Fail>(body);
    return ok;
}
