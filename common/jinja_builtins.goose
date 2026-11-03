// llama.cpp's Jinja engine: the builtin functions, filters and tests, and
// the methods of each type (a filter x | f(y) is the method f of x, called
// with x first).
// port-of: common/jinja/value.cpp (builtins)
import std;
import .jinja;
import .number;
import .jinja_lexer;
import .jinja_parser;
import .jinja_runtime;

namespace jinja;

// global functions
let B_RAISE_EXCEPTION = 0;
let B_DICT = 1;
let B_NAMESPACE = 2;
let B_STRFTIME_NOW = 3;
let B_RANGE = 4;
let B_TOJSON = 5;
// tests: test_is_<name>
let B_TEST_BOOLEAN = 10;
let B_TEST_CALLABLE = 11;
let B_TEST_ODD = 12;
let B_TEST_EVEN = 13;
let B_TEST_FALSE = 14;
let B_TEST_TRUE = 15;
let B_TEST_DIVISIBLEBY = 16;
let B_TEST_STRING = 17;
let B_TEST_INTEGER = 18;
let B_TEST_FLOAT = 19;
let B_TEST_NUMBER = 20;
let B_TEST_ITERABLE = 21;
let B_TEST_MAPPING = 22;
let B_TEST_LOWER = 23;
let B_TEST_UPPER = 24;
let B_TEST_NONE = 25;
let B_TEST_DEFINED = 26;
let B_TEST_UNDEFINED = 27;
let B_TEST_EQ = 28;
let B_TEST_GE = 29;
let B_TEST_GT = 30;
let B_TEST_LT = 31;
let B_TEST_NE = 32;
let B_TEST_IN = 33;
let B_TEST_TEST = 34;
let B_TEST_SAMEAS = 35;
let B_TEST_ESCAPED = 36;
let B_TEST_FILTER = 37;
// methods shared by several types
let B_DEFAULT = 40;
let B_EMPTY_ARRAY = 41;      // what an undefined or none gives for a list
let B_EMPTY_STRING = 42;
let B_EMPTY_STRING_NOARGS = 43;
let B_EMPTY_INT = 44;
let B_EMPTY_UNDEF = 45;
let B_UNDEF_JOIN = 46;
let B_UNDEF_STRIP = 47;
let B_NONE_STRING = 48;
// int, float, bool
let B_INT_ABS = 50;
let B_INT_INT = 51;
let B_INT_FLOAT = 52;
let B_FLOAT_ABS = 53;
let B_FLOAT_INT = 54;
let B_FLOAT_FLOAT = 55;
let B_FLOAT_STRING = 56;
let B_BOOL_INT = 57;
let B_BOOL_FLOAT = 58;
let B_BOOL_STRING = 59;
// string
let B_STR_UPPER = 60;
let B_STR_LOWER = 61;
let B_STR_STRIP = 62;
let B_STR_RSTRIP = 63;
let B_STR_LSTRIP = 64;
let B_STR_TITLE = 65;
let B_STR_CAPITALIZE = 66;
let B_STR_LENGTH = 67;
let B_STR_STARTSWITH = 68;
let B_STR_ENDSWITH = 69;
let B_STR_SPLIT = 70;
let B_STR_RSPLIT = 71;
let B_STR_REPLACE = 72;
let B_STR_FORMAT = 73;
let B_STR_INT = 74;
let B_STR_FLOAT = 75;
let B_STR_STRING = 76;
let B_STR_SLICE = 77;
let B_STR_SAFE = 78;
let B_STR_INDENT = 79;
let B_STR_JOIN = 80;
// array
let B_ARR_LIST = 90;
let B_ARR_FIRST = 91;
let B_ARR_LAST = 92;
let B_ARR_LENGTH = 93;
let B_ARR_SLICE = 94;
let B_ARR_SELECTATTR = 95;
let B_ARR_SELECT = 96;
let B_ARR_REJECTATTR = 97;
let B_ARR_REJECT = 98;
let B_ARR_SUM = 99;
let B_ARR_JOIN = 100;
let B_ARR_STRING = 101;
let B_ARR_MAP = 102;
let B_ARR_APPEND = 103;
let B_ARR_POP = 104;
let B_ARR_SORT = 105;
let B_ARR_REVERSE = 106;
let B_ARR_MIN = 107;
let B_ARR_MAX = 108;
let B_ARR_UNIQUE = 109;
// object
let B_OBJ_GET = 110;
let B_OBJ_KEYS = 111;
let B_OBJ_VALUES = 112;
let B_OBJ_ITEMS = 113;
let B_OBJ_STRING = 114;
let B_OBJ_LENGTH = 115;
let B_OBJ_DICTSORT = 116;
let B_OBJ_JOIN = 117;

struct Builtin { name: const u8[:], id: i64 }

let GLOBALS: Builtin[38] = [
    Builtin { "raise_exception", B_RAISE_EXCEPTION }, Builtin { "dict", B_DICT },
    Builtin { "namespace", B_NAMESPACE }, Builtin { "strftime_now", B_STRFTIME_NOW },
    Builtin { "range", B_RANGE }, Builtin { "tojson", B_TOJSON },
    Builtin { "test_is_boolean", B_TEST_BOOLEAN }, Builtin { "test_is_callable", B_TEST_CALLABLE },
    Builtin { "test_is_odd", B_TEST_ODD }, Builtin { "test_is_even", B_TEST_EVEN },
    Builtin { "test_is_false", B_TEST_FALSE }, Builtin { "test_is_true", B_TEST_TRUE },
    Builtin { "test_is_divisibleby", B_TEST_DIVISIBLEBY }, Builtin { "test_is_string", B_TEST_STRING },
    Builtin { "test_is_integer", B_TEST_INTEGER }, Builtin { "test_is_float", B_TEST_FLOAT },
    Builtin { "test_is_number", B_TEST_NUMBER }, Builtin { "test_is_iterable", B_TEST_ITERABLE },
    Builtin { "test_is_sequence", B_TEST_ITERABLE }, Builtin { "test_is_mapping", B_TEST_MAPPING },
    Builtin { "test_is_lower", B_TEST_LOWER }, Builtin { "test_is_upper", B_TEST_UPPER },
    Builtin { "test_is_none", B_TEST_NONE }, Builtin { "test_is_defined", B_TEST_DEFINED },
    Builtin { "test_is_undefined", B_TEST_UNDEFINED }, Builtin { "test_is_eq", B_TEST_EQ },
    Builtin { "test_is_equalto", B_TEST_EQ }, Builtin { "test_is_ge", B_TEST_GE },
    Builtin { "test_is_gt", B_TEST_GT }, Builtin { "test_is_greaterthan", B_TEST_GT },
    Builtin { "test_is_lt", B_TEST_LT }, Builtin { "test_is_lessthan", B_TEST_LT },
    Builtin { "test_is_ne", B_TEST_NE }, Builtin { "test_is_in", B_TEST_IN },
    Builtin { "test_is_test", B_TEST_TEST }, Builtin { "test_is_sameas", B_TEST_SAMEAS },
    Builtin { "test_is_escaped", B_TEST_ESCAPED }, Builtin { "test_is_filter", B_TEST_FILTER },
];

let INT_METHODS: Builtin[7] = [
    Builtin { "default", B_DEFAULT }, Builtin { "abs", B_INT_ABS }, Builtin { "int", B_INT_INT },
    Builtin { "float", B_INT_FLOAT }, Builtin { "safe", B_TOJSON }, Builtin { "string", B_TOJSON },
    Builtin { "tojson", B_TOJSON },
];

// Python's str() of a float: its repr
let FLOAT_METHODS: Builtin[7] = [
    Builtin { "default", B_DEFAULT }, Builtin { "abs", B_FLOAT_ABS }, Builtin { "int", B_FLOAT_INT },
    Builtin { "float", B_FLOAT_FLOAT }, Builtin { "safe", B_FLOAT_STRING }, Builtin { "string", B_FLOAT_STRING },
    Builtin { "tojson", B_TOJSON },
];

let STRING_METHODS: Builtin[23] = [
    Builtin { "default", B_DEFAULT }, Builtin { "upper", B_STR_UPPER }, Builtin { "lower", B_STR_LOWER },
    Builtin { "strip", B_STR_STRIP }, Builtin { "rstrip", B_STR_RSTRIP }, Builtin { "lstrip", B_STR_LSTRIP },
    Builtin { "title", B_STR_TITLE }, Builtin { "capitalize", B_STR_CAPITALIZE },
    Builtin { "length", B_STR_LENGTH }, Builtin { "startswith", B_STR_STARTSWITH },
    Builtin { "endswith", B_STR_ENDSWITH }, Builtin { "split", B_STR_SPLIT }, Builtin { "rsplit", B_STR_RSPLIT },
    Builtin { "replace", B_STR_REPLACE }, Builtin { "format", B_STR_FORMAT }, Builtin { "int", B_STR_INT },
    Builtin { "float", B_STR_FLOAT }, Builtin { "string", B_STR_STRING }, Builtin { "slice", B_STR_SLICE },
    Builtin { "safe", B_STR_SAFE }, Builtin { "tojson", B_TOJSON }, Builtin { "indent", B_STR_INDENT },
    Builtin { "join", B_STR_JOIN },
];

let BOOL_METHODS: Builtin[6] = [
    Builtin { "default", B_DEFAULT }, Builtin { "int", B_BOOL_INT }, Builtin { "float", B_BOOL_FLOAT },
    Builtin { "safe", B_BOOL_STRING }, Builtin { "string", B_BOOL_STRING }, Builtin { "tojson", B_TOJSON },
];

let ARRAY_METHODS: Builtin[22] = [
    Builtin { "default", B_DEFAULT }, Builtin { "list", B_ARR_LIST }, Builtin { "first", B_ARR_FIRST },
    Builtin { "last", B_ARR_LAST }, Builtin { "length", B_ARR_LENGTH }, Builtin { "slice", B_ARR_SLICE },
    Builtin { "selectattr", B_ARR_SELECTATTR }, Builtin { "select", B_ARR_SELECT },
    Builtin { "rejectattr", B_ARR_REJECTATTR }, Builtin { "reject", B_ARR_REJECT }, Builtin { "sum", B_ARR_SUM },
    Builtin { "join", B_ARR_JOIN }, Builtin { "string", B_ARR_STRING }, Builtin { "tojson", B_TOJSON },
    Builtin { "map", B_ARR_MAP }, Builtin { "append", B_ARR_APPEND }, Builtin { "pop", B_ARR_POP },
    Builtin { "sort", B_ARR_SORT }, Builtin { "reverse", B_ARR_REVERSE }, Builtin { "min", B_ARR_MIN },
    Builtin { "max", B_ARR_MAX }, Builtin { "unique", B_ARR_UNIQUE },
];

// no default: it got in gpt-oss's way
let OBJECT_METHODS: Builtin[10] = [
    Builtin { "get", B_OBJ_GET }, Builtin { "keys", B_OBJ_KEYS }, Builtin { "list", B_OBJ_KEYS },
    Builtin { "values", B_OBJ_VALUES }, Builtin { "items", B_OBJ_ITEMS }, Builtin { "tojson", B_TOJSON },
    Builtin { "string", B_OBJ_STRING }, Builtin { "length", B_OBJ_LENGTH }, Builtin { "dictsort", B_OBJ_DICTSORT },
    Builtin { "join", B_OBJ_JOIN },
];

let NONE_METHODS: Builtin[11] = [
    Builtin { "default", B_DEFAULT }, Builtin { "tojson", B_TOJSON }, Builtin { "string", B_NONE_STRING },
    Builtin { "safe", B_NONE_STRING }, Builtin { "items", B_EMPTY_ARRAY }, Builtin { "map", B_EMPTY_ARRAY },
    Builtin { "reject", B_EMPTY_ARRAY }, Builtin { "rejectattr", B_EMPTY_ARRAY },
    Builtin { "select", B_EMPTY_ARRAY }, Builtin { "selectattr", B_EMPTY_ARRAY }, Builtin { "unique", B_EMPTY_ARRAY },
];

let UNDEFINED_METHODS: Builtin[28] = [
    Builtin { "default", B_DEFAULT }, Builtin { "capitalize", B_EMPTY_STRING_NOARGS },
    Builtin { "first", B_EMPTY_UNDEF }, Builtin { "items", B_EMPTY_ARRAY }, Builtin { "join", B_UNDEF_JOIN },
    Builtin { "last", B_EMPTY_UNDEF }, Builtin { "length", B_EMPTY_INT }, Builtin { "list", B_EMPTY_ARRAY },
    Builtin { "lower", B_EMPTY_STRING_NOARGS }, Builtin { "map", B_EMPTY_ARRAY }, Builtin { "max", B_EMPTY_UNDEF },
    Builtin { "min", B_EMPTY_UNDEF }, Builtin { "reject", B_EMPTY_ARRAY }, Builtin { "rejectattr", B_EMPTY_ARRAY },
    Builtin { "replace", B_EMPTY_STRING }, Builtin { "reverse", B_EMPTY_ARRAY }, Builtin { "safe", B_EMPTY_STRING },
    Builtin { "select", B_EMPTY_ARRAY }, Builtin { "selectattr", B_EMPTY_ARRAY }, Builtin { "sort", B_EMPTY_ARRAY },
    Builtin { "string", B_EMPTY_STRING }, Builtin { "strip", B_UNDEF_STRIP }, Builtin { "sum", B_EMPTY_INT },
    Builtin { "title", B_EMPTY_STRING_NOARGS }, Builtin { "truncate", B_EMPTY_STRING },
    Builtin { "unique", B_EMPTY_ARRAY }, Builtin { "upper", B_EMPTY_STRING_NOARGS },
    Builtin { "wordcount", B_EMPTY_INT },
];

fn lookup(table: const Builtin[:], name: const u8[:]) -> i64 {
    for b in table { if b.name == name { return b.id; } }
    return -1;
}

fn global_builtin(name: const u8[:]) -> i64 { lookup(GLOBALS[..], name) }

// test_is_<name>
fn test_builtin(name: const u8[:]) -> i64 {
    var full: u8[>..] = [];
    full.append("test_is_");
    full.append(name);
    return lookup(GLOBALS[..], full);
}

// The method name of v, or -1. A function or keyword argument has none.
fn type_builtin(v: i64, name: const u8[:]) -> i64 {
    let k = kind(v);
    if k == K_INT { return lookup(INT_METHODS[..], name); }
    if k == K_FLOAT { return lookup(FLOAT_METHODS[..], name); }
    if k == K_STRING { return lookup(STRING_METHODS[..], name); }
    if k == K_BOOL { return lookup(BOOL_METHODS[..], name); }
    if k == K_ARRAY || k == K_TUPLE { return lookup(ARRAY_METHODS[..], name); }
    if k == K_OBJECT { return if vals[v].builtins { lookup(OBJECT_METHODS[..], name) } else { -1 }; }
    if k == K_NONE { return lookup(NONE_METHODS[..], name); }
    if k == K_UNDEF { return lookup(UNDEFINED_METHODS[..], name); }
    fail(str(tname(v), " has no builtins"));
    return -1;
}

// input's method name bound to input; undefined when it has none and
// undef_on_missing, an error otherwise.
fn try_builtin_func(c: Ctx, name: const u8[:], input: i64, undef_on_missing: bool) -> i64 {
    if stats {
        mark_used(input, false, 0);
        if name == "selectattr" { vals[input].ops |= OP_SELECTATTR; }
    }
    let id = type_builtin(input, name);
    if id >= 0 { return mk_func(name, id, input); }
    if undef_on_missing { return mk_undef_hint(name); }
    fail(str("Unknown (built-in) filter '", name, "' for type ", tname(input)));
    return -1;
}

// input | name(args).
recursive fn apply_filter_by_name(c: Ctx, filter: const u8[:], input: i64, a: Args) -> i64 {
    let name = if filter == "count" { "length" } else if filter == "d" { "default" }
               else if filter == "e" { "escape" } else if filter == "trim" { "strip" } else { filter };
    var inp = input;
    // the string filters take anything, as its text
    if kind(inp) != K_UNDEF && kind(inp) != K_STRING &&
       (name == "capitalize" || name == "lower" || name == "replace" || name == "strip" ||
        name == "title" || name == "upper" || name == "wordcount") {
        inp = copy_string(as_string(inp));
    }
    let f = try_builtin_func(c, name, inp, false);
    return invoke(f, a);
}

// A new string value with the parts of s.
fn copy_string(s: i64) -> i64 {
    let first = parts.len;
    push_parts(s);
    return mk_parts(first);
}

//
// arguments
//

fn fname(a: Args) -> const u8[:] { if a.name >= 0 { sbytes(a.name) } else { "" } }

fn get_pos(a: Args, k: i64) -> i64 {
    if a.n > k { return arg(a, k); }
    fail(str("Function '", fname(a), "' expected at least ", k + 1, " arguments, got ", a.n));
    return -1;
}

fn get_pos_or(a: Args, k: i64, d: i64) -> i64 { if a.n > k { arg(a, k) } else { d } }

fn get_kwarg(a: Args, key: const u8[:], d: i64) -> i64 {
    for k in a.n {
        let v = arg(a, k);
        if kind(v) == K_KWARG && sbytes(vals[v].a) == key { return vals[v].b; }
    }
    return d;
}

// The keyword argument key, or else the positional one at pos (if it is not
// a keyword argument), or undefined.
fn get_kwarg_or_pos(a: Args, key: const u8[:], pos: i64) -> i64 {
    let v = get_kwarg(a, key, -1);
    if v >= 0 && kind(v) != K_UNDEF { return v; }
    if pos < a.n && kind(arg(a, pos)) != K_KWARG { return arg(a, pos); }
    return if v >= 0 { v } else { mk_undef() };
}

// Fails on a keyword argument other than the names given.
fn ensure_kwargs(a: Args, n1: const u8[:], n2: const u8[:]) {
    for k in a.n {
        let v = arg(a, k);
        if kind(v) != K_KWARG { continue; }
        let key = sbytes(vals[v].a);
        if key != n1 && key != n2 {
            fail(str("Function '", fname(a), "' got an unexpected keyword argument '", key, "'"));
        }
    }
}

fn ensure_count(a: Args, lo: i64, hi: i64) {
    if a.n < lo || a.n > hi {
        fail(str("Function '", fname(a), "' expected between ", lo, " and ", hi, " arguments, got ", a.n));
    }
}

// The C++ type name of a value class, as typeid gives it.
fn class_name(w: u8) -> const u8[:] {
    return match w {
        'S' => "St10shared_ptrIN5jinja14value_string_tEE",
        'I' => "St10shared_ptrIN5jinja11value_int_tEE",
        'F' => "St10shared_ptrIN5jinja13value_float_tEE",
        'B' => "St10shared_ptrIN5jinja12value_bool_tEE",
        'A' => "St10shared_ptrIN5jinja13value_array_tEE",
        _ => "St10shared_ptrIN5jinja14value_object_tEE",
    };
}

// Whether v is of class w: S string, I int, F float, B bool, A array (or
// tuple), O object.
fn is_class(v: i64, w: u8) -> bool {
    let k = kind(v);
    return match w {
        'S' => k == K_STRING, 'I' => k == K_INT, 'F' => k == K_FLOAT, 'B' => k == K_BOOL,
        'A' => k == K_ARRAY || k == K_TUPLE, _ => k == K_OBJECT,
    };
}

fn ensure_val(a: Args, v: i64, w: u8) {
    if !is_class(v, w) {
        fail(str("Function '", fname(a), "' expected value of type ", class_name(w), ", got ", tname(v)));
    }
}

// The first `required` arguments there must be, of the classes in w.
fn ensure_vals(a: Args, w: const u8[:], required: i64) {
    ensure_count(a, required, 999);
    for k in required { if a.n > k { ensure_val(a, arg(a, k), w[k]); } }
}

//
// calling a builtin
//

recursive fn call_builtin(id: i64, a: Args) -> i64 {
    if id >= B_TEST_BOOLEAN && id <= B_TEST_FILTER { return call_test(id, a); }
    if id == B_ARR_SELECTATTR || id == B_ARR_SELECT || id == B_ARR_REJECTATTR || id == B_ARR_REJECT {
        return select_or_reject(a, id == B_ARR_REJECTATTR || id == B_ARR_REJECT, id == B_ARR_SELECTATTR || id == B_ARR_REJECTATTR);
    }
    if id == B_ARR_MAP { return array_map(a); }
    if id == B_ARR_SORT { return array_sort(a); }
    if id == B_ARR_UNIQUE || id == B_ARR_MIN || id == B_ARR_MAX { return array_pick(id, a); }
    if id == B_OBJ_DICTSORT { return object_dictsort(a); }
    if id >= B_STR_UPPER && id <= B_STR_JOIN { return call_string(id, a); }
    return call_other(id, a);
}

fn call_other(id: i64, a: Args) -> i64 {
    if id == B_RAISE_EXCEPTION {
        ensure_vals(a, "S", 1);
        fail(str("Jinja Exception: ", sbytes(arg(a, 0))));
    }
    if id == B_DICT { return to_object(a); }
    if id == B_NAMESPACE {
        // unlike a dict, a namespace has no methods: ns.items is the attribute
        let o = to_object(a);
        vals[o].builtins = false;
        return o;
    }
    if id == B_STRFTIME_NOW {
        ensure_vals(a, "S", 1);
        var buf: u8[100] = [0; 100];
        let n = number::c_strftime(buf[..], sbytes(arg(a, 0)), now);
        if n == 0 { fail("strftime_now: failed to format time"); }
        return mk_str(buf[..n], false);
    }
    if id == B_RANGE { return range(a); }
    if id == B_TOJSON { return tojson(a); }
    if id == B_DEFAULT {
        ensure_count(a, 2, 3);
        let check_bool = truthy(get_kwarg_or_pos(a, "boolean", 2));
        let v = get_pos(a, 0);
        let no_value = if check_bool { !truthy(v) } else { is_null(v) };
        return if no_value { get_pos(a, 1) } else { v };
    }
    if id == B_EMPTY_ARRAY { return mk_array(); }
    if id == B_EMPTY_STRING { return mk_empty(); }
    if id == B_EMPTY_STRING_NOARGS {
        ensure_kwargs(a, "", "");
        ensure_count(a, 1, 1);
        return mk_empty();
    }
    if id == B_EMPTY_INT { return mk_int(0); }
    if id == B_EMPTY_UNDEF { return mk_undef(); }
    if id == B_UNDEF_JOIN {
        ensure_kwargs(a, "d", "attribute");
        ensure_count(a, 1, 3);
        return mk_empty();
    }
    if id == B_UNDEF_STRIP {
        ensure_kwargs(a, "chars", "");
        ensure_count(a, 1, 2);
        return mk_empty();
    }
    if id == B_NONE_STRING { return mk_str("None", false); }
    if id == B_INT_ABS { ensure_vals(a, "I", 1); return mk_int(abs(vals[arg(a, 0)].i)); }
    if id == B_INT_INT { ensure_vals(a, "I", 1); return mk_int(vals[arg(a, 0)].i); }
    if id == B_INT_FLOAT { ensure_vals(a, "I", 1); return mk_float(vals[arg(a, 0)].i); }
    if id == B_FLOAT_ABS {
        ensure_vals(a, "F", 1);
        let v = vals[arg(a, 0)].f;
        return mk_float(if v < 0.0 { -v } else { v });
    }
    if id == B_FLOAT_INT { ensure_vals(a, "F", 1); return mk_int(cast_i64(vals[arg(a, 0)].f)); }
    if id == B_FLOAT_FLOAT { ensure_vals(a, "F", 1); return mk_float(vals[arg(a, 0)].f); }
    if id == B_FLOAT_STRING {
        ensure_vals(a, "F", 1);
        var out: u8[>..] = [];
        number::float_repr(out, vals[arg(a, 0)].f);
        return mk_str(out, false);
    }
    if id == B_BOOL_INT { ensure_vals(a, "B", 1); return mk_int(vals[arg(a, 0)].i); }
    if id == B_BOOL_FLOAT { ensure_vals(a, "B", 1); return mk_float(vals[arg(a, 0)].i); }
    if id == B_BOOL_STRING { ensure_vals(a, "B", 1); return mk_str(if vals[arg(a, 0)].i != 0 { "True" } else { "False" }, false); }
    if id == B_ARR_LIST {
        ensure_vals(a, "A", 1);
        let v = arg(a, 0);
        return mk_array_of(K_ARRAY, vals[v].a, vals[v].n);
    }
    if id == B_ARR_FIRST || id == B_ARR_LAST {
        ensure_vals(a, "A", 1);
        let v = arg(a, 0);
        if arr_len(v) == 0 { return mk_undef(); }
        return arr_at(v, if id == B_ARR_FIRST { 0 } else { arr_len(v) - 1 });
    }
    if id == B_ARR_LENGTH { ensure_vals(a, "A", 1); return mk_int(arr_len(arg(a, 0))); }
    if id == B_ARR_SLICE { return array_slice(a); }
    if id == B_ARR_SUM { not_implemented("Array sum builtin not implemented"); }
    if id == B_ARR_JOIN { return array_join(a); }
    if id == B_ARR_STRING || id == B_OBJ_STRING {
        ensure_vals(a, if id == B_ARR_STRING { "A" } else { "O" }, 1);
        if stats { mark_used(arg(a, 0), true, 0); }
        return copy_string(as_string(arg(a, 0)));
    }
    if id == B_ARR_APPEND {
        ensure_count(a, 2, 999);
        let v = get_pos(a, 0);
        if !is_array(v) { fail("append: first argument must be an array"); }
        if kind(v) == K_TUPLE { fail("Attempting to modify immutable type"); }
        arr_push(v, get_pos(a, 1));
        return v;
    }
    if id == B_ARR_POP {
        ensure_count(a, 1, 2);
        ensure_vals(a, "AI", 1);
        let v = arg(a, 0);
        var index = if a.n == 2 { as_int(arg(a, 1)) } else { -1 };
        if kind(v) == K_TUPLE { fail("Attempting to modify immutable type"); }
        let len = arr_len(v);
        if index < 0 { index += len; }
        if index < 0 || index >= len { fail(str("Index ", index, " out of bounds for array of size ", len)); }
        let r = arr_at(v, index);
        let x = vals[v];
        for k in index..len - 1 { items[x.a + k] = items[x.a + k + 1]; }
        vals[v].n = len - 1;
        return r;
    }
    if id == B_ARR_REVERSE {
        ensure_vals(a, "A", 1);
        let v = arg(a, 0);
        let r = new_val(Val { kind: kind(v), .. });
        for k in arr_len(v) { arr_push(r, arr_at(v, arr_len(v) - 1 - k)); }
        return r;
    }
    if id == B_OBJ_GET {
        ensure_count(a, 2, 3);
        let o = arg(a, 0);
        if kind(o) != K_OBJECT { fail("get: first argument must be an object"); }
        if kind(arg(a, 1)) != K_STRING { fail("get: second argument must be a string (key)"); }
        let d = if a.n == 3 { arg(a, 2) } else { mk_none() };
        let v = obj_get(o, arg(a, 1));
        return if v >= 0 { v } else { d };
    }
    if id == B_OBJ_KEYS || id == B_OBJ_VALUES || id == B_OBJ_ITEMS {
        ensure_vals(a, "O", 1);
        let o = arg(a, 0);
        let r = mk_array();
        for k in obj_len(o) {
            if id == B_OBJ_KEYS { arr_push(r, obj_key(o, k)); }
            else if id == B_OBJ_VALUES { arr_push(r, obj_value(o, k)); }
            else {
                let t = mk_tuple();
                arr_push(t, obj_key(o, k));
                arr_push(t, obj_value(o, k));
                arr_push(r, t);
            }
        }
        return r;
    }
    if id == B_OBJ_LENGTH { ensure_vals(a, "O", 1); return mk_int(obj_len(arg(a, 0))); }
    if id == B_OBJ_JOIN { not_implemented("object join not implemented"); }
    fail("unknown builtin");
    return -1;
}

fn call_test(id: i64, a: Args) -> i64 {
    if id == B_TEST_ODD || id == B_TEST_EVEN {
        ensure_vals(a, "I", 1);
        let odd = vals[arg(a, 0)].i % 2 != 0;
        return mk_bool(if id == B_TEST_ODD { odd } else { !odd });
    }
    if id == B_TEST_DIVISIBLEBY {
        ensure_vals(a, "II", 2);
        return mk_bool(vals[arg(a, 0)].i % vals[arg(a, 1)].i == 0);
    }
    if id == B_TEST_LOWER || id == B_TEST_UPPER {
        ensure_vals(a, "S", 1);
        let v = vals[arg(a, 0)];
        for k in v.n {
            for c in part_bytes(parts[v.a + k]) {
                if id == B_TEST_LOWER && c >= 'A' && c <= 'Z' { return mk_bool(false); }
                if id == B_TEST_UPPER && c >= 'a' && c <= 'z' { return mk_bool(false); }
            }
        }
        return mk_bool(true);
    }
    if id >= B_TEST_EQ && id <= B_TEST_NE {
        ensure_count(a, 2, 2);
        let op = match id { B_TEST_EQ => CMP_EQ, B_TEST_GE => CMP_GE, B_TEST_GT => CMP_GT, B_TEST_LT => CMP_LT, _ => CMP_NE };
        return mk_bool(compare(arg(a, 0), arg(a, 1), op));
    }
    if id == B_TEST_IN {
        ensure_count(a, 2, 999);
        return mk_bool(test_is_in(arg(a, 0), arg(a, 1)));
    }
    if id == B_TEST_TEST {
        ensure_vals(a, "S", 1);
        return mk_bool(test_builtin(sbytes(arg(a, 0))) >= 0);
    }
    if id == B_TEST_SAMEAS {
        // whether a and b are the same object (Python caches small ints)
        ensure_count(a, 2, 999);
        let x = arg(a, 0);
        let y = arg(a, 1);
        let kx = kind(x);
        let ky = kind(y);
        var res = false;
        if kx != K_UNDEF && ky != K_UNDEF {
            if kx == K_NONE && ky == K_NONE { res = true; }
            else if kx == K_BOOL && ky == K_BOOL { res = vals[x].i == vals[y].i; }
            else if kx == K_INT && ky == K_INT { res = vals[x].i >= -5 && vals[x].i <= 256 && vals[x].i == vals[y].i; }
            else { res = x == y; }
        }
        return mk_bool(res);
    }
    if id == B_TEST_ESCAPED { not_implemented("escaped test not implemented"); }
    if id == B_TEST_FILTER { not_implemented("filter test not implemented"); }
    ensure_count(a, 1, 999);
    let v = get_pos(a, 0);
    let k = kind(v);
    let r = match id {
        B_TEST_BOOLEAN => k == K_BOOL,
        B_TEST_CALLABLE => k == K_FUNC,
        B_TEST_FALSE => k == K_BOOL && vals[v].i == 0,
        B_TEST_TRUE => k == K_BOOL && vals[v].i != 0,
        B_TEST_STRING => k == K_STRING,
        B_TEST_INTEGER => k == K_INT,
        B_TEST_FLOAT => k == K_FLOAT,
        B_TEST_NUMBER => k == K_INT || k == K_FLOAT,
        B_TEST_ITERABLE => k == K_OBJECT || k == K_ARRAY || k == K_TUPLE || k == K_STRING || k == K_UNDEF,
        B_TEST_MAPPING => k == K_OBJECT,
        B_TEST_NONE => k == K_NONE,
        B_TEST_DEFINED => k != K_UNDEF,
        _ => k == K_UNDEF,
    };
    return mk_bool(r);
}

// needle in haystack (test_is_in).
fn test_is_in(needle: i64, haystack: i64) -> bool {
    let k = kind(haystack);
    if k == K_UNDEF { return false; }
    if k == K_ARRAY || k == K_TUPLE {
        for i in arr_len(haystack) { if equiv(needle, arr_at(haystack, i)) { return true; } }
        return false;
    }
    if k == K_STRING {
        if kind(needle) != K_STRING {
            fail(str("'in' test expects args[1] as string when args[0] is string, got args[1] as ", tname(needle)));
        }
        return find(sbytes(haystack), sbytes(needle)) >= 0;
    }
    if k == K_OBJECT { return obj_find(haystack, needle) >= 0; }
    fail(str("'in' test expects iterable as first argument, got ", tname(haystack)));
    return false;
}

let CMP_EQ = 0;
let CMP_GE = 1;
let CMP_GT = 2;
let CMP_LT = 3;
let CMP_NE = 4;

// value_compare: numbers as floats, strings (and a string with a number) by
// their text, bools for equality only; anything else is not equal.
fn compare(a: i64, b: i64, op: i64) -> bool {
    let ka = kind(a);
    let kb = kind(b);
    let na = ka == K_INT || ka == K_FLOAT;
    let nb = kb == K_INT || kb == K_FLOAT;
    if na && nb {
        let x = vals[a].f;
        let y = vals[b].f;
        return match op { CMP_EQ => x == y, CMP_GE => x >= y, CMP_GT => x > y, CMP_LT => x < y, _ => x != y };
    }
    if (kb == K_STRING && na) || (ka == K_STRING && nb) || (ka == K_STRING && kb == K_STRING) {
        let c = cmp_bytes(sbytes(as_string(a)), sbytes(as_string(b)));
        return match op { CMP_EQ => c == 0, CMP_GE => c >= 0, CMP_GT => c > 0, CMP_LT => c < 0, _ => c != 0 };
    }
    if ka == K_BOOL && kb == K_BOOL {
        if op == CMP_EQ { return vals[a].i == vals[b].i; }
        if op == CMP_NE { return vals[a].i != vals[b].i; }
        fail("Unsupported comparison operator for bool type");
    }
    return false;
}

// std::string's compare: bytes as unsigned, then length.
fn cmp_bytes(x: const u8[:], y: const u8[:]) -> i64 {
    for i in min(x.len, y.len) {
        if x[i] != y[i] { return if x[i] < y[i] { -1 } else { 1 }; }
    }
    return if x.len < y.len { -1 } else if x.len > y.len { 1 } else { 0 };
}

// val[attr] for an array or object (an attribute of digits indexes an
// array), or d.
fn get_attribute(val: i64, attr: i64, d: i64) -> i64 {
    if kind(attr) == K_UNDEF { return d; }
    let k = kind(val);
    if k == K_ARRAY || k == K_TUPLE {
        var idx = attr;
        if kind(attr) == K_STRING {
            let s = sbytes(attr);
            var digits = s.len > 0;
            for c in s { if c < '0' || c > '9' { digits = false; } }
            if digits {
                let v, ok = parse_int(s);
                idx = if ok { mk_int(v) } else { mk_undef() };
            }
        }
        let ik = kind(idx);
        if ik != K_INT && ik != K_BOOL { return d; }
        var i = vals[idx].i;
        let len = arr_len(val);
        if i < 0 { i += len; }
        if i < 0 || i >= len { return d; }
        return arr_at(val, i);
    }
    if k == K_OBJECT {
        let v = obj_get(val, attr);
        return if v >= 0 { v } else { d };
    }
    return d;
}

// dict() and namespace(): from pairs, an object, keyword arguments.
fn to_object(a: Args) -> i64 {
    let out = mk_object();
    let iter = get_pos_or(a, 0, -1);
    var iter_first = false;
    if iter >= 0 && is_array(iter) {
        iter_first = true;
        for i in arr_len(iter) {
            let it = arr_at(iter, i);
            if is_array(it) && arr_len(it) == 2 {
                obj_set(out, arr_at(it, 0), arr_at(it, 1));
            } else {
                fail(str("namespace/dict() iterable argument must consist of tuples, not ", tname(it)));
            }
        }
    } else if iter >= 0 && kind(iter) == K_OBJECT {
        iter_first = true;
        for i in obj_len(iter) { obj_set(out, obj_key(iter, i), obj_value(iter, i)); }
    }
    for k in a.n {
        let v = arg(a, k);
        if kind(v) == K_KWARG {
            obj_set(out, vals[v].a, vals[v].b);
        } else if !iter_first {
            fail(str("namespace/dict() arguments must be kwargs, dict and/or iterable of tuples, not ", tname(v)));
        }
        iter_first = false;
    }
    return out;
}

fn range(a: Args) -> i64 {
    ensure_count(a, 1, 3);
    ensure_vals(a, "III", 1);
    var start = 0;
    var stop = 0;
    var step = 1;
    if a.n == 1 {
        stop = as_int(arg(a, 0));
    } else {
        start = as_int(arg(a, 0));
        stop = as_int(arg(a, 1));
        if a.n == 3 { step = as_int(arg(a, 2)); }
    }
    if step == 0 { fail("range() step argument must not be zero"); }
    let out = mk_array();
    var i = start;
    if step > 0 { while i < stop { arr_push(out, mk_int(i)); i += step; } }
    else { while i > stop { arr_push(out, mk_int(i)); i += step; } }
    return out;
}

fn tojson(a: Args) -> i64 {
    ensure_count(a, 1, 5);
    let ascii = get_kwarg_or_pos(a, "ensure_ascii", 1);
    let indent_v = get_kwarg_or_pos(a, "indent", 2);
    let seps = get_kwarg_or_pos(a, "separators", 3);
    let sort = get_kwarg_or_pos(a, "sort_keys", 4);
    if stats { mark_used(get_pos(a, 0), true, 0); }
    let indent = if kind(indent_v) == K_INT { vals[indent_v].i } else { -1 };
    if truthy(sort) { not_implemented("tojson sort_keys=true not implemented"); }
    let ensure_ascii = truthy(ascii);
    var item_sep: u8[>..] = [];
    var key_sep: u8[>..] = [];
    let ns = if is_array(seps) { arr_len(seps) } else { 0 };
    if ns > 0 { item_sep.append(sbytes(as_string(arr_at(seps, 0)))); } else { item_sep.append(if indent < 0 { ", " } else { "," }); }
    if ns > 1 { key_sep.append(sbytes(as_string(arr_at(seps, 1)))); } else { key_sep.append(": "); }
    var out: u8[>..] = [];
    to_json(out, get_pos(a, 0), 0, indent, item_sep, key_sep);
    if ensure_ascii {
        var esc: u8[>..] = [];
        ensure_ascii_json(esc, out);
        return mk_str(esc, false);
    }
    return mk_str(out, false);
}

// The non-ASCII characters in the strings of JSON s as \u escapes, the
// rest as it is (json_ensure_ascii_preserving_format).
fn ensure_ascii_json(out: u8[>..]&, s: const u8[:]) {
    var in_string = false;
    var escaped = false;
    var pos = 0;
    while pos < s.len {
        let c = s[pos];
        if !in_string {
            out.push(c);
            if c == '"' { in_string = true; }
            pos++;
            continue;
        }
        if escaped { out.push(c); escaped = false; pos++; continue; }
        if c == '\\' { out.push(c); escaped = true; pos++; continue; }
        if c == '"' { out.push(c); in_string = false; pos++; continue; }
        if c < 0x80 { out.push(c); pos++; continue; }
        let cp, n = utf8_decode(s, pos);
        if n == 0 { out.append("\\ufffd"); pos++; continue; }
        if cp <= 0xFFFF {
            u16_escape(out, cp);
        } else {
            let v = cp - 0x10000;
            u16_escape(out, 0xD800 + ((v >> 10) & 0x3FF));
            u16_escape(out, 0xDC00 + (v & 0x3FF));
        }
        pos += n;
    }
}

fn u16_escape(out: u8[>..]&, v: i64) {
    out.append("\\u");
    for i in 4 { out.push("0123456789abcdef"[(v >> (12 - 4 * i)) & 15]); }
}

// The code point of the UTF-8 sequence at s[at], and its length; 0 if it
// is not a whole valid sequence (common_parse_utf8_codepoint).
fn utf8_decode(s: const u8[:], at: i64) -> i64, i64 {
    let c = s[at];
    if (c & 0x80) == 0 { return c, 1; }
    if (c & 0x40) == 0 { return 0, 0; }
    var n = 0;
    var cp = 0;
    if (c & 0x20) == 0 { n = 2; cp = c & 0x1F; }
    else if (c & 0x10) == 0 { n = 3; cp = c & 0x0F; }
    else if (c & 0x08) == 0 { n = 4; cp = c & 0x07; }
    else { return 0, 0; }
    for i in 1..n {
        if at + i >= s.len || (s[at + i] & 0xC0) != 0x80 { return 0, 0; }
        cp = (cp << 6) | (s[at + i] & 0x3F);
    }
    return cp, n;
}

// select, reject, selectattr, rejectattr.
recursive fn select_or_reject(a: Args, reject: bool, attr: bool) -> i64 {
    let test_pos = if attr { 2 } else { 1 };
    ensure_count(a, test_pos, 999);
    ensure_val(a, get_pos(a, 0), 'A');
    var name = -1;
    if attr {
        name = get_pos(a, 1);
        ensure_val(a, name, 'S');
    }
    var test = -1;
    if a.n > test_pos {
        ensure_val(a, arg(a, test_pos), 'S');
        test = test_builtin(sbytes(arg(a, test_pos)));
        if test < 0 { fail(str(fname(a), ": unknown test '", sbytes(arg(a, test_pos)), "'")); }
    }
    let arr = arg(a, 0);
    let out = mk_array();
    let undef = mk_undef();
    for i in arr_len(arr) {
        let item = arr_at(arr, i);
        let v = if attr { get_attribute(item, name, undef) } else { item };
        var selected = false;
        if test < 0 {
            selected = truthy(v);
        } else {
            let mark = argstack.len;
            argstack.push(v);
            for k in test_pos + 1..a.n { argstack.push(arg(a, k)); }
            let r = call_builtin(test, Args { ctx: a.ctx, at: mark, n: argstack.len - mark });
            argstack.resize(mark);
            selected = truthy(r);
        }
        if reject { selected = !selected; }
        if selected { arr_push(out, item); }
    }
    return out;
}

// map('filter', args...) or map(attribute='name', default=d).
recursive fn array_map(a: Args) -> i64 {
    ensure_count(a, 2, 999);
    let v = arg(a, 0);
    if !is_array(v) { fail("map: first argument must be an array"); }
    let out = new_val(Val { kind: kind(v), .. });
    if kind(arg(a, 1)) != K_KWARG {
        let filter = arg(a, 1);
        if kind(filter) != K_STRING { fail("map: filter name must be a string"); }
        let fargs = Args { ctx: a.ctx, at: a.at + 2, n: a.n - 2 };
        for i in arr_len(v) {
            let r = apply_filter_by_name(a.ctx, sbytes(filter), arr_at(v, i), fargs);
            arr_push(out, r);
        }
        return out;
    }
    let attribute = get_kwarg_or_pos(a, "attribute", 1);
    let d = get_kwarg(a, "default", mk_undef());
    for i in arr_len(v) { arr_push(out, get_attribute(arr_at(v, i), attribute, d)); }
    return out;
}

// sort(reverse=false, case_sensitive=false, attribute=none), as libstdc++'s
// std::sort does it for up to 16 items: an insertion sort.
recursive fn array_sort(a: Args) -> i64 {
    ensure_count(a, 1, 4);
    let v = get_pos(a, 0);
    if !is_array(v) { fail("sort: first argument must be an array"); }
    let reverse = truthy(get_kwarg_or_pos(a, "reverse", 1));
    get_kwarg_or_pos(a, "case_sensitive", 2);
    let attribute = get_kwarg_or_pos(a, "attribute", 3);
    let op = if reverse { CMP_GT } else { CMP_LT };
    let out = new_val(Val { kind: kind(v), .. });
    for i in arr_len(v) { arr_push(out, arr_at(v, i)); }
    let undef = mk_undef();
    let base = vals[out].a;
    let n = vals[out].n;
    for i in 1..n {
        let x = items[base + i];
        let kx = if kind(attribute) == K_UNDEF { x } else { get_attribute(x, attribute, undef) };
        let k0 = if kind(attribute) == K_UNDEF { items[base] } else { get_attribute(items[base], attribute, undef) };
        if compare(kx, k0, op) {
            var j = i;
            while j > 0 { items[base + j] = items[base + j - 1]; j--; }
            items[base] = x;
        } else {
            var j = i;
            while j > 0 {
                let p = items[base + j - 1];
                let kp = if kind(attribute) == K_UNDEF { p } else { get_attribute(p, attribute, undef) };
                if !compare(kx, kp, op) { break; }
                items[base + j] = p;
                j--;
            }
            items[base + j] = x;
        }
    }
    return out;
}

// unique, min and max.
recursive fn array_pick(id: i64, a: Args) -> i64 {
    ensure_count(a, 1, if id == B_ARR_UNIQUE { 3 } else { 4 });
    ensure_vals(a, "A", 1);
    let v = arg(a, 0);
    get_kwarg_or_pos(a, "case_sensitive", 1);
    let attribute = get_kwarg_or_pos(a, "attribute", 2);
    let undef = mk_undef();
    let by_attr = kind(attribute) != K_UNDEF;
    if id == B_ARR_UNIQUE {
        let out = mk_array();
        let seen = mk_array();
        for i in arr_len(v) {
            let item = arr_at(v, i);
            let key = if by_attr { get_attribute(item, attribute, undef) } else { item };
            var dup = false;
            for k in arr_len(seen) { if compare(arr_at(seen, k), key, CMP_EQ) { dup = true; break; } }
            if !dup { arr_push(seen, key); arr_push(out, item); }
        }
        return out;
    }
    if arr_len(v) == 0 { return undef; }
    var result = arr_at(v, 0);
    let op = if id == B_ARR_MIN { CMP_LT } else { CMP_GT };
    for i in arr_len(v) {
        let item = arr_at(v, i);
        let x = if by_attr { get_attribute(item, attribute, undef) } else { item };
        let y = if by_attr { get_attribute(result, attribute, undef) } else { result };
        if compare(x, y, op) { result = item; }
    }
    return result;
}

// dictsort(case_sensitive, by='key'|'value', reverse): a sorted copy.
recursive fn object_dictsort(a: Args) -> i64 {
    let input = get_pos(a, 0);
    get_kwarg_or_pos(a, "case_sensitive", 1);
    let by = get_kwarg_or_pos(a, "by", 2);
    let reverse = truthy(get_kwarg_or_pos(a, "reverse", 3));
    let by_value = kind(by) == K_STRING && sbytes(by) == "value";
    let out = mk_object();
    if kind(input) == K_OBJECT {
        for k in obj_len(input) { obj_append(out, obj_key(input, k), obj_value(input, k)); }
    }
    let op = if reverse { CMP_GT } else { CMP_LT };
    let base = vals[out].a;
    let n = vals[out].n;
    let w = if by_value { 1 } else { 0 };
    for i in 1..n {
        let xk = items[base + 2 * i];
        let xv = items[base + 2 * i + 1];
        let kx = items[base + 2 * i + w];
        if compare(kx, items[base + w], op) {
            var j = i;
            while j > 0 {
                items[base + 2 * j] = items[base + 2 * j - 2];
                items[base + 2 * j + 1] = items[base + 2 * j - 1];
                j--;
            }
            items[base] = xk;
            items[base + 1] = xv;
        } else {
            var j = i;
            while j > 0 && compare(kx, items[base + 2 * (j - 1) + w], op) {
                items[base + 2 * j] = items[base + 2 * j - 2];
                items[base + 2 * j + 1] = items[base + 2 * j - 1];
                j--;
            }
            items[base + 2 * j] = xk;
            items[base + 2 * j + 1] = xv;
        }
    }
    return out;
}

// Python's slice of a sequence of len items: the indices, in order.
fn slice_bounds(len: i64, start: i64, stop: i64, step: i64) -> i64, i64 {
    var a = start;
    var b = stop;
    if step > 0 {
        a = if a < 0 { max(len + a, 0) } else { min(a, len) };
        b = if b < 0 { max(len + b, 0) } else { min(b, len) };
    } else {
        a = if a < 0 { max(len + a, 0) } else { min(a, len - 1) };
        b = if b < -1 { max(len + b, -1) } else { min(b, len - 1) };
    }
    return a, b;
}

// The start, stop and step of slice(start, stop, step) after the value.
fn slice_args(a: Args, w: const u8[:]) -> i64, i64, i64 {
    ensure_count(a, 1, 4);
    ensure_vals(a, w, 2);
    let a0 = get_pos(a, 1);
    let a1 = get_pos_or(a, 2, mk_undef());
    let a2 = get_pos_or(a, 3, mk_undef());
    var start = 0;
    var stop = 0;
    var step = 1;
    if a.n == 1 {
        stop = as_int(a0);
    } else if a.n == 2 {
        start = as_int(a0);
        stop = as_int(a1);
    } else {
        start = as_int(a0);
        stop = as_int(a1);
        step = as_int(a2);
    }
    if step == 0 { fail("slice step cannot be zero"); }
    return start, stop, step;
}

fn array_slice(a: Args) -> i64 {
    let start, stop, step = slice_args(a, "AII");
    let v = arg(a, 0);
    let len = arr_len(v);
    let lo, hi = slice_bounds(len, start, stop, step);
    let out = new_val(Val { kind: kind(v), .. });
    let dir = if step > 0 { 1 } else { -1 };
    var i = lo;
    while dir * i < dir * hi {
        if i >= 0 && i < len { arr_push(out, arr_at(v, i)); }
        i += step;
    }
    return out;
}

fn array_join(a: Args) -> i64 {
    ensure_kwargs(a, "d", "attribute");
    ensure_count(a, 1, 3);
    let v = get_pos(a, 0);
    if !is_array(v) { fail("join() first argument must be an array"); }
    let delim_v = get_kwarg_or_pos(a, "d", 1);
    let attribute = get_kwarg_or_pos(a, "attribute", 2);
    let undef = mk_undef();
    var delim: u8[>..] = [];
    if kind(delim_v) != K_UNDEF { delim.append(sbytes(as_string(delim_v))); }
    var out: u8[>..] = [];
    let n = arr_len(v);
    for i in n {
        var item = arr_at(v, i);
        if kind(attribute) != K_UNDEF { item = get_attribute(item, attribute, undef); }
        let k = kind(item);
        if k != K_STRING && k != K_INT && k != K_FLOAT { fail("join() can only join arrays of strings or numerics"); }
        out.append(sbytes(as_string(item)));
        if i < n - 1 { out.append(delim); }
    }
    return mk_str(out, false);
}

//
// strings
//

fn call_string(id: i64, a: Args) -> i64 {
    if id == B_STR_UPPER || id == B_STR_LOWER || id == B_STR_TITLE || id == B_STR_CAPITALIZE {
        ensure_kwargs(a, "", "");
        ensure_count(a, 1, 1);
        ensure_vals(a, "S", 1);
        return transform(arg(a, 0), id);
    }
    if id == B_STR_STRIP {
        ensure_kwargs(a, "chars", "");
        ensure_count(a, 1, 2);
        if kind(get_pos(a, 0)) != K_STRING { fail("strip() first argument must be a string"); }
        return strip(arg(a, 0), true, true, get_kwarg_or_pos(a, "chars", 1));
    }
    if id == B_STR_RSTRIP || id == B_STR_LSTRIP {
        ensure_vals(a, "S", 1);
        return strip(arg(a, 0), id == B_STR_LSTRIP, id == B_STR_RSTRIP, get_kwarg_or_pos(a, "chars", 1));
    }
    if id == B_STR_LENGTH { ensure_vals(a, "S", 1); return mk_int(vals[arg(a, 0)].i); }
    if id == B_STR_STARTSWITH || id == B_STR_ENDSWITH {
        ensure_vals(a, "SS", 2);
        let s = sbytes(arg(a, 0));
        let p = sbytes(arg(a, 1));
        return mk_bool(if id == B_STR_STARTSWITH { starts_with(s, p) } else { ends_with(s, p) });
    }
    if id == B_STR_SPLIT || id == B_STR_RSPLIT { return split(a, id == B_STR_RSPLIT); }
    if id == B_STR_REPLACE { return replace(a); }
    if id == B_STR_FORMAT { return format_string(a); }
    if id == B_STR_INT {
        let input = get_pos(a, 0);
        let d = get_kwarg_or_pos(a, "default", 1);
        let base_v = get_kwarg_or_pos(a, "base", 2);
        let base = if kind(base_v) == K_UNDEF { 10 } else { as_int(base_v) };
        if base != 0 && (base < 2 || base > 36) { fail("int() base must be 0 or between 2 and 36"); }
        if kind(input) != K_STRING { fail("int() first argument must be a string"); }
        var ok = 0;
        let v = number::c_stoi(sbytes(input), base, ok);
        if ok != 0 { return mk_int(v); }
        return mk_int(if kind(d) == K_UNDEF { 0 } else { as_int(d) });
    }
    if id == B_STR_FLOAT {
        ensure_vals(a, "S", 1);
        let d = get_kwarg_or_pos(a, "default", 1);
        var used = 0;
        let v = number::c_strtod(sbytes(arg(a, 0)), used);
        if used > 0 { return mk_float(v); }
        return mk_float(if kind(d) == K_UNDEF { 0.0 } else { as_float(d) });
    }
    if id == B_STR_STRING { ensure_vals(a, "S", 1); return copy_string(arg(a, 0)); }
    if id == B_STR_SLICE {
        let start, stop, step = slice_args(a, "SII");
        let input = arg(a, 0);
        let s = sbytes(input);
        let lo, hi = slice_bounds(s.len, start, stop, step);
        var out: u8[>..] = [];
        let dir = if step > 0 { 1 } else { -1 };
        var i = lo;
        while dir * i < dir * hi {
            if i >= 0 && i < s.len { out.push(s[i]); }
            i += step;
        }
        let r = mk_str(out, false);
        mark_input_based_on(r, input);
        return r;
    }
    if id == B_STR_SAFE { ensure_vals(a, "S", 1); return arg(a, 0); }
    if id == B_STR_INDENT { return indent(a); }
    not_implemented("String join builtin not implemented");
    return -1;
}

// upper, lower, title and capitalize, part by part.
fn transform(s: i64, id: i64) -> i64 {
    let v = vals[s];
    let first = parts.len;
    for k in v.n {
        let p = parts[v.a + k];
        let off = text.len;
        var up_next = true;
        for i in p.len {
            var c = text[p.off + i];
            if id == B_STR_UPPER { if c >= 'a' && c <= 'z' { c -= 32; } }
            else if id == B_STR_LOWER { if c >= 'A' && c <= 'Z' { c += 32; } }
            else if id == B_STR_CAPITALIZE {
                if i == 0 { if c >= 'a' && c <= 'z' { c -= 32; } }
                else if c >= 'A' && c <= 'Z' { c += 32; }
            } else {
                if c == ' ' || (c >= 9 && c <= 13) { up_next = true; }
                else if up_next { if c >= 'a' && c <= 'z' { c -= 32; } up_next = false; }
                else if c >= 'A' && c <= 'Z' { c += 32; }
            }
            text.push(c);
        }
        parts.push(Part { off: off, len: p.len, input: p.input });
    }
    return mk_parts(first);
}

// Whether c is one of chars (a string), or white space when chars is undefined.
fn strip_match(c: u8, chars: i64) -> bool {
    if kind(chars) == K_UNDEF { return c == ' ' || (c >= 9 && c <= 13); }
    for x in sbytes(chars) { if x == c { return true; } }
    return false;
}

// string::strip: the parts at the ends lose what matches, and go when
// that empties them.
fn strip(s: i64, left: bool, right: bool, chars_v: i64) -> i64 {
    let chars = if kind(chars_v) == K_UNDEF { chars_v } else { as_string(chars_v) };
    let first = parts.len;
    push_parts(s);
    var lo = first;
    var hi = parts.len;
    if left {
        while lo < hi {
            while parts[lo].len > 0 && strip_match(text[parts[lo].off], chars) {
                parts[lo].off += 1;
                parts[lo].len -= 1;
            }
            if parts[lo].len > 0 { break; }
            lo++;
        }
    }
    if right {
        while hi > lo {
            let p = hi - 1;
            while parts[p].len > 0 && strip_match(text[parts[p].off + parts[p].len - 1], chars) { parts[p].len -= 1; }
            if parts[p].len > 0 { break; }
            hi--;
        }
    }
    // the parts kept, moved down to first
    let n = hi - lo;
    for k in n { parts[first + k] = parts[lo + k]; }
    parts.resize(first + n);
    return mk_parts(first);
}

fn split(a: Args, from_right: bool) -> i64 {
    ensure_count(a, 1, 3);
    let input = get_pos(a, 0);
    if kind(input) != K_STRING { fail(if from_right { "rsplit() first argument must be a string" } else { "split() first argument must be a string" }); }
    var delim: u8[>..] = [];
    if a.n > 1 { delim.append(sbytes(as_string(arg(a, 1)))); } else { delim.push(' '); }
    if delim.len == 0 { fail("empty separator"); }
    var maxsplit = if a.n > 2 { as_int(arg(a, 2)) } else { -1 };
    let s = sbytes(input);
    let out = mk_array();
    var lo = 0;
    var hi = s.len;
    if !from_right {
        loop {
            if maxsplit == 0 { break; }
            let p = find(s[lo..hi], delim[..]);
            if p < 0 { break; }
            arr_push(out, mk_str(s[lo..lo + p], false));
            lo += p + delim.len;
            maxsplit--;
        }
    } else {
        loop {
            if maxsplit == 0 { break; }
            let p = rfind(s[lo..hi], delim[..]);
            if p < 0 { break; }
            arr_push(out, mk_str(s[lo + p + delim.len..hi], false));
            hi = lo + p;
            maxsplit--;
        }
    }
    let rest = mk_str(s[lo..hi], false);
    mark_input_based_on(rest, input);
    arr_push(out, rest);
    if from_right {
        let n = arr_len(out);
        let base = vals[out].a;
        for k in n / 2 {
            let t = items[base + k];
            items[base + k] = items[base + n - 1 - k];
            items[base + n - 1 - k] = t;
        }
    }
    return out;
}

fn replace(a: Args) -> i64 {
    ensure_vals(a, "SSSI", 3);
    let s = sbytes(arg(a, 0));
    let old = sbytes(arg(a, 1));
    let new = sbytes(arg(a, 2));
    let count = if a.n > 3 { as_int(arg(a, 3)) } else { -1 };
    if count > 0 { not_implemented("String replace with count argument not implemented"); }
    var out: u8[>..] = [];
    if old == new {
        out.append(s);
    } else if old.len == 0 {
        out.append(new);
        for c in s { out.push(c); out.append(new); }
    } else {
        var pos = 0;
        loop {
            let p = find(s[pos..], old);
            if p < 0 { break; }
            out.append(s[pos..pos + p]);
            out.append(new);
            pos += p + old.len;
        }
        out.append(s[pos..]);
    }
    let r = mk_str(out, false);
    mark_input_based_on(r, arg(a, 0));
    return r;
}

// 'a {} b'.format(x): only {} placeholders, filled in turn.
fn format_string(a: Args) -> i64 {
    let fmt = get_pos(a, 0);
    if kind(fmt) != K_STRING { fail("format() first argument must be a string"); }
    let fmt_input = all_input(fmt);
    let s = sbytes(fmt);
    // the parts are gathered aside, as the arguments' text makes parts too
    let mark = gather.len;
    var lit = 0;
    var next_arg = 1;
    var i = 0;
    while i < s.len {
        if s[i] != '{' { i++; continue; }
        if i + 1 >= s.len || s[i + 1] != '}' { not_implemented("format() only supports simple '{}' placeholders"); }
        if i > lit { gather_bytes(s[lit..i], fmt_input); }
        let sv = as_string(get_pos(a, next_arg));
        let v = vals[sv];
        next_arg++;
        for k in v.n { gather.push(parts[v.a + k]); }
        i += 2;
        lit = i;
    }
    if s.len > lit { gather_bytes(s[lit..], fmt_input); }
    return take_gathered(mark);
}

// A part of new text, onto gather.
fn gather_bytes(b: const u8[:], input: bool) {
    let off = text.len;
    for c in b { text.push(c); }
    gather.push(Part { off: off, len: b.len, input: input });
}

// indent(width=4, first=false, blank=false): every line but the first (and
// empty ones) indented.
fn indent(a: Args) -> i64 {
    ensure_count(a, 1, 4);
    let input = get_pos(a, 0);
    let width = get_kwarg_or_pos(a, "width", 1);
    let first = truthy(get_kwarg_or_pos(a, "first", 2));
    let blank = truthy(get_kwarg_or_pos(a, "blank", 3));
    if kind(input) != K_STRING { fail("indent() first argument must be a string"); }
    var ind: u8[>..] = [];
    if kind(width) == K_INT { for i in vals[width].i { ind.push(' '); } }
    else if kind(width) == K_STRING { ind.append(sbytes(width)); }
    else { ind.append("    "); }
    let s = sbytes(input);
    var out: u8[>..] = [];
    var lo = 0;
    // the lines std::getline reads: a last newline ends a line, it does not start one
    while lo < s.len {
        var hi = lo;
        while hi < s.len && s[hi] != '\n' { hi++; }
        let line = s[lo..hi];
        if out.len > 0 { out.push('\n'); }
        if (if out.len == 0 { first } else { line.len > 0 || blank }) { out.append(ind); }
        out.append(line);
        lo = hi + 1;
    }
    if s.len > 0 && s[s.len - 1] == '\n' {
        out.push('\n');
        if blank { out.append(ind); }
    }
    let r = mk_str(out, false);
    mark_input_based_on(r, input);
    return r;
}
