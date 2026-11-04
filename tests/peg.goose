// llama.cpp's tests/peg-parser: test-gbnf-generation (the GBNF of each
// parser, as text) and cases of test-basic (what parsers make of inputs).
import std;
import common from "../common";
import common.json_schema;
import common.peg;

var checked = 0;
var failed = 0;

fn check(name: const u8[:], ok: bool) {
    checked++;
    if !ok { failed++; print("FAIL ", name); }
}

fn fresh() {
    peg::reset();
    json_schema::reset();
    peg::begin();
}

let SPACE = "space ::= | \" \" | \"\\n\"{1,2} [ \\t]{0,20}\n";

fn check_gbnf(name: const u8[:], root: i64, lazy: bool, expected: const u8[:]) {
    let a = peg::build(root);
    var got: u8[>..] = [];
    let ok = peg::grammar(a, lazy, got);
    check(name, ok && got == expected);
    if ok && got != expected { print("  expected:\n", expected, "  got:\n", got); }
    if !ok { print("  error: ", peg::err); }
}

fn gbnf_tests() {
    fresh();
    check_gbnf("literal grammar generation", peg::literal("hello"), false, str("root ::= \"hello\"\n", SPACE));
    fresh();
    check_gbnf("char class grammar", peg::chars("[a-z]", 1, 1), false, str("root ::= [a-z]\n", SPACE));
    fresh();
    {
        let a = peg::literal("hello");
        let b = peg::literal(" ");
        let s = peg::seq2(a, b);
        let c = peg::literal("world");
        check_gbnf("sequence grammar", peg::seq2(s, c), false, str("root ::= \"hello\" \" \" \"world\"\n", SPACE));
    }
    fresh();
    {
        let a = peg::literal("cat");
        let b = peg::literal("dog");
        check_gbnf("choice grammar", peg::choice2(a, b), false, str("root ::= \"cat\" | \"dog\"\n", SPACE));
    }
    fresh();
    check_gbnf("one_or_more grammar", peg::one_or_more(peg::literal("a")), false, str("root ::= \"a\"+\n", SPACE));
    fresh();
    check_gbnf("zero_or_more grammar", peg::zero_or_more(peg::literal("a")), false, str("root ::= \"a\"*\n", SPACE));
    fresh();
    {
        let a = peg::literal("hello");
        let b = peg::optional(peg::literal(" world"));
        check_gbnf("optional grammar", peg::seq2(a, b), false, str("root ::= \"hello\" \" world\"?\n", SPACE));
    }
    fresh();
    check_gbnf("until grammar", peg::until("</tag>"), false, str(
        "root ::= until-0\n", SPACE,
        "until-0 ::= | [<] until-0-01 | [^<] until-0\n",
        "until-0-01 ::= | [<] until-0-01 | [/] until-0-02 | [^/<] until-0\n",
        "until-0-02 ::= | [<] until-0-01 | [t] until-0-03 | [^<t] until-0\n",
        "until-0-03 ::= | [<] until-0-01 | [a] until-0-04 | [^<a] until-0\n",
        "until-0-04 ::= | [<] until-0-01 | [g] until-0-05 | [^<g] until-0\n",
        "until-0-05 ::= | [<] until-0-01 | [^<>] until-0\n"));
    fresh();
    check_gbnf("until grammar overlapping delimiter", peg::until("\n</parameter>\n"), false, str(
        "root ::= until-0\n", SPACE,
        "until-0 ::= | [\\n] until-0-01 | [^\\n] until-0\n",
        "until-0-01 ::= | [\\n] until-0-01 | [<] until-0-02 | [^\\n<] until-0\n",
        "until-0-02 ::= | [\\n] until-0-01 | [/] until-0-03 | [^\\n/] until-0\n",
        "until-0-03 ::= | [\\n] until-0-01 | [p] until-0-04 | [^\\np] until-0\n",
        "until-0-04 ::= | [\\n] until-0-01 | [a] until-0-05 | [^\\na] until-0\n",
        "until-0-05 ::= | [\\n] until-0-01 | [r] until-0-06 | [^\\nr] until-0\n",
        "until-0-06 ::= | [\\n] until-0-01 | [a] until-0-07 | [^\\na] until-0\n",
        "until-0-07 ::= | [\\n] until-0-01 | [m] until-0-08 | [^\\nm] until-0\n",
        "until-0-08 ::= | [\\n] until-0-01 | [e] until-0-09 | [^\\ne] until-0\n",
        "until-0-09 ::= | [\\n] until-0-01 | [t] until-0-10 | [^\\nt] until-0\n",
        "until-0-10 ::= | [\\n] until-0-01 | [e] until-0-11 | [^\\ne] until-0\n",
        "until-0-11 ::= | [\\n] until-0-01 | [r] until-0-12 | [^\\nr] until-0\n",
        "until-0-12 ::= | [\\n] until-0-01 | [>] until-0-13 | [^\\n>] until-0\n",
        "until-0-13 ::= | [^\\n] until-0\n"));
    fresh();
    check_gbnf("until grammar unicode delimiter", peg::until("<｜DSML｜"), false, str(
        "root ::= until-0\n", SPACE,
        "until-0 ::= | [<] until-0-01 | [^<] until-0\n",
        "until-0-01 ::= | [<] until-0-01 | [\\uFF5C] until-0-02 | [^<\\uFF5C] until-0\n",
        "until-0-02 ::= | [<] until-0-01 | [D] until-0-03 | [^<D] until-0\n",
        "until-0-03 ::= | [<] until-0-01 | [S] until-0-04 | [^<S] until-0\n",
        "until-0-04 ::= | [<] until-0-01 | [M] until-0-05 | [^<M] until-0\n",
        "until-0-05 ::= | [<] until-0-01 | [L] until-0-06 | [^<L] until-0\n",
        "until-0-06 ::= | [<] until-0-01 | [^<\\uFF5C] until-0\n"));
    fresh();
    check_gbnf("until grammar multiple delimiters", peg::until_one_of(["ab", "cd", "ef"]), false, str(
        "root ::= until-0\n", SPACE,
        "until-0 ::= | [a] until-0-01 | [c] until-0-03 | [e] until-0-05 | [^ace] until-0\n",
        "until-0-01 ::= | [a] until-0-01 | [c] until-0-03 | [e] until-0-05 | [^abce] until-0\n",
        "until-0-03 ::= | [a] until-0-01 | [c] until-0-03 | [e] until-0-05 | [^acde] until-0\n",
        "until-0-05 ::= | [a] until-0-01 | [c] until-0-03 | [e] until-0-05 | [^acef] until-0\n"));
    fresh();
    {
        let u = peg::until("</tag>");
        let l = peg::literal("</tag>");
        check_gbnf("ac grammar", peg::ac(peg::seq2(u, l), "</tag>"), false, str(
            "ac-3 ::= [<] ac-3-01 | [^<] ac-3\n",
            "ac-3-01 ::= [<] ac-3-01 | [/] ac-3-02 | [^/<] ac-3\n",
            "ac-3-02 ::= [<] ac-3-01 | [t] ac-3-03 | [^<t] ac-3\n",
            "ac-3-03 ::= [<] ac-3-01 | [a] ac-3-04 | [^<a] ac-3\n",
            "ac-3-04 ::= [<] ac-3-01 | [g] ac-3-05 | [^<g] ac-3\n",
            "ac-3-05 ::= [>] | [<] ac-3-01 | [^<>] ac-3\n",
            "root ::= ac-3\n", SPACE));
    }
    fresh();
    {
        let u = peg::until("\n</parameter>\n");
        let l = peg::literal("\n</parameter>\n");
        check_gbnf("ac grammar terminates at first delimiter", peg::ac(peg::seq2(u, l), "\n</parameter>\n"), false, str(
            "ac-3 ::= [\\n] ac-3-01 | [^\\n] ac-3\n",
            "ac-3-01 ::= [\\n] ac-3-01 | [<] ac-3-02 | [^\\n<] ac-3\n",
            "ac-3-02 ::= [\\n] ac-3-01 | [/] ac-3-03 | [^\\n/] ac-3\n",
            "ac-3-03 ::= [\\n] ac-3-01 | [p] ac-3-04 | [^\\np] ac-3\n",
            "ac-3-04 ::= [\\n] ac-3-01 | [a] ac-3-05 | [^\\na] ac-3\n",
            "ac-3-05 ::= [\\n] ac-3-01 | [r] ac-3-06 | [^\\nr] ac-3\n",
            "ac-3-06 ::= [\\n] ac-3-01 | [a] ac-3-07 | [^\\na] ac-3\n",
            "ac-3-07 ::= [\\n] ac-3-01 | [m] ac-3-08 | [^\\nm] ac-3\n",
            "ac-3-08 ::= [\\n] ac-3-01 | [e] ac-3-09 | [^\\ne] ac-3\n",
            "ac-3-09 ::= [\\n] ac-3-01 | [t] ac-3-10 | [^\\nt] ac-3\n",
            "ac-3-10 ::= [\\n] ac-3-01 | [e] ac-3-11 | [^\\ne] ac-3\n",
            "ac-3-11 ::= [\\n] ac-3-01 | [r] ac-3-12 | [^\\nr] ac-3\n",
            "ac-3-12 ::= [\\n] ac-3-01 | [>] ac-3-13 | [^\\n>] ac-3\n",
            "ac-3-13 ::= [\\n] | [^\\n] ac-3\n",
            "root ::= ac-3\n", SPACE));
    }
    fresh();
    check_gbnf("ac grammar multiple delimiters", peg::ac_one_of(peg::eps(), ["ab", "cd", "ef"]), false, str(
        "ac-1 ::= [a] ac-1-01 | [c] ac-1-03 | [e] ac-1-05 | [^ace] ac-1\n",
        "ac-1-01 ::= [b] | [a] ac-1-01 | [c] ac-1-03 | [e] ac-1-05 | [^abce] ac-1\n",
        "ac-1-03 ::= [d] | [a] ac-1-01 | [c] ac-1-03 | [e] ac-1-05 | [^acde] ac-1\n",
        "ac-1-05 ::= [f] | [a] ac-1-01 | [c] ac-1-03 | [e] ac-1-05 | [^acef] ac-1\n",
        "root ::= ac-1\n", SPACE));
    fresh();
    {
        let a = peg::literal("a");
        let b = peg::literal("b");
        check_gbnf("complex expressions with parentheses", peg::one_or_more(peg::choice2(a, b)), false, str("root ::= (\"a\" | \"b\")+\n", SPACE));
    }
    fresh();
    {
        let digit = peg::rule("digit", peg::chars("[0-9]", 1, 1), false);
        check_gbnf("rule references", peg::one_or_more(digit), false, str("digit ::= [0-9]\nroot ::= digit+\n", SPACE));
    }
    fresh();
    check_gbnf("escaping in literals", peg::literal("hello\nworld\n!"), false, str("root ::= \"hello\\nworld\\n!\"\n", SPACE));
    fresh();
    {
        let a = peg::literal("hello");
        let b = peg::literal("world");
        check_gbnf("operator<< (whitespace insertion)", peg::spaced(a, b), false, str("root ::= \"hello\" space \"world\"\n", SPACE));
    }
    fresh();
    {
        peg::rule("orphan", peg::literal("orphan"), false);
        let a = peg::literal("hello");
        let c = peg::rule("child", peg::literal(" world"), false);
        check_gbnf("emit only reachable rules", peg::seq2(a, c), false, str("child ::= \" world\"\nroot ::= \"hello\" child\n", SPACE));
    }
    fresh();
    {
        let a = peg::literal("a");
        let b = peg::literal("b");
        let c = peg::literal("c");
        let tg = peg::tag("t", peg::choice2(b, c));
        check_gbnf("tagged choice inside sequence gets parenthesized", peg::seq2(a, tg), false, str("root ::= \"a\" (\"b\" | \"c\")\n", SPACE));
    }
    fresh();
    {
        let a = peg::literal("a");
        let b = peg::literal("b");
        let tg = peg::tag("t", peg::seq2(a, b));
        let c = peg::literal("c");
        check_gbnf("tagged sequence inside choice gets parenthesized", peg::choice2(tg, c), false, str("root ::= \"a\" \"b\" | \"c\"\n", SPACE));
    }
    fresh();
    {
        let a = peg::literal("a");
        let b = peg::literal("b");
        check_gbnf("atomic choice inside repetition gets parenthesized", peg::one_or_more(peg::atomic(peg::choice2(a, b))), false, str("root ::= (\"a\" | \"b\")+\n", SPACE));
    }
    fresh();
    {
        let a = peg::literal("hello");
        let g = peg::gbnf(peg::literal("world"), "");
        check_gbnf("silent parser emits nothing in gbnf", peg::seq2(a, g), false, str("root ::= \"hello\"\n", SPACE));
    }
    fresh();
    {
        let a = peg::literal("a");
        let b = peg::literal("b");
        let c = peg::literal("c");
        let g = peg::gbnf(peg::choice2(b, c), "");
        let s = peg::seq2(a, g);
        let d = peg::literal("d");
        check_gbnf("silent choice inside sequence emits nothing", peg::seq2(s, d), false, str("root ::= \"a\" \"d\"\n", SPACE));
    }
    fresh();
    {
        let a = peg::literal("a");
        let tg = peg::tag("t", peg::gbnf(peg::literal("b"), ""));
        check_gbnf("silent wrapped in tag emits nothing", peg::seq2(a, tg), false, str("root ::= \"a\"\n", SPACE));
    }
    fresh();
    {
        let a = peg::literal("a");
        let g = peg::gbnf(peg::literal("b"), "[a-z]+");
        check_gbnf("gbnf parser emits custom grammar", peg::seq2(a, g), false, str("root ::= \"a\" [a-z]+\n", SPACE));
    }
    fresh();
    {
        let x = peg::literal("x");
        let a = peg::literal("a");
        let b = peg::literal("b");
        let tg = peg::tag("outer", peg::atomic(peg::choice2(a, b)));
        check_gbnf("nested transparent wrappers get parenthesized", peg::seq2(x, tg), false, str("root ::= \"x\" (\"a\" | \"b\")\n", SPACE));
    }
    fresh();
    {
        let node = json_schema::from_json("{\"type\": \"object\", \"properties\": {\"n\": {\"type\": \"integer\"}}, \"required\": [\"n\"]}");
        let open = peg::literal("<args>");
        let body = peg::schema(peg::json(), "args", node, false);
        // what llama.cpp writes for it
        check_gbnf("schema parser", peg::seq2(open, body), false, str(
            "args ::= \"{\" space args-n-kv space \"}\"\n",
            "args-n-kv ::= \"\\\"n\\\"\" space \":\" space integer\n",
            "integer ::= (\"-\"? integral-part)\n",
            "integral-part ::= [0] | [1-9] [0-9]{0,15}\n",
            "json-array ::= \"[\" space (\"]\" | json-value (space \",\" space json-value)* space \"]\")\n",
            "json-bool ::= \"true\" | \"false\"\n",
            "json-null ::= \"null\"\n",
            "json-number ::= \"-\"? (\"0\" | [1-9] [0-9]*) (\".\" [0-9]+)? ((\"e\" | \"E\") [+-]? [0-9]+)?\n",
            "json-object ::= \"{\" space (\"}\" | json-string space \":\" space json-value (space \",\" space json-string space \":\" space json-value)* space \"}\")\n",
            "json-string ::= \"\\\"\" ( [^\"\\\\] | \"\\\\\" ( [\"\\\\/ bfnrt] | \"u\" [0-9a-fA-F]{4} ) )* \"\\\"\"\n",
            "json-value ::= json-object | json-array | json-string | json-number | json-bool | json-null\n",
            "root ::= \"<args>\" args\n", SPACE));
    }
    for lazy in [false, true] {
        fresh();
        let a = peg::literal("a");
        let rule1 = peg::rule("rule-1", peg::seq2(a, peg::ref("rule-2")), false);
        let b = peg::literal("b");
        peg::rule("rule-2", peg::seq2(b, peg::ref("rule-3")), true);
        let c = peg::literal("c");
        peg::rule("rule-3", peg::seq2(c, peg::ref("rule-4")), false);
        peg::rule("rule-4", peg::literal("d"), true);
        if lazy {
            check_gbnf("emit only trigger rules (lazy)", rule1, true, str(
                "root ::= rule-2 | rule-4\nrule-2 ::= \"b\" rule-3\nrule-3 ::= \"c\" rule-4\nrule-4 ::= \"d\"\n", SPACE));
        } else {
            check_gbnf("emit only trigger rules", rule1, false, str(
                "root ::= rule-1\nrule-1 ::= \"a\" rule-2\nrule-2 ::= \"b\" rule-3\nrule-3 ::= \"c\" rule-4\nrule-4 ::= \"d\"\n", SPACE));
        }
    }
}

// parse cases

fn run(root: i64, text: const u8[:], flags: i64) -> peg::Res {
    let a = peg::build(root);
    let r, ok = peg::parse(a, text, flags, 0);
    if !ok { print("  parse error: ", peg::err); }
    return r;
}

fn chars_case(name: const u8[:], classes: const u8[:], text: const u8[:], want: i32) {
    fresh();
    let r = run(peg::chars(classes, 1, -1), text, 0);
    check(name, r.kind == want);
}

fn basic_tests() {
    chars_case("escape_sequence_newline", "[\\n\\t\\\\]", "\n", peg::SUCCESS);
    chars_case("escape_sequence_tab", "[\\n\\t\\\\]", "\t", peg::SUCCESS);
    chars_case("escape_sequence_backslash", "[\\n\\t\\\\]", "\\", peg::SUCCESS);
    chars_case("escape_sequence_space_fail", "[\\n\\t\\\\]", " ", peg::FAIL);
    chars_case("escaped_dash_a", "[a\\-z]", "a", peg::SUCCESS);
    chars_case("escaped_dash_literal", "[a\\-z]", "-", peg::SUCCESS);
    chars_case("escaped_dash_z", "[a\\-z]", "z", peg::SUCCESS);
    chars_case("escaped_dash_b_fail", "[a\\-z]", "b", peg::FAIL);

    fn hello_world() -> i64 {
        let a = peg::literal("hello");
        let b = peg::optional(peg::literal(" world"));
        return peg::seq2(a, b);
    }
    fresh();
    var r = run(hello_world(), "hello world", 0);
    check("optional_present", r.kind == peg::SUCCESS && r.end == 11);
    fresh();
    r = run(hello_world(), "hello", 0);
    check("optional_absent", r.kind == peg::SUCCESS && r.end == 5);
    fresh();
    r = run(hello_world(), "hello ", peg::FLAG_LENIENT);
    check("partial_match_need_more", r.kind == peg::NEED_MORE_INPUT);

    // JSON
    fresh();
    r = run(peg::json(), "{\"name\": \"test\", \"value\": 42, \"flag\": true}", 0);
    check("json object", r.kind == peg::SUCCESS && r.end == 43);
    fresh();
    r = run(peg::json(), "[1, 2.5e3, \"x\\u0041\", null, [false]]", 0);
    check("json array", r.kind == peg::SUCCESS && r.end == 36);
    fresh();
    r = run(peg::json(), "{\"name\": \"te", peg::FLAG_LENIENT);
    check("json partial", r.kind == peg::NEED_MORE_INPUT);
    fresh();
    r = run(peg::json(), "{\"name\" \"x\"}", 0);
    check("json invalid", r.kind == peg::FAIL);
    fresh();
    r = run(peg::json(), "3", peg::FLAG_LENIENT);
    check("json number not committed at end", r.kind == peg::NEED_MORE_INPUT);

    // until and tags
    fresh();
    {
        let open = peg::literal("<think>");
        let body = peg::tag("reasoning", peg::until("</think>"));
        let close = peg::literal("</think>");
        let rest = peg::tag("content", peg::rest());
        let p = peg::seq([open, body, close, rest]);
        r = run(p, "<think>hmm</think>answer", 0);
        var ok = r.kind == peg::SUCCESS && peg::result_nodes.len == 2;
        if ok {
            ok = peg::node_tag(peg::result_nodes[0]) == "reasoning" && peg::node_text(peg::result_nodes[0]) == "hmm"
                && peg::node_tag(peg::result_nodes[1]) == "content" && peg::node_text(peg::result_nodes[1]) == "answer";
        }
        check("tags", ok);
        fresh();
        let open2 = peg::literal("<think>");
        let body2 = peg::tag("reasoning", peg::until("</think>"));
        let p2 = peg::seq([open2, body2]);
        r = run(p2, "<think>hmm</thi", peg::FLAG_LENIENT);
        ok = r.kind == peg::SUCCESS && peg::result_nodes.len == 1 && peg::node_text(peg::result_nodes[0]) == "hmm";
        check("until stops before a partial delimiter", ok);
    }
    // invalid UTF-8 in until
    fresh();
    {
        let p = peg::tag("x", peg::until("|"));
        r = run(p, "a\xFFb|", 0);
        var san: u8[>..] = [];
        if peg::result_nodes.len == 1 { peg::sanitized_text(peg::result_nodes[0], san); }
        check("invalid utf-8 sanitized", r.kind == peg::SUCCESS && r.end == 3 && san == "a\xEF\xBF\xBDb");
    }
    // python dicts
    fresh();
    r = run(peg::python_value(), "{'a': True, \"b\": [1, None]}", 0);
    check("python dict", r.kind == peg::SUCCESS && r.end == 27);
}

fn main() {
    gbnf_tests();
    basic_tests();
    print(checked, " checked, ", failed, " failed");
    if failed > 0 { exit(1); }
}
