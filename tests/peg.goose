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
        "root ::= until-1\n", SPACE,
        "until-1 ::= | [<] until-1-01 | [^<] until-1\n",
        "until-1-01 ::= | [<] until-1-01 | [/] until-1-02 | [^/<] until-1\n",
        "until-1-02 ::= | [<] until-1-01 | [t] until-1-03 | [^<t] until-1\n",
        "until-1-03 ::= | [<] until-1-01 | [a] until-1-04 | [^<a] until-1\n",
        "until-1-04 ::= | [<] until-1-01 | [g] until-1-05 | [^<g] until-1\n",
        "until-1-05 ::= | [<] until-1-01 | [^<>] until-1\n"));
    fresh();
    check_gbnf("until grammar overlapping delimiter", peg::until("\n</parameter>\n"), false, str(
        "root ::= until-1\n", SPACE,
        "until-1 ::= | [\\n] until-1-01 | [^\\n] until-1\n",
        "until-1-01 ::= | [\\n] until-1-01 | [<] until-1-02 | [^\\n<] until-1\n",
        "until-1-02 ::= | [\\n] until-1-01 | [/] until-1-03 | [^\\n/] until-1\n",
        "until-1-03 ::= | [\\n] until-1-01 | [p] until-1-04 | [^\\np] until-1\n",
        "until-1-04 ::= | [\\n] until-1-01 | [a] until-1-05 | [^\\na] until-1\n",
        "until-1-05 ::= | [\\n] until-1-01 | [r] until-1-06 | [^\\nr] until-1\n",
        "until-1-06 ::= | [\\n] until-1-01 | [a] until-1-07 | [^\\na] until-1\n",
        "until-1-07 ::= | [\\n] until-1-01 | [m] until-1-08 | [^\\nm] until-1\n",
        "until-1-08 ::= | [\\n] until-1-01 | [e] until-1-09 | [^\\ne] until-1\n",
        "until-1-09 ::= | [\\n] until-1-01 | [t] until-1-10 | [^\\nt] until-1\n",
        "until-1-10 ::= | [\\n] until-1-01 | [e] until-1-11 | [^\\ne] until-1\n",
        "until-1-11 ::= | [\\n] until-1-01 | [r] until-1-12 | [^\\nr] until-1\n",
        "until-1-12 ::= | [\\n] until-1-01 | [>] until-1-13 | [^\\n>] until-1\n",
        "until-1-13 ::= | [^\\n] until-1\n"));
    fresh();
    check_gbnf("until grammar unicode delimiter", peg::until("<｜DSML｜"), false, str(
        "root ::= until-1\n", SPACE,
        "until-1 ::= | [<] until-1-01 | [^<] until-1\n",
        "until-1-01 ::= | [<] until-1-01 | [\\uFF5C] until-1-02 | [^<\\uFF5C] until-1\n",
        "until-1-02 ::= | [<] until-1-01 | [D] until-1-03 | [^<D] until-1\n",
        "until-1-03 ::= | [<] until-1-01 | [S] until-1-04 | [^<S] until-1\n",
        "until-1-04 ::= | [<] until-1-01 | [M] until-1-05 | [^<M] until-1\n",
        "until-1-05 ::= | [<] until-1-01 | [L] until-1-06 | [^<L] until-1\n",
        "until-1-06 ::= | [<] until-1-01 | [^<\\uFF5C] until-1\n"));
    fresh();
    check_gbnf("until grammar multiple delimiters", peg::until_one_of(["ab", "cd", "ef"]), false, str(
        "root ::= until-3\n", SPACE,
        "until-3 ::= | [a] until-3-01 | [c] until-3-03 | [e] until-3-05 | [^ace] until-3\n",
        "until-3-01 ::= | [a] until-3-01 | [c] until-3-03 | [e] until-3-05 | [^abce] until-3\n",
        "until-3-03 ::= | [a] until-3-01 | [c] until-3-03 | [e] until-3-05 | [^acde] until-3\n",
        "until-3-05 ::= | [a] until-3-01 | [c] until-3-03 | [e] until-3-05 | [^acef] until-3\n"));
    fresh();
    check_gbnf("through grammar", peg::through_s(peg::until("</tag>"), "</tag>"), false, str(
        "root ::= until-3\n", SPACE,
        "until-3 ::= [<] until-3-01 | [^<] until-3\n",
        "until-3-01 ::= [<] until-3-01 | [/] until-3-02 | [^/<] until-3\n",
        "until-3-02 ::= [<] until-3-01 | [t] until-3-03 | [^<t] until-3\n",
        "until-3-03 ::= [<] until-3-01 | [a] until-3-04 | [^<a] until-3\n",
        "until-3-04 ::= [<] until-3-01 | [g] until-3-05 | [^<g] until-3\n",
        "until-3-05 ::= [>] | [<] until-3-01 | [^<>] until-3\n"));
    fresh();
    check_gbnf("through grammar terminates at first delimiter", peg::through_s(peg::until("\n</parameter>\n"), "\n</parameter>\n"), false, str(
        "root ::= until-3\n", SPACE,
        "until-3 ::= [\\n] until-3-01 | [^\\n] until-3\n",
        "until-3-01 ::= [\\n] until-3-01 | [<] until-3-02 | [^\\n<] until-3\n",
        "until-3-02 ::= [\\n] until-3-01 | [/] until-3-03 | [^\\n/] until-3\n",
        "until-3-03 ::= [\\n] until-3-01 | [p] until-3-04 | [^\\np] until-3\n",
        "until-3-04 ::= [\\n] until-3-01 | [a] until-3-05 | [^\\na] until-3\n",
        "until-3-05 ::= [\\n] until-3-01 | [r] until-3-06 | [^\\nr] until-3\n",
        "until-3-06 ::= [\\n] until-3-01 | [a] until-3-07 | [^\\na] until-3\n",
        "until-3-07 ::= [\\n] until-3-01 | [m] until-3-08 | [^\\nm] until-3\n",
        "until-3-08 ::= [\\n] until-3-01 | [e] until-3-09 | [^\\ne] until-3\n",
        "until-3-09 ::= [\\n] until-3-01 | [t] until-3-10 | [^\\nt] until-3\n",
        "until-3-10 ::= [\\n] until-3-01 | [e] until-3-11 | [^\\ne] until-3\n",
        "until-3-11 ::= [\\n] until-3-01 | [r] until-3-12 | [^\\nr] until-3\n",
        "until-3-12 ::= [\\n] until-3-01 | [>] until-3-13 | [^\\n>] until-3\n",
        "until-3-13 ::= [\\n] | [^\\n] until-3\n"));
    fresh();
    {
        let e = peg::eps();
        let ab = peg::literal("ab");
        let cd = peg::literal("cd");
        let ef = peg::literal("ef");
        check_gbnf("through grammar multiple delimiters", peg::through_b(e, [peg::Branch { delim: ab }, peg::Branch { delim: cd }, peg::Branch { delim: ef }], false), false, str(
            "root ::= until-4\n", SPACE,
            "until-4 ::= [a] until-4-01 | [c] until-4-03 | [e] until-4-05 | [^ace] until-4\n",
            "until-4-01 ::= [b] | [a] until-4-01 | [c] until-4-03 | [e] until-4-05 | [^abce] until-4\n",
            "until-4-03 ::= [d] | [a] until-4-01 | [c] until-4-03 | [e] until-4-05 | [^acde] until-4\n",
            "until-4-05 ::= [f] | [a] until-4-01 | [c] until-4-03 | [e] until-4-05 | [^acef] until-4\n"));
    }
    fresh();
    {
        // (a + b) | cd: cd, then b, then a
        let cd = peg::literal("cd");
        let b = peg::literal("b");
        let a = peg::literal("a");
        let ch = peg::choice2(peg::seq2(a, b), cd);
        check_gbnf("until grammar with parser delimiters", peg::until_p(ch), false, str(
            "root ::= until-5\n", SPACE,
            "until-5 ::= | [a] until-5-01 | [c] until-5-03 | [^ac] until-5\n",
            "until-5-01 ::= | [a] until-5-01 | [c] until-5-03 | [^abc] until-5\n",
            "until-5-03 ::= | [a] until-5-01 | [c] until-5-03 | [^acd] until-5\n"));
    }
    fresh();
    {
        let b = peg::atomic(peg::literal("b"));
        let a = peg::literal("a");
        let tg = peg::tag("open", peg::seq2(a, b));
        check_gbnf("delimiters see through tags and atomics", peg::until_p(tg), false, str(
            "root ::= until-5\n", SPACE,
            "until-5 ::= | [a] until-5-01 | [^a] until-5\n",
            "until-5-01 ::= | [a] until-5-01 | [^ab] until-5\n"));
    }
    fresh();
    {
        let d = peg::literal("d");
        let b = peg::literal("b");
        let ch = peg::choice2(b, d);
        let a = peg::literal("a");
        check_gbnf("delimiters expand a choice in a sequence", peg::until_p(peg::seq2(a, ch)), false, str(
            "root ::= until-5\n", SPACE,
            "until-5 ::= | [a] until-5-01 | [^a] until-5\n",
            "until-5-01 ::= | [a] until-5-01 | [^abd] until-5\n"));
    }
    fresh();
    {
        let e = peg::eps();
        let ab = peg::literal("ab");
        let x = peg::literal("x");
        let cd = peg::literal("cd");
        check_gbnf("through grammar branches continue with their rest", peg::through_b(e, [peg::Branch { delim: ab, rest: x }, peg::Branch { delim: cd }], false), false, str(
            "root ::= until-4\n", SPACE,
            "until-4 ::= [a] until-4-01 | [c] until-4-03 | [^ac] until-4\n",
            "until-4-01 ::= [b] until-4-rest-0 | [a] until-4-01 | [c] until-4-03 | [^abc] until-4\n",
            "until-4-03 ::= [d] | [a] until-4-01 | [c] until-4-03 | [^acd] until-4\n",
            "until-4-rest-0 ::= \"x\"\n"));
    }
    fresh();
    {
        let e = peg::eps();
        let ab = peg::literal("ab");
        let x = peg::literal("x");
        check_gbnf("through grammar optional may end before a delimiter", peg::through_b(e, [peg::Branch { delim: ab, rest: x }], true), false, str(
            "root ::= until-3\n", SPACE,
            "until-3 ::= | [a] until-3-01 | [^a] until-3\n",
            "until-3-01 ::= | [b] until-3-rest-0 | [a] until-3-01 | [^ab] until-3\n",
            "until-3-rest-0 ::= \"x\"\n"));
    }
    for lazy in [false, true] {
        fresh();
        let ab = peg::literal("ab");
        let cd = peg::literal("cd");
        let scan = peg::until_p(peg::choice2(ab, cd));
        let two = peg::trigger_rule_sr("two", cd, peg::literal("y"));
        let one = peg::trigger_rule_sr("one", ab, peg::literal("x"));
        let root = peg::seq2(scan, peg::choice2(one, two));
        if !lazy {
            check_gbnf("trigger rules emit their start and rest when not lazy", root, false, str(
                "one ::= \"ab\" \"x\"\n",
                "root ::= until-3 (one | two)\n", SPACE,
                "two ::= \"cd\" \"y\"\n",
                "until-3 ::= | [a] until-3-01 | [c] until-3-03 | [^ac] until-3\n",
                "until-3-01 ::= | [a] until-3-01 | [c] until-3-03 | [^abc] until-3\n",
                "until-3-03 ::= | [a] until-3-01 | [c] until-3-03 | [^acd] until-3\n"));
        } else {
            check_gbnf("trigger rules make the lazy grammar scan for their starts", root, true, str(
                "one-rest ::= \"x\"\n",
                "root ::= trigger\n", SPACE,
                "trigger ::= | [a] trigger-01 | [c] trigger-03 | [^ac] trigger\n",
                "trigger-01 ::= | [b] one-rest | [a] trigger-01 | [c] trigger-03 | [^abc] trigger\n",
                "trigger-03 ::= | [d] two-rest | [a] trigger-01 | [c] trigger-03 | [^acd] trigger\n",
                "two-rest ::= \"y\"\n"));
        }
    }
    fresh();
    {
        peg::set_tokens(["<a>", "<b>"], [1, 2]);
        let x = peg::literal("x");
        let start = peg::seq2(peg::token("<a>"), x);
        let tool = peg::trigger_rule_sr("tool", start, peg::token("<b>"));
        let root = peg::seq2(peg::until_p(start), tool);
        check_gbnf("scanning grammar lets other named tokens through", root, true, str(
            "root ::= trigger\n", SPACE,
            "tool-rest ::= <[2]>\n",
            "trigger ::= | <[1]> trigger-01 | (. | <[2]>) trigger\n",
            "trigger-01 ::= | [x] tool-rest | <[1]> trigger-01 | ([^x] | <[2]>) trigger\n"));
        peg::clear_tokens();
    }
    fresh();
    {
        peg::set_tokens(["<a>", "<b>"], [1, 2]);
        let one = peg::tokens(["<a>", "<b>"], false);
        let other = peg::tokens(["<a>", "<b>"], true);
        check_gbnf("token sets", peg::seq2(one, other), false, str("root ::= (<[1]> | <[2]>) !<[1-2]>\n", SPACE));
        peg::clear_tokens();
    }
    fresh();
    {
        let e = peg::eps();
        let xab = peg::literal("xab");
        let one = peg::literal("1");
        let ab = peg::literal("ab");
        let two = peg::literal("2");
        check_gbnf("through grammar suffix delimiter offers both rests", peg::through_b(e, [peg::Branch { delim: xab, rest: one }, peg::Branch { delim: ab, rest: two }], false), false, str(
            "root ::= until-5\n", SPACE,
            "until-5 ::= [x] until-5-01 | [a] until-5-04 | [^ax] until-5\n",
            "until-5-01 ::= [x] until-5-01 | [a] until-5-02 | [^ax] until-5\n",
            "until-5-02 ::= [b] (until-5-rest-0 | until-5-rest-1) | [x] until-5-01 | [a] until-5-04 | [^abx] until-5\n",
            "until-5-04 ::= [b] until-5-rest-1 | [x] until-5-01 | [a] until-5-04 | [^abx] until-5\n",
            "until-5-rest-0 ::= \"1\"\n",
            "until-5-rest-1 ::= \"2\"\n"));
    }
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
