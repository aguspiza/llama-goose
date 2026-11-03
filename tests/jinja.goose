// llama.cpp's test-jinja: renders its template cases (tests/data/jinja_cases.txt,
// made by scripts/extract_jinja_cases.cpp) and compares with what llama.cpp's
// engine made of each; then its caps and string-parts checks.
//   jinja [tests/data/jinja_cases.txt] [-v]
import std;
import os;
import exception;
import common from "../common";
import common.jinja;
import common.jinja_lexer;
import common.jinja_parser;
import common.jinja_runtime;
import common.jinja_caps;

var data: u8[>..] = [];
var at = 0;

// The next field: "<byte length>\n<bytes>\n".
fn field() -> const u8[:] {
    var n = 0;
    while data[at] != '\n' { n = n * 10 + (data[at] - '0'); at += 1; }
    at += 1;
    let s = data[at..at + n];
    at += n + 1;
    return s;
}

var out: u8[>..] = [];

// Sets the members of the JSON object json as variables (global_from_json).
fn set_json(c: jinja::Ctx, json: const u8[:], mark_input: bool) {
    let v = jinja::from_json(json, mark_input);
    if jinja::kind(v) != jinja::K_OBJECT { jinja::fail_plain("global_from_json: input JSON value must be an object"); }
    for k in jinja::obj_len(v) { jinja::obj_set(c.env, jinja::obj_key(v, k), jinja::obj_value(v, k)); }
}

// Renders tmpl with the variables in the JSON object vars (none if empty):
// true and the output, or false and the error.
fn render(tmpl: const u8[:], vars: const u8[:]) -> bool, bool {
    var ok = false;
    var skipped = false;
    out.clear();
    fn body() {
        jinja::reset();
        let src, len = jinja::tokenize(tmpl);
        let p = jinja::parse(src, len);
        let c = jinja::new_ctx();
        if vars.len > 0 { set_json(c, vars, true); }
        let r = jinja::render(p, c);
        out.append(jinja::sbytes(r));
        ok = true;
    }
    exception::try<jinja::Fail>(body) { fail =>
        out.clear();
        out.append(jinja::err);
        skipped = fail.not_implemented;
    };
    return ok, skipped;
}

fn main() {
    let args = args();
    var path: const u8[:] = "tests/data/jinja_cases.txt";
    var verbose = false;
    for a, i in args {
        if i == 0 { continue; }
        if a == "-v" { verbose = true; } else { path = a; }
    }
    if !read_file(path, data) { print("cannot read ", path); exit(1); }

    var checked = 0;
    var failed = 0;
    var skipped = 0;
    while at < data.len {
        let what = field();
        let name = field();
        let tmpl = field();
        let vars = field();
        let expected = field();
        let status = field();
        let result = field();
        let ok, skip = render(tmpl, vars);
        checked++;
        var good = true;
        if status == "skip" {
            good = skip;
            if good { skipped++; }
        } else if what == "L" {
            good = !ok == (status == "error");
        } else if status == "error" {
            // llama.cpp failed: so must this, and for an error case with the expected message
            good = !ok && !skip && (what != "E" || find(out, expected) >= 0);
        } else {
            good = ok && out == result;
        }
        if !good {
            failed++;
            print("FAIL ", what, " ", name);
            if verbose {
                print("  template: ", tmpl);
                print("  vars:     ", vars);
                print("  llama.cpp ", status, ": ", result);
                print("  goose     ", if ok { "ok" } else if skip { "skip" } else { "error" }, ": ", out);
            }
        }
    }
    let more = test_caps() + test_stats() + test_string_parts();
    print(checked, " checked, ", failed + more, " failed, ", skipped, " not implemented in llama.cpp either");
    if failed + more > 0 { exit(1); }
}

var checks_failed = 0;

fn check(what: const u8[:], ok: bool) {
    if !ok { print("FAIL ", what); checks_failed++; }
}

fn parse(tmpl: const u8[:]) -> jinja::Program {
    let src, len = jinja::tokenize(tmpl);
    return jinja::parse(src, len);
}

// test-jinja's test_caps.
fn test_caps() -> i64 {
    checks_failed = 0;
    fn caps(tmpl: const u8[:]) -> jinja::Caps {
        var c = jinja::Caps { .. };
        fn body() { c = jinja::caps_get(parse(tmpl)); }
        exception::try<jinja::Fail>(body) { f => print("caps_get failed: ", jinja::err); checks_failed++; };
        return c;
    }
    var c = caps("{% for message in messages %}{{ message['role'] + ': ' + message['content'] }}{% endfor %}");
    check("string content: supports string content", c.supports_string_content);
    check("string content: does not support typed content", !c.supports_typed_content);
    c = caps("{% for message in messages %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] }}{% endfor %}{% endfor %}");
    check("typed content, raises on string: does not support string content", !c.supports_string_content);
    check("typed content, raises on string: supports typed content", c.supports_typed_content);
    c = caps("{% for message in messages %}{{ message['content'][0]['text'] }}{% endfor %}");
    check("typed content, silently drops string: does not support string content", !c.supports_string_content);
    check("typed content, silently drops string: supports typed content", c.supports_typed_content);
    c = caps("{% for message in messages %}{% if message['content'] %}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for part in message['content'] %}{{ part['text'] }}{% endfor %}{% endif %}{% endif %}{% endfor %}");
    check("string and typed content: supports string content", c.supports_string_content);
    check("string and typed content: supports typed content", c.supports_typed_content);
    return checks_failed;
}

// test-jinja's test_stats: what a render marks as read.
fn test_stats() -> i64 {
    checks_failed = 0;
    fn body() {
        jinja::reset();
        let tmpl = "{{val.num}} {{val.str}} {{val.arr[0]}} {{val.obj.key1}} {{val.nested | tojson}}";
        let p = parse(tmpl);
        let c = jinja::new_ctx();
        set_json(c, "{\"val\": {\"num\": 1, \"str\": \"abc\", \"arr\": [1, 2, 3], \"obj\": {\"key1\": 1, \"key2\": 2, \"key3\": 3}, \"nested\": {\"inner_key1\": [1, 2], \"inner_key2\": {\"a\": \"x\", \"b\": \"y\"}}, \"mixed\": {\"used\": 1, \"unused\": 2}}}", true);
        jinja::stats = true;
        jinja::render(p, c);
        jinja::stats = false;
        let v = jinja::get_var(c, "val");
        fn used(x: i64) -> bool { jinja::vals[x].used }
        fn at(x: i64, k: const u8[:]) -> i64 { jinja::obj_get_name(x, k) }
        check("num is used", used(at(v, "num")));
        check("str is used", used(at(v, "str")));
        check("arr is used", used(at(v, "arr")));
        check("arr[0] is used", used(jinja::arr_at(at(v, "arr"), 0)));
        check("arr[1] is not used", !used(jinja::arr_at(at(v, "arr"), 1)));
        check("obj is used", used(at(v, "obj")));
        check("obj.key1 is used", used(at(at(v, "obj"), "key1")));
        check("obj.key2 is not used", !used(at(at(v, "obj"), "key2")));
        check("inner_key1[0] is used", used(jinja::arr_at(at(at(v, "nested"), "inner_key1"), 0)));
        check("inner_key2.a is used", used(at(at(at(v, "nested"), "inner_key2"), "a")));
    }
    exception::try<jinja::Fail>(body) { f => print("stats render failed: ", jinja::err); checks_failed++; };
    jinja::stats = false;
    return checks_failed;
}

// test-jinja's test_string_parts: neighbours of the same kind are joined.
fn test_string_parts() -> i64 {
    checks_failed = 0;
    fn body() {
        jinja::reset();
        let p = parse("{{ val.a }}{{ val.b }}-{{ val.c }}");
        let c = jinja::new_ctx();
        set_json(c, "{\"val\": {\"a\": \"A\", \"b\": \"B\", \"c\": \"C\"}}", true);
        let rv = jinja::render(p, c);
        let r = jinja::vals[rv];
        check("3 parts after the merge", r.n == 3);
        if r.n == 3 {
            let p0 = jinja::parts[r.a];
            let p1 = jinja::parts[r.a + 1];
            let p2 = jinja::parts[r.a + 2];
            check("part 0 is the merged input", jinja::part_bytes(p0) == "AB" && p0.input);
            check("part 1 is from the template", jinja::part_bytes(p1) == "-" && !p1.input);
            check("part 2 is input", jinja::part_bytes(p2) == "C" && p2.input);
        }
    }
    exception::try<jinja::Fail>(body) { f => print("string parts render failed: ", jinja::err); checks_failed++; };
    return checks_failed;
}
