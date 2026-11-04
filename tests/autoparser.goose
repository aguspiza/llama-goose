// What the differential autoparser finds in each template of llama.cpp's
// models/templates, compared field by field with what llama.cpp's finds
// (tests/data/autoparser_cases.txt, made by scripts/gen-autoparser-cases.sh).
//   autoparser [-v] [name filter]
import std;
import os;
import exception;
import common from "../common";
import common.jinja;
import common.json_schema;
import common.chat_render;
import common.autoparser;
import common.peg;
import common.chat_peg;

var data: u8[>..] = [];
var at = 0;

fn field() -> const u8[:] {
    var n = 0;
    while data[at] != '\n' { n = n * 10 + (data[at] - '0'); at += 1; }
    at += 1;
    let s = data[at..at + n];
    at += n + 1;
    return s;
}

fn main() {
    let args = args();
    var verbose = false;
    var filter: u8[>..] = [];
    for a, i in args {
        if i == 0 { continue; }
        if a == "-v" { verbose = true; } else { filter.clear(); filter.append(a); }
    }
    if args.len > 1000 { var gr: u8[>..] = []; json_schema::to_grammar("{}", gr); }
    if !read_file("tests/data/autoparser_cases.txt", data) { print("cannot read the cases"); exit(1); }
    var checked = 0;
    var failed = 0;
    while at < data.len {
        field();
        let name = field();
        let ok = field() == "1";
        var expected: const u8[:][>..] = [];
        if ok { for k in 39 { expected.push(field()); } } else { expected.push(field()); }
        if filter.len > 0 && find(name, filter) < 0 { continue; }
        var src: u8[>..] = [];
        if !read_file(str("tests/data/templates/", name), src) { print("cannot read ", name); failed++; continue; }
        var t = chat_render::Template { .. };
        var parsed = false;
        fn parse_it() { t = chat_render::make_template(src, "<s>", "</s>"); parsed = true; }
        exception::try<jinja::Fail>(parse_it);
        checked++;
        if !parsed { print("FAIL ", name, ": does not parse: ", jinja::err); failed++; continue; }
        let got_ok = autoparser::analyze_template(t);
        if got_ok != ok {
            failed++;
            let mine: u8[>..] = if got_ok { str("works") } else { str("fails: ", autoparser::err) };
            let theirs: u8[>..] = if ok { str("works") } else { str("fails: ", expected[0]) };
            print("FAIL ", name, ": analysis ", mine, ", llama.cpp's ", theirs);
            continue;
        }
        if !ok { continue; }
        // the parser it makes, for a conversation with tools (that it runs)
        let tools = "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"}},\"required\":[\"city\"]}}}]";
        let g = chat_render::Gen { messages: "[{\"role\":\"user\",\"content\":\"Hi\"}]", tools: tools, add_generation_prompt: true, .. };
        if !autoparser::try_generate(t, g) && verbose { print("  generate failed: ", autoparser::err); }
        var k = 0;
        var bad = false;
        fn put(v: const u8[:]) {
            if v != expected[k] {
                if !bad { print("FAIL ", name); }
                bad = true;
                if verbose { print("  field ", k, ": llama.cpp '", expected[k], "', goose '", v, "'"); }
            }
            k++;
        }
        fn put_i(v: i64) { put(str(v)); }
        fn put_b(v: bool) { put(if v { "1" } else { "0" }); }
        put_i(autoparser::r_mode);
        put(autoparser::r_start);
        put(autoparser::r_end);
        put_i(autoparser::c_mode);
        put(autoparser::c_start);
        put(autoparser::c_end);
        put_i(autoparser::f_mode);
        put(autoparser::section_start);
        put(autoparser::section_end);
        put(autoparser::per_call_start);
        put(autoparser::per_call_end);
        put_b(autoparser::fun_name_is_key);
        put_b(autoparser::tools_array_wrapped);
        put_b(autoparser::openai_wrapper_trigger);
        put(autoparser::function_field);
        put(autoparser::name_field);
        put(autoparser::args_field);
        put(autoparser::id_field);
        put(autoparser::gen_id_field);
        put(autoparser::parameter_order);
        put(autoparser::fn_name_prefix);
        put(autoparser::fn_name_suffix);
        put(autoparser::fn_args_separator);
        put(autoparser::fn_close);
        put(autoparser::a_start);
        put(autoparser::a_end);
        put(autoparser::a_name_prefix);
        put(autoparser::a_name_suffix);
        put(autoparser::a_value_prefix);
        put(autoparser::a_value_suffix);
        put(autoparser::a_separator);
        put_b(autoparser::a_tolerate_ws);
        put_i(autoparser::id_pos);
        put(autoparser::id_prefix);
        put(autoparser::id_suffix);
        put(autoparser::user_start);
        put(autoparser::assistant_start);
        put(autoparser::preserved);
        put(autoparser::stops);
        if bad { failed++; }
    }
    print(checked, " checked, ", failed, " failed");
    if failed > 0 { exit(1); }
}
