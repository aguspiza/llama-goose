// llama.cpp's test-json-schema-to-grammar: converts its schemas and
// compares with the grammars llama.cpp's json_schema_to_grammar made of them
// (tests/data/schema_cases.txt, made by scripts/extract_schema_cases.cpp).
//   json_schema [tests/data/schema_cases.txt] [-v]
import std;
import os;
import common from "../common";

import common.json_schema;

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

fn main() {
    let args = args();
    var path: const u8[:] = "tests/data/schema_cases.txt";
    var verbose = false;
    for a, i in args {
        if i == 0 { continue; }
        if a == "-v" { verbose = true; } else { path = a; }
    }
    if !read_file(path, data) { print("cannot read ", path); exit(1); }
    var checked = 0;
    var failed = 0;
    while at < data.len {
        field();
        let name = field();
        let schema = field();
        let ok = field() == "1";
        let expected = field();
        var got: u8[>..] = [];
        let got_ok = json_schema::to_grammar(schema, got);
        checked++;
        if got_ok != ok || (ok && got != expected) {
            failed++;
            print("FAIL ", name);
            if verbose {
                print("  schema: ", schema);
                print("  llama.cpp ", ok, ":\n", expected);
                print("  goose     ", got_ok, ":\n", if got_ok { got[..] } else { json_schema::err[..] });
            }
        }
    }
    print(checked, " checked, ", failed, " failed");
    if failed > 0 { exit(1); }
}
