// llama.cpp's test-chat-template, its built-in templates part: applies each
// built-in template and each test case's template to the test conversation
// and compares with what libllama wrote (tests/data/chat_cases.txt, made by
// scripts/extract_chat_cases.cpp).
//   chat_template [tests/data/chat_cases.txt]
import std;
import os;
import llama from "../llama";
import llama.chat;

var data: u8[>..] = [];
var at = 0;

// The next field: "<byte length>\n<bytes>\n".
fn field() -> const u8[:] {
    var n = 0;
    while data[at] != '\n' { n = n * 10 + (data[at] - '0') as i64; at += 1; }
    at += 1;
    let s = data[at..at + n];
    at += n + 1;
    return s;
}

fn count() -> i64 {
    let v, ok = parse_int(field());
    assert(ok);
    return v;
}

fn main() {
    let args = args();
    let path = if args.len > 1 { args[1] } else { "tests/data/chat_cases.txt" };
    if !read_file(path, data) { print("cannot read ", path); exit(1); }

    var conv: chat::Message[>..] = [];
    for i in count() { conv.push(chat::Message { role: field(), content: field() }); }

    var out: u8[>..] = [];
    var failed = 0;
    var checked = 0;
    fn check(name: const u8[:], tmpl: const u8[:], msgs: const chat::Message[:], add_ass: bool, expected: const u8[:]) {
        let ok = chat::apply(tmpl, msgs, add_ass, out);
        let got = if ok { out[..] } else { "<ERROR>" };
        checked += 1;
        if got != expected {
            failed += 1;
            print("FAIL ", name, " add_ass=", add_ass);
            print("  expected: ", expected);
            print("  got:      ", got);
        }
    }

    let n_names = count();
    if n_names != chat::NAMES.len { print("FAIL: ", n_names, " built-in templates, Goose has ", chat::NAMES.len); failed += 1; }
    for i in n_names {
        let name = field();
        if i < chat::NAMES.len && chat::NAMES[i].name != name { print("FAIL: built-in template ", i, " is ", name); failed += 1; }
        check(name, name, conv, true, field());
        check(name, name, conv, false, field());
    }

    for i in count() {
        let name = field();
        let tmpl = field();
        field(); field(); field(); field(); field();    // jinja's expectations, bos, eos, supported_with_jinja
        var msgs: chat::Message[>..] = [];
        msgs.append(conv);
        for k in count() { msgs.push(chat::Message { role: field(), content: field() }); }
        check(name, tmpl, msgs, true, field());
        check(name, tmpl, msgs, false, field());
    }
    if chat::apply("INVALID TEMPLATE", conv, true, out) { print("FAIL: INVALID TEMPLATE was recognized"); failed += 1; }

    print(checked, " checked, ", failed, " failed");
    if failed > 0 { exit(1); }
}
