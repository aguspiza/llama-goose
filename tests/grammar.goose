// llama.cpp's test-grammar-parser and test-grammar-integration: parses
// their grammars, builds them, and runs their strings through them, and
// compares with what llama.cpp's grammar code made of each
// (tests/data/grammar_cases.txt, made by scripts/extract_grammar_cases.cpp).
//   grammar [tests/data/grammar_cases.txt] [-v]
import std;
import os;
import ggml from "../ggml";
import ggml_cpu from "../ggml/cpu";
import ggml_shim from "../ggml/shim";
import ggml.gguf;
import llama from "../llama";
import llama.unicode;
import llama.grammar;

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

// The parse as the extractor writes it: the symbols ("name id" lines, by
// name) and the rules (a line each, "type:value " per element).
fn dump(symbols: u8[>..]&, rules: u8[>..]&) {
    var order: i64[>..] = [];
    grammar::sorted_symbols(order);
    for k in order { format(symbols, grammar::symbol_name(k), " ", k, "\n"); }
    for r in grammar::prules {
        for e in grammar::pelems[r.at..r.at + r.len] { format(rules, e.t, ":", e.v, " "); }
        rules.push('\n');
    }
}

// test-grammar-integration's match_string: the input's characters (and
// 0xff-marked token ids) accepted in turn; whether the grammar can end there.
fn match_string(input: const u8[:]) -> bool {
    var off = 0;
    var piece: u8[>..] = [];
    while off < input.len {
        piece.clear();
        var token: i32 = 0;
        if input[off] == 0xff && off + 5 <= input.len {
            let v: u32 = (input[off + 1] as u32 << 24) | (input[off + 2] as u32 << 16) | (input[off + 3] as u32 << 8) | input[off + 4];
            token = v as! i32;
            format(piece, "<[", v, "]>");
            off += 5;
        } else {
            let cpt, next, ok = unicode::cpt_from_utf8(input, off);
            if ok {
                unicode::cpt_to_utf8(piece, cpt);
                off = next;
            } else {
                unicode::cpt_to_utf8(piece, 0xFFFD);
                off += 1;
            }
        }
        if !grammar::accept_token(token, piece) { return false; }
    }
    return grammar::can_end();
}

fn main() {
    let args = args();
    var path: const u8[:] = "tests/data/grammar_cases.txt";
    var verbose = false;
    for a, i in args {
        if i == 0 { continue; }
        if a == "-v" { verbose = true; } else { path = a; }
    }
    if !read_file(path, data) { print("cannot read ", path); exit(1); }

    var checked = 0;
    var failed = 0;
    var gtext: u8[>..] = [];     // the grammar of the M cases
    var g_ok = false;
    while at < data.len {
        let what = field();
        checked++;
        if what == "P" {
            let text = field();
            let ok = field() == "1";
            let symbols = field();
            let rules = field();
            let got_ok = grammar::parse(text, false);
            var got_symbols: u8[>..] = [];
            var got_rules: u8[>..] = [];
            dump(got_symbols, got_rules);
            if got_ok != ok || got_symbols != symbols || got_rules != rules {
                failed++;
                print("FAIL parse: ", text);
                if verbose {
                    print("  llama.cpp ", ok, "\n", symbols, rules);
                    print("  goose     ", got_ok, " ", grammar::err, "\n", got_symbols, got_rules);
                }
            }
        } else if what == "B" {
            let text = field();
            let root = field();
            let ok = field() == "1";
            let got = grammar::init(text, root, false);
            if got != ok {
                failed++;
                print("FAIL build (root ", root, "): ", text);
                if verbose { print("  llama.cpp ", ok, ", goose ", got, " ", grammar::err); }
            }
        } else if what == "G" {
            gtext.clear();
            gtext.append(field());
            checked--;
        } else {
            let input = field();
            let matched = field() == "1";
            g_ok = grammar::init(gtext, "root", false);
            let got = g_ok && match_string(input);
            if got != matched {
                failed++;
                print("FAIL match ", if matched { "(should match)" } else { "(should not match)" }, ": ", input);
                if verbose { print("  grammar: ", gtext); }
            }
        }
    }
    print(checked, " checked, ", failed, " failed");
    if failed > 0 { exit(1); }
}
