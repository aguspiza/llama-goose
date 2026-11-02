// llama.cpp's test-tokenizer-0: tokenizes the texts in <vocab>.inp and
// compares with the ids in <vocab>.out, for each vocab given.
//   tokenizer_0 <dir>/ggml-vocab-llama-spm.gguf ...
import std;
import os;
import ggml from "../ggml";
import ggml.gguf;
import llama from "../llama";
import llama.vocab;

fn main() {
    let args = args();
    var failed = 0;
    for a, k in args {
        if k == 0 { continue; }
        failed += check_vocab(a);
    }
    if failed > 0 { print(failed, " tests failed"); exit(1); }
    print("tokenizer tests ok");
}

fn check_vocab(path: const u8[:]) -> i64 {
    let ctx = gguf::init_from_file(path);
    if ctx.p == 0 { abort(str("cannot open ", path)); }
    vocab::load(ctx);
    var inp: u8[>..] = [];
    var outp: u8[>..] = [];
    if !read_file(str(path, ".inp"), inp) || !read_file(str(path, ".out"), outp) { abort("missing .inp/.out"); }
    let texts = split(inp, "\n__ggml_vocab_test__\n");
    let lines = split(outp, '\n');
    var failed = 0;
    var tokens: i32[>..] = [];
    for t, i in texts {
        tokens.clear();
        vocab::tokenize(t, false, false, tokens);
        var want: i32[>..] = [];
        each_word(lines[i]) { let v, ok = parse_int(it); want.push(v as i32); };
        if tokens != want {
            failed += 1;
            print("FAILED ", path, " test ", i, ": '", t, "'");
            print("  want ", want);
            print("  got  ", tokens);
        }
    }
    print(path, ": ", texts.len - failed, "/", texts.len, " ok");
    return failed;
}
