// llama.cpp's examples/simple: greedy generation from a prompt.
//   simple -m model.gguf [-n n_predict] [prompt]
// port-of: examples/simple/simple.cpp
import std;
import os;
import ggml from "../ggml";
import llama from "../llama";
import llama.vocab;
import llama.model;
import llama.context;

fn usage() {
    print("example usage:\n\n    simple -m model.gguf [-n n_predict] [prompt]\n");
    exit(1);
}

fn argmax(xs: const f32[:]) -> i32 {
    var best = 0;
    for x, i in xs { if x > xs[best] { best = i; } }
    return best as i32;
}

fn piece(out: u8[>..]&, t: i32) { vocab::token_to_piece(t, out, 0, true); }

fn main() {
    let args = args();
    var model_path: u8[>..] = [];
    var n_predict = 32;
    var prompt: u8[>..] = [];
    var i = 1;
    while i < args.len {
        let a = args[i];
        if a == "-m" && i + 1 < args.len { model_path.clear(); model_path.append(args[i + 1]); i += 2; continue; }
        if a == "-n" && i + 1 < args.len {
            let v, ok = parse_int(args[i + 1]);
            if !ok { usage(); }
            n_predict = v;
            i += 2;
            continue;
        }
        // the rest is the prompt
        for k in i..args.len {
            if k > i { prompt.push(' '); }
            prompt.append(args[k]);
        }
        break;
    }
    if model_path.len == 0 { usage(); }
    if prompt.len == 0 { prompt.append("Hello my name is"); }

    var m = model::Model { .. };
    model::load(m, model_path);

    var prompt_tokens: i32[>..] = [];
    vocab::tokenize(prompt, true, true, prompt_tokens);
    let n_prompt = prompt_tokens.len;

    var c = context::Context { .. };
    context::init(c, m, n_prompt + n_predict - 1);

    {
        var text: u8[>..] = [];
        for t in prompt_tokens { piece(text, t); }
        write_stdout(text);
    }

    // LLAMA_GOOSE_DUMP_LOGITS=file: append each step's logits to file (f32)
    var dump_path: u8[>..] = [];
    if env("LLAMA_GOOSE_DUMP_LOGITS", dump_path) {
        write_file(dump_path, "");
        write_stderr(str("prompt tokens: ", prompt_tokens, "\n"));
    }

    let t_start = clock();
    var n_decode = 0;
    var logits: f32[>..] = [];
    var batch: i32[>..] = [];
    batch.append(prompt_tokens);
    var n_pos = 0;
    while n_pos + batch.len < n_prompt + n_predict {
        context::decode(c, m, batch, logits);
        if dump_path.len > 0 { append_file(dump_path, bytes_of(logits)); }
        n_pos += batch.len;
        let t = argmax(logits);
        if vocab::is_eog(t) { break; }
        var text: u8[>..] = [];
        piece(text, t);
        write_stdout(text);
        flush_stdout();
        batch.clear();
        batch.push(t);
        n_decode += 1;
    }
    print("");
    let dt = clock() - t_start;
    write_stderr(str("main: decoded ", n_decode, " tokens in ", dt, " s, speed: ", n_decode / dt, " t/s\n"));
}
