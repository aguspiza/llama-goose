// llama.cpp's examples/batched: one prompt, then n_parallel sequences
// generated from it, one token of each per batch, in a cache unified for
// all of them (-kvu; the cache split per sequence is not ported).
//   batched -m model.gguf [-p prompt] [-n n_predict] [-np n_parallel]
//           [--temp t] [--top-k k] [--top-p p] [-s seed]
// port-of: examples/batched/batched.cpp
import std;
import os;
import ggml from "../ggml";
import ggml_cpu from "../ggml/cpu";
import ggml_shim from "../ggml/shim";
import llama from "../llama";
import llama.vocab;
import llama.model;
import llama.context;
import llama.sampler;
import common from "../common";
import common.console;

fn usage() {
    print("example usage:\n\n    batched -m model.gguf [-p prompt] [-n n_predict] [-np n_parallel] [--temp t] [--top-k k] [--top-p p] [-s seed]\n");
    exit(1);
}

fn piece(out: u8[>..]&, t: i32) { vocab::token_to_piece(t, out, 0, true); }

// The sequences' samplers, one chain each (top-k, top-p, temp, dist).
var chains: sampler::Chain[>..] = [];

fn main() {
    let args = console::utf8_args();     // the prompt in UTF-8, on Windows too
    var model_path: u8[>..] = [];
    var prompt: u8[>..] = [];
    prompt.append("Hello my name is");
    var n_predict = 32;
    var n_parallel = 1;
    var top_k = 40;
    var top_p: f32 = 0.95;
    var temp: f32 = 0.80;
    var seed: u32 = sampler::DEFAULT_SEED;
    var i = 1;
    while i < args.len {
        let a = args[i];
        if a == "-kvu" || a == "--kv-unified" { i++; continue; }
        guard i + 1 < args.len else { usage(); }
        let v = args[i + 1];
        if a == "-m" { model_path.clear(); model_path.append(v); }
        else if a == "-p" { prompt.clear(); prompt.append(v); }
        else if a == "-n" || a == "-np" || a == "--top-k" || a == "-s" || a == "--seed" {
            let x, ok = parse_int(v);
            if !ok { usage(); }
            if a == "-n" { n_predict = x; } else if a == "-np" { n_parallel = x; } else if a == "--top-k" { top_k = x; } else { seed = x as! u32; }
        } else if a == "--temp" || a == "--top-p" {
            let x, ok = parse_flt(v);
            if !ok { usage(); }
            if a == "--temp" { temp = x as f32; } else { top_p = x as f32; }
        } else { usage(); }
        i += 2;
    }
    if model_path.len == 0 { usage(); }
    guard n_parallel >= 1 && n_parallel <= 64 else { abort("n_parallel: 1 to 64"); }

    var m = model::Model { .. };
    model::load(m, model_path);

    var tokens: i32[>..] = [];
    vocab::tokenize(prompt, true, true, tokens);
    let n_kv_req = tokens.len + (n_predict - tokens.len) * n_parallel;

    var c = context::Context { .. };
    c.unified = true;
    context::init(c, m, n_kv_req);
    let n_batch = max(n_predict, n_parallel);
    let n_ubatch = min(512, n_batch);
    write_stderr(str("\nmain: n_predict = ", n_predict, ", n_ctx = ", c.n_ctx, ", n_batch = ", n_batch,
                     ", n_parallel = ", n_parallel, ", n_kv_req = ", n_kv_req, "\n"));
    guard n_kv_req <= c.n_ctx else { abort("n_kv_req > n_ctx, the required KV cache size is not big enough"); }

    for s in n_parallel {
        var ch = sampler::Chain { .. };
        var tk = sampler::new_sampler(sampler::KIND_TOP_K);
        tk.k = top_k;
        ch.samplers.push(tk);
        var tp = sampler::new_sampler(sampler::KIND_TOP_P);
        tp.a = top_p;
        ch.samplers.push(tp);
        var te = sampler::new_sampler(sampler::KIND_TEMP_EXT);
        te.a = temp;
        ch.samplers.push(te);
        ch.samplers.push(sampler::seeded(sampler::KIND_DIST, seed));
        chains.push(ch);
    }

    {
        var text: u8[>..] = [];
        text.push('\n');
        for t in tokens { piece(text, t); }
        write_stdout(text);
    }

    // the prompt, once, in all the sequences
    let all: u64 = if n_parallel == 64 { 0xFFFFFFFFFFFFFFFF } else { ((1 as u64) << n_parallel) - 1 };
    var batch: i32[>..] = [];
    var pos: i64[>..] = [];
    var seqs: u64[>..] = [];
    for t, j in tokens { batch.push(t); pos.push(j); seqs.push(all); }
    if model::has_encoder(m) { abort("encoder models: not ported here"); }
    var logits: f32[>..] = [];
    context::decode_seqs(c, m, batch, pos, seqs, logits, [(batch.len - 1) as i32], n_ubatch);

    if n_parallel > 1 { write_stdout(str("\n\nmain: generating ", n_parallel, " sequences ...\n")); }

    var streams: u8[>..] = [];           // each sequence's text, ended by a NUL, rebuilt as they grow
    var lens: i64[>..] = [];
    for s in n_parallel { lens.push(0); streams.push(0); }
    var texts: u8[>..] = [];
    // each live sequence's row in the last logits, -1 once it has finished
    var i_row: i64[>..] = [];
    for s in n_parallel { i_row.push(0); }
    var cands: sampler::Cand[>..] = [];
    let n_vocab = vocab::n_tokens();
    var n_cur = batch.len;
    var n_decode = 0;
    let t_start = clock();
    var outputs: i32[>..] = [];
    while n_cur <= n_predict {
        batch.clear();
        pos.clear();
        seqs.clear();
        outputs.clear();
        var n_rows = 0;
        for s in n_parallel {
            guard i_row[s] >= 0 else { continue; }
            let r = i_row[s];
            let tok = sampler::sample(chains[s], cands, logits[r * n_vocab..(r + 1) * n_vocab]);
            if vocab::is_eog(tok) || n_cur == n_predict {
                i_row[s] = -1;
                write_stdout("\n");
                if n_parallel > 1 { write_stderr(str("main: stream ", s, " finished at n_cur = ", n_cur, "\n")); }
                continue;
            }
            var p: u8[>..] = [];
            piece(p, tok);
            if n_parallel == 1 { write_stdout(p); flush_stdout(); }
            // streams[s] += piece: the texts are kept NUL-ended, one after another
            texts.clear();
            var at = 0;
            for k in n_parallel {
                texts.append(streams[at..at + lens[k]]);
                if k == s { texts.append(p); }
                texts.push(0);
                at += lens[k] + 1;
            }
            lens[s] += p.len;
            streams.clear();
            streams.append(texts);
            i_row[s] = n_rows;
            n_rows++;
            outputs.push(batch.len as i32);
            batch.push(tok);
            pos.push(n_cur);
            seqs.push((1 as u64) << s);
            n_decode++;
        }
        if batch.len == 0 { break; }
        n_cur++;
        context::decode_seqs(c, m, batch, pos, seqs, logits, outputs, n_ubatch);
    }

    if n_parallel > 1 {
        write_stdout("\n");
        var at = 0;
        for s in n_parallel {
            write_stdout(str("sequence ", s, ":\n\n", prompt, streams[at..at + lens[s]], "\n\n"));
            at += lens[s] + 1;
        }
    }
    let dt = clock() - t_start;
    write_stderr(str("main: decoded ", n_decode, " tokens in ", dt, " s, speed: ", n_decode / dt, " t/s\n"));
    write_stdout("\n");
}
