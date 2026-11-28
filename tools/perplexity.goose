// llama.cpp's llama-perplexity, its default mode: the perplexity of a text
// file in chunks of n_ctx tokens, scoring the second half of each chunk.
//   perplexity -m model.gguf -f text.txt [-c n_ctx] [-b n_batch] [--chunks n]
// The arguments are llama-perplexity's (common/arg; -h lists them).
// One sequence per batch: n_batch must not exceed n_ctx (llama.cpp packs
// several chunks into one batch when it does).
// port-of: tools/perplexity/perplexity.cpp
// port-of: tools/perplexity/main.cpp
import std;
import os;
import math;
import ggml from "../ggml";
import ggml_cpu from "../ggml/cpu";
import ggml_shim from "../ggml/shim";
import llama from "../llama";
import llama.vocab;
import llama.model;
import llama.context;
import common from "../common";
import common.download;
import common.console;
import common.arg;

fn fail(msg: const u8[:]) {
    write_stderr(str("error: ", msg, "\n"));
    exit(1);
}

// log_softmax of tokens[i + 1] from one row of logits.
fn log_softmax(logits: const f32[:], tok: i32) -> f64 {
    var max_logit = logits[0];
    for l in logits { max_logit = max(max_logit, l); }
    var sum_exp: f64 = 0.0;
    for l in logits { sum_exp += exp(l - max_logit); }
    return (logits[tok] - max_logit) - log(sum_exp);
}

fn fixed(out: u8[>..]&, v: f64, decimals: i64) { format_flt(out, v, decimals); }

struct Row { i: i32, tok: i32, logits: f32[] }    // i == -1: no more work
struct Score { i: i32, nll: f64 }

thread_fn scorer() {
    loop {
        let r = qget<Row>();
        guard r.i >= 0 else { break; }
        qput(Score { i: r.i, nll: -log_softmax(r.logits, r.tok) });
    }
}

fn main() {
    let args = console::utf8_args();
    arg::params.n_ctx = 512;
    arg::params.escape = false;
    if !arg::parse(args, arg::Example.Perplexity) { exit(1); }
    let p = arg::params;
    let model_path = p.model_path;
    let text = p.prompt;
    let n_ctx = p.n_ctx;
    if n_ctx <= 0 { fail("perplexity tool requires '--ctx-size' > 0"); }
    let n_batch = min(p.n_batch, n_ctx);
    let n_threads = if p.n_threads > 0 { p.n_threads as! i32 } else { ggml::default_threads() };   // postprocess_cpu_params
    let n_chunks = p.n_chunks;

    var m = model::Model { .. };
    model::load(m, model_path);
    let add_bos = vocab::add_bos;
    assert(!vocab::add_eos);

    var tokens: i32[>..] = [];
    let t_tok = clock();
    vocab::tokenize(text, true, false, tokens);
    write_stderr(str("perplexity: tokenization took ", (clock() - t_tok) * 1000.0, " ms\n"));
    if tokens.len < 2 * n_ctx {
        fail(str("you need at least ", 2 * n_ctx, " tokens to evaluate perplexity with a context of ", n_ctx,
                 "; the file tokenizes to only ", tokens.len));
    }

    var c = context::Context { n_threads: n_threads, n_ubatch: min(n_batch, p.n_ubatch), .. };
    context::init(c, m, n_ctx);

    let n_vocab = vocab::n_tokens();
    let n_chunk_max = tokens.len / n_ctx;
    let n_chunk = if n_chunks < 0 { n_chunk_max } else { min(n_chunks, n_chunk_max) };
    let num_batches = (n_ctx + n_batch - 1) / n_batch;
    let first = n_ctx / 2;
    write_stderr(str("perplexity: calculating perplexity over ", n_chunk, " chunks, n_ctx=", n_ctx,
                     ", batch_size=", n_batch, ", n_seq=1\n"));

    var count = 0;
    var nll: f64 = 0.0;
    var nll2: f64 = 0.0;
    var logits: f32[>..] = [];
    var all_logits: f32[>..] = [];
    var outputs: i32[>..] = [];
    var batch: i32[>..] = [];
    var line: u8[>..] = [];
    var nlls: f64[>..] = [];
    var workers: i64[>..] = [];
    for w in max(1, hardware_threads()) { workers.push(thread_spawn(scorer)); }
    let t0 = clock();

    for chunk in n_chunk {
        let start = chunk * n_ctx;
        let end = start + n_ctx;
        context::clear(c);
        all_logits.clear();
        for j in num_batches {
            let batch_start = start + j * n_batch;
            let batch_size = min(end - batch_start, n_batch);
            batch.clear();
            outputs.clear();
            for k in batch_size {
                // the first token of a chunk is BOS where the model wants one
                batch.push(if add_bos && j == 0 && k == 0 { vocab::special_bos_id } else { tokens[batch_start + k] });
                if j * n_batch + k >= first { outputs.push(k as i32); }
            }
            if outputs.len == 0 {
                // llama.cpp asks for no logits here; decode still computes the last
                context::decode(c, m, batch, logits);
            } else {
                context::decode(c, m, batch, logits, outputs);
                all_logits.append(logits);
            }
        }
        if chunk == 0 {
            write_stderr(str("perplexity: ", clock() - t0, " seconds per pass\n"));
        }
        // score tokens first + 1 .. end - 1 from the logits of first .. end - 2
        // on the workers, as llama.cpp does: with a large vocabulary this
        // costs as much as the model; summed in row order
        let n_rows = n_ctx - 1 - first;
        for t in n_rows {
            qput(Row { i: t as i32, tok: tokens[start + first + t + 1], logits: all_logits[t * n_vocab..(t + 1) * n_vocab] });
        }
        nlls.clear();
        nlls.resize(n_rows, 0.0);
        for done in n_rows {
            let s = qget<Score>();
            nlls[s.i] = s.nll;
        }
        for v in nlls {
            nll += v;
            nll2 += v * v;
        }
        count += n_ctx - first - 1;
        line.clear();
        format(line, "[", chunk + 1, "]");
        fixed(line, exp(nll / count), 4);
        line.push(',');
        write_stdout(line);
        flush_stdout();
    }
    write_stdout("\n");
    for w in workers { qput(Row { i: -1, tok: 0, logits: [] }); }
    for w in workers { thread_wait(w); }

    nll2 /= count;
    nll /= count;
    let ppl = exp(nll);
    nll2 -= nll * nll;
    if nll2 > 0.0 {
        nll2 = sqrt(nll2 / (count - 1));
        line.clear();
        format(line, "Final estimate: PPL = ");
        fixed(line, ppl, 4);
        format(line, " +/- ");
        fixed(line, nll2 * ppl, 5);
        format(line, "\n");
        write_stdout(line);
    } else {
        write_stderr("Unexpected negative standard deviation of log(prob)\n");
    }
}
