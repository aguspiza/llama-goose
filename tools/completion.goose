// A subset of llama.cpp's llama-completion: a prompt, then sampled tokens.
//   completion -m model.gguf -p prompt [-n n_predict] [-c n_ctx] [-b n_batch] [-s seed]
//              [--temp t] [--top-k k] [--top-p p] [--min-p p] [--typical p]
//              [--repeat-penalty r] [--repeat-last-n n] [--presence-penalty p]
//              [--frequency-penalty f] [--xtc-probability p] [--xtc-threshold t]
//              [--top-nsigma n] [--dynatemp-range r] [--dynatemp-exp e]
// port-of: tools/completion/completion.cpp
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

fn fail(msg: const u8[:]) {
    write_stderr(str("error: ", msg, "\n"));
    exit(1);
}

fn main() {
    let args = args();
    var model_path: u8[>..] = [];
    var prompt: u8[>..] = [];
    var n_predict = -1;
    var n_ctx = 4096;
    var n_batch = 2048;
    var n_threads: i32 = 0;
    var sp = sampler::Params { .. };

    var i = 1;
    fn value() -> const u8[:] {
        if i + 1 >= args.len { fail(str("missing value for ", args[i])); }
        i += 1;
        return args[i];
    }
    fn int_value() -> i64 {
        let v, ok = parse_int(value());
        if !ok { fail(str("bad number for ", args[i - 1])); }
        return v;
    }
    fn flt_value() -> f32 {
        let v, ok = parse_flt(value());
        if !ok { fail(str("bad number for ", args[i - 1])); }
        return v as f32;
    }
    while i < args.len {
        let a = args[i];
        if a == "-m" || a == "--model" { model_path.clear(); model_path.append(value()); }
        else if a == "-p" || a == "--prompt" { prompt.clear(); prompt.append(value()); }
        else if a == "-n" || a == "--n-predict" { n_predict = int_value(); }
        else if a == "-c" || a == "--ctx-size" { n_ctx = int_value(); }
        else if a == "-b" || a == "--batch-size" { n_batch = int_value(); }
        else if a == "-t" || a == "--threads" { n_threads = int_value() as! i32; }
        else if a == "-s" || a == "--seed" { sp.seed = (int_value() & 0xFFFFFFFF) as! u32; }
        else if a == "--temp" { sp.temp = max(flt_value(), 0.0); sp.user_set |= sampler::USER_TEMP; }
        else if a == "--top-k" { sp.top_k = int_value(); sp.user_set |= sampler::USER_TOP_K; }
        else if a == "--top-p" { sp.top_p = flt_value(); sp.user_set |= sampler::USER_TOP_P; }
        else if a == "--min-p" { sp.min_p = flt_value(); sp.user_set |= sampler::USER_MIN_P; }
        else if a == "--typical" { sp.typ_p = flt_value(); }
        else if a == "--repeat-penalty" { sp.penalty_repeat = flt_value(); sp.user_set |= sampler::USER_PENALTY_REPEAT; }
        else if a == "--repeat-last-n" { sp.penalty_last_n = int_value(); sp.user_set |= sampler::USER_PENALTY_LAST_N; }
        else if a == "--presence-penalty" { sp.penalty_present = flt_value(); }
        else if a == "--frequency-penalty" { sp.penalty_freq = flt_value(); }
        else if a == "--xtc-probability" { sp.xtc_probability = flt_value(); sp.user_set |= sampler::USER_XTC_PROBABILITY; }
        else if a == "--xtc-threshold" { sp.xtc_threshold = flt_value(); sp.user_set |= sampler::USER_XTC_THRESHOLD; }
        else if a == "--top-nsigma" { sp.top_n_sigma = flt_value(); }
        else if a == "--dynatemp-range" { sp.dynatemp_range = flt_value(); }
        else if a == "--dynatemp-exp" { sp.dynatemp_exponent = flt_value(); }
        else if a == "-no-cnv" || a == "--no-conversation" { }
        else { fail(str("unknown argument: ", a)); }
        i += 1;
    }
    if model_path.len == 0 { fail("no model (-m)"); }

    var m = model::Model { .. };
    model::load(m, model_path);

    var tokens: i32[>..] = [];
    vocab::tokenize(prompt, true, true, tokens);
    if tokens.len == 0 { fail("empty prompt"); }

    var c = context::Context { n_threads: n_threads, .. };
    context::init(c, m, n_ctx);
    if tokens.len > c.n_ctx - 4 { fail("the prompt is too long for the context"); }

    sampler::apply_model_defaults(sp, m.meta);
    var ch = sampler::Chain { .. };
    sampler::init(ch, sp);
    // the prompt counts for the penalties, as common_sampler accepts it
    for t in tokens { sampler::accept(ch, t); }

    {
        var text: u8[>..] = [];
        for t in tokens { vocab::token_to_piece(t, text, 0, false); }
        write_stdout(text);
        flush_stdout();
    }

    let t_start = clock();
    var logits: f32[>..] = [];
    var cands: sampler::Cand[>..] = [];
    // the prompt, in batches of n_batch
    var start = 0;
    while start < tokens.len {
        let end = min(start + n_batch, tokens.len);
        context::decode(c, m, tokens[start..end], logits);
        start = end;
    }
    var n_decode = 0;
    loop {
        if n_predict >= 0 && n_decode >= n_predict { break; }
        if c.n_past >= c.n_ctx { break; }
        let t = sampler::sample(ch, cands, logits);
        if vocab::is_eog(t) { write_stdout(" [end of text]\n"); break; }
        var text: u8[>..] = [];
        vocab::token_to_piece(t, text, 0, false);
        write_stdout(text);
        flush_stdout();
        n_decode += 1;
        context::decode(c, m, [t], logits);
    }
    write_stdout("\n\n");
    let dt = clock() - t_start;
    write_stderr(str("completion: decoded ", n_decode, " tokens in ", dt, " s, speed: ", n_decode / dt, " t/s\n"));
}
