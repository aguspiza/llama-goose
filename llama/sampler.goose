// Sampling: the chain llama.cpp's common_sampler builds, and its grammar.
// A given seed draws the same tokens as llama.cpp: the RNG is std::mt19937
// and the uniform draws follow libstdc++'s generate_canonical.
// Not ported yet: adaptive-p, infill, lazy grammars.
// Sorting is stable, by logit; std::partial_sort leaves tokens with exactly
// equal logits in an order of its own, so only those may come out in
// another order.
// port-of: src/llama-sampler.cpp common/sampling.cpp
import std;
import ggml from "../ggml";
import ggml.gguf;
import math;
import os;
import .vocab;
import .grammar;
namespace sampler;

let DEFAULT_SEED: u32 = 0xFFFFFFFF;

//
// std::mt19937
//

struct Mt19937 { mt: u32[624], index: i64 }

fn mt_seed(r: Mt19937&, seed: u32) {
    r.mt[0] = seed;
    for i in 1..624 {
        let prev = r.mt[i - 1];
        r.mt[i] = 1812433253 * (prev ^ (prev >> 30)) + i as u32;
    }
    r.index = 624;
}

fn mt_next(r: Mt19937&) -> u32 {
    if r.index >= 624 {
        for i in 624 {
            let y = (r.mt[i] & 0x80000000) | (r.mt[(i + 1) % 624] & 0x7FFFFFFF);
            var v = r.mt[(i + 397) % 624] ^ (y >> 1);
            if y & 1 != 0 { v ^= 0x9908B0DF; }
            r.mt[i] = v;
        }
        r.index = 0;
    }
    var y = r.mt[r.index];
    r.index += 1;
    y ^= y >> 11;
    y ^= (y << 7) & 0x9D2C5680;
    y ^= (y << 15) & 0xEFC60000;
    y ^= y >> 18;
    return y;
}

// std::uniform_real_distribution<double>(0, 1): generate_canonical<double, 53>
// takes two 32-bit draws.
fn uniform_f64(r: Mt19937&) -> f64 {
    let lo: f64 = mt_next(r);
    let hi: f64 = mt_next(r);
    let ret = (lo + hi * 4294967296.0) / 18446744073709551616.0;
    return if ret >= 1.0 { 0.9999999999999999 } else { ret };
}

// std::uniform_real_distribution<float>(0, 1): one draw.
fn uniform_f32(r: Mt19937&) -> f32 {
    let g: f32 = mt_next(r);
    let ret = g / 4294967296.0;
    return if ret >= 1.0 { 0.99999994 } else { ret };
}

// get_rng_seed: the default seed means "pick one".
fn rng_seed(seed: u32) -> u32 {
    if seed != DEFAULT_SEED { return seed; }
    return (random_seed() & 0xFFFFFFFF) as! u32;
}

//
// candidates
//

struct Cand { id: i32, logit: f32, p: f32 }

// The candidates: cands[start..start + size], as llama_token_data_array.
struct Cands { start: i64, size: i64, selected: i64, sorted: bool }

let NEG_INF: f32 = -1.0 / 0.0;

fn view(c: Cand[>..]&, a: Cands) -> Cand[:] { c[a.start..a.start + a.size] }

// Sorts by logit, highest first (the partial sorts of llama.cpp, in full).
fn sort_desc(xs: Cand[:]) { stable_sort(xs) { a, b => a.logit > b.logit }; }

fn temp_impl(xs: Cand[:], temp: f32) {
    guard xs.len > 0 else { return; }
    if temp <= 0.0 {
        // keep the token with the highest logit; the rest become -inf
        var max_i = 0;
        var max_l = xs[0].logit;
        for i in 1..xs.len {
            if xs[i].logit > max_l {
                xs[max_i].logit = NEG_INF;
                max_i = i;
                max_l = xs[i].logit;
            } else {
                xs[i].logit = NEG_INF;
            }
        }
        return;
    }
    for &x in xs { x.logit /= temp; }
}

fn softmax_impl(c: Cand[>..]&, a: Cands&, do_sort: bool) {
    assert(a.size > 0);
    if do_sort && !a.sorted {
        sort_desc(view(c, a));
        a.sorted = true;
    }
    let xs = view(c, a);
    var max_l = xs[0].logit;
    if !a.sorted { for x in xs { max_l = max(max_l, x.logit); } }
    var cum_sum: f32 = 0.0;
    for &x in xs {
        let p = expf(x.logit - max_l);
        x.p = p;
        cum_sum += p;
    }
    for &x in xs { x.p /= cum_sum; }
}

fn expf(x: f32) -> f32 { exp(x) }
fn logf(x: f32) -> f32 { log(x) }

//
// the samplers
//

let KIND_LOGIT_BIAS = 0;
let KIND_PENALTIES = 1;
let KIND_TOP_N_SIGMA = 2;
let KIND_TOP_K = 3;
let KIND_TYPICAL = 4;
let KIND_TOP_P = 5;
let KIND_MIN_P = 6;
let KIND_XTC = 7;
let KIND_TEMP_EXT = 8;
let KIND_GREEDY = 9;
let KIND_DIST = 10;
let KIND_DRY = 11;
let KIND_INFILL = 12;
let KIND_ADAPTIVE_P = 13;
let KIND_MIROSTAT = 14;
let KIND_MIROSTAT_V2 = 15;

// One sampler of the chain, with its parameters (a, b, c, k, n by kind), its
// RNG and mirostat's running target mu.
struct Sampler { kind: i64, a: f32, b: f32, c: f32, k: i64, min_keep: i64, rng: Mt19937, seed: u32, n: i64, mu: f32 }

// The parameters common_params_sampling gives, with its defaults.
struct Params {
    seed: u32 = 0xFFFFFFFF,
    n_probs: i64 = 0,
    min_keep: i64 = 0,
    top_k: i64 = 40,
    top_p: f32 = 0.95,
    min_p: f32 = 0.05,
    xtc_probability: f32 = 0.0,
    xtc_threshold: f32 = 0.10,
    typ_p: f32 = 1.0,
    temp: f32 = 0.80,
    dynatemp_range: f32 = 0.0,
    dynatemp_exponent: f32 = 1.0,
    penalty_last_n: i64 = 64,
    penalty_repeat: f32 = 1.0,
    penalty_freq: f32 = 0.0,
    penalty_present: f32 = 0.0,
    dry_multiplier: f32 = 0.0,
    dry_base: f32 = 1.75,
    dry_allowed_length: i64 = 2,
    dry_penalty_last_n: i64 = 64,
    // the sequence breakers, each ended by a NUL: newline, colon, quote, star
    dry_breakers: u8[..256] = [10, 0, 58, 0, 34, 0, 42, 0],
    mirostat: i64 = 0,
    mirostat_tau: f32 = 5.0,
    mirostat_eta: f32 = 0.10,
    top_n_sigma: f32 = -1.0,
    // the configurable part of the chain, in order (common_sampler_type)
    order: i64[..16] = [KIND_PENALTIES, KIND_DRY, KIND_TOP_N_SIGMA, KIND_TOP_K, KIND_TYPICAL,
                        KIND_TOP_P, KIND_MIN_P, KIND_XTC, KIND_TEMP_EXT],
    // the parameters the user gave (USER_* bits): the model's own
    // recommendations (general.sampling.*) do not override them
    user_set: u64 = 0,
}

let USER_SAMPLERS: u64 = 1;
let USER_TOP_K: u64 = 2;
let USER_TOP_P: u64 = 4;
let USER_MIN_P: u64 = 8;
let USER_XTC_PROBABILITY: u64 = 16;
let USER_XTC_THRESHOLD: u64 = 32;
let USER_TEMP: u64 = 64;
let USER_PENALTY_LAST_N: u64 = 128;
let USER_PENALTY_REPEAT: u64 = 256;
let USER_MIROSTAT: u64 = 512;
let USER_MIROSTAT_TAU: u64 = 1024;
let USER_MIROSTAT_ETA: u64 = 2048;

// A metadata value as llama_model_meta_val_str gives it (gguf_kv_to_str),
// for the numeric types and strings; false if the key is absent.
fn meta_str(meta: gguf::Ctx, key: const u8[:], out: u8[>..]&) -> bool {
    let id = gguf::find_key(meta, key);
    guard id >= 0 else { return false; }
    out.clear();
    match gguf::get_kv_type(meta, id) {
        gguf::UINT8 => format(out, gguf::get_val_u8(meta, id)),
        gguf::INT8 => format(out, gguf::get_val_i8(meta, id)),
        gguf::UINT16 => format(out, gguf::get_val_u16(meta, id)),
        gguf::INT16 => format(out, gguf::get_val_i16(meta, id)),
        gguf::UINT32 => format(out, gguf::get_val_u32(meta, id)),
        gguf::INT32 => format(out, gguf::get_val_i32(meta, id)),
        gguf::UINT64 => format(out, gguf::get_val_u64(meta, id)),
        gguf::INT64 => format(out, gguf::get_val_i64(meta, id)),
        gguf::FLOAT32 => format_flt(out, gguf::get_val_f32(meta, id), 6),     // std::to_string
        gguf::FLOAT64 => format_flt(out, gguf::get_val_f64(meta, id), 6),
        gguf::BOOL => out.append(if gguf::get_val_bool(meta, id) { "true" } else { "false" }),
        gguf::STRING => gguf::get_val_str(meta, id, out),
        _ => { return false; },
    }
    return true;
}

// The model's recommended sampling (common_params_sampling_apply_model_defaults).
fn apply_model_defaults(p: Params&, meta: gguf::Ctx) {
    var buf: u8[>..] = [];
    fn get_int(key: const u8[:], dst: i64&, bit: u64) {
        guard p.user_set & bit == 0 && meta_str(meta, key, buf) else { return; }
        let v, ok = parse_int(trim(buf));
        if ok { dst = v; }
    }
    fn get_float(key: const u8[:], dst: f32&, bit: u64) {
        guard p.user_set & bit == 0 && meta_str(meta, key, buf) else { return; }
        let v, ok = parse_flt(trim(buf));
        if ok { dst = v as f32; }
    }
    if p.user_set & USER_SAMPLERS == 0 && meta_str(meta, "general.sampling.sequence", buf) {
        var order: i64[..16] = [];
        each_split(buf, ';') { if order.len < order.cap { let k = kind_from_name(trim(it)); if k >= 0 { order.push(k); } } };
        if order.len > 0 { p.order = order; }
    }
    get_int("general.sampling.top_k", p.top_k, USER_TOP_K);
    get_float("general.sampling.top_p", p.top_p, USER_TOP_P);
    get_float("general.sampling.min_p", p.min_p, USER_MIN_P);
    get_float("general.sampling.xtc_probability", p.xtc_probability, USER_XTC_PROBABILITY);
    get_float("general.sampling.xtc_threshold", p.xtc_threshold, USER_XTC_THRESHOLD);
    get_float("general.sampling.temp", p.temp, USER_TEMP);
    get_int("general.sampling.penalty_last_n", p.penalty_last_n, USER_PENALTY_LAST_N);
    get_float("general.sampling.penalty_repeat", p.penalty_repeat, USER_PENALTY_REPEAT);
    get_int("general.sampling.mirostat", p.mirostat, USER_MIROSTAT);
    get_float("general.sampling.mirostat_tau", p.mirostat_tau, USER_MIROSTAT_TAU);
    get_float("general.sampling.mirostat_eta", p.mirostat_eta, USER_MIROSTAT_ETA);
}

// common_sampler_types_from_names, with its alternative names; -1 if unknown.
fn kind_from_name(name: const u8[:]) -> i64 {
    if name == "dry" { return KIND_DRY; }
    if name == "top_k" || name == "top-k" { return KIND_TOP_K; }
    if name == "top_p" || name == "top-p" || name == "nucleus" { return KIND_TOP_P; }
    if name == "top_n_sigma" || name == "top-n-sigma" || name == "top-nsigma" { return KIND_TOP_N_SIGMA; }
    if name == "min_p" || name == "min-p" { return KIND_MIN_P; }
    if name == "xtc" { return KIND_XTC; }
    if name == "typ_p" || name == "typical_p" || name == "typical" || name == "typ-p" || name == "typical-p" { return KIND_TYPICAL; }
    if name == "temperature" || name == "temp" { return KIND_TEMP_EXT; }
    if name == "penalties" { return KIND_PENALTIES; }
    if name == "infill" { return KIND_INFILL; }
    if name == "adaptive_p" || name == "adaptive-p" { return KIND_ADAPTIVE_P; }
    return -1;
}

struct Count { id: i32, n: i32 }

// The chain and its state: the penalties' window of recent tokens, and how
// often each appears in it (sorted by token).
struct Chain {
    samplers: Sampler[..16],
    prev: i32[..4096],           // the last penalty_last_n accepted tokens, oldest first
    counts: Count[..4096],
    dry_last: i32[..4096],       // the last dry_penalty_last_n accepted tokens, oldest first
    has_grammar: bool = false,   // the tokens are checked against the grammar (llama/grammar.goose)
}

// Constrains the chain's tokens to a GBNF grammar, whose start symbol is
// root: false, and grammar::err, if it does not parse.
fn set_grammar(ch: Chain&, text: const u8[:], root: const u8[:]) -> bool {
    ch.has_grammar = grammar::init(text, root, true);
    return ch.has_grammar;
}

// DRY's sequence breakers, processed as llama.cpp's multimap: a token that
// starts one, and the tokens (dry_tails[start..start + len]) that must
// follow it for the breaker to be complete; sorted by token. There is one
// chain per program, so they live here.
struct DryHead { token: i32, start: i64, len: i64 }
var dry_heads: DryHead[>..] = [];
var dry_tails: i32[>..] = [];

// The DRY breakers' head tokens and tails (llama_sampler_init_dry's
// get_overlapping_token_sequences): every token whose text holds a breaker
// starts one with no tail; a token whose text ends with the start of a
// breaker starts one whose tail is the rest of the breaker, tokenized (at
// most 20 tokens).
fn dry_process_breakers(breakers: const u8[:]) {
    dry_heads.clear();
    dry_tails.clear();
    let n_vocab = vocab::n_tokens();
    // the text of each token, once
    var words: u8[>..] = [];
    var ends: i64[>..] = [];
    for t in n_vocab {
        vocab::detokenize([t as i32], false, true, words);
        ends.push(words.len);
    }
    var seen: u8[>..] = [];
    var heads: DryHead[>..] = [];
    var start = 0;
    var n_breakers = 0;
    for c, i in breakers {
        guard c == 0 else { continue; }
        let brk = breakers[start..min(i, start + 40)];
        start = i + 1;
        guard brk.len > 0 else { continue; }
        // each distinct breaker once, at most 128 of them
        var dup = false;
        each_split(seen, 0) { if it == brk { dup = true; } };
        guard !dup else { continue; }
        seen.append(brk);
        seen.push(0);
        n_breakers += 1;
        guard n_breakers <= 128 else { break; }
        // the tails, by how much of the breaker the token's end matched
        var tail_start: i64[41] = [-1; 41];
        var tail_len: i64[41] = [0; 41];
        var w0 = 0;
        for t in n_vocab {
            let word = words[w0..ends[t]];
            w0 = ends[t];
            if find(word, brk) >= 0 {
                heads.push(DryHead { token: t as i32, start: 0, len: 0 });
                continue;
            }
            for pos in word.len {
                guard word[pos] == brk[0] else { continue; }
                var m = 1;
                var ok = true;
                while m < brk.len && m + pos < word.len {
                    if word[pos + m] != brk[m] { ok = false; break; }
                    m += 1;
                }
                guard ok else { continue; }
                if tail_start[m] < 0 {
                    var ids: i32[>..] = [];
                    vocab::tokenize(brk[m..], false, false, ids);
                    tail_start[m] = dry_tails.len;
                    tail_len[m] = min(ids.len, 20);
                    dry_tails.append(ids[..tail_len[m]]);
                }
                // not twice the same tail for a token
                var have = false;
                for h in heads {
                    if h.token == t as i32 && h.len == tail_len[m] && dry_tails[h.start..h.start + h.len] == dry_tails[tail_start[m]..tail_start[m] + tail_len[m]] { have = true; break; }
                }
                if !have { heads.push(DryHead { token: t as i32, start: tail_start[m], len: tail_len[m] }); }
            }
        }
    }
    stable_sort(heads) { a, b => a.token < b.token };
    dry_heads.append(heads);
}

// The first entry of dry_heads for token, or dry_heads.len.
fn dry_first(token: i32) -> i64 { lower_bound(dry_heads) { it.token < token } }

fn count_of(ch: Chain&, id: i32) -> i32 {
    let i = lower_bound(ch.counts) { it.id < id };
    return if i < ch.counts.len && ch.counts[i].id == id { ch.counts[i].n } else { 0 };
}

fn count_add(ch: Chain&, id: i32, d: i32) {
    let i = lower_bound(ch.counts) { it.id < id };
    if i < ch.counts.len && ch.counts[i].id == id {
        ch.counts[i].n += d;
        if ch.counts[i].n == 0 {
            for k in i..ch.counts.len - 1 { ch.counts[k] = ch.counts[k + 1]; }
            ch.counts.pop();
        }
        return;
    }
    assert(d > 0);
    ch.counts.push(Count { id: id, n: 0 });
    var k = ch.counts.len - 1;
    while k > i { ch.counts[k] = ch.counts[k - 1]; k -= 1; }
    ch.counts[i] = Count { id: id, n: d };
}

fn new_sampler(kind: i64) -> Sampler { Sampler { kind: kind, rng: Mt19937 { mt: [0; 624], index: 624 }, .. } }

fn seeded(kind: i64, seed: u32) -> Sampler {
    var s = new_sampler(kind);
    s.seed = seed;
    mt_seed(s.rng, rng_seed(seed));
    return s;
}

// common_sampler_init, without grammars.
fn init(ch: Chain&, p: Params) {
    ch.samplers.clear();
    ch.prev.clear();
    ch.counts.clear();
    // the model's suppressed tokens, as a logit bias of -inf
    if vocab::suppress_tokens.len > 0 { ch.samplers.push(new_sampler(KIND_LOGIT_BIAS)); }
    ch.dry_last.clear();
    guard p.penalty_last_n <= 4096 else { abort("penalty_last_n above 4096: not supported"); }
    if p.mirostat == 1 || p.mirostat == 2 {
        // the temperature, then mirostat, which draws the token
        var t = new_sampler(KIND_TEMP_EXT);
        t.a = p.temp;
        ch.samplers.push(t);
        var m = seeded(if p.mirostat == 1 { KIND_MIROSTAT } else { KIND_MIROSTAT_V2 }, p.seed);
        m.a = p.mirostat_tau;
        m.b = p.mirostat_eta;
        m.mu = 2.0 * p.mirostat_tau;
        m.k = 100;
        m.n = vocab::n_tokens();
        ch.samplers.push(m);
        return;
    }
    guard p.mirostat == 0 else { abort("unknown mirostat version"); }
    for kind in p.order {
        var s = new_sampler(kind);
        match kind {
            KIND_DRY => {
                // a DRY sampler that cannot penalize anything is left out
                let last_n = max(p.dry_penalty_last_n, 0);
                guard p.dry_multiplier != 0.0 && p.dry_base >= 1.0 && last_n != 0 else { continue; }
                guard last_n <= 4096 else { abort("dry_penalty_last_n above 4096: not supported"); }
                s.a = p.dry_multiplier; s.b = p.dry_base; s.k = p.dry_allowed_length; s.n = last_n;
                dry_process_breakers(p.dry_breakers);
            },
            KIND_PENALTIES => {
                s.k = if p.penalty_last_n < 0 { 0 } else { p.penalty_last_n };
                s.a = p.penalty_repeat; s.b = p.penalty_freq; s.c = p.penalty_present;
            },
            KIND_TOP_N_SIGMA => { s.a = p.top_n_sigma; },
            KIND_TOP_K => { s.k = p.top_k; },
            KIND_TYPICAL => { s.a = p.typ_p; s.min_keep = p.min_keep; },
            KIND_TOP_P => { s.a = p.top_p; s.min_keep = p.min_keep; },
            KIND_MIN_P => { s.a = p.min_p; s.min_keep = p.min_keep; },
            KIND_XTC => {
                s = seeded(KIND_XTC, p.seed);
                s.a = p.xtc_probability; s.b = p.xtc_threshold; s.min_keep = p.min_keep;
            },
            KIND_TEMP_EXT => { s.a = p.temp; s.b = p.dynatemp_range; s.c = p.dynatemp_exponent; },
            _ => { abort("infill and adaptive-p sampling: not ported yet"); },
        }
        ch.samplers.push(s);
    }
    // greedy when the chain ends in temp 0 or top-k 1 (and no probabilities are asked for)
    let last = if p.order.len > 0 { p.order[p.order.len - 1] } else { -1 };
    let greedy = p.n_probs == 0 && p.order.len > 0 &&
                 ((last == KIND_TEMP_EXT && p.temp == 0.0 && p.dynatemp_range == 0.0) ||
                  (last == KIND_TOP_K && p.top_k == 1));
    ch.samplers.push(if greedy { new_sampler(KIND_GREEDY) } else { seeded(KIND_DIST, p.seed) });
}

fn penalties_disabled(s: Sampler&) -> bool { s.k == 0 || (s.a == 1.0 && s.b == 0.0 && s.c == 0.0) }

fn apply(ch: Chain&, c: Cand[>..]&, s: Sampler&, a: Cands&) {
    match s.kind {
        KIND_LOGIT_BIAS => {
            for t in vocab::suppress_tokens {
                if t >= 0 && a.size > t && c[a.start + t].id == t {
                    c[a.start + t].logit += NEG_INF;
                } else {
                    for &x in view(c, a) { if x.id == t { x.logit += NEG_INF; break; } }
                }
            }
        },
        KIND_PENALTIES => {
            guard !penalties_disabled(s) else { return; }
            for &x in view(c, a) {
                let count = count_of(ch, x.id);
                guard count > 0 else { continue; }
                if x.logit <= 0.0 { x.logit *= s.a; } else { x.logit /= s.a; }
                x.logit -= (count as f32) * s.b + (if count > 0 { 1.0 } else { 0.0 }) * s.c;
            }
            a.sorted = false;
        },
        KIND_TOP_N_SIGMA => {
            guard s.a > 0.0 && a.size > 1 else { return; }
            let xs = view(c, a);
            var mx = xs[0].logit;
            var sum: f32 = 0.0;
            var valid = 0;
            for x in xs {
                if x.logit != NEG_INF { mx = max(mx, x.logit); sum += x.logit; valid += 1; }
            }
            let mean: f32 = if valid > 0 { sum / valid } else { 0.0 };
            var acc: f32 = 0.0;
            for x in xs {
                // pow(float, int) is a double, and float += double adds in double
                if x.logit != NEG_INF { acc = (acc as f64 + pow((x.logit - mean) as f64, 2.0)) as f32; }
            }
            let sd: f32 = if valid > 0 { sqrt(acc / valid) } else { 0.0 };
            for &x in xs { if x.logit < mx - s.a * sd { x.logit = NEG_INF; } }
        },
        KIND_TOP_K => {
            guard s.k > 0 else { return; }
            let k = min(s.k, a.size);
            if !a.sorted { sort_desc(view(c, a)); a.sorted = true; }
            a.size = k;
        },
        KIND_TYPICAL => {
            guard s.a < 1.0 else { return; }
            softmax_impl(c, a, true);
            let xs = view(c, a);
            var entropy: f32 = 0.0;
            for x in xs { entropy += -x.p * logf(x.p); }
            var order: i64[>..] = [];
            var shifted: f32[>..] = [];
            for x, i in xs { order.push(i); shifted.push(abs(-logf(x.p) - entropy)); }
            stable_sort(order) { i, j => shifted[i] < shifted[j] };
            var cum_sum: f32 = 0.0;
            var last_idx = order.len;
            for idx, i in order {
                cum_sum += xs[idx].p;
                if cum_sum > s.a && (s.min_keep == 0 || i >= s.min_keep - 1) { last_idx = i + 1; break; }
            }
            var kept: Cand[>..] = [];
            for i in last_idx { kept.push(xs[order[i]]); }
            for x, i in kept { xs[i] = x; }
            a.size = kept.len;
            a.sorted = false;
        },
        KIND_TOP_P => {
            guard s.a < 1.0 else { return; }
            softmax_impl(c, a, false);
            if !a.sorted { sort_desc(view(c, a)); a.sorted = true; }
            let xs = view(c, a);
            var cum_sum: f32 = 0.0;
            var last_idx = a.size;
            for x, i in xs {
                cum_sum += x.p;
                if cum_sum >= s.a && i + 1 >= s.min_keep { last_idx = i + 1; break; }
            }
            a.size = last_idx;
        },
        KIND_MIN_P => {
            guard s.a > 0.0 && a.size > 0 else { return; }
            var applied = false;
            if !a.sorted {
                let xs = view(c, a);
                var max_logit: f32 = -3.4028234663852886e38;
                for x in xs { max_logit = max(max_logit, x.logit); }
                let min_logit = max_logit + logf(s.a);
                var kept = 0;
                for x in xs { if x.logit >= min_logit { kept += 1; } }
                if kept > 0 && kept >= s.min_keep {
                    var n = 0;
                    for i in xs.len { if xs[i].logit >= min_logit { xs[n] = xs[i]; n += 1; } }
                    a.size = n;
                    applied = true;
                }
            }
            if !applied {
                if !a.sorted { sort_desc(view(c, a)); a.sorted = true; }
                let xs = view(c, a);
                let min_logit = xs[0].logit + logf(s.a);
                var i = 1;
                while i < xs.len {
                    if xs[i].logit < min_logit && i >= s.min_keep { break; }
                    i += 1;
                }
                a.size = i;
            }
        },
        KIND_XTC => {
            guard s.a > 0.0 && s.b <= 0.5 && a.size >= 2 else { return; }
            let chance = uniform_f32(s.rng);
            guard chance <= s.a else { return; }
            softmax_impl(c, a, true);
            let xs = view(c, a);
            var pos_last = 0;
            for x, i in xs { if x.p >= s.b { pos_last = i; } else { break; } }
            if a.size - pos_last >= s.min_keep && pos_last > 0 {
                a.start += pos_last;
                a.size -= pos_last;
            }
        },
        KIND_TEMP_EXT => {
            let temp = s.a;
            let delta = s.b;
            if delta > 0.0 {
                let min_temp = max(0.0, temp - delta);
                let max_temp = temp + delta;
                guard a.size > 1 else { return; }
                let max_entropy = -logf(1.0 / (a.size as f32));
                softmax_impl(c, a, true);
                var entropy: f32 = 0.0;
                for x in view(c, a) { if x.p > 0.0 { entropy -= x.p * logf(x.p); } }
                let normalized = entropy / max_entropy;
                let dyn_temp: f32 = min_temp + (max_temp - min_temp) * pow(normalized, s.c);
                temp_impl(view(c, a), dyn_temp);
                let xs = view(c, a);
                let max_l: f64 = xs[0].logit;
                var cum: f64 = 0.0;
                for &x in xs { let p = exp(x.logit - max_l); x.p = p as f32; cum += p; }
                for &x in xs { x.p = (x.p / cum) as f32; }
            } else {
                temp_impl(view(c, a), temp);
            }
        },
        KIND_DRY => { dry_apply(ch, c, s, a); },
        KIND_MIROSTAT => {
            softmax_impl(c, a, true);
            // the exponent of the Zipf law the top m tokens follow
            let xs = view(c, a);
            var sum_ti_bi: f32 = 0.0;
            var sum_ti_sq: f32 = 0.0;
            var i = 0;
            while i < s.k - 1 && i < a.size - 1 {
                let t_i = logf(((i + 2) as f32) / ((i + 1) as f32));
                let b_i = logf(xs[i].p / xs[i + 1].p);
                sum_ti_bi += t_i * b_i;
                sum_ti_sq += t_i * t_i;
                i += 1;
            }
            let s_hat = sum_ti_bi / sum_ti_sq;
            // k from s_hat and the target surprise
            let epsilon_hat = s_hat - 1.0;
            let n_vocab: f32 = s.n;
            let two: f32 = 2.0;
            let kf: f32 = pow((epsilon_hat * pow(two, s.mu)) / (1.0 - pow(n_vocab, -epsilon_hat)), 1.0 / s_hat);
            // int(k), as x86 converts: what is out of range becomes INT_MIN
            let ki: i64 = if kf != kf || kf >= 2147483648.0 || kf < -2147483648.0 { -2147483648 } else { kf as! i32 };
            let kk = min(max(ki, 1), a.size);
            if !a.sorted { sort_desc(view(c, a)); a.sorted = true; }
            a.size = kk;
            softmax_impl(c, a, true);
            mirostat_draw(c, s, a);
        },
        KIND_MIROSTAT_V2 => {
            softmax_impl(c, a, true);
            // the tokens whose surprise is above mu go
            let xs = view(c, a);
            var n = 0;
            while n < xs.len && !(-log2(xs[n].p) > s.mu) { n += 1; }
            a.size = max(n, 1);
            softmax_impl(c, a, true);
            mirostat_draw(c, s, a);
        },
        KIND_GREEDY => {
            let xs = view(c, a);
            a.selected = 0;
            for i in 1..xs.len { if xs[i].logit > xs[a.selected].logit { a.selected = i; } }
        },
        _ => {     // KIND_DIST
            if a.size == 0 { a.selected = -1; return; }
            a.selected = 0;
            let xs = view(c, a);
            if a.size == 1 {
                uniform_f64(s.rng);       // keep the RNG aligned with backend sampling
                xs[0].p = 1.0;
                return;
            }
            var max_l = xs[0].logit;
            if !a.sorted { for x in xs { max_l = max(max_l, x.logit); } }
            var sum_cum: f64 = 0.0;
            for &x in xs {
                let p = expf(x.logit - max_l);
                x.p = p;
                sum_cum += p;
            }
            let rnd = uniform_f64(s.rng);
            var sum_run: f64 = 0.0;
            let sum_tgt = sum_cum * rnd;
            var found = false;
            for &x, i in xs {
                if !found {
                    sum_run += x.p;
                    if sum_run >= sum_tgt { a.selected = i; found = true; }
                }
                x.p = (x.p / sum_cum) as f32;
            }
            if !found { a.selected = a.size - 1; }
        },
    }
}

// Mirostat's draw (llama_sample_dist: std::discrete_distribution over the
// probabilities), and its update of mu from the surprise of the token drawn.
fn mirostat_draw(c: Cand[>..]&, s: Sampler&, a: Cands&) {
    let xs = view(c, a);
    a.selected = discrete(xs, s.rng);
    let observed: f32 = -log2(xs[a.selected].p);
    let e = observed - s.a;
    s.mu = s.mu - s.b * e;
}

// std::discrete_distribution<int>(p...)(rng), as libstdc++ has it: the
// weights normalized in double, and the first cumulative sum not below a
// uniform draw (the last one being 1); with fewer than two weights, 0 and
// no draw.
fn discrete(xs: const Cand[:], rng: Mt19937&) -> i64 {
    guard xs.len >= 2 else { return 0; }
    var sum: f64 = 0.0;
    for x in xs { sum += x.p; }
    let u = uniform_f64(rng);
    var cum: f64 = 0.0;
    for x, i in xs {
        cum += x.p / sum;
        let cp = if i == xs.len - 1 { 1.0 } else { cum };
        if cp >= u { return i; }
    }
    return xs.len - 1;
}

// DRY ("don't repeat yourself"): a token that would extend a sequence
// repeated from earlier in the window, by more than dry_allowed_length
// tokens, is penalized by multiplier * base^(extra length); the sequence
// breakers end what counts as one sequence. llama_sampler_dry_apply.
fn dry_apply(ch: Chain&, c: Cand[>..]&, s: Sampler&, a: Cands&) {
    let n_last = ch.dry_last.len;
    let last_n_repeat = min(n_last, s.n);
    guard last_n_repeat > s.k else { return; }
    // the i-th token from the end
    fn rat(i: i64) -> i32 { ch.dry_last[n_last - 1 - i] }

    // 1: the last breaker limits how long a repeat can be
    var rep_limit = last_n_repeat;
    for i in last_n_repeat {
        let token = rat(i);
        var h = dry_first(token);
        guard h < dry_heads.len && dry_heads[h].token == token else { continue; }
        var longest = -1;
        while h < dry_heads.len && dry_heads[h].token == token {
            let e = dry_heads[h];
            h += 1;
            guard e.len > longest && e.len <= i else { continue; }
            var ok = true;
            for off in e.len {
                if dry_tails[e.start + off] != rat(i - off - 1) { ok = false; break; }
            }
            if ok { longest = e.len; }
        }
        if longest >= 0 {
            rep_limit = i - longest;
            break;
        }
    }
    guard rep_limit >= s.k else { return; }

    // 2: for each position, how long a suffix of the window repeats there
    // (the Z-algorithm, backwards)
    var repeat_count: i64[>..] = [];
    for i in last_n_repeat { repeat_count.push(0); }
    let last = last_n_repeat - 1;
    var rt = 0;
    var lt = 0;
    for k in 1..last_n_repeat {
        if k > rt {
            var n = 0;
            while n + k < last_n_repeat && rat(n) == rat(n + k) { n += 1; }
            repeat_count[last - k] = min(n, rep_limit);
            if n > 0 { lt = k; rt = k + n - 1; }
        } else {
            let p = k - lt;
            let right_part_len = rt - k + 1;
            if repeat_count[last - p] < right_part_len {
                repeat_count[last - k] = min(repeat_count[last - p], rep_limit);
            } else {
                var i = rt + 1;
                while i < last_n_repeat && rat(i) == rat(i - k) { i += 1; }
                repeat_count[last - k] = min(i - k, rep_limit);
                lt = k;
                rt = i - 1;
            }
        }
    }

    // 3: the longest repeat each next token would extend
    var max_repeat: Count[>..] = [];       // by token
    for i in last_n_repeat - 1 {
        let len = repeat_count[i];
        guard len >= s.k else { continue; }
        let token = rat(last_n_repeat - 2 - i);
        let j = lower_bound(max_repeat) { it.id < token };
        if j < max_repeat.len && max_repeat[j].id == token {
            if max_repeat[j].n < len { max_repeat[j].n = len as i32; }
        } else {
            max_repeat.push(Count { id: token, n: len as i32 });
            var q = max_repeat.len - 1;
            while q > j { max_repeat[q] = max_repeat[q - 1]; q -= 1; }
            max_repeat[j] = Count { id: token, n: len as i32 };
        }
    }

    // 4: the penalties, the exponent clamped so that base^exp stays a float
    let float_max_log: f32 = 88.7228391;
    var max_exponent = 0;
    if s.b > 1.000001 { max_exponent = (float_max_log / logf(s.b)) as! i32; }
    for &x in view(c, a) {
        let j = lower_bound(max_repeat) { it.id < x.id };
        guard j < max_repeat.len && max_repeat[j].id == x.id else { continue; }
        // a token that is a whole breaker is not penalized
        var h = dry_first(x.id);
        var single = false;
        while h < dry_heads.len && dry_heads[h].token == x.id {
            if dry_heads[h].len == 0 { single = true; break; }
            h += 1;
        }
        guard !single else { continue; }
        var repeat_exp: i64 = max_repeat[j].n - s.k;
        if max_exponent > 0 && repeat_exp > max_exponent { repeat_exp = max_exponent; }
        // std::pow(float, int) is a double
        let penalty = (s.a * pow(s.b as f64, repeat_exp as f64)) as f32;
        x.logit -= penalty;
    }
    a.sorted = false;
}

// llama_sampler_reset of the chain: the penalties and DRY forget what they
// saw, and the samplers that draw start their generators again from their
// seeds (a new random one for the default seed).
fn reset(ch: Chain&) {
    ch.prev.clear();
    ch.counts.clear();
    ch.dry_last.clear();
    for &s in ch.samplers {
        if s.kind == KIND_DIST || s.kind == KIND_XTC || s.kind == KIND_MIROSTAT || s.kind == KIND_MIROSTAT_V2 {
            mt_seed(s.rng, rng_seed(s.seed));
        }
        if s.kind == KIND_MIROSTAT || s.kind == KIND_MIROSTAT_V2 { s.mu = 2.0 * s.a; }
    }
}

// The windows of the penalties and DRY: their accept.
fn accept(ch: Chain&, token: i32) {
    for s in ch.samplers {
        if s.kind == KIND_DRY {
            if ch.dry_last.len >= s.n {
                for i in 0..ch.dry_last.len - 1 { ch.dry_last[i] = ch.dry_last[i + 1]; }
                ch.dry_last.pop();
            }
            ch.dry_last.push(token);
            continue;
        }
        guard s.kind == KIND_PENALTIES && s.k > 0 else { continue; }

        count_add(ch, token, 1);
        if ch.prev.len >= s.k {
            let old = ch.prev[0];
            for i in 0..ch.prev.len - 1 { ch.prev[i] = ch.prev[i + 1]; }
            ch.prev.pop();
            count_add(ch, old, -1);
        }
        ch.prev.push(token);
    }
}

// The chain run over all the logits: the token it selects.
fn run_chain(ch: Chain&, cands: Cand[>..]&, logits: const f32[:], with_grammar: bool) -> i32 {
    cands.clear();
    for l, i in logits { cands.push(Cand { id: i as i32, logit: l, p: 0.0 }); }
    if with_grammar {
        // the grammar first: what it cannot take is -inf
        var ids: i32[>..] = [];
        for c in cands { ids.push(c.id); }
        var rejected: bool[>..] = [];
        grammar::reject_tokens(ids, rejected);
        for r, i in rejected { if r { cands[i].logit = NEG_INF; } }
    }
    var a = Cands { start: 0, size: cands.len, selected: -1, sorted: false };
    for &s in ch.samplers { apply(ch, cands, s, a); }
    assert(a.selected >= 0 && a.selected < a.size);
    return cands[a.start + a.selected].id;
}

// common_sampler_sample and common_sampler_accept: a token from the logits,
// then accepted. With a grammar, the chain's token is checked against it,
// and if the grammar cannot take it, the chain runs again on what the
// grammar allows (rejection sampling). cands is scratch for the candidates.
fn sample(ch: Chain&, cands: Cand[>..]&, logits: const f32[:]) -> i32 {
    var token = run_chain(ch, cands, logits, false);
    if ch.has_grammar {
        var rejected: bool[>..] = [];
        grammar::reject_tokens([token], rejected);
        if rejected[0] { token = run_chain(ch, cands, logits, true); }
        grammar::accept_impl(token);
    }
    accept(ch, token);
    return token;
}

