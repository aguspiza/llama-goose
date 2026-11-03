// Sampling: the chain llama.cpp's common_sampler builds by default. A given
// seed draws the same tokens as llama.cpp: the RNG is std::mt19937 and the
// uniform draws follow libstdc++'s generate_canonical.
// Not ported yet: DRY, mirostat, adaptive-p, infill, grammars.
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

// One sampler of the chain, with its parameters (a, b, c by kind) and RNG.
struct Sampler { kind: i64, a: f32, b: f32, c: f32, k: i64, min_keep: i64, rng: Mt19937, seed: u32 }

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
}

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
    guard p.mirostat == 0 else { abort("mirostat sampling: not ported yet"); }
    guard p.penalty_last_n <= 4096 else { abort("penalty_last_n above 4096: not supported"); }
    for kind in p.order {
        var s = new_sampler(kind);
        match kind {
            KIND_DRY => {
                // a DRY sampler with multiplier 0 does nothing
                guard p.dry_multiplier == 0.0 else { abort("DRY sampling: not ported yet"); }
                continue;
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

// The penalties' window: llama_sampler_penalties_accept.
fn accept(ch: Chain&, token: i32) {
    for s in ch.samplers {
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

// llama_sampler_sample: a token from the logits, then accepted. cands is
// scratch for the candidates.
fn sample(ch: Chain&, cands: Cand[>..]&, logits: const f32[:]) -> i32 {
    cands.clear();
    for l, i in logits { cands.push(Cand { id: i as i32, logit: l, p: 0.0 }); }
    var a = Cands { start: 0, size: cands.len, selected: -1, sorted: false };
    for &s in ch.samplers { apply(ch, cands, s, a); }
    assert(a.selected >= 0 && a.selected < a.size);
    let token = cands[a.start + a.selected].id;
    accept(ch, token);
    return token;
}
