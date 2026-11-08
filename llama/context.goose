// A context: a model's KV cache, and the decoding of batches of tokens
// through the model's graph. One sequence; flash attention, as llama.cpp
// picks on the CPU. The graph of each architecture follows llama.cpp's
// src/models/<arch>.cpp operation by operation.
// port-of: src/llama-context.cpp src/llama-graph.cpp src/llama-kv-cache.cpp src/models/gemma3.cpp src/models/llama.cpp
// port-of: src/models/lfm2.cpp src/models/lfm-base.cpp src/llama-memory-recurrent.cpp src/models/lfm3.cpp
// port-of: src/models/phi2.cpp src/models/phi3.cpp src/models/stablelm.cpp src/models/smollm3.cpp src/models/granite.cpp
// port-of: src/models/mistral3.cpp src/models/granite-moe.cpp src/models/smallthinker.cpp
// port-of: src/models/mamba.cpp src/models/mamba2.cpp src/models/mamba-base.cpp
// port-of: src/models/falcon-h1.cpp src/models/granite-hybrid.cpp src/models/jamba.cpp src/models/plamo2.cpp
// port-of: src/models/rwkv6.cpp src/models/rwkv6-base.cpp src/models/rwkv7.cpp src/models/rwkv7-base.cpp src/models/minicpm3.cpp
// port-of: src/models/qwen2vl.cpp src/models/qwen3vl.cpp src/models/nemotron-h.cpp src/models/afmoe.cpp src/models/gemma3n.cpp src/models/gemma4.cpp src/models/t5.cpp
// port-of: src/models/maincoder.cpp src/models/nanbeige.cpp src/models/bitnet.cpp src/models/chatglm.cpp
// port-of: src/models/gemma.cpp src/models/olmo.cpp src/models/qwen.cpp src/models/refact.cpp src/models/mpt.cpp src/models/jais.cpp src/models/nemotron.cpp
// port-of: src/models/gpt2.cpp src/models/starcoder.cpp src/models/gptneox.cpp src/models/starcoder2.cpp src/models/bloom.cpp src/models/openelm.cpp
// port-of: src/models/arcee.cpp src/models/olmo2.cpp src/models/exaone4.cpp src/models/hunyuan-vl.cpp
// port-of: src/models/gemma2.cpp src/models/qwen2.cpp src/models/qwen3.cpp src/models/qwen35.cpp src/models/delta-net-base.cpp
import std;
import math;
import os;
import ggml from "../ggml";
import .model;
namespace context;

let MAX_NODES = 8192;

struct Context {
    n_ctx: i64,
    n_threads: i32 = 0,          // 0: ggml::default_threads()
    kv_ctx: ggml::Ctx,
    kv_buf: ggml::Buffer,
    k: ggml::Tensor[..512],      // per attention layer: [n_embd_k_gqa, n_ctx], f16
    v: ggml::Tensor[..512],
    r: ggml::Tensor[..512],      // per recurrent layer: its convolution's state, [n_embd_r, 1], f32
    s: ggml::Tensor[..512],      // and a gated delta net's, [n_embd_s, 1], f32
    n_past: i64,                 // the cells in use: cell i holds position i
    n_enc: i64,
    galloc: ggml::Gallocr,
    embd_enc: f32[>..],          // t5: the encoder's output, [n_embd, n_enc]
}

// llama.cpp pads the context to 256 cells.
fn pad_to(n: i64, to: i64) -> i64 { (n + to - 1) / to * to }

fn init(c: Context&, m: model::Model&, n_ctx: i64) {
    c.n_ctx = pad_to(n_ctx, 256);
    if c.n_threads <= 0 { c.n_threads = ggml::default_threads(); }
    let n_layer = m.hp.n_layer as i64;
    c.kv_ctx = ggml::init(4 * n_layer * ggml::tensor_overhead(), true);
    for il in n_layer {
        // keys and values for attention, a state for a recurrent layer; falcon-h1's layers have both
        var k = ggml::NONE;
        var v = ggml::NONE;
        var r = ggml::NONE;
        var s = ggml::NONE;
        if model::has_kv(m.hp, il) {
            k = ggml::new_tensor_2d(c.kv_ctx, ggml::TYPE_F16, model::n_embd_k_gqa(m.hp, il), c.n_ctx);
            v = ggml::new_tensor_2d(c.kv_ctx, ggml::TYPE_F16, model::n_embd_v_gqa(m.hp, il), c.n_ctx);
            ggml::set_name(k, str("cache_k_l", il));
            ggml::set_name(v, str("cache_v_l", il));
        }
        if m.hp.is_recr[il] {
            r = ggml::new_tensor_2d(c.kv_ctx, ggml::TYPE_F32, model::n_embd_r(m.hp), 1);
            ggml::set_name(r, str("cache_r_l", il));
            if model::n_embd_s(m.hp) > 0 {
                s = ggml::new_tensor_2d(c.kv_ctx, ggml::TYPE_F32, model::n_embd_s(m.hp), 1);
                ggml::set_name(s, str("cache_s_l", il));
            }
        }
        c.k.push(k);
        c.v.push(v);
        c.r.push(r);
        c.s.push(s);
    }
    c.kv_buf = ggml::backend_alloc_ctx_tensors(c.kv_ctx, m.backend);
    ggml::backend_buffer_clear(c.kv_buf, 0);
    c.galloc = ggml::gallocr_new(m.backend);
    c.n_past = 0;
}

// The inputs of one graph, set after it is allocated.
struct Inputs {
    tokens: ggml::Tensor,
    pos: ggml::Tensor,
    out_ids: ggml::Tensor,
    kv_idxs: ggml::Tensor,
    mask: ggml::Tensor,
    mask_swa: ggml::Tensor,
    s_copy: ggml::Tensor,        // the recurrent state rows to read: [0]
    attn_scale: ggml::Tensor,    // mistral3: [1, 1, n_tokens], the queries' scales
    pos_bucket: ggml::Tensor,    // t5: [n_kv, n_tokens], the relative position buckets
    cross_embd: ggml::Tensor,    // t5: the encoder's output
    cross_mask: ggml::Tensor,    // t5: [n_enc, n_tokens]
}

// Per graph: what the graph builders read.
struct Build {
    ctx: ggml::Ctx,
    gf: ggml::Graph,
    n_tokens: i64,
    n_kv: i64,
    inp: Inputs,
    // LLAMA_GOOSE_TRACE=1: tensors whose sums decode prints, to compare
    // with llama.cpp's eval callback
    tracing: bool,
    traced: ggml::Tensor[..64],
}

fn trace(b: Build&, name: const u8[:], t: ggml::Tensor) {
    guard b.tracing && b.traced.len < b.traced.cap else { return; }
    ggml::set_name(t, name);
    ggml::set_output(t);
    b.traced.push(t);
}

fn print_traced(b: Build&) {
    for t in b.traced {
        var vals: f32[>..] = [];
        vals.resize(ggml::nelements(t), 0.0);
        ggml::backend_get_f32(t, vals, 0);
        var s: f64 = 0.0;
        var s2: f64 = 0.0;
        for v, i in vals { s += v; s2 += v as f64 * v * (i % 7 + 1); }
        var name: u8[>..] = [];
        ggml::tensor_name(t, name);
        write_stderr(str("SUM ", name, " ", s, " ", s2, "\n"));
    }
}

fn build_norm_rms(b: Build&, m: model::Model&, cur: ggml::Tensor, w: ggml::Tensor) -> ggml::Tensor {
    let n = ggml::rms_norm(b.ctx, cur, m.hp.f_norm_rms_eps);
    return if w.p != 0 { ggml::mul(b.ctx, n, w) } else { n };
}

// LayerNorm, with an optional weight and bias (build_norm, LLM_NORM).
fn build_norm_ln(b: Build&, m: model::Model&, cur: ggml::Tensor, w: ggml::Tensor, bias: ggml::Tensor) -> ggml::Tensor {
    var n = ggml::norm(b.ctx, cur, m.hp.f_norm_eps);
    if w.p != 0 { n = ggml::mul(b.ctx, n, w); }
    return add_bias(b, n, bias);
}

// The layer's rope factors: its own, or the long or short ones by the
// context's size (llama_model::get_rope_factors).
fn rope_factors(m: model::Model&, c: Context&, ly: model::Layer) -> ggml::Tensor {
    if ly.rope_freqs.p != 0 { return ly.rope_freqs; }
    if c.n_ctx > m.hp.n_ctx_orig_yarn { return ly.rope_long; }
    return ly.rope_short;
}

// w * cur, times the weight's scale if it has one (build_lora_mm).
fn mul_mat_s(b: Build&, w: ggml::Tensor, cur: ggml::Tensor, w_s: ggml::Tensor) -> ggml::Tensor {
    let res = ggml::mul_mat(b.ctx, w, cur);
    return if w_s.p != 0 { ggml::mul(b.ctx, res, w_s) } else { res };
}

fn add_bias(b: Build&, cur: ggml::Tensor, bias: ggml::Tensor) -> ggml::Tensor {
    if bias.p != 0 { ggml::add(b.ctx, cur, bias) } else { cur }
}

// Q, K and V as [head, n_head(_kv), n_tokens] (build_qkv).
fn build_qkv(b: Build&, m: model::Model&, ly: model::Layer, cur: ggml::Tensor, il: i64) -> ggml::Tensor, ggml::Tensor, ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let head = hp.n_embd_head_k as i64;
    let n_head = hp.n_head[il] as i64;
    let n_head_kv = hp.n_head_kv[il] as i64;
    if ly.wqkv.p != 0 {
        var qkv = ggml::mul_mat(ctx, ly.wqkv, cur);
        if ly.bqkv.p != 0 {
            qkv = ggml::add(ctx, qkv, ly.bqkv);
        } else if ly.bq.p != 0 && ly.bk.p != 0 && ly.bv.p != 0 {
            abort("fused QKV with separate biases: not ported yet");
        }
        if hp.f_clamp_kqv > 0.0 { qkv = ggml::clamp(ctx, qkv, -hp.f_clamp_kqv, hp.f_clamp_kqv); }
        let ty = ggml::tensor_type(qkv);
        let nb1 = ggml::nb(qkv, 1);
        let q = ggml::view_4d(ctx, qkv, head, n_head, b.n_tokens, 1, ggml::row_size(ty, head), nb1, nb1 * b.n_tokens, 0);
        let k = ggml::view_4d(ctx, qkv, head, n_head_kv, b.n_tokens, 1, ggml::row_size(ty, head), nb1, nb1 * b.n_tokens,
                              ggml::row_size(ty, head * n_head));
        let v = ggml::view_4d(ctx, qkv, head, n_head_kv, b.n_tokens, 1, ggml::row_size(ty, head), nb1, nb1 * b.n_tokens,
                              ggml::row_size(ty, head * (n_head + n_head_kv)));
        return q, k, v;
    }
    var q = add_bias(b, mul_mat_s(b, ly.wq, cur, ly.wq_s), ly.bq);
    var k = add_bias(b, mul_mat_s(b, ly.wk, cur, ly.wk_s), ly.bk);
    var v = add_bias(b, mul_mat_s(b, ly.wv, cur, ly.wv_s), ly.bv);
    if hp.f_clamp_kqv > 0.0 {
        q = ggml::clamp(ctx, q, -hp.f_clamp_kqv, hp.f_clamp_kqv);
        k = ggml::clamp(ctx, k, -hp.f_clamp_kqv, hp.f_clamp_kqv);
        v = ggml::clamp(ctx, v, -hp.f_clamp_kqv, hp.f_clamp_kqv);
    }
    return ggml::reshape_3d(ctx, q, head, n_head, b.n_tokens),
           ggml::reshape_3d(ctx, k, head, n_head_kv, b.n_tokens),
           ggml::reshape_3d(ctx, v, head, n_head_kv, b.n_tokens);
}

// Q, K and V as [head * n_head(_kv), n_tokens], not split into heads
// (build_qkv with reshape false).
fn build_qkv_2d(b: Build&, m: model::Model&, ly: model::Layer, cur: ggml::Tensor, il: i64) -> ggml::Tensor, ggml::Tensor, ggml::Tensor {
    let ctx = b.ctx;
    if ly.wqkv.p != 0 {
        let hp .= m.hp;
        let n_embd_q = hp.n_embd_head_k as i64 * hp.n_head[il];
        let n_embd_k = hp.n_embd_head_k as i64 * hp.n_head_kv[il];
        let n_embd_v = hp.n_embd_head_v as i64 * hp.n_head_kv[il];
        let qkv = add_bias(b, ggml::mul_mat(ctx, ly.wqkv, cur), ly.bqkv);
        let ty = ggml::tensor_type(qkv);
        let nb1 = ggml::nb(qkv, 1);
        return ggml::cont(ctx, ggml::view_2d(ctx, qkv, n_embd_q, b.n_tokens, nb1, 0)),
               ggml::cont(ctx, ggml::view_2d(ctx, qkv, n_embd_k, b.n_tokens, nb1, ggml::row_size(ty, n_embd_q))),
               ggml::cont(ctx, ggml::view_2d(ctx, qkv, n_embd_v, b.n_tokens, nb1, ggml::row_size(ty, n_embd_q + n_embd_k)));
    }
    return add_bias(b, ggml::mul_mat(ctx, ly.wq, cur), ly.bq),
           add_bias(b, ggml::mul_mat(ctx, ly.wk, cur), ly.bk),
           add_bias(b, ggml::mul_mat(ctx, ly.wv, cur), ly.bv);
}

let FFN_SILU = 0;
let FFN_GELU = 1;
let FFN_RELU = 2;

// A mixture of experts: the router picks n_expert_used experts per token and
// weighs their outputs (build_moe_ffn). probs_in: the router's logits, when
// they come from elsewhere (smallthinker).
fn build_moe_ffn(b: Build&, m: model::Model&, ly: model::Layer, cur: ggml::Tensor, op: i32, norm_w: bool,
                 w_scale: f32, gating: i64, probs_in: ggml::Tensor = ggml::NONE) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_embd = ggml::ne(cur, 0);
    let n_tokens = ggml::ne(cur, 1);
    let n_expert = hp.n_expert as i64;
    let n_expert_used = hp.n_expert_used as i64;

    let logits = if probs_in.p != 0 { probs_in } else { ggml::mul_mat(ctx, ly.ffn_gate_inp, cur) };   // [n_expert, n_tokens]
    var probs = match gating {
        model::GATING_SOFTMAX => ggml::soft_max(ctx, logits),
        model::GATING_SIGMOID => ggml::sigmoid(ctx, logits),
        model::GATING_SOFTMAX_WEIGHT => logits,
        _ => { abort(str("expert gating function ", gating, ": not ported yet")); },
    };
    // a bias of the choice only: the weights come from the unbiased probabilities
    let selection_probs = if ly.ffn_exp_probs_b.p != 0 { ggml::add(ctx, probs, ly.ffn_exp_probs_b) } else { probs };
    let selected = ggml::argsort_top_k(ctx, selection_probs, n_expert_used as i32);     // [n_expert_used, n_tokens]
    probs = ggml::reshape_3d(ctx, probs, 1, n_expert, n_tokens);
    var weights = ggml::get_rows(ctx, probs, selected);                         // [1, n_expert_used, n_tokens]
    if gating == model::GATING_SOFTMAX_WEIGHT {
        weights = ggml::reshape_2d(ctx, weights, n_expert_used, n_tokens);
        weights = ggml::soft_max(ctx, weights);
        weights = ggml::reshape_3d(ctx, weights, 1, n_expert_used, n_tokens);
    }
    if norm_w {
        weights = ggml::reshape_2d(ctx, weights, n_expert_used, n_tokens);
        var sum = ggml::sum_rows(ctx, weights);
        sum = ggml::clamp(ctx, sum, 6.103515625e-5, 1.0 / 0.0);    // no division by zero: the smallest F16
        weights = ggml::div(ctx, weights, sum);
        weights = ggml::reshape_3d(ctx, weights, 1, n_expert_used, n_tokens);
    }
    if w_scale != 0.0 && w_scale != 1.0 { weights = ggml::scale(ctx, weights, w_scale); }
    ggml::build_forward_expand(b.gf, weights);

    let x = ggml::reshape_3d(ctx, cur, n_embd, 1, n_tokens);
    let up = ggml::mul_mat_id(ctx, ly.ffn_up_exps, x, selected);                // [n_ff, n_expert_used, n_tokens]
    let has_gate = ly.ffn_gate_exps.p != 0;
    var h = if has_gate { ggml::mul_mat_id(ctx, ly.ffn_gate_exps, x, selected) } else { up };
    h = match op {
        FFN_SILU => if has_gate { ggml::swiglu_split(ctx, h, up) } else { ggml::silu(ctx, h) },
        FFN_GELU => if has_gate { ggml::geglu_split(ctx, h, up) } else { ggml::gelu(ctx, h) },
        FFN_RELU => if has_gate { ggml::reglu_split(ctx, h, up) } else { ggml::relu(ctx, h) },
        _ => { abort("build_moe_ffn: unknown activation"); },
    };
    var experts = ggml::mul_mat_id(ctx, ly.ffn_down_exps, h, selected);       // [n_embd, n_expert_used, n_tokens]
    experts = ggml::mul(ctx, experts, weights);
    ggml::build_forward_expand(b.gf, experts);

    // the experts' outputs, added up: their views first, then the adds
    var views: ggml::Tensor[>..] = [];
    for i in n_expert_used {
        views.push(ggml::view_2d(ctx, experts, n_embd, n_tokens, ggml::nb(experts, 2), i * ggml::nb(experts, 1)));
        ggml::build_forward_expand(b.gf, views[i]);
    }
    var moe_out = views[0];
    for i in 1..n_expert_used {
        moe_out = ggml::add(ctx, moe_out, views[i]);
        ggml::build_forward_expand(b.gf, moe_out);
    }
    if n_expert_used == 1 { moe_out = ggml::cont(ctx, moe_out); }   // not a view
    return moe_out;
}

// The SwiGLU feed-forward, gate and up in parallel, with optional biases (build_ffn).
fn build_ffn_swiglu(b: Build&, ly: model::Layer, cur: ggml::Tensor) -> ggml::Tensor {
    let ctx = b.ctx;
    let up = add_bias(b, ggml::mul_mat(ctx, ly.ffn_up, cur), ly.ffn_up_b);
    let gate = add_bias(b, ggml::mul_mat(ctx, ly.ffn_gate, cur), ly.ffn_gate_b);
    return add_bias(b, ggml::mul_mat(ctx, ly.ffn_down, ggml::swiglu_split(ctx, gate, up)), ly.ffn_down_b);
}

// Writes k and v of this batch into the layer's cache, and attends over the
// first n_kv cells; q, k, v are [head, n_head(_kv), n_tokens].
fn build_attn(b: Build&, m: model::Model&, c: Context&, il: i64, wo: ggml::Tensor, wo_b: ggml::Tensor,
              q: ggml::Tensor, k_cur: ggml::Tensor, v_cur: ggml::Tensor, kq_scale: f32) -> ggml::Tensor {
    return add_bias(b, ggml::mul_mat(b.ctx, wo, build_kqv(b, m, c, il, q, k_cur, v_cur, kq_scale)), wo_b);
}

// build_attn before the output projection: the heads' outputs, [head * n_head, n_tokens].
fn build_kqv(b: Build&, m: model::Model&, c: Context&, il: i64,
             q: ggml::Tensor, k_cur: ggml::Tensor, v_cur: ggml::Tensor, kq_scale: f32,
             kq_b: ggml::Tensor = ggml::NONE) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_embd_k = model::n_embd_k_gqa(hp, il);
    let n_embd_v = model::n_embd_v_gqa(hp, il);
    ggml::build_forward_expand(b.gf, q);
    // the cache this layer reads: its own, or (gemma3n) an earlier layer's, with no k and v of its own
    let src = hp.kv_source[il];
    if src == il {
        ggml::build_forward_expand(b.gf, v_cur);
        ggml::build_forward_expand(b.gf, k_cur);
        // heads and their values merged into rows; k and v may be views of a fused QKV (cpy_k, cpy_v)
        let k_rows = ggml::view_2d(ctx, k_cur, n_embd_k, b.n_tokens, ggml::nb(k_cur, 2), 0);
        let v_rows = ggml::view_2d(ctx, v_cur, n_embd_v, b.n_tokens, ggml::nb(v_cur, 2), 0);
        ggml::build_forward_expand(b.gf, ggml::set_rows(ctx, c.k[il], k_rows, b.inp.kv_idxs));
        ggml::build_forward_expand(b.gf, ggml::set_rows(ctx, c.v[il], v_rows, b.inp.kv_idxs));
    }
    let cache_k = c.k[src];
    let cache_v = c.v[src];
    let tk = ggml::tensor_type(cache_k);
    let tv = ggml::tensor_type(cache_v);
    let n_head_kv = hp.n_head_kv[il] as i64;
    let hk = model::head_k(hp, il);
    let hv = model::head_v(hp, il);
    let k = ggml::view_4d(ctx, cache_k, hk, n_head_kv, b.n_kv, 1,
                          ggml::row_size(tk, hk), ggml::row_size(tk, n_embd_k),
                          ggml::row_size(tk, n_embd_k * c.n_ctx), 0);
    let v = ggml::view_4d(ctx, cache_v, hv, n_head_kv, b.n_kv, 1,
                          ggml::row_size(tv, hv), ggml::row_size(tv, n_embd_v),
                          ggml::row_size(tv, n_embd_v * c.n_ctx), 0);

    let qp = ggml::permute(ctx, q, 0, 2, 1, 3);
    let kp = ggml::permute(ctx, k, 0, 2, 1, 3);
    let vp = ggml::permute(ctx, v, 0, 2, 1, 3);
    let mask = if hp.is_swa[il] { b.inp.mask_swa } else { b.inp.mask };
    if kq_b.p != 0 { return build_attn_mha_no_fa(b, m, qp, kp, vp, kq_b, mask, kq_scale); }
    let softcap: f32 = if hp.attn_soft_cap { hp.f_attn_logit_softcapping } else { 0.0 };
    var cur = ggml::flash_attn_ext(ctx, qp, kp, vp, mask, kq_scale, hp.f_max_alibi_bias, softcap);
    ggml::prec_set_acc(cur, ggml::PREC_F32);
    cur = ggml::reshape_2d(ctx, cur, ggml::ne(cur, 0) * ggml::ne(cur, 1), ggml::ne(cur, 2) * ggml::ne(cur, 3));
    if il == 0 { trace(b, "kqv_out-0", cur); if v_cur.p != 0 { trace(b, "Vcur-0", v_cur); } }
    ggml::build_forward_expand(b.gf, cur);
    return cur;
}

// Attention without flash attention, for a bias of the scores (build_attn_mha);
// q, k, v: [head, n_tokens or n_kv, n_head]; the result: [head * n_head, n_tokens].
fn build_attn_mha_no_fa(b: Build&, m: model::Model&, q: ggml::Tensor, k: ggml::Tensor, v: ggml::Tensor,
                        kq_b: ggml::Tensor, mask: ggml::Tensor, kq_scale: f32) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    var kq = ggml::mul_mat(ctx, k, q);
    ggml::prec_set_acc(kq, ggml::PREC_F32);
    if hp.attn_soft_cap {
        kq = ggml::scale(ctx, kq, 1.0 / hp.f_attn_logit_softcapping);
        kq = ggml::tanh(ctx, kq);
        kq = ggml::scale(ctx, kq, hp.f_attn_logit_softcapping);
    }
    if kq_b.p != 0 { kq = ggml::add(ctx, kq, kq_b); }
    kq = ggml::soft_max_ext(ctx, kq, mask, kq_scale, hp.f_max_alibi_bias);
    let vt = ggml::cont(ctx, ggml::transpose(ctx, v));     // the cache's v is not transposed
    let kqv = ggml::mul_mat(ctx, vt, kq);
    let cur = ggml::permute(ctx, kqv, 0, 2, 1, 3);
    return ggml::cont_2d(ctx, cur, ggml::ne(cur, 0) * ggml::ne(cur, 1), ggml::ne(cur, 2) * ggml::ne(cur, 3));
}

// The relative position bucket of x seen from y (llama_relative_position_bucket).
fn relative_position_bucket(x: i64, y: i64, n_buckets_in: i64, bidirectional: bool) -> i32 {
    let max_distance: i64 = 128;
    var n_buckets = n_buckets_in;
    if bidirectional { n_buckets >>= 1; }
    let max_exact = n_buckets >> 1;
    var relative_position = (x - y) as i32;
    var relative_bucket: i32 = 0;
    if bidirectional {
        relative_bucket += if relative_position > 0 { n_buckets as i32 } else { 0 };
        relative_position = abs(relative_position);
    } else {
        relative_position = -min(relative_position, 0);
    }
    // floorf(max_exact + logf(1.0 * rp / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact))
    // the types of the C++: logf of a double, times an int64 in float, over a double log
    let rp: f64 = relative_position;
    let me: f64 = max_exact;
    let md: f64 = max_distance;
    let ratio: f64 = 1.0 * rp / me;
    let lf: f32 = log(ratio as f32);
    let nb: f32 = n_buckets - max_exact;
    let num: f32 = lf * nb;
    let den: f64 = log(1.0 * md / me);
    let large_f: f64 = me + num / den;
    var if_large = floor(large_f as f32) as i32;
    if_large = min(if_large, (n_buckets - 1) as i32);
    relative_bucket += if relative_position < max_exact { relative_position } else { if_large };
    return relative_bucket;
}

// The bias of the attention scores from the buckets (build_pos_bias): [n_kv, n_tokens, n_head].
fn build_pos_bias(b: Build&, pos_bucket: ggml::Tensor, attn_rel_b: ggml::Tensor) -> ggml::Tensor {
    let ctx = b.ctx;
    let p1d = ggml::reshape_1d(ctx, pos_bucket, ggml::ne(pos_bucket, 0) * ggml::ne(pos_bucket, 1));
    var bias = ggml::get_rows(ctx, attn_rel_b, p1d);
    bias = ggml::reshape_3d(ctx, bias, ggml::ne(bias, 0), ggml::ne(pos_bucket, 0), ggml::ne(pos_bucket, 1));
    bias = ggml::permute(ctx, bias, 2, 0, 1, 3);
    return ggml::cont(ctx, bias);
}

// A t5 feed-forward: GELU-gated with a gate, ReLU without.
fn build_ffn_t5(b: Build&, up_w: ggml::Tensor, gate_w: ggml::Tensor, down_w: ggml::Tensor, cur: ggml::Tensor) -> ggml::Tensor {
    let ctx = b.ctx;
    let up = ggml::mul_mat(ctx, up_w, cur);
    let h = if gate_w.p != 0 { ggml::geglu_split(ctx, ggml::mul_mat(ctx, gate_w, cur), up) } else { ggml::relu(ctx, up) };
    return ggml::mul_mat(ctx, down_w, h);
}

// models/t5.cpp, the decoder
fn build_t5_dec(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_embd_head = hp.n_embd_head_v as i64;
    let n_tokens = b.n_tokens;
    let n_layer = hp.dec_n_layer as i64;
    let embd_enc = b.inp.cross_embd;
    let n_enc = ggml::ne(embd_enc, 1);

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    for il in n_layer {
        let ly = m.layers[il];
        var inpSA = inpL;
        var cur = build_norm_rms(b, m, inpL, ly.attn_norm);
        {
            let q, k, v = build_qkv(b, m, ly, cur, il);
            let rel_b = if ly.attn_rel_b.p != 0 { ly.attn_rel_b } else { m.layers[0].attn_rel_b };
            let kq_b = build_pos_bias(b, b.inp.pos_bucket, rel_b);
            cur = add_bias(b, ggml::mul_mat(ctx, ly.wo, build_kqv(b, m, c, il, q, k, v, 1.0, kq_b)), ly.bo);
        }
        cur = ggml::add(ctx, cur, inpSA);
        var inpCA = cur;
        cur = build_norm_rms(b, m, cur, ly.attn_norm_cross);
        {
            var q = ggml::mul_mat(ctx, ly.wq_cross, cur);
            var k = ggml::mul_mat(ctx, ly.wk_cross, embd_enc);
            var v = ggml::mul_mat(ctx, ly.wv_cross, embd_enc);
            q = ggml::reshape_3d(ctx, q, n_embd_head, hp.n_head[il], n_tokens);
            k = ggml::reshape_3d(ctx, k, n_embd_head, hp.n_head_kv[il], n_enc);
            v = ggml::reshape_3d(ctx, v, n_embd_head, hp.n_head_kv[il], n_enc);
            ggml::build_forward_expand(b.gf, q);
            ggml::build_forward_expand(b.gf, k);
            ggml::build_forward_expand(b.gf, v);
            let qp = ggml::permute(ctx, q, 0, 2, 1, 3);
            let kp = ggml::permute(ctx, k, 0, 2, 1, 3);
            let vp = ggml::permute(ctx, v, 0, 2, 1, 3);
            var att = ggml::flash_attn_ext(ctx, qp, ggml::cast(ctx, kp, ggml::TYPE_F16), ggml::cast(ctx, vp, ggml::TYPE_F16),
                                           b.inp.cross_mask, 1.0, hp.f_max_alibi_bias, 0.0);
            ggml::prec_set_acc(att, ggml::PREC_F32);
            att = ggml::reshape_2d(ctx, att, ggml::ne(att, 0) * ggml::ne(att, 1), ggml::ne(att, 2) * ggml::ne(att, 3));
            cur = ggml::mul_mat(ctx, ly.wo_cross, att);
        }
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpCA = ggml::get_rows(ctx, inpCA, b.inp.out_ids);
        }
        let ffn_inp = ggml::add(ctx, cur, inpCA);
        cur = build_norm_rms(b, m, ffn_inp, ly.ffn_norm);
        cur = build_ffn_t5(b, ly.ffn_up, ly.ffn_gate, ly.ffn_down, cur);
        inpL = ggml::add(ctx, cur, ffn_inp);
    }
    let cur = build_norm_rms(b, m, inpL, m.output_norm);
    return ggml::mul_mat(ctx, m.output, cur);
}

// models/t5.cpp, the encoder: its output for every token, [n_embd, n_tokens]
fn build_t5_enc(b: Build&, m: model::Model&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_embd_head = hp.n_embd_head_v as i64;
    let n_tokens = b.n_tokens;
    let n_layer = hp.n_layer as i64;

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    for il in n_layer {
        let ly = m.layers[il];
        var inpSA = inpL;
        var cur = build_norm_rms(b, m, inpL, ly.attn_norm_enc);
        {
            var q = ggml::mul_mat(ctx, ly.wq_enc, cur);
            var k = ggml::mul_mat(ctx, ly.wk_enc, cur);
            var v = ggml::mul_mat(ctx, ly.wv_enc, cur);
            q = ggml::reshape_3d(ctx, q, n_embd_head, hp.n_head[il], n_tokens);
            k = ggml::reshape_3d(ctx, k, n_embd_head, hp.n_head_kv[il], n_tokens);
            v = ggml::reshape_3d(ctx, v, n_embd_head, hp.n_head_kv[il], n_tokens);
            let rel_b = if ly.attn_rel_b_enc.p != 0 { ly.attn_rel_b_enc } else { m.layers[0].attn_rel_b_enc };
            var kq_b = build_pos_bias(b, b.inp.pos_bucket, rel_b);
            ggml::build_forward_expand(b.gf, q);
            ggml::build_forward_expand(b.gf, k);
            ggml::build_forward_expand(b.gf, v);
            // no cache: k, v and the bias padded with zeros to the mask's width
            let n_kv_pad = ggml::ne(b.inp.mask, 0) - ggml::ne(k, 2);
            k = ggml::pad_ext(ctx, k, 0, 0, n_kv_pad as i32, 0);
            v = ggml::pad_ext(ctx, v, 0, 0, n_kv_pad as i32, 0);
            kq_b = ggml::pad_ext(ctx, kq_b, n_kv_pad as i32, 0, 0, 0);
            let att = build_attn_mha_no_fa(b, m, ggml::permute(ctx, q, 0, 2, 1, 3), ggml::permute(ctx, k, 0, 2, 1, 3),
                                           ggml::permute(ctx, v, 0, 2, 1, 3), kq_b, b.inp.mask, 1.0);
            cur = ggml::mul_mat(ctx, ly.wo_enc, att);
        }
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpSA = ggml::get_rows(ctx, inpSA, b.inp.out_ids);
        }
        let ffn_inp = ggml::add(ctx, cur, inpSA);
        cur = build_norm_rms(b, m, ffn_inp, ly.ffn_norm_enc);
        cur = build_ffn_t5(b, ly.ffn_up_enc, ly.ffn_gate_enc, ly.ffn_down_enc, cur);
        inpL = ggml::add(ctx, cur, ffn_inp);
    }
    return build_norm_rms(b, m, inpL, m.output_norm_enc);
}

// Runs tokens, at positions 0.., through the model's encoder, without its
// causal mask, and keeps its output for the decoder's cross attention
// (llama_encode).
fn encode(c: Context&, m: model::Model&, tokens: const i32[:]) {
    let n_tokens = tokens.len;
    guard n_tokens > 0 else { abort("encode: no tokens"); }
    let n_kv = pad_to(n_tokens, 256);
    let gctx = ggml::init(MAX_NODES * ggml::tensor_overhead() + ggml::graph_overhead_custom(MAX_NODES, false), true);
    var b = Build {
        ctx: gctx,
        gf: ggml::new_graph_custom(gctx, MAX_NODES, false),
        n_tokens: n_tokens,
        n_kv: n_kv,
        inp: Inputs {
            tokens: ggml::new_tensor_1d(gctx, ggml::TYPE_I32, n_tokens),
            pos: ggml::NONE,
            out_ids: ggml::new_tensor_1d(gctx, ggml::TYPE_I32, n_tokens),
            kv_idxs: ggml::NONE,
            mask: ggml::new_tensor_2d(gctx, ggml::TYPE_F16, n_kv, n_tokens),
            mask_swa: ggml::NONE,
            s_copy: ggml::NONE,
            attn_scale: ggml::NONE,
            pos_bucket: ggml::new_tensor_2d(gctx, ggml::TYPE_I32, n_tokens, n_tokens),
            cross_embd: ggml::NONE,
            cross_mask: ggml::NONE,
        },
        tracing: false,
        traced: [],
    };
    ggml::set_input(b.inp.tokens);
    ggml::set_input(b.inp.out_ids);
    ggml::set_input(b.inp.mask);
    ggml::set_input(b.inp.pos_bucket);
    let out = build_t5_enc(b, m);
    ggml::set_output(out);
    ggml::build_forward_expand(b.gf, out);
    guard ggml::gallocr_alloc_graph(c.galloc, b.gf) else { abort("encode: failed to allocate the graph"); }

    ggml::backend_set_i32(b.inp.tokens, tokens);
    var ids: i32[>..] = [];
    for t in n_tokens { ids.push(t as i32); }
    ggml::backend_set_i32(b.inp.out_ids, ids);
    var mask: f32[>..] = [];
    for t in n_tokens { for j in n_kv { mask.push(if j < n_tokens { 0.0 } else { NEG_INF }); } }
    ggml::backend_set_f16(b.inp.mask, mask);
    var buckets: i32[>..] = [];
    for j in n_tokens { for i in n_tokens { buckets.push(relative_position_bucket(i, j, m.hp.n_rel_attn_bkts, true)); } }
    ggml::backend_set_i32(b.inp.pos_bucket, buckets);

    ggml::backend_cpu_set_n_threads(m.backend, c.n_threads);
    guard ggml::backend_graph_compute(m.backend, b.gf) == 0 else { abort("encode: graph compute failed"); }
    c.embd_enc.clear();
    c.embd_enc.resize(ggml::nelements(out), 0.0);
    ggml::backend_get_f32(out, c.embd_enc, 0);
    c.n_enc = n_tokens;
    ggml::free(gctx);
}

fn rope(b: Build&, m: model::Model&, x: ggml::Tensor, freq_base: f32, freq_scale: f32,
        factors: ggml::Tensor = ggml::NONE, n_rot: i64 = -1) -> ggml::Tensor {
    let hp .= m.hp;
    let n_dims = if n_rot >= 0 { n_rot } else { hp.n_rot as i64 };
    return ggml::rope_ext(b.ctx, x, b.inp.pos, factors, n_dims as i32, hp.rope_type,
                          hp.n_ctx_orig_yarn as i32, freq_base, freq_scale, hp.yarn_ext_factor, hp.yarn_attn_factor,
                          hp.yarn_beta_fast, hp.yarn_beta_slow);
}

// models/gemma3.cpp
fn build_gemma3(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_embd_head = hp.n_embd_head_k as i64;
    let n_layer = hp.n_layer as i64;

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    inpL = ggml::scale(ctx, inpL, sqrt(hp.n_embd as f32));

    for il in n_layer {
        let ly = m.layers[il];
        let freq_base = if hp.is_swa[il] { hp.rope_freq_base_train_swa } else { hp.rope_freq_base_train };
        let freq_scale = if hp.is_swa[il] { hp.rope_freq_scale_train_swa } else { hp.rope_freq_scale_train };

        var cur = build_norm_rms(b, m, inpL, ly.attn_norm);

        // self-attention
        var q, k, v = build_qkv(b, m, ly, cur, il);
        q = build_norm_rms(b, m, q, ly.attn_q_norm);
        q = rope(b, m, q, freq_base, freq_scale);
        k = build_norm_rms(b, m, k, ly.attn_k_norm);
        k = rope(b, m, k, freq_base, freq_scale);
        // ref: https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/model.py#L315
        q = ggml::scale(ctx, q, hp.f_attention_scale);
        cur = build_attn(b, m, c, il, ly.wo, ly.bo, q, k, v, 1.0);

        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpL = ggml::get_rows(ctx, inpL, b.inp.out_ids);
        }
        cur = build_norm_rms(b, m, cur, ly.attn_post_norm);
        let sa_out = ggml::add(ctx, cur, inpL);

        // feed-forward: GELU-gated, gate and up in parallel
        cur = build_norm_rms(b, m, sa_out, ly.ffn_norm);
        let up = ggml::mul_mat(ctx, ly.ffn_up, cur);
        let gate = ggml::mul_mat(ctx, ly.ffn_gate, cur);
        cur = ggml::mul_mat(ctx, ly.ffn_down, ggml::geglu_split(ctx, gate, up));
        cur = build_norm_rms(b, m, cur, ly.ffn_post_norm);
        inpL = ggml::add(ctx, cur, sa_out);
    }

    var cur = build_norm_rms(b, m, inpL, m.output_norm);
    cur = ggml::mul_mat(ctx, m.output, cur);
    let cap = hp.f_final_logit_softcapping;
    if cap != 0.0 {
        cur = ggml::scale(ctx, cur, 1.0 / cap);
        cur = ggml::tanh(ctx, cur);
        cur = ggml::scale(ctx, cur, cap);
    }
    return cur;
}

// models/phi2.cpp
fn build_phi2(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_layer = hp.n_layer as i64;
    let n_embd_head = hp.n_embd_head_v as i64;

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    for il in n_layer {
        let ly = m.layers[il];
        var attn_norm_output = build_norm_ln(b, m, inpL, ly.attn_norm, ly.attn_norm_b);
        var q, k, v = build_qkv(b, m, ly, attn_norm_output, il);
        q = rope(b, m, q, hp.rope_freq_base_train, hp.rope_freq_scale_train);
        k = rope(b, m, k, hp.rope_freq_base_train, hp.rope_freq_scale_train);
        q = ggml::scale(ctx, q, 1.0 / sqrt(n_embd_head as f32));
        var cur = build_attn(b, m, c, il, ly.wo, ly.bo, q, k, v, 1.0);
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpL = ggml::get_rows(ctx, inpL, b.inp.out_ids);
            attn_norm_output = ggml::get_rows(ctx, attn_norm_output, b.inp.out_ids);
        }
        // the feed-forward reads the attention's input
        var ffn = add_bias(b, ggml::mul_mat(ctx, ly.ffn_up, attn_norm_output), ly.ffn_up_b);
        ffn = ggml::gelu(ctx, ffn);
        ffn = add_bias(b, ggml::mul_mat(ctx, ly.ffn_down, ffn), ly.ffn_down_b);
        cur = ggml::add(ctx, cur, ffn);
        inpL = ggml::add(ctx, cur, inpL);
    }
    let cur = build_norm_ln(b, m, inpL, m.output_norm, m.output_norm_b);
    return ggml::add(ctx, ggml::mul_mat(ctx, m.output, cur), m.output_b);
}

// models/phi3.cpp
fn build_phi3(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_layer = hp.n_layer as i64;
    let n_embd_head = hp.n_embd_head_v as i64;

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    for il in n_layer {
        let ly = m.layers[il];
        var residual = inpL;
        let factors = rope_factors(m, c, ly);
        let attn_norm_output = build_norm_rms(b, m, inpL, ly.attn_norm);
        var q, k, v = build_qkv(b, m, ly, attn_norm_output, il);
        q = rope(b, m, q, hp.rope_freq_base_train, hp.rope_freq_scale_train, factors);
        k = rope(b, m, k, hp.rope_freq_base_train, hp.rope_freq_scale_train, factors);
        q = ggml::scale(ctx, q, 1.0 / sqrt(n_embd_head as f32));
        var cur = build_attn(b, m, c, il, ly.wo, ly.bo, q, k, v, 1.0);
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            residual = ggml::get_rows(ctx, residual, b.inp.out_ids);
        }
        cur = ggml::add(ctx, cur, residual);
        residual = cur;
        cur = build_norm_rms(b, m, cur, ly.ffn_norm);
        // SwiGLU over the two halves of up
        cur = ggml::mul_mat(ctx, ly.ffn_down, ggml::swiglu(ctx, ggml::mul_mat(ctx, ly.ffn_up, cur)));
        inpL = ggml::add(ctx, residual, cur);
    }
    let cur = build_norm_rms(b, m, inpL, m.output_norm);
    return add_bias(b, ggml::mul_mat(ctx, m.output, cur), m.output_b);
}

// models/stablelm.cpp
fn build_stablelm(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_layer = hp.n_layer as i64;
    let n_embd_head = hp.n_embd_head_v as i64;

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    for il in n_layer {
        let ly = m.layers[il];
        var cur = build_norm_ln(b, m, inpL, ly.attn_norm, ly.attn_norm_b);
        var inpSA = cur;
        var q, k, v = build_qkv(b, m, ly, cur, il);
        if ly.attn_q_norm.p != 0 { q = build_norm_ln(b, m, q, ly.attn_q_norm, ggml::NONE); }
        if ly.attn_k_norm.p != 0 { k = build_norm_ln(b, m, k, ly.attn_k_norm, ggml::NONE); }
        q = rope(b, m, q, hp.rope_freq_base_train, hp.rope_freq_scale_train);
        k = rope(b, m, k, hp.rope_freq_base_train, hp.rope_freq_scale_train);
        cur = build_attn(b, m, c, il, ly.wo, ggml::NONE, q, k, v, 1.0 / sqrt(n_embd_head as f32));
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpL = ggml::get_rows(ctx, inpL, b.inp.out_ids);
            inpSA = ggml::get_rows(ctx, inpSA, b.inp.out_ids);
        }
        let ffn_inp = ggml::add(ctx, cur, inpL);
        // without its own norm the feed-forward reads the attention's normed input
        cur = if ly.ffn_norm.p != 0 { build_norm_ln(b, m, ffn_inp, ly.ffn_norm, ly.ffn_norm_b) } else { inpSA };
        cur = build_ffn_swiglu(b, ly, cur);
        inpL = ggml::add(ctx, cur, ffn_inp);
    }
    let cur = build_norm_ln(b, m, inpL, m.output_norm, m.output_norm_b);
    return ggml::mul_mat(ctx, m.output, cur);
}

// models/smollm3.cpp, models/granite.cpp: llama's graph; smollm3 skips the
// rope of every n_no_rope_layer_step-th layer, granite scales the
// embeddings, the residual branches and the logits.
fn build_smollm3_granite(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_embd_head = hp.n_embd_head_v as i64;
    assert(n_embd_head == hp.n_embd_head_k && n_embd_head == hp.n_rot);
    let n_layer = hp.n_layer as i64;
    let granite = hp.arch == model::ARCH_GRANITE || hp.arch == model::ARCH_GRANITE_MOE;
    let kq_scale: f32 = if hp.f_attention_scale == 0.0 { 1.0 / sqrt(n_embd_head as f32) } else { hp.f_attention_scale };

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    if hp.f_embedding_scale != 0.0 { inpL = ggml::scale(ctx, inpL, hp.f_embedding_scale); }
    for il in n_layer {
        let ly = m.layers[il];
        var inpSA = inpL;
        var cur = build_norm_rms(b, m, inpL, ly.attn_norm);
        var q, k, v = build_qkv(b, m, ly, cur, il);
        let use_rope = if granite { hp.rope_finetuned }
                       else { (il + 1) % hp.n_no_rope_layer_step != 0 };
        if use_rope {
            let factors = if granite { rope_factors(m, c, ly) } else { ggml::NONE };
            q = rope(b, m, q, hp.rope_freq_base_train, hp.rope_freq_scale_train, factors);
            k = rope(b, m, k, hp.rope_freq_base_train, hp.rope_freq_scale_train, factors);
        }
        cur = build_attn(b, m, c, il, ly.wo, ly.bo, q, k, v, kq_scale);
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpSA = ggml::get_rows(ctx, inpSA, b.inp.out_ids);
        }
        if hp.f_residual_scale != 0.0 { cur = ggml::scale(ctx, cur, hp.f_residual_scale); }
        let ffn_inp = ggml::add(ctx, cur, inpSA);
        cur = build_norm_rms(b, m, ffn_inp, ly.ffn_norm);
        if ly.ffn_gate_inp.p != 0 {
            // granitemoe, and its shared expert
            let moe_out = build_moe_ffn(b, m, ly, cur, FFN_SILU, true, hp.expert_weights_scale, model::GATING_SOFTMAX);
            cur = if hp.n_ff_shexp > 0 {
                ggml::add(ctx, moe_out, build_ffn_swiglu(b, model::Layer { ffn_up: ly.ffn_up_shexp, ffn_gate: ly.ffn_gate_shexp,
                                                                           ffn_down: ly.ffn_down_shexp, .. }, cur))
            } else { moe_out };
        } else {
            cur = build_ffn_swiglu(b, ly, cur);
        }
        if hp.f_residual_scale != 0.0 { cur = ggml::scale(ctx, cur, hp.f_residual_scale); }
        inpL = ggml::add(ctx, cur, ffn_inp);
    }
    let cur = build_norm_rms(b, m, inpL, m.output_norm);
    let logits = ggml::mul_mat(ctx, m.output, cur);
    return if granite { ggml::scale(ctx, logits, 1.0 / hp.f_logit_scale) } else { logits };
}

// models/gpt2.cpp, starcoder.cpp, gptneox.cpp, starcoder2.cpp, bloom.cpp
fn build_classic(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_embd_head = hp.n_embd_head_v as i64;
    let n_layer = hp.n_layer as i64;
    let has_rope = hp.rope_type >= 0;
    // jais scales by 1/head, not 1/sqrt(head)
    let head: f32 = n_embd_head;
    let kq_scale: f32 = if hp.arch == model::ARCH_JAIS { 1.0 / head } else { 1.0 / sqrt(head) };

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    if m.pos_embd.p != 0 { inpL = ggml::add(ctx, inpL, ggml::get_rows(ctx, m.pos_embd, b.inp.pos)); }
    if m.tok_norm.p != 0 { inpL = build_norm_ln(b, m, inpL, m.tok_norm, m.tok_norm_b); }
    for il in n_layer {
        let ly = m.layers[il];
        var cur = build_norm_ln(b, m, inpL, ly.attn_norm, ly.attn_norm_b);
        var q, k, v = build_qkv(b, m, ly, cur, il);
        if ly.attn_q_norm.p != 0 {
            // mpt: Q and K normed whole
            let n_head = hp.n_head[il] as i64;
            let n_head_kv = hp.n_head_kv[il] as i64;
            q = ggml::reshape_2d(ctx, q, n_embd_head * n_head, b.n_tokens);
            k = ggml::reshape_2d(ctx, k, n_embd_head * n_head_kv, b.n_tokens);
            q = build_norm_ln(b, m, q, ly.attn_q_norm, ly.attn_q_norm_b);
            k = build_norm_ln(b, m, k, ly.attn_k_norm, ly.attn_k_norm_b);
            q = ggml::reshape_3d(ctx, q, n_embd_head, n_head, b.n_tokens);
            k = ggml::reshape_3d(ctx, k, n_embd_head, n_head_kv, b.n_tokens);
        }
        if has_rope {
            q = rope(b, m, q, hp.rope_freq_base_train, hp.rope_freq_scale_train);
            k = rope(b, m, k, hp.rope_freq_base_train, hp.rope_freq_scale_train);
        }
        cur = build_attn(b, m, c, il, ly.wo, ly.bo, q, k, v, kq_scale);
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpL = ggml::get_rows(ctx, inpL, b.inp.out_ids);
        }
        if hp.use_par_res {
            // the feed-forward reads the layer's input, not the attention's output
            let attn_out = cur;
            cur = build_norm_ln(b, m, inpL, ly.ffn_norm, ly.ffn_norm_b);
            cur = build_ffn_gelu(b, ly, cur);
            cur = ggml::add(ctx, cur, inpL);
            inpL = ggml::add(ctx, cur, attn_out);
        } else {
            let ffn_inp = ggml::add(ctx, cur, inpL);
            cur = build_norm_ln(b, m, ffn_inp, ly.ffn_norm, ly.ffn_norm_b);
            cur = if hp.arch == model::ARCH_JAIS { build_ffn_swiglu(b, ly, cur) }
                  else if hp.arch == model::ARCH_NEMOTRON { build_ffn_relu_sqr(b, ly, cur) }
                  else { build_ffn_gelu(b, ly, cur) };
            inpL = ggml::add(ctx, cur, ffn_inp);
        }
    }
    let cur = build_norm_ln(b, m, inpL, m.output_norm, m.output_norm_b);
    return ggml::mul_mat(ctx, m.output, cur);
}

// A GELU feed-forward without gate, with biases (build_ffn, LLM_FFN_GELU, LLM_FFN_SEQ).
fn build_ffn_gelu(b: Build&, ly: model::Layer, cur: ggml::Tensor) -> ggml::Tensor {
    let ctx = b.ctx;
    let up = add_bias(b, ggml::mul_mat(ctx, ly.ffn_up, cur), ly.ffn_up_b);
    var h = ggml::gelu(ctx, up);
    if ly.ffn_act.p != 0 { h = ggml::div(ctx, h, ly.ffn_act); }
    return add_bias(b, ggml::mul_mat(ctx, ly.ffn_down, h), ly.ffn_down_b);
}

// A ReLU² feed-forward without gate, with biases (LLM_FFN_RELU_SQR, LLM_FFN_SEQ).
fn build_ffn_relu_sqr(b: Build&, ly: model::Layer, cur: ggml::Tensor) -> ggml::Tensor {
    let ctx = b.ctx;
    let up = add_bias(b, ggml::mul_mat(ctx, ly.ffn_up, cur), ly.ffn_up_b);
    return add_bias(b, ggml::mul_mat(ctx, ly.ffn_down, ggml::sqr(ctx, ggml::relu(ctx, up))), ly.ffn_down_b);
}

// models/bitnet.cpp
fn build_bitnet(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_embd_head = hp.n_embd_head_v as i64;
    let n_layer = hp.n_layer as i64;

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    for il in n_layer {
        let ly = m.layers[il];
        var inpSA = inpL;
        var cur = build_norm_rms(b, m, inpL, ly.attn_norm);
        var q, k, v = build_qkv(b, m, ly, cur, il);
        q = rope(b, m, q, hp.rope_freq_base_train, hp.rope_freq_scale_train);
        k = rope(b, m, k, hp.rope_freq_base_train, hp.rope_freq_scale_train);
        cur = build_kqv(b, m, c, il, q, k, v, 1.0 / sqrt(n_embd_head as f32));
        cur = build_norm_rms(b, m, cur, ly.attn_sub_norm);
        cur = add_bias(b, mul_mat_s(b, ly.wo, cur, ly.wo_s), ly.bo);
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpSA = ggml::get_rows(ctx, inpSA, b.inp.out_ids);
        }
        let ffn_inp = ggml::add(ctx, cur, inpSA);
        cur = build_norm_rms(b, m, ffn_inp, ly.ffn_norm);
        let up = mul_mat_s(b, ly.ffn_up, cur, ly.ffn_up_s);
        let gate = mul_mat_s(b, ly.ffn_gate, cur, ly.ffn_gate_s);
        cur = ggml::swiglu_split(ctx, gate, up);
        cur = build_norm_rms(b, m, cur, ly.ffn_sub_norm);
        cur = mul_mat_s(b, ly.ffn_down, cur, ly.ffn_down_s);
        inpL = ggml::add(ctx, cur, ffn_inp);
    }
    let cur = build_norm_rms(b, m, inpL, m.output_norm);
    return ggml::mul_mat(ctx, m.tok_embd, cur);
}

// models/chatglm.cpp
fn build_chatglm(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_embd_head = hp.n_embd_head_v as i64;
    let n_layer = hp.n_layer as i64;

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    for il in n_layer {
        let ly = m.layers[il];
        var inpSA = inpL;
        var cur = build_norm_rms(b, m, inpL, ly.attn_norm);
        var q, k, v = build_qkv(b, m, ly, cur, il);
        q = rope(b, m, q, hp.rope_freq_base_train, hp.rope_freq_scale_train);
        k = rope(b, m, k, hp.rope_freq_base_train, hp.rope_freq_scale_train);
        cur = build_attn(b, m, c, il, ly.wo, ggml::NONE, q, k, v, 1.0 / sqrt(n_embd_head as f32));
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpSA = ggml::get_rows(ctx, inpSA, b.inp.out_ids);
        }
        let ffn_inp = ggml::add(ctx, cur, inpSA);
        cur = build_norm_rms(b, m, ffn_inp, ly.ffn_norm);
        cur = ggml::mul_mat(ctx, ly.ffn_down, ggml::swiglu(ctx, ggml::mul_mat(ctx, ly.ffn_up, cur)));
        inpL = ggml::add(ctx, cur, ffn_inp);
    }
    let cur = build_norm_rms(b, m, inpL, m.output_norm);
    return ggml::mul_mat(ctx, m.output, cur);
}

// models/gemma.cpp
fn build_gemma(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_embd_head = hp.n_embd_head_v as i64;
    let n_layer = hp.n_layer as i64;

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    inpL = ggml::scale(ctx, inpL, sqrt(hp.n_embd as f32));
    for il in n_layer {
        let ly = m.layers[il];
        var cur = build_norm_rms(b, m, inpL, ly.attn_norm);
        var q, k, v = build_qkv(b, m, ly, cur, il);
        q = rope(b, m, q, hp.rope_freq_base_train, hp.rope_freq_scale_train);
        k = rope(b, m, k, hp.rope_freq_base_train, hp.rope_freq_scale_train);
        q = ggml::scale(ctx, q, 1.0 / sqrt(n_embd_head as f32));
        cur = build_attn(b, m, c, il, ly.wo, ggml::NONE, q, k, v, 1.0);
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpL = ggml::get_rows(ctx, inpL, b.inp.out_ids);
        }
        let sa_out = ggml::add(ctx, cur, inpL);
        cur = build_norm_rms(b, m, sa_out, ly.ffn_norm);
        let up = ggml::mul_mat(ctx, ly.ffn_up, cur);
        let gate = ggml::mul_mat(ctx, ly.ffn_gate, cur);
        cur = ggml::mul_mat(ctx, ly.ffn_down, ggml::geglu_split(ctx, gate, up));
        inpL = ggml::add(ctx, cur, sa_out);
    }
    let cur = build_norm_rms(b, m, inpL, m.output_norm);
    return ggml::mul_mat(ctx, m.output, cur);
}

// models/olmo.cpp, models/refact.cpp: llama's layers; olmo's norms are
// LayerNorms without weights, refact has ALiBi and no rope.
fn build_olmo_refact(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_embd_head = hp.n_embd_head_v as i64;
    let n_layer = hp.n_layer as i64;
    let olmo = hp.arch == model::ARCH_OLMO;
    fn norm(x: ggml::Tensor, w: ggml::Tensor) -> ggml::Tensor { if olmo { build_norm_ln(b, m, x, ggml::NONE, ggml::NONE) } else { build_norm_rms(b, m, x, w) } }

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    for il in n_layer {
        let ly = m.layers[il];
        var inpSA = inpL;
        var cur = norm(inpL, ly.attn_norm);
        var q, k, v = build_qkv(b, m, ly, cur, il);
        if olmo {
            q = rope(b, m, q, hp.rope_freq_base_train, hp.rope_freq_scale_train);
            k = rope(b, m, k, hp.rope_freq_base_train, hp.rope_freq_scale_train);
        }
        cur = build_attn(b, m, c, il, ly.wo, ggml::NONE, q, k, v, 1.0 / sqrt(n_embd_head as f32));
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpSA = ggml::get_rows(ctx, inpSA, b.inp.out_ids);
        }
        let ffn_inp = ggml::add(ctx, cur, inpSA);
        cur = norm(ffn_inp, ly.ffn_norm);
        cur = build_ffn_swiglu(b, ly, cur);
        inpL = ggml::add(ctx, cur, ffn_inp);
    }
    let cur = norm(inpL, m.output_norm);
    return ggml::mul_mat(ctx, m.output, cur);
}

// models/openelm.cpp
fn build_openelm(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_embd_head = hp.n_embd_head_v as i64;
    let n_layer = hp.n_layer as i64;

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    for il in n_layer {
        let ly = m.layers[il];
        var residual = inpL;
        var cur = build_norm_rms(b, m, inpL, ly.attn_norm);
        var q, k, v = build_qkv(b, m, ly, cur, il);
        q = build_norm_rms(b, m, q, ly.attn_q_norm);
        k = build_norm_rms(b, m, k, ly.attn_k_norm);
        q = rope(b, m, q, hp.rope_freq_base_train, hp.rope_freq_scale_train);
        k = rope(b, m, k, hp.rope_freq_base_train, hp.rope_freq_scale_train);
        cur = build_attn(b, m, c, il, ly.wo, ggml::NONE, q, k, v, 1.0 / sqrt(n_embd_head as f32));
        if il == n_layer - 1 {
            residual = ggml::get_rows(ctx, residual, b.inp.out_ids);
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
        }
        let ffn_inp = ggml::add(ctx, residual, cur);
        cur = build_norm_rms(b, m, ffn_inp, ly.ffn_norm);
        cur = build_ffn_swiglu(b, ly, cur);
        inpL = ggml::add(ctx, cur, ffn_inp);
    }
    let cur = build_norm_rms(b, m, inpL, m.output_norm);
    return ggml::mul_mat(ctx, m.output, cur);
}

// models/smallthinker.cpp
fn build_smallthinker(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_embd_head = hp.n_embd_head_v as i64;
    assert(n_embd_head == hp.n_embd_head_k && n_embd_head == hp.n_rot);
    let n_layer = hp.n_layer as i64;

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    for il in n_layer {
        let ly = m.layers[il];
        let freq_base = if hp.is_swa[il] { hp.rope_freq_base_train_swa } else { hp.rope_freq_base_train };
        let freq_scale = if hp.is_swa[il] { hp.rope_freq_scale_train_swa } else { hp.rope_freq_scale_train };
        var inpSA = inpL;
        let use_rope = hp.n_no_rope_layer_step == hp.n_layer || il % hp.n_no_rope_layer_step != 0;
        var probs = ggml::mul_mat(ctx, ly.ffn_gate_inp, inpL);     // [n_expert, n_tokens]
        var cur = build_norm_rms(b, m, inpL, ly.attn_norm);
        var q, k, v = build_qkv(b, m, ly, cur, il);
        if use_rope {
            q = rope(b, m, q, freq_base, freq_scale);
            k = rope(b, m, k, freq_base, freq_scale);
        }
        cur = build_attn(b, m, c, il, ly.wo, ly.bo, q, k, v, 1.0 / sqrt(n_embd_head as f32));
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpSA = ggml::get_rows(ctx, inpSA, b.inp.out_ids);
            probs = ggml::get_rows(ctx, probs, b.inp.out_ids);
        }
        let ffn_inp = ggml::add(ctx, cur, inpSA);
        cur = build_norm_rms(b, m, ffn_inp, ly.ffn_norm);
        cur = build_moe_ffn(b, m, ly, cur, FFN_RELU, true, hp.expert_weights_scale, hp.expert_gating_func, probs);
        inpL = ggml::add(ctx, cur, ffn_inp);
    }
    let cur = build_norm_rms(b, m, inpL, m.output_norm);
    return ggml::mul_mat(ctx, m.output, cur);
}

// models/arcee.cpp
fn build_arcee(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_embd_head = hp.n_embd_head_v as i64;
    let n_layer = hp.n_layer as i64;
    let kq_scale: f32 = if hp.f_attention_scale == 0.0 { 1.0 / sqrt(n_embd_head as f32) } else { hp.f_attention_scale };

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    for il in n_layer {
        let ly = m.layers[il];
        var inpSA = inpL;
        var cur = build_norm_rms(b, m, inpL, ly.attn_norm);
        let factors = rope_factors(m, c, ly);
        var q, k, v = build_qkv(b, m, ly, cur, il);
        q = rope(b, m, q, hp.rope_freq_base_train, hp.rope_freq_scale_train, factors);
        k = rope(b, m, k, hp.rope_freq_base_train, hp.rope_freq_scale_train, factors);
        cur = build_attn(b, m, c, il, ly.wo, ly.bo, q, k, v, kq_scale);
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpSA = ggml::get_rows(ctx, inpSA, b.inp.out_ids);
        }
        let ffn_inp = ggml::add(ctx, cur, inpSA);
        cur = build_norm_rms(b, m, ffn_inp, ly.ffn_norm);
        cur = ggml::sqr(ctx, ggml::relu(ctx, ggml::mul_mat(ctx, ly.ffn_up, cur)));
        cur = ggml::mul_mat(ctx, ly.ffn_down, cur);
        inpL = ggml::add(ctx, cur, ffn_inp);
    }
    let cur = build_norm_rms(b, m, inpL, m.output_norm);
    return ggml::mul_mat(ctx, m.output, cur);
}

// models/olmo2.cpp, models/exaone4.cpp: no norm before the attention and the
// feed-forward, one after each. olmo2 norms Q and K whole, exaone4 per head
// and, with sliding windows, ropes only the sliding window layers.
fn build_post_norm(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_embd_head = hp.n_embd_head_v as i64;
    assert(n_embd_head == hp.n_embd_head_k && n_embd_head == hp.n_rot);
    let n_layer = hp.n_layer as i64;
    let olmo2 = hp.arch == model::ARCH_OLMO2;

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    for il in n_layer {
        let ly = m.layers[il];
        var inpSA = inpL;
        var q = ggml::NONE;
        var k = ggml::NONE;
        var v = ggml::NONE;
        if olmo2 {
            q, k, v = build_qkv_2d(b, m, ly, inpL, il);
            q = build_norm_rms(b, m, q, ly.attn_q_norm);
            k = build_norm_rms(b, m, k, ly.attn_k_norm);
            q = ggml::reshape_3d(ctx, q, n_embd_head, hp.n_head[il], b.n_tokens);
            k = ggml::reshape_3d(ctx, k, n_embd_head, hp.n_head_kv[il], b.n_tokens);
            v = ggml::reshape_3d(ctx, v, n_embd_head, hp.n_head_kv[il], b.n_tokens);
            if hp.is_swa[il] {
                // olmo2's sliding window layers: no rope scaling
                q = ggml::rope_ext(ctx, q, b.inp.pos, ggml::NONE, hp.n_rot as i32, hp.rope_type, hp.n_ctx_orig_yarn as i32,
                                   hp.rope_freq_base_train_swa, 1.0, 0.0, 1.0, hp.yarn_beta_fast, hp.yarn_beta_slow);
                k = ggml::rope_ext(ctx, k, b.inp.pos, ggml::NONE, hp.n_rot as i32, hp.rope_type, hp.n_ctx_orig_yarn as i32,
                                   hp.rope_freq_base_train_swa, 1.0, 0.0, 1.0, hp.yarn_beta_fast, hp.yarn_beta_slow);
            } else {
                q = rope(b, m, q, hp.rope_freq_base_train, hp.rope_freq_scale_train);
                k = rope(b, m, k, hp.rope_freq_base_train, hp.rope_freq_scale_train);
            }
        } else {
            q, k, v = build_qkv(b, m, ly, inpL, il);
            q = build_norm_rms(b, m, q, ly.attn_q_norm);
            k = build_norm_rms(b, m, k, ly.attn_k_norm);
            if hp.is_swa[il] || hp.swa_type == model::SWA_NONE {
                let factors = rope_factors(m, c, ly);
                let freq_base = if hp.is_swa[il] { hp.rope_freq_base_train_swa } else { hp.rope_freq_base_train };
                let freq_scale = if hp.is_swa[il] { hp.rope_freq_scale_train_swa } else { hp.rope_freq_scale_train };
                q = rope(b, m, q, freq_base, freq_scale, factors);
                k = rope(b, m, k, freq_base, freq_scale, factors);
            }
        }
        var cur = build_attn(b, m, c, il, ly.wo, ggml::NONE, q, k, v, 1.0 / sqrt(n_embd_head as f32));
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpSA = ggml::get_rows(ctx, inpSA, b.inp.out_ids);
        }
        cur = build_norm_rms(b, m, cur, ly.attn_post_norm);
        let ffn_inp = ggml::add(ctx, cur, inpSA);
        cur = build_ffn_swiglu(b, ly, ffn_inp);
        cur = build_norm_rms(b, m, cur, ly.ffn_post_norm);
        inpL = ggml::add(ctx, cur, ffn_inp);
    }
    let cur = build_norm_rms(b, m, inpL, m.output_norm);
    return ggml::mul_mat(ctx, m.output, cur);
}

// models/hunyuan-vl.cpp: llama's graph with Q and K normed after their rope.
fn build_hunyuan_dense(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_embd_head = hp.n_embd_head_v as i64;
    assert(n_embd_head == hp.n_embd_head_k && n_embd_head == hp.n_rot);
    let n_layer = hp.n_layer as i64;

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    for il in n_layer {
        let ly = m.layers[il];
        var inpSA = inpL;
        var cur = build_norm_rms(b, m, inpL, ly.attn_norm);
        let factors = rope_factors(m, c, ly);
        var q, k, v = build_qkv(b, m, ly, cur, il);
        q = rope(b, m, q, hp.rope_freq_base_train, hp.rope_freq_scale_train, factors);
        k = rope(b, m, k, hp.rope_freq_base_train, hp.rope_freq_scale_train, factors);
        k = build_norm_rms(b, m, k, ly.attn_k_norm);
        q = build_norm_rms(b, m, q, ly.attn_q_norm);
        cur = build_attn(b, m, c, il, ly.wo, ly.bo, q, k, v, 1.0 / sqrt(n_embd_head as f32));
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpSA = ggml::get_rows(ctx, inpSA, b.inp.out_ids);
        }
        let ffn_inp = ggml::add(ctx, cur, inpSA);
        cur = build_norm_rms(b, m, ffn_inp, ly.ffn_norm);
        cur = build_ffn_swiglu(b, ly, cur);
        inpL = ggml::add(ctx, cur, ffn_inp);
    }
    let cur = build_norm_rms(b, m, inpL, m.output_norm);
    return ggml::mul_mat(ctx, m.output, cur);
}

// models/gemma2.cpp
fn build_gemma2(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_layer = hp.n_layer as i64;

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    inpL = ggml::scale(ctx, inpL, sqrt(hp.n_embd as f32));
    for il in n_layer {
        let ly = m.layers[il];
        let freq_base = if hp.is_swa[il] { hp.rope_freq_base_train_swa } else { hp.rope_freq_base_train };
        let freq_scale = if hp.is_swa[il] { hp.rope_freq_scale_train_swa } else { hp.rope_freq_scale_train };

        var cur = build_norm_rms(b, m, inpL, ly.attn_norm);
        var q, k, v = build_qkv(b, m, ly, cur, il);
        q = rope(b, m, q, freq_base, freq_scale);
        k = rope(b, m, k, freq_base, freq_scale);
        q = ggml::scale(ctx, q, hp.f_attention_scale);
        cur = build_attn(b, m, c, il, ly.wo, ggml::NONE, q, k, v, 1.0);

        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpL = ggml::get_rows(ctx, inpL, b.inp.out_ids);
        }
        cur = build_norm_rms(b, m, cur, ly.attn_post_norm);
        let sa_out = ggml::add(ctx, cur, inpL);

        // feed-forward: GELU-gated, gate and up in parallel
        cur = build_norm_rms(b, m, sa_out, ly.ffn_norm);
        let up = ggml::mul_mat(ctx, ly.ffn_up, cur);
        let gate = ggml::mul_mat(ctx, ly.ffn_gate, cur);
        cur = ggml::mul_mat(ctx, ly.ffn_down, ggml::geglu_split(ctx, gate, up));
        cur = build_norm_rms(b, m, cur, ly.ffn_post_norm);
        inpL = ggml::add(ctx, cur, sa_out);
    }

    var cur = build_norm_rms(b, m, inpL, m.output_norm);
    cur = ggml::mul_mat(ctx, m.output, cur);
    // final logit soft-capping
    cur = ggml::scale(ctx, cur, 1.0 / hp.f_final_logit_softcapping);
    cur = ggml::tanh(ctx, cur);
    return ggml::scale(ctx, cur, hp.f_final_logit_softcapping);
}

// models/llama.cpp
fn build_llama(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_embd_head = hp.n_embd_head_v as i64;
    assert(n_embd_head == hp.n_embd_head_k && n_embd_head == hp.n_rot);
    let n_layer = hp.n_layer as i64;
    let kq_scale: f32 = if hp.f_attention_scale == 0.0 { 1.0 / sqrt(n_embd_head as f32) } else { hp.f_attention_scale };

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    for il in n_layer {
        let ly = m.layers[il];
        var inpSA = inpL;
        var cur = build_norm_rms(b, m, inpL, ly.attn_norm);

        if il == 0 { trace(b, "attn_norm-0", cur); }
        var q, k, v = build_qkv(b, m, ly, cur, il);
        q = rope(b, m, q, hp.rope_freq_base_train, hp.rope_freq_scale_train, ly.rope_freqs);
        k = rope(b, m, k, hp.rope_freq_base_train, hp.rope_freq_scale_train, ly.rope_freqs);
        if b.inp.attn_scale.p != 0 { q = ggml::mul(ctx, q, b.inp.attn_scale); }
        if il == 0 { trace(b, "Qcur-0", q); trace(b, "Kcur-0", k); }
        cur = build_attn(b, m, c, il, ly.wo, ly.bo, q, k, v, kq_scale);
        if il == 0 { trace(b, "attn_out-0", cur); }

        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpSA = ggml::get_rows(ctx, inpSA, b.inp.out_ids);
        }
        let ffn_inp = ggml::add(ctx, cur, inpSA);
        cur = build_norm_rms(b, m, ffn_inp, ly.ffn_norm);
        cur = build_ffn_swiglu(b, ly, cur);
        inpL = ggml::add(ctx, cur, ffn_inp);
        if il == 0 { trace(b, "l_out-0", inpL); }
        // nanbeige: the output norm between loops of its layers
        if hp.n_loops > 1 && (il + 1) % hp.n_layer_phys == 0 && il + 1 < n_layer && !hp.skip_loop_final_norm {
            inpL = build_norm_rms(b, m, inpL, m.output_norm);
        }
    }
    let cur = build_norm_rms(b, m, inpL, m.output_norm);
    return ggml::mul_mat(ctx, m.output, cur);
}

// models/qwen2.cpp, qwen3.cpp, qwen2vl.cpp, qwen3vl.cpp: llama's graph
// without rope factors; qwen3 norms Q and K before their rope, qwen2 may add
// an output bias, the VL models use M-RoPE and qwen3vl adds its deepstack
// inputs (zeros for text) to its first layers.
fn build_qwen(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_embd_head = hp.n_embd_head_v as i64;
    assert(n_embd_head == hp.n_embd_head_k && n_embd_head == hp.n_rot);
    let n_layer = hp.n_layer as i64;

    let multi = n_pos_per_embd(m) > 1;
    let n_embd = hp.n_embd as i64;
    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    var inp_embd = inpL;
    if hp.n_deepstack_layers > 0 {
        // the token rows padded to the input's width (build_inp_embd)
        inp_embd = ggml::pad_ext(ctx, inpL, (n_embd * hp.n_deepstack_layers) as i32, 0, 0, 0);
        inpL = ggml::view_2d(ctx, inp_embd, n_embd, b.n_tokens, ggml::nb(inp_embd, 1), 0);
        inp_embd = inpL;
    }
    for il in n_layer {
        let ly = m.layers[il];
        var inpSA = inpL;
        var cur = build_norm_rms(b, m, inpL, ly.attn_norm);

        var q, k, v = build_qkv(b, m, ly, cur, il);
        if ly.attn_q_norm.p != 0 { q = build_norm_rms(b, m, q, ly.attn_q_norm); }
        q = if multi { rope_multi(b, m, q) } else { rope(b, m, q, hp.rope_freq_base_train, hp.rope_freq_scale_train) };
        if ly.attn_k_norm.p != 0 { k = build_norm_rms(b, m, k, ly.attn_k_norm); }
        k = if multi { rope_multi(b, m, k) } else { rope(b, m, k, hp.rope_freq_base_train, hp.rope_freq_scale_train) };
        cur = build_attn(b, m, c, il, ly.wo, ly.bo, q, k, v, 1.0 / sqrt(n_embd_head as f32));

        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpSA = ggml::get_rows(ctx, inpSA, b.inp.out_ids);
        }
        let ffn_inp = ggml::add(ctx, cur, inpSA);
        cur = build_norm_rms(b, m, ffn_inp, ly.ffn_norm);
        cur = build_ffn_swiglu(b, ly, cur);
        inpL = ggml::add(ctx, cur, ffn_inp);
        if il < hp.n_deepstack_layers {
            let ds = ggml::view_2d(ctx, inp_embd, n_embd, b.n_tokens, ggml::nb(inp_embd, 1), (il + 1) * n_embd * 4);
            inpL = ggml::add(ctx, inpL, ds);
        }
    }
    let cur = build_norm_rms(b, m, inpL, m.output_norm);
    return add_bias(b, ggml::mul_mat(ctx, m.output, cur), m.output_b);
}

// A mamba layer (build_mamba_layer, one sequence in the one state slot).
fn build_mamba_layer(b: Build&, m: model::Model&, c: Context&, ly: model::Layer, cur: ggml::Tensor, il: i64) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let d_conv = hp.ssm_d_conv as i64;
    let d_inner = hp.ssm_d_inner as i64;
    let d_state = hp.ssm_d_state as i64;
    let dt_rank = hp.ssm_dt_rank as i64;
    let n_head = d_inner;
    let head_dim: i64 = 1;
    let n_tokens = b.n_tokens;
    let conv_states_all = c.r[il];
    let ssm_states_all = c.s[il];

    var conv = ggml::get_rows(ctx, conv_states_all, b.inp.s_copy);
    conv = ggml::reshape_3d(ctx, conv, d_conv - 1, d_inner, 1);
    let x3 = ggml::reshape_3d(ctx, cur, ggml::ne(cur, 0), n_tokens, 1);
    let xz = ggml::mul_mat(ctx, ly.ssm_in, x3);       // {2*d_inner, n_tokens, 1}
    var x = ggml::view_3d(ctx, xz, d_inner, ggml::ne(xz, 1), ggml::ne(xz, 2), ggml::nb(xz, 1), ggml::nb(xz, 2), 0);
    let z = ggml::view_3d(ctx, xz, d_inner, ggml::ne(xz, 1), ggml::ne(xz, 2), ggml::nb(xz, 1), ggml::nb(xz, 2),
                          d_inner * ggml::element_size(xz));

    // the convolution; its last d_conv - 1 inputs back into the state
    let conv_x = ggml::concat(ctx, conv, ggml::transpose(ctx, x), 0);
    let last_conv = ggml::view_3d(ctx, conv_x, d_conv - 1, d_inner, 1, ggml::nb(conv_x, 1), ggml::nb(conv_x, 2),
                                  n_tokens * ggml::nb(conv_x, 0));
    ggml::build_forward_expand(b.gf, ggml::cpy(ctx, last_conv, ggml::view_1d(ctx, conv_states_all, (d_conv - 1) * d_inner, 0)));
    x = ggml::ssm_conv(ctx, conv_x, ly.ssm_conv1d);
    x = ggml::add(ctx, x, ly.ssm_conv1d_b);
    x = ggml::silu(ctx, x);

    // the state space model
    let x_db = ggml::mul_mat(ctx, ly.ssm_x, x);
    var dt = ggml::view_3d(ctx, x_db, dt_rank, n_tokens, 1, ggml::nb(x_db, 1), ggml::nb(x_db, 2), 0);
    var bm = ggml::view_4d(ctx, x_db, d_state, 1, n_tokens, 1, d_state * ggml::nb(x_db, 0), ggml::nb(x_db, 1), ggml::nb(x_db, 2),
                           ggml::element_size(x_db) * dt_rank);
    var cm = ggml::view_4d(ctx, x_db, d_state, 1, n_tokens, 1, d_state * ggml::nb(x_db, 0), ggml::nb(x_db, 1), ggml::nb(x_db, 2),
                           ggml::element_size(x_db) * (dt_rank + d_state));
    // FalconMamba and jamba norm dt, B and C
    if hp.ssm_dt_b_c_rms || (ly.ssm_dt_norm.p != 0 && ly.ssm_b_norm.p != 0 && ly.ssm_c_norm.p != 0) {
        dt = build_norm_rms(b, m, dt, ly.ssm_dt_norm);
        bm = build_norm_rms(b, m, bm, ly.ssm_b_norm);
        cm = build_norm_rms(b, m, cm, ly.ssm_c_norm);
    } else {
        dt = ggml::cont(ctx, dt);
    }
    dt = ggml::mul_mat(ctx, ly.ssm_dt, dt);
    dt = ggml::add(ctx, dt, ly.ssm_dt_b);
    let x_in = x;
    x = ggml::reshape_4d(ctx, x, head_dim, n_head, n_tokens, 1);
    let ssm = ggml::reshape_4d(ctx, ssm_states_all, d_state, head_dim, n_head, 1);
    let y_ssm = ggml::ssm_scan(ctx, ssm, x, dt, ly.ssm_a, bm, cm, b.inp.s_copy, 1);
    ggml::build_forward_expand(b.gf, y_ssm);
    // the last state back into the cache
    ggml::build_forward_expand(b.gf, ggml::cpy(ctx, ggml::view_1d(ctx, y_ssm, d_state * d_inner, ggml::nb(x, 3) * ggml::ne(x, 3)),
                                               ggml::view_1d(ctx, ssm_states_all, d_state * d_inner, 0)));
    var y = ggml::view_3d(ctx, y_ssm, d_inner, n_tokens, 1, ggml::nb(x, 2), ggml::nb(x, 3), 0);
    y = ggml::add(ctx, y, ggml::mul(ctx, x_in, ly.ssm_d));
    y = ggml::swiglu_split(ctx, ggml::cont(ctx, z), y);
    let out = ggml::mul_mat(ctx, ly.ssm_out, y);
    return ggml::reshape_2d(ctx, out, ggml::ne(out, 0), n_tokens);
}

// A mamba2 layer (build_mamba2_layer, one sequence in the one state slot).
fn build_mamba2_layer(b: Build&, m: model::Model&, c: Context&, ly: model::Layer, cur: ggml::Tensor, il: i64) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let d_conv = hp.ssm_d_conv as i64;
    let d_inner = hp.ssm_d_inner as i64;
    let d_state = hp.ssm_d_state as i64;
    let n_head = hp.ssm_dt_rank as i64;
    let head_dim = d_inner / n_head;
    let n_group = hp.ssm_n_group as i64;
    let n_tokens = b.n_tokens;
    let conv_states_all = c.r[il];
    let ssm_states_all = c.s[il];
    let conv_dim = d_inner + 2 * n_group * d_state;

    var conv = ggml::get_rows(ctx, conv_states_all, b.inp.s_copy);
    conv = ggml::reshape_3d(ctx, conv, d_conv - 1, conv_dim, 1);
    var zxBCdt = ggml::mul_mat(ctx, ly.ssm_in, cur);
    zxBCdt = ggml::reshape_3d(ctx, zxBCdt, ggml::ne(zxBCdt, 0), n_tokens, 1);
    let es = ggml::element_size(zxBCdt);
    let z = ggml::view_4d(ctx, zxBCdt, head_dim, n_head, n_tokens, 1, head_dim * ggml::nb(zxBCdt, 0),
                          ggml::nb(zxBCdt, 1), ggml::nb(zxBCdt, 2), 0);
    var xBC = ggml::view_3d(ctx, zxBCdt, conv_dim, n_tokens, 1, ggml::nb(zxBCdt, 1), ggml::nb(zxBCdt, 2), d_inner * es);
    var dt = ggml::view_3d(ctx, zxBCdt, n_head, n_tokens, 1, ggml::nb(zxBCdt, 1), ggml::nb(zxBCdt, 2), (2 * d_inner + 2 * n_group * d_state) * es);

    // the convolution; its last d_conv - 1 inputs back into the state
    let conv_x = ggml::concat(ctx, conv, ggml::transpose(ctx, xBC), 0);
    let row_count = (d_conv - 1) * conv_dim;
    let last_conv = ggml::view_3d(ctx, conv_x, d_conv - 1, conv_dim, 1, ggml::nb(conv_x, 1), ggml::nb(conv_x, 2),
                                  n_tokens * ggml::nb(conv_x, 0));
    ggml::build_forward_expand(b.gf, ggml::cpy(ctx, last_conv, ggml::view_2d(ctx, conv_states_all, row_count, 1, ggml::nb(conv_states_all, 1), 0)));
    xBC = ggml::ssm_conv(ctx, conv_x, ly.ssm_conv1d);
    xBC = ggml::add(ctx, xBC, ly.ssm_conv1d_b);
    xBC = ggml::silu(ctx, xBC);

    // the state space model: x, B and C are V, K and Q
    let ex = ggml::element_size(xBC);
    let x = ggml::view_4d(ctx, xBC, head_dim, n_head, n_tokens, 1, head_dim * ggml::nb(xBC, 0), ggml::nb(xBC, 1), ggml::nb(xBC, 2), 0);
    let bm = ggml::view_4d(ctx, xBC, d_state, n_group, n_tokens, 1, d_state * ggml::nb(xBC, 0), ggml::nb(xBC, 1), ggml::nb(xBC, 2), d_inner * ex);
    let cm = ggml::view_4d(ctx, xBC, d_state, n_group, n_tokens, 1, d_state * ggml::nb(xBC, 0), ggml::nb(xBC, 1), ggml::nb(xBC, 2),
                           (d_inner + n_group * d_state) * ex);
    dt = ggml::add(ctx, ggml::cont(ctx, dt), ly.ssm_dt_b);
    let ssm = ggml::reshape_4d(ctx, ssm_states_all, d_state, head_dim, n_head, ggml::ne(ssm_states_all, 1));
    let y_ssm = ggml::ssm_scan(ctx, ssm, x, dt, ly.ssm_a, bm, cm, b.inp.s_copy, 1);
    ggml::build_forward_expand(b.gf, y_ssm);
    // the last state back into the cache
    let n_d = d_state * d_inner;
    let y_row = ggml::row_size(ggml::tensor_type(y_ssm), n_d);
    ggml::build_forward_expand(b.gf, ggml::cpy(ctx,
        ggml::view_3d(ctx, y_ssm, n_d, 1, 1, y_row, y_row, ggml::nelements(x) * ggml::element_size(x)),
        ggml::view_3d(ctx, ssm_states_all, n_d, 1, 1, ggml::nb(ssm_states_all, 1), ggml::row_size(ggml::tensor_type(ssm_states_all), n_d), 0)));
    var y = ggml::view_4d(ctx, y_ssm, head_dim, n_head, n_tokens, 1, ggml::nb(x, 1), n_head * ggml::nb(x, 1),
                          n_tokens * n_head * ggml::nb(x, 1), 0);
    y = ggml::add(ctx, y, ggml::mul(ctx, x, ly.ssm_d));
    y = ggml::swiglu_split(ctx, ggml::cont(ctx, z), y);
    // grouped RMS norm
    if ly.ssm_norm.p != 0 {
        y = ggml::reshape_4d(ctx, y, d_inner / n_group, n_group, n_tokens, 1);
        y = build_norm_rms(b, m, y, ly.ssm_norm);
    }
    y = ggml::reshape_2d(ctx, y, d_inner, n_tokens);
    return ggml::mul_mat(ctx, ly.ssm_out, y);
}

// models/falcon-h1.cpp, granite-hybrid.cpp, jamba.cpp
fn build_hybrid(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_layer = hp.n_layer as i64;
    let n_embd_head = hp.n_embd_head_v as i64;
    let kq_scale: f32 = if hp.f_attention_scale == 0.0 { 1.0 / sqrt(n_embd_head as f32) } else { hp.f_attention_scale };
    let arch = hp.arch;

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    if hp.f_embedding_scale != 0.0 { inpL = ggml::scale(ctx, inpL, hp.f_embedding_scale); }
    for il in n_layer {
        let ly = m.layers[il];
        var inpSA = inpL;
        var cur = build_norm_rms(b, m, inpL, ly.attn_norm);
        if arch == model::ARCH_FALCON_H1 {
            // attention and mamba2, both on the normed input, added
            var q, k, v = build_qkv(b, m, ly, cur, il);
            q = rope(b, m, q, hp.rope_freq_base_train, hp.rope_freq_scale_train);
            k = rope(b, m, k, hp.rope_freq_base_train, hp.rope_freq_scale_train);
            let attn_out = build_attn(b, m, c, il, ly.wo, ggml::NONE, q, k, v, kq_scale);
            cur = build_norm_rms(b, m, inpL, ly.attn_norm);
            let ssm_out = build_mamba2_layer(b, m, c, ly, cur, il);
            cur = ggml::add(ctx, attn_out, ssm_out);
            inpSA = ggml::add(ctx, cur, inpSA);
            if il == n_layer - 1 {
                cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
                inpSA = ggml::get_rows(ctx, inpSA, b.inp.out_ids);
            }
            cur = build_norm_rms(b, m, inpSA, ly.ffn_norm);
            cur = build_ffn_swiglu(b, ly, cur);
            inpL = ggml::add(ctx, cur, inpSA);
            continue;
        }
        if hp.is_recr[il] {
            cur = if arch == model::ARCH_JAMBA { build_mamba_layer(b, m, c, ly, cur, il) } else { build_mamba2_layer(b, m, c, ly, cur, il) };
        } else {
            var q, k, v = build_qkv(b, m, ly, cur, il);
            if arch == model::ARCH_GRANITE_HYBRID && hp.rope_finetuned {
                let factors = rope_factors(m, c, ly);
                q = rope(b, m, q, hp.rope_freq_base_train, hp.rope_freq_scale_train, factors);
                k = rope(b, m, k, hp.rope_freq_base_train, hp.rope_freq_scale_train, factors);
            }
            cur = build_attn(b, m, c, il, ly.wo, ly.bo, q, k, v, kq_scale);
        }
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpSA = ggml::get_rows(ctx, inpSA, b.inp.out_ids);
        }
        if hp.f_residual_scale != 0.0 { cur = ggml::scale(ctx, cur, hp.f_residual_scale); }
        let ffn_inp = ggml::add(ctx, cur, inpSA);
        cur = build_norm_rms(b, m, ffn_inp, ly.ffn_norm);
        if ly.ffn_gate_inp.p != 0 {
            let jamba = arch == model::ARCH_JAMBA;
            let moe_out = build_moe_ffn(b, m, ly, cur, FFN_SILU, !jamba, hp.expert_weights_scale, model::GATING_SOFTMAX);
            cur = if !jamba && hp.n_ff_shexp > 0 {
                ggml::add(ctx, moe_out, build_ffn_swiglu(b, model::Layer { ffn_up: ly.ffn_up_shexp, ffn_gate: ly.ffn_gate_shexp,
                                                                           ffn_down: ly.ffn_down_shexp, .. }, cur))
            } else { moe_out };
        } else {
            cur = build_ffn_swiglu(b, ly, cur);
        }
        if hp.f_residual_scale != 0.0 { cur = ggml::scale(ctx, cur, hp.f_residual_scale); }
        inpL = ggml::add(ctx, ffn_inp, cur);
    }
    var cur = build_norm_rms(b, m, inpL, m.output_norm);
    cur = ggml::mul_mat(ctx, m.output, cur);
    if hp.f_logit_scale != 0.0 { cur = ggml::scale(ctx, cur, 1.0 / hp.f_logit_scale); }
    return cur;
}

// models/minicpm3.cpp
fn build_minicpm3(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_layer = hp.n_layer as i64;
    let n_tokens = b.n_tokens;
    let n_embd_base: f32 = 256.0;
    let scale_embd: f32 = 12.0;
    let scale_depth: f32 = 1.4;
    let head_k = hp.n_embd_head_k as i64;
    let head_v = hp.n_embd_head_v as i64;
    let kq_scale = 1.0 / sqrt(head_k as f32);
    let n_rope = hp.n_rot as i64;
    let n_nope = head_k - n_rope;
    let kv_lora_rank = hp.n_lora_kv as i64;
    let n_layer_f: f32 = n_layer;
    let scale_res = scale_depth / sqrt(n_layer_f);

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    inpL = ggml::scale(ctx, inpL, scale_embd);
    for il in n_layer {
        let ly = m.layers[il];
        let n_head = hp.n_head[il] as i64;
        var inpSA = inpL;
        let factors = rope_factors(m, c, ly);
        var cur = build_norm_rms(b, m, inpL, ly.attn_norm);

        var q = ggml::mul_mat(ctx, ly.wq_a, cur);
        q = build_norm_rms(b, m, q, ly.attn_q_a_norm);
        q = ggml::mul_mat(ctx, ly.wq_b, q);
        q = ggml::reshape_3d(ctx, q, head_k, n_head, n_tokens);

        let kv_pe = ggml::mul_mat(ctx, ly.wkv_a_mqa, cur);
        var kv_compressed = ggml::view_2d(ctx, kv_pe, kv_lora_rank, n_tokens, ggml::nb(kv_pe, 1), 0);
        var k_pe = ggml::view_3d(ctx, kv_pe, n_rope, 1, n_tokens, ggml::nb(kv_pe, 1), ggml::nb(kv_pe, 1),
                                 ggml::row_size(ggml::tensor_type(kv_pe), kv_lora_rank));
        kv_compressed = build_norm_rms(b, m, kv_compressed, ly.attn_kv_a_norm);
        let kv = ggml::mul_mat(ctx, ly.wkv_b, kv_compressed);
        let ty = ggml::tensor_type(kv);
        let k_nope = ggml::view_3d(ctx, kv, n_nope, n_head, n_tokens, ggml::row_size(ty, n_nope + head_v),
                                   ggml::row_size(ty, n_head * (n_nope + head_v)), 0);
        var v_states = ggml::view_3d(ctx, kv, head_v, n_head, n_tokens, ggml::row_size(ty, n_nope + head_v),
                                     ggml::row_size(ty, (n_nope + head_v) * n_head), ggml::row_size(ty, n_nope));
        v_states = ggml::cont(ctx, v_states);

        // rope the last n_rot dimensions of q, and k_pe
        q = rope(b, m, q, hp.rope_freq_base_train, hp.rope_freq_scale_train, factors);
        q = ggml::rope_set_offset(q, n_nope as i32);
        k_pe = rope(b, m, k_pe, hp.rope_freq_base_train, hp.rope_freq_scale_train, factors);
        let k_states = ggml::concat(ctx, k_nope, ggml::repeat_4d(ctx, k_pe, n_rope, n_head, n_tokens, 1), 0);
        cur = build_attn(b, m, c, il, ly.wo, ggml::NONE, q, k_states, v_states, kq_scale);
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpSA = ggml::get_rows(ctx, inpSA, b.inp.out_ids);
        }
        cur = ggml::scale(ctx, cur, scale_res);
        let ffn_inp = ggml::add(ctx, cur, inpSA);
        cur = build_norm_rms(b, m, ffn_inp, ly.ffn_norm);
        cur = build_ffn_swiglu(b, ly, cur);
        cur = ggml::scale(ctx, cur, scale_res);
        inpL = ggml::add(ctx, cur, ffn_inp);
    }
    var cur = build_norm_rms(b, m, inpL, m.output_norm);
    let n_embd_f: f32 = hp.n_embd;
    cur = ggml::scale(ctx, cur, n_embd_base / n_embd_f);
    return ggml::mul_mat(ctx, m.output, cur);
}

// rwkv6's attention (build_rwkv6_time_mix, one sequence).
fn build_rwkv6_time_mix(b: Build&, m: model::Model&, c: Context&, ly: model::Layer, cur_in: ggml::Tensor, x_prev: ggml::Tensor, il: i64) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_tokens = b.n_tokens;
    let n_embd = hp.n_embd as i64;
    let head_size = hp.wkv_head_size as i64;
    let n_head = n_embd / head_size;
    let n_head_kv = hp.n_head_kv[il] as i64;
    let is_qrwkv = ly.time_mix_first.p == 0;
    let f32s: i64 = 4;

    var sx = ggml::sub(ctx, x_prev, cur_in);
    sx = ggml::reshape_2d(ctx, sx, n_embd, n_tokens);
    var cur = ggml::reshape_2d(ctx, cur_in, n_embd, n_tokens);
    var xxx = ggml::add(ctx, ggml::mul(ctx, sx, ly.time_mix_lerp_x), cur);
    xxx = ggml::reshape_4d(ctx, ggml::tanh(ctx, ggml::mul_mat(ctx, ly.time_mix_w1, xxx)), ggml::ne(ly.time_mix_w1, 1) / 5, 1, 5, n_tokens);
    xxx = ggml::cont(ctx, ggml::permute(ctx, xxx, 0, 1, 3, 2));
    xxx = ggml::mul_mat(ctx, ggml::reshape_4d(ctx, ly.time_mix_w2, ggml::ne(ly.time_mix_w2, 0), ggml::ne(ly.time_mix_w2, 1), 1, 5), xxx);
    fn part(t: ggml::Tensor, i: i64) -> ggml::Tensor { ggml::view_2d(ctx, t, n_embd, n_tokens, ggml::nb(t, 1), n_embd * n_tokens * i * f32s) }
    var xw = ggml::NONE;
    var xk = ggml::NONE;
    var xv = ggml::NONE;
    var xr = ggml::NONE;
    var xg = ggml::NONE;
    if ly.time_mix_lerp_fused.p != 0 {
        sx = ggml::reshape_3d(ctx, sx, n_embd, 1, n_tokens);
        cur = ggml::reshape_3d(ctx, cur, n_embd, 1, n_tokens);
        xxx = ggml::add(ctx, ggml::mul(ctx, ggml::add(ctx, xxx, ly.time_mix_lerp_fused), sx), cur);
        xw = part(xxx, 0); xk = part(xxx, 1); xv = part(xxx, 2); xr = part(xxx, 3); xg = part(xxx, 4);
    } else {
        fn lerp(x: ggml::Tensor, l: ggml::Tensor) -> ggml::Tensor { ggml::add(ctx, ggml::mul(ctx, ggml::add(ctx, x, l), sx), cur) }
        xw = lerp(part(xxx, 0), ly.time_mix_lerp_w);
        xk = lerp(part(xxx, 1), ly.time_mix_lerp_k);
        xv = lerp(part(xxx, 2), ly.time_mix_lerp_v);
        xr = lerp(part(xxx, 3), ly.time_mix_lerp_r);
        xg = lerp(part(xxx, 4), ly.time_mix_lerp_g);
    }
    var r = add_bias(b, ggml::mul_mat(ctx, ly.time_mix_receptance, xr), ly.time_mix_receptance_b);
    var k = add_bias(b, ggml::mul_mat(ctx, ly.time_mix_key, xk), ly.time_mix_key_b);
    var v = add_bias(b, ggml::mul_mat(ctx, ly.time_mix_value, xv), ly.time_mix_value_b);
    var g = ggml::mul_mat(ctx, ly.time_mix_gate, xg);
    g = if is_qrwkv { ggml::sigmoid(ctx, g) } else { ggml::silu(ctx, g) };
    if n_head_kv != 0 && n_head_kv != n_head {
        assert(n_head % n_head_kv == 0);
        k = ggml::reshape_4d(ctx, k, head_size, 1, n_head_kv, n_tokens);
        v = ggml::reshape_4d(ctx, v, head_size, 1, n_head_kv, n_tokens);
        let tmp = ggml::new_tensor_4d(ctx, ggml::TYPE_F32, head_size, n_head / n_head_kv, n_head_kv, n_tokens);
        k = ggml::repeat(ctx, k, tmp);
        v = ggml::repeat(ctx, v, tmp);
    }
    k = ggml::reshape_3d(ctx, k, head_size, n_head, n_tokens);
    v = ggml::reshape_3d(ctx, v, head_size, n_head, n_tokens);
    r = ggml::reshape_3d(ctx, r, head_size, n_head, n_tokens);
    var w = ggml::mul_mat(ctx, ly.time_mix_decay_w2, ggml::tanh(ctx, ggml::mul_mat(ctx, ly.time_mix_decay_w1, xw)));
    w = ggml::add(ctx, w, ly.time_mix_decay);
    w = ggml::exp(ctx, ggml::neg(ctx, ggml::exp(ctx, w)));
    w = ggml::reshape_3d(ctx, w, head_size, n_head, n_tokens);
    if is_qrwkv { k = ggml::sub(ctx, k, ggml::mul(ctx, k, w)); }

    let states = c.s[il];
    let wkv_state = ggml::get_rows(ctx, states, b.inp.s_copy);
    let hs: f64 = head_size;
    let neg_half: f64 = -0.5;
    let wkv_output = if is_qrwkv { ggml::gated_linear_attn(ctx, k, v, r, w, wkv_state, pow(hs, neg_half) as f32) }
                     else { ggml::rwkv_wkv6(ctx, k, v, r, ly.time_mix_first, w, wkv_state) };
    cur = ggml::view_1d(ctx, wkv_output, n_embd * n_tokens, 0);
    let new_state = ggml::view_1d(ctx, wkv_output, n_embd * head_size, n_embd * n_tokens * f32s);
    ggml::build_forward_expand(b.gf, ggml::cpy(ctx, new_state, ggml::view_1d(ctx, states, model::n_embd_s(hp), 0)));
    if !is_qrwkv {
        cur = ggml::reshape_3d(ctx, cur, n_embd / n_head, n_head, n_tokens);
        cur = ggml::norm(ctx, cur, 64e-5);
        cur = ggml::reshape_2d(ctx, cur, n_embd, n_tokens);
        cur = ggml::add(ctx, ggml::mul(ctx, cur, ly.time_mix_ln), ly.time_mix_ln_b);
    } else {
        cur = ggml::reshape_2d(ctx, cur, n_embd, n_tokens);
    }
    cur = ggml::mul(ctx, cur, g);
    cur = ggml::mul_mat(ctx, ly.time_mix_output, cur);
    return ggml::reshape_3d(ctx, cur, n_embd, n_tokens, 1);
}

// rwkv7's attention (build_rwkv7_time_mix, one sequence); v_first: the
// first layer's values, which the later layers mix in.
fn build_rwkv7_time_mix(b: Build&, m: model::Model&, c: Context&, ly: model::Layer, cur_in: ggml::Tensor, x_prev: ggml::Tensor,
                        v_first: ggml::Tensor&, il: i64) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_tokens = b.n_tokens;
    let n_embd = hp.n_embd as i64;
    let head_size = hp.wkv_head_size as i64;
    let head_count = n_embd / head_size;
    let f32s: i64 = 4;
    let has_gating = ly.time_mix_g1.p != 0 && ly.time_mix_g2.p != 0;

    var sx = ggml::sub(ctx, x_prev, cur_in);
    let dummy = ggml::new_tensor_4d(ctx, ggml::TYPE_F32, n_embd, n_tokens, 1, if has_gating { 6 } else { 5 });
    sx = ggml::repeat(ctx, sx, dummy);
    let xxx = ggml::add(ctx, ggml::mul(ctx, sx, ly.time_mix_lerp_fused), cur_in);
    fn part(i: i64) -> ggml::Tensor { ggml::view_2d(ctx, xxx, n_embd, n_tokens, ggml::nb(xxx, 1), n_embd * n_tokens * i * f32s) }
    let xr = part(0);
    let xw = part(1);
    let xk = part(2);
    let xv = part(3);
    let xa = part(4);
    let xg = if has_gating { part(5) } else { ggml::NONE };

    var r = ggml::mul_mat(ctx, ly.time_mix_receptance, xr);
    var w = ggml::add(ctx, ggml::mul_mat(ctx, ly.time_mix_w2, ggml::tanh(ctx, ggml::mul_mat(ctx, ly.time_mix_w1, xw))), ly.time_mix_w0);
    w = ggml::exp(ctx, ggml::scale(ctx, ggml::sigmoid(ctx, w), -0.606531));
    var k = ggml::mul_mat(ctx, ly.time_mix_key, xk);
    var v = ggml::mul_mat(ctx, ly.time_mix_value, xv);
    if v_first.p == 0 {
        v_first = v;
    } else {
        let mix = ggml::sigmoid(ctx, ggml::add(ctx, ggml::mul_mat(ctx, ly.time_mix_v2, ggml::mul_mat(ctx, ly.time_mix_v1, xv)), ly.time_mix_v0));
        v = ggml::add(ctx, v, ggml::mul(ctx, ggml::sub(ctx, v_first, v), mix));
    }
    let g = if has_gating { ggml::mul_mat(ctx, ly.time_mix_g2, ggml::sigmoid(ctx, ggml::mul_mat(ctx, ly.time_mix_g1, xg))) } else { ggml::NONE };
    var a = ggml::sigmoid(ctx, ggml::add(ctx, ggml::mul_mat(ctx, ly.time_mix_a2, ggml::mul_mat(ctx, ly.time_mix_a1, xa)), ly.time_mix_a0));
    var kk = ggml::reshape_3d(ctx, ggml::mul(ctx, k, ly.time_mix_k_k), head_size, head_count, n_tokens);
    kk = ggml::l2_norm(ctx, kk, 1e-12);
    let ka = ggml::mul(ctx, k, ly.time_mix_k_a);
    k = ggml::add(ctx, k, ggml::sub(ctx, ggml::mul(ctx, a, ka), ka));
    r = ggml::reshape_3d(ctx, r, head_size, head_count, n_tokens);
    w = ggml::reshape_3d(ctx, w, head_size, head_count, n_tokens);
    k = ggml::reshape_3d(ctx, k, head_size, head_count, n_tokens);
    v = ggml::reshape_3d(ctx, v, head_size, head_count, n_tokens);
    a = ggml::reshape_3d(ctx, a, head_size, head_count, n_tokens);

    let states = c.s[il];
    let wkv_state = ggml::get_rows(ctx, states, b.inp.s_copy);
    let wkv_output = ggml::rwkv_wkv7(ctx, r, w, k, v, ggml::neg(ctx, kk), ggml::mul(ctx, kk, a), wkv_state);
    var cur = ggml::view_1d(ctx, wkv_output, n_embd * n_tokens, 0);
    let new_state = ggml::view_1d(ctx, wkv_output, n_embd * head_size, n_embd * n_tokens * f32s);
    ggml::build_forward_expand(b.gf, ggml::cpy(ctx, new_state, ggml::view_1d(ctx, states, model::n_embd_s(hp), 0)));
    if ly.time_mix_ln.p != 0 && ly.time_mix_ln_b.p != 0 {
        cur = ggml::reshape_3d(ctx, cur, n_embd / head_count, head_count, n_tokens);
        cur = ggml::norm(ctx, cur, 64e-5);
        cur = ggml::reshape_2d(ctx, cur, n_embd, n_tokens);
        cur = ggml::add(ctx, ggml::mul(ctx, cur, ly.time_mix_ln), ly.time_mix_ln_b);
    } else {
        cur = ggml::reshape_2d(ctx, cur, n_embd, n_tokens);
    }
    let rk = ggml::sum_rows(ctx, ggml::mul(ctx, ggml::mul(ctx, k, r), ggml::reshape_2d(ctx, ly.time_mix_r_k, head_size, head_count)));
    cur = ggml::add(ctx, cur, ggml::reshape_2d(ctx, ggml::mul(ctx, v, rk), n_embd, n_tokens));
    if has_gating { cur = ggml::mul(ctx, cur, g); }
    cur = ggml::mul_mat(ctx, ly.time_mix_output, cur);
    return ggml::reshape_3d(ctx, cur, n_embd, n_tokens, 1);
}

// models/rwkv6.cpp, models/rwkv7.cpp: each layer keeps its last two normed
// inputs (the token shift) and its attention's state.
fn build_rwkv(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_layer = hp.n_layer as i64;
    let n_embd = hp.n_embd as i64;
    let n_tokens = b.n_tokens;
    let rwkv7 = hp.arch == model::ARCH_RWKV7;
    var v_first = ggml::NONE;

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    inpL = build_norm_ln(b, m, inpL, m.tok_norm, m.tok_norm_b);
    for il in n_layer {
        let ly = m.layers[il];
        inpL = ggml::reshape_3d(ctx, inpL, n_embd, n_tokens, 1);
        // build_rwkv_token_shift_load
        var token_shift = ggml::get_rows(ctx, c.r[il], b.inp.s_copy);
        token_shift = ggml::reshape_3d(ctx, token_shift, n_embd, hp.token_shift_count, 1);
        let att_shift = ggml::view_3d(ctx, token_shift, n_embd, 1, 1, ggml::nb(token_shift, 1), ggml::nb(token_shift, 2), 0);
        let ffn_shift = ggml::view_3d(ctx, token_shift, n_embd, 1, 1, ggml::nb(token_shift, 1), ggml::nb(token_shift, 2),
                                      n_embd * ggml::element_size(token_shift));
        let att_norm = build_norm_ln(b, m, inpL, ly.attn_norm, ly.attn_norm_b);
        var x_prev = ggml::concat(ctx, att_shift,
                                  ggml::view_3d(ctx, att_norm, n_embd, n_tokens - 1, 1, ggml::nb(att_norm, 1), ggml::nb(att_norm, 2), 0), 1);
        var cur = if rwkv7 { build_rwkv7_time_mix(b, m, c, ly, att_norm, x_prev, v_first, il) }
                  else { build_rwkv6_time_mix(b, m, c, ly, att_norm, x_prev, il) };
        var ffn_inp = ggml::add(ctx, cur, inpL);
        var ffn_norm = build_norm_ln(b, m, ffn_inp, ly.attn_norm_2, ly.attn_norm_2_b);
        x_prev = ggml::concat(ctx, ffn_shift,
                              ggml::view_3d(ctx, ffn_norm, n_embd, n_tokens - 1, 1, ggml::nb(ffn_norm, 1), ggml::nb(ffn_norm, 2), 0), 1);
        token_shift = ggml::concat(ctx,
            ggml::view_3d(ctx, att_norm, n_embd, 1, 1, ggml::nb(att_norm, 1), ggml::nb(att_norm, 2), (n_tokens - 1) * n_embd * ggml::element_size(att_norm)),
            ggml::view_3d(ctx, ffn_norm, n_embd, 1, 1, ggml::nb(ffn_norm, 1), ggml::nb(ffn_norm, 2), (n_tokens - 1) * n_embd * ggml::element_size(ffn_norm)),
            1);
        // build_rwkv_token_shift_store
        ggml::build_forward_expand(b.gf, ggml::cpy(ctx, ggml::view_1d(ctx, token_shift, n_embd * hp.token_shift_count, 0),
                                                   ggml::view_1d(ctx, c.r[il], model::n_embd_r(hp), 0)));
        ffn_inp = ggml::reshape_2d(ctx, ffn_inp, n_embd, n_tokens);
        ffn_norm = ggml::reshape_2d(ctx, ffn_norm, n_embd, n_tokens);
        x_prev = ggml::reshape_2d(ctx, x_prev, n_embd, n_tokens);
        cur = ggml::reshape_2d(ctx, cur, n_embd, n_tokens);
        if il == n_layer - 1 {
            ffn_inp = ggml::get_rows(ctx, ffn_inp, b.inp.out_ids);
            ffn_norm = ggml::get_rows(ctx, ffn_norm, b.inp.out_ids);
            x_prev = ggml::get_rows(ctx, x_prev, b.inp.out_ids);
            if !rwkv7 { cur = ggml::get_rows(ctx, cur, b.inp.out_ids); }
        }
        // the channel mix
        let sx = ggml::sub(ctx, x_prev, ffn_norm);
        let xk = ggml::add(ctx, ggml::mul(ctx, sx, ly.channel_mix_lerp_k), ffn_norm);
        let kx = ggml::sqr(ctx, ggml::relu(ctx, ggml::mul_mat(ctx, ly.channel_mix_key, xk)));
        if rwkv7 {
            cur = ggml::mul_mat(ctx, ly.channel_mix_value, kx);
        } else {
            let xr = ggml::add(ctx, ggml::mul(ctx, sx, ly.channel_mix_lerp_r), ffn_norm);
            let rr = ggml::sigmoid(ctx, ggml::mul_mat(ctx, ly.channel_mix_receptance, xr));
            cur = ggml::mul(ctx, rr, ggml::mul_mat(ctx, ly.channel_mix_value, kx));
        }
        cur = ggml::add(ctx, cur, ffn_inp);
        if !rwkv7 && hp.rescale_every_n_layers != 0 && (il + 1) % hp.rescale_every_n_layers == 0 {
            cur = ggml::scale(ctx, cur, 0.5);
        }
        inpL = cur;
    }
    let cur = build_norm_ln(b, m, inpL, m.output_norm, m.output_norm_b);
    return ggml::mul_mat(ctx, m.output, cur);
}

// plamo2's mamba layer (build_plamo2_mamba_layer, one sequence).
fn build_plamo2_mamba_layer(b: Build&, m: model::Model&, c: Context&, ly: model::Layer, cur: ggml::Tensor, il: i64) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let d_conv = hp.ssm_d_conv as i64;
    let d_inner = hp.ssm_d_inner as i64;
    let d_state = hp.ssm_d_state as i64;
    let n_heads = hp.ssm_dt_rank as i64;
    let head_dim = d_inner / n_heads;
    let n_tokens = b.n_tokens;
    guard hp.ssm_n_group == 0 else { abort("plamo2: ssm.group_count must be 0"); }
    let conv_states_all = c.r[il];
    let ssm_states_all = c.s[il];

    var conv = ggml::get_rows(ctx, conv_states_all, b.inp.s_copy);
    conv = ggml::reshape_3d(ctx, conv, d_conv - 1, d_inner, 1);
    let x3 = ggml::reshape_3d(ctx, cur, ggml::ne(cur, 0), n_tokens, 1);
    var zx = ggml::mul_mat(ctx, ly.ssm_in, x3);
    zx = ggml::permute(ctx, zx, 0, 2, 1, 3);
    zx = ggml::cont_4d(ctx, zx, head_dim * 2, n_heads, n_tokens, 1);
    var x = ggml::view_4d(ctx, zx, head_dim, n_heads, n_tokens, 1, ggml::nb(zx, 1), ggml::nb(zx, 2), ggml::nb(zx, 3),
                          head_dim * ggml::element_size(zx));
    x = ggml::cont_3d(ctx, x, head_dim * n_heads, n_tokens, 1);
    let z = ggml::view_4d(ctx, zx, head_dim, n_heads, n_tokens, 1, ggml::nb(zx, 1), ggml::nb(zx, 2), ggml::nb(zx, 3), 0);

    // the convolution; its last d_conv - 1 inputs back into the state
    let conv_x = ggml::concat(ctx, conv, ggml::transpose(ctx, x), 0);
    let last_conv = ggml::view_3d(ctx, conv_x, d_conv - 1, d_inner, 1, ggml::nb(conv_x, 1), ggml::nb(conv_x, 2),
                                  n_tokens * ggml::nb(conv_x, 0));
    ggml::build_forward_expand(b.gf, ggml::cpy(ctx, last_conv, ggml::view_1d(ctx, conv_states_all, (d_conv - 1) * d_inner, 0)));
    x = ggml::ssm_conv(ctx, conv_x, ly.ssm_conv1d);
    x = ggml::silu(ctx, x);

    let x_bcdt = ggml::mul_mat(ctx, ly.ssm_x, x);
    let dt_dim = max(64, hp.n_embd as i64 / 16);
    let es = ggml::element_size(x_bcdt);
    var bm = ggml::view_3d(ctx, x_bcdt, d_state, n_tokens, 1, ggml::nb(x_bcdt, 1), ggml::nb(x_bcdt, 2), 0);
    var cm = ggml::view_3d(ctx, x_bcdt, d_state, n_tokens, 1, ggml::nb(x_bcdt, 1), ggml::nb(x_bcdt, 2), es * d_state);
    var dt = ggml::view_3d(ctx, x_bcdt, dt_dim, n_tokens, 1, ggml::nb(x_bcdt, 1), ggml::nb(x_bcdt, 2), es * 2 * d_state);
    bm = build_norm_rms(b, m, bm, ly.ssm_b_norm);
    cm = build_norm_rms(b, m, cm, ly.ssm_c_norm);
    dt = build_norm_rms(b, m, dt, ly.ssm_dt_norm);
    dt = ggml::mul_mat(ctx, ly.ssm_dt, dt);
    dt = ggml::add(ctx, dt, ly.ssm_dt_b);
    let a = ggml::reshape_2d(ctx, ly.ssm_a, 1, n_heads);
    let ex = ggml::element_size(x);
    x = ggml::view_4d(ctx, x, head_dim, n_heads, n_tokens, 1, head_dim * ex, head_dim * n_heads * ex, head_dim * n_heads * n_tokens * ex, 0);
    bm = ggml::view_4d(ctx, bm, d_state, 1, n_tokens, 1, d_state * ggml::nb(bm, 0), ggml::nb(bm, 1), ggml::nb(bm, 2), 0);
    cm = ggml::view_4d(ctx, cm, d_state, 1, n_tokens, 1, d_state * ggml::nb(cm, 0), ggml::nb(cm, 1), ggml::nb(cm, 2), 0);
    let ssm = ggml::reshape_4d(ctx, ssm_states_all, d_state, head_dim, n_heads, 1);
    let y_ssm = ggml::ssm_scan(ctx, ssm, x, dt, a, bm, cm, b.inp.s_copy, 1);
    ggml::build_forward_expand(b.gf, y_ssm);
    let n_s = n_heads * head_dim * d_state;
    ggml::build_forward_expand(b.gf, ggml::cpy(ctx,
        ggml::view_1d(ctx, y_ssm, n_s, n_heads * head_dim * n_tokens * ggml::element_size(y_ssm)),
        ggml::view_1d(ctx, ssm_states_all, n_s, 0)));
    var y = ggml::view_4d(ctx, y_ssm, head_dim, n_heads, n_tokens, 1, head_dim * ex, head_dim * n_heads * ex, head_dim * n_heads * n_tokens * ex, 0);
    let d = ggml::reshape_2d(ctx, ly.ssm_d, 1, n_heads);
    y = ggml::add(ctx, y, ggml::mul(ctx, x, d));
    y = ggml::swiglu_split(ctx, ggml::cont(ctx, z), y);
    y = ggml::view_3d(ctx, y, head_dim * n_heads, n_tokens, 1, ggml::nb(y, 2), ggml::nb(y, 3), 0);
    let out = ggml::mul_mat(ctx, ly.ssm_out, y);
    return ggml::reshape_2d(ctx, out, ggml::ne(out, 0), n_tokens);
}

// models/plamo2.cpp
fn build_plamo2(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_layer = hp.n_layer as i64;
    let n_embd_head_v = hp.n_embd_head_v as i64;

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    for il in n_layer {
        let ly = m.layers[il];
        var residual = inpL;
        var cur = build_norm_rms(b, m, inpL, ly.attn_norm);
        if hp.is_recr[il] {
            cur = build_plamo2_mamba_layer(b, m, c, ly, cur, il);
        } else {
            var q, k, v = build_qkv(b, m, ly, cur, il);
            q = build_norm_rms(b, m, q, ly.attn_q_norm);
            q = rope(b, m, q, hp.rope_freq_base_train, hp.rope_freq_scale_train);
            k = build_norm_rms(b, m, k, ly.attn_k_norm);
            k = rope(b, m, k, hp.rope_freq_base_train, hp.rope_freq_scale_train);
            cur = build_attn(b, m, c, il, ly.wo, ggml::NONE, q, k, v, 1.0 / sqrt(n_embd_head_v as f32));
        }
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            residual = ggml::get_rows(ctx, residual, b.inp.out_ids);
        }
        cur = build_norm_rms(b, m, cur, ly.attn_post_norm);
        cur = ggml::add(ctx, cur, residual);
        residual = cur;
        cur = build_norm_rms(b, m, cur, ly.ffn_norm);
        cur = ggml::mul_mat(ctx, ly.ffn_down, ggml::swiglu(ctx, ggml::mul_mat(ctx, ly.ffn_up, cur)));
        cur = build_norm_rms(b, m, cur, ly.ffn_post_norm);
        inpL = ggml::add(ctx, cur, residual);
    }
    let cur = build_norm_rms(b, m, inpL, m.output_norm);
    return ggml::mul_mat(ctx, m.output, cur);
}

// models/nemotron-h.cpp (without experts)
fn build_nemotron_h(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_layer = hp.n_layer as i64;
    let n_embd_head = hp.n_embd_head_v as i64;
    let kq_scale: f32 = if hp.f_attention_scale == 0.0 { 1.0 / sqrt(n_embd_head as f32) } else { hp.f_attention_scale };

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    ggml::build_forward_expand(b.gf, inpL);
    for il in n_layer {
        let ly = m.layers[il];
        var inpSA = inpL;
        var cur = build_norm_rms(b, m, inpL, ly.attn_norm);
        if hp.is_recr[il] {
            cur = build_mamba2_layer(b, m, c, ly, cur, il);
        } else if hp.n_ff[il] == 0 {
            let q, k, v = build_qkv(b, m, ly, cur, il);
            cur = build_attn(b, m, c, il, ly.wo, ly.bo, q, k, v, kq_scale);
        } else {
            cur = build_ffn_relu_sqr(b, ly, cur);
        }
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpSA = ggml::get_rows(ctx, inpSA, b.inp.out_ids);
        }
        inpL = ggml::add(ctx, cur, inpSA);
    }
    let cur = build_norm_rms(b, m, inpL, m.output_norm);
    return ggml::mul_mat(ctx, m.output, cur);
}

// models/afmoe.cpp
fn build_afmoe(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_layer = hp.n_layer as i64;
    let n_embd_head = hp.n_embd_head_v as i64;
    let kq_scale = 1.0 / sqrt(n_embd_head as f32);

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    inpL = ggml::scale(ctx, inpL, sqrt(hp.n_embd as f32));
    for il in n_layer {
        let ly = m.layers[il];
        let freq_base = if hp.is_swa[il] { hp.rope_freq_base_train_swa } else { hp.rope_freq_base_train };
        let freq_scale = if hp.is_swa[il] { hp.rope_freq_scale_train_swa } else { hp.rope_freq_scale_train };
        var inpSA = inpL;
        let use_rope = hp.n_no_rope_layer_step > 0 && (il + 1) % hp.n_no_rope_layer_step != 0;
        var cur = build_norm_rms(b, m, inpL, ly.attn_norm);
        var q, k, v = build_qkv(b, m, ly, cur, il);
        var gate = ggml::mul_mat(ctx, ly.attn_gate, cur);
        q = build_norm_rms(b, m, q, ly.attn_q_norm);
        k = build_norm_rms(b, m, k, ly.attn_k_norm);
        if use_rope {
            q = rope(b, m, q, freq_base, freq_scale);
            k = rope(b, m, k, freq_base, freq_scale);
        }
        cur = build_kqv(b, m, c, il, q, k, v, kq_scale);
        cur = ggml::mul(ctx, cur, ggml::sigmoid(ctx, gate));
        cur = ggml::mul_mat(ctx, ly.wo, cur);
        cur = build_norm_rms(b, m, cur, ly.attn_post_norm);
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpSA = ggml::get_rows(ctx, inpSA, b.inp.out_ids);
        }
        let ffn_inp = ggml::add(ctx, cur, inpSA);
        cur = build_norm_rms(b, m, ffn_inp, ly.ffn_norm);
        if il >= hp.n_layer_dense_lead {
            let moe_out = build_moe_ffn(b, m, ly, cur, FFN_SILU, hp.expert_weights_norm, hp.expert_weights_scale, hp.expert_gating_func);
            cur = if hp.n_expert_shared > 0 {
                ggml::add(ctx, moe_out, build_ffn_swiglu(b, model::Layer { ffn_up: ly.ffn_up_shexp, ffn_gate: ly.ffn_gate_shexp,
                                                                           ffn_down: ly.ffn_down_shexp, .. }, cur))
            } else { moe_out };
        } else {
            cur = build_ffn_swiglu(b, ly, cur);
        }
        cur = build_norm_rms(b, m, cur, ly.ffn_post_norm);
        inpL = ggml::add(ctx, cur, ffn_inp);
    }
    let cur = build_norm_rms(b, m, inpL, m.output_norm);
    return ggml::mul_mat(ctx, m.output, cur);
}

// x's slice idx along its 3rd dimension (gemma3n_view_2d_slice).
fn slice_2d(b: Build&, x: ggml::Tensor, idx: i64) -> ggml::Tensor {
    assert(idx < ggml::ne(x, 2));
    return ggml::view_2d(b.ctx, x, ggml::ne(x, 0), ggml::ne(x, 1), ggml::row_size(ggml::tensor_type(x), ggml::ne(x, 0)),
                         idx * ggml::ne(x, 0) * ggml::ne(x, 1) * ggml::element_size(x));
}

fn calc_magnitude(b: Build&, x: ggml::Tensor) -> ggml::Tensor { ggml::sqrt(b.ctx, ggml::sum_rows(b.ctx, ggml::sqr(b.ctx, x))) }

// gemma3n's AltUp router: [n_embd, n_tokens] -> [n_altup, n_tokens]
fn altup_router_modalities(b: Build&, m: model::Model&, ly: model::Layer, x: ggml::Tensor) -> ggml::Tensor {
    let n_embd_f: f32 = m.hp.n_embd;
    var r = build_norm_rms(b, m, x, ly.altup_router_norm);
    r = ggml::scale(b.ctx, r, 1.0 / n_embd_f);
    return ggml::tanh(b.ctx, ggml::mul_mat(b.ctx, ly.altup_router, r));
}

// models/gemma3n.cpp
fn build_gemma3n(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_layer = hp.n_layer as i64;
    let n_tokens = b.n_tokens;
    let n_embd = hp.n_embd as i64;
    let n_embd_head = hp.n_embd_head_k as i64;
    let n_altup = hp.n_altup;
    let n_embd_altup = hp.n_embd_altup;
    let act = hp.i_altup_act;
    let n_layer_sparsity: i64 = 10;
    let f_sparsity_std_mul: f32 = 1.6448533535003662;
    let n_embd_f: f32 = n_embd;
    let sqrt2: f32 = 2.0;

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    inpL = ggml::scale(ctx, inpL, sqrt(n_embd_f));

    // the per-layer inputs: [n_embd_altup, n_tokens, n_layer] (build_inp_per_layer, project_per_layer_inputs)
    var inp_per_layer = ggml::get_rows(ctx, m.per_layer_tok_embd, b.inp.tokens);
    inp_per_layer = ggml::reshape_3d(ctx, inp_per_layer, n_embd_altup, n_layer, n_tokens);
    let n_embd_altup_f: f32 = n_embd_altup;
    inp_per_layer = ggml::scale(ctx, inp_per_layer, sqrt(n_embd_altup_f));
    ggml::build_forward_expand(b.gf, inp_per_layer);
    {
        var proj = ggml::mul_mat(ctx, m.per_layer_model_proj, inpL);
        proj = ggml::scale(ctx, proj, 1.0 / sqrt(n_embd_f));
        proj = ggml::reshape_3d(ctx, proj, n_embd_altup, n_layer, n_tokens);
        proj = build_norm_rms(b, m, proj, m.per_layer_proj_norm);
        inp_per_layer = ggml::add(ctx, proj, inp_per_layer);
        inp_per_layer = ggml::scale(ctx, inp_per_layer, 1.0 / sqrt(sqrt2));
        inp_per_layer = ggml::cont(ctx, ggml::permute(ctx, inp_per_layer, 0, 2, 1, 3));
    }

    // the other AltUp inputs, projected from the first
    {
        let target_magnitude = calc_magnitude(b, inpL);
        let inp_repeated = ggml::repeat_4d(ctx, inpL, n_embd, n_tokens, n_altup - 1, 1);
        var altup_added = ggml::mul_mat(ctx, m.altup_proj, inp_repeated);
        let new_magnitude = calc_magnitude(b, altup_added);
        altup_added = ggml::div(ctx, ggml::mul(ctx, altup_added, target_magnitude), new_magnitude);
        inpL = ggml::concat(ctx, inpL, altup_added, 2);     // [n_embd, n_tokens, n_altup]
    }

    for il in n_layer {
        let ly = m.layers[il];
        let freq_base = if hp.is_swa[il] { hp.rope_freq_base_train_swa } else { hp.rope_freq_base_train };
        let freq_scale = if hp.is_swa[il] { hp.rope_freq_scale_train_swa } else { hp.rope_freq_scale_train };

        // altup_predict
        var predictions = ggml::NONE;
        {
            let activated = slice_2d(b, inpL, act);
            let modalities = altup_router_modalities(b, m, ly, activated);
            var all_coefs = ggml::mul_mat(ctx, ly.altup_predict_coef, modalities);
            all_coefs = ggml::reshape_3d(ctx, all_coefs, n_altup, n_altup, n_tokens);
            let cur_permuted = ggml::cont(ctx, ggml::permute(ctx, inpL, 1, 2, 0, 3));
            predictions = ggml::mul_mat(ctx, cur_permuted, all_coefs);
            predictions = ggml::cont(ctx, ggml::permute(ctx, predictions, 0, 2, 1, 3));
            predictions = ggml::add(ctx, predictions, inpL);
        }
        let active_prediction = slice_2d(b, predictions, act);
        var cur = build_norm_rms(b, m, active_prediction, ly.attn_norm);

        // laurel
        var laurel_out = ggml::mul_mat(ctx, ly.laurel_l, cur);
        laurel_out = ggml::mul_mat(ctx, ly.laurel_r, laurel_out);
        laurel_out = build_norm_rms(b, m, laurel_out, ly.laurel_post_norm);
        laurel_out = ggml::add(ctx, laurel_out, cur);

        // self-attention
        if model::has_kv(hp, il) {
            var q, k, v = build_qkv(b, m, ly, cur, il);
            q = build_norm_rms(b, m, q, ly.attn_q_norm);
            k = build_norm_rms(b, m, k, ly.attn_k_norm);
            v = ggml::rms_norm(ctx, v, hp.f_norm_rms_eps);
            q = rope(b, m, q, freq_base, freq_scale);
            k = rope(b, m, k, freq_base, freq_scale);
            cur = build_attn(b, m, c, il, ly.wo, ggml::NONE, q, k, v, hp.f_attention_scale);
        } else {
            // the keys and values of an earlier layer
            var q = ggml::NONE;
            if ly.wqkv.p != 0 {
                let qkv = ggml::mul_mat(ctx, ly.wqkv, cur);
                q = ggml::cont(ctx, ggml::view_2d(ctx, qkv, n_embd_head * hp.n_head[il], n_tokens, ggml::nb(qkv, 1), 0));
            } else {
                q = ggml::mul_mat(ctx, ly.wq, cur);
            }
            q = ggml::reshape_3d(ctx, q, n_embd_head, hp.n_head[il], n_tokens);
            q = build_norm_rms(b, m, q, ly.attn_q_norm);
            q = rope(b, m, q, freq_base, freq_scale);
            cur = build_attn(b, m, c, il, ly.wo, ggml::NONE, q, ggml::NONE, ggml::NONE, hp.f_attention_scale);
        }
        cur = build_norm_rms(b, m, cur, ly.attn_post_norm);
        cur = ggml::add(ctx, cur, active_prediction);
        let attn_laurel = ggml::scale(ctx, ggml::add(ctx, cur, laurel_out), 1.0 / sqrt(sqrt2));
        cur = build_norm_rms(b, m, attn_laurel, ly.ffn_norm);

        // feed-forward, with activation sparsity in the first layers
        {
            let up_proj = ggml::mul_mat(ctx, ly.ffn_up, cur);
            var gate_proj = ggml::mul_mat(ctx, ly.ffn_gate, cur);
            if il < n_layer_sparsity {
                // gaussian_topk
                let mean = ggml::mean(ctx, gate_proj);
                let n1: f32 = ggml::ne(gate_proj, 0) - 1;
                let std = ggml::sqrt(ctx, ggml::scale(ctx, ggml::sum_rows(ctx, ggml::sqr(ctx, ggml::sub(ctx, gate_proj, mean))), 1.0 / n1));
                let cutoff = ggml::add(ctx, mean, ggml::scale(ctx, std, f_sparsity_std_mul));
                gate_proj = ggml::relu(ctx, ggml::sub(ctx, gate_proj, cutoff));
            }
            gate_proj = ggml::gelu(ctx, gate_proj);
            cur = ggml::mul(ctx, up_proj, gate_proj);
            cur = ggml::mul_mat(ctx, ly.ffn_down, cur);
        }
        cur = build_norm_rms(b, m, cur, ly.ffn_post_norm);
        let attn_ffw_laurel_gated = ggml::add(ctx, cur, attn_laurel);

        // altup_correct
        var corrected = ggml::NONE;
        {
            let modalities = altup_router_modalities(b, m, ly, attn_ffw_laurel_gated);
            let innovation0 = ggml::sub(ctx, attn_ffw_laurel_gated, active_prediction);
            var all_coefs = ggml::mul_mat(ctx, ly.altup_correct_coef, modalities);
            all_coefs = ggml::scale_bias(ctx, all_coefs, 1.0, 1.0);
            all_coefs = ggml::transpose(ctx, all_coefs);
            all_coefs = ggml::cont_3d(ctx, all_coefs, 1, n_tokens, n_altup);
            let innovation = ggml::repeat_4d(ctx, innovation0, n_embd, n_tokens, n_altup, 1);
            corrected = ggml::mul(ctx, innovation, all_coefs);
            corrected = ggml::add(ctx, corrected, predictions);
        }
        var first_prediction = slice_2d(b, corrected, act);
        first_prediction = ggml::mul(ctx, first_prediction, ly.altup_correct_scale);
        first_prediction = ggml::mul_mat(ctx, ly.per_layer_inp_gate, first_prediction);
        first_prediction = ggml::gelu(ctx, first_prediction);
        first_prediction = ggml::mul(ctx, first_prediction, slice_2d(b, inp_per_layer, il));
        first_prediction = ggml::mul_mat(ctx, ly.per_layer_proj, first_prediction);
        first_prediction = build_norm_rms(b, m, first_prediction, ly.per_layer_post_norm);
        // corrected[1:] += first_prediction
        {
            let slice_first = slice_2d(b, corrected, 0);
            let ty = ggml::tensor_type(corrected);
            let slice_rest = ggml::view_3d(ctx, corrected, n_embd, n_tokens, n_altup - 1, ggml::row_size(ty, n_embd),
                                           ggml::row_size(ty, n_embd * n_tokens), n_embd * n_tokens * ggml::element_size(corrected));
            corrected = ggml::concat(ctx, slice_first, ggml::add(ctx, slice_rest, first_prediction), 2);
        }
        inpL = corrected;
    }

    // the AltUp inputs merged back into one
    var cur = inpL;
    {
        let target_magnitude = calc_magnitude(b, slice_2d(b, cur, act));
        let ty = ggml::tensor_type(cur);
        let alt_slice = ggml::view_3d(ctx, cur, n_embd, n_tokens, n_altup - 1, ggml::row_size(ty, n_embd),
                                      ggml::row_size(ty, n_embd * n_tokens), n_embd * n_tokens * ggml::element_size(cur));
        var altup_unembd = ggml::mul_mat(ctx, m.altup_unembd_proj, alt_slice);
        let new_magnitude = calc_magnitude(b, altup_unembd);
        altup_unembd = ggml::div(ctx, ggml::mul(ctx, altup_unembd, target_magnitude), new_magnitude);
        cur = slice_2d(b, cur, 0);
        for i in n_altup - 1 { cur = ggml::add(ctx, cur, slice_2d(b, altup_unembd, i)); }
        let n_altup_f: f32 = n_altup;
        cur = ggml::scale(ctx, cur, 1.0 / n_altup_f);
    }
    cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
    cur = build_norm_rms(b, m, cur, m.output_norm);
    cur = ggml::mul_mat(ctx, m.output, cur);
    // final logit soft-capping
    cur = ggml::scale(ctx, cur, 1.0 / hp.f_final_logit_softcapping);
    cur = ggml::tanh(ctx, cur);
    return ggml::scale(ctx, cur, hp.f_final_logit_softcapping);
}

// models/gemma4.cpp (without experts)
fn build_gemma4(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_layer = hp.n_layer as i64;
    let n_tokens = b.n_tokens;
    let n_embd_f: f32 = hp.n_embd;
    let n_epl = hp.n_embd_per_layer as i64;
    let two: f32 = 2.0;

    // the per-layer inputs first, as llama.cpp
    var inp_per_layer = ggml::NONE;
    if m.per_layer_tok_embd.p != 0 {
        inp_per_layer = ggml::get_rows(ctx, m.per_layer_tok_embd, b.inp.tokens);
        ggml::build_forward_expand(b.gf, inp_per_layer);
    }
    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    inpL = ggml::scale(ctx, inpL, sqrt(n_embd_f));
    if inp_per_layer.p != 0 {
        let n_epl_f: f32 = n_epl;
        inp_per_layer = ggml::scale(ctx, inp_per_layer, sqrt(n_epl_f));
        inp_per_layer = ggml::reshape_3d(ctx, inp_per_layer, n_epl, n_layer, ggml::ne(inp_per_layer, 1));
        // project_per_layer_inputs
        var proj = ggml::mul_mat(ctx, m.per_layer_model_proj, inpL);
        proj = ggml::scale(ctx, proj, 1.0 / sqrt(n_embd_f));
        proj = ggml::reshape_3d(ctx, proj, n_epl, n_layer, n_tokens);
        proj = build_norm_rms(b, m, proj, m.per_layer_proj_norm);
        inp_per_layer = ggml::add(ctx, proj, inp_per_layer);
        inp_per_layer = ggml::scale(ctx, inp_per_layer, 1.0 / sqrt(two));
        inp_per_layer = ggml::cont(ctx, ggml::permute(ctx, inp_per_layer, 0, 2, 1, 3));
    }

    for il in n_layer {
        let ly = m.layers[il];
        let n_embd_head = model::head_k(hp, il);
        assert(n_embd_head == model::head_v(hp, il));
        let n_head = hp.n_head[il] as i64;
        let n_head_kv = hp.n_head_kv[il] as i64;
        let freq_base = if hp.is_swa[il] { hp.rope_freq_base_train_swa } else { hp.rope_freq_base_train };
        let freq_scale = if hp.is_swa[il] { hp.rope_freq_scale_train_swa } else { hp.rope_freq_scale_train };
        let n_rot_l = model::rot(hp, il);
        var cur = build_norm_rms(b, m, inpL, ly.attn_norm);
        let factors = if hp.is_swa[il] { ggml::NONE } else { ly.rope_freqs };

        // Q, for the layers with and without keys and values of their own
        var qkv_fused = ggml::NONE;
        var q = ggml::NONE;
        if ly.wqkv.p != 0 {
            qkv_fused = ggml::mul_mat(ctx, ly.wqkv, cur);
            q = ggml::cont(ctx, ggml::view_2d(ctx, qkv_fused, n_embd_head * n_head, n_tokens, ggml::nb(qkv_fused, 1), 0));
        } else {
            q = ggml::mul_mat(ctx, ly.wq, cur);
        }
        q = ggml::reshape_3d(ctx, q, n_embd_head, n_head, n_tokens);
        q = build_norm_rms(b, m, q, ly.attn_q_norm);
        q = rope(b, m, q, freq_base, freq_scale, factors, n_rot_l);

        if model::has_kv(hp, il) {
            var k = ggml::NONE;
            var v = ggml::NONE;
            if qkv_fused.p != 0 {
                let q_dim = n_embd_head * n_head;
                let k_dim = n_embd_head * n_head_kv;
                let es = ggml::element_size(qkv_fused);
                k = ggml::cont(ctx, ggml::view_2d(ctx, qkv_fused, k_dim, n_tokens, ggml::nb(qkv_fused, 1), q_dim * es));
                v = ggml::cont(ctx, ggml::view_2d(ctx, qkv_fused, k_dim, n_tokens, ggml::nb(qkv_fused, 1), (q_dim + k_dim) * es));
            } else {
                k = ggml::mul_mat(ctx, ly.wk, cur);
                v = if ly.wv.p != 0 { ggml::mul_mat(ctx, ly.wv, cur) } else { k };   // no v_proj: v is k
            }
            k = ggml::reshape_3d(ctx, k, n_embd_head, n_head_kv, n_tokens);
            v = ggml::reshape_3d(ctx, v, n_embd_head, n_head_kv, n_tokens);
            k = build_norm_rms(b, m, k, ly.attn_k_norm);
            v = ggml::rms_norm(ctx, v, hp.f_norm_rms_eps);
            k = rope(b, m, k, freq_base, freq_scale, factors, n_rot_l);
            cur = build_attn(b, m, c, il, ly.wo, ggml::NONE, q, k, v, hp.f_attention_scale);
        } else {
            cur = build_attn(b, m, c, il, ly.wo, ggml::NONE, q, ggml::NONE, ggml::NONE, hp.f_attention_scale);
        }
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpL = ggml::get_rows(ctx, inpL, b.inp.out_ids);
        }
        cur = build_norm_rms(b, m, cur, ly.attn_post_norm);
        let attn_out = ggml::add(ctx, cur, inpL);
        cur = build_norm_rms(b, m, attn_out, ly.ffn_norm);
        let up = ggml::mul_mat(ctx, ly.ffn_up, cur);
        let gate = ggml::mul_mat(ctx, ly.ffn_gate, cur);
        cur = ggml::mul_mat(ctx, ly.ffn_down, ggml::geglu_split(ctx, gate, up));
        cur = build_norm_rms(b, m, cur, ly.ffn_post_norm);
        cur = ggml::add(ctx, cur, attn_out);
        // the per-layer input
        if inp_per_layer.p != 0 {
            let pe_in = cur;
            cur = ggml::mul_mat(ctx, ly.per_layer_inp_gate, cur);
            cur = ggml::gelu(ctx, cur);
            var inp_this_layer = slice_2d(b, inp_per_layer, il);
            if il == n_layer - 1 { inp_this_layer = ggml::get_rows(ctx, inp_this_layer, b.inp.out_ids); }
            cur = ggml::mul(ctx, cur, inp_this_layer);
            cur = ggml::mul_mat(ctx, ly.per_layer_proj, cur);
            cur = build_norm_rms(b, m, cur, ly.per_layer_post_norm);
            cur = ggml::add(ctx, pe_in, cur);
        }
        if ly.out_scale.p != 0 { cur = ggml::mul(ctx, cur, ly.out_scale); }
        inpL = cur;
    }
    var cur = build_norm_rms(b, m, inpL, m.output_norm);
    cur = ggml::mul_mat(ctx, m.output, cur);
    if hp.f_final_logit_softcapping != 0.0 {
        cur = ggml::scale(ctx, cur, 1.0 / hp.f_final_logit_softcapping);
        cur = ggml::tanh(ctx, cur);
        cur = ggml::scale(ctx, cur, hp.f_final_logit_softcapping);
    }
    return cur;
}

// models/mamba.cpp (mamba and mamba2)
fn build_mamba(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_layer = hp.n_layer as i64;

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    for il in n_layer {
        let ly = m.layers[il];
        var cur = build_norm_rms(b, m, inpL, ly.attn_norm);
        cur = if hp.arch == model::ARCH_MAMBA2 { build_mamba2_layer(b, m, c, ly, cur, il) } else { build_mamba_layer(b, m, c, ly, cur, il) };
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpL = ggml::get_rows(ctx, inpL, b.inp.out_ids);
        }
        inpL = ggml::add(ctx, cur, inpL);
    }
    let cur = build_norm_rms(b, m, inpL, m.output_norm);
    return ggml::mul_mat(ctx, m.output, cur);
}

// models/qwen35.cpp: a gated delta net layer (build_layer_attn_linear,
// one sequence, with the fused gated delta net llama.cpp picks on the CPU).
fn build_gated_delta_net(b: Build&, m: model::Model&, c: Context&, ly: model::Layer, cur: ggml::Tensor, il: i64) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let d_inner = hp.ssm_d_inner as i64;
    let head_k_dim = hp.ssm_d_state as i64;
    let num_k_heads = hp.ssm_n_group as i64;
    let num_v_heads = hp.ssm_dt_rank as i64;
    let head_v_dim = d_inner / num_v_heads;
    let n_tokens = b.n_tokens;

    var qkv_mixed = ggml::mul_mat(ctx, ly.wqkv, cur);
    qkv_mixed = ggml::reshape_3d(ctx, qkv_mixed, ggml::ne(qkv_mixed, 0), n_tokens, 1);
    let z = ggml::mul_mat(ctx, ly.attn_gate, cur);

    var beta = ggml::mul_mat(ctx, ly.ssm_beta, cur);
    beta = ggml::reshape_4d(ctx, beta, 1, num_v_heads, n_tokens, 1);
    beta = ggml::sigmoid(ctx, beta);

    var alpha = ggml::mul_mat(ctx, ly.ssm_alpha, cur);
    alpha = ggml::reshape_3d(ctx, alpha, num_v_heads, n_tokens, 1);
    let alpha_softplus = ggml::softplus(ctx, ggml::add(ctx, alpha, ly.ssm_dt));
    var gate = ggml::mul(ctx, alpha_softplus, ly.ssm_a);    // -A_log.exp() * softplus
    gate = ggml::reshape_4d(ctx, gate, 1, num_v_heads, n_tokens, 1);

    // the convolution over the layer's last d_conv - 1 inputs and this
    // batch's; the last d_conv - 1 become the new state (build_conv_state)
    let conv_states_all = c.r[il];
    let ssm_states_all = c.s[il];
    let conv_kernel_size = ggml::ne(ly.ssm_conv1d, 0);
    let conv_channels = d_inner + 2 * num_k_heads * head_k_dim;
    var conv_states = ggml::get_rows(ctx, conv_states_all, b.inp.s_copy);
    conv_states = ggml::reshape_3d(ctx, conv_states, conv_kernel_size - 1, conv_channels, 1);
    let conv_input = ggml::concat(ctx, conv_states, ggml::transpose(ctx, qkv_mixed), 0);
    let s_idx = ggml::ne(conv_input, 0) - ggml::ne(conv_states, 0);
    let conv_state_last = ggml::view_3d(ctx, conv_input, conv_kernel_size - 1, conv_channels, 1,
                                        ggml::nb(conv_input, 1), ggml::nb(conv_input, 2),
                                        ggml::row_size(ggml::tensor_type(conv_input), s_idx));
    let row_count = (conv_kernel_size - 1) * conv_channels;
    ggml::build_forward_expand(b.gf, ggml::cpy(ctx, conv_state_last,
                                               ggml::view_2d(ctx, conv_states_all, row_count, 1, ggml::nb(conv_states_all, 1), 0)));

    var state = ggml::get_rows(ctx, ssm_states_all, b.inp.s_copy);
    state = ggml::reshape_4d(ctx, state, head_v_dim, head_v_dim, num_v_heads, 1);

    let conv_out = ggml::silu(ctx, ggml::ssm_conv(ctx, conv_input, ly.ssm_conv1d));
    let ty = ggml::tensor_type(conv_out);
    let qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    let nb1_qkv = ggml::row_size(ty, qkv_dim);
    var q_conv = ggml::view_4d(ctx, conv_out, head_k_dim, num_k_heads, n_tokens, 1,
                               ggml::row_size(ty, head_k_dim), nb1_qkv, nb1_qkv * n_tokens, 0);
    var k_conv = ggml::view_4d(ctx, conv_out, head_k_dim, num_k_heads, n_tokens, 1,
                               ggml::row_size(ty, head_k_dim), nb1_qkv, nb1_qkv * n_tokens,
                               head_k_dim * num_k_heads * ggml::element_size(conv_out));
    let v_conv = ggml::view_4d(ctx, conv_out, head_v_dim, num_v_heads, n_tokens, 1,
                               ggml::row_size(ty, head_v_dim), nb1_qkv, nb1_qkv * n_tokens,
                               ggml::row_size(ty, 2 * head_k_dim * num_k_heads));
    // build_gdn_l2_norm
    let eps = hp.f_norm_rms_eps;
    let n_k: f32 = head_k_dim;
    q_conv = ggml::scale(ctx, ggml::rms_norm(ctx, q_conv, eps / n_k), 1.0 / sqrt(n_k));
    k_conv = ggml::scale(ctx, ggml::rms_norm(ctx, k_conv, eps / n_k), 1.0 / sqrt(n_k));

    // build_delta_net_fused, and the new state into the cache
    let result = ggml::gated_delta_net(ctx, q_conv, k_conv, v_conv, gate, beta, state, 1);
    let rt = ggml::tensor_type(result);
    let output = ggml::view_4d(ctx, result, head_v_dim, num_v_heads, n_tokens, 1,
                               ggml::row_size(rt, head_v_dim), ggml::row_size(rt, head_v_dim * num_v_heads),
                               ggml::row_size(rt, head_v_dim * num_v_heads * n_tokens), 0);
    let new_state = ggml::view_4d(ctx, result, head_v_dim, head_v_dim, num_v_heads, 1,
                                  ggml::row_size(rt, head_v_dim), ggml::row_size(rt, head_v_dim * head_v_dim),
                                  ggml::row_size(rt, head_v_dim * head_v_dim * num_v_heads),
                                  ggml::row_size(rt, head_v_dim * num_v_heads * n_tokens));
    ggml::build_forward_expand(b.gf, ggml::cpy(ctx, new_state,
                                               ggml::view_2d(ctx, ssm_states_all, model::n_embd_s(hp), 1, ggml::nb(ssm_states_all, 1), 0)));

    // the gated norm: norm(output) * silu(z)
    let z_4d = ggml::reshape_4d(ctx, z, head_v_dim, num_v_heads, n_tokens, 1);
    let normed = ggml::mul(ctx, build_norm_rms(b, m, output, ly.ssm_norm), ggml::silu(ctx, z_4d));
    let final_output = ggml::reshape_2d(ctx, normed, head_v_dim * num_v_heads, n_tokens);
    return ggml::mul_mat(ctx, ly.ssm_out, final_output);
}

// models/qwen35.cpp: an attention layer whose query carries a sigmoid gate
// per value, with IMROPE (build_layer_attn).
fn build_qwen35_attn(b: Build&, m: model::Model&, c: Context&, ly: model::Layer, cur: ggml::Tensor, il: i64) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_embd_head = hp.n_embd_head_v as i64;
    assert(n_embd_head == hp.n_embd_head_k);
    let n_head = hp.n_head[il] as i64;
    let n_head_kv = hp.n_head_kv[il] as i64;
    let n_tokens = b.n_tokens;

    let q_full = ggml::mul_mat(ctx, ly.wq, cur);
    var k = ggml::mul_mat(ctx, ly.wk, cur);
    var v = ggml::mul_mat(ctx, ly.wv, cur);
    let es = ggml::element_size(q_full);
    var q = ggml::view_3d(ctx, q_full, n_embd_head, n_head, n_tokens, es * n_embd_head * 2, es * n_embd_head * 2 * n_head, 0);
    q = build_norm_rms(b, m, q, ly.attn_q_norm);
    k = ggml::reshape_3d(ctx, k, n_embd_head, n_head_kv, n_tokens);
    k = build_norm_rms(b, m, k, ly.attn_k_norm);
    var gate = ggml::view_3d(ctx, q_full, n_embd_head, n_head, n_tokens, es * n_embd_head * 2, es * n_embd_head * 2 * n_head,
                             es * n_embd_head);
    gate = ggml::cont_2d(ctx, gate, n_embd_head * n_head, n_tokens);
    v = ggml::reshape_3d(ctx, v, n_embd_head, n_head_kv, n_tokens);

    q = rope_multi(b, m, q);
    k = rope_multi(b, m, k);
    let kq_scale: f32 = if hp.f_attention_scale == 0.0 { 1.0 / sqrt(n_embd_head as f32) } else { hp.f_attention_scale };
    var attn = build_kqv(b, m, c, il, q, k, v, kq_scale);
    attn = ggml::mul(ctx, attn, ggml::sigmoid(ctx, gate));
    return ggml::mul_mat(ctx, ly.wo, attn);
}

fn rope_multi(b: Build&, m: model::Model&, x: ggml::Tensor) -> ggml::Tensor {
    let hp .= m.hp;
    return ggml::rope_multi(b.ctx, x, b.inp.pos, ggml::NONE, hp.n_rot as i32, hp.rope_sections, hp.rope_type,
                            hp.n_ctx_orig_yarn as i32, hp.rope_freq_base_train, hp.rope_freq_scale_train, hp.yarn_ext_factor,
                            hp.yarn_attn_factor, hp.yarn_beta_fast, hp.yarn_beta_slow);
}

// models/qwen35.cpp
fn build_qwen35(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_layer = hp.n_layer as i64;

    var inpL = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    for il in n_layer {
        let ly = m.layers[il];
        var inpSA = inpL;
        var cur = build_norm_rms(b, m, inpL, ly.attn_norm);
        ggml::build_forward_expand(b.gf, cur);
        cur = if hp.is_recr[il] { build_gated_delta_net(b, m, c, ly, cur, il) } else { build_qwen35_attn(b, m, c, ly, cur, il) };
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            inpSA = ggml::get_rows(ctx, inpSA, b.inp.out_ids);
        }
        cur = ggml::add(ctx, cur, inpSA);
        let ffn_residual = cur;
        cur = build_norm_rms(b, m, cur, ly.attn_post_norm);
        cur = build_ffn_swiglu(b, ly, cur);
        inpL = ggml::add(ctx, cur, ffn_residual);
    }
    let cur = build_norm_rms(b, m, inpL, m.output_norm);
    return ggml::mul_mat(ctx, m.output, cur);
}

// A short convolution over the layer's last l_cache - 1 inputs, which its
// recurrent state keeps (build_lfm_shortconv, one sequence).
fn build_shortconv(b: Build&, m: model::Model&, c: Context&, ly: model::Layer, cur: ggml::Tensor, il: i64) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let d_conv = hp.n_shortconv_l_cache as i64 - 1;
    let n_embd = hp.n_embd as i64;

    var bcx = ggml::mul_mat(ctx, ly.shortconv_in_proj, cur);
    bcx = ggml::reshape_3d(ctx, bcx, ggml::ne(bcx, 0), b.n_tokens, 1);
    let chunk = ggml::ne(bcx, 0) / 3;
    let es = ggml::element_size(bcx);
    let nb1 = ggml::nb(bcx, 1);
    let nb2 = ggml::nb(bcx, 2);
    let bb = ggml::view_3d(ctx, bcx, chunk, b.n_tokens, 1, nb1, nb2, 0);
    let cc = ggml::view_3d(ctx, bcx, chunk, b.n_tokens, 1, nb1, nb2, chunk * es);
    let xx = ggml::view_3d(ctx, bcx, chunk, b.n_tokens, 1, nb1, nb2, 2 * chunk * es);
    var bx = ggml::transpose(ctx, ggml::mul(ctx, bb, xx));

    // the state's rows (build_rs), and the inputs it keeps in front of this batch's
    let state = c.r[il];
    let conv = ggml::reshape_3d(ctx, ggml::get_rows(ctx, state, b.inp.s_copy), d_conv, n_embd, 1);
    bx = ggml::concat(ctx, conv, bx, 0);

    // the last d_conv inputs become the new state
    let snap = ggml::view_3d(ctx, bx, d_conv, ggml::ne(bx, 1), ggml::ne(bx, 2), ggml::nb(bx, 1), ggml::nb(bx, 2),
                             (ggml::ne(bx, 0) - d_conv) * ggml::element_size(bx));
    ggml::build_forward_expand(b.gf, ggml::cpy(ctx, snap, ggml::view_2d(ctx, state, d_conv * n_embd, 1, ggml::nb(state, 1), 0)));

    let conv_out = ggml::ssm_conv(ctx, bx, ly.shortconv_conv);
    var y = ggml::mul(ctx, cc, conv_out);
    y = ggml::reshape_2d(ctx, y, ggml::ne(y, 0), b.n_tokens);
    return ggml::mul_mat(ctx, ly.shortconv_out_proj, y);
}

// models/lfm2.cpp
fn build_lfm2(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_layer = hp.n_layer as i64;
    let n_embd_head = hp.n_embd_head_v as i64;

    var cur = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    ggml::build_forward_expand(b.gf, cur);
    for il in n_layer {
        let ly = m.layers[il];
        var prev = cur;
        cur = build_norm_rms(b, m, cur, ly.attn_norm);
        if hp.is_recr[il] {
            cur = build_shortconv(b, m, c, ly, cur, il);
        } else {
            var q, k, v = build_qkv(b, m, ly, cur, il);
            q = build_norm_rms(b, m, q, ly.attn_q_norm);
            k = build_norm_rms(b, m, k, ly.attn_k_norm);
            q = rope(b, m, q, hp.rope_freq_base_train, hp.rope_freq_scale_train);
            k = rope(b, m, k, hp.rope_freq_base_train, hp.rope_freq_scale_train);
            cur = build_attn(b, m, c, il, ly.wo, ggml::NONE, q, k, v, 1.0 / sqrt(n_embd_head as f32));
        }
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            prev = ggml::get_rows(ctx, prev, b.inp.out_ids);
        }
        cur = ggml::add(ctx, prev, cur);
        let ffn_norm_out = build_norm_rms(b, m, cur, ly.ffn_norm);
        cur = ggml::add(ctx, cur, build_ffn_swiglu(b, ly, ffn_norm_out));
    }
    cur = build_norm_rms(b, m, cur, m.output_norm);
    return ggml::mul_mat(ctx, m.output, cur);
}

// models/lfm3.cpp
fn build_lfm3(b: Build&, m: model::Model&, c: Context&) -> ggml::Tensor {
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_layer = hp.n_layer as i64;
    let n_embd_head = hp.n_embd_head_k as i64;

    var cur = ggml::get_rows(ctx, m.tok_embd, b.inp.tokens);
    ggml::build_forward_expand(b.gf, cur);
    for il in n_layer {
        let ly = m.layers[il];
        var prev = cur;
        cur = build_norm_rms(b, m, cur, ly.attn_norm);
        if hp.is_recr[il] {
            cur = build_shortconv(b, m, c, ly, cur, il);
        } else {
            let n_head = hp.n_head[il] as i64;
            let swa = hp.is_swa[il];
            let freq_base = if swa { hp.rope_freq_base_train_swa } else { hp.rope_freq_base_train };
            let freq_scale = if swa { hp.rope_freq_scale_train_swa } else { hp.rope_freq_scale_train };
            let factors = if swa { ggml::NONE } else { ly.rope_freqs };
            var q, k, v = build_qkv(b, m, ly, cur, il);
            q = build_norm_rms(b, m, q, ly.attn_q_norm);
            k = build_norm_rms(b, m, k, ly.attn_k_norm);
            q = rope(b, m, q, freq_base, freq_scale, factors);
            k = rope(b, m, k, freq_base, freq_scale, factors);
            var attn = build_kqv(b, m, c, il, q, k, v, 1.0 / sqrt(n_embd_head as f32));
            // each head's output times the sigmoid of its gate
            let gate = ggml::sigmoid(ctx, ggml::mul_mat(ctx, ly.attn_gate, cur));
            let attn_3d = ggml::mul(ctx, ggml::reshape_3d(ctx, attn, n_embd_head, n_head, b.n_tokens),
                                    ggml::reshape_3d(ctx, gate, 1, n_head, b.n_tokens));
            attn = ggml::reshape_2d(ctx, attn_3d, n_embd_head * n_head, b.n_tokens);
            cur = ggml::mul_mat(ctx, ly.wo, attn);
        }
        if ly.attn_post_norm.p != 0 { cur = build_norm_rms(b, m, cur, ly.attn_post_norm); }
        if il == n_layer - 1 {
            cur = ggml::get_rows(ctx, cur, b.inp.out_ids);
            prev = ggml::get_rows(ctx, prev, b.inp.out_ids);
        }
        cur = ggml::add(ctx, prev, cur);
        let ffn_norm_out = build_norm_rms(b, m, cur, ly.ffn_norm);
        var ffn_out = build_ffn_swiglu(b, ly, ffn_norm_out);
        if ly.ffn_post_norm.p != 0 { ffn_out = build_norm_rms(b, m, ffn_out, ly.ffn_post_norm); }
        cur = ggml::add(ctx, cur, ffn_out);
    }
    cur = build_norm_rms(b, m, cur, m.output_norm);
    return ggml::mul_mat(ctx, m.output, cur);
}

let NEG_INF: f32 = -1.0 / 0.0;

// mask[t][j] for token t (at position pos0 + t) and cell j: 0, or -inf when
// the cell is in the token's future, unused, or (swa) out of its window.
// With ALiBi the kept cells hold minus their distance to the token.
fn fill_mask(mask: f32[>..]&, n_kv: i64, n_tokens: i64, pos0: i64, n_used: i64, swa: u32, alibi: bool = false) {
    mask.clear();
    for t in n_tokens {
        let p1 = pos0 + t;
        for j in n_kv {
            let masked = j >= n_used || j > p1 || (swa > 0 && p1 - j >= swa);
            mask.push(if masked { NEG_INF } else if alibi { -abs(p1 - j) as f32 } else { 0.0 });
        }
    }
}

fn n_pos_per_embd(m: model::Model&) -> i64 { if m.hp.rope_type == ggml::ROPE_TYPE_MROPE || m.hp.rope_type == ggml::ROPE_TYPE_IMROPE { 4 } else { 1 } }

fn any_recurrent(m: model::Model&) -> bool {
    for r in m.hp.is_recr { if r { return true; } }
    return false;
}

// Forgets everything: llama_memory_clear(mem, true).
fn clear(c: Context&) {
    ggml::backend_buffer_clear(c.kv_buf, 0);
    c.n_past = 0;
}

// Keeps the first n cells, forgets the rest: llama_memory_seq_rm(mem, 0, n,
// -1). A recurrent state cannot go back, so it is cleared unless n keeps
// all of it; the cells kept: n or 0.
fn keep_first(c: Context&, m: model::Model&, n: i64) -> i64 {
    if n >= c.n_past { return c.n_past; }
    if any_recurrent(m) { clear(c); return 0; }
    c.n_past = n;
    return n;
}

// Runs tokens, at positions n_past.., through the model; writes the logits
// of the last one to logits. With `outputs` (indices into tokens, in
// order), the logits of those tokens instead, one row of n_vocab after another.
fn decode(c: Context&, m: model::Model&, tokens: const i32[:], logits: f32[>..]&, outputs: const i32[:] = []) {
    let n_tokens = tokens.len;
    let n_outputs = if outputs.len > 0 { outputs.len } else { 1 };
    guard n_tokens > 0 && c.n_past + n_tokens <= c.n_ctx else { abort("decode: the context is full"); }
    let pos0 = c.n_past;

    let gctx = ggml::init(MAX_NODES * ggml::tensor_overhead() + ggml::graph_overhead_custom(MAX_NODES, false), true);
    var b = Build {
        ctx: gctx,
        gf: ggml::new_graph_custom(gctx, MAX_NODES, false),
        n_tokens: n_tokens,
        // llama.cpp's get_n_kv: the cells in use, padded to 256
        n_kv: min(c.n_ctx, max(256, pad_to(c.n_past + n_tokens, 256))),
        inp: Inputs {
            tokens: ggml::new_tensor_1d(gctx, ggml::TYPE_I32, n_tokens),
            pos: ggml::new_tensor_1d(gctx, ggml::TYPE_I32, n_tokens * n_pos_per_embd(m)),
            out_ids: ggml::new_tensor_1d(gctx, ggml::TYPE_I32, n_outputs),
            kv_idxs: ggml::new_tensor_1d(gctx, ggml::TYPE_I64, n_tokens),
            mask: ggml::NONE,
            mask_swa: ggml::NONE,
            s_copy: ggml::new_tensor_1d(gctx, ggml::TYPE_I32, 1),
            attn_scale: ggml::NONE,
            pos_bucket: ggml::NONE,
            cross_embd: ggml::NONE,
            cross_mask: ggml::NONE,
        },
        tracing: false,
        traced: [],
    };
    {
        var flag: u8[>..] = [];
        b.tracing = env("LLAMA_GOOSE_TRACE", flag) && flag == "1";
    }
    let has_swa = m.hp.swa_type != model::SWA_NONE;
    b.inp.mask = ggml::new_tensor_2d(gctx, ggml::TYPE_F16, b.n_kv, n_tokens);
    b.inp.mask_swa = if has_swa { ggml::new_tensor_2d(gctx, ggml::TYPE_F16, b.n_kv, n_tokens) } else { b.inp.mask };
    ggml::set_input(b.inp.tokens);
    ggml::set_input(b.inp.pos);
    ggml::set_input(b.inp.out_ids);
    ggml::set_input(b.inp.kv_idxs);
    ggml::set_input(b.inp.mask);
    if has_swa { ggml::set_input(b.inp.mask_swa); }
    ggml::set_input(b.inp.s_copy);
    if m.hp.arch == model::ARCH_T5 {
        guard c.n_enc > 0 else { abort("llama_encode must be called first"); }
        b.inp.pos_bucket = ggml::new_tensor_2d(gctx, ggml::TYPE_I32, b.n_kv, n_tokens);
        b.inp.cross_embd = ggml::new_tensor_2d(gctx, ggml::TYPE_F32, m.hp.n_embd, c.n_enc);
        b.inp.cross_mask = ggml::new_tensor_2d(gctx, ggml::TYPE_F16, c.n_enc, n_tokens);
        ggml::set_input(b.inp.pos_bucket);
        ggml::set_input(b.inp.cross_embd);
        ggml::set_input(b.inp.cross_mask);
    }
    if m.hp.f_attn_temp_scale != 0.0 {
        b.inp.attn_scale = ggml::new_tensor_3d(gctx, ggml::TYPE_F32, 1, 1, n_tokens);
        ggml::set_input(b.inp.attn_scale);
    }

    let out = match m.hp.arch {
        model::ARCH_GEMMA3 => build_gemma3(b, m, c),
        model::ARCH_LLAMA => build_llama(b, m, c),
        model::ARCH_LFM2 => build_lfm2(b, m, c),
        model::ARCH_LFM3 => build_lfm3(b, m, c),
        model::ARCH_QWEN2, model::ARCH_QWEN3 => build_qwen(b, m, c),
        model::ARCH_QWEN35 => build_qwen35(b, m, c),
        model::ARCH_GEMMA2 => build_gemma2(b, m, c),
        model::ARCH_PHI2 => build_phi2(b, m, c),
        model::ARCH_PHI3 => build_phi3(b, m, c),
        model::ARCH_STABLELM => build_stablelm(b, m, c),
        model::ARCH_SMOLLM3, model::ARCH_GRANITE, model::ARCH_GRANITE_MOE => build_smollm3_granite(b, m, c),
        model::ARCH_SMALLTHINKER => build_smallthinker(b, m, c),
        model::ARCH_GEMMA => build_gemma(b, m, c),
        model::ARCH_OLMO, model::ARCH_REFACT => build_olmo_refact(b, m, c),
        model::ARCH_QWEN => build_qwen(b, m, c),
        model::ARCH_MPT, model::ARCH_JAIS, model::ARCH_NEMOTRON => build_classic(b, m, c),
        model::ARCH_MAINCODER => build_hunyuan_dense(b, m, c),
        model::ARCH_NANBEIGE => build_llama(b, m, c),
        model::ARCH_BITNET => build_bitnet(b, m, c),
        model::ARCH_CHATGLM => build_chatglm(b, m, c),
        model::ARCH_MAMBA, model::ARCH_MAMBA2 => build_mamba(b, m, c),
        model::ARCH_FALCON_H1, model::ARCH_GRANITE_HYBRID, model::ARCH_JAMBA => build_hybrid(b, m, c),
        model::ARCH_PLAMO2 => build_plamo2(b, m, c),
        model::ARCH_RWKV6, model::ARCH_RWKV7 => build_rwkv(b, m, c),
        model::ARCH_MINICPM3 => build_minicpm3(b, m, c),
        model::ARCH_QWEN2VL, model::ARCH_QWEN3VL => build_qwen(b, m, c),
        model::ARCH_NEMOTRON_H => build_nemotron_h(b, m, c),
        model::ARCH_AFMOE => build_afmoe(b, m, c),
        model::ARCH_GEMMA3N => build_gemma3n(b, m, c),
        model::ARCH_GEMMA4 => build_gemma4(b, m, c),
        model::ARCH_T5 => build_t5_dec(b, m, c),
        model::ARCH_INTERNLM2, model::ARCH_EXAONE, model::ARCH_ERNIE4_5, model::ARCH_MISTRAL3 => build_llama(b, m, c),
        model::ARCH_ARCEE => build_arcee(b, m, c),
        model::ARCH_OLMO2, model::ARCH_EXAONE4 => build_post_norm(b, m, c),
        model::ARCH_HUNYUAN_DENSE => build_hunyuan_dense(b, m, c),
        model::ARCH_GPT2, model::ARCH_STARCODER, model::ARCH_GPTNEOX, model::ARCH_STARCODER2, model::ARCH_BLOOM => build_classic(b, m, c),
        model::ARCH_OPENELM => build_openelm(b, m, c),
        _ => { abort("decode: unknown architecture"); },
    };
    ggml::set_output(out);
    ggml::build_forward_expand(b.gf, out);
    guard ggml::gallocr_alloc_graph(c.galloc, b.gf) else { abort("decode: failed to allocate the graph"); }

    var pos: i32[>..] = [];
    var idxs: i64[>..] = [];
    for t in n_tokens { idxs.push(pos0 + t); }
    // M-RoPE: a token's position p is [p, p, p, 0], section after section
    for j in n_pos_per_embd(m) {
        for t in n_tokens { pos.push(if j < 3 { (pos0 + t) as i32 } else { 0 }); }
    }
    ggml::backend_set_i32(b.inp.tokens, tokens);
    if ggml::allocated(b.inp.pos) { ggml::backend_set_i32(b.inp.pos, pos); }   // bloom reads no positions
    if outputs.len > 0 { ggml::backend_set_i32(b.inp.out_ids, outputs); }
    else { ggml::backend_set_i32(b.inp.out_ids, [(n_tokens - 1) as i32]); }
    if ggml::allocated(b.inp.kv_idxs) { ggml::backend_set_i64(b.inp.kv_idxs, idxs); }   // no attention: no cache cells
    if any_recurrent(m) { ggml::backend_set_i32(b.inp.s_copy, [0]); }
    if b.inp.pos_bucket.p != 0 {
        // a cell's position, or -1 where it is empty, seen from each token
        var buckets: i32[>..] = [];
        for t in n_tokens {
            for j in b.n_kv {
                let p0: i64 = if j < pos0 + n_tokens { j } else { -1 };
                buckets.push(relative_position_bucket(p0, pos0 + t, m.hp.n_rel_attn_bkts, false));
            }
        }
        ggml::backend_set_i32(b.inp.pos_bucket, buckets);
        ggml::backend_set_f32(b.inp.cross_embd, c.embd_enc);
        var cmask: f32[>..] = [];
        cmask.resize(c.n_enc * n_tokens, 0.0);
        ggml::backend_set_f16(b.inp.cross_mask, cmask);
    }
    if b.inp.attn_scale.p != 0 {
        // llm_graph_input_attn_temp: log(floor(pos / floor_scale) + 1) * scale + 1
        var scales: f32[>..] = [];
        for t in n_tokens {
            let p: f32 = pos0 + t;
            let fl: f32 = floor(p / m.hp.n_attn_temp_floor_scale);
            scales.push((log(fl as f64 + 1.0) * m.hp.f_attn_temp_scale + 1.0) as f32);
        }
        ggml::backend_set_f32(b.inp.attn_scale, scales);
    }
    var mask: f32[>..] = [];
    fill_mask(mask, b.n_kv, n_tokens, pos0, pos0 + n_tokens, 0, m.hp.f_max_alibi_bias > 0.0);
    if ggml::allocated(b.inp.mask) { ggml::backend_set_f16(b.inp.mask, mask); }
    if has_swa {
        fill_mask(mask, b.n_kv, n_tokens, pos0, pos0 + n_tokens, m.hp.n_swa);
        ggml::backend_set_f16(b.inp.mask_swa, mask);
    }

    ggml::backend_cpu_set_n_threads(m.backend, c.n_threads);
    guard ggml::backend_graph_compute(m.backend, b.gf) == 0 else { abort("decode: graph compute failed"); }

    print_traced(b);
    logits.clear();
    logits.resize(ggml::nelements(out), 0.0);
    ggml::backend_get_f32(out, logits, 0);
    ggml::free(gctx);
    c.n_past += n_tokens;
}
