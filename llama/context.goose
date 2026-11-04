// A context: a model's KV cache, and the decoding of batches of tokens
// through the model's graph. One sequence; flash attention, as llama.cpp
// picks on the CPU. The graph of each architecture follows llama.cpp's
// src/models/<arch>.cpp operation by operation.
// port-of: src/llama-context.cpp src/llama-graph.cpp src/llama-kv-cache.cpp src/models/gemma3.cpp src/models/llama.cpp
// port-of: src/models/lfm2.cpp src/models/lfm-base.cpp src/llama-memory-recurrent.cpp
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
    r: ggml::Tensor[..512],      // per recurrent layer: its state, [n_embd_r, 1], f32
    n_past: i64,                 // the cells in use: cell i holds position i
    galloc: ggml::Gallocr,
}

// llama.cpp pads the context to 256 cells.
fn pad_to(n: i64, to: i64) -> i64 { (n + to - 1) / to * to }

fn init(c: Context&, m: model::Model&, n_ctx: i64) {
    c.n_ctx = pad_to(n_ctx, 256);
    if c.n_threads <= 0 { c.n_threads = ggml::default_threads(); }
    let n_layer = m.hp.n_layer as i64;
    c.kv_ctx = ggml::init(3 * n_layer * ggml::tensor_overhead(), true);
    for il in n_layer {
        if m.hp.is_recr[il] {
            c.k.push(ggml::NONE);
            c.v.push(ggml::NONE);
            c.r.push(ggml::new_tensor_2d(c.kv_ctx, ggml::TYPE_F32, model::n_embd_r(m.hp), 1));
            ggml::set_name(c.r[il], str("cache_r_l", il));
            continue;
        }
        c.k.push(ggml::new_tensor_2d(c.kv_ctx, ggml::TYPE_F16, model::n_embd_k_gqa(m.hp, il), c.n_ctx));
        c.v.push(ggml::new_tensor_2d(c.kv_ctx, ggml::TYPE_F16, model::n_embd_v_gqa(m.hp, il), c.n_ctx));
        c.r.push(ggml::NONE);
        ggml::set_name(c.k[il], str("cache_k_l", il));
        ggml::set_name(c.v[il], str("cache_v_l", il));
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
        let ty = ggml::tensor_type(qkv);
        let nb1 = ggml::nb(qkv, 1);
        let q = ggml::view_4d(ctx, qkv, head, n_head, b.n_tokens, 1, ggml::row_size(ty, head), nb1, nb1 * b.n_tokens, 0);
        let k = ggml::view_4d(ctx, qkv, head, n_head_kv, b.n_tokens, 1, ggml::row_size(ty, head), nb1, nb1 * b.n_tokens,
                              ggml::row_size(ty, head * n_head));
        let v = ggml::view_4d(ctx, qkv, head, n_head_kv, b.n_tokens, 1, ggml::row_size(ty, head), nb1, nb1 * b.n_tokens,
                              ggml::row_size(ty, head * (n_head + n_head_kv)));
        return q, k, v;
    }
    let q = add_bias(b, ggml::mul_mat(ctx, ly.wq, cur), ly.bq);
    let k = add_bias(b, ggml::mul_mat(ctx, ly.wk, cur), ly.bk);
    let v = add_bias(b, ggml::mul_mat(ctx, ly.wv, cur), ly.bv);
    return ggml::reshape_3d(ctx, q, head, n_head, b.n_tokens),
           ggml::reshape_3d(ctx, k, head, n_head_kv, b.n_tokens),
           ggml::reshape_3d(ctx, v, head, n_head_kv, b.n_tokens);
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
    let hp .= m.hp;
    let ctx = b.ctx;
    let n_embd_k = model::n_embd_k_gqa(hp, il);
    let n_embd_v = model::n_embd_v_gqa(hp, il);
    ggml::build_forward_expand(b.gf, q);
    ggml::build_forward_expand(b.gf, v_cur);
    ggml::build_forward_expand(b.gf, k_cur);

    ggml::build_forward_expand(b.gf, ggml::set_rows(ctx, c.k[il], ggml::reshape_2d(ctx, k_cur, n_embd_k, b.n_tokens), b.inp.kv_idxs));
    ggml::build_forward_expand(b.gf, ggml::set_rows(ctx, c.v[il], ggml::reshape_2d(ctx, v_cur, n_embd_v, b.n_tokens), b.inp.kv_idxs));

    let tk = ggml::tensor_type(c.k[il]);
    let tv = ggml::tensor_type(c.v[il]);
    let n_head_kv = hp.n_head_kv[il] as i64;
    let k = ggml::view_4d(ctx, c.k[il], hp.n_embd_head_k, n_head_kv, b.n_kv, 1,
                          ggml::row_size(tk, hp.n_embd_head_k), ggml::row_size(tk, n_embd_k),
                          ggml::row_size(tk, n_embd_k * c.n_ctx), 0);
    let v = ggml::view_4d(ctx, c.v[il], hp.n_embd_head_v, n_head_kv, b.n_kv, 1,
                          ggml::row_size(tv, hp.n_embd_head_v), ggml::row_size(tv, n_embd_v),
                          ggml::row_size(tv, n_embd_v * c.n_ctx), 0);

    let qp = ggml::permute(ctx, q, 0, 2, 1, 3);
    let kp = ggml::permute(ctx, k, 0, 2, 1, 3);
    let vp = ggml::permute(ctx, v, 0, 2, 1, 3);
    let mask = if hp.is_swa[il] { b.inp.mask_swa } else { b.inp.mask };
    var cur = ggml::flash_attn_ext(ctx, qp, kp, vp, mask, kq_scale, 0.0, 0.0);
    ggml::prec_set_acc(cur, ggml::PREC_F32);
    cur = ggml::reshape_2d(ctx, cur, ggml::ne(cur, 0) * ggml::ne(cur, 1), ggml::ne(cur, 2) * ggml::ne(cur, 3));
    if il == 0 { trace(b, "kqv_out-0", cur); trace(b, "Vcur-0", v_cur); }
    ggml::build_forward_expand(b.gf, cur);
    return add_bias(b, ggml::mul_mat(ctx, wo, cur), wo_b);
}

fn rope(b: Build&, m: model::Model&, x: ggml::Tensor, freq_base: f32, freq_scale: f32,
        factors: ggml::Tensor = ggml::NONE) -> ggml::Tensor {
    let hp .= m.hp;
    let ext_factor: f32 = if hp.yarn_ext_factor >= 0.0 { hp.yarn_ext_factor } else if hp.rope_scaling_yarn { 1.0 } else { 0.0 };
    guard ext_factor == 0.0 else { abort("YaRN rope scaling: not ported yet"); }
    let attn_factor = hp.yarn_attn_factor * hp.rope_attn_factor;
    return ggml::rope_ext(b.ctx, x, b.inp.pos, factors, hp.n_rot as i32, hp.rope_type,
                          hp.n_ctx_orig_yarn as i32, freq_base, freq_scale, ext_factor, attn_factor,
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

let NEG_INF: f32 = -1.0 / 0.0;

// mask[t][j] for token t (at position pos0 + t) and cell j: 0, or -inf when
// the cell is in the token's future, unused, or (swa) out of its window.
fn fill_mask(mask: f32[>..]&, n_kv: i64, n_tokens: i64, pos0: i64, n_used: i64, swa: u32) {
    mask.clear();
    for t in n_tokens {
        let p1 = pos0 + t;
        for j in n_kv {
            let masked = j >= n_used || j > p1 || (swa > 0 && p1 - j >= swa);
            mask.push(if masked { NEG_INF } else { 0.0 });
        }
    }
}

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
            pos: ggml::new_tensor_1d(gctx, ggml::TYPE_I32, n_tokens),
            out_ids: ggml::new_tensor_1d(gctx, ggml::TYPE_I32, n_outputs),
            kv_idxs: ggml::new_tensor_1d(gctx, ggml::TYPE_I64, n_tokens),
            mask: ggml::NONE,
            mask_swa: ggml::NONE,
            s_copy: ggml::new_tensor_1d(gctx, ggml::TYPE_I32, 1),
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

    let out = match m.hp.arch {
        model::ARCH_GEMMA3 => build_gemma3(b, m, c),
        model::ARCH_LLAMA => build_llama(b, m, c),
        model::ARCH_LFM2 => build_lfm2(b, m, c),
        _ => { abort("decode: unknown architecture"); },
    };
    ggml::set_output(out);
    ggml::build_forward_expand(b.gf, out);
    guard ggml::gallocr_alloc_graph(c.galloc, b.gf) else { abort("decode: failed to allocate the graph"); }

    var pos: i32[>..] = [];
    var idxs: i64[>..] = [];
    for t in n_tokens { pos.push((pos0 + t) as i32); idxs.push(pos0 + t); }
    ggml::backend_set_i32(b.inp.tokens, tokens);
    ggml::backend_set_i32(b.inp.pos, pos);
    if outputs.len > 0 { ggml::backend_set_i32(b.inp.out_ids, outputs); }
    else { ggml::backend_set_i32(b.inp.out_ids, [(n_tokens - 1) as i32]); }
    ggml::backend_set_i64(b.inp.kv_idxs, idxs);
    if any_recurrent(m) { ggml::backend_set_i32(b.inp.s_copy, [0]); }
    var mask: f32[>..] = [];
    fill_mask(mask, b.n_kv, n_tokens, pos0, pos0 + n_tokens, 0);
    ggml::backend_set_f16(b.inp.mask, mask);
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
