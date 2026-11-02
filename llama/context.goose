// A context: a model's KV cache, and the decoding of batches of tokens
// through the model's graph. One sequence; flash attention, as llama.cpp
// picks on the CPU. The graph of each architecture follows llama.cpp's
// src/models/<arch>.cpp operation by operation.
// port-of: src/llama-context.cpp src/llama-graph.cpp src/llama-kv-cache.cpp src/models/gemma3.cpp
import std;
import math;
import ggml from "../ggml";
import .model;
namespace context;

let MAX_NODES = 8192;

struct Context {
    n_ctx: i64,
    n_threads: i32 = 4,
    kv_ctx: ggml::Ctx,
    kv_buf: ggml::Buffer,
    k: ggml::Tensor[..512],      // per layer: [n_embd_k_gqa, n_ctx], f16
    v: ggml::Tensor[..512],
    n_past: i64,                 // the cells in use: cell i holds position i
    galloc: ggml::Gallocr,
}

// llama.cpp pads the context to 256 cells.
fn pad_to(n: i64, to: i64) -> i64 { (n + to - 1) / to * to }

fn init(c: Context&, m: model::Model&, n_ctx: i64) {
    c.n_ctx = pad_to(n_ctx, 256);
    let n_layer = m.hp.n_layer as i64;
    c.kv_ctx = ggml::init(2 * n_layer * ggml::tensor_overhead(), true);
    for il in n_layer {
        c.k.push(ggml::new_tensor_2d(c.kv_ctx, ggml::TYPE_F16, model::n_embd_k_gqa(m.hp, il), c.n_ctx));
        c.v.push(ggml::new_tensor_2d(c.kv_ctx, ggml::TYPE_F16, model::n_embd_v_gqa(m.hp, il), c.n_ctx));
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
}

// Per graph: what the graph builders read.
struct Build {
    ctx: ggml::Ctx,
    gf: ggml::Graph,
    n_tokens: i64,
    n_kv: i64,
    inp: Inputs,
}

fn build_norm_rms(b: Build&, m: model::Model&, cur: ggml::Tensor, w: ggml::Tensor) -> ggml::Tensor {
    let n = ggml::rms_norm(b.ctx, cur, m.hp.f_norm_rms_eps);
    return if w.p != 0 { ggml::mul(b.ctx, n, w) } else { n };
}

// Writes k and v of this batch into the layer's cache, and attends over the
// first n_kv cells; q, k, v are [head, n_head(_kv), n_tokens].
fn build_attn(b: Build&, m: model::Model&, c: Context&, il: i64, wo: ggml::Tensor,
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
    ggml::build_forward_expand(b.gf, cur);
    return ggml::mul_mat(ctx, wo, cur);
}

fn rope(b: Build&, m: model::Model&, x: ggml::Tensor, freq_base: f32, freq_scale: f32) -> ggml::Tensor {
    let hp .= m.hp;
    let ext_factor: f32 = if hp.yarn_ext_factor >= 0.0 { hp.yarn_ext_factor } else if hp.rope_scaling_yarn { 1.0 } else { 0.0 };
    guard ext_factor == 0.0 else { abort("YaRN rope scaling: not ported yet"); }
    let attn_factor = hp.yarn_attn_factor * hp.rope_attn_factor;
    return ggml::rope_ext(b.ctx, x, b.inp.pos, ggml::NONE, hp.n_rot as i32, hp.rope_type,
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
        var q = ggml::reshape_3d(ctx, ggml::mul_mat(ctx, ly.wq, cur), n_embd_head, hp.n_head[il], b.n_tokens);
        var k = ggml::reshape_3d(ctx, ggml::mul_mat(ctx, ly.wk, cur), n_embd_head, hp.n_head_kv[il], b.n_tokens);
        let v = ggml::reshape_3d(ctx, ggml::mul_mat(ctx, ly.wv, cur), n_embd_head, hp.n_head_kv[il], b.n_tokens);
        q = build_norm_rms(b, m, q, ly.attn_q_norm);
        q = rope(b, m, q, freq_base, freq_scale);
        k = build_norm_rms(b, m, k, ly.attn_k_norm);
        k = rope(b, m, k, freq_base, freq_scale);
        // ref: https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/model.py#L315
        q = ggml::scale(ctx, q, hp.f_attention_scale);
        cur = build_attn(b, m, c, il, ly.wo, q, k, v, 1.0);

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

// Runs tokens, at positions n_past.., through the model; writes the logits
// of the last one to logits.
fn decode(c: Context&, m: model::Model&, tokens: const i32[:], logits: f32[>..]&) {
    let n_tokens = tokens.len;
    guard n_tokens > 0 && c.n_past + n_tokens <= c.n_ctx else { abort("decode: the context is full"); }
    let pos0 = c.n_past;

    let gctx = ggml::init(MAX_NODES * ggml::tensor_overhead() + ggml::graph_overhead_custom(MAX_NODES, false), true);
    var b = Build {
        ctx: gctx,
        gf: ggml::new_graph_custom(gctx, MAX_NODES, false),
        n_tokens: n_tokens,
        n_kv: c.n_ctx,
        inp: Inputs {
            tokens: ggml::new_tensor_1d(gctx, ggml::TYPE_I32, n_tokens),
            pos: ggml::new_tensor_1d(gctx, ggml::TYPE_I32, n_tokens),
            out_ids: ggml::new_tensor_1d(gctx, ggml::TYPE_I32, 1),
            kv_idxs: ggml::new_tensor_1d(gctx, ggml::TYPE_I64, n_tokens),
            mask: ggml::new_tensor_2d(gctx, ggml::TYPE_F16, c.n_ctx, n_tokens),
            mask_swa: ggml::new_tensor_2d(gctx, ggml::TYPE_F16, c.n_ctx, n_tokens),
        },
    };
    ggml::set_input(b.inp.tokens);
    ggml::set_input(b.inp.pos);
    ggml::set_input(b.inp.out_ids);
    ggml::set_input(b.inp.kv_idxs);
    ggml::set_input(b.inp.mask);
    ggml::set_input(b.inp.mask_swa);

    let out = match m.hp.arch {
        model::ARCH_GEMMA3 => build_gemma3(b, m, c),
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
    ggml::backend_set_i32(b.inp.out_ids, [(n_tokens - 1) as i32]);
    ggml::backend_set_i64(b.inp.kv_idxs, idxs);
    var mask: f32[>..] = [];
    fill_mask(mask, c.n_ctx, n_tokens, pos0, pos0 + n_tokens, 0);
    ggml::backend_set_f16(b.inp.mask, mask);
    fill_mask(mask, c.n_ctx, n_tokens, pos0, pos0 + n_tokens, if m.hp.swa_type == model::SWA_STANDARD { m.hp.n_swa } else { 0 });
    ggml::backend_set_f16(b.inp.mask_swa, mask);

    ggml::backend_cpu_set_n_threads(m.backend, c.n_threads);
    guard ggml::backend_graph_compute(m.backend, b.gf) == 0 else { abort("decode: graph compute failed"); }

    logits.clear();
    logits.resize(ggml::ne(out, 0), 0.0);
    ggml::backend_get_f32(out, logits, 0);
    ggml::free(gctx);
    c.n_past += n_tokens;
}
