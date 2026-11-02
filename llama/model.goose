// A model: its hyperparameters and its weights, loaded from a GGUF file.
// So far the dense decoder architectures: gemma3.
// port-of: src/llama-model.cpp src/llama-hparams.cpp src/llama-hparams.h src/llama-model-loader.cpp
import std;
import math;
import ggml from "../ggml";
import ggml.gguf;
import .vocab;
namespace model;

let MAX_LAYERS = 512;

// enum llama_swa_type
let SWA_NONE = 0;
let SWA_STANDARD = 1;

let ARCH_UNKNOWN = 0;
let ARCH_GEMMA3 = 1;

struct HParams {
    arch: i64,
    n_ctx_train: u32,
    n_embd: u32,
    n_layer: u32,
    n_head: u32[..512],
    n_head_kv: u32[..512],
    n_ff: u32[..512],
    is_swa: bool[..512],
    n_embd_head_k: u32,
    n_embd_head_v: u32,
    n_rot: u32,
    n_ctx_orig_yarn: u32,
    rope_type: i32,
    rope_freq_base_train: f32,
    rope_freq_scale_train: f32,
    rope_freq_base_train_swa: f32 = 10000.0,
    rope_freq_scale_train_swa: f32 = 1.0,
    yarn_ext_factor: f32 = -1.0,
    yarn_attn_factor: f32 = 1.0,
    yarn_beta_fast: f32 = 32.0,
    yarn_beta_slow: f32 = 1.0,
    rope_attn_factor: f32 = 1.0,
    rope_scaling_yarn: bool,
    n_swa: u32,
    swa_type: i64,
    f_norm_rms_eps: f32,
    f_attention_scale: f32,
    f_final_logit_softcapping: f32,
}

fn n_embd_k_gqa(hp: HParams&, il: i64) -> i64 { hp.n_embd_head_k as i64 * hp.n_head_kv[il] }
fn n_embd_v_gqa(hp: HParams&, il: i64) -> i64 { hp.n_embd_head_v as i64 * hp.n_head_kv[il] }

struct Layer {
    attn_norm: ggml::Tensor,
    wq: ggml::Tensor,
    wk: ggml::Tensor,
    wv: ggml::Tensor,
    wo: ggml::Tensor,
    attn_post_norm: ggml::Tensor,
    attn_q_norm: ggml::Tensor,
    attn_k_norm: ggml::Tensor,
    ffn_norm: ggml::Tensor,
    ffn_gate: ggml::Tensor,
    ffn_up: ggml::Tensor,
    ffn_down: ggml::Tensor,
    ffn_post_norm: ggml::Tensor,
}

struct Model {
    hp: HParams,
    meta: gguf::Ctx,
    ctx: ggml::Ctx,          // the weights' tensors
    buf: ggml::Buffer,       // and their data
    backend: ggml::Backend,
    tok_embd: ggml::Tensor,
    output_norm: ggml::Tensor,
    output: ggml::Tensor,
    layers: Layer[..512],
}

//
// metadata
//

fn arch_name_of(arch: i64) -> const u8[:] {
    match arch { ARCH_GEMMA3 => "gemma3", _ => "unknown" }
}

fn arch_from_name(name: const u8[:]) -> i64 {
    if name == "gemma3" { return ARCH_GEMMA3; }
    return ARCH_UNKNOWN;
}

// "<arch>.<key>"
fn akey(m: Model&, key: const u8[:]) -> u8[>..] { str(arch_name_of(m.hp.arch), ".", key) }

fn get_u32(m: Model&, key: const u8[:], v: u32&, required: bool = true) -> bool {
    let k = akey(m, key);
    let id = vocab::key_of_type(m.meta, k, gguf::UINT32);
    if id < 0 {
        if required { abort(str("key not found in model: ", k)); }
        return false;
    }
    v = gguf::get_val_u32(m.meta, id);
    return true;
}

fn get_f32(m: Model&, key: const u8[:], v: f32&, required: bool = true) -> bool {
    let k = akey(m, key);
    let id = vocab::key_of_type(m.meta, k, gguf::FLOAT32);
    if id < 0 {
        if required { abort(str("key not found in model: ", k)); }
        return false;
    }
    v = gguf::get_val_f32(m.meta, id);
    return true;
}

fn get_str(m: Model&, key: const u8[:], out: u8[>..]&) -> bool {
    return vocab::get_str(m.meta, akey(m, key), out);
}

// A per-layer value given as one number or as an array of n (get_key_or_arr).
fn get_u32_or_arr(m: Model&, key: const u8[:], out: u32[..512]&, n: i64, required: bool = true) -> bool {
    let k = akey(m, key);
    let id = gguf::find_key(m.meta, k);
    if id < 0 {
        if required { abort(str("key not found in model: ", k)); }
        return false;
    }
    if gguf::get_kv_type(m.meta, id) == gguf::ARRAY {
        let at = gguf::get_arr_type(m.meta, id);
        guard at == gguf::UINT32 || at == gguf::INT32 else { abort(str("invalid array type for ", k)); }
        guard gguf::get_arr_n(m.meta, id) == n else { abort(str("array length ", gguf::get_arr_n(m.meta, id), " for key ", k, " != ", n)); }
        var vals: i32[>..] = [];
        vals.resize(n, 0);
        gguf::get_arr_i32(m.meta, id, vals);
        for i in n { out[i] = vals[i] as u32; }
        return true;
    }
    var v: u32 = 0;
    get_u32(m, key, v);
    for i in n { out[i] = v; }
    return true;
}

fn load_hparams(m: Model&) {
    let hp .= m.hp;
    get_u32(m, "context_length", hp.n_ctx_train);
    get_u32(m, "embedding_length", hp.n_embd);
    get_u32(m, "block_count", hp.n_layer);
    guard hp.n_layer > 0 && hp.n_layer <= MAX_LAYERS else { abort("bad block_count"); }
    let n = hp.n_layer as i64;
    for i in n { hp.n_head.push(0); hp.n_head_kv.push(0); hp.n_ff.push(0); hp.is_swa.push(false); }
    get_u32_or_arr(m, "feed_forward_length", hp.n_ff, n, false);
    get_u32_or_arr(m, "attention.head_count", hp.n_head, n, false);
    for i in n { hp.n_head_kv[i] = hp.n_head[i]; }
    get_u32_or_arr(m, "attention.head_count_kv", hp.n_head_kv, n, false);

    hp.n_ctx_orig_yarn = hp.n_ctx_train;
    get_u32(m, "rope.scaling.original_context_length", hp.n_ctx_orig_yarn, false);
    hp.rope_freq_base_train = 10000.0;
    get_f32(m, "rope.freq_base", hp.rope_freq_base_train, false);
    {
        var scaling: u8[>..] = [];
        if get_str(m, "rope.scaling.type", scaling) {
            guard scaling == "none" || scaling == "linear" || scaling == "yarn" || scaling == "longrope" else {
                abort(str("unknown rope scaling type: ", scaling));
            }
            hp.rope_scaling_yarn = scaling == "yarn";
        }
    }
    var ropescale: f32 = 0.0;
    if !get_f32(m, "rope.scaling.factor", ropescale, false) { get_f32(m, "rope.scale_linear", ropescale, false); }
    hp.rope_freq_scale_train = if ropescale == 0.0 { 1.0 } else { 1.0 / ropescale };
    get_f32(m, "rope.scaling.attn_factor", hp.rope_attn_factor, false);

    if hp.n_head[0] > 0 {
        hp.n_embd_head_k = hp.n_embd / hp.n_head[0];
        get_u32(m, "attention.key_length", hp.n_embd_head_k, false);
        hp.n_embd_head_v = hp.n_embd / hp.n_head[0];
        get_u32(m, "attention.value_length", hp.n_embd_head_v, false);
        hp.n_rot = hp.n_embd_head_k;
        get_u32(m, "rope.dimension_count", hp.n_rot, false);
    }

    match hp.arch {
        ARCH_GEMMA3 => load_hparams_gemma3(m),
        _ => abort("unknown architecture"),
    }
}

fn set_swa_pattern(hp: HParams&, n_pattern: u32, dense_first: bool) {
    for il in hp.n_layer as i64 {
        hp.is_swa[il] = if dense_first { n_pattern == 0 || il % n_pattern != 0 }
                        else { n_pattern == 0 || il % n_pattern < n_pattern - 1 };
    }
}

fn load_swa_pattern(m: Model&, n_pattern_default: u32, dense_first: bool = false) {
    let k = akey(m, "attention.sliding_window_pattern");
    let id = gguf::find_key(m.meta, k);
    if id >= 0 && gguf::get_kv_type(m.meta, id) == gguf::ARRAY {
        let n = gguf::get_arr_n(m.meta, id);
        var vals: u8[>..] = [];
        vals.resize(n, 0);
        gguf::get_arr_u8(m.meta, id, vals);
        for i in min(n, m.hp.n_layer as i64) { m.hp.is_swa[i] = vals[i] != 0; }
        return;
    }
    var n_pattern = n_pattern_default;
    get_u32(m, "attention.sliding_window_pattern", n_pattern, false);
    set_swa_pattern(m.hp, n_pattern, dense_first);
}

fn load_hparams_gemma3(m: Model&) {
    let found_swa = get_u32(m, "attention.sliding_window", m.hp.n_swa, false);
    if found_swa && m.hp.n_swa > 0 {
        m.hp.swa_type = SWA_STANDARD;
        load_swa_pattern(m, 6);
        get_f32(m, "rope.freq_base_swa", m.hp.rope_freq_base_train_swa, false);
    } else {
        m.hp.swa_type = SWA_NONE;
    }
    m.hp.f_final_logit_softcapping = 0.0;
    get_f32(m, "final_logit_softcapping", m.hp.f_final_logit_softcapping, false);
    get_f32(m, "attention.layer_norm_rms_epsilon", m.hp.f_norm_rms_eps);
    // ref: https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/config.py#L289
    let is_27b = m.hp.n_layer == 62;
    m.hp.f_attention_scale = if is_27b { 1.0 / sqrt((m.hp.n_embd / m.hp.n_head[0]) as f32) }
                             else { 1.0 / sqrt(m.hp.n_embd_head_k as f32) };
    m.hp.rope_type = ggml::ROPE_TYPE_NEOX;
}

//
// tensors
//

fn tensor(m: Model&, name: const u8[:], required: bool = true) -> ggml::Tensor {
    let t = ggml::get_tensor(m.ctx, name);
    if t.p == 0 && required { abort(str("missing tensor '", name, "'")); }
    return t;
}

fn layer_tensor(m: Model&, il: i64, name: const u8[:], required: bool = true) -> ggml::Tensor {
    return tensor(m, str("blk.", il, ".", name, ".weight"), required);
}

fn load_tensors_gemma3(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }       // tied to the input embeddings
    for il in m.hp.n_layer as i64 {
        m.layers.push(Layer {
            attn_norm: layer_tensor(m, il, "attn_norm"),
            wq: layer_tensor(m, il, "attn_q"),
            wk: layer_tensor(m, il, "attn_k"),
            wv: layer_tensor(m, il, "attn_v"),
            wo: layer_tensor(m, il, "attn_output"),
            attn_post_norm: layer_tensor(m, il, "post_attention_norm"),
            attn_q_norm: layer_tensor(m, il, "attn_q_norm"),
            attn_k_norm: layer_tensor(m, il, "attn_k_norm"),
            ffn_norm: layer_tensor(m, il, "ffn_norm"),
            ffn_gate: layer_tensor(m, il, "ffn_gate"),
            ffn_up: layer_tensor(m, il, "ffn_up"),
            ffn_down: layer_tensor(m, il, "ffn_down"),
            ffn_post_norm: layer_tensor(m, il, "post_ffw_norm"),
        });
    }
}

// Loads a model, and its vocabulary into the vocab module.
fn load(m: Model&, path: const u8[:]) {
    m.ctx = ggml::Ctx { p: 0 };
    m.meta = gguf::init_with_tensors(path, true, m.ctx);
    guard m.meta.p != 0 else { abort(str("failed to load model from ", path)); }

    var arch: u8[>..] = [];
    guard vocab::get_str(m.meta, "general.architecture", arch) else { abort("key not found in model: general.architecture"); }
    m.hp.arch = arch_from_name(arch);
    guard m.hp.arch != ARCH_UNKNOWN else { abort(str("unknown model architecture: '", arch, "'")); }

    load_hparams(m);
    vocab::load(m.meta);

    m.backend = ggml::backend_cpu_init();
    m.buf = ggml::backend_alloc_ctx_tensors(m.ctx, m.backend);
    guard gguf::load_tensor_data(m.meta, m.ctx, path) else { abort(str("failed to read the tensors of ", path)); }
    match m.hp.arch {
        ARCH_GEMMA3 => load_tensors_gemma3(m),
        _ => abort("unknown architecture"),
    }
}
