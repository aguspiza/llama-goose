// A model: its hyperparameters and its weights, loaded from a GGUF file.
// So far: afmoe, arcee, bitnet, bloom, chatglm, cohere2, ernie4_5, falcon-h1, gemma3n, gemma4, exaone, exaone4, gemma, gemma2, gemma3, glm4, gpt2,
// gptneox, granite, granitehybrid, granitemoe, hrm_text, hunyuan-dense, jamba, internlm2, jais, k2-horizon, limite, llama, mistral3,
// maincoder, mamba, mamba2, minicpm, minicpm3, nemotron_h, mpt, nanbeige, nemotron, olmo, olmo2, openelm, pangu-embedded, phi2, qwen, refact,
// phi3, plamo2, plamo3, qwen2vl, qwen3vl, rwkv6, rwkv7, smallthinker, spark2_5, t5, smollm3, stablelm, starcoder, starcoder2, lfm2 (attention and short convolutions), lfm3, qwen2, qwen3 and qwen35
// (attention and gated delta nets).
// port-of: src/llama-model.cpp src/llama-hparams.cpp src/llama-hparams.h src/llama-model-loader.cpp src/models/lfm3.cpp src/models/qwen2.cpp src/models/qwen3.cpp src/models/qwen35.cpp src/models/gemma2.cpp
// port-of: src/models/phi2.cpp src/models/phi3.cpp src/models/stablelm.cpp src/models/smollm3.cpp src/models/granite.cpp
// port-of: src/models/internlm2.cpp src/models/exaone.cpp src/models/ernie4-5.cpp src/models/arcee.cpp src/models/olmo2.cpp
// port-of: src/models/exaone4.cpp src/models/hunyuan-dense.cpp src/models/hunyuan-vl.cpp
// port-of: src/models/gpt2.cpp src/models/starcoder.cpp src/models/gptneox.cpp src/models/starcoder2.cpp src/models/bloom.cpp src/models/openelm.cpp
// port-of: src/models/mistral3.cpp src/models/granite-moe.cpp src/models/smallthinker.cpp
// port-of: src/models/gemma.cpp src/models/olmo.cpp src/models/qwen.cpp src/models/refact.cpp src/models/mpt.cpp src/models/jais.cpp src/models/nemotron.cpp
// port-of: src/models/maincoder.cpp src/models/nanbeige.cpp src/models/bitnet.cpp src/models/chatglm.cpp
// port-of: src/models/mamba.cpp src/models/mamba2.cpp src/models/mamba-base.cpp
// port-of: src/models/falcon-h1.cpp src/models/granite-hybrid.cpp src/models/jamba.cpp src/models/plamo2.cpp
// port-of: src/models/rwkv6.cpp src/models/rwkv6-base.cpp src/models/rwkv7.cpp src/models/rwkv7-base.cpp src/models/minicpm3.cpp
// port-of: src/models/qwen2vl.cpp src/models/qwen3vl.cpp src/models/nemotron-h.cpp src/models/afmoe.cpp src/models/gemma3n.cpp src/models/gemma4.cpp src/models/t5.cpp
// port-of: src/models/minicpm.cpp src/models/plamo3.cpp src/models/cohere2.cpp src/models/pangu-embed.cpp src/models/limite.cpp src/models/hrm-text.cpp src/models/spark2-5.cpp src/models/k2-horizon.cpp src/models/glm4.cpp
import std;
import math;
import ggml from "../ggml";
import ggml.gguf;
import .vocab;
import .unicode;
import .ecma_regex;
namespace model;

let MAX_LAYERS = 512;

// enum llama_expert_gating_func_type
enum Gating { None, Softmax, Sigmoid, SoftmaxWeight, SqrtSoftplus }

// The gating of the GGUF's expert_gating_func value.
fn gating_of(v: u32) -> Gating {
    match v { 1 => Gating.Softmax, 2 => Gating.Sigmoid, 3 => Gating.SoftmaxWeight, 4 => Gating.SqrtSoftplus, _ => Gating.None }
}

// enum llama_swa_type
enum Swa { None, Standard }

enum Arch {
    Unknown, Gemma3, Llama, Lfm2, Lfm3, Qwen2, Qwen3, Qwen35, Gemma2, Phi2, Phi3, Stablelm,
    Smollm3, Granite, Internlm2, Exaone, Ernie4_5, Arcee, Olmo2, Exaone4, HunyuanDense, Gpt2,
    Starcoder, Gptneox, Starcoder2, Bloom, Openelm, Mistral3, GraniteMoe, Smallthinker, Gemma,
    Olmo, Qwen, Refact, Mpt, Jais, Nemotron, Maincoder, Nanbeige, Bitnet, Chatglm, Mamba,
    Mamba2, FalconH1, GraniteHybrid, Jamba, Plamo2, Rwkv6, Rwkv7, Minicpm3, Qwen2vl, Qwen3vl,
    NemotronH, Afmoe, Gemma3n, Gemma4, T5, Minicpm, Plamo3, Cohere2, PanguEmbed, Limite,
    HrmText, Spark2_5, K2Horizon, Glm4,
}

struct HParams {
    arch: Arch,
    n_ctx_train: u32,
    n_embd: u32,
    n_layer: u32,
    n_head: u32[..512],
    n_head_kv: u32[..512],
    n_ff: u32[..512],
    is_swa: bool[..512],
    kv_source: i64[..512],       // the layer whose cache layer il reads (itself, or gemma3n's shared ones)
    n_layer_kv_from_start: i64 = -1,
    n_altup: i64 = 4,            // gemma3n's AltUp: the inputs,
    i_altup_act: i64 = 0,        // the active one,
    n_embd_altup: i64 = 256,     // the per-layer inputs' size
    is_recr: bool[..512],        // a recurrent layer (lfm2's short convolutions)
    n_shortconv_l_cache: u32,
    n_loops: u32 = 1,            // nanbeige: the layers run this many times,
    n_layer_phys: u32,           // n_layer / n_loops of them stored
    skip_loop_final_norm: bool,  // without the output norm between runs
    n_expert: u32,               // MoE: the experts,
    n_expert_used: u32,          // how many each token uses,
    n_ff_shexp: u32,             // the shared expert's size,
    expert_weights_scale: f32,   // a scale of the experts' weights
    expert_gating_func: Gating = Gating.None,  // and how the router's logits become them
    expert_weights_norm: bool,   // the weights divided by their sum
    n_expert_shared: u32,
    n_layer_dense_lead: u32,     // the first layers without experts
    ssm_d_conv: u32,             // gated delta nets: the convolution's width,
    ssm_d_inner: u32,            // the values' size,
    ssm_d_state: u32,            // a head's size,
    ssm_dt_rank: u32,            // the value heads
    ssm_n_group: u32,            // and the key heads
    n_lora_q: u32,               // minicpm3: the ranks of its Q and KV LoRA
    n_lora_kv: u32,
    wkv_head_size: u32,          // rwkv: a head's size,
    token_shift_count: u32 = 2,  // the shifted inputs a layer keeps
    rescale_every_n_layers: u32, // rwkv6: halve the residual every n layers
    ssm_dt_b_c_rms: bool,        // mamba: dt, B and C RMS-normed (FalconMamba)
    rope_sections: i32[4],
    n_deepstack_layers: u32,     // qwen3vl: image features added to its first layers
    n_embd_head_k: u32,
    n_embd_head_v: u32,
    n_rot: u32,
    n_embd_head_k_swa: u32,      // the sliding window layers' head and rope sizes
    n_embd_head_v_swa: u32,
    n_rot_swa: u32,
    n_embd_per_layer: u32,       // gemma4's per-layer inputs
    n_rel_attn_bkts: u32,        // t5: the relative position buckets,
    dec_start_token_id: i32 = -1, // the decoder's first token,
    dec_n_layer: u32,            // and its layers
    n_ctx_orig_yarn: u32,
    rope_type: i32,
    rope_freq_base_train: f32,
    rope_freq_scale_train: f32,
    rope_freq_base_train_swa: f32 = 10000.0,
    rope_freq_scale_train_swa: f32 = 1.0,
    yarn_ext_factor: f32 = -1.0,     // after load: the values rope reads (llama_context's cparams)
    yarn_attn_factor: f32 = 1.0,
    yarn_beta_fast: f32 = 32.0,
    yarn_beta_slow: f32 = 1.0,
    rope_attn_factor: f32 = 1.0,
    rope_scaling_alpha: f32,
    rope_scaling_yarn: bool,
    rope_scaling_none: bool,
    rope_yarn_log_mul: f32,
    n_swa: u32,
    swa_type: Swa = Swa.None,
    f_norm_rms_eps: f32,
    f_norm_eps: f32,             // LayerNorm's
    f_logit_scale: f32,          // granite: the logits divided by it,
    f_residual_scale: f32,       // each residual branch scaled by it,
    f_embedding_scale: f32,      // and the input embeddings
    n_no_rope_layer_step: u32,   // smollm3: every n-th layer without rope
    rope_finetuned: bool = true, // granite: false, no rope at all
    f_attention_scale: f32,
    attn_soft_cap: bool,
    f_attn_temp_scale: f32,      // mistral3: queries scaled by their position,
    n_attn_temp_floor_scale: u32, // in steps of this many
    f_clamp_kqv: f32,            // olmo, mpt: Q, K and V clamped to ±this
    f_max_alibi_bias: f32,       // bloom: ALiBi instead of rope
    use_par_res: bool,           // gptneox: attention and feed-forward in parallel
    f_attn_logit_softcapping: f32 = 50.0,
    f_final_logit_softcapping: f32,
    f_xsa_eps: f32,                          // limite: exclusive self attention's,
    final_logit_sigmoid_capping: f32[3],     // the logits' scale * sigmoid((x + shift) / temperature),
    n_mudd_ff: u32,                          // and the dense connections across layers:
    n_mudd_taps: u32,                        // each layer mixes this many earlier inputs,
    mudd_tap_idx: i32[..512],                // [n_layer][n_mudd_taps], -1: none
    n_hrm_layers_per_stack: u32,             // hrm_text: a low and a high stack of this many layers,
    n_hrm_h_cycles: u32,                     // the high one run this many times,
    n_hrm_l_cycles: u32,                     // the low one this many times before each
    n_norm_groups: u32,                      // k2-horizon: RMS norms over this many slices
}

// A layer's head and rope sizes: the sliding window ones on such layers.
fn head_k(hp: HParams&, il: i64) -> i64 { if hp.is_swa[il] { hp.n_embd_head_k_swa } else { hp.n_embd_head_k } }
fn head_v(hp: HParams&, il: i64) -> i64 { if hp.is_swa[il] { hp.n_embd_head_v_swa } else { hp.n_embd_head_v } }
fn rot(hp: HParams&, il: i64) -> i64 { if hp.is_swa[il] { hp.n_rot_swa } else { hp.n_rot } }

fn n_embd_k_gqa(hp: HParams&, il: i64) -> i64 { head_k(hp, il) * hp.n_head_kv[il] }
fn n_embd_v_gqa(hp: HParams&, il: i64) -> i64 { head_v(hp, il) * hp.n_head_kv[il] }

// A layer's tensors; p == 0 for the ones its architecture or file lacks.
struct Layer {
    attn_norm: ggml::Tensor,
    attn_norm_b: ggml::Tensor,
    attn_sub_norm: ggml::Tensor,     // bitnet: norms before attn_output and ffn_down,
    ffn_sub_norm: ggml::Tensor,
    wq_s: ggml::Tensor,              // and scales of the matmuls' results
    wk_s: ggml::Tensor,
    wv_s: ggml::Tensor,
    wo_s: ggml::Tensor,
    ffn_up_s: ggml::Tensor,
    ffn_gate_s: ggml::Tensor,
    ffn_down_s: ggml::Tensor,
    wqkv: ggml::Tensor,
    wq: ggml::Tensor,
    wk: ggml::Tensor,
    wv: ggml::Tensor,
    bqkv: ggml::Tensor,
    bq: ggml::Tensor,
    bk: ggml::Tensor,
    bv: ggml::Tensor,
    wo: ggml::Tensor,
    bo: ggml::Tensor,
    rope_freqs: ggml::Tensor,
    rope_long: ggml::Tensor,         // phi3, granite: the factors past n_ctx_orig_yarn
    rope_short: ggml::Tensor,        // and up to it
    attn_post_norm: ggml::Tensor,
    attn_q_norm: ggml::Tensor,
    attn_k_norm: ggml::Tensor,
    ffn_norm: ggml::Tensor,
    ffn_norm_b: ggml::Tensor,
    ffn_gate: ggml::Tensor,
    ffn_up: ggml::Tensor,
    ffn_down: ggml::Tensor,
    ffn_gate_b: ggml::Tensor,
    ffn_up_b: ggml::Tensor,
    ffn_down_b: ggml::Tensor,
    ffn_post_norm: ggml::Tensor,
    out_scale: ggml::Tensor,         // gemma4: the layer's output times this
    ffn_act: ggml::Tensor,           // mpt: the GELU's output divided by these scales
    attn_q_norm_b: ggml::Tensor,
    attn_k_norm_b: ggml::Tensor,
    attn_gate: ggml::Tensor,         // lfm3: a sigmoid gate per head on the attention's output; qwen35: z
    shortconv_conv: ggml::Tensor,
    shortconv_in_proj: ggml::Tensor,
    shortconv_out_proj: ggml::Tensor,
    ffn_gate_inp: ggml::Tensor,      // MoE: the router,
    ffn_exp_probs_b: ggml::Tensor,   // a bias of its choice,
    ffn_up_exps: ggml::Tensor,       // the experts,
    ffn_gate_exps: ggml::Tensor,
    ffn_down_exps: ggml::Tensor,
    ffn_up_shexp: ggml::Tensor,      // and a shared expert
    ffn_gate_shexp: ggml::Tensor,
    ffn_down_shexp: ggml::Tensor,
    attn_q_a_norm: ggml::Tensor,     // minicpm3's low rank Q and KV
    attn_kv_a_norm: ggml::Tensor,
    wq_a: ggml::Tensor,
    wq_b: ggml::Tensor,
    wkv_a_mqa: ggml::Tensor,
    wkv_b: ggml::Tensor,
    per_layer_inp_gate: ggml::Tensor,    // gemma3n's AltUp and Laurel
    per_layer_proj: ggml::Tensor,
    per_layer_post_norm: ggml::Tensor,
    altup_correct_coef: ggml::Tensor,
    altup_correct_scale: ggml::Tensor,
    altup_predict_coef: ggml::Tensor,
    altup_router: ggml::Tensor,
    altup_router_norm: ggml::Tensor,
    laurel_l: ggml::Tensor,
    laurel_r: ggml::Tensor,
    laurel_post_norm: ggml::Tensor,
    attn_norm_enc: ggml::Tensor,     // t5's encoder layers
    attn_rel_b_enc: ggml::Tensor,
    wq_enc: ggml::Tensor,
    wk_enc: ggml::Tensor,
    wv_enc: ggml::Tensor,
    wo_enc: ggml::Tensor,
    ffn_norm_enc: ggml::Tensor,
    ffn_gate_enc: ggml::Tensor,
    ffn_down_enc: ggml::Tensor,
    ffn_up_enc: ggml::Tensor,
    attn_rel_b: ggml::Tensor,        // and its decoder's
    attn_norm_cross: ggml::Tensor,
    wq_cross: ggml::Tensor,
    wk_cross: ggml::Tensor,
    wv_cross: ggml::Tensor,
    wo_cross: ggml::Tensor,
    attn_norm_2: ggml::Tensor,       // rwkv
    attn_norm_2_b: ggml::Tensor,
    time_mix_w0: ggml::Tensor,
    time_mix_w1: ggml::Tensor,
    time_mix_w2: ggml::Tensor,
    time_mix_a0: ggml::Tensor,
    time_mix_a1: ggml::Tensor,
    time_mix_a2: ggml::Tensor,
    time_mix_v0: ggml::Tensor,
    time_mix_v1: ggml::Tensor,
    time_mix_v2: ggml::Tensor,
    time_mix_g1: ggml::Tensor,
    time_mix_g2: ggml::Tensor,
    time_mix_k_k: ggml::Tensor,
    time_mix_k_a: ggml::Tensor,
    time_mix_r_k: ggml::Tensor,
    time_mix_lerp_x: ggml::Tensor,
    time_mix_lerp_w: ggml::Tensor,
    time_mix_lerp_k: ggml::Tensor,
    time_mix_lerp_v: ggml::Tensor,
    time_mix_lerp_r: ggml::Tensor,
    time_mix_lerp_g: ggml::Tensor,
    time_mix_lerp_fused: ggml::Tensor,
    time_mix_first: ggml::Tensor,
    time_mix_decay: ggml::Tensor,
    time_mix_decay_w1: ggml::Tensor,
    time_mix_decay_w2: ggml::Tensor,
    time_mix_key: ggml::Tensor,
    time_mix_key_b: ggml::Tensor,
    time_mix_value: ggml::Tensor,
    time_mix_value_b: ggml::Tensor,
    time_mix_receptance: ggml::Tensor,
    time_mix_receptance_b: ggml::Tensor,
    time_mix_gate: ggml::Tensor,
    time_mix_ln: ggml::Tensor,
    time_mix_ln_b: ggml::Tensor,
    time_mix_output: ggml::Tensor,
    channel_mix_lerp_k: ggml::Tensor,
    channel_mix_lerp_r: ggml::Tensor,
    channel_mix_key: ggml::Tensor,
    channel_mix_value: ggml::Tensor,
    channel_mix_receptance: ggml::Tensor,
    ssm_conv1d: ggml::Tensor,        // qwen35's gated delta nets, mamba's state space models
    ssm_conv1d_b: ggml::Tensor,
    ssm_in: ggml::Tensor,
    ssm_x: ggml::Tensor,
    ssm_dt: ggml::Tensor,            // qwen35: a bias; mamba: a weight,
    ssm_dt_b: ggml::Tensor,          // and its bias
    ssm_d: ggml::Tensor,
    ssm_dt_norm: ggml::Tensor,       // jamba
    ssm_b_norm: ggml::Tensor,
    ssm_c_norm: ggml::Tensor,
    ssm_a: ggml::Tensor,
    ssm_beta: ggml::Tensor,
    ssm_alpha: ggml::Tensor,
    ssm_norm: ggml::Tensor,
    ssm_out: ggml::Tensor,
    attn_ve_gate: ggml::Tensor,      // limite
    attn_xsa_alpha: ggml::Tensor,
    attn_resid_scale: ggml::Tensor,
    ffn_resid_scale: ggml::Tensor,
    mudd_down: ggml::Tensor,
    mudd_up: ggml::Tensor,
    mudd_up_b: ggml::Tensor,
}

struct Model {
    hp: HParams,
    meta: gguf::Ctx,
    ctx: ggml::Ctx,          // the weights' tensors
    bufs: ggml::Buffer[8],   // and the buffers allocated for their data, ending with p == 0
    host_weight_bytes: i64 = 0,   // the weights' host memory, as --fit counts it
    lazy_ple: bool = false,       // per_layer_tok_embd read as it is used (--lazy-mode)
    backend: ggml::Backend,
    tok_embd: ggml::Tensor,
    pos_embd: ggml::Tensor,  // gpt2, starcoder: learned positions
    tok_norm: ggml::Tensor,  // bloom: a LayerNorm of the embeddings
    tok_norm_b: ggml::Tensor,
    output_norm: ggml::Tensor,
    output_norm_enc: ggml::Tensor,       // t5's encoder
    altup_proj: ggml::Tensor,            // gemma3n
    altup_unembd_proj: ggml::Tensor,
    per_layer_tok_embd: ggml::Tensor,
    per_layer_model_proj: ggml::Tensor,
    per_layer_proj_norm: ggml::Tensor,
    output_norm_b: ggml::Tensor,
    output: ggml::Tensor,
    output_b: ggml::Tensor,
    value_embd: ggml::Tensor,    // limite
    hrm_z_l_init: ggml::Tensor,  // hrm_text: the low stack's first state
    layers: Layer[..512],
}

//
// metadata
//

fn arch_name_of(arch: Arch) -> const u8[:] {
    match arch { Gemma3 => "gemma3", Llama => "llama", Lfm2 => "lfm2", Lfm3 => "lfm3",
            Qwen2 => "qwen2", Qwen3 => "qwen3", Qwen35 => "qwen35", Gemma2 => "gemma2",
            Phi2 => "phi2", Phi3 => "phi3", Stablelm => "stablelm", Smollm3 => "smollm3", Granite => "granite",
            Internlm2 => "internlm2", Exaone => "exaone", Ernie4_5 => "ernie4_5", Arcee => "arcee", Olmo2 => "olmo2", Exaone4 => "exaone4", HunyuanDense => "hunyuan-dense",
            Gpt2 => "gpt2", Starcoder => "starcoder", Gptneox => "gptneox", Starcoder2 => "starcoder2", Bloom => "bloom", Openelm => "openelm", Mistral3 => "mistral3",
            GraniteMoe => "granitemoe", Smallthinker => "smallthinker",
            Gemma => "gemma", Olmo => "olmo", Qwen => "qwen", Refact => "refact", Mpt => "mpt", Jais => "jais", Nemotron => "nemotron",
            Maincoder => "maincoder", Nanbeige => "nanbeige", Bitnet => "bitnet", Chatglm => "chatglm", Mamba => "mamba", Mamba2 => "mamba2",
            FalconH1 => "falcon-h1", GraniteHybrid => "granitehybrid", Jamba => "jamba", Plamo2 => "plamo2", Rwkv6 => "rwkv6", Rwkv7 => "rwkv7", Minicpm3 => "minicpm3", Qwen2vl => "qwen2vl", Qwen3vl => "qwen3vl", NemotronH => "nemotron_h", Afmoe => "afmoe", Gemma3n => "gemma3n", Gemma4 => "gemma4", T5 => "t5", Minicpm => "minicpm", Plamo3 => "plamo3", Cohere2 => "cohere2", PanguEmbed => "pangu-embedded", Limite => "limite", HrmText => "hrm_text", Spark2_5 => "spark2_5", K2Horizon => "k2-horizon", Glm4 => "glm4", _ => "unknown" }
}

fn arch_from_name(name: const u8[:]) -> Arch {
    if name == "gemma3" { return Arch.Gemma3; }
    if name == "llama" { return Arch.Llama; }
    if name == "lfm2" { return Arch.Lfm2; }
    if name == "lfm3" { return Arch.Lfm3; }
    if name == "qwen2" { return Arch.Qwen2; }
    if name == "qwen3" { return Arch.Qwen3; }
    if name == "qwen35" { return Arch.Qwen35; }
    if name == "gemma2" { return Arch.Gemma2; }
    if name == "phi2" { return Arch.Phi2; }
    if name == "phi3" { return Arch.Phi3; }
    if name == "stablelm" { return Arch.Stablelm; }
    if name == "smollm3" { return Arch.Smollm3; }
    if name == "granite" { return Arch.Granite; }
    if name == "internlm2" { return Arch.Internlm2; }
    if name == "exaone" { return Arch.Exaone; }
    if name == "ernie4_5" { return Arch.Ernie4_5; }
    if name == "arcee" { return Arch.Arcee; }
    if name == "olmo2" { return Arch.Olmo2; }
    if name == "exaone4" { return Arch.Exaone4; }
    if name == "hunyuan-dense" { return Arch.HunyuanDense; }
    if name == "gpt2" { return Arch.Gpt2; }
    if name == "starcoder" { return Arch.Starcoder; }
    if name == "gptneox" { return Arch.Gptneox; }
    if name == "starcoder2" { return Arch.Starcoder2; }
    if name == "bloom" { return Arch.Bloom; }
    if name == "openelm" { return Arch.Openelm; }
    if name == "mistral3" { return Arch.Mistral3; }
    if name == "granitemoe" { return Arch.GraniteMoe; }
    if name == "smallthinker" { return Arch.Smallthinker; }
    if name == "gemma" { return Arch.Gemma; }
    if name == "olmo" { return Arch.Olmo; }
    if name == "qwen" { return Arch.Qwen; }
    if name == "refact" { return Arch.Refact; }
    if name == "mpt" { return Arch.Mpt; }
    if name == "jais" { return Arch.Jais; }
    if name == "nemotron" { return Arch.Nemotron; }
    if name == "maincoder" { return Arch.Maincoder; }
    if name == "nanbeige" { return Arch.Nanbeige; }
    if name == "bitnet" { return Arch.Bitnet; }
    if name == "chatglm" { return Arch.Chatglm; }
    if name == "mamba" { return Arch.Mamba; }
    if name == "mamba2" { return Arch.Mamba2; }
    if name == "falcon-h1" { return Arch.FalconH1; }
    if name == "granitehybrid" { return Arch.GraniteHybrid; }
    if name == "jamba" { return Arch.Jamba; }
    if name == "plamo2" { return Arch.Plamo2; }
    if name == "rwkv6" { return Arch.Rwkv6; }
    if name == "rwkv7" { return Arch.Rwkv7; }
    if name == "minicpm3" { return Arch.Minicpm3; }
    if name == "qwen2vl" { return Arch.Qwen2vl; }
    if name == "qwen3vl" { return Arch.Qwen3vl; }
    if name == "nemotron_h" { return Arch.NemotronH; }
    if name == "afmoe" { return Arch.Afmoe; }
    if name == "gemma3n" { return Arch.Gemma3n; }
    if name == "gemma4" { return Arch.Gemma4; }
    if name == "t5" { return Arch.T5; }
    if name == "minicpm" { return Arch.Minicpm; }
    if name == "plamo3" { return Arch.Plamo3; }
    if name == "cohere2" { return Arch.Cohere2; }
    if name == "pangu-embedded" { return Arch.PanguEmbed; }
    if name == "limite" { return Arch.Limite; }
    if name == "hrm_text" { return Arch.HrmText; }
    if name == "spark2_5" { return Arch.Spark2_5; }
    if name == "k2-horizon" { return Arch.K2Horizon; }
    if name == "glm4" { return Arch.Glm4; }
    return Arch.Unknown;
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
    for i in n { hp.n_head.push(0); hp.n_head_kv.push(0); hp.n_ff.push(0); hp.is_swa.push(false); hp.is_recr.push(false); hp.kv_source.push(i); }
    get_u32_or_arr(m, "feed_forward_length", hp.n_ff, n, false);
    get_u32(m, "expert_count", hp.n_expert, false);
    {
        var used: u32[..512] = [];
        for i in n { used.push(0); }
        get_u32_or_arr(m, "expert_used_count", used, n, false);
        for u in used { if u != used[0] { abort("a number of experts per layer: not ported yet"); } }
        hp.n_expert_used = used[0];
    }
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
            hp.rope_scaling_none = scaling == "none";
        }
    }
    var ropescale: f32 = 0.0;
    if !get_f32(m, "rope.scaling.factor", ropescale, false) { get_f32(m, "rope.scale_linear", ropescale, false); }
    hp.rope_freq_scale_train = if ropescale == 0.0 { 1.0 } else { 1.0 / ropescale };
    get_f32(m, "rope.scaling.attn_factor", hp.rope_attn_factor, false);
    get_f32(m, "rope.scaling.alpha", hp.rope_scaling_alpha, false);

    if hp.n_head[0] > 0 {
        hp.n_embd_head_k = hp.n_embd / hp.n_head[0];
        get_u32(m, "attention.key_length", hp.n_embd_head_k, false);
        hp.n_embd_head_v = hp.n_embd / hp.n_head[0];
        get_u32(m, "attention.value_length", hp.n_embd_head_v, false);
        hp.n_rot = hp.n_embd_head_k;
        get_u32(m, "rope.dimension_count", hp.n_rot, false);
        hp.n_embd_head_k_swa = hp.n_embd_head_k;
        hp.n_embd_head_v_swa = hp.n_embd_head_v;
        get_u32(m, "attention.key_length_swa", hp.n_embd_head_k_swa, false);
        get_u32(m, "attention.value_length_swa", hp.n_embd_head_v_swa, false);
        hp.n_rot_swa = hp.n_rot;
        get_u32(m, "rope.dimension_count_swa", hp.n_rot_swa, false);
    }

    match hp.arch {
        Gemma3 => load_hparams_gemma3(m),
        Llama => load_hparams_llama(m),
        Lfm2 => load_hparams_lfm2(m),
        Lfm3 => load_hparams_lfm3(m),
        Qwen2, Qwen3 => load_hparams_qwen(m),
        Qwen35 => load_hparams_qwen35(m),
        Gemma2 => load_hparams_gemma2(m),
        Phi2, Stablelm => load_hparams_layernorm(m),
        Phi3 => load_hparams_phi3(m),
        Smollm3 => load_hparams_smollm3(m),
        Granite => load_hparams_granite(m),
        Internlm2, Ernie4_5, Arcee => load_hparams_rms(m, ggml::ROPE_TYPE_NORMAL),
        Exaone => load_hparams_rms(m, ggml::ROPE_TYPE_NEOX),
        Olmo2 => load_hparams_olmo2(m),
        Exaone4 => load_hparams_exaone4(m),
        HunyuanDense => load_hparams_hunyuan_dense(m),
        Gpt2, Starcoder, Gptneox, Starcoder2, Bloom => load_hparams_classic(m),
        Openelm => load_hparams_rms(m, ggml::ROPE_TYPE_NEOX),
        Mistral3 => load_hparams_mistral3(m),
        GraniteMoe => load_hparams_granite(m),
        Smallthinker => load_hparams_smallthinker(m),
        Gemma, Qwen => load_hparams_rms(m, ggml::ROPE_TYPE_NEOX),
        Refact => load_hparams_refact(m),
        Olmo, Mpt, Jais, Nemotron => load_hparams_classic(m),
        Maincoder, Chatglm => load_hparams_rms(m, ggml::ROPE_TYPE_NORMAL),
        Bitnet => load_hparams_rms(m, ggml::ROPE_TYPE_NEOX),
        Nanbeige => load_hparams_nanbeige(m),
        Mamba, Mamba2 => load_hparams_mamba(m),
        FalconH1 => load_hparams_falcon_h1(m),
        GraniteHybrid => load_hparams_granite_hybrid(m),
        Jamba => load_hparams_jamba(m),
        Plamo2 => load_hparams_plamo2(m),
        Rwkv6, Rwkv7 => load_hparams_rwkv(m),
        Minicpm3 => load_hparams_minicpm3(m),
        Qwen2vl, Qwen3vl => load_hparams_qwenvl(m),
        NemotronH => load_hparams_nemotron_h(m),
        Afmoe => load_hparams_afmoe(m),
        Gemma3n => load_hparams_gemma3n(m),
        Gemma4 => load_hparams_gemma4(m),
        T5 => load_hparams_t5(m),
        Minicpm => load_hparams_minicpm(m),
        Plamo3 => load_hparams_plamo3(m),
        Cohere2 => load_hparams_cohere2(m),
        PanguEmbed => load_hparams_pangu_embed(m),
        Limite => load_hparams_limite(m),
        HrmText => load_hparams_hrm_text(m),
        Spark2_5 => load_hparams_spark2_5(m),
        K2Horizon => load_hparams_k2_horizon(m),
        Glm4 => load_hparams_glm4(m),
        _ => abort("unknown architecture"),
    }
    resolve_rope(hp);
}

// The rope settings llama_context derives from the model's (its cparams):
// YaRN's extrapolation, and its attention factor.
fn resolve_rope(hp: HParams&) {
    if hp.rope_scaling_none { hp.rope_freq_scale_train = 1.0; }    // never scale if the scaling type is none
    if hp.yarn_ext_factor < 0.0 { hp.yarn_ext_factor = if hp.rope_scaling_yarn { 1.0 } else { 0.0 }; }
    if hp.yarn_ext_factor != 0.0 {
        fn get_mscale(scale: f32, mscale: f32) -> f32 { if scale <= 1.0 { 1.0 } else { 0.1 * mscale * log(scale) + 1.0 } }
        let factor: f32 = 1.0 / hp.rope_freq_scale_train;
        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if hp.rope_yarn_log_mul != 0.0 {
            hp.yarn_attn_factor = get_mscale(factor, 1.0) / get_mscale(factor, hp.rope_yarn_log_mul);
        } else {
            hp.yarn_attn_factor = get_mscale(factor, 1.0);
        }
        // ggml's rope scales by this factor with YaRN on; cancel it
        hp.yarn_attn_factor *= 1.0 / (1.0 + 0.1 * log(factor));
    }
    hp.yarn_attn_factor *= hp.rope_attn_factor;
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
        for i in min(n, m.hp.n_layer) { m.hp.is_swa[i] = vals[i] != 0; }
        return;
    }
    var n_pattern = n_pattern_default;
    get_u32(m, "attention.sliding_window_pattern", n_pattern, false);
    set_swa_pattern(m.hp, n_pattern, dense_first);
}

// The architectures whose only key of their own is the RMS norm's epsilon:
// models/internlm2.cpp, exaone.cpp, ernie4-5.cpp (dense), arcee.cpp.
fn load_hparams_rms(m: Model&, rope_type: i32) {
    get_f32(m, "attention.layer_norm_rms_epsilon", m.hp.f_norm_rms_eps);
    if m.hp.arch == Arch.Ernie4_5 && gguf::find_key(m.meta, akey(m, "rope.dimension_sections")) >= 0 {
        abort("ernie4_5 with rope sections (VL): not ported yet");
    }
    m.hp.rope_type = rope_type;
}

// models/olmo2.cpp
fn load_hparams_olmo2(m: Model&) {
    let hp .= m.hp;
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    if get_u32(m, "attention.sliding_window", hp.n_swa, false) && hp.n_swa > 0 {
        hp.swa_type = Swa.Standard;
        load_swa_pattern(m, 4);
        hp.rope_freq_base_train_swa = hp.rope_freq_base_train;
        hp.rope_freq_scale_train_swa = 1.0;    // see build_olmo2
        get_f32(m, "rope.freq_base_swa", hp.rope_freq_base_train_swa, false);
    } else {
        hp.swa_type = Swa.None;
    }
    hp.rope_type = ggml::ROPE_TYPE_NEOX;
}

// models/exaone4.cpp: the 32B mixes sliding window layers (with rope) and
// full ones (without).
fn load_hparams_exaone4(m: Model&) {
    let hp .= m.hp;
    if hp.n_layer == 64 {
        hp.swa_type = Swa.Standard;
        hp.n_swa = 4096;
        load_swa_pattern(m, 4);
        hp.rope_freq_base_train_swa = hp.rope_freq_base_train;
        hp.rope_freq_scale_train_swa = hp.rope_freq_scale_train;
        get_f32(m, "rope.freq_base_swa", hp.rope_freq_base_train_swa, false);
    }
    get_u32(m, "attention.sliding_window", hp.n_swa, false);
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    var n_nextn: u32 = 0;
    if get_u32(m, "nextn_predict_layers", n_nextn, false) && n_nextn > 0 { abort("exaone4 MTP layers: not ported yet"); }
    hp.rope_type = ggml::ROPE_TYPE_NEOX;
}

// models/hunyuan-vl.cpp, which hunyuan-dense reuses
fn load_hparams_hunyuan_dense(m: Model&) {
    let hp .= m.hp;
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    if gguf::find_key(m.meta, akey(m, "rope.dimension_sections")) >= 0 { abort("hunyuan with rope sections (M-RoPE): not ported yet"); }
    if hp.rope_scaling_alpha > 0.0 {
        let dim: f32 = hp.n_embd_head_k;
        hp.rope_freq_base_train = hp.rope_freq_base_train * pow(hp.rope_scaling_alpha, dim / (dim - 2.0));
    }
    hp.rope_type = ggml::ROPE_TYPE_NEOX;
}

// The state space model's sizes (mamba, mamba2 and the hybrids).
fn load_ssm(m: Model&, groups: bool) {
    let hp .= m.hp;
    get_u32(m, "ssm.conv_kernel", hp.ssm_d_conv);
    get_u32(m, "ssm.inner_size", hp.ssm_d_inner);
    get_u32(m, "ssm.state_size", hp.ssm_d_state);
    get_u32(m, "ssm.time_step_rank", hp.ssm_dt_rank);
    if groups { get_u32(m, "ssm.group_count", hp.ssm_n_group); }
}

// models/mamba.cpp, models/mamba2.cpp: every layer recurrent, no attention.
fn load_hparams_mamba(m: Model&) {
    let hp .= m.hp;
    load_ssm(m, hp.arch == Arch.Mamba2);
    if hp.arch == Arch.Mamba { vocab::get_bool(m.meta, akey(m, "ssm.dt_b_c_rms"), hp.ssm_dt_b_c_rms); }
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    for il in hp.n_layer as i64 { hp.is_recr[il] = true; }
    hp.rope_type = -1;
}

// models/falcon-h1.cpp: attention and a mamba2 side by side in every layer.
fn load_hparams_falcon_h1(m: Model&) {
    let hp .= m.hp;
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    load_ssm(m, true);
    for il in hp.n_layer as i64 { hp.is_recr[il] = true; }
    hp.rope_type = ggml::ROPE_TYPE_NEOX;
}

// models/granite-hybrid.cpp: granite's attention layers, mamba2 where a
// layer has no KV heads.
fn load_hparams_granite_hybrid(m: Model&) {
    let hp .= m.hp;
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    get_f32(m, "logit_scale", hp.f_logit_scale, false);
    get_f32(m, "residual_scale", hp.f_residual_scale, false);
    get_f32(m, "embedding_scale", hp.f_embedding_scale, false);
    get_f32(m, "attention.scale", hp.f_attention_scale, false);
    load_ssm(m, true);
    vocab::get_bool(m.meta, akey(m, "rope.scaling.finetuned"), hp.rope_finetuned);
    for il in hp.n_layer as i64 { hp.is_recr[il] = hp.n_head_kv[il] == 0; }
    get_u32(m, "expert_shared_feed_forward_length", hp.n_ff_shexp, false);
    hp.rope_type = ggml::ROPE_TYPE_NORMAL;
}

// models/jamba.cpp: mamba layers where a layer has no KV heads; attention
// without positions.
fn load_hparams_jamba(m: Model&) {
    let hp .= m.hp;
    load_ssm(m, false);
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    for il in hp.n_layer as i64 { hp.is_recr[il] = hp.n_head_kv[il] == 0; }
    hp.rope_type = -1;
}

// models/plamo2.cpp: its own mamba layers where a layer has no KV heads.
fn load_hparams_plamo2(m: Model&) {
    let hp .= m.hp;
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    load_ssm(m, true);
    for il in hp.n_layer as i64 { hp.is_recr[il] = hp.n_head_kv[il] == 0; }
    hp.rope_type = ggml::ROPE_TYPE_NEOX;
}

// models/rwkv6.cpp, models/rwkv7.cpp: every layer recurrent.
fn load_hparams_rwkv(m: Model&) {
    let hp .= m.hp;
    get_f32(m, "attention.layer_norm_epsilon", hp.f_norm_eps, false);
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps, false);
    get_u32(m, "wkv.head_size", hp.wkv_head_size);
    if hp.arch == Arch.Rwkv6 { get_u32(m, "rescale_every_n_layers", hp.rescale_every_n_layers, false); }
    get_u32(m, "token_shift_count", hp.token_shift_count, false);
    guard hp.token_shift_count == 2 else { abort("GGML_ASSERT(hparams.token_shift_count == 2) failed"); }
    for il in hp.n_layer as i64 { hp.is_recr[il] = true; }
    hp.rope_type = -1;
}

// models/minicpm3.cpp: queries and keys through low rank projections (MLA).
fn load_hparams_minicpm3(m: Model&) {
    get_f32(m, "attention.layer_norm_rms_epsilon", m.hp.f_norm_rms_eps);
    get_u32(m, "attention.q_lora_rank", m.hp.n_lora_q);
    get_u32(m, "attention.kv_lora_rank", m.hp.n_lora_kv);
    m.hp.rope_type = ggml::ROPE_TYPE_NEOX;
}

// models/nemotron-h.cpp: mamba2 layers, attention layers (without rope) and
// feed-forward layers, each on its own.
fn load_hparams_nemotron_h(m: Model&) {
    let hp .= m.hp;
    load_ssm(m, true);
    for il in hp.n_layer as i64 { hp.is_recr[il] = hp.n_head_kv[il] == 0 && hp.n_ff[il] == 0; }
    get_f32(m, "attention.layer_norm_epsilon", hp.f_norm_eps);
    if !get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps, false) { hp.f_norm_rms_eps = hp.f_norm_eps; }
    var n_nextn: u32 = 0;
    if get_u32(m, "nextn_predict_layers", n_nextn, false) && n_nextn > 0 { abort("nemotron_h MTP layers: not ported yet"); }
    if hp.n_expert > 0 { abort("nemotron_h with experts (MoE): not ported yet"); }
    hp.rope_type = -1;
}

// models/afmoe.cpp: gated attention, sliding windows, a mixture of experts
// after the first dense layers.
fn load_hparams_afmoe(m: Model&) {
    let hp .= m.hp;
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    get_u32(m, "leading_dense_block_count", hp.n_layer_dense_lead, false);
    get_u32(m, "expert_shared_count", hp.n_expert_shared);
    var gating: u32 = 0;
    get_u32(m, "expert_gating_func", gating, false);
    hp.expert_gating_func = gating_of(gating);
    get_f32(m, "expert_weights_scale", hp.expert_weights_scale, false);
    vocab::get_bool(m.meta, akey(m, "expert_weights_norm"), hp.expert_weights_norm);
    var n_groups: u32 = 0;
    if get_u32(m, "expert_group_count", n_groups, false) && n_groups > 1 { abort("expert groups: not ported yet"); }
    get_u32(m, "attention.sliding_window", hp.n_swa, false);
    if hp.n_swa > 0 {
        hp.swa_type = Swa.Standard;
        load_swa_pattern(m, 4);
        hp.rope_freq_base_train_swa = hp.rope_freq_base_train;
        hp.rope_freq_scale_train_swa = hp.rope_freq_scale_train;
        get_f32(m, "rope.freq_base_swa", hp.rope_freq_base_train_swa, false);
    } else {
        hp.swa_type = Swa.None;
    }
    if hp.expert_gating_func == Gating.None { hp.expert_gating_func = Gating.Sigmoid; }
    hp.n_no_rope_layer_step = 4;    // llama_hparams' default
    hp.rope_type = ggml::ROPE_TYPE_NEOX;
}

// models/gemma3n.cpp: the layers past the 20th read the cache of the last
// layer before them of the same kind (sliding window or not).
fn load_hparams_gemma3n(m: Model&) {
    let hp .= m.hp;
    hp.swa_type = Swa.Standard;
    load_swa_pattern(m, 5);
    hp.n_layer_kv_from_start = 20;
    hp.f_attention_scale = 1.0;
    for il in hp.n_layer_kv_from_start..hp.n_layer as i64 {
        hp.kv_source[il] = hp.n_layer_kv_from_start - (if hp.is_swa[il] { 2 } else { 1 });
    }
    get_f32(m, "rope.freq_base_swa", hp.rope_freq_base_train_swa, false);
    get_u32(m, "attention.sliding_window", hp.n_swa);
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    hp.f_final_logit_softcapping = 30.0;     // llama_hparams' default
    hp.rope_type = ggml::ROPE_TYPE_NEOX;
}

// models/gemma4.cpp: its last shared_kv_layers read the cache of the last
// layer before them of the same kind; full and sliding window layers have
// their own head sizes.
fn load_hparams_gemma4(m: Model&) {
    let hp .= m.hp;
    hp.swa_type = Swa.Standard;
    {
        let k = akey(m, "attention.sliding_window_pattern");
        let id = gguf::find_key(m.meta, k);
        guard id >= 0 && gguf::get_kv_type(m.meta, id) == gguf::ARRAY else { abort(str("key not found in model: ", k)); }
        load_swa_pattern(m, 0);
    }
    var n_kv_shared: u32 = 0;
    get_u32(m, "attention.shared_kv_layers", n_kv_shared, false);
    hp.n_layer_kv_from_start = hp.n_layer as i64 - n_kv_shared;
    hp.f_attention_scale = 1.0;      // no scaling before attention
    guard hp.n_layer_kv_from_start >= 2 else { abort("GGML_ASSERT(hparams.n_layer_kv_from_start >= 2) failed"); }
    for il in hp.n_layer_kv_from_start..hp.n_layer as i64 {
        hp.kv_source[il] = hp.n_layer_kv_from_start - (if hp.is_swa[il] { 2 } else { 1 });
    }
    get_f32(m, "rope.freq_base_swa", hp.rope_freq_base_train_swa, false);
    get_u32(m, "attention.sliding_window", hp.n_swa);
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    get_u32(m, "embedding_length_per_layer_input", hp.n_embd_per_layer);
    get_u32(m, "attention.key_length_swa", hp.n_embd_head_k_swa);
    get_u32(m, "attention.value_length_swa", hp.n_embd_head_v_swa);
    hp.f_final_logit_softcapping = 30.0;     // llama_hparams' default
    get_f32(m, "final_logit_softcapping", hp.f_final_logit_softcapping, false);
    if hp.n_embd_head_k != hp.n_embd_head_v { abort("Gemma 4 requires n_embd_head_k == n_embd_head_v"); }
    if hp.n_embd_head_k_swa != hp.n_embd_head_v_swa { abort("Gemma 4 requires n_embd_head_k_swa == n_embd_head_v_swa"); }
    hp.rope_type = ggml::ROPE_TYPE_NEOX;
}

// models/t5.cpp: an encoder and a decoder; relative position buckets
// instead of rope.
fn load_hparams_t5(m: Model&) {
    let hp .= m.hp;
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    get_u32(m, "attention.relative_buckets_count", hp.n_rel_attn_bkts);
    var dec_start: u32 = 0;
    if get_u32(m, "decoder_start_token_id", dec_start, false) { hp.dec_start_token_id = dec_start as i32; }
    hp.dec_n_layer = hp.n_layer;
    get_u32(m, "decoder_block_count", hp.dec_n_layer, false);
    guard hp.dec_n_layer == hp.n_layer else { abort("t5 with a decoder of its own depth: not ported yet"); }
    hp.rope_type = -1;
}

// Whether the model encodes its input first (llama_model_has_encoder).
fn has_encoder(m: Model&) -> bool { m.hp.arch == Arch.T5 }

// The decoder's first token, or -1 (llama_model_decoder_start_token).
fn decoder_start_token(m: Model&) -> i32 { m.hp.dec_start_token_id }

// Whether layer il keeps keys and values: an attention layer, or any layer
// of falcon-h1 (llama_memory_hybrid's attention filter).
fn has_kv(hp: HParams&, il: i64) -> bool {
    if hp.kv_source[il] != il { return false; }     // gemma3n: it reads another layer's
    if hp.arch == Arch.NemotronH { return !hp.is_recr[il] && hp.n_ff[il] == 0; }     // not its feed-forward layers
    return !hp.is_recr[il] || hp.arch == Arch.FalconH1;
}

// models/nanbeige.cpp: the stored layers may run several times (loops),
// each run with its own cells in the cache.
fn load_hparams_nanbeige(m: Model&) {
    let hp .= m.hp;
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    get_u32(m, "num_loops", hp.n_loops, false);
    guard hp.n_loops >= 1 else { abort("nanbeige: num_loops must be >= 1"); }
    vocab::get_bool(m.meta, akey(m, "skip_loop_final_norm"), hp.skip_loop_final_norm);
    hp.n_layer_phys = hp.n_layer;
    guard hp.n_layer_phys * hp.n_loops <= MAX_LAYERS else { abort("nanbeige: too many layers with its loops"); }
    for j in 1..hp.n_loops as i64 {
        for i in hp.n_layer_phys as i64 {
            hp.n_head.push(hp.n_head[i]);
            hp.n_head_kv.push(hp.n_head_kv[i]);
            hp.n_ff.push(hp.n_ff[i]);
            hp.is_swa.push(hp.is_swa[i]);
            hp.is_recr.push(hp.is_recr[i]);
            hp.kv_source.push(hp.n_layer_phys * j + i);
        }
    }
    hp.n_layer = hp.n_layer_phys * hp.n_loops;
    hp.rope_type = ggml::ROPE_TYPE_NORMAL;
}

// models/refact.cpp: ALiBi instead of rope.
fn load_hparams_refact(m: Model&) {
    get_f32(m, "attention.layer_norm_rms_epsilon", m.hp.f_norm_rms_eps);
    m.hp.f_max_alibi_bias = 8.0;
    m.hp.rope_type = -1;
}

// models/gpt2.cpp, starcoder.cpp, gptneox.cpp, starcoder2.cpp, bloom.cpp,
// olmo.cpp, mpt.cpp, jais.cpp, nemotron.cpp: LayerNorm; some without rope
// (gpt2, starcoder, bloom, mpt, jais: learned positions or ALiBi).
fn load_hparams_classic(m: Model&) {
    let hp .= m.hp;
    get_f32(m, "attention.layer_norm_epsilon", hp.f_norm_eps);
    if hp.arch == Arch.Olmo || hp.arch == Arch.Mpt { get_f32(m, "attention.clamp_kqv", hp.f_clamp_kqv, false); }
    if hp.arch == Arch.Mpt || hp.arch == Arch.Jais { get_f32(m, "attention.max_alibi_bias", hp.f_max_alibi_bias, false); }
    if hp.arch == Arch.Gptneox {
        guard vocab::get_bool(m.meta, akey(m, "use_parallel_residual"), hp.use_par_res) else {
            abort(str("key not found in model: ", akey(m, "use_parallel_residual")));
        }
    }
    if hp.arch == Arch.Bloom { hp.f_max_alibi_bias = 8.0; }
    hp.rope_type = if hp.arch == Arch.Gptneox || hp.arch == Arch.Starcoder2 || hp.arch == Arch.Nemotron { ggml::ROPE_TYPE_NEOX }
                   else if hp.arch == Arch.Olmo { ggml::ROPE_TYPE_NORMAL } else { -1 };
}

// models/mistral3.cpp
fn load_hparams_mistral3(m: Model&) {
    let hp .= m.hp;
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    get_f32(m, "attention.temperature_scale", hp.f_attn_temp_scale, false);
    get_f32(m, "rope.scaling.yarn_beta_fast", hp.yarn_beta_fast, false);
    get_f32(m, "rope.scaling.yarn_beta_slow", hp.yarn_beta_slow, false);
    get_f32(m, "rope.scaling.yarn_log_multiplier", hp.rope_yarn_log_mul, false);
    var n_delay: u32 = 0;
    if get_u32(m, "realtime_delay_tokens", n_delay, false) { abort("mistral3 realtime (Voxtral): not ported yet"); }
    if hp.f_attn_temp_scale != 0.0 {
        hp.n_attn_temp_floor_scale = hp.n_ctx_orig_yarn;
        guard hp.n_attn_temp_floor_scale != 0 else { abort("invalid n_ctx_orig_yarn for attention temperature scaling"); }
    }
    var n_expert: u32 = 0;
    if get_u32(m, "expert_count", n_expert, false) && n_expert > 0 { abort("mistral3 with experts (MoE): not ported yet"); }
    hp.rope_type = ggml::ROPE_TYPE_NORMAL;
}

// models/smallthinker.cpp: the router reads the layer's input, before its norm.
fn load_hparams_smallthinker(m: Model&) {
    let hp .= m.hp;
    if get_u32(m, "attention.sliding_window", hp.n_swa, false) && hp.n_swa > 0 {
        hp.swa_type = Swa.Standard;
        hp.n_swa = 4096;
        load_swa_pattern(m, 4, true);
        hp.rope_freq_base_train_swa = hp.rope_freq_base_train;
        hp.rope_freq_scale_train_swa = hp.rope_freq_scale_train;
        get_f32(m, "rope.freq_base_swa", hp.rope_freq_base_train_swa, false);
        hp.n_no_rope_layer_step = 4;
    } else {
        hp.swa_type = Swa.None;
        hp.n_no_rope_layer_step = hp.n_layer;
    }
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    var gating: u32 = 0;
    get_u32(m, "expert_gating_func", gating, false);
    hp.expert_gating_func = gating_of(gating);
    guard hp.n_expert > 0 && hp.n_expert_used > 0 else { abort("n_expert and n_expert_used must be > 0 for SMALLTHINKER"); }
    hp.rope_type = ggml::ROPE_TYPE_NEOX;
}

// models/phi2.cpp, models/stablelm.cpp
fn load_hparams_layernorm(m: Model&) {
    get_f32(m, "attention.layer_norm_epsilon", m.hp.f_norm_eps);
    m.hp.rope_type = ggml::ROPE_TYPE_NEOX;
}

// models/phi3.cpp: llama.cpp turns its sliding window off.
fn load_hparams_phi3(m: Model&) {
    get_f32(m, "attention.layer_norm_rms_epsilon", m.hp.f_norm_rms_eps);
    m.hp.n_swa = 0;
    m.hp.swa_type = Swa.None;
    m.hp.rope_type = ggml::ROPE_TYPE_NEOX;
}

// models/smollm3.cpp
fn load_hparams_smollm3(m: Model&) {
    get_f32(m, "attention.layer_norm_rms_epsilon", m.hp.f_norm_rms_eps);
    m.hp.n_no_rope_layer_step = 4;
    m.hp.rope_type = ggml::ROPE_TYPE_NORMAL;
}

// models/granite.cpp
fn load_hparams_granite(m: Model&) {
    let hp .= m.hp;
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    get_f32(m, "logit_scale", hp.f_logit_scale);
    get_f32(m, "residual_scale", hp.f_residual_scale, false);
    get_f32(m, "embedding_scale", hp.f_embedding_scale, false);
    get_f32(m, "attention.scale", hp.f_attention_scale, false);
    if gguf::find_key(m.meta, akey(m, "deepstack_mapping")) >= 0 { abort("granite vision deepstack: not ported yet"); }
    vocab::get_bool(m.meta, akey(m, "rope.scaling.finetuned"), hp.rope_finetuned);
    get_u32(m, "expert_shared_feed_forward_length", hp.n_ff_shexp, false);    // granitemoe's shared expert
    hp.rope_type = ggml::ROPE_TYPE_NORMAL;
}

// models/minicpm.cpp: granite's graph, with defaults for the older GGUFs
fn load_hparams_minicpm(m: Model&) {
    let hp .= m.hp;
    hp.f_embedding_scale = 12.0;
    hp.f_residual_scale = 1.4 / sqrt(hp.n_layer as f32);
    hp.f_logit_scale = if hp.n_embd != 0 { 256.0 / hp.n_embd } else { 1.0 };
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    get_f32(m, "embedding_scale", hp.f_embedding_scale, false);
    get_f32(m, "residual_scale", hp.f_residual_scale, false);
    get_f32(m, "logit_scale", hp.f_logit_scale, false);
    hp.rope_finetuned = true;
    hp.rope_type = ggml::ROPE_TYPE_NORMAL;
}

// models/plamo3.cpp
fn load_hparams_plamo3(m: Model&) {
    let hp .= m.hp;
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    if get_u32(m, "attention.sliding_window", hp.n_swa, false) && hp.n_swa > 0 {
        hp.swa_type = Swa.Standard;
        get_f32(m, "rope.freq_base_swa", hp.rope_freq_base_train_swa, false);
        load_swa_pattern(m, 8);
    } else {
        hp.swa_type = Swa.None;
    }
    hp.rope_type = ggml::ROPE_TYPE_NEOX;
}

// models/cohere2.cpp: sliding windows; only their layers have rope
fn load_hparams_cohere2(m: Model&) {
    let hp .= m.hp;
    hp.swa_type = Swa.Standard;
    load_swa_pattern(m, 4);
    hp.rope_freq_base_train_swa = hp.rope_freq_base_train;
    hp.rope_freq_scale_train_swa = hp.rope_freq_scale_train;
    get_f32(m, "rope.freq_base_swa", hp.rope_freq_base_train_swa, false);
    get_u32(m, "attention.sliding_window", hp.n_swa);
    get_f32(m, "logit_scale", hp.f_logit_scale);
    get_f32(m, "attention.layer_norm_epsilon", hp.f_norm_eps);
    hp.rope_type = ggml::ROPE_TYPE_NORMAL;
}

// models/pangu-embed.cpp
fn load_hparams_pangu_embed(m: Model&) {
    get_f32(m, "attention.layer_norm_rms_epsilon", m.hp.f_norm_rms_eps);
    m.hp.rope_type = ggml::ROPE_TYPE_NEOX;
}

// models/limite.cpp: value embeddings, exclusive self attention, gates per
// head, learned residual scales, dense (MUDD) connections across layers.
fn load_hparams_limite(m: Model&) {
    let hp .= m.hp;
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    get_f32(m, "attention.scale", hp.f_attention_scale);
    get_f32(m, "attention.xsa_epsilon", hp.f_xsa_eps, false);
    let kc = akey(m, "final_logit_sigmoid_capping");
    let idc = gguf::find_key(m.meta, kc);
    if idc >= 0 {
        let n = gguf::get_arr_n(m.meta, idc);
        guard n <= 3 else { abort(str("array length ", n, " for key ", kc, " > 3")); }
        var vals: f32[>..] = [];
        vals.resize(n, 0.0);
        gguf::get_arr_f32(m.meta, idc, vals);
        for i in n { hp.final_logit_sigmoid_capping[i] = vals[i]; }
    }
    hp.swa_type = Swa.Standard;
    get_u32(m, "attention.sliding_window", hp.n_swa);
    load_swa_pattern(m, 4);
    get_u32(m, "mudd.tap_count", hp.n_mudd_taps, false);
    if hp.n_mudd_taps > 0 {
        get_u32(m, "mudd.feed_forward_length", hp.n_mudd_ff);
        let k = akey(m, "mudd.tap_indices");
        let id = gguf::find_key(m.meta, k);
        guard id >= 0 else { abort(str("key not found in model: ", k)); }
        let n = gguf::get_arr_n(m.meta, id);
        guard n <= 512 else { abort(str("array length ", n, " for key ", k, " > 512")); }
        var vals: i32[>..] = [];
        vals.resize(n, 0);
        gguf::get_arr_i32(m.meta, id, vals);
        for i in 512 { hp.mudd_tap_idx.push(if i < n { vals[i] } else { -1 }); }
    }
    hp.rope_type = ggml::ROPE_TYPE_NORMAL;
}

// models/hrm-text.cpp: a low and a high stack of layers, alternated over
// the same tokens. Each pass has its own cache slots: block_count of them.
fn load_hparams_hrm_text(m: Model&) {
    let hp .= m.hp;
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    get_f32(m, "embedding_scale", hp.f_embedding_scale, false);
    get_u32(m, "hrm.layers_per_stack", hp.n_hrm_layers_per_stack);
    get_u32(m, "hrm.h_cycles", hp.n_hrm_h_cycles);
    get_u32(m, "hrm.l_cycles", hp.n_hrm_l_cycles);
    assert(hp.n_hrm_layers_per_stack > 0 && hp.n_hrm_h_cycles > 0 && hp.n_hrm_l_cycles > 0);
    assert(hp.n_layer == hp.n_hrm_layers_per_stack * hp.n_hrm_h_cycles * (hp.n_hrm_l_cycles + 1));
    hp.rope_type = ggml::ROPE_TYPE_NEOX;
}

// models/spark2-5.cpp: sliding windows with their own rope, a gate per head
fn load_hparams_spark2_5(m: Model&) {
    let hp .= m.hp;
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    get_u32(m, "attention.sliding_window", hp.n_swa);
    hp.swa_type = Swa.Standard;
    let k = akey(m, "attention.sliding_window_pattern");
    let id = gguf::find_key(m.meta, k);
    guard id >= 0 && gguf::get_kv_type(m.meta, id) == gguf::ARRAY else { abort(str("key not found in model: ", k)); }
    load_swa_pattern(m, 0);
    hp.rope_freq_base_train_swa = hp.rope_freq_base_train;
    hp.rope_freq_scale_train_swa = hp.rope_freq_scale_train;
    get_f32(m, "rope.freq_base_swa", hp.rope_freq_base_train_swa, false);
    hp.rope_type = ggml::ROPE_TYPE_NEOX;
}

// models/k2-horizon.cpp: the dense models without Q/K norms or an output
// gate (the others, and the MoE and MoVA ones, are not ported yet).
fn load_hparams_k2_horizon(m: Model&) {
    let hp .= m.hp;
    get_f32(m, "rope.scaling.yarn_beta_fast", hp.yarn_beta_fast, false);
    get_f32(m, "rope.scaling.yarn_beta_slow", hp.yarn_beta_slow, false);
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    get_u32(m, "attention.group_norm_groups", hp.n_norm_groups, false);
    if hp.n_norm_groups == 0 { hp.n_norm_groups = 1; }
    if hp.n_expert > 0 { abort("k2-horizon with experts: not ported yet"); }
    var n_value_expert: u32 = 0;
    get_u32(m, "attention.value_expert_count", n_value_expert, false);
    if n_value_expert > 0 { abort("k2-horizon with value experts (MoVA): not ported yet"); }
    hp.rope_type = ggml::ROPE_TYPE_NEOX;
}

// models/glm4.cpp: M-RoPE when the file has rope sections (GLM-OCR). The
// NextN (MTP) layers at the end are not run, nor loaded.
fn load_hparams_glm4(m: Model&) {
    let hp .= m.hp;
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    if gguf::find_key(m.meta, akey(m, "rope.dimension_sections")) >= 0 { load_rope_sections(m); }
    var n_nextn: u32 = 0;
    get_u32(m, "nextn_predict_layers", n_nextn, false);
    assert(n_nextn <= hp.n_layer);
    hp.n_layer -= n_nextn;
    let use_mrope = hp.rope_sections[0] > 0 && hp.rope_sections[1] > 0;
    hp.rope_type = if use_mrope { ggml::ROPE_TYPE_MROPE } else { ggml::ROPE_TYPE_NORMAL };
}

// models/gemma2.cpp
fn load_hparams_gemma2(m: Model&) {
    let hp .= m.hp;
    hp.swa_type = Swa.Standard;
    hp.n_swa = 4096;      // gemma 2's default
    load_swa_pattern(m, 2);
    hp.attn_soft_cap = true;
    hp.rope_freq_base_train_swa = hp.rope_freq_base_train;
    hp.rope_freq_scale_train_swa = hp.rope_freq_scale_train;
    get_f32(m, "rope.freq_base_swa", hp.rope_freq_base_train_swa, false);
    get_u32(m, "attention.sliding_window", hp.n_swa, false);
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    get_f32(m, "attn_logit_softcapping", hp.f_attn_logit_softcapping, false);
    hp.f_final_logit_softcapping = 30.0;
    get_f32(m, "final_logit_softcapping", hp.f_final_logit_softcapping, false);
    // ref: https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/config.py#L173
    let is_27b = hp.n_layer == 46;
    hp.f_attention_scale = if is_27b { 1.0 / sqrt((hp.n_embd / hp.n_head[0]) as f32) }
                           else { 1.0 / sqrt(hp.n_embd_head_k as f32) };
    hp.rope_type = ggml::ROPE_TYPE_NEOX;
}

fn load_hparams_gemma3(m: Model&) {
    let found_swa = get_u32(m, "attention.sliding_window", m.hp.n_swa, false);
    if found_swa && m.hp.n_swa > 0 {
        m.hp.swa_type = Swa.Standard;
        load_swa_pattern(m, 6);
        get_f32(m, "rope.freq_base_swa", m.hp.rope_freq_base_train_swa, false);
    } else {
        m.hp.swa_type = Swa.None;
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

// models/llama.cpp
fn load_hparams_llama(m: Model&) {
    get_f32(m, "attention.layer_norm_rms_epsilon", m.hp.f_norm_rms_eps);
    m.hp.rope_type = ggml::ROPE_TYPE_NORMAL;
}

// models/qwen2.cpp, models/qwen3.cpp
fn load_hparams_qwen(m: Model&) {
    get_f32(m, "attention.layer_norm_rms_epsilon", m.hp.f_norm_rms_eps);
    m.hp.rope_type = ggml::ROPE_TYPE_NEOX;
}

// The four M-RoPE sections (get_key_or_arr of rope.dimension_sections).
fn load_rope_sections(m: Model&) {
    let k = akey(m, "rope.dimension_sections");
    let id = gguf::find_key(m.meta, k);
    guard id >= 0 else { abort(str("key not found in model: ", k)); }
    let n = gguf::get_arr_n(m.meta, id);
    guard n >= 4 else { abort(str("array length ", n, " for key ", k, " < 4")); }
    var vals: i32[>..] = [];
    vals.resize(n, 0);
    gguf::get_arr_i32(m.meta, id, vals);
    for i in 4 { m.hp.rope_sections[i] = vals[i]; }
}

// models/qwen2vl.cpp, models/qwen3vl.cpp: the text model, with M-RoPE.
// qwen2vl does not read the norm's epsilon (it stays 0, as in llama.cpp).
fn load_hparams_qwenvl(m: Model&) {
    let hp .= m.hp;
    if hp.arch == Arch.Qwen3vl {
        get_u32(m, "n_deepstack_layers", hp.n_deepstack_layers, false);
        get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    }
    load_rope_sections(m);
    hp.rope_type = if hp.arch == Arch.Qwen2vl { ggml::ROPE_TYPE_MROPE } else { ggml::ROPE_TYPE_IMROPE };
}

// models/qwen35.cpp: every full_attention_interval-th layer attends, the
// others are gated delta nets.
fn load_hparams_qwen35(m: Model&) {
    let hp .= m.hp;
    get_f32(m, "attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
    load_rope_sections(m);
    get_u32(m, "ssm.conv_kernel", hp.ssm_d_conv);
    get_u32(m, "ssm.inner_size", hp.ssm_d_inner);
    get_u32(m, "ssm.state_size", hp.ssm_d_state);
    get_u32(m, "ssm.time_step_rank", hp.ssm_dt_rank);
    get_u32(m, "ssm.group_count", hp.ssm_n_group);
    var n_nextn: u32 = 0;
    if get_u32(m, "nextn_predict_layers", n_nextn, false) && n_nextn > 0 { abort("qwen35 MTP layers: not ported yet"); }
    if gguf::find_key(m.meta, akey(m, "attention.recurrent_layers")) >= 0 { abort("qwen35 attention.recurrent_layers: not ported yet"); }
    var interval: u32 = 4;
    get_u32(m, "full_attention_interval", interval, false);
    for il in hp.n_layer as i64 { hp.is_recr[il] = (il + 1) % interval != 0; }
    hp.rope_type = ggml::ROPE_TYPE_IMROPE;
}

// models/lfm2.cpp: a layer without KV heads is a short convolution.
fn load_hparams_lfm2(m: Model&) {
    get_u32(m, "shortconv.l_cache", m.hp.n_shortconv_l_cache);
    get_f32(m, "attention.layer_norm_rms_epsilon", m.hp.f_norm_rms_eps);
    for il in m.hp.n_layer as i64 { m.hp.is_recr[il] = m.hp.n_head_kv[il] == 0; }
    var n_decision: u32 = 0;
    if get_u32(m, "decision.block_count", n_decision, false) && n_decision > 0 { abort("lfm2 decision heads: not ported yet"); }
    if get_u32(m, "attention.sliding_window", m.hp.n_swa, false) && m.hp.n_swa > 0 {
        m.hp.swa_type = Swa.Standard;
        for il in m.hp.n_layer as i64 { m.hp.is_swa[il] = !m.hp.is_recr[il]; }
    }
    m.hp.rope_type = ggml::ROPE_TYPE_NEOX;
}

// models/lfm3.cpp: lfm2's short convolutions, and attention that mixes
// full and sliding window layers.
fn load_hparams_lfm3(m: Model&) {
    get_u32(m, "shortconv.l_cache", m.hp.n_shortconv_l_cache);
    get_f32(m, "attention.layer_norm_rms_epsilon", m.hp.f_norm_rms_eps);
    for il in m.hp.n_layer as i64 { m.hp.is_recr[il] = m.hp.n_head_kv[il] == 0; }
    load_swa_pattern(m, 0);
    var any = false;
    for s in m.hp.is_swa { if s { any = true; } }
    guard any else { abort("LFM3 requires sliding window attention layers"); }
    m.hp.swa_type = Swa.Standard;
    get_u32(m, "attention.sliding_window", m.hp.n_swa);
    get_f32(m, "rope.freq_base_swa", m.hp.rope_freq_base_train_swa);
    var causal = true;
    if vocab::get_bool(m.meta, akey(m, "attention.causal"), causal) && !causal { abort("lfm3 decision models: not ported yet"); }
    m.hp.rope_type = ggml::ROPE_TYPE_NEOX;
}

// The size of a recurrent layer's convolution state (n_embd_r).
fn n_embd_r(hp: HParams&) -> i64 {
    if hp.wkv_head_size != 0 { return hp.token_shift_count as i64 * hp.n_embd; }     // rwkv
    if hp.n_shortconv_l_cache != 0 { return hp.n_embd as i64 * (hp.n_shortconv_l_cache - 1); }
    let d_conv: i64 = if hp.ssm_d_conv > 0 { hp.ssm_d_conv as i64 - 1 } else { 0 };
    return d_conv * (hp.ssm_d_inner + 2 * hp.ssm_n_group as i64 * hp.ssm_d_state);
}

// The size of a recurrent layer's other state, a gated delta net's (n_embd_s).
fn n_embd_s(hp: HParams&) -> i64 {
    if hp.wkv_head_size != 0 { return hp.n_embd as i64 * hp.wkv_head_size; }       // rwkv
    return hp.ssm_d_state as i64 * hp.ssm_d_inner;
}

//
// tensors
//

// The tensors the architecture's loader asked for, sorted once the model
// is loaded: the others of the file are left out (llama.cpp's unused tensors).
var used_tensors: i64[>..] = [];

fn tensor(m: Model&, name: const u8[:], required: bool = true) -> ggml::Tensor {
    let t = ggml::get_tensor(m.ctx, name);
    if t.p == 0 && required { abort(str("missing tensor '", name, "'")); }
    if t.p != 0 { used_tensors.push(t.p); }
    return t;
}

// Whether the model reads t (a tensor of its file).
fn is_used(t: ggml::Tensor) -> bool {
    let at = lower_bound(used_tensors) { it < t.p };
    return at < used_tensors.len && used_tensors[at] == t.p;
}

// The tensors of the file no loader asked for.
fn unused_tensors(m: Model&) -> ggml::Tensor[>..] {
    sort(used_tensors);
    var out: ggml::Tensor[>..] = [];
    var name: u8[>..] = [];
    for id in gguf::get_n_tensors(m.meta) {
        name.clear();
        gguf::get_tensor_name(m.meta, id, name);
        let t = ggml::get_tensor(m.ctx, name);
        if t.p != 0 && !is_used(t) { out.push(t); }
    }
    return out;
}

fn layer_tensor(m: Model&, il: i64, name: const u8[:], required: bool = true, suffix: const u8[:] = "weight") -> ggml::Tensor {
    return tensor(m, str("blk.", il, ".", name, ".", suffix), required);
}

fn layer_bias(m: Model&, il: i64, name: const u8[:]) -> ggml::Tensor { layer_tensor(m, il, name, false, "bias") }

// Q, K and V: one fused tensor or three, each with an optional bias
// (create_tensor_qkv).
fn load_qkv(m: Model&, il: i64, ly: Layer&) {
    ly.wqkv = layer_tensor(m, il, "attn_qkv", false);
    if ly.wqkv.p != 0 {
        ly.bqkv = layer_bias(m, il, "attn_qkv");
    } else {
        ly.wq = layer_tensor(m, il, "attn_q");
        ly.wk = layer_tensor(m, il, "attn_k");
        ly.wv = layer_tensor(m, il, "attn_v");
    }
    ly.bq = layer_bias(m, il, "attn_q");
    ly.bk = layer_bias(m, il, "attn_k");
    ly.bv = layer_bias(m, il, "attn_v");
}

fn load_tensors_llama(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    let rope_freqs = tensor(m, "rope_freqs.weight", false);   // one, shared by every layer
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        load_qkv(m, il, ly);
        ly.wo = layer_tensor(m, il, "attn_output");
        if m.hp.arch != Arch.Ernie4_5 { ly.bo = layer_bias(m, il, "attn_output"); }   // ernie4_5's graph does not add it
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        ly.rope_freqs = rope_freqs;
        if tensor(m, str("blk.", il, ".ffn_gate_inp.weight"), false).p != 0 { abort("llama with experts (MoE): not ported yet"); }
        ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        ly.ffn_gate_b = layer_bias(m, il, "ffn_gate");
        ly.ffn_down_b = layer_bias(m, il, "ffn_down");
        ly.ffn_up_b = layer_bias(m, il, "ffn_up");
        m.layers.push(ly);
    }
}

// qwen2: llama's layers with Q, K and V biases and an optional output
// bias; qwen3: no biases, Q and K normed per head.
fn load_tensors_qwen(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.hp.arch == Arch.Qwen2 || m.hp.arch == Arch.Qwen2vl { m.output_b = tensor(m, "output.bias", false); }
    if m.hp.arch == Arch.Qwen3vl && tensor(m, "cls_out.weight", false).p != 0 { abort("qwen3vl classifier: not ported yet"); }
    if m.output.p == 0 { m.output = m.tok_embd; }
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        load_qkv(m, il, ly);
        ly.wo = layer_tensor(m, il, "attn_output");
        if m.hp.arch == Arch.Qwen3 || m.hp.arch == Arch.Qwen3vl {
            ly.attn_k_norm = layer_tensor(m, il, "attn_k_norm");
            ly.attn_q_norm = layer_tensor(m, il, "attn_q_norm");
        }
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        m.layers.push(ly);
    }
}

fn load_tensors_qwen35(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    if tensor(m, "cls_out.weight", false).p != 0 { abort("qwen35 embedding projection: not ported yet"); }
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        ly.attn_post_norm = layer_tensor(m, il, "post_attention_norm");
        if !m.hp.is_recr[il] {
            load_qkv(m, il, ly);
            ly.wo = layer_tensor(m, il, "attn_output");
            ly.attn_q_norm = layer_tensor(m, il, "attn_q_norm");
            ly.attn_k_norm = layer_tensor(m, il, "attn_k_norm");
        } else {
            ly.wqkv = layer_tensor(m, il, "attn_qkv", false);
            ly.attn_gate = layer_tensor(m, il, "attn_gate", false);
            ly.ssm_conv1d = layer_tensor(m, il, "ssm_conv1d");
            ly.ssm_dt = layer_bias(m, il, "ssm_dt");
            ly.ssm_a = tensor(m, str("blk.", il, ".ssm_a"));
            ly.ssm_beta = layer_tensor(m, il, "ssm_beta");
            ly.ssm_alpha = layer_tensor(m, il, "ssm_alpha");
            ly.ssm_norm = layer_tensor(m, il, "ssm_norm");
            ly.ssm_out = layer_tensor(m, il, "ssm_out");
            if ly.ssm_dt.p == 0 { abort(str("missing tensor 'blk.", il, ".ssm_dt.bias'")); }
            if ly.wqkv.p == 0 || ly.attn_gate.p == 0 { abort("qwen35 without attn_qkv and attn_gate: not ported yet"); }
        }
        ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        m.layers.push(ly);
    }
}

// models/phi2.cpp: LayerNorm, fused QKV, a GELU feed-forward in parallel
// with the attention.
fn load_tensors_phi2(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output_norm_b = tensor(m, "output_norm.bias");
    m.output = tensor(m, "output.weight");
    m.output_b = tensor(m, "output.bias");
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        ly.attn_norm_b = layer_tensor(m, il, "attn_norm", true, "bias");
        load_qkv(m, il, ly);
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.bo = layer_tensor(m, il, "attn_output", true, "bias");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_down_b = layer_tensor(m, il, "ffn_down", true, "bias");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        ly.ffn_up_b = layer_tensor(m, il, "ffn_up", true, "bias");
        m.layers.push(ly);
    }
}

// models/phi3.cpp: fused QKV, and the gate and up of the SwiGLU in one tensor.
fn load_tensors_phi3(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    let rope_long = tensor(m, "rope_factors_long.weight", false);
    let rope_short = tensor(m, "rope_factors_short.weight", false);
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        load_qkv(m, il, ly);
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        ly.rope_long = rope_long;
        ly.rope_short = rope_short;
        if tensor(m, str("blk.", il, ".ffn_gate_inp.weight"), false).p != 0 { abort("phi3 with experts (MoE): not ported yet"); }
        m.layers.push(ly);
    }
}

// models/stablelm.cpp
fn load_tensors_stablelm(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm_b = tensor(m, "output_norm.bias");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight");
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        ly.attn_norm_b = layer_tensor(m, il, "attn_norm", true, "bias");
        load_qkv(m, il, ly);
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.attn_q_norm = layer_tensor(m, il, "attn_q_norm", false);
        ly.attn_k_norm = layer_tensor(m, il, "attn_k_norm", false);
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm", false);
        ly.ffn_norm_b = layer_bias(m, il, "ffn_norm");
        ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        m.layers.push(ly);
    }
}

// models/smollm3.cpp, models/granite.cpp: llama's layers, with optional
// biases; granite's rope factors (longrope: long and short).
fn load_tensors_smollm3_granite(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    let granite = m.hp.arch == Arch.Granite || m.hp.arch == Arch.GraniteMoe || m.hp.arch == Arch.Minicpm;
    var rope_freqs = ggml::NONE;
    var rope_long = ggml::NONE;
    var rope_short = ggml::NONE;
    if granite {
        rope_freqs = tensor(m, "rope_freqs.weight", false);
        rope_long = tensor(m, "rope_factors_long.weight", false);
        rope_short = tensor(m, "rope_factors_short.weight", false);
    }
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        load_qkv(m, il, ly);
        ly.wo = layer_tensor(m, il, "attn_output");
        if granite { ly.bo = layer_bias(m, il, "attn_output"); }
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        if m.hp.n_expert > 0 {
            // granitemoe
            ly.ffn_gate_inp = layer_tensor(m, il, "ffn_gate_inp");
            ly.ffn_gate_exps = layer_tensor(m, il, "ffn_gate_exps", false);
            ly.ffn_down_exps = layer_tensor(m, il, "ffn_down_exps");
            ly.ffn_up_exps = layer_tensor(m, il, "ffn_up_exps");
            if m.hp.n_ff_shexp > 0 {
                ly.ffn_gate_shexp = layer_tensor(m, il, "ffn_gate_shexp");
                ly.ffn_up_shexp = layer_tensor(m, il, "ffn_up_shexp");
                ly.ffn_down_shexp = layer_tensor(m, il, "ffn_down_shexp");
            }
            ly.rope_freqs = rope_freqs;
            ly.rope_long = rope_long;
            ly.rope_short = rope_short;
            m.layers.push(ly);
            continue;
        }
        ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        if granite {
            ly.ffn_gate_b = layer_bias(m, il, "ffn_gate");
            ly.ffn_down_b = layer_bias(m, il, "ffn_down");
            ly.ffn_up_b = layer_bias(m, il, "ffn_up");
            ly.rope_freqs = rope_freqs;
            ly.rope_long = rope_long;
            ly.rope_short = rope_short;
        }
        m.layers.push(ly);
    }
}

// models/arcee.cpp: llama's attention, a ReLU² feed-forward without gate.
fn load_tensors_arcee(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    let rope_freqs = tensor(m, "rope_freqs.weight", false);
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        load_qkv(m, il, ly);
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        ly.rope_freqs = rope_freqs;
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        m.layers.push(ly);
    }
}

// models/olmo2.cpp, models/exaone4.cpp: norms after the attention and the
// feed-forward, not before; Q and K normed.
fn load_tensors_post_norm(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    let olmo2 = m.hp.arch == Arch.Olmo2;
    m.output = tensor(m, "output.weight", olmo2);
    if m.output.p == 0 { m.output = m.tok_embd; }
    let rope_freqs = if olmo2 { ggml::NONE } else { tensor(m, "rope_freqs.weight", false) };
    for il in m.hp.n_layer as i64 {
        var ly = Layer { .. };
        load_qkv(m, il, ly);
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.rope_freqs = rope_freqs;
        ly.attn_q_norm = layer_tensor(m, il, "attn_q_norm");
        ly.attn_k_norm = layer_tensor(m, il, "attn_k_norm");
        ly.attn_post_norm = layer_tensor(m, il, "post_attention_norm");
        ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_post_norm = layer_tensor(m, il, "post_ffw_norm");
        m.layers.push(ly);
    }
}

// models/hunyuan-vl.cpp
fn load_tensors_hunyuan_dense(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        load_qkv(m, il, ly);
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.attn_k_norm = layer_tensor(m, il, "attn_k_norm");
        ly.attn_q_norm = layer_tensor(m, il, "attn_q_norm");
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        m.layers.push(ly);
    }
}

// models/gpt2.cpp, starcoder.cpp, gptneox.cpp, starcoder2.cpp, bloom.cpp:
// LayerNorms with biases, a GELU feed-forward without gate, biases everywhere.
fn load_tensors_classic(m: Model&) {
    let arch = m.hp.arch;
    m.tok_embd = tensor(m, "token_embd.weight");
    if arch == Arch.Gpt2 || arch == Arch.Starcoder || arch == Arch.Mpt { m.pos_embd = tensor(m, "position_embd.weight", arch != Arch.Mpt); }
    if arch == Arch.Bloom {
        m.tok_norm = tensor(m, "token_embd_norm.weight");
        m.tok_norm_b = tensor(m, "token_embd_norm.bias");
    }
    m.output_norm = tensor(m, "output_norm.weight");
    m.output_norm_b = tensor(m, "output_norm.bias", arch != Arch.Mpt);
    m.output = tensor(m, "output.weight", arch == Arch.Gptneox || arch == Arch.Jais || arch == Arch.Nemotron);
    if m.output.p == 0 { m.output = m.tok_embd; }
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        let opt = arch == Arch.Mpt || arch == Arch.Nemotron;     // their biases may be missing
        ly.attn_norm_b = layer_tensor(m, il, "attn_norm", arch != Arch.Mpt, "bias");
        if arch == Arch.Starcoder2 || arch == Arch.Nemotron {
            load_qkv(m, il, ly);
        } else {
            ly.wqkv = layer_tensor(m, il, "attn_qkv");
            ly.bqkv = layer_tensor(m, il, "attn_qkv", arch != Arch.Mpt, "bias");
        }
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.bo = layer_tensor(m, il, "attn_output", !opt, "bias");
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        ly.ffn_norm_b = layer_tensor(m, il, "ffn_norm", arch != Arch.Mpt, "bias");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_down_b = layer_tensor(m, il, "ffn_down", !opt, "bias");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        ly.ffn_up_b = layer_tensor(m, il, "ffn_up", !opt, "bias");
        if arch == Arch.Jais {
            ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
            ly.ffn_gate_b = layer_tensor(m, il, "ffn_gate", true, "bias");
        }
        if arch == Arch.Mpt {
            ly.attn_q_norm = layer_tensor(m, il, "attn_q_norm", false);
            ly.attn_q_norm_b = layer_bias(m, il, "attn_q_norm");
            ly.attn_k_norm = layer_tensor(m, il, "attn_k_norm", false);
            ly.attn_k_norm_b = layer_bias(m, il, "attn_k_norm");
            ly.ffn_act = tensor(m, str("blk.", il, ".ffn.act.scales"), false);
        }
        m.layers.push(ly);
    }
}

// models/openelm.cpp: heads and feed-forward sizes per layer.
fn load_tensors_openelm(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = m.tok_embd;
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        ly.wqkv = layer_tensor(m, il, "attn_qkv");
        ly.attn_q_norm = layer_tensor(m, il, "attn_q_norm");
        ly.attn_k_norm = layer_tensor(m, il, "attn_k_norm");
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        m.layers.push(ly);
    }
}

// models/smallthinker.cpp
fn load_tensors_smallthinker(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        load_qkv(m, il, ly);
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        ly.ffn_gate_inp = layer_tensor(m, il, "ffn_gate_inp");
        ly.ffn_gate_exps = layer_tensor(m, il, "ffn_gate_exps");
        ly.ffn_down_exps = layer_tensor(m, il, "ffn_down_exps");
        ly.ffn_up_exps = layer_tensor(m, il, "ffn_up_exps");
        m.layers.push(ly);
    }
}

// models/gemma.cpp
fn load_tensors_gemma(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = m.tok_embd;
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        load_qkv(m, il, ly);
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        m.layers.push(ly);
    }
}

// models/olmo.cpp: LayerNorms without weights.
fn load_tensors_olmo(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    for il in m.hp.n_layer as i64 {
        var ly = Layer { .. };
        load_qkv(m, il, ly);
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        m.layers.push(ly);
    }
}

// models/refact.cpp: it loads optional biases its graph does not add.
fn load_tensors_refact(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        load_qkv(m, il, ly);
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        if m.hp.n_expert > 0 { abort("refact with experts (MoE): not ported yet"); }
        ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        m.layers.push(ly);
    }
}

// models/nanbeige.cpp: llama's layers, stored once and repeated per loop.
fn load_tensors_nanbeige(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    let rope_freqs = tensor(m, "rope_freqs.weight", false);
    let n_phys = m.hp.n_layer_phys as i64;
    for il in n_phys {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        load_qkv(m, il, ly);
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.rope_freqs = rope_freqs;
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        m.layers.push(ly);
    }
    for i in n_phys..m.hp.n_layer as i64 { m.layers.push(m.layers[i % n_phys]); }
}

// models/bitnet.cpp: ternary weights; norms before attn_output and
// ffn_down, optional scales of the matmuls' results.
fn load_tensors_bitnet(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = m.tok_embd;
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        ly.attn_sub_norm = layer_tensor(m, il, "attn_sub_norm");
        ly.wq = layer_tensor(m, il, "attn_q");
        ly.wq_s = layer_tensor(m, il, "attn_q", false, "scale");
        ly.wk = layer_tensor(m, il, "attn_k");
        ly.wk_s = layer_tensor(m, il, "attn_k", false, "scale");
        ly.wv = layer_tensor(m, il, "attn_v");
        ly.wv_s = layer_tensor(m, il, "attn_v", false, "scale");
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.wo_s = layer_tensor(m, il, "attn_output", false, "scale");
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        ly.ffn_sub_norm = layer_tensor(m, il, "ffn_sub_norm");
        ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
        ly.ffn_gate_s = layer_tensor(m, il, "ffn_gate", false, "scale");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_down_s = layer_tensor(m, il, "ffn_down", false, "scale");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        ly.ffn_up_s = layer_tensor(m, il, "ffn_up", false, "scale");
        m.layers.push(ly);
    }
}

// models/chatglm.cpp: the gate and up of the SwiGLU in one tensor.
fn load_tensors_chatglm(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        load_qkv(m, il, ly);
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        m.layers.push(ly);
    }
}

// A mamba or mamba2 layer's tensors (mamba.cpp, mamba2.cpp).
fn load_mamba_layer(m: Model&, il: i64, ly: Layer&, mamba2: bool) {
    ly.ssm_in = layer_tensor(m, il, "ssm_in");
    ly.ssm_conv1d = layer_tensor(m, il, "ssm_conv1d");
    ly.ssm_conv1d_b = layer_tensor(m, il, "ssm_conv1d", true, "bias");
    ly.ssm_dt_b = layer_tensor(m, il, "ssm_dt", true, "bias");
    ly.ssm_a = tensor(m, str("blk.", il, ".ssm_a"));
    ly.ssm_d = tensor(m, str("blk.", il, ".ssm_d"));
    ly.ssm_out = layer_tensor(m, il, "ssm_out");
    if mamba2 {
        ly.ssm_norm = layer_tensor(m, il, "ssm_norm", m.hp.arch == Arch.Mamba2);
    } else {
        ly.ssm_x = layer_tensor(m, il, "ssm_x");
        ly.ssm_dt = layer_tensor(m, il, "ssm_dt");
        ly.ssm_dt_norm = layer_tensor(m, il, "ssm_dt_norm", false);
        ly.ssm_b_norm = layer_tensor(m, il, "ssm_b_norm", false);
        ly.ssm_c_norm = layer_tensor(m, il, "ssm_c_norm", false);
    }
}

// models/falcon-h1.cpp, granite-hybrid.cpp, jamba.cpp
fn load_tensors_hybrid(m: Model&) {
    let arch = m.hp.arch;
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    let rope_freqs = tensor(m, "rope_freqs.weight", false);
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        if m.hp.is_recr[il] {
            load_mamba_layer(m, il, ly, arch != Arch.Jamba);
            if arch != Arch.Jamba { ly.ssm_conv1d_b = layer_bias(m, il, "ssm_conv1d"); }   // optional there
        }
        if has_kv(m.hp, il) {
            load_qkv(m, il, ly);
            ly.wo = layer_tensor(m, il, "attn_output");
            if arch == Arch.GraniteHybrid { ly.bo = layer_bias(m, il, "attn_output"); }
        }
        // falcon-h1's ffn_norm has no ".weight"
        ly.ffn_norm = if arch == Arch.FalconH1 { tensor(m, str("blk.", il, ".ffn_norm")) } else { layer_tensor(m, il, "ffn_norm") };
        if arch != Arch.Jamba { ly.rope_freqs = rope_freqs; }
        ly.ffn_gate_inp = layer_tensor(m, il, "ffn_gate_inp", false);
        if ly.ffn_gate_inp.p != 0 || (arch == Arch.GraniteHybrid && m.hp.n_expert > 0) {
            ly.ffn_gate_inp = layer_tensor(m, il, "ffn_gate_inp");
            ly.ffn_gate_exps = layer_tensor(m, il, "ffn_gate_exps", arch == Arch.Jamba);
            ly.ffn_down_exps = layer_tensor(m, il, "ffn_down_exps");
            ly.ffn_up_exps = layer_tensor(m, il, "ffn_up_exps");
            if arch == Arch.GraniteHybrid && m.hp.n_ff_shexp > 0 {
                ly.ffn_gate_shexp = layer_tensor(m, il, "ffn_gate_shexp");
                ly.ffn_up_shexp = layer_tensor(m, il, "ffn_up_shexp");
                ly.ffn_down_shexp = layer_tensor(m, il, "ffn_down_shexp");
            }
        } else {
            ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
            ly.ffn_down = layer_tensor(m, il, "ffn_down");
            ly.ffn_up = layer_tensor(m, il, "ffn_up");
            if arch != Arch.Jamba {
                ly.ffn_gate_b = layer_bias(m, il, "ffn_gate");
                ly.ffn_down_b = layer_bias(m, il, "ffn_down");
                ly.ffn_up_b = layer_bias(m, il, "ffn_up");
            }
        }
        m.layers.push(ly);
    }
}

// models/plamo2.cpp: some norms have no ".weight" in their names.
fn load_tensors_plamo2(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    fn bare(il: i64, name: const u8[:]) -> ggml::Tensor { tensor(m, str("blk.", il, ".", name)) }
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        if m.hp.is_recr[il] {
            ly.ssm_in = layer_tensor(m, il, "ssm_in");
            ly.ssm_conv1d = layer_tensor(m, il, "ssm_conv1d");
            ly.ssm_x = layer_tensor(m, il, "ssm_x");
            ly.ssm_dt = layer_tensor(m, il, "ssm_dt");
            ly.ssm_dt_b = layer_tensor(m, il, "ssm_dt", true, "bias");
            ly.ssm_a = bare(il, "ssm_a");
            ly.ssm_d = bare(il, "ssm_d");
            ly.ssm_out = layer_tensor(m, il, "ssm_out");
            ly.ssm_dt_norm = bare(il, "ssm_dt_norm");
            ly.ssm_b_norm = bare(il, "ssm_b_norm");
            ly.ssm_c_norm = bare(il, "ssm_c_norm");
        } else {
            ly.wqkv = layer_tensor(m, il, "attn_qkv");
            ly.attn_q_norm = layer_tensor(m, il, "attn_q_norm");
            ly.attn_k_norm = layer_tensor(m, il, "attn_k_norm");
            ly.wo = layer_tensor(m, il, "attn_output");
        }
        ly.attn_post_norm = bare(il, "post_attention_norm");
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        ly.ffn_post_norm = bare(il, "post_ffw_norm");
        m.layers.push(ly);
    }
}

// models/minicpm3.cpp
fn load_tensors_minicpm3(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    let rope_long = tensor(m, "rope_factors_long.weight", false);
    let rope_short = tensor(m, "rope_factors_short.weight", false);
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        ly.attn_q_a_norm = layer_tensor(m, il, "attn_q_a_norm");
        ly.attn_kv_a_norm = layer_tensor(m, il, "attn_kv_a_norm");
        ly.wq_a = layer_tensor(m, il, "attn_q_a");
        ly.wq_b = layer_tensor(m, il, "attn_q_b");
        ly.wkv_a_mqa = layer_tensor(m, il, "attn_kv_a_mqa");
        ly.wkv_b = layer_tensor(m, il, "attn_kv_b");
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        ly.rope_long = rope_long;
        ly.rope_short = rope_short;
        m.layers.push(ly);
    }
}

// models/nemotron-h.cpp (without experts)
fn load_tensors_nemotron_h(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        if m.hp.is_recr[il] {
            load_mamba_layer(m, il, ly, true);
            ly.ssm_conv1d_b = layer_bias(m, il, "ssm_conv1d");
        } else if m.hp.n_ff[il] == 0 {
            load_qkv(m, il, ly);
            ly.wo = layer_tensor(m, il, "attn_output");
            ly.bo = layer_bias(m, il, "attn_output");
        } else {
            ly.ffn_down = layer_tensor(m, il, "ffn_down");
            ly.ffn_up = layer_tensor(m, il, "ffn_up");
            ly.ffn_down_b = layer_bias(m, il, "ffn_down");
            ly.ffn_up_b = layer_bias(m, il, "ffn_up");
        }
        m.layers.push(ly);
    }
}

// models/plamo3.cpp
fn load_tensors_plamo3(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        ly.wqkv = layer_tensor(m, il, "attn_qkv");
        ly.attn_q_norm = layer_tensor(m, il, "attn_q_norm");
        ly.attn_k_norm = layer_tensor(m, il, "attn_k_norm");
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.attn_post_norm = tensor(m, str("blk.", il, ".post_attention_norm"));    // no ".weight"
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        ly.ffn_post_norm = tensor(m, str("blk.", il, ".post_ffw_norm"));
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        m.layers.push(ly);
    }
}

// models/cohere2.cpp
fn load_tensors_cohere2(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = m.tok_embd;
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        load_qkv(m, il, ly);
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        m.layers.push(ly);
    }
}

// models/pangu-embed.cpp: llama's layers, biased attention
fn load_tensors_pangu_embed(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        load_qkv(m, il, ly);
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.bo = layer_tensor(m, il, "attn_output", true, "bias");
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        m.layers.push(ly);
    }
}

// models/limite.cpp
fn load_tensors_limite(m: Model&) {
    let hp .= m.hp;
    m.tok_embd = tensor(m, "token_embd.weight");
    m.value_embd = tensor(m, "value_embd.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    let n_taps = hp.n_mudd_taps as i64;
    for il in hp.n_layer as i64 {
        var ly = Layer { .. };
        load_qkv(m, il, ly);
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.attn_gate = layer_tensor(m, il, "attn_gate", false);
        ly.attn_ve_gate = layer_tensor(m, il, "attn_ve_gate", false);
        ly.attn_xsa_alpha = layer_tensor(m, il, "attn_xsa_alpha", false);
        ly.attn_resid_scale = layer_tensor(m, il, "attn_resid_scale");
        ly.ffn_resid_scale = layer_tensor(m, il, "ffn_resid_scale");
        ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        if n_taps > 0 && hp.mudd_tap_idx[il * n_taps] >= 0 {
            // rows [0, n_taps) mix the attention's input, rows [n_taps, 2 * n_taps) the residual
            ly.mudd_down = layer_tensor(m, il, "mudd_down");
            ly.mudd_up = layer_tensor(m, il, "mudd_up");
            ly.mudd_up_b = layer_tensor(m, il, "mudd_up", true, "bias");
        }
        m.layers.push(ly);
    }
}

// models/hrm-text.cpp: blocks [0, lps) are the low stack, [lps, 2 lps) the
// high one; every pass's slots alias them.
fn load_tensors_hrm_text(m: Model&) {
    let hp .= m.hp;
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    m.hrm_z_l_init = tensor(m, "hrm.z_l_init");
    let lps = hp.n_hrm_layers_per_stack as i64;
    let n_l = hp.n_hrm_l_cycles as i64;
    var stacks: Layer[>..] = [];
    for bid in 2 * lps {
        var ly = Layer { .. };
        load_qkv(m, bid, ly);
        ly.attn_gate = layer_tensor(m, bid, "attn_gate");
        ly.wo = layer_tensor(m, bid, "attn_output");
        ly.ffn_gate = layer_tensor(m, bid, "ffn_gate");
        ly.ffn_down = layer_tensor(m, bid, "ffn_down");
        ly.ffn_up = layer_tensor(m, bid, "ffn_up");
        stacks.push(ly);
    }
    for h in hp.n_hrm_h_cycles as i64 {
        for l in n_l + 1 {
            let base = if l == n_l { lps } else { 0 };
            for il in lps { m.layers.push(stacks[base + il]); }
        }
    }
}

// models/spark2-5.cpp
fn load_tensors_spark2_5(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        load_qkv(m, il, ly);
        ly.attn_gate = layer_tensor(m, il, "attn_gate");
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        m.layers.push(ly);
    }
}

// models/k2-horizon.cpp
fn load_tensors_k2_horizon(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        ly.wq = layer_tensor(m, il, "attn_q");
        ly.wk = layer_tensor(m, il, "attn_k");
        ly.wv = layer_tensor(m, il, "attn_v");
        if layer_tensor(m, il, "attn_q_norm", false).p != 0 || layer_tensor(m, il, "attn_gate", false).p != 0 {
            abort("k2-horizon with Q/K norms or an output gate: not ported yet");
        }
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        m.layers.push(ly);
    }
}

// models/glm4.cpp
fn load_tensors_glm4(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        load_qkv(m, il, ly);
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.attn_post_norm = layer_tensor(m, il, "post_attention_norm");
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        ly.ffn_post_norm = layer_tensor(m, il, "post_ffw_norm");
        m.layers.push(ly);
    }
}

// models/afmoe.cpp
fn load_tensors_afmoe(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        ly.attn_post_norm = layer_tensor(m, il, "post_attention_norm");
        load_qkv(m, il, ly);
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.attn_q_norm = layer_tensor(m, il, "attn_q_norm");
        ly.attn_k_norm = layer_tensor(m, il, "attn_k_norm");
        ly.attn_gate = layer_tensor(m, il, "attn_gate");
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        ly.ffn_post_norm = layer_tensor(m, il, "post_ffw_norm");
        if il >= m.hp.n_layer_dense_lead {
            ly.ffn_gate_inp = layer_tensor(m, il, "ffn_gate_inp");
            ly.ffn_exp_probs_b = layer_tensor(m, il, "exp_probs_b", true, "bias");
            ly.ffn_gate_exps = layer_tensor(m, il, "ffn_gate_exps");
            ly.ffn_down_exps = layer_tensor(m, il, "ffn_down_exps");
            ly.ffn_up_exps = layer_tensor(m, il, "ffn_up_exps");
            if m.hp.n_expert_shared > 0 {
                ly.ffn_gate_shexp = layer_tensor(m, il, "ffn_gate_shexp");
                ly.ffn_down_shexp = layer_tensor(m, il, "ffn_down_shexp");
                ly.ffn_up_shexp = layer_tensor(m, il, "ffn_up_shexp");
            }
        } else {
            ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
            ly.ffn_down = layer_tensor(m, il, "ffn_down");
            ly.ffn_up = layer_tensor(m, il, "ffn_up");
        }
        m.layers.push(ly);
    }
}

// models/gemma3n.cpp
fn load_tensors_gemma3n(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    m.altup_proj = tensor(m, "altup_proj.weight");
    m.altup_unembd_proj = tensor(m, "altup_unembd_proj.weight");
    m.per_layer_tok_embd = tensor(m, "per_layer_token_embd.weight");
    m.per_layer_model_proj = tensor(m, "per_layer_model_proj.weight");
    m.per_layer_proj_norm = tensor(m, "per_layer_proj_norm.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    for il in m.hp.n_layer as i64 {
        fn w(name: const u8[:]) -> ggml::Tensor { layer_tensor(m, il, name) }
        var ly = Layer { attn_norm: w("attn_norm"), .. };
        load_qkv(m, il, ly);
        ly.wo = w("attn_output");
        ly.attn_q_norm = w("attn_q_norm");
        ly.attn_k_norm = w("attn_k_norm");
        ly.attn_post_norm = w("post_attention_norm");
        ly.ffn_norm = w("ffn_norm");
        ly.ffn_gate = w("ffn_gate");
        ly.ffn_up = w("ffn_up");
        ly.ffn_down = w("ffn_down");
        ly.ffn_post_norm = w("post_ffw_norm");
        ly.per_layer_inp_gate = w("inp_gate");
        ly.per_layer_proj = w("proj");
        ly.per_layer_post_norm = w("post_norm");
        ly.altup_correct_coef = w("altup_correct_coef");
        ly.altup_correct_scale = w("altup_correct_scale");
        ly.altup_predict_coef = w("altup_predict_coef");
        ly.altup_router = w("altup_router");
        ly.altup_router_norm = w("altup_router_norm");
        ly.laurel_l = w("laurel_l");
        ly.laurel_r = w("laurel_r");
        ly.laurel_post_norm = w("laurel_post_norm");
        m.layers.push(ly);
    }
}

// models/gemma4.cpp (without experts)
fn load_tensors_gemma4(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    if m.hp.n_embd_per_layer > 0 {
        m.per_layer_tok_embd = tensor(m, "per_layer_token_embd.weight");
        m.per_layer_model_proj = tensor(m, "per_layer_model_proj.weight");
        m.per_layer_proj_norm = tensor(m, "per_layer_proj_norm.weight");
    }
    m.output_norm = tensor(m, "output_norm.weight");
    let rope_freqs = tensor(m, "rope_freqs.weight", false);
    for il in m.hp.n_layer as i64 {
        fn w(name: const u8[:], required: bool = true) -> ggml::Tensor { layer_tensor(m, il, name, required) }
        let kv = has_kv(m.hp, il);
        var ly = Layer { attn_norm: w("attn_norm"), .. };
        ly.wqkv = w("attn_qkv", false);
        if ly.wqkv.p == 0 {
            ly.wq = w("attn_q");
            ly.wk = w("attn_k", kv);
            ly.wv = w("attn_v", false);      // absent: v is k
        }
        ly.wo = w("attn_output");
        ly.attn_q_norm = w("attn_q_norm");
        ly.attn_k_norm = w("attn_k_norm", kv);
        ly.attn_post_norm = w("post_attention_norm");
        ly.out_scale = w("layer_output_scale", false);
        if !m.hp.is_swa[il] {
            // proportional rope on the full attention layers
            if rope_freqs.p == 0 { abort("missing tensor 'rope_freqs.weight'"); }
            ly.rope_freqs = rope_freqs;
        }
        ly.ffn_norm = w("ffn_norm");
        ly.ffn_gate = w("ffn_gate");
        ly.ffn_up = w("ffn_up");
        ly.ffn_down = w("ffn_down");
        ly.ffn_post_norm = w("post_ffw_norm");
        if w("ffn_gate_inp", false).p != 0 { abort("gemma4 with experts (MoE): not ported yet"); }
        if m.hp.n_embd_per_layer > 0 {
            ly.per_layer_inp_gate = w("inp_gate");
            ly.per_layer_proj = w("proj");
            ly.per_layer_post_norm = w("post_norm");
        }
        m.layers.push(ly);
    }
}

// models/t5.cpp
fn load_tensors_t5(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm_enc = tensor(m, "enc.output_norm.weight");
    m.output_norm = tensor(m, "dec.output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    for il in m.hp.n_layer as i64 {
        fn e(name: const u8[:], required: bool = true) -> ggml::Tensor { tensor(m, str("enc.blk.", il, ".", name, ".weight"), required) }
        fn d(name: const u8[:], required: bool = true) -> ggml::Tensor { tensor(m, str("dec.blk.", il, ".", name, ".weight"), required) }
        var ly = Layer { attn_norm_enc: e("attn_norm"), .. };
        ly.attn_rel_b_enc = e("attn_rel_b", false);
        ly.wq_enc = e("attn_q");
        ly.wk_enc = e("attn_k");
        ly.wv_enc = e("attn_v");
        ly.wo_enc = e("attn_o");
        ly.ffn_norm_enc = e("ffn_norm");
        ly.ffn_gate_enc = e("ffn_gate", false);
        ly.ffn_down_enc = e("ffn_down");
        ly.ffn_up_enc = e("ffn_up");
        ly.attn_norm = d("attn_norm");
        ly.attn_rel_b = d("attn_rel_b", false);
        ly.wq = d("attn_q");
        ly.wk = d("attn_k");
        ly.wv = d("attn_v");
        ly.wo = d("attn_o");
        ly.attn_norm_cross = d("cross_attn_norm");
        ly.wq_cross = d("cross_attn_q");
        ly.wk_cross = d("cross_attn_k");
        ly.wv_cross = d("cross_attn_v");
        ly.wo_cross = d("cross_attn_o");
        ly.ffn_norm = d("ffn_norm");
        ly.ffn_gate = d("ffn_gate", false);
        ly.ffn_down = d("ffn_down");
        ly.ffn_up = d("ffn_up");
        m.layers.push(ly);
    }
}

// models/rwkv6.cpp, models/rwkv7.cpp
fn load_tensors_rwkv(m: Model&) {
    let rwkv7 = m.hp.arch == Arch.Rwkv7;
    m.tok_embd = tensor(m, "token_embd.weight");
    m.tok_norm = tensor(m, "token_embd_norm.weight");
    m.tok_norm_b = tensor(m, "token_embd_norm.bias");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output_norm_b = tensor(m, "output_norm.bias");
    m.output = tensor(m, "output.weight");
    for il in m.hp.n_layer as i64 {
        fn w(name: const u8[:], required: bool = true) -> ggml::Tensor { layer_tensor(m, il, name, required) }
        var ly = Layer { attn_norm: w("attn_norm"), .. };
        ly.attn_norm_b = layer_tensor(m, il, "attn_norm", true, "bias");
        ly.attn_norm_2 = w("attn_norm_2");
        ly.attn_norm_2_b = layer_tensor(m, il, "attn_norm_2", true, "bias");
        ly.time_mix_w1 = w("time_mix_w1");
        ly.time_mix_w2 = w("time_mix_w2");
        ly.time_mix_lerp_fused = w("time_mix_lerp_fused", rwkv7);
        ly.time_mix_key = w("time_mix_key");
        ly.time_mix_value = w("time_mix_value");
        ly.time_mix_receptance = w("time_mix_receptance");
        ly.time_mix_ln = w("time_mix_ln");
        ly.time_mix_ln_b = layer_tensor(m, il, "time_mix_ln", true, "bias");
        ly.time_mix_output = w("time_mix_output");
        ly.channel_mix_lerp_k = w("channel_mix_lerp_k");
        ly.channel_mix_key = w("channel_mix_key");
        ly.channel_mix_value = w("channel_mix_value");
        if rwkv7 {
            ly.time_mix_w0 = w("time_mix_w0");
            ly.time_mix_a0 = w("time_mix_a0");
            ly.time_mix_a1 = w("time_mix_a1");
            ly.time_mix_a2 = w("time_mix_a2");
            ly.time_mix_v0 = w("time_mix_v0");
            ly.time_mix_v1 = w("time_mix_v1");
            ly.time_mix_v2 = w("time_mix_v2");
            ly.time_mix_g1 = w("time_mix_g1");
            ly.time_mix_g2 = w("time_mix_g2");
            ly.time_mix_k_k = w("time_mix_k_k");
            ly.time_mix_k_a = w("time_mix_k_a");
            ly.time_mix_r_k = w("time_mix_r_k");
        } else {
            ly.time_mix_lerp_x = w("time_mix_lerp_x");
            ly.time_mix_lerp_w = w("time_mix_lerp_w", false);
            ly.time_mix_lerp_k = w("time_mix_lerp_k", false);
            ly.time_mix_lerp_v = w("time_mix_lerp_v", false);
            ly.time_mix_lerp_r = w("time_mix_lerp_r", false);
            ly.time_mix_lerp_g = w("time_mix_lerp_g", false);
            if ly.time_mix_lerp_fused.p == 0 && ly.time_mix_lerp_w.p == 0 {
                abort("GGML_ASSERT(!(layer.time_mix_lerp_fused == NULL && layer.time_mix_lerp_w == NULL)) failed");
            }
            ly.time_mix_first = w("time_mix_first");
            ly.time_mix_decay = w("time_mix_decay");
            ly.time_mix_decay_w1 = w("time_mix_decay_w1");
            ly.time_mix_decay_w2 = w("time_mix_decay_w2");
            ly.time_mix_gate = w("time_mix_gate");
            ly.channel_mix_lerp_r = w("channel_mix_lerp_r");
            ly.channel_mix_receptance = w("channel_mix_receptance");
        }
        m.layers.push(ly);
    }
}

// models/mamba.cpp, models/mamba2.cpp
fn load_tensors_mamba(m: Model&) {
    let mamba2 = m.hp.arch == Arch.Mamba2;
    if !mamba2 && 2 * m.hp.n_embd != m.hp.ssm_d_inner { abort("only an expansion factor of 2 is supported for now"); }
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        load_mamba_layer(m, il, ly, mamba2);
        m.layers.push(ly);
    }
}

fn load_tensors_gemma2(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = m.tok_embd;
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        load_qkv(m, il, ly);
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.attn_post_norm = layer_tensor(m, il, "post_attention_norm");
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_post_norm = layer_tensor(m, il, "post_ffw_norm");
        m.layers.push(ly);
    }
}

fn load_tensors_gemma3(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }       // tied to the input embeddings
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        load_qkv(m, il, ly);
        ly.wo = layer_tensor(m, il, "attn_output");
        ly.attn_post_norm = layer_tensor(m, il, "post_attention_norm");
        ly.attn_q_norm = layer_tensor(m, il, "attn_q_norm");
        ly.attn_k_norm = layer_tensor(m, il, "attn_k_norm");
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_post_norm = layer_tensor(m, il, "post_ffw_norm");
        m.layers.push(ly);
    }
}

fn load_tensors_lfm2(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "token_embd_norm.weight");    // llama.cpp's fix for a wrong tensor name
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        if tensor(m, str("blk.", il, ".ffn_gate_inp.weight"), false).p != 0 { abort("lfm2 with experts (MoE): not ported yet"); }
        ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        if !m.hp.is_recr[il] {
            ly.attn_q_norm = layer_tensor(m, il, "attn_q_norm");
            ly.attn_k_norm = layer_tensor(m, il, "attn_k_norm");
            load_qkv(m, il, ly);
            ly.wo = layer_tensor(m, il, "attn_output");
        } else {
            ly.shortconv_conv = layer_tensor(m, il, "shortconv.conv");
            ly.shortconv_in_proj = layer_tensor(m, il, "shortconv.in_proj");
            ly.shortconv_out_proj = layer_tensor(m, il, "shortconv.out_proj");
        }
        m.layers.push(ly);
    }
}

fn load_tensors_lfm3(m: Model&) {
    m.tok_embd = tensor(m, "token_embd.weight");
    m.output_norm = tensor(m, "output_norm.weight");
    m.output = tensor(m, "output.weight", false);
    if m.output.p == 0 { m.output = m.tok_embd; }
    let rope_freqs = tensor(m, "rope_freqs.weight", false);
    for il in m.hp.n_layer as i64 {
        var ly = Layer { attn_norm: layer_tensor(m, il, "attn_norm"), .. };
        ly.ffn_norm = layer_tensor(m, il, "ffn_norm");
        // absent when the model has no sandwich norms
        ly.attn_post_norm = layer_tensor(m, il, "post_attention_norm", false);
        ly.ffn_post_norm = layer_tensor(m, il, "post_ffw_norm", false);
        if (ly.attn_post_norm.p == 0) != (ly.ffn_post_norm.p == 0) { abort("LFM3 requires both post norms or neither"); }
        ly.ffn_gate = layer_tensor(m, il, "ffn_gate");
        ly.ffn_down = layer_tensor(m, il, "ffn_down");
        ly.ffn_up = layer_tensor(m, il, "ffn_up");
        if !m.hp.is_recr[il] {
            ly.attn_q_norm = layer_tensor(m, il, "attn_q_norm");
            ly.attn_k_norm = layer_tensor(m, il, "attn_k_norm");
            load_qkv(m, il, ly);
            ly.wo = layer_tensor(m, il, "attn_output");
            ly.attn_gate = layer_tensor(m, il, "attn_gate");
            if !m.hp.is_swa[il] {
                if rope_freqs.p == 0 { abort("missing tensor 'rope_freqs.weight'"); }
                ly.rope_freqs = rope_freqs;
            }
        } else {
            ly.shortconv_conv = layer_tensor(m, il, "shortconv.conv");
            ly.shortconv_in_proj = layer_tensor(m, il, "shortconv.in_proj");
            ly.shortconv_out_proj = layer_tensor(m, il, "shortconv.out_proj");
        }
        m.layers.push(ly);
    }
}

// The weights read by mul_mat, which llama.cpp places in the CPU's repack
// buffers where it can (its create_tensor gives them GGML_OP_MUL_MAT).
fn matmul_weights(m: Model&) -> ggml::Tensor[>..] {
    var out: ggml::Tensor[>..] = [];
    fn add(t: ggml::Tensor) { if t.p != 0 { out.push(t); } }
    for ly in m.layers {
        add(ly.wqkv); add(ly.wq); add(ly.wk); add(ly.wv); add(ly.wo);
        add(ly.ffn_gate); add(ly.ffn_up); add(ly.ffn_down);
        add(ly.shortconv_in_proj); add(ly.shortconv_out_proj); add(ly.attn_gate);
        add(ly.ssm_beta); add(ly.ssm_alpha); add(ly.ssm_out);
        add(ly.attn_ve_gate); add(ly.mudd_down); add(ly.mudd_up);
        add(ly.ssm_in); add(ly.ssm_x); add(ly.ssm_dt);
        add(ly.wq_a); add(ly.wq_b); add(ly.wkv_a_mqa); add(ly.wkv_b);
        add(ly.wq_enc); add(ly.wk_enc); add(ly.wv_enc); add(ly.wo_enc); add(ly.ffn_gate_enc); add(ly.ffn_down_enc); add(ly.ffn_up_enc);
        add(ly.wq_cross); add(ly.wk_cross); add(ly.wv_cross); add(ly.wo_cross);
        add(ly.per_layer_inp_gate); add(ly.per_layer_proj); add(ly.altup_correct_coef); add(ly.altup_predict_coef);
        add(ly.altup_router); add(ly.laurel_l); add(ly.laurel_r);
        add(ly.time_mix_w1); add(ly.time_mix_w2); add(ly.time_mix_a1); add(ly.time_mix_a2); add(ly.time_mix_v1);
        add(ly.time_mix_v2); add(ly.time_mix_g1); add(ly.time_mix_g2); add(ly.time_mix_decay_w1); add(ly.time_mix_decay_w2);
        add(ly.time_mix_key); add(ly.time_mix_value); add(ly.time_mix_receptance); add(ly.time_mix_gate); add(ly.time_mix_output);
        add(ly.channel_mix_key); add(ly.channel_mix_receptance); add(ly.channel_mix_value);
        add(ly.ffn_gate_inp); add(ly.ffn_up_shexp); add(ly.ffn_gate_shexp); add(ly.ffn_down_shexp);
    }
    add(m.per_layer_model_proj); add(m.altup_proj); add(m.altup_unembd_proj);
    // a tied output is the token embeddings, read by get_rows
    if m.output.p != m.tok_embd.p { add(m.output); }
    return out;
}

// The experts' weights, read by mul_mat_id.
fn expert_weights(m: Model&) -> ggml::Tensor[>..] {
    var out: ggml::Tensor[>..] = [];
    for ly in m.layers {
        for t in [ly.ffn_up_exps, ly.ffn_gate_exps, ly.ffn_down_exps] { if t.p != 0 { out.push(t); } }
    }
    return out;
}

// llama_load_mode: how the weights get to memory. Auto is mmap on the CPU.
enum LoadMode { Auto, None, Mmap, Mlock, MmapMlock, Dio }

// llama_lazy_mode: whether the tensors an architecture marks (gemma4's
// per-layer embeddings) are read from the file as they are used, rows
// read ahead for each batch; Auto only for those above 4 GiB.
enum LazyMode { Off, Auto, On }

// What llama_model_params says of loading the weights (common/load sets
// them from the flags): the mode, whether they may go in the CPU's repack
// buffers (use_extra_bufts), and the patterns, comma-separated, of the
// tensors overridden to the CPU buffer type (tensor_buft_overrides, which
// on the CPU alone place nothing elsewhere).
var load_mode = LoadMode.Auto;
var lazy_mode = LazyMode.Auto;
var repack = true;
var cpu_overrides: u8[>..] = [];

// llama_load_mode's number, as the loader's C takes it; Auto resolved.
fn mode_number(mode: LoadMode) -> i32 {
    match mode { Auto, Mmap => 1, None => 0, Mlock => 2, MmapMlock => 3, Dio => 4 }
}

// Whether a tensor's name matches one of cpu_overrides (std::regex_search).
fn any_tensor_matches(m: Model&) -> bool {
    var stack: ecma::Frame[>..<] = [];
    var name: u8[>..] = [];
    var found = false;
    each_split(cpu_overrides, ',') {
        let pattern = it;
        var insts: ecma::Inst[>..] = [];
        var items: ecma::Item[>..] = [];
        var classes: ecma::Class[>..] = [];
        ecma::compile(unicode::cpts_from_utf8(pattern), insts, items, classes);
        let prog = ecma::Prog { insts: insts, items: items, classes: classes };
        for id in gguf::get_n_tensors(m.meta) {
            name.clear();
            gguf::get_tensor_name(m.meta, id, name);
            let text = unicode::cpts_from_utf8(name);
            let s, e = ecma::search(prog, text, 0, text.len, 0, false, false, stack);
            if s >= 0 { found = true; }
        }
    };
    return found;
}

// Loads a model, and its vocabulary into the vocab module.
fn load(m: Model&, path: const u8[:]) {
    m.ctx = ggml::Ctx { p: 0 };
    m.meta = gguf::init_with_tensors(path, true, m.ctx);
    guard m.meta.p != 0 else { abort(str("failed to load model from ", path)); }

    var arch: u8[>..] = [];
    guard vocab::get_str(m.meta, "general.architecture", arch) else { abort("key not found in model: general.architecture"); }
    m.hp.arch = arch_from_name(arch);
    guard m.hp.arch != Arch.Unknown else { abort(str("unknown model architecture: '", arch, "'")); }

    load_hparams(m);
    vocab::load(m.meta);

    match m.hp.arch {
        Gemma3 => load_tensors_gemma3(m),
        Llama => load_tensors_llama(m),
        Lfm2 => load_tensors_lfm2(m),
        Lfm3 => load_tensors_lfm3(m),
        Qwen2, Qwen3 => load_tensors_qwen(m),
        Qwen35 => load_tensors_qwen35(m),
        Gemma2 => load_tensors_gemma2(m),
        Phi2 => load_tensors_phi2(m),
        Phi3 => load_tensors_phi3(m),
        Stablelm => load_tensors_stablelm(m),
        Smollm3, Granite, GraniteMoe, Minicpm => load_tensors_smollm3_granite(m),
        Smallthinker => load_tensors_smallthinker(m),
        Gemma => load_tensors_gemma(m),
        Olmo => load_tensors_olmo(m),
        Qwen => load_tensors_qwen(m),
        Refact => load_tensors_refact(m),
        Mpt, Jais, Nemotron => load_tensors_classic(m),
        Maincoder => load_tensors_hunyuan_dense(m),
        Nanbeige => load_tensors_nanbeige(m),
        Bitnet => load_tensors_bitnet(m),
        Chatglm => load_tensors_chatglm(m),
        Mamba, Mamba2 => load_tensors_mamba(m),
        FalconH1, GraniteHybrid, Jamba => load_tensors_hybrid(m),
        Plamo2 => load_tensors_plamo2(m),
        Rwkv6, Rwkv7 => load_tensors_rwkv(m),
        Minicpm3 => load_tensors_minicpm3(m),
        Qwen2vl, Qwen3vl => load_tensors_qwen(m),
        NemotronH => load_tensors_nemotron_h(m),
        Afmoe => load_tensors_afmoe(m),
        Gemma3n => load_tensors_gemma3n(m),
        Gemma4 => load_tensors_gemma4(m),
        T5 => load_tensors_t5(m),
        Plamo3 => load_tensors_plamo3(m),
        Cohere2 => load_tensors_cohere2(m),
        PanguEmbed => load_tensors_pangu_embed(m),
        Limite => load_tensors_limite(m),
        HrmText => load_tensors_hrm_text(m),
        Spark2_5 => load_tensors_spark2_5(m),
        K2Horizon => load_tensors_k2_horizon(m),
        Glm4 => load_tensors_glm4(m),
        Internlm2, Exaone, Ernie4_5, Mistral3 => load_tensors_llama(m),
        Arcee => load_tensors_arcee(m),
        Olmo2, Exaone4 => load_tensors_post_norm(m),
        HunyuanDense => load_tensors_hunyuan_dense(m),
        Gpt2, Starcoder, Gptneox, Starcoder2, Bloom => load_tensors_classic(m),
        Openelm => load_tensors_openelm(m),
        _ => abort("unknown architecture"),
    }

    m.backend = ggml::backend_cpu_init();
    ggml::set_quiet(if llama::quiet { 1 } else { 0 });
    let overridden = cpu_overrides.len > 0 && any_tensor_matches(m);
    let tied = if m.output.p != 0 && m.output.p == m.tok_embd.p { m.tok_embd } else { ggml::NONE };
    var output = m.output;
    // TENSOR_READ_LAZY: gemma4's per-layer embeddings
    let ple = m.per_layer_tok_embd;
    m.lazy_ple = m.hp.arch == Arch.Gemma4 && ple.p != 0 &&
                 (lazy_mode == LazyMode.On || (lazy_mode == LazyMode.Auto && ggml::nbytes(ple) > 4 * 1024 * 1024 * 1024));
    let lazy = if m.lazy_ple { ple } else { ggml::NONE };
    guard ggml::load_weights(m.meta, m.ctx, path, matmul_weights(m), expert_weights(m), m.hp.n_expert_used, mode_number(load_mode),
                             repack, overridden, tied, lazy, unused_tensors(m), m.bufs, m.host_weight_bytes, output) else { abort(str("failed to load the tensors of ", path)); }
    if tied.p != 0 { m.output = output; }
    print_info(m);
}

fn print_info(m: Model&) {
    let hp .= m.hp;
    var out: u8[>..] = [];
    format(out, "print_info: arch                  = ", arch_name_of(hp.arch), "\n");
    format(out, "print_info: n_ctx_train           = ", hp.n_ctx_train, "\n");
    format(out, "print_info: n_embd                = ", hp.n_embd, "\n");
    format(out, "print_info: n_layer               = ", hp.n_layer, "\n");
    format(out, "print_info: n_head                = ", hp.n_head[0], "\n");
    format(out, "print_info: n_head_kv             = ", hp.n_head_kv[0], "\n");
    format(out, "print_info: n_rot                 = ", hp.n_rot, "\n");
    format(out, "print_info: n_swa                 = ", hp.n_swa, "\n");
    format(out, "print_info: n_embd_head_k         = ", hp.n_embd_head_k, "\n");
    format(out, "print_info: n_embd_head_v         = ", hp.n_embd_head_v, "\n");
    format(out, "print_info: f_norm_rms_eps        = ", hp.f_norm_rms_eps, "\n");
    format(out, "print_info: f_attn_scale          = ", hp.f_attention_scale, "\n");
    format(out, "print_info: n_ff                  = ", hp.n_ff[0], "\n");
    format(out, "print_info: rope type             = ", hp.rope_type, "\n");
    format(out, "print_info: freq_base_train       = ", hp.rope_freq_base_train, "\n");
    format(out, "print_info: freq_scale_train      = ", hp.rope_freq_scale_train, "\n");
    format(out, "print_info: n_ctx_orig_yarn       = ", hp.n_ctx_orig_yarn, "\n");
    if !llama::quiet { write_stderr(out); }
}
