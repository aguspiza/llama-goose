// Goose bindings for ggml. ggml objects cross as one-pointer structs, so a
// tensor cannot be passed where a context is expected.
namespace ggml;

struct Ctx { p: i64 }
struct Tensor { p: i64 }
struct Graph { p: i64 }
struct Backend { p: i64 }
struct Buffer { p: i64 }
struct Gallocr { p: i64 }

let NONE = Tensor { p: 0 };

// enum ggml_type
let TYPE_F32 = 0;
let TYPE_F16 = 1;
let TYPE_Q8_0 = 8;
let TYPE_I32 = 26;
let TYPE_I64 = 27;

// enum ggml_prec
let PREC_F32 = 10;

let ROPE_TYPE_NORMAL = 0;
let ROPE_TYPE_NEOX = 2;
let ROPE_TYPE_MROPE = 8;
let ROPE_TYPE_IMROPE = 40;

let DEFAULT_GRAPH_SIZE = 2048;

extern "ggml_free" fn free(ctx: Ctx);
extern "ggml_tensor_overhead" fn tensor_overhead() -> i64;
extern "ggml_graph_overhead" fn graph_overhead() -> i64;
extern "ggml_graph_overhead_custom" fn graph_overhead_custom(size: i64, grads: bool) -> i64;

// tensors
extern "ggml_new_tensor_1d" fn new_tensor_1d(ctx: Ctx, typ: i32, ne0: i64) -> Tensor;
extern "ggml_new_tensor_2d" fn new_tensor_2d(ctx: Ctx, typ: i32, ne0: i64, ne1: i64) -> Tensor;
extern "ggml_new_tensor_3d" fn new_tensor_3d(ctx: Ctx, typ: i32, ne0: i64, ne1: i64, ne2: i64) -> Tensor;
extern "ggml_new_tensor_4d" fn new_tensor_4d(ctx: Ctx, typ: i32, ne0: i64, ne1: i64, ne2: i64, ne3: i64) -> Tensor;
extern "ggml_nbytes" fn nbytes(t: Tensor) -> i64;
extern "ggml_nelements" fn nelements(t: Tensor) -> i64;
extern "ggml_row_size" fn row_size(typ: i32, ne: i64) -> i64;
extern "ggml_set_input" fn set_input(t: Tensor);
extern "ggml_set_output" fn set_output(t: Tensor);

// operations
extern "ggml_get_rows" fn get_rows(ctx: Ctx, a: Tensor, rows: Tensor) -> Tensor;
extern "ggml_set_rows" fn set_rows(ctx: Ctx, dst: Tensor, src: Tensor, rows: Tensor) -> Tensor;
extern "ggml_scale" fn scale(ctx: Ctx, a: Tensor, s: f32) -> Tensor;
extern "ggml_add" fn add(ctx: Ctx, a: Tensor, b: Tensor) -> Tensor;
extern "ggml_mul" fn mul(ctx: Ctx, a: Tensor, b: Tensor) -> Tensor;
extern "ggml_sub" fn sub(ctx: Ctx, a: Tensor, b: Tensor) -> Tensor;
extern "ggml_exp" fn exp(ctx: Ctx, a: Tensor) -> Tensor;
extern "ggml_sqrt" fn sqrt(ctx: Ctx, a: Tensor) -> Tensor;
extern "ggml_mean" fn mean(ctx: Ctx, a: Tensor) -> Tensor;
extern "ggml_scale_bias" fn scale_bias(ctx: Ctx, a: Tensor, s: f32, b: f32) -> Tensor;
extern "ggml_neg" fn neg(ctx: Ctx, a: Tensor) -> Tensor;
extern "ggml_repeat" fn repeat(ctx: Ctx, a: Tensor, b: Tensor) -> Tensor;
extern "ggml_repeat_4d" fn repeat_4d(ctx: Ctx, a: Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64) -> Tensor;
extern "ggml_l2_norm" fn l2_norm(ctx: Ctx, a: Tensor, eps: f32) -> Tensor;
extern "ggml_mul_mat" fn mul_mat(ctx: Ctx, a: Tensor, b: Tensor) -> Tensor;
extern "ggml_rms_norm" fn rms_norm(ctx: Ctx, a: Tensor, eps: f32) -> Tensor;
extern "ggml_norm" fn norm(ctx: Ctx, a: Tensor, eps: f32) -> Tensor;
extern "ggml_tanh" fn tanh(ctx: Ctx, a: Tensor) -> Tensor;
extern "ggml_sigmoid" fn sigmoid(ctx: Ctx, a: Tensor) -> Tensor;
extern "ggml_relu" fn relu(ctx: Ctx, a: Tensor) -> Tensor;
extern "ggml_sqr" fn sqr(ctx: Ctx, a: Tensor) -> Tensor;
extern "ggml_silu" fn silu(ctx: Ctx, a: Tensor) -> Tensor;
extern "ggml_softplus" fn softplus(ctx: Ctx, a: Tensor) -> Tensor;
extern "ggml_gelu" fn gelu(ctx: Ctx, a: Tensor) -> Tensor;
extern "ggml_gelu_erf" fn gelu_erf(ctx: Ctx, a: Tensor) -> Tensor;
extern "ggml_geglu_split" fn geglu_split(ctx: Ctx, a: Tensor, b: Tensor) -> Tensor;
extern "ggml_swiglu" fn swiglu(ctx: Ctx, a: Tensor) -> Tensor;
extern "ggml_swiglu_split" fn swiglu_split(ctx: Ctx, a: Tensor, b: Tensor) -> Tensor;
extern "ggml_soft_max" fn soft_max(ctx: Ctx, a: Tensor) -> Tensor;
extern "ggml_soft_max_ext" fn soft_max_ext(ctx: Ctx, a: Tensor, mask: Tensor, scale: f32, max_bias: f32) -> Tensor;
extern "ggml_reshape_1d" fn reshape_1d(ctx: Ctx, a: Tensor, ne0: i64) -> Tensor;
extern "ggml_argsort_top_k" fn argsort_top_k(ctx: Ctx, a: Tensor, k: i32) -> Tensor;
extern "ggml_sum_rows" fn sum_rows(ctx: Ctx, a: Tensor) -> Tensor;
extern "ggml_clamp" fn clamp(ctx: Ctx, a: Tensor, min: f32, max: f32) -> Tensor;
extern "ggml_div" fn div(ctx: Ctx, a: Tensor, b: Tensor) -> Tensor;
extern "ggml_mul_mat_id" fn mul_mat_id(ctx: Ctx, as_: Tensor, b: Tensor, ids: Tensor) -> Tensor;
extern "ggml_reglu_split" fn reglu_split(ctx: Ctx, a: Tensor, b: Tensor) -> Tensor;
extern "ggml_cont" fn cont(ctx: Ctx, a: Tensor) -> Tensor;
extern "ggml_cast" fn cast(ctx: Ctx, a: Tensor, typ: i32) -> Tensor;
extern "ggml_reshape_2d" fn reshape_2d(ctx: Ctx, a: Tensor, ne0: i64, ne1: i64) -> Tensor;
extern "ggml_reshape_3d" fn reshape_3d(ctx: Ctx, a: Tensor, ne0: i64, ne1: i64, ne2: i64) -> Tensor;
extern "ggml_reshape_4d" fn reshape_4d(ctx: Ctx, a: Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64) -> Tensor;
extern "ggml_cont_2d" fn cont_2d(ctx: Ctx, a: Tensor, ne0: i64, ne1: i64) -> Tensor;
extern "ggml_cont_3d" fn cont_3d(ctx: Ctx, a: Tensor, ne0: i64, ne1: i64, ne2: i64) -> Tensor;
extern "ggml_cont_4d" fn cont_4d(ctx: Ctx, a: Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64) -> Tensor;
extern "ggml_view_2d" fn view_2d(ctx: Ctx, a: Tensor, ne0: i64, ne1: i64, nb1: i64, offset: i64) -> Tensor;
extern "ggml_view_4d" fn view_4d(ctx: Ctx, a: Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64,
                                 nb1: i64, nb2: i64, nb3: i64, offset: i64) -> Tensor;
extern "ggml_view_3d" fn view_3d(ctx: Ctx, a: Tensor, ne0: i64, ne1: i64, ne2: i64, nb1: i64, nb2: i64, offset: i64) -> Tensor;
extern "ggml_transpose" fn transpose(ctx: Ctx, a: Tensor) -> Tensor;
extern "ggml_pad" fn pad_ext(ctx: Ctx, a: Tensor, p0: i32, p1: i32, p2: i32, p3: i32) -> Tensor;
extern "ggml_concat" fn concat(ctx: Ctx, a: Tensor, b: Tensor, dim: i32) -> Tensor;
extern "ggml_cpy" fn cpy(ctx: Ctx, a: Tensor, b: Tensor) -> Tensor;
extern "ggml_ssm_conv" fn ssm_conv(ctx: Ctx, sx: Tensor, c: Tensor) -> Tensor;
// the output: y [d_inner, n_seq_tokens, n_seqs], then K states
extern "ggml_ssm_scan" fn ssm_scan(ctx: Ctx, s: Tensor, x: Tensor, dt: Tensor, a: Tensor, b: Tensor, c: Tensor,
                                   ids: Tensor, n_states: i64) -> Tensor;
extern "ggml_view_1d" fn view_1d(ctx: Ctx, a: Tensor, ne0: i64, offset: i64) -> Tensor;
extern "ggml_element_size" fn element_size(t: Tensor) -> i64;
extern "ggml_permute" fn permute(ctx: Ctx, a: Tensor, ax0: i32, ax1: i32, ax2: i32, ax3: i32) -> Tensor;
extern "ggml_rope_ext" fn rope_ext(ctx: Ctx, a: Tensor, pos: Tensor, freqs: Tensor, n_dims: i32, mode: i32,
                                   n_ctx_orig: i32, freq_base: f32, freq_scale: f32, ext_factor: f32,
                                   attn_factor: f32, beta_fast: f32, beta_slow: f32) -> Tensor;
// rope only dimensions n_offs.. of a rope node
extern "ggml_rope_set_offset" fn rope_set_offset(a: Tensor, n_offs: i32) -> Tensor;
extern "ggml_flash_attn_ext" fn flash_attn_ext(ctx: Ctx, q: Tensor, k: Tensor, v: Tensor, mask: Tensor,
                                               scale: f32, max_bias: f32, logit_softcap: f32) -> Tensor;
// the output: the heads' values [S_v, H_v, n_tokens, n_seqs], then K states
extern "ggml_gated_delta_net" fn gated_delta_net(ctx: Ctx, q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
                                                 state: Tensor, n_states: i64) -> Tensor;
// RWKV's attentions: the output, then the new state
extern "ggml_rwkv_wkv6" fn rwkv_wkv6(ctx: Ctx, k: Tensor, v: Tensor, r: Tensor, tf: Tensor, td: Tensor, state: Tensor) -> Tensor;
extern "ggml_rwkv_wkv7" fn rwkv_wkv7(ctx: Ctx, r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor, state: Tensor) -> Tensor;
extern "ggml_gated_linear_attn" fn gated_linear_attn(ctx: Ctx, k: Tensor, v: Tensor, q: Tensor, g: Tensor, state: Tensor,
                                                     scale: f32) -> Tensor;
extern "ggml_prec_set_acc" fn prec_set_acc(a: Tensor, prec: i32) -> bool;

// graphs
extern "ggml_new_graph_custom" fn new_graph_custom(ctx: Ctx, size: i64, grads: bool) -> Graph;
extern "ggml_new_graph" fn new_graph(ctx: Ctx) -> Graph;
extern "ggml_build_forward_expand" fn build_forward_expand(g: Graph, t: Tensor);
extern "ggml_graph_n_nodes" fn graph_n_nodes(g: Graph) -> i32;

extern "ggml_backend_free" fn backend_free(b: Backend);
extern "ggml_backend_alloc_ctx_tensors" fn backend_alloc_ctx_tensors(ctx: Ctx, b: Backend) -> Buffer;
extern "ggml_backend_buffer_clear" fn backend_buffer_clear(buf: Buffer, value: u8);
extern "ggml_backend_buffer_free" fn backend_buffer_free(buf: Buffer);
extern "ggml_backend_graph_compute" fn backend_graph_compute(b: Backend, g: Graph) -> i32;
extern "ggml_gallocr_free" fn gallocr_free(g: Gallocr);
extern "ggml_gallocr_alloc_graph" fn gallocr_alloc_graph(g: Gallocr, graph: Graph) -> bool;


