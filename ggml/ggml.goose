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

extern fn free(ctx: Ctx) = "ggml_free";
extern fn tensor_overhead() -> i64 = "ggml_tensor_overhead";
extern fn graph_overhead() -> i64 = "ggml_graph_overhead";
extern fn graph_overhead_custom(size: i64, grads: bool) -> i64 = "ggml_graph_overhead_custom";

// tensors
extern fn new_tensor_1d(ctx: Ctx, typ: i32, ne0: i64) -> Tensor = "ggml_new_tensor_1d";
extern fn new_tensor_2d(ctx: Ctx, typ: i32, ne0: i64, ne1: i64) -> Tensor = "ggml_new_tensor_2d";
extern fn new_tensor_3d(ctx: Ctx, typ: i32, ne0: i64, ne1: i64, ne2: i64) -> Tensor = "ggml_new_tensor_3d";
extern fn new_tensor_4d(ctx: Ctx, typ: i32, ne0: i64, ne1: i64, ne2: i64, ne3: i64) -> Tensor = "ggml_new_tensor_4d";
extern fn nbytes(t: Tensor) -> i64 = "ggml_nbytes";
extern fn nelements(t: Tensor) -> i64 = "ggml_nelements";
extern fn row_size(typ: i32, ne: i64) -> i64 = "ggml_row_size";
extern fn set_input(t: Tensor) = "ggml_set_input";
extern fn set_output(t: Tensor) = "ggml_set_output";

// operations
extern fn get_rows(ctx: Ctx, a: Tensor, rows: Tensor) -> Tensor = "ggml_get_rows";
extern fn set_rows(ctx: Ctx, dst: Tensor, src: Tensor, rows: Tensor) -> Tensor = "ggml_set_rows";
extern fn scale(ctx: Ctx, a: Tensor, s: f32) -> Tensor = "ggml_scale";
extern fn add(ctx: Ctx, a: Tensor, b: Tensor) -> Tensor = "ggml_add";
extern fn mul(ctx: Ctx, a: Tensor, b: Tensor) -> Tensor = "ggml_mul";
extern fn sub(ctx: Ctx, a: Tensor, b: Tensor) -> Tensor = "ggml_sub";
extern fn exp(ctx: Ctx, a: Tensor) -> Tensor = "ggml_exp";
extern fn sqrt(ctx: Ctx, a: Tensor) -> Tensor = "ggml_sqrt";
extern fn mean(ctx: Ctx, a: Tensor) -> Tensor = "ggml_mean";
extern fn scale_bias(ctx: Ctx, a: Tensor, s: f32, b: f32) -> Tensor = "ggml_scale_bias";
extern fn neg(ctx: Ctx, a: Tensor) -> Tensor = "ggml_neg";
extern fn repeat(ctx: Ctx, a: Tensor, b: Tensor) -> Tensor = "ggml_repeat";
extern fn repeat_4d(ctx: Ctx, a: Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64) -> Tensor = "ggml_repeat_4d";
extern fn l2_norm(ctx: Ctx, a: Tensor, eps: f32) -> Tensor = "ggml_l2_norm";
extern fn mul_mat(ctx: Ctx, a: Tensor, b: Tensor) -> Tensor = "ggml_mul_mat";
extern fn rms_norm(ctx: Ctx, a: Tensor, eps: f32) -> Tensor = "ggml_rms_norm";
extern fn norm(ctx: Ctx, a: Tensor, eps: f32) -> Tensor = "ggml_norm";
extern fn tanh(ctx: Ctx, a: Tensor) -> Tensor = "ggml_tanh";
extern fn sigmoid(ctx: Ctx, a: Tensor) -> Tensor = "ggml_sigmoid";
extern fn relu(ctx: Ctx, a: Tensor) -> Tensor = "ggml_relu";
extern fn sqr(ctx: Ctx, a: Tensor) -> Tensor = "ggml_sqr";
extern fn silu(ctx: Ctx, a: Tensor) -> Tensor = "ggml_silu";
extern fn softplus(ctx: Ctx, a: Tensor) -> Tensor = "ggml_softplus";
extern fn gelu(ctx: Ctx, a: Tensor) -> Tensor = "ggml_gelu";
extern fn gelu_erf(ctx: Ctx, a: Tensor) -> Tensor = "ggml_gelu_erf";
extern fn geglu_split(ctx: Ctx, a: Tensor, b: Tensor) -> Tensor = "ggml_geglu_split";
extern fn swiglu(ctx: Ctx, a: Tensor) -> Tensor = "ggml_swiglu";
extern fn swiglu_split(ctx: Ctx, a: Tensor, b: Tensor) -> Tensor = "ggml_swiglu_split";
extern fn soft_max(ctx: Ctx, a: Tensor) -> Tensor = "ggml_soft_max";
extern fn soft_max_ext(ctx: Ctx, a: Tensor, mask: Tensor, scale: f32, max_bias: f32) -> Tensor = "ggml_soft_max_ext";
extern fn reshape_1d(ctx: Ctx, a: Tensor, ne0: i64) -> Tensor = "ggml_reshape_1d";
extern fn argsort_top_k(ctx: Ctx, a: Tensor, k: i32) -> Tensor = "ggml_argsort_top_k";
extern fn sum_rows(ctx: Ctx, a: Tensor) -> Tensor = "ggml_sum_rows";
extern fn clamp(ctx: Ctx, a: Tensor, min: f32, max: f32) -> Tensor = "ggml_clamp";
extern fn div(ctx: Ctx, a: Tensor, b: Tensor) -> Tensor = "ggml_div";
extern fn mul_mat_id(ctx: Ctx, as_: Tensor, b: Tensor, ids: Tensor) -> Tensor = "ggml_mul_mat_id";
extern fn reglu_split(ctx: Ctx, a: Tensor, b: Tensor) -> Tensor = "ggml_reglu_split";
extern fn cont(ctx: Ctx, a: Tensor) -> Tensor = "ggml_cont";
extern fn cast(ctx: Ctx, a: Tensor, typ: i32) -> Tensor = "ggml_cast";
extern fn reshape_2d(ctx: Ctx, a: Tensor, ne0: i64, ne1: i64) -> Tensor = "ggml_reshape_2d";
extern fn reshape_3d(ctx: Ctx, a: Tensor, ne0: i64, ne1: i64, ne2: i64) -> Tensor = "ggml_reshape_3d";
extern fn reshape_4d(ctx: Ctx, a: Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64) -> Tensor = "ggml_reshape_4d";
extern fn cont_2d(ctx: Ctx, a: Tensor, ne0: i64, ne1: i64) -> Tensor = "ggml_cont_2d";
extern fn cont_3d(ctx: Ctx, a: Tensor, ne0: i64, ne1: i64, ne2: i64) -> Tensor = "ggml_cont_3d";
extern fn cont_4d(ctx: Ctx, a: Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64) -> Tensor = "ggml_cont_4d";
extern fn view_2d(ctx: Ctx, a: Tensor, ne0: i64, ne1: i64, nb1: i64, offset: i64) -> Tensor = "ggml_view_2d";
extern fn view_4d(ctx: Ctx, a: Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64,
                  nb1: i64, nb2: i64, nb3: i64, offset: i64) -> Tensor = "ggml_view_4d";
extern fn view_3d(ctx: Ctx, a: Tensor, ne0: i64, ne1: i64, ne2: i64, nb1: i64, nb2: i64, offset: i64) -> Tensor = "ggml_view_3d";
extern fn transpose(ctx: Ctx, a: Tensor) -> Tensor = "ggml_transpose";
extern fn pad_ext(ctx: Ctx, a: Tensor, p0: i32, p1: i32, p2: i32, p3: i32) -> Tensor = "ggml_pad";
extern fn concat(ctx: Ctx, a: Tensor, b: Tensor, dim: i32) -> Tensor = "ggml_concat";
extern fn cpy(ctx: Ctx, a: Tensor, b: Tensor) -> Tensor = "ggml_cpy";
extern fn ssm_conv(ctx: Ctx, sx: Tensor, c: Tensor) -> Tensor = "ggml_ssm_conv";
// the output: y [d_inner, n_seq_tokens, n_seqs], then K states
extern fn ssm_scan(ctx: Ctx, s: Tensor, x: Tensor, dt: Tensor, a: Tensor, b: Tensor, c: Tensor,
                   ids: Tensor, n_states: i64) -> Tensor = "ggml_ssm_scan";
extern fn view_1d(ctx: Ctx, a: Tensor, ne0: i64, offset: i64) -> Tensor = "ggml_view_1d";
extern fn element_size(t: Tensor) -> i64 = "ggml_element_size";
extern fn permute(ctx: Ctx, a: Tensor, ax0: i32, ax1: i32, ax2: i32, ax3: i32) -> Tensor = "ggml_permute";
extern fn rope_ext(ctx: Ctx, a: Tensor, pos: Tensor, freqs: Tensor, n_dims: i32, mode: i32,
                   n_ctx_orig: i32, freq_base: f32, freq_scale: f32, ext_factor: f32,
                   attn_factor: f32, beta_fast: f32, beta_slow: f32) -> Tensor = "ggml_rope_ext";
// rope only dimensions n_offs.. of a rope node
extern fn rope_set_offset(a: Tensor, n_offs: i32) -> Tensor = "ggml_rope_set_offset";
extern fn flash_attn_ext(ctx: Ctx, q: Tensor, k: Tensor, v: Tensor, mask: Tensor,
                         scale: f32, max_bias: f32, logit_softcap: f32) -> Tensor = "ggml_flash_attn_ext";
// flash_attn_ext where each slice of q attends to its own list of k and v rows (kv_rows: [n_kv, n_slices], -1 for padding)
extern fn flash_attn_ext_rows(ctx: Ctx, q: Tensor, k: Tensor, v: Tensor, mask: Tensor, kv_rows: Tensor,
                              scale: f32, max_bias: f32, logit_softcap: f32) -> Tensor = "ggml_flash_attn_ext_rows";
// the output: the heads' values [S_v, H_v, n_tokens, n_seqs], then K states
extern fn gated_delta_net(ctx: Ctx, q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
                          state: Tensor, n_states: i64) -> Tensor = "ggml_gated_delta_net";
// RWKV's attentions: the output, then the new state
extern fn rwkv_wkv6(ctx: Ctx, k: Tensor, v: Tensor, r: Tensor, tf: Tensor, td: Tensor, state: Tensor) -> Tensor = "ggml_rwkv_wkv6";
extern fn rwkv_wkv7(ctx: Ctx, r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor, state: Tensor) -> Tensor = "ggml_rwkv_wkv7";
extern fn gated_linear_attn(ctx: Ctx, k: Tensor, v: Tensor, q: Tensor, g: Tensor, state: Tensor,
                            scale: f32) -> Tensor = "ggml_gated_linear_attn";
extern fn prec_set_acc(a: Tensor, prec: i32) -> bool = "ggml_prec_set_acc";

// graphs
extern fn new_graph_custom(ctx: Ctx, size: i64, grads: bool) -> Graph = "ggml_new_graph_custom";
extern fn new_graph(ctx: Ctx) -> Graph = "ggml_new_graph";
extern fn build_forward_expand(g: Graph, t: Tensor) = "ggml_build_forward_expand";
extern fn graph_n_nodes(g: Graph) -> i32 = "ggml_graph_n_nodes";

extern fn backend_free(b: Backend) = "ggml_backend_free";
extern fn backend_alloc_ctx_tensors(ctx: Ctx, b: Backend) -> Buffer = "ggml_backend_alloc_ctx_tensors";
extern fn backend_buffer_clear(buf: Buffer, value: u8) = "ggml_backend_buffer_clear";
extern fn backend_buffer_free(buf: Buffer) = "ggml_backend_buffer_free";
extern fn backend_graph_compute(b: Backend, g: Graph) -> i32 = "ggml_backend_graph_compute";
extern fn gallocr_free(g: Gallocr) = "ggml_gallocr_free";
extern fn gallocr_alloc_graph(g: Gallocr, graph: Graph) -> bool = "ggml_gallocr_alloc_graph";


