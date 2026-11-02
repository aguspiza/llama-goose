// Goose bindings for ggml. ggml objects cross as one-pointer structs, so a
// tensor cannot be passed where a context is expected.
namespace ggml;

struct Ctx { p: i64 }
struct Tensor { p: i64 }
struct Graph { p: i64 }

// enum ggml_type
let TYPE_F32 = 0;
let TYPE_F16 = 1;

extern "lg_init" fn init(mem_size: i64, no_alloc: bool) -> Ctx;
extern "ggml_free" fn free(ctx: Ctx);

extern "ggml_new_tensor_1d" fn new_tensor_1d(ctx: Ctx, typ: i32, ne0: i64) -> Tensor;
extern "ggml_new_tensor_2d" fn new_tensor_2d(ctx: Ctx, typ: i32, ne0: i64, ne1: i64) -> Tensor;
extern "ggml_mul_mat" fn mul_mat(ctx: Ctx, a: Tensor, b: Tensor) -> Tensor;

extern "ggml_new_graph" fn new_graph(ctx: Ctx) -> Graph;
extern "ggml_build_forward_expand" fn build_forward_expand(g: Graph, t: Tensor);
extern "ggml_graph_compute_with_ctx" fn graph_compute_with_ctx(ctx: Ctx, g: Graph, n_threads: i32) -> i32;

extern "lg_tensor_write_f32" fn write_f32(t: Tensor, src: const f32[:]);
extern "lg_tensor_read_f32" fn read_f32(t: Tensor, dst: f32[:]);
extern "lg_tensor_ne" fn ne(t: Tensor, i: i32) -> i64;
