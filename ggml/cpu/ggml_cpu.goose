// ggml's CPU backend (see ggml_cpu.build), and the bindings for its own
// functions, in namespace ggml with the rest.
import ggml_shim from "../shim";
import ggml from "..";
namespace ggml;

extern fn graph_compute_with_ctx(ctx: Ctx, g: Graph, n_threads: i32) -> i32 = "ggml_graph_compute_with_ctx";
// backends and buffers
extern fn backend_cpu_init() -> Backend = "ggml_backend_cpu_init";
extern fn backend_cpu_set_n_threads(b: Backend, n: i32) = "ggml_backend_cpu_set_n_threads";

// Allocates the tensors of ctx as llama.cpp does: the ones in matmul (read
// by mul_mat) and matmul_id (experts read by mul_mat_id, n_ids per token) in
// the CPU's extra buffer types (CPU_REPACK) where they fit, the rest in plain
// CPU memory. bufs gets the buffers, ending with p == 0.
extern fn alloc_weights(ctx: Ctx, matmul: const Tensor[:], matmul_id: const Tensor[:], n_ids: i64,
                        bufs: Buffer[:]) -> bool = "lg_alloc_weights";
extern fn default_threads() -> i32 = "lg_default_threads";
// The backend's threads kept between graphs, not started at each one.
extern fn use_threadpool(b: Backend, n_threads: i32) = "lg_use_threadpool";
extern fn set_quiet(q: i32) = "lg_set_quiet";
