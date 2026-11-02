// ggml's CPU backend (see ggml_cpu.build), and the bindings for its own
// functions, in namespace ggml with the rest.
import ggml_shim from "../shim";
import ggml from "..";
namespace ggml;

extern "ggml_graph_compute_with_ctx" fn graph_compute_with_ctx(ctx: Ctx, g: Graph, n_threads: i32) -> i32;
// backends and buffers
extern "ggml_backend_cpu_init" fn backend_cpu_init() -> Backend;
extern "ggml_backend_cpu_set_n_threads" fn backend_cpu_set_n_threads(b: Backend, n: i32);

// Allocates the tensors of ctx as llama.cpp does: the ones in matmul (read
// by mul_mat) in the CPU's extra buffer types (CPU_REPACK) where they fit,
// the rest in plain CPU memory. bufs gets the buffers, ending with p == 0.
extern "lg_alloc_weights" fn alloc_weights(ctx: Ctx, matmul: const Tensor[:], bufs: Buffer[:]) -> bool;
