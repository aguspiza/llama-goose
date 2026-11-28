// ggml's CPU backend (see ggml_cpu.build), and the bindings for its own
// functions, in namespace ggml with the rest.
import ggml_shim from "../shim";
import ggml from "..";
import ggml.gguf;
namespace ggml;

extern fn graph_compute_with_ctx(ctx: Ctx, g: Graph, n_threads: i32) -> i32 = "ggml_graph_compute_with_ctx";
// backends and buffers
extern fn backend_cpu_init() -> Backend = "ggml_backend_cpu_init";
extern fn backend_cpu_set_n_threads(b: Backend, n: i32) = "ggml_backend_cpu_set_n_threads";

// Places the tensors of ctx as llama.cpp does and reads them from path, the
// GGUF of meta: the ones in matmul (read by mul_mat) and matmul_id (experts
// read by mul_mat_id, n_ids per token) in the CPU's extra buffer types
// (CPU_REPACK) where they fit, when repack; the rest in plain CPU memory,
// with mode 1 or 3 (mmap, mmap+mlock) the file mapped in memory. mode is
// llama_load_mode's number (0 none, 2 mlock, 4 dio). warn_overrides: some
// tensors match a -ot override to the CPU (a warning with mmap). tied: the
// token embeddings when they are the output too (p == 0 if not); output
// gets the output, a copy of them where mul_mat takes another buffer type
// than their get_rows (llama.cpp's TENSOR_DUPLICATED). lazy: a tensor read
// from the mapping as it is used, whatever the mode (p == 0 for none).
// unused: tensors of ctx no one reads, neither placed nor read (with a
// warning). bufs gets the buffers allocated, ending with p == 0;
// host_bytes the CPU buffer type's share, as --fit counts it.
extern fn load_weights(meta: gguf::Ctx, ctx: Ctx, path: const u8[:], matmul: const Tensor[:], matmul_id: const Tensor[:],
                       n_ids: i64, mode: i32, repack: bool, warn_overrides: bool, tied: Tensor, lazy: Tensor,
                       unused: const Tensor[:], bufs: Buffer[:], host_bytes: i64&, output: Tensor&) -> bool = "lg_load_weights";
// llama_prefetch_rows: the pages of these rows of t read ahead (a lazy tensor's).
extern fn prefetch_rows(t: Tensor, rows: const i32[:]) = "lg_prefetch_rows";
extern fn default_threads() -> i32 = "lg_default_threads";
// The backend's threads kept between graphs, not started at each one.
extern fn use_threadpool(b: Backend, n_threads: i32) = "lg_use_threadpool";
extern fn set_quiet(q: i32) = "lg_set_quiet";

// For --fit: the host memory's total; the bytes ctx's tensors would take,
// given a buffer of no memory (to free) so that graphs that read them can
// be measured; and the buffer a graph would need.
extern fn host_memory_total() -> i64 = "lg_host_memory_total";
extern fn dummy_alloc(ctx: Ctx, size: i64&) -> Buffer = "lg_dummy_alloc";
extern fn gallocr_measure(g: Gallocr, graph: Graph) -> i64 = "lg_gallocr_measure";
