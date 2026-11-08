// The bindings whose C is shim.c (see ggml_shim.build), in namespace ggml
// with the rest; the gguf ones are in gguf_shim.goose.
import ggml from "..";
import .gguf_shim;
namespace ggml;

// contexts
extern fn init(mem_size: i64, no_alloc: bool) -> Ctx = "lg_init";
extern fn get_tensor(ctx: Ctx, name: const u8[:]) -> Tensor = "lg_get_tensor";
extern fn tensor_name(t: Tensor, out: u8[>..]&) = "lg_tensor_name";
extern fn set_name(t: Tensor, name: const u8[:]) = "lg_set_name";
extern fn ne(t: Tensor, i: i32) -> i64 = "lg_tensor_ne";
extern fn nb(t: Tensor, i: i32) -> i64 = "lg_tensor_nb";
extern fn tensor_type(t: Tensor) -> i32 = "lg_tensor_type";
extern fn allocated(t: Tensor) -> bool = "lg_tensor_allocated";
extern fn gallocr_new(b: Backend) -> Gallocr = "lg_gallocr_new";
// ggml_rope_multi; sections: four numbers
extern fn rope_multi(ctx: Ctx, a: Tensor, pos: Tensor, freqs: Tensor, n_dims: i32, sections: const i32[:],
                     mode: i32, n_ctx_orig: i32, freq_base: f32, freq_scale: f32, ext_factor: f32,
                     attn_factor: f32, beta_fast: f32, beta_slow: f32) -> Tensor = "lg_rope_multi";

// tensor data
extern fn write_f32(t: Tensor, src: const f32[:]) = "lg_tensor_write_f32";
extern fn read_f32(t: Tensor, dst: f32[:]) = "lg_tensor_read_f32";
extern fn backend_set_i32(t: Tensor, src: const i32[:]) = "lg_backend_tensor_set_i32";
extern fn backend_set_i64(t: Tensor, src: const i64[:]) = "lg_backend_tensor_set_i64";
extern fn backend_set_f32(t: Tensor, src: const f32[:]) = "lg_backend_tensor_set_f32";
extern fn backend_set_f16(t: Tensor, src: const f32[:]) = "lg_backend_tensor_set_f16";
// dst.len floats from element `offset` on.
extern fn backend_get_f32(t: Tensor, dst: f32[:], offset: i64) = "lg_backend_tensor_get_f32";
