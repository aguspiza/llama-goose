// The bindings whose C is shim.c (see ggml_shim.build), in namespace ggml
// with the rest; the gguf ones are in gguf_shim.goose.
import ggml from "..";
import .gguf_shim;
namespace ggml;

// contexts
extern "lg_init" fn init(mem_size: i64, no_alloc: bool) -> Ctx;
extern "lg_get_tensor" fn get_tensor(ctx: Ctx, name: const u8[:]) -> Tensor;
extern "lg_tensor_name" fn tensor_name(t: Tensor, out: u8[>..]&);
extern "lg_set_name" fn set_name(t: Tensor, name: const u8[:]);
extern "lg_tensor_ne" fn ne(t: Tensor, i: i32) -> i64;
extern "lg_tensor_nb" fn nb(t: Tensor, i: i32) -> i64;
extern "lg_tensor_type" fn tensor_type(t: Tensor) -> i32;
extern "lg_tensor_allocated" fn allocated(t: Tensor) -> bool;
extern "lg_gallocr_new" fn gallocr_new(b: Backend) -> Gallocr;
// ggml_rope_multi; sections: four numbers
extern "lg_rope_multi" fn rope_multi(ctx: Ctx, a: Tensor, pos: Tensor, freqs: Tensor, n_dims: i32, sections: const i32[:],
                                     mode: i32, n_ctx_orig: i32, freq_base: f32, freq_scale: f32, ext_factor: f32,
                                     attn_factor: f32, beta_fast: f32, beta_slow: f32) -> Tensor;

// tensor data
extern "lg_tensor_write_f32" fn write_f32(t: Tensor, src: const f32[:]);
extern "lg_tensor_read_f32" fn read_f32(t: Tensor, dst: f32[:]);
extern "lg_backend_tensor_set_i32" fn backend_set_i32(t: Tensor, src: const i32[:]);
extern "lg_backend_tensor_set_i64" fn backend_set_i64(t: Tensor, src: const i64[:]);
extern "lg_backend_tensor_set_f32" fn backend_set_f32(t: Tensor, src: const f32[:]);
extern "lg_backend_tensor_set_f16" fn backend_set_f16(t: Tensor, src: const f32[:]);
// dst.len floats from element `offset` on.
extern "lg_backend_tensor_get_f32" fn backend_get_f32(t: Tensor, dst: f32[:], offset: i64);
