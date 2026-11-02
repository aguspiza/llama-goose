// The gguf bindings whose C is shim.c.
import ggml from "..";
import ggml.gguf;
namespace gguf;

// Opens a GGUF file's metadata; p == 0 if it cannot be read.
extern "lg_gguf_init_meta" fn init_from_file(path: const u8[:]) -> Ctx;
// The same, and also makes its tensors in a new ggml context, written to ctx
// (no_alloc: without their data).
extern "lg_gguf_init" fn init_with_tensors(path: const u8[:], no_alloc: bool, ctx: ggml::Ctx&) -> Ctx;
extern "lg_gguf_find_key" fn find_key(ctx: Ctx, key: const u8[:]) -> i64;   // -1 if absent
extern "lg_gguf_get_key" fn get_key(ctx: Ctx, id: i64, out: u8[>..]&);
extern "lg_gguf_get_val_str" fn get_val_str(ctx: Ctx, id: i64, out: u8[>..]&);
extern "lg_gguf_get_arr_str" fn get_arr_str(ctx: Ctx, id: i64, i: i64, out: u8[>..]&);
// The first dst.len elements of an array value.
extern "lg_gguf_get_arr_i32" fn get_arr_i32(ctx: Ctx, id: i64, dst: i32[:]);
extern "lg_gguf_get_arr_f32" fn get_arr_f32(ctx: Ctx, id: i64, dst: f32[:]);
extern "lg_gguf_get_arr_u8" fn get_arr_u8(ctx: Ctx, id: i64, dst: u8[:]);
extern "lg_gguf_find_tensor" fn find_tensor(ctx: Ctx, name: const u8[:]) -> i64;
extern "lg_gguf_get_tensor_name" fn get_tensor_name(ctx: Ctx, id: i64, out: u8[>..]&);
// Reads every tensor of ctx (made by init_with_tensors and allocated in a
// backend buffer) from the file; false if it cannot be read.
extern "lg_load_tensor_data" fn load_tensor_data(meta: Ctx, ctx: ggml::Ctx, path: const u8[:]) -> bool;
