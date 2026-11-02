// Goose bindings for gguf, ggml's model file format: metadata and tensor index.
import .ggml;
namespace gguf;

struct Ctx { p: i64 }

// enum gguf_type
let UINT8 = 0;
let INT8 = 1;
let UINT16 = 2;
let INT16 = 3;
let UINT32 = 4;
let INT32 = 5;
let FLOAT32 = 6;
let BOOL = 7;
let STRING = 8;
let ARRAY = 9;
let UINT64 = 10;
let INT64 = 11;
let FLOAT64 = 12;

// Opens a GGUF file's metadata; p == 0 if it cannot be read.
extern "lg_gguf_init_meta" fn init_from_file(path: const u8[:]) -> Ctx;
// The same, and also makes its tensors in a new ggml context, written to ctx
// (no_alloc: without their data).
extern "lg_gguf_init" fn init_with_tensors(path: const u8[:], no_alloc: bool, ctx: ggml::Ctx&) -> Ctx;
extern "gguf_free" fn free(ctx: Ctx);

extern "gguf_get_version" fn get_version(ctx: Ctx) -> u32;
extern "gguf_get_alignment" fn get_alignment(ctx: Ctx) -> i64;
extern "gguf_get_data_offset" fn get_data_offset(ctx: Ctx) -> i64;

extern "gguf_get_n_kv" fn get_n_kv(ctx: Ctx) -> i64;
extern "lg_gguf_find_key" fn find_key(ctx: Ctx, key: const u8[:]) -> i64;   // -1 if absent
extern "lg_gguf_get_key" fn get_key(ctx: Ctx, id: i64, out: u8[>..]&);
extern "gguf_get_kv_type" fn get_kv_type(ctx: Ctx, id: i64) -> i32;
extern "gguf_get_arr_type" fn get_arr_type(ctx: Ctx, id: i64) -> i32;

extern "gguf_get_val_u8" fn get_val_u8(ctx: Ctx, id: i64) -> u8;
extern "gguf_get_val_i8" fn get_val_i8(ctx: Ctx, id: i64) -> i8;
extern "gguf_get_val_u16" fn get_val_u16(ctx: Ctx, id: i64) -> u16;
extern "gguf_get_val_i16" fn get_val_i16(ctx: Ctx, id: i64) -> i16;
extern "gguf_get_val_u32" fn get_val_u32(ctx: Ctx, id: i64) -> u32;
extern "gguf_get_val_i32" fn get_val_i32(ctx: Ctx, id: i64) -> i32;
extern "gguf_get_val_f32" fn get_val_f32(ctx: Ctx, id: i64) -> f32;
extern "gguf_get_val_u64" fn get_val_u64(ctx: Ctx, id: i64) -> u64;
extern "gguf_get_val_i64" fn get_val_i64(ctx: Ctx, id: i64) -> i64;
extern "gguf_get_val_f64" fn get_val_f64(ctx: Ctx, id: i64) -> f64;
extern "gguf_get_val_bool" fn get_val_bool(ctx: Ctx, id: i64) -> bool;
extern "lg_gguf_get_val_str" fn get_val_str(ctx: Ctx, id: i64, out: u8[>..]&);

extern "gguf_get_arr_n" fn get_arr_n(ctx: Ctx, id: i64) -> i64;
extern "lg_gguf_get_arr_str" fn get_arr_str(ctx: Ctx, id: i64, i: i64, out: u8[>..]&);
// The first dst.len elements of an array value.
extern "lg_gguf_get_arr_i32" fn get_arr_i32(ctx: Ctx, id: i64, dst: i32[:]);
extern "lg_gguf_get_arr_f32" fn get_arr_f32(ctx: Ctx, id: i64, dst: f32[:]);
extern "lg_gguf_get_arr_u8" fn get_arr_u8(ctx: Ctx, id: i64, dst: u8[:]);

extern "gguf_get_n_tensors" fn get_n_tensors(ctx: Ctx) -> i64;
extern "lg_gguf_find_tensor" fn find_tensor(ctx: Ctx, name: const u8[:]) -> i64;
extern "gguf_get_tensor_offset" fn get_tensor_offset(ctx: Ctx, id: i64) -> i64;
extern "lg_gguf_get_tensor_name" fn get_tensor_name(ctx: Ctx, id: i64, out: u8[>..]&);
extern "gguf_get_tensor_type" fn get_tensor_type(ctx: Ctx, id: i64) -> i32;
extern "gguf_get_tensor_size" fn get_tensor_size(ctx: Ctx, id: i64) -> i64;
