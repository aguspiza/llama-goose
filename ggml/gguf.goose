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

extern fn free(ctx: Ctx) = "gguf_free";

extern fn get_version(ctx: Ctx) -> u32 = "gguf_get_version";
extern fn get_alignment(ctx: Ctx) -> i64 = "gguf_get_alignment";
extern fn get_data_offset(ctx: Ctx) -> i64 = "gguf_get_data_offset";

extern fn get_n_kv(ctx: Ctx) -> i64 = "gguf_get_n_kv";
extern fn get_kv_type(ctx: Ctx, id: i64) -> i32 = "gguf_get_kv_type";
extern fn get_arr_type(ctx: Ctx, id: i64) -> i32 = "gguf_get_arr_type";

extern fn get_val_u8(ctx: Ctx, id: i64) -> u8 = "gguf_get_val_u8";
extern fn get_val_i8(ctx: Ctx, id: i64) -> i8 = "gguf_get_val_i8";
extern fn get_val_u16(ctx: Ctx, id: i64) -> u16 = "gguf_get_val_u16";
extern fn get_val_i16(ctx: Ctx, id: i64) -> i16 = "gguf_get_val_i16";
extern fn get_val_u32(ctx: Ctx, id: i64) -> u32 = "gguf_get_val_u32";
extern fn get_val_i32(ctx: Ctx, id: i64) -> i32 = "gguf_get_val_i32";
extern fn get_val_f32(ctx: Ctx, id: i64) -> f32 = "gguf_get_val_f32";
extern fn get_val_u64(ctx: Ctx, id: i64) -> u64 = "gguf_get_val_u64";
extern fn get_val_i64(ctx: Ctx, id: i64) -> i64 = "gguf_get_val_i64";
extern fn get_val_f64(ctx: Ctx, id: i64) -> f64 = "gguf_get_val_f64";
extern fn get_val_bool(ctx: Ctx, id: i64) -> bool = "gguf_get_val_bool";

extern fn get_arr_n(ctx: Ctx, id: i64) -> i64 = "gguf_get_arr_n";

extern fn get_n_tensors(ctx: Ctx) -> i64 = "gguf_get_n_tensors";
extern fn get_tensor_offset(ctx: Ctx, id: i64) -> i64 = "gguf_get_tensor_offset";
extern fn get_tensor_type(ctx: Ctx, id: i64) -> i32 = "gguf_get_tensor_type";
extern fn get_tensor_size(ctx: Ctx, id: i64) -> i64 = "gguf_get_tensor_size";

