// The few ggml calls Goose cannot make directly: structs passed by value,
// and fields read or written through a tensor pointer.
#include "ggml.h"
#include "ggml-cpu.h"
#include <string.h>

typedef struct { void * data; int64_t len; } lg_bytes;
typedef struct { float * data; int64_t len; } lg_f32s;

struct ggml_context * lg_init(int64_t mem_size, bool no_alloc) {
    struct ggml_init_params p = { (size_t) mem_size, NULL, no_alloc };
    return ggml_init(p);
}

// Copies between a Goose array and a tensor's own data (CPU memory only).
void lg_tensor_write(struct ggml_tensor * t, lg_bytes src) {
    memcpy(t->data, src.data, (size_t) src.len);
}

void lg_tensor_read(const struct ggml_tensor * t, lg_bytes dst) {
    memcpy(dst.data, t->data, (size_t) dst.len);
}

void lg_tensor_write_f32(struct ggml_tensor * t, lg_f32s src) {
    memcpy(t->data, src.data, (size_t) src.len * sizeof(float));
}

void lg_tensor_read_f32(const struct ggml_tensor * t, lg_f32s dst) {
    memcpy(dst.data, t->data, (size_t) dst.len * sizeof(float));
}

int64_t lg_tensor_ne(const struct ggml_tensor * t, int32_t i) { return t->ne[i]; }
int32_t lg_tensor_type(const struct ggml_tensor * t) { return (int32_t) t->type; }

//
// gguf: names cross as Goose slices (not NUL-terminated), strings come back
// appended to a Goose builder.
//
#include "gguf.h"
#include "goose.h"
#include <stdlib.h>

// A NUL-terminated copy of a Goose string, valid until the next call.
static const char * lg_cstr(lg_bytes s) {
    static char * buf = NULL;
    static int64_t cap = 0;
    if (s.len + 1 > cap) {
        cap = s.len + 1;
        buf = (char *) realloc(buf, (size_t) cap);
    }
    memcpy(buf, s.data, (size_t) s.len);
    buf[s.len] = 0;
    return buf;
}

static void lg_append_cstr(gs_rref out, const char * s) {
    gs_bld_append(out, s, (int64_t) strlen(s));
}

struct gguf_context * lg_gguf_init(lg_bytes path, bool no_alloc, struct ggml_context ** ctx) {
    struct gguf_init_params p = { no_alloc, ctx };
    return gguf_init_from_file(lg_cstr(path), p);
}

struct gguf_context * lg_gguf_init_meta(lg_bytes path) {
    struct gguf_init_params p = { true, NULL };
    return gguf_init_from_file(lg_cstr(path), p);
}

int64_t lg_gguf_find_key(const struct gguf_context * ctx, lg_bytes key) {
    return gguf_find_key(ctx, lg_cstr(key));
}

int64_t lg_gguf_find_tensor(const struct gguf_context * ctx, lg_bytes name) {
    return gguf_find_tensor(ctx, lg_cstr(name));
}

void lg_gguf_get_key(const struct gguf_context * ctx, int64_t id, gs_rref out) {
    lg_append_cstr(out, gguf_get_key(ctx, id));
}

void lg_gguf_get_val_str(const struct gguf_context * ctx, int64_t id, gs_rref out) {
    lg_append_cstr(out, gguf_get_val_str(ctx, id));
}

void lg_gguf_get_arr_str(const struct gguf_context * ctx, int64_t id, int64_t i, gs_rref out) {
    lg_append_cstr(out, gguf_get_arr_str(ctx, id, (size_t) i));
}

void lg_gguf_get_tensor_name(const struct gguf_context * ctx, int64_t id, gs_rref out) {
    lg_append_cstr(out, gguf_get_tensor_name(ctx, id));
}

// Copies the first dst.len elements of an array value.
typedef struct { int32_t * data; int64_t len; } lg_i32s;
void lg_gguf_get_arr_i32(const struct gguf_context * ctx, int64_t id, lg_i32s dst) {
    memcpy(dst.data, gguf_get_arr_data(ctx, id), (size_t) dst.len * sizeof(int32_t));
}
void lg_gguf_get_arr_f32(const struct gguf_context * ctx, int64_t id, lg_f32s dst) {
    memcpy(dst.data, gguf_get_arr_data(ctx, id), (size_t) dst.len * sizeof(float));
}
void lg_gguf_get_arr_u8(const struct gguf_context * ctx, int64_t id, lg_bytes dst) {
    memcpy(dst.data, gguf_get_arr_data(ctx, id), (size_t) dst.len);
}
