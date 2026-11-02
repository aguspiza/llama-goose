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

//
// backends, buffers, and tensor data
//
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <stdio.h>

#ifdef _WIN32
#define lg_fseek _fseeki64
#else
#define lg_fseek fseeko
#endif

struct ggml_tensor * lg_get_tensor(struct ggml_context * ctx, lg_bytes name) {
    return ggml_get_tensor(ctx, lg_cstr(name));
}

void lg_tensor_name(const struct ggml_tensor * t, gs_rref out) { lg_append_cstr(out, t->name); }
void lg_set_name(struct ggml_tensor * t, lg_bytes name) { ggml_set_name(t, lg_cstr(name)); }
int64_t lg_tensor_nb(const struct ggml_tensor * t, int32_t i) { return (int64_t) t->nb[i]; }

// Reads the data of every tensor in ctx from a GGUF file: at the file's data
// offset plus the tensor's own. False if the file cannot be read.
bool lg_load_tensor_data(const struct gguf_context * meta, struct ggml_context * ctx, lg_bytes path) {
    FILE * f = fopen(lg_cstr(path), "rb");
    if (!f) {
        return false;
    }
    const size_t data_offset = gguf_get_data_offset(meta);
    bool ok = true;
    void * tmp = NULL;
    size_t tmp_size = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t && ok; t = ggml_get_next_tensor(ctx, t)) {
        const int64_t id = gguf_find_tensor(meta, ggml_get_name(t));
        if (id < 0) {
            continue;
        }
        const size_t n = ggml_nbytes(t);
        if (n > tmp_size) {
            tmp = realloc(tmp, n);
            tmp_size = n;
        }
        ok = lg_fseek(f, (long long) (data_offset + gguf_get_tensor_offset(meta, id)), SEEK_SET) == 0 &&
             fread(tmp, 1, n, f) == n;
        if (ok) {
            ggml_backend_tensor_set(t, tmp, 0, n);
        }
    }
    free(tmp);
    fclose(f);
    return ok;
}

void lg_backend_tensor_set_i32(struct ggml_tensor * t, lg_i32s src) {
    ggml_backend_tensor_set(t, src.data, 0, (size_t) src.len * sizeof(int32_t));
}

typedef struct { int64_t * data; int64_t len; } lg_i64s;
void lg_backend_tensor_set_i64(struct ggml_tensor * t, lg_i64s src) {
    ggml_backend_tensor_set(t, src.data, 0, (size_t) src.len * sizeof(int64_t));
}

void lg_backend_tensor_set_f32(struct ggml_tensor * t, lg_f32s src) {
    ggml_backend_tensor_set(t, src.data, 0, (size_t) src.len * sizeof(float));
}

// f32 values into an f16 tensor.
void lg_backend_tensor_set_f16(struct ggml_tensor * t, lg_f32s src) {
    ggml_fp16_t * tmp = (ggml_fp16_t *) malloc((size_t) src.len * sizeof(ggml_fp16_t));
    ggml_fp32_to_fp16_row(src.data, tmp, src.len);
    ggml_backend_tensor_set(t, tmp, 0, (size_t) src.len * sizeof(ggml_fp16_t));
    free(tmp);
}

void lg_backend_tensor_get_f32(const struct ggml_tensor * t, lg_f32s dst, int64_t offset) {
    ggml_backend_tensor_get(t, dst.data, (size_t) offset * sizeof(float), (size_t) dst.len * sizeof(float));
}

ggml_gallocr_t lg_gallocr_new(ggml_backend_t backend) {
    return ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
}

//
// weights in the buffer types llama.cpp picks
//

// Whether a weight in buft can be read by mul_mat (llama.cpp's weight_buft_supported).
static bool lg_mul_mat_supported(struct ggml_tensor * w, ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev) {
    struct ggml_init_params p = { ggml_tensor_overhead() * 8, NULL, true };
    struct ggml_context * c = ggml_init(p);
    struct ggml_tensor * b = ggml_new_tensor_4d(c, GGML_TYPE_F32, w->ne[0], 512, w->ne[2], w->ne[3]);
    struct ggml_tensor * op = ggml_mul_mat(c, w, b);
    w->buffer = ggml_backend_buft_alloc_buffer(buft, 0);
    const bool ok = ggml_backend_dev_supports_op(dev, op);
    ggml_backend_buffer_free(w->buffer);
    w->buffer = NULL;
    ggml_free(c);
    return ok;
}

typedef struct { struct ggml_tensor ** data; int64_t len; } lg_tensors;
typedef ggml_backend_buffer_type_t * (*lg_get_extra_bufts_t)(ggml_backend_dev_t);

#define LG_MAX_BUFTS 8

// Allocates the tensors of ctx: each one in `matmul` (read by mul_mat) in the
// first of the CPU's extra buffer types (CPU_REPACK) that takes it, the rest
// in the CPU buffer type, as llama.cpp does. Returns the buffers, null-ended.
typedef struct { ggml_backend_buffer_t * data; int64_t len; } lg_buffers;
bool lg_alloc_weights(struct ggml_context * ctx, lg_tensors matmul, lg_buffers out) {
    ggml_backend_buffer_t * bufs = out.data;
    const int64_t n_bufs = out.len;
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    lg_get_extra_bufts_t get_extra = (lg_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");

    ggml_backend_buffer_type_t bufts[LG_MAX_BUFTS];
    int n = 0;
    if (get_extra) {
        for (ggml_backend_buffer_type_t * e = get_extra(dev); e && *e && n < LG_MAX_BUFTS - 1; e++) {
            bufts[n++] = *e;
        }
    }
    const int cpu = n;
    bufts[n++] = ggml_backend_dev_buffer_type(dev);
    if (n_bufs < n + 1) {
        return false;
    }

    // the buffer type of each tensor, in ctx order
    int64_t n_tensors = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        n_tensors++;
    }
    int * pick = (int *) malloc((size_t) n_tensors * sizeof(int));
    size_t size[LG_MAX_BUFTS] = {0};
    int64_t i = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t), i++) {
        pick[i] = cpu;
        for (int64_t k = 0; k < matmul.len; k++) {
            if (matmul.data[k] != t) {
                continue;
            }
            for (int b = 0; b < cpu; b++) {
                if (lg_mul_mat_supported(t, bufts[b], dev)) {
                    pick[i] = b;
                    break;
                }
            }
            break;
        }
        const size_t align = ggml_backend_buft_get_alignment(bufts[pick[i]]);
        size[pick[i]] += GGML_PAD(ggml_backend_buft_get_alloc_size(bufts[pick[i]], t), align);
    }

    struct ggml_tallocr talloc[LG_MAX_BUFTS];
    int64_t nb = 0;
    for (int b = 0; b < n; b++) {
        if (size[b] == 0) {
            continue;
        }
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(bufts[b], size[b] + ggml_backend_buft_get_alignment(bufts[b]));
        if (!buf) {
            free(pick);
            return false;
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        fprintf(stderr, "load_tensors: %10s model buffer size = %8.2f MiB\n", ggml_backend_buft_name(bufts[b]), size[b] / 1024.0 / 1024.0);
        bufs[nb++] = buf;
        talloc[b] = ggml_tallocr_new(buf);
    }
    bufs[nb] = NULL;

    i = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t), i++) {
        ggml_tallocr_alloc(&talloc[pick[i]], t);
    }
    free(pick);
    return true;
}
