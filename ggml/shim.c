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
