// The C behind ggml_cpu.goose's own bindings: what needs the CPU backend
// and the backend registry.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <stdio.h>
#include <stdlib.h>

//
// weights in the buffer types llama.cpp picks
//

// Quiet: no log messages from here or from ggml, errors only (what
// llama_log_set does with a callback that drops the rest).
static int lg_quiet = 0;

static void lg_log_errors(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (level >= GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

void lg_set_quiet(int q) {
    lg_quiet = q;
    ggml_log_set(q ? lg_log_errors : NULL, NULL);
}

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
        if (!lg_quiet) fprintf(stderr, "load_tensors: %10s model buffer size = %8.2f MiB\n", ggml_backend_buft_name(bufts[b]), size[b] / 1024.0 / 1024.0);
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

// common_cpu_get_num_math's answer on MinGW and on systems it has no special
// case for: half the logical CPUs above 4.
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
int lg_default_threads(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    int n = (int) si.dwNumberOfProcessors;
#else
    int n = (int) sysconf(_SC_NPROCESSORS_ONLN);
#endif
    return n > 0 ? (n <= 4 ? n : n / 2) : 4;
}
