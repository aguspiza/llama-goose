// The C behind ggml_cpu.goose's own bindings: what needs the CPU backend
// and the backend registry.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include "load.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define lg_fseek _fseeki64
#else
#define lg_fseek fseeko
#endif

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

// The same for an experts' weight read by mul_mat_id with n_ids experts per token.
static bool lg_mul_mat_id_supported(struct ggml_tensor * w, ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev, int64_t n_ids) {
    struct ggml_init_params p = { ggml_tensor_overhead() * 8, NULL, true };
    struct ggml_context * c = ggml_init(p);
    struct ggml_tensor * b = ggml_new_tensor_3d(c, GGML_TYPE_F32, w->ne[0], n_ids, 512);
    struct ggml_tensor * ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, n_ids, 512);
    struct ggml_tensor * op = ggml_mul_mat_id(c, w, b, ids);
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

typedef struct { ggml_backend_buffer_t * data; int64_t len; } lg_buffers;
typedef struct { void * data; int64_t len; } lg_bytes;

// llama_load_mode's values
enum { LG_LOAD_NONE = 0, LG_LOAD_MMAP = 1, LG_LOAD_MLOCK = 2, LG_LOAD_MMAP_MLOCK = 3, LG_LOAD_DIO = 4 };

// [first, last) of the file
typedef struct { size_t first, last; } lg_range;

static int lg_range_cmp(const void * a, const void * b) {
    const lg_range * x = (const lg_range *) a;
    const lg_range * y = (const lg_range *) b;
    return x->first < y->first ? -1 : x->first > y->first;
}

// merge_ranges: sorted, the overlapping and touching ones joined; their count
static int64_t lg_merge(lg_range * r, int64_t n) {
    qsort(r, (size_t) n, sizeof *r, lg_range_cmp);
    int64_t m = 0;
    for (int64_t i = 0; i < n; i++) {
        if (m > 0 && r[i].first <= r[m - 1].last) {
            if (r[i].last > r[m - 1].last) { r[m - 1].last = r[i].last; }
        } else {
            r[m++] = r[i];
        }
    }
    return m;
}

// sub_ranges: a less sub (both merged), into out; their count
static int64_t lg_sub(const lg_range * a, int64_t na, const lg_range * sub, int64_t ns, lg_range * out) {
    int64_t n = 0;
    for (int64_t i = 0; i < na; i++) {
        size_t cur = a[i].first;
        for (int64_t k = 0; k < ns; k++) {
            if (sub[k].last <= cur || sub[k].first >= a[i].last) {
                continue;
            }
            if (sub[k].first > cur) { out[n++] = (lg_range) { cur, sub[k].first }; }
            if (sub[k].last > cur) { cur = sub[k].last; }
        }
        if (cur < a[i].last) { out[n++] = (lg_range) { cur, a[i].last }; }
    }
    return n;
}

static int lg_buft_name_cmp(const void * a, const void * b) {
    return strcmp(ggml_backend_buft_name(*(ggml_backend_buffer_type_t const *) a),
                  ggml_backend_buft_name(*(ggml_backend_buffer_type_t const *) b));
}

// The model's mapping, kept while the process runs.
static void * lg_map;
static size_t lg_map_size;

// a tensor's group when it is the lazy one, and when no one reads it
#define LG_LAZY (-1)
#define LG_UNUSED (-2)

// get_mapping_ranges: buffers over the runs of the mapping that hold the
// tensors of group g (group[i] for the i-th of ctx) and none of the
// others', appended to mapped, with their names and sizes to show.
static bool lg_map_group(struct ggml_context * ctx, const size_t * offs, const int * group, int g,
                         ggml_backend_buffer_t * mapped, int64_t * n_mapped,
                         const char ** shown_name, size_t * shown_size, int64_t * n_shown) {
    int64_t n_tensors = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        n_tensors++;
    }
    lg_range * own = (lg_range *) malloc((size_t) (n_tensors + 1) * sizeof(lg_range));
    lg_range * other = (lg_range *) malloc((size_t) (n_tensors + 1) * sizeof(lg_range));
    lg_range * runs = (lg_range *) malloc((size_t) (2 * n_tensors + 1) * sizeof(lg_range));
    int64_t n_own = 0, n_other = 0;
    int64_t i = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t), i++) {
        const lg_range r = { offs[i], offs[i] + ggml_nbytes(t) };
        if (group[i] == g) { own[n_own++] = r; } else if (group[i] != LG_UNUSED) { other[n_other++] = r; }
    }
    n_own = lg_merge(own, n_own);
    n_other = lg_merge(other, n_other);
    const int64_t n_runs = lg_sub(own, n_own, other, n_other, runs);
    bool ok = true;
    for (int64_t r = 0; r < n_runs && ok; r++) {
        ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr((char *) lg_map + runs[r].first, runs[r].last - runs[r].first);
        ok = buf != NULL;
        if (ok) {
            ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            mapped[(*n_mapped)++] = buf;
            shown_name[*n_shown] = ggml_backend_buffer_name(buf);
            shown_size[(*n_shown)++] = runs[r].last - runs[r].first;
        }
    }
    free(own);
    free(other);
    free(runs);
    return ok;
}

// A tensor given its data in the mapping, in the mapped buffer that holds it.
static void lg_place(struct ggml_tensor * t, uint8_t * data, ggml_backend_buffer_t * mapped, int64_t n_mapped) {
    for (int64_t k = 0; k < n_mapped; k++) {
        const uint8_t * base = (const uint8_t *) ggml_backend_buffer_get_base(mapped[k]);
        if (base <= data && data + ggml_nbytes(t) <= base + ggml_backend_buffer_get_size(mapped[k])) {
            ggml_backend_tensor_alloc(mapped[k], t, data);
            return;
        }
    }
}

typedef struct { int32_t * data; int64_t len; } lg_i32s;

// llama_prefetch_rows: the pages of rows of a matrix in host memory read ahead.
void lg_prefetch_rows(const struct ggml_tensor * t, lg_i32s rows) {
    if (!t || !t->data || !t->buffer || !ggml_backend_buffer_is_host(t->buffer) || rows.len == 0) {
        return;
    }
    const size_t row_bytes = ggml_row_size(t->type, t->ne[0]);
    const void ** addr = (const void **) malloc((size_t) rows.len * sizeof(void *));
    size_t * size = (size_t *) malloc((size_t) rows.len * sizeof(size_t));
    for (int64_t i = 0; i < rows.len; i++) {
        addr[i] = (const char *) t->data + (size_t) rows.data[i] * t->nb[1];
        size[i] = row_bytes;
    }
    lg_prefetch(addr, size, (size_t) rows.len);
    free(addr);
    free(size);
}

// Places the tensors of ctx as llama.cpp does, and reads them from path,
// whose GGUF meta is: each tensor in `matmul` (read by mul_mat) or in
// `matmul_id` (experts read by mul_mat_id, n_ids per token) in the first of
// the CPU's extra buffer types (CPU_REPACK) that takes it, when repack; the
// rest in the CPU buffer type. With mode mmap (and mmap+mlock) those are
// buffers over the file mapped in memory, one for each run of the file that
// holds them; the others are allocated and their data copied. mlock locks
// the weights in RAM; dio reads with direct I/O. warn_overrides: tensors
// were overridden to the CPU, which mmap makes slower, llama.cpp warns.
// tied: the token embeddings when they are the output too (or NULL);
// output gets the output's tensor, a copy of them where mul_mat takes
// another buffer type than get_rows. lazy: a tensor read from the mapping
// as it is used (or NULL), mapped whatever the mode, neither read ahead
// nor locked. unused: tensors no one reads, neither placed nor read, with
// llama.cpp's warning. out gets the allocated
// buffers, null-ended; host_bytes what the CPU buffer type would allocate
// (llama.cpp's host memory whatever the mode, for --fit).
bool lg_load_weights(const struct gguf_context * meta, struct ggml_context * ctx, lg_bytes path_b,
                     lg_tensors matmul, lg_tensors matmul_id, int64_t n_ids, int32_t mode, bool repack,
                     bool warn_overrides, struct ggml_tensor * tied, struct ggml_tensor * lazy, lg_tensors unused,
                     lg_buffers out, int64_t * host_bytes, struct ggml_tensor ** output) {
    ggml_backend_buffer_t * bufs = out.data;
    const int64_t n_bufs = out.len;
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    lg_get_extra_bufts_t get_extra = (lg_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");

    ggml_backend_buffer_type_t bufts[LG_MAX_BUFTS];
    int n = 0;
    if (get_extra && repack) {
        for (ggml_backend_buffer_type_t * e = get_extra(dev); e && *e && n < LG_MAX_BUFTS - 1; e++) {
            bufts[n++] = *e;
        }
    }
    const int cpu = n;
    bufts[n++] = ggml_backend_dev_buffer_type(dev);
    if (n_bufs < n + 1) {
        return false;
    }
    const bool use_mmap = mode == LG_LOAD_MMAP || mode == LG_LOAD_MMAP_MLOCK;
    const bool use_mlock = mode == LG_LOAD_MLOCK || mode == LG_LOAD_MMAP_MLOCK;

    // the buffer type of each tensor and where its data is in the file, in ctx order
    int64_t n_tensors = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        n_tensors++;
    }
    int * pick = (int *) malloc((size_t) n_tensors * sizeof(int));
    size_t * offs = (size_t *) malloc((size_t) n_tensors * sizeof(size_t));
    const size_t data_offset = gguf_get_data_offset(meta);
    size_t size[LG_MAX_BUFTS] = {0};
    // the lazy tensor, read from the mapping as it is used (in CPU memory,
    // a group of its own: llama.cpp's lazy context)
    int64_t lazy_i = -1;
    size_t lazy_size = 0;
    int64_t i = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t), i++) {
        pick[i] = cpu;
        offs[i] = data_offset + gguf_get_tensor_offset(meta, gguf_find_tensor(meta, ggml_get_name(t)));
        bool skip = false;
        for (int64_t k = 0; k < unused.len; k++) {
            skip = skip || unused.data[k] == t;
        }
        if (skip) {
            pick[i] = LG_UNUSED;
            continue;
        }
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
        for (int64_t k = 0; k < matmul_id.len; k++) {
            if (matmul_id.data[k] != t) {
                continue;
            }
            for (int b = 0; b < cpu; b++) {
                if (lg_mul_mat_id_supported(t, bufts[b], dev, n_ids)) {
                    pick[i] = b;
                    break;
                }
            }
            break;
        }
        const size_t align = ggml_backend_buft_get_alignment(bufts[pick[i]]);
        const size_t bytes = GGML_PAD(ggml_backend_buft_get_alloc_size(bufts[pick[i]], t), align);
        if (t == lazy) {
            lazy_i = i;
            lazy_size = bytes;
        } else {
            size[pick[i]] += bytes;
        }
    }
    // a tied output, the token embeddings that get_rows reads, read by
    // mul_mat too: a copy of its own where that takes another buffer type
    // (llama.cpp's TENSOR_DUPLICATED), kept in a context of its own
    *output = tied;
    struct ggml_tensor * dup = NULL;
    int dup_pick = cpu;
    size_t dup_offs = 0;
    if (tied) {
        for (int b = 0; b < cpu; b++) {
            if (lg_mul_mat_supported(tied, bufts[b], dev)) {
                dup_pick = b;
                break;
            }
        }
        int tied_pick = cpu;
        i = 0;
        for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t), i++) {
            if (t == tied) {
                tied_pick = pick[i];
                dup_offs = offs[i];
            }
        }
        if (dup_pick != tied_pick) {
            struct ggml_init_params p = { ggml_tensor_overhead(), NULL, true };
            dup = ggml_dup_tensor(ggml_init(p), tied);
            ggml_set_name(dup, ggml_get_name(tied));
            size[dup_pick] += GGML_PAD(ggml_backend_buft_get_alloc_size(bufts[dup_pick], dup), ggml_backend_buft_get_alignment(bufts[dup_pick]));
            *output = dup;
        }
    }
    *host_bytes = (int64_t) lazy_size;
    for (int b = 0; b < n; b++) {
        if (ggml_backend_buft_is_host(bufts[b])) {
            *host_bytes += (int64_t) size[b];
        }
    }

    char * path = (char *) malloc((size_t) path_b.len + 1);
    memcpy(path, path_b.data, (size_t) path_b.len);
    path[path_b.len] = 0;
    // llama.cpp opens the file before it loads the tensors
    size_t dio_align = 1;
    lg_dio * dio = mode == LG_LOAD_DIO ? lg_dio_open(path, &dio_align) : NULL;

    static const char * mode_names[] = { "none", "mmap", "mlock", "mmap+mlock", "dio" };
    if (!lg_quiet) fprintf(stderr, "load_tensors: loading model tensors, this can take a while... (load_mode = %s)\n", mode_names[mode]);
    i = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t && !lg_quiet; t = ggml_get_next_tensor(ctx, t), i++) {
        if (pick[i] == LG_UNUSED) {
            fprintf(stderr, "model has unused tensor %s (size = %zu bytes) -- ignoring\n", ggml_get_name(t), ggml_nbytes(t));
        }
    }
    if (warn_overrides && use_mmap && !lg_quiet) {
        fprintf(stderr, "llama_model_loader: tensor overrides to CPU are used with mmap enabled - consider using --load-mode none for better performance\n");
    }
    if (lazy_i >= 0 && !lg_quiet) {
        fprintf(stderr, "add: tensor %s (size = %zu MiB) lazy read enabled\n", ggml_get_name(lazy), ggml_nbytes(lazy) / 1024 / 1024);
        fprintf(stderr, "load_tensors: enabling prefetch for '%s'\n", ggml_get_name(lazy));
    }
    bool ok = true;
    lg_mlock * map_lock = NULL;
    // mapped with mmap, or for the lazy tensor whatever the mode
    if (use_mmap || lazy_i >= 0) {
        const size_t lazy_first = lazy_i >= 0 ? offs[lazy_i] : 0;
        const size_t lazy_last = lazy_i >= 0 ? offs[lazy_i] + ggml_nbytes(lazy) : 0;
        lg_map = lg_map_file(path, &lg_map_size, use_mmap, lazy_first, lazy_last);
        ok = lg_map != NULL;
        if (ok && use_mlock && use_mmap) {
            map_lock = (lg_mlock *) calloc(1, sizeof *map_lock);
            map_lock->addr = lg_map;
        }
    }

    // the buffers, by the name of their type (llama.cpp's ctx_map order)
    ggml_backend_buffer_type_t by_name[LG_MAX_BUFTS];
    memcpy(by_name, bufts, (size_t) n * sizeof by_name[0]);
    qsort(by_name, (size_t) n, sizeof by_name[0], lg_buft_name_cmp);
    struct ggml_tallocr talloc[LG_MAX_BUFTS];
    // the buffers over the mapping, kept; each tensor's group: its buffer type, or LG_LAZY
    ggml_backend_buffer_t * mapped = (ggml_backend_buffer_t *) malloc((size_t) (2 * n_tensors + 2) * sizeof(ggml_backend_buffer_t));
    int64_t n_mapped = 0;
    int * group = (int *) malloc((size_t) n_tensors * sizeof(int));
    for (i = 0; i < n_tensors; i++) {
        group[i] = i == lazy_i ? LG_LAZY : pick[i];
    }
    int64_t nb = 0;
    const char ** shown_name = (const char **) malloc((size_t) (n + 2 * n_tensors + 1) * sizeof(char *));
    size_t * shown_size = (size_t *) malloc((size_t) (n + 2 * n_tensors + 1) * sizeof(size_t));
    int64_t n_shown = 0;
    for (int o = 0; o < n && ok; o++) {
        int b = 0;
        while (bufts[b] != by_name[o]) { b++; }
        if (size[b] == 0) {
            continue;
        }
        if (use_mmap && b == cpu) {
            ok = lg_map_group(ctx, offs, group, cpu, mapped, &n_mapped, shown_name, shown_size, &n_shown);
            continue;
        }
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(bufts[b], size[b] + ggml_backend_buft_get_alignment(bufts[b]));
        ok = buf != NULL;
        if (!ok) {
            break;
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        // mlock: the buffer's weights locked, when it is host memory
        if (use_mlock && !use_mmap && ggml_backend_buffer_is_host(buf)) {
            lg_mlock * lock = (lg_mlock *) calloc(1, sizeof *lock);
            lock->addr = ggml_backend_buffer_get_base(buf);
            lg_mlock_grow(lock, size[b]);
        }
        shown_name[n_shown] = ggml_backend_buft_name(bufts[b]);
        shown_size[n_shown++] = size[b];
        bufs[nb++] = buf;
        talloc[b] = ggml_tallocr_new(buf);
    }
    // the lazy context comes after the others
    if (ok && lazy_i >= 0) {
        ok = lg_map_group(ctx, offs, group, LG_LAZY, mapped, &n_mapped, shown_name, shown_size, &n_shown);
    }
    bufs[nb] = NULL;
    for (int64_t k = 0; k < n_shown && !lg_quiet; k++) {
        fprintf(stderr, "load_tensors: %12s model buffer size = %8.2f MiB\n", shown_name[k], shown_size[k] / 1024.0 / 1024.0);
    }
    free(shown_name);
    free(shown_size);

    i = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t && ok; t = ggml_get_next_tensor(ctx, t), i++) {
        if (!(use_mmap && pick[i] == cpu) && i != lazy_i && pick[i] != LG_UNUSED) {
            ggml_tallocr_alloc(&talloc[pick[i]], t);
        }
    }
    if (dup && ok) {
        ggml_tallocr_alloc(&talloc[dup_pick], dup);
    }

    // the data
    if (ok && use_mmap) {
        size_t used_first = (size_t) -1;
        size_t used_last = 0;
        i = 0;
        for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t), i++) {
            const size_t nt = ggml_nbytes(t);
            uint8_t * data = (uint8_t *) lg_map + offs[i];
            if (pick[i] == LG_UNUSED) {
                continue;
            }
            if (pick[i] != cpu) {
                ggml_backend_tensor_set(t, data, 0, nt);
                continue;
            }
            lg_place(t, data, mapped, n_mapped);
            // locking a lazy tensor would fault all of it in, which is what lazy avoids
            if (map_lock && i != lazy_i) {
                lg_mlock_grow(map_lock, offs[i] + nt);
            }
            if (offs[i] < used_first) { used_first = offs[i]; }
            if (offs[i] + nt > used_last) { used_last = offs[i] + nt; }
        }
        if (dup) {
            ggml_backend_tensor_set(dup, (uint8_t *) lg_map + dup_offs, 0, ggml_nbytes(dup));
        }
        // the pages the weights in place do not use: the metadata, and those copied out
        lg_unmap_fragment(lg_map, lg_map_size, 0, used_first == (size_t) -1 ? 0 : used_first);
        if (used_last != 0) {
            lg_unmap_fragment(lg_map, lg_map_size, used_last, lg_map_size);
        }
    } else if (ok) {
        const size_t align = dio_align;
        FILE * f = dio ? NULL : fopen(path, "rb");
        ok = dio || f;
        uint8_t * tmp = NULL;
        size_t tmp_size = 0;
        // the tensors of ctx and the copy, with where their data is; the lazy one stays in the mapping
        struct ggml_tensor ** ts = (struct ggml_tensor **) malloc((size_t) (n_tensors + 1) * sizeof(struct ggml_tensor *));
        size_t * at = (size_t *) malloc((size_t) (n_tensors + 1) * sizeof(size_t));
        int64_t n_read = 0;
        i = 0;
        for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t), i++) {
            if (i == lazy_i) {
                lg_place(t, (uint8_t *) lg_map + offs[i], mapped, n_mapped);
                continue;
            }
            if (pick[i] == LG_UNUSED) {
                continue;
            }
            ts[n_read] = t;
            at[n_read++] = offs[i];
        }
        if (dup) {
            ts[n_read] = dup;
            at[n_read++] = dup_offs;
        }
        for (int64_t r = 0; r < n_read && ok; r++) {
            struct ggml_tensor * t = ts[r];
            const size_t nt = ggml_nbytes(t);
            if (dio) {
                // the aligned blocks around the tensor, and its data inside them
                const size_t start = at[r] & ~(align - 1);
                const size_t pad = at[r] - start;
                const size_t len = (pad + nt + align - 1) & ~(align - 1);
                if (len + align > tmp_size) {
                    free(tmp);
                    tmp_size = len + align;
                    tmp = (uint8_t *) malloc(tmp_size);
                }
                uint8_t * dst = (uint8_t *) GGML_PAD((uintptr_t) tmp, align);
                ok = lg_dio_read(dio, dst, start, len);
                if (ok) {
                    ggml_backend_tensor_set(t, dst + pad, 0, nt);
                }
                continue;
            }
            ok = lg_fseek(f, (long long) at[r], SEEK_SET) == 0;
            if (ok && ggml_backend_buffer_is_host(t->buffer)) {
                ok = fread(t->data, 1, nt, f) == nt;
            } else if (ok) {
                if (nt > tmp_size) {
                    free(tmp);
                    tmp_size = nt;
                    tmp = (uint8_t *) malloc(tmp_size);
                }
                ok = fread(tmp, 1, nt, f) == nt;
                if (ok) {
                    ggml_backend_tensor_set(t, tmp, 0, nt);
                }
            }
        }
        free(tmp);
        free(ts);
        free(at);
        if (f) { fclose(f); }
    }
    if (dio) { lg_dio_close(dio); }
    free(path);
    free(pick);
    free(group);
    free(offs);
    return ok;
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

// A pool of threads the backend keeps from one graph to the next, as
// llama.cpp's programs attach to their context: without one, ggml starts
// its threads and joins them at each graph, each token.
void lg_use_threadpool(ggml_backend_t backend, int n_threads) {
    static struct ggml_threadpool * pool;
    if (pool) { ggml_threadpool_free(pool); }
    struct ggml_threadpool_params params = ggml_threadpool_params_default(n_threads);
    pool = ggml_threadpool_new(&params);
    ggml_backend_cpu_set_threadpool(backend, pool);
}

//
// the memory a model and a context would take, for --fit (common/fit)
//

// The host memory, as the CPU device reports it to llama.cpp's --fit: its total.
int64_t lg_host_memory_total(void) {
    size_t free = 0, total = 0;
    ggml_backend_dev_memory(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), &free, &total);
    return (int64_t) total;
}

// The bytes ctx's tensors would take in a CPU buffer, and a buffer of no
// memory given to each of them, so that a graph that reads them can be
// measured (llama.cpp's no_alloc dummy buffers). Gives the buffer, to free.
ggml_backend_buffer_t lg_dummy_alloc(struct ggml_context * ctx, int64_t * size) {
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    *size = (int64_t) ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft);
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, 0);
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        t->buffer = buf;
    }
    return buf;
}

// The size of the buffer galloc would allocate for graph, allocating nothing.
int64_t lg_gallocr_measure(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    size_t size = 0;
    ggml_gallocr_reserve_n_size(galloc, graph, NULL, NULL, &size);
    return (int64_t) size;
}
