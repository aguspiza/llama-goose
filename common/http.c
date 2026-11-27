// The C behind common/download: HTTP(S) requests through libcurl, which is
// loaded when the first request is made (so the programs run without it
// when they download nothing), the size of a file, and a symbolic link.
// libcurl's API is declared here, with its stable option numbers, so that
// building needs no curl headers.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#define http_isatty _isatty
#else
#include <dlfcn.h>
#include <unistd.h>
#include <sys/stat.h>
#define http_isatty isatty
#endif

typedef struct { char * data; int64_t len; } http_buf;            // a u8[:]
typedef struct { const char * data; int64_t len; } http_text;     // a const u8[:]

// libcurl's API, as curl/curl.h declares it
struct curl_slist;
typedef int (*curl_global_init_fn)(long);
typedef void * (*curl_easy_init_fn)(void);
typedef int (*curl_easy_setopt_fn)(void *, int, ...);
typedef int (*curl_easy_perform_fn)(void *);
typedef int (*curl_easy_getinfo_fn)(void *, int, ...);
typedef void (*curl_easy_cleanup_fn)(void *);
typedef const char * (*curl_easy_strerror_fn)(int);
typedef struct curl_slist * (*curl_slist_append_fn)(struct curl_slist *, const char *);
typedef void (*curl_slist_free_all_fn)(struct curl_slist *);

enum {
    CURLOPT_WRITEDATA         = 10001,
    CURLOPT_URL               = 10002,
    CURLOPT_ERRORBUFFER       = 10010,
    CURLOPT_WRITEFUNCTION     = 20011,
    CURLOPT_HTTPHEADER        = 10023,
    CURLOPT_HEADERDATA        = 10029,
    CURLOPT_NOPROGRESS        = 43,
    CURLOPT_NOBODY            = 44,
    CURLOPT_FOLLOWLOCATION    = 52,
    CURLOPT_XFERINFODATA      = 10057,
    CURLOPT_HEADERFUNCTION    = 20079,
    CURLOPT_RESUME_FROM_LARGE = 30116,
    CURLOPT_SSL_OPTIONS       = 216,
    CURLOPT_XFERINFOFUNCTION  = 20219,
    CURLINFO_RESPONSE_CODE    = 0x200000 + 2,
    CURLSSLOPT_NATIVE_CA      = 1 << 4,
    CURL_GLOBAL_DEFAULT       = 3,
    CURL_ERROR_SIZE           = 256,
};

static struct {
    int tried, ok;
    curl_easy_init_fn easy_init;
    curl_easy_setopt_fn easy_setopt;
    curl_easy_perform_fn easy_perform;
    curl_easy_getinfo_fn easy_getinfo;
    curl_easy_cleanup_fn easy_cleanup;
    curl_easy_strerror_fn easy_strerror;
    curl_slist_append_fn slist_append;
    curl_slist_free_all_fn slist_free_all;
} curl;

#ifdef _WIN32
static const char * const curl_names[] = { "libcurl-4.dll", "libcurl.dll", "libcurl-x64.dll" };
static void * lib_open(const char * name) { return (void *) LoadLibraryA(name); }
static void * lib_sym(void * lib, const char * name) { return (void *) GetProcAddress((HMODULE) lib, name); }
#else
#ifdef __APPLE__
static const char * const curl_names[] = { "libcurl.4.dylib", "libcurl.dylib" };
#else
static const char * const curl_names[] = { "libcurl.so.4", "libcurl.so", "libcurl-gnutls.so.4" };
#endif
static void * lib_open(const char * name) { return dlopen(name, RTLD_NOW); }
static void * lib_sym(void * lib, const char * name) { return dlsym(lib, name); }
#endif

static int curl_load(void) {
    if (curl.tried) return curl.ok;
    curl.tried = 1;
    void * lib = NULL;
    for (size_t i = 0; i < sizeof curl_names / sizeof curl_names[0] && !lib; i++) {
        lib = lib_open(curl_names[i]);
    }
    if (!lib) return 0;
    curl_global_init_fn global_init = (curl_global_init_fn) lib_sym(lib, "curl_global_init");
    curl.easy_init = (curl_easy_init_fn) lib_sym(lib, "curl_easy_init");
    curl.easy_setopt = (curl_easy_setopt_fn) lib_sym(lib, "curl_easy_setopt");
    curl.easy_perform = (curl_easy_perform_fn) lib_sym(lib, "curl_easy_perform");
    curl.easy_getinfo = (curl_easy_getinfo_fn) lib_sym(lib, "curl_easy_getinfo");
    curl.easy_cleanup = (curl_easy_cleanup_fn) lib_sym(lib, "curl_easy_cleanup");
    curl.easy_strerror = (curl_easy_strerror_fn) lib_sym(lib, "curl_easy_strerror");
    curl.slist_append = (curl_slist_append_fn) lib_sym(lib, "curl_slist_append");
    curl.slist_free_all = (curl_slist_free_all_fn) lib_sym(lib, "curl_slist_free_all");
    if (!global_init || !curl.easy_init || !curl.easy_setopt || !curl.easy_perform || !curl.easy_getinfo ||
        !curl.easy_cleanup || !curl.easy_strerror || !curl.slist_append || !curl.slist_free_all) {
        return 0;
    }
    if (global_init(CURL_GLOBAL_DEFAULT) != 0) return 0;
    curl.ok = 1;
    return 1;
}

// The last response: its body (when not written to a file), its headers
// (those of the last response after the redirects) and the error.
static struct { char * data; size_t len, cap; } body, headers;
static char error_text[CURL_ERROR_SIZE + 64];

static int grow_append(char ** data, size_t * len, size_t * cap, const char * p, size_t n) {
    if (*len + n + 1 > *cap) {
        size_t c = *cap ? *cap : 4096;
        while (c < *len + n + 1) c *= 2;
        char * d = (char *) realloc(*data, c);
        if (!d) return 0;
        *data = d;
        *cap = c;
    }
    memcpy(*data + *len, p, n);
    *len += n;
    (*data)[*len] = 0;
    return 1;
}

static char * cstr(http_text s) {
    char * c = (char *) malloc((size_t) s.len + 1);
    if (c) {
        memcpy(c, s.data, (size_t) s.len);
        c[s.len] = 0;
    }
    return c;
}

// The progress bar llama.cpp's downloads draw on a terminal.
// port-of: common/download.cpp (ProgressBar::on_start, on_update)
typedef struct {
    FILE * file;        // the file the body goes to, or NULL for memory
    int64_t offset;     // the bytes already there, when resuming
    int64_t step;       // the bytes since the bar was last drawn
    int bar;            // whether to draw it
    int drawn;          // whether its line was started
    int status;         // the status of the response being received
    char name[256];     // the file's name, cut at 39 characters
    size_t name_len;    // its length in characters
} transfer;

static void bar_name(transfer * t, const char * url) {
    const char * s = strrchr(url, '/');
    s = s ? s + 1 : url;
    size_t n = strcspn(s, "?");
    if (n > sizeof t->name - 8) n = sizeof t->name - 8;
    t->name_len = 0;
    size_t i = 0;
    for (; i < n; i++) {
        if (((unsigned char) s[i] & 0xC0) != 0x80) {
            if (t->name_len++ == 39) {
                break;
            }
        }
    }
    memcpy(t->name, s, i);
    t->name[i] = 0;
    if (i < n) strcat(t->name, "\xE2\x80\xA6");     // …
}

static void bar_draw(transfer * t, int64_t downloaded, int64_t total) {
    if (!t->drawn) {
        fputs("\n", stdout);
        t->drawn = 1;
    }
    size_t bar = (55 - t->name_len) * 2;
    size_t pct = (size_t) ((100 * downloaded) / total);
    size_t pos = (size_t) ((bar * downloaded) / total);
    printf("\rDownloading %s ", t->name);
    for (size_t i = 0; i < bar; i += 2) {
        fputs(i + 1 < pos ? "\xE2\x94\x80" : (i < pos ? "\xE2\x95\xB4" : " "), stdout);    // ─ ╴
    }
    printf("%4zu%%\033[K\r", pct);
    fflush(stdout);
}

static int on_progress(void * p, int64_t dltotal, int64_t dlnow, int64_t ultotal, int64_t ulnow) {
    (void) ultotal; (void) ulnow;
    transfer * t = (transfer *) p;
    if (t->bar && dltotal > 0) {
        int64_t total = t->offset + dltotal, now = t->offset + dlnow;
        if (t->step == 0 || now - t->step >= total / 1000 || now == total) {
            bar_draw(t, now, total);
            t->step = now;
        }
    }
    return 0;
}

static size_t on_body(char * p, size_t size, size_t n, void * user) {
    transfer * t = (transfer *) user;
    size_t len = size * n;
    if (t->file) {
        // an error page is not part of the file
        return t->status == 200 || t->status == 206 ? fwrite(p, 1, len, t->file) : len;
    }
    return grow_append(&body.data, &body.len, &body.cap, p, len) ? len : 0;
}

static size_t on_header(char * p, size_t size, size_t n, void * user) {
    transfer * t = (transfer *) user;
    size_t len = size * n;
    if (len >= 5 && memcmp(p, "HTTP/", 5) == 0) {     // a new response, after a redirect
        headers.len = 0;
        const char * sp = memchr(p, ' ', len);
        t->status = sp ? atoi(sp + 1) : 0;
    }
    return grow_append(&headers.data, &headers.len, &headers.cap, p, len) ? len : 0;
}

// One request: GET, or HEAD when head is 1, to url with the extra request
// headers in header_lines ("Name: value" lines, each ended by '\n').
// The body goes to the file at path (appended to, from byte resume_from
// when that is above 0), or to memory when path is empty; a progress bar
// is drawn on a terminal when bar is 1. Gives the HTTP status, -1 when no
// response came (see common_http_error), or -2 when libcurl is missing.
int64_t common_http_request(http_text url, http_text header_lines, int64_t head, http_text path,
                            int64_t resume_from, int64_t bar) {
    error_text[0] = 0;
    body.len = 0;
    headers.len = 0;
    if (!curl_load()) {
        snprintf(error_text, sizeof error_text, "libcurl was not found (%s)", curl_names[0]);
        return -2;
    }
    transfer t;
    memset(&t, 0, sizeof t);
    char * u = cstr(url);
    char * fpath = path.len > 0 ? cstr(path) : NULL;
    if (fpath) {
        t.file = fopen(fpath, "ab");
        if (!t.file) {
            snprintf(error_text, sizeof error_text, "cannot open %s for writing", fpath);
            free(u);
            free(fpath);
            return -1;
        }
    }
    t.offset = resume_from > 0 ? resume_from : 0;
    t.bar = bar && http_isatty(1);
    bar_name(&t, u);

    struct curl_slist * list = NULL;
    char * lines = cstr(header_lines);
    for (char * s = lines; s && *s;) {
        char * e = strchr(s, '\n');
        if (e) *e = 0;
        if (*s) list = curl.slist_append(list, s);
        s = e ? e + 1 : s + strlen(s);
    }

    void * h = curl.easy_init();
    char errbuf[CURL_ERROR_SIZE] = "";
    curl.easy_setopt(h, CURLOPT_URL, u);
    curl.easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);
    curl.easy_setopt(h, CURLOPT_HTTPHEADER, list);
    curl.easy_setopt(h, CURLOPT_ERRORBUFFER, errbuf);
    curl.easy_setopt(h, CURLOPT_HEADERFUNCTION, on_header);
    curl.easy_setopt(h, CURLOPT_HEADERDATA, &t);
    curl.easy_setopt(h, CURLOPT_WRITEFUNCTION, on_body);
    curl.easy_setopt(h, CURLOPT_WRITEDATA, &t);
#ifdef _WIN32
    // the system's certificate store, whichever TLS backend libcurl has
    curl.easy_setopt(h, CURLOPT_SSL_OPTIONS, (long) CURLSSLOPT_NATIVE_CA);
#endif
    if (head) curl.easy_setopt(h, CURLOPT_NOBODY, 1L);
    if (t.offset > 0) curl.easy_setopt(h, CURLOPT_RESUME_FROM_LARGE, (int64_t) t.offset);
    if (t.bar) {
        curl.easy_setopt(h, CURLOPT_NOPROGRESS, 0L);
        curl.easy_setopt(h, CURLOPT_XFERINFOFUNCTION, on_progress);
        curl.easy_setopt(h, CURLOPT_XFERINFODATA, &t);
    }
    int rc = curl.easy_perform(h);
    long status = -1;
    if (rc == 0) {
        curl.easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
    } else {
        snprintf(error_text, sizeof error_text, "%s", errbuf[0] ? errbuf : curl.easy_strerror(rc));
    }
    curl.easy_cleanup(h);
    curl.slist_free_all(list);
    if (t.file && fclose(t.file) != 0 && rc == 0) {
        snprintf(error_text, sizeof error_text, "error writing to %s", fpath);
        status = -1;
    }
    free(lines);
    free(u);
    free(fpath);
    return status;
}

// The last response's body, into out: its length (nothing is written when
// out is too small).
int64_t common_http_body(http_buf out) {
    if ((int64_t) body.len <= out.len && body.len > 0) memcpy(out.data, body.data, body.len);
    return (int64_t) body.len;
}

// The value of the last response's header name (case-insensitive), into
// out: its length, or -1 when it has none.
int64_t common_http_header(http_text name, http_buf out) {
    for (size_t at = 0; at < headers.len;) {
        char * line = headers.data + at;
        char * end = strchr(line, '\n');
        size_t n = end ? (size_t) (end - line) : headers.len - at;
        at += n + 1;
        if (n > (size_t) name.len && line[name.len] == ':') {
            int same = 1;
            for (int64_t i = 0; i < name.len && same; i++) {
                char a = line[i], b = name.data[i];
                if (a >= 'A' && a <= 'Z') a = (char) (a - 'A' + 'a');
                if (b >= 'A' && b <= 'Z') b = (char) (b - 'A' + 'a');
                same = a == b;
            }
            if (!same) continue;
            size_t s = (size_t) name.len + 1, e = n;
            while (s < e && (line[s] == ' ' || line[s] == '\t')) s++;
            while (e > s && (line[e - 1] == '\r' || line[e - 1] == ' ' || line[e - 1] == '\t')) e--;
            if ((int64_t) (e - s) <= out.len) memcpy(out.data, line + s, e - s);
            return (int64_t) (e - s);
        }
    }
    return -1;
}

// What went wrong in the last request, into out (cut at its size): the length.
int64_t common_http_error(http_buf out) {
    size_t n = strlen(error_text);
    if ((int64_t) n > out.len) n = (size_t) out.len;
    memcpy(out.data, error_text, n);
    return (int64_t) n;
}

// The size of the file at path, or -1 when there is none.
int64_t common_file_size(http_text path) {
    char * p = cstr(path);
    int64_t size = -1;
    FILE * f = p ? fopen(p, "rb") : NULL;
    if (f) {
#ifdef _WIN32
        if (_fseeki64(f, 0, SEEK_END) == 0) size = _ftelli64(f);
#else
        if (fseeko(f, 0, SEEK_END) == 0) size = (int64_t) ftello(f);
#endif
        fclose(f);
    }
    free(p);
    return size;
}

// A symbolic link at link to target (relative to the link's directory):
// 1 when it was made.
int64_t common_symlink(http_text target, http_text link) {
    char * t = cstr(target);
    char * l = cstr(link);
    int ok = 0;
    if (t && l) {
#ifdef _WIN32
        for (char * c = t; *c; c++) if (*c == '/') *c = '\\';
        ok = CreateSymbolicLinkA(l, t, 0x2 /* SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE */) != 0;
#else
        ok = symlink(t, l) == 0;
#endif
    }
    free(t);
    free(l);
    return ok;
}
