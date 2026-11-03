// The C behind jinja.goose: the libc formatting, parsing and time functions
// llama.cpp's Jinja engine calls, so that numbers and dates come out the same.
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { char * data; int64_t len; } lg_buf;            // a u8[:]
typedef struct { const char * data; int64_t len; } lg_text;     // a const u8[:]

// s with a terminating NUL, in buf (cut at its size).
static const char * lg_cstr(lg_text s, char * buf, size_t size) {
    size_t n = (size_t) s.len < size - 1 ? (size_t) s.len : size - 1;
    memcpy(buf, s.data, n);
    buf[n] = 0;
    return buf;
}

// snprintf(out, "%f", v), what std::to_string(double) writes: the length.
int64_t jinja_fmt_fixed(lg_buf out, double v) {
    return snprintf(out.data, (size_t) out.len, "%f", v);
}

// snprintf(out, "%.*e", prec, v): the length.
int64_t jinja_fmt_exp(lg_buf out, int64_t prec, double v) {
    return snprintf(out.data, (size_t) out.len, "%.*e", (int) prec, v);
}

// strtod of s, and how many bytes it read: -1 when out of range (ERANGE),
// where std::stod throws.
double jinja_strtod(lg_text s, int64_t * used) {
    char buf[512];
    const char * c = lg_cstr(s, buf, sizeof buf);
    char * end = NULL;
    errno = 0;
    double v = strtod(c, &end);
    *used = errno == ERANGE ? -1 : end - c;
    return v;
}

// strtol(s, base) as std::stoi reads it: the value, and false when nothing
// was read or it does not fit an int.
int64_t jinja_stoi(lg_text s, int64_t base, int64_t * ok) {
    char buf[512];
    const char * c = lg_cstr(s, buf, sizeof buf);
    char * end = NULL;
    long long v = strtoll(c, &end, (int) base);
    *ok = end != c && v >= -2147483648LL && v <= 2147483647LL;
    return v;
}

int64_t jinja_time(void) {
    return (int64_t) time(NULL);
}

// strftime(out, fmt, localtime(t)): the length, 0 when it does not fit.
int64_t jinja_strftime(lg_buf out, lg_text fmt, int64_t t) {
    char f[512];
    time_t tt = (time_t) t;
    return (int64_t) strftime(out.data, (size_t) out.len, lg_cstr(fmt, f, sizeof f), localtime(&tt));
}
