// stb_image behind mtmd's image decoding (mtmd-helper.cpp's
// mtmd_helper_bitmap_init_from_buf): any format stb reads, as RGB.
#include <stdint.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "sha256/sha256.h"

typedef struct { const unsigned char * data; int64_t len; } lg_text;
typedef struct { unsigned char * data; int64_t len; } lg_buf;

// The size of the image in buf, 0 if stb cannot read it.
int64_t mtmd_image_info(lg_text buf, int64_t * nx, int64_t * ny) {
    int x, y, n;
    if (buf.len > INT32_MAX || !stbi_info_from_memory(buf.data, (int) buf.len, &x, &y, &n)) { return 0; }
    *nx = x;
    *ny = y;
    return 1;
}

// The pixels of the image in buf, nx * ny * 3 bytes into out; 0 if it cannot be decoded.
int64_t mtmd_image_decode(lg_text buf, lg_buf out) {
    int x, y, n;
    if (buf.len > INT32_MAX) { return 0; }
    unsigned char * data = stbi_load_from_memory(buf.data, (int) buf.len, &x, &y, &n, 3);
    if (!data) { return 0; }
    int64_t size = (int64_t) x * y * 3;
    if (size != out.len) { stbi_image_free(data); return 0; }
    memcpy(out.data, data, (size_t) size);
    stbi_image_free(data);
    return 1;
}

// The SHA-256 of buf, 32 bytes into out (hash_sha256: a media file's id).
void mtmd_sha256(lg_text buf, lg_buf out) {
    if (out.len == 32) { sha256_hash(out.data, buf.data, (size_t) buf.len); }
}
