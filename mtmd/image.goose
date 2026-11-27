// Images: decoded with stb_image, resized as Pillow does (img_tool), cropped,
// and turned into the normalized floats an encoder reads.
// port-of: tools/mtmd/mtmd-image.cpp tools/mtmd/mtmd-impl.h tools/mtmd/mtmd-helper.cpp
import std;
import math;
namespace mtmd;

extern fn c_image_info(buf: const u8[:], nx: i64&, ny: i64&) -> i64 = "mtmd_image_info";
extern fn c_image_decode(buf: const u8[:], out: u8[:]) -> i64 = "mtmd_image_decode";

// RGB pixels, row after row (mtmd_image_u8).
struct ImageU8 { nx: i64, ny: i64, buf: u8[>..] }

// The same, as floats (mtmd_image_f32).
struct ImageF32 { nx: i64, ny: i64, buf: f32[>..] }

// The image a file's bytes hold, RGB; false if stb cannot read it.
fn decode_image(bytes: const u8[:], img: ImageU8&) -> bool {
    var nx = 0;
    var ny = 0;
    guard c_image_info(bytes, nx, ny) != 0 else { return false; }
    img.nx = nx;
    img.ny = ny;
    img.buf.clear();
    img.buf.resize(nx * ny * 3, 0);
    return c_image_decode(bytes, img.buf[..]) != 0;
}

enum Resize { Bilinear, Bicubic, Lanczos }

// Pillow's fixed point: 32 bits less 8 for a pixel, less 2 of headroom.
let PRECISION_BITS = 22;

fn sinc(v: f64) -> f64 {
    if v == 0.0 { return 1.0; }
    let pi_v = v * 3.141592653589793238462643383279502884;
    return sin(pi_v) / pi_v;
}

fn resample_filter(algo: Resize, x_in: f64) -> f64 {
    if algo == Resize.Lanczos {
        if -3.0 <= x_in && x_in < 3.0 { return sinc(x_in) * sinc(x_in / 3.0); }
        return 0.0;
    }
    let x = if x_in < 0.0 { -x_in } else { x_in };
    if algo == Resize.Bilinear { return if x < 1.0 { 1.0 - x } else { 0.0 }; }
    let a = -0.5;
    if x < 1.0 { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0; }
    if x < 2.0 { return (((x - 5.0) * x + 8.0) * x - 4.0) * a; }
    return 0.0;
}

fn clip8(v: i64) -> u8 { if v < 0 { 0 } else if v > 255 { 255 } else { v as u8 } }

// The filter of one dimension (precompute_weights): for each output pixel,
// its first input pixel and how many, in bounds, and their fixed-point
// weights, ksize each, in weights; returns ksize.
fn precompute_weights(algo: Resize, in_size: i64, out_size: i64, bounds: i64[>..]&, weights: i64[>..]&) -> i64 {
    let filter_support = if algo == Resize.Bilinear { 1.0 } else if algo == Resize.Bicubic { 2.0 } else { 3.0 };
    let scale: f64 = in_size as f64 / out_size;
    let filterscale = if scale < 1.0 { 1.0 } else { scale };
    let support = filter_support * filterscale;
    let ksize = (ceil(support) as i64) * 2 + 1;
    var pre: f64[>..] = [];
    pre.resize(out_size * ksize, 0.0);
    bounds.clear();
    bounds.resize(out_size * 2, 0);
    for xx in out_size {
        let center = (xx + 0.5) * scale;
        var ww = 0.0;
        let ss = 1.0 / filterscale;
        var xmin = (center - support + 0.5) as i64;
        if xmin < 0 { xmin = 0; }
        var xmax = (center + support + 0.5) as i64;
        if xmax > in_size { xmax = in_size; }
        xmax -= xmin;
        for x in xmax {
            let w = resample_filter(algo, ((x + xmin) - center + 0.5) * ss);
            pre[xx * ksize + x] = w;
            ww += w;
        }
        for x in xmax {
            if ww != 0.0 { pre[xx * ksize + x] /= ww; }
        }
        bounds[xx * 2] = xmin;
        bounds[xx * 2 + 1] = xmax;
    }
    let fxp_scale = (1 << PRECISION_BITS) as f64;
    weights.clear();
    for w in pre {
        // Pillow adds or takes 0.5, then truncates toward zero
        weights.push((w * fxp_scale + (if w < 0.0 { -0.5 } else { 0.5 })) as i32);
    }
    return ksize;
}

fn resample_horizontal(src: const u8[:], in_nx: i64, in_ny: i64, out_nx: i64, ksize: i64,
                       bounds: const i64[:], weights: const i64[:], out: u8[>..]&) {
    out.clear();
    out.resize(out_nx * in_ny * 3, 0);
    for yy in in_ny {
        let row = yy * in_nx * 3;
        for xx in out_nx {
            let xmin = bounds[xx * 2];
            let xcnt = bounds[xx * 2 + 1];
            var s0: i64 = 1 << (PRECISION_BITS - 1);
            var s1: i64 = 1 << (PRECISION_BITS - 1);
            var s2: i64 = 1 << (PRECISION_BITS - 1);
            for x in xcnt {
                let k = weights[xx * ksize + x];
                let p = row + (xmin + x) * 3;
                s0 += src[p] * k;
                s1 += src[p + 1] * k;
                s2 += src[p + 2] * k;
            }
            let o = (yy * out_nx + xx) * 3;
            out[o] = clip8(s0 >> PRECISION_BITS);
            out[o + 1] = clip8(s1 >> PRECISION_BITS);
            out[o + 2] = clip8(s2 >> PRECISION_BITS);
        }
    }
}

fn resample_vertical(src: const u8[:], in_nx: i64, out_ny: i64, ksize: i64,
                     bounds: const i64[:], weights: const i64[:], out: u8[>..]&) {
    let row_elems = in_nx * 3;
    out.clear();
    out.resize(row_elems * out_ny, 0);
    var acc: i64[>..] = [];
    acc.resize(row_elems, 0);
    for yy in out_ny {
        let ymin = bounds[yy * 2];
        let ycnt = bounds[yy * 2 + 1];
        for i in row_elems { acc[i] = 1 << (PRECISION_BITS - 1); }
        for y in ycnt {
            let w = weights[yy * ksize + y];
            let r = (ymin + y) * row_elems;
            for i in row_elems { acc[i] += src[r + i] * w; }
        }
        for i in row_elems { out[yy * row_elems + i] = clip8(acc[i] >> PRECISION_BITS); }
    }
}

// img resized to nx by ny (img_tool::resize with PAD_NONE, resize_pillow).
fn resize_image(img: ImageU8&, nx: i64, ny: i64, algo: Resize, dst: ImageU8&) {
    guard nx > 0 && nx <= 65536 && ny > 0 && ny <= 65536 else { abort(str("resize target ", nx, "x", ny, " is out of range (max 65536)")); }
    dst.nx = nx;
    dst.ny = ny;
    dst.buf.clear();
    let need_h = nx != img.nx;
    let need_v = ny != img.ny;
    var bh: i64[>..] = [];
    var wh: i64[>..] = [];
    var bv: i64[>..] = [];
    var wv: i64[>..] = [];
    let kh = if need_h { precompute_weights(algo, img.nx, nx, bh, wh) } else { 0 };
    let kv = if need_v { precompute_weights(algo, img.ny, ny, bv, wv) } else { 0 };
    if need_h && need_v {
        var tmp: u8[>..] = [];
        resample_horizontal(img.buf, img.nx, img.ny, nx, kh, bh, wh, tmp);
        resample_vertical(tmp, nx, ny, kv, bv, wv, dst.buf);
    } else if need_h {
        resample_horizontal(img.buf, img.nx, img.ny, nx, kh, bh, wh, dst.buf);
    } else if need_v {
        resample_vertical(img.buf, img.nx, ny, kv, bv, wv, dst.buf);
    } else {
        dst.buf.append(img.buf);
    }
}

// img resized into nx by ny keeping its aspect ratio, centered on black
// (img_tool::resize with PAD_CEIL); the same as resize_image when it fits.
fn resize_image_padded(img: ImageU8&, nx: i64, ny: i64, algo: Resize, dst: ImageU8&) {
    if img.nx == nx && img.ny == ny {
        dst.nx = nx;
        dst.ny = ny;
        dst.buf.clear();
        dst.buf.append(img.buf);
        return;
    }
    let scale_w: f32 = nx as f32 / img.nx;
    let scale_h: f32 = ny as f32 / img.ny;
    let scale = min(scale_w, scale_h);
    let new_w = min(ceil(img.nx * scale) as i64, nx);
    let new_h = min(ceil(img.ny * scale) as i64, ny);
    var resized = ImageU8 { nx: 0, ny: 0, buf: [] };
    resize_image(img, new_w, new_h, algo, resized);
    dst.nx = nx;
    dst.ny = ny;
    dst.buf.clear();
    dst.buf.resize(nx * ny * 3, 0);
    let ox = (nx - new_w) / 2;
    let oy = (ny - new_h) / 2;
    for y in new_h {
        for x in new_w {
            let d = ((y + oy) * nx + x + ox) * 3;
            let s = (y * new_w + x) * 3;
            for ch in 3 { dst.buf[d + ch] = resized.buf[s + ch]; }
        }
    }
}

// The w by h part of img at x, y (img_tool::crop).
fn crop_image(img: ImageU8&, x: i64, y: i64, w: i64, h: i64, dst: ImageU8&) {
    guard x >= 0 && y >= 0 && w > 0 && h > 0 && x + w <= img.nx && y + h <= img.ny else { abort("crop out of the image"); }
    dst.nx = w;
    dst.ny = h;
    dst.buf.clear();
    for i in h {
        let r = ((y + i) * img.nx + x) * 3;
        dst.buf.append(img.buf[r..r + w * 3]);
    }
}

// The pixels as floats in [0, 1], then, if normalized, less the mean over the
// standard deviation of each channel (from_u8, normalize).
fn to_f32(img: ImageU8&, normalized: bool, mean: const f32[:], std_: const f32[:], dst: ImageF32&) {
    dst.nx = img.nx;
    dst.ny = img.ny;
    dst.buf.clear();
    for v in img.buf { dst.buf.push(v / 255.0); }
    if normalized {
        for i in img.nx * img.ny {
            for ch in 3 { dst.buf[i * 3 + ch] = (dst.buf[i * 3 + ch] - mean[ch]) / std_[ch]; }
        }
    }
}
