// The libc number and time functions llama.cpp's common/ relies on, and
// the float formats they make: std::to_string's, and Python's repr (the
// fewest digits that read back), which Jinja's and JSON's output use.
// port-of: common/jinja/value.cpp (float_repr, float_json)
import std;

namespace number;

extern fn c_fmt_fixed(out: u8[:], v: f64) -> i64 = "jinja_fmt_fixed";
extern fn c_fmt_exp(out: u8[:], prec: i64, v: f64) -> i64 = "jinja_fmt_exp";
extern fn c_strtod(s: const u8[:], used: i64&) -> f64 = "jinja_strtod";
extern fn c_stoi(s: const u8[:], base: i64, ok: i64&) -> i64 = "jinja_stoi";
extern fn c_time() -> i64 = "jinja_time";
extern fn c_strftime(out: u8[:], fmt: const u8[:], t: i64) -> i64 = "jinja_strftime";

// value_float_t::as_string: std::to_string's six decimals, less the
// trailing zeros, but one.
fn format_float(out: u8[>..]&, v: f64) {
    var buf: u8[400] = [0; 400];
    let n = c_fmt_fixed(buf[..], v);
    var end = n;
    while end > 0 && buf[end - 1] == '0' { end--; }
    out.append(buf[..end]);
    if end > 0 && buf[end - 1] == '.' { out.push('0'); }
}

// A float as Python's repr() writes it: the fewest digits that read back as
// the same value (value.cpp float_repr).
fn float_repr(out: u8[>..]&, val: f64) {
    if val != val { out.append("nan"); return; }
    if val - val != 0.0 { out.append(if val < 0.0 { "-inf" } else { "inf" }); return; }
    let mag = abs(val);
    let is_pow2 = mag != 0.0 && frexp_half(mag);
    var digits: u8[..32] = [];
    var exp10 = 0;
    // DBL_DIG - 1 digits after the first to start with; 17 always read back
    var prec = if mag < 2.2250738585072014e-308 { 0 } else { 14 };
    while digits.len == 0 {
        var buf: u8[40] = [0; 40];
        let n = c_fmt_exp(buf[..], prec, mag);
        var d: u8[..32] = [];
        var e = 0;
        while e < n && buf[e] != 'e' {
            if buf[e] >= '0' && buf[e] <= '9' { d.push(buf[e]); }
            e++;
        }
        let ev, ok = parse_int(buf[e + 1..n]);
        exp10 = ev;
        if prec == 16 || reads_back(d[..], exp10 - prec, mag) {
            digits = d;
        } else if is_pow2 && d[d.len - 1] != '9' {
            d[d.len - 1] += 1;
            if reads_back(d[..], exp10 - prec, mag) { digits = d; }
        }
        prec++;
    }
    while digits.len > 1 && digits[digits.len - 1] == '0' { digits.pop(); }

    let n = digits.len;
    let point = exp10 + 1;
    if val < 0.0 || (val == 0.0 && 1.0 / val < 0.0) { out.push('-'); }
    if point > -4 && point <= 16 {
        if point <= 0 {
            out.append("0.");
            for i in -point { out.push('0'); }
            out.append(digits[..]);
        } else if point >= n {
            out.append(digits[..]);
            for i in point - n { out.push('0'); }
            out.append(".0");
        } else {
            out.append(digits[..point]);
            out.push('.');
            out.append(digits[point..]);
        }
    } else {
        out.push(digits[0]);
        if n > 1 { out.push('.'); out.append(digits[1..]); }
        out.push('e');
        out.push(if exp10 < 0 { '-' } else { '+' });
        let e = abs(exp10);
        if e < 10 { out.push('0'); }
        format(out, e);
    }
}

// Whether the digits times 10^e read back as mag.
fn reads_back(d: const u8[:], e: i64, mag: f64) -> bool {
    var s: u8[..64] = [];
    s.append(d);
    s.push('e');
    var t: u8[>..] = [];
    format(t, e);
    s.append(t);
    var used = 0;
    return c_strtod(s[..], used) == mag;
}

// frexp(mag) == 0.5: mag is a power of two.
fn frexp_half(mag: f64) -> bool {
    var m = mag;
    while m >= 1.0 { m *= 0.5; }
    while m < 0.5 { m *= 2.0; }
    return m == 0.5;
}

// A float as json.dumps writes it: NaN and Infinity, which JSON lacks.
fn float_json(out: u8[>..]&, v: f64) {
    if v != v { out.append("NaN"); return; }
    if v - v != 0.0 { out.append(if v < 0.0 { "-Infinity" } else { "Infinity" }); return; }
    float_repr(out, v);
}

// The time now; fixed_time instead when set (>= 0), for tests that compare
// with prompts made at another time.
var fixed_time: i64 = -1;
fn now() -> i64 { if fixed_time >= 0 { fixed_time } else { c_time() } }
