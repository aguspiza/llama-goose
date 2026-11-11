// A set of globals written to bytes and read back.
//
// A program whose globals hold the state of one of several users (the
// server's slots) lists them once, in a function that takes a Stash and
// calls keep on each. Called with an empty stash it writes them there;
// called with those bytes in a loading stash it brings them back. The
// globals must be flat values and arrays of them.
import std;

namespace stash;

struct Stash {
    loading: bool = false,
    at: i64 = 0,            // loading: where the next image starts
    bytes: u8[>..],         // one image per value kept, each after its length in 8 bytes
}

// Append the image of a.
fn put<T>(s: Stash&, a: const T[:]) {
    let start = s.bytes.len;
    s.bytes.resize(start + 8, 0);
    a.to_bytes(s.bytes);
    var n = s.bytes.len - start - 8;
    for i in 8 { s.bytes[start + i] = (n & 255) as! u8; n >>= 8; }
}

// The next image.
fn image(s: Stash&) -> const u8[:] {
    var n = 0;
    for i in 8 { n |= (s.bytes[s.at + i] as i64) << (8 * i); }
    s.at += 8 + n;
    return s.bytes[s.at - n..s.at];
}

fn keep<T>(s: Stash&, a: T[>..]&) {
    guard s.loading else { put(s, a[..]); return; }
    let v, ok = from_bytes<T[>..]>(image(s));
    guard ok else { abort("stash: not what was kept"); }
    a.clear();
    a.append(v);
}

fn keep<T>(s: Stash&, a: T[>..<]&) {
    guard s.loading else { put(s, a[..]); return; }
    let v, ok = from_bytes<T[>..]>(image(s));
    guard ok else { abort("stash: not what was kept"); }
    a.clear();
    a.append(v);
}

// A value that is not an array.
fn keep_value<T>(s: Stash&, v: T&) {
    guard s.loading else { var one = [v]; put(s, one[..]); return; }
    let a, ok = from_bytes<T[>..]>(image(s));
    guard ok && a.len == 1 else { abort("stash: not what was kept"); }
    v = a[0];
}
