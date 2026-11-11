// stash: a set of globals written to bytes and read back, as the server does
// with the state of the slot it stops and resumes serving.
import std;
import stash from "../tools/stash";

var failed = 0;

fn check(name: const u8[:], ok: bool) {
    if !ok { failed++; print("FAIL ", name); }
}

fn same(a: const i32[:], b: const i32[:]) -> bool {
    guard a.len == b.len else { return false; }
    for x, i in a { if x != b[i] { return false; } }
    return true;
}

struct Range { at: i64, len: i64 }
struct Wide { xs: i32[..8], on: bool = false, f: f32 = 0.0 }
enum Kind { None, Word { at: i64, len: i64 } }

// the globals of a pretend module, and the one place that lists them
var tokens: i32[>..] = [];
var ranges: Range[>..<] = [];
var text: u8[>..] = [];
var count = 0;
var wide = Wide { .. };
var kind = Kind.None;
var on = false;

fn state(s: stash::Stash&) {
    stash::keep(s, tokens);
    stash::keep(s, ranges);
    stash::keep(s, text);
    stash::keep_value(s, count);
    stash::keep_value(s, wide);
    stash::keep_value(s, kind);
    stash::keep_value(s, on);
}

fn set_first() {
    tokens.clear(); tokens.append([1, 2, 3]);
    ranges.clear(); ranges.push(Range { at: 4, len: 5 });
    text.clear(); text.append("first");
    count = 7;
    wide = Wide { xs: [1, 2, 3], on: true, f: 0.5 };
    kind = Kind.Word { at: 2, len: 9 };
    on = true;
}

fn is_first() -> bool {
    let word = match kind { Word w => w.at == 2 && w.len == 9, None => false };
    return same(tokens, [1, 2, 3]) && ranges.len == 1 && ranges[0].at == 4 && ranges[0].len == 5 && text == "first"
        && count == 7 && same(wide.xs, [1, 2, 3]) && wide.on && wide.f == 0.5 && word && on;
}

fn set_second() {
    tokens.clear();
    ranges.clear(); ranges.push(Range { at: 1, len: 1 }); ranges.push(Range { at: 2, len: 2 });
    text.clear(); text.append("the second one");
    count = -1;
    wide = Wide { .. };
    kind = Kind.None;
    on = false;
}

fn is_second() -> bool {
    let none = match kind { Word w => false, None => true };
    return tokens.len == 0 && ranges.len == 2 && ranges[1].at == 2 && text == "the second one"
        && count == -1 && wide.xs.len == 0 && !wide.on && none && !on;
}

// ---- the pieces: one image after another, each found by its length ----

fn test_images() {
    var s = stash::Stash { .. };
    let a: i32[..4] = [10, 20, 30];
    let none: i32[..4] = [];
    stash::put(s, a[..]);
    stash::put(s, none[..]);
    stash::put(s, "xyz");
    let first, ok1 = from_bytes<i32[>..]>(stash::image(s));
    check("image: the first array", ok1 && same(first, [10, 20, 30]));
    let second, ok2 = from_bytes<i32[>..]>(stash::image(s));
    check("image: an empty array", ok2 && second.len == 0);
    let third, ok3 = from_bytes<u8[>..]>(stash::image(s));
    check("image: the last one", ok3 && third == "xyz");
    check("image: all of the bytes read", s.at == s.bytes.len);
}

// ---- the interface: a state kept, changed and brought back ----

fn test_round_trip() {
    set_first();
    var s = stash::Stash { .. };
    state(s);
    check("saving leaves the state as it was", is_first());
    set_second();
    var back = stash::Stash { loading: true, .. };
    back.bytes.append(s.bytes);
    state(back);
    check("round trip", is_first());
    check("round trip: all of the bytes read", back.at == back.bytes.len);
}

// two states take turns in the same globals, as two slots do
fn test_swap() {
    set_first();
    var one = stash::Stash { .. };
    state(one);
    set_second();
    var two = stash::Stash { .. };
    state(two);
    one.loading = true;
    state(one);
    check("swap: the first is back", is_first());
    tokens.push(4);
    var one_later = stash::Stash { .. };
    state(one_later);
    two.loading = true;
    state(two);
    check("swap: the second is back", is_second());
    one_later.loading = true;
    state(one_later);
    check("swap: the first, with what it did since", same(tokens, [1, 2, 3, 4]) && count == 7 && text == "first");
}

fn main() {
    test_images();
    test_round_trip();
    test_swap();
    if failed > 0 { print(failed, " failed"); exit(1); }
    print("stash: all passed");
}
