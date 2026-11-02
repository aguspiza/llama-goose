// Unicode for the tokenizers: UTF-8, codepoint categories, NFD/NFC, and the
// pre-tokenizer splits (fast paths, and llama.cpp's std::regex fallback).
// port-of: src/unicode.cpp src/unicode.h
import std;
import .unicode_data;
import .ecma_regex;
namespace unicode;

let MAX_CODEPOINTS = 0x110000;

// Codepoint flags: one category bit, then the helper bits.
let UNDEFINED = 0x0001;
let NUMBER = 0x0002;       // \p{N}
let LETTER = 0x0004;       // \p{L}
let SEPARATOR = 0x0008;    // \p{Z}
let ACCENT_MARK = 0x0010;  // \p{M}
let PUNCTUATION = 0x0020;  // \p{P}
let SYMBOL = 0x0040;       // \p{S}
let CONTROL = 0x0080;      // \p{C}
let MASK_CATEGORIES = 0x00FF;
let WHITESPACE = 0x0100;
let LOWERCASE = 0x0200;
let UPPERCASE = 0x0400;
let NFD = 0x0800;

let OUT_OF_RANGE = 0xFFFFFFFF;

fn build_flags() -> u16[>..] {
    var f: u16[>..] = [];
    f.resize(MAX_CODEPOINTS, UNDEFINED);
    for i in 1..RANGE_START.len {
        let flags = at(RANGE_FLAGS, i - 1) as u16;
        for cpt in (at(RANGE_START, i - 1) as i64)..(at(RANGE_START, i) as i64) { f[cpt] = flags; }
    }
    for i in WHITESPACE_CPTS.len { f[at(WHITESPACE_CPTS, i) as i64] |= WHITESPACE; }
    for i in LOWER_TO.len { f[at(LOWER_TO, i) as i64] |= LOWERCASE; }
    for i in UPPER_TO.len { f[at(UPPER_TO, i) as i64] |= UPPERCASE; }
    for i in NFD_TO.len { f[at(NFD_TO, i) as i64] |= NFD; }
    return f;
}

let CPT_FLAGS = build_flags();

fn flags(cpt: u32) -> u16 {
    if cpt < MAX_CODEPOINTS { CPT_FLAGS[cpt] } else { UNDEFINED }
}

fn is(f: u16, bits: u16) -> bool { f & bits != 0 }

fn len_utf8(c: u8) -> i64 {
    let lookup: i64[16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4];
    return lookup[c >> 4];
}

// The codepoint at s[offset..], and where the next one starts; false where
// llama.cpp throws (bad lead or continuation byte, overlong, out of range).
fn cpt_from_utf8(s: const u8[:], offset: i64) -> u32, i64, bool {
    let c0 = s[offset] as u32;
    if c0 & 0x80 == 0 { return c0, offset + 1, true; }
    if c0 & 0x40 == 0 { return 0, offset, false; }
    fn cont(i: i64) -> bool { i < s.len && s[i] & 0xC0 == 0x80 }
    if c0 & 0x20 == 0 {
        guard cont(offset + 1) else { return 0, offset, false; }
        let r = (c0 & 0x1F) << 6 | s[offset + 1] & 0x3F;
        guard r >= 0x80 else { return 0, offset, false; }
        return r, offset + 2, true;
    }
    if c0 & 0x10 == 0 {
        guard cont(offset + 1) && cont(offset + 2) else { return 0, offset, false; }
        let r = (c0 & 0x0F) << 12 | ((s[offset + 1] & 0x3F) as u32) << 6 | s[offset + 2] & 0x3F;
        guard r >= 0x800 else { return 0, offset, false; }
        return r, offset + 3, true;
    }
    if c0 & 0x08 == 0 {
        guard cont(offset + 1) && cont(offset + 2) && cont(offset + 3) else { return 0, offset, false; }
        let r = (c0 & 0x07) << 18 | ((s[offset + 1] & 0x3F) as u32) << 12 |
                ((s[offset + 2] & 0x3F) as u32) << 6 | s[offset + 3] & 0x3F;
        guard r >= 0x10000 && r <= 0x10FFFF else { return 0, offset, false; }
        return r, offset + 4, true;
    }
    return 0, offset, false;
}

// Invalid bytes become U+FFFD, one per byte.
fn cpts_from_utf8(s: const u8[:]) -> u32[>..] {
    var out: u32[>..] = [];
    var offset = 0;
    while offset < s.len {
        let cpt, next, ok = cpt_from_utf8(s, offset);
        if ok { out.push(cpt); offset = next; } else { out.push(0xFFFD); offset += 1; }
    }
    return out;
}

fn cpt_to_utf8(out: u8[>..]&, cpt: u32) {
    if cpt <= 0x7F {
        out.push(cpt as! u8);
    } else if cpt <= 0x7FF {
        out.push((0xC0 | (cpt >> 6) & 0x1F) as! u8);
        out.push((0x80 | cpt & 0x3F) as! u8);
    } else if cpt <= 0xFFFF {
        out.push((0xE0 | (cpt >> 12) & 0x0F) as! u8);
        out.push((0x80 | (cpt >> 6) & 0x3F) as! u8);
        out.push((0x80 | cpt & 0x3F) as! u8);
    } else if cpt <= 0x10FFFF {
        out.push((0xF0 | (cpt >> 18) & 0x07) as! u8);
        out.push((0x80 | (cpt >> 12) & 0x3F) as! u8);
        out.push((0x80 | (cpt >> 6) & 0x3F) as! u8);
        out.push((0x80 | cpt & 0x3F) as! u8);
    } else {
        abort(str("invalid codepoint ", cpt));
    }
}

fn cpts_to_utf8(out: u8[>..]&, cpts: const u32[:]) { for c in cpts { cpt_to_utf8(out, c); } }

fn flags_from_utf8(s: const u8[:]) -> u16 {
    if s.len == 0 { return UNDEFINED; }
    let cpt, next, ok = cpt_from_utf8(s, 0);
    if !ok { abort("invalid UTF-8"); }
    return flags(cpt);
}

// GPT-2's byte-level encoding: each byte is shown as a printable codepoint.
fn build_byte_cpts() -> u32[256] {
    var map: u32[256] = [0; 256];
    var taken: bool[256] = [false; 256];
    for ch in 0x21..0x7F { map[ch] = ch as u32; taken[ch] = true; }
    for ch in 0xA1..0xAD { map[ch] = ch as u32; taken[ch] = true; }
    for ch in 0xAE..0x100 { map[ch] = ch as u32; taken[ch] = true; }
    var n: u32 = 0;
    for ch in 256 {
        if !taken[ch] { map[ch] = 256 + n; n += 1; }
    }
    return map;
}

let BYTE_CPT = build_byte_cpts();

fn build_cpt_bytes() -> i16[>..] {
    var map: i16[>..] = [];
    map.resize(0x144, -1);
    for ch in 256 { map[BYTE_CPT[ch]] = ch as i16; }
    return map;
}

let CPT_BYTE = build_cpt_bytes();

fn byte_to_utf8(out: u8[>..]&, b: u8) { cpt_to_utf8(out, BYTE_CPT[b]); }

// The byte a GPT-2 byte-level codepoint stands for; false for any other.
fn cpt_to_byte(cpt: u32) -> u8, bool {
    if cpt >= CPT_BYTE.len || CPT_BYTE[cpt] < 0 { return 0, false; }
    return CPT_BYTE[cpt] as! u8, true;
}

// The byte one GPT-2 byte-level character (as UTF-8) stands for.
fn utf8_to_byte(s: const u8[:]) -> u8, bool {
    guard s.len > 0 else { return 0, false; }
    let cpt, next, ok = cpt_from_utf8(s, 0);
    guard ok && next == s.len else { return 0, false; }
    return cpt_to_byte(cpt);
}

fn tolower(cpt: u32) -> u32 {
    let i, found = search(LOWER_FROM, cpt);
    return if found { at(LOWER_TO, i) as u32 } else { cpt };
}

fn is_han(cpt: u32) -> bool {
    return (cpt >= 0x4E00 && cpt <= 0x9FFF)      // CJK Unified Ideographs
        || (cpt >= 0x3400 && cpt <= 0x4DBF)      // extension A
        || (cpt >= 0x20000 && cpt <= 0x2A6DF)    // extension B
        || (cpt >= 0x2A700 && cpt <= 0x2B73F)    // extension C
        || (cpt >= 0x2B740 && cpt <= 0x2B81F)    // extension D
        || (cpt >= 0x2B820 && cpt <= 0x2CEAF)    // extension E
        || (cpt >= 0x2CEB0 && cpt <= 0x2EBEF)    // extension F
        || (cpt >= 0xF900 && cpt <= 0xFAFF)      // compatibility ideographs
        || (cpt >= 0x2F800 && cpt <= 0x2FA1F);   // compatibility supplement
}

fn normalize_nfd(cpts: const u32[:]) -> u32[>..] {
    var out: u32[>..] = [];
    for cpt in cpts {
        let i = upper_bound(NFD_FIRST, cpt) - 1;
        out.push(if at(NFD_FIRST, i) <= cpt && cpt <= at(NFD_LAST, i) { at(NFD_TO, i) as u32 } else { cpt });
    }
    return out;
}

fn canonical_class(cpt: u32) -> u8 {
    let i, found = search(CCC_CPT, cpt);
    return if found { at(CCC_CLASS, i) as u8 } else { 0 };
}

fn canonical_decompose(cpt: u32, out: u32[>..]&, pending: u32[>..<]&) {
    pending.push(cpt);
    while pending.len > 0 {
        let c = pending.pop();
        // Hangul syllables decompose by formula (UAX #15).
        if c >= 0xAC00 && c < 0xAC00 + 11172 {
            let index = c - 0xAC00;
            out.push(0x1100 + index / 588);
            out.push(0x1161 + (index % 588) / 28);
            if index % 28 != 0 { out.push(0x11A7 + index % 28); }
            continue;
        }
        let i, found = search(DEC_CPT, c);
        if !found { out.push(c); continue; }
        if at(DEC_SECOND, i) != 0 { pending.push(at(DEC_SECOND, i) as u32); }
        pending.push(at(DEC_FIRST, i) as u32);
    }
}

fn canonical_compose(first: u32, second: u32) -> u32 {
    if first >= 0x1100 && first < 0x1100 + 19 && second >= 0x1161 && second < 0x1161 + 21 {
        return 0xAC00 + ((first - 0x1100) * 21 + second - 0x1161) * 28;
    }
    if first >= 0xAC00 && first < 0xAC00 + 11172 && (first - 0xAC00) % 28 == 0 &&
       second > 0x11A7 && second < 0x11A7 + 28 {
        return first + second - 0x11A7;
    }
    let i, found = search(COMP_PAIR, (first as u64) << 32 | second);
    return if found { at(COMP_TO, i) as u32 } else { 0 };
}

fn normalize_nfc(cpts: const u32[:]) -> u32[>..] {
    var decomposed: u32[>..] = [];
    {
        var pending: u32[>..<] = [];
        for cpt in cpts { canonical_decompose(cpt, decomposed, pending); }
    }

    // Sort each run of combining marks by class; equal classes keep their order.
    var i = 0;
    while i < decomposed.len {
        if canonical_class(decomposed[i]) == 0 { i += 1; }
        let begin = i;
        while i < decomposed.len && canonical_class(decomposed[i]) != 0 { i += 1; }
        stable_sort(decomposed[begin..i]) { a, b => canonical_class(a) < canonical_class(b) };
    }

    var out: u32[>..] = [];
    var starter = 0;
    var previous_class: u8 = 0;
    for cpt in decomposed {
        let current_class = canonical_class(cpt);
        let composite = if out.len > 0 && (previous_class == 0 || previous_class < current_class) {
            canonical_compose(out[starter], cpt)
        } else { 0 };
        if composite != 0 {
            out[starter] = composite;
        } else {
            if current_class == 0 { starter = out.len; }
            out.push(cpt);
            previous_class = current_class;
        }
    }
    return out;
}

//
// Pre-tokenizer splits. Each one takes the text's codepoints and the lengths
// of the pieces split so far, and appends the lengths of the finer pieces.
//

// One piece being split: its codepoints run from ini to end.
struct Piece { cpts: const u32[:], ini: i64, end: i64, prev_end: i64, out: i64[>..]& }

fn cpt_at(p: Piece&, pos: i64) -> u32 {
    if p.ini <= pos && pos < p.end { p.cpts[pos] } else { OUT_OF_RANGE }
}

fn flags_at(p: Piece&, pos: i64) -> u16 {
    if p.ini <= pos && pos < p.end { flags(p.cpts[pos]) } else { 0 }
}

// Ends the current token at end; returns its length.
fn add_token(p: Piece&, end: i64) -> i64 {
    assert(p.prev_end <= end && end <= p.end);
    let len = end - p.prev_end;
    if len > 0 { p.out.push(len); }
    p.prev_end = end;
    return len;
}

// F(piece) for each piece of the split so far.
fn each_piece<F>(cpts: const u32[:], offsets: const i64[:], out: i64[>..]&) {
    var start = 0;
    for offset in offsets {
        var p = Piece { cpts: cpts, ini: start, end: start + offset, prev_end: start, out: out };
        assert(p.end <= cpts.len);
        start = p.end;
        F(p);
    }
}

// (?i:'s|'t|'re|'ve|'m|'ll|'d) at pos; the length matched (0 for none).
// long_s also folds U+017F (long s) to 's', as K2-Horizon does.
fn contraction_len(p: Piece&, pos: i64, long_s: bool = false) -> i64 {
    guard cpt_at(p, pos) == '\'' && pos + 1 < p.end else { return 0; }
    var next = tolower(cpt_at(p, pos + 1));
    if long_s && next == 0x017F { next = 's'; }
    if next == 's' || next == 't' || next == 'm' || next == 'd' { return 2; }
    if pos + 2 < p.end {
        let next2 = tolower(cpt_at(p, pos + 2));
        if (next == 'r' && next2 == 'e') || (next == 'v' && next2 == 'e') || (next == 'l' && next2 == 'l') {
            return 3;
        }
    }
    return 0;
}

// The whitespace tail shared by the llama3 family:
// \s*[\r\n]+ | \s+(?!\S) | \s+ | one codepoint.
fn whitespace_tail(p: Piece&, pos: i64) -> i64 {
    var n = 0;
    var last_end_r_or_n = 0;
    while is(flags_at(p, pos + n), WHITESPACE) {
        let c = cpt_at(p, pos + n);
        if c == '\r' || c == '\n' { last_end_r_or_n = pos + n + 1; }
        n += 1;
    }
    if last_end_r_or_n > 0 {                          // \s*[\r\n]+
        add_token(p, last_end_r_or_n);
        return last_end_r_or_n;
    }
    if n > 1 && cpt_at(p, pos + n) != OUT_OF_RANGE {  // \s+(?!\S)
        add_token(p, pos + n - 1);
        return pos + n - 1;
    }
    if n > 0 {                                        // \s+
        add_token(p, pos + n);
        return pos + n;
    }
    add_token(p, pos + 1);                            // no match
    return pos + 1;
}

// <space>?[^\s\p{L}\p{N}]+[\r\n]*, with `excluded` the categories it stops at.
// Its first test that a codepoint is there looks at pos (the llama3 family)
// or, with check_next, at the codepoint after the space (Kimi-K2).
// Returns the new position, or -1 when it does not match at pos.
fn punct_run(p: Piece&, pos: i64, excluded: u16, first_flags: u16, check_next: bool = false) -> i64 {
    let c = cpt_at(p, pos);
    var f2 = if c == ' ' { flags_at(p, pos + 1) } else { first_flags };
    let present = if check_next { f2 } else { first_flags };
    guard !is(f2, excluded) && present != 0 else { return -1; }
    var q = pos + (if c == ' ' { 1 } else { 0 });
    while !is(f2, excluded) && f2 != 0 { q += 1; f2 = flags_at(p, q); }
    var c2 = cpt_at(p, q);
    while c2 == '\r' || c2 == '\n' { q += 1; c2 = cpt_at(p, q); }
    add_token(p, q);
    return q;
}

// \p{N}{1,3}: digits in groups of up to three.
fn digits_by_three(p: Piece&, pos: i64) -> i64 {
    var q = pos;
    var ini = q;
    while is(flags_at(p, q), NUMBER) {
        q += 1;
        if q - ini >= 3 { add_token(p, q); ini = q; }
    }
    add_token(p, q);
    return q;
}

// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
fn split_gpt2(cpts: const u32[:], offsets: const i64[:], out: i64[>..]&) {
    each_piece(cpts, offsets, out) { p =>
        var pos = p.ini;
        while pos < p.end {
            let c = cpt_at(p, pos);
            let f = flags_at(p, pos);
            if c == '\'' && pos + 1 < p.end {
                let next = cpt_at(p, pos + 1);
                if next == 's' || next == 't' || next == 'm' || next == 'd' {
                    pos += add_token(p, pos + 2);
                    continue;
                }
                if pos + 2 < p.end {
                    let next2 = cpt_at(p, pos + 2);
                    if (next == 'r' && next2 == 'e') || (next == 'v' && next2 == 'e') || (next == 'l' && next2 == 'l') {
                        pos += add_token(p, pos + 3);
                        continue;
                    }
                }
            }
            var f2 = if c == ' ' { flags_at(p, pos + 1) } else { f };
            let space: i64 = if c == ' ' { 1 } else { 0 };
            if is(f2, LETTER) {                                       // <space>?\p{L}+
                pos += space;
                while is(f2, LETTER) { pos += 1; f2 = flags_at(p, pos); }
                add_token(p, pos);
                continue;
            }
            if is(f2, NUMBER) {                                       // <space>?\p{N}+
                pos += space;
                while is(f2, NUMBER) { pos += 1; f2 = flags_at(p, pos); }
                add_token(p, pos);
                continue;
            }
            if !is(f2, WHITESPACE | LETTER | NUMBER) && f2 != 0 {     // <space>?[^\s\p{L}\p{N}]+
                pos += space;
                while !is(f2, WHITESPACE | LETTER | NUMBER) && f2 != 0 { pos += 1; f2 = flags_at(p, pos); }
                add_token(p, pos);
                continue;
            }
            var n = 0;
            while is(flags_at(p, pos + n), WHITESPACE) { n += 1; }
            if n > 1 && cpt_at(p, pos + n) != OUT_OF_RANGE {          // \s+(?!\S)
                pos += n - 1;
                add_token(p, pos);
                continue;
            }
            if n > 0 {                                                // \s+
                pos += n;
                add_token(p, pos);
                continue;
            }
            pos += 1;                                                 // no match
            add_token(p, pos);
        }
    };
}

// The llama3 family:
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?L+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// where L is \p{L} (llama3, qwen2), [\p{L}\p{M}] (qwen35), or \p{L}, \p{M},
// ZWNJ and ZWJ (k2-horizon). qwen2 and qwen35 take one digit at a time.
let LETTERS_L = 0;
let LETTERS_LM = 1;
let LETTERS_K2_HORIZON = 2;

fn split_llama3_family(cpts: const u32[:], offsets: const i64[:], out: i64[>..]&,
                       letters: i64, digits_one_by_one: bool) {
    each_piece(cpts, offsets, out) { p =>
        fn is_letter(pos: i64) -> bool {
            let f = flags_at(p, pos);
            if letters == LETTERS_L { return is(f, LETTER); }
            if letters == LETTERS_K2_HORIZON {
                let c = cpt_at(p, pos);
                if c == 0x200C || c == 0x200D { return true; }
            }
            return is(f, LETTER | ACCENT_MARK);
        }
        let punct_excluded: u16 = if letters == LETTERS_LM { WHITESPACE | LETTER | ACCENT_MARK | NUMBER }
                             else { WHITESPACE | LETTER | NUMBER };
        var pos = p.ini;
        while pos < p.end {
            let c = cpt_at(p, pos);
            let f = flags_at(p, pos);
            let n = contraction_len(p, pos, letters == LETTERS_K2_HORIZON);
            if n > 0 { pos += add_token(p, pos + n); continue; }

            if !(c == '\r' || c == '\n' || is(f, NUMBER)) {           // [^\r\n\p{L}\p{N}]?L+
                if is_letter(pos) || is_letter(pos + 1) {
                    pos += 1;
                    while is_letter(pos) { pos += 1; }
                    add_token(p, pos);
                    continue;
                }
            }
            if is(f, NUMBER) {
                if digits_one_by_one {                                 // \p{N}
                    pos += 1;
                    add_token(p, pos);
                } else {                                               // \p{N}{1,3}
                    pos = digits_by_three(p, pos);
                }
                continue;
            }
            let q = punct_run(p, pos, punct_excluded, f);
            if q >= 0 { pos = q; continue; }
            pos = whitespace_tail(p, pos);
        }
    };
}

// Kimi-K2, from tokenization_kimi.py:
// [\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
// |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
// |\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
fn split_kimi_k2(cpts: const u32[:], offsets: const i64[:], out: i64[>..]&) {
    each_piece(cpts, offsets, out) { p =>
        fn non_han_letter(pos: i64) -> bool { is(flags_at(p, pos), LETTER) && !is_han(cpt_at(p, pos)) }
        var pos = p.ini;
        while pos < p.end {
            let c = cpt_at(p, pos);
            let f = flags_at(p, pos);

            if is_han(c) {                                            // [\p{Han}]+
                while is_han(cpt_at(p, pos)) { pos += 1; }
                add_token(p, pos);
                continue;
            }

            let leading_ok = !(c == '\r' || c == '\n' || is(f, LETTER) || is(f, NUMBER));
            let is_letter_pattern = (is(f, LETTER) && !is_han(c)) || (leading_ok && non_han_letter(pos + 1));
            var matched = false;
            if is_letter_pattern {
                var has_leading_char = false;
                if leading_ok { has_leading_char = true; pos += 1; }
                var has_letters = false;
                while non_han_letter(pos) { has_letters = true; pos += 1; }
                if has_letters || (!has_leading_char && non_han_letter(pos)) {
                    if !has_letters { pos += 1; }
                    while non_han_letter(pos) { pos += 1; }
                    if cpt_at(p, pos) == '\'' && pos + 1 < p.end {    // optional contraction
                        let next = tolower(cpt_at(p, pos + 1));
                        if next == 's' || next == 't' || next == 'm' || next == 'd' {
                            pos += 2;
                        } else if pos + 2 < p.end {
                            let next2 = tolower(cpt_at(p, pos + 2));
                            if (next == 'r' && next2 == 'e') || (next == 'v' && next2 == 'e') || (next == 'l' && next2 == 'l') {
                                pos += 3;
                            }
                        }
                    }
                    add_token(p, pos);
                    matched = true;
                } else if has_leading_char {
                    pos -= 1;                                         // no letters after it: back off
                }
            }
            if matched { continue; }

            if is(f, NUMBER) { pos = digits_by_three(p, pos); continue; }   // \p{N}{1,3}
            let q = punct_run(p, pos, WHITESPACE | LETTER | NUMBER, f, true);
            if q >= 0 { pos = q; continue; }
            pos = whitespace_tail(p, pos);
        }
    };
}

// AFMoE (and tiny_aya) digits: a run of digits splits into a leading group
// of len % 3 digits, then groups of three. Everything else stays together.
fn split_afmoe(cpts: const u32[:], offsets: const i64[:], out: i64[>..]&) {
    each_piece(cpts, offsets, out) { p =>
        var pos = p.ini;
        while pos < p.end {
            if is(flags_at(p, pos), NUMBER) {
                let digit_start = pos;
                var digit_count = 0;
                while is(flags_at(p, pos), NUMBER) && pos < p.end { digit_count += 1; pos += 1; }
                let remainder = digit_count % 3;
                var current = digit_start;
                if remainder > 0 { add_token(p, current + remainder); current += remainder; }
                while current < digit_start + digit_count { add_token(p, current + 3); current += 3; }
                continue;
            }
            pos += 1;
        }
        if p.prev_end < p.end { add_token(p, p.end); }
    };
}

// [^\n]+|[\n]+
fn split_newlines(cpts: const u32[:], offsets: const i64[:], out: i64[>..]&) {
    var start = 0;
    for offset in offsets {
        let end = start + offset;
        var pos = start;
        while pos < end {
            let is_newline = cpts[pos] == '\n';
            let run_start = pos;
            while pos < end && (cpts[pos] == '\n') == is_newline { pos += 1; }
            out.push(pos - run_start);
        }
        start = end;
    }
}

// The fast path for regex, if there is one; false otherwise.
fn split_custom(cpts: const u32[:], regex: const u8[:], offsets: const i64[:], out: i64[>..]&) -> bool {
    if regex == "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)" {
        split_gpt2(cpts, offsets, out);
    } else if regex == "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
           || regex == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" {
        split_llama3_family(cpts, offsets, out, LETTERS_L, false);
    } else if regex == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" {
        split_llama3_family(cpts, offsets, out, LETTERS_L, true);
    } else if regex == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" {
        split_llama3_family(cpts, offsets, out, LETTERS_LM, true);
    } else if regex == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" {
        split_llama3_family(cpts, offsets, out, LETTERS_K2_HORIZON, false);
    } else if regex == "\\p{Han}+" {
        split_kimi_k2(cpts, offsets, out);       // K2's first pattern stands for all of them
    } else if regex == "\\p{AFMoE_digits}" || regex == "\\d{1,3}(?=(?:\\d{3})*\\b)" {
        split_afmoe(cpts, offsets, out);
    } else if regex == "[^\\n]+|[\\n]+" {
        split_newlines(cpts, offsets, out);
    } else {
        return false;
    }
    return true;
}

// The \p{...} classes llama.cpp knows, and the category each one stands for.
let UCAT_NAMES = ["\\p{N}", "\\p{L}", "\\p{P}", "\\p{M}", "\\p{S}",
                  "\\p{Lu}", "\\p{Ll}", "\\p{Lt}", "\\p{Lm}", "\\p{Lo}"];
let UCAT_FLAGS: u16[10] = [NUMBER, LETTER, PUNCTUATION, ACCENT_MARK, SYMBOL,
                           LETTER, LETTER, LETTER, LETTER, LETTER];

// In the collapsed text every codepoint is one code: ASCII as itself, other
// whitespace as \v, and anything else as the code of its category (0xD0 for
// none). A \p{...} in the regex becomes a class of that code and the ASCII
// characters of the category.
fn collapsed_code(category: u16) -> u32 {
    match category {
        NUMBER => 0xD1,
        LETTER => 0xD2,
        PUNCTUATION => 0xD3,
        ACCENT_MARK => 0xD4,
        SYMBOL => 0xD5,
        _ => 0xD0,
    }
}

fn collapsed_ascii(category: u16) -> const u8[:] {
    match category {
        NUMBER => "0-9",
        LETTER => "A-Za-z",
        PUNCTUATION => "!-#%-*,-/:-;?-@\\[-\\]_\\{\\}",
        SYMBOL => "\\$\\+<->^`\\|\\~",
        _ => "",
    }
}

// The index in UCAT_NAMES of the \p{...} at regex[i..], or -1.
fn ucat_at(regex: const u8[:], i: i64) -> i64 {
    guard regex[i] == '\\' && i + 3 < regex.len && regex[i + 1] == 'p' && regex[i + 2] == '{' else { return -1; }
    let closing = find(regex[i + 3..], "}");
    guard closing >= 0 && i + 3 + closing <= i + 10 else { return -1; }
    let pat = regex[i..i + 3 + closing + 1];
    for name, k in UCAT_NAMES { if pat == name { return k; } }
    return -1;
}

fn uses_ucat(regex: const u8[:]) -> bool {
    for name in UCAT_NAMES { if find(regex, name) >= 0 { return true; } }
    return false;
}

// llama.cpp's rewrite of a regex for the collapsed text, byte by byte.
fn collapse_regex(regex: const u8[:]) -> u32[>..] {
    var out: u32[>..] = [];
    var inside = false;
    var i = 0;
    while i < regex.len {
        let c = regex[i];
        if c == '[' && (i == 0 || regex[i - 1] != '\\') {
            out.push('['); inside = true; i += 1; continue;
        }
        if inside && c == ']' && regex[i - 1] != '\\' {
            out.push(']'); inside = false; i += 1; continue;
        }
        let k = ucat_at(regex, i);
        if k >= 0 {
            let category = UCAT_FLAGS[k];
            if !inside { out.push('['); }
            out.push(collapsed_code(category));
            for b in collapsed_ascii(category) { out.push(b); }
            if !inside { out.push(']'); }
            i += find(regex[i..], "}") + 1;
            continue;
        }
        out.push(c);
        i += 1;
    }
    return out;
}

// One regex split with the general matcher (llama.cpp's std::regex path).
fn split_with_regex(cpts: const u32[:], regex: const u8[:], offsets: const i64[:], out: i64[>..]&) {
    var text: u32[>..] = [];
    var pattern: u32[>..] = [];
    if uses_ucat(regex) {
        for b in regex { if b >= 0x80 { abort("Regex includes both unicode categories and non-ASCII characters - not supported"); } }
        for cpt in cpts {
            if cpt < 128 { text.push(cpt); continue; }
            let f = flags(cpt);
            text.push(if is(f, WHITESPACE) { 0x0B } else { collapsed_code(f & MASK_CATEGORIES) });
        }
        pattern.append(collapse_regex(regex));
    } else {
        for cpt in cpts {
            text.push(if cpt > 0x7F && is(flags(cpt), WHITESPACE) { 0x0B } else { cpt });
        }
        pattern.append(cpts_from_utf8(regex));
    }
    var insts: ecma::Inst[>..] = [];
    var items: ecma::Item[>..] = [];
    var classes: ecma::Class[>..] = [];
    ecma::compile(pattern, insts, items, classes);
    let prog = ecma::Prog { insts: insts, items: items, classes: classes };
    ecma::split(prog, text, offsets, out);
}

// Splits text by each regex in turn, and appends the words to `words`,
// back to back, with the end of each in `ends`. With byte_encode each byte
// is written as its GPT-2 byte-level codepoint.
fn regex_split(text: const u8[:], regexes: const u8[:][:], byte_encode: bool,
               words: u8[>..]&, ends: i64[>..]&) {
    guard text.len > 0 else { return; }
    let cpts = cpts_from_utf8(text);
    var offsets: i64[>..] = [cpts.len];
    for regex in regexes {
        var next: i64[>..] = [];
        if !split_custom(cpts, regex, offsets, next) || next.len == 0 {
            next.clear();
            split_with_regex(cpts, regex, offsets, next);
        }
        offsets.clear();
        offsets.append(next);
    }

    var start = 0;
    var word: u8[>..] = [];
    for offset in offsets {
        word.clear();
        cpts_to_utf8(word, cpts[start..start + offset]);
        if byte_encode { for b in word { byte_to_utf8(words, b); } } else { words.append(word); }
        ends.push(words.len);
        start += offset;
    }
}
