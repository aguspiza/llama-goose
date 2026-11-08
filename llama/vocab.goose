// The vocabulary and the tokenizers. One vocabulary per process: a draft
// model for speculative decoding shares its target's.
// So far SPM, BPE, UGM, RWKV and PLaMo2 tokenize; WPM, PLaMo3 and the test
// tokenizer load but abort when asked to tokenize.
// port-of: src/llama-vocab.cpp src/llama-vocab.h
import std;
import math;
import dictionary;
import os;
import ggml from "../ggml";
import ggml.gguf;
import .unicode;
import .vocab_pre;
namespace vocab;

let TOKEN_NULL = -1;

// enum llama_vocab_type
let TYPE_NONE = 0;
let TYPE_SPM = 1;
let TYPE_BPE = 2;
let TYPE_WPM = 3;
let TYPE_UGM = 4;
let TYPE_RWKV = 5;
let TYPE_PLAMO2 = 6;
let TYPE_TEST = 7;
let TYPE_PLAMO3 = 8;

// enum llama_token_type, as the GGUF stores it
let TOKTYPE_UNDEFINED = 0;
let TOKTYPE_NORMAL = 1;
let TOKTYPE_UNKNOWN = 2;
let TOKTYPE_CONTROL = 3;
let TOKTYPE_USER_DEFINED = 4;
let TOKTYPE_UNUSED = 5;
let TOKTYPE_BYTE = 6;

// enum llama_token_attr
let ATTR_UNDEFINED = 0;
let ATTR_UNKNOWN = 1;
let ATTR_UNUSED = 2;
let ATTR_NORMAL = 4;
let ATTR_CONTROL = 8;
let ATTR_USER_DEFINED = 16;
let ATTR_BYTE = 32;
let ATTR_NORMALIZED = 64;
let ATTR_LSTRIP = 128;
let ATTR_RSTRIP = 256;
let ATTR_SINGLE_WORD = 512;

// enum llama_metaspace_prepend_scheme
let METASPACE_NEVER = 0;
let METASPACE_FIRST = 1;
let METASPACE_ALWAYS = 2;

var tokenizer_model: u8[>..];
var tokenizer_pre: u8[>..];
var vtype = TYPE_SPM;
var pre_type = PRE_DEFAULT;
var n_token_types: u32 = 0;
var max_token_len = 0;

// default LLaMA special tokens
var special_bos_id: i32 = 1;
var special_eos_id: i32 = 2;
var special_eot_id: i32 = TOKEN_NULL;
var special_eom_id: i32 = TOKEN_NULL;
var special_unk_id: i32 = 0;
var special_sep_id: i32 = TOKEN_NULL;
var special_pad_id: i32 = TOKEN_NULL;
var special_mask_id: i32 = TOKEN_NULL;
var linefeed_id: i32 = 13;
var special_fim_pre_id: i32 = TOKEN_NULL;
var special_fim_suf_id: i32 = TOKEN_NULL;
var special_fim_mid_id: i32 = TOKEN_NULL;
var special_fim_pad_id: i32 = TOKEN_NULL;
var special_fim_rep_id: i32 = TOKEN_NULL;
var special_fim_sep_id: i32 = TOKEN_NULL;

var add_space_prefix = false;
var add_bos = false;
var add_eos = false;
var add_sep = false;
var dna_auto_tags = false;
var ignore_merges = false;
var clean_spaces = false;
var remove_extra_whitespaces = false;
var escape_whitespaces = true;
var treat_whitespace_as_suffix = false;
var normalizer_nfc = false;
var normalizer_lowercase = true;
var normalizer_strip_accents = true;
var metaspace_prepend_scheme = METASPACE_NEVER;

// The tokens: token i's text is text_buf[text_end[i - 1]..text_end[i]].
var text_buf: u8[>..];
var text_end: i64[>..];
var scores: f32[>..];
var attrs: i32[>..];
var token_to_id = dictionary<const u8[:], i32> {};

// BPE merges, keyed by their "left right" text.
var merges_buf: u8[>..];
var bpe_ranks = dictionary<const u8[:], i32> {};

var eog_ids: i32[>..];            // sorted
var suppress_tokens: i32[>..];
var precompiled_charsmap: u8[>..];
var cache_special_tokens: i32[>..];
// token_to_piece(special = true) of every token: piece i is piece_buf[piece_end[i - 1]..piece_end[i]].
var piece_buf: u8[>..];
var piece_end: i64[>..];

// The BPE pre-tokenizer's regexes.
var bpe_regexes: const (const u8[:])[:] = PRE_REGEXES_0;
var bpe_byte_encode = true;

fn n_tokens() -> i64 { text_end.len }

fn token_text(id: i64) -> const u8[:] {
    let start = if id == 0 { 0 } else { text_end[id - 1] };
    return text_buf[start..text_end[id]];
}

fn token_piece(id: i64) -> const u8[:] {
    let start = if id == 0 { 0 } else { piece_end[id - 1] };
    return piece_buf[start..piece_end[id]];
}

fn text_to_token(text: const u8[:]) -> i32 {
    let id = token_to_id.get(text);
    return if id { id } else { TOKEN_NULL };
}

fn has_attr(id: i64, a: i32) -> bool { attrs[id] & a != 0 }
fn is_normal(id: i64) -> bool { has_attr(id, ATTR_NORMAL) }
fn is_unknown(id: i64) -> bool { has_attr(id, ATTR_UNKNOWN) }
fn is_control(id: i64) -> bool { has_attr(id, ATTR_CONTROL) }
fn is_byte(id: i64) -> bool { has_attr(id, ATTR_BYTE) }
fn is_user_defined(id: i64) -> bool { has_attr(id, ATTR_USER_DEFINED) }
fn is_unused(id: i64) -> bool { has_attr(id, ATTR_UNUSED) }

fn is_eog(id: i32) -> bool {
    if id == TOKEN_NULL { return false; }
    let i, found = binary_search(eog_ids, id);
    return found;
}

fn eog_insert(id: i32) {
    if !is_eog(id) { eog_ids.push(id); sort(eog_ids); }
}

fn eog_erase(id: i32) {
    let i, found = binary_search(eog_ids, id);
    guard found else { return; }
    for k in i..eog_ids.len - 1 { eog_ids[k] = eog_ids[k + 1]; }
    eog_ids.pop();
}

fn type_name() -> const u8[:] {
    match vtype {
        TYPE_NONE => "no vocab",
        TYPE_SPM => "SPM",
        TYPE_BPE => "BPE",
        TYPE_WPM => "WPM",
        TYPE_UGM => "UGM",
        TYPE_RWKV => "RWKV",
        TYPE_PLAMO2 => "PLaMo2",
        TYPE_PLAMO3 => "PLaMo3",
        TYPE_TEST => "TEST",
        _ => "unknown",
    }
}

fn hex_digit(v: i64) -> u8 { "0123456789ABCDEF"[v] }

fn hex_value(c: u8) -> i64 {
    if c >= '0' && c <= '9' { return c - '0'; }
    if c >= 'a' && c <= 'f' { return c - 'a' + 10; }
    if c >= 'A' && c <= 'F' { return c - 'A' + 10; }
    return -1;
}

// <0xXX> for a byte.
fn byte_token_text(out: u8[..6]&, ch: u8) {
    out.append("<0x");
    out.push(hex_digit(ch >> 4));
    out.push(hex_digit(ch & 15));
    out.push('>');
}

fn token_to_byte(id: i64) -> u8 {
    assert(vtype != TYPE_NONE && is_byte(id));
    guard vtype == TYPE_SPM || vtype == TYPE_UGM || vtype == TYPE_BPE else { abort("fatal error"); }
    // strtol of the two characters after "<0x"
    let t = token_text(id);
    var v = 0;
    for c in t[3..min(5, t.len)] {
        let h = hex_value(c);
        if h < 0 { break; }
        v = v * 16 + h;
    }
    return v as! u8;
}

fn byte_to_token(ch: u8) -> i32 {
    assert(vtype != TYPE_NONE);
    match vtype {
        TYPE_SPM, TYPE_UGM => {
            var buf: u8[..6] = [];
            byte_token_text(buf, ch);
            let id = text_to_token(buf);
            if id != TOKEN_NULL { return id; }
            let id2 = text_to_token([ch]);
            if id2 == TOKEN_NULL { abort("byte_to_token: no token for the byte"); }
            return id2;
        },
        TYPE_WPM, TYPE_BPE => {
            var buf: u8[>..] = [];
            unicode::byte_to_utf8(buf, ch);
            let id = text_to_token(buf);
            if id == TOKEN_NULL { abort("byte_to_token: no token for the byte"); }
            return id;
        },
        TYPE_PLAMO2, TYPE_PLAMO3 => {
            var buf: u8[..6] = [];
            byte_token_text(buf, ch);
            let id = text_to_token(buf);
            if id == TOKEN_NULL { abort("byte_to_token: no token for the byte"); }
            return id;
        },
        TYPE_TEST => { return TOKEN_NULL; },
        _ => { abort("fatal error"); },
    }
}

// The rank of the merge left+right, or -1.
fn find_bpe_rank(left: const u8[:], right: const u8[:], scratch: u8[>..]&) -> i32 {
    assert(find(left, " ") < 0 && find(right, " ") < 0);
    scratch.clear();
    scratch.append(left);
    scratch.push(' ');
    scratch.append(right);
    let r = bpe_ranks.get(scratch);
    return if r { r } else { -1 };
}

//
// Loading
//

fn key_type_name(t: i32) -> const u8[:] {
    match t {
        gguf::UINT8 => "u8", gguf::INT8 => "i8", gguf::UINT16 => "u16", gguf::INT16 => "i16",
        gguf::UINT32 => "u32", gguf::INT32 => "i32", gguf::FLOAT32 => "f32", gguf::BOOL => "bool",
        gguf::STRING => "str", gguf::ARRAY => "arr", gguf::UINT64 => "u64", gguf::INT64 => "i64",
        gguf::FLOAT64 => "f64", _ => "unknown",
    }
}

// The key's id if it is there with type t; -1 if it is not there. A key of
// another type aborts, as llama.cpp's loader throws.
fn key_of_type(ctx: gguf::Ctx, key: const u8[:], t: i32) -> i64 {
    let id = gguf::find_key(ctx, key);
    guard id >= 0 else { return -1; }
    let kt = gguf::get_kv_type(ctx, id);
    guard kt == t else {
        abort(str("key ", key, " has wrong type ", key_type_name(kt), " but expected type ", key_type_name(t)));
    }
    return id;
}

fn get_str(ctx: gguf::Ctx, key: const u8[:], out: u8[>..]&) -> bool {
    let id = key_of_type(ctx, key, gguf::STRING);
    guard id >= 0 else { return false; }
    out.clear();
    gguf::get_val_str(ctx, id, out);
    return true;
}

fn get_u32(ctx: gguf::Ctx, key: const u8[:], v: u32&) -> bool {
    let id = key_of_type(ctx, key, gguf::UINT32);
    guard id >= 0 else { return false; }
    v = gguf::get_val_u32(ctx, id);
    return true;
}

fn get_bool(ctx: gguf::Ctx, key: const u8[:], v: bool&) -> bool {
    let id = key_of_type(ctx, key, gguf::BOOL);
    guard id >= 0 else { return false; }
    v = gguf::get_val_bool(ctx, id);
    return true;
}

// An array key of element type t: its id, or -1 if absent; aborts on another type.
fn array_key(ctx: gguf::Ctx, key: const u8[:], t: i32) -> i64 {
    let id = gguf::find_key(ctx, key);
    guard id >= 0 else { return -1; }
    if gguf::get_kv_type(ctx, id) != gguf::ARRAY || gguf::get_arr_type(ctx, id) != t {
        abort(str("invalid gguf type for ", key));
    }
    return id;
}

fn read_merges(ctx: gguf::Ctx, merges_id: i64) {
    let n = gguf::get_arr_n(ctx, merges_id);
    var ends: i64[>..] = [];
    for i in n {
        gguf::get_arr_str(ctx, merges_id, i, merges_buf);
        ends.push(merges_buf.len);
    }
    var start = 0;
    for e, i in ends {
        let word = merges_buf[start..e];
        // llama.cpp splits at the first space after the first character; a
        // merge without one is ("", ""), which no lookup finds.
        let sep = if word.len > 0 { find(word[1..], " ") } else { -1 };
        let key = if sep >= 0 { word } else { merges_buf[start..start] };
        if !bpe_ranks.contains(key) { bpe_ranks.insert(key, i as i32); }
        start = e;
    }
}

// Reads the vocabulary from a model's metadata.
fn load(ctx: gguf::Ctx) {
    // the vocab type
    if !get_str(ctx, "tokenizer.ggml.model", tokenizer_model) { abort("key not found in model: tokenizer.ggml.model"); }
    get_str(ctx, "tokenizer.ggml.pre", tokenizer_pre);
    get_u32(ctx, "tokenizer.ggml.token_type_count", n_token_types);

    if tokenizer_model == "no_vocab" || tokenizer_model == "none" {
        vtype = TYPE_NONE;
        special_bos_id = TOKEN_NULL;
        special_eos_id = TOKEN_NULL;
        special_unk_id = TOKEN_NULL;
        special_sep_id = TOKEN_NULL;
        special_pad_id = TOKEN_NULL;
        special_mask_id = TOKEN_NULL;
        linefeed_id = TOKEN_NULL;
        // read vocab size from metadata
        var arch: u8[>..] = [];
        get_str(ctx, "general.architecture", arch);
        var n: u32 = 0;
        if get_u32(ctx, str(arch, ".vocab_size"), n) {
            log(str("load: adding ", n, " dummy tokens"));
            for i in n { text_end.push(0); scores.push(0.0); attrs.push(ATTR_UNDEFINED); }
        }
        return;
    }

    if tokenizer_model == "llama" {
        vtype = TYPE_SPM;
        special_bos_id = 1; special_eos_id = 2; special_unk_id = 0;
        special_sep_id = TOKEN_NULL; special_pad_id = TOKEN_NULL; special_mask_id = TOKEN_NULL;
    } else if tokenizer_model == "bert" {
        vtype = TYPE_WPM;
        special_bos_id = 101; special_eos_id = TOKEN_NULL; special_unk_id = 100;
        special_sep_id = 102; special_pad_id = 0; special_mask_id = 103;
        add_sep = true;
    } else if tokenizer_model == "gpt2" || tokenizer_model == "hybriddna" || tokenizer_model == "whitespace" {
        vtype = TYPE_BPE;
        let merges_id = gguf::find_key(ctx, "tokenizer.ggml.merges");
        if merges_id == -1 {
            // Kimi-K2 uses custom tokenization without BPE merges
            if tokenizer_pre != "kimi-k2" { abort("cannot find tokenizer merges in model file"); }
            log(str("load: Kimi-K2 tokenizer detected, skipping BPE merges"));
        } else {
            array_key(ctx, "tokenizer.ggml.merges", gguf::STRING);
            read_merges(ctx, merges_id);
        }
        special_bos_id = 11; special_eos_id = 11; special_unk_id = TOKEN_NULL;
        special_sep_id = TOKEN_NULL; special_pad_id = TOKEN_NULL; special_mask_id = TOKEN_NULL;
    } else if tokenizer_model == "t5" {
        vtype = TYPE_UGM;
        special_bos_id = TOKEN_NULL; special_eos_id = 1; special_unk_id = 2;
        special_sep_id = TOKEN_NULL; special_pad_id = 0; special_mask_id = TOKEN_NULL;
        let id = gguf::find_key(ctx, "tokenizer.ggml.precompiled_charsmap");
        if id != -1 {
            let at = gguf::get_arr_type(ctx, id);
            if gguf::get_kv_type(ctx, id) != gguf::ARRAY || (at != gguf::INT8 && at != gguf::UINT8) {
                abort("invalid gguf type for tokenizer.ggml.precompiled_charsmap");
            }
            precompiled_charsmap.resize(gguf::get_arr_n(ctx, id), 0);
            gguf::get_arr_u8(ctx, id, precompiled_charsmap);
            guard precompiled_charsmap.len >= 4 else { abort("precompiled_charsmap too small for xcda_blob_size header!"); }
            let blob_size = precompiled_charsmap[0] | (precompiled_charsmap[1] as i64) << 8 |
                            (precompiled_charsmap[2] as i64) << 16 | (precompiled_charsmap[3] as i64) << 24;
            guard blob_size + 4 < precompiled_charsmap.len else { abort("Index out of array bounds in precompiled charsmap!"); }
        }
    } else if tokenizer_model == "rwkv" {
        vtype = TYPE_RWKV;
        special_bos_id = TOKEN_NULL; special_eos_id = TOKEN_NULL; special_unk_id = TOKEN_NULL;
        special_sep_id = TOKEN_NULL; special_pad_id = TOKEN_NULL;
    } else if tokenizer_model == "test" {
        vtype = TYPE_TEST;
        special_bos_id = TOKEN_NULL; special_eos_id = TOKEN_NULL; special_unk_id = TOKEN_NULL;
        special_sep_id = TOKEN_NULL; special_pad_id = TOKEN_NULL; special_mask_id = TOKEN_NULL;
    } else if tokenizer_model == "plamo2" || tokenizer_model == "plamo3" {
        vtype = if tokenizer_model == "plamo2" { TYPE_PLAMO2 } else { TYPE_PLAMO3 };
        special_bos_id = 1; special_eos_id = 2; special_unk_id = 0;
        special_sep_id = TOKEN_NULL; special_pad_id = 3; special_mask_id = TOKEN_NULL;
    } else if tokenizer_model == "gemma4" {
        vtype = TYPE_BPE;
        let merges_id = gguf::find_key(ctx, "tokenizer.ggml.merges");
        if merges_id == -1 { abort("cannot find tokenizer merges in model file"); }
        array_key(ctx, "tokenizer.ggml.merges", gguf::STRING);
        read_merges(ctx, merges_id);
        special_bos_id = TOKEN_NULL; special_eos_id = TOKEN_NULL; special_unk_id = TOKEN_NULL;
        special_sep_id = TOKEN_NULL; special_pad_id = TOKEN_NULL; special_mask_id = TOKEN_NULL;
        tokenizer_pre.clear();
        tokenizer_pre.append("gemma4");
    } else {
        abort(str("unknown tokenizer: '", tokenizer_model, "'"));
    }

    // for now, only BPE models have pre-tokenizers
    if vtype == TYPE_BPE {
        add_space_prefix = false;
        escape_whitespaces = false;
        clean_spaces = true;
        if tokenizer_pre.len == 0 {
            log(str("load: missing pre-tokenizer type, using: 'default'"));
            log(str("load: GENERATION QUALITY WILL BE DEGRADED! CONSIDER REGENERATING THE MODEL"));
            pre_type = PRE_DEFAULT;
        } else if !apply_pre_name(tokenizer_pre) {
            abort(str("unknown pre-tokenizer type: '", tokenizer_pre, "'"));
        }
    } else if vtype == TYPE_SPM {
        pre_type = PRE_DEFAULT;
        add_space_prefix = true; clean_spaces = false; add_bos = true; add_eos = false;
    } else if vtype == TYPE_WPM {
        pre_type = PRE_DEFAULT;
        add_space_prefix = false; clean_spaces = true; add_bos = true; add_eos = false; add_sep = true;
    } else if vtype == TYPE_UGM {
        pre_type = PRE_DEFAULT;
        add_bos = false; add_eos = true;
    } else if vtype == TYPE_RWKV {
        pre_type = PRE_DEFAULT;
        add_space_prefix = false; clean_spaces = false; add_bos = false; add_eos = false;
    } else {
        pre_type = PRE_DEFAULT;
    }

    get_bool(ctx, "tokenizer.ggml.add_space_prefix", add_space_prefix);
    get_bool(ctx, "tokenizer.ggml.remove_extra_whitespaces", remove_extra_whitespaces);
    get_bool(ctx, "tokenizer.ggml.clean_spaces", clean_spaces);

    load_tokens(ctx);
    init_tokenizer();

    // the newline token: LLaMA "<0x0A>" == 10 == '\n', Falcon 193 == '\n'
    if vtype == TYPE_SPM {
        var buf: u8[..6] = [];
        byte_token_text(buf, '\n');
        let id = text_to_token(buf);
        let id2 = text_to_token("\n");
        if id != TOKEN_NULL { linefeed_id = id; }
        else if id2 != TOKEN_NULL { linefeed_id = id2; }
        else {
            log(str("load: SPM vocabulary, but newline token not found! Using special_pad_id instead."));
            linefeed_id = special_pad_id;
        }
    } else if vtype == TYPE_WPM {
        linefeed_id = special_pad_id;
    } else if vtype == TYPE_RWKV {
        var ids: i32[>..] = [];
        tokenize("\n", false, false, ids);
        guard ids.len > 0 else { abort("GGML_ASSERT(!ids.empty() && \"model vocab missing newline token\") failed"); }
        linefeed_id = ids[0];
    } else if vtype == TYPE_BPE || vtype == TYPE_PLAMO2 || vtype == TYPE_PLAMO3 || vtype == TYPE_UGM {
        // (llama.cpp tokenizes "\n" for the other types too, which are not ported yet)
        var ids: i32[>..] = [];
        tokenize("\n", false, false, ids);
        if ids.len == 0 {
            log(str("load: model vocab missing newline token, using special_pad_id instead"));
            linefeed_id = special_pad_id;
        } else {
            linefeed_id = ids[0];
        }
    }

    load_special_tokens(ctx);
    build_caches();
    set_attributes_by_name(ctx);
}

fn log(msg: const u8[:]) { if !llama::quiet { write_stderr(msg); write_stderr("\n"); } }

fn load_tokens(ctx: gguf::Ctx) {
    let token_idx = gguf::find_key(ctx, "tokenizer.ggml.tokens");
    if token_idx == -1 { abort("cannot find tokenizer vocab in model file"); }
    array_key(ctx, "tokenizer.ggml.tokens", gguf::STRING);
    let n = gguf::get_arr_n(ctx, token_idx);

    var score_values: f32[>..] = [];
    let score_idx = gguf::find_key(ctx, "tokenizer.ggml.scores");
    if score_idx != -1 {
        let at = if gguf::get_kv_type(ctx, score_idx) == gguf::ARRAY { gguf::get_arr_type(ctx, score_idx) } else { -1 };
        if at != gguf::INT32 && at != gguf::FLOAT32 { abort("invalid gguf type for tokenizer.ggml.scores"); }
        let n_scores = gguf::get_arr_n(ctx, score_idx);
        if n_scores < n {
            abort(str("Index out of array bounds for scores (", n_scores, " < ", n, ")"));
        }
        score_values.resize(n, 0.0);
        if at == gguf::INT32 {
            var ints: i32[>..] = [];
            ints.resize(n, 0);
            gguf::get_arr_i32(ctx, score_idx, ints);
            for v, i in ints { score_values[i] = v; }
        } else {
            gguf::get_arr_f32(ctx, score_idx, score_values);
        }
    }

    var toktypes: i32[>..] = [];
    let toktype_idx = array_key(ctx, "tokenizer.ggml.token_type", gguf::INT32);
    if toktype_idx != -1 {
        let n_toktypes = gguf::get_arr_n(ctx, toktype_idx);
        if n_toktypes < n {
            abort(str("Index out of array bounds for toktypes (", n_toktypes, " < ", n, ")"));
        }
        toktypes.resize(n, 0);
        gguf::get_arr_i32(ctx, toktype_idx, toktypes);
    }

    for i in n {
        let start = text_buf.len;
        gguf::get_arr_str(ctx, token_idx, i, text_buf);
        if text_buf.len == start {
            log(str("load_tokens: empty token at index ", i));
            format(text_buf, "[EMPTY_", i, "]");
        }
        text_end.push(text_buf.len);
        max_token_len = max(max_token_len, text_buf.len - start);
        scores.push(if score_values.len > 0 { score_values[i] } else { 0.0 });
        var attr: i32 = ATTR_NORMAL;
        if toktypes.len > 0 {
            attr = match toktypes[i] {
                TOKTYPE_UNKNOWN => ATTR_UNKNOWN,
                TOKTYPE_UNUSED => ATTR_UNUSED,
                TOKTYPE_NORMAL => ATTR_NORMAL,
                TOKTYPE_CONTROL => ATTR_CONTROL,
                TOKTYPE_USER_DEFINED => ATTR_USER_DEFINED,
                TOKTYPE_BYTE => ATTR_BYTE,
                _ => ATTR_UNDEFINED,
            };
        }
        attrs.push(attr);
    }
    // token_to_id[word] = i: a repeated text maps to its last id
    for i in n { token_to_id.insert(token_text(i), i as i32); }
    if token_to_id.count != n { abort("GGML_ASSERT(id_to_token.size() == token_to_id.size()) failed"); }
    if tokenizer_model == "hybriddna" { abort("hybriddna vocab: not ported yet"); }
}

fn init_tokenizer() {
    match vtype {
        TYPE_SPM, TYPE_WPM, TYPE_TEST => {},
        TYPE_UGM => ugm_build(),
        TYPE_RWKV => rwkv_build(),
        TYPE_PLAMO2, TYPE_PLAMO3 => plamo_build(),
        TYPE_BPE => {
            let regexes, byte_encode = pre_regexes(pre_type);
            bpe_regexes = regexes;
            bpe_byte_encode = byte_encode;
        },
        _ => { abort("unsupported vocab type"); },
    }
}

// A special token id from the GGUF, where it gives one in range.
fn read_special(ctx: gguf::Ctx, key: const u8[:], id: i32&) {
    if id >= 0 && id >= n_tokens() {
        log(str("load: default special token '", key, "' = ", id, " out of vocab range, disabling"));
        id = TOKEN_NULL;
    }
    var new_id: u32 = 0;
    guard get_u32(ctx, key, new_id) else { return; }
    if new_id >= n_tokens() {
        log(str("load: bad special token: '", key, "' = ", new_id, ", using default id ", id));
    } else {
        id = new_id as i32;
    }
}

fn mark_control(id: i64, text: const u8[:]) {
    if attrs[id] & ATTR_CONTROL == 0 {
        log(str("load: control-looking token: ", id, " '", text,
                  "' was not control-type; this is probably a bug in the model. its type will be overridden"));
        attrs[id] |= ATTR_CONTROL;
    }
}

fn is_one_of(text: const u8[:], names: const u8[:][:]) -> bool {
    for n in names { if text == n { return true; } }
    return false;
}

fn load_special_tokens(ctx: gguf::Ctx) {
    read_special(ctx, "tokenizer.ggml.bos_token_id", special_bos_id);
    read_special(ctx, "tokenizer.ggml.eos_token_id", special_eos_id);
    read_special(ctx, "tokenizer.ggml.eot_token_id", special_eot_id);
    read_special(ctx, "tokenizer.ggml.eom_token_id", special_eom_id);
    read_special(ctx, "tokenizer.ggml.unknown_token_id", special_unk_id);
    read_special(ctx, "tokenizer.ggml.seperator_token_id", special_sep_id);
    read_special(ctx, "tokenizer.ggml.padding_token_id", special_pad_id);
    read_special(ctx, "tokenizer.ggml.mask_token_id", special_mask_id);
    read_special(ctx, "tokenizer.ggml.fim_pre_token_id", special_fim_pre_id);
    read_special(ctx, "tokenizer.ggml.fim_suf_token_id", special_fim_suf_id);
    read_special(ctx, "tokenizer.ggml.fim_mid_token_id", special_fim_mid_id);
    read_special(ctx, "tokenizer.ggml.fim_pad_token_id", special_fim_pad_id);
    read_special(ctx, "tokenizer.ggml.fim_rep_token_id", special_fim_rep_id);
    read_special(ctx, "tokenizer.ggml.fim_sep_token_id", special_fim_sep_id);
    // deprecated
    read_special(ctx, "tokenizer.ggml.prefix_token_id", special_fim_pre_id);
    read_special(ctx, "tokenizer.ggml.suffix_token_id", special_fim_suf_id);
    read_special(ctx, "tokenizer.ggml.middle_token_id", special_fim_mid_id);

    var flag = true;
    if get_bool(ctx, "tokenizer.ggml.add_bos_token", flag) { add_bos = flag; }
    if get_bool(ctx, "tokenizer.ggml.add_eos_token", flag) { add_eos = flag; }
    if get_bool(ctx, "tokenizer.ggml.add_sep_token", flag) { add_sep = flag; }
    if get_bool(ctx, "tokenizer.ggml.dna_auto_tags", flag) { dna_auto_tags = flag; }
    // workaround for Gemma 4, ref: https://github.com/ggml-org/llama.cpp/pull/21500
    if pre_type == PRE_GEMMA4 && !add_bos {
        add_bos = true;
        log(str("load: override 'tokenizer.ggml.add_bos_token' to 'true' for Gemma4"));
    }

    get_bool(ctx, "tokenizer.ggml.normalizer.nfc", normalizer_nfc);
    get_bool(ctx, "tokenizer.ggml.normalizer.lowercase", normalizer_lowercase);
    normalizer_strip_accents = normalizer_lowercase;
    get_bool(ctx, "tokenizer.ggml.normalizer.strip_accents", normalizer_strip_accents);

    {
        var scheme: u8[>..] = [];
        if get_str(ctx, "tokenizer.ggml.metaspace.prepend_scheme", scheme) {
            if scheme == "never" { metaspace_prepend_scheme = METASPACE_NEVER; }
            else if scheme == "first" { metaspace_prepend_scheme = METASPACE_FIRST; }
            else if scheme == "always" { metaspace_prepend_scheme = METASPACE_ALWAYS; }
            else { abort(str("unknown metaspace prepend_scheme: '", scheme, "'")); }
            if vtype == TYPE_BPE && metaspace_prepend_scheme == METASPACE_ALWAYS {
                abort("metaspace prepend_scheme 'always' is not supported for BPE vocab");
            }
        }
    }

    {
        let id = array_key(ctx, "tokenizer.ggml.suppress_tokens", gguf::INT32);
        if id != -1 {
            var data: i32[>..] = [];
            data.resize(gguf::get_arr_n(ctx, id), 0);
            gguf::get_arr_i32(ctx, id, data);
            for t in data { if t >= 0 && t < n_tokens() { suppress_tokens.push(t); } }
        }
    }

    // Auto-detect special tokens by their text. llama.cpp walks its
    // unordered_map of texts, whose order is its C++ library's; this walks
    // the ids in order. They differ only where a vocab has two candidates
    // for the same role and its metadata names neither.
    let eot_names = ["<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>",
                     "<|end_of_text|>", "<EOT>", "_<EOT>", "[EOT]", "<｜end▁of▁sentence｜>", "<end_of_utterance>"];
    let fim_pre_names = ["<|fim_prefix|>", "<fim-prefix>", "<fim_prefix>", "<｜fim▁begin｜>", "<PRE>", "▁<PRE>",
                         "<|code_prefix|>", "<|prefix|>", "<|plamo:fim_prefix|>"];
    let fim_suf_names = ["<|fim_suffix|>", "<fim-suffix>", "<fim_suffix>", "<｜fim▁hole｜>", "<SUF>", "▁<SUF>",
                         "<|code_suffix|>", "<|suffix|>", "<|plamo:fim_suffix|>"];
    let fim_mid_names = ["<|fim_middle|>", "<fim-middle>", "<fim_middle>", "<｜fim▁end｜>", "<MID>", "▁<MID>",
                         "<|code_middle|>", "<|middle|>", "<|plamo:fim_middle|>"];
    let fim_pad_names = ["<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]"];
    let fim_rep_names = ["<|fim_repo|>", "<|repo_name|>", "<fim-repo>", "<REPO>", "<reponame>"];
    let fim_sep_names = ["<|file_sep|>", "<|plamo:file_separator|>"];
    for id in n_tokens() {
        let t = token_text(id);
        let id32 = id as i32;
        if special_eot_id == TOKEN_NULL && is_one_of(t, eot_names) { special_eot_id = id32; mark_control(id, t); }
        if special_eom_id == TOKEN_NULL && t == "<|eom_id|>" { special_eom_id = id32; mark_control(id, t); }
        if special_fim_pre_id == TOKEN_NULL && is_one_of(t, fim_pre_names) { special_fim_pre_id = id32; mark_control(id, t); }
        if special_fim_suf_id == TOKEN_NULL && is_one_of(t, fim_suf_names) { special_fim_suf_id = id32; mark_control(id, t); }
        if special_fim_mid_id == TOKEN_NULL && is_one_of(t, fim_mid_names) { special_fim_mid_id = id32; mark_control(id, t); }
        if special_fim_pad_id == TOKEN_NULL && is_one_of(t, fim_pad_names) { special_fim_pad_id = id32; mark_control(id, t); }
        if special_fim_rep_id == TOKEN_NULL && is_one_of(t, fim_rep_names) { special_fim_rep_id = id32; mark_control(id, t); }
        if special_fim_sep_id == TOKEN_NULL && is_one_of(t, fim_sep_names) { special_fim_sep_id = id32; mark_control(id, t); }
    }

    // auto-detect unused tokens: control tokens with the word "unused"
    {
        var n_unused = 0;
        for id in n_tokens() {
            if attrs[id] & ATTR_CONTROL == 0 { continue; }
            if attrs[id] & ATTR_UNUSED == 0 && find(token_text(id), "unused") >= 0 { attrs[id] |= ATTR_UNUSED; }
            if attrs[id] & ATTR_UNUSED != 0 { n_unused += 1; }
        }
        log(str("load: ", n_unused, " unused tokens"));
    }

    // the tokens that end generation, found by their text
    eog_ids.clear();
    if special_fim_pad_id != TOKEN_NULL { eog_insert(special_fim_pad_id); }
    if special_fim_rep_id != TOKEN_NULL { eog_insert(special_fim_rep_id); }
    if special_fim_sep_id != TOKEN_NULL { eog_insert(special_fim_sep_id); }

    let eog_names = ["<|eot_id|>", "<|eot|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>", "<|flush|>",
                     "<|calls|>", "<end_of_turn>", "<|endoftext|>", "</s>", "<|eom_id|>", "<EOT>", "_<EOT>",
                     "[EOT]", "[EOS]", "<|end_of_text|>", "<end_of_utterance>", "<eos>", "<turn|>",
                     "<|tool_response>", "<｜end▁of▁sentence｜>", "[e~[", "<|ifm|im_end|>"];
    for id in n_tokens() {
        let t = token_text(id);
        if is_one_of(t, eog_names) || (pre_type == PRE_FLINT && (t == "<|user|>" || t == "<|observation|>")) {
            eog_insert(id as i32);
            mark_control(id, t);
        }
    }

    // a quick hack for gpt-oss: always render these tokens
    for id in n_tokens() {
        let t = token_text(id);
        if t == "<|channel|>" || t == "<|message|>" || t == "<|start|>" || t == "<|constrain|>" {
            log(str("load: setting token '", t, "' (", id, ") attribute to USER_DEFINED (", ATTR_USER_DEFINED,
                      "), old attributes: ", attrs[id]));
            attrs[id] = ATTR_USER_DEFINED;
        }
    }

    if pre_type == PRE_SOPRANO { eog_insert(special_eos_id); }

    // sanity checks
    if special_eos_id != TOKEN_NULL && !is_eog(special_eos_id) {
        eog_insert(special_eos_id);
        log(str("load: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect"));
    }
    if special_eot_id != TOKEN_NULL && !is_eog(special_eot_id) {
        eog_insert(special_eot_id);
        log(str("load: special_eot_id is not in special_eog_ids - the tokenizer config may be incorrect"));
    }
    if special_eom_id != TOKEN_NULL && !is_eog(special_eom_id) {
        eog_insert(special_eom_id);
        log(str("load: special_eom_id is not in special_eog_ids - the tokenizer config may be incorrect"));
    }

    // o200k_harmony and solar-open: "<|end|>" is not EOG when the vocab also
    // has "<|return|>" and "<|call|>" ("<|calls|>" and "<|flush|>" for solar-open)
    {
        var has_return = false;
        var has_call = false;
        var has_end = false;
        var has_flush = false;
        var end_id: i32 = TOKEN_NULL;
        for tid in eog_ids {
            if tid < 0 || tid >= n_tokens() {
                log(str("load: EOG token id ", tid, " is out of range (vocab size ", n_tokens(), "), skipping"));
                continue;
            }
            let t = token_text(tid);
            if t == "<|return|>" { has_return = true; }
            else if t == "<|call|>" || t == "<|calls|>" { has_call = true; }
            else if t == "<|flush|>" { has_flush = true; }
            else if t == "<|end|>" { has_end = true; end_id = tid; }
        }
        if (has_return && has_call && has_end) || (has_call && has_flush && has_end) {
            eog_erase(end_id);
            attrs[end_id] = ATTR_USER_DEFINED;
            log(str("load: special_eog_ids contains both '<|return|>' and '<|call|>', or '<|calls|>' and '<|flush|>' tokens, removing '<|end|>' token from EOG list"));
        }
    }

    // gemma4, plamo and beam have a normal </s> token, unlike paddleocr
    {
        var has_normal_s_marker = token_to_id.contains("<rfl:parameter name=\"");
        var has_s = false;
        var s_id: i32 = TOKEN_NULL;
        for tid in eog_ids {
            if tid < 0 || tid >= n_tokens() { continue; }
            let t = token_text(tid);
            if t == "<|tool_response>" || t == "<|plamo:eos|>" { has_normal_s_marker = true; }
            else if t == "</s>" { has_s = true; s_id = tid; }
        }
        if has_normal_s_marker && has_s {
            eog_erase(s_id);
            attrs[s_id] = ATTR_NORMAL;
            log(str("load: '</s>' is a normal token here, removing it from EOG list"));
        }
    }
}

fn build_caches() {
    // the special tokens, longest text first
    for id in n_tokens() {
        if attrs[id] & (ATTR_CONTROL | ATTR_USER_DEFINED | ATTR_UNKNOWN) != 0 { cache_special_tokens.push(id as i32); }
    }
    stable_sort(cache_special_tokens) { a, b => token_text(a).len > token_text(b).len };

    var size_cache = 0;
    for id in n_tokens() {
        token_to_piece(id as i32, piece_buf, 0, true, false);
        piece_end.push(piece_buf.len);
    }
    size_cache = piece_buf.len;
    log(str("load: special tokens cache size = ", cache_special_tokens.len));
    log(str("load: token to piece cache size = ", size_cache));
}

fn contains_any(s: const u8[:], subs: const u8[:][:]) -> bool {
    for sub in subs { if find(s, sub) >= 0 { return true; } }
    return false;
}

fn set_token_attr(text: const u8[:], attr: i32, value: bool) {
    let id = token_to_id.get(text);
    guard id else { abort(str("token not found: ", text)); }
    let i: i64 = id;
    attrs[i] = if value { attrs[i] | attr } else { attrs[i] & ~attr };
}

// Per-token attributes the GGUF does not carry, set by model name.
fn set_attributes_by_name(ctx: gguf::Ctx) {
    var model_name: u8[>..] = [];
    var pre: u8[>..] = [];
    var general_arch: u8[>..] = [];
    get_str(ctx, "general.name", model_name);
    get_str(ctx, "tokenizer.ggml.pre", pre);
    get_str(ctx, "general.architecture", general_arch);
    to_lower(model_name);

    if contains_any(pre, ["jina-v2-de", "jina-v2-es", "jina-v2-code", "mmbert"])
       || contains_any(general_arch, ["nomic-bert-moe", "jina-bert-v3"]) {
        if !token_to_id.contains("<mask>") {
            log(str("load: Mask token is missing in vocab, please reconvert model!"));
        } else {
            set_token_attr("<mask>", ATTR_LSTRIP, true);
        }
    } else if contains_any(model_name, ["phi-3", "phi3"]) {
        for id in cache_special_tokens { attrs[id] |= ATTR_RSTRIP; }
        set_token_attr("</s>", ATTR_RSTRIP, true);
        for t in ["<unk>", "<s>", "<|endoftext|>"] { set_token_attr(t, ATTR_RSTRIP, false); }
    } else if contains_any(model_name, ["modern-bert"]) {
        if !token_to_id.contains("[MASK]") {
            log(str("load: Mask token missing in vocab!"));
        } else {
            set_token_attr("[MASK]", ATTR_LSTRIP, true);
        }
    }
}

//
// Tokenizing
//

// A heap on a grow-shrink array: F(a, b) is true when a ranks below b, and
// the top is the highest. These are std::push_heap and std::pop_heap.
fn heap_push<T, F>(h: T[>..<]&, v: T) {
    h.push(v);
    var i = h.len - 1;
    while i > 0 {
        let parent = (i - 1) / 2;
        if !F(h[parent], h[i]) { break; }
        swap(h[parent], h[i]);
        i = parent;
    }
}

fn heap_pop<T, F>(h: T[>..<]&) -> T {
    let top = h[0];
    let last = h.pop();
    if h.len > 0 {
        h[0] = last;
        var i = 0;
        loop {
            let l = 2 * i + 1;
            let r = l + 1;
            var best = i;
            if l < h.len && F(h[best], h[l]) { best = l; }
            if r < h.len && F(h[best], h[r]) { best = r; }
            if best == i { break; }
            swap(h[i], h[best]);
            i = best;
        }
    }
    return top;
}

// A piece of the text being tokenized, in a chain; n == 0 once merged away.
struct Symbol { prev: i64, next: i64, start: i64, n: i64 }

struct SpmBigram { left: i64, right: i64, score: f32, size: i64 }
struct Pair { left: i64, right: i64 }

fn spm_tokenize(text: const u8[:], out: i32[>..]&) {
    var symbols: Symbol[>..] = [];
    var offs = 0;
    while offs < text.len {
        let n = min(unicode::len_utf8(text[offs]), text.len - offs);
        let index = symbols.len;
        offs += n;
        symbols.push(Symbol { prev: index - 1, next: if offs == text.len { -1 } else { index + 1 }, start: offs - n, n: n });
    }

    var queue: SpmBigram[>..<] = [];
    var rev_merge = dictionary<const u8[:], Pair> {};

    fn try_add_bigram(left: i64, right: i64) {
        if left == -1 || right == -1 { return; }
        let piece = text[symbols[left].start..symbols[left].start + symbols[left].n + symbols[right].n];
        let token = text_to_token(piece);
        if token == TOKEN_NULL || token >= n_tokens() { return; }
        heap_push(queue, SpmBigram { left: left, right: right, score: scores[token], size: piece.len }) { l, r =>
            l.score < r.score || (l.score == r.score && l.left > r.left)
        };
        rev_merge.insert(piece, Pair { left: left, right: right });
    }

    // seed the work queue with all possible 2-character tokens
    for i in 1..symbols.len { try_add_bigram(i - 1, i); }

    // keep substituting the highest frequency pairs for as long as we can
    while queue.len > 0 {
        let b = heap_pop(queue) { l, r => l.score < r.score || (l.score == r.score && l.left > r.left) };
        let ln = symbols[b.left].n;
        let rn = symbols[b.right].n;
        // if one of the symbols already got merged, skip it
        if ln == 0 || rn == 0 || ln + rn != b.size { continue; }
        // merge the right symbol into the left one, and drop it from the chain
        symbols[b.left].n += rn;
        symbols[b.right].n = 0;
        symbols[b.left].next = symbols[b.right].next;
        if symbols[b.right].next >= 0 { symbols[symbols[b.right].next].prev = b.left; }
        try_add_bigram(symbols[b.left].prev, b.left);
        try_add_bigram(b.left, symbols[b.left].next);
    }

    // resegment each symbol; a symbol that is no token splits back into the
    // pair it was merged from, or into bytes
    var todo: i64[>..<] = [];
    var i = 0;
    while i != -1 {
        todo.push(i);
        while todo.len > 0 {
            let s = symbols[todo.pop()];
            let piece = text[s.start..s.start + s.n];
            let token = text_to_token(piece);
            if token != TOKEN_NULL { out.push(token); continue; }
            let p = rev_merge.get(piece);
            if !p {
                for c in piece { out.push(byte_to_token(c)); }
                continue;
            }
            todo.push(p.right);
            todo.push(p.left);
        }
        i = symbols[i].next;
    }
}

struct BpeBigram { left: i64, right: i64, start: i64, size: i64, rank: i32 }

fn bpe_less(l: BpeBigram, r: BpeBigram) -> bool {
    l.rank > r.rank || (l.rank == r.rank && l.left > r.left)
}

// The BPE merges of one word, appended to `symbols` (the word's own start at
// word_start in text) and chained after the symbols already there.
fn bpe_word(text: const u8[:], word_start: i64, word_end: i64, symbols: Symbol[>..]&, scratch: u8[>..]&) {
    let word = text[word_start..word_end];
    let first = symbols.len;
    var offset = 0;
    if ignore_merges && text_to_token(word) != TOKEN_NULL {
        symbols.push(Symbol { prev: -1, next: -1, start: word_start, n: word.len });
        offset = word.len;
    } else if (pre_type == PRE_GEMMA4 || pre_type == PRE_MMBERT) && all_newlines(word) {
        // fix for gemma 4, ref: https://github.com/ggml-org/llama.cpp/pull/21343
        if text_to_token(word) != TOKEN_NULL {
            symbols.push(Symbol { prev: -1, next: -1, start: word_start, n: word.len });
            offset = word.len;
        }
    }
    while offset < word.len {
        let n = min(word.len - offset, unicode::len_utf8(word[offset]));
        let index = symbols.len - first;
        offset += n;
        symbols.push(Symbol { prev: index - 1, next: if offset == word.len { -1 } else { index + 1 },
                              start: word_start + offset - n, n: n });
    }
    // within this function, symbol k of the word is symbols[first + k]
    let syms = symbols[first..];

    var queue: BpeBigram[>..<] = [];
    fn add_new_bigram(left: i64, right: i64) {
        if left == -1 || right == -1 { return; }
        let lt = text[syms[left].start..syms[left].start + syms[left].n];
        let rt = text[syms[right].start..syms[right].start + syms[right].n];
        let rank = find_bpe_rank(lt, rt, scratch);
        if rank < 0 { return; }
        heap_push(queue, BpeBigram { left: left, right: right, start: syms[left].start, size: lt.len + rt.len, rank: rank }) { l, r =>
            bpe_less(l, r)
        };
    }
    for i in 1..syms.len { add_new_bigram(i - 1, i); }

    while queue.len > 0 {
        let b = heap_pop(queue) { l, r => bpe_less(l, r) };
        let ls = syms[b.left];
        let rs = syms[b.right];
        if ls.n == 0 || rs.n == 0 { continue; }
        // skip a bigram whose text is out of date
        if ls.n + rs.n != b.size || text[ls.start..ls.start + ls.n] != text[b.start..b.start + ls.n] ||
           text[rs.start..rs.start + rs.n] != text[b.start + ls.n..b.start + b.size] { continue; }
        syms[b.left].n += rs.n;
        syms[b.right].n = 0;
        syms[b.left].next = rs.next;
        if rs.next >= 0 { syms[rs.next].prev = b.left; }
        add_new_bigram(syms[b.left].prev, b.left);
        add_new_bigram(b.left, syms[b.left].next);
    }
}

fn bpe_tokenize(raw: const u8[:], out: i32[>..]&) {
    // the text the regexes split
    var text: u8[>..] = [];
    if pre_type == PRE_NEMOTRON_3_5_EMBD {
        let zwnj = "\xe2\x80\x8c";
        format_replaced(text, raw, str("[", zwnj), "[");
        var t2: u8[>..] = [];
        format_replaced(t2, text, str("<", zwnj), "<");
        text.clear();
        text.append(t2);
    } else if pre_type == PRE_SOPRANO {
        var space = false;
        for cpt in unicode::cpts_from_utf8(raw) {
            if unicode::is(unicode::flags(cpt), unicode::WHITESPACE) {
                if !space { text.push(' '); }
                space = true;
            } else {
                // Unicode full lowercase expands LATIN CAPITAL LETTER I WITH DOT ABOVE.
                if cpt == 0x0130 { text.append("i\xcc\x87"); } else { unicode::cpt_to_utf8(text, unicode::tolower(cpt)); }
                space = false;
            }
        }
    } else {
        text.append(raw);
    }

    var words: u8[>..] = [];
    var ends: i64[>..] = [];
    unicode::regex_split(text, bpe_regexes, bpe_byte_encode, words, ends);

    if pre_type == PRE_MMBERT {
        // Metaspace: a text starts with an escaped space, and each escaped space starts a new word
        let space = "\xe2\x96\x81";
        var w2: u8[>..] = [];
        var e2: i64[>..] = [];
        var start = 0;
        for e in ends {
            let word = words[start..e];
            start = e;
            if all_newlines(word) { w2.append(word); e2.push(w2.len); continue; }
            var spaced: u8[>..] = [];
            if !starts_with(word, space) { spaced.append(space); }
            spaced.append(word);
            var s = 0;
            while s < spaced.len {
                let next = find(spaced[s + space.len..], space);
                let end = if next < 0 { spaced.len } else { s + space.len + next };
                w2.append(spaced[s..end]);
                e2.push(w2.len);
                s = end;
            }
        }
        words.clear();
        words.append(w2);
        ends.clear();
        ends.append(e2);
    }

    // merge each word, chaining the finished symbols in order
    var symbols: Symbol[>..] = [];
    var finals: Symbol[>..] = [];
    var scratch: u8[>..] = [];
    var start = 0;
    for e in ends {
        symbols.clear();
        bpe_word(words, start, e, symbols, scratch);
        for s in symbols {
            if s.n > 0 { finals.push(Symbol { prev: finals.len - 1, next: -1, start: s.start, n: s.n }); }
        }
        start = e;
    }

    for s in finals {
        if s.n == 0 { continue; }
        let piece = words[s.start..s.start + s.n];
        let token = text_to_token(piece);
        if token == TOKEN_NULL && pre_type == PRE_SOPRANO {
            out.push(special_unk_id);
        } else if token == TOKEN_NULL {
            for c in piece {
                var t: i32 = TOKEN_NULL;
                if bpe_byte_encode {
                    t = text_to_token([c]);
                } else {
                    // for non-byte-encoded BPE (e.g. gemma-4), byte tokens are <0xXX>
                    var buf: u8[..6] = [];
                    byte_token_text(buf, c);
                    t = text_to_token(buf);
                }
                if t != TOKEN_NULL { out.push(t); }
            }
        } else {
            out.push(token);
        }
    }
}

// A fragment of the text: a token, or (token == -1) the raw text at offset.
struct Fragment { token: i32, offset: i64, length: i64 }

fn all_newlines(s: const u8[:]) -> bool {
    for c in s { if c != '\n' { return false; } }
    return true;
}

fn is_c_space(c: u8) -> bool { c == ' ' || (c >= 0x09 && c <= 0x0D) }

// Splits the raw fragments at each special token, longest token first.
fn st_partition(raw: const u8[:], frags: Fragment[>..<]&, parse_special: bool) {
    var next: Fragment[>..<] = [];
    for special_id in cache_special_tokens {
        let a = attrs[special_id];
        // control and unknown tokens are text unless parse_special; user-defined ones always split
        if !parse_special && a & (ATTR_CONTROL | ATTR_UNKNOWN) != 0 { continue; }
        let st = token_text(special_id);
        next.clear();
        for f in frags {
            if f.token != -1 { next.push(f); continue; }
            var base_offset = f.offset;
            var base_length = f.length;
            loop {
                let rel = find(raw[base_offset..base_offset + base_length], st);
                if rel < 0 {
                    if base_length > 0 { next.push(Fragment { token: -1, offset: base_offset, length: base_length }); }
                    break;
                }
                let m = base_offset + rel;
                if m > base_offset {
                    var left_length = m - base_offset;
                    if a & ATTR_LSTRIP != 0 {
                        while left_length > 0 && is_c_space(raw[base_offset + left_length - 1]) { left_length -= 1; }
                    }
                    if left_length > 0 { next.push(Fragment { token: -1, offset: base_offset, length: left_length }); }
                }
                next.push(Fragment { token: special_id, offset: 0, length: 0 });
                guard m + st.len < base_offset + base_length else { break; }
                var right_offset = m + st.len;
                var right_length = base_length - ((m - base_offset) + st.len);
                if a & ATTR_RSTRIP != 0 {
                    while right_length > 0 && is_c_space(raw[right_offset]) { right_offset += 1; right_length -= 1; }
                }
                base_offset = right_offset;
                base_length = right_length;
            }
        }
        frags.clear();
        frags.append(next);
    }
}

// Appends the tokens of raw_text to out.
fn tokenize(raw_text: const u8[:], add_special: bool, parse_special: bool, out: i32[>..]&) {
    var frags: Fragment[>..<] = [];
    if raw_text.len > 0 {
        frags.push(Fragment { token: -1, offset: 0, length: raw_text.len });
        st_partition(raw_text, frags, parse_special);
    }
    let first_out = out.len;

    match vtype {
        TYPE_SPM => {
            // TODO (upstream): honor metaspace_prepend_scheme (add_space_prefix behaves like "always")
            var is_prev_special = true;     // prefix with space if first token
            if add_special && add_bos {
                assert(special_bos_id != TOKEN_NULL);
                out.push(special_bos_id);
                is_prev_special = true;
            }
            var text: u8[>..] = [];
            for f in frags {
                if f.token == -1 {
                    text.clear();
                    if add_space_prefix && is_prev_special { text.append("\xe2\x96\x81"); }
                    format_replaced(text, raw_text[f.offset..f.offset + f.length], " ", "\xe2\x96\x81");
                    spm_tokenize(text, out);
                    is_prev_special = false;
                } else {
                    out.push(f.token);
                    is_prev_special = true;
                }
            }
            if add_special && add_bos && out.len - first_out >= 2 && out[first_out + 1] == special_bos_id {
                log(str("tokenize: Added a BOS token to the prompt as specified by the model but the prompt also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. Are you sure this is what you want?"));
            }
            if add_special && add_eos {
                assert(special_eos_id != TOKEN_NULL);
                out.push(special_eos_id);
            }
        },
        TYPE_BPE => {
            if tokenizer_model == "hybriddna" || tokenizer_model == "whitespace" {
                abort(str(tokenizer_model, " tokenizer: not ported yet"));
            }
            if add_special && add_bos {
                assert(special_bos_id != TOKEN_NULL);
                out.push(special_bos_id);
            }
            var text: u8[>..] = [];
            for f, k in frags {
                if f.token == -1 {
                    text.clear();
                    let piece = raw_text[f.offset..f.offset + f.length];
                    let escaped = "\xe2\x96\x81";
                    // metaspace prepend_scheme "first": U+2581 only at the start of the input
                    let prepend = metaspace_prepend_scheme == METASPACE_FIRST && k == 0;
                    if escape_whitespaces {
                        format_replaced(text, piece, " ", escaped);
                    } else {
                        text.append(piece);
                    }
                    if prepend && !starts_with(text, escaped) {
                        var t2: u8[>..] = [];
                        t2.append(escaped);
                        t2.append(text);
                        text.clear();
                        text.append(t2);
                    }
                    bpe_tokenize(text, out);
                } else {
                    out.push(f.token);
                }
            }
            if add_special {
                if add_eos {
                    assert(special_eos_id != TOKEN_NULL);
                    out.push(special_eos_id);
                }
                let n = out.len - first_out;
                if add_bos && n >= 2 && out[first_out + 1] == special_bos_id {
                    log(str("tokenize: Added a BOS token to the prompt as specified by the model but the prompt also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. Are you sure this is what you want?"));
                }
                if add_eos && n >= 2 && out[out.len - 2] == special_eos_id {
                    log(str("tokenize: Added a EOS token to the prompt as specified by the model but the prompt also ends with a EOS token. So now the final prompt ends with 2 EOS tokens. Are you sure this is what you want?"));
                }
            }
        },
        TYPE_UGM => {
            if add_special && add_bos {
                assert(special_bos_id != TOKEN_NULL);
                out.push(special_bos_id);
            }
            for f in frags {
                if f.token == -1 { ugm_tokenize(raw_text[f.offset..f.offset + f.length], out); }
                else { out.push(f.token); }
            }
            if add_special && add_bos && out.len - first_out >= 2 && out[first_out + 1] == special_bos_id {
                log(str("tokenize: Added a BOS token to the prompt as specified by the model but the prompt also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. Are you sure this is what you want?"));
            }
            if add_special && add_eos {
                assert(special_eos_id != TOKEN_NULL);
                out.push(special_eos_id);
            }
        },
        TYPE_RWKV => {
            for f in frags {
                if f.token == -1 { rwkv_tokenize(raw_text[f.offset..f.offset + f.length], out); }
                else { out.push(f.token); }
            }
        },
        TYPE_PLAMO2, TYPE_PLAMO3 => {
            if add_special && add_bos {
                assert(special_bos_id != TOKEN_NULL);
                out.push(special_bos_id);
            }
            for f in frags {
                if f.token == -1 { plamo_tokenize(raw_text[f.offset..f.offset + f.length], out); }
                else { out.push(f.token); }
            }
            if add_special && add_bos && out.len - first_out >= 2 && out[first_out + 1] == special_bos_id {
                log(str("tokenize: Added a BOS token to the prompt as specified by the model but the prompt also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. Are you sure this is what you want?"));
            }
            if add_special && add_eos {
                assert(special_eos_id != TOKEN_NULL);
                out.push(special_eos_id);
            }
        },
        TYPE_NONE => { abort("fatal error"); },
        _ => { abort(str(type_name(), " tokenizer: not ported yet")); },
    }
}

// The text of a BPE byte-level token: each codepoint back to its byte.
fn decode_text(out: u8[>..]&, text: const u8[:]) {
    for cpt in unicode::cpts_from_utf8(text) {
        let b, ok = unicode::cpt_to_byte(cpt);
        if ok {
            out.push(b);
        } else {
            var utf8: u8[>..] = [];
            unicode::cpt_to_utf8(utf8, cpt);
            out.append("[UNK_BYTE_0x");
            for c in utf8 { out.push("0123456789abcdef"[c >> 4]); out.push("0123456789abcdef"[c & 15]); }
            out.append(text);
            out.push(']');
        }
    }
}

// Appends the text of a token, skipping up to lstrip leading spaces; returns
// the number of bytes appended.
fn token_to_piece(token: i32, out: u8[>..]&, lstrip: i64, special: bool, use_cache: bool = true) -> i64 {
    let attr_special = ATTR_UNKNOWN | ATTR_CONTROL;
    let attr = attrs[token];
    if !special && attr & attr_special != 0 { return 0; }

    let start = out.len;
    fn finish() -> i64 {
        // skip up to lstrip leading spaces
        var skip = 0;
        while skip < lstrip && start + skip < out.len && out[start + skip] == ' ' { skip += 1; }
        if skip > 0 {
            for i in start..out.len - skip { out[i] = out[i + skip]; }
            out.resize(out.len - skip);
        }
        return out.len - start;
    }

    if use_cache && piece_end.len > 0 {
        out.append(token_piece(token));
        return finish();
    }

    guard 0 <= token && token < n_tokens() else { return 0; }
    let t = token_text(token);
    match vtype {
        TYPE_WPM, TYPE_SPM, TYPE_UGM => {
            if attr & (attr_special | ATTR_USER_DEFINED) != 0 { out.append(t); return finish(); }
            if attr & ATTR_NORMAL != 0 { format_replaced(out, t, "\xe2\x96\x81", " "); return finish(); }
            if attr & ATTR_BYTE != 0 { out.push(token_to_byte(token)); return finish(); }
        },
        TYPE_BPE => {
            if attr & (attr_special | ATTR_USER_DEFINED) != 0 { out.append(t); return finish(); }
            if attr & ATTR_NORMAL != 0 {
                if pre_type == PRE_SOPRANO { out.append(t); return finish(); }
                if escape_whitespaces {
                    // SPM-style BPE: tokens contain ▁ for spaces
                    format_replaced(out, t, "\xe2\x96\x81", " ");
                    return finish();
                }
                decode_text(out, t);
                return finish();
            }
            if attr & ATTR_BYTE != 0 { out.push(token_to_byte(token)); return finish(); }
        },
        TYPE_TEST => {
            format_uint(out, token as u64, 16, 0, ' ');
            return finish();
        },
        TYPE_PLAMO2, TYPE_PLAMO3 => {
            if is_byte(token) && t.len == 6 && starts_with(t, "<0x") && t[5] == '>' &&
               hex_value(t[3]) >= 0 && hex_value(t[4]) >= 0 {
                out.push((hex_value(t[3]) * 16 + hex_value(t[4])) as! u8);
                return 1;
            }
            out.append(t);
            return finish();
        },
        TYPE_RWKV => {
            // no lstrip
            let n0 = out.len;
            rwkv_unescape(t, out);
            return out.len - n0;
        },
        _ => { abort("fatal error"); },
    }
    return 0;
}

//
// UGM (T5): the best scoring segmentation of the normalized text, by
// Viterbi over a trie of the tokens (llm_tokenizer_ugm)
//

// the tokens' trie (normal, user defined and unused tokens) and the user
// defined tokens' trie; node 0 is the root of each
var ugm_child = dictionary<i64, i32> {};
var ugm_value: i32[>..];
var ugm_ud_child = dictionary<i64, i32> {};
var ugm_ud_nodes: i64 = 0;
var ugm_unknown_score: f32 = 0.0;

fn trie_insert(child: dictionary<i64, i32>&, n_nodes: i64&, text: const u8[:]) -> i64 {
    var node: i64 = 0;
    for c in text {
        let key = node * 256 + c;
        let next = child.get(key);
        if next {
            node = next;
        } else {
            n_nodes += 1;
            child.insert(key, n_nodes as i32);
            node = n_nodes;
        }
    }
    return node;
}

fn ugm_build() {
    ugm_child.clear();
    ugm_value.clear();
    ugm_value.push(-1);
    ugm_ud_child.clear();
    ugm_ud_nodes = 0;
    var min_score: f32 = 3.4028234663852886e38;
    var n_nodes: i64 = 0;
    for id in n_tokens() {
        if is_normal(id) { min_score = min(min_score, scores[id]); }
        if is_normal(id) || is_user_defined(id) || is_unused(id) {
            let node = trie_insert(ugm_child, n_nodes, token_text(id));
            while ugm_value.len <= n_nodes { ugm_value.push(-1); }
            ugm_value[node] = id as i32;
        }
        if is_user_defined(id) { trie_insert(ugm_ud_child, ugm_ud_nodes, token_text(id)); }
    }
    ugm_unknown_score = min_score - 10.0;
}

fn u32_at(b: const u8[:], i: i64) -> u32 {
    let b0: u32 = b[i];
    return b0 | (b[i + 1] as u32) << 8 | (b[i + 2] as u32) << 16 | (b[i + 3] as u32) << 24;
}

// The normalized text of input[offset..]'s longest normalizable prefix, and
// how much of input it consumes (normalize_prefix).
fn ugm_normalize_prefix(input: const u8[:], offset: i64, out: u8[>..]&) -> i64 {
    if offset == input.len { return 0; }
    // a user defined token stays as it is
    {
        var node: i64 = 0;
        var o = offset;
        while o < input.len {
            let next = ugm_ud_child.get(node * 256 + input[o]);
            if !next { break; }
            node = next;
            o += 1;
        }
        if o > offset { out.append(input[offset..o]); return o - offset; }
    }
    // the precompiled charsmap: an XOR-compressed double array of replacements
    var longest_len: i64 = 0;
    var longest_offset: i64 = 0;
    let cm = precompiled_charsmap;
    var xcda_size: i64 = 0;
    var repl_start: i64 = 0;
    if cm.len > 0 {
        let blob = u32_at(cm, 0) as i64;
        xcda_size = blob / 4;
        repl_start = 4 + blob;
    }
    fn node(i: i64) -> u32 {
        guard i < xcda_size else { abort("Index out of array bounds in XCDA array!"); }
        return u32_at(cm, 4 + i * 4);
    }
    fn base(i: i64) -> i64 { let p = node(i); return (p >> 10) << ((p & (1 << 9)) >> 6); }
    fn lcheck(i: i64) -> u32 { node(i) & ((1 << 31) | 0xff) }
    fn leaf(i: i64) -> bool { (node(i) >> 8) & 1 != 0 }
    fn value(i: i64) -> i64 { node(i) & ((1 << 31) - 1) }
    if xcda_size > 0 {
        var ni: i64 = base(0);
        var p = offset;
        while p < input.len {
            let c = input[p];
            if c == 0 { break; }
            ni ^= c;
            if lcheck(ni) != c { break; }
            let is_leaf = leaf(ni);
            ni ^= base(ni);
            if is_leaf {
                longest_len = p - offset + 1;
                longest_offset = value(ni);
            }
            p += 1;
        }
    }
    if longest_len > 0 {
        let repl_size = cm.len - repl_start;
        guard longest_offset < repl_size else { abort("Index out of array bounds in precompiled charsmap!"); }
        var n: i64 = 0;
        while n < repl_size - longest_offset && cm[repl_start + longest_offset + n] != 0 { n += 1; }
        guard n < repl_size - longest_offset else { abort("Unterminated string in precompiled charsmap!"); }
        out.append(cm[repl_start + longest_offset..repl_start + longest_offset + n]);
        return longest_len;
    }
    let cpt, next, ok = unicode::cpt_from_utf8(input, offset);
    if ok { out.append(input[offset..next]); return next - offset; }
    out.append("\xEF\xBF\xBD");
    return 1;
}

fn ugm_normalize(input: const u8[:], normalized: u8[>..]&) {
    normalized.clear();
    let space = if escape_whitespaces { "\xE2\x96\x81" } else { " " };
    let shall_prepend_space = !treat_whitespace_as_suffix && add_space_prefix;
    let shall_append_space = treat_whitespace_as_suffix && add_space_prefix;
    let shall_merge_spaces = remove_extra_whitespaces;
    var is_space_prepended = false;
    var processing_non_ws = false;
    var piece: u8[>..] = [];
    var offset: i64 = 0;
    while offset < input.len {
        piece.clear();
        let consumed = ugm_normalize_prefix(input, offset, piece);
        for c in piece {
            if c != ' ' {
                if !processing_non_ws {
                    processing_non_ws = true;
                    if (shall_prepend_space && !is_space_prepended) || shall_merge_spaces {
                        normalized.append(space);
                        is_space_prepended = true;
                    }
                }
                normalized.push(c);
            } else {
                if processing_non_ws { processing_non_ws = false; }
                if !shall_merge_spaces { normalized.append(space); }
            }
        }
        offset += consumed;
    }
    if shall_append_space { normalized.append(space); }
}

fn ugm_tokenize(text: const u8[:], out: i32[>..]&) {
    let first = out.len;
    var normalized: u8[>..] = [];
    var valid = true;
    {
        var o: i64 = 0;
        while o < text.len {
            let cpt, next, ok = unicode::cpt_from_utf8(text, o);
            if !ok { valid = false; break; }
            o = next;
        }
    }
    if normalizer_nfc && valid {
        var canonical: u8[>..] = [];
        unicode::cpts_to_utf8(canonical, unicode::normalize_nfc(unicode::cpts_from_utf8(text)));
        ugm_normalize(canonical, normalized);
    } else {
        ugm_normalize(text, normalized);
    }
    let n = normalized.len;
    if n == 0 { return; }

    // the best tokenization ending at each offset: its last token, where that starts, its score
    var best_id: i32[>..] = [];
    var best_start: i64[>..] = [];
    var best_score: f64[>..] = [];
    best_id.resize(n + 1, special_unk_id);
    best_start.resize(n + 1, 0);
    best_score.resize(n + 1, -1.7976931348623157e308);
    best_score[0] = 0.0;
    var offset: i64 = 0;
    while offset < n {
        let n_units = min(unicode::len_utf8(normalized[offset]), n - offset);
        var single_found = false;
        let cur_score = best_score[offset];
        var prefix = offset;
        var node: i64 = -1;
        {
            let nx = ugm_child.get(normalized[prefix]);
            node = if nx { nx } else { -1 };
            prefix += 1;
        }
        while prefix <= n && node >= 0 {
            if ugm_value[node] >= 0 {
                if prefix - offset == n_units { single_found = true; }
                let id = ugm_value[node];
                let token_score: f64 = if is_user_defined(id) { 0.0 } else { scores[id] };
                let challenger = cur_score + token_score;
                if challenger > best_score[prefix] {
                    best_id[prefix] = id;
                    best_start[prefix] = offset;
                    best_score[prefix] = challenger;
                }
            }
            if prefix >= n { break; }
            let nx = ugm_child.get(node * 256 + normalized[prefix]);
            node = if nx { nx } else { -1 };
            prefix += 1;
        }
        if !single_found {
            let challenger = cur_score + ugm_unknown_score;
            let p = offset + n_units;
            if challenger > best_score[p] {
                best_id[p] = special_unk_id;
                best_start[p] = offset;
                best_score[p] = challenger;
            }
        }
        offset += n_units;
    }
    // back from the end
    var is_prev_unknown = false;
    var at = n;
    while true {
        let is_unknown = best_id[at] == special_unk_id;
        if !(is_prev_unknown && is_unknown) { out.push(best_id[at]); }
        if best_start[at] == 0 { break; }
        is_prev_unknown = is_unknown;
        at = best_start[at];
    }
    var i = first;
    var j = out.len - 1;
    while i < j {
        let t = out[i];
        out[i] = out[j];
        out[j] = t;
        i += 1;
        j -= 1;
    }
}

//
// RWKV: the longest token at each position, from a trie of the unescaped
// token texts (llm_tokenizer_rwkv)
//

// rwkv_child[node * 256 + byte]: the node a byte leads to; node 0 is the root.
var rwkv_child = dictionary<i64, i32> {};
var rwkv_value: i32[>..];     // a node's token, or -1

// A token's text with RWKV's escapes undone: \t \n \r \xHH and \<c> (llama_unescape_rwkv_token).
fn rwkv_unescape(escaped: const u8[:], out: u8[>..]&) {
    var escaping = false;
    var hex_remaining = 0;
    var hex_acc: i64 = 0;
    for c in escaped {
        if hex_remaining != 0 {
            let value: i64 = if c >= 'a' { c as i64 - 'a' + 10 } else { c as i64 - '0' };    // as uint8_t
            hex_acc = ((hex_acc << 4) + value) & 255;
            hex_remaining -= 1;
            if hex_remaining == 0 {
                out.push(hex_acc as! u8);
                hex_acc = 0;
            }
            continue;
        }
        if escaping {
            if c == 't' { out.push('\t'); }
            else if c == 'n' { out.push('\n'); }
            else if c == 'r' { out.push('\r'); }
            else if c == 'x' { hex_remaining = 2; }
            else { out.push(c); }
            escaping = false;
            continue;
        }
        if c == '\\' { escaping = true; continue; }
        out.push(c);
    }
}

fn rwkv_build() {
    rwkv_child.clear();
    rwkv_value.clear();
    rwkv_value.push(-1);
    var text: u8[>..] = [];
    for id in n_tokens() {
        text.clear();
        rwkv_unescape(token_text(id), text);
        var node: i64 = 0;
        for c in text {
            let key = node * 256 + c;
            let next = rwkv_child.get(key);
            if next {
                node = next;
            } else {
                rwkv_child.insert(key, rwkv_value.len as i32);
                node = rwkv_value.len;
                rwkv_value.push(-1);
            }
        }
        rwkv_value[node] = id as i32;    // a repeated text: its last id
    }
}

fn rwkv_tokenize(text: const u8[:], out: i32[>..]&) {
    fn child(node: i64, c: u8) -> i64 {
        let next = rwkv_child.get(node * 256 + c);
        return if next { next } else { -1 };
    }
    var position: i64 = 0;
    while position < text.len {
        var node = child(0, text[position]);
        if node < 0 {
            // no matching token: the unknown token
            out.push(special_unk_id);
            position += 1;
            continue;
        }
        // the longest matching token
        var token_id: i32 = 0;
        var token_length: i64 = 0;
        while node >= 0 {
            if rwkv_value[node] >= 0 {
                token_id = rwkv_value[node];
                token_length = position + 1;
            }
            if position + 1 >= text.len { break; }
            position += 1;
            node = child(node, text[position]);
        }
        out.push(token_id);
        position = token_length;
    }
}

//
// PLaMo2: the best scoring segmentation by dynamic programming over a table
// of every token's suffixes (llm_tokenizer_plamo2)
//

let PLAMO_INVALID_SCORE: i32 = -20000000;
let PLAMO_UNKNOWN_SCORE: i32 = -10000000;

var plamo_bytes: i32[>..];                          // the byte tokens <0xXX>
var plamo_to_suffix_id = dictionary<i64, i32> {};   // (first codepoint << 32 | rest's piece id) -> suffix id
var plamo_table: i32[>..];                          // rows of [piece length, token id, score, piece id]
var plamo_suffix_buf: u8[>..];                      // the suffixes' texts

fn plamo_build() {
    plamo_bytes.clear();
    plamo_bytes.resize(256, 0);
    plamo_to_suffix_id.clear();
    plamo_table.clear();
    plamo_suffix_buf.clear();

    // every token's text and its suffixes (codepoint by codepoint, re-encoded
    // as llama.cpp does), in order: (start, end, is the token itself)
    var starts: i64[>..] = [];
    var ends: i64[>..] = [];
    var whole: i32[>..] = [];       // the token id, or -1 for a suffix
    for id in n_tokens() {
        let t = token_text(id);
        if is_byte(id) {
            if t.len == 6 && starts_with(t, "<0x") && t[5] == '>' && hex_value(t[3]) >= 0 && hex_value(t[4]) >= 0 {
                plamo_bytes[hex_value(t[3]) * 16 + hex_value(t[4])] = id as i32;
            }
            continue;
        }
        let cpts = unicode::cpts_from_utf8(t);
        for i in cpts.len {
            starts.push(plamo_suffix_buf.len);
            unicode::cpts_to_utf8(plamo_suffix_buf, cpts[i..]);
            ends.push(plamo_suffix_buf.len);
            whole.push(if i == 0 { id as i32 } else { -1 });
        }
    }
    for b in 256 {
        if plamo_bytes[b] == 0 { abort(str("Byte token for <0x", b, "> is not set")); }
    }

    // suffix -> score: a token's own score, NaN for a suffix that is no token
    let nan: f32 = 0.0 / 0.0;
    var suffix_to_score = dictionary<const u8[:], f32> {};
    for k in starts.len {
        let s = plamo_suffix_buf[starts[k]..ends[k]];
        if whole[k] >= 0 { suffix_to_score.insert(s, scores[whole[k]]); }
        else if !suffix_to_score.contains(s) { suffix_to_score.insert(s, nan); }
    }

    // the suffixes, and "", in the order of their reversed texts
    var suffixes: (const u8[:])[>..] = [];
    suffix_to_score.each() { key, val => suffixes.push(key); };
    suffixes.push("");
    sort(suffixes) { a, b =>
        var i: i64 = 1;
        var r = a.len < b.len;
        while i <= min(a.len, b.len) {
            let ca = a[a.len - i];
            let cb = b[b.len - i];
            if ca != cb { r = ca < cb; break; }
            i += 1;
        }
        r
    };

    // each piece's id, and how its first codepoint extends a shorter suffix
    fn cpt_offsets(s: const u8[:]) -> i64[>..] {     // where each codepoint starts, and s.len
        var offs: i64[>..] = [];
        var o: i64 = 0;
        while o < s.len {
            offs.push(o);
            let cpt, next, ok = unicode::cpt_from_utf8(s, o);
            o = if ok { next } else { o + 1 };
        }
        offs.push(s.len);
        return offs;
    }
    var suffix_to_id = dictionary<const u8[:], i32> {};
    var num_pieces: i32 = 0;
    for suffix in suffixes {
        suffix_to_id.insert(suffix, num_pieces);
        if suffix.len > 0 {
            let offs = cpt_offsets(suffix);
            let first, next, ok = unicode::cpt_from_utf8(suffix, 0);
            let rest_id = suffix_to_id.get(suffix[offs[1]..]);
            let piece_code = (first as i64) << 32 | (if rest_id { rest_id as i64 } else { 0 });
            plamo_to_suffix_id.insert(piece_code, num_pieces);
            var pieces: i32 = 1;    // the sentinel row
            for n in 1..offs.len {
                if suffix_to_score.contains(suffix[0..offs[n]]) { pieces += 1; }
            }
            num_pieces += pieces;
        } else {
            num_pieces += 1;        // the empty suffix: its sentinel row
        }
    }

    // the table: each suffix's prefixes that are pieces, longest first, then a sentinel row
    for suffix in suffixes {
        let offs = cpt_offsets(suffix);
        var n = offs.len - 1;
        while n > 0 {
            let piece = suffix[0..offs[n]];
            let score = suffix_to_score.get(piece);
            if score {
                let id = token_to_id.get(piece);
                let sid = suffix_to_id.get(piece);
                let sc: f32 = score;
                let finite = sc == sc && sc - sc == 0.0;
                var scaled: i32 = PLAMO_INVALID_SCORE;
                if finite {
                    let d: f64 = sc as f64 * 1e4;
                    scaled = (if d >= 0.0 { floor(d + 0.5) } else { -floor(-d + 0.5) }) as i32;    // std::round
                }
                plamo_table.push(n as i32);
                plamo_table.push(if id { id } else { -1 });
                plamo_table.push(scaled);
                plamo_table.push(if sid { sid } else { 0 });
            }
            n -= 1;
        }
        plamo_table.push(1);
        plamo_table.push(-1);
        plamo_table.push(PLAMO_UNKNOWN_SCORE);
        plamo_table.push(0);
    }
}

// The bytes past the first of c's UTF-8.
fn utf8_extra(c: u32) -> i64 { (if c >= 0x80 { 1 } else { 0 }) + (if c >= 0x800 { 1 } else { 0 }) + (if c >= 0x10000 { 1 } else { 0 }) }

fn plamo_encode_cpts(cpts: const u32[:], out: i32[>..]&) {
    if cpts.len == 0 { return; }
    let n = cpts.len;
    let n_rows = plamo_table.len / 4;
    fn row(p: i64, k: i64) -> i32 { plamo_table[p * 4 + k] }
    var scores_dp: i64[>..] = [];
    scores_dp.resize(n + 1, 1 << 60);
    scores_dp[n] = 0;
    var path_len: i32[>..] = [];       // [token length, token id, number of tokens] per position
    var path_id: i32[>..] = [];
    var path_num: i32[>..] = [];
    path_len.resize(n + 1, 0);
    path_id.resize(n + 1, 0);
    path_num.resize(n + 1, 0);

    var suffix_id: i64 = 0;
    var i = n - 1;
    while i >= 0 {
        let c = cpts[i];
        // the next suffix id
        for p in suffix_id..n_rows {
            let code = (c as i64) << 32 | row(p, 3);
            let s = plamo_to_suffix_id.get(code);
            suffix_id = if s { s } else { 0 };
            if suffix_id > 0 || row(p, 2) == PLAMO_UNKNOWN_SCORE { break; }
        }
        // the best path from here
        for p in suffix_id..n_rows {
            let score = row(p, 2);
            if score > PLAMO_INVALID_SCORE {
                let piece_length = row(p, 0) as i64;
                let s = scores_dp[i + piece_length] - score;
                if s < scores_dp[i] {
                    scores_dp[i] = s;
                    path_len[i] = piece_length as i32;
                    path_id[i] = row(p, 1);
                    path_num[i] = path_num[i + piece_length] + 1;
                    if score == PLAMO_UNKNOWN_SCORE {
                        // the UTF-8 bytes it falls back to
                        path_num[i] += utf8_extra(c) as i32;
                    }
                }
            }
            if score == PLAMO_UNKNOWN_SCORE { break; }
        }
        i -= 1;
    }

    // the best path's tokens
    var pos: i64 = 0;
    while pos < n {
        if path_id[pos] >= 0 {
            out.push(path_id[pos]);
        } else {
            // byte tokens
            let c = cpts[pos];
            let s = 1 + utf8_extra(c);
            for k in s {
                var b: i64 = 0;
                if s == 1 { b = c; }
                else if k == 0 { b = (0xF00 >> s) & 0xFF; }
                else { b = 0x80; }
                out.push(plamo_bytes[b | ((c as i64 >> ((s - k - 1) * 6)) & 0x3F)]);
            }
        }
        assert(path_len[pos] > 0);
        pos += path_len[pos];
    }
}

fn plamo_tokenize(text: const u8[:], out: i32[>..]&) {
    var cpts = unicode::cpts_from_utf8(text);
    if vtype == TYPE_PLAMO3 { plamo3_tokenize(cpts, out); return; }
    // PLaMo-2: a leading U+FEFF is dropped
    if cpts.len > 0 && cpts[0] == 0xFEFF { plamo_encode_cpts(cpts[1..], out); return; }
    plamo_encode_cpts(cpts, out);
}

// PLaMo-3 cuts the text first: around <|plamo:...|> tags, runs of four or
// more of a character (but newlines) and of two or more spaces, and at
// U+EE00 (dropped); each segment is encoded alone.
fn plamo3_tokenize(cpts: const u32[:], out: i32[>..]&) {
    let prefix: u32[8] = ['<', '|', 'p', 'l', 'a', 'm', 'o', ':'];
    let n = cpts.len;
    var cut: bool[>..] = [];
    cut.resize(n + 1, false);
    // pass 1: <|plamo:...|>
    var i: i64 = 0;
    while i + 8 <= n {
        var matches = true;
        for k in 8 { if cpts[i + k] != prefix[k] { matches = false; break; } }
        if !matches { i += 1; continue; }
        // an empty body is valid; U+001C..U+001F count as whitespace (Python's \s)
        var j = i + 8;
        while j < n && j - (i + 8) < 64 && cpts[j] != '|' && (cpts[j] < 0x1C || cpts[j] > 0x1F) &&
              !unicode::is(unicode::flags(cpts[j]), unicode::WHITESPACE) {
            j += 1;
        }
        if j + 1 < n && cpts[j] == '|' && cpts[j + 1] == '>' {
            cut[i] = true;
            cut[j + 2] = true;
            i = j + 2;
        } else {
            i += 1;
        }
    }
    // pass 2: runs of repeated characters or spaces, never across a cut of pass 1
    i = 0;
    while i < n {
        let c = cpts[i];
        var run: i64 = 1;
        while i + run < n && cpts[i + run] == c && !cut[i + run] { run += 1; }
        if (c != '\n' && run >= 4) || (c == ' ' && run >= 2) {
            cut[i] = true;
            cut[i + run] = true;
        }
        i += run;
    }
    var seg_start: i64 = 0;
    for seg_end in n + 1 {
        let is_boundary = seg_end < n && cpts[seg_end] == 0xEE00;
        if seg_end == n || cut[seg_end] || is_boundary {
            if seg_start < seg_end { plamo_encode_cpts(cpts[seg_start..seg_end], out); }
            seg_start = seg_end + (if is_boundary { 1 } else { 0 });
        }
    }
}

// The text of tokens: appends
// Appends the text of tokens.
fn detokenize(tokens: const i32[:], remove_special: bool, unparse_special: bool, out: u8[>..]&) {
    if vtype == TYPE_NONE { return; }
    var toks = tokens;
    // remove the leading space
    var remove_space = add_space_prefix;
    if remove_special && add_bos && toks.len > 0 && toks[0] == special_bos_id {
        remove_space = false;
        toks = toks[1..];
    }
    if remove_special && add_eos && toks.len > 0 && toks[toks.len - 1] == special_eos_id {
        toks = toks[..toks.len - 1];
    }
    // metaspace prepend_scheme "first": remove the leading space of the first rendered piece
    var remove_first_space = vtype == TYPE_BPE && metaspace_prepend_scheme == METASPACE_FIRST;

    let start = out.len;
    for t in toks {
        token_to_piece(t, out, if remove_space || remove_first_space { 1 } else { 0 }, unparse_special);
        remove_space = false;
        if unparse_special || attrs[t] & (ATTR_UNKNOWN | ATTR_CONTROL) == 0 { remove_first_space = false; }
    }

    guard clean_spaces else { return; }
    // three passes over the text, in place, as llama.cpp makes them
    let text = out[start..];
    var total = text.len;

    // first pass: characters ?!.,
    let total1 = total;
    total = if total > 0 { 1 } else { 0 };
    for i in 1..total1 {
        let x = text[i];
        if text[i - 1] == ' ' && (x == '?' || x == '!' || x == '.' || x == ',') { total -= 1; }   // " ?", " !", " .", " ,"
        text[total] = x;
        total += 1;
    }

    // second pass: strip single apostrophe between spaces
    let total2 = total;
    total = if total > 0 { 1 } else { 0 };
    var i = 1;
    while i < total2 {
        let x = text[i];
        if x == '\'' && i + 1 < total2 && text[i - 1] == ' ' && text[i + 1] == ' ' {   // " ' "
            total -= 1;       // remove prev space
            i += 1;
            text[i] = 0;      // remove next space
        }
        text[total] = x;
        total += 1;
        i += 1;
    }

    // third pass: apostrophe contractions
    let total3 = total;
    total = if total > 0 { 1 } else { 0 };
    for i in 1..total3 {
        let x = text[i];
        if text[i - 1] == ' ' && x == '\'' && i + 1 < total3 {
            let x1 = text[i + 1];
            if x1 == 't' || x1 == 'd' {                       // " 't", " 'd": keep the space
            } else if x1 == 's' || x1 == 'm' {                // " 's", " 'm"
                total -= 1;
            } else if i + 2 < total3 {
                let x2 = text[i + 2];
                if (x1 == 'r' && x2 == 'e') || (x1 == 'v' && x2 == 'e') { total -= 1; }   // " 're", " 've"
            }
        }
        text[total] = x;
        total += 1;
    }
    out.resize(start + total);
}
