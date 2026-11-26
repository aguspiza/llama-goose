// Multimodal input, as llama.cpp's tools/mtmd: a prompt whose media markers
// become images, each encoded into the embeddings the text model reads in
// place of tokens, between the tokens its model wraps them in.
// port-of: tools/mtmd/mtmd.cpp tools/mtmd/mtmd-helper.cpp
import std;
import llama from "../llama";
import llama.vocab;
import .image;
import .clip;
namespace mtmd;

// mtmd_default_marker
let MARKER = "<__media__>";

let CHUNK_TEXT = 0;
let CHUNK_IMAGE = 1;

// A piece of the prompt (mtmd_input_chunk): text tokens, chunk_tokens[at..at + n],
// or an image's pixels, chunk_pixels[at..at + nx * ny * 3], which encode into n tokens.
struct Chunk { kind: i64, at: i64, n: i64, nx: i64, ny: i64 }

var chunks: Chunk[>..] = [];
var chunk_tokens: i32[>..] = [];
var chunk_pixels: f32[>..] = [];

// The images given for a prompt (mtmd_bitmap), decoded, in the order of its markers.
struct Bitmap { nx: i64, ny: i64, at: i64 }
var bitmaps: Bitmap[>..] = [];
var bitmap_pixels: u8[>..] = [];

fn clear_bitmaps() {
    bitmaps.clear();
    bitmap_pixels.clear();
}

// An image file's bytes decoded into the next bitmap; false if they are not an image.
fn add_bitmap(bytes: const u8[:]) -> bool {
    var img = ImageU8 { nx: 0, ny: 0, buf: [] };
    guard decode_image(bytes, img) else { return false; }
    bitmaps.push(Bitmap { nx: img.nx, ny: img.ny, at: bitmap_pixels.len });
    bitmap_pixels.append(img.buf);
    return true;
}

fn chunk_text(k: Chunk) -> const i32[:] { chunk_tokens[k.at..k.at + k.n] }
fn chunk_image(k: Chunk) -> const f32[:] { chunk_pixels[k.at..k.at + k.nx * k.ny * 3] }

// The token whose piece is text (lookup_token).
fn lookup_token(text: const u8[:]) -> i32 {
    var p: u8[>..] = [];
    for id in vocab::n_tokens() {
        p.clear();
        vocab::token_to_piece(id as i32, p, 0, true);
        if p == text { return id as i32; }
    }
    return vocab::TOKEN_NULL;
}

// Tokens appended to the last chunk if it is text, else in a new one (add_text).
fn add_tokens(toks: const i32[:]) {
    guard toks.len > 0 else { return; }
    if chunks.len == 0 || chunks[chunks.len - 1].kind != CHUNK_TEXT || chunks[chunks.len - 1].at + chunks[chunks.len - 1].n != chunk_tokens.len {
        chunks.push(Chunk { kind: CHUNK_TEXT, at: chunk_tokens.len, n: 0, nx: 0, ny: 0 });
    }
    chunk_tokens.append(toks);
    chunks[chunks.len - 1].n += toks.len;
}

fn add_text(text: const u8[:], parse_special: bool) {
    var toks: i32[>..] = [];
    vocab::tokenize(text, false, parse_special, toks);
    add_tokens(toks);
}

fn add_token(t: i32) { add_tokens([t]); }

fn add_image(c: Clip&, img: ImageF32&) {
    chunks.push(Chunk { kind: CHUNK_IMAGE, at: chunk_pixels.len, n: n_output_tokens(c, img.nx, img.ny), nx: img.nx, ny: img.ny });
    chunk_pixels.append(img.buf);
}

// An image's chunks, and the tokens around them (add_media).
fn add_media(c: Clip&, img: ImageU8&) {
    var pp = Preproc { .. };
    preprocess(c, img, pp);
    let n = pp.slice_nx * pp.slice_ny * 3;
    if c.proj == PROJ_GEMMA3 {
        // <start_of_image> ... <end_of_image>
        add_text("<start_of_image>", true);
        var f = ImageF32 { nx: pp.slice_nx, ny: pp.slice_ny, buf: [] };
        f.buf.append(pp.slices[..n]);
        add_image(c, f);
        add_text("<end_of_image>", true);
        return;
    }
    // idefics3: the slices first, row after row, each after its row and
    // column, a new line between rows; then the overview
    for y in pp.grid_y {
        for x in pp.grid_x {
            add_text(str("<fake_token_around_image><row_", y + 1, "_col_", x + 1, ">"), true);
            let i = y * pp.grid_x + x;
            var f = ImageF32 { nx: pp.slice_nx, ny: pp.slice_ny, buf: [] };
            f.buf.append(pp.slices[i * n..(i + 1) * n]);
            add_image(c, f);
        }
        if y != pp.grid_y - 1 { add_token(lookup_token("\n")); }
    }
    add_token(lookup_token("\n\n"));
    add_token(lookup_token("<fake_token_around_image>"));
    add_token(lookup_token("<global-img>"));
    add_image(c, pp.overview);
    add_token(lookup_token("<fake_token_around_image>"));
}

// The chunks of text, whose markers stand for images, in order (mtmd_tokenize);
// add_special: the vocabulary's BOS first (and its EOS last).
fn tokenize(c: Clip&, text: const u8[:], add_special: bool) {
    chunks.clear();
    chunk_tokens.clear();
    chunk_pixels.clear();
    var i_img = 0;
    var start = 0;
    var i = 0;
    while i <= text.len {
        let at_marker = i + MARKER.len <= text.len && text[i..i + MARKER.len] == MARKER;
        if at_marker || i == text.len {
            if i > start { add_text(text[start..i], true); }
            if !at_marker { break; }
            guard i_img < bitmaps.len else { abort("mtmd: more media markers in the text than images"); }
            let bm = bitmaps[i_img];
            var img = ImageU8 { nx: bm.nx, ny: bm.ny, buf: [] };
            img.buf.append(bitmap_pixels[bm.at..bm.at + bm.nx * bm.ny * 3]);
            add_media(c, img);
            i_img++;
            i += MARKER.len;
            start = i;
            continue;
        }
        i++;
    }
    guard i_img == bitmaps.len else { abort("mtmd: fewer media markers in the text than images"); }
    if add_special && vocab::add_bos {
        if chunks.len > 0 && chunks[0].kind == CHUNK_TEXT {
            var toks: i32[>..] = [];
            toks.push(vocab::special_bos_id);
            toks.append(chunk_text(chunks[0]));
            // the first chunk's tokens rewritten with BOS first, at the end of the list
            chunks[0].at = chunk_tokens.len;
            chunks[0].n = toks.len;
            chunk_tokens.append(toks);
        } else {
            let at = chunk_tokens.len;
            chunk_tokens.push(vocab::special_bos_id);
            chunks.push(Chunk { kind: CHUNK_TEXT, at: 0, n: 0, nx: 0, ny: 0 });
            var j = chunks.len - 1;
            while j > 0 { chunks[j] = chunks[j - 1]; j--; }
            chunks[0] = Chunk { kind: CHUNK_TEXT, at: at, n: 1, nx: 0, ny: 0 };
        }
    }
    if add_special && vocab::add_eos { add_token(vocab::special_eos_id); }
}

// Whether a model's image tokens see each other (mtmd_decode_use_non_causal).
fn use_non_causal(c: Clip&) -> bool { c.proj == PROJ_GEMMA3 }
