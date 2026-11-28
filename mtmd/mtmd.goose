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

enum ChunkKind { Text, Image }

// A piece of the prompt (mtmd_input_chunk): text tokens, chunk_tokens[at..at + n],
// or an image's pixels, chunk_pixels[at..at + nx * ny * 3], which encode into n
// tokens; media: the bitmap it comes from (-1 for text).
struct Chunk { kind: ChunkKind, at: i64, n: i64, nx: i64, ny: i64, media: i64 }

var chunks: Chunk[>..] = [];
var chunk_tokens: i32[>..] = [];
var chunk_pixels: f32[>..] = [];

// The images given for a prompt (mtmd_bitmap), decoded, in the order of its
// markers; each one's id, the SHA-256 of its file, in bitmap_ids[32 * i..].
struct Bitmap { nx: i64, ny: i64, at: i64 }
var bitmaps: Bitmap[>..] = [];
var bitmap_pixels: u8[>..] = [];
var bitmap_ids: u8[>..] = [];

fn clear_bitmaps() {
    bitmaps.clear();
    bitmap_pixels.clear();
    bitmap_ids.clear();
}

// An image file's bytes decoded into the next bitmap; false if they are not an image.
fn add_bitmap(bytes: const u8[:]) -> bool {
    var img = ImageU8 { nx: 0, ny: 0, buf: [] };
    guard decode_image(bytes, img) else { return false; }
    bitmaps.push(Bitmap { nx: img.nx, ny: img.ny, at: bitmap_pixels.len });
    bitmap_pixels.append(img.buf);
    let at = bitmap_ids.len;
    bitmap_ids.resize(at + 32, 0);
    c_sha256(bytes, bitmap_ids[at..at + 32]);
    return true;
}

fn bitmap_id(i: i64) -> const u8[:] { bitmap_ids[32 * i..32 * i + 32] }

// The bitmap whose chunks add_media is adding.
var cur_media = -1;

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
    if chunks.len == 0 || chunks[chunks.len - 1].kind != ChunkKind.Text || chunks[chunks.len - 1].at + chunks[chunks.len - 1].n != chunk_tokens.len {
        chunks.push(Chunk { kind: ChunkKind.Text, at: chunk_tokens.len, n: 0, nx: 0, ny: 0, media: -1 });
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
    chunks.push(Chunk { kind: ChunkKind.Image, at: chunk_pixels.len, n: n_output_tokens(c, img.nx, img.ny), nx: img.nx, ny: img.ny, media: cur_media });
    chunk_pixels.append(img.buf);
}

// An image's chunks, and the tokens around them (add_media).
fn add_media(c: Clip&, img: ImageU8&) {
    var pp = Preproc { .. };
    preprocess(c, img, pp);
    let n = pp.slice_nx * pp.slice_ny * 3;
    if c.proj == Projector.Gemma3 {
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
// add_special: the vocabulary's BOS first (and its EOS last). false when the
// markers are not as many as the images.
fn tokenize(c: Clip&, text: const u8[:], add_special: bool, marker: const u8[:] = MARKER) -> bool {
    chunks.clear();
    chunk_tokens.clear();
    chunk_pixels.clear();
    var i_img = 0;
    var start = 0;
    var i = 0;
    while i <= text.len {
        let at_marker = i + marker.len <= text.len && text[i..i + marker.len] == marker;
        if at_marker || i == text.len {
            if i > start { add_text(text[start..i], true); }
            if !at_marker { break; }
            guard i_img < bitmaps.len else { return markers_mismatch(text, marker); }
            let bm = bitmaps[i_img];
            var img = ImageU8 { nx: bm.nx, ny: bm.ny, buf: [] };
            img.buf.append(bitmap_pixels[bm.at..bm.at + bm.nx * bm.ny * 3]);
            cur_media = i_img;
            add_media(c, img);
            i_img++;
            i += marker.len;
            start = i;
            continue;
        }
        i++;
    }
    guard i_img == bitmaps.len else { return markers_mismatch(text, marker); }
    if add_special && vocab::add_bos {
        if chunks.len > 0 && chunks[0].kind == ChunkKind.Text {
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
            chunks.push(Chunk { kind: ChunkKind.Text, at: 0, n: 0, nx: 0, ny: 0, media: -1 });
            var j = chunks.len - 1;
            while j > 0 { chunks[j] = chunks[j - 1]; j--; }
            chunks[0] = Chunk { kind: ChunkKind.Text, at: at, n: 1, nx: 0, ny: 0, media: -1 };
        }
    }
    if add_special && vocab::add_eos { add_token(vocab::special_eos_id); }
    return true;
}

fn markers_mismatch(text: const u8[:], marker: const u8[:]) -> bool {
    var n = 0;
    var i = find(text, marker);
    while i >= 0 {
        n++;
        let j = find(text[i + marker.len..], marker);
        i = if j < 0 { -1 } else { i + marker.len + j };
    }
    write_stderr(str("mtmd_tokenize: error: number of bitmaps (", bitmaps.len, ") does not match number of markers (", n, ")\n"));
    return false;
}

// Whether a model's image tokens see each other (mtmd_decode_use_non_causal).
fn use_non_causal(c: Clip&) -> bool { c.proj == Projector.Gemma3 }
