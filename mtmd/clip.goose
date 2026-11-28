// A vision encoder (an mmproj file): its hyperparameters and weights, the
// graph of a vision transformer and its projector, and the encoding of an
// image into embeddings the text model reads. So far SigLIP with the
// projectors of gemma3 and idefics3 (SmolVLM).
// port-of: tools/mtmd/mtmd-model.cpp tools/mtmd/mtmd-model.h tools/mtmd/mtmd-graph.cpp tools/mtmd/models/siglip.cpp
// port-of: tools/mtmd/mtmd-subcontext.cpp tools/mtmd/mtmd-impl.h
import std;
import math;
import ggml from "../ggml";
import ggml.gguf;
import .image;
namespace mtmd;

// enum projector_type, the ones ported
enum Projector { Unknown, Gemma3, Idefics3 }

enum Ffn { Gelu, Silu, GeluQuick }

struct ClipLayer {
    q_w: ggml::Tensor, q_b: ggml::Tensor,
    k_w: ggml::Tensor, k_b: ggml::Tensor,
    v_w: ggml::Tensor, v_b: ggml::Tensor,
    o_w: ggml::Tensor, o_b: ggml::Tensor,
    ln_1_w: ggml::Tensor, ln_1_b: ggml::Tensor,
    ln_2_w: ggml::Tensor, ln_2_b: ggml::Tensor,
    ff_up_w: ggml::Tensor, ff_up_b: ggml::Tensor,
    ff_gate_w: ggml::Tensor, ff_gate_b: ggml::Tensor,
    ff_down_w: ggml::Tensor, ff_down_b: ggml::Tensor,
}

struct Clip {
    meta: gguf::Ctx,
    ctx: ggml::Ctx,
    buf: ggml::Buffer,
    backend: ggml::Backend,
    galloc: ggml::Gallocr,
    n_threads: i32,
    proj: Projector,
    n_embd: i64,
    n_head: i64,
    n_ff: i64,
    n_layer: i64,
    projection_dim: i64,
    eps: f32,
    ffn_op: Ffn,
    image_size: i64,
    patch_size: i64,
    n_merge: i64,
    image_longest_edge: i64,
    resize_algo: Resize,
    image_mean: f32[..3],
    image_std: f32[..3],
    patch_embd: ggml::Tensor,
    patch_bias: ggml::Tensor,
    pos_embd: ggml::Tensor,
    pre_ln_w: ggml::Tensor, pre_ln_b: ggml::Tensor,
    post_ln_w: ggml::Tensor, post_ln_b: ggml::Tensor,
    mm_fc_w: ggml::Tensor,                  // idefics3
    mm_input_proj_w: ggml::Tensor,          // gemma3
    mm_soft_emb_norm_w: ggml::Tensor,
    layers: ClipLayer[>..],
}

fn key_u32(c: Clip&, key: const u8[:], required: bool = true) -> i64 {
    let id = gguf::find_key(c.meta, key);
    if id < 0 {
        if required { abort(str("mmproj: key not found: ", key)); }
        return 0;
    }
    return gguf::get_val_u32(c.meta, id);
}

fn key_f32(c: Clip&, key: const u8[:]) -> f32 {
    let id = gguf::find_key(c.meta, key);
    guard id >= 0 else { abort(str("mmproj: key not found: ", key)); }
    return gguf::get_val_f32(c.meta, id);
}

fn key_bool(c: Clip&, key: const u8[:]) -> bool {
    let id = gguf::find_key(c.meta, key);
    return id >= 0 && gguf::get_val_bool(c.meta, id);
}

fn key_f32x3(c: Clip&, key: const u8[:], out: f32[..3]&) {
    let id = gguf::find_key(c.meta, key);
    guard id >= 0 && gguf::get_arr_n(c.meta, id) >= 3 else { abort(str("mmproj: ", key, " must have at least 3 elements")); }
    var v: f32[>..] = [];
    v.resize(3, 0.0);
    gguf::get_arr_f32(c.meta, id, v[..]);
    out.clear();
    for x in v { out.push(x); }
}

fn get_tensor(c: Clip&, name: const u8[:], required: bool = false) -> ggml::Tensor {
    let t = ggml::get_tensor(c.ctx, name);
    if t.p == 0 && required { abort(str("mmproj: missing tensor '", name, "'")); }
    return t;
}

fn layer_tensor(c: Clip&, il: i64, name: const u8[:], suffix: const u8[:], required: bool = false) -> ggml::Tensor {
    return get_tensor(c, str("v.blk.", il, ".", name, ".", suffix), required);
}

// Loads an mmproj file's vision encoder (mtmd_model_load).
fn load_clip(c: Clip&, path: const u8[:], n_threads: i32) {
    c.ctx = ggml::Ctx { p: 0 };
    c.meta = gguf::init_with_tensors(path, true, c.ctx);
    guard c.meta.p != 0 else { abort(str("failed to load the mmproj ", path)); }
    guard key_bool(c, "clip.has_vision_encoder") else { abort("mmproj: no vision encoder (audio is not ported yet)"); }
    var proj_name: u8[>..] = [];
    var id = gguf::find_key(c.meta, "clip.projector_type");
    if id < 0 { id = gguf::find_key(c.meta, "clip.vision.projector_type"); }
    if id >= 0 { gguf::get_val_str(c.meta, id, proj_name); }
    c.proj = if proj_name == "gemma3" { Projector.Gemma3 } else if proj_name == "idefics3" { Projector.Idefics3 } else { Projector.Unknown };
    guard c.proj != Projector.Unknown else { abort(str("mmproj: projector type '", proj_name, "' is not ported yet")); }

    c.n_embd = key_u32(c, "clip.vision.embedding_length");
    c.n_head = key_u32(c, "clip.vision.attention.head_count");
    c.n_ff = key_u32(c, "clip.vision.feed_forward_length");
    c.n_layer = key_u32(c, "clip.vision.block_count");
    c.projection_dim = key_u32(c, "clip.vision.projection_dim");
    c.eps = key_f32(c, "clip.vision.attention.layer_norm_epsilon");
    c.image_size = key_u32(c, "clip.vision.image_size");
    c.patch_size = key_u32(c, "clip.vision.patch_size");
    c.ffn_op = if key_bool(c, "clip.use_gelu") { Ffn.Gelu } else if key_bool(c, "clip.use_silu") { Ffn.Silu } else { Ffn.GeluQuick };
    key_f32x3(c, "clip.vision.image_mean", c.image_mean);
    key_f32x3(c, "clip.vision.image_std", c.image_std);
    let merge = key_u32(c, "clip.vision.projector.scale_factor", false);
    if c.proj == Projector.Gemma3 {
        c.n_merge = if merge > 0 { merge } else { 4 };
        c.resize_algo = Resize.Bilinear;
    } else {
        c.n_merge = merge;
        c.resize_algo = Resize.Lanczos;
        c.image_longest_edge = key_u32(c, "clip.vision.preproc_image_size", false);
    }

    c.pre_ln_w = get_tensor(c, "v.pre_ln.weight");
    c.pre_ln_b = get_tensor(c, "v.pre_ln.bias");
    c.post_ln_w = get_tensor(c, "v.post_ln.weight");
    c.post_ln_b = get_tensor(c, "v.post_ln.bias");
    c.patch_bias = get_tensor(c, "v.patch_embd.bias");
    c.patch_embd = get_tensor(c, "v.patch_embd.weight", true);
    c.pos_embd = get_tensor(c, "v.position_embd.weight");
    c.layers.clear();
    for il in c.n_layer {
        var ly = ClipLayer {
            q_w: layer_tensor(c, il, "attn_q", "weight"), q_b: layer_tensor(c, il, "attn_q", "bias"),
            k_w: layer_tensor(c, il, "attn_k", "weight"), k_b: layer_tensor(c, il, "attn_k", "bias"),
            v_w: layer_tensor(c, il, "attn_v", "weight"), v_b: layer_tensor(c, il, "attn_v", "bias"),
            o_w: layer_tensor(c, il, "attn_out", "weight", true), o_b: layer_tensor(c, il, "attn_out", "bias"),
            ln_1_w: layer_tensor(c, il, "ln1", "weight"), ln_1_b: layer_tensor(c, il, "ln1", "bias"),
            ln_2_w: layer_tensor(c, il, "ln2", "weight"), ln_2_b: layer_tensor(c, il, "ln2", "bias"),
            ff_up_w: layer_tensor(c, il, "ffn_up", "weight", true), ff_up_b: layer_tensor(c, il, "ffn_up", "bias"),
            ff_gate_w: layer_tensor(c, il, "ffn_gate", "weight"), ff_gate_b: layer_tensor(c, il, "ffn_gate", "bias"),
            ff_down_w: layer_tensor(c, il, "ffn_down", "weight", true), ff_down_b: layer_tensor(c, il, "ffn_down", "bias"),
        };
        guard ly.q_w.p != 0 && ly.k_w.p != 0 && ly.v_w.p != 0 else { abort("mmproj: a fused QKV is not ported yet"); }
        // old conversions named up and down the other way round
        if ggml::ne(ly.ff_down_w, 0) == c.n_embd {
            let t = ly.ff_up_w;
            ly.ff_up_w = ly.ff_down_w;
            ly.ff_down_w = t;
            let b = ly.ff_up_b;
            ly.ff_up_b = ly.ff_down_b;
            ly.ff_down_b = b;
        }
        c.layers.push(ly);
    }
    if c.proj == Projector.Idefics3 {
        c.mm_fc_w = get_tensor(c, "mm.model.fc.weight", true);
    } else {
        c.mm_input_proj_w = get_tensor(c, "mm.input_projection.weight", true);
        c.mm_soft_emb_norm_w = get_tensor(c, "mm.soft_emb_norm.weight", true);
    }

    c.backend = ggml::backend_cpu_init();
    c.buf = ggml::backend_alloc_ctx_tensors(c.ctx, c.backend);
    guard gguf::load_tensor_data(c.meta, c.ctx, path) else { abort(str("failed to read the tensors of ", path)); }
    c.galloc = ggml::gallocr_new(c.backend);
    c.n_threads = n_threads;
}

// The embeddings an image of nx by ny pixels becomes (n_mmproj_embd, n_output_tokens).
fn n_mmproj_embd(c: Clip&) -> i64 {
    return if c.proj == Projector.Idefics3 { ggml::ne(c.mm_fc_w, 1) } else { ggml::ne(c.mm_input_proj_w, 0) };
}

fn n_output_tokens(c: Clip&, nx: i64, ny: i64) -> i64 {
    let n = (nx / c.patch_size) * (ny / c.patch_size);
    return n / (c.n_merge * c.n_merge);
}

//
// the graph (mtmd_graph)
//

struct G { ctx: ggml::Ctx, gf: ggml::Graph, nx: i64, ny: i64 }

let CLIP_MAX_NODES = 8192;

fn build_norm(g: G&, c: Clip&, cur_in: ggml::Tensor, w: ggml::Tensor, b: ggml::Tensor, rms: bool) -> ggml::Tensor {
    var cur = if rms { ggml::rms_norm(g.ctx, cur_in, c.eps) } else { ggml::norm(g.ctx, cur_in, c.eps) };
    if w.p != 0 { cur = ggml::mul(g.ctx, cur, w); }
    if b.p != 0 { cur = ggml::add(g.ctx, cur, b); }
    return cur;
}

fn build_ffn(g: G&, c: Clip&, cur_in: ggml::Tensor, ly: ClipLayer) -> ggml::Tensor {
    let ctx = g.ctx;
    var tmp = ggml::mul_mat(ctx, ly.ff_up_w, cur_in);
    if ly.ff_up_b.p != 0 { tmp = ggml::add(ctx, tmp, ly.ff_up_b); }
    var cur = tmp;
    if ly.ff_gate_w.p != 0 {
        cur = ggml::mul_mat(ctx, ly.ff_gate_w, cur_in);
        if ly.ff_gate_b.p != 0 { cur = ggml::add(ctx, cur, ly.ff_gate_b); }
        cur = match c.ffn_op {
            Silu => ggml::swiglu_split(ctx, cur, tmp),
            Gelu => ggml::geglu_split(ctx, cur, tmp),
            _ => ggml::geglu_quick_split(ctx, cur, tmp),
        };
    } else {
        cur = match c.ffn_op {
            Silu => ggml::silu(ctx, cur),
            Gelu => ggml::gelu(ctx, cur),
            _ => ggml::gelu_quick(ctx, cur),
        };
    }
    cur = ggml::mul_mat(ctx, ly.ff_down_w, cur);
    if ly.ff_down_b.p != 0 { cur = ggml::add(ctx, cur, ly.ff_down_b); }
    return cur;
}

// Attention over all the positions, with flash attention (build_attn, as
// mtmd picks on the CPU); q, k, v: [d_head, n_head, n_pos, 1].
fn build_attn(g: G&, ly: ClipLayer, q_cur: ggml::Tensor, k_cur: ggml::Tensor, v_cur: ggml::Tensor, kq_scale: f32) -> ggml::Tensor {
    let ctx = g.ctx;
    ggml::build_forward_order(g.gf, q_cur);
    ggml::build_forward_order(g.gf, k_cur);
    ggml::build_forward_order(g.gf, v_cur);
    let q = ggml::permute(ctx, q_cur, 0, 2, 1, 3);
    let k = ggml::cast(ctx, ggml::permute(ctx, k_cur, 0, 2, 1, 3), ggml::TYPE_F16);
    let v = ggml::cast(ctx, ggml::permute(ctx, v_cur, 0, 2, 1, 3), ggml::TYPE_F16);
    var cur = ggml::flash_attn_ext(ctx, q, k, v, ggml::NONE, kq_scale, 0.0, 0.0);
    ggml::prec_set_acc(cur, ggml::PREC_F32);
    cur = ggml::reshape_2d(ctx, cur, ggml::ne(cur, 0) * ggml::ne(cur, 1), ggml::ne(cur, 2) * ggml::ne(cur, 3));
    cur = ggml::mul_mat(ctx, ly.o_w, cur);
    if ly.o_b.p != 0 { cur = ggml::add(ctx, cur, ly.o_b); }
    return cur;
}

// The vision transformer (build_vit): inp [n_embd, n_pos], learned
// positions added; the output [n_embd, n_pos].
fn build_vit(g: G&, c: Clip&, inp_in: ggml::Tensor, n_pos: i64) -> ggml::Tensor {
    let ctx = g.ctx;
    let d_head = c.n_embd / c.n_head;
    let kq_scale: f32 = 1.0 / sqrt(d_head as f32);
    var inp = inp_in;
    if c.pos_embd.p != 0 { inp = ggml::add(ctx, inp, c.pos_embd); }
    inp = ggml::reshape_2d(ctx, inp, c.n_embd, n_pos);
    var inpL = inp;
    if c.pre_ln_w.p != 0 { inpL = build_norm(g, c, inpL, c.pre_ln_w, c.pre_ln_b, false); }
    for ly in c.layers {
        var cur = build_norm(g, c, inpL, ly.ln_1_w, ly.ln_1_b, false);
        var q = ggml::mul_mat(ctx, ly.q_w, cur);
        if ly.q_b.p != 0 { q = ggml::add(ctx, q, ly.q_b); }
        var k = ggml::mul_mat(ctx, ly.k_w, cur);
        if ly.k_b.p != 0 { k = ggml::add(ctx, k, ly.k_b); }
        var v = ggml::mul_mat(ctx, ly.v_w, cur);
        if ly.v_b.p != 0 { v = ggml::add(ctx, v, ly.v_b); }
        q = ggml::reshape_4d(ctx, q, d_head, c.n_head, n_pos, 1);
        k = ggml::reshape_4d(ctx, k, d_head, c.n_head, n_pos, 1);
        v = ggml::reshape_4d(ctx, v, d_head, c.n_head, n_pos, 1);
        cur = build_attn(g, ly, q, k, v, kq_scale);
        cur = ggml::add(ctx, cur, inpL);
        inpL = cur;
        cur = build_norm(g, c, cur, ly.ln_2_w, ly.ln_2_b, false);
        cur = build_ffn(g, c, cur, ly);
        cur = ggml::add(ctx, inpL, cur);
        inpL = cur;
    }
    if c.post_ln_w.p != 0 { inpL = build_norm(g, c, inpL, c.post_ln_w, c.post_ln_b, false); }
    return inpL;
}

fn align_to(x: i64, n: i64) -> i64 { (x + n - 1) / n * n }

// The pixel shuffle (build_patch_merge_permute): each scale x scale block of
// patches into one, its embeddings side by side.
fn build_patch_merge_permute(g: G&, c: Clip&, cur_in: ggml::Tensor, scale: i64) -> ggml::Tensor {
    let ctx = g.ctx;
    let n_embd = ggml::ne(cur_in, 0);
    var width = g.nx / c.patch_size;
    var height = g.ny / c.patch_size;
    let pad_w = align_to(width, scale) - width;
    let pad_h = align_to(height, scale) - height;
    var cur = ggml::reshape_3d(ctx, cur_in, n_embd, width, height);
    if pad_w != 0 || pad_h != 0 {
        cur = ggml::zero_pad(ctx, cur, 0, pad_w as i32, pad_h as i32, 0);
        width += pad_w;
        height += pad_h;
    }
    cur = ggml::reshape_3d(ctx, cur, n_embd * scale, width / scale, height);
    cur = ggml::permute(ctx, cur, 0, 2, 1, 3);
    cur = ggml::cont_3d(ctx, cur, n_embd * scale * scale, height / scale, width / scale);
    cur = ggml::permute(ctx, cur, 0, 2, 1, 3);
    cur = ggml::cont_2d(ctx, cur, ggml::ne(cur, 0), ggml::ne(cur, 1) * ggml::ne(cur, 2));
    return cur;
}

// The graph of SigLIP and its projector (mtmd_graph_siglip::build), for an
// image of nx by ny; inp_raw is its input, [nx, ny, 3, 1].
fn build_siglip(g: G&, c: Clip&) -> ggml::Tensor, ggml::Tensor {
    let ctx = g.ctx;
    let n_px = g.nx / c.patch_size;
    let n_py = g.ny / c.patch_size;
    let n_patches = n_px * n_py;
    let inp_raw = ggml::new_tensor_4d(ctx, ggml::TYPE_F32, g.nx, g.ny, 3, 1);
    ggml::set_name(inp_raw, "inp_raw");
    ggml::set_input(inp_raw);
    var inp = ggml::conv_2d(ctx, c.patch_embd, inp_raw, c.patch_size as i32, c.patch_size as i32, 0, 0, 1, 1);
    inp = ggml::reshape_3d(ctx, inp, n_patches, c.n_embd, 1);
    inp = ggml::cont(ctx, ggml::transpose(ctx, inp));
    if c.patch_bias.p != 0 { inp = ggml::add(ctx, inp, c.patch_bias); }
    var cur = build_vit(g, c, inp, n_patches);
    if c.proj == Projector.Gemma3 {
        guard n_px == n_py else { abort("gemma3: the image must be square"); }
        let k = c.n_merge as i32;
        cur = ggml::transpose(ctx, cur);
        cur = ggml::cont_4d(ctx, cur, n_px, n_px, c.n_embd, 1);
        cur = ggml::pool_2d(ctx, cur, ggml::OP_POOL_AVG, k, k, k, k, 0.0, 0.0);
        cur = ggml::reshape_3d(ctx, cur, ggml::ne(cur, 0) * ggml::ne(cur, 0), c.n_embd, 1);
        cur = ggml::cont(ctx, ggml::transpose(ctx, cur));
        cur = ggml::rms_norm(ctx, cur, c.eps);
        cur = ggml::mul(ctx, cur, c.mm_soft_emb_norm_w);
        cur = ggml::mul_mat(ctx, ggml::cont(ctx, ggml::transpose(ctx, c.mm_input_proj_w)), cur);
    } else {
        cur = build_patch_merge_permute(g, c, cur, c.n_merge);
        cur = ggml::mul_mat(ctx, c.mm_fc_w, cur);
    }
    ggml::build_forward_expand(g.gf, cur);
    return inp_raw, cur;
}

// The embeddings of one image (mtmd_subcontext::encode): its pixels
// normalized, [n_mmproj_embd, n_output_tokens] into out.
// Its graph has memory of its own: init_graph's is the text model's, whose
// graph a context keeps between decodes.
fn encode_image(c: Clip&, img: ImageF32&, out: f32[>..]&) {
    let gctx = ggml::init(CLIP_MAX_NODES * ggml::tensor_overhead() + ggml::graph_overhead_custom(CLIP_MAX_NODES, false), true);
    var g = G { ctx: gctx, gf: ggml::new_graph_custom(gctx, CLIP_MAX_NODES, false), nx: img.nx, ny: img.ny };
    let inp_raw, emb = build_siglip(g, c);
    guard ggml::gallocr_alloc_graph(c.galloc, g.gf) else { abort("mtmd: failed to allocate the graph"); }
    // the pixels by channel: R, then G, then B
    let n = img.nx * img.ny;
    var planes: f32[>..] = [];
    planes.resize(3 * n, 0.0);
    for i in n {
        planes[i] = img.buf[3 * i];
        planes[n + i] = img.buf[3 * i + 1];
        planes[2 * n + i] = img.buf[3 * i + 2];
    }
    ggml::backend_set_f32(inp_raw, planes);
    ggml::backend_cpu_set_n_threads(c.backend, c.n_threads);
    guard ggml::backend_graph_compute(c.backend, g.gf) == 0 else { abort("mtmd: graph compute failed"); }
    out.clear();
    out.resize(ggml::nelements(emb), 0.0);
    ggml::backend_get_f32(emb, out, 0);
    ggml::free(gctx);
}

//
// preprocessing (mtmd_image_preprocessor_*)
//

// An image ready for the encoder: the overview, if the model takes one, and
// the slices, all of one size, grid_x by grid_y, row after row.
root struct Preproc {
    has_overview: bool,
    overview: ImageF32,
    n_slices: i64,
    slice_nx: i64,
    slice_ny: i64,
    slices: f32[>..],
    grid_x: i64,
    grid_y: i64,
}

fn add_slice(c: Clip&, img: ImageU8&, out: Preproc&) {
    var f = ImageF32 { nx: 0, ny: 0, buf: [] };
    to_f32(img, true, c.image_mean[..], c.image_std[..], f);
    if out.n_slices == 0 {
        out.slice_nx = f.nx;
        out.slice_ny = f.ny;
    }
    guard f.nx == out.slice_nx && f.ny == out.slice_ny else { abort("mtmd: slices of different sizes"); }
    out.slices.append(f.buf);
    out.n_slices++;
}

fn round_by(x: f32, f: i64) -> i64 { (round(x / f) as i64) * f }

// The overview and the slices of an image (preprocess).
fn preprocess(c: Clip&, img: ImageU8&, out: Preproc&) {
    out.has_overview = false;
    out.n_slices = 0;
    out.slices.clear();
    out.grid_x = 0;
    out.grid_y = 0;
    let sz = c.image_size;
    if c.proj == Projector.Gemma3 {
        // mtmd_image_preprocessor_fixed_size
        var r = ImageU8 { nx: 0, ny: 0, buf: [] };
        resize_image_padded(img, sz, sz, c.resize_algo, r);
        add_slice(c, r, out);
        return;
    }
    // mtmd_image_preprocessor_idefics3
    var refined = ImageU8 { nx: 0, ny: 0, buf: [] };
    var overview = ImageU8 { nx: 0, ny: 0, buf: [] };
    if c.image_longest_edge > 0 {
        let le = c.image_longest_edge;
        let aspect: f64 = img.nx as f64 / img.ny;
        var w = 0;
        var h = 0;
        if img.nx >= img.ny {
            w = le;
            h = (le / aspect) as i64;
            h += h % 2;
        } else {
            h = le;
            w = (le * aspect) as i64;
            w += w % 2;
        }
        let gx = (w + sz - 1) / sz;
        let gy = (h + sz - 1) / sz;
        var resized = ImageU8 { nx: 0, ny: 0, buf: [] };
        resize_image(img, w, h, c.resize_algo, resized);
        resize_image(resized, gx * sz, gy * sz, c.resize_algo, refined);
        resize_image(refined, sz, sz, c.resize_algo, overview);
        out.grid_x = gx;
        out.grid_y = gy;
    } else {
        // calc_size_preserved_ratio aligned to the tile; the llava-uhd slicing, bicubic
        let w = max(sz, round_by(img.nx, sz));
        let h = max(sz, round_by(img.ny, sz));
        resize_image(img, sz, sz, Resize.Bicubic, overview);
        resize_image_padded(img, w, h, Resize.Bicubic, refined);
        out.grid_x = (w + sz - 1) / sz;
        out.grid_y = (h + sz - 1) / sz;
    }
    out.has_overview = true;
    to_f32(overview, true, c.image_mean[..], c.image_std[..], out.overview);
    for y in out.grid_y {
        for x in out.grid_x {
            var s = ImageU8 { nx: 0, ny: 0, buf: [] };
            crop_image(refined, x * sz, y * sz, min(sz, refined.nx - x * sz), min(sz, refined.ny - y * sz), s);
            add_slice(c, s, out);
        }
    }
}
