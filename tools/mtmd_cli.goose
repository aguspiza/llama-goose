// llama.cpp's tools/mtmd/mtmd-cli, its single turn: a prompt and its images,
// through the model's chat template, then the response.
//   mtmd_cli -m model.gguf --mmproj mmproj.gguf --image file [--image file ...] -p prompt
//            [-n n_predict] [-c n_ctx] [-t threads] [--temp t] [-s seed] [-sys prompt]
// -m and --mmproj can be -hf <user>/<model>[:quant] [-hff file] [-hft token] [--offline]
// [--no-mmproj]: common/download, with the mmproj beside the model.
// The weights: -lm auto|none|mmap|mlock|mmap+mlock|dio, -lzm on|auto|off, --repack | -nr, -ot <pattern>=CPU, -cmoe, -ncmoe N: common/load.
// No -c (or -c 0): the model's context, fitted to the memory with -fit on|off, -fitt MiB, -fitc n: common/fit.
// The prompt holds a <__media__> marker for each image, or they go before it.
// port-of: tools/mtmd/mtmd-cli.cpp
import std;
import os;
import ggml from "../ggml";
import ggml_cpu from "../ggml/cpu";
import ggml_shim from "../ggml/shim";
import llama from "../llama";
import llama.vocab;
import llama.model;
import llama.context;
import llama.sampler;
import common from "../common";
import common.jinja;
import common.chat;
import common.download;
import common.load;
import common.fit;
import mtmd from "../mtmd";

fn usage() {
    write_stderr("usage: mtmd_cli -m model.gguf --mmproj mmproj.gguf --image file [--image file ...] -p prompt\n");
    write_stderr("                (or -hf <user>/<model>[:quant] [-hff file] [-hft token] [--offline] [--no-mmproj])\n");
    write_stderr("                [-n n_predict] [-c n_ctx] [-t threads] [--temp t] [-s seed] [-sys prompt]\n");
    exit(1);
}

fn fail(msg: const u8[:]) {
    write_stderr(str("error: ", msg, "\n"));
    exit(1);
}

fn piece(t: i32, special: bool) -> u8[>..] {
    var out: u8[>..] = [];
    vocab::token_to_piece(t, out, 0, special);
    return out;
}

fn main() {
    let a = args();
    var model_path: u8[>..] = [];
    var mmproj_path: u8[>..] = [];
    var images: u8[>..] = [];           // the image paths, each ended by a NUL
    var prompt: u8[>..] = [];
    var system_prompt: u8[>..] = [];
    var n_predict = -1;
    var n_ctx = 0;     // the model's, fitted to the memory (common_fit_params)
    var n_threads: i32 = 8;
    var sp = sampler::Params { .. };
    var no_mmproj = false;
    var i = 1;
    while i < a.len {
        let k = a[i];
        if k == "--offline" { download::offline = true; i += 1; continue; }
        if load::flag(k) { i += 1; continue; }
        if k == "--no-mmproj" { no_mmproj = true; i += 1; continue; }
        guard i + 1 < a.len else { usage(); }
        let v = a[i + 1];
        let n, ok = parse_int(v);
        if k == "-m" || k == "--model" { model_path.append(v); }
        else if k == "--mmproj" { mmproj_path.append(v); }
        else if download::takes_value(k) { download::set(k, v); }
        else if load::takes_value(k) { load::set(k, v); }
        else if k == "--image" { images.append(v); images.push(0); }
        else if k == "-p" || k == "--prompt" { prompt.append(v); }
        else if k == "-sys" || k == "--system-prompt" { system_prompt.append(v); }
        else if (k == "-n" || k == "--n-predict") && ok { n_predict = n; }
        else if (k == "-c" || k == "--ctx-size") && ok { n_ctx = n; fit::ctx_flag(n); }
        else if fit::takes_value(k) { fit::set(k, v); }
        else if (k == "-t" || k == "--threads") && ok { n_threads = n as! i32; }
        else if (k == "-s" || k == "--seed") && ok { sp.seed = (n & 0xFFFFFFFF) as! u32; }
        else if k == "--temp" {
            let f, fok = parse_flt(v);
            guard fok else { usage(); }
            sp.temp = max(f as f32, 0.0);
            sp.user_set |= sampler::USER_TEMP;
        }
        else { usage(); }
        i += 2;
    }
    download::resolve(model_path, mmproj_path, !no_mmproj);
    if model_path.len == 0 || mmproj_path.len == 0 || images.len == 0 { usage(); }

    var m = model::Model { .. };
    load::from_env();
    model::load(m, model_path);
    var c = context::Context { n_threads: n_threads, .. };
    context::init(c, m, fit::n_ctx(c, m, n_ctx));
    sampler::apply_model_defaults(sp, m.meta);
    var ch = sampler::Chain { .. };
    sampler::init(ch, sp);

    var clip = mtmd::Clip { .. };
    mtmd::load_clip(clip, mmproj_path, n_threads);
    guard mtmd::n_mmproj_embd(clip) == m.hp.n_embd else {
        fail(str("mismatch between text model (n_embd = ", m.hp.n_embd, ") and mmproj (n_embd = ", mtmd::n_mmproj_embd(clip), ")"));
    }

    // the chat template (common_chat_templates_init), as --jinja
    var tmpl: u8[>..] = [];
    var tool_use: u8[>..] = [];
    if !sampler::meta_str(m.meta, "tokenizer.chat_template", tmpl) { tmpl.clear(); }
    if !sampler::meta_str(m.meta, "tokenizer.chat_template.tool_use", tool_use) { tool_use.clear(); }
    let bos = if vocab::special_bos_id != vocab::TOKEN_NULL { piece(vocab::special_bos_id, true) } else { str("") };
    let eos = if vocab::special_eos_id != vocab::TOKEN_NULL { piece(vocab::special_eos_id, true) } else { str("") };
    var ts, ts_ok = common_chat::init(tmpl, tool_use, bos, eos, vocab::add_bos, vocab::add_eos);
    if !ts_ok { fail(str("failed to initialize the chat template: ", jinja::err)); }

    // a marker before the prompt for each image, unless it has them
    var n_images = 0;
    for b in images { if b == 0 { n_images++; } }
    var content: u8[>..] = [];
    if find(prompt, mtmd::MARKER) < 0 {
        for _ in n_images { content.append(mtmd::MARKER); }
    }
    content.append(prompt);

    var chat_text: u8[>..] = [];
    var chat_msgs: common_chat::Msg[>..] = [];
    var n_past = 0;
    var logits: f32[>..] = [];

    // a message through the template, its chunks evaluated (eval_message)
    fn eval_message(role: const u8[:], text: const u8[:], with_images: bool) {
        let add_bos = chat_msgs.len == 0;
        let r0 = chat_text.len;
        chat_text.append(role);
        let c0 = chat_text.len;
        chat_text.append(text);
        let msg = common_chat::Msg { role: chat_text[r0..c0], content: chat_text[c0..chat_text.len] };
        var formatted: u8[>..] = [];
        if !common_chat::format_single(ts, chat_msgs, msg, role == "user", formatted) {
            fail(str("failed to apply the chat template: ", jinja::err));
        }
        chat_msgs.push(msg);
        mtmd::clear_bitmaps();
        if with_images {
            var start = 0;
            for b, j in images {
                guard b == 0 else { continue; }
                var bytes: u8[>..] = [];
                if !read_file(images[start..j], bytes) || !mtmd::add_bitmap(bytes) {
                    fail(str("failed to load image ", images[start..j]));
                }
                start = j + 1;
            }
        }
        if !mtmd::tokenize(clip, formatted, add_bos) { fail("failed to tokenize the prompt"); }
        let n_batch = 2048;
        for k, ci in mtmd::chunks {
            let last = ci == mtmd::chunks.len - 1;
            if k.kind == mtmd::ChunkKind.Text {
                // mtmd_helper_eval_chunk_single: n_batch tokens at a time
                let toks = mtmd::chunk_text(k);
                var s = 0;
                while s < toks.len {
                    let e = min(s + n_batch, toks.len);
                    var pos: i64[>..] = [];
                    var seqs: u64[>..] = [];
                    for j in e - s { pos.push(n_past + j); seqs.push(1); }
                    context::decode_seqs(c, m, toks[s..e], pos, seqs, logits, [], c.n_ubatch);
                    n_past += e - s;
                    s = e;
                }
            } else {
                var img = mtmd::ImageF32 { nx: k.nx, ny: k.ny, buf: [] };
                img.buf.append(mtmd::chunk_image(k));
                var embd: f32[>..] = [];
                mtmd::encode_image(clip, img, embd);
                var pos: i64[>..] = [];
                for j in k.n { pos.push(n_past + j); }
                context::decode_embd(c, m, embd, pos, 0, c.n_ubatch, !mtmd::use_non_causal(clip));
                n_past += k.n;
                guard !last else { fail("the prompt must end with text"); }
            }
        }
        write_stdout("\n");
    }

    if system_prompt.len > 0 { eval_message("system", system_prompt, false); }
    eval_message("user", content, true);

    // generate_response
    let n_max = if n_predict < 0 { 1 << 62 } else { n_predict };
    var cands: sampler::Cand[>..] = [];
    for _ in n_max {
        let tok = sampler::sample(ch, cands, logits);
        sampler::accept(ch, tok);
        if vocab::is_eog(tok) { write_stdout("\n"); break; }
        write_stdout(piece(tok, false));
        flush_stdout();
        let pos: i64[..1] = [n_past];
        let seqs: u64[..1] = [1];
        context::decode_seqs(c, m, [tok], pos[..], seqs[..], logits);
        n_past++;
    }
    write_stdout("\n\n");
}
