// llama.cpp's examples/simple-chat: a chat with the model's built-in
// template, min-p 0.05, temperature 0.8. An empty line ends it.
//   simple_chat -m model.gguf [-c n_ctx] [-s seed]
// (-s is not in llama.cpp's: there the seed is always random.)
// port-of: examples/simple-chat/simple-chat.cpp
import std;
import os;
import ggml from "../ggml";
import ggml_cpu from "../ggml/cpu";
import ggml_shim from "../ggml/shim";
import ggml.gguf;
import llama from "../llama";
import llama.vocab;
import llama.model;
import llama.context;
import llama.sampler;
import llama.chat;

fn usage() {
    write_stdout("\nexample usage:\n\n    simple_chat -m model.gguf [-c context_size] [-ngl n_gpu_layers]\n\n");
    exit(1);
}

// A message: its role and content, as ranges of the text buffer.
struct Msg { assistant: bool, start: i64, end: i64 }

fn main() {
    let args = args();
    var model_path: u8[>..] = [];
    var n_ctx = 2048;
    var seed: u32 = 0xFFFFFFFF;
    var i = 1;
    while i < args.len {
        let a = args[i];
        guard i + 1 < args.len else { usage(); };
        i += 1;
        let v, ok = parse_int(args[i]);
        if a == "-m" { model_path.clear(); model_path.append(args[i]); }
        else if a == "-c" && ok { n_ctx = v; }
        else if a == "-ngl" && ok { }      // CPU only
        else if a == "-s" && ok { seed = (v & 0xFFFFFFFF) as! u32; }
        else { usage(); }
        i += 1;
    }
    if model_path.len == 0 { usage(); }

    llama::quiet = true;
    var m = model::Model { .. };
    model::load(m, model_path);
    var c = context::Context { n_threads: 4, .. };      // llama_context_default_params: GGML_DEFAULT_N_THREADS
    context::init(c, m, n_ctx);
    n_ctx = c.n_ctx;

    var ch = sampler::Chain { .. };
    var s = sampler::new_sampler(sampler::KIND_MIN_P);
    s.a = 0.05; s.min_keep = 1;
    ch.samplers.push(s);
    s = sampler::new_sampler(sampler::KIND_TEMP_EXT);
    s.a = 0.8;
    ch.samplers.push(s);
    ch.samplers.push(sampler::seeded(sampler::KIND_DIST, seed));

    var tmpl: u8[>..] = [];
    if !sampler::meta_str(m.meta, "tokenizer.chat_template", tmpl) { tmpl.clear(); }

    var text: u8[>..] = [];         // every message's content, back to back
    var msgs: Msg[>..] = [];
    var formatted: u8[>..] = [];
    var prev_len = 0;
    var user: u8[>..] = [];
    var tokens: i32[>..] = [];
    var logits: f32[>..] = [];
    var cands: sampler::Cand[>..] = [];
    var piece: u8[>..] = [];

    // The chat so far, through the template.
    fn format_chat(add_ass: bool) {
        var views: chat::Message[>..] = [];
        for msg in msgs {
            views.push(chat::Message { role: if msg.assistant { "assistant" } else { "user" }, content: text[msg.start..msg.end] });
        }
        if !chat::apply(tmpl, views, add_ass, formatted) {
            write_stderr("failed to apply the chat template\n");
            exit(1);
        }
    }

    loop {
        write_stdout("\x1b[32m> \x1b[0m");
        flush_stdout();
        user.clear();
        if !read_line(user) || user.len == 0 { break; }

        let start = text.len;
        text.append(user);
        msgs.push(Msg { assistant: false, start: start, end: text.len });
        format_chat(true);

        // the new part of the formatted chat
        write_stdout("\x1b[33m");
        let is_first = c.n_past == 0;
        tokens.clear();
        vocab::tokenize(formatted[prev_len..], is_first, true, tokens);
        let response_start = text.len;
        loop {
            if c.n_past + tokens.len > n_ctx {
                write_stdout("\x1b[0m\n");
                flush_stdout();
                write_stderr("context size exceeded\n");
                exit(0);
            }
            context::decode(c, m, tokens, logits);
            let t = sampler::sample(ch, cands, logits);
            if vocab::is_eog(t) { break; }
            piece.clear();
            vocab::token_to_piece(t, piece, 0, true);
            write_stdout(piece);
            flush_stdout();
            text.append(piece);
            tokens.clear();
            tokens.push(t);
        }
        write_stdout("\n\x1b[0m");

        msgs.push(Msg { assistant: true, start: response_start, end: text.len });
        format_chat(false);
        prev_len = formatted.len;
    }
}
