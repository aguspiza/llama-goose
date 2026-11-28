// llama.cpp's llama-completion: text from a prompt, or, in conversation
// mode (the default when the model has a chat template), a chat whose user
// turns are read line by line from stdin; an empty input (EOF) ends it.
//   completion -m model.gguf [-p prompt] [-sys system] [-cnv | -no-cnv] ...
// The arguments are llama-completion's (common/arg; -h lists them).
// No -c (or -c 0): the model's context, fitted to the memory with -fit on|off, -fitt MiB, -fitc n: common/fit.
// port-of: tools/completion/completion.cpp
// port-of: tools/completion/main.cpp
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
import llama.chat;
import llama.grammar;
import common from "../common";
import common.jinja;
import common.chat;
import common.console;
import common.json_schema;
import common.download;
import common.fit;
import common.arg;

fn fail(msg: const u8[:]) {
    write_stderr(str("error: ", msg, "\n"));
    console::cleanup();
    exit(1);
}

// common_token_to_piece.
fn piece(t: i32, special: bool) -> u8[>..] {
    var out: u8[>..] = [];
    vocab::token_to_piece(t, out, 0, special);
    return out;
}

// The first n messages of the chat through a built-in template
// (common_chat_templates_apply_legacy), into out.
fn legacy_apply(tmpl: const u8[:], n: i64, add_ass: bool, out: u8[>..]&) -> bool {
    // the views are of a copy, which out cannot be
    var copy_text: u8[>..] = [];
    var ends: i64[>..] = [];
    for k in n {
        copy_text.append(chat_msgs[k].role);
        ends.push(copy_text.len);
        copy_text.append(chat_msgs[k].content);
        ends.push(copy_text.len);
    }
    var views: chat::Message[>..] = [];
    var start = 0;
    for k in n {
        views.push(chat::Message { role: copy_text[start..ends[2 * k]], content: copy_text[ends[2 * k]..ends[2 * k + 1]] });
        start = ends[2 * k + 1];
    }
    return chat::apply(tmpl, views, add_ass, out);
}

enum Conversation { Auto, Off, On }

// The text of the chat's messages; the messages are views of it.
var chat_text: u8[>..] = [];
var chat_msgs: common_chat::Msg[>..] = [];

fn main() {
    let args = console::utf8_args();
    arg::usage_text.append(str("\nexample usage:\n\n  text generation:     ", args[0], " -m your_model.gguf -p \"I believe the meaning of life is\" -n 128 -no-cnv\n\n  chat (conversation): ", args[0], " -m your_model.gguf -sys \"You are a helpful assistant\"\n\n"));
    if !arg::parse(args, arg::Example.Completion) { exit(1); }
    let p = arg::params;
    var model_path = str(p.model_path);
    var prompt = str(p.prompt);
    let system_prompt = p.system_prompt;
    let chat_template = p.chat_template;
    let input_prefix = p.input_prefix;
    let input_suffix = p.input_suffix;
    let antiprompt_text = p.antiprompt;   // the reverse prompts, each ended by a NUL
    let n_predict = p.n_predict;
    var n_ctx = p.n_ctx;     // 0: the model's, fitted to the memory (common_fit_params)
    let n_batch = p.n_batch;
    var n_keep = p.n_keep;
    var ctx_shift = p.ctx_shift;
    let ga_n = p.grp_attn_n;     // self-extend: the group-attention factor
    let ga_w = p.grp_attn_w;     // and width
    let n_threads = if p.n_threads > 0 { p.n_threads as! i32 } else { ggml::default_threads() };   // postprocess_cpu_params
    var conversation = match p.conversation_mode { Auto => Conversation.Auto, Disabled => Conversation.Off, Enabled => Conversation.On };
    let single_turn = p.single_turn;
    var interactive = p.interactive;
    var interactive_first = p.interactive_first;
    let use_jinja = p.use_jinja;
    let enable_chat_template = p.enable_chat_template;
    let input_prefix_bos = p.input_prefix_bos;
    let special = p.special;
    let display_prompt = p.display_prompt;
    let escape = p.escape;
    let multiline_input = p.multiline_input;
    let verbose_prompt = p.verbose_prompt;
    let use_color = p.use_color;
    var sp = p.sampling;
    let grammar_text = p.grammar;
    console::init(use_color);

    var m = model::Model { .. };
    model::load(m, model_path);
    var c = context::Context { n_threads: n_threads, n_ubatch: min(n_batch, p.n_ubatch), .. };
    context::init(c, m, fit::n_ctx(c, m, n_ctx));
    n_ctx = c.n_ctx;
    if ctx_shift && !context::can_shift(c, m) {
        write_stderr("common_init_from_params: KV cache shifting is not supported for this context, disabling KV cache shifting\n");
        ctx_shift = false;
    }

    sampler::apply_model_defaults(sp, m.meta);
    var ch = sampler::Chain { .. };
    sampler::init(ch, sp);
    if grammar_text.len > 0 && !sampler::set_grammar(ch, grammar_text, "root") {
        write_stderr(str("parse: error parsing grammar: ", grammar::err, "\n\n", grammar_text, "\n"));
        fail("failed to parse grammar");
    }

    // the chat templates: the model's, or the one given
    var tmpl: u8[>..] = [];
    var tool_use: u8[>..] = [];
    if chat_template.len > 0 { tmpl.append(chat_template); }
    else if !sampler::meta_str(m.meta, "tokenizer.chat_template", tmpl) { tmpl.clear(); }
    if !sampler::meta_str(m.meta, "tokenizer.chat_template.tool_use", tool_use) { tool_use.clear(); }
    let bos = if vocab::special_bos_id != vocab::TOKEN_NULL { piece(vocab::special_bos_id, true) } else { str("") };
    let eos = if vocab::special_eos_id != vocab::TOKEN_NULL { piece(vocab::special_eos_id, true) } else { str("") };
    var ts, ts_ok = common_chat::init(tmpl, tool_use, bos, eos, vocab::add_bos, vocab::add_eos);
    if !ts_ok { fail(str("common_chat_templates_init: failed to initialize chat template: ", jinja::err)); }

    let has_chat_template = ts.has_explicit_template;
    if conversation == Conversation.Auto { conversation = if has_chat_template { Conversation.On } else { Conversation.Off }; }
    if conversation == Conversation.On && !has_chat_template {
        write_stderr("warning: chat template is not available or is not supported. This may cause the model to output suboptimal responses\n");
    }
    let chat_mode = conversation == Conversation.On && enable_chat_template;

    let add_bos = vocab::add_bos && !use_jinja;
    var embd_inp: i32[>..] = [];
    var waiting_for_first_input = false;
    var formatted: u8[>..] = [];

    // the message added to the chat, and what it adds to the prompt
    fn chat_add_and_format(role: const u8[:], content: const u8[:]) {
        let r0 = chat_text.len;
        chat_text.append(role);
        let c0 = chat_text.len;
        chat_text.append(content);
        let msg = common_chat::Msg { role: chat_text[r0..c0], content: chat_text[c0..chat_text.len] };
        formatted.clear();
        if !format_single(msg, role == "user") { fail(str("failed to apply the chat template: ", jinja::err)); }
        chat_msgs.push(msg);
    }
    fn format_single(msg: common_chat::Msg, add_ass: bool) -> bool {
        if use_jinja { return common_chat::format_single(ts, chat_msgs, msg, add_ass, formatted); }
        // the built-in templates (common_chat_templates_apply_legacy)
        var before: u8[>..] = [];
        if chat_msgs.len > 0 && !legacy_apply(tmpl, chat_msgs.len, false, before) { return false; }
        chat_msgs.push(msg);
        var after: u8[>..] = [];
        let ok = legacy_apply(tmpl, chat_msgs.len, add_ass, after);
        chat_msgs.pop();
        if !ok { return false; }
        if add_ass && before.len > 0 && before[before.len - 1] == '\n' { formatted.push('\n'); }
        formatted.append(after[min(before.len, after.len)..]);
        return true;
    }

    var prompt_text: u8[>..] = [];
    if chat_mode {
        if system_prompt.len > 0 { chat_add_and_format("system", system_prompt); }
        if prompt.len > 0 { chat_add_and_format("user", prompt); } else { waiting_for_first_input = true; }
        if system_prompt.len > 0 || prompt.len > 0 {
            if use_jinja {
                if !common_chat::apply(ts, chat_msgs, common_chat::Inputs { add_generation_prompt: prompt.len > 0, .. }) {
                    fail(str("failed to apply the chat template: ", jinja::err));
                }
                prompt_text.append(common_chat::prompt);
            } else if !legacy_apply(tmpl, chat_msgs.len, prompt.len > 0, prompt_text) {
                fail("this custom template is not supported, try using --jinja");
            }
        }
    } else {
        prompt_text.append(prompt);
    }
    vocab::tokenize(prompt_text, true, true, embd_inp);

    if !waiting_for_first_input && embd_inp.len == 0 {
        if add_bos { embd_inp.push(vocab::special_bos_id); }
        else { fail("input is empty"); }
    }
    if embd_inp.len > n_ctx - 4 { fail(str("prompt is too long (", embd_inp.len, " tokens, max ", n_ctx - 4, ")")); }
    if n_keep < 0 || n_keep > embd_inp.len { n_keep = embd_inp.len; } else if add_bos { n_keep += 1; }
    if verbose_prompt {
        write_stderr(str("main: prompt: '", prompt, "'\nmain: number of tokens in prompt = ", embd_inp.len, "\n"));
        for t in embd_inp {
            var line: u8[>..] = [];
            var num: u8[>..] = [];
            format(num, t);
            for k in max(0, 6 - num.len) { line.push(' '); }
            line.append(num);
            format(line, " -> '", piece(t, true), "'\n");
            write_stderr(line);
        }
    }

    if conversation == Conversation.On {
        if single_turn && prompt.len > 0 { interactive = false; interactive_first = false; }
        else { interactive_first = true; }
    }
    if interactive_first { interactive = true; }

    // the reverse prompts, and those that are one token
    var antiprompts: const u8[:][>..] = [];
    {
        var start = 0;
        for ch0, k in antiprompt_text {
            if ch0 == 0 { antiprompts.push(antiprompt_text[start..k]); start = k + 1; }
        }
    }
    var antiprompt_token: i32[>..] = [];
    for ap in antiprompts {
        var ids: i32[>..] = [];
        vocab::tokenize(ap, false, true, ids);
        if ids.len == 1 { antiprompt_token.push(ids[0]); }
    }

    var is_interacting = false;
    if interactive { is_interacting = interactive_first; }
    var is_antiprompt = false;
    var input_echo = true;
    var display = true;
    var n_remain = n_predict;
    var ga_i = 0;               // the positions grouped so far
    if ga_n != 1 {
        if ga_n <= 0 { fail("grp_attn_n must be positive"); }
        if ga_w % ga_n != 0 { fail("grp_attn_w must be a multiple of grp_attn_n"); }
    }
    var n_consumed = 0;
    var assistant: u8[>..] = [];
    // the sampler's recent tokens (common_sampler's prev), newest last
    var history: i32[>..<] = [];
    fn remember(t: i32) {
        if history.len == 64 {
            for k in 63 { history[k] = history[k + 1]; }
            history.pop();
        }
        history.push(t);
    }

    console::set_display(console::Display.Prompt);
    display = display_prompt;

    let t_start = clock();
    var n_generated = 0;
    var logits: f32[>..] = [];
    var cands: sampler::Cand[>..] = [];
    var embd: i32[>..] = [];
    var line: u8[>..] = [];
    var buffer: u8[>..] = [];
    while (n_remain != 0 && !is_antiprompt) || interactive {
        // predict
        if embd.len > 0 {
            let max_embd = n_ctx - 4;
            if embd.len > max_embd {
                let skipped = embd.len - max_embd;
                embd.resize(max_embd);
                console::set_display(console::Display.Error);
                write_stderr(str("<<input too long: skipped ", skipped, " token", if skipped != 1 { "s" } else { "" }, ">>"));
                console::set_display(console::Display.Reset);
            }
            if ga_n == 1 {
                // infinite text generation: when the context is full, the n_keep
                // first tokens kept and half of the rest after them dropped
                if c.n_past + embd.len >= n_ctx {
                    if !ctx_shift {
                        write_stderr("\n\nmain: context full and context shift is disabled => stopping\n");
                        break;
                    }
                    if n_predict == -2 {
                        write_stderr(str("\n\nmain: context full and n_predict == ", n_predict, " => stopping\n"));
                        break;
                    }
                    let n_left = c.n_past - n_keep;
                    let n_discard = n_left / 2;
                    context::seq_rm(c, 0, n_keep, n_keep + n_discard);
                    context::seq_add(c, 0, n_keep + n_discard, c.n_past, -n_discard);
                }
            } else {
                // Self-Extend: each window of ga_w positions from ga_i grouped into ga_w / ga_n
                while c.n_past >= ga_i + ga_w {
                    let n_past = c.n_past;
                    let ib = (ga_n * ga_i) / ga_w;
                    let bd = (ga_w / ga_n) * (ga_n - 1);
                    let dd = (ga_w / ga_n) - ib * bd - ga_w;
                    context::seq_add(c, 0, ga_i, n_past, ib * bd);
                    context::seq_div(c, 0, ga_i + ib * bd, ga_i + ib * bd + ga_w, ga_n);
                    context::seq_add(c, 0, ga_i + ib * bd + ga_w, n_past + ib * bd, dd);
                    c.n_past = n_past - bd;
                    ga_i += ga_w / ga_n;
                }
            }
            var start = 0;
            while start < embd.len {
                let end = min(start + n_batch, embd.len);
                context::decode(c, m, embd[start..end], logits);
                start = end;
            }
        }

        embd.clear();

        if embd_inp.len <= n_consumed && !is_interacting {
            let id = sampler::sample(ch, cands, logits);
            remember(id);
            embd.push(id);
            n_generated++;
            if conversation == Conversation.On && !waiting_for_first_input && !vocab::is_eog(id) {
                vocab::token_to_piece(id, assistant, 0, false);
            }
            input_echo = true;
            n_remain--;
        } else {
            // the prompt or the user's input, a batch at a time
            while embd_inp.len > n_consumed {
                embd.push(embd_inp[n_consumed]);
                sampler::accept(ch, embd_inp[n_consumed]);
                remember(embd_inp[n_consumed]);
                n_consumed++;
                if embd.len == n_batch { break; }
            }
        }

        if input_echo && display {
            var text: u8[>..] = [];
            for id in embd { vocab::token_to_piece(id, text, 0, special); }
            write_stdout(text);
            flush_stdout();
        }

        // the color back to normal when no input is left
        if input_echo && embd_inp.len == n_consumed {
            console::set_display(console::Display.Reset);
            display = true;
        }

        if embd_inp.len <= n_consumed {
            // a reverse prompt at the end of the output
            if antiprompts.len > 0 {
                let n_prev = min(32, history.len);
                var last_output: u8[>..] = [];
                for k in history.len - n_prev..history.len { vocab::token_to_piece(history[k], last_output, 0, true); }
                is_antiprompt = false;
                for ap in antiprompts {
                    let extra = if interactive { 0 } else { 2 };
                    let from = if last_output.len > ap.len + extra { last_output.len - (ap.len + extra) } else { 0 };
                    if find(last_output[from..], ap) >= 0 {
                        if interactive { is_interacting = true; }
                        is_antiprompt = true;
                        break;
                    }
                }
                if last_output.len > 0 {
                    let last = history[history.len - 1];
                    for t in antiprompt_token {
                        if t == last {
                            if interactive { is_interacting = true; }
                            is_antiprompt = true;
                            break;
                        }
                    }
                }
            }

            // the end of a turn
            if !waiting_for_first_input && history.len > 0 && vocab::is_eog(history[history.len - 1]) {
                if interactive {
                    if antiprompts.len > 0 {
                        var ids: i32[>..] = [];
                        vocab::tokenize(antiprompts[0], false, true, ids);
                        embd_inp.append(ids);
                        is_antiprompt = true;
                    }
                    if enable_chat_template { chat_add_and_format("assistant", assistant); }
                    is_interacting = true;
                    write_stdout("\n");
                }
            }

            if conversation == Conversation.On && !waiting_for_first_input {
                if prompt_text.len > 0 {
                    prompt_text.clear();
                    is_interacting = false;
                }
            }

            if (c.n_past > 0 || waiting_for_first_input) && is_interacting {
                if conversation == Conversation.On { write_stdout("\n> "); }
                if input_prefix_bos { embd_inp.push(vocab::special_bos_id); }
                buffer.clear();
                if input_prefix.len > 0 && conversation != Conversation.On { write_stdout(input_prefix); }
                console::set_display(console::Display.UserInput);
                display = display_prompt;
                loop {
                    let more = console::readline(line, multiline_input);
                    buffer.append(line);
                    if !more { break; }
                }
                console::set_display(console::Display.Reset);
                display = true;

                if buffer.len == 0 {
                    write_stdout("EOF by user\n");
                    break;
                }
                if buffer[buffer.len - 1] == '\n' { buffer.pop(); }

                if buffer.len > 0 {
                    if input_suffix.len > 0 && conversation != Conversation.On { write_stdout(input_suffix); }
                    if escape { arg::process_escapes(buffer); }
                    let format_chat = conversation == Conversation.On && enable_chat_template;
                    if format_chat { chat_add_and_format("user", buffer); }
                    var pfx: i32[>..] = [];
                    var inp: i32[>..] = [];
                    var sfx: i32[>..] = [];
                    vocab::tokenize(input_prefix, false, true, pfx);
                    vocab::tokenize(if format_chat { formatted[..] } else { buffer[..] }, false, format_chat, inp);
                    vocab::tokenize(input_suffix, false, true, sfx);
                    embd_inp.append(pfx);
                    embd_inp.append(inp);
                    embd_inp.append(sfx);
                    assistant.clear();
                    n_remain -= inp.len;
                }
                input_echo = false;
            }

            if c.n_past > 0 || waiting_for_first_input {
                if is_interacting {
                    sampler::reset(ch);
                    history.clear();
                }
                is_interacting = false;
                if waiting_for_first_input && single_turn { interactive = false; interactive_first = false; }
                waiting_for_first_input = false;
            }
        }

        // the end of the text
        if embd.len > 0 && vocab::is_eog(embd[embd.len - 1]) && !interactive {
            write_stdout(" [end of text]\n");
            break;
        }

        // interactive: back to the user after n_predict tokens
        if interactive && n_remain <= 0 && n_predict >= 0 {
            n_remain = n_predict;
            is_interacting = true;
        }
    }

    write_stdout("\n\n");
    flush_stdout();
    console::cleanup();
    let dt = clock() - t_start;
    write_stderr(str("completion: generated ", n_generated, " tokens in ", dt, " s, speed: ", n_generated / dt, " t/s\n"));
}
