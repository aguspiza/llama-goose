// llama.cpp's llama-completion: text from a prompt, or, in conversation
// mode (the default when the model has a chat template), a chat whose user
// turns are read line by line from stdin; an empty input (EOF) ends it.
//   completion -m model.gguf [-p prompt] [-sys system] [-cnv | -no-cnv] [-st] [-i | -if]
//              [-r reverse-prompt]... [-n n_predict] [-c n_ctx] [-b n_batch] [-t threads] [-s seed]
//              [--chat-template tmpl] [--jinja | --no-jinja] [-co on|off|auto] [-sp] [--keep n]
//              [--no-display-prompt] [--no-escape] [-mli] [--in-prefix s] [--in-suffix s] [--in-prefix-bos]
//              [--temp t] [--top-k k] [--top-p p] [--min-p p] [--typical p]
//              [--repeat-penalty r] [--repeat-last-n n] [--presence-penalty p]
//              [--frequency-penalty f] [--xtc-probability p] [--xtc-threshold t]
//              [--top-nsigma n] [--dynatemp-range r] [--dynatemp-exp e]
// port-of: tools/completion/completion.cpp
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

fn fail(msg: const u8[:]) {
    write_stderr(str("error: ", msg, "\n"));
    console::cleanup();
    exit(1);
}

// string_process_escapes: \n \r \t \' \" \\ and \xHH, in place.
fn process_escapes(s: u8[>..]&) {
    var out: u8[>..] = [];
    var i = 0;
    while i < s.len {
        let c = s[i];
        if c != '\\' || i + 1 >= s.len { out.push(c); i++; continue; }
        i++;
        let e = s[i];
        if e == 'n' { out.push('\n'); }
        else if e == 'r' { out.push('\r'); }
        else if e == 't' { out.push('\t'); }
        else if e == '\'' || e == '"' || e == '\\' { out.push(e); }
        else if e == 'x' && i + 2 < s.len && hex(s[i + 1]) >= 0 && hex(s[i + 2]) >= 0 {
            out.push((hex(s[i + 1]) * 16 + hex(s[i + 2])) as! u8);
            i += 2;
        } else {
            out.push('\\');
            out.push(e);
        }
        i++;
    }
    s.clear();
    s.append(out);
}

// process_escapes on each string of a list, each ended by a NUL.
fn process_list_escapes(list: u8[>..]&) {
    var out: u8[>..] = [];
    var start = 0;
    for c, k in list {
        if c != 0 { continue; }
        var one: u8[>..] = [];
        one.append(list[start..k]);
        process_escapes(one);
        out.append(one);
        out.push(0);
        start = k + 1;
    }
    list.clear();
    list.append(out);
}

fn hex(c: u8) -> i64 {
    if c >= '0' && c <= '9' { return c - '0'; }
    if c >= 'a' && c <= 'f' { return c - 'a' + 10; }
    if c >= 'A' && c <= 'F' { return c - 'A' + 10; }
    return -1;
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

let CNV_AUTO = -1;
let CNV_OFF = 0;
let CNV_ON = 1;

// The text of the chat's messages; the messages are views of it.
var chat_text: u8[>..] = [];
var chat_msgs: common_chat::Msg[>..] = [];

fn main() {
    let args = console::utf8_args();
    var model_path: u8[>..] = [];
    var prompt: u8[>..] = [];
    var system_prompt: u8[>..] = [];
    var chat_template: u8[>..] = [];
    var input_prefix: u8[>..] = [];
    var input_suffix: u8[>..] = [];
    var antiprompt_text: u8[>..] = [];     // the reverse prompts, each ended by a NUL
    var n_predict = -1;
    var n_ctx = 4096;
    var n_batch = 2048;
    var n_keep = 0;
    var n_threads: i32 = 0;
    var conversation = CNV_AUTO;
    var single_turn = false;
    var interactive = false;
    var interactive_first = false;
    var use_jinja = false;      // llama-completion's default: the built-in templates
    var enable_chat_template = true;
    var input_prefix_bos = false;
    var special = false;
    var display_prompt = true;
    var escape = true;
    var multiline_input = false;
    var verbose_prompt = false;
    var use_color = console::tty_can_use_colors();
    var sp = sampler::Params { .. };
    var breakers_given = false;
    var dry_breakers: u8[>..] = [];     // the --dry-sequence-breaker ones, each ended by a NUL
    var grammar_text: u8[>..] = [];

    var i = 1;
    fn value() -> const u8[:] {
        if i + 1 >= args.len { fail(str("missing value for ", args[i])); }
        i += 1;
        return args[i];
    }
    fn int_value() -> i64 {
        let v, ok = parse_int(value());
        if !ok { fail(str("bad number for ", args[i - 1])); }
        return v;
    }
    fn flt_value() -> f32 {
        let v, ok = parse_flt(value());
        if !ok { fail(str("bad number for ", args[i - 1])); }
        return v as f32;
    }
    while i < args.len {
        let a = args[i];
        if a == "-m" || a == "--model" { model_path.clear(); model_path.append(value()); }
        else if a == "-p" || a == "--prompt" { prompt.clear(); prompt.append(value()); }
        else if a == "-sys" || a == "--system-prompt" { system_prompt.clear(); system_prompt.append(value()); }
        else if a == "-n" || a == "--n-predict" { n_predict = int_value(); }
        else if a == "-c" || a == "--ctx-size" { n_ctx = int_value(); }
        else if a == "-b" || a == "--batch-size" { n_batch = int_value(); }
        else if a == "-t" || a == "--threads" { n_threads = int_value() as! i32; }
        else if a == "-s" || a == "--seed" { sp.seed = (int_value() & 0xFFFFFFFF) as! u32; }
        else if a == "--keep" { n_keep = int_value(); }
        else if a == "-cnv" || a == "--conversation" { conversation = CNV_ON; }
        else if a == "-no-cnv" || a == "--no-conversation" { conversation = CNV_OFF; }
        else if a == "-st" || a == "--single-turn" { single_turn = true; }
        else if a == "-i" || a == "--interactive" { interactive = true; }
        else if a == "-if" || a == "--interactive-first" { interactive_first = true; }
        else if a == "-r" || a == "--reverse-prompt" { antiprompt_text.append(value()); antiprompt_text.push(0); interactive = true; }
        else if a == "--chat-template" { chat_template.clear(); chat_template.append(value()); }
        else if a == "--jinja" { use_jinja = true; }
        else if a == "--no-jinja" { use_jinja = false; }
        else if a == "-co" || a == "--color" {
            let v = value();
            if v == "on" || v == "true" || v == "1" { use_color = true; }
            else if v == "off" || v == "false" || v == "0" { use_color = false; }
            else if v == "auto" { use_color = console::tty_can_use_colors(); }
            else { fail(str("unknown value for --color: '", v, "'")); }
        }
        else if a == "-sp" || a == "--special" { special = true; }
        else if a == "--no-display-prompt" { display_prompt = false; }
        else if a == "-e" || a == "--escape" { escape = true; }
        else if a == "--no-escape" { escape = false; }
        else if a == "-mli" || a == "--multiline-input" { multiline_input = true; }
        else if a == "--verbose-prompt" { verbose_prompt = true; }
        else if a == "--in-prefix" { input_prefix.clear(); input_prefix.append(value()); enable_chat_template = false; }
        else if a == "--in-suffix" { input_suffix.clear(); input_suffix.append(value()); enable_chat_template = false; }
        else if a == "--in-prefix-bos" { input_prefix_bos = true; enable_chat_template = false; }
        else if a == "--temp" { sp.temp = max(flt_value(), 0.0); sp.user_set |= sampler::USER_TEMP; }
        else if a == "--top-k" { sp.top_k = int_value(); sp.user_set |= sampler::USER_TOP_K; }
        else if a == "--top-p" { sp.top_p = flt_value(); sp.user_set |= sampler::USER_TOP_P; }
        else if a == "--min-p" { sp.min_p = flt_value(); sp.user_set |= sampler::USER_MIN_P; }
        else if a == "--typical" { sp.typ_p = flt_value(); }
        else if a == "--repeat-penalty" { sp.penalty_repeat = flt_value(); sp.user_set |= sampler::USER_PENALTY_REPEAT; }
        else if a == "--repeat-last-n" { sp.penalty_last_n = int_value(); sp.user_set |= sampler::USER_PENALTY_LAST_N; }
        else if a == "--presence-penalty" { sp.penalty_present = flt_value(); }
        else if a == "--frequency-penalty" { sp.penalty_freq = flt_value(); }
        else if a == "--xtc-probability" { sp.xtc_probability = flt_value(); sp.user_set |= sampler::USER_XTC_PROBABILITY; }
        else if a == "--xtc-threshold" { sp.xtc_threshold = flt_value(); sp.user_set |= sampler::USER_XTC_THRESHOLD; }
        else if a == "--top-nsigma" { sp.top_n_sigma = flt_value(); }
        else if a == "--dynatemp-range" { sp.dynatemp_range = flt_value(); }
        else if a == "--dynatemp-exp" { sp.dynatemp_exponent = flt_value(); }
        else if a == "--grammar" { grammar_text.clear(); grammar_text.append(value()); }
        else if a == "--grammar-file" {
            let path = value();
            grammar_text.clear();
            if !read_file(path, grammar_text) { fail(str("failed to open file '", path, "'")); }
        }
        else if a == "--mirostat" { sp.mirostat = int_value(); sp.user_set |= sampler::USER_MIROSTAT; }
        else if a == "--mirostat-lr" { sp.mirostat_eta = flt_value(); sp.user_set |= sampler::USER_MIROSTAT_ETA; }
        else if a == "--mirostat-ent" { sp.mirostat_tau = flt_value(); sp.user_set |= sampler::USER_MIROSTAT_TAU; }
        else if a == "--dry-multiplier" { sp.dry_multiplier = flt_value(); }
        else if a == "--dry-base" { let v = flt_value(); if v >= 1.0 { sp.dry_base = v; } }
        else if a == "--dry-allowed-length" { sp.dry_allowed_length = int_value(); }
        else if a == "--dry-penalty-last-n" {
            let v = int_value();
            if v < 0 { fail(str("invalid dry-penalty-last-n = ", v)); }
            sp.dry_penalty_last_n = v;
        }
        else if a == "--dry-sequence-breaker" {
            // the first one replaces the defaults; "none" leaves none
            if !breakers_given { sp.dry_breakers.clear(); breakers_given = true; }
            let v = value();
            if v == "none" { dry_breakers.clear(); } else { dry_breakers.append(v); dry_breakers.push(0); }
        }
        else { fail(str("unknown argument: ", a)); }
        i += 1;
    }
    if model_path.len == 0 { fail("--model is required"); }
    if escape {
        process_escapes(prompt);
        process_escapes(input_prefix);
        process_escapes(input_suffix);
        process_list_escapes(antiprompt_text);
        process_list_escapes(dry_breakers);
    }
    if breakers_given {
        if dry_breakers.len > sp.dry_breakers.cap { fail("the DRY sequence breakers are too long"); }
        sp.dry_breakers.append(dry_breakers);
    }
    console::init(use_color);

    var m = model::Model { .. };
    model::load(m, model_path);
    var c = context::Context { n_threads: n_threads, .. };
    context::init(c, m, n_ctx);
    n_ctx = c.n_ctx;

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
    if conversation == CNV_AUTO { conversation = if has_chat_template { CNV_ON } else { CNV_OFF }; }
    if conversation == CNV_ON && !has_chat_template {
        write_stderr("warning: chat template is not available or is not supported. This may cause the model to output suboptimal responses\n");
    }
    let chat_mode = conversation == CNV_ON && enable_chat_template;

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

    if conversation == CNV_ON {
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

    console::set_display(console::PROMPT);
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
                console::set_display(console::ERROR);
                write_stderr(str("<<input too long: skipped ", skipped, " token", if skipped != 1 { "s" } else { "" }, ">>"));
                console::set_display(console::RESET);
            }
            if c.n_past + embd.len >= n_ctx {
                write_stderr("\n\nmain: context full and context shift is disabled => stopping\n");
                break;
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
            if conversation == CNV_ON && !waiting_for_first_input && !vocab::is_eog(id) {
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
            console::set_display(console::RESET);
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

            if conversation == CNV_ON && !waiting_for_first_input {
                if prompt_text.len > 0 {
                    prompt_text.clear();
                    is_interacting = false;
                }
            }

            if (c.n_past > 0 || waiting_for_first_input) && is_interacting {
                if conversation == CNV_ON { write_stdout("\n> "); }
                if input_prefix_bos { embd_inp.push(vocab::special_bos_id); }
                buffer.clear();
                if input_prefix.len > 0 && conversation != CNV_ON { write_stdout(input_prefix); }
                console::set_display(console::USER_INPUT);
                display = display_prompt;
                loop {
                    let more = console::readline(line, multiline_input);
                    buffer.append(line);
                    if !more { break; }
                }
                console::set_display(console::RESET);
                display = true;

                if buffer.len == 0 {
                    write_stdout("EOF by user\n");
                    break;
                }
                if buffer[buffer.len - 1] == '\n' { buffer.pop(); }

                if buffer.len > 0 {
                    if input_suffix.len > 0 && conversation != CNV_ON { write_stdout(input_suffix); }
                    if escape { process_escapes(buffer); }
                    let format_chat = conversation == CNV_ON && enable_chat_template;
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
