// llama.cpp's command-line parser: the options of common_params_parser_init
// (arg_opts.goose) fill a common_params (Params): first from their
// environment variables, then from the command line; -h prints them by
// section and --completion-bash a bash completion script. An invalid
// argument or value prints the message llama.cpp prints and parse returns
// false (the programs then exit with 1).
// Not ported: the system config files (config.ini, read through
// common/preset), the remote presets of -hf repos, and the options of what
// the port lacks: such an option is listed, and fails when given, with
// "not ported" as its invalid value.
// port-of: common/arg.cpp (common_arg, common_params_parse, common_params_parse_ex, common_params_print_usage, common_params_print_completion, break_str_into_lines, common_arg_utils, parse_bool_value, parse_csv_row, read_file, list_builtin_chat_templates), common/arg.h, common/common.cpp (string_process_escapes), common/common.h (LLM_FFN_EXPS_REGEX, llm_ffn_block_regex)
import std;
import os;
import exception;
import llama from "../llama";
import llama.model;
import llama.sampler;
import llama.chat;
import .console;
import .download;
import .fit;
import .json_schema;
import .jinja;
import .chat;
import .number;
import .arg_opts;

namespace arg;

// llama_example: the program whose options are parsed.
enum Example {
    Batched, Debug, Common, Speculative, Completion, Cli, Embedding, Perplexity, Retrieval, Passkey,
    Imatrix, Bench, Server, CvectorGenerator, ExportLora, Mtmd, Lookup, Parallel, Tts, Diffusion,
    Finetune, FitParams, Results, ExportGraphOps, Download, Tokenize,
}

// How an option takes its value: none, a positive or negated name, an
// integer (std::stoi), a string, or two strings.
enum Kind { Void, Bool, Int, Str, StrStr }

enum ConvMode { Auto, Disabled, Enabled }            // common_conversation_mode
enum FaType { Auto, Disabled, Enabled }              // llama_flash_attn_type
enum ReasonFmt { None, Auto, Deepseek, DeepseekLegacy }  // common_reasoning_format

// common_params: the fields the port's programs read. The lists hold their
// strings each ended by a NUL.
root struct Params {
    n_predict: i64 = -1,
    n_ctx: i64 = 0,
    n_batch: i64 = 2048,
    n_ubatch: i64 = 512,
    n_keep: i64 = 0,
    n_chunks: i64 = -1,
    n_parallel: i64 = 1,
    n_sequences: i64 = 1,
    grp_attn_n: i64 = 1,
    grp_attn_w: i64 = 512,
    n_threads: i64 = -1,             // cpuparams.n_threads
    n_gpu_layers: i64 = -1,
    flash_attn: FaType = FaType.Auto,
    sampling: sampler::Params = sampler::Params { .. },
    grammar: u8[>..] = [],           // sampling.grammar's text
    model_path: u8[>..] = [],        // model.path
    mmproj_path: u8[>..] = [],       // mmproj.path
    model_alias: u8[>..] = [],       // a set: sorted, no repeats
    prompt: u8[>..] = [],
    system_prompt: u8[>..] = [],
    prompt_file: u8[>..] = [],
    input_prefix: u8[>..] = [],
    input_suffix: u8[>..] = [],
    antiprompt: u8[>..] = [],
    image: u8[>..] = [],             // --image, --audio, --video
    out_file: u8[>..] = [],
    server_base: u8[>..] = [],
    usage: bool = false,
    completion: bool = false,
    use_color: bool = false,
    special: bool = false,
    interactive: bool = false,
    interactive_first: bool = false,
    escape: bool = true,
    multiline_input: bool = false,
    cont_batching: bool = true,
    show_timings: bool = true,
    ctx_shift: bool = false,
    kv_unified: bool = false,
    input_prefix_bos: bool = false,
    verbose_prompt: bool = false,
    display_prompt: bool = true,
    single_turn: bool = false,
    conversation_mode: ConvMode = ConvMode.Auto,
    no_mmproj: bool = false,
    mmproj_use_gpu: bool = true,
    n_junk: i64 = 250,
    is_pp_shared: bool = false,
    n_ctx_checkpoints: i64 = 32,
    port: i64 = 9931,
    hostname: u8[>..] = "127.0.0.1",   // hostnames, comma-separated
    public_path: u8[>..] = [],
    api_keys: u8[>..] = [],
    chat_template: u8[>..] = [],
    use_jinja: bool = true,
    enable_chat_template: bool = true,
    reasoning_format: ReasonFmt = ReasonFmt.Deepseek,
    enable_reasoning: i64 = -1,
    prefill_assistant: bool = true,
    webui: bool = true,              // ui
    endpoint_slots: bool = true,
    endpoint_metrics: bool = false,
    media_path: u8[>..] = [],
}

var params = Params { .. };

// One option as common_arg holds it: its names, value hints, environment
// variable and help are spans of text.
struct Span { at: i64, len: i64 }
struct Arg {
    id: Opt,
    kind: Kind,
    args: Span,         // the names, separated by spaces
    neg: Span,          // the negated names
    hint: Span,
    hint2: Span,
    env: Span,
    help: Span,         // with its "(env: ...)" line
    examples: u64,      // EX_* bits
    flags: u64,         // SAMPLING, SPEC
}

let SAMPLING: u64 = 1;
let SPEC: u64 = 2;

var text: u8[>..] = [];
var opts: Arg[>..] = [];
var ex_cur = Example.Common;
var adding = false;     // whether the last option added takes the help being written

fn bit(ex: Example) -> u64 {
    return match ex {
        Batched => EX_BATCHED, Debug => EX_DEBUG, Common => EX_COMMON, Speculative => EX_SPECULATIVE,
        Completion => EX_COMPLETION, Cli => EX_CLI, Embedding => EX_EMBEDDING, Perplexity => EX_PERPLEXITY,
        Retrieval => EX_RETRIEVAL, Passkey => EX_PASSKEY, Imatrix => EX_IMATRIX, Bench => EX_BENCH,
        Server => EX_SERVER, CvectorGenerator => EX_CVECTOR_GENERATOR, ExportLora => EX_EXPORT_LORA,
        Mtmd => EX_MTMD, Lookup => EX_LOOKUP, Parallel => EX_PARALLEL, Tts => EX_TTS, Diffusion => EX_DIFFUSION,
        Finetune => EX_FINETUNE, FitParams => EX_FIT_PARAMS, Results => EX_RESULTS,
        ExportGraphOps => EX_EXPORT_GRAPH_OPS, Download => EX_DOWNLOAD, Tokenize => EX_TOKENIZE,
    };
}

fn span(s: const u8[:]) -> Span {
    let at = text.len;
    text.append(s);
    return Span { at: at, len: s.len };
}

fn get(s: Span) -> const u8[:] { text[s.at..s.at + s.len] }

// add_opt: the option is kept when it is in the program's examples or the
// common ones, and not excluded from the program; its help follows (h, hi,
// hf), until the next add or close.
fn add(id: Opt, kind: Kind, args: const u8[:], neg: const u8[:], hint: const u8[:], hint2: const u8[:],
       env: const u8[:], examples: u64, excludes: u64, flags: u64) -> bool {
    close();
    guard examples & (bit(ex_cur) | EX_COMMON) != 0 && excludes & bit(ex_cur) == 0 else { return false; }
    let a = span(args);
    let n = span(neg);
    let h1 = span(hint);
    let h2 = span(hint2);
    let e = span(env);
    opts.push(Arg { id: id, kind: kind, args: a, neg: n, hint: h1, hint2: h2, env: e,
                    help: Span { at: text.len, len: 0 }, examples: examples, flags: flags });
    adding = true;
    return true;
}

// set_env's line, and the end of the last option's help.
fn close() {
    guard adding else { return; }
    adding = false;
    let o = opts.len - 1;
    if opts[o].env.len > 0 {
        text.append("\n(env: ");
        let e = opts[o].env;
        var name: u8[>..] = [];
        name.append(text[e.at..e.at + e.len]);
        text.append(name);
        text.append(")");
    }
    opts[o].help.len = text.len - opts[o].help.at;
}

fn h(s: const u8[:]) { text.append(s); }
fn hi(v: i64) { text.append(str(v)); }
fn hf(v: f64, decimals: i64) { format_flt(text, v, decimals); }

// The default chain's names and chars, for the help.
fn sampler_names() -> u8[>..] {
    var out: u8[>..] = [];
    for k, i in params.sampling.order {
        if i > 0 { out.push(';'); }
        out.append(sampler::type_to_str(k));
    }
    return out;
}

fn sampler_chars() -> u8[>..] {
    var out: u8[>..] = [];
    for k in params.sampling.order { out.push(sampler::type_to_chr(k)); }
    return out;
}

// --samplers: common_sampler_types_from_names over the names separated by ;
fn set_samplers(v: const u8[:]) {
    params.sampling.order.clear();
    each_split(v, ';') {
        let k, ok = sampler::kind_from_name(it);
        if !ok {
            var lower: u8[>..] = [];
            for c in it { lower.push(if c >= 'A' && c <= 'Z' { c + 32 } else { c }); }
            write_stderr(str("common_sampler_types_from_names: unable to match sampler by name '", lower, "'\n"));
        } else if params.sampling.order.len < params.sampling.order.cap {
            params.sampling.order.push(k);
        }
    };
    params.sampling.user_set |= sampler::USER_SAMPLERS;
}

// --sampling-seq: common_sampler_types_from_chars
fn set_sampler_chars(v: const u8[:]) {
    params.sampling.order.clear();
    for c in v {
        let k, ok = sampler::kind_from_char(c);
        if !ok {
            write_stderr(str("common_sampler_types_from_chars: unable to match sampler by char '", [c], "'\n"));
        } else if params.sampling.order.len < params.sampling.order.cap {
            params.sampling.order.push(k);
        }
    }
}

// The default DRY sequence breakers, quoted, with a newline as \n.
fn dry_breakers() -> u8[>..] {
    var out: u8[>..] = [];
    var start = 0;
    for c, k in params.sampling.dry_breakers {
        if c != 0 { continue; }
        if out.len > 0 { out.append(", "); }
        out.push('\'');
        let b = params.sampling.dry_breakers[start..k];
        if b == "\n" { out.append("\\n"); } else { out.append(b); }
        out.push('\'');
        start = k + 1;
    }
    if out.len == 0 { out.append("none"); }
    return out;
}

// list_builtin_chat_templates
fn builtin_templates() -> u8[>..] {
    var out: u8[>..] = [];
    for t, i in chat::NAMES {
        if i > 0 { out.append(", "); }
        out.append(t.name);
    }
    return out;
}

let KV_CACHE_TYPES = "f32, f16, bf16, f8_e4m3, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1";

// common_arg::to_string: the names and hints, padded to column 40, then the
// help in lines of at most 70 characters, each further one indented.
fn to_string(o: Arg, out: u8[>..]&) {
    let start = out.len;
    var names: u8[>..] = [];
    names.append(get(o.args));
    if o.neg.len > 0 { names.push(' '); names.append(get(o.neg)); }
    var count = 0;
    each_split(names, ' ') { count += 1; };
    var i = 0;
    each_split(names, ' ') {
        if i == 0 {
            if count == 1 {
                out.append(it);
            } else {
                out.append(it);
                out.append(", ");
                for _ in max(0, 7 - (it.len + 2)) { out.push(' '); }
            }
        } else {
            out.append(it);
            if i < count - 1 { out.append(", "); }
        }
        i += 1;
    };
    if o.hint.len > 0 { out.push(' '); out.append(get(o.hint)); }
    if o.hint2.len > 0 { out.push(' '); out.append(get(o.hint2)); }
    if out.len - start > 40 - 3 {
        out.push('\n');
        for _ in 40 { out.push(' '); }
    } else {
        for _ in 40 - (out.len - start) { out.push(' '); }
    }
    var first = true;
    break_lines(get(o.help), 70) {
        if !first { for _ in 40 { out.push(' '); } }
        first = false;
        out.append(it);
        out.push('\n');
    };
}

// break_str_into_lines: each line of the text (getline's: no empty last
// one), longer ones wrapped at their words.
fn break_lines<F>(s: const u8[:], width: i64) {
    var start = 0;
    while start < s.len {
        var end = start;
        while end < s.len && s[end] != '\n' { end++; }
        let line = s[start..end];
        if line.len <= width {
            F(line);
        } else {
            var cur: u8[>..] = [];
            var k = 0;
            while k < line.len {
                while k < line.len && is_space(line[k]) { k++; }
                if k >= line.len { break; }
                let w0 = k;
                while k < line.len && !is_space(line[k]) { k++; }
                let word = line[w0..k];
                let sep = if cur.len > 0 { 1 } else { 0 };
                if cur.len + sep + word.len > width {
                    if cur.len > 0 { F(cur); }
                    cur.clear();
                    cur.append(word);
                } else {
                    if cur.len > 0 { cur.push(' '); }
                    cur.append(word);
                }
            }
            if cur.len > 0 { F(cur); }
        }
        start = end + 1;
    }
}

fn is_space(c: u8) -> bool { c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 11 || c == 12 }

fn is_truthy(v: const u8[:]) -> bool { v == "on" || v == "enabled" || v == "true" || v == "1" }
fn is_falsey(v: const u8[:]) -> bool { v == "off" || v == "disabled" || v == "false" || v == "0" }
fn is_autoy(v: const u8[:]) -> bool { v == "auto" || v == "-1" }

// The error of a handler, or of the parse as a whole; runtime is a
// std::runtime_error rather than an std::invalid_argument (the program then
// exits rather than returning false).
struct Fail { runtime: bool }
var err: u8[>..] = [];

fn invalid(msg: const u8[:]) {
    err.clear();
    err.append(msg);
    exception::throw(Fail { runtime: false });
}

fn runtime(msg: const u8[:]) {
    err.clear();
    err.append(msg);
    exception::throw(Fail { runtime: true });
}

fn parse_bool_value(v: const u8[:]) -> bool {
    if is_truthy(v) { return true; }
    if !is_falsey(v) { invalid("invalid boolean value"); }
    return false;
}

// std::stoi, std::stof and std::stoul: the number at the start of the text.
fn stoi(v: const u8[:]) -> i64 {
    var ok = 0;
    let n = number::c_stoi(v, 10, ok);
    if ok == 0 { invalid("stoi"); }
    return n;
}

extern fn c_stof(s: const u8[:], used: i64&) -> f32 = "arg_stof";
extern fn c_stoul(s: const u8[:], used: i64&) -> u64 = "arg_stoul";

fn stof(v: const u8[:]) -> f32 {
    var used = 0;
    let f = c_stof(v, used);
    if used == 0 { invalid("stof"); }
    if used < 0 { invalid("stof"); }
    return f;
}

fn stoul(v: const u8[:]) -> u64 {
    var used = 0;
    let n = c_stoul(v, used);
    if used <= 0 { invalid("stoul"); }
    return n;
}

fn is_finite(f: f32) -> bool { f == f && f - f == 0.0 }

// read_file, as a handler reads it.
fn read_into(path: const u8[:], out: u8[>..]&) {
    out.clear();
    if !read_file(path, out) { runtime(str("error: failed to open file '", path, "'\n")); }
}

// parse_csv_row: comma-separated fields, a quoted one keeping its commas,
// "" in it a quote; each appended to out with a NUL.
fn add_csv_row(v: const u8[:], out: u8[>..]&, skip_empty: bool) {
    var field: u8[>..] = [];
    var in_quotes = false;
    var i = 0;
    while i < v.len {
        let c = v[i];
        if c == '"' {
            if in_quotes && i + 1 < v.len && v[i + 1] == '"' { field.push('"'); i++; }
            else { in_quotes = !in_quotes; }
        } else if c == ',' && !in_quotes {
            if !skip_empty || field.len > 0 { out.append(field); out.push(0); }
            field.clear();
        } else {
            field.push(c);
        }
        i++;
    }
    if !skip_empty || field.len > 0 { out.append(field); out.push(0); }
}

fn strip(s: const u8[:]) -> const u8[:] {
    var a = 0;
    var b = s.len;
    while a < b && is_space(s[a]) { a++; }
    while b > a && is_space(s[b - 1]) { b--; }
    return s[a..b];
}

// model_alias as a std::set: sorted, each once.
fn add_alias(a: const u8[:]) {
    var names: u8[>..] = [];
    var placed = false;
    var start = 0;
    for c, k in params.model_alias {
        if c != 0 { continue; }
        let n = params.model_alias[start..k];
        if n == a { return; }
        if !placed && compare(a, n) < 0 { names.append(a); names.push(0); placed = true; }
        names.append(n);
        names.push(0);
        start = k + 1;
    }
    if !placed { names.append(a); names.push(0); }
    params.model_alias.clear();
    params.model_alias.append(names);
}

// -ot, -cmoe and -ncmoe with the CPU alone: the only buffer type is CPU,
// which the overrides already are; they check the names and warn with mmap,
// as llama.cpp does, and place nothing elsewhere. model::load reads them.
let FFN_EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";    // LLM_FFN_EXPS_REGEX

fn add_override(pattern: const u8[:]) {
    if model::cpu_overrides.len > 0 { model::cpu_overrides.push(','); }
    model::cpu_overrides.append(pattern);
}

var dry_cleared = false;

fn set_dry_breakers(v: const u8[:]) {
    if !dry_cleared { params.sampling.dry_breakers.clear(); dry_cleared = true; }
    if v == "none" { params.sampling.dry_breakers.clear(); return; }
    if params.sampling.dry_breakers.len + v.len + 1 > params.sampling.dry_breakers.cap {
        invalid("the DRY sequence breakers are too long");
    }
    params.sampling.dry_breakers.append(v);
    params.sampling.dry_breakers.push(0);
}

// The handler of option o with its value: v for a string (and v2 for a
// second one), n for an integer, b for a positive or negated name.
fn apply(o: Opt, v: const u8[:], v2: const u8[:], n: i64, b: bool) {
    match o {
        Help => { params.usage = true; },
        Version => {
            write_stderr("version: 1.0.0-dev (build 0, commit goose)\nbuilt with Goose for this target\n");
            exit(0);
        },
        CompletionBash => { params.completion = true; },
        VerbosePrompt => { params.verbose_prompt = true; },
        DisplayPrompt => { params.display_prompt = b; },
        Color => {
            if is_truthy(v) { params.use_color = true; }
            else if is_falsey(v) { params.use_color = false; }
            else if is_autoy(v) { params.use_color = console::tty_can_use_colors(); }
            else { invalid(str("error: unknown value for --color: '", v, "'\n")); }
        },
        Threads => {
            params.n_threads = n;
            if params.n_threads <= 0 { params.n_threads = cpu_count(); }
        },
        CtxSize => {
            params.n_ctx = n;
            fit::ctx_flag(n);
        },
        Predict => { params.n_predict = n; },
        BatchSize => { params.n_batch = n; },
        UbatchSize => { params.n_ubatch = n; },
        Keep => { params.n_keep = n; },
        CtxCheckpoints => { params.n_ctx_checkpoints = n; },
        KvUnified => { params.kv_unified = b; },
        ContextShift => { params.ctx_shift = b; },
        Chunks => { params.n_chunks = n; },
        FlashAttn => {
            if is_truthy(v) { params.flash_attn = FaType.Enabled; }
            else if is_falsey(v) { params.flash_attn = FaType.Disabled; }
            else if is_autoy(v) { params.flash_attn = FaType.Auto; }
            else { runtime(str("error: unknown value for --flash-attn: '", v, "'\n")); }
        },
        Prompt => { params.prompt.clear(); params.prompt.append(v); },
        SystemPrompt => { params.system_prompt.clear(); params.system_prompt.append(v); },
        ShowTimings => { params.show_timings = b; },
        File => {
            read_into(v, params.prompt);
            params.prompt_file.clear();
            params.prompt_file.append(v);
            if params.prompt.len > 0 && params.prompt[params.prompt.len - 1] == '\n' { params.prompt.resize(params.prompt.len - 1, 0); }
        },
        Escape => { params.escape = b; },
        ReversePrompt => { params.antiprompt.append(v); params.antiprompt.push(0); },
        Special => { params.special = true; },
        Conversation => { params.conversation_mode = if b { ConvMode.Enabled } else { ConvMode.Disabled }; },
        SingleTurn => { params.single_turn = true; },
        Interactive => { params.interactive = true; },
        InteractiveFirst => { params.interactive_first = true; },
        MultilineInput => { params.multiline_input = true; },
        InPrefixBos => { params.input_prefix_bos = true; params.enable_chat_template = false; },
        InPrefix => { params.input_prefix.clear(); params.input_prefix.append(v); params.enable_chat_template = false; },
        InSuffix => { params.input_suffix.clear(); params.input_suffix.append(v); params.enable_chat_template = false; },
        Samplers => { set_samplers(v); },
        SamplerSeq => { set_sampler_chars(v); },
        Seed => { params.sampling.seed = (stoul(v) & 0xFFFFFFFF) as! u32; },
        Temp => {
            params.sampling.temp = max(stof(v), 0.0);
            params.sampling.user_set |= sampler::USER_TEMP;
        },
        TopK => { params.sampling.top_k = n; params.sampling.user_set |= sampler::USER_TOP_K; },
        TopP => { params.sampling.top_p = stof(v); params.sampling.user_set |= sampler::USER_TOP_P; },
        MinP => { params.sampling.min_p = stof(v); params.sampling.user_set |= sampler::USER_MIN_P; },
        TopNsigma => { params.sampling.top_n_sigma = stof(v); },
        XtcProbability => { params.sampling.xtc_probability = stof(v); params.sampling.user_set |= sampler::USER_XTC_PROBABILITY; },
        XtcThreshold => { params.sampling.xtc_threshold = stof(v); params.sampling.user_set |= sampler::USER_XTC_THRESHOLD; },
        Typical => { params.sampling.typ_p = stof(v); },
        RepeatLastN => {
            if n < 0 { runtime(str("error: invalid repeat-last-n = ", n, "\n")); }
            params.sampling.penalty_last_n = n;
            params.sampling.user_set |= sampler::USER_PENALTY_LAST_N;
        },
        RepeatPenalty => {
            let f = stof(v);
            if !is_finite(f) || f <= 0.0 || !is_finite(1.0 / f) {
                runtime("error: repeat-penalty must be finite and greater than 0\n");
            }
            params.sampling.penalty_repeat = f;
            params.sampling.user_set |= sampler::USER_PENALTY_REPEAT;
        },
        PresencePenalty => {
            let f = stof(v);
            if !is_finite(f) { runtime("error: presence-penalty must be finite\n"); }
            params.sampling.penalty_present = f;
        },
        FrequencyPenalty => {
            let f = stof(v);
            if !is_finite(f) { runtime("error: frequency-penalty must be finite\n"); }
            params.sampling.penalty_freq = f;
        },
        DryMultiplier => { params.sampling.dry_multiplier = stof(v); },
        DryBase => {
            let f = stof(v);
            if f >= 1.0 { params.sampling.dry_base = f; }
        },
        DryAllowedLength => { params.sampling.dry_allowed_length = n; },
        DryPenaltyLastN => {
            if n < 0 { runtime(str("error: invalid dry-penalty-last-n = ", n, "\n")); }
            params.sampling.dry_penalty_last_n = n;
        },
        DrySequenceBreaker => { set_dry_breakers(v); },
        DynatempRange => { params.sampling.dynatemp_range = stof(v); },
        DynatempExp => { params.sampling.dynatemp_exponent = stof(v); },
        Mirostat => { params.sampling.mirostat = n; params.sampling.user_set |= sampler::USER_MIROSTAT; },
        MirostatLr => { params.sampling.mirostat_eta = stof(v); params.sampling.user_set |= sampler::USER_MIROSTAT_ETA; },
        MirostatEnt => { params.sampling.mirostat_tau = stof(v); params.sampling.user_set |= sampler::USER_MIROSTAT_TAU; },
        Grammar => { params.grammar.clear(); params.grammar.append(v); },
        GrammarFile => { read_into(v, params.grammar); },
        JsonSchema => { schema_grammar(v); },
        JsonSchemaFile => {
            var schema: u8[>..] = [];
            read_into(v, schema);
            schema_grammar(schema);
        },
        GrpAttnN => { params.grp_attn_n = n; },
        GrpAttnW => { params.grp_attn_w = n; },
        Repack => { model::repack = b; },
        Parallel => {
            if n == 0 { invalid("error: invalid value for n_parallel\n"); }
            params.n_parallel = n;
        },
        Parallel2 => { params.n_parallel = n; },
        Sequences => { params.n_sequences = n; },
        ContBatching => { params.cont_batching = b; },
        Mmproj => { params.mmproj_path.clear(); params.mmproj_path.append(v); },
        MmprojAuto => { params.no_mmproj = !b; },
        MmprojOffload => { params.mmproj_use_gpu = b; },
        Image => { add_csv_row(v, params.image, false); },
        LoadMode => {
            model::load_mode =
                if v == "auto" { model::LoadMode.Auto }
                else if v == "none" { model::LoadMode.None }
                else if v == "mmap" { model::LoadMode.Mmap }
                else if v == "mlock" { model::LoadMode.Mlock }
                else if v == "mmap+mlock" { model::LoadMode.MmapMlock }
                else if v == "dio" { model::LoadMode.Dio }
                else { invalid("invalid value"); model::LoadMode.Auto };
        },
        LazyMode => {
            model::lazy_mode =
                if v == "on" { model::LazyMode.On }
                else if v == "auto" { model::LazyMode.Auto }
                else if v == "off" { model::LazyMode.Off }
                else { invalid("invalid value"); model::LazyMode.Auto };
        },
        OverrideTensor => {
            // parse_tensor_buffer_overrides: <pattern>=<buffer type>, comma-separated
            each_split(v, ',') {
                let pattern, buft, found = split_once(it, '=');
                if !found { invalid("invalid value"); }
                if buft != "CPU" {
                    write_stdout("Available buffer types:\n  CPU\n");
                    invalid("unknown buffer type");
                }
                add_override(pattern);
            };
        },
        CpuMoe => { add_override(FFN_EXPS_REGEX); },
        NCpuMoe => {
            // llm_add_n_cpu_ffn_overrides: the experts of layers 0 to n - 1
            if n < 0 { invalid("invalid value"); }
            for i in n { add_override(str("blk\\.", i,FFN_EXPS_REGEX)); }
        },
        GpuLayers => {
            if v == "auto" { params.n_gpu_layers = -1; }
            else if v == "all" { params.n_gpu_layers = -2; }
            else { params.n_gpu_layers = stoi(v); }
            write_stderr("warning: no usable GPU found, --gpu-layers option will be ignored\n");
            write_stderr("warning: one possible reason is that llama.cpp was compiled without GPU support\n");
            write_stderr("warning: consult docs/build.md for compilation instructions\n");
        },
        Fit => {
            if is_truthy(v) { fit::enabled = true; }
            else if is_falsey(v) { fit::enabled = false; }
            else { runtime(str("error: unknown value for --fit: '", v, "'\n")); }
        },
        FitTarget => { fit_target(v); },
        FitCtx => { fit::min_ctx = n; },
        Alias => {
            each_split(v, ',') {
                let a = strip(it);
                if a.len > 0 { add_alias(a); }
            };
        },
        Model => { params.model_path.clear(); params.model_path.append(v); },
        HfRepo => { download::hf_repo.clear(); download::hf_repo.append(v); },
        HfFile => { download::hf_file.clear(); download::hf_file.append(v); },
        HfToken => { download::hf_token.clear(); download::hf_token.append(v); },
        Junk => { params.n_junk = n; },
        Pps => { params.is_pp_shared = true; },
        ServerBase => { params.server_base.clear(); params.server_base.append(v); },
        Output => { params.out_file.clear(); params.out_file.append(v); },
        Host => {
            var hosts: u8[>..] = [];
            add_csv_row(v, hosts, false);
            params.hostname.clear();
            var start = 0;
            for c, k in hosts {
                if c != 0 { continue; }
                let one = strip(hosts[start..k]);
                if one.len > 0 {
                    if params.hostname.len > 0 { params.hostname.push(','); }
                    params.hostname.append(one);
                }
                start = k + 1;
            }
            if params.hostname.len == 0 { invalid("--host requires at least one address"); }
        },
        Port => { params.port = n; },
        Path => { params.public_path.clear(); params.public_path.append(v); },
        Ui => { params.webui = b; },
        ApiKey => { add_csv_row(v, params.api_keys, true); },
        Metrics => { params.endpoint_metrics = true; },
        Slots => { params.endpoint_slots = b; },
        MediaPath => {
            params.media_path.clear();
            params.media_path.append(v);
            if !is_dir(params.media_path) { invalid(str("not a directory: ", v)); }
            if params.media_path.len > 0 && params.media_path[params.media_path.len - 1] != DIR_SEP { params.media_path.push(DIR_SEP); }
        },
        Jinja => { params.use_jinja = b; },
        ReasoningFormat => {
            if v == "none" { params.reasoning_format = ReasonFmt.None; }
            else if v == "auto" { params.reasoning_format = ReasonFmt.Auto; }
            else if v == "deepseek" { params.reasoning_format = ReasonFmt.Deepseek; }
            else if v == "deepseek-legacy" { params.reasoning_format = ReasonFmt.DeepseekLegacy; }
            else { runtime(str("Unknown reasoning format: ", v)); }
        },
        Reasoning => {
            if is_truthy(v) { params.enable_reasoning = 1; }
            else if is_falsey(v) { params.enable_reasoning = 0; }
            else if is_autoy(v) { params.enable_reasoning = -1; }
            else { invalid(str("error: unknown value for --reasoning: '", v, "'\n")); }
        },
        ChatTemplate => { params.chat_template.clear(); params.chat_template.append(v); },
        ChatTemplateFile => { read_into(v, params.chat_template); },
        PrefillAssistant => { params.prefill_assistant = b; },
        Offline => { download::offline = true; },
        _ => { invalid("not ported"); },
    }
}

extern fn c_dir_sep() -> i64 = "arg_dir_sep";
extern fn cpu_count() -> i64 = "arg_hardware_concurrency";
let DIR_SEP: u8 = c_dir_sep() as! u8;

fn schema_grammar(schema: const u8[:]) {
    var g: u8[>..] = [];
    if !json_schema::to_grammar(schema, g) { invalid(json_schema::err); }
    params.grammar.clear();
    params.grammar.append(g);
}

// -fitt: MiB per device, split at , and /; the first is the host's.
fn fit_target(v: const u8[:]) {
    var n_parts = 0;
    var first = -1;
    var start = 0;
    var i = 0;
    while i <= v.len {
        if i == v.len || v[i] == ',' || v[i] == '/' {
            if i > start || i == v.len {
                if first < 0 { first = start; }
                n_parts += 1;
            }
            while i < v.len && (v[i] == ',' || v[i] == '/') { i++; }
            start = i;
            if i == v.len { break; }
            continue;
        }
        i++;
    }
    var end = first;
    while end < v.len && v[end] != ',' && v[end] != '/' { end++; }
    let mib = stoul(v[first..end]);
    fit::target_mib = mib as i64;
}

// The tool's usage after llama.cpp's sections (the print_usage callback).
var usage_text: u8[>..] = [];

// common_params_parse: false after printing why the arguments are invalid.
// Before calling it a program may change params' defaults, which -h shows.
fn parse(args: const u8[:][>..]&, ex: Example) -> bool {
    // per-example defaults
    if ex == Example.Completion { params.use_jinja = false; }
    else if ex == Example.Mtmd { params.use_jinja = false; params.sampling.temp = 0.2; }
    else if ex == Example.Server { params.n_parallel = -1; }
    params.use_color = console::tty_can_use_colors();
    ex_cur = ex;
    text.clear();
    opts.clear();
    add_options(ex);

    var failed = false;
    var exit_code = 0;
    fn body() { parse_ex(args, ex); }
    exception::try<Fail>(body) { f =>
        write_stderr(str(err, "\n"));
        if f.runtime { exit_code = 1; } else { failed = true; }
    };
    if exit_code != 0 { exit(exit_code); }
    if failed { return false; }
    if params.usage {
        var out: u8[>..] = [];
        print_usage(ex, out);
        write_stdout(out);
        write_stdout(usage_text);
        exit(0);
    }
    if params.completion {
        var out: u8[>..] = [];
        print_completion(ex, out);
        write_stdout(out);
        exit(0);
    }
    return true;
}

fn find_opt(name: const u8[:]) -> i64, bool {
    for o, k in opts {
        if has_name(get(o.args), name) { return k, true; }
        if has_name(get(o.neg), name) { return k, false; }
    }
    return -1, false;
}

fn has_name(names: const u8[:], name: const u8[:]) -> bool {
    var found = false;
    each_split(names, ' ') { if it == name { found = true; } };
    return found;
}

// The option's environment variable (or its LLAMA_ARG_NO_ one, for an option
// with negated names, which gives "0") into out: whether it is set.
fn env_value(o: Arg, out: u8[>..]&) -> bool {
    guard o.env.len > 0 else { return false; }
    let name = get(o.env);
    if o.neg.len > 0 && starts_with(name, "LLAMA_ARG_") {
        var neg = str("LLAMA_ARG_NO_", name["LLAMA_ARG_".len..]);
        var tmp: u8[>..] = [];
        if env(neg, tmp) { out.clear(); out.append("0"); return true; }
    }
    out.clear();
    return env(name, out);
}

fn parse_ex(args: const u8[:][>..]&, ex: Example) {
    // environment variables
    for k in opts.len {
        var value: u8[>..] = [];
        guard env_value(opts[k], value) else { continue; }
        let o = opts[k];
        fn from_env() {
            match o.kind {
                Void => { if is_truthy(value) { apply(o.id, "", "", 0, true); } },
                Int => { apply(o.id, "", "", stoi(value), true); },
                Bool => { apply(o.id, "", "", 0, parse_bool_value(value)); },
                Str => { apply(o.id, value, "", 0, true); },
                _ => {},
            }
        }
        exception::try<Fail>(from_env) { f =>
            invalid(str("error while handling environment variable \"", get(o.env), "\": ", err, "\n\n"));
        };
    }

    // the command line
    var seen: u8[>..] = [];
    var i = 1;
    while i < args.len {
        var a = str(args[i]);
        if starts_with(a, "--") { for k in a.len { if a[k] == '_' { a[k] = '-'; } } }
        let k, positive = find_opt(a);
        if k < 0 { invalid(str("error: invalid argument: ", a)); }
        var dup = false;
        each_split(seen, 0) { if it == a { dup = true; } };
        if dup {
            if a != "--spec-type" {
                write_stderr(str("DEPRECATED: argument '", a, "' specified multiple times, use comma-separated values instead (only last value will be used)\n"));
            }
        } else {
            seen.append(a);
            seen.push(0);
        }
        let o = opts[k];
        var tmp: u8[>..] = [];
        if env_value(o, tmp) {
            write_stderr(str("warn: ", get(o.env), " environment variable is set, but will be overwritten by command line argument ", a, "\n"));
        }
        fn handle() {
            match o.kind {
                Void => { apply(o.id, "", "", 0, true); },
                Bool => { apply(o.id, "", "", 0, positive); },
                Int => {
                    if i + 1 >= args.len { invalid("expected value for argument"); }
                    i += 1;
                    apply(o.id, "", "", stoi(args[i]), true);
                },
                Str => {
                    if i + 1 >= args.len { invalid("expected value for argument"); }
                    i += 1;
                    apply(o.id, args[i], "", 0, true);
                },
                StrStr => {
                    if i + 1 >= args.len { invalid("expected value for argument"); }
                    i += 1;
                    if i + 1 >= args.len { invalid("expected value for argument"); }
                    i += 1;
                    apply(o.id, args[i - 1], args[i], 0, true);
                },
            }
        }
        exception::try<Fail>(handle) { f =>
            var usage: u8[>..] = [];
            to_string(o, usage);
            invalid(str("error while handling argument \"", a, "\": ", err, "\n\nusage:\n", usage, "\n\nto show complete usage, run with -h"));
        };
        i += 1;
    }

    // the model and its download
    if ex != Example.Server && ex != Example.Download && ex != Example.ExportGraphOps {
        // common_models_handler: the mmproj too, for the programs that take one
        let use_mmproj = bit(ex) & (EX_MTMD | EX_SERVER | EX_CLI | EX_TTS) != 0;
        download::resolve(params.model_path, params.mmproj_path, use_mmproj && !params.no_mmproj);
        if !params.usage && !params.completion && params.server_base.len == 0 && params.model_path.len == 0 {
            invalid("error: --model or -hf is required\n");
        }
    }

    if params.escape {
        process_escapes(params.prompt);
        process_escapes(params.input_prefix);
        process_escapes(params.input_suffix);
        process_list_escapes(params.antiprompt);
        var breakers: u8[>..] = [];
        breakers.append(params.sampling.dry_breakers);
        process_list_escapes(breakers);
        params.sampling.dry_breakers.clear();
        params.sampling.dry_breakers.append(breakers);
    }

    if params.chat_template.len > 0 && !verify_template(params.chat_template, params.use_jinja) {
        runtime(str("error: the supplied chat template is not supported: ", params.chat_template,
                    if params.use_jinja { "" } else { "\nnote: llama.cpp was started without --jinja, we only support commonly used templates" }, "\n"));
    }
}

// common_chat_verify_template: whether the template renders a user message.
fn verify_template(tmpl: const u8[:], use_jinja: bool) -> bool {
    if use_jinja {
        var ts, ok = common_chat::init(tmpl, "", "", "", false, false);
        let msgs = [common_chat::Msg { role: "user", content: "test" }];
        if ok { ok = common_chat::apply(ts, msgs, common_chat::Inputs { .. }); }
        if !ok { write_stderr(str("common_chat_verify_template: failed to apply template: ", jinja::err, "\n")); }
        return ok;
    }
    var out: u8[>..] = [];
    let msgs = [chat::Message { role: "user", content: "test" }];
    return chat::apply(tmpl, msgs, true, out);
}

// common_params_print_usage: the sections that have options.
fn print_usage(ex: Example, out: u8[>..]&) {
    var first = true;
    for section in 4 {
        var any = false;
        for o in opts {
            if section_of(o, ex) != section { continue; }
            if !any {
                if !first { out.append("\n\n"); }
                out.append("----- ");
                out.append(match section { 0 => "common params", 1 => "sampling params", 2 => "speculative params", _ => "example-specific params" });
                out.append(" -----\n\n");
                first = false;
                any = true;
            }
            to_string(o, out);
        }
    }
}

fn section_of(o: Arg, ex: Example) -> i64 {
    if o.flags & SAMPLING != 0 { return 1; }
    if o.flags & SPEC != 0 { return 2; }
    if o.examples & bit(ex) != 0 { return 3; }
    return 0;
}

// common_params_print_completion
fn print_completion(ex: Example, out: u8[>..]&) {
    out.append("_llama_completions() {\n");
    out.append("    local cur prev opts\n");
    out.append("    COMPREPLY=()\n");
    out.append("    cur=\"${COMP_WORDS[COMP_CWORD]}\"\n");
    out.append("    prev=\"${COMP_WORDS[COMP_CWORD-1]}\"\n\n");
    out.append("    opts=\"");
    for section in [0, 1, 2, 3] {
        for o in opts {
            if section_of(o, ex) != section { continue; }
            each_split(get(o.args), ' ') { out.append(it); out.push(' '); };
        }
    }
    out.append("\"\n\n");
    out.append("    case \"$prev\" in\n");
    out.append("        --model|-m)\n");
    out.append("            COMPREPLY=( $(compgen -f -X '!*.gguf' -- \"$cur\") $(compgen -d -- \"$cur\") )\n");
    out.append("            return 0\n");
    out.append("            ;;\n");
    out.append("        --grammar-file)\n");
    out.append("            COMPREPLY=( $(compgen -f -X '!*.gbnf' -- \"$cur\") $(compgen -d -- \"$cur\") )\n");
    out.append("            return 0\n");
    out.append("            ;;\n");
    out.append("        --chat-template-file)\n");
    out.append("            COMPREPLY=( $(compgen -f -X '!*.jinja' -- \"$cur\") $(compgen -d -- \"$cur\") )\n");
    out.append("            return 0\n");
    out.append("            ;;\n");
    out.append("        *)\n");
    out.append("            COMPREPLY=( $(compgen -W \"${opts}\" -- \"$cur\") )\n");
    out.append("            return 0\n");
    out.append("            ;;\n");
    out.append("    esac\n");
    out.append("}\n\n");
    for exe in EXECUTABLES {
        out.append("complete -F _llama_completions ");
        out.append(exe);
        out.push('\n');
    }
}

let EXECUTABLES = [
    "llama-batched", "llama-batched-bench", "llama-bench", "llama-cli", "llama-completion",
    "llama-convert-llama2c-to-ggml", "llama-cvector-generator", "llama-debug", "llama-diffusion-cli",
    "llama-embedding", "llama-eval-callback", "llama-export-lora", "llama-finetune", "llama-fit-params",
    "llama-gemma3-cli", "llama-gen-docs", "llama-gguf", "llama-gguf-hash", "llama-gguf-split", "llama-idle",
    "llama-imatrix", "llama-llava-cli", "llama-lookahead", "llama-lookup", "llama-lookup-create",
    "llama-lookup-merge", "llama-lookup-stats", "llama-minicpmv-cli", "llama-mtmd-cli", "llama-parallel",
    "llama-passkey", "llama-perplexity", "llama-q8dot", "llama-quantize", "llama-qwen2vl-cli",
    "llama-retrieval", "llama-save-load-state", "llama-server", "llama-simple", "llama-simple-chat",
    "llama-speculative", "llama-speculative-simple", "llama-tokenize", "llama-tts", "llama-vdot",
];

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
