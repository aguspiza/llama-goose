// llama.cpp's flags of how the weights are loaded and where they go, as
// they are with the CPU alone:
//   -lm, --load-mode auto|none|mmap|mlock|mmap+mlock|dio
//   -lzm, --lazy-mode on|auto|off        gemma4's per-layer embeddings read as they are used
//   --repack, -nr / --no-repack          the CPU's repack buffers, or not
//   -ot, --override-tensor <pattern>=<buffer type>,...
//   -cmoe, --cpu-moe; -ncmoe, --n-cpu-moe N   the experts (of the first N layers) on the CPU
// with their LLAMA_ARG_* environment variables. The only buffer type is
// CPU, which the overrides already are: they check the names and warn with
// mmap, as llama.cpp does, and place nothing elsewhere. model::load reads
// what they set.
// port-of: common/arg.cpp (--load-mode, --lazy-mode, --repack, --override-tensor, --cpu-moe, --n-cpu-moe, parse_tensor_buffer_overrides), common/common.h (LLM_FFN_EXPS_REGEX, llm_ffn_block_regex)
import std;
import os;
import llama from "../llama";
import llama.model;

namespace load;

let FFN_EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";

var mode_given = false;
var lazy_given = false;
var repack_given = false;
var overrides_given = false;

// Whether a is one of the flags that take a value.
fn takes_value(a: const u8[:]) -> bool {
    a == "-lm" || a == "--load-mode" || a == "-lzm" || a == "--lazy-mode" || a == "-ot" || a == "--override-tensor" || a == "-ncmoe" || a == "--n-cpu-moe"
}

// Whether a is one of the flags without a value; it is then set.
fn flag(a: const u8[:]) -> bool {
    if a == "--repack" || a == "-nr" || a == "--no-repack" {
        model::repack = a == "--repack";
        repack_given = true;
        return true;
    }
    if a == "-cmoe" || a == "--cpu-moe" {
        add_override(FFN_EXPS_REGEX);
        overrides_given = true;
        return true;
    }
    return false;
}

// Keeps the value of flag a (one that takes_value).
fn set(a: const u8[:], v: const u8[:]) {
    if a == "-lm" || a == "--load-mode" {
        set_mode(a, v);
        mode_given = true;
    } else if a == "-lzm" || a == "--lazy-mode" {
        set_lazy(a, v);
        lazy_given = true;
    } else if a == "-ot" || a == "--override-tensor" {
        add_overrides(a, v);
        overrides_given = true;
    } else {
        add_n_cpu_moe(a, v);
        overrides_given = true;
    }
}

fn fail(a: const u8[:], msg: const u8[:]) {
    write_stderr(str("error while handling argument \"", a, "\": ", msg, "\n"));
    exit(1);
}

fn fail_env(name: const u8[:], msg: const u8[:]) {
    write_stderr(str("error while handling environment variable \"", name, "\": ", msg, "\n"));
    exit(1);
}

fn set_mode(a: const u8[:], v: const u8[:]) {
    model::load_mode =
        if v == "auto" { model::LoadMode.Auto }
        else if v == "none" { model::LoadMode.None }
        else if v == "mmap" { model::LoadMode.Mmap }
        else if v == "mlock" { model::LoadMode.Mlock }
        else if v == "mmap+mlock" { model::LoadMode.MmapMlock }
        else if v == "dio" { model::LoadMode.Dio }
        else { fail(a, "invalid value"); model::LoadMode.Auto };
}

fn set_lazy(a: const u8[:], v: const u8[:]) {
    model::lazy_mode =
        if v == "on" { model::LazyMode.On }
        else if v == "auto" { model::LazyMode.Auto }
        else if v == "off" { model::LazyMode.Off }
        else { fail(a, "invalid value"); model::LazyMode.Auto };
}

fn add_override(pattern: const u8[:]) {
    if model::cpu_overrides.len > 0 { model::cpu_overrides.push(','); }
    model::cpu_overrides.append(pattern);
}

// parse_tensor_buffer_overrides: <pattern>=<buffer type>, comma-separated
fn add_overrides(a: const u8[:], v: const u8[:]) {
    each_split(v, ',') {
        let pattern, buft, found = split_once(it, '=');
        if !found { fail(a, "invalid value"); }
        if buft != "CPU" {
            write_stdout("Available buffer types:\n  CPU\n");
            fail(a, "unknown buffer type");
        }
        add_override(pattern);
    };
}

// llm_add_n_cpu_ffn_overrides: the experts of layers 0 to n - 1
fn add_n_cpu_moe(a: const u8[:], v: const u8[:]) {
    let n, ok = parse_int(v);
    if !ok || n < 0 { fail(a, "invalid value"); }
    for i in n { add_override(str("blk\\.", i, FFN_EXPS_REGEX)); }
}

// The flags not given, from their environment variables, as llama.cpp's
// argument parser takes them.
fn from_env() {
    var v: u8[>..] = [];
    if !mode_given && env("LLAMA_ARG_LOAD_MODE", v) { set_mode("-lm", v); }
    v.clear();
    if !lazy_given && env("LLAMA_ARG_LAZY_MODE", v) { set_lazy("-lzm", v); }
    v.clear();
    if !repack_given && env("LLAMA_ARG_REPACK", v) {
        if v == "on" || v == "enabled" || v == "true" || v == "1" { model::repack = true; }
        else if v == "off" || v == "disabled" || v == "false" || v == "0" { model::repack = false; }
        else { fail_env("LLAMA_ARG_REPACK", "invalid boolean value"); }
    }
    guard !overrides_given else { return; }
    v.clear();
    if env("LLAMA_ARG_OVERRIDE_TENSOR", v) { add_overrides("-ot", v); }
    v.clear();
    if env("LLAMA_ARG_CPU_MOE", v) && (v == "on" || v == "enabled" || v == "true" || v == "1") { add_override(FFN_EXPS_REGEX); }
    v.clear();
    if env("LLAMA_ARG_N_CPU_MOE", v) { add_n_cpu_moe("-ncmoe", v); }
}
