// llama.cpp's --fit on the CPU: a context size left to the model (-c 0, or
// no -c where 0 is the default) is the trained one, for each stream, unless
// the model, the cache and the compute buffers would then leave less than
// the target (-fitt, 1024 MiB) of the host memory's total; then it is cut,
// interpolated between that and -fitc (4096) per stream, down to a multiple
// of 256 per stream. A size the user set is kept, as one that cannot be
// fitted, with a warning. The memory is projected by context::projected_bytes.
// Not ported: the devices (layers and tensors placed on GPUs), -fitp.
// port-of: common/fit.cpp (common_params_fit_impl, its host-memory branch), common/common.cpp (common_fit_apply)
import std;
import os;
import llama from "../llama";
import llama.model;
import ggml from "../ggml";
import llama.context;

namespace fit;

var enabled = true;             // -fit on|off
var target_mib = 1024;          // -fitt: the host memory to leave free
var min_ctx = 4096;             // -fitc: the smallest context per stream; -c 0 asks for the full one

let FULL = 0x7FFFFFFF;          // min_ctx after -c 0: no cut

// Whether a is one of the flags above (all take a value).
fn takes_value(a: const u8[:]) -> bool {
    a == "-fit" || a == "--fit" || a == "-fitt" || a == "--fit-target" || a == "-fitc" || a == "--fit-ctx"
}

fn set(a: const u8[:], v: const u8[:]) {
    if a == "-fit" || a == "--fit" {
        guard v == "on" || v == "off" else { fail(str("error: unknown value for --fit: '", v, "'")); return; }
        enabled = v == "on";
        return;
    }
    // -fitt takes a value per device: the first is the host's
    let first, rest, more = split_once(v, ',');
    let n, ok = parse_int(first);
    guard ok && n >= 0 else { fail(str("error: invalid value for ", a, ": '", v, "'")); return; }
    if a == "-fitt" || a == "--fit-target" { target_mib = n; } else { min_ctx = n; }
}

// -c n: n == 0 asks for the trained context, which --fit then does not cut.
fn ctx_flag(n: i64) {
    if n == 0 { min_ctx = FULL; }
}

fn fail(msg: const u8[:]) {
    write_stderr(str(msg, "\n"));
    exit(1);
}

// The context to make for c (its sequences and whether the cache is unified
// set) with the model, n_ctx being the one asked for, 0 for the model's.
fn n_ctx(c: context::Context&, m: model::Model&, n_ctx: i64) -> i64 {
    let n_streams = if c.unified { 1 } else { max(1, c.n_seq_max) };
    let n_ctx_train = m.hp.n_ctx_train as i64;
    guard enabled else { return if n_ctx > 0 { n_ctx } else { n_ctx_train }; }
    let n_ctx_max = n_ctx_train * n_streams;
    let n_ctx_min = min(min_ctx * n_streams, FULL);
    let auto = n_ctx == 0;
    var cur = if auto { n_ctx_max } else { n_ctx };
    let margin = target_mib * 1024 * 1024;
    let total = ggml::host_memory_total();
    let used = context::projected_bytes(c, m, cur);
    if total - used >= margin { return cur; }
    if auto && n_ctx_max > n_ctx_min {
        let target = total - margin;
        cur = n_ctx_min;
        let used_min = context::projected_bytes(c, m, cur);
        if target > used_min {
            // linear interpolation between the smallest and the largest context
            cur += (n_ctx_max - n_ctx_min) * (target - used_min) / (used - used_min);
            let align = 256 * n_streams;
            return max(cur - cur % align, n_ctx_min);
        }
    }
    write_stderr("common_fit_params: failed to fit params to free device memory: was unable to fit model into system memory by reducing context, abort\n");
    return cur;
}
