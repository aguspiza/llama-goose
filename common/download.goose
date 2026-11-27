// llama.cpp's model downloads from Hugging Face: -hf <user>/<model>[:quant]
// (with -hff, -hft and --offline) picks the repository's GGUF file for the
// quant (Q4_K_M, then Q8_0, then the first, when none is given), with all
// its split parts and, for the programs that take one, its mmproj, and
// downloads them into the Hugging Face cache that huggingface_hub shares
// (blobs by their hash, linked from the commit's snapshot). A file already
// there is not downloaded again; with --offline, or when the Hub cannot be
// reached, the cached snapshot is used. The requests go through libcurl,
// loaded when the first one is made.
// Not ported: the cache's lock files (two programs downloading the same
// file at once), downloads from a plain URL (-mu), Docker registries, the
// repository's preset.ini and the speculative-decoding sidecars.
// port-of: common/download.cpp (common_download_get_hf_plan, find_best_model, find_best_sibling, get_gguf_split_info, common_download_file_single_online), common/hf-cache.cpp, common/arg.cpp (common_models_handler_apply), common/common.cpp (common_get_model_endpoint)
import std;
import os;
import .jdoc;

namespace download;

extern fn c_request(url: const u8[:], header_lines: const u8[:], head: i64, path: const u8[:], resume_from: i64, bar: i64) -> i64 = "common_http_request";
extern fn c_body(out: u8[:]) -> i64 = "common_http_body";
extern fn c_header(name: const u8[:], out: u8[:]) -> i64 = "common_http_header";
extern fn c_error(out: u8[:]) -> i64 = "common_http_error";
extern fn c_file_size(path: const u8[:]) -> i64 = "common_file_size";
extern fn c_symlink(target: const u8[:], link: const u8[:]) -> i64 = "common_symlink";

let PARTIAL_SUFFIX = ".downloadInProgress";

// The flags, as the programs pass them on.
var hf_repo: u8[>..] = [];      // -hf <user>/<model>[:quant]
var hf_file: u8[>..] = [];      // -hff: this file of the repository
var hf_token: u8[>..] = [];     // -hft, or HF_TOKEN
var offline = false;            // --offline

// Whether a is one of the flags above that takes a value.
fn takes_value(a: const u8[:]) -> bool {
    a == "-hf" || a == "-hfr" || a == "--hf-repo" || a == "-hff" || a == "--hf-file" || a == "-hft" || a == "--hf-token"
}

// Keeps the value of flag a (one that takes_value).
fn set(a: const u8[:], v: const u8[:]) {
    if a == "-hff" || a == "--hf-file" { hf_file.clear(); hf_file.append(v); }
    else if a == "-hft" || a == "--hf-token" { hf_token.clear(); hf_token.append(v); }
    else { hf_repo.clear(); hf_repo.append(v); }
}

fn fail(msg: const u8[:]) {
    write_stderr(str("error: ", msg, "\n"));
    exit(1);
}

// The flags not given, from their environment variables, as llama.cpp's
// argument parser takes them.
fn from_env() {
    var v: u8[>..] = [];
    if hf_repo.len == 0 && env("LLAMA_ARG_HF_REPO", v) { hf_repo.append(v); }
    v.clear();
    if hf_file.len == 0 && env("LLAMA_ARG_HF_FILE", v) { hf_file.append(v); }
    v.clear();
    if hf_token.len == 0 && env("HF_TOKEN", v) { hf_token.append(v); }
    v.clear();
    if env("LLAMA_OFFLINE", v) && (v == "1" || v == "true") { offline = true; }
}

// The Hub's address: MODEL_ENDPOINT, HF_ENDPOINT, or huggingface.co; with
// a '/' at the end.
fn endpoint() -> u8[>..] {
    var e: u8[>..] = [];
    if !env("MODEL_ENDPOINT", e) || e.len == 0 {
        e.clear();
        if !env("HF_ENDPOINT", e) || e.len == 0 { e.clear(); e.append("https://huggingface.co/"); }
    }
    if e[e.len - 1] != '/' { e.push('/'); }
    return e;
}

// The Hugging Face cache, where huggingface_hub keeps it.
fn cache_dir() -> u8[>..] {
    var v: u8[>..] = [];
    if env("LLAMA_CACHE", v) && v.len > 0 { return v; }
    v.clear();
    if env("HF_HUB_CACHE", v) && v.len > 0 { return v; }
    v.clear();
    if env("HUGGINGFACE_HUB_CACHE", v) && v.len > 0 { return v; }
    v.clear();
    if env("HF_HOME", v) && v.len > 0 { v.append("/hub"); return v; }
    v.clear();
    if env("XDG_CACHE_HOME", v) && v.len > 0 { v.append("/huggingface/hub"); return v; }
    v.clear();
    if (env("USERPROFILE", v) && v.len > 0) || (env("HOME", v) && v.len > 0) {
        v.append("/.cache/huggingface/hub");
        return v;
    }
    fail("Failed to determine HF cache directory");
    return v;
}

// The directory at path and those above it.
fn make_dirs(path: const u8[:]) {
    for c, i in path {
        if (c == '/' || c == '\\') && i > 0 && path[i - 1] != ':' { make_dir(path[..i]); }
    }
    make_dir(path);
}

fn parent(path: const u8[:]) -> const u8[:] {
    var end = 0;
    for c, i in path { if c == '/' || c == '\\' { end = i; } }
    return path[..end];
}

fn is_hex(s: const u8[:], n: i64) -> bool {
    guard s.len == n else { return false; }
    for c in s {
        guard (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') else { return false; }
    }
    return true;
}

// A path inside the snapshot: relative, without "." or ".." parts.
fn is_valid_subpath(p: const u8[:]) -> bool {
    guard p.len > 0 && p[0] != '/' && p[0] != '\\' && find(p, ":") < 0 && find(p, "\\") < 0 else { return false; }
    var ok = true;
    each_split(p, '/') { if it.len == 0 || it == "." || it == ".." { ok = false; } };
    return ok;
}

// The request headers, with the token when it is one of the Hub's.
fn request_headers(json: bool) -> u8[>..] {
    var h: u8[>..] = [];
    h.append("User-Agent: llama-goose\n");
    if json { h.append("Accept: application/json\n"); }
    if starts_with(hf_token, "hf_") { h.append(str("Authorization: Bearer ", hf_token, "\n")); }
    return h;
}

fn last_error() -> u8[>..] {
    var e: u8[>..] = [];
    e.resize(512, 0);
    let n = c_error(e);
    e.resize(n, 0);
    return e;
}

fn last_body() -> u8[>..] {
    var b: u8[>..] = [];
    let n = c_body(b[..]);
    b.resize(n, 0);
    c_body(b);
    return b;
}

fn last_header(name: const u8[:]) -> u8[>..], bool {
    var v: u8[>..] = [];
    v.resize(1024, 0);
    let n = c_header(name, v);
    guard n >= 0 && n <= 1024 else { v.clear(); return v, false; }
    v.resize(n, 0);
    return v, true;
}

// GET url's JSON into jdoc: the root value, or -1 (the error is logged).
fn api_get(url: const u8[:]) -> i64 {
    let status = c_request(url, request_headers(true), 0, "", 0, 0);
    guard status >= 0 else {
        write_stderr(str("api_get: error: ", last_error(), "\n"));
        return -1;
    }
    let body = last_body();
    guard status == 200 else {
        jdoc::reset();
        let e = jdoc::parse(body);
        let msg = if e >= 0 && jdoc::is_str(jdoc::get(e, "error")) { jdoc::str_of(jdoc::get(e, "error")) } else { body[..] };
        write_stderr(str("api_get: error: GET failed (", status, "): ", msg, "\n"));
        return -1;
    }
    jdoc::reset();
    let root = jdoc::parse(body);
    if root < 0 { write_stderr("api_get: JSON error\n"); }
    return root;
}

// A file of the repository: its path in the repository, where it goes
// (local: the blob, or the snapshot entry when there is no hash or it is
// there already), the snapshot entry, its URL and its hash, all as spans
// of text.
struct Span { at: i64 = 0, n: i64 = 0 }
struct File { path: Span, local: Span, final: Span, url: Span, oid: Span }

var text: u8[>..] = [];
var files: File[>..] = [];

fn keep(s: const u8[:]) -> Span {
    let at = text.len;
    text.append(s);
    return Span { at: at, n: s.len };
}

fn get(s: Span) -> const u8[:] { text[s.at..s.at + s.n] }

fn repo_folder(repo: const u8[:]) -> u8[>..] {
    var f: u8[>..] = [];
    f.append("models--");
    for c in repo { if c == '/' { f.append("--"); } else { f.push(c); } }
    return f;
}

// The branch's commit (main, or the first), kept in the cache's refs/.
fn repo_commit(repo: const u8[:], repo_path: const u8[:]) -> u8[>..] {
    var commit: u8[>..] = [];
    let root = api_get(str(endpoint(), "api/models/", repo, "/refs"));
    guard root >= 0 else { return commit; }
    let branches = jdoc::get(root, "branches");
    guard jdoc::is_arr(branches) else {
        write_stderr(str("get_repo_commit: missing 'branches' for '", repo, "'\n"));
        return commit;
    }
    var name: u8[>..] = [];
    for k in 0..jdoc::len(branches) {
        let b = jdoc::at(branches, k);
        let n = jdoc::get(b, "name");
        let c = jdoc::get(b, "targetCommit");
        guard jdoc::is_str(n) && jdoc::is_str(c) else { continue; }
        guard is_hex(jdoc::str_of(c), 40) else { continue; }
        guard is_valid_subpath(jdoc::str_of(n)) else { continue; }
        if jdoc::str_of(n) == "main" || name.len == 0 {
            name.clear(); name.append(jdoc::str_of(n));
            commit.clear(); commit.append(jdoc::str_of(c));
            if name == "main" { break; }
        }
    }
    guard commit.len > 0 else {
        write_stderr(str("get_repo_commit: no valid branch for '", repo, "'\n"));
        return commit;
    }
    let ref_path = str(repo_path, "/refs/", name);
    make_dirs(parent(ref_path));
    write_file_atomic(ref_path, commit);
    return commit;
}

// The repository's files, from the Hub, into files.
fn repo_files(repo: const u8[:], repo_path: const u8[:]) {
    let commit = repo_commit(repo, repo_path);
    guard commit.len > 0 else {
        write_stderr(str("get_repo_files: failed to resolve commit for ", repo, "\n"));
        return;
    }
    let ep = endpoint();
    let root = api_get(str(ep, "api/models/", repo, "/tree/", commit, "?recursive=true"));
    guard jdoc::is_arr(root) else { return; }
    for k in 0..jdoc::len(root) {
        let item = jdoc::at(root, k);
        guard jdoc::get_str(item, "type", "") == "file" && jdoc::is_str(jdoc::get(item, "path")) else { continue; }
        let path = jdoc::str_of(jdoc::get(item, "path"));
        guard is_valid_subpath(path) else {
            write_stderr(str("get_repo_files: skip invalid path: ", path, "\n"));
            continue;
        }
        let lfs = jdoc::get(item, "lfs");
        let oid = if jdoc::is_obj(lfs) { jdoc::get_str(lfs, "oid", "") } else { jdoc::get_str(item, "oid", "") };
        guard oid.len == 0 || is_hex(oid, 40) || is_hex(oid, 64) else { continue; }
        let final = str(repo_path, "/snapshots/", commit, "/", path);
        let local: u8[>..] = if oid.len > 0 && !file_exists(final) { str(repo_path, "/blobs/", oid) } else { copy(final) };
        files.push(File { path: keep(path), local: keep(local), final: keep(final),
                          url: keep(str(ep, repo, "/resolve/", commit, "/", path)), oid: keep(oid) });
    }
}

// The cached snapshot's files (of refs/main, or the first ref), into files.
fn cached_files(repo_path: const u8[:]) {
    var refs: u8[>..] = [];
    guard list_dir(str(repo_path, "/refs"), refs) else { return; }
    var commit: u8[>..] = [];
    each_split(refs, '\n') {
        var c: u8[>..] = [];
        if it.len > 0 && read_file(str(repo_path, "/refs/", it), c) {
            let first, rest, has_more = split_once(c, '\n');
            let line = trim(first);
            if is_hex(line, 40) && (it == "main" || commit.len == 0) { commit.clear(); commit.append(line); }
        }
    };
    guard commit.len > 0 else { return; }
    let snap = str(repo_path, "/snapshots/", commit);
    var dirs: u8[>..] = [];
    dirs.push(0);
    while dirs.len > 0 {
        // the directories still to list, NUL-ended, relative to snap
        var end = dirs.len - 1;
        var start = end;
        while start > 0 && dirs[start - 1] != 0 { start -= 1; }
        let rel = str(dirs[start..end]);
        dirs.resize(start, 0);
        var names: u8[>..] = [];
        let dir: u8[>..] = if rel.len > 0 { str(snap, "/", rel) } else { str(snap) };
        guard list_dir(dir, names) else { continue; }
        each_split(names, '\n') {
            if it.len > 0 {
                let r = if rel.len > 0 { str(rel, "/", it) } else { str(it) };
                let full = str(snap, "/", r);
                if is_dir(full) { dirs.append(r); dirs.push(0); }
                else { files.push(File { path: keep(r), local: keep(full), final: keep(full), url: keep(""), oid: keep("") }); }
            }
        };
    }
}

// A GGUF file name's split parts and quant: llama.cpp's gguf_split_info.
struct Split { prefix: Span, tag: Span, index: i64 = 0, count: i64 = 0 }

fn is_tag_char(c: u8) -> bool { (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' }

fn split_info(path: const u8[:]) -> Split {
    guard ends_with(path, ".gguf") else { return Split { prefix: Span {}, tag: Span {} }; }
    var prefix = path[..path.len - 5];
    var index = 1;
    var count = 1;
    // <prefix>-NNNNN-of-NNNNN
    let n = prefix.len;
    if n >= 16 && prefix[n - 15] == '-' && upper(prefix[n - 9..n - 5]) == "-OF-" {
        let a, ok_a = parse_int(prefix[n - 14..n - 9]);
        let b, ok_b = parse_int(prefix[n - 5..n]);
        var digits = true;
        for c in prefix[n - 14..n - 9] { if c < '0' || c > '9' { digits = false; } }
        for c in prefix[n - 5..n] { if c < '0' || c > '9' { digits = false; } }
        if ok_a && ok_b && digits { index = a; count = b; prefix = prefix[..n - 15]; }
    }
    // [-.]<tag> at the end
    var t = prefix.len;
    while t > 0 && is_tag_char(prefix[t - 1]) { t -= 1; }
    var tag: u8[>..] = [];
    if t > 0 && t < prefix.len && (prefix[t - 1] == '-' || prefix[t - 1] == '.') {
        tag.append(prefix[t..]);
        to_upper(tag);
    }
    return Split { prefix: keep(prefix), tag: keep(tag), index: index, count: count };
}

// The bits of a quant: Q4_0 -> 4, F16 -> 16, NVFP4 -> 4.
fn tag_bits(tag: const u8[:]) -> i64 {
    var v = 0;
    var seen = false;
    for c in tag {
        if c >= '0' && c <= '9' { v = v * 10 + (c - '0'); seen = true; }
        else if seen { break; }
    }
    return v;
}

fn quant_bits(path: const u8[:]) -> i64 { tag_bits(get(split_info(path).tag)) }

fn file_name(path: const u8[:]) -> const u8[:] {
    let s = rfind(path, "/");
    return if s >= 0 { path[s + 1..] } else { path };
}

// A model, not an imatrix or an mmproj (or another sidecar) beside it.
fn is_model(path: const u8[:]) -> bool {
    guard ends_with(path, ".gguf") else { return false; }
    let name = file_name(path);
    return find(name, "imatrix") < 0 && find(name, "mmproj") < 0 && find(name, "mtp-") < 0 &&
           find(name, "dflash-") < 0 && find(name, "eagle3-") < 0 && find(name, "dspark-") < 0;
}

fn upper(s: const u8[:]) -> u8[>..] {
    var u = str(s);
    to_upper(u);
    return u;
}

// Whether path has the tag as a whole token: (^|[-./])tag[.-], ignoring case.
fn has_tag(path: const u8[:], tag: const u8[:]) -> bool {
    let p = upper(path);
    let t = upper(tag);
    var at = 0;
    while at + t.len < p.len {
        let k = find(p[at..], t);
        guard k >= 0 else { return false; }
        let s = at + k;
        let e = s + t.len;
        if (s == 0 || p[s - 1] == '-' || p[s - 1] == '.' || p[s - 1] == '/') && e < p.len && (p[e] == '.' || p[e] == '-') { return true; }
        at = s + 1;
    }
    return false;
}

// The model file for the tag (Q4_K_M, then Q8_0, then the first, when no
// tag is given): the first part when it is split. -1 if none.
fn best_model(tag: const u8[:]) -> i64 {
    var tags: const u8[:][>..] = [];
    if tag.len > 0 { tags.push(tag); } else { tags.push("Q4_K_M"); tags.push("Q8_0"); }
    for t in tags {
        for f, i in files {
            let p = get(f.path);
            guard is_model(p) && has_tag(p, t) else { continue; }
            let sp = split_info(p);
            guard sp.count <= 1 || sp.index == 1 else { continue; }
            return i;
        }
    }
    if tag.len == 0 {
        for f, i in files {
            let p = get(f.path);
            guard is_model(p) else { continue; }
            let sp = split_info(p);
            guard sp.count <= 1 || sp.index == 1 else { continue; }
            return i;
        }
    }
    return -1;
}

// The model's mmproj: in its directory or one above it (the deepest),
// then the closest quant. -1 if none.
fn best_sibling(model: const u8[:], keyword: const u8[:]) -> i64 {
    let model_bits = quant_bits(model);
    let model_dir = parent(model);
    var best = -1;
    var best_depth = 0;
    var best_diff = 0;
    for f, i in files {
        let p = get(f.path);
        guard ends_with(p, ".gguf") && find(p, keyword) >= 0 else { continue; }
        let dir = parent(p);
        // the sibling's directory holds the model
        guard dir.len == 0 || dir == model_dir || starts_with(model_dir, str(dir, "/")) else { continue; }
        var depth = if dir.len > 0 { 1 } else { 0 };
        for c in dir { if c == '/' { depth += 1; } }
        var diff = quant_bits(p) - model_bits;
        if diff < 0 { diff = -diff; }
        if best < 0 || depth > best_depth || (depth == best_depth && diff < best_diff) {
            best = i;
            best_depth = depth;
            best_diff = diff;
        }
    }
    return best;
}

fn list_ggufs() {
    write_stderr("Available GGUF files:\n");
    for f in files {
        if ends_with(get(f.path), ".gguf") { write_stderr(str(" - ", get(f.path), "\n")); }
    }
}

// Downloads url to path, unless path or alt (the snapshot entry) is there:
// a HEAD for the size and whether the server resumes, then up to 3
// attempts into path + PARTIAL_SUFFIX, each resuming the last.
fn download_file(url: const u8[:], path: const u8[:], alt: const u8[:]) -> bool {
    if file_exists(path) || (alt.len > 0 && file_exists(alt)) { return true; }
    guard !offline else {
        write_stderr(str("common_download_file_single: required file is not available in cache (offline mode): ", path, "\n"));
        return false;
    }
    let headers = request_headers(false);
    let status = c_request(url, headers, 1, "", 0, 0);
    guard status >= 200 && status < 300 else {
        if status < 0 { write_stderr(str("download: ", last_error(), "\n")); }
        write_stderr(str("download: HEAD ", url, " failed, status: ", status, "\n"));
        return false;
    }
    let ranges, has_ranges = last_header("Accept-Ranges");
    let supports_ranges = has_ranges && ranges != "none";
    make_dirs(parent(path));
    let tmp = str(path, PARTIAL_SUFFIX);
    var delay = 2;
    for attempt in 0..3 {
        if attempt > 0 {
            write_stderr(str("common_download_file_single_online: retrying after ", delay, " seconds...\n"));
            sleep(delay);
            delay *= 2;
        }
        var existing = c_file_size(tmp);
        if existing > 0 && !supports_ranges { delete_file(tmp); existing = 0; }
        if existing < 0 { existing = 0; }
        let st = c_request(url, headers, 0, tmp, existing, 1);
        guard st == 200 || st == 206 else {
            if st < 0 { write_stderr(str("common_pull_file: download failed: ", last_error(), "\n")); }
            else { write_stderr(str("common_pull_file: download received non-successful status code: ", st, "\n")); }
            // a range the server cannot serve: start over
            if st == 416 { delete_file(tmp); }
            continue;
        }
        guard rename_file(tmp, path) else {
            write_stderr(str("common_download_file_single_online: unable to rename file: ", tmp, " to ", path, "\n"));
            return false;
        }
        return true;
    }
    write_stderr("common_download_file_single_online: download failed after 3 attempts\n");
    return false;
}

// The downloaded blob linked from (or, where links cannot be made, moved
// to) its snapshot entry: the entry's path.
var symlinks_disabled = false;

fn finalize(f: File) -> u8[>..] {
    let local = get(f.local);
    let final = get(f.final);
    if local == final || file_exists(final) || !file_exists(local) { return str(final); }
    make_dirs(parent(final));
    if !symlinks_disabled {
        // the blob from the entry's directory: up to the repository's folder
        // (snapshots/<commit>/<path>), then down to blobs/<oid>
        var target: u8[>..] = [];
        target.append("../../");
        for c in get(f.path) { if c == '/' { target.append("../"); } }
        target.append(str("blobs/", get(f.oid)));
        if c_symlink(target, final) != 0 { return str(final); }
        symlinks_disabled = true;
        write_stderr("finalize_file: failed to create symlink\nfinalize_file: switching to degraded mode\n");
    }
    if !rename_file(local, final) {
        write_stderr(str("finalize_file: failed to move file to snapshots: ", final, "\n"));
    }
    return str(final);
}

// resolve for a program that takes no mmproj.
fn resolve(model_path: u8[>..]&) {
    var none: u8[>..] = [];
    resolve(model_path, none, false);
}

// Downloads the files -hf names (when it is given) and gives the model's
// path in model_path, and the mmproj's in mmproj_path when want_mmproj and
// it is empty. Exits on failure, as llama.cpp's programs do.
fn resolve(model_path: u8[>..]&, mmproj_path: u8[>..]&, want_mmproj: bool) {
    from_env();
    guard hf_repo.len > 0 else { return; }
    var repo = str(hf_repo);
    var tag: u8[>..] = [];
    let c = rfind(hf_repo, ":");
    if c >= 0 { repo.resize(c, 0); tag.append(hf_repo[c + 1..]); }
    var slashes = 0;
    for ch in repo { if ch == '/' { slashes += 1; } }
    guard slashes == 1 && repo[0] != '/' && repo[repo.len - 1] != '/' else {
        fail("invalid HF repo format, expected <user>/<model>[:quant]");
        return;
    }
    let repo_path = str(cache_dir(), "/", repo_folder(repo));
    files.clear();
    text.clear();
    if !offline { repo_files(repo, repo_path); }
    if files.len == 0 { cached_files(repo_path); }
    guard files.len > 0 else {
        fail(str("failed to get the files of ", repo, " (not in the cache either)"));
        return;
    }

    var primary = -1;
    if hf_file.len > 0 {
        for f, i in files { if get(f.path) == hf_file { primary = i; break; } }
        if primary < 0 {
            write_stderr(str("common_download_get_hf_plan: file '", hf_file, "' not found in repository\n"));
            list_ggufs();
            exit(1);
        }
    } else {
        primary = best_model(tag);
        if primary < 0 {
            write_stderr(str("common_download_get_hf_plan: no GGUF files found in repository ", repo, "\n"));
            list_ggufs();
            exit(1);
        }
    }

    // the model's split parts, the first one first
    let p = get(files[primary].path);
    let sp = split_info(p);
    var parts: i64[>..] = [];
    if sp.count <= 1 {
        parts.push(primary);
    } else {
        for f, i in files {
            let s = split_info(get(f.path));
            if s.count == sp.count && get(s.prefix) == get(sp.prefix) { parts.push(i); }
        }
    }
    var mmproj = -1;
    if want_mmproj && mmproj_path.len == 0 { mmproj = best_sibling(p, "mmproj"); }

    for i in parts {
        let f = files[i];
        guard download_file(get(f.url), get(f.local), get(f.final)) else { fail(str("Download '", get(f.url), "' failed")); }
        let path = finalize(f);
        if i == primary { model_path.clear(); model_path.append(path); }
    }
    if mmproj >= 0 {
        let f = files[mmproj];
        guard download_file(get(f.url), get(f.local), get(f.final)) else { fail(str("Download '", get(f.url), "' failed")); }
        mmproj_path.clear();
        mmproj_path.append(finalize(f));
    }
}
