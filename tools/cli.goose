// llama.cpp's llama-cli: a chat in the console with a model that a server
// runs. It starts tools/server (the server beside this program) on a free
// port of 127.0.0.1, with the arguments it does not take itself, and talks
// to it over HTTP, as llama-cli does with its llama-server; with
// --server-base it talks to a server already running instead.
//   cli -m model.gguf [-sys prompt] [-p prompt] [-st] [-mli] [-co on|off|auto]
//       [--show-timings | --no-show-timings] [-o file] [the server's arguments]
//   cli --server-base http://host:port [...]
// Commands: /exit, /regen, /clear, /read <file>, /glob <pattern>.
// port-of: tools/cli/cli.cpp tools/cli/cli-context.cpp tools/cli/cli-ui.h tools/cli/cli-client.cpp tools/cli/cli-server.h
import std;
import os;
import common from "../common";
import common.console;
import common.jdoc;
import net from "./net";
import proc from "./proc";

// llama-cli's logo, in UTF-8
let LLAMA_ASCII_LOGO = str(
    "\n",
    "\xE2\x96\x84\xE2\x96\x84 \xE2\x96\x84\xE2\x96\x84\n",
    "\xE2\x96\x88\xE2\x96\x88 \xE2\x96\x88\xE2\x96\x88\n",
    "\xE2\x96\x88\xE2\x96\x88 \xE2\x96\x88\xE2\x96\x88  \xE2\x96\x80\xE2\x96\x80\xE2\x96\x88\xE2\x96\x84 \xE2\x96\x88\xE2\x96\x88\xE2\x96\x88\xE2\x96\x84\xE2\x96\x88\xE2\x96\x88\xE2\x96\x88\xE2\x96\x84  \xE2\x96\x80\xE2\x96\x80\xE2\x96\x88\xE2\x96\x84    \xE2\x96\x84\xE2\x96\x88\xE2\x96\x88\xE2\x96\x88\xE2\x96\x88 \xE2\x96\x88\xE2\x96\x88\xE2\x96\x88\xE2\x96\x88\xE2\x96\x84 \xE2\x96\x88\xE2\x96\x88\xE2\x96\x88\xE2\x96\x88\xE2\x96\x84\n",
    "\xE2\x96\x88\xE2\x96\x88 \xE2\x96\x88\xE2\x96\x88 \xE2\x96\x84\xE2\x96\x88\xE2\x96\x80\xE2\x96\x88\xE2\x96\x88 \xE2\x96\x88\xE2\x96\x88 \xE2\x96\x88\xE2\x96\x88 \xE2\x96\x88\xE2\x96\x88 \xE2\x96\x84\xE2\x96\x88\xE2\x96\x80\xE2\x96\x88\xE2\x96\x88    \xE2\x96\x88\xE2\x96\x88    \xE2\x96\x88\xE2\x96\x88 \xE2\x96\x88\xE2\x96\x88 \xE2\x96\x88\xE2\x96\x88 \xE2\x96\x88\xE2\x96\x88\n",
    "\xE2\x96\x88\xE2\x96\x88 \xE2\x96\x88\xE2\x96\x88 \xE2\x96\x80\xE2\x96\x88\xE2\x96\x84\xE2\x96\x88\xE2\x96\x88 \xE2\x96\x88\xE2\x96\x88 \xE2\x96\x88\xE2\x96\x88 \xE2\x96\x88\xE2\x96\x88 \xE2\x96\x80\xE2\x96\x88\xE2\x96\x84\xE2\x96\x88\xE2\x96\x88 \xE2\x96\x88\xE2\x96\x88 \xE2\x96\x80\xE2\x96\x88\xE2\x96\x88\xE2\x96\x88\xE2\x96\x88 \xE2\x96\x88\xE2\x96\x88\xE2\x96\x88\xE2\x96\x88\xE2\x96\x80 \xE2\x96\x88\xE2\x96\x88\xE2\x96\x88\xE2\x96\x88\xE2\x96\x80\n",
    "                                    \xE2\x96\x88\xE2\x96\x88    \xE2\x96\x88\xE2\x96\x88\n",
    "                                    \xE2\x96\x80\xE2\x96\x80    \xE2\x96\x80\xE2\x96\x80\n");

let FILE_GLOB_MAX_RESULTS = 100;

fn say(s: const u8[:]) { write_stdout(s); flush_stdout(); }
fn show_message(s: const u8[:]) { say(s); say("\n"); }

fn show_error(title: const u8[:], message: const u8[:] = "") {
    console::set_display(console::ERROR);
    say(str("Error: ", title, "\n"));
    console::set_display(console::RESET);
    if message.len > 0 { show_message(message); }
}

fn show_info(s: const u8[:]) {
    console::set_display(console::INFO);
    show_message(s);
    console::set_display(console::RESET);
}

// The server: where it is, and the one this program started (0 for none).
var host: u8[>..] = [];
var port = 0;
var prefix: u8[>..] = [];
var server = 0;
var model: u8[>..] = [];        // the model a --server-base request names, if any

fn path_of(p: const u8[:]) -> u8[>..] { str(prefix, p) }

// GET path: the body, or an error.
fn get(p: const u8[:], out: u8[>..]&) -> bool {
    var full = path_of(p);
    if model.len > 0 { full.append("?model="); full.append(model); }
    let status = net::request(host, port, "GET", full, "", out);
    return status >= 200 && status < 300;
}

// The message of an error response: error.message, or the text as it is.
fn error_message(text: const u8[:]) -> u8[>..] {
    jdoc::reset();
    let v = jdoc::parse(text);
    let e = jdoc::get(v, "error");
    if jdoc::is_obj(e) && jdoc::is_str(jdoc::get(e, "message")) { return str(jdoc::get_str(e, "message", "")); }
    return str(text);
}

// wait_health: true once /health answers 200.
fn wait_health() -> bool {
    var attempts = 0;
    loop {
        if proc::interrupts() > 0 { return false; }
        if server != 0 && !proc::alive(server) { return false; }
        var body: u8[>..] = [];
        let status = net::request(host, port, "GET", path_of("/health"), "", body);
        if status == 200 { return true; }
        if status < 0 {
            attempts++;
            // a server being started may take a while to listen
            if server == 0 && attempts >= 10 { show_error(str("failed to connect to http://", host, ":", port)); return false; }
        }
        sleep_ms(300);
    }
}

var model_name: u8[>..] = [];
var model_ftype: u8[>..] = [];
var build_info: u8[>..] = [];

// fetch_server_props: the model's name and type, and the server's build.
fn fetch_server_props() {
    var body: u8[>..] = [];
    guard get("/props", body) else { return; }
    jdoc::reset();
    let props = jdoc::parse(body);
    guard props >= 0 else { return; }
    model_name.clear();
    model_name.append(jdoc::get_str(props, "model_alias", ""));
    if model_name.len == 0 {
        let p = jdoc::get_str(props, "model_path", "");
        var at = 0;
        for c, i in p { if c == '/' || c == '\\' { at = i + 1; } }
        model_name.append(p[at..]);
    }
    model_ftype.clear();
    model_ftype.append(jdoc::get_str(props, "model_ftype", ""));
    build_info.clear();
    build_info.append(jdoc::get_str(props, "build_info", ""));
}

// list_and_ask_models: the model of a server that has several, asked.
fn list_and_ask_models() -> bool {
    var body: u8[>..] = [];
    if !get("/v1/models", body) { show_error(error_message(body)); return false; }
    jdoc::reset();
    let resp = jdoc::parse(body);
    let data = jdoc::get(resp, "data");
    if !jdoc::is_arr(data) {
        show_error("invalid response from /v1/models");
        show_message("This might be caused by an incorrect server-base endpoint URL");
        return false;
    }
    var names: u8[>..] = [];         // each ended by a NUL
    var shown: u8[>..] = [];
    var n = 0;
    for i in jdoc::len(data) {
        let m = jdoc::at(data, i);
        guard jdoc::is_str(jdoc::get(m, "id")) else { continue; }
        let id = jdoc::get_str(m, "id", "");
        names.append(id);
        names.push(0);
        shown.append(id);
        let aliases = jdoc::get(m, "aliases");
        if jdoc::is_arr(aliases) && jdoc::len(aliases) > 0 {
            shown.append(" (");
            for k in jdoc::len(aliases) {
                if k > 0 { shown.append(", "); }
                shown.append(jdoc::str_of(jdoc::at(aliases, k)));
            }
            shown.push(')');
        }
        shown.push(0);
        n++;
    }
    if n == 1 {
        model_name.clear();
        model_name.append(nth(names, 0));
        model.clear();
        model.append(model_name);
        return true;
    }
    var message: u8[>..] = [];
    message.append("\nAvailable models:");
    for i in n { format(message, "\n  ", i + 1, ". ", nth(shown, i)); }
    message.append("\n");
    show_message(message);
    loop {
        if proc::take_interrupts() > 0 { return false; }
        console::set_display(console::USER_INPUT);
        say("Select model by number: ");
        var line: u8[>..] = [];
        console::readline(line, false);
        console::set_display(console::RESET);
        guard line.len > 0 else { return false; }
        let idx, ok = parse_int(trim(line));
        if ok && idx > 0 && idx <= n {
            model_name.clear();
            model_name.append(nth(names, idx - 1));
            model.clear();
            model.append(model_name);
            show_message(str("Selected model: ", model_name));
            return true;
        }
        show_error("Invalid selection. Please enter a valid number.");
    }
}

fn nth(list: const u8[:], n: i64) -> const u8[:] {
    var start = 0;
    var k = 0;
    for b, i in list {
        guard b == 0 else { continue; }
        if k == n { return list[start..i]; }
        k++;
        start = i + 1;
    }
    return "";
}

// The chat: its messages as JSON objects, each ended by a NUL.
var messages: u8[>..] = [];
var system_prompt: u8[>..] = [];

fn push_message(role: const u8[:], content: const u8[:], reasoning: const u8[:] = "") {
    var o: u8[>..] = [];
    o.append("{\"role\":");
    jdoc::put_str(o, role);
    o.append(",\"content\":");
    jdoc::put_str(o, content);
    if reasoning.len > 0 {
        o.append(",\"reasoning_content\":");
        jdoc::put_str(o, reasoning);
    }
    o.push('}');
    messages.append(o);
    messages.push(0);
}

fn message_count() -> i64 {
    var n = 0;
    for b in messages { if b == 0 { n++; } }
    return n;
}

fn drop_last_message() {
    guard messages.len > 0 else { return; }
    var i = messages.len - 1;
    while i > 0 && messages[i - 1] != 0 { i--; }
    messages.resize(i, 0);
}

fn add_system_prompt() {
    if system_prompt.len > 0 { push_message("system", system_prompt); }
}

var output_file: u8[>..] = [];

fn write_output_file(s: const u8[:]) {
    if output_file.len > 0 { append_file(output_file, s); }
}

// The reply as it streams, shown: reasoning dimmed between its markers
// (ui::assistant_turn).
let MODE_CONTENT = 0;
let MODE_REASONING = 1;
var mode = MODE_CONTENT;
var trailing_newline = true;
var inside_reasoning = false;

fn push_text(m: i64, text: const u8[:]) {
    if m != mode {
        add_newline_if_needed();
        if m == MODE_CONTENT {
            if inside_reasoning {
                say("[End thinking]\n\n");
                inside_reasoning = false;
            }
            console::set_display(console::RESET);
        } else {
            console::set_display(console::REASONING);
            inside_reasoning = true;
            say("\n[Start thinking]\n\n");
        }
    }
    mode = m;
    guard text.len > 0 else { return; }
    trailing_newline = text[text.len - 1] == '\n';
    say(text);
}

fn add_newline_if_needed() { if !trailing_newline { say("\n"); } }

var reply_content: u8[>..] = [];
var reply_reasoning: u8[>..] = [];
var prompt_tps = 0.0;
var predicted_tps = 0.0;

// generate_completion: the reply to the chat so far, streamed and shown.
fn generate_completion() -> bool {
    reply_content.clear();
    reply_reasoning.clear();
    prompt_tps = 0.0;
    predicted_tps = 0.0;
    var body: u8[>..] = [];
    body.append("{\"messages\":[");
    var first = true;
    var start = 0;
    for b, i in messages {
        guard b == 0 else { continue; }
        if !first { body.push(','); }
        first = false;
        body.append(messages[start..i]);
        start = i + 1;
    }
    // timings_per_token: the timings come even when the reply is cut short
    body.append("],\"stream\":true,\"timings_per_token\":true");
    if model.len > 0 { body.append(",\"model\":"); jdoc::put_str(body, model); }
    body.push('}');

    mode = MODE_CONTENT;
    trailing_newline = true;
    inside_reasoning = false;
    console::set_display(console::RESET);
    var ok = true;
    let status = net::open_request(host, port, "POST", path_of("/v1/chat/completions"), body, "text/event-stream");
    if status < 0 {
        show_error(str("failed to connect to http://", host, ":", port));
        return false;
    }
    if status < 200 || status >= 300 {
        while net::read_more(60000) {}
        var text: u8[>..] = [];
        text.append(net::cbody);
        net::close_request();
        if text.len == 0 { format(text, "request failed with status ", status); }
        show_error(error_message(text));
        return false;
    }
    var payload: u8[>..] = [];
    loop {
        if proc::interrupts() > 0 { break; }
        let r = net::next_event(payload, 100);
        if r < 0 { break; }
        if r == 0 { continue; }
        if payload == "[DONE]" { continue; }
        jdoc::reset();
        let chunk = jdoc::parse(payload);
        guard chunk >= 0 else { continue; }
        if jdoc::contains(chunk, "error") {
            ok = false;
            var e: u8[>..] = [];
            jdoc::dump(e, chunk);
            show_error(error_message(e));
            continue;
        }
        let t = jdoc::get(chunk, "timings");
        if jdoc::is_obj(t) {
            prompt_tps = jdoc::get_num(t, "prompt_per_second", 0.0);
            predicted_tps = jdoc::get_num(t, "predicted_per_second", 0.0);
        }
        let choices = jdoc::get(chunk, "choices");
        guard jdoc::is_arr(choices) && jdoc::len(choices) > 0 else { continue; }
        let delta = jdoc::get(jdoc::at(choices, 0), "delta");
        guard delta >= 0 else { continue; }
        let r_text = jdoc::get(delta, "reasoning_content");
        if jdoc::is_str(r_text) && jdoc::str_of(r_text).len > 0 {
            reply_reasoning.append(jdoc::str_of(r_text));
            push_text(MODE_REASONING, jdoc::str_of(r_text));
        }
        let c_text = jdoc::get(delta, "content");
        if jdoc::is_str(c_text) && jdoc::str_of(c_text).len > 0 {
            reply_content.append(jdoc::str_of(c_text));
            push_text(MODE_CONTENT, jdoc::str_of(c_text));
        }
    }
    net::close_request();
    proc::take_interrupts();
    console::set_display(console::RESET);
    add_newline_if_needed();
    return ok;
}

// glob_match: * any run but '/', ** any run, ? one character but '/', and
// [...] a class ([!...] negated).
recursive fn glob_match(pat: const u8[:], s: const u8[:]) -> bool {
    var p = 0;
    var i = 0;
    while p < pat.len {
        let c = pat[p];
        if c == '*' {
            let any = p + 1 < pat.len && pat[p + 1] == '*';
            let rest = pat[if any { p + 2 } else { p + 1 }..];
            var j = i;
            loop {
                if glob_match(rest, s[j..]) { return true; }
                guard j < s.len && (any || s[j] != '/') else { return false; }
                j++;
            }
        }
        guard i < s.len else { return false; }
        if c == '?' {
            guard s[i] != '/' else { return false; }
        } else if c == '[' {
            var q = p + 1;
            let neg = q < pat.len && pat[q] == '!';
            if neg { q++; }
            var matched = false;
            while q < pat.len && pat[q] != ']' {
                if q + 2 < pat.len && pat[q + 1] == '-' && pat[q + 2] != ']' {
                    if s[i] >= pat[q] && s[i] <= pat[q + 2] { matched = true; }
                    q += 3;
                } else {
                    if s[i] == pat[q] { matched = true; }
                    q++;
                }
            }
            guard matched != neg else { return false; }
            p = q;
        } else if c != s[i] {
            return false;
        }
        p++;
        i++;
    }
    return i == s.len;
}

// The files under base (the current directory when empty), as paths
// relative to it with '/', into out, each ended by a NUL; directories are
// visited in the order found, breadth first.
fn walk(base: const u8[:], out: u8[>..]&) {
    var dirs: u8[>..] = [];         // relative paths still to list, each ended by a NUL
    dirs.push(0);                   // base itself
    var at = 0;
    while at < dirs.len {
        var end = at;
        while dirs[end] != 0 { end++; }
        var rel: u8[>..] = [];
        rel.append(dirs[at..end]);
        at = end + 1;
        var dir: u8[>..] = [];
        dir.append(base);
        if base.len > 0 && rel.len > 0 { dir.push('/'); }
        dir.append(rel);
        var names: u8[>..] = [];
        guard list_dir(if dir.len > 0 { dir[..] } else { "." }, names) else { continue; }
        var start = 0;
        for c, i in names {
            guard c == '\n' else { continue; }
            let name = names[start..i];
            start = i + 1;
            let r = str(rel, if rel.len > 0 { "/" } else { "" }, name);
            let full = str(dir, if dir.len > 0 { "/" } else { "" }, name);
            if is_dir(full) { dirs.append(r); dirs.push(0); } else { out.append(r); out.push(0); }
        }
    }
}

fn main() {
    let args = console::utf8_args();
    var forward: u8[>..] = [];      // the server's arguments, quoted for its command line
    var prompt: u8[>..] = [];
    var server_base: u8[>..] = [];
    var single_turn = false;
    var multiline = false;
    var show_timings = true;
    var use_color = console::tty_can_use_colors();
    var has_model = false;
    var i = 1;
    fn value() -> const u8[:] {
        if i + 1 >= args.len { show_error(str("missing value for ", args[i])); exit(1); }
        i += 1;
        return args[i];
    }
    fn quote(a: const u8[:]) {
        forward.push(' ');
        if a.len > 0 && find(a, " ") < 0 && find(a, "\"") < 0 && find(a, "\t") < 0 { forward.append(a); return; }
        forward.push('"');
        for c in a {
            if c == '"' { forward.append("\\\""); } else { forward.push(c); }
        }
        forward.push('"');
    }
    while i < args.len {
        let a = args[i];
        if a == "-sys" || a == "--system-prompt" { system_prompt.clear(); system_prompt.append(value()); }
        else if a == "-p" || a == "--prompt" { prompt.clear(); prompt.append(value()); }
        else if a == "-st" || a == "--single-turn" { single_turn = true; }
        else if a == "-mli" || a == "--multiline-input" { multiline = true; }
        else if a == "--show-timings" { show_timings = true; }
        else if a == "--no-show-timings" { show_timings = false; }
        else if a == "-o" || a == "--output-file" { output_file.clear(); output_file.append(value()); }
        else if a == "--server-base" { server_base.clear(); server_base.append(value()); }
        else if a == "-co" || a == "--color" {
            let v = value();
            use_color = if v == "on" || v == "true" || v == "1" { true } else if v == "off" || v == "false" || v == "0" { false } else { console::tty_can_use_colors() };
        }
        else if a == "-hf" || a == "--hf-repo" { show_error("-hf: downloading models is not ported yet; use -m <file.gguf>"); exit(1); }
        else {
            if a == "-m" || a == "--model" { has_model = true; }
            quote(a);
        }
        i += 1;
    }
    console::init(use_color);
    proc::catch_interrupts();

    if server_base.len > 0 {
        let h, p, pre, ok = net::parse_url(server_base);
        guard ok else { show_error(str("invalid --server-base: ", server_base)); exit(1); }
        host.append(h);
        port = p;
        prefix.append(pre);
        say(str("Connecting to server at ", server_base, " "));
    } else {
        if !has_model {
            show_error("no model specified", "use -m <file.gguf> to run a local model,\nor --server-base <url> to connect to a running llama-server");
            exit(1);
        }
        // the server beside this program, on a free port
        var exe = proc::exe_path();
        var dir_end = 0;
        for c, k in exe { if c == '/' || c == '\\' { dir_end = k + 1; } }
        let ext = if ends_with(exe, ".exe") { ".exe" } else { "" };
        let server_path = str(exe[..dir_end], "server", ext);
        guard file_exists(server_path) else { show_error(str("the server is not built: ", server_path)); exit(1); }
        let probe = net::listen("127.0.0.1", 0);
        port = net::port_of(probe);
        net::close(probe);
        host.append("127.0.0.1");
        var cmdline: u8[>..] = [];
        cmdline.push('"');
        cmdline.append(server_path);
        cmdline.push('"');
        cmdline.append(forward);
        format(cmdline, " --host 127.0.0.1 --port ", port);
        say("\n\nLoading model... ");
        server = proc::spawn(cmdline);
        guard server != 0 else { show_error("server start failed"); exit(1); }
    }
    if !wait_health() {
        if proc::interrupts() == 0 { show_error("the server exited before becoming ready"); }
        if server != 0 { proc::kill(server); }
        exit(1);
    }
    say("\n");
    if server_base.len > 0 && !list_and_ask_models() {
        exit(1);
    }
    fetch_server_props();
    let code = run(prompt, single_turn, multiline, show_timings);
    if server != 0 { proc::kill(server); }
    console::cleanup();
    exit(code);
}

fn run(prompt0: const u8[:], single_turn: bool, multiline: bool, show_timings: bool) -> i64 {
    add_system_prompt();
    var banner: u8[>..] = [];
    banner.append("\n");
    banner.append(LLAMA_ASCII_LOGO);
    banner.append("\n");
    format(banner, "build      : ", build_info, "\n");
    format(banner, "model      : ", model_name, "\n");
    if model_ftype.len > 0 { format(banner, "ftype      : ", model_ftype, "\n"); }
    banner.append("modalities : text\n");
    if system_prompt.len > 0 { banner.append("using custom system prompt\n"); }
    banner.append("\n");
    banner.append("available commands:\n");
    banner.append("  /exit or Ctrl+C     stop or exit\n");
    banner.append("  /regen              regenerate the last response\n");
    banner.append("  /clear              clear the chat history\n");
    banner.append("  /read <file>        add a text file\n");
    banner.append("  /glob <pattern>     add text files using globbing pattern\n");
    banner.append("\n");
    show_message(banner);

    var prompt: u8[>..] = [];
    prompt.append(prompt0);
    var cur_msg: u8[>..] = [];
    fn add_text_file(fname: const u8[:]) -> bool {
        var content: u8[>..] = [];
        if !read_file(fname, content) {
            show_error(str("file does not exist or cannot be opened: '", fname, "'"));
            return false;
        }
        format(cur_msg, "--- File: ", fname, " ---\n");
        cur_msg.append(content);
        show_message(str("Loaded text from '", fname, "'"));
        return true;
    }

    loop {
        var buffer: u8[>..] = [];
        console::set_display(console::USER_INPUT);
        if prompt.len == 0 {
            say("\n> ");
            var line: u8[>..] = [];
            loop {
                let more = console::readline(line, multiline);
                buffer.append(line);
                if !more { break; }
            }
        } else {
            buffer.append(prompt);
            if buffer.len > 500 { say(str("\n> ", buffer[..500], " ... (truncated)\n")); } else { say(str("\n> ", buffer, "\n")); }
            prompt.clear();
        }
        console::set_display(console::RESET);
        if proc::take_interrupts() > 0 { break; }
        // an empty read is the end of the input
        if buffer.len == 0 { break; }
        if buffer[buffer.len - 1] == '\n' { buffer.pop(); }
        if buffer.len == 0 { continue; }

        var add_user_msg = true;
        if starts_with(buffer, "/exit") {
            break;
        } else if starts_with(buffer, "/regen") {
            if message_count() >= 2 {
                drop_last_message();
                add_user_msg = false;
            } else {
                show_error("No message to regenerate.");
                continue;
            }
        } else if starts_with(buffer, "/clear") {
            messages.clear();
            add_system_prompt();
            show_message("Chat history cleared.");
            continue;
        } else if starts_with(buffer, "/read ") {
            let fname = trim(buffer[6..]);
            add_text_file(fname);
            write_output_file(str("User: Added text file: ", fname, "\n"));
            continue;
        } else if starts_with(buffer, "/glob ") {
            var pattern: u8[>..] = [];
            pattern.append(trim(buffer[6..]));
            var base: u8[>..] = [];
            // the fixed directories before the first wildcard
            var startglob = -1;
            for c, k in pattern { if startglob < 0 && (c == '!' || c == '[' || c == '*' || c == '?') { startglob = k; } }
            if startglob > 0 {
                var endpath = -1;
                for c, k in pattern[..startglob] { if c == '/' { endpath = k; } }
                if endpath >= 0 {
                    base.append(pattern[..endpath]);
                    var rest: u8[>..] = [];
                    rest.append(pattern[endpath + 1..]);
                    pattern.clear();
                    pattern.append(rest);
                }
            }
            var files: u8[>..] = [];
            walk(base, files);
            var count = 0;
            var s = 0;
            for b, k in files {
                guard b == 0 else { continue; }
                let rel = files[s..k];
                s = k + 1;
                guard glob_match(pattern, rel) else { continue; }
                let full = str(base, if base.len > 0 { "/" } else { "" }, rel);
                guard add_text_file(full) else { continue; }
                write_output_file(str("User: Added text file: ", full, "\n"));
                count++;
                if count >= FILE_GLOB_MAX_RESULTS {
                    show_error(str("Maximum number of globbed files allowed (", FILE_GLOB_MAX_RESULTS, ") reached."));
                    break;
                }
            }
            continue;
        } else {
            cur_msg.append(buffer);
        }

        if add_user_msg {
            push_message("user", cur_msg);
            write_output_file(str("User:\n", cur_msg, "\n\n"));
            cur_msg.clear();
        }
        generate_completion();
        push_message("assistant", reply_content, reply_reasoning);
        if output_file.len > 0 {
            var out: u8[>..] = [];
            out.append("Assistant:\n");
            if reply_reasoning.len > 0 {
                out.append("[Start thinking]\n\n");
                out.append(reply_reasoning);
                out.append("[End thinking]\n\n");
            }
            out.append(reply_content);
            if out.len > 0 && out[out.len - 1] != '\n' { out.push('\n'); }
            out.push('\n');
            write_output_file(out);
        }
        if show_timings {
            var t: u8[>..] = [];
            t.append("\n[ Prompt: ");
            format_flt(t, prompt_tps, 1);
            t.append(" t/s | Generation: ");
            format_flt(t, predicted_tps, 1);
            t.append(" t/s ]");
            show_info(t);
        }
        if single_turn { break; }
    }
    show_message("\n\nExiting...");
    return 0;
}
