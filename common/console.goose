// llama.cpp's console: the colors that tell the prompt, the user's input
// and the model's text apart, and reading the user's lines. Only the
// simple I/O (line by line), not the line editor.
// port-of: common/console.cpp (set_display, readline_simple), common/common.cpp (tty_can_use_colors)
import std;
import os;

namespace console;

extern "common_is_tty" fn is_tty(fd: i64) -> i64;
extern "common_args_utf8" fn c_args_utf8(out: u8[:]) -> i64;

// The program's arguments, in UTF-8 on Windows too (where llama.cpp's
// programs get them so from their manifest).
var arg_text: u8[>..] = [];

fn utf8_args() -> const u8[:][>..] {
    var out: const u8[:][>..] = [];
    var buf: u8[1] = [0];
    let need = c_args_utf8(buf[..]);
    if need < 0 {
        for a in args() { out.push(a); }
        return out;
    }
    arg_text.clear();
    arg_text.resize(need, 0);
    c_args_utf8(arg_text[..]);
    var start = 0;
    for c, i in arg_text {
        if c == 0 { out.push(arg_text[start..i]); start = i + 1; }
    }
    return out;
}

let RESET = 0;
let INFO = 1;
let PROMPT = 2;
let REASONING = 3;
let USER_INPUT = 4;
let ERROR = 5;

var advanced_display = false;
var current = RESET;

fn init(use_color: bool) { advanced_display = use_color; }

// Colors when stdout or stderr is a terminal, unless NO_COLOR or TERM=dumb
// say otherwise (tty_can_use_colors).
fn tty_can_use_colors() -> bool {
    var v: u8[>..] = [];
    if env("NO_COLOR", v) && v.len > 0 { return false; }
    v.clear();
    if env("TERM", v) && v == "dumb" { return false; }
    return is_tty(1) != 0 || is_tty(2) != 0;
}

// The color for what comes next, written only when it changes.
fn set_display(d: i64) {
    if !advanced_display || current == d { return; }
    write_stdout(match d {
        RESET => "\x1b[0m",
        INFO => "\x1b[35m",
        PROMPT => "\x1b[33m",
        REASONING => "\x1b[90m",
        USER_INPUT => "\x1b[1m\x1b[32m",
        _ => "\x1b[1m\x1b[31m",
    });
    current = d;
    flush_stdout();
}

fn cleanup() { set_display(RESET); }

// One line of input, with its newline, appended to line: whether more
// lines follow (multiline_input, or the other way round when the line ends
// in a backslash). A line ending in '/' returns control without a newline.
// At the end of the input, line is left empty and the answer is false.
fn readline(line: u8[>..]&, multiline_input: bool) -> bool {
    line.clear();
    if !read_line(line) { line.clear(); return false; }
    // the text mode of the C runtime reads \r\n as \n
    if line.len > 0 && line[line.len - 1] == '\r' { line.pop(); }
    var more = multiline_input;
    if line.len > 0 {
        let last = line[line.len - 1];
        if last == '/' {
            line[line.len - 1] = '\n';
            return false;
        }
        if last == '\\' {
            line.pop();
            more = !more;
        }
    }
    line.push('\n');
    return more;
}
