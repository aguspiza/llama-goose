// A child process, and Ctrl+C, for llama-cli: proc.c's calls.
// port-of: tools/cli/cli-server.h tools/cli/cli.cpp (the signal handler)
import std;
namespace proc;

// From now on a Ctrl+C is counted, not fatal; a second one before the count
// is taken ends the program.
extern "proc_catch_interrupts" fn catch_interrupts();
// The Ctrl+C counted since the last take; the count starts again at 0.
extern "proc_take_interrupts" fn take_interrupts() -> i64;
extern "proc_interrupts" fn interrupts() -> i64;
// Starts a command line with its output discarded: a handle, 0 if it can't.
extern "proc_spawn" fn spawn(cmdline: const u8[:]) -> i64;
extern "proc_alive" fn c_alive(h: i64) -> i64;
extern "proc_kill" fn kill(h: i64);
extern "proc_exe_path" fn c_exe_path(out: u8[:]) -> i64;

fn alive(h: i64) -> bool { c_alive(h) != 0 }

// This program's path.
fn exe_path() -> u8[>..] {
    var buf: u8[>..] = [];
    buf.resize(16384, 0);
    let n = c_exe_path(buf);
    buf.resize(n, 0);
    return buf;
}
