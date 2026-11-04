// HTTP/1.1 over TCP, as much as llama-server and its clients need: one
// request per connection (each response closes it), bodies by
// Content-Length, and server-sent events. The sockets are net.c's.
// port-of: tools/server/server-http.cpp
import std;
namespace net;

extern "net_listen" fn listen(host: const u8[:], port: i64) -> i64;
extern "net_port" fn port_of(fd: i64) -> i64;
extern "net_accept" fn accept(fd: i64) -> i64;
extern "net_connect" fn connect(host: const u8[:], port: i64) -> i64;
extern "net_recv" fn recv(fd: i64, buf: u8[:]) -> i64;
extern "net_send" fn c_send(fd: i64, data: const u8[:]) -> i64;
extern "net_readable" fn readable(fd: i64) -> i64;
extern "net_close" fn close(fd: i64);
// The index of the first of fds that is readable within timeout_ms, or -1.
extern "net_wait" fn wait(fds: const i64[:], timeout_ms: i64) -> i64;
extern "net_set_timeout" fn set_timeout(fd: i64, ms: i64);

fn send(fd: i64, data: const u8[:]) -> bool { c_send(fd, data) == 0 }

let MAX_HEAD = 64 * 1024;
let MAX_BODY = 256 * 1024 * 1024;

// The request read last: its parts are views of head and body.
var head: u8[>..] = [];         // the request line and the headers
var body: u8[>..] = [];
var method_end = 0;
var target_start = 0;
var target_end = 0;

fn method() -> const u8[:] { head[..method_end] }
// The path, without the query.
fn path() -> const u8[:] {
    let t = head[target_start..target_end];
    let q = find(t, "?");
    return if q >= 0 { t[..q] } else { t };
}
fn query() -> const u8[:] {
    let t = head[target_start..target_end];
    let q = find(t, "?");
    return if q >= 0 { t[q + 1..] } else { "" };
}

fn lower(c: u8) -> u8 { if c >= 'A' && c <= 'Z' { c + 32 } else { c } }

fn same_name(a: const u8[:], b: const u8[:]) -> bool {
    guard a.len == b.len else { return false; }
    for c, i in a { if lower(c) != lower(b[i]) { return false; } }
    return true;
}

// The value of the request's header named name (any case), or "".
fn header(name: const u8[:]) -> const u8[:] {
    var at = find(head, "\r\n");
    while at >= 0 && at + 2 < head.len {
        let start = at + 2;
        let rest = head[start..];
        let end = find(rest, "\r\n");
        let line = if end >= 0 { rest[..end] } else { rest };
        let colon = find(line, ":");
        if colon > 0 && same_name(line[..colon], name) { return trim(line[colon + 1..]); }
        guard end >= 0 else { break; }
        at = start + end;
    }
    return "";
}

// Reads a request from fd into head and body: 0, or the status to answer
// with (400, 413), or -1 when the connection closed first.
fn read_request(fd: i64) -> i64 {
    head.clear();
    body.clear();
    var buf: u8[>..] = [];
    buf.resize(16384, 0);
    var got: u8[>..] = [];
    var end = -1;
    while end < 0 {
        let n = recv(fd, buf);
        guard n > 0 else { return if got.len == 0 { -1 } else { 400 }; }
        got.append(buf[..n]);
        end = find(got, "\r\n\r\n");
        guard got.len <= MAX_HEAD || end >= 0 else { return 413; }
    }
    head.append(got[..end]);
    // the request line: METHOD SP target SP version
    let first = find(head, "\r\n");
    let line = if first >= 0 { head[..first] } else { head[..] };
    let sp1 = find(line, " ");
    guard sp1 > 0 else { return 400; }
    let after = line[sp1 + 1..];
    let sp2 = find(after, " ");
    guard sp2 > 0 else { return 400; }
    method_end = sp1;
    target_start = sp1 + 1;
    target_end = sp1 + 1 + sp2;

    var length = 0;
    let cl = header("Content-Length");
    if cl.len > 0 {
        let v, ok = parse_int(cl);
        guard ok && v >= 0 else { return 400; }
        guard v <= MAX_BODY else { return 413; }
        length = v;
    }
    guard same_name(header("Transfer-Encoding"), "") else { return 411; }
    if same_name(header("Expect"), "100-continue") && got.len - (end + 4) < length {
        send(fd, "HTTP/1.1 100 Continue\r\n\r\n");
    }
    body.append(got[end + 4..min(got.len, end + 4 + length)]);
    while body.len < length {
        let n = recv(fd, buf);
        guard n > 0 else { return 400; }
        body.append(buf[..min(n, length - body.len)]);
    }
    return 0;
}

fn status_text(status: i64) -> const u8[:] {
    return match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        411 => "Length Required",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        _ => "Unknown",
    };
}

fn start_head(out: u8[>..]&, status: i64, content_type: const u8[:]) {
    format(out, "HTTP/1.1 ", status, " ", status_text(status), "\r\n");
    out.append("Server: llama.cpp\r\n");
    out.append("Access-Control-Allow-Origin: *\r\n");
    out.append("Connection: close\r\n");
    if content_type.len > 0 { format(out, "Content-Type: ", content_type, "\r\n"); }
}

// A whole response.
fn respond(fd: i64, status: i64, content_type: const u8[:], data: const u8[:]) -> bool {
    var out: u8[>..] = [];
    start_head(out, status, content_type);
    format(out, "Content-Length: ", data.len, "\r\n\r\n");
    out.append(data);
    return send(fd, out);
}

// The answer to a CORS preflight (OPTIONS).
fn respond_options(fd: i64) -> bool {
    var out: u8[>..] = [];
    start_head(out, 200, "");
    out.append("Access-Control-Allow-Credentials: true\r\n");
    out.append("Access-Control-Allow-Methods: GET, POST\r\n");
    out.append("Access-Control-Allow-Headers: *\r\n");
    out.append("Content-Length: 0\r\n\r\n");
    return send(fd, out);
}

// The head of a stream of server-sent events; the body runs until the
// connection closes.
fn begin_events(fd: i64) -> bool {
    var out: u8[>..] = [];
    start_head(out, 200, "text/event-stream");
    out.append("Cache-Control: no-cache\r\n\r\n");
    return send(fd, out);
}

// One event: "data: <text>\n\n" (or what the caller gives whole).
fn send_event(fd: i64, data: const u8[:]) -> bool {
    var out: u8[>..] = [];
    format(out, "data: ", data, "\n\n");
    return send(fd, out);
}
