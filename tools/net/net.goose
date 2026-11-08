// HTTP/1.1 over TCP, as much as llama-server's clients need: one request
// per connection (each response closes it), bodies by Content-Length, and
// server-sent events; and the request parsing tools/server uses, its HTTP
// being goose-http's. The sockets are net.c's.
// port-of: tools/server/server-http.cpp
import std;
namespace net;

extern fn listen(host: const u8[:], port: i64) -> i64 = "lg_net_listen";
extern fn port_of(fd: i64) -> i64 = "lg_net_port";
extern fn accept(fd: i64) -> i64 = "lg_net_accept";
extern fn connect(host: const u8[:], port: i64) -> i64 = "lg_net_connect";
extern fn recv(fd: i64, buf: u8[:]) -> i64 = "lg_net_recv";
extern fn c_send(fd: i64, data: const u8[:]) -> i64 = "lg_net_send";
extern fn readable(fd: i64) -> i64 = "lg_net_readable";
// The index of the first of fds that is readable within timeout_ms, or -1.
extern fn wait(fds: const i64[:], timeout_ms: i64) -> i64 = "lg_net_wait";
extern fn close(fd: i64) = "lg_net_close";
// Whether the server is ready, for every thread: set once its model is loaded.
extern fn set_ready() = "lg_net_set_ready";
extern fn ready() -> i64 = "lg_net_ready";

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
    let st = request_line();
    guard st == 0 else { return st; }

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

// Parses the request line of head: METHOD SP target SP version. 0, or 400.
fn request_line() -> i64 {
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
    return 0;
}

// Takes a whole request, its head and its body, from raw into head and
// body: 0, or 400.
fn take_request(raw: const u8[:]) -> i64 {
    head.clear();
    body.clear();
    let end = find(raw, "\r\n\r\n");
    guard end >= 0 else { return 400; }
    head.append(raw[..end]);
    body.append(raw[end + 4..]);
    return request_line();
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

//
// the client side: requests to a server, and its server-sent events
//

// A server's base URL, http://host[:port][/prefix]: its host, port and path
// prefix (no trailing slash); false if it is not one.
fn parse_url(url: const u8[:]) -> u8[>..], i64, u8[>..], bool {
    var host: u8[>..] = [];
    var prefix: u8[>..] = [];
    guard starts_with(url, "http://") else { return host, 0, prefix, false; }
    var rest = url[7..];
    let slash = find(rest, "/");
    let authority = if slash >= 0 { rest[..slash] } else { rest };
    if slash >= 0 { prefix.append(rest[slash..]); }
    while prefix.len > 0 && prefix[prefix.len - 1] == '/' { prefix.pop(); }
    var port = 80;
    let colon = find(authority, ":");
    if colon >= 0 {
        host.append(authority[..colon]);
        let p, ok = parse_int(authority[colon + 1..]);
        guard ok && p > 0 && p < 65536 else { return host, 0, prefix, false; }
        port = p;
    } else {
        host.append(authority);
    }
    return host, port, prefix, host.len > 0;
}

// The connection of the request in progress, and what came of it.
var cfd = -1;
var craw: u8[>..] = [];         // received, not yet decoded
var cbody: u8[>..] = [];        // the body so far, decoded
var cchunked = false;
var cchunk_left = 0;            // of the current chunk; -1: a size line is next
var cdone = false;              // the body is complete
var clength = -1;               // Content-Length, -1 without one

// Sends a request and reads its head: the status, or -1 when the server
// cannot be reached or answers no HTTP.
fn open_request(host: const u8[:], port: i64, method: const u8[:], path: const u8[:], body: const u8[:], accept: const u8[:]) -> i64 {
    close_request();
    cfd = connect(host, port);
    guard cfd >= 0 else { return -1; }
    var req: u8[>..] = [];
    format(req, method, " ", path, " HTTP/1.1\r\nHost: ", host, ":", port, "\r\nConnection: close\r\nAccept: ", accept, "\r\n");
    if method == "POST" { format(req, "Content-Type: application/json\r\nContent-Length: ", body.len, "\r\n"); }
    req.append("\r\n");
    req.append(body);
    guard send(cfd, req) else { close_request(); return -1; }
    var buf: u8[>..] = [];
    buf.resize(16384, 0);
    var end = -1;
    while end < 0 {
        let n = recv(cfd, buf);
        guard n > 0 else { close_request(); return -1; }
        craw.append(buf[..n]);
        end = find(craw, "\r\n\r\n");
    }
    // the head: the status line and the fields
    var headtext: u8[>..] = [];
    headtext.append(craw[..end]);
    var rest: u8[>..] = [];
    rest.append(craw[end + 4..]);
    craw.clear();
    craw.append(rest);
    let line_end = find(headtext, "\r\n");
    let status_line = if line_end >= 0 { headtext[..line_end] } else { headtext[..] };
    let sp = find(status_line, " ");
    guard starts_with(status_line, "HTTP/") && sp > 0 else { close_request(); return -1; }
    let code, ok = parse_int(status_line[sp + 1..min(sp + 4, status_line.len)]);
    guard ok else { close_request(); return -1; }
    // the fields that frame the body
    var at = if line_end >= 0 { line_end + 2 } else { headtext.len };
    while at < headtext.len {
        let e = find(headtext[at..], "\r\n");
        let field = if e >= 0 { headtext[at..at + e] } else { headtext[at..] };
        let colon = find(field, ":");
        if colon > 0 {
            let name = field[..colon];
            let value = trim(field[colon + 1..]);
            if same_name(name, "Content-Length") { let v, vok = parse_int(value); if vok { clength = v; } }
            if same_name(name, "Transfer-Encoding") && find(value, "chunked") >= 0 { cchunked = true; }
        }
        guard e >= 0 else { break; }
        at += e + 2;
    }
    if method == "HEAD" || code == 204 || code == 304 || clength == 0 { cdone = true; }
    pump();
    return code;
}

// Moves what was received into the body, undoing the chunked coding.
fn pump() {
    if !cchunked {
        cbody.append(craw);
        craw.clear();
        if clength >= 0 && cbody.len >= clength { cdone = true; }
        return;
    }
    var at = 0;
    loop {
        if cdone { break; }
        if cchunk_left > 0 {
            let n = min(cchunk_left, craw.len - at);
            guard n > 0 else { break; }
            cbody.append(craw[at..at + n]);
            at += n;
            cchunk_left -= n;
            if cchunk_left == 0 { cchunk_left = -2; }      // its CRLF next
            continue;
        }
        let e = find(craw[at..], "\r\n");
        guard e >= 0 else { break; }
        if cchunk_left == -2 { at += e + 2; cchunk_left = 0; continue; }
        // a size line, maybe with extensions
        var size_text = craw[at..at + e];
        let semi = find(size_text, ";");
        if semi >= 0 { size_text = size_text[..semi]; }
        let size, ok = parse_int(trim(size_text), 16);
        at += e + 2;
        guard ok else { cdone = true; break; }
        if size == 0 { cdone = true; break; }
        cchunk_left = size;
    }
    var rest: u8[>..] = [];
    rest.append(craw[at..]);
    craw.clear();
    craw.append(rest);
}

// Reads more of the body, waiting at most timeout_ms: false once it is complete
// or the connection has ended.
fn read_more(timeout_ms: i64) -> bool {
    guard cfd >= 0 && !cdone else { return false; }
    if wait([cfd], timeout_ms) < 0 { return true; }
    var buf: u8[>..] = [];
    buf.resize(16384, 0);
    let n = recv(cfd, buf);
    guard n > 0 else { cdone = true; return false; }
    craw.append(buf[..n]);
    pump();
    return true;
}

fn close_request() {
    if cfd >= 0 { close(cfd); }
    cfd = -1;
    craw.clear();
    cbody.clear();
    cchunked = false;
    cchunk_left = 0;
    cdone = false;
    clength = -1;
}

// A whole request: the status (-1 without an answer), and the body into out.
fn request(host: const u8[:], port: i64, method: const u8[:], path: const u8[:], body: const u8[:], out: u8[>..]&) -> i64 {
    let status = open_request(host, port, method, path, body, "application/json");
    if status >= 0 {
        while read_more(60000) {}
        out.append(cbody);
    }
    close_request();
    return status;
}

// The next "data: " line of the events of the request opened, into payload:
// 1 for one, 0 when none came within timeout_ms, -1 at the end. The lines
// that are not data are skipped.
fn next_event(payload: u8[>..]&, timeout_ms: i64) -> i64 {
    loop {
        let nl = find(cbody, "\n");
        if nl >= 0 {
            var line: u8[>..] = [];
            line.append(cbody[..nl]);
            var rest: u8[>..] = [];
            rest.append(cbody[nl + 1..]);
            cbody.clear();
            cbody.append(rest);
            if line.len > 0 && line[line.len - 1] == '\r' { line.pop(); }
            guard starts_with(line, "data: ") else { continue; }
            payload.clear();
            payload.append(line[6..]);
            return 1;
        }
        guard !cdone && cfd >= 0 else { return -1; }
        let before = cbody.len;
        if !read_more(timeout_ms) { if cbody.len == before && find(cbody, "\n") < 0 { return -1; } continue; }
        if cbody.len == before { return 0; }
    }
}
