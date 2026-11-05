// tools/net: a request sent to a socket of this program, read back, and
// the response read by the client side.
import std;
import net from "../tools/net";

var failed = 0;

fn check(name: const u8[:], ok: bool) {
    if !ok { failed++; print("FAIL ", name); }
}

fn main() {
    let srv = net::listen("127.0.0.1", 0);
    check("listen", srv >= 0);
    let port = net::port_of(srv);
    check("port", port > 0);

    let cli = net::connect("127.0.0.1", port);
    check("connect", cli >= 0);
    check("send", net::send(cli, "POST /v1/chat/completions?x=1 HTTP/1.1\r\nHost: a\r\ncontent-length: 11\r\nX-Two:  b c \r\n\r\nhello world"));
    let con = net::accept(srv);
    check("accept", con >= 0);
    check("read", net::read_request(con) == 0);
    check("method", net::method() == "POST");
    check("path", net::path() == "/v1/chat/completions");
    check("query", net::query() == "x=1");
    check("header any case", net::header("Content-Length") == "11");
    check("header trimmed", net::header("x-two") == "b c");
    check("no header", net::header("Accept") == "");
    check("body", net::body == "hello world");

    check("respond", net::respond(con, 200, "application/json; charset=utf-8", "{\"a\":1}"));
    net::close(con);
    var got: u8[>..] = [];
    var buf: u8[>..] = [];
    buf.resize(4096, 0);
    loop {
        let n = net::recv(cli, buf);
        guard n > 0 else { break; }
        got.append(buf[..n]);
    }
    net::close(cli);
    check("status line", starts_with(got, "HTTP/1.1 200 OK\r\n"));
    check("length", find(got, "Content-Length: 7\r\n") >= 0);
    check("response body", ends_with(got, "\r\n\r\n{\"a\":1}"));

    // a body that comes after the head, in its own send
    let cli2 = net::connect("127.0.0.1", port);
    check("send head", net::send(cli2, "GET /health HTTP/1.1\r\nContent-Length: 3\r\n\r\n"));
    check("send body", net::send(cli2, "abc"));
    let con2 = net::accept(srv);
    check("read 2", net::read_request(con2) == 0);
    check("path 2", net::path() == "/health" && net::query() == "");
    check("body 2", net::body == "abc");
    net::close(con2);
    net::close(cli2);

    // a connection closed before a request
    let cli3 = net::connect("127.0.0.1", port);
    net::close(cli3);
    let con3 = net::accept(srv);
    check("closed", net::read_request(con3) == -1);
    net::close(con3);
    net::close(srv);

    // server base URLs
    {
        let h, p, pre, ok = net::parse_url("http://127.0.0.1:8080");
        check("url", ok && h == "127.0.0.1" && p == 8080 && pre == "");
        let h2, p2, pre2, ok2 = net::parse_url("http://example.local/api/v1/");
        check("url prefix", ok2 && h2 == "example.local" && p2 == 80 && pre2 == "/api/v1");
        let h3, p3, pre3, ok3 = net::parse_url("https://x");
        check("url https refused", !ok3);
    }

    // a chunked body, received in arbitrary pieces
    {
        net::close_request();
        net::cchunked = true;
        let wire = "6\r\ndata: \r\n9;ext=1\r\n{\"a\":1}\n\n\r\n0\r\n\r\n";
        for c in wire {
            net::craw.push(c);
            net::pump();
        }
        check("chunked body", net::cbody == "data: {\"a\":1}\n\n");
        check("chunked end", net::cdone);
        var payload: u8[>..] = [];
        check("event", net::next_event(payload, 0) == 1 && payload == "{\"a\":1}");
        check("events end", net::next_event(payload, 0) == -1);
        net::close_request();
    }

    if failed > 0 { print(failed, " failed"); exit(1); }
    print("net: all passed");
}
