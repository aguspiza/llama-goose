// TCP sockets for the server and its clients: listen, accept, connect,
// receive, send and close, on Winsock or POSIX. A socket is an int64_t,
// -1 when there is none.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET lg_sock;
#define LG_BAD INVALID_SOCKET
#define lg_closesocket closesocket
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int lg_sock;
#define LG_BAD (-1)
#define lg_closesocket close
#endif

typedef struct { char * data; int64_t len; } lg_buf;            // a u8[:]
typedef struct { const char * data; int64_t len; } lg_text;     // a const u8[:]

static int lg_started = 0;

static void lg_start(void) {
    if (lg_started) return;
    lg_started = 1;
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#else
    // a client that goes away mid-write must not kill the server
    signal(SIGPIPE, SIG_IGN);
#endif
}

static void lg_cstr(lg_text s, char * buf, size_t size) {
    size_t n = (size_t) s.len < size - 1 ? (size_t) s.len : size - 1;
    memcpy(buf, s.data, n);
    buf[n] = 0;
}

// A socket listening on host:port, or -1.
int64_t net_listen(lg_text host, int64_t port) {
    lg_start();
    char h[256], p[16];
    lg_cstr(host, h, sizeof h);
    snprintf(p, sizeof p, "%d", (int) port);
    struct addrinfo hints, * res = NULL;
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    if (getaddrinfo(h, p, &hints, &res) != 0) return -1;
    lg_sock s = LG_BAD;
    for (struct addrinfo * a = res; a; a = a->ai_next) {
        s = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (s == LG_BAD) continue;
#ifdef _WIN32
        // another server on the same port must make this one fail
        BOOL excl = TRUE;
        setsockopt(s, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (const char *) &excl, sizeof excl);
#else
        int yes = 1;
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof yes);
#endif
        if (bind(s, a->ai_addr, (int) a->ai_addrlen) == 0 && listen(s, 64) == 0) break;
        lg_closesocket(s);
        s = LG_BAD;
    }
    freeaddrinfo(res);
    return s == LG_BAD ? -1 : (int64_t) s;
}

// The port a listening socket got (for port 0), or -1.
int64_t net_port(int64_t fd) {
    struct sockaddr_storage a;
    socklen_t len = sizeof a;
    if (getsockname((lg_sock) fd, (struct sockaddr *) &a, &len) != 0) return -1;
    if (a.ss_family == AF_INET) return ntohs(((struct sockaddr_in *) &a)->sin_port);
    if (a.ss_family == AF_INET6) return ntohs(((struct sockaddr_in6 *) &a)->sin6_port);
    return -1;
}

static void lg_nodelay(lg_sock s) {
    int yes = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char *) &yes, sizeof yes);
}

// The next connection on a listening socket, or -1.
int64_t net_accept(int64_t fd) {
    lg_sock s = accept((lg_sock) fd, NULL, NULL);
    if (s == LG_BAD) return -1;
    lg_nodelay(s);
    return (int64_t) s;
}

// A connection to host:port, or -1.
int64_t net_connect(lg_text host, int64_t port) {
    lg_start();
    char h[256], p[16];
    lg_cstr(host, h, sizeof h);
    snprintf(p, sizeof p, "%d", (int) port);
    struct addrinfo hints, * res = NULL;
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(h, p, &hints, &res) != 0) return -1;
    lg_sock s = LG_BAD;
    for (struct addrinfo * a = res; a; a = a->ai_next) {
        s = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (s == LG_BAD) continue;
        if (connect(s, a->ai_addr, (int) a->ai_addrlen) == 0) break;
        lg_closesocket(s);
        s = LG_BAD;
    }
    freeaddrinfo(res);
    if (s == LG_BAD) return -1;
    lg_nodelay(s);
    return (int64_t) s;
}

// Up to buf.len bytes into buf: how many, 0 at the end, -1 on an error.
int64_t net_recv(int64_t fd, lg_buf buf) {
    int n = recv((lg_sock) fd, buf.data, (int) (buf.len > 1 << 30 ? 1 << 30 : buf.len), 0);
    return n < 0 ? -1 : n;
}

// All of data: 0, or -1 when the connection is gone.
int64_t net_send(int64_t fd, lg_text data) {
    int64_t at = 0;
    while (at < data.len) {
        int64_t left = data.len - at;
#ifdef MSG_NOSIGNAL
        int n = send((lg_sock) fd, data.data + at, (int) (left > 1 << 30 ? 1 << 30 : left), MSG_NOSIGNAL);
#else
        int n = send((lg_sock) fd, data.data + at, (int) (left > 1 << 30 ? 1 << 30 : left), 0);
#endif
        if (n <= 0) return -1;
        at += n;
    }
    return 0;
}

// Whether the peer has closed the connection (or sent more), without
// waiting: 1 when the socket is readable, 0 when not.
int64_t net_readable(int64_t fd) {
    fd_set set;
    FD_ZERO(&set);
    FD_SET((lg_sock) fd, &set);
    struct timeval tv = { 0, 0 };
    return select((int) fd + 1, &set, NULL, NULL, &tv) > 0 ? 1 : 0;
}

typedef struct { const int64_t * data; int64_t len; } lg_fds;    // a const i64[:]

// Waits up to timeout_ms for one of fds to be readable (a listening socket:
// to have a connection): its index, or -1.
int64_t net_wait(lg_fds fds, int64_t timeout_ms) {
    fd_set set;
    FD_ZERO(&set);
    int64_t top = 0;
    for (int64_t i = 0; i < fds.len && i < FD_SETSIZE; i++) {
        FD_SET((lg_sock) fds.data[i], &set);
        if (fds.data[i] > top) top = fds.data[i];
    }
    struct timeval tv = { (long) (timeout_ms / 1000), (long) (timeout_ms % 1000) * 1000 };
    if (select((int) top + 1, &set, NULL, NULL, &tv) <= 0) return -1;
    for (int64_t i = 0; i < fds.len && i < FD_SETSIZE; i++) {
        if (FD_ISSET((lg_sock) fds.data[i], &set)) return i;
    }
    return -1;
}

// A receive timeout, so that a client that stops halfway cannot hold the server.
void net_set_timeout(int64_t fd, int64_t ms) {
#ifdef _WIN32
    DWORD t = (DWORD) ms;
    setsockopt((lg_sock) fd, SOL_SOCKET, SO_RCVTIMEO, (const char *) &t, sizeof t);
#else
    struct timeval tv = { (long) (ms / 1000), (long) (ms % 1000) * 1000 };
    setsockopt((lg_sock) fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
#endif
}

void net_close(int64_t fd) {
    if (fd < 0) return;
#ifdef _WIN32
    shutdown((lg_sock) fd, SD_SEND);
#else
    shutdown((lg_sock) fd, SHUT_WR);
#endif
    lg_closesocket((lg_sock) fd);
}
