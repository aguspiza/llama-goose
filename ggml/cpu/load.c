// The operating system's side of loading a model: see load.h.
// port-of: src/llama-mmap.cpp (llama_mmap, llama_mlock, llama_file's direct I/O)
#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0602
#endif
#include <windows.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "load.h"

#ifdef _WIN32
// llama_format_win_err: the system's message, with its line break, in
// UTF-8 (the code page of llama.cpp's programs, whose FormatMessageA gives it)
static void lg_win_err(char * out, size_t n, DWORD err) {
    LPWSTR buf = NULL;
    const DWORD size = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                      NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPWSTR) &buf, 0, NULL);
    if (!size) {
        snprintf(out, n, "FormatMessageA failed");
        return;
    }
    const int len = WideCharToMultiByte(CP_UTF8, 0, buf, (int) size, out, (int) n - 1, NULL, NULL);
    out[len > 0 ? len : 0] = 0;
    LocalFree(buf);
}
#endif

//
// the mapping
//

#ifdef _WIN32
typedef BOOL (WINAPI * lg_prefetch_t)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);

static lg_prefetch_t lg_prefetch_fn(void) {
    return (lg_prefetch_t) (void *) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");
}

void * lg_map_file(const char * path, size_t * size, bool prefetch, size_t lazy_first, size_t lazy_last) {
    char msg[512];
    HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) {
        lg_win_err(msg, sizeof msg, GetLastError());
        fprintf(stderr, "failed to open %s: %s\n", path, msg);
        return NULL;
    }
    LARGE_INTEGER li;
    GetFileSizeEx(file, &li);
    *size = (size_t) li.QuadPart;
    HANDLE mapping = CreateFileMappingA(file, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(file);
    if (mapping == NULL) {
        lg_win_err(msg, sizeof msg, GetLastError());
        fprintf(stderr, "CreateFileMappingA failed: %s\n", msg);
        return NULL;
    }
    void * addr = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    const DWORD error = GetLastError();
    CloseHandle(mapping);    // the view keeps the mapping
    if (addr == NULL) {
        lg_win_err(msg, sizeof msg, error);
        fprintf(stderr, "MapViewOfFile failed: %s\n", msg);
        return NULL;
    }
    lg_prefetch_t fn = prefetch ? lg_prefetch_fn() : NULL;
    if (fn) {
        // the file but the lazy range (ranges_complement)
        WIN32_MEMORY_RANGE_ENTRY ranges[2];
        ULONG n = 0;
        if (lazy_first == lazy_last) {
            ranges[n++] = (WIN32_MEMORY_RANGE_ENTRY) { addr, (SIZE_T) *size };
        } else {
            if (lazy_first > 0) { ranges[n++] = (WIN32_MEMORY_RANGE_ENTRY) { addr, (SIZE_T) lazy_first }; }
            if (lazy_last < *size) { ranges[n++] = (WIN32_MEMORY_RANGE_ENTRY) { (char *) addr + lazy_last, (SIZE_T) (*size - lazy_last) }; }
        }
        if (n > 0 && !fn(GetCurrentProcess(), n, ranges, 0)) {
            lg_win_err(msg, sizeof msg, GetLastError());
            fprintf(stderr, "warning: PrefetchVirtualMemory failed: %s\n", msg);
        }
    }
    return addr;
}

void lg_unmap_fragment(void * addr, size_t size, size_t first, size_t last) {
    (void) addr; (void) size; (void) first; (void) last;
}
#else
// page-aligned posix_madvise over [beg, end), clamped to the file
static void lg_advise(void * addr, size_t size, size_t beg, size_t end, int advice, const char * name) {
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    beg &= ~(page - 1);
    end = (end + page - 1) & ~(page - 1);
    if (end > size) { end = size; }
    if (beg >= end) {
        return;
    }
    if (posix_madvise((char *) addr + beg, end - beg, advice)) {
        fprintf(stderr, "warning: posix_madvise(.., %s) failed: %s\n", name, strerror(errno));
    }
}

void * lg_map_file(const char * path, size_t * size, bool prefetch, size_t lazy_first, size_t lazy_last) {
    const int fd = open(path, O_RDONLY);
    if (fd == -1) {
        fprintf(stderr, "failed to open %s: %s\n", path, strerror(errno));
        return NULL;
    }
    struct stat st;
    fstat(fd, &st);
    *size = (size_t) st.st_size;
    const bool lazy = lazy_first != lazy_last;
    int flags = MAP_SHARED;
#ifdef __linux__
    if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)) {
        fprintf(stderr, "warning: posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: %s\n", strerror(errno));
    }
    // MAP_POPULATE would fault in the lazy range too
    if (prefetch && !lazy) { flags |= MAP_POPULATE; }
#endif
    void * addr = mmap(NULL, *size, PROT_READ, flags, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) {
        fprintf(stderr, "mmap failed: %s\n", strerror(errno));
        return NULL;
    }
    if (prefetch) {
        if (!lazy) {
            lg_advise(addr, *size, 0, *size, POSIX_MADV_WILLNEED, "POSIX_MADV_WILLNEED");
        } else {
            lg_advise(addr, *size, 0, lazy_first, POSIX_MADV_WILLNEED, "POSIX_MADV_WILLNEED");
            lg_advise(addr, *size, lazy_last, *size, POSIX_MADV_WILLNEED, "POSIX_MADV_WILLNEED");
        }
    }
    if (lazy) {
        lg_advise(addr, *size, lazy_first, lazy_last, POSIX_MADV_RANDOM, "POSIX_MADV_RANDOM");
    }
    return addr;
}

// The whole pages inside [first, last).
void lg_unmap_fragment(void * addr, size_t size, size_t first, size_t last) {
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    (void) size;
    first = (first + page - 1) & ~(page - 1);
    last &= ~(page - 1);
    if (last <= first) {
        return;
    }
    if (munmap((char *) addr + first, last - first)) {
        fprintf(stderr, "warning: munmap failed: %s\n", strerror(errno));
    }
}
#endif

// llama_prefetch: the ranges sorted, widened to whole pages and the
// touching ones joined, then read ahead in one call
typedef struct { uintptr_t first, last; } lg_span;

static int lg_span_cmp(const void * a, const void * b) {
    const lg_span * x = (const lg_span *) a;
    const lg_span * y = (const lg_span *) b;
    return x->first < y->first ? -1 : x->first > y->first;
}

void lg_prefetch(const void ** addr, const size_t * size, size_t n) {
#ifdef _WIN32
    lg_prefetch_t fn = lg_prefetch_fn();
    if (!fn) {
        return;
    }
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    const uintptr_t page = (uintptr_t) info.dwPageSize;
#else
    const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);
#endif
    lg_span * spans = (lg_span *) malloc((n > 0 ? n : 1) * sizeof(lg_span));
    size_t m = 0;
    for (size_t i = 0; i < n; i++) {
        if (!addr[i] || size[i] == 0) {
            continue;
        }
        const uintptr_t p = (uintptr_t) addr[i];
        spans[m++] = (lg_span) { p / page * page, (p + size[i] + page - 1) / page * page };
    }
    qsort(spans, m, sizeof *spans, lg_span_cmp);
    size_t k = 0;
    for (size_t i = 0; i < m; i++) {
        if (k > 0 && spans[i].first <= spans[k - 1].last) {
            if (spans[i].last > spans[k - 1].last) { spans[k - 1].last = spans[i].last; }
        } else {
            spans[k++] = spans[i];
        }
    }
#ifdef _WIN32
    WIN32_MEMORY_RANGE_ENTRY * entries = (WIN32_MEMORY_RANGE_ENTRY *) malloc((k > 0 ? k : 1) * sizeof *entries);
    for (size_t i = 0; i < k; i++) {
        entries[i].VirtualAddress = (PVOID) spans[i].first;
        entries[i].NumberOfBytes = (SIZE_T) (spans[i].last - spans[i].first);
    }
    if (k > 0 && !fn(GetCurrentProcess(), (ULONG_PTR) k, entries, 0)) {
        char msg[512];
        lg_win_err(msg, sizeof msg, GetLastError());
        fprintf(stderr, "llama_prefetch: PrefetchVirtualMemory failed: %s\n", msg);
    }
    free(entries);
#else
    for (size_t i = 0; i < k; i++) {
        if (madvise((void *) spans[i].first, spans[i].last - spans[i].first, MADV_WILLNEED) != 0) {
            fprintf(stderr, "llama_prefetch: madvise(MADV_WILLNEED) failed: %s\n", strerror(errno));
        }
    }
#endif
    free(spans);
}

//
// locking
//

#ifdef _WIN32
static size_t lg_lock_granularity(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (size_t) si.dwPageSize;
}

static bool lg_raw_lock(void * ptr, size_t len, size_t locked) {
    char msg[512];
    for (int tries = 1; ; tries++) {
        if (VirtualLock(ptr, len)) {
            return true;
        }
        if (tries == 2) {
            lg_win_err(msg, sizeof msg, GetLastError());
            fprintf(stderr, "warning: failed to VirtualLock %zu-byte buffer (after previously locking %zu bytes): %s\n",
                    len, locked, msg);
            return false;
        }
        // a working set big enough for the pages, and try again
        SIZE_T min_ws, max_ws;
        if (!GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws, &max_ws)) {
            lg_win_err(msg, sizeof msg, GetLastError());
            fprintf(stderr, "warning: GetProcessWorkingSetSize failed: %s\n", msg);
            return false;
        }
        const size_t increment = len + 1048576;
        if (!SetProcessWorkingSetSize(GetCurrentProcess(), min_ws + increment, max_ws + increment)) {
            lg_win_err(msg, sizeof msg, GetLastError());
            fprintf(stderr, "warning: SetProcessWorkingSetSize failed: %s\n", msg);
            return false;
        }
    }
}
#else
static size_t lg_lock_granularity(void) { return (size_t) sysconf(_SC_PAGESIZE); }

static bool lg_raw_lock(void * ptr, size_t len, size_t locked) {
    if (!mlock(ptr, len)) {
        return true;
    }
    const int err = errno;
    bool suggest = err == ENOMEM;
    struct rlimit lim;
    if (suggest && getrlimit(RLIMIT_MEMLOCK, &lim)) {
        suggest = false;
    }
    if (suggest && (uint64_t) lim.rlim_max > (uint64_t) lim.rlim_cur + len) {
        suggest = false;
    }
    fprintf(stderr, "warning: failed to mlock %zu-byte buffer (after previously locking %zu bytes): %s\n%s",
            len, locked, strerror(err), suggest ? "Try increasing RLIMIT_MEMLOCK ('ulimit -l' as root).\n" : "");
    return false;
}
#endif

void lg_mlock_grow(lg_mlock * m, size_t target) {
    if (m->failed) {
        return;
    }
    const size_t g = lg_lock_granularity();
    target = (target + g - 1) & ~(g - 1);
    if (target > m->size) {
        if (lg_raw_lock((uint8_t *) m->addr + m->size, target - m->size, m->size)) {
            m->size = target;
        } else {
            m->failed = true;
        }
    }
}

//
// direct I/O
//

#ifdef _WIN32
struct lg_dio { HANDLE h; size_t alignment; };

lg_dio * lg_dio_open(const char * path, size_t * alignment) {
    char msg[512];
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
                           OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, NULL);
    DWORD error = ERROR_SUCCESS;
    size_t align = 1;
    if (h == INVALID_HANDLE_VALUE) {
        error = GetLastError();
    } else {
        FILE_STORAGE_INFO info;
        if (!GetFileInformationByHandleEx(h, FileStorageInfo, &info, (DWORD) sizeof info)) {
            error = GetLastError();
        } else {
            align = info.LogicalBytesPerSector;
            if (info.PhysicalBytesPerSectorForAtomicity > align) { align = info.PhysicalBytesPerSectorForAtomicity; }
            if (info.PhysicalBytesPerSectorForPerformance > align) { align = info.PhysicalBytesPerSectorForPerformance; }
            if (info.FileSystemEffectivePhysicalBytesPerSectorForAtomicity > align) { align = info.FileSystemEffectivePhysicalBytesPerSectorForAtomicity; }
            if (align <= 1 || (align & (align - 1)) != 0) { error = ERROR_NOT_SUPPORTED; }
        }
    }
    if (error != ERROR_SUCCESS) {
        if (h != INVALID_HANDLE_VALUE) { CloseHandle(h); }
        lg_win_err(msg, sizeof msg, error);
        // llama.cpp trims the message's line break here
        for (size_t n = strlen(msg); n > 0 && (msg[n - 1] == '\r' || msg[n - 1] == '\n'); n--) { msg[n - 1] = 0; }
        fprintf(stderr, "Failed to open file '%s' with unbuffered I/O: %s. Falling back to buffered I/O\n", path, msg);
        return NULL;
    }
    fprintf(stderr, "init_direct_io: file '%s' opened with unbuffered I/O (alignment: %zu bytes)\n", path, align);
    lg_dio * f = (lg_dio *) malloc(sizeof *f);
    f->h = h;
    f->alignment = align;
    *alignment = align;
    return f;
}

bool lg_dio_read(lg_dio * f, void * dst, size_t offset, size_t len) {
    LARGE_INTEGER li;
    li.QuadPart = (LONGLONG) offset;
    if (!SetFilePointerEx(f->h, li, NULL, FILE_BEGIN)) {
        return false;
    }
    const size_t max_read = (size_t) MAXDWORD & ~(f->alignment - 1);
    size_t done = 0;
    while (done < len) {
        const DWORD want = (DWORD) (len - done < max_read ? len - done : max_read);
        DWORD got = 0;
        if (!ReadFile(f->h, (char *) dst + done, want, &got, NULL)) {
            return false;
        }
        done += got;
        if (got < want) {
            break;      // the end of the file, inside the last aligned block
        }
    }
    return true;
}

void lg_dio_close(lg_dio * f) {
    CloseHandle(f->h);
    free(f);
}
#else
struct lg_dio { int fd; size_t alignment; };

lg_dio * lg_dio_open(const char * path, size_t * alignment) {
#ifdef __linux__
    const int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd != -1) {
        struct stat st;
        fstat(fd, &st);
        lg_dio * f = (lg_dio *) malloc(sizeof *f);
        f->fd = fd;
        f->alignment = (size_t) st.st_blksize;
        *alignment = f->alignment;
        fprintf(stderr, "impl: file '%s' opened with direct I/O (alignment: %zu bytes)\n", path, f->alignment);
        return f;
    }
    fprintf(stderr, "Failed to open file '%s' with unbuffered I/O: %s. Falling back to buffered I/O\n", path, strerror(errno));
#else
    (void) path; (void) alignment;
#endif
    return NULL;
}

bool lg_dio_read(lg_dio * f, void * dst, size_t offset, size_t len) {
    size_t done = 0;
    while (done < len) {
        const ssize_t got = pread(f->fd, (char *) dst + done, len - done, (off_t) (offset + done));
        if (got < 0) {
            if (errno == EINTR) { continue; }
            return false;
        }
        done += (size_t) got;
        if ((size_t) got < len - (done - (size_t) got)) {
            break;      // the end of the file, inside the last aligned block
        }
    }
    return true;
}

void lg_dio_close(lg_dio * f) {
    close(f->fd);
    free(f);
}
#endif
