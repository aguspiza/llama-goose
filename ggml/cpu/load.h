// The operating system's side of loading a model (load.c): a file mapped
// into memory, memory locked in RAM, and reads that bypass the page cache.
// port-of: src/llama-mmap.h
#pragma once
#include <stdbool.h>
#include <stddef.h>

// The whole file mapped read-only (llama_mmap), its pages read ahead when
// prefetch, but for those of [lazy_first, lazy_last), which are read as
// they are used (a lazy tensor; empty when the two are equal); NULL if it
// cannot be mapped, with a message.
void * lg_map_file(const char * path, size_t * size, bool prefetch, size_t lazy_first, size_t lazy_last);
// llama_prefetch: the pages of n ranges of memory read ahead.
void lg_prefetch(const void ** addr, const size_t * size, size_t n);
// Gives back the pages of [first, last) of a mapping (POSIX; Windows keeps them).
void lg_unmap_fragment(void * addr, size_t size, size_t first, size_t last);

// Memory locked from addr, grown a page at a time (llama_mlock); after a
// failure, with a warning, it locks no more.
typedef struct { void * addr; size_t size; bool failed; } lg_mlock;
void lg_mlock_grow(lg_mlock * m, size_t target);

// A file read with direct I/O (llama_file's): reads of `alignment`-aligned
// offsets, lengths and addresses. NULL, with a warning, when the system
// refuses it: the caller then reads it buffered.
typedef struct lg_dio lg_dio;
lg_dio * lg_dio_open(const char * path, size_t * alignment);
bool lg_dio_read(lg_dio * f, void * dst, size_t offset, size_t len);
void lg_dio_close(lg_dio * f);
