#!/usr/bin/env bash
# Build Goose programs with MinGW gcc at -O1 -march=native.
#   scripts/build.sh tools/simple.goose [more.goose ...]
# This runs goose build's own commands (from --dry-run), with the program's
# C at the level of OPT.
# -O1 compiles the program's C about 4 times faster than -O3 and runs as
# fast: the heavy work is in ggml, whose C gets -O3 from its sidecar and is
# compiled once. -O0 makes the per-token Goose code slow.
# GOOSE  the Goose compiler, v2026.4 (default: ~/goose/goose.exe; the
#        `goose` on PATH may be a different program with the same name)
# OPT    gcc's optimization level for the program's C (default 1)
set -euo pipefail

opt=${OPT:-1}
goose=${GOOSE:-$HOME/goose/goose.exe}
export GOOSE_CC=gcc
# MinGW: gcc spills 32-byte AVX values to the stack with aligned moves, which
# Win64 does not align for, and the program crashes; this makes them unaligned.
avx_fix=$([[ $(uname -s) == MINGW* || $(uname -s) == MSYS* ]] && echo '-Wa,-muse-unaligned-vector-move' || true)

for program in "$@"; do
    # the dry run checks the program, reports its errors and writes its C;
    # it prints the commands that would compile the packages' C and then
    # the program's, and link
    cmds=$("$goose" build --dry-run "$program" | grep '^"')
    link=$(tail -1 <<< "$cmds")
    # the runtime is built by goose build itself: on its first use, a whole
    # goose build (it costs a second compile of the program, at -O0)
    for obj in $(grep -o '[^ "]*/build/runtime/[^ "]*\.o' <<< "$link"); do
        [[ -f $obj ]] || { "$goose" build -O0 "$program" > /dev/null; break; }
    done
    # the packages' objects that are missing or older than their source
    while read -r cmd; do
        src=$(grep -o -- ' -c [^ ]*' <<< "$cmd" | cut -c5- | tr -d '"')
        obj=$(grep -o -- ' -o [^ ]*' <<< "$cmd" | cut -c5- | tr -d '"')
        if [[ ! -f $obj || $src -nt $obj ]]; then eval "$cmd"; fi
    done < <(head -n -1 <<< "$cmds")
    # each package's objects are named by a hash of their source, and the
    # link reads them from the package's link-cc.rsp, which only a whole
    # goose build writes: its objects are put there, its libraries kept
    for rsp in $(grep -o '@[^ "]*link-cc\.rsp' <<< "$link" | cut -c2-); do
        dir=$(dirname "$rsp")
        objs=$(head -n -1 <<< "$cmds" | grep -o -- ' -o [^ ]*' | cut -c5- | tr -d '"' | grep -F "$dir/" || true)
        [[ -n $objs ]] || continue
        { sed 's/.*/"&"/' <<< "$objs"; grep -v '\.o"$' "$rsp" || true; } > "$rsp.new"
        mv "$rsp.new" "$rsp"
    done
    link=${link/ -O2 / -O$opt -march=native $avx_fix }
    eval "$link"
    echo "built ${program%.goose}.exe"
done
