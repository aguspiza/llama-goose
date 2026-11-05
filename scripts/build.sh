#!/usr/bin/env bash
# Build Goose programs with MinGW gcc at -O1 -march=native.
#   scripts/build.sh tools/simple.goose [more.goose ...]
# This runs goose build's own commands (from --dry-run), with the program's
# C at the level of OPT.
# -O1 compiles the program's C about 4 times faster than -O3 and runs as
# fast: the heavy work is in ggml, whose C gets -O3 from its sidecar and is
# compiled once. -O0 makes the per-token Goose code slow.
# GOOSE  the Goose compiler (default: the one in ~/coding/goose-lang; the
#        `goose` on PATH may be a different program with the same name)
# OPT    gcc's optimization level for the program's C (default 1)
set -euo pipefail

opt=${OPT:-1}
goose=${GOOSE:-$HOME/coding/goose-lang/tools/goose/goose-v2026.3-windows-x64/goose.exe}
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
    link=${link/ -O2 / -O$opt -march=native $avx_fix }
    eval "$link"
    echo "built ${program%.goose}.exe"
done
