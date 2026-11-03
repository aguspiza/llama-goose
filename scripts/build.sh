#!/usr/bin/env bash
# Build Goose programs with MinGW gcc at -O3 -march=native.
#   scripts/build.sh tools/simple.goose [more.goose ...]
# goose build compiles a program's C at -O2 at most, and gcc vectorizes at
# -O3, so this runs goose build's own commands (from --dry-run) with -O3.
# The packages' C (ggml) gets -O3 from its sidecar.
# GOOSE  the Goose compiler (default: the one in ~/coding/goose-lang; the
#        `goose` on PATH may be a different program with the same name)
set -euo pipefail

goose=${GOOSE:-$HOME/coding/goose-lang/tools/goose/goose-v2026.3-windows-x64/goose.exe}
export GOOSE_CC=gcc
# MinGW: gcc spills 32-byte AVX values to the stack with aligned moves, which
# Win64 does not align for, and the program crashes; this makes them unaligned.
avx_fix=$([[ $(uname -s) == MINGW* || $(uname -s) == MSYS* ]] && echo '-Wa,-muse-unaligned-vector-move' || true)

for program in "$@"; do
    # goose build first: it reports errors, builds the runtime and recompiles
    # the packages' C that changed (a quick -O0 build of the program)
    "$goose" build -O0 "$program" > /dev/null
    # then the last command of the dry run (compile and link) at -O3
    link=$("$goose" build --dry-run "$program" 2>/dev/null | grep '^"' | tail -1)
    link=${link/ -O2 / -O3 -march=native $avx_fix }
    eval "$link"
    echo "built ${program%.goose}.exe"
done
