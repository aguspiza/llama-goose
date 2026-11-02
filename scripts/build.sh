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

for program in "$@"; do
    # the last command of the dry run compiles and links the program
    link=$("$goose" build --dry-run "$program" 2>/dev/null | grep '^"' | tail -1)
    link=${link/ -O2 / -O3 -march=native }
    # the first time, goose build makes the runtime and the packages' C
    if ! eval "$link" 2>/dev/null; then
        "$goose" build -O0 "$program" > /dev/null
        eval "$link"
    fi
    echo "built ${program%.goose}.exe"
done
