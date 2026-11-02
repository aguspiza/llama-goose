#!/usr/bin/env bash
# Build Goose programs with MinGW gcc (the only C compiler ggml/ggml.build supports).
#   scripts/build.sh tests/ggml_smoke.goose [more.goose ...]
# GOOSE  the Goose compiler (default: the one in ~/coding/goose-lang; the
#        `goose` on PATH may be a different program with the same name)
set -euo pipefail

goose=${GOOSE:-$HOME/coding/goose-lang/tools/goose/goose-v2026.3-windows-x64/goose.exe}
for program in "$@"; do
    GOOSE_CC=gcc "$goose" build "$program"
done
