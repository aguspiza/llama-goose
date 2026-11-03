#!/usr/bin/env bash
# Runs llama.cpp's llama-completion and tools/completion with the same
# arguments and the same stdin, and compares what they write to stdout.
#   [IN=turns.txt] scripts/compare-completion.sh args...
# LLAMA_CPP_BIN is the directory of llama.cpp's binaries (default
# ../llama.cpp/build/bin). Fix the seed (-s) for the outputs to agree.
# llama.cpp writes stdout in text mode on Windows, so its \r\n are read as \n.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
bin=${LLAMA_CPP_BIN:-$root/../llama.cpp/build/bin}
in=${IN:-/dev/null}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

"$bin/llama-completion" "$@" < "$in" 2> "$tmp/ref.err" | tr -d '\r' > "$tmp/ref.out"
"$root/tools/completion.exe" "$@" < "$in" > "$tmp/goose.out" 2> "$tmp/goose.err"
if cmp -s "$tmp/ref.out" "$tmp/goose.out"; then
    echo "same output ($(wc -c < "$tmp/ref.out") bytes)"
else
    echo "outputs differ:"
    diff "$tmp/ref.out" "$tmp/goose.out" | head -40
    exit 1
fi
