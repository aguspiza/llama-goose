#!/usr/bin/env bash
# Regenerate tests/data/autoparser_cases.txt: what llama.cpp's differential
# autoparser finds in each template of models/templates (run after a sync
# with upstream).
#   scripts/gen-autoparser-cases.sh <llama.cpp source dir> <its build dir>
# The build must have static libraries (BUILD_SHARED_LIBS=OFF).
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
src=$(cd "$1" && pwd)
build=$(cd "$2" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

g++ -O1 -std=c++17 -I"$src/src" -I"$src/include" -I"$src/ggml/include" -I"$src/common" -I"$src/vendor" \
    "$root/scripts/extract_autoparser_cases.cpp" \
    "$build/common/libllama-common.a" "$build/common/libllama-common-base.a" \
    "$build/src/libllama.a" "$build/ggml/src/ggml.a" "$build/ggml/src/ggml-cpu.a" "$build/ggml/src/ggml-base.a" \
    -lws2_32 -o "$tmp/extract.exe"
"$tmp/extract.exe" "$root/tests/data/autoparser_cases.txt" "$src"/models/templates/*.jinja
cp "$src"/models/templates/*.jinja "$root/tests/data/templates/"
echo "wrote tests/data/autoparser_cases.txt"
