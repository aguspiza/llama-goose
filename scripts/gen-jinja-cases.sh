#!/usr/bin/env bash
# Regenerate tests/data/jinja_cases.txt from llama.cpp's test-jinja.cpp and
# what its Jinja engine makes of the cases (run after a sync with upstream).
#   scripts/gen-jinja-cases.sh <llama.cpp source dir> <its build dir>
# The build must have static libraries (BUILD_SHARED_LIBS=OFF).
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
src=$1
build=$2
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# the test functions of the template cases, test_whitespace_control up to
# test_hasher; the lexer checks go through the extractor's recorder
sed -n '/^static void test_whitespace_control(testing & t) {/,/^static void test_hasher(testing & t) {/p' \
    "$src/tests/test-jinja.cpp" | sed '$d' | sed 's/jinja::lexer()\.tokenize/rec_lexer_tokenize/' > "$tmp/cases.inc"
cp "$root/scripts/extract_jinja_cases.cpp" "$tmp/"
g++ -O1 -std=c++17 -I"$tmp" -I"$src/common" -I"$src/include" -I"$src/ggml/include" -I"$src/vendor" \
    "$tmp/extract_jinja_cases.cpp" \
    "$build/common/libllama-common.a" "$build/common/libllama-common-base.a" \
    "$build/src/libllama.a" "$build/ggml/src/ggml.a" "$build/ggml/src/ggml-cpu.a" "$build/ggml/src/ggml-base.a" \
    -o "$tmp/extract.exe"
"$tmp/extract.exe" "$root/tests/data/jinja_cases.txt"
echo "wrote tests/data/jinja_cases.txt"
