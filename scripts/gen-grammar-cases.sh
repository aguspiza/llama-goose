#!/usr/bin/env bash
# Regenerate tests/data/grammar_cases.txt from llama.cpp's test-grammar-parser.cpp
# and test-grammar-integration.cpp and what its grammar code makes of the
# cases (run after a sync with upstream).
#   scripts/gen-grammar-cases.sh <llama.cpp source dir> <its build dir>
# The build must have static libraries (BUILD_SHARED_LIBS=OFF).
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
src=$1
build=$2
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# the parser cases: main's body, without its return
sed -n '/^int main()/,/^}/p' "$src/tests/test-grammar-parser.cpp" | sed '1,2d;$d' | grep -v '^    return 0;' > "$tmp/parser.inc"
# the integration cases: the test functions, test_simple_grammar up to main
sed -n '/^static void test_simple_grammar() {/,/^int main() {/p' "$src/tests/test-grammar-integration.cpp" | sed '$d' > "$tmp/integration.inc"
cp "$root/scripts/extract_grammar_cases.cpp" "$tmp/"
g++ -O1 -std=c++17 -I"$tmp" -I"$src/src" -I"$src/include" -I"$src/ggml/include" -I"$src/common" -I"$src/vendor" \
    "$tmp/extract_grammar_cases.cpp" \
    "$build/common/libllama-common.a" "$build/common/libllama-common-base.a" \
    "$build/src/libllama.a" "$build/ggml/src/ggml.a" "$build/ggml/src/ggml-cpu.a" "$build/ggml/src/ggml-base.a" \
    -o "$tmp/extract.exe"
"$tmp/extract.exe" "$root/tests/data/grammar_cases.txt"
echo "wrote tests/data/grammar_cases.txt"
