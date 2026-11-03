#!/usr/bin/env bash
# Regenerate tests/data/schema_cases.txt from llama.cpp's
# test-json-schema-to-grammar.cpp and the grammars its json_schema_to_grammar
# makes of the cases (run after a sync with upstream).
#   scripts/gen-schema-cases.sh <llama.cpp source dir> <its build dir>
# The build must have static libraries (BUILD_SHARED_LIBS=OFF).
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
src=$1
build=$2
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# the test cases: trim, TestCase and test_all, up to main
sed -n '/^static std::string trim/,/^int main() {/p' "$src/tests/test-json-schema-to-grammar.cpp" | sed '$d' > "$tmp/cases.inc"
cp "$root/scripts/extract_schema_cases.cpp" "$tmp/"
g++ -O1 -std=c++17 -I"$tmp" -I"$src/src" -I"$src/include" -I"$src/ggml/include" -I"$src/common" -I"$src/vendor" \
    "$tmp/extract_schema_cases.cpp" \
    "$build/common/libllama-common.a" "$build/common/libllama-common-base.a" \
    "$build/src/libllama.a" "$build/ggml/src/ggml.a" "$build/ggml/src/ggml-cpu.a" "$build/ggml/src/ggml-base.a" \
    -o "$tmp/extract.exe"
"$tmp/extract.exe" "$root/tests/data/schema_cases.txt"
echo "wrote tests/data/schema_cases.txt"
