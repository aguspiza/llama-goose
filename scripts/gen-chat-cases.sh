#!/usr/bin/env bash
# Regenerate tests/data/chat_cases.txt from llama.cpp's test-chat-template.cpp
# and what its libllama and libcommon make of the cases (run after a sync with
# upstream).
#   scripts/gen-chat-cases.sh <llama.cpp source dir> <its build dir>
# The build must have static libraries (BUILD_SHARED_LIBS=OFF).
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
src=$1
build=$2
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

tests=$src/tests/test-chat-template.cpp
# the conversation and the test cases, as main_automated_tests declares them
sed -n '/^    std::vector<llama_chat_message> conversation {/,/^    };/p' "$tests" > "$tmp/cases.inc"
sed -n '/^    struct TestCase {/,/^    std::vector<char> formatted_chat/p' "$tests" | sed '$d' >> "$tmp/cases.inc"
cp "$root/scripts/extract_chat_cases.cpp" "$tmp/"
g++ -O1 -std=c++17 -I"$tmp" -I"$src/include" -I"$src/ggml/include" -I"$src/common" -I"$src/vendor" \
    "$tmp/extract_chat_cases.cpp" \
    "$build/common/libllama-common.a" "$build/common/libllama-common-base.a" \
    "$build/src/libllama.a" "$build/ggml/src/ggml.a" "$build/ggml/src/ggml-cpu.a" "$build/ggml/src/ggml-base.a" \
    -o "$tmp/extract.exe"
"$tmp/extract.exe" "$root/tests/data/chat_cases.txt"
echo "wrote tests/data/chat_cases.txt"
