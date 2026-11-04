#!/usr/bin/env bash
# Regenerate tests/data/peg_chat_cases.txt from llama.cpp's test-chat.cpp
# (test_template_output_peg_parsers) and what its libcommon makes of each
# case: prompt, grammar, triggers and parsed messages (run after a sync with
# upstream).
#   scripts/gen-peg-chat-cases.sh <llama.cpp source dir> <its build dir>
# The build must have static libraries (BUILD_SHARED_LIBS=OFF). It runs in
# the source dir, where test-chat.cpp finds models/templates.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
src=$(cd "$1" && pwd)
build=$(cd "$2" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# test-chat.cpp up to its own tests, then test_template_output_peg_parsers,
# with each peg_test_builder::run() recording its case (and no server)
t=$src/tests/test-chat.cpp
first=$(grep -n '^static void test_msgs_oaicompat_json_conversion' "$t" | cut -d: -f1)
peg=$(grep -n '^static void test_template_output_peg_parsers' "$t" | cut -d: -f1)
next=$(awk -v s="$peg" 'NR > s && /^static void / { print NR; exit }' "$t")
lfm2=$(grep -n '^static void test_lfm2_parser' "$t" | cut -d: -f1)
lfm2_end=$(awk -v s="$lfm2" 'NR > s && /^static void / { print NR; exit }' "$t")
{ sed -n "1,$((first - 1))p" "$t"; sed -n "${lfm2},$((lfm2_end - 1))p" "$t"; sed -n "${peg},$((next - 1))p" "$t"; } |
    sed -e '/server-chat.h/d' \
        -e 's|^class peg_test_builder;$|#include "peg_chat_record.inc"\nclass peg_test_builder;|' \
        -e 's|test_peg_parser(tester_.tmpls_.get(), \[this\](peg_test_case \& t) { t = tc_; }, tester_.detailed_debug_);|record_case(tester_.template_path(), tester_.tmpls_.get(), tc_);|' \
    > "$tmp/test-chat-recorded.cpp"
grep -q 'record_case(tester_' "$tmp/test-chat-recorded.cpp" || { echo "test-chat.cpp changed: run() not found"; exit 1; }
cp "$root/scripts/extract_peg_chat_cases.cpp" "$root/scripts/peg_chat_record.inc" "$tmp/"
g++ -O1 -std=c++17 -I"$tmp" -I"$src/tests" -I"$src/src" -I"$src/include" -I"$src/ggml/include" -I"$src/common" -I"$src/vendor" \
    "$tmp/extract_peg_chat_cases.cpp" \
    "$build/common/libllama-common.a" "$build/common/libllama-common-base.a" \
    "$build/src/libllama.a" "$build/ggml/src/ggml.a" "$build/ggml/src/ggml-cpu.a" "$build/ggml/src/ggml-base.a" \
    -lws2_32 -o "$tmp/extract.exe"
(cd "$src" && "$tmp/extract.exe" "$root/tests/data/peg_chat_cases.txt")
echo "wrote tests/data/peg_chat_cases.txt"
