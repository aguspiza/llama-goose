#!/usr/bin/env bash
# Build and run every program in tests/; each exits non-zero on failure.
# tokenizer_0 runs once per test vocab of llama.cpp (LLAMA_CPP_DIR/models,
# default ../llama.cpp/models), since a process holds one vocab.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
models=${LLAMA_CPP_DIR:-$root/../llama.cpp}/models
failed=0
cd "$root"          # tests read tests/data

run() {
    if "$@" > /dev/null 2>&1; then :; else failed=$((failed + 1)); echo "FAILED: ${*#$root/}"; fi
}

for program in "$root"/tests/*.goose; do
    "$root/scripts/build.sh" "$program" > /dev/null
    exe=${program%.goose}.exe
    if [[ $program == */tokenizer_0.goose ]]; then
        for inp in "$models"/ggml-vocab-*.gguf.inp; do
            [[ -e $inp ]] || continue
            case $inp in *bert*) continue ;; esac     # WPM: not ported yet
            run "$exe" "${inp%.inp}"
        done
    else
        run "$exe"
    fi
done
[[ $failed == 0 ]] && echo "all tests passed" || { echo "$failed failed"; exit 1; }
