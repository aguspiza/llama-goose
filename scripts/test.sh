#!/usr/bin/env bash
# Build and run every program in tests/; each exits non-zero on failure.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
failed=0
for program in "$root"/tests/*.goose; do
    "$root/scripts/build.sh" "$program" > /dev/null
    if "${program%.goose}.exe"; then :; else failed=$((failed + 1)); echo "FAILED: ${program#$root/}"; fi
done
[[ $failed == 0 ]] && echo "all tests passed" || { echo "$failed failed"; exit 1; }
