#!/usr/bin/env bash
# Weekly sync with llama.cpp: list what changed upstream since the commit in
# UPSTREAM, sorted by what has to happen to it here.
#
#   scripts/sync-upstream.sh            report only; writes sync-report.md
#   scripts/sync-upstream.sh --apply    also copy ggml and move UPSTREAM forward
#
# LLAMA_CPP_DIR  a llama.cpp clone to fetch into (default ../llama.cpp); only
#                `git fetch` runs there, its branches and files stay as they are
# UPSTREAM_URL   repository and branch to follow (default the fork's master)
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
llama=$(cd "${LLAMA_CPP_DIR:-$root/../llama.cpp}" && pwd)
url=${UPSTREAM_URL:-https://github.com/aguspiza/llama.cpp.git}
branch=${UPSTREAM_BRANCH:-master}
apply=false
[[ ${1:-} == --apply ]] && apply=true

old=$(cat "$root/UPSTREAM")
git -C "$llama" fetch -q "$url" "$branch"
new=$(git -C "$llama" rev-parse FETCH_HEAD)

if [[ $old == "$new" ]]; then
    echo "Already at $new; nothing to do."
    exit 0
fi

# Every ported Goose file names its llama.cpp sources on a `// port-of:` line.
declare -A ported_by=()
while IFS= read -r line; do
    file=${line%%:*}
    for src in ${line#*port-of:}; do
        ported_by[$src]+="${file#$root/} "
    done
done < <(grep -r --include='*.goose' -H '^// port-of:' "$root" | sed 's#:// port-of:#:port-of:#')

port=() todo=() vendored=() ignored=()
while IFS=$'\t' read -r status path rest; do
    path=${rest:-$path}            # renames: take the new name
    if [[ -n ${ported_by[$path]:-} ]]; then
        port+=("$status $path -> ${ported_by[$path]}")
    elif [[ $path == ggml/include/* || $path == ggml/src/ggml-cpu/* || $path =~ ^ggml/src/[^/]+$ ]]; then
        vendored+=("$status $path")
    elif [[ $path =~ ^(src|common|tools|include)/ && $path =~ \.(c|cpp|h|hpp)$ ]]; then
        todo+=("$status $path")
    else
        ignored+=("$status $path")
    fi
done < <(git -C "$llama" diff --name-status "$old" "$new")

section() {
    local title=$1; shift
    echo "## $title ($#)"
    echo
    for item in "$@"; do echo "- $item"; done
    echo
}

report=$root/sync-report.md
{
    echo "# Sync $(git -C "$llama" log -1 --format=%h "$old") -> $(git -C "$llama" log -1 --format='%h (%cd)' --date=short "$new")"
    echo
    echo "See one file's changes with: git -C $llama diff $old $new -- <path>"
    echo
    section "Ported files to convert again" "${port[@]}"
    section "Not ported yet" "${todo[@]}"
    section "ggml (copied as is with --apply)" "${vendored[@]}"
    echo "## Ignored: docs, CI, other backends, scripts (${#ignored[@]})"
} > "$report"

echo "Changes from ${old:0:9} to ${new:0:9}: ${#port[@]} to convert again, ${#todo[@]} not ported yet, ${#vendored[@]} in ggml, ${#ignored[@]} ignored."
echo "Report: $report"

$apply || exit 0

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git -C "$llama" archive "$new" ggml/include ggml/src ggml/CMakeLists.txt | tar -x -C "$tmp"
rm -rf "$root/ggml/include" "$root/ggml/src"
cp -r "$tmp/ggml/include" "$tmp/ggml/src" "$root/ggml/"

# What CMake would generate from ggml/src/ggml-version.h.in.
version=$(sed -n -E 's/^set\(GGML_VERSION_(MAJOR|MINOR|PATCH) ([0-9]+)\)/\2/p' "$tmp/ggml/CMakeLists.txt" | paste -sd.)
cat > "$root/ggml/units/ggml-version.h" <<EOF
// What CMake generates from src/ggml-version.h.in.
#pragma once

#define GGML_VERSION "$version"
#define GGML_COMMIT  "${new:0:9}"
EOF

# Sources ggml's CMake builds for an x86 CPU that the sidecar does not compile,
# itself or through a wrapper in units/.
covered=$( { sed -n -E 's#^ *"src/([^"]+)",#\1#p' "$root/ggml/ggml.build"
             sed -n -E 's#^ *"\.\./src/([^"]+)",#\1#p' "$root/ggml/cpu/ggml_cpu.build"
             sed -n -E 's#^\#include "\.\./src/([^"]+)"#\1#p' "$root/ggml/units/"*.c* ; } | sort -u)
missing=$( { sed -n -E 's#^ +(ggml[A-Za-z0-9_.-]*\.(c|cpp)|gguf\.cpp)\)?$#\1#p' "$root/ggml/src/CMakeLists.txt"
             grep -o -E 'ggml-cpu/[A-Za-z0-9_/-]+\.(c|cpp)' "$root/ggml/src/ggml-cpu/CMakeLists.txt" ; } \
           | grep -v -E 'arch/(arm|loongarch|powerpc|riscv|s390|wasm)|kleidiai|spacemit|cpu-feats|/cmake/' \
           | sort -u | comm -23 - <(echo "$covered"))
if [[ -n $missing ]]; then
    echo "ggml builds sources that ggml/ggml.build does not list; add them:"
    echo "$missing" | sed 's/^/  /'
fi

# The generated Goose: the Unicode tables and the BPE pre-tokenizers.
git -C "$llama" archive "$new" src/unicode-data.cpp src/llama-vocab.h src/llama-vocab.cpp | tar -x -C "$tmp"
python -I "$root/scripts/gen-unicode-data.py" "$tmp/src/unicode-data.cpp" "$root/llama/unicode_data.goose"
python -I "$root/scripts/gen-vocab-pre.py" "$tmp/src" "$root/llama/vocab_pre.goose"
# the tests' cases come from a llama.cpp build: scripts/gen-chat-cases.sh,
# gen-jinja-cases.sh, gen-grammar-cases.sh, gen-schema-cases.sh,
# gen-autoparser-cases.sh and gen-peg-chat-cases.sh

echo "$new" > "$root/UPSTREAM"
echo "ggml copied at ggml $version; UPSTREAM is now ${new:0:9}. Next: scripts/test.sh"
