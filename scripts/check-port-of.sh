#!/usr/bin/env bash
# Every llama.cpp source this port follows (see scripts/port-of.sh), at the
# commit in UPSTREAM, is either named on a `port-of:` line or listed in
# not-ported.txt, not both; and every path either names is a file there. Run
# by scripts/test.sh, so that porting a source means saying so.
#
# LLAMA_CPP_DIR  a llama.cpp clone that has the commit (default ../llama.cpp)
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
llama=$(cd "${LLAMA_CPP_DIR:-$root/../llama.cpp}" && pwd)
rev=$(cat "$root/UPSTREAM")
. "$root/scripts/port-of.sh"
load_port_of "$root"
load_not_ported "$root"

declare -A upstream=()
while IFS= read -r path; do upstream[$path]=1; done < <(git -C "$llama" ls-tree -r --name-only "$rev")

problems=()
for path in "${!upstream[@]}"; do
    followed "$path" || continue
    ported_of "$path"
    not_ported_of "$path"
    if [[ -z $by && -z $why ]]; then
        problems+=("neither ported nor in not-ported.txt: $path")
    elif [[ -n $by && -n ${not_ported[$path]+x} ]]; then
        problems+=("ported by $by and in not-ported.txt: $path")
    fi
done
for path in "${!ported_by[@]}"; do
    [[ -n ${upstream[$path]:-} ]] || problems+=("named by port-of but not in llama.cpp: $path (${ported_by[$path]% })")
done
for key in "${!not_ported[@]}"; do
    if [[ $key == *\** ]]; then
        found=false
        for path in "${!upstream[@]}"; do
            # shellcheck disable=SC2053
            if [[ $path == $key ]]; then found=true; break; fi
        done
        $found || problems+=("pattern of not-ported.txt matches nothing: $key")
    else
        [[ -n ${upstream[$key]:-} ]] || problems+=("in not-ported.txt but not in llama.cpp: $key")
    fi
done

if (( ${#problems[@]} > 0 )); then
    printf '%s\n' "${problems[@]}" | sort
    echo "port-of: ${#problems[@]} problems (llama.cpp ${rev:0:9})"
    exit 1
fi
echo "port-of: every followed source of llama.cpp ${rev:0:9} accounted for"
