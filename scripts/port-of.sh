# Sourced by check-port-of.sh and sync-upstream.sh: which llama.cpp sources
# this port follows, and how.
#
# A Goose or C file here names the llama.cpp sources it ports on `// port-of:`
# lines, separated by spaces or commas, each maybe followed by a note in
# parentheses. A header counts as ported with its .c or .cpp. The sources not
# ported are in not-ported.txt, each with why, a header with its .c or .cpp;
# a pattern there (tools/tuning/*) covers the files it matches that no
# port-of names.

# ported_by[path]: the files here that port it.
declare -A ported_by=()
# not_ported[path or pattern]: why.
declare -A not_ported=()
# its keys that are patterns
not_ported_patterns=()

load_port_of() {
    local root=$1 line file srcs src
    while IFS= read -r line; do
        file=${line%%:*}
        srcs=$(sed -e 's/([^)]*)//g' -e 's/,/ /g' <<< "${line#*port-of:}")
        for src in $srcs; do
            ported_by[$src]+="${file#$root/} "
        done
    done < <(grep -r -H --include='*.goose' --include='*.c' --include='*.h' \
                 --exclude-dir=src --exclude-dir=include --exclude-dir=vendor --exclude-dir=.git \
                 '^// port-of:' "$root" | sed 's#:// port-of:#:port-of:#')
}

load_not_ported() {
    local root=$1 path why
    while read -r path why; do
        [[ -z $path || $path == \#* ]] && continue
        not_ported[$path]=$why
        if [[ $path == *\** ]]; then not_ported_patterns+=("$path"); fi
    done < "$root/not-ported.txt"
}

# ported_of path: in by, the files here that port path, or nothing.
ported_of() {
    local path=$1
    by=${ported_by[$path]:-}
    if [[ -z $by && $path =~ \.(h|hpp)$ ]]; then
        local base=${path%.*}
        by=${ported_by[$base.cpp]:-${ported_by[$base.c]:-}}
    fi
}

# not_ported_of path: in why, why path is not ported (its entry, its .c or
# .cpp's, or a pattern's of not-ported.txt), or nothing.
not_ported_of() {
    local path=$1 key base=${1%.*}
    why=
    if [[ -n ${not_ported[$path]+x} ]]; then why=${not_ported[$path]:--}; return; fi
    if [[ $path =~ \.(h|hpp)$ ]]; then
        if [[ -n ${not_ported[$base.cpp]+x} ]]; then why=${not_ported[$base.cpp]:--}; return; fi
        if [[ -n ${not_ported[$base.c]+x} ]]; then why=${not_ported[$base.c]:--}; return; fi
    fi
    for key in "${not_ported_patterns[@]}"; do
        # shellcheck disable=SC2053
        if [[ $path == $key ]]; then why=${not_ported[$key]:--}; return; fi
    done
}

# Whether llama.cpp follows path here at all: the C and C++ of src/, common/, tools/, include/.
followed() { [[ $1 =~ ^(src|common|tools|include)/ && $1 =~ \.(c|cpp|h|hpp)$ ]]; }
