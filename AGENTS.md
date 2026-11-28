# AGENTS.md

How work is done in this port of llama.cpp to Goose. What it is and what is
done is in [README.md](README.md); how each part is checked, in
[docs/verification.md](docs/verification.md).

## Porting a llama.cpp source

- The Goose (or C) file that ports a llama.cpp source names it on a
  `// port-of:` line, with a note in parentheses when it ports only part of
  it: `// port-of: common/chat.cpp (format_single)`. A header counts as ported
  with its `.c` or `.cpp`.
- Porting more of a source into a file that already exists counts too: add
  the source to that file's `port-of:` line in the same change.
- Every source that is not ported is in [not-ported.txt](not-ported.txt), with
  why (not ported yet, replaced by something here, not needed). Porting one
  means moving it from there to a `port-of:` line.
- `scripts/check-port-of.sh`, part of `scripts/test.sh`, fails when a source
  of `src/`, `common/`, `tools/` or `include/` at the commit in `UPSTREAM` is in
  neither, when it is in both, or when either names a file llama.cpp does not
  have. The weekly sync (`scripts/sync-upstream.sh`) relies on these lines to
  say which upstream changes have to be converted again, so a missing one
  hides an upstream fix.

## Checking a change

- The target is llama.cpp's output byte for byte: the same text, logits, JSON
  and log lines, from the same arguments. A difference is a bug here until
  shown otherwise; the few known ones are written down in
  docs/verification.md with their reason.
- Regressions over many models go in stages: one model, then 10, then the
  rest, each run in its own temporary directory.
- Fix the root cause. A workaround is only a test to learn something: say so,
  keep a backup, and revert it.
- `scripts/test.sh` passes before a commit.

## Writing it down

- A change to what is ported updates its row in README.md ("What is missing",
  "What is ported", "Programs") and docs/verification.md: what it does and how
  it was compared, with the results.
- No local paths, user names or machine details in the repository: commits,
  docs, scripts and comments refer to files relative to the repository or to
  `../llama.cpp`.
