#!/usr/bin/env bash
# Prints the syntext commit Cargo.lock resolves `syntext-context` to: the
# revision whose C ABI is compiled into libturbospark_ffi.a.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# `source = "git+https://github.com/whit3rabbit/syntext?tag=<tag>#<commit>"`
# on the line after the package's name and version. The full commit after
# `#` is what cargo checked out, whether a `tag` or a `rev` pinned it.
rev="$(awk '
  $0 == "name = \"syntext-context\"" { found = 1; next }
  found && /^source = / {
    sub(/^.*#/, ""); sub(/"$/, ""); print; exit
  }
  /^\[\[package\]\]/ { found = 0 }
' "$root/Cargo.lock")"

case "$rev" in
  [0-9a-f]*) [ "${#rev}" -eq 40 ] || { echo "unexpected syntext-context revision '$rev' in Cargo.lock" >&2; exit 1; } ;;
  *) echo "syntext-context is not a git dependency in Cargo.lock" >&2; exit 1 ;;
esac
printf '%s\n' "$rev"
