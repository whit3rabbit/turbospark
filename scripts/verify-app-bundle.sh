#!/usr/bin/env bash
# Checks that an assembled TurboSpark.app can actually start, and that it was
# built to run on the macOS the Info.plist promises.
#
# WHY THIS EXISTS: v0.2.0 and v0.3.0 shipped a bundle that built, signed and
# packaged cleanly and then crashed on launch on every Mac ("could not load
# resource bundle"), because SwiftPM's generated resource accessor depends on
# the toolchain and nothing in CI ever ran the app. The packaging job only
# proved the files existed. This runs the binary.
#
# Usage: scripts/verify-app-bundle.sh [path/to/TurboSpark.app]   (default dist/TurboSpark.app)
#
# Checks, in order:
#   1. every Mach-O in the bundle targets a macOS no newer than the plist's
#      LSMinimumSystemVersion. A newer SDK does not lower the floor on its
#      own, and a framework built against a newer floor would load fine on
#      the CI host and fail on the oldest Mac the cask admits.
#   2. the resource bundle is under Contents/Resources (codesign refuses it
#      anywhere else).
#   3. every peer binary the `turbospark` front end execs is in the bundle.
#      0.6.0 shipped without turbospark-image because the packaging lists
#      were hand-kept and nothing compared them to the dispatch table.
#   4. the app stays alive for a few seconds. It is run with an empty HOME so
#      it cannot touch real user data.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
app="${1:-$root/dist/TurboSpark.app}"
smoke_seconds="${SMOKE_SECONDS:-10}"
contents="$app/Contents"
exe="$contents/MacOS/TurboSparkApp"

fail() { echo "verify-app-bundle: FAIL: $*" >&2; exit 1; }

[ -x "$exe" ] || fail "$exe does not exist or is not executable"

# 1. Deployment floor ---------------------------------------------------------
plist_min="$(/usr/libexec/PlistBuddy -c 'Print :LSMinimumSystemVersion' "$contents/Info.plist")"
echo "==> Info.plist LSMinimumSystemVersion: $plist_min"

# `a <= b` for dotted versions.
version_le() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" = "$1" ]; }

checked=0
while IFS= read -r f; do
  file -b "$f" | grep -q 'Mach-O' || continue
  # Fat files list one LC_BUILD_VERSION per slice; each one must pass.
  minos_list="$(otool -l "$f" | awk '/LC_BUILD_VERSION/{b=1} b&&$1=="minos"{print $2; b=0}')"
  [ -n "$minos_list" ] || continue
  for minos in $minos_list; do
    version_le "$minos" "$plist_min" \
      || fail "$f targets macOS $minos, newer than the plist's $plist_min"
  done
  checked=$((checked + 1))
done < <(find "$contents/MacOS" "$contents/Frameworks" -type f 2>/dev/null)
[ "$checked" -gt 0 ] || fail "found no Mach-O with a build version to check (otool output changed?)"
echo "==> $checked Mach-O files target macOS <= $plist_min"

# 2. Resource bundle placement ----------------------------------------------
ls -d "$contents"/Resources/*.bundle >/dev/null 2>&1 \
  || fail "no *.bundle under Contents/Resources"
if ls -d "$app"/*.bundle >/dev/null 2>&1; then
  fail "a .bundle sits at the app root; codesign rejects unsealed root contents"
fi

# 3. CLI peers -----------------------------------------------------------------
# turbospark-bench is a dev harness from crates/bench, deliberately not shipped.
front_end="$root/crates/cli/src/bin/turbospark.rs"
[ -f "$front_end" ] || fail "$front_end not found; run from a source checkout"
peers="$(grep -oE 'execute_peer\("[a-z0-9-]+"' "$front_end" | sed -E 's/.*"(.*)"/\1/' | sort -u | grep -vx turbospark-bench || true)"
[ -n "$peers" ] || fail "found no execute_peer calls in $front_end (dispatch changed?)"
for bin in turbospark $peers; do
  [ -x "$contents/MacOS/$bin" ] || fail "CLI binary $bin is dispatched by turbospark but missing from $contents/MacOS"
done
echo "==> CLI peers present: $(echo $peers | tr '\n' ' ')"

# 4. Launch smoke -----------------------------------------------------------
home="$(mktemp -d)"
log="$(mktemp)"
trap 'rm -rf "$home" "$log"; [ -n "${pid:-}" ] && kill "$pid" 2>/dev/null || true' EXIT

echo "==> launching for ${smoke_seconds}s with an empty HOME"
HOME="$home" "$exe" >"$log" 2>&1 &
pid=$!
for _ in $(seq 1 "$smoke_seconds"); do
  sleep 1
  if ! kill -0 "$pid" 2>/dev/null; then
    status=0; wait "$pid" || status=$?
    pid=""
    echo "--- app output ---" >&2; cat "$log" >&2; echo "------------------" >&2
    fail "app exited after under ${smoke_seconds}s (status $status); a GUI app that is healthy keeps running"
  fi
done
if grep -qiE 'could not load resource bundle|unable to find bundle' "$log"; then
  cat "$log" >&2
  fail "resource bundle lookup failed while the process was still alive"
fi
echo "==> OK: $app stayed up for ${smoke_seconds}s, deployment floor $plist_min"
