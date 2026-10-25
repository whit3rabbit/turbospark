#!/bin/sh
# Check that the bundled tslab can run here. Read-only: it reports, it fixes
# nothing. Exit 0 when every required check passes, 1 otherwise.
set -u

SKILL=$(cd "$(dirname "$0")/.." && pwd)
BIN="$SKILL/bin"
fail=0

ok() { printf '  ok    %s\n' "$1"; }
bad() { printf '  FAIL  %s\n' "$1"; fail=1; }
note() { printf '  note  %s\n' "$1"; }

echo "tslab skill at $SKILL"

if [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ]; then
    ok "macOS on Apple Silicon"
else
    bad "needs macOS on Apple Silicon (this is $(uname -s) $(uname -m))"
fi

for b in tslab tslab-worker; do
    if [ -x "$BIN/$b" ]; then ok "$b present"; else bad "$BIN/$b missing or not executable"; fi
done

# A downloaded zip is quarantined, and Gatekeeper then refuses to run the
# unsigned binaries. Clearing it is the person's call: say how, do not do it.
if xattr "$BIN/tslab" 2>/dev/null | grep -q com.apple.quarantine; then
    bad "binaries are quarantined. If you trust this build: xattr -dr com.apple.quarantine \"$SKILL\""
else
    ok "binaries not quarantined"
fi

if "$BIN/tslab" knobs --json >/dev/null 2>&1; then
    ok "tslab runs"
else
    bad "tslab did not run (see: \"$BIN/tslab\" knobs)"
fi

if [ -d "$SKILL/workloads" ]; then ok "workloads present"; else bad "workloads/ missing beside bin/"; fi

LAB=${TURBOSPARK_LAB_DIR:-${TURBOSPARK_HOME:-$HOME/.turbospark}/lab}
if mkdir -p "$LAB" 2>/dev/null && [ -w "$LAB" ]; then
    ok "results directory $LAB"
else
    bad "cannot write results directory $LAB"
fi

if [ -d "${TURBOSPARK_HOME:-$HOME/.turbospark}/models" ]; then
    ok "model store ${TURBOSPARK_HOME:-$HOME/.turbospark}/models"
else
    note "no model store at ${TURBOSPARK_HOME:-$HOME/.turbospark}/models: pass install paths to --model"
fi

# Optional tools: only tslab trace needs them.
if xcrun --find xctrace >/dev/null 2>&1; then ok "xctrace (tslab trace)"; else note "no xctrace: tslab trace unavailable (install Xcode)"; fi
if command -v python3 >/dev/null 2>&1; then ok "python3 (tslab trace)"; else note "no python3: tslab trace unavailable"; fi

if pmset -g batt 2>/dev/null | grep -q "AC Power"; then ok "on AC power"; else note "on battery: runs will be dirty unless --allow-battery"; fi

exit $fail
