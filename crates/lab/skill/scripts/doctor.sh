#!/bin/sh
# Check that tslab can run here and say which binary to use. Read-only: it
# reports, it fixes nothing. Exit 0 when every required check passes, 1
# otherwise.
#
# The skill runs in two places: unpacked from the zip, with bin/tslab beside
# this script's folder, or as crates/lab/skill in a TurboSpark checkout, where
# the binary is the checkout's release build.
set -u

# -P resolves symlinks (such as .claude/skills or .agents/skills) to the real
# folder, so a checkout is found from either path.
SKILL=$(cd "$(dirname "$0")/.." && pwd -P)
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

ROOT=$(cd "$SKILL/../../.." 2>/dev/null && pwd -P)
if [ -x "$SKILL/bin/tslab" ] || [ -d "$SKILL/bin" ]; then
    MODE=zip
    BIN="$SKILL/bin"
    WORKLOADS="$SKILL/workloads"
    ok "bundled binaries (see BUILD.txt for the build)"
elif [ -f "$ROOT/crates/lab/Cargo.toml" ]; then
    MODE=checkout
    BIN="${CARGO_TARGET_DIR:-$ROOT/target}/release"
    WORKLOADS="$ROOT/crates/lab/workloads"
    ok "TurboSpark checkout at $ROOT"
else
    bad "no bin/ beside the skill and no TurboSpark checkout around it"
    exit 1
fi

for b in tslab tslab-worker; do
    if [ -x "$BIN/$b" ]; then
        ok "$b present"
    elif [ "$MODE" = checkout ]; then
        bad "$BIN/$b missing: cargo build --release -p turbospark-lab"
    else
        bad "$BIN/$b missing or not executable"
    fi
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

if [ -d "$WORKLOADS" ]; then ok "workloads at $WORKLOADS"; else bad "workloads missing at $WORKLOADS"; fi

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

[ -x "$BIN/tslab" ] && echo "use: TSLAB=$BIN/tslab"
exit $fail
