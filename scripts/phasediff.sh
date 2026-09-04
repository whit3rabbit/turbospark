#!/bin/bash
# Bucket-level decode phase diff between this port and the Swift original.
#
# Both engines print a decode phase split (TURBOSPARK_PHASES=1 for Rust, phase profiling for Swift), but they
# do NOT divide by the same thing:
#
#   Swift  resets its counters at the prefill/decode boundary
#          (RealForwardRunner.beginDecodePhaseWindow) and divides by
#          generated tokens, so its numbers are decode-only.
#   This   port accumulates across every forward pass, prefill included,
#   port   and divides by that count (AGENTS.md Gotcha 21).
#
# So the prompt must be SHORT and the generation LONG, or this port's
# columns are an average over a range Swift's are not. At 20 prompt tokens
# and 1024 generated the residual skew is ~2%.
#
# Arms are interleaved pair by pair after a discarded warmup, because the
# first run after a build is a cold GPU (Gotcha 20) and consecutive batches
# carry thermal drift (CLAUDE.local.md).
#
# Usage: scripts/phasediff.sh [pairs] [slots]
# Env:   MODEL, SWIFT_CLI, RUST_CLI, PROMPT, MAX_NEW, OUT

set -u

PAIRS="${1:-3}"
SLOTS="${2:-16}"
MODEL="${MODEL:-$HOME/models/gemma4.gturbo}"
SWIFT_CLI="${SWIFT_CLI:-../Mference/.build/release/MferenceCLI}"
RUST_CLI="${RUST_CLI:-./target/release/turbospark-check}"
PROMPT="${PROMPT:-/tmp/phase-prompt.json}"
MAX_NEW="${MAX_NEW:-1024}"
OUT="${OUT:-/tmp/turbospark-phasediff}"

for path in "$SWIFT_CLI" "$RUST_CLI"; do
  [ -x "$path" ] || { echo "missing or not executable: $path" >&2; exit 2; }
done
[ -d "$MODEL" ] || { echo "missing install: $MODEL" >&2; exit 2; }
[ -f "$PROMPT" ] || { echo "missing prompt: $PROMPT" >&2; exit 2; }
mkdir -p "$OUT"

{
  echo "date    $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "chip    $(sysctl -n machdep.cpu.brand_string)"
  echo "power   $(pmset -g ps | head -1)"
  echo "rust    $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)')"
  echo "swift   $(git -C "$(dirname "$SWIFT_CLI")/../.." rev-parse --short HEAD 2>/dev/null)"
  echo "prompt  $PROMPT ($(wc -c < "$PROMPT") bytes)"
  echo "max-new $MAX_NEW   slots $SLOTS   pairs $PAIRS"
} | tee "$OUT/system.txt"
echo

if pgrep -fl 'Mference(CLI|Server|Mac|DecodeService)|turbospark-(check|server|bench)|mlx' \
     | grep -v phasediff.sh | grep -q .; then
  echo "another model process is running; results would be contaminated" >&2
  exit 2
fi

run_arm() {
  local engine="$1" tag="$2"
  local stem="$OUT/${engine}.${tag}"
  case "$engine" in
    swift)
      TURBOSPARK_PHASES=1 "$SWIFT_CLI" --model "$MODEL" --messages-file "$PROMPT" \
        --max-new "$MAX_NEW" --max-context 4096 --expert-cache-slots "$SLOTS" \
        --temperature 0.2 --top-k 64 --top-p 0.95 --seed 20260721 \
        > "$stem.stdout" 2> "$stem.stderr"
      ;;
    rust)
      TURBOSPARK_PHASES=1 "$RUST_CLI" --model "$MODEL" --messages-file "$PROMPT" \
        --max-new "$MAX_NEW" --max-context 4096 --expert-cache-slots "$SLOTS" \
        --temperature 0.2 --top-k 64 --top-p 0.95 --seed 20260721 \
        > "$stem.stdout" 2> "$stem.stderr"
      ;;
  esac
  local footer
  footer=$(grep '^\[stop=' "$stem.stderr" | tail -1)
  printf '  %-5s %-6s %s\n' "$engine" "$tag" "${footer:-NO FOOTER}"
}

run_arm swift warmup
run_arm rust  warmup
for ((p = 1; p <= PAIRS; p++)); do
  run_arm swift "p$p"
  run_arm rust  "p$p"
done

echo
echo "captures: $OUT/*.stderr"
