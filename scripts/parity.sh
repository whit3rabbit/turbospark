#!/bin/bash
# Swift-vs-Rust decode parity on one machine, one install, one session.
#
# Runs the frozen community protocol (real-generation-v1) through both
# engines: ../Mference's MferenceCLI and this port's mference-bench. Same
# prompts, seeds, sampling, budget, and expert-cache slots; one fresh
# process per measured run, arms interleaved pair by pair because
# consecutive batches carry thermal drift (CLAUDE.local.md).
#
# Usage: scripts/parity.sh [pairs]        (default 2 pairs per case)
# Env:   MODEL, SWIFT_CLI, SWIFT_PROMPTS, RUST_BENCH, OUT
#
# Reads the [stop=...] footer out of each run's stderr FILE, never the
# stream: `2>&1 >/dev/null` races the streamed generation text and
# silently truncates the capture.

set -u

PAIRS="${1:-2}"
MODEL="${MODEL:-$HOME/models/gemma4.gturbo}"
SWIFT_CLI="${SWIFT_CLI:-../Mference/.build/release/MferenceCLI}"
SWIFT_PROMPTS="${SWIFT_PROMPTS:-../Mference/docs/benchmark-prompts/real-generation-v1}"
RUST_BENCH="${RUST_BENCH:-./target/release/mference-bench}"
OUT="${OUT:-/tmp/mference-parity}"

CASES=(short-explanation:20260721 medium-review:20260722 long-synthesis:20260723)

for path in "$SWIFT_CLI" "$RUST_BENCH"; do
  [ -x "$path" ] || { echo "missing or not executable: $path" >&2; exit 2; }
done
[ -d "$MODEL" ] || { echo "missing install: $MODEL" >&2; exit 2; }
mkdir -p "$OUT"
: > "$OUT/rows.tsv"

# Provenance. A number without the machine state attached is not reusable.
{
  echo "date         $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "chip         $(sysctl -n machdep.cpu.brand_string)"
  echo "memory       $(($(sysctl -n hw.memsize) / 1073741824)) GB"
  echo "macos        $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
  echo "power        $(pmset -g ps | head -1)"
  echo "rust rev     $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)')"
  echo "swift rev    $(git -C "$(dirname "$SWIFT_CLI")/../.." rev-parse --short HEAD 2>/dev/null)"
  echo "install      $MODEL"
  echo "manifest sha $(shasum -a 256 "$MODEL/manifest.json" | cut -d' ' -f1)"
  echo "pairs        $PAIRS"
} | tee "$OUT/system.txt"
echo

# Fail loudly on a competing model process: it would show up as a slow arm.
if pgrep -fl 'Mference(CLI|Server|Mac|DecodeService)|mference-(check|server|bench)|mlx' \
     | grep -v parity.sh | grep -q .; then
  echo "another model process is running; results would be contaminated" >&2
  pgrep -fl 'Mference|mference|mlx' | grep -v parity.sh >&2
  exit 2
fi

# One measured run of one engine, in a fresh process, under /usr/bin/time
# -l. Its `peak memory footprint` line IS phys_footprint, the counter both
# engines' published memory numbers use, and the kernel reports it for any
# process -- so it works for Swift, whose CLI prints no memory line. Peak
# RSS comes along for free and is kept as a secondary column.
run_arm() {
  local engine="$1" case_id="$2" seed="$3" tag="$4"
  local stem="$OUT/${case_id}.${engine}.${tag}"
  local start elapsed
  start=$(date +%s)
  case "$engine" in
    swift)
      /usr/bin/time -l "$SWIFT_CLI" \
        --model "$MODEL" \
        --messages-file "$SWIFT_PROMPTS/${case_id}.json" \
        --max-new 1024 --max-context 4096 \
        --temperature 0.2 --top-k 64 --top-p 0.95 --seed "$seed" \
        > "$stem.stdout" 2> "$stem.stderr"
      ;;
    rust)
      # mference-bench does the protocol's discarded warmup inside the
      # process, so one launch is warmup + measured for this case.
      /usr/bin/time -l "$RUST_BENCH" --model "$MODEL" --case "$case_id" \
        > "$stem.stdout" 2> "$stem.stderr"
      ;;
  esac
  elapsed=$(( $(date +%s) - start ))

  local footer stop prefill_s new_tok decode_s tok_s rss_mib fp_mib
  footer=$(grep '^\[stop=' "$stem.stderr" | tail -1)
  [ -n "$footer" ] || { echo "  $engine/$case_id/$tag: NO FOOTER (see $stem.stderr)"; return 1; }
  stop=$(sed -n 's/.*stop=\([a-zA-Z]*\).*/\1/p' <<< "$footer")
  prefill_s=$(sed -n 's/.*prefill=[0-9]*tok\/\([0-9.]*\)s.*/\1/p' <<< "$footer")
  new_tok=$(sed -n 's/.*new=\([0-9]*\)tok.*/\1/p' <<< "$footer")
  decode_s=$(sed -n 's/.*decode=\([0-9.]*\)s.*/\1/p' <<< "$footer")
  tok_s=$(sed -n 's/.*tok\/s=\([0-9.]*\).*/\1/p' <<< "$footer")
  rss_mib=$(awk '/maximum resident set size/ {printf "%.0f", $1 / 1048576}' "$stem.stderr")
  fp_mib=$(awk '/peak memory footprint/ {printf "%.1f", $1 / 1048576}' "$stem.stderr")

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_id" "$engine" "$tag" "$stop" "$prefill_s" "$new_tok" "$decode_s" \
    "$tok_s" "${rss_mib:-}" "${fp_mib:-}" >> "$OUT/rows.tsv"
  printf '  %-6s %-8s stop=%-10s prefill %7ss  new=%-5s %8.2fs %8s tok/s  fp %7s MiB  rss %5s MiB  (%ds wall)\n' \
    "$engine" "$tag" "$stop" "$prefill_s" "$new_tok" "$decode_s" "$tok_s" \
    "${fp_mib:-?}" "${rss_mib:-?}" "$elapsed"
}

for entry in "${CASES[@]}"; do
  case_id="${entry%%:*}"
  seed="${entry##*:}"
  echo "== $case_id (seed $seed)"
  # Discarded warmup per engine: the first run after a build is a cold
  # GPU at low DVFS clocks and reads up to 53% off (AGENTS.md Gotcha 20).
  run_arm swift "$case_id" "$seed" warmup
  run_arm rust  "$case_id" "$seed" warmup
  for ((p = 1; p <= PAIRS; p++)); do
    run_arm swift "$case_id" "$seed" "p$p"
    run_arm rust  "$case_id" "$seed" "p$p"
  done
  echo
done

echo "== measured rows (warmups excluded)"
awk -F'\t' '
  $3 != "warmup" {
    if ($4 != "endOfTurn") bad[$1 "/" $2 "/" $3] = $4
    k = $1 "\t" $2
    prefill[k] += $5; sum[k] += $8; n[k]++
    if ($9 != "") { rss[k] = ($9 > rss[k] ? $9 : rss[k]) }
    if ($10 != "") { fp[k] = ($10 > fp[k] ? $10 : fp[k]) }
    tok[k] = $6
  }
  END {
    printf "%-18s %-6s %5s %8s %10s %9s %10s %9s\n", \
      "case", "engine", "runs", "new_tok", "prefill_s", "tok/s", "peak_fp", "peak_rss"
    for (k in sum) {
      split(k, f, "\t")
      printf "%-18s %-6s %5d %8s %9.2fs %9.3f %6.0f MiB %5d MiB\n", \
        f[1], f[2], n[k], tok[k], prefill[k] / n[k], sum[k] / n[k], fp[k], rss[k]
    }
    for (k in bad) printf "INVALID %s stopped %s (protocol requires endOfTurn)\n", k, bad[k]
  }
' "$OUT/rows.tsv" | sort

echo
echo "raw rows: $OUT/rows.tsv   captures: $OUT/*.stdout"
