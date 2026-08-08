#!/bin/bash
# Power baseline for one real install, over the frozen community protocol
# (ROADMAP Phase P1). Reports average watts and joules-per-token, split
# prefill vs decode, plus the residency and thermal columns the hygiene
# audit needs.
#
# Usage: scripts/power.sh [pairs]        (default 2 measured pairs per case)
# Env:   MODEL, RUST_BENCH, OUT, LABEL (ac|battery),
#        QOS (default|utility, comma-separated to interleave an A/B),
#        CASES (space-separated protocol case ids)
#
# NEEDS SUDO: powermetrics is root-only. It prompts once, up front.
#
# THE NUMBER IS ONLY AS GOOD AS ITS WINDOW. `turbospark-bench --model` opens
# a 13 GB mmap, compiles Metal pipelines, and runs a discarded 1024-token
# warmup before the measured run. Wrapping the process would fold all of
# that into the energy total, so the bench emits `[power-window ...]`
# markers around the measured run alone and this script integrates only
# the samples inside them. A row from a build without those markers is
# not a baseline.
#
# ONE powermetrics for the whole script, not one per arm: fewer sudo
# invocations, and the arms are marker-windowed out of the single log
# anyway.
#
# Reads the `[stop=...]` footer and the markers out of each run's stderr
# FILE, never the stream: `2>&1 >/dev/null` races the streamed generation
# text and silently truncates the capture (CLAUDE.local.md).

set -u

PAIRS="${1:-2}"
MODEL="${MODEL:-$HOME/models/gemma4.gturbo}"
RUST_BENCH="${RUST_BENCH:-./target/release/turbospark-bench}"
OUT="${OUT:-/tmp/mference-power}"
LABEL="${LABEL:-unlabelled}"
# The literal "default" means "set no QoS class", spelled as a token
# rather than as the empty string: macOS ships bash 3.2, where
# `read -ra` on an empty line leaves a ZERO-element array and indexing it
# under `set -u` aborts the script.
QOS="${QOS:-default}"
CASES="${CASES:-short-explanation medium-review long-synthesis}"

# powermetrics sample interval. Decode windows are ~26 s and would be
# fine at any interval; PREFILL is what sets this. The short-explanation
# case is 29 prompt tokens at ~21 ms each, so its prefill window is ~0.6 s
# and a 500 ms interval integrated it from 2 samples. At 200 ms the thin
# case gets 3 and decode gets ~130. The summary still warns on any window
# integrated from under 3 samples rather than quietly reporting it.
#
# The sampler is not free: it wakes 5x/s and its own CPU time lands in the
# very counters being read. That inflates absolute watts for every arm
# equally, so paired ratios are unaffected and absolute figures carry a
# small sampler overhead. Do not compare these watts to a figure taken at
# a different interval.
INTERVAL_MS=200

[ -x "$RUST_BENCH" ] || { echo "missing or not executable: $RUST_BENCH" >&2; exit 2; }
[ -d "$MODEL" ] || { echo "missing install: $MODEL" >&2; exit 2; }

# A competing model process would show up as both a slow arm and other
# people's watts. Same guard parity.sh uses.
if pgrep -fl 'Mference(CLI|Server|Mac|DecodeService)|mference-(check|server|bench)|mlx' \
     | grep -v power.sh | grep -q .; then
  echo "another model process is running; results would be contaminated" >&2
  pgrep -fl 'Mference|mference|mlx' | grep -v power.sh >&2
  exit 2
fi

# LABEL is what the write-up keys its AC-vs-battery comparison on, and a
# mislabelled capture is worse than a missing one: it reads as a real
# effect. The provenance block below records `pmset -g ps` either way, but
# nothing would force anyone to read it, so disagree loudly here instead.
DRAWING=$(pmset -g ps | head -1)
case "$LABEL:$DRAWING" in
  ac:*Battery*)    echo "LABEL=ac but $DRAWING -- plug in, or fix LABEL" >&2; exit 2;;
  battery:*AC*)    echo "LABEL=battery but $DRAWING -- unplug, or fix LABEL" >&2; exit 2;;
esac

mkdir -p "$OUT"
: > "$OUT/rows.tsv"

now_ms() { python3 -c 'import time; print(int(time.time() * 1000))'; }

# Whole-system wall watts from the battery gauge: no root, and it is the
# only counter here that includes DRAM, SSD, display and the rest of the
# SoC. `Combined Power` below is CPU+GPU+ANE ONLY, so the two are not the
# same quantity and the wall figure should sit above it by a roughly
# constant offset. Discharging only -- on AC the amperage goes positive or
# zero and this column is correctly blank.
sample_battery() {
  python3 - "$OUT/batt.tsv" <<'PY' &
import plistlib, subprocess, sys, time
with open(sys.argv[1], "w") as f:
    while True:
        try:
            raw = subprocess.run(["ioreg", "-rn", "AppleSmartBattery", "-a"],
                                 capture_output=True).stdout
            d = plistlib.loads(raw)[0]
            amps, volts = d.get("InstantAmperage", 0), d.get("Voltage", 0)
            # Discharge reads negative; sign convention is not documented
            # as stable, so take the magnitude and gate on ExternalConnected.
            watts = 0.0 if d.get("ExternalConnected") else abs(amps) * volts / 1e6
            print(f"{int(time.time() * 1000)}\t{watts:.3f}", file=f, flush=True)
        except Exception:
            pass
        time.sleep(1)
PY
  BATT_PID=$!
}

cleanup() {
  [ -n "${BATT_PID:-}" ] && kill "$BATT_PID" 2>/dev/null
  [ -n "${KEEPALIVE_PID:-}" ] && kill "$KEEPALIVE_PID" 2>/dev/null
  sudo -n pkill -x powermetrics 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "powermetrics needs root; sudo will prompt once."
sudo -v || exit 2
# A full three-case run is well past sudo's 5-minute timestamp timeout,
# and the cleanup at the end has to still be able to kill a root process.
while true; do sudo -n true; sleep 60; done 2>/dev/null &
KEEPALIVE_PID=$!

# Provenance. A number without the machine state attached is not reusable.
{
  echo "date         $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "chip         $(sysctl -n machdep.cpu.brand_string)"
  echo "memory       $(($(sysctl -n hw.memsize) / 1073741824)) GB"
  echo "macos        $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
  echo "power        $(pmset -g ps | head -1)"
  echo "label        $LABEL"
  echo "read qos     ${QOS:-default}"
  echo "rev          $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)')"
  echo "install      $MODEL"
  echo "manifest sha $(shasum -a 256 "$MODEL/manifest.json" | cut -d' ' -f1)"
  echo "pairs        $PAIRS"
  echo "interval     ${INTERVAL_MS} ms"
} | tee "$OUT/system.txt"
echo

# Keep only the lines the integration reads. `--line-buffered` matters:
# without it grep block-buffers into the file and the tail of the capture
# is lost when powermetrics is killed.
PM_T0=$(now_ms)
sudo powermetrics -s cpu_power,gpu_power,thermal -i "$INTERVAL_MS" 2>/dev/null \
  | grep --line-buffered -E '^\*\*\* Sampled system activity|^CPU Power:|^GPU Power:|^Combined Power|^Current pressure level:|-Cluster HW active residency:' \
  > "$OUT/pm.txt" &
sample_battery
sleep 2

# One measured run of one case, in a fresh process. `--case` is the
# protocol's fresh-process leg; the bench does its own discarded warmup
# inside the process, and the markers exclude it.
run_arm() {
  local case_id="$1" tag="$2"
  ARM_QOS="$3"
  local stem="$OUT/${case_id}.${tag}"
  # "default" is the absence of the seam, so it maps to an unset class.
  local qos_env="$ARM_QOS"
  [ "$qos_env" = "default" ] && qos_env=""
  MFERENCE_READ_QOS="$qos_env" "$RUST_BENCH" --model "$MODEL" --case "$case_id" \
    > "$stem.stdout" 2> "$stem.stderr"

  local footer stop prompt_tok prefill_s new_tok decode_s tok_s win_start win_end
  footer=$(grep '^\[stop=' "$stem.stderr" | tail -1)
  win_start=$(sed -n 's/.*phase=start unix_ms=\([0-9]*\).*/\1/p' "$stem.stderr" | tail -1)
  win_end=$(sed -n 's/.*phase=end unix_ms=\([0-9]*\).*/\1/p' "$stem.stderr" | tail -1)
  if [ -z "$footer" ] || [ -z "$win_start" ] || [ -z "$win_end" ]; then
    echo "  $case_id/$tag: NO FOOTER OR NO MARKERS (see $stem.stderr)"
    echo "  markers come from crates/bench/src/main.rs; rebuild --release"
    return 1
  fi
  stop=$(sed -n 's/.*stop=\([a-zA-Z]*\).*/\1/p' <<< "$footer")
  prompt_tok=$(sed -n 's/.*prefill=\([0-9]*\)tok.*/\1/p' <<< "$footer")
  prefill_s=$(sed -n 's/.*prefill=[0-9]*tok\/\([0-9.]*\)s.*/\1/p' <<< "$footer")
  new_tok=$(sed -n 's/.*new=\([0-9]*\)tok.*/\1/p' <<< "$footer")
  decode_s=$(sed -n 's/.*decode=\([0-9.]*\)s.*/\1/p' <<< "$footer")
  tok_s=$(sed -n 's/.*tok\/s=\([0-9.]*\).*/\1/p' <<< "$footer")

  # Prefill runs first inside the window and decode fills the rest, so the
  # footer's own split is enough to cut the window in two: no third marker.
  local prefill_end
  prefill_end=$(python3 -c "print(int($win_start + $prefill_s * 1000))")

  integrate "$case_id" "$tag" "$stop" "$win_start" "$prefill_end" "$prefill_s" "$prompt_tok" prefill
  integrate "$case_id" "$tag" "$stop" "$prefill_end" "$win_end" "$decode_s" "$new_tok" decode

  printf '  %-8s %-10s stop=%-10s prefill %ss/%stok  decode %ss/%stok  %s tok/s\n' \
    "$case_id" "$tag" "$stop" "$prefill_s" "$prompt_tok" "$decode_s" "$new_tok" "$tok_s"
}

# Integrates one phase window out of the single powermetrics log and
# appends a row. A sample is counted when its MIDPOINT falls in the
# window; powermetrics prints each sample's header with the elapsed time
# of the interval that just ENDED, so accumulating those elapsed figures
# from PM_T0 rebuilds the absolute timeline without parsing any dates.
integrate() {
  local case_id="$1" tag="$2" stop="$3" w0="$4" w1="$5" secs="$6" toks="$7" phase="$8"
  awk -v t0="$PM_T0" -v w0="$w0" -v w1="$w1" -v secs="$secs" -v toks="$toks" \
      -v case_id="$case_id" -v tag="$tag" -v stop="$stop" -v phase="$phase" \
      -v label="$LABEL" -v qos="${ARM_QOS:-default}" -v battf="$OUT/batt.tsv" '
    function flush_sample() {
      if (!have) return
      mid = t_prev + (t_now - t_prev) / 2
      if (mid >= w0 && mid <= w1) {
        dt = (t_now - t_prev) / 1000
        j += (comb / 1000) * dt; cpu += cpu_mw * dt; gpu += gpu_mw * dt
        ecl += e_res * dt; pcl += p_res * dt; n++; secs_seen += dt
        if (press != "" && press != "Nominal") thermal = press
      }
      comb = cpu_mw = gpu_mw = e_res = p_res = 0; p_n = 0; press = ""
      seen_cpu = seen_gpu = 0
    }
    # "... (1015.93ms elapsed) ***" -- the third field from the end, with
    # its parenthesis and unit stripped. Deliberately not a 3-argument
    # match(): that is a gawk extension and macOS ships BWK awk.
    /^\*\*\* Sampled system activity/ {
      flush_sample()
      elapsed = $(NF - 2); gsub(/[()a-zA-Z]/, "", elapsed)
      t_prev = (have ? t_now : t0)
      t_now = t_prev + elapsed
      have = 1
      next
    }
    # GPU Power is printed twice per sample (cpu_power block and gpu usage
    # block); take the first and ignore the echo.
    /^CPU Power:/      { if (!seen_cpu) { cpu_mw = $3; seen_cpu = 1 } ; next }
    /^GPU Power:/      { if (!seen_gpu) { gpu_mw = $3; seen_gpu = 1 } ; next }
    /^Combined Power/  { comb = $(NF - 1); next }
    /^Current pressure level:/ { press = $4; next }
    /^E-Cluster HW active residency:/ { e_res = $5 + 0; next }
    # M4 Max has TWO performance clusters (P0, P1); average them.
    /^P[0-9]-Cluster HW active residency:/ { p_res = (p_res * p_n + ($5 + 0)) / (p_n + 1); p_n++; next }
    END {
      flush_sample()
      if (n == 0) { print "  NO SAMPLES in " phase " window" > "/dev/stderr"; exit }
      # Wall watts from the battery gauge over the same window, when the
      # machine was discharging. Blank on AC by construction.
      bn = 0; bw = 0
      while ((getline line < battf) > 0) {
        split(line, b, "\t")
        if (b[1] >= w0 && b[1] <= w1 && b[2] + 0 > 0) { bw += b[2]; bn++ }
      }
      printf "%s\t%s\t%s\t%s\t%s\t%.2f\t%s\t%.1f\t%.2f\t%.4f\t%.0f\t%.0f\t%.1f\t%.1f\t%s\t%d\t%s\n",
        case_id, label, qos, tag, phase, secs, toks,
        j, j / secs_seen, (toks > 0 ? j / toks : 0),
        cpu / secs_seen, gpu / secs_seen, ecl / secs_seen, pcl / secs_seen,
        (thermal == "" ? "Nominal" : thermal), n,
        (bn > 0 ? sprintf("%.2f", bw / bn) : "")
    }
  ' "$OUT/pm.txt" >> "$OUT/rows.tsv"
}

# QOS may name SEVERAL arms, comma separated (`QOS=default,utility`).
# They alternate WITHIN each pair rather than running as two consecutive
# batches, because consecutive batches carry thermal drift and the paired
# delta is the only comparison worth making (CLAUDE.local.md).
IFS=',' read -ra QOS_ARMS <<< "$QOS"

for case_id in $CASES; do
  echo "== $case_id"
  # One discarded arm per case. The bench warms the GPU inside its own
  # process, but the FIRST process of a session also pays Metal pipeline
  # compilation and a cold page cache for this install's expert blobs
  # (AGENTS.md Gotcha 20).
  run_arm "$case_id" warmup "${QOS_ARMS[0]}"
  for ((p = 1; p <= PAIRS; p++)); do
    for arm in "${QOS_ARMS[@]}"; do
      run_arm "$case_id" "p$p.$arm" "$arm"
    done
  done
  echo
done

PM_T_END=$(now_ms)
cleanup
trap - EXIT INT TERM
sleep 1

# The windows are joined on a timeline reconstructed as PM_T0 plus the
# accumulated per-sample elapsed figures, because powermetrics timestamps
# its samples only to the second. That reconstruction is only as good as
# its drift against the wall clock, and drift would silently misalign the
# LAST arms of a long run while leaving the first ones correct. So measure
# it rather than assume it: the reconstructed end of the log should land
# on the wall-clock time the sampler was killed.
awk -v t0="$PM_T0" -v tend="$PM_T_END" '
  /^\*\*\* Sampled system activity/ { e = $(NF - 2); gsub(/[()a-zA-Z]/, "", e); sum += e; n++ }
  END {
    drift = (t0 + sum) - tend
    printf "timeline: %d samples, reconstructed span %.1fs, wall span %.1fs, drift %.1fs\n", \
      n, sum / 1000, (tend - t0) / 1000, drift / 1000
    if (drift < -2000 || drift > 2000)
      printf "WARNING drift over 2s: late windows may be misaligned; treat J/tok as approximate\n"
  }
' "$OUT/pm.txt"
echo

echo "== measured rows (warmups excluded), $LABEL, read qos ${QOS:-default}"
awk -F'\t' '
  $4 != "warmup" {
    # Grouped by QoS arm as well as case and phase: the arms alternate
    # within a pair, so averaging across them would erase the comparison.
    k = $1 "\t" $3 "\t" $5
    # Too few samples means the window is short against the interval and
    # the edge error stops being negligible.
    if ($16 < 3) little[$1 "/" $4 "/" $5] = $16
    if ($15 != "Nominal") thermal[$1 "/" $4] = $15
    secs[k] += $6; j[k] += $8; w[k] += $9; jt[k] += $10
    cpu[k] += $11; gpu[k] += $12; e[k] += $13; p[k] += $14
    if ($17 != "") { wall[k] += $17; walln[k]++ }
    n[k]++
  }
  END {
    printf "%-18s %-8s %-8s %5s %8s %9s %9s %8s %8s %7s %7s %9s\n", \
      "case", "qos", "phase", "runs", "secs", "joules", "watts", "J/tok", \
      "cpu_W", "gpu_W", "E%", "wall_W"
    for (k in n) {
      split(k, f, "\t")
      printf "%-18s %-8s %-8s %5d %8.2f %9.1f %9.2f %8.4f %8.2f %7.2f %7.1f %9s\n", \
        f[1], f[2], f[3], n[k], secs[k] / n[k], j[k] / n[k], w[k] / n[k], jt[k] / n[k], \
        cpu[k] / n[k] / 1000, gpu[k] / n[k] / 1000, e[k] / n[k], \
        (walln[k] > 0 ? sprintf("%.2f", wall[k] / walln[k]) : "n/a")
    }
    for (k in thermal) printf "WARNING %s left Nominal thermal pressure (%s)\n", k, thermal[k]
    for (k in little) printf "WARNING %s integrated only %d samples; shorten the interval\n", k, little[k]
  }
' "$OUT/rows.tsv" | sort

echo
echo "watts and J/tok are CPU+GPU+ANE (powermetrics Combined Power), NOT wall."
echo "wall_W is the battery gauge, which does include DRAM/SSD/display."
echo "raw rows: $OUT/rows.tsv   samples: $OUT/pm.txt"
