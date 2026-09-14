#!/bin/bash
# Power baseline for one real install, over the frozen community protocol
# (ROADMAP Phase P1). Reports average watts and joules-per-token, split
# prefill vs decode, plus the residency and thermal columns the hygiene
# audit needs.
#
# Usage: scripts/power.sh [pairs]        (default 2 measured pairs per case)
# Env:   MODEL, RUST_BENCH, OUT, LABEL (ac|battery),
#        ARMS (comma-separated arms to interleave; see ARMS below;
#              `nospec,spec` prices a speculative drafter, both arms greedy),
#        QOS (the Phase P1 spelling of ARMS, still honored),
#        CASES (space-separated protocol case ids),
#        COOLING (auto|max; see COOLING below)
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
# ARMS names the comparison axis, comma separated. THREE kinds of token are
# understood, because the arm name is a single column in rows.tsv:
#   default, utility                        -> TURBOSPARK_READ_QOS (Phase P1)
#   performance, balanced, efficiency       -> --power-profile   (Phase P2)
#   a bare number, e.g. 30 or 12.5          -> --max-tokens-per-sec
# "default" means "vary nothing", spelled as a token rather than as the
# empty string: macOS ships bash 3.2, where `read -ra` on an empty line
# leaves a ZERO-element array and indexing it under `set -u` aborts.
# QOS is still read so the Phase P1 invocations reproduce unchanged.
#
# THE QoS AXIS MAY NOT BE MIXED WITH THE OTHER TWO, and that is refused
# below rather than left to this comment: `utility` sets an environment
# variable where the others pass a flag, so a run containing both varies
# two things at once and the single arm column cannot say which.
#
# THE NUMERIC ARMS ARE THE RATE-CAP SWEEP (`ARMS=default,30,20,15,10`), and
# `default` doubles as their uncapped reference, so they need no new token.
# What makes them a clean axis is that the bench's profile stays at its
# `performance` default, and `runtime::rate_control_for` then builds a cap
# with NO thermal probe -- a numeric arm measures pacing alone, with the
# ladder out of the picture. They exist because the shipped `efficiency`
# cap of 10 tok/s measured 14.9% of the energy for 51.5% of the throughput
# and there were only TWO points on that curve to place it against
# (docs/POWER_BASELINE.md, "Forced cooling: the A/B, run").
#
# One cosmetic wart: the summary at the bottom pipes through `sort`, which
# orders the arm column lexically. Caps of equal digit width therefore read
# in ascending order and a run mixing `5` with `30` does not.
ARMS="${ARMS:-${QOS:-default}}"
CASES="${CASES:-short-explanation medium-review long-synthesis}"
# COOLING is an axis ORTHOGONAL to ARMS, and it gets its own column rather
# than an ARMS token for that reason: ARMS is one column and one comparison,
# while cooling is a property of the whole capture, like LABEL.
#   auto  the machine's own fan curve. Every row in docs/POWER_BASELINE.md
#         published before 2026-08-18 was taken this way.
#   max   fans pinned to 100% via ThermalForge (MIT, github.com/ProducerGuy/
#         ThermalForge) for the duration of the capture, restored on exit.
#
# WHY: this laptop cannot hold Nominal thermal pressure in performance mode
# for a whole case, so the Phase P2 performance-vs-efficiency A/B is
# inconclusive -- the performance arm spans 25% across byte-identical work
# because it is a thermal control loop's output rather than the workload's
# (docs/POWER_BASELINE.md, "the comparison does not"). That section names
# the missing condition outright: "A fair A/B needs hardware that can hold
# Nominal in performance mode". Pinned fans ARE that condition.
#
# WHAT IT DOES NOT DO: save power. Fans cost watts and a cooler chip boosts
# to a higher V/f point. A COOLING=max row is "energy per token with
# unlimited cooling", an upper-headroom operating point, NOT what a user on
# a real machine sees. Publish it beside the auto rows, never instead.
#
# The J/token column is unaffected either way: powermetrics Combined Power
# is CPU+GPU+ANE only, so fan draw never enters it. Fan draw DOES enter
# wall_W (the battery gauge), which is blank on AC by construction and is
# already documented as too noisy to publish (sd 20-40% of its own mean).
COOLING="${COOLING:-auto}"

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

# Classifies one ARMS token. ONE function, called by both the guard below
# and `run_arm`'s dispatch, because a guard that classifies differently
# from the code it guards is not a guard -- it would admit a token the
# dispatcher goes on to refuse, mid-capture and after the fan pin.
arm_kind() {
  case "$1" in
    default)                             echo neutral ;;
    utility)                             echo qos ;;
    performance | balanced | efficiency) echo profile ;;
    nospec | spec)                       echo spec ;;
    seq | chunked)                       echo chunk ;;
    *)
      # A bare number is a --max-tokens-per-sec cap. `=~` rather than a
      # `*[!0-9.]*` glob, which would accept "1.2.3"; bash 3.2 has `=~`
      # and needs the pattern UNQUOTED. The awk test is what rejects 0,
      # 0.0 and 00 alike, and it mirrors the bench's own `is_finite() &&
      # > 0.0` rather than inventing a second rule.
      if [[ $1 =~ ^[0-9]+([.][0-9]+)?$ ]] &&
         awk -v v="$1" 'BEGIN { exit !(v > 0) }'; then
        echo cap
      else
        echo unknown
      fi
      ;;
  esac
}

# ARMS is validated HERE -- before `sudo -v`, before the fans are pinned,
# and before a single token is generated.
#
# It used to be validated only inside `run_arm`, which echoes and returns 1
# on an unknown arm; the loop at the bottom ignores that return, so a typo
# produced a capture that ran for twenty minutes, exited 0, printed a
# summary, and was silently missing one arm of the comparison. A result
# that reads as complete and is not is worse than a failure, and it is the
# same species as a COOLING=max that pinned nothing: the guard has to sit
# where the mistake is made, not where its consequence shows up.
#
# It also sits AHEAD of the binary and install checks below, which is not
# the obvious order. Those probe the ENVIRONMENT; this validates an env var
# the caller typed, and needs nothing built to do it. Keeping it first is
# what makes `ARMS=bogus scripts/power.sh` a one-second test on any
# checkout -- and a guard nobody can cheaply exercise is how this class of
# bug survives in the first place.
IFS=',' read -ra ARM_LIST <<< "$ARMS"
SAW_QOS=""
SAW_FLAG=""
SAW_SPEC=""
SAW_CHUNK=""
SAW_OTHER=""
for arm in "${ARM_LIST[@]}"; do
  case "$(arm_kind "$arm")" in
    neutral) SAW_OTHER=1 ;;
    qos) SAW_QOS=1; SAW_OTHER=1 ;;
    profile | cap) SAW_FLAG=1; SAW_OTHER=1 ;;
    spec) SAW_SPEC=1 ;;
    chunk) SAW_CHUNK=1 ;;
    *)
      echo "unknown arm '$arm' in ARMS=$ARMS" >&2
      echo "want default|utility|performance|balanced|efficiency|nospec|spec," >&2
      echo "seq|chunked, or a positive number" >&2
      exit 2
      ;;
  esac
done
if [ -n "$SAW_QOS" ] && [ -n "$SAW_FLAG" ]; then
  echo "ARMS=$ARMS mixes the QoS axis (utility) with a profile or a rate cap" >&2
  echo "one axis per capture: the arm is a single column in rows.tsv" >&2
  exit 2
fi
# THE SPECULATION AXIS IS EXCLUSIVE OF EVERY OTHER ARM, `default` INCLUDED,
# and that last clause is the one worth stating. `nospec` is not `default`:
# both spec arms run `--shaping greedy`, because acceptance is exact only at
# temperature 0 and the frozen protocol samples at 0.2. So `ARMS=default,spec`
# would vary the SHAPING and the SPECULATION together and produce a delta
# nobody can attribute -- which is the same mistake, one layer up, as folding
# greedy into `--speculative` itself.
if [ -n "$SAW_SPEC" ] && [ -n "$SAW_OTHER" ]; then
  echo "ARMS=$ARMS mixes the speculation axis (nospec|spec) with another arm" >&2
  echo "the spec arms run --shaping greedy and the others do not, so pairing" >&2
  echo "them varies two things; use ARMS=nospec,spec on its own" >&2
  exit 2
fi
# THE PREFILL AXIS IS EXCLUSIVE TOO, and its reason is NOT the speculation
# axis's. `nospec` differs from `default` by `--shaping greedy`, so pairing
# them varies two things. `seq` is byte-for-byte the SAME invocation as
# `default`, so pairing THEM varies nothing at all and just puts one
# condition in two rows, which reads as a comparison. Different mechanism,
# same verdict.
#
# The `SAW_SPEC` clause is load-bearing and easy to drop: `chunk` sets
# neither SAW_OTHER nor SAW_FLAG, so the speculation guard above does NOT
# catch `ARMS=spec,chunked`. Without this clause that pair reaches the bench,
# which refuses `--speculative` alongside `--prefill-chunk` -- mid-capture,
# after `sudo -v` and after the fans are pinned, which is precisely what
# validating ARMS up here exists to prevent.
if [ -n "$SAW_CHUNK" ] && { [ -n "$SAW_OTHER" ] || [ -n "$SAW_SPEC" ]; }; then
  echo "ARMS=$ARMS mixes the prefill axis (seq|chunked) with another arm" >&2
  echo "seq is the SAME invocation as default, so naming both puts one" >&2
  echo "condition in two rows; and --prefill-chunk is refused alongside" >&2
  echo "--speculative. Use ARMS=seq,chunked on its own" >&2
  exit 2
fi

[ -x "$RUST_BENCH" ] || { echo "missing or not executable: $RUST_BENCH" >&2; exit 2; }
[ -d "$MODEL" ] || { echo "missing install: $MODEL" >&2; exit 2; }

# A competing model process would show up as both a slow arm and other
# people's watts. Include the renamed binaries and desktop app: the old
# Mference-only pattern silently admitted this engine's own competitors.
MODEL_PROCESSES='Mference(CLI|Server|Mac|DecodeService)|mference-(check|server|bench)|turbospark-(check|server|bench)|TurboSparkApp|mlx'
if pgrep -fl "$MODEL_PROCESSES" \
     | grep -v power.sh | grep -q .; then
  echo "another model process is running; results would be contaminated" >&2
  pgrep -fl "$MODEL_PROCESSES" | grep -v power.sh >&2
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

# Same reasoning as the LABEL guard immediately above: a capture that SAYS
# it pinned the fans and did not is worse than no capture, because it reads
# as a real effect. Refuse rather than fall back to auto.
case "$COOLING" in
  auto) ;;
  max)
    command -v thermalforge >/dev/null 2>&1 || {
      echo "COOLING=max needs the thermalforge binary, which is not on PATH" >&2
      echo "install it (github.com/ProducerGuy/ThermalForge), or use COOLING=auto" >&2
      exit 2
    }
    ;;
  *) echo "unknown COOLING=$COOLING (want auto|max)" >&2; exit 2;;
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
  if [ -n "${PROCESS_PID:-}" ]; then
    kill "$PROCESS_PID" 2>/dev/null
    wait "$PROCESS_PID" 2>/dev/null
    PROCESS_PID=""
  fi
  [ -n "${BATT_PID:-}" ] && kill "$BATT_PID" 2>/dev/null
  [ -n "${KEEPALIVE_PID:-}" ] && kill "$KEEPALIVE_PID" 2>/dev/null
  sudo -n pkill -x powermetrics 2>/dev/null
  # Restoring the fan curve is the one cleanup step whose omission leaves
  # the MACHINE in a bad state rather than just a stray process, so it is
  # gated on a flag set at the pin site: this runs on every exit path,
  # including the guards above that abort before anything was pinned.
  if [ -n "${FANS_PINNED:-}" ]; then
    if thermalforge auto >/dev/null 2>&1; then
      echo "fans restored to the machine's own curve"
      FANS_PINNED=""
    else
      echo "WARNING could not restore fans; run 'thermalforge auto' by hand" >&2
    fi
  fi
}
trap cleanup EXIT INT TERM

# Pinned AFTER the trap is armed, never before: between the pin and the
# trap there is no handler, so an interrupt in that window would leave the
# fans at 100% with nothing left running to put them back.
if [ "$COOLING" = max ]; then
  echo "pinning fans to maximum for the capture (COOLING=max)"
  # Assume restoration is needed before invoking the effectful command:
  # ThermalForge may change the fan state before failing or being interrupted.
  FANS_PINNED=1
  thermalforge max || { echo "thermalforge max failed" >&2; exit 2; }
  # The fans must reach speed before the first sample, or the early arms
  # are measured mid-ramp and the capture is not the single operating
  # point it claims to be. MEASURED on Mac16,5 2026-08-18: 1350 -> 5763
  # and 1451 -> 5689 RPM within 5 s against a 5777 target, so 10 s is
  # double the observed ramp. Re-measure on other hardware rather than
  # carrying this constant across (`thermalforge status` reports both
  # actual and target RPM, so the check is one command).
  sleep 10
fi

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
  echo "arms         ${ARMS}"
  echo "cooling      ${COOLING}$([ "$COOLING" = max ] && echo ' (fans pinned; NOT a shipping operating point)')"
  echo "rev          $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)')"
  echo "install      $MODEL"
  echo "manifest sha $(shasum -a 256 "$MODEL/manifest.json" | cut -d' ' -f1)"
  echo "pairs        $PAIRS"
  echo "interval     ${INTERVAL_MS} ms"
  echo "process log  processes.jsonl (1 s pause between snapshots; ps decayed CPU and cumulative CPU time)"
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
python3 "$(dirname "$0")/power_processes.py" "$OUT/processes.jsonl" \
  2> "$OUT/processes.stderr" &
PROCESS_PID=$!
sleep 2
kill -0 "$PROCESS_PID" 2>/dev/null || { echo "process sampler failed; see $OUT/processes.stderr" >&2; exit 2; }

# One measured run of one case, in a fresh process. `--case` is the
# protocol's fresh-process leg; the bench does its own discarded warmup
# inside the process, and the markers exclude it.
run_arm() {
  local case_id="$1" tag="$2"
  ARM="$3"
  local stem="$OUT/${case_id}.${tag}"
  # Each arm token names ONE seam to vary; everything else stays at its
  # default, so "default" runs the binary exactly as it ships.
  local qos_env=""
  local arm_args=()
  case "$(arm_kind "$ARM")" in
    neutral) ;;
    qos) qos_env="utility" ;;
    profile) arm_args=(--power-profile "$ARM") ;;
    cap) arm_args=(--max-tokens-per-sec "$ARM") ;;
    # BOTH arms are greedy; only `--speculative` differs. The bench refuses
    # `--speculative` without `--shaping greedy` up front, so a drifted
    # invocation here fails in a second rather than after a 20 s open.
    spec)
      if [ "$ARM" = "spec" ]; then
        arm_args=(--shaping greedy --speculative auto)
      else
        arm_args=(--shaping greedy)
      fi
      ;;
    # `seq` passes NOTHING and is deliberately the identical invocation to
    # `default`: the arm column is the only thing telling the two rows
    # apart, and that identity is what makes this a one-variable
    # comparison. `auto` resolves to DEFAULT_CHUNK_SIZE inside the bench,
    # which echoes the resolved number in its header.
    #
    # TURBOSPARK_ROUTED_BATCH and TURBOSPARK_BATCHED_GEMV are NOT set here.
    # They are read inside the runtime's chunk drivers, so they ride
    # whatever the caller exported and the bench header reports them on
    # both arms. Setting one here would fold a second variable into this
    # axis.
    chunk)
      if [ "$ARM" = "chunked" ]; then
        arm_args=(--prefill-chunk auto)
      fi
      ;;
    *)
      # Unreachable: the guard above refuses an unknown arm before the
      # capture starts. Kept as a backstop rather than deleted, because
      # its absence is what let a bad token get this far in the first
      # place -- but it must never be how a typo is DISCOVERED.
      echo "  unknown arm $ARM reached run_arm; the ARMS guard did not fire"
      return 1
      ;;
  esac
  # bash 3.2 aborts under `set -u` on "${arr[@]}" when arr is EMPTY, hence
  # the +expansion guard rather than a bare splat.
  TURBOSPARK_READ_QOS="$qos_env" "$RUST_BENCH" --model "$MODEL" --case "$case_id" \
    ${arm_args[@]+"${arm_args[@]}"} \
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
      -v label="$LABEL" -v qos="${ARM:-default}" -v battf="$OUT/batt.tsv" \
      -v cooling="$COOLING" '
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
      # `cooling` is APPENDED as field 18 rather than slotted in beside
      # `label`, where it belongs logically: the summary below indexes
      # $15/$16/$17 positionally, so inserting a column mid-row would move
      # the thermal and sample-count reads onto the wrong fields and the
      # warnings would go quiet instead of red.
      printf "%s\t%s\t%s\t%s\t%s\t%.2f\t%s\t%.1f\t%.2f\t%.4f\t%.0f\t%.0f\t%.1f\t%.1f\t%s\t%d\t%s\t%s\n",
        case_id, label, qos, tag, phase, secs, toks,
        j, j / secs_seen, (toks > 0 ? j / toks : 0),
        cpu / secs_seen, gpu / secs_seen, ecl / secs_seen, pcl / secs_seen,
        (thermal == "" ? "Nominal" : thermal), n,
        (bn > 0 ? sprintf("%.2f", bw / bn) : ""), cooling
    }
  ' "$OUT/pm.txt" >> "$OUT/rows.tsv"
}

# ARMS may name SEVERAL arms, comma separated
# (`ARMS=performance,efficiency`, `ARMS=default,30,20,15,10`). They
# alternate WITHIN each pair rather than running as two consecutive
# batches, because consecutive batches carry thermal drift and the paired
# delta is the only comparison worth making (CLAUDE.local.md). ARM_LIST was
# split and validated up at the guards, before anything was pinned.
for case_id in $CASES; do
  echo "== $case_id"
  # One discarded arm per case. The bench warms the GPU inside its own
  # process, but the FIRST process of a session also pays Metal pipeline
  # compilation and a cold page cache for this install's expert blobs
  # (AGENTS.md Gotcha 20).
  run_arm "$case_id" warmup "${ARM_LIST[0]}"
  for ((p = 1; p <= PAIRS; p++)); do
    for arm in "${ARM_LIST[@]}"; do
      run_arm "$case_id" "p$p.$arm" "$arm"
    done
  done
  echo
done

PM_T_END=$(now_ms)
kill -0 "$PROCESS_PID" 2>/dev/null || echo "WARNING process sampler stopped early; inspect processes.stderr" >&2
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

# THE CONTAMINATION FLOOR, and it is the tell DISPERSION CANNOT PROVIDE.
#
# AGENTS.md Gotcha 43's two tells are `cpu_W` against the install's own norm
# and dispersion between arms doing identical work. Dispersion caught the
# 2026-08-13 gpt-oss capture because that load DRIFTED: the desktop UI was
# busy early and idle late, so the two arms disagreed by 37%. A STEADY load
# contaminates both arms equally, so dispersion reads clean and only the norm
# comparison fires -- which needs a norm, i.e. a previous clean capture of the
# same install. The first capture of a new install has no norm at all.
#
# The minimum CPU power seen ANYWHERE in the log does not need one. The
# capture spans model opens, settling gaps and the pauses between arms, so a
# quiet machine touches near-idle at some point and a contaminated one never
# does. The 2026-08-21 ornith35b capture read a minimum of 3,361 mW across 484
# samples with a median of 10,154 -- no sample under 3 W in 106 seconds --
# while its arms agreed to 0.18% on `gpu_W`. Every published row here should
# have been checked this way and none was.
awk '
  /^CPU Power:/ { if ($3 + 0 < lo || n == 0) lo = $3 + 0; n++ }
  END {
    if (n == 0) { print "contamination floor: no CPU Power samples"; exit }
    printf "contamination floor: %d mW minimum CPU power over %d samples\n", lo, n
    # 2000 mW is CALIBRATED against three real captures on this machine, not
    # picked: the two clean DFlash2 ones read 94 mW (883 samples) and 392 mW
    # (1,363 samples, fans pinned), and the contaminated ornith35b one read
    # 3,361 mW (484). Nearly an order of magnitude of separation, and the
    # threshold sits in the gap. It is not zero because `powermetrics` itself
    # wakes 5x/s and the harness runs a shell between arms.
    if (lo > 2000) {
      printf "WARNING the CPU never fell below %.2f W anywhere in this capture,\n", lo / 1000
      print  "        including the gaps between generations. That is a STEADY"
      print  "        background load, which the per-arm dispersion below cannot"
      print  "        see because it contaminates every arm equally. `watts` and"
      print  "        `J/tok` are Combined Power and carry it; `gpu_W` and tok/s"
      print  "        are largely insulated. Find the consumer (`ps -A -o %cpu,comm -r"
      print  "        | head`) and re-run before publishing any energy row."
    }
  }
' "$OUT/pm.txt"
echo

echo "== measured rows (warmups excluded), $LABEL, arms ${ARMS}, cooling ${COOLING}"
awk -F'\t' '
  $4 != "warmup" {
    # Grouped by ARM as well as case and phase: the arms alternate
    # within a pair, so averaging across them would erase the comparison.
    k = $1 "\t" $3 "\t" $5
    # Too few samples means the window is short against the interval and
    # the edge error stops being negligible.
    if ($16 < 3) little[$1 "/" $4 "/" $5] = $16
    if ($15 != "Nominal") thermal[$1 "/" $4] = $15
    secs[k] += $6; j[k] += $8; w[k] += $9; jt[k] += $10
    cpu[k] += $11; gpu[k] += $12; e[k] += $13; p[k] += $14
    if ($17 != "") { wall[k] += $17; walln[k]++ }
    # PER-ARM EXTREMES, not just the mean. A mean over two arms doing
    # IDENTICAL work hides the one thing worth knowing about them, which is
    # whether they agreed: the 2026-08-13 gpt-oss capture averaged 1.7072
    # and 1.0799 J/tok into 1.3936, a number describing neither run, and the
    # spread was recoverable only by reading rows.tsv by hand.
    if (n[k] == 0 || $10 + 0 < jtlo[k]) jtlo[k] = $10 + 0
    if (n[k] == 0 || $10 + 0 > jthi[k]) jthi[k] = $10 + 0
    n[k]++
  }
  END {
    printf "%-18s %-8s %-8s %5s %8s %9s %9s %8s %7s %8s %7s %7s %9s\n", \
      "case", "arm", "phase", "runs", "secs", "joules", "watts", "J/tok", \
      "J/tok±", "cpu_W", "gpu_W", "E%", "wall_W"
    for (k in n) {
      split(k, f, "\t")
      # The spread the mean beside it hides, as a percentage of the mean.
      spread = (n[k] > 1 && jt[k] > 0) \
        ? 100 * (jthi[k] - jtlo[k]) / (jt[k] / n[k]) : 0
      printf "%-18s %-8s %-8s %5d %8.2f %9.1f %9.2f %8.4f %6.1f%% %8.2f %7.2f %7.1f %9s\n", \
        f[1], f[2], f[3], n[k], secs[k] / n[k], j[k] / n[k], w[k] / n[k], jt[k] / n[k], \
        spread, \
        cpu[k] / n[k] / 1000, gpu[k] / n[k] / 1000, e[k] / n[k], \
        (walln[k] > 0 ? sprintf("%.2f", wall[k] / walln[k]) : "n/a")
      # 10% is where a spread stops being run-to-run noise on this machine:
      # a clean capture reproduces to 0.4-2%, and the contaminated gpt-oss
      # one read 37%. A short window trips it for a different reason (too
      # few samples), which is why the sample-count warning stays separate.
      if (spread > 10) wide[f[1] "/" f[2] "/" f[3]] = spread
    }
    for (k in thermal) printf "WARNING %s left Nominal thermal pressure (%s)\n", k, thermal[k]
    for (k in little) printf "WARNING %s integrated only %d samples; shorten the interval\n", k, little[k]
    # ONE LINE, because this block is piped through `sort`: a continuation
    # line sorts away from the warning it belongs to.
    for (k in wide) \
      printf "WARNING %s J/tok spread %.1f%% across runs doing IDENTICAL work; the mean describes neither, read rows.tsv per arm\n", k, wide[k]
  }
' "$OUT/rows.tsv" | sort

echo
echo "watts and J/tok are CPU+GPU+ANE (powermetrics Combined Power), NOT wall."
echo "wall_W is the battery gauge, which does include DRAM/SSD/display."
if [ "$COOLING" = max ]; then
  echo
  echo "COOLING=max: fans were pinned. These rows are an UPPER-HEADROOM operating"
  echo "point (energy per token with unlimited cooling), not what a user sees."
  echo "Publish them BESIDE the COOLING=auto rows, never in place of them."
  echo "Fan draw is not in the J/tok column above; it lands in wall_W."
fi
echo "raw rows: $OUT/rows.tsv   samples: $OUT/pm.txt"
