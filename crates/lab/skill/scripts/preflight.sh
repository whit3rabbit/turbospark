#!/bin/sh
# Check that this host is quiet enough for tslab to produce clean runs. Read-only:
# it reports, it kills and fixes nothing. Exit 0 when nothing blocks, 1 when a
# run would be ledgered as dirty.
#
# tslab's own host guard (crates/lab/src/host.rs) rejects dirty runs after the
# fact. This looks first, so you do not spend a long round to learn that a
# build was running. Its thresholds copy the guard's defaults.
set -u

fail=0

ok() { printf '  ok    %s\n' "$1"; }
bad() { printf '  FAIL  %s\n' "$1"; fail=1; }
note() { printf '  note  %s\n' "$1"; }

echo "tslab preflight"

if pmset -g batt 2>/dev/null | grep -q "AC Power"; then
    ok "on AC power"
else
    bad "not on AC power: plug in, or pass --allow-battery to tslab and say so"
fi

# Processes the guard counts as foreign (host.rs MODEL_PROCESS_NAMES and the
# rustc check), plus cargo and xctrace, which start them. Matched on the
# first word of the command line's basename, so a path containing "cargo" does
# not match. `comm` is cut to 16 characters on macOS, so it cannot be used.
busy=$(ps -axwwo command= 2>/dev/null | awk '{print $1}' | sed 's|.*/||' | sort | uniq -c | awk '
    $2 ~ /^(rustc|cargo|xctrace|tslab|mlx_lm|mlx-lm|llama-server|llama-cli|mference|ollama|kapitsa|turbospark-server|turbospark-bench|turbospark-check|turbospark-startup-probe|tslab-worker|memory_oracle|quality_gate)$/ {
        printf "%s x%s ", $2, $1
    }')
if [ -z "$busy" ]; then
    ok "no build, trace or other model process running"
else
    bad "busy processes: $busy(finish or stop them; tslab-worker here may be another session's run)"
fi

LAB=${TURBOSPARK_LAB_DIR:-${TURBOSPARK_HOME:-$HOME/.turbospark}/lab}
# df on a path that does not exist yet fails, so check the nearest parent.
probe=$LAB
while [ ! -d "$probe" ] && [ "$probe" != / ]; do probe=$(dirname "$probe"); done
# Release builds are about 5 GB per worktree, and the guard refuses below 10 GiB
# free on the lab volume (--min-free-gb).
MIN_FREE_GB=10
free_kb=$(df -k "$probe" 2>/dev/null | awk 'NR==2 {print $4}')
if [ -n "$free_kb" ] && [ "$free_kb" -ge $((MIN_FREE_GB * 1024 * 1024)) ]; then
    ok "free disk on $probe: $((free_kb / 1024 / 1024)) GiB"
elif [ -n "$free_kb" ]; then
    bad "free disk on $probe: $((free_kb / 1024 / 1024)) GiB, under $MIN_FREE_GB GiB"
else
    note "could not read free disk on $probe"
fi

# Swap is recorded with each run but does not reject one. A cold page cache or
# swap in use is still why a kernel can look slow, so show it.
swap=$(sysctl -n vm.swapusage 2>/dev/null | sed -n 's/.*used = \([0-9.]*[MG]\).*/\1/p')
if [ -n "$swap" ]; then note "swap used: $swap (nonzero swap means memory pressure, read timings with care)"; fi

# The machine lock is a flock on <lab>/.machine.lock. lsof lists its holder.
if command -v lsof >/dev/null 2>&1 && [ -f "$LAB/.machine.lock" ]; then
    holder=$(lsof -t "$LAB/.machine.lock" 2>/dev/null | head -n 1)
    if [ -n "$holder" ]; then note "machine lock is held by pid $holder: a second tslab session will wait"; fi
fi

if [ "$fail" = 0 ]; then echo "ready to measure"; else echo "not ready: runs would be dirty"; fi
exit $fail
