# Power: watts and joules per token

Split out of the repository's `AGENTS.md`, which is loaded into every
session; this page is loaded when you follow the link. Nothing here is a
new fact, and `AGENTS.md` remains the map.

`scripts/power.sh` needs SUDO (powermetrics is root-only) and so cannot run
non-interactively. ~12 min per install. Before believing any row, read
`AGENTS.md` Gotchas 22, 28 and 43 together: thermal pressure rewrites both
throughput and energy and makes a throttled arm look GOOD, `powermetrics`
measures the MACHINE rather than your process, and a steady background load
is invisible to the dispersion check. `docs/POWER_BASELINE.md` holds the rows.

All commands run from the REPOSITORY ROOT, not from this directory.

```sh
# Power baseline over the frozen protocol (ROADMAP Phase P1): watts and
# joules-per-token, split prefill/decode. NEEDS SUDO (powermetrics is
# root-only) and so cannot be run non-interactively. ~12 min per install.
# Windows the capture with the `[power-window ...]` markers turbospark-bench
# emits, so the model open and the discarded warmup stay out of the total.
# Numbers and caveats: docs/BENCHMARKS.md.
LABEL=battery OUT=/tmp/power-gemma MODEL=~/models/gemma4.gturbo scripts/power.sh 2

# Same harness driving an interleaved A/B. Arms alternate WITHIN each
# pair, not as two consecutive batches, because consecutive batches carry
# thermal drift. `ARMS` names the axis and understands THREE kinds of
# token: `default`/`utility` set MFERENCE_READ_QOS (Phase P1, measured and
# rejected -- it loses on both joules and tok/s, kept as a documented dead
# end), `performance`/`balanced`/`efficiency` pass --power-profile
# (Phase P2), and a BARE NUMBER passes --max-tokens-per-sec. The QoS axis
# may not be mixed with the other two and the script refuses it (the arm is
# one column of rows.tsv, so a run varying two things cannot say which).
# An unknown arm is refused UP FRONT, before sudo and before the fans are
# pinned: `ARMS=bogus scripts/power.sh` is a one-second test that needs
# nothing built. `QOS` is still read as the old spelling of `ARMS`.
LABEL=battery MODEL=~/models/gemma4.gturbo CASES=short-explanation \
  ARMS=default,utility scripts/power.sh 3

# The Phase P2 gate: does the efficiency profile actually buy joules per
# token, and does performance stay where docs/POWER_BASELINE.md left it?
# Budget for it: an efficiency arm decodes at ~10 tok/s, so its window is
# several times longer than a performance arm's.
LABEL=ac MODEL=~/models/gemma4.gturbo CASES=short-explanation \
  ARMS=performance,efficiency scripts/power.sh 3

# What a speculative DRAFTER costs in joules, which is its own axis and is
# EXCLUSIVE of every other arm including `default`. Both arms pass
# `--shaping greedy` and differ only in `--speculative`, because acceptance
# is exact only at temperature 0 while the frozen protocol samples at 0.2 --
# so `ARMS=default,spec` would vary the shaping AND the speculation, and the
# script refuses it by name. A greedy row is NOT comparable to the sampled
# rows in docs/POWER_BASELINE.md; it is comparable to the other arm of its
# own capture, which is the whole point. Needs an install carrying a drafter
# (`mtp.*` or `dflash.*`). NOTE `turbospark-bench`'s `--speculative auto`
# reads the index and drives whichever drafter it finds, which is NOT what
# `turbospark-check` does since 2026-08-20: there `auto` enables an MTP head
# and only REPORTS a DFlash2 one, because that drafter measures 0.88x on
# prose (Gotcha 35's rule -- a harness that MEASURES a knob must not sense
# it, and here the harness is the arm that has to be able to turn it on).
LABEL=ac MODEL=~/models/qwen38-27b-dflash2.gturbo CASES=short-explanation \
  COOLING=max ARMS=nospec,spec scripts/power.sh 2

# The rate-cap SWEEP, which is what the numeric arms are for. That gate
# above measured the shipped `efficiency` cap of 10 tok/s at 14.9% of the
# energy for 51.5% of the throughput -- a poor trade, and unplaceable with
# only two points on the curve. `default` is the uncapped reference arm.
# Run it under COOLING=max: a governed arm's J/token is a thermal control
# loop's output rather than the cap's (Gotcha 28). gemma4 rather than a
# bigger install because the constants in `crates/runtime/src/power.rs` are
# GLOBAL and this one decodes ~44 tok/s, so the arms span a 4.4x range
# instead of museGlimmer's 1.9x. ~20 min; note each run pays the cap twice,
# since the discarded warmup is paced with the same RateControl.
LABEL=ac MODEL=~/models/gemma4.gturbo CASES=short-explanation COOLING=max \
  ARMS=default,30,20,15,10 OUT=/tmp/power-cap-sweep scripts/power.sh 3
```
