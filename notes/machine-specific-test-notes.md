---
uuid: "b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e09"
title: "Machine-specific test and benchmark gotchas"
summary: "syspolicyd stalls a cold cargo test at 0% CPU for tens of minutes (normal). Long oracles/gates need background execution. Always benchmark on AC power"
tags: ["testing", "benchmarking", "day-one"]
source: "CLAUDE.local.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## Why does my first test run look stuck?

After a large build, a full `cargo test --workspace` can appear hung at 0%
CPU for tens of minutes. That's `syspolicyd` (Gatekeeper) verifying each
freshly built test binary on first execution, roughly 30s each across
dozens of binaries. Sample `ps` twice, 20s apart: a different test binary
name each time means it's advancing normally. Don't kill the run.

A cold `cargo test --workspace` can take ~75 minutes on a laptop, almost
all of it this verification step, not compilation or actual test time.

## Don't

- Don't run memory oracles or long quality gates (e.g.
  `museglimmer_memory_oracle`, `quality_gate`) in the foreground. They
  exceed typical foreground command timeouts (often ~10 minutes). Run them
  as background tasks and poll or wait for completion.
- Don't benchmark or run a memory oracle on battery power. Sustained
  prefill (a prompt over ~2,000 tokens) raises thermal pressure above
  Nominal on battery, throttling clocks and distorting both tok/s and
  Joules/token in ways that don't reproduce on AC.
- Don't run consecutive batches when A/B-testing throughput. Interleave
  variants pair-by-pair instead. Thermal drift alone can create ~1.5 tok/s
  of artificial spread between a first and later run, and the FIRST timed
  run after any build is a cold GPU running at low clocks (measured 53%
  slower than warm on one real install). Always discard at least one
  warmup run.
- Don't pipe long-running background test output directly through `grep`,
  `sed`, or `tail`. Redirect stdout and stderr to separate files first
  (`> /tmp/out.log 2> /tmp/err.log`). Direct pipe buffering can delay
  capture, and stderr often carries the stop-phase footer a grep needs.

## Note

This page's specifics come from one developer's local, gitignored machine
notes (an M4 Max, 36 GB), not a committed project doc. The qualitative
gotchas (Gatekeeper stall, thermal throttling on battery, cold-GPU warmup)
generalize to any Apple Silicon Mac. The exact minute counts and tok/s
deltas do not.
