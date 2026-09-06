---
uuid: "7ff50071-f3fa-412b-bdb7-ea63b8795416"
title: "turbospark-bench"
summary: "The throughput benchmark harness: frozen protocol cases, a mach memory sampler for phys_footprint, and per-family memory oracles / quality gates run as #[ignore]d tests"
tags: ["crate", "bench"]
depends_on: ["35d68e3f-6f31-4c93-9e9a-8b6f2ee54a01", "b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e07"]
source: "crates/bench/CLAUDE.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## What does turbospark-bench do?

It's the `turbospark-bench` binary plus the test-side harness behind every
memory oracle and quality gate in the workspace. Key pieces: `protocol.rs`
(the frozen community benchmark cases: short-explanation, medium-review,
long-synthesis), `memory.rs` (a mach `phys_footprint` sampler), and
`real_model.rs`/`real_model_params.rs` (resolves each family's context
window and generation budget from the install's own manifest, and hands
that back beside the `RealForwardRunner`). `crates/bench` is a local dev
harness only: it never publishes to crates.io. See
[[ignored-tests-and-real-model-gates]] for how to run its gated tests, and
[[crate-bench-oracles-and-gates]] for the conventions inside those tests.

## Don't

- Don't assume a new family's memory ceiling lands in the same 1.6-2.2 GiB
  band as Gemma 4 or Qwen 3.6. `phys_footprint` is dominated by
  `slots x layers x expert_stride`, so DEPTH matters as much as expert
  size. Qwen3-30B-A3B peaks at 2,751 MiB not because its experts are big,
  but because it's 48 layers deep.
- Don't trust the first timed run after a build. The GPU is cold and runs
  at low clock states, up to 53% slower measured. Always discard at least
  one warmup run before recording a number.
- Don't compare absolute tok/s across sessions or power sources. Thermal
  state and AC-vs-battery both move it. Measure ratios back-to-back in one
  session instead.
- Don't let `--model` mode auto-detect Low Power Mode the way
  `turbospark-check`/`turbospark-server` do. This binary defaults to
  `PowerProfile::Performance` deliberately: a measurement tool that senses
  the environment it's supposed to be measuring can silently cap the
  "performance" arm of an A/B at reading speed and manufacture the exact
  efficiency gain the A/B was meant to prove.
- Don't read `TURBOSPARK_PHASES=1`'s phase report as a full account of a
  decode run. It covers only the inside of `produce`. The sampler and
  detokenizer run outside it. Subtract the phase total from the footer's
  `decode=` seconds before trusting the table as complete.
- Don't change `--expert-cache-slots` and expect published numbers to
  still apply. The protocol pins 16 slots, matching Swift's own default.
  32 buys roughly 15% more decode throughput and about 1.5 GB more
  footprint. Every frozen benchmark row is measured at 16 specifically.
- Don't assume a bench or `scripts/power.sh` row measured CHUNKED prefill.
  `--prefill-chunk off|auto|N` is the only way in and it defaults OFF, so
  every frozen row here is a sequential row and a chunked row is a NEW row
  rather than a re-freeze. The header prints `prefill=sequential` or
  `prefill=chunked` so a row cannot silently be the other one. See
  [[will-a-chunked-prefill-driver-pay-on-this-family]].
- Don't reach for `invocation::PrefillChunk` for a prefill knob here. Its
  `Default` is ON, so it would silently retire every frozen row. Use
  `Option<usize>` initialized `None`.
