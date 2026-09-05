---
uuid: "e2d4f6a8-1c3b-4e5d-9f7a-8b6c4d2e0a19"
title: "turbospark-runtime: dispatch order and env seams"
summary: "Routed expert slots MUST dispatch in the router's own ranking (FP addition is not associative). The A/B env vars are seams, not features, and both arms must produce identical tokens"
tags: ["crate", "runtime"]
source: "crates/runtime/CLAUDE.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## What are the correctness constraints on this crate's dispatch order?

Several places in `turbospark-runtime` treat "just an optimization" as a
correctness constraint, since floating-point addition is not associative
and this port has been bitten by it in the wild.

## Don't

- Don't reorder a layer's routed expert slots by anything other than the
  router's own ranking. Phase 2 reduces `blob[slot] * routing_w[slot]` in
  slot-index order, so slot order IS summation order. An earlier "hits
  first" ordering let cache state (not the prompt) decide the order,
  producing 4 distinct outputs across 6 warm runs of the identical prompt
  on one real install. Ask what an ordering change is a function of before
  landing it.
- Don't assume the expert-cache slot count (`ExpertCacheSlots::Auto`) can
  move output. It's throughput-only by design: routed slots dispatch in the
  router's own ranking regardless of slot count, so output has been
  byte-identical across 8/16/24/32 slots and cold vs warm since the
  ordering fix above.
- Don't treat `TURBOSPARK_PHASES=1`'s numbers as a per-token cost at one
  context length. It averages GPU phase timings over ALL forward passes,
  prefill included. To get a marginal per-token cost at long context, run
  two `--max-new` lengths and take the delta between totals.
- Don't treat any of `TURBOSPARK_SHARED_CB`, `TURBOSPARK_ROUTED_PIPELINE`,
  `TURBOSPARK_ROUTED_BATCH`, `TURBOSPARK_BATCHED_GEMV`, or
  `TURBOSPARK_PREFILL_CHUNK` as a feature flag with two acceptable
  behaviors. They're A/B seams: both arms of each MUST produce identical
  tokens, and the tests that exist for them assert exactly that (bit-for-bit
  or digest-identical), not "both arms work."
- Don't assume a batched or chunked driver silently falls back to the
  sequential engine on an unsupported family. `ChunkedPrefillRunner::prefill_chunk`
  refuses BY NAME when a family can't be served, so an explicit request for
  the chunked path finds out rather than silently measuring the sequential
  engine and reporting it as the chunked one.
- Don't move the rate-limiting `thread::sleep` in `decode` without
  understanding why it sits where it does. It runs AFTER the loop decides
  to continue (so the last token of a generation never pays for a sleep
  nobody waits through) and BEFORE the next `produce` call (so the idle
  window falls between forward passes, which is the point on the energy
  axis: the GPU must be idle during it). A timing test on this may only
  assert a lower bound on spacing, never an upper one.
- Don't assume a GGUF install's routed experts share one block type. A real
  sub-4-bit install can mix layouts per LAYER and per PHASE within one
  expert (`routed_layouts` and `moe_offsets` are resolved per layer at
  open, not once from the manifest's single `ggmlType`).

See [[crate-runtime]] for the core architecture and [[crate-runtime-3]] for
speculative decoding.
