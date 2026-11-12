# Router Batch Capture

## Overview
Bounded, batch-local router detail owned by `RouterHistogram` in the parent `router_hist.rs`. Where the legacy histogram and trace are unbounded flat records, this keeps per-batch, per-layer rows with token range, routes, completion flag and expert-cache IO counters. It is a diagnostic only, active when `TURBOSPARK_ROUTER_BATCHES=1`.

## Interface
- `BatchCapture` [batches.rs:BatchCapture]: serializable capture (version, `truncated`, `rows`); `new(limit)`, `begin(phase, start, count)`, `end(completed)`, `reset`, `record(layer, offset, routes)`, `io(layer, hits, misses, bytes)`
- `BatchRow` [batches.rs:BatchRow]: one batch and layer: request, batch, phase, layer, token range, trace range, routes, completed, cache hits/misses, bytes requested, io samples
- Subdirectories: None

## Invariants
- Total recorded route entries never exceed `limit`; the overflowing call sets `truncated` and records nothing [batches.rs:record]
- Empty route slices and records outside a `begin`/`end` scope are dropped [batches.rs:record]
- A row is `completed` only when `end(true)` ran and its route count equals `token_count`, so a truncated batch is never marked complete [batches.rs:end]
- `begin` first closes any open batch as incomplete; `reset` does the same and bumps the request counter [batches.rs:begin, reset]
- A new row opens only when the batch or layer changes, relying on in-order layer visits [batches.rs:record]
- `io` attaches only to the latest row of the current batch and layer [batches.rs:io]
- Default limit is 1,000,000 entries via `TURBOSPARK_ROUTER_BATCH_LIMIT` [router_hist.rs:RouterHistogram]

## Side Effects
- None here; the types are in-memory. The parent serializes them into the histogram JSON on drop, and env gating (`TURBOSPARK_ROUTER_BATCHES`, `TURBOSPARK_ROUTER_BATCH_LIMIT`) is in `router_hist.rs`.

## Verification
- Run: `cargo test -p turbospark-runtime router_hist`
