---
uuid: "2caf90c9-0aff-43b0-9a28-8ebb1a0895c0"
title: "turbospark-invocation"
summary: "Pure CLI argument parsing, no I/O. Adding a flag touches 5 places. Missing a parser match arm panics at runtime, not compile time"
tags: ["crate", "invocation"]
source: "crates/invocation/CLAUDE.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## What does turbospark-invocation do?

Pure CLI argument parsing: the `OPTIONS` table, the parser, `InvocationRequest`
(validated parameters), typed `InvocationFailure` errors, usage rendering,
and outcome routing. It performs no filesystem, environment, or process I/O,
and depends only on `foundation` (`turbospark-core`), which is what keeps it
pure: it can't read a machine or a model install, so several fields are
deliberate MIRRORS of types the engine owns (`MaxContext`,
`ExpertCacheSlots`, `PowerProfile`), and `crates/cli` maps between the two
spellings in exactly one place per type.

## Don't

- Don't add a new flag without touching all five places: the `OPTIONS`
  table (`options.rs`), the parser's first-pass dispatch match, its
  second-pass value-extraction match, `InvocationRequest`'s struct and
  literal, and `tests/usage_and_status.rs`'s hardcoded option-count
  assertion. Parser matches end in `unreachable!()`, so a missed arm is a
  RUNTIME PANIC, not a compile error.
- Don't assume a flag that parses cleanly is actually wired to anything.
  `--prefill-chunk` and `--rdadvise` both round-trip through
  `InvocationRequest` and get printed in `main.rs`'s resolved-request
  block, and neither is read by any binary. Grep for `request.<field>`
  outside `main.rs` before assuming a flag, or a `TURBOSPARK_*` env var,
  needs a new surface.
- Don't assume finding an unwired flag means wiring it is a one-line read.
  `--prefill-chunk` defaults to `Fixed(DEFAULT_CHUNK_SIZE)` rather than to
  off, so naively consuming it turns chunked prefill on by default. Giving
  it an off state needs a new `PrefillChunk` variant plus the five-place
  rule above.
- Don't add a new `PowerProfile`- or `ExpertCacheSlots`-shaped duplicated
  enum lightly just because this crate is pure. That's the intended
  pattern (three flags already take an `auto` keyword this way,
  `--reasoning`'s `ReasoningEffort` is the newest), but each one needs a
  mapping function in `crates/cli`, not a new cross-crate dependency edge.
- Don't confuse `--version` with a `Help` variant. It's a third
  `ParseOutcome`, sourced from `CARGO_PKG_VERSION` via `usage::VERSION`,
  and short-circuits at whichever of `--help`/`--version` is reached
  first.
