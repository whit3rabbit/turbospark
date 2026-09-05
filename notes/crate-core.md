---
uuid: "c88391d9-8fd2-4a21-adb9-ad806a107999"
title: "turbospark-core"
summary: "Shared primitives (TokenId=i32), RuntimeConfig with panicking allowed-set setters, chunk sizing, and SteeringMode. Imported downstream as `foundation`"
tags: ["crate", "core"]
source: "crates/core/CLAUDE.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## What does turbospark-core do?

Shared primitives, error types, `RuntimeConfig`, the allowed numeric sets
(`ALLOWED_CACHE_SLOTS`, `ALLOWED_CHUNK_SIZES`), automatic chunk-size
resolution, prefill chunking, and `SteeringMode` (the four directional-
steering edits: Ablate, Add, Clamp, Renorm). It has no internal workspace
dependencies, which is why nearly every other crate depends on it.

Every downstream crate imports it under the alias `foundation` (declared
per-crate in that crate's own `Cargo.toml`), not as `turbospark_core`.

## Don't

- Don't call a `RuntimeConfig` numeric setter with an arbitrary value and
  expect a `Result`. Setters PANIC when a value is outside the allowed set.
  There's no clamping and no fallible variant. Validate against
  `ALLOWED_CACHE_SLOTS` / `ALLOWED_CHUNK_SIZES` (in `src/runtime_config.rs`)
  first, or wrap with `catch_unwind`.
- Don't hardcode the allowed-set literals elsewhere. Read them from the
  const arrays in `src/runtime_config.rs`. `chunk_sizing.rs`'s resolution
  rule reads the same constants rather than redeclaring them.
- Don't widen or narrow `TokenId` from `i32` when wiring a new downstream
  crate. It's the interchange width every crate boundary assumes.
- Don't import this crate as `turbospark_core` in a new crate without
  checking what alias that crate's siblings already use. The alias is
  per-crate, not a workspace-wide convention.
- Don't move `SteeringMode` into `turbospark-compute` or `turbospark-gpu`
  even though both name it. It lives here because `compute` (the numerical
  contract) and `gpu` (the dispatch selector) can't see each other: `gpu`
  only carries `compute` as a dev-dependency, so an enum declared in
  `compute` would be unnameable from `gpu`'s dispatch code. Its
  discriminants are wire values the Metal kernel switches on, so reordering
  variants silently swaps two edits that both decode fluently without
  erroring.
