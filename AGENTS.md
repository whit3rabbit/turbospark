# AGENTS.md

This file is the short, always-loaded orientation for the TurboSpark Rust and
Swift workspace. Keep detailed procedures, measurements, and historical
evidence in the linked project docs. Do not turn this file into a changelog.

## First read

- Read the nearest crate or app instructions before changing code:
  `crates/<name>/AGENTS.md` or `swift/AGENTS.md`.
- Read `docs/DEVELOPMENT.md` before setup or build failures.
- Read `docs/TESTING.md` before adding or changing tests.
- Read `docs/BENCHMARKING.md` and `docs/BENCHMARKS.md` before quoting or
  re-freezing a number.
- Read `docs/NEW_MODEL.md` before adding a model family or checkpoint path.
- Read `docs/RELEASE.md` before cutting an artifact or tag.
- Read the relevant page under `.claude/docs/` for deep verification,
  benchmark, harness, install, power, cross-engine, or diagnostic work.

The root references are:

- [verification reference](.claude/docs/verification.md)
- [engineering guardrails](.claude/docs/engineering-gotchas.md)
- [benchmark reference](.claude/docs/benchmarks.md)
- [agent harness and repository workflow](.claude/docs/agent-harness.md)
- [model gates](.claude/docs/model-gates.md)
- [checkpoint installs](.claude/docs/checkpoint-installs.md)
- [diagnostics](.claude/docs/diagnostics.md)
- [cross-engine checks](.claude/docs/cross-engine-kl.md)
- [power measurement](.claude/docs/power-measurement.md)

## Workspace contract

- macOS is the primary development platform. Rust uses edition 2021, MSRV
  1.82, and a committed `Cargo.lock`.
- `crates/gpu` and the real runtime path require macOS and a Metal-capable
  device. `crates/vision-io` is intentionally part of the portable
  cross-target check.
- Keep source, comments, and docs ASCII. Do not use emojis or em dashes.
- Preserve concurrent work. Inspect `git status` before editing, do not reset,
  clean, or check out another session's files, and stage only intended paths.
- `target/`, build products, staged generated Swift FFI files, and local
  machine notes are not release source.
- Use `st` for repository search. Run `st index` when its index is stale or
  missing. Use exact searches before broad searches.
- If `mf.sqlite3` exists in the repository, use the memoryfield workflow
  before exploring the codebase.

## Safety invariants

- Decode flow is selected by `ArchConfig.family`, never by tensor names.
  A wrong family can produce fluent, incorrect output.
- Producers return logits. Sampling owns normalization. Never softmax in both
  the producer and sampler.
- Repeated Metal encoding loops use `gpu::autorelease_pool`. Without an
  inner pool, autoreleased command objects accumulate for the life of the
  process.
- KV cache sizing and ring capacity are part of correctness. Derive the ring
  from the layer mask and include capacity in pipeline-cache constants.
- Routed expert slots are dispatched in router ranking order. Slot order is
  reduction order.
- Token IDs crossing crate boundaries remain signed 32-bit
  (`turbospark_core::TokenId`). Resolve fixture IDs from a loaded tokenizer;
  never copy numeric IDs from fixture JSON.
- Family support is cross-layer. Trace the family enum and persisted string
  through model-io, runtime, FFI, Swift, catalog, CLI, server, docs, and tests.
- Unsafe code is intentional only in the documented ABI, mapping, and streaming
  boundaries. FFI entry points must stay inside the ABI guard and must not
  unwind across `extern "C"`.
- A platform claim is not a gate. When changing cfgs, dependency tables,
  memory mapping, kernels, or FFI, run the relevant cross-target and real
  hardware checks from the verification reference.
- A composite measurement inherits the limits of its components. Do not turn a
  projection, CPU run, synthetic fixture, or contaminated host reading into a
  real Metal, quality, or memory claim.
- Do not widen quality or resource tolerances without new evidence. Preserve
  negative findings and refusal paths in the canonical docs.

## Minimal verification

For ordinary Rust changes:

```sh
cargo build --workspace
cargo test --workspace
cargo fmt --check
cargo clippy --workspace --tests
```

For Swift or FFI changes:

```sh
make swift-lib
make swift-test
```

For app or release artifacts:

```sh
make swift-app
make app-bundle
make dmg
```

The commands above are the baseline only. Changes to decode, quantization,
sampling, KV, Metal, model intake, FFI, or real-install behavior require the
additional gates documented in [verification](.claude/docs/verification.md)
and [model gates](.claude/docs/model-gates.md). Do not claim a hardware gate
from compilation or a CPU-only result.

## Change boundaries

- Keep crate-local architecture and gotchas in the nearest crate file. Do not
  copy root rules into every module.
- Keep benchmark protocols and frozen rows in `docs/BENCHMARKING.md`,
  `docs/BENCHMARKS.md`, and `.claude/docs/benchmarks.md`. Do not add
  benchmark numbers to AGENTS.md files.
- Keep release and packaging procedures in `docs/RELEASE.md` and the
  verification reference.
- Keep project planning in the existing planning document, design rationale in
  `docs/`, and unfinished work in the issue tracker. Do not append
  session summaries or dated handoff notes here.
- For a contested file, re-check status immediately before staging. If a
  dependency on another uncommitted file makes a clean partial commit unsafe,
  report the dependency instead of staging unrelated work.
- Every new test must exercise the behavior it claims to cover. Mutation-check
  new guards when the behavior is subtle, and assert that a mutation applied
  before treating a survivor as evidence.

## Project map

- `crates/model-io`: checkpoint formats, manifests, mappings, and resident
  buffers.
- `crates/repack`: model intake and `.gturbo` writing.
- `crates/runtime`: family dispatch, forward passes, state, and sessions.
- `crates/gpu`: Metal kernels and dispatch.
- `crates/compute`, `crates/core`, `crates/selection`,
  `crates/tokenizer`, `crates/streaming`, `crates/invocation`: portable
  primitives and protocol boundaries.
- `crates/ffi`: the C ABI consumed by Swift.
- `crates/cli`, `crates/catalog`, `crates/server`: user-facing install,
  command, and service surfaces.
- `crates/bench`: measurement harnesses. Read its local pointer and the
  benchmark docs before running or interpreting a gate.
- `swift/TurboSparkApp`: the macOS app, persistence, tools, localization,
  and Swift-facing model controls.

## Final handoff

Report the files changed, focused checks run, any unrelated checkout failures,
and whether a real hardware or release gate was actually run. Keep claims
bounded by the evidence in the current checkout.
