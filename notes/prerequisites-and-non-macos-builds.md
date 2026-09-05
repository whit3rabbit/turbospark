---
uuid: "b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e11"
title: "Prerequisites and non-macOS builds"
summary: "The engine is Metal, no CPU decode path, no Intel build. Off macOS, nine crates are still portable and checkable with a pinned cargo check command"
tags: ["build", "day-one"]
source: "docs/DEVELOPMENT.md, AGENTS.md Gotcha 8"
created: "2026-09-04"
updated: "2026-09-04"
---

## Can I build or check any of this without a Mac?

Yes, partially. macOS on Apple Silicon is required to build and run the
actual inference engine: it's Metal-only, no CPU decode path, no Intel
build. But nine of the seventeen workspace crates are portable and can be
checked on Linux:

```sh
cargo check --target x86_64-unknown-linux-gnu \
  -p turbospark-core -p turbospark-compute -p turbospark-model-io \
  -p turbospark-streaming -p turbospark-selection -p turbospark-invocation \
  -p turbospark-window-fit -p turbospark-gpu -p turbospark-vision-io
```

Run this after touching a `cfg`, a dependency table, or anything `unsafe`.

## Don't

- Don't assume `crates/gpu` being Metal-only means it's excluded from this
  check on purpose. It's deliberately included: off macOS it must reduce to
  an empty shell (every module gated behind `#[cfg(target_os = "macos")]`),
  and it has silently failed to do that twice before, once when a module's
  cfg gate was dropped and once when `crates/streaming` declared `libc`
  under a macOS-only dependency table while calling it unconditionally. A
  clean check here is proof the reduction still works, not a formality.
- Don't judge portability from the `#[cfg]`s in `src/` alone. It's decided
  in the dependency TABLE: `crates/runtime` declares `model_io`, `gpu`,
  `compute` and `streaming` under a `[target.'cfg(target_os = "macos")']`
  block, so any crate depending on `runtime` inherits macOS-only status
  however portable its own source reads.
- Don't run the full `cargo test --workspace` on Linux expecting the same
  coverage as macOS. `crates/gpu` and everything depending on it compiles
  away to nothing there, so the same command stays green while covering far
  less.

## Full macOS prerequisites

macOS on Apple Silicon: 13.0+ for the binding, 14.0+ for the app, check
with `sw_vers -productVersion`.
Rust: stable, 1.82+, check with `cargo --version`.
Xcode command line tools: any recent, check with
`xcrun -sdk macosx metal --version`.
Swift: 5.9+, check with `swift --version`.

Full Xcode is not required, only the command line tools, unless you want
Instruments or the simulator.
