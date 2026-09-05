---
uuid: "b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e02"
title: "Building and running locally"
summary: "cargo build --workspace for Rust. make swift-lib MUST run once before any Swift build, or swift build fails on a missing header"
tags: ["build", "day-one", "swift"]
depends_on: ["b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e05", "b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e06", "b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e11"]
source: "docs/DEVELOPMENT.md, Makefile"
created: "2026-09-04"
updated: "2026-09-04"
---

## How do I build and run this locally?

The tree is two halves that build in one direction: a Rust workspace
(`crates/`) is the engine, and a C ABI over it (`crates/ffi`) is compiled to
a static library that two SwiftPM packages link (`swift/TurboSpark` the
binding, `swift/TurboSparkApp` the macOS app). Nothing Swift builds until
the Rust half is built and staged into the Swift package.

```sh
cargo build --workspace                 # the engine, no Swift needed
cargo run -p turbospark-cli --bin turbospark-check -- --help
```

To touch anything Swift, run the staging step first, exactly once per
checkout or worktree:

```sh
make swift-lib
```

This builds `crates/ffi` for `aarch64-apple-darwin` and copies
`libturbospark_ffi.a` and `turbospark.h` into
`swift/TurboSpark/Sources/CTurboSpark/`. Both copies are gitignored, so a
fresh clone or worktree starts without them.

```sh
make swift-app-build      # debug app build (runs swift-lib first)
make swift-app            # build and run the macOS app
cd swift/TurboSparkApp && swift build   # SwiftUI-only iteration, skip make
```

Running the CLI or server against a real model:

```sh
cargo run --release -p turbospark-cli --bin turbospark-check -- \
  --model ~/models/gemma4.gturbo --prompt "hi"
cargo run --release -p turbospark-server --bin turbospark-server -- \
  --model ~/models/gemma4.gturbo
```

A model install is not required to build or run the standing test suites.
Everything gated on a real checkpoint is `#[ignore]`d or reads an env var
and skips. See [[getting-a-model]] for installing one.

## Don't

- Don't run `make swift-app-build` / `make swift-app` for a SwiftUI-only
  change. `make swift-lib` `touch`es every Swift file in both packages to
  force a relink (SwiftPM does not treat the rebuilt `.a` as a build input),
  so going through `make` pays a full Swift rebuild every time. Call
  `swift build` directly from `swift/TurboSparkApp` instead.
- Don't build or test from a subdirectory of `swift/TurboSparkApp`. The
  package passes `-L../TurboSpark/Sources/CTurboSpark` as an unsafe linker
  flag resolved against the current directory. From anywhere else it fails
  with `linker command failed` and no error line above it.
- Don't use `--prompt` on an instruction-tuned model and expect coherent
  output. That's the chat template missing, not a decode bug. Use
  `--messages-file` (JSON conversation) instead.
- Don't expect Swift breakage to show up on a PR. `.github/workflows/ci.yml`
  compiles no Swift at all on `verify` (every PR). Only `package-macos`
  (push-to-main only) builds the app bundle and DMG. See
  [[what-ci-does-and-does-not-run]]. Build the app locally before merging
  anything that touches Swift.

## Prerequisites

macOS on Apple Silicon (13.0+ binding, 14.0+ app), Rust stable 1.82+
(`rust-toolchain.toml` pins it), Xcode command line tools (for the `metal`
shader compiler), Swift 5.9+. Full Xcode is not required. See
[[prerequisites-and-non-macos-builds]] for the engine's Metal dependency and
what can be checked off a Mac.
