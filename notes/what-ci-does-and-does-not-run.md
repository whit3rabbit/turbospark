---
uuid: "b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e06"
title: "What CI does and does not run"
summary: "verify (every PR) builds/tests Rust on macOS and checks a portable subset on Linux, and compiles NO Swift. package-macos (push-to-main only) builds the app+DMG"
tags: ["ci", "day-one"]
depends_on: ["b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e07"]
source: ".github/workflows/ci.yml, docs/DEVELOPMENT.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## What does CI actually check, and what does it miss?

Two jobs in `.github/workflows/ci.yml`, and the difference matters:

**`verify`** runs on every PR and push to `main`, matrixed over
`macos-latest` and `ubuntu-latest`, but the real work is macOS-only:
`cargo fmt --check`, `cargo clippy --workspace --tests`,
`cargo build --workspace`, `cargo test --workspace` all run only when
`matrix.os == 'macos-latest'`. The Ubuntu leg does exactly one thing: a
`cargo check` over the nine portable crates (core, compute, model-io,
streaming, selection, invocation, window-fit, gpu, vision-io). **This job
compiles no Swift at all.**

**`package-macos`** builds the release packaging path (app bundle + DMG via
`scripts/make-app-bundle.sh` and `scripts/make-dmg.sh`) on `macos-15`
specifically, pinned rather than `macos-latest`, because this is the ONLY
CI job that compiles Swift. It runs **push-to-main only**, never on a PR.

## Don't

- Don't assume a green PR means the Swift side builds. A Swift break lands
  on `main` via `package-macos`, not on your PR. Build the app locally
  (`make app-bundle` is the closest local approximation of that job) before
  merging anything under `swift/`.
- Don't assume `macos-latest` and the `package-macos` runner use the same
  SDK. `package-macos` is pinned to `macos-15` (Xcode with the macOS 15
  SDK) specifically because the app's deployment target and a
  `#available`-guarded symbol needed that SDK to resolve. A prior pin to
  `macos-14` shipped the app half broken on `main` while every local build
  stayed green. See `swift/CLAUDE.md` Gotcha 45 for the three failure modes
  this has already produced.
- Don't expect CI to run the `#[ignore]`d tests, memory oracles, quality
  gates, or any real-model smoke test. None of those run in CI at all. They
  need real multi-GB checkpoints and are opt-in locally. See
  [[ignored-tests-and-real-model-gates]].
- Don't be surprised the Linux leg exists mostly to catch a dropped `cfg`
  gate, not to prove real Linux support. `crates/gpu` is deliberately
  included in that portable-subset check despite being Metal-only: it must
  reduce to an empty shell off macOS, and has silently failed to do so
  twice (a lost `#[cfg(target_os = "macos")]` on a module, and `libc`
  declared macOS-only in `Cargo.toml` while called unconditionally).
