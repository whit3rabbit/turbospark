# turbospark-gpu

## Overview
Crate root for the Metal backend. The only direct source file is the build script that compiles the shader compositions ahead of time.

## Interface
- [build.rs:FLAGS] (`-std=metal3.1`, `-mmacosx-version-min=14.0`, `-ffast-math`), [build.rs:compile], [build.rs:tool] (invokes `xcrun`), [build.rs:digest] (SHA-256 keys), [build.rs:expand] (resolves string consts/`include_str!` via `syn`).
- Generates `precompiled_registry.rs` and per-shader `.metallib` blobs in `OUT_DIR` [build.rs].
- Children: [src](./src/EGM.md), [examples](./examples/EGM.md).

## Invariants
- The build script compiles the same shader compositions the dispatch modules submit at runtime, so byte blobs match runtime sources [build.rs] (module doc).
- A missing Xcode Metal toolchain only emits `cargo:warning` and falls back to runtime shader source; `TURBOSPARK_METAL_PRECOMPILE_STRICT=1` turns it into a build failure [build.rs:main].
- Non-macOS targets write an empty `precompiled_registry.rs` and never invoke `xcrun` [build.rs:main].
- A shader whose source starts with `// turbospark: precise-math` is compiled without `-ffast-math` [build.rs:compile].

## Side Effects
- On macOS targets, runs `xcrun metal` and `metallib` and writes artifacts to `OUT_DIR` [build.rs:compile], [build.rs:tool].

## Verification
- Run: `cargo test -p turbospark-gpu`
