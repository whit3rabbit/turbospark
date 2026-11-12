# image Blueprint

## System Overview
The `turbospark-image` crate (lib `turbospark_image`) holds the native
text-to-image generation pipeline. All code lives under `src/`; the crate root
has no direct source files. Parity and payload tests live in `tests/`.

## Subdirectories & System Boundaries
- src: [./src/EGM.md](./src/EGM.md) - pipeline, backends (CPU reference and Metal), packed tensor store, install builder, runtime lifecycle.
- src/shaders: [./src/shaders/EGM.md](./src/shaders/EGM.md) - Metal kernels embedded by the Metal backend.

## Global Invariants
- Unsafe code is forbidden in the crate [src/lib.rs:forbid(unsafe_code)].
- Requests use a fixed 9-step, batch 1, guidance 0.0 envelope [src/runtime.rs:ImageRequest::validate]. Install manifests carry a per-family envelope (Z-Image, Qwen-Image-2.1, Qwen-Image-2.1-Turbo, Krea 2 Turbo) [src/install.rs:ImageFamily::supported], but the request path only admits the Z-Image one, so only Z-Image installs can run `generate`.

## Global Side Effects
- Reads model installs from disk and writes install trees via staging and rename [src/builder.rs:build_image_install_with_progress].
- Uses the Metal GPU on macOS [src/lib.rs:cfg(target_os = "macos")].

## System Verification
- Run: `cargo test -p turbospark-image`
