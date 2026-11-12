# TurboSpark Blueprint

## System Overview
TurboSpark is a Rust and Swift workspace for local model inference on Apple Silicon. Rust crates under `crates/` hold the model intake, Metal and Core ML runtimes, audio and image pipelines, and the HTTP server. `crates/ffi` exposes a C ABI that the Swift packages under `swift/` wrap, and the macOS app is built on those. This blueprint indexes only code directories. Documentation trees (`docs/`, `documentation/`, `docs-site/`, `api-reference/`, `help-center/`, `changelog/`, `notes/`) are not covered, and `swift/TurboSparkApp/Vendor` and the Swift `Tests` directories were deliberately skipped. Existing `AGENTS.md`, `CLAUDE.md` and `README.md` files remain the human-maintained source of rules and are not modified by these contracts.

## Subdirectories & System Boundaries
- crates: [./crates/EGM.md](./crates/EGM.md) - the 22 Rust workspace crates, from portable primitives to Metal backends and user-facing surfaces.
- swift: [./swift/EGM.md](./swift/EGM.md) - the `TurboSpark` wrapper package over the C ABI and the `TurboSparkApp` macOS app.
- scripts: [./scripts/EGM.md](./scripts/EGM.md) - flat dev, release, benchmark and parity tooling (bash, Python, C, Swift, Rust).

## Global Invariants
- Decode flow is selected by `ArchConfig.family`, never by tensor names [crates/model-io/src/arch_config].
- Producers return logits and sampling owns normalization, so nothing softmaxes twice [crates/selection/src].
- Token ids crossing crate boundaries are signed 32-bit [crates/core/src].
- FFI entry points stay inside the ABI guard and never unwind across `extern "C"` [crates/ffi/src].
- The Rust static library must be staged with `make swift-lib` before the Swift packages build [swift/EGM.md].
- Contracts under this tree were generated from code reads of varying depth. The Swift app directories in particular are scan-based and should be treated as an index, not a verified API.

## Global Side Effects
- Metal and Core ML device use, and memory-mapped model weights.
- Hugging Face network traffic and writes under `~/.turbospark`.
- Local HTTP listener (server and the in-process FFI server).
- Release scripts write app bundles, DMGs and appcasts [scripts/EGM.md].

## System Verification
- Run: `cargo test --workspace`
- Run: `swift test --package-path swift/TurboSpark` and `swift test --package-path swift/TurboSparkApp`
