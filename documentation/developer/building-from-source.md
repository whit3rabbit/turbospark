---
description: Build TurboSpark CLI, HTTP server, and native Swift desktop application from source.
icon: hammer
---

# Building from Source

This guide covers building the complete TurboSpark workspace from source on macOS Apple Silicon.

## Prerequisites

* **Hardware**: Apple Silicon Mac (M1 or newer).
* **Operating System**: macOS 14 Sonoma or newer.
* **Xcode Command Line Tools**: `xcode-select --install`
* **Rust Toolchain**: Rust 2021 edition with Minimum Supported Rust Version (MSRV) **1.82** or newer:
  ```sh
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

---

## 1. Building the Rust Crates (CLI & Server)

Clone the repository and build with cargo:

```sh
git clone https://github.com/whit3rabbit/turbospark.git
cd turbospark

# Build all release binaries
cargo build --workspace --release
```

Release binaries are placed in `target/release/`:
* `target/release/turbospark`: Unified front-end command.
* `target/release/turbospark-check`: Inference test harness and interactive chat.
* `target/release/turbospark-model`: Model catalog manager and downloader.
* `target/release/turbospark-server`: Local API server.
* `target/release/turbospark-image`: Native Metal image generator.
* `target/release/turbospark-clone`: MegaTTS3 voice cloning CLI.
* `target/release/turbospark-music`: MiniMax music generation CLI.

---

## 2. Building the C / Swift Dynamic Library

The macOS app and Swift bindings communicate with the Rust engine through a C ABI dynamic library (`libturbospark.dylib`):

```sh
# Build C dynamic library for SwiftPM
make swift-lib
```

This compiles `crates/ffi` and places the dylib and generated C headers in the Swift package directory.

---

## 3. Building the Native macOS Desktop App

To compile `TurboSpark.app`:

```sh
# Build the native macOS application bundle
make swift-app

# Package the .app bundle with CLI binaries and resources
make app-bundle

# Verify the app bundle executes cleanly
scripts/verify-app-bundle.sh

# Build release DMG
make dmg
```

The resulting `TurboSpark.app` bundle will be located under `build/TurboSpark.app`.
