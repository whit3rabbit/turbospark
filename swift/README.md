# Swift Packages & macOS App

This directory contains the Swift packages for TurboSpark:
1. **`TurboSpark`**: The core SwiftPM package providing idiomatic Swift async/await bindings over the in-process C ABI (`crates/ffi`).
2. **`TurboSparkApp`**: The native macOS SwiftUI desktop chat and model management application.

Both targets run on Apple Silicon (macOS 13.0+) and link against the native Metal inference engine in-process with zero HTTP or IPC overhead.

---

## Directory Structure

```
swift/
├── TurboSpark/                 # SwiftPM library package
│   ├── Package.swift
│   ├── Sources/
│   │   ├── CTurboSpark/        # Staged C header and staticlib (gitignored)
│   │   └── TurboSpark/         # Swift async API (Session, Catalog, Types, Errors)
│   └── Tests/
│       └── TurboSparkTests/    # ABI surface and real-install integration tests
└── TurboSparkApp/              # macOS SwiftUI Application
    ├── Package.swift
    └── Sources/
        └── TurboSparkApp/
            ├── App/            # Entry point (TurboSparkApp.swift, RootView.swift)
            ├── Components/     # Badges, error banners, generate controls
            ├── Diagnostics/    # Inspector, telemetry, runner metrics, status HUD
            ├── Generation/     # Chat sidebar, message pane, prompt composer
            ├── Installation/   # Model catalog downloader, progress sheets, ETA
            ├── Presentation/   # Markdown rendering, document text extraction (PDF/Word/Excel)
            ├── State/          # AppModel state coordinator and persistence
            └── Theme/          # Typography, colors, and layout tokens
```

---

The repository-wide setup, the order the Rust and Swift halves build in, and
the day-to-day loop are [`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md). This
page is the Swift packages in detail.

## Prerequisites

- **macOS**: 13.0 (Ventura) or newer on Apple Silicon (M1/M2/M3/M4).
- **Rust toolchain**: Stable with target `aarch64-apple-darwin` (`rustup target add aarch64-apple-darwin`).
- **Xcode Command Line Tools**: `xcode-select --install` or full Xcode with Metal toolchain (`xcrun -sdk macosx metal`).
- **Swift**: Swift 5.9+ / Swift 6.

---

## Building and Running

All targets are driven conveniently from the repository root `Makefile`.

### 1. Build the FFI Static Library

Before building or testing any Swift package, build the Rust `turbospark-ffi` static library and stage the canonical header:

```sh
make swift-lib
```

This compiles `crates/ffi` for `aarch64-apple-darwin` with `MACOSX_DEPLOYMENT_TARGET=13.0` and stages `libturbospark_ffi.a` and `turbospark.h` into `swift/TurboSpark/Sources/CTurboSpark/`.

### 2. Build the macOS App

Debug build:
```sh
make swift-app-build
# or directly with SwiftPM:
# cd swift/TurboSparkApp && swift build
```

Release build:
```sh
make swift-app-release
# or directly with SwiftPM:
# cd swift/TurboSparkApp && swift build -c release
```

### 3. Run the macOS App

```sh
make swift-app
# or:
# make swift-demo
```

### 4. Run Tests

- **C ABI & Surface Tests** (no downloaded model required, ~1 second):
  ```sh
  make swift-test
  ```

- **End-to-End Real Model Tests** (requires a `.gturbo` model directory):
  ```sh
  make swift-test-real MODEL=~/models/gemma4.gturbo
  ```

- **Speculation Refusal Tests** (optional second MoE or sub-4-bit install):
  ```sh
  make swift-test-real MODEL=~/models/qwen38-27b-mtp.gturbo BLOCKED=~/models/ornith35b.gturbo
  ```

---

## Cleaning

To clean Swift build artifacts and staged C ABI headers:

```sh
make clean-swift
```

To clean both Rust and Swift build artifacts:

```sh
make clean
```

---

## Swift API Overview

### Session Generation

```swift
import TurboSpark

let options = OpenOptions()
let session = try await TurboSparkSession(modelPath: "~/models/gemma4.gturbo", options: options)

var genOptions = GenerateOptions()
genOptions.temperature = 0.2
genOptions.reasoning = .medium

let messages = [ChatMessage(role: .user, content: "Explain how flash attention works.")]
for try await event in session.generate(messages, options: genOptions) {
    switch event {
    case .prefill(let done, let total):
        print("Prefill: \(done)/\(total)")
    case .content(let text):
        print(text, terminator: "")
    case .reasoning(let thinking):
        print("Thinking: \(thinking)")
    case .finished(let result):
        print("\nCompleted: \(result.newTokens) tokens, stop reason: \(result.stopReason)")
    }
}

// Non-blocking cancellation from any thread or Task
session.cancel()
```

### Catalog Management

```swift
import TurboSpark

// List installed and available models
let installed = try TurboSparkCatalog.installed()
let available = try TurboSparkCatalog.available()

// Stream download and repack directly from Hugging Face
for try await event in TurboSparkCatalog.install("gemma4") {
    switch event {
    case .stage(let name):
        print("Stage: \(name)")
    case .bytes(let done, let total):
        print("Progress: \(done)/\(total) bytes")
    case .finished(let model):
        print("Installed \(model.alias) at \(model.path)")
    }
}
```

---

## Permission Gate

`TurboSparkApp` executes model-proposed shell commands, and `.auto` mode runs
some of them without asking. The barrier is a positive allowlist in
`TerminalCommandClassifier`, with a bundled 192 KiB linear classifier
(`CommandGate`) beside it that scores commands for hazard and obfuscation in
about 30 us.

That classifier only ever writes the reason shown on the approval sheet. Its
veto is disabled by default, because measured against this repository's own
contract lists it adds zero true positives and one to three false positives.
[`docs/PERMISSION_GATE.md`](../docs/PERMISSION_GATE.md) has the numbers, the
corpora it was trained on, the defect that withdrew the first models, and the
two porting landmines the oracle fixture exists to catch.

---

Detailed technical documentation for the C ABI and Swift binding contracts is available in [`docs/SWIFT_BINDINGS.md`](../docs/SWIFT_BINDINGS.md).
