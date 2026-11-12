# TurboSpark Swift Package

## Overview
Swift package (`swift/TurboSpark/Package.swift`, macOS 13+) that exposes the Rust engine to Swift. Target `CTurboSpark` is a `.systemLibrary` over the copied header; target `TurboSpark` depends on it and links `turbospark_ffi`, Metal, Foundation, QuartzCore and CoreML.

## Local Interface & Invariants
- Product `TurboSpark` library and test target `TurboSparkTests` [Package.swift:package].
- The staticlib and header are copied into `Sources/CTurboSpark` by `make swift-lib`; `swift build` fails until that has run once [Package.swift: header comment].
- Canonical header is `crates/ffi/include/turbospark.h` [Package.swift: header comment].

## Subdirectories & System Boundaries
- Sources: [./Sources/EGM.md](./Sources/EGM.md) - C module and Swift wrapper targets.

## Global Invariants
- Frameworks recorded by Rust dependencies are not carried by the staticlib, so the Swift link supplies them [Package.swift:linkerSettings].

## Global Side Effects
- None in the manifest itself; the library calls into the engine (see Sources).

## System Verification
- Run: `swift test --package-path swift/TurboSpark`
