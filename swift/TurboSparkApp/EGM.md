# TurboSparkApp Package

## Overview
Swift package (`swift/TurboSparkApp/Package.swift`, macOS 14+) for the macOS app. Targets: `TurboSparkAppResources`, `ComputerUseKit`, `TurboSparkComputerUse`, `TurboSparkApp`, benchmark executables `ZImageMLXBenchmark` and `QwenImageMLXBenchmark`, plus fixtures and `TurboSparkAppTests`.

## Local Interface & Invariants
- Depends on local `../TurboSpark`, OpenKind (sibling checkout, or pinned revision via `OPENKIND_RELEASE_REVISION` and `OPENKIND_PACKAGE_URL`), vendored ZImage, QwenImage, Qwen3TTS, SQLCipher, MarkdownUI, Syntext, Sparkle and FluidAudio [Package.swift:dependencies].
- Resources live in their own target with a hand-written lookup; the executable target must declare none or SwiftPM generates a colliding `Bundle.module` [Package.swift: targets comment].
- FluidAudio is pinned exactly (0.15.5) because later versions break this app's link [Package.swift: FluidAudio comment].
- Syntext is pinned exactly (2.7.2); see the manifest comment for why older tags must not be pinned [Package.swift: Syntext comment].

## Subdirectories & System Boundaries
- Sources: [./Sources/EGM.md](./Sources/EGM.md) - app, helper, kit and resource targets.
- Benchmarks: [./Benchmarks/EGM.md](./Benchmarks/EGM.md) - MLX image benchmark executables.

## Global Invariants
- UI strings and design rules are enforced per the repository AGENTS.md (design checks), not by this manifest.

## Global Side Effects
- Manifest reads environment variables `OPENKIND_RELEASE_REVISION` and `OPENKIND_PACKAGE_URL` at resolution time [Package.swift].

## System Verification
- Run: `swift test --package-path swift/TurboSparkApp`
