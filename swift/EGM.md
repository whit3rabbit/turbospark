# swift Blueprint

## System Overview
Swift side of TurboSpark: the `TurboSpark` package wraps the Rust C ABI, and the `TurboSparkApp` package is the macOS app built on it.

## Local Interface & Invariants
- `agentmode-mutations.sh` runs mutation checks for the AgentMode tests: each mutation asserts it applied exactly once, runs the `AgentModeTests` filter, and restores the file only if it still holds the script's mutated text [agentmode-mutations.sh:restore, mutate].
- It refuses to mutate a file with uncommitted changes [agentmode-mutations.sh:mutate].
- Side effects: temporarily edits source files and creates backups via `mktemp` [agentmode-mutations.sh:mutate].

## Subdirectories & System Boundaries
- TurboSpark: [./TurboSpark/EGM.md](./TurboSpark/EGM.md) - Swift wrapper package over the Rust FFI.
- TurboSparkApp: [./TurboSparkApp/EGM.md](./TurboSparkApp/EGM.md) - macOS app package.

## Global Invariants
- The app depends on the TurboSpark package by path; the Rust staticlib must be staged with `make swift-lib` first.

## Global Side Effects
- Engine FFI, network, Keychain and disk effects are documented in the child contracts.

## System Verification
- Run: `swift test --package-path swift/TurboSpark` and `swift test --package-path swift/TurboSparkApp`
