---
description: Test suites, verification gates, memory oracles, and release readiness checks.
icon: check-double
---

# Testing and Verification

TurboSpark enforces rigorous verification standards. A compilation success or CPU-only test run is never treated as evidence for a real Metal GPU, memory, or quality gate claim.

## Minimal Verification Workflow

Before submitting changes or staging commits:

### 1. Rust Workspace Checks
```sh
# Verify all crates compile cleanly
cargo build --workspace

# Run complete workspace unit and integration tests
cargo test --workspace

# Check formatting and linting
cargo fmt --check
cargo clippy --workspace --tests
```

### 2. Swift and FFI Checks
```sh
# Rebuild C dynamic library
make swift-lib

# Run Swift unit tests
make swift-test
```

### 3. Server API Contract Verification
Verify that server routes match the committed OpenAPI specifications:
```sh
# Verify audio route contract
cargo test -p turbospark-server --test audio_api -- openapi

# Verify unified server route contract
cargo test -p turbospark-server --test server_api_contract
```

### 4. Release App Bundle Verification
Packaging success alone does not guarantee launch stability. Always run the verification script:
```sh
make swift-app
make app-bundle
scripts/verify-app-bundle.sh
```

---

## Quality Gates and Memory Oracles

Every catalog row promoted to `verified` status must satisfy two independent empirical gates on real Apple Silicon hardware:

1. **Quality Gate**:
   * Evaluates token perplexity over deterministic reference prompt datasets.
   * Asserts stable logit digest hashes for both greedy and temperature-sampled generation.
   * Compares output distribution against upstream Python/MLX or llama.cpp baselines (cross-engine KL probe bound $\le 10^{-3}$ nats).

2. **Memory Oracle**:
   * Measures `phys_footprint` on an Apple Silicon device running the exact context window.
   * Asserts that peak memory consumption remains strictly beneath the committed ceiling threshold.
   * Asserts that the model reaches `EndOfTurn` without kernel panics or unbounded memory growth.
