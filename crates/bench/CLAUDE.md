# mrefrust-bench

Throughput benchmark harness (`mference-bench`), mach memory sampler (`memory.rs`), frozen community benchmark protocol (`protocol.rs`), real model benchmark runner (`real_model.rs`), and memory oracle integration test (`tests/memory_oracle.rs`).

## Directory & File Structure

```
crates/bench/
+-- Cargo.toml              # Crate manifest
+-- src/
|   +-- lib.rs              # Library entry point (mrefrust_bench)
|   +-- main.rs             # Binary entry point (mference-bench)
|   +-- memory.rs           # Mach memory sampler for physical footprint tracking
|   +-- protocol.rs         # Frozen community benchmark protocol definitions
|   \-- real_model.rs       # Real model benchmark runner driving RealForwardRunner
+-- tests/
|   +-- memory_oracle.rs    # Memory oracle asserting peak footprint ceiling & steady state
|   \-- mference_bench.rs   # Benchmark harness integration smoke test
\-- prompts/
    \-- real-generation-v1/ # Standardized benchmark protocol prompt fixtures
        +-- short-explanation.txt
        +-- medium-review.txt
        \-- long-synthesis.txt
```

## Key Modules

- `main.rs`: Handles CLI benchmark modes (scripted producer overhead vs real model protocol).
- `memory.rs`: Mach kernel task info sampler for tracking peak physical memory footprint (`phys_footprint`).
- `protocol.rs`: Frozen benchmark protocol case definitions and step evaluators.
- `real_model.rs`: Runs protocol cases against real `.gturbo` installs using `RealForwardRunner`.
- `tests/memory_oracle.rs`: Ignored test gated on `MREFRUST_GEMMA4_INSTALL_DIR` asserting per-chip memory ceilings and zero leak growth.

## Development & Test Commands

```sh
# Run fast unit and integration tests for mrefrust-bench
cargo test -p mrefrust-bench

# Run scripted producer throughput benchmark
cargo run -p mrefrust-bench --bin mference-bench -- <tokenizer-dir>

# Run real install benchmark (macOS, release mode required)
cargo run --release -p mrefrust-bench --bin mference-bench -- --model ~/models/gemma4.gturbo

# Run memory oracle test (macOS, takes ~10 mins, requires model env var)
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p mrefrust-bench --test memory_oracle --release -- --ignored --nocapture
```

## Crate Gotchas

1. **Footprint Accounting**: `phys_footprint` includes resident weight mapping (`mmap` pinned by Metal `newBufferWithBytesNoCopy`) + KV cache + expert slot capacity + process baseline.
2. **Cold GPU Benchmark Artifacts**: The first run after a build executes on a cold GPU at low DVFS clock states (up to 53% slower). Always discard at least one warmup run.
3. **Power Source & Cross-Session Ratios**: Thermal throttling and battery state (`pmset -g ps`) alter absolute tok/s. Always measure ratios back-to-back in the same session.
