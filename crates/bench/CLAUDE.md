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

# One protocol case per process (the protocol's fresh-process leg, and what
# a cross-engine comparison needs since Swift's CLI launches once per case)
cargo run --release -p mrefrust-bench --bin mference-bench -- \
  --model ~/models/gemma4.gturbo --case short-explanation

# Vary the routed-expert cache size (allowed 8/16/24/32, default 16, the
# same set and default MferenceCLI takes). 32 buys ~15% decode and ~1.5 GB
# of footprint, so it leaves the ~2 GB working-set claim behind; every
# published number is measured at the default.
cargo run --release -p mrefrust-bench --bin mference-bench -- \
  --model ~/models/gemma4.gturbo --case short-explanation --expert-cache-slots 32

# Head-to-head against ../Mference's MferenceCLI, same install, interleaved
# arms, one fresh process per run. Results: docs/BENCHMARKS.md
scripts/parity.sh

# Bucket-level decode phase diff against the Swift engine (both print a
# split under MFERENCE_PHASES=1, but Swift's is decode-only and this
# port's divides by all forward passes -- hence the short-prompt rule
# baked into the script). Not a benchmark; an attribution aid.
scripts/phasediff.sh [pairs] [slots]

# Run memory oracle test (macOS, takes ~10 mins, requires model env var)
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p mrefrust-bench --test memory_oracle --release -- --ignored --nocapture
```

## Crate Gotchas

1. **Footprint Accounting**: `phys_footprint` includes resident weight mapping (`mmap` pinned by Metal `newBufferWithBytesNoCopy`) + KV cache + expert slot capacity + process baseline.
2. **Cold GPU Benchmark Artifacts**: The first run after a build executes on a cold GPU at low DVFS clock states (up to 53% slower). Always discard at least one warmup run.
3. **Power Source & Cross-Session Ratios**: Thermal throttling and battery state (`pmset -g ps`) alter absolute tok/s. Always measure ratios back-to-back in the same session.
4. **Slot count is pinned, not defaulted-into**: `protocol::PROTOCOL_EXPERT_CACHE_SLOTS` (16) is what `docs/BENCHMARKS.md`, the memory oracle's per-chip rows, and Swift's own default all sit at. `--expert-cache-slots` exists so a Swift comparison can match a non-default setting, not so the protocol can drift; the oracle passes the constant explicitly for that reason. Output is not identical across slot counts (the hit/miss split permutes the phase-2 reduce order, FP addition is not associative), so compare within one count.
5. **The phase report does not account for a decode run**: `MFERENCE_PHASES=1` covers the inside of `produce` only; the sampler and detokenizer are outside it (AGENTS.md Gotcha 23). Subtract the phase total from the footer's `decode=` seconds before trusting a phase table as a full attribution.
