# mrefrust-bench

Throughput benchmark harness (`mference-bench`), Mach physical memory footprint sampler (`memory.rs`), frozen community benchmark protocol (`protocol.rs`), real model runner (`real_model.rs`), and memory oracle tests.

## Binary Execution

```sh
# Run scripted producer throughput benchmark
cargo run -p mrefrust-bench --bin mference-bench -- <tokenizer-dir>

# Run real model protocol benchmark (macOS, release mode required)
cargo run --release -p mrefrust-bench --bin mference-bench -- --model ~/models/gemma4.gturbo

# Run single protocol case per process
cargo run --release -p mrefrust-bench --bin mference-bench -- \
  --model ~/models/gemma4.gturbo --case short-explanation
```

## Key Modules

- `main.rs`: CLI benchmark modes (scripted producer overhead vs real model protocol runner).
- `memory.rs`: Mach kernel task info sampler tracking physical memory footprint (`phys_footprint`).
- `protocol.rs`: Frozen community benchmark protocol definitions and evaluation cases.
- `real_model.rs`: Real model benchmark runner driving `RealForwardRunner`.
- `tests/memory_oracle.rs`: Memory oracle asserting peak footprint ceiling and steady-state behavior for Gemma 4.
- `tests/qwen36_memory_oracle.rs`: Memory oracle asserting peak footprint ceiling for Qwen 3.6.
- `tests/quality_gate.rs` & `tests/qwen36_quality_gate.rs`: Quality gate evaluation verifying generated output digests.

## Development & Test Commands

```sh
# Run unit and integration tests for mrefrust-bench
cargo test -p mrefrust-bench

# Run memory oracle test (macOS, requires model environment variable)
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p mrefrust-bench --test memory_oracle --release -- --ignored --nocapture
```

## Crate Gotchas

1. **Footprint Measurement**: `phys_footprint` measures active process physical memory including memory-mapped weights pinned by Metal (`newBufferWithBytesNoCopy`), KV cache, expert slot buffers, and process base memory.
2. **Cold GPU DVFS Warmup**: Initial execution runs on a cold GPU at lower DVFS power/clock states. Always discard at least one initial warmup run.
3. **Assistant Slot Perplexity**: Teacher-forced perplexity is only valid when evaluated on tokens in the ASSISTANT slot, as instruction-tuned checkpoints mask prompt loss during training.
