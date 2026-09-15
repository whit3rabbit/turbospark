# turbospark-bench

Throughput benchmark harness (`turbospark-bench`), Mach physical memory footprint sampler (`memory.rs`), frozen community benchmark protocol (`protocol.rs`), real model runner (`real_model.rs`), and memory oracle / quality gate test suites.

Detailed benchmark methodology, frozen numbers, and hardware baselines are documented in [`docs/BENCHMARKING.md`](../../docs/BENCHMARKING.md) and [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md).

## Purpose & Role

`turbospark-bench` provides reproducible, frozen benchmarking of token generation throughput, prefill latency, and physical memory footprint across all supported model architectures. It includes memory oracle tests that guard against memory leaks or memory-pinning regressions and quality sensitivity gates that measure numeric drift.

## Binary Execution Modes

1. **Scripted Producer Overhead Mode**:
   Measures raw framework scheduling and detokenization throughput without GPU forward passes:
   ```sh
   cargo run -p turbospark-bench --bin turbospark-bench -- <tokenizer-dir>
   ```

2. **Real Model Protocol Benchmark**:
   Executes the full frozen community benchmark suite across a sequence of prompt lengths:
   ```sh
   cargo run --release -p turbospark-bench --bin turbospark-bench -- \
     --model ~/models/gemma4.gturbo
   ```

3. **Single Case Isolation Mode**:
   Runs a single benchmark scenario in a fresh process to isolate memory and DVFS effects:
   ```sh
   cargo run --release -p turbospark-bench --bin turbospark-bench -- \
     --model ~/models/gemma4.gturbo --case short-explanation
   ```

## Key Modules

- `main.rs`: Benchmark binary entry point and CLI mode dispatcher.
- `memory.rs`: Apple Silicon Mach kernel task info sampler querying `phys_footprint` (resident physical memory).
- `protocol.rs`: Frozen community benchmark definitions, prompt lengths, and generation targets.
- `model_mode.rs`: Command-line options for real-model execution.
- `real_model.rs`, `real_model_open.rs`, `real_model_params.rs`: Hardware runner initializing `RealForwardRunner` and executing timed generation steps.
- `scripted.rs`: Mock runner evaluating host tokenizer and detokenizer overhead.

## Development & Test Commands

```sh
# Run offline unit tests
cargo test -p turbospark-bench

# Run memory oracle test for Gemma 4 (macOS, requires model environment variable)
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test memory_oracle --release -- --ignored --nocapture

# Run quality gate test for Qwen 3.6
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-bench --test qwen36_quality_gate --release -- --ignored --nocapture
```

## Tests

This crate contains 57 test files in `tests/`:
- Memory oracles (verifying peak physical memory ceilings):
  `memory_oracle.rs` (Gemma 4), `qwen36_memory_oracle.rs`, `qwen38_memory_oracle.rs`, `qwen3moe_memory_oracle.rs`, `qwen3_dense_memory_oracle.rs`, `qwen4exp_memory_oracle.rs`, `minimax_memory_oracle.rs`, `mistral_memory_oracle.rs`, `spark_memory_oracle.rs`, `ternary_memory_oracle.rs`, `museglimmer_memory_oracle.rs`, `ornith9b_memory_oracle.rs`, `ornith35b_memory_oracle.rs`, `gptoss_memory_oracle.rs`, `vision_memory_oracle.rs`, `vision_sidecar_memory_oracle.rs`.
- Quality gates (evaluating token digest exactness and perplexity):
  `quality_gate.rs`, `qwen36_quality_gate.rs`, `qwen38_quality_gate.rs`, `qwen3moe_quality_gate.rs`, `qwen3_dense_quality_gate.rs`, `qwen4exp_quality_gate.rs`, `minimax_quality_gate.rs`, `mistral_quality_gate.rs`, `spark_quality_gate.rs`, `ternary_quality_gate.rs`, `museglimmer_quality_gate.rs`, `ornith9b_quality_gate.rs`, `ornith35b_quality_gate.rs`, `gptoss_quality_gate.rs`, `iq3_quality_gate.rs`.
- Architectural & speculative probes:
  `steering_probe.rs`, `steering_sweep.rs`, `dflash2_accept_length_probe.rs`, `mtp_head_probe.rs`, `mtp_accept_length_probe.rs`, `kv_quant_probe.rs`, `quality_sensitivity.rs`, `batched_forward_probe.rs`, `mapped_residency_eviction.rs`, `logit_dump.rs`, `vision_logit_dump.rs`.

## Crate Gotchas

1. **Physical Footprint Measurement (`phys_footprint`)**: Footprint measurements query Apple Silicon Mach kernel task info. This encompasses all physical memory committed to the process, including zero-copy `MTLBuffer` mappings around mapped model weights, the KV cache, expert slot buffers, and framework heap.
2. **Cold GPU DVFS Warmup**: Apple Silicon dynamically scales GPU clock frequencies (DVFS). Initial passes on a cold device execute at lower power states; reliable throughput benchmarks must discard at least one initial warmup iteration.
3. **Assistant Slot Perplexity**: Teacher-forced perplexity is only valid when evaluated strictly over tokens in the assistant turn slot, as instruction-tuned checkpoints mask prompt loss during training.
