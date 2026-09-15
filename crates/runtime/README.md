# turbospark-runtime

Raw-completion prefill and decode generation loops (`run_raw_completion`, `run_raw_completion_chunked`), `LogitProducer` trait definition, `ScriptedLogitProducer` test mock, and `RealForwardRunner` GPU forward-pass engine (macOS).

Downstream workspace crates import this package via the `runtime` alias:

```toml
[dependencies]
runtime = { package = "turbospark-runtime", path = "../runtime" }
```

## Purpose & Role

`turbospark-runtime` is the central execution engine of the workspace. It binds tokenization, KV cache management, model forward passes, candidate token selection, and streaming detokenization into a unified generation pipeline. On macOS with Apple Silicon, `RealForwardRunner` orchestrates GPU compute dispatches for all supported model architectures.

## Safety

- `#![forbid(unsafe_code)]` is enforced in `lib.rs`.
- Zero raw pointer manipulation or unchecked buffer casts.

## Key Modules

- `producer.rs`: `LogitProducer` trait definition and `ScriptedLogitProducer` deterministic test mock.
- `raw_completion.rs` & `raw_completion_chunked.rs`: Generation loop drivers orchestrating prefill chunking, step-by-step decode, stop token matching, and streaming callbacks.
- `real_forward.rs`: `RealForwardRunner` struct definition, execution lifecycle, and dispatch routing.
- `real_forward_open.rs`: Model open routines mapping `.gturbo` weight files and initializing device contexts.
- `real_forward_init.rs`: Architecture baseline validation, weight buffer binding, and expert streamer setup.
- `real_forward_api.rs` & `real_forward_traits.rs`: Public API abstractions for single-token decode and multi-token prefill.
- `real_forward_dispatch.rs` & `real_forward_dispatch_moe.rs`: Metal kernel dispatch orchestrators (`encode_gemv_any`, `encode_embed_any`, `encode_moe_phase1_any`, `encode_moe_phase2_any`).
- `real_forward_layout.rs`: Quantization dtype translation, block sizes, and MoE blob offset calculations.
- `real_forward_rollback.rs`: Atomic rollback of KV cache and recurrent state for speculative rejection.
- `real_forward_types.rs`: `RealForwardError`, `PhaseCounters`, `dispatch_profile_report`, and `DecodeScratch`.
- `real_forward_vision_api.rs` & `vision/`: Vision tower forward pass dispatches and multimodal token injection.
- `families/`: Dedicated per-architecture forward flows:
  - `gemma4/`: Gemma 4 attention, MoE routing, and layer norms.
  - `gptoss/`: Harmony and GPT-OSS architectures.
  - `llama/`: Llama 3, Mixtral MoE, and related dense models.
  - `museglimmer/`: Muse Glimmer architecture.
  - `qwen/`: Qwen 3.5, Qwen 3.6 (GDN hybrid), Qwen 3.8, Qwen 2, and Qwen3-MoE.
  - `qwen4/`: `qwen4_exp` QSA indexer and PLE n-gram routing flow.
  - `spark/`: Spark-X2.5-4B fused QKV and headwise gated architecture.
  - `synthetic/`: Synthetic fallback flow for headless unit tests.
- `steering.rs`: Multi-vector composed directional steering applied directly to residual streams during forward passes.
- `speculative.rs` & `speculation_policy.rs`: Speculative decoding drivers supporting DFlash block drafting and MTP multi-token prediction heads.
- `session_pool.rs`: Thread-safe session pooling for concurrent or reused model runners.
- `turn_stream.rs`: Push-callback stream adapter bridging generation loops with async channels and FFI callbacks.
- `resid_capture.rs`, `router_hist.rs`, `ffn_hist.rs`: Diagnostic inspection capturing internal activations, routing weights, and neuron firing histograms.
- `kv_prefix.rs` & `kv_write.rs`: Prefix KV cache matching and incremental KV cache writing.
- `pacing.rs` & `power.rs`: Decode pacing loops and Apple Silicon thermal/power throttling controls.
- `config.rs`: Runtime generation configuration and runner parameter tuning.

## Development & Test Commands

```sh
# Run all unit and integration tests for turbospark-runtime
cargo test -p turbospark-runtime

# Run a specific family integration test
cargo test -p turbospark-runtime --test real_forward_gemma4
```

## Tests

This crate contains 59 integration test suites in `tests/`:
- Model family forward passes: `real_forward_gemma4.rs`, `real_forward_gptoss.rs`, `real_forward_llama.rs`, `real_forward_llama_dense.rs`, `real_forward_minimax.rs`, `real_forward_muse.rs`, `real_forward_qwen.rs`, `real_forward_qwen2.rs`, `real_forward_qwen35.rs`, `real_forward_qwen3moe.rs`, `real_forward_qwen4.rs`, `real_forward_spark.rs`.
- Chunked prefill & prefix reuse: `chunked_prefill.rs`, `chunked_prefill_refusal.rs`, `prefix_reuse_real.rs`, `real_forward_gemma4_chunked.rs`, `real_forward_qwen35_chunked.rs`.
- Quantized KV cache forward tests: `real_forward_gemma4_kv_quant.rs`, `real_forward_gptoss_kv_quant.rs`, `real_forward_llama_kv_quant.rs`, `real_forward_qwen35_kv_quant.rs`, `real_forward_spark_kv_quant.rs`.
- Directional steering: `real_forward_gemma4_steered.rs`, `real_forward_gptoss_steered.rs`, `real_forward_llama_steered.rs`, `real_forward_museglimmer_steered.rs`, `real_forward_qwen35_steered_batched.rs`.
- Speculative decoding & drafting: `real_forward_qwen35_dflash.rs`, `real_forward_qwen35_mtp.rs`, `speculative.rs`.
- Vision tower integration: `vision_tower_parity.rs`, `vision_tower_synthetic.rs`, `vision_inject_synthetic.rs`, `vision_chunked_synthetic.rs`, `vision_sidecar_synthetic.rs`.
- Memory residency & eviction: `mapped_expert_residency.rs`, `mapped_vision_residency.rs`.

## Crate Gotchas

1. **Logit Producer Contract**: `LogitProducer::produce` must return raw, unnormalized logits. `selection::select` performs softmaxing internally. Returning probabilities destroys sampling temperature scaling (`softmax(softmax(z))`).
2. **Family Selection via ArchConfig**: Model execution flow keys strictly on `ArchConfig.family`, never on weight file names or tensor prefixes.
3. **GDN State Reset**: Calling `reset()` on a runner must rewind both the KV cache and `GdnStateManager` recurrent state (delta rule state `S` and conv tail buffer) to avoid state bleeding across independent conversation turns.
4. **Speculative Rollback Atomicity**: In speculative decoding with rejection, rollback must restore the KV cache position, GDN state, and step counter in a single atomic operation before evaluating subsequent draft tokens.
