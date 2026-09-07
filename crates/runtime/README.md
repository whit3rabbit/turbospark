# turbospark-runtime

Raw-completion prefill and decode generation loops (`run_raw_completion`, `run_raw_completion_chunked`), `LogitProducer` trait definition, `ScriptedLogitProducer` test mock, and `RealForwardRunner` GPU forward-pass engine (macOS).

Downstream workspace crates import this package via the `runtime` alias:

```toml
[dependencies]
runtime = { package = "turbospark-runtime", path = "../runtime" }
```

## Safety

- `#![forbid(unsafe_code)]` is enforced in this crate.

## Key Modules

- `producer.rs`: `LogitProducer` trait definition and `ScriptedLogitProducer` mock implementation.
- `raw_completion.rs`: Token generation loops (`run_raw_completion` and `run_raw_completion_chunked`), integrating producer, detokenizer, stop matcher, and selection sampler.
- `real_forward.rs`: `RealForwardRunner` struct definition, options handling, and dispatch orchestration.
- `real_forward_dispatch.rs`: Metal kernel dispatchers (`encode_gemv_any`, `encode_embed_any`); `real_forward_dispatch_moe.rs` holds the MoE pair (`encode_moe_phase1_any`, `encode_moe_phase2_any`).
- `real_forward_layout.rs`: Quantization/GGUF dtypes and MoE blob offset calculations.
- `real_forward_types.rs`: `RealForwardError`, `PhaseCounters`, `dispatch_profile_report`, and `DecodeScratch`.
- `real_forward_init.rs`: Architecture validation and expert streamer setup.
- `families/gemma4/`: Gemma 4 decode flow (`mod.rs`, `attn.rs`, `moe.rs`, `prefill.rs`, `state.rs`, and others).
- `families/qwen/`: Qwen 3.6 + dense `qwen3_5` decode flow (`mod.rs`, `attn.rs`, `moe.rs`, `prefill.rs`, `state.rs`, and others).
- `families/synthetic/`: Synthetic short-name fallback flow (`mod.rs`, `layer.rs`).
- `config.rs`: Runtime generation configuration and runner parameters.

## Development & Test Commands

```sh
# Run unit and integration tests for turbospark-runtime
cargo test -p turbospark-runtime
```

## Crate Gotchas

1. **Logit Producer Contract**: `LogitProducer::produce` MUST return raw unnormalized logits. `selection::select` performs softmaxing internally. Returning probabilities destroys sampling temperature scaling (`softmax(softmax(z))`).
2. **Family Selection**: Flow selection keys on `ArchConfig.family`, not on tensor naming. Gemma 4 and Qwen 3.6 both use standard weight names, so the family enum determines architectural routing.
3. **GDN State Reset**: Calling `reset()` on a runner must rewind both the KV cache and `GdnStateManager` recurrent state (delta rule state `S` and conv tail) to prevent context leakage across runs.
