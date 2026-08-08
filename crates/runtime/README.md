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
- `real_forward.rs`: `RealForwardRunner` short-name decode flow (`layer0.q_proj`).
- `real_forward_gemma4.rs`: `RealForwardRunner` verbatim Gemma 4 learned-weight decode flow (BF16 norms, per-head norms, INT8 router, shared expert, sandwich tail).
- `real_forward_qwen.rs` / `real_forward_qwen_attn.rs` / `real_forward_qwen_state.rs`: Qwen 3.6 decode flow (gated DeltaNet on mask-2 layers, gated full attention on mask-1, shared expert branch).
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
