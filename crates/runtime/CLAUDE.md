# mrefrust-runtime

Raw-completion prefill and decode generation loops (`run_raw_completion`, `run_raw_completion_chunked`), `LogitProducer` trait definition, `ScriptedLogitProducer` test mock, and `RealForwardRunner` GPU forward-pass engine (macOS).

## Safety

- `#![forbid(unsafe_code)]` is enforced in this crate.

## Directory & File Structure

```
crates/runtime/
+-- Cargo.toml                  # Crate manifest
+-- src/
|   +-- lib.rs                  # Library root
|   +-- producer.rs             # LogitProducer trait & ScriptedLogitProducer mock
|   +-- raw_completion.rs       # Generation loops (run_raw_completion & run_raw_completion_chunked)
|   +-- real_forward.rs         # RealForwardRunner short-name decode flow (layer0.q_proj)
|   +-- real_forward_gemma4.rs  # RealForwardRunner verbatim Gemma 4 learned-weight decode flow
|   +-- real_forward_qwen.rs    # RealForwardRunner Qwen 3.6 decode flow (state, MoE tail, head)
|   +-- real_forward_qwen_attn.rs # Qwen 3.6 per-layer blocks (gated DeltaNet, gated full attention)
|   +-- real_forward_qwen_state.rs # RealQwenState: open-time arch vetting, GDN buffers, scratch
|   +-- config.rs               # Runtime completion configuration
|   \-- error.rs                # RuntimeError enum definition
\-- tests/
    +-- chunked_prefill.rs      # Chunked prefill execution unit tests
    +-- golden_tokens.rs        # Golden token sequence reproducibility tests
    +-- raw_completion.rs       # Raw completion loop integration tests
    +-- real_forward.rs         # RealForwardRunner short-name integration tests
    +-- real_forward_gemma4.rs  # RealForwardRunner Gemma 4 learned-weight tests
    +-- real_forward_qwen.rs    # RealForwardRunner Qwen 3.6 decode tests
    \-- fixtures/
        \-- ChatMLTokenizer/    # Toy ChatML tokenizer fixture directory for integration tests
```

## Key Modules

- `producer.rs`: `LogitProducer` trait definition and `ScriptedLogitProducer` mock implementation.
- `raw_completion.rs`: Token generation loops (`run_raw_completion` and `run_raw_completion_chunked`), integrating producer, detokenizer, stop matcher, and selection sampler.
- `real_forward.rs`: `RealForwardRunner` handling synthetic short-name tensor indexing (`layer0.q_proj`).
- `real_forward_gemma4.rs`: `RealForwardRunner` handling verbatim real Gemma 4 checkpoint weight names (`language_model.model.layers.0...`), per-head norms, learned weights, and MoE routing.
- `real_forward_qwen.rs` / `real_forward_qwen_attn.rs` / `real_forward_qwen_state.rs`: the Qwen 3.6 flow -- gated DeltaNet on mask-2 layers, gated full attention on mask-1, one post-attention norm feeding router + shared expert + routed experts, no sandwich norms, no softcap. Selected from `ArchConfig.family`, never from tensor naming.
- `config.rs`: Runtime generation configuration and runner settings.
- `error.rs`: `RuntimeError` enum.

## Development & Test Commands

```sh
# Run unit and integration tests for mrefrust-runtime
cargo test -p mrefrust-runtime
```

## Crate Gotchas

1. **PRODUCE WRITES LOGITS, NEVER PROBABILITIES**: `LogitProducer::produce` must return raw, unnormalized logits. `selection::select` performs softmaxing internally. Returning probabilities destroys sampling temperature reweighting (`softmax(softmax(z))`).
2. **`produce_prefill` may skip the output head, `produce` never may.** The prefill loop in `raw_completion.rs` calls `produce_prefill` for every prompt token but the last, because only the last one's logits are read. `RealForwardRunner` implements that by skipping the final norm, full-vocab GEMV, softcap, and host readback. Any producer overriding it must still advance every other per-token side effect (KV cache, position, command buffer commit AND wait) exactly as `produce` does: the buffer wait is what stops the next token overwriting scratch the GPU is still reading. Unrelated to `ChunkedPrefillRunner::prefill_chunk`, which does produce usable logits.
3. **Flow selection keys on `ArchConfig.family`, not on tensor naming.** Gemma 4 and Qwen 3.6 both carry `language_model.model.embed_tokens.weight`, so the naming probe can only distinguish a real Gemma install from a synthetic short-name one. Within `Gemma4` the probe still applies; `Qwen36` always builds `RealQwenState`; `DeepseekV4Flash` is refused at open.
4. **`reset()` must rewind the GDN state, not just the KV cache.** A linear-attention layer keeps its whole history in `GdnStateManager`'s delta-rule state and conv tail; the KV cache holds nothing for it. Resetting one and not the other leaks the previous generation's context into every mask-2 layer, invisibly (output stays finite and deterministic).
5. **Borrow Checker Rule in `real_forward_gemma4.rs`**: Per-token forward functions interleave `let real = self.real.as_ref()` bindings with `&mut self` methods. Making a `&mut self` call invalidates existing `real` references under E0502; re-bind `real` immediately after any `&mut self` call.
6. **Phase Profiling Divisor**: `MFERENCE_PHASES=1` averages GPU phase timings over ALL forward passes (prefill tokens + decode tokens). To measure per-token decode cost at long contexts, run two tests with different `--max-new` lengths and calculate the delta.
7. **Execution Pipeline Flags**:
   - `MFERENCE_PHASES=1`: Prints GPU wait, router readback, expert `pread`, and routed bind timing breakdowns.
   - `MFERENCE_SHARED_CB=0`: Toggles overlapping the shared expert command buffer with host expert `pread`.
   - `MFERENCE_HIT_CB=0`: Toggles dispatching cache-hit expert GEMVs prior to expert `pread`.
   - `MFERENCE_ROUTED_PIPELINE=0`: Toggles one-layer-pipelined routed command buffer execution.
8. **The misses-first slot order in `real_forward_gemma4.rs` changes OUTPUT BYTES, and it is unconditional.** The Gemma flow orders a layer's routed slots misses first then hits so the hits' phase-1 GEMV can ride its own command buffer. That order is what `moe_phase2_down_reduce_k8` reduces in, and FP addition is not associative, so Gemma's generated text differs (deterministically) between expert-cache sizes and between a cold and a warm cache. `real_forward_qwen.rs` does no reordering and is byte-identical across slot counts. `MFERENCE_HIT_CB=0` toggles the separate command buffer, NOT the order, so it does not undo this: measured directly, it moves no digest. Consequences: any A/B that holds output constant compares within one slot count, and `crates/bench/tests/quality_gate.rs` freezes a separate digest per slot count instead of asserting them equal.
