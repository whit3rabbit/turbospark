# turbospark-runtime

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
|   +-- real_forward.rs         # RealForwardRunner struct, constructor, and dispatch
|   +-- real_forward_dispatch.rs# Dynamic Metal kernel dispatch helpers
|   +-- real_forward_init.rs    # Open-time arch vetting and expert streamer setup
|   +-- real_forward_layout.rs  # Quantization dtypes and MoE offset calculations
|   +-- real_forward_types.rs   # RealForwardError, PhaseCounters, DecodeScratch
|   +-- real_forward_utils.rs   # Type conversions, resident views, and host top-k
|   +-- families/               # Model-family-specific decode implementations
|   |   +-- mod.rs              # Re-exports model family submodules
|   |   +-- gemma4/             # Gemma 4 decode flow
|   |   |   +-- mod.rs          # Gemma 4 entry point & shared expert branch
|   |   |   +-- attn.rs         # Attention block & router GEMV pass
|   |   |   +-- moe.rs          # Routed MoE pass encoding
|   |   |   \-- state.rs        # RealGemmaState initialization
|   |   +-- llama/              # `llama` architecture (Mixtral) decode flow
|   |   |   +-- mod.rs          # Entry point & layer loop
|   |   |   +-- attn.rs         # Plain GQA attention block
|   |   |   +-- moe.rs          # Routed MoE pass (no shared expert)
|   |   |   \-- state.rs        # RealLlamaState & the dense-half refusal
|   |   +-- qwen/               # Qwen 3.6 decode flow
|   |   |   +-- mod.rs          # Qwen 3.6 entry point & layer loop
|   |   |   +-- attn.rs         # Gated DeltaNet & gated full attention blocks
|   |   |   +-- moe.rs          # Shared + routed MoE pass encoding
|   |   |   \-- state.rs        # RealQwenState initialization
|   |   \-- synthetic/          # Synthetic fallback decode flow
|   |       +-- mod.rs          # Synthetic entry point & host MoE FFN
|   |       \-- layer.rs        # Synthetic layer encoder
|   +-- config.rs               # Runtime completion configuration
|   \-- error.rs                # RuntimeError enum definition
\-- tests/
    +-- chunked_prefill.rs      # Chunked prefill execution unit tests
    +-- golden_tokens.rs        # Golden token sequence reproducibility tests
    +-- raw_completion.rs       # Raw completion loop integration tests
    +-- real_forward.rs         # RealForwardRunner short-name integration tests
    +-- real_forward_gemma4.rs  # RealForwardRunner Gemma 4 learned-weight tests
    +-- real_forward_llama.rs   # RealForwardRunner Mixtral-shaped decode tests
    +-- real_forward_qwen.rs    # RealForwardRunner Qwen 3.6 decode tests
    \-- fixtures/
        \-- ChatMLTokenizer/    # Toy ChatML tokenizer fixture directory for integration tests
```

## Key Modules

- `producer.rs`: `LogitProducer` trait definition and `ScriptedLogitProducer` mock implementation.
- `raw_completion.rs`: Token generation loops (`run_raw_completion` and `run_raw_completion_chunked`), integrating producer, detokenizer, stop matcher, and selection sampler.
- `real_forward.rs`: `RealForwardRunner` struct definition, options handling, and dispatch orchestration.
- `families/gemma4/`: Gemma 4 decode flow handling verbatim checkpoint weight names (`language_model.model.layers.0...`), per-head norms, learned weights, and MoE routing.
- `families/qwen/`: Qwen 3.6 decode flow — gated DeltaNet on mask-2 layers, gated full attention on mask-1, one post-attention norm feeding router + shared expert + routed experts, no sandwich norms, no softcap. Selected from `ArchConfig.family`, never from tensor naming.
- `families/llama/`: the `llama`-architecture decode flow (ROADMAP Phase M2), which is defined by its ABSENCES: plain GQA attention with no per-head norms and no output gate, a raw residual add with no sandwich norms, one post-attention norm feeding router and routed experts, no shared expert, no logit softcap, full-head NeoX RoPE at one base. **Mixtral-style MoE only.** One `general.architecture = "llama"` covers dense Llama 2/3.x and Mistral as well, and `RealLlamaState::build` refuses those by name: a dense install has no routed experts to stream, so it would need a GPU dense-FFN path (the current dense flow bridges FFN to CPU) and would abandon the memory ceiling the engine exists for. Phase 2's residual input is `scratch.zero_hidden` rather than a shared-expert output, so the routed sum is added to the stream exactly once.
- `families/synthetic/`: Short-name synthetic execution flow (`layer0.q_proj`).
- `config.rs`: Runtime generation configuration and runner settings.
- `error.rs`: `RuntimeError` enum.

## Development & Test Commands

```sh
# Run unit and integration tests for turbospark-runtime
cargo test -p turbospark-runtime
```

## Crate Gotchas

1. **PRODUCE WRITES LOGITS, NEVER PROBABILITIES**: `LogitProducer::produce` must return raw, unnormalized logits. `selection::select` performs softmaxing internally. Returning probabilities destroys sampling temperature reweighting (`softmax(softmax(z))`).
2. **`produce_prefill` may skip the output head, `produce` never may.** The prefill loop in `raw_completion.rs` calls `produce_prefill` for every prompt token but the last, because only the last one's logits are read. `RealForwardRunner` implements that by skipping the final norm, full-vocab GEMV, softcap, and host readback. Any producer overriding it must still advance every other per-token side effect (KV cache, position, command buffer commit AND wait) exactly as `produce` does: the buffer wait is what stops the next token overwriting scratch the GPU is still reading. Unrelated to `ChunkedPrefillRunner::prefill_chunk`, which does produce usable logits.
3. **Flow selection keys on `ArchConfig.family`, not on tensor naming.** Gemma 4 and Qwen 3.6 both carry `language_model.model.embed_tokens.weight`, so the naming probe can only distinguish a real Gemma install from a synthetic short-name one. Within `Gemma4` the probe still applies; `Qwen36` always builds `RealQwenState`; `DeepseekV4Flash` is refused at open.
4. **`reset()` must rewind the GDN state, not just the KV cache.** A linear-attention layer keeps its whole history in `GdnStateManager`'s delta-rule state and conv tail; the KV cache holds nothing for it. Resetting one and not the other leaks the previous generation's context into every mask-2 layer, invisibly (output stays finite and deterministic).
5. **Borrow Checker Rule in `families/gemma4/`**: Per-token forward functions interleave `let real = self.real.as_ref()` bindings with `&mut self` methods. Making a `&mut self` call invalidates existing `real` references under E0502; re-bind `real` immediately after any `&mut self` call.
6. **Phase Profiling Divisor**: `MFERENCE_PHASES=1` averages GPU phase timings over ALL forward passes (prefill tokens + decode tokens). To measure per-token decode cost at long contexts, run two tests with different `--max-new` lengths and calculate the delta.
7. **Execution Pipeline Flags**:
   - `MFERENCE_PHASES=1`: Prints GPU wait, router readback, expert `pread`, and routed bind timing breakdowns.
   - `MFERENCE_SHARED_CB=0`: Toggles overlapping the shared expert command buffer with host expert `pread`.
   - `MFERENCE_ROUTED_PIPELINE=0`: Toggles one-layer-pipelined routed command buffer execution.
   - `MFERENCE_ROUTER_HIST=/path.json`: Dumps a per-layer expert-selection histogram on runner drop (`router_hist.rs`, analyzed by `scripts/router_hist.py`). Diagnostic only; the 2026-08-08 measurement it exists for (domain-concentrated routing) came back negative, see `docs/EXPERT_ROUTING.md`.
8. **A layer's routed slots are dispatched in the ROUTER'S RANKING, and that is a correctness constraint, not a style choice.** Phase 2 reduces `blob[slot] * routing_w[slot]` in slot-index order and FP addition is not associative, so the slot order is the summation order. The Gemma flow used to order slots misses-first so the resident hits' phase-1 GEMV could ride its own command buffer (`MFERENCE_HIT_CB`, now removed); because the hit/miss split follows CACHE STATE rather than the prompt, the same prompt could decode to different text across warm runs in one process. Measured 2026-08-08: 4 distinct outputs in 6 runs on a Q8_0 GGUF install at 16 slots, 2 in 6 on the MLX install at 32. Both families are now byte-identical across 8/16/32 slots and cold vs warm. Before adding a decode-path optimization that reorders slots, ask what its ordering is a function of. See AGENTS.md Gotcha 27.
9. **Nothing about a GGUF install's block types is decided once. Every one of them is read per tensor, and the routed ones per LAYER and per PHASE.** Resident tensors key on the resident entry's dtype tag at the dispatch site (`encode_gemv_any`, `encode_embed_any`), because a real `Q4_K_M` mixes three block types in one file. The routed experts used to be the exception -- one `routed_layout: RoutedBlobLayout` off `manifest.quant.routedExpert`, on the sound premise that the blobs were uniform -- and ROADMAP Phase S broke the premise twice. Its candidate reads IQ3_XXS gate/up against an IQ4_NL down in the SAME expert (the two phases differ), and its layer 29 is IQ4_XS over Q8_0 (the layers differ), so the manifest's single `ggmlType` cannot name the kernel a dispatch wants. `routed_layouts: Vec<RoutedLayerLayout>` and `moe_offsets: Vec<MoeExpertOffsets>` are both per layer now, resolved at open from `packed_experts/layout.json`, which records a dtype per sub-tensor. Affine and GGUF are told apart by the presence of the SCALE COMPANIONS, not by the dtype string: an affine blob has nine sub-tensors and a GGUF blob three, so the test cannot drift, where a dtype allowlist would have to track that the affine writers spell their packed run `"U32"`. The dispatch sites still go through `encode_moe_phase1_any` / `encode_moe_phase2_any`, so a layout cannot disagree between two call sites and read one blob two ways. A GGUF entry has NO scale or bias companions, so its `scale_offset` is zero and `offset - index_size` underflows: resolve companion offsets inside the affine arm, never before the branch.
