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
|   |   +-- gptoss/             # `gpt-oss` decode flow (biases, sinks, YaRN, MXFP4 experts)
|   |   |   +-- mod.rs          # Entry point & layer loop
|   |   |   +-- attn.rs         # GQA + projection biases + YaRN rope + attention sinks
|   |   |   +-- moe.rs          # Routed MXFP4 pass; adds the ROUTER BIAS before the top-k
|   |   |   \-- state.rs        # RealGptOssState: YaRN table, per-layer router bias
|   |   +-- llama/              # `llama` architecture (Mixtral + dense) decode flow
|   |   |   +-- mod.rs          # Entry point & layer loop
|   |   |   +-- attn.rs         # Plain GQA attention block
|   |   |   +-- dense.rs        # Dense gated FFN (Mistral, Llama 2/3.x)
|   |   |   +-- moe.rs          # Routed MoE pass (no shared expert)
|   |   |   \-- state.rs        # RealLlamaState & the dense/MoE split
|   |   +-- qwen/               # Qwen 3.6 + dense `qwen3_5` decode flow
|   |   |   +-- mod.rs          # Entry point & layer loop
|   |   |   +-- attn.rs         # Gated DeltaNet & gated full attention blocks
|   |   |   +-- dense.rs        # Dense gated FFN (`qwen3_5`, ROADMAP's 1-bit entry)
|   |   |   +-- moe.rs          # Shared + routed MoE pass encoding
|   |   |   \-- state.rs        # RealQwenState & the dense/MoE split
|   |   \-- synthetic/          # Synthetic fallback decode flow
|   |       +-- mod.rs          # Synthetic entry point & host MoE FFN
|   |       \-- layer.rs        # Synthetic layer encoder
|   +-- config.rs               # Runtime completion configuration
|   +-- pacing.rs               # Decode-rate deadline arithmetic (Phase P2)
|   +-- power.rs                # Power profiles, thermal ladder, RateControl
|   \-- error.rs                # RuntimeError enum definition
\-- tests/
    +-- chunked_prefill.rs      # Chunked prefill execution unit tests
    +-- golden_tokens.rs        # Golden token sequence reproducibility tests
    +-- raw_completion.rs       # Raw completion loop integration tests
    +-- real_forward.rs         # RealForwardRunner short-name integration tests
    +-- real_forward_gemma4.rs  # RealForwardRunner Gemma 4 learned-weight tests
    +-- real_forward_llama.rs   # RealForwardRunner Mixtral-shaped decode tests
    +-- real_forward_llama_dense.rs # The DENSE half of the same architecture
    +-- real_forward_qwen3moe.rs# The same flow under the Qwen3-MoE family tag
    +-- real_forward_gptoss.rs  # The gpt-oss flow: perturb each input, require the logits to move
    +-- real_forward_qwen.rs    # RealForwardRunner Qwen 3.6 decode tests
    +-- real_forward_qwen35.rs  # The DENSE, ONE-BIT half of the same flow
    \-- fixtures/
        \-- ChatMLTokenizer/    # Toy ChatML tokenizer fixture directory for integration tests
```

## Key Modules

- `producer.rs`: `LogitProducer` trait definition and `ScriptedLogitProducer` mock implementation.
- `raw_completion.rs`: Token generation loops (`run_raw_completion` and `run_raw_completion_chunked`), integrating producer, detokenizer, stop matcher, and selection sampler.
- `real_forward.rs`: `RealForwardRunner` struct definition, options handling, and dispatch orchestration.
- `families/gemma4/`: Gemma 4 decode flow handling verbatim checkpoint weight names (`language_model.model.layers.0...`), per-head norms, learned weights, and MoE routing.
- `families/qwen/`: the hybrid linear/full-attention decode flow, serving **two families** — `QwenGdnMoe` and, since ROADMAP's 1-bit entry, the dense `QwenGdnDense` (Bonsai-27B, and since 2026-08-14 `Qwen/Qwen3.8-27B`, which shares its architecture exactly and differs only in quantization). Gated DeltaNet on mask-2 layers, gated full attention on mask-1, one post-attention norm feeding whatever the FFN half is, no sandwich norms, no softcap. Selected from `ArchConfig.family`, never from tensor naming. **The FFN is the only fork and it is read off `num_experts`** (`dense.rs`: `mlp.gate_proj` / `mlp.up_proj` / `silu_mul` / `mlp.down_proj` through `encode_gemv_any`, no new kernel), exactly as `families/llama/` splits Mixtral from Mistral. What licenses sharing rather than forking is measured, not assumed: every BEHAVIOURAL field of `qwen_gdn_dense_27b()` equals `qwen_gdn_moe_35b_a3b()`'s and every SHAPE field differs (`model-io`'s `qwen_gdn_dense_shares_the_moe_flows_behaviour_and_differs_in_shape`). `sharedExpertGated` is the one that legitimately parts company and `RealQwenState` checks it in BOTH directions, because it is not an independent axis: there is no shared expert to gate on a dense model. Two traps in the shared code. `intermediate_size` means the SHARED EXPERT's width on the MoE half and the DENSE FFN's width on the other, so the dense branch has to name it and never `moe_intermediate_size` (which a dense install sets to 0, encoding nothing). And a dense layer needs no mid-layer commit — nothing in it is data-dependent on a host readback the way the router's top-k is — so the whole token stays in one command buffer.
- `families/gptoss/`: the `gpt-oss` decode flow (ROADMAP M5), the FIFTH flow and the only one that is not a variation on an existing graph. Plain GQA like `llama`'s, plus four things that are each inside the layer: a BIAS on all four projections (a separate `bias_add_bf16_fp16` pass, applied BEFORE RoPE -- reversing those two is a different function that still reads fluently, since RoPE is linear and rotating a biased vector differs from biasing a rotated one by a rotation of the bias), YaRN rope through a PRECOMPUTED per-pair frequency table plus a magnitude scale of 1.3465736 (built once at open; it is position-independent), ATTENTION SINKS (one learned logit per QUERY head, not per KV head, added to the softmax denominator behind `FC_ATTN_HAS_SINKS`), and an alternating 128-token window on the EVEN layers. Two differences are NOT where a reader looks for them. The ROUTER bias lives in `moe.rs`, added on the host between the router readback and the top-k, because llama.cpp's `SOFTMAX_WEIGHT` selects on the biased RAW logits -- after the top-k it would select the wrong experts and still produce fluent text, and after the softmax it would be a different distribution over the right ones. The clamped SwiGLU and the per-expert biases are named nowhere in the flow at all: they ride inside the MXFP4 routed pair and arrive through `RoutedBlobLayout`, with `has_bias` derived from `offsets.gate_b != 0`. `RealGptOssState` refuses a tied head, a zero expert count (this architecture string has no dense half, unlike `llama`) and a missing YaRN block; all three are BACKSTOPS behind `arch_validation`, which compares the passed `ArchConfig` against the manifest field by field and fires first, so the tests for them patch `manifest.json` to agree before the flow's own check can run.
- `families/llama/`: the plain-GQA-plus-MoE decode flow (ROADMAP Phase M2), which serves **two families**, `Llama` (Mixtral) and `Qwen3Moe` (Qwen3-30B-A3B). It is defined by its ABSENCES: plain GQA attention with no per-head norms and no output gate, a raw residual add with no sandwich norms, one post-attention norm feeding router and routed experts, no shared expert, no logit softcap, full-head NeoX RoPE at one base. **BOTH HALVES OF THE ARCHITECTURE STRING RUN** since ROADMAP M4: one `general.architecture = "llama"` covers dense Llama 2/3.x and Mistral as well as the Mixtral MoEs, and `RealLlamaState` tells them apart by `num_experts` (`dense`), never by tensor naming. A dense layer swaps the router and routed experts for one gated FFN (`dense.rs`: `mlp.gate_proj` / `mlp.up_proj` / `silu_mul` / `mlp.down_proj`, all through `encode_gemv_any`, no new kernel) and is IDENTICAL above and below it -- embedding, both norms, attention, the raw residual and the head are the same code. Two non-obvious points: its width is `intermediate_size` and NOT `moe_intermediate_size` (Mixtral copies one `feed_forward_length` into both, so on the MoE half they are interchangeable and a dense checkpoint sets only the first), and a dense layer needs no mid-layer commit, because nothing in it is data-dependent on a host readback the way the router's top-k is. Phase 2's residual input is `scratch.zero_hidden` rather than a shared-expert output, so the routed sum is added to the stream exactly once. **`Qwen3Moe` differs in exactly two places, both carried by `RealLlamaState` and both keyed on `ArchConfig.family` rather than sniffed from tensor names**: it norms q and k PER HEAD before RoPE (`qk_norm`), and its RMS epsilon is 1e-6 against `llama`'s 1e-5 (`rms_eps`, which is not an `ArchConfig` field). A fourth copy of the flow with two lines changed would be a likelier source of a divergence bug than the shared one is.
- `families/synthetic/`: Short-name synthetic execution flow (`layer0.q_proj`).
- `config.rs`: Runtime generation configuration and runner settings.
- `power.rs`: ROADMAP Phase P2's policy: `PowerProfile`, `ThermalLevel`, the `stepped_cap` ladder, `RateControl`, and the two cfg-paired OS probes (`thermal_level`, `low_power_mode_enabled`) that call `crates/gpu`'s `NSProcessInfo` wrappers on macOS and return constants elsewhere.
- `pacing.rs`: the `Pacer`, pure absolute-deadline arithmetic. Reads no clock of its own (every method takes `now`), so it is testable at full speed.
- `error.rs`: `RuntimeError` enum.

## Development & Test Commands

```sh
# Run unit and integration tests for turbospark-runtime
cargo test -p turbospark-runtime
```

## Crate Gotchas

1. **PRODUCE WRITES LOGITS, NEVER PROBABILITIES**: `LogitProducer::produce` must return raw, unnormalized logits. `selection::select` performs softmaxing internally. Returning probabilities destroys sampling temperature reweighting (`softmax(softmax(z))`).
2. **`produce_prefill` may skip the output head, `produce` never may.** The prefill loop in `raw_completion.rs` calls `produce_prefill` for every prompt token but the last, because only the last one's logits are read. `RealForwardRunner` implements that by skipping the final norm, full-vocab GEMV, softcap, and host readback. Any producer overriding it must still advance every other per-token side effect (KV cache, position, command buffer commit AND wait) exactly as `produce` does: the buffer wait is what stops the next token overwriting scratch the GPU is still reading. Unrelated to `ChunkedPrefillRunner::prefill_chunk`, which does produce usable logits.
3. **Flow selection keys on `ArchConfig.family`, not on tensor naming.** `Llama` and `Qwen3Moe` share one flow and are told apart INSIDE it by the same field. Gemma 4 and Qwen 3.6 both carry `language_model.model.embed_tokens.weight`, so the naming probe can only distinguish a real Gemma install from a synthetic short-name one. Within `Gemma4` the probe still applies; `QwenGdnMoe` always builds `RealQwenState`; `DeepseekV4Flash` is refused at open.
4. **`reset()` must rewind the GDN state, not just the KV cache.** A linear-attention layer keeps its whole history in `GdnStateManager`'s delta-rule state and conv tail; the KV cache holds nothing for it. Resetting one and not the other leaks the previous generation's context into every mask-2 layer, invisibly (output stays finite and deterministic).
5. **Borrow Checker Rule in `families/gemma4/`**: Per-token forward functions interleave `let real = self.real.as_ref()` bindings with `&mut self` methods. Making a `&mut self` call invalidates existing `real` references under E0502; re-bind `real` immediately after any `&mut self` call.
6. **Phase Profiling Divisor**: `MFERENCE_PHASES=1` averages GPU phase timings over ALL forward passes (prefill tokens + decode tokens). To measure per-token decode cost at long contexts, run two tests with different `--max-new` lengths and calculate the delta.
7. **Execution Pipeline Flags**:
   - `MFERENCE_PHASES=1`: Prints GPU wait, router readback, expert `pread`, and routed bind timing breakdowns.
   - `MFERENCE_SHARED_CB=0`: Toggles overlapping the shared expert command buffer with host expert `pread`.
   - `MFERENCE_ROUTED_PIPELINE=0`: Toggles one-layer-pipelined routed command buffer execution.
   - `MFERENCE_ROUTER_HIST=/path.json`: Dumps a per-layer expert-selection histogram on runner drop (`router_hist.rs`, analyzed by `scripts/router_hist.py`). Diagnostic only; the 2026-08-08 measurement it exists for (domain-concentrated routing) came back negative, see `docs/EXPERT_ROUTING.md`.
   - `MFERENCE_ROUTER_TRACE=1`: adds the top-k ids IN PASS ORDER to that same file (`scripts/router_window.py` analyzes it). The counts cannot answer ROADMAP's speculative-decoding question, because a batched verify of M tokens reads the UNION of their routes and a histogram has already discarded which pass each selection came from.
8. **A layer's routed slots are dispatched in the ROUTER'S RANKING, and that is a correctness constraint, not a style choice.** Phase 2 reduces `blob[slot] * routing_w[slot]` in slot-index order and FP addition is not associative, so the slot order is the summation order. The Gemma flow used to order slots misses-first so the resident hits' phase-1 GEMV could ride its own command buffer (`MFERENCE_HIT_CB`, now removed); because the hit/miss split follows CACHE STATE rather than the prompt, the same prompt could decode to different text across warm runs in one process. Measured 2026-08-08: 4 distinct outputs in 6 runs on a Q8_0 GGUF install at 16 slots, 2 in 6 on the MLX install at 32. Both families are now byte-identical across 8/16/32 slots and cold vs warm. Before adding a decode-path optimization that reorders slots, ask what its ordering is a function of. See AGENTS.md Gotcha 27.
9. **The one `thread::sleep` in this crate is in `decode`, and where it sits is load-bearing.** ROADMAP Phase P2's rate cap paces AFTER the loop has decided to continue and BEFORE the next `produce`. After, so the final token of a generation never pays a sleep nobody waits through, and the stop branches break out above it. Before `produce`, so the idle window falls between forward passes rather than inside one, which is the entire point on the energy axis: the GPU has to be idle during it. It is also strictly downstream of `selection::select`, `history.push` and the progress callback, which is why pacing cannot move a token and why no quality gate is needed for a change to it (`raw_completion.rs`'s `pacing_polls_thermal_pressure_without_changing_the_tokens` is the guard). `RateControl::is_active` gates the whole block, so the default config executes the identical statement sequence it did before the feature existed. A timing test on this may only assert a LOWER bound: a cap is a floor on spacing, never a promise about the ceiling.

10. **Nothing about a GGUF install's block types is decided once. Every one of them is read per tensor, and the routed ones per LAYER and per PHASE.** Resident tensors key on the resident entry's dtype tag at the dispatch site (`encode_gemv_any`, `encode_embed_any`), because a real `Q4_K_M` mixes three block types in one file. The routed experts used to be the exception -- one `routed_layout: RoutedBlobLayout` off `manifest.quant.routedExpert`, on the sound premise that the blobs were uniform -- and ROADMAP Phase S broke the premise twice. Its candidate reads IQ3_XXS gate/up against an IQ4_NL down in the SAME expert (the two phases differ), and its layer 29 is IQ4_XS over Q8_0 (the layers differ), so the manifest's single `ggmlType` cannot name the kernel a dispatch wants. `routed_layouts: Vec<RoutedLayerLayout>` and `moe_offsets: Vec<MoeExpertOffsets>` are both per layer now, resolved at open from `packed_experts/layout.json`, which records a dtype per sub-tensor. Affine and GGUF are told apart by the presence of the SCALE COMPANIONS, not by the dtype string: an affine blob has nine sub-tensors and a GGUF blob three, so the test cannot drift, where a dtype allowlist would have to track that the affine writers spell their packed run `"U32"`. The dispatch sites still go through `encode_moe_phase1_any` / `encode_moe_phase2_any`, so a layout cannot disagree between two call sites and read one blob two ways. A GGUF entry has NO scale or bias companions, so its `scale_offset` is zero and `offset - index_size` underflows: resolve companion offsets inside the affine arm, never before the branch.

11. **A pure refactor of a family flow is a NUMERICS change, and one of them shipped a broken Qwen for a day.** `5279c88` split `real_forward_qwen.rs` into `families/qwen/`, moved no math on purpose, and inserted a Gemma-style sandwich norm anyway: `post_attention_layernorm` was applied to `scratch.o` before the residual add AND again to the stream below it. Qwen has no sandwich norms (`ffn_sandwich_norms: false` in `qwen_gdn_moe_35b_a3b()`), so the first application is both the wrong PLACE and the wrong TENSOR. Reference-answer perplexity read 255,408.97 against the frozen 6.2536 and every generation was token soup; Gemma was untouched and green throughout, which is exactly how a one-family regression hides in a workspace-wide `cargo test`. Nothing in the standing suite can see it -- the synthetic Qwen fixture's weights are untrained, so its output is meaningless by construction and a wrong norm still produces meaningless output. `qwen36_quality_gate` catches it in 77 seconds and was not run. The rule that follows: moving a decode flow between files earns the same three real-model gates as changing one, plus the family's quality gate, and "I only moved code" is the claim least worth believing about a file whose sibling implements a DIFFERENT architecture next door.

12. **The 1-bit affine GROUP SIZE is derived from the tensor's own companion planes, never named.** The resident index records a dtype tag and three regions and no group size, so the dtype-15 arms of `encode_gemv_any` and `encode_embed_any` compute it: one FP16 scale per group per row, hence `groups_per_row = scale_size / (2 * rows)` and `group_size = cols / groups_per_row` (`real_forward_utils::int1_group_size`). The alternative was a `BONSAI_GROUP_SIZE = 128` constant here, which is the shape of AGENTS.md Gotchas 37 and 38 -- a per-checkpoint number standing in for a per-tensor property, correct exactly as long as one checkpoint exercises it -- and `crates/gpu` had already declined the same constant for the same reason, taking the value as an argument. Two consequences worth knowing. The derivation doubles as the shape check the other affine arms spell out inline, and the group-is-a-whole-number-of-bytes conjunct is not decoration: the kernel ASSERTS that, so a malformed install reaching the dispatch would abort the process rather than return an error. And the embedding arm has to recover the ROW COUNT first (`size_bytes * 8 / hidden`), because its callers know the hidden size and the token id and never the vocabulary. **The SYMMETRIC (`+/-1`) kernel is deliberately unreachable from either arm**: it is a different summation order, so selecting it per tensor at dispatch time would make the generated bytes a function of which tensors happened to quantize symmetrically (Gotcha 8's territory). Wiring it is a repack-time decision and needs its own dtype tag.
