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
|   |   |   +-- mtp.rs          # The MTP head's draft step (MtpState, its own one-layer KV)
|   |   |   \-- state.rs        # RealQwenState & the dense/MoE split
|   |   \-- synthetic/          # Synthetic fallback decode flow
|   |       +-- mod.rs          # Synthetic entry point & host MoE FFN
|   |       \-- layer.rs        # Synthetic layer encoder
|   +-- config.rs               # Runtime completion configuration
|   +-- pacing.rs               # Decode-rate deadline arithmetic (Phase P2)
|   +-- power.rs                # Power profiles, thermal ladder, RateControl
|   \-- error.rs                # RuntimeError enum definition
\-- tests/
    +-- chunked_prefill.rs      # Chunked prefill loop unit tests (scripted producer)
    +-- real_forward_gemma4_chunked.rs # The REAL chunk driver, against a non-chunked reference
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
- `families/museglimmer/`: the `muse_glimmer` decode flow, the SIXTH flow and the seventh family (`mlx-community/Muse-Glimmer-30B-4bit`). Dense 52-layer GQA (32 q over 2 kv, head_dim 128), a three-sliding/one-full window at 2048, sandwich norms and a logit softcap. Its own flow on the `gpt-oss` precedent: TEN differences, every one inside the layer. Four are worth naming because no other flow has them. **Its four per-layer norms are CENTERED (`x * (1 + w)`) and its FINAL norm is PLAIN (`x * w`)** -- two conventions in one model, dispatched per tensor through `rmsnorm_bf16w_centered` and `rmsnorm_bf16w` (AGENTS.md Gotcha 50). **It carries TWO RMS epsilons**, 1e-5 on the input/pre-FFN/q-k/embedding norms and 1e-8 on the two POST norms, where every other flow carries one. **Its full-attention layers are NoPE** -- `layer_rope_theta` is literally 0 there in the checkpoint, carried as `full_rope_theta: 0.0`, and `RealMuseState` refuses an install that says otherwise, because a rotated NoPE layer is fluent and wrong. And **its attention output gate is its own tensor** (`self_attn.gate_proj`, fed from the layer's normed input), not Qwen's packing into `q_proj` -- which is why `attn_output_gate` is FALSE on this family despite it having a gate. Two published scalars (`qk_scale_factor` 3.87 on Q after its no-scale per-head norm, `output_multiplier` 26^-0.5 on the logits before the softcap) and both epsilons are family CONSTANTS in `state.rs` rather than `ArchConfig` fields, following the `rms_eps` precedent -- none is a binary fraction and `arch_validation` compares manifest floats with `!=` (Gotcha 24); `crates/repack/tests/museglimmer_config.rs` parses the real config and asserts all four offline.
- `families/synthetic/`: Short-name synthetic execution flow (`layer0.q_proj`).
- `config.rs`: Runtime generation configuration and runner settings.
- `power.rs`: ROADMAP Phase P2's policy: `PowerProfile`, `ThermalLevel`, the `stepped_cap` ladder, `RateControl`, and the two cfg-paired OS probes (`thermal_level`, `low_power_mode_enabled`) that call `crates/gpu`'s `NSProcessInfo` wrappers on macOS and return constants elsewhere.
- `pacing.rs`: the `Pacer`, pure absolute-deadline arithmetic. Reads no clock of its own (every method takes `now`), so it is testable at full speed.
- `context_policy.rs`: `MaxContext` and how `Auto` resolves, plus
  `kv_bytes_for_context`, which mirrors `KvCacheManager::new`'s allocation
  exactly. Portable for the same reason its sibling is -- every input is a
  parameter, so the whole policy is unit-tested anywhere. The one exception is
  `committed_bytes`, which reads the INSTALL (not the machine) to add the
  worst-case slot cache to the mapped weight region. See Gotcha 15.
- `expert_cache_policy.rs`: `ExpertCacheSlots` and how `Auto` resolves. Portable on purpose -- no `gpu`, no `cfg`, no probe of its own; `resolve` takes the machine's memory as a parameter, so its whole test suite runs on any platform rather than needing a Mac with an install on disk. See Gotcha 13.
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
   - `MFERENCE_FFN_HIST=/path.json`: dense-FFN activation census on runner drop (`ffn_hist.rs`, analyzed by `scripts/ffn_sparsity.py`; museGlimmer only, the one flow that feeds its capture). Redirects `silu_mul` into a per-layer capture buffer, so it changes no math and no output bytes; costs ~30% of decode throughput while on. The 2026-08-16 measurement it exists for (a PowerInfer-style neuron cache) came back negative, see `docs/ACTIVATION_SPARSITY.md`.
   - `MFERENCE_MTP_DRAFT=<depth>`: builds `families/qwen/mtp.rs`'s `MtpState` and lets `RealForwardRunner::mtp_draft_step` run (`docs/MTP_SPECULATIVE.md`, step 2; `qwen3_5` only). Unset, unparsable or 0 allocates NOTHING and encodes nothing, so the off path is identical in bytes and in footprint to the engine that shipped before the module existed -- which is what lets `qwen38_memory_oracle`'s frozen row stand rather than needing a new one. A depth asked for on an install with no head is an ERROR at open naming `mtp.fc.weight`, never a silent no-op: a caller that asked for speculation and quietly got none would measure the non-speculative engine and report it as the speculative one (Gotcha 14's argument, one feature over).
   - `MFERENCE_PREFILL_CHUNK=<tokens>`: routes prefill through `run_raw_completion_chunked` and `RealForwardRunner`'s chunk driver (Gotcha 14). An A/B seam like the two above it, not a feature flag: both arms must produce identical tokens. Unset, unparsable or 0 is the sequential path. `--prefill-chunk` exists in `crates/invocation`, is validated against `ALLOWED_CHUNK_SIZES`, and is wired to NOTHING on purpose -- it defaults to `Fixed(128)`, so wiring it turns chunked prefill on by default, which this phase has not earned across families yet.
8. **A layer's routed slots are dispatched in the ROUTER'S RANKING, and that is a correctness constraint, not a style choice.** Phase 2 reduces `blob[slot] * routing_w[slot]` in slot-index order and FP addition is not associative, so the slot order is the summation order. The Gemma flow used to order slots misses-first so the resident hits' phase-1 GEMV could ride its own command buffer (`MFERENCE_HIT_CB`, now removed); because the hit/miss split follows CACHE STATE rather than the prompt, the same prompt could decode to different text across warm runs in one process. Measured 2026-08-08: 4 distinct outputs in 6 runs on a Q8_0 GGUF install at 16 slots, 2 in 6 on the MLX install at 32. Both families are now byte-identical across 8/16/32 slots and cold vs warm. Before adding a decode-path optimization that reorders slots, ask what its ordering is a function of. See AGENTS.md Gotcha 27.
9. **The one `thread::sleep` in this crate is in `decode`, and where it sits is load-bearing.** ROADMAP Phase P2's rate cap paces AFTER the loop has decided to continue and BEFORE the next `produce`. After, so the final token of a generation never pays a sleep nobody waits through, and the stop branches break out above it. Before `produce`, so the idle window falls between forward passes rather than inside one, which is the entire point on the energy axis: the GPU has to be idle during it. It is also strictly downstream of `selection::select`, `history.push` and the progress callback, which is why pacing cannot move a token and why no quality gate is needed for a change to it (`raw_completion.rs`'s `pacing_polls_thermal_pressure_without_changing_the_tokens` is the guard). `RateControl::is_active` gates the whole block, so the default config executes the identical statement sequence it did before the feature existed. A timing test on this may only assert a LOWER bound: a cap is a floor on spacing, never a promise about the ceiling.

10. **Nothing about a GGUF install's block types is decided once. Every one of them is read per tensor, and the routed ones per LAYER and per PHASE.** Resident tensors key on the resident entry's dtype tag at the dispatch site (`encode_gemv_any`, `encode_embed_any`), because a real `Q4_K_M` mixes three block types in one file. The routed experts used to be the exception -- one `routed_layout: RoutedBlobLayout` off `manifest.quant.routedExpert`, on the sound premise that the blobs were uniform -- and ROADMAP Phase S broke the premise twice. Its candidate reads IQ3_XXS gate/up against an IQ4_NL down in the SAME expert (the two phases differ), and its layer 29 is IQ4_XS over Q8_0 (the layers differ), so the manifest's single `ggmlType` cannot name the kernel a dispatch wants. `routed_layouts: Vec<RoutedLayerLayout>` and `moe_offsets: Vec<MoeExpertOffsets>` are both per layer now, resolved at open from `packed_experts/layout.json`, which records a dtype per sub-tensor. Affine and GGUF are told apart by the presence of the SCALE COMPANIONS, not by the dtype string: an affine blob has nine sub-tensors and a GGUF blob three, so the test cannot drift, where a dtype allowlist would have to track that the affine writers spell their packed run `"U32"`. The dispatch sites still go through `encode_moe_phase1_any` / `encode_moe_phase2_any`, so a layout cannot disagree between two call sites and read one blob two ways. A GGUF entry has NO scale or bias companions, so its `scale_offset` is zero and `offset - index_size` underflows: resolve companion offsets inside the affine arm, never before the branch.

11. **A pure refactor of a family flow is a NUMERICS change, and one of them shipped a broken Qwen for a day.** `5279c88` split `real_forward_qwen.rs` into `families/qwen/`, moved no math on purpose, and inserted a Gemma-style sandwich norm anyway: `post_attention_layernorm` was applied to `scratch.o` before the residual add AND again to the stream below it. Qwen has no sandwich norms (`ffn_sandwich_norms: false` in `qwen_gdn_moe_35b_a3b()`), so the first application is both the wrong PLACE and the wrong TENSOR. Reference-answer perplexity read 255,408.97 against the frozen 6.2536 and every generation was token soup; Gemma was untouched and green throughout, which is exactly how a one-family regression hides in a workspace-wide `cargo test`. Nothing in the standing suite can see it -- the synthetic Qwen fixture's weights are untrained, so its output is meaningless by construction and a wrong norm still produces meaningless output. `qwen36_quality_gate` catches it in 77 seconds and was not run. The rule that follows: moving a decode flow between files earns the same three real-model gates as changing one, plus the family's quality gate, and "I only moved code" is the claim least worth believing about a file whose sibling implements a DIFFERENT architecture next door.

12. **The sub-4-bit affine GROUP SIZE is derived from the tensor's own companion planes, never named.** The resident index records a dtype tag and three regions and no group size, so the dtype-15 and dtype-16 arms of `encode_gemv_any` and `encode_embed_any` compute it: one FP16 scale per group per row, hence `groups_per_row = scale_size / (2 * rows)` and `group_size = cols / groups_per_row` (`real_forward_utils::affine_group_size`, which takes the WIDTH as a parameter -- the packed-size conjunct is the only thing that varies between one bit and two, and it is also the only thing that tells the two entry shapes apart, since both carry three planar regions with FP16 companions). The alternative was a `BONSAI_GROUP_SIZE = 128` constant here, which is the shape of AGENTS.md Gotchas 37 and 38 -- a per-checkpoint number standing in for a per-tensor property, correct exactly as long as one checkpoint exercises it -- and `crates/gpu` had already declined the same constant for the same reason, taking the value as an argument. Two consequences worth knowing. The derivation doubles as the shape check the other affine arms spell out inline, and the group-is-a-whole-number-of-bytes conjunct is not decoration: the kernel ASSERTS that, so a malformed install reaching the dispatch would abort the process rather than return an error. And the embedding arm has to recover the ROW COUNT first (`size_bytes * 8 / hidden`), because its callers know the hidden size and the token id and never the vocabulary. **ROADMAP's ternary entry added the dtype-16 arms beside these and they are the same code with one constant moved** (`prism-ml/Ternary-Bonsai-27B-mlx-2bit`, 2-bit affine at group 128, FP16 companions, the same dense `qwen3_5` architecture Bonsai-27B is). Its embedding arm is the one worth knowing about: swapping in the 1-bit lookup leaves every other case in `real_forward_qwen35.rs` green, because the wrong kernel still reads the table and still moves the logits when it is perturbed -- it strides by `D / 8` where a 2-bit table strides by `D / 4`, so it reads the WRONG ROW. `the_embedding_lookup_strides_at_the_tables_own_width` patches a byte range the wrong stride does not read for that token, and asserts the two ranges are disjoint before relying on it. **The SYMMETRIC (`+/-1`) kernel is deliberately unreachable from either arm**: it is a different summation order, so selecting it per tensor at dispatch time would make the generated bytes a function of which tensors happened to quantize symmetrically (Gotcha 8's territory). Wiring it is a repack-time decision and needs its own dtype tag.

13. **The expert-cache slot count is `auto` by DEFAULT, and what makes an environment-sensing default safe here is that the axis is throughput-only.** `ExpertCacheSlots::Auto` picks the largest allowed count whose `slots x sum(expert_stride)` fits a quarter of `physical - resident - 4 GiB`, and `open_with_slot_policy` resolves it inside `open_expert_streamers`, where the layout is already loaded. A slot-count change cannot move a digest or a perplexity: routed slots dispatch in the router's own ranking, so output has been byte-identical across 8/16/24/32 since Gotcha 8's fix, and that is what separates this from `--power-profile`, whose sensing default had to be kept away from harnesses for a numerics-adjacent reason as well as a measurement one.

    **Three things are load-bearing and each would be easy to undo by accident.**

    The FLOOR is `DEFAULT_CACHE_SLOTS`, not the bottom of `ALLOWED_CACHE_SLOTS`. Budgeting from headroom alone resolves a 13 GB install on a 16 GB machine to 8 slots, which is SLOWER than the 16 that shipped before the feature existed -- a regression aimed squarely at the users least able to absorb one. `auto_can_never_resolve_below_the_shipped_default` sweeps machine sizes, install sizes and expert granularities and asserts the rule directly rather than leaving the two example cases to imply it.

    `open_with_options` still takes a plain `usize` and is NOT widened to the policy type. That is the whole guarantee that no measuring caller can acquire the sensing default: `turbospark-bench`, both memory oracles, every quality gate and `logit_dump` pass `PROTOCOL_EXPERT_CACHE_SLOTS` through it, and widening the signature would put an `Auto` one keystroke away from a frozen footprint row (AGENTS.md Gotcha 35).

    The RESOLVED count is stored on the runner and reported by both binaries, never the request. Under `auto` the request carries no number at all, so a startup line echoing it would describe nothing -- and the count has to be readable beside any throughput or footprint figure, because it is worth 44.2 tok/s against 51.2 on one install (`docs/DECODE_BUDGET.md`).

    One quieter point: `bytes_per_slot` sums the REAL per-layer strides rather than `layers * max(stride)`. ROADMAP Phase S's candidate has a layer 29 at 1.6x its siblings, where the model-wide maximum over-states a slot's cost by 35% and would talk the policy out of a slot count that fits.

14. **The chunk driver batches the ATTENTION half of a layer and not the routed half, and the asymmetry is a hazard boundary rather than a stopping point.** `prefill_chunk_real_gemma4` (Gemma 4 only; every other family is refused BY NAME, never by falling back to the sequential loop) walks a chunk in micro-batches of `MAX_PREFILL_BATCH`, encodes all M tokens' norms, projections, RoPE, attention and router GEMV into ONE command buffer per layer, then runs the routed half per token. Measured 1.22x on the real install (`docs/BATCHED_PREFILL.md`, "Step 1, measured"); the win is the per-layer blocking wait paid once per micro-batch instead of once per token.

    **What decides which buffers need a per-token row is who WRITES them, not who reads them.** Command buffers on one queue execute in commit order, so a GPU-only intermediate (`normed`, `q`, `o`, `attn_out`, `h1`, `h2`, `moe_acts`, `ffn_normed`) is safe to reuse across a chunk's tokens -- token t+1's dispatch cannot start before token t's finishes, which is the same guarantee the shared-expert-into-routed chain has always relied on. Four buffers are not GPU-only and do need rows: `x` (the residual stream, so it crosses layers), `router_logits_f32` (all M are read back by the HOST after one commit), and `dense_x` / `routed_x` (written in the attention half, read in the routed half, with a commit between). `routing_w` and the routed argument buffer are the two the host WRITES per token, so they are banked `ROUTED_BANKS` deep instead.

    **Pipelining the routed buffers costs a plan that must AVOID the in-flight token's expert slots**, which is what `RoutedSlot::protect` carries and what `ExpertCache::plan`'s `avoiding_slots` was always for. Below `2 * top_k` slots the cache cannot guarantee room for those plus this token's misses, and it ASSERTS rather than degrading, so the driver falls back to retiring before it encodes. That fallback is a throughput choice and must stay a numerics no-op; `a_cache_too_small_to_pipeline_still_reproduces_the_sequential_logits` pins it.

    **Do not reach for the expert-union plan.** An earlier design had one `plan_experts_cached` over the chunk's union replacing M per-token plans, worth "25.2% of prefill cut 3.3x". Measured, prefill's union is 41.5 distinct experts per layer at M=16 against the 24.1 the sequential path already loads at 32 slots, so it saves nothing -- and 41.5 requests against 32 slots trips the assert above. The hit rate before and after the driver landed reads 81.2% against 81.4%, which is the third independent confirmation. See AGENTS.md Gotcha 54.

15. **The context window is sized by a POLICY too, and its two failure modes
    are deliberately different kinds of thing.** `MaxContext::Auto` is the
    default, and `resolve_max_context` answers it with
    `min(trained context, largest window fitting a quarter of the pool)`. Past
    the checkpoint's TRAINED context is a warning: RoPE extrapolates rather
    than failing, some checkpoints carry YaRN scaling meant to exceed it, and
    an install written before the trained context was recorded declares none
    at all, so refusing would be enforced on some installs and not others.
    Past what MEMORY holds is a refusal carrying the whole subtraction,
    because `KvCacheManager::new` allocates every layer up front and its
    failure is a Metal allocation error with no number in it naming the flag.

    **`kv_bytes_for_context` has to mirror that constructor, and the term that
    is easy to get wrong is the ring.** A sliding-window layer's capacity is
    `min(context, sliding_window + MAX_PREFILL_CHUNK_TOKENS)`, so past the cap
    it stops growing -- which means a model's per-token KV cost is decided by
    its FULL layers alone. Gemma 4 is 5 full layers of 30 and costs 20 KiB per
    token; a dense 7B costs 128. Get the ring wrong and the estimate is 6x
    high on Gemma. A linear layer contributes nothing at all (its history
    lives in `GdnStateManager`), and a compressed-attention install gives
    EVERY layer a placeholder rather than only its compressed ones. The
    dense-7B arm is cross-checked against the one measured KV figure in the
    repo: 32 layers at 8,192 comes out to exactly the 1,024 MiB
    `mistral_memory_oracle` records (AGENTS.md Gotcha 40).

    **`committed_bytes` is not the weight file's size**, and on a streamed MoE
    install the difference is most of the answer: Gemma 4's
    `model_weights.bin` is 1.26 GiB while its expert table is 12 GB, of which
    the slot cache pins `slots x sum(expert_stride)` -- about 3.0 GiB at the
    top of `ALLOWED_CACHE_SLOTS`. It adds the WORST case rather than the
    resolved count, because the slot policy resolves inside `open` and this
    has to decide first. Per-layer strides summed, never `layers x
    max(stride)`, for `model-io` Gotcha 2's reason.

    **`physical_memory()` answering 0 means the probe is unavailable, not that
    the machine has no memory.** That is what it returns off macOS, and
    reading it as an empty budget would refuse every explicit window and
    resolve every `Auto` to a context of ZERO -- which admits no prompt at
    all, on no information. An unknown machine imposes no bound, exactly as an
    unknown trained context imposes no ceiling.

16. **The MTP head runs on the trunk's encoders under a different tensor
    prefix, and everything it needs of its own is TWO buffers.** The draft
    step (`families/qwen/mtp.rs`, `docs/MTP_SPECULATIVE.md` step 2) calls
    `attn.rs`'s `encode_full_attention_block` and `dense.rs`'s
    `encode_qwen_layer_dense` with `MTP_PREFIX` in place of `TRUNK_PREFIX`,
    because the published head's single block is a trunk full-attention
    layer's shape field for field. No new kernel, no new dispatch shape.

    **Reusing the trunk's scratch is safe for exactly one reason and it is
    an ORDERING one.** `scratch.x` is re-initialised from the embedding at
    the top of every token, so clobbering it AFTER the trunk's logits have
    been read cannot affect anything -- which is also why
    `mtp_draft_step` must be called between the trunk's readback and the
    next `produce`, and why it reads `h_t` straight out of `scratch.x`
    rather than needing a copy. Move the call outside that window and the
    head silently drafts off the wrong hidden state.

    **Its KV is its own one-layer `KvCacheManager`, built from a CLONED
    `ArchConfig` at `num_layers: 1` and `full_attention_layer_mask:
    vec![1]`.** Widening the trunk's is not available: that one's sizing is
    what every family's frozen oracle peak is asserted against (Gotcha 15).
    The clone reuses the whole constructor and allocates about 4 KiB per
    token on this architecture. Note `reset()` has to rewind it too -- the
    trunk's reset does not reach it, which is Gotcha 4's leak shape one
    cache over.

    **Two `MTP_PREFIX` constants exist on purpose.** `families/qwen`'s is
    `"mtp"` and BUILDS names through `prefixed_layer_tensor`;
    `repack::classify`'s is `"mtp."` and MATCHES them with `starts_with`.
    Sharing one would be wrong at whichever site it was not written for.

    What a synthetic fixture can prove here is bounded, and
    `tests/real_forward_qwen35_mtp.rs` states the bound: untrained weights
    make a drafted token meaningless, so it asserts the step RUNS, that it
    reads the head's tensors and not trunk layer 0's (as a discriminating
    PAIR -- perturbing an `mtp.layers.0.*` tensor must move the draft and
    must not move the trunk), and a FROZEN DIGEST. That digest is not
    decoration: of four mutations checked, two (swapping `fc`'s input
    halves, and pointing the final norm at the wrong tensor) reddened the
    digest ALONE and every reachability case stayed green. Gotcha 51,
    demonstrated rather than cited.
