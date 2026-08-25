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
|   +-- raw_completion_chunked.rs # Chunked generation loop implementation
|   +-- token_sink.rs           # TokenSink abstractions for streaming completion tokens
|   +-- resid_capture.rs        # Per-layer residual stream at the last prompt token (steering)
|   +-- steering.rs             # SteeringPolicy + the GPU state that serves it
|   +-- real_forward.rs         # RealForwardRunner struct, constructor, and dispatch
|   +-- real_forward_open.rs    # RealForwardRunner open_inner implementation
|   +-- real_forward_rollback.rs# RollbackPoint state capture and rewind methods
|   +-- real_forward_traits.rs  # LogitProducer, SpeculativeProducer, ChunkedPrefillRunner impls
|   +-- real_forward_dispatch.rs# Dynamic Metal kernel dispatch helpers
|   +-- real_forward_dispatch_moe.rs # MoE Phase 1 and Phase 2 dispatch helpers
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
|   |   |   +-- prefill.rs      # Gemma 4 chunked prefill encoders
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
|   |   |   +-- mod.rs          # Entry point & DraftPolicies
|   |   |   +-- produce.rs      # Forward pass token decode (produce_real_qwen)
|   |   |   +-- batched.rs      # The M-ROW forward behind the MTP verify (step 4)
|   |   |   +-- batched_scratch.rs # Metal scratch buffer allocations for batched verify
|   |   |   +-- batched_layers.rs # Batched attention, linear, and dense layer encoders
|   |   |   +-- attn.rs         # Gated DeltaNet & gated full attention blocks
|   |   |   +-- dense.rs        # Dense gated FFN (`qwen3_5`, ROADMAP's 1-bit entry)
|   |   |   +-- moe.rs          # Shared + routed MoE pass encoding
|   |   |   +-- moe_batch.rs    # The M-ROW routed half behind the verify (Phase 3)
|   |   |   +-- mtp.rs          # The MTP head's draft step
|   |   |   +-- mtp_dump.rs     # Debug intermediate dump helper for MTP
|   |   |   +-- mtp_state.rs    # MTP state and policy definitions
|   |   |   +-- dflash.rs       # DFlash2 BLOCK drafter execution
|   |   |   +-- dflash_state.rs # DFlash2 state, shape derivation, policy definitions
|   |   |   +-- dflash_draft/   # DFlash2 draft passes and candidate selection
|   |   |   |   +-- mod.rs      # Draft round orchestration, cursor rewind, probe methods
|   |   |   |   +-- context_kv.rs # Context-KV projection pass
|   |   |   |   +-- forward.rs  # One-pass block forward pass
|   |   |   |   +-- layer.rs    # Sublayer attention and MLP encoders for DFlash
|   |   |   |   \-- select.rs   # Host candidate selection & codebook helpers
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
    +-- real_forward_qwen_moe_batched.rs # The BATCHED routed verify vs M sequential produce
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
- **The two sizing policies LIVE IN `crates/model-io` and are re-exported
  here.** `context_policy.rs` (`MaxContext`, `kv_bytes_for_context`,
  `largest_context_within`, `committed_bytes`) and `expert_cache_policy.rs`
  (`ExpertCacheSlots`) moved there when `crates/catalog` needed the same
  arithmetic to answer "would this install fit" BEFORE the install exists.
  Both are pure functions of an `ArchConfig` and a machine size and neither
  touches `gpu`, so the move cost nothing; `runtime::MaxContext` and
  `runtime::ExpertCacheSlots` still resolve, and Gotchas 13 and 15 below are
  still where their behaviour is documented, because this is the crate that
  consumes them. `power.rs` keeps the two OS probes (`physical_memory`,
  `recommended_max_working_set`) -- those are the one place that asks the
  machine, and the policies take the answer as a parameter.
- `error.rs`: `RuntimeError` enum.

## Development & Test Commands

```sh
# Run unit and integration tests for turbospark-runtime
cargo test -p turbospark-runtime
```

## Crate Gotchas

0. **THE SPECULATION POLICY LIVES HERE BECAUSE TWO FRONT ENDS NEED IT.**
   `speculation_policy.rs` (macOS-gated, like `families`) owns three
   decisions in a fixed order: `resolve_drafter` reads an install's resident
   index to say which drafter it carries, `draft_policies` turns that plus the
   request into the `DraftPolicies` to OPEN with, and `resolve_speculation`
   says whether this run may draft. It was `crates/cli`'s private
   `generate/speculation.rs` until 2026-08-21, when `turbospark-server` needed
   the same three and could reach none of them -- that crate has two binaries
   and no lib target.

   Two things about the shape. The enums here (`Speculation`,
   `SpeculativeDrafter`) are runtime-native and `crates/invocation` keeps its
   own, meeting in one mapping function per front end: that crate is pure and
   may not read an install or a machine, and every decision above needs one or
   the other. And `draft_policies`'s `note` arm is load-bearing rather than
   tidy -- a DFlash2-only install under `auto` must open with the MTP policy
   OFF so `resolve_speculation` can refuse with a message naming
   `--speculative-drafter dflash`, instead of the open failing with "carries
   no multi-token-prediction head" and sending someone holding a working
   drafter off to download a different one.

   **THE HEADLESS ARM BESIDE IT IS THE SAME ARGUMENT ON A WIDER INPUT, and it
   was missing until 2026-08-21.** The note only ever fires for a DFlash2
   install; every OTHER headless install still mapped a NAMED block onto
   `MtpDraftPolicy::Fixed` and failed at OPEN. Measured on the real
   `ornith35b`: `--speculative 2` reported "carries no multi-token-prediction
   head ... stream an install that adds the official checkpoint's last shard",
   sending a caller after a 4.4 GB shard that CANNOT help, because the batched
   verify is dense-only and that install routes to 256 experts. `auto` got the
   same install right in the same run, which is what localised it -- `auto`
   reaches `resolve_speculation` and a named block did not. Gotcha 16's
   architectural-blocker-before-the-missing-head ordering was correct all
   along and simply unreachable. `DrafterChoice` carries
   `install_has_mtp_head: Option<bool>` now and the block arm declines to ask
   the open for a head it knows is absent. Three things about it. `Some(false)`
   and never a bare falsy test: `None` is an unreadable index, and "nobody
   looked" must keep failing at open with the engine's own message rather than
   being explained by a guess. `resolve_drafter` reads the index even for an
   EXPLICITLY named drafter, because `--speculative-drafter mtp --speculative
   2` had the same bug by a second route -- passing the drafter through
   untouched is not the same as not looking at the install. And the hard fail
   is unchanged; only the reason improves, and on a DENSE headless install it
   is the same sentence it always was.


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
   - `MFERENCE_RESID_CAPTURE=/path.json`: lifts the residual stream at the OUTPUT of every layer, at the LAST PROMPT TOKEN, into a JSON header plus a raw `.f32` sidecar (`resid_capture.rs`, extracted by `scripts/extract_direction.py`). This is ROADMAP item 9's stated prerequisite, the activation-capture surface a steering direction is derived FROM. **TWO FLOWS since 2026-08-24**: the qwen one (both halves) and `families/llama/` (Mixtral, `qwen3moe`, and the dense Mistral / Llama 2 / 3.x half). The guard reads `steering::family_dispatches_steering`, the SAME predicate the steering refusal reads rather than a second list, because the capture and the edit land on one boundary and a family wired for one and not the other measures where it does not steer. It is guarded the way `MFERENCE_FFN_HIST` is and for its reason: a family whose flow contains no copy would write a file of ZEROS, which extracts as a zero direction, which `steering::inv_norm` then makes inert -- so the whole pipeline would run and steer nothing, with no error anywhere. It adds NO kernel (`encode_dflash_copy_rows` is already a generic strided FP16 row copy and the drafter already lifts `scratch.x` with it at this exact boundary) and no dispatch when unset. A copy cannot change what it copies, so output is byte-identical with it on -- measured on the real `qwen38-27b`, greedy md5 `f4654068...` both ways, not merely argued. **Which pass it keeps is the part to understand**: exactly one per generation, the FIRST with `skip_head` false, which is `produce(prompt[n-1])`. Keying on that transition rather than on "the last pass of the run" is what makes it independent of `--max-new`; the obvious alternative silently captures a GENERATED token's activation at any budget above 1, and a corpus half-captured at the wrong positions yields a plausible wrong direction. Re-armed by `reset()`, so one open can walk a corpus.
   - `MFERENCE_MTP_DRAFT=<depth>`: builds `families/qwen/mtp.rs`'s `MtpState` and lets `RealForwardRunner::mtp_draft_step` run (`docs/MTP_SPECULATIVE.md`, step 2; `qwen3_5` only). Unset, unparsable or 0 allocates NOTHING and encodes nothing, so the off path is identical in bytes and in footprint to the engine that shipped before the module existed -- which is what lets `qwen38_memory_oracle`'s frozen row stand rather than needing a new one. A depth asked for on an install with no head is an ERROR at open naming `mtp.fc.weight`, never a silent no-op: a caller that asked for speculation and quietly got none would measure the non-speculative engine and report it as the speculative one (Gotcha 14's argument, one feature over).
   - `MFERENCE_DFLASH_DRAFT=<block>`: builds `families/qwen/dflash.rs`'s `DflashState`, the SECOND drafter for this family and the first BLOCK drafter in the engine (`docs/DFLASH2.md`). Same off-path guarantee as the MTP knob, and the same refusal on an install without one. It proposes a whole block in ONE pass, so the loop calls `draft_block` and never `draft_step`; its KV holds TARGET-derived rows written by `dflash_context_write` from the trunk's aux capture, and it is `rewind_drafter`'s exception -- a target AHEAD of its cursor is normal, because that cursor advances only at the NEXT round's context write. Its residual stream is held DIVIDED by `DFLASH_RESIDUAL_SCALE`, with the norms reading it taking `DFLASH_RESIDUAL_EPS`, because the drafter's true residual peaks at 113,920 against FP16's 65,504 (AGENTS.md Gotcha 60); `dflash_select` REFUSES a non-finite row rather than proposing token 0 (Gotcha 59).
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
   - `MFERENCE_ROUTED_BATCH=1`: inside the chunk driver, runs each layer's routed half as ONE route-list dispatch pair per union-bounded sub-batch instead of per token (`docs/BATCHED_PREFILL.md` steps 2 and 3, `families/gemma4/moe_batch.rs`). INT4-affine blobs only; a GGUF install is refused by layout rather than looped.
   - `MFERENCE_BATCHED_GEMV=1`: the same driver's RESIDENT GEMVs as M-row GEMMs through `encode_gemm_any` (step 6, the 29.7% row of the prefill dispatch ranking). **It moves the four attention projections always and the shared expert's three only when `MFERENCE_ROUTED_BATCH` is also on** -- the per-token routed pass reads a single-row `h1` at offset 0 and that read is on the DECODE path's signature, so widening it would be a decode change. Norms, RoPE, attention, the residual adds and the router GEMV stay per token. INT4-affine only, refused by name otherwise (which the DEFAULT synthetic fixture triggers: it writes its shared MLP at eight bits where the real install declares four). Output is byte-identical on both arms and that is measured rather than structural -- the batched and single-row INT4 kernels agree bit-for-bit on a fixture built to see reassociation, against a positive control that does not.
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

    **THE RESIDENT GEMVS BATCH TOO, BEHIND A SECOND SEAM** (`docs/BATCHED_PREFILL.md` step 6, `MFERENCE_BATCHED_GEMV`). The four attention projections become M-row GEMMs through `encode_gemm_any`, and so do the shared expert's three when the routed half is batched as well; that second half is also where the host saving is largest, since the per-token shared branch opens and commits its OWN command buffer per token. Everything with no weights to amortize -- norms, per-head norms, RoPE, attention, the residual adds, the router GEMV -- still loops. Two hazards it added, both silent if unguarded: a batched K/V projection can STRADDLE a sliding-window ring's wrap (`k_slot` validates one row, so it would run past the layer's buffer), which `ring_spans` splits; and `batch_q` is sized at the model's WIDEST head, because Gemma 4's five full layers are 512-wide against the sliding window's 256 and no fixture here has `head_dim != full_head_dim` to catch a wrong sizing, so a length check at the dispatch stands in for the test that cannot exist.

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

    **THAT CACHE HAS TO BE PRIMED, AND NOTHING ABOUT IT FAILS IF IT IS
    NOT.** `encode_full_attention_block` derives its attention span from the
    `position` ARGUMENT (`position + 1`) and never from the cursor, so a
    draft taken at decode position `P` off an empty head attends over `P`
    rows nobody ever wrote: no error, finite logits, plausible tokens, and a
    depressed accept length that reads as a verdict about MTP rather than as
    a bug. `mtp_prime_step` is the fix -- the draft step without the
    full-vocab head, run once per prompt token, which is affordable for the
    same reason `produce_prefill` skips the head. Two consequences. The
    head's pair at `position` is `(h_position, token_at_position+1)`, and
    during a PROMPT both are known, so priming is the decode-time call with
    the guesswork removed and its rows land contiguously from 0 with no
    unwritten row (`h_0` exists, so row 0 does). And `mtp_draft_step` now
    REFUSES a step off the cursor in either direction, because a step past
    it reads unwritten rows while a step behind it silently re-drafts
    history, and both otherwise look like a working drafter.

    **`mtp_rewind_to` is the head's half of a speculative rollback and is
    deliberately NOT reached by `rollback`.** The two restore different
    things to different targets: the trunk goes back to where the block
    STARTED and replays the accepted prefix, while the head goes to where
    the accepted prefix ENDED and continues, because it cannot recompute
    those rows -- each needs the `scratch.x` that the trunk's replay has
    already overwritten. Only the caller knows both targets. A round should
    also take `depth + 1` head steps for `depth` proposals: the extra one is
    taken for its KV row alone, since a round where EVERY proposal is
    accepted needs a row the proposal-producing steps do not write, and that
    is the case a good drafter hits most often.

    **THE HEAD'S FIVE WHOLE-VECTOR NORMS ARE CENTERED (`x * (1 + w)`) AND THE
    TRUNK'S ARE PLAIN.** One model, two conventions -- Gotcha 50's rule
    arriving on a second family after `muse_glimmer`, so `mtp.rs` dispatches
    `encode_rms_norm_bf16w_centered` for `pre_fc_norm_embedding`,
    `pre_fc_norm_hidden`, `input_layernorm`, `post_attention_layernorm` and
    `mtp.norm`, and the PLAIN kernel for the trunk's own `model.norm` that
    produces the head's hidden input. Reading them plainly is not a subtle
    error: it put the true next-next token at median rank 248,308 of 248,320
    and gave 0 accepted of 7,168 proposals. Corrected, block 2 speculation
    pays 1.66x. **ALL SEVEN of the head's norms are centered, the per-head
    `q_norm`/`k_norm` included**, and that pair took a second pass because
    `encode_rms_norm_bf16w_perhead` had no centered sibling and the shared
    attention block resolves both by NAME -- the TRUNK's tensors of those
    exact names, at that exact shape, through that exact call, are plain.
    `QkNormConvention::{Plain, Centered}` is the parameter that lets the two
    call sites disagree; an enum rather than a bool so the call site states
    which question it is answering. **Deferring that pair on the strength of
    23/24 top-1 cost 21 to 75 percent of the speedup** and put two claims into
    `docs/MTP_SPECULATIVE.md` that were about the deviation rather than about
    the head. Facts and instruments: `docs/MTP.md`.

    **DETECTION PLACED AFTER AN OPT-IN GATE IS NOT DETECTION, AND A HEAD IS
    NECESSARY WITHOUT BEING SUFFICIENT.** `MtpState::build` used to return on
    `depth == 0` BEFORE looking at the index, so its `mtp.fc.weight` check only
    ever ran to word an error: an install that HAD a drafter decoded
    sequentially and silently unless an operator knew an env var.
    `MtpDraftPolicy` inverts that -- unset is `Auto`, which builds a head iff
    one is present, while an explicit `Fixed(n)` on a headless install stays an
    error because the caller named something this install cannot do. The second
    half is `speculation_blocker`: drafting needs a head, but VERIFYING needs
    `produce_batched`, which is INT4-only (and was dense-only until Phase 3
    landed the routed half), so a sub-4-bit install carrying a head would pass
    a head-presence check and then die at the first verify with generation
    under way. **The blocker still refuses MoE and that is now a POLICY rather
    than a capability**: the batched routed pair runs and is gated, but no
    published MoE conversion of this architecture carries an ingestible head,
    so nothing could drive it. Lifting the MoE arm is a drafter question, not
    a verify one. It reports the ARCHITECTURAL
    blocker before the missing head -- on a MoE install "no head" is true and
    useless, since no checkpoint would help. Two traps in writing such a probe:
    read a tensor the batched path really dispatches rather than the manifest,
    and do NOT probe layer 0 -- this architecture is three linear layers to one
    full, so layer 0 has no `self_attn.*` and probing it reports "not in the
    resident index" on a healthy install.

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

17. **`expect_err` DOES NOT WORK ON `open()`.** Clippy flags `.err().expect()`
    and suggests `expect_err`, which requires the OK type to be `Debug` -- and
    `RealForwardRunner` does not implement it. Every test asserting that an
    install is refused at open hits this, and the suggestion does not compile.
    Use `let Err(err) = RealForwardRunner::open(..) else { panic!(..) };`,
    which satisfies the lint and needs no `Debug`. The 29 existing
    `expect_err` call sites are all on `Result<(), E>`, where it is fine.

18. **Cancellation is a PARAMETER on two new functions, not a field on
    `GenerationConfig`, and the shape is what keeps it free.**
    `run_raw_completion_cancellable` and its chunked sibling take a
    `cancel: &dyn Fn() -> bool`; the two original entry points delegate to
    them passing a predicate that is always false. `GenerationConfig` has no
    `Default` and is built as a struct literal in over twenty places across
    thirteen files, so a field there would touch every one of them to buy
    nothing, while an extra parameter costs the existing call sites exactly
    zero. What licenses the claim that nothing moved is structural rather
    than hopeful: the added branch is `false` on every pre-existing path, so
    those functions execute the statement sequence they always did. Measured
    anyway, because that is the house rule -- greedy output on the real
    Gemma 4 install is md5-identical across pre- and post-change binaries
    (`b2f16611...`), as is the sampled arm (`0c383ac0...`).

    **THE CANCEL POLL SITS BESIDE `hit_stop_string || hit_max`, NOT AT THE
    TOP OF THE LOOP**, and moving it is the mistake to avoid. A cancelled
    run has to take the SAME exit path the other stop reasons take, because
    that path flushes the stop matcher's withheld tail. Break out early
    instead and whatever the matcher was holding back is silently dropped --
    the caller gets a truncated reply rather than a cancelled one, with
    nothing to say so. `cancelling_during_decode_still_flushes_the_withheld_tail`
    is the guard, and the mutation that breaks early reddens it.

    Cancellation is LAST in precedence: a run that would have stopped on its
    own terms this token reports why it really stopped, so a Stop pressed as
    the model finishes does not relabel a complete turn as a truncated one.
    Both halves are pinned (`a_real_stop_reason_wins_over_a_simultaneous_cancel`,
    `max_tokens_wins_over_a_simultaneous_cancel`).

    `StopReason::Cancelled` carries an ORDINARY `RawDecodeResult`: the tail
    is flushed and `kv_position` / `kv_backed_token_ids` describe the cache
    honestly, so a caller may keep the partial turn and continue from it. A
    cancel observed during PREFILL yields zero new tokens and a
    `decode_seconds` of exactly 0.0 rather than an unmeasured value, so
    nothing can plot a rate that was never measured.

    The chunked path's granularity is coarser BY CONSTRUCTION: a chunk is one
    indivisible `prefill_chunk` call and `PrefillChunkCommitState::require_clean`
    makes a mid-chunk bail unreachable, so a cancel lands on a chunk boundary
    -- on a 128-token chunk that is a longer wait than a caller might assume.


19. **The M-row batched forward is the MTP verify's engine, and both of its
    hazards are SILENT ONES that the losslessness gate cannot see.**
    `families/qwen/batched.rs` runs `tokens.len()` positions through the trunk
    in one pass (`docs/MTP_SPECULATIVE.md` step 4; measured 1.44x at block 2).
    Every GEMV becomes a GEMM through `encode_gemm_any`; norms, RoPE,
    attention and the recurrent step loop per token, which is what the
    measured compute split already assumes rather than a first cut. The
    multi-row GDN kernels (`gdn_conv_mix_prefill`, `gdn_delta_step_prefill`,
    and the `rows` argument on both GDN norms) already existed and were
    dispatched by nothing but `gdn_parity.rs`.

    **THE LAST ROW'S RESIDUAL MUST LAND AT ROW 0 OF `scratch.x`.** That is
    where a sequential run of the same tokens leaves it and where
    `mtp_draft_step` reads `h_t`. Get it wrong and the TRUNK is still
    perfectly correct -- the committed stream stays byte-identical to a
    non-speculative run and every losslessness assertion passes -- while the
    HEAD drafts off the first token of the block instead of the last. What
    moves is accept length alone: measured on the real install, 1.84 accepted
    per round fell to 1.10 and rollbacks went from 9 of 90 rounds to 62 of
    123, which reads as a verdict about MTP rather than as a bug in the pass.

    **THE DFLASH2 AUX CAPTURE IN THIS PASS HAS THE SAME SHAPE AND IS NOW
    PINNED.** The hook copies M residual rows into the drafter's capture and
    calls `note_capture(start_position, batch)`; a wrong aux column, row
    stride or base still fills the buffer with plausible residuals, leaves the
    trunk correct, and moves only the DRAFTER's quality.
    `the_batched_capture_writes_what_the_per_token_hook_writes`
    (`tests/real_forward_qwen35_dflash.rs`) drafts off a cache filled two ways
    -- all per-token primes, against a tail written by this hook -- and
    requires the draft logits to agree. It is a TOLERANCE with a
    discriminating arm beside it, not an equality: on the real install the
    batched and sequential kernels do not agree to the bit, and on this
    fixture they happen to agree exactly, so the wrong-rows arm is what gives
    the bound teeth. All three mutations above redden it and NOTHING else in
    the file -- the losslessness case and the frozen digest stay green under
    every one, which is this hazard demonstrated rather than asserted.

    **A ROLLBACK IS NOT FREE HERE AND NO COMPOSITE MODELS IT.** A sequential
    verify stops at the first rejection, so it has absorbed exactly the
    committed tokens and never rewinds. A batched pass cannot stop early, so a
    REJECTED round restores the whole gated-DeltaNet snapshot and replays the
    accepted prefix as a second batched pass. The probability of paying that
    rises with the block (10% at 2, 84% at 8, 98% at 15), which is why
    batching beats sequential at block 2 and LOSES to it at 8 and 15.

    **THE ROUTED HALF RUNS SINCE ROADMAP PHASE 3** and this paragraph used
    to open "dense only". `families/qwen/moe_batch.rs` drives the batched
    routed pair, bit-identical to M sequential `produce` calls on a routed
    trunk (`tests/real_forward_qwen_moe_batched.rs`). Two of its properties
    are family-specific and each is one mutation from a fluent wrong model:
    the GATED SHARED EXPERT seeds phase 2's accumulator rather than being
    added to a finished routed sum (which is why
    `encode_moe_prefill_phase2_fused` takes a residual at all -- FP addition
    is not associative, so seeding is a different function from appending),
    and the tail is ONE RAW RESIDUAL ADD, because this family has no
    sandwich norms and importing one from the file next door is Gotcha 11
    exactly. A MoE layer also costs a MID-LAYER COMMIT the dense path does
    not, the router's top-k being a host decision downstream of a readback.

    Three refusals remain, all by name: INT4 only (enforced in
    `encode_gemm_any` and again on the routed blob's `RoutedBlobLayout`; the
    1-bit and 2-bit checkpoints of this same architecture have no batched
    kernel), a block whose expert UNION outgrows the slot cache, and no KV
    wrap inside a block. A sequential fallback for any of them would be
    numerically identical and would make a "batched" verify measure the
    unbatched engine -- AGENTS.md Gotcha 35 one layer down.

    The union bound is what CAPS the block size and is not a limitation to
    work around: at top-8 a block of M tokens reads up to `8M` experts, so
    M=2 wants 16 and M=4 wants 32, and `ExpertCache::plan_if_possible`
    ASSERTS rather than degrading (AGENTS.md Gotcha 54) -- which is why the
    driver bounds its own sub-batches. Small blocks are what pay on this
    engine, measured independently on both speculative pages.

    **THE STEERING EDIT RUNS HERE TOO SINCE 2026-08-24, AND ONE FUNCTION
    SERVES BOTH PATHS ON PURPOSE.** `produce_batched` had no hook, so steering
    and speculation were refused together at open -- the verify is what
    COMMITS a speculative token, so a steered sequential path beside an
    unsteered batched one emits a run of tokens from two models, and no
    losslessness check can see it (the run and its speculative reference carry
    the same mixture). `steering.rs`'s `encode_steering` (which lived in `families/qwen/produce.rs` until the llama flow became its third call site) takes
    `rows` now and both sites call it, because two dispatch sites would have
    to agree on the mode, the alpha, the direction offset, the row stride AND
    the coefficient block, and a disagreement in any one of them is a fluent
    model that is not the one asked for.

    Three things this cost that are not obvious from the feature.

    **The coefficient buffer was one FP32 slot per LAYER**, which an M-row
    dispatch overruns -- at the last layer, off the end of the buffer. It is
    `num_layers * MAX_STEER_ROWS` now, and that constant IS `gpu::MAX_BATCH_ROWS`
    rather than a second 16 that agrees by luck: the batched INT4 GEMM caps the
    block for its own reason, so the steering refusal is a backstop that cannot
    currently fire, and a future rise in the kernel's cap carries the buffer
    with it. Nothing downstream can see a wrong stride -- each later layer
    overwrites the spill and a short GPU overrun lands in page slack -- so the
    partition is asserted as arithmetic in `steering.rs`'s own unit test.

    **The edit goes BEFORE the drafter's aux capture, in both paths.** The
    drafter predicts what the TARGET emits, and under an edit the target is the
    steered model, so a capture taken ahead of the edit hands it a residual no
    committed token came from. The order was the other way round in the
    per-token path and unobservable, since the two features could not both be
    on. Flipping one path alone reddens
    `the_batched_capture_agrees_with_the_per_token_hook_under_steering` and
    nothing else; the unsteered capture case cannot see it, because with the
    edit off the two orders are the same program.

    **The prediction that a steered target would collapse acceptance against
    an unsteered drafter is REFUTED** (MTP 1.30 -> 1.31 accepted per round,
    DFlash2 1.33 -> 1.29). Neither drafter's weights are steered and both are
    half-steered anyway, through their INPUT: the edit lands on every layer's
    output including the last, so `scratch.x` is already steered when the head
    reads `h_t` out of it. A drafter reading from anywhere upstream of the last
    steered layer would not inherit it that way.

    **NOTHING REACHES IT END TO END YET**, and that is the checkpoints
    rather than the code: `speculation_blocker` still refuses a MoE install,
    because drafting needs a HEAD and no published MoE conversion of this
    architecture carries an ingestible one. Ornith's lives in its BF16
    repo's last shard and is itself MoE, where `MtpState::REQUIRED` names
    the DENSE FFN tensors a `qwen3_5` head has.
