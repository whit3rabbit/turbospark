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
|   +-- config.rs               # Runtime completion configuration
|   +-- pacing.rs               # Decode-rate deadline arithmetic (Phase P2)
|   +-- power.rs                # Power profiles, thermal ladder, RateControl
|   +-- producer.rs             # LogitProducer trait & ScriptedLogitProducer mock
|   +-- raw_completion.rs       # Generation loops (run_raw_completion & run_raw_completion_chunked)
|   +-- raw_completion_chunked.rs # Chunked generation loop implementation
|   +-- token_sink.rs           # TokenSink abstractions for streaming completion tokens
|   +-- speculative.rs          # Multi-token speculative decoding generation loop orchestrator
|   +-- speculation_policy.rs   # Speculative decoding policy and drafter configuration
|   +-- speculation_policy_tests.rs # Unit tests for speculative decoding policy
|   +-- kv_prefix.rs            # What the KV holds, by token id, for cross-turn reuse
|   +-- router_hist.rs          # MoE expert activation routing histogram collector
|   +-- ffn_hist.rs             # Dense FFN neuron activation mass & sparsity collector
|   +-- resid_capture.rs        # Per-layer residual stream at the last prompt token (steering)
|   +-- vision/                 # The qwen3_5 vision tower's streamed forward pass (M-V4)
|   |   +-- mod.rs              # VisionTower, lazy open, the block-loop driver, VisionEmbedding
|   |   +-- shape.rs            # VisionShape: derived dims and the refusals they justify
|   |   +-- weights.rs          # The 9 resident tensors and the 12 per-block roles, FP16-checked
|   |   +-- scratch.rs          # VisionScratch: per-page buffers, sized and dropped per image
|   |   +-- block.rs            # One block: ln1/qkv/rope/attn/proj/res/ln2/fc1/gelu/fc2/res
|   |   \-- stages.rs           # Patch embed, the host-side position blend, the merger
|   +-- moe_prefill_pipeline.rs # Prefill pipeline for chunked MoE passes
|   +-- real_forward.rs         # RealForwardRunner struct, constructor, and dispatch
|   +-- real_forward_api.rs     # Public API methods and capabilities predicates
|   +-- real_forward_open.rs    # RealForwardRunner open_inner implementation
|   +-- real_forward_rollback.rs# RollbackPoint state capture and rewind methods
|   +-- real_forward_traits.rs  # LogitProducer, SpeculativeProducer, ChunkedPrefillRunner impls
|   +-- real_forward_dispatch.rs# Dynamic Metal kernel dispatch helpers
|   +-- real_forward_dispatch_moe.rs # MoE Phase 1 and Phase 2 dispatch helpers
|   +-- real_forward_init.rs    # Open-time arch vetting and expert streamer setup
|   +-- real_forward_layout.rs  # Quantization dtypes and MoE offset calculations
|   +-- real_forward_types.rs   # RealForwardError, PhaseCounters, DecodeScratch
|   +-- real_forward_utils.rs   # Type conversions, resident views, and host top-k
|   +-- real_forward_utils_tests.rs # Unit tests for type conversions and utilities
|   +-- steering.rs             # Directional steering dispatch & runtime layer application
|   +-- steering_tests.rs       # Unit tests for steering dispatch logic
|   +-- turn_stream.rs         # TurnSplitter/TurnEvent: the ONE turn-splitting adapter (docs/STREAMING.md)
|   +-- families/               # Model-family-specific decode implementations
|   |   +-- mod.rs              # Re-exports model family submodules
|   |   +-- gemma4/             # Gemma 4 decode flow
|   |   |   +-- mod.rs          # Gemma 4 entry point & shared expert branch
|   |   |   +-- attn.rs         # Attention block & router GEMV pass
|   |   |   +-- moe.rs          # Routed MoE pass encoding
|   |   |   +-- moe_batch.rs    # Batched routed MoE pass encoding
|   |   |   +-- prefill.rs      # Gemma 4 chunked prefill encoders
|   |   |   \-- state.rs        # RealGemmaState initialization
|   |   +-- gptoss/             # `gpt-oss` decode flow (biases, sinks, YaRN, MXFP4 experts)
|   |   |   +-- mod.rs          # Entry point & layer loop
|   |   |   +-- attn.rs         # GQA + projection biases + YaRN rope + attention sinks
|   |   |   +-- moe.rs          # Routed MXFP4 pass; adds the ROUTER BIAS before the top-k
|   |   |   +-- moe_batch.rs    # Batched routed MXFP4 pass (step 5, 2026-08-27)
|   |   |   +-- prefill.rs      # gpt-oss chunked prefill driver
|   |   |   \-- state.rs        # RealGptOssState: YaRN table, per-layer router bias
|   |   +-- llama/              # `llama` architecture (Mixtral + dense) decode flow
|   |   |   +-- mod.rs          # Entry point & layer loop
|   |   |   +-- attn.rs         # Plain GQA attention block
|   |   |   +-- dense.rs        # Dense gated FFN (Mistral, Llama 2/3.x)
|   |   |   +-- moe.rs          # Routed MoE pass (no shared expert)
|   |   |   +-- prefill.rs      # Dense-only chunked prefill (ChunkedPrefillRunner, 2026-08-26)
|   |   |   \-- state.rs        # RealLlamaState & the dense/MoE split
|   |   +-- museglimmer/        # Dense Muse Glimmer 30B decode flow
|   |   |   +-- mod.rs          # Entry point & layer loop
|   |   |   +-- attn.rs         # Dense GQA + attention output gate
|   |   |   \-- state.rs        # RealMuseState & norm convention configuration
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
|   |   +-- qwen4/              # `qwen4_exp` (Qwen3.8-Flash-Next) decode flow
|   |   |   +-- mod.rs          # Entry point & layer loop
|   |   |   +-- produce.rs      # Per-token forward pass
|   |   |   +-- prefill.rs      # Chunked prefill driver (the SEVENTH, 2026-09-05)
|   |   |   +-- attn.rs         # QSA indexer + block-selected sparse attention, GDN
|   |   |   +-- hc.rs           # Hyper-connections
|   |   |   +-- moe.rs          # Gated MoE: INT8 router GEMV + routed pass
|   |   |   +-- ple.rs          # The PLE n-gram chain (see Gotcha 14's host-write trap)
|   |   |   \-- state.rs        # RealQwen4State: QSA positions, PLE tails, widened rows
|   |   \-- synthetic/          # Synthetic fallback decode flow
|   |       +-- mod.rs          # Synthetic entry point & host MoE FFN
|   |       \-- layer.rs        # Synthetic layer encoder
|   \-- error.rs                # RuntimeError enum definition
\-- tests/
    +-- cancellation.rs         # Mid-generation cancellation integration tests
    +-- chunked_prefill.rs      # Chunked prefill loop unit tests (scripted producer)
    +-- chunked_prefill_refusal.rs # Chunked prefill capability refusal tests
    +-- gguf_install_refused.rs # Unsupported GGUF installs refused at open tests
    +-- golden_tokens.rs        # Golden token sequence reproducibility tests
    +-- mapped_expert_residency.rs # Mapped expert residency tests (Gemma)
    +-- mapped_expert_residency_gptoss.rs # Mapped expert residency tests (gpt-oss)
    +-- mapped_expert_residency_llama.rs # Mapped expert residency tests (Llama)
    +-- mapped_expert_residency_qwen.rs # Mapped expert residency tests (Qwen)
    +-- prefix_reuse_real.rs    # Real install KV prefix reuse tests
    +-- raw_completion.rs       # Raw completion loop integration tests
    +-- real_forward.rs         # RealForwardRunner short-name integration tests
    +-- real_forward_gemma4.rs  # RealForwardRunner Gemma 4 learned-weight tests
    +-- real_forward_gemma4_chunked.rs # The REAL chunk driver, against a non-chunked reference
    +-- real_forward_gemma4_steered.rs # Gemma 4 directional steering integration tests
    +-- real_forward_gptoss.rs  # The gpt-oss flow: perturb each input, require the logits to move
    +-- real_forward_gptoss_chunked.rs # gpt-oss chunked prefill integration tests
    +-- real_forward_gptoss_steered.rs # gpt-oss directional steering integration tests
    +-- real_forward_llama.rs   # RealForwardRunner Mixtral-shaped decode tests
    +-- real_forward_llama_dense.rs # The DENSE half of the same architecture
    +-- real_forward_llama_dense_chunked.rs # Dense Llama chunked prefill tests
    +-- real_forward_llama_moe_chunked.rs # MoE Llama chunked prefill tests
    +-- real_forward_llama_steered.rs # Llama directional steering integration tests
    +-- real_forward_muse.rs    # Real forward tests for Muse Glimmer flow
    +-- real_forward_museglimmer_chunked.rs # Muse Glimmer chunked prefill tests
    +-- real_forward_museglimmer_steered.rs # Muse Glimmer directional steering tests
    +-- real_forward_qwen.rs    # RealForwardRunner Qwen 3.6 decode tests
    +-- real_forward_qwen35.rs  # The DENSE, ONE-BIT half of the same flow
    +-- real_forward_qwen35_batched_onset.rs # Batched verify onset consistency tests
    +-- real_forward_qwen35_chunked.rs # Qwen 3.5 chunked prefill tests
    +-- real_forward_qwen35_dflash.rs # DFlash2 speculative decoding integration tests
    +-- real_forward_qwen35_mtp.rs # MTP speculative decoding integration tests
    +-- real_forward_qwen35_steered_batched.rs # Batched steered forward tests
    +-- real_forward_qwen3moe.rs# The same flow under the Qwen3-MoE family tag
    +-- real_forward_qwen_moe_batched.rs # The BATCHED routed verify vs M sequential produce
    +-- speculative.rs          # Speculative decoding loop integration tests
    +-- vision_inject_synthetic.rs # Synthetic vision injection integration tests
    +-- vision_tower_parity.rs  # CPU vs GPU vision tower parity tests
    +-- vision_tower_synthetic.rs # Vision tower forward pass on synthetic tensors
    \-- fixtures/
        \-- ChatMLTokenizer/    # Toy ChatML tokenizer fixture directory for integration tests
```

## Key Modules

- `producer.rs`: `LogitProducer` trait definition and `ScriptedLogitProducer` mock implementation.
- `raw_completion.rs`: Token generation loops (`run_raw_completion` and `run_raw_completion_chunked`), integrating producer, detokenizer, stop matcher, and selection sampler.
- `real_forward.rs`: `RealForwardRunner` struct definition, options handling, and dispatch orchestration.
- `families/gemma4/`: Gemma 4 decode flow handling verbatim checkpoint weight names (`language_model.model.layers.0...`), per-head norms, learned weights, and MoE routing. Steers on all three call sites its flow needs (`docs/OBLITERATION.md`): sequential decode, the chunked-prefill driver's per-token routed loop, and its batched-routed tail -- Gotcha 21.
- `families/qwen/`: the hybrid linear/full-attention decode flow, serving **two families** -- `QwenGdnMoe` and, since ROADMAP's 1-bit entry, the dense `QwenGdnDense` (Bonsai-27B, and since 2026-08-14 `Qwen/Qwen3.8-27B`, which shares its architecture exactly and differs only in quantization). Gated DeltaNet on mask-2 layers, gated full attention on mask-1, one post-attention norm feeding whatever the FFN half is, no sandwich norms, no softcap. Selected from `ArchConfig.family`, never from tensor naming. **The FFN is the only fork and it is read off `num_experts`** (`dense.rs`: `mlp.gate_proj` / `mlp.up_proj` / `silu_mul` / `mlp.down_proj` through `encode_gemv_any`, no new kernel), exactly as `families/llama/` splits Mixtral from Mistral. What licenses sharing rather than forking is measured, not assumed: every BEHAVIOURAL field of `qwen_gdn_dense_27b()` equals `qwen_gdn_moe_35b_a3b()`'s and every SHAPE field differs (`model-io`'s `qwen_gdn_dense_shares_the_moe_flows_behaviour_and_differs_in_shape`). `sharedExpertGated` is the one that legitimately parts company and `RealQwenState` checks it in BOTH directions, because it is not an independent axis: there is no shared expert to gate on a dense model. Two traps in the shared code. `intermediate_size` means the SHARED EXPERT's width on the MoE half and the DENSE FFN's width on the other, so the dense branch has to name it and never `moe_intermediate_size` (which a dense install sets to 0, encoding nothing). And a dense layer needs no mid-layer commit -- nothing in it is data-dependent on a host readback the way the router's top-k is -- so the whole token stays in one command buffer.
- `families/gptoss/`: the `gpt-oss` decode flow (ROADMAP M5), the FIFTH flow and the only one that is not a variation on an existing graph. Plain GQA like `llama`'s, plus four things that are each inside the layer: a BIAS on all four projections (a separate `bias_add_bf16_fp16` pass, applied BEFORE RoPE -- reversing those two is a different function that still reads fluently, since RoPE is linear and rotating a biased vector differs from biasing a rotated one by a rotation of the bias), YaRN rope through a PRECOMPUTED per-pair frequency table plus a magnitude scale of 1.3465736 (built once at open; it is position-independent), ATTENTION SINKS (one learned logit per QUERY head, not per KV head, added to the softmax denominator behind `FC_ATTN_HAS_SINKS`), and an alternating 128-token window on the EVEN layers. Two differences are NOT where a reader looks for them. The ROUTER bias lives in `moe.rs`, added on the host between the router readback and the top-k, because llama.cpp's `SOFTMAX_WEIGHT` selects on the biased RAW logits -- after the top-k it would select the wrong experts and still produce fluent text, and after the softmax it would be a different distribution over the right ones. The clamped SwiGLU and the per-expert biases are named nowhere in the flow at all: they ride inside the MXFP4 routed pair and arrive through `RoutedBlobLayout`, with `has_bias` derived from `offsets.gate_b != 0`. `RealGptOssState` refuses a tied head, a zero expert count (this architecture string has no dense half, unlike `llama`) and a missing YaRN block; all three are BACKSTOPS behind `arch_validation`, which compares the passed `ArchConfig` against the manifest field by field and fires first, so the tests for them patch `manifest.json` to agree before the flow's own check can run. Steers on its one call site: right after `encode_gpt_oss_layer_moe`'s raw residual add, on the post-mid-layer-commit "routed cb" pass the router's host-side top-k already forces (`docs/OBLITERATION.md`).
- `families/llama/`: the plain-GQA-plus-MoE decode flow (ROADMAP Phase M2), which serves **two families**, `Llama` (Mixtral) and `Qwen3Moe` (Qwen3-30B-A3B). It is defined by its ABSENCES: plain GQA attention with no per-head norms and no output gate, a raw residual add with no sandwich norms, one post-attention norm feeding router and routed experts, no shared expert, no logit softcap, full-head NeoX RoPE at one base. **BOTH HALVES OF THE ARCHITECTURE STRING RUN** since ROADMAP M4: one `general.architecture = "llama"` covers dense Llama 2/3.x and Mistral as well as the Mixtral MoEs, and `RealLlamaState` tells them apart by `num_experts` (`dense`), never by tensor naming. A dense layer swaps the router and routed experts for one gated FFN (`dense.rs`: `mlp.gate_proj` / `mlp.up_proj` / `silu_mul` / `mlp.down_proj`, all through `encode_gemv_any`, no new kernel) and is IDENTICAL above and below it -- embedding, both norms, attention, the raw residual and the head are the same code. Two non-obvious points: its width is `intermediate_size` and NOT `moe_intermediate_size` (Mixtral copies one `feed_forward_length` into both, so on the MoE half they are interchangeable and a dense checkpoint sets only the first), and a dense layer needs no mid-layer commit, because nothing in it is data-dependent on a host readback the way the router's top-k is. Phase 2's residual input is `scratch.zero_hidden` rather than a shared-expert output, so the routed sum is added to the stream exactly once. **`Qwen3Moe` differs in exactly two places, both carried by `RealLlamaState` and both keyed on `ArchConfig.family` rather than sniffed from tensor names**: it norms q and k PER HEAD before RoPE (`qk_norm`), and its RMS epsilon is 1e-6 against `llama`'s 1e-5 (`rms_eps`, which is not an `ArchConfig` field). A fourth copy of the flow with two lines changed would be a likelier source of a divergence bug than the shared one is. **The DENSE half also has a chunked-prefill driver** (`prefill.rs`, since 2026-08-26; Gotcha 14), the second `ChunkedPrefillRunner` implementation after Gemma 4's and structurally simpler, since a dense layer's no-mid-layer-commit property (this bullet's own "a dense layer needs no mid-layer commit" line) means a whole micro-batch runs every layer in ONE command buffer rather than one per layer.
- `families/museglimmer/`: the `muse_glimmer` decode flow, the SIXTH flow and the seventh family (`mlx-community/Muse-Glimmer-30B-4bit`). Dense 52-layer GQA (32 q over 2 kv, head_dim 128), a three-sliding/one-full window at 2048, sandwich norms and a logit softcap. Its own flow on the `gpt-oss` precedent: TEN differences, every one inside the layer. Four are worth naming because no other flow has them. **Its four per-layer norms are CENTERED (`x * (1 + w)`) and its FINAL norm is PLAIN (`x * w`)** -- two conventions in one model, dispatched per tensor through `rmsnorm_bf16w_centered` and `rmsnorm_bf16w` (AGENTS.md Gotcha 50). **It carries TWO RMS epsilons**, 1e-5 on the input/pre-FFN/q-k/embedding norms and 1e-8 on the two POST norms, where every other flow carries one. **Its full-attention layers are NoPE** -- `layer_rope_theta` is literally 0 there in the checkpoint, carried as `full_rope_theta: 0.0`, and `RealMuseState` refuses an install that says otherwise, because a rotated NoPE layer is fluent and wrong. And **its attention output gate is its own tensor** (`self_attn.gate_proj`, fed from the layer's normed input), not Qwen's packing into `q_proj` -- which is why `attn_output_gate` is FALSE on this family despite it having a gate. Two published scalars (`qk_scale_factor` 3.87 on Q after its no-scale per-head norm, `output_multiplier` 26^-0.5 on the logits before the softcap) and both epsilons are family CONSTANTS in `state.rs` rather than `ArchConfig` fields, following the `rms_eps` precedent -- none is a binary fraction and `arch_validation` compares manifest floats with `!=` (Gotcha 24); `crates/repack/tests/museglimmer_config.rs` parses the real config and asserts all four offline. Steers on its one call site: the FFN-half sandwich tail's residual add, the layer's true output -- no mid-layer commit, since this family has no router at all (`docs/OBLITERATION.md`).
- `families/synthetic/`: Short-name synthetic execution flow (`layer0.q_proj`).
- `config.rs`: Runtime generation configuration and runner settings.
- `power.rs`: ROADMAP Phase P2's policy: `PowerProfile`, `ThermalLevel`, the `thermal_cap` ladder, `RateControl`, and the cfg-paired OS probes that call `crates/gpu`'s wrappers on macOS and return constants elsewhere. **It also carries `MemoryPressure` and `memory_cap`, and `stepped_cap` takes BOTH ladders and applies the MINIMUM** -- not a precedence, because the two signals are independent (a machine can be cool and short of memory, or hot and comfortable) and each states a ceiling true on its own terms. `thermal_cap` stays a separate function so the five cases every published power row was measured under still assert exactly what they did. The memory probe reads the KERNEL'S VERDICT (`kern.memorystatus_vm_pressure_level`) and never a free-page count, which is what keeps it from contradicting `physical_memory`'s stated reason for declining `host_statistics64`: that argument is about a stable BUDGET, and a watcher exists to see the thing that moves. `MemoryPressure::from_raw` reads every unrecognized value -- including the 0 an unavailable sysctl returns -- as `Normal`, the OPPOSITE of `ThermalLevel::from_raw`'s clamp, because an unknown thermal level means hotter still while an unknown memory level means the kernel did not answer. Both probes ride the profile's existing stepping switch and the loop's existing 16-token poll, so `performance` (the default, and what every benchmark was measured under) polls nothing. See `docs/LOAD_GUARD.md`.
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
- `vision/`: the `qwen3_5` vision tower's streamed forward pass (ROADMAP M-V4, `docs/VISION.md`). Opens a 2-slot `PreadExpertStreamer` over the 27 blocks M-V3 wrote into `packed_vision/` and runs patch-embed, the block loop, and the merger, returning the `[merged_tokens, 5120]` FP16 rows M-V5 will inject. **NO NEW I/O CODE**, which was M-V3's design bet: `packed_vision/` reuses the `PackedExpertsLayout` schema verbatim and `StreamLayout` interprets neither "layer" nor "expert", so a tower is one layer of `depth` fixed-stride blobs and the existing streamer serves it unchanged. Reached through `RealForwardRunner::encode_image`, which OPENS THE TOWER LAZILY -- `arch.vision` is read by nothing else here, so an eager open would charge every text-only session on a vision install 58 MiB of pinned slots for a component it never touches, and `self.vision.is_none()` therefore means "no image yet" rather than "no tower" (that question is `arch.vision.is_active()`). It is also the ONLY reader of `vision.` resident tensors, which is what makes `readable_resident_dtype`'s name-scoped FP16 exception safe (Gotcha 24). **Its own mapped-residency arm since 2026-08-30** reads a SEPARATE seam, `TURBOSPARK_VISION_RESIDENCY=mapped`, never the routed `TURBOSPARK_EXPERT_RESIDENCY` -- see Gotcha 31 and `docs/EXPERT_RESIDENCY.md`.
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
   was missing until 2026-08-21 on the MTP path and until 2026-08-22 on the
   DFlash2 one.** The note only ever fires for a DFlash2
   install; every OTHER headless install still mapped a NAMED block onto
   `MtpDraftPolicy::Fixed` and failed at OPEN. Measured on the real
   `ornith35b`: `--speculative 2` reported "carries no multi-token-prediction
   head ... stream an install that adds the official checkpoint's last shard",
   sending a caller after a 4.4 GB shard that CANNOT help: that install routes
   to 256 experts, and no published MoE conversion of this architecture ships a
   drafter this port can ingest, so no shard helps. `auto` got the
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
   is unchanged; only the reason improves.

   **THE FIX WAS HALF-APPLIED FOR A DAY, and the missing half is the lesson.**
   The 2026-08-21 arm keyed on `install_has_mtp_head` and `DrafterChoice`
   carried nothing else, so `--speculative-drafter dflash --speculative 2`
   reproduced the identical failure through `DflashDraftPolicy::Fixed`: on the
   same `ornith35b` it reported "this install carries none (dflash.fc.weight
   is not in the resident index); stream it beside the trunk", which on a MoE
   checkpoint no artifact satisfies, since the published DFlash2 drafter
   targets the DENSE half. `install_has_dflash: Option<bool>` and a second
   guard closed it on 2026-08-22. Two drafters means two of everything on this
   path; grep for the sibling before calling one of these fixes done.

   **ROUTING A NAMED BLOCK THROUGH `Off` MOVES WHICH STRING THE CALLER SEES,
   and the two were not equal.** The reason now comes from the BLOCKER rather
   than from `Dflash/MtpState::build`, and both `build` errors named the
   artifact that fixes the problem while both blocker arms named only the
   obstacle. So the 2026-08-21 fix silently cost a dense headless install its
   "stream an install that adds the official checkpoint's last shard
   (docs/MTP.md)". Both arms carry their pointer now (2026-08-22), and the
   MoE arms return FIRST so neither pointer is ever offered where no artifact
   satisfies it.

   **NOTHING IN `speculation_policy_tests.rs` CAN GUARD THAT, and its own
   header says why**: it feeds fixture strings into `resolve_speculation` and
   never calls either blocker, so it pins the ROUTING and not the TEXT.
   Deleting the pointer from the real arm leaves all 37 of its cases green --
   measured, not inferred. The guard that sees it is
   `a_dense_int4_install_without_a_head_is_told_which_artifact_would_fix_it`
   (`tests/real_forward_qwen35_mtp.rs`), and it needs a fixture built at 4
   BITS: this file's `BITS = 1` is stopped by the INT4 arm and can never reach
   the no-head one. That is the only fixture in the repo that reaches
   `speculation_blocker`'s last arm at all, which is why the arm's text could
   rot unobserved.


1. **PRODUCE WRITES LOGITS, NEVER PROBABILITIES**: `LogitProducer::produce` must return raw, unnormalized logits. `selection::select` performs softmaxing internally. Returning probabilities destroys sampling temperature reweighting (`softmax(softmax(z))`).
2. **`produce_prefill` may skip the output head, `produce` never may.** The prefill loop in `raw_completion.rs` calls `produce_prefill` for every prompt token but the last, because only the last one's logits are read. `RealForwardRunner` implements that by skipping the final norm, full-vocab GEMV, softcap, and host readback. Any producer overriding it must still advance every other per-token side effect (KV cache, position, command buffer commit AND wait) exactly as `produce` does: the buffer wait is what stops the next token overwriting scratch the GPU is still reading. Unrelated to `ChunkedPrefillRunner::prefill_chunk`, which does produce usable logits.
3. **Flow selection keys on `ArchConfig.family`, not on tensor naming.** `Llama` and `Qwen3Moe` share one flow and are told apart INSIDE it by the same field. Gemma 4 and Qwen 3.6 both carry `language_model.model.embed_tokens.weight`, so the naming probe can only distinguish a real Gemma install from a synthetic short-name one. Within `Gemma4` the probe still applies; `QwenGdnMoe` always builds `RealQwenState`; `DeepseekV4Flash` is refused at open.
4. **`reset()` must rewind the GDN state, not just the KV cache.** A linear-attention layer keeps its whole history in `GdnStateManager`'s delta-rule state and conv tail; the KV cache holds nothing for it. Resetting one and not the other leaks the previous generation's context into every mask-2 layer, invisibly (output stays finite and deterministic).
5. **Borrow Checker Rule in `families/gemma4/`**: Per-token forward functions interleave `let real = self.real.as_ref()` bindings with `&mut self` methods. Making a `&mut self` call invalidates existing `real` references under E0502; re-bind `real` immediately after any `&mut self` call.
6. **Phase Profiling Divisor**: `TURBOSPARK_PHASES=1` averages GPU phase timings over ALL forward passes (prefill tokens + decode tokens). To measure per-token decode cost at long contexts, run two tests with different `--max-new` lengths and calculate the delta.
7. **Execution Pipeline Flags**:
   - `TURBOSPARK_PHASES=1`: Prints GPU wait, router readback, expert `pread`, and routed bind timing breakdowns.
   - `TURBOSPARK_DISPATCH_PROFILE=1`: ranks the individual dispatches INSIDE each command buffer
     (`crates/gpu/src/dispatch_profile.rs`), which a per-buffer number cannot tell you. Apple GPUs
     sample counters only at encoder boundaries, so this mode encodes one compute encoder per dispatch
     and waits on every buffer to resolve timestamps: read its module doc before quoting an absolute
     number. A debugging aid, never a throughput measurement.
   - `TURBOSPARK_SHARED_CB=0`: Toggles overlapping the shared expert command buffer with host expert `pread`.
   - `TURBOSPARK_ROUTED_PIPELINE=0`: Toggles one-layer-pipelined routed command buffer execution (Gemma 4's sequential decode). Since 2026-08-27 it also governs the `gpt-oss` chunked driver's PER-TOKEN routed loop: off means banks = 1 AND an empty `RoutedSlot::protect` set, the two moving together because retire-before-plan is what makes the empty set sound. Wired as the A/B seam that separated the batched arm's 28% miss drop -- and the answer EXONERATED the protect set (9,024 misses with it, 9,400 without, stdout md5-identical): the drop is real union dedup, this engine's one family-scoped exception to "the union saves nothing" (`docs/BATCHED_PREFILL.md`, step 5's miss-drop paragraphs).
   - `TURBOSPARK_ROUTER_HIST=/path.json`: Dumps a per-layer expert-selection histogram on runner drop (`router_hist.rs`, analyzed by `scripts/router_hist.py`). Diagnostic only; the 2026-08-08 measurement it exists for (domain-concentrated routing) came back negative, see `docs/EXPERT_ROUTING.md`.
   - `TURBOSPARK_ROUTER_TRACE=1`: adds the top-k ids IN PASS ORDER to that same file (`scripts/router_window.py` analyzes it). The counts cannot answer ROADMAP's speculative-decoding question, because a batched verify of M tokens reads the UNION of their routes and a histogram has already discarded which pass each selection came from.
   - `TURBOSPARK_PILOT_PROBE=1`: adds the ONE-LAYER-AHEAD prediction to that same file -- layer L+1's router run early, on layer L's post-attention residual (colibri's PILOT). Gemma only; one extra router GEMV per MoE layer, and nothing is written or dispatched when it is unset. Analyzed by `scripts/pilot_ceiling.py`, which also simulates the expert cache to price the guess in BYTES. The 2026-08-29 measurement it exists for came back negative (the predictor works at 70.6% recall and still reads 1.03x to 1.85x the expert bytes); `docs/EXPERT_ROUTING.md` has the sweep and the reversal condition.
   - `TURBOSPARK_PILOT_PROBE=self`: aims that probe at the layer it is already running in, so it reproduces the production router and recall MUST read 100%. It exists because the first wiring read 7.7% against a 6.25% random baseline -- an off-by-one in which layer the guess was filed against, which is indistinguishable from a real negative result (AGENTS.md Gotcha 57). Run it before believing any number this probe reports.
   - `TURBOSPARK_FFN_HIST=/path.json`: dense-FFN activation census on runner drop (`ffn_hist.rs`, analyzed by `scripts/ffn_sparsity.py`; museGlimmer only, the one flow that feeds its capture). Redirects `silu_mul` into a per-layer capture buffer, so it changes no math and no output bytes; costs ~30% of decode throughput while on. The 2026-08-16 measurement it exists for (a PowerInfer-style neuron cache) came back negative, see `docs/ACTIVATION_SPARSITY.md`.
   - `TURBOSPARK_RESID_CAPTURE=/path.json`: lifts the residual stream at the OUTPUT of every layer, at the LAST PROMPT TOKEN, into a JSON header plus a raw `.f32` sidecar (`resid_capture.rs`, extracted by `scripts/extract_direction.py`). This is ROADMAP item 9's stated prerequisite, the activation-capture surface a steering direction is derived FROM. **FIVE FLOWS since 2026-08-25**: the qwen one (both halves), `families/llama/` (Mixtral, `qwen3moe`, and the dense Mistral / Llama 2 / 3.x half), `families/gemma4/` (sequential decode, the chunked-prefill driver's per-token routed loop, and its batched-routed tail -- Gotcha 21), `families/gptoss/` (its one call site, the routed-MoE tail's raw residual add), and `families/museglimmer/` (its one call site, the FFN-half sandwich tail's residual add). The guard reads `steering::family_dispatches_steering`, the SAME predicate the steering refusal reads rather than a second list, because the capture and the edit land on one boundary and a family wired for one and not the other measures where it does not steer -- which is exactly why the last two picked up capture automatically the moment the predicate flipped to `true` for them, with no second list to update. It is guarded the way `TURBOSPARK_FFN_HIST` is and for its reason: a family whose flow contains no copy would write a file of ZEROS, which extracts as a zero direction, which `steering::inv_norm` then makes inert -- so the whole pipeline would run and steer nothing, with no error anywhere. It adds NO kernel (`encode_dflash_copy_rows` is already a generic strided FP16 row copy and the drafter already lifts `scratch.x` with it at this exact boundary) and no dispatch when unset. A copy cannot change what it copies, so output is byte-identical with it on -- measured on the real `qwen38-27b`, greedy md5 `f4654068...` both ways, not merely argued. **Which pass it keeps is the part to understand**: exactly one per generation, the FIRST with `skip_head` false, which is `produce(prompt[n-1])`. Keying on that transition rather than on "the last pass of the run" is what makes it independent of `--max-new`; the obvious alternative silently captures a GENERATED token's activation at any budget above 1, and a corpus half-captured at the wrong positions yields a plausible wrong direction. Re-armed by `reset()`, so one open can walk a corpus.
   - `TURBOSPARK_MTP_DRAFT=<depth>`: builds `families/qwen/mtp.rs`'s `MtpState` and lets `RealForwardRunner::mtp_draft_step` run (`docs/MTP_SPECULATIVE.md`, step 2; `qwen3_5` only). Unset, unparsable or 0 allocates NOTHING and encodes nothing, so the off path is identical in bytes and in footprint to the engine that shipped before the module existed -- which is what lets `qwen38_memory_oracle`'s frozen row stand rather than needing a new one. A depth asked for on an install with no head is an ERROR at open naming `mtp.fc.weight`, never a silent no-op: a caller that asked for speculation and quietly got none would measure the non-speculative engine and report it as the speculative one (Gotcha 14's argument, one feature over).
   - `TURBOSPARK_DFLASH_DRAFT=<block>`: builds `families/qwen/dflash.rs`'s `DflashState`, the SECOND drafter for this family and the first BLOCK drafter in the engine (`docs/DFLASH2.md`). Same off-path guarantee as the MTP knob, and the same refusal on an install without one. It proposes a whole block in ONE pass, so the loop calls `draft_block` and never `draft_step`; its KV holds TARGET-derived rows written by `dflash_context_write` from the trunk's aux capture, and it is `rewind_drafter`'s exception -- a target AHEAD of its cursor is normal, because that cursor advances only at the NEXT round's context write. Its residual stream is held DIVIDED by `DFLASH_RESIDUAL_SCALE`, with the norms reading it taking `DFLASH_RESIDUAL_EPS`, because the drafter's true residual peaks at 113,920 against FP16's 65,504 (AGENTS.md Gotcha 60); `dflash_select` REFUSES a non-finite row rather than proposing token 0 (Gotcha 59).
   - `TURBOSPARK_PREFILL_CHUNK=<tokens>`: routes prefill through `run_raw_completion_chunked` and `RealForwardRunner`'s chunk driver (Gotcha 14). An A/B seam like the two above it, not a feature flag: both arms must produce identical tokens. Unset, unparsable or 0 is the sequential path. **`--prefill-chunk` IS wired now (2026-08-26), and this env var still wins over it when set.** `crates/cli`'s `resolve_chunk_tokens` and `crates/server`'s `RealChatModel::run_completion` both check the env var first, then fall through to the flag's resolved value (`invocation::PrefillChunk::resolved`: `Fixed(n) -> n`, `Auto -> DEFAULT_CHUNK_SIZE`) ONLY when `RealForwardRunner::supports_chunked_prefill()` says the open install's family can serve it, and to the sequential path with no error otherwise -- the flag carries a default on every invocation whether or not the caller typed it, so an unsupported family must not become an error for a caller who asked for nothing. That predicate is the SAME one `ChunkedPrefillRunner::prefill_chunk`'s own hard refusal uses (`real_forward_api.rs`), so a caller deciding whether to route here and the driver's own refusal can never disagree. Two families today: Gemma 4 and the dense half of `llama` (Gotcha 14).
   - `TURBOSPARK_ROUTED_BATCH=1`: inside the chunk driver, runs each layer's routed half as ONE route-list dispatch pair per union-bounded sub-batch instead of per token (`docs/BATCHED_PREFILL.md` steps 2 and 3, `families/gemma4/moe_batch.rs`). **TWO families since 2026-08-27**: Gemma 4 on INT4-affine blobs, and `gpt-oss` on MXFP4 ones (step 5's first arm, `families/gptoss/moe_batch.rs`). Each refuses the OTHER's layout by name rather than looping, so neither can silently measure the per-token engine. `qwen3moe`'s Q4_K/Q6_K blobs are still refused by layout: that arm was scoped by measurement and deliberately not built (Gotcha 22).
   - `TURBOSPARK_BATCHED_GEMV=1`: the same driver's RESIDENT GEMVs as M-row GEMMs through `encode_gemm_any` (step 6, the 29.7% row of the prefill dispatch ranking). **It moves the four attention projections always and the shared expert's three only when `TURBOSPARK_ROUTED_BATCH` is also on** -- the per-token routed pass reads a single-row `h1` at offset 0 and that read is on the DECODE path's signature, so widening it would be a decode change. Norms, RoPE, attention, the residual adds and the router GEMV stay per token. INT4-affine only, refused by name otherwise (which the DEFAULT synthetic fixture triggers: it writes its shared MLP at eight bits where the real install declares four). Output is byte-identical on both arms and that is measured rather than structural -- the batched and single-row INT4 kernels agree bit-for-bit on a fixture built to see reassociation, against a positive control that does not. **Every OTHER chunked driver refuses this seam by name** (since 2026-08-28; it silently ignored it before, Gotcha 22's exact silent-ignore class): unlike `TURBOSPARK_ROUTED_BATCH`, which a dense driver legitimately ignores, this seam names the driver's RESIDENT GEMVs and every family has those -- so `gpt-oss`, both halves of `llama` and `muse_glimmer` each carry the refusal, pinned per family by `the_batched_gemv_seam_is_refused_by_name_on_this_family`.

  **THE DENSE QWEN DRIVER IS THE SECOND FAMILY WIRED TO IT (2026-08-29), AND ITS BYTE-IDENTITY CLAUSE ABOVE DOES NOT TRANSFER.** That clause is about two KERNELS agreeing; this family's M-row PASS differs from its per-token one by 6.2e-8 to 1.5e-5 nats with the argmax agreeing on every row, which is a batched-vs-cached SHAPE FLOOR every engine has (7.4e-6 measured on MLX for this same architecture) rather than a defect -- commit `e8deb6c`, which dissolved an open question three documents were carrying. So the DEFAULT arm is what must stay byte-identical and the batched arm's gate is `qwen38_quality_gate`. The wiring cost no dispatch code: `families/qwen/batched_layers.rs` already owned all three encoders for the MTP/DFlash2 verify, at the same `t * hidden * 2` row convention, and `MAX_PREFILL_BATCH` IS `gpu::MAX_BATCH_ROWS` (both 16) so a full micro-batch is one dispatch. What it added is three guards Gemma 4's arm does not need: the width refused UP FRONT (the 1-bit and 2-bit checkpoints of this architecture reach no M-row GEMM), `produce_batched`'s KV-wrap check copied in (no sliding window here, but "linear" still wraps at `max_context`), and an M-row scratch allocated only when the seam is ON -- stricter than `ensure_batched`, which also serves `TURBOSPARK_ROUTED_BATCH` and so allocates unconditionally. That last one is asserted with `gpu_buffer_allocations()` rather than argued, because `BatchedScratch` is ~10 MiB (mostly a `batch * vocab` logits plane this driver never reads, the head staying a single-row GEMV) against an oracle ceiling with 87 MiB of headroom that could not see it.
8. **A layer's routed slots are dispatched in the ROUTER'S RANKING, and that is a correctness constraint, not a style choice.** Phase 2 reduces `blob[slot] * routing_w[slot]` in slot-index order and FP addition is not associative, so the slot order is the summation order. The Gemma flow used to order slots misses-first so the resident hits' phase-1 GEMV could ride its own command buffer (hit-CB seam, now removed); because the hit/miss split follows CACHE STATE rather than the prompt, the same prompt could decode to different text across warm runs in one process. Measured 2026-08-08: 4 distinct outputs in 6 runs on a Q8_0 GGUF install at 16 slots, 2 in 6 on the MLX install at 32. Both families are now byte-identical across 8/16/32 slots and cold vs warm. Before adding a decode-path optimization that reorders slots, ask what its ordering is a function of.
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

14. **The chunk driver batches the ATTENTION half of a layer and not the routed half, and the asymmetry is a hazard boundary rather than a stopping point.** `prefill_chunk_real_gemma4` walks a chunk in micro-batches of `MAX_PREFILL_BATCH`, encodes all M tokens' norms, projections, RoPE, attention and router GEMV into ONE command buffer per layer, then runs the routed half per token. Measured 1.22x on the real install (`docs/BATCHED_PREFILL.md`, "Step 1, measured"); the win is the per-layer blocking wait paid once per micro-batch instead of once per token. **Every unsupported family is still refused BY NAME at `ChunkedPrefillRunner::prefill_chunk`, never by falling back to the sequential loop** -- that trait method has to stay a hard refusal so a caller who explicitly asks for the chunked driver on an install it can't serve finds out, rather than silently measuring the sequential engine and reporting it as the chunked one. What CAN fall back silently is a caller deciding whether to route here at all (Gotcha 7's `--prefill-chunk` default), which is a different question answered by `supports_chunked_prefill()` before the call is ever made.

    **A SECOND FLOW LANDED 2026-08-26, and it is structurally simpler than the first.** `prefill_chunk_real_llama_dense` (`families/llama/prefill.rs`) serves the DENSE half of `llama` (Mistral, Llama 2/3.x; `num_experts == 0`, checked via `RealLlamaState::dense`) -- MoE `llama` stays refused, same as every other unsupported family. A dense layer needs no router readback at all, so where Gemma 4 commits one command buffer PER LAYER to pipeline the routed half's host round trip, the dense driver runs an ENTIRE micro-batch, every layer, in ONE command buffer, committed once at the end. No `ROUTED_BANKS`, no expert-slot protection, no ring-wrap hazard (`RealLlamaState::build` already refuses any non-full-attention layer, so this family has no SWA ring to straddle). The row convention is the same as Gemma 4's, extended to one more function: `encode_llama_layer_dense` (`families/llama/dense.rs`) gained an `x_off: u64` parameter threaded into its residual add, following Gotcha 21's exact precedent, with every pre-existing call site passing `0` explicitly. `attn::encode_attention_block` needed NO change, since it never touches `scratch.x` directly (the caller reads/writes it around the call). Verified byte-identical against sequential on both the synthetic fixture (`real_forward_llama_dense_chunked.rs`, chunk-span sweep `[1, 2, 3, 4, 7, 11]`, plus an explicit "MoE is still refused" case) and on the real `~/.turbospark/models/mistral7b.gturbo` install (greedy and sampled stdout md5-identical against a pre-change binary).

    **A THIRD FLOW LANDED 2026-08-27, `muse_glimmer`, and it is the same shape as the second rather than a new one.** `prefill_chunk_real_muse` (`families/museglimmer/prefill.rs`) serves this family, which has no router at all (`families/museglimmer/mod.rs`'s own doc), so it runs an entire micro-batch in ONE command buffer exactly as the dense `llama` driver does -- no `ROUTED_BANKS`, no expert-slot protection. **No ring-wrap hazard either, and for a different reason than dense `llama`'s**: this family DOES have a sliding-window ring (three-sliding/one-full at 2048), but this driver keeps attention per-token and unbatched (no `TURBOSPARK_BATCHED_GEMV`-style widening of the projections), so the ring is addressed by `position` exactly as the sequential decode path already does it -- the hazard only exists for a batched K/V projection, which this driver never encodes. `attn::encode_attention_block` needed NO change, for the identical reason dense `llama`'s did not: it reads `scratch.normed` and writes `scratch.o`, never touching `scratch.x` directly. `mlp::encode_mlp_block` DOES touch `scratch.x` (its pre-FFN norm reads it, its residual add writes it), so it gained the same `x_off: u64` parameter `encode_llama_layer_dense` did, with the one pre-existing call site (the sequential flow) passing `0` explicitly. Verified byte-identical against sequential on both the synthetic fixture (`real_forward_museglimmer_chunked.rs`, chunk-span sweep `[1, 2, 3, 4, 7, 11]`, which crosses the 8-token sliding window the fixture uses) and on the real `~/models/museglimmer-30b.gturbo` install (greedy and sampled stdout md5-identical against a pre-change binary, at 40 new tokens each -- shortened from the usual 400 to fit under memory pressure from an unrelated process at verification time, still enough to prove multi-token decode continuity).

    **THE FIFTH FLOW LANDED 2026-08-27, `gpt-oss`, and it is the first to reuse the FULL pipelining pattern (`RoutedSlot`, banks, protect) a second time after Gemma 4.** `prefill_chunk_real_gpt_oss` (`families/gptoss/prefill.rs`) commits one command buffer PER LAYER for the attention-and-router half, exactly like Gemma 4's driver, because this family's router top-k also needs a host readback before the routed half can bind. It differs from Gemma 4's routed half in the same two ways the sequential decode flow already does: no shared expert (phase 2's residual seed is `scratch.zero_hidden`, matching the MoE half of `llama`) and the router BIAS added on the host between the readback and the top-k, inside the SAME `moe::encode_gpt_oss_layer_moe` call the per-token loop already makes, so nothing extra had to be threaded into the chunked driver for it. **No ring-wrap hazard, despite this family having a REAL alternating sliding window (unlike `llama`)**: attention stays per-token and unbatched in this driver, so there is no batched K/V projection to straddle the ring's wrap, and `attn::encode_attention_block` runs unchanged, addressing the ring by `position` exactly as the sequential path does. `RealGptOssState::router_logits_f32` and `moe_x` both widened from single-row to `MAX_PREFILL_BATCH` rows, matching `RealLlamaState`'s fields of the same names. Verified byte-identical against sequential on both the synthetic fixture (`real_forward_gptoss_chunked.rs`, chunk-span sweep including a span that lands exactly on the fixture's window and one that crosses it, plus a cache-too-small-to-pipeline case) and on the real `~/.turbospark/models/gptoss-20b.gturbo` install (greedy and sampled stdout md5-identical against a pre-change binary; prefill on a 75-token prompt dropped from 7.56s to 3.79s now that chunking engages).

    **THE FOURTH FLOW, THE MoE HALF OF `families/llama/`, LANDED THE SAME DAY AND IS THE SIMPLER OF THE TWO** (no shared expert, no router bias, no ring at all -- this architecture has no sliding-window layers, full stop). `prefill_chunk_real_llama_moe` (`families/llama/moe_prefill.rs`) is the same per-layer-`cb1`-plus-per-token-routed-loop shape, and `families/llama/moe.rs`'s `encode_llama_layer_moe` gained the same `RoutedSlot` parameter, `routed_blobs_banks` plumbing and `x_off`/bank-offset threading `encode_gemma4_layer_routed_moe` already has, returning the bound cache slots for the next token's `protect` set. `RealLlamaState::router_logits_f32` and `moe_x` widened the same way `RealGptOssState`'s did. **This is the pair that proves Step 1 needs no new kernel for ANY layout**: both drivers dispatch through `encode_moe_phase1_any` / `encode_moe_phase2_any`, the same layout-agnostic calls the sequential decode path already uses (Affine, GGUF Q4_K/Q6_K, MXFP4), so widening two more families cost zero new Metal code -- only the batched routed KERNEL (steps 2/3, `TURBOSPARK_ROUTED_BATCH`) stays INT4-affine-only and unwired for either. Verified byte-identical against sequential on the synthetic fixture (`real_forward_llama_moe_chunked.rs`, chunk-span sweep, a cache-too-small-to-pipeline case on a separate 2-expert/top-2 shape so top-k selects both experts every token and nothing ever misses after the first load, and a decode-continuation case) and on the real `Qwen/Qwen3-30B-A3B-GGUF` install pulled fresh for this verification (no install of this family's checkpoint remained on disk; `CLAUDE.local.md`'s old `qwen3moe-gguf.gturbo` reference had gone stale): greedy and sampled stdout md5-identical against a pre-change binary.

    **THE SIXTH FLOW, THE DENSE HALF OF THE QWEN LINEAR-ATTENTION FAMILY, LANDED 2026-08-29 AND IS NEITHER OF THE TWO SHAPES ABOVE.** `families/qwen/prefill.rs`'s `prefill_chunk_real_qwen_dense` serves `qwenGdnDense` (`qwen38-27b.gturbo`). The obvious precedent looked like `families/qwen/batched.rs` -- the M-row GEMM machinery this family already built for the MTP/DFlash2 verify pass -- and that is the WRONG one to copy: it implements steps 2-6 (GEMVs become GEMMs), sized for tiny drafter block depths and allocated only when a drafter is open. This driver is Step 1 again, same shape as dense `llama`'s and `muse_glimmer`'s: loop the EXISTING per-token kernels (`attn::encode_linear_block` for the gated-DeltaNet mask-2 layers, `attn::encode_full_attention_block` for the mask-1 ones, `dense::encode_qwen_layer_dense` for the FFN) inside a micro-batch, one command buffer for the whole thing since a dense layer has no router readback. **No new kernel, and -- a first among the Step-1 drivers -- no new buffer either**: every per-token intermediate the trunk's sequential flow already owns (`qwen.moe_x`, `qwen.h2`, the GDN scratch fields) is single-row and GPU-only, safe to reuse across tokens under commit-order execution (Gotcha 8 in `crates/gpu/CLAUDE.md`), exactly like `llama.moe_x`/`h2`.

    **The GDN recurrent state is the one thing no other Step-1 driver has to reason about, and it resolves for free.** `encode_linear_block`'s decode-shaped kernels advance `qwen.gdn.state_buffer(layer)` in place with no position argument, so calling it once per token, strictly in order, within one layer's inner loop before the next layer starts, reproduces sequential decode's math exactly -- Gotcha 4's constraint, satisfied by construction rather than by new machinery. It is also why cross-chunk continuity needs no handoff: the state buffer is the one sequential decode already reads and writes, so a prompt spanning several `prefill_chunk` calls carries it forward automatically.

    Two refusals are BY NAME, both specific to this family. An image prompt (`self.prompt_vision.is_some()`) is refused: the image injection in `produce.rs` is this family's only embedding call site (Gotcha 27), and this first cut stays text-only rather than growing a second site sight unseen. An open drafter (`self.real_mtp.is_some() || self.real_dflash.is_some()`) is refused too: the sequential dense branch fires the DFlash2 aux-capture hook on every pass, which this driver does not encode, so silently prefilling through it would leave the drafter reading a stale or empty capture. `supports_chunked_prefill()` folds both into its qwen clause so a caller routes around the driver entirely in the ordinary case; the driver keeps both checks as a backstop. `qwenGdnMoe` (Ornith 35B) is a follow-up, matching how `llama`'s two halves landed as separate steps.

    **`TURBOSPARK_BATCHED_GEMV` WAS WIRED THE SAME DAY** and this paragraph's "no new kernel, no new buffer" describes the DEFAULT arm alone; the seam's arm reuses `batched_layers.rs`'s three verify-pass encoders and allocates a `BatchedScratch` lazily. What stays refused is the WIDTH rather than the family (INT4-affine only), and the batched arm is NOT byte-identical on a real install -- see Gotcha 7's `TURBOSPARK_BATCHED_GEMV` bullet for the shape floor that makes that expected rather than a bug, and for the three guards the arm carries.

    Verified byte-identical against sequential on the synthetic fixture (`tests/real_forward_qwen35_chunked.rs`: chunk-span sweep `[1, 2, 3, 4, 7, 11]` crossing the fixture's GDN-then-full-attention layer mask, plus the batched arm's own sweep and allocation-counter case, and the width-refused, vision, open-drafter and MoE-still-refused cases) and on the real `~/models/qwen38-27b.gturbo` install: greedy and sampled stdout byte-identical against the sequential path, and `qwen38_quality_gate` / `qwen38_memory_oracle` both unmoved (reference-answer perplexity 4.9432, frozen digests all reproduced, peak footprint 663 MiB against the standing 750 MiB ceiling -- confirming the no-new-buffer claim rather than assuming it).

    **THE SEVENTH FLOW, `qwen4_exp`, LANDED 2026-09-05 AND FOUND A NEW CLASS OF HAZARD THIS PATTERN HAD NOT SEEN YET.** `families/qwen4/prefill.rs`'s `prefill_chunk_real_qwen4` is Step 1 again: per-layer `cb1` over the M-token attn_hc/GDN-or-QSA/hc_inject/mlp_hc/router loop, then a per-token routed-MoE loop with `RoutedSlot`/`routed_pipeline_banks`/`pending_routed` pipelining reused verbatim from this file. QSA needed NOTHING new: `encode_full_attention_block` already takes `pass: &mut PassEncoder` and already handles its own above-budget mid-layer commit (Gotcha 34), so calling it once per token in increasing order reproduces that unchanged, and the shared `qsa_positions` buffer stays safe for the same reason the routed half's pipelining does not disturb it -- a layer's whole attention-and-router half commits and waits before that layer's routed loop starts, so no two QSA layers' writes are ever in flight at once. GDN needed nothing either, matching dense qwen's own precedent.

    **PLE is where a REAL, SHIPPED bug lived, and it generalizes past this one family.** Four buffers needed the by-now-familiar M-row widening for the by-now-familiar reasons (`wide_x` crosses layers, `router_logits_f32` is read back in one host round trip, `hc_inject` and a new `moe_x` bridge the `cb1`/`"routed cb"` split for `mlp_hc` exactly as this paragraph's own rule below predicts). A FIFTH buffer, PLE's `ngram_emb`, needed the same widening for a DIFFERENT reason that the "who WRITES it" rule below does not by itself catch: its upload is `gpu::write_buffer_bytes`, a HOST write that executes the instant the encoding function runs, not a GPU dispatch queued for later -- so it does not respect command-buffer commit order AT ALL, where every other per-token buffer in this driver does. A single-row `ngram_emb` shipped in the first cut of this driver and left every token but the last in a micro-batch computing PLE from the WRONG token's n-gram embedding, silently: none of the `key_proj`/`value_proj` GEMVs that read it execute until the whole pass commits, by which point every token's host write has already landed, so the buffer holds only the LAST token's embedding for the pass's entire execution. Caught by `real_forward_qwen4_chunked.rs`'s `the_chunk_boundary_does_not_move_the_logits`, at chunk span 2, the first multi-token micro-batch it tried -- fixed the same way as `moe_x`: widen to `MAX_PREFILL_BATCH` rows, thread a row offset into `encode_ple_layer` for the write and both of its later reads. **The generalization: "who writes it, not who reads it" (this paragraph's own next rule) is about GPU dispatches specifically, and a HOST write masquerading as an ordinary per-token buffer populated by `write_buffer_bytes` is exactly the case that rule does not cover** -- any future family whose flow does its own host-side dequant-then-upload step (a second n-gram table, a second control-vector-style injection) owes this same check before assuming commit order protects it.

    Verified byte-identical against sequential on the synthetic fixture (`tests/real_forward_qwen4_chunked.rs`: whole-prompt and chunk-span sweep `[1, 2, 3, 4, 7, 11]`, a span crossing the QSA sparsity boundary at position 19, and the minimal-safe-slot-count case at `2 * top_k`) and against `real_forward_qwen4.rs`'s existing 17 cases, including its frozen digest, all unmoved by the buffer widening (logic did not change, only sizes grew).

    **AND THEN THE FIRST REAL-INSTALL RUN CRASHED, AT THE DEFAULT SLOT COUNT, WITH SEVEN GREEN SYNTHETIC CASES BEHIND IT.** `expert cache cannot place requested misses` (`crates/streaming/src/expert_cache.rs`), on an ordinary `turbospark-bench --model` prefill of a 426-token prompt. This is AGENTS.md Gotcha 64 on a second family, and the reason it reads as new is that the gotcha is written around `--expert-cache-slots 8`, as though 8 were the trigger. The trigger is `slots < 2 * top_k`. Gemma 4 routes top-8, so its default 16 leaves exactly 8 and fits BY ONE; this checkpoint routes **top-10** against the bench's pinned 16, so `16 >= 20` fails, `routed_pipeline_banks` degrades to `banks == 1`, the loop reserves the previous token's 10 slots through `RoutedSlot::protect`, and 6 places have to hold up to 10 misses.

    The reservation was never needed in that branch and the gotcha already said so: `protect` names slots a command buffer STILL IN FLIGHT is reading, and the `banks == 1` arm calls `retire_routed` BEFORE planning, so nothing is in flight when `protect` is consulted. `families/qwen4/prefill.rs` passes `HashSet::new()` there now, which is the same pairing `families/gptoss/prefill.rs`'s seam already makes (banks = 1 AND an empty protect set, together). **`families/{gemma4,gptoss,llama}/prefill.rs` still pass it unconditionally**, latent because none of them reaches the branch at its own default slot count.

    **WHAT LET IT SHIP IS THE PART TO CARRY, AND IT IS A NAMING FAILURE.** The suite had a case called `a_cache_too_small_to_pipeline_still_reproduces_the_sequential_logits` opening at `2 * TOP_K` -- a value that SATISFIES `slots >= 2 * top_k` and therefore takes the PIPELINED branch. The fallback had a test named after it and no test covering it, and the name is what stopped anyone looking. When the threshold is `>=`, a fixture at exactly the threshold is on the wrong side of it. `the_one_bank_fallback_reproduces_the_sequential_logits` opens at `TOP_K` instead, and its mutation (restoring the unconditional `previous_slots`) reproduces the real install's exact panic string on the synthetic fixture -- which is what says the fixture reaches the real path rather than merely a similar one.

    **What decides which buffers need a per-token row is who WRITES them, not who reads them.** Command buffers on one queue execute in commit order, so a GPU-only intermediate (`normed`, `q`, `o`, `attn_out`, `h1`, `h2`, `moe_acts`, `ffn_normed`) is safe to reuse across a chunk's tokens -- token t+1's dispatch cannot start before token t's finishes, which is the same guarantee the shared-expert-into-routed chain has always relied on. Four buffers are not GPU-only and do need rows: `x` (the residual stream, so it crosses layers), `router_logits_f32` (all M are read back by the HOST after one commit), and `dense_x` / `routed_x` (written in the attention half, read in the routed half, with a commit between). `routing_w` and the routed argument buffer are the two the host WRITES per token, so they are banked `ROUTED_BANKS` deep instead.

    **Pipelining the routed buffers costs a plan that must AVOID the in-flight token's expert slots**, which is what `RoutedSlot::protect` carries and what `ExpertCache::plan`'s `avoiding_slots` was always for. Below `2 * top_k` slots the cache cannot guarantee room for those plus this token's misses, and it ASSERTS rather than degrading, so the driver falls back to retiring before it encodes. That fallback is a throughput choice and must stay a numerics no-op; `a_cache_too_small_to_pipeline_still_reproduces_the_sequential_logits` pins it.

    **THE RESIDENT GEMVS BATCH TOO, BEHIND A SECOND SEAM** (`docs/BATCHED_PREFILL.md` step 6, `TURBOSPARK_BATCHED_GEMV`). The four attention projections become M-row GEMMs through `encode_gemm_any`, and so do the shared expert's three when the routed half is batched as well; that second half is also where the host saving is largest, since the per-token shared branch opens and commits its OWN command buffer per token. Everything with no weights to amortize -- norms, per-head norms, RoPE, attention, the residual adds, the router GEMV -- still loops. Two hazards it added, both silent if unguarded: a batched K/V projection can STRADDLE a sliding-window ring's wrap (`k_slot` validates one row, so it would run past the layer's buffer), which `ring_spans` splits; and `batch_q` is sized at the model's WIDEST head, because Gemma 4's five full layers are 512-wide against the sliding window's 256 and no fixture here has `head_dim != full_head_dim` to catch a wrong sizing, so a length check at the dispatch stands in for the test that cannot exist.

    **Do not reach for the expert-union plan.** An earlier design had one `plan_experts_cached` over the chunk's union replacing M per-token plans, worth "25.2% of prefill cut 3.3x". Measured, prefill's union is 41.5 distinct experts per layer at M=16 against the 24.1 the sequential path already loads at 32 slots, so it saves nothing -- and 41.5 requests against 32 slots trips the assert above. The hit rate before and after the driver landed reads 81.2% against 81.4%, which is the third independent confirmation. See Gotcha 14.

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
    ASSERTS rather than degrading (Gotcha 14) -- which is why the
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

20. **A PER-FAMILY CAPTURE HOOK HAS TWO HALVES AND THEY LIVE IN DIFFERENT
    PLACES.** The per-layer `encode_resid_capture` calls fill a buffer inside
    the layer loop; a separate `record_pass` after the command buffer is
    waited on is what keeps a snapshot. Wiring only the first gives
    `[resid-capture] no non-prefill pass ran; wrote nothing` -- loud, which is
    the good failure mode, and still a family half-wired. Grep for
    `record_pass` as well as for the encode when adding a family. The steering
    edit has NO second half, so the two hooks are not symmetric even though
    they share a boundary and are gated by one predicate
    (`steering::family_dispatches_steering`).

21. **`encode_steering` and `encode_resid_capture` hardcoded the edited row's
    offset at 0 for the life of the surface, and the first family with more
    than one row per `scratch.x` at a time found it.** Every caller before
    Gemma 4 -- the qwen flow's per-token pass, its M-row batched verify, and
    `families/llama/` -- reads or edits exactly one row and it always sits at
    offset 0, so both functions took the source/destination as
    `(&scratch.x, 0)` literally. Gemma 4's chunked-prefill driver
    (`prefill.rs`'s per-token routed loop, `moe_batch.rs`'s batched-routed
    tail) packs several prompt tokens into `scratch.x` for one layer at once,
    each at its own slot offset (`token * hidden * 2`), so a caller copying
    the qwen/llama call site verbatim would steer (or capture) row 0 on every
    call regardless of which token was actually being processed -- fluent,
    finite, and wrong for every token past the first of a micro-batch, with
    no error anywhere. Both functions now take an `x_off: u64` parameter;
    every pre-Gemma call site passes `0` explicitly, which the qwen38
    quality gate's frozen perplexity and digests confirm moves no bytes.

    **DISPATCH FOR EVERY ROW, NOT JUST THE ONE THAT MATTERS, AND LET COMMIT
    ORDER DO THE REST.** Steering wants every token of a micro-batch edited,
    so that half is straightforward. Capture only wants ONE snapshot (the
    micro-batch's last token, when `want_head` is true), and the fix is NOT
    to conditionally call `encode_resid_capture` only for that token: the
    destination is one fixed region per layer, dispatches within a layer
    execute in COMMIT ORDER (Gotcha 8's territory, one boundary over), and
    Gemma's per-token and per-sub-batch loops both process rows in strictly
    increasing token order -- so calling the copy unconditionally for every
    row leaves the LAST-processed row (which is always the micro-batch's
    last token) as the one `record_pass` reads back. A conditional call would
    need to know in advance which `t` is "last", which the per-token loop
    does not carry and the batched loop's sub-batch boundaries make awkward
    to compute; letting overwrite order do it is both simpler and matches
    how the sequential decode path already behaves (one call, one row, no
    condition needed because there is only one).

    Mutation-checked, per call site: reverting `prefill.rs`'s hook to a
    hardcoded 0 reddens ONLY the per-token chunked-vs-sequential case in
    `real_forward_gemma4_steered.rs`; reverting `moe_batch.rs`'s reddens ONLY
    the batched-routed case. Neither mutation touches the other's coverage,
    which is what says the two are independent call sites rather than one
    path exercising both.

22. **THE BATCHED ROUTED HALF SERVES TWO FAMILIES NOW, AND WHICH ONE WAS
    BUILT SECOND WAS DECIDED BY MEASUREMENT RATHER THAN BY THE ORDER THE
    WORK WAS SCOPED IN.** `docs/BATCHED_PREFILL.md` step 5 is titled "GGUF
    Routed Pair Widening" and the GGUF arm is the one NOT built.
    `families/gptoss/moe_batch.rs` (2026-08-27) drives
    `gpu::moe_prefill_batch_gguf`'s MXFP4 pair under the same
    `TURBOSPARK_ROUTED_BATCH` seam, measuring **1.31x** on the real 20B
    install against Gemma 4's 1.19x for the same step.

    **The three measurements that chose it, taken before either arm was
    written** (`TURBOSPARK_PHASES=1`, the frozen `long-synthesis` prompt, both
    real installs, slot count `auto`): `gpt-oss`'s routed pair is 61.4% of
    its prefill GPU device time where Gemma's is 38.2% and `qwen3moe`'s is
    36.5%; its un-batchable expert `pread` bucket is 8.2% where the two
    128-expert families run 25-37%; and with 32 experts at top-4 its
    measured union stays under the slot count at EVERY M, so it is the only
    family here that reaches M=16 -- both 128-expert families cap at M=8 on
    `union(M) <= slot_count`. `qwen3moe` would also have cost TWO kernels
    (Q4_K on gate/up, Q6_K on down) where MXFP4 covers both phases with one.
    Do the share-and-reachable-M arithmetic before widening a kernel to a
    third family; it is two runs and it inverted the planned order here.

    **The driver is Gemma's with two subtractions and one addition, and all
    three are already in this family's sequential flow.** No shared expert
    (phase 2 seeds from `batch_zero` and the routed sum reaches the stream
    through one raw residual add -- passing `x` there would add the residual
    twice), no sandwich norms (the tail is that one add rather than Gemma's
    norm/add/norm/add/scalar-mul), and the ROUTER BIAS added to each token's
    read-back logits BEFORE its top-k, which is `families/gptoss/moe.rs`'s
    first six lines. Getting that last one wrong selects the wrong experts
    and still reads fluently; the mutation that drops it reddens exactly the
    four batched cases in `real_forward_gptoss_chunked.rs` and none of the
    five per-token ones.

    **`BatchedRoutedScratch` is allocated on FIRST USE**, per the rule
    `families/gemma4/state.rs` states at length: a run that never asks for
    the batched half allocates none of it, so the frozen `gptoss_memory_oracle`
    row keeps describing the engine that shipped before this landed. It is
    ~1.1 MiB against a 5,700 MiB ceiling, so this is consistency with a rule
    rather than a memory win.

    **Its wide argument buffer comes from the MXFP4 pair's OWN shader
    library**, through `RoutedBlobsWideBuffer::new_for` / `bind_for` rather
    than the affine pair's `new` / `bind`. The two libraries declare
    `RoutedBlobsWide` identically, so one encoder works today; taking it from
    the function that will READ the buffer means a layout that ever diverged
    is a compile-time mismatch instead of a silently misread pointer array.

    Verified byte-identical against the sequential path on the synthetic
    fixture (chunk spans 1/2/3/5/8/11, several straddling the fixture's
    8-token sliding window, plus a two-slot case forcing the greedy
    sub-batch shrink) and on the real install: a THREE-way md5 identity --
    the PRE-CHANGE binary, the post-change binary with the seam off, and the
    post-change binary with it on (`7281650e...` greedy, `78aae4b3...`
    sampled). The pre-change arm is the one that says the DEFAULT path did
    not move; an on-vs-off comparison inside one binary cannot, since both
    of its arms carry whatever the change did.

    **WIRING A SECOND FAMILY TO THIS SEAM EXPOSED THAT THE UNWIRED ONE WAS
    SILENTLY IGNORING IT, and that is the part most worth carrying.**
    `TURBOSPARK_ROUTED_BATCH=1` on the real `qwen3moe` install ran to
    completion with no message and no batching, because
    `families/llama/moe_prefill.rs` had no branch reading the flag at all --
    so a caller who set it measured the per-token engine and would have
    reported the number under the batched arm's label. That is the
    `encode_gemm_any` doctrine's exact failure, in the one family that had a
    routed half and no batched kernel for it. Now a named refusal, pinned by
    `the_batched_routed_seam_is_refused_by_name_on_this_family`.

    Two things about how it was found. It came from checking a sentence in
    `docs/BATCHED_PREFILL.md` ("refused by layout, as it did before")
    against the binary, not from a test going red -- the sentence had been
    written from the AFFINE driver's guard and was never true of the family
    that has no driver. And the rule the fix follows is narrower than
    "refuse everywhere unwired": the DENSE drivers ignore the same flag and
    are right to, because `TURBOSPARK_ROUTED_BATCH` asks for the routed half
    as one dispatch pair and a dense family has no routed half to refer to.
    Refuse where the request is MEANINGFUL and unserved.

23. **EVERY TEST IN A PERTURBATION-STYLE FIXTURE FILE CAN BE SELF-RELATIVE,
    AND THEN THE FILE CATCHES ALMOST NOTHING.** The reachability pattern
    `docs/NEW_MODEL.md` recommends (perturb a tensor, require the logits to
    move) rebuilds its own baseline inside the same binary, so a mutation
    changing the MATH for every arm equally leaves every case green: the
    baseline carries it too. Measured on `tests/real_forward_muse.rs`,
    2026-08-15: six mutations, one reddened, and that one only because
    dropping the attention output gate makes a tensor UNREACHABLE. Dropping
    a Q scale, rotating a NoPE layer, swapping a norm convention, dropping
    an output multiplier and collapsing two epsilons all survived twelve
    tests.

    The fix is one number: a FROZEN DIGEST over a deterministic synthetic
    install's logits, the only assertion in such a file that compares
    against something computed BEFORE the mutation. It took the same six to
    five reddening. It is a CHANGE DETECTOR and not a correctness claim
    (untrained weights cannot say the arithmetic is right, only that it is
    what it was), so re-freezing needs a stated reason and a digest updated
    reflexively protects nothing.

    **Its POSITION matters, which the digest pattern does not warn about.**
    The obvious implementation reuses the file's `first_logits` helper, which
    produces at position 0, where a softmax over one key is exactly 1.0: so
    attention returns V alone and no q/k transform is observable. Measured,
    at a single position the correct model and one with flipped q/k norms
    digest IDENTICALLY (`312e17a0`); over eight positions they differ. A
    digest is only a change detector for changes its inputs can reach.

    Two limits worth knowing. Collapsing an RMS epsilon of 1e-5 into 1e-8
    survives even the digest: at FP16 with `mean_sq` near 1 those differ by
    ~5e-6 relative, an order of magnitude under FP16's resolution. Epsilon
    VALUES are pinnable offline against the checkpoint's `config.json`, but
    which epsilon reaches which norm site is a question only a real-model
    quality gate answers. And a file with no digest at all catches nothing of
    this class: `real_forward_qwen35.rs` was seven reachability cases, so
    flipping that flow's q/k norms to the centered convention left it green,
    left `real_forward_qwen.rs`'s eight green, and was caught only by
    `qwen38_quality_gate` on a real 14 GB install.

24. **THE VISION TOWER IS THE ONLY READER OF `vision.` TENSORS, AND THAT IS A
    CONSTRAINT RATHER THAN AN OBSERVATION.** `readable_resident_dtype` takes
    the tensor's NAME as well as its dtype tag, accepting FP16 (tag 2) under
    `vision.` and refusing it everywhere else. The hazard it scopes around is
    real and unchanged for text: `norm_view` and `read_bf16_host` are
    dtype-BLIND, resolving an unquantized tensor by BYTE WIDTH and decoding it
    as BF16, so a tag-2 tensor either of them reaches is MISREAD rather than
    rejected -- values wrong by up to 2^112, from an install that opened
    cleanly (AGENTS.md Gotcha 45). Granting tag 2 outright is not a narrower
    bug than the one the refusal prevents; it is the same one.

    What makes the scoping true is that `vision/weights.rs` is the whole of
    what reads those tensors, and its `fp16_view` checks the TAG as well as
    the width so it cannot quietly become the general reader. **Any new reader
    of a `vision.` tensor owes the same check**, and any text-path helper that
    grows a `vision.` call site reopens the case the exception was written to
    close.

    Two refusals in the same file exist for failures nothing else catches. A
    block sub-tensor whose dtype is not `"fp16"` is refused BY NAME, because a
    same-width BF16 run passes every length check and reads as a different
    number -- the mirror of the hazard above, pointing the other way. And a
    role offset that is not 4-BYTE ALIGNED is refused, because sub-tensors
    pack back-to-back with no per-role padding: alignment is an accident of
    the preceding tensors' sizes rather than a property of the format, the
    real tower's element counts are all even so every offset happens to land,
    and Metal's `setBuffer:offset:` requires it. A violation would be a
    validation-layer abort a long way from the layout that produced it.

25. **NO COMPOSITION TEST IN THIS REPO CAN SEE WHICH GELU THE VISION TOWER
    USES, AT EITHER LEVEL, AND BOTH LEVELS SAY SO AS AN ASSERTION.** The
    tower uses BOTH forms -- tanh in each block's MLP, the exact erf form in
    the merger -- and they agree to about 3e-4. A block's output already
    carries the accumulated FP16 error of two norms, five GEMMs and an
    attention; a whole TOWER's carries 27 of those, measured at 8.4e-3
    relative on the synthetic fixture. So the signal is more than an order of
    magnitude under the bound either parity assertion has to allow, and no
    tightening fixes it: the bound is measuring the storage, not the choice.

    Found by MUTATION rather than by review, twice. Swapping either GELU for
    the other leaves all nine cases in `tests/vision_tower_synthetic.rs` green,
    exactly as it leaves `crates/gpu/tests/vision_block_parity.rs` green one
    level down (that crate's Gotcha 9). `the_gelu_choice_is_invisible_at_this_
    bound` states the limitation as a test -- the two towers differ, and by
    less than the parity case's own bar -- so a reader cannot assume the
    composition test covers it. What DOES pin the choice is the pair of
    per-kernel cases in `crates/gpu/tests/vision_parity.rs` plus the one-line
    call site in `vision/block.rs` and `vision/stages.rs`, and nothing else.

26. **THE TOWER'S SECOND STREAMER SLOT IS NOT LOAD-BEARING YET, AND THE
    MUTATION THAT SAYS SO SURVIVES ON PURPOSE.** `VisionTower::run` reads
    block `n` into slot `n % VISION_SLOTS` and commits-and-waits per block, so
    the host cannot overwrite a slot the GPU is still reading -- because the
    GPU has finished. ONE slot would be correct today. The second is the shape
    the later read-pool prefetch needs, and alternating now makes that a
    one-line change rather than a restructure.

    Replacing `n % VISION_SLOTS` with `0` leaves every case in the synthetic
    gate green, which is the PREDICTED result and is recorded here so nobody
    reads the double buffer as doing something it is not. The same run's
    residency case still asserts `2 x block_stride`, because that is the
    footprint the milestone commits to rather than a consequence of
    correctness. A prefetch that removes the per-block wait makes the
    alternation load-bearing, and the case above becomes a real guard at that
    point rather than a documented no-op.

27. **THE IMAGE INJECTION IS A HOST WRITE INTO `scratch.x`, AND IT IS SOUND
    BECAUSE OF WHEN RATHER THAN BECAUSE OF A BARRIER.** At an image-pad
    position `families/qwen/produce.rs` blits the vision tower's FP16 row
    straight into the residual stream instead of encoding an embedding
    lookup -- the first code in this repo to write that buffer from outside
    the decode flow. What makes it safe is that `pass` is not committed until
    the first router wait far below, the buffer is shared storage, and a host
    write landing before commit is visible to every dispatch in that command
    buffer (Gotcha 8's rule in `crates/gpu`, read from the host side). Move it
    after any `pass.commit()` and it becomes a race the GPU wins silently,
    with the previous token's residual reaching layer 0.

    **The lookup is REPLACED, not blended.** The placeholder id carries no
    meaning, so adding the table's row would mix a text embedding into every
    patch. `a_span_position_ignores_its_token_id_and_a_text_position_does_not`
    is the guard and the PAIR is the point: the first half alone passes
    against a flow that ignores token ids entirely, the second alone against
    one that never blits.

    **This is the family's ONLY embedding call site**, which is what makes
    one seam sufficient. The qwen flow is the one `supports_chunked_prefill()`
    answers `false` for, so there is no chunked driver with a second call, and
    `produce_prefill` is the same inner function under `skip_head`. A family
    that acquires a chunked driver acquires a second site with it.

28. **THE mRoPE DISPATCH CONDITION IS A PROPERTY OF THE DATA, NOT A
    CLASSIFICATION OF THE TOKEN.** `RopePosition::Triple(t, h, w)` with
    `t == h == w` takes the PRE-EXISTING `encode_rope_neox_subdim`, and
    `get_rope_index` gives every text token of a mixed prompt exactly that --
    including the text between and after images (`docs/VISION_PHASE0.md` item
    2). So the divergence test IS the "is this an image pad" test: nothing
    separate has to be plumbed and the two cannot drift out of step.

    `position` keeps its other two jobs either way. It is still the KV slot
    index and still the `position + 1` attention span, so only the ANGLE
    moves and vision reaches this family without touching the cache at all.

    **THE DEGENERATE ARM IS A NO-OP TODAY AND THE MUTATION THAT SAYS SO
    SURVIVES ON PURPOSE.** The two kernels share `apply_neox_pair`, so
    `rope_mrope_interleaved` at `t == h == w` produces the identical bits;
    deleting the short-circuit leaves all nine cases in
    `tests/vision_inject_synthetic.rs` green, the frozen digest included. That
    is the PREDICTED result and it is the end-to-end confirmation of what
    `at_t_equals_h_equals_w_it_is_bit_identical_to_rope_neox_subdim` asserts
    in `crates/gpu`, reached here through the real trunk. The arm is kept for
    two reasons that are not numerical: it keeps a text token of a MIXED
    prompt on the dispatch path the pre-vision engine used, so a future change
    to the mRoPE kernel cannot reach one at all, and it makes the claim rest
    on which function is called rather than on the shader compiler continuing
    to agree. Gotcha 26's shape, one feature over.

29. **`PromptVision` SURVIVES BOTH `reset()` AND `rollback`, AND ONLY AN
    EXPLICIT `clear_prompt_vision` DROPS IT. M-V5 HAD THIS BACKWARDS AND IT
    SHIPPED A HALLUCINATING `--image`.** The original rule was "reset clears
    it", on the sound-sounding reasoning that a new generation is a new prompt
    and a bulk-OCR loop must not inherit page N's spans. But
    `run_raw_completion` calls `producer.reset()` at ENTRY, and a caller sets
    the map just BEFORE that call -- so the clear landed on the map for the
    very prompt about to be prefilled. Every image run prefilled placeholder
    embeddings and produced a fluent description of a page it had not been
    shown, with the right prompt length and no error anywhere.

    **The whole test suite missed it because every case drove `produce`
    directly**, this crate's synthetic file and `crates/bench`'s cross-engine
    dump alike. The path a front end actually takes -- set a map, run the
    ordinary loop -- was untested until M-V7's first real run.
    `an_injected_map_survives_the_generation_loops_own_reset` is the guard and
    it goes through `run_raw_completion`.

    What keeps a bulk-OCR loop safe now is the CALLER consuming the map per
    page (`crates/cli`'s loop calls `clear_prompt_vision` before building each
    one). A caller who forgets gets NO injection rather than the previous
    page's, which is the safe direction: a model handed placeholder embeddings
    answers vaguely instead of describing the wrong picture confidently.
    `only_an_explicit_clear_drops_the_injection_map` pins all three halves.

    Its six construction checks are refusals rather than tolerances because
    each is a wrong image reaching the model FLUENTLY: triples covering a
    different prompt, images and spans disagreeing in count, a span longer
    than its tower's merged-token count, overlapping spans, a span past the
    prompt end, and a `rope_delta` that would drive a decode position below
    zero. Not one of them fails at runtime on its own.

30. **PREFIX KV REUSE LIVES IN `kv_prefix.rs`, AND THE THREE THINGS THAT
    NEARLY MADE IT USELESS ARE ALL "THE OBVIOUS DESIGN IS THE WRONG ONE".**
    A chat client resends the whole transcript every turn and
    `run_raw_completion` reset and re-prefilled it every time, so a message
    cost the whole conversation again; on a streaming MoE install every
    replayed position also re-read its experts. `LogitProducer::try_reuse_prefix`
    is the seam, off by default, opted into per session with
    `RealForwardRunner::set_prefix_reuse`. Measured on the real 26B install:
    prefill 1.777s -> 0.153s (**11.63x**) on a 65-token prompt with a 54-token
    shared body, and turn 2's tokens byte-identical to the re-prefilled
    reference on every arm.

    **THE RECORD IS THE IDS THAT WERE FED, never a count derived from the
    caller's bookkeeping.** Whether the last sampled token was fed back,
    whether a chunked prefill ran to completion and whether generation
    stopped early all differ per caller, and the failure mode of getting it
    wrong is not a crash: it answers the next turn from a state belonging to
    a different conversation. `kv_prefix` records in `produce` and
    `prefill_chunk`, on success only, and TAINTS on anything ids cannot
    describe -- `set_prompt_vision` (a placeholder span has the same ids
    whatever picture filled it, so an id match would call two images equal)
    and `verify` (a speculative block whose accept count this seam is not
    told).

    **IT MUST BE THE LONGEST COMMON PREFIX, NOT AN ALL-OR-NOTHING MATCH.**
    The first design required the whole record to be a prefix of the new
    prompt, which is correct, ships, passes every test, and NEVER FIRES: the
    record covers the previous prompt AND its reply, and the next prompt
    carries that reply back as re-rendered TEXT, so re-tokenizing it does not
    reproduce the generated ids. Measured in the real chat REPL at 0/33 and
    0/49 over three turns. LCP plus a cursor rewind reads 13/33 and 29/49 on
    the same transcript, and the gap to the full prompt is just the
    generation-prompt suffix. The rewind is what buys it, and
    `try_reuse_prefix` refuses it in two cases: a family with RECURRENT state
    (the qwen flows' gated DeltaNet folds history non-invertibly and nobody
    took the ~60 MiB snapshot last turn) and a SLIDING-WINDOW ring past its
    slack (`max_safe_rewind`). Both return 0 and cost a full prefill.

    **AND IT HAS TO BE WIRED INTO THE LOOP CALLERS ACTUALLY TAKE.** Wiring it
    into `run_raw_completion` alone left the REPL at 0/33 for a third
    measurement: every family that supports chunked prefill routes through
    `run_raw_completion_chunked`, so the CLI and the server never reached the
    optimised path. Two arithmetic traps live in that second wiring, and
    `prefill_chunk_spans` makes both easy: the walk is given `reused` as its
    base so `start_position` is ABSOLUTE, while `token_offset` and
    `completed_count` count from the walk's own start and need `reused` added.

    **THE MEMORY SIDE IS A HIGHER TROUGH, NOT A HIGHER PEAK, and no frozen
    row can move.** `KvCacheManager::reset` calls `advise_dontneed` over
    every K/V buffer, so skipping it leaves those pages resident between
    turns. The buffers themselves are allocated once at `open` and a single
    turn already reaches the same high-water mark, so the PEAK the oracles
    measure is unchanged; what rises is the floor between turns. Reasoned
    rather than measured, and safe to leave that way because no harness opts
    in: the protocol, both oracles and both quality gates run one generation
    per process, and `--messages-file` does not enable reuse at all.

    **THE MUTATION CHECK IS THE POINT OF THAT LAST PARAGRAPH.** All three
    chunked mutations (spans from 0, position not made absolute, chunk sliced
    from the wrong base) SURVIVED the real-model suite, because every test in
    it drove the sequential loop and could not see the chunked one at all.
    `chunked_prefill_reuse_generates_the_same_tokens_and_actually_fires`
    closes it, and asserts the reuse HAPPENED as well as that the tokens
    match -- without that clause the equality is the trivial one, two full
    prefills agreeing with each other.

31. **THE VISION TOWER'S MAPPED RESIDENCY NEEDED A `base: u64` PARAMETER THE
    ROUTED FAMILIES DID NOT, BECAUSE THE TOWER MAPS ONE BUFFER RATHER THAN
    ONE PER LAYER.** `TURBOSPARK_VISION_RESIDENCY=mapped` (`vision/mod.rs`,
    `docs/EXPERT_RESIDENCY.md`) is its own seam, deliberately never
    `TURBOSPARK_EXPERT_RESIDENCY` -- reusing that variable would move the tower
    silently for anyone A/Bing routed residency, exactly the silent-ignore
    failure `mapped_residency_refusal`'s own doc comment (`real_forward_init.rs`)
    exists to prevent, one seam over. There is also no per-family refusal
    function for it: the tower is family-agnostic, so the only gate is
    whether it opens at all.

    The routed families map `Vec<Option<MetalBuffer>>`, one buffer per
    layer, because a routed install has many layers each with their own
    experts. The tower has exactly ONE pseudo-layer (`depth` blocks are one
    layer of a reused `PackedExpertsLayout`), so `VisionTower` maps a
    SINGLE buffer over the whole tower instead. That changes the addressing
    contract: `block::encode_block`'s twelve `roles.at(role)` reads are
    offsets relative to the START of ONE block's own blob, which is exactly
    right when `slot` is a pread'd per-block buffer (the existing arm,
    starting at 0) and wrong when `slot` is the mapped arm's single buffer
    spanning every block concatenated. `encode_block` therefore gained a
    `base: u64` parameter (`mapped_layer.expert_offset(n)` for the mapped
    arm, `0` for the pread arm) added to every role offset; the pread call
    site's `base = 0` is a no-op, which is what makes the change verifiable
    as byte-identical rather than merely argued
    (`tests/mapped_vision_residency.rs`).

    **Landed in two steps, mirroring this crate's own chunked-prefill
    discipline of separating a structural change from a latency one so a
    regression has one cause.** Step one wired the mapping with the
    per-block `commit_and_wait` unchanged for both arms. Step two dropped
    that wait for the MAPPED arm only: it has no `pread` step and no
    per-block-overwritten slot (the mapped buffer is read-only for the
    whole run), so the wait there was pure CPU-side serialization with no
    data-hazard purpose, and correctness rests on the same commit-order
    guarantee Gotcha 8 already relies on. The wait stays UNCONDITIONAL
    whenever a per-block host readback is requested
    (`run_with_stages`'s cross-engine capture, `TURBOSPARK_VISION_OVERFLOW`),
    since those need the GPU to have actually finished before reading
    `s.x` from the host -- skipping it there would read stale or
    in-flight bytes with no error.

    **ENGAGEMENT NEEDS ITS OWN PROOF, SEPARATE FROM BYTE-IDENTITY.**
    `RealForwardRunner::vision_residency_is_mapped()` reports which arm the
    tower actually took (`None` before the first image, `Some(bool)`
    after), because a byte-identity test alone cannot tell "the mapped arm
    ran and matched" from "the mapped arm silently fell through to
    pread" -- both pass parity trivially in the second case. Measured by
    mutation: forcing the tower to always take the pread branch reddens
    ONLY the engagement assertion in `tests/mapped_vision_residency.rs` and
    leaves the byte-identity one green, which is the silent-fallback
    failure the accessor exists to catch, demonstrated rather than argued.

32. **A SESSION POOL (`--session-slots`, ROADMAP section 4's Option 3) IS A
    SWAP, NEVER A COPY, AND A "SHALLOW MATCH" MEANS SOMETHING DIFFERENT ONCE
    ONE EXISTS.** `crate::session_pool::SessionPool` holds `session_slots -
    1` PARKED `SessionSlot`s (`kv: gpu::KvCacheManager`, `gdn:
    Option<gpu::GdnStateManager>`, `kv_prefix`, a monotonic `last_used`
    tick mirroring `streaming::ExpertCache`'s `slot_last_use`/`use_clock`
    pair) alongside the runner's existing single `kv`/`real_qwen.gdn`/
    `kv_prefix` fields, which stay the "live" slot. `RealForwardRunner::
    select_session` (called first thing inside `try_reuse_prefix`) scores
    every parked slot's `kv_prefix.common_prefix` against the incoming
    prompt and, if one beats the live session's own score, moves it onto
    the live fields via `std::mem::replace` -- an O(1) struct swap, never a
    memcpy, which is the entire reason KV cache sharing works here where a
    per-switch host copy was already rejected for that resource (see the
    ROADMAP research this Gotcha's own commit message cites). `reset()`
    (every `LogitProducer` caller's fallback when no match is found,
    including the speculative loop, which never calls `try_reuse_prefix`
    at all) is the OTHER half: instead of clearing the live session in
    place, it PARKS it and promotes the pool's LRU slot to become live
    before clearing THAT one. `session_slots <= 1` makes the pool empty and
    every new code path here a true no-op, which is the whole byte-identity
    guarantee for the shipped default.

    **TWO REAL BUGS SHIPPED IN THE FIRST DRAFT AND BOTH WERE FOUND ONLY BY
    A REAL-INSTALL TEST, NOT BY THE SYNTHETIC ONE WRITTEN ALONGSIDE IT.**
    `crates/runtime/tests/session_pool.rs` proves the swap/park mechanics
    with fully controlled token arrays and passed on the FIRST try; the
    real bug was in what "a shallow match" means against REAL tokenized
    text, which a hand-picked fixture cannot manufacture by accident.

    **Bug 1: a partial rewind is DESTRUCTIVE once there is somewhere else
    for the discarded content to go, and the original `try_reuse_prefix`
    had no way to tell "genuine continuation with minor tail divergence"
    (Gotcha 30's sanctioned case) from "coincidental overlap with an
    UNRELATED live session".** Measured on a real Gemma 4 install serving
    two interleaved conversations: two prompts sharing NOTHING but a chat
    template's opening tokens (`<bos><start_of_turn>user\n`, 6 tokens)
    still score a nonzero `common_prefix` against each other. Without a
    pool this is harmless -- the discarded tail was never going to be read
    again regardless of what overwrote it. WITH a pool, the same shallow
    match is destructive: it rewinds the live session by a few tokens and
    lets the new prompt's prefill overwrite the rest IN PLACE, which
    silently destroys a real, valuable conversation that had never been
    given a chance to be parked (parking only happens inside `reset()`,
    and this rewind path exists specifically to avoid calling it). The fix
    is a discriminator on the rewind, `back > keep` (discarding more than
    is kept is never what a genuine continuation looks like), refusing so
    the caller's `reset()` runs and parks the live session instead.

    **Bug 2, found by the SAME real-install test after fixing bug 1: the
    `back > keep` bar is too strict for a session `select_session` has
    ALREADY vetted, and applying it there undoes the swap's own verdict
    for no reason.** After bug 1's fix, the real install's second
    conversation's second turn STILL failed to find its own parked
    session -- `select_session` correctly swapped it in (proven by the
    debug trace: `best parked idx=0 score=15`), but the resulting `keep`
    (15 of a 42-token real session) still tripped `back > keep`, refusing
    a match the pool had already identified as the best available one.
    Refusing it does not preserve anything -- it forces ANOTHER `reset()`
    that re-parks the very session just swapped in, evicts whatever ELSE
    was parked to make room, and ends up reusing nothing at all; measured
    on the real install, this cascade needlessly evicted an unrelated
    THIRD conversation's session to serve a request that already had its
    own real match in hand. **15 of 42 (36%) is not a low bar by this
    codebase's own established norm**: the CLI's real `--chat` REPL
    measures 13/33 and 29/49 (39% and 59%) as WORKING reuse
    (`crate::kv_prefix`'s own module doc), both of which a flat `back >
    keep` bar would refuse. The fix: `select_session` returns whether it
    swapped, and `try_reuse_prefix` only applies the `back > keep` scrutiny
    when it did NOT -- once the pool has already vetted a candidate as the
    best match FOR THIS PROMPT, a small `keep` is not a coincidence to
    distrust, it is this codebase's own definition of reuse working.

    **The general shape, worth carrying past this feature**: a threshold
    meant to catch "this match is too coincidental to trust" has to be
    scoped to the case that was NEVER vetted by anything else. Applying it
    a second time to a decision another mechanism already made second-
    guesses that mechanism for free and can undo real, correct work -- and
    the failure mode is not a crash, it is quietly reusing NOTHING where
    something real was available, which reads as "the feature just doesn't
    help much" rather than as a bug with a specific, findable cause.

    **A live `GdnStateManager` swap, not the `GdnSnapshot`/`snapshot()`/
    `restore()` pair `RollbackPoint` uses for speculative rollback.** That
    pair is a real host memcpy (tens to ~150 MiB depending on family,
    `crates/gpu/src/gdn_state.rs`'s own doc), which would reintroduce for
    GDN state exactly the per-switch cost a swap-based pool exists to avoid
    paying for KV. A second LIVE `GdnStateManager` costs one extra
    allocation at open and zero cost per switch, swapped by
    `RealForwardRunner::swap_live_session` in the SAME `mem::replace` call
    that moves `kv`, gated on `self.real_qwen.as_mut()` so a non-GDN
    family's `SessionSlot::gdn` stays `None` throughout. **This family is
    also where the two refusals `try_reuse_prefix` already carries
    (`crates/runtime/CLAUDE.md` Gotcha 30's `real_qwen.is_some()` check)
    interact with pooling in a way worth knowing**: that check refuses ANY
    rewind unconditionally on a GDN family, so `try_reuse_prefix` can only
    ever succeed there via the `back == 0` exact-continuation path -- the
    `back > keep` discriminator (and its `!swapped` exemption) never even
    gets reached on this family, because the pre-existing guard answers
    first. `tests/session_pool.rs`'s
    `a_shallow_match_does_not_destroy_a_gdn_familys_recurrent_state` proves
    the swap itself (via `select_session`, unconditionally called before
    any rewind decision) still correctly moves the GDN state alongside the
    KV, independent of which downstream check ultimately allows reuse.

    **Eviction is reported (`LogitProducer::session_slot_evicted`,
    `RawDecodeResult::session_slot_evicted`) only when the LRU slot
    `reset()` promotes actually held real content** (`promoted.kv.position()
    > 0`) -- a freshly-allocated slot that has never served a request costs
    nothing to overwrite, and reporting an eviction for it would train an
    operator sizing `--session-slots` to distrust a signal that fires on
    every cold pool.

33. **`try_reuse_prefix`'S RECURRENT-STATE GUARD NAMES ONE QWEN FAMILY AND
    NOT THE OTHER, AND THE SECOND ONE HAS THE IDENTICAL HAZARD.** Found
    2026-09-04 while investigating `qwen4_exp`'s quality-gate determinism
    failure (see `docs/QWEN4_EXP.md`'s "The quality gate is BLOCKED"
    section; this gotcha is a real, separate gap the investigation
    surfaced, NOT the cause of that failure). The rewind guard at
    `real_forward_traits.rs`'s `back > 0` branch refuses any rewind when
    `self.real_qwen.is_some()`, for the reason Gotcha 30 states: gated
    DeltaNet folds history into a fixed-size accumulator through a
    non-invertible update, so there is no going back without a `RollbackPoint`
    snapshot, and rewinding the KV alone would leave that accumulator
    describing tokens the KV no longer holds -- fluent wrong output, not an
    error. `qwen4_exp` carries the SAME kind of recurrent state
    (`RealQwen4State::gdn: gpu::GdnStateManager`, reset the same two-line
    way `families/qwen4/state.rs::reset` handles it), and the guard has no
    `self.real_qwen4.is_some()` arm beside the one it has.

    **This is inert TODAY because prefix reuse is opt-in and nothing turns
    it on for this family yet.** `prefix_reuse_enabled` defaults `false` at
    open (`real_forward_open.rs`), `crates/bench`'s openers never call
    `set_prefix_reuse`, and this session found no other caller wired to this
    family either -- so `try_reuse_prefix` returns 0 on every path reaching
    it today regardless of this gap, and `reset()` runs unconditionally.
    That is also how this gotcha was distinguished from the quality gate's
    real bug: the same investigation confirmed `reset()` executes before
    every generation in that gate's flow, which ruled prefix reuse OUT as
    the cause there.

    **The gap becomes live the moment ANY caller enables prefix reuse for
    `qwen4_exp`** -- a future `--chat` REPL wiring, session pooling
    (Gotcha 32), or a benchmark opting in for its own reasons. At that
    point a shallow `keep` with `back > 0` would rewind the KV while
    leaving the GDN accumulator exactly where the discarded generation left
    it, which is Gotcha 30's failure mode arriving on the family the guard
    forgot. Add `|| self.real_qwen4.is_some()` to the existing check (or
    generalize both to one predicate over "families with recurrent state
    and no rollback snapshot") before wiring prefix reuse to this family for
    any reason.

34. **`qwen4_exp`'S QSA LAYER COMMITS MID-LAYER ABOVE THE INDEXER BUDGET,
    AND THE BUDGET BOUNDARY IS UNOBSERVABLE IN OUTPUT.** Since 2026-09-05
    `families/qwen4/attn.rs::encode_full_attention_block` takes `pass: &mut
    PassEncoder` (every other family's block takes `&PassEncoder`): above
    `index_top_k` complete blocks it scores the pooled blocks on the GPU,
    `mem::replace`s the caller's encoder with a fresh one, commits and waits
    on the old one for the score readback, runs `compute::select_blocks` on
    the host, uploads the position list and dispatches
    `attention_decode_indexed_partial`. That is the MoE router's own
    readback shape one sublayer earlier, and it means a QSA layer above
    budget costs two command buffers where every other layer costs one; the
    GPU time lands in the same `cb1` bucket so `TURBOSPARK_PHASES=1` reads
    the same, and the extra wait is in `gpu_wait`. Below budget the block is
    the dense kernel byte for byte: the indexer's own GEMV, key copy and
    block pooling run every token from position 0 but write only to the
    indexer's buffers, which the frozen quality-gate row (perplexity 8.7224,
    both digests) and `the_synthetic_flows_arithmetic_is_frozen` are the
    proof of.
    **A MUTATION THAT MOVES THE BOUNDARY ONE BLOCK EARLY SURVIVES EVERY
    TEST, AND THAT IS AN INVARIANT, NOT A GAP.** `complete_blocks >
    idx_block_topk` mutated to `>=` reddened nothing in
    `real_forward_qwen4.rs`: at exactly `top_k` complete blocks
    `select_blocks` keeps every block, the position list is the identity,
    and the indexed kernel is bit-identical to the dense one on that list
    (`crates/gpu/tests/attention_indexed_parity.rs` pins it). The only
    observable of that mutation is the extra commit, so no output-level test
    can or should see it; a throughput test could. The two mutations that
    DO matter -- never selecting, and uploading the identity list instead of
    the mask -- redden exactly `sparse_and_forced_dense_agree_below_budget_and_diverge_above`
    and `the_indexer_projection_moves_the_output_only_above_budget` (plus
    the capacity assert on the position buffer for the second), which is
    what those two tests are for.
    **`TURBOSPARK_QSA_FORCE_DENSE=1` / `set_qsa_force_dense` is a
    DIAGNOSTIC ARM, not a mode.** It attends densely above budget as if
    every block were selected. No reference engine for this 125B checkpoint
    fits this machine, so the KL between the two arms past 2,051 tokens is
    the one quantitative instrument the sparse path has on a real install
    (`crates/bench/tests/qwen4exp_qsa_probe.rs`): garbage on a broken
    kernel, small on a working one, and never zero.

35. **`TURBOSPARK_ROUTED_BATCH=1` hard-refuses `expert_cache_slots >= 32`**
    ("batched routed prefill/verify needs slot indices below 32") in
    `families/{gemma4,gptoss,qwen}/moe_batch.rs`. Since `ALLOWED_CACHE_SLOTS`
    widened past 32, `--expert-cache-slots auto` can now resolve at or above
    it by default, so this refusal is reachable without ever passing an
    explicit slot count. Pass `--expert-cache-slots 24` (or lower) explicitly
    when combining `TURBOSPARK_ROUTED_BATCH` with `auto`-sized installs.
