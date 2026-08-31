# turbospark-model-io

Model installation layout, `manifest.json` parsing and architecture validation (`ArchConfig`), packed expert layout metadata (`PackedExpertsLayout`), resident tensor index reader (`ResidentIndex`), memory-mapped resident weight buffer (`ResidentBuffer`), SHA-256 verification (`sha256.rs`), and install receipt validation (`InstallReceipt`).

## Safety

- Contains `unsafe` code specifically restricted to memory-mapping (`mmap`) inside `resident_buffer.rs`.

## Directory & File Structure

```
crates/model-io/
+-- Cargo.toml                  # Crate manifest
+-- src/
|   +-- lib.rs                  # Library root re-exporting model-io API
|   +-- manifest/               # Decodes and validates manifest.json
|   |   +-- mod.rs              # Manifest struct & load_manifest
|   |   +-- quant.rs            # ManifestQuantization & validate_quant
|   |   \-- types.rs            # ManifestLayer, ManifestModel, etc.
|   +-- arch_config/            # ArchConfig struct and field resolution
|   |   +-- mod.rs              # Re-exports ArchConfig types
|   |   +-- config.rs           # ArchConfig struct definition
|   |   +-- family.rs           # ModelFamily enum & family resolution
|   |   \-- sub_configs.rs      # LinearAttentionConfig, RopeScalingConfig, etc.
|   +-- arch_baselines/         # Canonical baselines (Gemma 4, Qwen 3.6, DeepSeek-V4, Muse Glimmer)
|   |   +-- mod.rs              # baseline_for_family & re-exports
|   |   +-- deepseek.rs         # DeepSeek-V4 baseline
|   |   +-- gemma.rs            # Gemma 4 baseline
|   |   +-- gpt_oss.rs          # gpt-oss baseline
|   |   +-- llama.rs            # Llama/Mixtral baseline
|   |   +-- muse_glimmer.rs     # Muse Glimmer 30B dense baseline
|   |   \-- qwen.rs             # Qwen 3.6 & Qwen 3.5 baselines
|   +-- arch_validation.rs      # Structural validation rules for architecture configs
|   +-- context_policy.rs       # MaxContext, kv_bytes_for_context, largest_context_within
|   +-- context_policy_tests.rs # Unit tests for context policy resolution
|   +-- load_guard.rs           # LoadGuard tiers, GuardBudget, LoadPolicy, the AutoFit floor
|   +-- load_guard_tests.rs     # Tier ordering, distinguishability, the default pin
|   +-- expert_cache_policy.rs  # ExpertCacheSlots and how Auto resolves
|   +-- packed_experts_layout.rs# Decodes packed_experts/layout.json for streamed MoE
|   +-- resident_index.rs       # Reads tensor index entries from model_weights.bin
|   +-- resident_buffer.rs      # Zero-copy mmap wrapper (ResidentBuffer)
|   +-- steering_set.rs         # SteeringSet and LayerDirection per-layer steering vectors
|   +-- sha256.rs               # Streaming SHA-256 checksum verifier
|   +-- install_receipt.rs      # Parses and validates .gturbo install receipts
|   \-- error.rs                # ModelIoError enum definition
\-- tests/
    +-- arch_config.rs          # Architecture config resolution unit tests
    +-- install_receipt.rs      # Install receipt parsing unit tests
    +-- manifest.rs             # Manifest JSON decoding & baseline validation tests
    +-- packed_experts_layout.rs# Layout decoding unit tests
    +-- resident_index.rs       # Binary resident index parsing tests
    \-- sha256.rs               # SHA-256 verification unit tests
```

## Key Modules

- `manifest/`: Decodes and validates `manifest.json`. Its `validate_quant` accepts FOUR shapes: the INT4/INT8 affine one at group 64 with BF16 companions, the 1-bit affine one at group 128 with FP16 companions (ROADMAP's 1-bit entry), the 2-bit affine one at group 128 with FP16 companions (its ternary entry), and (ROADMAP Phase G) a `scheme: "gguf"` slot whose declared block types are all in `EXECUTABLE_GGUF_TYPES` (`q8_0`, `q4_k`, `q6_k`, Phase S's `iq3_xxs`, `iq4_nl`, `iq4_xs`, Phase M2's `q5_k`, and M5's `mxfp4`) -- the block types this port has kernels for, which is deliberately narrower than the set the repack walk can WRITE. Widening it means landing kernels; `crates/runtime` applies the same rule again to the resident index's dtype tags. A slot declares its types with `ggmlType` (the dominant one) plus an optional `ggmlTypes` ARRAY when it carries more than one, which a mixed sub-4-bit install does; every member is checked, because checking only the dominant one would pass an install on the strength of its majority and fail at a dispatch thirty layers in. Several types are in the list on weaker grounds than a full kernel set: Q5_K has a resident GEMV and nothing else (Mixtral puts it on `attn_output` alone), Q6_K has a resident GEMV, an embedding lookup and -- since Phase M2 -- a routed PHASE 2 but no phase 1, IQ3_XXS and IQ4_XS have a routed phase 1 and no phase 2, IQ4_NL the reverse. Each covers what its real file asks for, and an install that used one elsewhere passes here and fails at the dispatch, by name. **MXFP4 is narrow in the OPPOSITE direction and is the first type this list and `crates/runtime`'s disagree about**: both routed phases and no resident GEMV at all, because `gpt-oss` puts it only in `ffn_*_exps`. The two lists are TWINS AND NOT COPIES -- this one reads the manifest's per-SLOT `ggmlType` and asks whether the type has the kernels its slot needs, while `EXECUTABLE_GGUF_DTYPES` reads a RESIDENT tensor's dtype tag and asks whether that tensor can be dispatched. So `"mxfp4"` is here and tag 14 is deliberately not there, and an install with MXFP4 attention passes this gate and is stopped by that backstop. AGENTS.md Gotcha 29 states the rule; `mxfp4_is_refused_as_a_resident_tensor_though_its_experts_run` asserts both halves on one install. **Each sub-4-bit shape is checked as ONE CONJUNCTION and that is the point of their being separate predicates rather than widenings of the affine one.** Note the 2-bit one does NOT subsume the `weightBits: 2` the affine arm already allows on `routedExpert`: that is BF16 at group 64, for the DeepSeek-V4-Flash dynamic quant, and the two 2-bit shapes share nothing but their width -- `the_cross_products_of_the_two_bit_shape_are_refused` pins that a gate collapsing them into one bit list is refused. Adding `1` and `2` to the bit lists, `128` to the group sizes and `fp16` to the companion types accepts a dozen combinations no kernel implements; the three real shapes are `(4|8, bf16, 64)`, `(1, fp16, 128)` and `(2, fp16, 128)`, because a checkpoint's bit width, companion dtype and group size travel together. The FP16-versus-BF16 axis is the dangerous one: the planes are the same width, so a wrong reading passes every length check and decodes the 1-bit checkpoint's 0.0271 scales as 1.7e-16, which is why the affine error message now names the observed triple instead of saying "unsupported". Both sub-4-bit shapes are accepted on ALL FIVE slots including `routedExpert`, which has no kernel at either width: both published checkpoints are dense, so three slots describe components they do not have and fall back to the type the rest of the model uses, and refusing them is exactly how M4's dense `llama` failed to open (`crates/repack` Gotcha 8). `tests/manifest.rs` moves ONE axis per case; one case moves two, and its comment says why (it is the only shape the bit-width conjunct alone refuses, and without it that conjunct can be deleted with the file still green).
- `arch_config/`: Architecture configuration structs and field resolution. `RopeScalingConfig` (ROADMAP M5) carries YaRN's four scalars as one grouped field with a `NONE`, following `LinearAttentionConfig`; `factor: 0.0` is the inactive sentinel because 1.0 would read as "declared, and the identity".
- `arch_baselines/`: Baseline specifications, one per `ModelFamily`: Gemma 4, Qwen 3.6, DeepSeek-V4-Flash, `llama` (Mixtral's, covering the dense half too), `qwen3moe`, `gptOss`, and `qwen35` (ROADMAP's 1-bit entry). **`qwen_gdn_dense_27b()` is the first baseline whose every BEHAVIOURAL field equals another family's** -- Qwen 3.6's, read off the checkpoint's own `config.json` rather than copied from the sibling -- **while every shape field differs**: hidden 5120 against 2048, 64 layers against 40, 24 q heads over 4 kv, and NO experts at all against 256 at top-8. That is what makes `qwen3_5` a variant sharing `families/qwen/`'s flow rather than a sixth flow, and it is asserted rather than commented (`tests/arch_config.rs::qwen_gdn_dense_shares_the_moe_flows_behaviour_and_differs_in_shape`), because a coherence smoke cannot see a behavioural field at all. **It is also the first baseline named for an ARCHITECTURE rather than a checkpoint, because it serves THREE** -- `prism-ml/Bonsai-27B-mlx-1bit`, `Qwen/Qwen3.8-27B` (2026-08-14) and `prism-ml/Ternary-Bonsai-27B-mlx-2bit` (2026-08-15), whose `text_config`s agree on 33 of 35 keys. The two that differ (`eos_token_id`, and the quantization block: 1-bit group 128, 2-bit group 128 and INT4 group 64) reach no field here, which `crates/repack`'s `every_published_checkpoint_parses_to_one_baseline` asserts offline. That is what made each of the later two a checkpoint rather than a new family: no field, no kernel and no flow moved for either. The ternary file goes further than Qwen3.8 did -- it shares Bonsai's `eos_token_id` as well, so those two differ in the quantization object ALONE. Two fields read oddly and both are correct: `intermediate_size` is the DENSE FFN width where Qwen 3.6's is its shared expert's, the same collision the dense `llama` half has; and `rope_scaling` is `NONE` even though the checkpoint declares `mrope_section [11, 11, 10]`, because `RopeScalingConfig` carries YaRN's scalars and mrope is not YaRN. **That was a claim awaiting the cross-engine check and is now SETTLED, by READING the reference rather than by measuring**: `mlx_lm.models.qwen3_5` calls `initialize_rope(..., scaling_config=rope_parameters)` and the checkpoint declares `rope_type: "default"`, which maps to a plain `nn.RoPE` at `int(head_dim * partial_rotary_factor) = 64` -- exactly what this port dispatches. So both mrope fields are read by NOTHING on the text path. The two cross-engine KLs since (1.6e-5 at one bit, 1.7e-5 at two, both ~2x their shape floor) are consistent with it and were not needed to establish it. A family gets a baseline when it gets a decode flow OR a name table, never as a placeholder -- `arch_validation` compares one field by field, so an invented baseline validates installs against fiction.
- `arch_baselines/muse_glimmer.rs`: `muse_glimmer_30b()`, the SEVENTH family's baseline and the first DENSE one that is not a half of an MoE architecture string. Three fields read oddly and all three are the file's own values. `full_rope_theta` is **0.0**, which is NOT an absent-field sentinel: `config.json` publishes a per-layer `layer_rope_theta` array reading 500000 on the sliding layers and 0 on the full ones, so the thirteen full layers are NoPE. `attn_output_gate` is **false** even though the architecture HAS an attention output gate, because that field asks the narrower question of whether `q_proj` emits packed `[query; gate]` rows (Qwen's shape) and this family's gate is its own tensor. And `embedding_scaled_by_sqrt_hidden` is false because the family NORMS its embedding row where Gemma scales it. `muse_glimmer_layer_mask` builds the `[0,0,0,1]` window pattern as a function rather than a 52-entry literal, so the repack parser can compare against the same generator it validates.
- `load_guard.rs`: the memory guardrail TIERS (`off`, `relaxed`, `balanced`,
  `strict`, and a custom byte ceiling) plus `LoadPolicy`, which pairs a tier
  with the floor under an automatically-sized context window. Pure and
  portable like the two sizing policies beside it, and read by all three
  budgets rather than by one: `context_policy`, the shared reserve, and
  `catalog::recommend::fit`. **`Relaxed` is the default and IS the pre-guard
  arithmetic** -- see Gotcha 3. User-facing page: `docs/LOAD_GUARD.md`.
- `arch_validation.rs`: Structural validation of architecture configs.
- `context_policy.rs` and `expert_cache_policy.rs`: the two sizing policies,
  **moved here from `crates/runtime`** (which re-exports every name, so
  `runtime::MaxContext` and `runtime::ExpertCacheSlots` still resolve). Both
  are pure functions of an `ArchConfig` and a machine size, neither touches
  `gpu`, and this is the lowest crate that already owns `ArchConfig`. They
  moved when `crates/catalog` needed the same arithmetic BEFORE an install
  exists -- to answer "would this fit" without a twenty-minute stream -- and
  `runtime` does not build on the platforms that crate does (its `model_io`
  dependency is macOS-only). The alternative was a second copy of the KV
  formula in a crate that could not see the first; see `context_policy`'s own
  doc for why a second copy gets the sliding-window ring wrong. Their gotchas
  are `crates/runtime/CLAUDE.md` 13 and 15, which stayed there with the flows
  that consume them.
- `arch_config/sub_configs.rs`: also `VisionConfig` (ROADMAP M-V3), following `LinearAttentionConfig` with a `NONE` and an `is_active()` reading `depth > 0`. **It describes what an INSTALL carries, not what the checkpoint's `vision_config` declares**, which is why `crates/repack`'s family config parsers leave it `NONE` and a separate `parse_vision_config` exists: five published `qwen3_5`-family checkpoints declare the identical tower and one of them ships none of it (`crates/repack` Gotcha 12). Every field is an integer count or a token id on purpose -- `arch_validation` compares manifest floats with `!=` against a ~1-ULP parser, so a non-binary-fraction float here could not round-trip (Gotcha 24 in AGENTS.md). `head_dim` is DERIVED (`hidden_size / num_heads`) rather than stored, because the checkpoint declares no such key and storing one would invent a third source for a value with two.
- `packed_experts_layout.rs`: Decodes `packed_experts/layout.json` for streamed MoE layouts. **The subdirectory is a PARAMETER since ROADMAP M-V3** (`load_from`, with `load` as a thin wrapper): the vision tower's `packed_vision/layout.json` carries this exact schema -- one `LayerLayout`, `experts` = the tower's blocks, free-form roles with a dtype each -- so it decodes here rather than through a second parser. A second copy would be a second place for the `expert_stride` fallback, the per-layer stride ceiling and the missing-entry check to drift, and Gotcha 2 is already about one of those being got wrong. `expert_stride` exists at TWO levels and they mean different things: `PackedExpertsLayout::expert_stride` is the model-wide maximum (what `manifest.json` declares and what a slot is sized from), while `LayerLayout::expert_stride` is what that layer's file is actually padded to. Address or size a layer with the second, never the first -- see Gotcha 2.
- `resident_index.rs`: Reads and parses tensor metadata entries from `model_weights.bin`.
- `resident_buffer.rs`: Zero-copy `mmap` wrapper (`ResidentBuffer`) for mapped model weights.
- `steering_set.rs`: `LayerDirection`, `SteeringSet`, and `LoadedSteeringDirection` defining portable per-layer steering vectors and their validation against an `ArchConfig`.
- `sha256.rs`: Streaming SHA-256 checksum calculator for installation integrity.
- `install_receipt.rs`: Parses and verifies `.gturbo` install receipts.

## Development & Test Commands

```sh
# Run tests for turbospark-model-io
cargo test -p turbospark-model-io
```

## Crate Gotchas

1. **Resident Memory Pinning**: While clean file-backed `mmap` pages are normally unpinned in host OS memory, wrapping `ResidentBuffer` into Metal buffers via `newBufferWithBytesNoCopy` pins the mapped virtual memory range. Resident weights count directly against process physical memory footprint (`phys_footprint`).
3. **`LoadGuard::Relaxed` IS THE PRE-GUARD ARITHMETIC, AND MOVING IT
   INVALIDATES PUBLISHED MEASUREMENTS WITHOUT FAILING ANYTHING.** Every frozen
   peak in `docs/BENCHMARKS.md`, every `measured` block in `catalog`'s
   `models.json`, and the memory oracles' ceilings describe an engine
   budgeting at `HEADROOM_RESERVE_BYTES` and `CONTEXT_BUDGET_FRACTION`. A
   default resolving to anything else does not break a build or redden a
   family gate: it leaves all of those rows quietly describing a
   configuration the engine no longer opens with, the same measurement-rot
   shape AGENTS.md Gotcha 58 is about.

   Three tests hold it, in three crates because no one crate can see all
   three numbers. `relaxed_is_exactly_todays_arithmetic` pins the reserve and
   the fraction here. `context_policy_tests` passes `LoadPolicy::default()`
   in all twelve PRE-EXISTING cases with their original expected windows, so
   that file staying green is the proof for the resolver. And
   `fit_tests::the_default_guard_reproduces_the_frozen_thresholds` pins the
   tight threshold, which this crate cannot state without depending on
   `catalog`.

   Two corollaries for a new tier. State its numbers RELATIVE to `Relaxed`'s
   rather than deriving them, because nothing has measured them and inventing
   precision reads as a finding. And assert the tiers stay DISTINGUISHABLE as
   well as ordered: an ordering sweep passes unchanged when every tier has
   collapsed onto one set of numbers, which is exactly the mutation that
   found the gap.

2. **The expert stride is PER LAYER, and it was model-wide until ROADMAP Phase S.** Every install written before then is uniform across layers, which makes a uniform-stride assumption invisible: the per-layer field simply falls back to the top-level one and nothing changes. It stops being invisible on a mixed sub-4-bit install. The Phase S candidate puts IQ3_XXS + IQ4_NL experts on 29 layers and IQ4_XS + Q8_0 on the thirtieth, whose blob is 1.6x the others, so padding every layer to the maximum writes 16.2 GB where 10.3 is needed and over-reads 29 of 30 layers by that factor on every cache miss. That inverts the phase's whole -24.2% into a +35% regression, which is why this is a prerequisite rather than a tuning step. The loader refuses a layer stride ABOVE the top-level value, because a slot is allocated from the latter.
