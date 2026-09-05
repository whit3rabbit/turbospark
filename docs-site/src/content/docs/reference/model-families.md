---
title: "Model Families and Architecture Resolution"
description: "Lookup table for 'does my checkpoint run, and through what': the nine ModelFamily variants, the GGUF and Hugging Face architecture strings that resolve to each, the per-family baseline and decode flow, and the validation and refusal rules."
diataxisType: "reference"
---

<!-- authored from source: model-io/src/arch_config/{family,config}.rs, model-io/src/arch_baselines/, model-io/src/arch_validation.rs, repack/src/arch_registry.rs, runtime/src/families/ -->

A model family is the discriminator that selects the tensor-name contract,
the layer graph shape, and the per-family kernel behaviour. It is the
`ModelFamily` enum in `crates/model-io/src/arch_config/family.rs`, recorded
in every install's `manifest.json` as `arch.family`, and read back at load.
An absent `arch.family` means Gemma 4, the format's original architecture.

The decode flow is chosen by `ArchConfig.family`, never by tensor naming.
This is a contract, not a convention: families share tensor names (Gemma 4
and Qwen 3.6 both carry `language_model.model.embed_tokens.weight`), so a
naming probe cannot tell them apart, and picking a neighbour's flow yields
fluent wrong output rather than an error.

## The three naming tables

A family answers to three different strings, and all three tables are
separate on purpose (`crates/repack/src/arch_registry.rs` module doc). The
same checkpoint can report all three:

| Table | Key | Example (Qwen 3.6) | Lives in |
|---|---|---|---|
| Wire string | `manifest.json -> arch.family` | `qwen36` | `ModelFamily::as_str` / `ModelFamily::parse` |
| GGUF | `general.architecture` | `qwen35moe` | `SUPPORTED_GGUF` in `arch_registry.rs` |
| HF | `config.json -> model_type` (root and `text_config`) | `qwen3_5_moe` | `SUPPORTED_HF` in `arch_registry.rs` |

The wire strings are an on-disk format constant. Two of them no longer match
their variant's Rust name (`QwenGdnMoe` writes `qwen36`, `QwenGdnDense`
writes `qwen35`) because renaming would invalidate every `.gturbo` install
ever written. A new family is free to pick a matching string.

## Family index

Nine `ModelFamily` variants plus the synthetic fixture flow. "Runs" means a
decode flow exists in `crates/runtime/src/families/` and real checkpoints
have been installed and gated.

| Family (variant) | Wire string | GGUF arch | HF model_type | Decode flow | State | Runs |
|---|---|---|---|---|---|---|
| Gemma 4 (`Gemma4`) | `gemma4` | `gemma4` | `gemma4`, `gemma4_text` | `families/gemma4/` | `RealGemmaState` | yes |
| Qwen 3.6 MoE (`QwenGdnMoe`) | `qwen36` | `qwen35moe` | `qwen3_5_moe`, `qwen3_5_moe_text` | `families/qwen/` | `RealQwenState` | yes |
| Qwen 3.5 dense (`QwenGdnDense`) | `qwen35` | `qwen35` | `qwen3_5`, `qwen3_5_text` | `families/qwen/` (shared) | `RealQwenState` | yes |
| Llama (`Llama`) | `llama` | `llama` | none (GGUF only) | `families/llama/` | `RealLlamaState` | yes (both halves) |
| Qwen3 MoE (`Qwen3Moe`) | `qwen3moe` | `qwen3moe` | none (GGUF only) | `families/llama/` (shared) | `RealLlamaState` | yes |
| gpt-oss (`GptOss`) | `gptOss` | `gpt-oss` | none (GGUF only) | `families/gptoss/` | `RealGptOssState` | yes |
| Muse Glimmer (`MuseGlimmer`) | `museGlimmer` | none | `muse_glimmer`, `muse_glimmer_text` | `families/museglimmer/` | `RealMuseState` | yes |
| Qwen4 Exp (`Qwen4Exp`) | `qwen4exp` | none (refused) | `qwen4_exp`, `qwen4_exp_text` | `families/qwen4/` | `RealQwen4State` | yes |
| DeepSeek-V4-Flash (`DeepseekV4Flash`) | `deepseekV4Flash` | none | none | none | none | refused |
| Synthetic (not a variant) | n/a | n/a | n/a | `families/synthetic/` | n/a | fixtures |

Layer-mask legend (`ArchConfig.full_attention_layer_mask`,
`crates/model-io/src/arch_config/config.rs`): 0 = sliding-window attention,
1 = full attention, 2 = gated-DeltaNet linear attention, 3 = compressed
sparse attention (CSA), 4 = heavily compressed attention (HCA). Only mask
values 0, 1 and 2 have kernels; 3 and 4 are refused (see
"Recognized-but-unported" below).

## Family entries

### Gemma 4 (`Gemma4`)

- Wire `gemma4`; GGUF `gemma4`; HF `gemma4`, `gemma4_text`.
- Baseline `gemma4_26b_a4b()` (`arch_baselines/gemma.rs`): 30 layers (every
  6th from index 5 full attention, the rest sliding at window 1024), hidden
  2816, 16 q heads over 8 SWA kv heads at head_dim 256 and 2 full-attention
  kv heads at full_head_dim 512, 128 routed experts at top-8,
  `intermediate_size` 2112 is the shared-expert width (3x the 704 routed
  width), logit softcap 30.0, tied embeddings, vocab 262,144.
- Flow: `families/gemma4/` (`RealGemmaState`). Verbatim checkpoint weight
  names, per-head q/k norms, shared-plus-routed MoE.
- Checkpoints known to run: the catalog's Gemma 4 26B-A4B, as MLX
  safetensors and as GGUF (`gemma4`).

### Qwen 3.6 MoE (`QwenGdnMoe`)

- Wire `qwen36` (frozen, version-shaped); GGUF `qwen35moe` (llama.cpp named
  the converter after the 3.5 series it shares a graph with); HF
  `qwen3_5_moe`, `qwen3_5_moe_text`.
- Baseline `qwen_gdn_moe_35b_a3b()` (`arch_baselines/qwen.rs`): 40 layers
  (30 gated-DeltaNet linear mask-2, every 4th layer full attention mask-1),
  hidden 2048, 16 q heads over 2 kv heads at head_dim 256, 256 routed
  experts at top-8 plus a sigmoid-gated shared expert, `partial_rotary_factor`
  0.25 at theta 1e7, `attn_output_gate` true (q_proj emits packed
  [query; gate] rows), no sandwich norms, no softcap, vocab 248,320.
- Flow: `families/qwen/` (`RealQwenState`). Gated DeltaNet on mask-2 layers,
  gated full attention on mask-1, one post-attention norm feeding the FFN.
- Checkpoints known to run: Qwen 3.6 35B-A3B (MLX safetensors and GGUF
  `qwen35moe`), `ornith-ai/Ornith-1.5-35B-A3B` (MLX).

### Qwen 3.5 dense (`QwenGdnDense`)

- Wire `qwen35` (frozen); GGUF `qwen35`; HF `qwen3_5`, `qwen3_5_text`.
- Baseline `qwen_gdn_dense_27b()`: 64 layers (48 linear, 16 full), hidden
  5120, 24 q heads over 4 kv heads, dense SwiGLU FFN (`num_experts` 0). Every
  behavioural field equals `qwen_gdn_moe_35b_a3b()`'s; every shape field
  differs. That is what licenses sharing the flow.
- Flow: `families/qwen/` (shared with `QwenGdnMoe`). The FFN is the only
  fork and it is read off `num_experts` (`RealQwenState::dense`), never off
  tensor naming.
- Checkpoints known to run: `prism-ml/Bonsai-27B-mlx-1bit`,
  `Qwen/Qwen3.8-27B` / `mlx-community/Qwen3.8-27B-4bit`,
  `prism-ml/Ternary-Bonsai-27B-mlx-2bit` (one architecture at three
  quantizations), and `ornith-ai/Ornith-1.5-9B-GGUF`, the first published
  `qwen35` GGUF.

### Llama (`Llama`) - dense and MoE halves under one string

- Wire `llama`; GGUF `llama`; no HF row (this family arrives as GGUF).
- Baseline `mixtral_8x7b()`: 32 full-attention GQA layers (32 q over 8 kv),
  8 routed experts at top-2, no shared expert, no softcap, no sliding
  window, untied head. The baseline is Mixtral's because every behavioural
  field is shared with the dense half and only shape fields differ.
- The shared-architecture trap: one `general.architecture = "llama"` string
  covers dense Llama 2/3.x and Mistral AND the Mixtral MoEs. Nothing in the
  architecture string says which half a file is; only `expert_count` does.
  `RealLlamaState` splits on `arch.num_experts == 0` (`families/llama/state.rs`),
  and a dense layer swaps the router and routed experts for one gated FFN
  (`families/llama/dense.rs`).
- Flow: `families/llama/` (`RealLlamaState`). Defined by its absences: plain
  GQA, raw residual add, one post-attention norm, no per-head norms, no
  output gate, full-head NeoX RoPE.
- Checkpoints known to run: `TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF` (MoE
  half), `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` and Mistral 7B (dense
  half).

### Qwen3 MoE (`Qwen3Moe`)

- Wire `qwen3moe`; GGUF `qwen3moe`; no HF row.
- Baseline `qwen3_30b_a3b()`: 48 full-attention layers, 32 q heads over 4 kv
  at head_dim 128 (head_dim is 128 while `hidden_size / num_heads` is 64),
  128 routed experts at top-8, no shared expert, `moe_intermediate_size` 768
  (one expert is 2.5 MiB at Q4_K, so 16 slots over 48 layers pin 1.90 GiB).
  Every number was read off the published GGUF header.
- Flow: `families/llama/` (shared). Differs from the `llama` family in
  exactly two places, both carried by `RealLlamaState` and keyed on
  `ArchConfig.family`: it norms q and k per head before RoPE, and its RMS
  epsilon is 1e-6 where `llama`'s is 1e-5.
- Checkpoints known to run: `Qwen/Qwen3-30B-A3B-GGUF` (Q4_K_M).

### gpt-oss (`GptOss`)

- Wire `gptOss`; GGUF `gpt-oss`; no HF row.
- Baseline `gpt_oss_20b()`: 24 layers, hidden 2880, 64 q heads over 8 kv at
  head_dim 64, 32 routed experts at top-4, alternating 128-token sliding
  window with EVEN layers sliding, vocab 201,088.
- Flow: `families/gptoss/` (`RealGptOssState`). The only flow here that is
  not a variation on an existing graph: per-projection biases on q/k/v/output
  and on the router and every routed expert, attention sinks (one learned
  logit per q head added to the softmax denominator), YaRN rope through a
  precomputed table, and a clamped SwiGLU (swish with alpha times
  `(up + 1)`).
- Checkpoints known to run: the published 12.1 GB `gpt-oss-20b` GGUF.

### Muse Glimmer (`MuseGlimmer`)

- Wire `museGlimmer`; no GGUF row; HF `muse_glimmer`, `muse_glimmer_text`.
  The `architectures` class-name scheme (`...ForConditionalGeneration`) is
  deliberately not consulted.
- Baseline `muse_glimmer_30b()` (`arch_baselines/muse_glimmer.rs`): 52 dense
  layers, hidden 6656, dense FFN width 19,968 (no experts), 32 q heads over
  2 kv at head_dim 128, alternating three-sliding/one-full window at 2048
  (`[0,0,0,1]` x 13, from `layer_types`), logit softcap 20.0, rope theta
  500,000 with `full_rope_theta` 0.0 (NoPE on the thirteen full layers),
  vocab 202,048.
- Flow: `families/museglimmer/` (`RealMuseState`). Ten differences inside
  the layer, including centered per-layer norms (`x * (1 + w)`) with a plain
  final norm, two RMS epsilons (1e-5 and 1e-8), an attention output gate as
  its own tensor (`self_attn.gate_proj`, so `attn_output_gate` is false), and
  a normed embedding row.
- Checkpoints known to run: `mlx-community/Muse-Glimmer-30B-4bit`. A
  vision-language model; this port ingests the text tower only.

### Qwen4 Exp (`Qwen4Exp`)

- Wire `qwen4exp`; no GGUF row (the GGUF walk refuses it: the MLX
  safetensors form is what this port ingests); HF `qwen4_exp`,
  `qwen4_exp_text`.
- Baseline `qwen4_exp_125b_a6b()` (`arch_baselines/qwen.rs`): 48 layers on a
  three-linear/one-full pattern (36 gated-DeltaNet mask-2, 12 full mask-1),
  hidden 2560, 512 routed experts at top-10 plus a gated shared expert (both
  widths 640), 24 q heads over 2 kv at head_dim 256, theta 1e7,
  `partial_rotary_factor` 0.25, `attn_output_gate` true, vocab 248,320.
- Three fields carry the whole reason this is not `QwenGdnMoe` at different
  shapes: `hyper_connections` active at `mult: 4` (the residual stream is
  `4 * hidden_size` wide and every residual add is a gated read/inject
  pair), `linear_attention.output_gate_sigmoid` true (every earlier family
  declares silu), and `ple` active (a hashed n-gram per-layer embedding at
  layer index 1, 30.8% of the checkpoint). Its full-attention layers also
  carry a query-sparse indexer selecting blocks of compressed keys.
- Flow: `families/qwen4/` (`RealQwen4State`); no final `model.norm`, the
  closing mixer carries it.
- Checkpoints known to run: `pipenetwork/Qwen3.8-Flash-Next-MLX-4bit` (512
  experts) and `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit` (pruned to 288;
  differs in `num_experts` and nothing else). Vision-language; text tower
  only.

### DeepSeek-V4-Flash (`DeepseekV4Flash`) - recognized, refused

- Wire `deepseekV4Flash`; no GGUF row; no HF row.
- Baseline `deepseek_v4_flash_284b_a13b()` exists and is registered in
  `known_architecture`, but the family has no decode flow:
  `RealForwardRunner::open` refuses it with "the DeepSeek-V4-Flash family
  has no decode flow yet" (`crates/runtime/src/real_forward_open.rs`), and
  the GGUF walk refuses it with `UnsupportedFamily` before streaming.
- The layer-level backstop names the missing kernels: `validate_arch_config`
  (`crates/runtime/src/real_forward_init.rs`) rejects any layer mask above 2
  with "compressed (DeepSeek CSA/HCA) attention layers are not supported
  yet". The baseline's mask is layers 0-1 sliding, then CSA (3) on even and
  HCA (4) on odd layers, 43 all-MoE layers with shared-KV MQA at 256 experts
  top-6.

### Synthetic (fixture flow, not a `ModelFamily` variant)

- `families/synthetic/` decodes short-name synthetic installs
  (`layer0.q_proj`) built by the `crates/repack` synthetic generators. Its
  weights are deterministic pseudo-random numbers: structurally valid output,
  semantically gibberish. It exists so tests need no multi-GB checkpoint.

## Selection and validation rules

### How a checkpoint resolves to a family

1. GGUF intake reads `general.architecture` and looks it up in
   `SUPPORTED_GGUF` (`gguf_arch_support`). Exact equality, never a prefix
   match: `qwen35` under a `starts_with` would resolve every dense file to
   `QwenGdnMoe`, a baseline with 256 experts and a decode flow with a router
   in it, i.e. fluent wrong output rather than an error.
2. HF intake reads `model_type` from the config root and then from
   `text_config` (`config_json_family`; multimodal wrappers carry both and
   disagree in suffix only). `refuse_foreign_config` guards each
   family-specific parser against being handed another family's file, with a
   message that quotes the raw `model_type`: "model_type {raw} resolves to
   the {found} family, not {expected}".
3. At load, the install's `manifest.json -> arch.family` is parsed by
   `ModelFamily::parse`; an unknown string is an error, an absent one means
   Gemma 4.

### Baseline comparison

Every install is validated field by field against its family's baseline:
`known_architecture(family)` (`crates/model-io/src/arch_baselines/mod.rs`)
returns the canonical `ArchConfig`, and `validate_arch`
(`crates/model-io/src/arch_validation.rs`) compares each manifest field
against it, failing with `ModelError::ArchMismatch` naming the field, the
expected and the actual value. A family gets a baseline when it gets a
decode flow or a name table, never as a placeholder: `arch_validation`
compares field by field, so an invented baseline validates installs against
fiction.

### Family-extension fallback rule

Optional manifest fields fall back as follows (`arch_validation.rs`):

| Field class | Absent means | Fields |
|---|---|---|
| Family extensions | the GEMMA baseline's value | `attnOutputGate`, `attentionScale`, `embeddingScaledBySqrtHidden`, `routerScaled`, `ffnSandwichNorms`, `sharedExpertGated`, `ropeNeoxSubdim` |
| Linear attention | 0 (none) | `linearNumKHeads`, `linearNumVHeads`, `linearKeyHeadDim`, `linearValueHeadDim`, `linearConvKernelSize`; `linearOutputGateSigmoid` absent means false (silu) |
| Compressed attention | 0 | `ca*` fields |
| Hyper-connections | 0 | `hcMult`, `hcSinkhornIters`, `hcEps`, `hcLowrank` |
| PLE table | `PleConfig::NONE`'s zeros | `ple*` fields, `pleLayerIds` compared as a whole list |
| RoPE scaling | the "no scaling" value | `ropeScalingFactor` 0.0, `ropeScalingOriginalContext` 0, betas 0.0 |
| Vision tower | `VisionConfig::NONE`'s zeros | `vision*` fields |

The Gemma fallback for family extensions is why the `.gturbo` writer writes
every extension field unconditionally: a Qwen install that omits one can
never load, because the omitted field is compared against Gemma's value
whatever family the manifest claims.

Float fields are compared with `!=` on `f64` and serde_json's default
parser is only accurate to about 1 ULP, so float values that are not binary
fractions can fail to round-trip (the real families' 1.0, 0.0625 and 2^-4.5
are fine).

### Refusal behaviour

`describe_gguf_architecture` produces one of three messages:

- Supported: "GGUF architecture {arch} is {family}".
- Recognized but no decode flow: names the missing work in one clause and
  points at `docs/NEW_MODEL.md`.
- Unknown: "not in this port's registry (crates/repack/src/arch_registry.rs)".

`Supported` in the registry answers "which family is this string";
`RealForwardRunner::open` answers "can it run". They are separate gates.

## Recognized-but-unported architectures

Planned GGUF rows carry a one-clause `needs` and a witness URL each; every
key was read off a real published file, and a network test re-reads each
witness. A `Planned` row is recognition and nothing more: callers still
refuse the checkpoint.

| GGUF arch | Needs |
|---|---|
| `llama4` | an expert granularity this engine can stream (16 experts of 77.8 MiB is 58.4 GiB of slot cache at 16 slots), plus RoPE frequency scaling and its interleaved chunked-attention layer graph |
| `deepseek2` | multi-head latent attention kernels (layer mask 3-4, unported) |
| `phi3` | dense-FFN GPU path and SuScaled (longrope) RoPE |

Planned architectures deliberately get no `ModelFamily` variant:
`known_architecture` is an exhaustive match returning a real baseline, so a
placeholder variant would validate installs against invented numbers. A
variant appears when a baseline and a flow do, not before. That is why this
table is keyed by string.

`DeepseekV4Flash` is the one case that went the other way: it has a variant
and a baseline but no flow, so it is refused by name at open and at the GGUF
walk, with the missing kernels (CSA/HCA, layer mask 3-4) named one gate
earlier.

## Cross-links

- [Add a model family](/guides/add-model-family/): the end-to-end bring-up
  checklist (what to map, what to specialize, what to measure).
- [The .gturbo format](/reference/gturbo-format/): what an install's
  `manifest.json` records, including `arch.family` and the quant slots.
- `docs/NEW_MODEL.md` in the repository: the same checklist at source.
