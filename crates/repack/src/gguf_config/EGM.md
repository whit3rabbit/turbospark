# GGUF Config

## Overview
Rebuilds a model_io `ArchConfig` from a parsed GGUF header. It starts from the family baseline (`known_architecture(family)`) and overrides only the shape fields GGUF actually determines, leaving behavioral fields untouched because llama.cpp hardcodes them. The module is split into a metadata accessor (`meta`), per-family layer-kind mask builders (`masks`), and KV-head / linear-attention extractors (`attention`). Header values are treated as untrusted input and are range-checked before they size any allocation.

## Interface
- `arch_from_gguf` [mod.rs:arch_from_gguf]: GGUF header to `ArchConfig`, dispatching on `family_for_architecture`
- `GgufConfigError` [mod.rs:GgufConfigError]: MissingArchitecture, UnsupportedArchitecture, MissingKey, BadValue, MissingTensor
- `Meta` [meta.rs:Meta]: prefix-scoped metadata accessor (`key`, `opt`, `u64`, `i64`, `opt_i64`, `opt_f64`)
- `kv_heads` [attention.rs:kv_heads]: scalar or per-layer `head_count_kv` to (sliding, full) counts
- `linear_attention` [attention.rs:linear_attention]: `ssm.*` keys to `LinearAttentionConfig` (gated DeltaNet dims)
- `gemma4_layer_mask` [masks.rs:gemma4_layer_mask]: bool array (true = slides) to mask 0/1
- `gpt_oss_layer_mask` [masks.rs:gpt_oss_layer_mask]: derived alternating window mask from `sliding_window` and optional period
- `lfm3_layer_mask` [masks.rs:lfm3_layer_mask]: kind 6 (short conv) where per-layer KV heads are 0 and the Gemma kind is full; zero KV heads on any other kind is refused (BadValue); else the Gemma mask
- `qwen_gdn_moe_layer_mask` [masks.rs:qwen_gdn_moe_layer_mask]: every `full_attention_interval`-th layer full (1), others linear (2)
- `qwen4_exp_layer_mask`, `derive_qwen4_exp_extensions`, `derive_lfm3_extensions`, `integer_array` [mod.rs]: private per-family derivations
- Subdirectories: None

## Invariants
- `block_count` must be in 1..=4096 and the trunk depth `block_count - nextn_predict_layers` must also be in 1..=4096, so mask allocation is bounded [mod.rs:MAX_MODEL_LAYERS]
- MTP head blocks are excluded from `num_layers`; `nextn_predict_layers` >= `block_count` is refused [mod.rs:arch_from_gguf]
- Vocab comes from `token_embd.weight` dims `[hidden, vocab]`, with row width a multiple of the ggml block size [mod.rs:vocab_size]
- `tie_word_embeddings` is true only when `output.weight` is absent [mod.rs:arch_from_gguf]
- `rope.freq_base` maps to `full_rope_theta` and `freq_base_swa` to `rope_theta` (swapped vs name similarity) [mod.rs:arch_from_gguf]
- Gemma inversion: sliding=true becomes mask 0; reversing it swaps every layer's kind and rope base [masks.rs:gemma4_layer_mask]
- Array lengths (masks, `head_count_kv`, compress ratios) must equal layer count [masks.rs, attention.rs:kv_heads, mod.rs:derive_qwen4_exp_extensions]
- `ssm.inner_size` must be an exact multiple of `ssm.time_step_rank` so `value_head_dim` is not floored [attention.rs:linear_attention]
- `Meta::i64` and `opt_i64` refuse values above i64::MAX and distinguish absent from unreadable [meta.rs:Meta]
- Unported families (DeepseekV4Flash, MuseGlimmer, Qwen3Vl, Flint, UnlimitedOcr2) return `UnsupportedArchitecture` rather than an invented mask [mod.rs:arch_from_gguf]
- Family-pinned epsilons and dims are enforced: MiniMax-M2 (1e-6), Qwen2 (1e-6), LFM3 (1e-5) [mod.rs:arch_from_gguf, derive_lfm3_extensions]
- DeepSeek2 with `q_lora_rank > 0` is refused (full V2/V3 branch not executed) [mod.rs:arch_from_gguf]
- `partial_rotary_factor` is never read from `rope.dimension_count` except for Qwen4Exp and consistency checks on Spark25 and LFM3 [mod.rs:arch_from_gguf]

## Side Effects
- None. Pure function of the in-memory header; no fs, env, network, or global state [mod.rs:arch_from_gguf]

## Verification
- Run: `cargo test -p turbospark-repack gguf_config`
