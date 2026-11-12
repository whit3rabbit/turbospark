# GGUF Tensor Name Mapping

## Overview
Maps GGUF tensor names onto the canonical HF-style names the repack pipeline uses, so classification, routed markers, and resident ordering are shared between safetensors and GGUF intakes. Dispatch is by `ModelFamily` in `mod.rs`; each per-family file supplies the `blk.<n>.` suffix table (and a few top-level rows). Every row was read off a real file or install, and unknown names are errors, never skips.

## Interface
- `map_gguf_name` [mod.rs:map_gguf_name]: `(name, ModelFamily) -> Result<GgufMapping, GgufNameError>`
- `GgufMapping` [mod.rs:GgufMapping]: Resident(name), Routed{layer, role}, RoutedFusedGateUp{layer}, Ignored{reason}, ZeroTailRopeFreqs
- `GgufNameError` [mod.rs:GgufNameError]: Unmapped, BadLayerIndex, PreMergeExperts, UnsupportedRopeScaling and related
- `gguf_architecture` [mod.rs:gguf_architecture]: family to GGUF `general.architecture` string (e.g. Qwen 3.6 is `qwen35moe`)
- `family_for_architecture` [mod.rs:family_for_architecture]: inverse lookup
- `map_gemma4_layer` [gemma4.rs:map_gemma4_layer]; `map_llama_layer`, `map_qwen2_layer`, `is_pre_merge_expert` [llama.rs]
- `map_qwen4_exp_layer`, `map_qwen4_exp_top_level`, `map_qwen_gdn_moe_layer`, `map_qwen_gdn_dense_layer`, `map_qwen3moe_layer`, `map_qwen3_dense_layer` [qwen.rs]
- `map_gpt_oss_layer` [gpt_oss.rs]; `map_deepseek2_layer` [deepseek2.rs]; `map_lfm3_layer`, `map_lfm3_top_level` [lfm3.rs]; `map_spark_layer` [spark.rs]; `map_layer` [minimax.rs]
- `layer_prefix` [mod.rs:layer_prefix]: canonical `language_model.model.layers.{n}.` (crate-private)
- Subdirectories: none

## Invariants
- An unrecognized name is `Unmapped`, never downgraded to a skip, since a missed routed marker silently becomes a resident tensor [mod.rs:GgufNameError, map_gguf_name]
- `rope_freqs.weight` is refused for every family except `Lfm3`, where the walk must verify its zero-tail pattern before it is dropped [mod.rs:map_gguf_name]
- Pre-merge Mixtral per-expert tensors are refused by name for `Llama` [mod.rs:map_gguf_name, llama.rs:is_pre_merge_expert]
- `DeepseekV4Flash`, `MuseGlimmer`, `Qwen3Vl`, `Flint`, `UnlimitedOcr2` have no GGUF table and always return `Unmapped`; `Qwen4Exp` maps only its own table, never Qwen 3.6's [mod.rs:map_gguf_name]
- Gemma 4 fuses routed gate/up (`RoutedFusedGateUp`); Qwen does not [mod.rs:GgufMapping]
- `blk.<n>` that does not parse yields `BadLayerIndex` [mod.rs:map_gguf_name]
- Qwen `A_log` and `dt_bias` take no `.weight` suffix [qwen.rs:map_qwen_gdn_moe_layer]

## Side Effects
- None (pure string mapping)

## Verification
- Run: `cargo test -p turbospark-repack`
