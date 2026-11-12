# arch_baselines (model-io)

## Overview
Compile-time `ArchConfig` baselines, one constructor per supported model family (`gemma4_12b_dense` is public but has no `ModelFamily` variant, so the registry never returns it). An installed manifest's `arch` block is checked field-by-field against the baseline resolved for its family. Private module of `turbospark-model-io`, re-exported from the crate root.

## Interface
- `known_architecture(ModelFamily) -> ArchConfig`, `all_known_architectures()` [arch_baselines/mod.rs:known_architecture]
- Constructors: `deepseek_v4_flash_284b_a13b` [deepseek.rs], `deepseek_v2_lite_16b`, `deepseek_vl2_tiny` [deepseek2.rs], `flint_40b_a2_5b`, `flint_layer_mask` [flint.rs], `gemma4_26b_a4b`, `gemma4_12b_dense` [gemma.rs], `gpt_oss_20b` [gpt_oss.rs], `lfm3_1_7b`, `lfm3_layer_mask` [lfm3.rs], `mixtral_8x7b` [llama.rs], `minimax_m2` [minimax.rs], `muse_glimmer_30b`, `muse_glimmer_layer_mask` [muse_glimmer.rs], `qwen_gdn_dense_27b`, `qwen_gdn_moe_35b_a3b`, `qwen3_30b_a3b`, `qwen3_4b`, `qwen2_5_7b`, `qwen4_exp_125b_a6b`, `qwen3_vl_4b` [qwen.rs], `spark_x25_4b`, `spark_layer_mask` [spark.rs], `unlimited_ocr2_12l` [unlimited_ocr2.rs]

## Invariants
- `known_architecture` matches every `ModelFamily` variant exhaustively, so a new family fails to compile until a baseline exists [arch_baselines/mod.rs:known_architecture]
- `all_known_architectures` lists baselines including the extra `deepseek_vl2_tiny`; keep it in sync with the family registry [arch_baselines/mod.rs:all_known_architectures]

## Side Effects
- None (pure constructors)

## Verification
- Run: `cargo test -p turbospark-model-io`
