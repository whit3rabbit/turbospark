# Synthetic Qwen4 Decode Fixture

## Overview
Builds a tiny, decode-capable `qwen4_exp` install for runtime tests. Every layer carries hyper-connection, GDN-or-attention, and MoE weights, plus a fully populated PLE layer with a real quantized n-gram table. Weights are untrained and deterministic, so tests can prove the flow runs, yields finite logits, and reads the tensors it claims to, but not output quality. Installs are written through the real repack pipeline (`write_gemma4_install` or the streamed qwen4_exp writer). `arch.rs` holds shape constants and the `ArchConfig`, `tensors.rs` generates layer tensors, `ngram.rs` generates PLE and n-gram tensors.

## Interface
- `build_synthetic_qwen4_exp_decode_install` [mod.rs:build_synthetic_qwen4_exp_decode_install]: default fixture, pre-packed INT8 router, real indexer budget
- `build_synthetic_qwen4_exp_decode_install_with_indexer_budget` [mod.rs:build_synthetic_qwen4_exp_decode_install_with_indexer_budget]: caller-chosen QSA `index_budget`
- `build_synthetic_qwen4_exp_decode_install_raw_router` [mod.rs:build_synthetic_qwen4_exp_decode_install_raw_router]: router and shared-expert gate shipped as raw BF16 (exercises router quantization)
- `build_synthetic_qwen4_exp_decode_install_with_vision` [mod.rs:build_synthetic_qwen4_exp_decode_install_with_vision]: attaches the toy vision tower (INT4); `streamed` selects the writer
- `tiny_qwen4_exp_decode_arch`, `tiny_qwen4_exp_decode_arch_with_indexer_budget` [arch.rs:tiny_qwen4_exp_decode_arch_with_indexer_budget]: the `ArchConfig`
- Shape constants `HIDDEN`, `HC_COUNT`, `HEAD_DIM`, `NUM_HEADS`, `NUM_KV_HEADS`, `NUM_EXPERTS`, `TOP_K`, `NUM_LAYERS` [arch.rs]
- `NGRAM_EOS_TOKEN_ID`, `PLE_LAYER` [ngram.rs]: re-exported for tests
- Layer mask `[2, 2, 1, 2]` (GDN, GDN+PLE, attention, GDN) [arch.rs:layer_mask]
- `build_tensors` [tensors.rs:build_tensors] (private): full tensor list plus `bits_overrides`
- `ple_layer_tensors` [ngram.rs:ple_layer_tensors] (private): PLE key/value/conv/norm tensors plus the n-gram table

## Invariants
- `indexer_budget` must be positive and a whole multiple of `IDX_COMPRESS` (4); `index_top_k = budget / IDX_COMPRESS` [arch.rs:tiny_qwen4_exp_decode_arch_with_indexer_budget]
- The default budget is 2048, the real checkpoint's, so default-fixture tests stay below budget and byte-identical [arch.rs:IDX_BUDGET]
- `attention_scale` is the binary fraction 0.125 because manifest floats are compared with `!=` [arch.rs:tiny_qwen4_exp_decode_arch_with_indexer_budget]
- Weights are deterministic (seeded per layer: `1000 * (l + 1)` plus offsets) [tensors.rs:build_tensors]
- Default router and shared-expert gate are INT8 triples with 8-bit overrides; raw-router mode ships both as BF16 with no override, matching the real REAP-288 checkpoint [tensors.rs:moe_tensors]
- Linear-attention `A_log` and `dt_bias` have no `.weight` suffix; the GDN gated norm is plain (near 1.0) while hyper-connection, attention, indexer, and PLE norms are centered (near 0.0) [tensors.rs:gdn_layer_tensors, tensors.rs:hyper_connection_tensors]
- N-gram rows are quantized INT4 affine at group 32 (low nibble first), the inverse of `compute::dequant_ngram_row`, and the multipliers come from `model_io::build_layer_multipliers` [ngram.rs:quantize_ngram_row, ngram.rs:ngram_tensors]
- `ple_embed_dim` equals `NGRAM_HEADS * NGRAM_HEAD_DIM` (128, equal to `HIDDEN`) and the PLE layer id is `PLE_LAYER + 1` [arch.rs:tiny_qwen4_exp_decode_arch_with_indexer_budget]
- `IDX_HEAD_DIM` must be at least `rotary_dim` (`HEAD_DIM / 4`) [arch.rs:IDX_HEAD_DIM]

## Side Effects
- Filesystem: writes a full install into the caller's `dir` via `write_gemma4_install` or `write_qwen4_exp_install_streamed` [mod.rs:build_synthetic_qwen4_exp_decode_install_inner, mod.rs:build_synthetic_qwen4_exp_decode_install_with_vision]
- Memory: assembles the whole synthetic safetensors blob in memory [mod.rs:build_synthetic_qwen4_exp_decode_install_inner]

## Verification
- Run: `cargo test -p turbospark-repack synthetic_qwen`
- Consumed by the decode flow tests in `crates/runtime` (`real_forward_qwen4.rs`): `cargo test -p turbospark-runtime qwen4`
