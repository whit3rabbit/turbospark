# GGUF Checkpoint Repack

## Overview
The GGUF repack walk: takes a parsed `GgufHeader` plus a `RangeSource` and produces a `.gturbo` install (resident weights, packed routed-expert layer files, manifest quant block). It classifies tensors into resident, routed, ignored, n-gram and rope-freqs sets, transcodes the resident set to the runtime's dtypes, and streams one expert layer at a time so a multi-GB checkpoint is never fully materialized. Source-convention fixes (V-head order, Llama rotary row order) and resume-by-provenance live here too.

## Interface
- `orchestrate_gguf_checkpoint` [mod.rs:orchestrate_gguf_checkpoint]: in-memory plan of a whole repack, returns `GgufRepackOutput`; fixture use only
- `write_gguf_install_streamed` [mod.rs:write_gguf_install_streamed]: streamed install write with no resume (no provenance)
- `write_gguf_install_streamed_resumable` [mod.rs:write_gguf_install_streamed_resumable]: streamed write that adopts leftover expert layers when provenance matches
- `ResumeProvenance` [resume.rs:ResumeProvenance]: repo + 40-hex commit + file identity; `new` returns `None` for non-hex commits
- `GgufRepackError` [types.rs:GgufRepackError]: Config/Header/Name/Download/Writer wrappers plus UnsupportedFamily, UnsupportedType, ShapeMismatch, MissingTensor, Io
- `GgufRepackOutput` [types.rs:GgufRepackOutput]: arch, resident specs, layers, expert stride, ignored notes, lossy BF16 narrowing counts
- `dtype_tag_for_ggml_type` [types.rs:dtype_tag_for_ggml_type]: ggml type id to `.gturbo` resident dtype tag, `None` if unsupported
- `FUSED_GATE_FIRST` [types.rs:FUSED_GATE_FIRST]: fused `ffn_gate_up_exps` halves are gate then up
- `gguf_manifest_quant` [manifest.rs:gguf_manifest_quant]: builds the `manifest.json` `quant` object (five slots)
- `minimax_gguf_sizing` / `MiniMaxSizing` [sizing.rs:minimax_gguf_sizing]: header-only storage and KV sizing for MiniMax M2
- `verify_plan` / `zero_tail_pattern` [rope_freqs.rs:zero_tail_pattern]: pure check of the LFM3 `rope_freqs.weight` zero-tail encoding
- `qwen4exp_tensor_is_transcoded` [transcode.rs:qwen4exp_tensor_is_transcoded]: whether Qwen4Exp converts a resident tensor to BF16 (shared with catalog probe)
- Internal: `plan::classify`, `expert_stride`, `plan_one_layer`, `plan_one_layer_shape`, `layer_file_bytes` [plan.rs]; `resident_entries`, `transcode_f32` [transcode.rs]; `apply_source_convention(_bytes)` [conventions.rs]; `write_gguf_ngram_table` [ngram.rs]

## Invariants
- `DeepseekV4Flash` is refused for GGUF repack [mod.rs:write_gguf_install_streamed_resumable]
- Blobs are numbered by position: routed layer must equal `seq + num_dense_leading_layers`, else refused [mod.rs:write_gguf_install_streamed_resumable]
- Leftover `layer_NN.bin` is adopted only if provenance matches the stored record AND the file size equals the header-derived expected size; otherwise leftovers are deleted [resume.rs:prepare] [plan.rs:layer_file_bytes]
- Only a full 40-hex commit counts as a source identity; a branch name yields `None` and no resume [resume.rs:ResumeProvenance::new]
- Blocks at or above `num_layers` (multi-token-prediction head) are ignored, not mapped into the trunk [plan.rs:head_block_index]
- Routed weight shapes must exactly match arch hidden/expert dims; rank-2 only for F32 bias roles [plan.rs:routed_body_dims]
- Header-derived byte sums use checked arithmetic; overflow is a refusal, not a wrapped stride [plan.rs:expert_stride] [plan.rs:per_expert_blob_bytes]
- Expert stride is the page-aligned max per-expert blob over all routed layers, known from the header alone [plan.rs:expert_stride]
- `logical_shape` panics when a dim exceeds `u32::MAX` rather than truncate [types.rs:logical_shape]
- LFM3 `rope_freqs.weight` is required, F32, and must be exactly 1.0 over the rotated pairs and >= 1e29 over the rest, checked before any byte is written [rope_freqs.rs:verify_plan]
- MiniMax router and correction bias must be F32 and are kept FP32 raw; other families' F32 router tensors become INT8 affine, other F32 becomes BF16 [plan.rs:classify] [transcode.rs:transcode_f32]
- V-head de-tiling (Qwen GDN families) and Llama q/k rotary row unpermute are applied to resident tensors; V/K head counts must divide evenly [conventions.rs:apply_source_convention] [conventions.rs:needs_rotary_unpermute]
- Qwen4Exp QSA indexer parts must be BF16 with exact shapes and are concatenated into one `index_qk_proj` entry [transcode.rs:resident_entries]
- Manifest slots for absent components fall back to the attention type; MiniMax router is raw FP32 [manifest.rs:gguf_manifest_quant]
- Qwen4Exp PLE table must be IQ4_NL with row/shard/hash-array lengths consistent, and the written header is re-read through the model-io loader [ngram.rs:write_gguf_ngram_table]

## Side Effects
- Writes the install dir via `StreamingGturboWriter`: `packed_experts/layer_NN.bin`, `layout.json`, resident `model_weights.bin`, manifest [mod.rs:write_gguf_install_streamed_resumable]
- Network/ranged reads of tensor bytes through `RangeSource` [types.rs:read_tensor]
- Writes and removes `.resume-provenance.json`; deletes stale `packed_experts/layer_*.bin` on mismatch [resume.rs:prepare] [resume.rs:finish]
- Writes n-gram table dir (rows blob, then atomic temp+rename header with `sync_all`), removing a previous header first [ngram.rs:write_gguf_ngram_table]
- Emits progress strings including BF16 precision-loss WARNINGs via the caller's callback [mod.rs:write_gguf_install_streamed_resumable]
- No global state, no env vars, no unsafe, no Metal

## Verification
- Run: `cargo test -p turbospark-repack`
