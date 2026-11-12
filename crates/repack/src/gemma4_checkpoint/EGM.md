# Gemma 4 Checkpoint Repack (safetensors walk)

## Overview
This module walks safetensors (MLX 4-bit and related) checkpoints and repacks them into a TurboSpark install: resident weights, per-layer routed-expert blobs, optional MTP/DFlash2/vision/ngram/hadamard sidecars, and a manifest. Despite the name it serves Gemma 4, Qwen, Flint, DeepSeek-VL2, LFM3 and others, selected by family. `mod.rs` holds the per-family install writers and staged atomic publish, `orchestrate.rs` the non-streamed classify-and-read path, and the `vision_*.rs` files are per-tower ingest readers dispatched from `vision.rs`.

## Interface
- `write_gemma4_install_streamed` [mod.rs:write_gemma4_install_streamed]: streamed install writer; stages in a sibling dir, then publishes
- `write_gemma4_install_streamed_in_place` (private) [mod.rs:write_gemma4_install_streamed_in_place]: classify, read resident set, ingest MTP/DFlash/vision/ngram/hadamard, write per-layer expert blobs, then manifest
- `write_gemma4_install` [mod.rs:write_gemma4_install]: non-streamed orchestrate and write of a full install
- Per-family streamed writers [mod.rs:write_qwen_gdn_moe_install_streamed]: `write_qwen_gdn_moe_install_streamed`, `write_qwen_gdn_dense_install_streamed`, `write_qwen4_exp_install_streamed`, `write_qwen2_dense_install_streamed`, `write_flint_install_streamed`, `write_deepseek_vl2_install_streamed`, `write_lfm3_install_streamed`, `write_qwen3_vl_install_streamed`, `write_muse_glimmer_install_streamed`
- `graft_qwen_gdn_dense_mtp_head` [mod.rs:graft_qwen_gdn_dense_mtp_head]: graft an MTP head from another repository onto an install
- `write_vision_sidecar` [mod.rs:write_vision_sidecar]: vision-only sidecar write from an HF-native repo
- `sibling_work_dir` / `publish_staged_install` (both private) [mod.rs:publish_staged_install]: staging dir creation and swap-with-backup publish
- `orchestrate_gemma4_checkpoint` / `orchestrate_gemma4_checkpoint_sharded` [orchestrate.rs:orchestrate_gemma4_checkpoint_sharded]: non-streamed walk returning `Gemma4RepackOutput` (resident entries, layers, expert stride, lossy narrowing, vision, ngram plan)
- `classify_all` / `ClassifiedNames` [orchestrate.rs:classify_all]: partition all shard tensor names by role
- `read_resident_entries_from_shards` / `ResidentRead` [orchestrate.rs:read_resident_entries_from_shards]: ordered resident entries plus lossy narrowing report
- `Gemma4Bucket` / `classify_for_family` / `classify_gemma4` [classify.rs:classify_for_family]: family-aware tensor name to bucket (resident, routed expert, MTP, DFlash, vision, ngram, hadamard sign, excluded, unknown)
- `canonicalize_vision_header` / `canonicalize_qwen2_header` / `canonicalize_deepseek_vl2_header` / `canonicalize_flint_header` [classify.rs]: rewrite foreign header namespaces onto the install namespace
- `Gemma4Error`, `parse_gemma4_config`, `Gemma4Quant`, `QuantMode`, `parse_gemma4_quantization` [config.rs:parse_gemma4_config]: error type, config to `ArchConfig`, per-tensor quant parsing (affine and mxfp4)
- `Gemma4Shards` [shards.rs:Gemma4Shards]: multi-shard view with one merged tensor registry; `GTURBO_PAGE_BYTES` (16 KiB) [shards.rs:GTURBO_PAGE_BYTES]
- `expert_stride_from_headers` / `plan_one_expert_layer` / `routed_expert_bits` (private) [expert_blobs.rs:plan_one_expert_layer]: per-layer expert blobs and the model-wide page-rounded stride
- `narrow_raw_to_bf16`, `convert_raw_to_fp16`, `quantize_gating_matrix_int8`, `pass_through_packed` [narrow.rs:narrow_raw_to_bf16]: per-tensor dtype handling for resident tensors
- `gemma4_manifest_quant` / `manifest_quant` / `manifest_quant_for` [manifest_quant.rs:manifest_quant_for]: manifest quant object per family
- `reblock_to_gguf_mxfp4` / `dequant_to_bf16` [mlx_mxfp4.rs:reblock_to_gguf_mxfp4]: exact re-encoding of MLX mxfp4 tensors
- `read_mtp_entries` [mtp.rs:read_mtp_entries], `read_dflash_entries` [dflash.rs:read_dflash_entries]: MTP head and DFlash2 drafter ingest (rank-2 to INT4, norms to BF16, codebooks raw)
- `parse_prism_hadamard` / `read_hadamard` [hadamard.rs:read_hadamard]: prism Hadamard contract and hadamard.bin
- `NgramTableSpec` / `NgramTableWriter` / `NgramPlan` / `write_ngram_table` [ngram.rs:write_ngram_table]: streamed n-gram table writer
- Vision dispatcher [vision.rs:read_vision_entries_quantized]: `VisionRead`, `read_vision_entries`, `read_vision_entries_quantized` (single dispatch by `VisionTowerKind`), `vision_should_ingest`, `vision_arch_for_manifest`, `read_tower_tensor`, `resident_shape`
- Per-family vision tower repackers (config parser and reader pairs, each builds packed blocks plus FP16 residents):
  - Gemma 4: `parse_gemma4_vision_config`, `read_gemma4_vision_entries` [vision_gemma4.rs:read_gemma4_vision_entries] (27 blocks, 13 roles)
  - CLIP / LLaVA-1.5: `parse_llava_vision_config`, `read_clip_vision_entries` [vision_clip.rs:read_clip_vision_entries]
  - LLaVA-OneVision SigLIP: `parse_llava_onevision_vision_config`, `read_onevision_vision_entries` [vision_onevision.rs:read_onevision_vision_entries]
  - DeepSeek-VL2 SigLIP: `parse_deepseek_vl2_vision_config`, `read_siglip_vision_entries` [vision_siglip.rs:read_siglip_vision_entries]
  - DeepSeek-OCR (SAM plus CLIP-L): `parse_deepseek_ocr_vision_config`, `read_deepseek_ocr_vision_entries` [vision_deepseek_ocr.rs:read_deepseek_ocr_vision_entries]
  - Pixtral: `parse_pixtral_vision_config`, `pixtral_special_token_ids`, `read_pixtral_vision_entries` [vision_pixtral.rs:read_pixtral_vision_entries]
  - LFM3-VL: `parse_lfm3_vl_vision_config`, `read_lfm3_vl_vision_entries` [vision_lfm3.rs:read_lfm3_vl_vision_entries]
  - MiniCPM-V: `read_minicpm_vision_entries` [vision_minicpm.rs:read_minicpm_vision_entries]
  - Muse-Glimmer: `read_muse_vision_entries` [vision_muse.rs:read_muse_vision_entries]

## Invariants
- A failed streamed write removes the staging dir and never touches the target dir; publish renames staging over the target with backup rollback [mod.rs:write_gemma4_install_streamed, mod.rs:publish_staged_install]
- MTP head, DFlash2 drafter, vision tower and ngram table arms must exist in both the streamed writer and `orchestrate_gemma4_checkpoint_sharded`, otherwise the install is silently headless [mod.rs:write_gemma4_install_streamed_in_place]
- Vision tower and hadamard.bin are written before `finish_streaming` because the manifest hashes every listed file [mod.rs:write_gemma4_install_streamed_in_place]
- A checkpoint with .signs tensors but no parsed hadamard contract is refused, not written sign-less [mod.rs:write_gemma4_install_streamed_in_place]
- Expert blobs are written one layer at a time from `num_dense_leading_layers` to `num_layers`; the stride is the max blob rounded to 16 KiB [mod.rs:write_gemma4_install_streamed_in_place, expert_blobs.rs:expert_stride_from_headers]
- Unknown tensor names abort the walk; duplicate routed (layer, role), ngram, or cross-shard tensors are refused, never last-wins [orchestrate.rs:classify_all, shards.rs:Gemma4Shards::new]
- Classification is by family, and MTP/DFlash/vision/ngram/hadamard tensors stay out of the trunk narrowing and ordering path [classify.rs:classify_for_family, orchestrate.rs:orchestrate_gemma4_checkpoint_sharded]
- Lossy BF16/FP16 narrowing is reported, never silent; FP16 conversion refuses overflow instead of clamping [mod.rs:write_gemma4_install_streamed_in_place, narrow.rs:convert_raw_to_fp16]
- Quant parsing: mxfp4 only at 4 bits group 32; affine 1/2-bit only at group 128, 4/8-bit only at group 64; routed experts must be 3- or 4-bit [config.rs:is_supported_affine_shape, expert_blobs.rs:routed_expert_bits]
- Vision ingest happens only when the request and the checkpoint's tensors both say so, and the manifest vision config is zeroed otherwise [vision.rs:vision_should_ingest, vision.rs:vision_arch_for_manifest]
- Every vision reader refuses unknown, duplicate, or missing tensor names and shape mismatches by name, and page-aligns `block_stride` [vision.rs:read_vision_entries_quantized, vision_clip.rs:read_clip_vision_entries]
- Fused q/k/v towers concatenate in q, k, v row order with no arithmetic [vision_clip.rs:read_clip_vision_entries, vision_muse.rs:read_muse_vision_entries, vision_pixtral.rs:read_pixtral_vision_entries]
- mxfp4 conversions are exact; exponent byte 0xFF and f32-overflowing elements are refused [mlx_mxfp4.rs:check_planes]

## Side Effects
- Creates hidden sibling staging and backup dirs `.<name>.<kind>.<pid>.<n>` beside the install dir; process-global `WORK_DIR_COUNTER` [mod.rs:sibling_work_dir]
- Renames and removes install directories (backup swap, `remove_dir_all` on failure) [mod.rs:publish_staged_install]
- Writes model_weights.bin, expert layers, manifest, hadamard.bin, packed_vision/ and ngram_table/ into the install dir [mod.rs:write_gemma4_install_streamed_in_place, ngram.rs:NgramTableWriter::finish]
- Reads checkpoint bytes via `RangeSource` (local file or network ranged reads) [shards.rs:Gemma4Shards::read]
- The non-streamed path holds all resident and expert blobs in memory [orchestrate.rs:orchestrate_gemma4_checkpoint_sharded]
- Classification, config, quant, mxfp4, and header canonicalizers are pure in-memory (header canonicalizers mutate the passed header only) [classify.rs, config.rs, mlx_mxfp4.rs]

## Verification
- Run: `cargo test -p turbospark-repack`
