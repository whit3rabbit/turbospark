# TurboSpark Repack (crate root)

## Overview
`turbospark-repack` turns checkpoints (HF safetensors, MLX, GGUF) into `.gturbo` installs: it parses per-family config.json into `ArchConfig`, quantizes matrices to affine INT4/INT8 itself (pre-quantized INT1/2/4/8 source tensors are repacked as-is), and writes the `model_weights.bin` resident index. It also verifies, peeks, and migrates existing installs. The crate root declares every module, re-exports the public API, and forbids unsafe code. Most config parsers refuse any behavior the hard-coded decode flow does not implement, since a wrong family or silently ignored key yields fluent but wrong output. Files named `synthetic_*` are tiny-install fixtures for tests.

## Interface
- `#![forbid(unsafe_code)]`, `#![recursion_limit = "512"]` [lib.rs:forbid]: crate-wide policy; the limit serves the single `json!` manifest literal
- `gguf_arch_support` / `ArchSupport` / `PlannedArch` [arch_registry.rs:gguf_arch_support]: GGUF architecture to Supported(ModelFamily) or Planned (recognition only, error text)
- `hf_family_for_model_type`, `config_json_family`, `refuse_foreign_config` [arch_registry.rs:refuse_foreign_config]: HF model_type routing and mismatch refusal
- `describe_headless_tts_trunk` [arch_registry.rs:describe_headless_tts_trunk]: routes a headless TTS trunk to the audio catalog
- `parse_qwen_gdn_moe_config`, `parse_qwen_gdn_dense_config`, `parse_qwen4_exp_config` [qwen36_config.rs:parse_qwen_gdn_moe_config]: Qwen GDN family configs plus the vision_config tower parsers
- `parse_qwen2_config` [qwen2_config.rs:parse_qwen2_config]: dense Qwen2/2.5 for the shared Llama flow
- `parse_qwen3_vl_config` [qwen3vl_config.rs:parse_qwen3_vl_config]: text-only qwen3_vl intake
- `parse_lfm3_config` [lfm3_config.rs:parse_lfm3_config]: LiquidAI lfm3 config
- `parse_flint_config`, `parse_flint_quantization`, `flint_admits_affine_shape` [flint_config.rs:parse_flint_config]: Recursive Flint config and quantization block
- `parse_muse_glimmer_config`, `MuseGlimmerScalars` [museglimmer_config.rs:parse_muse_glimmer_config]: muse_glimmer config plus scalars kept out of ArchConfig
- `parse_deepseek_vl2_config` [deepseek_vl2_config.rs:parse_deepseek_vl2_config]: non-MLA DeepSeek-VL2 config
- `parse_unlimited_ocr2_config`, `canonicalize_unlimited_ocr2_header` [unlimited_ocr2_config.rs:parse_unlimited_ocr2_config]: Unlimited-OCR-2 config and header renaming
- `write_unlimited_ocr2_tower`, `UOCR2_TOWER_BIN`, `UOCR2_TOWER_INDEX` [unlimited_ocr2_tower.rs:write_unlimited_ocr2_tower]: FP16 vision tower file plus offset index
- `DRAMA3_MODEL_TYPE`, `DRAMA3_SIDE_PREFIX`, `DRAMA3_TRUNK_DIR`, `DRAMA3_SIDE_FILE` [fish_drama3.rs:DRAMA3_MODEL_TYPE]: Fish Drama 3 headless trunk install layout
- `shift_bf16_exponent`, `SwigluRole`, `FLINT_SWIGLU_SHIFT` [flint_rebalance.rs:shift_bf16_exponent]: Flint FP16 range fix, in-place migration
- `orchestrate_llama_checkpoint`, `LlamaCheckpointDims`, `OrchestrateError` [hf_checkpoint.rs:orchestrate_llama_checkpoint]: Llama safetensors to INT4 ResidentTensorSpec
- `quantize_matrix_int4`, `quantize_matrix_int8`, `RepackError` [repack.rs:quantize_matrix_int4]: per-matrix affine quantization
- `build_resident_weights_bin`, `build_resident_weights_bin_mixed`, `write_resident_weights_bin_mixed` [resident_writer.rs:build_resident_weights_bin_mixed]: model_weights.bin index writers
- `read_resident_entries` [resident_reader.rs:read_resident_entries]: inverse of the writer (MTP head grafting)
- `parse_header`, `SafetensorsHeader` [safetensors_header.rs:parse_header]: safetensors header for ranged-download planning
- `GgufSet`, `gguf_shard_names` [gguf_set.rs:GgufSet]: merged address space over split GGUF shards
- `parse_control_vector`, `load_control_vector`, `write_control_vector` [control_vector.rs:parse_control_vector]: llama.cpp control vectors to SteeringSet
- `verify_install_full_sha256` [install_verifier.rs:verify_install_full_sha256]: rehash every manifest file
- `peek_manifest_arch` [manifest_peek.rs:peek_manifest_arch]: ArchConfig back from manifest.json
- `from_config_json`, `from_gguf`, `record` [trained_context.rs:record]: trained context length, stored as arch.trainedContext
- Test fixtures `build_synthetic_*` and `tiny_*_arch` [synthetic_tensors.rs:assemble_safetensors]: deepseek2, flint, lfm3, llama (`build_synthetic_llama_real_install`), muse (`tiny_muse_glimmer_arch`), spark, gemma4, unlimited_ocr2 tiny installs over shared helpers
- Subdirectories: [gemma4_checkpoint](./gemma4_checkpoint/EGM.md), [gguf_checkpoint](./gguf_checkpoint/EGM.md), [gguf_config](./gguf_config/EGM.md), [gguf_header](./gguf_header/EGM.md), [gguf_names](./gguf_names/EGM.md), [gturbo_writer](./gturbo_writer/EGM.md), [ranged_download](./ranged_download/EGM.md), [synthetic_model](./synthetic_model/EGM.md)

## Invariants
- No unsafe code anywhere in the crate [lib.rs:forbid]
- Registry lookups are exact string equality, never prefix, so qwen35 vs qwen35moe and gemma4 vs gemma4_unified never collapse; a Planned result must still be refused by the caller [arch_registry.rs:gguf_arch_support]
- A config with no recognizable model_type is accepted; only a positive family mismatch is refused [arch_registry.rs:refuse_foreign_config]
- Config parsers refuse unimplemented behavior by name (eps, activation, sliding window, MLA, rope type) instead of ignoring it [flint_config.rs:parse_flint_config]
- Layer counts are capped at 4096 and head and kv-head divisibility is checked [lfm3_config.rs:parse_lfm3_config]
- Unlimited-OCR v1 (no sparse_attention_config) is refused before the family check [unlimited_ocr2_config.rs:parse_unlimited_ocr2_config]
- Layer types map to mask codes (conv 6, full_attention 1, sliding_attention 0); any other type, including linear attention, is refused [lfm3_config.rs:layer_mask]
- Flint rebalance moves 2^6 from up_proj to down_proj, never the gate; zero, subnormal, inf, NaN and out-of-range exponents pass through unchanged [flint_rebalance.rs:shift_bf16_exponent]
- Only F32 and BF16 source dtypes are decoded; F16 is UnsupportedDtype; [hf_checkpoint.rs:decode_tensor_f32]
- `lm_head` is emitted only when embeddings are untied [hf_checkpoint.rs:orchestrate_llama_checkpoint]
- Quantization requires data.len() == rows*cols with cols a nonzero multiple of GROUP_SIZE [repack.rs:quantize_matrix_int4]
- Resident index layout constants come from model_io; the index region is padded to 16 KiB pages so data is page-aligned for NoCopy Metal buffers [resident_writer.rs:build_resident_weights_bin_mixed]
- Resident entries read back in file-offset order; GGUF block dtypes error out naming the tensor [resident_reader.rs:read_resident_entries]
- Manifest reads are size-capped before allocation (cap+1 bytes) [manifest_peek.rs:read_manifest_capped]; `parse_header` rejects a declared header length above the cap, and data_offsets with end before start [safetensors_header.rs:parse_header]
- Manifest keys must be non-empty relative Normal-component paths; the first hash mismatch aborts (case-insensitive compare) [install_verifier.rs:verify_install_full_sha256]
- Split GGUF requires split.count in 1..=1024 and the caller must name shard 1 [gguf_set.rs:gguf_shard_names]
- Control vector direction.N is block N (layer_base 1 default, 0 legacy, anything else refused); a non-integer layer_base is refused [control_vector.rs:layer_base]
- Trained context of zero or above u32::MAX reads as unknown; `sane` does the bound check, and text_config beats root [trained_context.rs:sane, trained_context.rs:from_config_json]
- Fixtures assert the walk-derived arch equals the tiny arch, and use binary-fraction attention_scale so validate_arch's exact f64 compare survives the manifest round trip [synthetic_flint.rs:tiny_flint_arch]
- Fixture quantized names end in .weight and group size is 64 [synthetic_tensors.rs:int4_triple]

## Side Effects
- Reads `<model_dir>/manifest.json` (capped) [manifest_peek.rs:read_manifest_capped]
- Rewrites manifest.json via manifest.json.tmp and atomic rename [trained_context.rs:record]
- Reads every manifest-listed install file in 1 MiB chunks [install_verifier.rs:verify_install_full_sha256]
- Rewrites model_weights.bin scale and bias planes and packed_experts/layer_*.bin in place [flint_rebalance.rs:rebalance_install_in_place]
- Reads model_weights.bin with seek and read_exact [resident_reader.rs:read_resident_entries]
- Reads a control-vector file, bounded to 16 MiB + 1 [control_vector.rs:load_control_vector]
- Writes trunk/ install and tts_side.safetensors (tmp file, fsync, rename) [fish_drama3.rs:write_drama3_side_safetensors]
- Writes uocr2_tower.bin and uocr2_tower.json; takes already canonicalized headers read-only; `canonicalize_unlimited_ocr2_header` mutates its `&mut` header in place [unlimited_ocr2_config.rs:canonicalize_unlimited_ocr2_header]
- Tensor bytes may come from disk or network via RangeSource [hf_checkpoint.rs:decode_tensor_f32]
- Fixtures write `.gturbo` installs into a caller-provided dir; no network [synthetic_spark.rs:build_synthetic_spark_install]
- Parsers, registry, quantizer and resident_writer builders are pure in-memory [repack.rs:quantize_matrix_int4]

## Verification
- Run: `cargo test -p turbospark-repack`
