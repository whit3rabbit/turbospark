# Synthetic Qwen Fixtures

## Overview
Test-fixture generators that build tiny Qwen-family (and Fish Drama 3 and vision sidecar) checkpoints in memory as safetensors blobs, then write `.gturbo` installs through the real repack writers (`write_qwen_gdn_dense_install`, `write_qwen_gdn_moe_install`, `write_gemma4_install[_streamed]`, the Drama intake). Weights are deterministic and untrained, so tests can verify layout, wiring and finite flow but not output quality. Covers dense GDN (qwen3_5, 1/2/4-bit, MTP, DFlash, vision), MoE (affine and mxfp4), qwen4_exp (n-gram PLE and a decode-capable variant), Drama 3, and a vision tower sidecar.

## Interface
- `build_synthetic_qwen_gdn_dense_install` and variants `_at_bits`, `_with_mtp[_streamed]`, `_with_dflash[_streamed|_wide_heads]`, `_with_both_drafters`, `_with_vision[_streamed]` [dense.rs]: write a dense GDN install to a dir, return `Result<ArchConfig, Box<dyn Error>>`
- `tiny_qwen_gdn_dense_arch` [dense_arch.rs:tiny_qwen_gdn_dense_arch]: arch for the dense fixture (shared shape consts, `group_for`, `companion_dtype`)
- `build_synthetic_qwen_gdn_moe_install`, `_with_mtp`, `tiny_qwen_gdn_moe_arch` [moe.rs]: Qwen 3.6 MoE install and arch
- `build_synthetic_qwen_gdn_moe_mxfp4_install` [moe_mxfp4.rs:build_synthetic_qwen_gdn_moe_mxfp4_install]: MoE laid out like the MiniCPM-V mxfp4 checkpoint, with decoy vision tensors
- `build_synthetic_qwen4_exp_install`, `_streamed`, `tiny_qwen4_exp_arch` [qwen4.rs]: n-gram table writer-wiring fixture (not a decode flow)
- `build_synthetic_qwen4_exp_decode_install` and variants, `tiny_qwen4_exp_decode_arch`, shape consts (`HC_COUNT`, `HEAD_DIM`, `NUM_EXPERTS`, `TOP_K`, ...) [mod.rs] (from qwen4_decode)
- `synthetic_fish_drama3_checkpoint`, `build_synthetic_fish_drama3_install`, `write_synthetic_fish_drama3`, `SyntheticDrama3`, `Drama3NormStorage`, `DRAMA3_FIXTURE_LAYERS` [drama3.rs]: Drama 3 checkpoint and streamed install
- `build_synthetic_vision_sidecar`, `tiny_vision_config` [vision.rs]: vision tower sidecar written to a dir
- Subdirectories: [qwen4_decode](./qwen4_decode/EGM.md)

## Invariants
- Dense fixture supports only 1-, 2- and 4-bit; other widths panic, and columns must be a whole number of quant groups [dense_tensors.rs:packed_triple]
- `group_for` must agree with the quantizer actually used (INT4 fixes its own group size) [dense_arch.rs:group_for]
- Vision dimensions are mutually indivisible by design so a transposed or swapped read changes byte count [vision.rs:V_HIDDEN]
- qwen4 n-gram fixture keeps `num_experts` 0 and underfills its table (heads * size <= total rows) [qwen4.rs:build_tensors]
- Drama 3 fixture is four layers, three linear then one full, with a global 8-bit default and per-path 4-bit overrides so the quant-key rename is observable [drama3.rs:DRAMA3_FIXTURE_LAYERS]
- mxfp4 fixture reuses the affine MoE arch shapes so installs can be compared [moe_mxfp4.rs:build_synthetic_qwen_gdn_moe_mxfp4_install]
- Builders return `Result<_, Box<dyn Error>>` and propagate write errors rather than panicking [vision.rs:build_synthetic_vision_sidecar]

## Side Effects
- Writes install files (`.gturbo` data, manifests, sidecars) into the caller-supplied directory [dense.rs:build_synthetic_qwen_gdn_dense_install]
- Writes `preprocessor_config.json` containing `{}` into the sidecar dir [vision.rs:build_synthetic_vision_sidecar]
- No env or network access

## Verification
- Run: `cargo test -p turbospark-repack`
