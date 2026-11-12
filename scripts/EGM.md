# scripts

## Overview
Flat collection of developer, release, benchmarking and parity tooling for the TurboSpark workspace. Nothing here is a library: each file is a standalone Python 3, bash, C, C++, Swift or Rust program with its own header comment. Scripts named `test_*` and `test-*` are their own tests and are not described here. There is no build manifest for this directory.

## Interface
Release and packaging (bash):
- `swift-lib.sh` builds `turbospark-ffi` and copies the static library and header for Swift [scripts/swift-lib.sh]
- `make-app-bundle.sh`, `make-dmg.sh`, `verify-app-bundle.sh`, `make-sparkle-appcast.sh`, `compile-strings.sh` [scripts/make-app-bundle.sh]
- `package-tslab-skill.sh`, `install-tslab-skill.sh` [scripts/install-tslab-skill.sh]
- `verify-music3.sh` [scripts/verify-music3.sh]

Repository checks (python):
- `check-design.py` design ratchet check (`make design-check`) [scripts/check-design.py]
- `check-gitbook-docs.py` markdown lint for GitBook Git Sync [scripts/check-gitbook-docs.py]
- `generate-settings-catalog.py` with `--check` for drift [scripts/generate-settings-catalog.py]
- `generate_app_icons.py` [scripts/generate_app_icons.py]

Benchmark, power and parity harnesses:
- `power.sh`, `power_processes.py`, `parity.sh`, `phasediff.sh`, `bench_api.py`, `bench_api_drive.sh`, `mlx_prefill.py`, `mlx_qmm_reference.py` [scripts/power.sh]
- Paired fresh-process A/B drivers: `attention_batch_pairs.py`, `moe_expert_schedule_pairs.py`, `moe_kernel_warmup_pairs.py`, `moe_startup_pairs.py`, `prefill_benchmark_pairs.py` [scripts/moe_startup_pairs.py]
- Cross-engine KL: `kld.py`, `kld_llamacpp.py`, `kld_mlx_affine.py`, `kld_mlx_vlm.py`, `kld_flint_mlx_vlm.py`, `llamacpp_logits.c` [scripts/kld.py]
- MoE routing and expert-cache analysis: `moe_route_replay.py`, `router_hist.py`, `router_window.py`, `expert_policy_replay.py`, `pilot_ceiling.py`, `prefill_bank_replay.py`, `ffn_sparsity.py`, `check_moe_route_mutations.py` [scripts/expert_policy_replay.py]
- Qwen4 capture checks: `check_qwen4_attention_intermediate_capture.py`, `check_qwen4_gdn_norm_capture.py`, `check_qwen4_layer_boundary_capture.py`, `capture_qwen4_llama_callback.cpp`, `mtp_bisect.py` [scripts/check_qwen4_gdn_norm_capture.py]

Oracles and reference transcriptions:
- Quantization oracles: `mlx_1bit_oracle.py`, `mlx_2bit_oracle.py`, `mlx_turboquant_oracle.py`, `ggml_mxfp4_oracle.c`, `ggml_q3_k_oracle.c`, `ggml_q5_k_oracle.c`, `ggml_tables.c`, `vendor_mlx_steel_qmm.py`, `omlx_steel_probe.py`, `omlx_steel_probe.swift` [scripts/vendor_mlx_steel_qmm.py]
- Vision tower references and probes: `*_tower_reference.py` (deepseek_ocr, deepseek_vl2, llava_onevision, minicpm_v, muse_glimmer, pixtral, qwen25_vl), `deepseek_vl2_trunk_reference.py`, `gemma4_vision_*.py`, `qwen3vl_vision_oracle.py`, `unlimited_ocr2_*.py`, `vision_tower_probe.py`, `lfm3_vl_tower_probe.py`, `llava_*.py`, `fetch_vision_tower.py`, `ov_header_probe.py`, `make_vision_test_page.py`, `check_ocr_page_output.py`, `ocr_server_gate.py` [scripts/vision_tower_probe.py]
- Z-Image evidence pipeline: `z_image_probe.py`, `z_image_download.py`, `z_image_capture.py`, `z_image_intra_block.py`, `z_image_checkpoint_block.py`, `z_image_compare.py`, `z_image_vae_compare.py`, `z_image_quantization.py`, `z_image_benchmark.py`, `z_image_benchmark_suite.py`, `z_image_benchmark_summary.py`, `z_image_summarize.py`, `z_image_evidence.py`, `z_image_quant_ref.rs` [scripts/z_image_evidence.py]
- Misc: `extract_direction.py` (steering directions), `skill_state_probe.py`, `fish_drama3_voiced_table.py` [scripts/extract_direction.py]

## Invariants
- `make-app-bundle.sh` runs under `set -euo pipefail` and requires `OPENKIND_RELEASE_REVISION` to be an exact published 40-character commit [scripts/make-app-bundle.sh:openkind_revision]
- `check-design.py` and `generate-settings-catalog.py --check` exit nonzero on drift [scripts/check-design.py:main] [scripts/generate-settings-catalog.py:--check]
- `vendor_mlx_steel_qmm.py` refuses to write when pinned digests do not match [scripts/vendor_mlx_steel_qmm.py:hashlib.sha256]
- `power.sh` needs sudo for `powermetrics` and measures one real install [scripts/power.sh:powermetrics]
- Measurement scripts report observed state and do not edit source [scripts/check_moe_route_mutations.py]

## Side Effects
- Writes the Swift FFI staticlib and header [scripts/swift-lib.sh:cp]
- Builds and codesigns `TurboSpark.app`, writes the DMG and the Sparkle appcast [scripts/make-app-bundle.sh:codesign] [scripts/make-dmg.sh] [scripts/make-sparkle-appcast.sh]
- Installs the tslab skill under `~/.claude/skills` by default [scripts/install-tslab-skill.sh:dest]
- Regenerates `crates/gpu/src/shaders/dequant_int4_steel.metal` [scripts/vendor_mlx_steel_qmm.py:OUTPUT]
- Regenerates oracle files under `crates/compute/tests/generated/` [scripts/mlx_1bit_oracle.py] [scripts/mlx_2bit_oracle.py] [scripts/unlimited_ocr2_dash_oracle.py]
- Network access to Hugging Face for downloads [scripts/z_image_download.py:snapshot_download] [scripts/fetch_vision_tower.py:urlopen]
- Benchmarks launch release binaries and local servers and sample power with `powermetrics` [scripts/power.sh] [scripts/bench_api_drive.sh]

## Verification
- Run: `python3 scripts/check-design.py` for the design check; for the Python tests run `python3 -m unittest discover -s scripts -p 'test_*.py'`
