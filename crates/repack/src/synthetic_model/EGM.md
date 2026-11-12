# Synthetic Model

## Overview
Builds tiny, deterministic Gemma-4-shaped `.gturbo` installs in memory (INT4-affine quantized, not trained) through the real resident-tensor and install writers. It exists so the runtime's real forward-pass tests have a loadable checkpoint when no trained one is available. Non-shape architecture fields come from Gemma 4's canonical baseline, so the models are honestly small, not invented. Variants cover dense, sliding-window, resident MoE, and streamed-expert MoE layouts.

## Interface
- `tiny_gemma4_arch` [arch.rs:tiny_gemma4_arch]: dense Gemma4 `ArchConfig` with caller-chosen vocab and layer count
- `HIDDEN_SIZE`, `NUM_HEADS`, `FULL_HEAD_DIM`, `INTERMEDIATE_SIZE` [arch.rs]: fixed toy dims (64, 2, 32, 64)
- `q_proj_name`, `k_proj_name`, `o_proj_name`, `gate_proj_name`, `up_proj_name`, `down_proj_name`, `router_name`, `embed_lm_head_name` [arch.rs]: resident tensor name helpers
- `expert_gate_proj_name`, `expert_up_proj_name`, `expert_down_proj_name` [arch.rs]: per-expert tensor names
- `build_synthetic_gemma4_install` [dense.rs:build_synthetic_gemma4_install]: dense install, returns the `ArchConfig`
- `build_synthetic_gemma4_swa_install` [dense.rs:build_synthetic_gemma4_swa_install]: alternating sliding (0) and full (1) layers
- `build_synthetic_gemma4_moe_install` [moe.rs:build_synthetic_gemma4_moe_install]: routed experts in the resident region, no dense FFN branch
- `build_synthetic_gemma4_moe_streamed_install` [moe.rs:build_synthetic_gemma4_moe_streamed_install]: same weights, experts in `packed_experts/layer_NN.bin`
- `deterministic_row`, `quantized_tensor` [arch.rs]: crate-private seeded xorshift rows and INT4 spec builder
- Subdirectories: None

## Invariants
- The contraction dims (`HIDDEN_SIZE`, `INTERMEDIATE_SIZE`) are multiples of 64, the quantization group size, so every GEMV contraction dim is valid; `NUM_HEADS` and `FULL_HEAD_DIM` are not [arch.rs:HIDDEN_SIZE]
- Weights are fully deterministic per seed (no RNG crate); layer seeds are `1000*(l+1)+k`, expert seeds `base+100+10*e+k` [arch.rs:deterministic_row, moe.rs]
- Streamed and resident MoE builders use identical seeds, so a runner must produce identical tokens on both [moe.rs:build_synthetic_gemma4_moe_streamed_install]
- Streamed expert blobs hold 9 sub-tensors per expert; bf16 scale/bias shapes are `[rows, ceil(cols/64)]`, not `[rows, cols]` [moe.rs]
- Expert stride is the max blob size rounded up to 16 KiB pages [moe.rs:build_synthetic_gemma4_moe_streamed_install]
- MoE variants set `moe_intermediate_size = INTERMEDIATE_SIZE` and write no shared dense FFN, a deliberate deviation from real Gemma 4 (see `DEVIATIONS.md`); `attention_k_eq_v` separately explains the missing V weights. `RealForwardRunner` runs routed-only when `num_experts > 0` [moe.rs]
- Tensor names must stay in sync with what `RealForwardRunner` looks up [arch.rs:embed_lm_head_name]
- The returned `ArchConfig` is the one the caller must pass to `load_manifest`, which validates against a supplied arch [dense.rs:build_synthetic_gemma4_install]

## Side Effects
- Writes a full `.gturbo` install (manifest, resident weights, optional expert blobs) under the caller-supplied `dir` [dense.rs, moe.rs via gturbo_writer]
- Allocates all weights in memory before writing [arch.rs:quantized_tensor]

## Verification
- Run: `cargo test -p turbospark-repack synthetic_model`
