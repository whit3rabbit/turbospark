# quant_gguf (turbospark-compute)

## Overview
CPU reference dequantizers, a few quantizers, and dequant GEMV routines for GGUF block formats (Q2_0 through Q8_0 and the K-quants). They serve as oracles for the Metal kernels.

## Interface
- Per-format constants and functions (`Q*_BLOCK_ELEMS`, `Q*_BLOCK_BYTES`, and for K-quants `Q*_K_SUB_ELEMS`):
  - [q2_0.rs:dequantize_q2_0], [q2_0.rs:dequant_q2_0_gemv]
  - [q2_k.rs:dequantize_q2_k], [q2_k.rs:quantize_q2_k], [q2_k.rs:dequant_q2_k_gemv]
  - [q3_k.rs:dequantize_q3_k], [q3_k.rs:quantize_q3_k], [q3_k.rs:dequant_q3_k_gemv], [q3_k.rs:q3_k_decode_scales]
  - [q4_0.rs:dequantize_q4_0]
  - [q4_k.rs:dequantize_q4_k], [q4_k.rs:quantize_q4_k], [q4_k.rs:dequant_q4_k_gemv]
  - [q5_0.rs:dequantize_q5_0]
  - [q5_k.rs:dequantize_q5_k], [q5_k.rs:dequant_q5_k_gemv]
  - [q6_k.rs:dequantize_q6_k], [q6_k.rs:quantize_q6_k], [q6_k.rs:dequant_q6_k_gemv]
  - [q8_0.rs:dequantize_q8_0], [q8_0.rs:quantize_q8_0], [q8_0.rs:dequant_q8_0_gemv]
- Comparison helper: [pearson.rs:pearson].
- Module wiring: [mod.rs].

## Invariants
- Lengths must be whole blocks and the byte slice must cover them (asserts), e.g. [q4_0.rs:dequantize_q4_0], [q2_k.rs:dequantize_q2_k].
- GEMV requires non-empty rows and `x.len() == n` [q5_k.rs:dequant_q5_k_gemv], [q6_k.rs:dequant_q6_k_gemv].
- This crate must not depend on repack, so block-layout helpers are duplicated rather than imported [q8_0.rs], [q4_k.rs], [q3_k.rs] (doc comments).
- `pearson` requires equal, non-empty inputs [pearson.rs:pearson].

## Side Effects
- None (pure CPU math).

## Verification
- Run: `cargo test -p turbospark-compute`
