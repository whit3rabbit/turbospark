# Synthetic GGUF Builder

## Overview
A fluent builder that serializes a spec-conformant GGUF v3 file in memory for test fixtures, with deterministic tensor payloads in several ggml block formats. `mod.rs` handles metadata, header, tensor table, alignment and value encoding; `tensors.rs` adds typed tensor helpers on `GgufBuilder`. Build also returns each tensor's absolute data byte range so tests can assert byte identity without re-deriving the layout.

## Interface
- `GgufBuilder` [mod.rs:GgufBuilder]: accumulator with `new`, `with_alignment`, `metadata`, `metadata_str`, `metadata_u32`, `metadata_f32`, `build`
- `GgufFileAndRanges` [mod.rs:GgufFileAndRanges]: `(Vec<u8>, Vec<(String, (u64, u64))>)` return of `build`
- `GgufBuilder::tensor` [tensors.rs:tensor]: raw tensor with caller bytes; dims fastest-varying first
- `q8_0_tensor`, `q4_k_tensor`, `q6_k_tensor`, `q3_k_tensor` [tensors.rs]: quantized helpers (ggml types 8, 12, 14, 11)
- `iq_tensor` [tensors.rs:iq_tensor]: random valid IQ3_XXS/IQ4_NL/IQ4_XS code points (types 18, 20, 23)
- `mxfp4_tensor` [tensors.rs:mxfp4_tensor]: random valid MXFP4 blocks (type 39), exponent 124
- `f32_upcast_bf16_tensor`, `f32_tensor` [tensors.rs]: F32 tensors, BF16-exact values and ordinary values
- `PendingTensor` [mod.rs:PendingTensor]: crate-private queued tensor
- Subdirectories: None

## Invariants
- Output is GGUF v3 little endian with magic `GGUF`; data region and each tensor offset are aligned to `alignment` (default 32) [mod.rs:GgufBuilder::build, DEFAULT_ALIGNMENT]
- `general.alignment` is emitted only when `with_alignment` was called, covering the absent-key default path [mod.rs:GgufBuilder]
- `with_alignment` panics on zero or non-power-of-two [mod.rs:with_alignment]
- Arrays must be homogeneous (assert_eq on element type id); empty arrays encode as U8 [mod.rs:write_value]
- Row length `dims[0]` must tile the block size (32 for Q8_0/MXFP4/IQ4_NL, 256 for K-quants/IQ3_XXS/IQ4_XS), else panic [tensors.rs:q8_0_tensor, q4_k_tensor, iq_tensor, mxfp4_tensor]
- `iq_tensor` panics on a ggml type other than 18, 20, 23 [tensors.rs:iq_tensor]
- Weights are kept small (Q8_0 quants in [-8, 7], K-quants in [-0.5, 0.5), small IQ/MXFP4 scales) so fixture forward passes stay inside FP16 [tensors.rs]
- Payloads are seed-deterministic; different seeds give different bytes [tensors.rs]

## Side Effects
- None. Pure in-memory serialization; panics (not errors) signal broken fixtures [mod.rs:build, tensors.rs]

## Verification
- Run: `cargo test -p turbospark-repack synthetic_gguf`
