# Synthetic GGUF Fixtures

## Overview
In-memory generators of tiny, structurally real GGUF v3 files for testing the GGUF header parser and repack walk without multi-GB downloads. Each generator hands already-packed, deterministic bytes to the shared `GgufBuilder` and returns the file bytes plus each tensor's data byte range, so tests assert structure, offsets and byte identity, never generated text. One module per architecture: Gemma 4 (with several quant mixtures), gpt-oss, MiniMax M2 and Qwen2.

## Interface
- `GgufBuilder`, `GgufFileAndRanges` [mod.rs:GgufBuilder] (re-exported from builder): serializer and its `(bytes, ranges)` return type
- `build_synthetic_gemma4_gguf` [gemma4.rs:build_synthetic_gemma4_gguf]: toy Gemma 4 MoE GGUF with the real tensor and metadata inventory
- `SyntheticGgufShape` [gemma4_shape.rs:SyntheticGgufShape]: dimension knobs; presets `k_quant`, `iq_mixed`, `mxfp4`, `q3_k`, and `Default` (two layers, sliding plus global)
- `QuantMix` [gemma4_shape.rs:QuantMix]: Q8_0, KQuant, Iq, Mxfp4, Q3K block-type mixtures
- `build_synthetic_gpt_oss_gguf` [gptoss.rs:build_synthetic_gpt_oss_gguf]: gpt-oss fixture with rank-2 routed biases, six routed roles, unfused gate/up, untied head
- `SyntheticGptOssShape` [gptoss.rs:SyntheticGptOssShape]: shape knobs for the gpt-oss fixture
- `build_synthetic_minimax_gguf` [minimax.rs:build_synthetic_minimax_gguf]: `minimax-m2` fixture with full-width norms, F32 routing and mixed K-quants
- `build_synthetic_qwen2_gguf` [qwen2.rs:build_synthetic_qwen2_gguf]: dense Qwen2 fixture with explicit RMS epsilon and qkv biases
- Subdirectories: [builder](./builder/EGM.md)

## Invariants
- Dimensions must tile the block size of every tensor's quant type (32 for Q8_0/MXFP4, 256 for K-quants), per expert as well as in total; enforced by panicking asserts in the tensor helpers, not by the shape type [builder/tensors.rs:q8_0_tensor, builder/tensors.rs:q4_k_tensor, builder/tensors.rs:mxfp4_tensor]
- Gemma fixture: gate and up fused as `ffn_gate_up_exps`, no `output.weight` (tied), no `attn_v` on global layers [gemma4.rs:build_synthetic_gemma4_gguf]
- gpt-oss gate/up and down bias lengths differ on purpose so a wrong-role read is a size mismatch [gptoss.rs:SyntheticGptOssShape]
- gpt-oss default has two layers with even layers sliding, to pin the window-mask phase [gptoss.rs:SyntheticGptOssShape]
- Writer only: no quantization, tensors laid out in push order at aligned offsets [mod.rs]
- Payloads are deterministic but meaningless [mod.rs]

## Side Effects
- None (pure in-memory construction; no fs, env, network or global state)

## Verification
- Run: `cargo test -p turbospark-repack`
