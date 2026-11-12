# gpu shaders

## Overview
Metal Shading Language sources for every kernel the `turbospark-gpu` crate dispatches: decode and indexed/TQ attention, dequantizing GEMV/GEMM for 1/2/4/8-bit and GGUF K-quant/IQ types, MoE decode and prefill, GDN, MLA, RoPE, RMSNorm, quantized KV, vision, Whisper, Conformer, Kokoro, Moonshine, Music 3, and utilities. The sources are loaded by Rust dispatch modules via `include_str!` and compiled at runtime, or compiled ahead of time by the build script.

## Interface
- Attention: [attention.metal], [attention_indexed.metal], [attention_tq.metal] (function constants `FC_ATTN_HEAD_DIM`, `FC_ATTN_NUM_Q_HEADS`, `FC_ATTN_NUM_KV_HEADS`, `FC_ATTN_USE_FC`, `FC_ATTN_SCALE`, `FC_ATTN_NUM_CHUNKS`, `FC_ATTN_RING_CAP`, `FC_ATTN_HAS_SINKS`, `FC_ATTN_GQA_GROUP`, `FC_ATTN_GQA_SIMDGROUPS`, `FC_ATTN_GQA_HEAD_DIM`, `FC_ATTN_GQA_UNROLL`, `FC_ATTN_TREE`, `FC_ATTN_ROWS_TILE`, `FC_ATTN_ROWS_HEADS` in [attention.metal]; each belongs in the Rust pipeline cache key).
- Dequant GEMV/GEMM: [dequant_1bit.metal], [dequant_2bit.metal], [dequant_int4.metal], [dequant_int4_batch.metal], [dequant_int4_fused.metal], [dequant_int4_mma.metal], [dequant_int4_mma8.metal], [dequant_int4_steel.metal], [dequant_int8.metal], [dequant_int8_batch.metal], [dequant_iq.metal] (+ [dequant_iq_lowbit_tables.metal]), [dequant_q2_k.metal], [dequant_q3_k.metal], [dequant_q4_k.metal], [dequant_q5_k.metal], [dequant_q6_k.metal], [dequant_q8_0.metal], [gemv_bf16.metal].
- MoE: [moe.metal] (19 kernels), [moe_gguf.metal] (15), [moe_prefill_batch.metal], [moe_prefill_batch_gguf.metal], [minimax_router.metal].
- Model blocks: [gdn.metal], [mla.metal], [rope.metal] (`rope_default_neox`, `rope_neox_subdim`, `rope_mrope_interleaved`, `rope_mrope_chunked`, `rope_proportional_neox`, ...), [rmsnorm.metal], [short_conv.metal], [hyper.metal], [ple.metal], [qsa_indexer.metal], [qsa_topk.metal], [hadamard.metal], [dflash_conv.metal], [logit.metal], [kv_quantize_tq.metal], [utility.metal].
- Audio and vision: [music3.metal] (`music3_linear`, `music3_attention`, `music3_norm`, ...), [music3_sequence.metal], [music3_attention_fallback.metal], [kokoro.metal], [whisper_conv.metal], [whisper_encoder.metal], [conformer_stream.metal], [moonshine_frontend.metal], [nemotron_diar.metal], [vision.metal].
- Loaders: Rust `include_str!` sites such as [../attention_decode.rs:SOURCE] and [../precompiled.rs].

## Invariants
- Function constants specialize pipelines, so every constant must be part of the Rust pipeline cache key or a wrong pipeline is silently reused (workspace rule; constants declared in [attention.metal]).
- Decode attention (`M_q = 1`) is split-KV tiled; the MPP prefill path handles `M_q > 1` separately [attention.metal] (header comment).
- Changing `num_chunks` above 1 changes floating-point rounding [attention.metal] (comment above the multi-chunk kernels).
- Precompiled IR is keyed by the exact composed MSL bytes; function constants stay unspecialized until the normal pipeline path runs [../precompiled.rs] (module doc).
- Shared shader sources must be statics because the in-process cache keys their addresses [../warmup.rs] (module doc).

## Side Effects
- None at rest. Compile flags `-std=metal3.1 -mmacosx-version-min=14.0 -ffast-math` are used by the build script via `xcrun` [../../build.rs:FLAGS], and `TURBOSPARK_METAL_PRECISE_MATH` selects precise math [../context/device.rs:MetalContext].

## Verification
- Run: `cargo test -p turbospark-gpu` (needs a Metal device; shaders compile through the crate)
