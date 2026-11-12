# turbospark-gpu (src)

## Overview
Metal GPU backend (macOS only; on other platforms the crate exports only `TokenId`). Holds per-kernel dispatch modules, resident-weight and KV-cache memory management, state managers for GDN, short-conv, DSV4 and QSA, and the opt-in ANE split. Kernels are validated against `turbospark-compute` CPU references. Shader sources are in the `shaders` child.

## Interface
- Context and passes: re-exported [context](./context/EGM.md) items `MetalContext`, `PassEncoder`, `CommittedPass`, `GpuError`, `autorelease_pool` [lib.rs]; `MetalBuffer` alias for `metal::Buffer`; `TokenId` re-export.
- Attention: [attention_decode.rs:encode_attention_decode], [attention_decode.rs:encode_attention_decode_gqa], [attention_decode.rs:encode_attention_decode_batch], [attention_decode.rs:encode_attention_decode_tree], [attention_decode.rs:AttentionScratch]; [attention_indexed.rs:encode_attention_decode_indexed]; [attention_tq.rs:encode_attention_decode_tq], [attention_tq.rs:TqAttentionScratch]; [mla.rs:encode_mla_attention_decode].
- Dequant GEMV/GEMM per format with `*ResidentMatrix` types and `encode_*` functions: int1, int2, int4 ([dequant_int4_steel.rs:encode_dequant_int4_gemm_steel], [dequant_int4_mma8.rs:encode_dequant_int4_mma8_resident], [dequant_int4_batch.rs]), int8, IQ, Q2_K to Q6_K, Q8_0, BF16 ([gemv_bf16.rs:Bf16ResidentMatrix]).
- MoE: [moe_decode.rs:encode_moe_phase1], [moe_decode.rs:encode_moe_phase2], [moe_decode.rs:MAX_STREAMED_EXPERTS], [moe_decode.rs:RoutedBlobsBuffer], [moe_prefill_batch.rs:encode_moe_prefill_phase1], [moe_prefill_batch.rs:MAX_PREFILL_EXPERT_BINDINGS]; GGUF variants in [moe_gguf](./moe_gguf/EGM.md) and [moe_prefill_batch_gguf.rs].
- State and memory: [kv_cache.rs:KvCacheManager], [kv_cache.rs:LayerKind], [resident_metal.rs:ResidentGpuWeights], [resident_metal.rs:wrap_page_aligned_no_copy], [gdn_state.rs:GdnStateManager], [short_conv_state.rs:ShortConvStateManager], [dsv4_state.rs:Dsv4StateManager], [qsa_indexer_state.rs:QsaIndexerCacheManager], [prefill_scratch.rs:PrefillChunkScratchLayout], [prefill_scratch.rs:MAX_BATCH_ATTENTION_SCRATCH_BYTES].
- Blocks: [gdn.rs] (`encode_gdn_*`), [rope.rs] (`encode_rope_*`), [rms_norm.rs] (`encode_rms_norm_*`), [utility.rs] (`encode_*` elementwise), [hadamard.rs:encode_hadamard_fwht], [kv_quantize.rs:encode_kv_quantize_tq], [logit_softmax.rs:encode_logit_softcap_softmax], [short_conv.rs], [hyper_connection.rs], [ple.rs:encode_ple_gate], [qsa_indexer.rs], [qsa_topk.rs], [minimax_router.rs:encode_minimax_router].
- Other models: [vision.rs] (`encode_vision_*`), [whisper_encoder.rs], [whisper_conv.rs], [conformer_stream.rs], [moonshine_frontend.rs], [nemotron_diar.rs], [dflash_conv.rs]; Music 3 in [music3.rs] (`Music3Device`, `Music3Weight`, `Music3Sequence`) and [music3](./music3/EGM.md).
- ANE split (feature `ane-prefill`): [ane_split.rs:encode_ane_dequant_rows], [ane_split.rs:HostEvent], [ane_split.rs:MAX_COMMAND_BUFFERS].
- Diagnostics and platform: [dispatch_profile.rs:report], [dispatch_profile.rs:reset], [warmup.rs:KernelWarmupPlan], [warmup.rs:KernelWarmup], [warmup.rs:KernelWarmupStats], [power_state.rs:thermal_state_raw], [power_state.rs:low_power_mode_enabled], [power_state.rs:memory_pressure_raw], [device_memory.rs:recommended_max_working_set], [device_memory.rs:process_footprint], [precompiled.rs], [bytes.rs:read_f32_buffer].
- Children: [context](./context/EGM.md), [moe_gguf](./moe_gguf/EGM.md), [music3](./music3/EGM.md), [shaders](./shaders/EGM.md).

## Invariants
- Repeated encoding loops must be wrapped in `autorelease_pool` or autoreleased Metal objects accumulate [context/error.rs:autorelease_pool].
- KV sizing is part of correctness: `max_context` and prefill chunk tokens must be positive; ring capacity derives from the layer mask [kv_cache.rs:KvCacheManager].
- Attention decode requires query heads divisible by KV heads, batch K/V buffers in FP16, and a live row count within capacity [attention_decode.rs:encode_attention_decode_batch].
- Split-KV GQA kernel constants (group, simdgroups, head_dim, unroll) are part of the pipeline cache key; a missing constant reuses the wrong pipeline (workspace rule) [attention_decode.rs].
- Metal JIT warmup is opt-in, and the gate (`TURBOSPARK_METAL_KERNEL_WARMUP=1`) lives in the runtime [../../runtime/src/real_forward_open.rs], not in [warmup.rs:KernelWarmupPlan].
- Warmup specializes via production builders and shared sources are statics because the pipeline cache keys addresses [warmup.rs] (module doc).
- The `power_state` unsafe probes live here because `crates/runtime` forbids unsafe code [power_state.rs] (module doc).
- Everything except the crate root is gated `#[cfg(target_os = "macos")]` [lib.rs].

## Side Effects
- Creates Metal buffers, wraps the weights mmap as a no-copy `MTLBuffer`, and commits command buffers [resident_metal.rs:ResidentGpuWeights], [context/pass.rs].
- Reads `TURBOSPARK_DISPATCH_PROFILE` for per-dispatch timing [dispatch_profile.rs:enabled].
- Reads kernel-selection env vars: `TURBOSPARK_ATTN_DECODE_KERNEL` [attention_decode.rs], `TURBOSPARK_GDN_DK128` [gdn.rs], `TURBOSPARK_VISION_MMA` and `TURBOSPARK_VISION_ATTN_MMA` [vision.rs].
- Calls `NSProcessInfo` for thermal and low-power state and reads process memory footprint [power_state.rs], [device_memory.rs:process_footprint].

## Verification
- Run: `cargo test -p turbospark-gpu` (needs a Metal device)
