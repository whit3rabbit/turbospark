//! Metal GPU backend: device/pipeline-cache context and per-kernel
//! dispatch, validated against `turbospark-compute`'s CPU reference kernels.
//! Ported from `Infrastructure/Metal` and the `.metal` shader sources under
//! `Metal/*`.
//!
//! macOS-only (per the workspace's Phase 6 decision: reuse the existing MSL
//! shaders nearly as-is via the `metal` crate, matching how Mference itself
//! compiles them from source at startup). On other platforms this crate
//! exposes nothing, so `cargo build --workspace` / `cargo test --workspace`
//! still succeed on Linux CI; only macOS gets the real implementation.
//!
//! Status: parity-tested dispatches (each against a `turbospark_compute`
//! CPU reference on real hardware): `rmsnorm_no_scale`,
//! `rope_proportional_neox`, `logit_softcap_fp16` (the port-local
//! cap-without-softmax head; `logit_softcap_softmax` stays vendored and
//! parity-tested but undispatched -- see `DEVIATIONS.md`),
//! `dequant_int4_gemv_simd` (staged, resident-offset-bound, and
//! encoder-level forms), `dequant_int8_gemv_simd`, the two-pass split-KV
//! decode attention (linear and ring KV addressing),
//! `utility.metal`'s elementwise kernels
//! (gelu/silu mul, residual add, scalar mul, logit softcap), and
//! `moe.metal`'s decode pair
//! (`moe_phase1_gate_up_act_u16load` + `moe_phase2_down_reduce_k8`,
//! reading expert blobs zero-copy through a `RoutedBlobs` argument
//! buffer). The memory-model pieces are production-wired:
//! `ResidentGpuWeights` (one `newBufferWithBytesNoCopy` MTLBuffer over
//! the resident mmap), `wrap_page_aligned_no_copy` (streamer slot wrap),
//! `KvCacheManager` (persistent K/V, `RealForwardRunner`'s cache),
//! `PassEncoder` (many kernels per command buffer, serial encoder), and
//! the pipeline cache keyed by function constants. Still vendored-only or
//! absent: the `sample` kernel and fused lm_head (no CPU reference — see
//! `DEVIATIONS.md`), `moe.metal`'s routers/top-k selectors/INT2 family,
//! the chunked-prefill tile pipeline (scratch allocated, kernel
//! descoped), `fused.metal`, and the GDN/DSV4 compute kernels (state
//! managers only).

#[cfg(target_os = "macos")]
mod attention_decode;
#[cfg(target_os = "macos")]
mod bytes;
#[cfg(target_os = "macos")]
mod context;
#[cfg(target_os = "macos")]
mod dequant_1bit_gemv;
#[cfg(target_os = "macos")]
mod dequant_2bit_gemv;
#[cfg(target_os = "macos")]
mod dequant_int4_batch;
#[cfg(target_os = "macos")]
mod dequant_int4_gemv;
#[cfg(target_os = "macos")]
mod dequant_int8_gemv;
#[cfg(target_os = "macos")]
mod dequant_iq_gemv;
#[cfg(target_os = "macos")]
mod dequant_q4_k_gemv;
#[cfg(target_os = "macos")]
mod dequant_q5_k_gemv;
#[cfg(target_os = "macos")]
mod dequant_q6_k_gemv;
#[cfg(target_os = "macos")]
mod dequant_q8_0_gemv;
#[cfg(target_os = "macos")]
mod device_memory;
#[cfg(target_os = "macos")]
mod dflash_conv;
#[cfg(target_os = "macos")]
mod dispatch_profile;
#[cfg(target_os = "macos")]
mod dsv4_state;
#[cfg(target_os = "macos")]
mod gdn;
#[cfg(target_os = "macos")]
mod gdn_shape;
#[cfg(target_os = "macos")]
mod gdn_state;
#[cfg(target_os = "macos")]
mod kv_cache;
#[cfg(target_os = "macos")]
mod kv_cache_mem;
#[cfg(target_os = "macos")]
mod logit_softmax;
#[cfg(target_os = "macos")]
mod moe_decode;
#[cfg(target_os = "macos")]
mod moe_gguf;
#[cfg(target_os = "macos")]
mod moe_prefill_batch;
#[cfg(target_os = "macos")]
mod moe_prefill_batch_gguf;
#[cfg(target_os = "macos")]
mod power_state;
#[cfg(target_os = "macos")]
mod prefill_scratch;
#[cfg(target_os = "macos")]
mod resident_metal;
#[cfg(target_os = "macos")]
mod rms_norm;
#[cfg(target_os = "macos")]
mod rope;
#[cfg(target_os = "macos")]
mod utility;
#[cfg(target_os = "macos")]
mod vision;

#[cfg(target_os = "macos")]
pub use attention_decode::{
    attention_decode, attention_decode_buffers, encode_attention_decode, AttentionScratch,
};
#[cfg(target_os = "macos")]
pub use bytes::{read_f32_buffer, read_f32_buffer_at};
#[cfg(target_os = "macos")]
pub use context::{
    autorelease_pool, dispatch_one_threadgroup_per_row, dispatch_one_threadgroup_per_row_offsets,
    dispatch_threads_3d, read_buffer_bytes, read_buffer_f16, read_buffer_f16_into,
    write_buffer_bytes, CommittedPass, GpuError, MetalContext, PassEncoder,
};
#[cfg(target_os = "macos")]
pub use dequant_1bit_gemv::{
    dequant_int1_gemv, dequant_int1_gemv_resident, dequant_int1_gemv_symmetric,
    encode_dequant_int1_gemv_resident, encode_embed_lookup_int1, int1_row_bytes, Int1AffineRowGpu,
    Int1ResidentMatrix, Int1SymmetricRowGpu,
};
#[cfg(target_os = "macos")]
pub use dequant_2bit_gemv::{
    dequant_int2_gemv, dequant_int2_gemv_resident, encode_dequant_int2_gemv_resident,
    encode_embed_lookup_int2, int2_row_bytes, Int2AffineRowGpu, Int2ResidentMatrix,
};
#[cfg(target_os = "macos")]
pub use dequant_int4_batch::{
    best_row_block, dequant_int4_gemm_pipeline_limits, encode_dequant_int4_gemm_mma_resident,
    encode_dequant_int4_gemm_mma_resident_skip_dequant,
    encode_dequant_int4_gemm_mma_resident_staged, encode_dequant_int4_gemm_resident,
    encode_dequant_int4_gemm_resident_blocked, GemmPipelineLimits, GEMM_THREADS_PER_GROUP,
    MAX_BATCH_ROWS, MAX_GEMM_ROW_BLOCK, MMA_MAX_BATCH_ROWS,
};
#[cfg(target_os = "macos")]
pub use dequant_int4_gemv::{
    dequant_int4_gemv, dequant_int4_gemv_resident, encode_dequant_int4_gemv_resident,
    encode_embed_lookup_int4, Int4AffineRowGpu, Int4ResidentMatrix,
};
#[cfg(target_os = "macos")]
pub use dequant_int8_gemv::{
    dequant_int8_gemv, dequant_int8_gemv_resident, encode_dequant_int8_gemv_resident,
    Int8AffineRowGpu, Int8ResidentMatrix,
};
#[cfg(target_os = "macos")]
pub use dequant_iq_gemv::{
    dequant_iq_gemv, IqBlockType, IQ3_XXS_BLOCK_BYTES, IQ3_XXS_BLOCK_ELEMS, IQ4_NL_BLOCK_BYTES,
    IQ4_NL_BLOCK_ELEMS, IQ4_XS_BLOCK_BYTES, IQ4_XS_BLOCK_ELEMS,
};
#[cfg(target_os = "macos")]
pub use dequant_q4_k_gemv::{
    dequant_q4_k_gemv, dequant_q4_k_gemv_resident, encode_dequant_q4_k_gemv_resident,
    encode_embed_lookup_q4_k, q4_k_row_bytes, Q4KResidentMatrix, Q4_K_BLOCK_BYTES,
    Q4_K_BLOCK_ELEMS,
};
#[cfg(target_os = "macos")]
pub use dequant_q5_k_gemv::{
    dequant_q5_k_gemv, dequant_q5_k_gemv_resident, encode_dequant_q5_k_gemv_resident,
    q5_k_row_bytes, Q5KResidentMatrix, Q5_K_BLOCK_BYTES, Q5_K_BLOCK_ELEMS,
};
#[cfg(target_os = "macos")]
pub use dequant_q6_k_gemv::{
    dequant_q6_k_gemv, dequant_q6_k_gemv_resident, encode_dequant_q6_k_gemv_resident,
    encode_embed_lookup_q6_k, q6_k_row_bytes, Q6KResidentMatrix, Q6_K_BLOCK_BYTES,
    Q6_K_BLOCK_ELEMS,
};
#[cfg(target_os = "macos")]
pub use dequant_q8_0_gemv::{
    dequant_q8_0_gemv, dequant_q8_0_gemv_resident, encode_dequant_q8_0_gemv_resident,
    encode_embed_lookup_q8_0, q8_0_row_bytes, Q8_0ResidentMatrix, Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_ELEMS,
};
#[cfg(target_os = "macos")]
pub use device_memory::recommended_max_working_set;
#[cfg(target_os = "macos")]
pub use dflash_conv::{
    encode_dflash_copy_rows, encode_dflash_grouped_conv, DFLASH_GROUP_SIZE, DFLASH_TAPS,
};
#[cfg(target_os = "macos")]
pub use dispatch_profile::{report as dispatch_profile_report, reset as dispatch_profile_reset};
#[cfg(target_os = "macos")]
pub use dsv4_state::{Dsv4StateManager, LayerCounters};
#[cfg(target_os = "macos")]
pub use gdn::{
    encode_gdn_conv_decode, encode_gdn_conv_prefill, encode_gdn_conv_tail_update,
    encode_gdn_delta_decode, encode_gdn_delta_prefill, encode_gdn_gated_norm, encode_gdn_in_proj,
    encode_gdn_qk_norm, GdnShape,
};
#[cfg(target_os = "macos")]
pub use gdn_state::{GdnSnapshot, GdnStateManager};
#[cfg(target_os = "macos")]
pub use kv_cache::{KvCacheManager, KvView, LayerKind};
#[cfg(target_os = "macos")]
pub use logit_softmax::{encode_logit_softcap_softmax, logit_softcap_softmax};
#[cfg(target_os = "macos")]
pub use moe_decode::{
    encode_moe_phase1, encode_moe_phase2, encode_router_gemv_gemma4, router_gemv_gemma4,
    MoeExpertOffsets, RoutedBlobsBuffer, MAX_STREAMED_EXPERTS,
};
#[cfg(target_os = "macos")]
pub use moe_gguf::{
    encode_moe_phase1_iq3_xxs, encode_moe_phase1_iq4_xs, encode_moe_phase1_mxfp4,
    encode_moe_phase1_q4_k, encode_moe_phase1_q8_0, encode_moe_phase2_iq4_nl,
    encode_moe_phase2_mxfp4, encode_moe_phase2_q4_k, encode_moe_phase2_q6_k,
    encode_moe_phase2_q8_0, mxfp4_row_bytes, Mxfp4Activation, MXFP4_BLOCK_BYTES, MXFP4_BLOCK_ELEMS,
};
#[cfg(target_os = "macos")]
pub use moe_prefill_batch::{
    encode_moe_prefill_phase1, encode_moe_prefill_phase2_fused, MoePrefillRoute,
    RoutedBlobsWideBuffer, MAX_PREFILL_EXPERT_BINDINGS,
};
#[cfg(target_os = "macos")]
pub use moe_prefill_batch_gguf::{
    bind_routed_blobs_wide as bind_routed_blobs_wide_mxfp4, encode_moe_prefill_phase1_mxfp4,
    encode_moe_prefill_phase2_fused_mxfp4, new_routed_blobs_wide as new_routed_blobs_wide_mxfp4,
};
#[cfg(target_os = "macos")]
pub use power_state::{
    low_power_mode_enabled, memory_pressure_raw, physical_memory, thermal_state_raw,
};
#[cfg(target_os = "macos")]
pub use prefill_scratch::{PrefillChunkScratchBuffers, PrefillChunkScratchLayout};
#[cfg(target_os = "macos")]
pub use resident_metal::{wrap_page_aligned_no_copy, ResidentGpuWeights};
#[cfg(target_os = "macos")]
pub use rms_norm::{
    encode_rms_norm_bf16w, encode_rms_norm_bf16w_centered, encode_rms_norm_bf16w_perhead,
    encode_rms_norm_bf16w_perhead_centered, encode_rms_norm_no_scale,
    encode_rms_norm_no_scale_perhead, rms_norm_bf16w_perhead, rms_norm_no_scale,
    rms_norm_no_scale_perhead,
};
#[cfg(target_os = "macos")]
pub use rope::{
    encode_rope_mrope_interleaved, encode_rope_neox_freqs, encode_rope_neox_subdim,
    encode_rope_proportional_neox, rope_mrope_interleaved, rope_neox_subdim,
    rope_proportional_neox,
};
#[cfg(target_os = "macos")]
pub use utility::{
    encode_bias_add, encode_gelu_mul, encode_logit_softcap, encode_residual_add, encode_scalar_mul,
    encode_sigmoid_gate_mul, encode_sigmoid_scalar_mul, encode_silu_mul, encode_split_q_gate,
    encode_steer_direction, SteerParams,
};
#[cfg(target_os = "macos")]
pub use vision::{
    encode_vision_attention, encode_vision_gelu, encode_vision_layer_norm, encode_vision_matmul,
    encode_vision_residual_add, encode_vision_rope_2d, GeluKind, MAX_ATTENTION_HEAD_DIM,
};

/// The Metal buffer handle, re-exported so downstream crates (e.g.
/// `crates/runtime`) can hold scratch buffers without their own `metal`
/// dependency.
#[cfg(target_os = "macos")]
pub use metal::Buffer as MetalBuffer;

// Token id width consumed from the core primitives, keeping the dependency
// edge live and documenting the interchange type this crate uses throughout.
pub use foundation::TokenId;
