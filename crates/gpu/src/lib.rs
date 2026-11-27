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
//! the pipeline cache keyed by function constants. `moe.metal`'s router
//! GEMV (`router_gemv_gemma4_r4`, INT8-weight) IS dispatched, across nearly
//! every family; its top-k selection stays host-side (the runtime's own
//! top-k, not this kernel). Still vendored-only or absent: the `sample`
//! kernel and fused lm_head (no CPU reference -- see `DEVIATIONS.md`),
//! `moe.metal`'s parallel top-k selector kernels and the whole DeepSeek-V4
//! INT2 family, the chunked-prefill tile pipeline (scratch allocated,
//! kernel descoped), `fused.metal`, and the DSV4 compute kernels (state
//! manager only).

#[cfg(all(target_os = "macos", feature = "ane-prefill"))]
pub mod ane_split;
#[cfg(target_os = "macos")]
mod attention_decode;
#[cfg(target_os = "macos")]
mod attention_indexed;
#[cfg(target_os = "macos")]
mod attention_tq;
#[cfg(target_os = "macos")]
mod bytes;
#[cfg(target_os = "macos")]
mod context;
// Both batch-GEMM modules import `metal`/`half` unconditionally, so on a
// non-macOS build they must compile to nothing like their GEMV siblings --
// leaving the gate off here broke the cross-target check from the day the
// modules landed (AGENTS.md/CLAUDE.md Gotcha 8: a mod without its gate).
#[cfg(target_os = "macos")]
mod conformer_stream;
#[cfg(target_os = "macos")]
mod dequant_1bit_gemm_batch;
#[cfg(target_os = "macos")]
mod dequant_1bit_gemv;
#[cfg(target_os = "macos")]
mod dequant_2bit_gemm_batch;
#[cfg(target_os = "macos")]
mod dequant_2bit_gemv;
#[cfg(target_os = "macos")]
mod dequant_ext_gemv;
#[cfg(target_os = "macos")]
mod dequant_int4_batch;
#[cfg(target_os = "macos")]
mod dequant_int4_fused;
#[cfg(target_os = "macos")]
mod dequant_int4_gemv;
#[cfg(target_os = "macos")]
mod dequant_int4_mma8;
#[cfg(target_os = "macos")]
mod dequant_int4_steel;
#[cfg(target_os = "macos")]
mod dequant_int8_batch;
#[cfg(target_os = "macos")]
mod dequant_int8_gemv;
#[cfg(target_os = "macos")]
mod dequant_iq_gemv;
#[cfg(target_os = "macos")]
mod dequant_kquant_gemm_batch;
#[cfg(target_os = "macos")]
mod dequant_legacy_gemv;
#[cfg(target_os = "macos")]
mod dequant_q2_k_gemv;
#[cfg(target_os = "macos")]
mod dequant_q3_k_gemv;
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
mod gemv_bf16;
#[cfg(target_os = "macos")]
mod hadamard;
#[cfg(target_os = "macos")]
mod hyper_connection;
#[cfg(target_os = "macos")]
mod inkling2_batch;
#[cfg(target_os = "macos")]
mod inkling2_dspark;
#[cfg(target_os = "macos")]
mod inkling2_ops;
#[cfg(target_os = "macos")]
mod kv_cache;
#[cfg(target_os = "macos")]
mod kv_cache_mem;
#[cfg(target_os = "macos")]
mod kv_quant_tables;
#[cfg(target_os = "macos")]
mod kv_quantize;
#[cfg(target_os = "macos")]
mod lamq;
#[cfg(target_os = "macos")]
pub mod launch_count;
#[cfg(target_os = "macos")]
mod logit_softmax;
#[cfg(target_os = "macos")]
mod mla;
#[cfg(target_os = "macos")]
mod moe_decode;
#[cfg(target_os = "macos")]
mod moe_gguf;
#[cfg(target_os = "macos")]
mod moe_inkling2;
#[cfg(target_os = "macos")]
mod moe_inkling2_lamq;
#[cfg(target_os = "macos")]
mod moe_prefill_batch;
#[cfg(target_os = "macos")]
mod moe_prefill_batch_gguf;
#[cfg(target_os = "macos")]
mod moonshine_frontend;
#[cfg(target_os = "macos")]
mod nemotron_diar;
#[cfg(target_os = "macos")]
mod ple;
#[cfg(target_os = "macos")]
mod power_state;
#[cfg(target_os = "macos")]
mod precompiled;
#[cfg(target_os = "macos")]
mod prefill_scratch;
#[cfg(target_os = "macos")]
mod qsa_indexer;
#[cfg(target_os = "macos")]
mod qsa_indexer_state;
#[cfg(target_os = "macos")]
mod qsa_topk;
#[cfg(target_os = "macos")]
mod resident_metal;
#[cfg(target_os = "macos")]
mod rms_norm;
#[cfg(target_os = "macos")]
mod rope;
#[cfg(target_os = "macos")]
mod short_conv;
#[cfg(target_os = "macos")]
mod short_conv_state;
#[cfg(target_os = "macos")]
pub mod submit_probe;
/// Lab-only idle-stall probes (KERN-05 Step 1a).
#[cfg(target_os = "macos")]
pub mod touch;
#[cfg(target_os = "macos")]
mod utility;
#[cfg(target_os = "macos")]
mod vision;
#[cfg(target_os = "macos")]
mod warmup;
#[cfg(target_os = "macos")]
mod whisper_conv;
#[cfg(target_os = "macos")]
mod whisper_encoder;
#[cfg(target_os = "macos")]
pub use warmup::{KernelWarmup, KernelWarmupPlan, KernelWarmupStats};

#[cfg(target_os = "macos")]
pub use attention_decode::{
    attention_decode, attention_decode_gqa_supported, encode_attention_decode,
    encode_attention_decode_batch, encode_attention_decode_gqa, encode_attention_decode_gqa_rows,
    encode_attention_decode_gqa_rows_with, encode_attention_decode_tree, AttentionScratch,
    BatchAttentionInputContract, BatchAttentionKvFormat, BatchAttentionKvLayout,
    BatchAttentionScratch, DecodeAttentionKernel, GqaRowsScratch, ATTENTION_TREE_MAX_DEPTH,
    GQA_ROWS_SCRATCH_CHUNKS, MAX_DECODE_ATTENTION_HEAD_DIM, MAX_GQA_GROUP,
};
#[cfg(target_os = "macos")]
pub use attention_indexed::{
    attention_decode_indexed, attention_decode_indexed_biased, encode_attention_decode_indexed,
    encode_attention_decode_indexed_biased, encode_attention_decode_indexed_from_count,
};
#[cfg(target_os = "macos")]
pub use attention_tq::{
    encode_attention_decode_indexed_tq, encode_attention_decode_indexed_tq_from_count,
    encode_attention_decode_tq, TqAttentionScratch,
};
#[cfg(target_os = "macos")]
pub use bytes::{half_slice_to_le_bytes, read_f32_buffer, read_f32_buffer_at, read_half_buffer};
#[cfg(target_os = "macos")]
pub use conformer_stream::{
    encode_copy as encode_conformer_copy,
    encode_depthwise_causal as encode_conformer_depthwise_causal,
    encode_glu_extend as encode_conformer_glu_extend,
    encode_relpos_attention as encode_conformer_relpos_attention,
    encode_silu as encode_conformer_silu, RelPosAttention as ConformerRelPosAttention,
    MAX_STREAM_HEAD_DIM, MAX_STREAM_KEYS,
};
#[cfg(target_os = "macos")]
pub use context::{
    autorelease_pool, dispatch_one_threadgroup_per_row, dispatch_one_threadgroup_per_row_offsets,
    dispatch_threads_3d, read_buffer_bytes, read_buffer_f16, read_buffer_f16_into,
    write_buffer_bytes, CommittedPass, GpuError, MetalCompilationStats, MetalContext, PassEncoder,
};
#[cfg(target_os = "macos")]
pub use dequant_1bit_gemm_batch::{dequant_int1_gemm_resident, encode_dequant_int1_gemm_resident};
#[cfg(target_os = "macos")]
pub use dequant_1bit_gemv::{
    dequant_int1_gemv, dequant_int1_gemv_resident, dequant_int1_gemv_symmetric,
    encode_dequant_int1_gemv_resident, encode_embed_lookup_int1, int1_row_bytes, Int1AffineRowGpu,
    Int1ResidentMatrix, Int1SymmetricRowGpu,
};
#[cfg(target_os = "macos")]
pub use dequant_2bit_gemm_batch::{dequant_int2_gemm_resident, encode_dequant_int2_gemm_resident};
#[cfg(target_os = "macos")]
pub use dequant_2bit_gemv::{
    dequant_int2_gemv, dequant_int2_gemv_resident, encode_dequant_int2_gemv_resident,
    encode_embed_lookup_int2, int2_row_bytes, Int2AffineRowGpu, Int2ResidentMatrix,
};
#[cfg(target_os = "macos")]
pub use dequant_ext_gemv::{
    dequant_ext_gemv_resident, encode_dequant_ext_gemv_resident, encode_embed_lookup_ext,
    ExtBlockType, ExtResidentMatrix, IQ1_NARROW_BLOCK_ELEMS, IQ1_XS_BLOCK_BYTES,
    IQ1_XXS_BLOCK_BYTES, IQ1_XXXS_BLOCK_BYTES, NVFP4_BLOCK_BYTES, NVFP4_BLOCK_ELEMS,
    Q1_0_BLOCK_BYTES, Q1_0_BLOCK_ELEMS, Q2_0_BLOCK_BYTES, Q2_0_BLOCK_ELEMS, TQ1_0_BLOCK_BYTES,
    TQ1_0_BLOCK_ELEMS, TQ2_0_BLOCK_BYTES, TQ2_0_BLOCK_ELEMS,
};
#[cfg(target_os = "macos")]
pub use dequant_int4_batch::{
    best_row_block, dequant_int4_gemm_mma_pipeline_limits, dequant_int4_gemm_pipeline_limits,
    encode_dequant_int4_gemm_mma_resident, encode_dequant_int4_gemm_mma_resident_skip_dequant,
    encode_dequant_int4_gemm_mma_resident_staged, encode_dequant_int4_gemm_mma_resident_wide,
    encode_dequant_int4_gemm_mma_resident_wide_skip_dequant, encode_dequant_int4_gemm_resident,
    encode_dequant_int4_gemm_resident_blocked, GemmPipelineLimits, GEMM_THREADS_PER_GROUP,
    MAX_BATCH_ROWS, MAX_GEMM_ROW_BLOCK, MMA_MAX_BATCH_ROWS,
};
#[cfg(target_os = "macos")]
pub use dequant_int4_fused::{encode_down_epi_gemv, encode_gate_up_act_gemv, DownEpilogue};
#[cfg(target_os = "macos")]
pub use dequant_int4_gemv::{
    dequant_int4_gemv, dequant_int4_gemv_resident, encode_dequant_int4_gemv_resident,
    encode_embed_lookup_int4, Int4AffineRowGpu, Int4ResidentMatrix,
};
#[cfg(target_os = "macos")]
pub use dequant_int4_mma8::{
    encode_dequant_int4_mma8_resident, int4_mma8_k_split, INT4_MMA8_MAX_ROWS,
};
#[cfg(target_os = "macos")]
pub use dequant_int4_steel::{
    encode_dequant_int4_gemm_small_n, encode_dequant_int4_gemm_steel,
    encode_dequant_int4_gemm_steel_edge, small_n_supports, steel_edge_supports, steel_supports,
    SteelTile, SMALL_N_MAX_ROWS, SMALL_N_MMA8_MAX_BATCH,
};
#[cfg(target_os = "macos")]
pub use dequant_int8_batch::{
    best_row_block_int8, encode_dequant_int8_gemm_resident,
    encode_dequant_int8_gemm_resident_blocked, INT8_MAX_BATCH_ROWS, INT8_MAX_GEMM_ROW_BLOCK,
};
#[cfg(target_os = "macos")]
pub use dequant_int8_gemv::{
    dequant_int8_gemv, dequant_int8_gemv_resident, encode_dequant_int8_gemv_resident,
    encode_embed_lookup_int8, Int8AffineRowGpu, Int8ResidentMatrix,
};
#[cfg(target_os = "macos")]
pub use dequant_iq_gemv::{
    dequant_iq_gemv, encode_dequant_iq_gemv_resident, encode_embed_lookup_iq1_m,
    encode_embed_lookup_iq4_xs, IqBlockType, IqResidentMatrix, IQ1_M_BLOCK_BYTES,
    IQ1_M_BLOCK_ELEMS, IQ1_S_BLOCK_BYTES, IQ1_S_BLOCK_ELEMS, IQ2_S_BLOCK_BYTES, IQ2_S_BLOCK_ELEMS,
    IQ2_XS_BLOCK_BYTES, IQ2_XS_BLOCK_ELEMS, IQ2_XXS_BLOCK_BYTES, IQ2_XXS_BLOCK_ELEMS,
    IQ3_S_BLOCK_BYTES, IQ3_S_BLOCK_ELEMS, IQ3_XXS_BLOCK_BYTES, IQ3_XXS_BLOCK_ELEMS,
    IQ4_NL_BLOCK_BYTES, IQ4_NL_BLOCK_ELEMS, IQ4_XS_BLOCK_BYTES, IQ4_XS_BLOCK_ELEMS,
};
#[cfg(target_os = "macos")]
pub use dequant_kquant_gemm_batch::{
    bf16_gemm_resident, dequant_kquant_gemm_resident, encode_bf16_gemm_resident,
    encode_dequant_kquant_gemm_resident, encode_dequant_q4_k_gemm_resident_choose, KQuantBatchType,
    KQuantResidentMatrix,
};
#[cfg(target_os = "macos")]
pub use dequant_legacy_gemv::{
    dequant_legacy_gemv_resident, encode_dequant_legacy_gemv_resident, encode_embed_lookup_legacy,
    LegacyBlockType, LegacyResidentMatrix, LEGACY_BLOCK_ELEMS, Q4_0_BLOCK_BYTES, Q4_1_BLOCK_BYTES,
    Q5_0_BLOCK_BYTES, Q5_1_BLOCK_BYTES,
};
#[cfg(target_os = "macos")]
pub use dequant_q2_k_gemv::{
    dequant_q2_k_gemv, encode_dequant_q2_k_gemv_resident, q2_k_row_bytes, Q2KResidentMatrix,
    Q2_K_BLOCK_BYTES, Q2_K_BLOCK_ELEMS,
};
#[cfg(target_os = "macos")]
pub use dequant_q3_k_gemv::{
    dequant_q3_k_gemv, dequant_q3_k_gemv_resident, encode_dequant_q3_k_gemv_resident,
    encode_embed_lookup_q3_k, q3_k_row_bytes, Q3KResidentMatrix, Q3_K_BLOCK_BYTES,
    Q3_K_BLOCK_ELEMS,
};
#[cfg(target_os = "macos")]
pub use dequant_q4_k_gemv::{
    dequant_q4_k_gemv, dequant_q4_k_gemv_resident, dequant_q4_k_gemv_resident_wide,
    dequant_q4_k_gemv_resident_wide_unroll, encode_dequant_q4_k_gemv_resident,
    encode_dequant_q4_k_gemv_resident_wide, encode_dequant_q4_k_gemv_resident_wide_unroll,
    encode_embed_lookup_q4_k, q4_k_row_bytes, q4_k_wide_selected, Q4KResidentMatrix,
    Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS,
};
#[cfg(target_os = "macos")]
pub use dequant_q5_k_gemv::{
    dequant_q5_k_gemv, dequant_q5_k_gemv_resident, encode_dequant_q5_k_gemv_resident,
    encode_embed_lookup_q5_k, q5_k_row_bytes, Q5KResidentMatrix, Q5_K_BLOCK_BYTES,
    Q5_K_BLOCK_ELEMS,
};
#[cfg(target_os = "macos")]
pub use dequant_q6_k_gemv::{
    dequant_q6_k_gemv, dequant_q6_k_gemv_resident, encode_dequant_q6_k_gemv_resident,
    encode_embed_lookup_q6_k, q6_k_row_bytes, Q6KResidentMatrix, Q6_K_BLOCK_BYTES,
    Q6_K_BLOCK_ELEMS,
};
#[cfg(target_os = "macos")]
pub use dequant_q8_0_gemv::{
    dequant_q8_0_gemv, dequant_q8_0_gemv_resident, dequant_q8_0_gemv_resident_wide,
    encode_dequant_q8_0_gemv_resident, encode_dequant_q8_0_gemv_resident_wide,
    encode_embed_lookup_q8_0, q8_0_row_bytes, Q8_0ResidentMatrix, Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_ELEMS,
};
#[cfg(target_os = "macos")]
pub use device_memory::{process_footprint, recommended_max_working_set};
#[cfg(target_os = "macos")]
pub use dflash_conv::{
    encode_copy_rows_fp16_strided, encode_dflash_copy_rows, encode_dflash_grouped_conv,
    encode_gather_rows_fp16, DFLASH_GROUP_SIZE, DFLASH_TAPS, GATHER_MAX_ROWS,
};
#[cfg(target_os = "macos")]
pub use dispatch_profile::{report as dispatch_profile_report, reset as dispatch_profile_reset};
#[cfg(target_os = "macos")]
pub use dsv4_state::{Dsv4StateManager, LayerCounters};
#[cfg(target_os = "macos")]
pub use gdn::{
    encode_gdn_conv_decode, encode_gdn_conv_prefill, encode_gdn_conv_prefill_tree,
    encode_gdn_conv_tail_update, encode_gdn_delta_decode, encode_gdn_delta_prefill,
    encode_gdn_delta_prefill_lazy, encode_gdn_delta_prefill_lazy_tree, encode_gdn_gated_norm,
    encode_gdn_gated_norm_sigmoid, encode_gdn_in_proj, encode_gdn_qk_norm,
    encode_gdn_qk_norm_with_rms_epsilon, encode_gdn_state_fold, gdn_tree_supported, GdnShape,
    GDN_TREE_MAX_ROWS,
};
#[cfg(target_os = "macos")]
pub use gdn_state::{GdnSnapshot, GdnStateManager};
#[cfg(target_os = "macos")]
pub use gemv_bf16::{encode_bf16_gemv_resident, encode_embed_lookup_bf16, Bf16ResidentMatrix};
#[cfg(target_os = "macos")]
pub use hadamard::{
    encode_hadamard_fwht, hadamard_shape_error, HADAMARD_HYBRID_THREADS, HADAMARD_MAX_BLOCK,
    HADAMARD_THREADS,
};
#[cfg(target_os = "macos")]
pub use hyper_connection::{encode_hc_inject_add, encode_hc_mix};
#[cfg(target_os = "macos")]
pub use inkling2_batch::{
    encode_inkling2_attention_prefill, encode_inkling2_attention_rows,
    encode_inkling2_bf16_gemm_rows_mma, encode_inkling2_bf16_gemv_rows,
    encode_inkling2_control_combine_rows, encode_inkling2_copy_rows,
    encode_inkling2_depth_mix_rows, encode_inkling2_l2_clip_rows, encode_inkling2_rel_table_rows,
    encode_inkling2_rms_rows, encode_inkling2_router_rows, encode_inkling2_router_select_rows,
    encode_inkling2_short_conv_rows, Inkling2GemvRowsSegment, Inkling2Rows,
    INKLING2_BATCH_MAX_ROWS, INKLING2_GEMV_MAX_ROWS,
};
#[cfg(target_os = "macos")]
pub use inkling2_dspark::{
    encode_dspark_attention, encode_dspark_bf16_to_half, encode_dspark_ctx_store,
    encode_dspark_init_x, encode_dspark_markov_argmax, encode_dspark_markov_gate,
    encode_dspark_markov_input, encode_dspark_markov_logits, encode_dspark_residual,
    encode_dspark_rms_f32, encode_dspark_rope_rows, encode_dspark_silu_mul,
    encode_dspark_tap_mix_norm, encode_dspark_transpose_bf16, DsparkAttnShape,
    DSPARK_LOGITS_PER_GROUP, DSPARK_MAX_RANK,
};
#[cfg(target_os = "macos")]
pub use inkling2_ops::{
    encode_inkling2_attention_decode, encode_inkling2_bf16_gemv_multi,
    encode_inkling2_bf16_gemv_multi2, encode_inkling2_control_combine, encode_inkling2_copy_half,
    encode_inkling2_depth_mix, encode_inkling2_depth_mix_norm, encode_inkling2_l2_clip,
    encode_inkling2_qkv_post, encode_inkling2_rel_table, encode_inkling2_router_select,
    encode_inkling2_short_conv_decode, encode_inkling2_short_conv_residual, Inkling2AttnShape,
    Inkling2ConvResidual, Inkling2GemvSegment, Inkling2QkvPostBuffers, Inkling2QkvPostShape,
    INKLING2_ATTN_MAX_HEAD_DIM, INKLING2_DEPTH_MAX_STATES, INKLING2_GEMV_MAX_SEGMENTS,
    INKLING2_MIX_NORM_MAX_H, INKLING2_ROUTER_MAX_EXPERTS, INKLING2_ROUTER_TOP_K,
};
#[cfg(target_os = "macos")]
pub use kv_cache::{KvCacheManager, LayerKind};
#[cfg(target_os = "macos")]
pub use kv_quant_tables::{KvQuantTables, TqSideTables};
#[cfg(target_os = "macos")]
pub use kv_quantize::encode_kv_quantize_tq;
#[cfg(target_os = "macos")]
pub use lamq::{
    encode_lamq_decode_row, encode_lamq_gemv, encode_lamq_gemv_rows, encode_lamq_rht,
    encode_lamq_rht_in_batch, encode_lamq_scatter, LamqMatrixView, LamqRhtDir, LamqRhtFactor,
    LAMQ_MAX_N, LAMQ_MAX_TABLE, LAMQ_RHT_BATCH_MAX,
};
#[cfg(target_os = "macos")]
pub use logit_softmax::{encode_logit_softcap_softmax, logit_softcap_softmax};
#[cfg(target_os = "macos")]
pub use mla::{
    encode_mla_absorb_q, encode_mla_attention_decode, encode_mla_cache_write, encode_mla_kv_norm,
    encode_mla_rope_q_pe, encode_mla_v_combine, mla_absorb_q, mla_attention_decode, mla_kv_norm,
    mla_rope_q_pe, mla_v_combine,
};
#[cfg(target_os = "macos")]
pub use moe_decode::{
    encode_moe_phase1, encode_moe_phase1_int3, encode_moe_phase2, encode_moe_phase2_int3,
    encode_router_gemv_gemma4, moe_decode_source, router_gemv_gemma4, MoeExpertOffsets,
    RoutedBlobsBuffer, MAX_STREAMED_EXPERTS,
};
#[cfg(target_os = "macos")]
pub use moe_gguf::{
    encode_inkling2_gguf_bind_rows, encode_inkling2_gguf_phase1, encode_inkling2_gguf_phase1_table,
    encode_inkling2_gguf_phase2, encode_inkling2_gguf_phase2_table, encode_moe_phase1_ext,
    encode_moe_phase1_iq1_m, encode_moe_phase1_iq2_s, encode_moe_phase1_iq2_xxs,
    encode_moe_phase1_iq3_xxs, encode_moe_phase1_iq4_xs, encode_moe_phase1_mxfp4,
    encode_moe_phase1_q2_0, encode_moe_phase1_q4_k, encode_moe_phase1_q8_0, encode_moe_phase2_ext,
    encode_moe_phase2_iq4_nl, encode_moe_phase2_mxfp4, encode_moe_phase2_q2_0_top10,
    encode_moe_phase2_q4_k, encode_moe_phase2_q6_k, encode_moe_phase2_q8_0, moe_gguf_source,
    mxfp4_row_bytes, Inkling2GgufExpertType, MoeExtType, Mxfp4Activation,
    INKLING2_GGUF_TABLE_BYTES, MXFP4_BLOCK_BYTES, MXFP4_BLOCK_ELEMS, PHASE2_FIXED_SLOTS,
};
#[cfg(target_os = "macos")]
pub use moe_inkling2::{
    encode_inkling2_moe_grouped_rows, encode_inkling2_moe_grouped_rows_mma,
    encode_inkling2_moe_phase1, encode_inkling2_moe_phase1_indexed,
    encode_inkling2_moe_phase1_indexed_rows, encode_inkling2_moe_phase2,
    encode_inkling2_moe_phase2_indexed, encode_inkling2_moe_phase2_indexed_rows,
    moe_inkling2_source, Inkling2ExpertType, Inkling2GroupScratch, Inkling2ResidentLayer,
    INKLING2_GROUP_MAX_EXPERTS, INKLING2_MAX_SLOTS, INKLING2_MOE_MAX_ROWS,
    INKLING2_OUTER_SCALE_BYTES, MOE_INKLING2_REFLECT_FUNCTION,
};
#[cfg(target_os = "macos")]
pub use moe_inkling2_lamq::{
    encode_inkling2_lamq_phase1, encode_inkling2_lamq_phase2, lamq_lut_pack,
    moe_inkling2_lamq_source, LamqExpertOffsets, LamqExperts, INKLING2_LAMQ_MAX_SLOTS,
    LAMQ_LUT_PACK, LAMQ_LUT_PACK_HALVES, MOE_INKLING2_LAMQ_REFLECT_FUNCTION,
};
#[cfg(target_os = "macos")]
pub use moe_prefill_batch::{
    encode_moe_prefill_phase1, encode_moe_prefill_phase1_wide, encode_moe_prefill_phase2_fused,
    encode_moe_prefill_phase2_fused_wide, MoePrefillRoute, RoutedBlobsWideBuffer,
    MAX_PREFILL_EXPERT_BINDINGS,
};
#[cfg(target_os = "macos")]
pub use moe_prefill_batch_gguf::{
    bind_routed_blobs_wide as bind_routed_blobs_wide_mxfp4, encode_moe_prefill_phase1_mxfp4,
    encode_moe_prefill_phase2_fused_mxfp4, new_routed_blobs_wide as new_routed_blobs_wide_mxfp4,
};
#[cfg(target_os = "macos")]
pub use moonshine_frontend::{
    encode_moonshine_conv1d, encode_moonshine_embed, encode_moonshine_groupnorm,
    encode_moonshine_rope, encode_moonshine_swiglu, encode_moonshine_tanh,
};
#[cfg(target_os = "macos")]
pub use nemotron_diar::encode_rope_neox_rows as encode_nemotron_rope_neox;
#[cfg(target_os = "macos")]
pub use ple::encode_ple_gate;
#[cfg(target_os = "macos")]
pub use power_state::{
    low_power_mode_enabled, memory_pressure_raw, physical_memory, thermal_state_raw,
};
#[cfg(target_os = "macos")]
pub use prefill_scratch::{
    BatchAttentionScratchLayout, BatchAttentionScratchLayoutError, PrefillChunkScratchBuffers,
    PrefillChunkScratchLayout, MAX_BATCH_ATTENTION_SCRATCH_BYTES,
};
#[cfg(target_os = "macos")]
pub use qsa_indexer::{
    encode_qsa_advance_blocks, encode_qsa_pool_blocks_mean, encode_qsa_score_blocks,
};
#[cfg(target_os = "macos")]
pub use qsa_indexer_state::QsaIndexerCacheManager;
#[cfg(target_os = "macos")]
pub use qsa_topk::encode_qsa_topk_positions;
#[cfg(target_os = "macos")]
pub use resident_metal::{wrap_page_aligned_no_copy, ResidentGpuWeights};
#[cfg(target_os = "macos")]
pub use rms_norm::{
    encode_rms_norm_bf16w, encode_rms_norm_bf16w_centered, encode_rms_norm_bf16w_grouped,
    encode_rms_norm_bf16w_grouped_centered, encode_rms_norm_bf16w_perhead,
    encode_rms_norm_bf16w_perhead_centered, encode_rms_norm_no_scale,
    encode_rms_norm_no_scale_perhead, rms_norm_bf16w_grouped_centered, rms_norm_bf16w_perhead,
    rms_norm_no_scale, rms_norm_no_scale_perhead,
};
#[cfg(target_os = "macos")]
pub use rope::{
    encode_rope_mrope_chunked, encode_rope_mrope_interleaved, encode_rope_neox_freqs,
    encode_rope_neox_subdim, encode_rope_neox_subdim_rows, encode_rope_neox_subdim_rows_positions,
    encode_rope_proportional_neox, encode_rope_proportional_neox_rows, rope_mrope_interleaved,
    rope_neox_subdim, rope_proportional_neox, ROPE_POSITION_ROWS_MAX,
};
#[cfg(target_os = "macos")]
pub use short_conv::{encode_short_conv_decode, encode_short_conv_prefill};
#[cfg(target_os = "macos")]
pub use short_conv_state::ShortConvStateManager;
#[cfg(target_os = "macos")]
pub use utility::{
    encode_bias_add, encode_gelu_erf_mul, encode_gelu_mul, encode_logit_softcap,
    encode_residual_add, encode_scalar_mul, encode_sigmoid, encode_sigmoid_gate_mul,
    encode_sigmoid_head_gate_mul, encode_sigmoid_scalar_mul, encode_silu, encode_silu_mul,
    encode_split_q_gate, encode_split_qkv, encode_steer_direction, SteerParams,
};
#[cfg(target_os = "macos")]
pub use vision::{
    encode_vision_attention, encode_vision_attention_relpos, encode_vision_attention_with,
    encode_vision_conv2d, encode_vision_gated_dwconv3x3, encode_vision_gather_rows,
    encode_vision_gelu, encode_vision_im2col, encode_vision_layer_norm, encode_vision_matmul,
    encode_vision_matmul_with, encode_vision_mul, encode_vision_quick_gelu,
    encode_vision_residual_add, encode_vision_rms_norm, encode_vision_rope_2d,
    encode_vision_rope_axial, encode_vision_sam_attention, encode_vision_silu,
    encode_vision_window_mean4_add, GeluKind, VisionAttentionKernel, VisionConvShape,
    VisionMatmulKernel, MAX_ATTENTION_HEAD_DIM, SAM_ATTENTION_HEAD_DIM, SAM_ATTENTION_MAX_KEYS,
    VISION_GATHER_ZERO,
};
#[cfg(target_os = "macos")]
pub use whisper_conv::{encode_whisper_conv1d3_gelu, whisper_conv1d3_gelu};
#[cfg(target_os = "macos")]
pub use whisper_encoder::{
    encode_add as encode_whisper_add, encode_attn_step as encode_whisper_attn_step,
    encode_gelu_erf as encode_whisper_gelu_erf, encode_gemv as encode_whisper_gemv,
    encode_layer_norm as encode_whisper_layer_norm,
    encode_matmul_bias as encode_whisper_matmul_bias,
    encode_softmax_rows as encode_whisper_softmax_rows,
    encode_transpose as encode_whisper_transpose,
    encode_transpose_pos as encode_whisper_transpose_pos, F32View, MAX_ATTN_STEP, ROW_THREADS,
};

/// The Metal buffer handle, re-exported so downstream crates (e.g.
/// `crates/runtime`) can hold scratch buffers without their own `metal`
/// dependency.
#[cfg(target_os = "macos")]
pub use metal::Buffer as MetalBuffer;

// Token id width consumed from the core primitives, keeping the dependency
// edge live and documenting the interchange type this crate uses throughout.
pub use foundation::TokenId;

#[cfg(target_os = "macos")]
mod minimax_router;
#[cfg(target_os = "macos")]
pub use minimax_router::encode_minimax_router;

#[cfg(target_os = "macos")]
mod music3;
#[cfg(target_os = "macos")]
pub use music3::{
    Music3Buffer, Music3DType, Music3Device, Music3Encoding, Music3Sequence, Music3Slot,
    Music3Table, Music3Weight,
};
