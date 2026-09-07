//! Destination-selected compute strategy plus CPU reference kernels.
//!
//! The kernel modules (`rms_norm`, `wht`, `rope`, `attention`, `quant`,
//! `quant_1bit`, `quant_2bit`, `quant_gguf`, `quant_gguf_iq`,
//! `moe`, `gdn`, `gating`, `sampling`, `tolerance`) are the numerical ground truth later GPU
//! kernels are validated against. Numerics parity with any upstream
//! implementation is out of scope for `ComputeStrategy` itself; only the
//! structural contracts of the decode and prefill areas are exercised there.
//!
//! The `turbospark-core` dependency is brought in under the alias `foundation` to avoid
//! colliding with the standard library `core` crate in the extern prelude.
#![forbid(unsafe_code)]

/// Causal multi-head and multi-query attention compute kernels.
pub mod attention;
/// BERT and XLM-RoBERTa encoder compute kernels.
pub mod encoder;
/// Gating and activation function compute kernels.
pub mod gating;
/// Gated-DeltaNet (GDN) linear-attention compute kernels.
pub mod gdn;
/// `qwen4_exp`'s hyper-connection mix (PORT-LOCAL; not `HyperConnectionConfig`'s
/// Sinkhorn-normalised mHC).
pub mod hyper_connection;
/// TurboQuant KV-cache codec: per-row norm, randomized Hadamard rotation,
/// Lloyd-Max codebook quantization (ported from mlx-vlm's
/// `_TurboQuantMSECodec`, see `docs/TRUBOQUANT.md`).
pub mod kv_quant;
/// CPU reference for causal attention over TurboQuant-quantized K/V rows.
pub mod kv_quant_attention;
/// Mixture-of-Experts (MoE) routing and FFN compute kernels.
pub mod moe;
/// `qwen4_exp`'s PLE (per-layer n-gram embedding) gate (PORT-LOCAL).
pub mod ple;
/// `qwen4_exp`'s QSA block indexer: pooling, scoring and top-k block
/// selection (PORT-LOCAL, groundwork -- no decode flow reads this yet).
pub mod qsa_indexer;
/// Quantization and dequantization primitives for affine INT4 and INT8 formats.
pub mod quant;
/// Quantization and dequantization primitives for the affine 1-bit format.
pub mod quant_1bit;
/// Quantization and dequantization primitives for the affine 2-bit format.
pub mod quant_2bit;
/// Quantization and dequantization primitives for GGUF Q4_K, Q6_K, and Q8_0 formats.
pub mod quant_gguf;
/// Quantization and dequantization primitives for GGUF IQ3_XXS, IQ4_XS, and IQ4_NL formats.
pub mod quant_gguf_iq;
/// Look up tables for IQ3_XXS and IQ4_NL quantization formats.
pub mod quant_gguf_iq_tables;
/// Dequantization primitives for the GGUF MXFP4 format (ROADMAP M5).
pub mod quant_gguf_mxfp4;
/// Root-Mean-Square Normalization (RMSNorm) compute kernels.
pub mod rms_norm;
/// Rotary Position Embedding (RoPE) compute kernels.
pub mod rope;
/// Softmax sampling and logit soft-capping compute kernels.
pub mod sampling;
/// Directional steering of a residual stream row (ablate, add, clamp,
/// renorm).
pub mod steering;
/// Numerical error measurement and tolerance checking utilities.
pub mod tolerance;

/// FP32 reference kernels for the qwen3_5 vision tower (ROADMAP M-V2).
pub mod vision;
/// Walsh-Hadamard Transform (WHT) compute kernels.
pub mod wht;

pub use attention::{causal_attention, causal_attention_with_sinks, indexed_attention};
pub use encoder::{
    cls_pool_and_normalize, cosine_similarity, encoder_block_forward, encoder_embeddings_lookup,
    EncoderLayerWeights, EncoderReferenceConfig,
};
pub use gating::{sigmoid_gate_mul, sigmoid_scalar_mul, split_q_gate};
pub use gdn::{gated_norm_sigmoid, sigmoid, silu, softplus, GdnDims, GdnReference, GDN_RMS_EPS};
pub use hyper_connection::{hc_inject_add, hc_mix};
pub use kv_quant::{
    codebook, dequantize_row, midpoints, pack_lsb_first, packed_words, quantize_index,
    quantize_row, rht_forward, rht_inverse, sign_vector, unpack_lsb_first, QuantizedRow, KEY_SEED,
    NORM_EPS, VALUE_SEED,
};
pub use kv_quant_attention::{causal_attention_tq, TqTables};
pub use moe::{apply_streamed_routed, gelu_tanh, run_ffn};
pub use ple::{dequant_ngram_row, dilated_conv_step, ple_gate};
pub use qsa_indexer::{pool_blocks_mean, score_blocks, select_blocks};
pub use quant::{
    bf16_to_f32, dequant_int4_gemv, dequant_int8_gemv, dequantize_int4_affine,
    dequantize_int8_affine, embed_lookup_int4, embed_lookup_int8, f32_to_bf16,
    quantize_int4_affine, quantize_int8_affine, Int4AffineRow, Int8AffineRow,
};
pub use quant_1bit::{
    asymmetric_group_count, dequant_int1_gemv, dequant_int1_gemv_symmetric, dequantize_int1_affine,
    embed_lookup_int1, f16_to_f32, f32_to_f16, is_symmetric, quantize_int1_affine_symmetric,
    Int1AffineRow, BONSAI_GROUP_SIZE,
};
// `f16_to_f32` / `f32_to_f16` are NOT re-exported here: `quant_2bit` shares
// `quant_1bit`'s pair rather than carrying a second copy, so re-exporting
// them twice at the crate root would be a name collision stating one fact.
pub use quant_2bit::{
    asymmetric_group_count_int2, dequant_int2_gemv, dequantize_int2_affine, embed_lookup_int2,
    is_ternary_symmetric, quantize_int2_affine_ternary, uses_fourth_level, Int2AffineRow,
    INT2_ELEMENTS_PER_BYTE, TERNARY_GROUP_SIZE,
};
pub use quant_gguf::{
    dequant_q4_k_gemv, dequant_q5_k_gemv, dequant_q6_k_gemv, dequant_q8_0_gemv, dequantize_q4_k,
    dequantize_q5_k, dequantize_q6_k, dequantize_q8_0, pearson, quantize_q4_k, quantize_q6_k,
    quantize_q8_0, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q4_K_SUB_ELEMS, Q5_K_BLOCK_BYTES,
    Q5_K_BLOCK_ELEMS, Q5_K_SUB_ELEMS, Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS, Q6_K_SUB_ELEMS,
    Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS,
};
pub use quant_gguf_iq::{
    dequant_iq3_xxs_gemv, dequant_iq4_nl_gemv, dequant_iq4_xs_gemv, dequantize_iq3_xxs,
    dequantize_iq4_nl, dequantize_iq4_xs, iq3xxs_signs, IQ3_XXS_BLOCK_BYTES, IQ3_XXS_BLOCK_ELEMS,
    IQ3_XXS_SUB_ELEMS, IQ4_NL_BLOCK_BYTES, IQ4_NL_BLOCK_ELEMS, IQ4_XS_BLOCK_BYTES,
    IQ4_XS_BLOCK_ELEMS, IQ4_XS_SUB_ELEMS,
};
pub use quant_gguf_iq_tables::{IQ3XXS_GRID, IQ4NL_VALUES};
pub use quant_gguf_mxfp4::{
    dequant_mxfp4_gemv, dequantize_mxfp4, mxfp4_scale, MXFP4_BLOCK_BYTES, MXFP4_BLOCK_ELEMS,
    MXFP4_VALUES,
};
pub use rms_norm::{rms_norm, rms_norm_centered, rms_norm_grouped_centered};
pub use rope::{
    mrope_component_selector, rope_mrope_interleaved, rope_neox, rope_neox_subdim, rope_paired,
    yarn_frequencies, YarnSpec,
};
pub use sampling::logit_softcap_softmax;
pub use steering::{
    direction_coefficient, inv_norm, renorm_gamma, steer_in_place, unit_coefficient,
};
pub use tolerance::{bounded_rel_error, max_abs_diff, rel_error, Tolerance};
pub use vision::{
    attention_scale, bidirectional_attention, gelu_erf, gelu_tanh_vision, layer_norm, matmul_bias,
    rope_vision_2d,
};
pub use wht::wht;

/// Marker for the destination-selected compute strategy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ComputeStrategy {
    _private: (),
}

impl ComputeStrategy {
    /// Construct the default compute strategy.
    pub fn new() -> Self {
        Self::default()
    }
}

// Token id width consumed from the core primitives, keeping the dependency
// edge live and documenting the interchange type used in later slices.
pub use foundation::TokenId;
