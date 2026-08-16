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
/// Gating and activation function compute kernels.
pub mod gating;
/// Gated-DeltaNet (GDN) linear-attention compute kernels.
pub mod gdn;
/// Mixture-of-Experts (MoE) routing and FFN compute kernels.
pub mod moe;
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
/// Numerical error measurement and tolerance checking utilities.
pub mod tolerance;
/// Walsh-Hadamard Transform (WHT) compute kernels.
pub mod wht;

pub use attention::{causal_attention, causal_attention_with_sinks};
pub use gating::{sigmoid_gate_mul, sigmoid_scalar_mul, split_q_gate};
pub use gdn::{sigmoid, silu, softplus, GdnDims, GdnReference, GDN_RMS_EPS};
pub use moe::{apply_streamed_routed, gelu_tanh, run_ffn};
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
pub use rms_norm::{rms_norm, rms_norm_centered};
pub use rope::{rope_neox, rope_neox_subdim, rope_paired, yarn_frequencies, YarnSpec};
pub use sampling::logit_softcap_softmax;
pub use tolerance::{bounded_rel_error, max_abs_diff, rel_error, Tolerance};
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
