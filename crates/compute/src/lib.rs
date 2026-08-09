//! Destination-selected compute strategy plus CPU reference kernels.
//!
//! The kernel modules (`rms_norm`, `wht`, `rope`, `attention`, `quant`,
//! `quant_gguf`, `quant_gguf_iq`,
//! `moe`, `gdn`, `gating`, `sampling`, `tolerance`) are the numerical ground truth later GPU
//! kernels are validated against. Numerics parity with any upstream
//! implementation is out of scope for `ComputeStrategy` itself; only the
//! structural contracts of the decode and prefill areas are exercised there.
//!
//! The `turbospark-core` dependency is brought in under the alias `foundation` to avoid
//! colliding with the standard library `core` crate in the extern prelude.
#![forbid(unsafe_code)]

pub mod attention;
pub mod gating;
pub mod gdn;
pub mod moe;
pub mod quant;
pub mod quant_gguf;
pub mod quant_gguf_iq;
pub mod quant_gguf_iq_tables;
pub mod rms_norm;
pub mod rope;
pub mod sampling;
pub mod tolerance;
pub mod wht;

pub use attention::causal_attention;
pub use gating::{sigmoid_gate_mul, sigmoid_scalar_mul, split_q_gate};
pub use gdn::{sigmoid, silu, softplus, GdnDims, GdnReference, GDN_RMS_EPS};
pub use moe::{apply_streamed_routed, gelu_tanh, run_ffn};
pub use quant::{
    bf16_to_f32, dequant_int4_gemv, dequant_int8_gemv, dequantize_int4_affine,
    dequantize_int8_affine, embed_lookup_int4, embed_lookup_int8, f32_to_bf16,
    quantize_int4_affine, quantize_int8_affine, Int4AffineRow, Int8AffineRow,
};
pub use quant_gguf::{
    dequant_q4_k_gemv, dequant_q6_k_gemv, dequant_q8_0_gemv, dequantize_q4_k, dequantize_q6_k,
    dequantize_q8_0, pearson, quantize_q4_k, quantize_q6_k, quantize_q8_0, Q4_K_BLOCK_BYTES,
    Q4_K_BLOCK_ELEMS, Q4_K_SUB_ELEMS, Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS, Q6_K_SUB_ELEMS,
    Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS,
};
pub use quant_gguf_iq::{
    dequant_iq3_xxs_gemv, dequant_iq4_nl_gemv, dequant_iq4_xs_gemv, dequantize_iq3_xxs,
    dequantize_iq4_nl, dequantize_iq4_xs, iq3xxs_signs, IQ3_XXS_BLOCK_BYTES, IQ3_XXS_BLOCK_ELEMS,
    IQ3_XXS_SUB_ELEMS, IQ4_NL_BLOCK_BYTES, IQ4_NL_BLOCK_ELEMS, IQ4_XS_BLOCK_BYTES,
    IQ4_XS_BLOCK_ELEMS, IQ4_XS_SUB_ELEMS,
};
pub use quant_gguf_iq_tables::{IQ3XXS_GRID, IQ4NL_VALUES};
pub use rms_norm::rms_norm;
pub use rope::{rope_neox, rope_neox_subdim, rope_paired};
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
