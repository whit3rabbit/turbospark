//! Norm kernel selection for the Gemma flow.
//!
//! Gemma 4 stores plain RMSNorm weights (`x * w`). Gemma 3 stores the
//! unbaked weight of `x * (1 + w)`: the GGUF walk subtracts llama.cpp's baked
//! `+1` (`gguf_checkpoint::conventions`), so the install holds the original
//! centered value and every norm here must add the 1 back. Picked by the
//! FAMILY because it is a property of the whole checkpoint, and it covers
//! every norm of Gemma 3 (the four per-layer norms, the per-head q/k norms
//! and the final norm). Do not use this for the weightless V norm, which
//! Gemma 3 does not have at all.

use model_io::{ArchConfig, ModelFamily};

/// Whether this arch's norm weights are offsets from unity.
pub(crate) fn centered_norms(arch: &ArchConfig) -> bool {
    arch.family == ModelFamily::Gemma3
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_norm(
    arch: &ArchConfig,
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    x: (&gpu::MetalBuffer, u64),
    weight: (&gpu::MetalBuffer, u64),
    out: (&gpu::MetalBuffer, u64),
    d: u32,
    eps: f32,
) -> Result<(), gpu::GpuError> {
    if centered_norms(arch) {
        gpu::encode_rms_norm_bf16w_centered(context, pass, x, weight, out, d, eps)
    } else {
        gpu::encode_rms_norm_bf16w(context, pass, x, weight, out, d, eps)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_norm_perhead(
    arch: &ArchConfig,
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    x: (&gpu::MetalBuffer, u64),
    weight: (&gpu::MetalBuffer, u64),
    out: (&gpu::MetalBuffer, u64),
    num_heads: u32,
    head_dim: u32,
    eps: f32,
) -> Result<(), gpu::GpuError> {
    if centered_norms(arch) {
        gpu::encode_rms_norm_bf16w_perhead_centered(
            context, pass, x, weight, out, num_heads, head_dim, eps,
        )
    } else {
        gpu::encode_rms_norm_bf16w_perhead(context, pass, x, weight, out, num_heads, head_dim, eps)
    }
}

/// Gemma 4 applies a weightless per-head RMS norm to V; Gemma 3 has none.
pub(crate) fn applies_v_norm(arch: &ArchConfig) -> bool {
    arch.family != ModelFamily::Gemma3
}
