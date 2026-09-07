//! Host-side dispatch for IQ codebook quant types (IQ3_XXS, IQ4_XS, IQ4_NL).

use super::{encode_phase1, encode_phase2};
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::moe_decode::{MoeExpertOffsets, RoutedBlobsBuffer};

/// Phase 1 over IQ3_XXS expert blobs (ROADMAP Phase S), the codebook type the
/// candidate checkpoint puts in 29 of its 30 `ffn_gate_up_exps` tensors.
///
/// `d_dim` must be a whole number of 256-element superblocks, as for Q4_K.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_phase1_iq3_xxs(
    context: &mut MetalContext,
    pass: &PassEncoder,
    routed: &RoutedBlobsBuffer,
    offsets: &MoeExpertOffsets,
    x: (&metal::Buffer, u64),
    acts: (&metal::Buffer, u64),
    d_dim: u32,
    f_dim: u32,
    top_k: u32,
    use_silu: bool,
) -> Result<(), GpuError> {
    assert_eq!(d_dim as usize % crate::IQ3_XXS_BLOCK_ELEMS, 0);
    assert!(top_k as usize <= super::PHASE2_FIXED_SLOTS);
    encode_phase1(
        context,
        pass,
        "moe_phase1_gate_up_act_iq3_xxs",
        routed,
        offsets,
        x,
        acts,
        d_dim,
        f_dim,
        top_k,
        use_silu,
    )
}

/// Phase 1 over IQ4_XS expert blobs. The candidate uses this on exactly one
/// layer; see `moe_gguf.metal`'s IQ header for why that is the real file's
/// shape rather than an omission here.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_phase1_iq4_xs(
    context: &mut MetalContext,
    pass: &PassEncoder,
    routed: &RoutedBlobsBuffer,
    offsets: &MoeExpertOffsets,
    x: (&metal::Buffer, u64),
    acts: (&metal::Buffer, u64),
    d_dim: u32,
    f_dim: u32,
    top_k: u32,
    use_silu: bool,
) -> Result<(), GpuError> {
    assert_eq!(d_dim as usize % crate::IQ4_XS_BLOCK_ELEMS, 0);
    assert!(top_k as usize <= super::PHASE2_FIXED_SLOTS);
    encode_phase1(
        context,
        pass,
        "moe_phase1_gate_up_act_iq4_xs",
        routed,
        offsets,
        x,
        acts,
        d_dim,
        f_dim,
        top_k,
        use_silu,
    )
}

/// Phase 2 over IQ4_NL expert blobs, which is what the candidate's
/// `ffn_down_exps` carries on 29 of 30 layers (the thirtieth is Q8_0, and uses
/// the pair above).
///
/// `f_dim` must be a whole number of 32-element blocks, as for Q8_0.
/// `top_k` must not exceed [`super::PHASE2_FIXED_SLOTS`].
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_phase2_iq4_nl(
    context: &mut MetalContext,
    pass: &PassEncoder,
    routed: &RoutedBlobsBuffer,
    offsets: &MoeExpertOffsets,
    acts: (&metal::Buffer, u64),
    routing_w: (&metal::Buffer, u64),
    residual: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    d_dim: u32,
    f_dim: u32,
    top_k: u32,
    use_silu: bool,
) -> Result<(), GpuError> {
    assert_eq!(f_dim as usize % crate::IQ4_NL_BLOCK_ELEMS, 0);
    encode_phase2(
        context,
        pass,
        "moe_phase2_down_reduce_k8_iq4_nl",
        routed,
        offsets,
        acts,
        routing_w,
        residual,
        y,
        d_dim,
        f_dim,
        top_k,
        use_silu,
    )
}
