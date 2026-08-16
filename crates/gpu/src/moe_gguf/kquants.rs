//! Host-side dispatch for standard GGUF quant types (Q8_0, Q4_K, Q6_K).

use super::{encode_phase1, encode_phase2};
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::moe_decode::{MoeExpertOffsets, RoutedBlobsBuffer};

/// Phase 1 over Q4_K expert blobs: for each of `top_k` slots,
/// `acts[slot * f_dim + f] = activation(gate_f(x)) * up_f(x)`.
///
/// `d_dim` must be a whole number of 256-element Q4_K superblocks. That is a
/// stronger requirement than either the Q8_0 pair's 32 or the vendored pair's
/// group of 64, and it is the only shape difference the host can see.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_phase1_q4_k(
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
    assert_eq!(d_dim as usize % crate::Q4_K_BLOCK_ELEMS, 0);
    assert!(top_k as usize <= crate::moe_decode::MAX_STREAMED_EXPERTS);
    encode_phase1(
        context,
        pass,
        "moe_phase1_gate_up_act_q4_k",
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

/// Phase 2 over Q4_K expert blobs: `y[d] = residual[d] + sum_slot
/// routing_w[slot] * down_d(acts[slot])`, reducing all eight slots
/// unconditionally (see the vendored sibling's contract).
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_phase2_q4_k(
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
    use_silu: bool,
) -> Result<(), GpuError> {
    assert_eq!(f_dim as usize % crate::Q4_K_BLOCK_ELEMS, 0);
    encode_phase2(
        context,
        pass,
        "moe_phase2_down_reduce_k8_q4_k",
        routed,
        offsets,
        acts,
        routing_w,
        residual,
        y,
        d_dim,
        f_dim,
        use_silu,
    )
}

/// Phase 1 over Q8_0 expert blobs: for each of `top_k` slots,
/// `acts[slot * f_dim + f] = activation(gate_f(x)) * up_f(x)`.
///
/// `d_dim` must be a whole number of 32-element Q8_0 blocks, which is a
/// weaker requirement than the vendored kernel's group of 64.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_phase1_q8_0(
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
    assert_eq!(d_dim as usize % crate::Q8_0_BLOCK_ELEMS, 0);
    assert!(top_k as usize <= crate::moe_decode::MAX_STREAMED_EXPERTS);
    encode_phase1(
        context,
        pass,
        "moe_phase1_gate_up_act_q8_0",
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

/// Phase 2 over Q8_0 expert blobs: `y[d] = residual[d] + sum_slot
/// routing_w[slot] * down_d(acts[slot])`, reducing all eight slots
/// unconditionally (see the vendored sibling's contract).
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_phase2_q8_0(
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
    use_silu: bool,
) -> Result<(), GpuError> {
    assert_eq!(f_dim as usize % crate::Q8_0_BLOCK_ELEMS, 0);
    encode_phase2(
        context,
        pass,
        "moe_phase2_down_reduce_k8_q8_0",
        routed,
        offsets,
        acts,
        routing_w,
        residual,
        y,
        d_dim,
        f_dim,
        use_silu,
    )
}

/// Phase 2 over Q6_K expert blobs, which is what Mixtral 8x7B's Q4_K_M
/// carries on the `ffn_down_exps` of 16 of its 32 layers (the other 16 are
/// Q4_K and use the pair above) -- ROADMAP Phase M2.
///
/// There is deliberately no Q6_K phase 1: that file's `ffn_gate_exps` and
/// `ffn_up_exps` are Q4_K on every layer. Same call the Q6_K GEMV made when it
/// shipped without an embedding or MoE sibling, and a checkpoint that needs
/// one fails at the dispatch site by name.
///
/// `f_dim` must be a whole number of 256-element superblocks.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_phase2_q6_k(
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
    use_silu: bool,
) -> Result<(), GpuError> {
    assert_eq!(f_dim as usize % crate::Q6_K_BLOCK_ELEMS, 0);
    encode_phase2(
        context,
        pass,
        "moe_phase2_down_reduce_k8_q6_k",
        routed,
        offsets,
        acts,
        routing_w,
        residual,
        y,
        d_dim,
        f_dim,
        use_silu,
    )
}
