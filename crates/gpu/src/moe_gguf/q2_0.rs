//! Host-side dispatch for the Qwen4Exp top-10 Q2_0 routed down kernel.

use super::encode_phase1;
use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::moe_decode::{
    constants_key, moe_function_constants, MoeExpertOffsets, RoutedBlobsBuffer,
};

/// Phase 1 over Q2_0 routed gate/up rows. Qwen4Exp's Q2_0 tier uses this
/// format for both projections as well as its down projection.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_phase1_q2_0(
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
    assert_eq!(d_dim as usize % turbospark_compute::Q2_0_BLOCK_ELEMS, 0);
    encode_phase1(
        context,
        pass,
        "moe_phase1_gate_up_act_q2_0",
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

/// Phase 2 over Q2_0 routed down rows, reducing exactly ten routed experts.
///
/// The dedicated Metal kernel has ten SIMD groups and ten reduction entries;
/// accepting another width would silently change the result.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_phase2_q2_0_top10(
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
    assert_eq!(top_k, 10, "Q2_0 routed phase 2 currently requires top_k=10");
    assert_eq!(f_dim as usize % turbospark_compute::Q2_0_BLOCK_ELEMS, 0);
    let pipeline = context.pipeline(
        super::SOURCE,
        "moe_phase2_down_reduce_k10_q2_0",
        &moe_function_constants(use_silu),
        &constants_key(use_silu),
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[
            (routed.buffer(), 0, 0),
            (acts.0, 2, acts.1),
            (routing_w.0, 3, routing_w.1),
            (residual.0, 4, residual.1),
            (y.0, 5, y.1),
        ],
        &[
            (&offsets.bytes(), 1),
            (u32_bytes(&d_dim), 6),
            (u32_bytes(&f_dim), 7),
        ],
        d_dim as u64,
        320,
    );
    Ok(())
}
