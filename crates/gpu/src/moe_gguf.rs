//! Host-side dispatch for `shaders/moe_gguf.metal`'s decode pair
//! (ROADMAP Phase G Stage 2): `moe_phase1_gate_up_act_q8_0` and
//! `moe_phase2_down_reduce_k8_q8_0`, the streamed routed-expert path for
//! GGUF block-quantized expert blobs.
//!
//! PORT-LOCAL, not vendored: the Swift engine has no GGUF intake. The
//! contract is `mrefrust_compute::dequant_q8_0_gemv` plus the same gated
//! activation the vendored pair uses, held by
//! `crates/gpu/tests/moe_gguf_parity.rs`.
//!
//! Two things are shared with [`crate::moe_decode`] rather than duplicated,
//! because they are the same objects and not merely similar ones. The
//! `RoutedBlobs` argument buffer ([`crate::RoutedBlobsBuffer`]) is one array
//! of eight pointers whatever the blob contains, so a GGUF dispatch binds the
//! one the runtime already owns. And [`crate::MoeExpertOffsets`] is the same
//! nine-field uniform, of which a GGUF blob uses only the three WEIGHT
//! offsets: there are no scale or bias planes to point at, so the host writes
//! zeros into the other six and the kernels never read them.
//!
//! The shader source is `moe.metal` CONCATENATED with `moe_gguf.metal`, in
//! that order, because the latter uses the former's `RoutedBlobs`,
//! `ExpertOffsets`, `moe_hidden_activation` and `moe_fc_*` declarations. That
//! is the `gdn.rs` arrangement and it carries the same two caveats: the
//! concatenation must be ONE `&'static str` constant (the pipeline cache keys
//! on its address, not its text), and `moe.metal`'s kernels are compiled
//! twice in this process.

use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::moe_decode::{
    constants_key, moe_function_constants, MoeExpertOffsets, RoutedBlobsBuffer,
};

const SOURCE: &str = concat!(
    include_str!("shaders/moe.metal"),
    include_str!("shaders/moe_gguf.metal")
);
const ROWS_PER_THREADGROUP: u64 = 8;
const THREADS_PER_GROUP: u64 = 256;

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
    assert_eq!(d_dim % 32, 0);
    assert!(top_k as usize <= crate::moe_decode::MAX_STREAMED_EXPERTS);
    let pipeline = context.pipeline(
        SOURCE,
        "moe_phase1_gate_up_act_q8_0",
        &moe_function_constants(use_silu),
        &constants_key(use_silu),
    )?;
    let rows = (top_k * f_dim) as u64;
    pass.encode_threadgroups(
        &pipeline,
        &[(routed.buffer(), 0, 0), (x.0, 2, x.1), (acts.0, 3, acts.1)],
        &[
            (&offsets.bytes(), 1),
            (u32_bytes(&d_dim), 4),
            (u32_bytes(&f_dim), 5),
            (u32_bytes(&top_k), 6),
        ],
        rows.div_ceil(ROWS_PER_THREADGROUP),
        THREADS_PER_GROUP,
    );
    Ok(())
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
    assert_eq!(f_dim % 32, 0);
    let pipeline = context.pipeline(
        SOURCE,
        "moe_phase2_down_reduce_k8_q8_0",
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
        THREADS_PER_GROUP,
    );
    Ok(())
}
