//! Host-side dispatch for `shaders/moe_gguf.metal`'s decode pair
//! (ROADMAP Phase G Stage 2): `moe_phase1_gate_up_act_q8_0` and
//! `moe_phase2_down_reduce_k8_q8_0`, the streamed routed-expert path for
//! GGUF block-quantized expert blobs.
//!
//! PORT-LOCAL, not vendored: the Swift engine has no GGUF intake. The
//! contract is `turbospark_compute::dequant_q8_0_gemv` plus the same gated
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
//! The shader source is `moe.metal`, then `dequant_q4_k.metal`, then
//! `moe_gguf.metal`, in that order: the last uses the first's `RoutedBlobs`,
//! `ExpertOffsets`, `moe_hidden_activation` and `moe_fc_*` declarations, and
//! the second's `dequant_q4_k_row_simd`. That is the `gdn.rs` arrangement and
//! it carries the same two caveats: the concatenation must be ONE
//! `&'static str` constant (the pipeline cache keys on its address, not its
//! text), and the two borrowed files' own kernels are compiled twice in this
//! process.
//!
//! Both block types get their own pair rather than one function-constant
//! kernel, for the reason the file header gives: the layouts share no
//! addressing. Q8_0 needs rows to be a whole number of 32 elements and Q4_K
//! of 256, which is the only difference visible from the host.

use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::moe_decode::{
    constants_key, moe_function_constants, MoeExpertOffsets, RoutedBlobsBuffer,
};

const SOURCE: &str = concat!(
    include_str!("shaders/moe.metal"),
    include_str!("shaders/dequant_q4_k.metal"),
    include_str!("shaders/moe_gguf.metal")
);
const ROWS_PER_THREADGROUP: u64 = 8;
const THREADS_PER_GROUP: u64 = 256;

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

/// The dispatch both phase-1 kernels share. Only the kernel name and the
/// block-size precondition differ between block types; the bindings, the
/// function constants and the threadgroup shape are identical, and keeping
/// one copy of them is what stops the two drifting apart.
#[allow(clippy::too_many_arguments)]
fn encode_phase1(
    context: &mut MetalContext,
    pass: &PassEncoder,
    kernel: &'static str,
    routed: &RoutedBlobsBuffer,
    offsets: &MoeExpertOffsets,
    x: (&metal::Buffer, u64),
    acts: (&metal::Buffer, u64),
    d_dim: u32,
    f_dim: u32,
    top_k: u32,
    use_silu: bool,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        kernel,
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

/// The dispatch both phase-2 kernels share; see [`encode_phase1`].
#[allow(clippy::too_many_arguments)]
fn encode_phase2(
    context: &mut MetalContext,
    pass: &PassEncoder,
    kernel: &'static str,
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
    let pipeline = context.pipeline(
        SOURCE,
        kernel,
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
