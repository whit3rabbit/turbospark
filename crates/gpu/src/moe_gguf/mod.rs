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

mod iq;
mod kquants;
mod mxfp4;

use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::moe_decode::{
    constants_key, moe_function_constants, MoeExpertOffsets, RoutedBlobsBuffer,
};

pub use iq::{encode_moe_phase1_iq3_xxs, encode_moe_phase1_iq4_xs, encode_moe_phase2_iq4_nl};
pub use kquants::{
    encode_moe_phase1_q4_k, encode_moe_phase1_q8_0, encode_moe_phase2_q4_k, encode_moe_phase2_q6_k,
    encode_moe_phase2_q8_0,
};
pub use mxfp4::{
    encode_moe_phase1_mxfp4, encode_moe_phase2_mxfp4, mxfp4_row_bytes, Mxfp4Activation,
    MXFP4_BLOCK_BYTES, MXFP4_BLOCK_ELEMS,
};

const SOURCE: &str = concat!(
    include_str!("../shaders/moe.metal"),
    include_str!("../shaders/dequant_q4_k.metal"),
    include_str!("../shaders/dequant_q6_k.metal"),
    include_str!("../shaders/dequant_iq.metal"),
    include_str!("../shaders/moe_gguf.metal")
);
const ROWS_PER_THREADGROUP: u64 = 8;
const THREADS_PER_GROUP: u64 = 256;

/// This module's compiled library (every GGUF block type's phase 1/2 pair,
/// MXFP4 included), for a caller that needs to reflect one of ITS
/// functions' `RoutedBlobs` argument-buffer layout rather than the vendored
/// pair's (S7: `RoutedBlobsBuffer::new_for`/`bind_for` in
/// [`crate::moe_decode`]). Any one of this library's entry points reflects
/// the same layout, since they share one struct declaration in one
/// compilation -- see this module's own doc comment for the concatenation.
pub fn moe_gguf_source() -> &'static str {
    SOURCE
}

/// The width every phase-2 kernel in this module reduces, and (via
/// `crate::moe_prefill_batch`/`crate::moe_prefill_batch_gguf`) the width
/// their batched-prefill siblings reduce too: `moe_gguf.metal`'s
/// `moe_phase2_down_reduce_k8_*` kernels (Q4_K, Q8_0, Q6_K, IQ4_NL, and the
/// masked MXFP4 arm) all declare `threadgroup float partial[8]` and reduce
/// exactly 8 simdgroups, unlike the vendored INT4-affine pair's own phase 2
/// (`moe_decode.rs`), which was widened to `MAX_STREAMED_EXPERTS` (16,
/// AGENTS.md Gotcha 13). A `top_k` past this width is silently truncated:
/// phase 1 (sized to `top_k * f_dim` rows) writes every slot, and phase 2
/// reduces only the first 8. Every phase-1 AND phase-2 dispatch in this
/// module, plus the batched-prefill pair, asserts `top_k` against this
/// constant rather than `MAX_STREAMED_EXPERTS` for that reason -- refusing
/// loudly at the point where a wider checkpoint would otherwise decode
/// fluent, wrong output.
pub const PHASE2_FIXED_SLOTS: usize = 8;

/// The dispatch every phase-1 kernel shares. Only the kernel name and the
/// block-size precondition differ between block types; the bindings, the
/// function constants and the threadgroup shape are identical, and keeping
/// one copy of them is what stops them drifting apart.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_phase1(
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

/// The dispatch both phase-2 kernels share; see [`encode_phase1`]. Asserts
/// `top_k` against [`PHASE2_FIXED_SLOTS`] rather than
/// `MAX_STREAMED_EXPERTS`: these kernels reduce a fixed 8 simdgroups
/// regardless of the width phase 1 was dispatched at (see that constant's
/// doc).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_phase2(
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
    top_k: u32,
    use_silu: bool,
) -> Result<(), GpuError> {
    assert!(top_k as usize <= PHASE2_FIXED_SLOTS);
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
