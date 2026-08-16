//! Host-side dispatch for MXFP4 expert blobs (ROADMAP M5).

use super::{ROWS_PER_THREADGROUP, SOURCE, THREADS_PER_GROUP};
use crate::bytes::{f32_bytes, u32_bytes};
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::moe_decode::{
    constants_key, moe_function_constants, MoeExpertOffsets, RoutedBlobsBuffer,
};

/// Elements per MXFP4 block. Mirrors
/// `turbospark_compute::MXFP4_BLOCK_ELEMS`; held equal by
/// `crates/gpu/tests/moe_gguf_parity.rs` rather than by an import, as every
/// other block type's pair of constants is.
///
/// These live HERE, and not in a `dequant_mxfp4_gemv.rs` beside the six other
/// block types', because MXFP4 has no resident GEMV: the only real file
/// carrying it (`gpt-oss-20b-MXFP4.gguf`) puts it in `ffn_*_exps` and nowhere
/// else, so this module is its sole consumer.
pub const MXFP4_BLOCK_ELEMS: usize = 32;
/// Bytes per MXFP4 block: one E8M0 exponent byte then 16 nibble-packed
/// indices. ODD, unlike every other block type here, which is why the kernel
/// reads bytes and never a `ushort`.
pub const MXFP4_BLOCK_BYTES: usize = 17;

/// Bytes in one MXFP4 row of `n` elements.
#[must_use]
pub fn mxfp4_row_bytes(n: usize) -> usize {
    assert_eq!(
        n % MXFP4_BLOCK_ELEMS,
        0,
        "N ({n}) is not a whole number of {MXFP4_BLOCK_ELEMS}-element blocks"
    );
    n / MXFP4_BLOCK_ELEMS * MXFP4_BLOCK_BYTES
}

/// How the MXFP4 phase-1 kernel activates, which is a FAMILY property rather
/// than a block-type one.
///
/// `gpt-oss` is the only checkpoint carrying MXFP4 experts and its activation
/// is llama.cpp's `swiglu_oai`, not the plain `activation(gate) * up` every
/// other pair here runs. Rather than a second kernel, the parameters ride as
/// uniforms: `constants_key` is one byte wide and adding two specialization
/// axes to it is how a pipeline cache hands back the wrong kernel (AGENTS.md
/// Gotcha 18's trap, one file over).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mxfp4Activation {
    /// Swish steepness. `0.0` selects the ordinary gated activation the
    /// block-type parity cases use; `gpt-oss` passes 1.702.
    pub alpha: f32,
    /// Pre-activation clamp. `gpt-oss` passes 7.0.
    pub limit: f32,
    /// Whether the blob carries per-expert bias planes at the three bias
    /// offsets, which every GGUF install before `gpt-oss` left at zero.
    pub has_bias: bool,
}

impl Mxfp4Activation {
    /// The plain gated activation, no biases: what a hypothetical MXFP4
    /// checkpoint that is not `gpt-oss` would use, and what the block-type
    /// parity cases assert.
    pub const PLAIN: Mxfp4Activation = Mxfp4Activation {
        alpha: 0.0,
        limit: 0.0,
        has_bias: false,
    };

    /// `gpt-oss`'s, with the two constants llama.cpp hardcodes.
    pub const GPT_OSS: Mxfp4Activation = Mxfp4Activation {
        alpha: 1.702,
        limit: 7.0,
        has_bias: true,
    };
}

/// Phase 1 over MXFP4 expert blobs (ROADMAP M5), which is what `gpt-oss`
/// streams. Unlike every other type here MXFP4 has NO resident GEMV and no
/// embedding lookup, because that file puts it in `ffn_*_exps` and nowhere
/// else; see `moe_gguf.metal`'s MXFP4 header.
///
/// `d_dim` must be a whole number of 32-element blocks, as for Q8_0.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_phase1_mxfp4(
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
    act: Mxfp4Activation,
) -> Result<(), GpuError> {
    assert_eq!(d_dim as usize % MXFP4_BLOCK_ELEMS, 0);
    assert!(top_k as usize <= crate::moe_decode::MAX_STREAMED_EXPERTS);
    let pipeline = context.pipeline(
        SOURCE,
        "moe_phase1_gate_up_act_mxfp4",
        &moe_function_constants(use_silu),
        &constants_key(use_silu),
    )?;
    let rows = (top_k * f_dim) as u64;
    let has_bias = u32::from(act.has_bias);
    pass.encode_threadgroups(
        &pipeline,
        &[(routed.buffer(), 0, 0), (x.0, 2, x.1), (acts.0, 3, acts.1)],
        &[
            (&offsets.bytes(), 1),
            (u32_bytes(&d_dim), 4),
            (u32_bytes(&f_dim), 5),
            (u32_bytes(&top_k), 6),
            (u32_bytes(&has_bias), 7),
            (f32_bytes(&act.alpha), 8),
            (f32_bytes(&act.limit), 9),
        ],
        rows.div_ceil(ROWS_PER_THREADGROUP),
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// Phase 2 over MXFP4 expert blobs, reducing all eight slots unconditionally
/// (see the vendored sibling's contract).
///
/// `f_dim` must be a whole number of 32-element blocks.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_phase2_mxfp4(
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
    has_bias: bool,
) -> Result<(), GpuError> {
    assert_eq!(f_dim as usize % MXFP4_BLOCK_ELEMS, 0);
    let pipeline = context.pipeline(
        SOURCE,
        "moe_phase2_down_reduce_k8_mxfp4",
        &moe_function_constants(use_silu),
        &constants_key(use_silu),
    )?;
    let has_bias = u32::from(has_bias);
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
            (u32_bytes(&has_bias), 8),
        ],
        d_dim as u64,
        THREADS_PER_GROUP,
    );
    Ok(())
}
