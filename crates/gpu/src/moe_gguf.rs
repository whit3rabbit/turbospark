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

use crate::bytes::{f32_bytes, u32_bytes};
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::moe_decode::{
    constants_key, moe_function_constants, MoeExpertOffsets, RoutedBlobsBuffer,
};

const SOURCE: &str = concat!(
    include_str!("shaders/moe.metal"),
    include_str!("shaders/dequant_q4_k.metal"),
    include_str!("shaders/dequant_q6_k.metal"),
    include_str!("shaders/dequant_iq.metal"),
    include_str!("shaders/moe_gguf.metal")
);
const ROWS_PER_THREADGROUP: u64 = 8;
const THREADS_PER_GROUP: u64 = 256;

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
    assert!(top_k as usize <= crate::moe_decode::MAX_STREAMED_EXPERTS);
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
    assert!(top_k as usize <= crate::moe_decode::MAX_STREAMED_EXPERTS);
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

/// The dispatch every phase-1 kernel shares. Only the kernel name and the
/// block-size precondition differ between block types; the bindings, the
/// function constants and the threadgroup shape are identical, and keeping
/// one copy of them is what stops them drifting apart.
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
