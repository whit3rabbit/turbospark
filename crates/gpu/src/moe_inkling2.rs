//! Host-side dispatch for `shaders/moe_inkling2.metal`: the routed-expert
//! decode pair for Inkling 2's native NVFP4 and MXFP8 experts.
//!
//! PORT-LOCAL. The blob layout, the activation and why phase 2 takes no
//! residual are in the shader header; the CPU contract is
//! `turbospark_compute::inkling2`, held by `tests/moe_inkling2_parity.rs`.
//!
//! A separate module from [`crate::moe_gguf`] rather than two more
//! `MoeExtType` arms: the activation, the per-expert outer scale and the
//! residual-free phase 2 are all Inkling's, and none of the GGUF kernels'
//! callers should be able to reach them by picking a block type. The dispatch
//! shape (8 rows per 256-thread threadgroup in phase 1, one threadgroup of 8
//! simdgroups per output row in phase 2) and the `RoutedBlobs` /
//! `ExpertOffsets` uniforms are the same objects the GGUF pair binds.

use crate::bytes::u32_bytes;
use crate::moe_decode::{constants_key, moe_function_constants};
use crate::{GpuError, MetalContext, MoeExpertOffsets, PassEncoder, RoutedBlobsBuffer};

const SOURCE: &str = concat!(
    include_str!("shaders/moe.metal"),
    include_str!("shaders/dequant_iq1_narrow_tables.metal"),
    include_str!("shaders/dequant_ext.metal"),
    include_str!("shaders/moe_inkling2.metal")
);
const ROWS_PER_THREADGROUP: u64 = 8;
const THREADS_PER_GROUP: u64 = 256;

/// The most routed slots phase 2 reduces: one simdgroup each in a
/// 256-thread threadgroup. Inkling 2 Mini routes exactly 8.
pub const INKLING2_MAX_SLOTS: usize = 8;

/// Bytes of the per-expert outer-scale record the repack writes at the front
/// of every blob: gate, up and down as f32, then 4 bytes of padding.
pub const INKLING2_OUTER_SCALE_BYTES: usize = 16;

/// Which native layout an Inkling 2 layer's experts use. Layers 20 to 23 of
/// the Mini are MXFP8; layers 0 to 19 are NVFP4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inkling2ExpertType {
    Nvfp4,
    Mxfp8,
}

impl Inkling2ExpertType {
    /// Elements per block; a routed row must be a whole number of them.
    pub fn block_elems(self) -> usize {
        match self {
            Inkling2ExpertType::Nvfp4 => 64,
            Inkling2ExpertType::Mxfp8 => 32,
        }
    }

    /// Bytes per block.
    pub fn block_bytes(self) -> usize {
        match self {
            Inkling2ExpertType::Nvfp4 => 36,
            Inkling2ExpertType::Mxfp8 => 33,
        }
    }

    /// Bytes a routed row of `n` elements occupies.
    pub fn row_bytes(self, n: usize) -> usize {
        n / self.block_elems() * self.block_bytes()
    }

    // Literals: the pipeline cache keys on the `&'static str`'s address.
    fn phase1_kernel(self) -> &'static str {
        match self {
            Inkling2ExpertType::Nvfp4 => "moe_inkling2_phase1_nvfp4",
            Inkling2ExpertType::Mxfp8 => "moe_inkling2_phase1_mxfp8",
        }
    }

    fn phase2_kernel(self) -> &'static str {
        match self {
            Inkling2ExpertType::Nvfp4 => "moe_inkling2_phase2_nvfp4",
            Inkling2ExpertType::Mxfp8 => "moe_inkling2_phase2_mxfp8",
        }
    }

    fn phase1_idx_kernel(self) -> &'static str {
        match self {
            Inkling2ExpertType::Nvfp4 => "moe_inkling2_phase1_idx_nvfp4",
            Inkling2ExpertType::Mxfp8 => "moe_inkling2_phase1_idx_mxfp8",
        }
    }

    fn phase2_idx_kernel(self) -> &'static str {
        match self {
            Inkling2ExpertType::Nvfp4 => "moe_inkling2_phase2_idx_nvfp4",
            Inkling2ExpertType::Mxfp8 => "moe_inkling2_phase2_idx_mxfp8",
        }
    }
}

/// This module's compiled library, for reflecting its `RoutedBlobs` layout
/// (`RoutedBlobsBuffer::new_for` / `bind_for`).
pub fn moe_inkling2_source() -> &'static str {
    SOURCE
}

/// A kernel name from this library to reflect the `RoutedBlobs` layout with.
pub const MOE_INKLING2_REFLECT_FUNCTION: &str = "moe_inkling2_phase1_nvfp4";

fn check(kind: Inkling2ExpertType, label: &str, dim: u32, top_k: u32) -> Result<(), GpuError> {
    if dim == 0 || dim as usize % kind.block_elems() != 0 {
        return Err(GpuError::InvalidInput(format!(
            "{label} ({dim}) must be a positive multiple of {} for Inkling 2 {kind:?}",
            kind.block_elems()
        )));
    }
    // Refused rather than truncated: phase 1 would write every slot and
    // phase 2 would silently reduce only the first eight.
    if top_k == 0 || top_k as usize > INKLING2_MAX_SLOTS {
        return Err(GpuError::InvalidInput(format!(
            "top_k {top_k} is outside 1..={INKLING2_MAX_SLOTS} for the Inkling 2 MoE kernels"
        )));
    }
    Ok(())
}

/// Phase 1: for each of `top_k` slots and each of `f_dim` rows,
/// `acts[slot * f_dim + f] = softcap(g, gate_cap) * sigmoid(g) *
/// softcap(u, up_cap)`, with `g` and `u` the slot's gate and up rows dotted
/// with `x` and multiplied by the blob's outer scales.
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_moe_phase1(
    context: &mut MetalContext,
    pass: &PassEncoder,
    kind: Inkling2ExpertType,
    routed: &RoutedBlobsBuffer,
    offsets: &MoeExpertOffsets,
    x: (&metal::Buffer, u64),
    acts: (&metal::Buffer, u64),
    d_dim: u32,
    f_dim: u32,
    top_k: u32,
    gate_cap: f32,
    up_cap: f32,
) -> Result<(), GpuError> {
    check(kind, "d_dim", d_dim, top_k)?;
    if f_dim == 0 {
        return Err(GpuError::InvalidInput("f_dim must be positive".into()));
    }
    let pipeline = context.pipeline(
        SOURCE,
        kind.phase1_kernel(),
        &moe_function_constants(false),
        &constants_key(false),
    )?;
    let mut caps = [0u8; 8];
    caps[..4].copy_from_slice(&gate_cap.to_le_bytes());
    caps[4..].copy_from_slice(&up_cap.to_le_bytes());
    let rows = u64::from(top_k) * u64::from(f_dim);
    pass.encode_threadgroups(
        &pipeline,
        &[(routed.buffer(), 0, 0), (x.0, 2, x.1), (acts.0, 3, acts.1)],
        &[
            (&offsets.bytes(), 1),
            (u32_bytes(&d_dim), 4),
            (u32_bytes(&f_dim), 5),
            (u32_bytes(&top_k), 6),
            (&caps, 7),
        ],
        rows.div_ceil(ROWS_PER_THREADGROUP),
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// Phase 2: `y[d] = sum over slots in order of routing_w[slot] * down_d(acts
/// of that slot) * outer`. No residual is added (see the shader header).
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_moe_phase2(
    context: &mut MetalContext,
    pass: &PassEncoder,
    kind: Inkling2ExpertType,
    routed: &RoutedBlobsBuffer,
    offsets: &MoeExpertOffsets,
    acts: (&metal::Buffer, u64),
    routing_w: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    d_dim: u32,
    f_dim: u32,
    top_k: u32,
) -> Result<(), GpuError> {
    check(kind, "f_dim", f_dim, top_k)?;
    if d_dim == 0 {
        return Err(GpuError::InvalidInput("d_dim must be positive".into()));
    }
    let pipeline = context.pipeline(
        SOURCE,
        kind.phase2_kernel(),
        &moe_function_constants(false),
        &constants_key(false),
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[
            (routed.buffer(), 0, 0),
            (acts.0, 2, acts.1),
            (routing_w.0, 3, routing_w.1),
            (y.0, 4, y.1),
        ],
        &[
            (&offsets.bytes(), 1),
            (u32_bytes(&d_dim), 5),
            (u32_bytes(&f_dim), 6),
            (u32_bytes(&top_k), 7),
        ],
        u64::from(d_dim),
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// One layer's experts resident in a single buffer, as the packed layer file
/// lays them out: expert `e` at byte `e * stride`, `n_experts` of them. The
/// indexed pair reads the slots' experts through a GPU-side `sel` list
/// (`u32[top_k]`, router ranking order) instead of host-bound blobs.
#[derive(Clone, Copy)]
pub struct Inkling2ResidentLayer<'a> {
    pub buffer: &'a metal::Buffer,
    pub stride: u64,
    pub n_experts: u32,
    pub sel: (&'a metal::Buffer, u64),
}

fn check_resident(kind: Inkling2ExpertType, l: &Inkling2ResidentLayer<'_>) -> Result<(), GpuError> {
    // NVFP4 rows are read as 32-bit words; a stride that is not a multiple
    // of 4 would misalign every expert past the first.
    if l.n_experts == 0
        || l.stride == 0
        || (kind == Inkling2ExpertType::Nvfp4 && l.stride % 4 != 0)
        || l.buffer.length() < l.stride * u64::from(l.n_experts)
    {
        return Err(GpuError::InvalidInput(format!(
            "resident expert layer: stride {} x {} experts does not fit a {} B buffer \
             (NVFP4 needs a 4-byte stride)",
            l.stride,
            l.n_experts,
            l.buffer.length()
        )));
    }
    Ok(())
}

/// [`encode_inkling2_moe_phase1`] over a resident layer and a GPU `sel` list.
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_moe_phase1_indexed(
    context: &mut MetalContext,
    pass: &PassEncoder,
    kind: Inkling2ExpertType,
    layer: &Inkling2ResidentLayer<'_>,
    offsets: &MoeExpertOffsets,
    x: (&metal::Buffer, u64),
    acts: (&metal::Buffer, u64),
    d_dim: u32,
    f_dim: u32,
    top_k: u32,
    gate_cap: f32,
    up_cap: f32,
) -> Result<(), GpuError> {
    check(kind, "d_dim", d_dim, top_k)?;
    check_resident(kind, layer)?;
    if f_dim == 0 {
        return Err(GpuError::InvalidInput("f_dim must be positive".into()));
    }
    let pipeline = context.pipeline(
        SOURCE,
        kind.phase1_idx_kernel(),
        &moe_function_constants(false),
        &constants_key(false),
    )?;
    let mut caps = [0u8; 8];
    caps[..4].copy_from_slice(&gate_cap.to_le_bytes());
    caps[4..].copy_from_slice(&up_cap.to_le_bytes());
    let rows = u64::from(top_k) * u64::from(f_dim);
    pass.encode_threadgroups(
        &pipeline,
        &[
            (layer.buffer, 0, 0),
            (x.0, 2, x.1),
            (acts.0, 3, acts.1),
            (layer.sel.0, 8, layer.sel.1),
        ],
        &[
            (&offsets.bytes(), 1),
            (u32_bytes(&d_dim), 4),
            (u32_bytes(&f_dim), 5),
            (u32_bytes(&top_k), 6),
            (&caps, 7),
            (&layer.stride.to_le_bytes(), 9),
            (u32_bytes(&layer.n_experts), 10),
        ],
        rows.div_ceil(ROWS_PER_THREADGROUP),
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// [`encode_inkling2_moe_phase2`] over a resident layer and a GPU `sel` list.
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_moe_phase2_indexed(
    context: &mut MetalContext,
    pass: &PassEncoder,
    kind: Inkling2ExpertType,
    layer: &Inkling2ResidentLayer<'_>,
    offsets: &MoeExpertOffsets,
    acts: (&metal::Buffer, u64),
    routing_w: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    d_dim: u32,
    f_dim: u32,
    top_k: u32,
) -> Result<(), GpuError> {
    check(kind, "f_dim", f_dim, top_k)?;
    check_resident(kind, layer)?;
    if d_dim == 0 {
        return Err(GpuError::InvalidInput("d_dim must be positive".into()));
    }
    let pipeline = context.pipeline(
        SOURCE,
        kind.phase2_idx_kernel(),
        &moe_function_constants(false),
        &constants_key(false),
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[
            (layer.buffer, 0, 0),
            (acts.0, 2, acts.1),
            (routing_w.0, 3, routing_w.1),
            (y.0, 4, y.1),
            (layer.sel.0, 8, layer.sel.1),
        ],
        &[
            (&offsets.bytes(), 1),
            (u32_bytes(&d_dim), 5),
            (u32_bytes(&f_dim), 6),
            (u32_bytes(&top_k), 7),
            (&layer.stride.to_le_bytes(), 9),
            (u32_bytes(&layer.n_experts), 10),
        ],
        u64::from(d_dim),
        THREADS_PER_GROUP,
    );
    Ok(())
}
