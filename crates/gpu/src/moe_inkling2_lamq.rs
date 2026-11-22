//! Host-side dispatch for `shaders/moe_inkling2_lamq.metal`: the routed-expert
//! decode pair for Inkling 2 LookAtMyQuant installs (`docs/LAMQ.md`). The CPU
//! contract is `turbospark_compute::lamq` plus `inkling2::situ`, held by
//! `tests/moe_inkling2_lamq_parity.rs`.
//!
//! Same slot binding as [`crate::moe_inkling2`] (32 rows per 1024-thread
//! threadgroup in phase 1, one threadgroup per output row in phase 2, `RoutedBlobs` or a resident layer with a `sel` list). The
//! differences: each expert's code comes from its blob's format record, the
//! input and the activations are already rotated F32 vectors (`lamq_rht`
//! dispatches around this pair), and every trellis LUT is bound in one pack.

use turbospark_compute::lamq::LamqCode;

use crate::bytes::u32_bytes;
use crate::moe_decode::{constants_key, moe_function_constants};
use crate::{GpuError, Inkling2ResidentLayer, MetalContext, PassEncoder, RoutedBlobsBuffer};

const SOURCE: &str = concat!(
    include_str!("shaders/moe.metal"),
    include_str!("shaders/lamq.metal"),
    include_str!("shaders/moe_inkling2_lamq.metal")
);
/// Phase 1 gate/up row pairs per threadgroup, one per 16-lane half of a
/// simdgroup (= the shader's `kLamqExpRowsPerTG`).
const ROWS_PER_THREADGROUP: u64 = 64;
const PHASE1_THREADS: u64 = ROWS_PER_THREADGROUP * 16;
/// Phase 2: one simdgroup per slot.
const PHASE2_THREADS: u64 = 256;

/// The most routed slots phase 2 reduces.
pub const INKLING2_LAMQ_MAX_SLOTS: usize = 8;

/// The packed LUT order and offsets (in halves) the shader's
/// `lamq_code_params` reads, followed by the rtr1 three-weight table.
pub const LAMQ_LUT_PACK: [(LamqCode, usize); 6] = [
    (LamqCode::Ht2, 0),
    (LamqCode::Ht3, 2048),
    (LamqCode::Rtr1, 4096),
    (LamqCode::Rtr2, 8192),
    (LamqCode::Rtr3, 12288),
    (LamqCode::Rtr4, 16384),
];
/// Where the rtr1 three-weight table starts (halves), after the six LUTs:
/// 16384 entries of four halves (`kLamqRtr1MtOff` in the shader).
pub const LAMQ_RTR1_MT_OFF: usize = 20480;
pub const LAMQ_LUT_PACK_HALVES: usize = LAMQ_RTR1_MT_OFF + (1 << 14) * 4;

/// This module's compiled library, for `RoutedBlobsBuffer::new_for`.
pub fn moe_inkling2_lamq_source() -> &'static str {
    SOURCE
}

/// A kernel name to reflect the `RoutedBlobs` layout with.
pub const MOE_INKLING2_LAMQ_REFLECT_FUNCTION: &str = "moe_inkling2_lamq_phase1";

/// Packs the six trellis LUTs (f16 bits) in `LAMQ_LUT_PACK` order, refusing a
/// missing or wrong-sized table.
pub fn lamq_lut_pack(tables: &[(LamqCode, &[u16])]) -> Result<Vec<u8>, GpuError> {
    let mut out = vec![0u8; LAMQ_LUT_PACK_HALVES * 2];
    for (code, off) in LAMQ_LUT_PACK {
        let t = tables
            .iter()
            .find(|(c, _)| *c == code)
            .map(|(_, t)| *t)
            .ok_or_else(|| GpuError::InvalidInput(format!("LAMQ LUT {} missing", code.name())))?;
        let want = code.table_rows() * code.g();
        if t.len() != want {
            return Err(GpuError::InvalidInput(format!(
                "LAMQ LUT {} has {} values, want {want}",
                code.name(),
                t.len()
            )));
        }
        for (i, v) in t.iter().enumerate() {
            out[2 * (off + i)..2 * (off + i) + 2].copy_from_slice(&v.to_le_bytes());
        }
        if code == LamqCode::Rtr1 {
            // Entry `idx` (14 bits) fixes three consecutive rtr1 columns: the
            // first's 12-bit state is idx >> 2, the second's idx >> 1, the
            // third's idx (k = 1), so slot t holds t[(idx >> (2 - t)) & 4095].
            // The same f16 values as the table, so the weights are exact.
            for idx in 0..1usize << 14 {
                for slot in 0..3 {
                    let v = t[(idx >> (2 - slot)) & 4095];
                    let at = LAMQ_RTR1_MT_OFF + idx * 4 + slot;
                    out[2 * at..2 * at + 2].copy_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
    Ok(out)
}

/// Byte offsets of a LAMQ expert blob's sub-tensors (uniform per layer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LamqExpertOffsets {
    pub fmt: u32,
    pub gu: u32,
    pub gu_s: u32,
    pub dn: u32,
    pub dn_s: u32,
}

impl LamqExpertOffsets {
    fn bytes(&self) -> Result<Vec<u8>, GpuError> {
        // The format record and the code rows are read as u32 words, scales
        // as f16.
        if self.fmt % 4 != 0
            || self.gu % 4 != 0
            || self.dn % 4 != 0
            || self.gu_s % 2 != 0
            || self.dn_s % 2 != 0
        {
            return Err(GpuError::InvalidInput(format!(
                "LAMQ expert offsets misaligned: {self:?}"
            )));
        }
        Ok([self.fmt, self.gu, self.gu_s, self.dn, self.dn_s]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect())
    }
}

/// Where the slots' experts are.
#[derive(Clone, Copy)]
pub enum LamqExperts<'a> {
    /// One streamed blob per slot. The caller marks every bound blob with
    /// `PassEncoder::use_read_buffer`, as for the NVFP4 pair: the kernels
    /// reach them through the argument buffer.
    Streamed(&'a RoutedBlobsBuffer),
    /// A resident layer and a GPU `sel` list.
    Resident(&'a Inkling2ResidentLayer<'a>),
}

fn check(d_dim: u32, f_dim: u32, top_k: u32, experts: &LamqExperts<'_>) -> Result<(), GpuError> {
    if d_dim == 0 || f_dim == 0 {
        return Err(GpuError::InvalidInput(
            "LAMQ experts need positive dims".into(),
        ));
    }
    if d_dim % 16 != 0 {
        return Err(GpuError::InvalidInput(format!(
            "LAMQ experts decode 16 columns at a time; width {d_dim} is not a multiple of 16"
        )));
    }
    if top_k == 0 || top_k as usize > INKLING2_LAMQ_MAX_SLOTS {
        return Err(GpuError::InvalidInput(format!(
            "top_k {top_k} is outside 1..={INKLING2_LAMQ_MAX_SLOTS} for the LAMQ expert pair"
        )));
    }
    if let LamqExperts::Resident(l) = experts {
        if l.n_experts == 0
            || l.stride == 0
            || l.stride % 4 != 0
            || l.buffer.length() < l.stride * u64::from(l.n_experts)
        {
            return Err(GpuError::InvalidInput(format!(
                "resident LAMQ layer: stride {} x {} experts does not fit a {} B buffer (stride must be 4-aligned)",
                l.stride,
                l.n_experts,
                l.buffer.length()
            )));
        }
    }
    Ok(())
}

/// Binds the expert source at buffer 0 (and `sel`, stride, count at 8-10 for
/// a resident layer), then dispatches.
#[allow(clippy::too_many_arguments)]
fn dispatch(
    context: &mut MetalContext,
    pass: &PassEncoder,
    experts: &LamqExperts<'_>,
    streamed_kernel: &'static str,
    resident_kernel: &'static str,
    buffers: &[(&metal::Buffer, u64, u64)],
    bytes: &[(&[u8], u64)],
    groups: u64,
    threads: u64,
) -> Result<(), GpuError> {
    let kernel = match experts {
        LamqExperts::Streamed(_) => streamed_kernel,
        LamqExperts::Resident(_) => resident_kernel,
    };
    let p = context.pipeline(
        SOURCE,
        kernel,
        &moe_function_constants(false),
        &constants_key(false),
    )?;
    let mut b: Vec<(&metal::Buffer, u64, u64)> = buffers.to_vec();
    let mut by: Vec<(&[u8], u64)> = bytes.to_vec();
    let stride_bytes;
    match experts {
        LamqExperts::Streamed(r) => b.push((r.buffer(), 0, 0)),
        LamqExperts::Resident(l) => {
            b.push((l.buffer, 0, 0));
            b.push((l.sel.0, 8, l.sel.1));
            stride_bytes = l.stride.to_le_bytes();
            by.push((&stride_bytes, 9));
            by.push((u32_bytes(&l.n_experts), 10));
        }
    }
    pass.encode_threadgroups(&p, &b, &by, groups, threads);
    Ok(())
}

/// Phase 1: `acts[slot * F + f] = situ(gate_f . x_rot, up_f . x_rot)` for each
/// slot, F32, with `x_rot` the gate-up-rotated F32 input.
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_lamq_phase1(
    context: &mut MetalContext,
    pass: &PassEncoder,
    experts: &LamqExperts<'_>,
    offsets: &LamqExpertOffsets,
    luts: (&metal::Buffer, u64),
    x_rot: (&metal::Buffer, u64),
    acts: (&metal::Buffer, u64),
    d_dim: u32,
    f_dim: u32,
    top_k: u32,
    gate_cap: f32,
    up_cap: f32,
) -> Result<(), GpuError> {
    check(d_dim, f_dim, top_k, experts)?;
    let off = offsets.bytes()?;
    let mut caps = [0u8; 8];
    caps[..4].copy_from_slice(&gate_cap.to_le_bytes());
    caps[4..].copy_from_slice(&up_cap.to_le_bytes());
    let rows = u64::from(top_k) * u64::from(f_dim);
    dispatch(
        context,
        pass,
        experts,
        "moe_inkling2_lamq_phase1",
        "moe_inkling2_lamq_phase1_idx",
        &[
            (x_rot.0, 2, x_rot.1),
            (acts.0, 3, acts.1),
            (luts.0, 11, luts.1),
        ],
        &[
            (&off, 1),
            (u32_bytes(&d_dim), 4),
            (u32_bytes(&f_dim), 5),
            (u32_bytes(&top_k), 6),
            (&caps, 7),
        ],
        rows.div_ceil(ROWS_PER_THREADGROUP),
        PHASE1_THREADS,
    )
}

/// Phase 2: `y[d] = sum over slots in order of routing_w[slot] *
/// (down_d . act_rot[slot])`, FP16 out, no residual.
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_lamq_phase2(
    context: &mut MetalContext,
    pass: &PassEncoder,
    experts: &LamqExperts<'_>,
    offsets: &LamqExpertOffsets,
    luts: (&metal::Buffer, u64),
    act_rot: (&metal::Buffer, u64),
    routing_w: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    d_dim: u32,
    f_dim: u32,
    top_k: u32,
) -> Result<(), GpuError> {
    check(d_dim, f_dim, top_k, experts)?;
    let off = offsets.bytes()?;
    dispatch(
        context,
        pass,
        experts,
        "moe_inkling2_lamq_phase2",
        "moe_inkling2_lamq_phase2_idx",
        &[
            (act_rot.0, 2, act_rot.1),
            (routing_w.0, 3, routing_w.1),
            (y.0, 4, y.1),
            (luts.0, 11, luts.1),
        ],
        &[
            (&off, 1),
            (u32_bytes(&d_dim), 5),
            (u32_bytes(&f_dim), 6),
            (u32_bytes(&top_k), 7),
        ],
        u64::from(d_dim),
        PHASE2_THREADS,
    )
}
