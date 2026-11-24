//! Inkling 2's routed-expert mode over the GGUF pairs: the same kernels every
//! GGUF family runs, compiled with function constants 112-114 so phase 1
//! applies the per-expert outer scales and the softcapped SwiGLU and phase 2
//! scales each down value and adds no residual (`moe.metal`'s
//! `moe_gguf_glu` / `moe_gguf_down` / `moe_gguf_residual`). Unset, those
//! constants leave every other family's pipelines unchanged; set, they are in
//! the pipeline key, so the two never share a pipeline.
//!
//! The blob contract adds one thing to the GGUF one: an outer-scale record (an
//! f32 each for gate, up and down) at the `gate_s` / `up_s` / `down_s`
//! offsets, which GGUF blobs otherwise leave unused. The Inkling GGUF repack
//! writes it from the file's per-expert `.scale` tensors, or 1.0.
//!
//! Types: every [`MoeExtType`] plus the hand-written Q8_0 and Q4_K pairs.
//! MXFP4 runs as a DOWN projection only, through its own phase 2 with
//! constant 112 and no bias (`mxfp4::encode_phase2_mxfp4_inkling`); its
//! phase 1 has its own activation, so an MXFP4 gate-up is refused.

use metal::{FunctionConstantValues, MTLDataType};

use super::{encode_phase1_with, encode_phase2_with, MoeExtType, PHASE2_FIXED_SLOTS};
use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::moe_decode::{moe_function_constants, MoeExpertOffsets, RoutedBlobsBuffer};

/// One routed GGUF block type, for the Inkling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inkling2GgufExpertType {
    Q8_0,
    Q4K,
    Ext(MoeExtType),
    /// MXFP4, as a DOWN projection only (one published UD mix ships it on
    /// one layer's down). Its phase 1 has its own activation, so a gate-up
    /// of this type has no Inkling pair and is refused.
    Mxfp4,
}

impl Inkling2GgufExpertType {
    /// From a `layout.json` sub-tensor dtype (the GGUF repack's lowercase
    /// ggml names). None for a type with no Inkling-mode pair.
    pub fn from_dtype(dtype: &str) -> Option<Self> {
        use MoeExtType as E;
        Some(match dtype {
            "q8_0" => Self::Q8_0,
            "q4_k" => Self::Q4K,
            "q5_k" => Self::Ext(E::Q5K),
            "q3_k" => Self::Ext(E::Q3K),
            "q2_k" => Self::Ext(E::Q2K),
            "q6_k" => Self::Ext(E::Q6K),
            "iq4_xs" => Self::Ext(E::Iq4Xs),
            "iq4_nl" => Self::Ext(E::Iq4Nl),
            "iq3_xxs" => Self::Ext(E::Iq3Xxs),
            "iq3_s" => Self::Ext(E::Iq3S),
            "iq2_s" => Self::Ext(E::Iq2S),
            "iq2_xs" => Self::Ext(E::Iq2Xs),
            "iq2_xxs" => Self::Ext(E::Iq2Xxs),
            "iq1_s" => Self::Ext(E::Iq1S),
            "iq1_m" => Self::Ext(E::Iq1M),
            "q4_0" => Self::Ext(E::Q4_0),
            "q4_1" => Self::Ext(E::Q4_1),
            "q5_0" => Self::Ext(E::Q5_0),
            "q5_1" => Self::Ext(E::Q5_1),
            "q2_0" => Self::Ext(E::Q2_0),
            "nvfp4" => Self::Ext(E::Nvfp4),
            "q1_0" => Self::Ext(E::Q1_0),
            "tq1_0" => Self::Ext(E::Tq1_0),
            "tq2_0" => Self::Ext(E::Tq2_0),
            // Unsloth fork types below IQ1_S (ggml 64 to 66).
            "iq1_xs" => Self::Ext(E::Iq1Xs),
            "iq1_xxs" => Self::Ext(E::Iq1Xxs),
            "iq1_xxxs" => Self::Ext(E::Iq1Xxxs),
            "mxfp4" => Self::Mxfp4,
            _ => return None,
        })
    }

    pub fn block_elems(self) -> usize {
        match self {
            Self::Q8_0 => 32,
            Self::Q4K => 256,
            Self::Ext(e) => e.block_elems(),
            Self::Mxfp4 => super::MXFP4_BLOCK_ELEMS,
        }
    }

    /// Bytes one row of `n` elements occupies.
    pub fn row_bytes(self, n: usize) -> usize {
        match self {
            Self::Q8_0 => n / 32 * 34,
            Self::Q4K => n / 256 * 144,
            Self::Ext(e) => e.row_bytes(n),
            Self::Mxfp4 => super::mxfp4_row_bytes(n),
        }
    }

    // Literals, as everywhere in this module: the pipeline cache keys on the
    // name.
    fn phase1_kernel(self) -> &'static str {
        match self {
            Self::Q8_0 => "moe_phase1_gate_up_act_q8_0",
            Self::Q4K => "moe_phase1_gate_up_act_q4_k",
            Self::Ext(e) => e.phase1_kernel(),
            // Refused before any dispatch (see `encode_inkling2_gguf_phase1_table`).
            Self::Mxfp4 => "moe_phase1_gate_up_act_mxfp4",
        }
    }

    fn phase2_kernel(self) -> &'static str {
        match self {
            Self::Q8_0 => "moe_phase2_down_reduce_k8_q8_0",
            Self::Q4K => "moe_phase2_down_reduce_k8_q4_k",
            Self::Ext(e) => e.phase2_kernel(),
            // Dispatched by its own encoder (different bindings).
            Self::Mxfp4 => "moe_phase2_down_reduce_k8_mxfp4",
        }
    }
}

/// The Inkling mode's function constants: the GGUF defaults (`use_silu`
/// false, unused there) plus 112 = true and, for phase 1, the two caps. The
/// key names every one of them, so no Inkling pipeline is ever reused for
/// another family or for other caps.
fn inkling_constants(caps: Option<(f32, f32)>) -> (FunctionConstantValues, Vec<u8>) {
    let values = moe_function_constants(false);
    let on = true;
    values.set_constant_value_at_index((&on as *const bool).cast(), MTLDataType::Bool, 112);
    let mut key = vec![0u8, 1u8];
    if let Some((g, u)) = caps {
        values.set_constant_value_at_index((&g as *const f32).cast(), MTLDataType::Float, 113);
        values.set_constant_value_at_index((&u as *const f32).cast(), MTLDataType::Float, 114);
        key.extend_from_slice(&g.to_le_bytes());
        key.extend_from_slice(&u.to_le_bytes());
    }
    (values, key)
}

fn check(kind: Inkling2GgufExpertType, label: &str, dim: u32, top_k: u32) -> Result<(), GpuError> {
    if dim == 0 || dim as usize % kind.block_elems() != 0 {
        return Err(GpuError::InvalidInput(format!(
            "{label} ({dim}) must be a positive multiple of {} for Inkling 2 GGUF {kind:?}",
            kind.block_elems()
        )));
    }
    // Phase 2 reduces a fixed eight slots; more would be truncated silently.
    if top_k == 0 || top_k as usize > PHASE2_FIXED_SLOTS {
        return Err(GpuError::InvalidInput(format!(
            "top_k {top_k} is outside 1..={PHASE2_FIXED_SLOTS} for the GGUF expert pair"
        )));
    }
    Ok(())
}

fn check_caps(gate_cap: f32, up_cap: f32) -> Result<(), GpuError> {
    if !(gate_cap.is_finite() && gate_cap > 0.0 && up_cap.is_finite() && up_cap > 0.0) {
        return Err(GpuError::InvalidInput(format!(
            "Inkling softcaps {gate_cap}, {up_cap} must be positive"
        )));
    }
    Ok(())
}

/// Phase 1 in the Inkling mode: `acts[slot * F + f] = softcap(g, gate_cap) *
/// sigmoid(g) * softcap(u, up_cap)`, `g` and `u` the slot's rows dotted with
/// FP16 `x` and multiplied by the blob's outer scales. The caller makes every
/// bound blob resident (`PassEncoder::use_read_buffer`).
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_gguf_phase1(
    context: &mut MetalContext,
    pass: &PassEncoder,
    kind: Inkling2GgufExpertType,
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
    encode_inkling2_gguf_phase1_table(
        context,
        pass,
        kind,
        (routed.buffer(), 0),
        offsets,
        x,
        acts,
        d_dim,
        f_dim,
        top_k,
        gate_cap,
        up_cap,
    )
}

/// [`encode_inkling2_gguf_phase1`] over a blob table at `table` (one of
/// [`encode_inkling2_gguf_bind_rows`]'s rows, or a bound
/// `RoutedBlobsBuffer` at offset 0).
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_gguf_phase1_table(
    context: &mut MetalContext,
    pass: &PassEncoder,
    kind: Inkling2GgufExpertType,
    routed: (&metal::Buffer, u64),
    offsets: &MoeExpertOffsets,
    x: (&metal::Buffer, u64),
    acts: (&metal::Buffer, u64),
    d_dim: u32,
    f_dim: u32,
    top_k: u32,
    gate_cap: f32,
    up_cap: f32,
) -> Result<(), GpuError> {
    if kind == Inkling2GgufExpertType::Mxfp4 {
        return Err(GpuError::InvalidInput(
            "MXFP4 has no Inkling gate-up pair (its phase 1 has its own activation); only an \
             MXFP4 down projection runs"
                .into(),
        ));
    }
    check(kind, "d_dim", d_dim, top_k)?;
    check_caps(gate_cap, up_cap)?;
    if f_dim == 0 {
        return Err(GpuError::InvalidInput("f_dim must be positive".into()));
    }
    let (values, key) = inkling_constants(Some((gate_cap, up_cap)));
    encode_phase1_with(
        context,
        pass,
        kind.phase1_kernel(),
        &values,
        &key,
        routed,
        offsets,
        x,
        acts,
        d_dim,
        f_dim,
        top_k,
    )
}

/// Phase 2 in the Inkling mode: `y[d] = sum over the eight slots of
/// routing_w[slot] * outer_down * down_d(acts[slot])`, no residual. Unused
/// slots need a zero routing weight, a valid blob and a finite acts row, as
/// for every GGUF phase 2.
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_gguf_phase2(
    context: &mut MetalContext,
    pass: &PassEncoder,
    kind: Inkling2GgufExpertType,
    routed: &RoutedBlobsBuffer,
    offsets: &MoeExpertOffsets,
    acts: (&metal::Buffer, u64),
    routing_w: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    d_dim: u32,
    f_dim: u32,
    top_k: u32,
) -> Result<(), GpuError> {
    encode_inkling2_gguf_phase2_table(
        context,
        pass,
        kind,
        (routed.buffer(), 0),
        offsets,
        acts,
        routing_w,
        y,
        d_dim,
        f_dim,
        top_k,
    )
}

/// [`encode_inkling2_gguf_phase2`] over a blob table at `table`.
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_gguf_phase2_table(
    context: &mut MetalContext,
    pass: &PassEncoder,
    kind: Inkling2GgufExpertType,
    routed: (&metal::Buffer, u64),
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
    if kind == Inkling2GgufExpertType::Mxfp4 {
        return super::mxfp4::encode_phase2_mxfp4_inkling(
            context, pass, routed, offsets, acts, routing_w, y, d_dim, f_dim, top_k,
        );
    }
    // The residual slot is bound to `acts` and never read in this mode.
    let (values, key) = inkling_constants(None);
    encode_phase2_with(
        context,
        pass,
        kind.phase2_kernel(),
        &values,
        &key,
        routed,
        offsets,
        acts,
        routing_w,
        acts,
        y,
        d_dim,
        f_dim,
    )
}

/// Bytes of one blob table (`RoutedBlobs`: `MAX_STREAMED_EXPERTS` GPU
/// addresses), the row stride of [`encode_inkling2_gguf_bind_rows`]'s output.
pub const INKLING2_GGUF_TABLE_BYTES: u64 = (crate::moe_decode::MAX_STREAMED_EXPERTS * 8) as u64;

/// One blob table per row for resident GGUF experts: row `r`'s slot `k`
/// points at `layer + sel[r * sel_stride + k] * stride` (slots past `top_k`
/// repeat the last, as `RoutedBlobsBuffer::bind_for` fills them). The tables
/// feed [`encode_inkling2_gguf_phase1_table`] and
/// [`encode_inkling2_gguf_phase2_table`], so a resident row runs the
/// streamed decode pair's own arithmetic. The caller makes `layer` resident
/// in the pass (`PassEncoder::use_read_buffer`): the pair reads it only
/// through the table.
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_gguf_bind_rows(
    context: &mut MetalContext,
    pass: &PassEncoder,
    layer: &metal::Buffer,
    stride: u64,
    sel: (&metal::Buffer, u64),
    sel_stride: u32,
    tables: (&metal::Buffer, u64),
    top_k: u32,
    n_rows: u32,
) -> Result<(), GpuError> {
    if top_k == 0 || top_k > sel_stride || top_k as usize > crate::moe_decode::MAX_STREAMED_EXPERTS
    {
        return Err(GpuError::InvalidInput(format!(
            "GGUF blob tables: top_k {top_k} with {sel_stride} ids per row"
        )));
    }
    if n_rows == 0 || tables.1 % 8 != 0 {
        return Err(GpuError::InvalidInput(
            "GGUF blob tables: no rows, or an unaligned table".into(),
        ));
    }
    let pipeline = context.pipeline(
        super::SOURCE,
        "inkling2_gguf_bind_rows",
        &FunctionConstantValues::new(),
        &[],
    )?;
    let threads = u64::from(n_rows) * crate::moe_decode::MAX_STREAMED_EXPERTS as u64;
    pass.encode_threadgroups(
        &pipeline,
        &[(layer, 0, 0), (sel.0, 1, sel.1), (tables.0, 2, tables.1)],
        &[
            (&stride.to_le_bytes(), 3),
            (u32_bytes(&top_k), 4),
            (u32_bytes(&sel_stride), 5),
            (u32_bytes(&n_rows), 6),
        ],
        threads.div_ceil(64),
        64,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The repack's lowercase ggml name for each type. Exhaustive on
    /// purpose: a new `MoeExtType` does not compile until it is named here,
    /// and the test below then fails until `from_dtype` maps it, so a file
    /// cannot pass the intake and fail at open for want of a name.
    fn name(e: MoeExtType) -> &'static str {
        use MoeExtType as E;
        match e {
            E::Q4_0 => "q4_0",
            E::Q4_1 => "q4_1",
            E::Q5_0 => "q5_0",
            E::Q5_1 => "q5_1",
            E::Q1_0 => "q1_0",
            E::Tq2_0 => "tq2_0",
            E::Tq1_0 => "tq1_0",
            E::Nvfp4 => "nvfp4",
            E::Q5K => "q5_k",
            E::Q3K => "q3_k",
            E::Q2K => "q2_k",
            E::Iq2Xs => "iq2_xs",
            E::Iq1S => "iq1_s",
            E::Iq3S => "iq3_s",
            E::Iq3Xxs => "iq3_xxs",
            E::Iq4Xs => "iq4_xs",
            E::Iq2S => "iq2_s",
            E::Iq2Xxs => "iq2_xxs",
            E::Iq1M => "iq1_m",
            E::Iq4Nl => "iq4_nl",
            E::Q6K => "q6_k",
            E::Q2_0 => "q2_0",
            E::Iq1Xs => "iq1_xs",
            E::Iq1Xxs => "iq1_xxs",
            E::Iq1Xxxs => "iq1_xxxs",
        }
    }

    #[test]
    fn every_ext_type_with_an_inkling_pair_is_named() {
        for &e in &MoeExtType::ALL {
            assert_eq!(
                Inkling2GgufExpertType::from_dtype(name(e)),
                Some(Inkling2GgufExpertType::Ext(e)),
                "{e:?} has Inkling-mode kernels but from_dtype does not map {:?}",
                name(e)
            );
        }
        assert_eq!(
            Inkling2GgufExpertType::from_dtype("q8_0"),
            Some(Inkling2GgufExpertType::Q8_0)
        );
        assert_eq!(
            Inkling2GgufExpertType::from_dtype("q4_k"),
            Some(Inkling2GgufExpertType::Q4K)
        );
        // MXFP4 maps (down only; its gate-up is refused at dispatch).
        assert_eq!(
            Inkling2GgufExpertType::from_dtype("mxfp4"),
            Some(Inkling2GgufExpertType::Mxfp4)
        );
    }
}
