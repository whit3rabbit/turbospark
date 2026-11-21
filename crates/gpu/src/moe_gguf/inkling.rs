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
//! MXFP4 is refused: its pair binds a bias plane and bakes `top_k` into a
//! function constant, and its phase 1 has its own activation.

use metal::{FunctionConstantValues, MTLDataType};

use super::{encode_phase1_with, encode_phase2_with, MoeExtType, PHASE2_FIXED_SLOTS};
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::moe_decode::{moe_function_constants, MoeExpertOffsets, RoutedBlobsBuffer};

/// One routed GGUF block type, for the Inkling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inkling2GgufExpertType {
    Q8_0,
    Q4K,
    Ext(MoeExtType),
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
            _ => return None,
        })
    }

    pub fn block_elems(self) -> usize {
        match self {
            Self::Q8_0 => 32,
            Self::Q4K => 256,
            Self::Ext(e) => e.block_elems(),
        }
    }

    /// Bytes one row of `n` elements occupies.
    pub fn row_bytes(self, n: usize) -> usize {
        match self {
            Self::Q8_0 => n / 32 * 34,
            Self::Q4K => n / 256 * 144,
            Self::Ext(e) => e.row_bytes(n),
        }
    }

    // Literals, as everywhere in this module: the pipeline cache keys on the
    // name.
    fn phase1_kernel(self) -> &'static str {
        match self {
            Self::Q8_0 => "moe_phase1_gate_up_act_q8_0",
            Self::Q4K => "moe_phase1_gate_up_act_q4_k",
            Self::Ext(e) => e.phase1_kernel(),
        }
    }

    fn phase2_kernel(self) -> &'static str {
        match self {
            Self::Q8_0 => "moe_phase2_down_reduce_k8_q8_0",
            Self::Q4K => "moe_phase2_down_reduce_k8_q4_k",
            Self::Ext(e) => e.phase2_kernel(),
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
    check(kind, "f_dim", f_dim, top_k)?;
    if d_dim == 0 {
        return Err(GpuError::InvalidInput("d_dim must be positive".into()));
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
