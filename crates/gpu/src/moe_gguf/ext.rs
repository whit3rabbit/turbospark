//! Host-side dispatch for `shaders/moe_gguf_ext.metal`: the routed-expert
//! decode pair for the element-addressed block types (legacy Q4_0, Q4_1,
//! Q5_0, Q5_1 and the ext family Q1_0, TQ2_0, TQ1_0, NVFP4). One entry point
//! per phase, keyed by [`MoeExtType`], because the dispatch shape is the
//! Q8_0 pair's and only the kernel name and block geometry differ.

use super::{encode_phase1, encode_phase2};
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::moe_decode::{MoeExpertOffsets, RoutedBlobsBuffer};

/// Which element-addressed block type a routed blob holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeExtType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q1_0,
    Tq2_0,
    Tq1_0,
    Nvfp4,
    Q5K,
    Q3K,
    Q2K,
    Iq2Xs,
    Iq1S,
    Iq3S,
    // The types below already had ONE hand-written phase in `moe_gguf.metal`;
    // the other phase is generated. Kernel names follow one convention, so
    // this API reaches both halves.
    Iq3Xxs,
    Iq4Xs,
    Iq2S,
    Iq2Xxs,
    Iq1M,
    Iq4Nl,
    Q6K,
    /// Q2_0 phase 1 is hand-written; its k8 phase 2 is generated (the
    /// hand-written phase 2 reduces exactly ten slots).
    Q2_0,
    /// Unsloth fork IQ1_XS, IQ1_XXS and IQ1_XXXS (ggml ids 64 to 66).
    Iq1Xs,
    Iq1Xxs,
    Iq1Xxxs,
}

impl MoeExtType {
    /// Every type, for tests that sweep them.
    pub const ALL: [MoeExtType; 25] = [
        MoeExtType::Q4_0,
        MoeExtType::Q4_1,
        MoeExtType::Q5_0,
        MoeExtType::Q5_1,
        MoeExtType::Q1_0,
        MoeExtType::Tq2_0,
        MoeExtType::Tq1_0,
        MoeExtType::Nvfp4,
        MoeExtType::Q5K,
        MoeExtType::Q3K,
        MoeExtType::Q2K,
        MoeExtType::Iq2Xs,
        MoeExtType::Iq1S,
        MoeExtType::Iq3S,
        MoeExtType::Iq3Xxs,
        MoeExtType::Iq4Xs,
        MoeExtType::Iq2S,
        MoeExtType::Iq2Xxs,
        MoeExtType::Iq1M,
        MoeExtType::Iq4Nl,
        MoeExtType::Q6K,
        MoeExtType::Q2_0,
        MoeExtType::Iq1Xs,
        MoeExtType::Iq1Xxs,
        MoeExtType::Iq1Xxxs,
    ];

    /// Elements per block; a routed row must be a whole number of them.
    pub fn block_elems(self) -> usize {
        match self {
            MoeExtType::Q4_0 | MoeExtType::Q4_1 | MoeExtType::Q5_0 | MoeExtType::Q5_1 => 32,
            MoeExtType::Q1_0 => 128,
            MoeExtType::Nvfp4 => 64,
            MoeExtType::Iq4Nl => 32,
            MoeExtType::Q2_0 => 64,
            MoeExtType::Tq2_0
            | MoeExtType::Tq1_0
            | MoeExtType::Q5K
            | MoeExtType::Q3K
            | MoeExtType::Q2K
            | MoeExtType::Iq2Xs
            | MoeExtType::Iq1S
            | MoeExtType::Iq3S
            | MoeExtType::Iq3Xxs
            | MoeExtType::Iq4Xs
            | MoeExtType::Iq2S
            | MoeExtType::Iq2Xxs
            | MoeExtType::Iq1M
            | MoeExtType::Q6K
            | MoeExtType::Iq1Xs
            | MoeExtType::Iq1Xxs
            | MoeExtType::Iq1Xxxs => 256,
        }
    }

    /// Bytes per block.
    pub fn block_bytes(self) -> usize {
        match self {
            MoeExtType::Q4_0 => 18,
            MoeExtType::Q4_1 => 20,
            MoeExtType::Q5_0 => 22,
            MoeExtType::Q5_1 => 24,
            MoeExtType::Q1_0 => 18,
            MoeExtType::Tq2_0 => 66,
            MoeExtType::Tq1_0 => 54,
            MoeExtType::Nvfp4 => 36,
            MoeExtType::Q5K => 176,
            MoeExtType::Q3K => 110,
            MoeExtType::Q2K => 84,
            MoeExtType::Iq2Xs => 74,
            MoeExtType::Iq1S => 50,
            MoeExtType::Iq3S => 110,
            MoeExtType::Iq3Xxs => 98,
            MoeExtType::Iq4Xs => 136,
            MoeExtType::Iq2S => 82,
            MoeExtType::Iq2Xxs => 66,
            MoeExtType::Iq1M => 56,
            MoeExtType::Iq4Nl => 18,
            MoeExtType::Q6K => 210,
            MoeExtType::Q2_0 => 18,
            MoeExtType::Iq1Xs => 46,
            MoeExtType::Iq1Xxs => 42,
            MoeExtType::Iq1Xxxs => 38,
        }
    }

    /// Bytes a routed row of `n` elements occupies.
    pub fn row_bytes(self, n: usize) -> usize {
        n / self.block_elems() * self.block_bytes()
    }

    // Every kernel this API reaches is `moe_phase1_gate_up_act_<suffix>` or
    // `moe_phase2_down_reduce_k8_<suffix>`, generated in `moe_gguf_ext.metal`
    // or hand-written in `moe_gguf.metal`. The names are literals because the
    // pipeline cache keys on the `&'static str`'s address.
    pub(crate) fn phase1_kernel(self) -> &'static str {
        match self {
            MoeExtType::Q4_0 => "moe_phase1_gate_up_act_q4_0",
            MoeExtType::Q4_1 => "moe_phase1_gate_up_act_q4_1",
            MoeExtType::Q5_0 => "moe_phase1_gate_up_act_q5_0",
            MoeExtType::Q5_1 => "moe_phase1_gate_up_act_q5_1",
            MoeExtType::Q1_0 => "moe_phase1_gate_up_act_q1_0",
            MoeExtType::Tq2_0 => "moe_phase1_gate_up_act_tq2_0",
            MoeExtType::Tq1_0 => "moe_phase1_gate_up_act_tq1_0",
            MoeExtType::Nvfp4 => "moe_phase1_gate_up_act_nvfp4",
            MoeExtType::Q5K => "moe_phase1_gate_up_act_q5_k",
            MoeExtType::Q3K => "moe_phase1_gate_up_act_q3_k",
            MoeExtType::Q2K => "moe_phase1_gate_up_act_q2_k",
            MoeExtType::Iq2Xs => "moe_phase1_gate_up_act_iq2_xs",
            MoeExtType::Iq1S => "moe_phase1_gate_up_act_iq1_s",
            MoeExtType::Iq3S => "moe_phase1_gate_up_act_iq3_s",
            MoeExtType::Iq3Xxs => "moe_phase1_gate_up_act_iq3_xxs",
            MoeExtType::Iq4Xs => "moe_phase1_gate_up_act_iq4_xs",
            MoeExtType::Iq2S => "moe_phase1_gate_up_act_iq2_s",
            MoeExtType::Iq2Xxs => "moe_phase1_gate_up_act_iq2_xxs",
            MoeExtType::Iq1M => "moe_phase1_gate_up_act_iq1_m",
            MoeExtType::Iq4Nl => "moe_phase1_gate_up_act_iq4_nl",
            MoeExtType::Q6K => "moe_phase1_gate_up_act_q6_k",
            MoeExtType::Q2_0 => "moe_phase1_gate_up_act_q2_0",
            MoeExtType::Iq1Xs => "moe_phase1_gate_up_act_iq1_xs",
            MoeExtType::Iq1Xxs => "moe_phase1_gate_up_act_iq1_xxs",
            MoeExtType::Iq1Xxxs => "moe_phase1_gate_up_act_iq1_xxxs",
        }
    }

    /// The ten-slot phase 2, for a model that routes exactly ten experts.
    /// `None` for the types whose phase 2 is hand-written (Q8_0, Q4_K, Q6_K,
    /// IQ4_NL, MXFP4) and for Q2_0, which has its own k10 entry point.
    fn phase2_kernel_k10(self) -> Option<&'static str> {
        Some(match self {
            MoeExtType::Q4_0 => "moe_phase2_down_reduce_k10_q4_0",
            MoeExtType::Q4_1 => "moe_phase2_down_reduce_k10_q4_1",
            MoeExtType::Q5_0 => "moe_phase2_down_reduce_k10_q5_0",
            MoeExtType::Q5_1 => "moe_phase2_down_reduce_k10_q5_1",
            MoeExtType::Q1_0 => "moe_phase2_down_reduce_k10_q1_0",
            MoeExtType::Tq2_0 => "moe_phase2_down_reduce_k10_tq2_0",
            MoeExtType::Tq1_0 => "moe_phase2_down_reduce_k10_tq1_0",
            MoeExtType::Nvfp4 => "moe_phase2_down_reduce_k10_nvfp4",
            MoeExtType::Q5K => "moe_phase2_down_reduce_k10_q5_k",
            MoeExtType::Q3K => "moe_phase2_down_reduce_k10_q3_k",
            MoeExtType::Q2K => "moe_phase2_down_reduce_k10_q2_k",
            MoeExtType::Iq2Xs => "moe_phase2_down_reduce_k10_iq2_xs",
            MoeExtType::Iq1S => "moe_phase2_down_reduce_k10_iq1_s",
            MoeExtType::Iq3S => "moe_phase2_down_reduce_k10_iq3_s",
            MoeExtType::Iq3Xxs => "moe_phase2_down_reduce_k10_iq3_xxs",
            MoeExtType::Iq4Xs => "moe_phase2_down_reduce_k10_iq4_xs",
            MoeExtType::Iq2S => "moe_phase2_down_reduce_k10_iq2_s",
            MoeExtType::Iq2Xxs => "moe_phase2_down_reduce_k10_iq2_xxs",
            MoeExtType::Iq1M => "moe_phase2_down_reduce_k10_iq1_m",
            MoeExtType::Iq1Xs => "moe_phase2_down_reduce_k10_iq1_xs",
            MoeExtType::Iq1Xxs => "moe_phase2_down_reduce_k10_iq1_xxs",
            MoeExtType::Iq1Xxxs => "moe_phase2_down_reduce_k10_iq1_xxxs",
            MoeExtType::Iq4Nl | MoeExtType::Q6K | MoeExtType::Q2_0 => return None,
        })
    }

    pub(crate) fn phase2_kernel(self) -> &'static str {
        match self {
            MoeExtType::Q4_0 => "moe_phase2_down_reduce_k8_q4_0",
            MoeExtType::Q4_1 => "moe_phase2_down_reduce_k8_q4_1",
            MoeExtType::Q5_0 => "moe_phase2_down_reduce_k8_q5_0",
            MoeExtType::Q5_1 => "moe_phase2_down_reduce_k8_q5_1",
            MoeExtType::Q1_0 => "moe_phase2_down_reduce_k8_q1_0",
            MoeExtType::Tq2_0 => "moe_phase2_down_reduce_k8_tq2_0",
            MoeExtType::Tq1_0 => "moe_phase2_down_reduce_k8_tq1_0",
            MoeExtType::Nvfp4 => "moe_phase2_down_reduce_k8_nvfp4",
            MoeExtType::Q5K => "moe_phase2_down_reduce_k8_q5_k",
            MoeExtType::Q3K => "moe_phase2_down_reduce_k8_q3_k",
            MoeExtType::Q2K => "moe_phase2_down_reduce_k8_q2_k",
            MoeExtType::Iq2Xs => "moe_phase2_down_reduce_k8_iq2_xs",
            MoeExtType::Iq1S => "moe_phase2_down_reduce_k8_iq1_s",
            MoeExtType::Iq3S => "moe_phase2_down_reduce_k8_iq3_s",
            MoeExtType::Iq3Xxs => "moe_phase2_down_reduce_k8_iq3_xxs",
            MoeExtType::Iq4Xs => "moe_phase2_down_reduce_k8_iq4_xs",
            MoeExtType::Iq2S => "moe_phase2_down_reduce_k8_iq2_s",
            MoeExtType::Iq2Xxs => "moe_phase2_down_reduce_k8_iq2_xxs",
            MoeExtType::Iq1M => "moe_phase2_down_reduce_k8_iq1_m",
            MoeExtType::Iq4Nl => "moe_phase2_down_reduce_k8_iq4_nl",
            MoeExtType::Q6K => "moe_phase2_down_reduce_k8_q6_k",
            MoeExtType::Q2_0 => "moe_phase2_down_reduce_k8_q2_0",
            MoeExtType::Iq1Xs => "moe_phase2_down_reduce_k8_iq1_xs",
            MoeExtType::Iq1Xxs => "moe_phase2_down_reduce_k8_iq1_xxs",
            MoeExtType::Iq1Xxxs => "moe_phase2_down_reduce_k8_iq1_xxxs",
        }
    }
}

fn require_tiling(kind: MoeExtType, label: &str, dim: u32) -> Result<(), GpuError> {
    if dim as usize % kind.block_elems() != 0 {
        return Err(GpuError::InvalidInput(format!(
            "{label} ({dim}) must be a multiple of {} for {kind:?}",
            kind.block_elems()
        )));
    }
    Ok(())
}

/// Phase 1 over routed gate/up rows of `kind`.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_phase1_ext(
    context: &mut MetalContext,
    pass: &PassEncoder,
    kind: MoeExtType,
    routed: &RoutedBlobsBuffer,
    offsets: &MoeExpertOffsets,
    x: (&metal::Buffer, u64),
    acts: (&metal::Buffer, u64),
    d_dim: u32,
    f_dim: u32,
    top_k: u32,
    use_silu: bool,
) -> Result<(), GpuError> {
    require_tiling(kind, "d_dim", d_dim)?;
    encode_phase1(
        context,
        pass,
        kind.phase1_kernel(),
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

/// Phase 2 over routed down rows of `kind`; reduces the fixed eight slots.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_phase2_ext(
    context: &mut MetalContext,
    pass: &PassEncoder,
    kind: MoeExtType,
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
    require_tiling(kind, "f_dim", f_dim)?;
    // Exactly ten routed experts take the ten-slot kernel; 9 would reduce a
    // slot nothing wrote, so it is refused rather than padded.
    if top_k > super::PHASE2_FIXED_SLOTS as u32 {
        let kernel = kind.phase2_kernel_k10().ok_or_else(|| {
            GpuError::InvalidInput(format!(
                "no ten-slot routed phase 2 exists for {kind:?}; top_k {top_k} exceeds the \
                 eight-slot kernel"
            ))
        })?;
        if top_k != 10 {
            return Err(GpuError::InvalidInput(format!(
                "routed phase 2 serves top_k up to 8 or exactly 10, got {top_k}"
            )));
        }
        return super::encode_phase2_k10(
            context, pass, kernel, routed, offsets, acts, routing_w, residual, y, d_dim, f_dim,
            use_silu,
        );
    }
    encode_phase2(
        context,
        pass,
        kind.phase2_kernel(),
        routed,
        offsets,
        acts,
        routing_w,
        residual,
        y,
        d_dim,
        f_dim,
        top_k,
        use_silu,
    )
}
