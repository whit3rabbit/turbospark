use crate::real_forward_layout::RoutedBlobLayout;

/// The routed-expert decode pair, dispatched for whichever blob layout this
/// install carries: the vendored INT4-affine `moe.metal` kernels or one of
/// the port-local GGUF pairs in `moe_gguf.metal` (ROADMAP Phase G Stage 2).
///
/// A pair of forwarders rather than a branch at each call site, so a layout
/// cannot disagree between two of them and read one blob two ways.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_moe_phase1_any(
    layout: RoutedBlobLayout,
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    routed: &gpu::RoutedBlobsBuffer,
    offsets: &gpu::MoeExpertOffsets,
    x: (&gpu::MetalBuffer, u64),
    acts: (&gpu::MetalBuffer, u64),
    d_dim: u32,
    f_dim: u32,
    top_k: u32,
    use_silu: bool,
) -> Result<(), gpu::GpuError> {
    match layout {
        RoutedBlobLayout::GgufQ8_0 => gpu::encode_moe_phase1_q8_0(
            context, pass, routed, offsets, x, acts, d_dim, f_dim, top_k, use_silu,
        ),
        RoutedBlobLayout::GgufQ4K => gpu::encode_moe_phase1_q4_k(
            context, pass, routed, offsets, x, acts, d_dim, f_dim, top_k, use_silu,
        ),
        RoutedBlobLayout::GgufIq3Xxs => gpu::encode_moe_phase1_iq3_xxs(
            context, pass, routed, offsets, x, acts, d_dim, f_dim, top_k, use_silu,
        ),
        RoutedBlobLayout::GgufIq4Xs => gpu::encode_moe_phase1_iq4_xs(
            context, pass, routed, offsets, x, acts, d_dim, f_dim, top_k, use_silu,
        ),
        // MXFP4 IS `gpt-oss` AND NOTHING ELSE, so the activation constants
        // come from that family here rather than being threaded through
        // every caller of this forwarder. `has_bias` is DERIVED rather than
        // assumed, though: a bias offset of 0 is what
        // `moe_offsets_from_layout` writes when the blob has no bias plane,
        // and 0 cannot be a real one because `gate_w` occupies it. If a
        // second MXFP4 checkpoint ever appears with a different activation,
        // this is the line that has to become a parameter.
        RoutedBlobLayout::GgufMxfp4 => gpu::encode_moe_phase1_mxfp4(
            context,
            pass,
            routed,
            offsets,
            x,
            acts,
            d_dim,
            f_dim,
            top_k,
            use_silu,
            gpu::Mxfp4Activation {
                has_bias: offsets.gate_b != 0,
                ..gpu::Mxfp4Activation::GPT_OSS
            },
        ),
        RoutedBlobLayout::Affine => gpu::encode_moe_phase1(
            context, pass, routed, offsets, x, acts, d_dim, f_dim, top_k, use_silu,
        ),
        // No real file puts IQ4_NL in gate/up, so there is no such kernel.
        // Reaching here means an install this port installed but cannot run;
        // `open()`'s dtype gate lets it through because the TYPE is
        // executable, just not in this position. Same shape as Q6_K.
        other => Err(gpu::GpuError::FunctionNotFound(format!(
            "routed phase 1 (gate/up) for {other:?}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_moe_phase2_any(
    layout: RoutedBlobLayout,
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    routed: &gpu::RoutedBlobsBuffer,
    offsets: &gpu::MoeExpertOffsets,
    acts: (&gpu::MetalBuffer, u64),
    routing_w: (&gpu::MetalBuffer, u64),
    residual: (&gpu::MetalBuffer, u64),
    y: (&gpu::MetalBuffer, u64),
    d_dim: u32,
    f_dim: u32,
    top_k: u32,
    use_silu: bool,
) -> Result<(), gpu::GpuError> {
    match layout {
        RoutedBlobLayout::GgufQ8_0 => gpu::encode_moe_phase2_q8_0(
            context, pass, routed, offsets, acts, routing_w, residual, y, d_dim, f_dim, use_silu,
        ),
        RoutedBlobLayout::GgufQ4K => gpu::encode_moe_phase2_q4_k(
            context, pass, routed, offsets, acts, routing_w, residual, y, d_dim, f_dim, use_silu,
        ),
        RoutedBlobLayout::GgufQ6K => gpu::encode_moe_phase2_q6_k(
            context, pass, routed, offsets, acts, routing_w, residual, y, d_dim, f_dim, use_silu,
        ),
        RoutedBlobLayout::GgufIq4Nl => gpu::encode_moe_phase2_iq4_nl(
            context, pass, routed, offsets, acts, routing_w, residual, y, d_dim, f_dim, use_silu,
        ),
        // The only layout narrow enough to waste phase-2 compute on unused
        // slots (`gpt-oss` routes top-4 of 32 experts against the kernel's
        // fixed 8); see the shader's own comment for the specialization.
        RoutedBlobLayout::GgufMxfp4 => gpu::encode_moe_phase2_mxfp4(
            context,
            pass,
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
            offsets.down_b != 0,
        ),
        RoutedBlobLayout::Affine => gpu::encode_moe_phase2(
            context, pass, routed, offsets, acts, routing_w, residual, y, d_dim, f_dim, use_silu,
        ),
        // No real file puts IQ3_XXS or IQ4_XS in `down`; see the phase-1
        // sibling's note.
        other => Err(gpu::GpuError::FunctionNotFound(format!(
            "routed phase 2 (down) for {other:?}"
        ))),
    }
}
