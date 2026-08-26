//! Dense MLP block pass encoding for Muse Glimmer.

use crate::real_forward::RealForwardError;
use crate::real_forward_dispatch::encode_gemv_any;
use crate::real_forward_types::DecodeScratch;
use crate::real_forward_utils::norm_view;

use super::layer_tensor;
use super::state::{POST_NORM_EPS, RMS_EPS};

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_mlp_block(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &model_io::ResidentIndex,
    scratch: &DecodeScratch,
    ffn_hist: Option<&crate::ffn_hist::FfnActHist>,
    layer: usize,
    hidden: usize,
    inter: usize,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;

    // The FFN's input norm, at the STANDARD epsilon again.
    let pre_ffn = norm_view(
        weights,
        index,
        &layer_tensor(layer, "pre_feedforward_layernorm.weight"),
        hidden,
    )?;
    gpu::encode_rms_norm_bf16w_centered(
        context,
        pass,
        (&scratch.x, 0),
        pre_ffn,
        (&scratch.ffn_normed, 0),
        hidden as u32,
        RMS_EPS,
    )
    .map_err(gpu_err)?;

    // A plain dense gated FFN. No router, no experts, and therefore
    // no mid-layer commit: nothing here is data-dependent on a host
    // readback the way an MoE top-k is, so the whole token stays in
    // one command buffer.
    for (suffix, out) in [
        ("mlp.gate_proj.weight", &scratch.ffn_gate),
        ("mlp.up_proj.weight", &scratch.ffn_up),
    ] {
        encode_gemv_any(
            context,
            pass,
            weights,
            index,
            &layer_tensor(layer, suffix),
            inter,
            hidden,
            (&scratch.ffn_normed, 0),
            (out, 0),
        )?;
    }
    // Under the activation census, `silu_mul` writes into a
    // per-layer region of the capture buffer instead of the shared
    // `ffn_act` scratch (which the next layer would overwrite), and
    // `down_proj` reads from the same region. Same kernel, same
    // inputs, different destination address: the math and the
    // generated text are byte-identical either way. See
    // `ffn_hist.rs`.
    let act = match ffn_hist {
        Some(hist) => (&hist.capture, (layer * inter * 2) as u64),
        None => (&scratch.ffn_act, 0),
    };
    gpu::encode_silu_mul(
        context,
        pass,
        (&scratch.ffn_gate, 0),
        (&scratch.ffn_up, 0),
        act,
        inter as u32,
    )
    .map_err(gpu_err)?;
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &layer_tensor(layer, "mlp.down_proj.weight"),
        hidden,
        inter,
        act,
        (&scratch.ffn_out, 0),
    )?;

    // SANDWICH TAIL, HALF TWO. Same shape as the attention half,
    // same POST epsilon, and it reuses `ffn_normed` because that
    // buffer's contents are dead by now.
    let post_ffn = norm_view(
        weights,
        index,
        &layer_tensor(layer, "post_feedforward_layernorm.weight"),
        hidden,
    )?;
    gpu::encode_rms_norm_bf16w_centered(
        context,
        pass,
        (&scratch.ffn_out, 0),
        post_ffn,
        (&scratch.ffn_normed, 0),
        hidden as u32,
        POST_NORM_EPS,
    )
    .map_err(gpu_err)?;
    gpu::encode_residual_add(
        context,
        pass,
        (&scratch.x, 0),
        (&scratch.ffn_normed, 0),
        hidden as u32,
    )
    .map_err(gpu_err)?;

    Ok(())
}
