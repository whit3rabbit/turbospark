//! `lfm3`'s FFN half: pre-FFN norm, silu SwiGLU (`w2(silu(w1 x) * w3 x)`),
//! then the post-FFN sandwich norm before the residual add.

use crate::real_forward::RealForwardError;
use crate::real_forward_dispatch::encode_gemv_any;
use crate::real_forward_types::DecodeScratch;
use crate::real_forward_utils::norm_view;

use super::layer_tensor;
use super::state::RMS_EPS;

/// `x` is read for the pre-FFN norm and written back with the normed FFN
/// output added.
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
    let pre_ffn = norm_view(
        weights,
        index,
        &layer_tensor(layer, "ffn_norm.weight"),
        hidden,
    )?;
    gpu::encode_rms_norm_bf16w(
        context,
        pass,
        (&scratch.x, 0),
        pre_ffn,
        (&scratch.ffn_normed, 0),
        hidden as u32,
        RMS_EPS,
    )
    .map_err(gpu_err)?;

    // `w1` is the gate, `w3` the up (`Lfm3MLP`).
    for (suffix, out) in [
        ("feed_forward.w1.weight", &scratch.ffn_gate),
        ("feed_forward.w3.weight", &scratch.ffn_up),
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
        &layer_tensor(layer, "feed_forward.w2.weight"),
        hidden,
        inter,
        act,
        (&scratch.ffn_out, 0),
    )?;

    // THE POST-FFN SANDWICH NORM, into `ffn_normed` (its pre-FFN contents
    // are consumed by now). Adding `ffn_out` raw is the spark/llama shape and
    // a different model here.
    let post_ffn = norm_view(
        weights,
        index,
        &layer_tensor(layer, "post_ffn_norm.weight"),
        hidden,
    )?;
    gpu::encode_rms_norm_bf16w(
        context,
        pass,
        (&scratch.ffn_out, 0),
        post_ffn,
        (&scratch.ffn_normed, 0),
        hidden as u32,
        RMS_EPS,
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
