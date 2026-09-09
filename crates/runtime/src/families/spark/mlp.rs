//! Dense MLP block pass encoding for `spark2_5`.
//!
//! Structurally muse's dense FFN minus the sandwich norms, with one kernel
//! difference that is the family's third first: the gated activation is
//! EXACT-ERF GELU (`encode_gelu_erf_mul`), not silu and not the tanh
//! approximation `encode_gelu_mul` computes. The reference refuses any
//! other `hidden_act`, and `RealSparkState::build` refuses any
//! `hiddenActivation` but `"gelu"` for exactly this reason: the tanh path
//! would be silent, finite, and a different distribution.

use crate::real_forward::RealForwardError;
use crate::real_forward_dispatch::encode_gemv_any;
use crate::real_forward_types::DecodeScratch;
use crate::real_forward_utils::norm_view;

use super::layer_tensor;
use super::state::RMS_EPS;

/// Encodes one layer's FFN half (pre-FFN norm, gated FFN with exact-erf
/// GELU, raw residual add) into `pass`. `x` is read as the residual for the
/// pre-FFN norm and written back with the FFN's output added, both at
/// `x_off` (Gotcha 21's convention: the residual stream is the only buffer
/// needing a row per token in a chunked-prefill driver).
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
    x_off: u64,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;

    // PLAIN norm (`x * w`), not centered: this family has no sandwich or
    // centered convention anywhere. Using muse's centered norm here decodes
    // fluently and is a different model.
    let pre_ffn = norm_view(
        weights,
        index,
        &layer_tensor(layer, "post_attention_layernorm.weight"),
        hidden,
    )?;
    gpu::encode_rms_norm_bf16w(
        context,
        pass,
        (&scratch.x, x_off),
        pre_ffn,
        (&scratch.ffn_normed, 0),
        hidden as u32,
        RMS_EPS,
    )
    .map_err(gpu_err)?;

    // A plain dense gated FFN: no router, no experts, no mid-layer commit,
    // the whole token in one command buffer.
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
    // Under the activation census, the activation writes into a per-layer
    // region of the capture buffer instead of the shared `ffn_act` scratch,
    // and `down_proj` reads from the same region. Same kernel, same inputs,
    // different destination address. See `ffn_hist.rs`.
    let act = match ffn_hist {
        Some(hist) => (&hist.capture, (layer * inter * 2) as u64),
        None => (&scratch.ffn_act, 0),
    };
    // EXACT-ERF GELU on the gate half, then the elementwise multiply with
    // `up` -- `down(gelu_erf(gate) * up)`, never the tanh approximation.
    gpu::encode_gelu_erf_mul(
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

    // RAW residual add: no post-FFN norm, no rescale -- the layer's output
    // joins the stream exactly as the `llama` dense branch's does. This add
    // is also the steering/capture boundary (`mod.rs`).
    gpu::encode_residual_add(
        context,
        pass,
        (&scratch.x, x_off),
        (&scratch.ffn_out, 0),
        hidden as u32,
    )
    .map_err(gpu_err)?;

    Ok(())
}
