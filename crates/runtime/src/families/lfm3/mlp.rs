//! `lfm3`'s FFN half: pre-FFN norm, silu SwiGLU (`w2(silu(w1 x) * w3 x)`),
//! then the post-FFN sandwich norm before the residual add.

use crate::real_forward::RealForwardError;
use crate::real_forward_dispatch::encode_gemv_any;
use crate::real_forward_types::DecodeScratch;
use crate::real_forward_utils::{entry, norm_view, resident_matrix};

use super::layer_tensor;
use super::state::RMS_EPS;

/// `TURBOSPARK_LFM3_FUSED_FFN`, default on, read once. `0` keeps the
/// three-dispatch gate / up / silu chain. Seeds `RealLfm3State::fused_ffn`.
pub(crate) fn lfm3_fused_ffn() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TURBOSPARK_LFM3_FUSED_FFN").as_deref() != Ok("0"))
}

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
    fused_ffn: bool,
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
    let gate_name = layer_tensor(layer, "feed_forward.w1.weight");
    let up_name = layer_tensor(layer, "feed_forward.w3.weight");
    let act = match ffn_hist {
        Some(hist) => (&hist.capture, (layer * inter * 2) as u64),
        None => (&scratch.ffn_act, 0),
    };
    // THE FUSED GATE+UP+SILU (default on): one dispatch for three, bit-exact
    // against the chain below because the kernel is `dequant_int4_gemv_simd`
    // with a second accumulator, rounded to half before the vendored
    // `silu_mul_fp16`. INT4 only, and not where `TURBOSPARK_INT4_MMA8` sends
    // the shape to the shared tile (the qwen guard). Down stays a plain
    // GEMV: `post_ffn_norm` sits between it and the residual add, so the
    // residual epilogue does not apply.
    let int4 = |name: &str| entry(index, name).map(|e| e.dtype == 4).unwrap_or(false);
    if fused_ffn
        && !crate::real_forward_dispatch::int4_mma8_takes(inter, hidden)
        && int4(&gate_name)
        && int4(&up_name)
    {
        foundation::engaged!("lfm3_fused_ffn");
        let gate_m = resident_matrix(weights, index, &gate_name, inter, hidden)?;
        let up_m = resident_matrix(weights, index, &up_name, inter, hidden)?;
        gpu::encode_gate_up_act_gemv(
            context,
            pass,
            &gate_m,
            &up_m,
            (&scratch.ffn_normed, 0),
            act,
            false,
        )
        .map_err(gpu_err)?;
    } else {
        for (name, out) in [(&gate_name, &scratch.ffn_gate), (&up_name, &scratch.ffn_up)] {
            encode_gemv_any(
                context,
                pass,
                weights,
                index,
                name,
                inter,
                hidden,
                (&scratch.ffn_normed, 0),
                (out, 0),
            )?;
        }
        gpu::encode_silu_mul(
            context,
            pass,
            (&scratch.ffn_gate, 0),
            (&scratch.ffn_up, 0),
            act,
            inter as u32,
        )
        .map_err(gpu_err)?;
    }
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
