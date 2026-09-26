//! One hyper-connection call (`attn_hc`, `mlp_hc`, or the final
//! `hyper_connection_mixer`): `mod.rs`'s "## Hyper-connections" pseudocode,
//! verified against `Qwen4ExpTextGatedResidual.forward` in
//! `modular_qwen4_exp.py` (`fc5c5bde8`). This runtime consumes llama.cpp GGUF
//! weights, whose converter already folds centered norm weights into the
//! stored scale, so the GPU path applies that scale directly.

use model_io::ResidentIndex;

use crate::families::qwen4::state::RealQwen4State;
use crate::real_forward_dispatch::encode_gemv_any;
use crate::real_forward_types::RealForwardError;
use crate::real_forward_utils::norm_view;

/// Encodes one `Qwen4ExpTextGatedResidual.forward` call.
///
/// `wide` is the `hc_count * hidden`-wide residual (`raw`), READ but never
/// written here except by [`gpu::encode_rms_norm_bf16w_grouped`]'s
/// destination (`qwen4.hc_normed`, a scratch buffer). The caller injects
/// the sublayer's output back into `wide` separately
/// (`gpu::encode_hc_inject_add`), after running the sublayer this call's
/// `mixed` output feeds.
///
/// `mixed_out` receives the `hidden`-wide mix (typically `scratch.normed`,
/// which every sublayer already reads its normalized input from).
/// `qwen4.hc_inject` receives the `hc_count`-wide inject gate at
/// `inject_row_offset` when `has_inject_gate` is true; the caller reads it
/// from there directly (`gpu::encode_hc_inject_add`'s `inject_w` argument,
/// at the same offset). `has_inject_gate` is false only for the final
/// `hyper_connection_mixer`, which has no `block_inject_weight` and returns
/// `mixed` alone -- `inject_row_offset` is ignored in that case.
///
/// `inject_row_offset` is `0` on the sequential decode path (single row);
/// inside a chunked-prefill micro-batch it is the calling token's own row
/// (`t * hc_count * 2`), because `mlp_hc`'s inject gate is read by a SECOND
/// `encode_hc_inject_add` call one command-buffer commit later than this one
/// writes it (`RealQwen4State::hc_inject`'s own doc has the full argument):
/// without a per-token row, a later token's `mlp_hc` call in the same
/// micro-batch would silently overwrite an earlier token's gate before it is
/// read back.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_hyper_connection(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    qwen4: &RealQwen4State,
    wide: (&gpu::MetalBuffer, u64),
    name_prefix: &str,
    hidden: usize,
    mixed_out: (&gpu::MetalBuffer, u64),
    has_inject_gate: bool,
    inject_row_offset: u64,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let hc_count = qwen4.hc_count;
    let wide_dim = hidden * hc_count;
    let name = |suffix: &str| format!("{name_prefix}.{suffix}");

    // GGUF stores the already-shifted scale for hc_norm, so apply weight
    // directly after reducing each hidden-width residual stream.
    let hc_norm_w = norm_view(weights, index, &name("hc_norm.weight"), wide_dim)?;
    gpu::encode_rms_norm_bf16w_grouped(
        context,
        pass,
        wide,
        hc_norm_w,
        (&qwen4.hc_normed, 0),
        hc_count as u32,
        hidden as u32,
        crate::families::qwen4::RMS_EPS,
    )
    .map_err(gpu_err)?;

    // w = silu(down(normed) / hc_count)
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &name("input_mix_weight_down.weight"),
        qwen4.hc_lowrank,
        wide_dim,
        (&qwen4.hc_normed, 0),
        (&qwen4.hc_low, 0),
    )?;
    gpu::encode_scalar_mul(
        context,
        pass,
        (&qwen4.hc_low, 0),
        1.0 / hc_count as f32,
        qwen4.hc_lowrank as u32,
    )
    .map_err(gpu_err)?;
    gpu::encode_silu(context, pass, (&qwen4.hc_low, 0), qwen4.hc_lowrank as u32)
        .map_err(gpu_err)?;

    // w = sigmoid(up(w)) -- NO division here; only the down-projection's
    // input and the inject weight's input are scaled by hc_count
    // (`docs/QWEN4_PHASE0.md` section 3's five easy-to-drop details, the
    // "two `/ C` divisions" one: there are exactly two, not three).
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &name("input_mix_weight_up.weight"),
        wide_dim,
        qwen4.hc_lowrank,
        (&qwen4.hc_low, 0),
        (&qwen4.hc_up, 0),
    )?;
    gpu::encode_sigmoid(context, pass, (&qwen4.hc_up, 0), wide_dim as u32).map_err(gpu_err)?;

    // mixed = (w.view(C,H) * normed.view(C,H)).mean(dim=C)
    gpu::encode_hc_mix(
        context,
        pass,
        (&qwen4.hc_up, 0),
        (&qwen4.hc_normed, 0),
        mixed_out,
        hc_count as u32,
        hidden as u32,
    )
    .map_err(gpu_err)?;

    if has_inject_gate {
        // inject_w = 2 * sigmoid(block_inject(normed) / hc_count)
        encode_gemv_any(
            context,
            pass,
            weights,
            index,
            &name("block_inject_weight.weight"),
            hc_count,
            wide_dim,
            (&qwen4.hc_normed, 0),
            (&qwen4.hc_inject, inject_row_offset),
        )?;
        gpu::encode_scalar_mul(
            context,
            pass,
            (&qwen4.hc_inject, inject_row_offset),
            1.0 / hc_count as f32,
            hc_count as u32,
        )
        .map_err(gpu_err)?;
        gpu::encode_sigmoid(
            context,
            pass,
            (&qwen4.hc_inject, inject_row_offset),
            hc_count as u32,
        )
        .map_err(gpu_err)?;
        gpu::encode_scalar_mul(
            context,
            pass,
            (&qwen4.hc_inject, inject_row_offset),
            2.0,
            hc_count as u32,
        )
        .map_err(gpu_err)?;
    }
    Ok(())
}
