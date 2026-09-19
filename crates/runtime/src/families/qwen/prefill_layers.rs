//! Layer encoders for Qwen dense chunked prefill: batched GEMV and per-token loops.

use model_io::{ArchConfig, ResidentIndex};

use super::attn::{
    encode_full_attention_block, encode_linear_block, QkNormConvention, RopePosition,
};
use super::batched_layers::{
    encode_dense_ffn_batched, encode_full_attention_block_batched, encode_linear_block_batched,
};
use super::{dense, RealQwenState, RMS_EPS, TRUNK_PREFIX};
use crate::real_forward_types::{DecodeScratch, RealForwardError};
use crate::resid_capture::encode_resid_capture;
use crate::steering::encode_steering;

/// Encodes one dense layer of a chunked prefill micro-batch using the batched GEMV path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_dense_layer_batched_prefill(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    scratch: &DecodeScratch,
    kv: &mut gpu::KvCacheManager,
    qwen: &RealQwenState,
    arch: &ArchConfig,
    resid_capture: Option<&crate::resid_capture::ResidCapture>,
    steering: Option<&crate::steering::SteeringState>,
    input_norm: (&gpu::MetalBuffer, u64),
    post_attn_norm: (&gpu::MetalBuffer, u64),
    layer: usize,
    hidden: usize,
    inter: usize,
    use_silu: bool,
    start_position: usize,
    m: usize,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let is_linear = arch.layer_is_linear(layer);
    let batched = qwen.batched_prefill();
    for t in 0..m {
        gpu::encode_rms_norm_bf16w(
            context,
            pass,
            (&scratch.x, (t * hidden * 2) as u64),
            input_norm,
            (&batched.normed, (t * hidden * 2) as u64),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
    }

    if is_linear {
        // The recurrence still runs in token order -- it is inside
        // `gdn_conv_prefill` / `gdn_delta_prefill` rather than in this loop.
        // No tape slot: this scratch records none (a prefill chunk is never
        // rolled back), and `None` is what keeps the copy from being asked
        // for.
        encode_linear_block_batched(
            context, pass, weights, index, arch, qwen, batched, layer, m, None,
        )?;
    } else {
        encode_full_attention_block_batched(
            context,
            pass,
            weights,
            index,
            arch,
            qwen,
            scratch,
            batched,
            kv,
            layer,
            start_position,
            m,
        )?;
    }

    for t in 0..m {
        let x_off = (t * hidden * 2) as u64;
        // RAW residual add, as in the per-token arm: this family has no sandwich norms.
        gpu::encode_residual_add(
            context,
            pass,
            (&scratch.x, x_off),
            (&batched.o, x_off),
            hidden as u32,
        )
        .map_err(gpu_err)?;
        gpu::encode_rms_norm_bf16w(
            context,
            pass,
            (&scratch.x, x_off),
            post_attn_norm,
            (&batched.moe_x, x_off),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
    }

    // Encodes its own per-row residual add back into `scratch.x`.
    encode_dense_ffn_batched(
        context, pass, weights, index, scratch, batched, layer, hidden, inter, use_silu, m,
    )?;

    // ONE M-row steering dispatch rather than m.
    encode_steering(context, pass, scratch, steering, layer, hidden, m, 0)?;
    for t in 0..m {
        encode_resid_capture(
            context,
            pass,
            scratch,
            resid_capture,
            layer,
            hidden,
            (t * hidden * 2) as u64,
        )?;
    }
    Ok(())
}

/// Encodes one dense layer of a chunked prefill micro-batch using the per-token unbatched loop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_dense_layer_per_token_prefill(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    scratch: &DecodeScratch,
    kv: &mut gpu::KvCacheManager,
    qwen: &RealQwenState,
    arch: &ArchConfig,
    resid_capture: Option<&crate::resid_capture::ResidCapture>,
    steering: Option<&crate::steering::SteeringState>,
    input_norm: (&gpu::MetalBuffer, u64),
    post_attn_norm: (&gpu::MetalBuffer, u64),
    layer: usize,
    hidden: usize,
    inter: usize,
    use_silu: bool,
    start_position: usize,
    m: usize,
    rope: &[RopePosition],
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let is_linear = arch.layer_is_linear(layer);
    // The loop below is driven BY `rope`, so a short slice would silently
    // encode fewer tokens than the micro-batch holds -- leaving the tail rows
    // of `scratch.x` carrying whatever the previous micro-batch left there,
    // which is finite and fluent and wrong. Checked here rather than left to
    // the caller, who builds it at length `m` one function up.
    //
    // No test reddens on deleting this: every caller is in-tree and correct,
    // so it is a backstop rather than a covered branch. Said plainly so a
    // future reader does not go looking for the guard's test.
    if rope.len() != m {
        return Err(RealForwardError::Unsupported(format!(
            "rope position table has {} entries for a micro-batch of {m}",
            rope.len()
        )));
    }
    for (t, &rope_position) in rope.iter().enumerate() {
        let position = start_position + t;
        let x_off = (t * hidden * 2) as u64;

        gpu::encode_rms_norm_bf16w(
            context,
            pass,
            (&scratch.x, x_off),
            input_norm,
            (&scratch.normed, 0),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
        // The folded twin of `produce.rs`'s norm-site transforms, on the same
        // single-row plan buffers the per-token arm norms into.
        if let Some(h) = qwen.hadamard.as_ref() {
            h.transform(
                context,
                pass,
                (&scratch.normed, 0),
                (&h.normed_h, 0),
                1,
                hidden as u32,
                true,
            )?;
        }

        if is_linear {
            // Mask-2: gated DeltaNet.
            encode_linear_block(context, pass, weights, index, arch, qwen, scratch, layer)?;
        } else {
            encode_full_attention_block(
                context,
                pass,
                weights,
                index,
                arch,
                qwen,
                scratch,
                kv,
                TRUNK_PREFIX,
                layer,
                position,
                QkNormConvention::Plain,
                rope_position,
            )?;
        }

        // RAW residual add, matching the sequential flow: this family has no sandwich norms.
        gpu::encode_residual_add(
            context,
            pass,
            (&scratch.x, x_off),
            (&scratch.o, 0),
            hidden as u32,
        )
        .map_err(gpu_err)?;

        gpu::encode_rms_norm_bf16w(
            context,
            pass,
            (&scratch.x, x_off),
            post_attn_norm,
            (&qwen.moe_x, 0),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
        if let Some(h) = qwen.hadamard.as_ref() {
            h.transform(
                context,
                pass,
                (&qwen.moe_x, 0),
                (&h.moe_x_h, 0),
                1,
                hidden as u32,
                true,
            )?;
        }

        dense::encode_qwen_layer_dense(
            context,
            pass,
            weights,
            index,
            scratch,
            qwen,
            &scratch.x,
            x_off,
            TRUNK_PREFIX,
            layer,
            hidden,
            inter,
            use_silu,
        )?;

        encode_steering(context, pass, scratch, steering, layer, hidden, 1, x_off)?;
        encode_resid_capture(context, pass, scratch, resid_capture, layer, hidden, x_off)?;
    }
    Ok(())
}

/// Encodes final rms norm and lm_head GEMV for Qwen dense chunked prefill.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_dense_chunk_head(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    scratch: &DecodeScratch,
    qwen: &RealQwenState,
    arch: &ArchConfig,
    embed_name: &str,
    hidden: usize,
    vocab: usize,
    m: usize,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    pass.relabel("qwen dense final cb (head)");
    let last_off = ((m - 1) * hidden * 2) as u64;
    let final_norm = crate::real_forward_utils::norm_view(
        weights,
        index,
        "language_model.model.norm.weight",
        hidden,
    )?;
    gpu::encode_rms_norm_bf16w(
        context,
        pass,
        (&scratch.x, last_off),
        final_norm,
        (&scratch.normed, 0),
        hidden as u32,
        RMS_EPS,
    )
    .map_err(gpu_err)?;
    let head_name = if arch.tie_word_embeddings {
        embed_name.to_string()
    } else {
        "language_model.lm_head.weight".to_string()
    };
    // The folded-head twin of `produce.rs`'s head site: one forward
    // transform between the final norm and the GEMV.
    let head_input = match qwen.hadamard.as_ref() {
        Some(h) if h.is_folded(&head_name) => {
            h.transform(
                context,
                pass,
                (&scratch.normed, 0),
                (&h.normed_h, 0),
                1,
                hidden as u32,
                true,
            )?;
            (&h.normed_h, 0)
        }
        _ => (&scratch.normed, 0),
    };
    crate::real_forward_dispatch::encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &head_name,
        vocab,
        hidden,
        head_input,
        (&scratch.logits, 0),
    )
}
