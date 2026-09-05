//! The PLE (per-layer n-gram embedding) sublayer, `mod.rs`'s "## PLE"
//! pseudocode, layer index [`RealQwen4State::ple_layer`] only.
//!
//! The 16 row lookups are HOST-side (dequant on the CPU, per the plan: 16
//! rows a token is not worth a GPU kernel until a phase table says so),
//! then one small upload starts the GPU half.

use half::f16;
use model_io::ResidentIndex;

use crate::families::qwen4::state::RealQwen4State;
use crate::families::qwen4::{layer_tensor, RMS_EPS};
use crate::real_forward_dispatch::encode_gemv_any;
use crate::real_forward_types::RealForwardError;
use crate::real_forward_utils::norm_view;

/// A `GdnShape` whose ONLY purpose is making `encode_gdn_conv_decode`'s
/// `channels`/`taps` derivation (`shape.qkv_dim()` / `shape.conv_kernel_size`)
/// equal PLE's own (`wide_dim`, 4): that dispatch was built for the GDN
/// chain and has no plain-channels-and-taps entry point, and its k/v head
/// split is otherwise meaningless here -- the conv reads only the two
/// derived numbers, never the split itself. `key_head_dim` is pinned at the
/// smallest `GdnShape::validate` accepts (32); `value_head_dim` is whatever
/// makes the two terms sum to `wide_dim`.
fn ple_conv_shape(wide_dim: usize) -> Result<gpu::GdnShape, RealForwardError> {
    const KEY_HEAD_DIM: usize = 32;
    let two_k = 2 * KEY_HEAD_DIM;
    if wide_dim <= two_k || (wide_dim - two_k) % 4 != 0 {
        return Err(RealForwardError::Unsupported(format!(
            "qwen4_exp's wide residual ({wide_dim}) does not fit the dilated conv's shape \
             adapter; it must exceed {two_k} by a multiple of 4"
        )));
    }
    let shape = gpu::GdnShape {
        num_k_heads: 1,
        num_v_heads: 1,
        key_head_dim: KEY_HEAD_DIM as u32,
        value_head_dim: (wide_dim - two_k) as u32,
        conv_kernel_size: 4,
    };
    shape.validate().map_err(RealForwardError::Gpu)?;
    Ok(shape)
}

/// Encodes one PLE layer step: the host-side n-gram hash and table lookup,
/// then the GPU half (`key`/`value`/`query`/`ple_gate`/dilated conv),
/// returning the `hc_count * hidden`-wide value the caller adds directly
/// into the residual (`docs/QWEN4_PHASE0.md` item 4: `hidden = hidden +
/// ple(hidden, input_ids)`, a plain add and NOT a hyper-connection
/// injection -- this sublayer has no `attn_hc`/`mlp_hc` of its own).
///
/// `emb_row_offset` is the byte offset into `qwen4.ngram_emb` this token's
/// dequantized n-gram embedding lands at and is read back from: `0` on the
/// sequential decode path (single row), `t * hidden * 2` inside a
/// chunked-prefill micro-batch. **This is not an optimization, it is a
/// correctness requirement**: `gpu::write_buffer_bytes` below is a HOST
/// write, executed the instant this function runs, not a GPU dispatch
/// queued for later -- so it does not respect command-buffer commit order
/// the way every other per-token buffer in this file does. A micro-batch
/// that calls this once per token into one uncommitted pass encodes EVERY
/// token's `key_proj`/`value_proj` GEMV before ANY of them execute, so a
/// single-row `ngram_emb` would hold only the LAST token's embedding by the
/// time the GPU actually runs the pass -- every earlier token's PLE output
/// would be computed from the wrong token's n-gram embedding, silently
/// (`prefill.rs`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_ple_layer(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    qwen4: &mut RealQwen4State,
    wide: (&gpu::MetalBuffer, u64),
    hidden: usize,
    heads_per_ngram: usize,
    token: i32,
    emb_row_offset: u64,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let hc_count = qwen4.hc_count;
    let wide_dim = hidden * hc_count;
    let layer = qwen4.ple_layer;
    let name = |suffix: &str| layer_tensor(layer, &format!("ple.{suffix}"));

    // --- HOST: hash, 16 row lookups, dequant, concatenate ---
    let ctx = qwen4.ngram_context.step(token as i64, qwen4.eos_token_id);
    let rows = model_io::ple_ngram_rows(
        &ctx,
        heads_per_ngram as i64,
        &qwen4.ngram_layout.multipliers,
        &qwen4.ngram_layout.head_vocab_sizes,
        &qwen4.ngram_layout.head_offsets,
    );
    let head_dim = qwen4.ngram_layout.head_dim as usize;
    let mut emb = Vec::with_capacity(hidden);
    let table = qwen4.ngram_table.data();
    for &gid in &rows {
        let offset = qwen4.ngram_layout.row_offset(gid).ok_or_else(|| {
            RealForwardError::Unsupported(format!(
                "PLE row id {gid} is outside the n-gram table's {} rows",
                qwen4.ngram_layout.rows
            ))
        })? as usize;
        let record_bytes = qwen4.ngram_layout.record_bytes as usize;
        let record = &table[offset..offset + record_bytes];
        let weight_bytes = qwen4.ngram_layout.weight_bytes as usize;
        let scale_bytes = qwen4.ngram_layout.scale_bytes as usize;
        let (packed, rest) = record.split_at(weight_bytes);
        let (scales, biases) = rest.split_at(scale_bytes);
        let dequanted = compute::dequant_ngram_row(
            packed,
            scales,
            biases,
            head_dim,
            qwen4.ngram_layout.group_size as usize,
        );
        emb.extend_from_slice(&dequanted);
    }
    if emb.len() != hidden {
        return Err(RealForwardError::Unsupported(format!(
            "PLE's 16-head concat produced {} values, expected hidden ({hidden})",
            emb.len()
        )));
    }
    let emb_f16: Vec<u8> = emb
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    gpu::write_buffer_bytes(&qwen4.ngram_emb, emb_row_offset as usize, &emb_f16);

    // --- GPU: key = norm_key(key_proj(emb)).view(C, H) ---
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &name("key_proj.weight"),
        wide_dim,
        hidden,
        (&qwen4.ngram_emb, emb_row_offset),
        (&qwen4.ple_key, 0),
    )?;
    let norm_key_w = norm_view(weights, index, &name("norm_key.weight"), wide_dim)?;
    gpu::encode_rms_norm_bf16w_grouped_centered(
        context,
        pass,
        (&qwen4.ple_key, 0),
        norm_key_w,
        (&qwen4.ple_key, 0),
        hc_count as u32,
        hidden as u32,
        RMS_EPS,
    )
    .map_err(gpu_err)?;

    // value = value_proj(emb), NO norm -- shared across every stream.
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &name("value_proj.weight"),
        hidden,
        hidden,
        (&qwen4.ngram_emb, emb_row_offset),
        (&qwen4.ple_value, 0),
    )?;

    // query = norm_query(hidden).view(C, H) -- reads the WIDE residual
    // directly; there is no query projection (the checkpoint's own tensor
    // list has key_proj and value_proj only).
    let norm_query_w = norm_view(weights, index, &name("norm_query.weight"), wide_dim)?;
    gpu::encode_rms_norm_bf16w_grouped_centered(
        context,
        pass,
        wide,
        norm_query_w,
        (&qwen4.ple_query, 0),
        hc_count as u32,
        hidden as u32,
        RMS_EPS,
    )
    .map_err(gpu_err)?;

    // gv = sigmoid(signed_sqrt((key . query) / sqrt(H))) * value
    gpu::encode_ple_gate(
        context,
        pass,
        (&qwen4.ple_key, 0),
        (&qwen4.ple_query, 0),
        (&qwen4.ple_value, 0),
        (&qwen4.ple_gv, 0),
        hc_count as u32,
        hidden as u32,
    )
    .map_err(gpu_err)?;

    // out = gv + silu(dilated_conv(norm_conv(gv))). The conv reads the
    // NORMED gated value; its output joins the UN-normed `ple_gv`
    // (`docs/QWEN4_PHASE0.md` item 4's own emphasis).
    let norm_conv_w = norm_view(weights, index, &name("norm_conv.weight"), wide_dim)?;
    gpu::encode_rms_norm_bf16w_grouped_centered(
        context,
        pass,
        (&qwen4.ple_gv, 0),
        norm_conv_w,
        (&qwen4.ple_conv_normed, 0),
        hc_count as u32,
        hidden as u32,
        RMS_EPS,
    )
    .map_err(gpu_err)?;

    let conv_shape = ple_conv_shape(wide_dim)?;
    let conv_w = norm_view(weights, index, &name("conv1d.weight"), wide_dim * 4)?;
    gpu::encode_gdn_conv_decode(
        context,
        pass,
        conv_shape,
        (&qwen4.ple_conv_tail, 0),
        (&qwen4.ple_conv_normed, 0),
        conv_w,
        (&qwen4.ple_conv_out, 0),
        // `dilation = ngram_size`, never the port's usual 1.
        // `ngram_context_len` is `ngram_size - 1` BY CONSTRUCTION
        // (`RealQwen4State::build`), so `+ 1` recovers `ngram_size`
        // directly -- unlike a derivation from `PLE_CONV_HISTORY`, which
        // is `(taps - 1) * dilation` and only coincides with `ngram_size`
        // through an unrelated factor (the exact trap
        // `RealQwen4State::ngram_context_len`'s own doc comment records
        // catching once already).
        (qwen4.ngram_context_len + 1) as u32,
    )
    .map_err(gpu_err)?;
    gpu::encode_silu(context, pass, (&qwen4.ple_conv_out, 0), wide_dim as u32).map_err(gpu_err)?;

    // hidden += gv + conv_out (both wide), a PLAIN add per the pseudocode
    // above -- the caller's residual (`wide`) is written in place.
    gpu::encode_residual_add(context, pass, wide, (&qwen4.ple_gv, 0), wide_dim as u32)
        .map_err(gpu_err)?;
    gpu::encode_residual_add(
        context,
        pass,
        wide,
        (&qwen4.ple_conv_out, 0),
        wide_dim as u32,
    )
    .map_err(gpu_err)?;
    Ok(())
}
