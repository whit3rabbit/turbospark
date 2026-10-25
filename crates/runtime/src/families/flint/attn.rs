//! Flint's attention block: grouped-query attention with the three things
//! that distinguish it from `families/llama/` and `families/gptoss/`, in the
//! order the layer applies them:
//!
//! 1. **Per-head QK RMSNorm**, plain weight (no `1 + w`), width `head_dim`,
//!    applied to q and k BEFORE RoPE. Learned norm weights make the reverse
//!    order a different function that still produces plausible text.
//! 2. **NoPE on the full-attention layers.** RoPE is skipped when the layer
//!    mask is 1, and only then. The dense lead layers are full-attention, so
//!    they skip it too. V is never rotated.
//! 3. **A learned attention sink** per QUERY head (`self_attn.softmax_offset`),
//!    added to the softmax denominator only.
//!
//! No projection bias, no YaRN, no output gate. The sliding window attends
//! `sliding_window` keys INCLUDING the current one (`kv_start = seq_len -
//! window`), which is llama.cpp's `LLAMA_SWA_TYPE_STANDARD` convention for
//! `n_swa = 513`: key `p0` is visible from query `p1` iff `p1 - p0 < n_swa`.

use model_io::{ArchConfig, ResidentIndex};

use crate::families::flint::{layer_tensor, RealFlintState};
use crate::kv_write::{encode_attention_any, encode_kv_commit, kv_write_target, KvHalf};
use crate::real_forward_dispatch::encode_gemv_any;
use crate::real_forward_types::{DecodeScratch, RealForwardError};
use crate::real_forward_utils::norm_view;

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_attention_block(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    arch: &ArchConfig,
    state: &RealFlintState,
    scratch: &DecodeScratch,
    kv: &gpu::KvCacheManager,
    layer: usize,
    position: usize,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let hidden = arch.hidden_size as usize;
    let num_heads = arch.num_heads as u32;
    let num_kv = arch.num_full_kv_heads as u32;
    let head_dim = arch.full_head_dim as u32;
    let q_dim = (num_heads * head_dim) as usize;
    let kv_dim = (num_kv * head_dim) as usize;
    let name = |suffix: &str| layer_tensor(layer, &format!("self_attn.{suffix}"));

    // K and V go STRAIGHT INTO the cache slot on an FP16 layer, or into the
    // staging row on a quantized one; the norm and RoPE below then run in
    // place at that same target (`crate::kv_write`'s doc).
    let (k_buf, k_off) = kv_write_target(kv, scratch, KvHalf::K, layer, position);
    let (v_buf, v_off) = kv_write_target(kv, scratch, KvHalf::V, layer, position);

    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &name("q_proj.weight"),
        q_dim,
        hidden,
        (&scratch.normed, 0),
        (&scratch.q, 0),
    )?;
    for (suffix, out) in [
        ("k_proj.weight", (k_buf, k_off)),
        ("v_proj.weight", (v_buf, v_off)),
    ] {
        encode_gemv_any(
            context,
            pass,
            weights,
            index,
            &name(suffix),
            kv_dim,
            hidden,
            (&scratch.normed, 0),
            out,
        )?;
    }

    // Per-head norm of q and k, before RoPE. V is not normed.
    for (data, heads, suffix) in [
        ((&scratch.q, 0u64), num_heads, "q_norm.weight"),
        ((k_buf, k_off), num_kv, "k_norm.weight"),
    ] {
        let w = norm_view(weights, index, &name(suffix), head_dim as usize)?;
        gpu::encode_rms_norm_bf16w_perhead(
            context,
            pass,
            data,
            w,
            data,
            heads,
            head_dim,
            state.rms_eps,
        )
        .map_err(gpu_err)?;
    }

    // Mask 1 is full attention (and NoPE here); mask 0 slides and rotates.
    let is_full = arch
        .full_attention_layer_mask
        .get(layer)
        .copied()
        .unwrap_or(1)
        == 1;
    if !is_full {
        let theta = arch.rope_theta as f32;
        for (data, heads) in [((&scratch.q, 0u64), num_heads), ((k_buf, k_off), num_kv)] {
            if arch.rope_neox_subdim {
                gpu::encode_rope_neox_subdim(
                    context,
                    pass,
                    data,
                    position as u32,
                    heads,
                    head_dim,
                    state.rotated_pairs * 2,
                    theta,
                )
                .map_err(gpu_err)?;
            } else {
                gpu::encode_rope_proportional_neox(
                    context,
                    pass,
                    data,
                    position as u32,
                    heads,
                    head_dim,
                    state.rotated_pairs,
                    theta,
                )
                .map_err(gpu_err)?;
            }
        }
    }

    let seq_len = (position + 1) as u32;
    let (kv_start, active_ring) = if is_full {
        (0, 0)
    } else {
        let ring = kv.ring_capacity(layer) as u32;
        (
            seq_len.saturating_sub(arch.sliding_window as u32),
            if ring > 0 && seq_len > ring { ring } else { 0 },
        )
    };

    encode_kv_commit(
        context, pass, kv, scratch, layer, position, head_dim, num_kv,
    )?;

    let sinks = norm_view(
        weights,
        index,
        &layer_tensor(layer, "self_attn.softmax_offset"),
        num_heads as usize,
    )?;
    encode_attention_any(
        context,
        pass,
        (&scratch.q, 0),
        kv,
        scratch,
        (&scratch.attn_out, 0),
        layer,
        position,
        head_dim,
        num_heads,
        num_kv,
        seq_len,
        kv_start,
        active_ring,
        arch.attention_scale as f32,
        Some(sinks),
    )?;

    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &name("o_proj.weight"),
        hidden,
        q_dim,
        (&scratch.attn_out, 0),
        (&scratch.o, 0),
    )
}
