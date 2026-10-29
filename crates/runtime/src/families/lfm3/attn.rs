//! `lfm3`'s attention block: separate q/k/v projections straight into the
//! cache, per-head q/k RMSNorm, PER-CLASS zero-tail RoPE, and spark's
//! headwise scalar output gate.
//!
//! The rope is `rope_proportional_neox` on both classes -- pair partner
//! `head_dim / 2` away and a head_dim divisor -- because HF's LFM3 rotary
//! zero-tails the inverse frequencies rather than shrinking the rotated
//! dimension. Full layers rotate `full_rotated_pairs` (32 of 64) at
//! `full_rope_theta`; sliding layers rotate every pair at `rope_theta`.
//! Spark's `rope_neox_subdim` here would pair the wrong elements and divide
//! by the wrong width: fluent, and a different model.

use model_io::{ArchConfig, ResidentIndex};

use crate::families::lfm3::layer_tensor;
use crate::families::lfm3::state::{RealLfm3State, RMS_EPS};
use crate::kv_write::{encode_attention_any, encode_kv_commit, kv_write_target, KvHalf};
use crate::real_forward_dispatch::encode_gemv_any;
use crate::real_forward_types::{DecodeScratch, RealForwardError};
use crate::real_forward_utils::norm_view;

/// Reads `scratch.normed` (the layer's `operator_norm` output) and writes
/// the block's output, pre-`post_operator_norm`, to `scratch.o`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_attention_block(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    arch: &ArchConfig,
    lfm3: &RealLfm3State,
    scratch: &DecodeScratch,
    kv: &gpu::KvCacheManager,
    layer: usize,
    position: usize,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let hidden = arch.hidden_size as usize;
    let num_heads = arch.num_heads as u32;
    let num_kv = arch.num_kv_heads as u32;
    let head_dim = arch.full_head_dim as u32;
    let q_dim = (num_heads * head_dim) as usize;
    let kv_dim = (num_kv * head_dim) as usize;
    let name = |suffix: &str| layer_tensor(layer, &format!("self_attn.{suffix}"));

    let (k_buf, k_off) = kv_write_target(kv, scratch, KvHalf::K, layer, position);
    let (v_buf, v_off) = kv_write_target(kv, scratch, KvHalf::V, layer, position);
    for (suffix, rows, out) in [
        ("q_proj.weight", q_dim, (&scratch.q, 0u64)),
        ("k_proj.weight", kv_dim, (k_buf, k_off)),
        ("v_proj.weight", kv_dim, (v_buf, v_off)),
        // The gate reads the same normed input the projections read, so it
        // is encoded here, before anything overwrites `scratch.normed`.
        ("gate_proj.weight", num_heads as usize, (&lfm3.attn_gate, 0)),
    ] {
        encode_gemv_any(
            context,
            pass,
            weights,
            index,
            &name(suffix),
            rows,
            hidden,
            (&scratch.normed, 0),
            out,
        )?;
    }

    // Per-head RMSNorm on q and k, BEFORE rope. V is not normed.
    for (data, heads, suffix) in [
        ((&scratch.q, 0u64), num_heads, "q_norm.weight"),
        ((k_buf, k_off), num_kv, "k_norm.weight"),
    ] {
        let w = norm_view(weights, index, &name(suffix), head_dim as usize)?;
        gpu::encode_rms_norm_bf16w_perhead(context, pass, data, w, data, heads, head_dim, RMS_EPS)
            .map_err(gpu_err)?;
    }

    let is_full = arch.layer_is_full(layer);
    let (pairs, theta) = rope_params_for(
        is_full,
        lfm3.full_rotated_pairs,
        head_dim,
        arch.full_rope_theta as f32,
        arch.rope_theta as f32,
    );
    for (data, heads) in [((&scratch.q, 0u64), num_heads), ((k_buf, k_off), num_kv)] {
        gpu::encode_rope_proportional_neox(
            context,
            pass,
            data,
            position as u32,
            heads,
            head_dim,
            pairs,
            theta,
        )
        .map_err(gpu_err)?;
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
        None,
    )?;

    // One sigmoid scalar per head, broadcast over head_dim, BEFORE o_proj.
    gpu::encode_sigmoid_head_gate_mul(
        context,
        pass,
        (&scratch.attn_out, 0),
        (&lfm3.attn_gate, 0),
        head_dim,
        q_dim as u32,
    )
    .map_err(gpu_err)?;

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

/// The per-class rope selection: `(rotated_pairs, theta)`. Named so a test
/// can pin it, for spark's reason: on untrained weights the frozen digest
/// cannot tell the two classes' parameters apart.
pub(crate) fn rope_params_for(
    is_full: bool,
    full_rotated_pairs: u32,
    head_dim: u32,
    full_theta: f32,
    sliding_theta: f32,
) -> (u32, f32) {
    if is_full {
        (full_rotated_pairs, full_theta)
    } else {
        (head_dim / 2, sliding_theta)
    }
}

#[cfg(test)]
mod tests {
    use super::rope_params_for;

    #[test]
    fn the_rope_selection_is_per_layer_class() {
        // The published checkpoint: 32 of 64 pairs at 1e6 on full layers,
        // all 64 at 1e4 on sliding ones.
        assert_eq!(
            rope_params_for(true, 32, 128, 1_000_000.0, 10_000.0),
            (32, 1_000_000.0)
        );
        assert_eq!(
            rope_params_for(false, 32, 128, 1_000_000.0, 10_000.0),
            (64, 10_000.0)
        );
    }
}
