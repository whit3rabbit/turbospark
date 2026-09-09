//! `spark2_5`'s attention block: fused QKV split straight into the cache, a
//! PER-CLASS rope decision, and the headwise scalar output gate.
//!
//! Three things here are not in any other family's block, and all three are
//! FLUENT failures if dropped:
//!
//! 1. The QKV projection is ONE fused tensor (`q_k_v_proj`, q then k then v
//!    along the output dim). It is read as one GEMV and split by
//!    `encode_split_qkv`, with k and v landing DIRECTLY in their cache
//!    slots -- the zero-copy property the per-projection families get from
//!    three GEMVs, kept here with one copy kernel instead.
//! 2. RoPE diverges PER LAYER CLASS in BOTH theta and width: full layers
//!    rotate their leading `full_rotary_dim` dims (64 of 256) at theta 5e6,
//!    sliding layers rotate the whole head at theta 1e4. The width divisor
//!    is the ROTARY dim (`rope_neox_subdim`'s convention, which is HF's
//!    partial-rotary semantics), not the head dim.
//! 3. The output gate is HEADWISE: `g_proj` emits one scalar per head,
//!    sigmoid, broadcast over head_dim (`encode_sigmoid_head_gate_mul`),
//!    applied to the attention output BEFORE `o_proj`.

use model_io::{ArchConfig, ResidentIndex};

use crate::families::spark::layer_tensor;
use crate::families::spark::state::RealSparkState;
use crate::kv_write::{encode_attention_any, encode_kv_commit, kv_write_target, KvHalf};
use crate::real_forward_dispatch::encode_gemv_any;
use crate::real_forward_types::{DecodeScratch, RealForwardError};

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_attention_block(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    arch: &ArchConfig,
    spark: &RealSparkState,
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

    // K and V land STRAIGHT IN their cache slots via the split below, on an
    // FP16 layer; on a TurboQuant-quantized layer `kv_write_target` hands
    // back the FP16 staging row and `encode_kv_commit` quantizes it in place
    // (`crate::kv_write`'s doc).
    let (k_buf, k_off) = kv_write_target(kv, scratch, KvHalf::K, layer, position);
    let (v_buf, v_off) = kv_write_target(kv, scratch, KvHalf::V, layer, position);

    // THE FUSED PROJECTION, as one GEMV: rows are q (4096) then k (1024)
    // then v (1024). Reading three neighbours' per-projection names here
    // would be an `entry()` failure at token 1, not a wrong number.
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &name("q_k_v_proj.weight"),
        q_dim + 2 * kv_dim,
        hidden,
        (&scratch.normed, 0),
        (&spark.qkv, 0),
    )?;

    // THE GATE'S PROJECTION, encoded while `scratch.normed` provably still
    // holds the layer's normed input: the reference reads the same `x` the
    // QKV projection reads, not the attention output.
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &name("g_proj.weight"),
        num_heads as usize,
        hidden,
        (&scratch.normed, 0),
        (&spark.attn_gate, 0),
    )?;

    // THE SPLIT: q into its scratch, k and v into the slots the attention
    // kernel and the next token will read them from.
    gpu::encode_split_qkv(
        context,
        pass,
        (&spark.qkv, 0),
        (&scratch.q, 0),
        (k_buf, k_off),
        (v_buf, v_off),
        q_dim as u32,
        kv_dim as u32,
    )
    .map_err(gpu_err)?;

    // **ROPE, PER LAYER CLASS.** Full layers: theta `full_rope_theta` over
    // the leading `full_rotary_dim` dims (the head's first quarter on the
    // published checkpoint). Sliding layers: theta `rope_theta` over the
    // whole head, which `rope_neox_subdim` expresses exactly (rotary_dim ==
    // head_dim). Using one class's theta on the other is fluent and wrong.
    let is_full = arch
        .full_attention_layer_mask
        .get(layer)
        .copied()
        .unwrap_or(1)
        == 1;
    let (rotary_dim, theta) = rope_params_for(
        is_full,
        spark.full_rotary_dim,
        head_dim,
        arch.full_rope_theta as f32,
        arch.rope_theta as f32,
    );
    // `rms_norm_eps` is unused in this block; the q/k path has NO norms at
    // all (no QK-norm in this family), so q and k go straight into rope.
    for (data, heads) in [((&scratch.q, 0u64), num_heads), ((k_buf, k_off), num_kv)] {
        gpu::encode_rope_neox_subdim(
            context,
            pass,
            data,
            position as u32,
            heads,
            head_dim,
            rotary_dim,
            theta,
        )
        .map_err(gpu_err)?;
    }

    // The window is on the SLIDING layers; the full ones read the whole
    // history. `ring_capacity` is 0 on a full layer, which the kernel reads
    // as identity addressing.
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

    // On a TurboQuant-quantized layer, quantize the staging row into the
    // real cache slot now that it is RoPE'd -- a no-op on an FP16 layer.
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

    // THE HEADWISE GATE, applied to the attention output BEFORE `o_proj`:
    // one sigmoid scalar per head, broadcast over head_dim. Applying it
    // after `o_proj` gates the wrong width and the wrong vector.
    gpu::encode_sigmoid_head_gate_mul(
        context,
        pass,
        (&scratch.attn_out, 0),
        (&spark.attn_gate, 0),
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

/// The per-class rope selection, as a NAMED function because it is the one
/// piece of this family's logic a fixture cannot test end to end: on
/// untrained weights the frozen digest is insensitive to swapping the two
/// classes' (width, theta) assignments (the residual stream's common-mode
/// term absorbs what the swap changes, below an FP16 ulp of the logits),
/// while skipping rope entirely DOES move it. So the digest proves rope
/// runs; only this function's own test proves it selects PER CLASS. Swapping
/// the two arms reddens `the_rope_selection_is_per_layer_class` in
/// milliseconds; on the real install it is the quality gate's question.
pub(crate) fn rope_params_for(
    is_full: bool,
    full_rotary_dim: u32,
    head_dim: u32,
    full_theta: f32,
    sliding_theta: f32,
) -> (u32, f32) {
    if is_full {
        (full_rotary_dim, full_theta)
    } else {
        // The sliding arm rotates the WHOLE head (rotary_dim == head_dim)
        // at the sliding theta -- the gemma4-shaped convention the baseline
        // documents, with `rope_neox_subdim`'s divisor degenerating to the
        // head itself.
        (head_dim, sliding_theta)
    }
}

#[cfg(test)]
mod tests {
    use super::rope_params_for;

    #[test]
    fn the_rope_selection_is_per_layer_class() {
        // The published checkpoint's own values.
        let (dim, theta) = rope_params_for(true, 64, 256, 5_000_000.0, 10_000.0);
        assert_eq!(dim, 64);
        assert_eq!(theta, 5_000_000.0);
        let (dim, theta) = rope_params_for(false, 64, 256, 5_000_000.0, 10_000.0);
        assert_eq!(dim, 256);
        assert_eq!(theta, 10_000.0);
    }
}
