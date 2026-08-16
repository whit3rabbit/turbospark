//! `muse_glimmer`'s attention block: GQA with a no-scale per-head q/k norm, a
//! second Q scale, a per-layer NoPE decision, and a separate output gate.
//!
//! Four things here are not in any other family's block, and all four are
//! FLUENT failures if dropped -- none produces an error:
//!
//! 1. The q/k norms are NO-SCALE, where Gemma's and `qwen3moe`'s are learned.
//! 2. Q takes a SECOND scale (`qk_scale_factor`) after that norm, on top of
//!    the `128^-0.5` the attention kernel applies.
//! 3. RoPE is skipped entirely on FULL-attention layers (NoPE).
//! 4. The output gate is its own projection off the layer's normed input.

use model_io::{ArchConfig, ResidentIndex};

use crate::families::museglimmer::layer_tensor;
use crate::families::museglimmer::state::{RealMuseState, QK_SCALE_FACTOR, RMS_EPS};
use crate::real_forward_dispatch::encode_gemv_any;
use crate::real_forward_types::{DecodeScratch, RealForwardError};

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_attention_block(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    arch: &ArchConfig,
    muse: &RealMuseState,
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

    // K and V are written STRAIGHT INTO the cache slot by the GEMV, which is
    // what keeps the decode path zero-copy.
    let (k_buf, k_off) = kv.k_slot(layer, position);
    let (v_buf, v_off) = kv.v_slot(layer, position);
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
        ("k_proj.weight", (k_buf, k_off as u64)),
        ("v_proj.weight", (v_buf, v_off as u64)),
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

    // THE OUTPUT GATE'S PROJECTION, encoded HERE rather than after
    // attention, because its input is `scratch.normed` -- the layer's normed
    // input -- and this is while that buffer provably still holds it. The
    // reference reads the same `x` the projections read, not the attention
    // output.
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &name("gate_proj.weight"),
        q_dim,
        hidden,
        (&scratch.normed, 0),
        (&muse.attn_gate, 0),
    )?;

    // NO-SCALE per-head norms on q and k, and NOT on v. The reference builds
    // ONE `RMSNormNoScale` and applies it to `queries` and `keys` only;
    // adding v "by analogy" is silent (`docs/NEW_MODEL.md` Phase 0 says so of
    // exactly this, in the other direction -- Gemma norms three).
    for (data, heads) in [
        ((&scratch.q, 0u64), num_heads),
        ((k_buf, k_off as u64), num_kv),
    ] {
        gpu::encode_rms_norm_no_scale_perhead(context, pass, data, data, heads, head_dim, RMS_EPS)
            .map_err(gpu_err)?;
    }

    // THE SECOND Q SCALE, on Q ALONE and after its norm. `qk_scale_factor`
    // is a separate published number from `attention_scale`, which the
    // attention kernel applies to the dot product; this one scales the
    // queries themselves. Applying it to k as well, or folding it into the
    // kernel's scale, both change the function.
    gpu::encode_scalar_mul(
        context,
        pass,
        (&scratch.q, 0),
        QK_SCALE_FACTOR,
        q_dim as u32,
    )
    .map_err(gpu_err)?;

    // **NoPE ON THE FULL-ATTENTION LAYERS.** `layer_rope_theta` is literally
    // 0 there in the checkpoint and the reference gates on
    // `bool(layer_rope_theta[i])`, so those thirteen layers rotate NOTHING.
    // `full_rope_theta` carries that zero and `RealMuseState::build` refuses
    // an install that does not.
    let is_full = arch
        .full_attention_layer_mask
        .get(layer)
        .copied()
        .unwrap_or(1)
        == 1;
    if !is_full {
        let theta = arch.rope_theta as f32;
        for (data, heads) in [
            ((&scratch.q, 0u64), num_heads),
            ((k_buf, k_off as u64), num_kv),
        ] {
            gpu::encode_rope_proportional_neox(
                context,
                pass,
                data,
                position as u32,
                heads,
                head_dim,
                muse.rotated_pairs,
                theta,
            )
            .map_err(gpu_err)?;
        }
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

    gpu::encode_attention_decode(
        context,
        pass,
        (&scratch.q, 0),
        k_buf,
        v_buf,
        &scratch.attn,
        (&scratch.attn_out, 0),
        head_dim,
        num_heads,
        num_kv,
        seq_len,
        kv_start,
        active_ring,
        arch.attention_scale as f32,
        None,
    )
    .map_err(gpu_err)?;

    // THE GATE, applied to the attention output BEFORE `o_proj`. The
    // reference is `output * sigmoid(gate)` then `o_proj(...)`; applying it
    // after `o_proj` gates the wrong width and the wrong vector.
    gpu::encode_sigmoid_gate_mul(
        context,
        pass,
        (&scratch.attn_out, 0),
        (&muse.attn_gate, 0),
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
