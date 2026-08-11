//! The `llama` architecture's attention block: plain grouped-query attention
//! (ROADMAP Phase M2).
//!
//! Written out rather than shared with the Gemma or Qwen blocks because what
//! it does NOT do is most of it. No per-head q/k/v norms (Gemma norms three,
//! Qwen two, this none), no packed query/gate split, no output gate, no
//! sliding window, no `attention_k_eq_v` aliasing of the V buffer, one RoPE
//! base for every layer. `docs/NEW_MODEL.md` Phase 0 asks for the layer as ten
//! lines of pseudocode; this file is those ten lines.

use model_io::{ArchConfig, ResidentIndex};

use crate::families::llama::{layer_tensor, RealLlamaState};
use crate::real_forward_dispatch::encode_gemv_any;
use crate::real_forward_types::{DecodeScratch, RealForwardError};

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_attention_block(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    arch: &ArchConfig,
    llama: &RealLlamaState,
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
    // what makes the decode path zero-copy; there is no separate V buffer to
    // alias, unlike Gemma's `attention_k_eq_v` layers.
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

    // QK-NORM, and note the ORDER: after the projections, before RoPE. Both
    // reference implementations norm the raw head then rotate; rotating
    // first and norming after is a different function that still produces
    // finite, plausible-looking text. `qwen3moe` only; `llama` skips it.
    if llama.qk_norm {
        let head_norm = |suffix: &str| {
            crate::real_forward_utils::norm_view(weights, index, &name(suffix), head_dim as usize)
        };
        for (data, heads, suffix) in [
            ((&scratch.q, 0u64), num_heads, "q_norm.weight"),
            ((k_buf, k_off as u64), num_kv, "k_norm.weight"),
        ] {
            gpu::encode_rms_norm_bf16w_perhead(
                context,
                pass,
                data,
                head_norm(suffix)?,
                data,
                heads,
                head_dim,
                llama.rms_eps,
            )
            .map_err(gpu_err)?;
        }
    }

    // One theta for every layer: this architecture publishes `rope.freq_base`
    // and no `freq_base_swa`, so `arch_from_gguf` sets both fields from it.
    let theta = arch.full_rope_theta as f32;
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
            llama.rotated_pairs,
            theta,
        )
        .map_err(gpu_err)?;
    }

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
        (position + 1) as u32,
        // No sliding window and no ring: every layer is full attention, so
        // the KV layout is linear and the start is 0.
        0,
        0,
        arch.attention_scale as f32,
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
