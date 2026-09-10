//! The `llama` architecture's attention block: plain grouped-query attention
//! (ROADMAP Phase M2).
//!
//! Q/K normalization is family-selected: none for Llama, per-head for
//! Qwen3, whole-projection for MiniMax. RoPE follows normalization and can
//! leave an unrotated tail. All use full attention without output gates.

use model_io::{ArchConfig, ResidentIndex};

use crate::families::llama::{layer_tensor, RealLlamaState};
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

    // K and V are written STRAIGHT INTO the cache slot by the GEMV on an
    // FP16 layer (zero-copy decode; no separate V buffer to alias, unlike
    // Gemma's `attention_k_eq_v` layers), or into a staging row on a
    // TurboQuant-quantized one -- `kv_write_target` picks which, and
    // everything below (the projection, the optional qk-norm, RoPE) runs
    // identically either way. `encode_kv_commit` quantizes staging into the
    // cache afterward; it is a no-op on an FP16 layer.
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

    // Normalize before RoPE: learned norm weights make the reverse order
    // a different function, even when its outputs look plausible.
    for (data, heads, suffix) in [
        ((&scratch.q, 0u64), num_heads, "q_norm.weight"),
        ((k_buf, k_off), num_kv, "k_norm.weight"),
    ] {
        use super::state::QkNorm;
        match llama.qk_norm {
            QkNorm::None => {}
            QkNorm::PerHead => {
                let w = crate::real_forward_utils::norm_view(
                    weights,
                    index,
                    &name(suffix),
                    head_dim as usize,
                )?;
                gpu::encode_rms_norm_bf16w_perhead(
                    context,
                    pass,
                    data,
                    w,
                    data,
                    heads,
                    head_dim,
                    llama.rms_eps,
                )
                .map_err(gpu_err)?;
            }
            QkNorm::Projection => {
                let width = heads * head_dim;
                let w = crate::real_forward_utils::norm_view(
                    weights,
                    index,
                    &name(suffix),
                    width as usize,
                )?;
                gpu::encode_rms_norm_bf16w(context, pass, data, w, data, width, llama.rms_eps)
                    .map_err(gpu_err)?;
            }
        }
    }

    // One theta for every layer: this architecture publishes `rope.freq_base`
    // and no `freq_base_swa`, so `arch_from_gguf` sets both fields from it.
    let theta = arch.full_rope_theta as f32;
    for (data, heads) in [((&scratch.q, 0u64), num_heads), ((k_buf, k_off), num_kv)] {
        if arch.rope_neox_subdim {
            gpu::encode_rope_neox_subdim(
                context,
                pass,
                data,
                position as u32,
                heads,
                head_dim,
                llama.rotated_pairs * 2,
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
                llama.rotated_pairs,
                theta,
            )
            .map_err(gpu_err)?;
        }
    }

    // No-op on an FP16 layer; quantizes the (now normed and RoPE'd) staging
    // row into the cache on a TurboQuant-quantized one.
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
        (position + 1) as u32,
        // No sliding window and no ring: every layer is full attention, so
        // the KV layout is linear and the start is 0.
        0,
        0,
        arch.attention_scale as f32,
        None,
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

impl crate::real_forward::RealForwardRunner {
    /// Encodes the attention and router GEMV pass (`cb1`) for a single token `t`
    /// at `position` within a micro-batch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_llama_layer_attn_and_router(
        &mut self,
        pass: &gpu::PassEncoder,
        layer: usize,
        position: usize,
        t: usize,
        hidden: usize,
        num_experts: usize,
    ) -> Result<(), RealForwardError> {
        let gpu_err = RealForwardError::Gpu;
        let x_off = (t * hidden * 2) as u64;
        let input_norm = crate::real_forward_utils::norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "input_layernorm.weight"),
            hidden,
        )?;
        let post_attn_norm = crate::real_forward_utils::norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "post_attention_layernorm.weight"),
            hidden,
        )?;
        let llama = self.real_llama.as_ref().expect("real llama state present");

        gpu::encode_rms_norm_bf16w(
            &mut self.context,
            pass,
            (&self.scratch.x, x_off),
            input_norm,
            (&self.scratch.normed, 0),
            hidden as u32,
            llama.rms_eps,
        )
        .map_err(gpu_err)?;

        encode_attention_block(
            &mut self.context,
            pass,
            &self.weights,
            &self.index,
            &self.arch,
            llama,
            &self.scratch,
            &self.kv,
            layer,
            position,
        )?;

        // RAW residual add, matching the sequential flow: this
        // architecture normalizes neither the attention output nor
        // the FFN output on the way back into the stream.
        gpu::encode_residual_add(
            &mut self.context,
            pass,
            (&self.scratch.x, x_off),
            (&self.scratch.o, 0),
            hidden as u32,
        )
        .map_err(gpu_err)?;

        // The post-attention norm feeds the router AND the routed
        // experts, and both happen after `cb1` commits, so it needs
        // this token's OWN row (`RealLlamaState::moe_x` is M-row,
        // matching `RealGemmaState::routed_x`).
        let llama = self.real_llama.as_ref().expect("real llama state present");
        gpu::encode_rms_norm_bf16w(
            &mut self.context,
            pass,
            (&self.scratch.x, x_off),
            post_attn_norm,
            (&llama.moe_x, x_off),
            hidden as u32,
            llama.rms_eps,
        )
        .map_err(gpu_err)?;

        super::router::encode(
            &mut self.context,
            pass,
            &self.weights,
            &self.index,
            llama,
            layer,
            t,
            hidden,
            num_experts,
        )?;

        Ok(())
    }
}
