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
            ((k_buf, k_off), num_kv, "k_norm.weight"),
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
    for (data, heads) in [((&scratch.q, 0u64), num_heads), ((k_buf, k_off), num_kv)] {
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

        let router_name = layer_tensor(layer, "mlp.gate.weight");
        let router = crate::real_forward_utils::entry(&self.index, &router_name)?;
        if router.dtype != 5 || router.size_bytes as usize != num_experts * hidden {
            return Err(RealForwardError::Unsupported(format!(
                "{router_name}: expected INT8 (dtype 5) {num_experts}x{hidden}, got \
                 dtype {} with {} packed bytes",
                router.dtype, router.size_bytes
            )));
        }
        let base = self.index.header.index_size;
        gpu::encode_router_gemv_gemma4(
            &mut self.context,
            pass,
            (
                self.weights.buffer(),
                self.weights.gpu_offset(router.file_offset - base),
            ),
            (
                self.weights.buffer(),
                self.weights.gpu_offset(router.scale_offset - base),
            ),
            (
                self.weights.buffer(),
                self.weights.gpu_offset(router.bias_offset - base),
            ),
            (&llama.moe_x, x_off),
            (&llama.router_ones, 0),
            (&llama.router_logits_f32, (t * num_experts * 4) as u64),
            num_experts as u32,
            hidden as u32,
        )
        .map_err(gpu_err)?;

        Ok(())
    }
}
