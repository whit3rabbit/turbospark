//! `gpt-oss`'s attention block (ROADMAP M5): plain grouped-query attention
//! plus THREE of the four things that make this a fifth flow.
//!
//! Structurally it is `families/llama/attn.rs` -- no per-head q/k norms, no
//! output gate, no packed query/gate split, no `attention_k_eq_v` aliasing.
//! What it adds, in the order the layer applies them:
//!
//! 1. **A BIAS ON ALL FOUR PROJECTIONS.** No GEMV kernel in this port takes
//!    one, so each is a separate `bias_add_bf16_fp16` pass rather than a
//!    seventh argument on seven kernels and four families' dispatch sites.
//!    Order matters and is llama.cpp's: bias FIRST, then RoPE.
//! 2. **YaRN RoPE**, through `rope_neox_freqs` and a precomputed per-pair
//!    frequency table, because YaRN's ramp between two correction dimensions
//!    is not expressible as the scalar theta the three older rope kernels
//!    take.
//! 3. **ATTENTION SINKS**: one learned logit per QUERY head, added to the
//!    softmax DENOMINATOR and to nothing else, so it drains probability mass
//!    without contributing a value row.
//!
//! And the window ALTERNATES, which is nearly free -- the SWA ring and the
//! layer mask already exist for Gemma (AGENTS.md Gotcha 18) -- but is not
//! free of thought: EVEN layers slide here, and inverting that phase gives a
//! model wrong only past 128 tokens of context, which no short smoke reaches.

use model_io::{ArchConfig, ResidentIndex};

use crate::families::gptoss::{layer_tensor, RealGptOssState};
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
    state: &RealGptOssState,
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

    // K and V go STRAIGHT INTO the cache slot on an FP16 layer, which is what
    // keeps the decode path zero-copy; on a TurboQuant-quantized layer they
    // go into the FP16 staging row instead (`crate::kv_write`'s doc). Their
    // biases and RoPE are then applied in place, at that same target, so the
    // eventual cache row (staging or slot) holds the finished values.
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

    // THE BIASES, BEFORE ROPE. Reversing the two is a different function that
    // still produces finite, plausible text: RoPE is linear in its input, so
    // rotating a biased vector and biasing a rotated one differ by a rotation
    // of the bias, which is a small, position-dependent, entirely wrong term.
    for (suffix, elems, out) in [
        ("q_proj.bias", q_dim, (&scratch.q, 0u64)),
        ("k_proj.bias", kv_dim, (k_buf, k_off)),
        ("v_proj.bias", kv_dim, (v_buf, v_off)),
    ] {
        let bias = norm_view(weights, index, &name(suffix), elems)?;
        gpu::encode_bias_add(context, pass, out, bias, elems as u32).map_err(gpu_err)?;
    }

    // V IS NOT ROTATED. Only q and k carry position.
    for (data, heads) in [((&scratch.q, 0u64), num_heads), ((k_buf, k_off), num_kv)] {
        gpu::encode_rope_neox_freqs(
            context,
            pass,
            data,
            position as u32,
            heads,
            head_dim,
            state.rotated_pairs,
            (&state.rope_frequencies, 0),
            state.rope_mscale,
        )
        .map_err(gpu_err)?;
    }

    // THE WINDOW. Mask 1 is full attention and mask 0 slides; the ring is
    // sized `sliding_window + max_prefill_chunk_tokens` by `KvCacheManager`,
    // and `ring_capacity` is 0 on a full layer, which the kernel reads as
    // identity addressing.
    let seq_len = (position + 1) as u32;
    let is_full = arch
        .full_attention_layer_mask
        .get(layer)
        .copied()
        .unwrap_or(1)
        == 1;
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
    // real cache slot now that it is normed and RoPE'd -- a no-op on an
    // FP16 layer (`crate::kv_write`'s doc).
    encode_kv_commit(
        context, pass, kv, scratch, layer, position, head_dim, num_kv,
    )?;

    // ONE SINK PER QUERY HEAD, checked for length at open. It joins the
    // softmax denominator in the COMBINE pass, behind `FC_ATTN_HAS_SINKS`
    // -- a function constant rather than a uniform, because an unbound
    // buffer is undefined behaviour, and its byte is in
    // `attention_constants_key` so a sink dispatch cannot silently reuse the
    // sinkless pipeline (AGENTS.md Gotcha 18's trap, one file over).
    let sinks = norm_view(
        weights,
        index,
        &layer_tensor(layer, "self_attn.sinks.weight"),
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
    )?;
    let o_bias = norm_view(weights, index, &name("o_proj.bias"), hidden)?;
    gpu::encode_bias_add(context, pass, (&scratch.o, 0), o_bias, hidden as u32).map_err(gpu_err)
}

impl crate::real_forward::RealForwardRunner {
    /// Encodes the attention and router GEMV pass (`cb1`) for a single token `t`
    /// at `position` within a micro-batch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_gpt_oss_layer_attn_and_router(
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
        let input_norm = norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "input_layernorm.weight"),
            hidden,
        )?;
        let post_attn_norm = norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "post_attention_layernorm.weight"),
            hidden,
        )?;
        let state = self
            .real_gpt_oss
            .as_ref()
            .expect("real gpt-oss state present");

        gpu::encode_rms_norm_bf16w(
            &mut self.context,
            pass,
            (&self.scratch.x, x_off),
            input_norm,
            (&self.scratch.normed, 0),
            hidden as u32,
            state.rms_eps,
        )
        .map_err(gpu_err)?;

        encode_attention_block(
            &mut self.context,
            pass,
            &self.weights,
            &self.index,
            &self.arch,
            state,
            &self.scratch,
            &self.kv,
            layer,
            position,
        )?;

        // RAW residual add, matching the sequential flow.
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
        // this token's OWN row (`RealGptOssState::moe_x` is M-row).
        let state = self
            .real_gpt_oss
            .as_ref()
            .expect("real gpt-oss state present");
        gpu::encode_rms_norm_bf16w(
            &mut self.context,
            pass,
            (&self.scratch.x, x_off),
            post_attn_norm,
            (&state.moe_x, x_off),
            hidden as u32,
            state.rms_eps,
        )
        .map_err(gpu_err)?;

        // The router GEMV runs on the GPU; its BIAS is added on the
        // host in `moe.rs`'s per-token routed loop, between the
        // readback and the top-k.
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
            (&state.moe_x, x_off),
            (&state.router_ones, 0),
            (&state.router_logits_f32, (t * num_experts * 4) as u64),
            num_experts as u32,
            hidden as u32,
        )
        .map_err(gpu_err)?;

        Ok(())
    }
}
