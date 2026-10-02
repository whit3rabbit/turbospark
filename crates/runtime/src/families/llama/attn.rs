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

/// Query rows prepared for the dense-Llama batch attention encoder.
///
/// K/V remain in their existing linear cache slots. The caller must commit
/// and wait for the pass that populated those slots before dispatching
/// attention.
#[allow(dead_code)] // The chunked-prefill caller consumes this after its batch dispatch is wired.
pub(crate) struct PreparedAttentionRows<'a> {
    pub(crate) query: &'a gpu::MetalBuffer,
    pub(crate) first_query_position: usize,
    pub(crate) row_count: usize,
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // Used by the dense chunked-prefill staging path.
pub(crate) fn encode_attention_inputs_batch<'a>(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    arch: &ArchConfig,
    llama: &'a RealLlamaState,
    scratch: &DecodeScratch,
    kv: &gpu::KvCacheManager,
    layer: usize,
    first_query_position: usize,
    rope_positions: &[crate::vision::RopePosition],
) -> Result<PreparedAttentionRows<'a>, RealForwardError> {
    let row_count = rope_positions.len();
    if row_count == 0 || row_count > crate::real_forward_types::MAX_PREFILL_BATCH {
        return Err(RealForwardError::Unsupported(format!(
            "dense Llama attention staging row count {row_count} is outside 1..={}",
            crate::real_forward_types::MAX_PREFILL_BATCH
        )));
    }
    let num_layers = usize::try_from(arch.num_layers)
        .map_err(|_| RealForwardError::Unsupported("invalid Llama layer count".into()))?;
    if arch.family != model_io::ModelFamily::Llama
        || arch.num_experts != 0
        || arch.full_attention_layer_mask.len() != num_layers
        || arch.full_attention_layer_mask.iter().any(|&mask| mask != 1)
        || layer >= num_layers
    {
        return Err(RealForwardError::Unsupported(
            "batch attention staging requires dense full-attention Llama".into(),
        ));
    }
    if (0..num_layers).any(|layer| kv.layer_quant(layer).is_some()) {
        return Err(RealForwardError::Unsupported(
            "dense Llama attention staging requires unquantized KV".into(),
        ));
    }
    let batch = llama.batch_attention_buffers.as_ref().ok_or_else(|| {
        RealForwardError::Unsupported(
            "batch attention buffers are unavailable for this Llama architecture".into(),
        )
    })?;

    let hidden = arch.hidden_size as usize;
    let q_dim = (arch.num_heads as usize)
        .checked_mul(arch.full_head_dim as usize)
        .ok_or_else(|| RealForwardError::Unsupported("query row size overflow".into()))?;
    let q_row_bytes = q_dim
        .checked_mul(2)
        .ok_or_else(|| RealForwardError::Unsupported("query row byte size overflow".into()))?;
    let required_q_bytes = q_row_bytes
        .checked_mul(row_count)
        .ok_or_else(|| RealForwardError::Unsupported("query batch size overflow".into()))?;
    if required_q_bytes > batch.q.length() as usize {
        return Err(RealForwardError::Unsupported(format!(
            "query batch needs {required_q_bytes} bytes but its buffer has {}",
            batch.q.length()
        )));
    }
    let required_input_bytes = hidden
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_mul(row_count))
        .ok_or_else(|| RealForwardError::Unsupported("input batch size overflow".into()))?;
    if required_input_bytes > scratch.x.length() as usize {
        return Err(RealForwardError::Unsupported(format!(
            "input batch needs {required_input_bytes} bytes but its buffer has {}",
            scratch.x.length()
        )));
    }
    let final_position = first_query_position
        .checked_add(row_count - 1)
        .ok_or_else(|| RealForwardError::Unsupported("query position range overflow".into()))?;
    u32::try_from(final_position).map_err(|_| {
        RealForwardError::Unsupported(format!(
            "query position {final_position} exceeds the GPU position range"
        ))
    })?;

    let input_norm = crate::real_forward_utils::norm_view(
        weights,
        index,
        &layer_tensor(layer, "input_layernorm.weight"),
        hidden,
    )?;
    for (row, &rope_position) in rope_positions.iter().enumerate() {
        let position = first_query_position + row;
        let input_offset = row * hidden * 2;
        gpu::encode_rms_norm_bf16w(
            context,
            pass,
            (&scratch.x, input_offset as u64),
            input_norm,
            (&scratch.normed, 0),
            hidden as u32,
            llama.rms_eps,
        )
        .map_err(RealForwardError::Gpu)?;

        let query_offset = row * q_row_bytes;
        encode_attention_input_row(
            context,
            pass,
            weights,
            index,
            arch,
            llama,
            scratch,
            kv,
            layer,
            position,
            rope_position,
            (&batch.q, query_offset as u64),
        )?;
    }

    Ok(PreparedAttentionRows {
        query: &batch.q,
        first_query_position,
        row_count,
    })
}

#[allow(clippy::too_many_arguments)]
fn encode_attention_input_row(
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
    rope_position: crate::vision::RopePosition,
    query: (&gpu::MetalBuffer, u64),
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
        query,
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

    // Qwen2 applies all three projection biases before RoPE. Applying them
    // after rotation would rotate the learned bias differently at every
    // position, producing finite but incorrect attention logits.
    if llama.qkv_bias {
        for (suffix, elems, out) in [
            ("q_proj.bias", q_dim, query),
            ("k_proj.bias", kv_dim, (k_buf, k_off)),
            ("v_proj.bias", kv_dim, (v_buf, v_off)),
        ] {
            let bias = crate::real_forward_utils::norm_view(weights, index, &name(suffix), elems)?;
            gpu::encode_bias_add(context, pass, out, bias, elems as u32).map_err(gpu_err)?;
        }
    }

    // Normalize before RoPE: learned norm weights make the reverse order
    // a different function, even when its outputs look plausible.
    for (data, heads, suffix) in [
        (query, num_heads, "q_norm.weight"),
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
    //
    // The mRoPE arm is `qwen3_vl`'s image positions and is the SAME
    // construction `families/qwen/attn.rs` runs for its tower, one flow over:
    // `position` keeps its KV-slot and attention-span jobs either way and
    // only the ANGLE moves, `t == h == w` still takes the pre-existing kernel
    // below, and the interleaved kernel shares `apply_neox_pair` with
    // `encode_rope_neox_subdim` -- at `rotated_pairs * 2 == head_dim`
    // (qwen3_vl's FULL rotary, `partial_rotary_factor` 1.0) its pair
    // structure is `rope_proportional_neox`'s exactly, which is what makes
    // the degenerate arm a bit-identity here too.
    let mrope = match rope_position {
        crate::vision::RopePosition::Sequential => None,
        crate::vision::RopePosition::Triple(t, h, w) if t == h && h == w => None,
        crate::vision::RopePosition::Triple(t, h, w) => Some((t as u32, h as u32, w as u32)),
    };
    let scalar = match rope_position {
        crate::vision::RopePosition::Sequential => position as u32,
        crate::vision::RopePosition::Triple(t, _, _) => t as u32,
    };
    let section = arch.vision.mrope_section;
    let theta = arch.full_rope_theta as f32;
    for (data, heads) in [(query, num_heads), ((k_buf, k_off), num_kv)] {
        if let Some(positions) = mrope {
            gpu::encode_rope_mrope_interleaved(
                context,
                pass,
                data,
                positions,
                heads,
                head_dim,
                llama.rotated_pairs * 2,
                (section[0] as u32, section[1] as u32, section[2] as u32),
                theta,
            )
            .map_err(gpu_err)?;
        } else if arch.rope_neox_subdim {
            gpu::encode_rope_neox_subdim(
                context,
                pass,
                data,
                scalar,
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
                scalar,
                heads,
                head_dim,
                llama.rotated_pairs,
                theta,
            )
            .map_err(gpu_err)?;
        }
    }

    Ok(())
}

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
    rope_position: crate::vision::RopePosition,
) -> Result<(), RealForwardError> {
    let hidden = arch.hidden_size as usize;
    let num_heads = arch.num_heads as u32;
    let num_kv = arch.num_full_kv_heads as u32;
    let head_dim = arch.full_head_dim as u32;
    let q_dim = (num_heads * head_dim) as usize;
    let name = |suffix: &str| layer_tensor(layer, &format!("self_attn.{suffix}"));

    encode_attention_input_row(
        context,
        pass,
        weights,
        index,
        arch,
        llama,
        scratch,
        kv,
        layer,
        position,
        rope_position,
        (&scratch.q, 0),
    )?;

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

#[cfg(all(test, target_os = "macos"))]
#[path = "attn_tests.rs"]
mod tests;

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

        // The MoE chunked driver serves no vision-capable family (the
        // qwen3_vl trunk is dense), so its angle is always the raw position.
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
            crate::vision::RopePosition::Sequential,
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
