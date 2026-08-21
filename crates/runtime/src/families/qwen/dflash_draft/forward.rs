use foundation::TokenId;

use super::{ring_spans, DFLASH_THETA, RESIDUAL_RESCALE};
use crate::families::qwen::dflash::{DFLASH_MASK_TOKEN, DFLASH_RESIDUAL_EPS, DFLASH_WINDOW};
use crate::families::qwen::RMS_EPS;
use crate::real_forward::{RealForwardError, RealForwardRunner};
use crate::real_forward_dispatch::{encode_embed_any, encode_gemm_any};
use crate::real_forward_utils::norm_view;

impl RealForwardRunner {
    /// The block forward proper: embeddings, `layers` drafter layers, the
    /// final norm, and the two head GEMMs. Assumes the context KV is
    /// current up to `base`.
    pub(crate) fn dflash_forward_block(
        &mut self,
        anchor: TokenId,
        base: usize,
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let vocab = arch.vocab_size as usize;
        let use_silu = arch.hidden_activation.contains("silu");
        let gpu_err = RealForwardError::Gpu;
        let embed_name = "language_model.model.embed_tokens.weight";
        let head_name = if arch.tie_word_embeddings {
            embed_name.to_string()
        } else {
            "language_model.lm_head.weight".to_string()
        };

        let pass = self.context.begin_pass_labeled("dflash draft");
        let (context, weights, index, dflash) = (
            &mut self.context,
            &self.weights,
            &self.index,
            self.real_dflash.as_ref().expect("checked"),
        );
        let s = &dflash.shape;
        let hidden = s.hidden;
        let rows = dflash.block + 1;
        let q_dim = s.num_heads * s.head_dim;
        let kv_dim = s.num_kv_heads * s.head_dim;
        let row_bytes = hidden as u64 * 2;
        let base_kernel_elems = 2 * 2 * hidden;
        let kv_spans = ring_spans(dflash.kv.capacity(0), base, rows);

        // Row 0 is the bonus token; rows 1..block are masks. Both enter at
        // 1/DFLASH_RESIDUAL_SCALE, which is where the residual stream's
        // scaled representation starts; the final norm takes it back out.
        encode_embed_any(
            context,
            &pass,
            weights,
            index,
            embed_name,
            (&dflash.x, 0),
            anchor as u32,
            hidden as u32,
            RESIDUAL_RESCALE,
        )?;
        for r in 1..rows {
            encode_embed_any(
                context,
                &pass,
                weights,
                index,
                embed_name,
                (&dflash.x, r as u64 * row_bytes),
                DFLASH_MASK_TOKEN as u32,
                hidden as u32,
                RESIDUAL_RESCALE,
            )?;
        }

        for layer in 0..s.layers {
            let name = |sfx: &str| format!("dflash.layers.{layer}.{sfx}");
            let input_norm = norm_view(weights, index, &name("input_layernorm.weight"), hidden)?;
            let post_norm = norm_view(
                weights,
                index,
                &name("post_attention_layernorm.weight"),
                hidden,
            )?;
            let q_norm = norm_view(weights, index, &name("self_attn.q_norm.weight"), s.head_dim)?;
            let k_norm = norm_view(weights, index, &name("self_attn.k_norm.weight"), s.head_dim)?;
            let attn_base = norm_view(
                weights,
                index,
                &name("attention_conv.base_kernel"),
                base_kernel_elems,
            )?;
            let mlp_base = norm_view(
                weights,
                index,
                &name("mlp_conv.base_kernel"),
                base_kernel_elems,
            )?;
            for r in 0..rows {
                let row = r as u64 * row_bytes;
                gpu::encode_rms_norm_bf16w(
                    context,
                    &pass,
                    (&dflash.x, row),
                    input_norm,
                    (&dflash.normed, row),
                    hidden as u32,
                    DFLASH_RESIDUAL_EPS,
                )
                .map_err(gpu_err)?;
            }

            // attention_conv.prepare: coefficients from the normed input,
            // side-0 convolution of it, and the sublayer reads the
            // CONVOLVED stream.
            encode_gemm_any(
                context,
                &pass,
                weights,
                index,
                &name("attention_conv.kernel_projection.weight"),
                s.conv_rows,
                hidden,
                (&dflash.normed, 0),
                (&dflash.coeffs, 0),
                rows,
            )?;
            gpu::encode_dflash_grouped_conv(
                context,
                &pass,
                (&dflash.normed, 0),
                (&dflash.coeffs, 0),
                attn_base,
                (&dflash.conv_p, 0),
                rows as u32,
                hidden as u32,
                0,
                1.0,
            )
            .map_err(gpu_err)?;

            // Attention: UNGATED q (the trunk's [q; gate] packing is a
            // trunk property this drafter does not share), k/v straight
            // into the slots, per-head norms, full-head NeoX rope, per-row
            // split-KV decode over the ring.
            encode_gemm_any(
                context,
                &pass,
                weights,
                index,
                &name("self_attn.q_proj.weight"),
                q_dim,
                hidden,
                (&dflash.conv_p, 0),
                (&dflash.q, 0),
                rows,
            )?;
            // Split at the ring's wrap, exactly as the context write is:
            // `k_slot` addresses `position % capacity` and a batched
            // projection writes ADJACENT slots, so a block straddling the
            // boundary would run past the layer's buffer.
            for is_k in [true, false] {
                let suffix = if is_k {
                    "self_attn.k_proj.weight"
                } else {
                    "self_attn.v_proj.weight"
                };
                for &(row0, count) in kv_spans.iter().filter(|s| s.1 > 0) {
                    let (buf, off) = if is_k {
                        dflash.kv.k_slot(layer, base + row0)
                    } else {
                        dflash.kv.v_slot(layer, base + row0)
                    };
                    encode_gemm_any(
                        context,
                        &pass,
                        weights,
                        index,
                        &name(suffix),
                        kv_dim,
                        hidden,
                        (&dflash.conv_p, (row0 * hidden) as u64 * 2),
                        (buf, off as u64),
                        count,
                    )?;
                }
            }
            // The per-row norms and RoPE run in their OWN loop, ahead of any
            // attention, because the attention below is NON-CAUSAL: row 0
            // reads row 8's key, so every row's key must already be normed
            // and rotated when the first row's attention is encoded. Folded
            // into one loop (which is what a causal block can do) row `r`
            // would read raw projections for every row above it.
            for r in 0..rows {
                let position = base + r;
                let q_row = (r * q_dim) as u64 * 2;
                let (k_buf, k_off) = dflash.kv.k_slot(layer, position);
                let k_row = k_off as u64;
                gpu::encode_rms_norm_bf16w_perhead(
                    context,
                    &pass,
                    (&dflash.q, q_row),
                    q_norm,
                    (&dflash.q, q_row),
                    s.num_heads as u32,
                    s.head_dim as u32,
                    RMS_EPS,
                )
                .map_err(gpu_err)?;
                gpu::encode_rms_norm_bf16w_perhead(
                    context,
                    &pass,
                    (k_buf, k_row),
                    k_norm,
                    (k_buf, k_row),
                    s.num_kv_heads as u32,
                    s.head_dim as u32,
                    RMS_EPS,
                )
                .map_err(gpu_err)?;
                for (data, heads) in [
                    ((&dflash.q, q_row), s.num_heads as u32),
                    ((k_buf, k_row), s.num_kv_heads as u32),
                ] {
                    gpu::encode_rope_proportional_neox(
                        context,
                        &pass,
                        data,
                        position as u32,
                        heads,
                        s.head_dim as u32,
                        (s.head_dim / 2) as u32,
                        DFLASH_THETA,
                    )
                    .map_err(gpu_err)?;
                }
            }
            // ATTENTION INSIDE THE DRAFT BLOCK IS NON-CAUSAL, which is the
            // block-diffusion semantics and not an optimization: the mask
            // rows are denoised JOINTLY, so every row sees every other row
            // as well as the committed context. Both references say so
            // outright -- llama.cpp calls `llama_set_causal_attn(ctx_dft,
            // false)` with the comment "DFlash needs non-causal attention",
            // and vLLM builds the block's attention the same way.
            //
            // This engine has no mask argument to flip: causality here IS
            // the span each row is given, so a whole-block span is the whole
            // change. Every row runs at `seq_len = base + rows` rather than
            // its own `position + 1`.
            let seq_len = (base + rows) as u32;
            let kv_start = seq_len.saturating_sub(DFLASH_WINDOW as u32);
            let ring = dflash.kv.ring_capacity(layer) as u32;
            let active_ring = if ring > 0 && seq_len > ring { ring } else { 0 };
            // One buffer per layer, so the slot's POSITION picks an offset
            // this call does not take: the kernel addresses the whole cache
            // itself, from `kv_start` and `active_ring`.
            let k_buf = dflash.kv.k_slot(layer, base).0;
            let v_buf = dflash.kv.v_slot(layer, base).0;
            for r in 0..rows {
                let q_row = (r * q_dim) as u64 * 2;
                gpu::encode_attention_decode(
                    context,
                    &pass,
                    (&dflash.q, q_row),
                    k_buf,
                    v_buf,
                    &dflash.attn,
                    (&dflash.attn_out, q_row),
                    s.head_dim as u32,
                    s.num_heads as u32,
                    s.num_kv_heads as u32,
                    seq_len,
                    kv_start,
                    active_ring,
                    (s.head_dim as f32).sqrt().recip(),
                    None,
                )
                .map_err(gpu_err)?;
            }
            encode_gemm_any(
                context,
                &pass,
                weights,
                index,
                &name("self_attn.o_proj.weight"),
                hidden,
                q_dim,
                (&dflash.attn_out, 0),
                (&dflash.sub_out, 0),
                rows,
            )?;
            gpu::encode_dflash_grouped_conv(
                context,
                &pass,
                (&dflash.sub_out, 0),
                (&dflash.coeffs, 0),
                attn_base,
                (&dflash.conv_f, 0),
                rows as u32,
                hidden as u32,
                1,
                RESIDUAL_RESCALE,
            )
            .map_err(gpu_err)?;
            for r in 0..rows {
                let row = r as u64 * row_bytes;
                gpu::encode_residual_add(
                    context,
                    &pass,
                    (&dflash.x, row),
                    (&dflash.conv_f, row),
                    hidden as u32,
                )
                .map_err(gpu_err)?;
                gpu::encode_rms_norm_bf16w(
                    context,
                    &pass,
                    (&dflash.x, row),
                    post_norm,
                    (&dflash.normed, row),
                    hidden as u32,
                    DFLASH_RESIDUAL_EPS,
                )
                .map_err(gpu_err)?;
            }

            // mlp_conv.prepare, the dense FFN, mlp_conv.finish.
            encode_gemm_any(
                context,
                &pass,
                weights,
                index,
                &name("mlp_conv.kernel_projection.weight"),
                s.conv_rows,
                hidden,
                (&dflash.normed, 0),
                (&dflash.coeffs, 0),
                rows,
            )?;
            gpu::encode_dflash_grouped_conv(
                context,
                &pass,
                (&dflash.normed, 0),
                (&dflash.coeffs, 0),
                mlp_base,
                (&dflash.conv_p, 0),
                rows as u32,
                hidden as u32,
                0,
                1.0,
            )
            .map_err(gpu_err)?;
            for (suffix, out) in [
                ("mlp.gate_proj.weight", &dflash.ffn_gate),
                ("mlp.up_proj.weight", &dflash.ffn_up),
            ] {
                encode_gemm_any(
                    context,
                    &pass,
                    weights,
                    index,
                    &name(suffix),
                    s.inter,
                    hidden,
                    (&dflash.conv_p, 0),
                    (out, 0),
                    rows,
                )?;
            }
            let act = if use_silu {
                gpu::encode_silu_mul
            } else {
                gpu::encode_gelu_mul
            };
            for r in 0..rows {
                let row = (r * s.inter) as u64 * 2;
                act(
                    context,
                    &pass,
                    (&dflash.ffn_gate, row),
                    (&dflash.ffn_up, row),
                    (&dflash.ffn_act, row),
                    s.inter as u32,
                )
                .map_err(gpu_err)?;
            }
            encode_gemm_any(
                context,
                &pass,
                weights,
                index,
                &name("mlp.down_proj.weight"),
                hidden,
                s.inter,
                (&dflash.ffn_act, 0),
                (&dflash.sub_out, 0),
                rows,
            )?;
            gpu::encode_dflash_grouped_conv(
                context,
                &pass,
                (&dflash.sub_out, 0),
                (&dflash.coeffs, 0),
                mlp_base,
                (&dflash.conv_f, 0),
                rows as u32,
                hidden as u32,
                1,
                RESIDUAL_RESCALE,
            )
            .map_err(gpu_err)?;
            for r in 0..rows {
                let row = r as u64 * row_bytes;
                gpu::encode_residual_add(
                    context,
                    &pass,
                    (&dflash.x, row),
                    (&dflash.conv_f, row),
                    hidden as u32,
                )
                .map_err(gpu_err)?;
            }
        }

        // Final norm, then the TRUNK's lm_head (the drafter shares it) and
        // the selector's hidden projection.
        let final_norm = norm_view(weights, index, "dflash.norm.weight", hidden)?;
        for r in 0..rows {
            let row = r as u64 * row_bytes;
            gpu::encode_rms_norm_bf16w(
                context,
                &pass,
                (&dflash.x, row),
                final_norm,
                (&dflash.normed, row),
                hidden as u32,
                DFLASH_RESIDUAL_EPS,
            )
            .map_err(gpu_err)?;
        }
        encode_gemm_any(
            context,
            &pass,
            weights,
            index,
            &head_name,
            vocab,
            hidden,
            (&dflash.normed, 0),
            (&dflash.logits, 0),
            rows,
        )?;
        encode_gemm_any(
            context,
            &pass,
            weights,
            index,
            "dflash.candidate_selector.hidden_projection.weight",
            s.rank,
            hidden,
            (&dflash.normed, 0),
            (&dflash.hproj, 0),
            rows,
        )?;
        pass.commit_and_wait();
        Ok(())
    }
}
