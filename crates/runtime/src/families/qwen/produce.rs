//! Forward pass token decode implementations for Qwen models.

use std::time::Instant;

use foundation::LogitValue;

use super::attn::{encode_full_attention_block, encode_linear_block, QkNormConvention};
use super::dense;
use super::moe;
use super::{layer_tensor, RMS_EPS, TRUNK_PREFIX};
use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::{encode_embed_any, encode_gemv_any};
use crate::real_forward_types::RealForwardError;
use crate::real_forward_utils::{entry, norm_view};

/// Lift this layer's OUTPUT -- the residual stream after the FFN join --
/// into the steering capture, if one is open.
///
/// Zero dispatches when it is not, which is what keeps
/// `MFERENCE_RESID_CAPTURE` unset identical in bytes and in footprint to the
/// engine that shipped before this module existed.
///
/// It reuses `encode_dflash_copy_rows` rather than adding a kernel: that one
/// is a generic strided FP16 row copy and the drafter already lifts
/// `scratch.x` with it at this exact boundary. A COPY cannot change what it
/// copies, so generated text is byte-identical with the capture on -- the
/// same guarantee `ffn_hist` gets by redirecting a destination, reached the
/// other way round because nothing writes the residual stream to a spare
/// buffer for us to redirect.
fn encode_resid_capture(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    scratch: &crate::real_forward_types::DecodeScratch,
    capture: Option<&crate::resid_capture::ResidCapture>,
    layer: usize,
    hidden: usize,
) -> Result<(), RealForwardError> {
    let Some(c) = capture else {
        return Ok(());
    };
    gpu::encode_dflash_copy_rows(
        context,
        pass,
        (&scratch.x, 0),
        (&c.capture, c.layer_offset(layer)),
        1,
        hidden as u32,
        c.hidden() as u32,
    )
    .map_err(RealForwardError::Gpu)
}

/// Apply this layer's directional-steering edit, if one is configured for it.
///
/// Placed at the same boundary as the capture above -- the residual stream
/// after the FFN join, which is this layer's OUTPUT -- so the direction is
/// applied where it was extracted from. Steering at a different boundary than
/// the capture would be a different edit than the one measured.
///
/// Zero dispatches when steering is off, and zero for a layer the direction
/// set does not cover: the per-layer table holds `None` there rather than a
/// zero vector, so a set steering three of sixty-four layers costs three
/// dispatches per token and not sixty-four.
fn encode_steering(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    scratch: &crate::real_forward_types::DecodeScratch,
    steering: Option<&crate::steering::SteeringState>,
    layer: usize,
    hidden: usize,
) -> Result<(), RealForwardError> {
    let Some(s) = steering else {
        return Ok(());
    };
    let Some(l) = s.layer(layer) else {
        return Ok(());
    };
    gpu::encode_steer_direction(
        context,
        pass,
        (&scratch.x, 0),
        (&s.directions, l.offset),
        // One FP32 slot per layer, so the coefficients of a whole pass
        // survive to be read back together after the commit.
        (&s.coeff, (layer * 4) as u64),
        &gpu::SteerParams {
            d_len: hidden as u32,
            rows: 1,
            row_stride: hidden as u32,
            mode: s.mode,
            alpha: s.alpha,
            inv_norm: l.inv_norm,
            target: s.target,
            gate_threshold: s.gate_threshold,
        },
    )
    .map_err(RealForwardError::Gpu)
}

impl RealForwardRunner {
    pub(crate) fn produce_real_qwen(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let result = self.produce_real_qwen_inner(token, position, logits);
        self.phases.calls += 1;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn produce_real_qwen_inner(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let hidden = arch.hidden_size as usize;
        let inter = arch.intermediate_size as usize;
        let moe_inter = arch.moe_intermediate_size as u32;
        let vocab = arch.vocab_size as usize;
        let num_experts = arch.num_experts as usize;
        let top_k = arch.top_k_experts as usize;
        let use_silu = arch.hidden_activation.contains("silu");
        let gpu_err = RealForwardError::Gpu;

        if position != self.kv.position() {
            return Err(RealForwardError::Unsupported(format!(
                "non-sequential position {position}; KV cache is at {}",
                self.kv.position()
            )));
        }
        if (token as usize) >= vocab {
            return Err(RealForwardError::Unsupported(format!(
                "token id {token} outside vocab {vocab}"
            )));
        }

        let embed_name = "language_model.model.embed_tokens.weight";
        let base = self.index.header.index_size;

        let mut pass = self.context.begin_pass_labeled("cb1 (attn+router)");
        encode_embed_any(
            &mut self.context,
            &pass,
            &self.weights,
            &self.index,
            embed_name,
            (&self.scratch.x, 0),
            token as u32,
            hidden as u32,
            1.0,
        )?;

        let qwen = self.real_qwen.as_ref().expect("real Qwen state present");
        let qwen_rotary_dim = qwen.rotary_dim;
        let qwen_shape = qwen.shape;
        let _ = (qwen_rotary_dim, qwen_shape);

        let (
            context,
            weights,
            index,
            arch,
            scratch,
            kv,
            qwen,
            dflash,
            resid_capture,
            steering,
            streamers,
            slot_buffers,
            routed_blobs,
            moe_offsets,
            routed_layouts,
            router_hist,
            phases,
        ) = (
            &mut self.context,
            &self.weights,
            &self.index,
            &self.arch,
            &self.scratch,
            &mut self.kv,
            self.real_qwen.as_ref().expect("checked above"),
            self.real_dflash.as_ref(),
            self.resid_capture.as_ref(),
            self.steering.as_ref(),
            &mut self.streamers,
            &self.slot_buffers,
            &self.routed_blobs,
            &self.moe_offsets,
            &self.routed_layouts,
            &mut self.router_hist,
            &mut self.phases,
        );

        for layer in 0..arch.num_layers as usize {
            let input_norm = norm_view(
                weights,
                index,
                &layer_tensor(layer, "input_layernorm.weight"),
                hidden,
            )?;
            gpu::encode_rms_norm_bf16w(
                context,
                &pass,
                (&scratch.x, 0),
                input_norm,
                (&scratch.normed, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;

            if arch.layer_is_linear(layer) {
                encode_linear_block(context, &pass, weights, index, arch, qwen, scratch, layer)?;
            } else {
                encode_full_attention_block(
                    context,
                    &pass,
                    weights,
                    index,
                    arch,
                    qwen,
                    scratch,
                    kv,
                    TRUNK_PREFIX,
                    layer,
                    position,
                    // The TRUNK's q/k norms are plain. Its MTP head's, which
                    // carry the same names through the same call, are not.
                    QkNormConvention::Plain,
                )?;
            }

            // The RAW attention output joins the residual stream. Qwen has
            // no sandwich norms (`ffn_sandwich_norms: false` in
            // `qwen_gdn_moe_35b_a3b()`), so normalizing `scratch.o` before this
            // add is a Gemma habit, not a Qwen one -- and it applies
            // `post_attention_layernorm`, a tensor that belongs to the
            // stream below, to the attention output as well. Doing both
            // took the reference-answer perplexity from 6.25 to 255,409.
            gpu::encode_residual_add(
                context,
                &pass,
                (&scratch.x, 0),
                (&scratch.o, 0),
                hidden as u32,
            )
            .map_err(gpu_err)?;

            // ONE post-attention norm feeds the router, the shared expert,
            // and the routed experts. Gemma splits this three ways; Qwen
            // does not, and adding the split would change every number.
            let post_attn = norm_view(
                weights,
                index,
                &layer_tensor(layer, "post_attention_layernorm.weight"),
                hidden,
            )?;
            gpu::encode_rms_norm_bf16w(
                context,
                &pass,
                (&scratch.x, 0),
                post_attn,
                (&qwen.moe_x, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;

            // THE DENSE HALF DIVERGES HERE AND NOWHERE ELSE. Everything above
            // -- embedding, both norms, both attention blocks, the raw
            // residual -- is the same code for a `qwen3_5` as for a Qwen 3.6,
            // and so is the head below. A dense layer also needs no mid-layer
            // commit, because nothing in it is data-dependent on a host
            // readback the way the router's top-k is, so the pass stays open
            // across the whole token.
            if qwen.dense {
                dense::encode_qwen_layer_dense(
                    context,
                    &pass,
                    weights,
                    index,
                    scratch,
                    qwen,
                    &scratch.x,
                    TRUNK_PREFIX,
                    layer,
                    hidden,
                    inter,
                    use_silu,
                )?;
                // THE DFLASH2 AUX CAPTURE, at the one point where this
                // layer's OUTPUT exists: the residual stream after the FFN
                // join. `docs/DFLASH2.md` pins which boundary "layer 5"
                // means (the OUTPUT of layer 5), and the capture lands in
                // the fc input's own layout. Zero dispatches when no
                // drafter is open; five small strided copies per token when
                // one is.
                if let Some(d) = dflash {
                    if let Some(aux) = d.aux_slot(layer) {
                        gpu::encode_dflash_copy_rows(
                            context,
                            &pass,
                            (&scratch.x, 0),
                            (&d.capture, (aux * hidden) as u64 * 2),
                            1,
                            hidden as u32,
                            (d.shape.aux_count * hidden) as u32,
                        )
                        .map_err(gpu_err)?;
                    }
                }
                encode_steering(context, &pass, scratch, steering, layer, hidden)?;
                encode_resid_capture(context, &pass, scratch, resid_capture, layer, hidden)?;
                continue;
            }

            let router_name = layer_tensor(layer, "mlp.gate.weight");
            let router = entry(index, &router_name)?;
            if router.dtype != 5 || router.size_bytes as usize != num_experts * hidden {
                return Err(RealForwardError::Unsupported(format!(
                    "{router_name}: expected INT8 (dtype 5) {num_experts}x{hidden}, got dtype {} \
                     with {} packed bytes",
                    router.dtype, router.size_bytes
                )));
            }
            gpu::encode_router_gemv_gemma4(
                context,
                &pass,
                (
                    weights.buffer(),
                    weights.gpu_offset(router.file_offset - base),
                ),
                (
                    weights.buffer(),
                    weights.gpu_offset(router.scale_offset - base),
                ),
                (
                    weights.buffer(),
                    weights.gpu_offset(router.bias_offset - base),
                ),
                (&qwen.moe_x, 0),
                (&qwen.router_ones, 0),
                (&qwen.router_logits_f32, 0),
                num_experts as u32,
                hidden as u32,
            )
            .map_err(gpu_err)?;

            let t_wait = Instant::now();
            phases.cb1_gpu_nanos += (pass.commit().wait_with_gpu_time() * 1e9) as u64;
            phases.gpu_wait_nanos += t_wait.elapsed().as_nanos() as u64;

            pass = context.begin_pass_labeled("routed cb");
            moe::encode_qwen_layer_moe(
                context,
                &pass,
                weights,
                index,
                scratch,
                qwen,
                streamers,
                slot_buffers,
                routed_blobs.as_ref(),
                moe_offsets,
                routed_layouts,
                router_hist,
                phases,
                layer,
                hidden,
                inter,
                moe_inter,
                num_experts,
                top_k,
                use_silu,
            )?;
            // The MoE half's post-FFN residual add happens INSIDE
            // `encode_qwen_layer_moe`, so this layer's output exists only
            // once that call returns -- the same boundary the dense branch
            // captures at, reached by a different route.
            encode_steering(context, &pass, scratch, steering, layer, hidden)?;
            encode_resid_capture(context, &pass, scratch, resid_capture, layer, hidden)?;
        }

        // Final norm + head. No softcap: Qwen has none, and the head must
        // hand `selection::select` raw logits either way.
        if !self.skip_head {
            pass.relabel("final cb (head)");
            let final_norm = norm_view(weights, index, "language_model.model.norm.weight", hidden)?;
            gpu::encode_rms_norm_bf16w(
                context,
                &pass,
                (&scratch.x, 0),
                final_norm,
                (&scratch.normed, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
            let head_name = if arch.tie_word_embeddings {
                embed_name.to_string()
            } else {
                "language_model.lm_head.weight".to_string()
            };
            encode_gemv_any(
                context,
                &pass,
                weights,
                index,
                &head_name,
                vocab,
                hidden,
                (&scratch.normed, 0),
                (&scratch.logits, 0),
            )?;
        } else {
            pass.relabel("final cb (prefill, no head)");
        }
        let t_wait = Instant::now();
        phases.final_cb_gpu_nanos += (pass.commit_and_wait_with_gpu_time() * 1e9) as u64;
        phases.final_wait_nanos += t_wait.elapsed().as_nanos() as u64;
        self.kv.advance();
        // The capture's bookkeeping, after the pass that filled it: one row
        // at this position, ready for a prime or a context write.
        if let Some(d) = self.real_dflash.as_mut() {
            d.note_capture(position, 1);
        }
        // The command buffer has been waited on, so every layer's capture
        // region is final. `record_pass` keeps at most one snapshot per
        // generation and decides which by `skip_head`, so this is called on
        // every pass rather than guarded here -- see `resid_capture.rs` on
        // why the LAST PROMPT token is the pass worth keeping and why
        // "whatever ran last" is the wrong rule.
        let skip_head = self.skip_head;
        if let Some(capture) = self.resid_capture.as_mut() {
            capture.record_pass(position, skip_head);
        }

        if self.skip_head {
            return Ok(());
        }
        if vocab != logits.len() {
            return Err(RealForwardError::Unsupported(format!(
                "vocab mismatch: model has {}, caller expected {}",
                vocab,
                logits.len()
            )));
        }
        gpu::read_buffer_f16_into(&scratch.logits, 0, logits);
        Ok(())
    }
}
