//! The `flint` decode flow for [`RealForwardRunner`]. Facts and open items:
//! `docs/FLINT_PHASE0.md`.
//!
//! A SEVENTH-ish FLOW rather than a family on an existing one: sinks, NoPE by
//! layer mask, per-head QK norm, a sigmoid router, an ungated shared expert
//! and a dense lead are each INSIDE the layer, and no existing flow has more
//! than two of them. The skeleton is `families/deepseek2/` (dense lead, then
//! shared expert seeding phase 2); the attention is `families/gptoss/`'s
//! minus biases and YaRN, plus the norm and NoPE.
//!
//! One decoder layer `i` of 49, cross-checked against the open llama.cpp port
//! (ggml-org/llama.cpp#30928), not a vendor modeling file:
//!
//! ```text
//! h      = rms_norm(x, input_layernorm)                 // eps 1e-6, plain weight
//! q,k,v  = h @ {q,k,v}_proj                             // 32 q over 4 kv, head_dim 128
//! q,k    = rms_norm_per_head(q, q_norm), (k, k_norm)    // BEFORE rope
//! q,k    = rope_neox(q, k, pos, theta 10000)            // only when mask[i] == 0
//!                                                       // full layers (mask 1) are NoPE
//! a      = attention(q, k, v, sinks = softmax_offset)   // scale 128^-0.5
//!                                                       // window 513 on sliding layers
//! x      = x + a @ o_proj                               // RAW, no sandwich norm
//! m      = rms_norm(x, post_attention_layernorm)
//! i < 2  : x = x + down(silu(gate m) * up m)            // dense, width 6144
//! i >= 2 : r = top6 by logit of (m @ router), weights = sigmoid(logit) / sum(sigmoid over
//!          the 6)                                        // INT8 router, no bias
//!          x = x + shared(m) + sum(r_w * expert(m))     // shared ungated, width 1024
//! ```
//!
//! Then a final norm and an UNTIED INT4 head returning raw logits, no softcap.
//! Layers 0 and 1 are dense AND full-attention NoPE: the two properties are
//! independent and this file ties neither to the other.
//!
//! Not wired, and recorded rather than half-built: chunked and batched
//! prefill (the sequential `produce` path serves prefill), mapped expert
//! residency, steering and speculation.

mod attn;
mod moe;
mod state;

pub(crate) use state::RealFlintState;

use std::time::Instant;

use foundation::LogitValue;

use crate::real_forward::{RealForwardError, RealForwardRunner};
use crate::real_forward_dispatch::{encode_embed_any, encode_gemv_any};
use crate::real_forward_utils::norm_view;
use crate::resid_capture::encode_resid_capture;
use crate::steering::encode_steering;

pub(crate) fn layer_tensor(layer: usize, suffix: &str) -> String {
    format!("language_model.model.layers.{layer}.{suffix}")
}

impl RealForwardRunner {
    pub(crate) fn produce_real_flint(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let result = self.produce_real_flint_inner(token, position, logits);
        self.phases.calls += 1;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn produce_real_flint_inner(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let hidden = arch.hidden_size as usize;
        let moe_inter = arch.moe_intermediate_size as u32;
        let vocab = arch.vocab_size as usize;
        let num_experts = arch.num_experts as usize;
        let top_k = arch.top_k_experts as usize;
        let routed_scale = arch.routed_scaling_factor as f32;
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
        let mut pass = self.context.begin_pass_labeled("cb1 (attn+router)");
        // No `sqrt(hidden)` embedding scale: that is Gemma's.
        encode_embed_any(
            &mut self.context,
            &pass,
            &self.weights,
            &self.index,
            embed_name,
            (&self.scratch.x, 0),
            token as u32,
            hidden as u32,
            vocab,
            1.0,
        )?;

        let state = self.real_flint.as_ref().expect("flint state present");
        let rms_eps = state.rms_eps;
        let lead = state.lead;

        for layer in 0..arch.num_layers as usize {
            let input_norm = norm_view(
                &self.weights,
                &self.index,
                &layer_tensor(layer, "input_layernorm.weight"),
                hidden,
            )?;
            gpu::encode_rms_norm_bf16w(
                &mut self.context,
                &pass,
                (&self.scratch.x, 0),
                input_norm,
                (&self.scratch.normed, 0),
                hidden as u32,
                rms_eps,
            )
            .map_err(gpu_err)?;

            attn::encode_attention_block(
                &mut self.context,
                &pass,
                &self.weights,
                &self.index,
                &arch,
                state,
                &self.scratch,
                &self.kv,
                layer,
                position,
            )?;

            // RAW residual add; there is no sandwich norm.
            gpu::encode_residual_add(
                &mut self.context,
                &pass,
                (&self.scratch.x, 0),
                (&self.scratch.o, 0),
                hidden as u32,
            )
            .map_err(gpu_err)?;

            let post_attn = norm_view(
                &self.weights,
                &self.index,
                &layer_tensor(layer, "post_attention_layernorm.weight"),
                hidden,
            )?;
            gpu::encode_rms_norm_bf16w(
                &mut self.context,
                &pass,
                (&self.scratch.x, 0),
                post_attn,
                (&state.moe_x, 0),
                hidden as u32,
                rms_eps,
            )
            .map_err(gpu_err)?;

            // THE DENSE LEAD diverges here and nowhere else: a plain SwiGLU
            // at the lead width, no router, no commit.
            if layer < lead {
                moe::encode_swiglu(
                    &mut self.context,
                    &pass,
                    &self.weights,
                    &self.index,
                    state,
                    "mlp",
                    layer,
                    hidden,
                    state.dense_inter,
                    (&self.scratch.o, 0),
                )?;
                gpu::encode_residual_add(
                    &mut self.context,
                    &pass,
                    (&self.scratch.x, 0),
                    (&self.scratch.o, 0),
                    hidden as u32,
                )
                .map_err(gpu_err)?;
                encode_steering(
                    &mut self.context,
                    &pass,
                    &self.scratch,
                    self.steering.as_ref(),
                    layer,
                    hidden,
                    1,
                    0,
                )?;
                encode_resid_capture(
                    &mut self.context,
                    &pass,
                    &self.scratch,
                    self.resid_capture.as_ref(),
                    layer,
                    hidden,
                    0,
                )?;
                continue;
            }

            // Router: the INT8 kernel over the transcoded `mlp.gate.weight`
            // (checked at open), then one commit so the host can rank.
            {
                let router = crate::real_forward_utils::entry(
                    &self.index,
                    &layer_tensor(layer, "mlp.gate.weight"),
                )?;
                let base = self.index.header.index_size;
                gpu::encode_router_gemv_gemma4(
                    &mut self.context,
                    &pass,
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
                    (&state.moe_x, 0),
                    (&state.router_ones, 0),
                    (&state.router_logits_f32, 0),
                    num_experts as u32,
                    hidden as u32,
                )
                .map_err(gpu_err)?;
            }

            let t_wait = Instant::now();
            let committed = pass.commit();
            let gpu_t = committed.wait_with_gpu_time();
            self.phases.cb1_gpu_nanos += (gpu_t * 1e9) as u64;
            self.phases.gpu_wait_nanos += t_wait.elapsed().as_nanos() as u64;

            pass = self.context.begin_pass_labeled("routed cb");
            moe::encode_flint_layer_moe(
                &mut self.context,
                &pass,
                &self.weights,
                &self.index,
                &self.scratch,
                state,
                &mut self.streamers,
                &self.slot_buffers,
                &self.mapped,
                self.routed_blobs.as_ref(),
                &self.routed_blobs_banks,
                &self.moe_offsets,
                &self.routed_layouts,
                &mut self.router_hist,
                &mut self.phases,
                layer,
                hidden,
                moe_inter,
                num_experts,
                top_k,
                routed_scale,
                &crate::moe_prefill_pipeline::RoutedSlot::sequential(),
            )?;
            encode_steering(
                &mut self.context,
                &pass,
                &self.scratch,
                self.steering.as_ref(),
                layer,
                hidden,
                1,
                0,
            )?;
            encode_resid_capture(
                &mut self.context,
                &pass,
                &self.scratch,
                self.resid_capture.as_ref(),
                layer,
                hidden,
                0,
            )?;
        }

        if !self.skip_head {
            pass.relabel("final cb (head)");
            let final_norm = norm_view(
                &self.weights,
                &self.index,
                "language_model.model.norm.weight",
                hidden,
            )?;
            gpu::encode_rms_norm_bf16w(
                &mut self.context,
                &pass,
                (&self.scratch.x, 0),
                final_norm,
                (&self.scratch.normed, 0),
                hidden as u32,
                rms_eps,
            )
            .map_err(gpu_err)?;
            // UNTIED, always: the state refuses an install claiming otherwise.
            encode_gemv_any(
                &mut self.context,
                &pass,
                &self.weights,
                &self.index,
                "language_model.lm_head.weight",
                vocab,
                hidden,
                (&self.scratch.normed, 0),
                (&self.scratch.logits, 0),
            )?;
        } else {
            pass.relabel("final cb (prefill, no head)");
        }
        let t_wait = Instant::now();
        self.phases.final_cb_gpu_nanos += (pass.commit_and_wait_with_gpu_time() * 1e9) as u64;
        self.phases.final_wait_nanos += t_wait.elapsed().as_nanos() as u64;
        self.kv.advance();
        if let Some(capture) = self.resid_capture.as_mut() {
            capture.record_pass(position, self.skip_head);
        }

        if self.skip_head {
            return Ok(());
        }
        // Raw logits, never probabilities (AGENTS.md Gotcha 16); no softcap.
        if vocab != logits.len() {
            return Err(RealForwardError::Unsupported(format!(
                "vocab mismatch: model has {}, caller expected {}",
                vocab,
                logits.len()
            )));
        }
        gpu::read_buffer_f16_into(&self.scratch.logits, 0, logits);
        Ok(())
    }
}
