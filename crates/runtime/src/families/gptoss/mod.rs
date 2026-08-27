//! The `gpt-oss` decode flow for [`RealForwardRunner`] (ROADMAP M5).
//!
//! A FIFTH FLOW rather than a sixth family on an existing one, and the reason
//! is that all four of its differences are INSIDE the layer. `qwen3moe` could
//! join `llama` because its answer to every structural question was "same
//! graph, two scalars different"; this one changes what a projection, a
//! rotation, a softmax and an expert each compute.
//!
//! One decoder layer, from llama.cpp's `src/models/openai-moe.cpp`, which is
//! what wrote these bytes:
//!
//! ```text
//! h      = rms_norm(x, input_layernorm)          // eps 1e-5
//! q,k,v  = h @ {q,k,v}_proj + bias               // 64 q over 8 kv, head_dim 64
//! q,k    = rope_yarn(q, k, pos)                  // base 150000, mscale 1.3465736
//! a      = attention(q, k, v, sinks) @ o_proj + bias
//!                                                // scale 0.125, window 128 on EVEN layers
//! x      = x + a                                 // RAW, no sandwich norm
//! m      = rms_norm(x, post_attention_layernorm)
//! r      = softmax_topk(m @ router + bias)[:4]   // 32 experts, top-4, on RAW logits
//! x      = x + sum(r_w * expert(m))              // MXFP4, clamped SwiGLU, per-expert bias
//! ```
//!
//! Then a final norm and an UNTIED head, with no softcap.
//!
//! WHERE EACH DIFFERENCE LIVES, since none of them is in this file:
//! the projection biases, YaRN and the sinks are in `attn.rs`; the router bias
//! is in `moe.rs` (it has to be, because it must land between the readback and
//! the top-k); and the clamped SwiGLU plus the per-expert biases are inside
//! the MXFP4 routed pair and reach the flow through `RoutedBlobLayout` without
//! being named here at all.
//!
//! Everything else is `families/llama/`'s graph: no per-head q/k norms, no
//! output gate, no shared expert, no sandwich norms, no logit softcap.

mod attn;
mod moe;
mod prefill;
mod state;

pub(crate) use state::RealGptOssState;

use std::time::Instant;

use foundation::LogitValue;

use crate::real_forward::{RealForwardError, RealForwardRunner};
use crate::real_forward_dispatch::{encode_embed_any, encode_gemv_any};
use crate::real_forward_utils::{entry, norm_view};
use crate::resid_capture::encode_resid_capture;
use crate::steering::encode_steering;

pub(crate) fn layer_tensor(layer: usize, suffix: &str) -> String {
    format!("language_model.model.layers.{layer}.{suffix}")
}

impl RealForwardRunner {
    pub(crate) fn produce_real_gpt_oss(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let result = self.produce_real_gpt_oss_inner(token, position, logits);
        self.phases.calls += 1;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn produce_real_gpt_oss_inner(
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
        // No `sqrt(hidden)` embedding scale: that is Gemma's, and this
        // architecture's manifest says so (`embeddingScaledBySqrtHidden`).
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

        let (
            context,
            weights,
            index,
            arch,
            scratch,
            kv,
            state,
            resid_capture,
            steering,
            streamers,
            slot_buffers,
            routed_blobs,
            routed_blobs_banks,
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
            self.real_gpt_oss
                .as_ref()
                .expect("real gpt-oss state present"),
            self.resid_capture.as_ref(),
            self.steering.as_ref(),
            &mut self.streamers,
            &self.slot_buffers,
            &self.routed_blobs,
            &self.routed_blobs_banks,
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
                state.rms_eps,
            )
            .map_err(gpu_err)?;

            attn::encode_attention_block(
                context, &pass, weights, index, arch, state, scratch, kv, layer, position,
            )?;

            // RAW residual add. Gemma normalizes the attention output before
            // adding it back (its sandwich norms); this architecture does not,
            // and getting that wrong is invisible in a greedy smoke.
            gpu::encode_residual_add(
                context,
                &pass,
                (&scratch.x, 0),
                (&scratch.o, 0),
                hidden as u32,
            )
            .map_err(gpu_err)?;

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
                (&state.moe_x, 0),
                hidden as u32,
                state.rms_eps,
            )
            .map_err(gpu_err)?;

            // The router GEMV runs on the GPU; its BIAS is added on the host
            // in `moe.rs`, between the readback and the top-k. See that file.
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
                (&state.moe_x, 0),
                (&state.router_ones, 0),
                (&state.router_logits_f32, 0),
                num_experts as u32,
                hidden as u32,
            )
            .map_err(gpu_err)?;

            let t_wait = Instant::now();
            phases.cb1_gpu_nanos += (pass.commit().wait_with_gpu_time() * 1e9) as u64;
            phases.gpu_wait_nanos += t_wait.elapsed().as_nanos() as u64;

            pass = context.begin_pass_labeled("routed cb");
            moe::encode_gpt_oss_layer_moe(
                context,
                &pass,
                scratch,
                state,
                streamers,
                slot_buffers,
                routed_blobs.as_ref(),
                routed_blobs_banks,
                moe_offsets,
                routed_layouts,
                router_hist,
                phases,
                layer,
                hidden,
                moe_inter,
                num_experts,
                top_k,
                &crate::moe_prefill_pipeline::RoutedSlot::sequential(),
            )?;
            // The layer's OUTPUT: `encode_gpt_oss_layer_moe` ends with the
            // raw residual add (no shared expert, so the routed sum is the
            // whole join) -- same boundary shape as `families/llama/`'s MoE
            // branch, reached the same way: after a mid-layer commit forced
            // by the router's host-side top-k.
            //
            // STEERING FIRST, THEN THE CAPTURE, matching every other flow.
            encode_steering(context, &pass, scratch, steering, layer, hidden, 1, 0)?;
            encode_resid_capture(context, &pass, scratch, resid_capture, layer, hidden, 0)?;
        }

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
                state.rms_eps,
            )
            .map_err(gpu_err)?;
            // UNTIED, always: `RealGptOssState::build` refuses an install
            // claiming otherwise, so there is no tie branch to get wrong.
            encode_gemv_any(
                context,
                &pass,
                weights,
                index,
                "language_model.lm_head.weight",
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
        // The command buffer has been waited on, so every layer's capture
        // region is final. `record_pass` decides whether THIS pass is the
        // one worth keeping (`crates/runtime` Gotcha 20 -- the encode half
        // above is not the whole hook).
        let skip_head = self.skip_head;
        if let Some(capture) = self.resid_capture.as_mut() {
            capture.record_pass(position, skip_head);
        }

        if self.skip_head {
            return Ok(());
        }
        // Raw logits, never probabilities: `selection::select` softmaxes
        // whatever it is handed (AGENTS.md Gotcha 16). No softcap here either.
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
