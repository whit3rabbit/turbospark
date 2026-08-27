//! The plain-GQA decode flow for [`RealForwardRunner`] (ROADMAP Phase M2,
//! completed by M4). BOTH HALVES of the `llama` architecture string run:
//! Mixtral 8x7B / 8x22B on routed experts, and Mistral / Llama 2 / 3.x on a
//! dense gated FFN. `RealLlamaState.dense` picks, off `num_experts`.
//!
//! **TWO FAMILIES RUN THROUGH THIS ONE FLOW**, `llama` (Mixtral) and
//! `qwen3moe` (Qwen3-30B-A3B), because the layer graph below is the same
//! graph for both. They differ in exactly two places, both carried by
//! [`RealLlamaState`] and both keyed on `ArchConfig.family`: Qwen3 norms q
//! and k per head before RoPE, and its RMS epsilon is 1e-6 against 1e-5.
//! A third copy of this file with two lines changed would be the more
//! likely source of a divergence bug than the shared one is.
//!
//! One decoder layer, which is the pseudocode `docs/NEW_MODEL.md` Phase 0
//! asks for before any of this is written:
//!
//! ```text
//! h      = rms_norm(x, input_layernorm)
//! q,k,v  = h @ {q,k,v}_proj                 // 32 q heads over 8 kv heads
//! q,k    = per_head_norm(q, k)               // qwen3moe ONLY, before rope
//! q,k    = rope(q, k, pos, base 1e6)        // full-head NeoX
//! a      = attention(q, k, v) @ o_proj      // no gate, no window, no norms
//! x      = x + a                            // RAW, no sandwich norm
//! m      = rms_norm(x, post_attention_layernorm)
//! r      = softmax_topk(m @ router)[:2]     // 8 experts, top-2
//! x      = x + sum(r_w * expert(m))         // no shared expert
//! ```
//!
//! The dense half replaces the last two lines with one gated FFN and nothing
//! else changes (`dense.rs`):
//!
//! ```text
//! x      = x + down_proj(silu(gate_proj @ m) * (up_proj @ m))
//! ```
//!
//! Then a final norm and an untied head, with no softcap.
//!
//! Every difference from the two existing flows is an ABSENCE, which is why
//! this file is the shortest of the three: Gemma's sandwich norms, per-head
//! q/k/v norms, router scales, shared expert and logit softcap are all gone,
//! and so are Qwen's linear layers, output gate and gated shared expert.

mod attn;
mod dense;
mod moe;
mod moe_prefill;
mod prefill;
mod state;

pub(crate) use state::RealLlamaState;

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
    pub(crate) fn produce_real_llama(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let result = self.produce_real_llama_inner(token, position, logits);
        self.phases.calls += 1;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn produce_real_llama_inner(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let hidden = arch.hidden_size as usize;
        let moe_inter = arch.moe_intermediate_size as u32;
        // The DENSE width, which is a different field. Mixtral publishes one
        // `feed_forward_length` and the walk copies it into both, so on the
        // MoE half these are equal and the distinction is invisible; a dense
        // checkpoint sets only this one.
        let dense_inter = arch.intermediate_size as usize;
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
            llama,
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
            self.real_llama.as_ref().expect("real llama state present"),
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
                llama.rms_eps,
            )
            .map_err(gpu_err)?;

            attn::encode_attention_block(
                context, &pass, weights, index, arch, llama, scratch, kv, layer, position,
            )?;

            // RAW residual add. Gemma normalizes the attention output before
            // adding it back (its sandwich norms) and Qwen normalizes it with
            // `post_attention_layernorm`; this architecture does neither, and
            // getting that wrong is invisible in a greedy smoke.
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
                (&llama.moe_x, 0),
                hidden as u32,
                llama.rms_eps,
            )
            .map_err(gpu_err)?;

            // THE DENSE HALF DIVERGES HERE AND NOWHERE ELSE. Everything
            // above -- embedding, both norms, attention, the raw residual --
            // is the same code for a Mistral as for a Mixtral, and so is the
            // head below. A dense layer also needs no mid-layer commit,
            // because nothing in it is data-dependent on a host readback.
            if llama.dense {
                dense::encode_llama_layer_dense(
                    context,
                    &pass,
                    weights,
                    index,
                    scratch,
                    llama,
                    layer,
                    hidden,
                    dense_inter,
                    use_silu,
                    0,
                )?;
                // The layer's OUTPUT: `encode_llama_layer_dense` ends with
                // the raw residual add, so the stream below this call is what
                // feeds the next block. That is the boundary llama.cpp's
                // `build_cvec` applies a control vector at, which is what
                // makes a vector written by this port and one written by
                // `repeng` the same edit here.
                //
                // STEERING FIRST, THEN THE CAPTURE, matching the qwen flow.
                // The two are unobservable in either order on the normal
                // configuration (a direction is extracted with steering OFF),
                // and stating one order in one place is what keeps the
                // question from having two answers.
                encode_steering(context, &pass, scratch, steering, layer, hidden, 1, 0)?;
                encode_resid_capture(context, &pass, scratch, resid_capture, layer, hidden, 0)?;
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
                (&llama.moe_x, 0),
                (&llama.router_ones, 0),
                (&llama.router_logits_f32, 0),
                num_experts as u32,
                hidden as u32,
            )
            .map_err(gpu_err)?;

            let t_wait = Instant::now();
            phases.cb1_gpu_nanos += (pass.commit().wait_with_gpu_time() * 1e9) as u64;
            phases.gpu_wait_nanos += t_wait.elapsed().as_nanos() as u64;

            pass = context.begin_pass_labeled("routed cb");
            moe::encode_llama_layer_moe(
                context,
                &pass,
                index,
                scratch,
                llama,
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
                use_silu,
                &crate::moe_prefill_pipeline::RoutedSlot::sequential(),
            )?;
            // Same boundary as the dense branch, reached by a different
            // route: this half's post-FFN residual add happens INSIDE
            // `encode_llama_layer_moe`, so the layer's output exists only
            // once that call returns. Note the pass here is the ROUTED
            // command buffer rather than cb1 -- the router's top-k forced a
            // commit above -- which is why the call sits after the encode
            // rather than beside the dense one.
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
                llama.rms_eps,
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
        // The command buffer has been waited on, so every layer's capture
        // region is final. **THE ENCODE HALF IS NOT THE WHOLE HOOK**: the
        // per-layer copies fill a buffer and this is what reads it back, and
        // wiring only the first half gave `[resid-capture] no non-prefill
        // pass ran; wrote nothing` -- loud, which is the good failure mode,
        // and still a family half-wired. `record_pass` keeps at most one
        // snapshot per generation and decides which by `skip_head`, so it is
        // called on every pass rather than guarded here.
        let skip_head = self.skip_head;
        if let Some(capture) = self.resid_capture.as_mut() {
            capture.record_pass(position, skip_head);
        }

        if self.skip_head {
            return Ok(());
        }
        // Raw logits, never probabilities: `selection::select` softmaxes
        // whatever it is handed (AGENTS.md Gotcha 16). There is no softcap to
        // apply here either.
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
