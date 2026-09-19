//! The plain-GQA decode flow for [`RealForwardRunner`] (ROADMAP Phase M2,
//! completed by M4). BOTH HALVES of the `llama` architecture string run:
//! Mixtral 8x7B / 8x22B on routed experts, and Mistral / Llama 2 / 3.x on a
//! dense gated FFN. `RealLlamaState.dense` picks, off `num_experts`.
//!
//! Llama/Mixtral, dense and MoE Qwen3, and MiniMax-M2 share this flow.
//! Family-selected state carries Q/K normalization, RMS epsilon, rotary
//! width, and router scoring. MiniMax normalizes whole projections and
//! selects experts using biased sigmoid scores with unbiased weights.
//!
//! One decoder layer (`docs/NEW_MODEL.md` Phase 0):
//!
//! ```text
//! h      = rms_norm(x, input_layernorm)
//! q,k,v  = h @ {q,k,v}_proj
//! q,k    = family_qk_norm(q, k)             // before RoPE
//! q,k    = rope(q, k, pos)                  // family rotary width/theta
//! a      = attention(q, k, v) @ o_proj
//! x      = x + a
//! m      = rms_norm(x, post_attention_layernorm)
//! r      = family_topk(m @ router)
//! x      = x + sum(r_w * expert(m))
//! ```
//!
//! Dense models replace routing and expert reduction with one gated FFN
//! (`dense.rs`). All use full attention, no shared expert or sandwich norm,
//! and a final norm followed by a raw-logit head.

mod attn;
mod dense;
mod moe;
mod moe_prefill;
mod prefill;
pub(crate) mod router;
mod state;

pub(crate) use state::RealLlamaState;

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

        // Decode positions continue PAST the prompt, so an image prompt's
        // rope angle is `position + rope_delta` (degenerate `(p, p, p)`),
        // never the raw cache index -- the same rule the qwen flow's
        // produce path runs, and the reason this seam exists on the llama
        // flow at all. KV slot and span stay the raw `position`.
        let rope_position = match self.prompt_vision.as_ref() {
            Some(pv) => {
                let (t, h, w) = pv.rope_position(position);
                crate::vision::RopePosition::Triple(t, h, w)
            }
            None => crate::vision::RopePosition::Sequential,
        };
        // This position's deepstack injection, resolved before the field
        // split below: `(buffer per declared merger, this position's merged
        // row)`. The chunked driver precomputes the same pair per token; a
        // sequentially-walked image prompt must get the same adds HERE, or
        // the two embed sites disagree (the equivalence
        // `tests/qwen3vl_vision.rs` holds both to).
        let (deepstack_buffers, deepstack_row) = match self.prompt_vision.as_ref() {
            Some(pv) => (pv.deepstack_buffers().to_vec(), pv.row_index_for(position)),
            None => (Vec::new(), None),
        };

        let mut pass = self.context.begin_pass_labeled("cb1 (attn+router)");
        // No `sqrt(hidden)` embedding scale: that is Gemma's, and this
        // architecture's manifest says so (`embeddingScaledBySqrtHidden`).
        //
        // An image-pad position of a sequentially-walked prompt blits the
        // tower's row INSTEAD of the lookup, the second site
        // `crate` Gotcha 27 documents for the qwen flow and the llama
        // flow's own twin. The host write lands before this pass's commit,
        // which is what makes it visible to every dispatch below. Text
        // positions -- including every decode step, whose positions are past
        // the last span -- take the table exactly as before.
        match self
            .prompt_vision
            .as_ref()
            .and_then(|pv| pv.row_for(position))
        {
            Some(row) => {
                gpu::write_buffer_bytes(&self.scratch.x, 0, row);
            }
            None => encode_embed_any(
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
            )?,
        }

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
            mapped,
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
            &self.mapped,
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
                context,
                &pass,
                weights,
                index,
                arch,
                llama,
                scratch,
                kv,
                layer,
                position,
                rope_position,
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
                // Merger `layer`'s rows raw-add at an image position after
                // the layer's output exists, the produce twin of the chunked
                // driver's post-layer loop -- same slot order, same buffer,
                // the reference's `h = layer(h)` then `_deepstack_process`.
                if layer < deepstack_buffers.len() {
                    if let Some(row) = deepstack_row {
                        gpu::encode_residual_add(
                            context,
                            &pass,
                            (&scratch.x, 0),
                            (&deepstack_buffers[layer], (row * hidden * 2) as u64),
                            hidden as u32,
                        )
                        .map_err(gpu_err)?;
                    }
                }
                continue;
            }

            router::encode(
                context,
                &pass,
                weights,
                index,
                llama,
                layer,
                0,
                hidden,
                num_experts,
            )?;

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
                mapped,
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
