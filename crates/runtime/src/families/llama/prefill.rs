//! Chunked prefill for the DENSE half of the `llama` architecture (Mistral,
//! Llama 2/3.x): the second [`crate::producer::ChunkedPrefillRunner`]
//! implementation after Gemma 4's, and structurally simpler than it.
//!
//! Gemma 4's driver (`families/gemma4/prefill.rs`) commits one command
//! buffer PER LAYER because its routed half needs the router's top-k read
//! back on the host before the experts can be bound. A dense layer has no
//! such host round trip (`dense.rs`'s own doc: "the whole token stays in
//! one command buffer and the layer loop never syncs"), so a whole
//! micro-batch runs every layer of every token in ONE command buffer,
//! committed once at the end.
//!
//! Only `scratch.x` needs a row per token (the residual stream, read and
//! written across the whole micro-batch); every other scratch buffer
//! (`normed`, `q`, `attn_out`, `o`, `ffn_gate`, `ffn_up`, `ffn_act`,
//! `llama.h2`, `llama.moe_x`) is a transient GPU-only intermediate reused at
//! offset 0 per token, safe because dispatches within one command buffer
//! execute in commit order (`crates/gpu/CLAUDE.md` Gotcha 8) -- the same
//! reasoning Gemma 4's driver already established.

use std::time::Instant;

use foundation::LogitValue;

use super::{attn, dense, layer_tensor};
use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::{encode_embed_any, encode_gemv_any};
use crate::real_forward_types::{RealForwardError, MAX_PREFILL_BATCH};
use crate::real_forward_utils::norm_view;
use crate::resid_capture::encode_resid_capture;
use crate::steering::encode_steering;

impl RealForwardRunner {
    /// Runs a whole prefill chunk through the dense `llama` flow, writing
    /// the logits for the position after its last token. Call only once
    /// [`RealLlamaState::dense`] is known true; an MoE install is refused at
    /// [`crate::producer::ChunkedPrefillRunner::prefill_chunk`], by name,
    /// before this is reached.
    pub(crate) fn prefill_chunk_real_llama_dense(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        if tokens.is_empty() {
            return Err(RealForwardError::Unsupported(
                "prefill_chunk called with an empty chunk".to_string(),
            ));
        }
        let mut offset = 0usize;
        while offset < tokens.len() {
            let take = (tokens.len() - offset).min(MAX_PREFILL_BATCH);
            let last = offset + take == tokens.len();
            self.prefill_micro_batch_llama_dense(
                &tokens[offset..offset + take],
                start_position + offset,
                last,
                logits,
            )?;
            offset += take;
        }
        Ok(())
    }

    fn prefill_micro_batch_llama_dense(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        want_head: bool,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let result =
            self.prefill_micro_batch_llama_dense_inner(tokens, start_position, want_head, logits);
        // One forward pass per TOKEN, matching every other phase divisor in
        // this port (`crates/runtime/CLAUDE.md` Gotcha 6).
        self.phases.calls += tokens.len() as u64;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn prefill_micro_batch_llama_dense_inner(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        want_head: bool,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let hidden = arch.hidden_size as usize;
        let dense_inter = arch.intermediate_size as usize;
        let vocab = arch.vocab_size as usize;
        let use_silu = arch.hidden_activation.contains("silu");
        let gpu_err = RealForwardError::Gpu;
        let m = tokens.len();

        if start_position != self.kv.position() {
            return Err(RealForwardError::Unsupported(format!(
                "non-sequential chunk start {start_position}; KV cache is at {}",
                self.kv.position()
            )));
        }
        if let Some(&bad) = tokens.iter().find(|&&t| (t as usize) >= vocab) {
            return Err(RealForwardError::Unsupported(format!(
                "token id {bad} outside vocab {vocab}"
            )));
        }

        let embed_name = "language_model.model.embed_tokens.weight";

        let pass = self.context.begin_pass_labeled("llama dense chunk cb");
        // No `sqrt(hidden)` embedding scale, matching the sequential flow:
        // this architecture's manifest declares `embeddingScaledBySqrtHidden`
        // false and `RealLlamaState::build` refuses an install that says
        // otherwise.
        for (t, &token) in tokens.iter().enumerate() {
            encode_embed_any(
                &mut self.context,
                &pass,
                &self.weights,
                &self.index,
                embed_name,
                (&self.scratch.x, (t * hidden * 2) as u64),
                token as u32,
                hidden as u32,
                1.0,
            )?;
        }

        let (context, weights, index, scratch, kv, llama, resid_capture, steering) = (
            &mut self.context,
            &self.weights,
            &self.index,
            &self.scratch,
            &mut self.kv,
            self.real_llama.as_ref().expect("real llama state present"),
            self.resid_capture.as_ref(),
            self.steering.as_ref(),
        );

        for layer in 0..arch.num_layers as usize {
            let input_norm = norm_view(
                weights,
                index,
                &layer_tensor(layer, "input_layernorm.weight"),
                hidden,
            )?;
            let post_attn_norm = norm_view(
                weights,
                index,
                &layer_tensor(layer, "post_attention_layernorm.weight"),
                hidden,
            )?;

            for t in 0..m {
                let position = start_position + t;
                let x_off = (t * hidden * 2) as u64;

                gpu::encode_rms_norm_bf16w(
                    context,
                    &pass,
                    (&scratch.x, x_off),
                    input_norm,
                    (&scratch.normed, 0),
                    hidden as u32,
                    llama.rms_eps,
                )
                .map_err(gpu_err)?;

                attn::encode_attention_block(
                    context, &pass, weights, index, &arch, llama, scratch, kv, layer, position,
                )?;

                // RAW residual add at this token's OWN row: this
                // architecture normalizes neither the attention output nor
                // the FFN output on the way back into the stream (Gotcha 11
                // in `crates/runtime/CLAUDE.md`).
                gpu::encode_residual_add(
                    context,
                    &pass,
                    (&scratch.x, x_off),
                    (&scratch.o, 0),
                    hidden as u32,
                )
                .map_err(gpu_err)?;

                gpu::encode_rms_norm_bf16w(
                    context,
                    &pass,
                    (&scratch.x, x_off),
                    post_attn_norm,
                    (&llama.moe_x, 0),
                    hidden as u32,
                    llama.rms_eps,
                )
                .map_err(gpu_err)?;

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
                    x_off,
                )?;

                // Steering first, then the capture, matching the sequential
                // flow's order (both are unobservable in either order on
                // the normal configuration: a direction is extracted with
                // steering off).
                encode_steering(context, &pass, scratch, steering, layer, hidden, 1, x_off)?;
                encode_resid_capture(context, &pass, scratch, resid_capture, layer, hidden, x_off)?;
            }
        }

        if want_head {
            pass.relabel("llama dense final cb (head)");
            let last_off = ((m - 1) * hidden * 2) as u64;
            let final_norm = norm_view(weights, index, "language_model.model.norm.weight", hidden)?;
            gpu::encode_rms_norm_bf16w(
                context,
                &pass,
                (&scratch.x, last_off),
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
            // No softcap: `RealLlamaState::build` refuses an install that
            // declares one, matching the sequential flow.
        } else {
            pass.relabel("llama dense chunk cb (no head)");
        }
        self.phases.final_cb_gpu_nanos += (pass.commit_and_wait_with_gpu_time() * 1e9) as u64;
        self.kv.advance_by(m);

        let skip_head = !want_head;
        let last_position = start_position + m - 1;
        if let Some(capture) = self.resid_capture.as_mut() {
            capture.record_pass(last_position, skip_head);
        }

        if !want_head {
            return Ok(());
        }
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
