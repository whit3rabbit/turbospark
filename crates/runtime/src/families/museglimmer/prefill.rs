//! Chunked prefill for `muse_glimmer`: the third
//! [`crate::producer::ChunkedPrefillRunner`] implementation, and structurally
//! the same shape as the dense half of `llama` (`families/llama/prefill.rs`).
//!
//! This family has no router at all (`families/museglimmer/mod.rs`'s own
//! doc: "no experts, no router, no shared expert"), so there is no host
//! round trip to pipeline around: a whole micro-batch runs every layer of
//! every token in ONE command buffer, committed once at the end, exactly as
//! the dense `llama` driver does.
//!
//! Only `scratch.x` needs a row per token (the residual stream, read and
//! written across the whole micro-batch); every other scratch buffer
//! (`normed`, `q`, `attn_out`, `o`, `o_normed`, `ffn_gate`, `ffn_up`,
//! `ffn_act`, `ffn_normed`, `muse.attn_gate`) is a transient GPU-only
//! intermediate reused at offset 0 per token, safe because dispatches within
//! one command buffer execute in commit order (`crates/gpu/CLAUDE.md`
//! Gotcha 8). `attn::encode_attention_block` needed NO change for the same
//! reason `families/llama/attn.rs`'s did not: it never touches `scratch.x`
//! directly, reading `scratch.normed` and writing `scratch.o` around it.
//! `mlp::encode_mlp_block` DOES touch `scratch.x` (its pre-FFN norm reads it
//! and its residual add writes it), so it gained the same `x_off: u64`
//! parameter `families/llama/dense.rs`'s `encode_llama_layer_dense` did,
//! with every pre-existing call site passing `0` explicitly.
//!
//! The three-sliding/one-full window, the centered-vs-plain norm split, NoPE
//! on full layers and the attention output gate all need no new logic here:
//! they are already correct per-token in `attn.rs`/`mlp.rs`/`mod.rs`, and
//! attention stays per-token and unbatched in this driver
//! (`TURBOSPARK_BATCHED_GEMV` is REFUSED by name, never silently ignored), so
//! the sliding-window ring is addressed by `position` exactly as the
//! sequential decode path already does it.

use std::time::Instant;

use foundation::LogitValue;

use super::state::{OUTPUT_MULTIPLIER, POST_NORM_EPS, RMS_EPS};
use super::{attn, layer_tensor, mlp};
use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::{encode_embed_any, encode_gemv_any};
use crate::real_forward_types::{RealForwardError, MAX_PREFILL_BATCH};
use crate::real_forward_utils::norm_view;
use crate::resid_capture::encode_resid_capture;
use crate::steering::encode_steering;

impl RealForwardRunner {
    /// Runs a whole prefill chunk through the `muse_glimmer` flow, writing
    /// the logits for the position after its last token.
    pub(crate) fn prefill_chunk_real_muse(
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
        // REFUSED BY NAME rather than ignored. This seam names the driver's
        // RESIDENT GEMVs, which every family has (this one's are INT4
        // affine, so the request is doubly plausible here); the M-row GEMM
        // is wired in Gemma 4's driver alone (step 6), so running the
        // per-token loop anyway would measure the unbatched engine under
        // the batched arm's label (crate Gotcha 22's rule).
        if self.batched_gemv_prefill {
            return Err(RealForwardError::Unsupported(
                "TURBOSPARK_BATCHED_GEMV is not wired for this family: the M-row \
                 resident GEMM exists in the gemma4 chunked driver alone \
                 (INT4-affine), and this driver keeps every resident GEMV per \
                 token"
                    .to_string(),
            ));
        }
        let mut offset = 0usize;
        while offset < tokens.len() {
            let take = (tokens.len() - offset).min(MAX_PREFILL_BATCH);
            let last = offset + take == tokens.len();
            self.prefill_micro_batch_muse(
                &tokens[offset..offset + take],
                start_position + offset,
                last,
                logits,
            )?;
            offset += take;
        }
        Ok(())
    }

    fn prefill_micro_batch_muse(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        want_head: bool,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let result = self.prefill_micro_batch_muse_inner(tokens, start_position, want_head, logits);
        self.phases.calls += tokens.len() as u64;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn prefill_micro_batch_muse_inner(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        want_head: bool,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let hidden = arch.hidden_size as usize;
        let inter = arch.intermediate_size as usize;
        let vocab = arch.vocab_size as usize;
        let softcap = arch.final_logit_softcap as f32;
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

        let pass = self.context.begin_pass_labeled("muse chunk cb");
        // No `sqrt(hidden)` embedding scale: this family norms the embedding
        // row instead, encoded below for every token, matching the
        // sequential flow's `embed_norm`.
        for (t, &token) in tokens.iter().enumerate() {
            let x_off = (t * hidden * 2) as u64;
            encode_embed_any(
                &mut self.context,
                &pass,
                &self.weights,
                &self.index,
                embed_name,
                (&self.scratch.x, x_off),
                token as u32,
                hidden as u32,
                1.0,
            )?;
            gpu::encode_rms_norm_no_scale(
                &mut self.context,
                &pass,
                (&self.scratch.x, x_off),
                (&self.scratch.x, x_off),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
        }

        let (context, weights, index, scratch, kv, muse, resid_capture, steering, ffn_hist) = (
            &mut self.context,
            &self.weights,
            &self.index,
            &self.scratch,
            &mut self.kv,
            self.real_muse.as_ref().expect("real muse state present"),
            self.resid_capture.as_ref(),
            self.steering.as_ref(),
            self.ffn_hist.as_ref(),
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

                // CENTERED, at the STANDARD epsilon, matching the sequential
                // flow exactly (`crates/runtime/CLAUDE.md`'s museGlimmer
                // bullet: this family's per-layer norms are `x * (1 + w)`).
                gpu::encode_rms_norm_bf16w_centered(
                    context,
                    &pass,
                    (&scratch.x, x_off),
                    input_norm,
                    (&scratch.normed, 0),
                    hidden as u32,
                    RMS_EPS,
                )
                .map_err(gpu_err)?;

                attn::encode_attention_block(
                    context, &pass, weights, index, &arch, muse, scratch, kv, layer, position,
                )?;

                // SANDWICH TAIL, HALF ONE: the attention output is
                // normalized (CENTERED, POST epsilon) before being added
                // back, this token's own row.
                gpu::encode_rms_norm_bf16w_centered(
                    context,
                    &pass,
                    (&scratch.o, 0),
                    post_attn_norm,
                    (&scratch.o_normed, 0),
                    hidden as u32,
                    POST_NORM_EPS,
                )
                .map_err(gpu_err)?;
                gpu::encode_residual_add(
                    context,
                    &pass,
                    (&scratch.x, x_off),
                    (&scratch.o_normed, 0),
                    hidden as u32,
                )
                .map_err(gpu_err)?;

                mlp::encode_mlp_block(
                    context, &pass, weights, index, scratch, ffn_hist, layer, hidden, inter, x_off,
                )?;

                // Steering first, then the capture, matching the sequential
                // flow's order and this token's own row.
                encode_steering(context, &pass, scratch, steering, layer, hidden, 1, x_off)?;
                encode_resid_capture(context, &pass, scratch, resid_capture, layer, hidden, x_off)?;
            }
        }

        if want_head {
            pass.relabel("muse final cb (head)");
            let last_off = ((m - 1) * hidden * 2) as u64;
            // **PLAIN, NOT CENTERED.** Matching the sequential flow's final
            // norm exactly (see the module doc on `mod.rs`).
            let final_norm = norm_view(weights, index, "language_model.model.norm.weight", hidden)?;
            gpu::encode_rms_norm_bf16w(
                context,
                &pass,
                (&scratch.x, last_off),
                final_norm,
                (&scratch.normed, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
            // Untied: `RealMuseState::build` refuses a tied install.
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
            // THE ORDER OF THESE TWO IS THE FUNCTION, matching the sequential
            // flow: `softcap(z * m)`, not `softcap(z) * m`.
            gpu::encode_scalar_mul(
                context,
                &pass,
                (&scratch.logits, 0),
                OUTPUT_MULTIPLIER,
                vocab as u32,
            )
            .map_err(gpu_err)?;
            gpu::encode_logit_softcap(context, &pass, (&scratch.logits, 0), softcap, vocab as u32)
                .map_err(gpu_err)?;
        } else {
            pass.relabel("muse chunk cb (no head)");
        }
        self.phases.final_cb_gpu_nanos += (pass.commit_and_wait_with_gpu_time() * 1e9) as u64;
        self.kv.advance_by(m);

        let skip_head = !want_head;
        let last_position = start_position + m - 1;
        if let Some(capture) = self.resid_capture.as_mut() {
            capture.record_pass(last_position, skip_head);
        }
        if let Some(hist) = self.ffn_hist.as_mut() {
            for t in 0..m {
                if t == m - 1 && want_head {
                    hist.record_pass();
                } else {
                    hist.note_prefill_pass();
                }
            }
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
