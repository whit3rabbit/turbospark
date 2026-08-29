//! Chunked prefill for the DENSE half of the qwen linear-attention flow
//! (`qwenGdnDense`, `qwen38-27b.gturbo`): the sixth
//! [`crate::producer::ChunkedPrefillRunner`] implementation, and the same
//! "step 1" shape as `families/llama/prefill.rs` and
//! `families/museglimmer/prefill.rs` -- loop the EXISTING per-token kernels
//! inside a micro-batch, batching command buffers rather than GEMVs.
//!
//! **This is a different design than `families/qwen/batched.rs`.** That file
//! implements `docs/BATCHED_PREFILL.md` steps 2-6 (GEMVs become GEMMs) for
//! the MTP/DFlash2 verify pass, sized for tiny block depths and allocated
//! only when a drafter is open. This driver needs neither: it calls
//! `attn::encode_linear_block` / `attn::encode_full_attention_block` and
//! `dense::encode_qwen_layer_dense` exactly as the sequential flow
//! (`produce.rs`) does, once per token, inside one command buffer per
//! micro-batch instead of one per token. **No new kernel, no new buffer.**
//!
//! Three properties make that safe.
//!
//! `attn.rs`'s two block encoders read `scratch.normed` / write `scratch.o`
//! at offset 0 and never touch `scratch.x` directly, so they need no change
//! (the same property that let `llama`'s and `muse_glimmer`'s attention
//! blocks land unmodified, `crates/runtime/CLAUDE.md` Gotcha 14).
//!
//! The GDN recurrent state (`qwen.gdn.state_buffer(layer)` /
//! `conv_tail_buffer(layer)`) is a persistent per-layer buffer that
//! `encode_linear_block`'s decode-shaped kernels advance in place, with no
//! position argument -- it does not know whether it is being called from a
//! chunk or from sequential decode, only that it is called once per token.
//! Calling it once per token, strictly in increasing `t` order, within one
//! layer's inner loop, before moving to the next layer, reproduces
//! sequential decode's math exactly. That ordering is what
//! `crates/runtime/CLAUDE.md` Gotcha 4 requires ("`reset()` must rewind the
//! GDN state, not just the KV cache") -- satisfied here by construction
//! rather than by a new mechanism, and it is also what makes cross-chunk
//! continuity free: the state buffer is the same one sequential decode
//! reads and writes, so a prompt spanning several micro-batches (or several
//! `prefill_chunk` calls) carries it forward automatically.
//!
//! `qwen.moe_x`, `qwen.h2`, and the GDN scratch fields on `RealQwenState`
//! are single-row GPU-only intermediates, safe to reuse per token within
//! one command buffer because a serial compute encoder runs dispatches in
//! commit order (`crates/gpu/CLAUDE.md` Gotcha 8) -- exactly the reasoning
//! `families/llama/prefill.rs` already established for `llama.moe_x`/`h2`.
//!
//! KV writes stay per-token via the existing `k_slot`/`v_slot` calls inside
//! `encode_full_attention_block`, so there is no batched-projection KV-wrap
//! hazard to guard against (that hazard belongs to the M-row GEMM path in
//! `batched_layers.rs`, which this driver does not use).
//!
//! Two refusals are BY NAME rather than silent, per this repo's convention:
//!
//! - **Vision.** The image injection in `produce.rs` is this family's only
//!   embedding call site today (Gotcha 27). This first cut stays text-only;
//!   `RealForwardRunner::supports_chunked_prefill` also excludes an install
//!   with a live `prompt_vision` map so callers route around this driver
//!   entirely rather than reaching the refusal.
//! - **An open drafter.** `produce.rs`'s dense branch fires the DFlash2
//!   aux-capture hook on every forward pass, which `dflash_prime_from_capture`
//!   later reads; this driver does not encode that hook, so it refuses
//!   rather than silently prefilling a prompt whose aux cache the drafter
//!   would read back empty.

use std::time::Instant;

use foundation::LogitValue;

use super::attn::{
    encode_full_attention_block, encode_linear_block, QkNormConvention, RopePosition,
};
use super::{dense, layer_tensor, RMS_EPS, TRUNK_PREFIX};
use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::{encode_embed_any, encode_gemv_any};
use crate::real_forward_types::{RealForwardError, MAX_PREFILL_BATCH};
use crate::real_forward_utils::norm_view;
use crate::resid_capture::encode_resid_capture;
use crate::steering::encode_steering;

impl RealForwardRunner {
    /// Runs a whole prefill chunk through the dense qwen flow, writing the
    /// logits for the position after its last token. Call only once
    /// `RealQwenState::dense` is known true; a MoE install is refused at
    /// [`crate::producer::ChunkedPrefillRunner::prefill_chunk`], by name,
    /// before this is reached.
    pub(crate) fn prefill_chunk_real_qwen_dense(
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
        if self.prompt_vision.is_some() {
            return Err(RealForwardError::Unsupported(
                "the qwen chunked prefill driver is text-only: this install has an image \
                 prompt attached, and the image injection's only embedding call site is the \
                 sequential flow"
                    .to_string(),
            ));
        }
        if self.real_mtp.is_some() || self.real_dflash.is_some() {
            return Err(RealForwardError::Unsupported(
                "the qwen chunked prefill driver does not encode the drafter's aux capture; \
                 open without MFERENCE_MTP_DRAFT / MFERENCE_DFLASH_DRAFT to use it, or prefill \
                 sequentially"
                    .to_string(),
            ));
        }
        // REFUSED BY NAME rather than ignored, matching every other chunked
        // driver (`crates/runtime/CLAUDE.md` Gotcha 22): this seam names the
        // driver's RESIDENT GEMVs, which every family has, and the M-row GEMM
        // for those is wired in Gemma 4's driver alone. Running the per-token
        // loop anyway would measure the unbatched engine under the batched
        // arm's label.
        if self.batched_gemv_prefill {
            return Err(RealForwardError::Unsupported(
                "MFERENCE_BATCHED_GEMV is not wired for this family: the M-row resident GEMM \
                 exists in the gemma4 chunked driver alone, and this driver keeps every \
                 resident GEMV per token"
                    .to_string(),
            ));
        }
        let mut offset = 0usize;
        while offset < tokens.len() {
            let take = (tokens.len() - offset).min(MAX_PREFILL_BATCH);
            let last = offset + take == tokens.len();
            self.prefill_micro_batch_qwen_dense(
                &tokens[offset..offset + take],
                start_position + offset,
                last,
                logits,
            )?;
            offset += take;
        }
        Ok(())
    }

    fn prefill_micro_batch_qwen_dense(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        want_head: bool,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let result =
            self.prefill_micro_batch_qwen_dense_inner(tokens, start_position, want_head, logits);
        // One forward pass per TOKEN, matching every other phase divisor in
        // this port (`crates/runtime/CLAUDE.md` Gotcha 6).
        self.phases.calls += tokens.len() as u64;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn prefill_micro_batch_qwen_dense_inner(
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
        let use_silu = arch.hidden_activation.contains("silu");
        let gpu_err = RealForwardError::Gpu;
        let m = tokens.len();

        if !self.real_qwen.as_ref().is_some_and(|s| s.dense) {
            return Err(RealForwardError::Unsupported(
                "prefill_chunk_real_qwen_dense called on a non-dense (MoE) qwen install"
                    .to_string(),
            ));
        }
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

        let pass = self.context.begin_pass_labeled("qwen dense chunk cb");
        // No `sqrt(hidden)` embedding scale, matching the sequential flow:
        // `RealQwenState::build` refuses an install that declares
        // `embeddingScaledBySqrtHidden`.
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

        let (context, weights, index, scratch, kv, qwen, resid_capture, steering) = (
            &mut self.context,
            &self.weights,
            &self.index,
            &self.scratch,
            &mut self.kv,
            self.real_qwen.as_ref().expect("checked dense above"),
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
            let is_linear = arch.layer_is_linear(layer);

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
                    RMS_EPS,
                )
                .map_err(gpu_err)?;

                if is_linear {
                    // Mask-2: gated DeltaNet. No position, no RoPE, no KV --
                    // the recurrent state advances once per call, in the
                    // order this loop calls it, which is this module's
                    // whole correctness argument (see the file doc comment).
                    encode_linear_block(
                        context, &pass, weights, index, &arch, qwen, scratch, layer,
                    )?;
                } else {
                    encode_full_attention_block(
                        context,
                        &pass,
                        weights,
                        index,
                        &arch,
                        qwen,
                        scratch,
                        kv,
                        TRUNK_PREFIX,
                        layer,
                        position,
                        QkNormConvention::Plain,
                        // Text-only: refused above when `prompt_vision` is set.
                        RopePosition::Sequential,
                    )?;
                }

                // RAW residual add, matching the sequential flow: this
                // family has no sandwich norms.
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
                    (&qwen.moe_x, 0),
                    hidden as u32,
                    RMS_EPS,
                )
                .map_err(gpu_err)?;

                dense::encode_qwen_layer_dense(
                    context,
                    &pass,
                    weights,
                    index,
                    scratch,
                    qwen,
                    &scratch.x,
                    x_off,
                    TRUNK_PREFIX,
                    layer,
                    hidden,
                    inter,
                    use_silu,
                )?;

                encode_steering(context, &pass, scratch, steering, layer, hidden, 1, x_off)?;
                encode_resid_capture(context, &pass, scratch, resid_capture, layer, hidden, x_off)?;
            }
        }

        if want_head {
            pass.relabel("qwen dense final cb (head)");
            let last_off = ((m - 1) * hidden * 2) as u64;
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
            // No softcap: `RealQwenState::build` refuses an install that
            // declares one, matching the sequential flow.
        } else {
            pass.relabel("qwen dense chunk cb (no head)");
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
