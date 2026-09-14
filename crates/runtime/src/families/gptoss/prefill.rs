//! Chunked prefill for `gpt-oss` (ROADMAP M5): the fifth
//! [`crate::producer::ChunkedPrefillRunner`] implementation, and the third
//! that pipelines a per-token routed half across a per-layer command
//! buffer (after Gemma 4's and the MoE half of `llama`'s).
//!
//! This is Gemma 4's Step 1 alone (`docs/BATCHED_PREFILL.md`), exactly as
//! `families/llama/moe_prefill.rs`'s header explains: all M tokens' norms,
//! projections, RoPE, attention and router GEMV go into ONE command buffer
//! per layer (`cb1`), then the routed half runs per token, pipelined via
//! [`crate::moe_prefill_pipeline::RoutedSlot`]. Steps 2/3 (the batched
//! routed KERNEL) are INT4-affine-only and not wired here; this driver
//! reuses the same per-token `encode_moe_phase1_any` / `encode_moe_phase2_any`
//! dispatch the sequential path already uses, which already resolves
//! `RoutedBlobLayout::GgufMxfp4` (this family's ONLY routed layout, per
//! `RealGptOssState::build`), so no new kernel is needed to widen it.
//!
//! **No ring-wrap hazard**, for the same reason as `families/llama/`'s
//! driver, despite this family HAVING a real sliding-window ring (unlike
//! `llama`): attention here stays per-token and unbatched
//! (`TURBOSPARK_BATCHED_GEMV` is REFUSED by name below, never silently
//! ignored), so there is no batched K/V
//! projection to straddle the ring's wrap. `attn::encode_attention_block`
//! is called unchanged, addressing the ring by `position` exactly as the
//! sequential decode path does.
//!
//! **The router bias must be added per token, before that token's top-k**,
//! replicating `moe.rs`'s own per-token host-side add exactly -- see that
//! file's module doc for why the placement (before top-k, not after) is
//! the correctness constraint. This driver's per-token routed loop calls
//! the SAME `moe::encode_gpt_oss_layer_moe`, which already does this
//! internally per call, so there is nothing extra to thread here beyond
//! `slot.token` selecting the right row of `router_logits_f32`.
//!
//! **No shared expert**, matching `llama`'s MoE half: phase 2's residual
//! seed is `scratch.zero_hidden`.

use std::time::Instant;

use foundation::LogitValue;

use crate::moe_prefill_pipeline::routed_pipeline_banks;
use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::{encode_embed_any, encode_gemv_any};
use crate::real_forward_types::{RealForwardError, MAX_PREFILL_BATCH};
use crate::real_forward_utils::norm_view;

impl RealForwardRunner {
    /// Runs a whole prefill chunk through the `gpt-oss` flow, writing the
    /// logits for the position after its last token.
    pub(crate) fn prefill_chunk_real_gpt_oss(
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
        // REFUSED BY NAME rather than ignored, and unlike
        // `TURBOSPARK_ROUTED_BATCH` this seam is meaningful on EVERY chunked
        // driver: it names the driver's RESIDENT GEMVs, and this family has
        // them (Q8_0 attention projections and head), where a dense family
        // merely has no routed half for the other flag to refer to. The
        // M-row GEMM is wired in Gemma 4's driver alone (INT4-affine,
        // step 6), so running the per-token loop anyway would hand back a
        // number from the unbatched engine under the batched arm's label --
        // the `encode_gemm_any` doctrine, found on this exact silent-ignore
        // class for `TURBOSPARK_ROUTED_BATCH` on qwen3moe (crate Gotcha 22).
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
            self.prefill_micro_batch_gpt_oss(
                &tokens[offset..offset + take],
                start_position + offset,
                last,
                logits,
            )?;
            offset += take;
        }
        Ok(())
    }

    fn prefill_micro_batch_gpt_oss(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        want_head: bool,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let result =
            self.prefill_micro_batch_gpt_oss_inner(tokens, start_position, want_head, logits);
        self.phases.calls += tokens.len() as u64;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn prefill_micro_batch_gpt_oss_inner(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        want_head: bool,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let hidden = arch.hidden_size as usize;
        let moe_inter = arch.moe_intermediate_size as u32;
        let vocab = arch.vocab_size as usize;
        let num_experts = arch.num_experts as usize;
        let top_k = arch.top_k_experts as usize;
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

        // `TURBOSPARK_ROUTED_PIPELINE=0` (the seam Gemma 4's sequential decode
        // has always read) means the same thing here: retire each token's
        // routed pass before the next plan (banks = 1) AND pass no protect
        // set. The two must move together -- with the previous pass retired
        // there is no in-flight slot for an empty protect set to endanger,
        // while an empty set at banks = 2 would let the planner evict a slot
        // the GPU is still reading. This is the A/B seam that separates the
        // batched arm's miss drop into its two candidate causes
        // (docs/BATCHED_PREFILL.md, step 5's miss-drop paragraph).
        // `banks == 1` does not make every slot count safe (see
        // `routed_pipeline_banks`'s doc and AGENTS.md Gotcha 64): a stale
        // `protect` set from the previous token can still exhaust the
        // cache. This family's top_k of 4 means `ALLOWED_CACHE_SLOTS`'
        // floor of 8 already satisfies `>= 2 * top_k`, so the CLI cannot
        // reach the failing branch here -- but a caller passing
        // `expert_cache_slots` directly below `2 * top_k` (a test or bench
        // harness, not the CLI) still can.
        let banks = if self.routed_pipeline {
            routed_pipeline_banks(self.expert_cache_slots, top_k)
        } else {
            1
        };
        if self.routed_batch_prefill {
            // Allocated on first use, so a run that never asks for the
            // batched half allocates none of its scratch. Idempotent, and
            // ahead of the layer loop so every `batched()` below it is
            // infallible -- `families/gemma4/prefill.rs`'s exact placement.
            let context = &mut self.context;
            self.real_gpt_oss
                .as_mut()
                .expect("real gpt-oss state present")
                .ensure_batched(context, &arch)?;
        }

        let embed_name = "language_model.model.embed_tokens.weight";

        let mut pass = self
            .context
            .begin_pass_labeled("gpt-oss chunk cb1 (attn+router)");
        // No `sqrt(hidden)` embedding scale, matching the sequential flow.
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
                vocab,
                1.0,
            )?;
        }

        let mut pending_routed: Option<gpu::CommittedPass> = None;
        for layer in 0..arch.num_layers as usize {
            for t in 0..m {
                let position = start_position + t;
                self.encode_gpt_oss_layer_attn_and_router(
                    &pass,
                    layer,
                    position,
                    t,
                    hidden,
                    num_experts,
                )?;
            }

            let cb1 = pass.commit();
            let t_wait = Instant::now();
            self.phases.cb1_gpu_nanos += (cb1.wait_with_gpu_time() * 1e9) as u64;
            self.phases.gpu_wait_nanos += t_wait.elapsed().as_nanos() as u64;

            // Nothing routed may still be in flight when the token loop
            // starts: token 0 protects no slots, so a survivor from the
            // previous layer could have its slots evicted under it.
            self.retire_routed(&mut pending_routed);

            if self.routed_batch_prefill {
                // Step 5: the whole layer's routed half as one MXFP4
                // route-list dispatch pair per union-bounded sub-batch
                // (`docs/BATCHED_PREFILL.md`). Same retire discipline as the
                // per-token path: committed here, retired after the NEXT
                // layer's router wait.
                pending_routed = Some(self.encode_gpt_oss_layer_routed_moe_batched(
                    layer,
                    hidden,
                    moe_inter,
                    num_experts,
                    top_k,
                    m,
                )?);
            } else {
                self.encode_gpt_oss_layer_routed_moe_pipelined(
                    &mut pending_routed,
                    layer,
                    hidden,
                    moe_inter,
                    num_experts,
                    top_k,
                    banks,
                    m,
                )?;
            }

            pass = self
                .context
                .begin_pass_labeled("gpt-oss chunk cb1 (attn+router)");
        }
        self.retire_routed(&mut pending_routed);

        if want_head {
            pass.relabel("gpt-oss final cb (head)");
            let last_off = ((m - 1) * hidden * 2) as u64;
            let final_norm = norm_view(
                &self.weights,
                &self.index,
                "language_model.model.norm.weight",
                hidden,
            )?;
            let rms_eps = self
                .real_gpt_oss
                .as_ref()
                .expect("real gpt-oss state present")
                .rms_eps;
            gpu::encode_rms_norm_bf16w(
                &mut self.context,
                &pass,
                (&self.scratch.x, last_off),
                final_norm,
                (&self.scratch.normed, 0),
                hidden as u32,
                rms_eps,
            )
            .map_err(gpu_err)?;
            // UNTIED, always: `RealGptOssState::build` refuses an install
            // claiming otherwise.
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
            // No softcap: `RealGptOssState::build` refuses an install that
            // declares one, matching the sequential flow.
        } else {
            pass.relabel("gpt-oss chunk cb (no head)");
        }
        let t_wait = Instant::now();
        self.phases.final_cb_gpu_nanos += (pass.commit_and_wait_with_gpu_time() * 1e9) as u64;
        self.phases.final_wait_nanos += t_wait.elapsed().as_nanos() as u64;
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
