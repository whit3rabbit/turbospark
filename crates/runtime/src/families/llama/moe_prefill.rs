//! Chunked prefill for the MoE half of the `llama` architecture (Mixtral,
//! Qwen3MoE): the fourth [`crate::producer::ChunkedPrefillRunner`]
//! implementation, and the second (after Gemma 4's) that pipelines a
//! per-token routed half across a per-layer command buffer.
//!
//! This is Gemma 4's Step 1 alone (`docs/BATCHED_PREFILL.md`): all M
//! tokens' norms, projections, RoPE, attention and router GEMV go into ONE
//! command buffer per layer (`cb1`), then the routed half runs per token,
//! pipelined via [`crate::moe_prefill_pipeline::RoutedSlot`] exactly as
//! Gemma 4's does. Steps 2/3 (the batched routed KERNEL,
//! `TURBOSPARK_ROUTED_BATCH`) are INT4-affine-only and are not wired here;
//! this driver reuses the same per-token `encode_moe_phase1_any` /
//! `encode_moe_phase2_any` dispatch the sequential decode path already
//! uses, which is layout-agnostic (Affine, GGUF, ...), so no new kernel is
//! needed to widen this family.
//!
//! **No ring-wrap hazard**, unlike Gemma 4's batched-GEMV seam: this
//! architecture has no sliding-window layers at all (`RealLlamaState::build`
//! refuses any non-full-attention layer), and attention here stays
//! per-token and unbatched regardless (`TURBOSPARK_BATCHED_GEMV` is REFUSED
//! by name below, never silently ignored), so there is no batched K/V
//! projection to straddle a ring in the first place.
//!
//! **No shared expert**, unlike Gemma 4's routed half: phase 2's residual
//! seed is `scratch.zero_hidden`, so there is no shared-expert branch to
//! overlap and no `ROUTED_BANKS`-deep shared-expert scratch to worry about.

use std::time::Instant;

use foundation::LogitValue;

use crate::moe_prefill_pipeline::routed_pipeline_banks;
use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::{encode_embed_any, encode_gemv_any};
use crate::real_forward_types::{RealForwardError, MAX_PREFILL_BATCH};
use crate::real_forward_utils::norm_view;

impl RealForwardRunner {
    /// Runs a whole prefill chunk through the MoE half of the `llama` flow,
    /// writing the logits for the position after its last token. Call only
    /// once [`super::RealLlamaState::dense`] is known false; the dense half
    /// has its own driver (`prefill.rs`).
    pub(crate) fn prefill_chunk_real_llama_moe(
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
        // REFUSED BY NAME rather than ignored, and the asymmetry with the
        // DENSE drivers is the point. `TURBOSPARK_ROUTED_BATCH` asks for the
        // routed half as one route-list dispatch pair; this family HAS a
        // routed half and no batched kernel for its Q4_K/Q6_K blobs, so
        // running the per-token loop anyway would hand back a number from
        // the unbatched engine under the batched arm's label -- the
        // `encode_gemm_any` doctrine, and the reason both wired families
        // refuse each other's layout by name. A dense driver ignores the
        // flag legitimately: there is no routed half for it to refer to.
        // Step 5's MXFP4 arm is wired (`families/gptoss/moe_batch.rs`); the
        // Q4_K/Q6_K one was scoped by measurement and deliberately not
        // built (`docs/BATCHED_PREFILL.md`, "Step 5's two arms").
        if self.routed_batch_prefill {
            return Err(RealForwardError::Unsupported(
                "TURBOSPARK_ROUTED_BATCH is not wired for this family: the batched \
                 routed pair exists for INT4-affine (gemma4) and MXFP4 (gpt-oss) \
                 blobs, and this install's routed experts are GGUF K-quants"
                    .to_string(),
            ));
        }
        // The same rule for the resident-GEMV seam, which unlike the one
        // above is meaningful on EVERY chunked driver: this family has
        // resident GEMVs whatever its routed layout, and the M-row GEMM is
        // wired in Gemma 4's driver alone (step 6).
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
            self.prefill_micro_batch_llama_moe(
                &tokens[offset..offset + take],
                start_position + offset,
                last,
                logits,
            )?;
            offset += take;
        }
        Ok(())
    }

    fn prefill_micro_batch_llama_moe(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        want_head: bool,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let result =
            self.prefill_micro_batch_llama_moe_inner(tokens, start_position, want_head, logits);
        self.phases.calls += tokens.len() as u64;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn prefill_micro_batch_llama_moe_inner(
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

        // `banks == 1` does not make every slot count safe here: see
        // `routed_pipeline_banks`'s doc and AGENTS.md Gotcha 64.
        // `--expert-cache-slots == top_k` panics on the first multi-token
        // prefill (Qwen3-30B-A3B's top_k of 8 hits it; Mixtral's top_k of 2
        // does not, since 8 >= 2 * 2).
        let banks = routed_pipeline_banks(self.expert_cache_slots, top_k);

        let embed_name = "language_model.model.embed_tokens.weight";

        let mut pass = self
            .context
            .begin_pass_labeled("llama moe chunk cb1 (attn+router)");
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
                1.0,
            )?;
        }

        let mut pending_routed: Option<gpu::CommittedPass> = None;
        for layer in 0..arch.num_layers as usize {
            for t in 0..m {
                let position = start_position + t;
                self.encode_llama_layer_attn_and_router(
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

            self.encode_llama_layer_routed_moe_pipelined(
                &mut pending_routed,
                layer,
                hidden,
                moe_inter,
                num_experts,
                top_k,
                use_silu,
                banks,
                m,
            )?;

            pass = self
                .context
                .begin_pass_labeled("llama moe chunk cb1 (attn+router)");
        }
        self.retire_routed(&mut pending_routed);

        if want_head {
            pass.relabel("llama moe final cb (head)");
            let last_off = ((m - 1) * hidden * 2) as u64;
            let final_norm = norm_view(
                &self.weights,
                &self.index,
                "language_model.model.norm.weight",
                hidden,
            )?;
            let rms_eps = self
                .real_llama
                .as_ref()
                .expect("real llama state present")
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
            let head_name = if arch.tie_word_embeddings {
                embed_name.to_string()
            } else {
                "language_model.lm_head.weight".to_string()
            };
            encode_gemv_any(
                &mut self.context,
                &pass,
                &self.weights,
                &self.index,
                &head_name,
                vocab,
                hidden,
                (&self.scratch.normed, 0),
                (&self.scratch.logits, 0),
            )?;
            // No softcap: `RealLlamaState::build` refuses an install that
            // declares one, matching the sequential flow.
        } else {
            pass.relabel("llama moe chunk cb (no head)");
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
