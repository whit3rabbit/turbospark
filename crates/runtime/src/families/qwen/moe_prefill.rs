//! Chunked prefill for the MoE half of the qwen linear-attention flow
//! (`qwenGdnMoe`: Qwen 3.6 35B-A3B, Ornith 35B, MiniCPM-V 4.7's trunk).
//!
//! The same "Step 1" shape as `families/qwen4/prefill.rs` and
//! `families/gemma4/prefill.rs`: loop the EXISTING per-token kernels inside a
//! micro-batch of at most [`MAX_PREFILL_BATCH`] tokens, one command buffer
//! per layer for the attention-and-router half, then a per-token routed-MoE
//! loop pipelined with `RoutedSlot` and the routed banks. No new kernel:
//! every dispatch is the one the sequential flow (`produce.rs`) already
//! makes, in the same order per token, so the output is byte-identical to
//! sequential prefill rather than merely close.
//!
//! **The win is the per-layer blocking wait.** Sequential prefill commits and
//! waits on `cb1` once per layer per TOKEN, because the router's top-k needs
//! a host readback. Here it is once per layer per MICRO-BATCH.
//!
//! **Two buffers had to widen to `MAX_PREFILL_BATCH` rows** (`state.rs`):
//! `router_logits_f32`, because the host reads every token's logits back
//! after the layer's single commit, and `moe_x`, because `cb1` writes every
//! token's post-attention norm and each token's routed pass reads its own
//! row one commit later. `scratch.x` was already that wide. The GPU-only
//! intermediates (`normed`, `o`, `h1`, `h2`, the GDN scratch) stay single
//! row: command buffers on one queue execute in commit order, so token t+1
//! cannot read them before token t is done (crate Gotcha 14's "who WRITES
//! it" rule). `routing_w` and the routed argument buffer are the two the
//! HOST writes per token, so they are banked, not widened.
//!
//! **GDN needs no change**, for the dense driver's reason: the per-token
//! encoders advance the recurrent state in place, so calling them once per
//! token, in order, within one layer before the next starts, is exactly
//! sequential decode's math.
//!
//! **`TURBOSPARK_QWEN_MOE_CHUNKED=0` is the kill switch**, read once at
//! open. It makes `supports_chunked_prefill` answer `false`, so the CLI and
//! server fall back to sequential prefill without an error, and it is the
//! same-binary reference arm for byte-identity checks on a real install.
//!
//! REFUSED BY NAME, never silently served by the per-token loop:
//! `TURBOSPARK_ROUTED_BATCH` and `TURBOSPARK_BATCHED_GEMV` (neither arm is
//! wired for this family's MoE half), a hadamard-folded install (its
//! single-row `moe_x_h` would need a row per token), an open drafter (the
//! aux capture is not encoded here), and `TURBOSPARK_QWEN_SHARED_READ_OVERLAP`
//! (its separate shared pass has not been checked against the pipelined
//! routed loop). `supports_chunked_prefill` excludes the install-level ones,
//! so the default `--prefill-chunk` routing never reaches these errors.

use std::collections::HashSet;
use std::time::Instant;

use foundation::LogitValue;

use super::attn::{
    encode_full_attention_block, encode_linear_block, QkNormConvention, RopePosition,
};
use super::{layer_tensor, moe, RMS_EPS, TRUNK_PREFIX};
use crate::moe_prefill_pipeline::routed_pipeline_banks;
use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::encode_embed_any;
use crate::real_forward_types::{RealForwardError, MAX_PREFILL_BATCH};
use crate::real_forward_utils::norm_view;
use crate::resid_capture::encode_resid_capture;
use crate::steering::encode_steering;

impl RealForwardRunner {
    /// Why this install cannot take the MoE chunked driver, or `None` when
    /// it can. Install-level only: `supports_chunked_prefill` asks the same
    /// question, so the two cannot disagree.
    pub(crate) fn qwen_moe_chunked_refusal(&self) -> Option<&'static str> {
        let qwen = self.real_qwen.as_ref()?;
        if qwen.dense {
            return Some("not a MoE qwen install");
        }
        if !qwen.moe_chunked_prefill {
            return Some("an install opened with TURBOSPARK_QWEN_MOE_CHUNKED=0 (the kill switch)");
        }
        if qwen.hadamard.is_some() {
            return Some(
                "a hadamard-folded MoE install: its moe_x_h transform buffer is single-row",
            );
        }
        if qwen.shared_read_overlap {
            return Some(
                "TURBOSPARK_QWEN_SHARED_READ_OVERLAP is not wired for the MoE chunked driver",
            );
        }
        if self.real_mtp.is_some() || self.real_dflash.is_some() {
            return Some("a drafter is open, and this driver does not encode its aux capture");
        }
        None
    }

    /// Runs a whole prefill chunk through the qwen MoE flow, writing the
    /// logits for the position after its last token. Only the final token of
    /// the final micro-batch runs the output head, as `produce_prefill` skips
    /// it for every prompt token but the last.
    pub(crate) fn prefill_chunk_real_qwen_moe(
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
        if let Some(reason) = self.qwen_moe_chunked_refusal() {
            return Err(RealForwardError::Unsupported(format!(
                "the qwen MoE chunked prefill driver cannot serve {reason}; prefill sequentially"
            )));
        }
        // Both seams name an arm this driver does not have. Serving the
        // per-token loop under either label would report the unbatched
        // engine as the batched one (crate Gotcha 22).
        if self.routed_batch_prefill {
            return Err(RealForwardError::Unsupported(
                "TURBOSPARK_ROUTED_BATCH is not wired for the qwen MoE chunked driver: it \
                 keeps its routed half as a per-token loop"
                    .to_string(),
            ));
        }
        if self.batched_gemv_prefill {
            return Err(RealForwardError::Unsupported(
                "TURBOSPARK_BATCHED_GEMV is not wired for the qwen MoE chunked driver: it \
                 keeps every resident GEMV per token"
                    .to_string(),
            ));
        }
        let mut offset = 0usize;
        while offset < tokens.len() {
            let take = (tokens.len() - offset).min(MAX_PREFILL_BATCH);
            let last = offset + take == tokens.len();
            self.prefill_micro_batch_qwen_moe(
                &tokens[offset..offset + take],
                start_position + offset,
                last,
                logits,
            )?;
            offset += take;
        }
        Ok(())
    }

    fn prefill_micro_batch_qwen_moe(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        want_head: bool,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let result =
            self.prefill_micro_batch_qwen_moe_inner(tokens, start_position, want_head, logits);
        // One forward pass per TOKEN, matching every other phase divisor in
        // this port (`crates/runtime/CLAUDE.md` Gotcha 6).
        self.phases.calls += tokens.len() as u64;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn prefill_micro_batch_qwen_moe_inner(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        want_head: bool,
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

        // `banks == 1` does not make every slot count safe on its own: see
        // `routed_pipeline_banks`'s doc and AGENTS.md Gotcha 64, and the
        // empty `protect` set below.
        let banks = if self.routed_pipeline {
            routed_pipeline_banks(self.expert_cache_slots, top_k)
        } else {
            1
        };

        let embed_name = "language_model.model.embed_tokens.weight";
        let mut pass = self
            .context
            .begin_pass_labeled("qwen moe chunk cb1 (attn+router)");

        // THE IMAGE INJECTION, mirroring `prefill.rs` (the dense driver) and
        // `produce.rs`: at an image-pad position the tower's row REPLACES the
        // table lookup. The host write is safe for the dense driver's two
        // reasons, both properties of the destination: each row of
        // `scratch.x` is disjoint, and the blit replaces the per-row lookup
        // rather than racing it. It also lands before this pass's first
        // commit (the end of layer 0's `cb1`), which is what makes it visible
        // to every dispatch below. No hadamard arm: a folded install is
        // refused above.
        for (t, &token) in tokens.iter().enumerate() {
            match self
                .prompt_vision
                .as_ref()
                .and_then(|pv| pv.row_for(start_position + t))
            {
                Some(row) => gpu::write_buffer_bytes(&self.scratch.x, t * hidden * 2, row),
                None => encode_embed_any(
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
                )?,
            }
        }
        // The mRoPE angle per row, resolved once per micro-batch before the
        // borrow splits below, exactly as the dense driver does.
        let rope: Vec<RopePosition> = (0..m)
            .map(|t| match self.prompt_vision.as_ref() {
                Some(pv) => {
                    let (rt, rh, rw) = pv.rope_position(start_position + t);
                    RopePosition::Triple(rt, rh, rw)
                }
                None => RopePosition::Sequential,
            })
            .collect();

        // Shared-expert overlap passes would wait on drop; that seam is
        // refused above, so this stays empty, but it is held to the final
        // wait like `produce.rs`'s so a future arm cannot drop one early.
        let mut shared_submissions: Vec<gpu::CommittedPass> = Vec::new();
        let mut pending_routed: Option<gpu::CommittedPass> = None;

        for layer in 0..arch.num_layers as usize {
            {
                // The attention-and-router half: every token of the
                // micro-batch, in order, into ONE pass.
                let (context, weights, index, scratch, kv, qwen) = (
                    &mut self.context,
                    &self.weights,
                    &self.index,
                    &self.scratch,
                    &mut self.kv,
                    self.real_qwen.as_ref().expect("checked above"),
                );
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
                for (t, &rope_position) in rope.iter().enumerate() {
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
                    if arch.layer_is_linear(layer) {
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
                            start_position + t,
                            QkNormConvention::Plain,
                            rope_position,
                        )?;
                    }
                    // RAW residual add: this family has no sandwich norms.
                    gpu::encode_residual_add(
                        context,
                        &pass,
                        (&scratch.x, x_off),
                        (&scratch.o, 0),
                        hidden as u32,
                    )
                    .map_err(gpu_err)?;
                    // ONE post-attention norm feeds the router, the shared
                    // expert and the routed experts, into this token's row.
                    gpu::encode_rms_norm_bf16w(
                        context,
                        &pass,
                        (&scratch.x, x_off),
                        post_attn_norm,
                        (&qwen.moe_x, x_off),
                        hidden as u32,
                        RMS_EPS,
                    )
                    .map_err(gpu_err)?;
                    moe::encode_qwen_router(
                        context,
                        &pass,
                        weights,
                        index,
                        qwen,
                        layer,
                        hidden,
                        num_experts,
                        x_off,
                        (t * num_experts * 4) as u64,
                    )?;
                }
            }

            let cb1 = pass.commit();
            let t_wait = Instant::now();
            self.phases.cb1_gpu_nanos += (cb1.wait_with_gpu_time() * 1e9) as u64;
            self.phases.gpu_wait_nanos += t_wait.elapsed().as_nanos() as u64;

            // Nothing routed may still be in flight when the token loop
            // starts: token 0 protects no slots, so a survivor from the
            // previous layer could have its slots evicted under it.
            self.retire_routed(&mut pending_routed);

            let mut previous_slots: HashSet<usize> = HashSet::new();
            for t in 0..m {
                if banks == 1 {
                    self.retire_routed(&mut pending_routed);
                }
                let slot = moe::RoutedSlot {
                    token: t,
                    bank: t % banks,
                    // EMPTY when the pipeline degraded to one bank, and that
                    // is a correctness requirement (AGENTS.md Gotcha 64): the
                    // previous buffer is already retired, and reserving its
                    // slots anyway can leave too few places for this token's
                    // misses, which asserts in `ExpertCache::plan`.
                    protect: if banks == 1 {
                        HashSet::new()
                    } else {
                        previous_slots.clone()
                    },
                };
                let routed_pass = self.context.begin_pass_labeled("routed cb");
                let used = {
                    let (
                        context,
                        weights,
                        index,
                        scratch,
                        qwen,
                        streamers,
                        slot_buffers,
                        mapped,
                        routed_blobs,
                        routed_blobs_banks,
                        moe_offsets,
                        routed_layouts,
                        router_hist,
                        phases,
                        steering,
                        resid_capture,
                    ) = (
                        &mut self.context,
                        &self.weights,
                        &self.index,
                        &self.scratch,
                        self.real_qwen.as_ref().expect("checked above"),
                        &mut self.streamers,
                        &self.slot_buffers,
                        &self.mapped,
                        &self.routed_blobs,
                        &self.routed_blobs_banks,
                        &self.moe_offsets,
                        &self.routed_layouts,
                        &mut self.router_hist,
                        &mut self.phases,
                        self.steering.as_ref(),
                        self.resid_capture.as_ref(),
                    );
                    let (shared, used) = moe::encode_qwen_layer_moe(
                        context,
                        &routed_pass,
                        weights,
                        index,
                        scratch,
                        qwen,
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
                        inter,
                        moe_inter,
                        num_experts,
                        top_k,
                        use_silu,
                        &slot,
                    )?;
                    shared_submissions.extend(shared);
                    // After the MoE residual add, in the sequential flow's
                    // order: steering, then the residual capture.
                    let x_off = (t * hidden * 2) as u64;
                    encode_steering(
                        context,
                        &routed_pass,
                        scratch,
                        steering,
                        layer,
                        hidden,
                        1,
                        x_off,
                    )?;
                    encode_resid_capture(
                        context,
                        &routed_pass,
                        scratch,
                        resid_capture,
                        layer,
                        hidden,
                        x_off,
                    )?;
                    used
                };

                if banks > 1 {
                    self.retire_routed(&mut pending_routed);
                }
                debug_assert!(pending_routed.is_none(), "routed pipeline depth is 1");
                pending_routed = Some(routed_pass.commit());
                previous_slots = used.into_iter().collect();
            }

            pass = self
                .context
                .begin_pass_labeled("qwen moe chunk cb1 (attn+router)");
        }
        self.retire_routed(&mut pending_routed);

        if want_head {
            let (context, weights, index, scratch, qwen) = (
                &mut self.context,
                &self.weights,
                &self.index,
                &self.scratch,
                self.real_qwen.as_ref().expect("checked above"),
            );
            // The dense driver's head encoder reads the LAST row of
            // `scratch.x`, which is the same final norm and head GEMV the
            // sequential flow runs. Hadamard is refused above, so its
            // folded-head arm is unreachable here.
            super::prefill_layers::encode_qwen_dense_chunk_head(
                context, &pass, weights, index, scratch, qwen, &arch, embed_name, hidden, vocab, m,
            )?;
            pass.relabel("qwen moe final cb (head)");
        } else {
            pass.relabel("qwen moe final cb (chunk, no head)");
        }
        let t_wait = Instant::now();
        self.phases.final_cb_gpu_nanos += (pass.commit_and_wait_with_gpu_time() * 1e9) as u64;
        self.phases.final_wait_nanos += t_wait.elapsed().as_nanos() as u64;
        drop(shared_submissions);
        self.kv.advance_by(m);

        let last_position = start_position + m - 1;
        if let Some(capture) = self.resid_capture.as_mut() {
            capture.record_pass(last_position, !want_head);
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
