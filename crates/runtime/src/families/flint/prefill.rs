//! Chunked prefill for `flint`: Gemma 4's Step 1 (`docs/BATCHED_PREFILL.md`)
//! on this family's layer, behind `TURBOSPARK_FLINT_CHUNKED_PREFILL`.
//!
//! **What the switch changes, and what it must not.** OFF (the default) the
//! chunk entry point runs the sequential path token by token, which is what a
//! Flint install did before this file existed, so the old behaviour is one
//! variable away and a lab A/B shares a chunk label across both arms. ON, each
//! layer of a micro-batch of M tokens is ONE command buffer:
//!
//! ```text
//! cb1 (per layer): for each of M tokens, in order
//!     input norm -> attention (per token: sinks, window, NoPE) -> residual add
//!     -> post-attention norm into moe_x[t]
//!     dense lead (layers 0, 1): SwiGLU -> residual add, still inside cb1
//!     MoE layers:               INT8 router GEMV into router_logits[t]
//! commit + wait cb1                       (MoE layers only)
//! routed loop, per token, pipelined across two banks:
//!     host sigmoid top-k, expert plan + pread, shared expert, phase 1/2, add
//! ```
//!
//! The same dispatches run in the same order as M sequential `produce` calls,
//! so the output is meant to be BYTE-IDENTICAL; the lab declares it
//! `bit_exact` and a runtime test compares prefill-then-decode with
//! all-produce. What moves is where the host waits: one blocking wait per
//! layer per micro-batch instead of one per token, and the `pread` for token
//! t+1 overlaps the GPU running token t's routed pass.
//!
//! **Attention stays per token.** The batch attention kernel rejects sinks and
//! the ring (`BatchAttentionInputContract::validate`), so, as in gpt-oss,
//! `attn::encode_attention_block` is called unchanged per token and addresses
//! the ring by `position`. There is therefore no batched K/V write to straddle
//! the ring's wrap. The single-row scratch (`normed`, `q`, `o`, `attn_out`) is
//! reused across the M tokens inside one command buffer, which is safe because
//! commands in a command buffer run in order.
//!
//! M-row state is exactly what the host reads or writes between commits:
//! `moe_x` and `router_logits_f32` (see `state.rs`). `shared_out` and `h2` are
//! GPU-only and stay single-row, as in gpt-oss.

use std::sync::OnceLock;
use std::time::Instant;

use foundation::LogitValue;

use crate::families::flint::{attn, layer_tensor, moe};
use crate::moe_prefill_pipeline::{routed_pipeline_banks, RoutedSlot};
use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::{encode_embed_any, encode_gemv_any};
use crate::real_forward_types::{RealForwardError, MAX_PREFILL_BATCH};
use crate::real_forward_utils::{entry, norm_view};
use crate::resid_capture::encode_resid_capture;
use crate::steering::encode_steering;

/// `TURBOSPARK_FLINT_CHUNKED_PREFILL=1` selects the layer-batched driver. Read
/// once: half the runtime's switches are cached in `OnceLock`s, and the lab
/// starts a fresh process per arm for that reason.
fn chunked_prefill_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("TURBOSPARK_FLINT_CHUNKED_PREFILL").as_deref() == Ok("1"))
}

impl RealForwardRunner {
    /// Runs a whole prefill chunk through the `flint` flow, writing the logits
    /// for the position after its last token.
    pub(crate) fn prefill_chunk_real_flint(
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
        // REFUSED BY NAME, never silently ignored (crate Gotcha 22): both
        // seams name machinery this family does not have, and running the
        // per-token loop anyway would report the unbatched engine's number
        // under the batched arm's label.
        if self.batched_gemv_prefill {
            return Err(RealForwardError::Unsupported(
                "TURBOSPARK_BATCHED_GEMV is not wired for flint: its resident GEMVs stay per token"
                    .to_string(),
            ));
        }
        if self.routed_batch_prefill {
            return Err(RealForwardError::Unsupported(
                "TURBOSPARK_ROUTED_BATCH is not wired for flint: its routed half stays per token"
                    .to_string(),
            ));
        }
        if !chunked_prefill_enabled() {
            return self.prefill_chunk_flint_sequential(tokens, start_position, logits);
        }
        // The lab's engagement check: a run that set the switch must have
        // reached this driver (`crates/lab/src/knobs.rs`).
        foundation::engaged!("flint_chunked_prefill");
        let mut offset = 0usize;
        while offset < tokens.len() {
            let take = (tokens.len() - offset).min(MAX_PREFILL_BATCH);
            let last = offset + take == tokens.len();
            self.prefill_micro_batch_flint(
                &tokens[offset..offset + take],
                start_position + offset,
                last,
                logits,
            )?;
            offset += take;
        }
        Ok(())
    }

    /// The pre-driver behaviour: one `produce` per token, the head only on the
    /// last. The OFF arm of the switch.
    fn prefill_chunk_flint_sequential(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let last = tokens.len() - 1;
        for (i, &token) in tokens.iter().enumerate() {
            let position = start_position + i;
            if i == last {
                self.produce_real_flint(token, position, logits)?;
            } else {
                self.skip_head = true;
                let result = self.produce_real_flint(token, position, logits);
                self.skip_head = false;
                result?;
            }
        }
        Ok(())
    }

    fn prefill_micro_batch_flint(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        want_head: bool,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let result =
            self.prefill_micro_batch_flint_inner(tokens, start_position, want_head, logits);
        self.phases.calls += tokens.len() as u64;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn prefill_micro_batch_flint_inner(
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
        let top_k = arch.top_k_experts as usize;
        let routed_scale = arch.routed_scaling_factor as f32;
        let num_experts = arch.num_experts as usize;
        let gpu_err = RealForwardError::Gpu;
        let m = tokens.len();
        let lead = self.real_flint.as_ref().expect("flint state present").lead;

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

        // `TURBOSPARK_ROUTED_PIPELINE=0` means banks = 1 AND no protect set; the
        // two move together (gpt-oss's driver explains why at length).
        let banks = if self.routed_pipeline {
            routed_pipeline_banks(self.expert_cache_slots, top_k)
        } else {
            1
        };

        let embed_name = "language_model.model.embed_tokens.weight";
        let mut pass = self
            .context
            .begin_pass_labeled("flint chunk cb1 (attn+router)");
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
                self.encode_flint_chunk_token(
                    &pass,
                    layer,
                    start_position + t,
                    t,
                    hidden,
                    num_experts,
                )?;
            }
            if layer < lead {
                // A dense layer has no router and no commit: it stays inside
                // this command buffer and the next layer's tokens follow it.
                continue;
            }

            let cb1 = pass.commit();
            let t_wait = Instant::now();
            self.phases.cb1_gpu_nanos += (cb1.wait_with_gpu_time() * 1e9) as u64;
            self.phases.gpu_wait_nanos += t_wait.elapsed().as_nanos() as u64;

            // Nothing routed may still be in flight when the token loop
            // starts: token 0 protects no slots, so a survivor from the
            // previous layer could have its slots evicted under it.
            self.retire_routed(&mut pending_routed);
            self.encode_flint_layer_routed_moe_pipelined(
                &mut pending_routed,
                layer,
                hidden,
                moe_inter,
                num_experts,
                top_k,
                routed_scale,
                banks,
                m,
            )?;
            pass = self
                .context
                .begin_pass_labeled("flint chunk cb1 (attn+router)");
        }
        self.retire_routed(&mut pending_routed);

        if want_head {
            pass.relabel("flint final cb (head)");
            let last_off = ((m - 1) * hidden * 2) as u64;
            let final_norm = norm_view(
                &self.weights,
                &self.index,
                "language_model.model.norm.weight",
                hidden,
            )?;
            let rms_eps = self
                .real_flint
                .as_ref()
                .expect("flint state present")
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
            pass.relabel("flint chunk cb (no head)");
        }
        let t_wait = Instant::now();
        self.phases.final_cb_gpu_nanos += (pass.commit_and_wait_with_gpu_time() * 1e9) as u64;
        self.phases.final_wait_nanos += t_wait.elapsed().as_nanos() as u64;
        self.kv.advance_by(m);

        if let Some(capture) = self.resid_capture.as_mut() {
            capture.record_pass(start_position + m - 1, !want_head);
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

    /// One token's half of a layer that fits in the layer's command buffer:
    /// norm, attention, residual add, post-attention norm into this token's
    /// row of `moe_x`, then the dense lead's FFN or the INT8 router GEMV into
    /// this token's row of `router_logits_f32`.
    #[allow(clippy::too_many_arguments)]
    fn encode_flint_chunk_token(
        &mut self,
        pass: &gpu::PassEncoder,
        layer: usize,
        position: usize,
        t: usize,
        hidden: usize,
        num_experts: usize,
    ) -> Result<(), RealForwardError> {
        let gpu_err = RealForwardError::Gpu;
        let x_off = (t * hidden * 2) as u64;
        let input_norm = norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "input_layernorm.weight"),
            hidden,
        )?;
        let post_attn_norm = norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "post_attention_layernorm.weight"),
            hidden,
        )?;
        let state = self.real_flint.as_ref().expect("flint state present");

        gpu::encode_rms_norm_bf16w(
            &mut self.context,
            pass,
            (&self.scratch.x, x_off),
            input_norm,
            (&self.scratch.normed, 0),
            hidden as u32,
            state.rms_eps,
        )
        .map_err(gpu_err)?;
        attn::encode_attention_block(
            &mut self.context,
            pass,
            &self.weights,
            &self.index,
            &self.arch,
            state,
            &self.scratch,
            &self.kv,
            layer,
            position,
        )?;
        // RAW residual add; there is no sandwich norm.
        gpu::encode_residual_add(
            &mut self.context,
            pass,
            (&self.scratch.x, x_off),
            (&self.scratch.o, 0),
            hidden as u32,
        )
        .map_err(gpu_err)?;
        gpu::encode_rms_norm_bf16w(
            &mut self.context,
            pass,
            (&self.scratch.x, x_off),
            post_attn_norm,
            (&state.moe_x, x_off),
            hidden as u32,
            state.rms_eps,
        )
        .map_err(gpu_err)?;

        if layer < state.lead {
            // The dense lead: a plain SwiGLU, read from this token's row.
            moe::encode_swiglu(
                &mut self.context,
                pass,
                &self.weights,
                &self.index,
                state,
                "mlp",
                layer,
                hidden,
                state.dense_inter,
                x_off,
                (&self.scratch.o, 0),
            )?;
            gpu::encode_residual_add(
                &mut self.context,
                pass,
                (&self.scratch.x, x_off),
                (&self.scratch.o, 0),
                hidden as u32,
            )
            .map_err(gpu_err)?;
            encode_steering(
                &mut self.context,
                pass,
                &self.scratch,
                self.steering.as_ref(),
                layer,
                hidden,
                1,
                x_off,
            )?;
            encode_resid_capture(
                &mut self.context,
                pass,
                &self.scratch,
                self.resid_capture.as_ref(),
                layer,
                hidden,
                x_off,
            )?;
            return Ok(());
        }

        // The router GEMV runs on the GPU into this token's logits row; the
        // host reads it back after the layer's commit.
        let router_name = layer_tensor(layer, "mlp.gate.weight");
        let router = entry(&self.index, &router_name)?;
        let base = self.index.header.index_size;
        gpu::encode_router_gemv_gemma4(
            &mut self.context,
            pass,
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
            (&state.moe_x, x_off),
            (&state.router_ones, 0),
            (&state.router_logits_f32, (t * num_experts * 4) as u64),
            num_experts as u32,
            hidden as u32,
        )
        .map_err(gpu_err)
    }

    /// Pipelined routed MoE loop across a micro-batch's tokens for one layer:
    /// gpt-oss's `encode_gpt_oss_layer_routed_moe_pipelined` with Flint's pass.
    #[allow(clippy::too_many_arguments)]
    fn encode_flint_layer_routed_moe_pipelined(
        &mut self,
        pending_routed: &mut Option<gpu::CommittedPass>,
        layer: usize,
        hidden: usize,
        moe_inter: u32,
        num_experts: usize,
        top_k: usize,
        routed_scale: f32,
        banks: usize,
        m: usize,
    ) -> Result<(), RealForwardError> {
        let mut previous_slots: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for t in 0..m {
            if banks == 1 {
                self.retire_routed(pending_routed);
            }
            let slot = RoutedSlot {
                token: t,
                bank: t % banks,
                // Empty when pipelining is off: retire-before-plan has
                // already run (banks == 1 above), so no slot is in flight
                // for the plan to avoid.
                protect: if self.routed_pipeline && banks > 1 {
                    previous_slots.clone()
                } else {
                    std::collections::HashSet::new()
                },
            };
            let routed_pass = self.context.begin_pass_labeled("flint routed cb");
            let (
                context,
                weights,
                index,
                scratch,
                state,
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
                &self.scratch,
                self.real_flint.as_ref().expect("flint state present"),
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
            let used = moe::encode_flint_layer_moe(
                context,
                &routed_pass,
                weights,
                index,
                scratch,
                state,
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
                routed_scale,
                &slot,
            )?;

            // This token's OWN row, matching the other chunked drivers.
            let x_off = (t * hidden * 2) as u64;
            encode_steering(
                &mut self.context,
                &routed_pass,
                &self.scratch,
                self.steering.as_ref(),
                layer,
                hidden,
                1,
                x_off,
            )?;
            encode_resid_capture(
                &mut self.context,
                &routed_pass,
                &self.scratch,
                self.resid_capture.as_ref(),
                layer,
                hidden,
                x_off,
            )?;

            if banks > 1 {
                self.retire_routed(pending_routed);
            }
            debug_assert!(pending_routed.is_none(), "routed pipeline depth is 1");
            *pending_routed = Some(routed_pass.commit());
            previous_slots = used.into_iter().collect();
        }
        Ok(())
    }
}
