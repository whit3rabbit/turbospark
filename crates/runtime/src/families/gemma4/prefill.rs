use std::time::Instant;

use foundation::LogitValue;

use super::moe;
use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::{encode_embed_any, encode_gemv_any};
use crate::real_forward_types::{RealForwardError, MAX_PREFILL_BATCH, ROUTED_BANKS};
use crate::real_forward_utils::norm_view;

const RMS_EPS: f32 = 1e-6;

impl RealForwardRunner {
    /// Runs a whole prefill chunk, writing the logits for the position
    /// after its last token. The chunk is walked in micro-batches of at
    /// most [`MAX_PREFILL_BATCH`]; only the final token of the final
    /// micro-batch runs the output head, exactly as `produce_prefill`
    /// skips it for every prompt token but the last.
    pub(crate) fn prefill_chunk_real_gemma4(
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
            self.prefill_micro_batch_gemma4(
                &tokens[offset..offset + take],
                start_position + offset,
                last,
                logits,
            )?;
            offset += take;
        }
        Ok(())
    }

    /// One micro-batch: every layer runs its attention-and-router half for
    /// ALL `tokens` into ONE command buffer, so the per-layer blocking wait
    /// is paid once per micro-batch instead of once per token, and then its
    /// routed half -- per token by default, or as one route-list dispatch
    /// pair per union-bounded sub-batch when `MFERENCE_ROUTED_BATCH=1`
    /// (`docs/BATCHED_PREFILL.md` steps 2 and 3, `moe_batch.rs`).
    ///
    /// The default keeps the routed half per token because the two things
    /// it needs from the host -- the routing weights and the argument
    /// buffer naming this token's eight expert slots -- are written
    /// outside the command queue, and because the batched pair is what
    /// the route-list kernels were built for.
    fn prefill_micro_batch_gemma4(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        want_head: bool,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let result =
            self.prefill_micro_batch_gemma4_inner(tokens, start_position, want_head, logits);
        // One forward pass per TOKEN, not per micro-batch: every
        // `MFERENCE_PHASES` bucket divides by this, and the whole point of
        // the comparison is cost per prompt token.
        self.phases.calls += tokens.len() as u64;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn prefill_micro_batch_gemma4_inner(
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
        let embed_scale = if arch.embedding_scaled_by_sqrt_hidden {
            (hidden as f32).sqrt()
        } else {
            1.0
        };
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

        // How many tokens' routed command buffers may be in flight at once.
        // Pipelining costs a plan that must AVOID the previous token's
        // slots, so the cache needs room for those plus this token's misses;
        // below `2 * top_k` slots it cannot guarantee that and
        // `ExpertCache::plan` aborts the process rather than degrading, so
        // a small cache falls back to retiring before it encodes.
        let banks = if self.expert_cache_slots >= 2 * top_k {
            ROUTED_BANKS
        } else {
            1
        };

        let embed_name = "language_model.model.embed_tokens.weight";
        let base = self.index.header.index_size;

        let mut pass = self.context.begin_pass_labeled("chunk cb1 (attn+router)");
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
                embed_scale,
            )?;
        }

        let mut pending_routed: Option<gpu::CommittedPass> = None;
        for layer in 0..arch.num_layers as usize {
            for t in 0..m {
                let position = start_position + t;
                self.encode_gemma4_layer_attn_and_router(
                    &pass,
                    layer,
                    position,
                    (position + 1) as u32,
                    base,
                    t,
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
                // Steps 2 and 3: the whole layer's routed half as one
                // route-list dispatch pair per union-bounded sub-batch
                // (`docs/BATCHED_PREFILL.md`). Same retire discipline as
                // the per-token path: committed here, retired after the
                // NEXT layer's router wait.
                pending_routed = Some(self.encode_gemma4_layer_routed_moe_batched(
                    layer,
                    hidden,
                    inter,
                    moe_inter,
                    num_experts,
                    top_k,
                    use_silu,
                    m,
                )?);
            } else {
                let mut previous_slots: std::collections::HashSet<usize> =
                    std::collections::HashSet::new();
                let h1 = self.real.as_ref().expect("real state present").h1.clone();
                for t in 0..m {
                    if banks == 1 {
                        self.retire_routed(&mut pending_routed);
                    }
                    let slot = moe::RoutedSlot {
                        token: t,
                        bank: t % banks,
                        protect: previous_slots.clone(),
                    };
                    if self.shared_cb_overlap {
                        self.encode_shared_expert_branch(
                            layer,
                            hidden,
                            inter,
                            use_silu,
                            &slot,
                            (&h1, 0),
                        )?;
                    }
                    let routed_pass = self.context.begin_pass_labeled("routed cb");
                    let used = self.encode_gemma4_layer_routed_moe(
                        &routed_pass,
                        layer,
                        hidden,
                        inter,
                        moe_inter,
                        num_experts,
                        top_k,
                        use_silu,
                        &slot,
                    )?;
                    if banks > 1 {
                        self.retire_routed(&mut pending_routed);
                    }
                    debug_assert!(pending_routed.is_none(), "routed pipeline depth is 1");
                    pending_routed = Some(routed_pass.commit());
                    previous_slots = used.into_iter().collect();
                }
            }

            pass = self.context.begin_pass_labeled("chunk cb1 (attn+router)");
        }
        self.retire_routed(&mut pending_routed);

        if want_head {
            pass.relabel("final cb (head)");
            let last_off = ((m - 1) * hidden * 2) as u64;
            let final_norm = norm_view(
                &self.weights,
                &self.index,
                "language_model.model.norm.weight",
                hidden,
            )?;
            gpu::encode_rms_norm_bf16w(
                &mut self.context,
                &pass,
                (&self.scratch.x, last_off),
                final_norm,
                (&self.scratch.normed, 0),
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
            if arch.final_logit_softcap > 0.0 {
                gpu::encode_logit_softcap(
                    &mut self.context,
                    &pass,
                    (&self.scratch.logits, 0),
                    arch.final_logit_softcap as f32,
                    vocab as u32,
                )
                .map_err(gpu_err)?;
            }
        } else {
            pass.relabel("final cb (chunk, no head)");
        }
        let t_wait = Instant::now();
        self.phases.final_cb_gpu_nanos += (pass.commit_and_wait_with_gpu_time() * 1e9) as u64;
        self.phases.final_wait_nanos += t_wait.elapsed().as_nanos() as u64;
        self.kv.advance_by(m);

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

    /// Waits out a pipelined routed command buffer and books its GPU time,
    /// leaving `pending` empty. A no-op when nothing is in flight.
    pub(crate) fn retire_routed(&mut self, pending: &mut Option<gpu::CommittedPass>) {
        if let Some(committed) = pending.take() {
            let t_retire = Instant::now();
            self.phases.routed_cb_gpu_nanos += (committed.wait_with_gpu_time() * 1e9) as u64;
            self.phases.pipeline_wait_nanos += t_retire.elapsed().as_nanos() as u64;
        }
    }
}
