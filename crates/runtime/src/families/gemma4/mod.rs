//! The real-checkpoint Gemma 4 decode flow for [`RealForwardRunner`].

mod attn;
mod moe;
mod moe_batch;
mod prefill;
mod state;

pub(crate) use state::RealGemmaState;

use std::time::Instant;

use foundation::LogitValue;

use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::{encode_embed_any, encode_gemm_any, encode_gemv_any};
use crate::real_forward_types::RealForwardError;
use crate::real_forward_utils::{layer_tensor, norm_view};
use crate::resid_capture::encode_resid_capture;
use crate::steering::encode_steering;

const RMS_EPS: f32 = 1e-6;

impl RealForwardRunner {
    /// The shared (dense) expert branch: INT8 gate/up on `dense_x`, gated
    /// activation, down projection, then `post_feedforward_layernorm_1`,
    /// encoded into its own command buffer and committed without waiting.
    ///
    /// `slot` names which token of a prefill micro-batch this is; it reads
    /// that token's `dense_x` row. `h1_out` is the row the down
    /// projection and `post_feedforward_layernorm_1` write into: the
    /// single-row `h1` on every existing path, and `batch_h1[t]` on the
    /// batched routed path, whose tail consumes all M tokens' shared
    /// outputs only after they have all been produced.
    pub(crate) fn encode_shared_expert_branch(
        &mut self,
        layer: usize,
        hidden: usize,
        inter: usize,
        use_silu: bool,
        slot: &moe::RoutedSlot,
        h1_out: (&gpu::MetalBuffer, u64),
    ) -> Result<(), RealForwardError> {
        let gpu_err = RealForwardError::Gpu;
        let x_off = (slot.token * hidden * 2) as u64;
        let shared_pass = self.context.begin_pass_labeled("shared-expert cb");
        let real = self.real.as_ref().expect("real state present");
        for (name, rows, cols, x_buf, y_buf) in [
            (
                layer_tensor(layer, "mlp.gate_proj.weight"),
                inter,
                hidden,
                &real.dense_x,
                &self.scratch.ffn_gate,
            ),
            (
                layer_tensor(layer, "mlp.up_proj.weight"),
                inter,
                hidden,
                &real.dense_x,
                &self.scratch.ffn_up,
            ),
        ] {
            encode_gemv_any(
                &mut self.context,
                &shared_pass,
                &self.weights,
                &self.index,
                &name,
                rows,
                cols,
                (x_buf, x_off),
                (y_buf, 0),
            )?;
        }
        let act = if use_silu {
            gpu::encode_silu_mul
        } else {
            gpu::encode_gelu_mul
        };
        act(
            &mut self.context,
            &shared_pass,
            (&self.scratch.ffn_gate, 0),
            (&self.scratch.ffn_up, 0),
            (&self.scratch.ffn_act, 0),
            inter as u32,
        )
        .map_err(gpu_err)?;
        encode_gemv_any(
            &mut self.context,
            &shared_pass,
            &self.weights,
            &self.index,
            &layer_tensor(layer, "mlp.down_proj.weight"),
            hidden,
            inter,
            (&self.scratch.ffn_act, 0),
            h1_out,
        )?;
        let post_ffn1 = norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "post_feedforward_layernorm_1.weight"),
            hidden,
        )?;
        gpu::encode_rms_norm_bf16w(
            &mut self.context,
            &shared_pass,
            h1_out,
            post_ffn1,
            h1_out,
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
        shared_pass.commit();
        Ok(())
    }

    /// The same branch for ALL `m` tokens of a prefill micro-batch, in one
    /// command buffer instead of `m` (`MFERENCE_BATCHED_GEMV`,
    /// `docs/BATCHED_PREFILL.md`).
    ///
    /// Two things are batched and one deliberately is not. The three
    /// projections become M-row GEMMs through `encode_gemm_any`, which is
    /// where the weight amortization lives; and the whole micro-batch shares
    /// ONE `begin_pass_labeled`, where the per-token form opens and commits
    /// `m` of them, which is host work the dispatch ranking never sees.
    /// The gated activation and `post_feedforward_layernorm_1` stay looped
    /// per row -- both are elementwise or per-row reductions with no weights
    /// to amortize, they are in the 21.2% of prefill that cannot batch at
    /// all, and looping them is what `families/qwen/batched_layers.rs`
    /// already does for the same pair of dispatches.
    ///
    /// `dense_x` is read from row 0 for all M rows because it already holds
    /// `MAX_PREFILL_BATCH` token-major rows, which is exactly the layout
    /// `encode_gemm_any` documents for its `x`.
    pub(crate) fn encode_shared_expert_branch_batched(
        &mut self,
        layer: usize,
        hidden: usize,
        inter: usize,
        use_silu: bool,
        m: usize,
    ) -> Result<(), RealForwardError> {
        let gpu_err = RealForwardError::Gpu;
        // Cloned up front so the `&mut self.context` borrows below do not
        // contend with the `&self.real` these live behind. Cheap: a
        // `MetalBuffer` clone is a handle, not the memory.
        let real = self.real.as_ref().expect("real state present");
        let dense_x = real.dense_x.clone();
        let batched = real.batched();
        let (gate, up, act, h1) = (
            batched.batch_ffn_gate.clone(),
            batched.batch_ffn_up.clone(),
            batched.batch_ffn_act.clone(),
            batched.batch_h1.clone(),
        );
        let shared_pass = self
            .context
            .begin_pass_labeled("shared-expert cb (batched)");
        for (suffix, out) in [("mlp.gate_proj.weight", &gate), ("mlp.up_proj.weight", &up)] {
            encode_gemm_any(
                &mut self.context,
                &shared_pass,
                &self.weights,
                &self.index,
                &layer_tensor(layer, suffix),
                inter,
                hidden,
                (&dense_x, 0),
                (out, 0),
                m,
            )?;
        }
        let activation = if use_silu {
            gpu::encode_silu_mul
        } else {
            gpu::encode_gelu_mul
        };
        for t in 0..m {
            let row = (t * inter * 2) as u64;
            activation(
                &mut self.context,
                &shared_pass,
                (&gate, row),
                (&up, row),
                (&act, row),
                inter as u32,
            )
            .map_err(gpu_err)?;
        }
        encode_gemm_any(
            &mut self.context,
            &shared_pass,
            &self.weights,
            &self.index,
            &layer_tensor(layer, "mlp.down_proj.weight"),
            hidden,
            inter,
            (&act, 0),
            (&h1, 0),
            m,
        )?;
        let post_ffn1 = norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "post_feedforward_layernorm_1.weight"),
            hidden,
        )?;
        for t in 0..m {
            let row = (t * hidden * 2) as u64;
            gpu::encode_rms_norm_bf16w(
                &mut self.context,
                &shared_pass,
                (&h1, row),
                post_ffn1,
                (&h1, row),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
        }
        shared_pass.commit();
        Ok(())
    }

    /// Times the whole forward pass into `phases.total_nanos`; the inner
    /// function accumulates the per-phase buckets it is carved into.
    pub(crate) fn produce_real_gemma4(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let result = self.produce_real_gemma4_inner(token, position, logits);
        self.phases.calls += 1;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn produce_real_gemma4_inner(
        &mut self,
        token: i32,
        position: usize,
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
        let seq_len = (position + 1) as u32;

        let embed_name = "language_model.model.embed_tokens.weight";
        let base = self.index.header.index_size;

        let mut pass = self.context.begin_pass_labeled("cb1 (attn+router)");
        encode_embed_any(
            &mut self.context,
            &pass,
            &self.weights,
            &self.index,
            embed_name,
            (&self.scratch.x, 0),
            token as u32,
            hidden as u32,
            embed_scale,
        )?;

        let mut pending_routed: Option<gpu::CommittedPass> = None;

        let sequential = moe::RoutedSlot::sequential();
        let h1 = self.real.as_ref().expect("real state present").h1.clone();
        for layer in 0..arch.num_layers as usize {
            self.encode_gemma4_layer_attn_and_router(&pass, layer, position, seq_len, base, 0)?;

            let cb1 = pass.commit();
            if self.shared_cb_overlap {
                self.encode_shared_expert_branch(
                    layer,
                    hidden,
                    inter,
                    use_silu,
                    &sequential,
                    (&h1, 0),
                )?;
            }

            let t_wait = Instant::now();
            self.phases.cb1_gpu_nanos += (cb1.wait_with_gpu_time() * 1e9) as u64;
            self.phases.gpu_wait_nanos += t_wait.elapsed().as_nanos() as u64;

            if let Some(pending) = pending_routed.take() {
                let t_retire = Instant::now();
                self.phases.routed_cb_gpu_nanos += (pending.wait_with_gpu_time() * 1e9) as u64;
                self.phases.pipeline_wait_nanos += t_retire.elapsed().as_nanos() as u64;
            }

            pass = self.context.begin_pass_labeled("routed cb");

            self.encode_gemma4_layer_routed_moe(
                &pass,
                layer,
                hidden,
                inter,
                moe_inter,
                num_experts,
                top_k,
                use_silu,
                &sequential,
            )?;
            // The layer's OUTPUT, on the SAME "routed cb" pass the routed
            // tail just encoded into: `encode_gemma4_layer_routed_moe`
            // ends with `layer_scalar`'s `encode_scalar_mul` over the
            // whole accumulated residual, which is the true end of this
            // layer's contribution and what the next layer's attention
            // reads. The sequential decode path is always token 0 of
            // `scratch.x`, so the offset is 0; the chunked-prefill driver
            // (`prefill.rs`) and the batched-routed tail
            // (`moe_batch.rs`) are the other two call sites, each at
            // their own token's offset.
            encode_steering(
                &mut self.context,
                &pass,
                &self.scratch,
                self.steering.as_ref(),
                layer,
                hidden,
                1,
                0,
            )?;
            encode_resid_capture(
                &mut self.context,
                &pass,
                &self.scratch,
                self.resid_capture.as_ref(),
                layer,
                hidden,
                0,
            )?;

            if self.routed_pipeline {
                debug_assert!(pending_routed.is_none(), "routed pipeline depth is 1");
                pending_routed = Some(pass.commit());
                pass = self.context.begin_pass_labeled("cb1 (attn+router)");
            }
        }

        if let Some(pending) = pending_routed.take() {
            let t_retire = Instant::now();
            self.phases.routed_cb_gpu_nanos += (pending.wait_with_gpu_time() * 1e9) as u64;
            self.phases.pipeline_wait_nanos += t_retire.elapsed().as_nanos() as u64;
        }

        if !self.skip_head {
            pass.relabel("final cb (head)");
            let final_norm = norm_view(
                &self.weights,
                &self.index,
                "language_model.model.norm.weight",
                hidden,
            )?;
            gpu::encode_rms_norm_bf16w(
                &mut self.context,
                &pass,
                (&self.scratch.x, 0),
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
            pass.relabel("final cb (prefill, no head)");
        }
        let t_wait = Instant::now();
        self.phases.final_cb_gpu_nanos += (pass.commit_and_wait_with_gpu_time() * 1e9) as u64;
        self.phases.final_wait_nanos += t_wait.elapsed().as_nanos() as u64;
        self.kv.advance();
        // The command buffer has been waited on, so every layer's capture
        // region is final. `record_pass` keeps at most one snapshot per
        // generation and decides which by `skip_head` -- called on every
        // pass rather than guarded here, matching the qwen and llama flows
        // (`crates/runtime/CLAUDE.md` Gotcha 20: the encode half alone
        // gives "no non-prefill pass ran; wrote nothing").
        let skip_head = self.skip_head;
        if let Some(capture) = self.resid_capture.as_mut() {
            capture.record_pass(position, skip_head);
        }

        if self.skip_head {
            return Ok(());
        }
        // Read the head STRAIGHT into the caller's slice. The owned-`Vec`
        // form of this cost a 512 KiB allocation and a second 512 KiB copy
        // per decoded token, outside every profiling bucket in this repo
        // (AGENTS.md Gotcha 23). The length check moves ahead of the read
        // because it was only ever comparing `vocab` to `logits.len()`.
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
