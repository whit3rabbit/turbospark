//! Chunked prefill for `qwen4_exp`: the same "Step 1" shape as
//! `families/gemma4/prefill.rs` -- loop the EXISTING per-token kernels
//! inside a micro-batch of at most [`MAX_PREFILL_BATCH`] tokens, one command
//! buffer per layer for the attention-and-router half, then a per-token
//! routed-MoE loop pipelined the same way gemma4's is. No new kernel.
//!
//! **Five buffers had to widen to `MAX_PREFILL_BATCH` rows for this to be
//! correct, not just fast.** Four are documented at their declaration in
//! `state.rs`: `wide_x` (the residual crosses layers, so every token needs
//! its own row at once, matching `DecodeScratch::x`'s reason exactly),
//! `router_logits_f32` (the host reads back every token's router logits in
//! one go after the layer's single commit), and `hc_inject` plus the new
//! `moe_x` (both bridge the `cb1`/`"routed cb"` split for `mlp_hc`: its
//! mixed output and inject gate are read one commit later than they are
//! written, so a chunked driver that encodes a whole micro-batch's `cb1`
//! before committing once would have every earlier token's value overwritten
//! by the last token's, were either single-row). The fifth, PLE's
//! `ngram_emb`, is a different and sharper case entirely -- see below.
//!
//! **The QSA indexer needs no new buffer or signature change at all.**
//! `encode_full_attention_block` already takes `pass: &mut PassEncoder` and
//! already handles its own above-budget mid-layer commit internally
//! (`attn.rs`'s own doc); calling it once per token, in increasing `t`
//! order, inside this driver's per-layer loop reproduces that exactly. The
//! shared `qsa_positions` buffer (`RealQwen4State`, one buffer for every QSA
//! layer) stays safe as long as this driver preserves gemma4's own ordering:
//! a layer's whole attention-and-router half commits and WAITS before that
//! layer's routed loop starts, and the next layer's pass is not created
//! until the current layer's routed loop finishes -- so no two QSA layers'
//! writes to the shared buffer are ever in flight at once
//! (`crates/runtime/CLAUDE.md` Gotcha 34).
//!
//! **GDN needs no change.** `encode_linear_block`'s kernels take no position
//! argument and advance `qwen4.gdn`'s recurrent state in place; calling it
//! once per token, strictly in increasing order, within one layer's inner
//! loop before the next layer starts, reproduces sequential decode's math
//! exactly (`families/qwen/prefill.rs`'s identical argument, one family
//! over).
//!
//! **PLE needed a FIFTH widened buffer, and it is the sharpest instance of
//! this whole pattern.** `ple.rs`'s `ngram_emb` upload
//! (`gpu::write_buffer_bytes`) is a HOST write, executed the instant
//! `encode_ple_layer` runs -- not a GPU dispatch queued for later, so it does
//! not respect command-buffer commit order AT ALL, unlike every other
//! per-token buffer in this family. A single-row `ngram_emb` left every
//! token's `key_proj`/`value_proj` GEMV reading whichever token's embedding
//! was written LAST, since none of those GEMVs execute until the whole pass
//! commits, by which point every token's host write in the micro-batch has
//! already landed -- every token but the last silently computed PLE from the
//! WRONG token's n-gram embedding. This shipped broken on the first cut of
//! this driver and was caught by `real_forward_qwen4_chunked.rs`'s
//! `the_chunk_boundary_does_not_move_the_logits`, at chunk span 2, the first
//! multi-token micro-batch the test tried. Fixed the same way as `moe_x`:
//! widen to `MAX_PREFILL_BATCH` rows, thread a row offset into
//! `encode_ple_layer` for both the write and its two subsequent reads. PLE's
//! two pieces of TRUE recurrent state (`ple_conv_tail`, `ngram_context`) are
//! unaffected and need no change: both advance once per token in host or GPU
//! program order, with no cross-token aliasing.
//!
//! Matches `produce.rs`'s current feature set rather than gemma4's: no
//! steering, no residual capture (this family's sequential decode calls
//! neither), no vision, no drafter -- all out of scope for this cut,
//! `mod.rs`'s own doc.
//!
//! BOTH BATCHING SEAMS ARE REFUSED BY NAME, never silently ignored, which
//! is the pair every other chunked driver carries (crate Gotcha 7's closing
//! clause). `TURBOSPARK_ROUTED_BATCH` is MEANINGFUL here rather than
//! vacuous -- this family has a routed half, unlike the dense drivers that
//! may ignore it legitimately -- and `TURBOSPARK_BATCHED_GEMV` names
//! resident GEMVs, which every family has. Neither is wired for this
//! driver, so serving the per-token loop under either arm's label would
//! report the unbatched engine as the batched one (crate Gotcha 22).

use std::collections::HashSet;
use std::time::Instant;

use foundation::LogitValue;

use super::attn::{encode_full_attention_block, encode_linear_block};
use super::hc::encode_hyper_connection;
use super::moe;
use super::ple::encode_ple_layer;
use super::{layer_tensor, TRUNK_PREFIX};
use crate::moe_prefill_pipeline::routed_pipeline_banks;
use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::{encode_embed_any, encode_gemv_any};
use crate::real_forward_types::{RealForwardError, MAX_PREFILL_BATCH};

impl RealForwardRunner {
    /// Runs a whole prefill chunk through the qwen4_exp flow, writing the
    /// logits for the position after its last token. The chunk is walked in
    /// micro-batches of at most [`MAX_PREFILL_BATCH`]; only the final token
    /// of the final micro-batch runs the output head, exactly as
    /// `produce_prefill` skips it for every prompt token but the last.
    pub(crate) fn prefill_chunk_real_qwen4(
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
        // REFUSED BY NAME rather than ignored, the same pair every other
        // chunked driver carries (crate Gotcha 7's closing clause, and
        // Gotcha 22's silent-ignore rule).
        //
        // `TURBOSPARK_ROUTED_BATCH` asks for the routed half as one
        // route-list dispatch pair. This family HAS a routed half, so the
        // request is MEANINGFUL here where a dense driver may ignore it
        // legitimately, and the batched pair is wired in the gemma4
        // (INT4-affine) and gpt-oss (MXFP4) drivers alone. Running the
        // per-token routed loop anyway would report the unbatched engine
        // under the batched arm's label.
        if self.routed_batch_prefill {
            return Err(RealForwardError::Unsupported(
                "TURBOSPARK_ROUTED_BATCH is not wired for this family: the batched \
                 routed pair exists in the gemma4 (INT4-affine) and gpt-oss (MXFP4) \
                 chunked drivers alone, and this driver keeps its routed half as a \
                 per-token loop"
                    .to_string(),
            ));
        }
        // The resident-GEMV seam, which unlike the one above is meaningful
        // on EVERY chunked driver: every family has resident GEMVs, and the
        // M-row GEMM is wired in the gemma4 and dense-qwen drivers alone
        // (step 6).
        if self.batched_gemv_prefill {
            return Err(RealForwardError::Unsupported(
                "TURBOSPARK_BATCHED_GEMV is not wired for this family: the M-row \
                 resident GEMM exists in the gemma4 and dense-qwen chunked drivers \
                 alone (INT4-affine), and this driver keeps every resident GEMV per \
                 token"
                    .to_string(),
            ));
        }
        let mut offset = 0usize;
        while offset < tokens.len() {
            let take = (tokens.len() - offset).min(MAX_PREFILL_BATCH);
            let last = offset + take == tokens.len();
            self.prefill_micro_batch_qwen4(
                &tokens[offset..offset + take],
                start_position + offset,
                last,
                logits,
            )?;
            offset += take;
        }
        Ok(())
    }

    fn prefill_micro_batch_qwen4(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        want_head: bool,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let result =
            self.prefill_micro_batch_qwen4_inner(tokens, start_position, want_head, logits);
        // One forward pass per TOKEN, not per micro-batch, matching every
        // other chunked driver's phase divisor (`crates/runtime/CLAUDE.md`
        // Gotcha 6).
        self.phases.calls += tokens.len() as u64;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn prefill_micro_batch_qwen4_inner(
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
        let heads_per_ngram = arch.ple.heads_per_ngram as usize;
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
        let banks = if self.routed_pipeline {
            routed_pipeline_banks(self.expert_cache_slots, top_k)
        } else {
            1
        };

        let embed_name = "language_model.model.embed_tokens.weight";
        let hc_count = self
            .real_qwen4
            .as_ref()
            .expect("real qwen4 state present")
            .hc_count;
        let wide_dim = hidden * hc_count;

        let mut pass = self.context.begin_pass_labeled("chunk cb1 (attn+router)");

        // Embed and TILE every token of the micro-batch into its own row of
        // `wide_x`: `embed_tokens(ids).repeat(1, 1, C)` per token, matching
        // the sequential flow's per-`c` tiling at that token's row offset.
        for (t, &token) in tokens.iter().enumerate() {
            let qwen4 = self.real_qwen4.as_ref().expect("checked above");
            let row_off = (t * wide_dim * 2) as u64;
            for c in 0..hc_count {
                encode_embed_any_qwen4(
                    &mut self.context,
                    &pass,
                    &self.weights,
                    &self.index,
                    embed_name,
                    (&qwen4.wide_x, row_off + (c * hidden * 2) as u64),
                    token as u32,
                    hidden as u32,
                )?;
            }
        }

        let mut pending_routed: Option<gpu::CommittedPass> = None;
        for layer in 0..arch.num_layers as usize {
            {
                // The attention-and-router half: every token of the
                // micro-batch, in order, into ONE evolving pass. QSA's own
                // internal above-budget mid-layer commit (`attn.rs`) may
                // swap `pass` out and back in transparently; nothing here
                // needs to know when that happens.
                let (context, weights, index, scratch, kv, qwen4, phases) = (
                    &mut self.context,
                    &self.weights,
                    &self.index,
                    &self.scratch,
                    &mut self.kv,
                    self.real_qwen4.as_mut().expect("checked above"),
                    &mut self.phases,
                );

                if layer == qwen4.ple_layer {
                    // Cheap Metal buffer handle retain, not a copy of the
                    // GPU contents (`crates/runtime/CLAUDE.md` Gotcha 5's
                    // shape) -- breaks the aliasing `encode_ple_layer`'s
                    // `qwen4: &mut` would otherwise create against `wide`.
                    let wide_buf = qwen4.wide_x.clone();
                    for (t, &token) in tokens.iter().enumerate() {
                        let x_off = (t * wide_dim * 2) as u64;
                        let emb_off = (t * hidden * 2) as u64;
                        encode_ple_layer(
                            context,
                            &pass,
                            weights,
                            index,
                            qwen4,
                            (&wide_buf, x_off),
                            hidden,
                            heads_per_ngram,
                            token,
                            emb_off,
                        )?;
                    }
                }

                for t in 0..m {
                    let x_off = (t * wide_dim * 2) as u64;
                    let hc_row = (t * hc_count * 2) as u64;
                    let position = start_position + t;

                    // attn_hc: mixed, raw (= wide_x's row t), inject_w (row t)
                    encode_hyper_connection(
                        context,
                        &pass,
                        weights,
                        index,
                        qwen4,
                        (&qwen4.wide_x, x_off),
                        &layer_tensor(layer, "attn_hyper_connection"),
                        hidden,
                        (&scratch.normed, 0),
                        true,
                        hc_row,
                    )?;

                    if arch.layer_is_linear(layer) {
                        encode_linear_block(
                            context,
                            &pass,
                            weights,
                            index,
                            &arch,
                            qwen4,
                            (&scratch.normed, 0),
                            scratch,
                            layer,
                        )?;
                    } else {
                        // `&mut pass`: above the QSA budget this commits and
                        // waits mid-layer for the indexer's score readback
                        // and hands back a fresh encoder (`attn.rs`'s own
                        // doc) -- unmodified from the sequential path.
                        encode_full_attention_block(
                            context,
                            &mut pass,
                            weights,
                            index,
                            &arch,
                            qwen4,
                            (&scratch.normed, 0),
                            scratch,
                            kv,
                            phases,
                            layer,
                            position,
                        )?;
                    }

                    gpu::encode_hc_inject_add(
                        context,
                        &pass,
                        (&qwen4.wide_x, x_off),
                        (&scratch.o, 0),
                        (&qwen4.hc_inject, hc_row),
                        hc_count as u32,
                        hidden as u32,
                    )
                    .map_err(gpu_err)?;

                    // mlp_hc: mixed -> qwen4.moe_x's row t (NOT
                    // scratch.normed -- see moe_x's own doc), raw (= updated
                    // wide_x's row t), inject_w -> hc_inject's row t.
                    let moe_x_off = (t * hidden * 2) as u64;
                    encode_hyper_connection(
                        context,
                        &pass,
                        weights,
                        index,
                        qwen4,
                        (&qwen4.wide_x, x_off),
                        &layer_tensor(layer, "mlp_hyper_connection"),
                        hidden,
                        (&qwen4.moe_x, moe_x_off),
                        true,
                        hc_row,
                    )?;

                    let logits_row_offset = (t * num_experts * 4) as u64;
                    moe::encode_moe_router(
                        context,
                        &pass,
                        weights,
                        index,
                        qwen4,
                        (&qwen4.moe_x, moe_x_off),
                        layer,
                        hidden,
                        num_experts,
                        logits_row_offset,
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
                    // is a correctness requirement here rather than a tidy-up
                    // (AGENTS.md Gotcha 64). `protect` names slots a command
                    // buffer STILL IN FLIGHT is reading; the `banks == 1`
                    // branch above has already called `retire_routed`, so the
                    // previous buffer is provably retired by the time this is
                    // consulted and the reservation is stale conservatism.
                    //
                    // Reserving anyway is not merely wasteful, it PANICS. At
                    // `banks == 1` the cache has `slots` places and the
                    // previous token holds `top_k` of them, leaving
                    // `slots - top_k` for a token that can miss on all
                    // `top_k`. This family routes top-10 and the bench pins
                    // 16 slots, so 6 places had to hold 10 misses and
                    // `ExpertCache::plan` asserted -- on the DEFAULT
                    // configuration, not an exotic one. Gemma 4 hides the
                    // same shape because top-8 of 16 leaves exactly 8.
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
                        qwen4,
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
                        self.real_qwen4.as_mut().expect("checked above"),
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
                    let moe_x_off = (t * hidden * 2) as u64;
                    let used = moe::encode_moe_layer(
                        context,
                        &routed_pass,
                        weights,
                        index,
                        scratch,
                        qwen4,
                        (&qwen4.moe_x, moe_x_off),
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

                    let x_off = (t * wide_dim * 2) as u64;
                    let hc_row = (t * hc_count * 2) as u64;
                    gpu::encode_hc_inject_add(
                        context,
                        &routed_pass,
                        (&qwen4.wide_x, x_off),
                        (&qwen4.h2, 0),
                        (&qwen4.hc_inject, hc_row),
                        hc_count as u32,
                        hidden as u32,
                    )
                    .map_err(gpu_err)?;
                    used
                };

                if banks > 1 {
                    self.retire_routed(&mut pending_routed);
                }
                debug_assert!(pending_routed.is_none(), "routed pipeline depth is 1");
                pending_routed = Some(routed_pass.commit());
                previous_slots = used.into_iter().collect();
            }

            pass = self.context.begin_pass_labeled("chunk cb1 (attn+router)");
        }
        self.retire_routed(&mut pending_routed);

        if want_head {
            pass.relabel("final cb (head)");
            let last_off = ((m - 1) * wide_dim * 2) as u64;
            let (context, weights, index, scratch, qwen4) = (
                &mut self.context,
                &self.weights,
                &self.index,
                &self.scratch,
                self.real_qwen4.as_ref().expect("checked above"),
            );
            // Exit: hyper_connection_mixer collapses the LAST token's wide
            // stream to `hidden`, INCLUDING its own hc_norm. No softcap
            // (`RealQwen4State::build` refuses an install that declares
            // one, matching the sequential flow).
            encode_hyper_connection(
                context,
                &pass,
                weights,
                index,
                qwen4,
                (&qwen4.wide_x, last_off),
                &format!("{TRUNK_PREFIX}.hyper_connection_mixer"),
                hidden,
                (&scratch.normed, 0),
                false,
                0,
            )?;
            let head_name = "language_model.lm_head.weight";
            encode_gemv_any(
                context,
                &pass,
                weights,
                index,
                head_name,
                vocab,
                hidden,
                (&scratch.normed, 0),
                (&scratch.logits, 0),
            )?;
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
}

/// A thin wrapper matching `encode_embed_any`'s shape at a fixed scale of
/// 1.0, matching `produce.rs`'s identical helper: `embedding_scaled_by_sqrt_hidden`
/// is refused true at `RealQwen4State::build`, so the constant is named here
/// rather than repeated at every `(t, c)` call site.
#[allow(clippy::too_many_arguments)]
fn encode_embed_any_qwen4(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &model_io::ResidentIndex,
    name: &str,
    out: (&gpu::MetalBuffer, u64),
    token: u32,
    hidden: u32,
) -> Result<(), RealForwardError> {
    encode_embed_any(context, pass, weights, index, name, out, token, hidden, 1.0)
}
