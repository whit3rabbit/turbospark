//! Forward pass token decode for `qwen4_exp`. The layer loop is
//! `mod.rs`'s "## decoder layer" pseudocode, encoded directly: PLE at its
//! one layer, two hyper-connection calls per layer (each read/inject
//! replacing a plain residual add), a mid-layer commit ONLY for the MoE
//! router readback (matching `families/qwen/produce.rs`'s shape), plus, on a
//! QSA layer ABOVE the indexer budget, a second commit inside
//! `encode_full_attention_block` for the block-score readback (GDN and
//! below-budget QSA are host-readback-free).

use std::time::Instant;

use foundation::LogitValue;

use super::attn::{encode_full_attention_block, encode_linear_block};
use super::hc::encode_hyper_connection;
use super::moe;
use super::ple::encode_ple_layer;
use super::{layer_tensor, TRUNK_PREFIX};
use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::encode_gemv_any;
use crate::real_forward_types::RealForwardError;

impl RealForwardRunner {
    pub(crate) fn produce_real_qwen4(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let result = self.produce_real_qwen4_inner(token, position, logits);
        self.phases.calls += 1;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn produce_real_qwen4_inner(
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
        let heads_per_ngram = arch.ple.heads_per_ngram as usize;
        let use_silu = arch.hidden_activation.contains("silu");
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

        let embed_name = "language_model.model.embed_tokens.weight";
        let hc_count = self
            .real_qwen4
            .as_ref()
            .expect("real qwen4 state present")
            .hc_count;

        let mut pass = self.context.begin_pass_labeled("cb1 (attn+router)");

        // Embed and TILE: `embed_tokens(ids).repeat(1, 1, C)` -- replicated,
        // not padded. `arch.embedding_scaled_by_sqrt_hidden` is refused
        // true at `RealQwen4State::build`, so the scale is always 1.0.
        for c in 0..hc_count {
            let qwen4 = self.real_qwen4.as_ref().expect("checked above");
            encode_embed_any_qwen4(
                &mut self.context,
                &pass,
                &self.weights,
                &self.index,
                embed_name,
                (&qwen4.wide_x, (c * hidden * 2) as u64),
                token as u32,
                hidden as u32,
                vocab,
            )?;
        }

        let (
            context,
            weights,
            index,
            scratch,
            kv,
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
            &mut self.kv,
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

        for layer in 0..arch.num_layers as usize {
            if layer == qwen4.ple_layer {
                // `encode_ple_layer` takes `qwen4: &mut` (it advances
                // `ngram_context`) alongside a `wide` buffer reference
                // derived from `qwen4.wide_x` -- an owned clone (a cheap
                // Metal buffer handle retain, not a copy of the GPU
                // contents) breaks the aliasing rather than reordering
                // around it (`crates/runtime/CLAUDE.md` Gotcha 5's shape).
                let wide_buf = qwen4.wide_x.clone();
                encode_ple_layer(
                    context,
                    &pass,
                    weights,
                    index,
                    qwen4,
                    (&wide_buf, 0),
                    hidden,
                    heads_per_ngram,
                    token,
                    0,
                )?;
            }

            // attn_hc: mixed, raw (= wide_x itself), inject_w
            encode_hyper_connection(
                context,
                &pass,
                weights,
                index,
                qwen4,
                (&qwen4.wide_x, 0),
                &layer_tensor(layer, "attn_hyper_connection"),
                hidden,
                (&scratch.normed, 0),
                true,
                0,
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
                // `&mut pass`: above the QSA budget this commits and waits
                // mid-layer for the indexer's score readback and hands back
                // a fresh encoder (`attn.rs`'s own doc).
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
                (&qwen4.wide_x, 0),
                (&scratch.o, 0),
                (&qwen4.hc_inject, 0),
                hc_count as u32,
                hidden as u32,
            )
            .map_err(gpu_err)?;

            // mlp_hc: mixed, raw (= updated wide_x), inject_w
            encode_hyper_connection(
                context,
                &pass,
                weights,
                index,
                qwen4,
                (&qwen4.wide_x, 0),
                &layer_tensor(layer, "mlp_hyper_connection"),
                hidden,
                (&scratch.normed, 0),
                true,
                0,
            )?;

            moe::encode_moe_router(
                context,
                &pass,
                weights,
                index,
                qwen4,
                (&scratch.normed, 0),
                layer,
                hidden,
                num_experts,
                0,
            )?;

            let t_wait = Instant::now();
            phases.cb1_gpu_nanos += (pass.commit().wait_with_gpu_time() * 1e9) as u64;
            phases.gpu_wait_nanos += t_wait.elapsed().as_nanos() as u64;

            pass = context.begin_pass_labeled("routed cb");
            moe::encode_moe_layer(
                context,
                &pass,
                weights,
                index,
                scratch,
                qwen4,
                (&scratch.normed, 0),
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
                &moe::RoutedSlot::sequential(),
            )?;

            gpu::encode_hc_inject_add(
                context,
                &pass,
                (&qwen4.wide_x, 0),
                (&qwen4.h2, 0),
                (&qwen4.hc_inject, 0),
                hc_count as u32,
                hidden as u32,
            )
            .map_err(gpu_err)?;

            // The "routed cb" pass carries over as the NEXT layer's
            // "cb1" implicitly -- no commit and no new pass here. It is
            // only committed again at the next router GEMV's readback
            // point (or, on the last layer, at the final head below).
            // Reassigning `pass` to a fresh `PassEncoder` here without
            // committing this one first would DROP it uncommitted: a
            // `PassEncoder` that is dropped rather than `.commit()`'d
            // ends its Metal encoder but never submits the command
            // buffer, silently discarding every dispatch encoded onto
            // it (`crates/gpu/CLAUDE.md` Gotcha 6).
        }

        // Exit: hyper_connection_mixer collapses the wide stream to
        // `hidden`, INCLUDING its own hc_norm (no separate final norm
        // tensor exists). No softcap.
        if !self.skip_head {
            pass.relabel("final cb (head)");
            encode_hyper_connection(
                context,
                &pass,
                weights,
                index,
                qwen4,
                (&qwen4.wide_x, 0),
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
            pass.relabel("final cb (prefill, no head)");
        }
        let t_wait = Instant::now();
        phases.final_cb_gpu_nanos += (pass.commit_and_wait_with_gpu_time() * 1e9) as u64;
        phases.final_wait_nanos += t_wait.elapsed().as_nanos() as u64;
        self.kv.advance();

        if self.skip_head {
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
/// 1.0 -- every other family's callers pass a scale they compute; this one
/// never does (`embedding_scaled_by_sqrt_hidden` is refused true at
/// `RealQwen4State::build`), so the constant is named here rather than
/// repeated at all `hc_count` call sites.
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
    vocab: usize,
) -> Result<(), RealForwardError> {
    crate::real_forward_dispatch::encode_embed_any(
        context, pass, weights, index, name, out, token, hidden, vocab, 1.0,
    )
}
