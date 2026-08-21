//! The M-row trunk forward: `docs/MTP_SPECULATIVE.md` step 4's verify pass.
//!
//! This is the FIRST batched math path on a decode flow in this repo, and
//! the distinction matters because one already exists that is not one:
//! `prefill_chunk_real_gemma4` batches COMMAND BUFFERS and still calls
//! `encode_attention_decode` once per token, which is why its win is the
//! per-layer blocking wait rather than the kernels. Here the GEMVs really do
//! become GEMMs.
//!
//! **WHAT IS BATCHED AND WHAT IS NOT COMES FROM THE MEASURED COMPUTE SPLIT**
//! (`docs/MTP_SPECULATIVE.md`: GEMV 93.1%, norms and elementwise 4.1%, GDN
//! recurrence 2.2%, attention 0.6%, un-amortizable floor 6.4%). Every GEMV is
//! batched; norms, RoPE, attention and the recurrent step loop per token.
//! That is not a first-cut simplification to be tightened later -- it is what
//! the composite those numbers feed already assumes, and widening attention
//! at decode context would buy 0.6% of a pass.
//!
//! Three refusals, all BY NAME rather than by falling back:
//!
//! 1. **Dense only.** A batched routed-expert pair does not exist, and the
//!    union of M tokens' experts is larger than what the slot cache already
//!    loads (AGENTS.md Gotcha 54), so there is nothing to fall back TO.
//! 2. **INT4 only**, enforced one level down in `encode_gemm_any`. The 1-bit
//!    and 2-bit checkpoints of this same architecture have no batched kernel.
//! 3. **No KV wrap.** M consecutive positions occupy M ADJACENT slots only
//!    while `position % capacity` does not roll over inside the block.
//!
//! The first two would otherwise be silent: a sequential fallback is
//! numerically identical, so it would pass every losslessness test while
//! making the measurement describe the wrong engine.

use foundation::LogitValue;

use super::batched_layers::{
    encode_dense_ffn_batched, encode_full_attention_block_batched, encode_linear_block_batched,
};
pub(crate) use super::BatchedScratch;
use crate::families::qwen::{layer_tensor, RMS_EPS};
use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::{encode_embed_any, encode_gemm_any};
use crate::real_forward_types::RealForwardError;
use crate::real_forward_utils::norm_view;

impl RealForwardRunner {
    /// Runs `tokens` through the trunk in ONE pass, writing `tokens.len() *
    /// vocab` logits.
    ///
    /// Token `m` occupies `start_position + m` and attends over
    /// `[0, start_position + m]`, so the causal structure is identical to
    /// running the same tokens sequentially -- and that is not arranged here,
    /// it falls out of `encode_attention_decode` taking its span from the
    /// position ARGUMENT rather than from the cache cursor. All M KV rows are
    /// written before any query reads, and a query simply does not look at
    /// the rows above its own span.
    ///
    /// The KV cursor advances by `tokens.len()`, exactly as the same tokens
    /// run one at a time would leave it, so `rollback` needs no batched
    /// variant.
    pub fn produce_batched(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let hidden = arch.hidden_size as usize;
        let inter = arch.intermediate_size as usize;
        let vocab = arch.vocab_size as usize;
        let use_silu = arch.hidden_activation.contains("silu");
        let batch = tokens.len();
        let gpu_err = RealForwardError::Gpu;

        if arch.num_experts != 0 {
            return Err(RealForwardError::Unsupported(
                "batched forward is dense-only: the routed-expert pair has no batched \
                 kernel, and the union of M tokens' experts exceeds what the slot cache \
                 already loads (AGENTS.md Gotcha 54), so there is nothing to batch"
                    .to_string(),
            ));
        }
        if batch == 0 {
            return Err(RealForwardError::Unsupported(
                "batched forward needs at least one token".to_string(),
            ));
        }
        // A DFlash2 install carries no MTP head but needs the SAME verify
        // pass, so its state owns a BatchedScratch of its own and either
        // drafter's scratch serves.
        let batched =
            match (&self.real_mtp, &self.real_dflash) {
                (Some(m), _) => &m.batched,
                (None, Some(d)) => &d.batched,
                (None, None) => return Err(RealForwardError::Unsupported(
                    "no batched scratch; set MFERENCE_MTP_DRAFT or MFERENCE_DFLASH_DRAFT before \
                     opening the model"
                        .to_string(),
                )),
            };
        if batch > batched.batch {
            return Err(RealForwardError::Unsupported(format!(
                "batched forward of {batch} rows against scratch sized for {}",
                batched.batch
            )));
        }
        if start_position != self.kv.position() {
            return Err(RealForwardError::Unsupported(format!(
                "non-sequential batched start {start_position}; KV cache is at {}",
                self.kv.position()
            )));
        }
        if logits.len() != batch * vocab {
            return Err(RealForwardError::Unsupported(format!(
                "batched logits buffer is {}, expected {batch} x {vocab}",
                logits.len()
            )));
        }
        for &t in tokens {
            if (t as usize) >= vocab {
                return Err(RealForwardError::Unsupported(format!(
                    "token id {t} outside vocab {vocab}"
                )));
            }
        }
        // M consecutive positions are M ADJACENT slots only while
        // `position % capacity` does not roll over inside the block. This
        // family has no sliding window so its full layers are linear, but
        // "linear" still wraps at `max_context`; a batched projection writing
        // across that boundary would scatter into row 0 with no symptom
        // beyond wrong attention.
        for layer in 0..arch.num_layers as usize {
            if arch.layer_is_linear(layer) {
                continue;
            }
            let capacity = self.kv.capacity(layer);
            if start_position % capacity + batch > capacity {
                return Err(RealForwardError::Unsupported(format!(
                    "batched forward of {batch} rows at position {start_position} wraps \
                     layer {layer}'s KV capacity {capacity}; the batched projection writes \
                     M adjacent slots and cannot straddle the boundary"
                )));
            }
        }

        let embed_name = "language_model.model.embed_tokens.weight";
        let pass = self.context.begin_pass_labeled("batched verify");
        let (context, weights, index, scratch, qwen, kv, dflash) = (
            &mut self.context,
            &self.weights,
            &self.index,
            &self.scratch,
            self.real_qwen
                .as_ref()
                .ok_or_else(|| RealForwardError::Unsupported("not a Qwen install".to_string()))?,
            &mut self.kv,
            self.real_dflash.as_ref(),
        );

        // An embedding lookup has no trip count to amortize, so it loops for
        // the same reason the norms below do.
        for (m, &token) in tokens.iter().enumerate() {
            encode_embed_any(
                context,
                &pass,
                weights,
                index,
                embed_name,
                (&scratch.x, (m * hidden) as u64 * 2),
                token as u32,
                hidden as u32,
                1.0,
            )?;
        }

        for layer in 0..arch.num_layers as usize {
            let input_norm = norm_view(
                weights,
                index,
                &layer_tensor(layer, "input_layernorm.weight"),
                hidden,
            )?;
            for m in 0..batch {
                gpu::encode_rms_norm_bf16w(
                    context,
                    &pass,
                    (&scratch.x, (m * hidden) as u64 * 2),
                    input_norm,
                    (&batched.normed, (m * hidden) as u64 * 2),
                    hidden as u32,
                    RMS_EPS,
                )
                .map_err(gpu_err)?;
            }

            if arch.layer_is_linear(layer) {
                encode_linear_block_batched(
                    context, &pass, weights, index, &arch, qwen, batched, layer, batch,
                )?;
            } else {
                encode_full_attention_block_batched(
                    context,
                    &pass,
                    weights,
                    index,
                    &arch,
                    qwen,
                    scratch,
                    batched,
                    kv,
                    layer,
                    start_position,
                    batch,
                )?;
            }

            let post_attn = norm_view(
                weights,
                index,
                &layer_tensor(layer, "post_attention_layernorm.weight"),
                hidden,
            )?;
            for m in 0..batch {
                let row = (m * hidden) as u64 * 2;
                // RAW residual add: this family has no sandwich norms, and
                // normalizing here took the Qwen 3.6 reference perplexity
                // from 6.25 to 255,409 once already (crate Gotcha 11).
                gpu::encode_residual_add(
                    context,
                    &pass,
                    (&scratch.x, row),
                    (&batched.o, row),
                    hidden as u32,
                )
                .map_err(gpu_err)?;
                gpu::encode_rms_norm_bf16w(
                    context,
                    &pass,
                    (&scratch.x, row),
                    post_attn,
                    (&batched.moe_x, row),
                    hidden as u32,
                    RMS_EPS,
                )
                .map_err(gpu_err)?;
            }

            encode_dense_ffn_batched(
                context, &pass, weights, index, scratch, batched, layer, hidden, inter, use_silu,
                batch,
            )?;

            // THE DFLASH2 AUX CAPTURE at M rows: the residual rows this
            // layer just produced, into the fc input's layout. Same point
            // as the per-token hook (`families/qwen/mod.rs`), same zero
            // dispatches when no drafter is open.
            if let Some(d) = dflash {
                if let Some(aux) = d.aux_slot(layer) {
                    gpu::encode_dflash_copy_rows(
                        context,
                        &pass,
                        (&scratch.x, 0),
                        (&d.capture, (aux * hidden) as u64 * 2),
                        batch as u32,
                        hidden as u32,
                        (d.shape.aux_count * hidden) as u32,
                    )
                    .map_err(gpu_err)?;
                }
            }
        }

        let final_norm = norm_view(weights, index, "language_model.model.norm.weight", hidden)?;
        for m in 0..batch {
            gpu::encode_rms_norm_bf16w(
                context,
                &pass,
                (&scratch.x, (m * hidden) as u64 * 2),
                final_norm,
                (&batched.normed, (m * hidden) as u64 * 2),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
        }
        let head_name = if arch.tie_word_embeddings {
            embed_name.to_string()
        } else {
            "language_model.lm_head.weight".to_string()
        };
        encode_gemm_any(
            context,
            &pass,
            weights,
            index,
            &head_name,
            vocab,
            hidden,
            (&batched.normed, 0),
            (&batched.logits, 0),
            batch,
        )?;

        pass.commit_and_wait();
        for _ in 0..batch {
            kv.advance();
        }
        // THE LAST ROW'S RESIDUAL HAS TO LAND AT ROW 0, because that is where
        // a SEQUENTIAL run of the same tokens leaves it and because
        // `mtp_draft_step` reads `h_t` from `scratch.x` at offset 0.
        //
        // Without this the head drafts off the FIRST token of the block
        // instead of the last, and the failure is invisible in every place
        // one would look: the trunk is untouched, so the committed stream
        // stays byte-identical to a non-speculative run and the losslessness
        // gate passes. What moves is the DRAFTER's quality -- measured on the
        // real install, accept length fell from 1.84 to 1.10 per round and
        // rollbacks went from 0 to 62 of 123 rounds, which reads as a verdict
        // about MTP rather than as a bug in the pass.
        //
        // A host copy rather than a blit: `commit_and_wait` has already run,
        // it is `hidden` halfs (10 KiB here), and it needs no kernel.
        if batch > 1 {
            let row = hidden * 2;
            let last = gpu::read_buffer_bytes(&scratch.x, (batch - 1) * row, row);
            gpu::write_buffer_bytes(&scratch.x, 0, &last);
        }
        gpu::read_buffer_f16_into(&batched.logits, 0, logits);
        // The dflash capture's bookkeeping, last so it cannot alias the
        // scratch borrow above: `batch` rows whose positions start at
        // `start_position`, ready for the next round's context write over
        // the accepted prefix.
        if let Some(d) = self.real_dflash.as_mut() {
            d.note_capture(start_position, batch);
        }
        Ok(())
    }
}
