//! Gemma 4 image spans in the chunked prefill (`docs/GEMMA4_VISION_PHASE0.md`).
//!
//! Two things an image changes in this trunk, and both are here:
//!
//! - **The rows.** At an image placeholder the tower's projected row IS the
//!   embedding. It is written straight into the residual, with NO
//!   `sqrt(hidden)` embed scale: the reference scatters the projector's
//!   output after the scaled lookup (Phase 0 trap 6).
//! - **The mask.** With `use_bidirectional_attention: "vision"`, a SLIDING
//!   layer lets every row of an image span attend to every other row of the
//!   same span, later ones included, inside the usual window counted from
//!   the row's own position. FULL layers stay causal. For a row at `p` in a
//!   span ending at `e`, the sliding keys are `p + 1 - window ..
//!   max(p + 1, e)`: one contiguous range, so the existing attention kernel
//!   serves it with a longer `seq_len` and an unchanged `kv_start`.
//!
//! # Why a span runs whole, and past the chunk's end
//!
//! Row `p`'s attention at layer `L` reads the keys of every later row of its
//! span at layer `L`, and those exist only once those rows have run layers
//! `0..L`. So a span cannot be split across micro-batches the way text is:
//! every layer runs over the WHOLE span before the next layer starts. The
//! span's rows live in their own residual buffer across layers, and each
//! layer runs in two phases: every row's q and K/V first, then attention and
//! the FFN. The FFN still runs 16 rows at a time, copied through
//! `scratch.x`, because the batched GEMMs and every per-row buffer the FFN
//! touches are sized for [`MAX_PREFILL_BATCH`].
//!
//! A span longer than what remains of the caller's chunk runs to its end
//! anyway. Its rows need no token ids (the map holds them), and the next
//! chunk skips the positions already in the cache.

use foundation::LogitValue;

use crate::moe_prefill_pipeline::routed_pipeline_banks;
use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::encode_gemv_any;
use crate::real_forward_types::{RealForwardError, MAX_PREFILL_BATCH};
use crate::real_forward_utils::norm_view;
use crate::resid_capture::encode_resid_capture;
use crate::steering::encode_steering;

const RMS_EPS: f32 = 1e-6;

impl RealForwardRunner {
    /// Why this runner cannot prefill an image span of `len` rows, if it
    /// cannot. Each arm is a path whose kernels or buffers assume the causal
    /// per-row order this file breaks.
    pub(crate) fn gemma4_span_refusal(&self, len: usize) -> Option<String> {
        if self.batched_gemv_prefill {
            return Some(
                "TURBOSPARK_BATCHED_GEMV cannot serve a Gemma 4 image span: its attention \
                 half norms each row's key just before that row attends, so later rows' keys \
                 are not ready. Unset it for image prompts"
                    .to_string(),
            );
        }
        if self.routed_batch_prefill {
            return Some(
                "TURBOSPARK_ROUTED_BATCH is not wired for Gemma 4 image spans; unset it for \
                 image prompts"
                    .to_string(),
            );
        }
        let layers = self.arch.num_layers as usize;
        if (0..layers).any(|l| self.kv.layer_quant(l).is_some()) {
            return Some(
                "--kv-bits is not supported with Gemma 4 image prompts: a span writes every \
                 row's keys before any row attends, and the quantized path stages one row"
                    .to_string(),
            );
        }
        let window = self.arch.sliding_window as usize;
        for layer in 0..layers {
            if self.arch.full_attention_layer_mask[layer] == 1 {
                continue;
            }
            let ring = self.kv.ring_capacity(layer);
            let need = window + len - 1;
            if ring > 0 && ring < need {
                return Some(format!(
                    "a {len}-token image span needs {need} sliding-window KV slots (window \
                     {window} plus the span's look-ahead), but layer {layer}'s ring holds \
                     {ring}; the ring is sized window + prefill chunk, so open with a prefill \
                     chunk of at least {len}"
                ));
            }
        }
        None
    }

    fn gemma4_span_attention_range(
        &self,
        layer: usize,
        position: usize,
        span_end: usize,
    ) -> (u32, u32) {
        span_attention_range(
            self.arch.full_attention_layer_mask[layer] == 1,
            self.arch.sliding_window as usize,
            position,
            span_end,
        )
    }

    /// Run the image span `[start, start + len)` through every layer and
    /// advance the cache past it.
    ///
    /// `head_row`, when set, writes the logits of that row of the span into
    /// `logits` (the caller's chunk ended inside or at the end of the span).
    pub(crate) fn prefill_image_span_gemma4(
        &mut self,
        start: usize,
        len: usize,
        head_row: Option<usize>,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        if start != self.kv.position() {
            return Err(RealForwardError::Unsupported(format!(
                "image span at {start} does not start at the cache position {}",
                self.kv.position()
            )));
        }
        if let Some(reason) = self.gemma4_span_refusal(len) {
            return Err(RealForwardError::Unsupported(reason));
        }
        let arch = self.arch.clone();
        let hidden = arch.hidden_size as usize;
        let inter = arch.intermediate_size as usize;
        let use_silu = arch.hidden_activation.contains("silu");
        let q_dim = (arch.num_heads * arch.head_dim.max(arch.full_head_dim)) as usize;
        let rows = self
            .prompt_vision
            .as_ref()
            .and_then(|pv| pv.span_rows(start))
            .ok_or_else(|| {
                RealForwardError::Unsupported(format!(
                    "no image rows are set for a span at {start}"
                ))
            })?;
        if rows.len() != len * hidden * 2 {
            return Err(RealForwardError::Unsupported(format!(
                "image span at {start}: {} bytes of rows for {len} x {hidden}",
                rows.len()
            )));
        }
        let started = std::time::Instant::now();
        let span_x = self.context.new_buffer_with_data(rows);
        let span_q = self.context.new_output_buffer((len * q_dim * 2) as u64);
        let x = self.scratch.x.clone();
        let dense = self.real.as_ref().expect("real state present").dense;
        let base = self.index.header.index_size;
        let span_end = start + len;
        let groups: Vec<(usize, usize)> = (0..len)
            .step_by(MAX_PREFILL_BATCH)
            .map(|g| (g, (len - g).min(MAX_PREFILL_BATCH)))
            .collect();
        let row = |r: usize| (r * hidden * 2) as u64;
        let q_row = |r: usize| (r * q_dim * 2) as u64;
        let banks = if !dense && self.routed_pipeline {
            routed_pipeline_banks(self.expert_cache_slots, arch.top_k_experts as usize)
        } else {
            1
        };
        let gpu_err = RealForwardError::Gpu;
        let mut pending_routed: Option<gpu::CommittedPass> = None;

        let mut pass = self.context.begin_pass_labeled("image span");
        for layer in 0..arch.num_layers as usize {
            // Phase A: every row's q and K/V, so phase B's look-ahead reads
            // keys that exist.
            for &(g, n) in &groups {
                gpu::encode_dflash_copy_rows(
                    &mut self.context,
                    &pass,
                    (&span_x, row(g)),
                    (&x, 0),
                    n as u32,
                    hidden as u32,
                    hidden as u32,
                )
                .map_err(gpu_err)?;
                for i in 0..n {
                    self.encode_gemma4_layer_qkv(
                        &pass,
                        layer,
                        start + g + i,
                        i,
                        (&span_q, q_row(g + i)),
                    )?;
                }
            }
            // Phase B: attention, the residual, then the FFN, 16 rows at a time.
            for &(g, n) in &groups {
                gpu::encode_dflash_copy_rows(
                    &mut self.context,
                    &pass,
                    (&span_x, row(g)),
                    (&x, 0),
                    n as u32,
                    hidden as u32,
                    hidden as u32,
                )
                .map_err(gpu_err)?;
                for i in 0..n {
                    let position = start + g + i;
                    let (seq_len, kv_start) =
                        self.gemma4_span_attention_range(layer, position, span_end);
                    self.encode_gemma4_layer_attend_and_router(
                        &pass,
                        layer,
                        position,
                        seq_len,
                        kv_start,
                        base,
                        i,
                        (&span_q, q_row(g + i)),
                    )?;
                }
                if dense {
                    self.encode_gemma4_dense_ffn_batched(&pass, layer, hidden, inter, use_silu, n)?;
                } else {
                    // The routing reads the router logits on the host.
                    pass.commit_and_wait();
                    self.retire_routed(&mut pending_routed);
                    self.encode_gemma4_routed_half_per_token(layer, n, banks, &mut pending_routed)?;
                    // Queue order puts this pass after the routed passes.
                    pass = self.context.begin_pass_labeled("image span");
                }
                for i in 0..n {
                    // The routed path applies these per token itself.
                    if dense {
                        encode_steering(
                            &mut self.context,
                            &pass,
                            &self.scratch,
                            self.steering.as_ref(),
                            layer,
                            hidden,
                            1,
                            row(i),
                        )?;
                        encode_resid_capture(
                            &mut self.context,
                            &pass,
                            &self.scratch,
                            self.resid_capture.as_ref(),
                            layer,
                            hidden,
                            row(i),
                        )?;
                    }
                }
                gpu::encode_dflash_copy_rows(
                    &mut self.context,
                    &pass,
                    (&x, 0),
                    (&span_x, row(g)),
                    n as u32,
                    hidden as u32,
                    hidden as u32,
                )
                .map_err(gpu_err)?;
            }
        }
        self.retire_routed(&mut pending_routed);

        if let Some(r) = head_row {
            if r >= len {
                return Err(RealForwardError::Unsupported(format!(
                    "head row {r} is outside a {len}-row span"
                )));
            }
            self.encode_gemma4_head_at(&pass, (&span_x, row(r)))?;
        }
        pass.commit_and_wait();
        self.kv.advance_by(len);
        self.phases.calls += len as u64;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        if let Some(capture) = self.resid_capture.as_mut() {
            capture.record_pass(
                span_end - 1,
                head_row.is_none() || !self.prefill_chunk_is_final,
            );
        }
        if head_row.is_some() {
            let vocab = arch.vocab_size as usize;
            if vocab != logits.len() {
                return Err(RealForwardError::Unsupported(format!(
                    "vocab mismatch: model has {vocab}, caller expected {}",
                    logits.len()
                )));
            }
            gpu::read_buffer_f16_into(&self.scratch.logits, 0, logits);
        }
        Ok(())
    }

    /// The final norm, the head and the softcap on one residual row, into
    /// `scratch.logits`. The same three dispatches the micro-batch drivers
    /// end with.
    fn encode_gemma4_head_at(
        &mut self,
        pass: &gpu::PassEncoder,
        x: (&gpu::MetalBuffer, u64),
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let hidden = arch.hidden_size as usize;
        let vocab = arch.vocab_size as usize;
        let final_norm = norm_view(
            &self.weights,
            &self.index,
            "language_model.model.norm.weight",
            hidden,
        )?;
        gpu::encode_rms_norm_bf16w(
            &mut self.context,
            pass,
            x,
            final_norm,
            (&self.scratch.normed, 0),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(RealForwardError::Gpu)?;
        let head_name = if arch.tie_word_embeddings {
            "language_model.model.embed_tokens.weight".to_string()
        } else {
            "language_model.lm_head.weight".to_string()
        };
        encode_gemv_any(
            &mut self.context,
            pass,
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
                pass,
                (&self.scratch.logits, 0),
                arch.final_logit_softcap as f32,
                vocab as u32,
            )
            .map_err(RealForwardError::Gpu)?;
        }
        Ok(())
    }

    /// A prefill chunk on a runner holding an image map: text runs go
    /// through the ordinary micro-batch drivers, each image span through
    /// [`Self::prefill_image_span_gemma4`], and positions a span already ran
    /// past the previous chunk's end are skipped.
    pub(crate) fn prefill_chunk_gemma4_with_images(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let end = start_position + tokens.len();
        let image_token = self.arch.vision.image_token_id as i32;
        let token_at = |p: usize| tokens[p - start_position];
        let span_at = |runner: &Self, p: usize| {
            runner
                .prompt_vision
                .as_ref()
                .and_then(|pv| pv.span_containing(p))
        };
        let mut p = start_position;

        // Positions an earlier span already put in the cache.
        if p < self.kv.position() {
            let done = self.kv.position().min(end);
            for q in p..done {
                if span_at(self, q).is_none() || token_at(q) != image_token {
                    return Err(RealForwardError::Unsupported(format!(
                        "chunk at {start_position} restarts at {q}, before the cache position \
                         {}, outside an image span",
                        self.kv.position()
                    )));
                }
            }
            p = done;
            if p == end {
                if self.prefill_chunk_is_final {
                    return Err(RealForwardError::Unsupported(
                        "a prompt cannot end on an image placeholder".to_string(),
                    ));
                }
                return Ok(());
            }
        }

        while p < end {
            if let Some(span) = span_at(self, p) {
                if span.start != p {
                    return Err(RealForwardError::Unsupported(format!(
                        "position {p} is inside the image span at {} but the span did not run",
                        span.start
                    )));
                }
                let span_end = span.start + span.len;
                if let Some(q) = (p..span_end.min(end)).find(|&q| token_at(q) != image_token) {
                    return Err(RealForwardError::Unsupported(format!(
                        "position {q} is in an image span but holds token {}, not the image \
                         placeholder {image_token}",
                        token_at(q)
                    )));
                }
                let head_row = (span_end >= end).then(|| end - 1 - span.start);
                self.prefill_image_span_gemma4(span.start, span.len, head_row, logits)?;
                p = span_end.min(end);
                continue;
            }
            let next_span = self
                .prompt_vision
                .as_ref()
                .and_then(|pv| pv.spans().iter().find(|s| s.start > p).map(|s| s.start))
                .unwrap_or(end)
                .min(end);
            let dense = self.real.as_ref().expect("real state present").dense;
            let mut q = p;
            while q < next_span {
                let take = (next_span - q).min(MAX_PREFILL_BATCH);
                let last = q + take == end;
                let run = &tokens[q - start_position..q - start_position + take];
                if dense {
                    self.prefill_micro_batch_gemma4_dense(run, q, last, logits)?;
                } else {
                    self.prefill_micro_batch_gemma4(run, q, last, logits)?;
                }
                q += take;
            }
            p = next_span;
        }
        Ok(())
    }
}

/// The attention range `(seq_len, kv_start)` of the image row at `position`
/// in a span ending at `span_end`.
///
/// Full layers stay causal (`create_masks_for_vision_model`: "global
/// layers: causal only"). Sliding layers reach to the span's end, with the
/// window still counted from the row's own position.
fn span_attention_range(
    is_full: bool,
    window: usize,
    position: usize,
    span_end: usize,
) -> (u32, u32) {
    let causal_end = (position + 1) as u32;
    if is_full {
        return (causal_end, 0);
    }
    let kv_start = causal_end.saturating_sub(window as u32);
    ((span_end as u32).max(causal_end), kv_start)
}

#[cfg(test)]
mod tests {
    use super::span_attention_range;

    #[test]
    fn full_layers_stay_causal_inside_a_span() {
        assert_eq!(span_attention_range(true, 8, 10, 20), (11, 0));
    }

    #[test]
    fn sliding_layers_reach_the_span_end_from_their_own_window() {
        // Row 10 of a span ending at 20, window 8: keys 3..20.
        assert_eq!(span_attention_range(false, 8, 10, 20), (20, 3));
        // The last row's range is exactly the causal one.
        assert_eq!(span_attention_range(false, 8, 19, 20), (20, 12));
        // Near the start the window floor is zero.
        assert_eq!(span_attention_range(false, 1024, 3, 9), (9, 0));
    }
}
