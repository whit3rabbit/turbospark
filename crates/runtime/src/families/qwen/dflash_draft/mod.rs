//! The DFlash2 draft pass: the context-KV writer, the one-pass block
//! forward, and the host-side candidate selector (`docs/DFLASH2.md`).
//!
//! A round is one command buffer for the context KV (its own pass; the
//! shape varies per round and it waits alone), one for the block forward
//! (embeddings, `layers` drafter layers, final norm, and TWO head GEMMs:
//! the trunk's `lm_head` for the selector's candidates and unary scores,
//! the drafter's `hidden_projection` for its edge scores), then a host
//! phase over two readbacks.
//!
//! The selector runs on the HOST at K=16, block <= 15: a top-16 SCAN (not
//! a sort -- this family's vocab is 248,320 and the 18.9 ms full sort in
//! `selection::select` is this repo's own lesson about that), ~17 codebook
//! row gathers and one bilinear score per step. Microseconds beside a pass
//! that reads a gigabyte of weights; vLLM needs a Triton walk kernel and a
//! radix top-k because it serves thousands of requests per step, and this
//! engine serves one.
//!
//! Precision note: every selector score is FP32 on the host over FP16
//! logits and BF16 codebooks. A drafter's arithmetic is a throughput axis,
//! never a correctness one -- the target verifies every proposal -- so no
//! bit-exactness claim crosses this file.

mod context_kv;
mod forward;
mod layer;
mod select;

pub(crate) use select::read_f16_rows;

use foundation::{LogitValue, TokenId};

use super::dflash::{DFLASH_MASK_TOKEN, DFLASH_RESIDUAL_SCALE};
use crate::real_forward::{RealForwardError, RealForwardRunner};

/// The drafter's RoPE: full-head NeoX at theta 1e7 (`rope_type: default` in
/// the published config), rotated_pairs = head_dim / 2.
pub(crate) const DFLASH_THETA: f32 = 1e7;

/// What the embedding rows and every `finish` conv output are multiplied by,
/// so the residual stream they build fits FP16. The reciprocal of
/// [`DFLASH_RESIDUAL_SCALE`], whose doc carries the measurement and the
/// argument that it cancels.
pub(crate) const RESIDUAL_RESCALE: f32 = 1.0 / DFLASH_RESIDUAL_SCALE;

/// Re-exported from [`crate::real_forward_utils`], where it moved when the
/// chunked-prefill driver's batched attention projections needed the same
/// split on Gemma 4's sliding-window rings. Its call sites here are
/// unchanged.
pub(crate) use crate::real_forward_utils::ring_spans;

impl RealForwardRunner {
    /// `SpeculativeProducer::prime_drafter` for the DFlash2 path: write the
    /// captured row for `position` into the drafter's cache.
    pub fn dflash_prime_from_capture(&mut self, position: usize) -> Result<(), RealForwardError> {
        let Some(d) = self.real_dflash.as_ref() else {
            return Err(RealForwardError::Unsupported(
                "no DFlash2 state; set MFERENCE_DFLASH_DRAFT before opening the model".to_string(),
            ));
        };
        if d.kv.position() != position {
            return Err(RealForwardError::Unsupported(format!(
                "DFlash2 prime at position {position} but its cache covers [0, {})",
                d.kv.position()
            )));
        }
        if d.capture_base != position || d.capture_rows < 1 {
            return Err(RealForwardError::Unsupported(format!(
                "DFlash2 prime at position {position} but the capture buffer holds {} rows at \
                 base {}",
                d.capture_rows, d.capture_base
            )));
        }
        self.dflash_context_write(1)
    }

    /// `SpeculativeProducer::rewind_drafter` for the DFlash2 path: a cursor
    /// move. Rows past the target are stale by construction and always
    /// rewritten before anything reads them again (the context write for
    /// committed positions, fresh query rows above the cursor).
    ///
    /// **A target AHEAD of the cursor is the normal case here, and that is
    /// the difference between a block drafter and a step-wise one.** The MTP
    /// head advances its cache as it drafts, so the loop's
    /// `rewind_drafter(base + accepted)` genuinely walks it BACK to the
    /// accepted end. This drafter's cursor is advanced only by
    /// `dflash_context_write`, which runs at the START of the next round --
    /// so between the two, the accepted prefix's slots still hold the
    /// DRAFT-written KV the block forward put there from mask embeddings,
    /// and the cursor legitimately lags. Advancing it here would claim
    /// those rows are target-derived when the pass that makes them so has
    /// not run.
    ///
    /// Refusing instead was unreachable for as long as acceptance was zero
    /// (`base + 0` is exactly the cursor), so this arm went unexercised
    /// until the FP16 overflow above it was fixed and the first round
    /// accepted all eight proposals.
    pub fn dflash_rewind_to(&mut self, position: usize) -> Result<(), RealForwardError> {
        let Some(d) = self.real_dflash.as_mut() else {
            return Err(RealForwardError::Unsupported(
                "no DFlash2 state to rewind".to_string(),
            ));
        };
        let cursor = d.kv.position();
        if position >= cursor {
            return Ok(());
        }
        if cursor - position > d.kv.max_safe_rewind() {
            return Err(RealForwardError::Unsupported(format!(
                "DFlash2 rewind of {} rows exceeds the ring's slack {}",
                cursor - position,
                d.kv.max_safe_rewind()
            )));
        }
        d.kv.rewind_by(cursor - position);
        Ok(())
    }

    /// One DFlash2 round: context-write the committed prefix, run the
    /// `block + 1` query rows through the drafter, walk the selector, and
    /// return `block` proposals in `proposals`.
    ///
    /// `base` is the position the anchor token occupies; the anchor is
    /// passed as a TOKEN because its embedding is the bonus row's input,
    /// matching the reference's `input_ids[query 0] = bonus_token`.
    pub fn dflash_draft_block(
        &mut self,
        anchor: TokenId,
        base: usize,
        proposals: &mut Vec<TokenId>,
    ) -> Result<(), RealForwardError> {
        let vocab = self.arch.vocab_size as usize;
        let Some(d) = self.real_dflash.as_ref() else {
            return Err(RealForwardError::Unsupported(
                "no DFlash2 state; set MFERENCE_DFLASH_DRAFT before opening the model".to_string(),
            ));
        };
        let block = d.block;
        proposals.clear();
        if anchor < 0 || anchor as usize >= vocab || DFLASH_MASK_TOKEN as usize >= vocab {
            return Err(RealForwardError::Unsupported(format!(
                "DFlash2 needs token ids {anchor} (anchor) and {DFLASH_MASK_TOKEN} (mask) inside \
                 vocab {vocab}"
            )));
        }

        gpu::autorelease_pool(|| {
            // The committed prefix's context KV, from the capture the
            // trunk's verify pass just filled. The write covers
            // `[capture_base, min(base, capture_end))`: EVERY accepted row
            // (their slots still hold draft-written KV from the previous
            // round's mask inputs; the reference overwrites each accepted
            // row with target-derived KV every round), capped at the
            // caller's base for the bisect's second alignment, which
            // drafts at the anchor's own position with the anchor's own
            // row still captured.
            let ctx_rows = {
                let d = self.real_dflash.as_ref().expect("checked above");
                let capture_end = d.capture_base + d.capture_rows;
                if d.kv.position() > base {
                    return Err(RealForwardError::Unsupported(format!(
                        "the drafter's cursor {} is past the round's base {base}",
                        d.kv.position()
                    )));
                }
                if base < d.capture_base {
                    return Err(RealForwardError::Unsupported(format!(
                        "the round's base {base} is behind the capture base {}",
                        d.capture_base
                    )));
                }
                base.min(capture_end) - d.capture_base
            };
            if ctx_rows > 0 {
                self.dflash_context_write(ctx_rows)?;
            }
            self.dflash_forward_block(anchor, base)
        })?;

        self.dflash_select(anchor, block, proposals)
    }

    /// Reads one row of the drafter's LAST head output, for the bisect
    /// probe (`docs/DFLASH2.md`): the rank of the true token in the
    /// drafter's own distribution is what separates a broken port from a
    /// weak drafter, and the selector's walk cannot be blamed until the
    /// unary itself is sane.
    pub fn dflash_probe_logits(
        &self,
        row: usize,
        out: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let Some(d) = self.real_dflash.as_ref() else {
            return Err(RealForwardError::Unsupported(
                "no DFlash2 state".to_string(),
            ));
        };
        let vocab = self.arch.vocab_size as usize;
        if out.len() != vocab {
            return Err(RealForwardError::Unsupported(format!(
                "probe buffer {} vs vocab {vocab}",
                out.len()
            )));
        }
        gpu::read_buffer_f16_into(&d.logits, row * vocab * 2, out);
        Ok(())
    }

    /// Every persistent draft buffer's `(name, nan count, min, max)` over
    /// its first `rows` rows, for localizing a non-finite draft pass.
    ///
    /// The per-layer buffers hold the LAST layer's values after a pass, so
    /// this says WHICH STAGE is non-finite and not which layer -- but the
    /// stage is the question when a whole pass reads NaN, and `capture`
    /// (written by the trunk, upstream of every line in this file) is in
    /// the list precisely so the drafter is not the first suspect.
    pub fn dflash_probe_buffers(&self) -> Vec<(&'static str, usize, f32, f32)> {
        let Some(d) = self.real_dflash.as_ref() else {
            return Vec::new();
        };
        let s = &d.shape;
        let rows = d.block + 1;
        let scan = |buf: &gpu::MetalBuffer, width: usize| {
            let v = read_f16_rows(buf, rows, width);
            let nans = v.iter().filter(|f| f.is_nan()).count();
            let fin = || v.iter().copied().filter(|f| f.is_finite());
            (
                nans,
                fin().fold(f32::INFINITY, f32::min),
                fin().fold(f32::NEG_INFINITY, f32::max),
            )
        };
        [
            ("capture", &d.capture, s.aux_count * s.hidden),
            ("ctx_combined", &d.ctx_combined, s.hidden),
            ("ctx_normed", &d.ctx_normed, s.hidden),
            ("x", &d.x, s.hidden),
            ("normed", &d.normed, s.hidden),
            ("coeffs", &d.coeffs, s.conv_rows),
            ("conv_p", &d.conv_p, s.hidden),
            ("q", &d.q, s.num_heads * s.head_dim),
            ("attn_out", &d.attn_out, s.num_heads * s.head_dim),
            ("sub_out", &d.sub_out, s.hidden),
            ("conv_f", &d.conv_f, s.hidden),
            ("ffn_gate", &d.ffn_gate, s.inter),
            ("ffn_up", &d.ffn_up, s.inter),
            ("ffn_act", &d.ffn_act, s.inter),
            ("hproj", &d.hproj, s.rank),
        ]
        .into_iter()
        .map(|(name, buf, width)| {
            let (nans, min, max) = scan(buf, width);
            (name, nans, min, max)
        })
        .collect()
    }
}
