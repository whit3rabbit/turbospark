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

use foundation::{LogitValue, TokenId};
use model_io::ResidentIndex;

use super::dflash::{
    DFLASH_MASK_TOKEN, DFLASH_RESIDUAL_SCALE, DFLASH_TOP_K, DFLASH_WINDOW,
};
use super::RMS_EPS;
use crate::real_forward::{RealForwardError, RealForwardRunner};
use crate::real_forward_dispatch::{encode_embed_any, encode_gemm_any};
use crate::real_forward_utils::{entry, norm_view};

/// The drafter's RoPE: full-head NeoX at theta 1e7 (`rope_type: default` in
/// the published config), rotated_pairs = head_dim / 2.
const DFLASH_THETA: f32 = 1e7;

/// What the embedding rows and every `finish` conv output are multiplied by,
/// so the residual stream they build fits FP16. The reciprocal of
/// [`DFLASH_RESIDUAL_SCALE`], whose doc carries the measurement and the
/// argument that it cancels.
const RESIDUAL_RESCALE: f32 = 1.0 / DFLASH_RESIDUAL_SCALE;

/// How a `rows`-row write starting at `base` splits across a RING cache's
/// wrap: `[(row offset within the write, row count); 2]`, the second span
/// empty whenever the write does not straddle.
///
/// `KvCacheManager::k_slot` addresses `position % capacity` and validates
/// ONE row, while both writers below hand a batched projection `rows`
/// ADJACENT slots -- so a straddling write runs past the layer's buffer
/// with no assertion in the way. `produce_batched` REFUSES that case on the
/// trunk, whose full layers only wrap at `max_context`; the drafter's cache
/// is a real ring of `DFLASH_WINDOW + DFLASH_RING_SLACK`, so it wraps every
/// 2,176 positions and refusing would end an ordinary generation. Splitting
/// costs one extra dispatch per projection on the one round that straddles
/// and is a no-op on every other round (`spans[1].1 == 0`, and `spans[0]`
/// is exactly the single call that used to be made).
fn ring_spans(capacity: usize, base: usize, rows: usize) -> [(usize, usize); 2] {
    let first = rows.min(capacity - base % capacity);
    [(0, first), (first, rows - first)]
}

impl RealForwardRunner {
    /// The context-KV write: capture rows `[0, rows)` become target-derived
    /// KV at positions `[capture_base, capture_base + rows)`, and the
    /// cursor advances to `capture_base + rows`.
    ///
    /// The rows are written from CAPTURE_BASE and not from the cursor
    /// because the write must cover the whole span the last verify
    /// accepted, part of which sits BEHIND the post-rewind cursor: the
    /// accepted proposals' slots still hold DRAFT-written KV (mask-token
    /// inputs), and the reference overwrites every accepted row with
    /// target-derived KV each round. Rows behind the cursor that were
    /// already correct are rewritten with identical values, which is
    /// idempotent by determinism.
    ///
    /// The recipe is the reference's `precompute_and_store_context_kv`,
    /// split into the per-layer projections this engine already dispatches
    /// rather than one fused GEMM: fc, `hidden_norm`, then per layer
    /// `k_proj`/`v_proj` straight into the slots (batched: the slot stride
    /// IS the kernel's row stride, asserted at build), per-head `k_norm` in
    /// place, RoPE in place at each row's own position.
    fn dflash_context_write(&mut self, rows: usize) -> Result<(), RealForwardError> {
        let (hidden, aux_count, layers, num_kv, head_dim) = {
            let d = self.real_dflash.as_ref().expect("caller checked");
            (
                d.shape.hidden,
                d.shape.aux_count,
                d.shape.layers,
                d.shape.num_kv_heads,
                d.shape.head_dim,
            )
        };
        let aux_width = hidden * aux_count;
        let kv_dim = num_kv * head_dim;
        let base_pos = self
            .real_dflash
            .as_ref()
            .expect("caller checked")
            .capture_base;
        // BEFORE the pass, not after it: this is a pure function of
        // `capture_base`, `rows` and the cursor, and refusing after
        // `commit_and_wait` would have already written the rows the
        // refusal says were not written.
        {
            let d = self.real_dflash.as_ref().expect("caller checked");
            if base_pos + rows < d.kv.position() {
                return Err(RealForwardError::Unsupported(format!(
                    "the DFlash2 context write would move the cursor BACKWARD: {} to {}",
                    d.kv.position(),
                    base_pos + rows
                )));
            }
        }

        let pass = self.context.begin_pass_labeled("dflash ctx kv");
        let (context, weights, index, dflash) = (
            &mut self.context,
            &self.weights,
            &self.index,
            self.real_dflash.as_ref().expect("caller checked"),
        );

        encode_gemm_any(
            context,
            &pass,
            weights,
            index,
            "dflash.fc.weight",
            hidden,
            aux_width,
            (&dflash.capture, 0),
            (&dflash.ctx_combined, 0),
            rows,
        )?;
        let hidden_norm = norm_view(weights, index, "dflash.hidden_norm.weight", hidden)?;
        for r in 0..rows {
            let row = (r * hidden) as u64 * 2;
            gpu::encode_rms_norm_bf16w(
                context,
                &pass,
                (&dflash.ctx_combined, row),
                hidden_norm,
                (&dflash.ctx_normed, row),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(RealForwardError::Gpu)?;
        }
        let spans = ring_spans(dflash.kv.capacity(0), base_pos, rows);
        for layer in 0..layers {
            let name = |s: &str| format!("dflash.layers.{layer}.{s}");
            let k_norm = norm_view(weights, index, &name("self_attn.k_norm.weight"), head_dim)?;
            for is_k in [true, false] {
                let suffix = if is_k {
                    "self_attn.k_proj.weight"
                } else {
                    "self_attn.v_proj.weight"
                };
                for &(row0, count) in spans.iter().filter(|s| s.1 > 0) {
                    let (buf, off) = if is_k {
                        dflash.kv.k_slot(layer, base_pos + row0)
                    } else {
                        dflash.kv.v_slot(layer, base_pos + row0)
                    };
                    encode_gemm_any(
                        context,
                        &pass,
                        weights,
                        index,
                        &name(suffix),
                        kv_dim,
                        hidden,
                        (&dflash.ctx_normed, (row0 * hidden * 2) as u64),
                        (buf, off as u64),
                        count,
                    )?;
                }
            }
            for r in 0..rows {
                // Per POSITION rather than `k_off + r * stride`: the same
                // wrap the spans above split is what makes the arithmetic
                // form wrong on a straddling round.
                let (k_buf, k_row) = dflash.kv.k_slot(layer, base_pos + r);
                let k_row = k_row as u64;
                gpu::encode_rms_norm_bf16w_perhead(
                    context,
                    &pass,
                    (k_buf, k_row),
                    k_norm,
                    (k_buf, k_row),
                    num_kv as u32,
                    head_dim as u32,
                    RMS_EPS,
                )
                .map_err(RealForwardError::Gpu)?;
                gpu::encode_rope_proportional_neox(
                    context,
                    &pass,
                    (k_buf, k_row),
                    (base_pos + r) as u32,
                    num_kv as u32,
                    head_dim as u32,
                    (head_dim / 2) as u32,
                    DFLASH_THETA,
                )
                .map_err(RealForwardError::Gpu)?;
            }
        }
        // The wait is unconditional: priming exists FOR the KV rows, and
        // they are not written until the GPU has run this pass.
        pass.commit_and_wait();
        let d = self.real_dflash.as_mut().expect("caller checked");
        let target = d.capture_base + rows;
        d.kv.advance_by(target - d.kv.position());
        // Consumed: a later draft must not rewrite these rows, which the
        // next trunk pass replaces wholesale.
        d.capture_rows = 0;
        Ok(())
    }

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
    pub fn dflash_rewind_to(&mut self, position: usize) -> Result<(), RealForwardError> {
        let Some(d) = self.real_dflash.as_mut() else {
            return Err(RealForwardError::Unsupported(
                "no DFlash2 state to rewind".to_string(),
            ));
        };
        let cursor = d.kv.position();
        if position > cursor {
            return Err(RealForwardError::Unsupported(format!(
                "DFlash2 rewind target {position} is ahead of the cache cursor {cursor}"
            )));
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

    /// The host half of the round: top-16 the mask rows' logits, gather the
    /// codebook rows, walk the greedy path.
    fn dflash_select(
        &self,
        anchor: TokenId,
        block: usize,
        proposals: &mut Vec<TokenId>,
    ) -> Result<(), RealForwardError> {
        let d = self.real_dflash.as_ref().expect("caller checked");
        let (rank, vocab) = (d.shape.rank, self.arch.vocab_size as usize);
        let pred = codebook_view(&self.weights, &self.index, PREDECESSOR_CODEBOOK)?;
        let succ = codebook_view(&self.weights, &self.index, SUCCESSOR_CODEBOOK)?;

        let mut row_logits = vec![LogitValue::from_f32(0.0); vocab];
        let hproj = read_f16_rows(&d.hproj, block + 1, rank);
        let mut prev_token = anchor;
        proposals.reserve_exact(block);
        for step in 0..block {
            let row = step + 1;
            gpu::read_buffer_f16_into(&d.logits, row * vocab * 2, &mut row_logits);
            // A NON-FINITE ROW IS REFUSED, not walked. `top_k` compares
            // `v <= val[k-1]`, and every comparison against NaN is false, so
            // a NaN row is ADMITTED at every candidate; then `score >
            // best_score` is false at every one of them and the walk keeps
            // `cand[0]`, which is token id 0. That is exactly what an FP16
            // overflow in the block forward looked like for the life of
            // this drafter: eight zeros a round, a plausible-looking
            // `loses` table, and no error anywhere. It is also why the
            // bisect probe's rank-of-the-true-token read a PERFECT median 0
            // -- ranking counts `v > target`, which NaN also fails.
            if let Some(bad) = row_logits.iter().position(|v| !v.to_f32().is_finite()) {
                return Err(RealForwardError::Unsupported(format!(
                    "the DFlash2 draft pass produced a non-finite logit (row {row}, id {bad}); \
                     the drafter's residual stream has left FP16's range, which \
                     DFLASH_RESIDUAL_SCALE exists to prevent"
                )));
            }
            let (cand, unary) = top_k(&row_logits, DFLASH_TOP_K);
            // One predecessor row (the previous step's chosen token, or the
            // anchor at step 0) scores against every successor candidate.
            let pred_row = read_bf16_row(pred, prev_token, rank);
            let proj = &hproj[row * rank..(row + 1) * rank];
            let mut best = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for (c, &candidate) in cand.iter().enumerate() {
                let succ_row = read_bf16_row(succ, candidate, rank);
                let mut dot = 0.0f32;
                for r in 0..rank {
                    dot += pred_row[r] * proj[r] * succ_row[r];
                }
                let score = unary[c] + dot;
                if score > best_score {
                    best_score = score;
                    best = c;
                }
            }
            prev_token = cand[best];
            proposals.push(prev_token);
        }
        Ok(())
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

    /// The block forward proper: embeddings, `layers` drafter layers, the
    /// final norm, and the two head GEMMs. Assumes the context KV is
    /// current up to `base`.
    fn dflash_forward_block(
        &mut self,
        anchor: TokenId,
        base: usize,
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let vocab = arch.vocab_size as usize;
        let use_silu = arch.hidden_activation.contains("silu");
        let gpu_err = RealForwardError::Gpu;
        let embed_name = "language_model.model.embed_tokens.weight";
        let head_name = if arch.tie_word_embeddings {
            embed_name.to_string()
        } else {
            "language_model.lm_head.weight".to_string()
        };

        let pass = self.context.begin_pass_labeled("dflash draft");
        let (context, weights, index, dflash) = (
            &mut self.context,
            &self.weights,
            &self.index,
            self.real_dflash.as_ref().expect("checked"),
        );
        let s = &dflash.shape;
        let hidden = s.hidden;
        let rows = dflash.block + 1;
        let q_dim = s.num_heads * s.head_dim;
        let kv_dim = s.num_kv_heads * s.head_dim;
        let row_bytes = hidden as u64 * 2;
        let base_kernel_elems = 2 * 2 * hidden;
        let kv_spans = ring_spans(dflash.kv.capacity(0), base, rows);

        // Row 0 is the bonus token; rows 1..block are masks. Both enter at
        // 1/DFLASH_RESIDUAL_SCALE, which is where the residual stream's
        // scaled representation starts; the final norm takes it back out.
        encode_embed_any(
            context,
            &pass,
            weights,
            index,
            embed_name,
            (&dflash.x, 0),
            anchor as u32,
            hidden as u32,
            RESIDUAL_RESCALE,
        )?;
        for r in 1..rows {
            encode_embed_any(
                context,
                &pass,
                weights,
                index,
                embed_name,
                (&dflash.x, r as u64 * row_bytes),
                DFLASH_MASK_TOKEN as u32,
                hidden as u32,
                RESIDUAL_RESCALE,
            )?;
        }

        // TEMPORARY DIAGNOSTIC (docs/DFLASH2.md section 8 item 1): run only
        // the first N drafter layers, so `dflash_probe_buffers` can say
        // which stage first goes non-finite.
        let layer_cap = std::env::var("MFERENCE_DFLASH_LAYERS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(s.layers);
        for layer in 0..s.layers.min(layer_cap) {
            let name = |sfx: &str| format!("dflash.layers.{layer}.{sfx}");
            let input_norm = norm_view(weights, index, &name("input_layernorm.weight"), hidden)?;
            let post_norm = norm_view(
                weights,
                index,
                &name("post_attention_layernorm.weight"),
                hidden,
            )?;
            let q_norm = norm_view(weights, index, &name("self_attn.q_norm.weight"), s.head_dim)?;
            let k_norm = norm_view(weights, index, &name("self_attn.k_norm.weight"), s.head_dim)?;
            let attn_base = norm_view(
                weights,
                index,
                &name("attention_conv.base_kernel"),
                base_kernel_elems,
            )?;
            let mlp_base = norm_view(
                weights,
                index,
                &name("mlp_conv.base_kernel"),
                base_kernel_elems,
            )?;
            for r in 0..rows {
                let row = r as u64 * row_bytes;
                gpu::encode_rms_norm_bf16w(
                    context,
                    &pass,
                    (&dflash.x, row),
                    input_norm,
                    (&dflash.normed, row),
                    hidden as u32,
                    RMS_EPS,
                )
                .map_err(gpu_err)?;
            }

            // attention_conv.prepare: coefficients from the normed input,
            // side-0 convolution of it, and the sublayer reads the
            // CONVOLVED stream.
            encode_gemm_any(
                context,
                &pass,
                weights,
                index,
                &name("attention_conv.kernel_projection.weight"),
                s.conv_rows,
                hidden,
                (&dflash.normed, 0),
                (&dflash.coeffs, 0),
                rows,
            )?;
            gpu::encode_dflash_grouped_conv(
                context,
                &pass,
                (&dflash.normed, 0),
                (&dflash.coeffs, 0),
                attn_base,
                (&dflash.conv_p, 0),
                rows as u32,
                hidden as u32,
                0,
                1.0,
            )
            .map_err(gpu_err)?;

            // Attention: UNGATED q (the trunk's [q; gate] packing is a
            // trunk property this drafter does not share), k/v straight
            // into the slots, per-head norms, full-head NeoX rope, per-row
            // split-KV decode over the ring.
            encode_gemm_any(
                context,
                &pass,
                weights,
                index,
                &name("self_attn.q_proj.weight"),
                q_dim,
                hidden,
                (&dflash.conv_p, 0),
                (&dflash.q, 0),
                rows,
            )?;
            // Split at the ring's wrap, exactly as the context write is:
            // `k_slot` addresses `position % capacity` and a batched
            // projection writes ADJACENT slots, so a block straddling the
            // boundary would run past the layer's buffer.
            for is_k in [true, false] {
                let suffix = if is_k {
                    "self_attn.k_proj.weight"
                } else {
                    "self_attn.v_proj.weight"
                };
                for &(row0, count) in kv_spans.iter().filter(|s| s.1 > 0) {
                    let (buf, off) = if is_k {
                        dflash.kv.k_slot(layer, base + row0)
                    } else {
                        dflash.kv.v_slot(layer, base + row0)
                    };
                    encode_gemm_any(
                        context,
                        &pass,
                        weights,
                        index,
                        &name(suffix),
                        kv_dim,
                        hidden,
                        (&dflash.conv_p, (row0 * hidden) as u64 * 2),
                        (buf, off as u64),
                        count,
                    )?;
                }
            }
            for r in 0..rows {
                let position = base + r;
                let q_row = (r * q_dim) as u64 * 2;
                let (k_buf, k_off) = dflash.kv.k_slot(layer, position);
                let v_buf = dflash.kv.v_slot(layer, position).0;
                let k_row = k_off as u64;
                gpu::encode_rms_norm_bf16w_perhead(
                    context,
                    &pass,
                    (&dflash.q, q_row),
                    q_norm,
                    (&dflash.q, q_row),
                    s.num_heads as u32,
                    s.head_dim as u32,
                    RMS_EPS,
                )
                .map_err(gpu_err)?;
                gpu::encode_rms_norm_bf16w_perhead(
                    context,
                    &pass,
                    (k_buf, k_row),
                    k_norm,
                    (k_buf, k_row),
                    s.num_kv_heads as u32,
                    s.head_dim as u32,
                    RMS_EPS,
                )
                .map_err(gpu_err)?;
                for (data, heads) in [
                    ((&dflash.q, q_row), s.num_heads as u32),
                    ((k_buf, k_row), s.num_kv_heads as u32),
                ] {
                    gpu::encode_rope_proportional_neox(
                        context,
                        &pass,
                        data,
                        position as u32,
                        heads,
                        s.head_dim as u32,
                        (s.head_dim / 2) as u32,
                        DFLASH_THETA,
                    )
                    .map_err(gpu_err)?;
                }
                let seq_len = (position + 1) as u32;
                let kv_start = seq_len.saturating_sub(DFLASH_WINDOW as u32);
                let ring = dflash.kv.ring_capacity(layer) as u32;
                let active_ring = if ring > 0 && seq_len > ring { ring } else { 0 };
                gpu::encode_attention_decode(
                    context,
                    &pass,
                    (&dflash.q, q_row),
                    k_buf,
                    v_buf,
                    &dflash.attn,
                    (&dflash.attn_out, q_row),
                    s.head_dim as u32,
                    s.num_heads as u32,
                    s.num_kv_heads as u32,
                    seq_len,
                    kv_start,
                    active_ring,
                    (s.head_dim as f32).sqrt().recip(),
                    None,
                )
                .map_err(gpu_err)?;
            }
            encode_gemm_any(
                context,
                &pass,
                weights,
                index,
                &name("self_attn.o_proj.weight"),
                hidden,
                q_dim,
                (&dflash.attn_out, 0),
                (&dflash.sub_out, 0),
                rows,
            )?;
            gpu::encode_dflash_grouped_conv(
                context,
                &pass,
                (&dflash.sub_out, 0),
                (&dflash.coeffs, 0),
                attn_base,
                (&dflash.conv_f, 0),
                rows as u32,
                hidden as u32,
                1,
                RESIDUAL_RESCALE,
            )
            .map_err(gpu_err)?;
            for r in 0..rows {
                let row = r as u64 * row_bytes;
                gpu::encode_residual_add(
                    context,
                    &pass,
                    (&dflash.x, row),
                    (&dflash.conv_f, row),
                    hidden as u32,
                )
                .map_err(gpu_err)?;
                gpu::encode_rms_norm_bf16w(
                    context,
                    &pass,
                    (&dflash.x, row),
                    post_norm,
                    (&dflash.normed, row),
                    hidden as u32,
                    RMS_EPS,
                )
                .map_err(gpu_err)?;
            }

            // mlp_conv.prepare, the dense FFN, mlp_conv.finish.
            encode_gemm_any(
                context,
                &pass,
                weights,
                index,
                &name("mlp_conv.kernel_projection.weight"),
                s.conv_rows,
                hidden,
                (&dflash.normed, 0),
                (&dflash.coeffs, 0),
                rows,
            )?;
            gpu::encode_dflash_grouped_conv(
                context,
                &pass,
                (&dflash.normed, 0),
                (&dflash.coeffs, 0),
                mlp_base,
                (&dflash.conv_p, 0),
                rows as u32,
                hidden as u32,
                0,
                1.0,
            )
            .map_err(gpu_err)?;
            for (suffix, out) in [
                ("mlp.gate_proj.weight", &dflash.ffn_gate),
                ("mlp.up_proj.weight", &dflash.ffn_up),
            ] {
                encode_gemm_any(
                    context,
                    &pass,
                    weights,
                    index,
                    &name(suffix),
                    s.inter,
                    hidden,
                    (&dflash.conv_p, 0),
                    (out, 0),
                    rows,
                )?;
            }
            let act = if use_silu {
                gpu::encode_silu_mul
            } else {
                gpu::encode_gelu_mul
            };
            for r in 0..rows {
                let row = (r * s.inter) as u64 * 2;
                act(
                    context,
                    &pass,
                    (&dflash.ffn_gate, row),
                    (&dflash.ffn_up, row),
                    (&dflash.ffn_act, row),
                    s.inter as u32,
                )
                .map_err(gpu_err)?;
            }
            encode_gemm_any(
                context,
                &pass,
                weights,
                index,
                &name("mlp.down_proj.weight"),
                hidden,
                s.inter,
                (&dflash.ffn_act, 0),
                (&dflash.sub_out, 0),
                rows,
            )?;
            gpu::encode_dflash_grouped_conv(
                context,
                &pass,
                (&dflash.sub_out, 0),
                (&dflash.coeffs, 0),
                mlp_base,
                (&dflash.conv_f, 0),
                rows as u32,
                hidden as u32,
                1,
                RESIDUAL_RESCALE,
            )
            .map_err(gpu_err)?;
            for r in 0..rows {
                let row = r as u64 * row_bytes;
                gpu::encode_residual_add(
                    context,
                    &pass,
                    (&dflash.x, row),
                    (&dflash.conv_f, row),
                    hidden as u32,
                )
                .map_err(gpu_err)?;
            }
        }

        // Final norm, then the TRUNK's lm_head (the drafter shares it) and
        // the selector's hidden projection.
        let final_norm = norm_view(weights, index, "dflash.norm.weight", hidden)?;
        for r in 0..rows {
            let row = r as u64 * row_bytes;
            gpu::encode_rms_norm_bf16w(
                context,
                &pass,
                (&dflash.x, row),
                final_norm,
                (&dflash.normed, row),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
        }
        encode_gemm_any(
            context,
            &pass,
            weights,
            index,
            &head_name,
            vocab,
            hidden,
            (&dflash.normed, 0),
            (&dflash.logits, 0),
            rows,
        )?;
        encode_gemm_any(
            context,
            &pass,
            weights,
            index,
            "dflash.candidate_selector.hidden_projection.weight",
            s.rank,
            hidden,
            (&dflash.normed, 0),
            (&dflash.hproj, 0),
            rows,
        )?;
        pass.commit_and_wait();
        Ok(())
    }
}

const PREDECESSOR_CODEBOOK: &str = "dflash.candidate_selector.predecessor_codebook";
const SUCCESSOR_CODEBOOK: &str = "dflash.candidate_selector.successor_codebook";

/// A resident BF16 codebook's (buffer, offset), sized off its own entry so
/// the host gathers can address rows by token id.
fn codebook_view<'a>(
    weights: &'a gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    name: &str,
) -> Result<(&'a gpu::MetalBuffer, u64), RealForwardError> {
    let e = entry(index, name)?;
    norm_view(weights, index, name, e.size_bytes as usize / 2)
}

/// The top-K candidates and their logits, by LINEAR SCAN with a running
/// K-list; ties prefer the lower token id (the reference walk's
/// tie-break). Sorting the whole vocab here would repeat
/// `selection::select`'s measured 18.9 ms mistake once per proposal step.
fn top_k(logits: &[LogitValue], k: usize) -> (Vec<TokenId>, Vec<f32>) {
    let mut idx = vec![0usize; k];
    let mut val = vec![f32::NEG_INFINITY; k];
    for (i, l) in logits.iter().enumerate() {
        let v = l.to_f32();
        // Strictly greater only: an equal logit keeps the earlier (lower)
        // token id, which is the tie-break the walk applies.
        if v <= val[k - 1] {
            continue;
        }
        let mut j = k - 1;
        idx[j] = i;
        val[j] = v;
        while j > 0 && (val[j - 1] < val[j] || (val[j - 1] == val[j] && idx[j - 1] > idx[j])) {
            idx.swap(j - 1, j);
            val.swap(j - 1, j);
            j -= 1;
        }
    }
    (idx.into_iter().map(|i| i as TokenId).collect(), val)
}

/// Reads `rows` rows of FP16 values off a GPU buffer into f32.
fn read_f16_rows(buffer: &gpu::MetalBuffer, rows: usize, width: usize) -> Vec<f32> {
    let raw = gpu::read_buffer_bytes(buffer, 0, rows * width * 2);
    raw.chunks_exact(2)
        .map(|c| LogitValue::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

/// Reads one BF16 codebook row as f32.
fn read_bf16_row(view: (&gpu::MetalBuffer, u64), token: TokenId, rank: usize) -> Vec<f32> {
    let raw = gpu::read_buffer_bytes(
        view.0,
        view.1 as usize + token as usize * rank * 2,
        rank * 2,
    );
    raw.chunks_exact(2)
        .map(|c| compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}
