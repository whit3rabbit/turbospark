//! What one prompt's images contribute to the trunk: the FP16 rows that
//! replace an embedding lookup, and the `(t, h, w)` triple every position
//! occupies (ROADMAP M-V5).
//!
//! # Two lookups, and they answer different questions
//!
//! [`PromptVision::row_for`] is about the EMBEDDING: at an image-pad position
//! the trunk blits a tower row into `scratch.x` instead of reading the table.
//! [`PromptVision::rope_position`] is about the ANGLE, and it applies to every
//! position rather than only to image ones -- text after an image gets a rope
//! position that is no longer its token index, because an image block advances
//! the clock by its largest axis rather than by its token count.
//!
//! # Why the second one covers text too, and why that costs nothing
//!
//! `get_rope_index` gives every TEXT token `t == h == w`
//! (`docs/VISION_PHASE0.md` item 2), so a text position still dispatches
//! `rope_neox_subdim` -- just at the prompt's own position rather than at the
//! token index. Only an image's own tokens diverge and reach
//! `rope_mrope_interleaved`. With no images at all the two are the same
//! number and nothing about the pre-vision engine changes, which is the
//! degenerate-equivalence invariant this milestone is judged on.
//!
//! # Everything is validated at construction
//!
//! Each check below is a wrong image silently reaching the model: a span
//! length that disagrees with a tower's merged-token count blits the previous
//! image's rows into part of this one, and an `out_hidden` that disagrees with
//! the trunk writes a row of the wrong width over its neighbour. None of that
//! fails at runtime -- it decodes fluently off the wrong picture.

use turbospark_vision_io::{ImageSpan, MropePositions};

use super::VisionEmbedding;
use crate::real_forward_types::RealForwardError;

/// Which angle an attention block rotates one position by.
///
/// DEFINED HERE rather than in the `qwen` flow that built it first, because
/// the `llama` flow's `qwen3_vl` trunk consumes the same seam: both families
/// rotate by a `(t, h, w)` triple on an image prompt and by the raw position
/// otherwise, and a second copy of the enum one family over would be
/// AGENTS.md Gotcha 61's duplicate-allowlist shape.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RopePosition {
    /// Rotate by the `position` argument. Every caller before ROADMAP M-V5,
    /// and every caller on a text-only prompt.
    Sequential,
    /// Rotate by this `(t, h, w)`, which an image prompt's own position table
    /// supplies ([`PromptVision::rope_position`]).
    ///
    /// **`t == h == w` here still takes the EXISTING kernel**, and that is the
    /// whole dispatch rule. It is a property of the DATA rather than a
    /// classification of the token: `get_rope_index` gives every text token of
    /// a mixed prompt the same number in all three slots, so the divergence
    /// test IS the "is this an image pad" test, with nothing extra to plumb
    /// and nothing to get out of step.
    Triple(i32, i32, i32),
}

/// One prompt's image rows and position table.
#[derive(Debug, Clone)]
pub struct PromptVision {
    /// Every span's FP16 rows, concatenated in span order, already in the
    /// little-endian byte form `gpu::write_buffer_bytes` wants. Stored as
    /// bytes rather than `u16` so the per-token blit is a slice and not a
    /// conversion.
    rows: Vec<u8>,
    /// Byte width of one row.
    row_bytes: usize,
    /// The image placeholder runs, in order and non-overlapping.
    spans: Vec<ImageSpan>,
    /// `spans[i]`'s first row index in `rows`, so a lookup needs no running
    /// sum. `prefix[i]` is the number of rows before span `i`.
    prefix: Vec<usize>,
    /// One `(t, h, w)` per prompt position.
    triples: Vec<(i32, i32, i32)>,
    /// Added to a position past the prompt to get its rope position. Usually
    /// NEGATIVE, because an image spends far fewer positions than tokens.
    rope_delta: i32,
    /// The deepstack mergers' rows, one entry per declared index, each
    /// `total_merged * out_hidden` FP16 values as little-endian bytes -- the
    /// same byte form [`Self::rows`] is in, and the same row width, because
    /// a deepstack merger's output width is the trunk's hidden size too.
    /// EMPTY for a tower without deepstack.
    deepstack: Vec<Vec<u8>>,
    /// The deepstack rows' GPU-resident twins, one [`gpu::MetalBuffer`] per
    /// [`Self::deepstack`] entry, uploaded by the runner at
    /// `set_prompt_vision` time so the trunk's per-layer adds bind GPU
    /// storage directly instead of re-writing bytes every micro-batch.
    /// Populated after construction (`attach_deepstack_buffers`); EMPTY
    /// until then and always for a tower without deepstack.
    ///
    /// Dropped WITH the map: `clear_prompt_vision` and `reset()` free the
    /// buffers through ordinary `Drop`, which is the right lifetime for
    /// storage that only the injection this map describes reads.
    deepstack_buffers: Vec<gpu::MetalBuffer>,
}

impl PromptVision {
    /// Build the map for one prompt.
    ///
    /// `embeddings` are the tower's output, one per image, in prompt order;
    /// `positions` is `turbospark_vision_io::mrope_position_triples`'s answer
    /// for the same id sequence, which is where the spans come from. Producing
    /// the spans in one place and checking them against the towers here is the
    /// point: the walk already has to find them, and finding them twice invites
    /// the two answers disagreeing.
    pub fn new(
        embeddings: &[VisionEmbedding],
        positions: &MropePositions,
        prompt_len: usize,
        hidden_size: usize,
    ) -> Result<Self, RealForwardError> {
        let refuse = |detail: String| Err(RealForwardError::Unsupported(detail));

        if positions.triples.len() != prompt_len {
            return refuse(format!(
                "mrope triples cover {} positions; the prompt has {prompt_len}",
                positions.triples.len()
            ));
        }
        if embeddings.len() != positions.spans.len() {
            return refuse(format!(
                "{} encoded image(s) against {} placeholder span(s) in the prompt",
                embeddings.len(),
                positions.spans.len()
            ));
        }

        let row_bytes = hidden_size * 2;
        let mut rows: Vec<u8> = Vec::new();
        let mut prefix = Vec::with_capacity(embeddings.len());
        let mut last_end = 0usize;

        // The deepstack halves must agree across every image: one declared
        // merger count, one row count per image, one output width. A
        // disagreement is not a runtime error later -- it is one image's
        // injection at another image's positions, which decodes fluently.
        let deepstack_len = embeddings.first().map(|e| e.deepstack.len()).unwrap_or(0);
        if let Some(mismatched) = embeddings
            .iter()
            .position(|e| e.deepstack.len() != deepstack_len)
        {
            return refuse(format!(
                "image {mismatched}: tower produced {} deepstack row set(s) against \
                 {deepstack_len} from image 0",
                embeddings[mismatched].deepstack.len()
            ));
        }
        let mut deepstack: Vec<Vec<u8>> = vec![Vec::new(); deepstack_len];

        for (i, (embedding, span)) in embeddings.iter().zip(positions.spans.iter()).enumerate() {
            if embedding.out_hidden != hidden_size {
                return refuse(format!(
                    "image {i}: tower emits {}-wide rows; the trunk's residual stream is \
                     {hidden_size}-wide",
                    embedding.out_hidden
                ));
            }
            if embedding.merged_tokens != span.len {
                return refuse(format!(
                    "image {i}: tower produced {} merged token(s) against a {}-token \
                     placeholder run at position {}",
                    embedding.merged_tokens, span.len, span.start
                ));
            }
            if embedding.rows.len() != embedding.merged_tokens * embedding.out_hidden {
                return refuse(format!(
                    "image {i}: {} row value(s) for {} x {}",
                    embedding.rows.len(),
                    embedding.merged_tokens,
                    embedding.out_hidden
                ));
            }
            for (k, ds) in embedding.deepstack.iter().enumerate() {
                if ds.rows.len() != embedding.merged_tokens * embedding.out_hidden {
                    return refuse(format!(
                        "image {i}: deepstack row set {k} carries {} value(s) for {} x {}",
                        ds.rows.len(),
                        embedding.merged_tokens,
                        embedding.out_hidden
                    ));
                }
                deepstack[k].reserve(ds.rows.len() * 2);
                for value in &ds.rows {
                    deepstack[k].extend_from_slice(&value.to_le_bytes());
                }
            }
            // Sorted and disjoint is what makes the binary search in
            // `row_for` correct. The walk produces them that way; checked
            // rather than assumed, because a caller may build spans by hand
            // (M-V6 does not exist yet, so today every caller does).
            if span.start < last_end {
                return refuse(format!(
                    "image {i}: placeholder run at {} overlaps the previous run, which ended \
                     at {last_end}",
                    span.start
                ));
            }
            if span.start + span.len > prompt_len {
                return refuse(format!(
                    "image {i}: placeholder run [{}, {}) runs past a {prompt_len}-token prompt",
                    span.start,
                    span.start + span.len
                ));
            }
            last_end = span.start + span.len;

            prefix.push(rows.len() / row_bytes.max(1));
            rows.reserve(embedding.rows.len() * 2);
            for value in &embedding.rows {
                rows.extend_from_slice(&value.to_le_bytes());
            }
        }

        // The decode rule below adds `rope_delta` to a raw position, so a
        // delta that could drive it negative is refused here rather than
        // saturated at the call site: a rope position of 0 for token 4,000 is
        // fluent and wrong, where a refusal names the malformed table.
        if prompt_len as i32 + positions.rope_delta < 0 {
            return refuse(format!(
                "rope_delta {} would put the first decode position before zero on a \
                 {prompt_len}-token prompt",
                positions.rope_delta
            ));
        }

        Ok(Self {
            rows,
            row_bytes,
            spans: positions.spans.clone(),
            prefix,
            triples: positions.triples.clone(),
            rope_delta: positions.rope_delta,
            deepstack,
            deepstack_buffers: Vec::new(),
        })
    }

    /// Upload the deepstack rows to GPU storage, one buffer per declared
    /// merger. Called by the runner at `set_prompt_vision` time, AFTER
    /// [`Self::new`] has validated every row count; a no-op when the tower
    /// has no deepstack.
    pub(crate) fn attach_deepstack_buffers(
        &mut self,
        context: &mut gpu::MetalContext,
    ) -> Result<(), RealForwardError> {
        if !self.deepstack_buffers.is_empty() {
            return Ok(());
        }
        self.deepstack_buffers = self
            .deepstack
            .iter()
            .map(|bytes| context.new_buffer_with_data(bytes.as_slice()))
            .collect();
        Ok(())
    }

    /// Deepstack merger `k`'s GPU rows, or `None` before
    /// `attach_deepstack_buffers` ran (which no production caller can see:
    /// `set_prompt_vision` attaches before installing the map).
    #[allow(dead_code)]
    pub(crate) fn deepstack_buffer(&self, k: usize) -> Option<&gpu::MetalBuffer> {
        self.deepstack_buffers.get(k)
    }

    /// Every deepstack buffer, in injection order. The llama dense prefill
    /// clones the retain handles out before its field split.
    pub(crate) fn deepstack_buffers(&self) -> &[gpu::MetalBuffer] {
        &self.deepstack_buffers
    }

    /// The FP16 row to blit at `position`, or `None` for a text position,
    /// where the trunk does its ordinary embedding lookup.
    pub(crate) fn row_for(&self, position: usize) -> Option<&[u8]> {
        let row = self.row_index_for(position)?;
        let start = row * self.row_bytes;
        Some(&self.rows[start..start + self.row_bytes])
    }

    /// The merged-row INDEX that `position` blits, across all spans, or
    /// `None` for a text position.
    ///
    /// The deepstack adds are the second consumer: they add from GPU
    /// buffers laid out in the same concatenated order, so both consumers
    /// must compute the same index, which is why one function answers both.
    pub(crate) fn row_index_for(&self, position: usize) -> Option<usize> {
        // The last span starting at or before `position`. Spans are disjoint
        // and sorted, so at most one can contain it.
        let i = self
            .spans
            .partition_point(|span| span.start <= position)
            .checked_sub(1)?;
        let span = self.spans[i];
        if position >= span.start + span.len {
            return None;
        }
        Some(self.prefix[i] + (position - span.start))
    }

    /// The `(t, h, w)` this position occupies.
    ///
    /// Inside the prompt this is the walk's own answer. Past it, decode
    /// continues at `(p, p, p)` with `p = position + rope_delta`, which is the
    /// reference's rule and is why `rope_delta` is carried at all: the prompt's
    /// positions advance more slowly than its token count, so a decode step
    /// that used its cache index would jump.
    pub(crate) fn rope_position(&self, position: usize) -> (i32, i32, i32) {
        match self.triples.get(position) {
            Some(&triple) => triple,
            None => {
                let p = position as i32 + self.rope_delta;
                (p, p, p)
            }
        }
    }

    /// Total image rows carried, for a caller reporting what it injected.
    pub fn image_rows(&self) -> usize {
        self.rows.len() / self.row_bytes.max(1)
    }

    /// The placeholder runs this map covers.
    pub fn spans(&self) -> &[ImageSpan] {
        &self.spans
    }
}
