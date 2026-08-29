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
        })
    }

    /// The FP16 row to blit at `position`, or `None` for a text position,
    /// where the trunk does its ordinary embedding lookup.
    pub(crate) fn row_for(&self, position: usize) -> Option<&[u8]> {
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
        let row = self.prefix[i] + (position - span.start);
        let start = row * self.row_bytes;
        Some(&self.rows[start..start + self.row_bytes])
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
