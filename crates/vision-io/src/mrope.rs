//! mRoPE position triples: the `(t, h, w)` position every token in a prompt
//! occupies once images are spliced in.
//!
//! A port of `get_rope_index` in mlx-vlm's `qwen3_vl/language.py`. Text runs
//! get the same value in all three slots, so they behave exactly as ordinary
//! 1-D positions; an image block spends ONE position per merged row and column
//! rather than one per token, which is what lets a 4,096-token image cost only
//! `max(llm_grid_h, llm_grid_w)` positions of context.

use crate::error::VisionIoError;
use foundation::TokenId;

/// Where one image's placeholder tokens sit in the id sequence.
///
/// M-V5 injects tower output at `[start, start + len)`; M-V6 produces the run
/// in the first place. Recorded here because this walk already has to find
/// them, and finding them twice invites the two answers disagreeing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSpan {
    /// Index of the first placeholder token.
    pub start: usize,
    /// Placeholder tokens, i.e. the image's merged token count.
    pub len: usize,
}

/// The special ids the walk keys on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionSpecialIds {
    /// Emitted before an image or video block. Counted, never walked from.
    pub vision_start: TokenId,
    /// The image placeholder, repeated once per merged token.
    pub image_pad: TokenId,
}

/// The result of the walk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MropePositions {
    /// One `(t, h, w)` triple per input id, in input order.
    pub triples: Vec<(i32, i32, i32)>,
    /// `max_position + 1 - seq_len`. Added to a cache position to get the next
    /// token's position during decode, since the prompt's positions advance
    /// more slowly than its token count.
    pub rope_delta: i32,
    /// The image placeholder runs, in order.
    pub spans: Vec<ImageSpan>,
}

/// Walk an id sequence and assign every token its mRoPE triple.
///
/// # The walk
///
/// Left to right. A text run of length `L` starting at position `k` occupies
/// `(k, k, k) .. (k+L-1, ...)`. The image block that follows starts at
/// `block_start = k + L` and its tokens take `(block_start + ti,
/// block_start + hi, block_start + wi)` over the `t -> h -> w` nest, with
/// `llm_grid_h = grid.h / merge` and `llm_grid_w = grid.w / merge`. The next
/// run resumes at one past the largest position used, so the block advances
/// the clock by `max(t, llm_grid_h, llm_grid_w)` rather than by its token
/// count.
///
/// # Two rules taken from the reference rather than reasoned out
///
/// **How many images the walk expects is counted from `vision_start` markers,
/// but WHERE each block sits is found from the placeholder token itself.** The
/// reference counts `vision_start`-followed-by-`image_pad` pairs to size its
/// loop and then calls `.index(image_token_id, st)` to place each one. The two
/// disagree on a malformed prompt: placeholders with no preceding
/// `vision_start` are not counted, so the loop does not run and they are
/// treated as ordinary text. That is reproduced here rather than corrected,
/// because the trunk this feeds is the reference's trunk -- being right about
/// a prompt the model was never trained on is worth less than agreeing with it
/// on every prompt it was.
///
/// **The grid supplies `t` verbatim.** `llm_grid_t` is the grid's own `t`, NOT
/// divided by anything; only the spatial axes are divided by the merge size.
///
/// # Errors
///
/// [`VisionIoError::PlaceholderMismatch`] when the counted image blocks and
/// the supplied grids disagree. This is checked rather than tolerated: the
/// reference indexes its grid list positionally and would read past the end or
/// silently leave a grid unused.
pub fn mrope_position_triples(
    ids: &[TokenId],
    grids: &[crate::patchify::GridThw],
    special: VisionSpecialIds,
    merge_size: usize,
) -> Result<MropePositions, VisionIoError> {
    if merge_size == 0 {
        return Err(VisionIoError::InvalidDimensions {
            detail: "merge size must be positive".into(),
        });
    }

    // Count the way the reference does: a placeholder that FOLLOWS a
    // vision-start marker. `ids[..len-1]` because a marker in the last slot
    // has nothing after it.
    let image_count = ids
        .iter()
        .enumerate()
        .take(ids.len().saturating_sub(1))
        .filter(|(i, &token)| token == special.vision_start && ids[i + 1] == special.image_pad)
        .count();
    if image_count != grids.len() {
        return Err(VisionIoError::PlaceholderMismatch {
            placeholders: image_count,
            grids: grids.len(),
        });
    }

    let mut triples: Vec<(i32, i32, i32)> = Vec::with_capacity(ids.len());
    let mut spans = Vec::with_capacity(image_count);
    // One past the largest position assigned so far, i.e. where the next run
    // starts. Zero before anything is assigned.
    let mut next_pos: i32 = 0;
    let mut cursor = 0usize;

    for grid in grids.iter().take(image_count) {
        let Some(offset) = ids[cursor..].iter().position(|&t| t == special.image_pad) else {
            // Unreachable while the count above matches: a counted marker is
            // followed by a placeholder at or after `cursor`. Refused rather
            // than assumed, because the alternative is an index panic.
            return Err(VisionIoError::PlaceholderMismatch {
                placeholders: spans.len(),
                grids: grids.len(),
            });
        };
        let block_start_idx = cursor + offset;
        let text_len = block_start_idx - cursor;
        for k in 0..text_len {
            let p = next_pos + k as i32;
            triples.push((p, p, p));
        }

        let block_start = next_pos + text_len as i32;
        let llm_h = grid.h / merge_size;
        let llm_w = grid.w / merge_size;
        let block_len = grid.t * llm_h * llm_w;
        if block_start_idx + block_len > ids.len() {
            return Err(VisionIoError::PlaceholderMismatch {
                placeholders: ids.len() - block_start_idx,
                grids: grids.len(),
            });
        }
        for ti in 0..grid.t {
            for hi in 0..llm_h {
                for wi in 0..llm_w {
                    triples.push((
                        block_start + ti as i32,
                        block_start + hi as i32,
                        block_start + wi as i32,
                    ));
                }
            }
        }
        spans.push(ImageSpan {
            start: block_start_idx,
            len: block_len,
        });

        // The block advances the clock by its LARGEST axis, which is why an
        // image costs far fewer positions than tokens.
        let span = grid.t.max(llm_h).max(llm_w) as i32;
        next_pos = block_start + span;
        cursor = block_start_idx + block_len;
    }

    for k in 0..ids.len() - cursor {
        let p = next_pos + k as i32;
        triples.push((p, p, p));
    }

    let max_position = triples
        .iter()
        .map(|&(t, h, w)| t.max(h).max(w))
        .max()
        .unwrap_or(-1);
    Ok(MropePositions {
        rope_delta: max_position + 1 - ids.len() as i32,
        triples,
        spans,
    })
}
