//! The vision tower's 2-D rotary frequency rows.

use crate::error::VisionIoError;
use crate::params::PreprocessParams;
use crate::patchify::GridThw;

/// The vision tower's rope base. **10,000, and deliberately not the trunk's.**
///
/// The qwen3_5 text trunk declares `rope_theta: 10000000` (1e7) in
/// `text_config.rope_parameters`. The tower's `VisionRotaryEmbedding` takes
/// the library default and no vision config field overrides it, so the two
/// halves of one checkpoint run different bases. Carrying the trunk's 1e7 in
/// here would rotate every patch by a nearly-constant angle and flatten the
/// tower's spatial signal, which produces plausible embeddings rather than an
/// error.
pub const VISION_ROPE_THETA: f32 = 10_000.0;

/// Per-token rotary frequency rows, `grid.patches()` of them, each
/// `head_dim / 2` wide, in merge-window order.
///
/// # Shape
///
/// The table is built over one axis at `head_dim / 4` frequencies
/// (`inv_freq[j] = theta^(-2j / (head_dim/2))`), and each token's row is
/// `concat(table[h], table[w])`: the height half first, then the width half.
/// At this family's `head_dim` of 72 that is 18 frequencies per axis and a
/// 36-wide row.
///
/// Swapping the two halves is the mistake to guard against. It keeps the row
/// width, keeps every value in range, and simply transposes the model's sense
/// of the image.
///
/// # Indices
///
/// `h` and `w` are FULL-RESOLUTION patch coordinates, not merged ones: the
/// rotation is applied inside the tower, before the spatial merge. They are
/// enumerated in the same `(t, wy, wx, ly, lx)` nest as the patch rows, so row
/// `i` here belongs to patch row `i` there.
pub fn vision_rope_freq_rows(
    grid: GridThw,
    head_dim: usize,
    theta: f32,
    params: &PreprocessParams,
) -> Result<Vec<f32>, VisionIoError> {
    if head_dim == 0 || head_dim % 4 != 0 {
        return Err(VisionIoError::InvalidDimensions {
            detail: format!("vision head_dim {head_dim} must be a positive multiple of 4"),
        });
    }
    let merge = params.merge_size;
    if grid.h == 0 || grid.w == 0 || grid.h % merge != 0 || grid.w % merge != 0 {
        return Err(VisionIoError::InvalidDimensions {
            detail: format!(
                "patch grid {}x{} is not a positive multiple of the merge size {merge}",
                grid.w, grid.h
            ),
        });
    }

    let half = head_dim / 2;
    let freqs = half / 2;
    let inv_freq: Vec<f32> = (0..freqs)
        .map(|j| 1.0 / theta.powf((2 * j) as f32 / half as f32))
        .collect();

    let mut out = Vec::with_capacity(grid.patches() * half);
    for _t in 0..grid.t {
        for wy in 0..grid.h / merge {
            for wx in 0..grid.w / merge {
                for ly in 0..merge {
                    for lx in 0..merge {
                        let row = (wy * merge + ly) as f32;
                        let col = (wx * merge + lx) as f32;
                        out.extend(inv_freq.iter().map(|f| row * f));
                        out.extend(inv_freq.iter().map(|f| col * f));
                    }
                }
            }
        }
    }
    Ok(out)
}

/// [`vision_rope_freq_rows`] at [`VISION_ROPE_THETA`].
pub fn vision_rope_freq_rows_default(
    grid: GridThw,
    head_dim: usize,
    params: &PreprocessParams,
) -> Result<Vec<f32>, VisionIoError> {
    vision_rope_freq_rows(grid, head_dim, VISION_ROPE_THETA, params)
}
