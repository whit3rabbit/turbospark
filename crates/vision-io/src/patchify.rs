//! Patch-row extraction in merge-window order.
//!
//! Adapted from the sconce vision crate's `extract_patches` (see `NOTICE`),
//! with ONE deliberate difference: the inner feature order. See
//! [`patch_rows`].

use crate::error::VisionIoError;
use crate::params::PreprocessParams;

/// A vision grid in patches: temporal groups, patch rows, patch columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridThw {
    pub t: usize,
    pub h: usize,
    pub w: usize,
}

impl GridThw {
    pub fn new(t: usize, h: usize, w: usize) -> Self {
        Self { t, h, w }
    }

    /// Patches in this grid, i.e. rows in the patch matrix.
    pub fn patches(&self) -> usize {
        self.t * self.h * self.w
    }

    /// Tokens this grid costs the trunk after the spatial merge.
    pub fn merged_tokens(&self, merge_size: usize) -> usize {
        self.patches() / (merge_size * merge_size)
    }
}

/// Extract patch rows from interleaved `(y, x, c)` f32 pixels.
///
/// Returns `(rows, grid)` where `rows` is `grid.patches() * params.patch_dim()`
/// floats.
///
/// # Two orders, and why the inner one differs from the reference
///
/// ROW order is merge-window: `(t, wy, wx, ly, lx)`. One whole `merge_size x
/// merge_size` window is emitted before the next, windows in row-major order
/// and patches row-major within a window. This matches the reference exactly
/// and is what the spatial-merge step reads.
///
/// INNER order is `(T, P_h, P_w, C)`. **The reference emits `(C, T, P_h,
/// P_w)`** -- its `transpose(0,1,4,7,5,8,3,2,6,9)` puts the channel axis ahead
/// of the temporal one. This crate transposes it, because the tower's
/// `patch_embed.proj.weight` ships as `[1152, 2, 16, 16, 3]`, i.e.
/// `[out, T, P, P, C]`: emitting rows in that same order lets the repack copy
/// the weight verbatim and the patch-embed GEMM read both operands row-major,
/// with no permutation at repack time and none per image at decode time
/// (`docs/VISION_PHASE0.md` item 4).
///
/// The cost of the choice is that the parity fixture and this function do not
/// compare elementwise, and that gap is bridged by an explicit index
/// permutation in the test rather than by a loosened tolerance. A wrong axis
/// order here is silent wrong numerics: the GEMM still has the right shape, the
/// tower still runs, and the image is simply read as a different image.
///
/// # Temporal duplication
///
/// A still image is one frame, and the tower's temporal patch spans
/// `temporal_patch_size` of them, so the frame is REPEATED to fill the patch
/// (the reference's `np.repeat`). It is repeated, not zero-padded: the two
/// halves of the temporal patch see identical pixels, which is what makes a
/// still image's temporal derivative zero rather than a step edge.
pub fn patch_rows(
    pixels: &[f32],
    height: usize,
    width: usize,
    params: &PreprocessParams,
) -> Result<(Vec<f32>, GridThw), VisionIoError> {
    let factor = params.spatial_factor();
    for (value, axis) in [(height, "height"), (width, "width")] {
        if value == 0 || value % factor != 0 {
            return Err(VisionIoError::InvalidDimensions {
                detail: format!("resized {axis} {value} is not a positive multiple of {factor}"),
            });
        }
    }
    let channels = params.in_channels;
    let expected = height * width * channels;
    if pixels.len() != expected {
        return Err(VisionIoError::InvalidDimensions {
            detail: format!(
                "{width}x{height}x{channels} needs {expected} samples, got {}",
                pixels.len()
            ),
        });
    }

    let patch = params.patch_size;
    let merge = params.merge_size;
    let grid = GridThw::new(1, height / patch, width / patch);
    let windows_h = grid.h / merge;
    let windows_w = grid.w / merge;
    let row_stride = width * channels;

    let mut out = Vec::with_capacity(grid.patches() * params.patch_dim());
    for _t_group in 0..grid.t {
        for wy in 0..windows_h {
            for wx in 0..windows_w {
                for ly in 0..merge {
                    for lx in 0..merge {
                        let patch_row = wy * merge + ly;
                        let patch_col = wx * merge + lx;
                        // (T, P_h, P_w, C): the frame repeats, then rows, then
                        // columns, and the channel triple is contiguous in the
                        // interleaved source, so each `py` is one copy.
                        for _tl in 0..params.temporal_patch_size {
                            for py in 0..patch {
                                let base = (patch_row * patch + py) * row_stride
                                    + patch_col * patch * channels;
                                out.extend_from_slice(&pixels[base..base + patch * channels]);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((out, grid))
}
