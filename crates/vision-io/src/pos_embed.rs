//! Bilinear interpolation of the tower's learned position-embedding grid onto
//! an image's own patch grid.
//!
//! The tower ships `pos_embed.weight` as a fixed `48 x 48` grid of 1152-wide
//! rows (2,304 entries). An image's grid is whatever `smart_resize` produced,
//! so every patch reads a bilinear blend of four table rows. This module
//! computes the four INDICES and the four WEIGHTS per patch; the gather and
//! the weighted sum happen in the tower, where the table lives.

use crate::error::VisionIoError;
use crate::params::PreprocessParams;
use crate::patchify::GridThw;

/// Four index planes and four weight planes, one entry per patch, in the same
/// merge-window order as the patch rows.
///
/// The four correspond to `(floor_h, floor_w)`, `(floor_h, ceil_w)`,
/// `(ceil_h, floor_w)`, `(ceil_h, ceil_w)`; the weights sum to one per patch.
#[derive(Debug, Clone, PartialEq)]
pub struct PosEmbedTable {
    /// Row offsets into the flattened `n_grid_per_side^2` table.
    pub indices: [Vec<usize>; 4],
    pub weights: [Vec<f32>; 4],
}

impl PosEmbedTable {
    /// Patches covered.
    pub fn len(&self) -> usize {
        self.indices[0].len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The interpolation table for one image grid.
///
/// # The mapping is endpoint-preserving, not half-pixel
///
/// Source coordinates are `linspace(0, n-1, count)`, i.e. `i * (n-1) /
/// (count-1)`, so the first and last patch land exactly on the first and last
/// table row. This is NOT the half-pixel convention (`(i + 0.5) * n / count -
/// 0.5`) that image resampling normally uses, including the bicubic resize two
/// modules over. Both are "bilinear interpolation"; they place every interior
/// sample differently, and the difference is a smooth shift that reads as a
/// slightly mis-registered image rather than as an error.
///
/// A `count` of one is the degenerate case the formula divides by zero on. The
/// reference's `linspace(0, n-1, 1)` yields `[0]`, so a single patch reads the
/// table's FIRST row rather than its centre, and that is reproduced here.
///
/// # Order
///
/// Computed on the row-major `h x w` grid and then permuted into merge-window
/// order, which is the reference's own two-step shape and which matters
/// because the interpolation is defined on the grid while the consumer reads
/// windows.
pub fn pos_embed_weights(
    grid: GridThw,
    n_grid_per_side: usize,
    params: &PreprocessParams,
) -> Result<PosEmbedTable, VisionIoError> {
    if n_grid_per_side == 0 {
        return Err(VisionIoError::InvalidDimensions {
            detail: "position embedding grid side must be positive".into(),
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

    let axis = |count: usize| -> (Vec<usize>, Vec<usize>, Vec<f32>) {
        let last = n_grid_per_side - 1;
        // `linspace`'s exact f32 spelling: `(stop - start) * (i / (num - 1))`,
        // the FRACTION first and the span second. Three algebraically equal
        // orderings were checked against `mx.linspace` at counts 4, 6, 10 and
        // 16; this is the only one that reproduces it, while `(i * last) /
        // (count - 1)` and `i * step` each disagree in the last f32 bit.
        //
        // That bit is not cosmetic here. It moves a bilinear weight by ~4e-7,
        // and the weighted sum of four table rows by ~1.6e-6 -- above the bar
        // the parity test holds, so the alternative to matching the spelling
        // is loosening a tolerance until it hides the difference.
        let denom = if count > 1 { (count - 1) as f32 } else { 1.0 };
        let mut floors = Vec::with_capacity(count);
        let mut ceils = Vec::with_capacity(count);
        let mut fracs = Vec::with_capacity(count);
        for i in 0..count {
            let pos = if count > 1 {
                last as f32 * (i as f32 / denom)
            } else {
                0.0
            };
            // The reference casts to int32, which truncates. `pos` is never
            // negative, so truncation and floor agree; spelled as floor
            // because that is the intent.
            let floor = (pos.floor() as usize).min(last);
            floors.push(floor);
            ceils.push((floor + 1).min(last));
            fracs.push(pos - floor as f32);
        }
        (floors, ceils, fracs)
    };
    let (h_floor, h_ceil, dh) = axis(grid.h);
    let (w_floor, w_ceil, dw) = axis(grid.w);

    // Row-major over the grid first; permuted below.
    let n = grid.h * grid.w;
    let mut indices = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    let mut weights = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    for y in 0..grid.h {
        let base = h_floor[y] * n_grid_per_side;
        let base_ceil = h_ceil[y] * n_grid_per_side;
        for x in 0..grid.w {
            indices[0].push(base + w_floor[x]);
            indices[1].push(base + w_ceil[x]);
            indices[2].push(base_ceil + w_floor[x]);
            indices[3].push(base_ceil + w_ceil[x]);
            weights[0].push((1.0 - dh[y]) * (1.0 - dw[x]));
            weights[1].push((1.0 - dh[y]) * dw[x]);
            weights[2].push(dh[y] * (1.0 - dw[x]));
            weights[3].push(dh[y] * dw[x]);
        }
    }

    let order = merge_window_order(grid, merge);
    Ok(PosEmbedTable {
        indices: core::array::from_fn(|k| order.iter().map(|&i| indices[k][i]).collect()),
        weights: core::array::from_fn(|k| order.iter().map(|&i| weights[k][i]).collect()),
    })
}

/// Row-major grid positions listed in merge-window order, repeated for each
/// temporal group. Entry `j` is the row-major index of the `j`-th patch.
pub(crate) fn merge_window_order(grid: GridThw, merge: usize) -> Vec<usize> {
    let mut order = Vec::with_capacity(grid.patches());
    for _t in 0..grid.t {
        for wy in 0..grid.h / merge {
            for wx in 0..grid.w / merge {
                for ly in 0..merge {
                    for lx in 0..merge {
                        order.push((wy * merge + ly) * grid.w + wx * merge + lx);
                    }
                }
            }
        }
    }
    order
}
