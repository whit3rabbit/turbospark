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
    // theta is the base of the frequency ladder: zero, negative or non-finite
    // gives inf or NaN rows the tower would rotate by without any error.
    if !(theta.is_finite() && theta > 0.0) {
        return Err(VisionIoError::InvalidDimensions {
            detail: format!("vision rope theta {theta} must be finite and positive"),
        });
    }
    params.check_geometry()?;
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

/// Mage-ViT's rotary rows for one still image: `[patches, head_dim / 2]`
/// angles in merge-window order, three runs of `(t, h, w)` at 4, 6 and 6
/// sixteenths of the row.
///
/// Two things differ from [`vision_rope_freq_rows`] besides the third axis:
/// the exponent is `j / size`, not `2j / size`, and the row feeds an
/// INTERLEAVED rotation that reads it twice (lane `l` of the head takes angle
/// `l % (head_dim / 2)`). The temporal run is all zeros: a still image is
/// frame 0. More than one frame is video, which is refused.
pub fn mage_vit_rope_rows(
    grid: GridThw,
    head_dim: usize,
    params: &PreprocessParams,
) -> Result<Vec<f32>, VisionIoError> {
    if head_dim == 0 || head_dim % 32 != 0 {
        return Err(VisionIoError::InvalidDimensions {
            detail: format!("mage_vit head_dim {head_dim} must be a positive multiple of 32"),
        });
    }
    if grid.t != 1 {
        return Err(VisionIoError::InvalidDimensions {
            detail: format!(
                "mage_vl video input is not supported; {} frames given, only a still image",
                grid.t
            ),
        });
    }
    params.check_geometry()?;
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
    let (t, axis) = (4 * half / 16, 6 * half / 16);
    let inv: Vec<f32> = (0..axis)
        .map(|j| 1.0 / VISION_ROPE_THETA.powf(j as f32 / axis as f32))
        .collect();
    let mut out = Vec::with_capacity(grid.patches() * half);
    for wy in 0..grid.h / merge {
        for wx in 0..grid.w / merge {
            for ly in 0..merge {
                for lx in 0..merge {
                    let row = (wy * merge + ly) as f32;
                    let col = (wx * merge + lx) as f32;
                    out.extend(std::iter::repeat_n(0.0f32, t));
                    out.extend(inv.iter().map(|f| row * f));
                    out.extend(inv.iter().map(|f| col * f));
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod mage_tests {
    use super::*;

    #[test]
    fn mage_rows_refuse_video_by_name_and_leave_time_at_frame_zero() {
        let params = PreprocessParams::from_preprocessor_config_json(
            r#"{"patch_size":16,"temporal_patch_size":1,"merge_size":2,"min_pixels":1024,
                "max_pixels":4000000,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]}"#,
        )
        .unwrap();
        let err = mage_vit_rope_rows(GridThw::new(2, 2, 2), 64, &params).unwrap_err();
        assert!(err.to_string().contains("video"), "{err}");
        assert!(mage_vit_rope_rows(GridThw::new(1, 2, 2), 48, &params).is_err());
        let rows = mage_vit_rope_rows(GridThw::new(1, 2, 2), 64, &params).unwrap();
        assert_eq!(rows.len(), 4 * 32);
        // The last patch (row 1, column 1): 8 zeros, then row and column.
        let last = &rows[3 * 32..];
        assert!(last[..8].iter().all(|&a| a == 0.0));
        assert_eq!((last[8], last[20]), (1.0, 1.0));
    }
}
