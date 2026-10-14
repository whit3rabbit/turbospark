//! NaFlex image preprocessing, as `Siglip2ImageProcessorFast` runs it.
//!
//! 1. Target size: the largest aspect-preserving size whose patch count
//!    fits `max_num_patches`, each side a multiple of the patch size
//!    ([`target_size`], a port of `get_image_size_for_max_num_patches`,
//!    including its f64 binary search).
//! 2. Resize: torchvision `F.resize(uint8, BILINEAR, antialias=True)`.
//!    On CPU that is PyTorch's separable uint8 antialiased kernel
//!    ([`resize_bilinear_aa_u8`]): f64 triangle weights widened by the
//!    downscale factor, quantized to int16 at a per-axis precision, integer
//!    accumulation, horizontal pass first with a uint8 round trip between
//!    passes. It is NOT PIL's bilinear (fixed 22-bit precision); the two
//!    differ by up to two 8-bit levels on the fixture images.
//! 3. Rescale and normalize fused, in f32: `(x - mean/r) / (std/r)` with
//!    `r = rescale_factor`, which is what `rescale_and_normalize` computes.
//! 4. Patchify: patches in row-major patch-grid order, each flattened as
//!    `(patch_row, patch_col, channel)`.
//!
//! The processor then pads to `max_num_patches` with zero rows and a 0
//! mask; [`NaflexInput`] keeps only the valid rows and reconstructs the
//! padded view on request, because padded keys are masked out of every
//! attention in the reference and contribute exactly zero.

use turbospark_vision_io::Rgb8Image;

use super::config::ProcessorConfig;
use crate::error::input_error;
use crate::Result;

/// Patch rows for one image, unpadded.
#[derive(Debug, Clone, PartialEq)]
pub struct NaflexInput {
    /// `[num_patches, patch_size * patch_size * 3]`.
    pub patches: Vec<f32>,
    /// `(patch rows, patch cols)`; `spatial_shapes` in the reference.
    pub spatial_shape: (usize, usize),
    pub patch_size: usize,
    /// The processor's padded length.
    pub max_num_patches: usize,
}

impl NaflexInput {
    pub fn num_patches(&self) -> usize {
        self.spatial_shape.0 * self.spatial_shape.1
    }

    pub fn patch_dim(&self) -> usize {
        self.patch_size * self.patch_size * 3
    }

    /// The reference `pixel_attention_mask`: 1 for valid rows, 0 for pad.
    pub fn pixel_attention_mask(&self) -> Vec<u32> {
        (0..self.max_num_patches)
            .map(|i| u32::from(i < self.num_patches()))
            .collect()
    }

    /// The reference `pixel_values`: valid rows then zero rows.
    pub fn padded_pixel_values(&self) -> Vec<f32> {
        let mut out = self.patches.clone();
        out.resize(self.max_num_patches * self.patch_dim(), 0.0);
        out
    }
}

/// `get_image_size_for_max_num_patches` with `eps = 1e-5`: returns
/// `(height, width)`. The binary search runs in f64 exactly like the
/// Python, so boundary cases land on the same side.
pub fn target_size(
    height: usize,
    width: usize,
    patch: usize,
    max_patches: usize,
) -> (usize, usize) {
    let p = patch as f64;
    let scaled = |scale: f64, size: usize| -> usize {
        let s = size as f64 * scale;
        let s = (s / p).ceil() * p;
        (s as usize).max(patch)
    };
    let eps = 1e-5f64;
    let (mut lo, mut hi) = (eps / 10.0, 100.0f64);
    while hi - lo >= eps {
        let scale = (lo + hi) / 2.0;
        let th = scaled(scale, height);
        let tw = scaled(scale, width);
        let n = (th as f64 / p) * (tw as f64 / p);
        if n <= max_patches as f64 {
            lo = scale;
        } else {
            hi = scale;
        }
    }
    (scaled(lo, height), scaled(lo, width))
}

/// One output index's input window and normalized f64 weights.
pub(crate) struct AxisWeights {
    pub start: usize,
    pub weights: Vec<f64>,
}

/// PyTorch's antialiased bilinear weights (`_compute_weights_aa` with
/// `interp_size = 2`, `align_corners = false`). Downscaling widens the
/// triangle by the scale factor; upscaling keeps support 1, which reduces
/// to half-pixel bilinear with edge clamping. Weights are renormalized
/// after the window is clipped to the input.
pub(crate) fn aa_bilinear_weights(in_size: usize, out_size: usize) -> Vec<AxisWeights> {
    let scale = in_size as f64 / out_size as f64;
    let (support, invscale) = if scale >= 1.0 {
        (scale, 1.0 / scale)
    } else {
        (1.0, 1.0)
    };
    (0..out_size)
        .map(|i| {
            let center = scale * (i as f64 + 0.5);
            // `as i64` truncates toward zero like the C++ cast.
            let xmin = ((center - support + 0.5) as i64).max(0);
            let xmax = ((center + support + 0.5) as i64).min(in_size as i64);
            let mut w: Vec<f64> = (xmin..xmax.max(xmin))
                .map(|j| {
                    let x = ((j as f64 - center + 0.5) * invscale).abs();
                    if x < 1.0 {
                        1.0 - x
                    } else {
                        0.0
                    }
                })
                .collect();
            let total: f64 = w.iter().sum();
            if total != 0.0 {
                for v in &mut w {
                    *v /= total;
                }
            }
            AxisWeights {
                start: xmin as usize,
                weights: w,
            }
        })
        .collect()
}

/// int16 fixed-point weights at the largest precision that keeps the
/// biggest weight under 2^15 (`compute_index_ranges_int16_weights`).
fn fixed_weights(axis: &[AxisWeights]) -> (Vec<Vec<i64>>, u32) {
    let wt_max = axis
        .iter()
        .flat_map(|a| a.weights.iter().copied())
        .fold(0.0f64, f64::max);
    let mut precision = 0u32;
    while precision < 22 {
        let next = (0.5 + wt_max * f64::from(1u32 << (precision + 1))) as i64;
        if next >= 1 << 15 {
            break;
        }
        precision += 1;
    }
    let scale = f64::from(1u32 << precision);
    let fixed = axis
        .iter()
        .map(|a| {
            a.weights
                .iter()
                .map(|&w| {
                    let v = w * scale;
                    (if v < 0.0 { v - 0.5 } else { v + 0.5 }) as i64
                })
                .collect()
        })
        .collect();
    (fixed, precision)
}

/// One separable pass over interleaved RGB8 `[rows, cols, 3]` along cols
/// (`horizontal`) or rows.
fn pass_u8(src: &[u8], rows: usize, cols: usize, out_len: usize, horizontal: bool) -> Vec<u8> {
    let in_len = if horizontal { cols } else { rows };
    let axis = aa_bilinear_weights(in_len, out_len);
    let (fixed, precision) = fixed_weights(&axis);
    let seed = 1i64 << (precision.max(1) - 1);
    let (orows, ocols) = if horizontal {
        (rows, out_len)
    } else {
        (out_len, cols)
    };
    let mut out = vec![0u8; orows * ocols * 3];
    for r in 0..orows {
        for c in 0..ocols {
            let (o, start) = if horizontal {
                (c, axis[c].start)
            } else {
                (r, axis[r].start)
            };
            for ch in 0..3 {
                let mut acc = seed;
                for (k, &w) in fixed[o].iter().enumerate() {
                    let idx = if horizontal {
                        (r * cols + start + k) * 3 + ch
                    } else {
                        ((start + k) * cols + c) * 3 + ch
                    };
                    acc += i64::from(src[idx]) * w;
                }
                out[(r * ocols + c) * 3 + ch] = (acc >> precision).clamp(0, 255) as u8;
            }
        }
    }
    out
}

/// torchvision `F.resize(img_u8, [new_h, new_w], BILINEAR, antialias=True)`
/// on CPU, bit-exact on the fixtures. An axis whose size is unchanged is
/// skipped, as in the reference (its weights would be the identity).
pub fn resize_bilinear_aa_u8(img: &Rgb8Image, new_h: usize, new_w: usize) -> Rgb8Image {
    let (h, w) = (img.height, img.width);
    let mut data = img.data.clone();
    let mut cur_w = w;
    if new_w != w {
        data = pass_u8(&data, h, w, new_w, true);
        cur_w = new_w;
    }
    if new_h != h {
        data = pass_u8(&data, h, cur_w, new_h, false);
    }
    Rgb8Image {
        width: new_w,
        height: new_h,
        data,
    }
}

/// The resized RGB8 image the processor patchifies.
pub fn resized_image(cfg: &ProcessorConfig, img: &Rgb8Image) -> Result<Rgb8Image> {
    if img.width == 0 || img.height == 0 {
        return Err(input_error(format!(
            "image is {}x{}; both sides must be nonzero",
            img.width, img.height
        )));
    }
    if img.data.len() != img.width * img.height * 3 {
        return Err(input_error(
            "RGB8 buffer length does not match its dimensions",
        ));
    }
    let (th, tw) = target_size(img.height, img.width, cfg.patch_size, cfg.max_num_patches);
    Ok(resize_bilinear_aa_u8(img, th, tw))
}

/// Full preprocessing of one RGB8 image.
pub fn preprocess(cfg: &ProcessorConfig, img: &Rgb8Image) -> Result<NaflexInput> {
    let resized = resized_image(cfg, img)?;
    let p = cfg.patch_size;
    let (gh, gw) = (resized.height / p, resized.width / p);
    if gh * gw > cfg.max_num_patches {
        return Err(input_error(format!(
            "{gh}x{gw} patches exceed max_num_patches {}",
            cfg.max_num_patches
        )));
    }
    // Fused rescale+normalize: mean and std are scaled by 1/rescale in
    // f32 first, then `(x - m) / s`, matching the reference's f32 ops.
    let inv = (1.0 / cfg.rescale_factor) as f32;
    let mean: [f32; 3] = std::array::from_fn(|c| cfg.image_mean[c] * inv);
    let std: [f32; 3] = std::array::from_fn(|c| cfg.image_std[c] * inv);
    let dim = p * p * 3;
    let mut patches = vec![0.0f32; gh * gw * dim];
    for py in 0..gh {
        for px in 0..gw {
            let row = &mut patches[(py * gw + px) * dim..(py * gw + px + 1) * dim];
            for y in 0..p {
                for x in 0..p {
                    let src = ((py * p + y) * resized.width + px * p + x) * 3;
                    for c in 0..3 {
                        row[(y * p + x) * 3 + c] =
                            (f32::from(resized.data[src + c]) - mean[c]) / std[c];
                    }
                }
            }
        }
    }
    Ok(NaflexInput {
        patches,
        spatial_shape: (gh, gw),
        patch_size: p,
        max_num_patches: cfg.max_num_patches,
    })
}
