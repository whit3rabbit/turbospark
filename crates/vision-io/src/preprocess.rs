//! The composed pipeline: smart resize, PIL bicubic, normalize, patchify.

use crate::decode::Rgb8Image;
use crate::error::VisionIoError;
use crate::normalize::rescale_and_normalize;
use crate::params::PreprocessParams;
use crate::patchify::{patch_rows, GridThw};
use crate::resize::resize_bicubic_pil;
use crate::smart_resize::resized_dims;

/// One preprocessed image: the patch matrix, its grid, and the token cost.
#[derive(Debug, Clone, PartialEq)]
pub struct PreprocessedImage {
    /// `grid.patches() * params.patch_dim()` floats, merge-window order with
    /// `(T, P_h, P_w, C)` inside each row. See [`crate::patchify::patch_rows`].
    pub patch_rows: Vec<f32>,
    pub grid: GridThw,
    /// Trunk tokens this image expands to, i.e. `grid.patches() / merge^2`.
    pub merged_tokens: usize,
    /// What [`resized_dims`] chose, in pixels. Reported because it is the one
    /// step whose output is not recoverable from the grid alone at a glance,
    /// and because a surprising token count is almost always a surprising
    /// resize.
    pub resized: (usize, usize),
}

/// Run the pipeline over decoded pixels.
///
/// The order is fixed and each step depends on the previous one's exact
/// output: resize decides the grid, normalization runs on the RESIZED uint8
/// samples (not on the source, and not on floats carried through the resize),
/// and patchify tiles the normalized plane. Normalizing before the resize
/// would run the bicubic kernel's `0..255` clamp on a signed signal and
/// destroy every negative sample.
pub fn preprocess(
    image: &Rgb8Image,
    params: &PreprocessParams,
) -> Result<PreprocessedImage, VisionIoError> {
    let (resized_h, resized_w) = resized_dims(image.height, image.width, params)?;
    let resized = resize_bicubic_pil(&image.data, image.width, image.height, resized_w, resized_h);
    let normalized = rescale_and_normalize(&resized, params);
    let (patch_rows, grid) = patch_rows(&normalized, resized_h, resized_w, params)?;
    Ok(PreprocessedImage {
        merged_tokens: grid.merged_tokens(params.merge_size),
        patch_rows,
        grid,
        resized: (resized_h, resized_w),
    })
}
