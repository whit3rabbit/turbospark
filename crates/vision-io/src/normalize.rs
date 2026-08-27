//! Rescale then per-channel normalize.

use crate::params::PreprocessParams;

/// `(sample * rescale_factor - mean[c]) / std[c]`, in that fixed order.
///
/// For this family's `mean = std = 0.5` and `rescale_factor = 1/255` that is
/// `2*(px/255) - 1`, but the general form is kept and the values come from the
/// checkpoint: the shorthand is a property of these two checkpoints, not of
/// the pipeline.
///
/// Input and output are both INTERLEAVED `(y, x, c)`, not planar. The patch
/// row's innermost axis is the channel, so keeping channels adjacent here
/// makes [`crate::patchify`] copy contiguous runs instead of gathering with a
/// plane stride.
pub fn rescale_and_normalize(src: &[u8], params: &PreprocessParams) -> Vec<f32> {
    let mut out = vec![0.0f32; src.len()];
    for (i, (slot, &sample)) in out.iter_mut().zip(src).enumerate() {
        let c = i % 3;
        *slot =
            (sample as f32 * params.rescale_factor - params.image_mean[c]) / params.image_std[c];
    }
    out
}
