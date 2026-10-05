//! Condition encoder and AR-to-Flow-VAE timeline fusion.
//!
//! Reference: `mlx_audio/music/models/minimax_music3/fusion.py`. The
//! fused frame hiddens `[1, frames, codebooks * hidden]` are split into
//! `num_condition_layers` slices, mixed by a softmax over learnable
//! layer logits, projected by a kernel-3 conv, and nearest-resampled
//! onto the latent (output-hop) timeline.

use crate::ops;
use crate::Result;
use crate::SpeechError;

use super::backend::{self, ComputeBackend};
use super::conv::{ConvSpec, MlxConv1d};
use super::precision::DType;
use super::weights::{Tensor, WeightStore};
use std::rc::Rc;

pub(crate) struct ConditionDims {
    pub hidden: usize,
    pub num_layers: usize,
    pub input_sampling_rate: usize,
    pub input_hop_length: usize,
    pub output_sampling_rate: usize,
    pub output_hop_length: usize,
}

pub(crate) struct ConditionEncoder {
    layer_weight_logits: Tensor,
    layer_scale: Tensor,
    proj: MlxConv1d,
    dims: ConditionDims,
    backend: Option<Rc<dyn ComputeBackend>>,
}

/// The latent length for `num_frames` AR frames: the reference computes
/// `frames * out_sr / in_sr * in_hop / out_hop` in Python floats and
/// truncates, with a floor of one.
pub(crate) fn latent_length_from_frames(
    num_frames: usize,
    input_sr: usize,
    input_hop: usize,
    output_sr: usize,
    output_hop: usize,
) -> usize {
    let value = num_frames as f64 * output_sr as f64 / input_sr as f64 * input_hop as f64
        / output_hop as f64;
    (value as usize).max(1)
}

/// Nearest resample of channel-major `[ch, length]` to `[ch, size]`.
///
/// The reference multiplies an int32 index range by the f64 length
/// ratio inside MLX, which promotes the product to f32 before the
/// truncating cast; mirror that exactly.
pub(crate) fn nearest_interpolate_1d(x: &[f32], ch: usize, length: usize, size: usize) -> Vec<f32> {
    if size == length {
        return x.to_vec();
    }
    let ratio = (length as f64 / size as f64) as f32;
    let mut out = Vec::with_capacity(ch * size);
    for c in 0..ch {
        for i in 0..size {
            let index = ((i as f32) * ratio) as i32;
            let index = index.clamp(0, length as i32 - 1) as usize;
            out.push(x[c * length + index]);
        }
    }
    out
}

impl ConditionEncoder {
    pub(crate) fn load(
        store: &mut WeightStore,
        base: &str,
        dims: ConditionDims,
    ) -> Result<ConditionEncoder> {
        let logits = store.tensor(&format!("{base}.layer_weight_logits"))?;
        if logits.data.len() != dims.num_layers {
            return Err(SpeechError::Tensor {
                name: format!("{base}.layer_weight_logits"),
                why: format!("expected {} entries", dims.num_layers),
            });
        }
        let scale = store.tensor(&format!("{base}.layer_scale"))?;
        if scale.data.len() != 1 {
            return Err(SpeechError::Tensor {
                name: format!("{base}.layer_scale"),
                why: "expected a single scalar".to_string(),
            });
        }
        let proj = MlxConv1d::load(
            store,
            &format!("{base}.proj"),
            ConvSpec {
                kernel: 3,
                stride: 1,
                padding: 1,
                dilation: 1,
            },
        )?;
        Ok(ConditionEncoder {
            layer_weight_logits: logits,
            layer_scale: scale,
            proj,
            dims,
            backend: store.backend(),
        })
    }

    /// `[1, frames, codebooks * hidden]` -> `[1, target, out_dim]`.
    pub(crate) fn dtype(&self, input: DType) -> DType {
        self.proj.dtype(input)
    }
    #[cfg(test)]
    pub(crate) fn forward(&self, frame_hiddens: &[f32], frames: usize) -> Result<Vec<f32>> {
        self.forward_typed(frame_hiddens, frames, self.layer_scale.dtype)
    }
    pub(crate) fn forward_typed(
        &self,
        frame_hiddens: &[f32],
        frames: usize,
        dtype: DType,
    ) -> Result<Vec<f32>> {
        let dims = &self.dims;
        let fused = dims.num_layers * dims.hidden;
        if frame_hiddens.len() != frames * fused {
            return Err(SpeechError::Input {
                why: format!(
                    "frame hiddens {} do not match {frames}x{fused}",
                    frame_hiddens.len()
                ),
            });
        }
        backend::trace(
            &self.backend,
            "flow.condition.input",
            frame_hiddens,
            dtype,
            &[1, frames, fused],
        );
        // Softmax over the layer mixing weights, in f32.
        let mut weights = self.layer_weight_logits.data.clone();
        ops::softmax_row(&mut weights);
        backend::trace(
            &self.backend,
            "flow.condition.weights_f32",
            &weights,
            DType::F32,
            &[dims.num_layers],
        );
        dtype.round_slice(&mut weights);
        backend::trace(
            &self.backend,
            "flow.condition.weights",
            &weights,
            dtype,
            &[dims.num_layers],
        );
        // Weighted sum over layers into a channel-major [hidden, frames]
        // buffer for the conv.
        let mut mixed = vec![0.0f32; dims.hidden * frames];
        for l in 0..dims.num_layers {
            let w = weights[l];
            for f in 0..frames {
                for d in 0..dims.hidden {
                    mixed[d * frames + f] = dtype.round(
                        mixed[d * frames + f]
                            + dtype.round(w * frame_hiddens[f * fused + l * dims.hidden + d]),
                    );
                }
            }
        }
        dtype.round_slice(&mut mixed);
        backend::trace(
            &self.backend,
            "flow.condition.reduced",
            &mixed,
            dtype,
            &[dims.hidden, frames],
        );
        let scale = dtype.round(self.layer_scale[0]);
        for v in &mut mixed {
            *v = dtype.round(*v * scale);
        }
        backend::trace(
            &self.backend,
            "flow.condition.scaled",
            &mixed,
            dtype,
            &[dims.hidden, frames],
        );
        let projected = self.proj.forward_typed(&mixed, frames, dtype)?;
        let out_ch = projected.len() / frames;
        let projected_dtype = self.proj.dtype(dtype);
        backend::trace(
            &self.backend,
            "flow.condition.projected",
            &projected,
            projected_dtype,
            &[out_ch, frames],
        );
        let target = latent_length_from_frames(
            frames,
            dims.input_sampling_rate,
            dims.input_hop_length,
            dims.output_sampling_rate,
            dims.output_hop_length,
        );
        let resampled = nearest_interpolate_1d(&projected, out_ch, frames, target);
        backend::trace(
            &self.backend,
            "flow.condition.resampled",
            &resampled,
            projected_dtype,
            &[out_ch, target],
        );
        // Back to row-major [1, target, out_dim].
        let mut out = vec![0.0f32; target * out_ch];
        for t in 0..target {
            for c in 0..out_ch {
                out[t * out_ch + c] = resampled[c * target + t];
            }
        }
        Ok(out)
    }
}
