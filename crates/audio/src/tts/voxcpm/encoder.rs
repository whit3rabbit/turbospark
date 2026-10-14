//! The VoxCPM local patch encoder (`VoxCPMLocEnc`).
//!
//! Port of `mlx_audio/tts/models/voxcpm/encoder.py` at the pinned
//! mlx-audio commit: a biased input projection from latent patches, a
//! learned special token prepended to every patch window, and a
//! vocabulary-free bidirectional MiniCPM pass whose CLS (position 0)
//! output becomes the patch embedding.
//!
//! The reference shapes this as a `(B*T, P+1, H)` batch of independent
//! rows; the port folds the rows into one sequence with a blocked
//! mask and per-block rope positions, which is bit-equivalent at
//! single batch.

use crate::lm::{Bias, Weights};
use crate::nn::Linear;
use crate::tts::voxcpm::minicpm::{self, Mask, MiniCpmModel};
use crate::{Result, SpeechError};

/// The loaded patch encoder.
pub struct VoxCpmLocEnc {
    special_token: Vec<f32>,
    in_proj: Linear,
    encoder: MiniCpmModel,
}

impl VoxCpmLocEnc {
    pub(crate) fn load(
        weights: &Weights,
        prefix: &str,
        input_dim: usize,
        encoder_config: &minicpm::Config,
    ) -> Result<Self> {
        let hidden = encoder_config.hidden_size;
        let special_token =
            weights.raw_tensor(&format!("{prefix}special_token"), &[1, 1, 1, hidden])?;
        Ok(Self {
            special_token,
            in_proj: weights.linear(
                &format!("{prefix}in_proj"),
                input_dim,
                hidden,
                Bias::Required,
            )?,
            encoder: MiniCpmModel::load(
                weights,
                &format!("{prefix}encoder."),
                encoder_config.clone(),
            )?,
        })
    }

    /// The CLS encoder over `patches` of `[rows, patch_size,
    /// latent_dim]` (row-major). Returns `[rows, hidden]`.
    pub fn forward(&mut self, patches: &[f32], rows: usize, patch_size: usize) -> Result<Vec<f32>> {
        let hidden = self.in_proj.output;
        let dim = self.in_proj.input;
        if patches.len() != rows * patch_size * dim {
            return Err(SpeechError::Input {
                why: format!(
                    "patch input length {} is not {rows} x {patch_size} x {dim}",
                    patches.len()
                ),
            });
        }
        // in_proj over the flattened (rows * patch) windows.
        let projected = self.in_proj.forward(patches, rows * patch_size);

        // Prepend the special token to every row: positions
        // [special, patch_0 .. patch_{P-1}] per row, folded into one
        // sequence of `rows * (P + 1)` positions.
        let block = patch_size + 1;
        let mut sequence = Vec::with_capacity(rows * block * hidden);
        for row in 0..rows {
            sequence.extend_from_slice(&self.special_token);
            let base = row * patch_size * hidden;
            sequence.extend_from_slice(&projected[base..base + patch_size * hidden]);
        }
        let out =
            self.encoder
                .forward(&sequence, rows * block, None, Mask::Blocks { len: block })?;

        // The CLS output (position 0 of every row).
        let mut cls = Vec::with_capacity(rows * hidden);
        for row in 0..rows {
            cls.extend_from_slice(&out[row * block * hidden..(row * block + 1) * hidden]);
        }
        Ok(cls)
    }
}
