//! Shared residual-VQ nearest-code lookup for the codec families.
//!
//! Both the SNAC and Descript DAC references use the same
//! `decode_latents` math (`snac/vq.py`, `descript/nn/quantize.py`):
//! L2-normalize the projected frame and every codebook row (norm
//! floored at 1e-12), then score squared Euclidean distance as
//! `|e|^2 - 2 e.c + |c|^2` over the normalized vectors. Ties resolve
//! to the first index, matching `argmax(-dist)`.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::wnconv::load_f32_shaped;
use crate::{Result, SpeechError};

/// A codebook with its normalized rows precomputed.
pub(crate) struct CodebookIndex {
    /// Raw rows `[size, dim]`; the emitted quantized vectors.
    codebook: Vec<f32>,
    /// Rows L2-normalized for the distance lookup.
    rows_norm: Vec<f32>,
    /// Squared norms of the normalized rows.
    rows_sq: Vec<f32>,
    dim: usize,
}

impl CodebookIndex {
    /// Loads `{prefix}.codebook.weight` with shape validation.
    pub fn load(file: &SafetensorsFile, prefix: &str, dim: usize) -> Result<Self> {
        let name = format!("{prefix}.codebook.weight");
        let desc = file.descriptor(&name).ok_or_else(|| SpeechError::Tensor {
            name: name.clone(),
            why: "missing codebook".to_string(),
        })?;
        if desc.shape.len() != 2 || desc.shape[1] != dim {
            return Err(SpeechError::Tensor {
                name: name.clone(),
                why: format!("expected codebook [size, {dim}], got {:?}", desc.shape),
            });
        }
        let size = desc.shape[0];
        let codebook = load_f32_shaped(file, &name, &[size, dim])?;
        let mut rows_norm = vec![0.0f32; codebook.len()];
        let mut rows_sq = vec![0.0f32; size];
        for r in 0..size {
            let row = &codebook[r * dim..(r + 1) * dim];
            let sq: f32 = row.iter().map(|v| v * v).sum();
            let scale = 1.0 / sq.sqrt().max(1e-12);
            for (dst, src) in rows_norm[r * dim..(r + 1) * dim].iter_mut().zip(row) {
                *dst = src * scale;
            }
            rows_sq[r] = rows_norm[r * dim..(r + 1) * dim]
                .iter()
                .map(|v| v * v)
                .sum();
        }
        Ok(CodebookIndex {
            codebook,
            rows_norm,
            rows_sq,
            dim,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Raw (unnormalized) codebook row; this is the emitted vector.
    pub fn raw_row(&self, idx: usize) -> &[f32] {
        &self.codebook[idx * self.dim..(idx + 1) * self.dim]
    }

    /// Nearest code for one frame `col [dim]`; returns
    /// `(index, distance)` with first-wins tie handling.
    pub fn nearest(&self, col: &[f32]) -> (usize, f32) {
        let scale = 1.0 / col.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        let e2: f32 = col.iter().map(|v| v * scale * (v * scale)).sum();
        let mut best_dist = f32::INFINITY;
        let mut best_idx = 0usize;
        let rows = self.rows_sq.len();
        for idx in 0..rows {
            let row = &self.rows_norm[idx * self.dim..(idx + 1) * self.dim];
            let mut dot = 0.0f32;
            for (a, &b) in col.iter().zip(row) {
                dot += a * scale * b;
            }
            let dist = e2 - 2.0 * dot + self.rows_sq[idx];
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }
        (best_idx, best_dist)
    }
}
