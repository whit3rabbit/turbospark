//! Three-axis adjacent-pair Rotary Position Embedding (RoPE) for Z-Image-Turbo.
//!
//! Computes 3D positional frequencies across temporal and spatial axes (t, h, w)
//! with theta=256.0 and dimensions [32, 48, 48] (summing to head dimension 128).
//! Unlike the vision encoder's NeoX pairing (which splits channels in halves),
//! this rotary embedding rotates strictly adjacent pairs: (x[2k], x[2k + 1]).

#![forbid(unsafe_code)]

pub const DEFAULT_ROPE_THETA: f64 = 256.0;
pub const DEFAULT_AXES_DIMS: [usize; 3] = [32, 48, 48];
pub const DEFAULT_AXES_LENS: [usize; 3] = [1536, 512, 512];
pub const TOTAL_HEAD_DIM: usize = 128;
pub const TOTAL_COMPLEX_PAIRS: usize = 64;

/// Precomputed frequency tables for 3-axis positional embeddings.
#[derive(Debug, Clone)]
pub struct RopeEmbedder {
    pub theta: f64,
    pub axes_dims: Vec<usize>,
    pub axes_lens: Vec<usize>,
    /// Table per axis: freqs_cis[axis][step * (dim / 2) + pair_idx] = (cos, sin)
    freqs_cis: Vec<Vec<(f32, f32)>>,
}

impl Default for RopeEmbedder {
    fn default() -> Self {
        Self::new(&DEFAULT_AXES_DIMS, &DEFAULT_AXES_LENS, DEFAULT_ROPE_THETA)
    }
}

impl RopeEmbedder {
    /// Create embedder and precompute cos/sin tables for all axes.
    pub fn new(axes_dims: &[usize], axes_lens: &[usize], theta: f64) -> Self {
        Self::try_new(axes_dims, axes_lens, theta).expect("invalid Z-Image RoPE configuration")
    }

    /// Fallible constructor for configuration supplied by a checkpoint.
    pub fn try_new(axes_dims: &[usize], axes_lens: &[usize], theta: f64) -> Result<Self, String> {
        if axes_dims.len() != axes_lens.len() || axes_dims.len() != 3 {
            return Err(
                "Z-Image RoPE requires exactly three matching axis dimensions and lengths"
                    .to_string(),
            );
        }
        if !theta.is_finite() || theta <= 0.0 {
            return Err(format!(
                "RoPE theta must be finite and positive, got {theta}"
            ));
        }
        let mut tables = Vec::with_capacity(axes_dims.len());
        for (&d, &e) in axes_dims.iter().zip(axes_lens.iter()) {
            if d == 0 || d % 2 != 0 || e == 0 {
                return Err(format!("RoPE axis dimension {d} must be positive and even and length {e} must be positive"));
            }
            let half = d / 2;
            let mut table = Vec::with_capacity(e * half);
            for t in 0..e {
                for j in 0..half {
                    let freq = 1.0 / theta.powf((2.0 * j as f64) / d as f64);
                    let angle = (t as f64) * freq;
                    table.push((angle.cos() as f32, angle.sin() as f32));
                }
            }
            tables.push(table);
        }

        Ok(Self {
            theta,
            axes_dims: axes_dims.to_vec(),
            axes_lens: axes_lens.to_vec(),
            freqs_cis: tables,
        })
    }

    /// Embed an array of 3D position coordinates [[t, h, w], ...].
    ///
    /// Returns a flat vector of (cos, sin) pairs of length `ids.len() * 64`.
    pub fn embed_ids(&self, ids: &[[i32; 3]]) -> Result<Vec<(f32, f32)>, String> {
        let total_pairs: usize = self.axes_dims.iter().map(|d| d / 2).sum();
        let mut out = Vec::with_capacity(ids.len() * total_pairs);

        for (idx, coord) in ids.iter().enumerate() {
            for (axis, &val) in coord.iter().enumerate() {
                let half = self.axes_dims[axis] / 2;
                let step = if val < 0 {
                    return Err(format!(
                        "token {idx} axis {axis} has negative coordinate {val}"
                    ));
                } else {
                    val as usize
                };
                if step >= self.axes_lens[axis] {
                    return Err(format!(
                        "token {idx} axis {axis} coordinate {step} exceeds axis length {}",
                        self.axes_lens[axis]
                    ));
                }
                let start = step * half;
                let end = start + half;
                out.extend_from_slice(&self.freqs_cis[axis][start..end]);
            }
        }

        Ok(out)
    }

    /// Apply adjacent-pair rotary embedding in place across a sequence.
    ///
    /// Shape of `x`: `[seq_len * num_heads * head_dim]`.
    /// `freqs_cis`: `[seq_len * (head_dim / 2)]` representing (cos, sin) per pair.
    pub fn apply_rotary_emb(
        x: &mut [f32],
        freqs_cis: &[(f32, f32)],
        num_heads: usize,
        head_dim: usize,
    ) -> Result<(), String> {
        if num_heads == 0 || head_dim == 0 || head_dim % 2 != 0 {
            return Err("num_heads and an even nonzero head_dim are required for RoPE".to_string());
        }
        let half = head_dim / 2;
        let seq_len = x.len() / (num_heads * head_dim);
        if x.len() != seq_len * num_heads * head_dim {
            return Err(format!(
                "tensor length {} not divisible by num_heads * head_dim ({})",
                x.len(),
                num_heads * head_dim
            ));
        }
        if freqs_cis.len() != seq_len * half {
            return Err(format!(
                "freqs_cis length {} does not match seq_len * (head_dim / 2) ({} * {} = {})",
                freqs_cis.len(),
                seq_len,
                half,
                seq_len * half
            ));
        }

        for t in 0..seq_len {
            let token_freqs = &freqs_cis[t * half..(t + 1) * half];
            for h in 0..num_heads {
                let offset = (t * num_heads + h) * head_dim;
                for (p, &(cos, sin)) in token_freqs.iter().enumerate().take(half) {
                    let r_idx = offset + 2 * p;
                    let i_idx = offset + 2 * p + 1;
                    let x_r = x[r_idx];
                    let x_i = x[i_idx];
                    x[r_idx] = x_r * cos - x_i * sin;
                    x[i_idx] = x_r * sin + x_i * cos;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_dimensions_and_step_zero() {
        let embedder = RopeEmbedder::default();
        assert_eq!(embedder.axes_dims, vec![32, 48, 48]);
        let ids = [[0, 0, 0]];
        let pairs = embedder.embed_ids(&ids).expect("embed step zero");
        assert_eq!(pairs.len(), TOTAL_COMPLEX_PAIRS);
        // At step 0, cos(0) = 1.0, sin(0) = 0.0 for all frequencies
        for &(cos, sin) in &pairs {
            assert!((cos - 1.0).abs() < 1e-6);
            assert!(sin.abs() < 1e-6);
        }
    }

    #[test]
    fn test_apply_rotary_emb_identity_at_zero() {
        let embedder = RopeEmbedder::default();
        let ids = [[0, 0, 0]];
        let freqs = embedder.embed_ids(&ids).unwrap();
        let mut x = vec![1.0f32; TOTAL_HEAD_DIM];
        RopeEmbedder::apply_rotary_emb(&mut x, &freqs, 1, TOTAL_HEAD_DIM).unwrap();
        for &val in &x {
            assert!((val - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_apply_rotary_emb_orthogonal_rotation() {
        let freqs = vec![(0.0f32, 1.0f32)]; // 90 degree rotation
        let mut x = vec![1.0f32, 1.0f32];
        RopeEmbedder::apply_rotary_emb(&mut x, &freqs, 1, 2).unwrap();
        // (1, 1) * (0, 1) = (-1, 1), exercising both adjacent-pair signs.
        assert!((x[0] + 1.0).abs() < 1e-6);
        assert!((x[1] - 1.0).abs() < 1e-6);
    }
}
