//! Native Z-Image-Turbo transformer-block math, modulation, and final layer.
//!
//! Implements:
//! - AdaLN modulation: linear projection from timestep embedding to (scale_msa, gate_msa, scale_mlp, gate_mlp)
//! - Single transformer block: pre-norm, Q/K/V linear, headwise Q/K RMSNorm, 3-axis RoPE,
//!   bidirectional scaled dot-product attention, post-norm gated residual, SwiGLU FFN with post-norm gated residual
//! - FinalLayer: non-affine LayerNorm, adaptive scale modulation, linear projection back to patch dimension.

#![forbid(unsafe_code)]

use crate::rope::RopeEmbedder;
use ndarray::ArrayView2;
use rayon::prelude::*;

pub const ADALN_EMBED_DIM: usize = 256;
pub const HEAD_DIM: usize = 128;
pub const NORM_EPS: f32 = 1e-5;

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
pub(crate) fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), weight.len(), "RMSNorm dimension mismatch");
    // GPU reductions are tree-shaped. A serial FP32 sum loses enough low
    // bits at width 3840 to breach the checkpoint parity tolerance.
    let mean_square = pairwise_sum_squares(x) / x.len() as f32;
    let scale = (mean_square + eps).sqrt().recip();
    x.iter()
        .zip(weight)
        .map(|(&value, &gain)| value * scale * gain)
        .collect()
}

fn pairwise_sum_squares(values: &[f32]) -> f32 {
    if values.len() <= 32 {
        return values.iter().map(|value| value * value).sum();
    }
    let middle = values.len() / 2;
    pairwise_sum_squares(&values[..middle]) + pairwise_sum_squares(&values[middle..])
}

/// Linear projection: y = weight * x + bias (if present).
///
/// Weight shape: `[out_features, in_features]` in row-major layout.
pub fn linear_forward(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    out_features: usize,
    in_features: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), in_features, "x dimension mismatch");
    assert_eq!(
        weight.len(),
        out_features * in_features,
        "weight dimension mismatch"
    );
    let mut out = vec![0.0f32; out_features];
    for o in 0..out_features {
        let mut sum = 0.0f32;
        let w_row = &weight[o * in_features..(o + 1) * in_features];
        for (xi, wi) in x.iter().zip(w_row.iter()) {
            sum += xi * wi;
        }
        // Match linear algebra backends: reduce the matrix product first,
        // then apply the bias as the separate affine operation.
        out[o] = bias.map_or(sum, |b| sum + b[o]);
    }
    out
}

/// Batched linear projection with input shape `[rows, in_features]`.
///
/// The checkpoint-scale reference needs matrix blocking to reuse weights
/// across image tokens. The scalar helper remains the operation-boundary
/// oracle for single rows and its numerical tests.
fn linear_forward_batch(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    rows: usize,
    out_features: usize,
    in_features: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), rows * in_features, "x batch dimension mismatch");
    assert_eq!(
        weight.len(),
        out_features * in_features,
        "weight dimension mismatch"
    );
    if let Some(bias) = bias {
        assert_eq!(bias.len(), out_features, "bias dimension mismatch");
    }

    let input = ArrayView2::from_shape((rows, in_features), x).expect("validated input shape");
    let threads = rayon::current_num_threads().min(out_features);
    let chunk_features = out_features.div_ceil(threads).div_ceil(64) * 64;
    let fragments: Vec<(usize, usize, Vec<f32>)> = weight
        .par_chunks(chunk_features * in_features)
        .enumerate()
        .map(|(chunk, weights)| {
            let start = chunk * chunk_features;
            let features = weights.len() / in_features;
            let weights = ArrayView2::from_shape((features, in_features), weights)
                .expect("validated weight chunk shape");
            let product = input.dot(&weights.t());
            let (values, offset) = product.into_raw_vec_and_offset();
            assert_eq!(offset, Some(0), "matrix product must use standard layout");
            (start, features, values)
        })
        .collect();

    let mut out = vec![0.0f32; rows * out_features];
    for (start, features, values) in fragments {
        for row in 0..rows {
            let source = &values[row * features..(row + 1) * features];
            let target =
                &mut out[row * out_features + start..row * out_features + start + features];
            target.copy_from_slice(source);
        }
    }
    if let Some(bias) = bias {
        out.par_chunks_mut(out_features).for_each(|row| {
            row.iter_mut()
                .zip(bias)
                .for_each(|(value, addend)| *value += addend)
        });
    }
    out
}

/// Modulation parameters produced by AdaLN linear projection.
#[derive(Debug, Clone)]
pub struct ModulationParams {
    pub scale_msa: Vec<f32>,
    pub gate_msa: Vec<f32>,
    pub scale_mlp: Vec<f32>,
    pub gate_mlp: Vec<f32>,
}

/// Adaptive LayerNorm modulation layer: Linear(min(dim, 256), 4 * dim).
#[derive(Debug, Clone)]
pub struct AdaLnModulation {
    pub dim: usize,
    pub in_dim: usize,
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl AdaLnModulation {
    pub fn new(dim: usize, weight: Vec<f32>, bias: Vec<f32>) -> Self {
        Self::try_new(dim, weight, bias).expect("invalid AdaLN checkpoint tensor shapes")
    }

    /// Build modulation from checkpoint tensors after validating their shape.
    pub fn try_new(dim: usize, weight: Vec<f32>, bias: Vec<f32>) -> Result<Self, String> {
        if dim == 0 {
            return Err("AdaLN dimension must be nonzero".to_string());
        }
        let in_dim = dim.min(ADALN_EMBED_DIM);
        if weight.len() != 4 * dim * in_dim || bias.len() != 4 * dim {
            return Err(format!(
                "AdaLN tensors have lengths weight={} bias={}, expected {} and {}",
                weight.len(),
                bias.len(),
                4 * dim * in_dim,
                4 * dim
            ));
        }
        Ok(Self {
            dim,
            in_dim,
            weight,
            bias,
        })
    }

    /// Project adaln_input into scale and gate parameters for MSA and MLP.
    pub fn modulate(&self, adaln_input: &[f32]) -> ModulationParams {
        assert_eq!(
            adaln_input.len(),
            self.in_dim,
            "AdaLN input dimension mismatch"
        );
        let raw = linear_forward(
            adaln_input,
            &self.weight,
            Some(&self.bias),
            4 * self.dim,
            self.in_dim,
        );
        let scale_msa_raw = &raw[0..self.dim];
        let gate_msa_raw = &raw[self.dim..2 * self.dim];
        let scale_mlp_raw = &raw[2 * self.dim..3 * self.dim];
        let gate_mlp_raw = &raw[3 * self.dim..4 * self.dim];

        let scale_msa = scale_msa_raw.iter().map(|&s| 1.0 + s).collect();
        let gate_msa = gate_msa_raw.iter().map(|&g| g.tanh()).collect();
        let scale_mlp = scale_mlp_raw.iter().map(|&s| 1.0 + s).collect();
        let gate_mlp = gate_mlp_raw.iter().map(|&g| g.tanh()).collect();

        ModulationParams {
            scale_msa,
            gate_msa,
            scale_mlp,
            gate_mlp,
        }
    }
}

/// Weights for one Z-Image transformer block.
#[derive(Debug, Clone)]
pub struct ZImageTransformerBlock {
    pub dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub norm_eps: f32,

    pub attention_norm1: Vec<f32>,
    pub to_q: Vec<f32>,
    pub to_k: Vec<f32>,
    pub to_v: Vec<f32>,
    pub norm_q: Vec<f32>,
    pub norm_k: Vec<f32>,
    pub to_out: Vec<f32>,
    pub attention_norm2: Vec<f32>,

    pub ffn_norm1: Vec<f32>,
    pub w1: Vec<f32>,
    pub w2: Vec<f32>,
    pub w3: Vec<f32>,
    pub ffn_norm2: Vec<f32>,

    pub modulation: Option<AdaLnModulation>,
}

impl ZImageTransformerBlock {
    /// Execute block forward pass.
    ///
    /// Input `x`: flat sequence of shape `[seq_len * dim]`.
    /// `mask`: optional attention mask (`true` means attend, `false` means pad).
    /// `freqs_cis`: flat 3-axis RoPE frequencies of shape `[seq_len * 64]`.
    /// `adaln_input`: optional conditioning embedding of shape `[256]` (or `[in_dim]`).
    pub fn forward(
        &self,
        x: &[f32],
        mask: Option<&[bool]>,
        freqs_cis: &[(f32, f32)],
        adaln_input: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        let seq_len = x.len() / self.dim;
        if x.len() != seq_len * self.dim {
            return Err(format!(
                "input length {} not divisible by dim {}",
                x.len(),
                self.dim
            ));
        }

        let mod_params = match (&self.modulation, adaln_input) {
            (Some(m), Some(inp)) => Some(m.modulate(inp)),
            _ => None,
        };

        // --- Attention branch ---
        // 1. Pre-norm and scale
        let mut x_norm1 = Vec::with_capacity(x.len());
        for t in 0..seq_len {
            let token_x = &x[t * self.dim..(t + 1) * self.dim];
            let normed = rms_norm(token_x, &self.attention_norm1, self.norm_eps);
            if let Some(mp) = &mod_params {
                let scaled: Vec<f32> = normed
                    .iter()
                    .zip(&mp.scale_msa)
                    .map(|(a, s)| a * s)
                    .collect();
                x_norm1.extend_from_slice(&scaled);
            } else {
                x_norm1.extend_from_slice(&normed);
            }
        }

        // 2. Q, K, V projections
        let mut q = linear_forward_batch(&x_norm1, &self.to_q, None, seq_len, self.dim, self.dim);
        let mut k = linear_forward_batch(&x_norm1, &self.to_k, None, seq_len, self.dim, self.dim);
        let v = linear_forward_batch(&x_norm1, &self.to_v, None, seq_len, self.dim, self.dim);

        // 3. Headwise Q / K RMSNorm
        q.par_chunks_mut(self.dim)
            .zip(k.par_chunks_mut(self.dim))
            .for_each(|(q_token, k_token)| {
                for h in 0..self.num_heads {
                    let head_start = h * self.head_dim;
                    let head_end = head_start + self.head_dim;

                    let q_normed =
                        rms_norm(&q_token[head_start..head_end], &self.norm_q, self.norm_eps);
                    q_token[head_start..head_end].copy_from_slice(&q_normed);

                    let k_normed =
                        rms_norm(&k_token[head_start..head_end], &self.norm_k, self.norm_eps);
                    k_token[head_start..head_end].copy_from_slice(&k_normed);
                }
            });

        // 4. Apply RoPE to Q and K
        RopeEmbedder::apply_rotary_emb(&mut q, freqs_cis, self.num_heads, self.head_dim)?;
        RopeEmbedder::apply_rotary_emb(&mut k, freqs_cis, self.num_heads, self.head_dim)?;

        // 5. Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut attn_out = vec![0.0f32; seq_len * self.dim];

        attn_out
            .par_chunks_mut(self.dim)
            .enumerate()
            .for_each(|(i, output_token)| {
                for h in 0..self.num_heads {
                    let q_offset = (i * self.num_heads + h) * self.head_dim;
                    let q_vec = &q[q_offset..q_offset + self.head_dim];

                    // Compute attention scores against all tokens j
                    let mut scores = vec![0.0f32; seq_len];
                    let mut max_score = f32::NEG_INFINITY;
                    for j in 0..seq_len {
                        if let Some(m) = mask {
                            if !m[j] {
                                scores[j] = f32::NEG_INFINITY;
                                continue;
                            }
                        }
                        let k_offset = (j * self.num_heads + h) * self.head_dim;
                        let k_vec = &k[k_offset..k_offset + self.head_dim];
                        let mut dot = 0.0f32;
                        for d in 0..self.head_dim {
                            dot += q_vec[d] * k_vec[d];
                        }
                        let s = dot * scale;
                        scores[j] = s;
                        if s > max_score {
                            max_score = s;
                        }
                    }

                    // Softmax
                    let mut sum_exp = 0.0f32;
                    let mut weights = vec![0.0f32; seq_len];
                    if max_score > f32::NEG_INFINITY {
                        for j in 0..seq_len {
                            if scores[j] > f32::NEG_INFINITY {
                                let e = (scores[j] - max_score).exp();
                                weights[j] = e;
                                sum_exp += e;
                            }
                        }
                    }
                    if sum_exp > 0.0 {
                        let inv_sum = 1.0 / sum_exp;
                        for w in &mut weights {
                            *w *= inv_sum;
                        }
                    }

                    // Weighted sum of values
                    let out_offset = h * self.head_dim;
                    let out_slice = &mut output_token[out_offset..out_offset + self.head_dim];
                    for (j, &w) in weights.iter().enumerate().take(seq_len) {
                        if w > 0.0 {
                            let v_offset = (j * self.num_heads + h) * self.head_dim;
                            let v_vec = &v[v_offset..v_offset + self.head_dim];
                            for d in 0..self.head_dim {
                                out_slice[d] += w * v_vec[d];
                            }
                        }
                    }
                }
            });

        // 6. to_out projection, post-attention norm, and gated residual add
        let projected =
            linear_forward_batch(&attn_out, &self.to_out, None, seq_len, self.dim, self.dim);
        let mut x_mid = vec![0.0f32; x.len()];
        x_mid
            .par_chunks_mut(self.dim)
            .zip(projected.par_chunks(self.dim))
            .zip(x.par_chunks(self.dim))
            .for_each(|((output_token, projected_token), orig_tok)| {
                let normed_proj = rms_norm(projected_token, &self.attention_norm2, self.norm_eps);

                if let Some(mp) = &mod_params {
                    for (((output, orig), normed), gate) in output_token
                        .iter_mut()
                        .zip(orig_tok)
                        .zip(&normed_proj)
                        .zip(&mp.gate_msa)
                    {
                        *output = orig + gate * normed;
                    }
                } else {
                    for ((output, orig), normed) in
                        output_token.iter_mut().zip(orig_tok).zip(&normed_proj)
                    {
                        *output = orig + normed;
                    }
                }
            });

        // --- FFN branch ---
        let intermediate_dim = self.w1.len() / self.dim;
        let mut ffn_input = vec![0.0f32; x.len()];
        ffn_input
            .par_chunks_mut(self.dim)
            .zip(x_mid.par_chunks(self.dim))
            .for_each(|(input_token, mid_tok)| {
                let ffn_norm = rms_norm(mid_tok, &self.ffn_norm1, self.norm_eps);
                if let Some(mp) = &mod_params {
                    for ((output, normed), scale) in
                        input_token.iter_mut().zip(&ffn_norm).zip(&mp.scale_mlp)
                    {
                        *output = normed * scale;
                    }
                } else {
                    input_token.copy_from_slice(&ffn_norm);
                }
            });

        let w1_out = linear_forward_batch(
            &ffn_input,
            &self.w1,
            None,
            seq_len,
            intermediate_dim,
            self.dim,
        );
        let w3_out = linear_forward_batch(
            &ffn_input,
            &self.w3,
            None,
            seq_len,
            intermediate_dim,
            self.dim,
        );
        let mut activation = vec![0.0f32; seq_len * intermediate_dim];
        activation
            .par_iter_mut()
            .zip(&w1_out)
            .zip(&w3_out)
            .for_each(|((output, left), right)| *output = silu(*left) * right);
        let ffn_output = linear_forward_batch(
            &activation,
            &self.w2,
            None,
            seq_len,
            self.dim,
            intermediate_dim,
        );
        let mut out = vec![0.0f32; x.len()];

        out.par_chunks_mut(self.dim)
            .zip(x_mid.par_chunks(self.dim))
            .zip(ffn_output.par_chunks(self.dim))
            .for_each(|((output_token, mid_tok), ffn_token)| {
                let normed_ffn = rms_norm(ffn_token, &self.ffn_norm2, self.norm_eps);

                if let Some(mp) = &mod_params {
                    for (((output, mid), normed), gate) in output_token
                        .iter_mut()
                        .zip(mid_tok)
                        .zip(&normed_ffn)
                        .zip(&mp.gate_mlp)
                    {
                        *output = mid + gate * normed;
                    }
                } else {
                    for ((output, mid), normed) in
                        output_token.iter_mut().zip(mid_tok).zip(&normed_ffn)
                    {
                        *output = mid + normed;
                    }
                }
            });

        Ok(out)
    }
}

/// FinalLayer: non-affine LayerNorm + modulation + biased output linear projection.
#[derive(Debug, Clone)]
pub struct FinalLayer {
    pub hidden_size: usize,
    pub out_channels: usize,
    pub linear_weight: Vec<f32>,
    pub linear_bias: Vec<f32>,
    pub ada_ln_weight: Vec<f32>,
    pub ada_ln_bias: Vec<f32>,
    pub eps: f32,
}

impl FinalLayer {
    pub fn new(
        hidden_size: usize,
        out_channels: usize,
        linear_weight: Vec<f32>,
        linear_bias: Vec<f32>,
        ada_ln_weight: Vec<f32>,
        ada_ln_bias: Vec<f32>,
    ) -> Self {
        Self {
            hidden_size,
            out_channels,
            linear_weight,
            linear_bias,
            ada_ln_weight,
            ada_ln_bias,
            eps: 1e-6,
        }
    }

    /// Forward pass through final layer: LayerNorm -> scale modulation -> biased linear projection.
    pub fn forward(&self, x: &[f32], adaln_input: &[f32]) -> Result<Vec<f32>, String> {
        let seq_len = x.len() / self.hidden_size;
        let in_dim = self.hidden_size.min(ADALN_EMBED_DIM);
        assert_eq!(adaln_input.len(), in_dim);

        // Modulation scale: 1.0 + linear(silu(adaln_input))
        let silu_input: Vec<f32> = adaln_input.iter().map(|&v| silu(v)).collect();
        let mod_out = linear_forward(
            &silu_input,
            &self.ada_ln_weight,
            Some(&self.ada_ln_bias),
            self.hidden_size,
            in_dim,
        );
        let scale: Vec<f32> = mod_out.iter().map(|&m| 1.0 + m).collect();

        let mut out = Vec::with_capacity(seq_len * self.out_channels);
        for t in 0..seq_len {
            let tok = &x[t * self.hidden_size..(t + 1) * self.hidden_size];
            // Non-affine LayerNorm
            let mean: f32 = tok.iter().sum::<f32>() / self.hidden_size as f32;
            let var: f32 =
                tok.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / self.hidden_size as f32;
            let inv_std = 1.0 / (var + self.eps).sqrt();

            let scaled: Vec<f32> = tok
                .iter()
                .zip(&scale)
                .map(|(&xv, &sc)| (xv - mean) * inv_std * sc)
                .collect();

            let proj = linear_forward(
                &scaled,
                &self.linear_weight,
                Some(&self.linear_bias),
                self.out_channels,
                self.hidden_size,
            );
            out.extend_from_slice(&proj);
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::{linear_forward, linear_forward_batch, pairwise_sum_squares};

    #[test]
    fn linear_adds_bias_after_dot_product() {
        let x = [16_777_216.0, 1.0, -16_777_216.0];
        let weight = [1.0, 1.0, 1.0];

        assert_eq!(linear_forward(&x, &weight, Some(&[1.0]), 1, 3), [1.0]);
    }

    #[test]
    fn batched_linear_matches_independent_rows() {
        let x = [1.0, 2.0, 3.0, -2.0, 0.5, 4.0];
        let weight: Vec<f32> = (0..130 * 3)
            .map(|index| (index % 17 - 8) as f32 * 0.125)
            .collect();
        let bias: Vec<f32> = (0..130).map(|index| index as f32 * 0.25).collect();
        let expected: Vec<f32> = x
            .chunks(3)
            .flat_map(|row| linear_forward(row, &weight, Some(&bias), 130, 3))
            .collect();

        assert_eq!(
            linear_forward_batch(&x, &weight, Some(&bias), 2, 130, 3),
            expected
        );
    }

    #[test]
    fn rms_reduction_is_pairwise() {
        let mut values = [1.0f32; 64];
        values[0] = 4096.0;

        assert_eq!(pairwise_sum_squares(&values), 16_777_248.0);
    }
}
