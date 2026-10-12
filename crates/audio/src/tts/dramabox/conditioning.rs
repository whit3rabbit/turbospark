//! The DramaBox text conditioner
//! (`mlx_audio/tts/models/dramabox/text_conditioning.py`).
//!
//! Reference: mlx-audio at commit
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/dramabox).
//! `FeatureExtractorV2` normalizes each of the Gemma hidden-state
//! layers per token over the hidden dim, concatenates the layers,
//! rescales, and projects to the cross dim; the 1D connector reorders
//! padded positions behind learnable registers, runs rope'd gated
//! attention blocks over a single learned register grid, and renormalizes.
//! The Gemma encoder itself is a separate downloadable model, so the
//! caller supplies the stacked hidden states `[tokens, hidden, layers]`.

use crate::nn::Linear;
use crate::{Result, SpeechError};
use turbospark_model_io::safetensors::SafetensorsFile;

use super::dit::{rms_norm, Attention, FeedForward};
use super::sharded::{load_linear_sharded, load_sharded};

/// `FeatureExtractorV2`: per-token per-layer RMS over the hidden dim,
/// layer concatenation, the sqrt(target/source) rescale, and one
/// biased projection to the cross dim.
pub(crate) struct FeatureExtractor {
    aggregate: Linear,
    hidden_dim: usize,
    num_layers: usize,
    cross_dim: usize,
    eps: f32,
}

impl FeatureExtractor {
    pub(crate) fn load(
        files: &[SafetensorsFile],
        name: &str,
        embedding_dim: usize,
        audio_inner_dim: usize,
        num_layers: usize,
    ) -> Result<Self> {
        Ok(Self {
            aggregate: load_linear_sharded(
                files,
                &format!("{name}.audio_aggregate_embed"),
                embedding_dim * num_layers,
                audio_inner_dim,
            )?,
            hidden_dim: embedding_dim,
            num_layers,
            cross_dim: audio_inner_dim,
            eps: 1e-6,
        })
    }

    /// `hidden_states [tokens, hidden * layers]` (layers concatenated
    /// per token) plus the binary mask `[tokens]` -> `[tokens, cross]`.
    /// `hidden_states [tokens, hidden * layers]`: the Gemma layers
    /// stacked per token with the hidden dim OUTER and the layer INNER
    /// (`mx.stack(hidden_states, axis=-1).reshape(B, T, D * L)`). The
    /// RMS normalization runs over the hidden dim PER LAYER, the mask
    /// zeroes invalid tokens, and the rescale multiplies by
    /// sqrt(cross_dim / hidden).
    pub(crate) fn forward(&self, encoded: &[f32], tokens: usize, mask: &[i32]) -> Vec<f32> {
        let hidden = self.hidden_dim;
        let layers = self.num_layers;
        let width = hidden * layers;
        let mut normed = vec![0.0f32; encoded.len()];
        for (token, mask_row) in mask.iter().enumerate().take(tokens) {
            let base = token * width;
            for layer in 0..layers {
                let mut sum = 0.0f32;
                for d in 0..hidden {
                    let value = encoded[base + d * layers + layer];
                    sum += value * value;
                }
                let inv = 1.0 / (sum / hidden as f32 + self.eps).sqrt();
                for d in 0..hidden {
                    normed[base + d * layers + layer] = encoded[base + d * layers + layer] * inv;
                }
            }
            if *mask_row == 0 {
                for slot in &mut normed[base..base + width] {
                    *slot = 0.0;
                }
            }
        }
        let scale = (self.cross_dim as f64 / self.hidden_dim as f64).sqrt() as f32;
        for value in normed.iter_mut() {
            *value *= scale;
        }
        self.aggregate.forward(&normed, tokens)
    }
}

/// `Embeddings1DConnector`: the register reorder, the rope'd blocks,
/// and the final RMSNorm.
pub(crate) struct Embeddings1DConnector {
    blocks: Vec<(Attention, FeedForward)>,
    registers: Vec<f32>,
    num_registers: usize,
    inner_dim: usize,
    heads: usize,
    norm_eps: f32,
    connector_theta: f64,
    connector_max_pos: f32,
}

impl Embeddings1DConnector {
    #[allow(clippy::too_many_arguments)]
    fn load(
        files: &[SafetensorsFile],
        name: &str,
        heads: usize,
        dim_head: usize,
        num_layers: usize,
        num_registers: usize,
        norm_eps: f32,
        connector_theta: f64,
        connector_max_pos: f32,
    ) -> Result<Self> {
        let inner_dim = heads * dim_head;
        let mut blocks = Vec::with_capacity(num_layers);
        for index in 0..num_layers {
            let block = format!("{name}.transformer_1d_blocks.{index}");
            blocks.push((
                Attention::load(
                    files,
                    &format!("{block}.attn1"),
                    inner_dim,
                    inner_dim,
                    heads,
                    dim_head,
                    norm_eps,
                    true,
                    true,
                )?,
                FeedForward::load(files, &format!("{block}.ff"), inner_dim)?,
            ));
        }
        let registers = if num_registers > 0 {
            load_sharded(
                files,
                &format!("{name}.learnable_registers"),
                &[num_registers, inner_dim],
            )?
        } else {
            Vec::new()
        };
        Ok(Self {
            blocks,
            registers,
            num_registers,
            inner_dim,
            heads,
            norm_eps,
            connector_theta,
            connector_max_pos,
        })
    }

    /// `hidden [tokens, inner]` plus the binary attention mask
    /// `[tokens]` -> `[tokens, inner]`. The register reorder walks
    /// valid tokens to the front in their original order (sort key
    /// `position - tokens` for valid, `position` for padding) and
    /// refills the tail with the tiled learnable registers; the
    /// blocks then run with the all-zero additive mask the reference
    /// returns (a no-op, so no mask is applied here).
    fn forward(&self, hidden: &[f32], tokens: usize, mask: &[i32]) -> Result<Vec<f32>> {
        let mut x = hidden.to_vec();
        if self.num_registers > 0 {
            if tokens % self.num_registers != 0 {
                return Err(SpeechError::Input {
                    why: format!(
                        "hidden state length {tokens} must be divisible by num_learnable_registers {}",
                        self.num_registers
                    ),
                });
            }
            let mut order: Vec<usize> = (0..tokens).collect();
            order.sort_by_key(|&position| {
                if mask[position] > 0 {
                    position as i64 - tokens as i64
                } else {
                    position as i64
                }
            });
            let mut reordered = vec![0.0f32; x.len()];
            for (out_row, &in_row) in order.iter().enumerate() {
                reordered[out_row * self.inner_dim..(out_row + 1) * self.inner_dim]
                    .copy_from_slice(&x[in_row * self.inner_dim..(in_row + 1) * self.inner_dim]);
            }
            let valid = mask.iter().filter(|&&m| m > 0).count();
            for row in valid..tokens {
                let register_row = row % self.num_registers;
                let base = row * self.inner_dim;
                x[base..base + self.inner_dim].copy_from_slice(
                    &self.registers
                        [register_row * self.inner_dim..(register_row + 1) * self.inner_dim],
                );
            }
        }

        // Connector rope over positions 0..tokens (the index rides as
        // both bounds of the middle-index average).
        let mut positions = vec![0.0f32; tokens * 2];
        for token in 0..tokens {
            positions[token * 2] = token as f32;
            positions[token * 2 + 1] = token as f32;
        }
        let (cos, sin) = super::dit::split_freqs_cis(
            &positions,
            tokens,
            1,
            &[self.connector_max_pos],
            self.connector_theta,
            self.inner_dim,
            self.heads,
            true,
        )?;

        for (attn, ff) in &self.blocks {
            let mut normed = x.clone();
            rms_norm(&mut normed, tokens, self.norm_eps);
            let attended = attn.forward(
                &normed,
                tokens,
                &normed,
                tokens,
                None,
                Some(&cos),
                Some(&sin),
                false,
            );
            for (slot, value) in x.iter_mut().zip(&attended) {
                *slot += value;
            }
            let mut fed = x.clone();
            rms_norm(&mut fed, tokens, self.norm_eps);
            let out = ff.forward(&fed, tokens);
            for (slot, value) in x.iter_mut().zip(&out) {
                *slot += value;
            }
        }
        // The connector ends with the plain rms_norm (eps 1e-6), not a
        // learned LayerNorm.
        rms_norm(&mut x, tokens, 1e-6);
        Ok(x)
    }
}

/// The dramabox text conditioner: hidden states in, cross-attention
/// context out (the mask the reference returns is dropped by
/// `generate`, which deliberately passes no DiT context mask).
pub(crate) struct TextConditioner {
    pub(crate) feature_extractor: FeatureExtractor,
    connector: Embeddings1DConnector,
    hidden_dim: usize,
    num_layers: usize,
}

impl TextConditioner {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load(
        files: &[SafetensorsFile],
        embedding_dim: usize,
        audio_inner_dim: usize,
        num_gemma_layers: usize,
        connector_layers: usize,
        connector_heads: usize,
        connector_head_dim: usize,
        connector_num_registers: usize,
        connector_theta: f64,
        connector_max_pos: f32,
    ) -> Result<Self> {
        // The connector input is the cross dim; the real config keeps
        // connector heads * head_dim equal to it (the register tile in
        // the reference requires it).
        if connector_heads * connector_head_dim != audio_inner_dim {
            return Err(SpeechError::BadConfig {
                field: "audio_cross_attention_dim".into(),
                why: format!(
                    "connector heads * head_dim = {} != cross dim {audio_inner_dim}",
                    connector_heads * connector_head_dim
                ),
            });
        }
        Ok(Self {
            feature_extractor: FeatureExtractor::load(
                files,
                "text_conditioner.feature_extractor",
                embedding_dim,
                audio_inner_dim,
                num_gemma_layers,
            )?,
            connector: Embeddings1DConnector::load(
                files,
                "text_conditioner.audio_connector",
                connector_heads,
                connector_head_dim,
                connector_layers,
                connector_num_registers,
                1e-6,
                connector_theta,
                connector_max_pos,
            )?,
            hidden_dim: embedding_dim,
            num_layers: num_gemma_layers,
        })
    }

    /// `hidden_states [tokens, hidden * layers]` (per token the Gemma
    /// layers concatenated), `attention_mask [tokens]` ->
    /// `[tokens, audio_inner_dim]`.
    pub(crate) fn forward(
        &self,
        hidden_states: &[f32],
        tokens: usize,
        attention_mask: &[i32],
    ) -> Result<Vec<f32>> {
        if tokens * self.hidden_dim * self.num_layers != hidden_states.len() {
            return Err(SpeechError::Input {
                why: format!(
                    "hidden states {} do not cover {tokens} tokens x {} dims x {} layers",
                    hidden_states.len(),
                    self.hidden_dim,
                    self.num_layers
                ),
            });
        }
        let features = self
            .feature_extractor
            .forward(hidden_states, tokens, attention_mask);
        self.connector.forward(&features, tokens, attention_mask)
    }
}
