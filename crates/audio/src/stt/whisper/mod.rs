//! OpenAI Whisper and Distil-Whisper ASR models.
//!
//! Reference: `mlx_audio/stt/models/whisper/whisper.py` (v0.5.7).
//! Architecture:
//! - Log-mel audio frontend ([`turbospark_audio::models::whisper`]), producing
//!   80 or 128 mel bands over 30-second windows.
//! - Audio encoder: two 1D convolutions (kernel size 3, stride 1 and stride 2),
//!   sinusoidal/learned positional embeddings, and residual transformer encoder
//!   blocks (LayerNorm -> multi-head self-attention -> LayerNorm -> MLP).
//! - Text decoder: learned token embeddings and positional embeddings, residual
//!   transformer decoder blocks (LayerNorm -> causal self-attention ->
//!   LayerNorm -> cross-attention over audio encoder features -> LayerNorm -> MLP),
//!   final LayerNorm, and projection to vocabulary logits.
//! - Autoregressive greedy decode starting from `<|startoftranscript|>` until
//!   `<|endoftext|>`.

use std::path::Path;

use crate::ops;
use crate::{Result, SpeechError};

/// SOT and control tokens for the Whisper tokenizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WhisperSpecialTokens {
    pub eot: u32,
    pub sot: u32,
    pub transcribe: u32,
    pub translate: u32,
    pub no_timestamps: u32,
    pub no_speech: u32,
}

impl Default for WhisperSpecialTokens {
    fn default() -> Self {
        Self {
            eot: 50257,
            sot: 50258,
            transcribe: 50359,
            translate: 50358,
            no_timestamps: 50363,
            no_speech: 50362,
        }
    }
}

/// Whisper model architecture parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperConfig {
    pub vocab_size: usize,
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
    pub max_target_positions: usize,
    pub d_model: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub decoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub decoder_ffn_dim: usize,
}

impl WhisperConfig {
    /// Whisper-tiny configuration (39M parameters).
    pub fn tiny() -> Self {
        Self {
            vocab_size: 51864,
            num_mel_bins: 80,
            max_source_positions: 1500,
            max_target_positions: 448,
            d_model: 384,
            encoder_layers: 4,
            decoder_layers: 4,
            encoder_attention_heads: 6,
            decoder_attention_heads: 6,
            encoder_ffn_dim: 1536,
            decoder_ffn_dim: 1536,
        }
    }

    /// Whisper-base configuration (74M parameters).
    pub fn base() -> Self {
        Self {
            vocab_size: 51864,
            num_mel_bins: 80,
            max_source_positions: 1500,
            max_target_positions: 448,
            d_model: 512,
            encoder_layers: 6,
            decoder_layers: 6,
            encoder_attention_heads: 8,
            decoder_attention_heads: 8,
            encoder_ffn_dim: 2048,
            decoder_ffn_dim: 2048,
        }
    }

    /// Whisper-small configuration (244M parameters).
    pub fn small() -> Self {
        Self {
            vocab_size: 51865,
            num_mel_bins: 80,
            max_source_positions: 1500,
            max_target_positions: 448,
            d_model: 768,
            encoder_layers: 12,
            decoder_layers: 12,
            encoder_attention_heads: 12,
            decoder_attention_heads: 12,
            encoder_ffn_dim: 3072,
            decoder_ffn_dim: 3072,
        }
    }

    /// Whisper-large-v3 configuration (1550M parameters).
    pub fn large_v3() -> Self {
        Self {
            vocab_size: 51866,
            num_mel_bins: 128,
            max_source_positions: 1500,
            max_target_positions: 448,
            d_model: 1280,
            encoder_layers: 32,
            decoder_layers: 32,
            encoder_attention_heads: 20,
            decoder_attention_heads: 20,
            encoder_ffn_dim: 5120,
            decoder_ffn_dim: 5120,
        }
    }

    /// Distil-large-v3 configuration (756M parameters, 2 decoder layers).
    pub fn distil_large_v3() -> Self {
        Self {
            vocab_size: 51865,
            num_mel_bins: 128,
            max_source_positions: 1500,
            max_target_positions: 448,
            d_model: 1280,
            encoder_layers: 32,
            decoder_layers: 2,
            encoder_attention_heads: 20,
            decoder_attention_heads: 20,
            encoder_ffn_dim: 5120,
            decoder_ffn_dim: 5120,
        }
    }

    /// Parse configuration from Hugging Face `config.json`.
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let as_usize = |key: &str| -> Option<usize> {
            v.get(key)
                .and_then(|x| x.as_u64())
                .and_then(|n| usize::try_from(n).ok())
        };

        let d_model = as_usize("d_model").unwrap_or(384);
        let config = Self {
            vocab_size: as_usize("vocab_size").unwrap_or(51864),
            num_mel_bins: as_usize("num_mel_bins").unwrap_or(80),
            max_source_positions: as_usize("max_source_positions").unwrap_or(1500),
            max_target_positions: as_usize("max_target_positions").unwrap_or(448),
            d_model,
            encoder_layers: as_usize("encoder_layers").unwrap_or(4),
            decoder_layers: as_usize("decoder_layers").unwrap_or(4),
            encoder_attention_heads: as_usize("encoder_attention_heads").unwrap_or(6),
            decoder_attention_heads: as_usize("decoder_attention_heads").unwrap_or(6),
            encoder_ffn_dim: as_usize("encoder_ffn_dim").unwrap_or(d_model * 4),
            decoder_ffn_dim: as_usize("decoder_ffn_dim").unwrap_or(d_model * 4),
        };
        config.validate()?;
        Ok(config)
    }

    /// Parse configuration from a `config.json` file on disk.
    pub fn from_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| SpeechError::BadConfig {
            field: path.display().to_string(),
            why: format!("failed to read file: {e}"),
        })?;
        let v: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| SpeechError::BadConfig {
                field: path.display().to_string(),
                why: format!("invalid JSON: {e}"),
            })?;
        Self::from_json(&v)
    }

    pub fn validate(&self) -> Result<()> {
        let bad = |field: &str, why: &str| SpeechError::BadConfig {
            field: field.to_string(),
            why: why.to_string(),
        };
        if self.d_model == 0 {
            return Err(bad("d_model", "cannot be 0"));
        }
        if self.encoder_attention_heads == 0 || self.d_model % self.encoder_attention_heads != 0 {
            return Err(bad(
                "encoder_attention_heads",
                "d_model must be divisible by head count",
            ));
        }
        if self.decoder_attention_heads == 0 || self.d_model % self.decoder_attention_heads != 0 {
            return Err(bad(
                "decoder_attention_heads",
                "d_model must be divisible by head count",
            ));
        }
        if self.num_mel_bins != 80 && self.num_mel_bins != 128 {
            return Err(bad("num_mel_bins", "must be 80 or 128"));
        }
        Ok(())
    }
}

/// Transformer encoder layer weights.
#[derive(Clone)]
pub struct EncoderLayerWeights {
    pub self_attn_q: Vec<f32>,
    pub self_attn_k: Vec<f32>,
    pub self_attn_v: Vec<f32>,
    pub self_attn_out: Vec<f32>,
    pub self_attn_q_bias: Vec<f32>,
    pub self_attn_v_bias: Vec<f32>,
    pub self_attn_out_bias: Vec<f32>,
    pub self_attn_layer_norm_w: Vec<f32>,
    pub self_attn_layer_norm_b: Vec<f32>,
    pub fc1_w: Vec<f32>,
    pub fc1_b: Vec<f32>,
    pub fc2_w: Vec<f32>,
    pub fc2_b: Vec<f32>,
    pub final_layer_norm_w: Vec<f32>,
    pub final_layer_norm_b: Vec<f32>,
}

/// Transformer decoder layer weights.
#[derive(Clone)]
pub struct DecoderLayerWeights {
    pub self_attn_q: Vec<f32>,
    pub self_attn_k: Vec<f32>,
    pub self_attn_v: Vec<f32>,
    pub self_attn_out: Vec<f32>,
    pub self_attn_q_bias: Vec<f32>,
    pub self_attn_v_bias: Vec<f32>,
    pub self_attn_out_bias: Vec<f32>,
    pub self_attn_layer_norm_w: Vec<f32>,
    pub self_attn_layer_norm_b: Vec<f32>,
    pub cross_attn_q: Vec<f32>,
    pub cross_attn_k: Vec<f32>,
    pub cross_attn_v: Vec<f32>,
    pub cross_attn_out: Vec<f32>,
    pub cross_attn_q_bias: Vec<f32>,
    pub cross_attn_v_bias: Vec<f32>,
    pub cross_attn_out_bias: Vec<f32>,
    pub cross_attn_layer_norm_w: Vec<f32>,
    pub cross_attn_layer_norm_b: Vec<f32>,
    pub fc1_w: Vec<f32>,
    pub fc1_b: Vec<f32>,
    pub fc2_w: Vec<f32>,
    pub fc2_b: Vec<f32>,
    pub final_layer_norm_w: Vec<f32>,
    pub final_layer_norm_b: Vec<f32>,
}

/// Complete in-memory weights for Whisper.
pub struct WhisperWeights {
    pub conv1_w: Vec<f32>,
    pub conv1_b: Vec<f32>,
    pub conv2_w: Vec<f32>,
    pub conv2_b: Vec<f32>,
    pub encoder_positions: Vec<f32>,
    pub encoder_layers: Vec<EncoderLayerWeights>,
    pub encoder_ln_w: Vec<f32>,
    pub encoder_ln_b: Vec<f32>,
    pub decoder_embed_tokens: Vec<f32>,
    pub decoder_positions: Vec<f32>,
    pub decoder_layers: Vec<DecoderLayerWeights>,
    pub decoder_ln_w: Vec<f32>,
    pub decoder_ln_b: Vec<f32>,
    pub proj_out: Option<Vec<f32>>,
}

fn run_layernorm(
    x: &[f32],
    rows: usize,
    cols: usize,
    w: &[f32],
    b: Option<&[f32]>,
    eps: f32,
) -> Vec<f32> {
    let mut out = x.to_vec();
    ops::layernorm(&mut out, rows, cols, w, b, eps);
    out
}

#[allow(clippy::too_many_arguments)]
fn run_mha(
    q_in: &[f32],
    k_in: &[f32],
    v_in: &[f32],
    q_len: usize,
    kv_len: usize,
    d_model: usize,
    heads: usize,
    mask: Option<&[f32]>,
) -> Vec<f32> {
    let head_dim = d_model / heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; q_len * d_model];
    let mut q_head = vec![0.0f32; q_len * head_dim];
    let mut k_head = vec![0.0f32; kv_len * head_dim];
    let mut v_head = vec![0.0f32; kv_len * head_dim];

    for h in 0..heads {
        for t in 0..q_len {
            q_head[t * head_dim..(t + 1) * head_dim].copy_from_slice(
                &q_in[t * d_model + h * head_dim..t * d_model + (h + 1) * head_dim],
            );
        }
        for t in 0..kv_len {
            k_head[t * head_dim..(t + 1) * head_dim].copy_from_slice(
                &k_in[t * d_model + h * head_dim..t * d_model + (h + 1) * head_dim],
            );
            v_head[t * head_dim..(t + 1) * head_dim].copy_from_slice(
                &v_in[t * d_model + h * head_dim..t * d_model + (h + 1) * head_dim],
            );
        }
        let o = ops::sdpa(
            &q_head, &k_head, &v_head, mask, q_len, kv_len, head_dim, head_dim, scale,
        );
        for t in 0..q_len {
            out[t * d_model + h * head_dim..t * d_model + (h + 1) * head_dim]
                .copy_from_slice(&o[t * head_dim..(t + 1) * head_dim]);
        }
    }
    out
}

/// The portable Whisper model runner.
pub struct WhisperModel {
    pub config: WhisperConfig,
    pub weights: WhisperWeights,
    pub specials: WhisperSpecialTokens,
}

impl WhisperModel {
    pub fn new(
        config: WhisperConfig,
        weights: WhisperWeights,
        specials: WhisperSpecialTokens,
    ) -> Self {
        Self {
            config,
            weights,
            specials,
        }
    }

    /// Encode log-mel frames into audio context features `[1500, d_model]`.
    pub fn encode(&self, mel_3000: &[Vec<f32>]) -> Result<Vec<f32>> {
        let n_mels = self.config.num_mel_bins;
        let d_model = self.config.d_model;
        let time_steps = mel_3000.len();

        if time_steps != 3000 {
            return Err(SpeechError::Input {
                why: format!("expected 3000 mel frames, got {time_steps}"),
            });
        }

        // Reshape mel to [n_mels, 3000] for 1D convolution
        let mut mel_planar = vec![0.0f32; n_mels * 3000];
        for (t, frame) in mel_3000.iter().enumerate() {
            for (m, &val) in frame.iter().enumerate() {
                mel_planar[m * 3000 + t] = val;
            }
        }

        // Conv1: in_channels = n_mels, out_channels = d_model, kernel = 3, stride = 1, pad = 1
        let mut x = ops::conv1d(
            &mel_planar,
            &self.weights.conv1_w,
            Some(&self.weights.conv1_b),
            n_mels,
            d_model,
            3,
            1,
            1,
            1,
            1,
        );
        ops::gelu_erf(&mut x);

        // Conv2: in_channels = d_model, out_channels = d_model, kernel = 3, stride = 2, pad = 1
        let mut x2 = ops::conv1d(
            &x,
            &self.weights.conv2_w,
            Some(&self.weights.conv2_b),
            d_model,
            d_model,
            3,
            2,
            1,
            1,
            1,
        );
        ops::gelu_erf(&mut x2);

        let out_steps = 1500;
        // Transpose from [d_model, 1500] to row-major [1500, d_model] and add positional embedding
        let mut hidden = vec![0.0f32; out_steps * d_model];
        for t in 0..out_steps {
            for c in 0..d_model {
                let pos_emb = self.weights.encoder_positions[t * d_model + c];
                hidden[t * d_model + c] = x2[c * out_steps + t] + pos_emb;
            }
        }

        let heads = self.config.encoder_attention_heads;

        // Encoder transformer layers
        for layer in &self.weights.encoder_layers {
            let norm = run_layernorm(
                &hidden,
                out_steps,
                d_model,
                &layer.self_attn_layer_norm_w,
                Some(&layer.self_attn_layer_norm_b),
                1e-5,
            );

            let q = ops::linear(
                &norm,
                &layer.self_attn_q,
                Some(&layer.self_attn_q_bias),
                out_steps,
                d_model,
                d_model,
            );
            let k = ops::linear(&norm, &layer.self_attn_k, None, out_steps, d_model, d_model);
            let v = ops::linear(
                &norm,
                &layer.self_attn_v,
                Some(&layer.self_attn_v_bias),
                out_steps,
                d_model,
                d_model,
            );

            let attn_out = run_mha(&q, &k, &v, out_steps, out_steps, d_model, heads, None);
            let proj_out = ops::linear(
                &attn_out,
                &layer.self_attn_out,
                Some(&layer.self_attn_out_bias),
                out_steps,
                d_model,
                d_model,
            );

            for i in 0..hidden.len() {
                hidden[i] += proj_out[i];
            }

            let ffn_norm = run_layernorm(
                &hidden,
                out_steps,
                d_model,
                &layer.final_layer_norm_w,
                Some(&layer.final_layer_norm_b),
                1e-5,
            );
            let mut ffn1 = ops::linear(
                &ffn_norm,
                &layer.fc1_w,
                Some(&layer.fc1_b),
                out_steps,
                d_model,
                self.config.encoder_ffn_dim,
            );
            ops::gelu_erf(&mut ffn1);
            let ffn2 = ops::linear(
                &ffn1,
                &layer.fc2_w,
                Some(&layer.fc2_b),
                out_steps,
                self.config.encoder_ffn_dim,
                d_model,
            );

            for i in 0..hidden.len() {
                hidden[i] += ffn2[i];
            }
        }

        // Final LayerNorm
        Ok(run_layernorm(
            &hidden,
            out_steps,
            d_model,
            &self.weights.encoder_ln_w,
            Some(&self.weights.encoder_ln_b),
            1e-5,
        ))
    }

    /// Compute next token logits given text tokens and pre-computed audio features.
    pub fn decode_step(&self, tokens: &[u32], audio_features: &[f32]) -> Result<Vec<f32>> {
        let seq_len = tokens.len();
        let d_model = self.config.d_model;
        let audio_len = self.config.max_source_positions;

        if seq_len == 0 || seq_len > self.config.max_target_positions {
            return Err(SpeechError::Input {
                why: format!("invalid sequence length {seq_len}"),
            });
        }

        // Embed tokens + positional embeddings
        let mut hidden = vec![0.0f32; seq_len * d_model];
        for (pos, &tok) in tokens.iter().enumerate() {
            let tok_idx = tok as usize;
            if tok_idx >= self.config.vocab_size {
                return Err(SpeechError::Input {
                    why: format!("token {tok} out of vocab bounds"),
                });
            }
            for d in 0..d_model {
                let token_emb = self.weights.decoder_embed_tokens[tok_idx * d_model + d];
                let pos_emb = self.weights.decoder_positions[pos * d_model + d];
                hidden[pos * d_model + d] = token_emb + pos_emb;
            }
        }

        let heads = self.config.decoder_attention_heads;

        // Construct causal mask
        let mut causal_mask = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                causal_mask[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }

        // Decoder transformer layers
        for layer in &self.weights.decoder_layers {
            let norm = run_layernorm(
                &hidden,
                seq_len,
                d_model,
                &layer.self_attn_layer_norm_w,
                Some(&layer.self_attn_layer_norm_b),
                1e-5,
            );
            let q = ops::linear(
                &norm,
                &layer.self_attn_q,
                Some(&layer.self_attn_q_bias),
                seq_len,
                d_model,
                d_model,
            );
            let k = ops::linear(&norm, &layer.self_attn_k, None, seq_len, d_model, d_model);
            let v = ops::linear(
                &norm,
                &layer.self_attn_v,
                Some(&layer.self_attn_v_bias),
                seq_len,
                d_model,
                d_model,
            );

            let self_attn = run_mha(
                &q,
                &k,
                &v,
                seq_len,
                seq_len,
                d_model,
                heads,
                Some(&causal_mask),
            );
            let self_proj = ops::linear(
                &self_attn,
                &layer.self_attn_out,
                Some(&layer.self_attn_out_bias),
                seq_len,
                d_model,
                d_model,
            );

            for i in 0..hidden.len() {
                hidden[i] += self_proj[i];
            }

            // Cross-attention over audio encoder features
            let cross_norm = run_layernorm(
                &hidden,
                seq_len,
                d_model,
                &layer.cross_attn_layer_norm_w,
                Some(&layer.cross_attn_layer_norm_b),
                1e-5,
            );
            let cq = ops::linear(
                &cross_norm,
                &layer.cross_attn_q,
                Some(&layer.cross_attn_q_bias),
                seq_len,
                d_model,
                d_model,
            );
            let ck = ops::linear(
                audio_features,
                &layer.cross_attn_k,
                None,
                audio_len,
                d_model,
                d_model,
            );
            let cv = ops::linear(
                audio_features,
                &layer.cross_attn_v,
                Some(&layer.cross_attn_v_bias),
                audio_len,
                d_model,
                d_model,
            );

            let cross_attn = run_mha(&cq, &ck, &cv, seq_len, audio_len, d_model, heads, None);
            let cross_proj = ops::linear(
                &cross_attn,
                &layer.cross_attn_out,
                Some(&layer.cross_attn_out_bias),
                seq_len,
                d_model,
                d_model,
            );

            for i in 0..hidden.len() {
                hidden[i] += cross_proj[i];
            }

            let ffn_norm = run_layernorm(
                &hidden,
                seq_len,
                d_model,
                &layer.final_layer_norm_w,
                Some(&layer.final_layer_norm_b),
                1e-5,
            );
            let mut ffn1 = ops::linear(
                &ffn_norm,
                &layer.fc1_w,
                Some(&layer.fc1_b),
                seq_len,
                d_model,
                self.config.decoder_ffn_dim,
            );
            ops::gelu_erf(&mut ffn1);
            let ffn2 = ops::linear(
                &ffn1,
                &layer.fc2_w,
                Some(&layer.fc2_b),
                seq_len,
                self.config.decoder_ffn_dim,
                d_model,
            );

            for i in 0..hidden.len() {
                hidden[i] += ffn2[i];
            }
        }

        // Final norm on last position
        let last_hidden = &hidden[(seq_len - 1) * d_model..seq_len * d_model];
        let normed_last = run_layernorm(
            last_hidden,
            1,
            d_model,
            &self.weights.decoder_ln_w,
            Some(&self.weights.decoder_ln_b),
            1e-5,
        );

        // Project to vocab logits
        let logits = if let Some(proj) = &self.weights.proj_out {
            ops::linear(&normed_last, proj, None, 1, d_model, self.config.vocab_size)
        } else {
            // Tied with decoder token embeddings
            ops::matmul(
                &normed_last,
                &self.weights.decoder_embed_tokens,
                1,
                d_model,
                self.config.vocab_size,
            )
        };

        Ok(logits)
    }

    /// Autoregressive greedy transcribe of 16 kHz audio samples.
    pub fn transcribe(&self, pcm_16k: &[f32], max_tokens: usize) -> Result<Vec<u32>> {
        let mut window = pcm_16k.to_vec();
        window.resize(turbospark_audio::whisper::WHISPER_WINDOW_SAMPLES, 0.0);

        let mel =
            turbospark_audio::whisper::whisper_log_mel_window(&window, self.config.num_mel_bins)?;

        let audio_features = self.encode(&mel)?;

        let mut tokens = vec![
            self.specials.sot,
            50259, // default language: English <|en|>
            self.specials.transcribe,
            self.specials.no_timestamps,
        ];

        let start_len = tokens.len();
        for _ in 0..max_tokens {
            let logits = self.decode_step(&tokens, &audio_features)?;
            let mut best_tok = 0u32;
            let mut best_val = f32::NEG_INFINITY;
            for (idx, &val) in logits.iter().enumerate() {
                if val > best_val {
                    best_val = val;
                    best_tok = idx as u32;
                }
            }

            if best_tok == self.specials.eot {
                break;
            }
            tokens.push(best_tok);
        }

        Ok(tokens[start_len..].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whisper_config_presets() {
        let tiny = WhisperConfig::tiny();
        assert_eq!(tiny.num_mel_bins, 80);
        assert_eq!(tiny.d_model, 384);
        assert_eq!(tiny.encoder_layers, 4);
        assert!(tiny.validate().is_ok());

        let base = WhisperConfig::base();
        assert_eq!(base.d_model, 512);
        assert_eq!(base.encoder_layers, 6);
        assert!(base.validate().is_ok());

        let large = WhisperConfig::large_v3();
        assert_eq!(large.num_mel_bins, 128);
        assert_eq!(large.d_model, 1280);
        assert_eq!(large.encoder_layers, 32);
        assert!(large.validate().is_ok());

        let distil = WhisperConfig::distil_large_v3();
        assert_eq!(distil.num_mel_bins, 128);
        assert_eq!(distil.decoder_layers, 2);
        assert!(distil.validate().is_ok());
    }

    #[test]
    fn test_whisper_config_validation() {
        let mut cfg = WhisperConfig::tiny();
        cfg.num_mel_bins = 64;
        assert!(cfg.validate().is_err());

        let mut cfg = WhisperConfig::tiny();
        cfg.encoder_attention_heads = 5; // 384 not divisible by 5
        assert!(cfg.validate().is_err());
    }
}
