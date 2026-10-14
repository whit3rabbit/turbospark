//! VoxCPM2 (2B, tokenizer-free, 48 kHz output).
//!
//! Port of `mlx_audio/tts/models/voxcpm2/` at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/voxcpm2)
//! (MLX 0.32.3). The voxcpm architecture with the V2 changes: the
//! conditioning concatenation replaces the additions (a
//! `fusion_concat_proj` on the residual side and a 2-wide `mu` token
//! pair in the DiT), the prompt grammar gains reference-audio
//! `[ref_start 103 | patches | ref_end 104]` prefixes and continuation
//! prompts with warmup patches and a decode-front trim, the stop head
//! gates on `real_steps > min_tokens` instead of the fixed `i > 5`,
//! the residual LM can drop rope (`no_rope`), and the AudioVAE decodes
//! with per-block sample-rate conditioning (16 kHz encode hop, 48 kHz
//! output). The MiniCPM backbone, flow solver, and causal-conv stack
//! are shared with [voxcpm](../voxcpm/index.html) (additive
//! `kv_channels` / `no_rope` there).
//!
//! Scope: single batch, non-streaming (the `streaming_prefix_len`
//! wiring only feeds the continuation prefix and the decode trim),
//! caller-owned text tokenization (the reference `_tokenize` plus the
//! multi-char Chinese split; the `(instruct)` voice-design wrapper is
//! a caller string concat). Zero-shot, reference-clone, continuation,
//! and combined prompts are ported. See [README.md](README.md).

use serde_json::Value;

use super::voxcpm::minicpm::MiniCpmModel;
use super::voxcpm::minicpm::{Config as MiniCpmConfig, Mask};
use super::voxcpm::FlowNoise;
use crate::lm::Weights;
use crate::nn::Linear;
use crate::{quant, Result, SpeechError};

mod audio_vae;
mod dit;

#[cfg(test)]
mod tests;

pub use audio_vae::AudioVaeConfig;

/// The reference special tokens.
pub const AUDIO_START_TOKEN: i32 = 101;
pub const REF_AUDIO_START_TOKEN: i32 = 103;
pub const REF_AUDIO_END_TOKEN: i32 = 104;

/// The encoder/diT sub-configs. `dit_mean_mode` (the config's
/// `mean_mode`) selects live delta-time values in the solver.
#[derive(Debug, Clone)]
pub struct SubConfig {
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub kv_channels: Option<usize>,
    pub dit_mean_mode: bool,
    pub solver: String,
}

impl SubConfig {
    fn from_json(value: &Value) -> Self {
        let i = |key: &str, default: usize| {
            value
                .get(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(default)
        };
        // `mean_mode` normalizes to `dit_mean_mode` in from_dict.
        let mean_mode = value
            .get("dit_mean_mode")
            .or_else(|| value.get("mean_mode"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        Self {
            hidden_dim: i("hidden_dim", 1024),
            ffn_dim: i("ffn_dim", 4096),
            num_heads: i("num_heads", 16),
            num_layers: i("num_layers", 4),
            kv_channels: value
                .get("kv_channels")
                .and_then(Value::as_u64)
                .map(|n| n as usize),
            dit_mean_mode: mean_mode,
            solver: value
                .get("cfm_config")
                .and_then(|c| c.get("solver"))
                .and_then(Value::as_str)
                .unwrap_or("euler")
                .to_owned(),
        }
    }
}

/// The resolved model config (`ModelArgs.from_dict`).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub lm_config: MiniCpmConfig,
    pub encoder_config: SubConfig,
    pub dit_config: SubConfig,
    pub audio_vae_config: AudioVaeConfig,
    pub patch_size: usize,
    pub feat_dim: usize,
    pub scalar_quantization_latent_dim: usize,
    pub scalar_quantization_scale: i32,
    pub residual_lm_num_layers: usize,
    pub residual_lm_no_rope: bool,
    pub max_length: usize,
}

impl ModelConfig {
    pub fn from_json(value: &Value) -> Result<Self> {
        let mut lm_value = value.get("lm_config").cloned().unwrap_or(Value::Null);
        if let Some(object) = lm_value.as_object_mut() {
            if let Some(rope_scaling) = object.remove("rope_scaling") {
                let get = |key: &str| rope_scaling.get(key).cloned().unwrap_or(Value::Null);
                if !object.contains_key("rope_scaling_type") {
                    let rope_type = match get("type") {
                        Value::String(text) => Value::String(text),
                        _ => Value::String("longrope".into()),
                    };
                    object.insert("rope_scaling_type".into(), rope_type);
                }
                object.insert("rope_long_factor".into(), get("long_factor"));
                object.insert("rope_short_factor".into(), get("short_factor"));
                if !object.contains_key("original_max_position_embeddings") {
                    object.insert(
                        "original_max_position_embeddings".into(),
                        get("original_max_position_embeddings"),
                    );
                }
            }
        }
        // The voxcpm2 LMConfig defaults `use_mup` to true (voxcpm to
        // false); restore that default when the key is absent.
        let lm_config = MiniCpmConfig::from_json(&lm_value);
        let lm_config = if value
            .get("lm_config")
            .and_then(|lm| lm.get("use_mup"))
            .is_none()
        {
            let mut config = lm_config;
            config.use_mup = true;
            config
        } else {
            lm_config
        };
        let i = |key: &str, default: usize| {
            value
                .get(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(default)
        };
        Ok(Self {
            encoder_config: SubConfig::from_json(
                value.get("encoder_config").unwrap_or(&Value::Null),
            ),
            dit_config: SubConfig::from_json(value.get("dit_config").unwrap_or(&Value::Null)),
            audio_vae_config: AudioVaeConfig::from_json(
                value.get("audio_vae_config").unwrap_or(&Value::Null),
            ),
            patch_size: i("patch_size", 4),
            feat_dim: i("feat_dim", 64),
            scalar_quantization_latent_dim: i("scalar_quantization_latent_dim", 512),
            scalar_quantization_scale: value
                .get("scalar_quantization_scale")
                .and_then(Value::as_i64)
                .map(|n| n as i32)
                .unwrap_or(9),
            residual_lm_num_layers: i("residual_lm_num_layers", 8),
            residual_lm_no_rope: value
                .get("residual_lm_no_rope")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            max_length: i("max_length", 8192),
            lm_config,
        })
    }

    /// The residual LM backbone config: base fields, the residual
    /// layer count, no vocabulary, optional no-rope.
    fn residual_config(&self) -> MiniCpmConfig {
        let mut config = self.lm_config.clone();
        config.num_hidden_layers = self.residual_lm_num_layers;
        config.vocab_size = 0;
        config.no_rope = self.residual_lm_no_rope;
        config
    }

    /// The patch encoder backbone config.
    fn encoder_backbone_config(&self) -> MiniCpmConfig {
        let mut config = self.lm_config.with_overrides(
            self.encoder_config.hidden_dim,
            self.encoder_config.ffn_dim,
            self.encoder_config.num_heads,
            self.encoder_config.num_layers,
        );
        config.kv_channels = self.encoder_config.kv_channels;
        config
    }

    /// The DiT denoiser backbone config.
    fn dit_backbone_config(&self) -> MiniCpmConfig {
        let mut config = self.lm_config.with_overrides(
            self.dit_config.hidden_dim,
            self.dit_config.ffn_dim,
            self.dit_config.num_heads,
            self.dit_config.num_layers,
        );
        config.kv_channels = self.dit_config.kv_channels;
        config
    }
}

/// `ScalarQuantizationLayer` (same as the voxcpm one).
struct ScalarQuantization {
    in_proj: Linear,
    out_proj: Linear,
    scale: f32,
}

impl ScalarQuantization {
    fn load(weights: &Weights, config: &ModelConfig) -> Result<Self> {
        let hidden = config.lm_config.hidden_size;
        let latent = config.scalar_quantization_latent_dim;
        Ok(Self {
            in_proj: weights.linear(
                "fsq_layer.in_proj",
                hidden,
                latent,
                crate::lm::Bias::Required,
            )?,
            out_proj: weights.linear(
                "fsq_layer.out_proj",
                latent,
                hidden,
                crate::lm::Bias::Required,
            )?,
            scale: config.scalar_quantization_scale as f32,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut h = self.in_proj.forward(x, rows);
        for value in &mut h {
            *value = (*value).tanh();
            *value = ((*value * self.scale).round()) / self.scale;
        }
        self.out_proj.forward(&h, rows)
    }
}

/// A continuation prompt: the text prefix ids and waveform (encoded
/// left-padded). `prompt_text + text` is tokenized as one string by
/// the caller, like the reference.
#[derive(Debug, Clone, Default)]
pub struct ContinuationPrompt {
    pub prompt_audio: Vec<f32>,
}

/// A reference-clone prompt: the reference waveform (encoded
/// right-padded). The reference text (if any) is part of the caller's
/// combined encode.
#[derive(Debug, Clone, Default)]
pub struct VoicePrompt {
    pub ref_audio: Vec<f32>,
}

/// `generate` parameters.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub max_tokens: usize,
    /// Generated-but-discarded conditioning patches (non-continuation
    /// modes only).
    pub warmup_patches: usize,
    /// The stop head gates on `real_steps > min_tokens`.
    pub min_tokens: usize,
    pub inference_timesteps: usize,
    pub cfg_value: f32,
    /// The continuation prefix length (patches kept from the prompt
    /// audio and trimmed from the decode front).
    pub streaming_prefix_len: usize,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: 2000,
            warmup_patches: 0,
            min_tokens: 2,
            inference_timesteps: 10,
            cfg_value: 2.0,
            streaming_prefix_len: 4,
        }
    }
}

/// One generation result.
#[derive(Debug, Clone)]
pub struct Generated {
    pub samples: Vec<f32>,
    /// The output sample rate (`out_sample_rate`, 48 kHz for the real
    /// model).
    pub sample_rate: u32,
    /// The reference `token_count`: the caller text ids (before the
    /// start token).
    pub token_count: usize,
    /// The collected patches `[steps * patch, feat_dim]` (after
    /// warmup/continuation assembly), the decoder input.
    pub latents: Vec<f32>,
    pub steps: usize,
    pub stopped: bool,
    /// Whether the run continued a prompt (prefix trimmed from the
    /// decode front).
    pub continued: bool,
}

/// The loaded VoxCPM2 model.
pub struct Model {
    pub config: ModelConfig,
    base_lm: MiniCpmModel,
    residual_lm: MiniCpmModel,
    feat_encoder: super::voxcpm::encoder::VoxCpmLocEnc,
    feat_decoder: dit::UnifiedCfm,
    fsq: ScalarQuantization,
    enc_to_lm_proj: Linear,
    lm_to_dit_proj: Linear,
    res_to_dit_proj: Linear,
    fusion_concat_proj: Linear,
    stop_proj: Linear,
    stop_head: Linear,
    audio_vae: audio_vae::AudioVae,
    lm_cache: Vec<crate::lm::cache::KvCache>,
    res_cache: Vec<crate::lm::cache::KvCache>,
}

impl Model {
    /// Opens a checkpoint directory (`config.json` + safetensors).
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        let config = ModelConfig::from_json(&config_value)?;
        let weights = Weights::open(dir)?;

        if config.dit_config.solver != "euler" {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "cfm solver {:?}; the port implements the reference euler path",
                    config.dit_config.solver
                ),
            });
        }
        if config.audio_vae_config.use_noise_block {
            return Err(SpeechError::Unsupported {
                why:
                    "audio_vae_config.use_noise_block is enabled; the noise block draws fresh mx.random.normal values per decode and has no deterministic Rust counterpart at this commit"
                        .into(),
            });
        }
        let hidden = config.lm_config.hidden_size;
        let base_lm = MiniCpmModel::load(&weights, "base_lm.", config.lm_config.clone())?;
        let residual_lm = MiniCpmModel::load(&weights, "residual_lm.", config.residual_config())?;
        let feat_encoder = super::voxcpm::encoder::VoxCpmLocEnc::load(
            &weights,
            "feat_encoder.",
            config.feat_dim,
            &config.encoder_backbone_config(),
        )?;
        let feat_decoder = dit::UnifiedCfm::load(
            &weights,
            "feat_decoder.",
            config.feat_dim,
            &config.dit_backbone_config(),
            config.dit_config.dit_mean_mode,
        )?;
        let fsq = ScalarQuantization::load(&weights, &config)?;
        let biased = |name: &str, input: usize, output: usize| -> Result<Linear> {
            weights.linear(name, input, output, crate::lm::Bias::Required)
        };
        let enc_to_lm_proj = biased("enc_to_lm_proj", config.encoder_config.hidden_dim, hidden)?;
        let lm_to_dit_proj = biased("lm_to_dit_proj", hidden, config.dit_config.hidden_dim)?;
        let res_to_dit_proj = biased("res_to_dit_proj", hidden, config.dit_config.hidden_dim)?;
        // The fusion projection takes the concatenated [hidden | hidden]
        // pair and returns one hidden width.
        let fusion_concat_proj = biased("fusion_concat_proj", hidden * 2, hidden)?;
        let stop_proj = biased("stop_proj", hidden, hidden)?;
        let stop_head = weights.linear("stop_head", hidden, 2, crate::lm::Bias::None)?;
        let audio_vae =
            audio_vae::AudioVae::load(weights.shards(), "audio_vae.", &config.audio_vae_config)?;
        let layers = |count: usize| crate::lm::make_prompt_cache(count);
        Ok(Self {
            lm_cache: layers(config.lm_config.num_hidden_layers),
            res_cache: layers(config.residual_lm_num_layers),
            base_lm,
            residual_lm,
            feat_encoder,
            feat_decoder,
            fsq,
            enc_to_lm_proj,
            lm_to_dit_proj,
            res_to_dit_proj,
            fusion_concat_proj,
            stop_proj,
            stop_head,
            audio_vae,
            config,
        })
    }

    /// The output sample rate (`out_sample_rate`).
    pub fn sample_rate(&self) -> u32 {
        self.audio_vae.out_sample_rate()
    }

    /// `_encode_wav` on an already-loaded waveform. The reference
    /// contract: an array input is at the OUTPUT sample rate and gets
    /// the scipy Fourier resample down to the encoder rate (16 kHz)
    /// before the patch padding (`right` or `left`) and the VAE
    /// encode. Returns `(rows, [rows * patch * latent_dim])`
    /// row-major.
    pub fn encode_wav(&mut self, audio: &[f32], padding_mode: &str) -> Result<(usize, Vec<f32>)> {
        let patch = self.config.patch_size;
        let latent = self.config.audio_vae_config.latent_dim;
        let encode_rate = self.config.audio_vae_config.sample_rate as f64;
        let out_rate = self.config.audio_vae_config.out_sample_rate as f64;
        let resampled;
        let audio: &[f32] = if (out_rate - encode_rate).abs() > f64::EPSILON {
            let num = (audio.len() as f64 * encode_rate / out_rate) as usize;
            resampled = audio_vae::fourier_resample(audio, num)?;
            &resampled
        } else {
            audio
        };
        let patch_len = patch * self.audio_vae.chunk_size();
        let padded;
        let samples: &[f32] = if audio.len() % patch_len != 0 {
            let pad = patch_len - audio.len() % patch_len;
            if padding_mode == "left" {
                let mut buffer = vec![0.0f32; pad];
                buffer.extend_from_slice(audio);
                padded = buffer;
            } else {
                let mut buffer = audio.to_vec();
                buffer.resize(audio.len() + pad, 0.0);
                padded = buffer;
            }
            &padded
        } else {
            audio
        };
        let latents = self.audio_vae.encode(samples);
        let frames = latents.len() / latent;
        let rows = frames / patch;
        let flat = super::voxcpm::transpose(&latents, latent, frames);
        Ok((rows, flat[..rows * patch * latent].to_vec()))
    }

    /// The reference `generate` over one combined text encode.
    /// `text_token_ids` is the caller-encoded text WITHOUT the start
    /// token: the bare `text` in zero-shot mode, `prompt_text + text`
    /// for a continuation, and `ref_text + text` combined with the
    /// reference audio (the reference encodes each string once).
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &mut self,
        text_token_ids: &[i32],
        params: &GenerateParams,
        noise: &mut dyn FlowNoise,
        voice: Option<&VoicePrompt>,
        continuation: Option<&ContinuationPrompt>,
    ) -> Result<Generated> {
        if params.max_tokens == 0 {
            return Err(SpeechError::Input {
                why: "max_tokens must be at least 1".into(),
            });
        }
        if params.inference_timesteps < 1 {
            return Err(SpeechError::Input {
                why: "inference_timesteps must be at least 1".into(),
            });
        }
        if voice.is_none() && continuation.is_none() && params.warmup_patches > 0 {
            return Err(SpeechError::Input {
                why: "warmup_patches apply only to reference/continuation prompts".into(),
            });
        }
        let hidden = self.config.lm_config.hidden_size;
        let patch = self.config.patch_size;
        let latent = self.config.audio_vae_config.latent_dim;
        // The reference scale embed: `scale_emb if use_mup else 1.0`
        // (inverted from voxcpm).
        let scale_emb = if self.config.lm_config.use_mup {
            self.config.lm_config.scale_emb as f32
        } else {
            1.0
        };

        for cache in self.lm_cache.iter_mut() {
            cache.reset();
        }
        for cache in self.res_cache.iter_mut() {
            cache.reset();
        }

        let prompt = self.build_prompt(
            text_token_ids,
            voice,
            continuation,
            scale_emb,
            params.streaming_prefix_len,
        )?;
        let total_length = prompt.combined.len() / hidden;

        let mut enc_outputs = self.base_lm.forward(
            &prompt.combined,
            total_length,
            Some(&mut self.lm_cache),
            Mask::Causal,
        )?;
        // fsq on audio rows, identity on text rows.
        let fsq_rows = self.fsq.forward(&enc_outputs, total_length);
        for row in 0..total_length {
            if prompt.audio_mask[row] == 0.0 {
                continue;
            }
            for col in 0..hidden {
                enc_outputs[row * hidden + col] = fsq_rows[row * hidden + col];
            }
        }
        let mut lm_hidden = enc_outputs[(total_length - 1) * hidden..].to_vec();

        // V2 residual input: fusion over [enc_outputs | audio_mask * feat].
        let mut fusion_in = vec![0.0f32; total_length * hidden * 2];
        for row in 0..total_length {
            for col in 0..hidden {
                fusion_in[row * hidden * 2 + col] = enc_outputs[row * hidden + col];
                fusion_in[row * hidden * 2 + hidden + col] =
                    prompt.audio_mask[row] * prompt.feat_embed[row * hidden + col];
            }
        }
        let residual_input = self.fusion_concat_proj.forward(&fusion_in, total_length);
        let residual_out = self.residual_lm.forward(
            &residual_input,
            total_length,
            Some(&mut self.res_cache),
            Mask::Causal,
        )?;
        let mut residual_hidden = residual_out[(total_length - 1) * hidden..].to_vec();

        let mut pred_feat_seq: Vec<f32> = prompt.continuation_prefix;
        let continued = prompt.continued;
        let warmup = if continued { 0 } else { params.warmup_patches };

        let mut prefix_feat_cond = prompt.prefix_feat_cond;
        let mut steps = 0usize;
        let mut stopped = false;

        for index in 0..(params.max_tokens + warmup) {
            // V2 DiT conditioning: the CONCATENATION of the two
            // projections (a 2-wide mu token pair).
            let dit_h = {
                let lm_part = self.lm_to_dit_proj.forward(&lm_hidden, 1);
                let res_part = self.res_to_dit_proj.forward(&residual_hidden, 1);
                let mut pair = lm_part;
                pair.extend_from_slice(&res_part);
                pair
            };
            let cond_in = super::voxcpm::transpose(&prefix_feat_cond, patch, latent);
            let pred_feat = self.feat_decoder.sample(
                &dit_h,
                params.inference_timesteps,
                patch,
                &cond_in,
                params.cfg_value,
                noise,
            )?;
            let pred = super::voxcpm::transpose(&pred_feat, latent, patch);
            if index >= warmup {
                pred_feat_seq.extend_from_slice(&pred);
            }

            let curr_embed = {
                let encoded = self.feat_encoder.forward(&pred, 1, patch)?;
                self.enc_to_lm_proj.forward(&encoded, 1)
            };

            let stop_flag = {
                let mut gated = self.stop_proj.forward(&lm_hidden, 1);
                for value in &mut gated {
                    *value *= 1.0 / (1.0 + (-*value).exp());
                }
                let logits = self.stop_head.forward(&gated, 1);
                crate::tts::sampling::argmax(&logits)
            };
            let real_steps = index.saturating_sub(warmup);
            if real_steps > params.min_tokens && stop_flag == 1 {
                stopped = true;
                steps = index + 1 - warmup;
                break;
            }

            let new_lm_out =
                self.base_lm
                    .forward(&curr_embed, 1, Some(&mut self.lm_cache), Mask::Causal)?;
            let mut lm_hidden_next = new_lm_out[(new_lm_out.len() - hidden)..].to_vec();
            lm_hidden_next = self.fsq.forward(&lm_hidden_next, 1);

            // V2 residual step: fusion over [lm_hidden | curr_embed].
            let mut step_fusion = vec![0.0f32; hidden * 2];
            step_fusion[..hidden].copy_from_slice(&lm_hidden_next);
            step_fusion[hidden..].copy_from_slice(&curr_embed);
            let curr_residual_input = self.fusion_concat_proj.forward(&step_fusion, 1);
            let new_res_out = self.residual_lm.forward(
                &curr_residual_input,
                1,
                Some(&mut self.res_cache),
                Mask::Causal,
            )?;
            residual_hidden = new_res_out[(new_res_out.len() - hidden)..].to_vec();

            lm_hidden = lm_hidden_next;
            prefix_feat_cond = pred;
        }
        if !stopped {
            steps = params.max_tokens;
        }

        // Decode all collected patches with the output sample-rate
        // conditioning, then trim the continuation prefix samples.
        let total_frames = pred_feat_seq.len() / latent;
        let latent_input = super::voxcpm::transpose(&pred_feat_seq, total_frames, latent);
        let mut samples = self.audio_vae.decode(&latent_input);
        if continued {
            let trim = patch
                * self.audio_vae.decode_chunk_size()
                * (params.streaming_prefix_len.saturating_sub(1));
            if trim < samples.len() {
                samples.drain(..trim);
            }
        }
        Ok(Generated {
            samples,
            sample_rate: self.sample_rate(),
            token_count: text_token_ids.len(),
            latents: pred_feat_seq,
            steps,
            stopped,
            continued,
        })
    }

    /// Builds the prompt embeddings, masks, and initial condition for
    /// the four reference modes.
    #[allow(clippy::too_many_arguments)]
    fn build_prompt(
        &mut self,
        text_token_ids: &[i32],
        voice: Option<&VoicePrompt>,
        continuation: Option<&ContinuationPrompt>,
        scale_emb: f32,
        streaming_prefix_len: usize,
    ) -> Result<PromptBundle> {
        let hidden = self.config.lm_config.hidden_size;
        let patch = self.config.patch_size;
        let latent = self.config.audio_vae_config.latent_dim;

        // Segments accumulate as (text rows, audio rows, patches).
        let mut text_token: Vec<i32> = Vec::new();
        let mut audio_feat: Vec<f32> = Vec::new();
        let mut text_mask: Vec<f32> = Vec::new();
        let mut audio_mask: Vec<f32> = Vec::new();
        let zero_patch = vec![0.0f32; patch * latent];

        let ref_rows = match voice {
            Some(voice) => {
                let (rows, patches) = self.encode_wav(&voice.ref_audio, "right")?;
                // [ref_start | zeros | ref_end] tokens with
                // [z1 | patches | z1] features.
                text_token.push(REF_AUDIO_START_TOKEN);
                text_token.extend(std::iter::repeat_n(0i32, rows));
                text_token.push(REF_AUDIO_END_TOKEN);
                text_mask.push(1.0);
                text_mask.extend(std::iter::repeat_n(0.0f32, rows));
                text_mask.push(1.0);
                audio_mask.push(0.0);
                audio_mask.extend(std::iter::repeat_n(1.0f32, rows));
                audio_mask.push(0.0);
                audio_feat.extend_from_slice(&zero_patch);
                audio_feat.extend_from_slice(&patches);
                audio_feat.extend_from_slice(&zero_patch);
                rows + 2
            }
            None => 0,
        };
        let _ = ref_rows;

        // Text rows: the ids plus the start token.
        let text_length = text_token_ids.len() + 1;
        text_token.extend_from_slice(text_token_ids);
        text_token.push(AUDIO_START_TOKEN);
        text_mask.extend(std::iter::repeat_n(1.0f32, text_length));
        audio_mask.extend(std::iter::repeat_n(0.0f32, text_length));
        for _ in 0..text_length {
            audio_feat.extend_from_slice(&zero_patch);
        }

        // Continuation rows: the prompt patches as audio rows.
        let mut continuation_prefix: Vec<f32> = Vec::new();
        let mut continued = false;
        if let Some(continuation) = continuation {
            let (rows, patches) = self.encode_wav(&continuation.prompt_audio, "left")?;
            text_token.extend(std::iter::repeat_n(0i32, rows));
            text_mask.extend(std::iter::repeat_n(0.0f32, rows));
            audio_mask.extend(std::iter::repeat_n(1.0f32, rows));
            audio_feat.extend_from_slice(&patches);
            continued = true;

            // The continuation prefix keeps the last
            // `streaming_prefix_len - 1` prompt patches.
            let context_len = streaming_prefix_len.saturating_sub(1).min(rows);
            let keep = context_len * patch * latent;
            let total = patches.len();
            continuation_prefix = patches[total - keep..].to_vec();
        }

        // Combined embeddings.
        let rows = text_token.len();
        let text_embed = self.base_lm.embed(&text_token, scale_emb)?;
        let encoded = self.feat_encoder.forward(&audio_feat, rows, patch)?;
        let feat_embed = self.enc_to_lm_proj.forward(&encoded, rows);
        let mut combined = vec![0.0f32; rows * hidden];
        for row in 0..rows {
            for col in 0..hidden {
                combined[row * hidden + col] = text_mask[row] * text_embed[row * hidden + col]
                    + audio_mask[row] * feat_embed[row * hidden + col];
            }
        }
        // The condition starts from the last patch row.
        let prefix_feat_cond = audio_feat[(rows - 1) * patch * latent..].to_vec();
        Ok(PromptBundle {
            combined,
            audio_mask,
            feat_embed,
            prefix_feat_cond,
            continuation_prefix,
            continued,
        })
    }
}

/// The prompt construction bundle.
struct PromptBundle {
    combined: Vec<f32>,
    audio_mask: Vec<f32>,
    feat_embed: Vec<f32>,
    prefix_feat_cond: Vec<f32>,
    continuation_prefix: Vec<f32>,
    continued: bool,
}
