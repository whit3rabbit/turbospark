//! VoxCPM 1.5 (alias `voxcpm1.5`): a MiniCPM latent-flow TTS family.
//!
//! Port of `mlx_audio/tts/models/voxcpm/` at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/voxcpm)
//! (MLX 0.32.3). The model is tokenizer-free in the codebook sense:
//! there are no audio tokens. A MiniCPM base LM over text embeddings
//! and a residual LM over encoded audio latents jointly condition a
//! patch-level flow-matching DiT (the `UnifiedCFM` Euler solver with
//! CFG zero-star and sway sampling); each sampled `[feat_dim x
//! patch_size]` latent patch feeds back through the CLS patch encoder,
//! a two-class stop head over the LM state ends the run after step 5,
//! and the concatenated patches are decoded to a waveform by the
//! causal-convolutional AudioVAE. Voice cloning splices VAE-encoded
//! reference-audio patches plus ref-text embeddings into the prefix.
//!
//! Scope, matching the reference single-prompt path: single batch,
//! non-streaming (the reference yields one `GenerationResult`),
//! caller-owned text tokenization (the reference encodes
//! `ref_text + text` with the checkpoint tokenizer; hand over the ids,
//! the `audio_start_token` 101 is appended here). Flow noise comes
//! from a [`FlowNoise`] provider so fixture tests replay the recorded
//! reference draws and standalone runs can seed the crate-local
//! threefry RNG. See [README.md](README.md) for the verification
//! evidence and open items.

use serde_json::Value;

use self::audio_vae::{AudioVae, AudioVaeConfig};
use self::dit::UnifiedCfm;
use self::encoder::VoxCpmLocEnc;
use self::minicpm::{Config as MiniCpmConfig, Mask, MiniCpmModel};
use crate::lm::cache::KvCache;
use crate::lm::Weights;
use crate::music::minimax_music3::rng;
use crate::nn::Linear;
use crate::{quant, Result, SpeechError};

pub(crate) mod audio_vae;
pub(crate) mod dit;
pub(crate) mod encoder;
pub(crate) mod minicpm;

#[cfg(test)]
mod tests;

/// The reference `audio_start_token`.
pub const AUDIO_START_TOKEN: i32 = 101;

/// The flow noise source: one standard-normal draw of `[feat_dim *
/// patch_size]` per generate step, in reference draw order (the
/// `(B, C, T)` initial Euler state).
pub trait FlowNoise {
    fn normal(&mut self, n: usize) -> Vec<f32>;
}

/// Seeded crate-local threefry noise for standalone runs.
pub struct SeededFlowNoise {
    keys: rng::KeySequence,
}

impl SeededFlowNoise {
    pub fn new(seed: u64) -> Self {
        Self {
            keys: rng::KeySequence::new(seed),
        }
    }
}

impl FlowNoise for SeededFlowNoise {
    fn normal(&mut self, n: usize) -> Vec<f32> {
        rng::normal(self.keys.next(), n)
    }
}

/// The encoder/diT sub-configs (`EncoderConfig`, `DiTConfig` with its
/// nested `CFMConfig`).
#[derive(Debug, Clone)]
pub struct SubConfig {
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    /// DiT only: the CFM parameters. Only the names are validated; the
    /// solver is the hardcoded Euler + sway + zero-star path.
    pub solver: String,
    pub t_scheduler: String,
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
        Self {
            hidden_dim: i("hidden_dim", 1024),
            ffn_dim: i("ffn_dim", 4096),
            num_heads: i("num_heads", 16),
            num_layers: i("num_layers", 8),
            solver: value
                .get("cfm_config")
                .and_then(|c| c.get("solver"))
                .and_then(Value::as_str)
                .unwrap_or("euler")
                .to_owned(),
            t_scheduler: value
                .get("cfm_config")
                .and_then(|c| c.get("t_scheduler"))
                .and_then(Value::as_str)
                .unwrap_or("log-norm")
                .to_owned(),
        }
    }
}

/// The resolved model config (`ModelArgs.from_dict`).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// The base LM config; also the field donor for the residual LM,
    /// encoder, and DiT backbone variants.
    pub lm_config: MiniCpmConfig,
    pub encoder_config: SubConfig,
    pub dit_config: SubConfig,
    pub audio_vae_config: AudioVaeConfig,
    pub patch_size: usize,
    pub feat_dim: usize,
    pub scalar_quantization_latent_dim: usize,
    pub scalar_quantization_scale: i32,
    pub residual_lm_num_layers: usize,
    pub max_length: usize,
}

impl ModelConfig {
    pub fn from_json(value: &Value) -> Result<Self> {
        // The reference from_dict flattens a nested `rope_scaling`
        // block before constructing the LMConfig.
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
        let lm_config = MiniCpmConfig::from_json(&lm_value);
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
            scalar_quantization_latent_dim: i("scalar_quantization_latent_dim", 256),
            scalar_quantization_scale: value
                .get("scalar_quantization_scale")
                .and_then(Value::as_i64)
                .map(|n| n as i32)
                .unwrap_or(9),
            residual_lm_num_layers: i("residual_lm_num_layers", 8),
            max_length: i("max_length", 8192),
            lm_config,
        })
    }

    /// The residual LM backbone config: the base LM fields with
    /// `num_hidden_layers = residual_lm_num_layers` and no vocabulary.
    fn residual_config(&self) -> MiniCpmConfig {
        let mut config = self.lm_config.clone();
        config.num_hidden_layers = self.residual_lm_num_layers;
        config.vocab_size = 0;
        config
    }

    /// The patch encoder backbone config.
    fn encoder_backbone_config(&self) -> MiniCpmConfig {
        self.lm_config.with_overrides(
            self.encoder_config.hidden_dim,
            self.encoder_config.ffn_dim,
            self.encoder_config.num_heads,
            self.encoder_config.num_layers,
        )
    }

    /// The DiT denoiser backbone config.
    fn dit_backbone_config(&self) -> MiniCpmConfig {
        self.lm_config.with_overrides(
            self.dit_config.hidden_dim,
            self.dit_config.ffn_dim,
            self.dit_config.num_heads,
            self.dit_config.num_layers,
        )
    }
}

/// Row-major transpose: `[rows, cols]` in, `[cols, rows]` out.
pub fn transpose(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; values.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = values[r * cols + c];
        }
    }
    out
}

/// `ScalarQuantizationLayer`: `out_proj(round(tanh(in_proj(x)) *
/// scale) / scale)`.
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

/// A loaded voice-clone prompt: the reference waveform at the model
/// sample rate. The reference encodes `ref_text + text` as ONE string,
/// so the caller encodes that concatenation and passes the ids to
/// [`Model::generate`]; this struct carries only the audio.
#[derive(Debug, Clone, Default)]
pub struct VoicePrompt {
    pub ref_audio: Vec<f32>,
}

/// `generate` parameters.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub max_tokens: usize,
    pub inference_timesteps: usize,
    pub cfg_value: f32,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            inference_timesteps: 10,
            cfg_value: 2.0,
        }
    }
}

/// One generation result.
#[derive(Debug, Clone)]
pub struct Generated {
    /// The decoded waveform samples.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// The caller text token count (the reference `token_count`; the
    /// appended start token and any reference tokens are excluded in
    /// zero-shot mode, included in clone mode exactly like the
    /// reference).
    pub token_count: usize,
    /// The sampled latent patches, `[steps * patch_size, feat_dim]`
    /// (the AudioVAE decoder input).
    pub latents: Vec<f32>,
    /// Sampled patch count.
    pub steps: usize,
    /// Whether the stop head ended the run (`i > 5` and class 1).
    pub stopped: bool,
}

/// The loaded VoxCPM model.
pub struct Model {
    pub config: ModelConfig,
    base_lm: MiniCpmModel,
    residual_lm: MiniCpmModel,
    feat_encoder: VoxCpmLocEnc,
    feat_decoder: UnifiedCfm,
    fsq: ScalarQuantization,
    enc_to_lm_proj: Linear,
    lm_to_dit_proj: Linear,
    res_to_dit_proj: Linear,
    stop_proj: Linear,
    stop_head: Linear,
    audio_vae: AudioVae,
    lm_cache: Vec<KvCache>,
    res_cache: Vec<KvCache>,
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
        let hidden = config.lm_config.hidden_size;
        let base_lm = MiniCpmModel::load(&weights, "base_lm.", config.lm_config.clone())?;
        let residual_lm = MiniCpmModel::load(&weights, "residual_lm.", config.residual_config())?;
        let feat_encoder = VoxCpmLocEnc::load(
            &weights,
            "feat_encoder.",
            config.feat_dim,
            &config.encoder_backbone_config(),
        )?;
        let feat_decoder = UnifiedCfm::load(
            &weights,
            "feat_decoder.",
            config.feat_dim,
            &config.dit_backbone_config(),
        )?;
        let fsq = ScalarQuantization::load(&weights, &config)?;
        let biased = |name: &str, input: usize, output: usize| -> Result<Linear> {
            weights.linear(name, input, output, crate::lm::Bias::Required)
        };
        let enc_to_lm_proj = biased("enc_to_lm_proj", config.encoder_config.hidden_dim, hidden)?;
        let lm_to_dit_proj = biased("lm_to_dit_proj", hidden, config.dit_config.hidden_dim)?;
        let res_to_dit_proj = biased("res_to_dit_proj", hidden, config.dit_config.hidden_dim)?;
        let stop_proj = biased("stop_proj", hidden, hidden)?;
        let stop_head = weights.linear("stop_head", hidden, 2, crate::lm::Bias::None)?;
        let audio_vae = AudioVae::load(weights.shards(), "audio_vae.", &config.audio_vae_config)?;
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
            stop_proj,
            stop_head,
            audio_vae,
            config,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.audio_vae.sample_rate()
    }

    /// `_encode_prompt_audio`: VAE-encode the reference waveform into
    /// patch rows. Returns `(rows, [rows * patch_size * feat_dim])`
    /// row-major; the input is left-padded to the
    /// `patch_size * hop_length` grid like the reference.
    pub fn encode_prompt_audio(&mut self, audio: &[f32]) -> Result<(usize, Vec<f32>)> {
        let patch = self.config.patch_size;
        let feat = self.config.feat_dim;
        let patch_len = patch * self.audio_vae.hop_length();
        let padded;
        let samples: &[f32] = if audio.len() % patch_len != 0 {
            let pad = patch_len - audio.len() % patch_len;
            let mut buffer = vec![0.0f32; pad];
            buffer.extend_from_slice(audio);
            padded = buffer;
            &padded
        } else {
            audio
        };
        // encode returns [latent_dim, frames] channel-major; the
        // reference keeps (T', D) row-major.
        let latents = self.audio_vae.encode(samples);
        let frames = latents.len() / feat;
        let rows = frames / patch;
        let flat = transpose(&latents, feat, frames);
        Ok((rows, flat[..rows * patch * feat].to_vec()))
    }

    /// The reference `generate` for one text. `text_token_ids` is the
    /// caller-encoded text without the start token (the reference
    /// appends `audio_start_token` 101 itself). With a
    /// [`VoicePrompt`], the run is voice cloning: the reference ids and
    /// VAE-encoded patches join the prefix.
    pub fn generate(
        &mut self,
        text_token_ids: &[i32],
        params: &GenerateParams,
        noise: &mut dyn FlowNoise,
        voice: Option<&VoicePrompt>,
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
        let hidden = self.config.lm_config.hidden_size;
        let patch = self.config.patch_size;
        let feat = self.config.feat_dim;
        let scale_emb = if self.config.lm_config.use_mup {
            1.0
        } else {
            self.config.lm_config.scale_emb as f32
        };

        // Reset the caches for this run.
        for cache in self.lm_cache.iter_mut() {
            cache.reset();
        }
        for cache in self.res_cache.iter_mut() {
            cache.reset();
        }

        let prompt = self.build_prompt(text_token_ids, voice, scale_emb)?;
        let total_length = prompt.combined.len() / hidden;

        let mut enc_outputs = self.base_lm.forward(
            &prompt.combined,
            total_length,
            Some(&mut self.lm_cache),
            Mask::Causal,
        )?;
        if let Some(clone) = &prompt.clone {
            // fsq on the audio rows, identity on the text rows.
            let fsq_rows = self.fsq.forward(&enc_outputs, total_length);
            for row in 0..total_length {
                let audio = clone.audio_mask[row] != 0.0;
                if !audio {
                    continue;
                }
                for col in 0..hidden {
                    enc_outputs[row * hidden + col] = fsq_rows[row * hidden + col];
                }
            }
        }
        let mut lm_hidden = enc_outputs[(total_length - 1) * hidden..].to_vec();
        if prompt.clone.is_none() {
            lm_hidden = self.fsq.forward(&lm_hidden, 1);
        }
        if let Some(clone) = &prompt.clone {
            for row in 0..total_length {
                for col in 0..hidden {
                    enc_outputs[row * hidden + col] +=
                        clone.audio_mask[row] * clone.feat_embed[row * hidden + col];
                }
            }
        }
        let residual_out = self.residual_lm.forward(
            &enc_outputs,
            total_length,
            Some(&mut self.res_cache),
            Mask::Causal,
        )?;
        let mut residual_hidden = residual_out[(total_length - 1) * hidden..].to_vec();

        let mut prefix_feat_cond = prompt.prefix_feat_cond;
        let mut pred_feat_seq: Vec<f32> = Vec::new();
        let mut steps = 0usize;
        let mut stopped = false;

        for index in 0..params.max_tokens {
            // DiT conditioning: lm_to_dit(lm_hidden) + res_to_dit(res).
            let dit_h = {
                let lm_part = self.lm_to_dit_proj.forward(&lm_hidden, 1);
                let res_part = self.res_to_dit_proj.forward(&residual_hidden, 1);
                lm_part
                    .iter()
                    .zip(&res_part)
                    .map(|(a, b)| a + b)
                    .collect::<Vec<f32>>()
            };
            let cond_in = transpose(&prefix_feat_cond, patch, feat);
            let pred_feat = self.feat_decoder.sample(
                &dit_h,
                params.inference_timesteps,
                patch,
                &cond_in,
                params.cfg_value,
                noise,
            )?;
            let pred = transpose(&pred_feat, feat, patch);
            pred_feat_seq.extend_from_slice(&pred);

            let curr_embed = {
                let encoded = self.feat_encoder.forward(&pred, 1, patch)?;
                self.enc_to_lm_proj.forward(&encoded, 1)
            };

            // Stop head over the LM state; the reference stops only
            // after step 5 (`i > 5`).
            let stop_flag = {
                let mut gated = self.stop_proj.forward(&lm_hidden, 1);
                for value in &mut gated {
                    *value *= 1.0 / (1.0 + (-*value).exp());
                }
                let logits = self.stop_head.forward(&gated, 1);
                crate::tts::sampling::argmax(&logits)
            };
            if index > 5 && stop_flag == 1 {
                stopped = true;
                steps = index + 1;
                break;
            }

            let new_lm_out =
                self.base_lm
                    .forward(&curr_embed, 1, Some(&mut self.lm_cache), Mask::Causal)?;
            let mut lm_hidden_next = new_lm_out[(new_lm_out.len() - hidden)..].to_vec();
            lm_hidden_next = self.fsq.forward(&lm_hidden_next, 1);

            let res_in: Vec<f32> = lm_hidden_next
                .iter()
                .zip(&curr_embed)
                .map(|(a, b)| a + b)
                .collect();
            let new_res_out =
                self.residual_lm
                    .forward(&res_in, 1, Some(&mut self.res_cache), Mask::Causal)?;
            residual_hidden = new_res_out[(new_res_out.len() - hidden)..].to_vec();

            lm_hidden = lm_hidden_next;
            prefix_feat_cond = pred;
        }
        if !stopped {
            steps = params.max_tokens;
        }

        // Decode all patches: [steps * patch, feat] -> [feat, total].
        let total_frames = steps * patch;
        let latents = pred_feat_seq;
        let latent_input = transpose(&latents, total_frames, feat);
        let samples = self.audio_vae.decode(&latent_input);
        // The reference counts the ids before the start token in
        // zero-shot mode but including it in clone mode (both are
        // `len(input_ids)` at their point in the reference flow).
        let token_count = if voice.is_some() {
            text_token_ids.len() + 1
        } else {
            text_token_ids.len()
        };
        Ok(Generated {
            samples,
            sample_rate: self.sample_rate(),
            token_count,
            latents,
            steps,
            stopped,
        })
    }

    /// Builds the prompt embeddings, masks, and the initial feature
    /// condition. Returns an internal bundle.
    fn build_prompt(
        &mut self,
        text_token_ids: &[i32],
        voice: Option<&VoicePrompt>,
        scale_emb: f32,
    ) -> Result<PromptBundle> {
        let hidden = self.config.lm_config.hidden_size;
        let patch = self.config.patch_size;
        let feat = self.config.feat_dim;
        match voice {
            Some(voice) => {
                // In clone mode `text_token_ids` already carries the
                // concatenated `ref_text + text` encode (the reference
                // tokenizes one string).
                let mut input_ids = text_token_ids.to_vec();
                input_ids.push(AUDIO_START_TOKEN);
                let text_length = input_ids.len();

                let (audio_rows, audio_flat) = self.encode_prompt_audio(&voice.ref_audio)?;
                let audio_length = audio_rows;

                // text rows: the ids then zeros; audio rows: the
                // encoded patches after text_length zero patches.
                let mut text_token = input_ids.clone();
                text_token.extend(std::iter::repeat_n(0i32, audio_length));

                let mut audio_feat = vec![0.0f32; text_length * patch * feat];
                audio_feat.extend_from_slice(&audio_flat);

                let mut text_mask = vec![1.0f32; text_length];
                text_mask.extend(std::iter::repeat_n(0.0f32, audio_length));
                let mut audio_mask = vec![0.0f32; text_length];
                audio_mask.extend(std::iter::repeat_n(1.0f32, audio_length));

                let text_embed = self.base_lm.embed(&text_token, scale_emb)?;
                let rows = text_length + audio_length;
                let encoded = self.feat_encoder.forward(&audio_feat, rows, patch)?;
                let feat_embed = self.enc_to_lm_proj.forward(&encoded, rows);

                let mut combined = vec![0.0f32; rows * hidden];
                for row in 0..rows {
                    for col in 0..hidden {
                        combined[row * hidden + col] = text_mask[row]
                            * text_embed[row * hidden + col]
                            + audio_mask[row] * feat_embed[row * hidden + col];
                    }
                }
                let prefix_feat_cond = audio_flat[(audio_length - 1) * patch * feat..].to_vec();
                Ok(PromptBundle {
                    combined,
                    clone: Some(CloneMix {
                        audio_mask,
                        feat_embed,
                    }),
                    prefix_feat_cond,
                })
            }
            None => {
                let mut input_ids = text_token_ids.to_vec();
                input_ids.push(AUDIO_START_TOKEN);
                let combined = self.base_lm.embed(&input_ids, scale_emb)?;
                Ok(PromptBundle {
                    prefix_feat_cond: vec![0.0f32; patch * feat],
                    combined,
                    clone: None,
                })
            }
        }
    }
}

/// The prompt construction bundle (`prompt` dict in the fixture
/// traces). `clone` carries the voice-clone masks and encoded features;
/// `None` in the zero-shot case.
pub(crate) struct PromptBundle {
    pub combined: Vec<f32>,
    pub clone: Option<CloneMix>,
    pub prefix_feat_cond: Vec<f32>,
}

/// The clone-branch mixing tensors.
pub(crate) struct CloneMix {
    pub audio_mask: Vec<f32>,
    pub feat_embed: Vec<f32>,
}
