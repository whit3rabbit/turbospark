//! KugelAudio TTS family (kugelaudio/kugelaudio-open, a VibeVoice
//! derivative).
//!
//! Port of `mlx_audio/tts/models/kugelaudio/` at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/kugelaudio):
//! a hybrid AR + diffusion architecture. A Qwen2.5 decoder (the shared
//! [`crate::lm`] qwen2 flavor: biased q/k/v, GQA, non-traditional rope,
//! final norm) runs autoregressively over a four-token speech vocabulary
//! (speech start / speech end / speech diffusion / EOS, greedy argmax
//! under a keep-set constraint). Each speech_diffusion token triggers
//! the VibeVoice diffusion head under the SDE-DPM-Solver++ stochastic
//! scheduler (v-prediction, cosine betas, classifier-free guidance
//! against a speech-start-only negative branch), and the sampled
//! acoustic latent feeds back through the speech connector into the
//! decoder stream. At the end the concatenated latents are unscaled by
//! the checkpoint's speech scaling/bias factors, decoded by the
//! convolutional VAE acoustic tokenizer decoder, and peak-normalized to
//! 0.95 when the decode exceeds unit amplitude.
//!
//! Scope of this port, matching the reference single-prompt path: batch
//! 1, non-streaming, and caller-owned text tokenization (the prompt
//! strings of `_build_prompt_tokens` are exported; the caller encodes
//! them with the checkpoint tokenizer and appends the speech start id).
//! Diffusion noise comes from a [`DiffusionNoise`] provider so fixture
//! tests replay the recorded reference draws and standalone runs can
//! seed the crate-local threefry RNG. See [README.md](README.md) for
//! the verification evidence and open items.

use serde_json::Value;

use self::acoustic_decoder::AcousticDecoder;
use self::diffusion_head::DiffusionHead;
use self::scheduler::SDEDpmSolver;
use crate::lm::engine::{Backbone, Input};
use crate::lm::{qwen2, Weights};
use crate::music::minimax_music3::rng;
use crate::nn::Linear;
use crate::nn::RmsNorm;
use crate::{ops, quant, Result, SpeechError};

mod acoustic_decoder;
mod diffusion_head;
mod scheduler;

#[cfg(test)]
mod tests;

/// Repurposed Qwen2.5 vision token ids.
pub const SPEECH_START_ID: i32 = 151652;
pub const SPEECH_END_ID: i32 = 151653;
pub const SPEECH_DIFFUSION_ID: i32 = 151654;
pub const EOS_TOKEN_ID: i32 = 151643;

/// The four tokens the argmax constraint keeps.
pub const VALID_SPEECH_TOKENS: [i32; 4] = [
    SPEECH_START_ID,
    SPEECH_END_ID,
    SPEECH_DIFFUSION_ID,
    EOS_TOKEN_ID,
];

/// When speech_end wins but the speech_diffusion logit is within this
/// margin, one final latent is generated to avoid cutting off the last
/// syllable.
pub const FINAL_LATENT_LOGIT_MARGIN: f32 = 5.0;

/// `_build_prompt_tokens` system prompt (leading space included).
pub const SYSTEM_PROMPT: &str = " Transform the text provided by various speakers into \
speech output, utilizing the distinct voice of each respective speaker.\n";

/// Formats the user text (`strip`, then the `Speaker 0:` prefix unless
/// the text already starts with `Speaker`).
pub fn format_speaker_text(text: &str) -> String {
    let formatted = text.trim();
    if formatted.starts_with("Speaker") {
        formatted.to_owned()
    } else {
        format!("Speaker 0: {formatted}")
    }
}

/// The text section of the prompt for a formatted text.
pub fn text_section(formatted_text: &str) -> String {
    format!(" Text input:\n {formatted_text}\n Speech output:\n")
}

/// The full prompt text the caller encodes
/// (`system_prompt + text_section`); the caller appends
/// [`SPEECH_START_ID`].
pub fn prompt_text(text: &str) -> String {
    format!(
        "{SYSTEM_PROMPT}{}",
        text_section(&format_speaker_text(text))
    )
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// The acoustic tokenizer config (`AcousticTokenizerConfig`); only the
/// fields the decoder path reads are kept.
#[derive(Debug, Clone)]
pub struct AcousticTokenizerConfig {
    pub channels: usize,
    pub vae_dim: usize,
    pub n_filters: usize,
    pub ratios: Vec<usize>,
    pub depths: Vec<usize>,
    pub layernorm_eps: f32,
    pub conv_bias: bool,
    pub layer_scale_init_value: f32,
}

impl AcousticTokenizerConfig {
    pub fn from_json(value: &Value) -> Self {
        let i = |key: &str, default: usize| {
            value
                .get(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(default)
        };
        let ratios: Vec<usize> = value
            .get("encoder_ratios")
            .and_then(Value::as_array)
            .map(|list| {
                list.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_else(|| vec![8, 5, 5, 4, 2, 2]);
        // decoder_ratios fall back to the encoder ratios; decoder_depths
        // to the reversed encoder depths.
        let depths: Vec<usize> = match value.get("decoder_depths") {
            Some(Value::String(text)) => text
                .split('-')
                .filter_map(|part| part.parse::<usize>().ok())
                .collect(),
            Some(Value::Array(list)) => list
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect(),
            _ => {
                let mut depths: Vec<usize> = match value.get("encoder_depths") {
                    Some(Value::String(text)) => text
                        .split('-')
                        .filter_map(|part| part.parse::<usize>().ok())
                        .collect(),
                    _ => vec![3, 3, 3, 3, 3, 3, 8],
                };
                depths.reverse();
                depths
            }
        };
        Self {
            channels: i("channels", 1),
            vae_dim: i("vae_dim", 64),
            n_filters: value
                .get("decoder_n_filters")
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or_else(|| i("encoder_n_filters", 32)),
            ratios,
            depths,
            layernorm_eps: value
                .get("layernorm_eps")
                .and_then(Value::as_f64)
                .unwrap_or(1e-5) as f32,
            conv_bias: value
                .get("conv_bias")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            layer_scale_init_value: value
                .get("layer_scale_init_value")
                .and_then(Value::as_f64)
                .unwrap_or(1e-6) as f32,
        }
    }
}

/// The diffusion head config (`DiffusionHeadConfig`).
#[derive(Debug, Clone)]
pub struct DiffusionHeadConfig {
    pub hidden_size: usize,
    pub head_layers: usize,
    pub head_ffn_ratio: f32,
    pub rms_norm_eps: f32,
    pub latent_size: usize,
    pub prediction_type: String,
    pub ddpm_num_steps: usize,
    pub ddpm_num_inference_steps: usize,
    pub ddpm_beta_schedule: String,
}

impl DiffusionHeadConfig {
    pub fn from_json(value: &Value) -> Self {
        let i = |key: &str, default: usize| {
            value
                .get(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(default)
        };
        Self {
            hidden_size: i("hidden_size", 3584),
            head_layers: i("head_layers", 4),
            head_ffn_ratio: value
                .get("head_ffn_ratio")
                .and_then(Value::as_f64)
                .unwrap_or(3.0) as f32,
            rms_norm_eps: value
                .get("rms_norm_eps")
                .and_then(Value::as_f64)
                .unwrap_or(1e-5) as f32,
            latent_size: i("latent_size", 64),
            prediction_type: value
                .get("prediction_type")
                .and_then(Value::as_str)
                .unwrap_or("v_prediction")
                .to_owned(),
            ddpm_num_steps: i("ddpm_num_steps", 1000),
            ddpm_num_inference_steps: i("ddpm_num_inference_steps", 10),
            ddpm_beta_schedule: value
                .get("ddpm_beta_schedule")
                .and_then(Value::as_str)
                .unwrap_or("cosine")
                .to_owned(),
        }
    }
}

/// The model config (`ModelConfig.from_dict`, including the
/// `acostic_vae_dim` typo handling).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub sample_rate: u32,
    pub acoustic_tokenizer: AcousticTokenizerConfig,
    pub decoder: crate::lm::DenseConfig,
    pub diffusion_head: DiffusionHeadConfig,
    pub acoustic_vae_dim: usize,
    pub tie_word_embeddings: bool,
}

impl ModelConfig {
    pub fn from_json(value: &Value) -> Result<Self> {
        let acoustic_value = value
            .get("acoustic_tokenizer_config")
            .cloned()
            .unwrap_or(Value::Null);
        let decoder_value = value.get("decoder_config").cloned().unwrap_or(Value::Null);
        let diffusion_value = value
            .get("diffusion_head_config")
            .cloned()
            .unwrap_or(Value::Null);
        let acoustic_vae_dim = value
            .get("acoustic_vae_dim")
            .or_else(|| value.get("acostic_vae_dim"))
            .and_then(Value::as_u64)
            .unwrap_or(64) as usize;
        let decoder = qwen2::from_json(&decoder_value)?;
        Ok(Self {
            sample_rate: value
                .get("sample_rate")
                .and_then(Value::as_u64)
                .unwrap_or(24000) as u32,
            acoustic_tokenizer: AcousticTokenizerConfig::from_json(&acoustic_value),
            decoder,
            diffusion_head: DiffusionHeadConfig::from_json(&diffusion_value),
            acoustic_vae_dim,
            tie_word_embeddings: value
                .get("tie_word_embeddings")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        })
    }
}

// ---------------------------------------------------------------------------
// Noise providers
// ---------------------------------------------------------------------------

/// The diffusion noise source: one standard-normal draw of
/// `[rows, cols]` per call, in reference draw order (the initial latent
/// noise of a `sample_speech_tokens` call, then one draw per scheduler
/// step).
pub trait DiffusionNoise {
    fn normal(&mut self, rows: usize, cols: usize) -> Vec<f32>;
}

/// Seeded crate-local threefry noise for standalone runs.
pub struct SeededNoise {
    keys: rng::KeySequence,
}

impl SeededNoise {
    pub fn new(seed: u64) -> Self {
        Self {
            keys: rng::KeySequence::new(seed),
        }
    }
}

impl DiffusionNoise for SeededNoise {
    fn normal(&mut self, rows: usize, cols: usize) -> Vec<f32> {
        rng::normal(self.keys.next(), rows * cols)
    }
}

// ---------------------------------------------------------------------------
// The loaded model
// ---------------------------------------------------------------------------

/// The speech connector (`SpeechConnector`): fc1 -> RMSNorm -> fc2,
/// both linears biased.
pub(crate) struct SpeechConnector {
    fc1: Linear,
    fc2: Linear,
    norm: RmsNorm,
}

impl SpeechConnector {
    fn load(weights: &Weights, input_dim: usize, output_dim: usize, eps: f32) -> Result<Self> {
        Ok(Self {
            // The connector tensors sit outside the backbone prefix.
            fc1: weights.linear_absolute(
                "acoustic_connector.fc1",
                input_dim,
                output_dim,
                crate::lm::Bias::Required,
            )?,
            norm: RmsNorm::new(
                weights.tensor_absolute("acoustic_connector.norm.weight", &[output_dim])?,
                eps,
            ),
            fc2: weights.linear_absolute(
                "acoustic_connector.fc2",
                output_dim,
                output_dim,
                crate::lm::Bias::Required,
            )?,
        })
    }

    /// `[rows, input] -> [rows, output]`.
    fn forward(&self, features: &[f32], rows: usize) -> Vec<f32> {
        let mut x = self.fc1.forward(features, rows);
        self.norm.apply(&mut x, rows);
        self.fc2.forward(&x, rows)
    }
}

/// The loaded kugelaudio model.
pub struct Model {
    pub config: ModelConfig,
    language_model: Backbone,
    lm_head: Option<Linear>,
    pub(crate) acoustic_decoder: AcousticDecoder,
    acoustic_connector: SpeechConnector,
    prediction_head: DiffusionHead,
    scheduler: SDEDpmSolver,
    speech_scaling_factor: f32,
    speech_bias_factor: f32,
}

/// Loads the shared qwen2 backbone, flipping the untied head off when
/// the checkpoint carries no `lm_head` (the tied fallback is inert for
/// `forward_hidden`; the family applies its own head through
/// `forward_hidden_and_logits`).
fn load_language_model(weights: &Weights, dense: &crate::lm::DenseConfig) -> Result<Backbone> {
    let mut dense = dense.clone();
    if !dense.tie_word_embeddings
        && !weights.contains("lm_head")
        && !weights.contains_absolute("lm_head")
    {
        dense.tie_word_embeddings = true;
    }
    Backbone::load(weights, dense)
}

/// Opens the checkpoint with the backbone prefix the converted saves
/// use (`language_model.` in the model tree, optionally under the raw
/// composite `model.` prefix); falls back to the unprefixed view.
fn open_backbone_weights(dir: &std::path::Path) -> Result<Weights> {
    let probe = Weights::open(dir)?;
    if probe.contains_absolute("language_model.embed_tokens") {
        return Ok(probe.with_prefix("language_model."));
    }
    if probe.contains_absolute("model.language_model.embed_tokens") {
        return Weights::open(dir).map(|w| w.with_prefix("model.language_model."));
    }
    Ok(probe)
}

impl Model {
    /// Opens a checkpoint directory (`config.json` + safetensors).
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        let parsed = ModelConfig::from_json(&config_value)?;
        let weights = open_backbone_weights(dir)?;
        Self::load_weights(&weights, parsed)
    }

    /// Loads from opened weights and the raw config dict. The backbone
    /// prefix must cover the language model (`language_model.` for the
    /// converted saves).
    pub fn load(weights: &Weights, config: &Value) -> Result<Self> {
        let parsed = ModelConfig::from_json(config)?;
        Self::load_weights(weights, parsed)
    }

    fn load_weights(weights: &Weights, config: ModelConfig) -> Result<Self> {
        let files = weights.shards();
        let decoder_config = config.decoder.clone();
        let language_model = load_language_model(weights, &decoder_config)?;
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(weights.linear_absolute(
                "lm_head",
                decoder_config.hidden_size,
                decoder_config.vocab_size,
                crate::lm::Bias::None,
            )?)
        };
        let acoustic_config = &config.acoustic_tokenizer;
        let acoustic_decoder = AcousticDecoder::load(
            files,
            "acoustic_tokenizer.decoder",
            acoustic_config.vae_dim,
            acoustic_config.channels,
            acoustic_config.n_filters,
            &acoustic_config.ratios,
            &acoustic_config.depths,
            acoustic_config.layernorm_eps,
            acoustic_config.conv_bias,
        )?;
        let acoustic_connector = SpeechConnector::load(
            weights,
            config.acoustic_vae_dim,
            decoder_config.hidden_size,
            decoder_config.rms_norm_eps,
        )?;
        let head_config = &config.diffusion_head;
        let prediction_head = DiffusionHead::load(
            files,
            "prediction_head",
            head_config.hidden_size,
            head_config.head_layers,
            head_config.head_ffn_ratio,
            head_config.rms_norm_eps,
            head_config.latent_size,
        )?;
        let scheduler = SDEDpmSolver::new(
            head_config.ddpm_num_steps,
            &head_config.ddpm_beta_schedule,
            &head_config.prediction_type,
        )?;
        // The scaling factors are checkpoint scalars with 1.0 / 0.0
        // defaults.
        // The scaling factors are top-level checkpoint scalars outside
        // the backbone prefix, with 1.0 / 0.0 defaults.
        let speech_scaling_factor = weights
            .tensor_absolute("speech_scaling_factor", &[])
            .ok()
            .and_then(|v| v.first().copied())
            .unwrap_or(1.0);
        let speech_bias_factor = weights
            .tensor_absolute("speech_bias_factor", &[])
            .ok()
            .and_then(|v| v.first().copied())
            .unwrap_or(0.0);
        Ok(Self {
            config,
            language_model,
            lm_head,
            acoustic_decoder,
            acoustic_connector,
            prediction_head,
            scheduler,
            speech_scaling_factor,
            speech_bias_factor,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// `get_lm_logits` over the post-norm hidden states: the untied
    /// checkpoint head or the tied embedding table.
    fn lm_logits(&self, hidden: &[f32], rows: usize) -> Vec<f32> {
        match &self.lm_head {
            Some(head) => head.forward(hidden, rows),
            None => ops::linear(
                hidden,
                self.language_model.embed_table(),
                None,
                rows,
                self.config.decoder.hidden_size,
                self.config.decoder.vocab_size,
            ),
        }
    }

    /// `sample_speech_tokens`: the SDE-DPM-Solver++ diffusion from pure
    /// noise, with classifier-free guidance when `cfg_scale > 1.0` and a
    /// negative condition is given. One initial noise draw plus one per
    /// scheduler step, in reference order.
    fn sample_speech_tokens(
        &mut self,
        condition: &[f32],
        cfg_scale: f32,
        neg_condition: Option<&[f32]>,
        noise: &mut dyn DiffusionNoise,
    ) -> Result<Vec<f32>> {
        let vae_dim = self.config.acoustic_vae_dim;
        let rows = condition.len() / self.config.decoder.hidden_size;
        let steps = self.config.diffusion_head.ddpm_num_inference_steps;
        self.scheduler.reset();
        self.scheduler.set_timesteps(steps);
        let timesteps: Vec<i64> = self.scheduler.timesteps().to_vec();

        if cfg_scale <= 1.0 || neg_condition.is_none() {
            let mut speech = noise.normal(rows, vae_dim);
            let mut prev_x0: Option<Vec<f32>> = None;
            for &t in &timesteps {
                let t_batch = vec![t as f32; rows];
                let eps = self
                    .prediction_head
                    .forward(&speech, &t_batch, condition, rows)?;
                let (next, x0_pred) = self.scheduler.step(
                    &eps,
                    &speech,
                    prev_x0.as_deref(),
                    &noise.normal(rows, vae_dim),
                )?;
                speech = next;
                prev_x0 = Some(x0_pred);
            }
            return Ok(speech);
        }

        // With CFG: the cond and uncond branches run batched through the
        // prediction head; the scheduler steps the unbatched guided eps.
        let neg_condition = neg_condition.expect("neg condition");
        let mut combined_condition = condition.to_vec();
        combined_condition.extend_from_slice(neg_condition);
        let mut speech = noise.normal(rows, vae_dim);
        let mut prev_x0: Option<Vec<f32>> = None;
        for &t in &timesteps {
            let mut combined_speech = speech.clone();
            combined_speech.extend_from_slice(&speech);
            let t_batch = vec![t as f32; 2 * rows];
            let eps = self.prediction_head.forward(
                &combined_speech,
                &t_batch,
                &combined_condition,
                2 * rows,
            )?;
            let width = eps.len() / (2 * rows);
            let mut guided = vec![0.0f32; rows * width];
            for row in 0..rows {
                for col in 0..width {
                    let cond = eps[row * width + col];
                    let uncond = eps[(rows + row) * width + col];
                    guided[row * width + col] = uncond + cfg_scale * (cond - uncond);
                }
            }
            let (next, x0_pred) = self.scheduler.step(
                &guided,
                &speech,
                prev_x0.as_deref(),
                &noise.normal(rows, vae_dim),
            )?;
            speech = next;
            prev_x0 = Some(x0_pred);
        }
        Ok(speech)
    }

    /// The reference `generate` for batch 1: the greedy constrained AR
    /// loop over the four speech tokens with the per-diffusion-token
    /// latent sampling and feedback, then the batch latent decode.
    /// `prompt_tokens` is the caller-encoded prompt INCLUDING the
    /// trailing speech start id. The final-latent margin branch
    /// generates one extra latent when speech_end wins while the
    /// diffusion logit sits within [`FINAL_LATENT_LOGIT_MARGIN`].
    pub fn generate(
        &mut self,
        prompt_tokens: &[i32],
        params: &GenerateParams,
        noise: &mut dyn DiffusionNoise,
    ) -> Result<Generated> {
        if prompt_tokens.is_empty() {
            return Err(SpeechError::Input {
                why: "empty prompt".into(),
            });
        }
        let cfg_scale = params.cfg_scale;
        let ddpm_steps = params.ddpm_steps;
        let saved_steps = self.config.diffusion_head.ddpm_num_inference_steps;
        if let Some(steps) = ddpm_steps {
            self.config.diffusion_head.ddpm_num_inference_steps = steps;
        }
        let result = self.generate_impl(prompt_tokens, cfg_scale, params.max_tokens, noise);
        self.config.diffusion_head.ddpm_num_inference_steps = saved_steps;
        result
    }

    fn generate_impl(
        &mut self,
        prompt_tokens: &[i32],
        cfg_scale: f32,
        max_tokens: usize,
        noise: &mut dyn DiffusionNoise,
    ) -> Result<Generated> {
        let hidden_size = self.config.decoder.hidden_size;

        // Initial forward pass through the language model (full prompt).
        let mut caches = crate::lm::make_prompt_cache(self.config.decoder.num_hidden_layers);
        let hidden_states = self
            .language_model
            .forward_hidden(Input::Tokens(prompt_tokens), Some(&mut caches))?;

        // Negative branch for CFG.
        let mut neg_caches = if cfg_scale > 1.0 {
            Some(crate::lm::make_prompt_cache(
                self.config.decoder.num_hidden_layers,
            ))
        } else {
            None
        };
        let mut neg_hidden = match neg_caches.as_mut() {
            Some(neg_caches) => Some(
                self.language_model
                    .forward_hidden(Input::Tokens(&[SPEECH_START_ID]), Some(neg_caches))?,
            ),
            None => None,
        };

        let mut all_speech_latents: Vec<f32> = Vec::new();
        let mut latent_count = 0usize;
        let mut token_count = prompt_tokens.len();
        let mut hidden_states = hidden_states;
        let mut step = 0usize;
        while step < max_tokens {
            let last_hidden = &hidden_states[hidden_states.len() - hidden_size..];
            let logits = self.lm_logits(last_hidden, 1);

            // Constrain to the valid speech tokens; the argmax keeps the
            // first maximal id like `mx.argmax`.
            let mut next_token_id = SPEECH_END_ID;
            let mut best = f32::NEG_INFINITY;
            for &token in VALID_SPEECH_TOKENS.iter() {
                let index = usize::try_from(token).map_err(|_| SpeechError::Input {
                    why: "negative speech token id".into(),
                })?;
                let score = logits.get(index).copied().unwrap_or(f32::NEG_INFINITY);
                if score > best {
                    best = score;
                    next_token_id = token;
                }
            }
            token_count += 1;
            step += 1;

            if next_token_id == SPEECH_END_ID || next_token_id == EOS_TOKEN_ID {
                let diff_logit = logits
                    .get(SPEECH_DIFFUSION_ID as usize)
                    .copied()
                    .unwrap_or(f32::NEG_INFINITY);
                let end_logit = logits
                    .get(next_token_id as usize)
                    .copied()
                    .unwrap_or(f32::NEG_INFINITY);
                if should_append_final_latent(latent_count > 0, diff_logit, end_logit) {
                    let condition = last_hidden.to_vec();
                    let neg_condition = if cfg_scale > 1.0 {
                        neg_hidden
                            .as_ref()
                            .map(|h| h[h.len() - hidden_size..].to_vec())
                    } else {
                        None
                    };
                    let final_latent = self.sample_speech_tokens(
                        &condition,
                        cfg_scale,
                        neg_condition.as_deref(),
                        noise,
                    )?;
                    all_speech_latents.extend_from_slice(&final_latent);
                    latent_count += 1;
                }
                // The loop condition ends the run.
                break;
            }

            if next_token_id == SPEECH_DIFFUSION_ID {
                let condition = last_hidden.to_vec();
                let neg_condition = if cfg_scale > 1.0 {
                    neg_hidden
                        .as_ref()
                        .map(|h| h[h.len() - hidden_size..].to_vec())
                } else {
                    None
                };
                let speech_latents = self.sample_speech_tokens(
                    &condition,
                    cfg_scale,
                    neg_condition.as_deref(),
                    noise,
                )?;
                all_speech_latents.extend_from_slice(&speech_latents);
                latent_count += 1;

                // Feed the latent back through the connector into the LM
                // (and the negative branch).
                let acoustic_embed = self.acoustic_connector.forward(&speech_latents, 1);
                hidden_states = self
                    .language_model
                    .forward_hidden(Input::Embeddings(&acoustic_embed), Some(&mut caches))?;
                if let Some(neg_caches) = neg_caches.as_mut() {
                    if cfg_scale > 1.0 {
                        neg_hidden = Some(self.language_model.forward_hidden(
                            Input::Embeddings(&acoustic_embed),
                            Some(neg_caches),
                        )?);
                    }
                }
            } else if next_token_id == SPEECH_START_ID {
                let token_embed = crate::lm::engine::lookup_embeddings(
                    self.language_model.embed_table(),
                    self.config.decoder.vocab_size,
                    hidden_size,
                    &[SPEECH_START_ID],
                )?;
                hidden_states = self
                    .language_model
                    .forward_hidden(Input::Embeddings(&token_embed), Some(&mut caches))?;
            }
        }

        if latent_count == 0 {
            return Ok(Generated {
                latents: Vec::new(),
                latent_frames: 0,
                samples: Vec::new(),
                sample_rate: self.config.sample_rate,
                token_count,
            });
        }

        // Unscale the latents (NaN scaling factors keep the raw latents;
        // the port treats NaN as "no scaling" like the reference guard).
        let mut latent_seq = all_speech_latents;
        if !self.speech_scaling_factor.is_nan() {
            for value in &mut latent_seq {
                *value = *value / self.speech_scaling_factor - self.speech_bias_factor;
            }
        }

        let frames = latent_seq.len() / self.config.acoustic_vae_dim;
        let samples = self.acoustic_decoder.forward(&latent_seq, frames)?;

        // Peak-normalize to 0.95 when the decode exceeds unit amplitude.
        let max_val = samples.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let samples = if max_val > 1.0 {
            samples.iter().map(|v| v * (0.95 / max_val)).collect()
        } else {
            samples
        };

        Ok(Generated {
            latents: latent_seq,
            latent_frames: frames,
            samples,
            sample_rate: self.config.sample_rate,
            token_count,
        })
    }
}

/// The stop-decision margin rule, factored for direct unit coverage.
pub fn should_append_final_latent(has_latents: bool, diff_logit: f32, end_logit: f32) -> bool {
    has_latents && diff_logit > end_logit - FINAL_LATENT_LOGIT_MARGIN
}

/// `generate` parameters (`generate(text, cfg_scale, max_tokens,
/// ddpm_steps)`).
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub cfg_scale: f32,
    pub max_tokens: usize,
    pub ddpm_steps: Option<usize>,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            cfg_scale: 3.0,
            max_tokens: 2048,
            ddpm_steps: None,
        }
    }
}

/// One generation result.
#[derive(Debug, Clone)]
pub struct Generated {
    /// The unscaled latents `[frames, acoustic_vae_dim]`.
    pub latents: Vec<f32>,
    pub latent_frames: usize,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub token_count: usize,
}
