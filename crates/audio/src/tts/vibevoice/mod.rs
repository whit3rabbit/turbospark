//! VibeVoice streaming TTS family.
//!
//! Port of `mlx_audio/tts/models/vibevoice/` at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/vibevoice):
//! a split Qwen2 decoder drives a latent diffusion loop. The lower
//! backbone layers (`language_model`, no final norm) embed the text in
//! windows of [`TTS_TEXT_WINDOW_SIZE`] tokens; their hidden states plus
//! a "text" type embedding feed the upper layers (`tts_language_model`,
//! with the final norm), whose last-position state conditions a
//! diffusion prediction head under the deterministic DPM-Solver++
//! scheduler (v-prediction, cosine betas, classifier-free guidance
//! against a zeros-input negative branch). Each sampled acoustic
//! latent feeds back through the speech connector as a "speech" type
//! embedding row, a binary classifier over the last hidden state ends
//! the run at p > 0.5, and the concatenated latents are unscaled by
//! the checkpoint's speech scaling/bias factors and decoded to a
//! waveform by the convolutional VAE acoustic tokenizer decoder.
//!
//! Scope of this port, matching the reference single-prompt path:
//! batch 1, non-streaming (the `stream` chunks are a decode-schedule
//! wrapper around the same latents), and caller-owned text
//! tokenization (the reference encodes `text.strip() + "\n"` without
//! special tokens; the caller encodes that string and hands over the
//! ids). Diffusion noise comes from a [`DiffusionNoise`] provider so
//! fixture tests replay the recorded reference draws and standalone
//! runs can seed the crate-local threefry RNG. Optional voice
//! conditioning loads the exported `voices/<name>.safetensors` caches
//! ([`Model::load_voice`]). The multi-speaker dialogue wrapper is a
//! caller loop over single segments. See [README.md](README.md) for
//! the verification evidence and open items.

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use self::acoustic_decoder::AcousticDecoder;
use self::diffusion_head::DiffusionHead;
use self::scheduler::DpmSolver;
use crate::lm::engine::{self, Backbone, Input};
use crate::lm::{qwen2, Weights};
use crate::music::minimax_music3::rng;
use crate::nn::Linear;
use crate::nn::RmsNorm;
use crate::{quant, Result, SpeechError};

mod acoustic_decoder;
mod diffusion_head;
mod scheduler;

#[cfg(test)]
mod tests;

/// Text tokens enter the upper backbone in windows of this many ids.
pub const TTS_TEXT_WINDOW_SIZE: usize = 5;
/// Speech latents sampled per outer generation round.
pub const TTS_SPEECH_WINDOW_SIZE: usize = 6;

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
        // decoder_ratios fall back to the encoder ratios; decoder_depths
        // to the reversed encoder depths.
        let ratios: Vec<usize> = match value.get("decoder_ratios") {
            Some(Value::Array(list)) if !list.is_empty() => list
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect(),
            _ => value
                .get("encoder_ratios")
                .and_then(Value::as_array)
                .map(|list| {
                    list.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_else(|| vec![8, 5, 5, 4, 2, 2]),
        };
        let depths: Vec<usize> = match value.get("decoder_depths") {
            Some(Value::String(text)) => text
                .split('-')
                .filter_map(|part| part.parse::<usize>().ok())
                .collect(),
            Some(Value::Array(list)) if !list.is_empty() => list
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
            hidden_size: i("hidden_size", 896),
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
            ddpm_num_inference_steps: i("ddpm_num_inference_steps", 20),
            ddpm_beta_schedule: value
                .get("ddpm_beta_schedule")
                .and_then(Value::as_str)
                .unwrap_or("cosine")
                .to_owned(),
        }
    }
}

/// The model config (`ModelConfig.from_dict`).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub sample_rate: u32,
    pub acoustic_tokenizer: AcousticTokenizerConfig,
    pub decoder: crate::lm::DenseConfig,
    pub diffusion_head: DiffusionHeadConfig,
    pub acoustic_vae_dim: usize,
    /// Upper-backbone layer count; the lower backbone takes the rest.
    pub tts_backbone_num_hidden_layers: usize,
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
        // The reference decoder rotary always builds default rope; any
        // declared rope_scaling is ignored, so it is dropped here.
        let mut decoder = qwen2::from_json(&decoder_value)?;
        decoder.rope_scaling = None;
        Ok(Self {
            sample_rate: value
                .get("sample_rate")
                .and_then(Value::as_u64)
                .unwrap_or(24000) as u32,
            acoustic_tokenizer: AcousticTokenizerConfig::from_json(&acoustic_value),
            decoder,
            diffusion_head: DiffusionHeadConfig::from_json(&diffusion_value),
            acoustic_vae_dim: value
                .get("acoustic_vae_dim")
                .and_then(Value::as_u64)
                .unwrap_or(64) as usize,
            tts_backbone_num_hidden_layers: value
                .get("tts_backbone_num_hidden_layers")
                .and_then(Value::as_u64)
                .unwrap_or(20) as usize,
        })
    }
}

// ---------------------------------------------------------------------------
// Noise providers
// ---------------------------------------------------------------------------

/// The diffusion noise source: one standard-normal draw of
/// `[rows, cols]` per call, in reference draw order (the initial latent
/// noise of a `sample_speech_tokens` call).
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
// Family parts
// ---------------------------------------------------------------------------

/// The speech connector (`SpeechConnector`): fc1 -> RMSNorm -> fc2,
/// both linears biased.
struct SpeechConnector {
    fc1: Linear,
    fc2: Linear,
    norm: RmsNorm,
}

impl SpeechConnector {
    fn load(weights: &Weights, input_dim: usize, output_dim: usize, eps: f32) -> Result<Self> {
        Ok(Self {
            // The connector tensors load prefix-free; any view works.
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

/// The end-of-speech head (`BinaryClassifier`): relu(fc1) then fc2 to
/// one logit, both linears biased.
struct BinaryClassifier {
    fc1: Linear,
    fc2: Linear,
}

impl BinaryClassifier {
    fn load(weights: &Weights, hidden: usize) -> Result<Self> {
        Ok(Self {
            fc1: weights.linear_absolute(
                "tts_eos_classifier.fc1",
                hidden,
                hidden,
                crate::lm::Bias::Required,
            )?,
            fc2: weights.linear_absolute(
                "tts_eos_classifier.fc2",
                hidden,
                1,
                crate::lm::Bias::Required,
            )?,
        })
    }

    /// The end-of-speech logit for one hidden row.
    fn logit(&self, hidden_row: &[f32]) -> f32 {
        let mut h = self.fc1.forward(hidden_row, 1);
        for value in &mut h {
            *value = value.max(0.0);
        }
        self.fc2.forward(&h, 1)[0]
    }
}

/// Reads a tensor from a voice safetensors file by exact name.
fn voice_tensor(file: &SafetensorsFile, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let descriptor = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_owned(),
        why: "voice cache is missing this tensor".into(),
    })?;
    let shape = descriptor.shape.clone();
    Ok((shape, file.load_as_f32(name)?))
}

/// One layer's cached keys and values, head-major
/// `[kv_heads, rows, head_dim]`.
struct KvRows {
    keys: Vec<f32>,
    values: Vec<f32>,
}

/// Normalizes one cached key or value tensor to head-major
/// `[kv_heads, rows, head_dim]`. The reference transposes every 4D
/// tensor on load, so Swift-style files store
/// `(B, kv_heads, rows, head_dim)`; files without the batch axis
/// (`ndim == 3`) are used as-is, which at batch 1 means
/// `(rows, kv_heads, head_dim)` row-major.
fn voice_kv(name: &str, shape: &[usize], data: Vec<f32>) -> Result<Vec<f32>> {
    match shape {
        // (1, kv_heads, rows, head_dim): already head-major once the
        // batch axis is dropped.
        [1, kv, rows, dim] => {
            debug_assert_eq!(kv * rows * dim, data.len());
            Ok(data)
        }
        // (rows, kv_heads, head_dim): rotate to head-major.
        [rows, kv, dim] => {
            let (rows, kv, dim) = (*rows, *kv, *dim);
            let mut out = vec![0.0f32; data.len()];
            for r in 0..rows {
                for kvh in 0..kv {
                    for d in 0..dim {
                        out[(kvh * rows + r) * dim + d] = data[(r * kv + kvh) * dim + d];
                    }
                }
            }
            Ok(out)
        }
        other => Err(SpeechError::Tensor {
            name: name.to_owned(),
            why: format!("expected a 3D or Swift-style 4D cache tensor, got shape {other:?}"),
        }),
    }
}

/// Reads one keys/values pair per layer from the voice file.
fn load_voice_kv(file: &SafetensorsFile, prefix: &str, layers: usize) -> Result<Vec<KvRows>> {
    (0..layers)
        .map(|i| {
            let (k_shape, k_data) = voice_tensor(file, &format!("{prefix}_key_{i}"))?;
            let (v_shape, v_data) = voice_tensor(file, &format!("{prefix}_value_{i}"))?;
            Ok(KvRows {
                keys: voice_kv(&format!("{prefix}_key_{i}"), &k_shape, k_data)?,
                values: voice_kv(&format!("{prefix}_value_{i}"), &v_shape, v_data)?,
            })
        })
        .collect()
}

/// Reads a hidden-state tensor `[1, rows, hidden]` from the voice file.
fn load_voice_hidden(
    file: &SafetensorsFile,
    name: &str,
    hidden: usize,
) -> Result<(usize, Vec<f32>)> {
    let (shape, data) = voice_tensor(file, name)?;
    if shape.len() != 3 || shape[0] != 1 || shape[2] != hidden {
        return Err(SpeechError::Tensor {
            name: name.to_owned(),
            why: format!("expected [1, rows, {hidden}]"),
        });
    }
    Ok((shape[1], data))
}

/// One loaded voice cache (`load_voice`): the upper-backbone hidden
/// states the run starts from plus the per-layer key/value rows for the
/// positive and negative branches.
struct VoiceCache {
    tts_hidden: Vec<f32>,
    neg_tts_hidden: Vec<f32>,
    lm_kv: Vec<KvRows>,
    tts_kv: Vec<KvRows>,
    neg_tts_kv: Vec<KvRows>,
    rows: usize,
}

/// Seeds one appending cache with precomputed key/value rows.
fn seed_cache(
    cache: &mut crate::lm::KvCache,
    rows: &KvRows,
    kv_heads: usize,
    len: usize,
    dim: usize,
) -> Result<()> {
    if rows.keys.len() != kv_heads * len * dim || rows.values.len() != kv_heads * len * dim {
        return Err(SpeechError::Tensor {
            name: "voice cache".into(),
            why: format!(
                "cached rows {} do not match {kv_heads} heads x {len} x {dim}",
                rows.keys.len()
            ),
        });
    }
    let _ = cache.update_and_fetch(&rows.keys, &rows.values, kv_heads, len, dim);
    Ok(())
}

// ---------------------------------------------------------------------------
// The loaded model
// ---------------------------------------------------------------------------

/// Reads an optional checkpoint scalar. An absent tensor keeps
/// `default`; a present but unreadable tensor is an error.
fn optional_scalar(weights: &Weights, name: &str, default: f32) -> Result<f32> {
    if !weights.contains_absolute(name) {
        return Ok(default);
    }
    Ok(weights
        .tensor_absolute(name, &[])?
        .first()
        .copied()
        .unwrap_or(default))
}

/// Opens the checkpoint and checks the backbone naming the converted
/// saves use (unprefixed, or under the raw composite `model.` prefix).
fn open_backbone_weights(dir: &std::path::Path) -> Result<Weights> {
    let probe = Weights::open(dir)?;
    if probe.contains_absolute("language_model.embed_tokens.weight") {
        return Ok(probe);
    }
    if probe.contains_absolute("model.language_model.embed_tokens.weight") {
        return Weights::open(dir).map(|w| w.with_prefix("model."));
    }
    Err(SpeechError::Tensor {
        name: "language_model.embed_tokens.weight".into(),
        why: "checkpoint carries no language_model embed table".into(),
    })
}

/// The loaded VibeVoice model.
pub struct Model {
    pub config: ModelConfig,
    /// The lower backbone layers (no final norm, with the embed table).
    language_model: Backbone,
    /// The upper backbone layers (final norm, no embed table).
    tts_language_model: Backbone,
    /// Type embeddings: row 0 is speech, row 1 is text.
    tts_input_types: Vec<f32>,
    pub(crate) acoustic_decoder: AcousticDecoder,
    acoustic_connector: SpeechConnector,
    prediction_head: DiffusionHead,
    tts_eos_classifier: BinaryClassifier,
    scheduler: DpmSolver,
    speech_scaling_factor: f32,
    speech_bias_factor: f32,
    voice: Option<VoiceCache>,
}

impl Model {
    /// Opens a checkpoint directory (`config.json` + safetensors).
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        let parsed = ModelConfig::from_json(&config_value)?;
        // The family tensors load from the unprefixed view; the two
        // backbone halves each take their own prefixed view.
        let base = open_backbone_weights(dir)?;
        let hidden = parsed.decoder.hidden_size;
        let acoustic_config = &parsed.acoustic_tokenizer;
        let acoustic_decoder = AcousticDecoder::load(
            base.shards(),
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
            &base,
            parsed.acoustic_vae_dim,
            hidden,
            parsed.decoder.rms_norm_eps,
        )?;
        let head_config = &parsed.diffusion_head;
        let prediction_head = DiffusionHead::load(
            base.shards(),
            "prediction_head",
            head_config.hidden_size,
            head_config.head_layers,
            head_config.head_ffn_ratio,
            head_config.rms_norm_eps,
            head_config.latent_size,
        )?;
        let tts_eos_classifier = BinaryClassifier::load(&base, hidden)?;
        let scheduler = DpmSolver::new(
            head_config.ddpm_num_steps,
            &head_config.ddpm_beta_schedule,
            &head_config.prediction_type,
        )?;
        let speech_scaling_factor = optional_scalar(&base, "speech_scaling_factor", 1.0)?;
        let speech_bias_factor = optional_scalar(&base, "speech_bias_factor", 0.0)?;
        let tts_input_types = base.tensor_absolute("tts_input_types.weight", &[2, hidden])?;

        if parsed.tts_backbone_num_hidden_layers == 0
            || parsed.tts_backbone_num_hidden_layers >= parsed.decoder.num_hidden_layers
        {
            return Err(SpeechError::BadConfig {
                field: "tts_backbone_num_hidden_layers".into(),
                why: format!(
                    "must be between 1 and {} (got {})",
                    parsed.decoder.num_hidden_layers - 1,
                    parsed.tts_backbone_num_hidden_layers
                ),
            });
        }
        let lm_layers = parsed.decoder.num_hidden_layers - parsed.tts_backbone_num_hidden_layers;
        let tts_layers = parsed.tts_backbone_num_hidden_layers;

        // Lower half: embed table + layers, no final norm. The reference
        // never reads a lower-half lm_head, so the config declares the
        // head tied to skip loading it.
        let mut lm_config = parsed.decoder.clone();
        lm_config.num_hidden_layers = lm_layers;
        lm_config.tie_word_embeddings = true;
        let lm_weights = open_backbone_weights(dir)?.with_prefix("language_model.");
        let language_model = Backbone::load_without_final_norm(&lm_weights, lm_config)?;

        // Upper half: layers + final norm; the reused embed table in the
        // checkpoint is never read, so it is not loaded.
        let mut tts_config = parsed.decoder.clone();
        tts_config.num_hidden_layers = tts_layers;
        tts_config.tie_word_embeddings = true;
        let tts_weights = open_backbone_weights(dir)?.with_prefix("tts_language_model.");
        let tts_language_model = Backbone::load_without_embed_table(&tts_weights, tts_config)?;

        Ok(Self {
            config: parsed,
            language_model,
            tts_language_model,
            tts_input_types,
            acoustic_decoder,
            acoustic_connector,
            prediction_head,
            tts_eos_classifier,
            scheduler,
            speech_scaling_factor,
            speech_bias_factor,
            voice: None,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Loads a VibeVoice voice cache (`voices/<name>.safetensors`):
    /// the upper-backbone hidden states plus per-layer key/value rows
    /// for the positive and negative branches. The `lm_hidden` and
    /// `neg_lm_*` entries the format also carries are never read by the
    /// reference generation loop and are not loaded here.
    pub fn load_voice(&mut self, path: &std::path::Path) -> Result<()> {
        let file = SafetensorsFile::open(path)?;
        let hidden = self.config.decoder.hidden_size;
        let (rows, tts_hidden) = load_voice_hidden(&file, "tts_lm_hidden", hidden)?;
        let (_, neg_tts_hidden) = load_voice_hidden(&file, "neg_tts_lm_hidden", hidden)?;
        let lm_kv = load_voice_kv(&file, "lm", self.language_model.config.num_hidden_layers)?;
        let tts_kv = load_voice_kv(
            &file,
            "tts_lm",
            self.tts_language_model.config.num_hidden_layers,
        )?;
        let neg_tts_kv = load_voice_kv(
            &file,
            "neg_tts_lm",
            self.tts_language_model.config.num_hidden_layers,
        )?;
        self.voice = Some(VoiceCache {
            tts_hidden,
            neg_tts_hidden,
            lm_kv,
            tts_kv,
            neg_tts_kv,
            rows,
        });
        Ok(())
    }

    /// The type-embedding rows for one position (`tts_input_types(id)`).
    fn type_embed(&self, type_id: i32, rows: usize) -> Result<Vec<f32>> {
        let hidden = self.config.decoder.hidden_size;
        let row = usize::try_from(type_id).map_err(|_| SpeechError::Input {
            why: "negative type id".into(),
        })?;
        if row >= 2 {
            return Err(SpeechError::Input {
                why: format!("type id {type_id} outside the 2-row table"),
            });
        }
        let source = &self.tts_input_types[row * hidden..(row + 1) * hidden];
        let mut out = Vec::with_capacity(rows * hidden);
        for _ in 0..rows {
            out.extend_from_slice(source);
        }
        Ok(out)
    }

    /// `sample_speech_tokens`: the deterministic DPM-Solver++ diffusion
    /// from pure noise, with classifier-free guidance. The guided pair
    /// stays in lockstep so both scheduler halves evolve identically;
    /// the port steps one half. One initial noise draw, in reference
    /// order.
    fn sample_speech_tokens(
        &mut self,
        condition: &[f32],
        neg_condition: &[f32],
        cfg_scale: f32,
        ddpm_steps: Option<usize>,
        noise: &mut dyn DiffusionNoise,
    ) -> Result<Vec<f32>> {
        let vae_dim = self.config.acoustic_vae_dim;
        let steps = ddpm_steps.unwrap_or(self.config.diffusion_head.ddpm_num_inference_steps);
        self.scheduler.reset();
        self.scheduler.set_timesteps(steps);
        let timesteps: Vec<i64> = self.scheduler.timesteps().to_vec();

        let mut combined_condition = condition.to_vec();
        combined_condition.extend_from_slice(neg_condition);
        let mut speech = noise.normal(1, vae_dim);
        let mut prev_x0: Option<Vec<f32>> = None;
        for &t in &timesteps {
            let mut combined_speech = speech.clone();
            combined_speech.extend_from_slice(&speech);
            let t_batch = vec![t as f32; 2];
            let eps =
                self.prediction_head
                    .forward(&combined_speech, &t_batch, &combined_condition, 2)?;
            let width = eps.len() / 2;
            let mut guided = vec![0.0f32; width];
            for col in 0..width {
                let cond = eps[col];
                let uncond = eps[width + col];
                guided[col] = uncond + cfg_scale * (cond - uncond);
            }
            let (next, x0_pred) = self.scheduler.step(&guided, &speech, prev_x0.as_deref())?;
            speech = next;
            prev_x0 = Some(x0_pred);
        }
        Ok(speech)
    }

    /// The reference `generate` for batch 1: the text-window and
    /// speech-window loop over the split backbones with the per-step
    /// latent sampling and feedback, ending on the binary EOS
    /// classifier or `max_tokens`. `text_token_ids` is the
    /// caller-encoded text (the reference encodes `text.strip() + "\n"`
    /// without special tokens).
    pub fn generate(
        &mut self,
        text_token_ids: &[i32],
        params: &GenerateParams,
        noise: &mut dyn DiffusionNoise,
    ) -> Result<Generated> {
        if params.ddpm_steps == Some(0) {
            return Err(SpeechError::Input {
                why: "ddpm_steps must be at least 1".into(),
            });
        }
        let hidden = self.config.decoder.hidden_size;
        let lm_layers = self.language_model.config.num_hidden_layers;
        let tts_layers = self.tts_language_model.config.num_hidden_layers;
        let kv_heads = self.config.decoder.num_key_value_heads;
        let head_dim = self.config.decoder.head_dim;

        // Caches: empty, or seeded from the loaded voice.
        let mut lm_cache = crate::lm::make_prompt_cache(lm_layers);
        let mut tts_cache = crate::lm::make_prompt_cache(tts_layers);
        let mut neg_tts_cache = crate::lm::make_prompt_cache(tts_layers);
        let mut tts_hidden: Vec<f32> = Vec::new();
        let mut neg_hidden: Vec<f32> = Vec::new();
        if let Some(voice) = &self.voice {
            if voice.lm_kv.len() != lm_layers
                || voice.tts_kv.len() != tts_layers
                || voice.neg_tts_kv.len() != tts_layers
            {
                return Err(SpeechError::Tensor {
                    name: "voice cache".into(),
                    why: "layer counts do not match the model".into(),
                });
            }
            for (cache, rows) in lm_cache.iter_mut().zip(&voice.lm_kv) {
                seed_cache(cache, rows, kv_heads, voice.rows, head_dim)?;
            }
            for (cache, rows) in tts_cache.iter_mut().zip(&voice.tts_kv) {
                seed_cache(cache, rows, kv_heads, voice.rows, head_dim)?;
            }
            for (cache, rows) in neg_tts_cache.iter_mut().zip(&voice.neg_tts_kv) {
                seed_cache(cache, rows, kv_heads, voice.rows, head_dim)?;
            }
            tts_hidden = voice.tts_hidden.clone();
            neg_hidden = voice.neg_tts_hidden.clone();
        }

        let mut speech_latents: Vec<f32> = Vec::new();
        let mut latent_count = 0usize;
        let mut text_pos = 0usize;
        let mut step = 0usize;
        let mut finished = false;

        while !finished && step < params.max_tokens {
            if text_pos < text_token_ids.len() {
                let end = (text_pos + TTS_TEXT_WINDOW_SIZE).min(text_token_ids.len());
                let window = end - text_pos;
                let text_embeds = engine::lookup_embeddings(
                    self.language_model.embed_table(),
                    self.config.decoder.vocab_size,
                    hidden,
                    &text_token_ids[text_pos..end],
                )?;
                let lm_out = self
                    .language_model
                    .forward_hidden(Input::Embeddings(&text_embeds), Some(&mut lm_cache))?;
                // Text rows carry the type-1 embedding.
                let type_embed = self.type_embed(1, window)?;
                let mut tts_in = vec![0.0f32; lm_out.len()];
                for (slot, value) in tts_in.iter_mut().enumerate() {
                    *value = lm_out[slot] + type_embed[slot];
                }
                let tts_out = self
                    .tts_language_model
                    .forward_hidden(Input::Embeddings(&tts_in), Some(&mut tts_cache))?;
                tts_hidden.extend_from_slice(&tts_out);

                // The negative branch starts from zeros plus the text
                // type embedding; with a loaded voice its state comes
                // from the cache and the text windows skip it.
                if neg_hidden.is_empty() {
                    let neg_in = self.type_embed(1, window)?;
                    let neg_out = self
                        .tts_language_model
                        .forward_hidden(Input::Embeddings(&neg_in), Some(&mut neg_tts_cache))?;
                    neg_hidden.extend_from_slice(&neg_out);
                }
                text_pos = end;
            }
            if tts_hidden.is_empty() || neg_hidden.is_empty() {
                break;
            }

            for _ in 0..TTS_SPEECH_WINDOW_SIZE {
                let condition = tts_hidden[tts_hidden.len() - hidden..].to_vec();
                let neg_condition = neg_hidden[neg_hidden.len() - hidden..].to_vec();
                let latent = self.sample_speech_tokens(
                    &condition,
                    &neg_condition,
                    params.cfg_scale,
                    params.ddpm_steps,
                    noise,
                )?;
                speech_latents.extend_from_slice(&latent);
                latent_count += 1;

                // Feed the latent back through the connector with the
                // speech (0) type embedding, on both branches.
                let acoustic_embed = self.acoustic_connector.forward(&latent, 1);
                let speech_type = self.type_embed(0, 1)?;
                let mut tts_in = acoustic_embed;
                for (index, value) in tts_in.iter_mut().enumerate() {
                    *value += speech_type[index];
                }
                let tts_out = self
                    .tts_language_model
                    .forward_hidden(Input::Embeddings(&tts_in), Some(&mut tts_cache))?;
                tts_hidden.extend_from_slice(&tts_out);
                let neg_out = self
                    .tts_language_model
                    .forward_hidden(Input::Embeddings(&tts_in), Some(&mut neg_tts_cache))?;
                neg_hidden.extend_from_slice(&neg_out);

                let eos_logit = self
                    .tts_eos_classifier
                    .logit(&tts_out[tts_out.len() - hidden..]);
                if 1.0 / (1.0 + (-eos_logit).exp()) > 0.5 {
                    finished = true;
                    break;
                }
                step += 1;
                if step >= params.max_tokens {
                    finished = true;
                    break;
                }
            }
        }

        if latent_count == 0 {
            return Ok(Generated {
                latents: Vec::new(),
                latent_frames: 0,
                samples: Vec::new(),
                sample_rate: self.config.sample_rate,
                token_count: 0,
            });
        }

        // Unscale and decode.
        let mut latent_seq = speech_latents;
        for value in &mut latent_seq {
            *value = *value / self.speech_scaling_factor - self.speech_bias_factor;
        }
        let samples = self.acoustic_decoder.forward(&latent_seq, latent_count)?;
        Ok(Generated {
            latents: latent_seq,
            latent_frames: latent_count,
            samples,
            sample_rate: self.config.sample_rate,
            token_count: latent_count,
        })
    }
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
            cfg_scale: 1.5,
            max_tokens: 512,
            ddpm_steps: None,
        }
    }
}

/// One generation result.
#[derive(Debug, Clone)]
pub struct Generated {
    /// The unscaled latents `[frames, acoustic_vae_dim]` (the decoder
    /// input, after the scaling/bias correction).
    pub latents: Vec<f32>,
    pub latent_frames: usize,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// The reference `token_count`: the number of sampled speech
    /// latents, equal to `latent_frames`.
    pub token_count: usize,
}
