//! CLAP HTSAT-unfused (`laion/clap-htsat-unfused`): a contrastive dual
//! tower mapping 48 kHz audio and text into one L2-normalized 512-dim
//! space.
//!
//! - [`features`]: `ClapFeatureExtractor` (rand_trunc + repeatpad, Slaney
//!   log-mel in dB), computed in f64 like the numpy reference.
//! - [`audio`]: the HTSAT Swin encoder, pooled by token mean.
//! - [`text`]: the RoBERTa encoder with its tanh pooler.
//! - [`ClapModel`]: both towers plus the `Linear -> ReLU -> Linear`
//!   projection heads and `logit_scale_a` / `logit_scale_t`.
//! - [`Clap`]: model + feature extractor + tokenizer, the end-user API.
//!
//! Normative reference: transformers 4.57.1 `ClapModel` in fp32. See
//! `src/clap/README.md` for the pinned revision, verified stages, and
//! divergences (deterministic long-clip crop, f64 frontend).

pub mod audio;
pub mod config;
pub mod features;
pub mod text;

#[cfg(test)]
mod tests;

use std::path::Path;

use turbospark_audio::nn::Linear;
use turbospark_audio::ops;

pub use audio::HtsatEncoder;
pub use config::{AudioConfig, ClapConfig, FeatureConfig, TextConfig};
pub use features::{load_wav_mono, ClapFeatureExtractor, LongAudioCrop, MelFeatures};
pub use text::RobertaEncoder;

use crate::error::input_error;
use crate::normalize::l2_normalize;
use crate::similarity::{dot, rank_descending};
use crate::tokenize::TextTokenizer;
use crate::weights::Weights;
use crate::Result;

/// Sample rate the unfused checkpoint's frontend expects.
pub const SAMPLE_RATE: u32 = 48_000;

/// The zero-shot audio classification pipeline's default
/// `hypothesis_template`. [`Clap::classify`] uses labels verbatim; format
/// them with this to reproduce the pipeline's prompts.
pub const PIPELINE_TEMPLATE: &str = "This is a sound of {}.";

const AUDIO_PREFIX: &str = "audio_model.audio_encoder.";
const TEXT_PREFIX: &str = "text_model.";

/// `ClapProjectionLayer`: `linear2(relu(linear1(x)))`.
struct Projection {
    linear1: Linear,
    linear2: Linear,
}

impl Projection {
    fn load(w: &Weights, base: &str, input: usize, dim: usize) -> Result<Self> {
        Ok(Self {
            linear1: w.linear(&format!("{base}.linear1"), input, dim, true, None)?,
            linear2: w.linear(&format!("{base}.linear2"), dim, dim, true, None)?,
        })
    }

    /// Projected and L2-normalized (`F.normalize`, eps 1e-12).
    fn embed(&self, pooled: &[f32]) -> Vec<f32> {
        let mut h = self.linear1.forward(pooled, 1);
        for v in h.iter_mut() {
            *v = v.max(0.0);
        }
        let mut out = self.linear2.forward(&h, 1);
        l2_normalize(&mut out);
        out
    }
}

/// Both towers, projections, and logit scales of a `ClapModel`
/// checkpoint. Tokenization and feature extraction live in [`Clap`].
pub struct ClapModel {
    config: ClapConfig,
    audio: HtsatEncoder,
    audio_projection: Projection,
    text: RobertaEncoder,
    text_projection: Projection,
    logit_scale_a: f32,
    logit_scale_t: f32,
}

impl std::fmt::Debug for ClapModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClapModel")
            .field("audio", &self.audio)
            .field("text", &self.text)
            .field("projection_dim", &self.config.projection_dim)
            .finish_non_exhaustive()
    }
}

impl ClapModel {
    /// Reads `config.json` and the safetensors weights under `dir`.
    pub fn load(dir: &Path) -> Result<Self> {
        let config = ClapConfig::from_file(&dir.join("config.json"))?;
        Self::from_weights(config, &Weights::open(dir)?)
    }

    pub fn from_weights(config: ClapConfig, w: &Weights) -> Result<Self> {
        let audio = HtsatEncoder::load(w, &config.audio, AUDIO_PREFIX)?;
        let text = RobertaEncoder::load(w, &config.text, TEXT_PREFIX)?;
        let dim = config.projection_dim;
        let audio_projection =
            Projection::load(w, "audio_projection", config.audio.hidden_size, dim)?;
        let text_projection = Projection::load(w, "text_projection", config.text.hidden_size, dim)?;
        let scalar = |n: &str| -> Result<f32> { Ok(w.load(n, &[])?[0]) };
        Ok(Self {
            logit_scale_a: scalar("logit_scale_a")?,
            logit_scale_t: scalar("logit_scale_t")?,
            config,
            audio,
            audio_projection,
            text,
            text_projection,
        })
    }

    pub fn config(&self) -> &ClapConfig {
        &self.config
    }

    pub fn audio_encoder(&self) -> &HtsatEncoder {
        &self.audio
    }

    pub fn text_encoder(&self) -> &RobertaEncoder {
        &self.text
    }

    /// `exp(logit_scale_a)`, the scale on `logits_per_audio`.
    pub fn audio_logit_scale(&self) -> f32 {
        self.logit_scale_a.exp()
    }

    /// `exp(logit_scale_t)`, the scale on `logits_per_text`.
    pub fn text_logit_scale(&self) -> f32 {
        self.logit_scale_t.exp()
    }

    /// `get_audio_features`: unit-norm `[projection_dim]` embedding.
    pub fn audio_embedding(&self, mel: &MelFeatures) -> Result<Vec<f32>> {
        self.audio_embedding_with(&mel.data, mel.frames, mel.mels, &mut |_, _| {})
    }

    /// [`Self::audio_embedding`] over raw `[frames, mels]` features, with
    /// the encoder's intermediate hook (see [`HtsatEncoder::forward_with`]).
    pub fn audio_embedding_with(
        &self,
        features: &[f32],
        frames: usize,
        mels: usize,
        hook: &mut dyn FnMut(&str, &[f32]),
    ) -> Result<Vec<f32>> {
        let pooled = self.audio.forward_with(features, frames, mels, hook)?;
        Ok(self.audio_projection.embed(&pooled))
    }

    /// `get_text_features` for one unpadded id sequence.
    pub fn text_embedding(&self, ids: &[u32]) -> Result<Vec<f32>> {
        self.text_embedding_with(ids, &mut |_, _| {})
    }

    pub fn text_embedding_with(
        &self,
        ids: &[u32],
        hook: &mut dyn FnMut(&str, &[f32]),
    ) -> Result<Vec<f32>> {
        let pooled = self.text.forward_with(ids, hook)?;
        Ok(self.text_projection.embed(&pooled))
    }

    /// `logits_per_audio` row for one audio embedding against `texts`
    /// (row-major `[n, projection_dim]`): `exp(logit_scale_a) * a . t`.
    pub fn logits_per_audio(&self, audio: &[f32], texts: &[f32]) -> Result<Vec<f32>> {
        let d = self.config.projection_dim;
        if audio.len() != d || texts.len() % d != 0 {
            return Err(input_error(format!(
                "embeddings must be {d} wide (audio {}, texts {})",
                audio.len(),
                texts.len()
            )));
        }
        let scale = self.audio_logit_scale();
        Ok(texts
            .chunks_exact(d)
            .map(|t| dot(audio, t) * scale)
            .collect())
    }
}

/// End-user CLAP: [`ClapModel`] plus the feature extractor and tokenizer
/// from the same checkpoint directory.
pub struct Clap {
    model: ClapModel,
    features: ClapFeatureExtractor,
    tokenizer: TextTokenizer,
}

impl std::fmt::Debug for Clap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Clap")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl Clap {
    /// Loads `config.json`, `preprocessor_config.json`, `tokenizer.json`,
    /// and `model.safetensors` from `dir`.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let model = ClapModel::load(dir)?;
        let features = ClapFeatureExtractor::from_file(&dir.join("preprocessor_config.json"))?;
        if features.config().feature_size != model.config().audio.num_mel_bins {
            return Err(crate::error::bad_config(
                "preprocessor_config.feature_size",
                format!(
                    "{} != audio_config.num_mel_bins {}",
                    features.config().feature_size,
                    model.config().audio.num_mel_bins
                ),
            ));
        }
        let frames = features.config().frames();
        let width = model.audio_encoder().geometry().spec_width;
        if frames > width {
            return Err(crate::error::bad_config(
                "preprocessor_config.max_length_s",
                format!("{frames} frames exceed the encoder's {width}"),
            ));
        }
        let tokenizer = TextTokenizer::from_file(&dir.join("tokenizer.json"))?;
        Ok(Self {
            model,
            features,
            tokenizer,
        })
    }

    pub fn model(&self) -> &ClapModel {
        &self.model
    }

    pub fn feature_extractor(&self) -> &ClapFeatureExtractor {
        &self.features
    }

    pub fn tokenizer(&self) -> &TextTokenizer {
        &self.tokenizer
    }

    /// Unit-norm audio embedding of mono PCM at 48 kHz. Clips longer than
    /// 10 s keep the first 10 s ([`LongAudioCrop::Head`]).
    pub fn embed_audio(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        self.embed_audio_with(pcm, LongAudioCrop::Head)
    }

    pub fn embed_audio_with(&self, pcm: &[f32], crop: LongAudioCrop) -> Result<Vec<f32>> {
        let mel = self.features.extract(pcm, crop)?;
        self.model.audio_embedding(&mel)
    }

    /// Loads a WAV file, mixes to mono, resamples to 48 kHz (see
    /// [`load_wav_mono`] for the resampler caveat), and embeds it.
    pub fn embed_audio_file(&self, path: &Path) -> Result<Vec<f32>> {
        let pcm = load_wav_mono(path, self.features.config().sampling_rate)?;
        self.embed_audio(&pcm)
    }

    /// Token ids `text` encodes to, with `<s>` and `</s>`.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(text, true)
    }

    /// Unit-norm text embedding. Text longer than the position table
    /// (512 tokens) is refused, not truncated.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        self.model.text_embedding(&self.tokenize(text)?)
    }

    /// Zero-shot classification as the HF pipeline scores it:
    /// `softmax(exp(logit_scale_a) * cos(audio, label))` over `labels`,
    /// sorted by descending probability (ties keep label order). Labels are
    /// embedded verbatim; see [`PIPELINE_TEMPLATE`].
    pub fn classify(&self, audio_embedding: &[f32], labels: &[&str]) -> Result<Vec<(String, f32)>> {
        if labels.is_empty() {
            return Err(input_error("no labels"));
        }
        let mut texts = Vec::with_capacity(labels.len() * self.model.config.projection_dim);
        for label in labels {
            texts.extend(self.embed_text(label)?);
        }
        let mut probs = self.model.logits_per_audio(audio_embedding, &texts)?;
        ops::softmax_row(&mut probs);
        Ok(rank_descending(&probs)
            .into_iter()
            .map(|i| (labels[i].to_string(), probs[i]))
            .collect())
    }
}
