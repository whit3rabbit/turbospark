//! Echo-TTS configuration (`config.py`): the DiT geometry, the sampler
//! defaults, and the model-level fields, with the reference `from_dict`
//! defaults.

use serde_json::Value;

use crate::{Result, SpeechError};

/// `EchoDiTConfig`.
#[derive(Debug, Clone)]
pub struct EchoDiTConfig {
    pub latent_size: usize,
    pub model_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub norm_eps: f32,
    pub text_vocab_size: usize,
    pub text_model_size: usize,
    pub text_num_layers: usize,
    pub text_num_heads: usize,
    pub text_intermediate_size: usize,
    pub speaker_patch_size: usize,
    pub speaker_model_size: usize,
    pub speaker_num_layers: usize,
    pub speaker_num_heads: usize,
    pub speaker_intermediate_size: usize,
    pub timestep_embed_size: usize,
    pub adaln_rank: usize,
    /// `enable_blockwise_modules`: the negated
    /// `delete_blockwise_modules` model flag.
    pub enable_blockwise_modules: bool,
}

impl Default for EchoDiTConfig {
    fn default() -> Self {
        EchoDiTConfig {
            latent_size: 80,
            model_size: 2048,
            num_layers: 24,
            num_heads: 16,
            intermediate_size: 5888,
            norm_eps: 1e-5,
            text_vocab_size: 256,
            text_model_size: 1280,
            text_num_layers: 14,
            text_num_heads: 10,
            text_intermediate_size: 3328,
            speaker_patch_size: 4,
            speaker_model_size: 1280,
            speaker_num_layers: 14,
            speaker_num_heads: 10,
            speaker_intermediate_size: 3328,
            timestep_embed_size: 512,
            adaln_rank: 256,
            enable_blockwise_modules: true,
        }
    }
}

impl EchoDiTConfig {
    fn from_dict(value: &Value) -> Self {
        let mut config = Self::default();
        let u = |key: &str| value.get(key).and_then(Value::as_u64).map(|n| n as usize);
        if let Some(v) = u("latent_size") {
            config.latent_size = v;
        }
        if let Some(v) = u("model_size") {
            config.model_size = v;
        }
        if let Some(v) = u("num_layers") {
            config.num_layers = v;
        }
        if let Some(v) = u("num_heads") {
            config.num_heads = v;
        }
        if let Some(v) = u("intermediate_size") {
            config.intermediate_size = v;
        }
        if let Some(v) = value.get("norm_eps").and_then(Value::as_f64) {
            config.norm_eps = v as f32;
        }
        if let Some(v) = u("text_vocab_size") {
            config.text_vocab_size = v;
        }
        if let Some(v) = u("text_model_size") {
            config.text_model_size = v;
        }
        if let Some(v) = u("text_num_layers") {
            config.text_num_layers = v;
        }
        if let Some(v) = u("text_num_heads") {
            config.text_num_heads = v;
        }
        if let Some(v) = u("text_intermediate_size") {
            config.text_intermediate_size = v;
        }
        if let Some(v) = u("speaker_patch_size") {
            config.speaker_patch_size = v;
        }
        if let Some(v) = u("speaker_model_size") {
            config.speaker_model_size = v;
        }
        if let Some(v) = u("speaker_num_layers") {
            config.speaker_num_layers = v;
        }
        if let Some(v) = u("speaker_num_heads") {
            config.speaker_num_heads = v;
        }
        if let Some(v) = u("speaker_intermediate_size") {
            config.speaker_intermediate_size = v;
        }
        if let Some(v) = u("timestep_embed_size") {
            config.timestep_embed_size = v;
        }
        if let Some(v) = u("adaln_rank") {
            config.adaln_rank = v;
        }
        config
    }

    fn valid(&self) -> Result<()> {
        if self.model_size % self.num_heads != 0 {
            return Err(SpeechError::BadConfig {
                field: "dit.model_size".into(),
                why: format!(
                    "{} not divisible by {} heads",
                    self.model_size, self.num_heads
                ),
            });
        }
        if self.text_model_size % self.text_num_heads != 0 {
            return Err(SpeechError::BadConfig {
                field: "dit.text_model_size".into(),
                why: format!(
                    "{} not divisible by {} text heads",
                    self.text_model_size, self.text_num_heads
                ),
            });
        }
        if self.speaker_model_size % self.speaker_num_heads != 0 {
            return Err(SpeechError::BadConfig {
                field: "dit.speaker_model_size".into(),
                why: format!(
                    "{} not divisible by {} speaker heads",
                    self.speaker_model_size, self.speaker_num_heads
                ),
            });
        }
        if self.timestep_embed_size % 2 != 0 {
            return Err(SpeechError::BadConfig {
                field: "dit.timestep_embed_size".into(),
                why: "embed_size must be even".into(),
            });
        }
        Ok(())
    }
}

/// `SamplerConfig`: the euler sampler controls with the reference
/// defaults.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub num_steps: usize,
    pub cfg_scale_text: f32,
    pub cfg_scale_speaker: f32,
    pub cfg_min_t: f32,
    pub cfg_max_t: f32,
    pub truncation_factor: Option<f32>,
    pub rescale_k: Option<f32>,
    pub rescale_sigma: Option<f32>,
    pub speaker_kv_scale: Option<f32>,
    pub speaker_kv_max_layers: Option<usize>,
    pub speaker_kv_min_t: Option<f32>,
    pub sequence_length: usize,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            num_steps: 40,
            cfg_scale_text: 3.0,
            cfg_scale_speaker: 8.0,
            cfg_min_t: 0.5,
            cfg_max_t: 1.0,
            truncation_factor: None,
            rescale_k: None,
            rescale_sigma: None,
            speaker_kv_scale: None,
            speaker_kv_max_layers: None,
            speaker_kv_min_t: None,
            sequence_length: 640,
        }
    }
}

impl SamplerConfig {
    fn from_dict(value: &Value) -> Self {
        let mut config = Self::default();
        let u = |key: &str| value.get(key).and_then(Value::as_u64).map(|n| n as usize);
        let f = |key: &str| value.get(key).and_then(Value::as_f64).map(|n| n as f32);
        let opt_f = |key: &str| match value.get(key) {
            None | Some(Value::Null) => None,
            Some(v) => v.as_f64().map(|n| n as f32),
        };
        let opt_u = |key: &str| match value.get(key) {
            None | Some(Value::Null) => None,
            Some(v) => v.as_u64().map(|n| n as usize),
        };
        if let Some(v) = u("num_steps") {
            config.num_steps = v;
        }
        if let Some(v) = f("cfg_scale_text") {
            config.cfg_scale_text = v;
        }
        if let Some(v) = f("cfg_scale_speaker") {
            config.cfg_scale_speaker = v;
        }
        if let Some(v) = f("cfg_min_t") {
            config.cfg_min_t = v;
        }
        if let Some(v) = f("cfg_max_t") {
            config.cfg_max_t = v;
        }
        config.truncation_factor = opt_f("truncation_factor");
        config.rescale_k = opt_f("rescale_k");
        config.rescale_sigma = opt_f("rescale_sigma");
        config.speaker_kv_scale = opt_f("speaker_kv_scale");
        config.speaker_kv_max_layers = opt_u("speaker_kv_max_layers");
        config.speaker_kv_min_t = opt_f("speaker_kv_min_t");
        if let Some(v) = u("sequence_length") {
            config.sequence_length = v;
        }
        config
    }
}

/// `ModelConfig`.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type: String,
    pub sample_rate: u32,
    pub max_text_length: usize,
    pub max_speaker_latent_length: usize,
    pub audio_downsample_factor: usize,
    pub normalize_text: bool,
    pub delete_blockwise_modules: bool,
    pub pca_filename: String,
    pub fish_codec_repo: String,
    pub model_path: Option<String>,
    pub dit: EchoDiTConfig,
    pub sampler: SamplerConfig,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            model_type: "echo_tts".into(),
            sample_rate: 44100,
            max_text_length: 768,
            max_speaker_latent_length: 6400,
            audio_downsample_factor: 2048,
            normalize_text: true,
            delete_blockwise_modules: false,
            pca_filename: "pca_state.safetensors".into(),
            fish_codec_repo: "jordand/fish-s1-dac-min".into(),
            model_path: None,
            dit: EchoDiTConfig::default(),
            sampler: SamplerConfig::default(),
        }
    }
}

impl ModelConfig {
    /// The reference `from_dict`.
    pub fn from_json(value: &Value) -> Result<Self> {
        let mut config = Self::default();
        if let Some(v) = value.get("model_type").and_then(Value::as_str) {
            config.model_type = v.to_owned();
        }
        if let Some(v) = value.get("sample_rate").and_then(Value::as_u64) {
            config.sample_rate = v as u32;
        }
        let u = |key: &str| value.get(key).and_then(Value::as_u64).map(|n| n as usize);
        if let Some(v) = u("max_text_length") {
            config.max_text_length = v;
        }
        if let Some(v) = u("max_speaker_latent_length") {
            config.max_speaker_latent_length = v;
        }
        if let Some(v) = u("audio_downsample_factor") {
            config.audio_downsample_factor = v;
        }
        if let Some(v) = value.get("normalize_text").and_then(Value::as_bool) {
            config.normalize_text = v;
        }
        if let Some(v) = value
            .get("delete_blockwise_modules")
            .and_then(Value::as_bool)
        {
            config.delete_blockwise_modules = v;
        }
        if let Some(v) = value.get("pca_filename").and_then(Value::as_str) {
            config.pca_filename = v.to_owned();
        }
        if let Some(v) = value.get("fish_codec_repo").and_then(Value::as_str) {
            config.fish_codec_repo = v.to_owned();
        }
        if let Some(v) = value.get("model_path").and_then(Value::as_str) {
            config.model_path = Some(v.to_owned());
        }
        config.dit = value
            .get("dit")
            .map(EchoDiTConfig::from_dict)
            .unwrap_or_default();
        config.dit.enable_blockwise_modules = !config.delete_blockwise_modules;
        config.sampler = value
            .get("sampler")
            .map(SamplerConfig::from_dict)
            .unwrap_or_default();
        // Zero divides in audio::get_speaker_latent_and_mask
        // (`audio.len() / frame_factor`); refuse it at load instead.
        if config.audio_downsample_factor == 0 {
            return Err(SpeechError::BadConfig {
                field: "audio_downsample_factor".into(),
                why: "must be positive".into(),
            });
        }
        config.dit.valid()?;
        Ok(config)
    }
}
