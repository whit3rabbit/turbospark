//! `config.json` and `preprocessor_config.json` for CLAP.
//!
//! Field defaults are the transformers 4.57.1 `ClapConfig`,
//! `ClapAudioConfig`, `ClapTextConfig`, and `ClapFeatureExtractor`
//! defaults, so a checkpoint that omits a field gets the value the
//! reference would use. Anything this port has not verified is refused at
//! load (feature fusion, non-absolute positions, a geometry that needs
//! window padding) rather than producing plausible noise.

use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use crate::error::{bad_config, EmbeddingError};
use crate::Result;

/// Top-level `ClapConfig`.
#[derive(Debug, Clone)]
pub struct ClapConfig {
    pub audio: AudioConfig,
    pub text: TextConfig,
    /// Output width of both projection heads. The top-level value
    /// overrides the sub-configs, as in `ClapConfig.__init__`.
    pub projection_dim: usize,
}

/// `ClapAudioConfig`, the HTSAT encoder.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    pub window_size: usize,
    pub num_mel_bins: usize,
    pub spec_size: usize,
    pub hidden_act: String,
    pub patch_size: Value,
    pub patch_stride: Value,
    pub hidden_size: usize,
    pub depths: Vec<usize>,
    pub num_attention_heads: Vec<usize>,
    pub enable_fusion: bool,
    pub patch_embed_input_channels: usize,
    pub flatten_patch_embeds: bool,
    pub patch_embeds_hidden_size: usize,
    pub enable_patch_layer_norm: bool,
    pub qkv_bias: bool,
    pub mlp_ratio: f64,
    pub layer_norm_eps: f64,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            window_size: 8,
            num_mel_bins: 64,
            spec_size: 256,
            hidden_act: "gelu".into(),
            patch_size: Value::from(4),
            patch_stride: Value::from(vec![4, 4]),
            hidden_size: 768,
            depths: vec![2, 2, 6, 2],
            num_attention_heads: vec![4, 8, 16, 32],
            enable_fusion: false,
            patch_embed_input_channels: 1,
            flatten_patch_embeds: true,
            patch_embeds_hidden_size: 96,
            enable_patch_layer_norm: true,
            qkv_bias: true,
            mlp_ratio: 4.0,
            layer_norm_eps: 1e-5,
        }
    }
}

/// `ClapTextConfig`, the RoBERTa encoder.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f64,
    pub pad_token_id: u32,
    pub position_embedding_type: String,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50265,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".into(),
            max_position_embeddings: 514,
            type_vocab_size: 1,
            layer_norm_eps: 1e-12,
            pad_token_id: 1,
            position_embedding_type: "absolute".into(),
        }
    }
}

/// Resolved HTSAT geometry, derived once from [`AudioConfig`].
#[derive(Debug, Clone, PartialEq)]
pub struct HtsatGeometry {
    /// `spec_size // num_mel_bins`: how many time chunks fold into rows.
    pub freq_ratio: usize,
    /// Time frames the mel is stretched to before folding (1024 for the
    /// real checkpoint).
    pub spec_width: usize,
    /// Square image side after the fold (`spec_size`).
    pub image: usize,
    pub patch: usize,
    /// Patch grid side (`image / patch`).
    pub grid: usize,
    /// Per stage: (resolution, dim, heads, depth, effective window,
    /// shift for odd blocks).
    pub stages: Vec<StageGeometry>,
    /// Channel width after the last stage.
    pub num_features: usize,
    pub mlp_hidden: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StageGeometry {
    pub resolution: usize,
    pub dim: usize,
    pub heads: usize,
    pub depth: usize,
    pub window: usize,
    pub shift: usize,
}

fn square(v: &Value, field: &str) -> Result<usize> {
    let parse = |x: &Value| {
        x.as_u64()
            .map(|n| n as usize)
            .ok_or_else(|| bad_config(field, format!("expected an integer, got {x}")))
    };
    match v {
        Value::Array(items) if items.len() == 2 => {
            let (a, b) = (parse(&items[0])?, parse(&items[1])?);
            if a != b {
                return Err(bad_config(
                    field,
                    format!("non-square {a}x{b} is not supported"),
                ));
            }
            Ok(a)
        }
        Value::Array(_) => Err(bad_config(field, "expected one integer or a pair")),
        other => parse(other),
    }
}

impl AudioConfig {
    /// Validates the config against what this port implements and
    /// resolves the per-stage geometry.
    pub fn geometry(&self) -> Result<HtsatGeometry> {
        if self.enable_fusion {
            return Err(EmbeddingError::Unsupported {
                why: "audio_config.enable_fusion=true (the fused CLAP checkpoints) is not \
                      implemented; this port covers the unfused rand_trunc path"
                    .into(),
            });
        }
        if self.hidden_act != "gelu" {
            return Err(bad_config(
                "audio_config.hidden_act",
                format!("{} (only gelu)", self.hidden_act),
            ));
        }
        if self.patch_embed_input_channels != 1 {
            return Err(bad_config(
                "audio_config.patch_embed_input_channels",
                "only 1 is supported",
            ));
        }
        if !self.flatten_patch_embeds {
            return Err(bad_config(
                "audio_config.flatten_patch_embeds",
                "only true is supported",
            ));
        }
        let patch = square(&self.patch_size, "audio_config.patch_size")?;
        let stride = square(&self.patch_stride, "audio_config.patch_stride")?;
        if patch != stride || patch == 0 {
            // patch != stride means a padded overlapping conv stem; no
            // published HTSAT checkpoint uses it.
            return Err(bad_config(
                "audio_config.patch_size",
                format!("patch {patch} != stride {stride} is not supported"),
            ));
        }
        let mels = self.num_mel_bins;
        if mels == 0 || self.spec_size % mels != 0 || self.spec_size < mels {
            return Err(bad_config(
                "audio_config.spec_size",
                format!(
                    "{} must be a positive multiple of num_mel_bins {mels}",
                    self.spec_size
                ),
            ));
        }
        let freq_ratio = self.spec_size / mels;
        // reshape_mel2img stretches time only; it stretches frequency when
        // spec_size / freq_ratio > num_mel_bins, which never happens when
        // spec_size is a multiple of num_mel_bins.
        debug_assert_eq!(self.spec_size / freq_ratio, mels);
        if self.spec_size % patch != 0 {
            return Err(bad_config(
                "audio_config.spec_size",
                "not a multiple of patch_size",
            ));
        }
        let grid = self.spec_size / patch;
        let n = self.depths.len();
        if n == 0 || self.num_attention_heads.len() != n {
            return Err(bad_config(
                "audio_config.num_attention_heads",
                "must list one head count per depths entry",
            ));
        }
        if self.window_size == 0 {
            return Err(bad_config("audio_config.window_size", "must be positive"));
        }
        let mut stages = Vec::with_capacity(n);
        let mut mlp_hidden = Vec::with_capacity(n);
        for i in 0..n {
            let resolution = grid >> i;
            if resolution << i != grid || resolution == 0 {
                return Err(bad_config(
                    "audio_config.depths",
                    "patch grid is not divisible by 2^stage",
                ));
            }
            let dim = self.patch_embeds_hidden_size << i;
            let heads = self.num_attention_heads[i];
            if heads == 0 || dim % heads != 0 {
                return Err(bad_config(
                    "audio_config.num_attention_heads",
                    format!("stage {i}: dim {dim} not divisible by {heads} heads"),
                ));
            }
            // ClapAudioLayer.set_shift_and_window_size: at resolution <=
            // window the shift is zeroed and the window shrinks to the
            // resolution. A shrunk window would disagree with the
            // relative-position table the reference sizes from the
            // configured window, so only resolution == window is allowed.
            let (window, shift) = if resolution <= self.window_size {
                if resolution != self.window_size {
                    return Err(bad_config(
                        "audio_config.window_size",
                        format!(
                            "stage {i} resolution {resolution} is below window {}",
                            self.window_size
                        ),
                    ));
                }
                (resolution, 0)
            } else {
                if resolution % self.window_size != 0 {
                    // The reference pads to a window multiple; unverified
                    // here, so refused.
                    return Err(bad_config(
                        "audio_config.window_size",
                        format!("stage {i} resolution {resolution} needs window padding"),
                    ));
                }
                (self.window_size, self.window_size / 2)
            };
            stages.push(StageGeometry {
                resolution,
                dim,
                heads,
                depth: self.depths[i],
                window,
                shift,
            });
            mlp_hidden.push((self.mlp_ratio * dim as f64) as usize);
        }
        let num_features = self.patch_embeds_hidden_size << (n - 1);
        if num_features != self.hidden_size {
            return Err(bad_config(
                "audio_config.hidden_size",
                format!(
                    "{} != patch_embeds_hidden_size * 2^(stages-1) = {num_features}",
                    self.hidden_size
                ),
            ));
        }
        Ok(HtsatGeometry {
            freq_ratio,
            spec_width: self.spec_size * freq_ratio,
            image: self.spec_size,
            patch,
            grid,
            stages,
            num_features,
            mlp_hidden,
        })
    }
}

impl TextConfig {
    pub fn validate(&self) -> Result<()> {
        if self.hidden_act != "gelu" {
            return Err(bad_config(
                "text_config.hidden_act",
                format!("{} (only gelu)", self.hidden_act),
            ));
        }
        if self.position_embedding_type != "absolute" {
            return Err(bad_config(
                "text_config.position_embedding_type",
                format!("{} (only absolute)", self.position_embedding_type),
            ));
        }
        if self.num_attention_heads == 0 || self.hidden_size % self.num_attention_heads != 0 {
            return Err(bad_config(
                "text_config.num_attention_heads",
                "hidden_size must be divisible by the head count",
            ));
        }
        if self.type_vocab_size == 0 {
            return Err(bad_config(
                "text_config.type_vocab_size",
                "must be at least 1",
            ));
        }
        if self.max_position_embeddings < self.pad_token_id as usize + 2 {
            return Err(bad_config(
                "text_config.max_position_embeddings",
                "too small for the padding_idx position offset",
            ));
        }
        Ok(())
    }

    /// Longest id sequence the position table covers: positions run from
    /// `pad + 1` to `pad + len`.
    pub fn max_tokens(&self) -> usize {
        self.max_position_embeddings - 1 - self.pad_token_id as usize
    }
}

impl ClapConfig {
    pub fn from_json(v: &Value) -> Result<Self> {
        let audio: AudioConfig = match v.get("audio_config") {
            Some(a) if !a.is_null() => {
                serde_json::from_value(a.clone()).map_err(|e| bad_config("audio_config", e))?
            }
            _ => AudioConfig::default(),
        };
        let text: TextConfig = match v.get("text_config") {
            Some(t) if !t.is_null() => {
                serde_json::from_value(t.clone()).map_err(|e| bad_config("text_config", e))?
            }
            _ => TextConfig::default(),
        };
        let projection_dim = match v.get("projection_dim") {
            None => 512,
            Some(p) => p
                .as_u64()
                .ok_or_else(|| bad_config("projection_dim", "not an integer"))?
                as usize,
        };
        let act = v
            .get("projection_hidden_act")
            .and_then(Value::as_str)
            .unwrap_or("relu");
        if act != "relu" {
            return Err(bad_config(
                "projection_hidden_act",
                format!("{act} (only relu)"),
            ));
        }
        if projection_dim == 0 {
            return Err(bad_config("projection_dim", "must be positive"));
        }
        let config = Self {
            audio,
            text,
            projection_dim,
        };
        config.audio.geometry()?;
        config.text.validate()?;
        Ok(config)
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        Self::from_json(&read_json(path)?)
    }
}

pub(crate) fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path).map_err(|e| EmbeddingError::Io {
        path: path.display().to_string(),
        why: e.to_string(),
    })?;
    serde_json::from_str(&text).map_err(|e| EmbeddingError::Io {
        path: path.display().to_string(),
        why: e.to_string(),
    })
}

/// `preprocessor_config.json` (`ClapFeatureExtractor`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct FeatureConfig {
    pub feature_size: usize,
    pub sampling_rate: u32,
    pub hop_length: usize,
    pub max_length_s: usize,
    pub fft_window_size: usize,
    pub frequency_min: f64,
    pub frequency_max: f64,
    pub top_db: Option<f64>,
    pub truncation: String,
    pub padding: String,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            feature_size: 64,
            sampling_rate: 48_000,
            hop_length: 480,
            max_length_s: 10,
            fft_window_size: 1024,
            frequency_min: 0.0,
            frequency_max: 14_000.0,
            top_db: None,
            // The ClapFeatureExtractor class default is "fusion"; the
            // unfused checkpoint's preprocessor_config.json says
            // rand_trunc, which is the only mode this port implements.
            truncation: "fusion".into(),
            padding: "repeatpad".into(),
        }
    }
}

impl FeatureConfig {
    pub fn from_json(v: &Value) -> Result<Self> {
        let c: Self =
            serde_json::from_value(v.clone()).map_err(|e| bad_config("preprocessor_config", e))?;
        c.validate()?;
        Ok(c)
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        Self::from_json(&read_json(path)?)
    }

    pub fn validate(&self) -> Result<()> {
        if self.truncation != "rand_trunc" {
            return Err(EmbeddingError::Unsupported {
                why: format!(
                    "feature truncation {:?}; only rand_trunc (the unfused checkpoint) is implemented",
                    self.truncation
                ),
            });
        }
        if self.padding != "repeatpad" {
            return Err(EmbeddingError::Unsupported {
                why: format!(
                    "feature padding {:?}; only repeatpad is implemented",
                    self.padding
                ),
            });
        }
        if self.top_db.is_some() {
            return Err(EmbeddingError::Unsupported {
                why: "preprocessor top_db clamp is not implemented".into(),
            });
        }
        let n = self.fft_window_size;
        if n < 2 || !n.is_power_of_two() {
            return Err(bad_config(
                "fft_window_size",
                format!("{n} is not a power of two"),
            ));
        }
        if self.hop_length == 0 || self.feature_size == 0 || self.max_length_s == 0 {
            return Err(bad_config(
                "hop_length",
                "hop, feature_size and max_length_s must be positive",
            ));
        }
        let nyquist = f64::from(self.sampling_rate) / 2.0;
        if !(0.0 <= self.frequency_min
            && self.frequency_min < self.frequency_max
            && self.frequency_max <= nyquist)
        {
            return Err(bad_config(
                "frequency_max",
                format!(
                    "need 0 <= fmin {} < fmax {} <= nyquist {nyquist}",
                    self.frequency_min, self.frequency_max
                ),
            ));
        }
        Ok(())
    }

    /// `max_length_s * sampling_rate`.
    pub fn max_samples(&self) -> usize {
        self.max_length_s * self.sampling_rate as usize
    }

    /// Frames of a centered STFT over `max_samples`.
    pub fn frames(&self) -> usize {
        1 + self.max_samples() / self.hop_length
    }
}
