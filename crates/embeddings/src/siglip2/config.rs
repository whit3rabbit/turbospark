//! `config.json` and `preprocessor_config.json` for SigLIP2 NaFlex.
//!
//! Missing fields take the defaults of HF transformers 4.57.1
//! `Siglip2TextConfig`, `Siglip2VisionConfig`, and
//! `Siglip2ImageProcessorFast`, because the pinned checkpoint's
//! `config.json` spells out almost nothing (`vision_config` is just its
//! `model_type`). Values this port has not verified against the reference
//! are refused rather than approximated.

use serde_json::Value;

use crate::error::{bad_config, EmbeddingError};
use crate::Result;

/// The only activation the reference checkpoints use, and the only one
/// verified here (`ops::gelu_tanh`).
pub const HIDDEN_ACT: &str = "gelu_pytorch_tanh";

/// `Siglip2Processor` text defaults: `padding="max_length"`,
/// `truncation=True`, `max_length=64`.
pub const TEXT_MAX_LENGTH: usize = 64;

/// Text tower hyperparameters (`Siglip2TextConfig`).
#[derive(Debug, Clone, PartialEq)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f32,
    /// Output width of the text `head`; defaults to `hidden_size`.
    pub projection_size: usize,
}

/// Vision tower hyperparameters (`Siglip2VisionConfig`).
#[derive(Debug, Clone, PartialEq)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    /// Learned position-embedding rows; a perfect square (the grid side).
    pub num_patches: usize,
    pub patch_size: usize,
    pub layer_norm_eps: f32,
}

impl VisionConfig {
    /// Side of the learned position grid (`int(num_patches ** 0.5)`).
    pub fn position_grid(&self) -> usize {
        (self.num_patches as f64).sqrt() as usize
    }
}

/// The dual-tower model config.
#[derive(Debug, Clone, PartialEq)]
pub struct Siglip2Config {
    pub text: TextConfig,
    pub vision: VisionConfig,
}

fn get_usize(obj: &Value, section: &str, key: &str, default: usize) -> Result<usize> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_u64()
            .map(|n| n as usize)
            .ok_or_else(|| bad_config(&format!("{section}.{key}"), "not a non-negative integer")),
    }
}

fn get_f32(obj: &Value, section: &str, key: &str, default: f32) -> Result<f32> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_f64()
            .map(|n| n as f32)
            .ok_or_else(|| bad_config(&format!("{section}.{key}"), "not a number")),
    }
}

fn require_str(obj: &Value, section: &str, key: &str, want: &str, default: &str) -> Result<()> {
    let got = match obj.get(key) {
        None | Some(Value::Null) => default,
        Some(v) => v
            .as_str()
            .ok_or_else(|| bad_config(&format!("{section}.{key}"), "not a string"))?,
    };
    if got != want {
        return Err(EmbeddingError::Unsupported {
            why: format!("{section}.{key} = {got:?}; only {want:?} is implemented"),
        });
    }
    Ok(())
}

fn nonzero(section: &str, pairs: &[(&str, usize)]) -> Result<()> {
    for &(key, v) in pairs {
        if v == 0 {
            return Err(bad_config(&format!("{section}.{key}"), "must be positive"));
        }
    }
    Ok(())
}

fn heads_divide(section: &str, hidden: usize, heads: usize) -> Result<()> {
    if hidden % heads != 0 {
        return Err(bad_config(
            &format!("{section}.num_attention_heads"),
            format!("{heads} does not divide hidden_size {hidden}"),
        ));
    }
    Ok(())
}

impl TextConfig {
    fn parse(v: &Value) -> Result<Self> {
        const S: &str = "text_config";
        require_str(v, S, "hidden_act", HIDDEN_ACT, HIDDEN_ACT)?;
        let hidden_size = get_usize(v, S, "hidden_size", 768)?;
        let cfg = Self {
            vocab_size: get_usize(v, S, "vocab_size", 32000)?,
            hidden_size,
            intermediate_size: get_usize(v, S, "intermediate_size", 3072)?,
            num_hidden_layers: get_usize(v, S, "num_hidden_layers", 12)?,
            num_attention_heads: get_usize(v, S, "num_attention_heads", 12)?,
            max_position_embeddings: get_usize(v, S, "max_position_embeddings", 64)?,
            layer_norm_eps: get_f32(v, S, "layer_norm_eps", 1e-6)?,
            projection_size: get_usize(v, S, "projection_size", hidden_size)?,
        };
        nonzero(
            S,
            &[
                ("vocab_size", cfg.vocab_size),
                ("hidden_size", cfg.hidden_size),
                ("intermediate_size", cfg.intermediate_size),
                ("num_attention_heads", cfg.num_attention_heads),
                ("max_position_embeddings", cfg.max_position_embeddings),
                ("projection_size", cfg.projection_size),
            ],
        )?;
        heads_divide(S, cfg.hidden_size, cfg.num_attention_heads)?;
        Ok(cfg)
    }
}

impl VisionConfig {
    fn parse(v: &Value) -> Result<Self> {
        const S: &str = "vision_config";
        require_str(v, S, "hidden_act", HIDDEN_ACT, HIDDEN_ACT)?;
        // Siglip2VisionTransformer reads `vision_use_head` with a True
        // default; without the head there is no pooled embedding.
        if let Some(use_head) = v.get("vision_use_head") {
            if use_head.as_bool() != Some(true) {
                return Err(EmbeddingError::Unsupported {
                    why: "vision_config.vision_use_head must be true: the pooled image \
                          embedding is the attention-pooling head's output"
                        .into(),
                });
            }
        }
        let cfg = Self {
            hidden_size: get_usize(v, S, "hidden_size", 768)?,
            intermediate_size: get_usize(v, S, "intermediate_size", 3072)?,
            num_hidden_layers: get_usize(v, S, "num_hidden_layers", 12)?,
            num_attention_heads: get_usize(v, S, "num_attention_heads", 12)?,
            num_channels: get_usize(v, S, "num_channels", 3)?,
            num_patches: get_usize(v, S, "num_patches", 256)?,
            patch_size: get_usize(v, S, "patch_size", 16)?,
            layer_norm_eps: get_f32(v, S, "layer_norm_eps", 1e-6)?,
        };
        nonzero(
            S,
            &[
                ("hidden_size", cfg.hidden_size),
                ("intermediate_size", cfg.intermediate_size),
                ("num_attention_heads", cfg.num_attention_heads),
                ("num_patches", cfg.num_patches),
                ("patch_size", cfg.patch_size),
            ],
        )?;
        heads_divide(S, cfg.hidden_size, cfg.num_attention_heads)?;
        if cfg.num_channels != 3 {
            return Err(EmbeddingError::Unsupported {
                why: format!(
                    "vision_config.num_channels = {}; the image frontend is RGB only",
                    cfg.num_channels
                ),
            });
        }
        let side = cfg.position_grid();
        if side * side != cfg.num_patches {
            return Err(bad_config(
                "vision_config.num_patches",
                format!("{} is not a perfect square", cfg.num_patches),
            ));
        }
        Ok(cfg)
    }
}

impl Siglip2Config {
    /// Parses a `Siglip2Model` `config.json`.
    pub fn from_json(v: &Value) -> Result<Self> {
        require_str(v, "config", "model_type", "siglip2", "")?;
        if v.get("quantization").is_some() || v.get("quantization_config").is_some() {
            return Err(EmbeddingError::Unsupported {
                why: "quantized SigLIP2 checkpoints are not verified; use the f32 upstream".into(),
            });
        }
        let empty = Value::Object(Default::default());
        let text = TextConfig::parse(v.get("text_config").unwrap_or(&empty))?;
        let vision = VisionConfig::parse(v.get("vision_config").unwrap_or(&empty))?;
        // logits are a dot product of the two embeddings.
        if text.projection_size != vision.hidden_size {
            return Err(bad_config(
                "text_config.projection_size",
                format!(
                    "{} must equal vision_config.hidden_size {}",
                    text.projection_size, vision.hidden_size
                ),
            ));
        }
        Ok(Self { text, vision })
    }
}

/// `preprocessor_config.json` as `Siglip2ImageProcessorFast` reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessorConfig {
    pub patch_size: usize,
    pub max_num_patches: usize,
    pub rescale_factor: f64,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

/// The only image processor this port reproduces.
pub const FAST_PROCESSOR: &str = "Siglip2ImageProcessorFast";

fn triple(v: &Value, key: &str, default: f32) -> Result<[f32; 3]> {
    match v.get(key) {
        None | Some(Value::Null) => Ok([default; 3]),
        Some(Value::Array(a)) if a.len() == 3 => {
            let mut out = [0.0f32; 3];
            for (o, x) in out.iter_mut().zip(a) {
                *o = x.as_f64().ok_or_else(|| bad_config(key, "not a number"))? as f32;
            }
            Ok(out)
        }
        Some(Value::Number(n)) => Ok([n.as_f64().unwrap_or(f64::NAN) as f32; 3]),
        Some(_) => Err(bad_config(key, "expected a number or a 3-element list")),
    }
}

fn flag(v: &Value, key: &str) -> Result<()> {
    match v.get(key) {
        None | Some(Value::Null) | Some(Value::Bool(true)) => Ok(()),
        Some(_) => Err(EmbeddingError::Unsupported {
            why: format!("preprocessor {key} must be true; only the resize+rescale+normalize path is verified"),
        }),
    }
}

impl ProcessorConfig {
    pub fn from_json(v: &Value) -> Result<Self> {
        // AutoProcessor picks the Fast (torchvision) processor from this
        // field when torchvision is installed. The slow one resizes with
        // PIL and lands on different bytes (see README), so it is refused.
        match v.get("image_processor_type").and_then(Value::as_str) {
            Some(FAST_PROCESSOR) => {}
            other => {
                return Err(EmbeddingError::Unsupported {
                    why: format!(
                        "image_processor_type {other:?}; only {FAST_PROCESSOR} is reproduced"
                    ),
                })
            }
        }
        // 2 is PILImageResampling.BILINEAR, mapped to torchvision bilinear.
        let resample = get_usize(v, "preprocessor", "resample", 2)?;
        if resample != 2 {
            return Err(EmbeddingError::Unsupported {
                why: format!("preprocessor resample {resample}; only 2 (bilinear) is implemented"),
            });
        }
        flag(v, "do_resize")?;
        flag(v, "do_rescale")?;
        flag(v, "do_normalize")?;
        if v.get("do_convert_rgb").and_then(Value::as_bool) == Some(false) {
            return Err(EmbeddingError::Unsupported {
                why: "do_convert_rgb false; the frontend always decodes to RGB".into(),
            });
        }
        let cfg = Self {
            patch_size: get_usize(v, "preprocessor", "patch_size", 16)?,
            max_num_patches: get_usize(v, "preprocessor", "max_num_patches", 256)?,
            rescale_factor: match v.get("rescale_factor") {
                None | Some(Value::Null) => 1.0 / 255.0,
                Some(x) => x
                    .as_f64()
                    .ok_or_else(|| bad_config("rescale_factor", "not a number"))?,
            },
            image_mean: triple(v, "image_mean", 0.5)?,
            image_std: triple(v, "image_std", 0.5)?,
        };
        nonzero(
            "preprocessor",
            &[
                ("patch_size", cfg.patch_size),
                ("max_num_patches", cfg.max_num_patches),
            ],
        )?;
        if !(cfg.rescale_factor.is_finite() && cfg.rescale_factor > 0.0) {
            return Err(bad_config("rescale_factor", "must be positive and finite"));
        }
        if cfg.image_std.iter().any(|&s| !(s.is_finite() && s != 0.0))
            || cfg.image_mean.iter().any(|m| !m.is_finite())
        {
            return Err(bad_config(
                "image_std",
                "mean/std must be finite, std nonzero",
            ));
        }
        Ok(cfg)
    }
}
