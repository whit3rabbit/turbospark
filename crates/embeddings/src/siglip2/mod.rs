//! SigLIP2 NaFlex dual tower (`google/siglip2-base-patch16-naflex`).
//!
//! Numeric reference: HF transformers 4.57.1 `Siglip2Model` with the
//! `Siglip2Processor` that `AutoProcessor` builds for the pinned checkpoint
//! (Fast torchvision image processor + `GemmaTokenizerFast`). See
//! `README.md` in this directory for the verified stages, the measured
//! divergences, and how the fixtures are regenerated.
//!
//! - [`processing`]: NaFlex sizing, torchvision-exact uint8 antialiased
//!   bilinear resize, normalize, patchify.
//! - [`vision`]: patch embedding, per-image position-grid resize, encoder,
//!   MAP pooling head.
//! - [`text`]: Gemma ids padded to 64, unmasked encoder, last-position
//!   pooling through `head`.
//!
//! Embeddings are L2-normalized as `Siglip2Model.forward` does, and
//! [`Siglip2::logit`] is `exp(logit_scale) * dot + logit_bias` over them.
//! Text is NOT lowercased: the reference processor and zero-shot pipeline
//! pass text through unchanged (the tokenizer config's `do_lower_case` is
//! ignored by `GemmaTokenizerFast`), so callers own any casing policy.

pub mod config;
mod encoder;
pub mod processing;
pub mod text;
pub mod vision;

#[cfg(test)]
mod tests;

use std::path::Path;

use turbospark_vision_io::{decode_image_file, Rgb8Image};

use self::config::{ProcessorConfig, Siglip2Config, TEXT_MAX_LENGTH};
use self::processing::NaflexInput;
use self::text::TextTower;
use self::vision::VisionTower;
use crate::error::{input_error, EmbeddingError};
use crate::normalize::l2_normalize;
use crate::tokenize::TextTokenizer;
use crate::weights::Weights;
use crate::Result;

fn read_json(path: &Path) -> Result<serde_json::Value> {
    let text = std::fs::read_to_string(path).map_err(|e| EmbeddingError::Io {
        path: path.display().to_string(),
        why: e.to_string(),
    })?;
    serde_json::from_str(&text).map_err(|e| EmbeddingError::Io {
        path: path.display().to_string(),
        why: format!("invalid JSON: {e}"),
    })
}

/// The Gemma tokenizer with the processor's fixed-length policy.
#[derive(Debug)]
pub struct Siglip2Tokenizer {
    inner: TextTokenizer,
    pad: u32,
    eos: u32,
}

impl Siglip2Tokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = TextTokenizer::from_file(path)?;
        let id = |tok: &str| {
            inner
                .token_to_id(tok)
                .ok_or_else(|| EmbeddingError::Tokenizer {
                    why: format!("{} has no {tok} token", path.display()),
                })
        };
        let (pad, eos) = (id("<pad>")?, id("<eos>")?);
        // Truncation below assumes the post-processor appends exactly one
        // trailing EOS, which is the checkpoint's TemplateProcessing.
        let plain = inner.encode("a", false)?;
        let mut expect = plain.clone();
        expect.push(eos);
        if inner.encode("a", true)? != expect {
            return Err(EmbeddingError::Unsupported {
                why: "tokenizer post-processor is not `$A <eos>`".into(),
            });
        }
        Ok(Self { inner, pad, eos })
    }

    /// `tokenizer(text, padding="max_length", truncation=True,
    /// max_length=len)`: content truncated to `len - 1`, `<eos>`, then
    /// `<pad>` to `len`.
    pub fn encode_padded(&self, text: &str, len: usize) -> Result<Vec<u32>> {
        if len == 0 {
            return Err(input_error("max length must be positive"));
        }
        let mut ids = self.inner.encode(text, true)?;
        if ids.len() > len {
            ids = self.inner.encode(text, false)?;
            ids.truncate(len - 1);
            ids.push(self.eos);
        }
        ids.resize(len, self.pad);
        Ok(ids)
    }

    pub fn pad_id(&self) -> u32 {
        self.pad
    }

    pub fn eos_id(&self) -> u32 {
        self.eos
    }
}

/// A loaded SigLIP2 NaFlex checkpoint.
#[derive(Debug)]
pub struct Siglip2 {
    config: Siglip2Config,
    processor: ProcessorConfig,
    vision: VisionTower,
    text: TextTower,
    tokenizer: Option<Siglip2Tokenizer>,
    logit_scale: f32,
    logit_bias: f32,
}

impl Siglip2 {
    /// Loads `config.json`, `preprocessor_config.json`, `model.safetensors`
    /// and, when present, `tokenizer.json` from `dir`. Without a tokenizer
    /// only [`Siglip2::embed_ids`] serves text.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let config = Siglip2Config::from_json(&read_json(&dir.join("config.json"))?)?;
        let processor =
            ProcessorConfig::from_json(&read_json(&dir.join("preprocessor_config.json"))?)?;
        if processor.patch_size != config.vision.patch_size {
            return Err(crate::error::bad_config(
                "patch_size",
                format!(
                    "preprocessor {} != vision_config {}",
                    processor.patch_size, config.vision.patch_size
                ),
            ));
        }
        let tok_path = dir.join("tokenizer.json");
        let tokenizer = if tok_path.is_file() {
            Some(Siglip2Tokenizer::from_file(&tok_path)?)
        } else {
            None
        };
        let w = Weights::open(dir)?;
        let logit_scale = w.load("logit_scale", &[1])?[0];
        let logit_bias = w.load("logit_bias", &[1])?[0];
        let vision = VisionTower::load(&w, &config.vision)?;
        let text = TextTower::load(w, &config.text)?;
        Ok(Self {
            config,
            processor,
            vision,
            text,
            tokenizer,
            logit_scale,
            logit_bias,
        })
    }

    pub fn config(&self) -> &Siglip2Config {
        &self.config
    }

    pub fn processor_config(&self) -> &ProcessorConfig {
        &self.processor
    }

    pub fn vision(&self) -> &VisionTower {
        &self.vision
    }

    pub fn text(&self) -> &TextTower {
        &self.text
    }

    pub fn tokenizer(&self) -> Option<&Siglip2Tokenizer> {
        self.tokenizer.as_ref()
    }

    /// `(logit_scale, logit_bias)` as stored (scale not yet exponentiated).
    pub fn logit_params(&self) -> (f32, f32) {
        (self.logit_scale, self.logit_bias)
    }

    /// The processor output for one RGB8 image.
    pub fn preprocess(&self, image: &Rgb8Image) -> Result<NaflexInput> {
        processing::preprocess(&self.processor, image)
    }

    /// L2-normalized embedding of preprocessed patches.
    pub fn embed_pixels(&self, input: &NaflexInput) -> Result<Vec<f32>> {
        let mut e = self.vision.forward(input)?;
        l2_normalize(&mut e);
        Ok(e)
    }

    /// L2-normalized image embedding (`image_embeds`).
    pub fn embed_image(&self, image: &Rgb8Image) -> Result<Vec<f32>> {
        self.embed_pixels(&self.preprocess(image)?)
    }

    /// Decodes an image file to RGB8 and embeds it. JPEG decode is not
    /// bit-identical to Pillow's; PNG is.
    pub fn embed_image_file(&self, path: &Path) -> Result<Vec<f32>> {
        let image = decode_image_file(path).map_err(|e| EmbeddingError::Io {
            path: path.display().to_string(),
            why: e.to_string(),
        })?;
        self.embed_image(&image)
    }

    /// The 64 ids the reference processor feeds the text tower.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let tok = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| EmbeddingError::Tokenizer {
                why: "checkpoint directory has no tokenizer.json".into(),
            })?;
        if TEXT_MAX_LENGTH > self.config.text.max_position_embeddings {
            return Err(input_error(format!(
                "processor pads to {TEXT_MAX_LENGTH} tokens but the tower holds {}",
                self.config.text.max_position_embeddings
            )));
        }
        tok.encode_padded(text, TEXT_MAX_LENGTH)
    }

    /// L2-normalized embedding of explicit ids (already padded as the
    /// caller wants; the last position is pooled).
    pub fn embed_ids(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let mut e = self.text.forward(ids)?;
        l2_normalize(&mut e);
        Ok(e)
    }

    /// L2-normalized text embedding (`text_embeds`).
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_ids(&self.tokenize(text)?)
    }

    /// `exp(logit_scale) * <image, text> + logit_bias` for normalized
    /// embeddings (one entry of `logits_per_image`).
    pub fn logit(&self, image_embed: &[f32], text_embed: &[f32]) -> Result<f32> {
        if image_embed.len() != text_embed.len() {
            return Err(input_error(format!(
                "embedding widths differ: {} vs {}",
                image_embed.len(),
                text_embed.len()
            )));
        }
        let dot: f32 = image_embed.iter().zip(text_embed).map(|(a, b)| a * b).sum();
        Ok(dot * self.logit_scale.exp() + self.logit_bias)
    }

    /// `logits_per_image`, row-major `[images.len(), texts.len()]`.
    pub fn logits(&self, images: &[Vec<f32>], texts: &[Vec<f32>]) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(images.len() * texts.len());
        for i in images {
            for t in texts {
                out.push(self.logit(i, t)?);
            }
        }
        Ok(out)
    }

    /// Sigmoid of each logit: SigLIP scores each pair independently, so
    /// these do not sum to one across texts.
    pub fn probabilities(&self, images: &[Vec<f32>], texts: &[Vec<f32>]) -> Result<Vec<f32>> {
        Ok(self
            .logits(images, texts)?
            .into_iter()
            .map(sigmoid)
            .collect())
    }
}

/// `1 / (1 + exp(-x))`.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
