//! Portable Whistle speech recognition.
//!
//! The deployment reference is Cactus-Compute/whistle at
//! b358ddadd89b7a713b5aa131f23032d3cca1b251. This family uses the
//! Needle 3 container and Cactus Quants, independently of MLX affine weights.
//! Text contracts: needle/model/architecture.py and export.py at
//! af2654dffa4987f7e2cbbacdf59c2f854fd18302. Speech operator reference:
//! Cactus-Compute/needle3 macos-arm64 at 2ae11323dc000f5e70c49f7403efa6af12ba9e67.

mod attention;
mod cact;
mod decoder;
mod encoder;
mod frontend;
mod layers;
mod stem;
mod tokenizer;

use crate::{Result, SpeechError};
use layers::DIM;
use std::path::Path;
use turbospark_core::TokenId;

const LANGUAGES: [&str; 7] = ["en", "de", "fr", "es", "it", "nl", "pl"];

/// The released Whistle checkpoint accepts 16 kHz mono clips up to 30 seconds.
pub struct Whistle {
    frontend: frontend::Frontend,
    stem: stem::Stem,
    encoder: encoder::Encoder,
    decoder: decoder::Decoder,
    tokenizer: tokenizer::Tokenizer,
    permutations: [Vec<usize>; 2],
    language_ids: [TokenId; 7],
}

/// A decoded transcript and its language code.
#[derive(Debug, Clone)]
pub struct WhistleTranscription {
    pub text: String,
    /// Empty for silent audio without a supplied language.
    pub language: String,
    pub tokens: Vec<TokenId>,
    /// True when the token budget was exhausted before end-of-sequence.
    pub truncated: bool,
}

/// Row-major acoustic embeddings at approximately 12.5 frames per second.
#[derive(Debug, Clone)]
pub struct WhistleEmbedding {
    pub values: Vec<f32>,
    pub frames: usize,
    pub width: usize,
}

impl Whistle {
    /// Load the verified Needle 3 Whistle geometry. Other Needle models are
    /// refused instead of being dispatched by similarly named tensors.
    pub fn load(path: &Path) -> Result<Self> {
        let a = cact::Archive::read(path)?;
        let expected = [
            0, 681, 28, 0, 8, 8199, 0, 512, 8, 2, 8, 48, 64, 320, 512, 4, 0, 0, 0, 3, 18432, 128,
            4, 4, 3, 0, 2, 2, 3, 0, 0, 2, 3, 7,
        ];
        if a.header.fields[1..expected.len()] != expected[1..]
            || a.header.fields[expected.len()..].iter().any(|&v| v != 0)
            || a.header.rope_theta != 100000.0
        {
            return Err(SpeechError::BadConfig {
                field: "whistle.geometry".into(),
                why: "unsupported Whistle checkpoint geometry".into(),
            });
        }
        let manifest = layers::floats(&a, 237, &[13])?;
        if manifest
            != [
                3246.0, 1.0, 8192.0, 8.0, 4.0, 9.0, 128.0, 3.0, 80.0, 16000.0, 512.0, 400.0, 160.0,
            ]
        {
            return Err(SpeechError::BadConfig {
                field: "whistle.audio_manifest".into(),
                why: "unsupported Whistle audio architecture".into(),
            });
        }
        let tokenizer = tokenizer::Tokenizer::parse(a.raw(680)?, 8199)?;
        let language_ids = LANGUAGES
            .map(|language| tokenizer.marker(&format!("<|{language}|>")))
            .into_iter()
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .unwrap();
        Ok(Self {
            frontend: frontend::Frontend::new(layers::floats(&a, 679, &[257, 80])?)?,
            stem: stem::Stem::load(&a)?,
            encoder: encoder::Encoder::load(&a)?,
            decoder: decoder::Decoder::load(&a)?,
            tokenizer,
            language_ids,
            permutations: [layers::permutation(&a, 226)?, layers::permutation(&a, 227)?],
        })
    }

    /// Encode text with the checkpoint's BPE tokenizer, without BOS or EOS.
    pub fn tokenize(&self, text: &str) -> Vec<TokenId> {
        self.tokenizer.encode(text)
    }

    /// Decode signed token IDs with the checkpoint's byte fallback.
    pub fn decode_tokens(&self, tokens: &[TokenId]) -> Result<String> {
        self.tokenizer.decode(tokens)
    }

    /// Encode finite normalized PCM at 16 kHz. The caller owns resampling.
    pub fn encode(&self, pcm: &[f32]) -> Result<WhistleEmbedding> {
        let features = self.frontend.features(pcm)?;
        let stem = self.stem.forward(&features);
        let values = self.encoder.forward(&stem, &self.permutations);
        finite(&values)?;
        let frames = values.len() / DIM;
        Ok(WhistleEmbedding {
            values,
            frames,
            width: DIM,
        })
    }

    /// Greedy transcription with optional language conditioning.
    pub fn transcribe(
        &self,
        pcm: &[f32],
        language: Option<&str>,
        max_tokens: usize,
    ) -> Result<WhistleTranscription> {
        if max_tokens == 0 || max_tokens > 318 {
            return Err(SpeechError::Input {
                why: "Whistle max_tokens must be in 1..=318".into(),
            });
        }
        if language.is_some_and(|s| !LANGUAGES.contains(&s)) {
            return Err(SpeechError::Input {
                why: "Whistle supports en, de, fr, es, it, nl, and pl".into(),
            });
        }
        frontend::validate_pcm(pcm)?;
        // Exact digital silence carries no speech. Avoid language and text
        // hallucinations while leaving quiet speech to the model.
        if pcm.iter().all(|&v| v == 0.0) {
            return Ok(WhistleTranscription {
                text: String::new(),
                language: language.unwrap_or("").into(),
                tokens: Vec::new(),
                truncated: false,
            });
        }
        let encoded = self.encode(pcm)?;
        let mut state = self.decoder.state(&encoded.values);
        let logits = self
            .decoder
            .step(self.tokenizer.bos, &mut state, &self.permutations);
        finite(&logits)?;
        let language_index = if let Some(language) = language {
            LANGUAGES
                .iter()
                .position(|&candidate| candidate == language)
                .unwrap()
        } else {
            self.language_ids
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| logits[**a as usize].total_cmp(&logits[**b as usize]))
                .unwrap()
                .0
        };
        let detected = LANGUAGES[language_index].to_owned();
        let mut token = self.language_ids[language_index];
        let mut tokens = Vec::new();
        let mut truncated = true;
        for _ in 0..max_tokens {
            let logits = self.decoder.step(token, &mut state, &self.permutations);
            finite(&logits)?;
            token = logits[..8192]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap()
                .0 as turbospark_core::TokenId;
            if token == self.tokenizer.eos {
                truncated = false;
                break;
            }
            tokens.push(token);
        }
        let text = self.tokenizer.decode(&tokens)?;
        Ok(WhistleTranscription {
            text,
            language: detected,
            tokens,
            truncated,
        })
    }
}

fn finite(values: &[f32]) -> Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(SpeechError::Tensor {
            name: "whistle.output".into(),
            why: "non-finite model output".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
