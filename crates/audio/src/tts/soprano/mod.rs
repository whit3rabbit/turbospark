//! Soprano TTS family (qwen3 backbone straight into a Vocos iSTFT
//! decoder).
//!
//! Port of `mlx_audio/tts/models/soprano/` (soprano.py, decoder.py,
//! text.py) at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/soprano):
//! a dense qwen3 backbone whose per-step post-norm hidden states (not
//! audio tokens) feed a ConvNeXt + iSTFT decoder, the sentence
//! preprocessing and `[STOP][TEXT]...[START]` prompt frame, the
//! hidden-state stack, and the per-sentence generate loop. See
//! [README.md](README.md) for the verification evidence and open items.
//!
//! Scope of this port, matching the reference single-segment path:
//! batch size 1, non-streaming decode (the whole sentence decodes to one
//! waveform after generation; the reference `stream_generate` yields the
//! same hidden stack incrementally), and prompt construction from
//! pre-tokenized ids. Text tokenization is a caller concern (the
//! reference loads a Hugging Face tokenizer in `post_load_hook` and
//! encodes `[STOP][TEXT]{sentence}[START]` with `add_special_tokens=
//! False`); so are the stop token ids (the reference takes the tokenizer
//! pad token, falling back to the `[STOP]` token id and the tokenizer
//! eos).

mod decoder;
mod text;

pub use decoder::{DecoderConfig, SopranoDecoder};
pub use text::clean_text;

use serde_json::Value;

use crate::lm;
use crate::lm::engine::Input;
use crate::tts::sampling;
use crate::{quant, Result, SpeechError};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// Reference sample rate.
pub const DEFAULT_SAMPLE_RATE: u32 = 32000;

/// The TTS `ModelConfig`: the qwen3 backbone args plus the decoder
/// config, sample rate, and the `model_path` hook the reference
/// dataclass carries.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub backbone: lm::DenseConfig,
    pub sample_rate: u32,
    pub decoder: DecoderConfig,
    /// The reference `model_path` field: when set and not containing
    /// `soprano-1.1` (case-insensitive), the post-init overrides the
    /// decoder geometry for the 1.0 checkpoints.
    pub model_path: Option<String>,
}

impl ModelConfig {
    pub fn from_json(value: &Value) -> Result<Self> {
        let backbone = lm::qwen3::from_json(value)?;
        let sample_rate = value
            .get("sample_rate")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_SAMPLE_RATE as u64) as u32;
        let mut decoder = match value.get("decoder_config") {
            Some(nested) if !nested.is_null() => DecoderConfig::from_json(nested)?,
            _ => DecoderConfig::default(),
        };
        let model_path = value
            .get("model_path")
            .and_then(Value::as_str)
            .map(str::to_owned);
        if let Some(path) = &model_path {
            // Reference `__post_init__`: pre-1.1 checkpoints use the
            // smaller decoder with a 3-wide input kernel.
            if !path.to_lowercase().contains("soprano-1.1") {
                decoder.decoder_dim = 512;
                decoder.decoder_intermediate_dim = 1536;
                decoder.input_kernel = 3;
            }
        }
        Ok(Self {
            backbone,
            sample_rate,
            decoder,
            model_path,
        })
    }
}

/// Sampling and generation options for one sentence, with the reference
/// `generate` defaults. The reference sampler is
/// `make_sampler(temperature, top_p)` (no min-p, top-k, or logits
/// processors) and the stop ids come from the tokenizer.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    /// 0.0 selects the exact greedy path (`make_sampler(0, ...)`).
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: usize,
    /// The tokenizer pad / `[STOP]` / eos ids the loop stops on. The
    /// stopping token is not appended and gets no hidden state (the
    /// reference breaks before the forward pass).
    pub stop_token_ids: Vec<i32>,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            temperature: 0.3,
            top_p: 0.95,
            max_tokens: 512,
            stop_token_ids: Vec::new(),
        }
    }
}

/// Why the generate loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    StopToken,
    MaxTokens,
}

/// One generated sentence: sampled ids, why the loop stopped, and the
/// decoded (and token-trimmed) mono waveform.
#[derive(Debug, Clone)]
pub struct Generated {
    pub tokens: Vec<i32>,
    pub samples: Vec<f32>,
    pub stopped_by: StopReason,
}

/// The loaded family model: qwen3 backbone, the family config, and the
/// vocoder decoder.
pub struct Model {
    pub config: ModelConfig,
    pub language_model: lm::Backbone,
    pub decoder: SopranoDecoder,
}

impl Model {
    /// Opens a checkpoint directory (`config.json` + safetensors).
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        // The backbone tensors keep the HF `model.` names (the reference
        // `sanitize` strips them at load time); the decoder tensors sit
        // at the root under `decoder.`.
        let backbone_weights = lm::Weights::open(dir)?;
        let decoder_weights = lm::Weights::open(dir)?.with_prefix("");
        Self::load_split(&backbone_weights, &decoder_weights, &config_value)
    }

    /// Loads both halves from one weights handle (same-prefix
    /// checkpoints).
    pub fn load(weights: &lm::Weights, config: &Value) -> Result<Self> {
        Self::load_split(weights, weights, config)
    }

    fn load_split(
        backbone_weights: &lm::Weights,
        decoder_weights: &lm::Weights,
        config_value: &Value,
    ) -> Result<Self> {
        let config = ModelConfig::from_json(config_value)?;
        let language_model = lm::Backbone::load(backbone_weights, config.backbone.clone())?;
        let decoder = SopranoDecoder::load(
            decoder_weights,
            &config.decoder,
            language_model.config.hidden_size,
        )?;
        Ok(Self {
            config,
            language_model,
            decoder,
        })
    }

    /// Reference `_preprocess_text` for one segment: clean the text,
    /// split sentences after terminal punctuation (the split consumes
    /// the whitespace run and the pieces keep their raw spacing, which
    /// can be empty), and merge short pieces (`min_length` 30 default)
    /// with the reference strip-on-concat semantics. Returns the prompt
    /// strings (the `[STOP][TEXT]{sentence}[START]` frame) in order;
    /// tokenization is the caller's.
    pub fn preprocess_text(text: &str, min_length: usize) -> Vec<String> {
        let cleaned = clean_text(text);
        let chars: Vec<char> = cleaned.chars().collect();
        let mut pieces: Vec<String> = Vec::new();
        let mut current = String::new();
        let mut index = 0usize;
        while index < chars.len() {
            let c = chars[index];
            current.push(c);
            // Split after [.!?] when whitespace follows; the whitespace
            // run is consumed by the split.
            if matches!(c, '.' | '!' | '?') {
                let mut end = index + 1;
                while end < chars.len() && chars[end].is_whitespace() {
                    end += 1;
                }
                if end > index + 1 {
                    pieces.push(std::mem::take(&mut current));
                    index = end;
                    continue;
                }
            }
            index += 1;
        }
        pieces.push(current);

        if min_length > 0 && pieces.len() > 1 {
            let mut merged: Vec<String> = Vec::new();
            let mut index = 0usize;
            while index < pieces.len() {
                let cur = pieces[index].clone();
                if cur.chars().count() < min_length {
                    if let Some(last) = merged.last_mut() {
                        let combined = format!("{last} {cur}");
                        *last = combined.trim().to_owned();
                    } else if index + 1 < pieces.len() {
                        let combined = format!("{} {}", cur, pieces[index + 1]);
                        pieces[index + 1] = combined.trim().to_owned();
                    } else {
                        merged.push(cur);
                    }
                } else {
                    merged.push(cur);
                }
                index += 1;
            }
            pieces = merged;
        }
        pieces
            .into_iter()
            .map(|sentence| format!("[STOP][TEXT]{sentence}[START]"))
            .collect()
    }

    /// Runs one sentence end to end: prefill capturing the last-position
    /// hidden state, greedy or sampled steps (each accepted token's
    /// forward pass contributes the next hidden row), then the decoder
    /// and the token-count trim. `seed` only feeds the categorical
    /// draws; temperature 0 never draws and is fully deterministic.
    pub fn generate_from_prompt(
        &mut self,
        prompt_ids: &[i32],
        params: &GenerateParams,
        seed: u64,
    ) -> Result<Generated> {
        let (tokens, hidden, stopped_by) = self.generate_hidden(prompt_ids, params, seed)?;
        let samples = self.decode_hidden(&hidden, tokens.len())?;
        Ok(Generated {
            tokens,
            samples,
            stopped_by,
        })
    }

    /// The generate loop alone: prefill, greedy or sampled steps with
    /// the stop conditions, returning the sampled ids and the stacked
    /// post-norm hidden states (`[1 + tokens, hidden]` row-major: the
    /// prefill row first, one row per accepted token) without decoding.
    pub fn generate_hidden(
        &mut self,
        prompt_ids: &[i32],
        params: &GenerateParams,
        seed: u64,
    ) -> Result<(Vec<i32>, Vec<f32>, StopReason)> {
        if prompt_ids.is_empty() {
            return Err(SpeechError::Input {
                why: "empty prompt".into(),
            });
        }
        let hidden_size = self.language_model.config.hidden_size;
        let mut caches = lm::make_prompt_cache(self.language_model.config.num_hidden_layers);
        // The reference prefills the whole prompt (no last-token split)
        // and yields the last hidden row before sampling anything.
        let (logits, hidden) = self
            .language_model
            .forward_hidden_and_logits(Input::Tokens(prompt_ids), Some(&mut caches))?;
        let vocab = self.language_model.config.vocab_size;
        if logits.len() != prompt_ids.len() * vocab {
            return Err(SpeechError::Tensor {
                name: "logits".into(),
                why: format!(
                    "expected {}x{} logits, got {} values",
                    prompt_ids.len(),
                    vocab,
                    logits.len()
                ),
            });
        }
        let mut hidden_stack: Vec<f32> = hidden[hidden.len() - hidden_size..].to_vec();
        let mut tokens: Vec<i32> = Vec::with_capacity(params.max_tokens);
        let mut keys = crate::music::minimax_music3::rng::KeySequence::new(seed);
        let mut step_logits = logits;

        let stopped_by = loop {
            // Sample from the last position's logits row.
            let last_row = &step_logits[step_logits.len() - vocab..];
            let logprobs = sampling::log_softmax(last_row);
            let next = if params.temperature == 0.0 {
                sampling::argmax(&logprobs)
            } else {
                let mut filtered = logprobs;
                if params.top_p > 0.0 && params.top_p < 1.0 {
                    sampling::apply_top_p(&mut filtered, params.top_p);
                }
                sampling::sample_categorical(&filtered, params.temperature, keys.next())
            };
            let next = i32::try_from(next).map_err(|_| SpeechError::Input {
                why: format!("sampled token {next} does not fit i32"),
            })?;
            // The stopping token is dropped: no push, no hidden row, no
            // forward pass (the reference breaks before stepping).
            if params.stop_token_ids.contains(&next) {
                break StopReason::StopToken;
            }
            let (next_logits, next_hidden) = self
                .language_model
                .forward_hidden_and_logits(Input::Tokens(&[next]), Some(&mut caches))?;
            hidden_stack.extend_from_slice(&next_hidden[next_hidden.len() - hidden_size..]);
            tokens.push(next);
            if tokens.len() == params.max_tokens {
                break StopReason::MaxTokens;
            }
            step_logits = next_logits;
        };
        Ok((tokens, hidden_stack, stopped_by))
    }

    /// Reference decode and trim: the hidden stack through the decoder,
    /// then keep the last `tokens * token_size` samples (the reference
    /// `audio_length = token_count * token_size - token_size` with its
    /// token count including the prefill row, so exactly one row per
    /// sampled token; a Python negative slice keeps everything when the
    /// decoder output is shorter).
    pub fn decode_hidden(&self, hidden: &[f32], tokens: usize) -> Result<Vec<f32>> {
        let positions = hidden.len() / self.language_model.config.hidden_size;
        let audio = self.decoder.forward(hidden, positions)?;
        let token_size = self.config.decoder.token_size;
        let audio_length = tokens * token_size;
        if audio_length > 0 {
            let keep = audio_length.min(audio.len());
            Ok(audio[audio.len() - keep..].to_vec())
        } else {
            Ok(audio)
        }
    }
}

#[cfg(test)]
mod tests;
