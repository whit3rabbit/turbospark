//! rumik-oss 1 TTS family (multilingual Indic).
//!
//! Port of `mlx_audio/tts/models/rumik_oss/rumik_oss.py` at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/rumik_oss):
//! a cohere2 backbone (shared [`crate::lm`] port) extended with a stop
//! predictor head and flattened Mimi codec tokens. For every 80 ms frame
//! the model emits `num_quantizers` codebook tokens in codebook order
//! (code-major, quantizer-minor unit-token layout), the round-robin
//! assembler regroups them into frames, and the frozen Mimi codec turns
//! the frames back into 24 kHz audio. See [README.md](README.md) for the
//! verification evidence and open items.
//!
//! Scope of this port, matching the reference single-segment path: batch
//! size 1, non-streaming decode (the frames decode to one waveform after
//! generation; the reference streaming decoder path is out of scope), and
//! prompt construction as a string. Text tokenization is a caller concern
//! (the real checkpoints ship a stock CohereTokenizer the portable crate
//! does not embed; the tokenizer itself prepends BOS), so
//! [`Model::generate_from_prompt`] takes the tokenized prompt ids. The
//! reference audio-head slice (`audio_head` / `make_sliced_sampler`) is a
//! memory optimization with bit-identical logits over the legal token set;
//! this port computes the full tied head and restricts through the audio
//! logits processor instead, which is the same math over the whole
//! vocabulary.

use serde_json::Value;

use crate::codec::mimi::Mimi;
use crate::lm::{self, cohere2::Cohere2Model};
use crate::music::minimax_music3::rng;
use crate::nn::{LayerNorm, Linear};
use crate::ops;
use crate::tts::sampling;
use crate::{quant, Result, SpeechError};

/// Reference Mimi checkpoint the codec loads from (kyutai layout; the
/// codec bundled with rumik-oss-1 is byte-identical to kyutai/mimi).
pub const MIMI_REPO: &str = "kyutai/moshiko-pytorch-bf16";
/// Reference sample rate (the deployed Mimi codec).
pub const DEFAULT_SAMPLE_RATE: u32 = 24000;
/// Reference frame rate: 80 ms per codec frame.
pub const DEFAULT_FRAME_RATE_HZ: f32 = 12.5;
/// Reference codebooks per frame.
pub const DEFAULT_NUM_QUANTIZERS: usize = 8;
/// Reference entries per codebook.
pub const DEFAULT_CODEBOOK_SIZE: usize = 2048;
/// Reference speaker roster (`ModelConfig.speakers` default).
pub const DEFAULT_SPEAKERS: [&str; 4] = ["Ira", "Aisha", "Siya", "Zoya"];

/// The TTS `ModelConfig`: the cohere2 backbone args plus the audio
/// vocabulary and codec fields the reference dataclass adds.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub backbone: lm::cohere2::Cohere2Config,
    pub num_quantizers: usize,
    pub codebook_size: usize,
    pub audio_start_token_id: Option<i32>,
    pub audio_end_token_id: Option<i32>,
    pub text_start_token_id: Option<i32>,
    pub first_unit_id: Option<i32>,
    pub last_unit_id: Option<i32>,
    pub frame_rate_hz: f32,
    pub speakers: Vec<String>,
    pub sample_rate: u32,
    pub mimi_repo: Option<String>,
}

impl ModelConfig {
    pub fn from_json(value: &Value) -> Result<Self> {
        let backbone = lm::cohere2::from_json(value)?;
        let list = |key: &str| -> Vec<String> {
            value
                .get(key)
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default()
        };
        let int_field = |key: &str| -> Option<i32> {
            value
                .get(key)
                .and_then(Value::as_i64)
                .and_then(|v| i32::try_from(v).ok())
        };
        Ok(Self {
            backbone,
            num_quantizers: value
                .get("num_quantizers")
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(DEFAULT_NUM_QUANTIZERS),
            codebook_size: value
                .get("codebook_size")
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(DEFAULT_CODEBOOK_SIZE),
            audio_start_token_id: int_field("audio_start_token_id"),
            audio_end_token_id: int_field("audio_end_token_id"),
            text_start_token_id: int_field("text_start_token_id"),
            first_unit_id: int_field("first_unit_id"),
            last_unit_id: int_field("last_unit_id"),
            frame_rate_hz: value
                .get("frame_rate_hz")
                .and_then(Value::as_f64)
                .map(|v| v as f32)
                .unwrap_or(DEFAULT_FRAME_RATE_HZ),
            speakers: {
                let names = list("speakers");
                if names.is_empty() {
                    DEFAULT_SPEAKERS.iter().map(|s| s.to_string()).collect()
                } else {
                    names
                }
            },
            sample_rate: value
                .get("sample_rate")
                .and_then(Value::as_u64)
                .map(|n| n as u32)
                .unwrap_or(DEFAULT_SAMPLE_RATE),
            mimi_repo: value
                .get("mimi_repo")
                .and_then(Value::as_str)
                .map(str::to_owned),
        })
    }
}

/// The resolved audio vocabulary the round-robin assembler and the logits
/// processor read: unit ids `[first, last]` plus the `</audio>` stop id.
/// `last` mirrors the reference `__post_init__` derivation when the
/// checkpoint omits it.
#[derive(Debug, Clone, Copy)]
pub struct AudioVocab {
    pub first_unit_id: i32,
    pub last_unit_id: i32,
    pub audio_end_token_id: i32,
    pub num_quantizers: usize,
}

impl AudioVocab {
    /// Resolves the ids from the config, refusing to guess when the
    /// checkpoint omits them (the reference would `int(None)`).
    pub fn from_config(config: &ModelConfig) -> Result<Self> {
        if config.num_quantizers == 0 || config.codebook_size == 0 {
            return Err(SpeechError::BadConfig {
                field: "num_quantizers".into(),
                why: "num_quantizers and codebook_size must be positive".into(),
            });
        }
        let first = config.first_unit_id.ok_or_else(|| SpeechError::BadConfig {
            field: "first_unit_id".into(),
            why: "the audio unit range is required for generation".into(),
        })?;
        let last = match config.last_unit_id {
            Some(last) => last,
            None => first + (config.codebook_size * config.num_quantizers) as i32 - 1,
        };
        if first < 0 || last < first {
            return Err(SpeechError::BadConfig {
                field: "first_unit_id".into(),
                why: format!("unit range [{first}, {last}] is empty or negative"),
            });
        }
        let end = config
            .audio_end_token_id
            .ok_or_else(|| SpeechError::BadConfig {
                field: "audio_end_token_id".into(),
                why: "the </audio> stop id is required for generation".into(),
            })?;
        if end < 0 {
            return Err(SpeechError::BadConfig {
                field: "audio_end_token_id".into(),
                why: format!("negative audio end id {end}"),
            });
        }
        Ok(Self {
            first_unit_id: first,
            last_unit_id: last,
            audio_end_token_id: end,
            num_quantizers: config.num_quantizers,
        })
    }

    pub fn is_unit_token(&self, token_id: i32) -> bool {
        self.first_unit_id <= token_id && token_id <= self.last_unit_id
    }

    /// Reference `push_audio_token`, the incremental round-robin frame
    /// assembler: `code = (id - first) / quantizers`,
    /// `quantizer = (id - first) % quantizers`. A token that breaks the
    /// codebook round robin drops the partial frame and resyncs (one bad
    /// token costs one frame, not the rest of the clip), and a completed
    /// frame is returned and the accumulator reset.
    pub fn push_audio_token(&self, token_id: i32, frame: &mut Vec<i32>) -> Option<Vec<i32>> {
        if !self.is_unit_token(token_id) {
            frame.clear();
            return None;
        }
        let offset = token_id - self.first_unit_id;
        let code = offset / self.num_quantizers as i32;
        let quantizer = offset % self.num_quantizers as i32;
        if quantizer as usize == frame.len() {
            frame.push(code);
            if frame.len() == self.num_quantizers {
                return Some(std::mem::take(frame));
            }
            return None;
        }
        // Out of order: resync to a fresh frame on quantizer 0, drop
        // otherwise (the reference returns `[code] if q == 0 else []`).
        frame.clear();
        if quantizer == 0 {
            frame.push(code);
        }
        None
    }
}

/// Why the generate loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    AudioEnd,
    MaxTokens,
}

/// One generated segment: sampled ids, the complete codec frames, the
/// level-major codes, and the decoded mono waveform.
#[derive(Debug, Clone)]
pub struct Generated {
    pub tokens: Vec<i32>,
    pub frames: Vec<Vec<i32>>,
    pub codes: Vec<Vec<i32>>,
    pub samples: Vec<f32>,
    pub stopped_by: StopReason,
}

/// Sampling and generation options for one segment, with the reference
/// `generate` defaults.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    /// 0.0 selects the exact greedy path (`make_sampler(0, ...)`).
    pub temperature: f32,
    /// Disabled at 1.0 (the reference default) like `make_sampler`.
    pub top_p: f32,
    /// `0` (or negative) disables, matching the reference default of 30.
    pub top_k: i32,
    pub max_tokens: usize,
    /// Tokens generated before `</audio>` may be emitted.
    pub min_tokens: usize,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 1.0,
            top_k: 30,
            max_tokens: 2048,
            min_tokens: 8,
        }
    }
}

/// The stop predictor head: `LayerNorm -> Linear -> GELU -> Linear`
/// over the post-final-norm hidden state (the same tensor the logits are
/// computed from), read as `sigmoid(output) > 0.5`.
struct StopPredictor {
    norm: LayerNorm,
    proj: Linear,
    out: Linear,
    hidden: usize,
}

impl StopPredictor {
    /// Loads the head. PyTorch `nn.Sequential` checkpoints number the
    /// children directly (`stop_predictor.0.weight`, the GELU taking
    /// index 2) while MLX-side saves nest them under `layers.`
    /// (`stop_predictor.layers.0.weight`, what the reference sanitize
    /// rewrites to); both spellings load.
    fn load(weights: &lm::Weights, hidden: usize) -> Result<Self> {
        let mid = (hidden / 4).max(64);
        let bases = |index: usize| -> [String; 2] {
            [
                format!("stop_predictor.layers.{index}"),
                format!("stop_predictor.{index}"),
            ]
        };
        let absolute_linear = |candidates: &[String; 2],
                               input: usize,
                               output: usize,
                               bias: lm::Bias|
         -> Result<Linear> {
            let mut last = None;
            for candidate in candidates {
                match weights.linear_absolute(candidate, input, output, bias) {
                    Ok(linear) => return Ok(linear),
                    Err(error) => last = Some(error),
                }
            }
            Err(last.unwrap_or_else(|| SpeechError::Tensor {
                name: "stop_predictor".into(),
                why: "no candidate name".into(),
            }))
        };
        // The LayerNorm rows are 1-D tensors, so they load through the
        // absolute tensor path; the linears keep the absolute linear path
        // with the mid width max(64, hidden / 4).
        let mut norm_weight = None;
        let mut norm_bias = None;
        for base in bases(0) {
            if weights.contains_absolute(&base) {
                norm_weight = Some(weights.tensor_absolute(&format!("{base}.weight"), &[hidden])?);
                if weights.contains_absolute(&format!("{base}.bias")) {
                    norm_bias = Some(weights.tensor_absolute(&format!("{base}.bias"), &[hidden])?);
                }
                break;
            }
        }
        let norm = LayerNorm::new(
            norm_weight.ok_or_else(|| SpeechError::Tensor {
                name: "stop_predictor.layers.0.weight".into(),
                why: "tensor is missing".into(),
            })?,
            norm_bias,
            1e-5,
        );
        let proj = absolute_linear(&bases(1), hidden, mid, lm::Bias::Required)?;
        let out = absolute_linear(&bases(3), mid, 1, lm::Bias::Required)?;
        Ok(Self {
            norm,
            proj,
            out,
            hidden,
        })
    }

    /// `P(utterance is over)` for one hidden row.
    fn probability(&self, hidden_row: &[f32]) -> Result<f32> {
        if hidden_row.len() != self.hidden {
            return Err(SpeechError::Input {
                why: format!(
                    "stop predictor hidden width {} does not match {}",
                    hidden_row.len(),
                    self.hidden
                ),
            });
        }
        let mut row = hidden_row.to_vec();
        self.norm.apply(&mut row, 1);
        let mid = self.proj.forward(&row, 1);
        let mut activated = mid;
        ops::gelu_erf(&mut activated);
        let out = self.out.forward(&activated, 1);
        Ok(ops::sigmoid(out[0]))
    }
}

/// The loaded family model: cohere2 backbone, stop predictor, and the
/// resolved audio vocabulary.
pub struct Model {
    pub config: ModelConfig,
    pub audio: AudioVocab,
    backbone: Cohere2Model,
    stop_predictor: StopPredictor,
}

impl Model {
    /// Opens a checkpoint directory (`config.json` + safetensors).
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        let weights = lm::Weights::open(dir)?;
        Self::load(&weights, &config_value)
    }

    /// Loads from opened weights and the raw config dict.
    pub fn load(weights: &lm::Weights, config: &Value) -> Result<Self> {
        let config = ModelConfig::from_json(config)?;
        let audio = AudioVocab::from_config(&config)?;
        let backbone = Cohere2Model::from_config(weights, config.backbone.clone())?;
        let stop_predictor = StopPredictor::load(weights, config.backbone.hidden_size)?;
        Ok(Self {
            config,
            audio,
            backbone,
            stop_predictor,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    pub fn frame_rate_hz(&self) -> f32 {
        self.config.frame_rate_hz
    }

    /// Reference `stop_probability` for one post-norm hidden row (the
    /// same tensor the logits are computed from).
    pub fn stop_probability(&self, hidden_row: &[f32]) -> Result<f32> {
        self.stop_predictor.probability(hidden_row)
    }

    /// Reference `resolve_voice`: the first speaker by default, a
    /// case-insensitive name match otherwise, an error listing the roster
    /// when nothing matches.
    pub fn resolve_voice(&self, voice: Option<&str>) -> Result<String> {
        let speakers = &self.config.speakers;
        match voice {
            None => Ok(speakers[0].clone()),
            Some(voice) => {
                let needle = voice.trim().to_ascii_lowercase();
                speakers
                    .iter()
                    .find(|name| name.to_ascii_lowercase() == needle)
                    .cloned()
                    .ok_or_else(|| SpeechError::Input {
                        why: format!(
                            "Unknown voice {voice:?}. Available: {}",
                            speakers.join(", ")
                        ),
                    })
            }
        }
    }

    /// Reference `build_prompt`:
    /// `<text>{speaker}: {text}<audio>` with an optional
    /// `<description="..."> ` prefix inserted before the text (only when
    /// the text does not embed one already).
    pub fn build_prompt(
        &self,
        text: &str,
        voice: Option<&str>,
        description: Option<&str>,
    ) -> Result<String> {
        let speaker = self.resolve_voice(voice)?;
        let text = text.trim();
        let text = match description {
            Some(description) if !text.contains("<description=") => {
                format!("<description=\"{}\"> {}", description.trim(), text)
            }
            _ => text.to_string(),
        };
        Ok(format!("<text>{speaker}: {text}<audio>"))
    }

    /// The generate loop alone over tokenized prompt ids: prefill, greedy
    /// or sampled steps with the audio logits processor (min-tokens gate,
    /// stop-predictor forcing, unit-token restriction), the round-robin
    /// frame assembly, and the stop conditions. Callers that stream decode
    /// intervals (out of scope here) drive this directly.
    pub fn generate_tokens_from_prompt(
        &mut self,
        prompt_ids: &[i32],
        params: &GenerateParams,
        seed: u64,
    ) -> Result<(Vec<i32>, StopReason)> {
        if prompt_ids.is_empty() {
            return Err(SpeechError::Input {
                why: "empty prompt".into(),
            });
        }
        // The reference generate_step never yields at max_tokens 0.
        if params.max_tokens == 0 {
            return Ok((Vec::new(), StopReason::MaxTokens));
        }
        let mut caches = self.backbone.make_cache();
        // The reference generate_step prefills everything except the last
        // token unconditionally (processors never see it), then steps the
        // last token through the processors.
        let (prefill, last) = prompt_ids.split_at(prompt_ids.len() - 1);
        if !prefill.is_empty() {
            self.backbone
                .forward(lm::engine::Input::Tokens(prefill), Some(&mut caches))?;
        }
        let (mut hidden, mut logits) = self
            .backbone
            .forward_hidden_and_logits(lm::engine::Input::Tokens(last), Some(&mut caches))?;

        let hidden_size = self.config.backbone.hidden_size;
        let mut step: usize = 0;
        // Hint only; capped so a huge caller max_tokens does not allocate up front.
        let mut tokens = Vec::with_capacity(params.max_tokens.min(4096));
        let mut keys = rng::KeySequence::new(seed);

        loop {
            let last_hidden = &hidden[hidden.len() - hidden_size..];
            // The reference processor consults the stop predictor only once
            // min_tokens is reached. Evaluating it earlier lets a predictor
            // error fail steps the reference never scores.
            let stop_probability = if step < params.min_tokens {
                0.0
            } else {
                self.stop_predictor.probability(last_hidden)?
            };
            let processed = process_audio_logits(
                &logits,
                step,
                params.min_tokens,
                &self.audio,
                stop_probability,
            );
            let logprobs = sampling::log_softmax(&processed);
            let next = if params.temperature == 0.0 {
                sampling::argmax(&logprobs)
            } else {
                let mut filtered = logprobs;
                if params.top_p > 0.0 && params.top_p < 1.0 {
                    sampling::apply_top_p(&mut filtered, params.top_p);
                }
                if params.top_k > 0 {
                    sampling::apply_top_k(&mut filtered, params.top_k as usize)?;
                }
                sampling::sample_categorical(&filtered, params.temperature, keys.next())
            };
            let next = i32::try_from(next).map_err(|_| SpeechError::Input {
                why: format!("sampled token {next} does not fit i32"),
            })?;
            tokens.push(next);
            let stopped_by = if next == self.audio.audio_end_token_id {
                StopReason::AudioEnd
            } else {
                step += 1;
                if step == params.max_tokens {
                    StopReason::MaxTokens
                } else {
                    let (next_hidden, next_logits) = self.backbone.forward_hidden_and_logits(
                        lm::engine::Input::Tokens(&[next]),
                        Some(&mut caches),
                    )?;
                    hidden = next_hidden;
                    logits = next_logits;
                    continue;
                }
            };
            return Ok((tokens, stopped_by));
        }
    }

    /// Runs one segment end to end: token loop, round-robin frame
    /// assembly, and one Mimi decode of all complete frames.
    /// `seed` only feeds the categorical draws; temperature 0 never draws
    /// and is fully deterministic.
    pub fn generate_from_prompt(
        &mut self,
        mimi: &Mimi,
        prompt_ids: &[i32],
        params: &GenerateParams,
        seed: u64,
    ) -> Result<Generated> {
        let (tokens, stopped_by) = self.generate_tokens_from_prompt(prompt_ids, params, seed)?;
        let frames = self.frames_from_tokens(&tokens);
        let codes = frames_to_codes(&frames).ok_or_else(|| SpeechError::Input {
            why: format!(
                "The model produced no complete audio frame ({} tokens, need at least {} per frame)",
                tokens.len(),
                self.num_quantizers()
            ),
        })?;
        let samples = decode_frames(mimi, &codes)?;
        Ok(Generated {
            tokens,
            frames,
            codes,
            samples,
            stopped_by,
        })
    }

    /// Reference `audio_tokens_to_codes`: generated ids into the
    /// level-major `[num_quantizers, num_frames]` codec codes, stopping at
    /// `</audio>`; incomplete tails are dropped exactly like the
    /// generate loop.
    pub fn audio_tokens_to_codes(&self, token_ids: &[i32]) -> Result<Vec<Vec<i32>>> {
        let frames = self.frames_from_tokens(token_ids);
        frames_to_codes(&frames).ok_or_else(|| SpeechError::Input {
            why: format!(
                "no complete codec frame in {} tokens (need at least {})",
                token_ids.len(),
                self.num_quantizers()
            ),
        })
    }

    /// The complete frames from a generated id list: generation order,
    /// stopping at `</audio>`, partial tails dropped.
    fn frames_from_tokens(&self, token_ids: &[i32]) -> Vec<Vec<i32>> {
        let mut frames = Vec::new();
        let mut frame = Vec::new();
        for &token in token_ids {
            if token == self.audio.audio_end_token_id {
                break;
            }
            if let Some(done) = self.audio.push_audio_token(token, &mut frame) {
                frames.push(done);
            }
        }
        frames
    }

    fn num_quantizers(&self) -> usize {
        self.audio.num_quantizers
    }
}

/// Reference `make_audio_logits_processor` for one step, over the full
/// vocabulary: before `min_tokens`, `</audio>` is masked out and the stop
/// predictor is not consulted; afterwards a stop probability above 0.5
/// forces `</audio>` (every other logit to negative infinity) and
/// otherwise sampling is restricted to the unit tokens plus `</audio>`.
fn process_audio_logits(
    logits: &[f32],
    step: usize,
    min_tokens: usize,
    audio: &AudioVocab,
    stop_probability: f32,
) -> Vec<f32> {
    let processed: Vec<f32> = logits
        .iter()
        .enumerate()
        .map(|(index, &logit)| {
            let id = index as i32;
            let is_unit = audio.is_unit_token(id);
            if step < min_tokens {
                if is_unit {
                    logit
                } else {
                    f32::NEG_INFINITY
                }
            } else if stop_probability > 0.5 {
                if id == audio.audio_end_token_id {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            } else if is_unit || id == audio.audio_end_token_id {
                logit
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();
    processed
}

/// Reference `_frames_to_codes`: complete frames `[T][Q]` into the
/// level-major codes `[Q][T]`; `None` when no frame completed.
pub fn frames_to_codes(frames: &[Vec<i32>]) -> Option<Vec<Vec<i32>>> {
    let count = frames.first()?.len();
    if frames.iter().any(|frame| frame.len() != count) {
        return None;
    }
    let mut codes = vec![Vec::with_capacity(frames.len()); count];
    for frame in frames {
        for (level, code) in frame.iter().enumerate() {
            codes[level].push(*code);
        }
    }
    Some(codes)
}

/// Reference `_decode_frames`: level-major codes through the Mimi codec
/// to mono samples.
pub fn decode_frames(mimi: &Mimi, codes: &[Vec<i32>]) -> Result<Vec<f32>> {
    if codes.is_empty() {
        return Err(SpeechError::Input {
            why: "no codec codes to decode".into(),
        });
    }
    mimi.decode(codes)
}

#[cfg(test)]
mod tests;
