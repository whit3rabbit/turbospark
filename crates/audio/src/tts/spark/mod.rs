//! Spark-TTS family (Spark-TTS-0.5B, qwen2 backbone + BiCodec).
//!
//! Port of `mlx_audio/tts/models/spark/` (spark.py, utils/token_parser.py)
//! at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/spark):
//! a qwen2 backbone (the shared [`crate::lm`] port) generates
//! `<|bicodec_semantic_i|>` audio tokens over a task-qualified prompt,
//! the generated ids are decoded back to code indices, and the BiCodec
//! turns semantic + global tokens into the waveform (the
//! [`crate::codec::spark_bicodec`] decode port). See [README.md](README.md)
//! for the verification evidence and open items.
//!
//! Scope of this port, matching the reference single-segment path:
//! batch size 1, non-streaming, and prompt construction at the string
//! level with caller-owned tokenization (the reference encodes the
//! prompt string with the bundled fast tokenizer, where every
//! `<|bicodec_semantic_i|>`-style marker is one token). The
//! [`AudioTokenMaps`] carry the marker-to-id lookup and the reverse
//! extraction the reference regex does. Voice-clone token extraction
//! from reference audio needs the unported BiCodec encode direction
//! (wav2vec2 features); reference prompts take caller-supplied codes.

use std::collections::HashMap;

use serde_json::Value;

use crate::codec::spark_bicodec::BiCodec;
use crate::lm;
use crate::music::minimax_music3::rng;
use crate::tts::sampling;
use crate::{quant, Result, SpeechError};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// The reference task-token map (`utils/token_parser.py`).
pub const TASK_TOKEN_MAP: &[(&str, &str)] = &[
    ("vc", "<|task_vc|>"),
    ("tts", "<|task_tts|>"),
    ("asr", "<|task_asr|>"),
    ("s2s", "<|task_s2s|>"),
    ("t2s", "<|task_t2s|>"),
    ("understand", "<|task_understand|>"),
    ("caption", "<|task_cap|>"),
    ("controllable_tts", "<|task_controllable_tts|>"),
    ("prompt_tts", "<|task_prompt_tts|>"),
    ("speech_edit", "<|task_edit|>"),
];

/// The `GENDER_MAP`.
pub const GENDER_MAP: &[(&str, i32)] = &[("female", 0), ("male", 1)];

/// The `LEVELS_MAP` (pitch and speed share it).
pub const LEVELS_MAP: &[(&str, i32)] = &[
    ("very_low", 0),
    ("low", 1),
    ("moderate", 2),
    ("high", 3),
    ("very_high", 4),
];

/// The generate-side `PITCH_MAP` / `SPEED_MAP`: the float attribute maps
/// to a level token label.
pub const PITCH_SPEED_LABELS: &[(&str, &str)] = &[
    ("0.0", "very_low"),
    ("0.5", "low"),
    ("1.0", "moderate"),
    ("1.5", "high"),
    ("2.0", "very_high"),
];

/// Number of codebook entries the semantic token map covers (the
/// published checkpoint's 8192-entry semantic codebook).
pub const SEMANTIC_TOKENS: usize = 8192;

/// Number of codebook entries the global token map covers (the
/// published speaker FSQ: 4^6).
pub const GLOBAL_TOKENS: usize = 4096;

/// `<|bicodec_semantic_{code}|>`.
pub fn semantic_token(code: i32) -> String {
    format!("<|bicodec_semantic_{code}|>")
}

/// `<|bicodec_global_{code}|>`.
pub fn global_token(code: i32) -> String {
    format!("<|bicodec_global_{code}|>")
}

/// The `<|bicodec_semantic_i|>` / `<|bicodec_global_i|>` vocabulary
/// lookup: the reference builds these strings, encodes them with the
/// bundled tokenizer, and regex-extracts the codes from the decoded
/// output. Ids arrive through a tokenizer-style lookup here, and the
/// reverse maps do the extraction.
#[derive(Debug, Clone, Default)]
pub struct AudioTokenMaps {
    semantic_by_id: HashMap<i32, i32>,
    global_by_id: HashMap<i32, i32>,
}

impl AudioTokenMaps {
    /// Builds both maps from a tokenizer-style lookup: `lookup(token)`
    /// returns the id the tokenizer assigns, or `None` when absent.
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<i32>) -> Self {
        let mut maps = Self::default();
        for index in 0..SEMANTIC_TOKENS {
            if let Some(id) = lookup(&semantic_token(index as i32)) {
                maps.semantic_by_id.insert(id, index as i32);
            }
        }
        for index in 0..GLOBAL_TOKENS {
            if let Some(id) = lookup(&global_token(index as i32)) {
                maps.global_by_id.insert(id, index as i32);
            }
        }
        maps
    }

    /// Builds both maps from explicit id tables (fixture and offline
    /// construction; `ids[i]` is the token id of code `i`).
    pub fn from_ids(
        semantic: &[Option<i32>],
        global: &[Option<i32>],
        semantic_codes: usize,
        global_codes: usize,
    ) -> Result<Self> {
        if semantic.len() != semantic_codes || global.len() != global_codes {
            return Err(SpeechError::Input {
                why: format!(
                    "audio token maps need {semantic_codes} semantic and \
                     {global_codes} global entries, got {} and {}",
                    semantic.len(),
                    global.len()
                ),
            });
        }
        let mut maps = Self::default();
        for (index, id) in semantic.iter().enumerate() {
            if let Some(id) = id {
                maps.semantic_by_id.insert(*id, index as i32);
            }
        }
        for (index, id) in global.iter().enumerate() {
            if let Some(id) = id {
                maps.global_by_id.insert(*id, index as i32);
            }
        }
        Ok(maps)
    }

    /// The semantic codes in generation order (the reference
    /// `re.findall(r"bicodec_semantic_(\d+)")`).
    pub fn extract_semantic(&self, tokens: &[i32]) -> Vec<i32> {
        tokens
            .iter()
            .filter_map(|id| self.semantic_by_id.get(id).copied())
            .collect()
    }

    /// The global codes in generation order (the reference
    /// `re.findall(r"bicodec_global_(\d+)")`).
    pub fn extract_global(&self, tokens: &[i32]) -> Vec<i32> {
        tokens
            .iter()
            .filter_map(|id| self.global_by_id.get(id).copied())
            .collect()
    }
}

/// The wrapper `ModelConfig` (spark.py dataclass defaults).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// The qwen2 backbone configuration (the dataclass fields feed the
    /// shared qwen2 port's defaults).
    pub backbone: lm::DenseConfig,
    pub sample_rate: u32,
    /// `eos_token_id`: the id `stream_generate` stops on (the yielded
    /// stream excludes it, exactly like the reference loop).
    pub eos_token_id: i32,
    /// `bos_token_id`, carried for loader parity.
    pub bos_token_id: i32,
    /// The backbone vocabulary (the head width when untied).
    pub vocab_size: usize,
}

impl ModelConfig {
    /// `ModelConfig` over a checkpoint `config.json` dict.
    pub fn from_json(value: &Value) -> Result<Self> {
        // Narrow with try_from: a bare `as` cast wraps an out-of-range id or
        // rate silently, and the wrapped value then drives generation.
        let out_of_range = |field: &str| SpeechError::BadConfig {
            field: field.into(),
            why: "value does not fit the field's integer type".into(),
        };
        let sample_rate = u32::try_from(
            value
                .get("sample_rate")
                .and_then(Value::as_u64)
                .unwrap_or(16000),
        )
        .map_err(|_| out_of_range("sample_rate"))?;
        let eos_token_id = i32::try_from(
            value
                .get("eos_token_id")
                .and_then(Value::as_i64)
                .unwrap_or(151645),
        )
        .map_err(|_| out_of_range("eos_token_id"))?;
        let bos_token_id = i32::try_from(
            value
                .get("bos_token_id")
                .and_then(Value::as_i64)
                .unwrap_or(151643),
        )
        .map_err(|_| out_of_range("bos_token_id"))?;
        let vocab_size = usize::try_from(
            value
                .get("vocab_size")
                .and_then(Value::as_u64)
                .unwrap_or(166000),
        )
        .map_err(|_| out_of_range("vocab_size"))?;
        Ok(Self {
            backbone: lm::qwen2::from_json(value)?,
            sample_rate,
            eos_token_id,
            bos_token_id,
            vocab_size,
        })
    }
}

/// Sampling and generation options with the reference `generate`
/// defaults.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct GenerateParams {
    /// 0.0 selects the exact greedy path (`make_sampler(0, ...)`).
    pub temperature: f32,
    /// The reference default 0.95.
    pub top_p: f32,
    /// The reference default 50; disabled at 0 or negative.
    pub top_k: i32,
    #[serde(default)]
    pub logit_bias: Vec<(i32, f32)>,
    /// The reference generate kwargs default 1.3.
    pub repetition_penalty: f32,
    /// The reference default 20.
    pub repetition_context_size: usize,
    pub max_tokens: usize,
    /// The tokenizer end-of-text id `stream_generate` stops on without
    /// yielding; defaults to the config `eos_token_id`.
    #[serde(default)]
    pub stop_token_ids: Vec<i32>,
    /// The reference hard-codes 128258 (the semantic-end token in the
    /// published vocabulary) as the per-segment break; the frame the
    /// loop appends before breaking.
    pub semantic_end_token_id: i32,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 50,
            logit_bias: Vec::new(),
            repetition_penalty: 1.3,
            repetition_context_size: 20,
            max_tokens: 3000,
            stop_token_ids: Vec::new(),
            semantic_end_token_id: 128258,
        }
    }
}

impl GenerateParams {
    /// The `generate` entry-point validation (`max_tokens must be
    /// positive`, `top_k must be non-negative`, and so on).
    pub fn validate(&self) -> Result<()> {
        if self.max_tokens == 0 {
            return Err(SpeechError::Input {
                why: "max_tokens must be positive".into(),
            });
        }
        if self.top_k < 0 {
            return Err(SpeechError::Input {
                why: "top_k must be non-negative".into(),
            });
        }
        if self.temperature < 0.0 {
            return Err(SpeechError::Input {
                why: "temperature must be non-negative".into(),
            });
        }
        if !(0.0..=1.0).contains(&self.top_p) {
            return Err(SpeechError::Input {
                why: "top_p must be between 0 and 1".into(),
            });
        }
        if self.repetition_penalty < 0.0 {
            return Err(SpeechError::Input {
                why: "repetition_penalty must be non-negative".into(),
            });
        }
        Ok(())
    }
}

/// Why the generate loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The sampled id hit the config/params stop set (`stream_generate`
    /// breaks without yielding it).
    EndOfText,
    /// The semantic-end token (the reference `next_token == 128258`
    /// break, appended before the loop ends).
    SemanticEnd,
    /// The frame budget ran out.
    MaxTokens,
}

/// One generated segment: the sampled ids, the extracted BiCodec codes,
/// and why the loop stopped.
#[derive(Debug, Clone)]
pub struct Generated {
    pub tokens: Vec<i32>,
    pub semantic_codes: Vec<i32>,
    pub global_codes: Vec<i32>,
    pub stopped_by: StopReason,
}

/// The loaded family model (`Model`).
pub struct Model {
    pub config: ModelConfig,
    backbone: lm::Backbone,
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
        let parsed = ModelConfig::from_json(config)?;
        Ok(Self {
            backbone: lm::qwen2::load(weights, config)?,
            config: parsed,
        })
    }

    /// `process_prompt` at the string level (voice cloning): the task
    /// token, the content block (reference transcript then target
    /// text), and the global/semantic token block. Tokenization stays
    /// with the caller.
    pub fn process_prompt(
        text: &str,
        ref_text: Option<&str>,
        global_codes: &[i32],
        semantic_codes: &[i32],
    ) -> String {
        let global_tokens: String = global_codes
            .iter()
            .map(|&code| global_token(code))
            .collect();
        let mut inputs = vec![task_token("tts"), "<|start_content|>".to_string()];
        if let Some(ref_text) = ref_text {
            let semantic_tokens: String = semantic_codes
                .iter()
                .map(|&code| semantic_token(code))
                .collect();
            inputs.push(ref_text.to_string());
            inputs.push(text.to_string());
            inputs.push("<|end_content|>".to_string());
            inputs.push("<|start_global_token|>".to_string());
            inputs.push(global_tokens);
            inputs.push("<|end_global_token|>".to_string());
            inputs.push("<|start_semantic_token|>".to_string());
            inputs.push(semantic_tokens);
        } else {
            inputs.push(text.to_string());
            inputs.push("<|end_content|>".to_string());
            inputs.push("<|start_global_token|>".to_string());
            inputs.push(global_tokens);
            inputs.push("<|end_global_token|>".to_string());
        }
        inputs.concat()
    }

    /// `process_prompt_control` at the string level (voice creation):
    /// gender + pitch + speed attribute tokens. `pitch`/`speed` take
    /// the level labels (`very_low`..`very_high`), the form the
    /// reference generate passes after its float label maps.
    pub fn process_prompt_control(
        gender: &str,
        pitch: &str,
        speed: &str,
        text: &str,
    ) -> Result<String> {
        let gender_id = map_value(GENDER_MAP, gender, "gender")?;
        let pitch_id = map_value(LEVELS_MAP, pitch, "pitch")?;
        let speed_id = map_value(LEVELS_MAP, speed, "speed")?;
        Ok([
            task_token("controllable_tts").as_str(),
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_style_label|>",
        ]
        .concat()
            + &format!(
                "<|gender_{gender_id}|><|pitch_label_{pitch_id}|><|speed_label_{speed_id}|>"
            )
            + "<|end_style_label|>")
    }

    /// Runs one segment end to end from prompt ids (`stream_generate`
    /// for a single non-streaming segment): prompt cache prefill of
    /// every token but the last, then the sampled loop with the logits
    /// processors and the `make_sampler` chain. `seed` only feeds the
    /// sampled draws; temperature 0 never draws and is fully
    /// deterministic. The waveform decode is
    /// [`BiCodec::detokenize`] over `Generated.semantic_codes` plus the
    /// global codes (from the reference audio on the clone path, or the
    /// extracted `global_codes` on the control path).
    pub fn generate(
        &mut self,
        token_maps: &AudioTokenMaps,
        prompt_ids: &[i32],
        params: &GenerateParams,
        seed: u64,
    ) -> Result<Generated> {
        params.validate()?;
        if prompt_ids.is_empty() {
            return Err(SpeechError::Input {
                why: "empty prompt".into(),
            });
        }
        let mut caches = lm::make_prompt_cache(self.backbone.config.num_hidden_layers);
        // The reference prefill feeds every prompt token but the last
        // (in 2048-token chunks) before the first sampled step.
        if prompt_ids.len() > 1 {
            self.backbone.forward(
                lm::Input::Tokens(&prompt_ids[..prompt_ids.len() - 1]),
                Some(&mut caches),
            )?;
        }
        let mut context: Vec<i32> = Vec::with_capacity(params.max_tokens + 1);
        context.push(prompt_ids[prompt_ids.len() - 1]);
        let mut logits = self.backbone.forward(
            lm::Input::Tokens(&[prompt_ids[prompt_ids.len() - 1]]),
            Some(&mut caches),
        )?;
        let width = self.backbone.config.vocab_size;
        let mut tokens = Vec::with_capacity(params.max_tokens);
        let mut stopped_by = StopReason::MaxTokens;
        let mut keys = rng::KeySequence::new(seed);
        for step in 0..params.max_tokens {
            let mut processed = logits[logits.len() - width..].to_vec();
            if !params.logit_bias.is_empty() {
                sampling::apply_logit_bias(&mut processed, &params.logit_bias)?;
            }
            if params.repetition_penalty != 0.0 {
                sampling::apply_repetition_penalty(
                    &mut processed,
                    &context,
                    params.repetition_penalty,
                    params.repetition_context_size,
                )?;
            }
            let logprobs = sampling::log_softmax(&processed);
            let next = if params.temperature == 0.0 {
                sampling::argmax(&logprobs)
            } else {
                // make_sampler chain order: top-p, then min-p (never
                // enabled here), then XTC (not ported, no family
                // enables it), then top-k.
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
            let stop_ids: &[i32] = if params.stop_token_ids.is_empty() {
                std::slice::from_ref(&self.config.eos_token_id)
            } else {
                &params.stop_token_ids
            };
            if stop_ids.contains(&next) {
                stopped_by = StopReason::EndOfText;
                break;
            }
            context.push(next);
            tokens.push(next);
            if next == params.semantic_end_token_id {
                stopped_by = StopReason::SemanticEnd;
                break;
            }
            if step + 1 == params.max_tokens {
                stopped_by = StopReason::MaxTokens;
                break;
            }
            logits = self
                .backbone
                .forward(lm::Input::Tokens(&[next]), Some(&mut caches))?;
        }
        Ok(Generated {
            semantic_codes: token_maps.extract_semantic(&tokens),
            global_codes: token_maps.extract_global(&tokens),
            tokens,
            stopped_by,
        })
    }
}

/// Decodes a generated segment to a mono waveform (`generate`'s final
/// `self._audio_tokenizer.detokenize(global_token_ids, pred_semantic_ids)`).
/// The global codes come from the reference audio on the clone path or
/// from `Generated.global_codes` on the control path.
pub fn decode_audio(
    bicodec: &BiCodec,
    semantic_codes: &[i32],
    global_codes: &[i32],
) -> Result<Vec<f32>> {
    bicodec.detokenize(semantic_codes, global_codes)
}

fn task_token(name: &str) -> String {
    TASK_TOKEN_MAP
        .iter()
        .find(|(key, _)| *key == name)
        .map(|(_, token)| (*token).to_string())
        .unwrap_or_else(|| format!("<|task_{name}|>"))
}

fn map_value(map: &[(&str, i32)], key: &str, field: &str) -> Result<i32> {
    map.iter()
        .find(|(name, _)| *name == key)
        .map(|(_, value)| *value)
        .ok_or_else(|| SpeechError::Input {
            why: format!(
                "{field} must be one of {:?}",
                map.iter().map(|(n, _)| *n).collect::<Vec<_>>()
            ),
        })
}

#[cfg(test)]
mod tests;
