//! Qwen3 SNAC TTS family (VyvoTTS-Baker-style wrapper).
//!
//! Port of `mlx_audio/tts/models/qwen3/qwen3.py` at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/qwen3):
//! a dense qwen3 backbone (shared [`crate::lm`] port) that interleaves
//! text with discrete SNAC audio codes above a dedicated audio-token
//! offset, plus the flat-code grouping, output parsing, prompt layouts,
//! and the segment generate loop. See [README.md](README.md) for the
//! verification evidence and open items.
//!
//! Scope of this port, matching the reference single-segment path:
//! batch size 1, non-streaming decode (the whole segment decodes to one
//! waveform after generation), and prompt construction from
//! pre-tokenized ids. Text tokenization is a caller concern (the
//! reference loads a Hugging Face tokenizer from the model directory in
//! its `post_load_hook`); voice names are likewise folded into the token
//! ids by the caller because the reference prefixes them before
//! tokenization (`"{voice}: {prompt}"`).

use serde_json::Value;

use crate::codec::snac::Snac;
use crate::lm;
use crate::lm::engine::Input;
use crate::music::minimax_music3::rng;
use crate::tts::sampling;
use crate::{quant, Result, SpeechError};

/// Qwen3 tokenizer length the family's special tokens are anchored at.
pub const TOKENIZER_LENGTH: i32 = 151669;
pub const START_OF_TEXT: i32 = 151643;
pub const END_OF_TEXT: i32 = 151645;
/// Start of speech; `parse_output` crops after its last occurrence.
pub const START_OF_SPEECH: i32 = TOKENIZER_LENGTH + 1; // 151670
/// End of speech; the generate loop stops on it.
pub const END_OF_SPEECH: i32 = TOKENIZER_LENGTH + 2; // 151671
pub const START_OF_HUMAN: i32 = TOKENIZER_LENGTH + 3; // 151672
pub const END_OF_HUMAN: i32 = TOKENIZER_LENGTH + 4; // 151673
/// Start of AI (zeroprompt wrapper).
pub const START_OF_AI: i32 = TOKENIZER_LENGTH + 5; // 151674
/// End of AI (zeroprompt wrapper).
pub const END_OF_AI: i32 = TOKENIZER_LENGTH + 6; // 151675
pub const PAD_TOKEN: i32 = TOKENIZER_LENGTH + 7; // 151676
/// Audio code offset: flat code c rides token `AUDIO_TOKENS_START + c`.
pub const AUDIO_TOKENS_START: i32 = TOKENIZER_LENGTH + 10; // 151679
/// Flat-code group stride: one SNAC frame is 7 consecutive codes.
pub const CODES_PER_GROUP: usize = 7;
/// Reference sample rate (the deployed SNAC 24 kHz codec).
pub const DEFAULT_SAMPLE_RATE: u32 = 24000;

/// The TTS `ModelConfig`: the qwen3 backbone args plus the tokenizer
/// name and sample rate the reference dataclass adds.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub backbone: lm::DenseConfig,
    pub tokenizer_name: Option<String>,
    pub sample_rate: u32,
}

impl ModelConfig {
    pub fn from_json(value: &Value) -> Result<Self> {
        let backbone = lm::qwen3::from_json(value)?;
        let tokenizer_name = value
            .get("tokenizer_name")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let sample_rate = value
            .get("sample_rate")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_SAMPLE_RATE as u64) as u32;
        Ok(Self {
            backbone,
            tokenizer_name,
            sample_rate,
        })
    }
}

/// Sampling and generation options for one segment, with the reference
/// `generate` defaults.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    /// 0.0 selects the exact greedy path (`make_sampler(0, ...)`).
    pub temperature: f32,
    pub top_p: f32,
    /// `-1` (or `0`) disables, matching the reference `kwargs.get("top_k", -1)`.
    pub top_k: i32,
    pub logit_bias: Vec<(i32, f32)>,
    pub repetition_penalty: f32,
    pub repetition_context_size: usize,
    pub max_tokens: usize,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            temperature: 0.6,
            top_p: 0.8,
            top_k: -1,
            logit_bias: Vec::new(),
            repetition_penalty: 1.3,
            repetition_context_size: 20,
            max_tokens: 1200,
        }
    }
}

/// Why the generate loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    EndOfSpeech,
    MaxTokens,
}

/// One generated segment: sampled ids, the parsed SNAC codes, and the
/// decoded mono waveform.
#[derive(Debug, Clone)]
pub struct Generated {
    pub tokens: Vec<i32>,
    pub codes: Vec<i32>,
    pub samples: Vec<f32>,
    pub stopped_by: StopReason,
}

/// The loaded family model: backbone plus the family config.
pub struct Model {
    pub config: ModelConfig,
    pub backbone: lm::Backbone,
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
        let backbone = lm::Backbone::load(weights, config.backbone.clone())?;
        Ok(Self { config, backbone })
    }

    /// Reference `prepare_input_ids`, batch of one:
    /// `[zeroprompt] [START_OF_HUMAN] [text] [END_OF_TEXT END_OF_HUMAN]`.
    /// The Python `voice` argument prefixes `"{voice}: "` before
    /// tokenization, so callers fold it into `text_ids`.
    pub fn prepare_input_ids(text_ids: &[i32], zeroprompt: Option<&[i32]>) -> Vec<i32> {
        let mut ids = Vec::with_capacity(zeroprompt.map_or(0, <[i32]>::len) + text_ids.len() + 3);
        if let Some(zeroprompt) = zeroprompt {
            ids.extend_from_slice(zeroprompt);
        }
        ids.push(START_OF_HUMAN);
        ids.extend_from_slice(text_ids);
        ids.push(END_OF_TEXT);
        ids.push(END_OF_HUMAN);
        ids
    }

    /// Reference `prepare_zeroprompt`: the voice-clone prefix
    /// `[SOH] [transcript] [EOT EOH] [SOA SOS] [audio codes] [EOS EOA]`
    /// with the codec codes offset by [`AUDIO_TOKENS_START`]. The
    /// reference prints a warning that cloning does not work reliably
    /// on this model; this port carries the layout for parity only.
    pub fn prepare_zeroprompt(
        snac: &Snac,
        ref_audio: &[f32],
        ref_text_ids: &[i32],
    ) -> Result<Vec<i32>> {
        let codes = encode_audio_to_codes(snac, ref_audio)?;
        let mut ids = Vec::with_capacity(ref_text_ids.len() + codes.len() + 9);
        ids.push(START_OF_HUMAN);
        ids.extend_from_slice(ref_text_ids);
        ids.push(END_OF_TEXT);
        ids.push(END_OF_HUMAN);
        ids.push(START_OF_AI);
        ids.push(START_OF_SPEECH);
        for code in codes {
            ids.push(code + AUDIO_TOKENS_START);
        }
        ids.push(END_OF_SPEECH);
        ids.push(END_OF_AI);
        Ok(ids)
    }

    /// Reference `parse_output` for the single row this port generates:
    /// take the ids after the LAST [`START_OF_SPEECH`], drop every
    /// [`END_OF_SPEECH`], trim to whole 7-code groups, and subtract
    /// [`AUDIO_TOKENS_START`]. Ids that were not audio tokens keep the
    /// subtraction (they can go negative), exactly like the reference;
    /// the codec refuses them downstream.
    pub fn parse_output(input_ids: &[i32]) -> Vec<i32> {
        let start = match input_ids.iter().rposition(|&t| t == START_OF_SPEECH) {
            Some(index) => index + 1,
            None => 0,
        };
        let cropped: Vec<i32> = input_ids[start..]
            .iter()
            .copied()
            .filter(|&t| t != END_OF_SPEECH)
            .collect();
        let keep = cropped.len() / CODES_PER_GROUP * CODES_PER_GROUP;
        cropped[..keep]
            .iter()
            .map(|&t| t - AUDIO_TOKENS_START)
            .collect()
    }

    /// Runs one segment end to end: prompt cache, greedy or sampled
    /// generate loop with the reference logits processors, stop at
    /// [`END_OF_SPEECH`] or `max_tokens`, then parse and decode all
    /// codes to one waveform. `seed` only feeds the categorical draws;
    /// temperature 0 never draws and is fully deterministic.
    ///
    /// The decode uses the zero-noise codec backbone; the reference
    /// draws fresh normals for the SNAC noise terms at every decode.
    /// Pass a noise source to [`decode_audio_from_codes`] to reproduce
    /// that behavior instead.
    pub fn generate(
        &mut self,
        snac: &Snac,
        text_ids: &[i32],
        params: &GenerateParams,
        seed: u64,
    ) -> Result<Generated> {
        let prompt = Self::prepare_input_ids(text_ids, None);
        self.generate_from_prompt(snac, &prompt, params, seed)
    }

    /// [`Model::generate`] against a raw prompt (the zeroprompt or
    /// fixture path, where the caller owns the exact ids).
    pub fn generate_from_prompt(
        &mut self,
        snac: &Snac,
        prompt_ids: &[i32],
        params: &GenerateParams,
        seed: u64,
    ) -> Result<Generated> {
        let (tokens, stopped_by) = self.generate_tokens_from_prompt(prompt_ids, params, seed)?;
        let mut input_ids = prompt_ids.to_vec();
        input_ids.extend_from_slice(&tokens);
        let codes = Self::parse_output(&input_ids);
        let samples = if codes.is_empty() {
            Vec::new()
        } else {
            decode_audio_from_codes(snac, &codes, &mut || 0.0)?
        };
        Ok(Generated {
            tokens,
            codes,
            samples,
            stopped_by,
        })
    }

    /// The generate loop alone: prefill, greedy or sampled steps with the
    /// reference logits processors, and the stop conditions. Returns the
    /// sampled ids and why the loop stopped, without parsing or decoding.
    /// Callers that stream decode intervals (out of scope here) or that
    /// feed deliberately non-audio prompts drive this directly.
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
        let mut caches = lm::make_prompt_cache(self.backbone.config.num_hidden_layers);
        // The reference prefills everything except the last token
        // unconditionally (processors never see it), then steps the last
        // token through the processors.
        let (prefill, last) = prompt_ids.split_at(prompt_ids.len() - 1);
        if !prefill.is_empty() {
            self.backbone
                .forward(Input::Tokens(prefill), Some(&mut caches))?;
        }
        let mut logits = self
            .backbone
            .forward(Input::Tokens(last), Some(&mut caches))?;
        if logits.len() != self.backbone.config.vocab_size {
            return Err(SpeechError::Tensor {
                name: "logits".into(),
                why: format!(
                    "expected {} values for one position, got {}",
                    self.backbone.config.vocab_size,
                    logits.len()
                ),
            });
        }

        // The reference logits processors accumulate every token that
        // passed through a step call: the last prompt token first, then
        // each generated token.
        let mut context: Vec<i32> = Vec::with_capacity(params.max_tokens + 1);
        context.push(last[last.len() - 1]);
        let mut tokens = Vec::with_capacity(params.max_tokens);
        let mut stopped_by = StopReason::MaxTokens;
        let mut keys = rng::KeySequence::new(seed);

        for step in 0..params.max_tokens {
            let mut processed = logits;
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
            context.push(next);
            tokens.push(next);
            if next == END_OF_SPEECH {
                stopped_by = StopReason::EndOfSpeech;
                break;
            }
            if step + 1 == params.max_tokens {
                stopped_by = StopReason::MaxTokens;
                break;
            }
            logits = self
                .backbone
                .forward(Input::Tokens(&[next]), Some(&mut caches))?;
        }
        Ok((tokens, stopped_by))
    }
}

/// The flat-code to SNAC-level split the qwen3 reference performs inline
/// in `decode_audio_from_codes` (it has no standalone
/// `codes_to_layers`): flat 7-code groups into three SNAC levels (one
/// L1 code, two L2 codes, four L3 codes per group). The reference group
/// count is `(len + 1) / 7`, which overruns on a remainder-of-6 tail;
/// this port refuses instead of indexing out of bounds.
pub fn codes_to_layers(code_list: &[i32]) -> Result<Vec<Vec<i32>>> {
    let groups = (code_list.len() + 1) / CODES_PER_GROUP;
    if groups * CODES_PER_GROUP > code_list.len() {
        return Err(SpeechError::Input {
            why: format!(
                "flat code list of {} is one code short of {groups} complete groups",
                code_list.len()
            ),
        });
    }
    let mut layer_1 = vec![0i32; groups];
    let mut layer_2 = vec![0i32; groups * 2];
    let mut layer_3 = vec![0i32; groups * 4];
    for (group, chunk) in code_list.chunks_exact(CODES_PER_GROUP).enumerate() {
        layer_1[group] = chunk[0];
        layer_2[group * 2] = chunk[1] - 4096;
        layer_3[group * 4] = chunk[2] - 2 * 4096;
        layer_3[group * 4 + 1] = chunk[3] - 3 * 4096;
        layer_2[group * 2 + 1] = chunk[4] - 4 * 4096;
        layer_3[group * 4 + 2] = chunk[5] - 5 * 4096;
        layer_3[group * 4 + 3] = chunk[6] - 6 * 4096;
    }
    Ok(vec![layer_1, layer_2, layer_3])
}

/// Reference `decode_audio_from_codes`: split into SNAC levels and
/// decode to mono samples. `noise` feeds the codec's optional noise
/// terms in consumption order; pass `|| 0.0` for the deterministic
/// zero-noise backbone (the deployed 24 kHz checkpoint enables them and
/// the reference draws fresh normals per decode).
pub fn decode_audio_from_codes(
    snac: &Snac,
    code_list: &[i32],
    noise: &mut dyn FnMut() -> f32,
) -> Result<Vec<f32>> {
    if code_list.is_empty() {
        return Err(SpeechError::Input {
            why: "empty code list".into(),
        });
    }
    let layers = codes_to_layers(code_list)?;
    snac.decode_with_noise(&layers, noise)
}

/// Reference `encode_audio_to_codes`: codec-encode mono samples and
/// interleave the levels back into flat 7-code groups with the per-slot
/// offsets. The inverse of [`codes_to_layers`].
pub fn encode_audio_to_codes(snac: &Snac, samples: &[f32]) -> Result<Vec<i32>> {
    let codes = snac.encode(samples)?;
    if codes.len() != 3 {
        return Err(SpeechError::Input {
            why: format!("expected 3 codec levels, got {}", codes.len()),
        });
    }
    let groups = codes[0].len();
    if codes[1].len() != groups * 2 || codes[2].len() != groups * 4 {
        return Err(SpeechError::Input {
            why: format!(
                "codec levels {} x {} x {} do not pack into {groups} groups",
                codes[0].len(),
                codes[1].len(),
                codes[2].len()
            ),
        });
    }
    let mut flat = Vec::with_capacity(groups * CODES_PER_GROUP);
    for group in 0..groups {
        flat.push(codes[0][group]);
        flat.push(codes[1][group * 2] + 4096);
        flat.push(codes[2][group * 4] + 2 * 4096);
        flat.push(codes[2][group * 4 + 1] + 3 * 4096);
        flat.push(codes[1][group * 2 + 1] + 4 * 4096);
        flat.push(codes[2][group * 4 + 2] + 5 * 4096);
        flat.push(codes[2][group * 4 + 3] + 6 * 4096);
    }
    Ok(flat)
}

#[cfg(test)]
mod tests;
