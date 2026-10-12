//! DramaBox TTS family (LTX-2.3 audio-only DiT flow matching).
//!
//! Port of `mlx_audio/tts/models/dramabox/` at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/dramabox):
//! a latent flow-matching loop over patchified audio-mel latents. Each
//! step runs the audio-only LTX transformer (AdaLN-single timestep
//! conditioning, gated split-rope self attention over the patch
//! stream, prompt-AdaLN-modulated cross attention to the text
//! context) as an x0 model, combines cond/uncond/STG predictions with
//! the multi-modal guider and its std rescale, and Euler-integrates
//! the LTX2 sigma schedule; the result unpatchifies into
//! `[channels, frames, mel_bins]` latents with the long-clip silence
//! prior patch.
//!
//! Scope of this port, matching the reference zero-reference path:
//! batch 1, f32, and the pipeline ENDS AT THE MEL LATENTS with a
//! documented refusal for the AudioVAE decoder and vocoder (they are
//! in-checkpoint modules not yet ported; see the README open items).
//! The Gemma-3 text encoder is a SEPARATE downloadable model
//! (`mlx-community/gemma-3-12b-it-8bit`), so text conditioning is
//! caller-owned: the caller supplies the stacked Gemma hidden states
//! `[tokens, hidden * layers]` and the attention mask, and the ported
//! conditioner (per-token per-layer RMS, register reorder, rope'd
//! connector blocks) produces the cross-attention context. The
//! reference-audio voice-clone path needs the AudioVAE encoder and is
//! not ported. The noise draw (`add_gaussian_noise`, one
//! standard-normal over the patched latent) comes from a
//! [`DiffusionNoise`] provider. See [README.md](README.md) for the
//! verification evidence and open items.

use serde_json::Value;

use self::conditioning::TextConditioner;
use self::diffusion::{
    auto_rescale_for_cfg, calculate_guided_prediction, euler_step, ltx2_sigmas, patch_grid_bounds,
    patch_long_clip_silence_prior, to_denoised, GuiderParams,
};
use self::dit::{
    adaln_embedding_coefficient, split_freqs_cis, AdaLayerNormSingle, TransformerBlock,
};
use self::sharded::{load_linear_sharded, load_sharded};
use crate::music::minimax_music3::rng;
use crate::nn::Linear;
use crate::{quant, Result, SpeechError};

mod conditioning;
mod diffusion;
mod dit;
mod sharded;

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Duration estimation (duration.py)
// ---------------------------------------------------------------------------

/// The nonverbal keyword durations, ported as whole-word
/// case-insensitive literal matches (the reference `\b`-anchored
/// alternations expanded to their literal variants).
const NONVERBAL_PATTERNS: &[(&str, f32)] = &[
    ("sigh", 0.8),
    ("sighs", 0.8),
    ("shaky breath", 1.0),
    ("breathing deeply", 1.0),
    ("gasp", 0.5),
    ("gasps", 0.5),
    ("burp", 0.5),
    ("burps", 0.5),
    ("yawn", 1.0),
    ("yawns", 1.0),
    ("pant", 0.8),
    ("pants", 0.8),
    ("wheeze", 0.8),
    ("wheezes", 0.8),
    ("cough", 0.8),
    ("coughs", 0.8),
    ("sniffle", 0.5),
    ("sniffles", 0.5),
    ("snort", 0.3),
    ("snorts", 0.3),
    ("groan", 0.8),
    ("groans", 0.8),
    ("long pause", 1.0),
    ("pause briefly", 0.3),
    ("pauses briefly", 0.3),
    ("pause", 0.5),
    ("pauses", 0.5),
    ("silence", 1.0),
    ("slam", 0.5),
    ("slams", 0.5),
    ("clap", 0.3),
    ("claps", 0.3),
    ("whistle", 1.0),
    ("whistles", 1.0),
    ("hum", 0.8),
    ("hums", 0.8),
    ("mutter", 1.5),
    ("mutters", 1.5),
    ("mumble", 1.0),
    ("mumbles", 1.0),
    ("whisper", 0.0),
    ("whispers", 0.0),
    ("gulp", 0.5),
    ("gulps", 0.5),
    ("swallow", 0.5),
    ("swallows", 0.5),
    ("steady himself", 1.0),
    ("steadies himself", 1.0),
    ("steady herself", 1.0),
    ("steadies herself", 1.0),
    ("catch his breath", 1.0),
    ("catches his breath", 1.0),
    ("catch her breath", 1.0),
    ("catches her breath", 1.0),
    ("compose himself", 0.8),
    ("composes himself", 0.8),
    ("compose herself", 0.8),
    ("composes herself", 0.8),
    ("demeanor shift", 0.5),
    ("demeanor shifts", 0.5),
    ("settle in", 0.5),
    ("settles in", 0.5),
    ("lean in", 0.3),
    ("leans in", 0.3),
    ("wipe his eyes", 0.5),
    ("wipes his eyes", 0.5),
    ("wipe her eyes", 0.5),
    ("wipes her eyes", 0.5),
];

/// The laugh verb base durations (the reference alternation groups
/// expanded to literal stems the matcher treats as prefixes).
const LAUGH_VERBS: &[(&str, f32)] = &[
    ("laugh", 1.5),
    ("cackle", 1.5),
    ("chuckle", 1.0),
    ("giggle", 1.0),
    ("snicker", 0.8),
];

fn count_word_occurrences(lower_text: &str, needle: &str) -> usize {
    let mut count = 0;
    let mut start = 0;
    while let Some(found) = lower_text[start..].find(needle) {
        let begin = start + found;
        let end = begin + needle.len();
        let before_ok = begin == 0
            || !lower_text[..begin]
                .chars()
                .next_back()
                .map(|c| c.is_ascii_alphanumeric())
                .unwrap_or(false);
        let after_ok = end == lower_text.len()
            || !lower_text[end..]
                .chars()
                .next()
                .map(|c| c.is_ascii_alphanumeric())
                .unwrap_or(false);
        if before_ok && after_ok {
            count += 1;
            start = end;
        } else {
            start = begin + 1;
        }
    }
    count
}

/// The prefix modifiers of `_contextual_laugh_duration` (checked
/// against the 40 chars after the match, case-insensitive).
const SHORT_LAUGH_MODIFIERS: &[&str] = &["briefly", "shortly", "once", "quickly"];
const LONG_LAUGH_MODIFIERS: &[&str] = &[
    "maniacally",
    "heartily",
    "uproariously",
    "uncontrollably",
    "hysterically",
    "darkly",
    "wickedly",
    "evilly",
    "loudly",
    "long",
];

fn laugh_context_multiplier(context: &str) -> f32 {
    // `^\s*(?:[a-z]+ly )?(modifier)`: one optional "-ly" word then a
    // modifier at the start.
    let context = context.trim_start();
    for modifier in SHORT_LAUGH_MODIFIERS {
        if context_starts_with_word(context, modifier) {
            return 0.4;
        }
    }
    for modifier in LONG_LAUGH_MODIFIERS {
        if context_starts_with_word(context, modifier) {
            return 1.2;
        }
    }
    1.0
}

/// `context.startswith(modifier + word boundary)` after the optional
/// "-ly" adverb the regex allows.
fn context_starts_with_word(context: &str, modifier: &str) -> bool {
    let mut rest = context;
    // Optionally skip one "[a-z]+ly " adverb.
    if let Some(space) = rest.find(' ') {
        let word = &rest[..space];
        if word.ends_with("ly") && word.len() > 2 && word.chars().all(|c| c.is_ascii_lowercase()) {
            rest = rest[space + 1..].trim_start();
        }
    }
    if let Some(stripped) = rest.strip_prefix(modifier) {
        if stripped.is_empty() || !stripped.chars().next().unwrap().is_ascii_alphanumeric() {
            return true;
        }
    }
    false
}

/// `_estimate_nonverbal_duration` plus the contextual laugh terms.
/// The quoted-text "haha" runs of the reference are ASCII-literal and
/// ported too.
pub fn estimate_nonverbal_duration(text: &str) -> f32 {
    let lower = text.to_lowercase();
    let mut extra = 0.0f32;
    for (pattern, duration) in NONVERBAL_PATTERNS {
        extra += duration * count_word_occurrences(&lower, pattern) as f32;
    }
    // Laugh verbs with the contextual multiplier.
    for (stem, base) in LAUGH_VERBS {
        let variants: &[&str] = match *stem {
            "laugh" => &["laugh", "laughs", "laughed", "laughing"],
            "cackle" => &["cackle", "cackles", "cackled", "cackling"],
            "chuckle" => &["chuckle", "chuckles", "chuckled", "chuckling"],
            "giggle" => &["giggle", "giggles", "giggled", "giggling"],
            _ => &["snicker", "snickers", "snickered", "snickering"],
        };
        for variant in variants {
            let mut start = 0;
            while let Some(found) = lower[start..].find(variant) {
                let begin = start + found;
                let end = begin + variant.len();
                let before_ok = begin == 0
                    || !lower[..begin]
                        .chars()
                        .next_back()
                        .map(|c| c.is_ascii_alphanumeric())
                        .unwrap_or(false);
                let after_ok = end == lower.len()
                    || !lower[end..]
                        .chars()
                        .next()
                        .map(|c| c.is_ascii_alphanumeric())
                        .unwrap_or(false);
                if before_ok && after_ok {
                    let context_end = (end + 40).min(lower.len());
                    extra += base * laugh_context_multiplier(&lower[end..context_end]);
                }
                start = begin + 1;
            }
        }
    }
    // Quoted "haha" runs: 0.2 per syllable beyond two.
    extra += quoted_laugh_runs(text);
    extra
}

/// The quoted-text `(?:h[ae]){3,}` run term.
fn quoted_laugh_runs(text: &str) -> f32 {
    let mut total = 0.0f32;
    let chars: Vec<char> = text.chars().collect();
    let mut quote: Option<char> = None;
    let mut segment: Vec<char> = Vec::new();
    for &ch in &chars {
        match quote {
            None if ch == '"' || ch == '\'' => quote = Some(ch),
            Some(open) if ch == open => {
                total += laugh_runs_in(&segment);
                quote = None;
                segment.clear();
            }
            Some(_) => segment.push(ch),
            None => {}
        }
    }
    total
}

fn laugh_runs_in(segment: &[char]) -> f32 {
    let lower: Vec<char> = segment.iter().map(|c| c.to_ascii_lowercase()).collect();
    let mut total = 0.0f32;
    let mut index = 0;
    while index < lower.len() {
        if lower[index] == 'h' && index + 1 < lower.len() && matches!(lower[index + 1], 'a' | 'e') {
            let mut syllables: i32 = 0;
            let mut cursor = index;
            while cursor + 1 < lower.len()
                && lower[cursor] == 'h'
                && matches!(lower[cursor + 1], 'a' | 'e')
            {
                syllables += 1;
                cursor += 2;
                // Optional separator: space or dash.
                if cursor + 1 < lower.len()
                    && (lower[cursor] == ' ' || lower[cursor] == '-')
                    && cursor + 1 < lower.len()
                    && lower[cursor + 1] == 'h'
                {
                    cursor += 1;
                }
            }
            if syllables >= 3 {
                total += 0.2 * (syllables.saturating_sub(2)) as f32;
            }
            index = cursor;
        } else {
            index += 1;
        }
    }
    total
}

/// `estimate_speech_duration`: the quoted-speech extraction, the
/// length-tiered characters-per-second, the sentence punctuation
/// pauses, the nonverbal terms, and the 2 second floor pad.
pub fn estimate_speech_duration(text: &str, speed: f32) -> f32 {
    let spoken = spoken_text(text);
    let mut chars_per_second = 14.0f32;
    let text_length = spoken.chars().count();
    if text_length < 40 {
        chars_per_second *= 0.6;
    } else if text_length < 80 {
        chars_per_second *= 0.8;
    }
    let mut duration = text_length as f32 / (chars_per_second * speed);
    duration += spoken.matches('.').count() as f32 * 0.3;
    duration += spoken.matches('!').count() as f32 * 0.3;
    duration += spoken.matches('?').count() as f32 * 0.3;
    duration += estimate_nonverbal_duration(text);
    // round(duration + 2.0, 1)
    let rounded = ((duration + 2.0) * 10.0).round() / 10.0;
    rounded.max(3.0)
}

/// The spoken-text extraction: double-quoted spans win, then
/// single-quoted spans longer than 3 words, then the text after the
/// first colon, then the whole text.
fn spoken_text(text: &str) -> String {
    let double_quoted = quoted_spans(text, '"');
    if !double_quoted.is_empty() {
        return double_quoted.join(" ");
    }
    let single_quoted: Vec<String> = single_quoted_spans(text)
        .into_iter()
        .filter(|span| span.split_whitespace().count() > 3)
        .collect();
    if !single_quoted.is_empty() {
        return single_quoted.join(" ");
    }
    if let Some(colon) = text.find(':') {
        return text[colon + 1..].trim().to_owned();
    }
    text.to_owned()
}

fn quoted_spans(text: &str, quote: char) -> Vec<String> {
    let mut spans = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find(quote) {
        if let Some(length) = rest[open + 1..].find(quote) {
            spans.push(rest[open + 1..open + 1 + length].to_owned());
            rest = &rest[open + length + 2..];
        } else {
            break;
        }
    }
    spans
}

/// The single-quote span rule `((?:[^']|'(?![\s.,!?)\]))+)'`: quotes
/// run until an apostrophe followed by whitespace/punctuation/end.
fn single_quoted_spans(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut spans = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        if chars[index] == '\'' {
            let mut span = Vec::new();
            let mut cursor = index + 1;
            let mut closed = false;
            while cursor < chars.len() {
                let ch = chars[cursor];
                if ch == '\'' {
                    let next_is_break = cursor + 1 >= chars.len()
                        || matches!(chars[cursor + 1], ' ' | '.' | ',' | '!' | '?' | ')' | ']');
                    if next_is_break {
                        closed = true;
                        break;
                    }
                    span.push(ch);
                    cursor += 1;
                } else {
                    span.push(ch);
                    cursor += 1;
                }
            }
            if closed {
                spans.push(span.into_iter().collect());
                index = cursor + 1;
            } else {
                index = cursor;
            }
        } else {
            index += 1;
        }
    }
    spans
}

/// `resolve_generation_duration`.
pub fn resolve_generation_duration(
    prompt: &str,
    speed: f32,
    duration_multiplier: f32,
    gen_duration: f32,
) -> f32 {
    if gen_duration > 0.0 {
        return gen_duration;
    }
    let estimated = estimate_speech_duration(prompt, speed) * duration_multiplier;
    // round(x, 1)
    let rounded = (estimated * 10.0).round() / 10.0;
    rounded.max(3.0)
}

/// `aligned_frame_count`: round to 25 fps, then to the 8-token grid.
pub fn aligned_frame_count(duration: f32, fps: f32) -> usize {
    let frames = (duration * fps).round() as i64 + 1;
    (((frames - 1 + 4) / 8) * 8 + 1) as usize
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// The transformer config (`TransformerConfig`); only the fields the
/// ported path reads are kept.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub num_layers: usize,
    pub audio_num_attention_heads: usize,
    pub audio_attention_head_dim: usize,
    pub audio_in_channels: usize,
    pub audio_out_channels: usize,
    pub audio_cross_attention_dim: usize,
    pub norm_eps: f32,
    pub positional_embedding_theta: f64,
    pub audio_positional_embedding_max_pos: f32,
    pub timestep_scale_multiplier: usize,
    pub use_middle_indices_grid: bool,
    pub rope_split: bool,
    pub frequencies_precision_float64: bool,
    pub apply_gated_attention: bool,
    pub cross_attention_adaln: bool,
    pub connector_layers: usize,
    pub connector_heads: usize,
    pub connector_head_dim: usize,
    pub connector_theta: f64,
    pub connector_max_pos: f32,
    pub connector_num_learnable_registers: usize,
}

impl TransformerConfig {
    fn from_json(value: &Value) -> Self {
        let i = |key: &str, default: usize| {
            value
                .get(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(default)
        };
        let f = |key: &str, default: f64| value.get(key).and_then(Value::as_f64).unwrap_or(default);
        let list_f = |key: &str, default: f32| -> f32 {
            value
                .get(key)
                .and_then(Value::as_array)
                .and_then(|list| list.first())
                .and_then(Value::as_f64)
                .map(|v| v as f32)
                .unwrap_or(default)
        };
        Self {
            num_layers: i("num_layers", 48),
            audio_num_attention_heads: i("audio_num_attention_heads", 32),
            audio_attention_head_dim: i("audio_attention_head_dim", 64),
            audio_in_channels: i("audio_in_channels", 128),
            audio_out_channels: i("audio_out_channels", 128),
            audio_cross_attention_dim: i("audio_cross_attention_dim", 2048),
            norm_eps: f("norm_eps", 1e-6) as f32,
            positional_embedding_theta: f("positional_embedding_theta", 10000.0),
            audio_positional_embedding_max_pos: list_f("audio_positional_embedding_max_pos", 20.0),
            timestep_scale_multiplier: i("timestep_scale_multiplier", 1000),
            use_middle_indices_grid: value
                .get("use_middle_indices_grid")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            rope_split: value
                .get("rope_type")
                .and_then(Value::as_str)
                .map(|t| t == "split")
                .unwrap_or(true),
            frequencies_precision_float64: matches!(
                value.get("frequencies_precision").and_then(Value::as_str),
                Some("float64")
            ) || value
                .get("frequencies_precision")
                .map(|v| !v.is_null() && v.as_str().is_none())
                .unwrap_or(false),
            apply_gated_attention: value
                .get("apply_gated_attention")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            cross_attention_adaln: value
                .get("cross_attention_adaln")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            connector_layers: i("connector_num_layers", 8),
            connector_heads: i("audio_connector_num_attention_heads", 32),
            connector_head_dim: i("audio_connector_attention_head_dim", 64),
            connector_theta: f("positional_embedding_theta", 10000.0),
            connector_max_pos: list_f("connector_positional_embedding_max_pos", 4096.0),
            connector_num_learnable_registers: i("connector_num_learnable_registers", 128),
        }
    }
}

/// The audio config (`AudioConfig`).
#[derive(Debug, Clone)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub latent_sample_rate: u32,
    pub hop_length: usize,
    pub latent_downsample_factor: usize,
    pub vae_channels: usize,
    pub mel_bins: usize,
    pub fps: f32,
}

impl AudioConfig {
    fn from_json(value: &Value) -> Self {
        let i = |key: &str, default: usize| {
            value
                .get(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(default)
        };
        Self {
            sample_rate: i("sample_rate", 48000) as u32,
            latent_sample_rate: i("latent_sample_rate", 16000) as u32,
            hop_length: i("hop_length", 160),
            latent_downsample_factor: i("latent_downsample_factor", 4),
            vae_channels: i("vae_channels", 8),
            mel_bins: i("mel_bins", 16),
            fps: value.get("fps").and_then(Value::as_f64).unwrap_or(25.0) as f32,
        }
    }
}

/// The inference defaults (`InferenceDefaults`).
#[derive(Debug, Clone)]
pub struct InferenceDefaults {
    pub cfg_scale: f32,
    pub stg_scale: f32,
    pub stg_block: usize,
    pub modality_scale: f32,
    pub duration_multiplier: f32,
    pub seed: u64,
    pub steps: usize,
    pub negative_prompt: String,
}

impl InferenceDefaults {
    fn from_json(value: &Value) -> Self {
        Self {
            cfg_scale: value
                .get("cfg_scale")
                .and_then(Value::as_f64)
                .unwrap_or(2.5) as f32,
            stg_scale: value
                .get("stg_scale")
                .and_then(Value::as_f64)
                .unwrap_or(1.5) as f32,
            stg_block: value
                .get("stg_block")
                .and_then(Value::as_u64)
                .unwrap_or(29) as usize,
            modality_scale: value
                .get("modality_scale")
                .and_then(Value::as_f64)
                .unwrap_or(1.0) as f32,
            duration_multiplier: value
                .get("duration_multiplier")
                .and_then(Value::as_f64)
                .unwrap_or(1.1) as f32,
            seed: value
                .get("seed")
                .and_then(Value::as_u64)
                .unwrap_or(42),
            steps: value.get("steps").and_then(Value::as_u64).unwrap_or(30) as usize,
            negative_prompt: value
                .get("negative_prompt")
                .and_then(Value::as_str)
                .unwrap_or("worst quality, inconsistent motion, blurry, jittery, distorted, robotic voice, echo, background noise, off-sync audio, repetitive speech")
                .to_owned(),
        }
    }
}

/// The model config (`ModelConfig.from_dict`).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub transformer: TransformerConfig,
    pub audio: AudioConfig,
    pub inference_defaults: InferenceDefaults,
    pub text_encoder_hidden_size: usize,
    /// The stacked Gemma layers the conditioner consumes; the
    /// reference hardcodes 49 and the field is absent in real configs.
    pub num_gemma_layers: usize,
    pub sample_rate: u32,
}

impl ModelConfig {
    pub fn from_json(value: &Value) -> Result<Self> {
        let transformer_value = value.get("transformer").cloned().unwrap_or(Value::Null);
        let audio_value = value.get("audio").cloned().unwrap_or(Value::Null);
        let defaults_value = value
            .get("inference_defaults")
            .cloned()
            .unwrap_or(Value::Null);
        Ok(Self {
            transformer: TransformerConfig::from_json(&transformer_value),
            audio: AudioConfig::from_json(&audio_value),
            inference_defaults: InferenceDefaults::from_json(&defaults_value),
            text_encoder_hidden_size: value
                .get("text_encoder_hidden_size")
                .and_then(Value::as_u64)
                .unwrap_or(3840) as usize,
            num_gemma_layers: value
                .get("num_gemma_layers")
                .and_then(Value::as_u64)
                .unwrap_or(49) as usize,
            sample_rate: value
                .get("sample_rate")
                .and_then(Value::as_u64)
                .map(|n| n as u32)
                .unwrap_or(48000),
        })
    }
}

// ---------------------------------------------------------------------------
// Noise provider
// ---------------------------------------------------------------------------

/// The initial-noise source: one standard-normal draw over the patched
/// latent `[tokens, channels * mel_bins]`, in reference draw order.
pub trait DiffusionNoise {
    fn normal(&mut self, rows: usize, cols: usize) -> Vec<f32>;
}

/// Seeded crate-local threefry noise for standalone runs.
pub struct SeededNoise {
    keys: rng::KeySequence,
}

impl SeededNoise {
    pub fn new(seed: u64) -> Self {
        Self {
            keys: rng::KeySequence::new(seed),
        }
    }
}

impl DiffusionNoise for SeededNoise {
    fn normal(&mut self, rows: usize, cols: usize) -> Vec<f32> {
        rng::normal(self.keys.next(), rows * cols)
    }
}

// ---------------------------------------------------------------------------
// The loaded model
// ---------------------------------------------------------------------------

/// The audio-only LTX transformer (`AudioOnlyLTXModel`) plus the text
/// conditioner. The AudioVAE and vocoder are refused; see the README.
pub struct Model {
    pub config: ModelConfig,
    pub(crate) patchify_proj: Linear,
    adaln_single: AdaLayerNormSingle,
    prompt_adaln_single: Option<AdaLayerNormSingle>,
    scale_shift_table: Vec<f32>,
    norm_eps: f32,
    proj_out: Linear,
    pub(crate) blocks: Vec<TransformerBlock>,
    text_conditioner: TextConditioner,
    inner_dim: usize,
    heads: usize,
}

impl Model {
    /// Opens a checkpoint directory: the transformer weights ride
    /// `model.safetensors` shards and the conditioner joins them.
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        let parsed = ModelConfig::from_json(&config_value)?;
        let weights = crate::lm::Weights::open(dir)?;
        Self::load_weights(&weights, parsed)
    }

    fn load_weights(weights: &crate::lm::Weights, config: ModelConfig) -> Result<Self> {
        let files = weights.shards();
        let t = &config.transformer;
        let inner_dim = t.audio_num_attention_heads * t.audio_attention_head_dim;
        let coefficient = adaln_embedding_coefficient(t.cross_attention_adaln);
        let patchify_proj = load_linear_sharded(
            files,
            "transformer.audio_patchify_proj",
            t.audio_in_channels,
            inner_dim,
        )?;
        let adaln_single = AdaLayerNormSingle::load(
            files,
            "transformer.audio_adaln_single",
            inner_dim,
            coefficient,
        )?;
        let prompt_adaln_single = if t.cross_attention_adaln {
            Some(AdaLayerNormSingle::load(
                files,
                "transformer.audio_prompt_adaln_single",
                inner_dim,
                2,
            )?)
        } else {
            None
        };
        let scale_shift_table = load_sharded(
            files,
            "transformer.audio_scale_shift_table",
            &[2, inner_dim],
        )?;
        // The reference LayerNorm is affine-free: no weight tensor in
        // the checkpoint; the port normalizes without one.
        let proj_out = load_linear_sharded(
            files,
            "transformer.audio_proj_out",
            inner_dim,
            t.audio_out_channels,
        )?;
        let mut blocks = Vec::with_capacity(t.num_layers);
        for index in 0..t.num_layers {
            blocks.push(TransformerBlock::load(
                files,
                &format!("transformer.transformer_blocks.{index}"),
                inner_dim,
                t.audio_num_attention_heads,
                t.audio_attention_head_dim,
                t.audio_cross_attention_dim,
                t.norm_eps,
                t.cross_attention_adaln,
                t.apply_gated_attention,
                t.rope_split,
            )?);
        }
        let text_encoder_hidden_size = config.text_encoder_hidden_size;
        let text_conditioner = TextConditioner::load(
            files,
            text_encoder_hidden_size,
            t.audio_cross_attention_dim,
            config.num_gemma_layers,
            t.connector_layers,
            t.connector_heads,
            t.connector_head_dim,
            t.connector_num_learnable_registers,
            t.connector_theta,
            t.connector_max_pos,
        )?;
        let heads = t.audio_num_attention_heads;
        let norm_eps = t.norm_eps;
        Ok(Self {
            config,
            patchify_proj,
            adaln_single,
            prompt_adaln_single,
            scale_shift_table,
            norm_eps,
            proj_out,
            blocks,
            text_conditioner,
            inner_dim,
            heads,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.audio.sample_rate
    }

    /// `target_shape_for_duration`: the token count for a duration.
    pub fn tokens_for_duration(&self, duration_with_pad: f32) -> usize {
        let frames = aligned_frame_count(duration_with_pad, self.config.audio.fps);
        let latents_per_second = self.config.audio.latent_sample_rate as f32
            / self.config.audio.hop_length as f32
            / self.config.audio.latent_downsample_factor as f32;
        let latent_duration = frames as f32 / self.config.audio.fps;
        (latent_duration * latents_per_second).round() as usize
    }

    /// The text conditioner: Gemma hidden states in, cross context out.
    pub fn encode_context(
        &self,
        hidden_states: &[f32],
        tokens: usize,
        attention_mask: &[i32],
    ) -> Result<Vec<f32>> {
        self.text_conditioner
            .forward(hidden_states, tokens, attention_mask)
    }

    /// One transformer pass (`X0Model` velocity output): the patch
    /// projection, the AdaLN blocks, and the scale/shift output head.
    #[allow(clippy::too_many_arguments)]
    fn run_pass(
        &self,
        latent: &[f32],
        tokens: usize,
        context: &[f32],
        ctx_tokens: usize,
        timestep_out: &[f32],
        prompt_timestep: &[f32],
        embedded_timestep: &[f32],
        cos: &[f32],
        sin: &[f32],
        skip_audio_self_attn: Option<usize>,
    ) -> Result<Vec<f32>> {
        let mut x = self.patchify_proj.forward(latent, tokens);
        for (index, block) in self.blocks.iter().enumerate() {
            block.forward(
                &mut x,
                tokens,
                context,
                ctx_tokens,
                timestep_out,
                prompt_timestep,
                cos,
                sin,
                Some(index) == skip_audio_self_attn,
            )?;
        }
        let dim = self.inner_dim;
        let mut shift = vec![0.0f32; tokens * dim];
        let mut scale = vec![0.0f32; tokens * dim];
        for token in 0..tokens {
            for col in 0..dim {
                shift[token * dim + col] =
                    self.scale_shift_table[col] + embedded_timestep[token * dim + col];
                scale[token * dim + col] =
                    self.scale_shift_table[dim + col] + embedded_timestep[token * dim + col];
            }
        }
        let mut out = x;
        layer_norm_affine_free(&mut out, tokens, dim, self.norm_eps);
        for row in 0..tokens {
            for col in 0..dim {
                out[row * dim + col] =
                    out[row * dim + col] * (1.0 + scale[row * dim + col]) + shift[row * dim + col];
            }
        }
        Ok(self.proj_out.forward(&out, tokens))
    }

    /// The reference zero-reference `generate` flow up to the mel
    /// latents: initial state, one recorded noise draw, the guided
    /// Euler loop over the LTX2 sigma schedule, and the unpatchify
    /// plus silence-prior patch. `context` and `negative_context` come
    /// from [`Model::encode_context`] over the caller-supplied Gemma
    /// hidden states.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &mut self,
        tokens: usize,
        context: &[f32],
        context_tokens: usize,
        negative_context: Option<&[f32]>,
        negative_context_tokens: usize,
        params: &GenerateParams,
        noise: &mut dyn DiffusionNoise,
    ) -> Result<Generated> {
        let t = self.config.transformer.clone();
        let width = t.audio_in_channels;
        let positions = patch_grid_bounds(
            tokens,
            self.config.audio.latent_downsample_factor,
            self.config.audio.hop_length,
            self.config.audio.latent_sample_rate as usize,
        );

        // Initial state: zero latents, all-one denoise mask, then the
        // one recorded Gaussian draw over the patched latent.
        let mut current = noise.normal(tokens, width);

        let guider = GuiderParams {
            cfg_scale: params.cfg_scale,
            stg_scale: params.stg_scale,
            rescale_scale: if params.rescale_scale_auto {
                auto_rescale_for_cfg(params.cfg_scale)
            } else {
                params.rescale_scale
            },
            modality_scale: params.modality_scale,
        };

        let sigmas = ltx2_sigmas(params.steps, width);
        let (cos, sin) = split_freqs_cis(
            &positions,
            tokens,
            1,
            &[t.audio_positional_embedding_max_pos],
            t.positional_embedding_theta,
            self.inner_dim,
            self.heads,
            t.frequencies_precision_float64,
        )?;

        for step_index in 0..sigmas.len() - 1 {
            let sigma = sigmas[step_index];
            // Per-token timesteps: the all-one denoise mask times the
            // scaled sigma; the prompt AdaLN sees one sigma token.
            let timesteps = vec![sigma * t.timestep_scale_multiplier as f32; tokens];
            let (timestep_out, embedded_timestep) = self.adaln_single.forward(&timesteps)?;
            let prompt_timestep: Vec<f32> = match &self.prompt_adaln_single {
                Some(prompt_adaln) => {
                    let sigma_scaled = vec![sigma * t.timestep_scale_multiplier as f32; 1];
                    let (out, _) = prompt_adaln.forward(&sigma_scaled)?;
                    out
                }
                None => Vec::new(),
            };

            let cond_velocity = self.run_pass(
                &current,
                tokens,
                context,
                context_tokens,
                &timestep_out,
                &prompt_timestep,
                &embedded_timestep,
                &cos,
                &sin,
                None,
            )?;
            let cond = to_denoised(&current, &cond_velocity, sigma);

            let plain_guidance =
                guider.cfg_scale == 1.0 && guider.stg_scale == 0.0 && guider.modality_scale == 1.0;
            let denoised = if plain_guidance {
                cond
            } else {
                let uncond_text = match negative_context {
                    Some(neg) => {
                        let pass = self.run_pass(
                            &current,
                            tokens,
                            neg,
                            negative_context_tokens,
                            &timestep_out,
                            &prompt_timestep,
                            &embedded_timestep,
                            &cos,
                            &sin,
                            None,
                        )?;
                        to_denoised(&current, &pass, sigma)
                    }
                    None => cond.clone(),
                };
                let uncond_perturbed = if params.stg_scale == 0.0 {
                    cond.clone()
                } else {
                    let pass = self.run_pass(
                        &current,
                        tokens,
                        context,
                        context_tokens,
                        &timestep_out,
                        &prompt_timestep,
                        &embedded_timestep,
                        &cos,
                        &sin,
                        Some(params.stg_block),
                    )?;
                    to_denoised(&current, &pass, sigma)
                };
                calculate_guided_prediction(&cond, &uncond_text, &uncond_perturbed, &cond, &guider)
            };

            current = euler_step(&current, &denoised, &sigmas, step_index)?;
        }

        // Unpatchify [tokens, width] -> [channels, frames, mel_bins]
        // and patch the silence prior.
        let channels = self.config.audio.vae_channels;
        let mel_bins = self.config.audio.mel_bins;
        if width != channels * mel_bins {
            return Err(SpeechError::BadConfig {
                field: "audio_in_channels".into(),
                why: format!("expected channels * mel_bins = {channels} * {mel_bins}, got {width}"),
            });
        }
        let mut latents = vec![0.0f32; current.len()];
        for token in 0..tokens {
            for (index, value) in current[token * width..(token + 1) * width]
                .iter()
                .enumerate()
            {
                let channel = index / mel_bins;
                let mel = index % mel_bins;
                latents[(channel * tokens + token) * mel_bins + mel] = *value;
            }
        }
        patch_long_clip_silence_prior(&mut latents, channels, tokens, mel_bins);

        Ok(Generated {
            latents,
            channels,
            frames: tokens,
            mel_bins,
            sample_rate: self.config.audio.sample_rate,
        })
    }
}

/// `nn.LayerNorm(affine=False)`: mean/variance over the last axis with
/// the loaded weight applied as a plain multiply (the checkpoint keeps
/// the weight tensor; the reference module has none, so an all-ones
/// weight is the exact case and the fixture ships one).
fn layer_norm_affine_free(x: &mut [f32], rows: usize, width: usize, eps: f32) {
    for row in 0..rows {
        let base = row * width;
        let mean = x[base..base + width].iter().sum::<f32>() / width as f32;
        let var = x[base..base + width]
            .iter()
            .map(|v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            / width as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for value in &mut x[base..base + width] {
            *value = (*value - mean) * inv;
        }
    }
}

/// `generate` parameters (the kwargs the reference generate reads for
/// the zero-reference path).
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub cfg_scale: f32,
    pub stg_scale: f32,
    pub stg_block: usize,
    pub rescale_scale_auto: bool,
    pub rescale_scale: f32,
    pub modality_scale: f32,
    pub steps: usize,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            cfg_scale: 2.5,
            stg_scale: 1.5,
            stg_block: 29,
            rescale_scale_auto: true,
            rescale_scale: 0.0,
            modality_scale: 1.0,
            steps: 30,
        }
    }
}

/// One generation result: the mel latents the AudioVAE decoder would
/// consume (see the README for the decode refusal).
#[derive(Debug, Clone)]
pub struct Generated {
    /// `[channels, frames, mel_bins]` row-major.
    pub latents: Vec<f32>,
    pub channels: usize,
    pub frames: usize,
    pub mel_bins: usize,
    pub sample_rate: u32,
}

/// The waveform-decode entry the AudioVAE decoder and vocoder own.
/// Refused: both modules ship in the checkpoint's audio-components
/// file and are not ported yet (see the family README).
pub fn decode_waveform_refusal() -> SpeechError {
    SpeechError::Unsupported {
        why: "the DramaBox AudioVAE decoder and vocoder are not ported yet; this port ends at the mel latents (see the family README)".into(),
    }
}
