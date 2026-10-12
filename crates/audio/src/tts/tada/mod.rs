//! TADA: Text-Audio Dual Alignment TTS family.
//!
//! Port of `mlx_audio/tts/models/tada/` at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/tada):
//! a Llama backbone (the shared [crate::lm] llama flavor, tied head)
//! consumes token embeddings plus acoustic features, an acoustic-mask
//! embedding, and two gray-coded duration embeddings; per step a
//! flow-matching prediction head (the VibeVoice head layout) integrates
//! a velocity field over a logSNR-spaced time grid with per-dimension
//! classifier-free guidance (acoustic and duration halves guided at
//! separately scheduled scales), the duration halves decode through a
//! gray code to the next frame's inter-frame gaps, and the gathered
//! features de-normalize into codec latents.
//!
//! Scope of this port, matching the reference zero-shot single-prompt
//! path: batch 1, caller-owned text tokenization (the reference drives
//! a Llama-3.2 tokenizer; [`build_prompt`] assembles the chat prefix
//! from caller-supplied special ids and encoded strings). The
//! tada-codec waveform decoder (and the torch wav2vec2 aligner behind
//! voice cloning) is a separate downloadable model
//! (`HumeAI/tada-codec`), so generation ends at the de-normalized
//! acoustic features plus durations with a documented refusal, exactly
//! like the breeze_tts/pocket_tts patterns. Diffusion noise comes from
//! a [`DiffusionNoise`] provider (one standard-normal draw per step,
//! scaled by the noise temperature). The reference text sampler is
//! unreachable from `generate` with its defaults (logits are only
//! computed in the free-continuation mode); the port implements that
//! mode greedy over the pad-suppressed, sign-aware repetition-penalized
//! logits, reusing the shared [crate::tts::sampling] processors. See
//! [README.md](README.md) for the verification evidence and open
//! items.

use serde_json::Value;

use self::diffusion_head::DiffusionHead;
use self::gray_code::decode_gray_code_to_time;
use crate::lm::engine::{self, Backbone, Input};
use crate::lm::{llama, Weights};
use crate::music::minimax_music3::rng;
use crate::nn::Linear;
use crate::tts::sampling;
use crate::{ops, quant, Result, SpeechError};

mod diffusion_head;
mod gray_code;

#[cfg(test)]
mod tests;

/// `normalize_text` (`text_utils.py`): common Unicode punctuation to
/// ASCII, then the punctuation rewrite chain, then sentence
/// capitalization. Matches the reference replacement order exactly.
pub fn normalize_text(text: &str) -> String {
    let substitutions: &[(&str, &str)] = &[
        // Quotes.
        ("\u{201c}", "\""),
        ("\u{201d}", "\""),
        ("\u{201e}", "\""),
        ("\u{201f}", "\""),
        ("\u{2018}", "'"),
        ("\u{2019}", "'"),
        ("\u{201a}", "'"),
        ("\u{201b}", "'"),
        // Dashes.
        ("\u{2013}", "-"),
        ("\u{2014}", "-"),
        ("\u{2015}", "-"),
        ("\u{2010}", "-"),
        ("\u{2011}", "-"),
        // Ellipsis.
        ("\u{2026}", "..."),
        // Misc punctuation.
        ("\u{2039}", "<"),
        ("\u{203a}", ">"),
        ("\u{ab}", "<<"),
        ("\u{bb}", ">>"),
    ];
    let mut text = text.to_owned();
    for (from, to) in substitutions {
        text = text.replace(from, to);
    }
    let mut text = text
        .replace("; ", ". ")
        .replace('"', "")
        .replace(':', ",")
        .replace(['(', ')'], "")
        .replace("--", "-")
        .replace('-', ", ")
        .replace(",,", ",")
        .replace(" '", " ")
        .replace("' ", " ")
        .replace("  ", " ");
    // Remove spaces before sentence-ending punctuation (`\s+([.,?!])`).
    text = collapse_space_before_punctuation(&text);
    // Lowercase, then capitalize after sentence-ending punctuation
    // (`([.!?]\s*)(\w)`).
    text = text.to_lowercase();
    text = capitalize_after_sentence_end(&text);
    // Capitalize the very first character.
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => text,
    }
}

/// `re.sub(r"\s+([.,?!])", r"\1", text)`: any whitespace run directly
/// before `.` `,` `?` `!` collapses to nothing.
fn collapse_space_before_punctuation(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut pending_space = false;
    for &ch in &chars {
        if ch.is_whitespace() {
            pending_space = true;
            continue;
        }
        if pending_space {
            if !matches!(ch, '.' | ',' | '?' | '!') {
                out.push(' ');
            }
            pending_space = false;
        }
        out.push(ch);
    }
    // Trailing whitespace survives the regex (no punctuation follows).
    if pending_space {
        out.push(' ');
    }
    out
}

/// `re.sub(r"([.!?]\s*)(\w)", upper, text)`: the first word character
/// after sentence-ending punctuation uppercases.
fn capitalize_after_sentence_end(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut armed = false;
    for &ch in &chars {
        if armed && ch.is_alphanumeric() {
            for upper in ch.to_uppercase() {
                out.push(upper);
            }
            armed = false;
            continue;
        }
        armed = matches!(ch, '.' | '!' | '?');
        out.push(ch);
    }
    out
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// The model config (`ModelConfig`); only the fields the zero-shot
/// generation path reads are kept. The codec decoder/encoder fields
/// ride the config for the separate tada-codec download and are not
/// consumed here.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// The Llama backbone geometry, parsed with the llama flavor.
    pub decoder: crate::lm::DenseConfig,
    pub acoustic_dim: usize,
    pub num_time_classes: usize,
    pub shift_acoustic: usize,
    pub head_layers: usize,
    pub head_ffn_ratio: f32,
    pub bottleneck_dim: Option<usize>,
    pub acoustic_mean: f32,
    pub acoustic_std: f32,
    pub sample_rate: u32,
}

impl ModelConfig {
    pub fn from_json(value: &Value) -> Result<Self> {
        let i = |key: &str, default: usize| {
            value
                .get(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(default)
        };
        // The reference rotary never applies rope_scaling at inference.
        let mut decoder = llama::from_json(value)?;
        decoder.rope_scaling = None;
        let num_time_classes = i("num_time_classes", 256);
        Ok(Self {
            decoder,
            acoustic_dim: i("acoustic_dim", 512),
            num_time_classes: num_time_classes.max(2),
            shift_acoustic: i("shift_acoustic", 5),
            head_layers: i("head_layers", 6),
            head_ffn_ratio: value
                .get("head_ffn_ratio")
                .and_then(Value::as_f64)
                .unwrap_or(4.0) as f32,
            bottleneck_dim: value
                .get("bottleneck_dim")
                .and_then(Value::as_u64)
                .map(|n| n as usize),
            acoustic_mean: value
                .get("acoustic_mean")
                .and_then(Value::as_f64)
                .unwrap_or(0.0) as f32,
            acoustic_std: value
                .get("acoustic_std")
                .and_then(Value::as_f64)
                .unwrap_or(1.5) as f32,
            sample_rate: value
                .get("sample_rate")
                .and_then(Value::as_u64)
                .unwrap_or(24000) as u32,
        })
    }
}

// ---------------------------------------------------------------------------
// Noise provider
// ---------------------------------------------------------------------------

/// The flow-matching noise source: one standard-normal draw of
/// `[rows, cols]` per call, in reference draw order (one per
/// generation step, before the flow integration).
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
// Prompt construction
// ---------------------------------------------------------------------------

/// The special token ids the chat prefix needs, resolved by the caller
/// from the checkpoint tokenizer.
#[derive(Debug, Clone, Copy)]
pub struct SpecialIds {
    pub bos: i32,
    pub eot: i32,
    pub start_header: i32,
    pub end_header: i32,
}

/// The assembled prompt: the token ids and the prefix length the skip
/// math needs (`prefix_text_tokens`, without the BOS, matching
/// PyTorch).
#[derive(Debug, Clone)]
pub struct Prompt {
    pub ids: Vec<i32>,
    pub prefix_len: usize,
}

/// Builds the reference input ids: `BOS + <|start_header_id|> system
/// <|end_header_id|> <|eot_id|> <|start_header_id|> assistant
/// <|end_header_id|> + text + <|eot_id|> * shift_acoustic`. `system`
/// and `assistant` are the caller-encoded header words; `text` is the
/// caller-encoded `normalize_text(text)` (the reference prompt text
/// splice, prompt text then user text). `prefix_len` counts the
/// structural prefix WITHOUT the BOS, matching PyTorch.
pub fn build_prompt(
    ids: &SpecialIds,
    system_tokens: &[i32],
    assistant_tokens: &[i32],
    text_tokens: &[i32],
    shift_acoustic: usize,
) -> Prompt {
    let mut all = Vec::with_capacity(
        8 + system_tokens.len() + assistant_tokens.len() + text_tokens.len() + shift_acoustic,
    );
    all.push(ids.bos);
    all.push(ids.start_header);
    all.extend_from_slice(system_tokens);
    all.push(ids.end_header);
    all.push(ids.eot);
    all.push(ids.start_header);
    all.extend_from_slice(assistant_tokens);
    all.push(ids.end_header);
    let prefix_len = all.len() - 1;
    all.extend_from_slice(text_tokens);
    for _ in 0..shift_acoustic {
        all.push(ids.eot);
    }
    Prompt {
        ids: all,
        prefix_len,
    }
}

// ---------------------------------------------------------------------------
// The loaded model
// ---------------------------------------------------------------------------

/// Reads an optional checkpoint scalar-backed field.
fn optional_linear(
    weights: &Weights,
    name: &str,
    input: usize,
    output: usize,
) -> Result<Option<Linear>> {
    if !weights.contains_absolute(name) {
        return Ok(None);
    }
    Ok(Some(weights.linear_absolute(
        name,
        input,
        output,
        crate::lm::Bias::Required,
    )?))
}

/// The loaded TADA model.
pub struct Model {
    pub config: ModelConfig,
    /// The Llama backbone under `model.` with its tied embed head.
    backbone: Backbone,
    acoustic_proj: Linear,
    time_start_embed: Vec<f32>,
    time_end_embed: Vec<f32>,
    acoustic_mask_emb: Vec<f32>,
    bottleneck_proj: Option<Linear>,
    prediction_head: DiffusionHead,
    /// Gray-code bit blocks per duration (`ceil(log2(num_time_classes))`).
    num_time_bits: usize,
}

impl Model {
    /// Opens a checkpoint directory (`config.json` + safetensors). The
    /// tada-codec decoder is a separate download and is not loaded; see
    /// the README.
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        let parsed = ModelConfig::from_json(&config_value)?;
        let weights = Weights::open(dir)?;
        Self::load_weights(&weights, parsed)
    }

    fn load_weights(weights: &Weights, config: ModelConfig) -> Result<Self> {
        let hidden = config.decoder.hidden_size;
        // Weights::open auto-detects the `model.` prefix from the embed
        // table; the backbone must exist under it.
        if !weights.contains("embed_tokens") {
            return Err(SpeechError::Tensor {
                name: "model.embed_tokens.weight".into(),
                why: "checkpoint carries no backbone embed table".into(),
            });
        }
        let backbone = Backbone::load(weights, config.decoder.clone())?;
        let acoustic_proj = weights.linear_absolute(
            "acoustic_proj",
            config.acoustic_dim,
            hidden,
            crate::lm::Bias::Required,
        )?;
        let time_start_embed = weights.tensor_absolute(
            "time_start_embed.weight",
            &[config.num_time_classes, hidden],
        )?;
        let time_end_embed =
            weights.tensor_absolute("time_end_embed.weight", &[config.num_time_classes, hidden])?;
        let acoustic_mask_emb =
            weights.tensor_absolute("acoustic_mask_emb.weight", &[2, hidden])?;
        let bottleneck_proj = match config.bottleneck_dim {
            Some(width) => optional_linear(weights, "bottleneck_proj", hidden, width)?,
            None => None,
        };
        let num_time_bits = (config.num_time_classes as f64).log2().ceil() as usize;
        let latent_size = config.acoustic_dim + 2 * num_time_bits;
        let head_hidden = config.bottleneck_dim.unwrap_or(hidden);
        let prediction_head = DiffusionHead::load(
            weights.shards(),
            "prediction_head",
            head_hidden,
            config.head_layers,
            config.head_ffn_ratio,
            config.decoder.rms_norm_eps,
            latent_size,
        )?;
        Ok(Self {
            config,
            backbone,
            acoustic_proj,
            time_start_embed,
            time_end_embed,
            acoustic_mask_emb,
            bottleneck_proj,
            prediction_head,
            num_time_bits,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    fn time_embed(&self, table: &[f32], id: i32, rows: usize) -> Result<Vec<f32>> {
        let hidden = self.config.decoder.hidden_size;
        let row = usize::try_from(id).map_err(|_| SpeechError::Input {
            why: "negative time id".into(),
        })?;
        if row >= self.config.num_time_classes {
            return Err(SpeechError::Input {
                why: format!(
                    "time id {id} outside the {}-class table",
                    self.config.num_time_classes
                ),
            });
        }
        let source = &table[row * hidden..(row + 1) * hidden];
        let mut out = Vec::with_capacity(rows * hidden);
        for _ in 0..rows {
            out.extend_from_slice(source);
        }
        Ok(out)
    }

    /// `forward_one_step`: token, acoustic, mask, and the two duration
    /// embeddings sum into the backbone input. `rows` covers the CFG
    /// pair; all inputs are row-major `[rows, width]`.
    #[allow(clippy::too_many_arguments)]
    fn forward_one_step(
        &mut self,
        input_ids: &[i32],
        acoustic_features: &[f32],
        acoustic_masks: &[i32],
        time_before: &[i32],
        time_after: &[i32],
        caches: Option<&mut crate::lm::LayerCaches>,
    ) -> Result<Vec<f32>> {
        let rows = input_ids.len();
        let hidden = self.config.decoder.hidden_size;
        let mut embeds = engine::lookup_embeddings(
            self.backbone.embed_table(),
            self.config.decoder.vocab_size,
            hidden,
            input_ids,
        )?;
        let acoustic = self.acoustic_proj.forward(acoustic_features, rows);
        for (slot, value) in embeds.iter_mut().zip(&acoustic) {
            *slot += value;
        }
        for (row, &mask) in acoustic_masks.iter().enumerate() {
            let mask_row = usize::try_from(mask).map_err(|_| SpeechError::Input {
                why: "negative mask id".into(),
            })?;
            if mask_row >= 2 {
                return Err(SpeechError::Input {
                    why: format!("mask id {mask} outside the 2-row table"),
                });
            }
            let source = &self.acoustic_mask_emb[mask_row * hidden..(mask_row + 1) * hidden];
            for (col, slot) in embeds[row * hidden..(row + 1) * hidden]
                .iter_mut()
                .enumerate()
            {
                *slot += source[col];
            }
        }
        for (row, &id) in time_before.iter().enumerate() {
            let embed = self.time_embed(&self.time_start_embed, id, 1)?;
            for (col, slot) in embeds[row * hidden..(row + 1) * hidden]
                .iter_mut()
                .enumerate()
            {
                *slot += embed[col];
            }
        }
        for (row, &id) in time_after.iter().enumerate() {
            let embed = self.time_embed(&self.time_end_embed, id, 1)?;
            for (col, slot) in embeds[row * hidden..(row + 1) * hidden]
                .iter_mut()
                .enumerate()
            {
                *slot += embed[col];
            }
        }
        self.backbone
            .forward_hidden(Input::Embeddings(&embeds), caches)
    }

    /// `_scheduled_cfg`: the guidance scale at time `t`.
    fn scheduled_cfg(base_scale: f32, t: f32, schedule: &str) -> f32 {
        if schedule == "constant" || base_scale == 1.0 {
            return base_scale;
        }
        if schedule == "linear" {
            return 1.0 + (base_scale - 1.0) * (1.0 - t);
        }
        if schedule == "cosine" {
            return 1.0 + (base_scale - 1.0) * 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
        }
        base_scale
    }

    /// `_build_time_schedule`: the Euler grid over `num_steps`
    /// intervals, endpoints included.
    fn build_time_schedule(num_steps: usize, schedule: &str) -> Vec<f32> {
        let points = num_steps + 1;
        match schedule {
            "cosine" => (0..points)
                .map(|i| {
                    let u = i as f32 / num_steps as f32;
                    0.5 * (1.0 - (std::f32::consts::PI * u).cos())
                })
                .collect(),
            "logsnr" => {
                let mut t: Vec<f32> = (0..points)
                    .map(|i| {
                        let log_snr = 5.0 + (i as f32 / num_steps as f32) * (-10.0);
                        // sigmoid(-log_snr / 2)
                        1.0 / (1.0 + (log_snr / 2.0).exp())
                    })
                    .collect();
                // Exact endpoints, matching the reference concat.
                t[0] = 0.0;
                t[num_steps] = 1.0;
                t
            }
            _ => (0..points).map(|i| i as f32 / num_steps as f32).collect(),
        }
    }

    /// `_compute_velocity`: the head over the (optionally doubled)
    /// batch with the two-half guidance.
    fn compute_velocity(
        &self,
        speech: &[f32],
        t: f32,
        cond: &[f32],
        neg_cond: &[f32],
        acoustic_cfg: f32,
        duration_cfg: f32,
    ) -> Result<Vec<f32>> {
        let latent = speech.len();
        if acoustic_cfg == 1.0 {
            let mut bottleneck = cond.to_vec();
            if let Some(proj) = &self.bottleneck_proj {
                bottleneck = proj.forward(&bottleneck, 1);
            }
            return self.prediction_head.forward(speech, &[t], &bottleneck, 1);
        }
        let mut combined_speech = speech.to_vec();
        combined_speech.extend_from_slice(speech);
        let mut combined_cond = cond.to_vec();
        combined_cond.extend_from_slice(neg_cond);
        if let Some(proj) = &self.bottleneck_proj {
            combined_cond = proj.forward(&combined_cond, 2);
        }
        let velocity =
            self.prediction_head
                .forward(&combined_speech, &[t, t], &combined_cond, 2)?;
        // The split point is the acoustic half of the LATENT (the
        // duration bits follow it); the batched rows are
        // [pos_row, neg_row].
        let acoustic = self.config.acoustic_dim;
        let mut guided = vec![0.0f32; latent];
        for (index, slot) in guided.iter_mut().enumerate() {
            let (pos, neg) = (velocity[index], velocity[latent + index]);
            let scale = if index < acoustic {
                acoustic_cfg
            } else {
                duration_cfg
            };
            *slot = neg + scale * (pos - neg);
        }
        Ok(guided)
    }

    /// `_solve_flow_matching`: Euler integration over the time grid.
    #[allow(clippy::too_many_arguments)]
    fn solve_flow_matching(
        &self,
        mut speech: Vec<f32>,
        cond: &[f32],
        neg_cond: &[f32],
        num_steps: usize,
        acoustic_cfg_scale: f32,
        duration_cfg_scale: f32,
        cfg_schedule: &str,
        time_schedule: &str,
    ) -> Result<Vec<f32>> {
        let t_span = Self::build_time_schedule(num_steps, time_schedule);
        let mut t_curr = t_span[0];
        for &t_next in &t_span[1..] {
            let dt = t_next - t_curr;
            let a_cfg = Self::scheduled_cfg(acoustic_cfg_scale, t_curr, cfg_schedule);
            let d_cfg = Self::scheduled_cfg(duration_cfg_scale, t_curr, cfg_schedule);
            let velocity = self.compute_velocity(&speech, t_curr, cond, neg_cond, a_cfg, d_cfg)?;
            for (slot, &v) in speech.iter_mut().zip(&velocity) {
                *slot += dt * v;
            }
            t_curr = t_next;
        }
        Ok(speech)
    }

    /// The reference zero-shot `generate`/`_generate_loop` for batch 1:
    /// the per-step forward, flow-matched feature prediction, and gray
    /// duration decode, ending at the de-normalized, skip-trimmed
    /// acoustic features and durations (the tada-codec waveform decode
    /// is a separate download and is refused; see the README).
    /// `prompt.ids` is the caller-built prompt ([`build_prompt`]).
    pub fn generate(
        &mut self,
        prompt: &Prompt,
        params: &GenerateParams,
        noise: &mut dyn DiffusionNoise,
    ) -> Result<Generated> {
        let shift = self.config.shift_acoustic;
        let mut ids = prompt.ids.clone();
        if params.num_extra_steps > 0 {
            let keep = ids.len().saturating_sub(shift);
            ids.truncate(keep);
        }
        let num_steps = (ids.len() + params.num_extra_steps).min(params.max_tokens);

        let hidden = self.config.decoder.hidden_size;
        let acoustic_dim = self.config.acoustic_dim;
        let total_dim = acoustic_dim + 2 * self.num_time_bits;

        let mut caches = crate::lm::make_prompt_cache(self.config.decoder.num_hidden_layers);
        // The CFG pair runs as two independent single-batch forwards
        // (the reference doubles the batch axis, which the shared
        // single-batch engine expresses as a second cache).
        let mut neg_caches = if params.acoustic_cfg_scale != 1.0 {
            Some(crate::lm::make_prompt_cache(
                self.config.decoder.num_hidden_layers,
            ))
        } else {
            None
        };
        // Current-step state: zeros until the first feature lands.
        let mut acoustic_features = vec![0.0f32; acoustic_dim];
        let mut acoustic_mask = 0i32;
        let mut time_before = 0i32;
        let mut time_after = 0i32;
        let mut neg_cond = vec![0.0f32; hidden];

        let mut all_features: Vec<Vec<f32>> = Vec::new();
        let mut all_time: Vec<i32> = Vec::new();
        let mut last_time_before: Option<i32> = None;

        for step in 0..num_steps {
            let token = if step < ids.len() {
                ids[step]
            } else {
                *ids.last().unwrap_or(&0)
            };
            let need_logits = params.num_extra_steps > 0 && step + 1 >= ids.len();

            let cond = if params.acoustic_cfg_scale != 1.0 {
                // Negative slice: structural tokens stay, text becomes
                // the pad id.
                let neg_token = if token == params.structural.start_header
                    || token == params.structural.end_header
                    || token == params.structural.eot
                {
                    token
                } else {
                    params.pad_token_id
                };
                let features = acoustic_features.clone();
                let neg_caches = neg_caches
                    .as_mut()
                    .expect("cfg branch without a negative cache");
                let neg = self.forward_one_step(
                    &[neg_token],
                    &features,
                    &[acoustic_mask],
                    &[time_before],
                    &[time_after],
                    Some(neg_caches),
                )?;
                neg_cond = neg;
                self.forward_one_step(
                    &[token],
                    &features,
                    &[acoustic_mask],
                    &[time_before],
                    &[time_after],
                    Some(&mut caches),
                )?
            } else {
                let features = acoustic_features.clone();
                self.forward_one_step(
                    &[token],
                    &features,
                    &[acoustic_mask],
                    &[time_before],
                    &[time_after],
                    Some(&mut caches),
                )?
            };

            // Flow matching from scaled Gaussian noise.
            let mut speech = noise.normal(1, total_dim);
            for slot in &mut speech {
                *slot *= params.noise_temperature;
            }
            let speech = self.solve_flow_matching(
                speech,
                &cond,
                &neg_cond,
                params.num_flow_matching_steps,
                params.acoustic_cfg_scale,
                params.duration_cfg_scale,
                &params.cfg_schedule,
                &params.time_schedule,
            )?;

            // Gray-code durations from the tail of the sample.
            let time_gray = &speech[acoustic_dim..];
            let predicted_before =
                decode_gray_code_to_time(&time_gray[..self.num_time_bits], self.num_time_bits)[0];
            let predicted_after =
                decode_gray_code_to_time(&time_gray[self.num_time_bits..], self.num_time_bits)[0];

            // Free continuation beyond the prompt ids (greedy over the
            // filtered logits; the reference sampler is unreachable
            // from `generate` defaults, see the module docs). The tied
            // head is `hidden @ embed_tokens.T`.
            if params.num_extra_steps > 0 && need_logits {
                let mut row = ops::linear(
                    &cond,
                    self.backbone.embed_table(),
                    None,
                    1,
                    self.config.decoder.hidden_size,
                    self.config.decoder.vocab_size,
                );
                if params.pad_token_id >= 0 && (params.pad_token_id as usize) < row.len() {
                    row[params.pad_token_id as usize] = f32::NEG_INFINITY;
                }
                if params.repetition_penalty != 1.0 {
                    sampling::apply_repetition_penalty(
                        &mut row,
                        &ids,
                        params.repetition_penalty,
                        20,
                    )?;
                }
                for value in &mut row {
                    *value /= params.temperature;
                }
                // The reference filter order: top-k, then top-p.
                if params.top_k > 0 {
                    sampling::apply_top_k(&mut row, params.top_k)?;
                }
                if params.top_p > 0.0 && params.top_p < 1.0 {
                    sampling::apply_top_p(&mut row, params.top_p);
                }
                let next = sampling::argmax(&row) as i32;
                ids.push(next);
                if params.eos_ids.contains(&next) {
                    break;
                }
            }

            // Update the state from step `shift` on.
            if step >= shift {
                acoustic_features.copy_from_slice(&speech[..acoustic_dim]);
                acoustic_mask = 1;
                all_features.push(acoustic_features.clone());
                time_before = predicted_before;
                time_after = predicted_after;
                all_time.push(time_before);
                last_time_before = Some(time_before);
            }
        }

        if all_features.is_empty() {
            return Ok(Generated {
                acoustic_features: Vec::new(),
                feature_frames: 0,
                time_before: Vec::new(),
                sample_rate: self.config.sample_rate,
            });
        }

        // Trailing duration repeat.
        if let Some(last) = last_time_before {
            all_time.push(last);
        }

        // De-normalize.
        let mut features: Vec<f32> = all_features.concat();
        for value in &mut features {
            *value = *value * self.config.acoustic_std + self.config.acoustic_mean;
        }

        // Zero-shot skip: the structural prefix features.
        let zero_shot_prompt_tokens = prompt
            .prefix_len
            .saturating_sub(params.num_transition_steps);
        let skip = zero_shot_prompt_tokens + params.num_transition_steps - 1;
        let frames = features.len() / acoustic_dim;
        let (features, time_before) = if skip > 0 && skip < frames {
            (
                features[skip * acoustic_dim..].to_vec(),
                all_time[skip..].to_vec(),
            )
        } else {
            (features, all_time)
        };

        Ok(Generated {
            feature_frames: features.len() / acoustic_dim,
            acoustic_features: features,
            time_before,
            sample_rate: self.config.sample_rate,
        })
    }
}

/// `generate` parameters (the reference `generate` kwargs the zero-shot
/// path reads).
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub acoustic_cfg_scale: f32,
    pub duration_cfg_scale: f32,
    pub cfg_schedule: String,
    pub noise_temperature: f32,
    pub num_flow_matching_steps: usize,
    pub time_schedule: String,
    pub num_transition_steps: usize,
    pub max_tokens: usize,
    /// Free-continuation steps beyond the prompt (the reference
    /// `num_extra_steps`; 0 in the default path). Continuation samples
    /// greedily.
    pub num_extra_steps: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    /// The ids the negative slice keeps (headers and eot).
    pub structural: SpecialIds,
    /// The pad id the negative slice and the sampler use; -1 disables
    /// pad suppression.
    pub pad_token_id: i32,
    /// Stop ids for the continuation mode.
    pub eos_ids: Vec<i32>,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            acoustic_cfg_scale: 1.6,
            duration_cfg_scale: 1.0,
            cfg_schedule: "cosine".to_owned(),
            noise_temperature: 0.9,
            num_flow_matching_steps: 20,
            time_schedule: "logsnr".to_owned(),
            num_transition_steps: 5,
            max_tokens: 1024,
            num_extra_steps: 0,
            temperature: 0.6,
            top_k: 0,
            top_p: 0.9,
            repetition_penalty: 1.1,
            structural: SpecialIds {
                bos: 1,
                eot: 128009,
                start_header: 128006,
                end_header: 128007,
            },
            pad_token_id: 128004,
            eos_ids: vec![128001, 128009],
        }
    }
}

/// One generation result: the codec latents and durations the
/// tada-codec decoder would expand (see the README for the refusal).
#[derive(Debug, Clone)]
pub struct Generated {
    /// De-normalized acoustic features `[frames, acoustic_dim]`.
    pub acoustic_features: Vec<f32>,
    pub feature_frames: usize,
    /// The per-frame inter-frame gap (`time_before`), one per frame
    /// after the skip, including the trailing repeat. In units of 50 Hz
    /// frames (`sample_rate / 50` samples each in the decoder).
    pub time_before: Vec<i32>,
    pub sample_rate: u32,
}

/// The waveform-decode entry the tada-codec decoder owns. Refused: the
/// decoder weights ship in the separate `HumeAI/tada-codec` download,
/// not the main checkpoint.
pub fn decode_waveform_refusal() -> SpeechError {
    SpeechError::Unsupported {
        why: "the TADA waveform decoder ships in the separate HumeAI/tada-codec download, not the main checkpoint; this port ends at the de-normalized acoustic features and durations (see the family README)".into(),
    }
}
