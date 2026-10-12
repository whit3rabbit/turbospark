//! OmniVoice: zero-shot multilingual TTS over a bidirectional Qwen3
//! backbone with iterative masked-diffusion decoding of 8 HiggsAudioV2
//! codebooks (24 kHz, `sample_rate / 960` tokens per second).
//!
//! Port of `mlx_audio/tts/models/omnivoice/` at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/omnivoice),
//! single batch, portable f32.
//!
//! The backbone is the shared [`crate::lm`] qwen3 engine with the causal
//! mask disabled ([`lm::Backbone::set_bidirectional`]); the audio side
//! is 8 independent `[V, hidden]` embedding tables summed per frame and
//! 8 independent prediction heads stacked per position. The iterative
//! unmasking schedule, the CFG combine over bidirectional conditional
//! and unconditional passes, the mask-token suppression, the
//! layer-penalized confidence, the Gumbel position noise, and the
//! argsort-rank reveal all follow `generation.py` exactly.
//!
//! Caller-owned intake: the text tokenizer (a HF `tokenizer.json`) and
//! the reference-audio encode (the Higgs codec encode path) are caller
//! concerns; [`Model::generate`] takes the style/text token ids, an
//! optional reference-code block, and the per-step Gumbel uniforms (a
//! seeded threefry stream stands in when the caller has no recorded
//! draws). Decoding ends at audio tokens plus the shared
//! [`crate::codec::higgs_audio`] decode.

use std::path::Path;

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::lm::{self, Input, Weights};
use crate::music::minimax_music3::rng;
use crate::nn::load_tensor;
use crate::tts::sampling::{argmax, log_softmax};
use crate::{ops, Result, SpeechError};

/// Combine the reference transcript and the text to synthesize
/// (`_combine_text`): strip both, join with one space, drop `\r`/`\n`,
/// collapse space/tab runs, and remove whitespace that touches a CJK
/// character on either side.
pub fn combine_text(text: &str, ref_text: Option<&str>) -> String {
    let joined = match ref_text {
        Some(ref_text) => format!("{} {}", ref_text.trim(), text.trim()),
        None => text.trim().to_owned(),
    };
    // Pass 1: drop newlines and collapse space/tab runs to one space.
    let mut collapsed = String::with_capacity(joined.len());
    let mut pending_space = false;
    for c in joined.chars() {
        if c == '\r' || c == '\n' {
            continue;
        }
        if c == ' ' || c == '\t' {
            pending_space = true;
            continue;
        }
        if pending_space {
            collapsed.push(' ');
            pending_space = false;
        }
        collapsed.push(c);
    }
    if pending_space {
        collapsed.push(' ');
    }
    // Pass 2: drop any space adjacent to a CJK character.
    let chars: Vec<char> = collapsed.chars().collect();
    let is_cjk = |c: char| ('\u{4e00}'..='\u{9fff}').contains(&c);
    let mut out = String::with_capacity(collapsed.len());
    for (i, &c) in chars.iter().enumerate() {
        if c == ' '
            && ((i > 0 && is_cjk(chars[i - 1])) || (i + 1 < chars.len() && is_cjk(chars[i + 1])))
        {
            continue;
        }
        out.push(c);
    }
    out
}

/// The style-prefix token string (`_tokenize_style_and_text`, string
/// half): the optional `<|denoise|>` tag for voice cloning, the language
/// tag block, and the instruction block. Tokenization is caller-owned.
pub fn build_style_text(language: &str, instruct: &str, denoise: bool) -> String {
    let mut style = String::new();
    if denoise {
        style.push_str("<|denoise|>");
    }
    style.push_str(&format!("<|lang_start|>{language}<|lang_end|>"));
    style.push_str(&format!("<|instruct_start|>{instruct}<|instruct_end|>"));
    style
}

/// The wrapped text string over the combined text. Tokenization is
/// caller-owned.
pub fn build_wrapped_text(text: &str, ref_text: Option<&str>) -> String {
    format!("<|text_start|>{}<|text_end|>", combine_text(text, ref_text))
}

/// The pipeline configuration (the checkpoint `config.json`).
#[derive(Debug, Clone)]
pub struct OmniVoiceConfig {
    pub model_type: String,
    /// 1025: 1024 codec tokens plus the mask token.
    pub audio_vocab_size: usize,
    pub audio_mask_id: usize,
    pub num_audio_codebook: usize,
    pub sample_rate: u32,
    /// The raw Qwen3 dict the backbone is parsed from.
    pub llm_config: Value,
}

impl Default for OmniVoiceConfig {
    fn default() -> Self {
        OmniVoiceConfig {
            model_type: "omnivoice".into(),
            audio_vocab_size: 1025,
            audio_mask_id: 1024,
            num_audio_codebook: 8,
            sample_rate: 24000,
            llm_config: Value::Null,
        }
    }
}

impl OmniVoiceConfig {
    pub fn from_json(value: &Value) -> Self {
        let mut config = Self::default();
        if let Some(v) = value.get("model_type").and_then(Value::as_str) {
            config.model_type = v.to_owned();
        }
        let field = |key: &str| value.get(key).and_then(Value::as_u64);
        if let Some(v) = field("audio_vocab_size") {
            config.audio_vocab_size = v as usize;
        }
        if let Some(v) = field("audio_mask_id") {
            config.audio_mask_id = v as usize;
        }
        if let Some(v) = field("num_audio_codebook") {
            config.num_audio_codebook = v as usize;
        }
        if let Some(v) = field("sample_rate") {
            config.sample_rate = v as u32;
        }
        if let Some(v) = value.get("llm_config") {
            config.llm_config = v.clone();
        }
        config
    }
}

/// The loaded model: the bidirectional backbone plus the per-codebook
/// audio embeddings and prediction heads.
pub struct Model {
    pub config: OmniVoiceConfig,
    backbone: lm::Backbone,
    /// `[codebook][vocab, hidden]` tables.
    audio_embeddings: Vec<Vec<f32>>,
    /// `[codebook][vocab, hidden]` head weights (bias-free).
    audio_heads: Vec<Vec<f32>>,
    hidden: usize,
}

impl Model {
    /// Opens the checkpoint directory (`config.json` plus
    /// `model.safetensors` in the converted `backbone.*` /
    /// `audio_embeddings.{i}` / `audio_heads.{i}` naming).
    pub fn open(dir: &Path) -> Result<Self> {
        let config_value = crate::quant::read_json(&dir.join("config.json"))?;
        Self::open_with_config(dir, config_value)
    }

    /// [`Self::open`] with a caller-supplied config document.
    pub fn open_with_config(dir: &Path, config_value: Value) -> Result<Self> {
        let config = OmniVoiceConfig::from_json(&config_value);
        let model_path = dir.join("model.safetensors");
        let file = SafetensorsFile::open(&model_path).map_err(SpeechError::from)?;
        let file = &file;
        let hidden = config
            .llm_config
            .get("hidden_size")
            .and_then(Value::as_u64)
            .ok_or_else(|| SpeechError::BadConfig {
                field: "llm_config.hidden_size".into(),
                why: "omnivoice requires a qwen3 llm_config".into(),
            })? as usize;
        let codebooks = config.num_audio_codebook;
        let vocab = config.audio_vocab_size;
        let mut audio_embeddings = Vec::with_capacity(codebooks);
        let mut audio_heads = Vec::with_capacity(codebooks);
        for i in 0..codebooks {
            audio_embeddings.push(load_tensor(
                file,
                &format!("audio_embeddings.{i}.weight"),
                &[vocab, hidden],
            )?);
            audio_heads.push(load_tensor(
                file,
                &format!("audio_heads.{i}.weight"),
                &[vocab, hidden],
            )?);
        }
        let weights = Weights::open_single(&model_path)?.with_prefix("backbone.");
        let dense_config = lm::qwen3::from_json(&config.llm_config)?;
        let mut backbone = lm::Backbone::load(&weights, dense_config)?;
        // The reference disables the causal mask on the whole stack.
        backbone.set_bidirectional(true);
        Ok(Model {
            config,
            backbone,
            audio_embeddings,
            audio_heads,
            hidden,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Tokens per frame: the codec runs 960 samples per frame.
    pub fn tokens_per_sec(&self) -> f32 {
        self.config.sample_rate as f32 / 960.0
    }

    /// The target frame count for a duration (`ceil(duration * tps)`).
    pub fn target_len_for(&self, duration_s: f32) -> usize {
        (duration_s * self.tokens_per_sec()).ceil() as usize
    }

    /// `_prepare_inference_inputs`: the `[L, C]` id grid over
    /// `[style | text | ref | target]` and the `[L]` audio mask marking
    /// the positions whose backbone embeddings are the summed audio
    /// tables instead of the text embedding.
    pub fn prepare_inference_inputs(
        &self,
        style_ids: &[i32],
        text_ids: &[i32],
        target_len: usize,
        ref_tokens: Option<&[Vec<i32>]>,
    ) -> Result<(Vec<i32>, Vec<bool>)> {
        let codebooks = self.config.num_audio_codebook;
        let mask_id = self.config.audio_mask_id as i32;
        let n_text = style_ids.len() + text_ids.len();
        let t_ref = ref_tokens.map_or(0, |r| r.len());
        let len = n_text + t_ref + target_len;
        let mut input_ids = vec![0i32; len * codebooks];
        let mut audio_mask = vec![false; len];
        let mut row = 0usize;
        // Text ids broadcast across every codebook column.
        for &id in style_ids.iter().chain(text_ids) {
            for c in 0..codebooks {
                input_ids[row * codebooks + c] = id;
            }
            row += 1;
        }
        if let Some(ref_tokens) = ref_tokens {
            for frame in ref_tokens {
                // A short frame would be zero-filled and a long one cut,
                // both corrupting the reference without an error.
                if frame.len() != codebooks {
                    return Err(SpeechError::Input {
                        why: format!(
                            "reference audio frame has {} codes, expected {codebooks}",
                            frame.len()
                        ),
                    });
                }
                for (c, &id) in frame.iter().enumerate() {
                    input_ids[row * codebooks + c] = id;
                }
                audio_mask[row] = true;
                row += 1;
            }
        }
        for r in row..len {
            for c in 0..codebooks {
                input_ids[r * codebooks + c] = mask_id;
            }
            audio_mask[r] = true;
        }
        Ok((input_ids, audio_mask))
    }

    /// `_prepare_embed_inputs` + the backbone pass + the 8 stacked
    /// prediction heads: returns the logits `[L, C, V]` row-major with
    /// the vocabulary innermost.
    pub fn forward(&mut self, input_ids: &[i32], audio_mask: &[bool]) -> Result<Vec<f32>> {
        let codebooks = self.config.num_audio_codebook;
        let len = audio_mask.len();
        if input_ids.len() != len * codebooks {
            return Err(SpeechError::Input {
                why: format!(
                    "input ids {} do not cover {} rows x {codebooks} codebooks",
                    input_ids.len(),
                    len
                ),
            });
        }
        let hidden = self.hidden;
        let vocab = self.config.audio_vocab_size;
        let embed_table = self.backbone.embed_table();
        let mut embeds = vec![0.0f32; len * hidden];
        for row in 0..len {
            let dst = &mut embeds[row * hidden..(row + 1) * hidden];
            if audio_mask[row] {
                // Audio rows: the sum of the per-codebook embeddings.
                for c in 0..codebooks {
                    let id = input_ids[row * codebooks + c];
                    let index = self.audio_row(id, vocab)?;
                    let table = &self.audio_embeddings[c];
                    let base = index * hidden;
                    for (d, slot) in dst.iter_mut().enumerate() {
                        *slot += table[base + d];
                    }
                }
            } else {
                // Text rows: the backbone embedding of the first column,
                // bounded by the backbone vocabulary.
                let id = input_ids[row * codebooks];
                let index = self.audio_row(id, self.backbone.config.vocab_size)?;
                let base = index * hidden;
                dst.copy_from_slice(&embed_table[base..base + hidden]);
            }
        }
        let states = self
            .backbone
            .forward_hidden(Input::Embeddings(&embeds), None)?;
        // Stack the 8 head projections: [L, C, V] with V innermost.
        let mut logits = vec![0.0f32; len * codebooks * vocab];
        for (c, head) in self.audio_heads.iter().enumerate() {
            let per_head = ops::linear(&states, head, None, len, hidden, vocab);
            for row in 0..len {
                let src = row * vocab;
                let dst = (row * codebooks + c) * vocab;
                logits[dst..dst + vocab].copy_from_slice(&per_head[src..src + vocab]);
            }
        }
        Ok(logits)
    }

    fn audio_row(&self, id: i32, vocab: usize) -> Result<usize> {
        let index = usize::try_from(id).map_err(|_| SpeechError::Input {
            why: format!("negative token id {id}"),
        })?;
        if index >= vocab {
            return Err(SpeechError::Input {
                why: format!("token id {id} outside vocab {vocab}"),
            });
        }
        Ok(index)
    }

    /// `iterative_unmask`: the masked-diffusion decode. `draws` carries
    /// one uniform block per step (the fixture replays the recorded
    /// `mx.random` stream): with `class_temperature > 0` each step
    /// consumes `T * C * V` uniforms for the token draw before the
    /// `T * C` position uniforms; at temperature 0 only the position
    /// block. `seed` derives a crate-local threefry stream when `draws`
    /// is `None` (documented deviation: not bit-reproducible against
    /// `mx.random`). Returns the target tokens `[T, C]` row-major with
    /// any surviving mask replaced by 0.
    #[allow(clippy::too_many_arguments)]
    pub fn iterative_unmask(
        &mut self,
        cond_input_ids: &[i32],
        cond_audio_mask: &[bool],
        target_len: usize,
        num_steps: usize,
        guidance_scale: f32,
        class_temperature: f32,
        position_temperature: f32,
        layer_penalty_factor: f32,
        t_shift: f32,
        draws: Option<&[Vec<f32>]>,
        seed: u64,
    ) -> Result<Vec<i32>> {
        // Zero steps would skip the loop and return the all-mask target
        // as zeros, a silent non-result.
        if num_steps == 0 {
            return Err(SpeechError::Input {
                why: "num_steps must be at least 1".into(),
            });
        }
        let codebooks = self.config.num_audio_codebook;
        let mask_id = self.config.audio_mask_id;
        let vocab = self.config.audio_vocab_size;
        let c_len = cond_audio_mask.len();
        // The target block is the tail of the sequence; a longer target
        // would underflow the offset below.
        if target_len > c_len {
            return Err(SpeechError::Input {
                why: format!("target_len {target_len} exceeds the {c_len}-position sequence"),
            });
        }
        let target_base = (c_len - target_len) * codebooks;
        let total_mask = target_len * codebooks;

        let mut cond_ids = cond_input_ids.to_vec();
        // The unconditional pass drops the [style | text] prefix and
        // keeps the target block only.
        let mut uncond_ids: Vec<i32> = cond_input_ids[target_base..].to_vec();
        let uncond_mask = cond_audio_mask[c_len - target_len..].to_vec();
        let layer_ids: Vec<f32> = (0..codebooks).map(|c| c as f32).collect();
        let timesteps = time_steps(num_steps, t_shift as f64);

        for step in 0..num_steps {
            let logits_all = self.forward(&cond_ids, cond_audio_mask)?;
            let cond_offset = target_base * vocab;
            let logits_cond = &logits_all[cond_offset..cond_offset + total_mask * vocab];
            let mut log_probs = if guidance_scale != 0.0 {
                let logits_uncond_all = self.forward(&uncond_ids, &uncond_mask)?;
                let logits_uncond = &logits_uncond_all[..total_mask * vocab];
                cfg_log_probs(logits_cond, logits_uncond, guidance_scale, vocab)
            } else {
                plain_log_probs(logits_cond, vocab)
            };
            // The mask token never wins an argmax.
            for cell in log_probs.chunks_mut(vocab) {
                cell[mask_id] = f32::NEG_INFINITY;
            }

            // Draw order per step: the class-uniform block first (only
            // at temperature > 0), then the position block.
            let step_uniforms = |count: usize| -> Vec<f32> {
                match draws {
                    Some(d) if step < d.len() => d[step].clone(),
                    _ => rng::uniform01(rng::Key::new(seed.wrapping_add(step as u64)), count),
                }
            };

            // Token choice: greedy argmax at temperature 0, otherwise
            // the top-ratio filter plus a Gumbel perturbation.
            let mut new_tokens = vec![0i32; total_mask];
            let class_draws = if class_temperature > 0.0 {
                Some(step_uniforms(total_mask * vocab))
            } else {
                None
            };
            for (cell_index, cell) in log_probs.chunks(vocab).enumerate() {
                let chosen = if let Some(uniforms) = &class_draws {
                    let filtered = filter_top_k(cell, 0.1);
                    let base = cell_index * vocab;
                    let mut scored = Vec::with_capacity(vocab);
                    for (v, &u) in filtered.iter().zip(&uniforms[base..base + vocab]) {
                        scored.push(v / class_temperature + gumbel(u));
                    }
                    argmax(&scored)
                } else {
                    argmax(cell)
                };
                new_tokens[cell_index] = chosen as i32;
            }

            // Confidence: the best masked-vocab log prob, penalized per
            // codebook layer, perturbed by the position Gumbel noise.
            let mut confidence = vec![0.0f32; total_mask];
            for (cell, slot) in log_probs.chunks(vocab).zip(confidence.iter_mut()) {
                *slot = cell.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            }
            // layer_ids broadcasts over the codebook axis: the penalty
            // follows each cell's COLUMN.
            for (index, value) in confidence.iter_mut().enumerate() {
                *value -= layer_ids[index % codebooks] * layer_penalty_factor;
            }
            if position_temperature > 0.0 {
                let uniforms = if class_temperature > 0.0 {
                    // The position block trails the class block.
                    match draws {
                        Some(d) if step < d.len() => {
                            d[step][total_mask * vocab..total_mask * vocab + total_mask].to_vec()
                        }
                        _ => rng::uniform01(
                            rng::Key::new(seed.wrapping_add(step as u64)),
                            total_mask,
                        ),
                    }
                } else {
                    step_uniforms(total_mask)
                };
                for (value, &u) in confidence.iter_mut().zip(&uniforms) {
                    *value = *value / position_temperature + gumbel(u);
                }
            }

            let dt = timesteps[step + 1] - timesteps[step];
            let mut k = ((total_mask as f64 * dt).ceil() as usize).max(1);
            if step == num_steps - 1 {
                k = total_mask;
            }

            // Reveal the k best-scoring still-masked cells.
            let mut score = confidence.clone();
            for (index, slot) in score.iter_mut().enumerate() {
                if cond_ids[target_base + index] != mask_id as i32 {
                    *slot = f32::NEG_INFINITY;
                }
            }
            let mut order: Vec<usize> = (0..total_mask).collect();
            order.sort_by(|&a, &b| {
                score[b]
                    .partial_cmp(&score[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut rank = vec![0usize; total_mask];
            for (position, &cell) in order.iter().enumerate() {
                rank[cell] = position;
            }
            for (index, &r) in rank.iter().enumerate() {
                let reveal = r < k && cond_ids[target_base + index] == mask_id as i32;
                if reveal {
                    cond_ids[target_base + index] = new_tokens[index];
                }
            }
            uncond_ids.copy_from_slice(&cond_ids[target_base..]);
        }

        let mut tokens = cond_ids[target_base..].to_vec();
        for value in &mut tokens {
            if *value == mask_id as i32 {
                *value = 0;
            }
        }
        Ok(tokens)
    }

    /// The single-utterance generate: assemble the inputs, run the
    /// iterative unmasking, and decode through the shared Higgs codec.
    /// `ref_tokens` (voice cloning) flips the style prefix to
    /// `<|denoise|>` at the caller and appends the reference code block.
    /// Returns the tokens `[T, C]` and the waveform.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &mut self,
        style_ids: &[i32],
        text_ids: &[i32],
        target_len: usize,
        ref_tokens: Option<&[Vec<i32>]>,
        num_steps: usize,
        guidance_scale: f32,
        class_temperature: f32,
        position_temperature: f32,
        layer_penalty_factor: f32,
        t_shift: f32,
        draws: Option<&[Vec<f32>]>,
        seed: u64,
        codec: &crate::codec::higgs_audio::HiggsAudioTokenizer,
    ) -> Result<(Vec<i32>, Vec<f32>)> {
        let (input_ids, audio_mask) =
            self.prepare_inference_inputs(style_ids, text_ids, target_len, ref_tokens)?;
        let tokens = self.iterative_unmask(
            &input_ids,
            &audio_mask,
            target_len,
            num_steps,
            guidance_scale,
            class_temperature,
            position_temperature,
            layer_penalty_factor,
            t_shift,
            draws,
            seed,
        )?;
        // Tokens [T, C] row-major -> per-book code rows for the codec.
        let codebooks = self.config.num_audio_codebook;
        let frames = tokens.len() / codebooks;
        let mut codes = vec![Vec::with_capacity(frames); codebooks];
        for frame in tokens.chunks(codebooks) {
            for (c, slot) in codes.iter_mut().enumerate() {
                slot.push(frame[c]);
            }
        }
        let audio = codec.decode_tokens(&codes)?;
        Ok((tokens, audio))
    }
}

/// `_gumbel_noise`'s noise term in f32: `-log(-log(u + 1e-10) + 1e-10)`.
fn gumbel(u: f32) -> f32 {
    -(-((u + 1e-10).ln()) + 1e-10).ln()
}

/// `_get_time_steps`: the cosine-shifted schedule
/// `t' = t_shift * t / (1 + (t_shift - 1) * t)` over `0..=1`.
pub fn time_steps(num_step: usize, t_shift: f64) -> Vec<f64> {
    (0..=num_step)
        .map(|i| {
            let t = i as f64 / num_step as f64;
            t_shift * t / (1.0 + (t_shift - 1.0) * t)
        })
        .collect()
}

/// The CFG combine: `log_softmax(c + g * (c - u))` per cell.
pub(crate) fn cfg_log_probs(
    cond: &[f32],
    uncond: &[f32],
    guidance_scale: f32,
    vocab: usize,
) -> Vec<f32> {
    let cells = cond.len() / vocab;
    let mut out = vec![0.0f32; cond.len()];
    for cell in 0..cells {
        let base = cell * vocab;
        let c_lp = log_softmax(&cond[base..base + vocab]);
        let u_lp = log_softmax(&uncond[base..base + vocab]);
        let mut combined = vec![0.0f32; vocab];
        for (v, slot) in combined.iter_mut().enumerate() {
            *slot = c_lp[v] + guidance_scale * (c_lp[v] - u_lp[v]);
        }
        out[base..base + vocab].copy_from_slice(&log_softmax(&combined));
    }
    out
}

/// The no-guidance path: a plain log softmax per cell.
fn plain_log_probs(logits: &[f32], vocab: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; logits.len()];
    for (slot, cell) in out.chunks_mut(vocab).zip(logits.chunks(vocab)) {
        slot.copy_from_slice(&log_softmax(cell));
    }
    out
}

/// `_filter_top_k`: keep the top `ceil(ratio * V)` entries by value and
/// set the rest to negative infinity (ties at the threshold kept).
fn filter_top_k(values: &[f32], ratio: f32) -> Vec<f32> {
    let len = values.len();
    let k = ((ratio * len as f32).ceil() as usize).clamp(1, len);
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted[k - 1];
    values
        .iter()
        .map(|&v| if v >= threshold { v } else { f32::NEG_INFINITY })
        .collect()
}

#[cfg(test)]
mod tests;
