//! Dia dialogue TTS family (Nari Labs Dia 1.6B).
//!
//! Port of `mlx_audio/tts/models/dia/` (dia.py, config.py, layers.py,
//! audio.py) at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/dia):
//! a two-stack sequence-to-sequence transformer. A byte-level text
//! encoder (`[S1]`/`[S2]` speaker marks replaced by the bytes 0x01/0x02,
//! right-padded to `text_length`, with a pad-compatibility attention
//! mask) conditions a nine-channel audio-code decoder through
//! precomputed cross attention. The decoder runs classifier-free
//! guidance (a zero-text unconditional row travels alongside the
//! conditional row), samples one code per channel per step under a
//! channel-delay pattern (`[0, 8, 9, ..., 15]`, BOS-forced before each
//! channel's delay elapses), masks the logits past code 1024, detects
//! EOS on channel 0, and keeps counting 30 extra steps while the
//! delayed channels receive their EOS/PAD bookkeeping (added on top of
//! the freshly sampled codes, which can push ids past the vocabulary;
//! the pinned MLX embeds such ids as zero rows and the port matches).
//! The sliced code grid drops its first frame and its final 30 frames,
//! clamps invalid codes to 0, and decodes through the Descript Audio
//! Codec.
//!
//! The DAC is a separate downloadable checkpoint
//! (`mlx-community/descript-audio-codec-44khz`); it is already ported
//! in [crate::codec::descript], so [`Model::generate`] takes a
//! caller-loaded [`Dac`]. See [README.md](README.md) for the
//! verification evidence and open items.

use self::audio::{audio_to_codebook, codebook_to_audio};
use self::model::{Decoder, Encoder};
use crate::codec::descript::Dac;
use crate::lm::Weights;
use crate::music::minimax_music3::rng;
use crate::tts::sampling;
use crate::{quant, Result, SpeechError};

mod audio;
mod config;
mod layers;
mod model;

#[cfg(test)]
mod tests;

pub use config::{DataConfig, DiaConfig, ModelConfig};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// The logits keep-mask width the reference hardcodes: codes `0..=1024`
/// (the 1024 codebook entries plus EOS) stay; PAD 1025 and BOS 1026
/// mask to negative infinity regardless of the configured vocabulary.
const KEEP_VOCAB: usize = 1025;

/// `extra_steps_after_eos`: the fixed countdown after channel-0 EOS;
/// the decode also drops exactly this many final frames.
pub const EXTRA_STEPS_AFTER_EOS: usize = 30;

/// The additive mask value the reference writes where attention is
/// denied (`mx.where(mask, scores, -1e9)`).
pub(crate) const MASK_DENIED: f32 = -1.0e9;

/// `generate` parameters; the `_generate` signature defaults.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    /// `None` uses `data.audio_length`.
    pub max_tokens: Option<usize>,
    pub cfg_scale: f32,
    pub temperature: f32,
    pub top_p: f32,
    pub use_cfg_filter: bool,
    pub cfg_filter_top_k: usize,
    /// The reference default `"\n"`.
    pub split_pattern: String,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: None,
            cfg_scale: 3.0,
            temperature: 1.3,
            top_p: 0.95,
            use_cfg_filter: true,
            cfg_filter_top_k: 35,
            split_pattern: "\n".to_owned(),
        }
    }
}

/// The per-step token source (`_sample_next_token`): one code per
/// channel from the kept CFG logits `[channels, vocab]`. Temperature 0
/// implementations take the per-channel argmax exactly like the
/// reference early return; sampled implementations apply the
/// `make_sampler` filter chain (top-p on the raw CFG logits, then
/// top-k) and draw. The draw itself uses the crate-local threefry
/// sequence, so a Rust run does not reproduce an MLX draw bit for bit;
/// the filtered support is what the fixtures pin.
pub trait SampleSource {
    fn sample(
        &mut self,
        logits: &[f32],
        channels: usize,
        vocab: usize,
        temperature: f32,
    ) -> Result<Vec<i32>>;
}

/// Temperature-0 argmax (first maximal index per channel, like
/// `mx.argmax`).
pub struct Greedy;

impl SampleSource for Greedy {
    fn sample(
        &mut self,
        logits: &[f32],
        channels: usize,
        vocab: usize,
        _temperature: f32,
    ) -> Result<Vec<i32>> {
        Ok((0..channels)
            .map(|c| {
                let row = &logits[c * vocab..(c + 1) * vocab];
                sampling::argmax(row) as i32
            })
            .collect())
    }
}

/// The sampled path: the reference `make_sampler(temp, top_p, top_k)`
/// chain over one channel row at a time. Filters run on the raw CFG
/// logits exactly like the reference call (the reference passes the
/// CFG output straight to the sampler, no log-softmax first).
pub struct Sampled {
    pub top_p: f32,
    pub top_k: usize,
    keys: rng::KeySequence,
}

impl Sampled {
    pub fn new(seed: u64, top_p: f32, top_k: usize) -> Self {
        Self {
            top_p,
            top_k,
            keys: rng::KeySequence::new(seed),
        }
    }
}

impl SampleSource for Sampled {
    fn sample(
        &mut self,
        logits: &[f32],
        channels: usize,
        vocab: usize,
        temperature: f32,
    ) -> Result<Vec<i32>> {
        let mut out = Vec::with_capacity(channels);
        for c in 0..channels {
            let mut row = logits[c * vocab..(c + 1) * vocab].to_vec();
            if self.top_p > 0.0 && self.top_p < 1.0 {
                sampling::apply_top_p(&mut row, self.top_p);
            }
            // An out-of-range top_k is an error in apply_top_k. Ignoring
            // it would sample the unfiltered distribution silently.
            if self.top_k > 0 {
                sampling::apply_top_k(&mut row, self.top_k)?;
            }
            let key = self.keys.next();
            out.push(sampling::sample_categorical(&row, temperature, key) as i32);
        }
        Ok(out)
    }
}

/// One generated segment.
#[derive(Debug, Clone)]
pub struct Generated {
    /// The sliced code grid `[channels][frames]` exactly as handed to
    /// the codec conversion (before the revert and clamp).
    pub codes: Vec<Vec<i32>>,
    /// Invalid codes clamped to 0 during the codec conversion.
    pub clamped_codes: usize,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// The reference `generation_step_index` return value.
    pub token_count: usize,
}

/// Self-attention cache length for one segment. Prefill installs
/// `1 + prompt_frames` positions and each decode step appends one, so
/// the cache must hold the prompt block plus the full budget. Sizing it
/// from `max_tokens` alone overflows when a voice prompt is longer than
/// the budget.
pub(crate) fn self_cache_capacity(prompt_frames: usize, max_tokens: usize) -> Result<usize> {
    prompt_frames
        .checked_add(max_tokens)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| SpeechError::Input {
            why: format!(
                "max_tokens {max_tokens} with a {prompt_frames}-frame voice prompt overflows the decoder cache"
            ),
        })
}

/// The loaded Dia transformer (the DAC travels separately).
pub struct Model {
    pub config: DiaConfig,
    encoder: Encoder,
    decoder: Decoder,
}

impl Model {
    /// Opens a checkpoint directory (`config.json` + safetensors).
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        let config = DiaConfig::from_json(&config_value)?;
        let weights = Weights::open(dir)?;
        Self::load(&weights, config)
    }

    /// Loads from opened weights and a parsed config. The checkpoint
    /// tensor names sit at the `DiaModel` root (`encoder.*`,
    /// `decoder.*`).
    pub fn load(weights: &Weights, config: DiaConfig) -> Result<Self> {
        let file = weights
            .shards()
            .first()
            .ok_or_else(|| SpeechError::Tensor {
                name: "model.safetensors".into(),
                why: "checkpoint has no shards".into(),
            })?;
        Self::validate(&config)?;
        Ok(Self {
            encoder: Encoder::load(file, &config)?,
            decoder: Decoder::load(file, &config)?,
            config,
        })
    }

    fn validate(config: &DiaConfig) -> Result<()> {
        let data = &config.data;
        if data.delay_pattern.len() != data.channels {
            return Err(SpeechError::BadConfig {
                field: "delay_pattern".into(),
                why: format!(
                    "{} delays for {} channels",
                    data.delay_pattern.len(),
                    data.channels
                ),
            });
        }
        let model = &config.model;
        // Zero kv heads would make the `%` below divide by zero. Refuse
        // it here so a bad checkpoint config fails at load, not mid-decode.
        if model.decoder.kv_heads == 0 {
            return Err(SpeechError::BadConfig {
                field: "kv_heads".into(),
                why: "must be positive".into(),
            });
        }
        if model.decoder.gqa_query_heads % model.decoder.kv_heads != 0 {
            return Err(SpeechError::BadConfig {
                field: "gqa_query_heads".into(),
                why: "must be divisible by kv_heads".into(),
            });
        }
        Ok(())
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.model.sample_rate
    }

    /// `_split_turns`: segments of at most two `[S1]`/`[S2]` chunks,
    /// pairwise merged.
    pub fn split_turns(text: &str) -> Vec<String> {
        let mut segments: Vec<String> = Vec::new();
        let mut cursor = 0usize;
        while let Some(s1) = text[cursor..].find("[S1]") {
            let s1_start = cursor + s1;
            let Some(s2_off) = text[s1_start + 4..].find("[S2]") else {
                break;
            };
            let s2_start = s1_start + 4 + s2_off;
            let rest = &text[s2_start + 4..];
            let next_s1 = rest.find("[S1]").unwrap_or(rest.len());
            let s1_chunk = text[s1_start + 4..s2_start].trim();
            let s2_chunk = rest[..next_s1].trim();
            segments.push(format!("[S1] {s1_chunk} [S2] {s2_chunk}"));
            cursor = s2_start + 4 + next_s1;
        }
        if segments.len() > 1 {
            let mut merged = Vec::new();
            let mut i = 0;
            while i < segments.len() {
                if i + 1 < segments.len() {
                    merged.push(format!("{} {}", segments[i], segments[i + 1]));
                    i += 2;
                } else {
                    merged.push(segments[i].clone());
                    i += 1;
                }
            }
            segments = merged;
        }
        segments
    }

    /// `_prepare_text_input`: the utf-8 bytes with `[S1]`/`[S2]`
    /// replaced by 0x01/0x02, right-padded (or truncated) to
    /// `text_length`. Returns the token rows for both batch rows (the
    /// unconditional row is all pad) and the per-row padding mask.
    pub fn prepare_text(&self, text: &str) -> (Vec<Vec<i32>>, Vec<Vec<bool>>) {
        let data = &self.config.data;
        let max_len = data.text_length;
        let bytes = text.as_bytes();
        let mut tokens: Vec<i32> = Vec::with_capacity(bytes.len());
        let mut i = 0usize;
        while i < bytes.len() {
            if bytes[i..].starts_with(b"[S1]") {
                tokens.push(1);
                i += 4;
            } else if bytes[i..].starts_with(b"[S2]") {
                tokens.push(2);
                i += 4;
            } else {
                tokens.push(bytes[i] as i32);
                i += 1;
            }
        }
        tokens.truncate(max_len);
        tokens.resize(max_len, data.text_pad_value);
        let mask: Vec<bool> = tokens.iter().map(|&t| t != data.text_pad_value).collect();
        let uncond = vec![0i32; max_len];
        let uncond_mask: Vec<bool> = uncond.iter().map(|&t| t != data.text_pad_value).collect();
        (vec![uncond, tokens], vec![uncond_mask, mask])
    }

    /// The reference `generate`: splits the prompt on `split_pattern`,
    /// expands two-speaker segments into turns, and synthesizes each
    /// segment. `ref_audio` is mono samples at the checkpoint sample
    /// rate; `ref_text` prepends to the text prompt and activates the
    /// voice-prompt prefill.
    pub fn generate(
        &mut self,
        text: &str,
        params: &GenerateParams,
        dac: &Dac,
        source: &mut dyn SampleSource,
        ref_audio: Option<&[f32]>,
        ref_text: Option<&str>,
    ) -> Result<Vec<Generated>> {
        let prompt = text.replace("\\n", "\n").replace("\\t", "\t");
        let mut segments: Vec<String> = Vec::new();
        for part in prompt.split(params.split_pattern.as_str()) {
            if part.contains("[S1]") && part.contains("[S2]") {
                segments.extend(Self::split_turns(part));
            } else {
                segments.push(part.to_owned());
            }
        }
        let mut results = Vec::with_capacity(segments.len());
        for segment in &segments {
            results.push(self.generate_segment(segment, params, dac, source, ref_audio, ref_text)?);
        }
        Ok(results)
    }

    /// The additive `[S, S]` pad-compatibility mask per row: allowed
    /// where query and key agree on their pad status.
    fn enc_masks(&self, src_padding_mask: &[Vec<bool>]) -> Vec<Vec<f32>> {
        let seq = self.config.data.text_length;
        let mut masks = Vec::with_capacity(2);
        for row_masks in &src_padding_mask[..2] {
            let mut mask = vec![0.0f32; seq * seq];
            for q in 0..seq {
                for k in 0..seq {
                    if row_masks[q] != row_masks[k] {
                        mask[q * seq + k] = MASK_DENIED;
                    }
                }
            }
            masks.push(mask);
        }
        masks
    }

    /// The decode-step cross mask: the query pad flags are fixed (the
    /// generated rows are never PAD), keys follow the text padding.
    fn cross_masks(&self, src_padding_mask: &[Vec<bool>], query_pad: &[bool; 2]) -> Vec<Vec<f32>> {
        let seq = self.config.data.text_length;
        let mut masks = Vec::with_capacity(2);
        for b in 0..2 {
            let mut mask = vec![0.0f32; seq];
            for (k, denied) in mask.iter_mut().enumerate() {
                let allowed = (query_pad[b] && !src_padding_mask[b][k])
                    || (!query_pad[b] && src_padding_mask[b][k]);
                *denied = if allowed { 0.0 } else { MASK_DENIED };
            }
            masks.push(mask);
        }
        masks
    }
    /// `_generate` for one segment.
    #[allow(clippy::too_many_lines)]
    fn generate_segment(
        &mut self,
        text: &str,
        params: &GenerateParams,
        dac: &Dac,
        source: &mut dyn SampleSource,
        ref_audio: Option<&[f32]>,
        ref_text: Option<&str>,
    ) -> Result<Generated> {
        let data = &self.config.data;
        let channels = data.channels;
        let bos = data.audio_bos_value;
        let eos = data.audio_eos_value;
        let pad = data.audio_pad_value;
        let delay: Vec<usize> = data.delay_pattern.clone();
        let max_delay = *delay.iter().max().unwrap_or(&0);
        let max_tokens = params.max_tokens.unwrap_or(data.audio_length);
        if max_tokens == 0 {
            return Err(SpeechError::Input {
                why: "max_tokens must be positive".into(),
            });
        }

        let text = match ref_text {
            Some(ref_text) => format!("{} {text}", ref_text.trim()),
            None => text.to_owned(),
        };
        let (src_tokens, src_padding_mask) = self.prepare_text(&text);
        let seq = self.config.data.text_length;
        let enc_masks = self.enc_masks(&src_padding_mask);
        let src_positions: Vec<i32> = (0..seq as i64).map(|v| v as i32).collect();
        let encoder_out = self
            .encoder
            .forward(&src_tokens, &src_positions, &enc_masks)?;

        let cross_caches = self
            .decoder
            .precompute_cross_kv(&encoder_out, seq, &src_positions);
        // The voice prompt is encoded first: its frame count sizes the
        // self-attention cache and the decoder grid.
        let prompt_codes = match ref_audio {
            Some(samples) => Some(audio_to_codebook(dac, samples, &delay, bos, pad)?),
            None => None,
        };
        let prompt_frames = prompt_codes.as_ref().map_or(0, Vec::len);
        let capacity = self_cache_capacity(prompt_frames, max_tokens)?;
        let mut self_caches = self.decoder.new_self_caches(capacity);

        // Decoder input grid `[2][positions][channels]`; the tail past
        // the written positions holds BOS placeholders that are never
        // read (the reference fills it with -1 sentinels for the same
        // effect).
        let mut generated: Vec<Vec<Vec<i32>>> = vec![
            vec![vec![bos; channels]; capacity],
            vec![vec![bos; channels]; capacity],
        ];
        let mut prompt_len_inc_bos = 1usize;
        let mut current_step = 0usize;
        let query_pad = [false, false];

        if let Some(prompt_codes) = &prompt_codes {
            let prefill_len = 1 + prompt_frames;
            for row in &mut generated {
                for (t, prompt_row) in prompt_codes.iter().enumerate() {
                    row[1 + t].copy_from_slice(prompt_row);
                }
            }
            prompt_len_inc_bos = prefill_len;
            current_step = prefill_len - 1;
            let positions: Vec<i32> = (0..prefill_len as i64).map(|v| v as i32).collect();
            // The per-position prefill query pad flags: true where
            // every channel of the row is PAD (BOS rows count as
            // non-pad).
            let prefill_pad: Vec<bool> = (0..prefill_len)
                .map(|t| generated[0][t].iter().all(|&v| v == pad))
                .collect();
            let mut self_masks: Vec<Vec<f32>> =
                vec![vec![MASK_DENIED; prefill_len * prefill_len]; 2];
            let mut cross_masks: Vec<Vec<f32>> = vec![vec![MASK_DENIED; prefill_len * seq]; 2];
            for b in 0..2 {
                for q in 0..prefill_len {
                    for k in 0..prefill_len {
                        let compatible = prefill_pad[q] == prefill_pad[k];
                        let causal = k <= q;
                        if compatible && causal {
                            self_masks[b][q * prefill_len + k] = 0.0;
                        }
                    }
                    for k in 0..seq {
                        let allowed = (!prefill_pad[q] && src_padding_mask[b][k])
                            || (prefill_pad[q] && !src_padding_mask[b][k]);
                        if allowed {
                            cross_masks[b][q * seq + k] = 0.0;
                        }
                    }
                }
            }
            let block: Vec<Vec<i32>> = (0..prefill_len).map(|t| generated[0][t].clone()).collect();
            let block = vec![block.clone(), block];
            self.decoder.prefill(
                &block,
                prefill_len,
                &positions,
                &self_masks,
                &cross_masks,
                &mut self_caches,
                &cross_caches,
            )?;
        }

        let cross_mask = self.cross_masks(&src_padding_mask, &query_pad);
        let vocab = self.config.model.tgt_vocab_size;
        let mut eos_detected = false;
        let mut eos_countdown: isize = -1;
        let mut generation_step_index = 0usize;
        let mut last_step: Option<usize> = None;

        for step in current_step..current_step + max_tokens {
            last_step = Some(step);
            let tgt_ids = generated[0][step].clone();
            let logits = self.decoder.decode_step(
                &tgt_ids,
                step as i32,
                &cross_mask,
                &mut self_caches,
                &cross_caches,
            );
            // CFG combine over the [2, C * V] logits block: the
            // unconditional row comes first.
            let mut cfg = vec![0.0f32; channels * vocab];
            let width = channels * vocab;
            for i in 0..width {
                let uncond = logits[i];
                let cond = logits[width + i];
                cfg[i] = cond + params.cfg_scale * (cond - uncond);
            }
            // Keep mask: codes past 1024 mask to -inf.
            for row in 0..channels {
                for v in KEEP_VOCAB..vocab {
                    cfg[row * vocab + v] = f32::NEG_INFINITY;
                }
            }
            let mut pred = source.sample(&cfg, channels, vocab, params.temperature)?;

            if ref_audio.is_none() {
                let step_index = step - current_step;
                for (c, &d) in delay.iter().enumerate() {
                    if step_index < d {
                        pred[c] = bos;
                    }
                }
            }

            for row in &mut generated {
                row[step + 1].copy_from_slice(&pred);
            }

            if !eos_detected && pred[0] == eos {
                eos_detected = true;
                eos_countdown = EXTRA_STEPS_AFTER_EOS as isize;
            }

            if eos_countdown > 0 {
                let step_after_eos = max_delay as isize - eos_countdown;
                for (i, &d) in delay.iter().enumerate() {
                    let d = d as isize;
                    if step_after_eos == d {
                        for row in &mut generated {
                            row[step + 1][i] += eos;
                        }
                    } else if step_after_eos > d {
                        for row in &mut generated {
                            row[step + 1][i] += pad;
                        }
                    }
                }
                eos_countdown -= 1;
                if eos_countdown == 0 {
                    break;
                }
            }

            generation_step_index = step - current_step + 1;
        }

        // output_codes = generated[:, prompt_len_inc_bos : last_step + 1].
        let mut codes: Vec<Vec<i32>> = vec![Vec::new(); channels];
        if let Some(last) = last_step {
            let end = last + 1;
            if end > prompt_len_inc_bos {
                for position in &generated[0][prompt_len_inc_bos..end] {
                    for (c, code) in position.iter().enumerate() {
                        codes[c].push(*code);
                    }
                }
            }
        }

        let (samples, clamped) = codebook_to_audio(&codes, dac, &delay, max_tokens)?;
        Ok(Generated {
            codes,
            clamped_codes: clamped,
            samples,
            sample_rate: self.config.model.sample_rate,
            token_count: generation_step_index,
        })
    }
}
