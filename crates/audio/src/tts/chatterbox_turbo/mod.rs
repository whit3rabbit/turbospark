//! Chatterbox Turbo text-to-speech (the distilled GPT-2 + meanflow
//! voice-clone family over the S3 token space).
//!
//! Reference: `mlx_audio/tts/models/chatterbox_turbo/` at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/chatterbox_turbo):
//! the GPT-2-based T3 speech-token generator ([`t3`]), the S3Gen
//! meanflow token decoder plus HiFT vocoder ([`flow`], [`hift`]), the
//! Kaldi-fbank CAMPPlus front end ([`fbank`], with the CAMPPlus network
//! under the turbo key tree in [`campplus`]), the LSTM voice encoder
//! ([`voice_encoder`]), and the conformer encoder fork ([`conformer`]).
//!
//! Pipeline: caller-owned text token ids feed the T3 (the reference
//! `generate` tokenizes the normalized chunk and passes the ids
//! verbatim - no start/stop text wrapping, unlike the sibling
//! family); the generated stream is filtered to ids below 6561 and
//! closed with three 4299 silence tokens; the meanflow flow + HiFT
//! render the 24 kHz waveform conditioned on the reference-voice
//! prompt (prompt tokens, length-matched prompt mel, CAMPPlus
//! embedding) in two Euler steps without classifier-free guidance.
//!
//! Scope, matching the established family choices:
//! - single batch, non-streaming: the reference `inference` path with
//!   `finalize=True`; the `stream_generate` chunked loop and
//!   `inference_stream` are not ported;
//! - caller-owned text tokenization: the real checkpoints ship a
//!   Hugging Face tokenizer the portable crate does not embed. The
//!   reference's portable text normalization ([`punc_norm`]) is ported
//!   so a caller reproduces the reference text, and the sentence
//!   chunking (`max_tokens // 8 * 4` chars) stays caller-side;
//! - caller-owned resampling and loudness: [`Model::prepare_conditionals`]
//!   takes the three reference wavebands directly (16 kHz full for the
//!   voice encoder and T3 prompt tokens, 24 kHz truncated to
//!   `dec_cond_len` for the S3Gen mel, and the 16 kHz resample of that
//!   band for its tokens); the `integrated_loudness` normalization is
//!   not portable and stays out;
//! - deterministic filter math, recorded-uniform replay for the
//!   sampled paths: the T3 categorical draws use the crate-local
//!   MLX-compatible threefry RNG, so a fixture seed replays the Python
//!   stream exactly; the HiFT sine randomness and the flow start
//!   windows are explicit inputs ([`hift::SineDraws`],
//!   [`flow::FlowNoise`]); `temperature == 0` is the explicit greedy
//!   mode (post-penalty argmax);
//! - the >30 s sliding-window path of the S3 tokenizer is the codec
//!   port's (`crate::codec::s3`), reused here unchanged, as are the
//!   S3Gen mel front end helpers and the shared conv/linear layers from
//!   the sibling chatterbox port (identical math at this commit; the
//!   CAMPPlus network, voice encoder, and conformer encoder run under
//!   the turbo checkpoint's key trees in [`campplus`],
//!   [`voice_encoder`], and [`conformer`]).

pub mod campplus;
pub mod conformer;
pub mod fbank;
pub mod flow;
pub mod hift;
pub mod t3;
pub mod voice_encoder;

#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::s3::{self, S3TokenizerV2};
use crate::music::minimax_music3::rng;
use crate::tts::chatterbox::s3gen::prompt_mel;
use crate::{quant, Result, SpeechError};

use self::campplus::CampPlusConfig;
use self::flow::{FlowConfig, FlowNoise, S3GenFlow};
use self::hift::{Hift, HiftConfig, SineDraws};
use self::t3::{T3Cond, T3Config, T3GenParams, T3 as T3Generator};
use self::voice_encoder::{VoiceEncConfig, VoiceEncoder};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// Sample rate of the speech tokenizer band.
pub const S3_SR: u32 = 16_000;
/// Sample rate of the vocoder band and the output audio.
pub const S3GEN_SR: u32 = 24_000;
/// Size of the speech token vocabulary; the generation crop keeps ids
/// strictly below this (the start 6561 and stop 6562 markers are
/// specials).
pub const SPEECH_VOCAB_SIZE: i32 = 6561;
/// The silence token appended to the cropped stream
/// (`s3gen.S3GEN_SIL`), three times.
pub const S3GEN_SIL: i32 = 4299;

/// The family configuration over the reference defaults
/// (`T3Config.turbo`, the `S3Token2Mel` constructor arguments, the
/// conditioning lengths of `ChatterboxTurboTTS`).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub t3: T3Config,
    pub s3_sr: u32,
    pub s3gen_sr: u32,
    /// `ENC_COND_LEN` (15 s of the 16 kHz band).
    pub enc_cond_len: usize,
    /// `DEC_COND_LEN` (10 s of the 24 kHz band).
    pub dec_cond_len: usize,
    pub flow: FlowConfig,
    pub hift: HiftConfig,
    pub campplus: CampPlusConfig,
    pub ve: VoiceEncConfig,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            t3: T3Config::default(),
            s3_sr: S3_SR,
            s3gen_sr: S3GEN_SR,
            enc_cond_len: 15 * S3_SR as usize,
            dec_cond_len: 10 * S3GEN_SR as usize,
            flow: FlowConfig::default(),
            hift: HiftConfig::default(),
            campplus: CampPlusConfig::default(),
            ve: VoiceEncConfig::default(),
        }
    }
}

impl ModelConfig {
    /// Parses a `config.json`-style object over the reference defaults.
    /// The `t3_config` key carries the T3 fields (with the GPT-2
    /// backbone under `gpt2_config`), `flow_config` the flow geometry,
    /// `hift_config` the vocoder widths, `campplus_config` and
    /// `ve_config` the speaker encoders.
    pub fn from_json(value: &Value) -> Result<Self> {
        let defaults = ModelConfig::default();
        let t3_value = value.get("t3_config").cloned().unwrap_or(Value::Null);
        let t3 = match &t3_value {
            Value::Object(_) => T3Config::from_json_with_defaults(&t3_value, defaults.t3.clone())?,
            _ => defaults.t3.clone(),
        };
        let int = |name: &str, fallback: usize| -> Result<usize> {
            match value.get(name) {
                None | Some(Value::Null) => Ok(fallback),
                Some(v) => v
                    .as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| SpeechError::BadConfig {
                        field: name.to_string(),
                        why: "expected an integer".to_string(),
                    }),
            }
        };
        let flow = value.get("flow_config").cloned().unwrap_or(Value::Null);
        let flow = match &flow {
            Value::Object(_) => FlowConfig::from_json_with_defaults(&flow, defaults.flow.clone())?,
            _ => defaults.flow,
        };
        let hift_value = value.get("hift_config").cloned().unwrap_or(Value::Null);
        let hift = match &hift_value {
            Value::Object(_) => {
                HiftConfig::from_json_with_defaults(&hift_value, defaults.hift.clone())?
            }
            _ => defaults.hift,
        };
        let campplus_value = value.get("campplus_config").cloned().unwrap_or(Value::Null);
        let campplus = match &campplus_value {
            Value::Object(_) => {
                CampPlusConfig::from_json_with_defaults(&campplus_value, defaults.campplus.clone())?
            }
            _ => defaults.campplus,
        };
        let ve_value = value.get("ve_config").cloned().unwrap_or(Value::Null);
        let ve = match &ve_value {
            Value::Object(_) => ve_config_from_json(&ve_value, defaults.ve.clone())?,
            _ => defaults.ve,
        };
        Ok(ModelConfig {
            t3,
            s3_sr: int("s3_sr", defaults.s3_sr as usize)? as u32,
            s3gen_sr: int("s3gen_sr", defaults.s3gen_sr as usize)? as u32,
            enc_cond_len: int("enc_cond_len", defaults.enc_cond_len)?,
            dec_cond_len: int("dec_cond_len", defaults.dec_cond_len)?,
            flow,
            hift,
            campplus,
            ve,
        })
    }
}

/// Parses the voice-encoder geometry (`ve_config`); the fixture shrinks
/// the hidden and embedding widths.
fn ve_config_from_json(value: &Value, defaults: VoiceEncConfig) -> Result<VoiceEncConfig> {
    let int = |name: &str, fallback: usize| -> Result<usize> {
        match value.get(name) {
            None | Some(Value::Null) => Ok(fallback),
            Some(v) => v
                .as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| SpeechError::BadConfig {
                    field: name.to_string(),
                    why: "expected an integer".to_string(),
                }),
        }
    };
    let boolean = |name: &str, fallback: bool| -> Result<bool> {
        match value.get(name) {
            None | Some(Value::Null) => Ok(fallback),
            Some(v) => v.as_bool().ok_or_else(|| SpeechError::BadConfig {
                field: name.to_string(),
                why: "expected a bool".to_string(),
            }),
        }
    };
    Ok(VoiceEncConfig {
        num_mels: int("num_mels", defaults.num_mels)?,
        sample_rate: int("sample_rate", defaults.sample_rate as usize)? as u32,
        speaker_embed_size: int("speaker_embed_size", defaults.speaker_embed_size)?,
        ve_hidden_size: int("ve_hidden_size", defaults.ve_hidden_size)?,
        ve_partial_frames: int("ve_partial_frames", defaults.ve_partial_frames)?,
        ve_final_relu: boolean("ve_final_relu", defaults.ve_final_relu)?,
    })
}

/// The S3Gen reference conditioning (the `embed_ref` result fields the
/// inference consumes).
#[derive(Debug, Clone)]
pub struct GenCond {
    /// Prompt speech tokens `[1, T]` flattened.
    pub prompt_token: Vec<i32>,
    /// Prompt mel frames, row-major `[T_mel, 80]`.
    pub prompt_feat: Vec<f32>,
    /// CAMPPlus speaker embedding `[spk_embed_dim]`.
    pub embedding: Vec<f32>,
}

/// The prepared conditionals for one voice (`chatterbox_turbo.
/// Conditionals`).
#[derive(Debug, Clone)]
pub struct Conditionals {
    pub t3: T3Cond,
    pub gen: GenCond,
}

/// One generated result: the raw T3 id stream (trailing stop removed),
/// the cropped speech tokens with the silence tail, and the decoded
/// 24 kHz waveform.
#[derive(Debug, Clone)]
pub struct Generated {
    pub ids: Vec<i32>,
    pub speech_tokens: Vec<i32>,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Generation options with the reference `generate` defaults.
#[derive(Debug, Clone, Default)]
pub struct GenerateParams {
    pub t3: T3GenParams,
    /// Seed for the T3 categorical draws (unused when temperature is
    /// 0, the explicit greedy mode).
    pub seed: u64,
}

/// The loaded family model.
pub struct Model {
    pub config: ModelConfig,
    pub t3: T3Generator,
    pub s3gen: S3GenFlow,
    pub hift: Hift,
    pub ve: VoiceEncoder,
    /// The S3 speech tokenizer (`_s3tokenizer`), required only by
    /// [`Model::prepare_conditionals`]; the generate path runs from
    /// prepared conditionals without it.
    pub s3_tokenizer: Option<S3TokenizerV2>,
}

/// Checkpoint layout helper: the fixture and a converted checkpoint
/// both keep one safetensors file per directory.
fn checkpoint_file(dir: &Path) -> Result<PathBuf> {
    let path = dir.join("model.safetensors");
    if path.exists() {
        return Ok(path);
    }
    Err(SpeechError::BadConfig {
        field: dir.display().to_string(),
        why: "no model.safetensors found".to_string(),
    })
}

impl Model {
    /// Opens a checkpoint directory (`config.json` +
    /// `model.safetensors`, plus the optional `s3_tokenizer.safetensors`
    /// the conditioning path needs; real installs load the shared
    /// S3TokenizerV2 checkpoint through [`Model::with_s3_tokenizer`]).
    pub fn open(dir: &Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        let config = ModelConfig::from_json(&config_value)?;
        let path = checkpoint_file(dir)?;
        let model = Self::load(&path, &config)?;
        let s3_path = dir.join("s3_tokenizer.safetensors");
        let s3_config = dir.join("s3_tokenizer_config.json");
        let s3_tokenizer = if s3_path.exists() && s3_config.exists() {
            let config_json = quant::read_json(&s3_config)?;
            let s3_config =
                s3::S3Config::from_json_with_defaults(&config_json, s3::S3Config::v2())?;
            let s3_file = SafetensorsFile::open(&s3_path).map_err(SpeechError::from)?;
            Some(S3TokenizerV2::load(s3_config, &s3_file)?)
        } else {
            None
        };
        Ok(Model {
            s3_tokenizer,
            ..model
        })
    }

    /// Loads every submodule from one checkpoint file with the
    /// reference prefixes (`t3.`, `s3gen.` split into the flow, vocoder,
    /// and speaker-encoder subtrees, `ve.`). The T3 backbone reopens
    /// the file through the shared LM `Weights` loader.
    pub fn load(path: &Path, config: &ModelConfig) -> Result<Self> {
        let file = SafetensorsFile::open(path).map_err(SpeechError::from)?;
        let t3 = T3Generator::load(path, &file, &config.t3)?;
        let s3gen = S3GenFlow::load(&config.flow, config.campplus.clone(), &file, "s3gen")?;
        let hift = Hift::load(&config.hift, &file, "s3gen.mel2wav")?;
        let ve = VoiceEncoder::load(&file, "ve.", config.ve.clone())?;
        Ok(Model {
            config: config.clone(),
            t3,
            s3gen,
            hift,
            ve,
            s3_tokenizer: None,
        })
    }

    /// Loads the S3 speech tokenizer from its own checkpoint directory
    /// (`config.json` + safetensors), the shape the real
    /// `mlx-community/S3TokenizerV2` install takes.
    pub fn with_s3_tokenizer(mut self, dir: &Path) -> Result<Self> {
        let config_json = quant::read_json(&dir.join("config.json"))?;
        let s3_config = s3::S3Config::from_json_with_defaults(&config_json, s3::S3Config::v2())?;
        let path = dir.join("model.safetensors");
        let file = SafetensorsFile::open(&path).map_err(SpeechError::from)?;
        self.s3_tokenizer = Some(S3TokenizerV2::load(s3_config, &file)?);
        Ok(self)
    }

    /// The turbo prompt layout: the caller's text token ids pass
    /// through verbatim (`generate` feeds the tokenizer output
    /// straight into `inference_turbo`; there is no start/stop text
    /// wrapping in this family).
    pub fn prepare_text_tokens(&self, text_ids: &[i32]) -> Vec<i32> {
        text_ids.to_vec()
    }

    /// `S3Token2Mel.embed_ref` + the T3 conditioning token path of
    /// `prepare_conditionals`, from the three reference wavebands:
    /// `wav_16k_full` (the voice-encoder band, up to `enc_cond_len`),
    /// `wav_24k` (the S3Gen mel band, up to `dec_cond_len`), and
    /// `wav_16k_from_24k` (the S3Gen token band, truncated to the
    /// decoder conditioning length scaled to 16 kHz). Requires the S3
    /// tokenizer.
    pub fn prepare_conditionals(
        &mut self,
        wav_16k_full: &[f32],
        wav_24k: &[f32],
        wav_16k_from_24k: &[f32],
    ) -> Result<Conditionals> {
        let tokenizer = self
            .s3_tokenizer
            .as_ref()
            .ok_or_else(|| SpeechError::Input {
                why: "prepare_conditionals requires the S3 speech tokenizer".to_string(),
            })?;
        let n_mels = self.config.flow.output_size;
        // --- S3Gen band: mel (24 kHz), tokens (16 kHz band trimmed to
        // the decoder conditioning length at the 16 kHz rate) ---
        let wav_24k = &wav_24k[..wav_24k.len().min(self.config.dec_cond_len)];
        let mut mel = prompt_mel(wav_24k, n_mels, self.config.s3gen_sr)?;
        let mel_len = mel.len();
        let dec_trim =
            self.config.dec_cond_len * self.config.s3_sr as usize / self.config.s3gen_sr as usize;
        let mut log_mel = s3::log_mel_spectrogram(
            &wav_16k_from_24k[..wav_16k_from_24k.len().min(dec_trim)],
            128,
        )?;
        // The reference tokenizer front end drops the final centered
        // frame (`spec[:, :-1]`); the codec helper keeps it.
        log_mel.truncate(log_mel.len().saturating_sub(1));
        // The codec tokenizer consumes channel-major `[n_mels, frames]`.
        let n_frames = log_mel.len();
        let mut mel_flat = vec![0.0f32; n_frames * 128];
        for (t, frame) in log_mel.iter().enumerate() {
            for (m, &v) in frame.iter().enumerate() {
                mel_flat[m * n_frames + t] = v;
            }
        }
        let (mut tokens, _token_len) = tokenizer.quantize(&mel_flat, n_frames)?;

        // Speaker embedding: CAMPPlus over the 16 kHz resample through
        // the turbo fbank (float64 Kaldi front end + per-bin mean
        // removal).
        let fbank = fbank::mean_normalized_fbank(wav_16k_from_24k);
        let frames = fbank.len();
        let feat_dim = fbank.first().map_or(0, Vec::len);
        let mut flat = Vec::with_capacity(frames * feat_dim);
        for row in &fbank {
            flat.extend_from_slice(row);
        }
        let embedding = self.s3gen.speaker_embed(&flat, frames);

        // Align mel/token lengths: mel frames must equal twice the
        // token count; truncate whichever side is longer.
        let expected_tokens = mel_len / 2;
        let mut actual = tokens.len();
        if actual != expected_tokens {
            if actual < expected_tokens {
                // Tokens are shorter: truncate the mel to 2 * tokens.
                let keep = 2 * actual;
                mel.truncate(keep);
            } else {
                tokens.truncate(expected_tokens);
                actual = expected_tokens;
            }
        }
        let mut prompt_feat = Vec::with_capacity(mel.len() * n_mels);
        for frame in &mel {
            prompt_feat.extend_from_slice(frame);
        }

        // --- T3 band: prompt tokens from the 16 kHz full band ---
        let wav_16k = &wav_16k_full[..wav_16k_full.len().min(self.config.enc_cond_len)];
        let mut t3_log_mel = s3::log_mel_spectrogram(wav_16k, 128)?;
        t3_log_mel.truncate(t3_log_mel.len().saturating_sub(1));
        let t3_frames = t3_log_mel.len();
        let mut t3_flat = vec![0.0f32; t3_frames * 128];
        for (t, frame) in t3_log_mel.iter().enumerate() {
            for (m, &v) in frame.iter().enumerate() {
                t3_flat[m * t3_frames + t] = v;
            }
        }
        let (mut t3_tokens, _) = tokenizer.quantize(&t3_flat, t3_frames)?;
        let plen = self.config.t3.speech_cond_prompt_len;
        t3_tokens.truncate(plen);

        // --- Voice encoder speaker embedding ---
        let speaker = self.ve.embeds_from_wav(wav_16k_full, Some(20.0))?;
        let _ = actual;
        Ok(Conditionals {
            t3: T3Cond {
                speaker_emb: speaker,
                cond_prompt_speech_tokens: Some(t3_tokens),
            },
            gen: GenCond {
                prompt_token: tokens,
                prompt_feat,
                embedding,
            },
        })
    }

    /// Runs the full pipeline: T3 generation, the `< 6561` crop with
    /// the three-silence tail, the meanflow S3Gen flow, the HiFT
    /// vocoder, and the 20 ms fade-in. `sine_draws` injects the vocoder
    /// sine randomness and `flow_noise` the flow start windows (`None`
    /// = crate RNG draws; see [`hift::SineDraws`] and
    /// [`flow::FlowNoise`]).
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &mut self,
        text_ids: &[i32],
        conds: &Conditionals,
        params: &GenerateParams,
        sine_draws: Option<&SineDraws>,
        flow_noise: Option<&FlowNoise>,
    ) -> Result<Generated> {
        let prompt = self.prepare_text_tokens(text_ids);
        let mut keys = rng::KeySequence::new(params.seed);
        let ids = self
            .t3
            .inference_turbo(&conds.t3, &prompt, &params.t3, &mut keys)?;
        let speech_tokens = self.sanitize_speech_tokens(&ids);
        if speech_tokens.is_empty() {
            return Err(SpeechError::Input {
                why: "generation produced no speech tokens below the vocabulary bound".to_string(),
            });
        }
        let mel_rows = self.s3gen.infer(
            &speech_tokens,
            &conds.gen.prompt_token,
            &conds.gen.prompt_feat,
            &conds.gen.embedding,
            self.s3gen.n_timesteps(),
            flow_noise,
            &mut keys,
        )?;
        let mut samples = self.hift.infer(&mel_rows, sine_draws)?;
        crate::tts::chatterbox::hift::trim_fade(&mut samples);
        Ok(Generated {
            ids,
            speech_tokens,
            samples,
            sample_rate: S3GEN_SR,
        })
    }

    /// The turbo crop (`generate`): keep the ids strictly below 6561 in
    /// stream order (the whole stream, not a start/stop span) and
    /// append the three 4299 silence tokens.
    fn sanitize_speech_tokens(&self, ids: &[i32]) -> Vec<i32> {
        let mut tokens: Vec<i32> = ids
            .iter()
            .copied()
            .filter(|&t| t < SPEECH_VOCAB_SIZE)
            .collect();
        tokens.extend_from_slice(&[S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]);
        tokens
    }
}

/// `chatterbox_turbo.punc_norm`, the portable text cleanup run before
/// tokenization. The turbo replacement table differs from the sibling
/// family's: no ellipsis-to-comma for three ASCII dots, no spaced-dash
/// or semicolon handling, and the sentence enders are ASCII only.
pub fn punc_norm(text: &str) -> String {
    if text.is_empty() {
        return "You need to add some text for me to talk.".to_string();
    }
    let mut out = text.to_string();
    // Capitalise the first character when it is lowercase.
    let mut chars = out.chars();
    if let Some(first) = chars.next() {
        if first.is_lowercase() {
            let upper: String = first.to_uppercase().collect();
            out = format!("{upper}{}", &out[first.len_utf8()..]);
        }
    }
    // Collapse whitespace runs to single spaces.
    let mut collapsed = String::with_capacity(out.len());
    let mut in_space = false;
    for c in out.chars() {
        if c.is_whitespace() {
            if !in_space {
                collapsed.push(' ');
                in_space = true;
            }
        } else {
            collapsed.push(c);
            in_space = false;
        }
    }
    let mut text = collapsed;
    // Replace uncommon/LLM punctuation.
    for (old, new) in [
        ("\u{2026}", ", "),
        (":", ","),
        ("\u{2014}", "-"),
        ("\u{2013}", "-"),
        (" ,", ","),
        ("\u{201c}", "\""),
        ("\u{201d}", "\""),
        ("\u{2018}", "'"),
        ("\u{2019}", "'"),
    ] {
        text = text.replace(old, new);
    }
    // Add a full stop when no sentence ender closes the text.
    let trimmed = text.trim_end_matches(' ');
    let enders = ['.', '!', '?', '-', ','];
    let needs_stop = !trimmed.chars().last().is_some_and(|c| enders.contains(&c));
    if needs_stop {
        format!("{trimmed}.")
    } else {
        trimmed.to_string()
    }
}
