//! Chatterbox text-to-speech (voice-clone TTS over the S3 token space).
//!
//! Reference: `mlx_audio/tts/models/chatterbox/` at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/chatterbox):
//! the T3 llama speech-token generator ([`t3`]), the S3Gen flow +
//! HiFT vocoder ([`flow`], [`hift`], [`conformer`]), the CAMPPlus
//! speaker encoder ([`campplus`]), and the LSTM voice encoder
//! ([`voice_encoder`]).
//!
//! Pipeline: text token ids (caller-owned; see below) are wrapped in
//! the T3 start/stop text tokens, the T3 generates speech tokens with
//! classifier-free guidance, the invalid-token crop filters the stream,
//! and the S3Gen flow + HiFT render the 24 kHz waveform conditioned on
//! the reference-voice prompt (prompt tokens, length-matched prompt
//! mel, CAMPPlus embedding).
//!
//! Scope, matching the established family choices:
//! - single batch, non-streaming (the reference `generate` is a
//!   single-yield generator; `finalize=True` everywhere);
//! - caller-owned text tokenization: the real checkpoints ship a
//!   Hugging Face tokenizer the portable crate does not embed. The
//!   reference's portable text normalization ([`punc_norm`]) and the
//!   exact prompt layout ([`Model::prepare_text_tokens`]) are ported so
//!   a caller reproduces the reference ids from its own tokenizer;
//! - caller-owned resampling: [`Model::prepare_conditionals`] takes the
//!   three reference wavebands (16 kHz full, 24 kHz truncated to
//!   `dec_cond_len`, 16 kHz resampled from the 24 kHz band) instead of
//!   running the polyphase resampler;
//! - deterministic filter math, recorded-uniform replay for the
//!   sampled paths: the T3 categorical draws use the crate-local
//!   MLX-compatible threefry RNG, so a fixture seed replays the Python
//!   stream exactly; the HiFT sine randomness (initial phase and noise
//!   bed) and the CFM noise window are explicit inputs;
//! - the >30 s sliding-window path of the reference S3 tokenizer is
//!   the codec port's (`crate::codec::s3`), reused here unchanged.
//!
//! `chatterbox_turbo` is a separate spec task and is not ported here.

pub mod campplus;
pub mod conformer;
pub mod flow;
pub mod hift;
pub mod s3gen;
pub mod t3;
pub mod voice_encoder;

#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::s3::{self, S3TokenizerV2};
use crate::music::minimax_music3::rng;
use crate::{quant, Result, SpeechError};

use self::campplus::{CampPlus, CampPlusConfig};
use self::flow::{ChatterboxFlow, FlowConfig};
use self::hift::{Hift, HiftConfig, SineDraws};
use self::t3::{T3Cond, T3Config, T3GenParams, T3 as T3Generator};
use self::voice_encoder::{VoiceEncConfig, VoiceEncoder};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// Sample rate of the speech tokenizer band.
pub const S3_SR: u32 = 16_000;
/// Sample rate of the vocoder band and the output audio.
pub const S3GEN_SR: u32 = 24_000;
/// Size of the speech token vocabulary; tokens at or above this are
/// specials (start 6561, stop 6562).
pub const SPEECH_VOCAB_SIZE: i32 = 6561;

/// The family `ModelConfig` (`chatterbox/config.py::ModelConfig` plus
/// the submodule geometry keys the fixture overrides).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub t3: T3Config,
    /// `text_preprocessing`: "legacy" (lowercase-normalizing) or the v3
    /// "NFKD,fullcase" (case-preserving); only the `capitalize` flag of
    /// [`punc_norm`] is portable.
    pub text_preprocessing: String,
    pub s3_sr: u32,
    pub s3gen_sr: u32,
    pub enc_cond_len: usize,
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
            text_preprocessing: "legacy".to_string(),
            s3_sr: S3_SR,
            s3gen_sr: S3GEN_SR,
            enc_cond_len: 6 * S3_SR as usize,
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
    /// The multilingual checkpoints widen the text vocabulary; the
    /// `t3_config` key carries the T3 fields, `multilingual` alone
    /// selects the 2454-token English-plus-accent table (v2 shape).
    pub fn from_json(value: &Value) -> Result<Self> {
        let defaults = ModelConfig::default();
        let multilingual = value
            .get("multilingual")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            || value.get("vocab_size").and_then(Value::as_u64) == Some(2454);
        let t3_value = value.get("t3_config").cloned().unwrap_or(Value::Null);
        let t3 = match &t3_value {
            Value::Object(_) => {
                let mut t3 = T3Config::from_json_with_defaults(&t3_value, defaults.t3.clone())?;
                if multilingual && t3.text_tokens_dict_size == defaults.t3.text_tokens_dict_size {
                    // The older converted multilingual checkpoints only
                    // recorded `multilingual`; widen to the v2 table.
                    t3.text_tokens_dict_size = 2454;
                }
                t3
            }
            _ if multilingual => {
                let mut t3 = defaults.t3.clone();
                t3.text_tokens_dict_size = 2454;
                t3
            }
            _ => defaults.t3.clone(),
        };
        let text_preprocessing = match value.get("text_preprocessing") {
            None | Some(Value::Null) => {
                // v3 checkpoints default to case-preserving preprocessing.
                if value.get("t3_model").and_then(Value::as_str) == Some("v3") {
                    "NFKD,fullcase".to_string()
                } else {
                    defaults.text_preprocessing
                }
            }
            Some(v) => v
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| SpeechError::BadConfig {
                    field: "text_preprocessing".to_string(),
                    why: "expected a string".to_string(),
                })?,
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
            text_preprocessing,
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

/// The S3Gen reference conditioning (`S3Token2Mel.embed_ref` result).
#[derive(Debug, Clone)]
pub struct GenCond {
    /// Prompt speech tokens `[1, T]` flattened.
    pub prompt_token: Vec<i32>,
    /// Prompt mel frames, row-major `[T_mel, 80]`.
    pub prompt_feat: Vec<f32>,
    /// CAMPPlus speaker embedding `[192]`.
    pub embedding: Vec<f32>,
}

/// The prepared conditionals for one voice (`chatterbox.Conditionals`).
#[derive(Debug, Clone)]
pub struct Conditionals {
    pub t3: T3Cond,
    pub gen: GenCond,
}

/// One generated result: the raw T3 id stream, the cropped speech
/// tokens, and the decoded 24 kHz waveform.
#[derive(Debug, Clone)]
pub struct Generated {
    pub ids: Vec<i32>,
    pub speech_tokens: Vec<i32>,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Generation options with the reference `generate` defaults.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub exaggeration: f32,
    pub t3: T3GenParams,
    /// Seed for the T3 categorical draws (unused when temperature is 0).
    pub seed: u64,
}

impl Default for GenerateParams {
    fn default() -> Self {
        GenerateParams {
            exaggeration: 0.5,
            t3: T3GenParams::default(),
            seed: 0,
        }
    }
}

/// Why the text stream carries no usable speech tokens.
#[derive(Debug, Clone)]
pub enum GenerateError {
    /// The T3 stopped on its first pick: no speech frames.
    NoSpeechTokens,
}

/// The loaded family model.
pub struct Model {
    pub config: ModelConfig,
    pub t3: T3Generator,
    pub flow: ChatterboxFlow,
    pub hift: Hift,
    pub campplus: CampPlus,
    pub ve: VoiceEncoder,
    /// The S3 speech tokenizer (`_s3_tokenizer`), required only by
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
    /// reference prefixes (`t3.`, `flow.`, `mel2wav.`,
    /// `speaker_encoder.`, `ve.`). The T3 backbone reopens the file
    /// through the shared LM `Weights` loader.
    pub fn load(path: &Path, config: &ModelConfig) -> Result<Self> {
        let file = SafetensorsFile::open(path).map_err(SpeechError::from)?;
        let t3 = T3Generator::load(path, &file, &config.t3)?;
        let flow = ChatterboxFlow::load(&config.flow, &file, "flow")?;
        let hift = Hift::load(&config.hift, &file, "mel2wav")?;
        let campplus = CampPlus::load(&file, "speaker_encoder.", config.campplus.clone())?;
        let ve = VoiceEncoder::load(&file, "ve.", config.ve.clone())?;
        Ok(Model {
            config: config.clone(),
            t3,
            flow,
            hift,
            campplus,
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

    /// Builds the exact reference text prompt: `[start_text_token]
    /// text [stop_text_token]`. The caller tokenizes (see
    /// [`punc_norm`] for the portable normalization and
    /// `ModelConfig::text_preprocessing` for the case convention).
    pub fn prepare_text_tokens(&self, text_ids: &[i32]) -> Vec<i32> {
        let mut ids = Vec::with_capacity(text_ids.len() + 2);
        ids.push(self.config.t3.start_text_token);
        ids.extend_from_slice(text_ids);
        ids.push(self.config.t3.stop_text_token);
        ids
    }

    /// `S3Token2Mel.embed_ref` + the T3 conditioning token path of
    /// `prepare_conditionals`, from the three reference wavebands:
    /// `wav_16k_full` (the voice-encoder band, up to `enc_cond_len`),
    /// `wav_24k` (the S3Gen mel band, up to `dec_cond_len`), and
    /// `wav_16k_from_24k` (the S3Gen token band). `exaggeration` feeds
    /// the emotion scalar. Requires the S3 tokenizer.
    pub fn prepare_conditionals(
        &mut self,
        wav_16k_full: &[f32],
        wav_24k: &[f32],
        wav_16k_from_24k: &[f32],
        exaggeration: f32,
    ) -> Result<Conditionals> {
        let tokenizer = self
            .s3_tokenizer
            .as_ref()
            .ok_or_else(|| SpeechError::Input {
                why: "prepare_conditionals requires the S3 speech tokenizer".to_string(),
            })?;
        let n_mels = self.config.flow.output_size;
        // --- S3Gen band: mel (24 kHz), tokens (16 kHz resampled) ---
        let wav_24k = &wav_24k[..wav_24k.len().min(self.config.dec_cond_len)];
        let mut mel = s3gen::prompt_mel(wav_24k, n_mels, self.config.s3gen_sr)?;
        let mel_len = mel.len();
        let mut log_mel = s3::log_mel_spectrogram(wav_16k_from_24k, 128)?;
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

        // Speaker embedding: CAMPPlus over the 16 kHz resample.
        let fbank_rows = campplus::kaldi_fbank(wav_16k_from_24k)?;
        let frames = fbank_rows.len();
        let feat_dim = fbank_rows.first().map_or(0, Vec::len);
        let mut fbank = Vec::with_capacity(frames * feat_dim);
        for row in &fbank_rows {
            fbank.extend_from_slice(row);
        }
        // Per-feature mean removal over frames.
        for c in 0..feat_dim {
            let mean = (0..frames).map(|f| fbank[f * feat_dim + c]).sum::<f32>() / frames as f32;
            for f in 0..frames {
                fbank[f * feat_dim + c] -= mean;
            }
        }
        let embedding = self.campplus.inference(&fbank, frames);

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
                emotion_adv: exaggeration,
            },
            gen: GenCond {
                prompt_token: tokens,
                prompt_feat,
                embedding,
            },
        })
    }

    /// Runs the full pipeline: text prompt wrap, T3 generation, the
    /// invalid-token crop, the S3Gen flow, the HiFT vocoder, and the
    /// 20 ms fade-in. `sine_draws` injects the vocoder sine randomness
    /// (`None` = deterministic zeros; see [`hift::SineDraws`]).
    pub fn generate(
        &mut self,
        text_ids: &[i32],
        conds: &Conditionals,
        params: &GenerateParams,
        sine_draws: Option<&SineDraws>,
    ) -> Result<Generated> {
        let prompt = self.prepare_text_tokens(text_ids);
        let mut keys = rng::KeySequence::new(params.seed);
        let (ids, _stopped) = self
            .t3
            .inference(&conds.t3, &prompt, &params.t3, &mut keys)?;
        let speech_tokens = self.sanitize_speech_tokens(&ids)?;
        if speech_tokens.is_empty() {
            return Err(SpeechError::Input {
                why: "generation stopped with no speech tokens between the start and stop markers"
                    .to_string(),
            });
        }
        let mel_rows = self.flow.infer(
            &speech_tokens,
            &conds.gen.prompt_token,
            &conds.gen.prompt_feat,
            &conds.gen.embedding,
            self.flow.n_timesteps(),
        )?;
        let mut samples = self.hift.infer(&mel_rows, sine_draws)?;
        hift::trim_fade(&mut samples);
        Ok(Generated {
            ids,
            speech_tokens,
            samples,
            sample_rate: S3GEN_SR,
        })
    }

    /// `drop_invalid_tokens` + the `>= SPEECH_VOCAB_SIZE` filter: take
    /// the stream between the first start marker (exclusive) and the
    /// first stop marker (exclusive), then drop the specials keeping
    /// order.
    fn sanitize_speech_tokens(&self, ids: &[i32]) -> Result<Vec<i32>> {
        let cropped = drop_invalid_tokens(ids);
        Ok(cropped
            .into_iter()
            .filter(|&t| t < SPEECH_VOCAB_SIZE)
            .collect())
    }
}

/// `chatterbox.drop_invalid_tokens`: the stream strictly between the
/// first SOS (6561) and the first EOS (6562); the stream starts with
/// SOS by construction, so the crop yields the speech content only.
pub fn drop_invalid_tokens(ids: &[i32]) -> Vec<i32> {
    let sos = SPEECH_VOCAB_SIZE;
    let eos = SPEECH_VOCAB_SIZE + 1;
    let start = ids
        .iter()
        .position(|&t| t == sos)
        .map_or(0, |index| index + 1);
    let end = ids.iter().position(|&t| t == eos).unwrap_or(ids.len());
    if start >= end {
        return Vec::new();
    }
    ids[start..end].to_vec()
}

/// `chatterbox.punc_norm`, the portable text cleanup run before
/// tokenization. `capitalize` mirrors the reference flag: the legacy
/// preprocessing capitalizes the first letter, the v3
/// `NFKD,fullcase` convention does not.
pub fn punc_norm(text: &str, capitalize: bool) -> String {
    if text.is_empty() {
        return "You need to add some text for me to talk.".to_string();
    }
    let mut out = text.to_string();
    if capitalize {
        let mut chars = out.chars();
        if let Some(first) = chars.next() {
            if first.is_lowercase() {
                let upper: String = first.to_uppercase().collect();
                out = format!("{upper}{}", &out[first.len_utf8()..]);
            }
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
        ("...", ", "),
        ("\u{2026}", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
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
    let enders = [
        '.', '!', '?', '-', ',', '\u{3001}', '\u{ff0c}', '\u{3002}', '\u{ff1f}', '\u{ff01}',
    ];
    let needs_stop = trimmed
        .chars()
        .last()
        .map_or(true, |c| !enders.contains(&c));
    if needs_stop {
        format!("{trimmed}.")
    } else {
        trimmed.to_string()
    }
}
