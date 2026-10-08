//! Qwen2-Audio offline transcription: whisper-family log-mel frontend,
//! attention-pooling audio encoder, linear projector, and a Qwen2 language
//! model over a spliced audio-and-text prompt.
//!
//! Reference: `mlx_audio/stt/models/qwen2_audio/` (qwen2_audio.py,
//! config.py) at mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`.
//!
//! This port covers the offline single-audio transcription path of
//! upstream `generate`: 16 kHz mono audio padded to 30 seconds, the inline
//! log-mel frontend (400-point FFT, 160 hop, 128 HTK-mel bands, log10 with
//! a 1e-10 floor, peak clamp at max - 8, `(x + 4) / 4`), the conv +
//! transformer tower with non-causal pair averaging, the projector, the
//! default chat prompt with 750 `<|AUDIO|>` placeholders spliced with the
//! projected features, then a causal prefill and greedy decode until
//! `<|im_end|>`. Multi-audio batching, streaming, and the general
//! audio-understanding task prompts are refused; see the family README.
//!
//! Two divergence notes (both documented in the README): the reference
//! pipeline evaluates the tower and the spliced embeddings in bfloat16
//! while this port computes in f32 from the same bf16-rounded weights, and
//! the 400-point real FFT runs as a plain f64 DFT instead of numpy's
//! pocketfft. Token ids stay signed 32-bit at crate boundaries.

pub(crate) mod encoder;
pub(crate) mod language_model;

use std::fs;
use std::path::Path;

use serde_json::Value;
use tokenizers::AddedToken;
use turbospark_model_io::safetensors::SafetensorsFile;
use turbospark_tokenizer::Tokenizer;

use crate::nn::bad_config;
use crate::quant::QuantScheme;
use crate::{Result, SpeechError};

use encoder::{AudioTower, Projector};
use language_model::LanguageModel;

/// Immutable Hugging Face checkpoint profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen2AudioProfile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

/// The pinned profile: smoke-verified upstream on the shared reference
/// WAV (mlx-audio 0.5.7 returned the expected phrase).
pub const QWEN2_AUDIO_7B_INSTRUCT_4BIT: Qwen2AudioProfile = Qwen2AudioProfile {
    name: "Qwen2-Audio-7B-Instruct 4-bit",
    repository: "mlx-community/Qwen2-Audio-7B-Instruct-4bit",
    revision: "c65570002626f41b4dc08b7b54f42f99f3e82e7f",
};

// Frontend constants, pinned by the upstream `_init_mel_constants` and
// `_extract_features`.
/// FFT size per frame.
const N_FFT: usize = 400;
/// Hop between frames.
const HOP: usize = 160;
/// Input sample rate.
const SAMPLE_RATE: usize = 16_000;
/// One fixed 30-second window of PCM; audio is zero padded or truncated.
const MAX_SAMPLES: usize = 480_000;
/// Mel bands.
const N_MELS: usize = 128;
/// Spectrum bins (N_FFT / 2 + 1).
const N_FREQS: usize = N_FFT / 2 + 1;
/// Frames per window after the centered STFT: `(480400 - 400) / 160 + 1`.
const MEL_FRAMES: usize = 3_001;
/// Audio embedding rows the tower emits (and `<|AUDIO|>` placeholders):
/// 1500 conv output frames pair-averaged to 750.
const NUM_AUDIO_TOKENS: usize = 750;
/// Center padding of the reflect-padded STFT.
const STFT_PAD: usize = N_FFT / 2;

/// Default transcription instruction (upstream `_build_prompt` default).
pub const DEFAULT_USER_PROMPT: &str = "Please transcribe the speech.";

/// Max generated tokens (upstream `max_tokens` default).
const MAX_NEW_TOKENS: usize = 4096;

/// Parsed and pinned checkpoint configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen2AudioConfig {
    pub audio: AudioEncoderConfig,
    pub text: TextDecoderConfig,
    pub audio_token_id: u32,
    pub quantization: QuantScheme,
}

/// Audio tower geometry, pinned to the verified variant. A key that is
/// absent takes the pinned default (upstream `EncoderConfig.from_dict`
/// fills the same defaults); a key that is present must equal the pinned
/// value.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioEncoderConfig {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
}

/// Qwen2 decoder geometry with the upstream `TextConfig` defaults. The
/// pinned checkpoint's `text_config` omits most fields, so the defaults
/// are load-bearing.
#[derive(Debug, Clone, PartialEq)]
pub struct TextDecoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

fn unsupported(why: impl Into<String>) -> SpeechError {
    SpeechError::Unsupported { why: why.into() }
}

/// Reads an integer field that must equal `pinned` when present; missing
/// takes `pinned` (the upstream dataclass default).
fn pinned_int(value: &Value, key: &str, pinned: usize) -> Result<usize> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(pinned),
        Some(found) => {
            let found = found
                .as_u64()
                .and_then(|number| usize::try_from(number).ok())
                .ok_or_else(|| bad_config(key, "must be a non-negative integer"))?;
            if found == pinned {
                Ok(pinned)
            } else {
                Err(bad_config(
                    key,
                    format!("expected the pinned value {pinned}, found {found}"),
                ))
            }
        }
    }
}

fn require(value: &Value, key: &str, expected: &str) -> Result<()> {
    if value.get(key).and_then(Value::as_str) == Some(expected) {
        Ok(())
    } else {
        Err(bad_config(
            key,
            format!("expected {expected:?}, found {:?}", value.get(key)),
        ))
    }
}

impl Qwen2AudioConfig {
    pub fn from_json(root: &Value) -> Result<Self> {
        require(root, "model_type", "qwen2_audio")?;
        if root
            .get("architectures")
            .and_then(Value::as_array)
            .is_none_or(|items| {
                !items
                    .iter()
                    .any(|item| item.as_str() == Some("Qwen2AudioForConditionalGeneration"))
            })
        {
            return Err(bad_config(
                "architectures",
                "expected the Qwen2AudioForConditionalGeneration architecture",
            ));
        }

        // Quantization is part of the pinned profile: refuse any other
        // scheme instead of silently mis-decoding.
        let quant = root
            .get("quantization")
            .ok_or_else(|| bad_config("quantization", "is missing from config.json"))?;
        let bits = quant
            .get("bits")
            .and_then(Value::as_u64)
            .ok_or_else(|| bad_config("quantization.bits", "must be an integer"))?;
        let group_size = quant
            .get("group_size")
            .and_then(Value::as_u64)
            .and_then(|number| usize::try_from(number).ok())
            .ok_or_else(|| bad_config("quantization.group_size", "must be a positive integer"))?;
        if bits != 4 || group_size != 64 {
            return Err(unsupported(format!(
                "only the verified 4-bit group-64 scheme is accepted, found {bits}-bit \
                 groups of {group_size}"
            )));
        }

        let audio_json = root
            .get("audio_config")
            .ok_or_else(|| bad_config("audio_config", "is missing"))?;
        require(audio_json, "model_type", "qwen2_audio_encoder")?;
        if let Some(activation) = audio_json.get("activation_function") {
            if *activation != Value::String("gelu".into()) {
                return Err(bad_config(
                    "audio_config.activation_function",
                    format!("expected \"gelu\", found {activation}"),
                ));
            }
        }
        let audio = AudioEncoderConfig {
            d_model: pinned_int(audio_json, "d_model", 1280)?,
            encoder_layers: pinned_int(audio_json, "encoder_layers", 32)?,
            encoder_attention_heads: pinned_int(audio_json, "encoder_attention_heads", 20)?,
            encoder_ffn_dim: pinned_int(audio_json, "encoder_ffn_dim", 5120)?,
            num_mel_bins: pinned_int(audio_json, "num_mel_bins", 128)?,
            max_source_positions: pinned_int(audio_json, "max_source_positions", 1500)?,
        };

        let text_json = root
            .get("text_config")
            .ok_or_else(|| bad_config("text_config", "is missing"))?;
        require(text_json, "model_type", "qwen2")?;
        if let Some(hidden_act) = text_json.get("hidden_act") {
            if *hidden_act != Value::String("silu".into()) {
                return Err(bad_config(
                    "text_config.hidden_act",
                    format!("expected \"silu\", found {hidden_act}"),
                ));
            }
        }
        if text_json
            .get("rope_scaling")
            .is_some_and(|value| !value.is_null())
        {
            return Err(unsupported(
                "rope_scaling is refused; the pinned decoder uses plain RoPE",
            ));
        }
        if text_json.get("use_sliding_window").and_then(Value::as_bool) == Some(true) {
            return Err(unsupported("sliding-window attention is refused"));
        }
        let tie_word_embeddings = text_json
            .get("tie_word_embeddings")
            .map(|value| {
                value.as_bool().ok_or_else(|| {
                    bad_config("text_config.tie_word_embeddings", "must be a boolean")
                })
            })
            .transpose()?
            .unwrap_or(false);
        if tie_word_embeddings {
            return Err(unsupported(
                "only tie_word_embeddings=false decoders are verified; the pinned checkpoint \
                 carries a separate quantized lm_head",
            ));
        }
        if let Some(attention_bias) = text_json.get("attention_bias") {
            if attention_bias.as_bool() != Some(true) {
                return Err(unsupported(
                    "only attention_bias=true decoders are verified; the attention projections \
                     carry plain biases",
                ));
            }
        }
        let text = TextDecoderConfig {
            hidden_size: pinned_int(text_json, "hidden_size", 4096)?,
            intermediate_size: pinned_int(text_json, "intermediate_size", 11008)?,
            num_hidden_layers: pinned_int(text_json, "num_hidden_layers", 32)?,
            num_attention_heads: pinned_int(text_json, "num_attention_heads", 32)?,
            num_key_value_heads: pinned_int(text_json, "num_key_value_heads", 32)?,
            vocab_size: pinned_int(text_json, "vocab_size", 156032)?,
            rms_norm_eps: text_json
                .get("rms_norm_eps")
                .map(|value| {
                    value
                        .as_f64()
                        .filter(|eps| *eps > 0.0)
                        .ok_or_else(|| bad_config("text_config.rms_norm_eps", "must be positive"))
                })
                .transpose()?
                .unwrap_or(1.0e-5) as f32,
            rope_theta: text_json
                .get("rope_theta")
                .map(|value| {
                    value
                        .as_f64()
                        .filter(|theta| *theta > 0.0)
                        .ok_or_else(|| bad_config("text_config.rope_theta", "must be positive"))
                })
                .transpose()?
                .unwrap_or(10_000.0) as f32,
        };
        if text.hidden_size % text.num_attention_heads != 0
            || text.num_attention_heads % text.num_key_value_heads != 0
        {
            return Err(bad_config(
                "text_config",
                "attention head geometry is inconsistent",
            ));
        }

        // config.json carries the id under `audio_token_index`; upstream
        // reads `audio_token_id` and defaults to 151646.
        let audio_token_id = root
            .get("audio_token_id")
            .or_else(|| root.get("audio_token_index"))
            .and_then(Value::as_u64)
            .unwrap_or(151_646);
        if audio_token_id != 151_646 {
            return Err(bad_config(
                "audio_token_index",
                format!("expected the pinned <|AUDIO|> id 151646, found {audio_token_id}"),
            ));
        }

        Ok(Self {
            audio,
            text,
            audio_token_id: 151_646,
            quantization: QuantScheme {
                bits: 4,
                group_size: 64,
            },
        })
    }
}

/// The inline HTK-mel filterbank of the upstream `_init_mel_constants`:
/// `mel = 2595 log10(1 + f / 700)` edges on integer bins
/// `floor((N_FFT + 1) * hz / sample_rate)`, no area normalization.
pub(crate) fn mel_filterbank() -> Vec<f32> {
    let fmax = SAMPLE_RATE as f64 / 2.0;
    let mel_max = 2595.0 * (1.0 + fmax / 700.0).log10();
    let step = mel_max / (N_MELS + 1) as f64;
    let hz_points: Vec<f64> = (0..N_MELS + 2)
        .map(|index| {
            if index == N_MELS + 1 {
                mel_max
            } else {
                index as f64 * step
            }
        })
        .map(|mel| 700.0 * (10.0f64.powf(mel / 2595.0) - 1.0))
        .collect();
    let bins: Vec<i64> = hz_points
        .iter()
        .map(|&hz| ((N_FFT + 1) as f64 * hz / SAMPLE_RATE as f64).floor() as i64)
        .collect();
    let mut bank = vec![0.0f32; N_MELS * N_FREQS];
    for mel in 1..=N_MELS {
        let (lower, center, upper) = (bins[mel - 1], bins[mel], bins[mel + 1]);
        if center > lower {
            for bin in lower..center {
                bank[(mel - 1) * N_FREQS + bin as usize] =
                    ((bin - lower) as f64 / (center - lower) as f64) as f32;
            }
        }
        if upper > center {
            for bin in center..upper {
                bank[(mel - 1) * N_FREQS + bin as usize] =
                    ((upper - bin) as f64 / (upper - center) as f64) as f32;
            }
        }
    }
    bank
}

/// Real DFT of one 400-point frame: returns the power spectrum
/// `[N_FREQS]`. numpy's rfft is f64 pocketfft; the plain f64 DFT matches
/// it far inside the fixture gates and keeps the frontend portable.
fn frame_power(frame: &[f64], cos_table: &[f64], sin_table: &[f64]) -> [f64; N_FREQS] {
    let mut power = [0.0f64; N_FREQS];
    for (bin, slot) in power.iter_mut().enumerate() {
        let table_base = bin * N_FFT;
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (n, &sample) in frame.iter().enumerate() {
            re += sample * cos_table[table_base + n];
            im -= sample * sin_table[table_base + n];
        }
        *slot = re * re + im * im;
    }
    power
}

/// Precomputed `exp(-2 pi i k n / N)` twiddle tables for the f64 DFT.
fn dft_tables() -> (Vec<f64>, Vec<f64>) {
    let mut cos_table = vec![0.0f64; N_FREQS * N_FFT];
    let mut sin_table = vec![0.0f64; N_FREQS * N_FFT];
    for bin in 0..N_FREQS {
        for n in 0..N_FFT {
            let angle = -2.0 * std::f64::consts::PI * (bin * n) as f64 / N_FFT as f64;
            cos_table[bin * N_FFT + n] = angle.cos();
            sin_table[bin * N_FFT + n] = angle.sin();
        }
    }
    (cos_table, sin_table)
}

/// The inline log-mel frontend of the upstream `_extract_features`.
///
/// Pads or truncates to the fixed 30-second window, reflect-pads by
/// `N_FFT / 2`, takes the power spectrum with a periodic Hann window,
/// projects onto the inline filterbank, then applies `log10` with a 1e-10
/// floor, the global peak clamp at `max - 8`, and `(x + 4) / 4`. Returns
/// band-major features `[num_mels, frames]` (the transposed layout the
/// tower consumes) and the fixed audio token count.
pub(crate) fn extract_features(samples: &[f32]) -> Result<(Vec<f32>, usize)> {
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(SpeechError::Input {
            why: "waveform contains non-finite samples".into(),
        });
    }
    let mut audio = vec![0.0f32; MAX_SAMPLES];
    let copied = samples.len().min(MAX_SAMPLES);
    audio[..copied].copy_from_slice(&samples[..copied]);

    // Reflect padding about both edges, edge value excluded (numpy
    // `mode="reflect"`).
    let mut padded = vec![0.0f32; MAX_SAMPLES + 2 * STFT_PAD];
    for offset in 0..STFT_PAD {
        padded[offset] = audio[STFT_PAD - offset];
        padded[MAX_SAMPLES + STFT_PAD + offset] = audio[MAX_SAMPLES - 2 - offset];
    }
    padded[STFT_PAD..STFT_PAD + MAX_SAMPLES].copy_from_slice(&audio);

    let n_frames = (padded.len() - N_FFT) / HOP + 1;
    let bank = mel_filterbank();
    let (cos_table, sin_table) = dft_tables();
    let mut window = vec![0.0f64; N_FFT];
    for (index, slot) in window.iter_mut().enumerate() {
        *slot = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * index as f64 / N_FFT as f64).cos());
    }

    // Band-major output [num_mels, frames]; the reference transposes the
    // frame-major log-mel into the tower's [B, n_mels, T] input.
    let mut log_mel = vec![0.0f64; n_frames * N_MELS];
    let mut frame = vec![0.0f64; N_FFT];
    let mut mel = vec![0.0f64; N_MELS];
    for t in 0..n_frames {
        let base = t * HOP;
        for (offset, slot) in frame.iter_mut().enumerate() {
            *slot = f64::from(padded[base + offset]) * window[offset];
        }
        let power = frame_power(&frame, &cos_table, &sin_table);
        for (m, slot) in mel.iter_mut().enumerate() {
            let row = &bank[m * N_FREQS..(m + 1) * N_FREQS];
            let mut sum = 0.0f64;
            for (bin, &weight) in row.iter().enumerate() {
                sum += power[bin] * f64::from(weight);
            }
            *slot = sum.max(1e-10).log10();
        }
        let row_base = t * N_MELS;
        log_mel[row_base..row_base + N_MELS].copy_from_slice(&mel);
    }

    let peak = log_mel.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let clamp = peak - 8.0;
    let mut features = vec![0.0f32; n_frames * N_MELS];
    for (t, frame_values) in log_mel.chunks_exact(N_MELS).enumerate() {
        for (m, &value) in frame_values.iter().enumerate() {
            let normalized = (value.max(clamp) + 4.0) / 4.0;
            features[m * n_frames + t] = normalized as f32;
        }
    }
    Ok((features, NUM_AUDIO_TOKENS))
}

/// Assembles the offline chat prompt string for one audio window,
/// mirroring the upstream `_build_prompt` content and the pinned chat
/// template (which prepends the default system message for a non-system
/// first message and appends the generation prompt).
pub fn build_prompt_string(num_audio_tokens: usize, user_prompt: Option<&str>) -> String {
    let user_prompt = user_prompt.unwrap_or(DEFAULT_USER_PROMPT);
    let audio = format!(
        "Audio 1: <|audio_bos|>{}<|audio_eos|>",
        "<|AUDIO|>".repeat(num_audio_tokens)
    );
    format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
         <|im_start|>user\n{audio}\n{user_prompt}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

/// Loads the checkpoint tokenizer: `tokenizer.json` plus every token from
/// `tokenizer_config.json added_tokens_decoder` that the fast file is
/// missing (the pinned tokenizer.json carries only the three Qwen chat
/// tokens; the audio and timestamp control tokens exist only in the slow
/// config). Added in id order, so the contiguity of the pinned added-token
/// block makes each id stick. Also resolves the generation EOS id from
/// `eos_token`.
fn load_tokenizer(model_dir: &Path) -> Result<(Tokenizer, u32)> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|error| bad_config("tokenizer.json", error))?;
    let config: Value = serde_json::from_slice(
        &fs::read(model_dir.join("tokenizer_config.json")).map_err(|error| SpeechError::Input {
            why: format!("cannot read tokenizer_config.json: {error}"),
        })?,
    )
    .map_err(|error| bad_config("tokenizer_config.json", error.to_string()))?;
    require(&config, "tokenizer_class", "Qwen2Tokenizer")?;

    let decoder = config
        .get("added_tokens_decoder")
        .and_then(Value::as_object)
        .ok_or_else(|| bad_config("tokenizer_config.json", "added_tokens_decoder is missing"))?;
    let mut entries = decoder
        .iter()
        .map(|(raw_id, entry)| {
            let id = raw_id
                .parse::<u32>()
                .map_err(|error| bad_config("tokenizer_config.json.added_tokens_decoder", error))?;
            let content = entry
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    bad_config(
                        "tokenizer_config.json.added_tokens_decoder",
                        format!("token {id} has no content"),
                    )
                })?
                .to_owned();
            let field = |name| entry.get(name).and_then(Value::as_bool).unwrap_or(false);
            let token = AddedToken::from(content.clone(), field("special"))
                .single_word(field("single_word"))
                .lstrip(field("lstrip"))
                .rstrip(field("rstrip"))
                .normalized(field("normalized"));
            Ok((id, content, token, field("special")))
        })
        .collect::<Result<Vec<_>>>()?;
    entries.sort_by_key(|(id, _, _, _)| *id);
    for (id, content, token, special) in entries {
        if tokenizer.token_to_id(&content) == Some(id) {
            continue;
        }
        if special {
            tokenizer.add_special_tokens(&[token]);
        } else {
            tokenizer.add_tokens(&[token]);
        }
        if tokenizer.token_to_id(&content) != Some(id) {
            return Err(bad_config(
                "tokenizer_config.json.added_tokens_decoder",
                format!("added token {content} did not retain id {id}"),
            ));
        }
    }

    let eos_token = config
        .get("eos_token")
        .and_then(Value::as_str)
        .ok_or_else(|| bad_config("tokenizer_config.json.eos_token", "is missing"))?;
    let eos_id = tokenizer.token_to_id(eos_token).ok_or_else(|| {
        bad_config(
            "tokenizer_config.json.eos_token",
            format!("{eos_token} is not in the vocabulary"),
        )
    })?;
    Ok((tokenizer, eos_id))
}

/// Encoded audio features for one window: the band-major mel input, the
/// pooled tower output, and the projected rows the prompt splices in.
pub struct AudioFeatures {
    /// `[num_mels, frames]` band-major log-mel input to the tower.
    pub input_features: Vec<f32>,
    /// `[frames, d_model]` pooled tower output (the projector input).
    pub tower_output: Vec<f32>,
    /// `[frames, hidden]` projected features.
    pub projected: Vec<f32>,
    pub frames: usize,
    pub hidden: usize,
}

/// The audio path only: mel frontend, tower, and projector. Kept separate
/// from the language model so a caller can encode audio and drop the tower
/// (about 0.6B f32 parameters) before the much larger decoder loads.
pub struct Qwen2AudioAudioTower {
    config: Qwen2AudioConfig,
    tower: AudioTower,
    projector: Projector,
}

impl Qwen2AudioAudioTower {
    /// Loads the tower and projector from an already-downloaded model
    /// folder. Never downloads.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config = load_config(model_dir)?;
        let shards = open_checkpoint(model_dir)?;
        let tower = AudioTower::load(
            &shards,
            config.audio.d_model,
            config.audio.encoder_layers,
            config.audio.encoder_attention_heads,
            config.audio.encoder_ffn_dim,
            config.audio.num_mel_bins,
            config.audio.max_source_positions,
        )?;
        let projector = Projector::load(&shards, config.audio.d_model, config.text.hidden_size)?;
        Ok(Self {
            config,
            tower,
            projector,
        })
    }

    pub fn profile(&self) -> Qwen2AudioProfile {
        QWEN2_AUDIO_7B_INSTRUCT_4BIT
    }

    pub fn config(&self) -> &Qwen2AudioConfig {
        &self.config
    }

    /// Runs the frontend, tower, and projector for one mono 16 kHz window.
    pub fn encode(&self, samples: &[f32]) -> Result<AudioFeatures> {
        let (input_features, frames) = extract_features(samples)?;
        let tower_output = self.tower.forward(&input_features, MEL_FRAMES);
        let projected = self.projector.forward(&tower_output, frames);
        Ok(AudioFeatures {
            input_features,
            tower_output,
            projected,
            frames,
            hidden: self.config.text.hidden_size,
        })
    }
}

/// The language-model path only: the Qwen2 decoder, the checkpoint
/// tokenizer, and the prompt/splice/decode protocol.
pub struct Qwen2AudioLanguageModel {
    config: Qwen2AudioConfig,
    lm: LanguageModel,
    tokenizer: Tokenizer,
    eos_token_id: u32,
}

impl Qwen2AudioLanguageModel {
    /// Loads the decoder from an already-downloaded model folder. The
    /// dequantized f32 weights need roughly 31 GB; load it after the audio
    /// tower has been used and dropped.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config = load_config(model_dir)?;
        let shards = open_checkpoint(model_dir)?;
        let lm = LanguageModel::load(
            &shards,
            "language_model",
            config.text.hidden_size,
            config.text.intermediate_size,
            config.text.num_hidden_layers,
            config.text.num_attention_heads,
            config.text.num_key_value_heads,
            config.text.vocab_size,
            config.text.rms_norm_eps,
            config.text.rope_theta,
            config.quantization,
        )?;
        let (tokenizer, eos_token_id) = load_tokenizer(model_dir)?;
        Ok(Self {
            config,
            lm,
            tokenizer,
            eos_token_id,
        })
    }

    pub fn profile(&self) -> Qwen2AudioProfile {
        QWEN2_AUDIO_7B_INSTRUCT_4BIT
    }

    pub fn config(&self) -> &Qwen2AudioConfig {
        &self.config
    }

    /// Token id of the `<|AUDIO|>` placeholder.
    pub fn audio_token_id(&self) -> u32 {
        self.config.audio_token_id
    }

    /// Builds the offline prompt token ids for one audio window and the
    /// positions carrying audio features (mirrors `_build_prompt` with the
    /// default instruction).
    pub fn build_prompt(
        &self,
        num_audio_tokens: usize,
        user_prompt: Option<&str>,
    ) -> Result<(Vec<i32>, Vec<usize>)> {
        if num_audio_tokens == 0 {
            return Err(SpeechError::Input {
                why: "no audio tokens to splice into the prompt".into(),
            });
        }
        let prompt = build_prompt_string(num_audio_tokens, user_prompt);
        let encoded = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|error| SpeechError::Input {
                why: format!("Qwen2-Audio prompt tokenization failed: {error}"),
            })?;
        let token_ids = encoded
            .get_ids()
            .iter()
            .map(|&id| {
                i32::try_from(id).map_err(|_| SpeechError::Input {
                    why: "Qwen2-Audio prompt token id exceeds signed 32-bit range".into(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let audio_positions = token_ids
            .iter()
            .enumerate()
            .filter_map(|(position, &id)| {
                (id == self.config.audio_token_id as i32).then_some(position)
            })
            .collect::<Vec<_>>();
        if audio_positions.len() != num_audio_tokens {
            return Err(SpeechError::Input {
                why: format!(
                    "prompt has {} audio placeholders for {num_audio_tokens} encoded frames",
                    audio_positions.len()
                ),
            });
        }
        Ok((token_ids, audio_positions))
    }

    /// Merges text embeddings with the projected audio features at the
    /// audio positions (mirrors `get_input_embeddings`; audio positions
    /// contribute their feature row, other positions embed id 0 where the
    /// placeholder stood).
    pub fn splice(
        &self,
        prompt_ids: &[i32],
        audio_positions: &[usize],
        features: &AudioFeatures,
    ) -> Result<Vec<f32>> {
        let hidden = self.config.text.hidden_size;
        if features.projected.len() != features.frames * hidden {
            return Err(SpeechError::Tensor {
                name: "projected".into(),
                why: "feature width does not match the decoder hidden size".into(),
            });
        }
        let text_ids: Vec<i32> = prompt_ids
            .iter()
            .map(|&id| {
                if id == self.config.audio_token_id as i32 {
                    0
                } else {
                    id
                }
            })
            .collect();
        let mut embeds = self.lm.embed(&text_ids)?;
        for (index, &position) in audio_positions.iter().enumerate() {
            if position >= prompt_ids.len() || index >= features.frames {
                return Err(SpeechError::Input {
                    why: "audio position outside the prompt or beyond the encoded features".into(),
                });
            }
            let source = &features.projected[index * hidden..(index + 1) * hidden];
            embeds[position * hidden..(position + 1) * hidden].copy_from_slice(source);
        }
        Ok(embeds)
    }

    /// Greedy decode from pre-filled prompt embeddings until the tokenizer
    /// EOS (`<|im_end|>` for the pinned checkpoint); returns the generated
    /// ids with special tokens skipped during detokenization.
    pub fn generate_from_embeddings(&self, embeds: &[f32], rows: usize) -> Result<Vec<u32>> {
        let (last_hidden, mut cache) = self.lm.prefill(embeds, rows);
        let mut logits = self.lm.logits(&last_hidden);
        let mut next = crate::nn::argmax(&logits);
        let mut generated: Vec<u32> = Vec::new();
        for _ in 0..MAX_NEW_TOKENS {
            if next as u32 == self.eos_token_id {
                break;
            }
            let token = i32::try_from(next).map_err(|_| SpeechError::Input {
                why: "generated token id exceeds signed 32-bit range".into(),
            })?;
            generated.push(token as u32);
            let embedding = self.lm.embed(&[token])?;
            let hidden = self.lm.step(&embedding, &mut cache);
            logits = self.lm.logits(&hidden);
            next = crate::nn::argmax(&logits);
        }
        Ok(generated)
    }

    /// Detokenizes generated ids with special tokens skipped, matching the
    /// upstream `generate` text.
    pub fn decode_generated(&self, generated: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(generated, true)
            .map_err(|error| SpeechError::Input {
                why: format!("Qwen2-Audio output detokenization failed: {error}"),
            })
    }

    /// Transcribes spliced prompt embeddings end to end: prefill, greedy
    /// decode, detokenize.
    pub fn transcribe_embeddings(&self, embeds: &[f32], rows: usize) -> Result<String> {
        let generated = self.generate_from_embeddings(embeds, rows)?;
        self.decode_generated(&generated)
    }
}

/// Loaded Qwen2-Audio model (offline single-window path). Holds the audio
/// tower and the language model together; prefer the split
/// [`Qwen2AudioAudioTower`] plus [`Qwen2AudioLanguageModel`] flow when
/// resident memory matters.
pub struct Qwen2Audio {
    config: Qwen2AudioConfig,
    tower: Qwen2AudioAudioTower,
    language_model: Qwen2AudioLanguageModel,
}

impl Qwen2Audio {
    /// Loads the full pinned model from an already-downloaded model
    /// folder. Never downloads; refuses unverified config knobs.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config = load_config(model_dir)?;
        // Both halves re-read the config; the parse is deterministic and
        // cheap next to the weight loads.
        let tower = Qwen2AudioAudioTower::load(model_dir)?;
        let language_model = Qwen2AudioLanguageModel::load(model_dir)?;
        debug_assert_eq!(tower.config, config);
        debug_assert_eq!(language_model.config, config);
        Ok(Self {
            config,
            tower,
            language_model,
        })
    }

    pub fn profile(&self) -> Qwen2AudioProfile {
        QWEN2_AUDIO_7B_INSTRUCT_4BIT
    }

    pub fn config(&self) -> &Qwen2AudioConfig {
        &self.config
    }

    /// Transcribes one mono 16 kHz waveform through the offline path.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        let features = self.tower.encode(samples)?;
        let (prompt_ids, audio_positions) =
            self.language_model.build_prompt(features.frames, None)?;
        let embeds = self
            .language_model
            .splice(&prompt_ids, &audio_positions, &features)?;
        self.language_model
            .transcribe_embeddings(&embeds, prompt_ids.len())
    }
}

/// Reads and parses `config.json` from a model folder.
fn load_config(model_dir: &Path) -> Result<Qwen2AudioConfig> {
    let config_json =
        fs::read_to_string(model_dir.join("config.json")).map_err(|error| SpeechError::Input {
            why: format!("cannot read config.json: {error}"),
        })?;
    let root: Value = serde_json::from_str(&config_json)
        .map_err(|error| bad_config("config.json", error.to_string()))?;
    Qwen2AudioConfig::from_json(&root)
}

/// Opens the checkpoint shards: the `model.safetensors.index.json`
/// `weight_map` order when present, otherwise `weights.safetensors` (the
/// pinned single-file layout), otherwise `model.safetensors`.
fn open_checkpoint(model_dir: &Path) -> Result<Vec<SafetensorsFile>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let mut names: Vec<String> = Vec::new();
    if index_path.is_file() {
        let value: Value = serde_json::from_slice(
            &fs::read(&index_path)
                .map_err(|error| bad_config("model.safetensors.index.json", error))?,
        )
        .map_err(|error| bad_config("model.safetensors.index.json", error))?;
        let map = value
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| bad_config("model.safetensors.index.json", "weight_map missing"))?;
        for shard in map.values().filter_map(Value::as_str) {
            if !names.iter().any(|name| name == shard) {
                names.push(shard.to_owned());
            }
        }
        names.sort();
    } else if model_dir.join("weights.safetensors").is_file() {
        names.push("weights.safetensors".to_owned());
    } else {
        names.push("model.safetensors".to_owned());
    }
    names
        .iter()
        .map(|name| {
            SafetensorsFile::open(&model_dir.join(name)).map_err(|error| SpeechError::BadConfig {
                field: name.clone(),
                why: error.to_string(),
            })
        })
        .collect()
}

/// Natural log of the softmax value at `index`, for the per-step
/// confidence records the fixture carries.
#[cfg(test)]
pub(crate) fn log_softmax_top(logits: &[f32], index: usize) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&value| (value - max).exp()).sum();
    logits[index] - max - sum.ln()
}

#[cfg(test)]
mod tests {
    use super::{
        build_prompt_string, extract_features, mel_filterbank, Qwen2AudioAudioTower,
        Qwen2AudioConfig, Qwen2AudioLanguageModel, N_FREQS, N_MELS, QWEN2_AUDIO_7B_INSTRUCT_4BIT,
    };
    use serde_json::Value;
    use std::path::Path;

    const FIXTURE: &str = include_str!("../../../testdata/qwen2_audio_reference.json");

    fn load_fixture() -> Value {
        serde_json::from_str(FIXTURE).expect("valid qwen2_audio reference fixture")
    }

    #[derive(serde::Deserialize, Clone)]
    struct Spots {
        shape: Vec<usize>,
        rows: Vec<usize>,
        columns: Vec<usize>,
        values: Vec<Vec<f32>>,
    }

    /// Compares `[T, C]` spot rows against a row-major actual buffer and
    /// returns the worst absolute difference (the caller applies gates so
    /// one run measures every stage).
    fn compare_spots(actual: &[f32], spots: &Spots, label: &str) -> f32 {
        let columns = spots.shape[1];
        assert!(
            actual.len() >= spots.shape[0] * columns,
            "{label} size {} < {}",
            actual.len(),
            spots.shape[0] * columns
        );
        let mut worst = 0.0f32;
        for (row_index, &row) in spots.rows.iter().enumerate() {
            for (column_index, &column) in spots.columns.iter().enumerate() {
                let expected = spots.values[row_index][column_index];
                worst = worst.max((actual[row * columns + column] - expected).abs());
            }
        }
        worst
    }

    #[derive(serde::Deserialize)]
    struct VectorSpots {
        #[serde(default)]
        length: usize,
        indices: Vec<usize>,
        values: Vec<f32>,
    }

    /// 1-D variant of [`compare_spots`]; returns the worst difference.
    fn compare_vector(actual: &[f32], spots: &VectorSpots, label: &str) -> f32 {
        if spots.length > 0 {
            assert_eq!(actual.len(), spots.length, "{label} length");
        }
        let mut worst = 0.0f32;
        for (&index, &expected) in spots.indices.iter().zip(&spots.values) {
            worst = worst.max((actual[index] - expected).abs());
        }
        worst
    }

    #[test]
    fn profile_pin_is_immutable() {
        assert_eq!(
            QWEN2_AUDIO_7B_INSTRUCT_4BIT.repository,
            "mlx-community/Qwen2-Audio-7B-Instruct-4bit"
        );
        assert_eq!(
            QWEN2_AUDIO_7B_INSTRUCT_4BIT.revision,
            "c65570002626f41b4dc08b7b54f42f99f3e82e7f"
        );
    }

    #[test]
    fn fixture_provenance_pins_the_reference_run() {
        let fixture = load_fixture();
        assert_eq!(
            fixture["provenance"]["revision"],
            "c65570002626f41b4dc08b7b54f42f99f3e82e7f"
        );
        assert_eq!(
            fixture["provenance"]["source"],
            "mlx-audio qwen2_audio at commit e1b19b9054bf163f5d812221a54fcc346f1890e9"
        );
        assert_eq!(
            fixture["provenance"]["repository"],
            "mlx-community/Qwen2-Audio-7B-Instruct-4bit"
        );
        let wav = crate::wav::read_wav_f32(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("testdata/qwen3_forced_aligner_reference.wav"),
        )
        .expect("reference WAV loads");
        let digest = turbospark_model_io::hash_data(
            &std::fs::read(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("testdata/qwen3_forced_aligner_reference.wav"),
            )
            .expect("reference WAV bytes"),
        );
        assert_eq!(digest, fixture["provenance"]["audio_sha256"]);
        assert_eq!(wav.sample_rate, 16_000);
    }

    #[test]
    fn fixture_transcript_and_greedy_shape() {
        let fixture = load_fixture();
        assert_eq!(
            fixture["transcript"],
            "The quick brown fox jumps over the lazy dog."
        );
        let generated: Vec<i64> = fixture["generated_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_i64().unwrap())
            .collect();
        assert_eq!(
            generated,
            vec![785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13]
        );
        // 750 <|AUDIO|> placeholders out of 783 prompt tokens.
        assert_eq!(fixture["prompt"]["num_audio_tokens"], 750);
        assert_eq!(
            fixture["prompt"]["token_ids"].as_array().unwrap().len(),
            783
        );
        assert_eq!(
            fixture["prompt"]["audio_token_positions"]
                .as_array()
                .unwrap()
                .len(),
            750
        );
        // The first decision is confident: argmax 785 ("The") by a wide margin.
        assert_eq!(fixture["first_logits"]["argmax"], 785);
        assert_eq!(fixture["steps"].as_array().unwrap().len(), 10);
    }

    #[test]
    fn pinned_config_parses_and_refuses_unverified_knobs() {
        // Minimal replica of the pinned config.json (whose text_config
        // omits most geometry; the upstream defaults apply).
        let root = serde_json::json!({
            "model_type": "qwen2_audio",
            "architectures": ["Qwen2AudioForConditionalGeneration"],
            "audio_token_index": 151646,
            "quantization": {"group_size": 64, "bits": 4},
            "audio_config": {
                "model_type": "qwen2_audio_encoder",
                "num_mel_bins": 128,
                "encoder_layers": 32,
                "encoder_attention_heads": 20,
                "encoder_ffn_dim": 5120,
                "d_model": 1280,
                "activation_function": "gelu",
                "scale_embedding": false,
                "max_source_positions": 1500
            },
            "text_config": {
                "model_type": "qwen2",
                "bos_token_id": 151643,
                "eos_token_id": 151645,
                "intermediate_size": 11008,
                "max_position_embeddings": 8192,
                "rope_theta": 10000,
                "rms_norm_eps": 1e-05,
                "sliding_window": 32768,
                "vocab_size": 156032
            }
        });
        let config = Qwen2AudioConfig::from_json(&root).expect("pinned config parses");
        assert_eq!(config.audio.d_model, 1280);
        assert_eq!(config.audio.encoder_layers, 32);
        assert_eq!(config.text.hidden_size, 4096);
        assert_eq!(config.text.num_hidden_layers, 32);
        assert_eq!(config.text.num_attention_heads, 32);
        assert_eq!(config.text.num_key_value_heads, 32);
        assert_eq!(config.text.vocab_size, 156032);
        assert_eq!(config.text.rms_norm_eps, 1.0e-5);
        assert_eq!(config.text.rope_theta, 10_000.0);
        assert_eq!(config.audio_token_id, 151_646);
        assert_eq!(config.quantization.bits, 4);
        assert_eq!(config.quantization.group_size, 64);

        // Top-level refusals.
        for (field, value) in [
            ("model_type", Value::String("qwen3_audio".into())),
            ("audio_token_index", serde_json::json!(151647)),
        ] {
            let mut broken = root.clone();
            broken[field] = value;
            assert!(Qwen2AudioConfig::from_json(&broken).is_err(), "{field}");
        }
        let mut broken = root.clone();
        broken["architectures"] = serde_json::json!(["Qwen2AudioForConditionalGenerationXX"]);
        assert!(
            Qwen2AudioConfig::from_json(&broken).is_err(),
            "architectures"
        );

        // Quantization scheme refusals.
        for quant in [
            serde_json::json!({"group_size": 64, "bits": 8}),
            serde_json::json!({"group_size": 32, "bits": 4}),
        ] {
            let mut broken = root.clone();
            broken["quantization"] = quant;
            assert!(
                Qwen2AudioConfig::from_json(&broken).is_err(),
                "quantization"
            );
        }
        let mut broken = root.clone();
        broken.as_object_mut().unwrap().remove("quantization");
        assert!(
            Qwen2AudioConfig::from_json(&broken).is_err(),
            "missing quantization"
        );

        // Present-but-divergent audio geometry is refused; absent keys
        // default to the pinned values.
        let mut broken = root.clone();
        broken["audio_config"]["encoder_layers"] = serde_json::json!(24);
        assert!(
            Qwen2AudioConfig::from_json(&broken).is_err(),
            "encoder_layers"
        );
        let mut broken = root.clone();
        broken["audio_config"]["activation_function"] = Value::String("relu".into());
        assert!(
            Qwen2AudioConfig::from_json(&broken).is_err(),
            "activation_function"
        );
        let mut minimal_audio = root.clone();
        minimal_audio["audio_config"] = serde_json::json!({"model_type": "qwen2_audio_encoder"});
        let config = Qwen2AudioConfig::from_json(&minimal_audio).expect("defaults fill audio");
        assert_eq!(config.audio.d_model, 1280);
        assert_eq!(config.audio.max_source_positions, 1500);

        // Decoder knob refusals.
        let mut broken = root.clone();
        broken["text_config"]["tie_word_embeddings"] = Value::Bool(true);
        assert!(
            Qwen2AudioConfig::from_json(&broken).is_err(),
            "tie_word_embeddings"
        );
        let mut broken = root.clone();
        broken["text_config"]["attention_bias"] = Value::Bool(false);
        assert!(
            Qwen2AudioConfig::from_json(&broken).is_err(),
            "attention_bias"
        );
        let mut broken = root.clone();
        broken["text_config"]["use_sliding_window"] = Value::Bool(true);
        assert!(
            Qwen2AudioConfig::from_json(&broken).is_err(),
            "sliding window"
        );
        let mut broken = root.clone();
        broken["text_config"]["rope_scaling"] = serde_json::json!({"type": "yarn"});
        assert!(
            Qwen2AudioConfig::from_json(&broken).is_err(),
            "rope_scaling"
        );
        let mut broken = root.clone();
        broken["text_config"]["hidden_act"] = Value::String("gelu".into());
        assert!(Qwen2AudioConfig::from_json(&broken).is_err(), "hidden_act");
        let mut broken = root.clone();
        broken["text_config"]["hidden_size"] = serde_json::json!(3584);
        assert!(Qwen2AudioConfig::from_json(&broken).is_err(), "hidden_size");
        let mut broken = root.clone();
        broken["text_config"]["model_type"] = Value::String("qwen3".into());
        assert!(
            Qwen2AudioConfig::from_json(&broken).is_err(),
            "decoder model_type"
        );
    }

    #[test]
    fn prompt_string_matches_the_upstream_template() {
        // The pinned chat template prepends the default system message for
        // a non-system first message and appends the generation prompt.
        assert_eq!(
            build_prompt_string(3, None),
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|AUDIO|><|AUDIO|><|audio_eos|>\n\
             Please transcribe the speech.<|im_end|>\n\
             <|im_start|>assistant\n"
        );
        assert_eq!(
            build_prompt_string(1, Some("What is said?")),
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n\
             What is said?<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    /// Silence is a deterministic all-floor frontend: every mel value is
    /// exactly `(log10(1e-10) + 4) / 4 = -1.5` after the peak clamp.
    #[test]
    fn silence_yields_the_floor_value_everywhere() {
        let (features, tokens) = extract_features(&[0.0f32; 16_000]).expect("encodes silence");
        assert_eq!(tokens, 750);
        assert_eq!(features.len(), N_MELS * 3_001);
        assert!(features.iter().all(|&value| value == -1.5));
    }

    /// Filterbank fidelity at the integer-bin edges: the upstream inline
    /// construction leaves 21 low mel bands with all-zero weights (their
    /// integer bin edges coincide, and an edge weight of `(f - lower) /
    /// (center - lower)` evaluates to zero at the shared bin), and gives
    /// every other band a contiguous nonzero triangle with an exact 1.0
    /// peak.
    #[test]
    fn filterbank_has_the_upstream_triangle_shape() {
        let bank = mel_filterbank();
        assert_eq!(bank.len(), N_MELS * N_FREQS);
        let empty: Vec<usize> = (0..N_MELS)
            .filter(|&mel| {
                bank[mel * N_FREQS..(mel + 1) * N_FREQS]
                    .iter()
                    .all(|&w| w == 0.0)
            })
            .collect();
        assert_eq!(
            empty,
            vec![0, 2, 3, 5, 6, 8, 10, 12, 13, 15, 17, 19, 21, 24, 26, 28, 31, 34, 38, 42, 49]
        );
        for mel in 0..N_MELS {
            let row = &bank[mel * N_FREQS..(mel + 1) * N_FREQS];
            let Some(first) = row.iter().position(|w| *w > 0.0) else {
                continue;
            };
            let last = row.iter().rposition(|w| *w > 0.0).unwrap();
            assert!(
                row.iter()
                    .skip(first)
                    .take(last - first + 1)
                    .all(|&w| w > 0.0),
                "band {mel} has a hole"
            );
            assert!(row.contains(&1.0), "band {mel} never peaks");
        }
        assert!(bank[N_FREQS] > 0.0, "band 1 starts at bin 0");
        // Highest band reaches the top bin region.
        let top = &bank[(N_MELS - 1) * N_FREQS..N_MELS * N_FREQS];
        assert!(top[N_FREQS - 1] > 0.0 || top[N_FREQS - 2] > 0.0);
        assert_eq!(N_FREQS, 201);
    }

    /// Loads the pinned checkpoint when TURBOSPARK_QWEN2_AUDIO_MODEL_DIR
    /// is set; the run reproduces every fixture stage and the exact
    /// transcript. Minutes on CPU f32 (the dequantized decoder needs about
    /// 31 GB of resident memory); never downloads.
    #[test]
    #[ignore = "requires the pinned checkpoint in TURBOSPARK_QWEN2_AUDIO_MODEL_DIR"]
    fn pinned_checkpoint_matches_the_fixture_stages_and_transcript() {
        let Some(model_dir) = std::env::var_os("TURBOSPARK_QWEN2_AUDIO_MODEL_DIR") else {
            eprintln!("skipping: TURBOSPARK_QWEN2_AUDIO_MODEL_DIR is unset");
            return;
        };
        let fixture = load_fixture();
        let audio_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_forced_aligner_reference.wav");
        let waveform = crate::wav::read_wav_f32(&audio_path).expect("reference WAV loads");
        assert_eq!(waveform.sample_rate, 16_000);

        // Stage 1: frontend, tower, projector. The tower is dropped before
        // the decoder loads.
        let first_logits_watch = || {
            let watch: Vec<usize> = fixture["first_logits"]["watch_indices"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_u64().unwrap() as usize)
                .collect();
            let values: Vec<f64> = fixture["first_logits"]["watch_values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap())
                .collect();
            watch.into_iter().zip(values).collect::<Vec<_>>()
        };

        let projected_rows: Vec<f32>;
        // (stage, worst diff, gate) accumulated and asserted at the end so
        // one run measures every stage.
        let mut measurements: Vec<(&'static str, f32, f32)> = Vec::new();
        {
            let tower = Qwen2AudioAudioTower::load(Path::new(&model_dir)).expect("tower loads");
            assert_eq!(tower.profile(), QWEN2_AUDIO_7B_INSTRUCT_4BIT);
            let features = tower.encode(&waveform.samples).expect("encodes");

            // Frontend: fixture spots are band-major [num_mels, frames].
            let spots: Spots = serde_json::from_value(fixture["input_features"].clone()).unwrap();
            let worst = compare_spots(&features.input_features, &spots, "input_features");
            eprintln!("input_features worst spot diff {worst:.3e}");
            measurements.push(("input_features", worst, 5.0e-3));

            // Encoder attention-pooling output.
            let spots: Spots = serde_json::from_value(fixture["encoder_output"].clone()).unwrap();
            let worst = compare_spots(&features.tower_output, &spots, "encoder_output");
            eprintln!("encoder_output worst spot diff {worst:.3e}");
            measurements.push(("encoder_output", worst, 1.0e-1));

            // Projector output.
            let spots: Spots = serde_json::from_value(fixture["projected"].clone()).unwrap();
            let worst = compare_spots(&features.projected, &spots, "projected");
            eprintln!("projected worst spot diff {worst:.3e}");
            measurements.push(("projected", worst, 1.5e-1));
            projected_rows = features.projected;
        }

        // Stage 2: decoder.
        let model = Qwen2AudioLanguageModel::load(Path::new(&model_dir)).expect("decoder loads");
        assert_eq!(model.profile(), QWEN2_AUDIO_7B_INSTRUCT_4BIT);
        assert_eq!(model.audio_token_id(), 151_646);
        assert_eq!(model.eos_token_id, 151_645, "tokenizer eos is <|im_end|>");

        let frames = super::NUM_AUDIO_TOKENS;
        let (prompt_ids, audio_positions) = model.build_prompt(frames, None).expect("prompt");
        let prompt_ids_expected: Vec<i64> = fixture["prompt"]["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_i64().unwrap())
            .collect();
        let computed_prompt: Vec<i64> = prompt_ids.iter().map(|&id| i64::from(id)).collect();
        assert_eq!(computed_prompt, prompt_ids_expected, "prompt token ids");
        let expected_positions: Vec<usize> = fixture["prompt"]["audio_token_positions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(audio_positions, expected_positions, "audio positions");

        let features = super::AudioFeatures {
            input_features: Vec::new(),
            tower_output: Vec::new(),
            projected: projected_rows,
            frames,
            hidden: model.config.text.hidden_size,
        };
        let embeds = model
            .splice(&prompt_ids, &audio_positions, &features)
            .expect("splice");
        // Spliced embedding rows: the fixture concatenates rows 0, mid,
        // last into one flat [1, 3 * hidden] spot vector.
        let rows_spots: Spots =
            serde_json::from_value(fixture["inputs_embeds_rows"].clone()).unwrap();
        let rows = prompt_ids.len();
        let hidden = model.config.text.hidden_size;
        assert_eq!(rows_spots.shape, vec![1, 3 * hidden]);
        let packed_rows = [0usize, rows / 2, rows - 1];
        let mut worst_embed = 0.0f32;
        for &column in &rows_spots.columns {
            let expected = rows_spots.values[0][rows_spots
                .columns
                .iter()
                .position(|&c| c == column)
                .unwrap()];
            let segment = column / hidden;
            let within = column % hidden;
            let actual = embeds[packed_rows[segment] * hidden + within];
            worst_embed = worst_embed.max((actual - expected).abs());
        }
        eprintln!("inputs_embeds worst spot diff {worst_embed:.3e}");
        measurements.push(("inputs_embeds_rows", worst_embed, 1.0e-1));

        let (last_hidden, mut cache) = model.lm.prefill(&embeds, rows);
        let hidden_spots: VectorSpots =
            serde_json::from_value(fixture["prefill_hidden_last_row"].clone()).unwrap();
        let worst_hidden = compare_vector(&last_hidden, &hidden_spots, "prefill_hidden_last_row");
        eprintln!("prefill hidden worst diff {worst_hidden:.3e}");
        // The post-norm rows carry the full 32-layer bf16-vs-f32
        // divergence; measured worst 2.6e-1 against values of magnitude
        // 1-10, far below the O(10) error a weight-layout bug produces.
        measurements.push(("prefill_hidden_last_row", worst_hidden, 5.0e-1));

        let logits = model.lm.logits(&last_hidden);
        assert_eq!(
            crate::nn::argmax(&logits),
            fixture["first_logits"]["argmax"].as_u64().unwrap() as usize,
            "first decision"
        );
        let mut worst_logit = 0.0f32;
        for (index, expected) in first_logits_watch() {
            worst_logit = worst_logit.max((logits[index] - expected as f32).abs());
        }
        eprintln!("first logits worst watch diff {worst_logit:.3e}");
        measurements.push(("first_logits_watch", worst_logit, 5.0e-1));

        // Greedy loop with the cached prefill. Fixture `steps[i]` is the
        // i-th decision: step 0 is the prefill argmax, step i > 0 the
        // argmax after stepping token i - 1.
        let mut decision_logits = logits;
        let mut next = crate::nn::argmax(&decision_logits);
        let mut generated: Vec<i64> = Vec::new();
        let mut step_index = 0usize;
        let mut worst_logprob = 0.0f32;
        while next as u32 != model.eos_token_id && generated.len() < super::MAX_NEW_TOKENS {
            let record = &fixture["steps"][step_index];
            assert!(
                !record.is_null(),
                "reference generated more tokens than the fixture records"
            );
            assert_eq!(
                next as i64,
                record["token"].as_i64().unwrap(),
                "step {step_index} argmax"
            );
            let reference_logprob = record["top_logprob"].as_f64().unwrap() as f32;
            let computed = super::log_softmax_top(&decision_logits, next);
            worst_logprob = worst_logprob.max((computed - reference_logprob).abs());
            let token = i32::try_from(next).unwrap();
            generated.push(i64::from(token));
            let embedding = model.lm.embed(&[token]).unwrap();
            let hidden = model.lm.step(&embedding, &mut cache);
            decision_logits = model.lm.logits(&hidden);
            next = crate::nn::argmax(&decision_logits);
            step_index += 1;
        }
        let expected_generated: Vec<i64> = fixture["generated_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_i64().unwrap())
            .collect();
        assert_eq!(generated, expected_generated, "generated token ids");
        eprintln!("step logprob worst diff {worst_logprob:.3e}");
        measurements.push(("step_top_logprob", worst_logprob, 2.5e-1));

        let transcript = model
            .decode_generated(&generated.iter().map(|&id| id as u32).collect::<Vec<_>>())
            .expect("decodes");
        assert_eq!(
            transcript, fixture["transcript"],
            "transcript must match the reference exactly"
        );

        // Tolerances: every gate must absorb the measured bf16-vs-f32
        // divergence with headroom while staying well below error
        // magnitudes a real packing or layout bug would produce.
        for (stage, worst, gate) in &measurements {
            assert!(
                worst < gate,
                "stage {stage} worst diff {worst:.3e} exceeds gate {gate:.3e}"
            );
        }
        for (stage, worst, gate) in &measurements {
            eprintln!("gate {stage}: worst {worst:.3e} < {gate:.3e}");
        }
    }
}
