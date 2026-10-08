//! Voxtral Realtime 4B streaming speech-to-text, offline buffered path.
//!
//! Reference: `mlx_audio/stt/models/voxtral_realtime/` (voxtral_realtime.py
//! 606 lines, encoder.py, decoder.py, audio.py, config.py, tokenizer.py) at
//! mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`.
//!
//! Ported: the offline buffered branch of upstream `generate` with the
//! default 480 ms transcription delay. 16 kHz mono input is padded with
//! silence in 1280-sample token units (`_pad_audio_streaming`), turned into
//! a 128-band Slaney log-mel spectrogram, encoded by the 32-layer causal
//! sliding-window transformer, 4x-downsampled and projected by the adapter,
//! summed per position with the `[BOS] + STREAMING_PAD` prompt embeddings,
//! prefilled through the 26-layer Mistral-style decoder, and greedily
//! decoded until EOS (or until the audio positions run out), then decoded
//! with the Tekken BPE. The real-time chunked `StreamingSession` with its
//! delay and commit semantics is refused; see the family README.
//!
//! The pinned checkpoint quantizes the encoder and decoder linears to
//! 4-bit affine groups of 64; the conv stem, adapter projection, norms, and
//! the tied token embedding matrix stay plain (F16/F32) in that
//! distribution. Token ids are signed 32-bit at crate boundaries.

pub(crate) mod decoder;
pub(crate) mod encoder;
pub(crate) mod tokenizer;

use std::fs;
use std::path::Path;

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::mel::MelScale;
use crate::nn::{bad_config, load_tensor, Linear, RmsNorm};
use crate::ops;
use crate::quant::QuantScheme;
use crate::{Result, SpeechError};

use decoder::{Decoder, DecoderGeometry};
use encoder::{AudioEncoder, EncoderGeometry};
use tokenizer::TekkenTokenizer;

/// Immutable Hugging Face checkpoint profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoxtralRealtimeProfile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

/// The pinned profile: smoke-verified upstream on the shared reference WAV
/// (mlx-audio 0.5.7 generated the exact expected phrase).
pub const VOXTRAL_MINI_4B_REALTIME_4BIT: VoxtralRealtimeProfile = VoxtralRealtimeProfile {
    name: "Voxtral Mini 4B Realtime 4-bit",
    repository: "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit",
    revision: "fdebf7b2af834a1db4b8a3c99ab7480b333adf9e",
};

// Frontend and streaming constants, pinned by the upstream `config.py`
// defaults and `voxtral_realtime.py`.
/// Input sample rate.
pub(crate) const SAMPLE_RATE: usize = 16_000;
/// Mel bands.
pub(crate) const N_MELS: usize = 128;
/// STFT window size (also the FFT size).
pub(crate) const WINDOW_SIZE: usize = 400;
/// STFT hop.
pub(crate) const HOP_LENGTH: usize = 160;
/// Raw samples per audio token (`SAMPLE_RATE / FRAME_RATE`).
pub(crate) const RAW_AUDIO_LENGTH_PER_TOK: usize = 1280;
/// Conv frames per audio token (`RAW_AUDIO_LENGTH_PER_TOK / HOP_LENGTH`).
const AUDIO_LENGTH_PER_TOK: usize = 8;
/// Mel clamp maximum (upstream `global_log_mel_max`).
const GLOBAL_LOG_MEL_MAX: f32 = 1.5;
/// Default transcription delay (upstream `ModelConfig` default).
const TRANSCRIPTION_DELAY_MS: f32 = 480.0;
/// Left silence padding in audio tokens (upstream `n_left_pad_tokens`).
const N_LEFT_PAD_TOKENS: usize = 32;
/// Special token ids (upstream `ModelConfig` defaults).
const BOS_TOKEN_ID: i32 = 1;
const EOS_TOKEN_ID: i32 = 2;
const STREAMING_PAD_TOKEN_ID: i32 = 32;
/// Max generated tokens (upstream `max_tokens` default).
const MAX_NEW_TOKENS: usize = 4096;

/// Parsed and pinned checkpoint configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct VoxtralRealtimeConfig {
    pub encoder: EncoderGeometry,
    pub decoder: DecoderGeometry,
    /// The adapter output width; must equal the decoder width.
    pub adapter_dim: usize,
    pub quantization: QuantScheme,
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

/// Reads a float field that must equal `pinned` when present (bit-equal
/// after f64 parse); missing takes `pinned`.
fn pinned_float(value: &Value, key: &str, pinned: f64) -> Result<f64> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(pinned),
        Some(found) => {
            let found = found
                .as_f64()
                .ok_or_else(|| bad_config(key, "must be a number"))?;
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

fn require_true(value: &Value, key: &str) -> Result<()> {
    if value.get(key).and_then(Value::as_bool) == Some(true) {
        Ok(())
    } else {
        Err(bad_config(
            key,
            format!("expected true, found {:?}", value.get(key)),
        ))
    }
}

impl VoxtralRealtimeConfig {
    /// Parses `config.json`. The upstream `from_dict` fills missing keys
    /// from dataclass defaults; this port accepts a missing key as the
    /// pinned default and refuses any present-but-different value, so an
    /// unverified geometry variant cannot load silently.
    pub fn from_json(root: &Value) -> Result<Self> {
        require(root, "model_type", "voxtral_realtime")?;

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
        if quant.get("mode").and_then(Value::as_str) != Some("affine") {
            return Err(unsupported("only the affine quantization mode is verified"));
        }
        if bits != 4 || group_size != 64 {
            return Err(unsupported(format!(
                "only the verified 4-bit group-64 scheme is accepted, found {bits}-bit \
                 groups of {group_size}"
            )));
        }
        let scheme = QuantScheme {
            bits: 4,
            group_size: 64,
        };

        // Audio encoding args can sit at the top level or nested inside
        // encoder_args (upstream ModelConfig.__post_init__ moves them out).
        let encoder_json = root
            .get("encoder_args")
            .ok_or_else(|| bad_config("encoder_args", "is missing"))?;
        let audio_json = root
            .get("audio_encoding_args")
            .or_else(|| encoder_json.get("audio_encoding_args"))
            .ok_or_else(|| bad_config("audio_encoding_args", "is missing"))?;
        pinned_int(audio_json, "sampling_rate", SAMPLE_RATE)?;
        pinned_float(audio_json, "frame_rate", 12.5)?;
        pinned_int(audio_json, "num_mel_bins", N_MELS)?;
        pinned_int(audio_json, "hop_length", HOP_LENGTH)?;
        pinned_int(audio_json, "window_size", WINDOW_SIZE)?;
        pinned_float(audio_json, "global_log_mel_max", GLOBAL_LOG_MEL_MAX as f64)?;

        let encoder = EncoderGeometry {
            dim: pinned_int(encoder_json, "dim", 1280)?,
            n_layers: pinned_int(encoder_json, "n_layers", 32)?,
            n_heads: pinned_int(encoder_json, "n_heads", 32)?,
            head_dim: pinned_int(encoder_json, "head_dim", 64)?,
            hidden_dim: pinned_int(encoder_json, "hidden_dim", 5120)?,
            norm_eps: pinned_float(encoder_json, "norm_eps", 1e-5)? as f32,
            rope_theta: pinned_float(encoder_json, "rope_theta", 1_000_000.0)? as f32,
            sliding_window: pinned_int(encoder_json, "sliding_window", 750)?,
            downsample_factor: pinned_int(encoder_json, "downsample_factor", 4)?,
        };
        if encoder_json.get("causal").is_some()
            && encoder_json.get("causal") != Some(&Value::Bool(true))
        {
            return Err(unsupported("only the causal encoder is verified"));
        }

        let decoder_json = root
            .get("decoder")
            .ok_or_else(|| bad_config("decoder", "is missing"))?;
        let decoder = DecoderGeometry {
            dim: pinned_int(decoder_json, "dim", 3072)?,
            n_layers: pinned_int(decoder_json, "n_layers", 26)?,
            n_heads: pinned_int(decoder_json, "n_heads", 32)?,
            n_kv_heads: pinned_int(decoder_json, "n_kv_heads", 8)?,
            head_dim: pinned_int(decoder_json, "head_dim", 128)?,
            hidden_dim: pinned_int(decoder_json, "hidden_dim", 9216)?,
            vocab_size: pinned_int(decoder_json, "vocab_size", 131_072)?,
            norm_eps: pinned_float(decoder_json, "norm_eps", 1e-5)? as f32,
            rope_theta: pinned_float(decoder_json, "rope_theta", 1_000_000.0)? as f32,
            sliding_window: pinned_int(decoder_json, "sliding_window", 8192)?,
            ada_dim: pinned_int(decoder_json, "ada_rms_norm_t_cond_dim", 32)?,
        };
        require_true(decoder_json, "tied_embeddings")?;
        require_true(decoder_json, "ada_rms_norm_t_cond")?;
        if decoder.n_heads % decoder.n_kv_heads != 0 {
            return Err(bad_config("decoder", "KV head geometry is inconsistent"));
        }

        Ok(Self {
            encoder,
            adapter_dim: decoder.dim,
            decoder,
            quantization: scheme,
        })
    }
}

/// Silence padding in 1280-sample token units (upstream
/// `_pad_audio_streaming`, offline mode): `n_left_pad_tokens` samples of
/// silence on the left, then alignment to a 1280 multiple plus
/// `n_right_pad_tokens` tokens of silence on the right.
pub(crate) fn pad_audio_streaming(
    audio: &[f32],
    n_left_pad_tokens: usize,
    n_right_pad_tokens: usize,
) -> Vec<f32> {
    let mult_of = RAW_AUDIO_LENGTH_PER_TOK;
    let align_pad = (mult_of - (audio.len() % mult_of)) % mult_of;
    let left_pad = n_left_pad_tokens * mult_of;
    let right_pad = align_pad + n_right_pad_tokens * mult_of;
    let mut out = vec![0.0f32; left_pad + audio.len() + right_pad];
    out[left_pad..left_pad + audio.len()].copy_from_slice(audio);
    out
}

/// Audio tokens for a raw sample count (upstream `_num_audio_tokens`).
pub(crate) fn num_audio_tokens(audio_len: usize) -> usize {
    let frames = if audio_len % HOP_LENGTH != 0 {
        ((audio_len as f64 / HOP_LENGTH as f64) - 1.0).ceil() as usize
    } else {
        audio_len / HOP_LENGTH
    };
    ((frames as f64) / AUDIO_LENGTH_PER_TOK as f64).ceil() as usize
}

/// Delay tokens for a delay in milliseconds (upstream `_num_delay_tokens`).
pub(crate) fn num_delay_tokens(delay_ms: f32) -> i32 {
    let delay_len = (f64::from(delay_ms) / 1000.0 * SAMPLE_RATE as f64) as usize;
    num_audio_tokens(delay_len) as i32
}

/// The `[BOS] + STREAMING_PAD x (n_left + n_delay)` prompt ids.
pub(crate) fn prompt_ids(n_left: usize, n_delay: i32) -> Vec<i32> {
    let mut ids = vec![BOS_TOKEN_ID];
    ids.resize(1 + n_left + n_delay as usize, STREAMING_PAD_TOKEN_ID);
    ids
}

/// Per-position cos/sin tables for interleaved (GPT-J style) RoPE over the
/// absolute positions `start..start + len`.
pub(crate) struct RopeTables {
    pub(crate) start: usize,
    pub(crate) cos: Vec<f32>,
    pub(crate) sin: Vec<f32>,
}

impl RopeTables {
    pub(crate) fn new(start: usize, len: usize, rotary_dim: usize, theta: f32) -> Self {
        let (cos, sin) = ops::rope_tables_range(start, len, rotary_dim, theta);
        Self { start, cos, sin }
    }
}

/// Per-layer rotating KV store: keeps the last `capacity` (== the
/// attention sliding window) entries in temporal order. Query row at
/// absolute position `p` reads exactly the retained positions in
/// `(p - capacity, p]`, which is the set the upstream `RotatingKVCache`
/// mask selects; trimming to the window makes the cut implicit.
pub(crate) struct RingCache {
    pub(crate) keys: Vec<Vec<f32>>,
    pub(crate) values: Vec<Vec<f32>>,
    len: usize,
    start: usize,
    capacity: usize,
}

impl RingCache {
    pub(crate) fn new(kv_heads: usize, capacity: usize) -> Self {
        Self {
            keys: vec![Vec::new(); kv_heads],
            values: vec![Vec::new(); kv_heads],
            len: 0,
            start: 0,
            capacity,
        }
    }

    /// Appends one chunk of head-major `[heads, rows, head_dim]` rotated
    /// keys and values WITHOUT trimming. Every key appended in this chunk
    /// must stay visible to this chunk's own queries (upstream answers
    /// query `p` before any later key exists), so trimming happens only in
    /// [`RingCache::trim_to_capacity`] after the chunk's attention ran.
    pub(crate) fn append(
        &mut self,
        keys: &[f32],
        values: &[f32],
        heads: usize,
        rows: usize,
        head_dim: usize,
    ) {
        for head in 0..heads {
            let begin = head * rows * head_dim;
            let end = begin + rows * head_dim;
            self.keys[head].extend_from_slice(&keys[begin..end]);
            self.values[head].extend_from_slice(&values[begin..end]);
        }
        self.len += rows;
    }

    /// Trims the retained entries to the last `capacity` from the front.
    /// Call after a chunk's attention: the next chunk's first query at
    /// absolute position `start + len` needs exactly the keys in
    /// `(start + len - capacity, start + len)`.
    pub(crate) fn trim_to_capacity(&mut self, heads: usize, head_dim: usize) {
        let drop = self.len.saturating_sub(self.capacity);
        if drop > 0 {
            for head in 0..heads {
                self.keys[head].drain(..drop * head_dim);
                self.values[head].drain(..drop * head_dim);
            }
            self.len -= drop;
            self.start += drop;
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Absolute position of the first retained entry.
    pub(crate) fn start(&self) -> usize {
        self.start
    }

    /// Cache index range `[lo, hi)` visible to a query at `query_pos`:
    /// causal (nothing after the query is stored) intersected with the
    /// sliding window `(query_pos - capacity, query_pos]`.
    pub(crate) fn visible_range(&self, query_pos: usize) -> (usize, usize) {
        let low = (query_pos + 1)
            .saturating_sub(self.capacity)
            .max(self.start);
        let high = query_pos + 1;
        (
            low - self.start,
            high.min(self.start + self.len) - self.start,
        )
    }
}

/// Loads an RMSNorm weight with the `[width]` shape check.
pub(crate) fn load_norm(
    files: &[SafetensorsFile],
    name: &str,
    width: usize,
    eps: f32,
) -> Result<RmsNorm> {
    let file = files
        .iter()
        .find(|file| file.contains_tensor(&format!("{name}.weight")))
        .ok_or_else(|| SpeechError::Tensor {
            name: format!("{name}.weight"),
            why: "tensor is missing".into(),
        })?;
    Ok(RmsNorm::new(
        load_tensor(file, &format!("{name}.weight"), &[width])?,
        eps,
    ))
}

/// Loads a plain (never quantized) linear from the shard carrying it,
/// with the `[output, input]` shape check.
pub(crate) fn load_plain_linear_sharded(
    files: &[SafetensorsFile],
    base: &str,
    input: usize,
    output: usize,
    has_bias: bool,
) -> Result<Linear> {
    let file = files
        .iter()
        .find(|file| file.contains_tensor(&format!("{base}.weight")))
        .ok_or_else(|| SpeechError::Tensor {
            name: format!("{base}.weight"),
            why: "tensor is missing".into(),
        })?;
    let weight = load_tensor(file, &format!("{base}.weight"), &[output, input])?;
    let bias = if has_bias {
        Some(load_tensor(file, &format!("{base}.bias"), &[output])?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias, input, output))
}

/// The log-mel frontend (upstream `audio.py`): periodic Hann window of 400,
/// reflect padding of 200 on both edges, 400-point real FFT at hop 160 with
/// the last frame dropped, projection onto the Slaney 128-band filterbank
/// (0 to 8000 Hz), `log10` with a 1e-10 floor, a global lower clamp at
/// `1.5 - 8`, and the `(x + 4) / 4` scale. Returns the band-major
/// spectrogram `[128, frames]` after the even-frame trim (the first frame
/// is dropped when the count is odd).
pub(crate) fn mel_spectrogram(padded: &[f32]) -> Result<(Vec<f32>, usize)> {
    if padded.iter().any(|sample| !sample.is_finite()) {
        return Err(SpeechError::Input {
            why: "padded waveform contains non-finite samples".into(),
        });
    }
    if padded.len() < 2 {
        return Err(SpeechError::Input {
            why: "reflect padding needs at least two samples".into(),
        });
    }
    let bank = crate::mel::mel_filterbank(
        N_MELS,
        WINDOW_SIZE,
        SAMPLE_RATE as u32,
        0.0,
        Some(8_000.0),
        MelScale::Slaney,
    )
    .map_err(|error| SpeechError::Audio(error.to_string()))?;
    debug_assert_eq!(bank.num_bins, WINDOW_SIZE / 2 + 1);

    // Reflect padding about both edges, edge value excluded (numpy
    // `mode="reflect"`).
    let pad = WINDOW_SIZE / 2;
    let mut frames_input = vec![0.0f32; padded.len() + 2 * pad];
    for offset in 0..pad {
        frames_input[offset] = padded[pad - offset];
        frames_input[padded.len() + pad + offset] = padded[padded.len() - 2 - offset];
    }
    frames_input[pad..pad + padded.len()].copy_from_slice(padded);

    // Periodic Hann window, f32 like the reference.
    let window: Vec<f32> = (0..WINDOW_SIZE)
        .map(|n| {
            0.5 * (1.0 - (2.0 * std::f64::consts::PI * n as f64 / WINDOW_SIZE as f64).cos() as f32)
        })
        .collect();

    // f64 twiddle tables for the 400-point real DFT (numpy's pocketfft
    // stands in for MLX rfft; agreement is far inside the fixture gates).
    let n_bins = WINDOW_SIZE / 2 + 1;
    let mut cos_table = vec![0.0f64; n_bins * WINDOW_SIZE];
    let mut sin_table = vec![0.0f64; n_bins * WINDOW_SIZE];
    for bin in 0..n_bins {
        for n in 0..WINDOW_SIZE {
            let angle = -2.0 * std::f64::consts::PI * (bin * n) as f64 / WINDOW_SIZE as f64;
            cos_table[bin * WINDOW_SIZE + n] = angle.cos();
            sin_table[bin * WINDOW_SIZE + n] = angle.sin();
        }
    }

    let n_frames_raw = 1 + (frames_input.len() - WINDOW_SIZE) / HOP_LENGTH;
    // The reference drops the final STFT frame before the mel projection.
    let n_frames = n_frames_raw - 1;
    let mut power = vec![0.0f32; n_bins];
    let mut frame = vec![0.0f64; WINDOW_SIZE];
    let mut mel = vec![0.0f32; N_MELS * n_frames];
    for t in 0..n_frames {
        let base = t * HOP_LENGTH;
        for (offset, slot) in frame.iter_mut().enumerate() {
            *slot = f64::from(frames_input[base + offset] * window[offset]);
        }
        for (bin, slot) in power.iter_mut().enumerate() {
            let table_base = bin * WINDOW_SIZE;
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for (n, &sample) in frame.iter().enumerate() {
                re += sample * cos_table[table_base + n];
                im -= sample * sin_table[table_base + n];
            }
            *slot = (re * re + im * im) as f32;
        }
        // Filterbank projection: mel[m] = sum_f weights[m, f] * power[f].
        for m in 0..N_MELS {
            let row = &bank.weights[m * n_bins..(m + 1) * n_bins];
            let mut sum = 0.0f32;
            for (f, &weight) in row.iter().enumerate() {
                sum += weight * power[f];
            }
            let logged = sum.max(1e-10).log10();
            let clamped = logged.max(GLOBAL_LOG_MEL_MAX - 8.0);
            mel[m * n_frames + t] = (clamped + 4.0) / 4.0;
        }
    }

    // Even-frame trim: drop the first frame when the count is odd.
    if n_frames % 2 != 0 {
        let mut trimmed = vec![0.0f32; N_MELS * (n_frames - 1)];
        for m in 0..N_MELS {
            trimmed[m * (n_frames - 1)..(m + 1) * (n_frames - 1)]
                .copy_from_slice(&mel[m * n_frames + 1..m * n_frames + n_frames]);
        }
        return Ok((trimmed, n_frames - 1));
    }
    Ok((mel, n_frames))
}

/// One prepared audio window: everything between the waveform and the
/// decoder prompt.
pub struct PreparedAudio {
    /// The silence-padded waveform.
    pub padded: Vec<f32>,
    /// Band-major log-mel `[128, mel_frames]`.
    pub mel: Vec<f32>,
    pub mel_frames: usize,
    /// Row-major adapter output `[adapter_len, decoder dim]`.
    pub adapter: Vec<f32>,
    pub adapter_len: usize,
    /// Conv-frame count before the downsample.
    pub conv_frames: usize,
    /// Total audio positions the decode loop may run (`conv_frames / 4`
    /// when it divides exactly, matching the upstream count).
    pub n_audio_total: usize,
    /// Delay tokens implied by the transcription delay.
    pub n_delay: i32,
}

/// Loaded Voxtral Realtime model (offline buffered path).
pub struct VoxtralRealtime {
    config: VoxtralRealtimeConfig,
    encoder: AudioEncoder,
    decoder: Decoder,
    tokenizer: TekkenTokenizer,
    n_delay: i32,
}

impl VoxtralRealtime {
    /// Load the pinned profile from an already-downloaded model folder.
    /// Never downloads; refuses unverified quantization schemes and
    /// unverified geometry knobs.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let shards = open_checkpoint(model_dir)?;
        let config = load_config_from_shards(&shards, model_dir)?;
        let encoder = AudioEncoder::load(
            &shards,
            &config.encoder,
            config.adapter_dim,
            config.quantization,
        )?;
        let mut decoder = Decoder::load(&shards, &config.decoder, config.quantization)?;
        let n_delay = num_delay_tokens(TRANSCRIPTION_DELAY_MS);
        decoder.precompute_ada_scales(n_delay);
        let tokenizer = TekkenTokenizer::from_model_path(model_dir)?;
        Ok(Self {
            config,
            encoder,
            decoder,
            tokenizer,
            n_delay,
        })
    }

    pub fn profile(&self) -> VoxtralRealtimeProfile {
        VOXTRAL_MINI_4B_REALTIME_4BIT
    }

    pub fn config(&self) -> &VoxtralRealtimeConfig {
        &self.config
    }

    /// Runs the offline buffered path up to the adapter: silence padding,
    /// log-mel, conv stem, causal encoder, 4x downsample, projection.
    pub fn prepare(&self, samples: &[f32]) -> Result<PreparedAudio> {
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(SpeechError::Input {
                why: "waveform contains non-finite samples".into(),
            });
        }
        let n_delay = num_delay_tokens(TRANSCRIPTION_DELAY_MS);
        let n_right = (n_delay as usize + 1) + 10;
        let padded = pad_audio_streaming(samples, N_LEFT_PAD_TOKENS, n_right);
        let (mel, mel_frames) = mel_spectrogram(&padded)?;
        let conv = self.encoder.conv_stem(&mel);
        let conv_frames = conv.len() / self.config.encoder.dim;
        let encoded = if conv_frames <= self.config.encoder.sliding_window {
            self.encoder.encode_full(&conv, conv_frames)?
        } else {
            self.encoder.encode_chunked(&conv, conv_frames)?
        };
        let (adapter, adapter_len) = self.encoder.downsample_and_project(&encoded, conv_frames);
        Ok(PreparedAudio {
            padded,
            mel,
            mel_frames,
            adapter,
            adapter_len,
            conv_frames,
            n_audio_total: conv_frames / self.config.encoder.downsample_factor,
            n_delay,
        })
    }

    /// Transcribes one mono 16 kHz waveform through the offline buffered
    /// path and returns the stripped transcript text.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        let prepared = self.prepare(samples)?;
        let prompt = prompt_ids(N_LEFT_PAD_TOKENS, self.n_delay);
        let prompt_len = prompt.len();
        if prepared.adapter_len < prompt_len {
            return Err(SpeechError::Input {
                why: format!(
                    "adapter produced {} tokens for a {}-token prompt",
                    prepared.adapter_len, prompt_len
                ),
            });
        }
        let hidden = self.config.decoder.dim;
        // Per-position sum of the audio embedding and the prompt token
        // embedding (upstream `prefix_embeds = adapter_out[:prompt_len] +
        // prompt_text_embeds`).
        let mut embeds = self.decoder.embed(&prompt)?;
        for (position, id) in prompt.iter().enumerate() {
            let tok_embed = self.decoder.embed(&[*id])?;
            for column in 0..hidden {
                embeds[position * hidden + column] =
                    prepared.adapter[position * hidden + column] + tok_embed[column];
            }
        }
        let (last, mut cache) = self.decoder.prefill(&embeds, prompt_len);
        let mut logits = self.decoder.tied_logits(&last);
        let mut next = crate::nn::argmax(&logits) as i32;
        let mut generated: Vec<i32> = Vec::new();
        let mut broke = false;
        for pos in prompt_len..prepared.n_audio_total {
            generated.push(next);
            if next == EOS_TOKEN_ID || generated.len() > MAX_NEW_TOKENS {
                broke = true;
                break;
            }
            let mut embed = self.decoder.embed(&[next])?;
            if pos < prepared.adapter_len {
                for column in 0..hidden {
                    embed[column] += prepared.adapter[pos * hidden + column];
                }
            }
            let step_hidden = self.decoder.step(&embed, pos, &mut cache);
            logits = self.decoder.tied_logits(&step_hidden);
            next = crate::nn::argmax(&logits) as i32;
        }
        if !broke {
            // The loop ran out of audio positions; read the pending token
            // (the upstream for/else tail).
            generated.push(next);
        }
        if generated.last() == Some(&EOS_TOKEN_ID) {
            generated.pop();
        }
        let text = self.tokenizer.decode(&generated);
        Ok(text.trim().to_owned())
    }
}

/// Reads and parses `config.json` from a model folder.
fn load_config(model_dir: &Path) -> Result<VoxtralRealtimeConfig> {
    let config_json =
        fs::read_to_string(model_dir.join("config.json")).map_err(|error| SpeechError::Input {
            why: format!("cannot read config.json: {error}"),
        })?;
    let root: Value = serde_json::from_str(&config_json)
        .map_err(|error| bad_config("config.json", error.to_string()))?;
    VoxtralRealtimeConfig::from_json(&root)
}

/// Parses the config from the already-open shards' model directory. Kept
/// separate so the gated test can load encoder and decoder in two passes.
fn load_config_from_shards(
    _shards: &[SafetensorsFile],
    model_dir: &Path,
) -> Result<VoxtralRealtimeConfig> {
    load_config(model_dir)
}

/// Opens the checkpoint shards: the `model.safetensors.index.json`
/// `weight_map` order when present, otherwise `model.safetensors`, then
/// `weights.safetensors`.
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
    } else if model_dir.join("model.safetensors").is_file() {
        names.push("model.safetensors".to_owned());
    } else {
        names.push("weights.safetensors".to_owned());
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

/// Test-only shared helpers: a minimal safetensors writer the family unit
/// tests use to build tiny fake checkpoints.
#[cfg(test)]
pub(crate) mod tests_support {
    use std::path::Path;

    /// Writes a minimal safetensors file: F32 or U32 tensors keyed by
    /// name. The format is an 8-byte little-endian header length, a JSON
    /// header of `name -> {dtype, shape, data_offsets}`, then the data
    /// blob.
    pub(crate) fn write_safetensors(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = data.len();
            data.extend_from_slice(bytes);
            header.insert(
                (*name).to_owned(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, data.len()],
                }),
            );
        }
        let header_json = serde_json::to_string(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        out.extend_from_slice(header_json.as_bytes());
        out.extend_from_slice(&data);
        std::fs::write(path, out).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::{
        num_audio_tokens, num_delay_tokens, pad_audio_streaming, prompt_ids, VoxtralRealtimeConfig,
        VOXTRAL_MINI_4B_REALTIME_4BIT,
    };
    use serde_json::Value;
    use std::path::Path;

    const FIXTURE: &str = include_str!("../../../testdata/voxtral_realtime_reference.json");

    fn load_fixture() -> Value {
        serde_json::from_str(FIXTURE).expect("valid voxtral_realtime reference fixture")
    }

    #[derive(serde::Deserialize)]
    struct Spots {
        shape: Vec<usize>,
        rows: Vec<usize>,
        columns: Vec<usize>,
        values: Vec<Vec<f32>>,
    }

    fn compare_spots(
        actual: &[f32],
        row_stride: usize,
        spots: &Spots,
        label: &str,
        gate: f32,
    ) -> f32 {
        assert_eq!(
            spots.shape[1], row_stride,
            "{label} stride must match the fixture column count"
        );
        let mut worst = 0.0f32;
        for (row_index, &row) in spots.rows.iter().enumerate() {
            for (column_index, &column) in spots.columns.iter().enumerate() {
                let expected = spots.values[row_index][column_index];
                let diff = (actual[row * row_stride + column] - expected).abs();
                worst = worst.max(diff);
                assert!(
                    diff < gate,
                    "{label} [{row},{column}] differs by {diff} (gate {gate})"
                );
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

    fn compare_vector(actual: &[f32], spots: &VectorSpots, label: &str, gate: f32) -> f32 {
        if spots.length > 0 {
            assert_eq!(actual.len(), spots.length, "{label} length");
        }
        let mut worst = 0.0f32;
        for (&index, &expected) in spots.indices.iter().zip(&spots.values) {
            let diff = (actual[index] - expected).abs();
            worst = worst.max(diff);
            assert!(
                diff < gate,
                "{label}[{index}] differs by {diff} (gate {gate})"
            );
        }
        worst
    }

    fn smoke_waveform() -> crate::waveform::Waveform {
        let audio_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_forced_aligner_reference.wav");
        let waveform = crate::wav::read_wav_f32(&audio_path).expect("reference WAV loads");
        assert_eq!(waveform.sample_rate, 16_000);
        waveform
    }

    #[test]
    fn profile_pin_is_immutable() {
        assert_eq!(
            VOXTRAL_MINI_4B_REALTIME_4BIT.repository,
            "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
        );
        assert_eq!(
            VOXTRAL_MINI_4B_REALTIME_4BIT.revision,
            "fdebf7b2af834a1db4b8a3c99ab7480b333adf9e"
        );
    }

    #[test]
    fn fixture_provenance_pins_the_reference_run() {
        let fixture = load_fixture();
        assert_eq!(
            fixture["provenance"]["revision"],
            "fdebf7b2af834a1db4b8a3c99ab7480b333adf9e"
        );
        assert_eq!(
            fixture["provenance"]["source"],
            "mlx-audio voxtral_realtime at commit e1b19b9054bf163f5d812221a54fcc346f1890e9"
        );
        assert_eq!(
            fixture["provenance"]["repository"],
            "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
        );
        // The fixture was generated from the shared smoke WAV (digest of
        // the file bytes).
        let audio_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_forced_aligner_reference.wav");
        let digest = turbospark_model_io::hash_file(&audio_path, 1 << 20).expect("hashes");
        assert_eq!(fixture["provenance"]["audio_sha256"], digest);
    }

    #[test]
    fn fixture_transcript_and_greedy_shape() {
        let fixture = load_fixture();
        assert_eq!(
            fixture["transcript"],
            "The quick brown fox jumps over the lazy dog."
        );
        // The raw decode carries the leading space of the first content
        // token; the port strips like the reference.
        assert_eq!(
            fixture["transcript_raw"],
            " The quick brown fox jumps over the lazy dog."
        );
        let generated: Vec<i64> = fixture["generated_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_i64().unwrap())
            .collect();
        // The smoke decode ends when the audio positions run out (no EOS),
        // with trailing STREAMING_PAD ids the Tekken decode skips.
        assert_eq!(fixture["prompt"]["prompt_len"], 39);
        assert_eq!(fixture["encoder"]["n_audio_total"], 84);
        assert_eq!(generated.len(), 46);
        assert_eq!(*generated.last().unwrap(), 32);
        assert_eq!(fixture["steps"].as_array().unwrap().len(), 46);
        // The tail of the decode trails off into STREAMING_PAD ids the
        // Tekken decode skips.
        for step in fixture["steps"].as_array().unwrap().iter().rev().take(8) {
            assert_eq!(step["token"], 32, "trailing decisions must be pads");
        }
        // Token bytes cross-check: the " The" payload.
        assert_eq!(fixture["token_bytes_hex"][3], "20546865");
    }

    /// The always-on frontend gate: Rust padding reproduces the reference
    /// padded waveform byte for byte, and the Rust mel matches the fixture
    /// spots. No checkpoint involved.
    #[test]
    fn padding_and_mel_match_the_reference_frontend() {
        let fixture = load_fixture();
        let waveform = smoke_waveform();
        let padded = pad_audio_streaming(&waveform.samples, 32, 17);
        let digest = turbospark_model_io::hash_data(
            &padded
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        assert_eq!(fixture["padding"]["sha256_f32le"], digest, "padded digest");
        assert_eq!(fixture["padding"]["padded_samples"], padded.len());
        let spots: VectorSpots =
            serde_json::from_value(fixture["padding"]["spots"].clone()).unwrap();
        for (&index, &expected) in spots.indices.iter().zip(&spots.values) {
            assert_eq!(
                padded[index].to_bits(),
                expected.to_bits(),
                "padded_waveform[{index}] must match bitwise"
            );
        }

        let (mel, mel_frames) = super::mel_spectrogram(&padded).expect("mel computes");
        assert_eq!(mel_frames, 672);
        assert_eq!(fixture["mel"]["shape"][0], mel_frames);
        assert_eq!(fixture["mel"]["shape"][1], super::N_MELS);
        // The fixture stores the frame-major transpose; compare via the
        // row-stride form (row = frame, stride = bands).
        let spots: Spots = serde_json::from_value(fixture["mel"].clone()).unwrap();
        let mut worst_mel = 0.0f32;
        for (row_index, &frame) in spots.rows.iter().enumerate() {
            for (column_index, &band) in spots.columns.iter().enumerate() {
                let expected = spots.values[row_index][column_index];
                let diff = (mel[band * mel_frames + frame] - expected).abs();
                worst_mel = worst_mel.max(diff);
                assert!(
                    diff < 2.0e-4,
                    "mel [frame {frame}, band {band}] differs by {diff}"
                );
            }
        }
        eprintln!("mel worst spot diff {worst_mel:.3e}");
    }

    /// Silence pads through the whole frontend to the floor value: log10
    /// of the 1e-10 clamp is -10, the lower clamp at 1.5 - 8 binds, and
    /// the scale lands at (-6.5 + 4) / 4.
    #[test]
    fn silence_yields_the_floor_value_everywhere() {
        let padded = vec![0.0f32; 1280 * 4];
        let (mel, frames) = super::mel_spectrogram(&padded).expect("mel computes");
        assert_eq!(frames, 32);
        for value in &mel {
            assert!((value - (-0.625)).abs() < 1e-6, "silence floor {value}");
        }
    }

    /// Token-count and padding arithmetic against the upstream formulas.
    #[test]
    fn token_count_and_padding_math() {
        assert_eq!(num_delay_tokens(480.0), 6);
        assert_eq!(num_delay_tokens(0.0), 0);
        // 7680 samples: exact multiple of 160 -> 48 frames -> 6 tokens.
        assert_eq!(num_audio_tokens(7680), 6);
        // 7679 samples: ceil(7679/160 - 1) = ceil(46.99) = 47 -> 6 tokens.
        assert_eq!(num_audio_tokens(7679), 6);
        // 1 sample: ceil(1/160 - 1) = 0 -> 0 tokens.
        assert_eq!(num_audio_tokens(1), 0);
        assert_eq!(num_audio_tokens(1280), 1);

        // Padding: 44715 raw samples -> align 1025 + left 40960 + right
        // 21760 = 107520.
        let audio = vec![1.0f32; 44715];
        let padded = pad_audio_streaming(&audio, 32, 17);
        assert_eq!(padded.len(), 107_520);
        assert_eq!(&padded[..40_960], &vec![0.0f32; 40_960][..]);
        assert_eq!(&padded[40_960..40_960 + 44_715], &audio[..]);
        assert!(padded[40_960 + 44_715..].iter().all(|v| *v == 0.0));
        // Already aligned input gets no alignment pad.
        let aligned = pad_audio_streaming(&vec![1.0f32; 1280], 0, 1);
        assert_eq!(aligned.len(), 1280 * 2);

        // Prompt ids: [BOS] then (32 + 6) pads of id 32.
        let ids = prompt_ids(32, 6);
        assert_eq!(ids.len(), 39);
        assert_eq!(ids[0], 1);
        assert!(ids[1..].iter().all(|&id| id == 32));
    }

    #[test]
    fn pinned_config_parses_and_refuses_unverified_knobs() {
        let root: Value = serde_json::from_str(
            r#"{
            "decoder": {
                "dim": 3072, "n_layers": 26, "head_dim": 128, "hidden_dim": 9216,
                "n_heads": 32, "n_kv_heads": 8, "vocab_size": 131072,
                "norm_eps": 1e-05, "rope_theta": 1000000.0, "sliding_window": 8192,
                "tied_embeddings": true, "ada_rms_norm_t_cond": true,
                "ada_rms_norm_t_cond_dim": 32
            },
            "encoder_args": {
                "audio_encoding_args": {
                    "sampling_rate": 16000, "frame_rate": 12.5, "num_mel_bins": 128,
                    "hop_length": 160, "window_size": 400, "chunk_length_s": null,
                    "global_log_mel_max": 1.5, "transcription_format": "streaming"
                },
                "dim": 1280, "n_layers": 32, "head_dim": 64, "hidden_dim": 5120,
                "n_heads": 32, "vocab_size": 131072, "n_kv_heads": 32,
                "use_biases": true, "use_cache": false, "rope_theta": 1000000.0,
                "causal": true, "norm_eps": 1e-05, "pos_embed": "rope",
                "max_source_positions": null, "ffn_type": "swiglu",
                "norm_type": "rms_norm", "sliding_window": 750, "downsample_factor": 4
            },
            "model_type": "voxtral_realtime",
            "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}
        }"#,
        )
        .unwrap();
        let config = VoxtralRealtimeConfig::from_json(&root).expect("pinned config parses");
        assert_eq!(config.encoder.dim, 1280);
        assert_eq!(config.encoder.sliding_window, 750);
        assert_eq!(config.decoder.vocab_size, 131_072);
        assert_eq!(config.decoder.sliding_window, 8192);
        assert_eq!(config.adapter_dim, 3072);
        assert_eq!(config.quantization.bits, 4);
        assert_eq!(config.quantization.group_size, 64);
        // A second copy with audio_encoding_args lifted to the top level
        // (the other placement the reference accepts).
        let mut lifted = root.clone();
        lifted["audio_encoding_args"] = lifted["encoder_args"]["audio_encoding_args"].clone();
        assert!(VoxtralRealtimeConfig::from_json(&lifted).is_ok());

        // model_type and quantization refusals.
        for mutation in [
            ("model_type", serde_json::json!("voxtral_tts")),
            (
                "quantization",
                serde_json::json!({"group_size": 32, "bits": 4, "mode": "affine"}),
            ),
            (
                "quantization",
                serde_json::json!({"group_size": 64, "bits": 8, "mode": "affine"}),
            ),
            (
                "quantization",
                serde_json::json!({"group_size": 64, "bits": 4, "mode": "affine-custom"}),
            ),
        ] {
            let mut broken = root.clone();
            let field = mutation.0;
            broken[field] = mutation.1;
            assert!(
                load_config_value(&broken).is_err(),
                "refusal failed for {field}"
            );
        }
        // Missing quantization refuses.
        let mut broken = root.clone();
        broken.as_object_mut().unwrap().remove("quantization");
        assert!(load_config_value(&broken).is_err());

        // Encoder knob refusals: any present-but-different value.
        for (field, value) in [
            ("dim", serde_json::json!(1024)),
            ("n_layers", serde_json::json!(16)),
            ("head_dim", serde_json::json!(80)),
            ("hidden_dim", serde_json::json!(4096)),
            ("sliding_window", serde_json::json!(512)),
            ("downsample_factor", serde_json::json!(2)),
            ("rope_theta", serde_json::json!(10000.0)),
        ] {
            let mut broken = root.clone();
            broken["encoder_args"][field] = value;
            assert!(
                load_config_value(&broken).is_err(),
                "encoder refusal failed for {field}"
            );
        }
        // Frontend knob refusals.
        for (field, value) in [
            ("sampling_rate", serde_json::json!(24000)),
            ("num_mel_bins", serde_json::json!(80)),
            ("hop_length", serde_json::json!(128)),
            ("window_size", serde_json::json!(512)),
            ("global_log_mel_max", serde_json::json!(4.0)),
        ] {
            let mut broken = root.clone();
            broken["encoder_args"]["audio_encoding_args"][field] = value;
            assert!(
                load_config_value(&broken).is_err(),
                "frontend refusal failed for {field}"
            );
        }
        // Non-causal encoder refuses.
        let mut broken = root.clone();
        broken["encoder_args"]["causal"] = serde_json::json!(false);
        assert!(load_config_value(&broken).is_err(), "causal");

        // Decoder knob refusals.
        for (field, value) in [
            ("dim", serde_json::json!(2048)),
            ("n_layers", serde_json::json!(24)),
            ("n_kv_heads", serde_json::json!(32)),
            ("vocab_size", serde_json::json!(32768)),
            ("sliding_window", serde_json::json!(4096)),
        ] {
            let mut broken = root.clone();
            broken["decoder"][field] = value;
            assert!(
                load_config_value(&broken).is_err(),
                "decoder refusal failed for {field}"
            );
        }
        let mut broken = root.clone();
        broken["decoder"]["tied_embeddings"] = serde_json::json!(false);
        assert!(load_config_value(&broken).is_err(), "tied_embeddings");
        let mut broken = root.clone();
        broken["decoder"]["ada_rms_norm_t_cond"] = serde_json::json!(false);
        assert!(load_config_value(&broken).is_err(), "ada_rms_norm_t_cond");
        let mut broken = root.clone();
        broken.as_object_mut().unwrap().remove("decoder");
        assert!(load_config_value(&broken).is_err(), "missing decoder");
    }

    fn load_config_value(root: &Value) -> crate::Result<VoxtralRealtimeConfig> {
        VoxtralRealtimeConfig::from_json(root)
    }

    /// Real checkpoint gate. Loads encoder and decoder in two passes to
    /// bound resident memory, reproduces every fixture stage, and requires
    /// the exact transcript. Minutes on CPU f32; never downloads.
    #[test]
    #[ignore = "requires the pinned checkpoint in TURBOSPARK_VOXTRAL_REALTIME_MODEL_DIR"]
    fn pinned_checkpoint_matches_the_fixture_stages_and_transcript() {
        let Some(model_dir) = std::env::var_os("TURBOSPARK_VOXTRAL_REALTIME_MODEL_DIR") else {
            eprintln!("skipping: TURBOSPARK_VOXTRAL_REALTIME_MODEL_DIR is unset");
            return;
        };
        let model_dir = Path::new(&model_dir);
        let fixture = load_fixture();
        let waveform = smoke_waveform();

        // Stage 1: encoder. Dropped before the decoder loads.
        let adapter_rows: Vec<f32>;
        let adapter_len: usize;
        let conv_frames: usize;
        let mut measurements: Vec<(&'static str, f32, f32)> = Vec::new();
        {
            let shards = super::open_checkpoint(model_dir).expect("checkpoint opens");
            let config = super::load_config(model_dir).expect("config parses");
            assert_eq!(config.encoder.dim, 1280);
            assert_eq!(config.encoder.sliding_window, 750);
            assert_eq!(config.decoder.dim, 3072);
            assert_eq!(config.decoder.n_layers, 26);
            assert_eq!(config.decoder.vocab_size, 131_072);
            assert_eq!(
                config.quantization,
                crate::quant::QuantScheme {
                    bits: 4,
                    group_size: 64
                }
            );
            let encoder = super::encoder::AudioEncoder::load(
                &shards,
                &config.encoder,
                config.adapter_dim,
                config.quantization,
            )
            .expect("encoder loads");

            // Padding (exactness asserted by the always-on test).
            let n_delay = super::num_delay_tokens(super::TRANSCRIPTION_DELAY_MS);
            let padded = super::pad_audio_streaming(&waveform.samples, 32, n_delay as usize + 11);
            let (mel, mel_frames) = super::mel_spectrogram(&padded).expect("mel computes");
            let mel_spots: Spots = serde_json::from_value(fixture["mel"].clone()).unwrap();
            for (row_index, &frame) in mel_spots.rows.iter().enumerate() {
                for (column_index, &band) in mel_spots.columns.iter().enumerate() {
                    let diff = (mel[band * mel_frames + frame]
                        - mel_spots.values[row_index][column_index])
                        .abs();
                    assert!(diff < 2.0e-4, "mel [{frame},{band}] diff {diff}");
                }
            }

            let conv = encoder.conv_stem(&mel);
            conv_frames = conv.len() / config.encoder.dim;
            assert_eq!(
                conv_frames, fixture["encoder"]["conv_frames"],
                "conv frames"
            );
            let encoded = if conv_frames <= config.encoder.sliding_window {
                encoder
                    .encode_full(&conv, conv_frames)
                    .expect("encode_full")
            } else {
                encoder
                    .encode_chunked(&conv, conv_frames)
                    .expect("encode_chunked")
            };
            // The chunked path must agree with the full path on in-window
            // audio (the fixture generator measured a bitwise-zero
            // difference upstream).
            if conv_frames <= config.encoder.sliding_window {
                let chunked = encoder.encode_chunked(&conv, conv_frames).expect("chunked");
                let worst = encoded
                    .iter()
                    .zip(&chunked)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                eprintln!("encoder chunked-vs-full worst diff {worst:.3e}");
                measurements.push(("encoder_chunked_vs_full", worst, 1.0e-2));
            }
            let (adapter, len) = encoder.downsample_and_project(&encoded, conv_frames);
            adapter_len = len;
            assert_eq!(
                adapter_len, fixture["encoder"]["adapter_len"],
                "adapter len"
            );
            let spots: Spots =
                serde_json::from_value(fixture["encoder"]["adapter"].clone()).unwrap();
            let worst = compare_spots(&adapter, config.adapter_dim, &spots, "adapter", 2.0e-2);
            eprintln!("adapter worst spot diff {worst:.3e}");
            measurements.push(("adapter", worst, 2.0e-2));
            adapter_rows = adapter;
        }

        // Stage 2: decoder plus tokenizer.
        let shards = super::open_checkpoint(model_dir).expect("checkpoint opens");
        let config = super::load_config(model_dir).expect("config parses");
        let mut decoder =
            super::decoder::Decoder::load(&shards, &config.decoder, config.quantization)
                .expect("decoder loads");
        let n_delay = super::num_delay_tokens(super::TRANSCRIPTION_DELAY_MS);
        decoder.precompute_ada_scales(n_delay);
        let tokenizer = super::tokenizer::TekkenTokenizer::from_model_path(model_dir)
            .expect("tekken tokenizer loads");

        let prompt = super::prompt_ids(super::N_LEFT_PAD_TOKENS, n_delay);
        let expected_prompt: Vec<i64> = fixture["prompt"]["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_i64().unwrap())
            .collect();
        let computed_prompt: Vec<i64> = prompt.iter().map(|&id| i64::from(id)).collect();
        assert_eq!(computed_prompt, expected_prompt, "prompt token ids");
        let prompt_len = prompt.len();
        assert_eq!(prompt_len, fixture["prompt"]["prompt_len"]);

        let hidden = config.decoder.dim;
        let mut embeds = decoder.embed(&prompt).expect("prompt embeds");
        for (position, id) in prompt.iter().enumerate() {
            let tok_embed = decoder.embed(&[*id]).expect("tok embed");
            for column in 0..hidden {
                embeds[position * hidden + column] =
                    adapter_rows[position * hidden + column] + tok_embed[column];
            }
        }

        let (last, mut cache) = decoder.prefill(&embeds, prompt_len);
        let hidden_spots: VectorSpots =
            serde_json::from_value(fixture["prefill_hidden_last_row"].clone()).unwrap();
        let worst_hidden = compare_vector(&last, &hidden_spots, "prefill_hidden_last_row", 5.0e-2);
        eprintln!("prefill hidden worst diff {worst_hidden:.3e}");
        measurements.push(("prefill_hidden_last_row", worst_hidden, 5.0e-2));

        let mut logits = decoder.tied_logits(&last);
        let mut next = crate::nn::argmax(&logits) as i64;
        assert_eq!(
            next,
            fixture["first_logits"]["argmax"].as_i64().unwrap(),
            "first decision"
        );
        let watch: Vec<usize> = fixture["first_logits"]["watch_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_u64().unwrap() as usize)
            .collect();
        let watch_values: Vec<f64> = fixture["first_logits"]["watch_values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap())
            .collect();
        let mut worst_logit = 0.0f32;
        for (&index, &expected) in watch.iter().zip(&watch_values) {
            worst_logit = worst_logit.max((logits[index] - expected as f32).abs());
        }
        eprintln!("first logits worst watch diff {worst_logit:.3e}");
        measurements.push(("first_logits_watch", worst_logit, 2.0e-1));

        // Greedy loop mirroring the upstream for/else: fixture `steps[i]`
        // is the i-th decision; the loop ends on EOS or when the audio
        // positions run out, in which case the pending token is read.
        let n_audio_total = conv_frames / config.encoder.downsample_factor;
        assert_eq!(n_audio_total as i64, fixture["encoder"]["n_audio_total"]);
        let mut generated: Vec<i64> = Vec::new();
        let mut broke = false;
        for (step_index, pos) in (prompt_len..n_audio_total).enumerate() {
            assert!(
                step_index < fixture["steps"].as_array().unwrap().len(),
                "computed more decisions than the fixture records"
            );
            assert_eq!(
                next,
                fixture["steps"][step_index]["token"].as_i64().unwrap(),
                "step {step_index} argmax"
            );
            let reference_top = fixture["steps"][step_index]["top_logprob"]
                .as_f64()
                .unwrap() as f32;
            let computed = log_softmax_top(&logits, next as usize);
            measurements.push(("step_logprob", (computed - reference_top).abs(), 5.0e-2));
            generated.push(next);
            if next == super::EOS_TOKEN_ID as i64 || generated.len() > super::MAX_NEW_TOKENS {
                broke = true;
                break;
            }
            let token = i32::try_from(next).unwrap();
            let mut embed = decoder.embed(&[token]).expect("embed");
            if pos < adapter_len {
                for column in 0..hidden {
                    embed[column] += adapter_rows[pos * hidden + column];
                }
            }
            let step_hidden = decoder.step(&embed, pos, &mut cache);
            logits = decoder.tied_logits(&step_hidden);
            next = i64::from(crate::nn::argmax(&logits) as i32);
        }
        if !broke {
            generated.push(next);
        }
        let expected_generated: Vec<i64> = fixture["generated_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_i64().unwrap())
            .collect();
        assert_eq!(generated, expected_generated, "generated token ids");

        // Tekken byte decode of every generated id against the fixture.
        let bytes_hex: Vec<String> = fixture["token_bytes_hex"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect();
        for (&id, hex) in generated.iter().zip(&bytes_hex) {
            let expected = hex_decode_test(hex);
            assert_eq!(
                tokenizer.token_bytes(id as i32),
                expected,
                "token bytes for id {id}"
            );
        }

        // Full end to end on a fresh combined load (the staged decoder is
        // dropped first to bound peak memory).
        drop(decoder);
        drop(tokenizer);
        drop(shards);
        let model = super::VoxtralRealtime::load(model_dir).expect("combined model loads");
        assert_eq!(model.profile(), VOXTRAL_MINI_4B_REALTIME_4BIT);
        let transcript = model.transcribe(&waveform.samples).expect("transcribes");
        assert_eq!(
            transcript, fixture["transcript"],
            "transcript must match the reference exactly"
        );

        // Tolerances: each gate must absorb the measured f32-vs-reference
        // divergence with headroom while staying far below error
        // magnitudes a real layout or packing bug would produce.
        let mut worst_step = 0.0f32;
        for (stage, worst, gate) in &measurements {
            if *stage == "step_logprob" {
                worst_step = worst_step.max(*worst);
            }
            assert!(
                worst < gate,
                "stage {stage} worst diff {worst:.3e} exceeds gate {gate:.3e}"
            );
        }
        eprintln!("worst per-step logprob diff {worst_step:.3e}");
        for (stage, worst, gate) in &measurements {
            eprintln!("gate {stage}: worst {worst:.3e} < {gate:.3e}");
        }
    }

    fn log_softmax_top(logits: &[f32], index: usize) -> f32 {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|&value| (value - max).exp()).sum();
        logits[index] - max - sum.ln()
    }

    fn hex_decode_test(hex: &str) -> Vec<u8> {
        (0..hex.len() / 2)
            .map(|index| u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).unwrap())
            .collect()
    }
}
