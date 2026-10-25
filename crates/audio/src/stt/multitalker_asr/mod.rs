//! Multitalker Parakeet: speaker-conditioned cache-aware streaming RNN-T.
//!
//! Reference: NVIDIA NeMo Speech `218d0b443441b7d8aa1c832f47df310f186cf0a4`
//! (`EncDecMultiTalkerRNNTBPEModel`, `SpeakerKernelMixin`,
//! `ConformerEncoder.cache_aware_stream_step`) running
//! `nvidia/multitalker-parakeet-streaming-0.6b-v1` at
//! `8749fc71fd6e2d88ef230159bbf2aea69b524ee1`. This family departs from the
//! crate's mlx-audio 0.5.7 reference rule because mlx-audio has no
//! speaker-kernel model; see `README.md`.
//!
//! The checkpoint is a fine-tune of Nemotron's cache-aware FastConformer, so
//! the encoder, predictor and joint come from [`super::nemotron_asr`]. What
//! this module adds is the speaker conditioning NeMo applies as a forward
//! pre-hook on encoder layer 0:
//!
//! ```text
//! x = x + spk_ff(x * align(spk_targets, T, 1.0))
//! x = x + bg_ff (x * align(bg_targets,  T, 0.0))   // uses the updated x
//! ```
//!
//! where `spk_ff`/`bg_ff` are `Linear -> ReLU -> Linear` and `align` left-pads
//! a short mask with the default or keeps the last `T` values of a long one.
//! Each speaker stream keeps its own encoder caches and decoder state; the
//! weights are shared.

pub mod frontend;
pub mod scheduler;
pub mod seglst;

use std::path::Path;

use serde_json::Value;
use turbospark_audio::nemo_mel::{
    nemo_log_mel_spectrogram_with_padding, NemoMelNormalization, NemoMelOptions,
};
use turbospark_audio::stft::{StftOptions, StftPaddingMode};
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::models::stt::nemotron_asr::encoder::{
    EncoderStreamCache, NemotronEncoder, StreamGeometry,
};
pub use crate::models::stt::nemotron_asr::encoder::{StreamEncoderWeights, StreamLayerWeights};
use crate::models::stt::nemotron_asr::rnnt::{
    greedy_resume, GreedyStreamState, Joint, NemotronPredictor,
};
use crate::models::stt::nemotron_asr::NemotronAsrConfig;
use crate::nn::{symmetric_hann, Linear};
use crate::{Result, SpeechError};

/// A pinned checkpoint for this family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultitalkerProfile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

/// The Frozen E checkpoint. The repository ships only a `.nemo`; this crate
/// loads the F32 safetensors written by
/// `scripts/convert_multitalker_nemo.py convert`.
pub const MULTITALKER_PARAKEET_STREAMING_V1: MultitalkerProfile = MultitalkerProfile {
    name: "Multitalker Parakeet Streaming 0.6B v1",
    repository: "nvidia/multitalker-parakeet-streaming-0.6b-v1",
    revision: "8749fc71fd6e2d88ef230159bbf2aea69b524ee1",
};

/// `Linear -> ReLU -> Linear` speaker kernel (`spk_kernel_type: ff`).
struct SpeakerKernel {
    first: Linear,
    second: Linear,
}

impl SpeakerKernel {
    fn load(file: &SafetensorsFile, prefix: &str, hidden: usize) -> Result<Self> {
        Ok(Self {
            first: Linear::load(file, &format!("{prefix}.0"), hidden, hidden, true)?,
            second: Linear::load(file, &format!("{prefix}.3"), hidden, hidden, true)?,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut h = self.first.forward(x, rows);
        for v in &mut h {
            *v = v.max(0.0);
        }
        self.second.forward(&h, rows)
    }
}

/// NeMo `solve_length_mismatch`: a mask shorter than `rows` is left-padded
/// with `default`; a longer one keeps its last `rows` values.
pub fn align_mask(mask: &[f32], rows: usize, default: f32) -> Vec<f32> {
    if mask.len() >= rows {
        mask[mask.len() - rows..].to_vec()
    } else {
        let mut out = vec![default; rows - mask.len()];
        out.extend_from_slice(mask);
        out
    }
}

/// Whole-recording log-mel features, frame-major `[frames, mels]`, exactly
/// as NeMo's `CacheAwareStreamingAudioBuffer` holds them: centered STFT with
/// zero (constant) padding, no normalization, and the final centered frame
/// (past `samples / hop`) zeroed.
#[derive(Debug, Clone, PartialEq)]
pub struct Features {
    pub frames: usize,
    pub mels: usize,
    pub data: Vec<f32>,
}

impl Features {
    /// Frame-major rows `[start, start + count)`, with frames before 0
    /// filled with zeros (NeMo's zero pre-encode cache on the first chunk).
    pub fn frame_major(&self, start: isize, count: usize) -> Vec<f32> {
        let mut out = vec![0.0; count * self.mels];
        for i in 0..count {
            let frame = start + i as isize;
            if frame >= 0 && (frame as usize) < self.frames {
                let f = frame as usize;
                out[i * self.mels..(i + 1) * self.mels]
                    .copy_from_slice(&self.data[f * self.mels..(f + 1) * self.mels]);
            }
        }
        out
    }

    /// Channel-major `[mels, count]` view of frames `[start, start + count)`,
    /// the layout the diarizer takes.
    pub fn channel_major(&self, start: usize, count: usize) -> Vec<f32> {
        let mut out = vec![0.0; self.mels * count];
        for i in 0..count {
            let f = start + i;
            for m in 0..self.mels {
                out[m * count + i] = self.data[f * self.mels + m];
            }
        }
        out
    }
}

/// One chunk after subsampling and the pre-encode drop.
#[derive(Debug, Clone, PartialEq)]
pub struct PreEncoded {
    /// `[rows, hidden]`.
    pub data: Vec<f32>,
    pub rows: usize,
    /// Rows backed by real input (NeMo `length`).
    pub valid: usize,
}

/// One emitted token at a global encoder frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token {
    pub id: usize,
    pub frame: usize,
}

/// One speaker's streaming state.
#[derive(Clone)]
pub struct SpeakerStream {
    cache: EncoderStreamCache,
    decoder: GreedyStreamState,
    tokens: Vec<Token>,
    decoded_frames: usize,
}

impl SpeakerStream {
    /// Every token emitted so far.
    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }

    /// Encoder frames decoded so far (NeMo `dec_state.decoded_length`).
    pub fn decoded_frames(&self) -> usize {
        self.decoded_frames
    }

    /// Valid rows in the attention cache (NeMo `cache_last_channel_len`).
    pub fn cache_len(&self) -> usize {
        self.cache.attn_len
    }
}

/// Runs the streaming encoder layers for several speakers in one call, on
/// a device that holds each speaker's attention and conv caches. Not
/// bit-exact against the CPU `stream_layers`; an implementation documents
/// its tolerance.
pub trait SpeakerEncoder: Send {
    /// `inputs[b]` is the layer-0 input (`[rows, hidden]`, from
    /// [`MultitalkerAsr::apply_speaker_kernels`]) of the speaker whose
    /// caches live in slot `slots[b]`; `cache_lens[b]` is that speaker's
    /// valid attention-cache rows before this step and `valid` the chunk's
    /// real rows. Returns each speaker's `[rows, hidden]` encoder output in
    /// the same order. A slot's caches start zeroed and change only when
    /// the slot is stepped.
    fn stream_layers(
        &mut self,
        slots: &[usize],
        inputs: &[&[f32]],
        rows: usize,
        valid: usize,
        cache_lens: &[usize],
    ) -> Result<Vec<Vec<f32>>>;
}

/// Lends a device encoder to a session, so its resident weights outlive
/// one recording; the caller resets its caches between sessions.
impl<T: SpeakerEncoder + ?Sized> SpeakerEncoder for &mut T {
    fn stream_layers(
        &mut self,
        slots: &[usize],
        inputs: &[&[f32]],
        rows: usize,
        valid: usize,
        cache_lens: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        (**self).stream_layers(slots, inputs, rows, valid, cache_lens)
    }
}

/// One speaker step's result.
#[derive(Debug, Clone, PartialEq)]
pub struct StepOutput {
    /// Layer-0 input after the speaker kernels, `[rows, hidden]`.
    pub layer0_input: Vec<f32>,
    /// Encoder output `[rows, hidden]`; only the first `encoded_len` rows are decoded.
    pub encoded: Vec<f32>,
    pub encoded_len: usize,
    /// Tokens emitted by this step.
    pub tokens: Vec<Token>,
}

/// A loaded Multitalker Parakeet checkpoint.
pub struct MultitalkerAsr {
    config: NemotronAsrConfig,
    encoder: NemotronEncoder,
    geometry: StreamGeometry,
    spk_kernel: SpeakerKernel,
    bg_kernel: SpeakerKernel,
    predictor: NemotronPredictor,
    joint: Joint,
    punctuation: Vec<char>,
}

fn bad(field: &str, why: impl Into<String>) -> SpeechError {
    SpeechError::BadConfig {
        field: field.to_string(),
        why: why.into(),
    }
}

/// Checks the `speaker_kernels` block is the one implemented here:
/// feed-forward kernels on layer 0 with a background kernel.
fn check_speaker_kernels(value: &Value, hidden: usize) -> Result<()> {
    let k = value
        .get("speaker_kernels")
        .ok_or_else(|| bad("speaker_kernels", "missing from config.json"))?;
    let layers: Option<Vec<u64>> = k
        .get("layers")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_u64).collect());
    if k.get("type").and_then(Value::as_str) != Some("ff")
        || layers.as_deref() != Some(&[0])
        || k.get("background").and_then(Value::as_bool) != Some(true)
        || k.get("hidden").and_then(Value::as_u64) != Some(hidden as u64)
    {
        return Err(SpeechError::Unsupported {
            why: "only feed-forward speaker kernels on layer 0 with a background kernel \
                  (the Multitalker Parakeet v1 graph) are implemented"
                .into(),
        });
    }
    Ok(())
}

/// Reads `supported_punctuation`, the converter's copy of NeMo's
/// `extract_punctuation_from_vocab` set. Required: without it the
/// hypothesis text, and so the SegLST word split, differs from NeMo's.
fn supported_punctuation(value: &Value) -> Result<Vec<char>> {
    let field = "supported_punctuation";
    let list = value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| bad(field, "missing; reconvert with convert_multitalker_nemo.py"))?;
    list.iter()
        .map(|v| {
            let s = v.as_str().unwrap_or_default();
            let mut chars = s.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) => Ok(c),
                _ => Err(bad(field, format!("entry {v} is not a single character"))),
            }
        })
        .collect()
}

/// NeMo's `decode_tokens_to_str_with_strip_punctuation` after the
/// SentencePiece decode: `re.sub(r"(\s)(<punct>)", r"\2", text)`, which
/// deletes one whitespace character before each supported mark, scanning
/// left to right without overlap.
pub fn strip_space_before_punctuation(text: &str, punctuation: &[char]) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_whitespace() && chars.peek().is_some_and(|n| punctuation.contains(n)) {
            continue;
        }
        out.push(c);
    }
    out
}

impl MultitalkerAsr {
    /// Opens a converted snapshot (`config.json` with `model_type:
    /// "multitalker_asr"` and an F32 `model.safetensors`).
    pub fn open(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let value: Value = serde_json::from_slice(&std::fs::read(&config_path).map_err(|e| {
            bad(
                "config.json",
                format!("failed to read {}: {e}", config_path.display()),
            )
        })?)
        .map_err(|e| bad("config.json", format!("invalid JSON: {e}")))?;
        if value.get("model_type").and_then(Value::as_str) != Some("multitalker_asr") {
            return Err(SpeechError::Unsupported {
                why: "config is not a Multitalker Parakeet checkpoint".into(),
            });
        }
        let config = NemotronAsrConfig::from_json_prompt_free(&value)?;
        check_speaker_kernels(&value, config.encoder_hidden)?;
        let punctuation = supported_punctuation(&value)?;
        let file = SafetensorsFile::open(&model_dir.join("model.safetensors"))?;
        // Frozen E ran FP32; refuse quantized tensors rather than mis-decode them.
        if file.contains_tensor("encoder.layers.0.self_attn.linear_q.scales") {
            return Err(SpeechError::Unsupported {
                why: "quantized Multitalker weights are not supported; use the F32 conversion"
                    .into(),
            });
        }
        let encoder = NemotronEncoder::load(&file, &config)?;
        let geometry = encoder.stream_geometry(config.default_attention_context)?;
        let hidden = config.encoder_hidden;
        Ok(Self {
            spk_kernel: SpeakerKernel::load(&file, "spk_kernels.0", hidden)?,
            bg_kernel: SpeakerKernel::load(&file, "bg_spk_kernels.0", hidden)?,
            predictor: NemotronPredictor::load(
                &file,
                config.decoder_hidden,
                config.decoder_layers,
                config.vocabulary.len(),
            )?,
            joint: Joint::load(
                &file,
                hidden,
                config.decoder_hidden,
                config.joint_hidden,
                config.vocabulary.len() + 1,
            )?,
            encoder,
            geometry,
            config,
            punctuation,
        })
    }

    pub fn config(&self) -> &NemotronAsrConfig {
        &self.config
    }

    /// Encoder output rows per full chunk (NeMo `streaming_cfg.valid_out_len`).
    pub fn chunk_rows(&self) -> usize {
        self.geometry.chunk
    }

    /// Mel frames per full chunk step (`chunk_rows * subsampling_factor`).
    pub fn shift_frames(&self) -> usize {
        self.geometry.chunk * self.config.subsampling_factor
    }

    /// Pre-encode look-back in mel frames (`subsampling_factor + 1`).
    pub fn pre_encode_cache_frames(&self) -> usize {
        self.config.subsampling_factor + 1
    }

    /// Leading pre-encoded rows dropped every step
    /// (`1 + (pre_encode_cache - 1) / subsampling_factor`).
    pub fn drop_extra_pre_encoded(&self) -> usize {
        1 + (self.pre_encode_cache_frames() - 1) / self.config.subsampling_factor
    }

    /// The checkpoint's frontend: centered STFT with zero (constant) padding,
    /// no normalization, and NeMo's mask on frames past `samples / hop`.
    fn mel_options(&self) -> NemoMelOptions {
        NemoMelOptions {
            stft: StftOptions {
                fft_size: self.config.n_fft,
                hop: self.config.hop_samples,
                window: symmetric_hann(self.config.window_samples),
                center: true,
            },
            sample_rate: self.config.sample_rate,
            num_mels: self.config.num_mels,
            normalize: NemoMelNormalization::None,
            preemphasis: self.config.preemphasis,
            log_zero_guard_value: self.config.log_zero_guard_value,
            pad_to: 0,
            pad_value: 0.0,
            normalize_valid_frames: true,
        }
    }

    /// A frontend that turns PCM into the same frames as
    /// [`features`](Self::features) while it arrives, without the
    /// whole-file length cap. The caller checks the sample rate.
    pub fn live_frontend(&self) -> Result<frontend::LiveFrontend> {
        frontend::LiveFrontend::new(self.mel_options())
    }

    /// Whole-recording features for mono PCM at the model rate.
    pub fn features(&self, samples: &[f32], sample_rate: u32) -> Result<Features> {
        if sample_rate != self.config.sample_rate {
            return Err(SpeechError::Input {
                why: format!(
                    "Multitalker expects {} Hz PCM, received {sample_rate} Hz",
                    self.config.sample_rate
                ),
            });
        }
        if samples.is_empty() || samples.iter().any(|s| !s.is_finite()) {
            return Err(SpeechError::Input {
                why: "Multitalker input must be nonempty finite mono PCM".into(),
            });
        }
        let options = self.mel_options();
        let mel =
            nemo_log_mel_spectrogram_with_padding(samples, &options, StftPaddingMode::Constant)?;
        let frames = mel.len();
        Ok(Features {
            frames,
            mels: self.config.num_mels,
            data: mel.into_iter().flatten().collect(),
        })
    }

    /// A fresh speaker stream (zero caches, start-of-sequence decoder).
    pub fn new_stream(&self) -> SpeakerStream {
        SpeakerStream {
            cache: self.encoder.new_stream_cache(&self.geometry),
            decoder: GreedyStreamState::new(&self.predictor, self.config.vocabulary.len()),
            tokens: Vec::new(),
            decoded_frames: 0,
        }
    }

    /// Subsamples one frame-major chunk and drops `drop` leading rows. The
    /// result is shared by every speaker of the step.
    pub fn pre_encode(
        &self,
        chunk: &[f32],
        frames: usize,
        valid_frames: usize,
        drop: usize,
    ) -> Result<PreEncoded> {
        let (data, rows, valid) = self.encoder.pre_encode(chunk, frames, valid_frames, drop)?;
        Ok(PreEncoded { data, rows, valid })
    }

    /// Layer-0 input after the speaker kernels (NeMo's pre-hook).
    pub fn apply_speaker_kernels(&self, pre: &PreEncoded, spk: &[f32], bg: &[f32]) -> Vec<f32> {
        let hidden = self.config.encoder_hidden;
        let rows = pre.rows;
        let spk = align_mask(spk, rows, 1.0);
        let bg = align_mask(bg, rows, 0.0);
        let mut x = pre.data.clone();
        let masked = mask_rows(&x, &spk, hidden);
        add(&mut x, &self.spk_kernel.forward(&masked, rows));
        let masked = mask_rows(&x, &bg, hidden);
        add(&mut x, &self.bg_kernel.forward(&masked, rows));
        x
    }

    /// One speaker's step on a shared pre-encoded chunk with that speaker's
    /// target and background activity. `keep_all` is NeMo's
    /// `keep_all_outputs` (true only on the final chunk).
    pub fn step(
        &self,
        pre: &PreEncoded,
        spk: &[f32],
        bg: &[f32],
        stream: &mut SpeakerStream,
        keep_all: bool,
    ) -> Result<StepOutput> {
        let layer0_input = self.apply_speaker_kernels(pre, spk, bg);
        let encoded = self.encoder.stream_layers(
            &self.geometry,
            &layer0_input,
            pre.rows,
            pre.valid,
            &mut stream.cache,
        )?;
        self.decode_step(pre, layer0_input, encoded, stream, keep_all)
    }

    /// [`step`](Self::step) after a [`SpeakerEncoder`] ran the encoder
    /// layers for this speaker: advances the stream's attention-cache
    /// length as `stream_layers` does (the device holds the caches
    /// themselves), then decodes. `layer0_input` comes from
    /// [`apply_speaker_kernels`](Self::apply_speaker_kernels).
    pub fn step_encoded(
        &self,
        pre: &PreEncoded,
        layer0_input: Vec<f32>,
        encoded: Vec<f32>,
        stream: &mut SpeakerStream,
        keep_all: bool,
    ) -> Result<StepOutput> {
        if encoded.len() != pre.rows * self.config.encoder_hidden {
            return Err(SpeechError::Audio(format!(
                "device encoder returned {} values for {} rows",
                encoded.len(),
                pre.rows
            )));
        }
        stream.cache.attn_len = (stream.cache.attn_len + pre.rows).min(self.geometry.attn_frames);
        self.decode_step(pre, layer0_input, encoded, stream, keep_all)
    }

    /// The streaming encoder's weights, for a [`SpeakerEncoder`] device.
    pub fn stream_weights(&self) -> StreamEncoderWeights<'_> {
        self.encoder.stream_weights(&self.geometry)
    }

    /// Greedy RNN-T over this step's encoder rows, resumed from the
    /// stream's decoder state.
    fn decode_step(
        &self,
        pre: &PreEncoded,
        layer0_input: Vec<f32>,
        encoded: Vec<f32>,
        stream: &mut SpeakerStream,
        keep_all: bool,
    ) -> Result<StepOutput> {
        let encoded_len = if keep_all {
            pre.valid
        } else {
            pre.valid.min(self.geometry.chunk)
        };
        let tokens: Vec<Token> = greedy_resume(
            &self.predictor,
            &self.joint,
            &encoded,
            encoded_len,
            self.config.encoder_hidden,
            self.config.vocabulary.len(),
            self.config.max_symbols,
            &mut stream.decoder,
            stream.decoded_frames,
        )?
        .into_iter()
        .map(|(id, frame)| Token { id, frame })
        .collect();
        stream.decoded_frames += encoded_len;
        stream.tokens.extend_from_slice(&tokens);
        Ok(StepOutput {
            layer0_input,
            encoded,
            encoded_len,
            tokens,
        })
    }

    /// NeMo's hypothesis text for token ids: a SentencePiece-style decode
    /// (`▁` becomes a space, the first piece's leading space is dropped,
    /// `<unk>` decodes to ` ⁇ `), then [`strip_space_before_punctuation`].
    pub fn detokenize(&self, ids: &[usize]) -> String {
        let mut text = String::new();
        for &id in ids {
            match self.config.vocabulary.get(id).map(String::as_str) {
                Some("<unk>") => text.push_str(" \u{2047} "),
                Some(piece) => text.push_str(&piece.replace('\u{2581}', " ")),
                None => {}
            }
        }
        let text = text.strip_prefix(' ').unwrap_or(&text);
        strip_space_before_punctuation(text, &self.punctuation)
    }
}

fn mask_rows(x: &[f32], mask: &[f32], hidden: usize) -> Vec<f32> {
    let mut out = x.to_vec();
    for (row, m) in out.chunks_exact_mut(hidden).zip(mask) {
        for v in row {
            *v *= m;
        }
    }
    out
}

fn add(target: &mut [f32], values: &[f32]) {
    for (t, v) in target.iter_mut().zip(values) {
        *t += v;
    }
}

#[cfg(test)]
mod tests;
