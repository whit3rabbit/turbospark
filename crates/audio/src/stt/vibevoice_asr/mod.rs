//! VibeVoice-ASR offline transcription: acoustic and semantic tokenizer
//! encoders, speech connectors, and a Qwen2 language model.
//!
//! Reference: `mlx_audio/stt/models/vibevoice_asr/` (vibevoice_asr.py,
//! audio_encoder.py, config.py) at mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`.
//!
//! This port covers the offline single-window path of upstream `generate`
//! (resample to 24 kHz, encode the two tokenizer features, splice them into
//! the default chat prompt, prefill, greedily decode until one of the two
//! Qwen EOS tokens, decode with special tokens skipped, strip). The
//! streaming chunk protocol, diarization controls, and hotword folding are
//! refused; see the family README for the full refusal list.
//!
//! The upstream loader cannot load the pinned checkpoint: its
//! `AutoTokenizer` path validates `config.json` against transformers'
//! native VibeVoiceConfig and rejects the 1.5B geometry, so upstream
//! inference only works after a model_type override plus a direct
//! `Qwen2Tokenizer` load. This port reads the raw checkpoint tensors with
//! shape checks instead of going through the MLX module tree.

pub(crate) mod language_model;
pub(crate) mod resample_poly;
pub(crate) mod tokenizer_encoder;

use std::fs;
use std::path::Path;

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::nn::{bad_config, open_shards, Linear, RmsNorm};
use crate::{Result, SpeechError};

use language_model::Qwen2;
use tokenizer_encoder::TokenizerEncoder;

/// Immutable Hugging Face checkpoint profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VibeVoiceAsrProfile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

pub const VIBEVOICE_ASR_STREAMING_1_5B: VibeVoiceAsrProfile = VibeVoiceAsrProfile {
    name: "VibeVoice-ASR-Streaming 1.5B",
    repository: "microsoft/VibeVoice-ASR-Streaming-1.5B",
    revision: "4262d23d8a539a6530cf64fbd0b1751ef9a30853",
};

/// Max generated tokens for the offline window (upstream `max_tokens`).
const MAX_NEW_TOKENS: usize = 8192;
/// Upstream `stream_generate` stops on `<|endoftext|>` and `<|im_end|>`.
const EOS_TOKEN_IDS: [u32; 2] = [151_643, 151_645];

const SYSTEM_PROMPT: &str =
    "You are a helpful assistant that transcribes audio input into text output in JSON format.";
const TRANSCRIBE_KEYS: &str = "Start time, End time, Speaker ID, Content";

/// One speaker-tagged stretch of a decoded transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeakerSegment {
    pub speaker: u32,
    pub text: String,
}

/// Splits a decoded transcript on `Speaker N:` markers.
///
/// The pinned model emits `Speaker 0:`-style markers inline (upstream
/// leaves them in the text and its own `parse_transcription` only handles
/// JSON output, returning nothing for this format). This splitter is a
/// port-side utility: each `Speaker <digits>:` run starts a segment that
/// runs to the next marker or the end of the text.
pub fn parse_speaker_segments(text: &str) -> Vec<SpeakerSegment> {
    let mut markers: Vec<(usize, u32, usize)> = Vec::new();
    let bytes = text.as_bytes();
    let mut search = 0;
    while let Some(found) = text[search..].find("Speaker ") {
        let start = search + found;
        let mut cursor = start + "Speaker ".len();
        let mut speaker: Option<u32> = None;
        while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
            let digit = u32::from(bytes[cursor] - b'0');
            speaker = Some(match speaker {
                Some(current) => current
                    .checked_mul(10)
                    .and_then(|value| value.checked_add(digit))
                    .unwrap_or(u32::MAX),
                None => digit,
            });
            cursor += 1;
        }
        // A marker needs at least one digit and a colon.
        if let Some(speaker) = speaker.filter(|_| cursor > start + "Speaker ".len()) {
            if cursor < bytes.len() && bytes[cursor] == b':' {
                markers.push((start, speaker, cursor + 1));
            }
        }
        search = start + "Speaker ".len();
    }
    markers
        .iter()
        .enumerate()
        .map(|(index, (_start, speaker, text_start))| {
            let text_end = markers
                .get(index + 1)
                .map_or(text.len(), |(next_start, _, _)| *next_start);
            SpeakerSegment {
                speaker: *speaker,
                text: text[*text_start..text_end].trim().to_owned(),
            }
        })
        .collect()
}

/// Parsed and pinned checkpoint configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct VibeVoiceAsrConfig {
    pub sample_rate: u32,
    pub speech_tok_compress_ratio: u32,
    pub acoustic: TokenizerSideConfig,
    pub semantic: TokenizerSideConfig,
    pub decoder: DecoderConfig,
    pub chunk_frames: Option<u32>,
    pub lookahead_frames: Option<u32>,
}

/// One tokenizer encoder side (acoustic or semantic).
#[derive(Debug, Clone, PartialEq)]
pub struct TokenizerSideConfig {
    pub vae_dim: usize,
    pub n_filters: usize,
    pub ratios: Vec<usize>,
    pub depths: Vec<usize>,
    pub eps: f32,
}

/// Qwen2 decoder geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct DecoderConfig {
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

fn positive(value: &Value, key: &str) -> Result<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|number| usize::try_from(number).ok())
        .filter(|&number| number > 0)
        .ok_or_else(|| bad_config(key, "must be a positive integer"))
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
        Err(bad_config(key, "expected true"))
    }
}

impl VibeVoiceAsrConfig {
    pub fn from_json(root: &Value) -> Result<Self> {
        require(root, "model_type", "vibevoice")?;
        if root
            .get("architectures")
            .and_then(Value::as_array)
            .is_none_or(|items| {
                !items
                    .iter()
                    .any(|item| item.as_str() == Some("VibeVoiceForASRStreamingTraining"))
            })
        {
            return Err(bad_config(
                "architectures",
                "expected the VibeVoiceForASRStreamingTraining architecture",
            ));
        }
        require_true(root, "use_semantic_feature")?;

        // Tokenizer encoder geometry, pinned to the verified variant.
        let mut sides = Vec::with_capacity(2);
        for (key, model_type, vae_dim) in [
            (
                "acoustic_tokenizer_config",
                "vibevoice_acoustic_tokenizer",
                64usize,
            ),
            (
                "semantic_tokenizer_config",
                "vibevoice_semantic_tokenizer",
                128usize,
            ),
        ] {
            let side = root.get(key).ok_or_else(|| bad_config(key, "is missing"))?;
            require(side, "model_type", model_type)?;
            if side.get("causal").and_then(Value::as_bool) != Some(true) {
                return Err(unsupported("only the causal tokenizer encoder is verified"));
            }
            require(side, "mixer_layer", "depthwise_conv")?;
            require(side, "layernorm", "RMSNorm")?;
            require(side, "pad_mode", "constant")?;
            require_true(side, "conv_bias")?;
            require_true(side, "layernorm_elementwise_affine")?;
            if side.get("disable_last_norm").and_then(Value::as_bool) != Some(true) {
                return Err(unsupported(
                    "only disable_last_norm=true encoders are verified",
                ));
            }
            let ratios = side
                .get("encoder_ratios")
                .and_then(Value::as_array)
                .ok_or_else(|| bad_config("encoder_ratios", "must be an integer array"))?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|number| usize::try_from(number).ok())
                        .filter(|&number| number > 0)
                        .ok_or_else(|| bad_config("encoder_ratios", "must be positive integers"))
                })
                .collect::<Result<Vec<_>>>()?;
            let depths = side
                .get("encoder_depths")
                .and_then(Value::as_str)
                .ok_or_else(|| bad_config("encoder_depths", "must be a dash-separated string"))?
                .split('-')
                .map(|part| {
                    part.parse::<usize>()
                        .map_err(|_| bad_config("encoder_depths", "must contain positive integers"))
                })
                .collect::<Result<Vec<_>>>()?;
            let config_vae_dim = positive(side, "vae_dim")?;
            if config_vae_dim != vae_dim {
                return Err(bad_config(
                    "vae_dim",
                    format!("expected {vae_dim}, found {config_vae_dim}"),
                ));
            }
            sides.push(TokenizerSideConfig {
                vae_dim,
                n_filters: positive(side, "encoder_n_filters")?,
                ratios,
                depths,
                eps: root
                    .get(key)
                    .and_then(|side| side.get("layernorm_eps"))
                    .and_then(Value::as_f64)
                    .filter(|value| *value > 0.0)
                    .ok_or_else(|| bad_config("layernorm_eps", "must be positive"))?
                    as f32,
            });
        }

        let decoder_json = root
            .get("decoder_config")
            .ok_or_else(|| bad_config("decoder_config", "is missing"))?;
        require(decoder_json, "model_type", "qwen2")?;
        require(decoder_json, "hidden_act", "silu")?;
        if decoder_json
            .get("tie_word_embeddings")
            .and_then(Value::as_bool)
            != Some(true)
        {
            return Err(unsupported(
                "only tie_word_embeddings=true decoders are verified; the tied embedding \
                 matrix is the logits head",
            ));
        }
        if decoder_json
            .get("rope_scaling")
            .is_some_and(|value| !value.is_null())
        {
            return Err(unsupported(
                "rope_scaling is refused; the pinned decoder uses plain RoPE",
            ));
        }
        if decoder_json
            .get("use_sliding_window")
            .and_then(Value::as_bool)
            == Some(true)
            || decoder_json
                .get("sliding_window")
                .is_some_and(|value| !value.is_null())
        {
            return Err(unsupported("sliding-window attention is refused"));
        }
        let decoder = DecoderConfig {
            hidden_size: positive(decoder_json, "hidden_size")?,
            intermediate_size: positive(decoder_json, "intermediate_size")?,
            num_hidden_layers: positive(decoder_json, "num_hidden_layers")?,
            num_attention_heads: positive(decoder_json, "num_attention_heads")?,
            num_key_value_heads: positive(decoder_json, "num_key_value_heads")?,
            vocab_size: positive(decoder_json, "vocab_size")?,
            rms_norm_eps: decoder_json
                .get("rms_norm_eps")
                .and_then(Value::as_f64)
                .filter(|value| *value > 0.0)
                .ok_or_else(|| bad_config("rms_norm_eps", "must be positive"))?
                as f32,
            rope_theta: decoder_json
                .get("rope_theta")
                .and_then(Value::as_f64)
                .filter(|value| *value > 0.0)
                .ok_or_else(|| bad_config("rope_theta", "must be positive"))?
                as f32,
        };
        if decoder.hidden_size % decoder.num_attention_heads != 0
            || decoder.num_attention_heads % decoder.num_key_value_heads != 0
        {
            return Err(bad_config(
                "decoder_config",
                "attention head geometry is inconsistent",
            ));
        }

        let sample_rate = positive(root, "target_sample_rate")?;
        let speech_tok_compress_ratio = positive(root, "speech_tok_compress_ratio")?;
        Ok(Self {
            sample_rate: u32::try_from(sample_rate)
                .map_err(|_| bad_config("target_sample_rate", "exceeds u32"))?,
            speech_tok_compress_ratio: u32::try_from(speech_tok_compress_ratio)
                .map_err(|_| bad_config("speech_tok_compress_ratio", "exceeds u32"))?,
            acoustic: sides[0].clone(),
            semantic: sides[1].clone(),
            decoder,
            chunk_frames: root
                .get("chunk_frames")
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok()),
            lookahead_frames: root
                .get("lookahead_frames")
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok()),
        })
    }

    /// Parses `preprocessor_config.json` and refuses the knob values this
    /// port has not verified.
    pub fn from_preprocessor(root: &Value) -> Result<(u32, u32)> {
        let sample_rate = positive(root, "target_sample_rate")?;
        let compress_ratio = positive(root, "speech_tok_compress_ratio")?;
        if root.get("normalize_audio").and_then(Value::as_bool) != Some(false) {
            return Err(unsupported(
                "normalize_audio must be false: the pinned streaming checkpoint trains with \
                 normalization off and the reference loudness normalizer path is unverified",
            ));
        }
        Ok((
            u32::try_from(sample_rate)
                .map_err(|_| bad_config("target_sample_rate", "exceeds u32"))?,
            u32::try_from(compress_ratio)
                .map_err(|_| bad_config("speech_tok_compress_ratio", "exceeds u32"))?,
        ))
    }
}

/// `Linear -> RMSNorm -> Linear` projector to the LM width.
struct SpeechConnector {
    fc1: Linear,
    norm: RmsNorm,
    fc2: Linear,
}

impl SpeechConnector {
    fn load(
        files: &[SafetensorsFile],
        prefix: &str,
        input_dim: usize,
        hidden: usize,
    ) -> Result<Self> {
        Ok(Self {
            fc1: tokenizer_encoder::load_linear_sharded(
                files,
                &format!("{prefix}.fc1"),
                input_dim,
                hidden,
                true,
            )?,
            norm: tokenizer_encoder::load_rms_norm_sharded(
                files,
                &format!("{prefix}.norm"),
                hidden,
                1.0e-6,
            )?,
            fc2: tokenizer_encoder::load_linear_sharded(
                files,
                &format!("{prefix}.fc2"),
                hidden,
                hidden,
                true,
            )?,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut hidden = self.fc1.forward(x, rows);
        self.norm.apply(&mut hidden, rows);
        self.fc2.forward(&hidden, rows)
    }
}

/// Speech features for one window: the per-side latents, the per-side
/// connector outputs, and their sum.
pub struct SpeechEncoding {
    pub frames: usize,
    /// `[frames, 64]` acoustic tokenizer latents.
    pub acoustic_tokens: Vec<f32>,
    /// `[frames, 128]` semantic tokenizer latents.
    pub semantic_tokens: Vec<f32>,
    /// `[frames, hidden]` combined features the prompt splices in.
    pub combined: Vec<f32>,
}

/// Loaded VibeVoice-ASR model (offline single-window path).
pub struct VibeVoiceAsr {
    config: VibeVoiceAsrConfig,
    acoustic_tokenizer: TokenizerEncoder,
    semantic_tokenizer: TokenizerEncoder,
    acoustic_connector: SpeechConnector,
    semantic_connector: SpeechConnector,
    language_model: Qwen2,
    tokenizer: turbospark_tokenizer::Tokenizer,
    speech_pad_id: i32,
}

impl VibeVoiceAsr {
    /// Load the pinned profile from an already-downloaded model folder.
    /// Never downloads; refuses MLX-quantized tensors and unverified
    /// config knobs.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config_json = fs::read_to_string(model_dir.join("config.json")).map_err(|error| {
            SpeechError::Input {
                why: format!("cannot read config.json: {error}"),
            }
        })?;
        let root: Value = serde_json::from_str(&config_json)
            .map_err(|error| bad_config("config.json", error.to_string()))?;
        let mut config = VibeVoiceAsrConfig::from_json(&root)?;
        let preprocessor: Value = serde_json::from_slice(
            &fs::read(model_dir.join("preprocessor_config.json")).map_err(|error| {
                SpeechError::Input {
                    why: format!("cannot read preprocessor_config.json: {error}"),
                }
            })?,
        )
        .map_err(|error| bad_config("preprocessor_config.json", error.to_string()))?;
        let (sample_rate, compress_ratio) = VibeVoiceAsrConfig::from_preprocessor(&preprocessor)?;
        if sample_rate != config.sample_rate || compress_ratio != config.speech_tok_compress_ratio {
            return Err(bad_config(
                "preprocessor_config.json",
                "sample rate or compression ratio disagrees with config.json",
            ));
        }
        // Streaming protocol metadata is parsed for provenance only; the
        // chunked streaming path is refused (README).
        config.chunk_frames = preprocessor
            .get("chunk_frames")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok());
        config.lookahead_frames = preprocessor
            .get("lookahead_frames")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok());

        let shards = open_shards(model_dir)?;
        // Refuse MLX affine-quantized checkpoints: only the plain BF16
        // distribution is verified.
        for file in &shards {
            for name in file.tensor_names() {
                if name.ends_with(".scales") || name.ends_with(".biases") {
                    return Err(unsupported(format!(
                        "tensor {name} indicates an MLX affine-quantized checkpoint; this \
                         port verifies only the plain BF16 pinned distribution"
                    )));
                }
            }
        }

        let decoder = config.decoder.clone();
        let acoustic = TokenizerEncoder::load(
            &shards,
            "model.acoustic_tokenizer.encoder",
            config.acoustic.vae_dim,
            config.acoustic.n_filters,
            &config.acoustic.ratios,
            &config.acoustic.depths,
            config.acoustic.eps,
            true,
        )?;
        let semantic = TokenizerEncoder::load(
            &shards,
            "model.semantic_tokenizer.encoder",
            config.semantic.vae_dim,
            config.semantic.n_filters,
            &config.semantic.ratios,
            &config.semantic.depths,
            config.semantic.eps,
            true,
        )?;
        let acoustic_connector = SpeechConnector::load(
            &shards,
            "model.acoustic_connector",
            config.acoustic.vae_dim,
            decoder.hidden_size,
        )?;
        let semantic_connector = SpeechConnector::load(
            &shards,
            "model.semantic_connector",
            config.semantic.vae_dim,
            decoder.hidden_size,
        )?;
        let language_model = Qwen2::load(
            &shards,
            "model.language_model",
            decoder.hidden_size,
            decoder.intermediate_size,
            decoder.num_hidden_layers,
            decoder.num_attention_heads,
            decoder.num_key_value_heads,
            decoder.vocab_size,
            decoder.rms_norm_eps,
            decoder.rope_theta,
        )?;
        let tokenizer = crate::stt::qwen3_asr::load_tokenizer(model_dir)?;
        let resolve = |token: &str| -> Result<i32> {
            tokenizer
                .token_to_id(token)
                .and_then(|id| i32::try_from(id).ok())
                .ok_or_else(|| SpeechError::Input {
                    why: format!("tokenizer is missing the required token {token}"),
                })
        };
        // The speech delimiters appear in the prompt as literal text; the
        // pad id is the one used as a token id. Resolving all three pins
        // the tokenizer's special-token set at load.
        let _ = resolve("<|object_ref_start|>")?;
        let _ = resolve("<|object_ref_end|>")?;
        let speech_pad_id = resolve("<|box_start|>")?;
        Ok(Self {
            config,
            acoustic_tokenizer: acoustic,
            semantic_tokenizer: semantic,
            acoustic_connector,
            semantic_connector,
            language_model,
            tokenizer,
            speech_pad_id,
        })
    }

    pub fn profile(&self) -> VibeVoiceAsrProfile {
        VIBEVOICE_ASR_STREAMING_1_5B
    }

    pub fn config(&self) -> &VibeVoiceAsrConfig {
        &self.config
    }

    /// Resamples mono 16 kHz input to the model's 24 kHz rate with the
    /// pinned kaiser-best polyphase filter (the mlx-audio frontend).
    pub fn resample_input(&self, samples: &[f32]) -> Result<Vec<f32>> {
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(SpeechError::Input {
                why: "waveform contains non-finite samples".into(),
            });
        }
        resample_poly::resample_mono_polyphase(samples, 16_000, self.config.sample_rate)
            .map_err(|error| SpeechError::Audio(error.to_string()))
    }

    /// Encodes one 24 kHz mono window (mirrors `Model.encode_speech`).
    pub fn encode_speech(&self, audio_24k: &[f32]) -> Result<SpeechEncoding> {
        if audio_24k.is_empty() {
            return Err(SpeechError::Input {
                why: "VibeVoice cannot transcribe empty audio".into(),
            });
        }
        let acoustic_tokens = self.acoustic_tokenizer.forward(audio_24k);
        let frames = acoustic_tokens.len() / self.config.acoustic.vae_dim;
        let semantic_tokens = self.semantic_tokenizer.forward(audio_24k);
        if semantic_tokens.len() / self.config.semantic.vae_dim != frames {
            return Err(SpeechError::Tensor {
                name: "tokenizer encoder outputs".into(),
                why: "acoustic and semantic encoders disagree on frame count".into(),
            });
        }
        let acoustic_features = self.acoustic_connector.forward(&acoustic_tokens, frames);
        let semantic_features = self.semantic_connector.forward(&semantic_tokens, frames);
        let mut combined = vec![0.0f32; frames * self.config.decoder.hidden_size];
        for index in 0..combined.len() {
            combined[index] = acoustic_features[index] + semantic_features[index];
        }
        Ok(SpeechEncoding {
            frames,
            acoustic_tokens,
            semantic_tokens,
            combined,
        })
    }

    /// Builds the default offline prompt for `frames` speech frames and the
    /// window duration in seconds; returns the token ids and the positions
    /// carrying speech features (mirrors `_build_prompt_tokens` with no
    /// context).
    pub fn build_prompt(
        &self,
        frames: usize,
        audio_duration: f64,
    ) -> Result<(Vec<i32>, Vec<usize>)> {
        if frames == 0 {
            return Err(SpeechError::Input {
                why: "no speech frames to splice into the prompt".into(),
            });
        }
        let user = format!(
            "<|object_ref_start|>{}<|object_ref_end|>\nThis is a {audio_duration:.2} seconds \
             audio, please transcribe it with these keys: {TRANSCRIBE_KEYS}",
            "<|box_start|>".repeat(frames),
        );
        let prompt = format!(
            "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n\
             <|im_start|>assistant\n"
        );
        let encoded = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|error| SpeechError::Input {
                why: format!("VibeVoice prompt tokenization failed: {error}"),
            })?;
        let token_ids = encoded
            .get_ids()
            .iter()
            .map(|&id| {
                i32::try_from(id).map_err(|_| SpeechError::Input {
                    why: "VibeVoice prompt token id exceeds signed 32-bit range".into(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let pad_positions = token_ids
            .iter()
            .enumerate()
            .filter_map(|(position, &id)| (id == self.speech_pad_id).then_some(position))
            .collect::<Vec<_>>();
        if pad_positions.len() != frames {
            return Err(SpeechError::Input {
                why: format!(
                    "prompt has {} speech placeholders for {frames} encoded frames",
                    pad_positions.len()
                ),
            });
        }
        Ok((token_ids, pad_positions))
    }

    /// Merges text embeddings with speech features at the pad positions.
    fn merge_embeddings(
        &self,
        prompt_ids: &[i32],
        pad_positions: &[usize],
        speech: &SpeechEncoding,
    ) -> Result<Vec<f32>> {
        let mut embeds = self.language_model.embed(prompt_ids)?;
        let hidden = self.config.decoder.hidden_size;
        for (index, &position) in pad_positions.iter().enumerate() {
            let source = &speech.combined[index * hidden..(index + 1) * hidden];
            embeds[position * hidden..(position + 1) * hidden].copy_from_slice(source);
        }
        Ok(embeds)
    }

    /// Greedy decode from pre-filled prompt embeddings; returns the
    /// generated ids and one record per generated token (argmax id and its
    /// log-probability after the softmax the reference applies).
    fn greedy_from_embeddings(&self, embeds: &[f32], rows: usize) -> Result<GreedyRun> {
        let (_last_hidden, mut cache) = self.language_model.prefill(embeds, rows);
        let first_logits = self.language_model.tied_logits(&_last_hidden);
        let mut generated: Vec<i32> = Vec::new();
        let mut steps: Vec<StepRecord> = Vec::new();
        let mut next = crate::nn::argmax(&first_logits);
        for _ in 0..MAX_NEW_TOKENS {
            if EOS_TOKEN_IDS.contains(&(next as u32)) {
                break;
            }
            let token = i32::try_from(next).map_err(|_| SpeechError::Input {
                why: "VibeVoice generated token id exceeds signed 32-bit range".into(),
            })?;
            generated.push(token);
            let embedding = self.language_model.embed(&[token])?;
            let hidden = self.language_model.step(&embedding, &mut cache);
            let logits = self.language_model.tied_logits(&hidden);
            let argmax = crate::nn::argmax(&logits);
            steps.push(StepRecord {
                token: argmax as u32,
                top_logit: logits[argmax],
                top_logprob: log_softmax_top(&logits, argmax),
            });
            next = argmax;
        }
        Ok(GreedyRun { generated, steps })
    }

    /// Transcribes one mono 16 kHz waveform through the offline path and
    /// returns the stripped transcript text.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        let audio_24k = self.resample_input(samples)?;
        let speech = self.encode_speech(&audio_24k)?;
        let duration = audio_24k.len() as f64 / f64::from(self.config.sample_rate);
        let (prompt_ids, pad_positions) = self.build_prompt(speech.frames, duration)?;
        let embeds = self.merge_embeddings(&prompt_ids, &pad_positions, &speech)?;
        let run = self.greedy_from_embeddings(&embeds, prompt_ids.len())?;
        let ids: Vec<u32> = run
            .generated
            .iter()
            .map(|&id| {
                u32::try_from(id).map_err(|_| SpeechError::Input {
                    why: "generated token id does not fit the tokenizer interface".into(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let decoded = self
            .tokenizer
            .decode(&ids, true)
            .map_err(|error| SpeechError::Input {
                why: format!("VibeVoice output detokenization failed: {error}"),
            })?;
        Ok(decoded.trim().to_owned())
    }
}

/// One greedy step: the argmax token and its confidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepRecord {
    pub token: u32,
    pub top_logit: f32,
    pub top_logprob: f32,
}

/// Result of the offline greedy decode over one window.
pub struct GreedyRun {
    pub generated: Vec<i32>,
    pub steps: Vec<StepRecord>,
}

fn log_softmax_top(logits: &[f32], index: usize) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&value| (value - max).exp()).sum();
    logits[index] - max - sum.ln()
}

#[cfg(test)]
mod tests {
    use super::{
        parse_speaker_segments, VibeVoiceAsr, VibeVoiceAsrConfig, VIBEVOICE_ASR_STREAMING_1_5B,
    };
    use serde_json::Value;
    use std::path::Path;

    const FIXTURE: &str = include_str!("../../../testdata/vibevoice_asr_reference.json");

    fn load_fixture() -> Value {
        serde_json::from_str(FIXTURE).expect("valid vibevoice_asr reference fixture")
    }

    #[derive(serde::Deserialize)]
    struct Spots {
        shape: Vec<usize>,
        rows: Vec<usize>,
        columns: Vec<usize>,
        values: Vec<Vec<f32>>,
    }

    fn compare_spots(actual: &[f32], spots: &Spots, label: &str, gate: f32) -> f32 {
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
                let diff = (actual[row * columns + column] - expected).abs();
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

    #[test]
    fn profile_pin_is_immutable() {
        assert_eq!(
            VIBEVOICE_ASR_STREAMING_1_5B.repository,
            "microsoft/VibeVoice-ASR-Streaming-1.5B"
        );
        assert_eq!(
            VIBEVOICE_ASR_STREAMING_1_5B.revision,
            "4262d23d8a539a6530cf64fbd0b1751ef9a30853"
        );
    }

    #[test]
    fn fixture_provenance_pins_the_reference_run() {
        let fixture = load_fixture();
        assert_eq!(
            fixture["provenance"]["revision"],
            "4262d23d8a539a6530cf64fbd0b1751ef9a30853"
        );
        assert_eq!(
            fixture["provenance"]["source"],
            "mlx-audio vibevoice_asr at commit e1b19b9054bf163f5d812221a54fcc346f1890e9"
        );
        assert_eq!(
            fixture["provenance"]["repository"],
            "microsoft/VibeVoice-ASR-Streaming-1.5B"
        );
    }

    #[test]
    fn fixture_transcript_carries_the_speaker_marker() {
        let fixture = load_fixture();
        assert_eq!(
            fixture["transcript"],
            "Speaker 0:The quick brown fox jumps over the lazy dog."
        );
        assert_eq!(
            fixture["transcript_raw"],
            " Speaker 0:The quick brown fox jumps over the lazy dog."
        );
        let generated: Vec<i64> = fixture["generated_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_i64().unwrap())
            .collect();
        // The final token is the streaming chunk delimiter the reference
        // emits before EOS; the text decoder drops it as a special token.
        assert_eq!(generated.last(), Some(&151_665));
        assert_eq!(generated.len(), 14);
    }

    #[test]
    fn speaker_marker_parsing_splits_and_trims() {
        let segments = parse_speaker_segments(
            " Speaker 0:The quick brown fox. Speaker 1: jumps over the lazy dog.",
        );
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].speaker, 0);
        assert_eq!(segments[0].text, "The quick brown fox.");
        assert_eq!(segments[1].speaker, 1);
        assert_eq!(segments[1].text, "jumps over the lazy dog.");

        let none = parse_speaker_segments("plain text without markers");
        assert!(none.is_empty());

        let trailing = parse_speaker_segments("Speaker 12:hello");
        assert_eq!(trailing.len(), 1);
        assert_eq!(trailing[0].speaker, 12);
        assert_eq!(trailing[0].text, "hello");

        // "Speaker" without digits or without a colon is plain text.
        assert!(parse_speaker_segments("Speaker: x").is_empty());
        assert!(parse_speaker_segments("Speaking loudly: x").is_empty());
        let two_digit_overflow = parse_speaker_segments("Speaker 99999999999999999999: x");
        assert_eq!(two_digit_overflow[0].speaker, u32::MAX);
    }

    #[test]
    fn pinned_config_parses_and_refuses_unverified_knobs() {
        let mut root = serde_json::json!({
            "model_type": "vibevoice",
            "architectures": ["VibeVoiceForASRStreamingTraining"],
            "use_semantic_feature": true,
            "target_sample_rate": 24000,
            "speech_tok_compress_ratio": 3200,
            "acoustic_tokenizer_config": {
                "model_type": "vibevoice_acoustic_tokenizer",
                "causal": true,
                "mixer_layer": "depthwise_conv",
                "layernorm": "RMSNorm",
                "pad_mode": "constant",
                "conv_bias": true,
                "layernorm_elementwise_affine": true,
                "disable_last_norm": true,
                "vae_dim": 64,
                "encoder_n_filters": 32,
                "encoder_ratios": [8, 5, 5, 4, 2, 2],
                "encoder_depths": "3-3-3-3-3-3-8",
                "layernorm_eps": 1e-5
            },
            "semantic_tokenizer_config": {
                "model_type": "vibevoice_semantic_tokenizer",
                "causal": true,
                "mixer_layer": "depthwise_conv",
                "layernorm": "RMSNorm",
                "pad_mode": "constant",
                "conv_bias": true,
                "layernorm_elementwise_affine": true,
                "disable_last_norm": true,
                "vae_dim": 128,
                "encoder_n_filters": 32,
                "encoder_ratios": [8, 5, 5, 4, 2, 2],
                "encoder_depths": "3-3-3-3-3-3-8",
                "layernorm_eps": 1e-5
            },
            "decoder_config": {
                "model_type": "qwen2",
                "hidden_act": "silu",
                "hidden_size": 1536,
                "intermediate_size": 8960,
                "num_hidden_layers": 28,
                "num_attention_heads": 12,
                "num_key_value_heads": 2,
                "vocab_size": 151936,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "tie_word_embeddings": true
            }
        });
        let config = VibeVoiceAsrConfig::from_json(&root).expect("pinned config parses");
        assert_eq!(config.acoustic.vae_dim, 64);
        assert_eq!(config.semantic.vae_dim, 128);
        assert_eq!(config.decoder.num_key_value_heads, 2);
        assert_eq!(config.decoder.vocab_size, 151936);
        assert_eq!(config.sample_rate, 24000);

        // model_type refusals.
        for (field, value) in [
            ("model_type", Value::String("vibevoice_tts".into())),
            ("use_semantic_feature", Value::Bool(false)),
        ] {
            let mut broken = root.clone();
            broken[field] = value;
            assert!(VibeVoiceAsrConfig::from_json(&broken).is_err(), "{field}");
        }
        let mut broken = root.clone();
        broken["architectures"] = serde_json::json!(["VibeVoiceForTTSTraining"]);
        assert!(
            VibeVoiceAsrConfig::from_json(&broken).is_err(),
            "architectures"
        );

        // Encoder knob refusals.
        for field in [
            "causal",
            "conv_bias",
            "layernorm_elementwise_affine",
            "disable_last_norm",
        ] {
            let mut broken = root.clone();
            broken["acoustic_tokenizer_config"][field] = Value::Bool(false);
            assert!(VibeVoiceAsrConfig::from_json(&broken).is_err(), "{field}");
        }
        for (field, value) in [
            ("mixer_layer", Value::String("regular_conv".into())),
            ("layernorm", Value::String("LayerNorm".into())),
            ("pad_mode", Value::String("reflect".into())),
            (
                "model_type",
                Value::String("vibevoice_semantic_tokenizer".into()),
            ),
        ] {
            let mut broken = root.clone();
            broken["acoustic_tokenizer_config"][field] = value;
            assert!(VibeVoiceAsrConfig::from_json(&broken).is_err(), "{field}");
        }

        // Decoder knob refusals.
        {
            let mut broken = root.clone();
            broken["decoder_config"]["tie_word_embeddings"] = Value::Bool(false);
            assert!(
                VibeVoiceAsrConfig::from_json(&broken).is_err(),
                "tie_word_embeddings"
            );
        }
        {
            let mut broken = root.clone();
            broken["decoder_config"]["rope_scaling"] = serde_json::json!({"type": "yarn"});
            assert!(
                VibeVoiceAsrConfig::from_json(&broken).is_err(),
                "rope_scaling"
            );
        }
        {
            let mut broken = root.clone();
            broken["decoder_config"]["use_sliding_window"] = Value::Bool(true);
            assert!(
                VibeVoiceAsrConfig::from_json(&broken).is_err(),
                "sliding window"
            );
        }
        {
            let mut broken = root.clone();
            broken["decoder_config"]["hidden_act"] = Value::String("gelu".into());
            assert!(
                VibeVoiceAsrConfig::from_json(&broken).is_err(),
                "hidden_act"
            );
        }
        {
            let mut broken = root.clone();
            broken["decoder_config"]["model_type"] = Value::String("qwen3".into());
            assert!(
                VibeVoiceAsrConfig::from_json(&broken).is_err(),
                "decoder model_type"
            );
        }
    }

    #[test]
    fn preprocessor_normalization_is_refused() {
        let value: Value = serde_json::from_str(
            r#"{"target_sample_rate": 24000, "speech_tok_compress_ratio": 3200,
                "normalize_audio": true}"#,
        )
        .unwrap();
        let error = VibeVoiceAsrConfig::from_preprocessor(&value).unwrap_err();
        assert!(error.to_string().contains("normalize_audio"));
        let pinned: Value = serde_json::from_str(
            r#"{"target_sample_rate": 24000, "speech_tok_compress_ratio": 3200,
                "normalize_audio": false, "chunk_frames": 22, "lookahead_frames": 4}"#,
        )
        .unwrap();
        assert_eq!(
            VibeVoiceAsrConfig::from_preprocessor(&pinned).unwrap(),
            (24000, 3200)
        );
    }

    /// The always-on frontend gate: the Rust kaiser-best polyphase port
    /// must reproduce the reference 24 kHz waveform from the shared smoke
    /// WAV, sample for sample where the fixture recorded spots, with a
    /// matching byte digest.
    #[test]
    fn resampled_frontend_matches_the_reference_waveform() {
        let fixture = load_fixture();
        let audio_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_forced_aligner_reference.wav");
        let waveform = crate::wav::read_wav_f32(&audio_path).expect("reference WAV loads");
        assert_eq!(waveform.sample_rate, 16_000);
        let resampled =
            super::resample_poly::resample_mono_polyphase(&waveform.samples, 16_000, 24_000)
                .expect("resamples");
        let spots: VectorSpots =
            serde_json::from_value(fixture["resampled_24k"]["spots"].clone()).unwrap();
        let worst = compare_vector(&resampled, &spots, "resampled_24k", 1.0e-6);
        eprintln!("resampled_24k worst spot diff {worst:.3e}");
        // Byte-level digest: only passes when the f64 pipeline is exact.
        assert_eq!(
            turbospark_model_io::hash_data(
                &resampled
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>(),
            ),
            fixture["resampled_24k"]["sha256_f32le"],
            "resampled waveform digest (worst spot {worst:.3e})"
        );
    }

    /// Loads the pinned checkpoint when TURBOSPARK_VIBEVOICE_ASR_MODEL_DIR
    /// is set; the run reproduces every fixture stage and the exact
    /// transcript including the "Speaker 0:" prefix. Minutes on CPU f32;
    /// never downloads.
    #[test]
    #[ignore = "requires the pinned checkpoint in TURBOSPARK_VIBEVOICE_ASR_MODEL_DIR"]
    fn pinned_checkpoint_matches_the_fixture_stages_and_transcript() {
        let Some(model_dir) = std::env::var_os("TURBOSPARK_VIBEVOICE_ASR_MODEL_DIR") else {
            eprintln!("skipping: TURBOSPARK_VIBEVOICE_ASR_MODEL_DIR is unset");
            return;
        };
        let model = VibeVoiceAsr::load(Path::new(&model_dir)).expect("pinned checkpoint loads");
        assert_eq!(model.profile(), VIBEVOICE_ASR_STREAMING_1_5B);
        let fixture = load_fixture();

        let audio_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_forced_aligner_reference.wav");
        let waveform = crate::wav::read_wav_f32(&audio_path).expect("reference WAV loads");
        assert_eq!(waveform.sample_rate, 16_000);

        let audio_24k = model.resample_input(&waveform.samples).expect("resample");
        let spots: VectorSpots =
            serde_json::from_value(fixture["resampled_24k"]["spots"].clone()).unwrap();
        compare_vector(&audio_24k, &spots, "resampled_24k", 1.0e-6);
        assert_eq!(audio_24k.len(), fixture["resampled_24k"]["spots"]["length"]);

        let speech = model.encode_speech(&audio_24k).expect("encode");
        let acoustic: Spots = serde_json::from_value(fixture["acoustic_tokens"].clone()).unwrap();
        compare_spots(
            &speech.acoustic_tokens,
            &acoustic,
            "acoustic_tokens",
            5.0e-4,
        );
        let semantic: Spots = serde_json::from_value(fixture["semantic_tokens"].clone()).unwrap();
        compare_spots(
            &speech.semantic_tokens,
            &semantic,
            "semantic_tokens",
            5.0e-4,
        );
        let combined: Spots = serde_json::from_value(fixture["speech_features"].clone()).unwrap();
        compare_spots(&speech.combined, &combined, "speech_features", 5.0e-3);

        let duration = audio_24k.len() as f64 / 24_000.0;
        let (prompt_ids, pad_positions) =
            model.build_prompt(speech.frames, duration).expect("prompt");
        let expected_prompt: Vec<i64> = fixture["prompt"]["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_i64().unwrap())
            .collect();
        let computed_prompt: Vec<i64> = prompt_ids.iter().map(|&id| i64::from(id)).collect();
        assert_eq!(computed_prompt, expected_prompt, "prompt token ids");
        assert_eq!(
            pad_positions.len(),
            fixture["prompt"]["speech_pad_positions"],
            "pad positions"
        );

        let embeds = model
            .merge_embeddings(&prompt_ids, &pad_positions, &speech)
            .expect("merge");
        let rows = prompt_ids.len();
        let (last_hidden, mut cache) = model.language_model.prefill(&embeds, rows);
        let hidden_spots: VectorSpots =
            serde_json::from_value(fixture["prefill_hidden_last_row"].clone()).unwrap();
        compare_vector(
            &last_hidden,
            &hidden_spots,
            "prefill_hidden_last_row",
            2.0e-2,
        );

        let logits = model.language_model.tied_logits(&last_hidden);
        let argmax = crate::nn::argmax(&logits) as i64;
        assert_eq!(
            argmax,
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
        assert!(
            worst_logit < 3.0e-1,
            "first logits worst diff {worst_logit}"
        );

        // Greedy loop with the cached prefill.
        let mut next = crate::nn::argmax(&logits);
        let mut generated: Vec<i64> = Vec::new();
        let mut step_index = 0usize;
        let mut worst_logprob = 0.0f32;
        while next as u32 != 151_643
            && next as u32 != 151_645
            && generated.len() < super::MAX_NEW_TOKENS
        {
            let token = i32::try_from(next).unwrap();
            generated.push(i64::from(token));
            let embedding = model.language_model.embed(&[token]).unwrap();
            let hidden = model.language_model.step(&embedding, &mut cache);
            let step_logits = model.language_model.tied_logits(&hidden);
            let step_argmax = crate::nn::argmax(&step_logits);
            let record = &fixture["steps"][step_index];
            if record.is_null() {
                break;
            }
            assert_eq!(
                step_argmax as i64,
                record["token"].as_i64().unwrap(),
                "step {step_index} argmax"
            );
            let reference_top = record["top_logprob"].as_f64().unwrap() as f32;
            let computed = super::log_softmax_top(&step_logits, step_argmax);
            worst_logprob = worst_logprob.max((computed - reference_top).abs());
            step_index += 1;
            next = step_argmax;
        }
        let expected_generated: Vec<i64> = fixture["generated_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_i64().unwrap())
            .collect();
        assert_eq!(generated, expected_generated, "generated token ids");
        eprintln!("step logprob worst diff {worst_logprob:.3e}");
        assert!(
            worst_logprob < 5.0e-1,
            "step confidence worst diff {worst_logprob}"
        );

        let transcript = model.transcribe(&waveform.samples).expect("transcribes");
        assert_eq!(
            transcript, fixture["transcript"],
            "transcript must match the reference"
        );
    }
}
