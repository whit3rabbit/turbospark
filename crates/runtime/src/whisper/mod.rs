//! The whisper speech runner: open, mel windowing, the CPU reference
//! encoder path, and segment assembly.
//!
//! Family-gated by [`SpeechFamily::Whisper`](model_io::speech_family::SpeechFamily):
//! the runtime never dispatches on tensor names. This module is the CPU
//! reference execution path, which the speech-to-text design makes the
//! parity partner for the whisper Metal kernels; the Metal encoder path
//! lands with the shared encoder-block kernels it consumes.
//!
//! Windowing: 30-second windows (480000 samples) striding by 29 seconds,
//! so each window carries one second of overlap into the next and a word
//! cut at a window edge is whole inside the following window. Each window
//! decodes independently and emits one segment covering its exclusive
//! region; the final window covers its full length. A window whose decode
//! comes back empty contributes no segment. The same PCM always yields the
//! same segments.

pub mod decode;
#[cfg(target_os = "macos")]
pub mod metal;
#[cfg(test)]
mod tests;
pub mod weights;

pub use decode::{StopReason, WindowDecoder};

use std::fs;
use std::path::Path;
use std::sync::Mutex;

use audio::whisper::{whisper_log_mel_window, WHISPER_SAMPLE_RATE, WHISPER_WINDOW_SAMPLES};
use compute::vision::layer_norm;
use compute::whisper::{
    whisper_conv_frontend, whisper_encoder_layer, WhisperConvWeights, WhisperEncoderLayerWeights,
};
use model_io::safetensors::SafetensorsFile;
use model_io::speech_family::SpeechFamily;
use model_io::whisper_config::{WhisperConfig, WhisperSpecialTokens};
use tokenizer::Tokenizer;

use self::weights::WhisperWeights;

/// One transcript segment with absolute window offsets in seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct WhisperSegment {
    pub index: usize,
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub text: String,
}

/// A complete transcription of one PCM input.
#[derive(Debug, Clone, PartialEq)]
pub struct WhisperTranscription {
    pub segments: Vec<WhisperSegment>,
    /// The language token's code (`"en"` for English-only models, the
    /// detected or selected code otherwise).
    pub language: String,
}

/// Runtime runner for the whisper speech family. Device selection happens
/// at open: the Metal engine (`metal::WhisperMetalEngine`) is built when a
/// Metal device initializes and `TURBOSPARK_WHISPER_DEVICE` does not force
/// `cpu`; otherwise -- or on any engine error -- the runner executes the
/// CPU reference path. `transcribe` runs entirely on one device; each
/// device path is deterministic for the same PCM.
pub struct WhisperRunner {
    pub config: WhisperConfig,
    pub tokens: WhisperSpecialTokens,
    pub weights: WhisperWeights,
    pub tokenizer: Option<Tokenizer>,
    /// The Metal engine behind a mutex: `transcribe(&self)` takes the FFI
    /// contract's `&self`, and one engine serves one transcription at a
    /// time (the engine's caches and scratch are single-window state).
    pub(crate) metal: Option<Mutex<metal::WhisperMetalEngine>>,
}

/// Window stride: 30-second windows carrying one second of overlap.
pub(crate) const WINDOW_STRIDE_SAMPLES: usize =
    WHISPER_WINDOW_SAMPLES - WHISPER_SAMPLE_RATE as usize;

/// Phase timing for the Metal path, on when `TURBOSPARK_WHISPER_PROFILE=1`.
/// Printed to stderr, one line per window: mel, encode, cross caches, and
/// the decode loop with its token count. Attribution only; never a
/// benchmark row.
pub(crate) fn whisper_profile() -> bool {
    std::env::var("TURBOSPARK_WHISPER_PROFILE").as_deref() == Ok("1")
}

impl WhisperRunner {
    /// Opens a speech install directory containing `config.json`,
    /// `model.safetensors`, and `tokenizer.json`.
    pub fn open(model_dir: &Path) -> Result<Self, String> {
        let config_path = model_dir.join("config.json");
        let config_str = fs::read_to_string(&config_path)
            .map_err(|e| format!("failed to read {}: {e}", config_path.display()))?;
        let config = WhisperConfig::from_json_str(&config_str)
            .map_err(|e| format!("failed to parse {}: {e}", config_path.display()))?;

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("failed to load tokenizer.json: {e}"))?;
        // Tokenizers can reassign added-token IDs while loading. Use the
        // resolved table so prompts and decoded IDs share one identity.
        let tokenizer_str = tokenizer
            .to_string(false)
            .map_err(|e| format!("failed to serialize loaded tokenizer: {e}"))?;
        let tokens = WhisperSpecialTokens::from_tokenizer_json(&tokenizer_str)
            .map_err(|e| format!("tokenizer.json is not a whisper tokenizer: {e}"))?;
        tokens
            .validate_config(&config)
            .map_err(|e| format!("tokenizer/config mismatch: {e}"))?;
        if tokenizer
            .get_vocab(true)
            .values()
            .any(|&id| id as usize >= config.vocab_size)
        {
            return Err("loaded tokenizer ID exceeds config vocab_size".to_string());
        }

        let safetensors_path = model_dir.join("model.safetensors");
        let file = SafetensorsFile::open(&safetensors_path)
            .map_err(|e| format!("failed to open {}: {e}", safetensors_path.display()))?;
        let weights = WhisperWeights::load_from_safetensors(&file, &config)
            .map_err(|e| format!("failed to load whisper weights: {e}"))?;

        let metal = Self::try_metal(&weights, &config);
        Ok(Self {
            config,
            tokens,
            weights,
            tokenizer: Some(tokenizer),
            metal,
        })
    }

    /// Builds the Metal engine unless the device is unavailable or the
    /// environment forces the CPU path. Any failure falls back silently to
    /// the CPU reference: the transcription contract does not change, only
    /// the device.
    fn try_metal(
        weights: &WhisperWeights,
        config: &WhisperConfig,
    ) -> Option<Mutex<metal::WhisperMetalEngine>> {
        if std::env::var("TURBOSPARK_WHISPER_DEVICE").as_deref() == Ok("cpu") {
            return None;
        }
        match metal::WhisperMetalEngine::new(weights, config) {
            Ok(engine) => Some(Mutex::new(engine)),
            Err(_) => None,
        }
    }

    /// Whether `transcribe` executes on Metal (the engine built at open).
    pub fn using_metal(&self) -> bool {
        self.metal.is_some()
    }

    /// Constructs a runner from parts (synthetic tests); the special
    /// tokens come in explicitly because there is no tokenizer.json to
    /// read them from, and the CPU reference path is selected -- there is
    /// no checkpoint-backed engine to build.
    pub fn from_parts(
        config: WhisperConfig,
        tokens: WhisperSpecialTokens,
        weights: WhisperWeights,
    ) -> Self {
        Self {
            config,
            tokens,
            weights,
            tokenizer: None,
            metal: None,
        }
    }

    /// Encoder forward for one log-mel window. `mel` is band-major,
    /// `[n_mels * frames]`, exactly what
    /// [`audio::whisper::whisper_log_mel_window`] emits frame-major, so
    /// the runner transposes on the way in. Returns `[frames/2, d_model]`.
    pub fn encode_window(&self, mel_frame_major: &[Vec<f32>]) -> Result<Vec<f32>, String> {
        let n_mels = self.config.n_mels;
        let frames = mel_frame_major.len();
        if frames == 0 || mel_frame_major.iter().any(|frame| frame.len() != n_mels) {
            return Err(format!("mel window must carry {n_mels} bands per frame"));
        }
        // Transpose [frame][band] -> [band][frame] for the conv.
        let mut mel = vec![0.0f32; n_mels * frames];
        for (t, frame) in mel_frame_major.iter().enumerate() {
            for (b, &v) in frame.iter().enumerate() {
                mel[b * frames + t] = v;
            }
        }

        let d = self.config.d_model;
        let conv = WhisperConvWeights {
            conv1_weight: &self.weights.conv1,
            conv1_bias: &self.weights.conv1_bias,
            conv2_weight: &self.weights.conv2,
            conv2_bias: &self.weights.conv2_bias,
        };
        let conv_out = whisper_conv_frontend(&mel, n_mels, frames, &conv, d);
        let seq = conv_out.len() / d;
        if seq != self.config.max_source_positions {
            return Err(format!(
                "conv front end produced {seq} positions, config says {}",
                self.config.max_source_positions
            ));
        }

        // Transpose [d, seq] -> [seq, d] and add the positional rows.
        let mut hidden = vec![0.0f32; seq * d];
        for t in 0..seq {
            for i in 0..d {
                hidden[t * d + i] = conv_out[i * seq + t] + self.weights.enc_positions[t * d + i];
            }
        }

        for layer in &self.weights.enc_layers {
            let w = WhisperEncoderLayerWeights {
                q_weight: &layer.q,
                q_bias: &layer.q_bias,
                k_weight: &layer.k,
                v_weight: &layer.v,
                v_bias: &layer.v_bias,
                out_weight: &layer.out,
                out_bias: &layer.out_bias,
                ln1_weight: &layer.ln1_weight,
                ln1_bias: &layer.ln1_bias,
                fc1_weight: &layer.fc1,
                fc1_bias: &layer.fc1_bias,
                fc2_weight: &layer.fc2,
                fc2_bias: &layer.fc2_bias,
                ln2_weight: &layer.ln2_weight,
                ln2_bias: &layer.ln2_bias,
            };
            hidden = whisper_encoder_layer(
                &hidden,
                seq,
                &w,
                d,
                self.config.num_attention_heads,
                self.config.encoder_ffn_dim,
                self.config.layer_norm_eps,
            );
        }

        // Final encoder layer norm, per row.
        let mut out = vec![0.0f32; seq * d];
        for t in 0..seq {
            out[t * d..(t + 1) * d].copy_from_slice(&layer_norm(
                &hidden[t * d..(t + 1) * d],
                &self.weights.enc_ln_weight,
                &self.weights.enc_ln_bias,
                self.config.layer_norm_eps,
            ));
        }
        Ok(out)
    }

    /// Vocab logits for one decoder stream row: final layer norm, then a
    /// dot product with every (tied) token-embedding row. f64 accumulation
    /// keeps the argmax stable across platforms.
    pub(crate) fn logits(&self, hidden: &[f32]) -> Result<Vec<f32>, String> {
        let d = self.config.d_model;
        if hidden.len() != d {
            return Err(format!(
                "decoder stream row is {} wide, config says {d}",
                hidden.len()
            ));
        }
        let normed = layer_norm(
            hidden,
            &self.weights.dec_ln_weight,
            &self.weights.dec_ln_bias,
            self.config.layer_norm_eps,
        );
        let vocab = self.config.vocab_size;
        let mut out = Vec::with_capacity(vocab);
        for v in 0..vocab {
            let row = &self.weights.embed_tokens[v * d..(v + 1) * d];
            let acc: f64 = normed
                .iter()
                .zip(row)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            out.push(acc as f32);
        }
        Ok(out)
    }

    /// Resolves the language selection to a token id. `Some(code)` picks
    /// that language's token from the tokenizer's added tokens (`"en"`,
    /// `"de"`, ...); `None` auto-detects on the first encoded window.
    pub(crate) fn language_token(
        &self,
        language: Option<&str>,
        encoder_out: Option<&[f32]>,
        seq: usize,
    ) -> Result<u32, String> {
        match language {
            Some(code) if !code.is_empty() => {
                let needle = format!("<|{code}|>");
                let tokenizer = self
                    .tokenizer
                    .as_ref()
                    .ok_or_else(|| "language selection needs a tokenizer".to_string())?;
                for (id, token) in tokenizer.get_added_tokens_decoder() {
                    if token.content == needle && self.tokens.is_language(id) {
                        return Ok(id);
                    }
                }
                Err(format!(
                    "language {code:?} has no token in this model's vocabulary"
                ))
            }
            _ => {
                let encoder_out = encoder_out
                    .ok_or_else(|| "language auto-detection needs an encoded window".to_string())?;
                decode::detect_language(self, encoder_out, seq)
            }
        }
    }

    /// Transcribes 16 kHz mono f32 PCM. `language` is `None` or `"auto"`
    /// for auto-detection (scored once, on the first window), or a
    /// language code such as `"en"`.
    pub fn transcribe(
        &self,
        pcm: &[f32],
        language: Option<&str>,
    ) -> Result<WhisperTranscription, String> {
        if pcm.is_empty() {
            return Err("pcm is empty".to_string());
        }
        if pcm.iter().any(|sample| !sample.is_finite()) {
            return Err("pcm samples must be finite".to_string());
        }
        if !self.tokens.multilingual
            && language.is_some_and(|code| !matches!(code, "" | "auto" | "en"))
        {
            return Err("English-only Whisper models require language en or auto".to_string());
        }
        if let Some(engine) = &self.metal {
            let mut engine = engine.lock().expect("whisper Metal engine lock");
            return self.transcribe_metal(&mut engine, pcm, language);
        }
        let n_mels = self.config.n_mels;
        let mut segments: Vec<WhisperSegment> = Vec::new();
        let mut chosen_language: Option<u32> = None;
        let mut window_start = 0usize;

        while window_start < pcm.len() {
            let window_len = (pcm.len() - window_start).min(WHISPER_WINDOW_SAMPLES);
            // Zero-pad a tail window to the full 30 seconds, the way the
            // reference pads short audio.
            let mut window = vec![0.0f32; WHISPER_WINDOW_SAMPLES];
            window[..window_len].copy_from_slice(&pcm[window_start..window_start + window_len]);

            let mel = whisper_log_mel_window(&window, n_mels)
                .map_err(|e| format!("mel frontend failed: {e}"))?;
            let encoder_out = self.encode_window(&mel)?;
            let seq = encoder_out.len() / self.config.d_model;

            let language_token = match chosen_language {
                Some(token) => token,
                None => match language {
                    // English-only vocabularies configure no language
                    // token; the prompt builder drops it and the report
                    // below pins the language to "en" (0 is never read).
                    _ if !self.tokens.multilingual => 0,
                    Some(code) if !code.is_empty() && code != "auto" => {
                        self.language_token(Some(code), None, seq)?
                    }
                    _ => self.language_token(None, Some(&encoder_out), seq)?,
                },
            };
            chosen_language = Some(language_token);

            let prompt = decode::build_prompt(&self.tokens, language_token);
            let decoder = WindowDecoder::new(self, &encoder_out, seq, &prompt)?;
            let decoded = decoder.decode()?;

            // The segment covers the window's exclusive region: from this
            // window's start to the next window's start, or the full real
            // (unpadded) length for the final window.
            let start_seconds = window_start as f64 / f64::from(WHISPER_SAMPLE_RATE);
            let exclusive_end = if window_start + WHISPER_WINDOW_SAMPLES < pcm.len() {
                window_start + WINDOW_STRIDE_SAMPLES
            } else {
                window_start + window_len
            };
            let end_seconds = exclusive_end as f64 / f64::from(WHISPER_SAMPLE_RATE);

            let text = match &self.tokenizer {
                Some(tokenizer) => tokenizer
                    .decode(&decoded.text_tokens, true)
                    .map_err(|error| format!("transcript token decoding failed: {error}"))?,
                None => String::new(),
            };
            if !text.trim().is_empty() {
                segments.push(WhisperSegment {
                    index: segments.len(),
                    start_seconds,
                    end_seconds,
                    text: text.trim().to_string(),
                });
            }

            if window_start + WHISPER_WINDOW_SAMPLES >= pcm.len() {
                break;
            }
            window_start += WINDOW_STRIDE_SAMPLES;
        }

        let language = if !self.tokens.multilingual {
            "en".to_string()
        } else {
            self.language_code(chosen_language.ok_or("no window decoded")?)
        };
        Ok(WhisperTranscription { segments, language })
    }

    /// The Metal twin of `transcribe`: identical windowing (30-second
    /// windows, one-second carry, zero-padded tail) and segment assembly,
    /// with the encoder, cross caches, and decoder running on the engine.
    /// Kept as a sibling rather than parameterized so the verified CPU
    /// path's control flow stays untouched; `whisper::tests` pins the two
    /// paths to the same synthetic-fixture tokens.
    fn transcribe_metal(
        &self,
        engine: &mut metal::WhisperMetalEngine,
        pcm: &[f32],
        language: Option<&str>,
    ) -> Result<WhisperTranscription, String> {
        let n_mels = self.config.n_mels;
        let mut segments: Vec<WhisperSegment> = Vec::new();
        let mut chosen_language: Option<u32> = None;
        let mut window_start = 0usize;

        while window_start < pcm.len() {
            let window_len = (pcm.len() - window_start).min(WHISPER_WINDOW_SAMPLES);
            let mut window = vec![0.0f32; WHISPER_WINDOW_SAMPLES];
            window[..window_len].copy_from_slice(&pcm[window_start..window_start + window_len]);

            let profile = whisper_profile();
            let phase_started = std::time::Instant::now();
            let mel = whisper_log_mel_window(&window, n_mels)
                .map_err(|e| format!("mel frontend failed: {e}"))?;
            // Transpose [frame][band] -> [band][frame] for the conv, then
            // run the encoder on the engine.
            let frames = mel.len();
            let mut mel_band_major = vec![0.0f32; n_mels * frames];
            for (t, frame) in mel.iter().enumerate() {
                for (b, &v) in frame.iter().enumerate() {
                    mel_band_major[b * frames + t] = v;
                }
            }
            let mel_ms = phase_started.elapsed().as_secs_f64() * 1e3;
            let encode_started = std::time::Instant::now();
            engine.encode(&mel_band_major, frames)?;
            let encode_ms = encode_started.elapsed().as_secs_f64() * 1e3;
            let cross_started = std::time::Instant::now();
            engine.build_cross_caches()?;
            let cross_ms = cross_started.elapsed().as_secs_f64() * 1e3;

            let language_token = match chosen_language {
                Some(token) => token,
                None => match language {
                    _ if !self.tokens.multilingual => 0,
                    Some(code) if !code.is_empty() && code != "auto" => {
                        self.language_token(Some(code), None, self.config.max_source_positions)?
                    }
                    _ => metal::detect_language_metal(
                        engine,
                        &self.weights.embed_tokens,
                        &self.weights.dec_positions,
                        &self.tokens,
                    )?,
                },
            };
            chosen_language = Some(language_token);

            let prompt = decode::build_prompt(&self.tokens, language_token);
            let decode_started = std::time::Instant::now();
            let decoder = metal::MetalWindowDecoder::new(
                engine,
                &self.weights.embed_tokens,
                &self.weights.dec_positions,
                self.tokens,
                &prompt,
            )?;
            let decoded = decoder.decode()?;
            if profile {
                let decode_ms = decode_started.elapsed().as_secs_f64() * 1e3;
                eprintln!(
                    "[whisper] window at {:.2}s: mel {mel_ms:.1} ms, encode {encode_ms:.1} ms, cross {cross_ms:.1} ms, decode {decode_ms:.1} ms ({} tokens)",
                    window_start as f64 / f64::from(WHISPER_SAMPLE_RATE),
                    decoded.text_tokens.len()
                );
            }

            let start_seconds = window_start as f64 / f64::from(WHISPER_SAMPLE_RATE);
            let exclusive_end = if window_start + WHISPER_WINDOW_SAMPLES < pcm.len() {
                window_start + WINDOW_STRIDE_SAMPLES
            } else {
                window_start + window_len
            };
            let end_seconds = exclusive_end as f64 / f64::from(WHISPER_SAMPLE_RATE);

            let text = match &self.tokenizer {
                Some(tokenizer) => tokenizer
                    .decode(&decoded.text_tokens, true)
                    .map_err(|error| format!("transcript token decoding failed: {error}"))?,
                None => String::new(),
            };
            if !text.trim().is_empty() {
                segments.push(WhisperSegment {
                    index: segments.len(),
                    start_seconds,
                    end_seconds,
                    text: text.trim().to_string(),
                });
            }

            if window_start + WHISPER_WINDOW_SAMPLES >= pcm.len() {
                break;
            }
            window_start += WINDOW_STRIDE_SAMPLES;
        }

        let language = if !self.tokens.multilingual {
            "en".to_string()
        } else {
            self.language_code(chosen_language.ok_or("no window decoded")?)
        };
        if whisper_profile() {
            match gpu::dispatch_profile_report(1) {
                Some(report) => eprintln!("{report}"),
                None => eprintln!("[whisper] dispatch profile: no samples"),
            }
        }
        Ok(WhisperTranscription { segments, language })
    }

    /// The language code for a token id, read back from the tokenizer's
    /// added tokens; `"en"` is the fallback for English-only models whose
    /// `<|en|>` token may be absent.
    pub(crate) fn language_code(&self, token: u32) -> String {
        if let Some(tokenizer) = self.tokenizer.as_ref() {
            if let Some(added) = tokenizer.get_added_tokens_decoder().get(&token) {
                let inner = added
                    .content
                    .strip_prefix("<|")
                    .and_then(|s| s.strip_suffix("|>"));
                if let Some(code) = inner {
                    return code.to_string();
                }
            }
        }
        "en".to_string()
    }

    /// The speech family this runner executes; guards the open path.
    pub fn family() -> SpeechFamily {
        SpeechFamily::Whisper
    }
}
