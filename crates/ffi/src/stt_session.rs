//! STT (speech-to-text) sessions: a resident whisper model plus streaming
//! PCM streams, owned as raw handles the app opens and closes explicitly.
//!
//! Transport contract (mirrors the speech-to-text design): audio crosses
//! the ABI only as bounded chunks. Each append carries at most two seconds
//! of base64 f32 little-endian 16 kHz mono PCM; the stream buffers prior
//! audio internally and callers never resend. On the CPU reference path
//! the decode runs at `finish`, which returns the complete deterministic
//! segment list; append returns the buffered extent so the UI can show
//! progress, and the stable-prefix/tail-revision machinery is where the
//! Metal encoder path plugs in. Cancel discards the buffer.

use std::path::PathBuf;
use std::sync::Arc;

use runtime::{WhisperRunner, WhisperTranscription};

/// Resident whisper model.
pub struct SttModel {
    pub(crate) runner: Arc<WhisperRunner>,
}

/// One streaming transcription session over a resident model.
pub struct SttStream {
    /// Each stream owns the runner independently of the FFI model handle.
    model: Arc<WhisperRunner>,
    /// Explicit language code, or None for auto-detection.
    pub language: Option<String>,
    pub pcm: Vec<f32>,
    pub finished: bool,
}

/// Options for opening a stream, from JSON.
#[derive(Debug, Default, serde::Deserialize)]
pub struct SttStreamOptions {
    /// Language code (`"en"`, `"de"`, ...) or omitted/`"auto"` for
    /// auto-detection on the first window.
    #[serde(default)]
    pub language: Option<String>,
}

/// Maximum PCM bytes one append may carry: two seconds at 16 kHz f32.
pub const MAX_APPEND_BYTES: usize = 2 * 16_000 * 4;

impl SttModel {
    /// Opens a speech install directory.
    pub fn open(model_dir: &str) -> Result<Self, String> {
        let runner = WhisperRunner::open(PathBuf::from(model_dir).as_path())?;
        Ok(Self {
            runner: Arc::new(runner),
        })
    }

    pub(crate) fn has_streams(&self) -> bool {
        Arc::strong_count(&self.runner) > 1
    }
}

impl SttStream {
    pub fn open(model: &SttModel, options: &SttStreamOptions) -> Self {
        Self {
            model: Arc::clone(&model.runner),
            language: options
                .language
                .as_deref()
                .filter(|c| !c.is_empty() && *c != "auto")
                .map(str::to_string),
            pcm: Vec::new(),
            finished: false,
        }
    }

    /// Validates and buffers one bounded PCM chunk. Frame and byte limits
    /// are enforced before any decode work.
    pub fn append(&mut self, pcm_base64: &str) -> Result<serde_json::Value, String> {
        if self.finished {
            return Err("stream is already finished".to_string());
        }
        // Bound the encoded payload before the decoder reserves its buffer.
        // Ignore whitespace as the shared base64 decoder does, but do not
        // let whitespace padding inflate its allocation.
        let max_encoded = MAX_APPEND_BYTES.div_ceil(3) * 4;
        let encoded_len = pcm_base64
            .bytes()
            .filter(|byte| !byte.is_ascii_whitespace())
            .take(max_encoded + 1)
            .count();
        if encoded_len > max_encoded {
            return Err(format!(
                "encoded pcm chunk is too large; the cap is {MAX_APPEND_BYTES} decoded bytes"
            ));
        }
        let mut encoded = String::with_capacity(encoded_len);
        for part in pcm_base64.split_ascii_whitespace() {
            encoded.push_str(part);
        }
        let raw = turbospark_server::vision::base64_decode(&encoded)
            .map_err(|e| format!("pcm is not valid base64: {e}"))?;
        if raw.len() > MAX_APPEND_BYTES {
            return Err(format!(
                "pcm chunk is {} bytes; the cap is {MAX_APPEND_BYTES} (two seconds of 16 kHz f32)",
                raw.len()
            ));
        }
        if raw.len() % 4 != 0 {
            return Err(format!(
                "pcm chunk is {} bytes, not a whole number of f32 samples",
                raw.len()
            ));
        }
        let samples: Vec<f32> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        if samples.iter().any(|s| !s.is_finite()) {
            return Err("pcm chunk carries a non-finite sample".to_string());
        }
        self.pcm.extend_from_slice(&samples);
        Ok(serde_json::json!({
            "bufferedSamples": self.pcm.len(),
            "bufferedSeconds": self.pcm.len() as f64 / 16_000.0,
        }))
    }

    /// Transcribes everything buffered, deterministically, and marks the
    /// stream finished.
    pub fn finish(&mut self) -> Result<WhisperTranscription, String> {
        if self.finished {
            return Err("stream is already finished".to_string());
        }
        let transcription = self.model.transcribe(&self.pcm, self.language.as_deref())?;
        self.finished = true;
        self.pcm = Vec::new();
        Ok(transcription)
    }
}

/// Cancel path: drops the buffered audio and marks the stream finished.
pub fn cancel(stream: &mut SttStream) {
    stream.pcm = Vec::new();
    stream.finished = true;
}

/// Serializes a transcription to the wire JSON shape the design pins:
/// `{ "segments": [...], "language_detected": string }`.
pub fn transcription_to_json(t: &WhisperTranscription) -> serde_json::Value {
    serde_json::json!({
        "segments": t.segments.iter().map(|s| serde_json::json!({
            "index": s.index,
            "startSeconds": s.start_seconds,
            "endSeconds": s.end_seconds,
            "text": s.text,
        })).collect::<Vec<_>>(),
        "languageDetected": t.language,
    })
}

#[cfg(test)]
pub(crate) fn model_for_testing() -> SttModel {
    use model_io::whisper_config::{WhisperConfig, WhisperSpecialTokens};
    use runtime::whisper::weights::WhisperWeights;

    // Ownership and append tests never run the encoder or decoder. Empty
    // tensors keep these handle tests independent of checkpoint fixtures.
    let config = WhisperConfig::from_json_str(
        r#"{"model_type":"whisper","d_model":8,"encoder_layers":1,"decoder_layers":1,
            "encoder_attention_heads":2,"decoder_attention_heads":2,
            "encoder_ffn_dim":32,"decoder_ffn_dim":32,"vocab_size":51865,
            "num_mel_bins":80,"max_source_positions":1500,"max_target_positions":448,
            "activation_function":"gelu","scale_embedding":false}"#,
    )
    .expect("handle fixture config");
    let weights = WhisperWeights {
        conv1: Vec::new(),
        conv1_bias: Vec::new(),
        conv2: Vec::new(),
        conv2_bias: Vec::new(),
        enc_positions: Vec::new(),
        enc_layers: Vec::new(),
        enc_ln_weight: Vec::new(),
        enc_ln_bias: Vec::new(),
        embed_tokens: Vec::new(),
        dec_positions: Vec::new(),
        dec_layers: Vec::new(),
        dec_ln_weight: Vec::new(),
        dec_ln_bias: Vec::new(),
    };
    SttModel {
        runner: Arc::new(WhisperRunner::from_parts(
            config,
            WhisperSpecialTokens::MULTILINGUAL,
            weights,
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_stream() -> SttStream {
        SttStream::open(&model_for_testing(), &SttStreamOptions::default())
    }

    fn encode_bytes(bytes: &[u8]) -> String {
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let b0 = chunk[0] as u32;
            let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
            let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
            let n = (b0 << 16) | (b1 << 8) | b2;
            out.push(ALPHABET[(n >> 18) as usize & 63] as char);
            out.push(ALPHABET[(n >> 12) as usize & 63] as char);
            if chunk.len() > 1 {
                out.push(ALPHABET[(n >> 6) as usize & 63] as char);
            } else {
                out.push('=');
            }
            if chunk.len() > 2 {
                out.push(ALPHABET[n as usize & 63] as char);
            } else {
                out.push('=');
            }
        }
        out
    }

    fn encode(samples: &[f32]) -> String {
        let mut bytes = Vec::with_capacity(samples.len() * 4);
        for s in samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        encode_bytes(&bytes)
    }

    #[test]
    fn append_enforces_the_two_second_byte_cap() {
        let mut stream = test_stream();
        // Three seconds of silence: over the cap, refused before decode.
        let encoded = encode(&[0.0f32; 3 * 16_000]);
        let err = stream.append(&encoded).unwrap_err();
        assert!(err.contains("the cap is"), "{err}");
        // Exactly two seconds: accepted.
        let encoded = encode(&[0.25f32; 2 * 16_000]);
        let status = stream.append(&encoded).unwrap();
        assert_eq!(status["bufferedSamples"], 32_000);
        assert_eq!(status["bufferedSeconds"], 2.0);
        assert_eq!(stream.pcm.len(), 32_000);
    }

    #[test]
    fn append_refuses_partial_frames_and_non_finite_samples() {
        let mut stream = test_stream();
        // 7 bytes is not a whole number of f32 samples.
        let encoded = encode_bytes(&[0u8; 7]);
        let err = stream.append(&encoded).unwrap_err();
        assert!(err.contains("not a whole number of f32 samples"), "{err}");
        // A NaN sample is refused by value.
        let encoded = encode(&[f32::NAN]);
        let err = stream.append(&encoded).unwrap_err();
        assert!(err.contains("non-finite"), "{err}");
        // Finish on an empty stream still decodes (the runner refuses an
        // empty pcm with its own error), so only append is exercised here.
    }

    #[test]
    fn append_bounds_encoded_input_before_decoding_and_preserves_whitespace() {
        let mut stream = test_stream();
        let oversized_invalid = "!".repeat(MAX_APPEND_BYTES.div_ceil(3) * 4 + 1);
        let error = stream.append(&oversized_invalid).unwrap_err();
        assert!(error.contains("encoded pcm chunk is too large"), "{error}");
        assert!(stream.pcm.is_empty());

        let encoded = encode(&[0.5; 2 * 16_000]);
        let spaced = encoded
            .as_bytes()
            .chunks(64)
            .map(|part| std::str::from_utf8(part).unwrap())
            .collect::<Vec<_>>()
            .join(" \n");
        assert_eq!(stream.append(&spaced).unwrap()["bufferedSamples"], 32_000);
        cancel(&mut stream);
        assert!(stream.finished);
        assert_eq!(
            stream.pcm.capacity(),
            0,
            "cancellation releases buffered audio"
        );
    }
}
