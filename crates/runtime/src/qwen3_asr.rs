//! The Qwen3-ASR speech runner: open, greedy transcription, and the
//! clip-level segment the session API consumes.
//!
//! Family-gated by
//! [`SpeechFamily::Qwen3Asr`](model_io::speech_family::SpeechFamily): the
//! runtime never dispatches on tensor names. This module is the portable
//! CPU path (the CPU audio tower plus the f32-dequantized text decoder);
//! the resident packed 8-bit Metal decoder stays opt-in in
//! [`crate::qwen3_asr_metal`] and nothing here selects it.

use std::path::Path;

use audio::stt::qwen3_asr::Qwen3Asr;
use model_io::speech_family::SpeechFamily;

use crate::whisper::{WhisperSegment, WhisperTranscription};

/// Resident Qwen3-ASR model on the portable CPU path.
pub struct Qwen3AsrRunner {
    model: Qwen3Asr,
}

impl Qwen3AsrRunner {
    /// The speech family this runner executes; guards the open path.
    pub fn family() -> SpeechFamily {
        SpeechFamily::Qwen3Asr
    }

    /// Opens one installed speech directory. The audio crate's loader
    /// re-validates the config, the tokenizer assets, and every tensor
    /// shape exactly as the catalog probe did at install time.
    pub fn open(model_dir: &Path) -> Result<Self, String> {
        let model = Qwen3Asr::load(model_dir).map_err(|error| error.to_string())?;
        Ok(Self { model })
    }

    /// Preserve language and token confidence for the additive audio API.
    pub fn transcribe_with_details(
        &self,
        samples: &[f32],
        language: Option<&str>,
    ) -> Result<audio::stt::qwen3_asr::Qwen3AsrTranscription, String> {
        self.model
            .transcribe_with_details(samples, language, 512)
            .map_err(|e| e.to_string())
    }

    /// Transcribes mono 16 kHz PCM. Qwen3-ASR has no alignment output, so
    /// the session contract returns one clip-level segment covering the
    /// whole buffer, with no word-level timing claim. `language` is
    /// `None`/`"auto"` for the model's default detection or a supported
    /// language name such as `"English"`; the transcription preserves the
    /// language reported by the decoder.
    pub fn transcribe(
        &self,
        samples: &[f32],
        language: Option<&str>,
    ) -> Result<WhisperTranscription, String> {
        let requested = language.filter(|code| !code.is_empty() && *code != "auto");
        let details = self
            .model
            .transcribe_with_details(samples, requested, 512)
            .map_err(|error| error.to_string())?;
        Ok(WhisperTranscription {
            segments: vec![WhisperSegment {
                index: 0,
                start_seconds: 0.0,
                end_seconds: samples.len() as f64 / 16_000.0,
                text: details.text,
            }],
            language: details.language,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Qwen3AsrRunner;
    use model_io::speech_family::SpeechFamily;

    #[test]
    fn family_wire_string_is_stable() {
        assert_eq!(Qwen3AsrRunner::family(), SpeechFamily::Qwen3Asr);
        assert_eq!(Qwen3AsrRunner::family().as_str(), "qwen3_asr");
    }

    /// The runner seam the FFI dispatches through: open the pinned
    /// checkpoint and transcribe to the single clip-level segment the
    /// session contract promises.
    #[test]
    #[ignore = "requires the pinned Qwen3-ASR checkpoint and the reference WAV"]
    fn pinned_checkpoint_transcribes_through_the_runner() {
        let model_dir = std::env::var_os("TURBOSPARK_QWEN3_ASR_DIR")
            .expect("set TURBOSPARK_QWEN3_ASR_DIR to the pinned checkpoint directory");
        let wav = std::env::var_os("TURBOSPARK_QWEN3_ASR_WAV")
            .expect("set TURBOSPARK_QWEN3_ASR_WAV to the reference speech clip");
        let audio = audio::read_wav_f32(std::path::Path::new(&wav)).unwrap();
        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.channels, 1);

        let runner = Qwen3AsrRunner::open(std::path::Path::new(&model_dir)).unwrap();
        let transcription = runner.transcribe(&audio.samples, None).unwrap();
        eprintln!(
            "qwen3_asr_runner text={:?} language={:?}",
            transcription.segments[0].text, transcription.language
        );
        assert_eq!(transcription.segments.len(), 1);
        assert_eq!(transcription.segments[0].index, 0);
        assert_eq!(transcription.segments[0].start_seconds, 0.0);
        assert_eq!(
            transcription.segments[0].end_seconds,
            audio.samples.len() as f64 / 16_000.0
        );
        assert_eq!(
            transcription.segments[0].text,
            "The quick brown fox jumps over the lazy dog."
        );
    }
}
