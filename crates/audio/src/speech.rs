//! The speech model contract: what a speech-to-text, text-to-speech or
//! music model family must provide, and the refusal every task returns until
//! one exists.
//!
//! **NO AUDIO MODEL FAMILY IS IMPLEMENTED.** [`open_model`] refuses every
//! path with [`AudioError::NeedsModel`], and [`capabilities`] reports every
//! task inactive with the same reason, so the app can show a disabled
//! control with a sentence rather than a control that fails. The app's
//! interim fallback (Apple Speech, `AVSpeechSynthesizer`) lives in Swift and
//! is labelled as such; it is not routed through here.
//!
//! When a family lands it must be selected by `ArchConfig.family`, never by
//! tensor names (root AGENTS.md safety invariants), its Metal forward belongs
//! in `runtime`, and its install path is the reserved
//! `models/audio/<alias>.gturbo` (`catalog::store::audio_install_path`).

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::AudioError;
use crate::formats::{REFUSED_EXTENSIONS, SUPPORTED_EXTENSIONS};

/// The reason every model task is refused today. One string so the app,
/// the FFI and the tests cannot word it three ways.
pub const NO_AUDIO_MODEL_REASON: &str =
    "no speech-to-text, text-to-speech or music model family is supported yet; \
     audio installs under models/audio are reserved";

/// Whether one task can run, and if not, why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskStatus {
    pub active: bool,
    pub reason: Option<String>,
}

impl TaskStatus {
    fn refused(reason: &str) -> Self {
        Self {
            active: false,
            reason: Some(reason.to_string()),
        }
    }
}

/// What this build of the engine can do with audio.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioCapabilities {
    /// File extensions decode, peaks and convert accept.
    pub decode_extensions: Vec<String>,
    /// Audio extensions refused with a "convert it" message.
    pub refused_extensions: Vec<String>,
    pub speech_to_text: TaskStatus,
    pub text_to_speech: TaskStatus,
    pub music: TaskStatus,
}

/// The engine's audio capability table.
pub fn capabilities() -> AudioCapabilities {
    AudioCapabilities {
        decode_extensions: SUPPORTED_EXTENSIONS.iter().map(|s| s.to_string()).collect(),
        refused_extensions: REFUSED_EXTENSIONS.iter().map(|s| s.to_string()).collect(),
        speech_to_text: TaskStatus::refused(NO_AUDIO_MODEL_REASON),
        text_to_speech: TaskStatus::refused(NO_AUDIO_MODEL_REASON),
        music: TaskStatus::refused(NO_AUDIO_MODEL_REASON),
    }
}

/// Options for one transcription.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct TranscribeOptions {
    /// BCP-47 hint such as `en`; `None` lets the model detect it.
    pub language: Option<String>,
    /// Emit per-segment timestamps.
    pub timestamps: bool,
}

/// One timed span of a transcript.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptSegment {
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub text: String,
}

/// A finished transcription.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Transcript {
    pub text: String,
    pub language: Option<String>,
    pub segments: Vec<TranscriptSegment>,
}

/// Options for one synthesis.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct SynthesizeOptions {
    /// Model-specific voice or speaker id.
    pub voice: Option<String>,
    /// 1.0 is the model's natural rate.
    pub rate: Option<f32>,
}

/// Synthesized mono audio.
#[derive(Debug, Clone, PartialEq)]
pub struct SynthesizedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// A speech-to-text model. Input is 16 kHz mono (`ConvertOptions::speech`),
/// so a model never resamples and every caller normalizes the same way.
pub trait SpeechToText: Send {
    fn transcribe(
        &mut self,
        samples_16k_mono: &[f32],
        options: &TranscribeOptions,
    ) -> Result<Transcript, AudioError>;
}

/// A text-to-speech model.
pub trait TextToSpeech: Send {
    fn synthesize(
        &mut self,
        text: &str,
        options: &SynthesizeOptions,
    ) -> Result<SynthesizedAudio, AudioError>;
}

/// An opened audio model, by task.
pub enum AudioModel {
    SpeechToText(Box<dyn SpeechToText>),
    TextToSpeech(Box<dyn TextToSpeech>),
}

impl std::fmt::Debug for AudioModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AudioModel::SpeechToText(_) => f.write_str("AudioModel::SpeechToText"),
            AudioModel::TextToSpeech(_) => f.write_str("AudioModel::TextToSpeech"),
        }
    }
}

/// Opens an audio model install. Refuses every path today; see the module
/// docs for where a real family plugs in.
pub fn open_model(path: &Path) -> Result<AudioModel, AudioError> {
    let _ = path;
    Err(AudioError::NeedsModel(NO_AUDIO_MODEL_REASON.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_model_task_is_refused_with_the_shared_reason() {
        let caps = capabilities();
        for status in [&caps.speech_to_text, &caps.text_to_speech, &caps.music] {
            assert!(!status.active);
            assert_eq!(status.reason.as_deref(), Some(NO_AUDIO_MODEL_REASON));
        }
        match open_model(Path::new("/nonexistent")) {
            Err(AudioError::NeedsModel(reason)) => assert_eq!(reason, NO_AUDIO_MODEL_REASON),
            other => panic!("expected NeedsModel, got {other:?}"),
        }
    }

    #[test]
    fn capabilities_serialize_in_camel_case() {
        let json = serde_json::to_value(capabilities()).unwrap();
        assert!(json["decodeExtensions"]
            .as_array()
            .unwrap()
            .contains(&"wav".into()));
        assert_eq!(json["speechToText"]["active"], false);
    }
}
