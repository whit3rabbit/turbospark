//! The audio model session behind `TsAudioSession`.
//!
//! Separate from the text and image sessions for the reason the image one
//! is: a different backend, memory envelope and output shape. Every open is
//! refused today (`turbospark_audio::speech::open_model`), so no session is
//! ever constructed outside this crate's own tests; the type exists so the
//! ABI, the Swift wrapper and the app's UI states are in place when a model
//! family lands, and a host written now does not change shape then.

use std::path::Path;
use std::sync::Mutex;

use serde::Serialize;
use turbospark_audio::speech::{SynthesizeOptions, TranscribeOptions, Transcript};
use turbospark_audio::wav::{write_mono, WavSampleFormat};
use turbospark_audio::{load_speech_samples, open_model, AudioError, AudioModel};

use crate::heavy::HeavyWorkGuard;

/// Opaque handle. One model, serialized by the mutex.
#[derive(Debug)]
pub struct AudioSession {
    model: Mutex<AudioModel>,
}

/// What `ts_audio_synthesize` wrote.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SynthesisReport {
    pub sample_rate: u32,
    pub frames: u64,
    pub duration_seconds: f64,
}

impl AudioSession {
    pub fn open(model_dir: &str) -> Result<Self, AudioError> {
        let model = open_model(Path::new(model_dir))?;
        Ok(Self::with_model(model))
    }

    /// Wraps an already-built model. For tests, and for the day a family's
    /// constructor lives somewhere `open_model` dispatches to.
    pub fn with_model(model: AudioModel) -> Self {
        Self {
            model: Mutex::new(model),
        }
    }

    pub fn transcribe(
        &self,
        audio_path: &Path,
        options: &TranscribeOptions,
    ) -> Result<Transcript, AudioError> {
        let samples = load_speech_samples(audio_path)?;
        let _gate = HeavyWorkGuard::acquire();
        let mut model = self
            .model
            .lock()
            .map_err(|_| AudioError::Decode("audio session is poisoned".to_string()))?;
        match &mut *model {
            AudioModel::SpeechToText(stt) => stt.transcribe(&samples, options),
            AudioModel::TextToSpeech(_) => Err(AudioError::InvalidOption(
                "this audio session is a text-to-speech model".to_string(),
            )),
        }
    }

    pub fn synthesize(
        &self,
        text: &str,
        options: &SynthesizeOptions,
        destination: &Path,
    ) -> Result<SynthesisReport, AudioError> {
        let _gate = HeavyWorkGuard::acquire();
        let audio = {
            let mut model = self
                .model
                .lock()
                .map_err(|_| AudioError::Decode("audio session is poisoned".to_string()))?;
            match &mut *model {
                AudioModel::TextToSpeech(tts) => tts.synthesize(text, options)?,
                AudioModel::SpeechToText(_) => {
                    return Err(AudioError::InvalidOption(
                        "this audio session is a speech-to-text model".to_string(),
                    ))
                }
            }
        };
        write_mono(
            destination,
            &audio.samples,
            audio.sample_rate,
            WavSampleFormat::Float32,
        )?;
        Ok(SynthesisReport {
            sample_rate: audio.sample_rate,
            frames: audio.samples.len() as u64,
            duration_seconds: audio.samples.len() as f64 / f64::from(audio.sample_rate.max(1)),
        })
    }
}
