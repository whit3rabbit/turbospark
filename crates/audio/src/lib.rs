//! Portable audio engine for TurboSpark: decode, resample, analysis, WAV
//! output, and the speech model session contract.
//!
//! The split with the macOS app is deliberate (docs/AUDIO_UI.md). Swift owns
//! what only the OS can do: microphone and app-audio capture, playback,
//! permissions, and AAC encoding through Apple's encoder. This crate owns
//! everything that is arithmetic or inference: turning any supported file
//! into PCM, resampling it to the 16 kHz mono shape speech models read,
//! waveform peaks and meter levels, trimmed WAV output, and the
//! speech-to-text / text-to-speech session types.
//!
//! Nothing here touches Metal, macOS, or a model install, so it builds and
//! tests for a non-macOS target, and the portable-subset CI check asserts
//! that. A future audio model family that runs on Metal belongs in
//! `runtime`, dispatched by `ArchConfig.family`; [`speech`] is the contract
//! it will implement, not the place its kernels go.

#![forbid(unsafe_code)]

pub mod analysis;
pub mod convert;
pub mod decode;
pub mod error;
pub mod formats;
pub mod resample;
pub mod speech;
pub mod wav;

pub use analysis::{display_level, peaks, rms, PeaksReport};
pub use convert::{convert, load_speech_samples, ConvertOptions, ConvertReport};
pub use decode::{probe, DecodedStream, ProbeReport};
pub use error::AudioError;
pub use formats::{classify_extension, FileClass};
pub use resample::Resampler;
pub use speech::{capabilities, open_model, AudioCapabilities, AudioModel, TaskStatus};
pub use wav::{WavSampleFormat, WavWriter};

/// The sample rate speech-to-text models read. Whisper-class encoders are
/// trained on 16 kHz log-mel input, and the Apple recognizer accepts it.
pub const SPEECH_SAMPLE_RATE: u32 = 16_000;
