//! Unified portable audio crate: DSP primitives (waveform, WAV I/O, resampling,
//! FFT, STFT, mel spectrograms), neural audio ops, and models organized by
//! role (music, STT, TTS, VAD, STS, codec, LID).
//!
//! # Module map
//!
//! ## Signal processing primitives
//! - [`waveform`]: the [`Waveform`] value every DSP module consumes and
//!   produces: interleaved f32 plus sample rate and channel count.
//! - [`wav`]: strict RIFF/WAVE reader and writer (u8, i16, i24, i32, f32).
//! - [`conversion`]: channel layout moves -- mixdown, duplication, planar
//!   interchange, channel extraction -- and the mono-resample pipelines.
//! - [`resample`]: mono resampling, linear and torchaudio-compatible sinc-Hann
//!   polyphase.
//! - [`fft`]: radix-2 FFT with a real-FFT wrapper.
//! - [`stft`]: STFT and inverse with window-sum-of-squares normalization.
//! - [`mel`]: mel filterbank construction and log-mel frontends.
//! - [`nemo_mel`]: NeMo-compatible pre-emphasis and centered Slaney power-mel.
//! - [`whisper`]: OpenAI Whisper log-mel filterbank and windowing frontend.
//! - [`dsp`]: windows, peak/rms, gain, normalize.
//!
//! ## Shared neural tensor ops and quantization
//! - [`ops`]: portable f32 tensor kernels (linear, norms, rope, attention pieces, convolutions).
//! - [`quant`]: MLX groupwise affine dequantization (2/3/4/6/8-bit) for U32-packed safetensors.
//!
//! ## Audio models organized by role
//! - [`music`]: Music generation (e.g. MiniMax Music 0.5 flow-matching DiT).
//! - [`stt`]: Speech-to-text (Whisper, Moonshine, Parakeet, MMS, Granite Speech, Qwen3 ASR, Nemotron ASR).
//! - [`tts`]: Text-to-speech (Kokoro).
//! - [`vad`]: Voice activity detection & diarization (Silero VAD, Sortformer, Nemotron Diarization).
//! - [`sts`]: Speech-to-speech & audio enhancement (DeepFilterNet).
//! - [`codec`]: Audio codecs, vocoders, and neural tokenizers.
//! - [`lid`]: Spoken language identification.
//! - [`models`]: Re-exports preserving task-qualified and flat module paths.

#![forbid(unsafe_code)]

extern crate self as turbospark_audio;

pub mod conversion;
pub mod dsp;
pub mod error;
pub mod fft;
pub mod mel;
pub mod nemo_mel;
pub mod resample;
pub mod stft;
pub mod wav;
pub mod waveform;
pub mod whisper;

// Shared neural ops & quantization
pub mod ops;
pub mod quant;

// Audio models by role
pub mod codec;
pub mod lid;
pub mod music;
pub mod sts;
pub mod stt;
pub mod tts;
pub mod vad;

// Grouped models module for backward compatibility
pub mod models;

pub use conversion::{
    deinterleave_to_planar, duplicate_mono_to_interleaved, extract_interleaved_channel,
    interleave_planar, mixdown_interleaved_to_mono_average, to_mono_resampled, MonoMixAccumulation,
    MonoResampleStrategy,
};
pub use dsp::{hann_window, normalize_peak, peak, rms, scale_by_gain};
pub use error::{AudioError, AudioModelError, SpeechError};
pub use fft::{real_fft_forward, real_fft_inverse, ComplexF32, RealFftPlan};
pub use mel::{
    hz_to_mel, log_mel_spectrogram, mel_filterbank, mel_spectrogram, mel_to_hz, MelFilterbank,
    MelScale, MelSpectrogramOptions,
};
pub use nemo_mel::{
    nemo_log_mel_spectrogram, nemo_log_mel_spectrogram_with_padding, NemoMelNormalization,
    NemoMelOptions,
};
pub use resample::{
    resample_mono_linear, resample_mono_sinc_hann, Accumulation, KernelComputation, SincHannOptions,
};
pub use stft::{istft, stft, stft_with_modes, StftOptions, StftPaddingMode, StftWindowPlacement};
pub use wav::{read_wav_f32, read_wav_f32_bytes, write_wav_f32, write_wav_i16};
pub use waveform::Waveform;

/// Convenience alias for model results.
pub type Result<T> = std::result::Result<T, SpeechError>;
/// Convenience alias for DSP results.
pub type AudioResult<T> = std::result::Result<T, AudioError>;
