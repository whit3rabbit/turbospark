//! Failure modes for audio intake and DSP.
//!
//! Hand-rolled in the shape `vision_io::VisionIoError` uses: struct-style
//! variants carrying the diagnostic values by name, and a manual `Display`
//! that puts those values in the message. The rule the variants are written
//! to is that a reader who sees only the message can tell which input caused
//! it -- a bare "invalid audio" sends someone back to the file.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum AudioError {
    /// The byte stream is not a RIFF/WAVE file. `detail` names what was
    /// found where the RIFF or WAVE identifier was expected.
    NotWav { detail: String },
    /// The WAVE container is structurally broken: a truncated header, a
    /// chunk length that runs past the file, or a data chunk with a size
    /// that is not a whole number of frames. `detail` names the break.
    MalformedWav { detail: String },
    /// The file is a WAVE container but carries a format this crate does not
    /// decode. `format_tag` is the raw `wFormatTag` (1 = PCM, 3 = IEEE float,
    /// 0xFFFE = extensible).
    UnsupportedWavFormat { format_tag: u16 },
    /// The format is decodable in principle but the bits-per-sample value is
    /// not one this crate implements (8, 16, 24, or 32 for PCM; 32 for
    /// float).
    UnsupportedBitsPerSample { format_tag: u16, bits: u16 },
    /// A channel count or sample rate that cannot describe real audio
    /// (zero channels, zero or absurd rate). `field` names which one.
    InvalidWavLayout { field: String, value: u32 },
    /// An operation received a shape disagreement -- interleaved length not
    /// a multiple of channels, planar buffers of unequal frame counts, a
    /// spectrum whose length does not match the FFT size. `expected` and
    /// `actual` are the two values that disagreed.
    ShapeMismatch {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A numeric parameter is outside the range the algorithm is defined
    /// for -- a zero sample rate, a rolloff outside (0, 1], a non-positive
    /// filter width. `why` carries the required range.
    InvalidParameter {
        name: String,
        value: String,
        why: String,
    },
    /// An FFT size that is not a power of two. The radix-2 plan refuses it
    /// rather than silently padding.
    NonPowerOfTwoSize { size: usize },
    /// A requested buffer is too large to allocate as one allocation, so
    /// the operation would fail or fragment the heap. Caught before the
    /// allocation so the message names the request.
    BufferTooLarge { what: &'static str, samples: usize },
}

impl fmt::Display for AudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioError::NotWav { detail } => write!(f, "not a RIFF/WAVE stream: {detail}"),
            AudioError::MalformedWav { detail } => {
                write!(f, "malformed WAVE container: {detail}")
            }
            AudioError::UnsupportedWavFormat { format_tag } => write!(
                f,
                "unsupported WAVE format tag {format_tag:#06x}; this crate decodes PCM (0x0001) and IEEE float (0x0003)"
            ),
            AudioError::UnsupportedBitsPerSample { format_tag, bits } => write!(
                f,
                "unsupported WAVE sample width {bits} bits for format tag {format_tag:#06x}; decodable widths are 8, 16, 24, and 32"
            ),
            AudioError::InvalidWavLayout { field, value } => {
                write!(f, "unusable WAVE layout, {field} is {value}")
            }
            AudioError::ShapeMismatch {
                what,
                expected,
                actual,
            } => write!(
                f,
                "shape mismatch for {what}: expected {expected}, got {actual}"
            ),
            AudioError::InvalidParameter { name, value, why } => {
                write!(f, "invalid {name} = {value}: {why}")
            }
            AudioError::NonPowerOfTwoSize { size } => {
                write!(f, "FFT size {size} is not a power of two")
            }
            AudioError::BufferTooLarge { what, samples } => {
                write!(f, "{what} would allocate {samples} samples in one buffer")
            }
        }
    }
}

impl std::error::Error for AudioError {}

/// Validates a sample rate the way every entry point in this crate needs:
/// positive and within a range where f32 time math stays meaningful.
pub(crate) fn check_sample_rate(rate: u32) -> Result<(), AudioError> {
    if rate == 0 || rate > 768_000 {
        return Err(AudioError::InvalidWavLayout {
            field: "sample rate".to_string(),
            value: rate,
        });
    }
    Ok(())
}

/// Validates a positive channel count.
pub(crate) fn check_channel_count(channels: usize) -> Result<(), AudioError> {
    if channels == 0 || channels > 256 {
        return Err(AudioError::InvalidWavLayout {
            field: "channel count".to_string(),
            value: channels as u32,
        });
    }
    Ok(())
}

/// Errors raised by the audio and speech models.
#[derive(Debug)]
pub enum SpeechError {
    /// A config field is missing or contradicts the checkpoint.
    BadConfig { field: String, why: String },
    /// A tensor is missing, misshaped, or undecodable.
    Tensor { name: String, why: String },
    /// The caller passed an input the model cannot consume.
    Input { why: String },
    /// Frontend failure passed through from the DSP layer.
    Audio(String),
    /// The checkpoint requests a feature this port refuses (for example an
    /// unverified quantization scheme).
    Unsupported { why: String },
    /// Wrapped model_io failure.
    ModelIo(turbospark_model_io::ModelError),
}

impl std::fmt::Display for SpeechError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpeechError::BadConfig { field, why } => {
                write!(f, "bad config field {field}: {why}")
            }
            SpeechError::Tensor { name, why } => write!(f, "tensor {name}: {why}"),
            SpeechError::Input { why } => write!(f, "invalid input: {why}"),
            SpeechError::Audio(why) => write!(f, "audio frontend: {why}"),
            SpeechError::Unsupported { why } => write!(f, "unsupported: {why}"),
            SpeechError::ModelIo(e) => write!(f, "model io: {e}"),
        }
    }
}

impl std::error::Error for SpeechError {}

impl From<turbospark_model_io::ModelError> for SpeechError {
    fn from(e: turbospark_model_io::ModelError) -> Self {
        SpeechError::ModelIo(e)
    }
}

impl From<AudioError> for SpeechError {
    fn from(e: AudioError) -> Self {
        SpeechError::Audio(e.to_string())
    }
}

/// Alias for audio model errors.
pub type AudioModelError = SpeechError;
