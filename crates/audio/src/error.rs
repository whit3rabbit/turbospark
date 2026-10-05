//! One error type for every fallible call, mapped to ABI codes in the FFI.

use std::fmt;

/// Why an audio operation did not happen.
#[derive(Debug)]
pub enum AudioError {
    /// The file could not be read or written.
    Io(std::io::Error),
    /// The container or codec has no decoder here (for example Opus). The
    /// message names the extension so the app can say "convert it".
    Unsupported(String),
    /// The file opened but its contents did not decode.
    Decode(String),
    /// An option was outside its allowed set (zero rate, inverted range).
    InvalidOption(String),
    /// A trim range selected no audio.
    EmptyRange,
    /// No model family serves this task yet. Carries the user-facing reason.
    NeedsModel(String),
}

impl fmt::Display for AudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioError::Io(e) => write!(f, "audio I/O failed: {e}"),
            AudioError::Unsupported(what) => write!(
                f,
                "unsupported audio format ({what}); convert to M4A or WAV and try again"
            ),
            AudioError::Decode(detail) => write!(f, "audio decode failed: {detail}"),
            AudioError::InvalidOption(detail) => write!(f, "invalid audio option: {detail}"),
            AudioError::EmptyRange => write!(f, "the selected range contains no audio"),
            AudioError::NeedsModel(reason) => write!(f, "{reason}"),
        }
    }
}

impl std::error::Error for AudioError {}

impl From<std::io::Error> for AudioError {
    fn from(e: std::io::Error) -> Self {
        AudioError::Io(e)
    }
}
