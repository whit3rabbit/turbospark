//! Failure modes for embedding model intake and inference.
//!
//! Struct-style variants that name the offending field, tensor, or input,
//! so a reader who sees only the message can tell what to fix.

use std::fmt;

use turbospark_audio::SpeechError;
use turbospark_model_io::ModelError;

#[derive(Debug)]
pub enum EmbeddingError {
    /// A config field is missing or contradicts the checkpoint.
    BadConfig { field: String, why: String },
    /// A tensor is missing, misshaped, or undecodable.
    Tensor { name: String, why: String },
    /// The caller passed an input the model cannot consume.
    Input { why: String },
    /// The checkpoint requests a feature this port refuses (for example an
    /// unverified quantization scheme or pooling mode).
    Unsupported { why: String },
    /// Tokenizer load or encode failure.
    Tokenizer { why: String },
    /// A file could not be read.
    Io { path: String, why: String },
    /// Failure passed through from the shared audio kernels and loaders.
    Kernel(SpeechError),
    /// Wrapped model_io failure.
    ModelIo(ModelError),
}

impl fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmbeddingError::BadConfig { field, why } => {
                write!(f, "bad config field {field}: {why}")
            }
            EmbeddingError::Tensor { name, why } => write!(f, "tensor {name}: {why}"),
            EmbeddingError::Input { why } => write!(f, "invalid input: {why}"),
            EmbeddingError::Unsupported { why } => write!(f, "unsupported: {why}"),
            EmbeddingError::Tokenizer { why } => write!(f, "tokenizer: {why}"),
            EmbeddingError::Io { path, why } => write!(f, "io {path}: {why}"),
            EmbeddingError::Kernel(e) => write!(f, "{e}"),
            EmbeddingError::ModelIo(e) => write!(f, "model io: {e}"),
        }
    }
}

impl std::error::Error for EmbeddingError {}

impl From<SpeechError> for EmbeddingError {
    fn from(e: SpeechError) -> Self {
        EmbeddingError::Kernel(e)
    }
}

impl From<ModelError> for EmbeddingError {
    fn from(e: ModelError) -> Self {
        EmbeddingError::ModelIo(e)
    }
}

/// A `BadConfig` error naming `field`.
pub fn bad_config(field: &str, why: impl fmt::Display) -> EmbeddingError {
    EmbeddingError::BadConfig {
        field: field.to_owned(),
        why: why.to_string(),
    }
}

/// A `Tensor` error naming `name`.
pub fn tensor_error(name: &str, why: impl fmt::Display) -> EmbeddingError {
    EmbeddingError::Tensor {
        name: name.to_owned(),
        why: why.to_string(),
    }
}

/// An `Input` error.
pub fn input_error(why: impl fmt::Display) -> EmbeddingError {
    EmbeddingError::Input {
        why: why.to_string(),
    }
}
