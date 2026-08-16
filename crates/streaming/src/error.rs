//! Failure modes for the expert streamer. Ported from
//! `Infrastructure/Streaming/ExpertStreamer.swift` (`StreamerError`).

use std::fmt;

// `AllocationFailed` is a slot's page-aligned backing allocation failing at
// open. It used to be raised as `PreadFailed`, which the two share nothing
// but the moment to justify: "pread failed: posix_memalign failed with 12"
// sends the reader looking at the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamerError {
    OpenFailed { path: String, detail: String },
    SizeMismatch { expected: u64, actual: u64 },
    OffsetOutOfRange { offset: u64 },
    PreadFailed { detail: String },
    SlotOutOfRange { slot: usize },
    AllocationFailed { detail: String },
}

impl fmt::Display for StreamerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamerError::OpenFailed { path, detail } => {
                write!(f, "open({path}) failed: {detail}")
            }
            StreamerError::SizeMismatch { expected, actual } => {
                write!(f, "file size mismatch: expected {expected}, got {actual}")
            }
            StreamerError::OffsetOutOfRange { offset } => {
                write!(f, "offset {offset} is outside the streamed range")
            }
            StreamerError::PreadFailed { detail } => write!(f, "pread failed: {detail}"),
            StreamerError::SlotOutOfRange { slot } => {
                write!(f, "expert cache slot {slot} is out of range")
            }
            StreamerError::AllocationFailed { detail } => {
                write!(f, "expert slot allocation failed: {detail}")
            }
        }
    }
}

impl std::error::Error for StreamerError {}
