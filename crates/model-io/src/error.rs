//! Failure modes for the validation gates in model install loading. Ported
//! from `ModelError` in `Infrastructure/ModelIO/ModelTypes.swift`.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    PartialInstall {
        path: String,
    },
    NotAGTurboDirectory,
    UnsupportedCapability {
        capability: String,
    },
    UnsupportedVersion {
        major: i64,
        minor: i64,
    },
    UnknownFlag {
        name: String,
    },
    ArchMismatch {
        field: String,
        expected: String,
        actual: String,
    },
    ExpertStrideNotPageAligned {
        stride: u64,
        page_size: u64,
    },
    MissingFile {
        name: String,
    },
    ChecksumMismatch {
        file: String,
    },
    TensorNotFound {
        name: String,
    },
    TensorSizeMismatch {
        name: String,
        expected: u64,
        actual: u64,
    },
    IndexCorrupt {
        detail: String,
    },
    IoFailed {
        call: String,
        detail: String,
    },
    TrustedReceiptInvalid {
        detail: String,
    },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::PartialInstall { path } => {
                write!(
                    f,
                    "model.gturbo directory at {path} is missing manifest.json"
                )
            }
            ModelError::NotAGTurboDirectory => {
                write!(f, "manifest.json magic does not equal \"GTURBO\"")
            }
            ModelError::UnsupportedCapability { capability } => write!(
                f,
                "model capability {capability:?} is not supported by the text-model loader"
            ),
            ModelError::UnsupportedVersion { major, minor } => write!(
                f,
                "manifest version {major}.{minor} is not supported (need 1.x)"
            ),
            ModelError::UnknownFlag { name } => {
                write!(f, "manifest.flags contains unknown key \"{name}\"")
            }
            ModelError::ArchMismatch {
                field,
                expected,
                actual,
            } => {
                write!(f, "manifest.arch.{field} = {actual}; expected {expected}")
            }
            ModelError::ExpertStrideNotPageAligned { stride, page_size } => write!(
                f,
                "expertStride {stride} is not a multiple of page size {page_size}"
            ),
            ModelError::MissingFile { name } => {
                write!(f, "model.gturbo is missing required file {name}")
            }
            ModelError::ChecksumMismatch { file } => write!(
                f,
                "SHA-256 of {file} does not match manifest.files[{file}].sha256"
            ),
            ModelError::TensorNotFound { name } => {
                write!(f, "no IndexEntry named {name} in model_weights.bin")
            }
            ModelError::TensorSizeMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "tensor {name} size {actual} does not match expected {expected}"
            ),
            ModelError::IndexCorrupt { detail } => write!(f, "resident index is corrupt: {detail}"),
            ModelError::IoFailed { call, detail } => write!(f, "{call} failed: {detail}"),
            ModelError::TrustedReceiptInvalid { detail } => {
                write!(f, "trusted install receipt invalid: {detail}")
            }
        }
    }
}

impl std::error::Error for ModelError {}
