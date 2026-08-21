//! Parses the GGUF v3 file header: a fixed 24-byte preamble, a metadata
//! key/value section, and a tensor table describing each tensor's element
//! type, dimensions, and offset within the data region that follows. See
//! <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md> for the spec.

/// GGML tensor types, quantization formats, and block size helpers.
pub mod ggml;
/// Binary parser for GGUF headers and key-value metadata.
pub mod parser;
/// Data types and structures representing parsed GGUF metadata and tensors.
pub mod types;

pub use ggml::{ggml_type_block, ggml_type_name};
pub use parser::parse_header;
pub use types::{
    GgufHeader, GgufHeaderError, GgufTensorInfo, GgufValue, DEFAULT_ALIGNMENT,
    DEFAULT_MAX_HEADER_BYTES, SUPPORTED_VERSION,
};
