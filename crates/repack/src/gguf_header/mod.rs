//! Parses the GGUF v3 file header: a fixed 24-byte preamble, a metadata
//! key/value section, and a tensor table describing each tensor's element
//! type, dimensions, and offset within the data region that follows. See
//! <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md> for the spec.

pub mod ggml;
pub mod parser;
pub mod types;

pub use ggml::{ggml_type_block, ggml_type_name};
pub use parser::parse_header;
pub use types::{
    GgufHeader, GgufHeaderError, GgufTensorInfo, GgufValue, DEFAULT_ALIGNMENT,
    DEFAULT_MAX_HEADER_BYTES, SUPPORTED_VERSION,
};
