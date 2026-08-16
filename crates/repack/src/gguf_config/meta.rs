//! Helper wrapper around GGUF header metadata lookups.

use crate::gguf_header::{GgufHeader, GgufValue};

use super::GgufConfigError;

/// Scoped metadata accessor that resolves keys under a given prefix.
pub struct Meta<'a> {
    /// Reference to the parsed GGUF header.
    pub header: &'a GgufHeader,
    /// Architecture or namespace prefix (e.g. "llama" or "qwen2").
    pub prefix: &'a str,
}

impl Meta<'_> {
    /// Constructs a full metadata key by appending `suffix` to `prefix`.
    pub fn key(&self, suffix: &str) -> String {
        format!("{}.{suffix}", self.prefix)
    }

    /// Retrieves an optional metadata value for the prefixed key.
    pub fn opt(&self, suffix: &str) -> Option<&GgufValue> {
        self.header.metadata.get(&self.key(suffix))
    }

    /// Reads a required `u64` metadata value for the prefixed key.
    pub fn u64(&self, suffix: &str) -> Result<u64, GgufConfigError> {
        let key = self.key(suffix);
        self.opt(suffix)
            .ok_or(GgufConfigError::MissingKey { key: key.clone() })?
            .as_u64()
            .ok_or(GgufConfigError::BadValue {
                key,
                detail: "not an unsigned integer".to_string(),
            })
    }

    /// Reads a required `i64` metadata value for the prefixed key, casting from `u64`.
    pub fn i64(&self, suffix: &str) -> Result<i64, GgufConfigError> {
        Ok(self.u64(suffix)? as i64)
    }

    /// Reads an optional `i64` metadata value for the prefixed key, casting from `u64`.
    pub fn opt_i64(&self, suffix: &str) -> Option<i64> {
        self.opt(suffix)
            .and_then(GgufValue::as_u64)
            .map(|v| v as i64)
    }

    /// Reads an optional `f64` metadata value for the prefixed key.
    pub fn opt_f64(&self, suffix: &str) -> Option<f64> {
        self.opt(suffix).and_then(GgufValue::as_f64)
    }
}
