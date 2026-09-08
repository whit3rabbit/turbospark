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

    /// Reads a required `i64` metadata value for the prefixed key, converting
    /// from `u64`. A value above `i64::MAX` is refused rather than wrapped
    /// negative -- a header field this port cannot represent should never
    /// reach a shape or a mask as a plausible-looking negative number.
    pub fn i64(&self, suffix: &str) -> Result<i64, GgufConfigError> {
        let key = self.key(suffix);
        i64::try_from(self.u64(suffix)?).map_err(|_| GgufConfigError::BadValue {
            key,
            detail: "exceeds i64".to_string(),
        })
    }

    /// Reads an optional `i64` metadata value for the prefixed key, converting
    /// from `u64`.
    ///
    /// Returns `Ok(None)` only when the key is genuinely absent. A key that
    /// IS present but is not representable as `i64` is an error, not a
    /// silent absence: collapsing "no such key" and "present but unreadable"
    /// into one `None` is exactly the class of bug this port has hit before
    /// (`crates/repack` CLAUDE.md's control-vector `layer_base` reader draws
    /// the same distinction, for the same reason) -- a caller that then
    /// falls back to a baseline default would read a corrupt or hostile
    /// field as if the checkpoint had simply not published it.
    pub fn opt_i64(&self, suffix: &str) -> Result<Option<i64>, GgufConfigError> {
        let Some(value) = self.opt(suffix) else {
            return Ok(None);
        };
        let raw = value.as_u64().ok_or_else(|| GgufConfigError::BadValue {
            key: self.key(suffix),
            detail: "not an unsigned integer".to_string(),
        })?;
        i64::try_from(raw)
            .map(Some)
            .map_err(|_| GgufConfigError::BadValue {
                key: self.key(suffix),
                detail: "exceeds i64".to_string(),
            })
    }

    /// Reads an optional `f64` metadata value for the prefixed key.
    pub fn opt_f64(&self, suffix: &str) -> Option<f64> {
        self.opt(suffix).and_then(GgufValue::as_f64)
    }
}
