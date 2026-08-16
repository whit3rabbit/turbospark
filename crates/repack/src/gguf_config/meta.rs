//! Helper wrapper around GGUF header metadata lookups.

use crate::gguf_header::{GgufHeader, GgufValue};

use super::GgufConfigError;

pub struct Meta<'a> {
    pub header: &'a GgufHeader,
    pub prefix: &'a str,
}

impl Meta<'_> {
    pub fn key(&self, suffix: &str) -> String {
        format!("{}.{suffix}", self.prefix)
    }

    pub fn opt(&self, suffix: &str) -> Option<&GgufValue> {
        self.header.metadata.get(&self.key(suffix))
    }

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

    pub fn i64(&self, suffix: &str) -> Result<i64, GgufConfigError> {
        Ok(self.u64(suffix)? as i64)
    }

    pub fn opt_i64(&self, suffix: &str) -> Option<i64> {
        self.opt(suffix)
            .and_then(GgufValue::as_u64)
            .map(|v| v as i64)
    }

    pub fn opt_f64(&self, suffix: &str) -> Option<f64> {
        self.opt(suffix).and_then(GgufValue::as_f64)
    }
}
