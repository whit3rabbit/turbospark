//! Ranged reads over a remote (or, for tests, in-memory) byte source, and
//! the two-step plan (fetch the length prefix, then fetch exactly the
//! header) that lets the repacker read a safetensors header without
//! downloading the file it's attached to.

use crate::safetensors_header::{parse_header, SafetensorsHeader, SafetensorsHeaderError};

#[derive(Debug, Clone, PartialEq)]
pub enum DownloadError {
    Request(String),
    UnexpectedStatus { status: u16 },
    ShortRead { expected: u64, actual: u64 },
    Header(SafetensorsHeaderError),
}

impl std::fmt::Display for DownloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DownloadError::Request(detail) => write!(f, "request failed: {detail}"),
            DownloadError::UnexpectedStatus { status } => {
                write!(f, "unexpected HTTP status {status}")
            }
            DownloadError::ShortRead { expected, actual } => {
                write!(f, "short read: expected {expected} bytes, got {actual}")
            }
            DownloadError::Header(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for DownloadError {}

impl From<SafetensorsHeaderError> for DownloadError {
    fn from(e: SafetensorsHeaderError) -> Self {
        DownloadError::Header(e)
    }
}

/// A source of byte ranges, addressed by absolute file offset. Implemented
/// by [`HttpRangeSource`] for real installs and by an in-memory slice in
/// tests, so the planning logic in this module never needs a live network
/// connection to exercise.
pub trait RangeSource {
    fn read_range(&self, start: u64, end_exclusive: u64) -> Result<Vec<u8>, DownloadError>;
}

/// Fetches the safetensors header from `source` without downloading the
/// tensor data that follows it: first the 8-byte length prefix, then
/// exactly the header bytes it declares.
pub fn fetch_safetensors_header(
    source: &dyn RangeSource,
) -> Result<SafetensorsHeader, DownloadError> {
    let prefix = source.read_range(0, 8)?;
    if prefix.len() < 8 {
        return Err(DownloadError::ShortRead {
            expected: 8,
            actual: prefix.len() as u64,
        });
    }
    let header_len = u64::from_le_bytes(prefix[0..8].try_into().unwrap());
    let full = source.read_range(0, 8 + header_len)?;
    Ok(parse_header(
        &full,
        crate::safetensors_header::DEFAULT_MAX_HEADER_BYTES,
    )?)
}

/// HTTP-backed [`RangeSource`] using the `blocking` `reqwest` client. Issues
/// one `Range: bytes=start-(end-1)` GET per call and requires a `206
/// Partial Content` response, so a server that silently ignores range
/// requests (and would otherwise hand back the whole file) is caught rather
/// than treated as success.
pub struct HttpRangeSource {
    url: String,
    client: reqwest::blocking::Client,
}

impl HttpRangeSource {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            client: reqwest::blocking::Client::new(),
        }
    }
}

impl RangeSource for HttpRangeSource {
    fn read_range(&self, start: u64, end_exclusive: u64) -> Result<Vec<u8>, DownloadError> {
        let end_inclusive = end_exclusive.saturating_sub(1);
        let response = self
            .client
            .get(&self.url)
            .header(
                reqwest::header::RANGE,
                format!("bytes={start}-{end_inclusive}"),
            )
            .send()
            .map_err(|e| DownloadError::Request(e.to_string()))?;
        if response.status().as_u16() != 206 {
            return Err(DownloadError::UnexpectedStatus {
                status: response.status().as_u16(),
            });
        }
        let bytes = response
            .bytes()
            .map_err(|e| DownloadError::Request(e.to_string()))?;
        let expected = end_exclusive - start;
        if bytes.len() as u64 != expected {
            return Err(DownloadError::ShortRead {
                expected,
                actual: bytes.len() as u64,
            });
        }
        Ok(bytes.to_vec())
    }
}

/// In-memory [`RangeSource`] over a byte slice, for tests.
pub struct MemoryRangeSource<'a> {
    data: &'a [u8],
}

impl<'a> MemoryRangeSource<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }
}

impl RangeSource for MemoryRangeSource<'_> {
    fn read_range(&self, start: u64, end_exclusive: u64) -> Result<Vec<u8>, DownloadError> {
        let start = start as usize;
        let end = end_exclusive as usize;
        if end > self.data.len() {
            return Err(DownloadError::ShortRead {
                expected: end_exclusive - start as u64,
                actual: self.data.len().saturating_sub(start) as u64,
            });
        }
        Ok(self.data[start..end].to_vec())
    }
}
