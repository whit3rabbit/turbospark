//! Ranged reads over a remote (or, for tests, in-memory) byte source, and
//! the two-step plan (fetch the length prefix, then fetch exactly the
//! header) that lets the repacker read a safetensors header without
//! downloading the file it's attached to.

use crate::gguf_header::{
    parse_header as parse_gguf, GgufHeader, GgufHeaderError, DEFAULT_MAX_HEADER_BYTES as GGUF_CAP,
};
use crate::safetensors_header::{parse_header, SafetensorsHeader, SafetensorsHeaderError};

#[derive(Debug, Clone, PartialEq)]
pub enum DownloadError {
    Request(String),
    UnexpectedStatus {
        status: u16,
    },
    ShortRead {
        expected: u64,
        actual: u64,
    },
    Header(SafetensorsHeaderError),
    GgufHeader(GgufHeaderError),
    /// The file ended before its own header did.
    TruncatedGguf {
        have: u64,
        needed: u64,
    },
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
            DownloadError::GgufHeader(e) => write!(f, "{e}"),
            DownloadError::TruncatedGguf { have, needed } => write!(
                f,
                "file is {have} bytes but its GGUF header runs to at least {needed}"
            ),
        }
    }
}

impl From<GgufHeaderError> for DownloadError {
    fn from(e: GgufHeaderError) -> Self {
        DownloadError::GgufHeader(e)
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

/// First speculative read for [`fetch_gguf_header`]. A real header is
/// dominated by `tokenizer.ggml.tokens` and runs to a few MB, so this is
/// sized to usually take two or three requests rather than one: asking for
/// the whole cap up front would pull 64 MB off every checkpoint.
pub const GGUF_INITIAL_FETCH_BYTES: u64 = 1 << 20;

/// Fetches a GGUF header from `source` without downloading the tensor data
/// that follows it.
///
/// GGUF has no length prefix, so unlike [`fetch_safetensors_header`] this
/// cannot be a two-step plan: the header's length is only known once the
/// variable-length metadata section has been walked. The parser reports the
/// offset it wanted, but that offset advances one FIELD at a time, so
/// following it literally would be one HTTP round trip per metadata value.
/// This grows geometrically instead and uses `needed` only as a floor.
pub fn fetch_gguf_header(source: &dyn RangeSource) -> Result<GgufHeader, DownloadError> {
    let mut want = GGUF_INITIAL_FETCH_BYTES;
    loop {
        // A range past EOF is a short read, not a failure: small files are
        // legitimate, and the actual length is what bounds the retry.
        let (buf, at_eof) = match source.read_range(0, want) {
            Ok(b) => (b, false),
            Err(DownloadError::ShortRead { actual, .. }) if actual > 0 => {
                (source.read_range(0, actual)?, true)
            }
            Err(e) => return Err(e),
        };
        let have = buf.len() as u64;
        match parse_gguf(&buf, GGUF_CAP) {
            Ok(header) => return Ok(header),
            Err(GgufHeaderError::TooShort { needed }) => {
                if at_eof {
                    return Err(DownloadError::TruncatedGguf { have, needed });
                }
                want = needed.max(want.saturating_mul(2)).min(GGUF_CAP);
                if have >= want {
                    return Err(DownloadError::TruncatedGguf { have, needed });
                }
            }
            Err(e) => return Err(e.into()),
        }
    }
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
