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

/// The largest single `Range` GET this client will issue. A GGUF's resident
/// core contains whole tensors far larger than this (Gemma 4's Q8_0
/// embedding table is 785 MB in ONE tensor), and a single response body that
/// long is where a CDN drops the connection: the first attempt at the real
/// 26.9 GB Q8_0 checkpoint died with "error decoding response body" ~2.5 GB
/// in. Splitting bounds what a retry has to re-fetch as well as making the
/// drop less likely.
const MAX_RANGE_BYTES: u64 = 64 * 1024 * 1024;

/// Attempts per chunk before giving up. Transport failures on a multi-GB
/// walk are expected rather than exceptional; a format error is not retried
/// because it will not change.
///
/// Raised from 4 to 8 in ROADMAP Phase M2. It did NOT fix the failure it was
/// raised for (a 26 GB Mixtral walk that died three times at the same layer,
/// ~19 GB in), and it is kept only because a longer walk deserves a longer
/// budget: the cost of being wrong is asymmetric, since another four attempts
/// cost seconds of backoff while giving up costs the whole walk, which has no
/// resume. Two hypotheses about that failure were tested and refuted -- bad
/// offsets (the ranges end exactly at EOF and `curl` fetched every failing
/// 64 MiB chunk at HTTP 206) and connection reuse (disabling pooling changed
/// nothing) -- so do not read this constant as the fix.
const RANGE_ATTEMPTS: usize = 8;

/// Splits `[start, end_exclusive)` into successive chunks of at most `cap`
/// bytes. Pure, so the boundary arithmetic is testable without a network.
fn chunk_ranges(start: u64, end_exclusive: u64, cap: u64) -> Vec<(u64, u64)> {
    assert!(cap > 0, "chunk cap must be positive");
    let mut out = Vec::new();
    let mut at = start;
    while at < end_exclusive {
        let next = (at + cap).min(end_exclusive);
        out.push((at, next));
        at = next;
    }
    out
}

/// HTTP-backed [`RangeSource`] using the `blocking` `reqwest` client. Issues
/// `Range: bytes=start-(end-1)` GETs and requires a `206 Partial Content`
/// response, so a server that silently ignores range requests (and would
/// otherwise hand back the whole file) is caught rather than treated as
/// success.
///
/// A call is split into [`MAX_RANGE_BYTES`] chunks, each retried up to
/// [`RANGE_ATTEMPTS`] times. Callers see one contiguous `Vec` either way.
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

    /// One `Range` GET, no retry. Length is checked here so a truncated body
    /// is a retryable error rather than silent corruption.
    fn read_chunk(&self, start: u64, end_exclusive: u64) -> Result<Vec<u8>, DownloadError> {
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

impl RangeSource for HttpRangeSource {
    fn read_range(&self, start: u64, end_exclusive: u64) -> Result<Vec<u8>, DownloadError> {
        let mut out = Vec::with_capacity((end_exclusive - start) as usize);
        for (chunk_start, chunk_end) in chunk_ranges(start, end_exclusive, MAX_RANGE_BYTES) {
            let mut last = None;
            for attempt in 0..RANGE_ATTEMPTS {
                match self.read_chunk(chunk_start, chunk_end) {
                    Ok(bytes) => {
                        out.extend_from_slice(&bytes);
                        last = None;
                        break;
                    }
                    // A range the server will not serve at all is not going
                    // to start working; everything else is transport.
                    Err(e @ DownloadError::UnexpectedStatus { .. }) => return Err(e),
                    Err(e) => {
                        std::thread::sleep(std::time::Duration::from_millis(250 << attempt.min(4)));
                        last = Some(e);
                    }
                }
            }
            if let Some(e) = last {
                return Err(e);
            }
        }
        Ok(out)
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

#[cfg(test)]
mod chunk_tests {
    use super::chunk_ranges;

    #[test]
    fn chunks_cover_the_range_exactly_and_in_order() {
        // The property that matters: concatenating the chunks reproduces the
        // original range with no gap, no overlap, and none over the cap.
        for (start, end, cap) in [(0, 0, 8), (0, 1, 8), (7, 8, 8), (0, 24, 8), (5, 23, 7)] {
            let chunks = chunk_ranges(start, end, cap);
            assert_eq!(
                chunks.iter().map(|(a, b)| b - a).sum::<u64>(),
                end - start,
                "total length for {start}..{end} cap {cap}"
            );
            let mut at = start;
            for (a, b) in &chunks {
                assert_eq!(*a, at, "gap or overlap in {start}..{end} cap {cap}");
                assert!(b - a <= cap && b > a);
                at = *b;
            }
            assert_eq!(at, end);
        }
        assert!(
            chunk_ranges(4, 4, 8).is_empty(),
            "empty range yields no GET"
        );
    }
}
