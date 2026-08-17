//! HTTP range download implementation using reqwest.

use std::sync::Arc;

use super::chunks::{
    chunk_ranges, fill_chunks, MAX_RANGE_BYTES, RANGE_ATTEMPTS, RANGE_CONCURRENCY,
};
use super::{DownloadError, RangeSource};

/// Callback invoked on each downloaded chunk with the chunk's byte count.
pub type ByteProgressCallback = Arc<dyn Fn(u64) + Send + Sync>;

/// HTTP-backed [`RangeSource`] using the `blocking` `reqwest` client. Issues
/// `Range: bytes=start-(end-1)` GETs and requires a `206 Partial Content`
/// response, so a server that silently ignores range requests (and would
/// otherwise hand back the whole file) is caught rather than treated as
/// success.
///
/// A call is split into [`MAX_RANGE_BYTES`] chunks, up to
/// [`RANGE_CONCURRENCY`] of them in flight at once, each retried up to
/// [`RANGE_ATTEMPTS`] times. Callers see one contiguous `Vec` either way.
pub struct HttpRangeSource {
    url: String,
    client: reqwest::blocking::Client,
    on_bytes: Option<ByteProgressCallback>,
}

impl HttpRangeSource {
    pub fn new(url: impl Into<String>) -> Self {
        let client = reqwest::blocking::Client::builder()
            // `http1_only` IS the optimization; [`RANGE_CONCURRENCY`] on its
            // own is not. Every one of these URLs redirects to the Xet LFS
            // bridge, which speaks HTTP/2, and reqwest would then multiplex
            // all the concurrent chunk GETs onto ONE connection -- so one
            // CloudFront edge, so the one per-edge rate cap, so the
            // concurrency buys exactly nothing. There is no error and no
            // warning in that case, only the old wall clock (AGENTS.md
            // Gotcha 46). HTTP/1.1 forces a connection per in-flight
            // request, which is what the measurement in
            // [`RANGE_CONCURRENCY`]'s table was taken over.
            .http1_only()
            .pool_max_idle_per_host(RANGE_CONCURRENCY)
            .build()
            // Matches `Client::new`, which panics on the same failure.
            .expect("blocking HTTP client");
        Self {
            url: url.into(),
            client,
            on_bytes: None,
        }
    }

    pub fn with_progress(url: impl Into<String>, on_bytes: ByteProgressCallback) -> Self {
        let client = reqwest::blocking::Client::builder()
            .http1_only()
            .pool_max_idle_per_host(RANGE_CONCURRENCY)
            .build()
            .expect("blocking HTTP client");
        Self {
            url: url.into(),
            client,
            on_bytes: Some(on_bytes),
        }
    }

    /// One `Range` GET, no retry, written straight into `dst`. Length is
    /// checked here so a truncated body is a retryable error rather than
    /// silent corruption.
    fn read_chunk(
        &self,
        start: u64,
        end_exclusive: u64,
        dst: &mut [u8],
    ) -> Result<(), DownloadError> {
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
        dst.copy_from_slice(&bytes);
        if let Some(on_bytes) = &self.on_bytes {
            on_bytes(bytes.len() as u64);
        }
        Ok(())
    }

    /// [`Self::read_chunk`] under the retry ladder: [`RANGE_ATTEMPTS`]
    /// attempts with an exponential backoff capped at four seconds. A range
    /// the server will not serve at all is not going to start working, so
    /// `UnexpectedStatus` is fatal; everything else is treated as transport.
    fn read_chunk_retrying(
        &self,
        start: u64,
        end_exclusive: u64,
        dst: &mut [u8],
    ) -> Result<(), DownloadError> {
        let mut last = None;
        for attempt in 0..RANGE_ATTEMPTS {
            match self.read_chunk(start, end_exclusive, dst) {
                Ok(()) => return Ok(()),
                Err(e @ DownloadError::UnexpectedStatus { .. }) => return Err(e),
                Err(e) => {
                    last = Some(e);
                    if attempt + 1 < RANGE_ATTEMPTS {
                        std::thread::sleep(std::time::Duration::from_millis(250 << attempt.min(4)));
                    }
                }
            }
        }
        Err(last.expect("RANGE_ATTEMPTS is nonzero, so a failed loop recorded an error"))
    }
}

impl RangeSource for HttpRangeSource {
    fn read_range(&self, start: u64, end_exclusive: u64) -> Result<Vec<u8>, DownloadError> {
        let chunks = chunk_ranges(start, end_exclusive, MAX_RANGE_BYTES);
        let mut out = vec![0u8; (end_exclusive - start) as usize];
        fill_chunks(
            &mut out,
            &chunks,
            RANGE_CONCURRENCY,
            |chunk_start, chunk_end, dst| self.read_chunk_retrying(chunk_start, chunk_end, dst),
        )?;
        Ok(out)
    }
}
