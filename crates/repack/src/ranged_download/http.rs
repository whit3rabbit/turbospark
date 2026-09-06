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
    token: Option<String>,
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
            token: None,
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
            token: None,
        }
    }

    /// Attach a bearer authentication token for gated repositories.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Attach an optional bearer authentication token for gated repositories.
    pub fn with_optional_token(mut self, token: Option<impl Into<String>>) -> Self {
        self.token = token.map(|t| t.into());
        self
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
        let mut request = self.client.get(&self.url).header(
            reqwest::header::RANGE,
            format!("bytes={start}-{end_inclusive}"),
        );
        if let Some(token) = &self.token {
            request = request.bearer_auth(token);
        }
        let response = request
            .send()
            .map_err(|e| DownloadError::Request(e.to_string()))?;
        if response.status().as_u16() != 206 {
            let status = response.status().as_u16();
            // `Retry-After` is read HERE because it is a property of this
            // response and is gone by the time the ladder sees the error.
            // Seconds form only: the HTTP-date form is legal and no CDN in
            // this walk's path uses it, so parsing one would be untested code
            // guarding an unobserved case.
            let retry_after = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.trim().parse::<u64>().ok());
            return Err(DownloadError::UnexpectedStatus {
                status,
                retry_after,
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
    /// attempts with an exponential backoff capped at four seconds.
    ///
    /// **A STATUS IS FATAL OR THROTTLED, AND FOR THE LIFE OF THIS SURFACE
    /// EVERY STATUS WAS FATAL.** The reasoning this replaces was sound for the
    /// case it was written for -- "a range the server will not serve at all is
    /// not going to start working" is true of 404 and 416 -- and it swept in
    /// the one status that means the exact opposite. 429 is the canonical
    /// "back off and retry"; treating it as permanent aborts a twenty-minute
    /// walk on its first occurrence, and this walk CANNOT RESUME, so the whole
    /// stream restarts.
    ///
    /// Found by M-V3's own gate: three consecutive 16 GB `qwen38` streams died
    /// on a 429 from the Xet CDN bridge (NOT from `huggingface.co/resolve`,
    /// whose `ratelimit` header read 2998 of 3000 remaining at the moment of
    /// failure -- the two have separate quotas and only one of them is
    /// observable from a response header).
    ///
    /// 5xx joins it for the same reason: a gateway error is the server saying
    /// "not now". Everything else stays fatal, so a 404 still fails on the
    /// first attempt rather than eight times.
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
                Err(DownloadError::UnexpectedStatus {
                    status,
                    retry_after,
                }) if throttled_status(status) => {
                    last = Some(DownloadError::UnexpectedStatus {
                        status,
                        retry_after,
                    });
                    if attempt + 1 < RANGE_ATTEMPTS {
                        std::thread::sleep(throttle_backoff(attempt, retry_after));
                    }
                }
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

/// Whether a non-206 status is the server asking to be retried later.
///
/// 429 is the whole point (see [`HttpRangeSource::read_chunk_retrying`]); the
/// 5xx pair are gateway failures that a CDN recovers from. A LIST rather than
/// a `>= 500` range, so a future status is fatal until someone establishes it
/// is not -- the failure this replaces came from a rule that was too broad,
/// and widening it back by default would repeat that in the other direction.
fn throttled_status(status: u16) -> bool {
    matches!(status, 429 | 500 | 502 | 503 | 504)
}

/// How long to wait before re-issuing a throttled range.
///
/// The server's own `Retry-After` wins when it sent one, because it knows its
/// window and this side is guessing. Otherwise an exponential backoff from ONE
/// SECOND, which is deliberately slower than the 250ms transport ladder: that
/// one recovers from a dropped connection, and re-hammering a rate limiter at
/// 250ms simply spends the retries before the window moves.
///
/// Capped at 30s so eight attempts span about a minute at worst, which is
/// bounded against a walk that already takes twenty.
fn throttle_backoff(attempt: usize, retry_after: Option<u64>) -> std::time::Duration {
    const CAP: u64 = 30;
    let seconds = match retry_after {
        Some(after) => after.min(CAP),
        None => (1u64 << attempt.min(5)).min(CAP),
    };
    std::time::Duration::from_secs(seconds)
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

#[cfg(test)]
mod tests {
    use super::{throttle_backoff, throttled_status};

    /// 429 retries and 404 does not, which is the whole correction.
    ///
    /// The old ladder returned on ANY non-206, so a rate limit killed a
    /// twenty-minute unresumable walk on its first occurrence. Both directions
    /// are asserted, because widening the rule until everything retries would
    /// make a 404 cost eight backoffs and is the same mistake mirrored.
    #[test]
    fn a_rate_limit_is_throttled_and_a_missing_range_is_fatal() {
        assert!(
            throttled_status(429),
            "429 is the canonical back-off status"
        );
        for s in [500, 502, 503, 504] {
            assert!(
                throttled_status(s),
                "{s} is a gateway failure, not a verdict"
            );
        }
        for s in [400, 401, 403, 404, 416, 451] {
            assert!(!throttled_status(s), "{s} will not start working");
        }
    }

    /// The server's own window wins over this side's guess.
    #[test]
    fn retry_after_overrides_the_backoff_and_both_are_capped() {
        assert_eq!(throttle_backoff(0, Some(7)).as_secs(), 7);
        assert_eq!(
            throttle_backoff(5, Some(3)).as_secs(),
            3,
            "the header wins even late"
        );
        // Capped, so a server naming an hour cannot park a walk for one.
        assert_eq!(throttle_backoff(0, Some(3600)).as_secs(), 30);
    }

    /// Without a header the wait grows, and starts a full second rather than
    /// the transport ladder's 250ms.
    ///
    /// Re-hammering a rate limiter four times a second spends every retry
    /// before the window moves, which is exactly how eight attempts can fail
    /// in under three seconds and read as a permanent refusal.
    #[test]
    fn the_default_backoff_grows_from_one_second_and_saturates() {
        let secs: Vec<u64> = (0..8)
            .map(|a| throttle_backoff(a, None).as_secs())
            .collect();
        assert_eq!(secs, vec![1, 2, 4, 8, 16, 30, 30, 30]);
        assert!(
            secs[0] >= 1,
            "starting below a second re-hammers the limiter"
        );
        // Bounded: eight attempts span about two minutes, against a walk that
        // already takes twenty.
        assert!(
            secs.iter().sum::<u64>() < 150,
            "the ladder must stay bounded"
        );
    }
}
