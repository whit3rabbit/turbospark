//! HTTP range download implementation using reqwest.

use std::fs::{self, File, OpenOptions};
use std::io::{Read as _, Seek as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use super::chunks::{
    chunk_ranges, fill_chunks, MAX_RANGE_BYTES, RANGE_ATTEMPTS, RANGE_CONCURRENCY,
};
use super::{DownloadError, RangeSource};

/// Callback invoked on each downloaded chunk with the chunk's byte count.
pub type ByteProgressCallback = Arc<dyn Fn(u64) + Send + Sync>;

/// A cloneable cancellation flag for a download walk.
///
/// One walk holds one flag; the caller clones it out BEFORE starting so a
/// UI thread can `cancel()` it while the walk is blocking in `read_range`.
/// Every chunk read checks the flag, so a walk stops within one chunk
/// window (a few seconds at the chunk sizes this module issues) rather
/// than at the next tensor or shard boundary. Not a future and not a
/// channel. Pausing parks the walk at a checkpoint without discarding its
/// buffers or files; cancellation wakes paused workers so they can exit.
#[derive(Clone)]
pub struct CancelFlag(Arc<DownloadControl>);

struct DownloadControl {
    cancelled: AtomicBool,
    paused: Mutex<bool>,
    wake: Condvar,
}

impl CancelFlag {
    pub fn new() -> Self {
        Self(Arc::new(DownloadControl {
            cancelled: AtomicBool::new(false),
            paused: Mutex::new(false),
            wake: Condvar::new(),
        }))
    }

    pub fn cancel(&self) {
        // Use the same lock as the wait to avoid losing a cancel between
        // checking the predicate and parking a worker.
        let _paused = self.0.paused.lock().unwrap_or_else(|p| p.into_inner());
        self.0
            .cancelled
            .store(true, std::sync::atomic::Ordering::Release);
        self.0.wake.notify_all();
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.cancelled.load(std::sync::atomic::Ordering::Acquire)
    }

    /// In-flight requests finish; subsequent requests wait without losing work.
    pub fn pause(&self) -> bool {
        let mut paused = self.0.paused.lock().unwrap_or_else(|p| p.into_inner());
        if self.is_cancelled() {
            return false;
        }
        *paused = true;
        true
    }

    pub fn resume(&self) -> bool {
        let mut paused = self.0.paused.lock().unwrap_or_else(|p| p.into_inner());
        if self.is_cancelled() {
            return false;
        }
        *paused = false;
        self.0.wake.notify_all();
        true
    }

    /// Wait before issuing a request, never while holding an HTTP response.
    /// A long pause must not consume that response's timeout or retry budget.
    pub fn checkpoint(&self) -> Result<(), DownloadError> {
        let mut paused = self.0.paused.lock().unwrap_or_else(|p| p.into_inner());
        while *paused && !self.is_cancelled() {
            paused = self.0.wake.wait(paused).unwrap_or_else(|p| p.into_inner());
        }
        if self.is_cancelled() {
            Err(DownloadError::Cancelled)
        } else {
            Ok(())
        }
    }

    /// Identity test for the guard that unregisters its own flag when the
    /// walk ends.
    pub fn same_flag(&self, other: &CancelFlag) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Default for CancelFlag {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for CancelFlag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancelFlag")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

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
    /// Set by `with_cancel`: when the flag fires, chunk reads abort with
    /// `DownloadError::Cancelled`. `None` (the default) is every existing
    /// caller -- the CLI has nothing to cancel a walk with, so nothing
    /// there changes shape.
    cancel: Option<CancelFlag>,
    /// Optional durable cache for completed ranges. The catalog enables this
    /// only for immutable repository revisions, so a URL identifies stable
    /// bytes across retries and app launches.
    cache_dir: Option<PathBuf>,
}

/// `connect_timeout` bounds the TCP/TLS handshake; `timeout` bounds the
/// ENTIRE request (reqwest's blocking client has no separate read timeout in
/// 0.12.28 -- checked in the vendored source -- so this is the only knob
/// that bounds a stalled read). Sized for one [`MAX_RANGE_BYTES`] chunk: a
/// slower-than-~110 KB/s edge still fails within the window rather than
/// hanging indefinitely, and the retry ladder (`RANGE_ATTEMPTS`) is what
/// recovers from a single slow or dropped chunk.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

fn build_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
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
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(REQUEST_TIMEOUT)
        .build()
        // Matches `Client::new`, which panics on the same failure.
        .expect("blocking HTTP client")
}

impl HttpRangeSource {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            client: build_client(),
            on_bytes: None,
            token: None,
            cancel: None,
            cache_dir: None,
        }
    }

    pub fn with_progress(url: impl Into<String>, on_bytes: ByteProgressCallback) -> Self {
        Self {
            url: url.into(),
            client: build_client(),
            on_bytes: Some(on_bytes),
            token: None,
            cancel: None,
            cache_dir: None,
        }
    }

    /// Attach a cancel flag: once `CancelFlag::cancel` has been called on
    /// it, every chunk read this source issues aborts with
    /// `DownloadError::Cancelled`.
    pub fn with_cancel(mut self, cancel: CancelFlag) -> Self {
        self.cancel = Some(cancel);
        self
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

    /// Persist completed ranges below `directory` so a later walk can reuse
    /// them. Cache entries carry their own SHA-256 and are ignored on any I/O
    /// or integrity failure. The final install still performs its normal
    /// manifest verification.
    pub fn with_cache_dir(mut self, directory: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(directory.into());
        self
    }

    fn cache_path(&self, start: u64, end_exclusive: u64) -> Option<PathBuf> {
        let root = self.cache_dir.as_ref()?;
        let source = model_io::hash_data(self.url.as_bytes());
        Some(
            root.join(source)
                .join(format!("{start}-{end_exclusive}.range")),
        )
    }

    fn read_cached_chunk(&self, start: u64, end_exclusive: u64, dst: &mut [u8]) -> bool {
        let Some(path) = self.cache_path(start, end_exclusive) else {
            return false;
        };
        let expected_len = dst.len() as u64 + 64;
        let result = (|| -> Result<(), std::io::Error> {
            let mut file = File::open(&path)?;
            if file.metadata()?.len() != expected_len {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "cached range has the wrong length",
                ));
            }
            let mut expected_hash = [0u8; 64];
            file.read_exact(&mut expected_hash)?;
            file.read_exact(dst)?;
            if model_io::hash_data(dst).as_bytes() != expected_hash {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "cached range hash mismatch",
                ));
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                if let Some(on_bytes) = &self.on_bytes {
                    on_bytes(dst.len() as u64);
                }
                true
            }
            Err(_) => {
                let _ = fs::remove_file(path);
                false
            }
        }
    }

    fn persist_cached_chunk(&self, start: u64, end_exclusive: u64, bytes: &[u8]) {
        static CACHE_TEMP_ID: AtomicU64 = AtomicU64::new(0);
        let Some(path) = self.cache_path(start, end_exclusive) else {
            return;
        };
        let Some(parent) = path.parent() else {
            return;
        };
        if fs::create_dir_all(parent).is_err() {
            return;
        }
        let suffix = CACHE_TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let temp = parent.join(format!(".range-{}-{suffix}.tmp", std::process::id()));
        let write = (|| -> Result<(), std::io::Error> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp)?;
            file.write_all(model_io::hash_data(bytes).as_bytes())?;
            file.write_all(bytes)?;
            match fs::rename(&temp, &path) {
                Ok(()) => Ok(()),
                Err(_) if path.is_file() => {
                    let _ = fs::remove_file(&temp);
                    Ok(())
                }
                Err(error) => Err(error),
            }
        })();
        if write.is_err() {
            let _ = fs::remove_file(temp);
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
        if let Some(cancel) = &self.cancel {
            cancel.checkpoint()?;
        }
        if self.read_cached_chunk(start, end_exclusive, dst) {
            return Ok(());
        }
        let end_inclusive = end_exclusive.saturating_sub(1);
        let mut request = self.client.get(&self.url).header(
            reqwest::header::RANGE,
            format!("bytes={start}-{end_inclusive}"),
        );
        if let Some(token) = &self.token {
            request = request.bearer_auth(token);
        }
        let mut response = request
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
        // Read straight into `dst` rather than buffering the whole body via
        // `response.bytes()` and then copying: with `RANGE_CONCURRENCY`
        // chunks in flight, that doubled peak memory above the caller's own
        // buffer (an extra ~512 MiB at the default chunk size and
        // concurrency), which is exactly the overhead the disjoint-slice
        // design in `fill_chunks` exists to avoid. A manual loop rather than
        // `read_exact` so a short body reports how much it actually got
        // (`read_exact` only distinguishes "all" from "not all").
        let expected = end_exclusive - start;
        let mut read = 0usize;
        while read < dst.len() {
            match response.read(&mut dst[read..]) {
                Ok(0) => break,
                Ok(n) => read += n,
                Err(e) => return Err(DownloadError::Request(e.to_string())),
            }
        }
        if read as u64 != expected {
            return Err(DownloadError::ShortRead {
                expected,
                actual: read as u64,
            });
        }
        self.persist_cached_chunk(start, end_exclusive, dst);
        if let Some(on_bytes) = &self.on_bytes {
            on_bytes(read as u64);
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
                // A cancel is not a network condition: retrying it would
                // spend the ladder's backoffs pretending the caller did not
                // just ask the walk to stop.
                Err(e @ DownloadError::Cancelled) => return Err(e),
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

    /// Download a complete remote file into a sibling `.partial` file using
    /// the same bounded concurrent ranges, retry ladder, cancellation, and
    /// durable cache as `read_range`. Only a complete file is renamed into
    /// `destination`.
    pub fn download_to(&self, destination: &Path, total_bytes: u64) -> Result<u64, DownloadError> {
        if destination.exists() {
            return Err(DownloadError::Request(format!(
                "refusing to overwrite {}",
                destination.display()
            )));
        }
        let parent = destination.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).map_err(|e| DownloadError::Request(e.to_string()))?;
        let name = destination
            .file_name()
            .ok_or_else(|| DownloadError::Request("download destination has no file name".into()))?
            .to_string_lossy();
        let partial = parent.join(format!(".{name}.partial"));
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&partial)
            .map_err(|e| DownloadError::Request(e.to_string()))?;
        file.set_len(total_bytes)
            .map_err(|e| DownloadError::Request(e.to_string()))?;

        let chunks = chunk_ranges(0, total_bytes, MAX_RANGE_BYTES);
        let queue = Mutex::new(chunks.iter().copied().enumerate().collect::<Vec<_>>());
        let output = Mutex::new(file);
        let failures = Mutex::new(Vec::<(usize, DownloadError)>::new());
        let workers = RANGE_CONCURRENCY.clamp(1, chunks.len().max(1));
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| {
                    while let Some((index, (start, end_exclusive))) =
                        queue.lock().expect("download queue lock").pop()
                    {
                        let mut bytes = vec![0u8; (end_exclusive - start) as usize];
                        let result = self
                            .read_chunk_retrying(start, end_exclusive, &mut bytes)
                            .and_then(|()| {
                                let mut output = output.lock().expect("download output lock");
                                output
                                    .seek(std::io::SeekFrom::Start(start))
                                    .and_then(|_| output.write_all(&bytes))
                                    .map_err(|e| DownloadError::Request(e.to_string()))
                            });
                        if let Err(error) = result {
                            failures
                                .lock()
                                .expect("download failure lock")
                                .push((index, error));
                        }
                    }
                });
            }
        });
        let mut failures = failures.into_inner().expect("download failure lock");
        failures.sort_by_key(|(index, _)| *index);
        if let Some((_, error)) = failures.into_iter().next() {
            return Err(error);
        }
        output
            .into_inner()
            .expect("download output lock")
            .sync_all()
            .map_err(|e| DownloadError::Request(e.to_string()))?;
        fs::rename(&partial, destination).map_err(|e| DownloadError::Request(e.to_string()))?;
        Ok(total_bytes)
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
        if let Some(cancel) = &self.cancel {
            cancel.checkpoint()?;
        }
        if start > end_exclusive {
            return Err(DownloadError::InvalidRange {
                start,
                end_exclusive,
            });
        }
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
    use std::fs;
    use std::io::Write as _;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    use super::super::{DownloadError, RangeSource};
    use super::{
        throttle_backoff, throttled_status, ByteProgressCallback, CancelFlag, HttpRangeSource,
    };

    fn temp_dir(label: &str) -> std::path::PathBuf {
        static ID: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "turbospark-range-{label}-{}-{}",
            std::process::id(),
            ID.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn seed_cache(source: &HttpRangeSource, start: u64, bytes: &[u8]) {
        let path = source
            .cache_path(start, start + bytes.len() as u64)
            .expect("cache path");
        fs::create_dir_all(path.parent().unwrap()).expect("cache parent");
        let mut file = fs::File::create(path).expect("cache file");
        file.write_all(model_io::hash_data(bytes).as_bytes())
            .expect("cache hash");
        file.write_all(bytes).expect("cache bytes");
    }

    #[test]
    fn an_inverted_range_is_rejected_before_any_request() {
        let source = HttpRangeSource::new("http://127.0.0.1:1/nothing");
        assert_eq!(
            source.read_range(2, 1),
            Err(DownloadError::InvalidRange {
                start: 2,
                end_exclusive: 1,
            })
        );
    }

    #[test]
    fn a_verified_cached_range_skips_the_network_and_reports_progress() {
        let root = temp_dir("hit");
        let observed = Arc::new(AtomicU64::new(0));
        let progress_observed = Arc::clone(&observed);
        let progress: ByteProgressCallback = Arc::new(move |bytes| {
            progress_observed.fetch_add(bytes, Ordering::Relaxed);
        });
        let source = HttpRangeSource::with_progress("http://127.0.0.1:1/nothing", progress)
            .with_cache_dir(&root);
        seed_cache(&source, 0, b"cached bytes");
        assert_eq!(
            source.read_range(0, 12).expect("cache hit"),
            b"cached bytes"
        );
        assert_eq!(observed.load(Ordering::Relaxed), 12);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn a_corrupt_cached_range_is_removed_instead_of_trusted() {
        let root = temp_dir("corrupt");
        let source = HttpRangeSource::new("http://127.0.0.1:1/nothing").with_cache_dir(&root);
        seed_cache(&source, 0, b"good");
        let path = source.cache_path(0, 4).expect("cache path");
        fs::write(&path, b"bad").expect("corrupt cache");
        let mut out = [0u8; 4];
        assert!(!source.read_cached_chunk(0, 4, &mut out));
        assert!(!path.exists(), "corrupt cache entry must be discarded");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn a_cached_complete_file_is_published_from_a_partial_file() {
        let root = temp_dir("file");
        let destination = root.join("model.safetensors");
        let cache = root.join("cache");
        let source = HttpRangeSource::new("http://127.0.0.1:1/nothing").with_cache_dir(&cache);
        seed_cache(&source, 0, b"complete file");
        assert_eq!(
            source
                .download_to(&destination, 13)
                .expect("cache download"),
            13
        );
        assert_eq!(
            fs::read(&destination).expect("published file"),
            b"complete file"
        );
        assert!(!root.join(".model.safetensors.partial").exists());
        let _ = fs::remove_dir_all(root);
    }

    /// A fired flag aborts `read_range` BEFORE any request: the port here
    /// (:1) is unreachable, so a probe that reached the network would fail
    /// with `Request`, not `Cancelled`. That distinction IS the test --
    /// cancellation is the walk's own decision, not a transport failure,
    /// and the retry ladder must never see it.
    #[test]
    fn a_fired_cancel_flag_aborts_read_range_before_any_request() {
        let flag = CancelFlag::new();
        let source = HttpRangeSource::new("http://127.0.0.1:1/nothing").with_cancel(flag.clone());
        assert!(!flag.is_cancelled());
        assert!(
            !matches!(source.read_range(0, 64), Err(DownloadError::Cancelled)),
            "before cancel() the walk must still try to read"
        );
        flag.cancel();
        assert!(flag.is_cancelled());
        assert_eq!(
            source.read_range(0, 64),
            Err(DownloadError::Cancelled),
            "after cancel() the flag must stop the walk before any request"
        );
    }

    /// A flag attached to no source must not lose its identity: the guard
    /// that deregisters a finished walk compares by `Arc` identity, so a
    /// clone and its original are the same flag and two fresh flags are
    /// not.
    #[test]
    fn clones_share_state_and_fresh_flags_are_distinct() {
        let flag = CancelFlag::new();
        let clone = flag.clone();
        let other = CancelFlag::new();
        assert!(flag.same_flag(&clone));
        assert!(clone.same_flag(&flag));
        assert!(!flag.same_flag(&other));
        clone.cancel();
        assert!(flag.is_cancelled(), "state must flow through the clone");
        assert!(!other.is_cancelled());
    }

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
