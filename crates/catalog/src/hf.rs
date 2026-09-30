//! The Hugging Face surface: resolve URLs, the repository file list, and
//! small-file GETs.
//!
//! Model pulls use ranged reads through `repack::HttpRangeSource`. Image
//! packing has a different contract: the packer needs complete safetensors
//! shards in a temporary Diffusers tree before it can emit its own packed
//! payload. `Client::download_to` is the deliberately narrow full-file arm
//! for that path, with the same retry ladder and authentication as metadata
//! calls. It never buffers a weight file in memory.
//!
//! Two things worth knowing about the endpoints:
//!
//! - `resolve/<rev>/<path>` 302s into the Xet LFS bridge for anything
//!   LFS-backed and serves small files directly. Both are fine for a plain
//!   GET; only ranged reads care about the difference.
//! - `api/models/<repo>/revision/<rev>` returns a `siblings` array naming
//!   every file in the repo. That is how [`probe`](crate::probe) learns which
//!   sidecars exist rather than guessing a list, which is the failure that
//!   cost a 20-minute re-stream (AGENTS.md Gotcha 47).

use crate::hub_validation::{HubMetadataValidator, HubValidationError, ValidationReport};
use serde_json::Value;
use std::io::Read;

/// Typed bounded GET failures for the explicit live catalog path.
///
/// Other Hugging Face callers retain the legacy string-returning methods.
#[derive(Debug)]
pub(crate) enum HubRequestError {
    Offline,
    Transport(String),
    HttpStatus {
        status: u16,
        retry_after_secs: Option<u64>,
    },
    ResponseTooLarge,
    BodyRead(String),
}

/// An in-process override for [`hf_endpoint`], set by [`set_hf_endpoint_override`].
///
/// This exists instead of `std::env::set_var`/`remove_var` because this
/// crate's callers mutate the endpoint from a GUI thread (`ts_hf_endpoint_set`,
/// a user switching mirrors in settings) while `hf_endpoint()` is read from
/// worker threads for the whole life of a multi-minute install or probe walk
/// -- a data race on the process environment table with no synchronization,
/// which several libc implementations document as unsound under concurrent
/// access. A `RwLock` around a plain value has no such hazard.
static HF_ENDPOINT_OVERRIDE: std::sync::RwLock<Option<String>> = std::sync::RwLock::new(None);

/// The base URL for Hugging Face endpoints. Prefers the in-process override
/// set via [`set_hf_endpoint_override`], falling back to `$HF_ENDPOINT`
/// (read once at process start by most callers, so this still honors a
/// mirror set before this crate ever runs), defaulting to
/// `https://huggingface.co`.
pub fn hf_endpoint() -> String {
    if let Ok(guard) = HF_ENDPOINT_OVERRIDE.read() {
        if let Some(url) = guard.as_ref() {
            return url.trim_end_matches('/').to_string();
        }
    }
    std::env::var("HF_ENDPOINT")
        .unwrap_or_else(|_| "https://huggingface.co".to_string())
        .trim_end_matches('/')
        .to_string()
}

/// Sets (`Some`) or clears (`None`) the in-process override [`hf_endpoint`]
/// prefers over `$HF_ENDPOINT`. The synchronized replacement for mutating
/// the process environment directly; see [`hf_endpoint`]'s doc for why that
/// matters here specifically.
pub fn set_hf_endpoint_override(value: Option<String>) {
    if let Ok(mut guard) = HF_ENDPOINT_OVERRIDE.write() {
        *guard = value;
    }
}

/// A repository coordinate: `owner/name` at a revision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoRef {
    pub repo: String,
    pub revision: String,
}

impl RepoRef {
    /// Creates a new repository reference from owner/name and revision.
    pub fn new(repo: impl Into<String>, revision: impl Into<String>) -> Self {
        Self {
            repo: repo.into(),
            revision: revision.into(),
        }
    }

    /// Parse `owner/name` or `owner/name@revision`. An omitted revision is
    /// `main`, which is what every Hugging Face URL means by default and
    /// what the floating catalog rows already record.
    pub fn parse(text: &str) -> Result<Self, String> {
        let (repo, revision) = match text.split_once('@') {
            Some((repo, rev)) if !rev.is_empty() => (repo, rev),
            Some((_, _)) => return Err(format!("{text:?}: empty revision after '@'")),
            None => (text, "main"),
        };
        if repo.split('/').count() != 2 || repo.split('/').any(str::is_empty) {
            return Err(format!(
                "{repo:?} is not owner/name (try mlx-community/gemma-4-26b-a4b-it-4bit)"
            ));
        }
        Ok(Self::new(repo, revision))
    }

    /// The `resolve` URL for one file in this repository.
    pub fn file_url(&self, path: &str) -> String {
        format!(
            "{}/{}/resolve/{}/{}",
            hf_endpoint(),
            self.repo,
            self.revision,
            path
        )
    }

    /// The API URL for this repository's metadata, including `siblings`.
    pub fn api_url(&self) -> String {
        format!(
            "{}/api/models/{}/revision/{}",
            hf_endpoint(),
            self.repo,
            self.revision
        )
    }

    /// The same URL with `?blobs=true`, which adds a `size` to every sibling.
    ///
    /// One request for every file's length, where [`Client::content_length`]
    /// is one request per file. That difference decides whether ranking a
    /// repository's ten published quantizations is one round trip or ten, and
    /// [`crate::recommend`] does it across twenty repositories at once.
    pub fn api_url_with_sizes(&self) -> String {
        format!("{}?blobs=true", self.api_url())
    }
}

impl std::fmt::Display for RepoRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.repo, self.revision)
    }
}

/// One row of the popular-models listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PopularRepo {
    /// `owner/name`.
    pub id: String,
    pub downloads: u64,
    /// The checkpoint this artifact was converted from, when the card names
    /// one. **A GGUF repository carries no `tokenizer.json`**, so without
    /// this a discovered candidate has nowhere to get its sidecars and is
    /// refused for a reason that has nothing to do with whether it would run
    /// (`crates/catalog/CLAUDE.md` Gotcha 4).
    pub base_model: Option<String>,
}

/// A file in a repository, with its length when the listing carried one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoFile {
    pub name: String,
    pub size: Option<u64>,
}

/// A blocking HTTP client for the small-file endpoints.
///
/// Carries the `HF_TOKEN` bearer when one is set. Without it a gated
/// repository answers 401, which reads like a bug in this code rather than
/// like a missing credential, so [`Client::get`] says so by name.
pub struct Client {
    inner: reqwest::blocking::Client,
    token: Option<String>,
}

impl Default for Client {
    fn default() -> Self {
        Self::new()
    }
}

impl Client {
    /// Creates a new Hugging Face HTTP client, resolving an authentication
    /// token from flags, environment, store, or system cache.
    pub fn new() -> Self {
        Self::with_token(None)
    }

    /// Creates a client with a caller-supplied request timeout.
    ///
    /// Interactive callers use a shorter bound than model installation: a
    /// recommendation refresh may inspect several independent repositories,
    /// and one unavailable host must not leave the UI spinning indefinitely.
    pub fn with_timeout(timeout: std::time::Duration) -> Self {
        Self::with_token_and_timeout(None, timeout)
    }

    /// Creates a new Hugging Face HTTP client with an optional explicit token override.
    pub fn with_token(explicit_token: Option<String>) -> Self {
        Self::with_token_and_timeout(explicit_token, std::time::Duration::from_secs(120))
    }

    fn with_token_and_timeout(
        explicit_token: Option<String>,
        timeout: std::time::Duration,
    ) -> Self {
        let inner = reqwest::blocking::Client::builder()
            .timeout(timeout)
            .build()
            .expect("blocking HTTP client");
        let token = crate::auth::resolve_hf_token(explicit_token.as_deref());
        Self { inner, token }
    }

    /// The resolved authentication token, if any.
    pub fn token(&self) -> Option<&str> {
        self.token.as_deref()
    }

    /// Whether a token was found, so `probe` can say "no HF_TOKEN is set"
    /// beside a 401 rather than leaving the user to guess.
    pub fn has_token(&self) -> bool {
        self.token.is_some()
    }

    fn send_response(&self, url: &str) -> Result<reqwest::blocking::Response, reqwest::Error> {
        let mut request = self.inner.get(url);
        if let Some(token) = &self.token {
            request = request.bearer_auth(token);
        }
        request.send()
    }

    /// A request under the same retry ladder
    /// `repack::HttpRangeSource::read_chunk_retrying` carries for weight
    /// ranges (`crates/repack/CLAUDE.md` Gotcha 14) -- a SEPARATE
    /// implementation rather than a shared one, because this module's own
    /// doc says why the two clients stay apart (a second client sharing the
    /// ranged downloader's chunking/retry/`http1_only()` machinery would be
    /// the wrong tool for a KB-scale GET). Found live: a real `qwen4_exp`
    /// pull's sidecar fetch (`vocab.json`) hit a 429 and aborted the whole
    /// 68 GiB walk on its first occurrence, the exact failure mode Gotcha 14
    /// already named and fixed -- for the OTHER client. This one had never
    /// received the same fix, because nothing had hit it here before.
    ///
    /// Every status this returns is handed to the caller UNCHANGED once it
    /// is not throttled (or attempts run out), so `get`/`get_optional`/
    /// `content_length`'s own 401/403/404 interpretation is untouched.
    fn send_retrying(&self, url: &str) -> Result<reqwest::blocking::Response, String> {
        self.send_retrying_response(url)
            .map_err(|error| format!("GET {url}: {error}"))
    }

    fn send_retrying_response(
        &self,
        url: &str,
    ) -> Result<reqwest::blocking::Response, reqwest::Error> {
        const ATTEMPTS: usize = 8;
        for attempt in 0..ATTEMPTS {
            let response = self.send_response(url)?;
            let status = response.status().as_u16();
            if !throttled_status(status) || attempt + 1 == ATTEMPTS {
                return Ok(response);
            }
            let retry_after = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.trim().parse::<u64>().ok());
            std::thread::sleep(throttle_backoff(attempt, retry_after));
        }
        unreachable!("ATTEMPTS is nonzero, so the loop always returns by its last iteration")
    }

    /// One small file. Fails with a message that names the likely cause for
    /// the two statuses that are not bugs: 401/403 (gated, needs `HF_TOKEN`)
    /// and 404 (the file is not in this repo at this revision, which is the
    /// sidecar-list trap).
    pub fn get(&self, url: &str) -> Result<Vec<u8>, String> {
        self.get_with_limit(url, None)
    }

    /// One small file with a hard response-body ceiling. Metadata callers use
    /// this before parsing so an oversized Hub response cannot be fully
    /// buffered before validation rejects it.
    pub fn get_bounded(&self, url: &str, max_bytes: usize) -> Result<Vec<u8>, String> {
        self.get_with_limit(url, Some(max_bytes))
    }

    /// Bounded GET for live catalog search and trending. It keeps connection
    /// failures, HTTP status, retry metadata, and body-limit failures typed so
    /// the Hub layer can preserve actionable failure state.
    pub(crate) fn get_bounded_for_hub(
        &self,
        url: &str,
        max_bytes: usize,
    ) -> Result<Vec<u8>, HubRequestError> {
        let response = self.send_retrying_response(url).map_err(|error| {
            if error.is_connect() {
                HubRequestError::Offline
            } else {
                HubRequestError::Transport(format!("GET {url}: {error}"))
            }
        })?;
        let status = response.status().as_u16();
        if status != 200 {
            return Err(HubRequestError::HttpStatus {
                status,
                retry_after_secs: response
                    .headers()
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| value.trim().parse::<u64>().ok()),
            });
        }
        read_response_bounded_for_hub(response, max_bytes)
    }

    /// Opens one explicit repository file as a stream for the exact-source
    /// transfer path. The caller consumes bounded chunks and owns cancellation,
    /// hashing, and staging policy; the response is never buffered here.
    pub(crate) fn open_stream_for_hub(
        &self,
        url: &str,
    ) -> Result<reqwest::blocking::Response, HubRequestError> {
        let response = self.send_retrying_response(url).map_err(|error| {
            if error.is_connect() {
                HubRequestError::Offline
            } else {
                HubRequestError::Transport(format!("GET {url}: {error}"))
            }
        })?;
        if response.status().as_u16() != 200 {
            return Err(HubRequestError::HttpStatus {
                status: response.status().as_u16(),
                retry_after_secs: response
                    .headers()
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| value.trim().parse::<u64>().ok()),
            });
        }
        Ok(response)
    }

    fn get_with_limit(&self, url: &str, max_bytes: Option<usize>) -> Result<Vec<u8>, String> {
        let response = self.send_retrying(url)?;
        let status = response.status().as_u16();
        match status {
            200 => match max_bytes {
                Some(max_bytes) => read_response_bounded(response, url, max_bytes),
                None => response
                    .bytes()
                    .map(|b| b.to_vec())
                    .map_err(|e| format!("reading {url}: {e}")),
            },
            401 | 403 if !self.has_token() => Err(format!(
                "GET {url}: HTTP {status}. This repository is gated and no HF_TOKEN \
                 is set; accept its licence on huggingface.co, then export a token."
            )),
            401 | 403 => Err(format!(
                "GET {url}: HTTP {status}. HF_TOKEN is set, so either it lacks access \
                 to this repository or its licence has not been accepted."
            )),
            404 => Err(format!(
                "GET {url}: HTTP 404. That file is not in this repository at this \
                 revision -- file lists differ per repository, not per model family."
            )),
            _ => Err(format!("GET {url}: HTTP {status}")),
        }
    }

    /// `get` returning `None` on 404 rather than an error, for the files a
    /// caller is probing for rather than requiring.
    pub fn get_optional(&self, url: &str) -> Result<Option<Vec<u8>>, String> {
        let response = self.send_retrying(url)?;
        match response.status().as_u16() {
            200 => response
                .bytes()
                .map(|b| Some(b.to_vec()))
                .map_err(|e| format!("reading {url}: {e}")),
            404 => Ok(None),
            other => Err(format!("GET {url}: HTTP {other}")),
        }
    }

    /// Stream one repository file into an already-selected temporary path.
    ///
    /// The image source materializer needs complete safetensors shards before
    /// its packer can emit the checked image payload. The response body is
    /// copied in bounded chunks rather than collected as one large allocation.
    pub fn download_to(&self, url: &str, destination: &std::path::Path) -> Result<u64, String> {
        let mut response = self.send_retrying(url)?;
        let status = response.status().as_u16();
        if status != 200 {
            return Err(match status {
                401 | 403 if !self.has_token() => format!("GET {url}: HTTP {status}. This repository is gated and no HF_TOKEN is set; accept its licence on huggingface.co, then export a token."),
                401 | 403 => format!("GET {url}: HTTP {status}. HF_TOKEN is set, so either it lacks access to this repository or its licence has not been accepted."),
                404 => format!(
                    "GET {url}: HTTP 404. That file is not in this repository at this revision."
                ),
                _ => format!("GET {url}: HTTP {status}"),
            });
        }
        let parent = destination
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("creating download directory {}: {e}", parent.display()))?;
        let partial = destination.with_extension(format!(
            "{}partial-{}",
            destination
                .extension()
                .and_then(|extension| extension.to_str())
                .map(|extension| format!("{extension}."))
                .unwrap_or_default(),
            std::process::id()
        ));
        let mut file = std::fs::File::create(&partial)
            .map_err(|e| format!("creating download {}: {e}", partial.display()))?;
        let bytes =
            std::io::copy(&mut response, &mut file).map_err(|e| format!("reading {url}: {e}"))?;
        file.sync_all()
            .map_err(|e| format!("syncing download {}: {e}", partial.display()))?;
        drop(file);
        if destination.exists() {
            let _ = std::fs::remove_file(&partial);
            return Err(format!(
                "refusing to overwrite downloaded file {}",
                destination.display()
            ));
        }
        std::fs::rename(&partial, destination).map_err(|e| {
            let _ = std::fs::remove_file(&partial);
            format!("publishing download {}: {e}", destination.display())
        })?;
        Ok(bytes)
    }

    /// Every filename in the repository at this revision.
    ///
    /// This is the one call that makes probing a repository possible without
    /// knowing anything about it in advance: it says whether the weights are
    /// safetensors or GGUF, which `.gguf` files are on offer, and which
    /// tokenizer sidecars actually exist.
    pub fn file_list(&self, repo: &RepoRef) -> Result<Vec<String>, String> {
        self.file_list_report(repo).map(|report| {
            report_rejections(&report);
            report.valid
        })
    }

    /// Validated file names plus per-file rejections for callers that can
    /// present metadata diagnostics while retaining unrelated safe siblings.
    pub fn file_list_report(
        &self,
        repo: &RepoRef,
    ) -> Result<ValidationReport<Vec<String>>, String> {
        let validator = HubMetadataValidator::default();
        let body = self.get_bounded(&repo.api_url(), validator.limits().max_response_bytes)?;
        validate_file_list_payload(&body, &validator, &repo.repo, &repo.revision)
    }

    /// Every filename in the repository, with its length.
    ///
    /// The names come back sorted like [`Self::file_list`]'s, so the two are
    /// interchangeable where only names are wanted. `size` is `None` for a
    /// sibling the API listed without one rather than 0, because a zero-byte
    /// file and an unreported length are different facts and only one of them
    /// should sort to the bottom of a size ranking (the same reasoning the
    /// probe's unsized ggml types get -- `crates/catalog/CLAUDE.md`).
    pub fn file_list_with_sizes(&self, repo: &RepoRef) -> Result<Vec<RepoFile>, String> {
        self.file_list_with_sizes_report(repo).map(|report| {
            report_rejections(&report);
            report.valid
        })
    }

    /// Validated file names and sizes plus per-file rejections.
    pub fn file_list_with_sizes_report(
        &self,
        repo: &RepoRef,
    ) -> Result<ValidationReport<Vec<RepoFile>>, String> {
        let validator = HubMetadataValidator::default();
        let body = self.get_bounded(
            &repo.api_url_with_sizes(),
            validator.limits().max_response_bytes,
        )?;
        validate_file_list_with_sizes_payload(&body, &validator, &repo.repo, &repo.revision)
    }

    /// The most-downloaded GGUF text-generation repositories.
    ///
    /// The entry point for [`crate::recommend::discover`], and the one call in
    /// this module that does not start from a repository the caller already
    /// named. Adapted from shoehorn's `popular_gguf_repos` (see `NOTICE`); the
    /// changes are that it goes through this module's client rather than
    /// shelling out to `curl`, and that it asks for `cardData` so the
    /// sidecar repository comes back in the same request.
    pub fn popular_gguf_repos(&self, limit: usize) -> Result<Vec<PopularRepo>, String> {
        self.popular_gguf_repos_report(limit).map(|report| {
            report_rejections(&report);
            report.valid
        })
    }

    /// Validated popular repository rows plus rejected entries from the Hub
    /// response. The legacy vector method remains available to callers that
    /// cannot display row-level diagnostics.
    pub fn popular_gguf_repos_report(
        &self,
        limit: usize,
    ) -> Result<ValidationReport<Vec<PopularRepo>>, String> {
        let url = format!(
            "{}/api/models?filter=gguf&pipeline_tag=text-generation\
             &sort=downloads&direction=-1&cardData=true&limit={limit}",
            hf_endpoint()
        );
        let validator = HubMetadataValidator::default();
        let body = self.get_bounded(&url, validator.limits().max_response_bytes)?;
        validate_popular_list_payload(&body, &validator)
    }

    /// The `Content-Length` of a file, without fetching it.
    ///
    /// `probe` reports this and `tests/catalog_network.rs` asserts it, which
    /// is the only rot detector a `main`-pinned row has: a re-upload changes
    /// the size long before anybody notices the numbers moved.
    pub fn content_length(&self, url: &str) -> Result<Option<u64>, String> {
        let mut request = self.inner.head(url);
        if let Some(token) = &self.token {
            request = request.bearer_auth(token);
        }
        let response = request.send().map_err(|e| format!("HEAD {url}: {e}"))?;
        if !response.status().is_success() {
            return Err(format!("HEAD {url}: HTTP {}", response.status().as_u16()));
        }
        // `x-linked-size` is the LFS object's real length and is what a
        // Xet-backed file reports; `content-length` covers the rest.
        let headers = response.headers();
        let read = |key: &str| {
            headers
                .get(key)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
        };
        Ok(read("x-linked-size").or_else(|| read("content-length")))
    }
}

fn read_response_bounded(
    response: reqwest::blocking::Response,
    url: &str,
    max_bytes: usize,
) -> Result<Vec<u8>, String> {
    let max_u64 = u64::try_from(max_bytes).unwrap_or(u64::MAX);
    if response
        .content_length()
        .is_some_and(|content_length| content_length > max_u64)
    {
        return Err(format!(
            "GET {url}: response exceeds the {max_bytes}-byte metadata limit"
        ));
    }
    read_bounded_body(response, max_bytes).map_err(|error| format!("GET {url}: {error}"))
}

fn read_response_bounded_for_hub(
    response: reqwest::blocking::Response,
    max_bytes: usize,
) -> Result<Vec<u8>, HubRequestError> {
    let max_u64 = u64::try_from(max_bytes).unwrap_or(u64::MAX);
    if response
        .content_length()
        .is_some_and(|content_length| content_length > max_u64)
    {
        return Err(HubRequestError::ResponseTooLarge);
    }
    let read_limit = u64::try_from(max_bytes)
        .unwrap_or(u64::MAX)
        .saturating_add(1);
    let mut body = Vec::with_capacity(max_bytes.min(64 * 1024));
    response
        .take(read_limit)
        .read_to_end(&mut body)
        .map_err(|error| HubRequestError::BodyRead(error.to_string()))?;
    if body.len() > max_bytes {
        return Err(HubRequestError::ResponseTooLarge);
    }
    Ok(body)
}

fn read_bounded_body<R: Read>(reader: R, max_bytes: usize) -> std::io::Result<Vec<u8>> {
    let read_limit = u64::try_from(max_bytes)
        .unwrap_or(u64::MAX)
        .saturating_add(1);
    let mut body = Vec::with_capacity(max_bytes.min(64 * 1024));
    reader.take(read_limit).read_to_end(&mut body)?;
    if body.len() > max_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("response exceeds the {max_bytes}-byte metadata limit"),
        ));
    }
    Ok(body)
}

fn validate_file_list_payload(
    bytes: &[u8],
    validator: &HubMetadataValidator,
    expected_repo: &str,
    expected_revision: &str,
) -> Result<ValidationReport<Vec<String>>, String> {
    let report =
        validate_repository_for_request(bytes, validator, expected_repo, expected_revision)?;
    let mut valid: Vec<String> = report
        .valid
        .files
        .into_iter()
        .map(|file| file.path)
        .collect();
    valid.sort();
    Ok(ValidationReport {
        valid,
        rejected: report.rejected,
    })
}

fn validate_file_list_with_sizes_payload(
    bytes: &[u8],
    validator: &HubMetadataValidator,
    expected_repo: &str,
    expected_revision: &str,
) -> Result<ValidationReport<Vec<RepoFile>>, String> {
    let report =
        validate_repository_for_request(bytes, validator, expected_repo, expected_revision)?;
    let mut valid: Vec<RepoFile> = report
        .valid
        .files
        .into_iter()
        .map(|file| RepoFile {
            name: file.path,
            size: file.size_bytes,
        })
        .collect();
    valid.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(ValidationReport {
        valid,
        rejected: report.rejected,
    })
}

fn validate_repository_for_request(
    bytes: &[u8],
    validator: &HubMetadataValidator,
    expected_repo: &str,
    expected_revision: &str,
) -> Result<ValidationReport<crate::hub_validation::HubRepoMetadata>, String> {
    let report = validator
        .validate_repository_payload(bytes)
        .map_err(|error| error.to_string())?;
    if report.valid.repo_id != expected_repo {
        return Err(HubValidationError::InvalidResponse {
            entry: report.valid.repo_id,
            rule: "repository_id_matches_request",
        }
        .to_string());
    }
    // A floating ref such as `main` resolves to the response SHA. An
    // immutable caller ref must resolve to that exact commit.
    if is_commit_sha(expected_revision)
        && !report
            .valid
            .revision
            .eq_ignore_ascii_case(expected_revision)
    {
        return Err(HubValidationError::InvalidResponse {
            entry: format!("{expected_repo}@{}", report.valid.revision),
            rule: "revision_matches_request",
        }
        .to_string());
    }
    Ok(report)
}

fn is_commit_sha(revision: &str) -> bool {
    revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_popular_list_payload(
    bytes: &[u8],
    validator: &HubMetadataValidator,
) -> Result<ValidationReport<Vec<PopularRepo>>, String> {
    let report = validator
        .validate_search_payload(bytes)
        .map_err(|error| error.to_string())?;
    let raw: Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("parsing the popular-model listing: {error}"))?;
    let rows = raw
        .as_array()
        .expect("validated search response is an array");
    let valid = report
        .valid
        .into_iter()
        .map(|entry| {
            let raw_entry = rows.iter().find(|row| {
                row.get("id").and_then(Value::as_str) == Some(entry.repo_id.as_str())
                    && row
                        .get("sha")
                        .and_then(Value::as_str)
                        .is_some_and(|sha| sha.eq_ignore_ascii_case(&entry.revision))
            });
            let base_model = raw_entry
                .and_then(|row| row.get("cardData"))
                .and_then(|card_data| card_data.get("base_model"))
                .cloned()
                .and_then(base_model);
            PopularRepo {
                id: entry.repo_id,
                downloads: entry.downloads.unwrap_or(0),
                base_model,
            }
        })
        .collect();
    Ok(ValidationReport {
        valid,
        rejected: report.rejected,
    })
}

fn report_rejections<T>(report: &ValidationReport<T>) {
    for rejected in &report.rejected {
        eprintln!("{rejected}");
    }
}

/// Whether a non-2xx status is the server asking to be retried later, rather
/// than a verdict about the request. Mirrors
/// `repack::ranged_download::http::throttled_status` exactly (429 plus the
/// 5xx gateway pair); duplicated rather than shared because that function is
/// private to a crate this one does not depend on for exactly the reason
/// [`Client`]'s own doc gives.
fn throttled_status(status: u16) -> bool {
    matches!(status, 429 | 500 | 502 | 503 | 504)
}

/// How long to wait before re-issuing a throttled small-file GET. Mirrors
/// `repack::ranged_download::http::throttle_backoff`: the server's own
/// `Retry-After` wins when it sent one, otherwise an exponential backoff
/// from one second (not the sub-second range a plain retry ladder would use,
/// which re-hammers a rate limiter faster than its window moves), capped at
/// 30s so eight attempts stay bounded at about two minutes.
fn throttle_backoff(attempt: usize, retry_after: Option<u64>) -> std::time::Duration {
    const CAP: u64 = 30;
    let seconds = match retry_after {
        Some(after) => after.min(CAP),
        None => (1u64 << attempt.min(5)).min(CAP),
    };
    std::time::Duration::from_secs(seconds)
}

/// Read `cardData.base_model`, which the API spells as either a string or a
/// list of them and which a caller that assumed one shape gets `None` for
/// half the time. A list means the artifact was merged or converted from
/// several; the FIRST is the one whose tokenizer a conversion carries.
fn base_model(value: serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) if !s.trim().is_empty() => Some(s),
        serde_json::Value::Array(items) => items.into_iter().find_map(|v| match v {
            serde_json::Value::String(s) if !s.trim().is_empty() => Some(s),
            _ => None,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        base_model, read_bounded_body, throttle_backoff, throttled_status,
        validate_file_list_payload, validate_file_list_with_sizes_payload,
        validate_popular_list_payload, PopularRepo, RepoFile,
    };
    use crate::{HubMetadataValidator, HubValidationError, HubValidationLimits};
    use std::io::Read;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    struct CountingReader {
        bytes: Vec<u8>,
        read_bytes: Arc<AtomicUsize>,
    }

    impl Read for CountingReader {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            let count = buffer.len().min(self.bytes.len());
            buffer[..count].copy_from_slice(&self.bytes[..count]);
            self.bytes.drain(..count);
            self.read_bytes.fetch_add(count, Ordering::SeqCst);
            Ok(count)
        }
    }

    #[test]
    fn bounded_metadata_read_consumes_no_more_than_limit_plus_one() {
        let read_bytes = Arc::new(AtomicUsize::new(0));
        let reader = CountingReader {
            bytes: vec![b'x'; 100],
            read_bytes: Arc::clone(&read_bytes),
        };

        let result = read_bounded_body(reader, 8);

        assert!(result.is_err());
        assert_eq!(read_bytes.load(Ordering::SeqCst), 9);
    }

    #[test]
    fn file_list_payload_validation_keeps_safe_rows_and_reports_rejections() {
        let validator = HubMetadataValidator::default();
        let body = format!(
            r#"{{"id":"owner/model","sha":"{}","siblings":[
                {{"rfilename":"config.json","size":4}},
                {{"rfilename":"../escape.gguf","size":8}}
            ]}}"#,
            "0123456789abcdef0123456789abcdef01234567"
        );

        let report =
            validate_file_list_payload(body.as_bytes(), &validator, "owner/model", "main").unwrap();

        assert_eq!(report.valid, vec!["config.json"]);
        assert_eq!(report.rejected.len(), 1);
        assert_eq!(report.rejected[0].entry, "../escape.gguf");
        assert_eq!(report.rejected[0].rule, "safe_relative_path");
    }

    #[test]
    fn sized_file_list_payload_preserves_unknown_sizes_without_zeroing_them() {
        let validator = HubMetadataValidator::default();
        let body = format!(
            r#"{{"id":"owner/model","sha":"{}","siblings":[
                {{"rfilename":"weights.gguf","size":8}},
                {{"rfilename":"tokenizer.json"}}
            ]}}"#,
            "0123456789abcdef0123456789abcdef01234567"
        );

        let report = validate_file_list_with_sizes_payload(
            body.as_bytes(),
            &validator,
            "owner/model",
            "main",
        )
        .unwrap();

        assert_eq!(
            report.valid,
            vec![
                RepoFile {
                    name: "tokenizer.json".to_string(),
                    size: None
                },
                RepoFile {
                    name: "weights.gguf".to_string(),
                    size: Some(8)
                }
            ]
        );
    }

    #[test]
    fn popular_list_validation_filters_only_invalid_repositories() {
        let validator = HubMetadataValidator::default();
        let body = format!(
            r#"[
                {{"id":"owner/good","sha":"{}","downloads":12,"cardData":{{"base_model":"base/source"}}}},
                {{"id":"owner/../bad","sha":"{}","downloads":8}},
                {{"id":"owner/also-good","sha":"{}","downloads":null}}
            ]"#,
            "0123456789abcdef0123456789abcdef01234567",
            "0123456789abcdef0123456789abcdef01234567",
            "abcdef0123456789abcdef0123456789abcdef01"
        );

        let report = validate_popular_list_payload(body.as_bytes(), &validator).unwrap();

        assert_eq!(report.valid.len(), 2);
        assert_eq!(
            report.valid[0],
            PopularRepo {
                id: "owner/good".to_string(),
                downloads: 12,
                base_model: Some("base/source".to_string())
            }
        );
        assert_eq!(report.valid[1].id, "owner/also-good");
        assert_eq!(report.valid[1].downloads, 0);
        assert_eq!(report.rejected.len(), 1);
        assert_eq!(report.rejected[0].entry, "owner/../bad");
        assert_eq!(report.rejected[0].rule, "repository_id_shape");
    }

    #[test]
    fn oversized_json_is_rejected_at_the_reader_before_full_buffering() {
        let read_bytes = Arc::new(AtomicUsize::new(0));
        let reader = CountingReader {
            bytes: vec![b' '; 100],
            read_bytes: Arc::clone(&read_bytes),
        };
        let validator = HubMetadataValidator::new(HubValidationLimits {
            max_response_bytes: 8,
            ..HubValidationLimits::default()
        });
        let body = read_bounded_body(reader, validator.limits().max_response_bytes);

        assert!(body.is_err());
        assert_eq!(read_bytes.load(Ordering::SeqCst), 9);
        assert!(matches!(
            validator.validate_search_payload(&[b' '; 9]),
            Err(HubValidationError::OversizedResponse { .. })
        ));
    }

    /// 429 and the 5xx pair retry; nothing else does -- the exact set
    /// `repack`'s own `throttled_status` uses, which is what lets this
    /// module's retry ladder reproduce that fix rather than a narrower or
    /// wider version of it.
    #[test]
    fn throttled_status_matches_the_ranged_downloaders_set() {
        assert!(throttled_status(429));
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

    /// The server's own `Retry-After` wins over the guessed backoff, and
    /// both are capped so a server naming an hour cannot park a caller for
    /// one.
    #[test]
    fn throttle_backoff_prefers_retry_after_and_caps_both_arms() {
        assert_eq!(throttle_backoff(0, Some(7)).as_secs(), 7);
        assert_eq!(throttle_backoff(5, Some(3)).as_secs(), 3);
        assert_eq!(throttle_backoff(0, Some(3600)).as_secs(), 30);
        let secs: Vec<u64> = (0..8)
            .map(|a| throttle_backoff(a, None).as_secs())
            .collect();
        assert_eq!(secs, vec![1, 2, 4, 8, 16, 30, 30, 30]);
    }

    /// Both shapes the API actually returns. Measured live 2026-08-18:
    /// `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` answers a one-element
    /// LIST and `antirez/deepseek-v4-gguf` answers a bare STRING, so a reader
    /// written against either one alone is wrong about real repositories
    /// rather than about a hypothetical.
    #[test]
    fn a_base_model_is_read_from_a_string_or_a_list() {
        assert_eq!(
            base_model(serde_json::json!("deepseek-ai/DeepSeek-V4-Flash")),
            Some("deepseek-ai/DeepSeek-V4-Flash".to_string())
        );
        assert_eq!(
            base_model(serde_json::json!(["Qwen/Qwen3-Coder-30B-A3B-Instruct"])),
            Some("Qwen/Qwen3-Coder-30B-A3B-Instruct".to_string())
        );
        assert_eq!(base_model(serde_json::json!(null)), None);
        assert_eq!(base_model(serde_json::json!([])), None);
        // Blank is absent, not a repository named "".
        assert_eq!(base_model(serde_json::json!("  ")), None);
        assert_eq!(
            base_model(serde_json::json!(["", "owner/real"])),
            Some("owner/real".to_string())
        );
    }
}
