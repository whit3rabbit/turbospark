//! The Hugging Face surface: resolve URLs, the repository file list, and
//! small-file GETs.
//!
//! **This module deliberately does NOT download weights.** Everything here is
//! KB-scale: a JSON file list, a `config.json`, a tokenizer sidecar. Weight
//! bytes go through `repack::HttpRangeSource`, which has the chunking, the
//! retry ladder and the `http1_only()` client setting that the Xet bridge's
//! per-edge rate cap makes load-bearing (AGENTS.md Gotcha 46). A second
//! client here that fetched a multi-GB body would quietly lose all three.
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

use serde::Deserialize;

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
            "https://huggingface.co/{}/resolve/{}/{}",
            self.repo, self.revision, path
        )
    }

    /// The API URL for this repository's metadata, including `siblings`.
    pub fn api_url(&self) -> String {
        format!(
            "https://huggingface.co/api/models/{}/revision/{}",
            self.repo, self.revision
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

#[derive(Debug, Deserialize)]
struct Sibling {
    rfilename: String,
    /// Present only under `?blobs=true`.
    #[serde(default)]
    size: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct RepoInfo {
    #[serde(default)]
    siblings: Vec<Sibling>,
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

#[derive(Debug, Deserialize)]
struct CardData {
    /// Either a bare string or a list; the API is not consistent about which,
    /// and a caller that expects one gets `None` for half of Hugging Face.
    #[serde(default)]
    base_model: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct ListedModel {
    id: String,
    #[serde(default)]
    downloads: Option<u64>,
    #[serde(default)]
    #[serde(rename = "cardData")]
    card_data: Option<CardData>,
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
    /// Creates a new Hugging Face HTTP client, reading `HF_TOKEN` from the environment if present.
    pub fn new() -> Self {
        let inner = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("blocking HTTP client");
        let token = ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]
            .iter()
            .find_map(|k| std::env::var(k).ok())
            .filter(|t| !t.trim().is_empty());
        Self { inner, token }
    }

    /// Whether a token was found, so `probe` can say "no HF_TOKEN is set"
    /// beside a 401 rather than leaving the user to guess.
    pub fn has_token(&self) -> bool {
        self.token.is_some()
    }

    fn send(&self, url: &str) -> Result<reqwest::blocking::Response, String> {
        let mut request = self.inner.get(url);
        if let Some(token) = &self.token {
            request = request.bearer_auth(token);
        }
        request.send().map_err(|e| format!("GET {url}: {e}"))
    }

    /// [`Self::send`] under the same retry ladder
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
        const ATTEMPTS: usize = 8;
        for attempt in 0..ATTEMPTS {
            let response = self.send(url)?;
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
        let response = self.send_retrying(url)?;
        let status = response.status().as_u16();
        match status {
            200 => response
                .bytes()
                .map(|b| b.to_vec())
                .map_err(|e| format!("reading {url}: {e}")),
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

    /// Every filename in the repository at this revision.
    ///
    /// This is the one call that makes probing a repository possible without
    /// knowing anything about it in advance: it says whether the weights are
    /// safetensors or GGUF, which `.gguf` files are on offer, and which
    /// tokenizer sidecars actually exist.
    pub fn file_list(&self, repo: &RepoRef) -> Result<Vec<String>, String> {
        let body = self.get(&repo.api_url())?;
        let info: RepoInfo = serde_json::from_slice(&body)
            .map_err(|e| format!("parsing the file list for {repo}: {e}"))?;
        let mut names: Vec<String> = info.siblings.into_iter().map(|s| s.rfilename).collect();
        names.sort();
        Ok(names)
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
        let body = self.get(&repo.api_url_with_sizes())?;
        let info: RepoInfo = serde_json::from_slice(&body)
            .map_err(|e| format!("parsing the file list for {repo}: {e}"))?;
        let mut files: Vec<RepoFile> = info
            .siblings
            .into_iter()
            .map(|s| RepoFile {
                name: s.rfilename,
                size: s.size,
            })
            .collect();
        files.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(files)
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
        let url = format!(
            "https://huggingface.co/api/models?filter=gguf&pipeline_tag=text-generation\
             &sort=downloads&direction=-1&cardData=true&limit={limit}"
        );
        let body = self.get(&url)?;
        let listed: Vec<ListedModel> = serde_json::from_slice(&body)
            .map_err(|e| format!("parsing the popular-model listing: {e}"))?;
        Ok(listed
            .into_iter()
            .map(|m| PopularRepo {
                id: m.id,
                downloads: m.downloads.unwrap_or(0),
                base_model: m.card_data.and_then(|c| c.base_model).and_then(base_model),
            })
            .collect())
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
    use super::{base_model, throttle_backoff, throttled_status};

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
