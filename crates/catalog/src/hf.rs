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
}

impl std::fmt::Display for RepoRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.repo, self.revision)
    }
}

#[derive(Debug, Deserialize)]
struct Sibling {
    rfilename: String,
}

#[derive(Debug, Deserialize)]
struct RepoInfo {
    #[serde(default)]
    siblings: Vec<Sibling>,
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

    /// One small file. Fails with a message that names the likely cause for
    /// the two statuses that are not bugs: 401/403 (gated, needs `HF_TOKEN`)
    /// and 404 (the file is not in this repo at this revision, which is the
    /// sidecar-list trap).
    pub fn get(&self, url: &str) -> Result<Vec<u8>, String> {
        let response = self.send(url)?;
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
        let response = self.send(url)?;
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
