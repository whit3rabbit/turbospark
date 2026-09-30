//! Explicit Hugging Face search and trending results composed with the
//! bundled catalog.
//!
//! Constructing a [`HubClient`] does not perform I/O. The only network calls
//! in this module occur at [`HubClient::search`] and [`HubClient::trending`].
//! Live list metadata is validated before composition, and result composition
//! never probes a repository or changes a curated target's revision.

use crate::catalog::Catalog;
use crate::entry::CatalogEntry;
use crate::hf::{hf_endpoint, Client, RepoRef};
use crate::hub_validation::{
    HubMetadataValidator, HubSearchEntry, HubValidationError, RejectedHubEntry,
};
use crate::recommend::{
    recommend_catalog, CountedSource, Evidence, Fit, FitVerdict, GgufVariant, Machine, Origin,
    Recommendation,
};
use crate::store::Store;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Maximum list size accepted by the default metadata validator.
pub const MAX_HUB_PAGE_ENTRIES: u32 = 1_000;
/// Page size for the on-demand trending feed.
pub const DEFAULT_TRENDING_LIMIT: u32 = 30;
const SEARCH_FRESHNESS: Duration = Duration::from_secs(15 * 60);
const TRENDING_FRESHNESS: Duration = Duration::from_secs(60 * 60);
const CACHE_RETENTION_MULTIPLIER: u32 = 2;
const CACHE_ENVELOPE_OVERHEAD_BYTES: usize = 4 * 1024;
static CACHE_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy)]
enum CacheKind {
    Search,
    Trending,
}

impl CacheKind {
    fn freshness(self) -> Duration {
        match self {
            Self::Search => SEARCH_FRESHNESS,
            Self::Trending => TRENDING_FRESHNESS,
        }
    }
}

#[derive(Debug)]
struct ValidatedList {
    entries: Vec<HubSearchEntry>,
    rejected: Vec<RejectedHubEntry>,
    fetched_at: SystemTime,
    from_cache: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheEnvelope {
    /// Seconds since the Unix epoch, kept as an integer for a stable JSON
    /// envelope and deterministic freshness comparisons.
    fetched_at: u64,
    payload: Value,
}

/// Ordering requested from the Hugging Face model-list endpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum HubSort {
    /// Hugging Face's current trending score.
    #[default]
    Trending,
    /// Most downloads first.
    Downloads,
    /// Most likes first.
    Likes,
}

impl HubSort {
    fn api_value(self) -> &'static str {
        match self {
            Self::Trending => "trendingScore",
            Self::Downloads => "downloads",
            Self::Likes => "likes",
        }
    }
}

/// A user-submitted search phrase and bounded result-page options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubQuery {
    pub text: String,
    pub limit: u32,
    pub sort: HubSort,
}

impl HubQuery {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            limit: DEFAULT_TRENDING_LIMIT,
            sort: HubSort::Trending,
        }
    }

    pub fn with_limit(mut self, limit: u32) -> Self {
        self.limit = limit;
        self
    }

    pub fn with_sort(mut self, sort: HubSort) -> Self {
        self.sort = sort;
        self
    }
}

/// Whether a user action may use a fresh cached response or must fetch again.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Refresh {
    /// Use a fresh cached response when a cache is available.
    #[default]
    AllowCache,
    /// Skip any cached response and fetch again.
    Force,
}

/// One source label attached to a composed repository result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryProvenance {
    Bundled,
    LiveHub,
}

/// A merged row. Multiple bundled actions are retained when several curated
/// aliases share one canonical repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubEntry {
    /// Display name. A bundled model name is preferred when one is available.
    pub name: String,
    /// Repository identity as supplied by the first live or bundled source.
    pub repo_id: String,
    pub provenance: Vec<EntryProvenance>,
    pub bundled_targets: Vec<BundledTarget>,
    pub live_target: Option<LiveTarget>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
}

/// One curated action and the recommendation evidence for that exact row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundledTarget {
    pub alias: String,
    pub name: String,
    pub repo: RepoRef,
    pub file: Option<String>,
    pub install_args: Vec<String>,
    pub evidence: String,
    pub fit: FitSummary,
}

/// One live repository at the immutable revision returned by the Hub.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveTarget {
    pub repo: RepoRef,
    pub evidence: String,
    pub fit: FitSummary,
    /// Populated by the explicit repository-detail or probe flow.
    pub variants: Option<Vec<GgufVariant>>,
}

/// Display projection of the existing recommendation fit and evidence source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FitSummary {
    pub verdict: String,
    pub is_estimate: bool,
    pub counted_bytes: Option<u64>,
    pub counted_source: String,
    pub mapped_bytes: Option<u64>,
    pub notes: Vec<String>,
}

/// One composed page plus entry-level validation diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubPage {
    pub entries: Vec<HubEntry>,
    pub rejected: Vec<RejectedHubEntry>,
    pub fetched_at: SystemTime,
    /// True when the validated live payload came from the store cache.
    pub from_cache: bool,
}

/// A failed live request or an invalid whole response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HubError {
    Network(String),
    RateLimited {
        retry_after_secs: Option<u64>,
    },
    InvalidRequest {
        field: &'static str,
        rule: &'static str,
    },
    InvalidResponse {
        entry: String,
        rule: &'static str,
    },
}

impl std::fmt::Display for HubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Network(error) => write!(f, "hub request failed: {error}"),
            Self::RateLimited { retry_after_secs } => match retry_after_secs {
                Some(seconds) => write!(f, "hub rate limit reached; retry after {seconds}s"),
                None => write!(f, "hub rate limit reached"),
            },
            Self::InvalidRequest { field, rule } => {
                write!(
                    f,
                    "hub request field {field:?} failed validation rule {rule}"
                )
            }
            Self::InvalidResponse { entry, rule } => {
                write!(
                    f,
                    "hub response entry {entry:?} failed validation rule {rule}"
                )
            }
        }
    }
}

impl std::error::Error for HubError {}

/// Fetches validated Hub list responses and composes them with local rows.
///
/// The bundled catalog and fit inputs are snapshots for this service instance.
/// A new `HubClient` can be created when the catalog or machine configuration
/// changes. The HF client is borrowed so other catalog services can reuse the
/// same authenticated transport.
pub struct HubClient<'a> {
    client: &'a Client,
    catalog: Catalog,
    machine: Machine,
    context: u32,
    slots: model_io::ExpertCacheSlots,
    store: Option<Store>,
}

impl<'a> HubClient<'a> {
    pub fn new(
        client: &'a Client,
        catalog: Catalog,
        machine: Machine,
        context: u32,
        slots: model_io::ExpertCacheSlots,
    ) -> Self {
        Self {
            client,
            catalog,
            machine,
            context,
            slots,
            store: Store::default_store().ok(),
        }
    }

    /// Uses an explicit model-store root for the validated hub-response cache.
    /// The default constructor resolves the configured store automatically.
    pub fn with_store(mut self, store: Store) -> Self {
        self.store = Some(store);
        self
    }

    /// Fetches one explicit live search and merges matching bundled rows.
    pub fn search(&self, query: &HubQuery, refresh: Refresh) -> Result<HubPage, HubError> {
        if query.text.trim().is_empty() {
            return Err(HubError::InvalidRequest {
                field: "text",
                rule: "non_empty_search",
            });
        }
        if query.limit == 0 || query.limit > MAX_HUB_PAGE_ENTRIES {
            return Err(HubError::InvalidRequest {
                field: "limit",
                rule: "within_page_limit",
            });
        }

        let url = models_url(Some(&query.text), query.sort, query.limit)?;
        let fetched = self.fetch_list(&url, CacheKind::Search, refresh)?;
        let local_matches = self.catalog.find(&query.text);
        Ok(self.compose(fetched, local_matches))
    }

    /// Fetches the current trending page and appends bundled recommendations.
    pub fn trending(&self, refresh: Refresh) -> Result<HubPage, HubError> {
        let url = models_url(None, HubSort::Trending, DEFAULT_TRENDING_LIMIT)?;
        let fetched = self.fetch_list(&url, CacheKind::Trending, refresh)?;
        let local_recommendations = self.catalog.entries().collect();
        Ok(self.compose(fetched, local_recommendations))
    }

    fn fetch_list(
        &self,
        url: &str,
        kind: CacheKind,
        refresh: Refresh,
    ) -> Result<ValidatedList, HubError> {
        let validator = HubMetadataValidator::default();
        if refresh == Refresh::AllowCache {
            if let Some(cached) = self.read_cache(url, kind, &validator) {
                return Ok(cached);
            }
        }

        let body = self
            .client
            .get_bounded(url, validator.limits().max_response_bytes)
            .map_err(map_transport_error)?;
        let report = validator
            .validate_search_payload(&body)
            .map_err(map_validation_error)?;
        let fetched_at = truncate_to_seconds(SystemTime::now());
        if report.rejected.is_empty() {
            self.write_cache(
                url,
                &body,
                fetched_at,
                validator.limits().max_response_bytes,
            );
        }
        Ok(ValidatedList {
            entries: report.valid,
            rejected: report.rejected,
            fetched_at,
            from_cache: false,
        })
    }

    fn read_cache(
        &self,
        url: &str,
        kind: CacheKind,
        validator: &HubMetadataValidator,
    ) -> Option<ValidatedList> {
        let path = self.cache_path(url)?;
        let max_envelope_bytes = validator
            .limits()
            .max_response_bytes
            .checked_add(CACHE_ENVELOPE_OVERHEAD_BYTES)?;
        let Some(bytes) = read_bounded_cache(&path, max_envelope_bytes) else {
            let _ = fs::remove_file(&path);
            return None;
        };
        let Ok(envelope) = serde_json::from_slice::<CacheEnvelope>(&bytes) else {
            let _ = fs::remove_file(&path);
            return None;
        };
        let Some(fetched_at) = unix_seconds_to_system_time(envelope.fetched_at) else {
            let _ = fs::remove_file(&path);
            return None;
        };
        let now = SystemTime::now();
        let Ok(age) = now.duration_since(fetched_at) else {
            let _ = fs::remove_file(&path);
            return None;
        };
        if age > kind.freshness().saturating_mul(CACHE_RETENTION_MULTIPLIER) {
            let _ = fs::remove_file(&path);
            return None;
        }

        let Ok(payload) = serde_json::to_vec(&envelope.payload) else {
            let _ = fs::remove_file(&path);
            return None;
        };
        let Ok(report) = validator.validate_search_payload(&payload) else {
            let _ = fs::remove_file(&path);
            return None;
        };
        // A cache file is written only for a fully accepted page. If it was
        // edited later, discard it rather than turning its bad rows into an
        // apparently fresh partial result.
        if !report.rejected.is_empty() {
            let _ = fs::remove_file(&path);
            return None;
        }
        if age > kind.freshness() {
            return None;
        }

        Some(ValidatedList {
            entries: report.valid,
            rejected: report.rejected,
            fetched_at,
            from_cache: true,
        })
    }

    fn write_cache(&self, url: &str, body: &[u8], fetched_at: SystemTime, max_bytes: usize) {
        let Some(path) = self.cache_path(url) else {
            return;
        };
        let Ok(payload) = serde_json::from_slice::<Value>(body) else {
            return;
        };
        let Some(fetched_at) = system_time_to_unix_seconds(fetched_at) else {
            return;
        };
        let envelope = CacheEnvelope {
            fetched_at,
            payload,
        };
        let Ok(bytes) = serde_json::to_vec(&envelope) else {
            return;
        };
        let Some(max_envelope_bytes) = max_bytes.checked_add(CACHE_ENVELOPE_OVERHEAD_BYTES) else {
            return;
        };
        if bytes.len() > max_envelope_bytes {
            return;
        }

        let Some(parent) = path.parent() else {
            return;
        };
        if fs::create_dir_all(parent).is_err() {
            return;
        }
        let sequence = CACHE_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("hub-cache");
        let temporary = parent.join(format!(
            ".{file_name}.tmp-{}-{sequence}",
            std::process::id()
        ));
        let Ok(mut file) = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        else {
            return;
        };
        if file.write_all(&bytes).is_err() || file.sync_all().is_err() {
            drop(file);
            let _ = fs::remove_file(&temporary);
            return;
        }
        drop(file);
        if fs::rename(&temporary, &path).is_err() {
            let _ = fs::remove_file(&temporary);
        }
    }

    fn cache_path(&self, url: &str) -> Option<PathBuf> {
        let store = self.store.as_ref()?;
        Some(
            store
                .root()
                .join("hub-cache")
                .join(format!("{}.json", cache_key(url, self.client.token()))),
        )
    }

    fn compose(&self, fetched: ValidatedList, bundled_rows: Vec<&CatalogEntry>) -> HubPage {
        let mut entries = Vec::<HubEntry>::new();
        let mut by_repo = HashMap::<String, usize>::new();

        let mut rejected = fetched.rejected;
        for live in fetched.entries {
            let key = canonical_repo_id(&live.repo_id);
            if let Some(index) = by_repo.get(&key).copied() {
                let existing = &mut entries[index];
                let existing_revision = existing
                    .live_target
                    .as_ref()
                    .map(|target| target.repo.revision.as_str());
                if existing_revision != Some(live.revision.as_str()) {
                    rejected.push(RejectedHubEntry {
                        entry: live.repo_id,
                        rule: "one_immutable_revision_per_repository",
                    });
                    continue;
                }
                existing.downloads = max_option(existing.downloads, live.downloads);
                existing.likes = max_option(existing.likes, live.likes);
                continue;
            }

            let index = entries.len();
            by_repo.insert(key, index);
            entries.push(HubEntry {
                name: live.repo_id.clone(),
                repo_id: live.repo_id.clone(),
                provenance: vec![EntryProvenance::LiveHub],
                bundled_targets: Vec::new(),
                live_target: Some(LiveTarget {
                    repo: RepoRef::new(&live.repo_id, &live.revision),
                    evidence: Evidence::Discovered.as_str().to_string(),
                    fit: unknown_live_fit(),
                    variants: None,
                }),
                downloads: live.downloads,
                likes: live.likes,
            });
        }

        let mut recommendations =
            recommend_catalog(&bundled_rows, &self.machine, self.context, self.slots);
        for recommendation in recommendations.drain(..) {
            let Origin::Catalog(alias) = &recommendation.origin else {
                continue;
            };
            let Some(entry) = self.catalog.get(alias) else {
                continue;
            };
            let target = bundled_target(entry, &recommendation);
            let key = canonical_repo_id(&entry.source.repo);
            let index = match by_repo.get(&key).copied() {
                Some(index) => index,
                None => {
                    let index = entries.len();
                    by_repo.insert(key, index);
                    entries.push(HubEntry {
                        name: target.name.clone(),
                        repo_id: entry.source.repo.clone(),
                        provenance: vec![EntryProvenance::Bundled],
                        bundled_targets: Vec::new(),
                        live_target: None,
                        downloads: None,
                        likes: None,
                    });
                    index
                }
            };
            let row = &mut entries[index];
            if row.bundled_targets.is_empty() && row.live_target.is_some() {
                row.name = target.name.clone();
            }
            row.bundled_targets.push(target);
        }

        for entry in &mut entries {
            entry.provenance.clear();
            if !entry.bundled_targets.is_empty() {
                entry.provenance.push(EntryProvenance::Bundled);
            }
            if entry.live_target.is_some() {
                entry.provenance.push(EntryProvenance::LiveHub);
            }
        }

        HubPage {
            entries,
            rejected,
            fetched_at: fetched.fetched_at,
            from_cache: fetched.from_cache,
        }
    }
}

fn cache_key(url: &str, token: Option<&str>) -> String {
    let token_fingerprint = token.map(|token| {
        let mut fingerprint = Sha256::new();
        fingerprint.update(b"turbospark-hub-token-v1\0");
        fingerprint.update(token.as_bytes());
        fingerprint.finalize()
    });

    let mut key = Sha256::new();
    key.update(b"turbospark-hub-cache-v1\0");
    key.update(url.as_bytes());
    key.update([0]);
    if let Some(fingerprint) = token_fingerprint {
        key.update(b"authenticated\0");
        key.update(fingerprint);
    } else {
        key.update(b"anonymous\0");
    }
    let digest = key.finalize();
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

fn read_bounded_cache(path: &Path, max_bytes: usize) -> Option<Vec<u8>> {
    let file = std::fs::File::open(path).ok()?;
    let read_limit = u64::try_from(max_bytes).ok()?.checked_add(1)?;
    let mut bytes = Vec::with_capacity(max_bytes.min(64 * 1024));
    file.take(read_limit).read_to_end(&mut bytes).ok()?;
    (bytes.len() <= max_bytes).then_some(bytes)
}

fn system_time_to_unix_seconds(time: SystemTime) -> Option<u64> {
    time.duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_secs())
}

fn unix_seconds_to_system_time(seconds: u64) -> Option<SystemTime> {
    UNIX_EPOCH.checked_add(Duration::from_secs(seconds))
}

fn truncate_to_seconds(time: SystemTime) -> SystemTime {
    system_time_to_unix_seconds(time)
        .and_then(unix_seconds_to_system_time)
        .unwrap_or(time)
}

fn models_url(search: Option<&str>, sort: HubSort, limit: u32) -> Result<String, HubError> {
    let base = hf_endpoint();
    let mut url = reqwest::Url::parse(&format!("{}/api/models", base.trim_end_matches('/')))
        .map_err(|error| HubError::Network(format!("invalid HF endpoint {base:?}: {error}")))?;
    {
        let mut query = url.query_pairs_mut();
        if let Some(search) = search {
            query.append_pair("search", search);
        }
        query
            .append_pair("sort", sort.api_value())
            .append_pair("direction", "-1")
            .append_pair("limit", &limit.to_string());
    }
    Ok(url.into())
}

fn canonical_repo_id(repo_id: &str) -> String {
    repo_id.to_ascii_lowercase()
}

fn max_option(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn bundled_target(entry: &CatalogEntry, recommendation: &Recommendation) -> BundledTarget {
    BundledTarget {
        alias: entry.alias.clone(),
        name: entry.name.clone(),
        repo: RepoRef::new(&entry.source.repo, &entry.source.revision),
        file: entry.source.file.clone(),
        install_args: recommendation.origin.install_args(),
        evidence: recommendation.evidence.as_str().to_string(),
        fit: fit_summary(&recommendation.fit, &recommendation.notes),
    }
}

fn fit_summary(fit: &Fit, notes: &[String]) -> FitSummary {
    let (counted_source, is_estimate) = match fit.counted_source {
        CountedSource::Measured => ("measured", false),
        CountedSource::Estimated => ("estimated", true),
        CountedSource::Unknown => ("unknown", false),
    };
    FitSummary {
        verdict: fit.verdict.as_str().to_string(),
        is_estimate,
        counted_bytes: (fit.counted_source != CountedSource::Unknown).then_some(fit.counted),
        counted_source: counted_source.to_string(),
        mapped_bytes: Some(fit.mapped),
        notes: notes.to_vec(),
    }
}

fn unknown_live_fit() -> FitSummary {
    FitSummary {
        verdict: FitVerdict::Unknown.as_str().to_string(),
        is_estimate: false,
        counted_bytes: None,
        counted_source: "unknown".to_string(),
        mapped_bytes: None,
        notes: vec![
            "Search metadata has no authoritative artifact size or checkpoint shape.".to_string(),
        ],
    }
}

fn map_transport_error(error: String) -> HubError {
    if error.contains("HTTP 429") {
        HubError::RateLimited {
            retry_after_secs: None,
        }
    } else if error.contains("response exceeds the") {
        HubError::InvalidResponse {
            entry: "<response>".to_string(),
            rule: "response_size_within_limit",
        }
    } else {
        HubError::Network(error)
    }
}

fn map_validation_error(error: HubValidationError) -> HubError {
    match error {
        HubValidationError::OversizedResponse { .. } => HubError::InvalidResponse {
            entry: "<response>".to_string(),
            rule: "response_size_within_limit",
        },
        HubValidationError::InvalidResponse { entry, rule } => {
            HubError::InvalidResponse { entry, rule }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_encodes_search_as_one_query_value() {
        let url = models_url(Some("org/model & sort=likes"), HubSort::Downloads, 17).unwrap();
        let parsed = reqwest::Url::parse(&url).unwrap();
        let pairs: Vec<_> = parsed.query_pairs().collect();
        assert_eq!(
            pairs.iter().find(|(key, _)| key == "search").unwrap().1,
            "org/model & sort=likes"
        );
        assert_eq!(pairs.iter().filter(|(key, _)| key == "limit").count(), 1);
        assert_eq!(
            pairs.iter().find(|(key, _)| key == "sort").unwrap().1,
            "downloads"
        );
    }

    #[test]
    fn trending_sort_uses_hub_wire_value() {
        let url = models_url(None, HubSort::Trending, DEFAULT_TRENDING_LIMIT).unwrap();
        let parsed = reqwest::Url::parse(&url).unwrap();
        let sort = parsed
            .query_pairs()
            .find(|(key, _)| key == "sort")
            .unwrap()
            .1;
        assert_eq!(sort, "trendingScore");
    }

    #[test]
    fn canonical_repo_identity_ignores_ascii_case_only() {
        assert_eq!(
            canonical_repo_id("Owner/Model-Name"),
            canonical_repo_id("owner/model-name")
        );
        assert_ne!(
            canonical_repo_id("owner/model-name"),
            canonical_repo_id("owner/model_name")
        );
    }

    #[test]
    fn cache_identity_separates_anonymous_and_authenticated_partitions() {
        let url = "https://huggingface.co/api/models?search=owner%2Fmodel";
        let anonymous = cache_key(url, None);
        let credential_one = cache_key(url, Some("hf_fixture_one"));
        let credential_two = cache_key(url, Some("hf_fixture_two"));

        assert_ne!(anonymous, credential_one);
        assert_ne!(credential_one, credential_two);
    }
}
