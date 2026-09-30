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
use std::collections::HashMap;
use std::time::SystemTime;

/// Maximum list size accepted by the default metadata validator.
pub const MAX_HUB_PAGE_ENTRIES: u32 = 1_000;
/// Page size for the on-demand trending feed.
pub const DEFAULT_TRENDING_LIMIT: u32 = 30;

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

/// Cache policy reserved for the cache implementation in task 1.3.
///
/// Until that task lands, both variants deliberately issue a fresh request.
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
    /// Always false until the validated-response cache is implemented.
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
        }
    }

    /// Fetches one explicit live search and merges matching bundled rows.
    pub fn search(&self, query: &HubQuery, _refresh: Refresh) -> Result<HubPage, HubError> {
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
        let (live, rejected) = self.fetch_list(&url)?;
        let local_matches = self.catalog.find(&query.text);
        Ok(self.compose(live, rejected, local_matches))
    }

    /// Fetches the current trending page and appends bundled recommendations.
    pub fn trending(&self, _refresh: Refresh) -> Result<HubPage, HubError> {
        let url = models_url(None, HubSort::Trending, DEFAULT_TRENDING_LIMIT)?;
        let (live, rejected) = self.fetch_list(&url)?;
        let local_recommendations = self.catalog.entries().collect();
        Ok(self.compose(live, rejected, local_recommendations))
    }

    fn fetch_list(
        &self,
        url: &str,
    ) -> Result<(Vec<HubSearchEntry>, Vec<RejectedHubEntry>), HubError> {
        let validator = HubMetadataValidator::default();
        let body = self
            .client
            .get_bounded(url, validator.limits().max_response_bytes)
            .map_err(map_transport_error)?;
        let report = validator
            .validate_search_payload(&body)
            .map_err(map_validation_error)?;
        Ok((report.valid, report.rejected))
    }

    fn compose(
        &self,
        live_rows: Vec<HubSearchEntry>,
        mut rejected: Vec<RejectedHubEntry>,
        bundled_rows: Vec<&CatalogEntry>,
    ) -> HubPage {
        let mut entries = Vec::<HubEntry>::new();
        let mut by_repo = HashMap::<String, usize>::new();

        for live in live_rows {
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
            fetched_at: SystemTime::now(),
            from_cache: false,
        }
    }
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
}
