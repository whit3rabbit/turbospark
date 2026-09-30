//! Explicit Hugging Face search and trending results composed with the
//! bundled catalog.
//!
//! Constructing a [`HubClient`] does not perform I/O. The only network calls
//! in this module occur at [`HubClient::search`] and [`HubClient::trending`].
//! Live list metadata is validated before composition, and result composition
//! never probes a repository or changes a curated target's revision.

use crate::catalog::Catalog;
use crate::entry::CatalogEntry;
use crate::hf::{hf_endpoint, Client, HubRequestError, RepoRef};
use crate::hub_validation::{
    normalize_sha256, valid_relative_path, valid_repo_id, valid_revision, HubFileMetadata,
    HubMetadataValidator, HubSearchEntry, HubValidationError, RejectedHubEntry,
};
use crate::probe::{ProbeReport, Verdict};
use crate::quant::{group_variants, quant_label, QuantLabel, ShardSetIssue, ShardSetStatus};
use crate::recommend::{
    fit as calculate_fit, recommend_catalog, CountedSource, Evidence, Fit, FitVerdict, Machine,
    Origin, Recommendation, Shape,
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
    pub variants: Option<Vec<HubGgufVariant>>,
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

/// A filename-derived GGUF variant enriched with authoritative file sizes and
/// the existing recommendation fit calculation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubGgufVariant {
    /// Immutable repository revision resolved from the repository response.
    pub repo: RepoRef,
    pub label: QuantLabel,
    pub files: Vec<crate::hf::RepoFile>,
    pub total_bytes: Option<u64>,
    pub fit: FitSummary,
    pub installability: VariantInstallability,
}

/// An owner-supplied source group. `role` is opaque to catalog transport.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedSourceGroup {
    pub role: String,
    pub repo: RepoRef,
    pub files: Vec<PinnedSourceFile>,
}

/// An explicit all-or-nothing transfer request from a modality-owned catalog.
/// `owner_id` and each source `role` are opaque values preserved in the receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedArtifactPlan {
    pub owner_id: String,
    pub sources: Vec<PinnedSourceGroup>,
}

/// All verified files from one source group, with the owner's role unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedSourceGroup {
    pub role: String,
    pub repo: RepoRef,
    pub files: Vec<VerifiedSourceFile>,
}

/// A staged file whose byte count and SHA-256 were verified before receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedSourceFile {
    pub path: String,
    pub staged_path: PathBuf,
    pub size: u64,
    pub sha256: String,
}

/// Receipt returned only after every source group has transferred and passed
/// identity, size, and digest checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadReceipt {
    pub owner_id: String,
    pub sources: Vec<VerifiedSourceGroup>,
    /// The unique child staging directory created and owned by this transfer.
    pub staging_root: PathBuf,
}

/// Progress for one explicit pinned-source transfer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubDownloadProgress {
    pub owner_id: String,
    pub completed_files: u32,
    pub total_files: u32,
    pub current_path: Option<String>,
    pub completed_bytes: u64,
    pub total_bytes: u64,
}

/// One owner-pinned path and any exact size or SHA-256 expectations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedSourceFile {
    pub path: String,
    pub expected_size: Option<u64>,
    pub expected_sha256: Option<String>,
}

/// Repository source metadata resolved and checked against an owner source set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSourceIdentity {
    pub repo: RepoRef,
    pub files: Vec<ResolvedSourceFile>,
}

/// One exact path with authority-backed source metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSourceFile {
    pub path: String,
    pub authoritative_size: Option<u64>,
    pub source_sha256: Option<String>,
}

/// Why an owner-pinned source and a resolved live source do not match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceIdentityMismatch {
    Repository,
    Revision,
    FileSet,
    MissingAuthoritativeSize,
    Size,
    MissingPinnedDigest,
    Digest,
    DuplicatePath,
}

/// Why one GGUF file group can or cannot reach the existing install gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VariantInstallability {
    Ready,
    SupportUnverified {
        files: Vec<String>,
    },
    Unsupported {
        files: Vec<String>,
        reason: String,
    },
    UnknownSize {
        files: Vec<String>,
    },
    IncompleteShardSet {
        files: Vec<String>,
        expected_count: usize,
        present_indices: Vec<usize>,
    },
    InconsistentShardSet {
        files: Vec<String>,
        issues: Vec<ShardSetIssue>,
    },
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
    Offline,
    Network(String),
    /// The caller cancelled an explicit transfer. Its owned staging root is
    /// removed before this value is returned.
    Cancelled,
    /// A staged body failed a source size or digest invariant.
    TransferIntegrity {
        entry: String,
        rule: &'static str,
    },
    /// Local staging could not be created, written, synced, or removed.
    Staging(String),
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
            Self::Offline => write!(f, "hub is offline; check the network and retry"),
            Self::Network(error) => write!(f, "hub request failed: {error}"),
            Self::Cancelled => write!(f, "hub transfer cancelled by the caller"),
            Self::TransferIntegrity { entry, rule } => {
                write!(
                    f,
                    "hub transfer entry {entry:?} failed integrity rule {rule}"
                )
            }
            Self::Staging(error) => write!(f, "hub transfer staging failed: {error}"),
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

    /// Lists GGUF variants for an explicit repository detail action.
    ///
    /// Missing sibling sizes are resolved with authoritative `HEAD` content
    /// lengths. A fit remains unknown until a matching explicit probe supplies
    /// checkpoint shape; repository size alone is not a fit estimate.
    pub fn repo_variants(&self, repo: &RepoRef) -> Result<Vec<HubGgufVariant>, HubError> {
        self.enrich_repo_variants(repo, None)
    }

    /// Lists variants and applies one explicit probe report only to the exact
    /// complete file group it describes.
    pub fn repo_variants_with_probe(
        &self,
        repo: &RepoRef,
        probe: &ProbeReport,
    ) -> Result<Vec<HubGgufVariant>, HubError> {
        self.enrich_repo_variants(repo, Some(probe))
    }

    /// Resolves one owner-pinned source group at its immutable revision.
    ///
    /// This is an explicit source-resolution request. It reads the repository
    /// listing once, resolves missing listed sizes with `HEAD`, and returns
    /// only after the complete owner-pinned set matches validated metadata.
    pub fn resolve_source_identity(
        &self,
        expected: &PinnedSourceGroup,
    ) -> Result<ResolvedSourceIdentity, HubError> {
        let endpoint = hf_endpoint();
        self.resolve_source_identity_at(expected, &endpoint)
    }

    pub(crate) fn resolve_source_identity_at(
        &self,
        expected: &PinnedSourceGroup,
        endpoint: &str,
    ) -> Result<ResolvedSourceIdentity, HubError> {
        validate_pinned_source_group(expected)?;

        let validator = HubMetadataValidator::default();
        let metadata_url = source_repository_metadata_url(endpoint, &expected.repo)?;
        let body = self
            .client
            .get_bounded_for_hub(&metadata_url, validator.limits().max_response_bytes)
            .map_err(|error| {
                map_hub_request_error(error, &metadata_url, self.client.has_token())
            })?;
        let report = validator
            .validate_source_repository_payload(&body)
            .map_err(map_validation_error)?;

        let repo_id = canonical_repo_id(&expected.repo.repo);
        if canonical_repo_id(&report.valid.repo_id) != repo_id {
            return Err(HubError::InvalidResponse {
                entry: report.valid.repo_id,
                rule: "repository_id_matches_owner_source",
            });
        }
        let revision = expected.repo.revision.to_ascii_lowercase();
        if report.valid.revision != revision {
            return Err(HubError::InvalidResponse {
                entry: format!("{}@{}", report.valid.repo_id, report.valid.revision),
                rule: "immutable_revision_matches_owner_source",
            });
        }

        let mut files = Vec::with_capacity(expected.files.len());
        for pinned in &expected.files {
            if report
                .rejected
                .iter()
                .any(|rejected| rejected.entry == pinned.path)
            {
                return Err(HubError::InvalidResponse {
                    entry: pinned.path.clone(),
                    rule: "owner_source_file_metadata_valid",
                });
            }
            let Some(source) = report
                .valid
                .files
                .iter()
                .find(|file| file.path == pinned.path)
            else {
                return Err(HubError::InvalidResponse {
                    entry: pinned.path.clone(),
                    rule: "owner_source_file_present",
                });
            };

            let authoritative_size = match source.size_bytes {
                Some(size) => Some(size),
                None => {
                    let url = source_file_url_at(endpoint, &expected.repo, &pinned.path)?;
                    self.client
                        .content_length(&url)
                        .map_err(HubError::Network)?
                }
            };
            let Some(authoritative_size) = authoritative_size else {
                return Err(HubError::InvalidResponse {
                    entry: pinned.path.clone(),
                    rule: "authoritative_source_size_available",
                });
            };
            if authoritative_size > validator.limits().max_file_size_bytes {
                return Err(HubError::InvalidResponse {
                    entry: pinned.path.clone(),
                    rule: "file_size_within_limit",
                });
            }

            files.push(ResolvedSourceFile {
                path: pinned.path.clone(),
                authoritative_size: Some(authoritative_size),
                source_sha256: source.sha256.clone(),
            });
        }

        let resolved = ResolvedSourceIdentity {
            repo: RepoRef::new(repo_id, revision),
            files,
        };
        matches_exact_source(expected, &resolved).map_err(|mismatch| {
            HubError::InvalidResponse {
                entry: expected.repo.to_string(),
                rule: source_mismatch_rule(mismatch),
            }
        })?;

        let checked_files: Vec<_> = resolved
            .files
            .iter()
            .map(|file| HubFileMetadata {
                path: file.path.clone(),
                size_bytes: file.authoritative_size,
            })
            .collect();
        validator
            .checked_file_total(&checked_files)
            .map_err(map_validation_error)?;

        Ok(resolved)
    }

    /// Transfers an explicit owner-pinned plan into a unique child directory
    /// below `staging_parent`. The parent remains caller-owned; only the child
    /// created by this invocation is eligible for rollback cleanup.
    pub fn download_pinned_artifacts(
        &self,
        plan: &PinnedArtifactPlan,
        staging_parent: &Path,
        progress: &mut dyn FnMut(HubDownloadProgress),
        cancel: &crate::install::CancelFlag,
    ) -> Result<DownloadReceipt, HubError> {
        crate::hub_transfer::download_pinned_artifacts(self, plan, staging_parent, progress, cancel)
    }

    pub(crate) fn client(&self) -> &Client {
        self.client
    }

    fn enrich_repo_variants(
        &self,
        repo: &RepoRef,
        probe: Option<&ProbeReport>,
    ) -> Result<Vec<HubGgufVariant>, HubError> {
        if !valid_repo_id(&repo.repo) {
            return Err(HubError::InvalidRequest {
                field: "repo",
                rule: "repository_id_shape",
            });
        }
        let requested_revision_is_commit = valid_revision(&repo.revision);
        if repo.revision != "main" && !requested_revision_is_commit {
            return Err(HubError::InvalidRequest {
                field: "revision",
                rule: "main_or_immutable_commit",
            });
        }

        let validator = HubMetadataValidator::default();
        let metadata_url = repo.api_url_with_sizes();
        let body = self
            .client
            .get_bounded_for_hub(&metadata_url, validator.limits().max_response_bytes)
            .map_err(|error| {
                map_hub_request_error(error, &metadata_url, self.client.has_token())
            })?;
        let report = validator
            .validate_repository_payload(&body)
            .map_err(map_validation_error)?;
        if canonical_repo_id(&report.valid.repo_id) != canonical_repo_id(&repo.repo) {
            return Err(HubError::InvalidResponse {
                entry: report.valid.repo_id,
                rule: "repository_id_matches_request",
            });
        }
        if requested_revision_is_commit
            && !report.valid.revision.eq_ignore_ascii_case(&repo.revision)
        {
            return Err(HubError::InvalidResponse {
                entry: format!("{}@{}", repo.repo, report.valid.revision),
                rule: "revision_matches_request",
            });
        }
        let resolved_repo = RepoRef::new(&report.valid.repo_id, &report.valid.revision);
        let repo_files: Vec<_> = report
            .valid
            .files
            .into_iter()
            .map(|file| crate::hf::RepoFile {
                name: file.path,
                size: file.size_bytes,
            })
            .collect();

        let mut variants = Vec::new();
        for group in group_variants(&repo_files) {
            variants.push(self.enrich_variant(&resolved_repo, group, &validator, probe));
        }
        Ok(variants)
    }

    fn enrich_variant(
        &self,
        repo: &RepoRef,
        group: crate::quant::VariantFiles,
        validator: &HubMetadataValidator,
        probe: Option<&ProbeReport>,
    ) -> HubGgufVariant {
        let matching_probe =
            probe.filter(|candidate| probe_matches_variant(repo, &group, candidate));
        let group_files = file_names(&group.files);
        let mut files = group.files;
        for file in &mut files {
            if file.size.is_none() {
                file.size = self
                    .client
                    .content_length(&repo.file_url(&file.name))
                    .ok()
                    .flatten();
            }
        }

        let complete = matches!(
            &group.shard_set,
            ShardSetStatus::SingleFile | ShardSetStatus::Complete { .. }
        );
        let total_result = if complete {
            checked_variant_total(&files, validator)
        } else {
            Ok(None)
        };
        let total_bytes = total_result.as_ref().ok().copied().flatten();
        let fit = matching_probe
            .zip(total_bytes)
            .filter(|(candidate, _)| has_sufficient_fit_shape(candidate))
            .map(|(candidate, bytes)| variant_fit_summary(candidate, bytes, self))
            .unwrap_or_else(unknown_variant_fit);

        let installability = match group.shard_set {
            ShardSetStatus::Incomplete {
                expected_count,
                present_indices,
            } => VariantInstallability::IncompleteShardSet {
                files: group_files,
                expected_count,
                present_indices,
            },
            ShardSetStatus::Inconsistent { issues } => {
                VariantInstallability::InconsistentShardSet {
                    files: group_files,
                    issues,
                }
            }
            ShardSetStatus::SingleFile | ShardSetStatus::Complete { .. } => match total_result {
                Ok(None) => VariantInstallability::UnknownSize {
                    files: missing_file_sizes(&files),
                },
                Err(error) => size_failure_installability(error),
                Ok(Some(_)) => match matching_probe.map(|candidate| &candidate.verdict) {
                    None => VariantInstallability::SupportUnverified { files: group_files },
                    Some(Verdict::Runnable) => VariantInstallability::Ready,
                    Some(Verdict::Refused(reason)) => VariantInstallability::Unsupported {
                        files: group_files,
                        reason: reason.clone(),
                    },
                },
            },
        };

        HubGgufVariant {
            repo: repo.clone(),
            label: group.label,
            files,
            total_bytes,
            fit,
            installability,
        }
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
            .get_bounded_for_hub(url, validator.limits().max_response_bytes)
            .map_err(|error| map_hub_request_error(error, url, self.client.has_token()))?;
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

/// Checks that a fully resolved live source exactly satisfies an owner pin.
///
/// Path identity is case-sensitive and set-based. Repository IDs and commit
/// hashes are canonicalized according to the Hub's case-insensitive IDs and
/// hexadecimal revision representation.
pub fn matches_exact_source(
    expected: &PinnedSourceGroup,
    candidate: &ResolvedSourceIdentity,
) -> Result<(), SourceIdentityMismatch> {
    if !valid_repo_id(&expected.repo.repo) || !valid_repo_id(&candidate.repo.repo) {
        return Err(SourceIdentityMismatch::Repository);
    }
    if canonical_repo_id(&expected.repo.repo) != canonical_repo_id(&candidate.repo.repo) {
        return Err(SourceIdentityMismatch::Repository);
    }
    if !valid_revision(&expected.repo.revision)
        || !valid_revision(&candidate.repo.revision)
        || !expected
            .repo
            .revision
            .eq_ignore_ascii_case(&candidate.repo.revision)
    {
        return Err(SourceIdentityMismatch::Revision);
    }
    if expected.files.is_empty() {
        return Err(SourceIdentityMismatch::FileSet);
    }

    let mut expected_by_path = HashMap::with_capacity(expected.files.len());
    for file in &expected.files {
        if !valid_relative_path(&file.path) {
            return Err(SourceIdentityMismatch::FileSet);
        }
        if expected_by_path.insert(file.path.as_str(), file).is_some() {
            return Err(SourceIdentityMismatch::DuplicatePath);
        }
    }

    let mut candidate_by_path = HashMap::with_capacity(candidate.files.len());
    for file in &candidate.files {
        if !valid_relative_path(&file.path) {
            return Err(SourceIdentityMismatch::FileSet);
        }
        if candidate_by_path.insert(file.path.as_str(), file).is_some() {
            return Err(SourceIdentityMismatch::DuplicatePath);
        }
    }
    if expected_by_path.len() != candidate_by_path.len()
        || expected_by_path
            .keys()
            .any(|path| !candidate_by_path.contains_key(path))
    {
        return Err(SourceIdentityMismatch::FileSet);
    }

    let validator = HubMetadataValidator::default();
    let mut checked_files = Vec::with_capacity(candidate.files.len());
    for file in &candidate.files {
        let Some(size) = file.authoritative_size else {
            return Err(SourceIdentityMismatch::MissingAuthoritativeSize);
        };
        let source_sha256 = file
            .source_sha256
            .as_deref()
            .map(normalize_sha256)
            .transpose()
            .map_err(|_| SourceIdentityMismatch::Digest)?;
        checked_files.push(HubFileMetadata {
            path: file.path.clone(),
            size_bytes: Some(size),
        });

        let Some(pinned) = expected_by_path.get(file.path.as_str()).copied() else {
            return Err(SourceIdentityMismatch::FileSet);
        };
        if pinned
            .expected_size
            .is_some_and(|expected_size| expected_size != size)
        {
            return Err(SourceIdentityMismatch::Size);
        }
        if let Some(expected_sha256) = pinned.expected_sha256.as_deref() {
            let expected_sha256 =
                normalize_sha256(expected_sha256).map_err(|_| SourceIdentityMismatch::Digest)?;
            let Some(source_sha256) = source_sha256 else {
                return Err(SourceIdentityMismatch::MissingPinnedDigest);
            };
            if expected_sha256 != source_sha256 {
                return Err(SourceIdentityMismatch::Digest);
            }
        }
    }
    validator
        .checked_file_total(&checked_files)
        .map_err(|_| SourceIdentityMismatch::Size)?;
    Ok(())
}

pub(crate) fn validate_pinned_source_group(expected: &PinnedSourceGroup) -> Result<(), HubError> {
    let invalid = |field, rule| HubError::InvalidRequest { field, rule };
    if !valid_repo_id(&expected.repo.repo) {
        return Err(invalid("repo", "repository_id_shape"));
    }
    if !valid_revision(&expected.repo.revision) {
        return Err(invalid("revision", "immutable_commit_revision"));
    }
    let validator = HubMetadataValidator::default();
    if expected.files.is_empty() {
        return Err(invalid("files", "non_empty_source_set"));
    }
    if expected.files.len() > validator.limits().max_repo_files {
        return Err(invalid("files", "file_count_within_limit"));
    }

    let mut seen = std::collections::HashSet::with_capacity(expected.files.len());
    for file in &expected.files {
        if !valid_relative_path(&file.path) {
            return Err(invalid("path", "safe_relative_path"));
        }
        if !seen.insert(file.path.as_str()) {
            return Err(invalid("path", "unique_file_path"));
        }
        if file
            .expected_size
            .is_some_and(|size| size > validator.limits().max_file_size_bytes)
        {
            return Err(invalid("expected_size", "file_size_within_limit"));
        }
        if file
            .expected_sha256
            .as_deref()
            .is_some_and(|digest| normalize_sha256(digest).is_err())
        {
            return Err(invalid("expected_sha256", "sha256_digest_shape"));
        }
    }
    Ok(())
}

pub(crate) fn source_repository_metadata_url(
    endpoint: &str,
    repo: &RepoRef,
) -> Result<String, HubError> {
    let mut url = reqwest::Url::parse(&format!("{}/", endpoint.trim_end_matches('/')))
        .map_err(|error| HubError::Network(format!("invalid HF endpoint {endpoint:?}: {error}")))?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| HubError::InvalidRequest {
                field: "repo",
                rule: "repository_id_shape",
            })?;
        segments.push("api").push("models");
        for segment in repo.repo.split('/') {
            segments.push(segment);
        }
        segments.push("revision").push(&repo.revision);
    }
    url.query_pairs_mut().append_pair("blobs", "true");
    Ok(url.into())
}

pub(crate) fn source_file_url_at(
    endpoint: &str,
    repo: &RepoRef,
    path: &str,
) -> Result<String, HubError> {
    let mut url = reqwest::Url::parse(&format!("{}/", endpoint.trim_end_matches('/')))
        .map_err(|error| HubError::Network(format!("invalid HF endpoint {endpoint:?}: {error}")))?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| HubError::InvalidRequest {
                field: "path",
                rule: "safe_relative_path",
            })?;
        for segment in repo.repo.split('/') {
            segments.push(segment);
        }
        segments.push("resolve").push(&repo.revision);
        for segment in path.split('/') {
            segments.push(segment);
        }
    }
    Ok(url.into())
}

pub(crate) fn source_mismatch_rule(mismatch: SourceIdentityMismatch) -> &'static str {
    match mismatch {
        SourceIdentityMismatch::Repository => "repository_id_matches_owner_source",
        SourceIdentityMismatch::Revision => "immutable_revision_matches_owner_source",
        SourceIdentityMismatch::FileSet => "complete_file_set_matches_owner_source",
        SourceIdentityMismatch::MissingAuthoritativeSize => "authoritative_source_size_available",
        SourceIdentityMismatch::Size => "source_size_matches_owner_pin",
        SourceIdentityMismatch::MissingPinnedDigest => "pinned_source_digest_available",
        SourceIdentityMismatch::Digest => "source_digest_matches_owner_pin",
        SourceIdentityMismatch::DuplicatePath => "unique_file_path",
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

fn unknown_variant_fit() -> FitSummary {
    FitSummary {
        verdict: FitVerdict::Unknown.as_str().to_string(),
        is_estimate: false,
        counted_bytes: None,
        counted_source: "unknown".to_string(),
        mapped_bytes: None,
        notes: vec![
            "Variant fit stays unknown until an explicit probe provides checkpoint shape."
                .to_string(),
        ],
    }
}

fn file_names(files: &[crate::hf::RepoFile]) -> Vec<String> {
    files.iter().map(|file| file.name.clone()).collect()
}

fn missing_file_sizes(files: &[crate::hf::RepoFile]) -> Vec<String> {
    files
        .iter()
        .filter(|file| file.size.is_none())
        .map(|file| file.name.clone())
        .collect()
}

fn checked_variant_total(
    files: &[crate::hf::RepoFile],
    validator: &HubMetadataValidator,
) -> Result<Option<u64>, HubValidationError> {
    let metadata: Vec<_> = files
        .iter()
        .map(|file| HubFileMetadata {
            path: file.name.clone(),
            size_bytes: file.size,
        })
        .collect();
    validator.checked_file_total(&metadata)
}

fn probe_matches_variant(
    repo: &RepoRef,
    group: &crate::quant::VariantFiles,
    probe: &ProbeReport,
) -> bool {
    if probe.kind != crate::entry::SourceKind::Gguf
        || canonical_repo_id(&probe.repo.repo) != canonical_repo_id(&repo.repo)
        || probe.repo.revision != repo.revision
        || !matches!(
            group.shard_set,
            ShardSetStatus::SingleFile | ShardSetStatus::Complete { .. }
        )
    {
        return false;
    }
    let Some(probed_file) = probe.file.as_deref() else {
        return false;
    };
    group.files.iter().any(|file| file.name == probed_file)
        && quant_label(probed_file).as_ref() == Some(&group.label)
}

fn has_sufficient_fit_shape(probe: &ProbeReport) -> bool {
    probe
        .arch
        .as_ref()
        .is_some_and(|arch| arch.num_experts <= 0 || probe.expert_stride.is_some())
}

fn variant_fit_summary(probe: &ProbeReport, total_bytes: u64, hub: &HubClient<'_>) -> FitSummary {
    let shape = Shape {
        install_bytes: total_bytes,
        expert_stride: probe.expert_stride,
        arch: probe.arch.clone(),
        measured_counted: None,
    };
    let mut fit = calculate_fit(
        &shape,
        hub.machine.physical_bytes,
        hub.context,
        hub.slots,
        hub.machine.load_guard,
    );
    if matches!(probe.verdict, Verdict::Refused(_)) {
        fit.verdict = FitVerdict::Refused;
    }
    fit_summary(&fit, &probe.warnings)
}

fn size_failure_installability(error: HubValidationError) -> VariantInstallability {
    match error {
        HubValidationError::InvalidResponse { entry, rule } => VariantInstallability::Unsupported {
            files: vec![entry],
            reason: format!("file size validation failed: {rule}"),
        },
        HubValidationError::OversizedResponse {
            actual_bytes,
            max_bytes,
        } => VariantInstallability::Unsupported {
            files: Vec::new(),
            reason: format!("file metadata response size {actual_bytes} exceeds {max_bytes}"),
        },
    }
}

pub(crate) fn map_hub_request_error(
    error: HubRequestError,
    url: &str,
    has_token: bool,
) -> HubError {
    match error {
        HubRequestError::Offline => HubError::Offline,
        HubRequestError::Transport(message) => HubError::Network(message),
        HubRequestError::HttpStatus {
            status: 429,
            retry_after_secs,
        } => HubError::RateLimited { retry_after_secs },
        HubRequestError::HttpStatus {
            status: status @ (401 | 403),
            ..
        } if !has_token => HubError::Network(format!(
            "GET {url}: HTTP {status}. This repository is gated and no HF_TOKEN \
             is set; accept its licence on huggingface.co, then export a token."
        )),
        HubRequestError::HttpStatus {
            status: status @ (401 | 403),
            ..
        } => HubError::Network(format!(
            "GET {url}: HTTP {status}. HF_TOKEN is set, so either it lacks access \
             to this repository or its licence has not been accepted."
        )),
        HubRequestError::HttpStatus { status: 404, .. } => HubError::Network(format!(
            "GET {url}: HTTP 404. That file is not in this repository at this \
             revision -- file lists differ per repository, not per model family."
        )),
        HubRequestError::HttpStatus { status, .. } => {
            HubError::Network(format!("GET {url}: HTTP {status}"))
        }
        HubRequestError::ResponseTooLarge => HubError::InvalidResponse {
            entry: "<response>".to_string(),
            rule: "response_size_within_limit",
        },
        HubRequestError::BodyRead(message) => HubError::Network(format!("GET {url}: {message}")),
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

    #[test]
    fn checked_variant_total_rejects_u64_overflow() {
        let validator = HubMetadataValidator::new(crate::hub_validation::HubValidationLimits {
            max_file_size_bytes: u64::MAX,
            max_file_set_size_bytes: u64::MAX,
            ..crate::hub_validation::HubValidationLimits::default()
        });
        let files = [
            crate::hf::RepoFile {
                name: "first-Q4_K_M.gguf".to_string(),
                size: Some(u64::MAX),
            },
            crate::hf::RepoFile {
                name: "second-Q4_K_M.gguf".to_string(),
                size: Some(1),
            },
        ];

        let error = checked_variant_total(&files, &validator).expect_err("overflow is rejected");
        assert_eq!(
            error,
            HubValidationError::InvalidResponse {
                entry: "second-Q4_K_M.gguf".to_string(),
                rule: "checked_size_total",
            }
        );
        assert_eq!(
            size_failure_installability(error),
            VariantInstallability::Unsupported {
                files: vec!["second-Q4_K_M.gguf".to_string()],
                reason: "file size validation failed: checked_size_total".to_string(),
            }
        );
    }
}
