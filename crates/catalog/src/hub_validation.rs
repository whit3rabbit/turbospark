//! Validation for untrusted Hugging Face catalog metadata.
//!
//! This module deliberately owns no HTTP behavior. Callers pass response
//! bytes obtained through the existing [`crate::Client`] and must validate
//! both network responses and cached payloads before exposing the result.

use serde_json::Value;
use std::collections::HashSet;

/// Default maximum response body accepted by the hub metadata boundary.
pub const DEFAULT_MAX_HUB_RESPONSE_BYTES: usize = 8 * 1024 * 1024;
/// Default maximum number of search or trending entries in one response.
pub const DEFAULT_MAX_HUB_SEARCH_ENTRIES: usize = 1_000;
/// Default maximum number of files listed for one repository.
pub const DEFAULT_MAX_HUB_REPO_FILES: usize = 10_000;
/// Default maximum size of one declared remote file (1 TiB).
pub const DEFAULT_MAX_HUB_FILE_SIZE_BYTES: u64 = 1 << 40;
/// Default maximum checked total for one selected file set (1 TiB).
pub const DEFAULT_MAX_HUB_FILE_SET_SIZE_BYTES: u64 = 1 << 40;
const MAX_HUB_PATH_BYTES: usize = 1_024;

/// Configurable ceilings for the catalog metadata trust boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HubValidationLimits {
    pub max_response_bytes: usize,
    pub max_search_entries: usize,
    pub max_repo_files: usize,
    pub max_file_size_bytes: u64,
    pub max_file_set_size_bytes: u64,
}

impl Default for HubValidationLimits {
    fn default() -> Self {
        Self {
            max_response_bytes: DEFAULT_MAX_HUB_RESPONSE_BYTES,
            max_search_entries: DEFAULT_MAX_HUB_SEARCH_ENTRIES,
            max_repo_files: DEFAULT_MAX_HUB_REPO_FILES,
            max_file_size_bytes: DEFAULT_MAX_HUB_FILE_SIZE_BYTES,
            max_file_set_size_bytes: DEFAULT_MAX_HUB_FILE_SET_SIZE_BYTES,
        }
    }
}

/// A per-entry validation failure. `entry` is the remote ID or file path
/// where available, and `rule` names the rule that rejected it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RejectedHubEntry {
    pub entry: String,
    pub rule: &'static str,
}

impl std::fmt::Display for RejectedHubEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "hub entry {:?} failed validation rule {}",
            self.entry, self.rule
        )
    }
}

/// Valid entries and isolated invalid entries from one response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationReport<T> {
    pub valid: T,
    pub rejected: Vec<RejectedHubEntry>,
}

/// One validated row from a Hugging Face model search or feed response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubSearchEntry {
    pub repo_id: String,
    /// Lowercase, immutable 40-character commit ID.
    pub revision: String,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
}

/// A validated file from a repository's `siblings` listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubFileMetadata {
    pub path: String,
    pub size_bytes: Option<u64>,
}

/// Validated repository identity and the safe files from its metadata response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubRepoMetadata {
    pub repo_id: String,
    /// Lowercase, immutable 40-character commit ID returned by the Hub.
    pub revision: String,
    pub files: Vec<HubFileMetadata>,
}

/// One validated repository sibling with an optional LFS SHA-256 identity.
///
/// This source-resolution view is separate from `HubFileMetadata` so the
/// existing public repository and GGUF variant contracts stay unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HubSourceFileMetadata {
    pub path: String,
    pub size_bytes: Option<u64>,
    pub sha256: Option<String>,
    pub remote_validator: Option<String>,
}

/// Validated repository identity and source metadata for exact owner matching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HubSourceRepoMetadata {
    pub repo_id: String,
    pub revision: String,
    pub files: Vec<HubSourceFileMetadata>,
}

/// A whole-response validation failure. Entry-level failures are returned in
/// [`ValidationReport::rejected`] so valid neighbors remain usable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HubValidationError {
    OversizedResponse {
        actual_bytes: usize,
        max_bytes: usize,
    },
    InvalidResponse {
        entry: String,
        rule: &'static str,
    },
}

impl std::fmt::Display for HubValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OversizedResponse {
                actual_bytes,
                max_bytes,
            } => write!(
                f,
                "hub response is {actual_bytes} bytes, above the {max_bytes}-byte limit"
            ),
            Self::InvalidResponse { entry, rule } => {
                write!(
                    f,
                    "hub response entry {entry:?} failed validation rule {rule}"
                )
            }
        }
    }
}

impl std::error::Error for HubValidationError {}

/// Pure validation for remote and cached Hub payloads.
///
/// Search and repository methods parse bounded JSON, validate response shape,
/// and isolate invalid entries. The same methods should be called on cache
/// reads so persisted data does not bypass current validation rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HubMetadataValidator {
    limits: HubValidationLimits,
}

impl Default for HubMetadataValidator {
    fn default() -> Self {
        Self::new(HubValidationLimits::default())
    }
}

impl HubMetadataValidator {
    pub const fn new(limits: HubValidationLimits) -> Self {
        Self { limits }
    }

    pub const fn limits(&self) -> HubValidationLimits {
        self.limits
    }

    /// Validates a Hub model-list payload, including search and trending data.
    /// Invalid rows are isolated in `rejected`; invalid top-level shape or an
    /// over-limit entry count rejects the response as a whole.
    pub fn validate_search_payload(
        &self,
        bytes: &[u8],
    ) -> Result<ValidationReport<Vec<HubSearchEntry>>, HubValidationError> {
        let value = self.parse_bounded_json(bytes)?;
        let entries = value
            .as_array()
            .ok_or_else(|| invalid("<response>", "search_array_shape"))?;
        if entries.len() > self.limits.max_search_entries {
            return Err(invalid("<response>", "entry_count_within_limit"));
        }

        let mut valid = Vec::with_capacity(entries.len());
        let mut rejected = Vec::new();
        for (index, entry) in entries.iter().enumerate() {
            let label = entry
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .unwrap_or_else(|| format!("results[{index}]"));
            let Some(object) = entry.as_object() else {
                rejected.push(rejected_entry(label, "search_entry_shape"));
                continue;
            };
            let Some(repo_id) = object.get("id").and_then(Value::as_str) else {
                rejected.push(rejected_entry(label, "repository_id_shape"));
                continue;
            };
            if !valid_repo_id(repo_id) {
                rejected.push(rejected_entry(label, "repository_id_shape"));
                continue;
            }
            let Some(revision) = object.get("sha").and_then(Value::as_str) else {
                rejected.push(rejected_entry(label, "immutable_revision"));
                continue;
            };
            if !valid_revision(revision) {
                rejected.push(rejected_entry(label, "immutable_revision"));
                continue;
            }
            let downloads = match optional_u64(object.get("downloads")) {
                Ok(value) => value,
                Err(rule) => {
                    rejected.push(rejected_entry(label, rule));
                    continue;
                }
            };
            let likes = match optional_u64(object.get("likes")) {
                Ok(value) => value,
                Err(rule) => {
                    rejected.push(rejected_entry(label, rule));
                    continue;
                }
            };
            valid.push(HubSearchEntry {
                repo_id: repo_id.to_owned(),
                revision: revision.to_ascii_lowercase(),
                downloads,
                likes,
            });
        }

        Ok(ValidationReport { valid, rejected })
    }

    /// Validates a repository detail response with its `siblings` file list.
    /// Invalid sibling files are isolated while the repository identity and
    /// unrelated safe files remain available.
    pub fn validate_repository_payload(
        &self,
        bytes: &[u8],
    ) -> Result<ValidationReport<HubRepoMetadata>, HubValidationError> {
        let value = self.parse_bounded_json(bytes)?;
        let object = value
            .as_object()
            .ok_or_else(|| invalid("<repository>", "repository_metadata_shape"))?;
        if !object.contains_key("id")
            || !object.contains_key("sha")
            || !object.contains_key("siblings")
        {
            return Err(invalid("<repository>", "repository_metadata_shape"));
        }
        let raw_id = object.get("id").and_then(Value::as_str);
        let entry = raw_id.unwrap_or("<repository>");
        let Some(repo_id) = raw_id.filter(|id| valid_repo_id(id)) else {
            return Err(invalid(entry, "repository_id_shape"));
        };
        let Some(revision) = object.get("sha").and_then(Value::as_str) else {
            return Err(invalid(repo_id, "immutable_revision"));
        };
        if !valid_revision(revision) {
            return Err(invalid(repo_id, "immutable_revision"));
        }
        let Some(siblings) = object.get("siblings").and_then(Value::as_array) else {
            return Err(invalid(repo_id, "siblings_array_shape"));
        };
        if siblings.len() > self.limits.max_repo_files {
            return Err(invalid(repo_id, "file_count_within_limit"));
        }

        let mut files = Vec::with_capacity(siblings.len());
        let mut rejected = Vec::new();
        let mut seen = HashSet::with_capacity(siblings.len());
        for (index, sibling) in siblings.iter().enumerate() {
            let label = sibling
                .get("rfilename")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .unwrap_or_else(|| format!("siblings[{index}]"));
            let Some(sibling) = sibling.as_object() else {
                rejected.push(rejected_entry(label, "file_metadata_shape"));
                continue;
            };
            let Some(path) = sibling.get("rfilename").and_then(Value::as_str) else {
                rejected.push(rejected_entry(label, "file_path_shape"));
                continue;
            };
            if !valid_relative_path(path) {
                rejected.push(rejected_entry(label, "safe_relative_path"));
                continue;
            }
            let size_bytes = match optional_u64(sibling.get("size")) {
                Ok(value) => value,
                Err(_) => {
                    rejected.push(rejected_entry(label, "nonnegative_size"));
                    continue;
                }
            };
            if size_bytes.is_some_and(|size| size > self.limits.max_file_size_bytes) {
                rejected.push(rejected_entry(label, "file_size_within_limit"));
                continue;
            }
            if !seen.insert(path.to_owned()) {
                rejected.push(rejected_entry(label, "unique_file_path"));
                continue;
            }
            files.push(HubFileMetadata {
                path: path.to_owned(),
                size_bytes,
            });
        }

        Ok(ValidationReport {
            valid: HubRepoMetadata {
                repo_id: repo_id.to_owned(),
                revision: revision.to_ascii_lowercase(),
                files,
            },
            rejected,
        })
    }

    /// Validates repository metadata and retains SHA-256 identities for LFS
    /// siblings without changing the legacy public repository metadata shape.
    pub(crate) fn validate_source_repository_payload(
        &self,
        bytes: &[u8],
    ) -> Result<ValidationReport<HubSourceRepoMetadata>, HubValidationError> {
        let report = self.validate_repository_payload(bytes)?;
        let value = self.parse_bounded_json(bytes)?;
        let Some(siblings) = value.get("siblings").and_then(Value::as_array) else {
            return Err(invalid("<repository>", "siblings_array_shape"));
        };
        let by_path: std::collections::HashMap<&str, &Value> = siblings
            .iter()
            .filter_map(|sibling| Some((sibling.get("rfilename")?.as_str()?, sibling)))
            .collect();

        let mut valid_files = Vec::with_capacity(report.valid.files.len());
        let mut rejected = report.rejected;
        for file in report.valid.files {
            let Some(sibling) = by_path.get(file.path.as_str()).copied() else {
                rejected.push(rejected_entry(file.path, "file_metadata_shape"));
                continue;
            };
            let validated = sibling_sha256(sibling).and_then(|sha256| {
                let blob = match sibling.get("blobId") {
                    None | Some(Value::Null) => None,
                    Some(Value::String(blob)) if valid_revision(blob) => {
                        Some(blob.to_ascii_lowercase())
                    }
                    Some(_) => return Err("remote_validator_shape"),
                };
                let remote_validator = sha256.clone().or(blob);
                Ok((sha256, remote_validator))
            });
            match validated {
                Ok((sha256, remote_validator)) => valid_files.push(HubSourceFileMetadata {
                    path: file.path,
                    size_bytes: file.size_bytes,
                    sha256,
                    remote_validator,
                }),
                Err(rule) => rejected.push(rejected_entry(file.path, rule)),
            }
        }

        Ok(ValidationReport {
            valid: HubSourceRepoMetadata {
                repo_id: report.valid.repo_id,
                revision: report.valid.revision,
                files: valid_files,
            },
            rejected,
        })
    }

    /// Returns a checked total for a selected file set, or `None` if any size
    /// is unknown. Invalid paths, sizes, zero-byte GGUFs, overflow, and a
    /// configured set-size ceiling produce a named validation error.
    pub fn checked_file_total(
        &self,
        files: &[HubFileMetadata],
    ) -> Result<Option<u64>, HubValidationError> {
        if files.len() > self.limits.max_repo_files {
            return Err(invalid("<file-set>", "file_count_within_limit"));
        }
        let mut total = 0u64;
        let mut unknown_size = false;
        let mut seen = HashSet::with_capacity(files.len());
        for file in files {
            if !valid_relative_path(&file.path) {
                return Err(invalid(&file.path, "safe_relative_path"));
            }
            if !seen.insert(file.path.as_str()) {
                return Err(invalid(&file.path, "unique_file_path"));
            }
            let Some(size) = file.size_bytes else {
                unknown_size = true;
                continue;
            };
            if size > self.limits.max_file_size_bytes {
                return Err(invalid(&file.path, "file_size_within_limit"));
            }
            if file.path.to_ascii_lowercase().ends_with(".gguf") && size == 0 {
                return Err(invalid(&file.path, "nonzero_gguf_size"));
            }
            total = total
                .checked_add(size)
                .ok_or_else(|| invalid(&file.path, "checked_size_total"))?;
            if total > self.limits.max_file_set_size_bytes {
                return Err(invalid(&file.path, "file_set_size_within_limit"));
            }
        }
        Ok((!unknown_size).then_some(total))
    }

    fn parse_bounded_json(&self, bytes: &[u8]) -> Result<Value, HubValidationError> {
        if bytes.len() > self.limits.max_response_bytes {
            return Err(HubValidationError::OversizedResponse {
                actual_bytes: bytes.len(),
                max_bytes: self.limits.max_response_bytes,
            });
        }
        serde_json::from_slice(bytes).map_err(|_| invalid("<response>", "valid_json"))
    }
}

fn optional_u64(value: Option<&Value>) -> Result<Option<u64>, &'static str> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => number.as_u64().map(Some).ok_or("nonnegative_count"),
        Some(_) => Err("nonnegative_count"),
    }
}

pub(crate) fn valid_repo_id(id: &str) -> bool {
    let mut parts = id.split('/');
    let Some(owner) = parts.next() else {
        return false;
    };
    let Some(name) = parts.next() else {
        return false;
    };
    parts.next().is_none() && valid_repo_component(owner) && valid_repo_component(name)
}

fn valid_repo_component(component: &str) -> bool {
    !component.is_empty()
        && component.len() <= 96
        && component != "."
        && component != ".."
        && !component.starts_with('.')
        && !component.ends_with('.')
        && component
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

pub(crate) fn valid_revision(revision: &str) -> bool {
    revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub(crate) fn valid_relative_path(path: &str) -> bool {
    if path.is_empty()
        || path.len() > MAX_HUB_PATH_BYTES
        || path.starts_with('/')
        || path.ends_with('/')
        || path
            .chars()
            .any(|ch| ch.is_control() || matches!(ch, '\\' | ':' | '%' | '?' | '#'))
    {
        return false;
    }
    path.split('/')
        .all(|part| !part.is_empty() && part != "." && part != "..")
}

fn sibling_sha256(sibling: &Value) -> Result<Option<String>, &'static str> {
    let Some(lfs) = sibling.get("lfs") else {
        return Ok(None);
    };
    if lfs.is_null() {
        return Ok(None);
    }
    let Some(lfs) = lfs.as_object() else {
        return Err("lfs_metadata_shape");
    };

    let direct = match lfs.get("sha256") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => Some(value.as_str()),
        Some(_) => return Err("sha256_digest_shape"),
    };
    let oid = match lfs.get("oid") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => {
            Some(value.strip_prefix("sha256:").ok_or("sha256_digest_shape")?)
        }
        Some(_) => return Err("sha256_digest_shape"),
    };

    let direct = direct.map(normalize_sha256).transpose()?;
    let oid = oid.map(normalize_sha256).transpose()?;
    if direct.is_some() && oid.is_some() && direct != oid {
        return Err("sha256_digest_conflict");
    }
    Ok(direct.or(oid))
}

pub(crate) fn normalize_sha256(value: &str) -> Result<String, &'static str> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("sha256_digest_shape");
    }
    Ok(value.to_ascii_lowercase())
}

fn invalid(entry: impl Into<String>, rule: &'static str) -> HubValidationError {
    HubValidationError::InvalidResponse {
        entry: entry.into(),
        rule,
    }
}

fn rejected_entry(entry: impl Into<String>, rule: &'static str) -> RejectedHubEntry {
    RejectedHubEntry {
        entry: entry.into(),
        rule,
    }
}
