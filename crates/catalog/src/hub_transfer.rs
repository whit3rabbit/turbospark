//! Explicit all-or-nothing transport for modality-owned pinned source sets.

use crate::hf::hf_endpoint;
use crate::hf::HubRequestError;
use crate::hub::{
    map_hub_request_error, matches_exact_source, source_file_url_at, validate_pinned_source_group,
    DownloadReceipt, HubClient, HubDownloadProgress, HubError, PinnedArtifactPlan,
    ResolvedSourceIdentity, VerifiedSourceFile, VerifiedSourceGroup,
};
use crate::hub_validation::{normalize_sha256, HubMetadataValidator};
use crate::install::CancelFlag;
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static STAGING_COUNTER: AtomicU64 = AtomicU64::new(0);
const TRANSFER_BUFFER_BYTES: usize = 64 * 1024;

struct PreparedSource<'a> {
    expected: &'a crate::hub::PinnedSourceGroup,
    resolved: ResolvedSourceIdentity,
}

struct OwnedStagingRoot {
    path: PathBuf,
    retained: bool,
}

impl OwnedStagingRoot {
    fn create(parent: &Path) -> Result<Self, HubError> {
        let parent = fs::canonicalize(parent).map_err(|error| {
            HubError::Staging(format!(
                "resolving staging parent {}: {error}",
                parent.display()
            ))
        })?;
        if !parent.is_dir() {
            return Err(HubError::Staging(format!(
                "staging parent {} is not a directory",
                parent.display()
            )));
        }

        for _ in 0..128 {
            let sequence = STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = parent.join(format!(
                ".turbospark-pinned-transfer-{}-{sequence}",
                std::process::id()
            ));
            match create_private_directory(&path) {
                Ok(()) => {
                    return Ok(Self {
                        path,
                        retained: false,
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(staging_io_error(&path, error));
                }
            }
        }
        Err(HubError::Staging(format!(
            "could not allocate a unique transfer directory below {}",
            parent.display()
        )))
    }

    fn retain(mut self) -> PathBuf {
        self.retained = true;
        self.path.clone()
    }

    fn remove(&mut self) -> Result<(), std::io::Error> {
        match fs::remove_dir_all(&self.path) {
            Ok(()) => {
                self.retained = true;
                Ok(())
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                self.retained = true;
                Ok(())
            }
            Err(error) => Err(error),
        }
    }
}

impl Drop for OwnedStagingRoot {
    fn drop(&mut self) {
        if !self.retained {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

#[cfg(unix)]
fn create_private_directory(path: &Path) -> Result<(), std::io::Error> {
    use std::os::unix::fs::DirBuilderExt;

    let mut builder = fs::DirBuilder::new();
    builder.mode(0o700);
    builder.create(path)
}

#[cfg(not(unix))]
fn create_private_directory(path: &Path) -> Result<(), std::io::Error> {
    fs::create_dir(path)
}

/// Resolves every source before staging, then transfers only the owner-pinned
/// paths. The caller-supplied directory is a parent and is never removed.
pub(super) fn download_pinned_artifacts(
    hub: &HubClient<'_>,
    plan: &PinnedArtifactPlan,
    staging_parent: &Path,
    progress: &mut dyn FnMut(HubDownloadProgress),
    cancel: &CancelFlag,
) -> Result<DownloadReceipt, HubError> {
    let validator = HubMetadataValidator::default();
    if plan.sources.is_empty() {
        return Err(HubError::InvalidRequest {
            field: "sources",
            rule: "non_empty_source_set",
        });
    }

    let mut total_files = 0u32;
    for source in &plan.sources {
        validate_pinned_source_group(source)?;
        let count = u32::try_from(source.files.len()).map_err(|_| HubError::InvalidRequest {
            field: "sources",
            rule: "file_count_within_limit",
        })?;
        total_files = total_files
            .checked_add(count)
            .filter(|count| *count <= validator.limits().max_repo_files as u32)
            .ok_or(HubError::InvalidRequest {
                field: "sources",
                rule: "file_count_within_limit",
            })?;
    }

    cancel.checkpoint().map_err(|_| HubError::Cancelled)?;
    let endpoint = hf_endpoint();
    let mut prepared = Vec::with_capacity(plan.sources.len());
    let mut total_bytes = 0u64;
    for expected in &plan.sources {
        cancel.checkpoint().map_err(|_| HubError::Cancelled)?;
        let resolved = hub.resolve_source_identity_at(expected, &endpoint)?;
        matches_exact_source(expected, &resolved).map_err(|mismatch| {
            HubError::InvalidResponse {
                entry: expected.repo.to_string(),
                rule: crate::hub::source_mismatch_rule(mismatch),
            }
        })?;
        for file in &resolved.files {
            let Some(size) = file.authoritative_size else {
                return Err(HubError::InvalidResponse {
                    entry: file.path.clone(),
                    rule: "authoritative_source_size_available",
                });
            };
            total_bytes = total_bytes
                .checked_add(size)
                .ok_or(HubError::InvalidRequest {
                    field: "sources",
                    rule: "total_file_size_within_limit",
                })?;
            if total_bytes > validator.limits().max_file_set_size_bytes {
                return Err(HubError::InvalidRequest {
                    field: "sources",
                    rule: "total_file_size_within_limit",
                });
            }
        }
        prepared.push(PreparedSource { expected, resolved });
    }

    cancel.checkpoint().map_err(|_| HubError::Cancelled)?;
    let mut staging = OwnedStagingRoot::create(staging_parent)?;
    let transfer = transfer_prepared_sources(
        hub,
        plan,
        &prepared,
        &endpoint,
        &staging.path,
        total_files,
        total_bytes,
        progress,
        cancel,
    );
    match transfer {
        Ok(sources) => {
            cancel.checkpoint().map_err(|_| HubError::Cancelled)?;
            let staging_root = staging.retain();
            Ok(DownloadReceipt {
                owner_id: plan.owner_id.clone(),
                sources,
                staging_root,
            })
        }
        Err(error) => {
            if let Err(cleanup_error) = staging.remove() {
                return Err(HubError::Staging(format!(
                    "{error}; removing owned staging root {} failed: {cleanup_error}",
                    staging.path.display()
                )));
            }
            Err(error)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn transfer_prepared_sources(
    hub: &HubClient<'_>,
    plan: &PinnedArtifactPlan,
    prepared: &[PreparedSource<'_>],
    endpoint: &str,
    staging_root: &Path,
    total_files: u32,
    total_bytes: u64,
    progress: &mut dyn FnMut(HubDownloadProgress),
    cancel: &CancelFlag,
) -> Result<Vec<VerifiedSourceGroup>, HubError> {
    let client = hub.client();
    let mut complete_files = 0u32;
    let mut complete_bytes = 0u64;
    let mut verified_sources = Vec::with_capacity(prepared.len());

    progress(HubDownloadProgress {
        owner_id: plan.owner_id.clone(),
        completed_files: 0,
        total_files,
        current_path: None,
        completed_bytes: 0,
        total_bytes,
    });
    cancel.checkpoint().map_err(|_| HubError::Cancelled)?;

    for (source_index, source) in prepared.iter().enumerate() {
        let source_root = staging_root.join(format!("source-{source_index:04}"));
        fs::create_dir(&source_root).map_err(|error| staging_io_error(&source_root, error))?;
        let mut verified_files = Vec::with_capacity(source.expected.files.len());

        for pinned in &source.expected.files {
            cancel.checkpoint().map_err(|_| HubError::Cancelled)?;
            let resolved = source
                .resolved
                .files
                .iter()
                .find(|file| file.path == pinned.path)
                .ok_or(HubError::TransferIntegrity {
                    entry: pinned.path.clone(),
                    rule: "resolved_file_set_matches_plan",
                })?;
            let expected_size = resolved
                .authoritative_size
                .ok_or(HubError::InvalidResponse {
                    entry: pinned.path.clone(),
                    rule: "authoritative_source_size_available",
                })?;
            let destination = append_relative_path(&source_root, &pinned.path);
            let parent = destination.parent().ok_or(HubError::InvalidRequest {
                field: "path",
                rule: "safe_relative_path",
            })?;
            fs::create_dir_all(parent).map_err(|error| staging_io_error(parent, error))?;

            let url = source_file_url_at(endpoint, &source.resolved.repo, &pinned.path)?;
            let mut response = client
                .open_stream_for_hub(&url)
                .map_err(|error| map_hub_request_error(error, &url, client.has_token()))?;
            cancel.checkpoint().map_err(|_| HubError::Cancelled)?;

            let mut staged = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&destination)
                .map_err(|error| staging_io_error(&destination, error))?;
            let mut hasher = Sha256::new();
            let mut file_bytes = 0u64;
            let mut buffer = vec![0u8; TRANSFER_BUFFER_BYTES];
            loop {
                cancel.checkpoint().map_err(|_| HubError::Cancelled)?;
                let read = response.read(&mut buffer).map_err(|error| {
                    map_hub_request_error(
                        HubRequestError::BodyRead(error.to_string()),
                        &url,
                        client.has_token(),
                    )
                })?;
                if read == 0 {
                    break;
                }
                let chunk = &buffer[..read];
                file_bytes =
                    file_bytes
                        .checked_add(read as u64)
                        .ok_or(HubError::TransferIntegrity {
                            entry: pinned.path.clone(),
                            rule: "downloaded_size_matches_authoritative_source",
                        })?;
                if file_bytes > expected_size {
                    return Err(HubError::TransferIntegrity {
                        entry: pinned.path.clone(),
                        rule: "downloaded_size_matches_authoritative_source",
                    });
                }
                staged
                    .write_all(chunk)
                    .map_err(|error| staging_io_error(&destination, error))?;
                hasher.update(chunk);
                complete_bytes =
                    complete_bytes
                        .checked_add(read as u64)
                        .ok_or(HubError::TransferIntegrity {
                            entry: pinned.path.clone(),
                            rule: "download_progress_within_u64",
                        })?;
                progress(HubDownloadProgress {
                    owner_id: plan.owner_id.clone(),
                    completed_files: complete_files,
                    total_files,
                    current_path: Some(pinned.path.clone()),
                    completed_bytes: complete_bytes,
                    total_bytes,
                });
                cancel.checkpoint().map_err(|_| HubError::Cancelled)?;
            }
            staged
                .flush()
                .map_err(|error| staging_io_error(&destination, error))?;
            staged
                .sync_all()
                .map_err(|error| staging_io_error(&destination, error))?;
            if file_bytes != expected_size
                || pinned
                    .expected_size
                    .is_some_and(|expected| expected != file_bytes)
            {
                return Err(HubError::TransferIntegrity {
                    entry: pinned.path.clone(),
                    rule: "downloaded_size_matches_authoritative_source",
                });
            }

            let sha256 = digest_hex(hasher.finalize().as_slice());
            if let Some(expected_digest) = pinned.expected_sha256.as_deref() {
                let expected_digest =
                    normalize_sha256(expected_digest).map_err(|_| HubError::InvalidRequest {
                        field: "expected_sha256",
                        rule: "sha256_digest_shape",
                    })?;
                if sha256 != expected_digest {
                    return Err(HubError::TransferIntegrity {
                        entry: pinned.path.clone(),
                        rule: "downloaded_sha256_matches_pinned_source",
                    });
                }
            }
            if resolved
                .source_sha256
                .as_deref()
                .is_some_and(|source| source != sha256)
            {
                return Err(HubError::TransferIntegrity {
                    entry: pinned.path.clone(),
                    rule: "downloaded_sha256_matches_source_metadata",
                });
            }

            complete_files = complete_files
                .checked_add(1)
                .ok_or(HubError::InvalidRequest {
                    field: "sources",
                    rule: "file_count_within_limit",
                })?;
            verified_files.push(VerifiedSourceFile {
                path: pinned.path.clone(),
                staged_path: destination,
                size: file_bytes,
                sha256,
            });
            progress(HubDownloadProgress {
                owner_id: plan.owner_id.clone(),
                completed_files: complete_files,
                total_files,
                current_path: None,
                completed_bytes: complete_bytes,
                total_bytes,
            });
            cancel.checkpoint().map_err(|_| HubError::Cancelled)?;
        }

        verify_source_file_coverage(source.expected, &verified_files)?;
        verified_sources.push(VerifiedSourceGroup {
            role: source.expected.role.clone(),
            repo: source.resolved.repo.clone(),
            files: verified_files,
        });
    }

    if verified_sources.len() != plan.sources.len() || complete_files != total_files {
        return Err(HubError::TransferIntegrity {
            entry: plan.owner_id.clone(),
            rule: "staged_file_set_matches_plan",
        });
    }
    Ok(verified_sources)
}

fn append_relative_path(root: &Path, path: &str) -> PathBuf {
    let mut destination = root.to_path_buf();
    for segment in path.split('/') {
        destination.push(segment);
    }
    destination
}

fn verify_source_file_coverage(
    expected: &crate::hub::PinnedSourceGroup,
    verified: &[VerifiedSourceFile],
) -> Result<(), HubError> {
    if verified.len() != expected.files.len()
        || verified
            .iter()
            .zip(&expected.files)
            .any(|(verified, expected)| verified.path != expected.path)
    {
        return Err(HubError::TransferIntegrity {
            entry: expected.repo.to_string(),
            rule: "staged_file_set_matches_plan",
        });
    }
    Ok(())
}

fn digest_hex(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

fn staging_io_error(path: &Path, error: std::io::Error) -> HubError {
    HubError::Staging(format!("{}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::verify_source_file_coverage;
    use crate::hf::RepoRef;
    use crate::hub::{PinnedSourceFile, PinnedSourceGroup, VerifiedSourceFile};
    use std::path::PathBuf;

    fn expected(paths: &[&str]) -> PinnedSourceGroup {
        PinnedSourceGroup {
            role: "opaque".to_string(),
            repo: RepoRef::new("owner/model", "0123456789abcdef0123456789abcdef01234567"),
            files: paths
                .iter()
                .map(|path| PinnedSourceFile {
                    path: (*path).to_string(),
                    expected_size: None,
                    expected_sha256: None,
                })
                .collect(),
        }
    }

    fn verified(paths: &[&str]) -> Vec<VerifiedSourceFile> {
        paths
            .iter()
            .map(|path| VerifiedSourceFile {
                path: (*path).to_string(),
                staged_path: PathBuf::from("staged").join(path),
                size: 1,
                sha256: "0".repeat(64),
            })
            .collect()
    }

    #[test]
    fn source_coverage_rejects_missing_extra_and_case_changed_paths() {
        let expected_group = expected(&["a.bin", "nested/b.bin"]);
        assert!(verify_source_file_coverage(
            &expected_group,
            &verified(&["a.bin", "nested/b.bin"])
        )
        .is_ok());
        assert!(matches!(
            verify_source_file_coverage(&expected_group, &verified(&["a.bin"])),
            Err(crate::hub::HubError::TransferIntegrity {
                rule: "staged_file_set_matches_plan",
                ..
            })
        ));
        assert!(matches!(
            verify_source_file_coverage(&expected(&["a.bin"]), &verified(&["a.bin", "extra.bin"])),
            Err(crate::hub::HubError::TransferIntegrity {
                rule: "staged_file_set_matches_plan",
                ..
            })
        ));
        assert!(matches!(
            verify_source_file_coverage(&expected(&["weights.gguf"]), &verified(&["Weights.gguf"])),
            Err(crate::hub::HubError::TransferIntegrity {
                rule: "staged_file_set_matches_plan",
                ..
            })
        ));
    }
}
