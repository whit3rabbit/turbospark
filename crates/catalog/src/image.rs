//! Curated Diffusers image sources, separate from the text-model catalog.

use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::Deserialize;

use crate::{CancelFlag, Client, RepoRef, INSTALL_CANCELLED};
use repack::HttpRangeSource;

const EMBEDDED: &str = include_str!("image_models.json");

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct ImageCatalogEntry {
    pub alias: String,
    pub model_id: String,
    pub revision: String,
    pub required_files: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ImageCatalog {
    entries: Vec<ImageCatalogEntry>,
}

impl ImageCatalog {
    pub fn embedded() -> Result<Self, String> {
        let entries: Vec<ImageCatalogEntry> = serde_json::from_str(EMBEDDED)
            .map_err(|e| format!("parsing embedded image catalog: {e}"))?;
        if entries.is_empty() {
            return Err("embedded image catalog is empty".to_string());
        }
        for entry in &entries {
            validate(entry)?;
        }
        Ok(Self { entries })
    }

    pub fn entries(&self) -> impl Iterator<Item = &ImageCatalogEntry> {
        self.entries.iter()
    }

    pub fn get(&self, alias: &str) -> Option<&ImageCatalogEntry> {
        self.entries.iter().find(|entry| entry.alias == alias)
    }
}

/// Materializes the bounded Diffusers source tree consumed by the image
/// packer. Large safetensors files use bounded concurrent ranges written to a
/// sibling partial file. Completed ranges are cached with SHA-256 under the
/// staging tree, so an interrupted immutable-revision install can reuse them.
pub fn materialize_image_source(
    client: &Client,
    repo: &RepoRef,
    destination: &Path,
    cancel: Option<&CancelFlag>,
    mut progress: impl FnMut(&str),
    on_bytes: Arc<dyn Fn(u64, u64) + Send + Sync>,
) -> Result<u64, String> {
    if destination.exists() && !destination.is_dir() {
        return Err(format!(
            "image source staging path is not a directory: {}",
            destination.display()
        ));
    }
    if repo.revision.len() != 40 || !repo.revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!(
            "image repository revision {:?} is not an immutable 40-hex commit",
            repo.revision
        ));
    }
    let files = client.file_list_with_sizes(repo)?;
    let selected = select_files(
        &files
            .iter()
            .map(|file| file.name.clone())
            .collect::<Vec<_>>(),
    )?;
    let selected = selected
        .into_iter()
        .map(|name| {
            let listed = files
                .iter()
                .find(|file| file.name == name)
                .ok_or_else(|| format!("image repository stopped listing {name}"))?;
            let size = match listed.size {
                Some(size) => size,
                None => client
                    .content_length(&repo.file_url(&name))?
                    .ok_or_else(|| format!("{name}: missing content length"))?,
            };
            Ok((name, size))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let total = selected.iter().try_fold(0u64, |sum, (_, size)| {
        sum.checked_add(*size)
            .ok_or_else(|| "image source download byte count overflowed u64".to_string())
    })?;
    let completed = Arc::new(AtomicU64::new(0));
    std::fs::create_dir_all(destination)
        .map_err(|e| format!("creating {}: {e}", destination.display()))?;
    let cache = destination.join(".download-cache");
    for (remote, expected_size) in selected {
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(INSTALL_CANCELLED.to_string());
        }
        let local = safe_join(destination, &remote)?;
        let stage = format!("downloading {remote}");
        progress(&stage);
        // Rebuild even a size-matching source file from the verified range
        // cache. Size alone cannot prove that a source left by a crashed pack
        // is still intact; cache entries carry hashes and remain local I/O.
        if local.exists() {
            std::fs::remove_file(&local)
                .map_err(|e| format!("removing incomplete {}: {e}", local.display()))?;
        }
        let done = Arc::clone(&completed);
        let report = Arc::clone(&on_bytes);
        let range_progress: repack::ByteProgressCallback = Arc::new(move |delta| {
            let cumulative = done.fetch_add(delta, Ordering::Relaxed) + delta;
            report(cumulative, total);
        });
        let mut source = HttpRangeSource::with_progress(repo.file_url(&remote), range_progress)
            .with_optional_token(client.token())
            .with_cache_dir(&cache);
        if let Some(cancel) = cancel {
            source = source.with_cancel(cancel.clone());
        }
        source
            .download_to(&local, expected_size)
            .map_err(|e| format!("downloading {remote}: {e}"))?;
        progress(&stage);
    }
    if cancel.is_some_and(CancelFlag::is_cancelled) {
        return Err(INSTALL_CANCELLED.to_string());
    }
    Ok(completed.load(Ordering::Relaxed))
}

fn select_files(files: &[String]) -> Result<Vec<String>, String> {
    let mut selected = Vec::new();
    let mut has_scheduler = false;
    let mut components = [false; 3];
    for file in files {
        let path = Path::new(file);
        if !safe_relative(path) {
            return Err(format!("Hugging Face file path is unsafe: {file}"));
        }
        let is_metadata = file == "scheduler/scheduler_config.json"
            || file == "scheduler/config.json"
            || file.ends_with("/config.json")
            || file.ends_with(".index.json");
        let is_tokenizer = file == "tokenizer"
            || file.starts_with("tokenizer/")
            || file.starts_with("tokenizer_config.")
            || file == "special_tokens_map.json";
        let component = if file.starts_with("text_encoder/") {
            Some(0)
        } else if file.starts_with("transformer/") {
            Some(1)
        } else if file.starts_with("vae/") {
            Some(2)
        } else {
            None
        };
        let is_weight = component.is_some() && file.ends_with(".safetensors");
        if is_metadata || is_tokenizer || is_weight {
            if file == "scheduler/scheduler_config.json" || file == "scheduler/config.json" {
                has_scheduler = true;
            }
            if let Some(index) = component.filter(|_| is_weight) {
                components[index] = true;
            }
            selected.push(file.clone());
        }
    }
    if !has_scheduler {
        return Err("image repository is missing scheduler/scheduler_config.json".to_string());
    }
    for (name, present) in [
        ("text_encoder", components[0]),
        ("transformer", components[1]),
        ("vae", components[2]),
    ] {
        if !present {
            return Err(format!("image repository is missing {name} safetensors"));
        }
    }
    selected.sort();
    selected.dedup();
    Ok(selected)
}

fn safe_join(root: &Path, remote: &str) -> Result<PathBuf, String> {
    let path = Path::new(remote);
    if !safe_relative(path) {
        return Err(format!("Hugging Face file path is unsafe: {remote}"));
    }
    Ok(root.join(path))
}

fn safe_relative(path: &Path) -> bool {
    !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn validate(entry: &ImageCatalogEntry) -> Result<(), String> {
    if entry.alias.trim().is_empty() || entry.model_id.split('/').count() != 2 {
        return Err(format!("invalid image catalog identity {:?}", entry.alias));
    }
    if entry.revision.len() != 40 || !entry.revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!(
            "image catalog entry {} is not pinned to a 40-hex revision",
            entry.alias
        ));
    }
    if entry.required_files.is_empty()
        || entry
            .required_files
            .iter()
            .any(|file| file.is_empty() || file.starts_with('/') || file.contains(".."))
    {
        return Err(format!(
            "image catalog entry {} has invalid required files",
            entry.alias
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::ImageCatalog;

    #[test]
    fn the_embedded_image_catalog_is_pinned_and_separate() {
        let catalog = ImageCatalog::embedded().expect("image catalog");
        let entry = catalog.get("z-image-turbo").expect("Z-Image row");
        assert_eq!(entry.model_id, "Tongyi-MAI/Z-Image-Turbo");
        assert_eq!(entry.revision.len(), 40);
        assert!(entry
            .required_files
            .iter()
            .all(|file| !file.contains("text-generation")));

        for (alias, model_id, revision) in [
            (
                "z-image-turbo-mlx-2bit",
                "andrevp/Z-Image-Turbo-MLX-2bit",
                "32b4e9ceb3a813485027b1ea942f199608fb8200",
            ),
            (
                "z-image-turbo-mlx-4bit",
                "andrevp/Z-Image-Turbo-MLX-4bit",
                "9adc576198c9126874792d35569b53cf2f45a03c",
            ),
            (
                "z-image-turbo-mlx-8bit",
                "andrevp/Z-Image-Turbo-MLX-8bit",
                "c9f70995562299b1eda9b9145a94dd7a5a1ae0d6",
            ),
            (
                "z-image-turbo-mlx-fp16",
                "andrevp/Z-Image-Turbo-MLX",
                "e186d7d65d66883270671fcee05324178928ea03",
            ),
        ] {
            let entry = catalog.get(alias).expect("published MLX image row");
            assert_eq!(entry.model_id, model_id);
            assert_eq!(entry.revision, revision);
            assert_eq!(entry.required_files.len(), 17);
        }
    }
}
