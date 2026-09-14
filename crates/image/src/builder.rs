//! Atomic assembly of a checked image-generation install.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[path = "builder_files.rs"]
mod builder_files;
#[path = "builder_manifest.rs"]
mod builder_manifest;

use crate::packed::PackedTensorReport;
use builder_files::{collect_paths, copy_tree, future_canonical_path, staging_path, write_receipt};
use builder_manifest::make_manifest;

const COMPONENTS_DIR: &str = "components";
const TEXT_ENCODER_INDEX: &str = "model.safetensors.index.json";
const TRANSFORMER_INDEX: &str = "diffusion_pytorch_model.safetensors.index.json";
const VAE_INDEX: &str = "diffusion_pytorch_model.safetensors.index.json";

/// Source tree expected by [`build_image_install`].
///
/// The source is a Diffusers-style export with tokenizer assets under
/// `tokenizer/`, the scheduler config under `scheduler/scheduler_config.json`,
/// and indexed or conventional single-file safetensors under `text_encoder/`,
/// `transformer/`, and `vae/`. The builder also accepts the install component
/// spellings `config.json` and `vae_decoder/` for local fixtures. It does not
/// download or mutate this tree.
#[derive(Debug, Clone)]
pub struct ImageInstallSpec {
    pub source_root: PathBuf,
    pub output_root: PathBuf,
    pub model_id: String,
    pub model_revision: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageInstallReport {
    pub output_root: PathBuf,
    pub component_reports: BTreeMap<String, PackedTensorReport>,
    pub file_count: usize,
    pub total_bytes: u64,
}

/// Pack and publish a complete image install without exposing a partial tree.
pub fn build_image_install(spec: &ImageInstallSpec) -> Result<ImageInstallReport, String> {
    validate_spec(spec)?;
    let staging = staging_path(&spec.output_root)?;
    fs::create_dir(&staging).map_err(|e| format!("failed to create staging install: {e}"))?;

    let result = build_staged(spec, &staging);
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    let report = result?;
    fs::rename(&staging, &spec.output_root).map_err(|e| {
        let _ = fs::remove_dir_all(&staging);
        format!(
            "failed to publish image install {}: {e}",
            spec.output_root.display()
        )
    })?;
    Ok(ImageInstallReport {
        output_root: spec.output_root.clone(),
        ..report
    })
}

fn validate_spec(spec: &ImageInstallSpec) -> Result<(), String> {
    if !spec.source_root.is_dir() {
        return Err(format!(
            "image source root is not a directory: {}",
            spec.source_root.display()
        ));
    }
    if spec.output_root.exists() {
        return Err(format!(
            "refusing to overwrite existing image install {}",
            spec.output_root.display()
        ));
    }
    if spec.model_id.trim().is_empty() || spec.model_revision.trim().is_empty() {
        return Err("image install requires model_id and model_revision".to_string());
    }
    if let Some(parent) = spec.output_root.parent() {
        if !parent.is_dir() {
            return Err(format!(
                "image install output parent is not a directory: {}",
                parent.display()
            ));
        }
    }
    Ok(())
}

fn build_staged(spec: &ImageInstallSpec, staging: &Path) -> Result<ImageInstallReport, String> {
    let components = staging.join(COMPONENTS_DIR);
    fs::create_dir(&components)
        .map_err(|e| format!("failed to create components directory: {e}"))?;
    copy_tree(
        &spec.source_root.join("tokenizer"),
        &components.join("tokenizer"),
    )?;

    let mut component_reports = BTreeMap::new();
    for (name, source_dir, index_name) in [
        ("text_encoder", "text_encoder", TEXT_ENCODER_INDEX),
        ("transformer", "transformer", TRANSFORMER_INDEX),
        (
            "vae_decoder",
            if spec.source_root.join("vae").is_dir() {
                "vae"
            } else {
                "vae_decoder"
            },
            VAE_INDEX,
        ),
    ] {
        let report = crate::packed::pack_component(
            &spec.source_root.join(source_dir),
            index_name,
            &components.join(name),
        )?;
        component_reports.insert(name.to_string(), report);
    }

    let scheduler_dst = components.join("scheduler");
    fs::create_dir(&scheduler_dst)
        .map_err(|e| format!("failed to create scheduler component: {e}"))?;
    let scheduler_config = [
        spec.source_root.join("scheduler/scheduler_config.json"),
        spec.source_root.join("scheduler/config.json"),
    ]
    .into_iter()
    .find(|path| path.is_file())
    .ok_or_else(|| {
        format!(
            "image scheduler source is missing scheduler/scheduler_config.json: {}",
            spec.source_root.display()
        )
    })?;
    fs::copy(scheduler_config, scheduler_dst.join("config.json"))
        .map_err(|e| format!("failed to copy scheduler config: {e}"))?;

    let manifest = make_manifest(spec, staging, &component_reports)?;
    manifest.validate()?;
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|e| format!("failed to serialize image manifest: {e}"))?;
    fs::write(staging.join("manifest.json"), manifest_bytes)
        .map_err(|e| format!("failed to write image manifest: {e}"))?;
    write_receipt(spec, staging, &manifest)?;
    let final_directory_path = future_canonical_path(&spec.output_root)?;
    let validation = manifest.verify_files_for_publication(staging, &final_directory_path)?;
    let total_bytes = collect_paths(staging)?
        .into_iter()
        .map(|path| {
            fs::metadata(&path)
                .map(|metadata| metadata.len())
                .map_err(|error| format!("failed to stat {}: {error}", path.display()))
        })
        .try_fold(0u64, |total, size| {
            size?
                .checked_add(total)
                .ok_or_else(|| "image install byte count overflowed u64".to_string())
        })?;

    Ok(ImageInstallReport {
        output_root: PathBuf::new(),
        component_reports,
        file_count: validation.file_count,
        total_bytes,
    })
}

#[cfg(test)]
#[path = "builder_tests.rs"]
mod tests;
