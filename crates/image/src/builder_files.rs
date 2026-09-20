//! Filesystem assembly and receipt binding for image installs.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::install::{ImageManifest, ImageManifestFile, IMAGE_RECEIPT_NAME};
use crate::packed::{PackedIndex, PackedTensorReport, PACKED_DATA_NAME, PACKED_INDEX_NAME};
use crate::runtime::{IMAGE_MLX_QUANTIZATION, IMAGE_QUANTIZATION, IMAGE_UNQUANTIZED};

pub(super) fn write_receipt(
    spec: &super::ImageInstallSpec,
    staging: &Path,
    manifest: &ImageManifest,
) -> Result<(), String> {
    let manifest_path = staging.join("manifest.json");
    let manifest_sha256 = model_io::hash_file(&manifest_path, 1 << 20)
        .map_err(|e| format!("failed to hash image manifest: {e}"))?;
    let manifest_size = fs::metadata(&manifest_path)
        .map_err(|e| format!("failed to stat image manifest: {e}"))?
        .len();
    let mut files = BTreeMap::new();
    files.insert(
        "manifest.json".to_string(),
        model_io::InstallReceiptFileEntry {
            size: manifest_size,
            sha256: manifest_sha256.clone(),
        },
    );
    for file in &manifest.files {
        files.insert(
            file.path.clone(),
            model_io::InstallReceiptFileEntry {
                size: file.size_bytes,
                sha256: file.sha256.clone(),
            },
        );
    }
    let receipt = model_io::VerifiedInstallReceipt {
        schema_version: 1,
        manifest_sha256,
        model_directory_path: future_canonical_path(&spec.output_root)?,
        source_repo_id: Some(spec.model_id.clone()),
        source_revision: Some(spec.model_revision.clone()),
        verification_timestamp: format!(
            "unix:{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| format!("system clock is before UNIX epoch: {e}"))?
                .as_secs()
        ),
        tool_version: env!("CARGO_PKG_VERSION").to_string(),
        files,
    };
    let bytes = serde_json::to_vec_pretty(&receipt)
        .map_err(|e| format!("failed to serialize image install receipt: {e}"))?;
    fs::write(staging.join(IMAGE_RECEIPT_NAME), bytes)
        .map_err(|e| format!("failed to write image install receipt: {e}"))
}

pub(super) fn future_canonical_path(output: &Path) -> Result<String, String> {
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let name = output
        .file_name()
        .ok_or_else(|| "image install output has no file name".to_string())?;
    Ok(parent
        .canonicalize()
        .map_err(|e| format!("failed to canonicalize image install parent: {e}"))?
        .join(name)
        .display()
        .to_string())
}

pub(super) fn collect_files(
    staging: &Path,
    reports: &BTreeMap<String, PackedTensorReport>,
) -> Result<Vec<ImageManifestFile>, String> {
    let mut files = Vec::new();
    for path in collect_paths(staging)? {
        let rel = path
            .strip_prefix(staging)
            .map_err(|e| format!("install file is outside staging: {e}"))?;
        let relative = rel.to_string_lossy().replace('\\', "/");
        if relative == "manifest.json" || relative == IMAGE_RECEIPT_NAME {
            continue;
        }
        let size_bytes = fs::metadata(&path)
            .map_err(|e| format!("failed to stat {}: {e}", path.display()))?
            .len();
        let owner = owner_for_path(&relative)?;
        let (storage_dtype, quantization, inventory) = file_metadata(&path, &relative)?;
        let sha256 = if relative.ends_with(PACKED_DATA_NAME) {
            let component = relative
                .strip_prefix("components/")
                .and_then(|rest| rest.split('/').next())
                .ok_or_else(|| format!("packed payload {relative} has no component"))?;
            reports
                .get(component)
                .ok_or_else(|| format!("packed payload {relative} has no pack report"))?
                .data_sha256
                .clone()
        } else {
            model_io::hash_file(&path, 1 << 20)
                .map_err(|e| format!("failed to hash {}: {e}", path.display()))?
        };
        files.push(ImageManifestFile {
            path: relative,
            owner,
            size_bytes,
            sha256,
            storage_dtype,
            quantization,
            tensor_inventory_sha256: inventory,
        });
    }
    files.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(files)
}

fn file_metadata(
    path: &Path,
    relative: &str,
) -> Result<(String, serde_json::Value, String), String> {
    if relative.ends_with(PACKED_INDEX_NAME) {
        let bytes = fs::read(path)
            .map_err(|e| format!("failed to read packed index {}: {e}", path.display()))?;
        let index: PackedIndex = serde_json::from_slice(&bytes)
            .map_err(|e| format!("failed to parse packed index {}: {e}", path.display()))?;
        return Ok((
            "index".to_string(),
            serde_json::json!({"scheme": "packed-index", "version": index.version}),
            index.tensor_inventory_sha256,
        ));
    }
    if relative.ends_with(PACKED_DATA_NAME) {
        let index_path = path
            .parent()
            .ok_or_else(|| format!("packed payload {} has no parent", path.display()))?
            .join(PACKED_INDEX_NAME);
        let index_bytes = fs::read(&index_path)
            .map_err(|e| format!("failed to read {}: {e}", index_path.display()))?;
        let index: PackedIndex = serde_json::from_slice(&index_bytes)
            .map_err(|e| format!("failed to parse packed index {}: {e}", index_path.display()))?;
        let observed_affine_bit_widths = observed_mlx_bit_widths(&index);
        let scheme = quantization_scheme(&index);
        return Ok((
            "mixed".to_string(),
            serde_json::json!({
                "scheme": scheme,
                "label": quantization_label(&index),
                "group_size": 64,
                "affine_bit_widths": crate::packed::MLX_AFFINE_BITS,
                "observed_affine_bit_widths": observed_affine_bit_widths,
            }),
            index.tensor_inventory_sha256,
        ));
    }
    Ok((
        "metadata".to_string(),
        serde_json::json!({"scheme": "none"}),
        model_io::hash_data(b""),
    ))
}

pub(super) fn observed_mlx_bit_widths(index: &PackedIndex) -> Vec<u8> {
    let mut widths: Vec<u8> = index
        .tensors
        .values()
        .filter(|tensor| tensor.storage_dtype == "MLX_AFFINE")
        .filter_map(|tensor| {
            tensor
                .quantization
                .as_ref()
                .map(|quantization| quantization.bits)
        })
        .collect();
    widths.sort_unstable();
    widths.dedup();
    widths
}

pub(super) fn quantization_scheme(index: &PackedIndex) -> &'static str {
    if index
        .tensors
        .values()
        .any(|tensor| tensor.storage_dtype == "MLX_AFFINE")
    {
        IMAGE_MLX_QUANTIZATION
    } else if index
        .tensors
        .values()
        .any(|tensor| tensor.storage_dtype == "INT4_AFFINE")
    {
        IMAGE_QUANTIZATION
    } else {
        IMAGE_UNQUANTIZED
    }
}

pub(super) fn quantization_label(index: &PackedIndex) -> String {
    let scheme = quantization_scheme(index);
    if scheme != IMAGE_MLX_QUANTIZATION {
        return scheme.to_string();
    }
    let suffix = observed_mlx_bit_widths(index)
        .into_iter()
        .map(|width| width.to_string())
        .collect::<Vec<_>>()
        .join("-");
    format!("{scheme}-bits-{suffix}")
}

pub(super) fn owner_for_path(path: &str) -> Result<String, String> {
    let component = path
        .strip_prefix("components/")
        .and_then(|rest| rest.split('/').next())
        .ok_or_else(|| format!("install file {path} is outside components"))?;
    let owner = match component {
        "tokenizer" | "text_encoder" => "text_encoder_stage",
        "transformer" => "transformer_stage",
        "scheduler" => "pipeline_control",
        "vae_decoder" => "vae_stage",
        other => return Err(format!("unknown image component directory {other}")),
    };
    Ok(owner.to_string())
}

pub(super) fn copy_tree(source: &Path, destination: &Path) -> Result<(), String> {
    if !source.is_dir() {
        return Err(format!(
            "image component source is not a directory: {}",
            source.display()
        ));
    }
    fs::create_dir(destination)
        .map_err(|e| format!("failed to create {}: {e}", destination.display()))?;
    for entry in
        fs::read_dir(source).map_err(|e| format!("failed to read {}: {e}", source.display()))?
    {
        let entry = entry.map_err(|e| format!("failed to read image source entry: {e}"))?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let file_type = entry
            .file_type()
            .map_err(|e| format!("failed to inspect {}: {e}", source_path.display()))?;
        if file_type.is_symlink() {
            return Err(format!(
                "refusing symlink in image source: {}",
                source_path.display()
            ));
        }
        if file_type.is_dir() {
            copy_tree(&source_path, &destination_path)?;
        } else if file_type.is_file() {
            copy_file(&source_path, &destination_path)?;
        } else {
            return Err(format!(
                "unsupported image source entry: {}",
                source_path.display()
            ));
        }
    }
    Ok(())
}

fn copy_file(source: &Path, destination: &Path) -> Result<(), String> {
    fs::copy(source, destination).map(|_| ()).map_err(|e| {
        format!(
            "failed to copy {} to {}: {e}",
            source.display(),
            destination.display()
        )
    })
}

pub(super) fn collect_paths(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    collect_paths_inner(root, &mut files)?;
    Ok(files)
}

fn collect_paths_inner(root: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in
        fs::read_dir(root).map_err(|e| format!("failed to read {}: {e}", root.display()))?
    {
        let entry = entry.map_err(|e| format!("failed to read install entry: {e}"))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|e| format!("failed to inspect {}: {e}", path.display()))?;
        if file_type.is_symlink() {
            return Err(format!(
                "refusing symlink in image install: {}",
                path.display()
            ));
        }
        if file_type.is_dir() {
            collect_paths_inner(&path, files)?;
        } else if file_type.is_file() {
            files.push(path);
        }
    }
    Ok(())
}

pub(super) fn staging_path(output: &Path) -> Result<PathBuf, String> {
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let name = output
        .file_name()
        .ok_or_else(|| "image install output has no file name".to_string())?
        .to_string_lossy();
    let staging = parent.join(format!(".{name}.turbospark-partial-{}", std::process::id()));
    if staging.exists() {
        return Err(format!(
            "staging install already exists: {}",
            staging.display()
        ));
    }
    Ok(staging)
}
