//! Curated image catalog and install entry points for native GUI hosts.

use std::path::PathBuf;
use std::sync::Arc;

use serde::Serialize;

use super::install::ActiveInstall;

#[derive(Serialize)]
struct ImageCatalogRow {
    alias: String,
    #[serde(rename = "modelID")]
    model_id: String,
    revision: String,
    quantization: String,
}

#[derive(Serialize)]
struct ImageInstalledRow {
    alias: String,
    #[serde(rename = "modelID")]
    model_id: String,
    revision: String,
    path: PathBuf,
    width: u32,
    height: u32,
    #[serde(rename = "schedulerSteps")]
    scheduler_steps: u32,
    quantization: String,
}

pub(crate) fn catalog_json() -> Result<String, String> {
    let catalog = catalog::ImageCatalog::embedded()?;
    let rows = catalog
        .entries()
        .map(|entry| ImageCatalogRow {
            alias: entry.alias.clone(),
            model_id: entry.model_id.clone(),
            revision: entry.revision.clone(),
            quantization: quantization_label(&entry.alias),
        })
        .collect::<Vec<_>>();
    serde_json::to_string(&rows).map_err(|e| e.to_string())
}

/// Downloads and packs one curated image row into the shared image store.
/// Image installs intentionally do not enter the text `installed.json` index.
pub(crate) fn install(
    alias: &str,
    mut on_stage: impl FnMut(&str),
    on_bytes: Arc<dyn Fn(u64, u64) + Send + Sync>,
) -> Result<String, String> {
    let active = ActiveInstall::register_image();
    let catalog = catalog::ImageCatalog::embedded()?;
    let entry = catalog
        .get(alias)
        .ok_or_else(|| format!("no image catalog row named {alias:?}"))?
        .clone();
    let store = catalog::Store::default_store()?;
    let output = store.image_install_path(alias);
    if output.exists() {
        return Err(format!(
            "{} already holds an image install",
            output.display()
        ));
    }
    let parent = output
        .parent()
        .ok_or_else(|| "image install has no parent directory".to_string())?;
    std::fs::create_dir_all(parent)
        .map_err(|e| format!("creating image install parent {}: {e}", parent.display()))?;
    let source = parent.join(format!(".{alias}.{}.image-source", entry.revision));
    let repo = catalog::RepoRef::new(&entry.model_id, &entry.revision);
    on_stage("preparing image source");
    let downloaded = catalog::materialize_image_source(
        &catalog::Client::new(),
        &repo,
        &source,
        Some(active.cancel_flag()),
        |stage| on_stage(stage),
        Arc::clone(&on_bytes),
    )?;
    on_stage(&format!("downloaded {downloaded} image source bytes"));

    let report = image::build_image_install_with_progress(
        &image::ImageInstallSpec {
            source_root: source.clone(),
            output_root: output.clone(),
            model_id: entry.model_id.clone(),
            model_revision: entry.revision.clone(),
        },
        |stage| on_stage(stage),
    )?;
    let _ = std::fs::remove_dir_all(&source);
    let manifest = image::ImageManifest::load(&report.output_root)?;
    manifest.validate()?;
    let path = report
        .output_root
        .canonicalize()
        .unwrap_or(report.output_root);
    serde_json::to_string(&ImageInstalledRow {
        alias: entry.alias,
        model_id: entry.model_id,
        revision: entry.revision,
        path,
        width: manifest.supported.width,
        height: manifest.supported.height,
        scheduler_steps: manifest.supported.scheduler_steps,
        quantization: image::image_quantization_label(&manifest)?,
    })
    .map_err(|e| e.to_string())
}

fn quantization_label(alias: &str) -> String {
    if alias.ends_with("-fp16") {
        return "unquantized".to_string();
    }
    alias
        .rsplit_once('-')
        .and_then(|(_, bits)| bits.strip_suffix("bit"))
        .and_then(|bits| bits.parse::<u8>().ok())
        .map(|bits| format!("mlx-affine-linear-weights-group-64-bits-{bits}"))
        .unwrap_or_else(|| "unquantized".to_string())
}

#[cfg(test)]
mod tests {
    use super::quantization_label;

    #[test]
    fn image_catalog_labels_match_the_published_widths() {
        assert_eq!(
            quantization_label("z-image-turbo-mlx-2bit"),
            "mlx-affine-linear-weights-group-64-bits-2"
        );
        assert_eq!(quantization_label("z-image-turbo-mlx-fp16"), "unquantized");
    }
}
