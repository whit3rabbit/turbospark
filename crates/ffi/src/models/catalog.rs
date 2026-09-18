//! Catalog and local store model management.

use catalog::{Catalog, Store};
use serde::Serialize;
use std::path::{Path, PathBuf};

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

#[derive(Serialize)]
struct CatalogRow<'a> {
    #[serde(flatten)]
    entry: &'a catalog::CatalogEntry,
    /// Whether this row is already installed, so a GUI needs one call rather
    /// than two and a join.
    installed: bool,
}

pub(crate) fn catalog_json() -> Result<String, String> {
    let catalog = Catalog::embedded()?;
    let installed = Store::default_store()
        .map(|s| s.installed())
        .unwrap_or_default();
    let rows: Vec<CatalogRow> = catalog
        .entries()
        .map(|entry| CatalogRow {
            installed: installed.contains_key(&entry.alias),
            entry,
        })
        .collect();
    serde_json::to_string(&rows).map_err(|e| e.to_string())
}

pub(crate) fn installed_json() -> Result<String, String> {
    let store = Store::default_store()?;
    let rows: Vec<_> = store.installed().into_values().collect();
    serde_json::to_string(&rows).map_err(|e| e.to_string())
}

/// Image installs are deliberately not recorded in the text model index.
/// Discover only valid image manifests under the shared machine store so the
/// app can offer compatible rows without making image artifacts text models.
pub(crate) fn image_installed_json() -> Result<String, String> {
    let store = Store::default_store()?;
    let models = store.root().join("models");
    let mut rows = image_installed_rows(&models);
    rows.sort_by(|a, b| a.alias.cmp(&b.alias));
    serde_json::to_string(&rows).map_err(|e| e.to_string())
}

/// Deletes one valid image install from the shared image store.
///
/// Image installs are absent from the text index, so deletion resolves the
/// alias from the same validated manifest listing used by the app. That keeps
/// arbitrary aliases from becoming filesystem paths.
pub(crate) fn delete_image(alias: &str) -> Result<(), String> {
    let store = Store::default_store()?;
    let models = store.root().join("models");
    let row = image_installed_rows(&models)
        .into_iter()
        .find(|row| row.alias == alias)
        .ok_or_else(|| format!("image model {alias:?} is not installed"))?;
    if row.path.exists() {
        std::fs::remove_dir_all(&row.path)
            .map_err(|e| format!("failed to remove {}: {e}", row.path.display()))?;
    }
    Ok(())
}

fn image_installed_rows(models: &Path) -> Vec<ImageInstalledRow> {
    let Ok(entries) = std::fs::read_dir(models) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_dir() {
                return None;
            }
            let name = path.file_name()?.to_str()?;
            let alias = name.strip_suffix(".image.gturbo")?;
            if alias.is_empty() {
                return None;
            }
            let manifest = image::ImageManifest::load(&path).ok()?;
            manifest.validate().ok()?;
            let source = manifest.source.as_object()?;
            let model_id = source.get("model_id")?.as_str()?.to_string();
            let revision = source.get("model_revision")?.as_str()?.to_string();
            Some(ImageInstalledRow {
                alias: alias.to_string(),
                model_id,
                revision,
                path,
                width: manifest.supported.width,
                height: manifest.supported.height,
                scheduler_steps: manifest.supported.scheduler_steps,
                quantization: image::image_quantization_label(&manifest).ok()?,
            })
        })
        .collect()
}

/// The bytes an install of `alias` will read off the network, so a GUI can
/// show a determinate progress bar and a free-space warning before the walk
/// starts.
pub(crate) fn install_bytes(alias: &str) -> Result<(u64, u64), String> {
    let catalog = Catalog::embedded()?;
    let entry = catalog
        .get(alias)
        .ok_or_else(|| format!("no catalog row named {alias:?}"))?;
    Ok((entry.download_bytes, entry.install_bytes))
}

/// Deletes an installed model from the store and drops its directory.
pub(crate) fn delete(alias: &str) -> Result<(), String> {
    let store = Store::default_store()?;
    delete_from_store(&store, alias)
}

fn delete_from_store(store: &Store, alias: &str) -> Result<(), String> {
    let path = store
        .installed()
        .get(alias)
        .map(|model| model.path.clone())
        .ok_or_else(|| format!("model {alias:?} is not installed"))?;
    if path.exists() {
        std::fs::remove_dir_all(&path)
            .map_err(|e| format!("failed to remove {}: {e}", path.display()))?;
    }
    store.forget(alias)
}

#[cfg(test)]
mod tests {
    use super::*;
    use catalog::InstalledModel;

    fn temp_root(tag: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!(
            "turbospark-ffi-delete-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::remove_dir_all(&root).ok();
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn image_listing_skips_missing_and_invalid_manifests() {
        let root = temp_root("image-listing");
        let models = root.join("models");
        std::fs::create_dir_all(models.join("not-an-image.gturbo")).unwrap();
        std::fs::create_dir_all(models.join("broken.image.gturbo")).unwrap();
        std::fs::write(
            models.join("broken.image.gturbo/manifest.json"),
            "{\"not\":\"an image manifest\"}",
        )
        .unwrap();

        assert!(image_installed_rows(&models).is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }

    fn record(store: &Store, alias: &str, path: &std::path::Path) {
        store
            .record(&InstalledModel {
                alias: alias.to_string(),
                repo: "owner/name".to_string(),
                revision: "main".to_string(),
                path: path.to_path_buf(),
                family: "llama".to_string(),
                install_bytes: 1,
                installed_on: "2026-09-14".to_string(),
                status: "runs".to_string(),
                kind: None,
            })
            .unwrap();
    }

    #[test]
    fn delete_removes_a_recorded_install() {
        let root = temp_root("recorded");
        let store = Store::new(&root);
        let install = root.join("models/model.gturbo");
        std::fs::create_dir_all(&install).unwrap();
        record(&store, "model", &install);

        delete_from_store(&store, "model").unwrap();

        assert!(!install.exists());
        assert!(store.installed().is_empty());
    }

    #[test]
    fn delete_rejects_an_existing_unrecorded_directory() {
        let root = temp_root("unrecorded");
        let store = Store::new(root.join("store"));
        let victim = root.join("Documents");
        std::fs::create_dir_all(&victim).unwrap();
        let sentinel = victim.join("valuable.txt");
        std::fs::write(&sentinel, "keep").unwrap();

        let error = delete_from_store(&store, victim.to_str().unwrap()).unwrap_err();

        assert!(error.contains("not installed"), "got {error:?}");
        assert!(sentinel.is_file(), "unrecorded directory was modified");
    }
}
