//! Catalog and local store model management.

use catalog::{Catalog, Store};
use serde::Serialize;

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
