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
    let path = store
        .resolve(alias)
        .ok_or_else(|| format!("model {alias:?} is not installed"))?;
    if path.exists() {
        std::fs::remove_dir_all(&path)
            .map_err(|e| format!("failed to remove {}: {e}", path.display()))?;
    }
    store.forget(alias)
}
