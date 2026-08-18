//! The model-management surface: browse, probe, install.
//!
//! Portable. Nothing here decodes, so a GUI can list the catalog and install
//! a model on a platform where it could not then run one -- which is the
//! honest behaviour, since the artifact is the same either way.
//!
//! The catalog and store rows are `crates/catalog`'s own `Serialize` types
//! passed straight through, so a GUI's rows and `turbospark-model list`'s
//! rows are the same data and cannot drift. `ProbeReport` is NOT
//! `Serialize` (it carries an `ArchConfig` and a `ModelFamily`), so its
//! projection below is written by hand -- deliberately narrow, since a GUI
//! needs the verdict and the arithmetic behind it rather than the whole
//! config.

use std::sync::Arc;

use catalog::{Catalog, Client, InstallPlan, RepoRef, Store, Verdict};
use serde::Serialize;
use serde_json::json;

/// The install-progress event kinds. Mirrored in `turbospark.h`.
pub const TS_INSTALL_STAGE: i32 = 0;
pub const TS_INSTALL_BYTES: i32 = 1;

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

/// The header-only probe: what this engine makes of an arbitrary repository,
/// reading kilobytes rather than gigabytes.
///
/// `repo` and `sidecarRepo` accept `owner/name` or `owner/name@revision`.
pub(crate) fn probe_json(
    repo: &str,
    file: Option<&str>,
    sidecar_repo: Option<&str>,
) -> Result<String, String> {
    let repo = parse_repo(repo)?;
    let sidecars = sidecar_repo.map(parse_repo).transpose()?;
    let report = catalog::probe(&Client::new(), &repo, file, sidecars.as_ref())?;

    // A hand-built projection rather than a derived one. The two omissions
    // are deliberate: the full `ArchConfig` is engine internals a GUI has no
    // use for, and `slot_cache_bytes` is included BECAUSE it is the number
    // that decides whether a model fits at all -- `slots x layers x stride`,
    // not the model's size.
    Ok(json!({
        "repo": repo.repo,
        "revision": repo.revision,
        "file": report.file,
        "downloadBytes": report.download_bytes,
        "architecture": report.architecture,
        "family": report.family.map(|f| f.as_str()),
        "runnable": report.verdict.is_runnable(),
        "refusedBecause": match &report.verdict {
            Verdict::Runnable => None,
            Verdict::Refused(why) => Some(why.clone()),
        },
        "types": report.types.iter().map(|t| json!({
            "name": t.name,
            "tensors": t.tensors,
            // Null rather than zero when this port cannot size the type,
            // which is a much louder statement: an unsized type is usually
            // the one carrying the model.
            "bytes": t.bytes,
            "executable": t.executable,
        })).collect::<Vec<_>>(),
        "affine": report.affine.map(|(bits, group)| json!({"bits": bits, "groupSize": group})),
        "expertStride": report.expert_stride,
        "slotCacheBytes": report.slot_cache_bytes().iter()
            .map(|(slots, bytes)| json!({"slots": slots, "bytes": bytes}))
            .collect::<Vec<_>>(),
        "sidecarsPresent": report.sidecars_present,
        "sidecarsMissing": report.sidecars_missing,
        "chatTemplate": report.chat_template,
        "warnings": report.warnings,
    })
    .to_string())
}

/// `owner/name` or `owner/name@revision`, defaulting to `main`.
fn parse_repo(spec: &str) -> Result<RepoRef, String> {
    let (repo, revision) = match spec.split_once('@') {
        Some((r, rev)) => (r, rev),
        None => (spec, "main"),
    };
    if repo.split('/').count() != 2 || repo.starts_with('/') || repo.ends_with('/') {
        return Err(format!(
            "repository must be owner/name (optionally @revision), got {spec:?}"
        ));
    }
    Ok(RepoRef::new(repo, revision))
}

/// Installs the catalog row named `alias` into the store.
///
/// **THE WALK CANNOT RESUME.** It streams 4 to 25 GB and a failure restarts
/// it from the beginning, which is why the first stage line says so and why
/// a GUI must surface that before starting rather than after failing.
///
/// **THE BYTE CALLBACK IS CALLED FROM WORKER THREADS.** `HttpRangeSource`
/// splits a large range into concurrent chunks, so byte progress arrives
/// concurrently and out of order while the STAGE lines arrive on this
/// thread. That is why `on_bytes` is a separate `Fn` bound rather than
/// another arm of the same `FnMut`.
pub(crate) fn install(
    alias: &str,
    mut on_stage: impl FnMut(&str),
    on_bytes: Arc<dyn Fn(u64) + Send + Sync>,
) -> Result<String, String> {
    let catalog = Catalog::embedded()?;
    let entry = catalog
        .get(alias)
        .ok_or_else(|| format!("no catalog row named {alias:?}"))?;
    let plan = InstallPlan::from_entry(entry);
    let store = Store::default_store()?;
    let dir = store.install_path(alias);

    on_stage(
        "this walk streams the checkpoint and CANNOT RESUME: a failure restarts it \
         from the beginning",
    );

    let installed = catalog::install_with_byte_progress(
        &plan,
        &dir,
        &Client::new(),
        |line| on_stage(line),
        Some(on_bytes),
    )?;
    catalog::record(&store, &installed)?;
    serde_json::to_string(&installed.model).map_err(|e| e.to_string())
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
