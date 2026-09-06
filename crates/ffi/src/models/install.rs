//! Model installation pipeline: catalog downloads and Hugging Face repo streaming.

use std::sync::Arc;

use catalog::{Catalog, Client, InstallPlan, Store, Verdict};

use super::probe::parse_repo;

/// The install-progress event kinds. Mirrored in `turbospark.h`.
pub const TS_INSTALL_STAGE: i32 = 0;
pub const TS_INSTALL_BYTES: i32 = 1;

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

/// Probes and installs an arbitrary Hugging Face model repository.
pub(crate) fn install_repo(
    repo: &str,
    alias: &str,
    file: Option<&str>,
    sidecar_repo: Option<&str>,
    mut on_stage: impl FnMut(&str),
    on_bytes: Arc<dyn Fn(u64) + Send + Sync>,
) -> Result<String, String> {
    let weights = parse_repo(repo)?;
    let sidecars = match sidecar_repo {
        Some(text) => parse_repo(text)?,
        None => weights.clone(),
    };
    let client = Client::new();
    let report = catalog::probe(&client, &weights, file, Some(&sidecars))?;
    if !report.verdict.is_runnable() {
        return Err(match report.verdict {
            Verdict::Refused(why) => format!("model would not run here: {why}"),
            Verdict::Runnable => unreachable!(),
        });
    }
    let plan = InstallPlan::from_probe(alias, &report, sidecars);
    let store = Store::default_store()?;
    let dir = store.install_path(alias);
    if dir.join("manifest.json").is_file() {
        return Err(format!("{} already holds an install", dir.display()));
    }
    on_stage(
        "this walk streams the checkpoint and CANNOT RESUME: a failure restarts it \
         from the beginning",
    );
    let installed = catalog::install_with_byte_progress(
        &plan,
        &dir,
        &client,
        |line| on_stage(line),
        Some(on_bytes),
    )?;
    catalog::record(&store, &installed)?;
    serde_json::to_string(&installed.model).map_err(|e| e.to_string())
}
