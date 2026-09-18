//! Model installation pipeline: catalog downloads and Hugging Face repo streaming.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use catalog::{CancelFlag, Catalog, Client, InstallPlan, Store, Verdict};

use super::probe::parse_repo;

/// The install-progress event kinds. Mirrored in `turbospark.h`.
pub const TS_INSTALL_STAGE: i32 = 0;
pub const TS_INSTALL_BYTES: i32 = 1;

/// The cancel flag of every in-flight walk.
///
/// A walk registers a FRESH flag at entry and drops it at exit, so a stale
/// cancel can never poison the next install: "is anything running" IS
/// whether this vec is non-empty, and `cancel_active_installs` signals
/// every walk in it (in practice one -- `ts_install` blocks its caller's
/// thread -- but nothing here assumes that).
static ACTIVE_INSTALLS: Mutex<Vec<CancelFlag>> = Mutex::new(Vec::new());

/// Signals every in-flight install walk to stop. Returns how many walks
/// were running; `ts_install_cancel` publishes that as its verdict.
pub(crate) fn cancel_active_installs() -> usize {
    let active = ACTIVE_INSTALLS.lock().unwrap_or_else(|p| p.into_inner());
    for flag in active.iter() {
        flag.cancel();
    }
    active.len()
}

/// Registers one walk: hands out its cancel flag and guarantees
/// deregistration on every exit path, panic included.
pub(crate) struct ActiveInstall {
    flag: CancelFlag,
}

impl ActiveInstall {
    pub(crate) fn register() -> Self {
        let flag = CancelFlag::new();
        ACTIVE_INSTALLS
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(flag.clone());
        Self { flag }
    }

    pub(crate) fn cancel_flag(&self) -> &CancelFlag {
        &self.flag
    }
}

impl Drop for ActiveInstall {
    fn drop(&mut self) {
        let mut active = ACTIVE_INSTALLS.lock().unwrap_or_else(|p| p.into_inner());
        active.retain(|f| !f.same_flag(&self.flag));
        // In the guard, so a walk cancelled mid-stream and one that failed
        // on the network both count: the number's only reader wants to know
        // that A walk which was running has exited, whatever ended it.
        INSTALLS_FINISHED.fetch_add(1, Ordering::Relaxed);
    }
}

/// Counter bumped once per finished install walk (any outcome: success,
/// network failure, cancel), so a caller that saw `ts_install_cancel`
/// report a running walk can tell that the walk has since exited.
static INSTALLS_FINISHED: AtomicUsize = AtomicUsize::new(0);

/// How many install walks have finished since process start.
pub(crate) fn installs_finished() -> usize {
    INSTALLS_FINISHED.load(Ordering::Relaxed)
}

/// Installs the catalog row named `alias` into the store.
///
/// **THE WALK CANNOT RESUME.** It streams 4 to 25 GB and a failure restarts
/// it from the beginning, which is why the first stage line says so and why
/// a GUI must surface that before starting rather than after failing.
///
/// **THE WALK CAN NOW BE CANCELLED** (`ts_install_cancel`): the flag is
/// checked at every step boundary and inside every ranged chunk read, so a
/// cancelled walk aborts with the `install cancelled` error within seconds
/// rather than streaming on for tens of minutes. The partial directory it
/// leaves behind is a failed install's partial directory -- same as a walk
/// that died on a network error, and no more usable.
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
    let active = ActiveInstall::register();
    let cancel = active.flag.clone();
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
        Some(&cancel),
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
    let active = ActiveInstall::register();
    let cancel = active.flag.clone();
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
        Some(&cancel),
    )?;
    catalog::record(&store, &installed)?;
    serde_json::to_string(&installed.model).map_err(|e| e.to_string())
}

#[cfg(test)]
mod cancel_tests {
    use super::*;

    /// The full cancel seam, no network: a walk whose flag is ALREADY fired
    /// must return the `install cancelled` error, and a registered walk
    /// must both receive the signal and leave the registry exactly once.
    ///
    /// `install_with_byte_progress`'s entry check is what makes the first
    /// half offline: with the flag fired before the call, the walk aborts
    /// before creating a directory or touching HTTP. The sidecar list is
    /// empty so that removing the ENTRY check does not simply move the hit
    /// to the per-file check -- the mutation falls through to the
    /// tokenizer verify instead and fails with a DIFFERENT message, which
    /// is what reddens the test.
    #[test]
    fn a_pre_cancelled_walk_errors_as_cancelled_and_the_registry_stays_consistent() {
        let plan = InstallPlan {
            alias: "cancel-seam-test".to_string(),
            weights: catalog::RepoRef::new("owner/nothing", "deadbeef"),
            file: Some("model.gguf".to_string()),
            kind: catalog::SourceKind::Gguf,
            sidecars: catalog::RepoRef::new("owner/nothing", "deadbeef"),
            sidecar_files: Vec::new(),
            install_bytes: 0,
            status: "unlisted".to_string(),
            mtp: None,
            reuse_trunk_from: None,
            vision_only: false,
            vision_file: None,
            // A cancelled-at-entry walk reads no weights, so the tower
            // question never comes up; every other hand-built plan in
            // `crates/catalog` defaults this false too.
            include_vision: false,
        };
        let flag = CancelFlag::new();
        flag.cancel();
        let dir =
            std::env::temp_dir().join(format!("turbospark-cancel-seam-{}", std::process::id()));
        let error = catalog::install_with_byte_progress(
            &plan,
            &dir,
            &Client::new(),
            |_| {},
            None,
            Some(&flag),
        )
        .expect_err("a pre-cancelled walk must not run");
        assert_eq!(error, catalog::INSTALL_CANCELLED, "got: {error}");
        assert!(
            !dir.join("manifest.json").exists(),
            "a cancelled walk must not have produced an install"
        );

        // Registry mechanics: register one walk, signal it, and watch the
        // guard deregister it exactly once. The finished counter is what
        // the GUI's cancel path polls to learn the walk actually exited.
        let before = installs_finished();
        let active = ActiveInstall::register();
        assert_eq!(cancel_active_installs(), 1, "one walk is in flight");
        assert!(
            active.flag.is_cancelled(),
            "the signal must reach the walk's own flag"
        );
        drop(active);
        assert_eq!(
            installs_finished(),
            before + 1,
            "exiting must bump the counter"
        );
        assert_eq!(
            cancel_active_installs(),
            0,
            "the exited walk must have deregistered itself"
        );
    }
}
