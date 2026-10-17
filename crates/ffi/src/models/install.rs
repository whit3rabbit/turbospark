//! Model installation pipeline: catalog downloads and Hugging Face repo streaming.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use catalog::library::{LibraryRecord, LibraryRegistry, LocationEnv, INTERNAL_LIBRARY_ID};
use catalog::{
    CancelFlag, Catalog, Client, InstallKind, InstallPlan, InstallTarget, InstalledModel, Store,
    StoreError, Verdict,
};

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
static ACTIVE_INSTALLS: Mutex<Vec<(CancelFlag, bool)>> = Mutex::new(Vec::new());

/// Signals every in-flight install walk to stop. Returns how many walks
/// were running; `ts_install_cancel` publishes that as its verdict.
pub(crate) fn cancel_active_installs() -> usize {
    let active = ACTIVE_INSTALLS.lock().unwrap_or_else(|p| p.into_inner());
    for (flag, _) in active.iter() {
        flag.cancel();
    }
    active.len()
}

/// Refuses a model-store move while any install walk (text, image or audio)
/// is registered. A walk resolves its destination root once and keeps
/// writing there, so a relocation that deletes the old tree under it would
/// fail the walk or publish into a directory that was just removed.
pub(crate) fn refuse_while_installing() -> Result<(), String> {
    refuse_while_installing_for("move the model store")
}

/// [`refuse_while_installing`] for another change that must not race a walk,
/// worded for it: `action` completes "cannot ... while N install(s) are
/// running". Library changes use it (4.1), so one registry of running walks
/// answers for both. A running model move (4.4) refuses the same changes,
/// worded for the move.
pub(crate) fn refuse_while_installing_for(action: &str) -> Result<(), String> {
    // Installs first, then the move: the one lock order every caller uses.
    let installs = ACTIVE_INSTALLS.lock().unwrap_or_else(|p| p.into_inner());
    let moving = ACTIVE_MOVE.lock().unwrap_or_else(|p| p.into_inner());
    match busy_reason(installs.len(), moving.is_some(), action) {
        None => Ok(()),
        Some(why) => Err(why),
    }
}

/// Why `action` cannot start while `installs` walks run or a model move
/// does, in the one wording every busy refusal shares. `None` when nothing
/// runs.
fn busy_reason(installs: usize, moving: bool, action: &str) -> Option<String> {
    if installs > 0 {
        Some(format!(
            "cannot {action} while {installs} install(s) are running; \
             wait for them to finish or cancel them first"
        ))
    } else if moving {
        Some(format!(
            "cannot {action} while a model move is running; \
             wait for it to finish or cancel it first"
        ))
    } else {
        None
    }
}

/// The cancel flag of the model move in flight, if any (4.4).
///
/// One move at a time, and apart from [`ACTIVE_INSTALLS`] on purpose:
/// `ts_install_cancel`, `ts_install_pause` and `ts_installs_finished` speak of
/// install walks, and cancelling an install must not end a move. The two
/// registries still answer for each other's busy checks. When both are locked,
/// installs come first.
static ACTIVE_MOVE: Mutex<Option<Arc<AtomicBool>>> = Mutex::new(None);

/// Refuses an install while a model move runs. A move rewrites the
/// registry and the markers of two libraries, and a marker has no guard
/// against a lost update (2.2), so an install recording a row meanwhile could
/// lose it. Checked before the install's work and not atomic against a move
/// that begins afterwards, like [`refuse_while_installing_for`]; the move's own
/// registration is atomic against installs that are already registered.
pub(crate) fn refuse_while_moving(action: &str) -> Result<(), String> {
    let moving = ACTIVE_MOVE.lock().unwrap_or_else(|p| p.into_inner());
    match busy_reason(0, moving.is_some(), action) {
        None => Ok(()),
        Some(why) => Err(why),
    }
}

/// Signals the model move in flight to stop. Returns whether one was running.
/// The move notices at its next chunk or file boundary.
pub(crate) fn cancel_active_move() -> bool {
    let moving = ACTIVE_MOVE.lock().unwrap_or_else(|p| p.into_inner());
    match moving.as_ref() {
        Some(flag) => {
            flag.store(true, Ordering::Release);
            true
        }
        None => false,
    }
}

/// Registers one model move: hands out its FRESH cancel flag, so a stale
/// cancel can never poison the next move, and guarantees deregistration on
/// every exit path, panic included.
pub(crate) struct ActiveMove {
    cancel: Arc<AtomicBool>,
}

impl ActiveMove {
    /// Registers a move, or says why `action` cannot run: an install walk or
    /// another move is running. Checked and registered under both locks, so
    /// an install that registered before this call is seen, and two moves
    /// cannot both register.
    pub(crate) fn register(action: &str) -> Result<Self, String> {
        let installs = ACTIVE_INSTALLS.lock().unwrap_or_else(|p| p.into_inner());
        let mut moving = ACTIVE_MOVE.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(why) = busy_reason(installs.len(), moving.is_some(), action) {
            return Err(why);
        }
        let cancel = Arc::new(AtomicBool::new(false));
        *moving = Some(Arc::clone(&cancel));
        Ok(Self { cancel })
    }

    /// The flag `catalog::library::move_install` polls.
    pub(crate) fn cancel_flag(&self) -> &AtomicBool {
        &self.cancel
    }
}

impl Drop for ActiveMove {
    fn drop(&mut self) {
        *ACTIVE_MOVE.lock().unwrap_or_else(|p| p.into_inner()) = None;
    }
}

/// These signals never wait for the worker itself. Only its next checkpoint
/// parks, so the UI remains responsive while an HTTP chunk is in flight.
pub(crate) fn pause_active_installs() -> usize {
    let active = ACTIVE_INSTALLS.lock().unwrap_or_else(|p| p.into_inner());
    active
        .iter()
        .filter(|(flag, pausable)| *pausable && flag.pause())
        .count()
}

pub(crate) fn resume_active_installs() -> usize {
    let active = ACTIVE_INSTALLS.lock().unwrap_or_else(|p| p.into_inner());
    active
        .iter()
        .filter(|(flag, pausable)| *pausable && flag.resume())
        .count()
}

/// Registers one walk: hands out its cancel flag and guarantees
/// deregistration on every exit path, panic included.
pub(crate) struct ActiveInstall {
    flag: CancelFlag,
}

impl ActiveInstall {
    pub(crate) fn register() -> Self {
        Self::register_with_pause(true)
    }

    pub(crate) fn register_image() -> Self {
        // Image packing has separate controls. A text download's Pause
        // button must not silently park an unrelated image install.
        Self::register_with_pause(false)
    }

    fn register_with_pause(pausable: bool) -> Self {
        let flag = CancelFlag::new();
        ACTIVE_INSTALLS
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push((flag.clone(), pausable));
        Self { flag }
    }

    pub(crate) fn cancel_flag(&self) -> &CancelFlag {
        &self.flag
    }
}

impl Drop for ActiveInstall {
    fn drop(&mut self) {
        let mut active = ACTIVE_INSTALLS.lock().unwrap_or_else(|p| p.into_inner());
        active.retain(|(f, _)| !f.same_flag(&self.flag));
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

/// `model`, as the install entry points answer: every key the row has, and the
/// library keys the installed listing adds to its rows (4.2), in snake_case
/// like the row: `library` (the library's id, "internal" for This Mac, whose
/// own row names none), `library_name` and `install_id`. The name is raw, as
/// in every listing, because it can come from a drive anyone could have
/// written.
pub(crate) fn installed_value(
    model: &InstalledModel,
    library_name: &str,
) -> Result<serde_json::Value, String> {
    let mut value = serde_json::to_value(model).map_err(|e| e.to_string())?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| "an installed row is not an object".to_string())?;
    object.insert(
        "library".to_string(),
        serde_json::json!(model.library.as_deref().unwrap_or(INTERNAL_LIBRARY_ID)),
    );
    object.insert("library_name".to_string(), serde_json::json!(library_name));
    object.insert(
        "install_id".to_string(),
        serde_json::json!(model.install_id()),
    );
    Ok(value)
}

/// The name of the library `model` was installed into, for [`installed_value`],
/// when the installer did not hand it back: This Mac's, or the list's, or the
/// id when the library was removed since.
pub(crate) fn library_name_of(store: &Store, model: &InstalledModel) -> String {
    match model
        .library
        .as_deref()
        .filter(|id| *id != INTERNAL_LIBRARY_ID)
    {
        None => LibraryRecord::internal(store.root()).name,
        Some(id) => LibraryRegistry::load(store.root())
            .libraries()
            .into_iter()
            .find(|record| record.id == id)
            .map_or_else(|| id.to_string(), |record| record.name),
    }
}

/// Installs the catalog row named `alias` into `library`, This Mac for
/// `None`. Where it lands is settled before the first request: a library that
/// is not connected, one a newer TurboSpark wrote, or an alias another library
/// holds fails here, before anything is downloaded.
///
/// **IMMUTABLE-REVISION NETWORK RANGES RESUME.** Completed, SHA-256 checked
/// ranges survive a failed or cancelled walk. Conversion and repacking still
/// restart because their output is published atomically rather than exposed
/// as a partially usable install.
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
    library: Option<&str>,
    mut on_stage: impl FnMut(&str),
    on_bytes: Arc<dyn Fn(u64) + Send + Sync>,
) -> Result<String, StoreError> {
    let active = ActiveInstall::register();
    let cancel = active.flag.clone();
    let catalog = Catalog::embedded().map_err(StoreError::Io)?;
    let entry = catalog
        .get(alias)
        .ok_or_else(|| StoreError::Io(format!("no catalog row named {alias:?}")))?;
    let plan = InstallPlan::from_entry(entry);
    let store = Store::default_store().map_err(StoreError::Io)?;
    let env = LocationEnv::system(store.root()).map_err(StoreError::Io)?;
    let target = catalog::install_destination(&store, library, InstallKind::Text, alias, &env)?;

    on_stage(
        "failed or cancelled immutable-revision downloads reuse completed ranges; \
         conversion restarts, and pause keeps in-memory progress",
    );

    let installed = catalog::install_text_into(
        &plan,
        &target,
        &Client::new(),
        |line| on_stage(line),
        Some(on_bytes),
        Some(&cancel),
        &store,
        &env,
    )?;
    result_json(&installed.model, &target)
}

/// Probes and installs an arbitrary Hugging Face model repository into
/// `library`, This Mac for `None`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn install_repo(
    repo: &str,
    alias: &str,
    file: Option<&str>,
    sidecar_repo: Option<&str>,
    library: Option<&str>,
    mut on_stage: impl FnMut(&str),
    on_bytes: Arc<dyn Fn(u64) + Send + Sync>,
) -> Result<String, StoreError> {
    let active = ActiveInstall::register();
    let cancel = active.flag.clone();
    let weights = parse_repo(repo).map_err(StoreError::Io)?;
    let sidecars = match sidecar_repo {
        Some(text) => parse_repo(text).map_err(StoreError::Io)?,
        None => weights.clone(),
    };
    // Before the probe, so a refused destination costs no network request.
    let store = Store::default_store().map_err(StoreError::Io)?;
    let env = LocationEnv::system(store.root()).map_err(StoreError::Io)?;
    let target = catalog::install_destination(&store, library, InstallKind::Text, alias, &env)?;
    let client = Client::new();
    let report =
        catalog::probe(&client, &weights, file, Some(&sidecars)).map_err(StoreError::Io)?;
    if !report.verdict.is_runnable() {
        return Err(StoreError::Io(match report.verdict {
            Verdict::Refused(why) => format!("model would not run here: {why}"),
            Verdict::Runnable => unreachable!(),
        }));
    }
    let plan = InstallPlan::from_probe(alias, &report, sidecars);
    if target.dir.join("manifest.json").is_file() {
        return Err(StoreError::Io(format!(
            "{} already holds an install",
            target.dir.display()
        )));
    }
    on_stage(
        "failed or cancelled immutable-revision downloads reuse completed ranges; \
         conversion restarts, and pause keeps in-memory progress",
    );
    let installed = catalog::install_text_into(
        &plan,
        &target,
        &client,
        |line| on_stage(line),
        Some(on_bytes),
        Some(&cancel),
        &store,
        &env,
    )?;
    result_json(&installed.model, &target)
}

/// The answer of a text install: the row recorded, with the library keys.
fn result_json(model: &InstalledModel, target: &InstallTarget) -> Result<String, StoreError> {
    installed_value(model, &target.library_name)
        .and_then(|value| serde_json::to_string(&value).map_err(|e| e.to_string()))
        .map_err(StoreError::Io)
}

#[cfg(test)]
mod result_tests {
    use super::*;
    use catalog::ModelModality;

    fn row(library: Option<&str>) -> InstalledModel {
        InstalledModel {
            alias: "qwen".to_string(),
            repo: "owner/qwen".to_string(),
            revision: "main".to_string(),
            path: std::path::PathBuf::from("/Volumes/T7/TurboSpark/text/qwen.gturbo"),
            family: "qwen3".to_string(),
            install_bytes: 4,
            installed_on: "2026-10-16".to_string(),
            status: "runs".to_string(),
            kind: None,
            modality: ModelModality::Text,
            variant: None,
            library: library.map(str::to_string),
            relative_path: library.map(|_| std::path::PathBuf::from("text/qwen.gturbo")),
        }
    }

    /// The answer of an install keeps every key the row has and adds the
    /// library keys of the installed listing. A row of This Mac names no
    /// library of its own, and is told as "internal".
    #[test]
    fn an_install_answer_names_its_library_beside_the_row() {
        let this_mac = installed_value(&row(None), "This Mac").unwrap();
        assert_eq!(this_mac["library"], "internal");
        assert_eq!(this_mac["library_name"], "This Mac");
        assert_eq!(this_mac["install_id"], "internal:text/qwen");
        assert_eq!(this_mac["alias"], "qwen");
        assert!(this_mac.get("relative_path").is_none(), "{this_mac}");

        let id = "0f6c2a3e-1b7d-4c55-9e0a-7d2f1c3b4a59";
        let in_library = installed_value(&row(Some(id)), "T7").unwrap();
        assert_eq!(in_library["library"], id);
        assert_eq!(in_library["library_name"], "T7");
        assert_eq!(in_library["install_id"], format!("{id}:text/qwen"));
        assert_eq!(in_library["relative_path"], "text/qwen.gturbo");
        // The name stays raw: display escapes it, as for every listing.
        let hostile = installed_value(&row(Some(id)), "A\u{1b}[31m").unwrap();
        assert_eq!(hostile["library_name"], "A\u{1b}[31m");
    }

    /// The name an audio install answers with comes from the list: This Mac's
    /// for a row with no library, the library's, and for a library the list no
    /// longer has, its id rather than nothing.
    #[test]
    fn the_library_name_of_a_row_comes_from_the_list() {
        use std::os::unix::fs::MetadataExt;

        /// Removes its folder when the test ends, passing or not.
        struct Cleanup(std::path::PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let base = std::env::temp_dir().join(format!(
            "turbospark-ffi-install-name-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("Volumes").join("T7")).unwrap();
        std::fs::create_dir_all(base.join("home")).unwrap();
        let _cleanup = Cleanup(base.clone());
        // Canonical, because the temp dir is behind the /var symlink and the
        // location rules compare canonical paths.
        let base = base.canonicalize().unwrap();
        let volume = base.join("Volumes").join("T7");
        let table = format!(
            "/dev/disk3s1s1 on / (apfs, sealed, local, read-only)\n\
             /dev/disk20s1 on {} (hfs, local, journaled)\n",
            volume.display()
        );
        let env = LocationEnv {
            mount_source: catalog::library::MountSource::Fixed(
                catalog::library::parse_mount_table(&table),
            ),
            volumes_root: base.join("Volumes"),
            home_device: std::fs::metadata(&base).unwrap().dev().wrapping_add(1),
        };
        let home = base.join("home");
        let id = LibraryRegistry::load(&home)
            .add(&volume, Some("T7"), &env)
            .unwrap()
            .view
            .record
            .id;
        let store = Store::new(&home);

        assert_eq!(library_name_of(&store, &row(None)), "This Mac");
        assert_eq!(
            library_name_of(&store, &row(Some(INTERNAL_LIBRARY_ID))),
            "This Mac"
        );
        assert_eq!(library_name_of(&store, &row(Some(&id))), "T7");
        let gone = "5a1d9c70-3e2b-4f68-8a14-6b0c2d9e7f31";
        assert_eq!(library_name_of(&store, &row(Some(gone))), gone);
    }
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
            vision_from_bytes: false,
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
        // Pause is scoped to text installs; image packing has its own UI.
        let image = ActiveInstall::register_image();
        assert_eq!(unsafe { crate::ts_install_pause() }, 0);
        assert_eq!(unsafe { crate::ts_install_resume() }, 0);
        // A store move must refuse while any walk is registered, image
        // walks included, and succeed to proceed once they are gone.
        assert!(refuse_while_installing().unwrap_err().contains("install"));
        drop(image);
        assert!(refuse_while_installing().is_ok());

        let before = installs_finished();
        let active = ActiveInstall::register();
        assert_eq!(unsafe { crate::ts_install_pause() }, 1);
        let worker_flag = active.flag.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || tx.send(worker_flag.checkpoint()).unwrap());
        let premature = rx.recv_timeout(std::time::Duration::from_millis(100));
        assert_eq!(unsafe { crate::ts_install_resume() }, 1);
        assert!(rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap()
            .is_ok());
        worker.join().unwrap();
        assert!(
            premature.is_err(),
            "the ABI pause must park the registered walk"
        );
        assert_eq!(unsafe { crate::ts_install_pause() }, 1);
        assert_eq!(cancel_active_installs(), 1, "one walk is in flight");
        assert_eq!(
            unsafe { crate::ts_install_resume() },
            0,
            "resume cannot undo cancellation"
        );
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
