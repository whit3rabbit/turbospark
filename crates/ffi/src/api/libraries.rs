//! Model library C ABI entry points: list, add, remove, set the default,
//! discover a library on a volume, forget an install, move an install between
//! libraries, and import a moved store.
//!
//! `catalog::library` decides every rule. This file resolves the home and the
//! live location environment, serializes each outcome, and spells each
//! refusal as a stable lowercase code ahead of a sentence, `"<code>: <human
//! sentence>"`, so the app can localize by code and only log the sentence.
//!
//! The wire is camelCase like the rest of this crate's own JSON. Library and
//! marker names, aliases, and copy-failure reasons can come from a drive
//! anyone could have written. Sentences are escaped and capped here, because
//! they are made to be shown. Names and aliases are left as they are, because
//! they are identities, and the app truncates them for display.

use std::io::Write;
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;

use catalog::library::{
    discover_at, move_install, AddedLibrary, ImportedStore, LibraryError, LibraryMarker,
    LibraryRegistry, LibraryView, LoadReport, LocationEnv, LocationRefusal, MoveError, MoveOutcome,
    Refreshed, INTERNAL_LIBRARY_ID,
};
use catalog::{Store, StoreError};
use serde::Serialize;

use crate::abi::{self, guard_result};
use crate::models::{self, TS_INSTALL_BYTES};
use crate::strings;
use crate::TsInstallCallback;

/// The most characters of an error sentence. Marker errors and copy-failure
/// reasons echo untrusted values and paths.
pub(super) const MESSAGE_LIMIT: usize = 300;

/// A status code and the message the thread's error slot keeps.
type Fault = (c_int, String);

// ------------------------------------------------------------------ wire shapes

/// One library as the app lists it. `name` is the English literal for This
/// Mac, so the app shows its own label when `isInternal`. `root` is `null`
/// while the library is not connected.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct LibraryJson<'a> {
    id: &'a str,
    name: &'a str,
    root: Option<String>,
    available: bool,
    duplicate: bool,
    model_count: usize,
    is_default: bool,
    last_seen: Option<&'a str>,
    is_internal: bool,
}

impl<'a> From<&'a LibraryView> for LibraryJson<'a> {
    fn from(view: &'a LibraryView) -> Self {
        Self {
            id: &view.record.id,
            name: &view.record.name,
            root: view
                .root
                .as_ref()
                .map(|root| root.to_string_lossy().into_owned()),
            available: view.available,
            duplicate: view.duplicate,
            model_count: view.model_count,
            is_default: view.record.default_for_downloads,
            last_seen: view.record.last_seen.as_deref(),
            is_internal: view.record.is_internal(),
        }
    }
}

/// The result of adding: the library, and the aliases of marker rows that
/// were not recorded because the alias already belongs to another library.
#[derive(Serialize)]
struct AddedJson<'a> {
    library: LibraryJson<'a>,
    skipped: &'a [String],
}

/// The result of importing a moved store. `copyFailures` leaves the
/// conversion pending: the copies are retried only when the import is called
/// again.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ImportedJson<'a> {
    library: LibraryJson<'a>,
    skipped: &'a [String],
    custom: &'a [String],
    needs_adoption: &'a [String],
    copied_to_home: &'a [String],
    copy_failures: Vec<CopyFailureJson<'a>>,
    restored_catalog_override: bool,
}

#[derive(Serialize)]
struct CopyFailureJson<'a> {
    key: &'a str,
    reason: String,
}

impl<'a> From<&'a ImportedStore> for ImportedJson<'a> {
    fn from(imported: &'a ImportedStore) -> Self {
        Self {
            library: LibraryJson::from(&imported.view),
            skipped: &imported.skipped,
            custom: &imported.custom,
            needs_adoption: &imported.needs_adoption,
            copied_to_home: &imported.copied_to_home,
            copy_failures: imported
                .copy_failures
                .iter()
                .map(|(key, reason)| CopyFailureJson {
                    key,
                    reason: clean(reason, MESSAGE_LIMIT),
                })
                .collect(),
            restored_catalog_override: imported.restored_catalog_override,
        }
    }
}

/// What `ts_library_discover_json` says of the marker it found.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct MarkerJson<'a> {
    id: &'a str,
    name: &'a str,
    created_on: &'a str,
    model_count: usize,
    read_only: bool,
}

// ------------------------------------------------------------------ entry points

/// Every library, This Mac first, after locating each one.
#[no_mangle]
pub unsafe extern "C" fn ts_libraries_json(out: *mut *mut c_char) -> c_int {
    guard_result(|| {
        require_output(out)?;
        let (store, env) = context()?;
        let mut registry = LibraryRegistry::load(store.root());
        // `load` may have set an unparseable list aside, which the refresh's
        // own re-read would no longer report.
        let mut notes = report_notes(registry.load_report());
        let Refreshed {
            views,
            persist_error,
        } = registry.refresh(&env);
        for note in report_notes(registry.load_report()) {
            if !notes.contains(&note) {
                notes.push(note);
            }
        }
        if let Some(error) = &persist_error {
            notes.push(format!(
                "could not record when each library was last seen: {}",
                describe(error, "the location").1
            ));
        }
        log_notes(&notes);
        let rows: Vec<LibraryJson> = views.iter().map(LibraryJson::from).collect();
        emit_json(&rows, out)
    })
}

/// Adds the folder at `path` as a library, or adopts the library already
/// there with its models.
#[no_mangle]
pub unsafe extern "C" fn ts_library_add(
    path: *const c_char,
    name: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        // Before the library is added: a result that cannot reach the caller
        // would leave a change nobody was told about.
        require_output(out)?;
        let path = strings::required(path, "path").map_err(invalid)?;
        let name = strings::optional(name, "name").map_err(invalid)?;
        refuse_while_installing()?;
        let (store, env) = context()?;
        let mut registry = LibraryRegistry::load(store.root());
        let AddedLibrary { view, skipped } = registry
            .add(Path::new(path), name, &env)
            .map_err(|error| library_fault(&error, path))?;
        emit_json(
            &AddedJson {
                library: LibraryJson::from(&view),
                skipped: &skipped,
            },
            out,
        )
    })
}

/// Stops listing the library `id` and its models. Never touches a file.
#[no_mangle]
pub unsafe extern "C" fn ts_library_remove(id: *const c_char) -> c_int {
    guard_result(|| {
        let id = strings::required(id, "id").map_err(invalid)?;
        refuse_while_installing()?;
        let (store, _) = context()?;
        LibraryRegistry::load(store.root())
            .remove(id)
            .map_err(|error| library_fault(&error, "the location"))
    })
}

/// Makes `id` the only default for new downloads. "internal" is This Mac.
#[no_mangle]
pub unsafe extern "C" fn ts_library_set_default(id: *const c_char) -> c_int {
    guard_result(|| {
        let id = strings::required(id, "id").map_err(invalid)?;
        refuse_while_installing()?;
        let (store, _) = context()?;
        LibraryRegistry::load(store.root())
            .set_default(id)
            .map_err(|error| library_fault(&error, "the location"))
    })
}

/// The library marker in the standard folder at the top of the volume
/// `volume_root`, as a summary object, or `null` when there is none. Reads
/// one file, so connecting a drive never walks its contents.
#[no_mangle]
pub unsafe extern "C" fn ts_library_discover_json(
    volume_root: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        require_output(out)?;
        let volume = strings::required(volume_root, "volumeRoot").map_err(invalid)?;
        match discover_at(Path::new(volume)) {
            Some(marker) => emit_json(&summary(&marker), out),
            None => strings::emit("null", out).map_err(invalid),
        }
    })
}

/// Stops listing the install `install_id` (`<library>:<modality>/<alias>`).
/// Deletes no file. An install that is not listed is not an error.
#[no_mangle]
pub unsafe extern "C" fn ts_model_forget(install_id: *const c_char) -> c_int {
    guard_result(|| {
        let install_id = strings::required(install_id, "installId").map_err(invalid)?;
        refuse_while_installing()?;
        let (store, env) = context()?;
        store
            .forget_install(install_id, &env)
            .map_err(|error| store_fault(&error))
    })
}

/// Moves the install `install_id` (`<library>:<modality>/<alias>`) into the
/// library `library_id` ("internal" is This Mac): copies it, checks the copy,
/// lists it in its new library, and only then deletes the original.
/// `catalog::library::move_install` applies every rule and guarantees that a
/// refusal, a failure or a cancel leaves the original where it was.
///
/// `free_bytes` is the destination's free space as the app measured it. A
/// negative value means it could not be measured, and the move is refused as
/// `free_space_unknown`, as the command line refuses when `df` cannot be read:
/// the free-space check is the only one, so a guess would defeat it.
///
/// Progress reaches `cb` as `TS_INSTALL_BYTES` (`done` of `total`, first
/// `0` of `total`) on the calling thread. No stage lines are sent. The move is
/// registered with the running installs: it is refused as `busy` while an
/// install or another move runs, and while it runs library changes and
/// installs are refused.
#[no_mangle]
pub unsafe extern "C" fn ts_model_move(
    install_id: *const c_char,
    library_id: *const c_char,
    free_bytes: i64,
    cb: TsInstallCallback,
    userdata: *mut c_void,
    result_json: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        // Before the move: a result that cannot reach the caller would leave
        // a move nobody was told about.
        require_output(result_json)?;
        let install_id = strings::required(install_id, "installId").map_err(invalid)?;
        let library_id = strings::required(library_id, "libraryId").map_err(invalid)?;
        let free = u64::try_from(free_bytes).map_err(|_| {
            coded(
                "free_space_unknown",
                "the free space of the destination could not be measured, so nothing was moved",
            )
        })?;
        // Registered before anything is read, so a library change that comes
        // after this point is refused until the move is over.
        let active = models::ActiveMove::register("move a model").map_err(busy)?;
        let (store, env) = context()?;
        require_known_libraries(&store, install_id, library_id)?;
        let progress = |done: u64, total: u64| {
            if let Some(report) = cb {
                // SAFETY: the caller's own callback and userdata, called on
                // this thread while the call is open.
                unsafe { report(userdata, TS_INSTALL_BYTES, std::ptr::null(), 0, done, total) };
            }
        };
        let outcome = move_install(
            &store,
            install_id,
            library_id,
            &env,
            free,
            &progress,
            active.cancel_flag(),
        )
        .map_err(|error| move_fault(&error))?;
        emit_json(
            &outcome_json(&outcome, &install_id_in(install_id, library_id)),
            result_json,
        )
    })
}

/// Stops the model move in flight at its next file or chunk boundary, which
/// then answers `cancelled` with the original untouched and no partial copy.
/// Returns 1 when a move was running and has been signalled, 0 when none was,
/// which is not an error. A cancel that comes after the copy was published
/// cannot undo the move, and it ends `moved`.
#[no_mangle]
pub unsafe extern "C" fn ts_model_move_cancel() -> c_int {
    abi::guard_value(0, || c_int::from(models::cancel_active_move()))
}

/// Registers the `models` folder of a store Move Store moved to `root` as a
/// library, keeping every model where it is, and copies its embedding and OCR
/// models into the home.
#[no_mangle]
pub unsafe extern "C" fn ts_store_import_moved_root(
    root: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        require_output(out)?;
        let root = strings::required(root, "root").map_err(invalid)?;
        refuse_while_installing()?;
        let (store, env) = context()?;
        let imported = LibraryRegistry::load(store.root())
            .import_moved_root(Path::new(root), &env)
            .map_err(|error| library_fault(&error, root))?;
        emit_json(&ImportedJson::from(&imported), out)
    })
}

// ---------------------------------------------------------------------- helpers

fn invalid(message: String) -> Fault {
    (abi::TS_ERR_INVALID_ARGUMENT, message)
}

/// The out-pointer, checked before any work, as `ts_install` checks its own.
fn require_output(out: *mut *mut c_char) -> Result<(), Fault> {
    if out.is_null() {
        return Err(invalid("output pointer must not be null".to_string()));
    }
    Ok(())
}

/// The store at the default root, and the live location environment for it.
/// Debug builds honor the `TURBOSPARK_TEST_*` overrides inside the latter.
fn context() -> Result<(Store, LocationEnv), Fault> {
    let store = Store::default_store().map_err(|e| coded("io", &e))?;
    let env = LocationEnv::system(store.root()).map_err(|e| coded("io", &e))?;
    Ok((store, env))
}

/// A change to the libraries is refused while any install walk or model move
/// runs: a walk resolves its destination once and keeps writing there, so a
/// library removed or imported under it would strand what it writes. Checked
/// before any work; not atomic against an install that begins afterwards, so
/// the app must also stop queueing installs while it changes libraries. It is
/// `TS_ERR_BUSY`, because waiting for the walk and retrying can succeed.
fn refuse_while_installing() -> Result<(), Fault> {
    models::refuse_while_installing_for("change model libraries").map_err(busy)
}

/// An install is refused while a model move runs (4.4), for the move's sake:
/// both rewrite the registry and library markers, which keep no guard against
/// a lost update. Checked before the install's work, and not atomic against a
/// move that begins afterwards, as for [`refuse_while_installing`].
pub(super) fn refuse_while_moving() -> Result<(), Fault> {
    models::refuse_while_moving("install a model").map_err(busy)
}

/// The `busy` refusal: `TS_ERR_BUSY`, for work that waiting can clear.
fn busy(why: String) -> Fault {
    (abi::TS_ERR_BUSY, format!("busy: {why}"))
}

/// Refuses a move naming a library the list does not have, before the move
/// reads anything: the destination as `unknown_library`, as for an install,
/// and the library of the install id as `unknown_install`, since no install
/// can be in a library that is gone. Both come from the registry, never from
/// the wording of a later error. This Mac is always listed.
fn require_known_libraries(store: &Store, install_id: &str, library_id: &str) -> Result<(), Fault> {
    let listed = LibraryRegistry::load(store.root()).libraries();
    let known = |id: &str| id == INTERNAL_LIBRARY_ID || listed.iter().any(|record| record.id == id);
    if !known(library_id) {
        return Err(library_fault(
            &LibraryError::UnknownLibrary(library_id.to_string()),
            "the location",
        ));
    }
    if install_id
        .split_once(':')
        .is_some_and(|(source, _)| !known(source))
    {
        return Err(move_fault(&MoveError::UnknownInstall(
            install_id.to_string(),
        )));
    }
    Ok(())
}

/// The id of the install `install_id` once it is in `library_id`: the same
/// `<modality>/<alias>` behind the destination's id. The move has already
/// checked the id's form.
fn install_id_in(install_id: &str, library_id: &str) -> String {
    let rest = install_id
        .split_once(':')
        .map_or(install_id, |(_, rest)| rest);
    format!("{library_id}:{rest}")
}

/// How a move that put the install in its destination ended, with the
/// install's new id and the fields of the outcome. `staleMarker` names the
/// source library whose marker could not be rewritten and still lists the
/// install, which is harmless; `path` is the original's folder when it could
/// not be deleted.
fn outcome_json(outcome: &MoveOutcome, install_id: &str) -> serde_json::Value {
    match outcome {
        MoveOutcome::Moved => serde_json::json!({
            "outcome": "moved",
            "installId": install_id,
        }),
        MoveOutcome::SourceMarkerStale(library) => serde_json::json!({
            "outcome": "source_marker_stale",
            "installId": install_id,
            "staleMarker": library,
        }),
        MoveOutcome::SourceCleanupFailed { path, stale_marker } => serde_json::json!({
            "outcome": "source_cleanup_failed",
            "installId": install_id,
            "path": path.to_string_lossy(),
            "staleMarker": stale_marker,
        }),
    }
}

fn emit_json<T: Serialize>(value: &T, out: *mut *mut c_char) -> Result<(), Fault> {
    let json = serde_json::to_string(value).map_err(|e| (abi::TS_ERR_JSON, e.to_string()))?;
    // SAFETY: `out` was checked non-null, and the caller promises it points
    // to a writable `char *`, as for every out-parameter in this crate.
    unsafe { strings::emit(&json, out) }.map_err(invalid)
}

fn summary(marker: &LibraryMarker) -> MarkerJson<'_> {
    MarkerJson {
        id: &marker.id,
        name: &marker.name,
        created_on: &marker.created_on,
        model_count: marker.models.len(),
        read_only: marker.is_read_only(),
    }
}

/// A refusal: the stable `code`, then the sentence.
pub(super) fn coded(code: &str, sentence: &str) -> Fault {
    (
        abi::TS_ERR_INVALID_ARGUMENT,
        format!("{code}: {}", clean(sentence, MESSAGE_LIMIT)),
    )
}

/// The fault for a failed library operation. `place` names the location an
/// add or an import was about.
fn library_fault(error: &LibraryError, place: &str) -> Fault {
    let (code, sentence) = describe(error, place);
    coded(code, &sentence)
}

/// The fault for a failed change to an install.
pub(super) fn store_fault(error: &StoreError) -> Fault {
    let code = match error {
        StoreError::NotConnected { .. } => "not_connected",
        StoreError::LibraryDisconnected { .. } => "library_disconnected",
        StoreError::LibraryReadOnly { .. } => "library_read_only",
        StoreError::HomeOnly => "home_only",
        StoreError::AliasElsewhere { .. } => "alias_elsewhere",
        StoreError::Io(_) => "io",
    };
    coded(code, &error.to_string())
}

/// The fault for a refused or stopped move (4.4). Every `MoveError` has its
/// own stable code, and the match names them all with no wildcard, so a
/// variant added to the catalog is a compile error here until it has a code.
/// `cancelled` is `TS_ERR_CANCELLED` and `busy` is `TS_ERR_BUSY`; the rest are
/// `TS_ERR_INVALID_ARGUMENT`. A drive lost during the move is `not_connected`,
/// as the catalog reports it: there is no `library_disconnected` for a move.
fn move_fault(error: &MoveError) -> Fault {
    let invalid = abi::TS_ERR_INVALID_ARGUMENT;
    let (status, code) = match error {
        MoveError::Busy(_) => (abi::TS_ERR_BUSY, "busy"),
        MoveError::InsufficientSpace { .. } => (invalid, "insufficient_space"),
        MoveError::NotConnected(_) => (invalid, "not_connected"),
        MoveError::LibraryReadOnly(_) => (invalid, "library_read_only"),
        MoveError::HomeOnly => (invalid, "home_only"),
        MoveError::SameLibrary => (invalid, "same_library"),
        MoveError::UnknownInstall(_) => (invalid, "unknown_install"),
        MoveError::DestinationTaken(_) => (invalid, "destination_taken"),
        MoveError::Cancelled => (abi::TS_ERR_CANCELLED, "cancelled"),
        MoveError::VerifyFailed { .. } => (invalid, "verify_failed"),
        MoveError::Io(_) => (invalid, "io"),
    };
    // The sentence can carry a library's name or a path from a drive.
    let (_, message) = coded(code, &error.to_string());
    (status, message)
}

/// The fault for a failed install into a library (4.3). The four errors the
/// app localizes (`not_connected`, `library_disconnected`,
/// `library_read_only`, `home_only`) carry their stable code, as for a
/// forgotten install. Any other failure is what it was before libraries: its
/// own message, cancelled or a generic failure ([`abi::install_error`]), with
/// no code, so a refusal such as an alias another library holds still reads
/// as it did.
pub(super) fn install_fault(error: StoreError) -> Fault {
    match error {
        StoreError::Io(message) => abi::install_error(message),
        coded => store_fault(&coded),
    }
}

/// Reads the options of an install entry point: a JSON object whose keys are
/// all known to `T` (which denies unknown fields). Serde would also read an
/// array as the fields in order, so `["<id>"]` would be a destination and a
/// misspelt object would not be the only way to get one by accident.
pub(super) fn parse_options<T: serde::de::DeserializeOwned>(json: &str) -> Result<T, Fault> {
    let invalid = |why: String| {
        (
            abi::TS_ERR_INVALID_ARGUMENT,
            format!("options is not valid install options: {why}"),
        )
    };
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|e| invalid(e.to_string()))?;
    if !value.is_object() {
        return Err(invalid("expected a JSON object".to_string()));
    }
    serde_json::from_value(value).map_err(|e| invalid(e.to_string()))
}

/// Refuses a library `id` that the list does not have, as `unknown_library`,
/// before any install work. An install entry point takes an id the app read
/// from `ts_libraries_json`, which another window can remove meanwhile.
/// `None` and "internal" are This Mac, which is always listed. The code comes
/// from the registry, never from the wording of a later error.
pub(super) fn require_listed(library: Option<&str>) -> Result<(), Fault> {
    let Some(id) = library.filter(|id| *id != INTERNAL_LIBRARY_ID) else {
        return Ok(());
    };
    let store = Store::default_store().map_err(|e| coded("io", &e))?;
    if LibraryRegistry::load(store.root())
        .libraries()
        .iter()
        .any(|record| record.id == id)
    {
        return Ok(());
    }
    Err(library_fault(
        &LibraryError::UnknownLibrary(id.to_string()),
        "the location",
    ))
}

/// The stable code of a library error and a sentence for it. The codes are
/// the contract: the app maps them to localized text, so a variant is never
/// renamed, only added.
fn describe(error: &LibraryError, place: &str) -> (&'static str, String) {
    match error {
        LibraryError::Refused(refusal) => (refusal_code(refusal), refusal_sentence(place, refusal)),
        LibraryError::UnknownLibrary(id) => {
            ("unknown_library", format!("no library has the id {id}"))
        }
        LibraryError::Internal => ("internal_library", "This Mac cannot be removed".to_string()),
        LibraryError::Marker(error) => ("marker", error.to_string()),
        LibraryError::NotConnected(path) => (
            "not_connected",
            format!("{} is not connected", path.display()),
        ),
        LibraryError::Io(reason) => ("io", reason.clone()),
    }
}

fn refusal_code(refusal: &LocationRefusal) -> &'static str {
    match refusal {
        LocationRefusal::SameDiskAsHome => "same_disk_as_home",
        LocationRefusal::Network => "network",
        LocationRefusal::SmallFileLimit { .. } => "small_file_limit",
        LocationRefusal::ReadOnly => "read_only",
        LocationRefusal::NotADirectory => "not_a_directory",
        LocationRefusal::NestedIn { .. } => "nested",
        LocationRefusal::InsideHome => "inside_home",
        LocationRefusal::UnknownVolume => "unknown_volume",
    }
}

/// Why `place` cannot be a library, naming the rule it broke.
fn refusal_sentence(place: &str, refusal: &LocationRefusal) -> String {
    match refusal {
        LocationRefusal::SameDiskAsHome => {
            format!("{place} is on the same disk as This Mac, which already covers it")
        }
        LocationRefusal::Network => format!(
            "{place} is on a network share; libraries must be on a directly connected drive"
        ),
        LocationRefusal::SmallFileLimit { fs_type } => format!(
            "{place} is on a drive formatted as {fs_type}, which cannot store a single file \
             of 4 GiB or larger"
        ),
        LocationRefusal::ReadOnly => format!(
            "{place} is on a read-only drive; a library needs a drive TurboSpark can write to"
        ),
        LocationRefusal::NotADirectory => format!("{place} is not an existing folder"),
        // Also an unreadable mount table and a mount nested in another volume.
        LocationRefusal::UnknownVolume => format!(
            "TurboSpark could not identify this drive, so it cannot check that {place} is on \
             a directly connected drive that can hold large files"
        ),
        LocationRefusal::NestedIn { library } => {
            format!("{place} overlaps the library {library}, which already covers it")
        }
        LocationRefusal::InsideHome => {
            format!("{place} overlaps the TurboSpark home folder, which This Mac already covers")
        }
    }
}

/// What reading the list set aside, as one line each.
fn report_notes(report: &LoadReport) -> Vec<String> {
    let mut notes = Vec::new();
    if let Some(path) = &report.quarantined {
        notes.push(format!(
            "the library list could not be parsed and was set aside as {}",
            path.display()
        ));
    }
    for dropped in &report.dropped {
        notes.push(format!("ignored a library record: {dropped}"));
    }
    if let Some(unreadable) = &report.unreadable {
        notes.push(format!("could not read the library list: {unreadable}"));
    }
    notes
}

/// Bookkeeping trouble is a diagnostic, never a listing error (3.4): the
/// views are current whether or not the dates could be recorded. The write
/// ignores its own failure, because `eprintln!` panics on a closed stderr.
fn log_notes(notes: &[String]) {
    let mut stderr = std::io::stderr();
    for note in notes {
        let _ = writeln!(stderr, "warning: {}", clean(note, MESSAGE_LIMIT));
    }
}

/// `text` with control and bidirectional formatting characters escaped, cut
/// to `limit` characters. Shown raw, a name from a drive could move the
/// cursor, recolor a terminal, or reorder what the user reads.
pub(super) fn clean(text: &str, limit: usize) -> String {
    let escaped: String = text
        .chars()
        .map(|c| {
            if c.is_control() || is_bidi_control(c) {
                c.escape_default().to_string()
            } else {
                c.to_string()
            }
        })
        .collect();
    if escaped.chars().count() <= limit {
        return escaped;
    }
    let kept: String = escaped.chars().take(limit.saturating_sub(3)).collect();
    format!("{kept}...")
}

/// The explicit direction marks, embeddings, overrides, and isolates.
fn is_bidi_control(c: char) -> bool {
    matches!(
        c,
        '\u{061c}' | '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}'
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use catalog::library::{LibraryRecord, MarkerError};
    use std::path::PathBuf;

    /// Every kind has its own stable code. Swift maps these strings, so a
    /// rename here is an ABI break.
    #[test]
    fn each_location_refusal_has_its_own_stable_code() {
        let refusals = [
            (LocationRefusal::SameDiskAsHome, "same_disk_as_home"),
            (LocationRefusal::Network, "network"),
            (
                LocationRefusal::SmallFileLimit {
                    fs_type: "msdos".to_string(),
                },
                "small_file_limit",
            ),
            (LocationRefusal::ReadOnly, "read_only"),
            (LocationRefusal::NotADirectory, "not_a_directory"),
            (
                LocationRefusal::NestedIn {
                    library: "Fast".to_string(),
                },
                "nested",
            ),
            (LocationRefusal::InsideHome, "inside_home"),
            (LocationRefusal::UnknownVolume, "unknown_volume"),
        ];
        for (refusal, code) in &refusals {
            let (status, message) = library_fault(&LibraryError::Refused(refusal.clone()), "/V/X");
            assert_eq!(status, abi::TS_ERR_INVALID_ARGUMENT);
            assert!(message.starts_with(&format!("{code}: ")), "{message}");
            assert!(
                message.contains("/V/X"),
                "the sentence names the place: {message}"
            );
        }
        let sentences: Vec<String> = refusals
            .iter()
            .map(|(refusal, _)| refusal_sentence("/V/X", refusal))
            .collect();
        for (at, sentence) in sentences.iter().enumerate() {
            assert!(!sentences[at + 1..].contains(sentence), "{sentence}");
        }
        assert!(sentences[2].contains("msdos") && sentences[2].contains("4 GiB"));
        assert!(sentences[5].contains("Fast"));
    }

    #[test]
    fn library_errors_and_install_errors_carry_their_codes() {
        let library = [
            (
                LibraryError::UnknownLibrary("x".to_string()),
                "unknown_library",
            ),
            (LibraryError::Internal, "internal_library"),
            (LibraryError::Marker(MarkerError::NotAFile), "marker"),
            (
                LibraryError::NotConnected(PathBuf::from("/Volumes/Gone")),
                "not_connected",
            ),
            (LibraryError::Io("disk full".to_string()), "io"),
        ];
        for (error, code) in &library {
            let (_, message) = library_fault(error, "the location");
            assert!(message.starts_with(&format!("{code}: ")), "{message}");
        }
        // The install-side kinds the move and install entry points will share.
        let store = [
            (
                StoreError::NotConnected {
                    library: "T7".to_string(),
                },
                "not_connected",
            ),
            (
                StoreError::LibraryDisconnected {
                    library: "T7".to_string(),
                },
                "library_disconnected",
            ),
            (
                StoreError::LibraryReadOnly {
                    library: "T7".to_string(),
                },
                "library_read_only",
            ),
            (StoreError::HomeOnly, "home_only"),
            (
                StoreError::AliasElsewhere {
                    alias: "qwen".to_string(),
                    place: "This Mac".to_string(),
                },
                "alias_elsewhere",
            ),
            (StoreError::Io("x".to_string()), "io"),
        ];
        for (error, code) in &store {
            let (status, message) = store_fault(error);
            assert_eq!(status, abi::TS_ERR_INVALID_ARGUMENT);
            assert!(message.starts_with(&format!("{code}: ")), "{message}");
        }
    }

    /// Every way a move ends without moving has its own stable code, and the
    /// status the app acts on: `cancelled` is a cancel and `busy` is a retry,
    /// the rest are refusals. Swift maps these strings, so a rename here is an
    /// ABI break. The table has one row per `MoveError` variant (the mapping
    /// itself has no wildcard, so a new variant fails the build first).
    #[test]
    fn each_move_error_has_its_own_stable_code_and_status() {
        let invalid = abi::TS_ERR_INVALID_ARGUMENT;
        let table = [
            (
                MoveError::Busy("loaded".to_string()),
                abi::TS_ERR_BUSY,
                "busy",
            ),
            (
                MoveError::InsufficientSpace {
                    needed: 5_000,
                    available: 10,
                },
                invalid,
                "insufficient_space",
            ),
            (
                MoveError::NotConnected("T7".to_string()),
                invalid,
                "not_connected",
            ),
            (
                MoveError::LibraryReadOnly("T7".to_string()),
                invalid,
                "library_read_only",
            ),
            (MoveError::HomeOnly, invalid, "home_only"),
            (MoveError::SameLibrary, invalid, "same_library"),
            (
                MoveError::UnknownInstall("internal:text/qwen".to_string()),
                invalid,
                "unknown_install",
            ),
            (
                MoveError::DestinationTaken("T9".to_string()),
                invalid,
                "destination_taken",
            ),
            (MoveError::Cancelled, abi::TS_ERR_CANCELLED, "cancelled"),
            (
                MoveError::VerifyFailed {
                    expected: 10,
                    found: 9,
                    expected_files: 2,
                    found_files: 2,
                },
                invalid,
                "verify_failed",
            ),
            (MoveError::Io("disk on fire".to_string()), invalid, "io"),
        ];
        for (error, status, code) in &table {
            let (got, message) = move_fault(error);
            assert_eq!(got, *status, "{code}");
            assert!(message.starts_with(&format!("{code}: ")), "{message}");
            assert!(
                message.len() > code.len() + 2,
                "a sentence follows: {message}"
            );
        }
        let mut codes: Vec<&str> = table.iter().map(|(_, _, code)| *code).collect();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), table.len(), "two variants share a code");
        // The space refusal carries both figures, for the app to show.
        let (_, space) = move_fault(&table[1].0);
        assert!(
            space.contains("10 bytes") && space.contains("5000"),
            "{space}"
        );
    }

    /// A move's sentence echoes library names and paths from a drive: the
    /// code survives, and the sentence is escaped and capped.
    #[test]
    fn a_move_sentence_is_escaped_and_capped_behind_an_intact_code() {
        let hostile = format!("T7\u{1b}[31m\u{202e}{}", "x".repeat(1_000));
        for error in [
            MoveError::NotConnected(hostile.clone()),
            MoveError::Io(hostile),
        ] {
            let (_, message) = move_fault(&error);
            assert!(!message.contains('\u{1b}') && !message.contains('\u{202e}'));
            let (code, sentence) = message.split_once(": ").unwrap();
            assert!(code == "not_connected" || code == "io", "{message}");
            assert!(sentence.chars().count() <= MESSAGE_LIMIT, "{message}");
        }
    }

    /// Each outcome keeps the fields of its Rust variant, in camelCase, and
    /// every one names the install's new id.
    #[test]
    fn each_move_outcome_serializes_with_its_variant_fields() {
        let id = "T7:text/qwen";
        assert_eq!(
            outcome_json(&MoveOutcome::Moved, id),
            serde_json::json!({"outcome": "moved", "installId": id})
        );
        assert_eq!(
            outcome_json(&MoveOutcome::SourceMarkerStale("T9".to_string()), id),
            serde_json::json!({
                "outcome": "source_marker_stale", "installId": id, "staleMarker": "T9"
            })
        );
        let cleanup = |stale: Option<&str>| MoveOutcome::SourceCleanupFailed {
            path: PathBuf::from("/Volumes/T9/TurboSpark/text/qwen.gturbo"),
            stale_marker: stale.map(str::to_string),
        };
        assert_eq!(
            outcome_json(&cleanup(None), id),
            serde_json::json!({
                "outcome": "source_cleanup_failed", "installId": id,
                "path": "/Volumes/T9/TurboSpark/text/qwen.gturbo", "staleMarker": null
            })
        );
        assert_eq!(outcome_json(&cleanup(Some("T9")), id)["staleMarker"], "T9");
    }

    /// The new id is the same `<modality>/<alias>` behind the destination's id.
    #[test]
    fn a_moved_install_is_named_by_its_destination() {
        assert_eq!(
            install_id_in("0f6c:text/qwen", "internal"),
            "internal:text/qwen"
        );
        assert_eq!(install_id_in("internal:image/z", "5a1d"), "5a1d:image/z");
    }

    /// An install fails with one of the four library codes ahead of its
    /// sentence, and `home_only` is among them: no install entry point can
    /// return it yet, so this is where its mapping is pinned. Any other
    /// failure is what it was before libraries, with no code.
    #[test]
    fn an_install_fault_codes_the_library_errors_and_leaves_the_rest() {
        for (error, code) in [
            (
                StoreError::NotConnected {
                    library: "T7".to_string(),
                },
                "not_connected",
            ),
            (
                StoreError::LibraryDisconnected {
                    library: "T7".to_string(),
                },
                "library_disconnected",
            ),
            (
                StoreError::LibraryReadOnly {
                    library: "T7".to_string(),
                },
                "library_read_only",
            ),
            (StoreError::HomeOnly, "home_only"),
            (
                StoreError::AliasElsewhere {
                    alias: "qwen".to_string(),
                    place: "This Mac".to_string(),
                },
                "alias_elsewhere",
            ),
        ] {
            let (status, message) = install_fault(error);
            assert_eq!(status, abi::TS_ERR_INVALID_ARGUMENT, "{code}");
            assert!(message.starts_with(&format!("{code}: ")), "{message}");
        }
        assert_eq!(
            install_fault(StoreError::Io("disk full".to_string())),
            (abi::TS_ERR_GENERATE, "disk full".to_string())
        );
        assert_eq!(
            install_fault(StoreError::Io(catalog::INSTALL_CANCELLED.to_string())),
            (
                abi::TS_ERR_CANCELLED,
                catalog::INSTALL_CANCELLED.to_string()
            )
        );
    }

    /// The code is data and the sentence is shown, so only the sentence is
    /// escaped and capped, and the code always survives.
    #[test]
    fn a_sentence_is_escaped_and_capped_behind_an_intact_code() {
        let hostile = format!("a\u{1b}[31m\u{202e}b{}", "x".repeat(1_000));
        let (_, message) = coded("marker", &hostile);

        assert!(
            message.starts_with("marker: a\\u{1b}[31m\\u{202e}b"),
            "{message}"
        );
        assert!(!message.contains('\u{1b}') && !message.contains('\u{202e}'));
        let sentence = message.strip_prefix("marker: ").unwrap();
        assert_eq!(sentence.chars().count(), MESSAGE_LIMIT);
        assert!(sentence.ends_with("..."));
        assert_eq!(clean("Fast", 64), "Fast");
        assert_eq!(clean("caf\u{e9}", 64), "caf\u{e9}");
    }

    #[test]
    fn a_library_serializes_with_camel_case_keys_and_null_for_what_is_absent() {
        let record = LibraryRecord {
            id: "0f6c2a3e-1b7d-4c55-9e0a-7d2f1c3b4a59".to_string(),
            name: "T7".to_string(),
            volume_relative_path: PathBuf::from("TurboSpark"),
            last_mount_path: PathBuf::from("/Volumes/T7"),
            default_for_downloads: true,
            added_on: "2026-10-14".to_string(),
            last_seen: None,
        };
        let view = LibraryView {
            record,
            root: None,
            available: false,
            duplicate: false,
            model_count: 3,
        };

        let json = serde_json::to_value(LibraryJson::from(&view)).unwrap();

        let mut keys: Vec<&str> = json
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "available",
                "duplicate",
                "id",
                "isDefault",
                "isInternal",
                "lastSeen",
                "modelCount",
                "name",
                "root"
            ]
        );
        assert!(json["root"].is_null() && json["lastSeen"].is_null());
        assert_eq!(json["modelCount"], 3);
        assert_eq!(json["isDefault"], true);
        assert_eq!(json["isInternal"], false);
    }
}
