//! Model library C ABI entry points: list, add, remove, set the default,
//! discover a library on a volume, forget an install, and import a moved store.
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
use std::os::raw::{c_char, c_int};
use std::path::Path;

use catalog::library::{
    discover_at, AddedLibrary, ImportedStore, LibraryError, LibraryMarker, LibraryRegistry,
    LibraryView, LoadReport, LocationEnv, LocationRefusal, Refreshed,
};
use catalog::{Store, StoreError};
use serde::Serialize;

use crate::abi::{self, guard_result};
use crate::models;
use crate::strings;

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

/// A change to the libraries is refused while any install walk runs: a walk
/// resolves its destination once and keeps writing there, so a library
/// removed or imported under it would strand what it writes. Checked before
/// any work; not atomic against an install that begins afterwards, so the app
/// must also stop queueing installs while it changes libraries. It is
/// `TS_ERR_BUSY`, because waiting for the walk and retrying can succeed.
fn refuse_while_installing() -> Result<(), Fault> {
    models::refuse_while_installing_for("change model libraries")
        .map_err(|why| (abi::TS_ERR_BUSY, format!("busy: {why}")))
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
fn store_fault(error: &StoreError) -> Fault {
    let code = match error {
        StoreError::NotConnected { .. } => "not_connected",
        StoreError::LibraryDisconnected { .. } => "library_disconnected",
        StoreError::LibraryReadOnly { .. } => "library_read_only",
        StoreError::HomeOnly => "home_only",
        StoreError::Io(_) => "io",
    };
    coded(code, &error.to_string())
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
            (StoreError::Io("x".to_string()), "io"),
        ];
        for (error, code) in &store {
            let (status, message) = store_fault(error);
            assert_eq!(status, abi::TS_ERR_INVALID_ARGUMENT);
            assert!(message.starts_with(&format!("{code}: ")), "{message}");
        }
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
