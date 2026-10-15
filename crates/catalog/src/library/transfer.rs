//! Moving one install from one library to another (7.1, 7.3, 7.4).
//!
//! A move copies the install into the destination's staging folder, checks
//! the copy against the source, publishes it with a same-volume rename,
//! points the destination's marker and then the registry at it, drops it
//! from the source's marker, and only then deletes the source. Any failure
//! or cancel before the registry points at the copy removes the copy, so
//! the source stays where it was, registered and loadable.
//!
//! Drive contents are untrusted (2.2 SECURITY). The source is walked without
//! following anything and a link or special file in it refuses the move,
//! the copy never follows a link, and the path to the source is walked
//! again, unfollowed, right before it is deleted. As for installs (2.8), std
//! has no `openat`/`O_NOFOLLOW` descent and this crate forbids unsafe code,
//! so a live local process that swaps a folder for a link between a check
//! and its use is an accepted residual race.

use super::{
    new_library_id, LibraryMarker, LibraryRecord, LibraryRegistry, LocationEnv, PortableRow,
    SharedMountTable, INTERNAL_LIBRARY_ID,
};
use crate::audio_catalog::{AudioCatalog, RECEIPT_FILENAME};
use crate::store::{connected_library, install_in, portable_modality};
use crate::{EntryKind, InstalledModel, ModelModality, Store, StoreError};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

/// How much is copied between cancel checks and progress reports, so a
/// cancel is answered within one chunk even on a slow drive. Small in tests,
/// so their small files take several chunks.
const CHUNK_BYTES: usize = if cfg!(test) { 1024 } else { 8 * 1024 * 1024 };

/// Why [`move_install`] refused a move or stopped one. Every refusal comes
/// before anything is copied, and every error leaves the source where it
/// was, still registered and loadable (7.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoveError {
    /// The install is loaded, being installed, or already being moved
    /// (7.5). [`move_install`] never returns it: the caller checks, and
    /// returns it.
    Busy(String),
    /// The destination has less free space than the install takes (7.4).
    InsufficientSpace {
        needed: u64,
        available: u64,
    },
    /// The source or destination library is not connected, or its drive
    /// went away during the move. Its name.
    NotConnected(String),
    /// A newer TurboSpark wrote the library's marker, so this build neither
    /// adds a row to it nor drops one. Its name.
    LibraryReadOnly(String),
    /// Embedding and OCR models stay on This Mac (2.5).
    HomeOnly,
    /// The install is already in the destination library.
    SameLibrary,
    /// No recorded install, and no image folder, has this id.
    UnknownInstall(String),
    /// The destination library already holds an install with this alias
    /// and modality. Its name.
    DestinationTaken(String),
    Cancelled,
    /// The copy does not match the source: its total bytes `found` against
    /// `expected`, and its file count `found_files` against
    /// `expected_files`.
    VerifyFailed {
        expected: u64,
        found: u64,
        expected_files: u64,
        found_files: u64,
    },
    Io(String),
}

impl std::fmt::Display for MoveError {
    // Names are quoted with escapes: an adopted library's name comes from
    // an untrusted marker and must not put control characters on a terminal.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Busy(reason) | Self::Io(reason) => write!(f, "{reason}"),
            Self::InsufficientSpace { needed, available } => write!(
                f,
                "the destination has {available} bytes free and the model needs {needed}"
            ),
            Self::NotConnected(library) => write!(
                f,
                "the library {library:?} is not connected; connect its drive and try again"
            ),
            Self::LibraryReadOnly(library) => write!(
                f,
                "the library {library:?} was written by a newer TurboSpark and is read-only here"
            ),
            Self::HomeOnly => write!(f, "embedding and OCR models stay on This Mac"),
            Self::SameLibrary => write!(f, "the model is already in that library"),
            Self::UnknownInstall(id) => write!(f, "no installed model has the id {id:?}"),
            Self::DestinationTaken(library) => write!(
                f,
                "the library {library:?} already holds a model with this name"
            ),
            Self::Cancelled => write!(f, "the move was cancelled"),
            Self::VerifyFailed {
                expected,
                found,
                expected_files,
                found_files,
            } => write!(
                f,
                "the copy did not match the original: {found} of {expected} bytes, \
                 {found_files} of {expected_files} files"
            ),
        }
    }
}

impl std::error::Error for MoveError {}

impl From<StoreError> for MoveError {
    fn from(error: StoreError) -> Self {
        match error {
            StoreError::NotConnected { library } | StoreError::LibraryDisconnected { library } => {
                Self::NotConnected(library)
            }
            StoreError::LibraryReadOnly { library } => Self::LibraryReadOnly(library),
            StoreError::HomeOnly => Self::HomeOnly,
            StoreError::Io(reason) => Self::Io(reason),
        }
    }
}

/// How a move that put the install in its destination ended. The registry
/// points at the destination in every case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoveOutcome {
    Moved,
    /// Moved, but the source library's marker could not be rewritten and
    /// still lists the install. A marker with an extra row is tolerated (a
    /// marker never LACKS a row the registry holds for its library). The
    /// library's name.
    SourceMarkerStale(String),
    /// Moved, but the source folder at `path` could not be deleted, or was
    /// no longer reached without a link, and is left over.
    /// `stale_marker` names the source library when its marker could not
    /// be rewritten either, as for [`Self::SourceMarkerStale`].
    SourceCleanupFailed {
        path: PathBuf,
        stale_marker: Option<String>,
    },
}

/// Moves the install `install_id` (`<library>:<modality>/<alias>`, as
/// [`InstalledModel::install_id`] spells it) into the library
/// `dest_library`, an id from the library list or [`INTERNAL_LIBRARY_ID`]
/// for This Mac (7.1). Text models, vision towers and audio models, music
/// included, move with their registry row. Image installs are not
/// recorded, so `<library>:image/<alias>` names the folder
/// `<root>/image/<alias>.gturbo`.
///
/// Refused before anything is copied: an embedding or OCR id
/// ([`MoveError::HomeOnly`]), the same library, a source or destination
/// that is not connected or is read-only (the source's marker must lose the
/// row), a destination that already holds the alias and modality, a source
/// holding a link or anything but plain files and folders, and a source
/// larger than `free_bytes`, the destination's free space as the caller
/// measured it (7.4). A row is moved only from its alias's own install
/// folder (`<modality>/<alias>.gturbo`, a vision tower's
/// `text/<alias>.gturbo-vision`, or on This Mac the flat names from before
/// modality folders), since the source is deleted afterward: a This Mac row
/// outside `<home>/models` (`pull --out`), or a row naming a modality
/// folder, is refused. A catalog audio install without its receipt moves
/// only to This Mac (2.9).
///
/// A drive unplugged during the copy is [`MoveError::NotConnected`], as for
/// installs. A copy left on a destination drive that went away cannot be
/// removed until the drive is back.
///
/// `progress` gets the bytes copied and the total, first `(0, total)`, and
/// `cancel` is checked between files and between chunks of a file (7.2).
///
/// Busy checks are the CALLER's: the native interface refuses while the
/// install is loaded or being installed or moved, and the app adds its
/// session checks (7.5). This crate cannot see what is loaded or running.
pub fn move_install(
    store: &Store,
    install_id: &str,
    dest_library: &str,
    env: &LocationEnv,
    free_bytes: u64,
    progress: &dyn Fn(u64, u64),
    cancel: &AtomicBool,
) -> Result<MoveOutcome, MoveError> {
    let plan = plan_move(store, install_id, dest_library, env, free_bytes)?;
    run_move(store, &plan, env, progress, cancel, &|_| {})
}

/// One end of a move: This Mac, or a connected library this build may write.
struct Side {
    /// The library's record, `None` for This Mac.
    record: Option<LibraryRecord>,
    name: String,
    /// This Mac's `<home>/models`, or the canonical library root.
    root: PathBuf,
}

impl Side {
    /// `library`, `None` for This Mac, refused when it is not listed, not
    /// connected, or read-only.
    fn connect(
        store: &Store,
        listed: &[LibraryRecord],
        library: Option<&str>,
        table: &SharedMountTable,
    ) -> Result<Self, MoveError> {
        let Some(id) = library else {
            return Ok(Self {
                record: None,
                name: LibraryRecord::internal(store.root()).name,
                root: store.models_root(),
            });
        };
        let record = listed
            .iter()
            .find(|record| record.id == id)
            .ok_or_else(|| MoveError::Io(format!("no library with the id {id:?} is listed")))?;
        let (root, marker) = connected_library(record, table)?;
        if marker.is_read_only() {
            return Err(MoveError::LibraryReadOnly(record.name.clone()));
        }
        Ok(Self {
            record: Some(record.clone()),
            name: record.name.clone(),
            root,
        })
    }
}

/// Total bytes and file count of a folder, which the copy must match.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Tally {
    bytes: u64,
    files: u64,
}

/// A move that passed every check, worked out before anything is copied.
struct Plan {
    alias: String,
    modality: ModelModality,
    source: Side,
    /// The install folder in `source.root`.
    source_relative: PathBuf,
    destination: Side,
    /// Where the install goes in `destination.root`.
    dest_relative: PathBuf,
    /// What the source holds.
    expected: Tally,
    /// The registry row at the destination, `None` for an image, which is
    /// not recorded.
    moved: Option<InstalledModel>,
    /// `moved` as the destination's marker lists it, in a library.
    portable: Option<PortableRow>,
}

impl Plan {
    fn source_dir(&self) -> PathBuf {
        self.source.root.join(&self.source_relative)
    }

    fn dest_dir(&self) -> PathBuf {
        self.destination.root.join(&self.dest_relative)
    }
}

/// Every check of [`move_install`], before anything is copied or created.
fn plan_move(
    store: &Store,
    install_id: &str,
    dest_library: &str,
    env: &LocationEnv,
    free_bytes: u64,
) -> Result<Plan, MoveError> {
    let unknown = || MoveError::UnknownInstall(install_id.to_string());
    let (library, rest) = install_id.split_once(':').ok_or_else(unknown)?;
    let (modality, alias) = rest.split_once('/').ok_or_else(unknown)?;
    let modality = match modality {
        "text" => ModelModality::Text,
        "image" => ModelModality::Image,
        "audio" => ModelModality::Audio,
        // Their installs and paths are home-only (2.5).
        "embedding" | "embeddings" | "ocr" => return Err(MoveError::HomeOnly),
        _ => return Err(unknown()),
    };
    let external = |id: &str| (id != INTERNAL_LIBRARY_ID).then(|| id.to_string());
    let (from, to) = (external(library), external(dest_library));
    if from == to {
        return Err(MoveError::SameLibrary);
    }
    // Images are not recorded, so an image with no row is its folder.
    let row = store
        .read_rows()
        .into_iter()
        .find(|row| row.install_id() == install_id);
    if row.is_none() && modality != ModelModality::Image {
        return Err(unknown());
    }

    let listed = LibraryRegistry::load(store.root()).libraries();
    let table = SharedMountTable::new(env);
    let source = Side::connect(store, &listed, from.as_deref(), &table)?;
    let destination = Side::connect(store, &listed, to.as_deref(), &table)?;
    let kind = row.as_ref().and_then(|row| row.kind.as_deref());

    // The install folder the alias owns in the source library.
    let own = relative_to(
        &source.root,
        &install_path(&source.root, modality, kind, alias)?,
    )?;
    let source_relative = match &row {
        // Not recorded: the folder comes from the alias.
        None => own,
        Some(row) => {
            let recorded = recorded_relative(row, &source)?;
            // The source is deleted after the move, so a row may name only
            // its alias's own install folder: an untrusted marker row or a
            // hand edit naming `audio` would otherwise take every install in
            // it. This Mac also keeps the flat `<alias>.gturbo` and
            // `<alias>.gturbo-vision` of installs from before modality
            // folders.
            let legacy = source.record.is_none()
                && modality == ModelModality::Text
                && own
                    .file_name()
                    .is_some_and(|name| recorded.as_path() == Path::new(name));
            if recorded != own && !legacy {
                return Err(MoveError::Io(format!(
                    "{} is not the install folder of {alias:?}, so it is not moved",
                    source.root.join(&recorded).display()
                )));
            }
            recorded
        }
    };
    // The containment walk (2.2 SECURITY): every folder from the root to
    // the install is a plain folder, examined without following it.
    let (source_dir, present) = install_in(&source.root, &source_relative).ok_or_else(|| {
        MoveError::Io(format!(
            "{} is not a path inside its library",
            source_relative.display()
        ))
    })?;
    if !present {
        let absent = matches!(
            fs::symlink_metadata(&source_dir),
            Err(e) if e.kind() == io::ErrorKind::NotFound
        );
        if row.is_none() && absent {
            return Err(unknown());
        }
        return Err(MoveError::Io(format!(
            "{} is missing, or is reached through a link, which a move never follows",
            source_dir.display()
        )));
    }
    // A catalog audio install without its receipt cannot be checked or
    // deleted in a library (2.9), so it goes only to This Mac, which adopts
    // it. Audio with no catalog profile, music among it, has no receipt.
    if modality == ModelModality::Audio && destination.record.is_some() {
        let receipted = fs::symlink_metadata(source_dir.join(RECEIPT_FILENAME))
            .is_ok_and(|found| found.file_type().is_file());
        // The catalog is parsed only when the receipt is missing.
        if !receipted
            && AudioCatalog::embedded()
                .map_err(MoveError::Io)?
                .get(alias)
                .is_some()
        {
            return Err(MoveError::Io(format!(
                "{alias:?} has no audio install receipt, so it can move only to This Mac; \
                 adopt it there first"
            )));
        }
    }

    let dest_dir = install_path(&destination.root, modality, kind, alias)?;
    let dest_relative = relative_to(&destination.root, &dest_dir)?;
    // Walked unfollowed too: a modality folder that is a link would put
    // the copy outside the library, and the publish rename refuses it.
    if let Some(modality_dir) = dest_dir.parent() {
        match fs::symlink_metadata(modality_dir) {
            Ok(found) if !found.file_type().is_dir() => return Err(not_plain(modality_dir)),
            _ => {}
        }
    }
    match fs::symlink_metadata(&dest_dir) {
        Ok(_) => return Err(MoveError::DestinationTaken(destination.name)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(io_error("reading", &dest_dir, e)),
    }

    let expected = tally(&source_dir)?;
    if expected.bytes > free_bytes {
        return Err(MoveError::InsufficientSpace {
            needed: expected.bytes,
            available: free_bytes,
        });
    }

    let moved = row.map(|row| InstalledModel {
        path: dest_dir.clone(),
        library: to.clone(),
        relative_path: to.as_ref().map(|_| dest_relative.clone()),
        ..row
    });
    let portable = match (&moved, &destination.record) {
        (Some(moved), Some(_)) => Some(
            PortableRow::from_installed(moved, &destination.root)
                .map_err(|e| MoveError::Io(e.to_string()))?,
        ),
        _ => None,
    };
    Ok(Plan {
        alias: alias.to_string(),
        modality,
        source,
        source_relative,
        destination,
        dest_relative,
        expected,
        moved,
        portable,
    })
}

/// Where the recorded install `row` is in the root of `source`. A library
/// row is placed by its relative path, or by its recorded path when it has
/// none, as listing places it. A This Mac row outside `<home>/models` is a
/// custom location (`pull --out`), in no library, and is not moved.
fn recorded_relative(row: &InstalledModel, source: &Side) -> Result<PathBuf, MoveError> {
    let recorded = || {
        row.path
            .strip_prefix(&source.root)
            .ok()
            .map(Path::to_path_buf)
    };
    let relative = match &source.record {
        Some(_) => row.relative_path.clone().or_else(recorded),
        None => recorded(),
    };
    relative.ok_or_else(|| {
        MoveError::Io(format!(
            "{} is not in {:?}, so it cannot be moved",
            row.path.display(),
            source.name
        ))
    })
}

/// Where an install of `alias` goes in the library at `root`: the layout
/// every library shares, with a vision tower in the text folder under its
/// own suffix.
fn install_path(
    root: &Path,
    modality: ModelModality,
    kind: Option<&str>,
    alias: &str,
) -> Result<PathBuf, MoveError> {
    let path = match modality {
        ModelModality::Text if kind == Some(EntryKind::VisionTower.as_str()) => {
            Store::vision_install_path_in(root, alias)
        }
        ModelModality::Text => Store::install_path_in(root, alias),
        ModelModality::Image => Store::image_install_path_in(root, alias),
        ModelModality::Audio => Store::audio_install_path_in(root, alias),
    };
    path.map_err(MoveError::from)
}

fn relative_to(root: &Path, path: &Path) -> Result<PathBuf, MoveError> {
    path.strip_prefix(root)
        .map(Path::to_path_buf)
        .map_err(|_| MoveError::Io(format!("{} is not in its library", path.display())))
}

/// Copies, verifies, publishes, records, and deletes the source. `after_copy`
/// sees the staged copy before it is verified, the seam the tests use to
/// break a copy or swap a link in.
fn run_move(
    store: &Store,
    plan: &Plan,
    env: &LocationEnv,
    progress: &dyn Fn(u64, u64),
    cancel: &AtomicBool,
    after_copy: &dyn Fn(&Path),
) -> Result<MoveOutcome, MoveError> {
    let name = new_library_id()
        .map_err(|e| MoveError::Io(format!("naming the move's staging folder: {e}")))?;
    let root = &plan.destination.root;
    let staging = store
        .staging_parent_for(root)
        .map_err(|e| lost_drive(store, plan, root, e.into()))?
        .join(format!("move-{name}"));
    create_private_dir(&staging)
        .map_err(|e| lost_drive(store, plan, root, io_error("creating", &staging, e)))?;
    // A cancel or failure before publishing removes the copy, so no partial
    // copy is left (7.3). Removing never follows a link. Classified first:
    // a removed staging folder would read as an unplugged drive.
    if let Err(error) = stage(store, plan, &staging, progress, cancel, after_copy) {
        let error = lost_drive(store, plan, &staging, error);
        let _ = fs::remove_dir_all(&staging);
        return Err(error);
    }

    let Some(moved) = &plan.moved else {
        // An image has no registry or marker row.
        return Ok(finish(plan, env, None));
    };
    // 1. The destination's marker first, so a marker never lacks a row the
    //    registry holds for its library.
    let mut replaced = None;
    if let (Some(record), Some(portable)) = (&plan.destination.record, &plan.portable) {
        let root = &plan.destination.root;
        match set_marker_row(
            record,
            root,
            &plan.alias,
            plan.modality,
            Some(portable.clone()),
        ) {
            Ok(previous) => replaced = Some(previous),
            Err(reason) => {
                let error = format!(
                    "writing the marker of the library {:?}: {reason}",
                    record.name
                );
                return Err(remove_published(plan, MoveError::Io(error)));
            }
        }
    }
    // 2. Then the registry, in one write that replaces the source's row.
    if let Err(reason) = store.record_rows(std::slice::from_ref(moved)) {
        let mut error = format!("recording the move: {reason}");
        if let (Some(record), Some(previous)) = (&plan.destination.record, replaced) {
            let root = &plan.destination.root;
            if let Err(undo) = set_marker_row(record, root, &plan.alias, plan.modality, previous) {
                error = format!(
                    "{error}; the marker of the library {:?} still lists the copy: {undo}",
                    record.name
                );
            }
        }
        return Err(remove_published(plan, MoveError::Io(error)));
    }
    // 3. The source's marker. One that cannot be rewritten keeps an extra
    //    row, which the invariant tolerates, so the move stands.
    let mut stale = None;
    if let Some(record) = &plan.source.record {
        if set_marker_row(record, &plan.source.root, &plan.alias, plan.modality, None).is_err() {
            stale = Some(record.name.clone());
        }
    }
    Ok(finish(plan, env, stale))
}

/// Copies the source into `staging`, checks the copy, and publishes it.
fn stage(
    store: &Store,
    plan: &Plan,
    staging: &Path,
    progress: &dyn Fn(u64, u64),
    cancel: &AtomicBool,
    after_copy: &dyn Fn(&Path),
) -> Result<(), MoveError> {
    progress(0, plan.expected.bytes);
    let mut copier = Copier {
        buffer: vec![0; CHUNK_BYTES],
        copied: 0,
        total: plan.expected.bytes,
        progress,
        cancel,
    };
    copier.copy_dir(&plan.source_dir(), staging)?;
    after_copy(staging);
    let found = tally(staging)?;
    if found != plan.expected {
        return Err(MoveError::VerifyFailed {
            expected: plan.expected.bytes,
            found: found.bytes,
            expected_files: plan.expected.files,
            found_files: found.files,
        });
    }
    // Nothing is published yet, so a cancel after the last chunk still stops.
    check_cancel(cancel)?;
    store
        .publish_staged(staging, &plan.dest_dir(), &plan.destination.name)
        .map_err(MoveError::from)
}

/// Deletes the source, the last step, and says how the move ended.
fn finish(plan: &Plan, env: &LocationEnv, stale_marker: Option<String>) -> MoveOutcome {
    match delete_source(plan, env) {
        Err(path) => MoveOutcome::SourceCleanupFailed { path, stale_marker },
        Ok(()) => stale_marker.map_or(MoveOutcome::Moved, MoveOutcome::SourceMarkerStale),
    }
}

/// `error` from a step of the copy, classified as an install classifies
/// its own ([`Store::install_failure`]): when the source's or the
/// destination's drive can no longer be examined, it was unplugged, and the
/// move is not connected rather than failed. `at` is a path the step used
/// that existed before it failed. Errors other than IO pass through.
fn lost_drive(store: &Store, plan: &Plan, at: &Path, error: MoveError) -> MoveError {
    let MoveError::Io(reason) = error else {
        return error;
    };
    // This Mac does not unplug.
    if plan.source.record.is_some() && fs::symlink_metadata(&plan.source.root).is_err() {
        return MoveError::NotConnected(plan.source.name.clone());
    }
    let destination = &plan.destination;
    store
        .install_failure(at, &destination.root, &destination.name, reason)
        .into()
}

/// Deletes the source folder. Its library is located again and the path
/// walked again, unfollowed, right before: a drive anyone could have
/// written may have had a folder on the way swapped for a link since the
/// copy, and nothing is ever deleted through one (2.2 SECURITY). On failure,
/// returns the folder left over.
fn delete_source(plan: &Plan, env: &LocationEnv) -> Result<(), PathBuf> {
    let root = match &plan.source.record {
        None => plan.source.root.clone(),
        Some(record) => connected_library(record, &SharedMountTable::new(env))
            .map(|(root, _)| root)
            .map_err(|_| plan.source_dir())?,
    };
    match install_in(&root, &plan.source_relative) {
        Some((path, true)) => fs::remove_dir_all(&path).map_err(|_| path),
        Some((path, false)) => Err(path),
        None => Err(plan.source_dir()),
    }
}

/// Removes the published copy after recording it failed, and returns
/// `error`, noting a copy that could not be removed. The copy is this
/// move's own, but the path to it is walked again unfollowed first, so
/// nothing is deleted through a link swapped in meanwhile.
fn remove_published(plan: &Plan, error: MoveError) -> MoveError {
    let removed = match install_in(&plan.destination.root, &plan.dest_relative) {
        Some((path, true)) => fs::remove_dir_all(path).map_err(|e| e.to_string()),
        _ => Err("it is no longer a plain folder in its library".to_string()),
    };
    match removed {
        Ok(()) => error,
        Err(reason) => MoveError::Io(format!(
            "{error}; the copy at {} is left over: {reason}",
            plan.dest_dir().display()
        )),
    }
}

/// Sets the row for `alias` and `modality` in the marker of the library
/// `record` at `root` to `row`, or drops it for `None`, and returns the row
/// it replaced. The marker is read again first, since markers have no
/// lost-update guard (2.2), and one that is gone or is now another
/// library's is not written.
fn set_marker_row(
    record: &LibraryRecord,
    root: &Path,
    alias: &str,
    modality: ModelModality,
    row: Option<PortableRow>,
) -> Result<Option<PortableRow>, String> {
    let mut marker = match LibraryMarker::read(root) {
        Ok(Some(marker)) if marker.id == record.id => marker,
        Ok(_) => return Err("the marker is gone or replaced".to_string()),
        Err(e) => return Err(e.to_string()),
    };
    let held = |kept: &PortableRow| kept.alias == alias && portable_modality(kept) == modality;
    let previous = marker
        .models
        .iter()
        .position(held)
        .map(|at| marker.models.remove(at));
    marker.models.retain(|kept| !held(kept));
    if previous.is_none() && row.is_none() {
        return Ok(None);
    }
    marker.models.extend(row);
    marker.write(root).map_err(|e| e.to_string())?;
    Ok(previous)
}

/// Copies a folder into the staged copy in chunks, reporting progress and
/// checking cancel between files and between chunks.
struct Copier<'a> {
    buffer: Vec<u8>,
    copied: u64,
    total: u64,
    progress: &'a dyn Fn(u64, u64),
    cancel: &'a AtomicBool,
}

impl Copier<'_> {
    /// Copies what the folder `from` holds into the folder `to`. Nothing is
    /// followed: a link or special file, which the scan before the copy
    /// refused, fails the move if one appeared since.
    fn copy_dir(&mut self, from: &Path, to: &Path) -> Result<(), MoveError> {
        for entry in fs::read_dir(from).map_err(|e| io_error("reading", from, e))? {
            check_cancel(self.cancel)?;
            let entry = entry.map_err(|e| io_error("reading", from, e))?;
            let source = entry.path();
            let target = to.join(entry.file_name());
            let kind = entry
                .file_type()
                .map_err(|e| io_error("reading", &source, e))?;
            if kind.is_dir() {
                fs::create_dir(&target).map_err(|e| io_error("creating", &target, e))?;
                self.copy_dir(&source, &target)?;
            } else if kind.is_file() {
                self.copy_file(&source, &target)?;
            } else {
                return Err(not_plain(&source));
            }
        }
        Ok(())
    }

    fn copy_file(&mut self, from: &Path, to: &Path) -> Result<(), MoveError> {
        let listed = fs::symlink_metadata(from).map_err(|e| io_error("reading", from, e))?;
        let mut input = File::open(from).map_err(|e| io_error("opening", from, e))?;
        let opened = input.metadata().map_err(|e| io_error("reading", from, e))?;
        // Opening follows a link, so the file opened must be the plain file
        // listed: one swapped for a link meanwhile is refused, never copied
        // from wherever the link points.
        if !listed.file_type().is_file()
            || (opened.dev(), opened.ino()) != (listed.dev(), listed.ino())
        {
            return Err(not_plain(from));
        }
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(to)
            .map_err(|e| io_error("creating", to, e))?;
        loop {
            check_cancel(self.cancel)?;
            let read = match input.read(&mut self.buffer) {
                Ok(0) => break,
                Ok(read) => read,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(io_error("reading", from, e)),
            };
            output
                .write_all(&self.buffer[..read])
                .map_err(|e| io_error("writing", to, e))?;
            self.copied += read as u64;
            (self.progress)(self.copied, self.total);
        }
        // On the drive before the source can be deleted.
        output.sync_all().map_err(|e| io_error("writing", to, e))
    }
}

/// What the folder `dir` holds, walked without following anything. A link,
/// or anything but a plain file or folder, is refused: a move copies plain
/// files and folders, and a link on a drive anyone could have written could
/// point anywhere.
fn tally(dir: &Path) -> Result<Tally, MoveError> {
    let mut tally = Tally::default();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(at) = pending.pop() {
        for entry in fs::read_dir(&at).map_err(|e| io_error("reading", &at, e))? {
            let entry = entry.map_err(|e| io_error("reading", &at, e))?;
            let path = entry.path();
            let kind = entry
                .file_type()
                .map_err(|e| io_error("reading", &path, e))?;
            if kind.is_dir() {
                pending.push(path);
            } else if kind.is_file() {
                // Not followed, like the entry's type.
                let found = entry
                    .metadata()
                    .map_err(|e| io_error("reading", &path, e))?;
                tally.bytes += found.len();
                tally.files += 1;
            } else {
                return Err(not_plain(&path));
            }
        }
    }
    Ok(tally)
}

fn check_cancel(cancel: &AtomicBool) -> Result<(), MoveError> {
    if cancel.load(Ordering::Relaxed) {
        Err(MoveError::Cancelled)
    } else {
        Ok(())
    }
}

/// Creates `path` at mode 0700 from the start, like install staging, so a
/// staged copy is never readable by other local users.
fn create_private_dir(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    fs::DirBuilder::new().mode(0o700).create(path)
}

fn not_plain(path: &Path) -> MoveError {
    MoveError::Io(format!(
        "{} is a link or not a plain file or folder, and a move copies only plain files and folders",
        path.display()
    ))
}

fn io_error(action: &str, path: &Path, error: io::Error) -> MoveError {
    MoveError::Io(format!("{action} {}: {error}", path.display()))
}

#[cfg(test)]
#[path = "transfer_tests.rs"]
mod tests;
