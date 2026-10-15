//! The library list, `<home>/libraries.json`.
//!
//! The list lives in the home on the Mac, so unplugging a drive never takes
//! it along (1.1). "This Mac" is implicit: it is never written into the
//! `libraries` array, only its default flag is, as `internal_is_default`.
//!
//! The CLI and the app both write the list, and `std` file locking needs a
//! newer Rust than the MSRV, so every write re-reads the file just before
//! its rename and applies only its own change to that fresh copy. A write
//! from another process is then lost only if it lands inside that window.

use super::locate::{locate, Located};
use super::marker::{replace_file, validate_id, validate_relative_path};
use super::{
    new_library_id, validate_location, LibraryError, LibraryMarker, LibraryRecord, LibraryView,
    LocationEnv, LocationRefusal, INTERNAL_LIBRARY_ID, STANDARD_LIBRARY_FOLDER,
};
use crate::{InstalledModel, ModelModality, Store};
use serde::{Deserialize, Serialize};
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

/// The list's file name inside the home.
pub const LIBRARIES_FILE: &str = "libraries.json";

/// The format this build reads and writes. A list with a greater format is
/// read but never rewritten, because rewriting it would drop the fields this
/// build does not know.
const LIBRARIES_FORMAT: u32 = 1;

/// What reading the list set aside rather than trusting.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LoadReport {
    /// Where an unparseable `libraries.json` was moved.
    pub quarantined: Option<PathBuf>,
    /// One line per record dropped as invalid.
    pub dropped: Vec<String>,
    /// Why an existing `libraries.json` could not be read at all.
    pub unreadable: Option<String>,
}

/// The outcome of [`LibraryRegistry::add`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddedLibrary {
    pub view: LibraryView,
    /// Aliases of marker rows not recorded, because the alias and modality
    /// already belong to another library or repeat within the marker.
    pub skipped: Vec<String>,
}

/// The library list of one home, and the only writer of `libraries.json`.
#[derive(Debug, Clone)]
pub struct LibraryRegistry {
    home: PathBuf,
    list: LibraryList,
    report: LoadReport,
}

/// The parsed list. Records are the external libraries only.
#[derive(Debug, Clone)]
struct LibraryList {
    format: u32,
    records: Vec<LibraryRecord>,
    internal_is_default: bool,
}

/// The file as read. Records stay raw so one bad record is dropped alone
/// rather than failing the whole list.
#[derive(Deserialize)]
struct StoredList {
    format: u32,
    #[serde(default)]
    libraries: Vec<serde_json::Value>,
    #[serde(default = "this_mac_is_default")]
    internal_is_default: bool,
}

#[derive(Serialize)]
struct WrittenList<'a> {
    format: u32,
    libraries: &'a [LibraryRecord],
    internal_is_default: bool,
}

/// A list written before any default was chosen starts with This Mac (1.9).
fn this_mac_is_default() -> bool {
    true
}

impl LibraryList {
    fn only_this_mac() -> Self {
        Self {
            format: LIBRARIES_FORMAT,
            records: Vec::new(),
            internal_is_default: true,
        }
    }

    /// Exactly one default (1.9). A hand edit can leave none or several:
    /// This Mac wins when it is flagged, else the first flagged library,
    /// else This Mac.
    fn normalize_default(&mut self) {
        let chosen = if self.internal_is_default {
            None
        } else {
            self.records
                .iter()
                .position(|record| record.default_for_downloads)
        };
        self.internal_is_default = chosen.is_none();
        for (index, record) in self.records.iter_mut().enumerate() {
            record.default_for_downloads = Some(index) == chosen;
        }
    }

    fn contains(&self, id: &str) -> bool {
        self.records.iter().any(|record| record.id == id)
    }
}

impl LibraryRegistry {
    /// The list in `home`. A missing file is only This Mac, as the default.
    /// Never fails: an unparseable file is moved aside and an unreadable one
    /// is reported in [`Self::load_report`], and both read as only This Mac.
    pub fn load(home: &Path) -> Self {
        let (list, report) = read_list(home).unwrap_or_else(|reason| {
            let report = LoadReport {
                unreadable: Some(reason),
                ..LoadReport::default()
            };
            (LibraryList::only_this_mac(), report)
        });
        Self {
            home: home.to_path_buf(),
            list,
            report,
        }
    }

    /// What the most recent read of the list set aside.
    pub fn load_report(&self) -> &LoadReport {
        &self.report
    }

    /// This Mac first, then every external library in list order.
    pub fn libraries(&self) -> Vec<LibraryRecord> {
        let mut this_mac = LibraryRecord::internal(&self.home);
        this_mac.default_for_downloads = self.list.internal_is_default;
        std::iter::once(this_mac)
            .chain(self.list.records.iter().cloned())
            .collect()
    }

    /// Adds the folder at `path` as a library (1.3), or adopts the library
    /// already there with its models (1.6).
    ///
    /// The library root is `path` when it holds a marker, else its
    /// `TurboSpark` child when that holds one, else `path` itself when it is
    /// named `TurboSpark`, else a new `TurboSpark` child. A root is never the
    /// top of a volume. A marker that cannot be read refuses the add, and
    /// nothing ever overwrites it. Adopting a library already listed
    /// returns it unchanged.
    pub fn add(
        &mut self,
        path: &Path,
        name: Option<&str>,
        env: &LocationEnv,
    ) -> Result<AddedLibrary, LibraryError> {
        // `validate_location` takes a canonical path, and the nesting rules
        // compare canonical paths, so no symlink or `..` spelling slips past.
        let chosen = path
            .canonicalize()
            .map_err(|_| LibraryError::Refused(LocationRefusal::NotADirectory))?;
        let home = self
            .home
            .canonicalize()
            .unwrap_or_else(|_| self.home.clone());
        // Checked apart from the disk rule: TURBOSPARK_HOME can put the home
        // on an external drive (1.7).
        if chosen.starts_with(&home) {
            return Err(LibraryError::Refused(LocationRefusal::InsideHome));
        }
        let volume = validate_location(&chosen, env).map_err(LibraryError::Refused)?;
        let (root, adopted) = choose_root(&chosen, &volume)?;

        self.reread()?;
        self.ensure_writable()?;
        if let Some(marker) = &adopted {
            if let Some(listed) = self.record(&marker.id) {
                return Ok(AddedLibrary {
                    view: self.view(&listed, env),
                    skipped: Vec::new(),
                });
            }
        }
        // A library inside another would list the same files twice, and
        // removing either would forget models the other still holds.
        if let Some(library) = self.covering_library(&root, env) {
            return Err(LibraryError::Refused(LocationRefusal::NestedIn { library }));
        }
        if home.starts_with(&root) {
            return Err(LibraryError::Refused(LocationRefusal::InsideHome));
        }

        let volume_name = volume
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let name = name
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::to_string)
            .or_else(|| {
                adopted
                    .as_ref()
                    .map(|marker| marker.name.trim().to_string())
                    .filter(|name| !name.is_empty())
            })
            .unwrap_or(volume_name);
        // Worked out before anything is created, so a root that cannot be
        // recorded leaves no folder or marker on the drive. `volume` is the
        // volume root of `chosen` (`volume_root_of`), and the root is
        // `chosen` or its standard child on the same device.
        let relative = root
            .strip_prefix(&volume)
            .map(Path::to_path_buf)
            .unwrap_or_default();
        if validate_relative_path(&relative).is_err() {
            return Err(LibraryError::Io(format!(
                "{} is not a folder below the top of its drive",
                root.display()
            )));
        }
        let marker = match adopted {
            Some(marker) => marker,
            None => create_library(&root, &name)?,
        };
        let record = LibraryRecord {
            id: marker.id.clone(),
            name,
            volume_relative_path: relative,
            last_mount_path: volume,
            default_for_downloads: false,
            added_on: crate::install::today(),
            last_seen: None,
        };

        // Rows before the list: if the list write fails, the rows are
        // forgotten again, and a retry adopts the marker afresh.
        let skipped = self.merge_rows(&marker, &root)?;
        let committed = self.commit(|list| {
            // Another process may have added this library meanwhile.
            if !list.contains(&record.id) {
                list.records.push(record.clone());
            }
            Ok(())
        });
        if let Err(e) = committed {
            let _ = Store::new(&self.home).forget_library_rows(&record.id);
            return Err(e);
        }
        let listed = self.record(&record.id).unwrap_or(record);
        Ok(AddedLibrary {
            view: self.view(&listed, env),
            skipped,
        })
    }

    /// Stops listing the library `id` and its models (1.8). Never touches a
    /// file: the folder and its marker stay as they are, so adding the drive
    /// again restores the models from the marker (1.6).
    pub fn remove(&mut self, id: &str) -> Result<(), LibraryError> {
        if id == INTERNAL_LIBRARY_ID {
            return Err(LibraryError::Internal);
        }
        self.reread()?;
        self.ensure_writable()?;
        if !self.list.contains(id) {
            return Err(LibraryError::UnknownLibrary(id.to_string()));
        }
        // Rows first: if the list write then fails, the library is still
        // listed and a retry finishes. The other order would strand rows
        // under an id no list knows.
        Store::new(&self.home)
            .forget_library_rows(id)
            .map_err(LibraryError::Io)?;
        // Removing the default hands it back to This Mac in the commit's
        // normalization.
        self.commit(|list| {
            list.records.retain(|record| record.id != id);
            Ok(())
        })
    }

    /// Makes `id`, This Mac or a listed library, the only default for new
    /// downloads (1.9).
    pub fn set_default(&mut self, id: &str) -> Result<(), LibraryError> {
        self.commit(|list| {
            if id != INTERNAL_LIBRARY_ID && !list.contains(id) {
                return Err(LibraryError::UnknownLibrary(id.to_string()));
            }
            list.internal_is_default = id == INTERNAL_LIBRARY_ID;
            for record in &mut list.records {
                record.default_for_downloads = record.id == id;
            }
            Ok(())
        })
    }

    fn record(&self, id: &str) -> Option<LibraryRecord> {
        self.list
            .records
            .iter()
            .find(|record| record.id == id)
            .cloned()
    }

    fn reread(&mut self) -> Result<(), LibraryError> {
        let (list, report) = read_list(&self.home).map_err(LibraryError::Io)?;
        self.list = list;
        self.report = report;
        Ok(())
    }

    fn ensure_writable(&self) -> Result<(), LibraryError> {
        if self.list.format > LIBRARIES_FORMAT {
            return Err(LibraryError::Io(format!(
                "{} is format {}, written by a newer TurboSpark",
                self.home.join(LIBRARIES_FILE).display(),
                self.list.format
            )));
        }
        Ok(())
    }

    /// Writes the list with `change` applied to it as it is on disk now,
    /// not to this registry's copy, so a library another process added or
    /// removed since stays as that process left it.
    fn commit(
        &mut self,
        change: impl FnOnce(&mut LibraryList) -> Result<(), LibraryError>,
    ) -> Result<(), LibraryError> {
        #[cfg(test)]
        if let Some(hook) = BEFORE_COMMIT.with(|slot| slot.borrow_mut().take()) {
            hook();
        }
        self.reread()?;
        self.ensure_writable()?;
        let mut fresh = self.list.clone();
        change(&mut fresh)?;
        fresh.normalize_default();
        let bytes = serde_json::to_vec_pretty(&WrittenList {
            format: LIBRARIES_FORMAT,
            libraries: &fresh.records,
            internal_is_default: fresh.internal_is_default,
        })
        .map_err(|e| LibraryError::Io(format!("serializing the library list: {e}")))?;
        let path = self.home.join(LIBRARIES_FILE);
        fs::create_dir_all(&self.home)
            .and_then(|()| replace_file(&path, &bytes))
            .map_err(|e| LibraryError::Io(format!("writing {}: {e}", path.display())))?;
        self.list = fresh;
        Ok(())
    }

    /// The listed library whose root is `root`, contains it, or sits inside
    /// it, compared where the library is now.
    ///
    /// A library that is not connected has no place to compare. Another
    /// drive often mounts under the same name ("T7", "Untitled"), and its
    /// folders are not that library's. Its recorded place still counts when
    /// a marker there exists but cannot be read, since that could be the
    /// library itself. A missing folder, no marker, or another id there means
    /// the place is free.
    fn covering_library(&self, root: &Path, env: &LocationEnv) -> Option<String> {
        self.list
            .records
            .iter()
            .find(|record| {
                let other = match locate(record, env) {
                    Located::Available { root, .. } => root,
                    Located::NotConnected => {
                        let recorded = record.last_mount_path.join(&record.volume_relative_path);
                        let unreadable = fs::symlink_metadata(&recorded).is_ok()
                            && LibraryMarker::read(&recorded).is_err();
                        if !unreadable {
                            return false;
                        }
                        recorded
                    }
                };
                let other = other.canonicalize().unwrap_or(other);
                root.starts_with(&other) || other.starts_with(root)
            })
            .map(|record| record.name.clone())
    }

    /// Records the marker's rows as installs in its library, now at `root`,
    /// and returns the aliases it skipped.
    ///
    /// Alias plus modality is unique across every library, so a row already
    /// held by This Mac or another library is skipped rather than replacing
    /// that row and orphaning its folder. A repeat within the marker is
    /// skipped too. A row this library already has is kept as it is. The
    /// rows land in one write, so a failure records none of them.
    fn merge_rows(&self, marker: &LibraryMarker, root: &Path) -> Result<Vec<String>, LibraryError> {
        let store = Store::new(&self.home);
        let rows = store.read_rows();
        let mut seen: Vec<(String, ModelModality)> = Vec::new();
        let mut merged: Vec<InstalledModel> = Vec::new();
        let mut skipped = Vec::new();
        for portable in &marker.models {
            let row = portable
                .to_installed(root, &marker.id)
                .map_err(LibraryError::Marker)?;
            let key = (row.alias.clone(), row.effective_modality());
            if seen.contains(&key) {
                skipped.push(row.alias);
                continue;
            }
            let held = rows
                .iter()
                .find(|existing| existing.alias == key.0 && existing.effective_modality() == key.1);
            seen.push(key);
            match held {
                Some(existing) if existing.library.as_deref() == Some(marker.id.as_str()) => {}
                Some(_) => skipped.push(row.alias),
                None => merged.push(row),
            }
        }
        // No write when nothing is new, so adding an empty library never
        // creates or touches the record.
        if !merged.is_empty() {
            store.record_all(&merged).map_err(LibraryError::Io)?;
        }
        Ok(skipped)
    }

    fn view(&self, record: &LibraryRecord, env: &LocationEnv) -> LibraryView {
        let (root, available, duplicate) = match locate(record, env) {
            Located::Available { root, duplicate } => (Some(root), true, duplicate),
            Located::NotConnected => (None, false, false),
        };
        let rows = Store::new(&self.home).read_rows();
        LibraryView {
            record: record.clone(),
            root,
            available,
            duplicate,
            model_count: self.model_count(record, &rows),
        }
    }

    /// The rows of `record`. A row with no library is This Mac's only when it
    /// sits under `<home>/models`: one elsewhere is a custom location.
    fn model_count(&self, record: &LibraryRecord, rows: &[InstalledModel]) -> usize {
        if record.is_internal() {
            let models = self.home.join("models");
            rows.iter()
                .filter(|row| row.library.is_none() && row.path.starts_with(&models))
                .count()
        } else {
            rows.iter()
                .filter(|row| row.library.as_deref() == Some(record.id.as_str()))
                .count()
        }
    }
}

/// Where the library for the chosen folder `chosen` on the volume `volume`
/// lives, with the marker to adopt there, if any.
fn choose_root(
    chosen: &Path,
    volume: &Path,
) -> Result<(PathBuf, Option<LibraryMarker>), LibraryError> {
    // A root at the volume top would have an empty relative path, so a
    // marker there is never adopted and the drive gets the standard folder.
    let at_top = chosen == volume;
    if !at_top {
        if let Some(marker) = read_marker(chosen)? {
            return Ok((chosen.to_path_buf(), Some(marker)));
        }
    }
    let standard = chosen.join(STANDARD_LIBRARY_FOLDER);
    // `validate_location` vouched for the device of `chosen` only. A share
    // mounted at the standard folder is inside this drive by path but is
    // another drive, so it must not be read or registered as this one.
    // Followed, so a symlink onto another device is caught here too.
    if let (Ok(inner), Ok(outer)) = (fs::metadata(&standard), fs::metadata(chosen)) {
        if inner.dev() != outer.dev() {
            return Err(LibraryError::Refused(LocationRefusal::UnknownVolume));
        }
    }
    // Not followed: a symlinked standard folder could lead the library off
    // this drive, past every check `validate_location` made.
    let standard_kind = fs::symlink_metadata(&standard).map(|metadata| metadata.file_type());
    let standard_is_folder = standard_kind.as_ref().is_ok_and(|kind| kind.is_dir());
    if standard_is_folder {
        if let Some(marker) = read_marker(&standard)? {
            return Ok((standard, Some(marker)));
        }
    }
    if !at_top && chosen.file_name() == Some(OsStr::new(STANDARD_LIBRARY_FOLDER)) {
        return Ok((chosen.to_path_buf(), None));
    }
    match standard_kind {
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok((standard, None)),
        _ if standard_is_folder => Ok((standard, None)),
        _ => Err(LibraryError::Refused(LocationRefusal::NotADirectory)),
    }
}

/// The marker in `root`. One that exists but cannot be read refuses the add
/// rather than being replaced: it may describe a library and its models.
fn read_marker(root: &Path) -> Result<Option<LibraryMarker>, LibraryError> {
    LibraryMarker::read(root).map_err(LibraryError::Marker)
}

/// Creates the folder `root` if needed and a fresh marker in it. Only called
/// after reading found no marker there.
fn create_library(root: &Path, name: &str) -> Result<LibraryMarker, LibraryError> {
    match fs::create_dir(root) {
        Err(e) if e.kind() != io::ErrorKind::AlreadyExists => {
            return Err(LibraryError::Io(format!(
                "creating {}: {e}",
                root.display()
            )));
        }
        _ => {}
    }
    let id =
        new_library_id().map_err(|e| LibraryError::Io(format!("creating a library id: {e}")))?;
    let marker = LibraryMarker::new(id, name.to_string());
    marker.write(root).map_err(LibraryError::Marker)?;
    Ok(marker)
}

/// The list in `home` with its read report. A missing file is only This
/// Mac. An unparseable one is moved aside first, so no write ever replaces
/// it. Fails only when the file exists but cannot be read or moved aside.
///
/// The format is read before anything else. A newer list is never moved
/// aside, even when this build cannot parse the rest of it: it is a newer
/// TurboSpark's valid list, and it stays read-only here.
fn read_list(home: &Path) -> Result<(LibraryList, LoadReport), String> {
    let path = home.join(LIBRARIES_FILE);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Ok((LibraryList::only_this_mac(), LoadReport::default()));
        }
        Err(e) => return Err(format!("reading {}: {e}", path.display())),
    };
    let value: Option<serde_json::Value> = serde_json::from_slice(&bytes).ok();
    let format = value
        .as_ref()
        .and_then(|value| value.get("format"))
        .and_then(serde_json::Value::as_u64);
    let stored = match (format, value) {
        (Some(format), Some(value)) if format > u64::from(LIBRARIES_FORMAT) => {
            match serde_json::from_value::<StoredList>(value) {
                Ok(stored) => stored,
                Err(e) => {
                    let mut list = LibraryList::only_this_mac();
                    list.format = u32::try_from(format).unwrap_or(u32::MAX);
                    let report = LoadReport {
                        unreadable: Some(format!(
                            "{} is format {format}, written by a newer TurboSpark: {e}",
                            path.display()
                        )),
                        ..LoadReport::default()
                    };
                    return Ok((list, report));
                }
            }
        }
        (Some(format), Some(value)) if format > 0 => {
            match serde_json::from_value::<StoredList>(value) {
                Ok(stored) => stored,
                Err(_) => return quarantined(&path),
            }
        }
        _ => return quarantined(&path),
    };
    let mut report = LoadReport::default();
    let mut records: Vec<LibraryRecord> = Vec::new();
    for (index, value) in stored.libraries.into_iter().enumerate() {
        match parse_record(value) {
            Ok(record) if records.iter().any(|kept| kept.id == record.id) => report
                .dropped
                .push(format!("libraries[{index}] repeats the id {:?}", record.id)),
            Ok(record) => records.push(record),
            Err(reason) => report.dropped.push(format!("libraries[{index}]: {reason}")),
        }
    }
    let mut list = LibraryList {
        format: stored.format,
        records,
        internal_is_default: stored.internal_is_default,
    };
    list.normalize_default();
    Ok((list, report))
}

/// One stored record, held to the rules a marker's id and relative paths
/// meet, since a record's root is joined from its relative path and its id
/// is embedded in install ids.
fn parse_record(value: serde_json::Value) -> Result<LibraryRecord, String> {
    let record: LibraryRecord = serde_json::from_value(value).map_err(|e| e.to_string())?;
    validate_id(&record.id).map_err(|e| e.to_string())?;
    if validate_relative_path(&record.volume_relative_path).is_err() {
        return Err(format!(
            "the library {:?} has an unsafe volume_relative_path {:?}",
            record.id, record.volume_relative_path
        ));
    }
    // A relative mount path would resolve against whatever the working
    // directory happens to be.
    if !record.last_mount_path.is_absolute() {
        return Err(format!(
            "the library {:?} has a relative last_mount_path {:?}",
            record.id, record.last_mount_path
        ));
    }
    Ok(record)
}

/// Only This Mac, after moving the unparseable list at `path` aside.
fn quarantined(path: &Path) -> Result<(LibraryList, LoadReport), String> {
    let report = LoadReport {
        quarantined: Some(quarantine(path)?),
        ..LoadReport::default()
    };
    Ok((LibraryList::only_this_mac(), report))
}

/// Moves the unparseable list at `path` to `libraries.json.corrupt-<secs>`
/// beside it, as the app's JSON stores do, and returns the new path.
fn quarantine(path: &Path) -> Result<PathBuf, String> {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    let base = format!("{LIBRARIES_FILE}.corrupt-{secs}");
    let mut target = path.with_file_name(&base);
    // A second quarantine in the same second must not replace the first.
    let mut copy = 1;
    while fs::symlink_metadata(&target).is_ok() {
        target = path.with_file_name(format!("{base}-{copy}"));
        copy += 1;
    }
    fs::rename(path, &target).map_err(|e| format!("moving aside {}: {e}", path.display()))?;
    Ok(target)
}

#[cfg(test)]
thread_local! {
    /// Runs once inside the next commit on this thread, before its re-read,
    /// so a test can stand in another process writing the list mid-call.
    static BEFORE_COMMIT: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn before_next_commit(hook: impl FnOnce() + 'static) {
    BEFORE_COMMIT.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::library::{
        parse_mount_table, LibraryMarker, LocationRefusal, MarkerError, MountSource, PortableRow,
        INTERNAL_LIBRARY_ID, MARKER_FILE, STANDARD_LIBRARY_FOLDER,
    };
    use crate::{InstalledModel, ModelModality, Store};
    use std::collections::BTreeMap;
    use std::fs;
    use std::os::unix::fs::MetadataExt;

    const ID: &str = "0f6c2a3e-1b7d-4c55-9e0a-7d2f1c3b4a59";
    const OTHER_ID: &str = "7a1e9c04-52b3-4f8d-a6e1-3c9b0d2f5e17";
    const LOCAL: &str = "hfs, local, journaled";

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ts-library-registry-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        // The temp dir is behind the /var symlink, and the location rules
        // compare canonical paths.
        dir.canonicalize().unwrap()
    }

    /// A temp home plus fake volumes under `<base>/Volumes`, with a fixed
    /// mount table built from real-format lines. `home_device` differs from
    /// the temp dir's device, so the same-disk rule does not fire for the
    /// fake volumes, which share it.
    struct Fixture {
        base: PathBuf,
        home: PathBuf,
        volumes_root: PathBuf,
        env: LocationEnv,
    }

    impl Fixture {
        /// `volumes` are mount table entries as `(name, flags)`. A folder
        /// under the volumes root without an entry is an unknown volume.
        fn new(test: &str, volumes: &[(&str, &str)]) -> Self {
            let base = scratch(test);
            let home = base.join("home");
            fs::create_dir_all(&home).unwrap();
            let volumes_root = base.join("Volumes");
            let mut table = String::from("/dev/disk3s1s1 on / (apfs, sealed, local, read-only)\n");
            for (index, (name, flags)) in volumes.iter().enumerate() {
                let volume = volumes_root.join(name);
                fs::create_dir_all(&volume).unwrap();
                table.push_str(&format!(
                    "/dev/disk{}s1 on {} ({flags})\n",
                    index + 20,
                    volume.display()
                ));
            }
            let home_device = fs::metadata(&base).unwrap().dev().wrapping_add(1);
            Self {
                env: LocationEnv {
                    mount_source: MountSource::Fixed(parse_mount_table(&table)),
                    volumes_root: volumes_root.clone(),
                    home_device,
                },
                base,
                home,
                volumes_root,
            }
        }

        fn volume(&self, name: &str) -> PathBuf {
            self.volumes_root.join(name)
        }

        fn registry(&self) -> LibraryRegistry {
            LibraryRegistry::load(&self.home)
        }

        fn list_path(&self) -> PathBuf {
            self.home.join(LIBRARIES_FILE)
        }

        fn rows(&self) -> Vec<InstalledModel> {
            Store::new(&self.home).read_rows()
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.base);
        }
    }

    fn portable(alias: &str, relative: &str, modality: ModelModality) -> PortableRow {
        PortableRow {
            alias: alias.to_string(),
            repo: format!("owner/{alias}"),
            revision: "main".to_string(),
            relative_path: PathBuf::from(relative),
            family: "qwen3".to_string(),
            install_bytes: 4_096,
            installed_on: "2026-10-14".to_string(),
            status: "verified".to_string(),
            kind: None,
            modality,
            variant: None,
        }
    }

    fn installed(
        alias: &str,
        modality: ModelModality,
        path: PathBuf,
        library: Option<&str>,
    ) -> InstalledModel {
        InstalledModel {
            alias: alias.to_string(),
            repo: format!("owner/{alias}"),
            revision: "main".to_string(),
            path,
            family: "qwen3".to_string(),
            install_bytes: 4_096,
            installed_on: "2026-10-14".to_string(),
            status: "verified".to_string(),
            kind: None,
            modality,
            variant: None,
            library: library.map(str::to_string),
            relative_path: None,
        }
    }

    /// A marker with `id` and `rows` written into `root`, which is created.
    fn marker_at(root: &Path, id: &str, name: &str, rows: Vec<PortableRow>) {
        fs::create_dir_all(root).unwrap();
        let mut marker = LibraryMarker::new(id.to_string(), name.to_string());
        marker.models = rows;
        marker.write(root).unwrap();
    }

    /// Every file under `dir` with its bytes, and every folder, by path
    /// relative to `dir`.
    fn snapshot(dir: &Path) -> BTreeMap<PathBuf, Option<Vec<u8>>> {
        let mut found = BTreeMap::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(at) = stack.pop() {
            for entry in fs::read_dir(&at).unwrap() {
                let path = entry.unwrap().path();
                let relative = path.strip_prefix(dir).unwrap().to_path_buf();
                if path.is_dir() {
                    found.insert(relative, None);
                    stack.push(path);
                } else {
                    found.insert(relative, Some(fs::read(&path).unwrap()));
                }
            }
        }
        found
    }

    fn defaults(registry: &LibraryRegistry) -> Vec<String> {
        registry
            .libraries()
            .into_iter()
            .filter(|record| record.default_for_downloads)
            .map(|record| record.id)
            .collect()
    }

    fn ids(registry: &LibraryRegistry) -> Vec<String> {
        registry
            .libraries()
            .into_iter()
            .map(|record| record.id)
            .collect()
    }

    fn names(registry: &LibraryRegistry) -> Vec<String> {
        registry
            .libraries()
            .into_iter()
            .map(|record| record.name)
            .collect()
    }

    fn refused(result: Result<AddedLibrary, LibraryError>) -> LocationRefusal {
        match result {
            Err(LibraryError::Refused(refusal)) => refusal,
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_missing_list_means_only_this_mac_as_the_default() {
        let fx = Fixture::new("missing", &[]);
        let registry = fx.registry();

        let libraries = registry.libraries();
        assert_eq!(libraries.len(), 1);
        assert!(libraries[0].is_internal());
        assert!(libraries[0].default_for_downloads);
        assert_eq!(
            libraries[0]
                .last_mount_path
                .join(&libraries[0].volume_relative_path),
            fx.home.join("models")
        );
        assert_eq!(registry.load_report(), &LoadReport::default());
        // Reading never creates the file.
        assert!(!fx.list_path().exists());
    }

    #[test]
    fn adding_a_drive_creates_the_standard_folder_with_a_fresh_marker() {
        let fx = Fixture::new("fresh", &[("T7", LOCAL)]);
        let mut registry = fx.registry();

        let added = registry.add(&fx.volume("T7"), None, &fx.env).unwrap();

        let root = fx.volume("T7").join(STANDARD_LIBRARY_FOLDER);
        let marker = LibraryMarker::read(&root).unwrap().unwrap();
        assert_ne!(marker.id, INTERNAL_LIBRARY_ID);
        assert_eq!(marker.name, "T7");
        assert!(marker.models.is_empty());
        let record = &added.view.record;
        assert_eq!(record.id, marker.id);
        assert_eq!(record.name, "T7");
        assert_eq!(record.volume_relative_path, PathBuf::from("TurboSpark"));
        assert_eq!(record.last_mount_path, fx.volume("T7"));
        assert!(!record.default_for_downloads);
        assert_eq!(record.added_on, crate::install::today());
        assert_eq!(record.last_seen, None);
        assert_eq!(added.view.root, Some(root));
        assert!(added.view.available);
        assert!(!added.view.duplicate);
        assert_eq!(added.view.model_count, 0);
        assert!(added.skipped.is_empty());

        // Persisted, and This Mac stays out of the array.
        let reloaded = fx.registry();
        assert_eq!(reloaded.libraries(), registry.libraries());
        assert_eq!(
            ids(&reloaded),
            vec![INTERNAL_LIBRARY_ID.to_string(), marker.id.clone()]
        );
        let json: serde_json::Value =
            serde_json::from_slice(&fs::read(fx.list_path()).unwrap()).unwrap();
        assert_eq!(json["format"], 1);
        assert_eq!(json["internal_is_default"], true);
        let stored = json["libraries"].as_array().unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0]["id"], marker.id.as_str());
    }

    #[test]
    fn a_chosen_folder_gets_a_standard_child_unless_it_is_one() {
        let fx = Fixture::new("folders", &[("T7", LOCAL)]);
        let stuff = fx.volume("T7").join("Stuff");
        let named = fx
            .volume("T7")
            .join("Archive")
            .join(STANDARD_LIBRARY_FOLDER);
        fs::create_dir_all(&stuff).unwrap();
        fs::create_dir_all(&named).unwrap();
        let mut registry = fx.registry();

        let in_stuff = registry.add(&stuff, Some("Models"), &fx.env).unwrap();
        assert_eq!(
            in_stuff.view.root,
            Some(stuff.join(STANDARD_LIBRARY_FOLDER))
        );
        assert_eq!(in_stuff.view.record.name, "Models");
        assert_eq!(
            in_stuff.view.record.volume_relative_path,
            PathBuf::from("Stuff/TurboSpark")
        );
        let marker = LibraryMarker::read(&stuff.join(STANDARD_LIBRARY_FOLDER))
            .unwrap()
            .unwrap();
        assert_eq!(marker.name, "Models");

        // A folder already named TurboSpark becomes the root itself.
        let in_named = registry.add(&named, None, &fx.env).unwrap();
        assert_eq!(in_named.view.root, Some(named.clone()));
        assert_eq!(
            in_named.view.record.volume_relative_path,
            PathBuf::from("Archive/TurboSpark")
        );
        assert!(!named.join(STANDARD_LIBRARY_FOLDER).exists());
        assert!(LibraryMarker::read(&named).unwrap().is_some());
    }

    #[test]
    fn adoption_merges_marker_rows_and_skips_conflicts() {
        let fx = Fixture::new("adopt", &[("T7", LOCAL)]);
        let root = fx.volume("T7").join(STANDARD_LIBRARY_FOLDER);
        marker_at(
            &root,
            ID,
            "Archive",
            vec![
                portable("qwen", "text/qwen.gturbo", ModelModality::Text),
                // A repeat inside the marker.
                portable("qwen", "text/qwen-copy.gturbo", ModelModality::Text),
                // Already This Mac's.
                portable("flux", "image/flux.gturbo", ModelModality::Image),
                portable("whisper", "audio/whisper.gturbo", ModelModality::Audio),
                // Same alias, other modality: not a conflict.
                portable("qwen", "audio/qwen.gturbo", ModelModality::Audio),
                // Already another library's.
                portable("kokoro", "audio/kokoro.gturbo", ModelModality::Audio),
            ],
        );
        let marker_bytes = fs::read(root.join(MARKER_FILE)).unwrap();
        let store = Store::new(&fx.home);
        let this_mac_flux = installed(
            "flux",
            ModelModality::Image,
            fx.home.join("models/image/flux.gturbo"),
            None,
        );
        let other_kokoro = installed(
            "kokoro",
            ModelModality::Audio,
            PathBuf::from("/Volumes/Other/TurboSpark/audio/kokoro.gturbo"),
            Some(OTHER_ID),
        );
        store.record(&this_mac_flux).unwrap();
        store.record(&other_kokoro).unwrap();
        let mut registry = fx.registry();

        let added = registry.add(&fx.volume("T7"), None, &fx.env).unwrap();

        assert_eq!(added.view.record.id, ID);
        // The marker names an adopted library when the caller does not.
        assert_eq!(added.view.record.name, "Archive");
        assert_eq!(added.skipped, vec!["qwen", "flux", "kokoro"]);
        assert_eq!(added.view.model_count, 3);
        let rows = fx.rows();
        assert!(rows.contains(&this_mac_flux));
        assert!(rows.contains(&other_kokoro));
        let ours: Vec<(String, ModelModality, PathBuf, Option<PathBuf>)> = rows
            .iter()
            .filter(|row| row.library.as_deref() == Some(ID))
            .map(|row| {
                (
                    row.alias.clone(),
                    row.modality,
                    row.path.clone(),
                    row.relative_path.clone(),
                )
            })
            .collect();
        assert_eq!(
            ours,
            vec![
                (
                    "qwen".to_string(),
                    ModelModality::Text,
                    root.join("text/qwen.gturbo"),
                    Some(PathBuf::from("text/qwen.gturbo"))
                ),
                (
                    "whisper".to_string(),
                    ModelModality::Audio,
                    root.join("audio/whisper.gturbo"),
                    Some(PathBuf::from("audio/whisper.gturbo"))
                ),
                (
                    "qwen".to_string(),
                    ModelModality::Audio,
                    root.join("audio/qwen.gturbo"),
                    Some(PathBuf::from("audio/qwen.gturbo"))
                ),
            ]
        );
        // Adoption never rewrites the marker.
        assert_eq!(fs::read(root.join(MARKER_FILE)).unwrap(), marker_bytes);
    }

    #[test]
    fn a_marker_in_the_chosen_folder_is_adopted_in_place() {
        let fx = Fixture::new("adopt-in-place", &[("T7", LOCAL)]);
        let root = fx.volume("T7").join("Models");
        marker_at(&root, ID, "Shelf", Vec::new());
        let mut registry = fx.registry();

        let added = registry.add(&root, None, &fx.env).unwrap();

        assert_eq!(added.view.record.id, ID);
        assert_eq!(added.view.record.name, "Shelf");
        assert_eq!(added.view.root, Some(root.clone()));
        assert_eq!(
            added.view.record.volume_relative_path,
            PathBuf::from("Models")
        );
        assert!(!root.join(STANDARD_LIBRARY_FOLDER).exists());
    }

    #[test]
    fn a_marker_at_the_top_of_a_volume_is_never_adopted() {
        let fx = Fixture::new("top-marker", &[("T7", LOCAL)]);
        let volume = fx.volume("T7");
        marker_at(&volume, OTHER_ID, "Top", Vec::new());
        let top_bytes = fs::read(volume.join(MARKER_FILE)).unwrap();
        let mut registry = fx.registry();

        let added = registry.add(&volume, None, &fx.env).unwrap();

        // A root at the volume top would have an empty relative path, so
        // the drive gets the standard folder instead.
        assert_ne!(added.view.record.id, OTHER_ID);
        assert_eq!(added.view.root, Some(volume.join(STANDARD_LIBRARY_FOLDER)));
        assert_eq!(fs::read(volume.join(MARKER_FILE)).unwrap(), top_bytes);
    }

    #[test]
    fn re_adding_a_registered_library_returns_it_without_a_duplicate() {
        let fx = Fixture::new("re-add", &[("T7", LOCAL)]);
        let root = fx.volume("T7").join(STANDARD_LIBRARY_FOLDER);
        marker_at(
            &root,
            ID,
            "T7",
            vec![portable("qwen", "text/qwen.gturbo", ModelModality::Text)],
        );
        let mut registry = fx.registry();
        let first = registry.add(&fx.volume("T7"), None, &fx.env).unwrap();
        let rows = fx.rows();
        let list = fs::read(fx.list_path()).unwrap();

        // By the drive and by the library folder itself.
        let again = registry.add(&fx.volume("T7"), None, &fx.env).unwrap();
        let direct = registry.add(&root, Some("Renamed"), &fx.env).unwrap();

        assert_eq!(again.view, first.view);
        assert_eq!(direct.view, first.view);
        assert!(again.skipped.is_empty());
        assert_eq!(
            ids(&registry),
            vec![INTERNAL_LIBRARY_ID.to_string(), ID.to_string()]
        );
        assert_eq!(fx.rows(), rows);
        assert_eq!(fs::read(fx.list_path()).unwrap(), list);
    }

    #[test]
    fn an_unreadable_marker_refuses_the_add_and_is_never_overwritten() {
        let fx = Fixture::new("bad-marker", &[("T7", LOCAL)]);
        let chosen = fx.volume("T7").join("Lib");
        fs::create_dir_all(&chosen).unwrap();
        fs::write(chosen.join(MARKER_FILE), "not json").unwrap();
        let standard = fx.volume("T7").join(STANDARD_LIBRARY_FOLDER);
        fs::create_dir_all(&standard).unwrap();
        fs::write(standard.join(MARKER_FILE), "{").unwrap();
        let mut registry = fx.registry();

        for (path, marker) in [
            (chosen.clone(), chosen.join(MARKER_FILE)),
            (fx.volume("T7"), standard.join(MARKER_FILE)),
        ] {
            let before = fs::read(&marker).unwrap();
            let result = registry.add(&path, None, &fx.env);
            assert!(
                matches!(result, Err(LibraryError::Marker(MarkerError::Malformed(_)))),
                "{}: {result:?}",
                path.display()
            );
            assert_eq!(fs::read(&marker).unwrap(), before);
        }
        assert!(!chosen.join(STANDARD_LIBRARY_FOLDER).exists());
        assert!(!fx.list_path().exists());
        assert_eq!(ids(&registry), vec![INTERNAL_LIBRARY_ID.to_string()]);
    }

    #[test]
    fn every_location_refusal_is_reported_and_writes_nothing() {
        let fx = Fixture::new(
            "refusals",
            &[
                ("T7", LOCAL),
                ("NAS", "smbfs, nodev, nosuid"),
                ("NO NAME", "msdos, local, nodev, nosuid"),
                ("Installer", "hfs, local, read-only"),
            ],
        );
        // A folder under the volumes root with no mount entry of its own.
        fs::create_dir_all(fx.volume("Loose")).unwrap();
        let file = fx.volume("T7").join("file.txt");
        fs::write(&file, "x").unwrap();
        // A standard folder that leads off the drive.
        let link = fx.volume("T7").join("Link");
        fs::create_dir_all(&link).unwrap();
        let elsewhere = fx.base.join("elsewhere");
        fs::create_dir_all(&elsewhere).unwrap();
        std::os::unix::fs::symlink(&elsewhere, link.join(STANDARD_LIBRARY_FOLDER)).unwrap();
        // A standard folder on another device, as a share mounted there
        // would be. Tests cannot mount, so a link to `/dev` (devfs, its own
        // device) stands in, as in the location tests.
        let mounted = fx.volume("T7").join("Mounted");
        fs::create_dir_all(&mounted).unwrap();
        std::os::unix::fs::symlink("/dev", mounted.join(STANDARD_LIBRARY_FOLDER)).unwrap();
        assert_ne!(
            fs::metadata("/dev").unwrap().dev(),
            fs::metadata(&mounted).unwrap().dev(),
            "/dev must be its own device"
        );
        let inside_home = fx.home.join("models");
        fs::create_dir_all(&inside_home).unwrap();
        let mut registry = fx.registry();

        let cases = [
            (fx.volume("NAS"), LocationRefusal::Network),
            (
                fx.volume("NO NAME"),
                LocationRefusal::SmallFileLimit {
                    fs_type: "msdos".to_string(),
                },
            ),
            (fx.volume("Installer"), LocationRefusal::ReadOnly),
            (fx.volume("Loose"), LocationRefusal::UnknownVolume),
            (file, LocationRefusal::NotADirectory),
            (
                fx.volume("T7").join("missing"),
                LocationRefusal::NotADirectory,
            ),
            (link, LocationRefusal::NotADirectory),
            (mounted, LocationRefusal::UnknownVolume),
            (inside_home.clone(), LocationRefusal::InsideHome),
            (fx.home.clone(), LocationRefusal::InsideHome),
        ];
        for (path, expected) in cases {
            assert_eq!(
                refused(registry.add(&path, None, &fx.env)),
                expected,
                "{}",
                path.display()
            );
        }

        // The home's own disk.
        let same_disk = LocationEnv {
            home_device: fs::metadata(&fx.base).unwrap().dev(),
            ..fx.env.clone()
        };
        assert_eq!(
            refused(registry.add(&fx.volume("T7"), None, &same_disk)),
            LocationRefusal::SameDiskAsHome
        );

        assert!(!fx.list_path().exists());
        assert_eq!(ids(&registry), vec![INTERNAL_LIBRARY_ID.to_string()]);
        for volume in ["T7", "NAS", "NO NAME", "Installer", "Loose"] {
            assert!(
                !fx.volume(volume).join(STANDARD_LIBRARY_FOLDER).exists(),
                "{volume}"
            );
        }
        assert!(!inside_home.join(STANDARD_LIBRARY_FOLDER).exists());
        assert!(fs::read_dir(&elsewhere).unwrap().next().is_none());
    }

    #[test]
    fn a_library_that_would_contain_the_home_is_refused() {
        let fx = Fixture::new("contains-home", &[("T7", LOCAL)]);
        // A home moved onto the drive, as TURBOSPARK_HOME allows.
        let standard = fx.volume("T7").join(STANDARD_LIBRARY_FOLDER);
        let home = standard.join("home");
        fs::create_dir_all(&home).unwrap();
        let mut registry = LibraryRegistry::load(&home);

        assert_eq!(
            refused(registry.add(&fx.volume("T7"), None, &fx.env)),
            LocationRefusal::InsideHome
        );
        assert!(!standard.join(MARKER_FILE).exists());
        assert!(!home.join(LIBRARIES_FILE).exists());
    }

    #[test]
    fn a_library_inside_or_around_another_is_refused() {
        let fx = Fixture::new("nested", &[("T7", LOCAL), ("T8", LOCAL), ("T9", LOCAL)]);
        let mut registry = fx.registry();
        // T7: a library at the standard folder.
        registry.add(&fx.volume("T7"), None, &fx.env).unwrap();
        let inner = fx.volume("T7").join(STANDARD_LIBRARY_FOLDER).join("text");
        fs::create_dir_all(&inner).unwrap();
        // T8: a library inside a folder named TurboSpark.
        let shelf = fx.volume("T8").join(STANDARD_LIBRARY_FOLDER).join("Inner");
        marker_at(&shelf, OTHER_ID, "Shelf", Vec::new());
        registry.add(&shelf, None, &fx.env).unwrap();
        // T9: a library that is not connected, compared at its recorded
        // place because the marker there cannot be read and so could be its
        // own.
        let offline_root = fx.volume("T9").join(STANDARD_LIBRARY_FOLDER);
        let offline_inner = offline_root.join("models");
        fs::create_dir_all(&offline_inner).unwrap();
        fs::write(offline_root.join(MARKER_FILE), "{").unwrap();
        let mut list: serde_json::Value =
            serde_json::from_slice(&fs::read(fx.list_path()).unwrap()).unwrap();
        list["libraries"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "id": ID,
                "name": "Offline",
                "volume_relative_path": "TurboSpark",
                "last_mount_path": fx.volume("T9"),
                "default_for_downloads": false,
                "added_on": "2026-10-14"
            }));
        fs::write(fx.list_path(), serde_json::to_vec(&list).unwrap()).unwrap();
        let mut registry = fx.registry();
        let before = fs::read(fx.list_path()).unwrap();

        // Inside a library.
        assert_eq!(
            refused(registry.add(&inner, None, &fx.env)),
            LocationRefusal::NestedIn {
                library: "T7".to_string()
            }
        );
        // Around a library: the new root `T8/TurboSpark` contains Inner.
        assert_eq!(
            refused(registry.add(&fx.volume("T8"), None, &fx.env)),
            LocationRefusal::NestedIn {
                library: "Shelf".to_string()
            }
        );
        // Inside a library that is not connected.
        assert_eq!(
            refused(registry.add(&offline_inner, None, &fx.env)),
            LocationRefusal::NestedIn {
                library: "Offline".to_string()
            }
        );

        assert_eq!(fs::read(fx.list_path()).unwrap(), before);
        assert!(!inner.join(STANDARD_LIBRARY_FOLDER).exists());
        assert!(!fx
            .volume("T8")
            .join(STANDARD_LIBRARY_FOLDER)
            .join(MARKER_FILE)
            .exists());
        assert!(!offline_inner.join(STANDARD_LIBRARY_FOLDER).exists());
    }

    #[test]
    fn removing_forgets_the_library_and_its_rows_and_leaves_its_files() {
        let fx = Fixture::new("remove", &[("T7", LOCAL)]);
        let root = fx.volume("T7").join(STANDARD_LIBRARY_FOLDER);
        marker_at(
            &root,
            ID,
            "T7",
            vec![
                portable("qwen", "text/qwen.gturbo", ModelModality::Text),
                portable("whisper", "audio/whisper.gturbo", ModelModality::Audio),
            ],
        );
        for install in ["text/qwen.gturbo", "audio/whisper.gturbo"] {
            fs::create_dir_all(root.join(install)).unwrap();
            fs::write(root.join(install).join("manifest.json"), install).unwrap();
        }
        let this_mac = installed(
            "gemma",
            ModelModality::Text,
            fx.home.join("models/text/gemma.gturbo"),
            None,
        );
        Store::new(&fx.home).record(&this_mac).unwrap();
        let mut registry = fx.registry();
        registry.add(&fx.volume("T7"), None, &fx.env).unwrap();
        let library_rows: Vec<InstalledModel> = fx
            .rows()
            .into_iter()
            .filter(|row| row.library.as_deref() == Some(ID))
            .collect();
        assert_eq!(library_rows.len(), 2);
        let files = snapshot(&fx.volume("T7"));

        registry.remove(ID).unwrap();

        assert_eq!(ids(&registry), vec![INTERNAL_LIBRARY_ID.to_string()]);
        assert_eq!(ids(&fx.registry()), vec![INTERNAL_LIBRARY_ID.to_string()]);
        assert_eq!(fx.rows(), vec![this_mac.clone()]);
        assert_eq!(snapshot(&fx.volume("T7")), files);

        // The marker kept the rows, so adding the drive again restores them.
        let again = registry.add(&fx.volume("T7"), None, &fx.env).unwrap();
        assert_eq!(again.view.record.id, ID);
        assert_eq!(again.view.model_count, 2);
        let restored: Vec<InstalledModel> = fx
            .rows()
            .into_iter()
            .filter(|row| row.library.as_deref() == Some(ID))
            .collect();
        assert_eq!(restored, library_rows);
        assert!(fx.rows().contains(&this_mac));
    }

    #[test]
    fn this_mac_and_unknown_libraries_cannot_be_removed() {
        let fx = Fixture::new("remove-refused", &[("T7", LOCAL)]);
        let mut registry = fx.registry();
        registry.add(&fx.volume("T7"), None, &fx.env).unwrap();
        let list = fs::read(fx.list_path()).unwrap();

        assert_eq!(
            registry.remove(INTERNAL_LIBRARY_ID),
            Err(LibraryError::Internal)
        );
        assert_eq!(
            registry.remove(OTHER_ID),
            Err(LibraryError::UnknownLibrary(OTHER_ID.to_string()))
        );
        assert_eq!(fs::read(fx.list_path()).unwrap(), list);
    }

    #[test]
    fn exactly_one_library_is_the_default() {
        let fx = Fixture::new("defaults", &[("T7", LOCAL), ("T8", LOCAL)]);
        let mut registry = fx.registry();
        assert_eq!(defaults(&registry), vec![INTERNAL_LIBRARY_ID]);
        let a = registry
            .add(&fx.volume("T7"), None, &fx.env)
            .unwrap()
            .view
            .record
            .id;
        let b = registry
            .add(&fx.volume("T8"), None, &fx.env)
            .unwrap()
            .view
            .record
            .id;
        // Adding never moves the default.
        assert_eq!(defaults(&registry), vec![INTERNAL_LIBRARY_ID]);

        registry.set_default(&a).unwrap();
        assert_eq!(defaults(&registry), vec![a.clone()]);
        assert_eq!(defaults(&fx.registry()), vec![a.clone()]);

        registry.set_default(&b).unwrap();
        assert_eq!(defaults(&fx.registry()), vec![b.clone()]);

        assert_eq!(
            registry.set_default(OTHER_ID),
            Err(LibraryError::UnknownLibrary(OTHER_ID.to_string()))
        );
        assert_eq!(defaults(&fx.registry()), vec![b.clone()]);

        // Removing the default hands it back to This Mac.
        registry.remove(&b).unwrap();
        assert_eq!(defaults(&fx.registry()), vec![INTERNAL_LIBRARY_ID]);

        registry.set_default(&a).unwrap();
        registry.set_default(INTERNAL_LIBRARY_ID).unwrap();
        assert_eq!(defaults(&fx.registry()), vec![INTERNAL_LIBRARY_ID]);
    }

    #[test]
    fn a_hand_edited_list_is_read_with_exactly_one_default() {
        let fx = Fixture::new("defaults-edited", &[]);
        let record = |id: &str, default: bool| {
            serde_json::json!({
                "id": id,
                "name": id,
                "volume_relative_path": "TurboSpark",
                "last_mount_path": "/Volumes/T7",
                "default_for_downloads": default,
                "added_on": "2026-10-14"
            })
        };
        let write = |internal: Option<bool>, records: Vec<serde_json::Value>| {
            let mut list = serde_json::json!({ "format": 1, "libraries": records });
            if let Some(internal) = internal {
                list["internal_is_default"] = internal.into();
            }
            fs::write(fx.list_path(), serde_json::to_vec(&list).unwrap()).unwrap();
        };

        // Two flagged: the first wins.
        write(Some(false), vec![record(ID, true), record(OTHER_ID, true)]);
        assert_eq!(defaults(&fx.registry()), vec![ID]);
        // None flagged: This Mac.
        write(Some(false), vec![record(ID, false)]);
        assert_eq!(defaults(&fx.registry()), vec![INTERNAL_LIBRARY_ID]);
        // This Mac flagged too: This Mac.
        write(Some(true), vec![record(ID, true)]);
        assert_eq!(defaults(&fx.registry()), vec![INTERNAL_LIBRARY_ID]);
        // Absent flag: This Mac, as on a first run.
        write(None, vec![record(ID, false)]);
        assert_eq!(defaults(&fx.registry()), vec![INTERNAL_LIBRARY_ID]);
    }

    /// Stands in another process adding the drive `volume` once this
    /// thread's next commit has started.
    fn another_writer_adds(fx: &Fixture, volume: &str) {
        let home = fx.home.clone();
        let env = fx.env.clone();
        let volume = fx.volume(volume);
        before_next_commit(move || {
            LibraryRegistry::load(&home)
                .add(&volume, None, &env)
                .unwrap();
        });
    }

    #[test]
    fn every_write_applies_to_the_list_as_it_is_on_disk() {
        let fx = Fixture::new(
            "lost-update",
            &[("T7", LOCAL), ("T8", LOCAL), ("T9", LOCAL), ("T10", LOCAL)],
        );
        let mut registry = fx.registry();

        another_writer_adds(&fx, "T8");
        let t7 = registry
            .add(&fx.volume("T7"), None, &fx.env)
            .unwrap()
            .view
            .record
            .id;
        assert_eq!(names(&fx.registry()), vec!["This Mac", "T8", "T7"]);
        assert_eq!(names(&registry), vec!["This Mac", "T8", "T7"]);

        another_writer_adds(&fx, "T9");
        registry.remove(&t7).unwrap();
        assert_eq!(names(&fx.registry()), vec!["This Mac", "T8", "T9"]);

        another_writer_adds(&fx, "T10");
        let t8 = fx.registry().libraries()[1].id.clone();
        registry.set_default(&t8).unwrap();
        let reloaded = fx.registry();
        assert_eq!(names(&reloaded), vec!["This Mac", "T8", "T9", "T10"]);
        assert_eq!(defaults(&reloaded), vec![t8]);
    }

    #[test]
    fn a_stale_temp_file_does_not_affect_reading_the_list() {
        let fx = Fixture::new("stale-temp", &[("T7", LOCAL), ("T8", LOCAL)]);
        let mut registry = fx.registry();
        registry.add(&fx.volume("T7"), None, &fx.env).unwrap();
        let before = fx.registry().libraries();
        let list = fs::read(fx.list_path()).unwrap();
        // Writers that crashed after part of their temp file and before the
        // rename, one named as this process's next temp file would be.
        for name in [
            format!("{LIBRARIES_FILE}.tmp-{}-0", std::process::id()),
            format!("{LIBRARIES_FILE}.tmp-1-0"),
        ] {
            fs::write(fx.home.join(name), &list[..list.len() / 2]).unwrap();
        }

        let reloaded = fx.registry();
        assert_eq!(reloaded.libraries(), before);
        assert_eq!(reloaded.load_report(), &LoadReport::default());

        // And the next write still lands.
        registry.add(&fx.volume("T8"), None, &fx.env).unwrap();
        assert_eq!(names(&fx.registry()), vec!["This Mac", "T7", "T8"]);
    }

    #[test]
    fn an_unparseable_list_is_quarantined_rather_than_overwritten() {
        let fx = Fixture::new("corrupt", &[("T7", LOCAL)]);
        fs::write(fx.list_path(), "{ not json").unwrap();

        let mut registry = fx.registry();

        assert_eq!(ids(&registry), vec![INTERNAL_LIBRARY_ID.to_string()]);
        assert_eq!(defaults(&registry), vec![INTERNAL_LIBRARY_ID]);
        let quarantined = registry.load_report().quarantined.clone().unwrap();
        assert!(quarantined
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("libraries.json.corrupt-"));
        assert_eq!(quarantined.parent(), Some(fx.home.as_path()));
        assert_eq!(fs::read(&quarantined).unwrap(), b"{ not json");
        assert!(!fx.list_path().exists());

        registry.add(&fx.volume("T7"), None, &fx.env).unwrap();
        assert_eq!(fs::read(&quarantined).unwrap(), b"{ not json");

        // A second one in the same second keeps the first.
        fs::write(fx.list_path(), "[]").unwrap();
        let second = fx.registry().load_report().quarantined.clone().unwrap();
        assert_ne!(second, quarantined);
        assert_eq!(fs::read(&second).unwrap(), b"[]");
        assert_eq!(fs::read(&quarantined).unwrap(), b"{ not json");
    }

    #[test]
    fn invalid_records_are_dropped_and_reported() {
        let fx = Fixture::new("invalid-records", &[]);
        let record = |id: &str, relative: &str| {
            serde_json::json!({
                "id": id,
                "name": "T7",
                "volume_relative_path": relative,
                "last_mount_path": "/Volumes/T7",
                "default_for_downloads": false,
                "added_on": "2026-10-14"
            })
        };
        let list = serde_json::json!({
            "format": 1,
            "internal_is_default": true,
            "libraries": [
                record(ID, "TurboSpark"),
                record(INTERNAL_LIBRARY_ID, "TurboSpark"),
                record("a:b", "TurboSpark"),
                record("a/b", "TurboSpark"),
                record("", "TurboSpark"),
                record(OTHER_ID, "/Volumes/T7/TurboSpark"),
                record(OTHER_ID, "../TurboSpark"),
                record(OTHER_ID, "TurboSpark/../x"),
                record(OTHER_ID, ""),
                { "id": OTHER_ID, "name": "no other fields" },
                // A repeated id.
                record(ID, "Other"),
                {
                    "id": OTHER_ID,
                    "name": "T7",
                    "volume_relative_path": "TurboSpark",
                    "last_mount_path": "Volumes/T7",
                    "default_for_downloads": false,
                    "added_on": "2026-10-14"
                },
            ]
        });
        fs::write(fx.list_path(), serde_json::to_vec(&list).unwrap()).unwrap();

        let registry = fx.registry();

        assert_eq!(
            ids(&registry),
            vec![INTERNAL_LIBRARY_ID.to_string(), ID.to_string()]
        );
        assert_eq!(
            registry.libraries()[1].volume_relative_path,
            PathBuf::from("TurboSpark")
        );
        assert_eq!(
            registry.load_report().dropped.len(),
            11,
            "{:?}",
            registry.load_report()
        );
        assert_eq!(registry.load_report().quarantined, None);
    }

    #[test]
    fn a_newer_list_is_read_but_never_rewritten() {
        let fx = Fixture::new("newer", &[("T7", LOCAL)]);
        let list = serde_json::json!({
            "format": 2,
            "internal_is_default": true,
            "future": "field",
            "libraries": [{
                "id": ID,
                "name": "T9",
                "volume_relative_path": "TurboSpark",
                "last_mount_path": "/Volumes/T9",
                "default_for_downloads": false,
                "added_on": "2026-10-14"
            }]
        });
        fs::write(fx.list_path(), serde_json::to_vec(&list).unwrap()).unwrap();
        let row = installed(
            "qwen",
            ModelModality::Text,
            PathBuf::from("/Volumes/T9/TurboSpark/text/qwen.gturbo"),
            Some(ID),
        );
        Store::new(&fx.home).record(&row).unwrap();
        let before = fs::read(fx.list_path()).unwrap();
        let mut registry = fx.registry();

        assert_eq!(
            ids(&registry),
            vec![INTERNAL_LIBRARY_ID.to_string(), ID.to_string()]
        );
        assert!(matches!(
            registry.add(&fx.volume("T7"), None, &fx.env),
            Err(LibraryError::Io(_))
        ));
        assert!(matches!(registry.remove(ID), Err(LibraryError::Io(_))));
        assert!(matches!(registry.set_default(ID), Err(LibraryError::Io(_))));
        assert_eq!(fs::read(fx.list_path()).unwrap(), before);
        assert_eq!(fx.rows(), vec![row]);
        assert!(!fx.volume("T7").join(STANDARD_LIBRARY_FOLDER).exists());
    }

    #[test]
    fn this_mac_counts_its_own_rows_only() {
        let fx = Fixture::new("count", &[]);
        let store = Store::new(&fx.home);
        for row in [
            installed(
                "qwen",
                ModelModality::Text,
                fx.home.join("models/text/qwen.gturbo"),
                None,
            ),
            installed(
                "flux",
                ModelModality::Image,
                fx.home.join("models/image/flux.gturbo"),
                None,
            ),
            // A custom location (`pull --out`) is not This Mac's.
            installed(
                "custom",
                ModelModality::Text,
                fx.base.join("elsewhere/custom.gturbo"),
                None,
            ),
            installed(
                "whisper",
                ModelModality::Audio,
                PathBuf::from("/Volumes/T7/TurboSpark/audio/whisper.gturbo"),
                Some(ID),
            ),
        ] {
            store.record(&row).unwrap();
        }
        let registry = fx.registry();
        let this_mac = &registry.libraries()[0];

        let view = registry.view(this_mac, &fx.env);

        assert_eq!(view.model_count, 2);
        assert!(view.available);
        assert_eq!(view.root, Some(fx.home.join("models")));
    }

    #[test]
    fn another_drive_with_the_same_name_is_not_inside_an_offline_library() {
        let fx = Fixture::new("same-name", &[("T7", LOCAL)]);
        let mut registry = fx.registry();
        let first = registry
            .add(&fx.volume("T7"), None, &fx.env)
            .unwrap()
            .view
            .record;
        let standard = fx.volume("T7").join(STANDARD_LIBRARY_FOLDER);

        // That drive is unplugged, and another drive also named T7, with an
        // empty TurboSpark folder, mounts in its place.
        fs::remove_dir_all(&standard).unwrap();
        fs::create_dir(&standard).unwrap();
        let fresh = registry.add(&fx.volume("T7"), None, &fx.env).unwrap();

        assert_ne!(fresh.view.record.id, first.id);
        assert!(fresh.view.available);
        assert!(!registry.view(&first, &fx.env).available);

        // And a third T7 that carries its own library is adopted with its
        // models.
        fs::remove_dir_all(&standard).unwrap();
        marker_at(
            &standard,
            OTHER_ID,
            "Backup",
            vec![portable("qwen", "text/qwen.gturbo", ModelModality::Text)],
        );
        let adopted = registry.add(&fx.volume("T7"), None, &fx.env).unwrap();

        assert_eq!(adopted.view.record.id, OTHER_ID);
        assert_eq!(adopted.view.model_count, 1);
        assert_eq!(
            ids(&fx.registry()),
            vec![
                INTERNAL_LIBRARY_ID.to_string(),
                first.id.clone(),
                fresh.view.record.id.clone(),
                OTHER_ID.to_string(),
            ]
        );
    }

    #[test]
    fn a_failed_list_write_forgets_the_rows_the_add_merged() {
        let fx = Fixture::new("rollback", &[("T7", LOCAL)]);
        marker_at(
            &fx.volume("T7").join(STANDARD_LIBRARY_FOLDER),
            ID,
            "T7",
            vec![
                portable("qwen", "text/qwen.gturbo", ModelModality::Text),
                portable("whisper", "audio/whisper.gturbo", ModelModality::Audio),
            ],
        );
        let this_mac = installed(
            "gemma",
            ModelModality::Text,
            fx.home.join("models/text/gemma.gturbo"),
            None,
        );
        Store::new(&fx.home).record(&this_mac).unwrap();
        let mut registry = fx.registry();
        // A newer TurboSpark writes the list after this add merged its rows
        // and before its list write, which must then refuse.
        let newer = serde_json::to_vec(&serde_json::json!({
            "format": 2,
            "internal_is_default": true,
            "libraries": []
        }))
        .unwrap();
        let list_path = fx.list_path();
        let written = newer.clone();
        before_next_commit(move || fs::write(&list_path, &written).unwrap());

        let result = registry.add(&fx.volume("T7"), None, &fx.env);

        assert!(matches!(result, Err(LibraryError::Io(_))), "{result:?}");
        assert_eq!(fx.rows(), vec![this_mac]);
        assert_eq!(fs::read(fx.list_path()).unwrap(), newer);
    }

    #[test]
    fn a_failed_row_write_lists_nothing() {
        let fx = Fixture::new("row-write-fails", &[("T7", LOCAL)]);
        let root = fx.volume("T7").join(STANDARD_LIBRARY_FOLDER);
        marker_at(
            &root,
            ID,
            "T7",
            vec![portable("qwen", "text/qwen.gturbo", ModelModality::Text)],
        );
        let marker_bytes = fs::read(root.join(MARKER_FILE)).unwrap();
        // A record that cannot be replaced.
        fs::create_dir_all(fx.home.join("installed.json")).unwrap();
        let mut registry = fx.registry();

        let result = registry.add(&fx.volume("T7"), None, &fx.env);

        assert!(matches!(result, Err(LibraryError::Io(_))), "{result:?}");
        assert!(!fx.list_path().exists());
        assert_eq!(ids(&fx.registry()), vec![INTERNAL_LIBRARY_ID.to_string()]);
        assert_eq!(fs::read(root.join(MARKER_FILE)).unwrap(), marker_bytes);
    }

    #[test]
    fn list_writes_replace_the_file_rather_than_rewriting_it_in_place() {
        use std::io::Read;
        let fx = Fixture::new("replace-list", &[("T7", LOCAL), ("T8", LOCAL)]);
        let mut registry = fx.registry();
        registry.add(&fx.volume("T7"), None, &fx.env).unwrap();
        let before = fs::read_to_string(fx.list_path()).unwrap();
        let mut reader = fs::File::open(fx.list_path()).unwrap();

        registry.add(&fx.volume("T8"), None, &fx.env).unwrap();

        // A reader that opened the list before the write still sees every
        // byte of the old one, so a write cut short never leaves it partial.
        let mut seen = String::new();
        reader.read_to_string(&mut seen).unwrap();
        assert_eq!(seen, before, "the old list was rewritten in place");
        assert_eq!(names(&fx.registry()), vec!["This Mac", "T7", "T8"]);
    }

    #[test]
    fn removing_a_library_with_no_rows_leaves_the_install_record_alone() {
        let fx = Fixture::new("remove-empty", &[("T7", LOCAL), ("T8", LOCAL)]);
        let record = fx.home.join("installed.json");
        let mut registry = fx.registry();

        let t7 = registry
            .add(&fx.volume("T7"), None, &fx.env)
            .unwrap()
            .view
            .record
            .id;
        assert!(!record.exists());
        registry.remove(&t7).unwrap();
        assert!(!record.exists());

        // With other installs recorded, the record is not rewritten either.
        let this_mac = installed(
            "gemma",
            ModelModality::Text,
            fx.home.join("models/text/gemma.gturbo"),
            None,
        );
        Store::new(&fx.home).record(&this_mac).unwrap();
        let bytes = fs::read(&record).unwrap();
        let inode = fs::metadata(&record).unwrap().ino();
        let t8 = registry
            .add(&fx.volume("T8"), None, &fx.env)
            .unwrap()
            .view
            .record
            .id;
        registry.remove(&t8).unwrap();
        assert_eq!(fs::read(&record).unwrap(), bytes);
        assert_eq!(fs::metadata(&record).unwrap().ino(), inode);
    }

    #[test]
    fn a_newer_list_this_build_cannot_parse_is_never_quarantined() {
        let fx = Fixture::new("newer-unparsed", &[("T7", LOCAL)]);
        let list = serde_json::json!({
            "format": 3,
            "libraries": { "a": "shape this build does not know" }
        });
        fs::write(fx.list_path(), serde_json::to_vec(&list).unwrap()).unwrap();
        let before = fs::read(fx.list_path()).unwrap();

        let mut registry = fx.registry();

        assert_eq!(registry.load_report().quarantined, None);
        assert!(registry.load_report().unreadable.is_some());
        assert_eq!(ids(&registry), vec![INTERNAL_LIBRARY_ID.to_string()]);
        assert!(matches!(
            registry.add(&fx.volume("T7"), None, &fx.env),
            Err(LibraryError::Io(_))
        ));
        assert!(matches!(
            registry.set_default(INTERNAL_LIBRARY_ID),
            Err(LibraryError::Io(_))
        ));
        assert_eq!(fs::read(fx.list_path()).unwrap(), before);
        let entries: Vec<String> = fs::read_dir(&fx.home)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(entries, vec![LIBRARIES_FILE]);
    }
}
