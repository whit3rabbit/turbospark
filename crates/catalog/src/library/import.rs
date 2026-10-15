//! Converting a store that Move Store moved into a library (9.2, 9.4, 9.5).
//!
//! Move Store copied `models`, `installed.json`, and `models.json` into a
//! new root and made that root the store home. The import registers the
//! root's `models` folder as a library where it is, so no text, image, or
//! audio model moves, and records the moved store's installs as rows of that
//! library. Embedding and OCR models install only into This Mac (2.5), so
//! their folders are copied into the home, as is the catalog override the
//! move took along. Nothing under the moved root is changed or deleted, apart
//! from the library marker written into its `models` folder.

use super::LibraryRegistry;
use crate::audio_catalog::{AudioCatalog, RECEIPT_FILENAME};
use crate::library::marker::validate_relative_path;
use crate::library::transfer::copy_folder;
use crate::library::{
    new_library_id, replace_file, validate_alias, volume_root_of, LibraryError, LibraryMarker,
    LibraryRecord, LibraryView, LocationEnv, LocationRefusal, MarkerError, PortableRow,
    STANDARD_LIBRARY_FOLDER,
};
use crate::store::{external_library, install_in, portable_modality};
use crate::{InstalledModel, ModelModality, Store};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// The folders under a models folder whose installs stay on This Mac (2.5).
const HOME_ONLY_FOLDERS: [&str; 2] = ["embeddings", "ocr"];

/// The catalog override in a store root, which Move Store moved along.
const CATALOG_OVERRIDE: &str = "models.json";

/// A store root's record of its installs.
const INSTALL_RECORD: &str = "installed.json";

/// The outcome of [`LibraryRegistry::import_moved_root`]. A design
/// addition: the design returned the view alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportedStore {
    /// The library the moved store's `models` folder now is.
    pub view: LibraryView,
    /// Aliases of rows not recorded: the alias and modality already belong
    /// to This Mac, a custom location, or another library (the 1.2
    /// decision), the row repeats an earlier one, its path cannot be kept in
    /// the library, or it names a library the moved store had listed.
    pub skipped: Vec<String>,
    /// Aliases of rows outside `<root>/models`, such as `pull --out`
    /// installs, recorded with no library, as custom locations, so they stay
    /// listed too.
    pub custom: Vec<String>,
    /// Aliases of catalog audio installs in the library that have no audio
    /// receipt, such as music installed on This Mac before the move. They
    /// stay listed (9.2), but a library cannot check or delete them (2.9),
    /// so the app asks the user to move them to This Mac and adopt them
    /// there.
    pub needs_adoption: Vec<String>,
    /// `embeddings/<alias>` and `ocr/<alias>` for each install folder copied
    /// into the home (9.5).
    pub copied_to_home: Vec<String>,
    /// Whether `<root>/models.json` was copied into the home, which had none.
    pub restored_catalog_override: bool,
    /// Each home copy that failed, as `(embeddings/<alias>, reason)`, or
    /// `models.json` for the catalog override. A failed copy leaves nothing
    /// in the home and does not stop the import, and the next import tries
    /// it again. Reasons name paths on the moved store's drive, so a caller
    /// showing one truncates it.
    pub copy_failures: Vec<(String, String)>,
}

/// What an import did besides listing the library.
#[derive(Default)]
struct Done {
    skipped: Vec<String>,
    custom: Vec<String>,
    copied_to_home: Vec<String>,
    restored_catalog_override: bool,
    copy_failures: Vec<(String, String)>,
}

/// The rows an import records.
#[derive(Default)]
struct Plan {
    /// Rows of the library, each with its marker row.
    library: Vec<(InstalledModel, PortableRow)>,
    /// Rows outside the library with no library, kept as custom locations.
    custom: Vec<InstalledModel>,
    skipped: Vec<String>,
}

impl LibraryRegistry {
    /// Registers the `models` folder of the store Move Store moved to `root`
    /// as a library, with the moved store's installs as its rows (9.2), and
    /// copies its embedding and OCR models into the home (9.5). `env` is a
    /// parameter, as for every other location lookup (a refinement of the
    /// design's signature).
    ///
    /// The location refusals do not apply. Those govern a place the user
    /// picks now, and Move Store accepted places a library may not be, the
    /// boot disk among them, so a store it moved there imports and stays
    /// listed. Only nesting is refused ([`LocationRefusal::NestedIn`]): a
    /// models folder inside or around a connected library would list the
    /// same files twice. A root that is not there, as on a drive that is not
    /// connected, is [`LibraryError::NotConnected`]. A root that is the home
    /// is This Mac already, and is returned as it is.
    ///
    /// The moved store's `installed.json` is its only list of text and audio
    /// installs, so one that exists but cannot be read or parsed refuses the
    /// import, which a later import retries, rather than list the library
    /// with none of them. A marker that cannot be read refuses it too, and is
    /// never overwritten (2.2). Every refusal comes before any marker,
    /// registry, or list write. A refusal found when they are read again
    /// after the home copies leaves those copies made, which is harmless:
    /// they never replace anything, and the next import skips them.
    ///
    /// Idempotent by marker id: a `models` folder whose marker names a
    /// listed library returns that library, and only makes the home copies
    /// an earlier import could not. They never replace anything, so once all
    /// are made it writes nothing.
    ///
    /// Rows:
    /// - A row under `<root>/models` becomes a row of the library, at its
    ///   path relative to that folder. An audio row is placed at its managed
    ///   path, `audio/<alias>.gturbo`, where the store always found it.
    /// - A row elsewhere with no library is recorded as it is, a custom
    ///   location (the choice for rows outside `<root>/models`).
    /// - A row whose alias and modality are already This Mac's, a custom
    ///   location's, or another library's is skipped and reported (1.2), as
    ///   are a repeat and a row naming a library the moved store had listed.
    /// - Rows the marker lists that `installed.json` does not, as after an
    ///   import on another Mac, are recorded too, as adding a library adopts
    ///   a marker's rows.
    ///
    /// Catalog audio without its receipt is recorded and reported in
    /// [`ImportedStore::needs_adoption`] rather than left out.
    ///
    /// Home copies never fail the import: users moved stores off a full
    /// disk, and listing their models must not wait on the home's space. A
    /// failed copy is reported in [`ImportedStore::copy_failures`].
    ///
    /// The steps keep every invariant when one fails or the process stops
    /// between two of them:
    /// 1. Every read and check, with nothing written.
    /// 2. The home copies, each through a staging folder and a rename, never
    ///    replacing anything there.
    /// 3. The marker, the list, and the home's rows read again, since the
    ///    copies can take minutes, and the rows worked out from them.
    /// 4. The marker, with every row the library is about to hold, so it
    ///    never lacks a row the registry holds for it (2.7).
    /// 5. The registry rows, in one write. If it fails, the marker keeps
    ///    extra rows, which is allowed, and the next import reuses its id.
    /// 6. The library list, re-read before its rename. If it cannot be
    ///    written, the library's rows are forgotten again, as when adding a
    ///    library, and the marker keeps them for the next import.
    pub fn import_moved_root(
        &mut self,
        root: &Path,
        env: &LocationEnv,
    ) -> Result<ImportedStore, LibraryError> {
        let moved = match root.canonicalize() {
            Ok(moved) => moved,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Err(LibraryError::NotConnected(root.to_path_buf()));
            }
            Err(e) => return Err(io_error("reading", root, e)),
        };
        let home = self
            .home
            .canonicalize()
            .unwrap_or_else(|_| self.home.clone());
        // Before the models folder is looked for, so a home that has none yet
        // is This Mac too. Listing This Mac's own folder as a library would
        // list each of its installs twice.
        if moved == home {
            let this_mac = self.libraries().remove(0);
            return self.outcome(this_mac, &home, env, Done::default());
        }
        let models = moved_models(root, &moved)?;

        // 1. Reads and checks.
        let existing = LibraryMarker::read(&models).map_err(LibraryError::Marker)?;
        self.reread()?;
        self.ensure_writable()?;
        if let Some(listed) = existing.as_ref().and_then(|marker| self.record(&marker.id)) {
            let done = home_copies(&models, &moved, &self.home);
            return self.outcome(listed, &models, env, done);
        }
        refuse_read_only(existing.as_ref())?;
        let moved_rows = read_install_record(&moved)?;
        let volume = volume_root_of(&models, env).map_err(|e| io_error("reading", &models, e))?;
        let relative = models
            .strip_prefix(&volume)
            .map(Path::to_path_buf)
            .unwrap_or_default();
        if validate_relative_path(&relative).is_err() {
            return Err(LibraryError::Io(format!(
                "{} is not a folder below the top of its drive",
                models.display()
            )));
        }
        // The import's own library is never compared with itself here: a
        // marker id that is listed returned above, so every listed library
        // is another one.
        if let Some(library) = self.covering_library(&models, env) {
            return Err(LibraryError::Refused(LocationRefusal::NestedIn { library }));
        }

        // 2. The home copies.
        let mut done = home_copies(&models, &moved, &self.home);
        #[cfg(test)]
        if let Some(hook) = AFTER_COPIES.with(|slot| slot.borrow_mut().take()) {
            hook();
        }

        // 3. Read again after the copies.
        let existing = LibraryMarker::read(&models).map_err(LibraryError::Marker)?;
        self.reread()?;
        self.ensure_writable()?;
        if let Some(listed) = existing.as_ref().and_then(|marker| self.record(&marker.id)) {
            // Another process imported it meanwhile.
            return self.outcome(listed, &models, env, done);
        }
        refuse_read_only(existing.as_ref())?;
        let fresh = existing.is_none();
        let mut marker = match existing {
            Some(marker) => marker,
            None => {
                let id = new_library_id()
                    .map_err(|e| LibraryError::Io(format!("creating a library id: {e}")))?;
                LibraryMarker::new(id, default_name(&models, &volume))
            }
        };
        let store = Store::new(&self.home);
        let plan = plan_rows(root, &models, &marker, moved_rows, &store.read_rows());

        // 4. The marker.
        let mut updated = fresh;
        for (_, portable) in &plan.library {
            if marker.models.contains(portable) {
                continue;
            }
            let modality = portable_modality(portable);
            marker.models.retain(|kept| {
                !(kept.alias == portable.alias && portable_modality(kept) == modality)
            });
            marker.models.push(portable.clone());
            updated = true;
        }
        if updated {
            marker.write(&models).map_err(LibraryError::Marker)?;
        }
        // 5. The registry rows.
        let rows: Vec<InstalledModel> = plan
            .library
            .iter()
            .map(|(row, _)| row.clone())
            .chain(plan.custom.iter().cloned())
            .collect();
        if !rows.is_empty() {
            store.record_rows(&rows).map_err(LibraryError::Io)?;
        }
        // 6. The list.
        let name = Some(marker.name.trim())
            .filter(|name| !name.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| default_name(&models, &volume));
        let today = crate::install::today();
        let record = LibraryRecord {
            id: marker.id.clone(),
            name,
            volume_relative_path: relative,
            last_mount_path: volume,
            default_for_downloads: false,
            added_on: today.clone(),
            // Connected right now, as for an added library (3.8).
            last_seen: Some(today),
        };
        let committed = self.commit(|list| {
            // Another process may have imported it meanwhile.
            if !list.contains(&record.id) {
                list.records.push(record.clone());
            }
            Ok(())
        });
        if let Err(e) = committed {
            let _ = store.forget_library_rows(&record.id);
            return Err(e);
        }
        let listed = self.record(&record.id).unwrap_or(record);
        done.skipped = plan.skipped;
        done.custom = plan.custom.into_iter().map(|row| row.alias).collect();
        self.outcome(listed, &models, env, done)
    }

    /// The outcome for the library `record`, whose rows are checked for
    /// receipt-less catalog audio where it is now, else at `models`.
    fn outcome(
        &self,
        record: LibraryRecord,
        models: &Path,
        env: &LocationEnv,
        done: Done,
    ) -> Result<ImportedStore, LibraryError> {
        let view = self.view(&record, env);
        let needs_adoption = if record.is_internal() {
            Vec::new()
        } else {
            let rows: Vec<InstalledModel> = Store::new(&self.home)
                .read_rows()
                .into_iter()
                .filter(|row| row.library.as_deref() == Some(record.id.as_str()))
                .collect();
            receipt_less(view.root.as_deref().unwrap_or(models), &rows)?
        };
        Ok(ImportedStore {
            view,
            skipped: done.skipped,
            custom: done.custom,
            needs_adoption,
            copied_to_home: done.copied_to_home,
            restored_catalog_override: done.restored_catalog_override,
            copy_failures: done.copy_failures,
        })
    }
}

/// The `models` folder of the moved store at the canonical `moved`, which
/// the caller gave as `root`.
fn moved_models(root: &Path, moved: &Path) -> Result<PathBuf, LibraryError> {
    let models = moved.join("models");
    // Not followed: a linked models folder would register a library where
    // Move Store never put one.
    match fs::symlink_metadata(&models) {
        Ok(found) if found.file_type().is_dir() => Ok(models),
        Ok(_) => Err(LibraryError::Io(format!(
            "{} is not a plain folder",
            models.display()
        ))),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            Err(LibraryError::NotConnected(root.to_path_buf()))
        }
        Err(e) => Err(io_error("reading", &models, e)),
    }
}

/// The rows of the moved store's `installed.json`, or none when there is no
/// such file.
///
/// Stricter than the store's own reader, which reads a record it cannot
/// parse as empty so an install can still be recorded. Here that would list
/// the library without its models for good, since a later import finds the
/// library listed and records nothing, so a record that cannot be read or
/// parsed, as a drive's read error can make it, refuses the import.
fn read_install_record(moved: &Path) -> Result<Vec<InstalledModel>, LibraryError> {
    let path = moved.join(INSTALL_RECORD);
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(io_error("reading", &path, e)),
    };
    serde_json::from_str(&text).map_err(|e| {
        LibraryError::Io(format!(
            "{} cannot be parsed, so its models cannot be listed: {e}",
            path.display()
        ))
    })
}

/// A newer TurboSpark's marker cannot take the moved store's rows.
fn refuse_read_only(marker: Option<&LibraryMarker>) -> Result<(), LibraryError> {
    match marker {
        Some(marker) if marker.is_read_only() => Err(LibraryError::Marker(MarkerError::ReadOnly {
            format: marker.format(),
        })),
        _ => Ok(()),
    }
}

/// Sorts the moved store's rows `moved`, and the rows `marker` lists, into
/// the library's and custom locations, as `import_moved_root` describes.
/// `given` is the root as the caller spelled it, `models` its canonical
/// models folder, and `held` the home's rows.
fn plan_rows(
    given: &Path,
    models: &Path,
    marker: &LibraryMarker,
    moved: Vec<InstalledModel>,
    held: &[InstalledModel],
) -> Plan {
    // Move Store wrote canonical paths, but a row recorded through another
    // spelling of the root is still under it.
    let spellings = [models.to_path_buf(), given.join("models")];
    let listed = marker
        .models
        .iter()
        .filter_map(|portable| portable.to_installed(models, &marker.id).ok());
    let mut plan = Plan::default();
    let mut seen: Vec<(String, ModelModality)> = Vec::new();
    for (row, from_marker) in moved
        .into_iter()
        .map(|row| (row, false))
        .chain(listed.map(|row| (row, true)))
    {
        let key = (row.alias.clone(), row.effective_modality());
        if seen.contains(&key) {
            // The moved store's own row wins over the marker's.
            if !from_marker {
                plan.skipped.push(row.alias);
            }
            continue;
        }
        seen.push(key.clone());
        let held = held
            .iter()
            .find(|existing| existing.alias == key.0 && existing.effective_modality() == key.1);
        // A library the moved store listed is not one this home knows.
        if external_library(&row).is_some() && !from_marker {
            plan.skipped.push(row.alias);
            continue;
        }
        match place(&row, &spellings) {
            Some(relative) => {
                let row = InstalledModel {
                    path: models.join(&relative),
                    library: Some(marker.id.clone()),
                    relative_path: Some(relative),
                    ..row
                };
                // The alias and relative path checks every marker row meets.
                let Ok(portable) = PortableRow::from_installed(&row, models) else {
                    plan.skipped.push(row.alias);
                    continue;
                };
                // Alias plus modality is unique across libraries (1.2). A row
                // already this library's is from an import that stopped early.
                match held {
                    Some(existing) if existing.library.as_deref() == Some(marker.id.as_str()) => {}
                    Some(_) => {
                        plan.skipped.push(row.alias);
                        continue;
                    }
                    None => {}
                }
                plan.library.push((row, portable));
            }
            None => {
                let row = InstalledModel {
                    library: None,
                    relative_path: None,
                    ..row
                };
                match held {
                    // Recorded by an import that stopped early.
                    Some(existing) if *existing == row => {}
                    Some(_) => {
                        plan.skipped.push(row.alias);
                        continue;
                    }
                    None => {}
                }
                plan.custom.push(row);
            }
        }
    }
    plan
}

/// Where `row` is inside the moved store's models folder, spelled any of
/// `spellings`, or `None` when it is elsewhere.
fn place(row: &InstalledModel, spellings: &[PathBuf]) -> Option<PathBuf> {
    // The store always found an audio install at its managed path, whatever
    // its row recorded, and a library re-derives it there too (2.9).
    if row.effective_modality() == ModelModality::Audio {
        return Some(
            Path::new(ModelModality::Audio.as_str()).join(format!("{}.gturbo", row.alias)),
        );
    }
    spellings
        .iter()
        .find_map(|models| row.path.strip_prefix(models).ok())
        .map(Path::to_path_buf)
}

/// Copies into the home at `home` what it lacks of the embedding and OCR
/// installs under `models` and of the catalog override in the moved root
/// `moved` (9.5). The originals stay, and nothing in the home is replaced.
///
/// Never fails: each copy that fails leaves nothing in the home and is
/// reported, for the next import to try again. Nothing is read through a
/// link: an install folder that is a link is reported rather than copied,
/// and a link or special file inside an install fails its copy, as it fails
/// a move. Any other entry that is not a folder named like an alias is not
/// an install and is passed over.
fn home_copies(models: &Path, moved: &Path, home: &Path) -> Done {
    let mut done = Done::default();
    let home_models = Store::new(home).models_root();
    for folder in HOME_ONLY_FOLDERS {
        let from = models.join(folder);
        // Not followed: a linked folder could lead anywhere.
        if !fs::symlink_metadata(&from).is_ok_and(|found| found.file_type().is_dir()) {
            continue;
        }
        let entries = match fs::read_dir(&from) {
            Ok(entries) => entries,
            Err(e) => {
                let reason = format!("reading {}: {e}", from.display());
                done.copy_failures.push((folder.to_string(), reason));
                continue;
            }
        };
        // Each install folder, and whether it is a link. The entry's own
        // type, not followed.
        let mut aliases: Vec<(String, bool)> = entries
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let kind = entry.file_type().ok()?;
                let linked = kind.is_symlink();
                (kind.is_dir() || linked).then_some((entry.file_name(), linked))
            })
            .filter_map(|(name, linked)| Some((name.into_string().ok()?, linked)))
            // A dot name is a staging folder, never an install.
            .filter(|(name, _)| validate_alias(name).is_ok())
            .collect();
        aliases.sort();
        let into = home_models.join(folder);
        for (alias, linked) in aliases {
            let target = into.join(&alias);
            // The home's own install wins and is never replaced.
            if fs::symlink_metadata(&target).is_ok() {
                continue;
            }
            let name = format!("{folder}/{alias}");
            if linked {
                let reason = format!(
                    "{} is a link, which the import never follows",
                    from.join(&alias).display()
                );
                done.copy_failures.push((name, reason));
                continue;
            }
            let copied = fs::create_dir_all(&into)
                .map_err(|e| format!("creating {}: {e}", into.display()))
                .and_then(|()| copy_into_place(&from.join(&alias), &target));
            match copied {
                Ok(()) => done.copied_to_home.push(name),
                Err(reason) => done.copy_failures.push((name, reason)),
            }
        }
    }
    match restore_catalog_override(moved, home) {
        Ok(restored) => done.restored_catalog_override = restored,
        Err(reason) => done
            .copy_failures
            .push((CATALOG_OVERRIDE.to_string(), reason)),
    }
    done
}

/// Copies the install `source` to `target` through a staging folder beside
/// it, so a copy cut short is never found as an install. A failure removes
/// the staging folder.
fn copy_into_place(source: &Path, target: &Path) -> Result<(), String> {
    let name = new_library_id().map_err(|e| format!("naming a staging folder: {e}"))?;
    let staging = target.with_file_name(format!(".import-{name}"));
    let published = copy_folder(source, &staging)
        .map_err(|e| e.to_string())
        .and_then(|()| {
            // Again, right before the rename, which would replace an empty
            // folder created meanwhile.
            if fs::symlink_metadata(target).is_ok() {
                return Err(format!("{} appeared during the import", target.display()));
            }
            fs::rename(&staging, target)
                .map_err(|e| format!("publishing {}: {e}", target.display()))
        });
    if published.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    published
}

/// Copies `<moved>/models.json` to `<home>/models.json` when the home has
/// none, since Move Store took it along. Returns whether it did.
fn restore_catalog_override(moved: &Path, home: &Path) -> Result<bool, String> {
    let target = home.join(CATALOG_OVERRIDE);
    if fs::symlink_metadata(&target).is_ok() {
        return Ok(false);
    }
    let source = moved.join(CATALOG_OVERRIDE);
    // Not followed: only the file Move Store wrote.
    if !fs::symlink_metadata(&source).is_ok_and(|found| found.file_type().is_file()) {
        return Ok(false);
    }
    let bytes = fs::read(&source).map_err(|e| format!("reading {}: {e}", source.display()))?;
    fs::create_dir_all(home).map_err(|e| format!("creating {}: {e}", home.display()))?;
    replace_file(&target, &bytes).map_err(|e| format!("writing {}: {e}", target.display()))?;
    Ok(true)
}

/// Aliases among `rows`, in the library at `root`, of catalog audio
/// installs with no audio receipt, by the test a move applies (2.10). An
/// install whose folder is gone, or is reached through a link, is missing
/// rather than in need of adoption.
fn receipt_less(root: &Path, rows: &[InstalledModel]) -> Result<Vec<String>, LibraryError> {
    let mut catalog = None;
    let mut found = Vec::new();
    for row in rows
        .iter()
        .filter(|row| row.effective_modality() == ModelModality::Audio)
    {
        let Some((dir, true)) = row
            .relative_path
            .as_deref()
            .and_then(|relative| install_in(root, relative))
        else {
            continue;
        };
        let receipt = dir.join(RECEIPT_FILENAME);
        if fs::symlink_metadata(&receipt).is_ok_and(|found| found.file_type().is_file()) {
            continue;
        }
        // Parsed only when a receipt is missing.
        let catalog = match &mut catalog {
            Some(catalog) => catalog,
            None => catalog.insert(AudioCatalog::embedded().map_err(LibraryError::Io)?),
        };
        if catalog.get(&row.alias).is_some() {
            found.push(row.alias.clone());
        }
    }
    Ok(found)
}

/// A new library's name: the moved store's folder, else its volume.
fn default_name(models: &Path, volume: &Path) -> String {
    models
        .parent()
        .and_then(Path::file_name)
        .or_else(|| volume.file_name())
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| STANDARD_LIBRARY_FOLDER.to_string())
}

fn io_error(action: &str, path: &Path, error: io::Error) -> LibraryError {
    LibraryError::Io(format!("{action} {}: {error}", path.display()))
}

#[cfg(test)]
thread_local! {
    /// Runs once inside the next import on this thread, right after its
    /// home copies, so a test can stand in another process writing then.
    static AFTER_COPIES: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn after_next_copies(hook: impl FnOnce() + 'static) {
    AFTER_COPIES.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

#[cfg(test)]
#[path = "import_tests.rs"]
mod tests;
