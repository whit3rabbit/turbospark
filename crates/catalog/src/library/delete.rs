//! Deleting an install where its library is now (3.1, 3.7).
//!
//! A registry row keeps the path its install had when it was recorded. For
//! an install in a library that is the mount point of the time, so later it
//! can name a folder on a different drive mounted there, or, behind a link
//! planted on the drive, a folder outside the library. Nothing here deletes
//! through that path. The library is located now, every folder from its root
//! to the install is examined without following it (`install_in`), and the
//! install is renamed to a tombstone inside the library's own `.staging`
//! folder before its row is forgotten, so a failure to forget it puts the
//! folder back. Only then is the tombstone removed.
//!
//! As for installs and moves (2.8, 2.10), std has no `openat`/`O_NOFOLLOW`
//! descent and this crate forbids unsafe code, so a live local process that
//! swaps a folder for a link between a check and its use is an accepted
//! residual race. A rename never follows its last component, so what such a
//! link can move is the link itself.

use super::transfer::{create_private_dir, install_path};
use super::{new_library_id, Availability, LibraryRegistry, LocationEnv, INTERNAL_LIBRARY_ID};
use crate::store::{external_library, install_in};
use crate::{InstalledEntry, InstalledModel, ModelModality, Store, StoreError};
use std::fs;
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};

/// The file every image install carries. Named in `turbospark_image::install`,
/// which this crate does not depend on.
const IMAGE_MANIFEST: &str = "manifest.json";

/// The most of an image manifest that is read, for the revision it names.
const MANIFEST_LIMIT: u64 = 1 << 20;

impl Store {
    /// Deletes the text install `install_id` (`<library>:text/<alias>`, as
    /// [`InstalledModel::install_id`] spells it): its folder, its registry
    /// row, and in a library its marker entry. A vision tower is a text
    /// install under its own suffix, and deletes only its own folder.
    ///
    /// The install is placed where its library is now, never at the path its
    /// row recorded:
    ///
    /// - A library that is not connected is [`StoreError::NotConnected`],
    ///   one a newer TurboSpark wrote is [`StoreError::LibraryReadOnly`], and
    ///   a root that leads off its drive is refused. Nothing is touched.
    /// - A row may name only its alias's own folder, as a move requires.
    /// - A folder that is not there as a plain folder, because it is gone or
    ///   because a link stands in the way, is not followed or deleted: the
    ///   row is only forgotten. A path that cannot be examined at all is an
    ///   error, since a drive that answers badly must not lose its rows.
    /// - Otherwise the folder is renamed into a tombstone folder in the
    ///   library's `.staging`, the row is forgotten (`Store::forget_install`),
    ///   and the tombstone is removed. A failure after the rename puts the
    ///   folder back, and records the row again if it was already dropped.
    ///
    /// On This Mac the folder is deleted at its recorded path, as it always
    /// was, but only when that path is the alias's own folder in the models
    /// folder, or a custom location that holds nothing the store manages: not
    /// the home or the models folder, another install, or a library. A row
    /// whose folder is gone is forgotten.
    ///
    /// `env` locates libraries, as for [`Store::forget_install`], which this
    /// runs. Audio installs have their own receipt-checked delete.
    pub fn delete_install(&self, install_id: &str, env: &LocationEnv) -> Result<(), StoreError> {
        let entries = self.installed_with_availability(env);
        let Some(entry) = entries.iter().find(|entry| entry.install_id == install_id) else {
            return Err(unknown(install_id));
        };
        match entry.row.effective_modality() {
            ModelModality::Text => {}
            ModelModality::Audio => {
                return Err(StoreError::Io(format!(
                    "{install_id:?} is an audio model, which is deleted through the audio \
                     catalog, where its receipt is checked"
                )))
            }
            ModelModality::Image => {
                return Err(StoreError::Io(format!(
                    "{install_id:?} is an image install, which is deleted as an image"
                )))
            }
        }
        match external_library(&entry.row) {
            Some(library) => self.delete_in_library(entry, library, env),
            None => self.delete_on_this_mac(&entries, entry, env),
        }
    }

    /// Deletes the image install `install_id` (`<library>:image/<alias>`) of
    /// a library: `<root>/image/<alias>.gturbo`, and the MLX source folder the
    /// installer keeps beside it, `.<alias>.<revision>.image-source`, which
    /// This Mac's delete removes too. Image installs are not recorded, so
    /// there is no row to forget.
    ///
    /// Placed and removed as [`Store::delete_install`] does. A folder that is
    /// not there, is reached through a link, or has no plain manifest is not
    /// an image install ([`StoreError::UnknownInstall`]). The kept source is
    /// found by the revision the manifest names, which a drive supplies, so it
    /// must be one plain folder of the image folder to go. This Mac's image
    /// installs are listed by their manifests and deleted by path or alias,
    /// and an id of This Mac is refused here.
    pub fn delete_image_install(
        &self,
        install_id: &str,
        env: &LocationEnv,
    ) -> Result<(), StoreError> {
        let (library, rest) = install_id
            .split_once(':')
            .ok_or_else(|| unknown(install_id))?;
        let alias = rest
            .strip_prefix("image/")
            .ok_or_else(|| unknown(install_id))?;
        if library == INTERNAL_LIBRARY_ID {
            return Err(StoreError::Io(
                "the image installs of This Mac are deleted by path or alias, not by library"
                    .to_string(),
            ));
        }
        let (root, _) = self.library_for_install(Some(library), env)?;
        let own = relative_to(&root, &Store::image_install_path_in(&root, alias)?)?;
        if !matches!(install_in(&root, &own), Some((_, true))) {
            return Err(unexamined(&root, &own).unwrap_or_else(|| unknown(install_id)));
        }
        let Some(manifest) = read_manifest(&root.join(&own)) else {
            return Err(unknown(install_id));
        };
        let mut folders = vec![own];
        folders.extend(kept_source(&root, alias, &manifest));
        delete_through_tombstone(self, &root, &folders, || Ok(()), || {})
    }

    fn delete_in_library(
        &self,
        entry: &InstalledEntry,
        library: &str,
        env: &LocationEnv,
    ) -> Result<(), StoreError> {
        if let Availability::NotConnected { library } = &entry.availability {
            return Err(StoreError::NotConnected {
                library: library.clone(),
            });
        }
        // Located now, whatever the row recorded. Also refuses a root that
        // leads off its drive, which listing calls missing, and a library
        // that is read-only or no longer listed.
        let (root, _) = self.library_for_install(Some(library), env)?;
        let row = &entry.row;
        let own = install_path(&root, ModelModality::Text, row.kind.as_deref(), &row.alias)
            .map_err(|error| StoreError::Io(error.to_string()))?;
        let own = relative_to(&root, &own)?;
        let recorded = row
            .relative_path
            .clone()
            .or_else(|| row.path.strip_prefix(&root).ok().map(Path::to_path_buf));
        if recorded.as_deref() != Some(own.as_path()) {
            return Err(not_own(&row.path, &row.alias));
        }
        if !matches!(install_in(&root, &own), Some((_, true))) {
            // Gone, or behind a link, which is never followed: only the row
            // goes, and no file is touched.
            if let Some(error) = unexamined(&root, &own) {
                return Err(error);
            }
            return self.forget_install(&entry.install_id, env);
        }
        let restored = InstalledModel {
            path: root.join(&own),
            ..row.clone()
        };
        delete_through_tombstone(
            self,
            &root,
            &[own],
            || self.forget_install(&entry.install_id, env),
            || {
                let _ = self.record_in_library(std::slice::from_ref(&restored), env);
            },
        )
    }

    fn delete_on_this_mac(
        &self,
        entries: &[InstalledEntry],
        entry: &InstalledEntry,
        env: &LocationEnv,
    ) -> Result<(), StoreError> {
        match &entry.availability {
            Availability::NotConnected { library } => Err(StoreError::NotConnected {
                library: library.clone(),
            }),
            Availability::Missing => self.forget_install(&entry.install_id, env),
            Availability::Available => {
                self.check_own_folder(entries, entry, env)?;
                let path = &entry.row.path;
                // A link at the path is removed itself, never followed.
                fs::remove_dir_all(path).map_err(|error| {
                    StoreError::Io(format!("failed to remove {}: {error}", path.display()))
                })?;
                self.forget_install(&entry.install_id, env)
            }
        }
    }

    /// Whether the recorded path of a This Mac row may be deleted. A hand
    /// edit of `installed.json`, or a bug elsewhere, must not turn a delete
    /// into the removal of the home, the models folder, another install or a
    /// library.
    fn check_own_folder(
        &self,
        entries: &[InstalledEntry],
        entry: &InstalledEntry,
        env: &LocationEnv,
    ) -> Result<(), StoreError> {
        let row = &entry.row;
        let path = &row.path;
        let refuse =
            |why: &str| StoreError::Io(format!("{} {why}, so it is not deleted", path.display()));
        if !path.is_absolute()
            || path
                .components()
                .any(|part| matches!(part, Component::ParentDir | Component::CurDir))
        {
            return Err(refuse("is not a plain absolute path"));
        }
        let models = self.models_root();
        // In the models folder, only the alias's own folder, or the flat name
        // from before modality folders, as for a move (2.10).
        if let Ok(recorded) = path.strip_prefix(&models) {
            let own = install_path(
                &models,
                ModelModality::Text,
                row.kind.as_deref(),
                &row.alias,
            )
            .map_err(|error| StoreError::Io(error.to_string()))?;
            let own = relative_to(&models, &own)?;
            let flat = own
                .file_name()
                .is_some_and(|name| recorded == Path::new(name));
            if recorded != own && !flat {
                return Err(not_own(path, &row.alias));
            }
        } else if let Some(record) = LibraryRegistry::load(self.root())
            .libraries()
            .into_iter()
            .filter(|record| !record.is_internal())
            .find(|record| {
                covers(
                    &record.last_mount_path.join(&record.volume_relative_path),
                    path,
                )
            })
        {
            // A row with no library that sits where a listed library is, or
            // was, mounted is one an older writer stripped of its library.
            // The drive there may be another drive now, mounted under the
            // same name, so it is not deleted by the path it recorded.
            return Err(StoreError::Io(format!(
                "{} is inside the library {:?}, so it is not deleted here; \
                 remove it through the library",
                path.display(),
                record.name
            )));
        }
        // And never a folder above anything else the store manages.
        let mut protected = vec![self.root().to_path_buf(), models];
        protected.extend(
            entries
                .iter()
                .filter(|other| other.install_id != entry.install_id)
                .map(|other| other.row.path.clone()),
        );
        protected.extend(
            LibraryRegistry::load(self.root())
                .libraries()
                .into_iter()
                .filter(|record| !record.is_internal())
                .map(|record| record.last_mount_path.join(&record.volume_relative_path)),
        );
        protected.extend(
            self.connected_libraries(env)
                .into_iter()
                .map(|(_, root)| root),
        );
        if protected.iter().any(|inner| covers(path, inner)) {
            return Err(refuse("holds other installs, a library or the model store"));
        }
        Ok(())
    }
}

/// Removes the folders `folders` of the library at the canonical `root`
/// (relative to it, each already found to be a plain folder) through
/// tombstones in the library's own `.staging`, where a rename stays on the
/// drive. `forget` runs once they are all aside, and only after it succeeds
/// are they removed; a failure of either puts them back first, and
/// `restore` records the row again, since `forget` may already have dropped
/// it. Every folder is walked again, unfollowed, right before its rename: a
/// slow drive takes a while to read, and anyone with the drive could have
/// swapped a link in meanwhile.
fn delete_through_tombstone(
    store: &Store,
    root: &Path,
    folders: &[PathBuf],
    forget: impl FnOnce() -> Result<(), StoreError>,
    restore: impl FnOnce(),
) -> Result<(), StoreError> {
    let io = |action: &str, path: &Path, error: io::Error| {
        StoreError::Io(format!("{action} {}: {error}", path.display()))
    };
    let staging = store.staging_parent_for(root)?;
    let name = new_library_id()
        .map_err(|error| StoreError::Io(format!("naming the deletion folder: {error}")))?;
    let tombstones = staging.join(format!("delete-{name}"));
    create_private_dir(&tombstones).map_err(|error| io("creating", &tombstones, error))?;
    // Where each folder was, and where it is now.
    let mut moved: Vec<(PathBuf, PathBuf)> = Vec::new();
    for (index, relative) in folders.iter().enumerate() {
        let tombstone = tombstones.join(index.to_string());
        let failure = match install_in(root, relative) {
            Some((path, true)) => match fs::rename(&path, &tombstone) {
                Ok(()) => {
                    moved.push((path, tombstone));
                    continue;
                }
                Err(error) => io("moving aside", &path, error),
            },
            _ => StoreError::Io(format!(
                "{} changed while it was being deleted, so it is not deleted",
                root.join(relative).display()
            )),
        };
        return Err(put_back(&moved, &tombstones, failure));
    }
    if let Err(error) = forget() {
        let error = put_back(&moved, &tombstones, error);
        restore();
        return Err(error);
    }
    if let Err(error) = fs::remove_dir_all(&tombstones) {
        let first = moved.first().map_or(tombstones.as_path(), |(path, _)| path);
        let error = put_back(&moved, &tombstones, io("removing", first, error));
        restore();
        return Err(error);
    }
    Ok(())
}

/// Renames what was moved aside back to where it was, newest first, removes
/// the tombstone folder, and returns `error`, naming anything that could not
/// be put back and is left in the tombstone folder.
fn put_back(moved: &[(PathBuf, PathBuf)], tombstones: &Path, error: StoreError) -> StoreError {
    let stuck: Vec<String> = moved
        .iter()
        .rev()
        .filter_map(|(path, tombstone)| {
            fs::rename(tombstone, path).err().map(|reason| {
                format!(
                    "{} could not be put back from {}: {reason}",
                    path.display(),
                    tombstone.display()
                )
            })
        })
        .collect();
    if stuck.is_empty() {
        // Only an empty folder is removed, so nothing is lost by trying.
        let _ = fs::remove_dir(tombstones);
        return error;
    }
    StoreError::Io(format!("{error}; {}", stuck.join("; ")))
}

fn unknown(install_id: &str) -> StoreError {
    StoreError::UnknownInstall {
        install_id: install_id.to_string(),
    }
}

fn not_own(path: &Path, alias: &str) -> StoreError {
    StoreError::Io(format!(
        "{} is not the install folder of {alias:?}, so it is not deleted",
        path.display()
    ))
}

fn relative_to(root: &Path, path: &Path) -> Result<PathBuf, StoreError> {
    path.strip_prefix(root)
        .map(Path::to_path_buf)
        .map_err(|_| StoreError::Io(format!("{} is not in its library", path.display())))
}

/// An error for a path `install_in` found not to be a plain folder, when the
/// reason is that something on the way could not be examined, as a drive
/// answering with an IO error does while it goes away. A folder that is
/// absent, or is a link or a file, is not one: those are answers.
fn unexamined(root: &Path, relative: &Path) -> Option<StoreError> {
    let mut at = root.to_path_buf();
    for part in relative.components() {
        at.push(part);
        match fs::symlink_metadata(&at) {
            Ok(found) if found.file_type().is_dir() => {}
            Ok(_) => return None,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return None,
            Err(error) => {
                return Some(StoreError::Io(format!(
                    "cannot examine {}: {error}",
                    at.display()
                )))
            }
        }
    }
    None
}

/// Whether `outer` is `inner` or a folder above it, by spelling or, when
/// both exist, by where they lead: a link or a different spelling of the home
/// must not hide a folder from the check.
fn covers(outer: &Path, inner: &Path) -> bool {
    if inner.starts_with(outer) {
        return true;
    }
    match (outer.canonicalize(), inner.canonicalize()) {
        (Ok(outer), Ok(inner)) => inner.starts_with(outer),
        _ => false,
    }
}

/// The manifest of the image install `install`, when it is a plain file of
/// a sane size that parses. The listing reads an install no other way.
fn read_manifest(install: &Path) -> Option<serde_json::Value> {
    let path = install.join(IMAGE_MANIFEST);
    if !fs::symlink_metadata(&path).ok()?.file_type().is_file() {
        return None;
    }
    let mut bytes = Vec::new();
    fs::File::open(&path)
        .ok()?
        .take(MANIFEST_LIMIT + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() as u64 > MANIFEST_LIMIT {
        return None;
    }
    serde_json::from_slice(&bytes).ok()
}

/// The MLX source folder kept beside the image install `alias`, relative to
/// the library root, when it is there as a plain folder. Its name is built
/// from the revision in the manifest, which a drive supplies, so a revision
/// that is not one plain path component names nothing.
fn kept_source(root: &Path, alias: &str, manifest: &serde_json::Value) -> Option<PathBuf> {
    let revision = manifest.get("source")?.get("model_revision")?.as_str()?;
    let name = format!(".{alias}.{revision}.image-source");
    let one_component = matches!(
        Path::new(&name).components().collect::<Vec<_>>().as_slice(),
        [Component::Normal(_)]
    );
    if !one_component || name.contains('\0') {
        return None;
    }
    let relative = Path::new("image").join(name);
    matches!(install_in(root, &relative), Some((_, true))).then_some(relative)
}

#[cfg(test)]
#[path = "delete_tests.rs"]
mod tests;
