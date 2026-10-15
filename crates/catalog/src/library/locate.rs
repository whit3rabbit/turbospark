//! Where a library sits on its drive.
//!
//! A library is recorded as a volume root plus a path relative to it, so a
//! drive that is renamed or remounted elsewhere is searched at the same
//! relative place. Everything here works from paths, `st_dev`, and the
//! injected mount table, so tests drive it with a temp folder of fake
//! volumes.

use super::mount_table::mount_entry_at;
use super::{LibraryMarker, LibraryRecord, LocationEnv, MountEntry};
use std::cell::OnceCell;
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};

/// The folder a library is created in at the top of a drive, and the only
/// place discovery looks on a drive with no known library (4.5).
pub const STANDARD_LIBRARY_FOLDER: &str = "TurboSpark";

/// Where a library is right now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Located {
    /// `duplicate` is set when the library had moved and more than one local
    /// volume carries its marker id, as after a drive is cloned.
    Available {
        root: PathBuf,
        duplicate: bool,
    },
    NotConnected,
}

/// Finds `record`'s library root by its marker id. Read-only.
///
/// The recorded root is checked first, and a match there ends the search
/// without touching another volume or the mount table. Otherwise the same
/// relative path is checked on each locally mounted volume, so a renamed or
/// remounted drive is still found (4.2). Nothing else on a drive is read
/// (4.5). A duplicate is therefore only reported while relocating, when two
/// or more volumes carry the id. A marker that is absent, unreadable, or
/// carries another id is simply not this library, because a drive anyone
/// could have written must not turn a refresh into an error (3.4).
pub fn locate(record: &LibraryRecord, env: &LocationEnv) -> Located {
    locate_with(record, &env.volumes_root, || env.mount_table())
}

/// The mount table of one operation over many libraries, read the first
/// time a library needs it and reused after, so locating N libraries that
/// are offline or have moved spawns `/sbin/mount` once rather than N times.
/// A failed read is reused too, and every library then skips the scan, as
/// [`locate`] does alone.
pub(super) struct SharedMountTable<'a> {
    env: &'a LocationEnv,
    table: OnceCell<Result<Vec<MountEntry>, String>>,
}

impl<'a> SharedMountTable<'a> {
    pub(super) fn new(env: &'a LocationEnv) -> Self {
        Self {
            env,
            table: OnceCell::new(),
        }
    }

    /// [`locate`], reading the table at most once across every call.
    pub(super) fn locate(&self, record: &LibraryRecord) -> Located {
        locate_with(record, &self.env.volumes_root, || {
            self.table.get_or_init(|| self.env.mount_table()).clone()
        })
    }
}

/// [`locate`] with the mount table read through `read_table`, which runs at
/// most once and only when the recorded root does not match.
fn locate_with(
    record: &LibraryRecord,
    volumes_root: &Path,
    read_table: impl FnOnce() -> Result<Vec<MountEntry>, String>,
) -> Located {
    let recorded = record.last_mount_path.join(&record.volume_relative_path);
    // This Mac has no marker and never moves. A library still where it was
    // needs no other volume.
    if record.is_internal() || carries_id(&recorded, &record.id) {
        return Located::Available {
            root: recorded,
            duplicate: false,
        };
    }
    let volumes = volumes(volumes_root);
    if volumes.is_empty() {
        return Located::NotConnected;
    }
    // A network share can hang a file read for minutes, past the 5 s bound
    // (4.1, 4.6), and reading one can raise a privacy prompt for a share
    // the user never added. With no table, no volume is known to be local.
    let Ok(table) = read_table() else {
        return Located::NotConnected;
    };
    let mut found = None;
    let mut copies: Vec<PathBuf> = Vec::new();
    for volume in volumes {
        if !is_local_mount(&volume, &table) {
            continue;
        }
        let candidate = volume.join(&record.volume_relative_path);
        if !carries_id(&candidate, &record.id) {
            continue;
        }
        // Two volumes can reach one folder through a symlink, which is one
        // library rather than a copy.
        let identity = candidate
            .canonicalize()
            .unwrap_or_else(|_| candidate.clone());
        if !copies.contains(&identity) {
            copies.push(identity);
        }
        found.get_or_insert(candidate);
    }
    match found {
        Some(root) => Located::Available {
            root,
            duplicate: copies.len() > 1,
        },
        None => Located::NotConnected,
    }
}

/// Whether `volume` has its own local mount, looked up as
/// `validate_location` does. A folder with no entry of its own, such as a
/// mount point left behind by an unclean eject, sits on the boot disk, and
/// the `/` entry must not vouch for it.
fn is_local_mount(volume: &Path, table: &[MountEntry]) -> bool {
    mount_entry_at(table, volume).is_some_and(|entry| entry.local)
}

/// The marker of the library in the standard folder at the top of the
/// volume at `volume_root`, if any. Reads only
/// `<volume_root>/TurboSpark/library.json`, so connecting a drive never
/// walks its contents (4.5). Like [`locate`], an unreadable marker is no
/// library rather than an error.
pub fn discover_at(volume_root: &Path) -> Option<LibraryMarker> {
    let folder = volume_root.join(STANDARD_LIBRARY_FOLDER);
    // Not followed: a crafted drive could point the folder off the drive,
    // even onto a network share the user never added.
    if !std::fs::symlink_metadata(&folder).is_ok_and(|metadata| metadata.is_dir()) {
        return None;
    }
    LibraryMarker::read(&folder).ok().flatten()
}

/// The path of the library root `root` below its volume root, which a
/// record keeps as `volume_relative_path`. `root` must exist and be
/// canonical, as for [`volume_root_of`].
pub fn volume_relative_path(root: &Path, env: &LocationEnv) -> std::io::Result<PathBuf> {
    let volume = volume_root_of(root, env)?;
    root.strip_prefix(&volume)
        .map(Path::to_path_buf)
        .map_err(|_| std::io::Error::other(format!("{} is not below its volume", root.display())))
}

fn carries_id(root: &Path, id: &str) -> bool {
    matches!(LibraryMarker::read(root), Ok(Some(marker)) if marker.id == id)
}

/// The volume folders under `volumes_root`, sorted by name so a duplicate
/// resolves to the same copy on every refresh.
fn volumes(volumes_root: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(volumes_root) else {
        return Vec::new();
    };
    let mut volumes: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        // Symlinks are skipped: `/Volumes/Macintosh HD` links to `/`, and
        // the internal disk never holds an external library.
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .map(|entry| entry.path())
        .collect();
    volumes.sort();
    volumes
}

/// The volume root of `path`, which must exist and be canonical.
///
/// A direct child of `env.volumes_root` is a volume root, which makes fake
/// volumes under a temp folder behave like `/Volumes/<name>`. Otherwise the
/// root is the highest ancestor still on the path's device. On APFS every
/// internal path, including the firmlinked Data volume, reports the device
/// of `/`, so an internal folder resolves to `/`.
pub fn volume_root_of(path: &Path, env: &LocationEnv) -> std::io::Result<PathBuf> {
    if let Ok(rest) = path.strip_prefix(&env.volumes_root) {
        if let Some(Component::Normal(volume)) = rest.components().next() {
            return Ok(env.volumes_root.join(volume));
        }
    }
    let device = std::fs::metadata(path)?.dev();
    let mut root = path;
    for ancestor in path.ancestors().skip(1) {
        if std::fs::metadata(ancestor)?.dev() != device {
            break;
        }
        root = ancestor;
    }
    Ok(root.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::library::MountSource;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ts-library-locate-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // The temp dir is behind the /var symlink, and the volume-root rules
        // compare canonical paths.
        dir.canonicalize().unwrap()
    }

    fn env_with_volumes_root(volumes_root: PathBuf) -> LocationEnv {
        LocationEnv {
            mount_source: MountSource::Fixed(Vec::new()),
            volumes_root,
            home_device: 0,
        }
    }

    #[test]
    fn a_direct_child_of_the_volumes_root_is_the_volume_root() {
        let base = scratch("direct-child");
        let volumes_root = base.join("Volumes");
        let library = volumes_root.join("Embedding Gemma Transcribe/TurboSpark/models");
        std::fs::create_dir_all(&library).unwrap();
        let env = env_with_volumes_root(volumes_root.clone());
        let volume = volumes_root.join("Embedding Gemma Transcribe");

        // Fake volumes share the temp dir's device, so only the direct-child
        // rule can stop the walk at the fake volume rather than at `/`.
        assert_eq!(volume_root_of(&library, &env).unwrap(), volume);
        assert_eq!(volume_root_of(&volume, &env).unwrap(), volume);

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn outside_the_volumes_root_the_highest_ancestor_on_the_same_device_is_the_root() {
        let base = scratch("same-device");
        let folder = base.join("Models");
        std::fs::create_dir_all(&folder).unwrap();
        // A volumes root that does not contain the folder, so the device walk
        // decides. On macOS every internal path, including the firmlinked
        // Data volume, reports the same device as `/`.
        let env = env_with_volumes_root(base.join("Volumes"));

        assert_eq!(volume_root_of(&folder, &env).unwrap(), PathBuf::from("/"));

        let _ = std::fs::remove_dir_all(&base);
    }

    const ID: &str = "0f6c2a3e-1b7d-4c55-9e0a-7d2f1c3b4a59";
    const OTHER_ID: &str = "7a1e9c04-52b3-4f8d-a6e1-3c9b0d2f5e17";

    /// A library folder at `root` carrying a marker with `id`.
    fn library_at(root: &Path, id: &str) {
        std::fs::create_dir_all(root).unwrap();
        LibraryMarker::new(id.to_string(), "T7".to_string())
            .write(root)
            .unwrap();
    }

    /// An env whose fixed mount table has the local `/` that every live
    /// table has, plus the named volumes under `volumes_root`.
    fn env_with_mounts(volumes_root: &Path, local: &[&str], network: &[&str]) -> LocationEnv {
        let mut entries = vec![MountEntry {
            mount_point: PathBuf::from("/"),
            fs_type: "apfs".to_string(),
            local: true,
            read_only: true,
        }];
        for (names, is_local, fs_type) in [(local, true, "apfs"), (network, false, "smbfs")] {
            entries.extend(names.iter().map(|name| MountEntry {
                mount_point: volumes_root.join(name),
                fs_type: fs_type.to_string(),
                local: is_local,
                read_only: false,
            }));
        }
        LocationEnv {
            mount_source: MountSource::Fixed(entries),
            volumes_root: volumes_root.to_path_buf(),
            home_device: 0,
        }
    }

    /// `env`'s table, counting each read.
    fn counted<'a>(
        env: &'a LocationEnv,
        reads: &'a std::cell::Cell<usize>,
    ) -> impl FnOnce() -> Result<Vec<MountEntry>, String> + 'a {
        move || {
            reads.set(reads.get() + 1);
            env.mount_table()
        }
    }

    /// A record of the library `id`, last found at the standard folder on
    /// the volume `last_mount_path`.
    fn record(id: &str, last_mount_path: &Path) -> LibraryRecord {
        LibraryRecord {
            id: id.to_string(),
            name: "T7".to_string(),
            volume_relative_path: PathBuf::from(STANDARD_LIBRARY_FOLDER),
            last_mount_path: last_mount_path.to_path_buf(),
            default_for_downloads: false,
            added_on: "2026-10-14".to_string(),
            last_seen: None,
        }
    }

    #[test]
    fn locate_finds_the_library_at_its_recorded_mount_path() {
        let base = scratch("recorded");
        let volumes_root = base.join("Volumes");
        let volume = volumes_root.join("T7");
        let root = volume.join(STANDARD_LIBRARY_FOLDER);
        library_at(&root, ID);
        // A local clone that a scan would report as a duplicate.
        library_at(
            &volumes_root.join("Clone").join(STANDARD_LIBRARY_FOLDER),
            ID,
        );
        let env = env_with_mounts(&volumes_root, &["T7", "Clone"], &[]);

        // Found where it was, the library touches no other volume and no
        // mount table, so a dead network mount cannot stall a refresh.
        let reads = std::cell::Cell::new(0);
        assert_eq!(
            locate_with(&record(ID, &volume), &volumes_root, counted(&env, &reads)),
            Located::Available {
                root: root.clone(),
                duplicate: false
            }
        );
        assert_eq!(reads.get(), 0);
        assert_eq!(
            locate(&record(ID, &volume), &env),
            Located::Available {
                root: root.clone(),
                duplicate: false
            }
        );

        // A recorded mount path that names the volume through a symlink is
        // returned in the recorded spelling.
        let alias = base.join("alias");
        std::os::unix::fs::symlink(&volume, &alias).unwrap();
        assert_eq!(
            locate(&record(ID, &alias), &env),
            Located::Available {
                root: alias.join(STANDARD_LIBRARY_FOLDER),
                duplicate: false
            }
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn locate_finds_the_library_after_its_volume_is_renamed() {
        let base = scratch("renamed");
        let volumes_root = base.join("Volumes");
        let volume = volumes_root.join("T7");
        library_at(&volume.join(STANDARD_LIBRARY_FOLDER), ID);
        let env = env_with_mounts(&volumes_root, &["T7 1"], &[]);
        let recorded = record(ID, &volume);

        // macOS mounts a second drive with the same name as "T7 1".
        let renamed = volumes_root.join("T7 1");
        std::fs::rename(&volume, &renamed).unwrap();

        assert_eq!(
            locate(&recorded, &env),
            Located::Available {
                root: renamed.join(STANDARD_LIBRARY_FOLDER),
                duplicate: false
            }
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_cloned_marker_reports_a_duplicate_while_relocating() {
        let base = scratch("cloned");
        let volumes_root = base.join("Volumes");
        let first = volumes_root.join("A");
        let second = volumes_root.join("B");
        // APFS lists a folder in name-hash order, which puts G before A, so
        // this clone pins the sort by name.
        let listed_first = volumes_root.join("G");
        for volume in [&first, &second, &listed_first] {
            library_at(&volume.join(STANDARD_LIBRARY_FOLDER), ID);
        }
        let env = env_with_mounts(&volumes_root, &["A", "B", "G"], &[]);

        // With the recorded mount gone, the first volume by name wins, so
        // repeated refreshes agree, and the table is read once.
        let reads = std::cell::Cell::new(0);
        assert_eq!(
            locate_with(
                &record(ID, &volumes_root.join("Gone")),
                &volumes_root,
                counted(&env, &reads)
            ),
            Located::Available {
                root: first.join(STANDARD_LIBRARY_FOLDER),
                duplicate: true
            }
        );
        assert_eq!(reads.get(), 1);

        // Found at its recorded mount, the library is not checked for clones.
        assert_eq!(
            locate(&record(ID, &second), &env),
            Located::Available {
                root: second.join(STANDARD_LIBRARY_FOLDER),
                duplicate: false
            }
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn one_folder_reached_from_two_volumes_is_not_a_duplicate() {
        let base = scratch("one-folder");
        let volumes_root = base.join("Volumes");
        let root = volumes_root.join("A").join(STANDARD_LIBRARY_FOLDER);
        library_at(&root, ID);
        std::fs::create_dir_all(volumes_root.join("B")).unwrap();
        std::os::unix::fs::symlink(&root, volumes_root.join("B").join(STANDARD_LIBRARY_FOLDER))
            .unwrap();
        let env = env_with_mounts(&volumes_root, &["A", "B"], &[]);

        assert_eq!(
            locate(&record(ID, &volumes_root.join("Gone")), &env),
            Located::Available {
                root,
                duplicate: false
            }
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn only_volumes_with_their_own_local_mount_entry_are_scanned() {
        let base = scratch("local-only");
        let volumes_root = base.join("Volumes");
        // A network share, and a folder with no mount of its own, such as a
        // mount point left behind by an unclean eject. Both carry the
        // library's marker, and the local `/` entry covers the second by
        // prefix but not exactly.
        library_at(
            &volumes_root.join("Share").join(STANDARD_LIBRARY_FOLDER),
            ID,
        );
        library_at(
            &volumes_root.join("Stale").join(STANDARD_LIBRARY_FOLDER),
            ID,
        );
        let env = env_with_mounts(&volumes_root, &[], &["Share"]);
        let gone = record(ID, &volumes_root.join("Gone"));

        assert_eq!(locate(&gone, &env), Located::NotConnected);

        // A local volume is scanned, and the skipped ones do not count as
        // copies of it.
        let local = volumes_root.join("T7").join(STANDARD_LIBRARY_FOLDER);
        library_at(&local, ID);
        let env = env_with_mounts(&volumes_root, &["T7"], &["Share"]);
        assert_eq!(
            locate(&gone, &env),
            Located::Available {
                root: local,
                duplicate: false
            }
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn an_unreadable_mount_table_means_no_scan() {
        let base = scratch("no-table");
        let volumes_root = base.join("Volumes");
        let volume = volumes_root.join("T7");
        let root = volume.join(STANDARD_LIBRARY_FOLDER);
        library_at(&root, ID);
        let unreadable = || Err("running /sbin/mount: refused".to_string());

        assert_eq!(
            locate_with(
                &record(ID, &volumes_root.join("Gone")),
                &volumes_root,
                unreadable
            ),
            Located::NotConnected
        );
        // The recorded root needs no table.
        assert_eq!(
            locate_with(&record(ID, &volume), &volumes_root, unreadable),
            Located::Available {
                root,
                duplicate: false
            }
        );

        // With no volume folder to scan, the table is not read at all.
        let reads = std::cell::Cell::new(0);
        let env = env_with_mounts(&base.join("NoVolumes"), &[], &[]);
        assert_eq!(
            locate_with(
                &record(ID, &volume.join("Gone")),
                &env.volumes_root,
                counted(&env, &reads)
            ),
            Located::NotConnected
        );
        assert_eq!(reads.get(), 0);

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn locate_matches_only_the_marker_id_at_the_recorded_relative_path() {
        let base = scratch("mismatch");
        let volumes_root = base.join("Volumes");
        let volume = volumes_root.join("T7");
        // Another library at the recorded place, and this library's marker
        // at a place the record does not name.
        library_at(&volume.join(STANDARD_LIBRARY_FOLDER), OTHER_ID);
        library_at(&volumes_root.join("T8/Elsewhere"), ID);
        let env = env_with_mounts(&volumes_root, &["T7", "T8"], &[]);

        assert_eq!(locate(&record(ID, &volume), &env), Located::NotConnected);

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn relocating_looks_at_the_recorded_relative_path_on_each_volume() {
        let base = scratch("relative-scan");
        let volumes_root = base.join("Volumes");
        // The same id at the standard folder, which this record does not
        // name, so it is a decoy rather than a copy.
        library_at(&volumes_root.join("A").join(STANDARD_LIBRARY_FOLDER), ID);
        let root = volumes_root.join("B/Stuff").join(STANDARD_LIBRARY_FOLDER);
        library_at(&root, ID);
        let env = env_with_mounts(&volumes_root, &["A", "B"], &[]);
        let nested = LibraryRecord {
            volume_relative_path: PathBuf::from("Stuff/TurboSpark"),
            ..record(ID, &volumes_root.join("Gone"))
        };

        assert_eq!(
            locate(&nested, &env),
            Located::Available {
                root,
                duplicate: false
            }
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_later_mount_on_the_same_point_decides_whether_it_is_local() {
        let base = scratch("remounted");
        let volumes_root = base.join("Volumes");
        let root = volumes_root.join("T7").join(STANDARD_LIBRARY_FOLDER);
        library_at(&root, ID);
        let gone = record(ID, &volumes_root.join("Gone"));
        let mounted = |fs_type: &str, local: bool| MountEntry {
            mount_point: volumes_root.join("T7"),
            fs_type: fs_type.to_string(),
            local,
            read_only: false,
        };
        let env_with = |entries: Vec<MountEntry>| LocationEnv {
            mount_source: MountSource::Fixed(entries),
            volumes_root: volumes_root.clone(),
            home_device: 0,
        };

        // A share mounted over the local volume hides it.
        let shadowed = env_with(vec![mounted("apfs", true), mounted("smbfs", false)]);
        assert_eq!(locate(&gone, &shadowed), Located::NotConnected);

        // The reverse order leaves a local volume on top.
        let local_on_top = env_with(vec![mounted("smbfs", false), mounted("apfs", true)]);
        assert_eq!(
            locate(&gone, &local_on_top),
            Located::Available {
                root,
                duplicate: false
            }
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_symlinked_volume_entry_is_not_scanned() {
        let base = scratch("symlinked-volume");
        let volumes_root = base.join("Volumes");
        std::fs::create_dir_all(&volumes_root).unwrap();
        // Like `/Volumes/Macintosh HD -> /`: the link's target holds the
        // library, but only real volume folders are searched.
        let target = base.join("boot");
        library_at(&target.join(STANDARD_LIBRARY_FOLDER), ID);
        std::os::unix::fs::symlink(&target, volumes_root.join("Macintosh HD")).unwrap();
        let env = env_with_mounts(&volumes_root, &["Macintosh HD"], &[]);

        assert_eq!(
            locate(&record(ID, &volumes_root.join("Gone")), &env),
            Located::NotConnected
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn unreadable_markers_are_silently_not_a_match() {
        let base = scratch("unreadable");
        let volumes_root = base.join("Volumes");
        let recorded = volumes_root.join("A");
        let recorded_root = recorded.join(STANDARD_LIBRARY_FOLDER);
        std::fs::create_dir_all(&recorded_root).unwrap();
        std::fs::write(recorded_root.join(crate::library::MARKER_FILE), "not json").unwrap();
        // A folder where the marker file should be.
        std::fs::create_dir_all(
            volumes_root
                .join("B")
                .join(STANDARD_LIBRARY_FOLDER)
                .join(crate::library::MARKER_FILE),
        )
        .unwrap();
        let good = volumes_root.join("C").join(STANDARD_LIBRARY_FOLDER);
        library_at(&good, ID);
        let env = env_with_mounts(&volumes_root, &["A", "B", "C"], &[]);

        assert_eq!(
            locate(&record(ID, &recorded), &env),
            Located::Available {
                root: good.clone(),
                duplicate: false
            }
        );

        std::fs::remove_dir_all(&good).unwrap();
        assert_eq!(locate(&record(ID, &recorded), &env), Located::NotConnected);

        // No volumes folder at all is not connected too, not an error.
        let missing = env_with_mounts(&base.join("NoVolumes"), &[], &[]);
        assert_eq!(
            locate(&record(ID, &recorded), &missing),
            Located::NotConnected
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn this_mac_is_available_at_its_home_root_without_a_marker() {
        let base = scratch("internal");
        let home = base.join("home");
        let env = env_with_volumes_root(base.join("Volumes"));

        // No home, no models folder, and no marker: This Mac is not located
        // by marker, so it is available at today's layout regardless.
        assert_eq!(
            locate(&LibraryRecord::internal(&home), &env),
            Located::Available {
                root: home.join("models"),
                duplicate: false
            }
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn discovery_reads_only_the_standard_folder_at_the_top_of_a_volume() {
        let base = scratch("discover");
        let volume = base.join("Volumes/T7");
        // Decoys: a marker at the volume top, in another folder, and nested
        // inside the standard folder.
        library_at(&volume, OTHER_ID);
        library_at(&volume.join("Other"), OTHER_ID);
        library_at(&volume.join(STANDARD_LIBRARY_FOLDER).join("sub"), OTHER_ID);

        // Only decoys: nothing is discovered.
        assert_eq!(discover_at(&volume), None);

        library_at(&volume.join(STANDARD_LIBRARY_FOLDER), ID);
        let found = discover_at(&volume).expect("the standard marker");
        assert_eq!(found.id, ID);

        // A malformed standard marker and a missing volume discover nothing.
        std::fs::write(
            volume
                .join(STANDARD_LIBRARY_FOLDER)
                .join(crate::library::MARKER_FILE),
            "{",
        )
        .unwrap();
        assert_eq!(discover_at(&volume), None);
        assert_eq!(discover_at(&base.join("Volumes/Gone")), None);

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn discovery_does_not_follow_a_symlinked_standard_folder() {
        let base = scratch("discover-symlink");
        // A valid library off the drive, as a share or another disk would be.
        let elsewhere = base.join("Share/Library");
        library_at(&elsewhere, ID);
        let volume = base.join("Volumes/T7");
        std::fs::create_dir_all(&volume).unwrap();
        std::os::unix::fs::symlink(&elsewhere, volume.join(STANDARD_LIBRARY_FOLDER)).unwrap();

        assert_eq!(discover_at(&volume), None);

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn volume_relative_path_is_the_library_root_below_its_volume() {
        let base = scratch("relative");
        let volumes_root = base.join("Volumes");
        let top = volumes_root.join("T7").join(STANDARD_LIBRARY_FOLDER);
        let nested = volumes_root.join("T7/Stuff/Models");
        let internal = base.join("Models");
        for root in [&top, &nested, &internal] {
            std::fs::create_dir_all(root).unwrap();
        }
        let env = env_with_volumes_root(volumes_root);

        assert_eq!(
            volume_relative_path(&top, &env).unwrap(),
            PathBuf::from(STANDARD_LIBRARY_FOLDER)
        );
        assert_eq!(
            volume_relative_path(&nested, &env).unwrap(),
            PathBuf::from("Stuff/Models")
        );
        // The split must rejoin to the root, which is how locate rebuilds it.
        for root in [&top, &nested, &internal] {
            let relative = volume_relative_path(root, &env).unwrap();
            assert!(relative.is_relative(), "{}", relative.display());
            assert_eq!(&volume_root_of(root, &env).unwrap().join(relative), root);
        }

        let _ = std::fs::remove_dir_all(&base);
    }
}
