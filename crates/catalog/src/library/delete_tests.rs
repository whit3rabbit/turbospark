//! Deleting installs where their library is now (4.6), offline. This Mac and
//! a library, T7, on a fake volume located through a fixed mount table. The
//! hazard under test: a row records the path an install had when it was
//! recorded, and that path can later answer for another drive, or for a
//! folder outside the library, so nothing is ever deleted through it.

use crate::library::{
    parse_mount_table, Availability, LibraryMarker, LibraryRegistry, LocationEnv, MountSource,
    MARKER_FILE,
};
use crate::{InstalledModel, ModelModality, Store, StoreError};
use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::{symlink, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

// ------------------------------------------------------------------ fixture

#[derive(Clone, Copy, PartialEq, Eq)]
enum Place {
    Library,
    ThisMac,
}

/// An install a test made.
struct Install {
    id: String,
    dir: PathBuf,
}

/// A temp home with This Mac, and the library T7 on the fake volume
/// `<base>/Volumes/T7`.
struct Fixture {
    base: PathBuf,
    home: PathBuf,
    /// The library root, `<base>/Volumes/T7/TurboSpark`.
    root: PathBuf,
    store: Store,
    env: LocationEnv,
    library: String,
}

impl Fixture {
    fn new() -> Self {
        let base = std::env::temp_dir().join(format!(
            "catalog-library-delete-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        // Canonical, because the temp dir is behind the /var symlink and the
        // location rules compare canonical paths.
        let base = base.canonicalize().unwrap();
        let home = base.join("home");
        let volume = base.join("Volumes/T7");
        fs::create_dir_all(&home).unwrap();
        fs::create_dir_all(&volume).unwrap();
        let table = format!(
            "/dev/disk3s1s1 on / (apfs, sealed, local, read-only)\n\
             /dev/disk20s1 on {} (hfs, local, journaled)\n",
            volume.display()
        );
        // `home_device` differs from the temp dir's device so the same-disk
        // refusal does not fire for the fake volume.
        let env = LocationEnv {
            mount_source: MountSource::Fixed(parse_mount_table(&table)),
            volumes_root: base.join("Volumes"),
            home_device: fs::metadata(&base).unwrap().dev().wrapping_add(1),
        };
        let library = LibraryRegistry::load(&home)
            .add(&volume, Some("T7"), &env)
            .unwrap()
            .view
            .record
            .id;
        Self {
            root: volume.join("TurboSpark"),
            store: Store::new(&home),
            base,
            home,
            env,
            library,
        }
    }

    fn volume(&self) -> PathBuf {
        self.base.join("Volumes/T7")
    }

    fn staging(&self) -> PathBuf {
        self.root.join(".staging")
    }

    /// Makes a text install of `alias`: a nested folder of files, an empty one
    /// among them, and its row in the registry and, in the library, its
    /// marker. A `kind` of `vision-tower` makes the sidecar folder.
    fn text(&self, place: Place, alias: &str, kind: Option<&str>) -> Install {
        let (root, library) = match place {
            Place::Library => (self.root.clone(), Some(self.library.clone())),
            Place::ThisMac => (self.store.models_root(), None),
        };
        let suffix = if kind.is_some() {
            "gturbo-vision"
        } else {
            "gturbo"
        };
        let relative = PathBuf::from("text").join(format!("{alias}.{suffix}"));
        let dir = root.join(&relative);
        self.fill(&dir, alias);
        self.record(alias, kind, &dir, library.as_deref(), Some(relative));
        Install {
            id: format!("{}:text/{alias}", library.as_deref().unwrap_or("internal")),
            dir,
        }
    }

    /// Records a text row for `alias` at `dir`. A row in the library keeps
    /// the `relative` path it is given.
    fn record(
        &self,
        alias: &str,
        kind: Option<&str>,
        dir: &Path,
        library: Option<&str>,
        relative: Option<PathBuf>,
    ) {
        let row = InstalledModel {
            alias: alias.to_string(),
            repo: "owner/name".to_string(),
            revision: "main".to_string(),
            path: dir.to_path_buf(),
            family: "qwen3".to_string(),
            install_bytes: 3_000,
            installed_on: "2026-10-16".to_string(),
            status: "runs".to_string(),
            kind: kind.map(str::to_string),
            modality: ModelModality::Text,
            variant: None,
            library: library.map(str::to_string),
            relative_path: library.and(relative),
        };
        self.store.record_in_library(&[row], &self.env).unwrap();
    }

    fn fill(&self, dir: &Path, alias: &str) {
        fs::create_dir_all(dir.join("weights")).unwrap();
        fs::write(dir.join("config.json"), format!("{{\"alias\": {alias:?}}}")).unwrap();
        fs::write(dir.join("weights/part-0.bin"), vec![7u8; 3000]).unwrap();
        fs::write(dir.join("empty.txt"), b"").unwrap();
    }

    /// An image install in the library, `image/<alias>.gturbo`, with the
    /// manifest the listing reads. `kept` also makes the MLX source folder
    /// the installer keeps beside it. Image installs are not recorded.
    fn image(&self, alias: &str, revision: &str, kept: bool) -> PathBuf {
        let dir = self.root.join("image").join(format!("{alias}.gturbo"));
        fs::create_dir_all(dir.join("weights")).unwrap();
        fs::write(
            dir.join("manifest.json"),
            serde_json::json!({
                "source": {"model_id": "andrevp/Z-Image", "model_revision": revision}
            })
            .to_string(),
        )
        .unwrap();
        fs::write(dir.join("weights/part-0.bin"), vec![3u8; 2000]).unwrap();
        if kept {
            let source = self
                .root
                .join("image")
                .join(format!(".{alias}.{revision}.image-source"));
            fs::create_dir_all(source.join("text_encoder")).unwrap();
            fs::write(source.join("text_encoder/model.safetensors"), b"source").unwrap();
        }
        dir
    }

    fn image_id(&self, alias: &str) -> String {
        format!("{}:image/{alias}", self.library)
    }

    /// The aliases in `installed.json`, sorted.
    fn recorded(&self) -> Vec<String> {
        let mut aliases: Vec<String> = fs::read(self.home.join("installed.json"))
            .map(|bytes| serde_json::from_slice::<Vec<InstalledModel>>(&bytes).unwrap())
            .unwrap_or_default()
            .into_iter()
            .map(|row| row.alias)
            .collect();
        aliases.sort();
        aliases
    }

    /// The aliases in the library's marker, sorted.
    fn marked(&self) -> Vec<String> {
        let mut aliases: Vec<String> = LibraryMarker::read(&self.root)
            .unwrap()
            .unwrap()
            .models
            .into_iter()
            .map(|row| row.alias)
            .collect();
        aliases.sort();
        aliases
    }

    /// Takes the drive away as unplugging it would.
    fn unplug(&self) {
        fs::rename(self.volume(), self.base.join("unplugged-T7")).unwrap();
    }

    fn mark_newer(&self) {
        let path = self.root.join(MARKER_FILE);
        let mut json: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        json["format"] = serde_json::json!(2);
        fs::write(&path, serde_json::to_vec_pretty(&json).unwrap()).unwrap();
    }

    /// A folder outside every library with a file in it, and what it holds.
    fn outside(&self, name: &str) -> PathBuf {
        let dir = self.base.join(name);
        fs::create_dir_all(dir.join("inner")).unwrap();
        fs::write(dir.join("inner/precious.bin"), b"precious").unwrap();
        dir
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.base);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Node {
    Dir,
    File(Vec<u8>),
    Link(PathBuf),
}

/// Everything under `dir` with its bytes, a link as a link, empty when `dir`
/// is gone.
fn tree(dir: &Path) -> BTreeMap<PathBuf, Node> {
    fn walk(root: &Path, at: &Path, found: &mut BTreeMap<PathBuf, Node>) {
        let Ok(entries) = fs::read_dir(at) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let key = path.strip_prefix(root).unwrap().to_path_buf();
            let kind = entry.file_type().unwrap();
            if kind.is_symlink() {
                found.insert(key, Node::Link(fs::read_link(&path).unwrap()));
            } else if kind.is_dir() {
                found.insert(key, Node::Dir);
                walk(root, &path, found);
            } else {
                found.insert(key, Node::File(fs::read(&path).unwrap()));
            }
        }
    }
    let mut found = BTreeMap::new();
    walk(dir, dir, &mut found);
    found
}

fn names(dir: &Path) -> Vec<String> {
    let mut found: Vec<String> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    found.sort();
    found
}

fn is_gone(path: &Path) -> bool {
    fs::symlink_metadata(path).is_err()
}

fn io_reason(result: Result<(), StoreError>) -> String {
    match result {
        Err(StoreError::Io(reason)) => reason,
        other => panic!("wanted an IO refusal, got {other:?}"),
    }
}

/// A folder at a restricted mode until dropped, which restores 0o755, so a
/// failing test still leaves its temp tree removable. Declared after the
/// [`Fixture`] it is in, so it drops first.
struct Locked(PathBuf);

impl Locked {
    fn new(dir: &Path) -> Self {
        fs::set_permissions(dir, fs::Permissions::from_mode(0o555)).unwrap();
        Self(dir.to_path_buf())
    }
}

impl Drop for Locked {
    fn drop(&mut self) {
        let _ = fs::set_permissions(&self.0, fs::Permissions::from_mode(0o755));
    }
}

fn not_connected() -> Result<(), StoreError> {
    Err(StoreError::NotConnected {
        library: "T7".to_string(),
    })
}

// ----------------------------------------------------------- deleting in a library

/// The folder, the registry row and the marker entry go together, and the
/// install next to it is untouched. A link inside the install is removed
/// with it and not followed, and the staging folder it was renamed into holds
/// nothing afterwards.
#[test]
fn a_text_install_is_deleted_with_its_row_and_marker_entry() {
    let fx = Fixture::new();
    let qwen = fx.text(Place::Library, "qwen", None);
    let other = fx.text(Place::Library, "other", None);
    let outside = fx.outside("outside");
    symlink(&outside, qwen.dir.join("linked")).unwrap();
    let kept = tree(&other.dir);
    let outside_before = tree(&outside);
    assert_eq!(fx.recorded(), ["other", "qwen"]);
    assert_eq!(fx.marked(), ["other", "qwen"]);

    fx.store.delete_install(&qwen.id, &fx.env).unwrap();

    assert!(is_gone(&qwen.dir));
    assert_eq!(fx.recorded(), ["other"]);
    assert_eq!(fx.marked(), ["other"]);
    assert_eq!(tree(&other.dir), kept, "the install beside it is untouched");
    assert_eq!(tree(&outside), outside_before, "a link inside was followed");
    assert_eq!(names(&fx.staging()), Vec::<String>::new());
    // Deleting it again has nothing left to find.
    assert_eq!(
        fx.store.delete_install(&qwen.id, &fx.env),
        Err(StoreError::UnknownInstall {
            install_id: qwen.id.clone()
        })
    );
}

/// A vision tower is its own install in the text folder, under its own
/// suffix. Deleting it removes that folder and nothing else: not a text
/// install, and not an unrecorded folder named for the tower's alias.
#[test]
fn a_vision_tower_deletes_only_its_own_folder() {
    let fx = Fixture::new();
    let text = fx.text(Place::Library, "qwen", None);
    let tower = fx.text(Place::Library, "qwen-vision", Some("vision-tower"));
    // The folder a tower row of the wrong kind would resolve to.
    let witness = fx.root.join("text/qwen-vision.gturbo");
    fx.fill(&witness, "witness");
    let (text_before, witness_before) = (tree(&text.dir), tree(&witness));

    fx.store.delete_install(&tower.id, &fx.env).unwrap();

    assert!(is_gone(&tower.dir));
    assert_eq!(tree(&text.dir), text_before);
    assert_eq!(tree(&witness), witness_before);
    assert_eq!(fx.recorded(), ["qwen"]);
    assert_eq!(fx.marked(), ["qwen"]);
}

/// An image install is not recorded, so there is nothing to forget: the
/// folder and the MLX source tree the installer kept beside it go, the
/// registry and the marker are not rewritten, and the next image is intact.
#[test]
fn an_image_install_is_deleted_with_its_kept_source() {
    let fx = Fixture::new();
    fx.text(Place::Library, "qwen", None);
    let pic = fx.image("pic", "r1", true);
    let other = fx.image("other", "r1", true);
    let other_source = fx.root.join("image/.other.r1.image-source");
    let registry = fs::read(fx.home.join("installed.json")).unwrap();
    let marker = fs::read(fx.root.join(MARKER_FILE)).unwrap();
    let (other_before, source_before) = (tree(&other), tree(&other_source));

    fx.store
        .delete_image_install(&fx.image_id("pic"), &fx.env)
        .unwrap();

    assert!(is_gone(&pic));
    assert!(is_gone(&fx.root.join("image/.pic.r1.image-source")));
    assert_eq!(tree(&other), other_before);
    assert_eq!(tree(&other_source), source_before);
    assert_eq!(fs::read(fx.home.join("installed.json")).unwrap(), registry);
    assert_eq!(fs::read(fx.root.join(MARKER_FILE)).unwrap(), marker);
    assert_eq!(names(&fx.staging()), Vec::<String>::new());
}

/// The kept source folder is named from the manifest's revision, which a
/// drive supplies. A revision that walks out of the image folder, and a
/// source that is a link, are not deleted.
#[test]
fn a_hostile_revision_or_linked_source_is_left_alone() {
    let fx = Fixture::new();
    let outside = fx.outside("outside");
    let pic = fx.image("pic", "x/../../../../outside", false);
    let other = fx.image("other", "r1", false);
    fs::create_dir_all(fx.root.join("image/.pic.x")).unwrap();
    symlink(&outside, fx.root.join("image/.other.r1.image-source")).unwrap();
    let before = tree(&outside);

    fx.store
        .delete_image_install(&fx.image_id("pic"), &fx.env)
        .unwrap();
    fx.store
        .delete_image_install(&fx.image_id("other"), &fx.env)
        .unwrap();

    assert!(is_gone(&pic) && is_gone(&other));
    assert_eq!(tree(&outside), before);
    assert!(
        fs::symlink_metadata(fx.root.join("image/.other.r1.image-source"))
            .unwrap()
            .file_type()
            .is_symlink(),
        "a linked source is left where it is"
    );
    assert!(fx.root.join("image/.pic.x").is_dir());
}

// -------------------------------------------------------------- what is refused

/// A library that is not connected is refused with its name, and nothing on
/// the way is touched. The row stays, and the install is where it was when
/// the drive comes back.
#[test]
fn an_unplugged_library_is_refused_and_keeps_its_rows() {
    let fx = Fixture::new();
    let qwen = fx.text(Place::Library, "qwen", None);
    let pic = fx.image("pic", "r1", false);
    let (qwen_before, pic_before) = (tree(&qwen.dir), tree(&pic));
    fx.unplug();
    let before = tree(&fx.base);

    assert_eq!(fx.store.delete_install(&qwen.id, &fx.env), not_connected());
    assert_eq!(
        fx.store.delete_image_install(&fx.image_id("pic"), &fx.env),
        not_connected()
    );

    assert_eq!(tree(&fx.base), before);
    assert_eq!(fx.recorded(), ["qwen"]);
    fs::rename(fx.base.join("unplugged-T7"), fx.volume()).unwrap();
    assert_eq!(tree(&qwen.dir), qwen_before);
    assert_eq!(tree(&pic), pic_before);
}

/// The row records the mount point of the time. A different drive now mounted
/// there, holding a folder at the recorded path, is not this library, and
/// nothing on it is touched, whether or not it carries a library marker of
/// its own.
#[test]
fn a_different_drive_at_the_recorded_path_is_never_touched() {
    let fx = Fixture::new();
    let qwen = fx.text(Place::Library, "qwen", None);
    fx.image("pic", "r1", false);
    fx.unplug();
    // The recorded path of the row now answers for another drive.
    let other_root = fx.volume().join("TurboSpark");
    let imposter = other_root.join("text/qwen.gturbo");
    fx.fill(&imposter, "imposter");
    fx.fill(&other_root.join("image/pic.gturbo"), "imposter");
    fs::write(other_root.join("image/pic.gturbo/manifest.json"), b"{}").unwrap();
    assert_eq!(
        qwen.dir, imposter,
        "the row's recorded path is the other drive's folder"
    );

    for marker in [false, true] {
        if marker {
            LibraryMarker::new(
                "7c1e4b52-9d03-4a86-b1f7-2e5a8c3d6f90".to_string(),
                "Other".to_string(),
            )
            .write(&other_root)
            .unwrap();
        }
        let before = tree(&fx.volume());

        assert_eq!(fx.store.delete_install(&qwen.id, &fx.env), not_connected());
        assert_eq!(
            fx.store.delete_image_install(&fx.image_id("pic"), &fx.env),
            not_connected()
        );

        assert_eq!(
            tree(&fx.volume()),
            before,
            "marker on the other drive: {marker}"
        );
        assert_eq!(fx.recorded(), ["qwen"], "the row stays");
    }
}

/// A link planted where the library's folders are is never followed. The row
/// of an install that is not reached as a plain folder is listed as missing,
/// and deleting it forgets the row and the marker entry and touches no file:
/// not the folder the link points at, and not the link.
#[test]
fn a_planted_link_is_never_followed() {
    // The modality folder is a link to a folder outside.
    let fx = Fixture::new();
    let qwen = fx.text(Place::Library, "qwen", None);
    let outside = fx.outside("outside");
    fs::create_dir_all(outside.join("qwen.gturbo")).unwrap();
    fs::write(outside.join("qwen.gturbo/weights.bin"), b"theirs").unwrap();
    let parked = fx.base.join("parked-text");
    fs::rename(fx.root.join("text"), &parked).unwrap();
    symlink(&outside, fx.root.join("text")).unwrap();
    let (outside_before, parked_before) = (tree(&outside), tree(&parked));
    let entry = &fx.store.installed_with_availability(&fx.env)[0];
    assert_eq!(
        entry.availability,
        Availability::Missing,
        "listed as missing"
    );

    fx.store.delete_install(&qwen.id, &fx.env).unwrap();

    assert_eq!(tree(&outside), outside_before);
    assert_eq!(tree(&parked), parked_before);
    assert!(fs::symlink_metadata(fx.root.join("text"))
        .unwrap()
        .file_type()
        .is_symlink());
    assert_eq!(fx.recorded(), Vec::<String>::new());
    assert_eq!(fx.marked(), Vec::<String>::new());
    assert!(!fx.staging().exists(), "nothing was renamed");

    // The install folder itself is a link to a folder outside.
    let fx = Fixture::new();
    let qwen = fx.text(Place::Library, "qwen", None);
    let outside = fx.outside("outside");
    let parked = fx.base.join("parked-qwen");
    fs::rename(&qwen.dir, &parked).unwrap();
    symlink(&outside, &qwen.dir).unwrap();
    let (outside_before, parked_before) = (tree(&outside), tree(&parked));

    fx.store.delete_install(&qwen.id, &fx.env).unwrap();

    assert_eq!(tree(&outside), outside_before);
    assert_eq!(tree(&parked), parked_before);
    assert!(fs::symlink_metadata(&qwen.dir)
        .unwrap()
        .file_type()
        .is_symlink());
    assert_eq!(fx.recorded(), Vec::<String>::new());

    // An image folder, or its install, that is a link is not an install.
    let fx = Fixture::new();
    let outside = fx.outside("outside");
    fx.image("pic", "r1", false);
    fs::rename(fx.root.join("image/pic.gturbo"), outside.join("pic.gturbo")).unwrap();
    symlink(outside.join("pic.gturbo"), fx.root.join("image/pic.gturbo")).unwrap();
    let before = tree(&outside);
    let unknown = |alias: &str| {
        Err(StoreError::UnknownInstall {
            install_id: fx.image_id(alias),
        })
    };
    assert_eq!(
        fx.store.delete_image_install(&fx.image_id("pic"), &fx.env),
        unknown("pic")
    );
    let parked = fx.base.join("parked-image");
    fs::rename(fx.root.join("image"), &parked).unwrap();
    symlink(&outside, fx.root.join("image")).unwrap();
    assert_eq!(
        fx.store.delete_image_install(&fx.image_id("pic"), &fx.env),
        unknown("pic")
    );
    assert_eq!(tree(&outside), before);
}

/// Staging is on the drive, and a staging folder that is a link would send
/// the renamed install off it. It is refused, with the install where it was.
#[test]
fn a_linked_staging_folder_is_refused() {
    let fx = Fixture::new();
    let qwen = fx.text(Place::Library, "qwen", None);
    let outside = fx.outside("outside");
    symlink(&outside, fx.staging()).unwrap();
    let (before, install) = (tree(&outside), tree(&qwen.dir));

    let reason = io_reason(fx.store.delete_install(&qwen.id, &fx.env));

    assert!(reason.contains("not a plain folder"), "{reason}");
    assert_eq!(tree(&outside), before);
    assert_eq!(tree(&qwen.dir), install);
    assert_eq!(fx.recorded(), ["qwen"]);
}

/// A library root that is a link off its drive is not followed: every row in
/// it is missing, a delete is refused rather than treated as a forget, and
/// the folder it leads to is untouched, markers included.
#[test]
fn a_library_root_linked_off_its_drive_is_refused() {
    let fx = Fixture::new();
    let qwen = fx.text(Place::Library, "qwen", None);
    fx.image("pic", "r1", false);
    let parked = fx.base.join("parked-library");
    fs::rename(&fx.root, &parked).unwrap();
    symlink(&parked, &fx.root).unwrap();
    let before = tree(&fx.base);

    for result in [
        fx.store.delete_install(&qwen.id, &fx.env),
        fx.store.delete_image_install(&fx.image_id("pic"), &fx.env),
    ] {
        let reason = io_reason(result);
        assert!(reason.contains("leads off its drive"), "{reason}");
    }

    assert_eq!(tree(&fx.base), before);
    assert_eq!(fx.recorded(), ["qwen"]);
}

/// A library a newer TurboSpark wrote is read-only here: nothing is deleted
/// or forgotten, a missing row included, because the marker would keep it.
#[test]
fn a_read_only_library_is_refused() {
    let fx = Fixture::new();
    let qwen = fx.text(Place::Library, "qwen", None);
    let gone = fx.text(Place::Library, "gone", None);
    fx.image("pic", "r1", false);
    fs::remove_dir_all(&gone.dir).unwrap();
    fx.mark_newer();
    let before = tree(&fx.base);
    let refused = Err(StoreError::LibraryReadOnly {
        library: "T7".to_string(),
    });

    assert_eq!(fx.store.delete_install(&qwen.id, &fx.env), refused);
    assert_eq!(fx.store.delete_install(&gone.id, &fx.env), refused);
    assert_eq!(
        fx.store.delete_image_install(&fx.image_id("pic"), &fx.env),
        refused
    );

    assert_eq!(tree(&fx.base), before);
}

/// A row may name only its alias's own install folder, as a move requires
/// (2.10): a marker row naming the whole `audio` folder, another install's
/// folder, or the wrong suffix would otherwise take other installs with it.
#[test]
fn a_row_that_names_another_folder_is_refused() {
    let fx = Fixture::new();
    let victim = fx.text(Place::Library, "victim", None);
    let audio = fx.root.join("audio/something.gturbo");
    fx.fill(&audio, "audio");
    let tower = fx.root.join("text/c.gturbo-vision");
    fx.fill(&tower, "c");
    let text = |alias: &str, relative: &str| {
        fx.record(
            alias,
            None,
            &fx.root.join(relative),
            Some(&fx.library),
            Some(PathBuf::from(relative)),
        );
        format!("{}:text/{alias}", fx.library)
    };
    let ids = [
        text("a", "text/victim.gturbo"),
        text("b", "audio"),
        // The tower's folder, but the row is not a vision tower.
        text("c", "text/c.gturbo-vision"),
        text("d", "text"),
    ];
    let before = tree(&fx.base);

    for id in &ids {
        let reason = io_reason(fx.store.delete_install(id, &fx.env));
        assert!(
            reason.contains("is not the install folder"),
            "{id}: {reason}"
        );
    }

    assert_eq!(tree(&fx.base), before, "nothing was deleted");
    assert!(victim.dir.is_dir());
    assert_eq!(fx.recorded().len(), 5);
}

/// An id that is not a recorded install, an image folder that is not there,
/// and a library that is not listed each say so, and an image id of This Mac
/// is not the catalog's to delete.
#[test]
fn ids_that_name_nothing_are_reported() {
    let fx = Fixture::new();
    fx.text(Place::Library, "qwen", None);
    let unknown = |id: &str| {
        Err(StoreError::UnknownInstall {
            install_id: id.to_string(),
        })
    };
    for id in [
        "garbage",
        "",
        &format!("{}:text/nope", fx.library),
        "internal:text/nope",
    ] {
        assert_eq!(fx.store.delete_install(id, &fx.env), unknown(id), "{id:?}");
    }
    let nope = fx.image_id("nope");
    assert_eq!(
        fx.store.delete_image_install(&nope, &fx.env),
        unknown(&nope)
    );
    for id in ["garbage", &format!("{}:text/qwen", fx.library)] {
        assert_eq!(fx.store.delete_image_install(id, &fx.env), unknown(id));
    }

    let reason = io_reason(fx.store.delete_image_install("not-listed:image/x", &fx.env));
    assert!(reason.contains("is listed"), "{reason}");
    let reason = io_reason(fx.store.delete_image_install("internal:image/x", &fx.env));
    assert!(reason.contains("This Mac"), "{reason}");
    // An alias that could name a folder outside the library never does.
    let reason = io_reason(
        fx.store
            .delete_image_install(&fx.image_id("../escape"), &fx.env),
    );
    assert!(reason.contains("not a valid install alias"), "{reason}");
    // Audio installs are deleted by the audio catalog, which checks receipts.
    let sound = fx.root.join("audio/sound.gturbo");
    fx.fill(&sound, "sound");
    let relative = PathBuf::from("audio/sound.gturbo");
    let row = InstalledModel {
        alias: "sound".to_string(),
        repo: "owner/name".to_string(),
        revision: "main".to_string(),
        path: sound.clone(),
        family: "whisper".to_string(),
        install_bytes: 3_000,
        installed_on: "2026-10-16".to_string(),
        status: "runs".to_string(),
        kind: None,
        modality: ModelModality::Audio,
        variant: None,
        library: Some(fx.library.clone()),
        relative_path: Some(relative),
    };
    fx.store.record_in_library(&[row], &fx.env).unwrap();
    let before = tree(&sound);
    let reason = io_reason(
        fx.store
            .delete_install(&format!("{}:audio/sound", fx.library), &fx.env),
    );
    assert!(reason.contains("audio"), "{reason}");
    assert_eq!(tree(&sound), before);
}

/// A folder already gone from a connected library is simply forgotten: the
/// registry and the marker lose the row, no file is touched, and no staging
/// folder is made for it.
#[test]
fn a_missing_folder_is_forgotten_and_nothing_is_touched() {
    let fx = Fixture::new();
    let gone = fx.text(Place::Library, "gone", None);
    let other = fx.text(Place::Library, "other", None);
    fs::remove_dir_all(&gone.dir).unwrap();
    let before = tree(&other.dir);

    fx.store.delete_install(&gone.id, &fx.env).unwrap();

    assert_eq!(fx.recorded(), ["other"]);
    assert_eq!(fx.marked(), ["other"]);
    assert_eq!(tree(&other.dir), before);
    assert!(!fx.staging().exists());
}

/// If the row cannot be forgotten once the folder has been renamed aside,
/// the folder comes back exactly as it was, nothing is left in staging, and
/// the row and the marker entry are still there.
#[test]
fn a_failed_forget_puts_the_folder_back() {
    let fx = Fixture::new();
    let qwen = fx.text(Place::Library, "qwen", None);
    let before = tree(&qwen.dir);
    let marker = fs::read(fx.root.join(MARKER_FILE)).unwrap();
    // The registry is written through a temp file in the home.
    let _locked = Locked::new(&fx.home);

    let reason = io_reason(fx.store.delete_install(&qwen.id, &fx.env));

    assert!(reason.contains("installed.json"), "{reason}");
    assert_eq!(tree(&qwen.dir), before);
    assert_eq!(names(&fx.staging()), Vec::<String>::new());
    assert_eq!(fx.recorded(), ["qwen"]);
    assert_eq!(fs::read(fx.root.join(MARKER_FILE)).unwrap(), marker);
}

/// If the renamed folder cannot be removed after the row was forgotten, the
/// folder comes back and the row is recorded again, and the error says why.
/// The removal may have got part of the way before it failed, so the folder
/// can come back with some files gone: it is listed, and a delete finishes it.
#[test]
fn a_failed_removal_puts_the_folder_and_row_back() {
    let fx = Fixture::new();
    let qwen = fx.text(Place::Library, "qwen", None);
    let _locked = Locked::new(&qwen.dir.join("weights"));

    let reason = io_reason(fx.store.delete_install(&qwen.id, &fx.env));

    assert!(reason.contains("removing"), "{reason}");
    assert_eq!(
        fs::read(qwen.dir.join("weights/part-0.bin")).unwrap(),
        vec![7u8; 3000]
    );
    assert_eq!(names(&fx.staging()), Vec::<String>::new());
    assert_eq!(fx.recorded(), ["qwen"]);
    assert_eq!(fx.marked(), ["qwen"]);
    let listed = fx.store.installed_with_availability(&fx.env);
    assert_eq!(listed[0].availability, Availability::Available);
}

// ------------------------------------------------------------------ This Mac

/// This Mac's installs go where they always went: the managed folder, the
/// flat name from before modality folders, a vision tower, and a custom
/// location (`pull --out`), each with its row, and nothing else.
#[test]
fn this_mac_installs_are_deleted_at_their_own_folders() {
    let fx = Fixture::new();
    let qwen = fx.text(Place::ThisMac, "qwen", None);
    let tower = fx.text(Place::ThisMac, "qwen-vision", Some("vision-tower"));
    let keep = fx.text(Place::ThisMac, "keep", None);
    let flat = fx.store.models_root().join("flat.gturbo");
    fx.fill(&flat, "flat");
    fx.record("flat", None, &flat, None, None);
    let custom = fx.base.join("elsewhere/odd");
    fx.fill(&custom, "odd");
    fx.record("odd", None, &custom, None, None);
    let kept = tree(&keep.dir);

    for (id, dir) in [
        (qwen.id.as_str(), &qwen.dir),
        (tower.id.as_str(), &tower.dir),
        ("internal:text/flat", &flat),
        ("internal:text/odd", &custom),
    ] {
        fx.store.delete_install(id, &fx.env).unwrap();
        assert!(is_gone(dir), "{id}");
    }

    assert_eq!(fx.recorded(), ["keep"]);
    assert_eq!(tree(&keep.dir), kept);
    assert!(
        fx.base.join("elsewhere").is_dir(),
        "only the install folder"
    );
}

/// A row whose folder is gone is forgotten, and a link is unlinked and never
/// followed.
#[test]
fn a_missing_or_linked_this_mac_install_touches_nothing_else() {
    let fx = Fixture::new();
    let gone = fx.text(Place::ThisMac, "gone", None);
    let linked = fx.text(Place::ThisMac, "linked", None);
    fs::remove_dir_all(&gone.dir).unwrap();
    let outside = fx.outside("outside");
    fs::remove_dir_all(&linked.dir).unwrap();
    symlink(&outside, &linked.dir).unwrap();
    let before = tree(&outside);

    fx.store.delete_install(&gone.id, &fx.env).unwrap();
    fx.store.delete_install(&linked.id, &fx.env).unwrap();

    assert!(is_gone(&linked.dir));
    assert_eq!(tree(&outside), before);
    assert_eq!(fx.recorded(), Vec::<String>::new());
}

/// The recorded path of a This Mac row is deleted only when it is the
/// alias's own folder in the models folder, or a custom folder that holds
/// nothing the store manages: not the models folder or the home, not a
/// modality folder or another install's folder, not a folder that contains
/// another install or a library, and not a path spelled with `..`.
#[test]
fn this_mac_refuses_a_row_that_covers_other_installs() {
    let fx = Fixture::new();
    let keep = fx.text(Place::ThisMac, "keep", None);
    let models = fx.store.models_root();
    let nested = fx.base.join("custom/inner");
    fx.fill(&nested, "inner");
    fx.record("inner", None, &nested, None, None);
    fs::create_dir_all(fx.base.join("custom")).unwrap();
    let dotted = models.join("text/../text/keep.gturbo");
    let hostile: [(&str, PathBuf); 8] = [
        ("models", models.clone()),
        ("modality", models.join("text")),
        ("home", fx.home.clone()),
        ("world", fx.base.clone()),
        ("dotted", dotted),
        ("neighbor", models.join("text/keep.gturbo")),
        ("wrapper", fx.base.join("custom")),
        ("drive", fx.volume()),
    ];
    for (alias, path) in &hostile {
        fx.record(alias, None, path, None, None);
    }
    let before = tree(&fx.base);

    for (alias, _) in &hostile {
        let reason = io_reason(
            fx.store
                .delete_install(&format!("internal:text/{alias}"), &fx.env),
        );
        assert!(reason.contains("so it is not deleted"), "{alias}: {reason}");
    }

    assert_eq!(tree(&fx.base), before, "nothing was deleted");
    assert_eq!(fx.recorded().len(), 1 + 1 + hostile.len());
    assert!(keep.dir.is_dir());
}

/// A custom install under a volume that is not mounted is not connected, and
/// is refused with the volume's name instead of being reported as gone.
#[test]
fn a_custom_install_on_an_unmounted_volume_is_not_connected() {
    let fx = Fixture::new();
    let gone = fx.base.join("Volumes/Gone/model");
    fx.record("away", None, &gone, None, None);

    assert_eq!(
        fx.store.delete_install("internal:text/away", &fx.env),
        Err(StoreError::NotConnected {
            library: "Gone".to_string()
        })
    );
    assert_eq!(fx.recorded(), ["away"]);
}

// ------------------------------------------------- guards found by the review

/// A row an older writer stripped of its library and relative path records
/// the mount point of the time. With its library unplugged and another drive
/// mounted under the same name, that path answers for the other drive and
/// the row lists as available. It sits where the library is, so it is not
/// deleted by the path it recorded.
#[test]
fn a_stripped_row_inside_a_library_root_is_not_deleted_by_its_recorded_path() {
    let fx = Fixture::new();
    let qwen = fx.text(Place::Library, "qwen", None);
    // The older writer rewrote installed.json without the library fields.
    let rows_path = fx.home.join("installed.json");
    let mut rows: Vec<serde_json::Value> =
        serde_json::from_slice(&fs::read(&rows_path).unwrap()).unwrap();
    for row in &mut rows {
        let object = row.as_object_mut().unwrap();
        object.remove("library");
        object.remove("relative_path");
    }
    fs::write(&rows_path, serde_json::to_vec(&rows).unwrap()).unwrap();
    fx.unplug();
    // Another drive, mounted where the library was, with the same folder.
    fx.fill(&qwen.dir, "imposter");
    let before = tree(&fx.volume());
    let id = "internal:text/qwen";
    let listed = fx.store.installed_with_availability(&fx.env);
    let entry = listed.iter().find(|entry| entry.install_id == id).unwrap();
    assert_eq!(
        entry.availability,
        Availability::Available,
        "the imposter's folder lists as the row's own"
    );

    let reason = io_reason(fx.store.delete_install(id, &fx.env));

    assert!(reason.contains("inside the library"), "{reason}");
    assert_eq!(tree(&fx.volume()), before, "the other drive is untouched");
    assert_eq!(fx.recorded(), ["qwen"], "the row stays");
}

/// A marker is a drive's own word, and a row in it can name any plain
/// relative path. A row of another modality that names a text install's
/// folder must not delete it: the text delete takes only text rows.
#[test]
fn a_row_of_another_modality_cannot_delete_a_text_installs_folder() {
    let fx = Fixture::new();
    let qwen = fx.text(Place::Library, "qwen", None);
    let before = tree(&qwen.dir);
    // The same alias as the text install: alias plus modality is what names
    // an install, so the hostile row's own folder, if the text delete took
    // it, would be the text install's.
    let alias = "qwen";
    for modality in [ModelModality::Audio, ModelModality::Image] {
        let row = InstalledModel {
            alias: alias.to_string(),
            repo: "owner/name".to_string(),
            revision: "main".to_string(),
            path: qwen.dir.clone(),
            family: "whisper".to_string(),
            install_bytes: 3_000,
            installed_on: "2026-10-16".to_string(),
            status: "runs".to_string(),
            kind: None,
            modality,
            variant: None,
            library: Some(fx.library.clone()),
            relative_path: Some(PathBuf::from("text/qwen.gturbo")),
        };
        fx.store.record_in_library(&[row], &fx.env).unwrap();
        let kind = if modality == ModelModality::Audio {
            "audio"
        } else {
            "image"
        };

        let result = fx
            .store
            .delete_install(&format!("{}:{kind}/{alias}", fx.library), &fx.env);

        assert!(result.is_err(), "{kind}: {result:?}");
        assert_eq!(
            tree(&qwen.dir),
            before,
            "{kind}: the text install is intact"
        );
    }
}

/// The kept source folder is named from the manifest's revision. A revision
/// with a slash makes the name two components, `.pic.a` and `b.image-source`,
/// and a plain folder at that nested path is not the kept source: it stays.
#[test]
fn a_revision_with_a_slash_never_names_a_nested_folder() {
    let fx = Fixture::new();
    let pic = fx.image("pic", "a/b", false);
    let nested = fx.root.join("image/.pic.a/b.image-source");
    fs::create_dir_all(&nested).unwrap();
    fs::write(nested.join("keep.bin"), b"keep").unwrap();
    let before = tree(&fx.root.join("image/.pic.a"));

    fx.store
        .delete_image_install(&fx.image_id("pic"), &fx.env)
        .unwrap();

    assert!(is_gone(&pic));
    assert_eq!(tree(&fx.root.join("image/.pic.a")), before);
}
