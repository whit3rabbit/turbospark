//! Importing a store that Move Store moved (2.11), offline. A temp home and a
//! moved store on the fake volume T7, located through a fixed mount table,
//! or on the boot disk with the real home device. A write is made to fail
//! with a folder mode that [`Locked`] restores, or through the registry's
//! commit hook.

use super::super::before_next_commit;
use super::*;
use crate::library::{
    parse_mount_table, validate_location, Availability, LocationRefusal, MountSource,
    INTERNAL_LIBRARY_ID, LIBRARIES_FILE, MARKER_FILE,
};
use crate::EntryKind;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

const LOCAL: &str = "apfs, local, journaled";
/// Catalog music, which This Mac installs without an audio receipt.
const MUSIC: &str = "minimax-music3-4bit";
/// The folder Move Store moved the store into.
const STORE: &str = "TurboSparkStore";
const ID: &str = "0f6c2a3e-1b7d-4c55-9e0a-7d2f1c3b4a59";
const OTHER_ID: &str = "7a1e9c04-52b3-4f8d-a6e1-3c9b0d2f5e17";

struct Fixture {
    base: PathBuf,
    home: PathBuf,
    env: LocationEnv,
}

impl Fixture {
    /// A temp home and the fake volume T7. The home device is not the temp
    /// dir's, so T7 is not on the home's disk.
    fn new() -> Self {
        let base = scratch();
        let volume = base.join("Volumes/T7");
        fs::create_dir_all(&volume).unwrap();
        let table = format!(
            "/dev/disk3s1s1 on / (apfs, sealed, local, read-only)\n\
             /dev/disk20s1 on {} ({LOCAL})\n",
            volume.display()
        );
        let home_device = fs::metadata(&base).unwrap().dev().wrapping_add(1);
        Self::with(base, &table, home_device)
    }

    /// The boot disk: the temp dir's real device is the home's.
    fn on_home_disk() -> Self {
        let base = scratch();
        let home_device = fs::metadata(&base).unwrap().dev();
        Self::with(
            base,
            "/dev/disk3s1s1 on / (apfs, sealed, local, read-only)\n",
            home_device,
        )
    }

    fn with(base: PathBuf, table: &str, home_device: u64) -> Self {
        let home = base.join("home");
        fs::create_dir_all(&home).unwrap();
        Self {
            env: LocationEnv {
                mount_source: MountSource::Fixed(parse_mount_table(table)),
                volumes_root: base.join("Volumes"),
                home_device,
            },
            base,
            home,
        }
    }

    fn store(&self) -> Store {
        Store::new(&self.home)
    }

    fn import(&self, root: &Path) -> Result<ImportedStore, LibraryError> {
        LibraryRegistry::load(&self.home).import_moved_root(root, &self.env)
    }

    fn ids(&self) -> Vec<String> {
        LibraryRegistry::load(&self.home)
            .libraries()
            .into_iter()
            .map(|record| record.id)
            .collect()
    }

    fn rows_of(&self, id: &str) -> Vec<InstalledModel> {
        self.store()
            .read_rows()
            .into_iter()
            .filter(|row| row.library.as_deref() == Some(id))
            .collect()
    }

    /// Where a moved store on T7 is.
    fn on_t7(&self) -> PathBuf {
        self.base.join("Volumes/T7").join(STORE)
    }

    /// A custom-location install outside every store.
    fn custom(&self) -> PathBuf {
        self.base.join("elsewhere/custom.gturbo")
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.base);
    }
}

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "catalog-library-import-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    // The temp dir is behind the /var symlink, and the import records
    // canonical paths.
    dir.canonicalize().unwrap()
}

/// Sets a file's or folder's mode, and restores it on drop so the temp dir
/// can be removed, and the next step read, even when an assertion fails.
struct Locked {
    path: PathBuf,
    mode: u32,
}

impl Locked {
    fn new(path: &Path, mode: u32) -> Self {
        let previous = fs::metadata(path).unwrap().permissions().mode() & 0o7777;
        fs::set_permissions(path, fs::Permissions::from_mode(mode)).unwrap();
        Self {
            path: path.to_path_buf(),
            mode: previous,
        }
    }
}

impl Drop for Locked {
    fn drop(&mut self) {
        let _ = fs::set_permissions(&self.path, fs::Permissions::from_mode(self.mode));
    }
}

/// A store as Move Store left it.
struct Moved {
    root: PathBuf,
    models: PathBuf,
    /// The catalog audio alias installed with its receipt.
    audio: String,
}

/// Writes a moved store at `root`, as Move Store left one:
/// - recorded: a text model, a vision tower, catalog audio with its
///   receipt, catalog music without one, and a custom-location install at
///   `custom`, outside the store
/// - unrecorded: an image folder, an embedding, and an OCR install
/// - the catalog override
fn moved_store(root: &Path, custom: &Path) -> Moved {
    let models = root.join("models");
    let audio = AudioCatalog::embedded()
        .unwrap()
        .entries()
        .map(|profile| profile.identity.alias.clone())
        .find(|alias| alias != MUSIC)
        .unwrap();
    let vision = Some(EntryKind::VisionTower.as_str());
    let mut rows = Vec::new();
    for (alias, modality, relative, kind) in [
        (
            "qwen",
            ModelModality::Text,
            "text/qwen.gturbo".to_string(),
            None,
        ),
        (
            "llava",
            ModelModality::Text,
            "text/llava.gturbo-vision".to_string(),
            vision,
        ),
        (
            audio.as_str(),
            ModelModality::Audio,
            format!("audio/{audio}.gturbo"),
            None,
        ),
        (
            MUSIC,
            ModelModality::Audio,
            format!("audio/{MUSIC}.gturbo"),
            None,
        ),
    ] {
        let dir = models.join(relative);
        install_files(&dir);
        rows.push(row(alias, modality, dir, kind));
    }
    fs::write(
        models
            .join(format!("audio/{audio}.gturbo"))
            .join(RECEIPT_FILENAME),
        b"{}",
    )
    .unwrap();
    install_files(custom);
    rows.push(row(
        "custom",
        ModelModality::Text,
        custom.to_path_buf(),
        None,
    ));
    for unrecorded in [
        "image/flux.gturbo",
        "embeddings/gemma-embed",
        "ocr/unlimited-ocr",
    ] {
        install_files(&models.join(unrecorded));
    }
    fs::write(
        root.join("installed.json"),
        serde_json::to_vec_pretty(&rows).unwrap(),
    )
    .unwrap();
    fs::write(
        root.join(CATALOG_OVERRIDE),
        b"{\"schema_version\": 1, \"models\": []}",
    )
    .unwrap();
    Moved {
        root: root.to_path_buf(),
        models,
        audio,
    }
}

/// Adds `extra` to the moved store's `installed.json`.
fn record_moved(moved: &Moved, extra: InstalledModel) {
    let mut rows = Store::new(&moved.root).read_rows();
    rows.push(extra);
    fs::write(
        moved.root.join("installed.json"),
        serde_json::to_vec_pretty(&rows).unwrap(),
    )
    .unwrap();
}

/// A small install: files in a nested folder.
fn install_files(dir: &Path) {
    fs::create_dir_all(dir.join("weights")).unwrap();
    fs::write(dir.join("config.json"), format!("{:?}", dir.file_name())).unwrap();
    fs::write(dir.join("weights/part-0.bin"), vec![7u8; 3000]).unwrap();
}

fn row(alias: &str, modality: ModelModality, path: PathBuf, kind: Option<&str>) -> InstalledModel {
    InstalledModel {
        alias: alias.to_string(),
        repo: format!("owner/{alias}"),
        revision: "main".to_string(),
        path,
        family: "qwen3".to_string(),
        install_bytes: 4_096,
        installed_on: "2026-10-14".to_string(),
        status: "verified".to_string(),
        kind: kind.map(str::to_string),
        modality,
        variant: None,
        library: None,
        relative_path: None,
    }
}

/// Every entry under `dir` by relative path, not followed: a file's bytes,
/// `None` for anything else.
fn snapshot(dir: &Path) -> BTreeMap<PathBuf, Option<Vec<u8>>> {
    let mut found = BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(at) = stack.pop() {
        for entry in fs::read_dir(&at).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            let relative = path.strip_prefix(dir).unwrap().to_path_buf();
            let kind = entry.file_type().unwrap();
            if kind.is_dir() {
                stack.push(path);
                found.insert(relative, None);
            } else if kind.is_file() {
                found.insert(relative, Some(fs::read(&path).unwrap()));
            } else {
                found.insert(relative, None);
            }
        }
    }
    found
}

/// A file's bytes and inode: every write replaces the file, so a rewrite
/// with the same bytes still shows.
fn file_state(path: &Path) -> Option<(Vec<u8>, u64)> {
    Some((fs::read(path).ok()?, fs::metadata(path).ok()?.ino()))
}

fn names(dir: &Path) -> Vec<String> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    names
}

fn aliases(rows: &[InstalledModel]) -> Vec<String> {
    rows.iter().map(|row| row.alias.clone()).collect()
}

/// The observable completion: importing a moved store twice yields one
/// library holding every original row, listed and available, a marker
/// listing them, home copies of the embedding and OCR installs and of the
/// catalog override, and the moved store as it was but for its marker.
#[test]
fn importing_a_moved_store_twice_lists_every_row_in_one_library() {
    let fx = Fixture::new();
    let moved = moved_store(&fx.on_t7(), &fx.custom());
    let before = snapshot(&moved.root);

    let first = fx.import(&moved.root).unwrap();

    let id = first.view.record.id.clone();
    assert_ne!(id, INTERNAL_LIBRARY_ID);
    assert_eq!(fx.ids(), vec![INTERNAL_LIBRARY_ID.to_string(), id.clone()]);
    let record = &first.view.record;
    assert_eq!(record.name, STORE);
    assert_eq!(record.last_mount_path, fx.base.join("Volumes/T7"));
    assert_eq!(record.volume_relative_path, Path::new(STORE).join("models"));
    assert!(!record.default_for_downloads);
    assert_eq!(record.last_seen, Some(crate::install::today()));
    assert_eq!(first.view.root, Some(moved.models.clone()));
    assert!(first.view.available);
    assert_eq!(first.view.model_count, 4);
    assert!(first.skipped.is_empty(), "{:?}", first.skipped);
    assert_eq!(first.custom, vec!["custom"]);
    assert_eq!(first.needs_adoption, vec![MUSIC]);
    assert_eq!(
        first.copied_to_home,
        vec!["embeddings/gemma-embed", "ocr/unlimited-ocr"]
    );
    assert!(first.restored_catalog_override);
    assert!(first.copy_failures.is_empty(), "{:?}", first.copy_failures);

    // Every row inside the store is the library's, at its path in it.
    let expected: Vec<(String, PathBuf)> = vec![
        ("qwen".to_string(), "text/qwen.gturbo".into()),
        ("llava".to_string(), "text/llava.gturbo-vision".into()),
        (
            moved.audio.clone(),
            format!("audio/{}.gturbo", moved.audio).into(),
        ),
        (MUSIC.to_string(), format!("audio/{MUSIC}.gturbo").into()),
    ];
    let ours: Vec<(String, PathBuf)> = fx
        .rows_of(&id)
        .into_iter()
        .map(|row| {
            let relative = row.relative_path.unwrap();
            assert_eq!(row.path, moved.models.join(&relative));
            (row.alias, relative)
        })
        .collect();
    assert_eq!(ours, expected);
    let marker = LibraryMarker::read(&moved.models).unwrap().unwrap();
    assert_eq!(marker.id, id);
    assert_eq!(marker.name, STORE);
    let listed: Vec<(String, PathBuf)> = marker
        .models
        .iter()
        .map(|portable| (portable.alias.clone(), portable.relative_path.clone()))
        .collect();
    assert_eq!(listed, expected);
    // The custom-location row is recorded as it was, with no library.
    let custom = row("custom", ModelModality::Text, fx.custom(), None);
    assert!(fx.store().read_rows().contains(&custom));
    // Every row listed and available, the custom one as a custom location.
    let entries = fx.store().installed_with_availability(&fx.env);
    assert_eq!(entries.len(), 5);
    for entry in &entries {
        assert_eq!(
            entry.availability,
            Availability::Available,
            "{}",
            entry.install_id
        );
        let library = match entry.row.alias.as_str() {
            "custom" => "custom location",
            _ => STORE,
        };
        assert_eq!(entry.library_name, library, "{}", entry.install_id);
    }

    // The home copies, byte for byte, with no staging folder left.
    for folder in ["embeddings/gemma-embed", "ocr/unlimited-ocr"] {
        assert_eq!(
            snapshot(&fx.home.join("models").join(folder)),
            snapshot(&moved.models.join(folder))
        );
    }
    assert_eq!(names(&fx.home.join("models/embeddings")), ["gemma-embed"]);
    assert_eq!(names(&fx.home.join("models/ocr")), ["unlimited-ocr"]);
    assert_eq!(
        fs::read(fx.home.join(CATALOG_OVERRIDE)).unwrap(),
        fs::read(moved.root.join(CATALOG_OVERRIDE)).unwrap()
    );
    // The originals as they were, the marker the only addition.
    let mut after = snapshot(&moved.root);
    assert!(after
        .remove(&Path::new("models").join(MARKER_FILE))
        .is_some());
    assert_eq!(after, before);

    // Again: the same library, and nothing written anywhere.
    let watched = [
        fx.home.join("installed.json"),
        fx.home.join(LIBRARIES_FILE),
        fx.home.join(CATALOG_OVERRIDE),
        moved.models.join(MARKER_FILE),
    ];
    let states: Vec<_> = watched.iter().map(|path| file_state(path)).collect();
    let home = snapshot(&fx.home);
    let store = snapshot(&moved.root);

    let second = fx.import(&moved.root).unwrap();

    assert_eq!(second.view, first.view);
    assert!(second.skipped.is_empty());
    assert!(second.custom.is_empty());
    assert!(second.copied_to_home.is_empty());
    assert!(!second.restored_catalog_override);
    assert!(
        second.copy_failures.is_empty(),
        "{:?}",
        second.copy_failures
    );
    assert_eq!(second.needs_adoption, vec![MUSIC]);
    assert_eq!(fx.ids(), vec![INTERNAL_LIBRARY_ID.to_string(), id]);
    for (path, state) in watched.iter().zip(&states) {
        assert!(state.is_some(), "{} is missing", path.display());
        assert_eq!(&file_state(path), state, "{} was written", path.display());
    }
    assert_eq!(snapshot(&fx.home), home);
    assert_eq!(snapshot(&moved.root), store);
}

/// Move Store accepted the boot disk, where a library cannot be added
/// (1.10). A store it moved there imports with no refusal and stays listed.
#[test]
fn a_moved_store_on_the_home_disk_imports_and_stays_listed() {
    let fx = Fixture::on_home_disk();
    let moved = moved_store(&fx.base.join(STORE), &fx.custom());
    assert_eq!(
        validate_location(&moved.models, &fx.env),
        Err(LocationRefusal::SameDiskAsHome)
    );

    let imported = fx.import(&moved.root).unwrap();

    let record = &imported.view.record;
    assert_eq!(
        record.last_mount_path.join(&record.volume_relative_path),
        moved.models
    );
    assert_eq!(imported.view.root, Some(moved.models.clone()));
    assert!(imported.view.available);
    // Still listed, and available, when the list is next refreshed.
    let refreshed = LibraryRegistry::load(&fx.home).refresh(&fx.env);
    let view = refreshed
        .views
        .iter()
        .find(|view| view.record.id == record.id)
        .unwrap();
    assert!(view.available);
    assert_eq!(view.model_count, 4);
    let entries = fx.store().installed_with_availability(&fx.env);
    assert_eq!(entries.len(), 5);
    assert!(
        entries
            .iter()
            .all(|entry| entry.availability == Availability::Available),
        "{entries:?}"
    );
}

/// Alias plus modality stays unique across libraries (1.2): a moved row
/// whose alias This Mac or another library holds is skipped and reported,
/// and the holder's row stays as it was. So is a row naming a library the
/// moved store had listed itself.
#[test]
fn an_alias_held_elsewhere_is_skipped_and_reported() {
    let fx = Fixture::new();
    let moved = moved_store(&fx.on_t7(), &fx.custom());
    record_moved(
        &moved,
        InstalledModel {
            library: Some(OTHER_ID.to_string()),
            relative_path: Some("text/phi.gturbo".into()),
            ..row(
                "phi",
                ModelModality::Text,
                PathBuf::from("/Volumes/Old/TurboSpark/text/phi.gturbo"),
                None,
            )
        },
    );
    let this_mac = row(
        "qwen",
        ModelModality::Text,
        fx.home.join("models/text/qwen.gturbo"),
        None,
    );
    let other = InstalledModel {
        library: Some(OTHER_ID.to_string()),
        relative_path: Some(format!("audio/{MUSIC}.gturbo").into()),
        ..row(
            MUSIC,
            ModelModality::Audio,
            PathBuf::from(format!("/Volumes/Other/TurboSpark/audio/{MUSIC}.gturbo")),
            None,
        )
    };
    fx.store()
        .record_rows(&[this_mac.clone(), other.clone()])
        .unwrap();

    let imported = fx.import(&moved.root).unwrap();

    assert_eq!(imported.skipped, vec!["qwen", MUSIC, "phi"]);
    let rows = fx.store().read_rows();
    assert!(rows.contains(&this_mac));
    assert!(rows.contains(&other));
    let id = &imported.view.record.id;
    let ours = aliases(&fx.rows_of(id));
    assert_eq!(ours, vec!["llava".to_string(), moved.audio.clone()]);
    // The marker lists what the library holds, and no more.
    let marker = LibraryMarker::read(&moved.models).unwrap().unwrap();
    let listed: Vec<String> = marker
        .models
        .iter()
        .map(|portable| portable.alias.clone())
        .collect();
    assert_eq!(listed, ours);
    assert!(imported.needs_adoption.is_empty());
}

/// A moved store that is not there, as on a drive that is not connected,
/// is not connected, and nothing is written: the conversion can wait for
/// the drive (9.3).
#[test]
fn a_moved_store_that_is_not_there_is_not_connected_and_writes_nothing() {
    let fx = Fixture::new();
    let root = fx.base.join("Volumes/T9").join(STORE);

    assert_eq!(
        fx.import(&root),
        Err(LibraryError::NotConnected(root.clone()))
    );
    // The folder without its models folder is not the store either.
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join(CATALOG_OVERRIDE), b"{}").unwrap();
    assert_eq!(
        fx.import(&root),
        Err(LibraryError::NotConnected(root.clone()))
    );

    assert_eq!(snapshot(&fx.home), BTreeMap::new());
    assert_eq!(fx.ids(), vec![INTERNAL_LIBRARY_ID.to_string()]);
}

/// A marker in the moved models folder that cannot be read may describe a
/// library and its models, so the import refuses and never replaces it
/// (2.2), before copying, recording, or listing anything.
#[test]
fn an_unreadable_marker_refuses_the_import_and_is_never_overwritten() {
    let fx = Fixture::new();
    let moved = moved_store(&fx.on_t7(), &fx.custom());
    fs::write(moved.models.join(MARKER_FILE), "{").unwrap();
    let before = snapshot(&moved.root);

    let result = fx.import(&moved.root);

    assert!(
        matches!(result, Err(LibraryError::Marker(MarkerError::Malformed(_)))),
        "{result:?}"
    );
    assert_eq!(snapshot(&moved.root), before);
    assert_eq!(snapshot(&fx.home), BTreeMap::new());
}

/// The home's own embedding install and catalog override win: neither is
/// replaced, and only what the home lacks is copied.
#[test]
fn the_home_keeps_its_own_embedding_and_catalog_override() {
    let fx = Fixture::new();
    let moved = moved_store(&fx.on_t7(), &fx.custom());
    let own = fx.home.join("models/embeddings/gemma-embed");
    fs::create_dir_all(&own).unwrap();
    fs::write(own.join("config.json"), b"the home's own").unwrap();
    fs::write(fx.home.join(CATALOG_OVERRIDE), b"the home's own override").unwrap();
    let kept = (
        snapshot(&own),
        fs::read(fx.home.join(CATALOG_OVERRIDE)).unwrap(),
    );

    let imported = fx.import(&moved.root).unwrap();

    assert_eq!(imported.copied_to_home, vec!["ocr/unlimited-ocr"]);
    assert!(!imported.restored_catalog_override);
    assert_eq!(
        (
            snapshot(&own),
            fs::read(fx.home.join(CATALOG_OVERRIDE)).unwrap()
        ),
        kept
    );
}

/// The marker is written before the registry, so a marker that cannot be
/// written leaves no row recorded for its library (2.7). Once it can be,
/// the import finishes.
#[test]
fn a_marker_that_cannot_be_written_leaves_no_row_recorded() {
    let fx = Fixture::new();
    let moved = moved_store(&fx.on_t7(), &fx.custom());

    let result = {
        let _locked = Locked::new(&moved.models, 0o555);
        fx.import(&moved.root)
    };

    assert!(
        matches!(result, Err(LibraryError::Marker(MarkerError::Io(_)))),
        "{result:?}"
    );
    assert!(fx.store().read_rows().is_empty());
    assert!(!fx.home.join(LIBRARIES_FILE).exists());
    assert!(!moved.models.join(MARKER_FILE).exists());

    let imported = fx.import(&moved.root).unwrap();
    assert_eq!(fx.rows_of(&imported.view.record.id).len(), 4);
}

/// An import that stopped after its marker or its registry rows finishes
/// under the marker's id: the rows already recorded are this library's
/// rather than conflicts, rows only the marker lists are recorded too, and
/// nothing is listed twice.
#[test]
fn an_import_that_stopped_early_finishes_under_the_same_id() {
    let fx = Fixture::new();
    let moved = moved_store(&fx.on_t7(), &fx.custom());
    let qwen = PortableRow::from_installed(
        &row(
            "qwen",
            ModelModality::Text,
            moved.models.join("text/qwen.gturbo"),
            None,
        ),
        &moved.models,
    )
    .unwrap();
    // A row another Mac installed into the library after converting it.
    install_files(&moved.models.join("text/phi.gturbo"));
    let phi = PortableRow::from_installed(
        &row(
            "phi",
            ModelModality::Text,
            moved.models.join("text/phi.gturbo"),
            None,
        ),
        &moved.models,
    )
    .unwrap();
    let mut marker = LibraryMarker::new(ID.to_string(), "Earlier".to_string());
    marker.models = vec![qwen.clone(), phi];
    marker.write(&moved.models).unwrap();
    fx.store()
        .record_rows(&[qwen.to_installed(&moved.models, ID).unwrap()])
        .unwrap();

    let imported = fx.import(&moved.root).unwrap();

    assert_eq!(imported.view.record.id, ID);
    assert_eq!(imported.view.record.name, "Earlier");
    assert!(imported.skipped.is_empty(), "{:?}", imported.skipped);
    assert_eq!(
        fx.ids(),
        vec![INTERNAL_LIBRARY_ID.to_string(), ID.to_string()]
    );
    let all = vec![
        "qwen".to_string(),
        "llava".to_string(),
        moved.audio.clone(),
        MUSIC.to_string(),
        "phi".to_string(),
    ];
    assert_eq!(aliases(&fx.rows_of(ID)), all);
    let marker = LibraryMarker::read(&moved.models).unwrap().unwrap();
    let mut listed: Vec<String> = marker
        .models
        .iter()
        .map(|portable| portable.alias.clone())
        .collect();
    listed.sort();
    let mut all = all;
    all.sort();
    assert_eq!(listed, all);
}

/// A library list that cannot be written lists nothing and forgets the
/// library rows the import recorded, and the marker keeps them for the next
/// import. The custom-location row stays recorded, and the retry reports it
/// as imported rather than as a conflict with itself.
#[test]
fn a_failed_list_write_forgets_the_rows_the_import_recorded() {
    let fx = Fixture::new();
    let moved = moved_store(&fx.on_t7(), &fx.custom());
    // A newer TurboSpark writes the list after the rows and before the list
    // write, which must then refuse.
    let newer = serde_json::to_vec(&serde_json::json!({
        "format": 2,
        "internal_is_default": true,
        "libraries": []
    }))
    .unwrap();
    let list_path = fx.home.join(LIBRARIES_FILE);
    let written = newer.clone();
    before_next_commit(move || fs::write(&list_path, &written).unwrap());

    let result = fx.import(&moved.root);

    assert!(matches!(result, Err(LibraryError::Io(_))), "{result:?}");
    let marker = LibraryMarker::read(&moved.models).unwrap().unwrap();
    assert!(fx.rows_of(&marker.id).is_empty());
    assert_eq!(marker.models.len(), 4);
    assert_eq!(fs::read(fx.home.join(LIBRARIES_FILE)).unwrap(), newer);
    let custom = row("custom", ModelModality::Text, fx.custom(), None);
    assert_eq!(fx.store().read_rows(), vec![custom.clone()]);

    // The newer TurboSpark's list is gone again, as after a downgrade.
    fs::remove_file(fx.home.join(LIBRARIES_FILE)).unwrap();
    let retried = fx.import(&moved.root).unwrap();

    assert_eq!(retried.view.record.id, marker.id);
    assert!(retried.skipped.is_empty(), "{:?}", retried.skipped);
    assert_eq!(retried.custom, vec!["custom"]);
    assert_eq!(fx.rows_of(&marker.id).len(), 4);
    let rows = fx.store().read_rows();
    assert_eq!(rows.iter().filter(|row| **row == custom).count(), 1);
}

/// The home copies follow no link: an embedding install holding a link, or
/// an install folder that is itself a link, is not copied, and is
/// reported, with nothing read or written through the link. The rest of the
/// import goes ahead.
#[test]
fn a_link_in_a_home_only_install_is_not_copied_and_is_reported() {
    let fx = Fixture::new();
    let moved = moved_store(&fx.on_t7(), &fx.custom());
    let outside = fx.base.join("outside.bin");
    fs::write(&outside, b"not the store's").unwrap();
    std::os::unix::fs::symlink(
        &outside,
        moved.models.join("embeddings/gemma-embed/weights/link.bin"),
    )
    .unwrap();
    let outside_install = fx.base.join("outside-install");
    install_files(&outside_install);
    std::os::unix::fs::symlink(
        &outside_install,
        moved.models.join("embeddings/linked-embed"),
    )
    .unwrap();

    let imported = fx.import(&moved.root).unwrap();

    let failed: Vec<&str> = imported
        .copy_failures
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(
        failed,
        ["embeddings/gemma-embed", "embeddings/linked-embed"]
    );
    assert_eq!(imported.copied_to_home, vec!["ocr/unlimited-ocr"]);
    assert!(imported.restored_catalog_override);
    assert_eq!(
        names(&fx.home.join("models/embeddings")),
        Vec::<String>::new()
    );
    assert_eq!(fs::read(&outside).unwrap(), b"not the store's");
    assert_eq!(fx.rows_of(&imported.view.record.id).len(), 4);
}

/// A home copy that fails part way, as on a full disk or an unreadable
/// file, removes its partial copy and is reported, and the import goes on:
/// listing the moved store's models must not wait on the home's space. The
/// next import, the library already listed, copies it.
#[test]
fn a_failed_home_copy_leaves_nothing_and_the_next_import_copies_it() {
    let fx = Fixture::new();
    let moved = moved_store(&fx.on_t7(), &fx.custom());
    let unreadable = moved
        .models
        .join("embeddings/gemma-embed/weights/part-0.bin");

    let first = {
        let _locked = Locked::new(&unreadable, 0o000);
        fx.import(&moved.root).unwrap()
    };

    let failed: Vec<&str> = first
        .copy_failures
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(failed, ["embeddings/gemma-embed"]);
    assert_eq!(first.copied_to_home, vec!["ocr/unlimited-ocr"]);
    assert_eq!(
        names(&fx.home.join("models/embeddings")),
        Vec::<String>::new()
    );
    assert_eq!(fx.rows_of(&first.view.record.id).len(), 4);

    let second = fx.import(&moved.root).unwrap();

    assert_eq!(second.view.record.id, first.view.record.id);
    assert_eq!(second.copied_to_home, vec!["embeddings/gemma-embed"]);
    assert!(
        second.copy_failures.is_empty(),
        "{:?}",
        second.copy_failures
    );
    assert_eq!(
        snapshot(&fx.home.join("models/embeddings/gemma-embed")),
        snapshot(&moved.models.join("embeddings/gemma-embed"))
    );
    assert_eq!(names(&fx.home.join("models/embeddings")), ["gemma-embed"]);
}

/// The moved store's record is its models. One that exists but cannot be
/// read or parsed refuses the import before anything is written, rather than
/// listing the library with none of its models: the conversion stays
/// pending, and once the record is repaired every row is listed.
#[test]
fn an_unreadable_install_record_refuses_the_import_and_writes_nothing() {
    let fx = Fixture::new();
    let moved = moved_store(&fx.on_t7(), &fx.custom());
    let record = moved.root.join("installed.json");
    let good = fs::read(&record).unwrap();
    fs::write(&record, b"[{").unwrap();
    let before = snapshot(&moved.root);

    let result = fx.import(&moved.root);

    match &result {
        Err(LibraryError::Io(reason)) => assert!(reason.contains("installed.json"), "{reason}"),
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert_eq!(snapshot(&moved.root), before);
    assert_eq!(snapshot(&fx.home), BTreeMap::new());

    // A record that cannot be read, as after a drive's read error, is not
    // an empty one either.
    fs::write(&record, &good).unwrap();
    let before = snapshot(&moved.root);
    let result = {
        let _locked = Locked::new(&record, 0o000);
        fx.import(&moved.root)
    };
    match &result {
        Err(LibraryError::Io(reason)) => assert!(reason.contains("installed.json"), "{reason}"),
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert_eq!(snapshot(&moved.root), before);
    assert_eq!(snapshot(&fx.home), BTreeMap::new());

    let imported = fx.import(&moved.root).unwrap();
    assert_eq!(imported.view.model_count, 4);
    assert_eq!(imported.custom, vec!["custom"]);
}

/// The marker and the list are read again after the home copies, which can
/// take minutes: a library another process imported from the same folder
/// meanwhile is returned as it is, with nothing written, rather than
/// replaced by a second library under a new id.
#[test]
fn a_library_imported_during_the_copies_is_returned_without_writing() {
    let fx = Fixture::new();
    let moved = moved_store(&fx.on_t7(), &fx.custom());
    let watched = vec![
        fx.home.join("installed.json"),
        fx.home.join(LIBRARIES_FILE),
        moved.models.join(MARKER_FILE),
    ];
    type Seen = Option<(String, Vec<Option<(Vec<u8>, u64)>>)>;
    let seen: Rc<RefCell<Seen>> = Rc::new(RefCell::new(None));
    {
        let (home, root, env) = (fx.home.clone(), moved.root.clone(), fx.env.clone());
        let (watched, seen) = (watched.clone(), seen.clone());
        // Another process imports the same store while the copies run.
        after_next_copies(move || {
            let other = LibraryRegistry::load(&home)
                .import_moved_root(&root, &env)
                .unwrap();
            let states = watched.iter().map(|path| file_state(path)).collect();
            *seen.borrow_mut() = Some((other.view.record.id, states));
        });
    }

    let imported = fx.import(&moved.root).unwrap();

    let (id, states) = seen.borrow_mut().take().expect("the hook ran");
    assert_eq!(imported.view.record.id, id);
    assert_eq!(imported.view.model_count, 4);
    assert_eq!(fx.ids(), vec![INTERNAL_LIBRARY_ID.to_string(), id]);
    for (path, state) in watched.iter().zip(&states) {
        assert!(state.is_some(), "{} is missing", path.display());
        assert_eq!(&file_state(path), state, "{} was written", path.display());
    }
    // The copies this import made are still reported.
    assert_eq!(
        imported.copied_to_home,
        vec!["embeddings/gemma-embed", "ocr/unlimited-ocr"]
    );
}

/// A library inside another would list the same files twice, so a moved
/// store inside a connected library, or holding one, is refused with that
/// library's name before anything is written.
#[test]
fn a_moved_store_inside_or_around_a_library_is_refused() {
    // Inside: the moved store is in T7's library folder.
    let fx = Fixture::new();
    LibraryRegistry::load(&fx.home)
        .add(&fx.base.join("Volumes/T7"), None, &fx.env)
        .unwrap();
    let library_root = fx.base.join("Volumes/T7/TurboSpark");
    let inside = moved_store(&library_root.join(STORE), &fx.custom());
    let home = snapshot(&fx.home);
    let store = snapshot(&inside.root);

    assert_eq!(
        fx.import(&inside.root),
        Err(LibraryError::Refused(LocationRefusal::NestedIn {
            library: "T7".to_string()
        }))
    );
    assert_eq!(snapshot(&fx.home), home);
    assert_eq!(snapshot(&inside.root), store);

    // Around: a library was added inside the moved store's models folder.
    let fx = Fixture::new();
    let around = moved_store(&fx.on_t7(), &fx.custom());
    let shelf = around.models.join("shelf");
    fs::create_dir_all(&shelf).unwrap();
    LibraryRegistry::load(&fx.home)
        .add(&shelf, Some("Shelf"), &fx.env)
        .unwrap();
    let home = snapshot(&fx.home);
    let store = snapshot(&around.root);

    assert_eq!(
        fx.import(&around.root),
        Err(LibraryError::Refused(LocationRefusal::NestedIn {
            library: "Shelf".to_string()
        }))
    );
    assert_eq!(snapshot(&fx.home), home);
    assert_eq!(snapshot(&around.root), store);
}

/// Receipt-less catalog music whose folder is gone stays listed, as
/// missing, but is not reported as needing adoption: there is nothing to
/// adopt.
#[test]
fn receipt_less_music_whose_folder_is_gone_is_missing_not_in_need_of_adoption() {
    let fx = Fixture::new();
    let moved = moved_store(&fx.on_t7(), &fx.custom());
    fs::remove_dir_all(moved.models.join(format!("audio/{MUSIC}.gturbo"))).unwrap();

    let imported = fx.import(&moved.root).unwrap();

    assert!(
        imported.needs_adoption.is_empty(),
        "{:?}",
        imported.needs_adoption
    );
    let music = fx
        .store()
        .installed_with_availability(&fx.env)
        .into_iter()
        .find(|entry| entry.row.alias == MUSIC)
        .unwrap();
    assert_eq!(
        music.row.library.as_deref(),
        Some(imported.view.record.id.as_str())
    );
    assert_eq!(music.availability, Availability::Missing);
}

/// A recorded root that is the home is This Mac already, so the import
/// returns This Mac and writes nothing, rather than listing the same folder
/// a second time as a library.
#[test]
fn a_moved_root_that_is_the_home_is_this_mac() {
    let fx = Fixture::new();
    let qwen = fx.home.join("models/text/qwen.gturbo");
    install_files(&qwen);
    fx.store()
        .record_rows(&[row("qwen", ModelModality::Text, qwen, None)])
        .unwrap();
    let before = snapshot(&fx.home);

    let imported = fx.import(&fx.home).unwrap();

    assert!(imported.view.record.is_internal());
    assert_eq!(imported.view.model_count, 1);
    assert!(imported.copied_to_home.is_empty());
    assert_eq!(snapshot(&fx.home), before);
    assert_eq!(fx.ids(), vec![INTERNAL_LIBRARY_ID.to_string()]);

    // A home with no models folder yet is This Mac too, not a store that
    // is not connected.
    let fresh = Fixture::new();
    let imported = fresh.import(&fresh.home).unwrap();
    assert!(imported.view.record.is_internal());
    assert_eq!(snapshot(&fresh.home), BTreeMap::new());
}
