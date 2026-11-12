//! Moving installs between libraries (2.10), offline. This Mac and two
//! libraries, T7 and T8, on fake volumes located through a fixed mount
//! table. A step is made to fail with folder modes that [`Locked`] restores,
//! or through the seam `run_move` calls on the copy before verifying it.

use super::*;
use crate::audio_catalog::{AudioCatalog, RECEIPT_FILENAME};
use crate::directory_bytes;
use crate::library::{
    parse_mount_table, Availability, LibraryMarker, LibraryRegistry, MountSource,
    INTERNAL_LIBRARY_ID, MARKER_FILE,
};
use std::cell::{Cell, RefCell};
use std::fs;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Text,
    Audio,
    /// Not recorded: a folder under `<root>/image`.
    Image,
}

impl Kind {
    fn modality(self) -> ModelModality {
        match self {
            Self::Text => ModelModality::Text,
            Self::Audio => ModelModality::Audio,
            Self::Image => ModelModality::Image,
        }
    }
}

/// An install a test made.
struct Source {
    install_id: String,
    alias: String,
    dir: PathBuf,
    modality: ModelModality,
}

/// A temp home with This Mac, and the libraries T7 and T8 on fake volumes.
struct Fixture {
    base: PathBuf,
    store: Store,
    env: LocationEnv,
    t7: String,
    t8: String,
}

impl Fixture {
    fn new() -> Self {
        let base = std::env::temp_dir().join(format!(
            "catalog-library-move-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        // Canonical, because the temp dir is behind the /var symlink and the
        // location rules compare canonical paths.
        let base = base.canonicalize().unwrap();
        let home = base.join("home");
        fs::create_dir_all(&home).unwrap();
        let volumes = base.join("Volumes");
        for name in ["T7", "T8"] {
            fs::create_dir_all(volumes.join(name)).unwrap();
        }
        // `home_device` differs from the temp dir's device so the same-disk
        // refusal does not fire for the fake volumes.
        let table = format!(
            "/dev/disk3s1s1 on / (apfs, sealed, local, read-only)\n\
             /dev/disk20s1 on {} (hfs, local, journaled)\n\
             /dev/disk21s1 on {} (hfs, local, journaled)\n",
            volumes.join("T7").display(),
            volumes.join("T8").display()
        );
        let env = LocationEnv {
            mount_source: MountSource::Fixed(parse_mount_table(&table)),
            volumes_root: volumes.clone(),
            home_device: fs::metadata(&base).unwrap().dev().wrapping_add(1),
        };
        let add = |name: &str| {
            LibraryRegistry::load(&home)
                .add(&volumes.join(name), Some(name), &env)
                .unwrap()
                .view
                .record
                .id
        };
        let (t7, t8) = (add("T7"), add("T8"));
        Self {
            base,
            store: Store::new(&home),
            env,
            t7,
            t8,
        }
    }

    fn direction(&self, name: &str) -> (Option<&str>, Option<&str>) {
        let (t7, t8) = (Some(self.t7.as_str()), Some(self.t8.as_str()));
        match name {
            "This Mac to T7" => (None, t7),
            "T7 to This Mac" => (t7, None),
            "T7 to T8" => (t7, t8),
            _ => panic!("no direction {name}"),
        }
    }

    /// The root of `library`, `None` for This Mac.
    fn root(&self, library: Option<&str>) -> PathBuf {
        match library {
            None => self.store.models_root(),
            Some(id) if id == self.t7 => self.base.join("Volumes/T7/TurboSpark"),
            Some(_) => self.base.join("Volumes/T8/TurboSpark"),
        }
    }

    fn name(&self, library: Option<&str>) -> &'static str {
        match library {
            None => "This Mac",
            Some(id) if id == self.t7 => "T7",
            Some(_) => "T8",
        }
    }

    /// Where `alias` lands in `library`.
    fn destination(&self, library: Option<&str>, source: &Source) -> PathBuf {
        self.root(library)
            .join(source.modality.as_str())
            .join(format!("{}.gturbo", source.alias))
    }

    /// `move-` folders left where moves stage: the home for This Mac, and
    /// each library's `.staging`.
    fn leftovers(&self) -> Vec<String> {
        [
            self.store.root().to_path_buf(),
            self.root(Some(&self.t7)).join(".staging"),
            self.root(Some(&self.t8)).join(".staging"),
        ]
        .iter()
        .flat_map(|dir| names(dir))
        .filter(|name| name.starts_with("move-"))
        .collect()
    }

    /// Installs `alias` as `kind` into `library`: files in a nested folder,
    /// an empty one among them, and for a text or audio model its row in the
    /// registry and in the library's marker.
    fn install(&self, library: Option<&str>, kind: Kind, alias: &str) -> Source {
        let modality = kind.modality();
        let relative = PathBuf::from(modality.as_str()).join(format!("{alias}.gturbo"));
        let dir = self.root(library).join(&relative);
        fs::create_dir_all(dir.join("weights")).unwrap();
        fs::write(dir.join("config.json"), format!("{{\"alias\": {alias:?}}}")).unwrap();
        fs::write(dir.join("weights/part-0.bin"), vec![7u8; 3000]).unwrap();
        fs::write(dir.join("weights/part-1.bin"), vec![9u8; 5000]).unwrap();
        fs::write(dir.join("empty.txt"), b"").unwrap();
        if kind == Kind::Audio {
            // A catalog audio install's receipt. Its contents are not read.
            fs::write(dir.join(RECEIPT_FILENAME), b"{}").unwrap();
        }
        if kind != Kind::Image {
            let relative = library.map(|_| relative.as_path());
            self.record(recorded_row(alias, modality, None, &dir, library, relative));
        }
        Source {
            install_id: format!(
                "{}:{}/{alias}",
                library.unwrap_or(INTERNAL_LIBRARY_ID),
                modality.as_str()
            ),
            alias: alias.to_string(),
            dir,
            modality,
        }
    }

    /// Records `row` in the registry and, in a library, its marker.
    fn record(&self, row: InstalledModel) {
        self.store.record_in_library(&[row], &self.env).unwrap();
    }

    /// What a failed move must leave as it was.
    fn state(&self, source: &Path) -> State {
        let read = |path: PathBuf| fs::read_to_string(path).ok();
        State {
            registry: read(self.store.root().join("installed.json")),
            t7_marker: read(self.root(Some(&self.t7)).join(MARKER_FILE)),
            t8_marker: read(self.root(Some(&self.t8)).join(MARKER_FILE)),
            source: snapshot(source),
        }
    }

    fn marker(&self, library: &str) -> LibraryMarker {
        LibraryMarker::read(&self.root(Some(library)))
            .unwrap()
            .unwrap()
    }

    fn in_marker(&self, library: &str, alias: &str) -> bool {
        self.marker(library)
            .models
            .iter()
            .any(|row| row.alias == alias)
    }

    /// Rewrites the marker as a newer TurboSpark would, which makes the
    /// library read-only to this build.
    fn mark_newer(&self, library: &str) {
        let path = self.root(Some(library)).join(MARKER_FILE);
        let mut json: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        json["format"] = serde_json::json!(2);
        fs::write(&path, serde_json::to_vec_pretty(&json).unwrap()).unwrap();
    }

    /// Takes the volume `name` away.
    fn unplug(&self, name: &str) {
        fs::rename(
            self.base.join("Volumes").join(name),
            self.base.join(format!("unplugged-{name}")),
        )
        .unwrap();
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.base);
    }
}

/// The registry, both markers, and every source file with its bytes. The
/// JSON files are text so a failure shows what changed.
#[derive(Debug, PartialEq, Eq)]
struct State {
    registry: Option<String>,
    t7_marker: Option<String>,
    t8_marker: Option<String>,
    source: Vec<(PathBuf, Vec<u8>)>,
}

/// A path at a restricted mode until dropped, which restores an open one,
/// so a failing test still leaves its temp tree removable. Declared after
/// the [`Fixture`] it is in, so it drops first.
struct Locked {
    path: PathBuf,
    restore: u32,
}

impl Locked {
    /// A read-only folder (0o555), restored to 0o755.
    fn new(dir: &Path) -> Self {
        Self::set(dir, 0o555, 0o755)
    }

    /// An unreadable file (0o000), restored to 0o644.
    fn unreadable(file: &Path) -> Self {
        Self::set(file, 0o000, 0o644)
    }

    fn set(path: &Path, mode: u32, restore: u32) -> Self {
        fs::set_permissions(path, fs::Permissions::from_mode(mode)).unwrap();
        Self {
            path: path.to_path_buf(),
            restore,
        }
    }
}

impl Drop for Locked {
    fn drop(&mut self) {
        let _ = fs::set_permissions(&self.path, fs::Permissions::from_mode(self.restore));
    }
}

/// A registry row for `alias` at `path`, in `library` at `relative`.
fn recorded_row(
    alias: &str,
    modality: ModelModality,
    kind: Option<&str>,
    path: &Path,
    library: Option<&str>,
    relative: Option<&Path>,
) -> InstalledModel {
    InstalledModel {
        alias: alias.to_string(),
        repo: "owner/name".to_string(),
        revision: "main".to_string(),
        path: path.to_path_buf(),
        family: "qwen3".to_string(),
        install_bytes: 8_000,
        installed_on: "2026-10-15".to_string(),
        status: "runs".to_string(),
        kind: kind.map(str::to_string),
        modality,
        variant: None,
        library: library.map(str::to_string),
        relative_path: relative.map(Path::to_path_buf),
    }
}

/// Every file under `dir` with its bytes, empty when `dir` is gone.
fn snapshot(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    fn walk(root: &Path, at: &Path, found: &mut Vec<(PathBuf, Vec<u8>)>) {
        let Ok(entries) = fs::read_dir(at) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if entry.file_type().unwrap().is_dir() {
                walk(root, &path, found);
            } else {
                let bytes = fs::read(&path).unwrap();
                found.push((path.strip_prefix(root).unwrap().to_path_buf(), bytes));
            }
        }
    }
    let mut found = Vec::new();
    walk(dir, dir, &mut found);
    found.sort();
    found
}

fn names(dir: &Path) -> Vec<String> {
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default()
}

/// A move of `install_id` to `to` (This Mac for `None`) through the seam,
/// with `after_copy` run on the copy before it is verified. Returns the
/// progress reports too. With `cancel_on_progress`, the first report of
/// copied bytes cancels the move.
fn move_with(
    fx: &Fixture,
    install_id: &str,
    to: Option<&str>,
    free_bytes: u64,
    cancel_on_progress: bool,
    after_copy: &dyn Fn(&Path),
) -> (Result<MoveOutcome, MoveError>, Vec<(u64, u64)>) {
    let reports = RefCell::new(Vec::new());
    let cancel = AtomicBool::new(false);
    let progress = |copied: u64, total: u64| {
        reports.borrow_mut().push((copied, total));
        if cancel_on_progress && copied > 0 {
            cancel.store(true, Ordering::Relaxed);
        }
    };
    let to = to.unwrap_or(INTERNAL_LIBRARY_ID);
    let result = plan_move(&fx.store, install_id, to, &fx.env, free_bytes)
        .and_then(|plan| run_move(&fx.store, &plan, &fx.env, &progress, &cancel, after_copy));
    (result, reports.into_inner())
}

/// What every failed move leaves: no copy in staging or at the
/// destination, and the registry, both markers and the source's files as
/// they were, so the source is still registered and loadable (7.3).
fn assert_untouched(fx: &Fixture, source: &Source, to: Option<&str>, before: &State, case: &str) {
    assert!(fx.leftovers().is_empty(), "{case}: {:?}", fx.leftovers());
    let dest = fx.destination(to, source);
    assert!(
        fs::symlink_metadata(&dest).is_err(),
        "{case}: a copy is left at {}",
        dest.display()
    );
    assert_eq!(&fx.state(&source.dir), before, "{case}");
}

/// The completion check of 2.10 for success: a text row, an audio row and
/// an image folder move from This Mac to a drive, back, and between drives,
/// with their files, registry row and markers, and list at the destination.
#[test]
fn installs_move_with_their_rows_and_markers_in_every_direction() {
    for kind in [Kind::Text, Kind::Audio, Kind::Image] {
        for direction in ["This Mac to T7", "T7 to This Mac", "T7 to T8"] {
            let case = format!("{kind:?} {direction}");
            let fx = Fixture::new();
            let (from, to) = fx.direction(direction);
            let source = fx.install(from, kind, "qwen");
            let original = fx.store.read_rows();
            let payload = snapshot(&source.dir);
            let total = directory_bytes(&source.dir);
            let reports = RefCell::new(Vec::new());

            // Exactly enough free space is enough.
            let moved = move_install(
                &fx.store,
                &source.install_id,
                to.unwrap_or(INTERNAL_LIBRARY_ID),
                &fx.env,
                total,
                &|copied, all| reports.borrow_mut().push((copied, all)),
                &AtomicBool::new(false),
            );

            assert_eq!(moved, Ok(MoveOutcome::Moved), "{case}");
            let dest = fx.destination(to, &source);
            assert_eq!(snapshot(&dest), payload, "{case}");
            assert!(fs::symlink_metadata(&source.dir).is_err(), "{case}");
            assert!(fx.leftovers().is_empty(), "{case}: {:?}", fx.leftovers());
            let reports = reports.into_inner();
            assert_eq!(reports.first(), Some(&(0, total)), "{case}");
            assert_eq!(reports.last(), Some(&(total, total)), "{case}");

            if kind == Kind::Image {
                assert!(fx.store.read_rows().is_empty(), "{case}");
                assert!(!fx.in_marker(&fx.t7, "qwen"), "{case}");
                assert!(!fx.in_marker(&fx.t8, "qwen"), "{case}");
                continue;
            }
            let relative = PathBuf::from(source.modality.as_str()).join("qwen.gturbo");
            let expected = InstalledModel {
                path: dest.clone(),
                library: to.map(str::to_string),
                relative_path: to.map(|_| relative.clone()),
                ..original[0].clone()
            };
            assert_eq!(fx.store.read_rows(), vec![expected], "{case}");
            if let Some(to) = to {
                let marker = fx.marker(to);
                assert!(
                    marker
                        .models
                        .iter()
                        .any(|row| row.alias == "qwen" && row.relative_path == relative),
                    "{case}: {:?}",
                    marker.models
                );
            }
            if let Some(from) = from {
                assert!(!fx.in_marker(from, "qwen"), "{case}");
            }
            let id = format!(
                "{}:{}/qwen",
                to.unwrap_or(INTERNAL_LIBRARY_ID),
                source.modality.as_str()
            );
            let listed = fx.store.installed_with_availability(&fx.env);
            let entry = listed
                .iter()
                .find(|entry| entry.install_id == id)
                .unwrap_or_else(|| panic!("{case}: {id} is not listed"));
            assert_eq!(entry.availability, Availability::Available, "{case}");
            assert_eq!(entry.library_name, fx.name(to), "{case}");
            assert_eq!(entry.row.path, dest, "{case}");
        }
    }
}

/// Cancelled from the progress callback in the middle of the copy, a move
/// removes its copy and leaves everything else as it was.
#[test]
fn a_cancelled_move_removes_its_copy_and_leaves_the_source_registered() {
    for direction in ["This Mac to T7", "T7 to This Mac"] {
        let fx = Fixture::new();
        let (from, to) = fx.direction(direction);
        let source = fx.install(from, Kind::Text, "qwen");
        let before = fx.state(&source.dir);

        let (result, reports) = move_with(&fx, &source.install_id, to, u64::MAX, true, &|_| {});

        assert_eq!(result, Err(MoveError::Cancelled), "{direction}");
        let (copied, total) = *reports.last().unwrap();
        assert!(0 < copied && copied < total, "{direction}: {reports:?}");
        assert_untouched(&fx, &source, to, &before, direction);
    }
}

/// A file the copy cannot read fails the move part way, and the part
/// already copied is removed.
#[test]
fn a_copy_that_fails_part_way_is_removed() {
    let fx = Fixture::new();
    let source = fx.install(Some(&fx.t7), Kind::Audio, "voice");
    let before = fx.state(&source.dir);

    let unreadable = Locked::unreadable(&source.dir.join("weights/part-1.bin"));
    let (result, _) = move_with(
        &fx,
        &source.install_id,
        Some(&fx.t8),
        u64::MAX,
        false,
        &|_| {},
    );
    drop(unreadable);

    match &result {
        Err(MoveError::Io(reason)) => assert!(reason.contains("part-1.bin"), "{reason}"),
        other => panic!("expected an IO error, got {other:?}"),
    }
    assert_untouched(&fx, &source, Some(&fx.t8), &before, "unreadable");
}

/// A copy that lost a file, or lost bytes of one, is refused and removed.
/// A lost empty file leaves the byte total equal, and a truncated file
/// leaves the file count equal, so both are compared.
#[test]
fn a_copy_that_does_not_match_its_source_is_removed() {
    for case in ["lost part-0.bin", "lost empty.txt", "truncated part-1.bin"] {
        let fx = Fixture::new();
        let source = fx.install(None, Kind::Text, "qwen");
        let before = fx.state(&source.dir);
        let total = directory_bytes(&source.dir);

        let (result, _) = move_with(
            &fx,
            &source.install_id,
            Some(&fx.t7),
            u64::MAX,
            false,
            &|copy| match case {
                "lost part-0.bin" => fs::remove_file(copy.join("weights/part-0.bin")).unwrap(),
                "lost empty.txt" => fs::remove_file(copy.join("empty.txt")).unwrap(),
                _ => fs::OpenOptions::new()
                    .write(true)
                    .open(copy.join("weights/part-1.bin"))
                    .unwrap()
                    .set_len(1000)
                    .unwrap(),
            },
        );

        let (lost, found_files) = match case {
            "lost part-0.bin" => (3000, 3),
            "lost empty.txt" => (0, 3),
            _ => (4000, 4),
        };
        assert_eq!(
            result,
            Err(MoveError::VerifyFailed {
                expected: total,
                found: total - lost,
                expected_files: 4,
                found_files,
            }),
            "{case}"
        );
        assert_untouched(&fx, &source, Some(&fx.t7), &before, case);
    }
}

enum Expect {
    Exactly(MoveError),
    IoMentioning(&'static str),
}

/// Every refusal comes before anything is copied: no progress, no staging
/// folder, and nothing changed.
#[test]
fn moves_are_refused_before_anything_is_copied() {
    for case in [
        "source not connected",
        "destination not connected",
        "same library",
        "same library on This Mac",
        "read-only destination",
        "read-only source",
        "insufficient space",
        "alias at the destination",
        "image alias at the destination",
        "link in the install",
        "linked install folder",
    ] {
        let fx = Fixture::new();
        let (t7, t8) = (Some(fx.t7.as_str()), Some(fx.t8.as_str()));
        let (from, to, kind) = match case {
            "source not connected" | "read-only source" => (t7, None, Kind::Text),
            "same library" => (t7, t7, Kind::Text),
            "same library on This Mac" => (None, None, Kind::Text),
            "image alias at the destination" | "linked install folder" => (t7, t8, Kind::Image),
            _ => (None, t8, Kind::Text),
        };
        let source = fx.install(from, kind, "qwen");
        let total = directory_bytes(&source.dir);
        let dest = fx.destination(to, &source);
        let outside = fx.base.join("outside");
        let mut free = u64::MAX;
        let expected = match case {
            "source not connected" => {
                fx.unplug("T7");
                Expect::Exactly(MoveError::NotConnected("T7".to_string()))
            }
            "destination not connected" => {
                fx.unplug("T8");
                Expect::Exactly(MoveError::NotConnected("T8".to_string()))
            }
            "same library" | "same library on This Mac" => Expect::Exactly(MoveError::SameLibrary),
            "read-only destination" => {
                fx.mark_newer(&fx.t8);
                Expect::Exactly(MoveError::LibraryReadOnly("T8".to_string()))
            }
            "read-only source" => {
                fx.mark_newer(&fx.t7);
                Expect::Exactly(MoveError::LibraryReadOnly("T7".to_string()))
            }
            "insufficient space" => {
                free = total - 1;
                Expect::Exactly(MoveError::InsufficientSpace {
                    needed: total,
                    available: total - 1,
                })
            }
            "alias at the destination" | "image alias at the destination" => {
                fs::create_dir_all(&dest).unwrap();
                Expect::Exactly(MoveError::DestinationTaken("T8".to_string()))
            }
            "link in the install" => {
                fs::write(&outside, b"outside the library").unwrap();
                std::os::unix::fs::symlink(&outside, source.dir.join("weights/link.bin")).unwrap();
                Expect::IoMentioning("link.bin")
            }
            "linked install folder" => {
                fs::rename(&source.dir, &outside).unwrap();
                std::os::unix::fs::symlink(&outside, &source.dir).unwrap();
                Expect::IoMentioning("qwen.gturbo")
            }
            _ => unreachable!(),
        };
        let before = fx.state(&source.dir);

        let (result, reports) = move_with(&fx, &source.install_id, to, free, false, &|_| {});

        match (&expected, &result) {
            (Expect::Exactly(error), _) => assert_eq!(result.as_ref(), Err(error), "{case}"),
            (Expect::IoMentioning(text), Err(MoveError::Io(reason))) => {
                assert!(reason.contains(text), "{case}: {reason}")
            }
            (_, other) => panic!("{case}: expected an IO error, got {other:?}"),
        }
        assert!(reports.is_empty(), "{case}: progress was reported");
        assert!(fx.leftovers().is_empty(), "{case}");
        for library in [&fx.t7, &fx.t8] {
            let staging = fx.root(Some(library)).join(".staging");
            assert!(
                !staging.exists(),
                "{case}: {} was created",
                staging.display()
            );
        }
        assert_eq!(fx.state(&source.dir), before, "{case}");
        match case {
            "alias at the destination" | "image alias at the destination" => {
                assert!(names(&dest).is_empty(), "{case}")
            }
            "same library" | "same library on This Mac" => {}
            _ => assert!(fs::symlink_metadata(&dest).is_err(), "{case}"),
        }
    }
}

/// Embedding and OCR stay on This Mac, and an id no install has is refused,
/// before anything is staged.
#[test]
fn home_only_and_unknown_ids_are_refused() {
    let fx = Fixture::new();
    let unknown = |id: &str| MoveError::UnknownInstall(id.to_string());
    for (id, expected) in [
        ("internal:embedding/gemma", MoveError::HomeOnly),
        ("internal:ocr/reader", MoveError::HomeOnly),
        ("internal:text/nothing", unknown("internal:text/nothing")),
        ("internal:image/nothing", unknown("internal:image/nothing")),
        ("not an id", unknown("not an id")),
    ] {
        let (result, reports) = move_with(&fx, id, Some(&fx.t7), u64::MAX, false, &|_| {});
        assert_eq!(result, Err(expected), "{id}");
        assert!(reports.is_empty(), "{id}");
        assert!(!fx.root(Some(&fx.t7)).join(".staging").exists(), "{id}");
    }
}

/// A registry write that fails after the copy was published removes the
/// copy and the destination's marker row, so the source stays registered.
#[test]
fn a_failed_registry_write_removes_the_published_copy() {
    for direction in ["This Mac to T7", "T7 to T8"] {
        let fx = Fixture::new();
        let (from, to) = fx.direction(direction);
        let source = fx.install(from, Kind::Text, "qwen");
        let total = directory_bytes(&source.dir);
        let before = fx.state(&source.dir);

        let locked = Locked::new(fx.store.root());
        let (result, reports) = move_with(&fx, &source.install_id, to, u64::MAX, false, &|_| {});
        drop(locked);

        match &result {
            Err(MoveError::Io(reason)) => {
                assert!(reason.contains("installed.json"), "{direction}: {reason}")
            }
            other => panic!("{direction}: expected an IO error, got {other:?}"),
        }
        // The copy was complete, so the failure came after it.
        assert_eq!(reports.last(), Some(&(total, total)), "{direction}");
        assert_untouched(&fx, &source, to, &before, direction);
    }
}

/// The destination's marker is written before the registry: when it cannot
/// be written, the registry was never pointed at the copy.
#[test]
fn a_destination_marker_that_cannot_be_written_leaves_the_registry_unchanged() {
    let fx = Fixture::new();
    let root = fx.root(Some(&fx.t8));
    let source = fx.install(None, Kind::Text, "qwen");
    // With the modality and staging folders already there, a locked root
    // still lets the copy publish, and only the marker write fails.
    fs::create_dir(root.join("text")).unwrap();
    fx.store.staging_parent_for(&root).unwrap();
    let before = fx.state(&source.dir);

    let locked = RefCell::new(None);
    let (result, _) = move_with(
        &fx,
        &source.install_id,
        Some(&fx.t8),
        u64::MAX,
        false,
        &|_| *locked.borrow_mut() = Some(Locked::new(&root)),
    );
    drop(locked);

    match &result {
        Err(MoveError::Io(reason)) => assert!(reason.contains("marker"), "{reason}"),
        other => panic!("expected an IO error, got {other:?}"),
    }
    assert_untouched(&fx, &source, Some(&fx.t8), &before, "marker");
}

/// A source folder that cannot be deleted after the move is reported, with
/// the registry and markers already at the destination.
#[test]
fn a_source_that_cannot_be_deleted_is_reported_after_the_move() {
    let fx = Fixture::new();
    let source = fx.install(Some(&fx.t7), Kind::Text, "qwen");
    let payload = snapshot(&source.dir);

    let locked = Locked::new(&fx.root(Some(&fx.t7)).join("text"));
    let (result, _) = move_with(&fx, &source.install_id, None, u64::MAX, false, &|_| {});
    drop(locked);

    assert_eq!(
        result,
        Ok(MoveOutcome::SourceCleanupFailed {
            path: source.dir.clone(),
            stale_marker: None,
        })
    );
    let dest = fx.destination(None, &source);
    assert_eq!(snapshot(&dest), payload);
    let rows = fx.store.read_rows();
    assert_eq!(rows.len(), 1);
    assert_eq!((rows[0].library.as_deref(), &rows[0].path), (None, &dest));
    assert!(!fx.in_marker(&fx.t7, "qwen"));
}

/// A source marker that cannot be rewritten keeps an extra row, which the
/// invariant tolerates, and the move still finishes and says so.
#[test]
fn a_source_marker_that_cannot_be_rewritten_is_reported_after_the_move() {
    let fx = Fixture::new();
    let source = fx.install(Some(&fx.t7), Kind::Audio, "voice");

    let locked = Locked::new(&fx.root(Some(&fx.t7)));
    let (result, _) = move_with(
        &fx,
        &source.install_id,
        Some(&fx.t8),
        u64::MAX,
        false,
        &|_| {},
    );
    drop(locked);

    assert_eq!(result, Ok(MoveOutcome::SourceMarkerStale("T7".to_string())));
    assert!(fs::symlink_metadata(&source.dir).is_err());
    let rows = fx.store.read_rows();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].library.as_deref(), Some(fx.t8.as_str()));
    assert!(fx.in_marker(&fx.t8, "voice"));
    assert!(fx.in_marker(&fx.t7, "voice"));
}

/// The path to the source is walked again right before it is deleted: a
/// folder on the way swapped for a link meanwhile is never deleted through
/// (2.2 SECURITY), and the move reports the source as left over.
#[test]
fn the_source_is_never_deleted_through_a_link() {
    let fx = Fixture::new();
    let source = fx.install(Some(&fx.t7), Kind::Text, "qwen");
    let payload = snapshot(&source.dir);
    let text = fx.root(Some(&fx.t7)).join("text");
    let outside = fx.base.join("outside");

    let (result, _) = move_with(&fx, &source.install_id, None, u64::MAX, false, &|_| {
        fs::rename(&text, &outside).unwrap();
        std::os::unix::fs::symlink(&outside, &text).unwrap();
    });

    assert_eq!(
        result,
        Ok(MoveOutcome::SourceCleanupFailed {
            path: source.dir.clone(),
            stale_marker: None,
        })
    );
    assert_eq!(snapshot(&outside.join("qwen.gturbo")), payload);
    assert_eq!(snapshot(&fx.destination(None, &source)), payload);
    assert!(fs::symlink_metadata(&text)
        .unwrap()
        .file_type()
        .is_symlink());
}

/// A row moves only its own install folder. One naming its whole modality
/// folder, by a hand edit on This Mac or as an untrusted marker row adopted
/// from a drive, is refused before anything is staged, and the other
/// install in that folder keeps its files and its row.
#[test]
fn a_row_naming_more_than_its_own_install_folder_is_refused() {
    for case in ["This Mac text folder", "T8 audio folder"] {
        let fx = Fixture::new();
        let (from, to, kind) = if case.starts_with("This Mac") {
            (None, Some(fx.t7.as_str()), Kind::Text)
        } else {
            (Some(fx.t8.as_str()), None, Kind::Audio)
        };
        let neighbor = fx.install(from, kind, "llama");
        let modality = kind.modality();
        let folder = neighbor.dir.parent().unwrap().to_path_buf();
        let relative = PathBuf::from(modality.as_str());
        let relative = from.map(|_| relative.as_path());
        fx.record(recorded_row(
            "qwen", modality, None, &folder, from, relative,
        ));
        let install_id = format!(
            "{}:{}/qwen",
            from.unwrap_or(INTERNAL_LIBRARY_ID),
            modality.as_str()
        );
        let before = fx.state(&folder);

        let (result, reports) = move_with(&fx, &install_id, to, u64::MAX, false, &|_| {});

        match &result {
            Err(MoveError::Io(reason)) => {
                assert!(
                    reason.contains("not the install folder"),
                    "{case}: {reason}"
                )
            }
            other => panic!("{case}: expected a refusal, got {other:?}"),
        }
        assert!(reports.is_empty(), "{case}: progress was reported");
        assert!(fx.leftovers().is_empty(), "{case}");
        for library in [&fx.t7, &fx.t8] {
            let staging = fx.root(Some(library)).join(".staging");
            assert!(!staging.exists(), "{case}: {}", staging.display());
        }
        assert_eq!(fx.state(&folder), before, "{case}");
        let dest = fx.root(to).join(modality.as_str()).join("qwen.gturbo");
        assert!(fs::symlink_metadata(&dest).is_err(), "{case}");
    }
}

/// This Mac's flat folders from before modality folders, and a vision
/// tower's own suffix, are a row's own install folder: each moves to its
/// standard place in the library.
#[test]
fn legacy_and_vision_tower_folders_move_to_their_standard_places() {
    let tower = Some(crate::EntryKind::VisionTower.as_str());
    for (folder, kind, standard) in [
        ("qwen.gturbo", None, "text/qwen.gturbo"),
        ("qwen.gturbo-vision", tower, "text/qwen.gturbo-vision"),
        ("text/qwen.gturbo-vision", tower, "text/qwen.gturbo-vision"),
    ] {
        let fx = Fixture::new();
        let dir = fx.store.models_root().join(folder);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("weights.bin"), vec![3u8; 2000]).unwrap();
        fx.record(recorded_row(
            "qwen",
            ModelModality::Text,
            kind,
            &dir,
            None,
            None,
        ));
        let payload = snapshot(&dir);

        let (result, _) = move_with(
            &fx,
            "internal:text/qwen",
            Some(&fx.t7),
            u64::MAX,
            false,
            &|_| {},
        );

        assert_eq!(result, Ok(MoveOutcome::Moved), "{folder}");
        let dest = fx.root(Some(&fx.t7)).join(standard);
        assert_eq!(snapshot(&dest), payload, "{folder}");
        assert!(fs::symlink_metadata(&dir).is_err(), "{folder}");
        let rows = fx.store.read_rows();
        assert_eq!(rows.len(), 1, "{folder}");
        assert_eq!(rows[0].path, dest, "{folder}");
        assert_eq!(rows[0].relative_path.as_deref(), Some(Path::new(standard)));
    }
}

/// A catalog audio install without its receipt cannot be checked or deleted
/// in a library (2.9), so it moves only to This Mac, where it is adopted.
/// Into a library it is refused before anything is staged, and with its
/// receipt the same install moves. Audio with no catalog profile, music
/// among it, has no receipt and moves anywhere.
#[test]
fn receipt_less_catalog_audio_moves_only_to_this_mac() {
    let catalog = AudioCatalog::embedded().unwrap();
    let alias = catalog.entries().next().unwrap().identity.alias.clone();
    assert!(catalog.get("song").is_none());
    for direction in ["This Mac to T7", "T7 to T8", "T7 to This Mac"] {
        let fx = Fixture::new();
        let (from, to) = fx.direction(direction);
        let song = fx.install(from, Kind::Audio, "song");
        fs::remove_file(song.dir.join(RECEIPT_FILENAME)).unwrap();
        let (result, _) = move_with(&fx, &song.install_id, to, u64::MAX, false, &|_| {});
        assert_eq!(result, Ok(MoveOutcome::Moved), "{direction}: song");

        let source = fx.install(from, Kind::Audio, &alias);
        fs::remove_file(source.dir.join(RECEIPT_FILENAME)).unwrap();
        let before = fx.state(&source.dir);

        let (result, reports) = move_with(&fx, &source.install_id, to, u64::MAX, false, &|_| {});

        if to.is_none() {
            assert_eq!(result, Ok(MoveOutcome::Moved), "{direction}");
            continue;
        }
        match &result {
            Err(MoveError::Io(reason)) => assert!(
                reason.contains("receipt") && reason.contains("This Mac"),
                "{direction}: {reason}"
            ),
            other => panic!("{direction}: expected a refusal, got {other:?}"),
        }
        assert!(reports.is_empty(), "{direction}: progress was reported");
        assert_untouched(&fx, &source, to, &before, direction);

        fs::write(source.dir.join(RECEIPT_FILENAME), b"{}").unwrap();
        let (result, _) = move_with(&fx, &source.install_id, to, u64::MAX, false, &|_| {});
        assert_eq!(result, Ok(MoveOutcome::Moved), "{direction}");
    }
}

/// A drive unplugged in the middle of the copy, the destination's or the
/// source's, is not connected, as an install that loses its drive is, and
/// the source stays registered and as it was.
#[test]
fn a_drive_lost_mid_copy_is_not_connected() {
    for direction in ["This Mac to T7", "T7 to This Mac"] {
        let fx = Fixture::new();
        let (from, to) = fx.direction(direction);
        let source = fx.install(from, Kind::Text, "qwen");
        let payload = snapshot(&source.dir);
        let registry = fx.state(&source.dir).registry;
        let unplugged = Cell::new(false);
        // Three of the files have bytes, so whatever order the folder lists
        // them in, another is still to be read or written after the first
        // report.
        let progress = |copied: u64, _: u64| {
            if copied > 0 && !unplugged.replace(true) {
                fx.unplug("T7");
            }
        };

        let result = plan_move(
            &fx.store,
            &source.install_id,
            to.unwrap_or(INTERNAL_LIBRARY_ID),
            &fx.env,
            u64::MAX,
        )
        .and_then(|plan| {
            run_move(
                &fx.store,
                &plan,
                &fx.env,
                &progress,
                &AtomicBool::new(false),
                &|_| {},
            )
        });

        assert!(unplugged.get(), "{direction}");
        assert_eq!(
            result,
            Err(MoveError::NotConnected("T7".to_string())),
            "{direction}"
        );
        let now = fs::read_to_string(fx.store.root().join("installed.json")).ok();
        assert_eq!(now, registry, "{direction}");
        let source_now = match from {
            Some(_) => fx.base.join("unplugged-T7/TurboSpark/text/qwen.gturbo"),
            None => source.dir.clone(),
        };
        assert_eq!(snapshot(&source_now), payload, "{direction}");
        assert!(
            fx.leftovers().is_empty(),
            "{direction}: {:?}",
            fx.leftovers()
        );
        if from.is_some() {
            let dest = fx.destination(None, &source);
            assert!(fs::symlink_metadata(&dest).is_err(), "{direction}");
        }
    }
}

/// A cancel inside a file that takes several chunks stops at the next
/// chunk, not at the end of the file.
#[test]
fn a_cancel_inside_a_file_stops_at_the_next_chunk() {
    let fx = Fixture::new();
    let dir = fx.store.image_install_path("big");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("weights.bin"), vec![5u8; 5 * CHUNK_BYTES]).unwrap();
    let source = Source {
        install_id: "internal:image/big".to_string(),
        alias: "big".to_string(),
        dir,
        modality: ModelModality::Image,
    };
    let before = fx.state(&source.dir);
    let total = 5 * CHUNK_BYTES as u64;

    let (result, reports) = move_with(
        &fx,
        &source.install_id,
        Some(&fx.t7),
        u64::MAX,
        true,
        &|_| {},
    );

    assert_eq!(result, Err(MoveError::Cancelled));
    assert_eq!(reports, vec![(0, total), (CHUNK_BYTES as u64, total)]);
    assert_untouched(&fx, &source, Some(&fx.t7), &before, "chunk");
}

/// A cancel that comes after the last chunk, before the copy is published,
/// still stops the move and removes the copy.
#[test]
fn a_cancel_after_the_copy_stops_before_publishing() {
    let fx = Fixture::new();
    let source = fx.install(None, Kind::Text, "qwen");
    let before = fx.state(&source.dir);
    let cancel = AtomicBool::new(false);

    let result =
        plan_move(&fx.store, &source.install_id, &fx.t7, &fx.env, u64::MAX).and_then(|plan| {
            run_move(&fx.store, &plan, &fx.env, &|_, _| {}, &cancel, &|_| {
                cancel.store(true, Ordering::Relaxed)
            })
        });

    assert_eq!(result, Err(MoveError::Cancelled));
    assert_untouched(&fx, &source, Some(&fx.t7), &before, "after the copy");
}

/// When the source's marker cannot be rewritten AND its folder cannot be
/// deleted, the outcome reports both.
#[test]
fn both_source_failures_after_a_move_are_reported() {
    let fx = Fixture::new();
    let source = fx.install(Some(&fx.t7), Kind::Text, "qwen");
    let root = fx.root(Some(&fx.t7));

    let text = Locked::new(&root.join("text"));
    let marker = Locked::new(&root);
    let (result, _) = move_with(&fx, &source.install_id, None, u64::MAX, false, &|_| {});
    drop(marker);
    drop(text);

    assert_eq!(
        result,
        Ok(MoveOutcome::SourceCleanupFailed {
            path: source.dir.clone(),
            stale_marker: Some("T7".to_string()),
        })
    );
    assert!(fx.in_marker(&fx.t7, "qwen"));
    let rows = fx.store.read_rows();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].path, fx.destination(None, &source));
}
