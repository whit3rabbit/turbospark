//! The audio catalog in a library (2.9), offline. The library sits on a fake
//! volume located through a fixed mount table, and the transfer is a stand-in
//! that stages fixture bytes wherever the installer asks, so each refusal is
//! checked against whether a transfer was asked for at all.

use super::*;
use crate::library::{
    move_install, parse_mount_table, system_mount_reads, LibraryMarker, LibraryRegistry,
    LocationEnv, MountSource, MoveOutcome, INTERNAL_LIBRARY_ID, LIBRARIES_FILE, MARKER_FILE,
};
use crate::{ModelModality, StoreError, VerifiedSourceFile, VerifiedSourceGroup};
use sha2::{Digest, Sha256};
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

/// A temp home and a library on the fake volume `<base>/Volumes/T7`.
struct Library {
    base: PathBuf,
    store: Store,
    /// Locates the library where its volume is mounted now.
    env: LocationEnv,
    id: String,
    /// The library root on the volume as it is mounted now.
    root: PathBuf,
}

/// A fixed mount table with the local volumes `volumes`. `home_device`
/// differs from the temp dir's device so the same-disk refusal does not
/// fire for the fake volumes.
fn env_for(base: &Path, volumes: &[&Path]) -> LocationEnv {
    let mut table = "/dev/disk3s1s1 on / (apfs, sealed, local, read-only)\n".to_string();
    for (index, volume) in volumes.iter().enumerate() {
        table.push_str(&format!(
            "/dev/disk{}s1 on {} (hfs, local, journaled)\n",
            20 + index,
            volume.display()
        ));
    }
    LocationEnv {
        mount_source: MountSource::Fixed(parse_mount_table(&table)),
        volumes_root: base.join("Volumes"),
        home_device: fs::metadata(base).unwrap().dev().wrapping_add(1),
    }
}

impl Library {
    fn new() -> Self {
        let base = std::env::temp_dir().join(format!(
            "catalog-audio-library-{}-{}",
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
        let volume = base.join("Volumes").join("T7");
        fs::create_dir_all(&volume).unwrap();
        let env = env_for(&base, &[&volume]);
        let added = LibraryRegistry::load(&home)
            .add(&volume, Some("T7"), &env)
            .unwrap();
        Self {
            root: volume.join("TurboSpark"),
            base,
            store: Store::new(&home),
            env,
            id: added.view.record.id,
        }
    }

    fn volume(&self) -> PathBuf {
        self.root.parent().unwrap().to_path_buf()
    }

    /// Takes the volume away, keeping its contents to plug back in.
    fn unplug(&self) {
        fs::rename(self.volume(), self.base.join("unplugged")).unwrap();
    }

    /// Mounts the unplugged volume again under `name`.
    fn plug_in_as(&mut self, name: &str) {
        let volume = self.base.join("Volumes").join(name);
        fs::rename(self.base.join("unplugged"), &volume).unwrap();
        self.env = env_for(&self.base, &[&volume]);
        self.root = volume.join("TurboSpark");
    }

    fn marker(&self) -> LibraryMarker {
        LibraryMarker::read(&self.root).unwrap().unwrap()
    }

    /// Rewrites the marker as a newer TurboSpark would, which makes the
    /// library read-only to this build.
    fn mark_newer(&self) {
        let path = self.root.join(MARKER_FILE);
        let mut json: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        json["format"] = serde_json::json!(2);
        fs::write(&path, serde_json::to_vec_pretty(&json).unwrap()).unwrap();
    }

    fn install_path(&self, alias: &str) -> PathBuf {
        self.root.join("audio").join(format!("{alias}.gturbo"))
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.base);
    }
}

/// A temp folder removed when the test ends, passing or not.
struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A folder made read-only (0o555) until dropped, which restores 0o755, so
/// a failing test still leaves its temp tree removable. Declared after the
/// [`Library`] it is in, so it drops first.
struct Locked(PathBuf);

impl Locked {
    fn new(dir: &Path) -> Self {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o555)).unwrap();
        Self(dir.to_path_buf())
    }
}

impl Drop for Locked {
    fn drop(&mut self) {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&self.0, fs::Permissions::from_mode(0o755));
    }
}

/// The embedded catalog with its first text-to-speech profile (no terms to
/// accept) pinned to small fixture files, and those files.
fn pinned_fixture() -> (AudioCatalog, AudioProfileIdentity, Vec<(String, Vec<u8>)>) {
    pinned_fixture_where(|p| p.identity.task == AudioTask::TextToSpeech)
}

/// [`pinned_fixture`] for the first profile `pick` accepts.
fn pinned_fixture_where(
    pick: impl Fn(&AudioProfile) -> bool,
) -> (AudioCatalog, AudioProfileIdentity, Vec<(String, Vec<u8>)>) {
    let mut catalog = AudioCatalog::embedded().unwrap();
    let profile = catalog.entries.iter_mut().find(|p| pick(p)).unwrap();
    assert!(profile.acknowledgments.is_empty());
    let mut files = Vec::new();
    for asset in &mut profile.assets {
        let bytes = format!("fixture {}", asset.path).into_bytes();
        asset.size = bytes.len() as u64;
        asset.sha256 = format!("{:x}", Sha256::digest(&bytes));
        files.push((asset.path.clone(), bytes));
    }
    profile.identity.asset_fingerprint = asset_fingerprint(
        &profile.assets,
        &profile.frontend,
        profile.install_subdir.as_deref(),
    )
    .unwrap();
    let identity = profile.identity.clone();
    (catalog, identity, files)
}

/// What a pinned transfer leaves: the fixture files staged in a fresh
/// transfer folder under `parent`, and the receipt for them.
fn stage(
    catalog: &AudioCatalog,
    identity: &AudioProfileIdentity,
    files: &[(String, Vec<u8>)],
    parent: &Path,
) -> DownloadReceipt {
    stage_plan(&catalog.pinned_plan(identity).unwrap(), files, parent)
}

/// [`stage`] for the transfer of `plan`, whose one source is the files.
fn stage_plan(
    plan: &PinnedArtifactPlan,
    files: &[(String, Vec<u8>)],
    parent: &Path,
) -> DownloadReceipt {
    let staging_root = parent.join(format!(
        ".turbospark-pinned-transfer-test-{}",
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let source_root = staging_root.join("source-0000");
    let mut verified = Vec::new();
    for (path, bytes) in files {
        let staged_path = source_root.join(path);
        fs::create_dir_all(staged_path.parent().unwrap()).unwrap();
        fs::write(&staged_path, bytes).unwrap();
        verified.push(VerifiedSourceFile {
            path: path.clone(),
            staged_path,
            size: bytes.len() as u64,
            sha256: format!("{:x}", Sha256::digest(bytes)),
        });
    }
    DownloadReceipt {
        owner_id: plan.owner_id.clone(),
        staging_root,
        download_cache: Vec::new(),
        sources: vec![VerifiedSourceGroup {
            role: plan.sources[0].role.clone(),
            repo: plan.sources[0].repo.clone(),
            files: verified,
        }],
    }
}

struct Run {
    result: Result<InstalledModel, StoreError>,
    /// `(staging_parent, cache_parent)` of each transfer asked for.
    transfers: Vec<(PathBuf, PathBuf)>,
}

/// An install with the stand-in transfer, which records where it was asked
/// to stage and cache.
fn install(
    fx: &Library,
    catalog: &AudioCatalog,
    identity: &AudioProfileIdentity,
    files: &[(String, Vec<u8>)],
    library: Option<&str>,
) -> Run {
    let mut transfers = Vec::new();
    let result = catalog.install_with(
        &fx.store,
        identity,
        &[],
        library,
        &fx.env,
        &CancelFlag::new(),
        |plan, staging_parent, cache_parent| {
            assert_eq!(plan, &catalog.pinned_plan(identity).unwrap());
            transfers.push((staging_parent.to_path_buf(), cache_parent.to_path_buf()));
            Ok(stage(catalog, identity, files, staging_parent))
        },
    );
    Run { result, transfers }
}

/// Every file under `dir` with its bytes, so a test can tell it was left
/// exactly as it was.
fn snapshot(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    fn walk(root: &Path, at: &Path, found: &mut Vec<(PathBuf, Vec<u8>)>) {
        for entry in fs::read_dir(at).unwrap().flatten() {
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

#[test]
fn an_install_lands_in_the_library_and_lists_resolves_verifies_and_deletes_there() {
    let fx = Library::new();
    let (catalog, identity, files) = pinned_fixture();
    let run = install(&fx, &catalog, &identity, &files, Some(&fx.id));
    let installed = run.result.unwrap();

    // Staged on the library's drive, so publication is a same-volume
    // rename, with the range cache in the home.
    assert_eq!(
        run.transfers,
        vec![(fx.root.join(".staging"), fx.store.root().to_path_buf())]
    );
    let path = fx.install_path(&identity.alias);
    let relative = PathBuf::from(format!("audio/{}.gturbo", identity.alias));
    assert_eq!(installed.path, path);
    assert_eq!(installed.library.as_deref(), Some(fx.id.as_str()));
    assert_eq!(installed.relative_path.as_deref(), Some(relative.as_path()));
    assert_eq!(installed.modality, ModelModality::Audio);
    assert!(path.join(RECEIPT_FILENAME).is_file());
    assert_eq!(fx.store.audio_records(), vec![installed.clone()]);
    let marker = fx.marker();
    assert_eq!(marker.models.len(), 1);
    assert_eq!(marker.models[0].alias, identity.alias);
    assert_eq!(marker.models[0].relative_path, relative);
    // The transfer folder is gone, and nothing landed in the home's models.
    assert!(names(&fx.root.join(".staging")).is_empty());
    assert!(!fx.store.models_root().exists());

    let report = catalog.installed_report_with(&fx.store, &fx.env);
    assert_eq!(report.installed, vec![installed.clone()]);
    assert!(report.not_connected.is_empty());
    assert!(report.incompatible.is_empty(), "{:?}", report.incompatible);
    assert!(report.needs_adoption.is_empty());
    assert!(report.other_audio.is_empty());

    assert_eq!(
        catalog.resolve_with(&fx.store, &identity, &fx.env).unwrap(),
        path
    );
    assert_eq!(
        catalog.verify_with(&fx.store, &identity, &fx.env).unwrap(),
        path
    );
    let mut reports = 0;
    let observed = catalog
        .verify_observed_with(&fx.store, &identity, &fx.env, &|| false, &mut |_, _, _| {
            reports += 1
        })
        .unwrap();
    assert_eq!(observed, path);
    assert!(reports > 0);

    catalog.delete_with(&fx.store, &identity, &fx.env).unwrap();
    assert!(!path.exists());
    // No tombstone is left beside it, and the row left the registry and
    // the marker together.
    assert!(names(&fx.root.join("audio")).is_empty());
    assert!(fx.store.audio_records().is_empty());
    assert!(fx.marker().models.is_empty());
    assert!(catalog.resolve_with(&fx.store, &identity, &fx.env).is_err());
}

/// The completion check of 2.9: unplugged, the install is listed as not
/// connected with no error, and plugged back in under another name it
/// resolves and deletes at the new mount point, not its recorded path.
#[test]
fn an_unplugged_library_lists_as_not_connected_and_resolves_at_its_new_mount() {
    let mut fx = Library::new();
    let (catalog, identity, files) = pinned_fixture();
    let installed = install(&fx, &catalog, &identity, &files, Some(&fx.id))
        .result
        .unwrap();
    let payload = snapshot(&installed.path);
    fx.unplug();

    let report = catalog.installed_report_with(&fx.store, &fx.env);
    assert!(report.incompatible.is_empty(), "{:?}", report.incompatible);
    assert!(report.installed.is_empty());
    assert_eq!(
        report.not_connected,
        vec![AudioNotConnectedInstall {
            record: installed.clone(),
            identity: Some(identity.clone()),
            library_name: "T7".to_string(),
        }]
    );

    let not_connected = StoreError::NotConnected {
        library: "T7".to_string(),
    }
    .to_string();
    let resolved = catalog.resolve_with(&fx.store, &identity, &fx.env);
    assert_eq!(resolved, Err(not_connected.clone()));
    let verified = catalog.verify_with(&fx.store, &identity, &fx.env);
    assert_eq!(verified, Err(not_connected.clone()));
    let deleted = catalog.delete_with(&fx.store, &identity, &fx.env);
    assert_eq!(deleted, Err(not_connected));
    assert_eq!(fx.store.audio_records(), vec![installed.clone()]);
    assert!(!fx.volume().exists(), "the unplugged volume was recreated");

    fx.plug_in_as("T7 Renamed");
    let moved = fx.install_path(&identity.alias);
    assert_ne!(moved, installed.path);
    // The refused delete left every file as it was.
    assert_eq!(snapshot(&moved), payload);
    assert_eq!(
        catalog.resolve_with(&fx.store, &identity, &fx.env).unwrap(),
        moved
    );
    assert_eq!(
        catalog.verify_with(&fx.store, &identity, &fx.env).unwrap(),
        moved
    );
    let report = catalog.installed_report_with(&fx.store, &fx.env);
    assert!(report.not_connected.is_empty());
    assert!(report.incompatible.is_empty(), "{:?}", report.incompatible);
    assert_eq!(report.installed.len(), 1);
    assert_eq!(report.installed[0].path, moved);

    catalog.delete_with(&fx.store, &identity, &fx.env).unwrap();
    assert!(!moved.exists());
    assert!(fx.store.audio_records().is_empty());
    assert!(fx.marker().models.is_empty());
}

/// A folder missing from a connected library is a fault, reported the way
/// a missing This Mac folder is, and not "not connected".
#[test]
fn a_folder_missing_from_a_connected_library_is_incompatible() {
    let fx = Library::new();
    let (catalog, identity, files) = pinned_fixture();
    let installed = install(&fx, &catalog, &identity, &files, Some(&fx.id))
        .result
        .unwrap();
    fs::remove_dir_all(&installed.path).unwrap();

    let report = catalog.installed_report_with(&fx.store, &fx.env);
    assert!(report.not_connected.is_empty());
    assert!(report.installed.is_empty());
    assert_eq!(report.incompatible.len(), 1);
    assert_eq!(report.incompatible[0].record.alias, identity.alias);
    assert!(
        report.incompatible[0]
            .error
            .contains("reading audio directory"),
        "{}",
        report.incompatible[0].error
    );
}

/// A drive anyone could have written may link the library's `audio` folder
/// or an install folder elsewhere. Listing, resolving and deleting refuse
/// the link, and nothing is read or deleted through it (2.2 SECURITY).
#[test]
fn linked_folders_in_a_library_are_refused_and_never_followed() {
    for linked in ["audio folder", "install folder"] {
        let fx = Library::new();
        let (catalog, identity, files) = pinned_fixture();
        let installed = install(&fx, &catalog, &identity, &files, Some(&fx.id))
            .result
            .unwrap();
        let victim = if linked == "audio folder" {
            fx.root.join("audio")
        } else {
            installed.path.clone()
        };
        let outside = fx.base.join("outside");
        fs::rename(&victim, &outside).unwrap();
        std::os::unix::fs::symlink(&outside, &victim).unwrap();
        let before = snapshot(&outside);

        let report = catalog.installed_report_with(&fx.store, &fx.env);
        assert!(report.installed.is_empty(), "{linked}");
        assert!(report.not_connected.is_empty(), "{linked}");
        assert_eq!(report.incompatible.len(), 1, "{linked}");
        let error = &report.incompatible[0].error;
        assert!(error.contains("must not be a symlink"), "{linked}: {error}");
        assert!(
            catalog.resolve_with(&fx.store, &identity, &fx.env).is_err(),
            "{linked}"
        );
        assert!(
            catalog.verify_with(&fx.store, &identity, &fx.env).is_err(),
            "{linked}"
        );
        assert!(
            catalog.delete_with(&fx.store, &identity, &fx.env).is_err(),
            "{linked}"
        );
        assert_eq!(snapshot(&outside), before, "{linked}");
        assert!(fs::symlink_metadata(&victim)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(fx.store.audio_records(), vec![installed], "{linked}");
    }
}

/// Every library refusal comes before the transfer is asked for: an
/// unplugged library, a read-only one, and an alias held elsewhere in
/// either direction. Nothing is written.
#[test]
fn library_installs_are_refused_before_any_transfer() {
    for case in ["unplugged", "read-only", "on This Mac", "in the library"] {
        let fx = Library::new();
        let (catalog, identity, files) = pinned_fixture();
        let destination = match case {
            "on This Mac" | "in the library" => {
                let held = if case == "on This Mac" {
                    None
                } else {
                    Some(fx.id.as_str())
                };
                install(&fx, &catalog, &identity, &files, held)
                    .result
                    .unwrap();
                if case == "on This Mac" {
                    Some(fx.id.as_str())
                } else {
                    None
                }
            }
            _ => Some(fx.id.as_str()),
        };
        match case {
            "unplugged" => fx.unplug(),
            "read-only" => fx.mark_newer(),
            _ => {}
        }
        let rows = fx.store.audio_records();
        let home_entries = names(fx.store.root());

        let run = install(&fx, &catalog, &identity, &files, destination);
        let expected = match case {
            "unplugged" => StoreError::NotConnected {
                library: "T7".to_string(),
            },
            "read-only" => StoreError::LibraryReadOnly {
                library: "T7".to_string(),
            },
            _ => {
                let held_by = if case == "on This Mac" {
                    "This Mac"
                } else {
                    "the library \"T7\""
                };
                match &run.result {
                    Err(held @ StoreError::AliasElsewhere { place, .. })
                        if place.contains(held_by) =>
                    {
                        held.clone()
                    }
                    other => panic!("{case}: expected a refusal naming {held_by}, got {other:?}"),
                }
            }
        };
        assert_eq!(run.result, Err(expected), "{case}");
        assert!(
            run.transfers.is_empty(),
            "{case}: the transfer was asked for"
        );
        assert_eq!(fx.store.audio_records(), rows, "{case}");
        assert_eq!(names(fx.store.root()), home_entries, "{case}");
        if case != "in the library" {
            assert!(!fx.root.join("audio").exists(), "{case}");
            assert!(!fx.root.join(".staging").exists(), "{case}");
        }
    }
}

/// The public installer refuses an unplugged library the same way, with no
/// progress callback, so no request was made.
#[test]
fn the_installer_refuses_an_unplugged_library_with_no_progress() {
    let fx = Library::new();
    let (catalog, identity, _) = pinned_fixture();
    fx.unplug();
    let mut progress = 0;
    let result = catalog.install_acknowledged_into(
        &Client::with_timeout(std::time::Duration::from_secs(1)),
        &Catalog::embedded().unwrap(),
        &fx.store,
        &identity,
        &[],
        Some(&fx.id),
        &fx.env,
        &mut |_| progress += 1,
        &CancelFlag::new(),
    );
    assert_eq!(
        result,
        Err(StoreError::NotConnected {
            library: "T7".to_string()
        })
    );
    assert_eq!(progress, 0);
    assert!(!fx.store.root().join(".download-cache").exists());
}

/// `None` is This Mac and today's behavior: the same path and staging, a
/// row with no library fields, the marker untouched, and the calls that
/// predate libraries resolving and deleting it.
#[test]
fn this_mac_installs_resolve_list_and_delete_as_before() {
    let fx = Library::new();
    let (catalog, identity, files) = pinned_fixture();
    let run = install(&fx, &catalog, &identity, &files, None);
    let installed = run.result.unwrap();
    let home = fx.store.root().to_path_buf();
    assert_eq!(run.transfers, vec![(home.clone(), home.clone())]);
    assert_eq!(installed.path, fx.store.audio_install_path(&identity.alias));
    assert_eq!(installed.library, None);
    assert_eq!(installed.relative_path, None);
    assert_eq!(fx.store.audio_records(), vec![installed.clone()]);
    assert!(fx.marker().models.is_empty());
    assert!(!fx.root.join(".staging").exists());
    assert!(!names(&home)
        .iter()
        .any(|name| name.starts_with(".turbospark-pinned-transfer")));

    assert_eq!(
        catalog.resolve(&fx.store, &identity).unwrap(),
        installed.path
    );
    assert_eq!(
        catalog.installed_report(&fx.store).installed,
        vec![installed.clone()]
    );
    assert_eq!(
        catalog.installed_report_with(&fx.store, &fx.env),
        catalog.installed_report(&fx.store)
    );
    catalog.delete(&fx.store, &identity).unwrap();
    assert!(!installed.path.exists());
    assert!(fx.store.audio_records().is_empty());
}

/// A report locates every library through one mount table read, so N rows
/// on N unplugged drives spawn `/sbin/mount` once. Lives here because only
/// unit tests can count the spawns.
#[test]
fn the_report_reads_the_mount_table_once_for_any_number_of_offline_libraries() {
    let base = std::env::temp_dir().join(format!(
        "catalog-audio-one-table-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&base);
    let _cleanup = TempDir(base.clone());
    let home = base.join("home");
    let volumes_root = base.join("Volumes");
    // A volume folder, so locating an offline library has a volume to scan
    // and needs the table. The live table has no entry for it.
    fs::create_dir_all(volumes_root.join("Other")).unwrap();
    fs::create_dir_all(&home).unwrap();
    let ids = [
        "0f6c2a3e-1b7d-4c55-9e0a-7d2f1c3b4a59",
        "7a1e9c04-52b3-4f8d-a6e1-3c9b0d2f5e17",
        "c3d9e2f1-6a4b-4e8c-9f7d-1b2a3c4d5e6f",
    ];
    let records: Vec<serde_json::Value> = ids
        .iter()
        .enumerate()
        .map(|(index, id)| {
            serde_json::json!({
                "id": id,
                "name": format!("Gone{index}"),
                "volume_relative_path": "TurboSpark",
                "last_mount_path": volumes_root.join(format!("Gone{index}")),
                "default_for_downloads": false,
                "added_on": "2026-10-14"
            })
        })
        .collect();
    let list = serde_json::json!({
        "format": 1,
        "internal_is_default": true,
        "libraries": records,
    });
    fs::write(
        home.join(LIBRARIES_FILE),
        serde_json::to_vec(&list).unwrap(),
    )
    .unwrap();
    let rows: Vec<InstalledModel> = ids
        .iter()
        .enumerate()
        .map(|(index, id)| InstalledModel {
            alias: format!("voice{index}"),
            repo: "owner/name".to_string(),
            revision: "main".to_string(),
            path: volumes_root.join(format!("Gone{index}/TurboSpark/audio/voice{index}.gturbo")),
            family: "kokoro".to_string(),
            install_bytes: 1,
            installed_on: "2026-10-14".to_string(),
            status: "unverified".to_string(),
            kind: None,
            modality: ModelModality::Audio,
            variant: None,
            library: Some(id.to_string()),
            relative_path: Some(PathBuf::from(format!("audio/voice{index}.gturbo"))),
        })
        .collect();
    fs::write(
        home.join("installed.json"),
        serde_json::to_vec(&rows).unwrap(),
    )
    .unwrap();
    let live = LocationEnv {
        mount_source: MountSource::System,
        volumes_root,
        home_device: 0,
    };

    let reads = system_mount_reads();
    let report = AudioCatalog::embedded()
        .unwrap()
        .installed_report_with(&Store::new(&home), &live);
    assert_eq!(system_mount_reads(), reads + 1);
    assert!(report.incompatible.is_empty(), "{:?}", report.incompatible);
    let names: Vec<_> = report
        .not_connected
        .iter()
        .map(|entry| entry.library_name.as_str())
        .collect();
    assert_eq!(names, ["Gone0", "Gone1", "Gone2"]);
}

/// A forget that fails half way puts everything back. With the library root
/// read-only, `forget_install` writes the registry and then fails to rewrite
/// the marker, so the delete must rename the folder back from its tombstone
/// and record the registry row again.
#[test]
fn a_delete_whose_marker_cannot_be_rewritten_restores_the_folder_and_the_row() {
    let fx = Library::new();
    let (catalog, identity, files) = pinned_fixture();
    let installed = install(&fx, &catalog, &identity, &files, Some(&fx.id))
        .result
        .unwrap();
    let payload = snapshot(&installed.path);
    let marker = fx.marker();

    let locked = Locked::new(&fx.root);
    let deleted = catalog.delete_with(&fx.store, &identity, &fx.env);
    drop(locked);

    let error = deleted.unwrap_err();
    assert!(error.contains("still lists it"), "{error}");
    assert_eq!(snapshot(&installed.path), payload);
    let alias_folder = format!("{}.gturbo", identity.alias);
    assert_eq!(names(&fx.root.join("audio")), [alias_folder]);
    assert_eq!(fx.store.audio_records(), vec![installed]);
    assert_eq!(fx.marker(), marker);
}

/// A read-only library refuses a delete up front. The `audio` folder is made
/// read-only too, so a delete that went ahead would fail with a different
/// error rather than succeed.
#[test]
fn a_delete_in_a_read_only_library_is_refused_and_touches_nothing() {
    let fx = Library::new();
    let (catalog, identity, files) = pinned_fixture();
    let installed = install(&fx, &catalog, &identity, &files, Some(&fx.id))
        .result
        .unwrap();
    fx.mark_newer();
    let payload = snapshot(&installed.path);
    let marker = fs::read(fx.root.join(MARKER_FILE)).unwrap();

    let locked = Locked::new(&fx.root.join("audio"));
    let deleted = catalog.delete_with(&fx.store, &identity, &fx.env);
    drop(locked);

    let read_only = StoreError::LibraryReadOnly {
        library: "T7".to_string(),
    };
    assert_eq!(deleted, Err(read_only.to_string()));
    assert_eq!(snapshot(&installed.path), payload);
    assert_eq!(fx.store.audio_records(), vec![installed]);
    assert_eq!(fs::read(fx.root.join(MARKER_FILE)).unwrap(), marker);
}

/// A library row that records another folder than its managed one is not
/// trusted: it is listed as incompatible, and neither resolved nor deleted.
#[test]
fn a_library_row_naming_another_folder_is_refused() {
    let fx = Library::new();
    let (catalog, identity, files) = pinned_fixture();
    let installed = install(&fx, &catalog, &identity, &files, Some(&fx.id))
        .result
        .unwrap();
    let payload = snapshot(&installed.path);
    let mut row = installed.clone();
    row.relative_path = Some(PathBuf::from("audio/other.gturbo"));
    fx.store.record_rows(std::slice::from_ref(&row)).unwrap();

    let expected = "does not name its library's managed path";
    let report = catalog.installed_report_with(&fx.store, &fx.env);
    assert!(report.installed.is_empty());
    assert!(report.not_connected.is_empty());
    assert_eq!(report.incompatible.len(), 1);
    let error = &report.incompatible[0].error;
    assert!(error.contains(expected), "{error}");
    let resolved = catalog
        .resolve_with(&fx.store, &identity, &fx.env)
        .unwrap_err();
    assert!(resolved.contains(expected), "{resolved}");
    let deleted = catalog
        .delete_with(&fx.store, &identity, &fx.env)
        .unwrap_err();
    assert!(deleted.contains(expected), "{deleted}");
    assert_eq!(snapshot(&installed.path), payload);
    assert_eq!(fx.store.audio_records(), vec![row]);
}

// MiniMax Music 3 through the music installer (2.8), whose 4-bit profile is
// also an audio catalog profile.

const MUSIC_ALIAS: &str = "minimax-music3-4bit";

/// The embedded catalog with the MiniMax Music 3 4-bit profile pinned to
/// small fixture files, and those files.
fn music_fixture() -> (AudioCatalog, AudioProfileIdentity, Vec<(String, Vec<u8>)>) {
    pinned_fixture_where(|p| p.identity.alias == MUSIC_ALIAS)
}

/// A music install with `catalog` as the audio catalog and a stand-in
/// transfer that stages `files` for the music plan it is asked for, after
/// `tamper` had its way with the staged tree.
fn install_music(
    fx: &Library,
    catalog: &AudioCatalog,
    files: &[(String, Vec<u8>)],
    library: Option<&str>,
    tamper: impl FnOnce(&Path),
) -> Result<InstalledModel, StoreError> {
    crate::music::install_music_with(
        &fx.store,
        MUSIC_ALIAS,
        library,
        &fx.env,
        || Ok(catalog.clone()),
        &CancelFlag::new(),
        |plan, staging_parent, _| {
            let staged = stage_plan(plan, files, staging_parent);
            tamper(&staged.staging_root.join("source-0000"));
            Ok(staged)
        },
    )
}

/// The receipt adoption writes for `identity`'s profile in `catalog`.
fn adoption_receipt(catalog: &AudioCatalog, identity: &AudioProfileIdentity) -> Vec<u8> {
    serde_json::to_vec_pretty(&new_audio_receipt(catalog.profile(identity).unwrap())).unwrap()
}

/// Only This Mac adopts an install without its receipt, so a catalog
/// profile installed into a library by the music installer carries the
/// receipt adoption would write, from the moment it is published. The
/// library then lists, resolves, verifies and moves it like any audio
/// catalog install, rather than listing it as incompatible for good.
#[test]
fn a_library_music_install_carries_its_audio_receipt_and_moves() {
    let mut fx = Library::new();
    let (catalog, identity, files) = music_fixture();
    let installed = install_music(&fx, &catalog, &files, Some(&fx.id), |_| {}).unwrap();
    let path = fx.install_path(MUSIC_ALIAS);
    assert_eq!(installed.path, path);
    assert_eq!(installed.library.as_deref(), Some(fx.id.as_str()));
    assert_eq!(
        fs::read(path.join(RECEIPT_FILENAME)).unwrap(),
        adoption_receipt(&catalog, &identity)
    );
    assert!(names(&fx.root.join(".staging")).is_empty());

    let report = catalog.installed_report_with(&fx.store, &fx.env);
    assert!(report.incompatible.is_empty(), "{:?}", report.incompatible);
    assert!(report.needs_adoption.is_empty());
    let listed: Vec<_> = report.installed.iter().map(|row| &row.alias).collect();
    assert_eq!(listed, [MUSIC_ALIAS]);
    assert_eq!(
        catalog.resolve_with(&fx.store, &identity, &fx.env).unwrap(),
        path
    );
    assert_eq!(
        catalog.verify_with(&fx.store, &identity, &fx.env).unwrap(),
        path
    );

    // Into another library, to This Mac, and into a library again. A move
    // into a library refuses catalog audio without its receipt (2.10).
    let (t7, t8) = (fx.volume(), fx.base.join("Volumes").join("T8"));
    fs::create_dir_all(&t8).unwrap();
    fx.env = env_for(&fx.base, &[&t7, &t8]);
    let other = LibraryRegistry::load(fx.store.root())
        .add(&t8, Some("T8"), &fx.env)
        .unwrap()
        .view
        .record
        .id;
    let mut install_id = installed.install_id();
    for to in [other.as_str(), INTERNAL_LIBRARY_ID, fx.id.as_str()] {
        let moved = move_install(
            &fx.store,
            &install_id,
            to,
            &fx.env,
            u64::MAX,
            &|_, _| {},
            &AtomicBool::new(false),
        );
        assert_eq!(moved, Ok(MoveOutcome::Moved), "to {to}");
        let rows = fx.store.audio_records();
        assert_eq!(rows.len(), 1, "to {to}");
        install_id = rows[0].install_id();
        let report = catalog.installed_report_with(&fx.store, &fx.env);
        assert_eq!(report.installed.len(), 1, "to {to}: {report:?}");
    }
    assert_eq!(
        catalog.resolve_with(&fx.store, &identity, &fx.env).unwrap(),
        path
    );
}

/// A This Mac music install (`turbospark-model pull-audio`) publishes the
/// receipt adoption would write, like a library install, so the app lists it
/// as installed rather than asking for adoption or reporting it incompatible.
/// Both audio catalog music profiles, the Turbo one included: the CLI once
/// recorded Turbo as `affine-4bit-turbo-8step`, which the app's profile
/// match refused, so the app showed the install as needing attention.
#[test]
fn a_this_mac_music_install_carries_its_audio_receipt() {
    for alias in [MUSIC_ALIAS, "minimax-music3-turbo-4bit-8step"] {
        let fx = Library::new();
        let (catalog, identity, files) = pinned_fixture_where(|p| p.identity.alias == alias);
        let installed = crate::music::install_music_with(
            &fx.store,
            alias,
            None,
            &fx.env,
            || Ok(catalog.clone()),
            &CancelFlag::new(),
            |plan, staging_parent, _| Ok(stage_plan(plan, &files, staging_parent)),
        )
        .unwrap();
        assert_eq!(installed.path, fx.store.audio_install_path(alias));
        assert_eq!(installed.variant.as_deref(), Some("affine-4bit"), "{alias}");
        assert_eq!(
            fs::read(installed.path.join(RECEIPT_FILENAME)).unwrap(),
            adoption_receipt(&catalog, &identity),
            "{alias}"
        );

        let report = catalog.installed_report_with(&fx.store, &fx.env);
        assert!(
            report.incompatible.is_empty(),
            "{alias}: {:?}",
            report.incompatible
        );
        assert!(report.needs_adoption.is_empty(), "{alias}");
        let listed: Vec<_> = report.installed.iter().map(|row| &row.alias).collect();
        assert_eq!(listed, [alias]);
    }
}

/// A row an earlier CLI recorded for Turbo, `affine-4bit-turbo-8step` and no
/// receipt, is still offered for adoption rather than reported incompatible.
#[test]
fn an_earlier_cli_turbo_record_is_offered_for_adoption() {
    let alias = "minimax-music3-turbo-4bit-8step";
    let fx = Library::new();
    let (catalog, identity, files) = pinned_fixture_where(|p| p.identity.alias == alias);
    let installed = crate::music::install_music_with(
        &fx.store,
        alias,
        None,
        &fx.env,
        || Ok(catalog.clone()),
        &CancelFlag::new(),
        |plan, staging_parent, _| Ok(stage_plan(plan, &files, staging_parent)),
    )
    .unwrap();
    fs::remove_file(installed.path.join(RECEIPT_FILENAME)).unwrap();
    let mut earlier = installed.clone();
    earlier.variant = Some("affine-4bit-turbo-8step".into());
    fx.store.record(&earlier).unwrap();

    let report = catalog.installed_report_with(&fx.store, &fx.env);
    assert!(report.incompatible.is_empty(), "{:?}", report.incompatible);
    assert_eq!(report.needs_adoption.len(), 1);
    assert_eq!(report.needs_adoption[0].identity, identity);
    catalog
        .adopt_legacy(&fx.store, &identity, &CancelFlag::new())
        .unwrap();
    assert_eq!(
        fs::read(installed.path.join(RECEIPT_FILENAME)).unwrap(),
        adoption_receipt(&catalog, &identity)
    );
}

/// The receipt vouches for the bytes the transfer hashed, and a resolve
/// trusts it after checking sizes alone. So a staged tree that is not
/// exactly the profile's pinned assets, a digest or a file off, is not
/// published at all, rather than published under a receipt for other bytes.
#[test]
fn a_library_music_install_that_is_not_its_profile_is_refused() {
    for case in ["another digest", "an extra file"] {
        let fx = Library::new();
        let (mut catalog, _, files) = music_fixture();
        if case == "another digest" {
            let profile = catalog
                .entries
                .iter_mut()
                .find(|p| p.identity.alias == MUSIC_ALIAS)
                .unwrap();
            profile.assets[0].sha256 = "0".repeat(64);
        }
        let result = install_music(&fx, &catalog, &files, Some(&fx.id), |source| {
            if case == "an extra file" {
                fs::write(source.join("extra.json"), b"{}").unwrap();
            }
        });
        match &result {
            Err(StoreError::Io(message)) => {
                assert!(message.contains("audio profile"), "{case}: {message}")
            }
            other => panic!("{case}: expected a refusal, got {other:?}"),
        }
        assert!(!fx.install_path(MUSIC_ALIAS).exists(), "{case}");
        assert!(names(&fx.root.join(".staging")).is_empty(), "{case}");
        assert!(fx.store.audio_records().is_empty(), "{case}");
        assert!(fx.marker().models.is_empty(), "{case}");
    }
}

/// The receipt helper on its own: a transfer that omits a pinned file, lists
/// one twice (so another went unhashed), or staged a file elsewhere gets no
/// receipt, and neither does a profile with terms to accept, which adoption
/// refuses too. An alias with no audio profile gets none and no error.
#[test]
fn a_transfer_receipt_needs_each_pinned_file_once_where_it_was_staged() {
    let base = std::env::temp_dir().join(format!(
        "catalog-audio-transfer-receipt-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _cleanup = TempDir(base.clone());
    let (catalog, identity, files) = music_fixture();
    let plan = crate::music::MusicCatalog::embedded()
        .unwrap()
        .pinned_plan(MUSIC_ALIAS)
        .unwrap();
    let staged = stage_plan(&plan, &files, &base);
    let source = staged.staging_root.join("source-0000");
    let verified = &staged.sources[0].files;
    let receipt = source.join(RECEIPT_FILENAME);

    assert_eq!(
        catalog.write_transfer_receipt("minimax-music3-mxfp8", verified, &source),
        Ok(false)
    );
    let mut twice = verified.clone();
    twice[1] = twice[0].clone();
    let mut elsewhere = verified.clone();
    elsewhere[0].staged_path = base.join(&elsewhere[0].path);
    let cases = [
        ("omitted", verified[1..].to_vec()),
        ("twice", twice),
        ("elsewhere", elsewhere),
    ];
    for (case, listed) in cases {
        let refused = catalog.write_transfer_receipt(MUSIC_ALIAS, &listed, &source);
        assert!(refused.is_err(), "{case}");
        assert!(!receipt.exists(), "{case}");
    }
    let mut with_terms = catalog.clone();
    with_terms
        .entries
        .iter_mut()
        .find(|p| p.identity.alias == MUSIC_ALIAS)
        .unwrap()
        .acknowledgments
        .push(AudioAcknowledgment {
            id: "fixture_terms".to_string(),
            summary: "Fixture terms".to_string(),
        });
    let refused = with_terms.write_transfer_receipt(MUSIC_ALIAS, verified, &source);
    assert!(refused.is_err_and(|error| error.contains("terms")));
    assert!(!receipt.exists());
    assert_eq!(
        catalog.write_transfer_receipt(MUSIC_ALIAS, verified, &source),
        Ok(true)
    );
    assert_eq!(
        fs::read(&receipt).unwrap(),
        adoption_receipt(&catalog, &identity)
    );
}
