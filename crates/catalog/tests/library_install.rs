//! Installs routed to a destination library (2.1, 2.6), offline: a local
//! fixture server stands in for the Hub, and the library sits on a fake
//! volume located through a fixed mount table.
//!
//! The load-bearing assertions are the refusals that must happen before a
//! single byte moves: a library that is not connected, one a newer
//! TurboSpark wrote, and an alias another library already holds. Each is
//! checked against both the transfer callback and the fixture server's
//! request log, so a refusal that came after the transfer started fails.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use turbospark_catalog::library::{
    parse_mount_table, Availability, LibraryMarker, LibraryRegistry, LocationEnv, MountSource,
    MARKER_FILE,
};
use turbospark_catalog::{
    embedded_music_entry, install_destination, install_music_into, install_text_into,
    install_with_byte_progress, record_into, set_hf_endpoint_override, AudioCatalog, CancelFlag,
    Catalog, Client, InstallKind, InstallPlan, InstalledModel, ModelModality, RepoRef, SourceKind,
    Store, StoreError,
};

const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
const MUSIC_ALIAS: &str = "minimax-music3-mxfp8";
static ENDPOINT_LOCK: Mutex<()> = Mutex::new(());

/// One repository the fixture server answers for, at one revision.
struct FixtureRepo {
    id: String,
    revision: String,
    files: BTreeMap<String, Vec<u8>>,
}

/// Answers Hub metadata and file requests by path, so the order a transfer
/// asks in does not matter, and logs every request it saw.
struct FixtureServer {
    endpoint: String,
    requests: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl FixtureServer {
    fn start(repos: Vec<FixtureRepo>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("fixture bind");
        listener
            .set_nonblocking(true)
            .expect("set nonblocking fixture listener");
        let address = listener.local_addr().expect("fixture address");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let thread_requests = Arc::clone(&requests);
        let thread_stop = Arc::clone(&stop);
        let thread = thread::spawn(move || {
            while !thread_stop.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream
                            .set_nonblocking(false)
                            .expect("accepted fixture stream is blocking");
                        stream
                            .set_read_timeout(Some(Duration::from_secs(3)))
                            .expect("set fixture stream timeout");
                        let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
                        let mut request_line = String::new();
                        if reader.read_line(&mut request_line).is_err() {
                            continue;
                        }
                        let mut range = None;
                        loop {
                            let mut line = String::new();
                            if reader.read_line(&mut line).is_err()
                                || line == "\r\n"
                                || line.is_empty()
                            {
                                break;
                            }
                            if let Some((name, value)) = line.trim_end().split_once(':') {
                                if name.eq_ignore_ascii_case("range") {
                                    range = Some(value.trim().to_string());
                                }
                            }
                        }
                        let mut parts = request_line.split_whitespace();
                        let method = parts.next().unwrap_or("?").to_string();
                        let target = parts.next().unwrap_or("/").to_string();
                        thread_requests
                            .lock()
                            .expect("requests mutex")
                            .push(format!("{method} {target}"));
                        let wire = answer(&method, &target, range.as_deref(), &repos);
                        let _ = stream.write_all(&wire);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(error) => panic!("accept fixture request: {error}"),
                }
            }
        });
        Self {
            endpoint: format!("http://{address}"),
            requests,
            stop,
            thread: Some(thread),
        }
    }

    fn requests(&self) -> Vec<String> {
        self.requests.lock().expect("requests mutex").clone()
    }
}

impl Drop for FixtureServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn answer(method: &str, target: &str, range: Option<&str>, repos: &[FixtureRepo]) -> Vec<u8> {
    let path = target.split('?').next().unwrap_or(target);
    // Hub repository ids are case-insensitive, and the pinned transfer asks
    // with the lowercased id.
    let lowered = path.to_ascii_lowercase();
    for repo in repos {
        let id = repo.id.to_ascii_lowercase();
        if lowered.starts_with(&format!("/api/models/{id}/")) {
            let siblings: Vec<_> = repo
                .files
                .iter()
                .map(|(name, bytes)| {
                    serde_json::json!({
                        "rfilename": name,
                        "size": bytes.len() as u64,
                        "blobId": "a".repeat(40),
                    })
                })
                .collect();
            let body = serde_json::json!({
                "id": repo.id,
                "sha": repo.revision,
                "siblings": siblings,
            })
            .to_string();
            return response(200, "OK", body.as_bytes(), &[]);
        }
        let prefix = format!("/{id}/resolve/{}/", repo.revision);
        if !lowered.starts_with(&prefix) {
            continue;
        }
        let name = &path[prefix.len()..];
        let Some(bytes) = repo.files.get(name) else {
            return response(404, "Not Found", b"", &[]);
        };
        if method.eq_ignore_ascii_case("HEAD") {
            return response(
                200,
                "OK",
                b"",
                &[("x-linked-size", bytes.len().to_string())],
            );
        }
        if let Some(range) = range {
            let (start, end) = range
                .strip_prefix("bytes=")
                .and_then(|range| range.split_once('-'))
                .expect("range endpoints");
            let start: usize = start.parse().expect("range start");
            let end: usize = end
                .parse::<usize>()
                .expect("range end")
                .min(bytes.len() - 1);
            return response(
                206,
                "Partial Content",
                &bytes[start..=end],
                &[(
                    "Content-Range",
                    format!("bytes {start}-{end}/{}", bytes.len()),
                )],
            );
        }
        return response(200, "OK", bytes, &[]);
    }
    response(404, "Not Found", b"", &[])
}

fn response(status: u16, reason: &str, body: &[u8], headers: &[(&str, String)]) -> Vec<u8> {
    let mut wire = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (name, value) in headers {
        wire.push_str(&format!("{name}: {value}\r\n"));
    }
    wire.push_str("\r\n");
    let mut wire = wire.into_bytes();
    wire.extend_from_slice(body);
    wire
}

struct EndpointReset;

impl Drop for EndpointReset {
    fn drop(&mut self) {
        set_hf_endpoint_override(None);
    }
}

fn client_for(server: &FixtureServer) -> (EndpointReset, Client) {
    set_hf_endpoint_override(Some(server.endpoint.clone()));
    (EndpointReset, Client::with_timeout(Duration::from_secs(3)))
}

/// A temp home plus the fake volume `<base>/Volumes/T7` with a library added
/// on it, through a fixed mount table. `home_device` differs from the temp
/// dir's device so the same-disk refusal does not fire for the fake volume.
struct LibraryFixture {
    base: PathBuf,
    store: Store,
    volume: PathBuf,
    env: LocationEnv,
    id: String,
}

impl LibraryFixture {
    fn new(tag: &str) -> Self {
        let base = std::env::temp_dir().join(format!(
            "turbospark-library-install-{tag}-{}",
            std::process::id()
        ));
        std::fs::remove_dir_all(&base).ok();
        std::fs::create_dir_all(&base).unwrap();
        // Canonical, because the temp dir is behind the /var symlink and the
        // location rules compare canonical paths.
        let base = base.canonicalize().unwrap();
        let home = base.join("home");
        std::fs::create_dir_all(&home).unwrap();
        let volumes_root = base.join("Volumes");
        let volume = volumes_root.join("T7");
        std::fs::create_dir_all(&volume).unwrap();
        let table = format!(
            "/dev/disk3s1s1 on / (apfs, sealed, local, read-only)\n\
             /dev/disk20s1 on {} (hfs, local, journaled)\n",
            volume.display()
        );
        let env = LocationEnv {
            mount_source: MountSource::Fixed(parse_mount_table(&table)),
            volumes_root,
            home_device: std::fs::metadata(&base).unwrap().dev().wrapping_add(1),
        };
        let added = LibraryRegistry::load(&home)
            .add(&volume, Some("T7"), &env)
            .unwrap();
        Self {
            base,
            store: Store::new(&home),
            volume,
            env,
            id: added.view.record.id,
        }
    }

    fn library_root(&self) -> PathBuf {
        self.volume.join("TurboSpark")
    }

    fn marker(&self) -> LibraryMarker {
        LibraryMarker::read(&self.library_root()).unwrap().unwrap()
    }

    fn rows_on_disk(&self) -> Vec<InstalledModel> {
        std::fs::read(self.store.root().join("installed.json"))
            .map(|bytes| serde_json::from_slice(&bytes).unwrap())
            .unwrap_or_default()
    }

    fn unplug(&self) {
        std::fs::remove_dir_all(&self.volume).unwrap();
    }

    /// Rewrites the marker as a newer TurboSpark would, which makes the
    /// library read-only to this build.
    fn mark_newer(&self) {
        let path = self.library_root().join(MARKER_FILE);
        let mut json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        json["format"] = serde_json::json!(2);
        std::fs::write(&path, serde_json::to_vec_pretty(&json).unwrap()).unwrap();
    }

    /// Everything under `dir` relative to it, sorted, `None` when absent.
    fn listing(dir: &Path) -> Option<Vec<PathBuf>> {
        fn walk(root: &Path, at: &Path, found: &mut Vec<PathBuf>) {
            for entry in std::fs::read_dir(at).unwrap().flatten() {
                let path = entry.path();
                found.push(path.strip_prefix(root).unwrap().to_path_buf());
                if entry.file_type().unwrap().is_dir() {
                    walk(root, &path, found);
                }
            }
        }
        if !dir.exists() {
            return None;
        }
        let mut found = Vec::new();
        walk(dir, dir, &mut found);
        found.sort();
        Some(found)
    }
}

impl Drop for LibraryFixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.base).ok();
    }
}

fn audio_row(alias: &str, path: &Path, library: Option<&str>) -> InstalledModel {
    InstalledModel {
        alias: alias.to_string(),
        repo: "owner/name".to_string(),
        revision: "main".to_string(),
        path: path.to_path_buf(),
        family: "minimax_music3".to_string(),
        install_bytes: 1,
        installed_on: "2026-10-15".to_string(),
        status: "unverified".to_string(),
        kind: None,
        modality: ModelModality::Audio,
        variant: None,
        library: library.map(str::to_string),
        relative_path: None,
    }
}

// Text and vision towers.

/// A synthetic GGUF and the ChatML tokenizer sidecars under `owner/model`.
fn text_repo() -> (FixtureRepo, InstallPlan) {
    let (gguf, _) = repack::build_synthetic_gemma4_gguf(repack::SyntheticGgufShape::default());
    let tokenizer_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../tokenizer/tests/fixtures/ChatMLTokenizer");
    let repo = RepoRef::new("owner/model", REVISION);
    let plan = InstallPlan {
        alias: "qwen".to_string(),
        weights: repo.clone(),
        file: Some("model-Q8_0.gguf".to_string()),
        kind: SourceKind::Gguf,
        sidecars: repo,
        sidecar_files: vec![
            "tokenizer.json".to_string(),
            "tokenizer_config.json".to_string(),
        ],
        include_vision: false,
        vision_from_bytes: false,
        install_bytes: gguf.len() as u64,
        status: "unlisted".to_string(),
        mtp: None,
        reuse_trunk_from: None,
        vision_only: false,
        vision_file: None,
    };
    let files = BTreeMap::from([
        ("model-Q8_0.gguf".to_string(), gguf),
        (
            "tokenizer.json".to_string(),
            std::fs::read(tokenizer_dir.join("tokenizer.json")).unwrap(),
        ),
        (
            "tokenizer_config.json".to_string(),
            std::fs::read(tokenizer_dir.join("tokenizer_config.json")).unwrap(),
        ),
    ]);
    (
        FixtureRepo {
            id: "owner/model".to_string(),
            revision: REVISION.to_string(),
            files,
        },
        plan,
    )
}

/// The text install the CLI and the native interface run: the destination
/// first, then the walk into it, then the record in its library.
#[test]
fn a_text_install_lands_in_the_library_and_is_recorded_there() {
    let _lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fx = LibraryFixture::new("text");
    let (repo, plan) = text_repo();
    let server = FixtureServer::start(vec![repo]);
    let (_reset, client) = client_for(&server);

    let target =
        install_destination(&fx.store, Some(&fx.id), InstallKind::Text, "qwen", &fx.env).unwrap();
    let root = fx.library_root();
    assert_eq!(target.root, root);
    assert_eq!(target.dir, root.join("text").join("qwen.gturbo"));
    assert_eq!(target.relative_path, PathBuf::from("text/qwen.gturbo"));
    assert_eq!(target.library_id.as_deref(), Some(fx.id.as_str()));
    assert_eq!(target.library_name, "T7");

    let bytes = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&bytes);
    let installed = install_with_byte_progress(
        &plan,
        &target.dir,
        &client,
        |_| {},
        Some(Arc::new(move |delta| {
            counted.fetch_add(delta as usize, Ordering::Relaxed);
        })),
        None,
    )
    .expect("offline text install");
    assert!(bytes.load(Ordering::Relaxed) > 0);
    let recorded = record_into(&fx.store, &installed.model, &target, &fx.env).unwrap();

    assert!(target.dir.join("model_weights.bin").is_file());
    assert_eq!(recorded.library.as_deref(), Some(fx.id.as_str()));
    assert_eq!(
        recorded.relative_path.as_deref(),
        Some(Path::new("text/qwen.gturbo"))
    );
    assert_eq!(fx.rows_on_disk(), vec![recorded.clone()]);
    let marker = fx.marker();
    assert_eq!(marker.models.len(), 1);
    assert_eq!(marker.models[0].alias, "qwen");
    assert_eq!(
        marker.models[0].relative_path,
        PathBuf::from("text/qwen.gturbo")
    );
    let listed = fx.store.installed_with_availability(&fx.env);
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].library_name, "T7");
    assert_eq!(listed[0].availability, Availability::Available);
    // Nothing of a library install lands in the home's models folder.
    assert_eq!(LibraryFixture::listing(&fx.store.models_root()), None);
}

/// [`install_text_into`], the walk-and-record step the CLI and the native
/// interface share, lands a text install in its library and records it
/// there, in the registry and the marker alike.
#[test]
fn install_text_into_walks_into_the_library_and_records_it() {
    let _lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fx = LibraryFixture::new("text-into");
    let (repo, plan) = text_repo();
    let server = FixtureServer::start(vec![repo]);
    let (_reset, client) = client_for(&server);

    let target =
        install_destination(&fx.store, Some(&fx.id), InstallKind::Text, "qwen", &fx.env).unwrap();
    let installed = install_text_into(
        &plan,
        &target,
        &client,
        |_| {},
        None,
        None,
        &fx.store,
        &fx.env,
    )
    .expect("offline text install");

    assert!(target.dir.join("model_weights.bin").is_file());
    assert_eq!(installed.model.path, target.dir);
    assert_eq!(installed.model.library.as_deref(), Some(fx.id.as_str()));
    assert_eq!(
        installed.model.relative_path.as_deref(),
        Some(Path::new("text/qwen.gturbo"))
    );
    assert_eq!(fx.rows_on_disk(), vec![installed.model.clone()]);
    let marker = fx.marker();
    assert_eq!(marker.models.len(), 1);
    assert_eq!(marker.models[0].alias, "qwen");
}

/// Into This Mac the helper is today's install: the same folder, and a row
/// with no library fields.
#[test]
fn install_text_into_this_mac_installs_as_before() {
    let _lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fx = LibraryFixture::new("text-into-mac");
    let (repo, plan) = text_repo();
    let server = FixtureServer::start(vec![repo]);
    let (_reset, client) = client_for(&server);

    let target = install_destination(&fx.store, None, InstallKind::Text, "qwen", &fx.env).unwrap();
    let installed = install_text_into(
        &plan,
        &target,
        &client,
        |_| {},
        None,
        None,
        &fx.store,
        &fx.env,
    )
    .expect("offline text install");

    assert_eq!(target.dir, fx.store.install_path("qwen"));
    assert!(target.dir.join("model_weights.bin").is_file());
    assert_eq!(installed.model.library, None);
    assert_eq!(installed.model.relative_path, None);
    assert_eq!(fx.rows_on_disk(), vec![installed.model.clone()]);
    assert!(fx.marker().models.is_empty());
}

/// A drive unplugged after the destination was settled ends the install as
/// a disconnect before any request, and the walk never rebuilds the
/// library's folders on the disk under the mount point.
#[test]
fn install_text_into_an_unplugged_library_is_a_disconnect_before_any_request() {
    let _lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fx = LibraryFixture::new("text-into-unplugged");
    let (repo, plan) = text_repo();
    let server = FixtureServer::start(vec![repo]);
    let (_reset, client) = client_for(&server);

    let target =
        install_destination(&fx.store, Some(&fx.id), InstallKind::Text, "qwen", &fx.env).unwrap();
    fx.unplug();
    let error = install_text_into(
        &plan,
        &target,
        &client,
        |_| {},
        None,
        None,
        &fx.store,
        &fx.env,
    )
    .unwrap_err();

    assert_eq!(
        error,
        StoreError::LibraryDisconnected {
            library: "T7".to_string()
        }
    );
    assert!(server.requests().is_empty(), "{:?}", server.requests());
    assert!(!fx.volume.exists());
    assert!(fx.rows_on_disk().is_empty());
}

#[test]
fn this_mac_destinations_are_todays_paths() {
    let fx = LibraryFixture::new("this-mac");
    for (kind, path) in [
        (InstallKind::Text, fx.store.install_path("qwen")),
        (
            InstallKind::VisionTower,
            fx.store.vision_install_path("qwen"),
        ),
        (InstallKind::Image, fx.store.image_install_path("qwen")),
        (InstallKind::Audio, fx.store.audio_install_path("qwen")),
    ] {
        let target = install_destination(&fx.store, None, kind, "qwen", &fx.env).unwrap();
        assert_eq!(target.dir, path, "{kind:?}");
        assert_eq!(target.root, fx.store.models_root());
        assert_eq!(target.library_id, None);
        assert_eq!(target.library_name, "This Mac");
    }
    let vision = install_destination(
        &fx.store,
        Some(&fx.id),
        InstallKind::VisionTower,
        "qwen",
        &fx.env,
    )
    .unwrap();
    assert_eq!(
        vision.dir,
        fx.library_root().join("text").join("qwen.gturbo-vision")
    );
    assert_eq!(
        vision.relative_path,
        PathBuf::from("text/qwen.gturbo-vision")
    );
}

/// A vision tower recorded into a library keeps its kind and its
/// `.gturbo-vision` folder in the text namespace, in the registry row and
/// the marker row alike.
#[test]
fn a_vision_tower_is_recorded_into_the_library_with_its_kind() {
    let fx = LibraryFixture::new("record-vision");
    let target = install_destination(
        &fx.store,
        Some(&fx.id),
        InstallKind::VisionTower,
        "tower",
        &fx.env,
    )
    .unwrap();
    std::fs::create_dir_all(&target.dir).unwrap();
    let mut tower = audio_row("tower", &target.dir, None);
    tower.modality = ModelModality::Text;
    tower.kind = Some("vision-tower".to_string());

    let recorded = record_into(&fx.store, &tower, &target, &fx.env).unwrap();
    let relative = PathBuf::from("text/tower.gturbo-vision");
    assert_eq!(recorded.library.as_deref(), Some(fx.id.as_str()));
    assert_eq!(recorded.relative_path.as_deref(), Some(relative.as_path()));
    assert_eq!(recorded.kind.as_deref(), Some("vision-tower"));
    assert_eq!(fx.rows_on_disk(), vec![recorded]);
    let marker = fx.marker();
    assert_eq!(marker.models.len(), 1);
    assert_eq!(marker.models[0].alias, "tower");
    assert_eq!(marker.models[0].kind.as_deref(), Some("vision-tower"));
    assert_eq!(marker.models[0].modality, ModelModality::Text);
    assert_eq!(marker.models[0].relative_path, relative);
}

/// A This Mac install records exactly the row it recorded before libraries:
/// no library and no relative path.
#[test]
fn recording_into_this_mac_keeps_the_internal_row_shape() {
    let fx = LibraryFixture::new("record-this-mac");
    let target =
        install_destination(&fx.store, None, InstallKind::Audio, "voice", &fx.env).unwrap();
    std::fs::create_dir_all(&target.dir).unwrap();
    let row = audio_row("voice", &target.dir, None);
    let recorded = record_into(&fx.store, &row, &target, &fx.env).unwrap();
    assert_eq!(recorded, row);
    assert_eq!(fx.rows_on_disk(), vec![row]);
    assert!(fx.marker().models.is_empty());
}

/// An install whose folder is not the target's, as when a link was swapped
/// in under the library during the walk, is never recorded there.
#[test]
fn an_install_outside_its_library_target_is_not_recorded() {
    let fx = LibraryFixture::new("record-elsewhere");
    let target =
        install_destination(&fx.store, Some(&fx.id), InstallKind::Text, "qwen", &fx.env).unwrap();
    let elsewhere = fx.base.join("elsewhere").join("qwen.gturbo");
    std::fs::create_dir_all(&elsewhere).unwrap();
    let mut row = audio_row("qwen", &elsewhere, None);
    row.modality = ModelModality::Text;
    assert!(matches!(
        record_into(&fx.store, &row, &target, &fx.env),
        Err(StoreError::Io(_))
    ));
    assert!(fx.rows_on_disk().is_empty());
    assert!(fx.marker().models.is_empty());
}

/// A modality folder a drive's owner replaced with a link would put the
/// walk's writes outside the library, so the destination is refused.
#[test]
fn a_linked_modality_folder_in_the_library_is_refused() {
    let fx = LibraryFixture::new("linked-modality");
    let elsewhere = fx.base.join("elsewhere");
    std::fs::create_dir_all(&elsewhere).unwrap();
    std::os::unix::fs::symlink(&elsewhere, fx.library_root().join("text")).unwrap();
    assert!(matches!(
        install_destination(&fx.store, Some(&fx.id), InstallKind::Text, "qwen", &fx.env),
        Err(StoreError::Io(_))
    ));
}

#[test]
fn a_text_install_into_an_unplugged_or_read_only_library_is_refused_up_front() {
    let fx = LibraryFixture::new("text-refused");
    fx.mark_newer();
    assert_eq!(
        install_destination(&fx.store, Some(&fx.id), InstallKind::Text, "qwen", &fx.env),
        Err(StoreError::LibraryReadOnly {
            library: "T7".to_string()
        })
    );
    fx.unplug();
    assert_eq!(
        install_destination(&fx.store, Some(&fx.id), InstallKind::Text, "qwen", &fx.env),
        Err(StoreError::NotConnected {
            library: "T7".to_string()
        })
    );
}

/// The 1.2 decision: alias plus modality stays unique across libraries. A
/// row in another library refuses the install (offer a move instead) rather
/// than letting the record replace it and orphan its folder. The same
/// library, and another modality, are fine.
#[test]
fn an_alias_recorded_in_another_library_is_refused() {
    let fx = LibraryFixture::new("alias-elsewhere");
    let here = fx.store.install_path("qwen");
    std::fs::create_dir_all(&here).unwrap();
    let mut row = audio_row("qwen", &here, None);
    row.modality = ModelModality::Text;
    fx.store.record(&row).unwrap();

    let refused = install_destination(&fx.store, Some(&fx.id), InstallKind::Text, "qwen", &fx.env);
    let Err(StoreError::Io(message)) = refused else {
        panic!("expected a refusal, got {refused:?}");
    };
    assert!(message.contains("This Mac"), "{message}");
    // Reinstalling into This Mac, and another modality, are allowed.
    install_destination(&fx.store, None, InstallKind::Text, "qwen", &fx.env).unwrap();
    install_destination(&fx.store, Some(&fx.id), InstallKind::Audio, "qwen", &fx.env).unwrap();

    // And the other way round: a row in T7 refuses This Mac.
    let target = install_destination(
        &fx.store,
        Some(&fx.id),
        InstallKind::Audio,
        "voice",
        &fx.env,
    )
    .unwrap();
    std::fs::create_dir_all(&target.dir).unwrap();
    record_into(
        &fx.store,
        &audio_row("voice", &target.dir, None),
        &target,
        &fx.env,
    )
    .unwrap();
    let refused = install_destination(&fx.store, None, InstallKind::Audio, "voice", &fx.env);
    let Err(StoreError::Io(message)) = refused else {
        panic!("expected a refusal, got {refused:?}");
    };
    assert!(message.contains("T7"), "{message}");
    install_destination(
        &fx.store,
        Some(&fx.id),
        InstallKind::Audio,
        "voice",
        &fx.env,
    )
    .unwrap();
    // Unplugged, the row still refuses: it is recorded, not looked for.
    fx.unplug();
    assert!(matches!(
        install_destination(&fx.store, None, InstallKind::Audio, "voice", &fx.env),
        Err(StoreError::Io(_))
    ));
}

/// Image installs are not recorded, so their folders are what says an alias
/// is taken: This Mac's, and every connected library's.
#[test]
fn an_image_folder_in_another_library_is_refused() {
    let fx = LibraryFixture::new("image-elsewhere");
    std::fs::create_dir_all(fx.store.image_install_path("z-image")).unwrap();
    let refused = install_destination(
        &fx.store,
        Some(&fx.id),
        InstallKind::Image,
        "z-image",
        &fx.env,
    );
    assert!(matches!(refused, Err(StoreError::Io(ref message)) if message.contains("This Mac")));

    std::fs::create_dir_all(fx.library_root().join("image").join("flux.gturbo")).unwrap();
    let refused = install_destination(&fx.store, None, InstallKind::Image, "flux", &fx.env);
    assert!(matches!(refused, Err(StoreError::Io(ref message)) if message.contains("T7")));
    // The library's own folder is the installer's "already installed" case.
    install_destination(&fx.store, Some(&fx.id), InstallKind::Image, "flux", &fx.env).unwrap();
}

// Music, through the pinned transfer.

/// The pinned files of the MiniMax Music 3 MXFP8 profile, as small fixtures.
fn music_repo() -> FixtureRepo {
    music_repo_of(MUSIC_ALIAS)
}

/// [`music_repo`] for the profile `alias`.
fn music_repo_of(alias: &str) -> FixtureRepo {
    let entry = embedded_music_entry(alias).unwrap();
    let files = entry
        .required_files
        .iter()
        .map(|path| (path.clone(), format!("fixture {path}").into_bytes()))
        .collect();
    FixtureRepo {
        id: entry.model_id,
        revision: entry.revision,
        files,
    }
}

struct MusicRun {
    result: Result<InstalledModel, StoreError>,
    progress_calls: usize,
}

fn install_music(fx: &LibraryFixture, client: &Client, library: Option<&str>) -> MusicRun {
    let mut progress_calls = 0;
    let result = install_music_into(
        client,
        &Catalog::embedded().unwrap(),
        &fx.store,
        MUSIC_ALIAS,
        library,
        &fx.env,
        &mut |_| progress_calls += 1,
        &CancelFlag::new(),
    );
    MusicRun {
        result,
        progress_calls,
    }
}

/// Transfer staging folders a pinned transfer leaves under `parent`.
fn transfer_stages(parent: &Path) -> Vec<String> {
    std::fs::read_dir(parent)
        .map(|entries| {
            entries
                .flatten()
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .filter(|name| name.starts_with(".turbospark-pinned-transfer-"))
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn music_installs_into_the_library_and_keeps_its_cache_in_the_home() {
    let _lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fx = LibraryFixture::new("music");
    let server = FixtureServer::start(vec![music_repo()]);
    let (_reset, client) = client_for(&server);

    // Observed while the transfer runs: afterwards the stage is gone
    // either way.
    let root = fx.library_root();
    let (mut on_library, mut in_home) = (false, false);
    let installed = install_music_into(
        &client,
        &Catalog::embedded().unwrap(),
        &fx.store,
        MUSIC_ALIAS,
        Some(&fx.id),
        &fx.env,
        &mut |_| {
            on_library |= !transfer_stages(&root.join(".staging")).is_empty();
            in_home |= !transfer_stages(fx.store.root()).is_empty();
        },
        &CancelFlag::new(),
    )
    .expect("offline music install");
    assert!(on_library, "the transfer did not stage on the library");
    assert!(!in_home, "the transfer staged in the home");

    let destination = root.join("audio").join(format!("{MUSIC_ALIAS}.gturbo"));
    assert_eq!(installed.path, destination);
    assert!(destination.join("config.json").is_file());
    assert_eq!(installed.library.as_deref(), Some(fx.id.as_str()));
    let relative = PathBuf::from(format!("audio/{MUSIC_ALIAS}.gturbo"));
    assert_eq!(installed.relative_path.as_deref(), Some(relative.as_path()));
    assert_eq!(fx.rows_on_disk(), vec![installed.clone()]);
    let marker = fx.marker();
    assert_eq!(marker.models.len(), 1);
    assert_eq!(marker.models[0].alias, MUSIC_ALIAS);
    assert_eq!(marker.models[0].modality, ModelModality::Audio);
    assert_eq!(marker.models[0].relative_path, relative);

    // Staged on the library's drive and cleaned up there; nothing staged in
    // the home and nothing under its models folder.
    assert!(transfer_stages(&root.join(".staging")).is_empty());
    assert!(transfer_stages(fx.store.root()).is_empty());
    assert_eq!(LibraryFixture::listing(&fx.store.models_root()), None);
    // The pinned range cache lives in the home, never on the library.
    assert!(fx
        .store
        .root()
        .join(".download-cache")
        .join("pinned")
        .is_dir());
    assert!(!root.join(".staging").join(".download-cache").exists());
    assert!(!root.join(".download-cache").exists());

    // MXFP8 is no audio catalog profile, so it carries no audio receipt and
    // is listed as other audio, never as incompatible.
    let audio = AudioCatalog::embedded().unwrap();
    assert!(audio.get(MUSIC_ALIAS).is_none());
    assert!(!destination.join("audio-receipt.json").exists());
    let report = audio.installed_report_with(&fx.store, &fx.env);
    assert!(report.incompatible.is_empty(), "{:?}", report.incompatible);
    let listed: Vec<_> = report.other_audio.iter().map(|row| &row.alias).collect();
    assert_eq!(listed, [MUSIC_ALIAS]);
}

/// The 4-bit profile is also an audio catalog profile, so into a library it
/// is published with that profile's receipt, which vouches for the pinned
/// sizes and digests. These fixture bytes are not the pinned ones (the real
/// shards are gigabytes), so the install is refused with nothing published,
/// instead of carrying a receipt for bytes it does not hold.
#[test]
fn a_library_music_install_that_is_not_its_audio_profile_is_refused() {
    let _lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let alias = "minimax-music3-4bit";
    assert!(AudioCatalog::embedded().unwrap().get(alias).is_some());
    let fx = LibraryFixture::new("music-profile");
    let server = FixtureServer::start(vec![music_repo_of(alias)]);
    let (_reset, client) = client_for(&server);

    let result = install_music_into(
        &client,
        &Catalog::embedded().unwrap(),
        &fx.store,
        alias,
        Some(&fx.id),
        &fx.env,
        &mut |_| {},
        &CancelFlag::new(),
    );
    match &result {
        Err(StoreError::Io(message)) => assert!(message.contains("audio profile"), "{message}"),
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert!(!server.requests().is_empty(), "the transfer never ran");
    let root = fx.library_root();
    assert!(!root.join("audio").join(format!("{alias}.gturbo")).exists());
    assert!(transfer_stages(&root.join(".staging")).is_empty());
    assert!(fx.rows_on_disk().is_empty());
    assert!(fx.marker().models.is_empty());
}

/// `None` is This Mac and today's behavior: the same path, staging in the
/// home, and a row with no library fields.
#[test]
fn music_into_this_mac_installs_as_before() {
    let _lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fx = LibraryFixture::new("music-this-mac");
    let server = FixtureServer::start(vec![music_repo()]);
    let (_reset, client) = client_for(&server);

    let installed = install_music(&fx, &client, None).result.unwrap();
    assert_eq!(installed.path, fx.store.audio_install_path(MUSIC_ALIAS));
    assert!(installed.path.join("config.json").is_file());
    assert_eq!(installed.library, None);
    assert_eq!(installed.relative_path, None);
    assert_eq!(fx.rows_on_disk(), vec![installed]);
    assert!(fx.marker().models.is_empty());
    assert!(transfer_stages(fx.store.root()).is_empty());
    assert!(!fx.library_root().join(".staging").exists());
}

/// A refused music install transfers nothing: no progress callback, no
/// request to the Hub, and no file written anywhere.
fn assert_refused_before_any_transfer(
    fx: &LibraryFixture,
    server: &FixtureServer,
    run: MusicRun,
    expected: StoreError,
) {
    assert_eq!(run.result, Err(expected));
    assert_eq!(run.progress_calls, 0, "the transfer callback fired");
    assert!(
        server.requests().is_empty(),
        "the Hub was asked: {:?}",
        server.requests()
    );
    assert!(transfer_stages(fx.store.root()).is_empty());
    assert!(!fx.store.root().join(".download-cache").exists());
}

#[test]
fn music_into_an_unplugged_library_fails_before_any_transfer() {
    let _lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fx = LibraryFixture::new("music-unplugged");
    let server = FixtureServer::start(vec![music_repo()]);
    let (_reset, client) = client_for(&server);
    fx.unplug();

    let run = install_music(&fx, &client, Some(&fx.id));
    assert_refused_before_any_transfer(
        &fx,
        &server,
        run,
        StoreError::NotConnected {
            library: "T7".to_string(),
        },
    );
    assert!(!fx.volume.exists(), "the unplugged volume was recreated");
    assert!(fx.rows_on_disk().is_empty());
}

#[test]
fn music_into_a_read_only_library_fails_before_any_transfer() {
    let _lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fx = LibraryFixture::new("music-read-only");
    let server = FixtureServer::start(vec![music_repo()]);
    let (_reset, client) = client_for(&server);
    fx.mark_newer();

    let run = install_music(&fx, &client, Some(&fx.id));
    assert_refused_before_any_transfer(
        &fx,
        &server,
        run,
        StoreError::LibraryReadOnly {
            library: "T7".to_string(),
        },
    );
    assert!(!fx.library_root().join("audio").exists());
}

#[test]
fn music_already_in_another_library_fails_before_any_transfer() {
    let _lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fx = LibraryFixture::new("music-elsewhere");
    let server = FixtureServer::start(vec![music_repo()]);
    let (_reset, client) = client_for(&server);
    let here = fx.store.audio_install_path(MUSIC_ALIAS);
    std::fs::create_dir_all(&here).unwrap();
    fx.store
        .record(&audio_row(MUSIC_ALIAS, &here, None))
        .unwrap();
    let rows = fx.rows_on_disk();

    let run = install_music(&fx, &client, Some(&fx.id));
    let message = match &run.result {
        Err(StoreError::Io(message)) => message.clone(),
        other => panic!("expected a refusal, got {other:?}"),
    };
    assert!(message.contains("This Mac"), "{message}");
    assert_refused_before_any_transfer(&fx, &server, run, StoreError::Io(message));
    assert_eq!(fx.rows_on_disk(), rows);
    assert!(!fx.library_root().join("audio").exists());
}

// Resolving installs in a library (10.7).

/// Records a text and an audio install in the library, with their folders,
/// and returns their folder paths under the library root.
fn record_resolvable_installs(fx: &LibraryFixture) -> (PathBuf, PathBuf) {
    let text = fx.library_root().join("text").join("qq-text.gturbo");
    let audio = fx.library_root().join("audio").join("qq-song.gturbo");
    for (path, alias, relative, modality) in [
        (&text, "qq-text", "text/qq-text.gturbo", ModelModality::Text),
        (
            &audio,
            "qq-song",
            "audio/qq-song.gturbo",
            ModelModality::Audio,
        ),
    ] {
        std::fs::create_dir_all(path).unwrap();
        let mut row = audio_row(alias, path, Some(&fx.id));
        row.modality = modality;
        row.relative_path = Some(PathBuf::from(relative));
        fx.store.record(&row).unwrap();
    }
    (text, audio)
}

/// An install in a connected library resolves to where the library is now,
/// whatever path was recorded: a drive that mounted under another name
/// leaves the recorded path pointing nowhere.
#[test]
fn a_library_install_resolves_to_where_its_library_is_now() {
    let fx = LibraryFixture::new("resolve-current");
    let (text, audio) = record_resolvable_installs(&fx);
    // The recorded paths name a mount point that is gone.
    let mut rows: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(fx.store.root().join("installed.json")).unwrap())
            .unwrap();
    for row in &mut rows {
        let stale = format!(
            "/Volumes/T7 stale/TurboSpark/{}",
            row["relative_path"].as_str().unwrap()
        );
        row["path"] = serde_json::json!(stale);
    }
    std::fs::write(
        fx.store.root().join("installed.json"),
        serde_json::to_vec(&rows).unwrap(),
    )
    .unwrap();

    assert_eq!(fx.store.resolve_text_with("qq-text", &fx.env), Some(text));
    assert_eq!(fx.store.resolve_audio_with("qq-song", &fx.env), Some(audio));
    // Neither modality answers to the other's alias.
    assert_eq!(fx.store.resolve_text_with("qq-song", &fx.env), None);
    assert_eq!(fx.store.resolve_audio_with("qq-text", &fx.env), None);
}

/// A library that is not connected resolves to nothing, and the message
/// names the drive, for a text and an audio install alike; an alias nobody
/// has and an install that is connected get no message.
#[test]
fn an_install_on_an_unplugged_library_is_not_resolved_and_names_its_drive() {
    let fx = LibraryFixture::new("resolve-unplugged");
    record_resolvable_installs(&fx);
    let sentence = |alias: &str| {
        format!("Model {alias} is on T7, which is not connected. Connect the drive and try again.")
    };
    for (alias, modality) in [
        ("qq-text", ModelModality::Text),
        ("qq-song", ModelModality::Audio),
    ] {
        assert_eq!(
            fx.store.not_connected_message(alias, modality, &fx.env),
            None,
            "{alias} is connected"
        );
    }

    fx.unplug();

    assert_eq!(fx.store.resolve_text_with("qq-text", &fx.env), None);
    assert_eq!(fx.store.resolve_audio_with("qq-song", &fx.env), None);
    for (alias, modality) in [
        ("qq-text", ModelModality::Text),
        ("qq-song", ModelModality::Audio),
    ] {
        assert_eq!(
            fx.store
                .not_connected_message(alias, modality, &fx.env)
                .as_deref(),
            Some(sentence(alias).as_str())
        );
    }
    // The wrong modality, and an alias no row has, are not this library's.
    assert_eq!(
        fx.store
            .not_connected_message("qq-song", ModelModality::Text, &fx.env),
        None
    );
    assert_eq!(
        fx.store
            .not_connected_message("nobody", ModelModality::Text, &fx.env),
        None
    );
}
