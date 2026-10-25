//! Moving installs between libraries through the C ABI (4.4): `ts_model_move`
//! and `ts_model_move_cancel`, driven through the same function bodies the
//! header exposes, against a temp home, fake volumes, and small fixture
//! installs. `catalog::library::move_install` decides every transfer rule and
//! is tested in the catalog; what is tested here is the native interface
//! around it: the progress callback, the cancel, the running-install registry,
//! and the stable code of every refusal.
//!
//! The location rules read the mount table and the home folder's device.
//! Debug builds take both from the `TURBOSPARK_TEST_*` overrides, set here in
//! process, so a temp folder stands in for `/Volumes` and a fixture file for
//! `/sbin/mount`. Release builds ignore the overrides by design, so this file
//! is debug-only. The store root and the environment are process-wide, so
//! every test holds one lock for its whole run.
#![cfg(debug_assertions)]

use std::ffi::{CStr, CString};
use std::fs;
use std::os::raw::{c_char, c_int, c_void};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use catalog::library::{LibraryMarker, LocationEnv, MARKER_FILE};
use catalog::{InstalledModel, ModelModality, Store};
use serde_json::{json, Value};
use turbospark_ffi::{
    abi, ts_audio_catalog_json, ts_audio_install_with_options, ts_catalog_json,
    ts_embedding_install, ts_image_install, ts_install, ts_install_cancel,
    ts_install_repo_with_options, ts_installed_json, ts_last_error, ts_libraries_json,
    ts_library_add, ts_library_remove, ts_library_set_default, ts_model_forget, ts_model_move,
    ts_model_move_cancel, ts_store_import_moved_root, ts_store_relocate, ts_store_root_set,
    ts_string_free, TS_INSTALL_BYTES,
};

/// More free space than any fixture needs.
const PLENTY: i64 = 1 << 40;

/// The store root and the environment are process-wide.
static SERIAL: Mutex<()> = Mutex::new(());

// ------------------------------------------------------------------ fixture

/// A library the fixture added, on the volume `name`.
struct Lib {
    id: String,
    name: &'static str,
}

/// What a fixture install is: a recorded text model, or an image folder,
/// which is not recorded.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Text,
    Image,
}

impl Kind {
    fn folder(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
        }
    }
}

/// An install a test made, with the bytes it held.
struct Install {
    id: String,
    dir: PathBuf,
    files: Vec<(PathBuf, Vec<u8>)>,
}

impl Install {
    /// Bytes in all its files, which is what a move reports as its total.
    fn bytes(&self) -> u64 {
        self.files.iter().map(|(_, bytes)| bytes.len() as u64).sum()
    }
}

/// A temp home and a folder of fake volumes, removed when the test ends,
/// passing or not. Declared last, `_serial` is released after the cleanup.
struct Fixture {
    base: PathBuf,
    home: PathBuf,
    volumes: PathBuf,
    _serial: MutexGuard<'static, ()>,
}

impl Fixture {
    fn new(tag: &str) -> Self {
        let serial = SERIAL
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let base =
            std::env::temp_dir().join(format!("turbospark-ffi-moves-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        // The temp dir is behind the /var symlink, and the volume rules
        // compare canonical paths.
        let base = base.canonicalize().unwrap();
        let home = base.join("home");
        let volumes = base.join("Volumes");
        fs::create_dir_all(&home).unwrap();
        for volume in ["T7", "T9", "Spare"] {
            fs::create_dir_all(volumes.join(volume)).unwrap();
        }
        let at = |volume: &str| volumes.join(volume).display().to_string();
        let table = format!(
            "/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)\n\
             /dev/disk9s1 on {} (apfs, local, nodev, nosuid, journaled, noowners)\n\
             /dev/disk10s1 on {} (apfs, local, nodev, nosuid, journaled, noowners)\n\
             /dev/disk11s1 on {} (apfs, local, nodev, nosuid, journaled, noowners)\n",
            at("T7"),
            at("T9"),
            at("Spare"),
        );
        let mounts = base.join("mounts.txt");
        fs::write(&mounts, table).unwrap();
        let device = fs::metadata(&base).unwrap().dev();
        let fixture = Self {
            base,
            home,
            volumes,
            _serial: serial,
        };
        std::env::set_var("TURBOSPARK_TEST_MOUNT_TABLE", &mounts);
        std::env::set_var("TURBOSPARK_TEST_VOLUMES_ROOT", &fixture.volumes);
        // Unlike the temp folder's, so the fake volumes are other disks.
        std::env::set_var(
            "TURBOSPARK_TEST_HOME_DEVICE",
            device.wrapping_add(1).to_string(),
        );
        let home = c(fixture.home.to_str().unwrap());
        assert_eq!(unsafe { ts_store_root_set(home.as_ptr()) }, abi::TS_OK);
        fixture
    }

    fn volume(&self, name: &str) -> PathBuf {
        self.volumes.join(name)
    }

    /// The library root `add_library` makes on the volume `name`.
    fn root(&self, name: &str) -> PathBuf {
        self.volume(name).join("TurboSpark")
    }

    /// The models folder of `place`, This Mac for `None`.
    fn models(&self, place: Option<&Lib>) -> PathBuf {
        match place {
            Some(lib) => self.root(lib.name),
            None => self.home.join("models"),
        }
    }

    fn add_library(&self, name: &'static str) -> Lib {
        let path = c(self.volume(name).to_str().unwrap());
        let label = c(name);
        let added = reply(|out| unsafe { ts_library_add(path.as_ptr(), label.as_ptr(), out) }).ok();
        Lib {
            id: added.json()["library"]["id"].as_str().unwrap().to_string(),
            name,
        }
    }

    /// Takes the volume `name` away as unplugging its drive would. The volumes
    /// folder is then made unwritable, as the real `/Volumes` is, so nothing
    /// can rebuild the drive's folders on the disk under the mount point.
    fn unplug(&self, name: &str) {
        fs::remove_dir_all(self.volume(name)).unwrap();
        fs::set_permissions(&self.volumes, fs::Permissions::from_mode(0o555)).unwrap();
    }

    /// Rewrites the marker of the library on `name` as a newer TurboSpark
    /// would, which makes the library read-only to this build.
    fn mark_newer(&self, name: &str) {
        let path = self.root(name).join(MARKER_FILE);
        let mut json: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        json["format"] = json!(2);
        fs::write(&path, serde_json::to_vec_pretty(&json).unwrap()).unwrap();
    }

    fn store(&self) -> Store {
        Store::new(&self.home)
    }

    /// Makes an install of `alias` in `place`: a nested folder of files, an
    /// empty one among them, and for a text model its row in the registry and
    /// in the library's marker, as an install would leave them. The library
    /// must be added first.
    fn install(&self, place: Option<&Lib>, kind: Kind, alias: &str) -> Install {
        let relative = PathBuf::from(kind.folder()).join(format!("{alias}.gturbo"));
        let dir = self.models(place).join(&relative);
        fs::create_dir_all(dir.join("weights")).unwrap();
        fs::write(dir.join("config.json"), format!("{{\"alias\": {alias:?}}}")).unwrap();
        fs::write(dir.join("weights/part-0.bin"), vec![7u8; 3000]).unwrap();
        fs::write(dir.join("weights/part-1.bin"), vec![9u8; 5000]).unwrap();
        fs::write(dir.join("empty.txt"), b"").unwrap();
        if kind == Kind::Text {
            let row = InstalledModel {
                alias: alias.to_string(),
                repo: "owner/name".to_string(),
                revision: "main".to_string(),
                path: dir.clone(),
                family: "qwen3".to_string(),
                install_bytes: 8_000,
                installed_on: "2026-10-16".to_string(),
                status: "runs".to_string(),
                kind: None,
                modality: ModelModality::Text,
                variant: None,
                library: place.map(|lib| lib.id.clone()),
                relative_path: place.map(|_| relative.clone()),
            };
            let env = LocationEnv::system(&self.home).unwrap();
            self.store().record_in_library(&[row], &env).unwrap();
        }
        let id = format!("{}:{}/{alias}", id_of(place), kind.folder());
        Install {
            id,
            files: snapshot(&dir),
            dir,
        }
    }

    /// The rows of the home's `installed.json`.
    fn recorded(&self) -> Vec<Value> {
        match fs::read_to_string(self.home.join("installed.json")) {
            Ok(text) => serde_json::from_str(&text).unwrap(),
            Err(_) => Vec::new(),
        }
    }

    /// The aliases in the marker of the library on `name`.
    fn marker_aliases(&self, name: &str) -> Vec<String> {
        LibraryMarker::read(&self.root(name))
            .unwrap()
            .unwrap()
            .models
            .into_iter()
            .map(|row| row.alias)
            .collect()
    }

    /// `move-` folders a move left where it stages: the home, and each
    /// library's `.staging`.
    fn leftovers(&self) -> Vec<String> {
        let mut found = Vec::new();
        for dir in [
            self.home.clone(),
            self.root("T7").join(".staging"),
            self.root("T9").join(".staging"),
        ] {
            for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                if name.starts_with("move-") {
                    found.push(name);
                }
            }
        }
        found
    }

    /// What a refused or cancelled move must leave exactly as it was.
    fn state(&self, source: &Install) -> State {
        let read = |path: PathBuf| fs::read_to_string(path).ok();
        State {
            registry: read(self.home.join("installed.json")),
            t7_marker: read(self.root("T7").join(MARKER_FILE)),
            t9_marker: read(self.root("T9").join(MARKER_FILE)),
            source: snapshot(&source.dir),
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        // Writable again, so the tree below can be removed.
        let _ = fs::set_permissions(&self.volumes, fs::Permissions::from_mode(0o755));
        unsafe { ts_store_root_set(ptr::null()) };
        for name in [
            "TURBOSPARK_TEST_MOUNT_TABLE",
            "TURBOSPARK_TEST_VOLUMES_ROOT",
            "TURBOSPARK_TEST_HOME_DEVICE",
        ] {
            std::env::remove_var(name);
        }
        let _ = fs::remove_dir_all(&self.base);
    }
}

#[derive(Debug, PartialEq)]
struct State {
    registry: Option<String>,
    t7_marker: Option<String>,
    t9_marker: Option<String>,
    source: Vec<(PathBuf, Vec<u8>)>,
}

/// Takes a folder's write permission away, and gives it back when dropped.
struct Locked(PathBuf);

impl Locked {
    fn new(path: PathBuf) -> Self {
        fs::set_permissions(&path, fs::Permissions::from_mode(0o555)).unwrap();
        Self(path)
    }
}

impl Drop for Locked {
    fn drop(&mut self) {
        let _ = fs::set_permissions(&self.0, fs::Permissions::from_mode(0o755));
    }
}

/// Points the Hub at a port nothing listens on while it lives, so an install
/// that a regression let through fails at once instead of reaching the real
/// network.
struct Offline;

impl Offline {
    fn new() -> Self {
        catalog::set_hf_endpoint_override(Some("http://127.0.0.1:9".to_string()));
        Self
    }
}

impl Drop for Offline {
    fn drop(&mut self) {
        catalog::set_hf_endpoint_override(None);
    }
}

fn id_of(place: Option<&Lib>) -> &str {
    place.map_or("internal", |lib| lib.id.as_str())
}

/// Every file under `dir` with its bytes, sorted, empty when `dir` is gone.
fn snapshot(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    fn walk(root: &Path, at: &Path, found: &mut Vec<(PathBuf, Vec<u8>)>) {
        for entry in fs::read_dir(at).into_iter().flatten().flatten() {
            let path = entry.path();
            if entry.file_type().unwrap().is_dir() {
                walk(root, &path, found);
            } else {
                found.push((
                    path.strip_prefix(root).unwrap().to_path_buf(),
                    fs::read(&path).unwrap(),
                ));
            }
        }
    }
    let mut found = Vec::new();
    walk(dir, dir, &mut found);
    found.sort();
    found
}

fn c(text: &str) -> CString {
    CString::new(text).unwrap()
}

/// Reads and frees a `char **` out-parameter.
unsafe fn take(out: *mut c_char) -> String {
    let text = CStr::from_ptr(out).to_str().unwrap().to_string();
    ts_string_free(out);
    text
}

fn last_error() -> String {
    unsafe {
        let length = ts_last_error(ptr::null_mut(), 0);
        let mut buffer = vec![0u8; length + 1];
        ts_last_error(buffer.as_mut_ptr() as *mut c_char, buffer.len());
        CStr::from_ptr(buffer.as_ptr() as *const c_char)
            .to_str()
            .unwrap()
            .to_string()
    }
}

/// What one entry point returned: its status, its output, and its error.
struct Reply {
    code: c_int,
    text: Option<String>,
    error: String,
}

impl Reply {
    fn ok(self) -> Self {
        assert_eq!(self.code, abi::TS_OK, "{}", self.error);
        self
    }

    fn json(&self) -> Value {
        serde_json::from_str(self.text.as_deref().expect("the call wrote output")).unwrap()
    }
}

/// A call whose result goes through a `char **`. On a failure the output must
/// stay untouched, which `Reply::text` shows.
fn reply(call: impl FnOnce(*mut *mut c_char) -> c_int) -> Reply {
    let mut out: *mut c_char = ptr::null_mut();
    let code = call(&mut out);
    let error = if code == abi::TS_OK {
        String::new()
    } else {
        last_error()
    };
    let text = (!out.is_null()).then(|| unsafe { take(out) });
    Reply { code, text, error }
}

/// A refusal carries its stable code first, then a sentence, and writes no
/// output.
fn assert_refused(reply: &Reply, code: &str) {
    assert_eq!(
        reply.code,
        abi::TS_ERR_INVALID_ARGUMENT,
        "{code}: {}",
        reply.error
    );
    assert!(
        reply.error.starts_with(&format!("{code}: ")),
        "wanted the code {code:?} first, got {:?}",
        reply.error
    );
    assert!(
        reply.error.len() > code.len() + 2,
        "a sentence follows the code: {:?}",
        reply.error
    );
    assert!(reply.text.is_none(), "a refusal writes no output");
}

/// A refusal because an install or a move runs: its own status, the `busy`
/// code, and a sentence about what runs.
fn assert_busy(reply: &Reply, running: &str) {
    assert_eq!(reply.code, abi::TS_ERR_BUSY, "{}", reply.error);
    assert!(
        reply.error.starts_with("busy: "),
        "wanted the code busy first, got {:?}",
        reply.error
    );
    assert!(reply.error.contains(running), "{:?}", reply.error);
    assert!(reply.text.is_none(), "a refusal writes no output");
}

// ---------------------------------------------------------------------- calls

/// One progress callback call.
#[derive(Clone, Debug)]
struct Event {
    kind: c_int,
    done: u64,
    total: u64,
    text_is_null: bool,
}

/// What a test sees of one call through its callback, and what it does there:
/// optionally cancel the move on the calling thread at the first byte update
/// that has progress, and optionally park the call at its first update.
struct Probe {
    events: Mutex<Vec<Event>>,
    cancel_on_progress: AtomicBool,
    /// What `ts_model_move_cancel` answered when the callback called it, -1
    /// until it did.
    cancel_answer: AtomicI32,
    gate: Option<Gate>,
}

struct Gate {
    started: Mutex<Sender<()>>,
    release: Mutex<Receiver<()>>,
    first: AtomicBool,
}

impl Probe {
    fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            cancel_on_progress: AtomicBool::new(false),
            cancel_answer: AtomicI32::new(-1),
            gate: None,
        }
    }

    /// A probe that cancels the move from inside its first progress update.
    fn cancelling() -> Self {
        let probe = Self::new();
        probe.cancel_on_progress.store(true, Ordering::SeqCst);
        probe
    }

    fn userdata(&self) -> *mut c_void {
        self as *const Probe as *mut c_void
    }

    fn events(&self) -> Vec<Event> {
        self.events.lock().unwrap().clone()
    }

    /// `(done, total)` of every byte update.
    fn bytes(&self) -> Vec<(u64, u64)> {
        self.events()
            .iter()
            .filter(|event| event.kind == TS_INSTALL_BYTES)
            .map(|event| (event.done, event.total))
            .collect()
    }
}

unsafe extern "C" fn probe_callback(
    userdata: *mut c_void,
    kind: c_int,
    text: *const c_char,
    _len: usize,
    done: u64,
    total: u64,
) {
    let probe = &*(userdata as *const Probe);
    probe.events.lock().unwrap().push(Event {
        kind,
        done,
        total,
        text_is_null: text.is_null(),
    });
    if kind == TS_INSTALL_BYTES
        && done > 0
        && probe.cancel_on_progress.swap(false, Ordering::SeqCst)
    {
        probe
            .cancel_answer
            .store(ts_model_move_cancel(), Ordering::SeqCst);
    }
    if let Some(gate) = &probe.gate {
        if gate.first.swap(false, Ordering::SeqCst) {
            gate.started.lock().unwrap().send(()).unwrap();
            let _ = gate
                .release
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(60));
        }
    }
}

fn mv_with(install_id: &str, library_id: &str, free: i64, probe: Option<&Probe>) -> Reply {
    let (install_id, library_id) = (c(install_id), c(library_id));
    let callback: turbospark_ffi::TsInstallCallback = probe.map(|_| probe_callback as _);
    reply(|out| unsafe {
        ts_model_move(
            install_id.as_ptr(),
            library_id.as_ptr(),
            free,
            callback,
            probe.map_or(ptr::null_mut(), Probe::userdata),
            out,
        )
    })
}

/// A move with plenty of room, watched by `probe`.
fn mv(install: &Install, to: Option<&Lib>, probe: &Probe) -> Reply {
    mv_with(&install.id, id_of(to), PLENTY, Some(probe))
}

/// A call with no output.
fn status(code: c_int) -> Reply {
    let error = if code == abi::TS_OK {
        String::new()
    } else {
        last_error()
    };
    Reply {
        code,
        text: None,
        error,
    }
}

fn forget(install_id: &str) -> Reply {
    let install_id = c(install_id);
    status(unsafe { ts_model_forget(install_id.as_ptr()) })
}

fn set_default(id: &str) -> Reply {
    let id = c(id);
    status(unsafe { ts_library_set_default(id.as_ptr()) })
}

fn remove(id: &str) -> Reply {
    let id = c(id);
    status(unsafe { ts_library_remove(id.as_ptr()) })
}

fn add(path: &Path) -> Reply {
    let path = c(path.to_str().unwrap());
    reply(|out| unsafe { ts_library_add(path.as_ptr(), ptr::null(), out) })
}

fn import(root: &Path) -> Reply {
    let root = c(root.to_str().unwrap());
    reply(|out| unsafe { ts_store_import_moved_root(root.as_ptr(), out) })
}

fn rows_of(reply: Reply) -> Vec<Value> {
    reply.ok().json().as_array().unwrap().clone()
}

// ---------------------------------------------------------------- held calls

/// A call parked in its callback while it is registered as running. Dropped
/// without `finish`, it is cancelled and released, so a failed assertion
/// cannot leave a thread waiting.
struct Held<T: Send + 'static> {
    _probe: Arc<Probe>,
    release: Sender<()>,
    thread: Option<JoinHandle<T>>,
}

impl<T: Send + 'static> Held<T> {
    /// Runs `run` on a thread with the address of a probe that parks it at
    /// its first callback, and returns once it is parked.
    fn start(run: impl FnOnce(usize) -> T + Send + 'static) -> Self {
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let mut probe = Probe::new();
        probe.gate = Some(Gate {
            started: Mutex::new(started_tx),
            release: Mutex::new(release_rx),
            first: AtomicBool::new(true),
        });
        let probe = Arc::new(probe);
        let address = Arc::as_ptr(&probe) as usize;
        let thread = thread::spawn(move || run(address));
        let held = Self {
            _probe: probe,
            release: release_tx,
            thread: Some(thread),
        };
        started_rx
            .recv_timeout(Duration::from_secs(60))
            .expect("the call reached its first callback");
        held
    }

    /// Lets the call go and returns what it returned.
    fn finish(mut self) -> T {
        self.release.send(()).unwrap();
        self.thread.take().unwrap().join().unwrap()
    }
}

impl<T: Send + 'static> Drop for Held<T> {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            unsafe {
                ts_model_move_cancel();
                ts_install_cancel();
            }
            let _ = self.release.send(());
            let _ = thread.join();
        }
    }
}

/// A move parked in its first progress update, after it registered.
fn held_move(install: &Install, to: &Lib) -> Held<Reply> {
    let (install_id, library_id) = (c(&install.id), c(&to.id));
    Held::start(move |address| {
        reply(|out| unsafe {
            ts_model_move(
                install_id.as_ptr(),
                library_id.as_ptr(),
                PLENTY,
                Some(probe_callback),
                address as *mut c_void,
                out,
            )
        })
    })
}

/// A real `ts_install` walk, parked in its first stage line while it is
/// registered as running. Cancelled before it is released, it stops at its
/// entry check without a network request, and returns its status code.
fn held_install() -> Held<c_int> {
    let catalog = reply(|out| unsafe { ts_catalog_json(out) }).ok().json();
    let alias = c(catalog[0]["alias"].as_str().unwrap());
    Held::start(move |address| {
        let mut out: *mut c_char = ptr::null_mut();
        let code = unsafe {
            ts_install(
                alias.as_ptr(),
                Some(probe_callback),
                address as *mut c_void,
                &mut out,
            )
        };
        if !out.is_null() {
            unsafe { ts_string_free(out) };
        }
        code
    })
}

// ----------------------------------------------------------------- the moves

/// An install moves from library to library and to and from This Mac. Each
/// time the callback reports bytes from 0 of the total up to the total, the
/// call ends `moved` and names the install's new id, the files are in the
/// destination byte for byte, the original is gone, and the registry row and
/// both library markers follow.
#[test]
fn a_move_reports_progress_and_the_install_follows_in_every_direction() {
    let fx = Fixture::new("directions");
    let t7 = fx.add_library("T7");
    let t9 = fx.add_library("T9");
    let qwen = fx.install(Some(&t7), Kind::Text, "qwen");
    let total = qwen.bytes();
    assert_eq!(fx.marker_aliases("T7"), ["qwen"]);

    // T7 to T9, T9 to This Mac, This Mac to T7.
    let mut at: Option<&Lib> = Some(&t7);
    for to in [Some(&t9), None, Some(&t7)] {
        let id = format!("{}:text/qwen", id_of(at));
        let probe = Probe::new();

        let answer = mv_with(&id, id_of(to), PLENTY, Some(&probe)).ok().json();

        let new_id = format!("{}:text/qwen", id_of(to));
        assert_eq!(
            answer,
            json!({"outcome": "moved", "installId": new_id}),
            "{id} to {}",
            id_of(to)
        );
        // Progress: only byte updates, from nothing to everything, never
        // going back, with a report for each file copied.
        let events = probe.events();
        assert!(events
            .iter()
            .all(|e| e.kind == TS_INSTALL_BYTES && e.text_is_null));
        let bytes = probe.bytes();
        assert_eq!(bytes.first(), Some(&(0, total)), "{bytes:?}");
        assert_eq!(bytes.last(), Some(&(total, total)), "{bytes:?}");
        assert!(bytes.len() >= 3, "{bytes:?}");
        assert!(
            bytes.windows(2).all(|pair| pair[0].0 <= pair[1].0),
            "{bytes:?}"
        );
        assert!(bytes.iter().all(|(_, of)| *of == total), "{bytes:?}");

        // The files, the original, and the staging folders.
        let landed = fx.models(to).join("text/qwen.gturbo");
        assert_eq!(snapshot(&landed), qwen.files, "{id}");
        assert!(!fx.models(at).join("text/qwen.gturbo").exists(), "{id}");
        assert!(fx.leftovers().is_empty(), "{:?}", fx.leftovers());

        // The registry row, and the markers: gone from the source's, in the
        // destination's.
        let rows = fx.recorded();
        assert_eq!(rows.len(), 1, "{rows:?}");
        assert_eq!(rows[0]["alias"], "qwen");
        assert_eq!(rows[0]["path"], landed.to_str().unwrap());
        match to {
            Some(lib) => {
                assert_eq!(rows[0]["library"], lib.id);
                assert_eq!(rows[0]["relative_path"], "text/qwen.gturbo");
                assert_eq!(fx.marker_aliases(lib.name), ["qwen"]);
            }
            None => assert!(rows[0].get("library").is_none(), "{rows:?}"),
        }
        if let Some(lib) = at {
            assert!(fx.marker_aliases(lib.name).is_empty(), "{id}");
        }

        // The listing shows it where it now is, under the id the move named.
        let listed = rows_of(reply(|out| unsafe { ts_installed_json(out) }));
        let row = listed.iter().find(|row| row["alias"] == "qwen").unwrap();
        assert_eq!(row["install_id"], new_id);
        assert_eq!(row["availability"], "available");
        assert_eq!(
            row["location_kind"],
            if to.is_some() { "library" } else { "internal" }
        );
        at = to;
    }
}

/// An image install has no registry row or marker entry: the folder moves,
/// and the answer names its new id.
#[test]
fn an_image_install_moves_as_a_folder() {
    let fx = Fixture::new("image");
    let t7 = fx.add_library("T7");
    let image = fx.install(Some(&t7), Kind::Image, "z-image");
    let probe = Probe::new();

    let answer = mv(&image, None, &probe).ok().json();

    assert_eq!(
        answer,
        json!({"outcome": "moved", "installId": "internal:image/z-image"})
    );
    let landed = fx.models(None).join("image/z-image.gturbo");
    assert_eq!(snapshot(&landed), image.files);
    assert!(!image.dir.exists());
    assert!(fx.recorded().is_empty(), "{:?}", fx.recorded());
    assert!(fx.marker_aliases("T7").is_empty());
    assert_eq!(probe.bytes().last(), Some(&(image.bytes(), image.bytes())));
}

/// A call with no callback moves just the same, and the space needed is the
/// boundary: one byte short is refused, exactly enough moves.
#[test]
fn a_move_needs_no_callback_and_exactly_enough_space() {
    let fx = Fixture::new("boundary");
    let t7 = fx.add_library("T7");
    let qwen = fx.install(None, Kind::Text, "qwen");
    let total = qwen.bytes() as i64;

    let short = mv_with(&qwen.id, &t7.id, total - 1, None);

    assert_refused(&short, "insufficient_space");
    assert!(
        short.error.contains(&format!("{} bytes free", total - 1))
            && short.error.contains(&format!("needs {total}")),
        "{}",
        short.error
    );
    assert_eq!(snapshot(&qwen.dir), qwen.files);
    assert!(!fx.root("T7").join("text").exists());
    assert!(fx.leftovers().is_empty());

    mv_with(&qwen.id, &t7.id, total, None).ok();
    assert_eq!(
        snapshot(&fx.root("T7").join("text/qwen.gturbo")),
        qwen.files
    );
    assert!(!qwen.dir.exists());
}

// ---------------------------------------------------------------------- cancel

/// A cancel called from the progress callback, on the calling thread, stops
/// the copy partway: the call is `cancelled` with its own status, the
/// original is exactly as it was and still listed, the destination holds
/// nothing, nothing is recorded, and the next move is not poisoned by the
/// cancel.
#[test]
fn a_cancel_from_the_callback_leaves_the_original_and_the_destination_clean() {
    let fx = Fixture::new("cancel");
    let t7 = fx.add_library("T7");
    let t9 = fx.add_library("T9");
    let qwen = fx.install(Some(&t7), Kind::Text, "qwen");
    let before = fx.state(&qwen);
    let probe = Probe::cancelling();

    let cancelled = mv(&qwen, Some(&t9), &probe);

    assert_eq!(cancelled.code, abi::TS_ERR_CANCELLED, "{}", cancelled.error);
    assert!(
        cancelled.error.starts_with("cancelled: "),
        "{}",
        cancelled.error
    );
    assert!(cancelled.text.is_none());
    assert_eq!(
        probe.cancel_answer.load(Ordering::SeqCst),
        1,
        "the move was running when it was cancelled"
    );
    // Stopped partway, not after the last file.
    let bytes = probe.bytes();
    let last = bytes.last().unwrap();
    assert!(last.0 > 0 && last.0 < last.1, "{bytes:?}");
    // Everything is as it was: files, rows, markers, and no leftover copy.
    assert_eq!(fx.state(&qwen), before);
    assert!(!fx.root("T9").join("text/qwen.gturbo").exists());
    assert!(fx.leftovers().is_empty(), "{:?}", fx.leftovers());
    assert_eq!(fx.recorded().len(), 1);
    assert_eq!(fx.recorded()[0]["library"], t7.id);
    assert_eq!(fx.marker_aliases("T7"), ["qwen"]);
    assert!(fx.marker_aliases("T9").is_empty());

    // Nothing runs now, so a cancel does nothing and says so, and the
    // flag of the move that was cancelled does not reach the next one.
    assert_eq!(unsafe { ts_model_move_cancel() }, 0);
    mv(&qwen, Some(&t9), &Probe::new()).ok();
    assert_eq!(
        snapshot(&fx.root("T9").join("text/qwen.gturbo")),
        qwen.files
    );
    assert!(!qwen.dir.exists());
}

// ------------------------------------------------------------------------ busy

/// A move is refused while an install runs. The refusal changes nothing and
/// calls nothing back, cancelling the move that is not running does nothing,
/// and the install, which is cancelled the way it always was, is untouched by
/// it. When the install is over the same move goes through.
#[test]
fn a_move_is_refused_while_an_install_runs() {
    let fx = Fixture::new("busy-install");
    let t7 = fx.add_library("T7");
    let qwen = fx.install(None, Kind::Text, "qwen");
    let before = fx.state(&qwen);
    let held = held_install();
    let probe = Probe::new();

    let refused = mv(&qwen, Some(&t7), &probe);

    assert_busy(&refused, "install");
    assert!(probe.events().is_empty(), "a refused move reports nothing");
    assert_eq!(fx.state(&qwen), before);
    assert!(!fx.root("T7").join("text").exists());
    // There is no move to cancel, and the install is not the move's to stop.
    assert_eq!(unsafe { ts_model_move_cancel() }, 0);
    // Cancelled the way an install always was, it ends as cancelled.
    assert_eq!(unsafe { ts_install_cancel() }, 1, "one walk was running");
    assert_eq!(held.finish(), abi::TS_ERR_CANCELLED);

    // Over: the same move goes through.
    mv(&qwen, Some(&t7), &probe).ok();
    assert!(!qwen.dir.exists());
}

/// Cancelling a move is not cancelling an install. With no move running, the
/// call signals nothing, and an install that runs meanwhile is let go with no
/// cancel of its own, so it is not cancelled: it goes on to the Hub, which
/// here is a closed port, and fails there. A cancel button left over from a
/// move would otherwise stop an unrelated install.
#[test]
fn cancelling_a_move_does_not_stop_a_running_install() {
    let _fx = Fixture::new("cancel-isolation");
    let _offline = Offline::new();
    let held = held_install();

    assert_eq!(unsafe { ts_model_move_cancel() }, 0, "no move was running");

    let ended = held.finish();
    assert_ne!(
        ended,
        abi::TS_ERR_CANCELLED,
        "a move's cancel stopped an install"
    );
}

/// While a move runs, a library change, a second move, an install of every
/// kind, and a store relocation are refused, and none of them changes
/// anything. Reading is never refused. Cancelling an install does not stop the
/// move. When the move is over, the same calls go through.
#[test]
fn library_changes_installs_and_other_moves_are_refused_while_a_move_runs() {
    let fx = Fixture::new("busy-move");
    let _offline = Offline::new();
    let t7 = fx.add_library("T7");
    let t9 = fx.add_library("T9");
    let qwen = fx.install(Some(&t7), Kind::Text, "qwen");
    let other = fx.install(None, Kind::Text, "other");
    let identity = {
        let catalog = reply(|out| unsafe { ts_audio_catalog_json(out) })
            .ok()
            .json();
        c(&catalog[0]["identity"].to_string())
    };
    let moved_root = fx.volume("Spare");
    fs::create_dir_all(moved_root.join("models")).unwrap();

    let held = held_move(&qwen, &t9);

    // Library changes.
    let before = fx.recorded();
    let libraries = rows_of(reply(|out| unsafe { ts_libraries_json(out) }));
    for refused in [
        add(&fx.volume("Spare")),
        remove(&t9.id),
        set_default(&t9.id),
        forget(&qwen.id),
        import(&moved_root),
    ] {
        assert_busy(&refused, "model move");
    }
    // A second move, whichever way it points.
    assert_busy(&mv(&other, Some(&t9), &Probe::new()), "model move");
    assert_busy(&mv(&qwen, None, &Probe::new()), "model move");
    // Every install entry point, before any request.
    let alias = c("tinyllama");
    let repo = c("owner/model@0123456789abcdef0123456789abcdef01234567");
    let file = c("model-Q8_0.gguf");
    let installs = [
        reply(|out| unsafe { ts_install(alias.as_ptr(), None, ptr::null_mut(), out) }),
        reply(|out| unsafe {
            ts_install_repo_with_options(
                repo.as_ptr(),
                alias.as_ptr(),
                file.as_ptr(),
                ptr::null(),
                ptr::null(),
                None,
                ptr::null_mut(),
                out,
            )
        }),
        reply(|out| unsafe { ts_image_install(alias.as_ptr(), None, ptr::null_mut(), out) }),
        reply(|out| unsafe {
            ts_audio_install_with_options(
                identity.as_ptr(),
                ptr::null(),
                None,
                ptr::null_mut(),
                out,
            )
        }),
        reply(|out| unsafe {
            // Refused before the pin is checked, so any revision does.
            let revision = c("0");
            ts_embedding_install(
                alias.as_ptr(),
                revision.as_ptr(),
                None,
                ptr::null_mut(),
                out,
            )
        }),
    ];
    for refused in &installs {
        assert_busy(refused, "model move");
    }
    // A store relocation would pull the home from under the move.
    let relocated = reply(|out| unsafe {
        let to = c(fx.base.join("elsewhere").to_str().unwrap());
        ts_store_relocate(to.as_ptr(), None, ptr::null_mut(), out)
    });
    assert_ne!(relocated.code, abi::TS_OK);
    assert!(
        relocated.error.contains("model move"),
        "{}",
        relocated.error
    );
    assert!(!fx.base.join("elsewhere").exists());
    // Nothing changed behind the refusals; reading is never refused.
    assert_eq!(fx.recorded(), before);
    assert_eq!(
        rows_of(reply(|out| unsafe { ts_libraries_json(out) })),
        libraries
    );
    rows_of(reply(|out| unsafe { ts_installed_json(out) }));
    // No install walk is running, and stopping one does not stop the move.
    assert_eq!(unsafe { ts_install_cancel() }, 0);

    let moved = held.finish();

    assert_eq!(
        moved.ok().json(),
        json!({"outcome": "moved", "installId": format!("{}:text/qwen", t9.id)})
    );
    assert!(!qwen.dir.exists());
    // Over: the same calls go through.
    set_default(&t9.id).ok();
    forget(&other.id).ok();
    add(&fx.volume("Spare")).ok();
    remove(&t9.id).ok();
}

// -------------------------------------------------------------------- refusals

/// A destination or a source that is not connected is `not_connected`, naming
/// the library, and one a newer TurboSpark wrote is `library_read_only`. Each
/// is settled before a byte is copied, and the original stays listed.
#[test]
fn an_unplugged_or_read_only_library_is_refused_before_copying() {
    let fx = Fixture::new("libraries");
    let t7 = fx.add_library("T7");
    let t9 = fx.add_library("T9");
    let on_mac = fx.install(None, Kind::Text, "mac");
    let on_t9 = fx.install(Some(&t9), Kind::Text, "drive");

    // A read-only destination, then a read-only source.
    fx.mark_newer("T9");
    let before = fx.state(&on_mac);
    let probe = Probe::new();
    assert_refused(&mv(&on_mac, Some(&t9), &probe), "library_read_only");
    assert_refused(&mv(&on_t9, Some(&t7), &probe), "library_read_only");
    assert_eq!(fx.state(&on_mac), before);
    assert_eq!(snapshot(&on_t9.dir), on_t9.files);
    assert!(probe.events().is_empty());

    // A destination that is gone, then a source that is.
    fx.unplug("T9");
    let before = fx.state(&on_mac);
    let refused = mv(&on_mac, Some(&t9), &probe);
    assert_refused(&refused, "not_connected");
    assert!(refused.error.contains("T9"), "{}", refused.error);
    assert_eq!(fx.state(&on_mac), before);
    let refused = mv(&on_t9, None, &probe);
    assert_refused(&refused, "not_connected");
    assert!(refused.error.contains("T9"), "{}", refused.error);
    assert!(!fx.volume("T9").exists(), "the drive was rebuilt");
    assert!(probe.events().is_empty());
    assert_eq!(fx.recorded().len(), 2);
}

/// The ids a move is asked for are checked against the registry: a library
/// the list does not have is `unknown_library` as a destination, and an
/// install nothing has is `unknown_install`, also when its id names a library
/// that is gone. Embedding and OCR ids are `home_only`, and an install is not
/// moved to the library it is in.
#[test]
fn unknown_ids_and_home_only_installs_are_refused_by_their_codes() {
    let fx = Fixture::new("ids");
    let t7 = fx.add_library("T7");
    let qwen = fx.install(None, Kind::Text, "qwen");
    let drive = fx.install(Some(&t7), Kind::Text, "drive");
    let gone = "5a1d9c70-3e2b-4f68-8a14-6b0c2d9e7f31";
    let before = fx.state(&qwen);
    let probe = Probe::new();
    let to =
        |install_id: &str, library_id: &str| mv_with(install_id, library_id, PLENTY, Some(&probe));

    // A destination the list does not have, also an empty one.
    for library in [gone, "", "T7"] {
        let refused = to(&qwen.id, library);
        assert_refused(&refused, "unknown_library");
    }
    // An install nothing has: no row, a form that is none, an image folder
    // that is not there, and a library that is gone.
    for install_id in [
        "internal:text/ghost",
        "internal:image/ghost",
        "nonsense",
        "",
        &format!("{gone}:text/qwen"),
        &format!("{gone}:image/qwen"),
    ] {
        assert_refused(&to(install_id, &t7.id), "unknown_install");
    }
    // Embedding and OCR models stay on This Mac.
    for install_id in ["internal:embedding/gemma", "internal:ocr/paddle"] {
        assert_refused(&to(install_id, &t7.id), "home_only");
    }
    // Already there.
    assert_refused(&to(&qwen.id, "internal"), "same_library");
    assert_refused(&to(&drive.id, &t7.id), "same_library");

    assert_eq!(fx.state(&qwen), before);
    assert_eq!(snapshot(&drive.dir), drive.files);
    assert!(probe.events().is_empty());
    assert!(fx.leftovers().is_empty());
}

/// A destination that already holds the alias is `destination_taken`, an
/// install whose folder is not where its row says is `io`, and a free-space
/// figure the app could not measure is `free_space_unknown`, not a guess. All
/// leave the original alone, and none leaves the move registered as running.
#[test]
fn a_taken_destination_a_missing_folder_and_unknown_space_are_refused() {
    let fx = Fixture::new("states");
    let t7 = fx.add_library("T7");
    let qwen = fx.install(None, Kind::Text, "qwen");
    // The alias is already in the destination, as a folder.
    fs::create_dir_all(fx.root("T7").join("text/qwen.gturbo")).unwrap();
    // A row whose folder was never there.
    let ghost = fx.install(None, Kind::Text, "ghost");
    fs::remove_dir_all(&ghost.dir).unwrap();
    let before = fx.state(&qwen);
    let probe = Probe::new();

    let taken = mv(&qwen, Some(&t7), &probe);
    assert_refused(&taken, "destination_taken");
    assert!(taken.error.contains("T7"), "{}", taken.error);

    let missing = mv(&ghost, Some(&t7), &probe);
    assert_refused(&missing, "io");
    assert!(missing.error.contains("missing"), "{}", missing.error);

    fs::remove_dir_all(fx.root("T7").join("text/qwen.gturbo")).unwrap();
    for free in [-1, i64::MIN] {
        let unknown = mv_with(&qwen.id, &t7.id, free, Some(&probe));
        assert_refused(&unknown, "free_space_unknown");
        assert!(
            unknown.error.contains("nothing was moved"),
            "{}",
            unknown.error
        );
    }

    assert_eq!(fx.state(&qwen), before);
    assert!(probe.events().is_empty());
    assert!(fx.leftovers().is_empty());
    // Not registered by any of them: the move goes through now.
    mv(&qwen, Some(&t7), &probe).ok();
    assert!(!qwen.dir.exists());
}

/// A null id, a null library, and a null out-pointer are refused up front,
/// before the move registers or anything is read.
#[test]
fn bad_arguments_are_refused_before_anything_moves() {
    let fx = Fixture::new("arguments");
    let t7 = fx.add_library("T7");
    let qwen = fx.install(None, Kind::Text, "qwen");
    let before = fx.state(&qwen);
    let (id, library) = (c(&qwen.id), c(&t7.id));

    let calls = unsafe {
        [
            reply(|out| {
                ts_model_move(
                    ptr::null(),
                    library.as_ptr(),
                    PLENTY,
                    None,
                    ptr::null_mut(),
                    out,
                )
            }),
            reply(|out| {
                ts_model_move(id.as_ptr(), ptr::null(), PLENTY, None, ptr::null_mut(), out)
            }),
        ]
    };
    for call in &calls {
        assert_eq!(call.code, abi::TS_ERR_INVALID_ARGUMENT, "{}", call.error);
        assert!(call.text.is_none());
    }
    let no_output = unsafe {
        ts_model_move(
            id.as_ptr(),
            library.as_ptr(),
            PLENTY,
            None,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    assert_eq!(no_output, abi::TS_ERR_INVALID_ARGUMENT, "{}", last_error());

    assert_eq!(fx.state(&qwen), before);
    assert!(!fx.root("T7").join("text").exists());
}

// -------------------------------------------------------------------- outcomes

/// A move that cannot rewrite the original library's marker, or cannot delete
/// the original folder, still put the install in its destination, and says
/// which: the answer is success with the leftover named, so the app does not
/// show a failed move for an install that is in its new library.
#[test]
fn a_move_that_cannot_clean_up_still_reports_the_destination() {
    let fx = Fixture::new("outcomes");
    let t7 = fx.add_library("T7");

    // The source library's root cannot take the marker's replacement.
    let stale = fx.install(Some(&t7), Kind::Text, "stale");
    {
        let _locked = Locked::new(fx.root("T7"));
        let answer = mv_with(&stale.id, "internal", PLENTY, None).ok().json();
        assert_eq!(
            answer,
            json!({
                "outcome": "source_marker_stale",
                "installId": "internal:text/stale",
                "staleMarker": "T7",
            })
        );
    }
    assert_eq!(
        snapshot(&fx.models(None).join("text/stale.gturbo")),
        stale.files
    );
    assert!(!stale.dir.exists());
    assert_eq!(fx.recorded()[0]["alias"], "stale");
    assert!(fx.recorded()[0].get("library").is_none());

    // The folder that holds the original cannot lose an entry.
    let stuck = fx.install(Some(&t7), Kind::Text, "stuck");
    let leftover = {
        let _locked = Locked::new(fx.root("T7").join("text"));
        let answer = mv_with(&stuck.id, "internal", PLENTY, None).ok().json();
        assert_eq!(answer["outcome"], "source_cleanup_failed", "{answer}");
        assert_eq!(answer["installId"], "internal:text/stuck");
        assert_eq!(answer["staleMarker"], Value::Null, "{answer}");
        PathBuf::from(answer["path"].as_str().unwrap())
    };
    assert_eq!(leftover, stuck.dir);
    assert_eq!(
        snapshot(&fx.models(None).join("text/stuck.gturbo")),
        stuck.files
    );
    // Both are listed in their destination, This Mac. The marker lost the
    // second, and still lists the first: that row is the harmless extra the
    // stale outcome names.
    let rows = fx.recorded();
    assert_eq!(rows.len(), 2, "{rows:?}");
    assert!(rows.iter().all(|row| row.get("library").is_none()));
    assert_eq!(fx.marker_aliases("T7"), ["stale"]);
}

// ---------------------------------------------------------------------- header

/// The header documents the move: the entry points, the answer's fields, and
/// every code the app maps, in the library block.
#[test]
fn the_header_documents_the_move_and_its_codes() {
    let header = include_str!("../include/turbospark.h");
    let block = &header[header.find("---- model libraries").unwrap()..];
    let block = &block[..block.find("int32_t ts_libraries_json(").unwrap()];
    for needle in [
        "ts_model_move",
        "insufficient_space",
        "verify_failed",
        "destination_taken",
        "same_library",
        "unknown_install",
        "unknown_library",
        "free_space_unknown",
        "library_read_only",
        "home_only",
        "not_connected",
        "cancelled",
        "TS_ERR_CANCELLED",
        "busy",
    ] {
        assert!(
            block.contains(needle),
            "the library block never says {needle}"
        );
    }
    // The comment above the declaration of the move itself.
    let declaration = header.find("int32_t ts_model_move(").unwrap();
    let comment = &header[header.find("/* Moves the install `install_id`").unwrap()..declaration];
    for needle in [
        "source_marker_stale",
        "source_cleanup_failed",
        "installId",
        "staleMarker",
        "\"path\"",
        "TS_INSTALL_BYTES",
        "TS_ERR_BUSY",
        "TS_ERR_CANCELLED",
        "NEGATIVE",
    ] {
        assert!(
            comment.contains(needle),
            "ts_model_move never says {needle}"
        );
    }
    assert!(header.contains("int32_t ts_model_move_cancel(void);"));
}
