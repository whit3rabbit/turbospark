//! Deleting installs through the C ABI (4.6): `ts_model_delete_install`,
//! `ts_image_delete_install`, and the older `ts_model_delete` and
//! `ts_image_delete`, driven through the same function bodies the header
//! exposes, against a temp home, fake volumes, and small fixture installs.
//! `catalog::Store::delete_install` decides every rule and is tested in the
//! catalog; what is tested here is the native interface around it: the stable
//! code of every refusal, the busy guards, and that no entry point deletes
//! through the path a row recorded.
//!
//! The location rules read the mount table and the home folder's device.
//! Debug builds take both from the `TURBOSPARK_TEST_*` overrides, set here in
//! process, so a temp folder stands in for `/Volumes` and a fixture file for
//! `/sbin/mount`. Release builds ignore the overrides by design, so this file
//! is debug-only. The store root and the environment are process-wide, so
//! every test holds one lock for its whole run.
#![cfg(debug_assertions)]

use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::fs;
use std::os::raw::{c_char, c_int, c_void};
use std::os::unix::fs::{symlink, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use catalog::library::{LibraryMarker, LocationEnv, MARKER_FILE};
use catalog::{InstalledModel, ModelModality, Store};
use serde_json::{json, Value};
use turbospark_ffi::{
    abi, ts_audio_adopt, ts_audio_catalog_json, ts_audio_delete, ts_audio_delete_legacy,
    ts_catalog_json, ts_image_delete, ts_image_delete_install, ts_image_installed_json, ts_install,
    ts_install_cancel, ts_installed_json, ts_last_error, ts_library_add, ts_model_delete,
    ts_model_delete_install, ts_model_move, ts_model_move_cancel, ts_store_root_set,
    ts_string_free,
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Text,
    /// A vision-tower sidecar, `text/<alias>.gturbo-vision`.
    Tower,
}

/// An install a test made.
struct Install {
    id: String,
    dir: PathBuf,
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
        let base = std::env::temp_dir().join(format!(
            "turbospark-ffi-deletes-{tag}-{}",
            std::process::id()
        ));
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

    fn add_library(&self, name: &'static str) -> Lib {
        let path = c(self.volume(name).to_str().unwrap());
        let label = c(name);
        let added = reply(|out| unsafe { ts_library_add(path.as_ptr(), label.as_ptr(), out) }).ok();
        Lib {
            id: added.json()["library"]["id"].as_str().unwrap().to_string(),
            name,
        }
    }

    /// Takes the volume `name` away, to give back with [`Self::replug`].
    fn unplug(&self, name: &str) {
        fs::rename(self.volume(name), self.base.join(format!("away-{name}"))).unwrap();
    }

    fn replug(&self, name: &str) {
        fs::rename(self.base.join(format!("away-{name}")), self.volume(name)).unwrap();
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

    /// Makes a text install of `alias` in `place` (This Mac for `None`): a
    /// nested folder of files, an empty one among them, and its row in the
    /// registry and in the library's marker, as an install would leave them.
    /// The library must be added first.
    fn install(&self, place: Option<&Lib>, kind: Kind, alias: &str) -> Install {
        let root = match place {
            Some(lib) => self.root(lib.name),
            None => self.home.join("models"),
        };
        let suffix = match kind {
            Kind::Text => "gturbo",
            Kind::Tower => "gturbo-vision",
        };
        let relative = PathBuf::from("text").join(format!("{alias}.{suffix}"));
        let dir = root.join(&relative);
        fs::create_dir_all(dir.join("weights")).unwrap();
        fs::write(dir.join("config.json"), format!("{{\"alias\": {alias:?}}}")).unwrap();
        fs::write(dir.join("weights/part-0.bin"), vec![7u8; 3000]).unwrap();
        fs::write(dir.join("empty.txt"), b"").unwrap();
        let row = InstalledModel {
            alias: alias.to_string(),
            repo: "owner/name".to_string(),
            revision: "main".to_string(),
            path: dir.clone(),
            family: "qwen3".to_string(),
            install_bytes: 3_000,
            installed_on: "2026-10-16".to_string(),
            status: "runs".to_string(),
            kind: (kind == Kind::Tower).then(|| "vision-tower".to_string()),
            modality: ModelModality::Text,
            variant: None,
            library: place.map(|lib| lib.id.clone()),
            relative_path: place.map(|_| relative.clone()),
        };
        let env = LocationEnv::system(&self.home).unwrap();
        self.store().record_in_library(&[row], &env).unwrap();
        Install {
            id: format!("{}:text/{alias}", id_of(place)),
            dir,
        }
    }

    /// A valid image install `alias` in `place`, built by the production
    /// builder from a synthetic source, as `model` (an `andrevp/` model keeps
    /// its MLX source tree beside the install, and `kept` makes that folder).
    fn image(&self, place: Option<&Lib>, alias: &str, model: &str, kept: bool) -> Install {
        let parent = match place {
            Some(lib) => self.root(lib.name).join("image"),
            None => self.home.join("models/image"),
        };
        let source = self.base.join(format!("source-{alias}"));
        write_synthetic_source(&source);
        fs::create_dir_all(&parent).unwrap();
        let dir = parent.join(format!("{alias}.gturbo"));
        image::build_image_install(&image::ImageInstallSpec {
            source_root: source.clone(),
            output_root: dir.clone(),
            model_id: model.to_string(),
            model_revision: "synthetic-revision".to_string(),
        })
        .expect("build a synthetic image install");
        fs::remove_dir_all(source).unwrap();
        if kept {
            let kept = parent.join(format!(".{alias}.synthetic-revision.image-source"));
            fs::create_dir_all(kept.join("text_encoder")).unwrap();
            fs::write(kept.join("text_encoder/model.safetensors"), b"source").unwrap();
        }
        Install {
            id: format!("{}:image/{alias}", id_of(place)),
            dir: dir.canonicalize().unwrap(),
        }
    }

    /// The aliases of the rows of the home's `installed.json`, sorted.
    fn recorded(&self) -> Vec<String> {
        let rows: Vec<Value> = match fs::read_to_string(self.home.join("installed.json")) {
            Ok(text) => serde_json::from_str(&text).unwrap(),
            Err(_) => Vec::new(),
        };
        let mut aliases: Vec<String> = rows
            .iter()
            .map(|row| row["alias"].as_str().unwrap().to_string())
            .collect();
        aliases.sort();
        aliases
    }

    /// The aliases in the marker of the library on `name`, sorted.
    fn marked(&self, name: &str) -> Vec<String> {
        let mut aliases: Vec<String> = LibraryMarker::read(&self.root(name))
            .unwrap()
            .unwrap()
            .models
            .into_iter()
            .map(|row| row.alias)
            .collect();
        aliases.sort();
        aliases
    }

    /// Anything left where deletes rename an install aside.
    fn staged(&self, name: &str) -> Vec<String> {
        let mut found: Vec<String> = fs::read_dir(self.root(name).join(".staging"))
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        found.sort();
        found
    }

    /// A folder outside every library with a file in it.
    fn outside(&self, name: &str) -> PathBuf {
        let dir = self.base.join(name);
        fs::create_dir_all(dir.join("inner")).unwrap();
        fs::write(dir.join("inner/precious.bin"), b"precious").unwrap();
        dir
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        // Writable again, so the tree below can be removed.
        let _ = fs::set_permissions(&self.home, fs::Permissions::from_mode(0o755));
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

/// Takes a folder's write permission away, and gives it back when dropped.
struct Locked(PathBuf);

impl Locked {
    fn new(path: &Path) -> Self {
        fs::set_permissions(path, fs::Permissions::from_mode(0o555)).unwrap();
        Self(path.to_path_buf())
    }
}

impl Drop for Locked {
    fn drop(&mut self) {
        let _ = fs::set_permissions(&self.0, fs::Permissions::from_mode(0o755));
    }
}

fn id_of(place: Option<&Lib>) -> &str {
    place.map_or("internal", |lib| lib.id.as_str())
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
        for entry in fs::read_dir(at).into_iter().flatten().flatten() {
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

fn is_gone(path: &Path) -> bool {
    fs::symlink_metadata(path).is_err()
}

fn write_synthetic_source(source: &Path) {
    fs::create_dir_all(source.join("tokenizer")).unwrap();
    fs::create_dir_all(source.join("scheduler")).unwrap();
    fs::write(source.join("tokenizer/tokenizer.json"), b"tokenizer").unwrap();
    fs::write(
        source.join("tokenizer/tokenizer_config.json"),
        b"{\"chat_template\": \"test\"}",
    )
    .unwrap();
    fs::write(
        source.join("scheduler/scheduler_config.json"),
        b"{\"shift\": 3.0}",
    )
    .unwrap();
    for (component, index) in [
        ("text_encoder", "model.safetensors.index.json"),
        (
            "transformer",
            "diffusion_pytorch_model.safetensors.index.json",
        ),
        ("vae", "diffusion_pytorch_model.safetensors.index.json"),
    ] {
        let dir = source.join(component);
        fs::create_dir_all(&dir).unwrap();
        let file = index.strip_suffix(".index.json").unwrap();
        let mut payload = Vec::new();
        for value in std::iter::repeat_n(0.25f32, 64).chain(std::iter::repeat_n(1.0f32, 64)) {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let header = json!({
            "linear.weight": {"dtype": "F32", "shape": [1, 64], "data_offsets": [0, 256]},
            "norm.weight": {"dtype": "F32", "shape": [64], "data_offsets": [256, 512]}
        });
        let header = serde_json::to_vec(&header).unwrap();
        let mut shard = (header.len() as u64).to_le_bytes().to_vec();
        shard.extend_from_slice(&header);
        shard.extend_from_slice(&payload);
        fs::write(dir.join(file), shard).unwrap();
        if component != "vae" {
            fs::write(
                dir.join(index),
                serde_json::to_vec(&json!({
                    "weight_map": {"linear.weight": file, "norm.weight": file}
                }))
                .unwrap(),
            )
            .unwrap();
        }
    }
}

// ------------------------------------------------------------------- C helpers

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

/// A call whose result goes through a `char **`.
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

/// A refusal carries its stable code first, then a sentence.
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
}

fn delete(install: &str) -> Reply {
    let install = c(install);
    status(unsafe { ts_model_delete_install(install.as_ptr()) })
}

fn delete_image(install: &str) -> Reply {
    let install = c(install);
    status(unsafe { ts_image_delete_install(install.as_ptr()) })
}

fn delete_by_alias(alias: &str) -> Reply {
    let alias = c(alias);
    status(unsafe { ts_model_delete(alias.as_ptr()) })
}

fn delete_image_by(identifier: &str) -> Reply {
    let identifier = c(identifier);
    status(unsafe { ts_image_delete(identifier.as_ptr()) })
}

/// The rows `ts_installed_json` lists, by alias.
fn listed(alias: &str) -> Option<Value> {
    let rows = reply(|out| unsafe { ts_installed_json(out) }).ok().json();
    rows.as_array()
        .unwrap()
        .iter()
        .find(|row| row["alias"] == alias)
        .cloned()
}

/// The aliases `ts_image_installed_json` lists, sorted.
fn listed_images() -> Vec<String> {
    let rows = reply(|out| unsafe { ts_image_installed_json(out) })
        .ok()
        .json();
    let mut aliases: Vec<String> = rows
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["alias"].as_str().unwrap().to_string())
        .collect();
    aliases.sort();
    aliases
}

// ---------------------------------------------------------------- held calls

/// One progress callback call, parked at its first update.
struct Probe {
    gate: Option<Gate>,
}

struct Gate {
    started: Mutex<Sender<()>>,
    release: Mutex<Receiver<()>>,
    first: AtomicBool,
}

unsafe extern "C" fn probe_callback(
    userdata: *mut c_void,
    _kind: c_int,
    _text: *const c_char,
    _len: usize,
    _done: u64,
    _total: u64,
) {
    let probe = &*(userdata as *const Probe);
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
        let probe = Arc::new(Probe {
            gate: Some(Gate {
                started: Mutex::new(started_tx),
                release: Mutex::new(release_rx),
                first: AtomicBool::new(true),
            }),
        });
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
fn held_move(install: &Install, to: &Lib) -> Held<c_int> {
    let (install_id, library_id) = (c(&install.id), c(&to.id));
    Held::start(move |address| {
        let mut out: *mut c_char = ptr::null_mut();
        let code = unsafe {
            ts_model_move(
                install_id.as_ptr(),
                library_id.as_ptr(),
                PLENTY,
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

// ------------------------------------------------------------------- the deletes

/// A text install, a vision tower and an image install in a connected
/// library are each deleted: the folder, the row and the marker entry go,
/// the kept source goes with its image, nothing else changes, and nothing is
/// left where the folder was renamed aside.
#[test]
fn a_text_vision_and_image_install_are_deleted_from_a_library() {
    let fx = Fixture::new("deletes");
    let t7 = fx.add_library("T7");
    let qwen = fx.install(Some(&t7), Kind::Text, "qwen");
    let tower = fx.install(Some(&t7), Kind::Tower, "qwen-vision");
    let keep = fx.install(Some(&t7), Kind::Text, "keep");
    let model = "andrevp/Z-Image-Turbo-MLX-4bit";
    let pic = fx.image(Some(&t7), "pic", model, true);
    let keep_pic = fx.image(Some(&t7), "keep-pic", model, true);
    let kept_source = fx
        .root("T7")
        .join("image/.pic.synthetic-revision.image-source");
    let (keep_before, keep_pic_before) = (tree(&keep.dir), tree(&keep_pic.dir));
    assert_eq!(fx.marked("T7"), ["keep", "qwen", "qwen-vision"]);
    assert_eq!(listed_images(), ["keep-pic", "pic"]);

    delete(&qwen.id).ok();
    delete(&tower.id).ok();
    delete_image(&pic.id).ok();

    assert!(is_gone(&qwen.dir) && is_gone(&tower.dir));
    assert!(is_gone(&pic.dir) && is_gone(&kept_source));
    assert_eq!(fx.recorded(), ["keep"]);
    assert_eq!(fx.marked("T7"), ["keep"]);
    assert!(listed("qwen").is_none() && listed("qwen-vision").is_none());
    assert!(listed("keep").is_some());
    assert_eq!(listed_images(), ["keep-pic"]);
    assert_eq!(tree(&keep.dir), keep_before);
    assert_eq!(tree(&keep_pic.dir), keep_pic_before);
    assert_eq!(fx.staged("T7"), Vec::<String>::new());
    // Gone is gone: the same ids name nothing now.
    assert_refused(&delete(&qwen.id), "unknown_install");
    assert_refused(&delete_image(&pic.id), "unknown_install");
}

/// A library that is not connected is refused as `not_connected`, with the
/// row still listed, as not connected, and nothing deleted. The folders are
/// as they were when the drive comes back.
#[test]
fn an_unplugged_library_is_refused_and_keeps_its_rows() {
    let fx = Fixture::new("unplugged");
    let t7 = fx.add_library("T7");
    let qwen = fx.install(Some(&t7), Kind::Text, "qwen");
    let pic = fx.image(Some(&t7), "pic", "Tongyi-MAI/Z-Image-Turbo", false);
    let (qwen_before, pic_before) = (tree(&qwen.dir), tree(&pic.dir));
    fx.unplug("T7");

    assert_refused(&delete(&qwen.id), "not_connected");
    assert_refused(&delete_image(&pic.id), "not_connected");
    // The old entry point names an alias held by a library, and goes no
    // further than refusing it.
    assert_refused(&delete_by_alias("qwen"), "use_install_id");

    assert_eq!(fx.recorded(), ["qwen"]);
    assert_eq!(listed("qwen").unwrap()["availability"], "not_connected");
    fx.replug("T7");
    assert_eq!(tree(&qwen.dir), qwen_before);
    assert_eq!(tree(&pic.dir), pic_before);
    assert_eq!(fx.marked("T7"), ["qwen"]);
}

/// The hazard this task closes: the library is away and a DIFFERENT drive is
/// mounted at the path the row recorded, holding a folder there. Deleting by
/// install id is refused, and so is the older call by alias, which used to
/// remove the recorded path and the row. The other drive is byte for byte as
/// it was, whatever the outcome.
#[test]
fn a_different_drive_at_the_recorded_path_is_never_touched() {
    let fx = Fixture::new("other-drive");
    let t7 = fx.add_library("T7");
    let qwen = fx.install(Some(&t7), Kind::Text, "qwen");
    fx.image(Some(&t7), "pic", "Tongyi-MAI/Z-Image-Turbo", false);
    fx.unplug("T7");
    // The recorded path now answers for another drive.
    let other = fx.volume("T7").join("TurboSpark");
    fs::create_dir_all(other.join("text/qwen.gturbo/weights")).unwrap();
    fs::write(other.join("text/qwen.gturbo/weights/part-0.bin"), b"theirs").unwrap();
    fs::create_dir_all(other.join("image/pic.gturbo")).unwrap();
    fs::write(other.join("image/pic.gturbo/manifest.json"), b"{}").unwrap();
    assert_eq!(qwen.dir, other.join("text/qwen.gturbo"));

    // Whether or not the other drive carries a library marker of its own.
    for marker in [false, true] {
        if marker {
            LibraryMarker::new(
                "7c1e4b52-9d03-4a86-b1f7-2e5a8c3d6f90".to_string(),
                "Other".to_string(),
            )
            .write(&other)
            .unwrap();
        }
        let before = tree(&fx.volume("T7"));

        assert_refused(&delete(&qwen.id), "not_connected");
        assert_refused(
            &delete_image(&format!("{}:image/pic", t7.id)),
            "not_connected",
        );
        assert_refused(&delete_by_alias("qwen"), "use_install_id");
        // This Mac's image delete never looks at a library.
        let by_path = delete_image_by(other.join("image/pic.gturbo").to_str().unwrap());
        assert_eq!(by_path.code, abi::TS_ERR_INVALID_ARGUMENT);
        assert!(by_path.error.contains("not installed"), "{}", by_path.error);

        assert_eq!(
            tree(&fx.volume("T7")),
            before,
            "marker on the other drive: {marker}"
        );
        assert_eq!(fx.recorded(), ["qwen"], "the row stays");
    }
}

/// A link planted where the library's folders are is never followed. An
/// install that is not reached as a plain folder is listed as missing, and
/// deleting it forgets the row and the marker entry and touches no file: not
/// the folder the link points at, and not the link. An image reached through
/// a link is no install.
#[test]
fn a_planted_link_is_never_followed() {
    let fx = Fixture::new("links");
    let t7 = fx.add_library("T7");
    let qwen = fx.install(Some(&t7), Kind::Text, "qwen");
    let pic = fx.image(Some(&t7), "pic", "Tongyi-MAI/Z-Image-Turbo", false);
    // The text folder is now a link to a folder outside the library.
    let outside = fx.outside("outside");
    fs::create_dir_all(outside.join("qwen.gturbo")).unwrap();
    fs::write(outside.join("qwen.gturbo/weights.bin"), b"theirs").unwrap();
    let parked = fx.base.join("parked-text");
    fs::rename(fx.root("T7").join("text"), &parked).unwrap();
    symlink(&outside, fx.root("T7").join("text")).unwrap();
    // So is the image folder.
    let parked_image = fx.base.join("parked-image");
    fs::rename(fx.root("T7").join("image"), &parked_image).unwrap();
    symlink(&outside, fx.root("T7").join("image")).unwrap();
    let (outside_before, parked_before, parked_image_before) =
        (tree(&outside), tree(&parked), tree(&parked_image));
    assert_eq!(listed("qwen").unwrap()["availability"], "missing");
    assert!(
        listed_images().is_empty(),
        "a linked image folder lists none"
    );

    assert_refused(&delete_image(&pic.id), "unknown_install");
    delete(&qwen.id).ok();

    assert_eq!(tree(&outside), outside_before);
    assert_eq!(tree(&parked), parked_before);
    assert_eq!(tree(&parked_image), parked_image_before);
    for link in ["text", "image"] {
        assert!(
            fs::symlink_metadata(fx.root("T7").join(link))
                .unwrap()
                .file_type()
                .is_symlink(),
            "{link} is still the planted link"
        );
    }
    assert!(listed("qwen").is_none());
    assert_eq!(fx.recorded(), Vec::<String>::new());
    assert_eq!(fx.marked("T7"), Vec::<String>::new());
}

/// A library root that is a link off its drive is not followed: its rows are
/// missing, a delete is refused rather than treated as a forget, and the
/// folder the link leads to is untouched.
#[test]
fn a_library_root_linked_off_its_drive_is_refused() {
    let fx = Fixture::new("root-link");
    let t7 = fx.add_library("T7");
    let qwen = fx.install(Some(&t7), Kind::Text, "qwen");
    let pic = fx.image(Some(&t7), "pic", "Tongyi-MAI/Z-Image-Turbo", false);
    let parked = fx.base.join("parked-library");
    fs::rename(fx.root("T7"), &parked).unwrap();
    symlink(&parked, fx.root("T7")).unwrap();
    let before = tree(&fx.base);

    for refused in [delete(&qwen.id), delete_image(&pic.id)] {
        assert_refused(&refused, "io");
        assert!(
            refused.error.contains("leads off its drive"),
            "{}",
            refused.error
        );
    }

    assert_eq!(tree(&fx.base), before);
    assert_eq!(fx.recorded(), ["qwen"]);
}

/// A library a newer TurboSpark wrote is read-only here: nothing is deleted
/// or forgotten, because its marker would keep listing the row.
#[test]
fn a_read_only_library_is_refused() {
    let fx = Fixture::new("read-only");
    let t7 = fx.add_library("T7");
    let qwen = fx.install(Some(&t7), Kind::Text, "qwen");
    let pic = fx.image(Some(&t7), "pic", "Tongyi-MAI/Z-Image-Turbo", false);
    fx.mark_newer("T7");
    let before = tree(&fx.base);

    assert_refused(&delete(&qwen.id), "library_read_only");
    assert_refused(&delete_image(&pic.id), "library_read_only");

    assert_eq!(tree(&fx.base), before);
}

/// If the row cannot be forgotten once the folder has been renamed aside,
/// the folder is back where it was, and the error comes back with its code.
#[test]
fn a_failed_forget_puts_the_folder_back() {
    let fx = Fixture::new("rollback");
    let t7 = fx.add_library("T7");
    let qwen = fx.install(Some(&t7), Kind::Text, "qwen");
    let before = tree(&qwen.dir);
    // The registry is written through a temp file in the home.
    let _locked = Locked::new(&fx.home);

    let refused = delete(&qwen.id);

    assert_refused(&refused, "io");
    assert!(
        refused.error.contains("installed.json"),
        "{}",
        refused.error
    );
    assert_eq!(tree(&qwen.dir), before);
    assert_eq!(fx.staged("T7"), Vec::<String>::new());
    assert_eq!(fx.recorded(), ["qwen"]);
    assert_eq!(fx.marked("T7"), ["qwen"]);
}

/// This Mac's installs are deleted as before, by alias and by install id,
/// text and image, with the row gone and nothing else touched. A library's
/// image is not found by `ts_image_delete`, which looks at This Mac only.
#[test]
fn this_mac_deletes_still_work() {
    let fx = Fixture::new("this-mac");
    let t7 = fx.add_library("T7");
    let by_alias = fx.install(None, Kind::Text, "by-alias");
    let by_id = fx.install(None, Kind::Text, "by-id");
    let tower = fx.install(None, Kind::Tower, "tower");
    let keep = fx.install(None, Kind::Text, "keep");
    let model = "Tongyi-MAI/Z-Image-Turbo";
    let home_a = fx.image(None, "home-a", model, false);
    let home_b = fx.image(None, "home-b", model, false);
    let home_c = fx.image(None, "home-c", model, false);
    let library_pic = fx.image(Some(&t7), "drive-pic", model, false);
    let (keep_before, drive_before) = (tree(&keep.dir), tree(&library_pic.dir));

    delete_by_alias("by-alias").ok();
    delete(&by_id.id).ok();
    delete(&tower.id).ok();
    delete_image_by("home-a").ok();
    delete_image_by(home_b.dir.to_str().unwrap()).ok();
    delete_image(&home_c.id).ok();

    for dir in [
        &by_alias.dir,
        &by_id.dir,
        &tower.dir,
        &home_a.dir,
        &home_b.dir,
        &home_c.dir,
    ] {
        assert!(is_gone(dir), "{}", dir.display());
    }
    assert_eq!(fx.recorded(), ["keep"]);
    assert_eq!(tree(&keep.dir), keep_before);
    // A library's image is not This Mac's to delete.
    for refused in [
        delete_image_by("drive-pic"),
        delete_image_by(library_pic.dir.to_str().unwrap()),
    ] {
        assert_eq!(refused.code, abi::TS_ERR_INVALID_ARGUMENT);
        assert!(refused.error.contains("not installed"), "{}", refused.error);
    }
    assert_eq!(tree(&library_pic.dir), drive_before);
    // Nothing named so is installed.
    let missing = delete_by_alias("nope");
    assert_refused(&missing, "unknown_install");
    assert!(missing.error.contains("not installed"), "{}", missing.error);
    assert_refused(&delete("internal:text/nope"), "unknown_install");
}

/// Ids that name nothing say so with their own codes, an id of a library the
/// list does not have is `unknown_library` from the registry, and a null
/// pointer is a plain argument error.
#[test]
fn ids_that_name_nothing_have_their_own_codes() {
    let fx = Fixture::new("unknown");
    let t7 = fx.add_library("T7");
    let qwen = fx.install(Some(&t7), Kind::Text, "qwen");
    let pic = fx.image(Some(&t7), "pic", "Tongyi-MAI/Z-Image-Turbo", false);

    for id in [
        "",
        "garbage",
        "internal:text/nope",
        &format!("{}:text/nope", t7.id),
    ] {
        assert_refused(&delete(id), "unknown_install");
    }
    // The wrong kind of id for the entry point names nothing there.
    assert_refused(&delete(&pic.id), "unknown_install");
    assert_refused(&delete_image(&qwen.id), "unknown_install");
    assert_refused(&delete_image("garbage"), "unknown_install");
    assert_refused(
        &delete_image(&format!("{}:image/nope", t7.id)),
        "unknown_install",
    );
    // A library that is not listed.
    assert_refused(&delete("not-listed:text/qwen"), "unknown_library");
    assert_refused(&delete_image("not-listed:image/pic"), "unknown_library");
    // A name is not an id, as for installs and moves.
    assert_refused(&delete("T7:text/qwen"), "unknown_library");
    // An alias that would walk out of the library is no install.
    assert_refused(&delete_image(&format!("{}:image/../qwen", t7.id)), "io");

    let null = status(unsafe { ts_model_delete_install(ptr::null()) });
    assert_eq!(null.code, abi::TS_ERR_INVALID_ARGUMENT);
    let null = status(unsafe { ts_image_delete_install(ptr::null()) });
    assert_eq!(null.code, abi::TS_ERR_INVALID_ARGUMENT);
    assert!(qwen.dir.is_dir() && pic.dir.is_dir(), "nothing was deleted");
}

/// Text a drive supplies (a library's name) reaches the message of a refusal
/// through the same escape and cap the code is placed in front of.
#[test]
fn a_library_name_from_a_drive_is_escaped_and_capped() {
    let fx = Fixture::new("hostile");
    let hostile = format!("evil\u{1b}[31m\u{202e}{}", "x".repeat(2000));
    let id = "0f6c2a3e-1b7d-4c55-9e0a-7d2f1c3b4a59";
    let root = fx.root("T7");
    fs::create_dir_all(root.join("text/qwen.gturbo")).unwrap();
    let mut marker = LibraryMarker::new(id.to_string(), hostile);
    marker.models = vec![catalog::library::PortableRow {
        alias: "qwen".to_string(),
        repo: "owner/qwen".to_string(),
        revision: "main".to_string(),
        relative_path: PathBuf::from("text/qwen.gturbo"),
        family: "llama".to_string(),
        install_bytes: 1,
        installed_on: "2026-10-16".to_string(),
        status: "verified".to_string(),
        kind: None,
        modality: ModelModality::Text,
        variant: None,
    }];
    marker.write(&root).unwrap();
    let path = c(fx.volume("T7").to_str().unwrap());
    reply(|out| unsafe { ts_library_add(path.as_ptr(), ptr::null(), out) }).ok();
    fx.unplug("T7");

    let refused = delete(&format!("{id}:text/qwen"));

    assert_refused(&refused, "not_connected");
    assert!(refused.error.contains("evil"), "{}", refused.error);
    assert!(!refused.error.contains('\u{1b}') && !refused.error.contains('\u{202e}'));
    let sentence = refused.error.strip_prefix("not_connected: ").unwrap();
    assert!(
        sentence.chars().count() <= 300,
        "{}",
        sentence.chars().count()
    );
}

// ------------------------------------------------------------------------ busy

/// Every delete, and the audio entry points that write This Mac's rows, are
/// refused as `busy` while an install runs and while a model move runs, with
/// nothing changed. When the work is over, the same deletes go through.
#[test]
fn deletes_and_audio_writes_are_refused_while_an_install_or_a_move_runs() {
    let fx = Fixture::new("busy");
    let t7 = fx.add_library("T7");
    let t9 = fx.add_library("T9");
    let qwen = fx.install(Some(&t7), Kind::Text, "qwen");
    let local = fx.install(None, Kind::Text, "local");
    let pic = fx.image(Some(&t7), "pic", "Tongyi-MAI/Z-Image-Turbo", false);
    let home_pic = fx.image(None, "home-pic", "Tongyi-MAI/Z-Image-Turbo", false);
    let identity = {
        let catalog = reply(|out| unsafe { ts_audio_catalog_json(out) })
            .ok()
            .json();
        c(&catalog[0]["identity"].to_string())
    };
    let state = || {
        (
            fs::read_to_string(fx.home.join("installed.json")).ok(),
            fs::read_to_string(fx.root("T7").join(MARKER_FILE)).ok(),
            tree(&fx.base.join("Volumes")),
            tree(&fx.home.join("models")),
        )
    };
    let refused_calls = |running: &str| {
        let adopt = reply(|out| unsafe { ts_audio_adopt(identity.as_ptr(), out) });
        assert_busy(&adopt, running);
        assert!(adopt.text.is_none(), "a refusal writes no output");
        for refused in [
            delete(&qwen.id),
            delete(&local.id),
            delete_image(&pic.id),
            delete_image(&home_pic.id),
            delete_by_alias("local"),
            delete_image_by("home-pic"),
            status(unsafe { ts_audio_delete(identity.as_ptr()) }),
            status(unsafe { ts_audio_delete_legacy(identity.as_ptr()) }),
        ] {
            assert_busy(&refused, running);
        }
    };
    let before = state();

    // An install runs.
    let held = held_install();
    refused_calls("install");
    assert_eq!(state(), before);
    assert_eq!(unsafe { ts_install_cancel() }, 1, "one walk was running");
    assert_eq!(held.finish(), abi::TS_ERR_CANCELLED);

    // A model move runs.
    let held = held_move(&qwen, &t9);
    refused_calls("model move");
    assert_eq!(held.finish(), abi::TS_OK);

    // Over: the same deletes go through. The move put qwen in T9.
    let moved = format!("{}:text/qwen", t9.id);
    delete(&moved).ok();
    delete(&local.id).ok();
    delete_image(&pic.id).ok();
    delete_image(&home_pic.id).ok();
    assert_eq!(fx.recorded(), Vec::<String>::new());
    assert!(is_gone(&local.dir) && is_gone(&pic.dir) && is_gone(&home_pic.dir));
}
