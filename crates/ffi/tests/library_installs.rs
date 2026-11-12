//! Install destinations through the C ABI (4.3): `ts_install_with_options`,
//! `ts_install_repo_with_options`, `ts_image_install_with_options` and the
//! `library` option of `ts_audio_install_with_options`, driven through the
//! same function bodies the header exposes, against a temp home, fake
//! volumes, and a local fixture Hub.
//!
//! The location rules read the mount table and the home folder's device.
//! Debug builds take both from the `TURBOSPARK_TEST_*` overrides, set here in
//! process, so a temp folder stands in for `/Volumes` and a fixture file for
//! `/sbin/mount`. Release builds ignore the overrides by design, so this file
//! is debug-only. The Hub endpoint, the store root and the environment are
//! process-wide, so every test holds one lock for its whole run.
//!
//! The load-bearing assertions are the refusals that must come before a
//! single request: each is checked against the fixture Hub's count of
//! connections, so a refusal that came after the first request fails.
#![cfg(debug_assertions)]

use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::os::raw::{c_char, c_int, c_void};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use catalog::library::{LibraryMarker, MARKER_FILE};
use serde_json::{json, Value};
use turbospark_ffi::{
    abi, ts_audio_catalog_json, ts_audio_install_with_options, ts_image_install_with_options,
    ts_image_installed_json, ts_install_cancel, ts_install_repo_with_options,
    ts_install_with_options, ts_installed_json, ts_last_error, ts_library_add, ts_library_remove,
    ts_store_root_set, ts_string_free, TS_INSTALL_STAGE,
};

/// The audio catalog alias the refusal tests install. Its files are real
/// multi-gigabyte pins, so no test here lets its transfer start.
const MUSIC: &str = "minimax-music3-4bit";
/// The image row the fixture Hub stands in for.
const IMAGE: &str = "z-image-turbo";
/// The curated text row the fixture Hub stands in for.
const TEXT: &str = "tinyllama";
/// A revision the fixture repository `owner/model` answers to.
const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

/// The store root, the Hub endpoint and the environment are process-wide.
static SERIAL: Mutex<()> = Mutex::new(());

// ------------------------------------------------------------------ fixture

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
            "turbospark-ffi-installs-{tag}-{}",
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
        for volume in ["T7", "T9"] {
            fs::create_dir_all(volumes.join(volume)).unwrap();
        }
        let at = |volume: &str| volumes.join(volume).display().to_string();
        let table = format!(
            "/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)\n\
             /dev/disk9s1 on {} (apfs, local, nodev, nosuid, journaled, noowners)\n\
             /dev/disk10s1 on {} (apfs, local, nodev, nosuid, journaled, noowners)\n",
            at("T7"),
            at("T9"),
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

    /// Adds the volume `name` as a library of the same name and returns its id.
    fn add_library(&self, name: &str) -> String {
        let path = c(self.volume(name).to_str().unwrap());
        let name = c(name);
        let added = reply(|out| unsafe { ts_library_add(path.as_ptr(), name.as_ptr(), out) }).ok();
        added.json()["library"]["id"].as_str().unwrap().to_string()
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
fn assert_refused(reply: &Reply, code: &str, entry: &str) {
    assert_eq!(
        reply.code,
        abi::TS_ERR_INVALID_ARGUMENT,
        "{entry}: {code}: {}",
        reply.error
    );
    assert!(
        reply.error.starts_with(&format!("{code}: ")),
        "{entry}: wanted the code {code:?} first, got {:?}",
        reply.error
    );
    assert!(
        reply.error.len() > code.len() + 2,
        "{entry}: a sentence follows the code: {:?}",
        reply.error
    );
    assert!(reply.text.is_none(), "{entry}: a refusal writes no output");
}

fn rows_of(reply: Reply) -> Vec<Value> {
    reply.ok().json().as_array().unwrap().clone()
}

// ------------------------------------------------------------ the install watch

/// The text a stage line must contain, and the action to run once it does.
type Trigger = (String, Box<dyn FnOnce() + Send>);

/// What a test sees of one install through its callback: every stage line,
/// a count of byte updates, and optionally one action to run, on the calling
/// thread, at the first stage line that contains some text.
struct Watch {
    stages: Mutex<Vec<String>>,
    bytes: AtomicUsize,
    trigger: Mutex<Option<Trigger>>,
}

impl Watch {
    fn new() -> Self {
        Self {
            stages: Mutex::new(Vec::new()),
            bytes: AtomicUsize::new(0),
            trigger: Mutex::new(None),
        }
    }

    /// A watch that runs `action` at the first stage line containing `needle`.
    fn at(needle: &str, action: impl FnOnce() + Send + 'static) -> Self {
        let watch = Self::new();
        *watch.trigger.lock().unwrap() = Some((needle.to_string(), Box::new(action)));
        watch
    }

    fn userdata(&self) -> *mut c_void {
        self as *const Watch as *mut c_void
    }

    fn stages(&self) -> Vec<String> {
        self.stages.lock().unwrap().clone()
    }
}

unsafe extern "C" fn watch_callback(
    userdata: *mut c_void,
    kind: c_int,
    text: *const c_char,
    len: usize,
    _done: u64,
    _total: u64,
) {
    let watch = &*(userdata as *const Watch);
    // Byte updates arrive from worker threads.
    if kind != TS_INSTALL_STAGE {
        watch.bytes.fetch_add(1, Ordering::SeqCst);
        return;
    }
    let line =
        String::from_utf8_lossy(std::slice::from_raw_parts(text as *const u8, len)).into_owned();
    watch.stages.lock().unwrap().push(line.clone());
    let mut trigger = watch.trigger.lock().unwrap();
    if trigger
        .as_ref()
        .is_some_and(|(needle, _)| line.contains(needle.as_str()))
    {
        let (_, action) = trigger.take().unwrap();
        // Released first: the action may call back into the ABI.
        drop(trigger);
        action();
    }
}

// ----------------------------------------------------------- the fixture Hub

/// One repository the fixture Hub answers for.
struct FixtureRepo {
    id: String,
    /// The revisions its files are served under, a floating one among them.
    revisions: Vec<String>,
    /// The immutable commit its metadata names.
    sha: String,
    files: BTreeMap<String, Vec<u8>>,
}

/// Answers Hub metadata and file requests by path, counting every connection
/// and logging every request line. Also the process's Hugging Face endpoint
/// until it is dropped.
struct Hub {
    connections: Arc<AtomicUsize>,
    requests: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Hub {
    fn start(repos: Vec<FixtureRepo>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("fixture bind");
        listener.set_nonblocking(true).expect("nonblocking hub");
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let connections = Arc::new(AtomicUsize::new(0));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (counted, logged, stopped) = (
            Arc::clone(&connections),
            Arc::clone(&requests),
            Arc::clone(&stop),
        );
        let thread = thread::spawn(move || {
            while !stopped.load(Ordering::Acquire) {
                let Ok((mut stream, _)) = listener.accept() else {
                    thread::sleep(Duration::from_millis(2));
                    continue;
                };
                counted.fetch_add(1, Ordering::AcqRel);
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request = String::new();
                if reader.read_line(&mut request).is_err() {
                    continue;
                }
                let mut range = None;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
                        break;
                    }
                    if let Some((name, value)) = line.trim_end().split_once(':') {
                        if name.eq_ignore_ascii_case("range") {
                            range = Some(value.trim().to_string());
                        }
                    }
                }
                let mut parts = request.split_whitespace();
                let method = parts.next().unwrap_or("GET").to_string();
                let target = parts.next().unwrap_or("/").to_string();
                logged.lock().unwrap().push(format!("{method} {target}"));
                let _ = stream.write_all(&answer(&method, &target, range.as_deref(), &repos));
            }
        });
        catalog::set_hf_endpoint_override(Some(endpoint));
        Self {
            connections,
            requests,
            stop,
            thread: Some(thread),
        }
    }

    /// Connections so far. Called after the entry point returned, by which
    /// time any connection it made is in the listener's queue or counted; the
    /// short wait lets the accept loop drain the queue.
    fn connections(&self) -> usize {
        thread::sleep(Duration::from_millis(50));
        self.connections.load(Ordering::Acquire)
    }

    fn requests(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }
}

impl Drop for Hub {
    fn drop(&mut self) {
        catalog::set_hf_endpoint_override(None);
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn answer(method: &str, target: &str, range: Option<&str>, repos: &[FixtureRepo]) -> Vec<u8> {
    let path = target.split('?').next().unwrap_or(target);
    // Hub repository ids are case-insensitive.
    let lowered = path.to_ascii_lowercase();
    for repo in repos {
        let id = repo.id.to_ascii_lowercase();
        if lowered.starts_with(&format!("/api/models/{id}/")) {
            let siblings: Vec<_> = repo
                .files
                .iter()
                .map(|(name, bytes)| {
                    json!({
                        "rfilename": name,
                        "size": bytes.len() as u64,
                        "blobId": "a".repeat(40),
                    })
                })
                .collect();
            let body = json!({"id": repo.id, "sha": repo.sha, "siblings": siblings}).to_string();
            return response(200, body.as_bytes(), &[]);
        }
        for revision in &repo.revisions {
            let prefix = format!("/{id}/resolve/{}/", revision.to_ascii_lowercase());
            if !lowered.starts_with(&prefix) {
                continue;
            }
            let Some(bytes) = repo.files.get(&path[prefix.len()..]) else {
                return response(404, b"", &[]);
            };
            if method.eq_ignore_ascii_case("HEAD") {
                return response(200, b"", &[("x-linked-size", bytes.len().to_string())]);
            }
            let Some((start, end)) = range
                .and_then(|range| range.strip_prefix("bytes="))
                .and_then(|range| range.split_once('-'))
            else {
                return response(200, bytes, &[]);
            };
            let start: usize = start.parse().unwrap();
            let end = end.parse::<usize>().unwrap().min(bytes.len() - 1);
            return response(
                206,
                &bytes[start..=end],
                &[(
                    "Content-Range",
                    format!("bytes {start}-{end}/{}", bytes.len()),
                )],
            );
        }
    }
    response(404, b"", &[])
}

fn response(status: u16, body: &[u8], headers: &[(&str, String)]) -> Vec<u8> {
    let mut wire = format!(
        "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n",
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

// ------------------------------------------------------------ fixture content

/// A synthetic GGUF, padded past the probe's first 1 MiB header read as every
/// real checkpoint is: a shorter file answers that read short, and the reader
/// then backs off for seconds before asking for the size.
#[cfg(target_os = "macos")]
fn gguf() -> Vec<u8> {
    let (mut gguf, _) = repack::build_synthetic_gemma4_gguf(repack::SyntheticGgufShape::default());
    gguf.resize(gguf.len().max(1 << 21), 0);
    gguf
}

/// The ChatML tokenizer files every text install fetches.
#[cfg(target_os = "macos")]
fn tokenizer_files() -> BTreeMap<String, Vec<u8>> {
    let dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../tokenizer/tests/fixtures/ChatMLTokenizer");
    ["tokenizer.json", "tokenizer_config.json"]
        .into_iter()
        .map(|name| (name.to_string(), fs::read(dir.join(name)).unwrap()))
        .collect()
}

/// `owner/model` at [`REVISION`]: the synthetic GGUF and its tokenizer.
#[cfg(target_os = "macos")]
fn owner_model() -> FixtureRepo {
    let mut files = tokenizer_files();
    files.insert("model-Q8_0.gguf".to_string(), gguf());
    FixtureRepo {
        id: "owner/model".to_string(),
        revisions: vec![REVISION.to_string()],
        sha: REVISION.to_string(),
        files,
    }
}

/// The repositories the curated row `alias` names, served with the synthetic
/// GGUF and tokenizer under the row's own file names, so the row installs
/// offline through its real plan.
#[cfg(target_os = "macos")]
fn curated_repos(alias: &str) -> Vec<FixtureRepo> {
    let catalog = catalog::Catalog::embedded().unwrap();
    let plan = catalog::InstallPlan::from_entry(catalog.get(alias).unwrap());
    let sha = "b".repeat(40);
    let revisions = |revision: &str| vec![revision.to_string(), sha.clone()];
    let mut weights = BTreeMap::new();
    weights.insert(plan.file.clone().unwrap(), gguf());
    let mut sidecars = tokenizer_files();
    sidecars.retain(|name, _| plan.sidecar_files.contains(name));
    assert_eq!(sidecars.len(), plan.sidecar_files.len(), "{plan:?}");
    vec![
        FixtureRepo {
            id: plan.weights.repo.clone(),
            revisions: revisions(&plan.weights.revision),
            sha: sha.clone(),
            files: weights,
        },
        FixtureRepo {
            id: plan.sidecars.repo.clone(),
            revisions: revisions(&plan.sidecars.revision),
            sha: sha.clone(),
            files: sidecars,
        },
    ]
}

/// A safetensors file of F32 tensors, each `[64]` or `[1, 64]` of 0.25.
fn safetensors(tensors: &[(&str, &[u64])]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for (name, shape) in tensors {
        let count: u64 = shape.iter().product();
        let start = payload.len() as u64;
        for _ in 0..count {
            payload.extend_from_slice(&0.25f32.to_le_bytes());
        }
        header.insert(
            name.to_string(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [start, payload.len() as u64]}),
        );
    }
    let header = serde_json::to_vec(&Value::Object(header)).unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.extend_from_slice(&payload);
    file
}

/// The repository of the curated image row [`IMAGE`], with a tiny Diffusers
/// export under the row's own pinned file names, so the row installs offline
/// through its real download and packing.
fn image_repo() -> FixtureRepo {
    let catalog = catalog::ImageCatalog::embedded().unwrap();
    let entry = catalog.get(IMAGE).unwrap();
    let mut files = BTreeMap::new();
    for name in &entry.required_files {
        let bytes = match name.as_str() {
            "scheduler/scheduler_config.json" => b"{\"shift\": 3.0}".to_vec(),
            "tokenizer/tokenizer_config.json" => b"{\"chat_template\": \"test\"}".to_vec(),
            "tokenizer/tokenizer.json" => b"tokenizer".to_vec(),
            "tokenizer/merges.txt" | "tokenizer/vocab.json" => b"x".to_vec(),
            name if name.ends_with("/config.json") => b"{}".to_vec(),
            name if name.ends_with(".safetensors.index.json") => {
                // The index names a shard by its file name in the component.
                let first = name
                    .rsplit('/')
                    .next()
                    .unwrap()
                    .replace(".safetensors.index.json", "-00001-of-00003.safetensors");
                serde_json::to_vec(&json!({
                    "weight_map": {"linear.weight": first, "norm.weight": first}
                }))
                .unwrap()
            }
            name if name.ends_with("-00001-of-00003.safetensors")
                || name == "vae/diffusion_pytorch_model.safetensors" =>
            {
                safetensors(&[("linear.weight", &[1, 64]), ("norm.weight", &[64])])
            }
            // The index maps nothing into the other shards.
            name if name.ends_with(".safetensors") => safetensors(&[]),
            other => panic!("no fixture content for {other}"),
        };
        files.insert(name.clone(), bytes);
    }
    FixtureRepo {
        id: entry.model_id.clone(),
        revisions: vec![entry.revision.clone()],
        sha: entry.revision.clone(),
        files,
    }
}

// --------------------------------------------------------------- entry points

fn opt(text: &Option<CString>) -> *const c_char {
    text.as_ref().map_or(ptr::null(), |text| text.as_ptr())
}

/// `{"library": id}`.
fn library(id: &str) -> String {
    json!({ "library": id }).to_string()
}

fn install_curated(alias: &str, options: Option<&str>, watch: &Watch) -> Reply {
    let (alias, options) = (c(alias), options.map(c));
    reply(|out| unsafe {
        ts_install_with_options(
            alias.as_ptr(),
            opt(&options),
            Some(watch_callback),
            watch.userdata(),
            out,
        )
    })
}

fn install_repo(alias: &str, options: Option<&str>, watch: &Watch) -> Reply {
    let repo = c(&format!("owner/model@{REVISION}"));
    let (alias, file, options) = (c(alias), c("model-Q8_0.gguf"), options.map(c));
    reply(|out| unsafe {
        ts_install_repo_with_options(
            repo.as_ptr(),
            alias.as_ptr(),
            file.as_ptr(),
            ptr::null(),
            opt(&options),
            Some(watch_callback),
            watch.userdata(),
            out,
        )
    })
}

fn install_image(alias: &str, options: Option<&str>, watch: &Watch) -> Reply {
    let (alias, options) = (c(alias), options.map(c));
    reply(|out| unsafe {
        ts_image_install_with_options(
            alias.as_ptr(),
            opt(&options),
            Some(watch_callback),
            watch.userdata(),
            out,
        )
    })
}

/// The identity of the audio catalog profile [`MUSIC`], as JSON.
fn music_identity() -> CString {
    let catalog = reply(|out| unsafe { ts_audio_catalog_json(out) })
        .ok()
        .json();
    let row = catalog
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["identity"]["alias"] == MUSIC)
        .unwrap_or_else(|| panic!("no audio profile {MUSIC}"));
    c(&row["identity"].to_string())
}

fn install_audio(options: Option<&str>, watch: &Watch) -> Reply {
    let (identity, options) = (music_identity(), options.map(c));
    reply(|out| unsafe {
        ts_audio_install_with_options(
            identity.as_ptr(),
            opt(&options),
            Some(watch_callback),
            watch.userdata(),
            out,
        )
    })
}

/// Every install entry point with a destination, asked for the same
/// `options`, labelled. The alias of the repository install is `mine`.
fn every_entry_point(options: &str) -> Vec<(&'static str, Reply)> {
    let watch = Watch::new();
    vec![
        (
            "ts_install_with_options",
            install_curated(TEXT, Some(options), &watch),
        ),
        (
            "ts_install_repo_with_options",
            install_repo("mine", Some(options), &watch),
        ),
        (
            "ts_image_install_with_options",
            install_image(IMAGE, Some(options), &watch),
        ),
        (
            "ts_audio_install_with_options",
            install_audio(Some(options), &watch),
        ),
    ]
}

// ------------------------------------------------------------------ landings

/// An options install of a curated row lands in the library: the folder, the
/// row, the marker entry and the listing are all there, and nothing is in the
/// home. The result carries the library keys `ts_installed_json` rows carry.
#[cfg(target_os = "macos")]
#[test]
fn a_curated_text_install_with_options_lands_in_the_requested_library() {
    let fx = Fixture::new("curated");
    let id = fx.add_library("T7");
    let hub = Hub::start(curated_repos(TEXT));
    let watch = Watch::new();

    let installed = install_curated(TEXT, Some(&library(&id)), &watch)
        .ok()
        .json();

    let dir = fx.root("T7").join("text").join("tinyllama.gturbo");
    assert!(dir.join("model_weights.bin").is_file());
    assert_eq!(installed["alias"], TEXT);
    assert_eq!(installed["path"], dir.to_str().unwrap());
    assert_eq!(installed["library"], id);
    assert_eq!(installed["library_name"], "T7");
    assert_eq!(installed["install_id"], format!("{id}:text/{TEXT}"));
    assert_eq!(installed["relative_path"], "text/tinyllama.gturbo");
    // The registry row and the marker entry.
    let rows = fx.recorded();
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(rows[0]["library"], id);
    assert_eq!(rows[0]["relative_path"], "text/tinyllama.gturbo");
    assert_eq!(fx.marker_aliases("T7"), [TEXT]);
    // Listed in that library, and nothing landed in the home's models.
    let listed = rows_of(reply(|out| unsafe { ts_installed_json(out) }));
    let found = listed.iter().find(|row| row["alias"] == TEXT).unwrap();
    assert_eq!(found["location_kind"], "library", "{found}");
    assert_eq!(found["availability"], "available", "{found}");
    assert_eq!(found["install_id"], installed["install_id"]);
    assert!(!fx.home.join("models").join("text").exists());
    // It went through the Hub, reporting stages and bytes.
    assert!(hub.connections() > 0);
    assert!(!watch.stages().is_empty());
    assert!(watch.bytes.load(Ordering::SeqCst) > 0);
}

/// The repository variant lands in the library the same way, and a vision
/// tower or a second alias is not needed to see it.
#[cfg(target_os = "macos")]
#[test]
fn a_repo_install_with_options_lands_in_the_requested_library() {
    let fx = Fixture::new("repo");
    let id = fx.add_library("T7");
    let hub = Hub::start(vec![owner_model()]);
    let watch = Watch::new();

    let installed = install_repo("mine", Some(&library(&id)), &watch)
        .ok()
        .json();

    let dir = fx.root("T7").join("text").join("mine.gturbo");
    assert!(dir.join("model_weights.bin").is_file());
    assert_eq!(installed["path"], dir.to_str().unwrap());
    assert_eq!(installed["library"], id);
    assert_eq!(installed["library_name"], "T7");
    assert_eq!(installed["install_id"], format!("{id}:text/mine"));
    let rows = fx.recorded();
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(rows[0]["alias"], "mine");
    assert_eq!(rows[0]["library"], id);
    assert_eq!(fx.marker_aliases("T7"), ["mine"]);
    assert!(!fx.home.join("models").join("text").exists());
    assert!(hub.connections() > 0);
    assert!(!watch.stages().is_empty());
}

/// No `library`, an absent or empty options string, `null`, and "internal"
/// all mean This Mac, as the entry points without options always did, and the
/// originals still install there.
#[cfg(target_os = "macos")]
#[test]
fn an_install_without_a_library_installs_into_this_mac() {
    let fx = Fixture::new("this-mac");
    fx.add_library("T7");
    let hub = Hub::start({
        let mut repos = curated_repos(TEXT);
        repos.push(owner_model());
        repos
    });
    let watch = Watch::new();

    for (alias, options) in [
        ("a", None),
        ("b", Some("")),
        ("c", Some("{}")),
        ("d", Some(r#"{"library": null}"#)),
        ("e", Some(r#"{"library": "internal"}"#)),
    ] {
        let installed = install_repo(alias, options, &watch).ok().json();
        let dir = fx.home.join("models/text").join(format!("{alias}.gturbo"));
        assert!(dir.join("model_weights.bin").is_file(), "{options:?}");
        assert_eq!(
            installed["path"],
            dir.canonicalize().unwrap().to_str().unwrap()
        );
        assert_eq!(installed["library"], "internal", "{options:?}");
        assert_eq!(installed["library_name"], "This Mac");
        assert_eq!(installed["install_id"], format!("internal:text/{alias}"));
    }
    // The row of a This Mac install carries no library of its own.
    let rows = fx.recorded();
    assert_eq!(rows.len(), 5, "{rows:?}");
    assert!(
        rows.iter().all(|row| row.get("library").is_none()),
        "{rows:?}"
    );
    assert!(fx.marker_aliases("T7").is_empty());
    assert!(!fx.root("T7").join("text").exists());

    // The original entry point, which has no destination, still installs there.
    let curated = {
        let alias = c(TEXT);
        reply(|out| unsafe {
            turbospark_ffi::ts_install(alias.as_ptr(), Some(watch_callback), watch.userdata(), out)
        })
        .ok()
        .json()
    };
    let dir = fx.home.join("models/text/tinyllama.gturbo");
    assert!(dir.join("model_weights.bin").is_file());
    assert_eq!(curated["library"], "internal");
    assert_eq!(curated["install_id"], "internal:text/tinyllama");
    assert!(hub.connections() > 0);
}

/// An image install lands in the library's image folder, built through its
/// staging folder and published there, and is listed with its library.
#[test]
fn an_image_install_with_options_lands_in_the_requested_library() {
    let fx = Fixture::new("image");
    let id = fx.add_library("T7");
    let hub = Hub::start(vec![image_repo()]);
    let watch = Watch::new();

    let installed = install_image(IMAGE, Some(&library(&id)), &watch)
        .ok()
        .json();

    let dir = fx.root("T7").join("image").join("z-image-turbo.gturbo");
    assert!(dir.join("manifest.json").is_file(), "{installed}");
    assert_eq!(installed["alias"], IMAGE);
    assert_eq!(installed["path"], dir.to_str().unwrap());
    assert_eq!(installed["library"], id);
    assert_eq!(installed["libraryName"], "T7");
    assert_eq!(installed["installId"], format!("{id}:image/{IMAGE}"));
    // Staging is left empty, and the home's models folder is untouched.
    assert_eq!(
        fs::read_dir(fx.root("T7").join(".staging"))
            .unwrap()
            .count(),
        0
    );
    assert!(!fx.home.join("models").exists());
    // Listed under that library.
    let listed = rows_of(reply(|out| unsafe { ts_image_installed_json(out) }));
    assert_eq!(listed.len(), 1, "{listed:?}");
    assert_eq!(listed[0]["library"], id);
    assert_eq!(listed[0]["installId"], installed["installId"]);
    assert!(hub.connections() > 0);
    assert!(!watch.stages().is_empty());

    // A second install of it is refused as already installed, not rebuilt.
    let again = install_image(IMAGE, Some(&library(&id)), &watch);
    assert_ne!(again.code, abi::TS_OK);
    assert!(
        again.error.contains("already holds an image install"),
        "{}",
        again.error
    );
}

/// An image install with no library is the install this entry point's
/// original makes: This Mac's image folder, and `internal` in the result.
#[test]
fn an_image_install_with_no_library_installs_into_this_mac() {
    let fx = Fixture::new("image-this-mac");
    fx.add_library("T7");
    let _hub = Hub::start(vec![image_repo()]);
    let watch = Watch::new();

    let installed = install_image(IMAGE, None, &watch).ok().json();

    let dir = fx.home.join("models/image/z-image-turbo.gturbo");
    assert!(dir.join("manifest.json").is_file(), "{installed}");
    assert_eq!(installed["library"], "internal");
    assert_eq!(installed["libraryName"], "This Mac");
    assert_eq!(installed["installId"], format!("internal:image/{IMAGE}"));
    assert!(!fx.root("T7").join("image").exists());
}

// ------------------------------------------------------------------- refusals

/// The destination is settled before the first request, for every entry
/// point: a library that is not connected is `not_connected`, and the Hub is
/// never asked, nor is the drive's folder rebuilt under its mount point.
#[test]
fn a_library_that_is_not_connected_is_refused_before_any_request() {
    let fx = Fixture::new("not-connected");
    let id = fx.add_library("T7");
    fx.unplug("T7");
    let hub = Hub::start(Vec::new());

    for (entry, reply) in every_entry_point(&library(&id)) {
        assert_refused(&reply, "not_connected", entry);
        assert!(reply.error.contains("T7"), "{entry}: {}", reply.error);
    }

    assert_eq!(hub.connections(), 0, "a refusal reached the network");
    assert!(hub.requests().is_empty());
    assert!(!fx.volume("T7").exists());
    assert!(fx.recorded().is_empty());
}

/// A library a newer TurboSpark wrote is `library_read_only`, before any
/// request, and nothing is written to it.
#[test]
fn a_read_only_library_is_refused_before_any_request() {
    let fx = Fixture::new("read-only");
    let id = fx.add_library("T7");
    fx.mark_newer("T7");
    let hub = Hub::start(Vec::new());
    let before = fs::read(fx.root("T7").join(MARKER_FILE)).unwrap();

    for (entry, reply) in every_entry_point(&library(&id)) {
        assert_refused(&reply, "library_read_only", entry);
    }

    assert_eq!(hub.connections(), 0, "a refusal reached the network");
    assert_eq!(fs::read(fx.root("T7").join(MARKER_FILE)).unwrap(), before);
    for folder in ["text", "image", "audio", ".staging"] {
        assert!(!fx.root("T7").join(folder).exists(), "{folder}");
    }
    assert!(fx.recorded().is_empty());
}

/// An id the library list does not have is `unknown_library`, also before any
/// request, however it is spelled: absent from the list, or empty.
#[test]
fn an_unknown_library_is_refused_before_any_request() {
    let fx = Fixture::new("unknown");
    fx.add_library("T7");
    let hub = Hub::start(Vec::new());

    for id in ["5a1d9c70-3e2b-4f68-8a14-6b0c2d9e7f31", ""] {
        for (entry, reply) in every_entry_point(&library(id)) {
            assert_refused(&reply, "unknown_library", entry);
        }
    }

    assert_eq!(hub.connections(), 0, "a refusal reached the network");
    assert!(fx.recorded().is_empty());
}

/// An alias another library, or This Mac, already holds is refused with its
/// own code, `alias_elsewhere`, so the app can offer a move, also before any
/// request.
#[test]
fn an_alias_held_by_another_library_is_refused_before_any_request() {
    let fx = Fixture::new("alias-elsewhere");
    let id = fx.add_library("T7");
    let hub = Hub::start(Vec::new());
    // This Mac holds each alias, as a recorded row or an image folder.
    let text = fx.home.join("models").join("text");
    let rows: Vec<Value> = [TEXT, "mine"]
        .iter()
        .map(|alias| {
            json!({
                "alias": alias, "repo": "owner/x", "revision": "main",
                "path": text.join(format!("{alias}.gturbo")), "family": "qwen3",
                "install_bytes": 1, "installed_on": "2026-10-01", "status": "runs",
                "kind": null, "modality": "text",
            })
        })
        .chain([json!({
            "alias": MUSIC, "repo": "owner/x", "revision": "main",
            "path": fx.home.join("models/audio").join(format!("{MUSIC}.gturbo")),
            "family": "minimax_music3", "install_bytes": 1, "installed_on": "2026-10-01",
            "status": "unverified", "kind": null, "modality": "audio",
        })])
        .collect();
    fs::write(
        fx.home.join("installed.json"),
        serde_json::to_vec(&rows).unwrap(),
    )
    .unwrap();
    fs::create_dir_all(fx.home.join("models/image").join(format!("{IMAGE}.gturbo"))).unwrap();

    for (entry, reply) in every_entry_point(&library(&id)) {
        assert_refused(&reply, "alias_elsewhere", entry);
        assert!(
            reply.error.contains("already installed in This Mac"),
            "{entry}: {}",
            reply.error
        );
    }

    assert_eq!(hub.connections(), 0, "a refusal reached the network");
    assert!(fx.marker_aliases("T7").is_empty());
}

/// Options are parsed strictly, as the audio options always were: a misspelt
/// key must not install into This Mac, a wrong type is an error, and nothing
/// is requested or written.
#[test]
fn malformed_options_are_refused_before_any_work() {
    let fx = Fixture::new("options");
    fx.add_library("T7");
    let hub = Hub::start(Vec::new());

    for bad in [
        "not json",
        r#"{"libary": "x"}"#,
        r#"{"library": 3}"#,
        r#"{"library": ["a"]}"#,
        r#"["a"]"#,
    ] {
        for (entry, reply) in every_entry_point(bad) {
            assert_eq!(
                reply.code,
                abi::TS_ERR_INVALID_ARGUMENT,
                "{entry}: {bad}: {}",
                reply.error
            );
            // Refused as options, not read as a destination: serde would take
            // the array `["a"]` for the fields in order.
            assert!(
                reply
                    .error
                    .starts_with("options is not valid install options"),
                "{entry}: {bad}: {}",
                reply.error
            );
            assert!(reply.text.is_none(), "{entry}: {bad} left an output");
        }
    }

    // A null alias and a null result pointer are refused up front too.
    let watch = Watch::new();
    let options = c(r#"{"library": "internal"}"#);
    let mut out: *mut c_char = ptr::null_mut();
    for code in unsafe {
        [
            ts_install_with_options(
                ptr::null(),
                options.as_ptr(),
                None,
                ptr::null_mut(),
                &mut out,
            ),
            ts_image_install_with_options(
                ptr::null(),
                options.as_ptr(),
                None,
                ptr::null_mut(),
                &mut out,
            ),
            ts_install_with_options(
                c(TEXT).as_ptr(),
                options.as_ptr(),
                None,
                watch.userdata(),
                ptr::null_mut(),
            ),
            ts_image_install_with_options(
                c(IMAGE).as_ptr(),
                options.as_ptr(),
                None,
                watch.userdata(),
                ptr::null_mut(),
            ),
        ]
    } {
        assert_eq!(code, abi::TS_ERR_INVALID_ARGUMENT);
    }
    assert!(out.is_null());
    assert_eq!(hub.connections(), 0);
    assert!(fx.recorded().is_empty());
}

// ------------------------------------------------------------ mid-install loss

/// A drive that disappears while the walk runs is `library_disconnected`:
/// the walk fails on the missing drive, nothing is recorded, and the drive's
/// folders are not rebuilt.
#[cfg(target_os = "macos")]
#[test]
fn a_drive_lost_during_a_text_install_is_library_disconnected() {
    let fx = Fixture::new("lost-text");
    let id = fx.add_library("T7");
    let hub = Hub::start(vec![owner_model()]);
    let (volume, volumes) = (fx.volume("T7"), fx.volumes.clone());
    // The walk's first stage line, after it made its folder on the drive.
    let watch = Watch::at("installing mine from", move || {
        fs::remove_dir_all(&volume).unwrap();
        fs::set_permissions(&volumes, fs::Permissions::from_mode(0o555)).unwrap();
    });

    let lost = install_repo("mine", Some(&library(&id)), &watch);

    assert_refused(
        &lost,
        "library_disconnected",
        "ts_install_repo_with_options",
    );
    assert!(lost.error.contains("T7"), "{}", lost.error);
    assert!(!fx.volume("T7").exists(), "the drive was rebuilt");
    assert!(fx.recorded().is_empty(), "{:?}", fx.recorded());
    assert!(hub.connections() > 0, "the drive was lost before the walk");
}

/// The same for an image install, lost while its source downloads into the
/// library's staging folder.
#[test]
fn a_drive_lost_during_an_image_install_is_library_disconnected() {
    let fx = Fixture::new("lost-image");
    let id = fx.add_library("T7");
    let hub = Hub::start(vec![image_repo()]);
    let (volume, volumes) = (fx.volume("T7"), fx.volumes.clone());
    let watch = Watch::at("downloading ", move || {
        fs::remove_dir_all(&volume).unwrap();
        fs::set_permissions(&volumes, fs::Permissions::from_mode(0o555)).unwrap();
    });

    let lost = install_image(IMAGE, Some(&library(&id)), &watch);

    assert_refused(
        &lost,
        "library_disconnected",
        "ts_image_install_with_options",
    );
    assert!(lost.error.contains("T7"), "{}", lost.error);
    assert!(!fx.volume("T7").exists(), "the drive was rebuilt");
    assert!(
        hub.connections() > 0,
        "the drive was lost before the download"
    );
}

// ------------------------------------------------------- the running-install registry

/// An options install is registered with the walks `ts_install_cancel` and the
/// library changes look at: while it runs a library change is `busy`, and a
/// cancel ends it as cancelled with nothing recorded.
#[cfg(target_os = "macos")]
#[test]
fn a_text_install_with_options_is_registered_as_running_and_can_be_cancelled() {
    let fx = Fixture::new("cancel-text");
    let id = fx.add_library("T7");
    let _hub = Hub::start(vec![owner_model()]);
    let busy = Arc::new(AtomicI32::new(-1));
    let cancelled = Arc::new(AtomicI32::new(-1));
    let (during, cancel_saw) = (Arc::clone(&busy), Arc::clone(&cancelled));
    let removed = c(&id);
    let watch = Watch::at("installing mine from", move || {
        during.store(
            unsafe { ts_library_remove(removed.as_ptr()) },
            Ordering::SeqCst,
        );
        cancel_saw.store(unsafe { ts_install_cancel() }, Ordering::SeqCst);
    });

    let ended = install_repo("mine", Some(&library(&id)), &watch);

    assert_eq!(
        busy.load(Ordering::SeqCst),
        abi::TS_ERR_BUSY,
        "the walk was not registered"
    );
    assert_eq!(cancelled.load(Ordering::SeqCst), 1, "no walk was cancelled");
    assert_eq!(ended.code, abi::TS_ERR_CANCELLED, "{}", ended.error);
    assert!(ended.text.is_none());
    assert!(fx.recorded().is_empty(), "{:?}", fx.recorded());
    assert!(fx.marker_aliases("T7").is_empty());
}

/// The same for an image install, whose walk registers apart from the text
/// walks but in the same registry.
#[test]
fn an_image_install_with_options_is_registered_as_running_and_can_be_cancelled() {
    let fx = Fixture::new("cancel-image");
    let id = fx.add_library("T7");
    let _hub = Hub::start(vec![image_repo()]);
    let busy = Arc::new(AtomicI32::new(-1));
    let cancelled = Arc::new(AtomicI32::new(-1));
    let (during, cancel_saw) = (Arc::clone(&busy), Arc::clone(&cancelled));
    let removed = c(&id);
    let watch = Watch::at("preparing image source", move || {
        during.store(
            unsafe { ts_library_remove(removed.as_ptr()) },
            Ordering::SeqCst,
        );
        cancel_saw.store(unsafe { ts_install_cancel() }, Ordering::SeqCst);
    });

    let ended = install_image(IMAGE, Some(&library(&id)), &watch);

    assert_eq!(
        busy.load(Ordering::SeqCst),
        abi::TS_ERR_BUSY,
        "the walk was not registered"
    );
    assert_eq!(cancelled.load(Ordering::SeqCst), 1, "no walk was cancelled");
    assert_eq!(ended.code, abi::TS_ERR_CANCELLED, "{}", ended.error);
    assert!(!fx
        .root("T7")
        .join("image")
        .join("z-image-turbo.gturbo")
        .exists());
}

// --------------------------------------------------------------------- header

fn squash(text: &str) -> String {
    text.split_whitespace().collect()
}

/// The parameter types the Rust export `name` takes in `source`, as C spells
/// them, whitespace removed.
fn exported(source: &str, name: &str) -> Vec<String> {
    let start = format!("pub unsafe extern \"C\" fn {name}(");
    let at = source
        .find(&start)
        .unwrap_or_else(|| panic!("{name} is not exported"));
    let rest = &source[at + start.len()..];
    let close = rest.find(") -> c_int").unwrap();
    rest[..close]
        .split(',')
        .map(str::trim)
        .filter(|param| !param.is_empty())
        .map(|param| {
            let rust = param.split_once(':').unwrap().1.trim();
            let c_type = match rust {
                "*const c_char" => "const char *",
                "*mut *mut c_char" => "char **",
                "*mut c_void" => "void *",
                "TsInstallCallback" | "crate::TsInstallCallback" => "TsInstallCallback",
                other => panic!("{name}: no C spelling known for {other}"),
            };
            squash(c_type)
        })
        .collect()
}

/// The parameter types the header declares for `<returns> <name>(...)`.
fn declared(header: &str, returns: &str, name: &str) -> Option<Vec<String>> {
    let from = header.find(&format!("{returns} {name}("))?;
    let rest = &header[from..];
    let open = rest.find('(').unwrap();
    let close = rest.find(");").unwrap();
    Some(
        rest[open + 1..close]
            .split(',')
            .map(str::trim)
            .filter(|param| !param.is_empty())
            .map(|param| {
                // Drop the parameter's name, keep its type.
                let name_at = param
                    .rfind(|ch: char| !(ch.is_alphanumeric() || ch == '_'))
                    .map_or(0, |at| at + 1);
                squash(&param[..name_at])
            })
            .collect(),
    )
}

/// Every install entry point that takes a destination is declared in the
/// header with the parameters it exports, and the header documents the
/// option and the codes the app maps.
#[test]
fn the_header_declares_the_install_destination_exports() {
    let header = include_str!("../include/turbospark.h");
    let read =
        |file: &str| fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(file)).unwrap();
    let (models, audio) = (read("src/api/models.rs"), read("src/api/audio.rs"));
    for (source, returns, name) in [
        (&models, "int32_t", "ts_install_with_options"),
        (&models, "int32_t", "ts_install_repo_with_options"),
        (&models, "int32_t", "ts_image_install_with_options"),
        (&audio, "int", "ts_audio_install_with_options"),
    ] {
        let in_header = declared(header, returns, name)
            .unwrap_or_else(|| panic!("{name} is not declared in the header"));
        assert_eq!(
            in_header,
            exported(source, name),
            "{name}: the header and the export disagree"
        );
    }

    // The documented codes and the option name, in the library block.
    let block = &header[header.find("---- model libraries").unwrap()..];
    let block = &block[..block.find("int32_t ts_libraries_json(").unwrap()];
    for needle in [
        "options_json",
        "\"library\"",
        "unknown_library",
        "library_disconnected",
        "library_read_only",
        "home_only",
        "not_connected",
    ] {
        assert!(
            block.contains(needle),
            "the library block never says {needle}"
        );
    }
}

// ------------------------------------------------------ links on the drive

/// Plants links in the folder a text install of `alias` is about to use in
/// the library T7, as a cancelled install or anyone who can write the drive
/// could leave them, installs through the options entry point, and checks
/// nothing outside the library was written.
#[cfg(target_os = "macos")]
fn install_over_planted_links(tag: &str, curated: bool) {
    let fx = Fixture::new(tag);
    let id = fx.add_library("T7");
    let alias = if curated { TEXT } else { "mine" };
    let dir = fx.root("T7").join("text").join(format!("{alias}.gturbo"));
    let outside = fx.base.join("outside");
    let victim_dir = outside.join("victim-dir");
    fs::create_dir_all(&victim_dir).unwrap();
    let victim = outside.join("victim.txt");
    fs::write(&victim, b"keep").unwrap();
    fs::create_dir_all(&dir).unwrap();
    let mut files = vec![
        "tokenizer.json",
        "tokenizer_config.json",
        "model_weights.bin",
    ];
    if curated {
        // A repository install refuses a manifest it finds, as "already
        // holds an install", so only the curated path gets one.
        files.push("manifest.json");
    }
    for name in files {
        std::os::unix::fs::symlink(&victim, dir.join(name)).unwrap();
    }
    for name in ["packed_experts", ".download-cache"] {
        std::os::unix::fs::symlink(&victim_dir, dir.join(name)).unwrap();
    }
    let hub = Hub::start(if curated {
        curated_repos(TEXT)
    } else {
        vec![owner_model()]
    });
    let watch = Watch::new();
    let options = library(&id);

    let installed = if curated {
        install_curated(TEXT, Some(&options), &watch)
    } else {
        install_repo("mine", Some(&options), &watch)
    }
    .ok()
    .json();

    assert_eq!(installed["library"], id);
    assert_eq!(
        fs::read(&victim).unwrap(),
        b"keep",
        "{tag}: a link was written through"
    );
    assert_eq!(
        fs::read_dir(&victim_dir).unwrap().count(),
        0,
        "{tag}: a folder outside the library was written into"
    );
    let weights = fs::symlink_metadata(dir.join("model_weights.bin")).unwrap();
    assert!(
        weights.file_type().is_file(),
        "{tag}: the weights are a plain file"
    );
    drop(hub);
}

/// A leftover folder with links in it must not make a text install write
/// outside the library, through either entry point (found by the review of
/// 4.3). The folder is cleared, links unlinked rather than followed.
#[cfg(target_os = "macos")]
#[test]
fn links_planted_in_a_library_install_folder_are_not_written_through() {
    install_over_planted_links("links-curated", true);
    install_over_planted_links("links-repo", false);
}
