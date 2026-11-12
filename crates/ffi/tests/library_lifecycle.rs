//! The library lifecycle end to end, through the C ABI and without hardware
//! (task 7.1).
//!
//! One scenario: add a library on a fake volume, install a model into it,
//! rename the volume, unplug it, plug it back under another name, remove the
//! library, and add the same folder again. After every step the listings
//! (`ts_libraries_json`, `ts_installed_json`) are checked against the step
//! table below, and so is the registry on disk. The install must keep one
//! install id through all of it (4.2), removing the library must leave every
//! file on the drive as it was (1.8), and adding the folder again must bring
//! the rows back from the drive's marker with the same ids (1.6). A second
//! drive, `Decoy`, is mounted throughout with a library of its own that is
//! never added, so finding the renamed drive again must go by the marker's id
//! and not by the shape of the folder.
//!
//! `crates/cli/tests/library_lifecycle.rs` runs the same scenario through the
//! real `turbospark-model` binary against the same table. The two crates are
//! separate and the CLI crate may not depend on this one, so agreement
//! between them is established by two tests over one scenario, not by one
//! process calling both. `the_step_table_is_the_command_lines` fails when the
//! text between the table markers differs from the CLI test's, so edit both.
//! As a cheap second check, every listing here is also compared with the
//! catalog layer's own (`LibraryRegistry::refresh`,
//! `Store::installed_with_availability`), which is what the command line
//! renders from.
//!
//! The location rules read the mount table and the home folder's device.
//! Debug builds take both from the `TURBOSPARK_TEST_*` overrides, set here in
//! process, so a temp folder stands in for `/Volumes` and a fixture file for
//! `/sbin/mount`. Release builds ignore the overrides by design, so this file
//! is debug-only. The environment, the store root and the Hub endpoint are
//! process-wide, so the scenario holds one lock for its whole run, and the
//! environment it changes is restored afterwards. Nothing here touches a real
//! volume, the real home folder, or the network beyond a loopback fixture
//! Hub. The synthetic checkpoint comes from `repack`, which is macOS-only.
#![cfg(all(debug_assertions, target_os = "macos"))]

use std::collections::BTreeMap;
use std::ffi::{CStr, CString, OsString};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::os::raw::{c_char, c_int};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use catalog::library::{Availability, LibraryMarker, LibraryRegistry, LocationEnv};
use catalog::{InstalledModel, ModelModality, Store};
use serde_json::{json, Value};
use turbospark_ffi::{
    abi, ts_install_repo_with_options, ts_installed_json, ts_last_error, ts_libraries_json,
    ts_library_add, ts_library_remove, ts_store_root_set, ts_string_free,
};

// STEP TABLE BEGIN
/// The name of the library the scenario adds.
const LIBRARY: &str = "Fast";
/// The alias of the one model the scenario installs into it.
const ALIAS: &str = "qq-lifecycle";
/// The install id of [`ALIAS`]: `<library>:<modality>/<alias>`, where `{id}`
/// is the id the library's marker carries. It is the same at every step.
const INSTALL_ID: &str = "{id}:text/qq-lifecycle";

/// What one step of the scenario leaves true. Each test performs the step its
/// own way and then checks its listings against these facts.
struct Step {
    /// The action that leads to this step:
    ///
    /// - `empty`: nothing yet.
    /// - `added`: the volume `T7` is added as the library.
    /// - `installed`: the model is installed into the library.
    /// - `renamed`: the volume is renamed to `T7 1`, with the same device.
    /// - `unplugged`: the volume is taken away, with its files.
    /// - `replugged`: the same volume comes back as `T7 2`.
    /// - `removed`: the library is removed, and its files stay.
    /// - `re_added`: the volume `T7 2` is added again.
    name: &'static str,
    /// The volume the library was last found on, `None` while it is not
    /// listed. Its root is `<volumes>/<volume>/TurboSpark`.
    volume: Option<&'static str>,
    /// Whether the library is connected, that is located on a volume now.
    connected: bool,
    /// The models the library lists.
    models: usize,
    /// The availability of the install: `available`, `not_connected` or
    /// `missing`. `None` while it is not listed.
    install: Option<&'static str>,
}

const STEPS: [Step; 8] = [
    Step {
        name: "empty",
        volume: None,
        connected: false,
        models: 0,
        install: None,
    },
    Step {
        name: "added",
        volume: Some("T7"),
        connected: true,
        models: 0,
        install: None,
    },
    Step {
        name: "installed",
        volume: Some("T7"),
        connected: true,
        models: 1,
        install: Some("available"),
    },
    Step {
        name: "renamed",
        volume: Some("T7 1"),
        connected: true,
        models: 1,
        install: Some("available"),
    },
    Step {
        name: "unplugged",
        volume: Some("T7 1"),
        connected: false,
        models: 1,
        install: Some("not_connected"),
    },
    Step {
        name: "replugged",
        volume: Some("T7 2"),
        connected: true,
        models: 1,
        install: Some("available"),
    },
    Step {
        name: "removed",
        volume: None,
        connected: false,
        models: 0,
        install: None,
    },
    Step {
        name: "re_added",
        volume: Some("T7 2"),
        connected: true,
        models: 1,
        install: Some("available"),
    },
];
// STEP TABLE END

/// A revision the fixture repository `owner/model` answers to.
const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

/// The id of a library on another drive, the volume `Decoy`, which is
/// mounted all along and never added. A library is found again by the id its
/// marker carries, so after a rename the scan must pass this drive by, and
/// must leave it untouched.
const DECOY_ID: &str = "5a1d9c70-3e2b-4f68-8a14-6b0c2d9e7f31";

/// The store root, the Hub endpoint and the environment are process-wide.
static SERIAL: Mutex<()> = Mutex::new(());

/// The environment variables the fixture sets or clears, so it can put them
/// back.
const ENVIRONMENT: [&str; 6] = [
    "TURBOSPARK_TEST_MOUNT_TABLE",
    "TURBOSPARK_TEST_VOLUMES_ROOT",
    "TURBOSPARK_TEST_HOME_DEVICE",
    "HF_HOME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
];

// ------------------------------------------------------------------ fixture

/// A temp home, a folder of fake volumes, and a mount table naming the
/// mounted ones, removed when the test ends, passing or not. Declared last,
/// `_serial` is released after the cleanup.
struct Fixture {
    base: PathBuf,
    home: PathBuf,
    volumes: PathBuf,
    /// Where an unplugged drive waits, outside the volumes folder.
    unplugged: PathBuf,
    mounts: PathBuf,
    /// What [`ENVIRONMENT`] held before, put back on drop.
    saved: Vec<(&'static str, Option<OsString>)>,
    _serial: MutexGuard<'static, ()>,
}

impl Fixture {
    fn new(tag: &str) -> Self {
        let serial = SERIAL
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let base = std::env::temp_dir().join(format!(
            "turbospark-ffi-lifecycle-{tag}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        // The temp dir is behind the /var symlink, and the volume rules
        // compare canonical paths.
        let base = base.canonicalize().unwrap();
        let device = fs::metadata(&base).unwrap().dev();
        // Built, and so dropped on a panic, before the environment changes.
        let fixture = Self {
            home: base.join("home"),
            volumes: base.join("Volumes"),
            unplugged: base.join("Unplugged"),
            mounts: base.join("mounts.txt"),
            saved: ENVIRONMENT
                .iter()
                .map(|name| (*name, std::env::var_os(name)))
                .collect(),
            base,
            _serial: serial,
        };
        for dir in [
            &fixture.home,
            &fixture.volumes,
            &fixture.unplugged,
            &fixture.base.join("hf"),
        ] {
            fs::create_dir_all(dir).unwrap();
        }
        fs::create_dir_all(fixture.volume("T7")).unwrap();
        let decoy = fixture.root("Decoy");
        fs::create_dir_all(&decoy).unwrap();
        LibraryMarker::new(DECOY_ID.to_string(), "Decoy".to_string())
            .write(&decoy)
            .unwrap();
        fixture.mount(Some("T7"));
        std::env::set_var("TURBOSPARK_TEST_MOUNT_TABLE", &fixture.mounts);
        std::env::set_var("TURBOSPARK_TEST_VOLUMES_ROOT", &fixture.volumes);
        // Unlike the temp folder's, so the fake volumes are other disks.
        std::env::set_var(
            "TURBOSPARK_TEST_HOME_DEVICE",
            device.wrapping_add(1).to_string(),
        );
        // Nothing reads a real Hugging Face token, or hands one to the
        // fixture Hub.
        std::env::set_var("HF_HOME", fixture.base.join("hf"));
        std::env::remove_var("HF_TOKEN");
        std::env::remove_var("HUGGING_FACE_HUB_TOKEN");
        let home = c(fixture.home.to_str().unwrap());
        assert_eq!(unsafe { ts_store_root_set(home.as_ptr()) }, abi::TS_OK);
        fixture
    }

    fn volume(&self, name: &str) -> PathBuf {
        self.volumes.join(name)
    }

    /// The library root `ts_library_add` makes on the volume `name`.
    fn root(&self, name: &str) -> PathBuf {
        self.volume(name).join("TurboSpark")
    }

    /// Writes the mount table: the boot disk, the decoy drive, and the drive
    /// of the scenario mounted at the volume `name`, if any. The drive is
    /// always `/dev/disk9s1`, so a renamed or replugged drive keeps its
    /// device.
    fn mount(&self, name: Option<&str>) {
        let mut table = format!(
            "/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)\n\
             /dev/disk10s1 on {} (apfs, local, nodev, nosuid, journaled, noowners)\n",
            self.volume("Decoy").display()
        );
        if let Some(name) = name {
            table.push_str(&format!(
                "/dev/disk9s1 on {} (apfs, local, nodev, nosuid, journaled, noowners)\n",
                self.volume(name).display()
            ));
        }
        fs::write(&self.mounts, table).unwrap();
    }

    /// The drive mounts again under another name, as macOS does when the
    /// name is taken: the folder moves and the mount table names the new
    /// mount point.
    fn rename_volume(&self, from: &str, to: &str) {
        fs::rename(self.volume(from), self.volume(to)).unwrap();
        self.mount(Some(to));
    }

    /// The drive is pulled out: its folder, with every file, leaves the
    /// volumes folder and the mount table forgets it.
    fn unplug(&self, name: &str) {
        fs::rename(self.volume(name), self.unplugged.join(name)).unwrap();
        self.mount(None);
    }

    /// The drive that was pulled out as `name` is plugged in as `as_name`.
    fn plug(&self, name: &str, as_name: &str) {
        fs::rename(self.unplugged.join(name), self.volume(as_name)).unwrap();
        self.mount(Some(as_name));
    }

    /// The records of `libraries.json`.
    fn recorded_libraries(&self) -> Vec<Value> {
        match fs::read_to_string(self.home.join("libraries.json")) {
            Ok(text) => {
                let list: Value = serde_json::from_str(&text).unwrap();
                list["libraries"].as_array().cloned().unwrap_or_default()
            }
            Err(_) => Vec::new(),
        }
    }

    /// The rows of `installed.json`.
    fn recorded_installs(&self) -> Vec<InstalledModel> {
        match fs::read(self.home.join("installed.json")) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap(),
            Err(_) => Vec::new(),
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        unsafe { ts_store_root_set(ptr::null()) };
        for (name, value) in &self.saved {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
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

// ----------------------------------------------------------------- the fixture Hub

/// Serves one repository, `owner/model` at [`REVISION`], the way the Hub
/// answers metadata, HEAD, and ranged file requests, on a loopback port. It
/// is the process's Hugging Face endpoint until it is dropped.
struct FileHub {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl FileHub {
    fn start(files: BTreeMap<String, Vec<u8>>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("hub bind");
        listener.set_nonblocking(true).expect("nonblocking hub");
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let thread = thread::spawn(move || {
            while !stopped.load(Ordering::Acquire) {
                let Ok((mut stream, _)) = listener.accept() else {
                    thread::sleep(Duration::from_millis(2));
                    continue;
                };
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
                let _ = stream.write_all(&answer(&method, &target, range.as_deref(), &files));
            }
        });
        catalog::set_hf_endpoint_override(Some(endpoint));
        Self {
            stop,
            thread: Some(thread),
        }
    }
}

impl Drop for FileHub {
    fn drop(&mut self) {
        catalog::set_hf_endpoint_override(None);
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn answer(
    method: &str,
    target: &str,
    range: Option<&str>,
    files: &BTreeMap<String, Vec<u8>>,
) -> Vec<u8> {
    let path = target
        .split('?')
        .next()
        .unwrap_or(target)
        .to_ascii_lowercase();
    if path.starts_with("/api/models/owner/model/") {
        let siblings: Vec<_> = files
            .iter()
            .map(|(name, bytes)| {
                json!({
                    "rfilename": name,
                    "size": bytes.len() as u64,
                    "blobId": "a".repeat(40),
                })
            })
            .collect();
        let body = json!({"id": "owner/model", "sha": REVISION, "siblings": siblings});
        return response(200, body.to_string().as_bytes(), &[]);
    }
    let prefix = format!("/owner/model/resolve/{REVISION}/");
    let Some(bytes) = path.strip_prefix(&prefix).and_then(|name| {
        files
            .iter()
            .find(|(file, _)| file.to_ascii_lowercase() == name)
            .map(|(_, bytes)| bytes)
    }) else {
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
    response(
        206,
        &bytes[start..=end],
        &[(
            "Content-Range",
            format!("bytes {start}-{end}/{}", bytes.len()),
        )],
    )
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

/// The repository the install pulls from: a synthetic GGUF and the ChatML
/// tokenizer files.
fn repository() -> BTreeMap<String, Vec<u8>> {
    let (mut gguf, _) = repack::build_synthetic_gemma4_gguf(repack::SyntheticGgufShape::default());
    // Past the probe's first header read (1 MiB), as every real checkpoint
    // is. A shorter file answers that read short, and the reader then backs
    // off for seconds before asking for the size.
    gguf.resize(gguf.len().max(1 << 21), 0);
    let tokenizer =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../tokenizer/tests/fixtures/ChatMLTokenizer");
    BTreeMap::from([
        ("model-Q8_0.gguf".to_string(), gguf),
        (
            "tokenizer.json".to_string(),
            fs::read(tokenizer.join("tokenizer.json")).unwrap(),
        ),
        (
            "tokenizer_config.json".to_string(),
            fs::read(tokenizer.join("tokenizer_config.json")).unwrap(),
        ),
    ])
}

/// Every file and folder under `root` with the bytes of each file.
fn snapshot(root: &Path) -> BTreeMap<PathBuf, Option<Vec<u8>>> {
    let mut found = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let meta = fs::symlink_metadata(&path).unwrap();
            if meta.is_dir() {
                pending.push(path.clone());
                found.insert(path, None);
            } else {
                found.insert(path.clone(), Some(fs::read(&path).unwrap()));
            }
        }
    }
    found
}

// ------------------------------------------------------------------ entry points

/// `ts_library_add` that must succeed: its answer.
fn add(path: &Path, name: Option<&str>) -> Value {
    let (path, name) = (c(path.to_str().unwrap()), name.map(c));
    let name = name.as_ref().map_or(ptr::null(), |name| name.as_ptr());
    reply(|out| unsafe { ts_library_add(path.as_ptr(), name, out) })
        .ok()
        .json()
}

/// Installs [`ALIAS`] from the fixture Hub into the library `library`: the
/// answer.
fn install_into(library: &str) -> Value {
    let repo = c(&format!("owner/model@{REVISION}"));
    let (alias, file) = (c(ALIAS), c("model-Q8_0.gguf"));
    let options = c(&json!({ "library": library }).to_string());
    reply(|out| unsafe {
        ts_install_repo_with_options(
            repo.as_ptr(),
            alias.as_ptr(),
            file.as_ptr(),
            ptr::null(),
            options.as_ptr(),
            None,
            ptr::null_mut(),
            out,
        )
    })
    .ok()
    .json()
}

/// `ts_libraries_json`, which locates every library and refreshes the list.
fn library_list() -> Vec<Value> {
    reply(|out| unsafe { ts_libraries_json(out) })
        .ok()
        .json()
        .as_array()
        .unwrap()
        .clone()
}

/// `ts_installed_json`: every text install with its place and availability.
fn install_list() -> Vec<Value> {
    reply(|out| unsafe { ts_installed_json(out) })
        .ok()
        .json()
        .as_array()
        .unwrap()
        .clone()
}

// ------------------------------------------------------------------ the scenario

/// What the scenario learns on the way: the library's id, and the drive as it
/// was before the library was removed.
#[derive(Default)]
struct Journey {
    id: String,
    drive_before_removal: BTreeMap<PathBuf, Option<Vec<u8>>>,
}

/// Performs the step `name` through the C ABI.
fn perform(fixture: &Fixture, name: &str, journey: &mut Journey) {
    match name {
        "empty" => {}
        "added" => {
            let added = add(&fixture.volume("T7"), Some(LIBRARY));
            assert_eq!(
                added["library"]["root"],
                fixture.root("T7").to_str().unwrap()
            );
            // The id the drive's marker carries is the id the list records.
            let marker: Value =
                serde_json::from_slice(&fs::read(fixture.root("T7").join("library.json")).unwrap())
                    .unwrap();
            assert_eq!(added["library"]["id"], marker["id"]);
            let records = fixture.recorded_libraries();
            assert_eq!(records.len(), 1, "{records:?}");
            assert_eq!(records[0]["id"], marker["id"]);
            journey.id = marker["id"].as_str().unwrap().to_string();
        }
        "installed" => {
            let installed = install_into(&journey.id);
            let dir = fixture.root("T7").join("text").join("qq-lifecycle.gturbo");
            assert!(dir.join("model_weights.bin").is_file());
            // The answer names the install by the id every later listing
            // gives it.
            assert_eq!(
                installed["install_id"],
                INSTALL_ID.replace("{id}", &journey.id)
            );
            assert_eq!(installed["library"], journey.id);
            assert_eq!(installed["path"], dir.to_str().unwrap());
            // The drive's marker lists it by a path relative to the library.
            let marker: Value =
                serde_json::from_slice(&fs::read(fixture.root("T7").join("library.json")).unwrap())
                    .unwrap();
            assert_eq!(marker["models"][0]["alias"], ALIAS);
            assert_eq!(
                marker["models"][0]["relative_path"],
                "text/qq-lifecycle.gturbo"
            );
        }
        "renamed" => fixture.rename_volume("T7", "T7 1"),
        "unplugged" => fixture.unplug("T7 1"),
        "replugged" => fixture.plug("T7 1", "T7 2"),
        "removed" => {
            journey.drive_before_removal = snapshot(&fixture.volume("T7 2"));
            let id = c(&journey.id);
            let removed = unsafe { ts_library_remove(id.as_ptr()) };
            assert_eq!(removed, abi::TS_OK, "{}", last_error());
            // Every file on the drive is as it was (1.8).
            assert_eq!(
                snapshot(&fixture.volume("T7 2")),
                journey.drive_before_removal
            );
        }
        "re_added" => {
            // The volume, not the library folder: the standard folder on it
            // is found by its marker (1.6).
            let added = add(&fixture.volume("T7 2"), None);
            assert_eq!(added["library"]["id"], journey.id);
            assert_eq!(added["library"]["name"], LIBRARY);
            assert_eq!(added["library"]["modelCount"], 1);
            assert_eq!(added["skipped"], json!([]));
            // Adding wrote nothing to the drive.
            assert_eq!(
                snapshot(&fixture.volume("T7 2")),
                journey.drive_before_removal
            );
        }
        other => panic!("no action for the step {other}"),
    }
}

/// The availability word of an entry, as the catalog layer sees it.
fn word(availability: &Availability) -> &'static str {
    match availability {
        Availability::Available => "available",
        Availability::NotConnected { .. } => "not_connected",
        Availability::Missing => "missing",
    }
}

/// Checks the listings and the registry against the facts of `step`.
fn check(fixture: &Fixture, step: &Step, journey: &Journey) {
    let name = step.name;
    let install_id = INSTALL_ID.replace("{id}", &journey.id);
    let volumes = &fixture.volumes;

    // The installs first, then the libraries: the first listing after a
    // rename must find the library on its own, without the other having
    // recorded where it went.
    let installs = install_list();
    let libraries = library_list();

    // The libraries: This Mac, then the one the scenario adds.
    let this_mac = &libraries[0];
    assert_eq!(this_mac["id"], "internal", "{name}");
    assert_eq!(this_mac["name"], "This Mac", "{name}");
    assert_eq!(
        this_mac["root"],
        fixture.home.join("models").to_str().unwrap(),
        "{name}"
    );
    assert_eq!(this_mac["available"], true, "{name}");
    assert_eq!(this_mac["duplicate"], false, "{name}");
    assert_eq!(this_mac["modelCount"], 0, "{name}");
    assert_eq!(this_mac["isDefault"], true, "{name}");
    assert_eq!(this_mac["isInternal"], true, "{name}");
    assert_eq!(
        libraries.len(),
        1 + usize::from(step.volume.is_some()),
        "{name}: {libraries:?}"
    );
    if let Some(volume) = step.volume {
        let library = &libraries[1];
        assert_eq!(library["id"], journey.id, "{name}");
        assert_eq!(library["name"], LIBRARY, "{name}");
        // The root is where the library is now, and null while it is away.
        let root = volumes.join(volume).join("TurboSpark");
        if step.connected {
            assert_eq!(library["root"], root.to_str().unwrap(), "{name}");
        } else {
            assert!(library["root"].is_null(), "{name}: {library}");
        }
        assert_eq!(library["available"], step.connected, "{name}");
        assert_eq!(library["duplicate"], false, "{name}");
        assert_eq!(library["modelCount"], step.models, "{name}");
        assert_eq!(library["isDefault"], false, "{name}");
        assert_eq!(library["isInternal"], false, "{name}");
    }

    // The installs.
    match step.install {
        Some(availability) => {
            assert_eq!(installs.len(), 1, "{name}: {installs:?}");
            let install = &installs[0];
            assert_eq!(install["alias"], ALIAS, "{name}");
            assert_eq!(install["install_id"], install_id, "{name}");
            assert_eq!(install["library"], journey.id, "{name}");
            assert_eq!(install["library_name"], LIBRARY, "{name}");
            assert_eq!(install["availability"], availability, "{name}");
            assert_eq!(install["location_kind"], "library", "{name}");
            if availability == "not_connected" {
                assert_eq!(install["connect_name"], LIBRARY, "{name}");
            } else {
                assert!(install["connect_name"].is_null(), "{name}: {install}");
            }
            if availability == "available" {
                // Where the install is now, which follows the drive.
                let dir = volumes
                    .join(step.volume.unwrap())
                    .join("TurboSpark")
                    .join("text")
                    .join("qq-lifecycle.gturbo");
                assert_eq!(install["path"], dir.to_str().unwrap(), "{name}");
            }
        }
        None => assert!(installs.is_empty(), "{name}: {installs:?}"),
    }

    // The registry the listings read, which the app and the command line read
    // too (10.1): the list names the library by its marker's id and the
    // volume it was last found on, and the install row carries the same
    // library id, so the install keeps one identity wherever the drive is
    // mounted (4.2).
    let records = fixture.recorded_libraries();
    match step.volume {
        Some(volume) => {
            assert_eq!(records.len(), 1, "{name}: {records:?}");
            assert_eq!(records[0]["id"], journey.id, "{name}");
            assert_eq!(records[0]["name"], LIBRARY, "{name}");
            assert_eq!(records[0]["volume_relative_path"], "TurboSpark", "{name}");
            assert_eq!(
                records[0]["last_mount_path"],
                volumes.join(volume).to_str().unwrap(),
                "{name}: where the library was last found"
            );
        }
        None => assert!(records.is_empty(), "{name}: {records:?}"),
    }
    let rows = fixture.recorded_installs();
    match step.install {
        Some(_) => {
            assert_eq!(rows.len(), 1, "{name}: {rows:?}");
            assert_eq!(
                rows[0].library.as_deref(),
                Some(journey.id.as_str()),
                "{name}"
            );
            assert_eq!(rows[0].alias, ALIAS, "{name}");
            assert_eq!(rows[0].install_id(), install_id, "{name}");
        }
        None => assert!(rows.is_empty(), "{name}: {rows:?}"),
    }

    agrees_with_the_catalog(name, &libraries, &installs);
}

/// The listings above are the catalog layer's, which the command line renders
/// from: the same libraries, ids, names, roots and counts, and the same
/// installs with the same ids, places and availability.
fn agrees_with_the_catalog(name: &str, libraries: &[Value], installs: &[Value]) {
    let store = Store::default_store().unwrap();
    let env = LocationEnv::system(store.root()).unwrap();
    let views = LibraryRegistry::load(store.root()).refresh(&env).views;
    assert_eq!(libraries.len(), views.len(), "{name}");
    for (json, view) in libraries.iter().zip(&views) {
        assert_eq!(json["id"], view.record.id, "{name}");
        assert_eq!(json["name"], view.record.name, "{name}");
        assert_eq!(json["available"], view.available, "{name}");
        assert_eq!(json["duplicate"], view.duplicate, "{name}");
        assert_eq!(json["modelCount"], view.model_count, "{name}");
        assert_eq!(
            json["isDefault"], view.record.default_for_downloads,
            "{name}"
        );
        let root = view.root.as_ref().map(|root| root.to_str().unwrap());
        assert_eq!(json["root"], json!(root), "{name}");
    }

    let mut entries: Vec<_> = store
        .installed_with_availability(&env)
        .into_iter()
        .filter(|entry| entry.row.effective_modality() == ModelModality::Text)
        .collect();
    entries.sort_by(|a, b| a.row.alias.cmp(&b.row.alias));
    assert_eq!(installs.len(), entries.len(), "{name}");
    for (json, entry) in installs.iter().zip(&entries) {
        assert_eq!(json["alias"], entry.row.alias, "{name}");
        assert_eq!(json["install_id"], entry.install_id, "{name}");
        assert_eq!(json["library_name"], entry.library_name, "{name}");
        assert_eq!(json["availability"], word(&entry.availability), "{name}");
        assert_eq!(json["path"], entry.row.path.to_str().unwrap(), "{name}");
    }
}

/// The whole lifecycle through the C ABI. The install keeps its id from the
/// install to the end (4.2), removing the library leaves the drive
/// byte-identical (1.8), and adding the folder again lists the model again
/// from the marker alone (1.6).
#[test]
fn a_library_keeps_its_install_ids_through_a_rename_a_removal_and_a_re_add() {
    let fixture = Fixture::new("scenario");
    // Dropped before the fixture, which releases the lock last.
    let _hub = FileHub::start(repository());
    let decoy_before = snapshot(&fixture.volume("Decoy"));
    let mut journey = Journey::default();

    for step in &STEPS {
        perform(&fixture, step.name, &mut journey);
        check(&fixture, step, &journey);
    }

    // The model's files went to the drive and nowhere else, and survived the
    // renames, the unplug, and the removal.
    let weights = fixture
        .root("T7 2")
        .join("text")
        .join("qq-lifecycle.gturbo")
        .join("model_weights.bin");
    assert!(weights.is_file(), "{}", weights.display());
    assert!(!fixture.home.join("models").join("text").exists());
    // Nothing recreated a folder at a mount point the drive had left.
    for gone in ["T7", "T7 1"] {
        assert!(!fixture.volume(gone).exists(), "{gone}");
    }
    // The other drive was never listed, searched into, or written.
    assert_eq!(snapshot(&fixture.volume("Decoy")), decoy_before);
}

/// The text between the table markers of `source`, without trailing blanks.
fn step_table(source: &str) -> String {
    // Written in pieces so this function's own text holds neither marker.
    let begin = concat!("// STEP TABLE", " BEGIN");
    let end = concat!("// STEP TABLE", " END");
    let start = source.find(begin).expect("a begin marker") + begin.len();
    let stop = source[start..].find(end).expect("an end marker") + start;
    source[start..stop]
        .lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n")
}

/// The two tests run one scenario against one table, but the table is
/// written out twice, once per crate. The copies must not drift: a step added
/// to one and not the other would make the two surfaces look like they agree
/// on a scenario they no longer share.
#[test]
fn the_step_table_is_the_command_lines() {
    let ours = step_table(include_str!("library_lifecycle.rs"));
    let theirs = step_table(include_str!("../../cli/tests/library_lifecycle.rs"));
    assert!(ours.contains("const STEPS: [Step; 8]"), "{ours}");
    assert_eq!(
        ours, theirs,
        "crates/ffi/tests/library_lifecycle.rs and crates/cli/tests/library_lifecycle.rs \
         must hold the same step table"
    );
}
