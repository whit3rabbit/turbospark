//! The library lifecycle end to end, through the real `turbospark-model`
//! binary and without hardware (task 7.1).
//!
//! One scenario: add a library on a fake volume, install a model into it,
//! rename the volume, unplug it, plug it back under another name, remove the
//! library, and add the same folder again. After every step the command
//! line's listings are checked against the step table below, and so is the
//! registry on disk. The install must keep one install id through all of it
//! (4.2), removing the library must leave every file on the drive as it was
//! (1.8), and adding the folder again must bring the rows back from the
//! drive's marker with the same ids (1.6). A second drive, `Decoy`, is
//! mounted throughout with a library of its own that is never added, so
//! finding the renamed drive again must go by the marker's id and not by
//! the shape of the folder.
//!
//! `crates/ffi/tests/library_lifecycle.rs` runs the same scenario through the
//! C ABI against the same table. The command line and the C ABI are separate
//! crates and this crate may not depend on the other, so agreement between
//! them is established by two tests over one scenario, not by one process
//! calling both. The FFI test compares the text between the table markers
//! below with its own copy and fails when they differ, so edit both.
//!
//! The location rules read the mount table and the home folder's device.
//! Debug builds take both from the `TURBOSPARK_TEST_*` overrides, set here on
//! each spawned command only, so a temp folder stands in for `/Volumes` and a
//! fixture file for `/sbin/mount`. Release builds ignore the overrides by
//! design (a shipped binary must not be able to bypass the network refusal),
//! so this file is debug-only. Nothing here touches a real volume, the real
//! home folder, or the network beyond a loopback fixture Hub.
#![cfg(debug_assertions)]

use std::collections::BTreeMap;
use std::fs;
use std::net::TcpListener;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use catalog::library::LibraryMarker;

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

/// A temp home, a folder of fake volumes, and a mount table naming the
/// mounted ones, removed when the test ends, passing or not.
struct Fixture {
    base: PathBuf,
    home: PathBuf,
    volumes: PathBuf,
    /// Where an unplugged drive waits, outside the volumes folder.
    unplugged: PathBuf,
    mounts: PathBuf,
    /// The real device of the temp folder. The fake volumes share it, so a
    /// home device unlike it is what makes them "other disks".
    device: u64,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.base);
    }
}

struct Outcome {
    code: i32,
    stdout: String,
    stderr: String,
}

impl Fixture {
    fn new(name: &str) -> Self {
        let base = std::env::temp_dir().join(format!(
            "turbospark-library-lifecycle-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        // The temp dir is behind the /var symlink, and the volume rules
        // compare canonical paths.
        let base = base.canonicalize().unwrap();
        let fixture = Self {
            home: base.join("home"),
            volumes: base.join("Volumes"),
            unplugged: base.join("Unplugged"),
            mounts: base.join("mounts.txt"),
            device: fs::metadata(&base).unwrap().dev(),
            base,
        };
        for dir in [
            &fixture.home,
            &fixture.volumes,
            &fixture.unplugged,
            &fixture.base.join("user"),
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
        fixture
    }

    fn volume(&self, name: &str) -> PathBuf {
        self.volumes.join(name)
    }

    /// The library root `library add` makes on the volume `name`.
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

    /// The binary, with this home, the fake volumes, and a home device
    /// unlike the temp folder's, so the fake volumes count as drives other
    /// than the Mac's own disk. `HOME` and `HF_HOME` point into the temp
    /// folder and the token variables are removed, so nothing reads the real
    /// home or hands a real token to the fixture Hub.
    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_turbospark-model"));
        command
            .env("TURBOSPARK_HOME", &self.home)
            .env("TURBOSPARK_TEST_MOUNT_TABLE", &self.mounts)
            .env("TURBOSPARK_TEST_VOLUMES_ROOT", &self.volumes)
            .env(
                "TURBOSPARK_TEST_HOME_DEVICE",
                self.device.wrapping_add(1).to_string(),
            )
            .env("HOME", self.base.join("user"))
            .env("HF_HOME", self.base.join("hf"))
            .env_remove("HF_TOKEN")
            .env_remove("HUGGING_FACE_HUB_TOKEN")
            .stdin(Stdio::null());
        command
    }

    fn run(&self, args: &[&str]) -> Outcome {
        outcome(self.command().args(args))
    }

    /// [`Self::run`] with `hub` as the Hugging Face endpoint.
    fn run_against(&self, hub: &FileHub, args: &[&str]) -> Outcome {
        outcome(self.command().env("HF_ENDPOINT", &hub.endpoint).args(args))
    }

    /// A command that must succeed.
    fn succeed(&self, args: &[&str]) -> Outcome {
        let done = self.run(args);
        assert_eq!(done.code, 0, "{args:?}: {}\n{}", done.stdout, done.stderr);
        done
    }

    /// The records of `libraries.json`.
    fn recorded_libraries(&self) -> Vec<serde_json::Value> {
        match fs::read_to_string(self.home.join("libraries.json")) {
            Ok(text) => {
                let list: serde_json::Value = serde_json::from_str(&text).unwrap();
                list["libraries"].as_array().cloned().unwrap_or_default()
            }
            Err(_) => Vec::new(),
        }
    }

    /// The rows of `installed.json`.
    fn recorded_installs(&self) -> Vec<catalog::InstalledModel> {
        match fs::read(self.home.join("installed.json")) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap(),
            Err(_) => Vec::new(),
        }
    }
}

fn outcome(command: &mut Command) -> Outcome {
    let out = command.output().expect("the binary runs");
    Outcome {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).to_string(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
    }
}

/// Serves one repository, `owner/model` at [`REVISION`], the way the Hub
/// answers metadata, HEAD, and ranged file requests, on a loopback port.
struct FileHub {
    endpoint: String,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl FileHub {
    fn start(files: BTreeMap<String, Vec<u8>>) -> Self {
        use std::io::{BufRead, BufReader, Write};
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
        Self {
            endpoint,
            stop,
            thread: Some(thread),
        }
    }
}

impl Drop for FileHub {
    fn drop(&mut self) {
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
                serde_json::json!({
                    "rfilename": name,
                    "size": bytes.len() as u64,
                    "blobId": "a".repeat(40),
                })
            })
            .collect();
        let body = serde_json::json!({
            "id": "owner/model",
            "sha": REVISION,
            "siblings": siblings,
        });
        return response(200, &body.to_string().into_bytes(), &[]);
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

// ------------------------------------------------------------------ the listings

/// The cells of one table line. Columns are separated by two or more spaces
/// and a cell holds single spaces only ("T7 1", "not connected").
fn cells(line: &str) -> Vec<String> {
    line.trim()
        .split("  ")
        .map(str::trim)
        .filter(|cell| !cell.is_empty())
        .map(str::to_string)
        .collect()
}

/// One line of `library list`.
#[derive(Debug, PartialEq, Eq)]
struct LibraryLine {
    name: String,
    location: String,
    connected: bool,
    models: usize,
    default: bool,
    duplicate: bool,
}

/// `library list`, parsed. A warning on stderr fails the scenario: nothing in
/// it corrupts a file the listing reads.
fn library_list(fixture: &Fixture) -> Vec<LibraryLine> {
    let listed = fixture.succeed(&["library", "list"]);
    assert!(!listed.stderr.contains("warning:"), "{}", listed.stderr);
    let mut lines = listed.stdout.lines();
    assert_eq!(
        cells(lines.next().expect("a header line")),
        ["NAME", "LOCATION", "STATUS", "MODELS"],
        "{}",
        listed.stdout
    );
    lines
        .map(|line| {
            let cells = cells(line);
            let connected = match cells[2].as_str() {
                "connected" => true,
                status if status.starts_with("not connected") => false,
                status => panic!("an unknown status {status:?} in {line:?}"),
            };
            LibraryLine {
                name: cells[0].clone(),
                location: cells[1].clone(),
                connected,
                models: cells[3].parse().unwrap(),
                default: cells.iter().any(|cell| cell == "(default)"),
                duplicate: cells.iter().any(|cell| cell.starts_with("note:")),
            }
        })
        .collect()
}

/// One line of the installed section of `list`.
#[derive(Debug, PartialEq, Eq)]
struct InstallLine {
    alias: String,
    kind: String,
    library: String,
    /// `available`, `not_connected` or `missing`, as the C ABI spells them.
    availability: String,
}

/// The installs `list` shows for [`ALIAS`].
fn install_list(fixture: &Fixture) -> Vec<InstallLine> {
    let listed = fixture.succeed(&["list", "--filter", ALIAS]);
    assert!(!listed.stderr.contains("warning:"), "{}", listed.stderr);
    // The alias matches no curated row, so nothing but installs can show.
    if listed.stdout.trim() == "no models match" {
        return Vec::new();
    }
    let (_, section) = listed
        .stdout
        .split_once("installed:\n")
        .unwrap_or_else(|| panic!("no installed section in:\n{}", listed.stdout));
    let mut lines = section.lines();
    assert_eq!(
        cells(lines.next().expect("a header line")),
        ["ALIAS", "KIND", "LOCATION", "STATUS"],
        "{section}"
    );
    lines
        .map(|line| {
            let cells = cells(line);
            InstallLine {
                alias: cells[0].clone(),
                kind: cells[1].clone(),
                library: cells[2].clone(),
                availability: cells[3].replace(' ', "_"),
            }
        })
        .collect()
}

// ------------------------------------------------------------------ the scenario

/// What the scenario learns on the way: the library's id, and the drive as it
/// was before the library was removed.
#[derive(Default)]
struct Journey {
    id: String,
    drive_before_removal: BTreeMap<PathBuf, Option<Vec<u8>>>,
}

/// Performs the step `name` through the command line.
fn perform(fixture: &Fixture, hub: &FileHub, name: &str, journey: &mut Journey) {
    match name {
        "empty" => {}
        "added" => {
            let path = fixture.volume("T7");
            let added =
                fixture.succeed(&["library", "add", path.to_str().unwrap(), "--name", LIBRARY]);
            assert!(
                added
                    .stdout
                    .contains(&fixture.root("T7").display().to_string()),
                "{}",
                added.stdout
            );
            // The id the drive's marker carries is the id the list records.
            let marker: serde_json::Value =
                serde_json::from_slice(&fs::read(fixture.root("T7").join("library.json")).unwrap())
                    .unwrap();
            let records = fixture.recorded_libraries();
            assert_eq!(records.len(), 1, "{records:?}");
            assert_eq!(records[0]["id"], marker["id"]);
            journey.id = marker["id"].as_str().unwrap().to_string();
        }
        "installed" => {
            let repo = format!("owner/model@{REVISION}");
            let pulled = fixture.run_against(
                hub,
                &[
                    "pull",
                    "--repo",
                    &repo,
                    "--alias",
                    ALIAS,
                    "--file",
                    "model-Q8_0.gguf",
                    "--library",
                    LIBRARY,
                    "--force",
                ],
            );
            assert_eq!(pulled.code, 0, "{}\n{}", pulled.stdout, pulled.stderr);
            let dir = fixture.root("T7").join("text").join("qq-lifecycle.gturbo");
            assert!(dir.join("model_weights.bin").is_file());
            // The drive's marker lists it by a path relative to the library.
            let marker: serde_json::Value =
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
            let removed = fixture.succeed(&["library", "remove", LIBRARY, "--yes"]);
            assert!(removed.stdout.contains("untouched"), "{}", removed.stdout);
            // Every file on the drive is as it was (1.8).
            assert_eq!(
                snapshot(&fixture.volume("T7 2")),
                journey.drive_before_removal
            );
        }
        "re_added" => {
            // The volume, not the library folder: the standard folder on it
            // is found by its marker (1.6).
            let path = fixture.volume("T7 2");
            let added = fixture.succeed(&["library", "add", path.to_str().unwrap()]);
            assert!(
                added
                    .stdout
                    .contains(&format!("added the library {LIBRARY}")),
                "{}",
                added.stdout
            );
            assert!(added.stdout.contains("with 1 model"), "{}", added.stdout);
            // Adding listed nothing it should not have, and wrote nothing to
            // the drive.
            assert_eq!(
                snapshot(&fixture.volume("T7 2")),
                journey.drive_before_removal
            );
        }
        other => panic!("no action for the step {other}"),
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
    let installs = install_list(fixture);
    let libraries = library_list(fixture);

    let expected_installs: Vec<InstallLine> = step
        .install
        .map(|availability| InstallLine {
            alias: ALIAS.to_string(),
            kind: "text".to_string(),
            library: LIBRARY.to_string(),
            availability: availability.to_string(),
        })
        .into_iter()
        .collect();
    assert_eq!(installs, expected_installs, "{name}: the install listing");

    let mut expected_libraries = vec![LibraryLine {
        name: "This Mac".to_string(),
        location: fixture.home.join("models").display().to_string(),
        connected: true,
        models: 0,
        default: true,
        duplicate: false,
    }];
    if let Some(volume) = step.volume {
        expected_libraries.push(LibraryLine {
            name: LIBRARY.to_string(),
            location: volumes
                .join(volume)
                .join("TurboSpark")
                .display()
                .to_string(),
            connected: step.connected,
            models: step.models,
            default: false,
            duplicate: false,
        });
    }
    assert_eq!(libraries, expected_libraries, "{name}: the library listing");

    // The registry the listings read, which the app reads too (10.1): the
    // list names the library by its marker's id and the volume it was last
    // found on, and the install row carries the same library id, so the
    // install keeps one identity wherever the drive is mounted (4.2).
    let records = fixture.recorded_libraries();
    match step.volume {
        Some(volume) => {
            assert_eq!(records.len(), 1, "{name}: {records:?}");
            assert_eq!(records[0]["id"], journey.id, "{name}");
            assert_eq!(records[0]["name"], LIBRARY, "{name}");
            assert_eq!(records[0]["volume_relative_path"], "TurboSpark", "{name}");
            assert_eq!(
                records[0]["last_mount_path"],
                volumes.join(volume).display().to_string(),
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
}

/// The whole lifecycle through the command line. The install keeps its id
/// from the install to the end (4.2), removing the library leaves the drive
/// byte-identical (1.8), and adding the folder again lists the model again
/// from the marker alone (1.6).
#[test]
fn a_library_keeps_its_install_ids_through_a_rename_a_removal_and_a_re_add() {
    let fixture = Fixture::new("scenario");
    let hub = FileHub::start(repository());
    let decoy_before = snapshot(&fixture.volume("Decoy"));
    let mut journey = Journey::default();

    for step in &STEPS {
        perform(&fixture, &hub, step.name, &mut journey);
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
