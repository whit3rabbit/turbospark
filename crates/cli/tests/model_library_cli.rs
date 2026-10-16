//! `turbospark-model library`: list, add, remove, and default, driven
//! through the real binary against fake volumes.
//!
//! The location rules read the mount table and the home folder's device.
//! Debug builds take both from the `TURBOSPARK_TEST_*` overrides, so a temp
//! folder stands in for `/Volumes` and a fixture file for `/sbin/mount`.
//! Release builds ignore the overrides by design (a shipped binary must not
//! be able to bypass the network refusal), so this file is debug-only.
#![cfg(debug_assertions)]

use std::collections::BTreeMap;
use std::fs;
use std::net::TcpListener;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// A temp home, a folder of fake volumes, and a mount table naming them,
/// removed when the test ends, passing or not.
struct Fixture {
    base: PathBuf,
    home: PathBuf,
    volumes: PathBuf,
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

/// A folder made read-only (0o555) until dropped, which restores 0o755, so
/// a failing test still leaves its temp tree removable. Declared after the
/// [`Fixture`] it is in, so it drops first.
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

struct Outcome {
    code: i32,
    stdout: String,
    stderr: String,
}

impl Fixture {
    fn new(name: &str) -> Self {
        let base = std::env::temp_dir().join(format!(
            "turbospark-model-library-{name}-{}",
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
        for volume in ["T7", "T9", "Share", "Stick", "Locked", "Unlisted"] {
            fs::create_dir_all(volumes.join(volume)).unwrap();
        }
        let at = |volume: &str| volumes.join(volume).display().to_string();
        // `Unlisted` has no line, so its drive cannot be identified.
        let table = format!(
            "/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)\n\
             /dev/disk9s1 on {} (apfs, local, nodev, nosuid, journaled, noowners)\n\
             /dev/disk10s1 on {} (apfs, local, nodev, nosuid, journaled, noowners)\n\
             //user@nas/share on {} (smbfs, nodev, nosuid, mounted by user)\n\
             /dev/disk11s1 on {} (msdos, local, nodev, nosuid, noowners)\n\
             /dev/disk12s1 on {} (apfs, local, read-only, journaled)\n",
            at("T7"),
            at("T9"),
            at("Share"),
            at("Stick"),
            at("Locked"),
        );
        let mounts = base.join("mounts.txt");
        fs::write(&mounts, table).unwrap();
        let device = fs::metadata(&base).unwrap().dev();
        Self {
            base,
            home,
            volumes,
            mounts,
            device,
        }
    }

    fn volume(&self, name: &str) -> PathBuf {
        self.volumes.join(name)
    }

    /// The binary with this home, the fake volumes, and `home_device` as the
    /// home folder's device.
    fn command_with_home_device(&self, home_device: u64) -> Command {
        self.binary_with_home_device("turbospark-model", home_device)
    }

    /// [`Self::command_with_home_device`] for another binary of this crate.
    fn binary_with_home_device(&self, binary: &str, home_device: u64) -> Command {
        let mut path = std::env::current_exe().expect("test binary path");
        path.pop(); // deps/
        path.pop();
        path.push(binary);
        let mut command = Command::new(path);
        command
            .env("TURBOSPARK_HOME", &self.home)
            .env("TURBOSPARK_TEST_MOUNT_TABLE", &self.mounts)
            .env("TURBOSPARK_TEST_VOLUMES_ROOT", &self.volumes)
            .env("TURBOSPARK_TEST_HOME_DEVICE", home_device.to_string())
            .stdin(Stdio::null());
        command
    }

    /// A home device unlike the temp folder's, so the fake volumes count as
    /// drives other than the Mac's own disk.
    fn run(&self, args: &[&str]) -> Outcome {
        self.run_with_home_device(self.device.wrapping_add(1), args)
    }

    fn run_with_home_device(&self, home_device: u64, args: &[&str]) -> Outcome {
        let out = self
            .command_with_home_device(home_device)
            .args(args)
            .output()
            .expect("the binary runs");
        Outcome {
            code: out.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&out.stdout).to_string(),
            stderr: String::from_utf8_lossy(&out.stderr).to_string(),
        }
    }

    /// [`Self::run`] with `hub` as the Hugging Face endpoint.
    fn run_against(&self, hub: &Hub, args: &[&str]) -> Outcome {
        let out = self
            .command_with_home_device(self.device.wrapping_add(1))
            .env("HF_ENDPOINT", &hub.endpoint)
            .args(args)
            .output()
            .expect("the binary runs");
        Outcome {
            code: out.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&out.stdout).to_string(),
            stderr: String::from_utf8_lossy(&out.stderr).to_string(),
        }
    }

    /// `library add` that must succeed.
    fn add(&self, volume: &str, name: &str) -> Outcome {
        let path = self.volume(volume);
        let added = self.run(&["library", "add", path.to_str().unwrap(), "--name", name]);
        assert_eq!(added.code, 0, "add {volume}: {}", added.stderr);
        added
    }

    /// The id `libraries.json` records for the library named `name`.
    fn id_of(&self, name: &str) -> String {
        let text = fs::read_to_string(self.home.join("libraries.json")).unwrap();
        let list: serde_json::Value = serde_json::from_str(&text).unwrap();
        list["libraries"]
            .as_array()
            .unwrap()
            .iter()
            .find(|record| record["name"] == name)
            .and_then(|record| record["id"].as_str())
            .unwrap_or_else(|| panic!("no library named {name} in {text}"))
            .to_string()
    }
}

/// A local port standing in for the Hub. It answers nothing: it counts each
/// connection and closes it, so a command that reaches the network fails
/// fast, and a count of zero means the command made no network request.
struct Hub {
    endpoint: String,
    connections: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Hub {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("hub bind");
        listener.set_nonblocking(true).expect("nonblocking hub");
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let connections = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (counted, stopped) = (Arc::clone(&connections), Arc::clone(&stop));
        let thread = thread::spawn(move || {
            while !stopped.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        counted.fetch_add(1, Ordering::AcqRel);
                        drop(stream);
                    }
                    Err(_) => thread::sleep(Duration::from_millis(2)),
                }
            }
        });
        Self {
            endpoint,
            connections,
            stop,
            thread: Some(thread),
        }
    }

    /// Connections so far. Called after the command exits, by which time
    /// any connection it made is in the listener's queue or counted; the
    /// short wait lets the accept loop drain the queue.
    fn connections(&self) -> usize {
        thread::sleep(Duration::from_millis(50));
        self.connections.load(Ordering::Acquire)
    }
}

impl Drop for Hub {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// The one stdout line that contains `needle`.
fn line_with<'a>(stdout: &'a str, needle: &str) -> &'a str {
    let lines: Vec<&str> = stdout
        .lines()
        .filter(|line| line.contains(needle))
        .collect();
    assert_eq!(lines.len(), 1, "one line with {needle:?} in:\n{stdout}");
    lines[0]
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

#[test]
fn the_usage_text_documents_the_library_commands() {
    let fixture = Fixture::new("usage");
    let help = fixture.run(&["--help"]);
    assert_eq!(help.code, 0);
    for command in [
        "library list",
        "library add <PATH>",
        "library remove <NAME>",
        "library default <NAME>",
    ] {
        assert!(help.stdout.contains(command), "{command}: {}", help.stdout);
    }

    let bare = fixture.run(&["library"]);
    assert_eq!(bare.code, 2, "{}", bare.stderr);
    let unknown = fixture.run(&["library", "rename"]);
    assert_eq!(unknown.code, 2, "{}", unknown.stderr);
    // A flag on the wrong library command is refused, not ignored.
    let path = fixture.volume("T7");
    let wrong = fixture.run(&["library", "add", path.to_str().unwrap(), "--yes"]);
    assert_eq!(wrong.code, 2, "{}", wrong.stderr);
    assert!(wrong.stderr.contains("--yes"), "{}", wrong.stderr);
    assert!(!fixture.home.join("libraries.json").exists());
}

/// Add on a fake volume, list it, make it the default, then remove it and
/// find every file on the drive as it was (1.8, 10.5).
#[test]
fn a_library_is_added_listed_made_default_and_removed_with_its_files_intact() {
    let fixture = Fixture::new("lifecycle");

    let empty = fixture.run(&["library", "list"]);
    assert_eq!(empty.code, 0, "{}", empty.stderr);
    let this_mac = line_with(&empty.stdout, "This Mac");
    assert!(this_mac.contains("connected"), "{this_mac}");
    assert!(this_mac.contains("(default)"), "{this_mac}");

    let added = fixture.add("T7", "Fast");
    let root = fixture.volume("T7").join("TurboSpark");
    assert!(added.stdout.contains("Fast"), "{}", added.stdout);
    assert!(
        added.stdout.contains(&root.display().to_string()),
        "{}",
        added.stdout
    );
    assert!(root.join("library.json").is_file());

    let listed = fixture.run(&["library", "list"]);
    assert_eq!(listed.code, 0, "{}", listed.stderr);
    let fast = line_with(&listed.stdout, "Fast");
    assert!(fast.contains(&root.display().to_string()), "{fast}");
    assert!(fast.contains("connected"), "{fast}");
    assert!(!fast.contains("not connected"), "{fast}");
    assert!(!fast.contains("(default)"), "{fast}");
    // This Mac is listed first.
    assert!(
        listed.stdout.find("This Mac").unwrap() < listed.stdout.find("Fast").unwrap(),
        "{}",
        listed.stdout
    );

    // Names resolve case-insensitively.
    let made_default = fixture.run(&["library", "default", "fast"]);
    assert_eq!(made_default.code, 0, "{}", made_default.stderr);
    assert!(
        made_default.stdout.contains("Fast"),
        "{}",
        made_default.stdout
    );
    let listed = fixture.run(&["library", "list"]);
    assert!(
        line_with(&listed.stdout, "Fast").contains("(default)"),
        "{}",
        listed.stdout
    );
    assert!(
        !line_with(&listed.stdout, "This Mac").contains("(default)"),
        "{}",
        listed.stdout
    );

    // Models on the drive, which removing the library must leave alone.
    let model = root.join("text").join("tiny.gturbo");
    fs::create_dir_all(&model).unwrap();
    fs::write(model.join("weights.bin"), b"weights").unwrap();
    fs::write(model.join("config.json"), b"{}").unwrap();
    let before = snapshot(&fixture.volume("T7"));

    // Like `rm`: without --yes the name must be typed back.
    let unconfirmed = fixture.run(&["library", "remove", "Fast"]);
    assert_eq!(unconfirmed.code, 1, "{}", unconfirmed.stderr);
    assert!(
        unconfirmed.stderr.contains("not confirmed"),
        "{}",
        unconfirmed.stderr
    );
    let listed = fixture.run(&["library", "list"]);
    line_with(&listed.stdout, "Fast");

    let removed = fixture.run(&["library", "remove", "Fast", "--yes"]);
    assert_eq!(removed.code, 0, "{}", removed.stderr);
    assert!(removed.stdout.contains("untouched"), "{}", removed.stdout);
    assert_eq!(snapshot(&fixture.volume("T7")), before);

    let listed = fixture.run(&["library", "list"]);
    assert_eq!(listed.code, 0, "{}", listed.stderr);
    assert!(!listed.stdout.contains("Fast"), "{}", listed.stdout);
    // Removing the default hands it back to This Mac.
    assert!(
        line_with(&listed.stdout, "This Mac").contains("(default)"),
        "{}",
        listed.stdout
    );
}

/// One sentence per refusal, each naming the rule it hit (1.4, 1.5, 1.7,
/// 1.10), and nothing created on a refused drive.
#[test]
fn each_refused_location_gets_its_own_sentence() {
    let fixture = Fixture::new("refusals");
    let refuse = |path: &Path, expected: &[&str]| {
        let outcome = fixture.run(&["library", "add", path.to_str().unwrap()]);
        assert_eq!(outcome.code, 1, "{}: {}", path.display(), outcome.stderr);
        let errors: Vec<&str> = outcome
            .stderr
            .lines()
            .filter(|line| line.starts_with("error:"))
            .collect();
        assert_eq!(errors.len(), 1, "{}", outcome.stderr);
        for phrase in expected {
            assert!(errors[0].contains(phrase), "{phrase:?} in {}", errors[0]);
        }
        errors[0].to_string()
    };

    let network = refuse(
        &fixture.volume("Share"),
        &["network share", "directly connected drive"],
    );
    let fat = refuse(&fixture.volume("Stick"), &["4 GiB", "msdos"]);
    let read_only = refuse(&fixture.volume("Locked"), &["read-only"]);
    let unknown = refuse(
        &fixture.volume("Unlisted"),
        &["could not identify this drive"],
    );
    let missing = refuse(&fixture.base.join("missing"), &["not an existing folder"]);
    let inside_home = refuse(&fixture.home, &["TurboSpark home folder", "This Mac"]);

    // A folder on the home's own disk: the temp dir's real device as the
    // home device.
    let plain = fixture.base.join("Plain");
    fs::create_dir_all(&plain).unwrap();
    let same_disk =
        fixture.run_with_home_device(fixture.device, &["library", "add", plain.to_str().unwrap()]);
    assert_eq!(same_disk.code, 1, "{}", same_disk.stderr);
    assert!(
        same_disk
            .stderr
            .contains("is on the same disk as This Mac, which already covers it"),
        "{}",
        same_disk.stderr
    );
    // The same fake volume that passes with another home device.
    let t7 = fixture.volume("T7");
    let same_disk_volume =
        fixture.run_with_home_device(fixture.device, &["library", "add", t7.to_str().unwrap()]);
    assert_eq!(same_disk_volume.code, 1, "{}", same_disk_volume.stderr);
    assert!(
        same_disk_volume.stderr.contains("same disk as This Mac"),
        "{}",
        same_disk_volume.stderr
    );

    // Nothing was created or listed by any refusal.
    for volume in ["Share", "Stick", "Locked", "Unlisted", "T7"] {
        assert!(
            !fixture.volume(volume).join("TurboSpark").exists(),
            "{volume}"
        );
    }
    assert!(!plain.join("TurboSpark").exists());
    assert!(!fixture.home.join("libraries.json").exists());

    // Inside a library already listed names that library.
    fixture.add("T7", "Fast");
    let inner = fixture.volume("T7").join("TurboSpark").join("inner");
    fs::create_dir_all(&inner).unwrap();
    let nested = refuse(&inner, &["Fast", "already covers"]);

    let sentences = [
        &network,
        &fat,
        &read_only,
        &unknown,
        &missing,
        &inside_home,
        &nested,
    ];
    for (index, sentence) in sentences.iter().enumerate() {
        for other in &sentences[index + 1..] {
            assert_ne!(sentence, other);
        }
    }
}

/// This Mac cannot be removed, and a name that matches no library or more
/// than one is refused rather than guessed.
#[test]
fn this_mac_and_unknown_or_ambiguous_names_are_refused() {
    let fixture = Fixture::new("names");
    for target in ["This Mac", "this mac", "internal"] {
        let outcome = fixture.run(&["library", "remove", target, "--yes"]);
        assert_eq!(outcome.code, 1, "{target}: {}", outcome.stderr);
        assert!(
            outcome.stderr.contains("This Mac cannot be removed"),
            "{target}: {}",
            outcome.stderr
        );
    }

    for args in [
        &["library", "remove", "nope", "--yes"][..],
        &["library", "default", "nope"][..],
    ] {
        let outcome = fixture.run(args);
        assert_eq!(outcome.code, 1, "{args:?}: {}", outcome.stderr);
        assert!(
            outcome.stderr.contains("no library named \"nope\""),
            "{args:?}: {}",
            outcome.stderr
        );
    }

    fixture.add("T7", "Twin");
    fixture.add("T9", "twin");
    let first = fixture.id_of("Twin");
    let second = fixture.id_of("twin");
    for args in [
        &["library", "remove", "TWIN", "--yes"][..],
        &["library", "default", "twin"][..],
    ] {
        let outcome = fixture.run(args);
        assert_eq!(outcome.code, 1, "{args:?}: {}", outcome.stderr);
        assert!(
            outcome.stderr.contains("2 libraries are named"),
            "{args:?}: {}",
            outcome.stderr
        );
        // The ids are the way to pick one.
        assert!(outcome.stderr.contains(&first), "{}", outcome.stderr);
        assert!(outcome.stderr.contains(&second), "{}", outcome.stderr);
    }
    assert_eq!(fixture.id_of("Twin"), first, "nothing was removed");

    let by_id = fixture.run(&["library", "default", &second]);
    assert_eq!(by_id.code, 0, "{}", by_id.stderr);
    let back = fixture.run(&["library", "default", "this mac"]);
    assert_eq!(back.code, 0, "{}", back.stderr);
    let listed = fixture.run(&["library", "list"]);
    assert!(
        line_with(&listed.stdout, "This Mac").contains("(default)"),
        "{}",
        listed.stdout
    );
    let removed = fixture.run(&["library", "remove", &first, "--yes"]);
    assert_eq!(removed.code, 0, "{}", removed.stderr);
    let listed = fixture.run(&["library", "list"]);
    assert!(!listed.stdout.contains("Twin"), "{}", listed.stdout);
    line_with(&listed.stdout, "twin");
}

/// Listing never fails for bookkeeping (3.4): a refresh that cannot record
/// when a library was last seen still lists every library, and warns.
#[test]
fn list_survives_a_failed_last_seen_write() {
    let fixture = Fixture::new("persist");
    fixture.add("T7", "Fast");
    // An old date, so the refresh has something to record.
    let list_path = fixture.home.join("libraries.json");
    let text = fs::read_to_string(&list_path).unwrap();
    let mut list: serde_json::Value = serde_json::from_str(&text).unwrap();
    list["libraries"][0]["last_seen"] = serde_json::json!("2000-01-01");
    fs::write(&list_path, serde_json::to_vec_pretty(&list).unwrap()).unwrap();
    let before = fs::read(&list_path).unwrap();

    let _locked = Locked::new(&fixture.home);
    // Root ignores the mode, and then nothing here can fail to write.
    let probe = fixture.home.join("probe");
    if fs::write(&probe, b"").is_ok() {
        let _ = fs::remove_file(&probe);
        eprintln!("skipped: the home stays writable for this user");
        return;
    }

    let listed = fixture.run(&["library", "list"]);
    assert_eq!(listed.code, 0, "{}", listed.stderr);
    line_with(&listed.stdout, "This Mac");
    let fast = line_with(&listed.stdout, "Fast");
    assert!(fast.contains("connected"), "{fast}");
    assert!(listed.stderr.contains("warning:"), "{}", listed.stderr);
    assert!(!listed.stderr.contains("error:"), "{}", listed.stderr);
    assert_eq!(fs::read(&list_path).unwrap(), before);
}

/// A list that cannot be parsed is set aside, with a warning naming where,
/// and the listing still succeeds with This Mac.
#[test]
fn list_warns_about_a_set_aside_library_list() {
    let fixture = Fixture::new("quarantine");
    fs::write(fixture.home.join("libraries.json"), b"not json").unwrap();
    let listed = fixture.run(&["library", "list"]);
    assert_eq!(listed.code, 0, "{}", listed.stderr);
    line_with(&listed.stdout, "This Mac");
    let warning = line_with(&listed.stderr, "warning:");
    assert!(warning.contains("libraries.json.corrupt-"), "{warning}");
}

/// Adopting a library already on a drive reports the models it skipped
/// because another library holds their names, and shows the drive's name
/// with control characters escaped and its length capped.
#[test]
fn adopting_a_drive_library_reports_skipped_models_and_escapes_its_name() {
    let fixture = Fixture::new("adopt");
    let hostile = format!("Archive\u{1b}[31m{}", "x".repeat(100));
    let root = fixture.volume("T9").join("TurboSpark");
    fs::create_dir_all(&root).unwrap();
    let row = |alias: &str| {
        serde_json::json!({
            "alias": alias,
            "repo": format!("owner/{alias}"),
            "revision": "main",
            "relative_path": format!("text/{alias}.gturbo"),
            "family": "qwen3",
            "install_bytes": 4,
            "installed_on": "2026-10-01",
            "status": "runs",
            "modality": "text",
        })
    };
    let marker = serde_json::json!({
        "format": 1,
        "id": "6d1f0c7e-2a4b-4c8d-9e3f-5a6b7c8d9e0f",
        "name": hostile,
        "created_on": "2026-10-01",
        "models": [row("tiny"), row("kept")],
    });
    fs::write(
        root.join("library.json"),
        serde_json::to_vec_pretty(&marker).unwrap(),
    )
    .unwrap();
    // This Mac already holds `tiny`.
    let this_mac_row = serde_json::json!([{
        "alias": "tiny",
        "repo": "owner/tiny",
        "revision": "main",
        "path": fixture.home.join("models/text/tiny.gturbo"),
        "family": "qwen3",
        "install_bytes": 4,
        "installed_on": "2026-10-01",
        "status": "runs",
        "kind": null,
        "modality": "text",
    }]);
    fs::write(
        fixture.home.join("installed.json"),
        serde_json::to_vec_pretty(&this_mac_row).unwrap(),
    )
    .unwrap();

    let path = fixture.volume("T9");
    let added = fixture.run(&["library", "add", path.to_str().unwrap()]);
    assert_eq!(added.code, 0, "{}", added.stderr);
    let skipped = line_with(&added.stdout, "another library");
    assert!(skipped.contains("tiny"), "{skipped}");
    assert!(!skipped.contains("kept"), "{skipped}");

    let listed = fixture.run(&["library", "list"]);
    assert_eq!(listed.code, 0, "{}", listed.stderr);
    for output in [&added.stdout, &listed.stdout] {
        assert!(!output.contains('\u{1b}'), "{output}");
        assert!(output.contains("Archive\\u{1b}[31m"), "{output}");
        assert!(!output.contains(&"x".repeat(64)), "{output}");
    }
}

/// What a pull into the unplugged library "T7" fails with (10.4).
const T7_NOT_CONNECTED: &str =
    "the library \"T7\" is not connected; connect its drive and try again";

/// A registry row for the text install `alias` at `path`, in the library
/// `(id, path relative to its root)`, or with no library for `None`.
fn text_row(alias: &str, path: &Path, library: Option<(&str, &str)>) -> serde_json::Value {
    let mut row = serde_json::json!({
        "alias": alias,
        "repo": format!("owner/{alias}"),
        "revision": "main",
        "path": path,
        "family": "qwen3",
        "install_bytes": 4,
        "installed_on": "2026-10-01",
        "status": "runs",
        "kind": null,
        "modality": "text",
    });
    if let Some((id, relative)) = library {
        row["library"] = serde_json::json!(id);
        row["relative_path"] = serde_json::json!(relative);
    }
    row
}

impl Fixture {
    fn write_rows(&self, rows: &[serde_json::Value]) {
        fs::write(
            self.home.join("installed.json"),
            serde_json::to_vec_pretty(rows).unwrap(),
        )
        .unwrap();
    }
}

/// A pull into a library whose drive is gone fails naming the library,
/// before any network request (10.4). Every pull command takes the library,
/// and `pull --repo`, whose probe is itself a request, checks it first too.
#[test]
fn a_pull_into_an_unplugged_library_fails_before_any_network_request() {
    let fixture = Fixture::new("pull-unplugged");
    fixture.add("T7", "T7");
    fs::remove_dir_all(fixture.volume("T7")).unwrap();
    let hub = Hub::start();
    for args in [
        &["pull", "tinyllama", "--library", "T7"][..],
        &[
            "pull",
            "--repo",
            "owner/name",
            "--alias",
            "mine",
            "--library",
            "t7",
        ],
        &["pull-vision", "qwen38-vision-tower", "--library", "T7"],
        &["pull-image", "--alias", "z-image-turbo", "--library", "T7"],
        &["pull-audio", "minimax-music3-mxfp8", "--library", "T7"],
    ] {
        let outcome = fixture.run_against(&hub, args);
        assert_eq!(outcome.code, 1, "{args:?}: {}", outcome.stderr);
        assert!(
            outcome.stderr.contains(T7_NOT_CONNECTED),
            "{args:?}: {}",
            outcome.stderr
        );
        assert_eq!(hub.connections(), 0, "{args:?} reached the network");
    }
    // Nothing rebuilt the library's folder on the disk under the mount point.
    assert!(!fixture.volume("T7").exists());
    assert!(!fixture.home.join("installed.json").exists());
}

/// The destination's other refusals also come before the network: a
/// library a newer TurboSpark wrote, an alias another library holds, and an
/// alias that is not a plain name. For `pull --repo` these used to follow
/// the probe's metadata request.
#[test]
fn a_refused_destination_fails_before_any_network_request() {
    let fixture = Fixture::new("pull-refused");
    fixture.add("T7", "T7");
    fixture.add("T9", "T9");
    let marker = fixture.volume("T7").join("TurboSpark").join("library.json");
    let mut json: serde_json::Value = serde_json::from_slice(&fs::read(&marker).unwrap()).unwrap();
    json["format"] = serde_json::json!(2);
    fs::write(&marker, serde_json::to_vec_pretty(&json).unwrap()).unwrap();
    let text = fixture.home.join("models").join("text");
    fixture.write_rows(&[
        text_row("tinyllama", &text.join("tinyllama.gturbo"), None),
        text_row("mine", &text.join("mine.gturbo"), None),
    ]);
    let rows = fs::read(fixture.home.join("installed.json")).unwrap();
    let hub = Hub::start();
    let repo = |alias: &'static str, library: Option<&'static str>| {
        let mut args = vec!["pull", "--repo", "owner/name", "--alias", alias];
        if let Some(name) = library {
            args.extend(["--library", name]);
        }
        args
    };
    for (args, expected) in [
        (
            vec!["pull", "tinyllama", "--library", "T7"],
            "is read-only here",
        ),
        (repo("fresh", Some("T7")), "is read-only here"),
        (
            vec!["pull", "tinyllama", "--library", "T9"],
            "already installed in This Mac",
        ),
        (repo("mine", Some("T9")), "already installed in This Mac"),
        (repo("../escape", None), "is not a valid install alias"),
    ] {
        let outcome = fixture.run_against(&hub, &args);
        assert_eq!(outcome.code, 1, "{args:?}: {}", outcome.stderr);
        assert!(
            outcome.stderr.contains(expected),
            "{args:?}: {}",
            outcome.stderr
        );
        assert_eq!(hub.connections(), 0, "{args:?} reached the network");
    }
    assert_eq!(fs::read(fixture.home.join("installed.json")).unwrap(), rows);
}

/// Without `--library` a pull goes to the default library, and when that
/// library is not connected, to This Mac with a note saying so (2.2), and
/// then on to the network. Where each pull was headed shows before any
/// request: an install already in that folder stops it there.
#[test]
fn a_pull_falls_back_to_this_mac_when_the_default_library_is_unplugged() {
    let fixture = Fixture::new("pull-default");
    fixture.add("T7", "T7");
    let made_default = fixture.run(&["library", "default", "T7"]);
    assert_eq!(made_default.code, 0, "{}", made_default.stderr);
    let hub = Hub::start();
    let holds = |dir: &Path| {
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join("manifest.json"), b"{}").unwrap();
        format!("{} already holds an install", dir.display())
    };

    let in_library = fixture
        .volume("T7")
        .join("TurboSpark")
        .join("text")
        .join("tinyllama.gturbo");
    let expected = holds(&in_library);
    let connected = fixture.run_against(&hub, &["pull", "tinyllama"]);
    assert_eq!(connected.code, 1, "{}", connected.stderr);
    assert!(connected.stderr.contains(&expected), "{}", connected.stderr);
    assert!(!connected.stderr.contains("note:"), "{}", connected.stderr);

    fs::remove_dir_all(fixture.volume("T7")).unwrap();
    let on_mac = fixture
        .home
        .join("models")
        .join("text")
        .join("tinyllama.gturbo");
    let expected = holds(&on_mac);
    let fallback = fixture.run_against(&hub, &["pull", "tinyllama"]);
    assert_eq!(fallback.code, 1, "{}", fallback.stderr);
    assert!(fallback.stderr.contains(&expected), "{}", fallback.stderr);
    let note = line_with(&fallback.stderr, "note:");
    assert!(
        note.contains("the default library \"T7\" is not connected"),
        "{note}"
    );
    assert!(note.contains("This Mac"), "{note}");
    assert_eq!(hub.connections(), 0);

    // With nothing in the way, the pull carries on to the network.
    fs::remove_dir_all(&on_mac).unwrap();
    let onward = fixture.run_against(&hub, &["pull", "tinyllama"]);
    assert_eq!(onward.code, 1, "{}", onward.stderr);
    line_with(&onward.stderr, "note: the default library");
    assert!(
        !onward.stderr.contains("connect its drive"),
        "{}",
        onward.stderr
    );
    assert!(
        hub.connections() > 0,
        "the pull stopped before the network: {}",
        onward.stderr
    );
    assert!(!fixture.volume("T7").exists());
}

/// `--out` and `--library` each name a destination, so together they are a
/// usage error, and nothing is created.
#[test]
fn out_and_library_together_are_a_usage_error() {
    let fixture = Fixture::new("pull-out");
    fixture.add("T7", "T7");
    let out = fixture.base.join("out");
    for command in [
        &["pull", "tinyllama"][..],
        &["pull-vision", "qwen38-vision-tower"],
        &["pull-image", "--alias", "z-image-turbo"],
    ] {
        let mut args = command.to_vec();
        args.extend(["--out", out.to_str().unwrap(), "--library", "T7"]);
        let outcome = fixture.run(&args);
        assert_eq!(outcome.code, 2, "{args:?}: {}", outcome.stderr);
        let first = outcome.stderr.lines().next().unwrap_or_default();
        assert!(first.contains("--out and --library"), "{args:?}: {first}");
    }
    assert!(!out.exists());
}

/// `list` shows every install with its library and whether it can be
/// opened now (10.2): available, not connected, and missing, plus a `pull
/// --out` row, which has no library and is not This Mac's. A library name
/// from a drive is escaped and capped.
#[test]
fn list_shows_each_install_with_its_location_and_status() {
    let fixture = Fixture::new("list-installed");
    fixture.add("T7", "T7");
    let hostile = format!("Away\u{1b}[31m{}", "x".repeat(100));
    fixture.add("T9", &hostile);
    let (t7, t9) = (fixture.id_of("T7"), fixture.id_of(&hostile));
    let t7_root = fixture.volume("T7").join("TurboSpark");
    let t9_root = fixture.volume("T9").join("TurboSpark");
    fs::create_dir_all(t7_root.join("text").join("qq-ready.gturbo")).unwrap();
    let custom = fixture.base.join("elsewhere").join("qq-custom.gturbo");
    fs::create_dir_all(&custom).unwrap();
    let gone = fixture
        .home
        .join("models")
        .join("text")
        .join("qq-gone.gturbo");
    fixture.write_rows(&[
        text_row(
            "qq-ready",
            &t7_root.join("text").join("qq-ready.gturbo"),
            Some((&t7, "text/qq-ready.gturbo")),
        ),
        text_row(
            "qq-offline",
            &t9_root.join("text").join("qq-offline.gturbo"),
            Some((&t9, "text/qq-offline.gturbo")),
        ),
        text_row("qq-gone", &gone, None),
        text_row("qq-custom", &custom, None),
    ]);
    fs::remove_dir_all(fixture.volume("T9")).unwrap();

    let listed = fixture.run(&["list"]);
    assert_eq!(listed.code, 0, "{}", listed.stderr);
    // The catalog is still listed.
    assert!(listed.stdout.contains("gemma4"), "{}", listed.stdout);
    let header = line_with(&listed.stdout, "LOCATION");
    assert!(header.contains("STATUS"), "{header}");
    for (alias, location, status) in [
        ("qq-ready", "T7", "available"),
        ("qq-offline", "Away\\u{1b}[31m", "not connected"),
        ("qq-gone", "This Mac", "missing"),
        ("qq-custom", "custom location", "available"),
    ] {
        let line = line_with(&listed.stdout, alias);
        assert!(line.contains(location), "{location} in {line}");
        assert!(line.contains(status), "{status} in {line}");
    }
    assert!(!line_with(&listed.stdout, "qq-ready").contains("not connected"));
    assert!(!listed.stdout.contains('\u{1b}'), "{}", listed.stdout);
    assert!(
        !listed.stdout.contains(&"x".repeat(64)),
        "{}",
        listed.stdout
    );

    // `--filter` narrows the installs too, even with no catalog row left.
    let filtered = fixture.run(&["list", "--filter", "qq-off"]);
    assert_eq!(filtered.code, 0, "{}", filtered.stderr);
    line_with(&filtered.stdout, "qq-offline");
    assert!(!filtered.stdout.contains("qq-ready"), "{}", filtered.stdout);
}

/// Serves one repository, `owner/model` at [`REVISION`], the way the Hub
/// answers metadata, HEAD, and ranged file requests.
struct FileHub {
    endpoint: String,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

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

/// A text pull into a library lands in it end to end (10.3): the folder is
/// on the library's drive, the row and the drive's marker list it, and
/// `list` shows it in that library, available. Nothing lands in This Mac.
#[test]
fn a_text_pull_installs_into_the_named_library() {
    let fixture = Fixture::new("pull-into");
    fixture.add("T7", "T7");
    let (mut gguf, _) = repack::build_synthetic_gemma4_gguf(repack::SyntheticGgufShape::default());
    // Past the probe's first header read (1 MiB), as every real checkpoint
    // is. A shorter file answers that read short, and the reader then backs
    // off for seconds before asking for the size.
    gguf.resize(gguf.len().max(1 << 21), 0);
    let tokenizer =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../tokenizer/tests/fixtures/ChatMLTokenizer");
    let files = BTreeMap::from([
        ("model-Q8_0.gguf".to_string(), gguf),
        (
            "tokenizer.json".to_string(),
            fs::read(tokenizer.join("tokenizer.json")).unwrap(),
        ),
        (
            "tokenizer_config.json".to_string(),
            fs::read(tokenizer.join("tokenizer_config.json")).unwrap(),
        ),
    ]);
    let hub = FileHub::start(files);
    let repo = format!("owner/model@{REVISION}");
    let out = fixture
        .command_with_home_device(fixture.device.wrapping_add(1))
        .env("HF_ENDPOINT", &hub.endpoint)
        .args([
            "pull",
            "--repo",
            &repo,
            "--alias",
            "qwen",
            "--file",
            "model-Q8_0.gguf",
            "--library",
            "T7",
            "--force",
        ])
        .output()
        .expect("the binary runs");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stdout}\n{stderr}");

    let root = fixture.volume("T7").join("TurboSpark");
    let dir = root.join("text").join("qwen.gturbo");
    assert!(dir.join("model_weights.bin").is_file());
    assert!(stdout.contains(&dir.display().to_string()), "{stdout}");
    let rows: serde_json::Value =
        serde_json::from_slice(&fs::read(fixture.home.join("installed.json")).unwrap()).unwrap();
    assert_eq!(rows[0]["alias"], "qwen");
    assert_eq!(rows[0]["library"], fixture.id_of("T7"));
    assert_eq!(rows[0]["relative_path"], "text/qwen.gturbo");
    let marker: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("library.json")).unwrap()).unwrap();
    assert_eq!(marker["models"][0]["alias"], "qwen");
    assert!(!fixture.home.join("models").join("text").exists());

    let listed = fixture.run(&["list", "--filter", "qwen"]);
    let line = line_with(&listed.stdout, "qwen ");
    assert!(
        line.contains("T7") && line.contains("available"),
        "{}",
        listed.stdout
    );
}

/// A tiny local Diffusers export the image packer accepts: tokenizer,
/// scheduler, and one F32 shard per weight component.
fn write_image_source(source: &Path) {
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
        let header = serde_json::to_vec(&serde_json::json!({
            "linear.weight": {"dtype": "F32", "shape": [1, 64], "data_offsets": [0, 256]},
            "norm.weight": {"dtype": "F32", "shape": [64], "data_offsets": [256, 512]}
        }))
        .unwrap();
        let mut shard = (header.len() as u64).to_le_bytes().to_vec();
        shard.extend_from_slice(&header);
        shard.extend_from_slice(&payload);
        fs::write(dir.join(file), shard).unwrap();
        if component != "vae" {
            fs::write(
                dir.join(index),
                serde_json::to_vec(&serde_json::json!({
                    "weight_map": {"linear.weight": file, "norm.weight": file}
                }))
                .unwrap(),
            )
            .unwrap();
        }
    }
}

/// An image pull into a library builds in the library's staging folder and
/// publishes into its image folder, leaving nothing in staging or in This
/// Mac. Its folder then holds the alias: another pull into the library, or
/// into This Mac, is refused before any work.
#[test]
fn an_image_pull_installs_into_the_named_library() {
    let fixture = Fixture::new("pull-image");
    fixture.add("T7", "T7");
    let source = fixture.base.join("export");
    write_image_source(&source);
    let before = snapshot(&source);
    let pull = |library: Option<&str>| {
        let mut args = vec![
            "pull-image",
            "--source",
            source.to_str().unwrap(),
            "--alias",
            "pic",
            "--model-id",
            "Tongyi-MAI/Z-Image-Turbo",
            "--model-revision",
            "f332072aa78be7aecdf3ee76d5c247082da564a6",
        ];
        if let Some(name) = library {
            args.extend(["--library", name]);
        }
        fixture.run(&args)
    };

    let installed = pull(Some("T7"));
    assert_eq!(installed.code, 0, "{}", installed.stderr);
    let root = fixture.volume("T7").join("TurboSpark");
    let dir = root.join("image").join("pic.gturbo");
    assert!(dir.join("manifest.json").is_file(), "{}", installed.stdout);
    assert!(
        installed.stdout.contains(&dir.display().to_string()),
        "{}",
        installed.stdout
    );
    assert!(
        fs::read_dir(root.join(".staging"))
            .unwrap()
            .next()
            .is_none(),
        "staging is left empty"
    );
    assert!(!fixture.home.join("models").exists());
    assert_eq!(snapshot(&source), before, "the export is not touched");

    let again = pull(Some("T7"));
    assert_eq!(again.code, 1, "{}", again.stderr);
    assert!(
        again.stderr.contains("already holds an image install"),
        "{}",
        again.stderr
    );
    let elsewhere = pull(None);
    assert_eq!(elsewhere.code, 1, "{}", elsewhere.stderr);
    assert!(
        elsewhere
            .stderr
            .contains("already installed in the library \"T7\""),
        "{}",
        elsewhere.stderr
    );
    assert!(!fixture.home.join("models").exists());
}

/// A recorded text install of `alias` on This Mac, with a nested folder,
/// and its registry row as the only one. Returns its folder.
fn this_mac_text_install(fixture: &Fixture, alias: &str) -> PathBuf {
    let dir = fixture
        .home
        .join("models")
        .join("text")
        .join(format!("{alias}.gturbo"));
    fs::create_dir_all(dir.join("tokenizer")).unwrap();
    fs::write(dir.join("manifest.json"), b"{\"format\": 1}").unwrap();
    fs::write(dir.join("model_weights.bin"), vec![7u8; 5000]).unwrap();
    fs::write(dir.join("tokenizer").join("tokenizer.json"), b"{}").unwrap();
    fixture.write_rows(&[text_row(alias, &dir, None)]);
    dir
}

/// Every entry under `root` by its path relative to `root`, with the bytes
/// of each file, so a copy compares equal to its original.
fn relative_snapshot(root: &Path) -> BTreeMap<PathBuf, Option<Vec<u8>>> {
    snapshot(root)
        .into_iter()
        .map(|(path, bytes)| (path.strip_prefix(root).unwrap().to_path_buf(), bytes))
        .collect()
}

/// The aliases the marker of the library on `volume` lists.
fn marker_aliases(fixture: &Fixture, volume: &str) -> Vec<String> {
    let path = fixture
        .volume(volume)
        .join("TurboSpark")
        .join("library.json");
    let marker: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    marker["models"]
        .as_array()
        .map(|rows| {
            rows.iter()
                .map(|row| row["alias"].as_str().unwrap().to_string())
                .collect()
        })
        .unwrap_or_default()
}

/// The `move-*` folders a move left in the staging folder of `root`.
fn move_leftovers(root: &Path) -> Vec<PathBuf> {
    match fs::read_dir(root.join(".staging")) {
        Ok(entries) => entries
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("move-"))
            })
            .collect(),
        Err(_) => Vec::new(),
    }
}

#[test]
fn the_usage_text_documents_move_and_its_missing_process_guard() {
    let fixture = Fixture::new("move-usage");
    let help = fixture.run(&["--help"]);
    assert_eq!(help.code, 0);
    assert!(
        help.stdout.contains("move <ALIAS> --to <LIBRARY>"),
        "{}",
        help.stdout
    );
    assert!(
        help.stdout.contains("no cross-process guard"),
        "{}",
        help.stdout
    );

    for args in [
        &["move", "qq-move"][..],
        &["move", "--to", "T7"],
        &["move", "qq-move", "--to", "T7", "--yes"],
        &["move", "qq-move", "--to", "T7", "--kind", "embedding"],
    ] {
        let outcome = fixture.run(args);
        assert_eq!(outcome.code, 2, "{args:?}: {}", outcome.stderr);
    }
}

/// A text install moves This Mac to T7, T7 to T9, and T9 back to This Mac
/// (10.6, 7.1). Each time the files land whole at the destination, `list`
/// shows the new library, the original is gone, and the markers follow.
#[test]
fn a_text_install_moves_between_libraries_and_back_to_this_mac() {
    let fixture = Fixture::new("move-text");
    fixture.add("T7", "T7");
    fixture.add("T9", "T9");
    let original = this_mac_text_install(&fixture, "qq-move");
    let files = relative_snapshot(&original);
    let t7 = fixture.volume("T7").join("TurboSpark");
    let t9 = fixture.volume("T9").join("TurboSpark");

    let mut from = original.clone();
    for (target, to, shown) in [
        ("T7", t7.join("text").join("qq-move.gturbo"), "T7"),
        // Library names resolve ignoring case, as for `library default`.
        ("t9", t9.join("text").join("qq-move.gturbo"), "T9"),
        ("This Mac", original.clone(), "This Mac"),
    ] {
        let moved = fixture.run(&["move", "qq-move", "--to", target]);
        assert_eq!(moved.code, 0, "{target}: {}", moved.stderr);
        assert!(
            moved.stdout.contains(&format!("moved qq-move to {shown}")),
            "{target}: {}",
            moved.stdout
        );
        // Plain progress lines, since stderr is not a terminal here.
        assert!(moved.stderr.contains("[move] 100%"), "{}", moved.stderr);
        assert_eq!(relative_snapshot(&to), files, "{target}");
        assert!(!from.exists(), "{target}: the original is deleted");

        let listed = fixture.run(&["list", "--filter", "qq-move"]);
        assert_eq!(listed.code, 0, "{}", listed.stderr);
        let line = line_with(&listed.stdout, "qq-move");
        assert!(line.contains(shown), "{target}: {line}");
        assert!(line.contains("available"), "{target}: {line}");

        let listed_in = |library: &str| {
            if shown == library {
                vec!["qq-move".to_string()]
            } else {
                Vec::new()
            }
        };
        assert_eq!(marker_aliases(&fixture, "T7"), listed_in("T7"), "{target}");
        assert_eq!(marker_aliases(&fixture, "T9"), listed_in("T9"), "{target}");
        for root in [&t7, &t9, &fixture.home] {
            assert!(move_leftovers(root).is_empty(), "{}", root.display());
        }
        from = to;
    }

    let rows: serde_json::Value =
        serde_json::from_slice(&fs::read(fixture.home.join("installed.json")).unwrap()).unwrap();
    assert_eq!(rows.as_array().unwrap().len(), 1, "{rows}");
    assert!(rows[0]["library"].is_null(), "{rows}");
    assert_eq!(rows[0]["path"], original.display().to_string());
}

/// An image install has no registry row: its folder is found in the one
/// library that holds it, and moves without touching any marker.
#[test]
fn an_image_folder_moves_from_a_library_to_this_mac() {
    let fixture = Fixture::new("move-image");
    fixture.add("T7", "T7");
    let root = fixture.volume("T7").join("TurboSpark");
    let source = root.join("image").join("pic.gturbo");
    fs::create_dir_all(source.join("transformer")).unwrap();
    fs::write(source.join("manifest.json"), b"{}").unwrap();
    fs::write(
        source.join("transformer").join("weights.bin"),
        vec![3u8; 3000],
    )
    .unwrap();
    let files = relative_snapshot(&source);
    let marker = fs::read(root.join("library.json")).unwrap();

    let moved = fixture.run(&["move", "pic", "--to", "this mac"]);
    assert_eq!(moved.code, 0, "{}", moved.stderr);
    assert!(
        moved.stdout.contains("moved pic to This Mac"),
        "{}",
        moved.stdout
    );
    let destination = fixture.home.join("models").join("image").join("pic.gturbo");
    assert_eq!(relative_snapshot(&destination), files);
    assert!(!source.exists());
    assert_eq!(fs::read(root.join("library.json")).unwrap(), marker);
    assert!(move_leftovers(&root).is_empty());
    assert!(move_leftovers(&fixture.home).is_empty());
}

/// A destination with less free space than the install is refused before
/// anything is copied, naming both sizes (7.4), and nothing changes.
#[test]
fn a_move_is_refused_before_copying_when_the_destination_is_too_small() {
    let fixture = Fixture::new("move-space");
    fixture.add("T7", "T7");
    let original = this_mac_text_install(&fixture, "qq-move");
    let needed: u64 = snapshot(&original)
        .values()
        .flatten()
        .map(|bytes| bytes.len() as u64)
        .sum();
    let home = snapshot(&fixture.home);
    let drive = snapshot(&fixture.volume("T7"));

    let out = fixture
        .command_with_home_device(fixture.device.wrapping_add(1))
        .env("TURBOSPARK_TEST_FREE_BYTES", "10")
        .args(["move", "qq-move", "--to", "T7"])
        .output()
        .expect("the binary runs");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    let error = line_with(&stderr, "error:");
    assert!(error.contains("10 B free"), "{error}");
    assert!(
        error.contains(&format!("needs {}", catalog::human_bytes(needed))),
        "{error}"
    );
    assert!(error.contains("nothing was moved"), "{error}");
    // Byte for byte as before, with no staging folder left on either side.
    assert_eq!(snapshot(&fixture.home), home);
    assert_eq!(snapshot(&fixture.volume("T7")), drive);
}

/// An alias held by installs of more than one kind is a usage error listing
/// them, and `--kind` picks one.
#[test]
fn an_alias_of_several_kinds_needs_kind_to_move() {
    let fixture = Fixture::new("move-kinds");
    fixture.add("T7", "T7");
    fixture.add("T9", "T9");
    let text = this_mac_text_install(&fixture, "dup");
    let image = fixture
        .volume("T7")
        .join("TurboSpark")
        .join("image")
        .join("dup.gturbo");
    fs::create_dir_all(&image).unwrap();
    fs::write(image.join("manifest.json"), b"{}").unwrap();

    let ambiguous = fixture.run(&["move", "dup", "--to", "T9"]);
    assert_eq!(ambiguous.code, 2, "{}", ambiguous.stderr);
    let first = ambiguous.stderr.lines().next().unwrap_or_default();
    for phrase in ["--kind", "text in This Mac", "image in T7"] {
        assert!(first.contains(phrase), "{phrase:?} in {first}");
    }
    assert!(text.is_dir() && image.is_dir());

    let moved = fixture.run(&["move", "dup", "--kind", "image", "--to", "T9"]);
    assert_eq!(moved.code, 0, "{}", moved.stderr);
    assert!(fixture
        .volume("T9")
        .join("TurboSpark")
        .join("image")
        .join("dup.gturbo")
        .join("manifest.json")
        .is_file());
    assert!(!image.exists());
    assert!(text.is_dir(), "the text install stays");
}

/// Embedding and OCR models stay on This Mac (2.5).
#[test]
fn an_embedding_model_is_refused_as_home_only() {
    let fixture = Fixture::new("move-embedding");
    fixture.add("T7", "T7");
    let embedding = fixture
        .home
        .join("models")
        .join("embeddings")
        .join("gemma-embed");
    fs::create_dir_all(&embedding).unwrap();
    fs::write(embedding.join("model.safetensors"), b"weights").unwrap();

    let refused = fixture.run(&["move", "gemma-embed", "--to", "T7"]);
    assert_eq!(refused.code, 1, "{}", refused.stderr);
    let error = line_with(&refused.stderr, "error:");
    assert!(
        error.contains("embedding and OCR models stay on This Mac"),
        "{error}"
    );
    assert!(embedding.join("model.safetensors").is_file());
    assert!(!fixture
        .volume("T7")
        .join("TurboSpark")
        .join("embeddings")
        .exists());
}

/// A destination or source library that is not connected names the drive
/// to connect, and nothing moves.
#[test]
fn a_move_with_an_unplugged_library_names_the_drive_to_connect() {
    let fixture = Fixture::new("move-unplugged");
    fixture.add("T7", "T7");
    fixture.add("T9", "T9");
    let original = this_mac_text_install(&fixture, "qq-move");
    let files = relative_snapshot(&original);
    let t9 = fixture.volume("T9").join("TurboSpark");
    let away = t9.join("text").join("qq-away.gturbo");
    fixture.write_rows(&[
        text_row("qq-move", &original, None),
        text_row(
            "qq-away",
            &away,
            Some((&fixture.id_of("T9"), "text/qq-away.gturbo")),
        ),
    ]);
    let rows = fs::read(fixture.home.join("installed.json")).unwrap();
    fs::remove_dir_all(fixture.volume("T7")).unwrap();
    fs::remove_dir_all(fixture.volume("T9")).unwrap();

    let to_offline = fixture.run(&["move", "qq-move", "--to", "T7"]);
    assert_eq!(to_offline.code, 1, "{}", to_offline.stderr);
    assert!(
        line_with(&to_offline.stderr, "error:").contains(T7_NOT_CONNECTED),
        "{}",
        to_offline.stderr
    );
    assert_eq!(relative_snapshot(&original), files);

    let from_offline = fixture.run(&["move", "qq-away", "--to", "This Mac"]);
    assert_eq!(from_offline.code, 1, "{}", from_offline.stderr);
    let error = line_with(&from_offline.stderr, "error:");
    assert!(
        error.contains("qq-away is in the library \"T9\", which is not connected"),
        "{error}"
    );
    assert!(error.contains("connect its drive"), "{error}");
    assert!(!fixture
        .home
        .join("models")
        .join("text")
        .join("qq-away.gturbo")
        .exists());
    assert_eq!(fs::read(fixture.home.join("installed.json")).unwrap(), rows);
    // Nothing rebuilt either library's folder under its mount point.
    assert!(!fixture.volume("T7").exists() && !fixture.volume("T9").exists());
}

// Running a model whose library is not connected (10.7).

/// What the run and serve commands say about `alias` in the unplugged
/// library T7.
fn connect_t7(alias: &str) -> String {
    format!("Model {alias} is on T7, which is not connected. Connect the drive and try again.")
}

impl Fixture {
    /// `binary` of this crate with this home and the fake volumes.
    fn run_binary(&self, binary: &str, args: &[&str]) -> Outcome {
        let out = self
            .binary_with_home_device(binary, self.device.wrapping_add(1))
            .args(args)
            .output()
            .expect("the binary runs");
        Outcome {
            code: out.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&out.stdout).to_string(),
            stderr: String::from_utf8_lossy(&out.stderr).to_string(),
        }
    }

    /// Mounts the volume `from` again as `to`, as macOS does when another
    /// drive already has the name: the folder moves and the mount table
    /// names the new mount point.
    fn remount(&self, from: &str, to: &str) {
        fs::rename(self.volume(from), self.volume(to)).unwrap();
        let table = fs::read_to_string(&self.mounts).unwrap();
        let old = format!("{} (", self.volume(from).display());
        assert!(table.contains(&old), "{table}");
        let new = format!("{} (", self.volume(to).display());
        fs::write(&self.mounts, table.replace(&old, &new)).unwrap();
    }
}

/// `turbospark-check --model` with an alias whose library is unplugged
/// fails naming the drive to connect, rather than trying the bare alias as
/// a path and reporting no model there.
#[cfg(target_os = "macos")]
#[test]
fn checking_a_model_on_an_unplugged_library_names_the_drive_to_connect() {
    let fixture = Fixture::new("check-unplugged");
    fixture.add("T7", "T7");
    let root = fixture.volume("T7").join("TurboSpark");
    fixture.write_rows(&[text_row(
        "qq-away",
        &root.join("text").join("qq-away.gturbo"),
        Some((&fixture.id_of("T7"), "text/qq-away.gturbo")),
    )]);
    fs::remove_dir_all(fixture.volume("T7")).unwrap();

    let outcome = fixture.run_binary("turbospark-check", &["--model", "qq-away", "--prompt", "hi"]);
    assert_eq!(outcome.code, 1, "{}", outcome.stderr);
    assert!(
        outcome.stderr.contains(&connect_t7("qq-away")),
        "{}",
        outcome.stderr
    );
    assert!(
        !outcome.stderr.contains("manifest.json"),
        "{}",
        outcome.stderr
    );
}

/// After the library's drive mounts under a new name, its alias opens where
/// the library is now, not at the recorded path. The folder holds no model,
/// so the run fails reading the manifest at the path it resolved, which is
/// what this asserts.
#[cfg(target_os = "macos")]
#[test]
fn checking_a_model_follows_its_library_to_a_renamed_drive() {
    let fixture = Fixture::new("check-renamed");
    fixture.add("T7", "T7");
    let recorded = fixture
        .volume("T7")
        .join("TurboSpark")
        .join("text")
        .join("qq-moved.gturbo");
    fs::create_dir_all(&recorded).unwrap();
    fixture.write_rows(&[text_row(
        "qq-moved",
        &recorded,
        Some((&fixture.id_of("T7"), "text/qq-moved.gturbo")),
    )]);
    fixture.remount("T7", "T7 1");
    let now = fixture
        .volume("T7 1")
        .join("TurboSpark")
        .join("text")
        .join("qq-moved.gturbo");

    let outcome = fixture.run_binary("turbospark-check", &["--model", "qq-moved", "--prompt", "hi"]);
    assert_eq!(outcome.code, 1, "{}", outcome.stderr);
    let expected = format!("no manifest.json at {}", now.join("manifest.json").display());
    assert!(outcome.stderr.contains(&expected), "{}", outcome.stderr);
}

/// `turbospark-music` names the drive to connect for an audio alias in an
/// unplugged library, rather than calling it not installed.
#[test]
fn generating_music_with_a_model_on_an_unplugged_library_names_the_drive() {
    let fixture = Fixture::new("music-unplugged");
    fixture.add("T7", "T7");
    let root = fixture.volume("T7").join("TurboSpark");
    let mut row = text_row(
        "qq-song",
        &root.join("audio").join("qq-song.gturbo"),
        Some((&fixture.id_of("T7"), "audio/qq-song.gturbo")),
    );
    row["modality"] = serde_json::json!("audio");
    row["family"] = serde_json::json!("minimax_music3");
    fixture.write_rows(&[row]);
    fs::remove_dir_all(fixture.volume("T7")).unwrap();
    let output = fixture.base.join("song.wav");

    let outcome = fixture.run_binary(
        "turbospark-music",
        &[
            "generate",
            "--model",
            "qq-song",
            "--caption",
            "soft piano",
            "--lyrics",
            "[instrumental]",
            "--duration",
            "0.04",
            "--steps",
            "1",
            "--output",
            output.to_str().unwrap(),
        ],
    );
    assert_eq!(outcome.code, 1, "{}", outcome.stderr);
    assert!(
        outcome.stderr.contains(&connect_t7("qq-song")),
        "{}",
        outcome.stderr
    );
    assert!(!output.exists());
}
