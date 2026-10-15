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
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

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
        let mut path = std::env::current_exe().expect("test binary path");
        path.pop(); // deps/
        path.pop();
        path.push("turbospark-model");
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
