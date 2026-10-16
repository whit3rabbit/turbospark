//! Library management through the C ABI (4.1), driven through the same
//! function bodies the header exposes, against a temp home and fake volumes.
//!
//! The location rules read the mount table and the home folder's device.
//! Debug builds take both from the `TURBOSPARK_TEST_*` overrides, set here in
//! process, so a temp folder stands in for `/Volumes` and a fixture file for
//! `/sbin/mount`. Release builds ignore the overrides by design (a shipped
//! binary must not be able to bypass the network refusal), so this file is
//! debug-only. The home is the process-local root `ts_store_root_set`
//! installs. That root and the environment are process-wide, so every test
//! holds one lock for its whole run.
#![cfg(debug_assertions)]

use std::ffi::{CStr, CString};
use std::fs;
use std::os::raw::{c_char, c_int, c_void};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use catalog::library::{LibraryMarker, PortableRow};
use catalog::ModelModality;
use serde_json::Value;
use turbospark_ffi::{
    abi, ts_catalog_json, ts_install, ts_install_cancel, ts_last_error, ts_libraries_json,
    ts_library_add, ts_library_discover_json, ts_library_remove, ts_library_set_default,
    ts_model_forget, ts_store_import_moved_root, ts_store_root_set, ts_string_free,
    TS_INSTALL_STAGE,
};

/// A library id as a marker would carry it.
const ID: &str = "0f6c2a3e-1b7d-4c55-9e0a-7d2f1c3b4a59";
/// A catalog audio alias, which a library cannot check without a receipt.
const MUSIC: &str = "minimax-music3-4bit";

/// The store root and the environment are process-wide.
static SERIAL: Mutex<()> = Mutex::new(());

// ------------------------------------------------------------------ fixture

/// A temp home and a folder of fake volumes, removed when the test ends,
/// passing or not. Declared last, `_serial` is released after the cleanup.
struct Fixture {
    base: PathBuf,
    home: PathBuf,
    volumes: PathBuf,
    /// The real device of the temp folder. The fake volumes share it, so a
    /// home device unlike it is what makes them "other disks".
    device: u64,
    _serial: MutexGuard<'static, ()>,
}

impl Fixture {
    fn new(tag: &str) -> Self {
        let serial = SERIAL
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let base = std::env::temp_dir().join(format!(
            "turbospark-ffi-libraries-{tag}-{}",
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
        for volume in [
            "T7", "T9", "Extra", "Share", "Stick", "Locked", "Unlisted", "Old",
        ] {
            fs::create_dir_all(volumes.join(volume)).unwrap();
        }
        let at = |volume: &str| volumes.join(volume).display().to_string();
        // `Unlisted` and `Old` have no line: the first is a drive that cannot
        // be identified, the second a moved store, which no location rule
        // judges.
        let table = format!(
            "/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)\n\
             /dev/disk9s1 on {} (apfs, local, nodev, nosuid, journaled, noowners)\n\
             /dev/disk10s1 on {} (apfs, local, nodev, nosuid, journaled, noowners)\n\
             /dev/disk13s1 on {} (apfs, local, nodev, nosuid, journaled, noowners)\n\
             //user@nas/share on {} (smbfs, nodev, nosuid, mounted by user)\n\
             /dev/disk11s1 on {} (msdos, local, nodev, nosuid, noowners)\n\
             /dev/disk12s1 on {} (apfs, local, read-only, journaled)\n",
            at("T7"),
            at("T9"),
            at("Extra"),
            at("Share"),
            at("Stick"),
            at("Locked"),
        );
        let mounts = base.join("mounts.txt");
        fs::write(&mounts, table).unwrap();
        let device = fs::metadata(&base).unwrap().dev();
        let fixture = Self {
            base,
            home,
            volumes,
            device,
            _serial: serial,
        };
        std::env::set_var("TURBOSPARK_TEST_MOUNT_TABLE", &mounts);
        std::env::set_var("TURBOSPARK_TEST_VOLUMES_ROOT", &fixture.volumes);
        fixture.set_home_device(device.wrapping_add(1));
        let home = c(fixture.home.to_str().unwrap());
        assert_eq!(unsafe { ts_store_root_set(home.as_ptr()) }, abi::TS_OK);
        fixture
    }

    fn volume(&self, name: &str) -> PathBuf {
        self.volumes.join(name)
    }

    /// The device the home counts as being on. The default is unlike the
    /// temp folder's, so the fake volumes are drives other than the Mac's
    /// own disk.
    fn set_home_device(&self, device: u64) {
        std::env::set_var("TURBOSPARK_TEST_HOME_DEVICE", device.to_string());
    }

    /// The rows of the home's `installed.json`.
    fn recorded(&self) -> Vec<Value> {
        match fs::read_to_string(self.home.join("installed.json")) {
            Ok(text) => serde_json::from_str(&text).unwrap(),
            Err(_) => Vec::new(),
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
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

fn list_reply() -> Reply {
    reply(|out| unsafe { ts_libraries_json(out) })
}

fn list() -> Vec<Value> {
    list_reply().ok().json().as_array().unwrap().clone()
}

fn add(path: &Path, name: Option<&str>) -> Reply {
    let path = c(path.to_str().unwrap());
    let name = name.map(c);
    reply(|out| unsafe {
        ts_library_add(
            path.as_ptr(),
            name.as_ref().map_or(ptr::null(), |name| name.as_ptr()),
            out,
        )
    })
}

fn remove(id: &str) -> Reply {
    let id = c(id);
    status(unsafe { ts_library_remove(id.as_ptr()) })
}

fn set_default(id: &str) -> Reply {
    let id = c(id);
    status(unsafe { ts_library_set_default(id.as_ptr()) })
}

fn discover(volume: &Path) -> Reply {
    let volume = c(volume.to_str().unwrap());
    reply(|out| unsafe { ts_library_discover_json(volume.as_ptr(), out) })
}

fn forget(install_id: &str) -> Reply {
    let install_id = c(install_id);
    status(unsafe { ts_model_forget(install_id.as_ptr()) })
}

fn import(root: &Path) -> Reply {
    let root = c(root.to_str().unwrap());
    reply(|out| unsafe { ts_store_import_moved_root(root.as_ptr(), out) })
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

fn assert_busy(reply: &Reply) {
    assert_eq!(reply.code, abi::TS_ERR_BUSY, "{}", reply.error);
    assert!(
        reply.error.starts_with("busy: "),
        "wanted the code busy first, got {:?}",
        reply.error
    );
    assert!(reply.error.contains("install"), "{:?}", reply.error);
    assert!(reply.text.is_none(), "a refusal writes no output");
}

fn named<'a>(rows: &'a [Value], name: &str) -> &'a Value {
    rows.iter()
        .find(|row| row["name"] == name)
        .unwrap_or_else(|| panic!("no library named {name} in {rows:?}"))
}

fn is_date(value: &Value) -> bool {
    value.as_str().is_some_and(|text| {
        text.len() == 10
            && text.char_indices().all(|(at, ch)| {
                if at == 4 || at == 7 {
                    ch == '-'
                } else {
                    ch.is_ascii_digit()
                }
            })
    })
}

/// The text install a library on a drive can already hold.
fn qwen_row() -> PortableRow {
    PortableRow {
        alias: "qwen".to_string(),
        repo: "owner/qwen".to_string(),
        revision: "main".to_string(),
        relative_path: PathBuf::from("text/qwen.gturbo"),
        family: "qwen3".to_string(),
        install_bytes: 4_096,
        installed_on: "2026-10-14".to_string(),
        status: "verified".to_string(),
        kind: None,
        modality: ModelModality::Text,
        variant: None,
    }
}

/// A library already on the drive: its marker lists one model whose folder
/// holds a file. Returns the library root.
fn existing_library(volume: &Path) -> PathBuf {
    let root = volume.join("TurboSpark");
    let install = root.join("text/qwen.gturbo");
    fs::create_dir_all(&install).unwrap();
    fs::write(install.join("weights.bin"), b"weights").unwrap();
    let mut marker = LibraryMarker::new(ID.to_string(), "Studio".to_string());
    marker.models = vec![qwen_row()];
    marker.write(&root).unwrap();
    root
}

fn qwen_id() -> String {
    format!("{ID}:text/qwen")
}

// -------------------------------------------------------------------- listing

#[test]
fn a_fresh_home_lists_only_this_mac_as_the_default() {
    let fx = Fixture::new("fresh");
    let rows = list();
    assert_eq!(rows.len(), 1, "{rows:?}");
    let this_mac = &rows[0];
    assert_eq!(this_mac["id"], "internal");
    assert_eq!(this_mac["isInternal"], true);
    assert_eq!(this_mac["available"], true);
    assert_eq!(this_mac["duplicate"], false);
    assert_eq!(this_mac["isDefault"], true);
    assert_eq!(this_mac["modelCount"], 0);
    assert!(this_mac["lastSeen"].is_null(), "{this_mac}");
    assert_eq!(
        this_mac["root"],
        fx.home.join("models").to_str().unwrap(),
        "This Mac is the store's models folder"
    );
}

/// A folder made read-only until dropped, which restores it so a failing test
/// still leaves its temp tree removable. Declared after the [`Fixture`] it is
/// in, so it drops first.
struct Locked(PathBuf);

impl Locked {
    fn new(dir: &Path) -> Self {
        fs::set_permissions(dir, fs::Permissions::from_mode(0o555)).unwrap();
        Self(dir.to_path_buf())
    }
}

impl Drop for Locked {
    fn drop(&mut self) {
        let _ = fs::set_permissions(&self.0, fs::Permissions::from_mode(0o755));
    }
}

/// Bookkeeping trouble is a diagnostic, never a listing error (3.4).
#[test]
fn a_damaged_library_list_still_lists_this_mac() {
    let fx = Fixture::new("damaged-list");
    fs::write(fx.home.join("libraries.json"), b"{ not json").unwrap();

    let rows = list();

    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(rows[0]["id"], "internal");
    let set_aside = fs::read_dir(&fx.home)
        .unwrap()
        .flatten()
        .any(|entry| fs::read(entry.path()).is_ok_and(|bytes| bytes == b"{ not json"));
    assert!(set_aside, "the damaged list is moved aside, not deleted");
}

#[test]
fn a_list_that_cannot_record_dates_is_still_a_list() {
    let fx = Fixture::new("unrecorded-dates");
    add(&fx.volume("T7"), Some("Fast")).ok();
    // A date the refresh would rewrite, then a home it cannot write to.
    let path = fx.home.join("libraries.json");
    let mut stored: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    stored["libraries"][0]["last_seen"] = serde_json::json!("2020-01-01");
    fs::write(&path, stored.to_string()).unwrap();
    let _locked = Locked::new(&fx.home);

    let rows = list();

    let fast = named(&rows, "Fast");
    assert_eq!(fast["available"], true);
    assert!(
        is_date(&fast["lastSeen"]) && fast["lastSeen"] != "2020-01-01",
        "the view shows the date the refresh saw: {fast}"
    );
    assert!(
        fs::read_to_string(&path).unwrap().contains("2020-01-01"),
        "the date could not be written"
    );
}

#[test]
fn adding_a_drive_returns_its_view_and_lists_it_after_this_mac() {
    let fx = Fixture::new("add");
    let t7 = fx.volume("T7");

    let added = add(&t7, Some("Fast")).ok().json();
    let library = &added["library"];
    assert_eq!(library["name"], "Fast");
    assert_eq!(library["root"], t7.join("TurboSpark").to_str().unwrap());
    assert_eq!(library["available"], true);
    assert_eq!(library["duplicate"], false);
    assert_eq!(library["isDefault"], false);
    assert_eq!(library["isInternal"], false);
    assert_eq!(library["modelCount"], 0);
    assert!(is_date(&library["lastSeen"]), "{library}");
    assert_eq!(added["skipped"], serde_json::json!([]));
    let id = library["id"].as_str().unwrap().to_string();
    assert!(!id.is_empty() && id != "internal");
    assert!(
        t7.join("TurboSpark/library.json").is_file(),
        "adding writes the marker that makes the drive recognizable"
    );

    // A missing name and an empty one both derive it from the drive.
    let derived = add(&fx.volume("T9"), None).ok().json();
    assert_eq!(derived["library"]["name"], "T9");
    let empty = add(&fx.volume("Extra"), Some("")).ok().json();
    assert_eq!(empty["library"]["name"], "Extra");

    // Adding a library that is already listed returns it unchanged.
    let again = add(&t7, Some("Other name")).ok().json();
    assert_eq!(again["library"]["id"], id);
    assert_eq!(again["library"]["name"], "Fast");

    let rows = list();
    let names: Vec<&str> = rows
        .iter()
        .map(|row| row["name"].as_str().unwrap())
        .collect();
    assert_eq!(names.len(), 4, "{names:?}");
    assert_eq!(rows[0]["id"], "internal", "This Mac comes first");
    assert_eq!(&names[1..], ["Fast", "T9", "Extra"]);
}

#[test]
fn adding_a_drive_with_a_library_reports_its_models() {
    let fx = Fixture::new("adopt");
    existing_library(&fx.volume("Extra"));

    let added = add(&fx.volume("Extra"), None).ok().json();

    // Adopted rather than created: the marker's name and its model.
    assert_eq!(added["library"]["id"], ID);
    assert_eq!(added["library"]["name"], "Studio");
    assert_eq!(added["library"]["modelCount"], 1);
    assert_eq!(added["skipped"], serde_json::json!([]));
}

#[test]
fn an_adopted_marker_row_that_clashes_is_reported_as_skipped() {
    let fx = Fixture::new("skipped");
    // This Mac already holds a text model by the same alias, so the drive's
    // row is not recorded (the alias stays unique across libraries).
    let home_model = fx.home.join("models/text/qwen.gturbo");
    fs::create_dir_all(&home_model).unwrap();
    let row = serde_json::json!([{
        "alias": "qwen", "repo": "owner/qwen", "revision": "main",
        "path": home_model, "family": "qwen3", "install_bytes": 4096,
        "installed_on": "2026-10-14", "status": "verified"
    }]);
    fs::write(fx.home.join("installed.json"), row.to_string()).unwrap();
    existing_library(&fx.volume("Extra"));

    let added = add(&fx.volume("Extra"), None).ok().json();

    assert_eq!(added["skipped"], serde_json::json!(["qwen"]));
    assert_eq!(added["library"]["modelCount"], 0);
}

// ------------------------------------------------------------------- refusals

#[test]
fn every_location_refusal_round_trips_its_code() {
    let fx = Fixture::new("refusals");

    for (volume, code) in [
        ("Share", "network"),
        ("Stick", "small_file_limit"),
        ("Locked", "read_only"),
        ("Unlisted", "unknown_volume"),
    ] {
        let path = fx.volume(volume);
        assert_refused(&add(&path, None), code);
        assert!(
            !path.join("TurboSpark").exists(),
            "a refused {volume} gets no library folder"
        );
    }

    // Not a folder: nothing there, and a plain file.
    assert_refused(&add(&fx.base.join("nowhere"), None), "not_a_directory");
    let file = fx.base.join("a-file");
    fs::write(&file, b"x").unwrap();
    assert_refused(&add(&file, None), "not_a_directory");

    // Inside the home, however deep.
    assert_refused(&add(&fx.home, None), "inside_home");
    fs::create_dir_all(fx.home.join("deeper")).unwrap();
    assert_refused(&add(&fx.home.join("deeper"), None), "inside_home");

    // On the Mac's own disk: the home's device is the folder's.
    fx.set_home_device(fx.device);
    assert_refused(&add(&fx.volume("T7"), None), "same_disk_as_home");
    fx.set_home_device(fx.device.wrapping_add(1));

    // Inside a library that is listed.
    add(&fx.volume("T7"), Some("Fast")).ok();
    let inner = fx.volume("T7").join("TurboSpark/sub");
    fs::create_dir_all(&inner).unwrap();
    assert_refused(&add(&inner, None), "nested");

    // Every refusal left the list as it was.
    assert_eq!(list().len(), 2);
}

#[test]
fn library_errors_round_trip_their_codes() {
    let fx = Fixture::new("library-errors");

    assert_refused(&remove("no-such-library"), "unknown_library");
    assert_refused(&set_default("no-such-library"), "unknown_library");
    assert_refused(&remove("internal"), "internal_library");

    // A marker that exists but cannot be read is never overwritten.
    let marker = fx.volume("Extra").join("TurboSpark/library.json");
    fs::create_dir_all(marker.parent().unwrap()).unwrap();
    fs::write(&marker, b"{ not a marker").unwrap();
    assert_refused(&add(&fx.volume("Extra"), None), "marker");
    assert_eq!(fs::read(&marker).unwrap(), b"{ not a marker");
}

/// A refusal whose sentence carries text from a drive keeps its code first,
/// holds no raw control or direction character, and stays within the code
/// plus a 300 character sentence.
fn assert_plain_refusal(reply: &Reply, code: &str) {
    assert_refused(reply, code);
    let error = &reply.error;
    assert!(
        !error
            .chars()
            .any(|c| c.is_control()
                || matches!(c, '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')),
        "a raw control or direction character reached the message: {error:?}"
    );
    assert!(
        error.chars().count() <= code.len() + ": ".len() + 300,
        "the sentence is capped, got {} characters",
        error.chars().count()
    );
}

/// Text a drive supplies (a library's name, a path) reaches the message of a
/// refusal through the same escape and cap the code is placed in front of,
/// for a library refusal and for a forget alike. Pinned on the entry points
/// themselves, because a refusal that skipped that step would still carry
/// its code and pass every other test.
#[test]
fn untrusted_text_is_escaped_and_capped_behind_an_intact_code() {
    let fx = Fixture::new("hostile-text");
    let hostile = format!("evil\u{1b}[31m\u{202e}{}", "x".repeat(2000));

    // A library on a drive whose marker names it so, adopted as is.
    let root = fx.volume("T7").join("TurboSpark");
    fs::create_dir_all(root.join("text/qwen.gturbo")).unwrap();
    let mut marker = LibraryMarker::new(ID.to_string(), hostile.clone());
    marker.models = vec![qwen_row()];
    marker.write(&root).unwrap();
    add(&fx.volume("T7"), None).ok();

    // Nesting inside it names it in the refusal.
    let inner = root.join("sub");
    fs::create_dir_all(&inner).unwrap();
    let nested = add(&inner, None);
    assert_plain_refusal(&nested, "nested");
    assert!(nested.error.contains("evil"), "{:?}", nested.error);

    // So does forgetting one of its installs while it is away.
    let away = fx.volume("T7").join("Away");
    fs::rename(&root, &away).unwrap();
    let forgotten = forget(&qwen_id());
    assert_plain_refusal(&forgotten, "not_connected");
    assert!(forgotten.error.contains("evil"), "{:?}", forgotten.error);
    assert_eq!(fx.recorded().len(), 1, "a refused forget keeps the row");
    fs::rename(&away, &root).unwrap();

    // A path with control and direction characters in its name.
    let path = fx
        .base
        .join(format!("no\u{1b}[31m\u{202e}where{}", "y".repeat(2000)));
    let missing = add(&path, None);
    assert_plain_refusal(&missing, "not_a_directory");
    assert!(missing.error.contains("where"), "{:?}", missing.error);
}

#[test]
fn a_list_from_a_newer_turbospark_is_refused_as_io() {
    let fx = Fixture::new("newer-list");
    fs::write(
        fx.home.join("libraries.json"),
        br#"{"format": 99, "libraries": []}"#,
    )
    .unwrap();

    let refused = set_default("internal");

    assert_refused(&refused, "io");
    assert!(refused.error.contains("newer"), "{:?}", refused.error);
}

// ---------------------------------------------------------------- remove, default

#[test]
fn removing_a_library_keeps_every_file_and_adding_it_again_restores_its_models() {
    let fx = Fixture::new("remove");
    let root = existing_library(&fx.volume("Extra"));
    let marker_before = fs::read(root.join("library.json")).unwrap();
    add(&fx.volume("Extra"), None).ok();
    assert_eq!(fx.recorded().len(), 1, "the drive's model is listed");

    remove(ID).ok();

    let rows = list();
    assert_eq!(rows.len(), 1, "only This Mac is left: {rows:?}");
    assert!(fx.recorded().is_empty(), "its models stop being listed");
    assert_eq!(
        fs::read(root.join("text/qwen.gturbo/weights.bin")).unwrap(),
        b"weights",
        "the model's files stay"
    );
    assert_eq!(
        fs::read(root.join("library.json")).unwrap(),
        marker_before,
        "the marker stays as it was"
    );

    let again = add(&fx.volume("Extra"), None).ok().json();
    assert_eq!(again["library"]["id"], ID);
    assert_eq!(again["library"]["modelCount"], 1);
    assert_eq!(fx.recorded().len(), 1);
}

#[test]
fn a_second_default_replaces_the_first() {
    let fx = Fixture::new("default");
    let t7 = add(&fx.volume("T7"), Some("Fast")).ok().json();
    let t9 = add(&fx.volume("T9"), Some("Spare")).ok().json();
    let (t7, t9) = (
        t7["library"]["id"].as_str().unwrap().to_string(),
        t9["library"]["id"].as_str().unwrap().to_string(),
    );
    let defaults = || -> Vec<String> {
        list()
            .iter()
            .filter(|row| row["isDefault"] == true)
            .map(|row| row["name"].as_str().unwrap().to_string())
            .collect()
    };
    assert_eq!(defaults(), ["This Mac"], "This Mac is the initial default");

    set_default(&t7).ok();
    assert_eq!(defaults(), ["Fast"]);

    set_default(&t9).ok();
    assert_eq!(
        defaults(),
        ["Spare"],
        "the first default is replaced, not joined"
    );

    set_default("internal").ok();
    assert_eq!(defaults(), ["This Mac"]);

    // Removing the default hands it back to This Mac.
    set_default(&t7).ok();
    remove(&t7).ok();
    assert_eq!(defaults(), ["This Mac"]);
}

// -------------------------------------------------------------------- discovery

#[test]
fn discovery_summarizes_a_marker_or_answers_null() {
    let fx = Fixture::new("discover");
    let extra = fx.volume("Extra");

    // Nothing to find on a drive with no library, or at a path with no drive.
    assert_eq!(discover(&extra).ok().text.as_deref(), Some("null"));
    assert_eq!(
        discover(&fx.base.join("nowhere")).ok().text.as_deref(),
        Some("null")
    );

    existing_library(&extra);
    let found = discover(&extra).ok().json();
    assert_eq!(found["id"], ID);
    assert_eq!(found["name"], "Studio");
    assert_eq!(found["modelCount"], 1);
    assert_eq!(found["readOnly"], false);
    assert!(is_date(&found["createdOn"]), "{found}");
    // Looking is not adding.
    assert_eq!(list().len(), 1);

    // A marker a newer TurboSpark wrote can be shown but not changed.
    let marker = extra.join("TurboSpark/library.json");
    let mut value: Value = serde_json::from_slice(&fs::read(&marker).unwrap()).unwrap();
    value["format"] = serde_json::json!(2);
    fs::write(&marker, value.to_string()).unwrap();
    assert_eq!(discover(&extra).ok().json()["readOnly"], true);

    let no_volume = reply(|out| unsafe { ts_library_discover_json(ptr::null(), out) });
    assert_eq!(
        no_volume.code,
        abi::TS_ERR_INVALID_ARGUMENT,
        "{}",
        no_volume.error
    );
}

/// A drive is untrusted: discovery answers `null` for a marker it cannot
/// read and for a library folder that is a link, and never follows the link.
#[test]
fn discovery_answers_null_for_a_marker_it_cannot_trust() {
    let fx = Fixture::new("discover-untrusted");

    // A marker that is not a marker.
    let extra = fx.volume("Extra");
    let marker = extra.join("TurboSpark/library.json");
    fs::create_dir_all(marker.parent().unwrap()).unwrap();
    fs::write(&marker, b"{ not a marker").unwrap();
    assert_eq!(discover(&extra).ok().text.as_deref(), Some("null"));

    // A real library elsewhere, reached through a link planted as the
    // standard folder.
    let elsewhere = existing_library(&fx.volume("T9"));
    assert_eq!(discover(&fx.volume("T9")).ok().json()["id"], ID);
    let stick = fx.volume("T7");
    std::os::unix::fs::symlink(&elsewhere, stick.join("TurboSpark")).unwrap();
    assert_eq!(discover(&stick).ok().text.as_deref(), Some("null"));

    // The marker file itself a link.
    let linked = fx.volume("Old");
    fs::create_dir_all(linked.join("TurboSpark")).unwrap();
    std::os::unix::fs::symlink(
        elsewhere.join("library.json"),
        linked.join("TurboSpark/library.json"),
    )
    .unwrap();
    assert_eq!(discover(&linked).ok().text.as_deref(), Some("null"));

    // Looking at a drive never adds it.
    assert_eq!(list().len(), 1);
}

// ----------------------------------------------------------------------- forget

#[test]
fn forgetting_an_install_drops_its_row_and_marker_entry_but_not_its_files() {
    let fx = Fixture::new("forget");
    let root = existing_library(&fx.volume("Extra"));
    add(&fx.volume("Extra"), None).ok();
    assert_eq!(fx.recorded().len(), 1);

    forget(&qwen_id()).ok();

    assert!(fx.recorded().is_empty(), "the row is gone");
    let marker: Value =
        serde_json::from_slice(&fs::read(root.join("library.json")).unwrap()).unwrap();
    assert_eq!(
        marker["models"],
        serde_json::json!([]),
        "so is its marker entry"
    );
    assert!(
        root.join("text/qwen.gturbo/weights.bin").is_file(),
        "forgetting never deletes a model"
    );
    assert_eq!(named(&list(), "Studio")["modelCount"], 0);

    // Forgetting what is not listed is not an error.
    forget(&qwen_id()).ok();
    forget("no-such-library:text/nothing").ok();
}

#[test]
fn forgetting_is_refused_for_a_library_that_cannot_be_written() {
    let fx = Fixture::new("forget-refused");
    let root = existing_library(&fx.volume("Extra"));
    add(&fx.volume("Extra"), None).ok();

    // A drive that is not connected.
    let away = fx.volume("Extra").join("Away");
    fs::rename(&root, &away).unwrap();
    assert_refused(&forget(&qwen_id()), "not_connected");
    assert_eq!(fx.recorded().len(), 1, "a refused forget keeps the row");
    fs::rename(&away, &root).unwrap();

    // A marker a newer TurboSpark wrote.
    let marker = root.join("library.json");
    let mut value: Value = serde_json::from_slice(&fs::read(&marker).unwrap()).unwrap();
    value["format"] = serde_json::json!(2);
    fs::write(&marker, value.to_string()).unwrap();
    assert_refused(&forget(&qwen_id()), "library_read_only");
    assert_eq!(fx.recorded().len(), 1);
}

// ----------------------------------------------------------------------- import

/// A store Move Store moved to `Old`: a text install, a catalog music install
/// without its receipt, an embedding install, an OCR "install" that is a link
/// (so its copy fails), a custom-location row, and the catalog override.
fn moved_store(fx: &Fixture) -> PathBuf {
    let root = fx.volume("Old");
    let models = root.join("models");
    let qwen = models.join("text/qwen.gturbo");
    let music = models.join(format!("audio/{MUSIC}.gturbo"));
    for install in [&qwen, &music] {
        fs::create_dir_all(install).unwrap();
        fs::write(install.join("weights.bin"), b"weights").unwrap();
    }
    let embedding = models.join("embeddings/gemma");
    fs::create_dir_all(&embedding).unwrap();
    fs::write(embedding.join("weights.bin"), b"embedding").unwrap();
    let outside = fx.base.join("outside");
    fs::create_dir_all(&outside).unwrap();
    fs::create_dir_all(models.join("ocr")).unwrap();
    std::os::unix::fs::symlink(&outside, models.join("ocr/linked")).unwrap();
    let custom = fx.base.join("custom/scratch");
    fs::create_dir_all(&custom).unwrap();
    let row = |alias: &str, path: &Path, modality: &str| {
        serde_json::json!({
            "alias": alias, "repo": format!("owner/{alias}"), "revision": "main",
            "path": path, "family": "qwen3", "install_bytes": 4096,
            "installed_on": "2026-10-14", "status": "verified", "modality": modality
        })
    };
    let rows = serde_json::json!([
        row("qwen", &qwen, "text"),
        row(MUSIC, &music, "audio"),
        row("scratch", &custom, "text"),
    ]);
    fs::write(root.join("installed.json"), rows.to_string()).unwrap();
    fs::write(root.join("models.json"), b"{}").unwrap();
    root
}

#[test]
fn importing_a_moved_store_reports_every_outcome() {
    let fx = Fixture::new("import");
    let root = moved_store(&fx);

    let imported = import(&root).ok().json();

    let library = &imported["library"];
    assert_eq!(library["name"], "Old", "named for the moved store's folder");
    assert_eq!(library["root"], root.join("models").to_str().unwrap());
    assert_eq!(library["available"], true);
    assert_eq!(library["isInternal"], false);
    assert_eq!(library["modelCount"], 2, "the text and the music install");
    assert_eq!(imported["skipped"], serde_json::json!([]));
    assert_eq!(imported["custom"], serde_json::json!(["scratch"]));
    assert_eq!(imported["needsAdoption"], serde_json::json!([MUSIC]));
    assert_eq!(
        imported["copiedToHome"],
        serde_json::json!(["embeddings/gemma"])
    );
    assert_eq!(imported["restoredCatalogOverride"], true);
    let failures = imported["copyFailures"].as_array().unwrap();
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert_eq!(failures[0]["key"], "ocr/linked");
    assert!(
        failures[0]["reason"].as_str().unwrap().contains("link"),
        "{failures:?}"
    );

    // Copied into the home, and the originals left alone.
    assert!(fx
        .home
        .join("models/embeddings/gemma/weights.bin")
        .is_file());
    assert!(fx.home.join("models.json").is_file());
    assert!(root.join("models/embeddings/gemma/weights.bin").is_file());
    assert!(root.join("models/text/qwen.gturbo/weights.bin").is_file());
    assert_eq!(list().len(), 2);

    // Again: the same library, and the failed copy is tried again.
    let again = import(&root).ok().json();
    assert_eq!(again["library"]["id"], library["id"]);
    assert_eq!(again["copiedToHome"], serde_json::json!([]));
    assert_eq!(again["restoredCatalogOverride"], false);
    assert_eq!(again["copyFailures"][0]["key"], "ocr/linked");
    assert_eq!(list().len(), 2, "still one library for the moved store");
}

#[test]
fn import_refusals_round_trip_their_codes() {
    let fx = Fixture::new("import-refusals");

    // A root that is not there, as on a drive that is not connected, and one
    // with no models folder.
    assert_refused(&import(&fx.base.join("Volumes/Gone")), "not_connected");
    assert_refused(&import(&fx.volume("Old")), "not_connected");

    // A models folder inside a library that is listed.
    add(&fx.volume("T7"), Some("Fast")).ok();
    let inner = fx.volume("T7").join("TurboSpark/old");
    fs::create_dir_all(inner.join("models")).unwrap();
    assert_refused(&import(&inner), "nested");

    // A marker that cannot be read, and an install record that cannot.
    let broken_marker = fx.volume("T9").join("moved");
    fs::create_dir_all(broken_marker.join("models")).unwrap();
    fs::write(broken_marker.join("models/library.json"), b"{ no").unwrap();
    assert_refused(&import(&broken_marker), "marker");
    let broken_record = fx.volume("Extra").join("moved");
    fs::create_dir_all(broken_record.join("models")).unwrap();
    fs::write(broken_record.join("installed.json"), b"not json").unwrap();
    assert_refused(&import(&broken_record), "io");

    // Every refusal left the list as it was.
    assert_eq!(list().len(), 2);
}

// ------------------------------------------------------------------------ busy

/// Holds an install walk open: its first stage line blocks until released.
struct Gate {
    started: Mutex<Sender<()>>,
    release: Mutex<Receiver<()>>,
    first: AtomicBool,
}

unsafe extern "C" fn hold(
    userdata: *mut c_void,
    kind: c_int,
    _text: *const c_char,
    _len: usize,
    _done: u64,
    _total: u64,
) {
    if kind != TS_INSTALL_STAGE {
        return;
    }
    let gate = &*(userdata as *const Gate);
    if gate.first.swap(false, Ordering::SeqCst) {
        gate.started.lock().unwrap().send(()).unwrap();
        let _ = gate
            .release
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(60));
    }
}

/// A real `ts_install` walk, parked in its stage callback while it is
/// registered as running. Cancelled before it is released, it stops at its
/// entry check without a network request.
struct HeldInstall {
    _gate: Arc<Gate>,
    release: Sender<()>,
    thread: Option<JoinHandle<c_int>>,
}

impl HeldInstall {
    fn start() -> Self {
        let catalog = reply(|out| unsafe { ts_catalog_json(out) }).ok().json();
        let alias = c(catalog[0]["alias"].as_str().unwrap());
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let gate = Arc::new(Gate {
            started: Mutex::new(started_tx),
            release: Mutex::new(release_rx),
            first: AtomicBool::new(true),
        });
        let address = Arc::as_ptr(&gate) as usize;
        let thread = thread::spawn(move || {
            let mut out: *mut c_char = ptr::null_mut();
            let code =
                unsafe { ts_install(alias.as_ptr(), Some(hold), address as *mut c_void, &mut out) };
            if !out.is_null() {
                unsafe { ts_string_free(out) };
            }
            code
        });
        let held = Self {
            _gate: gate,
            release: release_tx,
            thread: Some(thread),
        };
        started_rx
            .recv_timeout(Duration::from_secs(60))
            .expect("the install walk reached its first stage line");
        held
    }

    /// Cancels the walk, lets it go, and returns its status code.
    fn finish(mut self) -> c_int {
        assert_eq!(unsafe { ts_install_cancel() }, 1, "one walk was running");
        self.release.send(()).unwrap();
        self.thread.take().unwrap().join().unwrap()
    }
}

impl Drop for HeldInstall {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            unsafe { ts_install_cancel() };
            let _ = self.release.send(());
            let _ = thread.join();
        }
    }
}

#[test]
fn library_changes_are_refused_while_an_install_runs() {
    let fx = Fixture::new("busy");
    existing_library(&fx.volume("Extra"));
    add(&fx.volume("Extra"), None).ok();
    let spare = add(&fx.volume("T9"), Some("Spare")).ok().json();
    let spare = spare["library"]["id"].as_str().unwrap().to_string();
    let before = list();
    let recorded = fx.recorded();
    let moved = fx.volume("Old");
    fs::create_dir_all(moved.join("models")).unwrap();

    let held = HeldInstall::start();

    assert_busy(&add(&fx.volume("T7"), Some("Fast")));
    assert_busy(&remove(ID));
    assert_busy(&set_default(&spare));
    assert_busy(&forget(&qwen_id()));
    assert_busy(&import(&moved));
    // Nothing changed behind the refusals.
    assert!(!fx.volume("T7").join("TurboSpark").exists());
    assert!(!moved.join("models/library.json").exists());
    assert_eq!(list(), before);
    assert_eq!(fx.recorded(), recorded);
    // Reading is never refused.
    assert_eq!(list_reply().ok().json().as_array().unwrap().len(), 3);
    assert_eq!(
        discover(&fx.volume("Extra")).ok().json()["id"],
        ID,
        "discovery reads while an install runs"
    );

    assert_eq!(held.finish(), abi::TS_ERR_CANCELLED);

    // The same calls go through once nothing is installing.
    add(&fx.volume("T7"), Some("Fast")).ok();
    set_default(&spare).ok();
    forget(&qwen_id()).ok();
    remove(ID).ok();
    assert_eq!(list().len(), 3);
    assert!(fx.recorded().is_empty());
}

// --------------------------------------------------------------------- arguments

#[test]
fn bad_arguments_are_refused_before_anything_changes() {
    let fx = Fixture::new("arguments");
    let t7 = c(fx.volume("T7").to_str().unwrap());
    let root = fx.volume("Old");
    fs::create_dir_all(root.join("models")).unwrap();
    let old = c(root.to_str().unwrap());

    // A null out-pointer is found before the library is added, not after,
    // when the result could no longer reach the caller.
    let no_output = status(unsafe { ts_library_add(t7.as_ptr(), ptr::null(), ptr::null_mut()) });
    assert_eq!(
        no_output.code,
        abi::TS_ERR_INVALID_ARGUMENT,
        "{}",
        no_output.error
    );
    assert!(!fx.volume("T7").join("TurboSpark").exists());
    let no_output = status(unsafe { ts_store_import_moved_root(old.as_ptr(), ptr::null_mut()) });
    assert_eq!(
        no_output.code,
        abi::TS_ERR_INVALID_ARGUMENT,
        "{}",
        no_output.error
    );
    assert!(!root.join("models/library.json").exists());
    assert_eq!(
        status(unsafe { ts_libraries_json(ptr::null_mut()) }).code,
        abi::TS_ERR_INVALID_ARGUMENT
    );

    // Required strings.
    let missing = reply(|out| unsafe { ts_library_add(ptr::null(), ptr::null(), out) });
    assert_eq!(
        missing.code,
        abi::TS_ERR_INVALID_ARGUMENT,
        "{}",
        missing.error
    );
    assert_eq!(
        status(unsafe { ts_library_remove(ptr::null()) }).code,
        abi::TS_ERR_INVALID_ARGUMENT
    );
    assert_eq!(
        status(unsafe { ts_library_set_default(ptr::null()) }).code,
        abi::TS_ERR_INVALID_ARGUMENT
    );
    assert_eq!(
        status(unsafe { ts_model_forget(ptr::null()) }).code,
        abi::TS_ERR_INVALID_ARGUMENT
    );
    let missing = reply(|out| unsafe { ts_store_import_moved_root(ptr::null(), out) });
    assert_eq!(
        missing.code,
        abi::TS_ERR_INVALID_ARGUMENT,
        "{}",
        missing.error
    );
    assert_eq!(list().len(), 1);
}

// ------------------------------------------------------------------------ header

fn squash(text: &str) -> String {
    text.split_whitespace().collect()
}

/// The library exports in `src/api/libraries.rs` with their parameter types
/// as C spells them, whitespace removed.
fn exports(source: &str) -> Vec<(String, Vec<String>)> {
    const START: &str = "pub unsafe extern \"C\" fn ";
    source
        .match_indices(START)
        .map(|(at, _)| {
            let rest = &source[at + START.len()..];
            let name: String = rest
                .chars()
                .take_while(|ch| ch.is_alphanumeric() || *ch == '_')
                .collect();
            let open = rest.find('(').unwrap();
            let close = rest.find(") -> c_int").unwrap();
            let params = rest[open + 1..close]
                .split(',')
                .map(str::trim)
                .filter(|param| !param.is_empty())
                .map(|param| {
                    let rust = param.split_once(':').unwrap().1.trim();
                    let c_type = match rust {
                        "*const c_char" => "const char *",
                        "*mut *mut c_char" => "char **",
                        other => panic!("{name}: no C spelling known for {other}"),
                    };
                    squash(c_type)
                })
                .collect();
            (name, params)
        })
        .collect()
}

/// The parameter types the header declares for `int32_t <name>(...)`.
fn declared(header: &str, name: &str) -> Option<Vec<String>> {
    let from = header.find(&format!("int32_t {name}("))?;
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

#[test]
fn the_header_declares_exactly_the_library_exports() {
    let header = include_str!("../include/turbospark.h");
    let source =
        fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/api/libraries.rs"))
            .unwrap();
    let found = exports(&source);

    let mut names: Vec<&str> = found.iter().map(|(name, _)| name.as_str()).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        [
            "ts_libraries_json",
            "ts_library_add",
            "ts_library_discover_json",
            "ts_library_remove",
            "ts_library_set_default",
            "ts_model_forget",
            "ts_store_import_moved_root",
        ],
        "an export added or dropped here needs its header line and this list"
    );
    for (name, params) in &found {
        let in_header = declared(header, name)
            .unwrap_or_else(|| panic!("{name} is not declared in the header"));
        assert_eq!(
            &in_header, params,
            "{name}: the header and the export disagree"
        );
    }
}
