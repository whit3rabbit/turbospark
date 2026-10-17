//! Availability reported through the C ABI (4.2): the installed, image and
//! audio reports, driven through the same function bodies the header exposes,
//! against a temp home and fake volumes.
//!
//! Debug builds take the mount table, the volumes folder and the home's device
//! from the `TURBOSPARK_TEST_*` overrides, set here in process, so a temp
//! folder stands in for `/Volumes` and a fixture file for `/sbin/mount`.
//! Release builds ignore the overrides by design, so this file is debug-only.
//! The home is the process-local root `ts_store_root_set` installs. That root
//! and the environment are process-wide, so every test holds one lock for its
//! whole run.
#![cfg(debug_assertions)]

use std::ffi::{CStr, CString};
use std::fs;
use std::os::raw::{c_char, c_int};
use std::os::unix::fs::{symlink, MetadataExt};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::{Mutex, MutexGuard};

use catalog::audio_catalog::{AudioCatalog, AudioProfile, AudioTask};
use catalog::library::{LibraryMarker, PortableRow};
use catalog::ModelModality;
use serde_json::{json, Value};
use turbospark_ffi::{
    abi, ts_audio_installed_json, ts_audio_resolve, ts_image_installed_json, ts_installed_json,
    ts_last_error, ts_library_add, ts_store_root_set, ts_string_free,
};

/// A library id as a marker would carry it.
const ID: &str = "0f6c2a3e-1b7d-4c55-9e0a-7d2f1c3b4a59";
/// A second library.
const OTHER_ID: &str = "5a1d9c70-3e2b-4f68-8a14-6b0c2d9e7f31";

/// The store root and the environment are process-wide.
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
            "turbospark-ffi-reports-{tag}-{}",
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
        for volume in ["T7", "T9", "Extra"] {
            fs::create_dir_all(volumes.join(volume)).unwrap();
        }
        let at = |volume: &str| volumes.join(volume).display().to_string();
        let table = format!(
            "/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)\n\
             /dev/disk9s1 on {} (apfs, local, nodev, nosuid, journaled, noowners)\n\
             /dev/disk10s1 on {} (apfs, local, nodev, nosuid, journaled, noowners)\n\
             /dev/disk13s1 on {} (apfs, local, nodev, nosuid, journaled, noowners)\n",
            at("T7"),
            at("T9"),
            at("Extra"),
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

    /// The home's `installed.json`, replaced by `rows`.
    fn write_installed(&self, rows: Vec<Value>) {
        fs::write(
            self.home.join("installed.json"),
            Value::Array(rows).to_string(),
        )
        .unwrap();
    }

    /// The rows now in the home's `installed.json`.
    fn recorded(&self) -> Vec<Value> {
        serde_json::from_slice(&fs::read(self.home.join("installed.json")).unwrap()).unwrap()
    }

    /// Takes the library at `root` away, as unplugging its drive would.
    fn unplug(&self, root: &Path) -> PathBuf {
        let away = root.parent().unwrap().join("Away");
        fs::rename(root, &away).unwrap();
        away
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

fn rows_of(reply: Reply) -> Vec<Value> {
    reply.ok().json().as_array().unwrap().clone()
}

fn installed() -> Vec<Value> {
    rows_of(reply(|out| unsafe { ts_installed_json(out) }))
}

fn images() -> Vec<Value> {
    rows_of(reply(|out| unsafe { ts_image_installed_json(out) }))
}

fn audio_report() -> Value {
    reply(|out| unsafe { ts_audio_installed_json(out) })
        .ok()
        .json()
}

fn resolve(identity: &Value) -> Reply {
    let identity = c(&identity.to_string());
    reply(|out| unsafe { ts_audio_resolve(identity.as_ptr(), out) })
}

fn add(volume: &Path) {
    let path = c(volume.to_str().unwrap());
    reply(|out| unsafe { ts_library_add(path.as_ptr(), ptr::null(), out) }).ok();
}

fn row<'a>(rows: &'a [Value], alias: &str) -> &'a Value {
    rows.iter()
        .find(|row| row["alias"] == alias)
        .unwrap_or_else(|| panic!("no row {alias} in {rows:?}"))
}

fn aliases(rows: &[Value]) -> Vec<&str> {
    rows.iter()
        .map(|row| row["alias"].as_str().unwrap())
        .collect()
}

// ------------------------------------------------------------- text fixtures

/// A text row as `installed.json` holds it.
fn row_json(alias: &str, path: &Path) -> Value {
    json!({
        "alias": alias, "repo": format!("owner/{alias}"), "revision": "main",
        "path": path, "family": "llama", "install_bytes": 4096,
        "installed_on": "2026-10-14", "status": "verified"
    })
}

/// A text install a library on a drive can hold.
fn text_row(alias: &str) -> PortableRow {
    PortableRow {
        alias: alias.to_string(),
        repo: format!("owner/{alias}"),
        revision: "main".to_string(),
        relative_path: PathBuf::from(format!("text/{alias}.gturbo")),
        family: "llama".to_string(),
        install_bytes: 4_096,
        installed_on: "2026-10-14".to_string(),
        status: "verified".to_string(),
        kind: None,
        modality: ModelModality::Text,
        variant: None,
    }
}

/// A library already on `volume`: its marker lists `rows`. Returns its root.
fn library_on(volume: &Path, id: &str, name: &str, rows: Vec<PortableRow>) -> PathBuf {
    let root = volume.join("TurboSpark");
    fs::create_dir_all(&root).unwrap();
    let mut marker = LibraryMarker::new(id.to_string(), name.to_string());
    marker.models = rows;
    marker.write(&root).unwrap();
    root
}

/// A text install folder with one file in it.
fn text_folder(root: &Path, alias: &str) {
    let folder = root.join("text").join(format!("{alias}.gturbo"));
    fs::create_dir_all(&folder).unwrap();
    fs::write(folder.join("weights.bin"), b"weights").unwrap();
}

// ---------------------------------------------------------- the installed report

/// The report lists the rows an offline library holds, with its name and
/// "not connected", and a folder gone from a connected library as "missing".
/// This Mac's rows keep every key they always had.
#[test]
fn an_offline_library_row_stays_listed_and_a_missing_folder_is_marked() {
    let fx = Fixture::new("installed");
    let home_install = fx.home.join("models/text/homey.gturbo");
    fs::create_dir_all(&home_install).unwrap();
    fx.write_installed(vec![row_json("homey", &home_install)]);
    let root = library_on(
        &fx.volume("T7"),
        ID,
        "Studio",
        vec![text_row("qwen"), text_row("gone")],
    );
    text_folder(&root, "qwen");
    text_folder(&root, "gone");
    add(&fx.volume("T7"));
    fs::remove_dir_all(root.join("text/gone.gturbo")).unwrap();

    let rows = installed();

    assert_eq!(aliases(&rows), ["gone", "homey", "qwen"]);
    let homey = row(&rows, "homey");
    assert_eq!(homey["library"], "internal");
    assert_eq!(homey["library_name"], "This Mac");
    assert_eq!(homey["install_id"], "internal:text/homey");
    assert_eq!(homey["availability"], "available");
    assert_eq!(homey["location_kind"], "internal");
    assert!(homey["connect_name"].is_null(), "{homey}");
    // The keys an older reader depends on, unchanged.
    assert_eq!(homey["repo"], "owner/homey");
    assert_eq!(homey["revision"], "main");
    assert_eq!(homey["path"], home_install.to_str().unwrap());
    assert_eq!(homey["family"], "llama");
    assert_eq!(homey["install_bytes"], 4096);
    assert_eq!(homey["installed_on"], "2026-10-14");
    assert_eq!(homey["status"], "verified");
    assert_eq!(homey["modality"], "text");

    let qwen = row(&rows, "qwen");
    assert_eq!(qwen["library"], ID);
    assert_eq!(qwen["library_name"], "Studio");
    assert_eq!(qwen["install_id"], format!("{ID}:text/qwen"));
    assert_eq!(qwen["availability"], "available");
    assert_eq!(qwen["location_kind"], "library");
    assert_eq!(
        qwen["path"],
        root.join("text/qwen.gturbo").to_str().unwrap()
    );
    let gone = row(&rows, "gone");
    assert_eq!(gone["availability"], "missing", "{gone}");
    assert_eq!(gone["library_name"], "Studio");

    // The drive goes away: the rows stay, marked, with the library to
    // connect, and nothing is an error.
    fx.unplug(&root);
    let rows = installed();
    assert_eq!(aliases(&rows), ["gone", "homey", "qwen"]);
    for alias in ["qwen", "gone"] {
        let offline = row(&rows, alias);
        assert_eq!(offline["availability"], "not_connected", "{offline}");
        assert_eq!(offline["library_name"], "Studio");
        assert_eq!(offline["connect_name"], "Studio");
        assert_eq!(offline["library"], ID);
        assert_eq!(offline["location_kind"], "library");
    }
    assert_eq!(
        row(&rows, "qwen")["path"],
        root.join("text/qwen.gturbo").to_str().unwrap(),
        "an offline row keeps the last path recorded"
    );
    assert_eq!(row(&rows, "homey")["availability"], "available");
}

/// A row's path is where the install is now: a library that reappears under
/// another volume name reports its rows there.
#[test]
fn a_library_that_reappears_elsewhere_reports_where_its_rows_are_now() {
    let fx = Fixture::new("relocated");
    let root = library_on(&fx.volume("T7"), ID, "Studio", vec![text_row("qwen")]);
    text_folder(&root, "qwen");
    add(&fx.volume("T7"));
    let now = fx.volume("T9").join("TurboSpark");
    fs::rename(&root, &now).unwrap();

    let rows = installed();

    let qwen = row(&rows, "qwen");
    assert_eq!(qwen["availability"], "available", "{qwen}");
    assert_eq!(qwen["path"], now.join("text/qwen.gturbo").to_str().unwrap());
}

/// Rows outside a library say what they are, so the app localizes the place
/// instead of printing an English literal, and the report holds text only.
#[test]
fn rows_outside_a_library_report_a_location_kind() {
    let fx = Fixture::new("kinds");
    let custom = fx.base.join("elsewhere/odd.gturbo");
    fs::create_dir_all(&custom).unwrap();
    let mut ghost = row_json(
        "ghost",
        &fx.volume("Gone").join("TurboSpark/text/ghost.gturbo"),
    );
    ghost["library"] = json!("not-listed");
    ghost["relative_path"] = json!("text/ghost.gturbo");
    let mut image = row_json("pic", &fx.home.join("models/image/pic.gturbo"));
    image["modality"] = json!("image");
    let mut sound = row_json("sound", &fx.home.join("models/audio/sound.gturbo"));
    sound["modality"] = json!("audio");
    fx.write_installed(vec![
        row_json("odd", &custom),
        row_json("odd-offline", &fx.volume("Gone").join("odd.gturbo")),
        row_json("lost", &fx.home.join("models/text/lost.gturbo")),
        ghost,
        image,
        sound,
    ]);

    let rows = installed();

    assert_eq!(aliases(&rows), ["ghost", "lost", "odd", "odd-offline"]);
    let odd = row(&rows, "odd");
    assert_eq!(odd["location_kind"], "custom");
    assert!(odd["library"].is_null(), "no library holds it: {odd}");
    assert_eq!(odd["library_name"], "custom location");
    assert_eq!(odd["availability"], "available");
    assert_eq!(odd["install_id"], "internal:text/odd");
    // Under a volume that is not mounted, the volume is what to connect.
    let offline = row(&rows, "odd-offline");
    assert_eq!(offline["location_kind"], "custom");
    assert_eq!(offline["availability"], "not_connected");
    assert_eq!(offline["connect_name"], "Gone");
    let ghost = row(&rows, "ghost");
    assert_eq!(ghost["location_kind"], "unknown");
    assert_eq!(ghost["library"], "not-listed");
    assert_eq!(ghost["library_name"], "unknown library");
    assert_eq!(ghost["availability"], "missing");
    // This Mac's own row with no folder is listed, marked missing.
    let lost = row(&rows, "lost");
    assert_eq!(lost["location_kind"], "internal");
    assert_eq!(lost["availability"], "missing");
}

/// A library's name comes from a marker on a drive, so a library can be
/// called "This Mac" or "custom location". What a row is located by is its
/// ids and paths: such a library still reports `library`, and This Mac's own
/// row still reports `internal`.
#[test]
fn a_library_name_cannot_pass_for_a_location_kind() {
    let fx = Fixture::new("kinds-by-name");
    let own = fx.home.join("models/text/own.gturbo");
    fs::create_dir_all(&own).unwrap();
    fx.write_installed(vec![row_json("own", &own)]);
    let this_mac = library_on(&fx.volume("T7"), ID, "This Mac", vec![text_row("qq-a")]);
    text_folder(&this_mac, "qq-a");
    add(&fx.volume("T7"));
    let custom = library_on(
        &fx.volume("Extra"),
        OTHER_ID,
        "custom location",
        vec![text_row("qq-b")],
    );
    text_folder(&custom, "qq-b");
    add(&fx.volume("Extra"));

    let rows = installed();

    assert_eq!(aliases(&rows), ["own", "qq-a", "qq-b"]);
    let own = row(&rows, "own");
    assert_eq!(own["location_kind"], "internal");
    assert_eq!(own["library"], "internal");
    assert_eq!(own["library_name"], "This Mac");
    for (alias, id, name) in [
        ("qq-a", ID, "This Mac"),
        ("qq-b", OTHER_ID, "custom location"),
    ] {
        let found = row(&rows, alias);
        assert_eq!(found["location_kind"], "library", "{found}");
        assert_eq!(found["library"], id, "{found}");
        assert_eq!(found["library_name"], name, "{found}");
        assert_eq!(found["availability"], "available", "{found}");
        assert_eq!(found["install_id"], format!("{id}:text/{alias}"));
    }
}

// ------------------------------------------------------------ the image report

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

/// A valid image install at `<parent>/<alias>.gturbo`, built by the
/// production builder from a synthetic source kept outside `parent`.
fn build_image(fx: &Fixture, parent: &Path, alias: &str) -> PathBuf {
    build_image_of(fx, parent, alias, "Tongyi-MAI/Z-Image-Turbo")
}

/// [`build_image`] of the model `model_id`. An `andrevp/` model keeps its MLX
/// source tree in a folder beside the install.
fn build_image_of(fx: &Fixture, parent: &Path, alias: &str, model_id: &str) -> PathBuf {
    let source = fx.base.join(format!("source-{alias}"));
    write_synthetic_source(&source);
    fs::create_dir_all(parent).unwrap();
    let output = parent.join(format!("{alias}.gturbo"));
    image::build_image_install(&image::ImageInstallSpec {
        source_root: source.clone(),
        output_root: output.clone(),
        model_id: model_id.to_string(),
        model_revision: "synthetic-revision".to_string(),
    })
    .expect("build a synthetic image install");
    fs::remove_dir_all(source).unwrap();
    output.canonicalize().unwrap()
}

/// Image installs are listed from This Mac and from every library that is
/// connected, each tagged with its library, and a library that is not
/// connected is skipped without an error.
#[test]
fn image_installs_are_listed_from_every_connected_library() {
    let fx = Fixture::new("images");
    let home_image = build_image(&fx, &fx.home.join("models/image"), "home-pic");
    let root = library_on(&fx.volume("T7"), ID, "Studio", Vec::new());
    add(&fx.volume("T7"));
    let library_image = build_image(&fx, &root.join("image"), "drive-pic");
    let other = library_on(&fx.volume("Extra"), OTHER_ID, "Spare", Vec::new());
    add(&fx.volume("Extra"));
    let spare_image = build_image(&fx, &other.join("image"), "spare-pic");

    let rows = images();

    assert_eq!(aliases(&rows), ["drive-pic", "home-pic", "spare-pic"]);
    let home = row(&rows, "home-pic");
    assert_eq!(home["path"], home_image.to_str().unwrap());
    assert_eq!(home["library"], "internal");
    assert_eq!(home["libraryName"], "This Mac");
    assert_eq!(home["installId"], "internal:image/home-pic");
    // The row keeps the shape older readers know.
    assert_eq!(home["modelID"], "Tongyi-MAI/Z-Image-Turbo");
    assert_eq!(home["revision"], "synthetic-revision");
    assert_eq!(home["width"], 1024);
    assert!(home["sourcePath"].is_null());
    assert!(home["family"].is_string() && home["quantization"].is_string());
    let drive = row(&rows, "drive-pic");
    assert_eq!(drive["path"], library_image.to_str().unwrap());
    assert_eq!(drive["library"], ID);
    assert_eq!(drive["libraryName"], "Studio");
    assert_eq!(drive["installId"], format!("{ID}:image/drive-pic"));
    let spare = row(&rows, "spare-pic");
    assert_eq!(spare["path"], spare_image.to_str().unwrap());
    assert_eq!(spare["library"], OTHER_ID);

    // One drive is unplugged: its image is gone from the list, the others
    // stay, and the call still succeeds.
    fx.unplug(&root);
    let rows = images();
    assert_eq!(aliases(&rows), ["home-pic", "spare-pic"]);
}

/// A drive anyone could have written must not point the listing outside the
/// library: a linked `image` folder, a linked install, a linked manifest, a
/// linked library root and a linked source folder each add nothing, and a
/// plain source folder is reported.
#[test]
fn links_planted_on_a_drive_are_not_followed() {
    let fx = Fixture::new("image-links");
    // A valid install outside every library, which each link points at.
    let outside = build_image(&fx, &fx.base.join("outside"), "outsider");
    let root = library_on(&fx.volume("T7"), ID, "Studio", Vec::new());
    add(&fx.volume("T7"));
    let kept = build_image(&fx, &root.join("image"), "kept");
    symlink(&outside, root.join("image/linked.gturbo")).unwrap();
    // An install whose manifest is a link to a valid manifest elsewhere.
    let relinked = build_image(&fx, &root.join("image"), "linked-manifest");
    let copy = fx.base.join("manifest-copy.json");
    fs::copy(relinked.join("manifest.json"), &copy).unwrap();
    fs::remove_file(relinked.join("manifest.json")).unwrap();
    symlink(&copy, relinked.join("manifest.json")).unwrap();
    // A model that keeps its source tree: a plain folder is reported, and a
    // link in its place is not followed (the install itself still lists).
    let model = "andrevp/Z-Image-Turbo-MLX-4bit";
    let with_source = build_image_of(&fx, &root.join("image"), "with-source", model);
    let source = root.join("image/.with-source.synthetic-revision.image-source");
    fs::create_dir_all(&source).unwrap();
    build_image_of(&fx, &root.join("image"), "linked-source", model);
    symlink(
        fx.base.join("outside"),
        root.join("image/.linked-source.synthetic-revision.image-source"),
    )
    .unwrap();
    // A manifest whose revision, which the drive supplies, walks the kept
    // source folder's name out of the library: the folders it would reach
    // exist, and must not be examined or reported.
    let escaping = build_image_of(&fx, &root.join("image"), "escaping", model);
    let manifest_path = escaping.join("manifest.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest["source"]["model_revision"] = json!("x/../../../evil");
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    fs::create_dir_all(root.join("image/.escaping.x")).unwrap();
    fs::create_dir_all(fx.volume("T7").join("evil.image-source")).unwrap();
    // A library whose whole `image` folder is a link.
    let other = library_on(&fx.volume("Extra"), OTHER_ID, "Spare", Vec::new());
    add(&fx.volume("Extra"));
    symlink(fx.base.join("outside"), other.join("image")).unwrap();
    // A library whose root became a link to a folder off its drive.
    let third = library_on(
        &fx.volume("T9"),
        "7c1e4b52-9d03-4a86-b1f7-2e5a8c3d6f90",
        "Third",
        Vec::new(),
    );
    add(&fx.volume("T9"));
    build_image(&fx, &third.join("image"), "third-pic");
    let parked = fx.base.join("parked-library");
    fs::rename(&third, &parked).unwrap();
    symlink(&parked, &third).unwrap();

    let rows = images();

    assert_eq!(
        aliases(&rows),
        ["escaping", "kept", "linked-source", "with-source"],
        "{rows:?}"
    );
    assert!(
        row(&rows, "escaping")["sourcePath"].is_null(),
        "a revision cannot name a folder outside the library: {rows:?}"
    );
    assert_eq!(row(&rows, "kept")["path"], kept.to_str().unwrap());
    assert_eq!(
        row(&rows, "with-source")["path"],
        with_source.to_str().unwrap()
    );
    assert_eq!(
        row(&rows, "with-source")["sourcePath"],
        source.to_str().unwrap()
    );
    assert!(row(&rows, "linked-source")["sourcePath"].is_null());
}

// ------------------------------------------------------------- the audio report

/// The smallest audio profile a library row can be checked against: no terms
/// to accept, and a task whose record shape the report knows. Its files are
/// made sparse, so this costs no disk.
fn audio_profile() -> AudioProfile {
    AudioCatalog::embedded()
        .unwrap()
        .entries()
        .filter(|profile| {
            profile.acknowledgments.is_empty()
                && matches!(
                    profile.identity.task,
                    AudioTask::SpeechToText | AudioTask::TextToSpeech | AudioTask::Music
                )
        })
        .min_by_key(|profile| profile.assets.iter().map(|asset| asset.size).sum::<u64>())
        .cloned()
        .expect("a catalog profile without terms")
}

fn audio_row(profile: &AudioProfile) -> PortableRow {
    PortableRow {
        alias: profile.identity.alias.clone(),
        repo: profile.identity.repository.clone(),
        revision: profile.identity.revision.clone(),
        relative_path: PathBuf::from(format!("audio/{}.gturbo", profile.identity.alias)),
        family: profile.family.clone(),
        install_bytes: profile.assets.iter().map(|asset| asset.size).sum(),
        installed_on: "2026-10-14".to_string(),
        status: "verified".to_string(),
        kind: None,
        modality: ModelModality::Audio,
        variant: None,
    }
}

/// The install `profile` leaves in the library at `root`: every pinned file
/// at its pinned size, and the receipt the quick check reads. The bytes are
/// holes, because the quick check compares sizes and the receipt, and the
/// full hash is what `ts_audio_verify` is for.
fn audio_install(root: &Path, profile: &AudioProfile) {
    let folder = root
        .join("audio")
        .join(format!("{}.gturbo", profile.identity.alias));
    let mut files = serde_json::Map::new();
    for asset in &profile.assets {
        let path = folder.join(&asset.path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::File::create(&path)
            .unwrap()
            .set_len(asset.size)
            .unwrap();
        files.insert(
            asset.path.clone(),
            json!({"size": asset.size, "sha256": asset.sha256}),
        );
    }
    let receipt = json!({
        "schema_version": 1,
        "identity": profile.identity,
        "family": profile.family,
        "files": files,
    });
    fs::write(
        folder.join("audio-receipt.json"),
        serde_json::to_vec_pretty(&receipt).unwrap(),
    )
    .unwrap();
}

/// A library on `volume` holding `profile`'s audio install. Returns its root.
fn audio_library(volume: &Path, name: &str, profile: &AudioProfile) -> PathBuf {
    let root = library_on(volume, ID, name, vec![audio_row(profile)]);
    audio_install(&root, profile);
    add(volume);
    root
}

fn bucket<'a>(report: &'a Value, name: &str) -> &'a Vec<Value> {
    report[name]
        .as_array()
        .unwrap_or_else(|| panic!("no {name} bucket in {report}"))
}

/// A connected library's audio install is reported at the path it has now,
/// including after the drive came back under another volume, and resolves
/// there.
#[test]
fn a_connected_library_audio_install_reports_and_resolves_at_its_current_path() {
    let fx = Fixture::new("audio-connected");
    let profile = audio_profile();
    let alias = profile.identity.alias.clone();
    let root = audio_library(&fx.volume("T7"), "Studio", &profile);
    let identity = serde_json::to_value(&profile.identity).unwrap();
    let path = root.join("audio").join(format!("{alias}.gturbo"));

    let report = audio_report();

    assert!(bucket(&report, "not_connected").is_empty(), "{report}");
    assert!(bucket(&report, "incompatible").is_empty(), "{report}");
    let found = bucket(&report, "installed");
    assert_eq!(found.len(), 1, "{report}");
    assert_eq!(found[0]["alias"], alias.as_str());
    assert_eq!(found[0]["library"], ID);
    assert_eq!(found[0]["path"], path.to_str().unwrap());
    let resolved = resolve(&identity).ok();
    assert_eq!(resolved.text.as_deref(), Some(path.to_str().unwrap()));

    // The drive comes back under another name: the recorded path is stale and
    // the report and the resolve both use where the library is now.
    let now = fx.volume("T9").join("TurboSpark");
    fs::rename(&root, &now).unwrap();
    let moved = now.join("audio").join(format!("{alias}.gturbo"));
    let report = audio_report();
    let found = bucket(&report, "installed");
    assert_eq!(found.len(), 1, "{report}");
    assert_eq!(found[0]["path"], moved.to_str().unwrap());
    assert_eq!(
        resolve(&identity).ok().text.as_deref(),
        Some(moved.to_str().unwrap())
    );
}

/// An unplugged library's audio install is in the `not_connected` bucket with
/// the library to connect, is not an error, and resolving it names the
/// library behind a stable code.
#[test]
fn an_offline_library_audio_install_is_not_connected_rather_than_an_error() {
    let fx = Fixture::new("audio-offline");
    let profile = audio_profile();
    let root = audio_library(&fx.volume("T7"), "Studio", &profile);
    let identity = serde_json::to_value(&profile.identity).unwrap();
    fx.unplug(&root);

    let report = audio_report();

    let offline = bucket(&report, "not_connected");
    assert_eq!(offline.len(), 1, "{report}");
    assert_eq!(offline[0]["library_name"], "Studio");
    assert_eq!(
        offline[0]["record"]["alias"],
        profile.identity.alias.as_str()
    );
    assert_eq!(offline[0]["record"]["library"], ID);
    assert!(bucket(&report, "incompatible").is_empty(), "{report}");
    assert!(bucket(&report, "installed").is_empty(), "{report}");

    let refused = resolve(&identity);
    assert_eq!(
        refused.code,
        abi::TS_ERR_INVALID_ARGUMENT,
        "{}",
        refused.error
    );
    assert!(refused.text.is_none(), "a refusal writes no output");
    assert!(
        refused.error.starts_with("not_connected: "),
        "{}",
        refused.error
    );
    assert!(refused.error.contains("Studio"), "{}", refused.error);
}

/// A row an older writer stripped of its library is given the library back
/// before the audio report reads it, so it is reported as installed rather
/// than as incompatible, which is what a This Mac row at a foreign path is.
#[test]
fn an_audio_row_an_older_writer_stripped_is_restored_before_the_report() {
    let fx = Fixture::new("audio-stripped");
    let profile = audio_profile();
    let alias = profile.identity.alias.clone();
    let root = audio_library(&fx.volume("T7"), "Studio", &profile);
    // An older writer rewrote the registry and dropped what it did not know.
    let stripped: Vec<Value> = fx
        .recorded()
        .into_iter()
        .map(|mut row| {
            let object = row.as_object_mut().unwrap();
            object.remove("library");
            object.remove("relative_path");
            row
        })
        .collect();
    assert_eq!(stripped.len(), 1);
    fx.write_installed(stripped);

    let report = audio_report();

    assert!(bucket(&report, "incompatible").is_empty(), "{report}");
    let found = bucket(&report, "installed");
    assert_eq!(found.len(), 1, "{report}");
    assert_eq!(found[0]["alias"], alias.as_str());
    assert_eq!(found[0]["library"], ID);
    assert_eq!(
        found[0]["path"],
        root.join("audio")
            .join(format!("{alias}.gturbo"))
            .to_str()
            .unwrap()
    );
}

/// Names and aliases from a drive stay raw in the JSON, and the sentence of an
/// error is escaped and capped behind an intact code.
#[test]
fn a_hostile_library_name_is_raw_in_json_and_escaped_in_an_error() {
    let fx = Fixture::new("hostile");
    let profile = audio_profile();
    let hostile = format!("evil\u{1b}[31m\u{202e}{}", "x".repeat(2000));
    let root = library_on(
        &fx.volume("T7"),
        ID,
        &hostile,
        vec![text_row("qwen"), audio_row(&profile)],
    );
    text_folder(&root, "qwen");
    audio_install(&root, &profile);
    add(&fx.volume("T7"));
    fx.unplug(&root);

    let rows = installed();
    let qwen = row(&rows, "qwen");
    assert_eq!(
        qwen["library_name"],
        hostile.as_str(),
        "names are identities"
    );
    assert_eq!(qwen["connect_name"], hostile.as_str());

    let refused = resolve(&serde_json::to_value(&profile.identity).unwrap());
    let error = &refused.error;
    assert!(error.starts_with("not_connected: "), "{error}");
    assert!(error.contains("evil"), "the library is named: {error}");
    assert_plain(error, "not_connected: ");
}

/// A message behind `prefix` holds no raw control or direction character and
/// no more than 300 characters of sentence.
fn assert_plain(error: &str, prefix: &str) {
    assert!(
        !error
            .chars()
            .any(|ch| ch.is_control()
                || matches!(ch, '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')),
        "a raw control or direction character reached the message: {error:?}"
    );
    assert!(
        error.chars().count() <= prefix.len() + 300,
        "the sentence is capped, got {} characters",
        error.chars().count()
    );
}

/// A library whose root leads off its drive is a fault, not an unplugged
/// drive, so resolving is refused without the not-connected code. The
/// sentence still names a drive-supplied library, so it is escaped and capped
/// like every other.
#[test]
fn a_library_linked_off_its_drive_refuses_resolve_as_a_plain_capped_sentence() {
    let fx = Fixture::new("audio-linked-root");
    let profile = audio_profile();
    // The name also says what an unplugged drive's message says, so a
    // verdict reached by matching the sentence would call this unplugged.
    let hostile = format!(
        "evil\u{1b}[31m\u{202e} is not connected; connect its drive {}",
        "x".repeat(2000)
    );
    let root = audio_library(&fx.volume("T7"), &hostile, &profile);
    let parked = fx.base.join("parked-library");
    fs::rename(&root, &parked).unwrap();
    symlink(&parked, &root).unwrap();

    let refused = resolve(&serde_json::to_value(&profile.identity).unwrap());

    assert_eq!(
        refused.code,
        abi::TS_ERR_INVALID_ARGUMENT,
        "{}",
        refused.error
    );
    assert!(refused.text.is_none(), "a refusal writes no output");
    let error = &refused.error;
    assert!(
        !error.is_empty() && !error.starts_with("not_connected"),
        "{error}"
    );
    assert!(error.contains("evil"), "the library is named: {error}");
    assert_plain(error, "");
    let report = audio_report();
    assert!(bucket(&report, "not_connected").is_empty(), "{report}");
    assert!(bucket(&report, "installed").is_empty(), "{report}");
}
