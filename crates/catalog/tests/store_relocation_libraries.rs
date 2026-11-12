//! `relocate_default_store` refuses while a model library is listed.
//!
//! A separate test binary on purpose: `set_default_root` is process-wide and
//! the relocation tests in `store.rs` set it too. Here every test holds one
//! lock for as long as it owns the override.

use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use turbospark_catalog::library::{parse_mount_table, LibraryRegistry, LocationEnv, MountSource};
use turbospark_catalog::{
    default_root, relocate_default_store, set_default_root, InstalledModel, ModelModality, Store,
};

static OVERRIDE: Mutex<()> = Mutex::new(());

/// A home with one text install, a fake drive `T7`, and the store root
/// override pointing at the home until the fixture drops.
struct Fixture {
    base: PathBuf,
    home: PathBuf,
    volume: PathBuf,
    env: LocationEnv,
    _lock: MutexGuard<'static, ()>,
}

impl Fixture {
    fn new(tag: &str) -> Self {
        let lock = OVERRIDE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Canonical, because the temp dir is behind the /var symlink and the
        // location rules compare canonical paths.
        let base = std::env::temp_dir().join(format!(
            "turbospark-relocation-libraries-{tag}-{}",
            std::process::id()
        ));
        std::fs::remove_dir_all(&base).ok();
        std::fs::create_dir_all(&base).unwrap();
        let base = base.canonicalize().unwrap();
        let home = base.join("home");
        let volumes_root = base.join("Volumes");
        let volume = volumes_root.join("T7");
        std::fs::create_dir_all(&volume).unwrap();

        let store = Store::new(&home);
        let model = store.install_path("qwen");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(model.join("weights.bin"), [1, 2, 3, 4]).unwrap();
        store
            .record(&InstalledModel {
                alias: "qwen".to_string(),
                repo: "owner/name".to_string(),
                revision: "main".to_string(),
                path: model,
                family: "qwen3".to_string(),
                install_bytes: 4,
                installed_on: "2026-10-18".to_string(),
                status: "runs".to_string(),
                kind: None,
                modality: ModelModality::Text,
                variant: None,
                library: None,
                relative_path: None,
            })
            .unwrap();

        // The fake drive shares the temp dir's device, so the home gets
        // another one or the same-disk refusal would fire.
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
        set_default_root(Some(home.clone())).unwrap();
        Self {
            base,
            home,
            volume,
            env,
            _lock: lock,
        }
    }

    fn add_library(&self) -> String {
        LibraryRegistry::load(&self.home)
            .add(&self.volume, Some("T7"), &self.env)
            .unwrap()
            .view
            .record
            .id
    }

    fn destination(&self) -> PathBuf {
        self.base.join("destination")
    }

    fn relocate(&self) -> Result<turbospark_catalog::StoreRelocation, String> {
        relocate_default_store(&self.destination(), |_, _| {})
    }

    /// Nothing moved: the install is where it was, its row is recorded, the
    /// destination was never created, and the store root is still the home.
    fn assert_untouched(&self) {
        let weights = self.home.join("models/text/qwen.gturbo/weights.bin");
        assert_eq!(std::fs::read(weights).unwrap(), [1, 2, 3, 4]);
        assert!(Store::new(&self.home).installed().contains_key("qwen"));
        assert!(!self.destination().exists(), "the destination was created");
        assert_eq!(default_root().as_deref(), Some(self.home.as_path()));
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        set_default_root(None).ok();
        std::fs::remove_dir_all(&self.base).ok();
    }
}

/// With a library listed the relocation refuses before it creates the
/// destination or touches the home; once the library is removed the same call
/// relocates, so the refusal is the library's and nothing else's.
#[test]
fn a_listed_library_refuses_the_relocation_until_it_is_removed() {
    let fx = Fixture::new("listed");
    let id = fx.add_library();

    let refused = fx.relocate().unwrap_err();
    assert!(
        refused.contains("while model libraries are listed"),
        "{refused}"
    );
    assert!(refused.contains("model libraries replace"), "{refused}");
    fx.assert_untouched();
    // The library is still listed, with its record in place.
    assert!(LibraryRegistry::load(&fx.home)
        .libraries()
        .iter()
        .any(|record| record.id == id));

    LibraryRegistry::load(&fx.home).remove(&id).unwrap();
    let relocation = fx.relocate().unwrap();
    assert_eq!(relocation.destination, fx.destination());
    assert!(fx
        .destination()
        .join("models/text/qwen.gturbo/weights.bin")
        .is_file());
    assert!(!fx.home.join("models").exists());
}

/// A list that cannot be read cannot vouch that no library exists.
#[test]
fn an_unreadable_library_list_refuses_the_relocation() {
    let fx = Fixture::new("unreadable");
    // A folder where the file belongs: `read` fails and it is not NotFound.
    std::fs::create_dir(fx.home.join("libraries.json")).unwrap();

    let refused = fx.relocate().unwrap_err();
    assert!(refused.contains("library list cannot be read"), "{refused}");
    fx.assert_untouched();
}

/// A list that does not parse is set aside by the load, and the relocation
/// still refuses that once, naming why.
#[test]
fn an_unparseable_library_list_refuses_the_relocation() {
    let fx = Fixture::new("unparseable");
    std::fs::write(fx.home.join("libraries.json"), b"not json").unwrap();

    let refused = fx.relocate().unwrap_err();
    assert!(refused.contains("library list cannot be read"), "{refused}");
    fx.assert_untouched();
    assert!(set_aside(&fx.home), "the list was not set aside");
}

fn set_aside(home: &Path) -> bool {
    std::fs::read_dir(home).unwrap().flatten().any(|entry| {
        entry
            .file_name()
            .to_string_lossy()
            .contains("libraries.json.corrupt-")
    })
}
