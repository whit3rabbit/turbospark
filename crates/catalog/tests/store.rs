//! Store layout, the install record, and alias resolution.
//!
//! The resolution ORDER is the load-bearing assertion in this file. Every
//! other test here is bookkeeping; that one is the difference between
//! `--model foo` running the model at `./foo` and running some other model
//! that happens to be recorded under that alias.

use std::path::PathBuf;

use turbospark_catalog::{InstalledModel, Store};

fn temp_root(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-store-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn row(alias: &str, path: &std::path::Path) -> InstalledModel {
    InstalledModel {
        alias: alias.to_string(),
        repo: "owner/name".to_string(),
        revision: "main".to_string(),
        path: path.to_path_buf(),
        family: "llama".to_string(),
        install_bytes: 123,
        installed_on: "2026-08-15".to_string(),
        status: "runs".to_string(),
        kind: None,
    }
}

#[test]
fn an_install_path_is_the_alias_under_models() {
    let root = temp_root("path");
    let store = Store::new(&root);
    assert_eq!(
        store.install_path("gemma4"),
        root.join("models").join("gemma4.gturbo")
    );
}

#[test]
fn the_record_round_trips_and_drops_rows_whose_directory_is_gone() {
    let root = temp_root("record");
    let store = Store::new(&root);
    let present = root.join("models").join("here.gturbo");
    std::fs::create_dir_all(&present).unwrap();
    let absent = root.join("models").join("gone.gturbo");

    store.record(&row("here", &present)).unwrap();
    store.record(&row("gone", &absent)).unwrap();

    let installed = store.installed();
    assert!(installed.contains_key("here"));
    // A directory deleted by hand leaves a stale row rather than a corrupt
    // one, and `installed` is where that is reconciled.
    assert!(
        !installed.contains_key("gone"),
        "a row whose directory is gone must not be listed as installed"
    );

    store.forget("here").unwrap();
    assert!(store.installed().is_empty());
}

/// A corrupt record must not stop a pull. The record is a convenience index
/// over directories that are self-describing; treating it as authoritative
/// would make one bad write cost every install.
#[test]
fn a_corrupt_record_reads_as_empty_rather_than_failing() {
    let root = temp_root("corrupt");
    std::fs::write(root.join("installed.json"), "{not json at all").unwrap();
    assert!(Store::new(&root).installed().is_empty());
}

/// **The order that matters.** An existing directory beats a recorded alias
/// of the same name, because a path that silently resolved to an alias would
/// run a DIFFERENT model than the one named on the command line, fluently and
/// with no error.
#[test]
fn an_existing_directory_beats_an_alias_of_the_same_name() {
    let root = temp_root("resolve");
    let store = Store::new(&root);

    let recorded = root.join("models").join("collide.gturbo");
    std::fs::create_dir_all(&recorded).unwrap();
    store.record(&row("collide", &recorded)).unwrap();
    assert_eq!(store.resolve("collide").unwrap(), recorded);

    // Now make a DIRECTORY with that literal name and re-resolve from inside
    // its parent, so the string is both a valid relative path and a valid
    // alias. The path must win.
    let scratch = temp_root("resolve-cwd");
    let shadow = scratch.join("collide");
    std::fs::create_dir_all(&shadow).unwrap();
    assert_eq!(
        store.resolve(shadow.to_str().unwrap()).unwrap(),
        shadow,
        "an existing directory must win over a recorded alias"
    );
}

#[test]
fn an_install_in_the_default_location_resolves_without_a_record() {
    let root = temp_root("implicit");
    let store = Store::new(&root);
    let path = store.install_path("byhand");
    std::fs::create_dir_all(&path).unwrap();
    assert_eq!(
        store.resolve("byhand").unwrap(),
        path,
        "a store populated by hand should still resolve"
    );
    assert!(store.resolve("never-installed").is_none());
}

/// `resolve_model_arg` backs `turbospark-check --model`, so its one hard
/// requirement is that it never fails a run that used to work: an
/// unresolvable name comes back unchanged and the caller reports the same
/// "no such install" it always did.
#[test]
fn resolving_a_model_argument_passes_an_unknown_name_through_unchanged() {
    let unknown = "/definitely/not/an/install/anywhere";
    assert_eq!(
        turbospark_catalog::resolve_model_arg(unknown),
        PathBuf::from(unknown)
    );
}

#[test]
fn directory_bytes_sums_a_tree() {
    let root = temp_root("bytes");
    std::fs::create_dir_all(root.join("packed_experts")).unwrap();
    std::fs::write(root.join("a.bin"), vec![0u8; 100]).unwrap();
    std::fs::write(root.join("packed_experts").join("b.bin"), vec![0u8; 250]).unwrap();
    assert_eq!(turbospark_catalog::directory_bytes(&root), 350);
}

#[test]
fn hf_token_storage_and_lifecycle() {
    let root = temp_root("hf-token");
    let store = Store::new(&root);

    // Initial state: none
    assert!(store.get_hf_token().is_none());

    // Save token
    store.set_hf_token("hf_test_secret_token_12345").unwrap();
    assert_eq!(
        store.get_hf_token().as_deref(),
        Some("hf_test_secret_token_12345")
    );

    // File permissions on Unix should be 0600
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::metadata(store.hf_token_path())
            .unwrap()
            .permissions();
        assert_eq!(perms.mode() & 0o777, 0o600);
    }

    // Overwrite with trimmed whitespace
    store.set_hf_token("  hf_updated_token  ").unwrap();
    assert_eq!(store.get_hf_token().as_deref(), Some("hf_updated_token"));

    // Setting empty clears it
    store.set_hf_token("   ").unwrap();
    assert!(store.get_hf_token().is_none());

    // Clear explicitly
    store.set_hf_token("hf_token_to_clear").unwrap();
    assert!(store.get_hf_token().is_some());
    store.clear_hf_token().unwrap();
    assert!(store.get_hf_token().is_none());
}

#[test]
fn hf_token_resolution_order() {
    use turbospark_catalog::{resolve_hf_token_with_source, HfTokenSource};

    // Explicit argument wins over anything
    let (tok, src) = resolve_hf_token_with_source(Some("hf_explicit_arg")).unwrap();
    assert_eq!(tok, "hf_explicit_arg");
    assert_eq!(src, HfTokenSource::Explicit);
}
