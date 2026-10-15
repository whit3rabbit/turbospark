//! Store layout, the install record, and alias resolution.
//!
//! The resolution ORDER is the load-bearing assertion in this file. Every
//! other test here is bookkeeping; that one is the difference between
//! `--model foo` running the model at `./foo` and running some other model
//! that happens to be recorded under that alias.

use std::path::PathBuf;

use turbospark_catalog::{set_default_root, InstalledModel, ModelModality, Store};

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
        modality: ModelModality::Text,
        variant: None,
        library: None,
        relative_path: None,
    }
}

/// `installed.json` exactly as the writer before model libraries produced it
/// (`serde_json::to_string_pretty` of the rows). It holds a variant text row,
/// a vision tower, a legacy image row an older CLI rewrote with the default
/// `"modality": "text"`, an audio row, and a row left behind on an unmounted
/// Move Store volume. Every one must keep parsing as an internal install and
/// write back byte for byte, or older readers and diffs of the file change.
const PRE_LIBRARY_INSTALLED_JSON: &str = r#"[
  {
    "alias": "qwen3-4b",
    "repo": "Qwen/Qwen3-4B",
    "revision": "main",
    "path": "/Users/example/.turbospark/models/text/qwen3-4b.gturbo",
    "family": "qwen3",
    "install_bytes": 4400000000,
    "installed_on": "2026-08-15",
    "status": "runs",
    "kind": null,
    "modality": "text",
    "variant": "Q8_0"
  },
  {
    "alias": "qwen3-vl-tower",
    "repo": "Qwen/Qwen3-VL-4B-Instruct",
    "revision": "0123456789abcdef0123456789abcdef01234567",
    "path": "/Users/example/.turbospark/models/text/qwen3-vl-tower.gturbo",
    "family": "qwen3_vl",
    "install_bytes": 800000000,
    "installed_on": "2026-08-16",
    "status": "runs",
    "kind": "vision-tower",
    "modality": "text"
  },
  {
    "alias": "legacy-image",
    "repo": "owner/image",
    "revision": "main",
    "path": "/Users/example/.turbospark/models/legacy-image.image.gturbo",
    "family": "z_image",
    "install_bytes": 6000000000,
    "installed_on": "2026-05-01",
    "status": "runs",
    "kind": "image",
    "modality": "text"
  },
  {
    "alias": "whisper-small",
    "repo": "openai/whisper-small",
    "revision": "main",
    "path": "/Users/example/.turbospark/models/audio/whisper-small.gturbo",
    "family": "whisper",
    "install_bytes": 480000000,
    "installed_on": "2026-09-02",
    "status": "unverified",
    "kind": "speech",
    "modality": "audio"
  },
  {
    "alias": "away",
    "repo": "owner/name",
    "revision": "main",
    "path": "/Volumes/Models/turbospark/models/text/away.gturbo",
    "family": "llama",
    "install_bytes": 123,
    "installed_on": "2026-07-04",
    "status": "runs",
    "kind": null,
    "modality": "text"
  }
]"#;

#[test]
fn a_pre_library_registry_parses_unchanged_as_internal_rows() {
    let rows: Vec<InstalledModel> = serde_json::from_str(PRE_LIBRARY_INSTALLED_JSON).unwrap();
    assert_eq!(rows.len(), 5);
    for row in &rows {
        assert_eq!(row.library, None, "{} must read as internal", row.alias);
        assert_eq!(row.relative_path, None, "{}", row.alias);
    }
    let ids: Vec<String> = rows.iter().map(InstalledModel::install_id).collect();
    assert_eq!(
        ids,
        [
            "internal:text/qwen3-4b",
            "internal:text/qwen3-vl-tower",
            "internal:image/legacy-image",
            "internal:audio/whisper-small",
            "internal:text/away",
        ]
    );
    assert_eq!(
        serde_json::to_string_pretty(&rows).unwrap(),
        PRE_LIBRARY_INSTALLED_JSON,
        "an internal row must write back exactly as the pre-library writer did"
    );
}

/// The id is `<library>:<modality>/<alias>` and is the key app metadata
/// follows across locations (8.1), so its spelling is pinned here. A
/// legacy image row with no `modality` key must still resolve to `image`,
/// the same modality every reader already gives it.
#[test]
fn install_id_is_library_modality_and_alias() {
    let path = std::path::Path::new("/models/x.gturbo");
    let text = row("gemma4", path);
    assert_eq!(text.install_id(), "internal:text/gemma4");
    assert!(text
        .install_id()
        .starts_with(turbospark_catalog::library::INTERNAL_LIBRARY_ID));

    let legacy_image: InstalledModel = serde_json::from_value(serde_json::json!({
        "alias": "old-image",
        "repo": "owner/image",
        "revision": "main",
        "path": "/models/old-image.image.gturbo",
        "family": "z_image",
        "install_bytes": 1,
        "installed_on": "2026-01-01",
        "status": "runs",
        "kind": "image"
    }))
    .unwrap();
    assert_eq!(legacy_image.modality, ModelModality::Text);
    assert_eq!(legacy_image.install_id(), "internal:image/old-image");

    let mut external = row("whisper", path);
    external.modality = ModelModality::Audio;
    external.library = Some("5f0c2a9e-7d41-4c3b-9a55-0e6f1b2c3d4e".to_string());
    external.relative_path = Some(PathBuf::from("audio/whisper.gturbo"));
    assert_eq!(
        external.install_id(),
        "5f0c2a9e-7d41-4c3b-9a55-0e6f1b2c3d4e:audio/whisper"
    );

    // One alias may exist once per modality, so the ids must differ.
    let mut same_alias_image = row("gemma4", path);
    same_alias_image.modality = ModelModality::Image;
    assert_ne!(same_alias_image.install_id(), text.install_id());
}

/// The new fields go through the real writer: an external row keeps them,
/// an internal row is written without either key.
#[test]
fn library_fields_round_trip_and_internal_rows_omit_them() {
    let root = temp_root("library-fields");
    let store = Store::new(&root);
    let internal = row("here", &store.install_path("here"));
    let mut external = row(
        "there",
        std::path::Path::new("/Volumes/Ext/m/text/there.gturbo"),
    );
    external.library = Some("5f0c2a9e-7d41-4c3b-9a55-0e6f1b2c3d4e".to_string());
    external.relative_path = Some(PathBuf::from("text/there.gturbo"));
    store.record(&internal).unwrap();
    store.record(&external).unwrap();

    let text = std::fs::read_to_string(root.join("installed.json")).unwrap();
    let raw: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
    let internal_raw = raw.iter().find(|r| r["alias"] == "here").unwrap();
    assert!(internal_raw.get("library").is_none(), "{internal_raw}");
    assert!(
        internal_raw.get("relative_path").is_none(),
        "{internal_raw}"
    );
    let external_raw = raw.iter().find(|r| r["alias"] == "there").unwrap();
    assert_eq!(
        external_raw["library"],
        "5f0c2a9e-7d41-4c3b-9a55-0e6f1b2c3d4e"
    );
    assert_eq!(external_raw["relative_path"], "text/there.gturbo");

    let decoded: Vec<InstalledModel> = serde_json::from_str(&text).unwrap();
    assert_eq!(decoded, vec![internal, external]);
}

/// Library fields do not change the text listing: it still lists every text
/// row whose directory exists, keeps its recorded path, and hides the rest.
/// Availability-aware listing is a separate call.
#[test]
fn library_fields_do_not_change_the_text_listing() {
    let root = temp_root("library-listing");
    let store = Store::new(&root);
    let mounted = root.join("Ext").join("m").join("text").join("there.gturbo");
    std::fs::create_dir_all(&mounted).unwrap();
    let mut external = row("there", &mounted);
    external.library = Some("5f0c2a9e-7d41-4c3b-9a55-0e6f1b2c3d4e".to_string());
    external.relative_path = Some(PathBuf::from("text/there.gturbo"));
    let mut offline = row("away", &root.join("Gone").join("text").join("away.gturbo"));
    offline.library = Some("0a1b2c3d-0000-4000-8000-000000000000".to_string());
    offline.relative_path = Some(PathBuf::from("text/away.gturbo"));
    store.record(&external).unwrap();
    store.record(&offline).unwrap();

    let installed = store.installed();
    assert_eq!(installed.keys().collect::<Vec<_>>(), ["there"]);
    assert_eq!(installed["there"], external);
}

#[test]
fn an_install_path_is_the_alias_under_models() {
    let root = temp_root("path");
    let store = Store::new(&root);
    assert_eq!(
        store.install_path("gemma4"),
        root.join("models").join("text").join("gemma4.gturbo")
    );
}

#[test]
fn variant_label_is_optional_and_legacy_records_keep_their_json_shape() {
    let path = std::path::Path::new("/models/model.gturbo");
    let legacy_json = serde_json::to_value(row("model", path)).unwrap();
    let mut legacy_object = legacy_json.as_object().unwrap().clone();
    legacy_object.remove("variant");
    let legacy: InstalledModel = serde_json::from_value(legacy_object.into()).unwrap();
    assert_eq!(legacy.variant, None);
    assert!(!serde_json::to_value(&legacy)
        .unwrap()
        .as_object()
        .unwrap()
        .contains_key("variant"));

    let mut variant = row("model-q8", path);
    variant.variant = Some("Q8_0".to_string());
    let encoded = serde_json::to_vec(&variant).unwrap();
    let decoded: InstalledModel = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded.variant.as_deref(), Some("Q8_0"));
}

#[test]
fn an_image_install_has_its_own_namespace_and_resolves_without_a_record() {
    let root = temp_root("image-path");
    let store = Store::new(&root);
    let path = store.image_install_path("z-image");
    assert_eq!(
        path,
        root.join("models").join("image").join("z-image.gturbo")
    );
    std::fs::create_dir_all(&path).unwrap();
    assert_eq!(store.resolve_image("z-image").unwrap(), path);
}

#[test]
fn modality_paths_share_a_root_but_do_not_share_aliases() {
    let root = temp_root("modalities");
    let store = Store::new(&root);
    assert_eq!(
        store.audio_install_path("whisper"),
        root.join("models").join("audio").join("whisper.gturbo")
    );

    let text = store.install_path("same-name");
    let image = store.image_install_path("same-name");
    std::fs::create_dir_all(&text).unwrap();
    std::fs::create_dir_all(&image).unwrap();
    assert_eq!(store.resolve("same-name"), Some(text));
    assert_eq!(store.resolve_image("same-name"), Some(image));
}

#[test]
fn relocating_the_default_store_verifies_and_rewrites_install_paths() {
    let root = temp_root("relocation");
    let source = root.join("source");
    let destination = root.join("destination");
    let store = Store::new(&source);
    let text_model = store.install_path("text-model");
    let image_model = store.image_install_path("image-model");
    let audio_model = store.audio_install_path("audio-model");

    std::fs::create_dir_all(&text_model).unwrap();
    std::fs::create_dir_all(&image_model).unwrap();
    std::fs::create_dir_all(&audio_model).unwrap();
    std::fs::write(text_model.join("weights.bin"), [1, 2, 3, 4]).unwrap();
    std::fs::write(image_model.join("weights.bin"), [5, 6]).unwrap();
    std::fs::write(audio_model.join("weights.bin"), [7, 8, 9]).unwrap();
    store.record(&row("text-model", &text_model)).unwrap();
    std::fs::write(source.join("hf_token"), "hf_test").unwrap();
    let cached_hub_response = b"validated hub cache fixture";
    let hub_cache_file = source.join("hub-cache").join("query.json");
    std::fs::create_dir_all(hub_cache_file.parent().unwrap()).unwrap();
    std::fs::write(&hub_cache_file, cached_hub_response).unwrap();
    let expected_source = source.canonicalize().unwrap();

    set_default_root(Some(source.clone())).unwrap();
    let outcome = turbospark_catalog::relocate_default_store(&destination, |done, total| {
        assert!(done <= total);
    });
    set_default_root(None).unwrap();
    let relocation = outcome.unwrap();

    assert_eq!(relocation.source, expected_source);
    assert_eq!(relocation.destination, destination);
    assert!(relocation.bytes >= 9);
    assert!(relocation.files >= 3);
    assert!(
        !source.join("models").exists(),
        "the managed model tree should be removed from the source"
    );
    assert!(
        source.join("hub-cache/query.json").is_file(),
        "private hub cache entries should remain at the original store"
    );
    assert_eq!(
        std::fs::read(source.join("hub-cache/query.json")).unwrap(),
        cached_hub_response
    );
    assert!(
        !destination.join("hub-cache").exists(),
        "model-store relocation must not export private hub metadata"
    );
    assert_eq!(
        std::fs::read_to_string(source.join("hf_token")).unwrap(),
        "hf_test"
    );
    assert!(
        !destination.join("hf_token").exists(),
        "model-store relocation must not export credentials"
    );
    assert!(destination
        .join("models/image/image-model.gturbo/weights.bin")
        .is_file());
    assert!(destination
        .join("models/audio/audio-model.gturbo/weights.bin")
        .is_file());

    let installed = Store::new(&destination).installed();
    assert_eq!(
        installed["text-model"].path,
        destination.join("models/text/text-model.gturbo")
    );
}

#[cfg(unix)]
#[test]
fn relocation_does_not_follow_a_destination_symlink_added_during_copy() {
    use std::os::unix::fs::symlink;

    let root = temp_root("relocation-symlink-race");
    let source = root.join("source");
    let destination = root.join("destination");
    let victim = root.join("victim");
    let model = Store::new(&source).install_path("model");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("weights.bin"), [1, 2, 3, 4]).unwrap();
    std::fs::write(source.join("models.json"), "{}").unwrap();
    std::fs::write(&victim, "keep me").unwrap();

    let mut planted = false;
    set_default_root(Some(source.clone())).unwrap();
    let outcome = turbospark_catalog::relocate_default_store(&destination, |_, _| {
        if !planted {
            symlink(&victim, destination.join("models.json")).unwrap();
            planted = true;
        }
    });
    set_default_root(None).unwrap();

    let error = outcome.unwrap_err();
    assert!(error.contains("without replacing an existing entry"));
    assert_eq!(std::fs::read_to_string(&victim).unwrap(), "keep me");
    assert!(source.join("models.json").is_file());
}

#[test]
fn legacy_flat_paths_remain_readable_by_their_modality() {
    let root = temp_root("legacy-paths");
    let store = Store::new(&root);
    let text = root.join("models").join("legacy.gturbo");
    let image = root.join("models").join("legacy-image.image.gturbo");
    std::fs::create_dir_all(&text).unwrap();
    std::fs::create_dir_all(&image).unwrap();
    assert_eq!(store.resolve("legacy"), Some(text));
    assert_eq!(store.resolve_image("legacy-image"), Some(image));
}

#[test]
fn legacy_image_registry_rows_do_not_enter_the_text_index() {
    let root = temp_root("legacy-image-row");
    let store = Store::new(&root);
    let path = root.join("models").join("legacy-image.image.gturbo");
    std::fs::create_dir_all(&path).unwrap();

    let mut image_row = row("legacy-image", &path);
    image_row.kind = Some("image".to_string());
    std::fs::write(
        root.join("installed.json"),
        serde_json::to_vec(&vec![image_row]).unwrap(),
    )
    .unwrap();

    assert!(!store.installed().contains_key("legacy-image"));
    assert!(store.resolve("legacy-image").is_none());
    assert_eq!(store.resolve_image("legacy-image"), Some(path));
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

/// A row on an unmounted external volume looks exactly like a row whose
/// directory was deleted. Recording any other install must not prune it, or
/// the install is forgotten for good before the volume comes back.
#[test]
fn recording_an_install_keeps_rows_whose_volume_is_offline() {
    let root = temp_root("offline-volume");
    let store = Store::new(&root);
    let offline = root.join("unmounted-volume").join("away.gturbo");
    let mut offline_audio = row(
        "away-audio",
        &root.join("unmounted-volume").join("a.gturbo"),
    );
    offline_audio.modality = ModelModality::Audio;
    store.record(&row("away", &offline)).unwrap();
    store.record(&offline_audio).unwrap();

    let present = store.install_path("here");
    std::fs::create_dir_all(&present).unwrap();
    store.record(&row("here", &present)).unwrap();

    let on_disk: Vec<InstalledModel> =
        serde_json::from_slice(&std::fs::read(root.join("installed.json")).unwrap()).unwrap();
    let aliases: Vec<&str> = on_disk.iter().map(|r| r.alias.as_str()).collect();
    assert!(
        aliases.contains(&"away") && aliases.contains(&"away-audio"),
        "rows whose directory is missing must survive another install: {aliases:?}"
    );
    assert!(!store.installed().contains_key("away"));

    // The volume comes back: the same row lists again with no re-install.
    std::fs::create_dir_all(&offline).unwrap();
    assert_eq!(store.installed()["away"].path, offline);

    // Re-recording the same alias and modality still replaces, not appends.
    store.record(&row("away", &offline)).unwrap();
    let on_disk: Vec<InstalledModel> =
        serde_json::from_slice(&std::fs::read(root.join("installed.json")).unwrap()).unwrap();
    assert_eq!(on_disk.iter().filter(|r| r.alias == "away").count(), 1);
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
