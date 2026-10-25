//! Store layout, the install record, and alias resolution.
//!
//! The resolution ORDER is the load-bearing assertion in this file. Every
//! other test here is bookkeeping; that one is the difference between
//! `--model foo` running the model at `./foo` and running some other model
//! that happens to be recorded under that alias.

use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::PathBuf;

use turbospark_catalog::library::{
    parse_mount_table, Availability, LibraryMarker, LibraryRegistry, LocationEnv, MountSource,
    PortableRow, INTERNAL_LIBRARY_ID, MARKER_FILE,
};
use turbospark_catalog::{
    set_default_root, InstalledEntry, InstalledModel, ModelModality, Store, StoreError,
};

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
/// an internal row is written without either key. The external row is in a
/// listed, connected library, since recording refuses any other.
#[test]
fn library_fields_round_trip_and_internal_rows_omit_them() {
    let fx = LibraryFixture::new("library-fields");
    let store = &fx.store;
    let internal = row("here", &store.install_path("here"));
    let mut external = row("there", &fx.root().join("text").join("there.gturbo"));
    external.library = Some(fx.id.clone());
    external.relative_path = Some(PathBuf::from("text/there.gturbo"));
    store.record(&internal).unwrap();
    store.record(&external).unwrap();

    let text = std::fs::read_to_string(store.root().join("installed.json")).unwrap();
    let raw: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
    let internal_raw = raw.iter().find(|r| r["alias"] == "here").unwrap();
    assert!(internal_raw.get("library").is_none(), "{internal_raw}");
    assert!(
        internal_raw.get("relative_path").is_none(),
        "{internal_raw}"
    );
    let external_raw = raw.iter().find(|r| r["alias"] == "there").unwrap();
    assert_eq!(external_raw["library"], fx.id.as_str());
    assert_eq!(external_raw["relative_path"], "text/there.gturbo");

    let decoded: Vec<InstalledModel> = serde_json::from_str(&text).unwrap();
    assert_eq!(decoded, vec![internal, external]);
}

/// Library fields do not change the text listing: it still lists every text
/// row whose directory exists, keeps its recorded path, and hides the rest.
/// Availability-aware listing is a separate call. The rows are written as
/// any writer could have left them, since the reader must cope with rows of
/// libraries it does not know.
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
    std::fs::write(
        root.join("installed.json"),
        serde_json::to_vec_pretty(&[external.clone(), offline]).unwrap(),
    )
    .unwrap();

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

/// Because a corrupt record reads as empty, a write truncated by a crash
/// would cost every install at the next `record`. So a write replaces the
/// record through a rename and never rewrites it in place: a reader that
/// opened the record before the write still sees every byte of the old one
/// (6.4).
#[test]
fn recording_replaces_the_record_rather_than_rewriting_it_in_place() {
    use std::io::Read;
    let root = temp_root("replace-record");
    let store = Store::new(&root);
    store
        .record(&row("first", &root.join("models").join("first.gturbo")))
        .unwrap();
    let record = root.join("installed.json");
    let before = std::fs::read_to_string(&record).unwrap();
    let mut reader = std::fs::File::open(&record).unwrap();

    store
        .record(&row("second", &root.join("models").join("second.gturbo")))
        .unwrap();

    let mut seen = String::new();
    reader.read_to_string(&mut seen).unwrap();
    assert_eq!(seen, before, "the old record was rewritten in place");
    let on_disk: Vec<InstalledModel> =
        serde_json::from_slice(&std::fs::read(&record).unwrap()).unwrap();
    let aliases: Vec<&str> = on_disk.iter().map(|r| r.alias.as_str()).collect();
    assert_eq!(aliases, ["first", "second"]);
}

/// A writer that crashed between its temp file and the rename leaves the
/// temp file behind. The record it was replacing stays whole and readable,
/// and the next write still lands with every row.
#[test]
fn a_stale_temp_record_does_not_affect_the_existing_rows() {
    let root = temp_root("stale-temp-record");
    let store = Store::new(&root);
    let here = store.install_path("here");
    std::fs::create_dir_all(&here).unwrap();
    store.record(&row("here", &here)).unwrap();
    store
        .record(&row("away", &root.join("unmounted").join("away.gturbo")))
        .unwrap();
    let record = root.join("installed.json");
    let before = std::fs::read(&record).unwrap();
    // Half-written temp files, one named as a temp file of this process
    // could be.
    for name in [
        format!("installed.json.tmp-{}-0", std::process::id()),
        "installed.json.tmp-1-0".to_string(),
    ] {
        std::fs::write(root.join(name), &before[..before.len() / 2]).unwrap();
    }

    assert_eq!(std::fs::read(&record).unwrap(), before);
    assert_eq!(store.installed()["here"].path, here);

    store
        .record(&row("next", &root.join("models").join("next.gturbo")))
        .unwrap();
    let on_disk: Vec<InstalledModel> =
        serde_json::from_slice(&std::fs::read(&record).unwrap()).unwrap();
    let aliases: Vec<&str> = on_disk.iter().map(|r| r.alias.as_str()).collect();
    assert_eq!(aliases, ["here", "away", "next"]);
}

#[test]
fn writing_the_record_leaves_no_temp_file_behind() {
    let root = temp_root("no-temp-record");
    let store = Store::new(&root);
    let mut audio = row("voice", &root.join("models").join("voice.gturbo"));
    audio.modality = ModelModality::Audio;

    store
        .record(&row("a", &root.join("models").join("a.gturbo")))
        .unwrap();
    store
        .record(&row("b", &root.join("models").join("b.gturbo")))
        .unwrap();
    store.record(&audio).unwrap();
    store.forget("a").unwrap();
    store.forget_audio("voice").unwrap();

    let names: Vec<String> = std::fs::read_dir(&root)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, ["installed.json"]);
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

// Library-aware install roots, staging, and publication (2.4, 2.5, 2.6).

const UNLISTED_ID: &str = "7a1e9c04-52b3-4f8d-a6e1-3c9b0d2f5e17";

/// A temp home plus the fake volume `<base>/Volumes/T7` with a library added
/// on it, through a fixed mount table. `home_device` differs from the temp
/// dir's device so the same-disk refusal does not fire for the fake volume,
/// which shares that device.
struct LibraryFixture {
    base: PathBuf,
    store: Store,
    volume: PathBuf,
    env: LocationEnv,
    id: String,
}

impl LibraryFixture {
    fn new(tag: &str) -> Self {
        // Canonical, because the temp dir is behind the /var symlink and the
        // location rules compare canonical paths.
        let base = temp_root(tag).canonicalize().unwrap();
        let home = base.join("home");
        std::fs::create_dir_all(&home).unwrap();
        let volumes_root = base.join("Volumes");
        let volume = volumes_root.join("T7");
        std::fs::create_dir_all(&volume).unwrap();
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
        let added = LibraryRegistry::load(&home)
            .add(&volume, Some("T7"), &env)
            .unwrap();
        Self {
            base,
            store: Store::new(&home),
            volume,
            env,
            id: added.view.record.id,
        }
    }

    /// The library root as an install would get it.
    fn root(&self) -> PathBuf {
        self.store
            .library_root_for_install(Some(&self.id), &self.env)
            .unwrap()
    }

    fn record_bytes(&self) -> Option<Vec<u8>> {
        std::fs::read(self.store.root().join("installed.json")).ok()
    }
}

impl Drop for LibraryFixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.base).ok();
    }
}

#[test]
fn install_paths_in_a_library_root_mirror_the_internal_layout() {
    let root = temp_root("library-paths");
    let store = Store::new(&root);
    let external = PathBuf::from("/Volumes/T7/TurboSpark");

    assert_eq!(
        Store::install_path_in(&external, "qwen").unwrap(),
        external.join("text").join("qwen.gturbo")
    );
    assert_eq!(
        Store::image_install_path_in(&external, "z-image").unwrap(),
        external.join("image").join("z-image.gturbo")
    );
    assert_eq!(
        Store::audio_install_path_in(&external, "whisper").unwrap(),
        external.join("audio").join("whisper.gturbo")
    );
    assert_eq!(
        Store::vision_install_path_in(&external, "qwen").unwrap(),
        external.join("text").join("qwen.gturbo-vision")
    );

    // The internal root is `<home>/models`, so the same helpers give exactly
    // today's paths and every existing row keeps resolving.
    let models = store.models_root();
    assert_eq!(models, root.join("models"));
    assert_eq!(
        Store::install_path_in(&models, "qwen").unwrap(),
        store.install_path("qwen")
    );
    assert_eq!(
        Store::image_install_path_in(&models, "z-image").unwrap(),
        store.image_install_path("z-image")
    );
    assert_eq!(
        Store::audio_install_path_in(&models, "whisper").unwrap(),
        store.audio_install_path("whisper")
    );
    assert_eq!(
        Store::vision_install_path_in(&models, "qwen").unwrap(),
        store.vision_install_path("qwen")
    );
}

#[test]
fn an_alias_that_could_leave_the_library_root_is_refused() {
    let external = PathBuf::from("/Volumes/T7/TurboSpark");
    for alias in [
        "",
        ".hidden",
        "..",
        "../escape",
        "nested/alias",
        "/absolute",
    ] {
        for path in [
            Store::install_path_in(&external, alias),
            Store::image_install_path_in(&external, alias),
            Store::audio_install_path_in(&external, alias),
            Store::vision_install_path_in(&external, alias),
        ] {
            assert!(
                matches!(path, Err(StoreError::Io(_))),
                "{alias:?} gave {path:?}"
            );
        }
    }
}

#[test]
fn this_mac_is_the_install_root_without_a_library() {
    let fx = LibraryFixture::new("root-internal");
    let models = fx.store.models_root();
    assert_eq!(
        fx.store.library_root_for_install(None, &fx.env).unwrap(),
        models
    );
    assert_eq!(
        fx.store
            .library_root_for_install(Some(INTERNAL_LIBRARY_ID), &fx.env)
            .unwrap(),
        models
    );
}

#[test]
fn a_connected_library_is_the_install_root_at_its_canonical_path() {
    let fx = LibraryFixture::new("root-connected");
    let root = fx.root();
    assert_eq!(root, fx.volume.join("TurboSpark"));
    assert_eq!(root, root.canonicalize().unwrap());
    assert_eq!(
        Store::install_path_in(&root, "qwen").unwrap(),
        fx.volume
            .join("TurboSpark")
            .join("text")
            .join("qwen.gturbo")
    );
}

/// The registry refuses a linked library folder when a drive is added, but a
/// drive anyone could have written may swap one in later. A root leading off
/// its drive is refused, even to a folder carrying the library's marker.
#[test]
fn a_library_root_linked_off_its_drive_is_refused() {
    let fx = LibraryFixture::new("root-linked");
    let root = fx.volume.join("TurboSpark");
    let elsewhere = fx.base.join("elsewhere");
    std::fs::rename(&root, &elsewhere).unwrap();
    std::os::unix::fs::symlink(&elsewhere, &root).unwrap();

    assert!(matches!(
        fx.store.library_root_for_install(Some(&fx.id), &fx.env),
        Err(StoreError::Io(_))
    ));
}

#[test]
fn an_unplugged_library_is_not_connected() {
    let fx = LibraryFixture::new("root-unplugged");
    std::fs::remove_dir_all(&fx.volume).unwrap();
    let refused = fx.store.library_root_for_install(Some(&fx.id), &fx.env);
    assert_eq!(
        refused,
        Err(StoreError::NotConnected {
            library: "T7".to_string()
        })
    );
    assert!(refused.unwrap_err().to_string().contains("T7"));
}

#[test]
fn an_unlisted_library_is_refused() {
    let fx = LibraryFixture::new("root-unlisted");
    assert!(matches!(
        fx.store
            .library_root_for_install(Some(UNLISTED_ID), &fx.env),
        Err(StoreError::Io(_))
    ));
}

#[test]
fn a_library_written_by_a_newer_turbospark_refuses_installs() {
    let fx = LibraryFixture::new("root-read-only");
    let marker = fx.volume.join("TurboSpark").join(MARKER_FILE);
    let mut json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&marker).unwrap()).unwrap();
    json["format"] = serde_json::json!(2);
    std::fs::write(&marker, serde_json::to_vec_pretty(&json).unwrap()).unwrap();

    assert_eq!(
        fx.store.library_root_for_install(Some(&fx.id), &fx.env),
        Err(StoreError::LibraryReadOnly {
            library: "T7".to_string()
        })
    );
}

/// A listing only reads a library, so a library a newer TurboSpark wrote is
/// listed with its root although installs into it are refused, and This Mac,
/// which is every store's own, is not one of the libraries listed.
#[test]
fn connected_libraries_are_listed_at_their_canonical_roots_without_this_mac() {
    let fx = LibraryFixture::new("connected-libraries");

    let listed = fx.store.connected_libraries(&fx.env);

    assert_eq!(listed.len(), 1, "{listed:?}");
    assert_eq!(listed[0].0.id, fx.id);
    assert_eq!(listed[0].1, fx.volume.join("TurboSpark"));
    assert_eq!(listed[0].1, listed[0].1.canonicalize().unwrap());

    let marker = fx.volume.join("TurboSpark").join(MARKER_FILE);
    let mut json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&marker).unwrap()).unwrap();
    json["format"] = serde_json::json!(2);
    std::fs::write(&marker, serde_json::to_vec_pretty(&json).unwrap()).unwrap();
    assert!(fx
        .store
        .library_root_for_install(Some(&fx.id), &fx.env)
        .is_err());
    let listed = fx.store.connected_libraries(&fx.env);
    assert_eq!(listed.len(), 1, "a read-only library can still be read");
    assert_eq!(listed[0].1, fx.volume.join("TurboSpark"));
}

/// Not connected, or leading off its drive, is left out with no error: a
/// listing never raises for a drive that is away.
#[test]
fn a_library_that_is_away_or_linked_off_its_drive_is_not_listed_as_connected() {
    let fx = LibraryFixture::new("connected-libraries-away");
    let root = fx.volume.join("TurboSpark");
    let elsewhere = fx.base.join("elsewhere");
    std::fs::rename(&root, &elsewhere).unwrap();
    std::os::unix::fs::symlink(&elsewhere, &root).unwrap();
    assert!(fx.store.connected_libraries(&fx.env).is_empty());

    std::fs::remove_dir_all(&fx.volume).unwrap();
    assert!(fx.store.connected_libraries(&fx.env).is_empty());
}

#[test]
fn embedding_and_ocr_installs_stay_in_this_mac() {
    let fx = LibraryFixture::new("home-only");
    assert_eq!(Store::home_only(None), Ok(()));
    assert_eq!(Store::home_only(Some(INTERNAL_LIBRARY_ID)), Ok(()));
    assert_eq!(Store::home_only(Some(&fx.id)), Err(StoreError::HomeOnly));
    // Refused by kind, before any lookup: an unlisted id is refused the same.
    assert_eq!(
        Store::home_only(Some(UNLISTED_ID)),
        Err(StoreError::HomeOnly)
    );
}

/// Internal installs must not change at all: their staging parent is the
/// home they have always staged in, and nothing new appears under models.
#[test]
fn internal_installs_stage_in_the_home_as_before() {
    let root = temp_root("staging-internal");
    let store = Store::new(&root);
    assert_eq!(
        store.staging_parent_for(&store.models_root()).unwrap(),
        store.root()
    );

    std::fs::create_dir_all(store.models_root()).unwrap();
    assert_eq!(
        store.staging_parent_for(&store.models_root()).unwrap(),
        store.root()
    );
    assert_eq!(
        std::fs::read_dir(store.models_root()).unwrap().count(),
        0,
        "an internal install created something under models"
    );
}

#[test]
fn a_library_install_stages_in_the_librarys_private_staging_folder() {
    let fx = LibraryFixture::new("staging-external");
    let root = fx.root();

    let staging = fx.store.staging_parent_for(&root).unwrap();
    assert_eq!(staging, root.join(".staging"));
    let metadata = std::fs::symlink_metadata(&staging).unwrap();
    assert!(metadata.is_dir());
    assert_eq!(metadata.permissions().mode() & 0o777, 0o700);
    // Same volume as the destination, so publication is a rename.
    assert_eq!(metadata.dev(), std::fs::metadata(&root).unwrap().dev());
    // Reused rather than recreated.
    assert_eq!(fx.store.staging_parent_for(&root).unwrap(), staging);
}

#[test]
fn a_symlinked_or_non_folder_staging_entry_is_refused() {
    let fx = LibraryFixture::new("staging-symlink");
    let root = fx.root();
    let elsewhere = fx.base.join("elsewhere");
    std::fs::create_dir_all(&elsewhere).unwrap();
    std::os::unix::fs::symlink(&elsewhere, root.join(".staging")).unwrap();

    assert!(matches!(
        fx.store.staging_parent_for(&root),
        Err(StoreError::Io(_))
    ));
    assert_eq!(std::fs::read_dir(&elsewhere).unwrap().count(), 0);

    std::fs::remove_file(root.join(".staging")).unwrap();
    std::fs::write(root.join(".staging"), b"not a folder").unwrap();
    assert!(matches!(
        fx.store.staging_parent_for(&root),
        Err(StoreError::Io(_))
    ));
}

#[test]
fn publishing_moves_the_staged_install_into_the_library_and_records_nothing() {
    let fx = LibraryFixture::new("publish");
    let root = fx.root();
    let staged = fx.store.staging_parent_for(&root).unwrap().join("job");
    std::fs::create_dir_all(&staged).unwrap();
    std::fs::write(staged.join("weights.bin"), b"weights").unwrap();
    let destination = Store::install_path_in(&root, "qwen").unwrap();
    let record = fx.record_bytes();

    fx.store
        .publish_staged(&staged, &destination, "T7")
        .unwrap();

    assert_eq!(
        std::fs::read(destination.join("weights.bin")).unwrap(),
        b"weights"
    );
    assert!(!staged.exists());
    // The caller records the row, and only after a publish succeeds.
    assert_eq!(fx.record_bytes(), record);

    // An existing install is never replaced.
    std::fs::create_dir_all(&staged).unwrap();
    assert!(matches!(
        fx.store.publish_staged(&staged, &destination, "T7"),
        Err(StoreError::Io(_))
    ));
    assert!(staged.is_dir());
    assert!(destination.join("weights.bin").is_file());
}

/// This Mac's models folder may not exist yet in a fresh home. That is not a
/// disconnected drive, so publishing creates it.
#[test]
fn publishing_into_this_mac_creates_its_models_folder_in_a_fresh_home() {
    let root = temp_root("publish-internal");
    let store = Store::new(&root);
    let staged = store
        .staging_parent_for(&store.models_root())
        .unwrap()
        .join("job");
    std::fs::create_dir_all(&staged).unwrap();
    assert!(!store.models_root().exists());

    let destination = Store::install_path_in(&store.models_root(), "qwen").unwrap();
    store
        .publish_staged(&staged, &destination, "This Mac")
        .unwrap();
    assert!(destination.is_dir());
    assert_eq!(destination, store.install_path("qwen"));
}

/// The home does not unplug, so a staged install missing from it is an IO
/// error, never a disconnected drive named "This Mac".
#[test]
fn a_staged_install_missing_from_this_mac_is_an_io_error() {
    let root = temp_root("publish-internal-missing");
    let store = Store::new(&root);
    let staged = store
        .staging_parent_for(&store.models_root())
        .unwrap()
        .join("job");
    let destination = Store::install_path_in(&store.models_root(), "qwen").unwrap();

    let published = store.publish_staged(&staged, &destination, "This Mac");
    assert!(matches!(published, Err(StoreError::Io(_))), "{published:?}");
    assert!(!destination.exists());
}

/// rename(2) quietly replaces an EMPTY folder, so an existing install folder
/// that happens to be empty must still be refused rather than replaced.
#[test]
fn publishing_never_replaces_an_empty_existing_install_folder() {
    let fx = LibraryFixture::new("publish-empty-destination");
    let root = fx.root();
    let staged = fx.store.staging_parent_for(&root).unwrap().join("job");
    std::fs::create_dir_all(&staged).unwrap();
    std::fs::write(staged.join("weights.bin"), b"weights").unwrap();
    let destination = Store::install_path_in(&root, "qwen").unwrap();
    std::fs::create_dir_all(&destination).unwrap();

    let published = fx.store.publish_staged(&staged, &destination, "T7");
    assert!(matches!(published, Err(StoreError::Io(_))), "{published:?}");
    assert!(staged.join("weights.bin").is_file());
    assert_eq!(std::fs::read_dir(&destination).unwrap().count(), 0);
}

/// Another drive remounted under the same name can bring its own TurboSpark
/// folder, so the root is there while this install's staged folder is not.
/// That is still the drive going away mid-install, and nothing is created on
/// the drive that is there now.
#[test]
fn a_staged_install_gone_from_a_present_library_root_ends_disconnected() {
    let fx = LibraryFixture::new("publish-staged-gone");
    let kept = fx.store.models_root().join("text").join("kept.gturbo");
    std::fs::create_dir_all(&kept).unwrap();
    fx.store.record(&row("kept", &kept)).unwrap();
    let record = fx.record_bytes();
    assert!(record.is_some());

    let root = fx.root();
    let staged = fx.store.staging_parent_for(&root).unwrap().join("job");
    std::fs::create_dir_all(&staged).unwrap();
    let destination = Store::install_path_in(&root, "qwen").unwrap();
    std::fs::remove_dir_all(&staged).unwrap();

    assert_eq!(
        fx.store.publish_staged(&staged, &destination, "T7"),
        Err(StoreError::LibraryDisconnected {
            library: "T7".to_string()
        })
    );
    assert_eq!(fx.record_bytes(), record);
    assert!(root.is_dir());
    assert!(
        !root.join("text").exists(),
        "publishing created a modality folder on the drive"
    );
}

#[test]
fn publishing_through_a_symlinked_modality_folder_is_refused() {
    let fx = LibraryFixture::new("publish-symlink");
    let root = fx.root();
    let elsewhere = fx.base.join("elsewhere");
    std::fs::create_dir_all(&elsewhere).unwrap();
    std::os::unix::fs::symlink(&elsewhere, root.join("text")).unwrap();
    let staged = fx.store.staging_parent_for(&root).unwrap().join("job");
    std::fs::create_dir_all(&staged).unwrap();

    let destination = Store::install_path_in(&root, "qwen").unwrap();
    assert!(matches!(
        fx.store.publish_staged(&staged, &destination, "T7"),
        Err(StoreError::Io(_))
    ));
    assert!(staged.is_dir());
    assert_eq!(std::fs::read_dir(&elsewhere).unwrap().count(), 0);
}

/// A staged install that is a link is never published, wherever it points:
/// renaming the link into the library would list an install whose files
/// live outside it. A staged file is not an install either.
#[test]
fn publishing_refuses_a_staged_link_or_file() {
    let fx = LibraryFixture::new("publish-staged-link");
    let root = fx.root();
    let staging = fx.store.staging_parent_for(&root).unwrap();
    let elsewhere = fx.base.join("elsewhere").join("job");
    std::fs::create_dir_all(&elsewhere).unwrap();
    std::fs::write(elsewhere.join("weights.bin"), b"weights").unwrap();
    let link = staging.join("job");
    std::os::unix::fs::symlink(&elsewhere, &link).unwrap();
    let file = staging.join("file");
    std::fs::write(&file, b"not an install").unwrap();

    for staged in [&link, &file] {
        let destination = Store::install_path_in(&root, "qwen").unwrap();
        assert!(
            matches!(
                fx.store.publish_staged(staged, &destination, "T7"),
                Err(StoreError::Io(_))
            ),
            "{} was published",
            staged.display()
        );
        assert!(std::fs::symlink_metadata(&destination).is_err());
    }
    assert!(std::fs::symlink_metadata(&link)
        .unwrap()
        .file_type()
        .is_symlink());
    assert_eq!(
        std::fs::read(elsewhere.join("weights.bin")).unwrap(),
        b"weights"
    );
}

/// Unplugging the drive takes the staging folder with it. The install ends
/// as disconnected, and the registry is exactly as it was (2.4).
#[test]
fn an_install_whose_staging_vanished_ends_disconnected_and_records_nothing() {
    let fx = LibraryFixture::new("publish-unplugged");
    let kept = fx.store.models_root().join("text").join("kept.gturbo");
    std::fs::create_dir_all(&kept).unwrap();
    fx.store.record(&row("kept", &kept)).unwrap();
    let record = fx.record_bytes();
    assert!(record.is_some());

    let root = fx.root();
    let staged = fx.store.staging_parent_for(&root).unwrap().join("job");
    std::fs::create_dir_all(&staged).unwrap();
    std::fs::write(staged.join("weights.bin"), b"weights").unwrap();
    let destination = Store::install_path_in(&root, "qwen").unwrap();

    std::fs::remove_dir_all(&fx.volume).unwrap();
    let published = fx.store.publish_staged(&staged, &destination, "T7");

    assert_eq!(
        published,
        Err(StoreError::LibraryDisconnected {
            library: "T7".to_string()
        })
    );
    assert!(published.unwrap_err().to_string().contains("T7"));
    assert_eq!(fx.record_bytes(), record);
    assert!(fx.store.installed().contains_key("kept"));
    assert!(
        !fx.volume.exists(),
        "publishing recreated the unplugged volume"
    );
}

/// A library root that is gone is never recreated on the disk under it,
/// even when the staged install sits somewhere that still exists.
#[test]
fn a_vanished_library_root_is_never_recreated_by_publishing() {
    let fx = LibraryFixture::new("publish-root-gone");
    let root = fx.root();
    let staged = fx.base.join("still-here").join("job");
    std::fs::create_dir_all(&staged).unwrap();
    let destination = Store::install_path_in(&root, "qwen").unwrap();
    std::fs::remove_dir_all(&root).unwrap();

    assert_eq!(
        fx.store.publish_staged(&staged, &destination, "T7"),
        Err(StoreError::LibraryDisconnected {
            library: "T7".to_string()
        })
    );
    assert!(!root.exists(), "publishing recreated the library root");
    assert!(staged.is_dir());
}

// Availability-aware listing and markers kept in step (2.7).

impl LibraryFixture {
    /// The library root on the fake volume, canonical because `base` is.
    fn library_root(&self) -> PathBuf {
        self.volume.join("TurboSpark")
    }

    fn marker(&self) -> LibraryMarker {
        LibraryMarker::read(&self.library_root()).unwrap().unwrap()
    }

    fn marker_bytes(&self) -> Vec<u8> {
        std::fs::read(self.library_root().join(MARKER_FILE)).unwrap()
    }

    /// Rewrites the marker as a newer TurboSpark would, which makes the
    /// library read-only to this build.
    fn mark_newer(&self) {
        let path = self.library_root().join(MARKER_FILE);
        let mut json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        json["format"] = serde_json::json!(2);
        std::fs::write(&path, serde_json::to_vec_pretty(&json).unwrap()).unwrap();
    }

    /// The marker's rows as alias and modality, in marker order.
    fn marker_keys(&self) -> Vec<(String, ModelModality)> {
        self.marker()
            .models
            .into_iter()
            .map(|portable| (portable.alias, portable.modality))
            .collect()
    }

    /// A row for `alias` installed in this library, its folder created, as
    /// an installer records it: no `relative_path` of its own.
    fn library_row(&self, alias: &str, modality: ModelModality) -> InstalledModel {
        let root = self.root();
        let path = match modality {
            ModelModality::Text => Store::install_path_in(&root, alias),
            ModelModality::Image => Store::image_install_path_in(&root, alias),
            ModelModality::Audio => Store::audio_install_path_in(&root, alias),
        }
        .unwrap();
        std::fs::create_dir_all(&path).unwrap();
        let mut installed = row(alias, &path);
        installed.modality = modality;
        installed.library = Some(self.id.clone());
        installed
    }

    fn rows_on_disk(&self) -> Vec<InstalledModel> {
        serde_json::from_slice(&self.record_bytes().unwrap()).unwrap()
    }

    /// Replaces `installed.json` with `rows` as another writer would,
    /// bypassing the store.
    fn write_record(&self, rows: &[InstalledModel]) {
        std::fs::write(
            self.store.root().join("installed.json"),
            serde_json::to_vec_pretty(rows).unwrap(),
        )
        .unwrap();
    }

    fn record_inode(&self) -> Option<u64> {
        std::fs::metadata(self.store.root().join("installed.json"))
            .ok()
            .map(|metadata| metadata.ino())
    }

    fn entries(&self) -> Vec<InstalledEntry> {
        self.store.installed_with_availability(&self.env)
    }
}

fn entry<'a>(entries: &'a [InstalledEntry], alias: &str) -> &'a InstalledEntry {
    entries
        .iter()
        .find(|entry| entry.row.alias == alias)
        .unwrap_or_else(|| panic!("no entry for {alias}: {entries:?}"))
}

fn not_connected(library: &str) -> Availability {
    Availability::NotConnected {
        library: library.to_string(),
    }
}

/// Recording an install into a library adds its portable row to the
/// library's marker, keyed by alias and modality like the registry, and
/// forgetting it by install id takes the row out of both. Files are never
/// touched, and This Mac rows never touch a marker.
#[test]
fn recording_into_a_library_adds_its_marker_row_and_forgetting_removes_it() {
    let fx = LibraryFixture::new("marker-sync");
    assert!(fx.marker().models.is_empty());
    let qwen = fx.library_row("qwen", ModelModality::Text);

    // The String API every installer calls today keeps the marker in step.
    fx.store.record(&qwen).unwrap();
    let marker = fx.marker();
    assert_eq!(marker.models.len(), 1);
    assert_eq!(
        marker.models[0],
        PortableRow::from_installed(&qwen, &fx.root()).unwrap()
    );
    assert_eq!(
        marker.models[0].relative_path,
        PathBuf::from("text/qwen.gturbo")
    );
    // The registry row says where it is in the library, as the marker does.
    let recorded = fx.rows_on_disk();
    assert_eq!(recorded.len(), 1);
    assert_eq!(
        recorded[0].relative_path,
        Some(PathBuf::from("text/qwen.gturbo"))
    );
    assert_eq!(recorded[0].library.as_deref(), Some(fx.id.as_str()));

    let voice = fx.library_row("voice", ModelModality::Audio);
    fx.store
        .record_in_library(std::slice::from_ref(&voice), &fx.env)
        .unwrap();
    assert_eq!(
        fx.marker_keys(),
        [
            ("qwen".to_string(), ModelModality::Text),
            ("voice".to_string(), ModelModality::Audio)
        ]
    );

    // Recording the same alias and modality again replaces its marker row.
    let mut newer = qwen.clone();
    newer.revision = "0123456789abcdef0123456789abcdef01234567".to_string();
    fx.store.record(&newer).unwrap();
    let marker = fx.marker();
    assert_eq!(marker.models.len(), 2);
    let qwen_row = marker.models.iter().find(|p| p.alias == "qwen").unwrap();
    assert_eq!(qwen_row.revision, newer.revision);

    // A This Mac row never touches a marker.
    let before = fx.marker_bytes();
    let here = fx.store.install_path("here");
    std::fs::create_dir_all(&here).unwrap();
    fx.store.record(&row("here", &here)).unwrap();
    assert_eq!(fx.marker_bytes(), before);

    // Forgetting by install id drops the registry row and the marker row,
    // and leaves the install folder where it is.
    fx.store
        .forget_install(&newer.install_id(), &fx.env)
        .unwrap();
    let aliases: Vec<String> = fx.rows_on_disk().into_iter().map(|r| r.alias).collect();
    assert_eq!(aliases, ["voice", "here"]);
    assert_eq!(
        fx.marker_keys(),
        [("voice".to_string(), ModelModality::Audio)]
    );
    assert!(qwen.path.is_dir());

    // An id no row has changes nothing.
    let record = fx.record_bytes();
    let marker = fx.marker_bytes();
    fx.store
        .forget_install(&format!("{}:text/qwen", fx.id), &fx.env)
        .unwrap();
    assert_eq!(fx.record_bytes(), record);
    assert_eq!(fx.marker_bytes(), marker);

    // Forgetting a This Mac row leaves the marker alone.
    fx.store
        .forget_install("internal:text/here", &fx.env)
        .unwrap();
    assert_eq!(fx.marker_bytes(), marker);
    let aliases: Vec<String> = fx.rows_on_disk().into_iter().map(|r| r.alias).collect();
    assert_eq!(aliases, ["voice"]);
    assert!(here.is_dir());
}

/// A library whose marker a newer TurboSpark wrote, or that is not
/// connected, refuses the record before anything is written, so the registry
/// never holds a row its marker lacks.
#[test]
fn recording_into_a_read_only_or_unplugged_library_records_nothing() {
    let fx = LibraryFixture::new("marker-refusals");
    let kept = fx.store.install_path("kept");
    std::fs::create_dir_all(&kept).unwrap();
    fx.store.record(&row("kept", &kept)).unwrap();
    let record = fx.record_bytes();
    let qwen = fx.library_row("qwen", ModelModality::Text);
    let marker_path = fx.library_root().join(MARKER_FILE);

    fx.mark_newer();
    let newer = fx.marker_bytes();
    assert_eq!(
        fx.store
            .record_in_library(std::slice::from_ref(&qwen), &fx.env),
        Err(StoreError::LibraryReadOnly {
            library: "T7".to_string()
        })
    );
    let refused = fx.store.record(&qwen).unwrap_err();
    assert!(refused.contains("read-only"), "{refused}");
    assert_eq!(fx.record_bytes(), record);
    assert_eq!(fx.marker_bytes(), newer);

    // A marker that cannot be read is a drive being unplugged, as for an
    // install root.
    std::fs::write(&marker_path, b"not json").unwrap();
    assert_eq!(
        fx.store
            .record_in_library(std::slice::from_ref(&qwen), &fx.env),
        Err(StoreError::NotConnected {
            library: "T7".to_string()
        })
    );
    assert_eq!(fx.record_bytes(), record);

    std::fs::remove_dir_all(&fx.volume).unwrap();
    assert_eq!(
        fx.store
            .record_in_library(std::slice::from_ref(&qwen), &fx.env),
        Err(StoreError::NotConnected {
            library: "T7".to_string()
        })
    );
    assert_eq!(fx.record_bytes(), record);
}

/// A row of a library that is not listed would have no marker to go into,
/// so re-adding the library would drop the row and leave its folder
/// unlisted on the drive (1.6). Recording refuses it before writing
/// anything, as when an install finishes just after its library was
/// removed, or when the library list cannot be read.
#[test]
fn recording_into_a_library_that_is_not_listed_writes_nothing() {
    let fx = LibraryFixture::new("record-unlisted");
    let kept = fx.store.install_path("kept");
    std::fs::create_dir_all(&kept).unwrap();
    fx.store.record(&row("kept", &kept)).unwrap();
    let qwen = fx.library_row("qwen", ModelModality::Text);
    let here = row("here", &fx.store.install_path("here"));

    // The library is removed while the install finishes.
    LibraryRegistry::load(fx.store.root())
        .remove(&fx.id)
        .unwrap();
    let record = fx.record_bytes();
    let marker = fx.marker_bytes();

    let refused = fx.store.record(&qwen).unwrap_err();
    assert!(refused.contains("is listed"), "{refused}");
    // One refused row refuses the batch, This Mac rows included.
    match fx
        .store
        .record_in_library(&[here.clone(), qwen.clone()], &fx.env)
    {
        Err(StoreError::Io(reason)) => assert!(reason.contains(&fx.id), "{reason}"),
        other => panic!("expected an unlisted library refusal, got {other:?}"),
    }
    let mut unknown = qwen.clone();
    unknown.library = Some(UNLISTED_ID.to_string());
    assert!(matches!(
        fx.store
            .record_in_library(std::slice::from_ref(&unknown), &fx.env),
        Err(StoreError::Io(_))
    ));
    assert_eq!(fx.record_bytes(), record);
    assert_eq!(fx.marker_bytes(), marker);

    // A list that cannot be read cannot vouch for any library, and the
    // refusal says why.
    let list = fx.store.root().join("libraries.json");
    std::fs::remove_file(&list).unwrap();
    std::fs::create_dir(&list).unwrap();
    match fx
        .store
        .record_in_library(std::slice::from_ref(&qwen), &fx.env)
    {
        Err(StoreError::Io(reason)) => assert!(reason.contains("cannot be read"), "{reason}"),
        other => panic!("expected an unreadable list refusal, got {other:?}"),
    }
    assert_eq!(fx.record_bytes(), record);
    assert_eq!(fx.marker_bytes(), marker);
}

/// The marker is written before the registry, so a failure between them
/// can only leave the marker with a row the registry lacks, which re-adding
/// the library tolerates. A marker that cannot be written records nothing.
#[test]
fn a_marker_that_cannot_be_written_records_nothing() {
    let fx = LibraryFixture::new("marker-unwritable");
    let qwen = fx.library_row("qwen", ModelModality::Text);
    let root = fx.library_root();
    assert!(fx.record_bytes().is_none());

    // The marker can be read but not replaced.
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o555)).unwrap();
    let recorded = fx
        .store
        .record_in_library(std::slice::from_ref(&qwen), &fx.env);
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o755)).unwrap();

    assert!(matches!(recorded, Err(StoreError::Io(_))), "{recorded:?}");
    assert!(fx.record_bytes().is_none(), "the registry was written");
    assert!(fx.marker().models.is_empty());
}

/// Forgetting writes the registry before the marker, for the same reason:
/// a registry that cannot be written leaves the marker still listing the
/// row the registry still holds.
#[test]
fn forgetting_writes_the_registry_before_the_marker() {
    let fx = LibraryFixture::new("forget-order");
    let qwen = fx.library_row("qwen", ModelModality::Text);
    fx.store.record(&qwen).unwrap();
    let record = fx.record_bytes();
    let home = fx.store.root().to_path_buf();

    // The registry can be read but not replaced, as on a full disk.
    std::fs::set_permissions(&home, std::fs::Permissions::from_mode(0o555)).unwrap();
    let forgot = fx.store.forget_install(&qwen.install_id(), &fx.env);
    std::fs::set_permissions(&home, std::fs::Permissions::from_mode(0o755)).unwrap();

    assert!(matches!(forgot, Err(StoreError::Io(_))), "{forgot:?}");
    assert_eq!(fx.record_bytes(), record);
    assert_eq!(
        fx.marker_keys(),
        [("qwen".to_string(), ModelModality::Text)],
        "the marker lost a row the registry still holds"
    );
}

/// Forgetting a row whose library is not connected would leave its marker
/// listing it, so it is refused with the library to connect, and nothing
/// changes.
#[test]
fn forgetting_an_install_whose_library_is_unplugged_is_refused() {
    let fx = LibraryFixture::new("forget-unplugged");
    let qwen = fx.library_row("qwen", ModelModality::Text);
    fx.store.record(&qwen).unwrap();
    let record = fx.record_bytes();
    std::fs::remove_dir_all(&fx.volume).unwrap();

    assert_eq!(
        fx.store.forget_install(&qwen.install_id(), &fx.env),
        Err(StoreError::NotConnected {
            library: "T7".to_string()
        })
    );
    assert_eq!(fx.record_bytes(), record);
}

/// Forgetting a row in a library a newer TurboSpark wrote would leave its
/// marker listing the row, since this build cannot rewrite that marker, so
/// it is refused before anything is written.
#[test]
fn forgetting_an_install_in_a_read_only_library_is_refused() {
    let fx = LibraryFixture::new("forget-read-only");
    let qwen = fx.library_row("qwen", ModelModality::Text);
    fx.store.record(&qwen).unwrap();
    fx.mark_newer();
    let record = fx.record_bytes();
    let marker = fx.marker_bytes();

    assert_eq!(
        fx.store.forget_install(&qwen.install_id(), &fx.env),
        Err(StoreError::LibraryReadOnly {
            library: "T7".to_string()
        })
    );
    assert_eq!(fx.record_bytes(), record);
    assert_eq!(fx.marker_bytes(), marker);
}

/// A library root that leads off its drive, as a drive anyone could have
/// written can arrange, is connected but cannot be trusted. Its rows are
/// missing, not waiting for a drive to be connected, and their recorded
/// paths are not resolved through the link.
#[test]
fn rows_of_a_library_whose_root_leads_off_its_drive_are_missing() {
    let fx = LibraryFixture::new("listing-linked-root");
    let qwen = fx.library_row("qwen", ModelModality::Text);
    fx.store.record(&qwen).unwrap();
    let recorded = fx.rows_on_disk();
    let root = fx.library_root();
    let elsewhere = fx.base.join("elsewhere");
    std::fs::rename(&root, &elsewhere).unwrap();
    std::os::unix::fs::symlink(&elsewhere, &root).unwrap();
    assert!(root.join("text").join("qwen.gturbo").is_dir());

    let entries = fx.entries();
    let listed = entry(&entries, "qwen");
    assert_eq!(listed.availability, Availability::Missing);
    assert_eq!(listed.library_name, "T7");
    assert_eq!(listed.row, recorded[0]);
}

/// Every row stays listed (3.1, 10.2): an unplugged library's rows as not
/// connected with its name, a deleted folder in a connected library as
/// missing (3.7). The text listing is exactly what it was.
#[test]
fn listing_shows_offline_rows_as_not_connected_and_deleted_folders_as_missing() {
    let fx = LibraryFixture::new("listing");
    let qwen = fx.library_row("qwen", ModelModality::Text);
    fx.store.record(&qwen).unwrap();
    let gone = fx.library_row("gone", ModelModality::Image);
    fx.store.record(&gone).unwrap();
    std::fs::remove_dir_all(&gone.path).unwrap();
    let here = fx.store.install_path("here");
    std::fs::create_dir_all(&here).unwrap();
    fx.store.record(&row("here", &here)).unwrap();
    let lost = fx.store.install_path("lost");
    fx.store.record(&row("lost", &lost)).unwrap();

    let entries = fx.entries();
    assert_eq!(entries.len(), 4);
    let listed = entry(&entries, "qwen");
    assert_eq!(listed.availability, Availability::Available);
    assert_eq!(listed.library_name, "T7");
    assert_eq!(listed.install_id, format!("{}:text/qwen", fx.id));
    assert_eq!(listed.row.path, fx.root().join("text").join("qwen.gturbo"));
    let listed = entry(&entries, "gone");
    assert_eq!(listed.availability, Availability::Missing);
    assert_eq!(listed.library_name, "T7");
    assert_eq!(listed.install_id, format!("{}:image/gone", fx.id));
    let listed = entry(&entries, "here");
    assert_eq!(listed.availability, Availability::Available);
    assert_eq!(listed.library_name, "This Mac");
    assert_eq!(listed.install_id, "internal:text/here");
    let listed = entry(&entries, "lost");
    assert_eq!(listed.availability, Availability::Missing);
    assert_eq!(listed.library_name, "This Mac");

    // The text listing still lists available text rows at their recorded
    // paths and hides the rest.
    let recorded = fx.rows_on_disk();
    let installed = fx.store.installed();
    assert_eq!(installed.keys().collect::<Vec<_>>(), ["here", "qwen"]);
    assert_eq!(
        &installed["qwen"],
        recorded.iter().find(|r| r.alias == "qwen").unwrap()
    );

    // Unplugged: the drive's folder leaves the volumes root.
    std::fs::rename(&fx.volume, fx.base.join("unplugged")).unwrap();
    let entries = fx.entries();
    assert_eq!(entries.len(), 4);
    for alias in ["qwen", "gone"] {
        let listed = entry(&entries, alias);
        assert_eq!(listed.availability, not_connected("T7"), "{alias}");
        assert_eq!(listed.library_name, "T7");
        // The last resolved path is kept, not rewritten.
        assert_eq!(
            &listed.row,
            recorded.iter().find(|r| r.alias == alias).unwrap()
        );
    }
    assert_eq!(
        entry(&entries, "qwen").install_id,
        format!("{}:text/qwen", fx.id)
    );
    assert_eq!(
        entry(&entries, "here").availability,
        Availability::Available
    );

    // The CLI asks before resolving, so it can name the drive (10.7).
    assert_eq!(
        fx.store.availability_of("qwen", &fx.env),
        Some(not_connected("T7"))
    );
    assert_eq!(
        fx.store.availability_of("here", &fx.env),
        Some(Availability::Available)
    );
    assert_eq!(
        fx.store.availability_of("lost", &fx.env),
        Some(Availability::Missing)
    );
    // Only text rows answer to an alias.
    assert_eq!(fx.store.availability_of("gone", &fx.env), None);
    assert_eq!(fx.store.availability_of("nope", &fx.env), None);
    assert_eq!(fx.store.installed().keys().collect::<Vec<_>>(), ["here"]);
}

/// A library row's `path` is where its library is now, joined from its
/// relative path, rather than the last path recorded.
#[test]
fn a_library_rows_path_is_where_its_library_is_now() {
    let fx = LibraryFixture::new("listing-path");
    let mut qwen = fx.library_row("qwen", ModelModality::Text);
    qwen.relative_path = Some(PathBuf::from("text/qwen.gturbo"));
    qwen.path = PathBuf::from("/Volumes/Old Name/TurboSpark/text/qwen.gturbo");
    fx.store
        .record_in_library(std::slice::from_ref(&qwen), &fx.env)
        .unwrap();

    let entries = fx.entries();
    let listed = entry(&entries, "qwen");
    assert_eq!(listed.availability, Availability::Available);
    assert_eq!(listed.row.path, fx.root().join("text").join("qwen.gturbo"));
}

/// An older writer keeps `path` but drops `library` and `relative_path`.
/// Listing gives them back from the marker of the connected library the path
/// is in, records that, and never writes the marker.
#[test]
fn a_row_an_older_writer_stripped_regains_its_library() {
    let fx = LibraryFixture::new("listing-restore");
    let qwen = fx.library_row("qwen", ModelModality::Text);
    fx.store.record(&qwen).unwrap();
    let mut stripped = fx.rows_on_disk();
    for row in &mut stripped {
        row.library = None;
        row.relative_path = None;
    }
    // A folder under the library root that the marker does not list.
    let stray = fx.root().join("text").join("stray.gturbo");
    std::fs::create_dir_all(&stray).unwrap();
    stripped.push(row("stray", &stray));
    fx.write_record(&stripped);
    let marker = fx.marker_bytes();

    let entries = fx.entries();
    let restored = entry(&entries, "qwen");
    assert_eq!(restored.row.library.as_deref(), Some(fx.id.as_str()));
    assert_eq!(
        restored.row.relative_path,
        Some(PathBuf::from("text/qwen.gturbo"))
    );
    assert_eq!(restored.install_id, format!("{}:text/qwen", fx.id));
    assert_eq!(restored.library_name, "T7");
    assert_eq!(restored.availability, Availability::Available);

    // Recorded, so the next reader sees the library too.
    let on_disk = fx.rows_on_disk();
    let qwen_now = on_disk.iter().find(|r| r.alias == "qwen").unwrap();
    assert_eq!(qwen_now.library.as_deref(), Some(fx.id.as_str()));
    assert_eq!(
        qwen_now.relative_path,
        Some(PathBuf::from("text/qwen.gturbo"))
    );
    assert_eq!(fx.marker_bytes(), marker);

    // A folder the marker does not list stays a custom location.
    let listed = entry(&entries, "stray");
    assert_eq!(listed.row.library, None);
    assert_eq!(listed.library_name, "custom location");
    assert_eq!(listed.install_id, "internal:text/stray");
    assert_eq!(listed.availability, Availability::Available);
    assert_eq!(
        on_disk.iter().find(|r| r.alias == "stray").unwrap().library,
        None
    );
}

/// Listing writes the registry only to restore stripped rows. Otherwise it
/// leaves `installed.json` as it is, and never creates one.
#[test]
fn listing_writes_the_registry_only_to_restore_stripped_rows() {
    let fx = LibraryFixture::new("listing-writes");
    assert!(fx.entries().is_empty());
    assert!(fx.record_bytes().is_none());

    let qwen = fx.library_row("qwen", ModelModality::Text);
    fx.store.record(&qwen).unwrap();
    let written = fx.record_inode();
    fx.entries();
    assert_eq!(fx.record_inode(), written, "a listing rewrote the registry");

    let mut stripped = fx.rows_on_disk();
    stripped[0].library = None;
    stripped[0].relative_path = None;
    fx.write_record(&stripped);
    let stripped_inode = fx.record_inode();
    fx.entries();
    assert_ne!(fx.record_inode(), stripped_inode);

    let restored = fx.record_inode();
    fx.entries();
    assert_eq!(fx.record_inode(), restored);
}

/// A row with no library outside the home is a custom location (`pull
/// --out`), never This Mac. One under a volume that is not mounted is not
/// connected, named by the volume.
#[test]
fn rows_outside_the_home_without_a_library_are_custom_locations() {
    let fx = LibraryFixture::new("listing-custom");
    let out = fx.base.join("elsewhere").join("out.gturbo");
    std::fs::create_dir_all(&out).unwrap();
    let away = fx.env.volumes_root.join("Gone").join("away.gturbo");
    let loose = fx.volume.join("loose").join("loose.gturbo");
    let absent = fx.base.join("elsewhere").join("absent.gturbo");
    for (alias, path) in [
        ("out", &out),
        ("away", &away),
        ("loose", &loose),
        ("absent", &absent),
    ] {
        fx.store.record(&row(alias, path)).unwrap();
    }

    let entries = fx.entries();
    for listed in &entries {
        assert_eq!(listed.library_name, "custom location", "{listed:?}");
        assert_eq!(listed.row.library, None);
        assert_eq!(
            listed.install_id,
            format!("internal:text/{}", listed.row.alias)
        );
    }
    assert_eq!(entry(&entries, "out").availability, Availability::Available);
    assert_eq!(entry(&entries, "out").row.path, out);
    assert_eq!(entry(&entries, "away").availability, not_connected("Gone"));
    assert_eq!(entry(&entries, "loose").availability, Availability::Missing);
    assert_eq!(
        entry(&entries, "absent").availability,
        Availability::Missing
    );
    assert_eq!(fx.store.installed().keys().collect::<Vec<_>>(), ["out"]);
}

/// Rows can outlive their library's place in the list, as after a hand edit
/// of `libraries.json`. With no root to look in they are missing, under a
/// name that says why, and can still be forgotten.
#[test]
fn rows_of_a_library_no_longer_listed_are_missing() {
    let fx = LibraryFixture::new("listing-unknown");
    let path = fx.root().join("text").join("orphan.gturbo");
    std::fs::create_dir_all(&path).unwrap();
    let mut orphan = row("orphan", &path);
    orphan.library = Some(UNLISTED_ID.to_string());
    // Written as another writer left it: recording refuses an unlisted id.
    fx.write_record(std::slice::from_ref(&orphan));

    let entries = fx.entries();
    let listed = entry(&entries, "orphan");
    assert_eq!(listed.availability, Availability::Missing);
    assert_eq!(listed.library_name, "unknown library");
    assert_eq!(listed.install_id, format!("{UNLISTED_ID}:text/orphan"));

    fx.store
        .forget_install(&orphan.install_id(), &fx.env)
        .unwrap();
    assert!(fx.rows_on_disk().is_empty());
    assert!(path.is_dir());
}

/// `relative_path` in a registry row is unchecked input. A `..` component,
/// or a linked folder anywhere along the path inside the library, could lead
/// out of the library, so the row is missing and the link is not followed,
/// even when what it leads to is a real install folder.
#[test]
fn a_library_row_that_could_lead_out_of_its_library_is_missing() {
    let fx = LibraryFixture::new("listing-escape");
    let root = fx.root();
    let outside = fx.base.join("outside");
    std::fs::create_dir_all(outside.join("evil.gturbo")).unwrap();
    std::fs::create_dir_all(root.join("text")).unwrap();
    // A linked install folder, and a linked modality folder.
    std::os::unix::fs::symlink(
        outside.join("evil.gturbo"),
        root.join("text").join("linked.gturbo"),
    )
    .unwrap();
    std::os::unix::fs::symlink(&outside, root.join("image")).unwrap();
    let in_library = |alias: &str, modality: ModelModality, relative: &str| {
        let mut installed = row(alias, &root.join(relative));
        installed.modality = modality;
        installed.library = Some(fx.id.clone());
        installed.relative_path = Some(PathBuf::from(relative));
        installed
    };
    let traversal = in_library(
        "traversal",
        ModelModality::Text,
        "../../../outside/evil.gturbo",
    );
    let linked = in_library("linked", ModelModality::Text, "text/linked.gturbo");
    let via = in_library("via", ModelModality::Image, "image/evil.gturbo");
    fx.write_record(&[traversal.clone(), linked, via]);
    assert!(root.join("../../../outside/evil.gturbo").is_dir());
    assert!(root.join("text/linked.gturbo").is_dir());
    assert!(root.join("image/evil.gturbo").is_dir());

    let entries = fx.entries();
    assert_eq!(entries.len(), 3);
    for listed in &entries {
        assert_eq!(listed.availability, Availability::Missing, "{listed:?}");
        assert_eq!(listed.library_name, "T7");
    }
    // A path that does not join is left as recorded rather than resolved.
    assert_eq!(entry(&entries, "traversal").row.path, traversal.path);
}
