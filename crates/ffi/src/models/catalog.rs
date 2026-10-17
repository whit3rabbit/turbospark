//! Catalog and local store model management.

use catalog::library::{
    Availability, LibraryRecord, LibraryRegistry, LocationEnv, INTERNAL_LIBRARY_ID,
};
use catalog::{Catalog, InstalledEntry, ModelModality, Store};
use serde::Serialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize)]
struct ImageInstalledRow {
    alias: String,
    #[serde(rename = "modelID")]
    model_id: String,
    revision: String,
    path: PathBuf,
    width: u32,
    height: u32,
    #[serde(rename = "schedulerSteps")]
    scheduler_steps: u32,
    quantization: String,
    #[serde(rename = "sourcePath")]
    source_path: Option<PathBuf>,
    /// `catalog::image_family_for_model_id`, so a host does not restate the
    /// model-id-to-family routing (it drifts silently when a family is added).
    family: String,
}

#[derive(Serialize)]
struct CatalogRow<'a> {
    #[serde(flatten)]
    entry: &'a catalog::CatalogEntry,
    /// Whether this row is already installed, so a GUI needs one call rather
    /// than two and a join.
    installed: bool,
}

pub(crate) fn catalog_json() -> Result<String, String> {
    let catalog = Catalog::embedded()?;
    let installed = Store::default_store()
        .map(|s| s.installed())
        .unwrap_or_default();
    let rows: Vec<CatalogRow> = catalog
        .entries()
        .map(|entry| CatalogRow {
            installed: installed.contains_key(&entry.alias),
            entry,
        })
        .collect();
    serde_json::to_string(&rows).map_err(|e| e.to_string())
}

/// The text installs, with where each is and whether it can be opened now
/// (3.1). A row in a library that is not connected, or whose folder is gone,
/// stays listed and says so, because an unplugged drive is a state and not a
/// fault (3.4): the app filters its pickers, and this list does not.
pub(crate) fn installed_json() -> Result<String, String> {
    let store = Store::default_store()?;
    let env = LocationEnv::system(store.root())?;
    installed_json_in(&store, &env)
}

/// Every key a row had before libraries is kept. The additions are
/// `library` (the library's id, `"internal"` for This Mac, null for a custom
/// location), `library_name`, `install_id`, `availability` (`available`,
/// `not_connected` or `missing`), `location_kind` (`internal`, `library`,
/// `custom` or `unknown`) and `connect_name` (the library or volume to
/// connect, else null). They are snake_case like the keys beside them.
/// Names come from a drive anyone could have written, so they are raw here
/// and the app truncates them for display.
fn installed_json_in(store: &Store, env: &LocationEnv) -> Result<String, String> {
    // Only text rows, as before: image and audio rows have their own lists.
    let mut entries: Vec<InstalledEntry> = store
        .installed_with_availability(env)
        .into_iter()
        .filter(|entry| entry.row.effective_modality() == ModelModality::Text)
        .collect();
    entries.sort_by(|a, b| a.row.alias.cmp(&b.row.alias));
    let listed: Vec<String> = LibraryRegistry::load(store.root())
        .libraries()
        .into_iter()
        .map(|record| record.id)
        .collect();
    let models = store.models_root();
    let rows = entries
        .iter()
        .map(|entry| installed_row(entry, &listed, &models))
        .collect::<Result<Vec<_>, _>>()?;
    serde_json::to_string(&rows).map_err(|e| e.to_string())
}

fn installed_row(
    entry: &InstalledEntry,
    listed: &[String],
    models: &Path,
) -> Result<Value, String> {
    let row = &entry.row;
    // The place is told from the row's ids and path, as the catalog does, and
    // never from `library_name`: a drive's marker can name its library "This
    // Mac" or "custom location".
    let external = row
        .library
        .as_deref()
        .filter(|id| *id != INTERNAL_LIBRARY_ID);
    let (location_kind, library) = match external {
        Some(id) if listed.iter().any(|known| known == id) => ("library", Some(id)),
        Some(id) => ("unknown", Some(id)),
        None if row.path.starts_with(models) => ("internal", Some(INTERNAL_LIBRARY_ID)),
        None => ("custom", None),
    };
    let (availability, connect_name) = match &entry.availability {
        Availability::Available => ("available", None),
        Availability::NotConnected { library } => ("not_connected", Some(library.as_str())),
        Availability::Missing => ("missing", None),
    };
    let mut value = serde_json::to_value(row).map_err(|e| e.to_string())?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| "an installed row is not an object".to_string())?;
    object.insert("library".to_string(), json!(library));
    object.insert("library_name".to_string(), json!(entry.library_name));
    object.insert("install_id".to_string(), json!(entry.install_id));
    object.insert("availability".to_string(), json!(availability));
    object.insert("location_kind".to_string(), json!(location_kind));
    object.insert("connect_name".to_string(), json!(connect_name));
    Ok(value)
}

/// An image install and the library holding it, as the app lists it. The row
/// keeps the keys it had before libraries, and its library is camelCase like
/// they are.
#[derive(Debug, Serialize)]
struct LibraryImageRow {
    #[serde(flatten)]
    row: ImageInstalledRow,
    /// The library's id, `"internal"` for This Mac.
    library: String,
    #[serde(rename = "libraryName")]
    library_name: String,
    /// `<library>:image/<alias>`, the identity a move or a forget takes.
    #[serde(rename = "installId")]
    install_id: String,
}

impl LibraryImageRow {
    fn new(row: ImageInstalledRow, library: &str, library_name: &str) -> Self {
        Self {
            install_id: format!("{library}:image/{}", row.alias),
            library: library.to_string(),
            library_name: library_name.to_string(),
            row,
        }
    }
}

/// Image installs are deliberately not recorded in the text model index.
/// Discover only valid image manifests under the shared machine store, and in
/// every library that is connected, so the app can offer compatible rows
/// without making image artifacts text models. A library that is not
/// connected is skipped without a word: it has nothing to say until its drive
/// is back (3.4).
pub(crate) fn image_installed_json() -> Result<String, String> {
    let store = Store::default_store()?;
    let env = LocationEnv::system(store.root())?;
    image_installed_json_in(&store, &env)
}

fn image_installed_json_in(store: &Store, env: &LocationEnv) -> Result<String, String> {
    let this_mac = LibraryRecord::internal(store.root()).name;
    let mut rows: Vec<LibraryImageRow> =
        image_installed_rows(&store.modality_root(ModelModality::Image))
            .into_iter()
            .chain(image_installed_rows(&store.models_root()))
            .map(|row| LibraryImageRow::new(row, INTERNAL_LIBRARY_ID, &this_mac))
            .collect();
    for (record, root) in store.connected_libraries(env) {
        rows.extend(
            library_image_rows(&root)
                .into_iter()
                .map(|row| LibraryImageRow::new(row, &record.id, &record.name)),
        );
    }
    rows.sort_by(|a, b| {
        a.row
            .alias
            .cmp(&b.row.alias)
            .then_with(|| a.row.path.cmp(&b.row.path))
    });
    serde_json::to_string(&rows).map_err(|e| e.to_string())
}

/// Deletes one valid image install from the shared image store.
///
/// Image installs are absent from the text index, so deletion resolves the
/// path (or an unambiguous legacy alias) from the same validated manifest
/// listing used by the app. That keeps arbitrary paths from becoming deletion
/// targets and preserves the selected row when canonical and legacy aliases
/// collide.
pub(crate) fn delete_image(identifier: &str) -> Result<(), String> {
    let store = Store::default_store()?;
    let canonical = store.modality_root(ModelModality::Image);
    let legacy = store.models_root();
    let rows = image_installed_rows(&canonical)
        .into_iter()
        .chain(image_installed_rows(&legacy))
        .collect();
    let row = image_row_for_delete(rows, identifier)?;
    if let Some(source_path) = row.source_path.as_ref().filter(|path| path.exists()) {
        std::fs::remove_dir_all(source_path)
            .map_err(|e| format!("failed to remove {}: {e}", source_path.display()))?;
    }
    if row.path.exists() {
        std::fs::remove_dir_all(&row.path)
            .map_err(|e| format!("failed to remove {}: {e}", row.path.display()))?;
    }
    Ok(())
}

fn image_row_for_delete(
    rows: Vec<ImageInstalledRow>,
    identifier: &str,
) -> Result<ImageInstalledRow, String> {
    if let Some(index) = rows
        .iter()
        .position(|row| row.path == Path::new(identifier))
    {
        return Ok(rows.into_iter().nth(index).expect("index came from rows"));
    }

    let mut matches = rows.into_iter().filter(|row| row.alias == identifier);
    let row = matches
        .next()
        .ok_or_else(|| format!("image model {identifier:?} is not installed"))?;
    if matches.next().is_some() {
        return Err(format!(
            "image model alias {identifier:?} is ambiguous; delete by its installed path"
        ));
    }
    Ok(row)
}

/// Valid image installs under a user-chosen folder, for locations outside the
/// managed store (an external drive, a shared models directory).
///
/// Looks at `root` and its immediate subdirectories only. A folder like this
/// can be a whole drive, and the text-model scan's recursive walk is the cost
/// to avoid repeating. `root` may also be an install bundle itself, so a
/// folder picked by hand resolves to that one install. Rows go through the
/// same manifest validation as the managed store and report the retained MLX
/// source tree when the `.{alias}.{revision}.image-source` sibling exists, so
/// an install copied with its source runs and one copied without it is
/// reported without a source path.
pub(crate) fn image_installed_in_json(root: &Path) -> Result<String, String> {
    if !root.is_dir() {
        return Err(format!("{} is not a directory", root.display()));
    }
    let mut rows = Vec::new();
    let is_bundle = |path: &Path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".gturbo"))
    };
    if is_bundle(root) {
        if let Some(parent) = root.parent() {
            rows.extend(
                image_installed_rows(parent)
                    .into_iter()
                    .filter(|row| row.path == root),
            );
        }
    } else {
        rows.extend(image_installed_rows(root));
        if let Ok(children) = std::fs::read_dir(root) {
            for child in children.filter_map(Result::ok) {
                let path = child.path();
                let hidden = child.file_name().to_string_lossy().starts_with('.');
                // Bundles were listed above, and the dot-directories are the
                // retained source trees, which hold no installs.
                if path.is_dir() && !hidden && !is_bundle(&path) {
                    rows.extend(image_installed_rows(&path));
                }
            }
        }
    }
    rows.sort_by(|a, b| a.alias.cmp(&b.alias).then_with(|| a.path.cmp(&b.path)));
    rows.dedup_by(|a, b| a.path == b.path);
    serde_json::to_string(&rows).map_err(|e| e.to_string())
}

fn image_installed_rows(models: &Path) -> Vec<ImageInstalledRow> {
    let Ok(entries) = std::fs::read_dir(models) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_dir() {
                return None;
            }
            image_row(path, false)
        })
        .collect()
}

/// The valid image installs in the `image` folder of the library whose
/// canonical root is `root`.
///
/// The library is a drive anyone could have written, so nothing is followed:
/// the `image` folder, each install, its manifest and a kept source folder
/// must be plain folders and files, never links. Otherwise a link planted on
/// the drive would list, and later open, a folder that is not in the library
/// (2.2 SECURITY). The same walk `catalog` makes before it reads or deletes
/// anything in a library.
fn library_image_rows(root: &Path) -> Vec<ImageInstalledRow> {
    let images = root.join("image");
    if !is_plain_dir(&images) {
        return Vec::new();
    }
    let Ok(entries) = std::fs::read_dir(&images) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        // `DirEntry::file_type` reports a link as a link.
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .filter_map(|entry| image_row(entry.path(), true))
        .collect()
}

/// Whether `path` is a folder itself, not a link to one.
fn is_plain_dir(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok_and(|found| found.file_type().is_dir())
}

/// Whether `path` is a file itself, not a link to one.
fn is_plain_file(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok_and(|found| found.file_type().is_file())
}

/// The image install at `path`, when its name and manifest are valid. With
/// `plain`, its manifest and the kept source folder beside it must not be
/// links either (see [`library_image_rows`]).
fn image_row(path: PathBuf, plain: bool) -> Option<ImageInstalledRow> {
    let name = path.file_name()?.to_str()?;
    let alias = name
        .strip_suffix(".image.gturbo")
        .or_else(|| name.strip_suffix(".gturbo"))?;
    if alias.is_empty() {
        return None;
    }
    if plain && !is_plain_file(&path.join(image::install::IMAGE_MANIFEST_NAME)) {
        return None;
    }
    let manifest = image::ImageManifest::load(&path).ok()?;
    manifest.validate().ok()?;
    let source = manifest.source.as_object()?;
    let model_id = source.get("model_id")?.as_str()?.to_string();
    let revision = source.get("model_revision")?.as_str()?.to_string();
    let source_path = if super::image_install::retains_mlx_source_tree(&model_id) {
        // The revision comes from the manifest, which on a drive anyone could
        // have written. A `/` or `..` in it would walk this name out of the
        // folder, so it counts only as one plain path component.
        let kept = format!(".{alias}.{revision}.image-source");
        let one_component = matches!(
            Path::new(&kept).components().collect::<Vec<_>>().as_slice(),
            [std::path::Component::Normal(_)]
        );
        path.parent()
            .filter(|_| one_component && !kept.contains('\0'))
            .map(|parent| parent.join(&kept))
            .filter(|source| {
                if plain {
                    is_plain_dir(source)
                } else {
                    source.is_dir()
                }
            })
    } else {
        None
    };
    Some(ImageInstalledRow {
        alias: alias.to_string(),
        family: catalog::image_family_for_model_id(&model_id).to_string(),
        model_id,
        revision,
        path,
        width: manifest.supported.width,
        height: manifest.supported.height,
        scheduler_steps: manifest.supported.scheduler_steps,
        quantization: image::image_quantization_label(&manifest).ok()?,
        source_path,
    })
}

/// The bytes an install of `alias` will read off the network, so a GUI can
/// show a determinate progress bar and a free-space warning before the walk
/// starts.
pub(crate) fn install_bytes(alias: &str) -> Result<(u64, u64), String> {
    let catalog = Catalog::embedded()?;
    let entry = catalog
        .get(alias)
        .ok_or_else(|| format!("no catalog row named {alias:?}"))?;
    Ok((entry.download_bytes, entry.install_bytes))
}

/// Deletes an installed model from the store and drops its directory.
pub(crate) fn delete(alias: &str) -> Result<(), String> {
    let store = Store::default_store()?;
    delete_from_store(&store, alias)
}

fn delete_from_store(store: &Store, alias: &str) -> Result<(), String> {
    let path = store
        .installed()
        .get(alias)
        .map(|model| model.path.clone())
        .ok_or_else(|| format!("model {alias:?} is not installed"))?;
    if path.exists() {
        std::fs::remove_dir_all(&path)
            .map_err(|e| format!("failed to remove {}: {e}", path.display()))?;
    }
    store.forget(alias)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use catalog::InstalledModel;

    fn temp_root(tag: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!(
            "turbospark-ffi-delete-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::remove_dir_all(&root).ok();
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn image_listing_skips_missing_and_invalid_manifests() {
        let root = temp_root("image-listing");
        let models = root.join("models");
        std::fs::create_dir_all(models.join("not-an-image.gturbo")).unwrap();
        std::fs::create_dir_all(models.join("broken.image.gturbo")).unwrap();
        std::fs::write(
            models.join("broken.image.gturbo/manifest.json"),
            "{\"not\":\"an image manifest\"}",
        )
        .unwrap();

        assert!(image_installed_rows(&models).is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }

    /// A minimal valid source tree: the production builder packs it, so the
    /// scan below runs against a manifest the engine itself wrote. The image
    /// installer's tests materialize it as a downloaded source.
    pub(in crate::models) fn write_synthetic_source(source: &Path) {
        use std::fs;
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
            let header = serde_json::json!({
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
                    serde_json::to_vec(&serde_json::json!({
                        "weight_map": {"linear.weight": file, "norm.weight": file}
                    }))
                    .unwrap(),
                )
                .unwrap();
            }
        }
    }

    fn build_synthetic_install(root: &Path, alias: &str) -> PathBuf {
        build_synthetic_install_for(root, alias, "Tongyi-MAI/Z-Image-Turbo")
    }

    fn build_synthetic_install_for(root: &Path, alias: &str, model_id: &str) -> PathBuf {
        let source = root.join(format!("source-{alias}"));
        write_synthetic_source(&source);
        let output = root.join(format!("{alias}.gturbo"));
        image::build_image_install(&image::ImageInstallSpec {
            source_root: source.clone(),
            output_root: output.clone(),
            model_id: model_id.to_string(),
            model_revision: "synthetic-revision".to_string(),
        })
        .expect("build synthetic image install");
        std::fs::remove_dir_all(source).unwrap();
        output.canonicalize().unwrap()
    }

    fn scanned_paths(root: &Path) -> Vec<PathBuf> {
        let json = image_installed_in_json(root).expect("scan");
        let rows: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
        rows.iter()
            .map(|row| PathBuf::from(row["path"].as_str().unwrap()))
            .collect()
    }

    #[test]
    fn scanning_a_folder_finds_valid_installs_at_depth_zero_and_one_only() {
        let root = temp_root("image-scan");
        let drive = root.join("drive");
        let nested = drive.join("models");
        let too_deep = nested.join("archive");
        std::fs::create_dir_all(&too_deep).unwrap();
        let top = build_synthetic_install(&drive, "top-level");
        let child = build_synthetic_install(&nested, "one-down");
        let _buried = build_synthetic_install(&too_deep, "two-down");
        // A broken manifest is not an install.
        std::fs::create_dir_all(drive.join("broken.gturbo")).unwrap();
        std::fs::write(drive.join("broken.gturbo/manifest.json"), "{}").unwrap();

        let found = scanned_paths(&drive.canonicalize().unwrap());
        assert!(found.contains(&top), "{found:?}");
        assert!(found.contains(&child), "{found:?}");
        assert_eq!(
            found.len(),
            2,
            "depth two and broken installs stay out: {found:?}"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn scanning_an_install_bundle_directly_returns_just_that_install() {
        let root = temp_root("image-scan-bundle");
        let one = build_synthetic_install(&root, "one");
        let _two = build_synthetic_install(&root, "two");

        assert_eq!(scanned_paths(&one), vec![one.clone()]);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn scanning_reports_the_retained_source_only_when_it_sits_beside_the_install() {
        let root = temp_root("image-scan-source");
        // Z-Image rows from `andrevp/` keep their source; this synthetic row
        // is `Tongyi-MAI/`, which does not, so a stray sibling directory must
        // not be reported as one either.
        let install = build_synthetic_install(&root, "plain");
        std::fs::create_dir_all(root.join(".plain.synthetic-revision.image-source")).unwrap();

        let json = image_installed_in_json(&root.canonicalize().unwrap()).unwrap();
        let rows: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["path"].as_str().unwrap(), install.to_str().unwrap());
        assert!(rows[0]["sourcePath"].is_null(), "{}", rows[0]);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn scanning_reports_the_retained_source_for_a_family_that_keeps_it() {
        let root = temp_root("image-scan-kept");
        // `andrevp/` rows keep their MLX source tree beside the install, the
        // layout an install copied to another drive carries with it.
        let install = build_synthetic_install_for(&root, "kept", "andrevp/Z-Image-Turbo-MLX-4bit");
        let source = root.join(".kept.synthetic-revision.image-source");
        std::fs::create_dir_all(&source).unwrap();

        let json = image_installed_in_json(&root.canonicalize().unwrap()).unwrap();
        let rows: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
        assert_eq!(rows.len(), 1, "{rows:?}");
        assert_eq!(rows[0]["path"].as_str().unwrap(), install.to_str().unwrap());
        assert_eq!(
            rows[0]["sourcePath"].as_str().map(PathBuf::from),
            Some(source.canonicalize().unwrap()),
        );
        // Without the source beside it, the same install reports none.
        std::fs::remove_dir_all(&source).unwrap();
        let json = image_installed_in_json(&root.canonicalize().unwrap()).unwrap();
        let rows: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
        assert!(rows[0]["sourcePath"].is_null(), "{}", rows[0]);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn scanning_a_missing_or_file_path_is_an_error_not_an_empty_list() {
        let root = temp_root("image-scan-missing");
        assert!(image_installed_in_json(&root.join("nope")).is_err());
        let file = root.join("file");
        std::fs::write(&file, b"x").unwrap();
        assert!(image_installed_in_json(&file).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    fn image_row(alias: &str, path: &str) -> ImageInstalledRow {
        ImageInstalledRow {
            alias: alias.to_string(),
            model_id: "owner/image".to_string(),
            revision: "main".to_string(),
            path: PathBuf::from(path),
            width: 1024,
            height: 1024,
            scheduler_steps: 9,
            quantization: "test".to_string(),
            source_path: None,
            family: "z-image".to_string(),
        }
    }

    #[test]
    fn image_delete_path_preserves_identity_across_duplicate_aliases() {
        let canonical = image_row("duplicate", "/models/image/duplicate.gturbo");
        let legacy = image_row("duplicate", "/models/duplicate.image.gturbo");

        let selected =
            image_row_for_delete(vec![canonical, legacy], "/models/duplicate.image.gturbo")
                .unwrap();

        assert_eq!(selected.path, Path::new("/models/duplicate.image.gturbo"));
    }

    #[test]
    fn image_delete_rejects_an_ambiguous_alias() {
        let rows = vec![
            image_row("duplicate", "/models/image/duplicate.gturbo"),
            image_row("duplicate", "/models/duplicate.image.gturbo"),
        ];

        let error = image_row_for_delete(rows, "duplicate").unwrap_err();

        assert!(error.contains("ambiguous"), "got {error:?}");
    }

    fn record(store: &Store, alias: &str, path: &std::path::Path) {
        store
            .record(&InstalledModel {
                alias: alias.to_string(),
                repo: "owner/name".to_string(),
                revision: "main".to_string(),
                path: path.to_path_buf(),
                family: "llama".to_string(),
                install_bytes: 1,
                installed_on: "2026-09-14".to_string(),
                status: "runs".to_string(),
                kind: None,
                modality: ModelModality::Text,
                variant: None,
                library: None,
                relative_path: None,
            })
            .unwrap();
    }

    #[test]
    fn delete_removes_a_recorded_install() {
        let root = temp_root("recorded");
        let store = Store::new(&root);
        let install = root.join("models/model.gturbo");
        std::fs::create_dir_all(&install).unwrap();
        record(&store, "model", &install);

        delete_from_store(&store, "model").unwrap();

        assert!(!install.exists());
        assert!(store.installed().is_empty());
    }

    #[test]
    fn delete_rejects_an_existing_unrecorded_directory() {
        let root = temp_root("unrecorded");
        let store = Store::new(root.join("store"));
        let victim = root.join("Documents");
        std::fs::create_dir_all(&victim).unwrap();
        let sentinel = victim.join("valuable.txt");
        std::fs::write(&sentinel, "keep").unwrap();

        let error = delete_from_store(&store, victim.to_str().unwrap()).unwrap_err();

        assert!(error.contains("not installed"), "got {error:?}");
        assert!(sentinel.is_file(), "unrecorded directory was modified");
    }
}
