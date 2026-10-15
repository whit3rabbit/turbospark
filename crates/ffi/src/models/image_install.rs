//! Curated image catalog and install entry points for native GUI hosts.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use catalog::library::LocationEnv;
use catalog::StoreError;
use serde::Serialize;

use super::install::ActiveInstall;

#[derive(Serialize)]
struct ImageCatalogRow {
    alias: String,
    #[serde(rename = "modelID")]
    model_id: String,
    revision: String,
    family: String,
    quantization: String,
}

#[derive(Serialize)]
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
    family: String,
}

pub(crate) fn catalog_json() -> Result<String, String> {
    let catalog = catalog::ImageCatalog::embedded()?;
    let rows = catalog
        .entries()
        .map(|entry| ImageCatalogRow {
            alias: entry.alias.clone(),
            model_id: entry.model_id.clone(),
            revision: entry.revision.clone(),
            family: catalog::image_family_for_model_id(&entry.model_id).to_string(),
            quantization: quantization_label(&entry.alias),
        })
        .collect::<Vec<_>>();
    serde_json::to_string(&rows).map_err(|e| e.to_string())
}

/// Downloads and packs one curated image row into the shared image store of
/// `library`, This Mac for `None`. Image installs intentionally do not enter
/// the text `installed.json` index.
pub(crate) fn install(
    alias: &str,
    library: Option<&str>,
    on_stage: impl FnMut(&str),
    on_bytes: Arc<dyn Fn(u64, u64) + Send + Sync>,
) -> Result<String, String> {
    let active = ActiveInstall::register_image();
    let store = catalog::Store::default_store()?;
    let env = LocationEnv::system(store.root())?;
    let client = catalog::Client::new();
    install_with(
        &store,
        library,
        &env,
        alias,
        on_stage,
        on_bytes,
        |repo, source, on_stage, on_bytes| {
            catalog::materialize_image_source(
                &client,
                repo,
                source,
                Some(active.cancel_flag()),
                on_stage,
                on_bytes,
            )
        },
    )
    .map_err(|error| error.to_string())
}

/// [`install`] with its store and library environment given, and with
/// `materialize` fetching the source tree into the folder it is given, which
/// is the network download in [`install`] and a local fixture in tests.
///
/// Where the install lands is settled by `catalog::install_destination`
/// before `materialize` runs, so a library that is not connected or is
/// read-only, or an alias another library holds (its image folders count,
/// as image installs are not recorded), fails before any download or
/// callback.
///
/// This Mac keeps exactly today's layout: the source tree under
/// `models/image/.<alias>.<revision>.image-source` and the install built in
/// place. Into a library, the source tree and the build stage in the
/// library's staging folder and publish into `<root>/image` with
/// same-volume renames, a kept MLX source tree first, so a listed install
/// never lacks it. A drive that goes away mid-install ends with
/// `StoreError::LibraryDisconnected`, and nothing partial is listed.
fn install_with(
    store: &catalog::Store,
    library: Option<&str>,
    env: &LocationEnv,
    alias: &str,
    mut on_stage: impl FnMut(&str),
    on_bytes: Arc<dyn Fn(u64, u64) + Send + Sync>,
    materialize: impl FnOnce(
        &catalog::RepoRef,
        &Path,
        &mut dyn FnMut(&str),
        Arc<dyn Fn(u64, u64) + Send + Sync>,
    ) -> Result<u64, String>,
) -> Result<String, StoreError> {
    let catalog = catalog::ImageCatalog::embedded().map_err(StoreError::Io)?;
    let entry = catalog
        .get(alias)
        .ok_or_else(|| StoreError::Io(format!("no image catalog row named {alias:?}")))?
        .clone();
    let target =
        catalog::install_destination(store, library, catalog::InstallKind::Image, alias, env)?;
    let output = &target.dir;
    if output.exists() {
        return Err(StoreError::Io(format!(
            "{} already holds an image install",
            output.display()
        )));
    }
    let parent = output
        .parent()
        .ok_or_else(|| StoreError::Io("image install has no parent directory".to_string()))?;
    let source_name = format!(".{alias}.{}.image-source", entry.revision);
    let repo = catalog::RepoRef::new(&entry.model_id, &entry.revision);
    // The Swift MLX runtime consumes the original safetensors tree. Keep it
    // beside the packed install so MLX-backed models do not download a second
    // copy into the Hugging Face cache. The packed install remains available
    // to the native runtime as a fallback. deepsweet rows are mflux
    // conversions whose trees the catalog normalizes for the same purpose,
    // and the Qwen-Image-2.1 row keeps the mlx-community processor layout.
    let retained = retains_mlx_source_tree(&entry.model_id);
    let spec = |source_root: &Path, output_root: &Path| image::ImageInstallSpec {
        source_root: source_root.to_path_buf(),
        output_root: output_root.to_path_buf(),
        model_id: entry.model_id.clone(),
        model_revision: entry.revision.clone(),
    };

    let (path, mlx_source_path) = if target.library_id.is_none() {
        std::fs::create_dir_all(parent).map_err(|e| {
            StoreError::Io(format!(
                "creating image install parent {}: {e}",
                parent.display()
            ))
        })?;
        let source = parent.join(&source_name);
        on_stage("preparing image source");
        let downloaded = materialize(&repo, &source, &mut on_stage, Arc::clone(&on_bytes))
            .map_err(StoreError::Io)?;
        on_stage(&format!("downloaded {downloaded} image source bytes"));
        let report = image::build_image_install_with_progress(&spec(&source, output), |stage| {
            on_stage(stage)
        })
        .map_err(StoreError::Io)?;
        let mlx_source_path = if retained {
            let _ = std::fs::remove_dir_all(source.join(".download-cache"));
            source.canonicalize().ok()
        } else {
            let _ = std::fs::remove_dir_all(&source);
            None
        };
        let path = report
            .output_root
            .canonicalize()
            .unwrap_or(report.output_root);
        (path, mlx_source_path)
    } else {
        let failed = |staging: &Path, error: String| {
            store.install_failure(staging, &target.root, &target.library_name, error)
        };
        let staging = store
            .staging_parent_for(&target.root)
            .map_err(|error| failed(&target.root, error.to_string()))?;
        // A kept source tree whose install was removed by hand would block
        // publishing this one. Refused before downloading, rather than
        // deleting a folder on a drive anyone could have written.
        let kept_source = parent.join(&source_name);
        if retained && std::fs::symlink_metadata(&kept_source).is_ok() {
            return Err(StoreError::Io(format!(
                "{} is left from an earlier install of {alias:?}; remove it and try again",
                kept_source.display()
            )));
        }
        let source = staging.join(&source_name);
        clear_unsafe_leftover(&source)
            .map_err(|error| failed(&staging, format!("clearing {}: {error}", source.display())))?;
        on_stage("preparing image source");
        let downloaded = materialize(&repo, &source, &mut on_stage, Arc::clone(&on_bytes))
            .map_err(|error| failed(&staging, error))?;
        on_stage(&format!("downloaded {downloaded} image source bytes"));
        // Built in staging too, so a partial build never shows in the
        // library. A build an earlier attempt left unpublished is replaced;
        // a link there is removed, not followed.
        let staged = staging.join(format!("{alias}.gturbo"));
        let _ = std::fs::remove_dir_all(&staged);
        let report = image::build_image_install_with_progress(&spec(&source, &staged), |stage| {
            on_stage(stage)
        })
        .map_err(|error| failed(&staging, error))?;
        let published = publish_into_library(
            store,
            &target,
            &source,
            &kept_source,
            retained,
            &report.output_root,
        );
        if published.is_err() {
            let _ = std::fs::remove_dir_all(&report.output_root);
        }
        (output.clone(), published?)
    };
    let manifest = image::ImageManifest::load(&path).map_err(StoreError::Io)?;
    manifest.validate().map_err(StoreError::Io)?;
    serde_json::to_string(&ImageInstalledRow {
        alias: entry.alias,
        family: catalog::image_family_for_model_id(&entry.model_id).to_string(),
        model_id: entry.model_id,
        revision: entry.revision,
        path,
        width: manifest.supported.width,
        height: manifest.supported.height,
        scheduler_steps: manifest.supported.scheduler_steps,
        quantization: image::image_quantization_label(&manifest).map_err(StoreError::Io)?,
        source_path: mlx_source_path,
    })
    .map_err(|e| StoreError::Io(e.to_string()))
}

/// Makes `source`, a fixed name in a library's staging folder, safe to
/// download into. A drive anyone could have written may hold a link there,
/// or a folder from an earlier attempt with a link inside, and writing or
/// publishing through either would put the source tree outside the library
/// (the 2.2 SECURITY note). A link or other non-folder at `source` is
/// removed itself, never followed. A leftover folder is kept, for the resume
/// cache inside it, only when it holds nothing but plain folders and files.
/// Otherwise it is removed whole, which removes the links in it without
/// following them.
fn clear_unsafe_leftover(source: &Path) -> std::io::Result<()> {
    let found = match std::fs::symlink_metadata(source) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        found => found?,
    };
    if !found.file_type().is_dir() {
        return std::fs::remove_file(source);
    }
    let mut folders = vec![source.to_path_buf()];
    while let Some(folder) = folders.pop() {
        for entry in std::fs::read_dir(&folder)? {
            let entry = entry?;
            // Not followed: `file_type` reports a link as a link.
            let kind = entry.file_type()?;
            if kind.is_dir() {
                folders.push(entry.path());
            } else if !kind.is_file() {
                return std::fs::remove_dir_all(source);
            }
        }
    }
    Ok(())
}

/// Publishes a library image install built in staging: the kept MLX source
/// tree to `kept_source` first, when the row keeps one, then the install to
/// `target.dir`. Returns where the kept source went.
fn publish_into_library(
    store: &catalog::Store,
    target: &catalog::InstallTarget,
    source: &Path,
    kept_source: &Path,
    retained: bool,
    staged_install: &Path,
) -> Result<Option<PathBuf>, StoreError> {
    if !retained {
        let _ = std::fs::remove_dir_all(source);
        store.publish_staged(staged_install, &target.dir, &target.library_name)?;
        return Ok(None);
    }
    let _ = std::fs::remove_dir_all(source.join(".download-cache"));
    store.publish_staged(source, kept_source, &target.library_name)?;
    if let Err(error) = store.publish_staged(staged_install, &target.dir, &target.library_name) {
        // Back into staging, so a retry is not refused by this attempt's
        // own leftover source. With the drive gone this fails too, and the
        // drive keeps the leftover.
        let _ = std::fs::rename(kept_source, source);
        return Err(error);
    }
    Ok(Some(kept_source.to_path_buf()))
}

/// Whether an install keeps its original MLX safetensors tree beside the
/// packed install. The install path and the installed-row rescan
/// (`catalog.rs`) must agree, or a row would install with a source tree and
/// then rescan without one. The Turbo prefix is scoped to its repository, not
/// the publisher.
pub(super) fn retains_mlx_source_tree(model_id: &str) -> bool {
    model_id.starts_with("andrevp/")
        || model_id.starts_with("deepsweet/")
        || model_id.starts_with("mlx-community/")
        || model_id.starts_with("baa-ai/Krea-2-Turbo")
        || model_id.starts_with(catalog::QWEN_IMAGE_21_TURBO_PREFIX)
}

fn quantization_label(alias: &str) -> String {
    if alias.ends_with("-fp16") {
        return "unquantized".to_string();
    }
    alias
        .rsplit_once('-')
        .and_then(|(_, bits)| bits.strip_suffix("bit").or_else(|| bits.strip_prefix('q')))
        .and_then(|bits| bits.parse::<u8>().ok())
        .map(|bits| format!("mlx-affine-linear-weights-group-64-bits-{bits}"))
        .unwrap_or_else(|| "unquantized".to_string())
}

#[cfg(test)]
mod tests {
    use super::quantization_label;

    #[test]
    fn image_catalog_labels_match_the_published_widths() {
        assert_eq!(
            quantization_label("z-image-turbo-mlx-2bit"),
            "mlx-affine-linear-weights-group-64-bits-2"
        );
        assert_eq!(quantization_label("z-image-turbo-mlx-fp16"), "unquantized");
        assert_eq!(
            quantization_label("z-image-turbo-mlx-q4"),
            "mlx-affine-linear-weights-group-64-bits-4"
        );
        assert_eq!(
            quantization_label("z-image-mlx-q8"),
            "mlx-affine-linear-weights-group-64-bits-8"
        );
        // The label is the transformer width, matching what the installed
        // manifest reports; the Q8 text encoder is not part of it.
        assert_eq!(
            quantization_label("qwen-image-2.1-turbo-mlx-mixed-4bit"),
            "mlx-affine-linear-weights-group-64-bits-4"
        );
    }

    #[test]
    fn the_retained_source_tree_covers_the_turbo_repository_only() {
        use super::retains_mlx_source_tree;
        assert!(retains_mlx_source_tree(
            "Sawfwair/Qwen-Image-2.1-Turbo-MLX-Mixed-4bit"
        ));
        assert!(!retains_mlx_source_tree("Sawfwair/Something-Else"));
        assert!(retains_mlx_source_tree(
            "mlx-community/Qwen-Image-2.1-MLX-4bit"
        ));
        assert!(!retains_mlx_source_tree("Tongyi-MAI/Z-Image-Turbo"));
    }
}

/// Image installs routed to a library (2.1, 2.6). The download is a local
/// fixture handed in as `materialize`: everything around it, the
/// destination, staging, build, and publication, is the production path.
#[cfg(test)]
mod library_tests {
    use std::cell::Cell;
    use std::os::unix::fs::MetadataExt;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use catalog::library::{parse_mount_table, LibraryRegistry, MountSource, MARKER_FILE};

    use super::super::catalog::tests::write_synthetic_source;
    use super::*;

    /// Kept: `andrevp/` rows keep their MLX source tree beside the install.
    const KEPT: &str = "z-image-turbo-mlx-fp16";
    /// Dropped: `Tongyi-MAI/` rows do not.
    const DROPPED: &str = "z-image-turbo";

    /// A temp home and a library added on the fake volume
    /// `<base>/Volumes/T7`, through a fixed mount table.
    struct Fixture {
        base: PathBuf,
        store: catalog::Store,
        volume: PathBuf,
        env: LocationEnv,
        id: String,
    }

    impl Fixture {
        fn new(tag: &str) -> Self {
            let base = std::env::temp_dir().join(format!(
                "turbospark-ffi-image-library-{tag}-{}",
                std::process::id()
            ));
            std::fs::remove_dir_all(&base).ok();
            std::fs::create_dir_all(&base).unwrap();
            let base = base.canonicalize().unwrap();
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
                // Unlike the temp dir's, so the fake volume is not refused
                // as the home's disk.
                home_device: std::fs::metadata(&base).unwrap().dev().wrapping_add(1),
            };
            let added = LibraryRegistry::load(&home)
                .add(&volume, Some("T7"), &env)
                .unwrap();
            Self {
                base,
                store: catalog::Store::new(&home),
                volume,
                env,
                id: added.view.record.id,
            }
        }

        fn root(&self) -> PathBuf {
            self.volume.join("TurboSpark")
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.base).ok();
        }
    }

    struct Run {
        result: Result<String, StoreError>,
        /// Where `materialize` was asked to put the source, once per call.
        materialized: Vec<PathBuf>,
        stages: usize,
        bytes: usize,
    }

    fn run(fx: &Fixture, library: Option<&str>, alias: &str) -> Run {
        let stages = Cell::new(0);
        let bytes = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&bytes);
        let mut materialized = Vec::new();
        let result = install_with(
            &fx.store,
            library,
            &fx.env,
            alias,
            |_| stages.set(stages.get() + 1),
            Arc::new(move |_, _| {
                counted.fetch_add(1, Ordering::Relaxed);
            }),
            |_, source, on_stage, on_bytes| {
                materialized.push(source.to_path_buf());
                on_stage("downloading fixture");
                write_synthetic_source(source);
                on_bytes(1, 1);
                Ok(1)
            },
        );
        Run {
            result,
            materialized,
            stages: stages.get(),
            bytes: bytes.load(Ordering::Relaxed),
        }
    }

    fn row(json: &str) -> serde_json::Value {
        serde_json::from_str(json).unwrap()
    }

    fn path(value: &serde_json::Value) -> Option<PathBuf> {
        value.as_str().map(PathBuf::from)
    }

    fn revision(alias: &str) -> String {
        catalog::ImageCatalog::embedded()
            .unwrap()
            .get(alias)
            .unwrap()
            .revision
            .clone()
    }

    fn entries(dir: &Path) -> Vec<String> {
        std::fs::read_dir(dir)
            .map(|found| {
                found
                    .flatten()
                    .map(|entry| entry.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn an_image_install_lands_in_the_library_with_its_kept_source_beside_it() {
        let fx = Fixture::new("kept");
        let run = run(&fx, Some(&fx.id), KEPT);
        let installed = row(&run.result.expect("library image install"));

        let root = fx.root();
        let source_name = format!(".{KEPT}.{}.image-source", revision(KEPT));
        // Downloaded into the library's staging folder, not the home.
        assert_eq!(
            run.materialized,
            vec![root.join(".staging").join(&source_name)]
        );
        let install = root.join("image").join(format!("{KEPT}.gturbo"));
        let kept = root.join("image").join(&source_name);
        assert_eq!(path(&installed["path"]), Some(install.clone()));
        assert_eq!(path(&installed["sourcePath"]), Some(kept.clone()));
        assert!(install.join("manifest.json").is_file());
        assert!(kept.is_dir());
        assert!(!kept.join(".download-cache").exists());
        // Staging is empty again, and the home's models folder untouched.
        assert!(entries(&root.join(".staging")).is_empty());
        assert!(!fx.store.models_root().exists());

        // The library's image folder lists it exactly as This Mac's would.
        let listed: Vec<serde_json::Value> = serde_json::from_str(
            &super::super::catalog::image_installed_in_json(&root.join("image")).unwrap(),
        )
        .unwrap();
        assert_eq!(listed.len(), 1, "{listed:?}");
        assert_eq!(listed[0]["path"], installed["path"]);
        assert_eq!(listed[0]["sourcePath"], installed["sourcePath"]);
    }

    #[test]
    fn a_library_image_install_drops_a_source_it_does_not_keep() {
        let fx = Fixture::new("dropped");
        let installed = row(&run(&fx, Some(&fx.id), DROPPED).result.unwrap());
        let root = fx.root();
        assert_eq!(
            path(&installed["path"]),
            Some(root.join("image").join(format!("{DROPPED}.gturbo")))
        );
        assert!(installed["sourcePath"].is_null());
        assert_eq!(
            entries(&root.join("image")),
            vec![format!("{DROPPED}.gturbo")]
        );
        assert!(entries(&root.join(".staging")).is_empty());
    }

    /// `None` is This Mac with today's layout: the source beside the
    /// install under `models/image`, and nothing on any library.
    #[test]
    fn this_mac_image_installs_keep_todays_layout() {
        let fx = Fixture::new("this-mac");
        let run = run(&fx, None, KEPT);
        let installed = row(&run.result.unwrap());
        let image = fx.store.models_root().join("image");
        let source = image.join(format!(".{KEPT}.{}.image-source", revision(KEPT)));
        assert_eq!(run.materialized, vec![source.clone()]);
        assert_eq!(
            path(&installed["path"]),
            Some(fx.store.image_install_path(KEPT))
        );
        assert_eq!(path(&installed["sourcePath"]), Some(source));
        assert!(!fx.root().join(".staging").exists());
        assert!(!fx.root().join("image").exists());
    }

    fn assert_refused_before_the_download(run: Run, expected: StoreError) {
        assert_eq!(run.result, Err(expected));
        assert!(run.materialized.is_empty(), "the source was downloaded");
        assert_eq!(run.stages, 0, "a stage callback fired");
        assert_eq!(run.bytes, 0, "a byte callback fired");
    }

    #[test]
    fn an_unplugged_or_read_only_library_is_refused_before_the_download() {
        let fx = Fixture::new("refused");
        let marker = fx.root().join(MARKER_FILE);
        let mut json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&marker).unwrap()).unwrap();
        json["format"] = serde_json::json!(2);
        std::fs::write(&marker, serde_json::to_vec_pretty(&json).unwrap()).unwrap();
        assert_refused_before_the_download(
            run(&fx, Some(&fx.id), KEPT),
            StoreError::LibraryReadOnly {
                library: "T7".to_string(),
            },
        );
        assert!(!fx.root().join(".staging").exists());

        std::fs::remove_dir_all(&fx.volume).unwrap();
        assert_refused_before_the_download(
            run(&fx, Some(&fx.id), KEPT),
            StoreError::NotConnected {
                library: "T7".to_string(),
            },
        );
        assert!(!fx.volume.exists(), "the unplugged volume was recreated");
    }

    /// The message of a refused run, which must name `place`.
    fn refusal_naming(run: &Run, place: &str) -> StoreError {
        match &run.result {
            Err(StoreError::Io(message)) if message.contains(place) => {
                StoreError::Io(message.clone())
            }
            other => panic!("expected a refusal naming {place}, got {other:?}"),
        }
    }

    /// Image installs are not recorded, so an alias is taken by its folder:
    /// one in This Mac refuses a library install and the other way round.
    #[test]
    fn an_image_alias_in_another_library_is_refused_before_the_download() {
        let fx = Fixture::new("elsewhere");
        std::fs::create_dir_all(fx.store.image_install_path(KEPT)).unwrap();
        let into_library = run(&fx, Some(&fx.id), KEPT);
        let expected = refusal_naming(&into_library, "This Mac");
        assert_refused_before_the_download(into_library, expected);

        let theirs = fx.root().join("image").join(format!("{DROPPED}.gturbo"));
        std::fs::create_dir_all(theirs).unwrap();
        let into_this_mac = run(&fx, None, DROPPED);
        let expected = refusal_naming(&into_this_mac, "T7");
        assert_refused_before_the_download(into_this_mac, expected);
    }

    /// An install that appears at the destination while this one downloads
    /// stops this one at publication, and the kept source it had already
    /// published goes back to staging, so the retry is not refused by it.
    #[test]
    fn a_failed_publication_takes_its_kept_source_back_to_staging() {
        let fx = Fixture::new("publish-race");
        let root = fx.root();
        let install = root.join("image").join(format!("{KEPT}.gturbo"));
        let source_name = format!(".{KEPT}.{}.image-source", revision(KEPT));
        let result = install_with(
            &fx.store,
            Some(&fx.id),
            &fx.env,
            KEPT,
            |_| {},
            Arc::new(|_, _| {}),
            |_, source, _, _| {
                write_synthetic_source(source);
                std::fs::create_dir_all(&install).unwrap();
                Ok(1)
            },
        );
        assert!(
            matches!(&result, Err(StoreError::Io(message)) if message.contains("already exists")),
            "{result:?}"
        );
        assert!(!root.join("image").join(&source_name).exists());
        assert!(root.join(".staging").join(&source_name).is_dir());
        assert_eq!(
            entries(&root.join(".staging")),
            vec![source_name.clone()],
            "the staged build was left behind"
        );

        std::fs::remove_dir(&install).unwrap();
        let retried = run(&fx, Some(&fx.id), KEPT);
        assert!(retried.result.is_ok(), "{:?}", retried.result);
    }

    /// A folder outside the library with one file in it, the target of the
    /// links a drive's owner plants in the tests below.
    fn outside(fx: &Fixture) -> PathBuf {
        let outside = fx.base.join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("sentinel"), b"keep").unwrap();
        outside
    }

    /// Every link anywhere under `dir`, found without following any.
    fn links_under(dir: &Path) -> Vec<PathBuf> {
        let mut links = Vec::new();
        let mut folders = vec![dir.to_path_buf()];
        while let Some(folder) = folders.pop() {
            for entry in std::fs::read_dir(&folder).unwrap().flatten() {
                let kind = entry.file_type().unwrap();
                if kind.is_symlink() {
                    links.push(entry.path());
                } else if kind.is_dir() {
                    folders.push(entry.path());
                }
            }
        }
        links
    }

    /// Nothing was written through a planted link, none was published, and
    /// what the install reports is inside the library.
    fn assert_contained(fx: &Fixture, outside: &Path, run: Run) {
        let installed = row(&run.result.expect("the install proceeds"));
        assert_eq!(entries(outside), vec!["sentinel".to_string()]);
        assert_eq!(std::fs::read(outside.join("sentinel")).unwrap(), b"keep");
        let root = fx.root();
        assert_eq!(links_under(&root), Vec::<PathBuf>::new());
        for key in ["path", "sourcePath"] {
            let found = path(&installed[key]).unwrap();
            assert!(found.starts_with(&root), "{key} {}", found.display());
            let metadata = std::fs::symlink_metadata(&found).unwrap();
            assert!(metadata.file_type().is_dir(), "{key} is not a folder");
        }
    }

    /// The staged source has a fixed name, so a drive's owner can plant a
    /// link there. It is removed, never written or published through.
    #[test]
    fn a_link_planted_at_the_staged_source_is_removed_not_followed() {
        let fx = Fixture::new("planted-link");
        let outside = outside(&fx);
        let staging = fx.store.staging_parent_for(&fx.root()).unwrap();
        let source_name = format!(".{KEPT}.{}.image-source", revision(KEPT));
        std::os::unix::fs::symlink(&outside, staging.join(&source_name)).unwrap();

        let run = run(&fx, Some(&fx.id), KEPT);
        assert_contained(&fx, &outside, run);
    }

    /// A leftover source folder with a link inside is cleared before the
    /// download, so nothing is written through the link.
    #[test]
    fn a_link_inside_a_leftover_staged_source_is_removed_not_followed() {
        let fx = Fixture::new("nested-link");
        let outside = outside(&fx);
        let staging = fx.store.staging_parent_for(&fx.root()).unwrap();
        let leftover = staging.join(format!(".{KEPT}.{}.image-source", revision(KEPT)));
        std::fs::create_dir_all(leftover.join("tokenizer")).unwrap();
        std::fs::write(leftover.join("tokenizer/tokenizer.json"), b"old").unwrap();
        std::os::unix::fs::symlink(&outside, leftover.join("transformer")).unwrap();

        let run = run(&fx, Some(&fx.id), KEPT);
        assert_contained(&fx, &outside, run);
    }

    /// The only link sits two folders down, so a walk that stopped at the
    /// leftover's top level would keep it and publish the link.
    #[test]
    fn a_link_deep_inside_a_leftover_staged_source_is_removed_not_followed() {
        let fx = Fixture::new("deep-link");
        let outside = outside(&fx);
        let staging = fx.store.staging_parent_for(&fx.root()).unwrap();
        let leftover = staging.join(format!(".{KEPT}.{}.image-source", revision(KEPT)));
        std::fs::create_dir_all(leftover.join("tokenizer")).unwrap();
        std::fs::write(leftover.join("tokenizer/tokenizer.json"), b"old").unwrap();
        std::os::unix::fs::symlink(&outside, leftover.join("tokenizer/nested")).unwrap();

        let run = run(&fx, Some(&fx.id), KEPT);
        assert_contained(&fx, &outside, run);
    }

    /// A kept source left beside a removed install would block publishing,
    /// so it is refused up front rather than deleted on the drive.
    #[test]
    fn a_leftover_kept_source_is_refused_before_the_download() {
        let fx = Fixture::new("leftover");
        let leftover = fx
            .root()
            .join("image")
            .join(format!(".{KEPT}.{}.image-source", revision(KEPT)));
        std::fs::create_dir_all(leftover.join("transformer")).unwrap();
        let run = run(&fx, Some(&fx.id), KEPT);
        let expected = refusal_naming(&run, "left from an earlier install");
        assert_refused_before_the_download(run, expected);
        assert!(
            leftover.join("transformer").is_dir(),
            "the leftover was touched"
        );
    }
}
