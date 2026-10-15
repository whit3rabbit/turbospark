//! Curated, pinned MiniMax Music 3 checkpoint profiles and their install path.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::install::{install_destination, record_into, InstallKind};
use crate::library::LocationEnv;
use crate::{
    AudioCatalog, CancelFlag, Catalog, Client, DownloadReceipt, HubClient, HubDownloadProgress,
    HubError, InstalledModel, Machine, ModelModality, PinnedArtifactPlan, PinnedSourceFile,
    PinnedSourceGroup, RepoRef, Store, StoreError,
};

const EMBEDDED: &str = include_str!("music_models.json");
const SOURCE_ROLE: &str = "minimax_music3";

/// The quantization profiles published for MiniMax Music 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum MusicQuantization {
    #[serde(rename = "bf16")]
    Bf16,
    #[serde(rename = "affine-8bit")]
    Affine8Bit,
    #[serde(rename = "affine-6bit")]
    Affine6Bit,
    #[serde(rename = "affine-4bit")]
    Affine4Bit,
    #[serde(rename = "mxfp8")]
    Mxfp8,
    #[serde(rename = "mxfp4")]
    Mxfp4,
    #[serde(rename = "nvfp4")]
    Nvfp4,
}

impl MusicQuantization {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Affine8Bit => "affine-8bit",
            Self::Affine6Bit => "affine-6bit",
            Self::Affine4Bit => "affine-4bit",
            Self::Mxfp8 => "mxfp8",
            Self::Mxfp4 => "mxfp4",
            Self::Nvfp4 => "nvfp4",
        }
    }
}

/// One immutable Hugging Face profile accepted by the audio installer.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicCatalogEntry {
    pub alias: String,
    pub model_id: String,
    pub revision: String,
    pub quantization: MusicQuantization,
    pub recommended: bool,
    pub experimental: bool,
    pub required_files: Vec<String>,
}

/// Parsed and validated list of only the supported music profiles.
#[derive(Debug, Clone)]
pub struct MusicCatalog {
    entries: Vec<MusicCatalogEntry>,
}

impl MusicCatalog {
    pub fn embedded() -> Result<Self, String> {
        let entries: Vec<MusicCatalogEntry> = serde_json::from_str(EMBEDDED)
            .map_err(|error| format!("parsing embedded MiniMax Music 3 catalog: {error}"))?;
        if entries.len() != 7 {
            return Err(format!(
                "MiniMax Music 3 catalog has {} rows; expected all seven profiles",
                entries.len()
            ));
        }
        let mut aliases = HashSet::new();
        for entry in &entries {
            validate_entry(entry)?;
            if !aliases.insert(entry.alias.as_str()) {
                return Err(format!("duplicate MiniMax Music 3 alias {:?}", entry.alias));
            }
        }
        if !entries
            .iter()
            .any(|entry| entry.quantization == MusicQuantization::Mxfp8 && entry.recommended)
        {
            return Err("the MiniMax Music 3 MXFP8 profile must be recommended".to_string());
        }
        Ok(Self { entries })
    }

    pub fn entries(&self) -> impl Iterator<Item = &MusicCatalogEntry> {
        self.entries.iter()
    }

    pub fn get(&self, alias: &str) -> Option<&MusicCatalogEntry> {
        self.entries.iter().find(|entry| entry.alias == alias)
    }

    /// Convert a selected curated row to the shared exact-file pinned plan.
    pub fn pinned_plan(&self, alias: &str) -> Result<PinnedArtifactPlan, String> {
        let entry = self
            .get(alias)
            .ok_or_else(|| format!("no MiniMax Music 3 profile named {alias:?}"))?;
        let repo = RepoRef::new(&entry.model_id, &entry.revision);
        Ok(PinnedArtifactPlan {
            owner_id: format!("music:{alias}"),
            sources: vec![PinnedSourceGroup {
                role: SOURCE_ROLE.to_string(),
                repo,
                files: entry
                    .required_files
                    .iter()
                    .map(|path| PinnedSourceFile {
                        path: path.clone(),
                        expected_size: None,
                        expected_sha256: None,
                    })
                    .collect(),
            }],
        })
    }
}

fn validate_entry(entry: &MusicCatalogEntry) -> Result<(), String> {
    if entry.alias.is_empty()
        || !entry
            .alias
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        return Err(format!("invalid MiniMax Music 3 alias {:?}", entry.alias));
    }
    if entry.model_id
        != format!(
            "mlx-community/MiniMax-Music3-{}",
            entry.quantization_repo_suffix()
        )
    {
        return Err(format!(
            "{} points to an unapproved MiniMax Music 3 repository {}",
            entry.alias, entry.model_id
        ));
    }
    if entry.revision.len() != 40 || !entry.revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!(
            "{} revision is not an immutable 40-hex commit",
            entry.alias
        ));
    }
    if entry.experimental
        != matches!(
            entry.quantization,
            MusicQuantization::Mxfp4 | MusicQuantization::Nvfp4
        )
    {
        return Err(format!(
            "{} has an incorrect experimental-profile marker",
            entry.alias
        ));
    }
    if entry.recommended != (entry.quantization == MusicQuantization::Mxfp8) {
        return Err(format!(
            "{} has an incorrect recommended-profile marker",
            entry.alias
        ));
    }
    if entry.required_files.is_empty() {
        return Err(format!("{} has no required files", entry.alias));
    }
    let mut files = HashSet::new();
    for path in &entry.required_files {
        let safe = !path.is_empty()
            && !Path::new(path).is_absolute()
            && Path::new(path)
                .components()
                .all(|part| matches!(part, std::path::Component::Normal(_)));
        if !safe || !files.insert(path.as_str()) {
            return Err(format!(
                "{} has an unsafe or duplicate path {path:?}",
                entry.alias
            ));
        }
    }
    for required in [
        "LICENSE",
        "config.json",
        "model.safetensors.index.json",
        "scheduler/scheduler_config.json",
        "tokenizer/tokenizer.json",
    ] {
        if !files.contains(required) {
            return Err(format!(
                "{} is missing required file {required}",
                entry.alias
            ));
        }
    }
    if !entry
        .required_files
        .iter()
        .any(|path| path.starts_with("model-") && path.ends_with(".safetensors"))
    {
        return Err(format!("{} has no safetensors shard", entry.alias));
    }
    Ok(())
}

impl MusicCatalogEntry {
    fn quantization_repo_suffix(&self) -> &'static str {
        match self.quantization {
            MusicQuantization::Bf16 => "bf16",
            MusicQuantization::Affine8Bit => "8bit",
            MusicQuantization::Affine6Bit => "6bit",
            MusicQuantization::Affine4Bit => "4bit",
            MusicQuantization::Mxfp8 => "mxfp8",
            MusicQuantization::Mxfp4 => "mxfp4",
            MusicQuantization::Nvfp4 => "nvfp4",
        }
    }
}

pub fn embedded_music_entries() -> Result<Vec<MusicCatalogEntry>, String> {
    Ok(MusicCatalog::embedded()?.entries)
}

pub fn embedded_music_entry(alias: &str) -> Result<MusicCatalogEntry, String> {
    MusicCatalog::embedded()?
        .get(alias)
        .cloned()
        .ok_or_else(|| format!("no MiniMax Music 3 profile named {alias:?}"))
}

/// Download and atomically install one approved profile into This Mac's
/// audio store. The source revision and exact file list come from the
/// embedded catalog.
pub fn install_music(
    client: &Client,
    catalog: &Catalog,
    store: &Store,
    alias: &str,
    progress: &mut dyn FnMut(HubDownloadProgress),
    cancel: &CancelFlag,
) -> Result<InstalledModel, String> {
    let env = LocationEnv::system(store.root())?;
    install_music_into(client, catalog, store, alias, None, &env, progress, cancel)
        .map_err(|error| error.to_string())
}

/// [`install_music`] into `library`, This Mac for `None`, with `env`
/// locating it (2.1, 2.6).
///
/// Where the install lands is settled by [`install_destination`] before
/// anything is downloaded, so a library that is not connected or is
/// read-only, or an alias another library holds, fails before the first
/// request and progress callback. Into a library the transfer stages on the
/// library's drive (`Store::staging_parent_for`) and publishes with a
/// same-volume rename (`Store::publish_staged`), while the pinned range
/// cache stays in the home, where a retry into any library finds it. A
/// profile the audio catalog also lists is published into a library with
/// the audio receipt adoption would write, and refused when its bytes are
/// not that profile's pins. A drive that goes away mid-install ends with
/// [`StoreError::LibraryDisconnected`] and no row. This Mac keeps exactly
/// today's staging, paths, and errors.
#[allow(clippy::too_many_arguments)]
pub fn install_music_into(
    client: &Client,
    catalog: &Catalog,
    store: &Store,
    alias: &str,
    library: Option<&str>,
    env: &LocationEnv,
    progress: &mut dyn FnMut(HubDownloadProgress),
    cancel: &CancelFlag,
) -> Result<InstalledModel, StoreError> {
    install_music_with(
        store,
        alias,
        library,
        env,
        AudioCatalog::embedded,
        cancel,
        |plan, staging_parent, cache_parent| {
            HubClient::new(
                client,
                catalog.clone(),
                Machine::default(),
                4096,
                model_io::ExpertCacheSlots::Auto,
            )
            .with_store(store.clone())
            .download_pinned_artifacts_into(
                plan,
                staging_parent,
                cache_parent,
                progress,
                cancel,
            )
        },
    )
}

/// [`install_music_into`] with the audio catalog and the transfer,
/// `download(plan, staging_parent, cache_parent)`, passed in, so tests can
/// pin a profile to fixture bytes and stand in for the Hub. The audio
/// catalog is read only for an install into a library.
pub(crate) fn install_music_with(
    store: &Store,
    alias: &str,
    library: Option<&str>,
    env: &LocationEnv,
    audio_catalog: impl FnOnce() -> Result<AudioCatalog, String>,
    cancel: &CancelFlag,
    download: impl FnOnce(&PinnedArtifactPlan, &Path, &Path) -> Result<DownloadReceipt, HubError>,
) -> Result<InstalledModel, StoreError> {
    let music = MusicCatalog::embedded().map_err(StoreError::Io)?;
    let entry = music
        .get(alias)
        .ok_or_else(|| StoreError::Io(format!("no MiniMax Music 3 profile named {alias:?}")))?;
    let plan = music.pinned_plan(alias).map_err(StoreError::Io)?;
    let target = install_destination(store, library, InstallKind::Audio, alias, env)?;
    let destination = &target.dir;
    if destination.exists() {
        return Err(StoreError::Io(format!(
            "{} already exists; remove it before pulling again",
            destination.display()
        )));
    }
    // A library's root is never created here: with its drive gone that
    // would rebuild it on the disk under the mount point. Publishing makes
    // the audio folder.
    let into_library = target.library_id.is_some();
    if !into_library {
        std::fs::create_dir_all(store.root()).map_err(|error| {
            StoreError::Io(format!(
                "creating audio store {}: {error}",
                store.root().display()
            ))
        })?;
        std::fs::create_dir_all(destination.parent().expect("audio install has parent"))
            .map_err(|error| StoreError::Io(format!("creating audio install parent: {error}")))?;
    }
    // Only This Mac adopts a catalog profile's install that lacks its audio
    // receipt, so into a library the receipt is published with the install.
    // Read before the transfer, so a catalog that fails to parse wastes no
    // download.
    let audio = if into_library {
        Some(audio_catalog().map_err(StoreError::Io)?)
    } else {
        None
    };
    // Classified by the store's one rule, so a step that failed because the
    // drive went away reads as a disconnect. This Mac's are IO errors.
    let failed = |staging: &Path, error: String| {
        store.install_failure(staging, &target.root, &target.library_name, error)
    };
    let staging_parent = store
        .staging_parent_for(&target.root)
        .map_err(|error| failed(&target.root, error.to_string()))?;

    let receipt = download(&plan, &staging_parent, store.root()).map_err(|error| match error {
        HubError::Staging(_) => failed(&staging_parent, error.to_string()),
        other => StoreError::Io(other.to_string()),
    })?;
    if cancel.is_cancelled() {
        let _ = std::fs::remove_dir_all(&receipt.staging_root);
        return Err(StoreError::Io(crate::INSTALL_CANCELLED.to_string()));
    }
    let published = match &audio {
        // Written into the staged tree, so one rename publishes it complete.
        Some(audio) => verified_music_source(&receipt, entry)
            .and_then(|staged| {
                audio.write_transfer_receipt(alias, &receipt.sources[0].files, &staged)?;
                Ok(staged)
            })
            .map_err(|error| failed(&receipt.staging_root, error))
            .and_then(|staged| store.publish_staged(&staged, destination, &target.library_name)),
        None => promote_music_source(&receipt, entry, destination).map_err(StoreError::Io),
    };
    if let Err(error) = published {
        let _ = std::fs::remove_dir_all(&receipt.staging_root);
        return Err(error);
    }
    if let Err(error) = std::fs::remove_dir_all(&receipt.staging_root) {
        // Classified before the published install is removed, which would
        // otherwise read as the drive going away.
        let error = failed(
            &receipt.staging_root,
            format!("removing music staging tree: {error}"),
        );
        let _ = std::fs::remove_dir_all(destination);
        return Err(error);
    }

    let installed = InstalledModel {
        alias: entry.alias.clone(),
        repo: entry.model_id.clone(),
        revision: entry.revision.clone(),
        path: destination.clone(),
        family: "minimax_music3".to_string(),
        install_bytes: crate::directory_bytes(destination),
        installed_on: crate::install::today(),
        status: "unverified".to_string(),
        kind: None,
        modality: ModelModality::Audio,
        variant: Some(entry.quantization.as_str().to_string()),
        library: None,
        relative_path: None,
    };
    let recorded = match record_into(store, &installed, &target, env) {
        Ok(recorded) => recorded,
        Err(error) => {
            let error = failed(
                destination,
                format!("recording MiniMax Music 3 install: {error}"),
            );
            let _ = std::fs::remove_dir_all(destination);
            return Err(error);
        }
    };
    // Best effort: the published install holds these bytes now, and a
    // leftover cache costs disk, not correctness.
    let _ = receipt.discard_download_cache();
    Ok(recorded)
}

/// Hub repository ids are ASCII case-insensitive and the transfer layer
/// returns them lowercased (`hub.rs` compares them the same way); the
/// revision stays an exact immutable pin.
fn same_repository_pin(repo: &str, revision: &str, entry: &MusicCatalogEntry) -> bool {
    repo.eq_ignore_ascii_case(&entry.model_id) && revision == entry.revision
}

fn promote_music_source(
    receipt: &DownloadReceipt,
    entry: &MusicCatalogEntry,
    destination: &Path,
) -> Result<(), String> {
    let staged_root = verified_music_source(receipt, entry)?;
    std::fs::rename(&staged_root, destination).map_err(|error| {
        format!(
            "installing audio profile {}: {error}",
            destination.display()
        )
    })
}

/// The staged source tree of `receipt`, once the receipt is exactly the
/// pinned profile `entry`: owner, source, repository pin, and file set.
fn verified_music_source(
    receipt: &DownloadReceipt,
    entry: &MusicCatalogEntry,
) -> Result<PathBuf, String> {
    if receipt.owner_id != format!("music:{}", entry.alias) {
        return Err("music transfer returned an unexpected owner id".to_string());
    }
    if receipt.sources.len() != 1 {
        return Err("music transfer returned an unexpected source count".to_string());
    }
    let staged = &receipt.sources[0];
    if staged.role != SOURCE_ROLE {
        return Err(format!(
            "music transfer returned unexpected role {:?}",
            staged.role
        ));
    }
    if !same_repository_pin(&staged.repo.repo, &staged.repo.revision, entry) {
        return Err(format!(
            "music transfer returned an unexpected repository pin: got {}@{}, expected {}@{}",
            staged.repo.repo, staged.repo.revision, entry.model_id, entry.revision
        ));
    }
    if staged.files.len() != entry.required_files.len() {
        return Err("music transfer returned an incomplete pinned file set".to_string());
    }
    let staged_root = receipt.staging_root.join("source-0000");
    if !staged_root.is_dir() {
        return Err("music transfer returned a missing staging directory".to_string());
    }
    let mut remaining: HashSet<&str> = entry.required_files.iter().map(String::as_str).collect();
    for file in &staged.files {
        if !remaining.remove(file.path.as_str()) {
            return Err(format!(
                "music transfer returned an unexpected or duplicate file {:?}",
                file.path
            ));
        }
        let expected_path = staged_root.join(&file.path);
        if file.staged_path != expected_path || !file.staged_path.is_file() {
            return Err(format!(
                "music transfer returned an invalid staged path for {:?}",
                file.path
            ));
        }
    }
    if !remaining.is_empty() {
        return Err(format!(
            "music transfer omitted pinned files: {}",
            remaining.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
    Ok(staged_root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{VerifiedSourceFile, VerifiedSourceGroup};

    #[test]
    fn bundled_music_profiles_are_exactly_the_seven_approved_pins() {
        let catalog = MusicCatalog::embedded().unwrap();
        let entries: Vec<_> = catalog.entries().collect();
        assert_eq!(entries.len(), 7);
        assert_eq!(
            entries
                .iter()
                .filter(|entry| entry.recommended)
                .map(|entry| entry.quantization)
                .collect::<Vec<_>>(),
            vec![MusicQuantization::Mxfp8]
        );
        assert_eq!(
            entries
                .iter()
                .filter(|entry| entry.experimental)
                .map(|entry| entry.quantization)
                .collect::<Vec<_>>(),
            vec![MusicQuantization::Mxfp4, MusicQuantization::Nvfp4]
        );
        for entry in entries {
            assert_eq!(entry.revision.len(), 40, "{}", entry.alias);
            let plan = catalog.pinned_plan(&entry.alias).unwrap();
            assert_eq!(plan.sources[0].repo.repo, entry.model_id);
            assert_eq!(plan.sources[0].repo.revision, entry.revision);
            assert_eq!(
                plan.sources[0]
                    .files
                    .iter()
                    .map(|file| file.path.as_str())
                    .collect::<Vec<_>>(),
                entry
                    .required_files
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
            );
        }
        assert!(catalog.pinned_plan("owner/arbitrary-model").is_err());
    }

    #[test]
    fn repository_pin_ignores_repo_case_but_not_revision_or_name() {
        let entry = embedded_music_entry("minimax-music3-4bit").unwrap();
        assert!(same_repository_pin(
            &entry.model_id,
            &entry.revision,
            &entry
        ));
        assert!(same_repository_pin(
            &entry.model_id.to_ascii_lowercase(),
            &entry.revision,
            &entry
        ));
        let mut other_revision = entry.revision.clone();
        other_revision.replace_range(
            ..1,
            if entry.revision.starts_with('0') {
                "1"
            } else {
                "0"
            },
        );
        assert!(!same_repository_pin(
            &entry.model_id,
            &other_revision,
            &entry
        ));
        assert!(!same_repository_pin(
            "mlx-community/MiniMax-Music3-8bit",
            &entry.revision,
            &entry
        ));
    }

    #[test]
    fn catalog_rejects_foreign_repositories_and_experimental_mislabels() {
        let mut foreign = embedded_music_entry("minimax-music3-mxfp8").unwrap();
        foreign.model_id = "someone/another-model".to_string();
        assert!(validate_entry(&foreign).is_err());

        let mut mislabeled = embedded_music_entry("minimax-music3-nvfp4").unwrap();
        mislabeled.experimental = false;
        assert!(validate_entry(&mislabeled).is_err());
    }

    #[test]
    fn pre_cancelled_pull_leaves_no_install_or_transfer_stage() {
        let root = std::env::temp_dir().join(format!(
            "catalog-minimax-music3-cancel-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let store = Store::new(&root);
        let client = Client::new();
        let catalog = Catalog::embedded().unwrap();
        let cancel = CancelFlag::new();
        cancel.cancel();
        let result = install_music(
            &client,
            &catalog,
            &store,
            "minimax-music3-mxfp8",
            &mut |_| {},
            &cancel,
        );
        assert!(result.unwrap_err().contains("cancelled"));
        assert!(!store.audio_install_path("minimax-music3-mxfp8").exists());
        let residue: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.starts_with(".turbospark-pinned-transfer-"))
            .collect();
        assert!(residue.is_empty(), "unexpected staging residue {residue:?}");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn verified_music_receipt_requires_the_exact_pin_and_promotes_atomically() {
        let root = std::env::temp_dir().join(format!(
            "catalog-minimax-music3-promote-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let entry = embedded_music_entry("minimax-music3-mxfp8").unwrap();
        let staging_root = root.join(".turbospark-pinned-transfer-test");
        let source_root = staging_root.join("source-0000");
        let mut files = Vec::new();
        for path in entry.required_files.iter().rev() {
            let staged_path = source_root.join(path);
            std::fs::create_dir_all(staged_path.parent().unwrap()).unwrap();
            std::fs::write(&staged_path, b"fixture").unwrap();
            files.push(VerifiedSourceFile {
                path: path.clone(),
                staged_path,
                size: 7,
                sha256: "fixture-digest".to_string(),
            });
        }
        let receipt = DownloadReceipt {
            owner_id: format!("music:{}", entry.alias),
            sources: vec![VerifiedSourceGroup {
                role: SOURCE_ROLE.to_string(),
                repo: RepoRef::new(&entry.model_id, &entry.revision),
                files,
            }],
            staging_root: staging_root.clone(),
            download_cache: Vec::new(),
        };
        let destination = root.join("audio").join(format!("{}.gturbo", entry.alias));
        std::fs::create_dir_all(destination.parent().unwrap()).unwrap();
        let mut malformed = receipt.clone();
        malformed.sources[0].files[0].path = "unexpected.json".to_string();
        let rejected_destination = root.join("rejected");
        assert!(promote_music_source(&malformed, &entry, &rejected_destination).is_err());
        assert!(!rejected_destination.exists());

        promote_music_source(&receipt, &entry, &destination).unwrap();
        assert!(destination.join("config.json").is_file());
        assert!(!source_root.exists());
        std::fs::remove_dir_all(&staging_root).unwrap();
        assert!(!staging_root.exists());
        std::fs::remove_dir_all(root).unwrap();
    }
}
