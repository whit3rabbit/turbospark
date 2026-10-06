//! Task-aware pinned audio profiles and verified managed installs.

use crate::{
    CancelFlag, Catalog, Client, DownloadReceipt, HubDownloadProgress, InstalledModel,
    PinnedArtifactPlan, Store,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioTask {
    SpeechToText,
    TextToSpeech,
    Music,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioProfileIdentity {
    pub task: AudioTask,
    pub alias: String,
    pub repository: String,
    pub revision: String,
    pub asset_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioAsset {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioFrontendProvenance {
    pub repository: String,
    pub revision: String,
    pub license: String,
    pub license_path: String,
    pub license_size: u64,
    pub license_sha256: String,
    pub bundled_license_path: String,
    pub bundled_license_size: u64,
    pub bundled_license_sha256: String,
    pub resources: Vec<AudioFrontendResource>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioFrontendResource {
    pub repository: String,
    pub revision: String,
    pub path: String,
    pub size: u64,
    pub sha256: String,
    pub license: String,
    pub bundled_path: String,
    pub bundled_size: u64,
    pub bundled_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioCapabilities {
    pub languages: Vec<String>,
    pub voices: Vec<String>,
    pub max_input_seconds: Option<u32>,
    pub supports_lyrics: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioPcmFormat {
    pub sample_rate: u32,
    pub channels: u32,
    pub interleaved: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioReadiness {
    ImplementedUnqualified,
    RuntimePending,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioProfile {
    pub identity: AudioProfileIdentity,
    pub family: String,
    pub display_name: String,
    pub capabilities: AudioCapabilities,
    pub pcm_format: AudioPcmFormat,
    pub download_bytes: u64,
    pub resident_memory_bytes: u64,
    pub resident_memory_evidence: String,
    pub readiness: AudioReadiness,
    pub evidence: Vec<String>,
    pub assets: Vec<AudioAsset>,
    pub frontend: Option<AudioFrontendProvenance>,
}

#[derive(Debug, Clone)]
pub struct AudioCatalog {
    entries: Vec<AudioProfile>,
}
const ASSETS: &str = include_str!("audio_assets.json");
const RECEIPT_FILENAME: &str = "audio-receipt.json";

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioInstalledReport {
    pub installed: Vec<InstalledModel>,
    pub needs_adoption: Vec<AudioLegacyInstall>,
    pub incompatible: Vec<AudioInstalledRecordError>,
    pub other_audio: Vec<InstalledModel>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioInstalledRecordError {
    pub record: InstalledModel,
    pub identity: Option<AudioProfileIdentity>,
    pub error: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioLegacyInstall {
    pub record: InstalledModel,
    pub identity: AudioProfileIdentity,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AssetManifest {
    schema_version: u32,
    profiles: Vec<ManifestProfile>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestProfile {
    alias: String,
    assets: Vec<AudioAsset>,
    source: Option<ManifestSource>,
    frontend: Option<AudioFrontendProvenance>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestSource {
    repository: String,
    revision: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AudioInstallReceipt {
    schema_version: u32,
    identity: AudioProfileIdentity,
    family: String,
    files: std::collections::BTreeMap<String, model_io::speech_receipt::FileEntry>,
}

impl AudioTask {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SpeechToText => "speech_to_text",
            Self::TextToSpeech => "text_to_speech",
            Self::Music => "music",
        }
    }
}

impl AudioCatalog {
    pub fn embedded() -> Result<Self, String> {
        Self::from_manifest(ASSETS)
    }

    fn from_manifest(json: &str) -> Result<Self, String> {
        let manifest: AssetManifest =
            serde_json::from_str(json).map_err(|e| format!("invalid audio asset manifest: {e}"))?;
        if manifest.schema_version != 1 || manifest.profiles.len() != 3 {
            return Err("audio manifest must have schema 1 and exactly three profiles".into());
        }
        let mut entries = Vec::new();
        let mut aliases = std::collections::HashSet::new();
        for row in manifest.profiles {
            if !aliases.insert(row.alias.clone()) {
                return Err(format!("duplicate audio alias {:?}", row.alias));
            }
            validate_assets(&row.assets)?;
            let (
                task,
                repository,
                revision,
                family,
                display_name,
                required,
                rate,
                channels,
                capabilities,
                readiness,
            ) = match row.alias.as_str() {
                "whisper-base" => {
                    if row.source.is_some() || row.frontend.is_some() {
                        return Err("Whisper source must come from the speech catalog".into());
                    }
                    let speech = crate::embedded_speech_entry(&row.alias)?;
                    (
                        AudioTask::SpeechToText,
                        speech.model_id,
                        speech.revision,
                        speech.format.speech_family().as_str(),
                        "Whisper Base",
                        speech.required_files,
                        16000,
                        1,
                        AudioCapabilities {
                            languages: vec!["auto".into(), "multilingual".into()],
                            voices: vec![],
                            max_input_seconds: Some(1800),
                            supports_lyrics: false,
                        },
                        AudioReadiness::ImplementedUnqualified,
                    )
                }
                "kokoro-82m-bf16" => {
                    let source = row.source.ok_or("Kokoro is missing its source pin")?;
                    let frontend = row
                        .frontend
                        .as_ref()
                        .ok_or("Kokoro is missing frontend provenance")?;
                    validate_frontend(frontend)?;
                    (
                        AudioTask::TextToSpeech,
                        source.repository,
                        source.revision,
                        "kokoro",
                        "Kokoro 82M BF16",
                        vec![
                            "README.md".into(),
                            "config.json".into(),
                            "kokoro-v1_0.safetensors".into(),
                            "voices/af_heart.safetensors".into(),
                        ],
                        24000,
                        1,
                        AudioCapabilities {
                            languages: vec!["en-US".into()],
                            voices: vec!["af_heart".into()],
                            max_input_seconds: None,
                            supports_lyrics: false,
                        },
                        AudioReadiness::RuntimePending,
                    )
                }
                "minimax-music3-4bit" => {
                    if row.source.is_some() || row.frontend.is_some() {
                        return Err("MiniMax source must come from the music catalog".into());
                    }
                    let music = crate::embedded_music_entry(&row.alias)?;
                    (
                        AudioTask::Music,
                        music.model_id,
                        music.revision,
                        "minimax_music3",
                        "MiniMax Music 3 4-bit",
                        music.required_files,
                        44100,
                        2,
                        AudioCapabilities {
                            languages: vec![],
                            voices: vec![],
                            max_input_seconds: Some(360),
                            supports_lyrics: true,
                        },
                        AudioReadiness::RuntimePending,
                    )
                }
                alias => return Err(format!("unapproved audio profile {alias:?}")),
            };
            if !immutable_revision(&revision) {
                return Err(format!("{} does not have an immutable revision", row.alias));
            }
            for name in &required {
                if !row.assets.iter().any(|a| &a.path == name) {
                    return Err(format!("{} is missing asset {name}", row.alias));
                }
            }
            // Exact graphs prevent unrelated or obsolete assets from becoming install requirements.
            if row.assets.len() != required.len() {
                return Err(format!("{} has an unexpected asset graph", row.alias));
            }
            let download_bytes = row
                .assets
                .iter()
                .try_fold(0u64, |n, a| n.checked_add(a.size))
                .ok_or("audio download size overflow")?;
            let weight_bytes = row
                .assets
                .iter()
                .filter(|a| a.path.ends_with(".safetensors") && !a.path.starts_with("voices/"))
                .try_fold(0u64, |n, a| n.checked_add(a.size))
                .ok_or("audio weight size overflow")?;
            let (resident_memory_bytes,resident_memory_evidence)=match task {
                AudioTask::SpeechToText => (weight_bytes.saturating_mul(2).saturating_add(16_000*1800*4).saturating_add(256<<20),"estimate: twice checkpoint weights, 30-minute PCM, and 256 MiB working reserve"),
                AudioTask::TextToSpeech => (weight_bytes.saturating_mul(2).saturating_add(256<<20),"estimate: twice checkpoint weights and 256 MiB voice/working reserve"),
                AudioTask::Music => (weight_bytes.saturating_add(2<<30),"estimate: packed checkpoint weights and 2 GiB working reserve; full duration peak unqualified"),
            };
            let identity = AudioProfileIdentity {
                task,
                alias: row.alias,
                repository,
                revision,
                asset_fingerprint: asset_fingerprint(&row.assets, &row.frontend)?,
            };
            entries.push(AudioProfile {identity,family:family.into(),display_name:display_name.into(),capabilities,pcm_format:AudioPcmFormat {sample_rate:rate,channels,interleaved:true},download_bytes,resident_memory_bytes,resident_memory_evidence:resident_memory_evidence.into(),readiness,evidence:vec!["Immutable source asset plan; runtime/reference and measured resident gates remain unqualified".into()],assets:row.assets,frontend:row.frontend});
        }
        Ok(Self { entries })
    }

    pub fn entries(&self) -> impl Iterator<Item = &AudioProfile> {
        self.entries.iter()
    }
    pub fn get(&self, alias: &str) -> Option<&AudioProfile> {
        self.entries.iter().find(|p| p.identity.alias == alias)
    }

    /// Compare every identity field before dispatch; a saved selection never upgrades its pin.
    pub fn profile(&self, identity: &AudioProfileIdentity) -> Result<&AudioProfile, String> {
        self.get(&identity.alias).filter(|p|p.identity==*identity).ok_or_else(||format!("obsolete or unknown audio profile identity for {:?}; select the current catalog profile",identity.alias))
    }

    pub fn pinned_plan(
        &self,
        identity: &AudioProfileIdentity,
    ) -> Result<PinnedArtifactPlan, String> {
        let p = self.profile(identity)?;
        Ok(PinnedArtifactPlan {
            owner_id: format!(
                "audio:{}:{}:{}",
                identity.task.as_str(),
                identity.alias,
                identity.asset_fingerprint
            ),
            sources: vec![crate::PinnedSourceGroup {
                role: p.family.clone(),
                repo: crate::RepoRef::new(&identity.repository, &identity.revision),
                files: p
                    .assets
                    .iter()
                    .map(|a| crate::PinnedSourceFile {
                        path: a.path.clone(),
                        expected_size: Some(a.size),
                        expected_sha256: Some(a.sha256.clone()),
                    })
                    .collect(),
            }],
        })
    }

    /// Download alone never constructs or loads a runtime model.
    pub fn install(
        &self,
        client: &Client,
        catalog: &Catalog,
        store: &Store,
        identity: &AudioProfileIdentity,
        progress: &mut dyn FnMut(HubDownloadProgress),
        cancel: &CancelFlag,
    ) -> Result<InstalledModel, String> {
        let plan = self.pinned_plan(identity)?;
        cancel.checkpoint().map_err(|e| e.to_string())?;
        if store.audio_install_path(&identity.alias).exists() {
            return Err("audio install already exists; delete it before downloading again".into());
        }
        std::fs::create_dir_all(store.root()).map_err(|e| format!("creating audio store: {e}"))?;
        let hub = crate::HubClient::new(
            client,
            catalog.clone(),
            crate::Machine::default(),
            4096,
            model_io::ExpertCacheSlots::Auto,
        )
        .with_store(store.clone());
        let receipt = hub
            .download_pinned_artifacts(&plan, store.root(), progress, cancel)
            .map_err(|e| e.to_string())?;
        let published = self.publish_verified(store, identity, &receipt, cancel);
        let cleanup = std::fs::remove_dir_all(&receipt.staging_root);
        match (published, cleanup) {
            (Ok(installed), Ok(())) => Ok(installed),
            (Ok(_), Err(e)) => Err(format!(
                "audio install published, but removing transfer staging failed: {e}"
            )),
            (Err(e), _) => Err(e),
        }
    }

    /// Independently verify the transfer receipt and bytes before a single directory rename.
    /// The caller owns staging, including cleanup after cancellation or failure.
    pub fn publish_verified(
        &self,
        store: &Store,
        identity: &AudioProfileIdentity,
        receipt: &DownloadReceipt,
        cancel: &CancelFlag,
    ) -> Result<InstalledModel, String> {
        let profile = self.profile(identity)?;
        let plan = self.pinned_plan(identity)?;
        cancel.checkpoint().map_err(|e| e.to_string())?;
        let destination = store.audio_install_path(&identity.alias);
        if destination.exists() {
            return Err("audio install already exists; publication cannot overwrite it".into());
        }
        if receipt.owner_id != plan.owner_id || receipt.sources.len() != 1 {
            return Err(
                "audio transfer owner or source count does not match requested identity".into(),
            );
        }
        let group = &receipt.sources[0];
        let expected = &plan.sources[0];
        if group.role != expected.role
            || group.repo != expected.repo
            || group.files.len() != profile.assets.len()
        {
            return Err("audio transfer does not match the pinned source graph".into());
        }
        let source_root = receipt.staging_root.join("source-0000");
        require_directory(&receipt.staging_root)?;
        let actual = regular_paths(&source_root)?;
        let pinned: std::collections::BTreeSet<_> =
            profile.assets.iter().map(|a| a.path.clone()).collect();
        if actual != pinned {
            return Err("audio staging has an incomplete or unexpected asset graph".into());
        }
        let mut remaining = pinned;
        for file in &group.files {
            if !remaining.remove(&file.path) {
                return Err("audio transfer has an unexpected or duplicate asset".into());
            }
            let asset = profile
                .assets
                .iter()
                .find(|a| a.path == file.path)
                .ok_or("audio transfer has a foreign asset")?;
            if file.staged_path != source_root.join(&file.path)
                || file.size != asset.size
                || file.sha256 != asset.sha256
            {
                return Err(format!(
                    "audio transfer metadata does not match {}",
                    asset.path
                ));
            }
        }
        verify_asset_bytes(profile, &source_root, true, Some(cancel))?;
        if identity.task == AudioTask::SpeechToText {
            crate::install_speech_receipt(
                &source_root,
                &profile
                    .assets
                    .iter()
                    .map(|a| a.path.clone())
                    .collect::<Vec<_>>(),
                Some(identity.repository.clone()),
                Some(identity.revision.clone()),
            )?;
        }
        let native_receipt = new_audio_receipt(profile);
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(source_root.join(RECEIPT_FILENAME))
            .map_err(|e| format!("creating audio receipt: {e}"))?;
        use std::io::Write;
        file.write_all(&serde_json::to_vec_pretty(&native_receipt).map_err(|e| e.to_string())?)
            .and_then(|_| file.sync_all())
            .map_err(|e| format!("writing audio receipt: {e}"))?;
        cancel.checkpoint().map_err(|e| e.to_string())?;
        std::fs::create_dir_all(store.audio_models_dir())
            .map_err(|e| format!("creating audio namespace: {e}"))?;
        std::fs::rename(&source_root, &destination)
            .map_err(|e| format!("publishing audio install: {e}"))?;
        let installed = InstalledModel {
            alias: identity.alias.clone(),
            repo: identity.repository.clone(),
            revision: identity.revision.clone(),
            path: destination.clone(),
            family: profile.family.clone(),
            install_bytes: crate::directory_bytes(&destination),
            installed_on: crate::install::today(),
            status: "unverified".into(),
            kind: if identity.task == AudioTask::SpeechToText {
                Some("speech".into())
            } else {
                None
            },
            modality: crate::ModelModality::Audio,
            variant: if identity.task == AudioTask::Music {
                Some("affine-4bit".into())
            } else {
                None
            },
        };
        if let Err(e) = store.record(&installed) {
            // Restore the caller's complete staging tree when the registry could not publish.
            std::fs::rename(&destination, &source_root).map_err(|rollback| {
                format!("recording audio install failed: {e}; restoring staging failed: {rollback}")
            })?;
            return Err(format!("recording audio install failed: {e}"));
        }
        Ok(installed)
    }

    /// Fast picker lookup: receipt identity, graph, and sizes. This does not detect same-size corruption.
    pub fn resolve(
        &self,
        store: &Store,
        identity: &AudioProfileIdentity,
    ) -> Result<PathBuf, String> {
        let profile = self.profile(identity)?;
        let path = store.audio_install_path(&identity.alias);
        read_receipt(profile, &path)?;
        verify_asset_bytes(profile, &path, false, None)?;
        Ok(path)
    }

    /// Full byte verification for the runtime-open boundary, separate from picker resolution.
    pub fn verify(
        &self,
        store: &Store,
        identity: &AudioProfileIdentity,
    ) -> Result<PathBuf, String> {
        let path = self.resolve(store, identity)?;
        verify_asset_bytes(self.profile(identity)?, &path, true, None)?;
        Ok(path)
    }

    /// Explicitly adopt a legacy exact-pin install after independent byte verification.
    /// A store status is not qualification evidence and never promotes readiness.
    pub fn adopt_legacy(
        &self,
        store: &Store,
        identity: &AudioProfileIdentity,
        cancel: &CancelFlag,
    ) -> Result<InstalledModel, String> {
        let profile = self.profile(identity)?;
        cancel.checkpoint().map_err(|e| e.to_string())?;
        let mut matching: Vec<_> = store
            .audio_records()
            .into_iter()
            .filter(|row| row.alias == identity.alias)
            .collect();
        if matching.len() != 1 {
            return Err("legacy adoption requires one unambiguous Audio store record".into());
        }
        let mut row = matching.pop().expect("one matching record");
        validate_profile_record(profile, store, &row)?;
        validate_legacy_graph(profile, &row.path)?;
        let receipt_exists = !audio_receipt_missing(&row.path)?;
        if receipt_exists {
            read_receipt(profile, &row.path)?;
        }
        verify_asset_bytes(profile, &row.path, true, Some(cancel))?;
        cancel.checkpoint().map_err(|e| e.to_string())?;
        let created = if receipt_exists {
            false
        } else {
            write_adoption_receipt(profile, &row.path, cancel)?;
            true
        };
        row.status = unqualified_status(&row.status);
        row.install_bytes = crate::directory_bytes(&row.path);
        if let Err(error) = store.record(&row) {
            if created {
                std::fs::remove_file(row.path.join(RECEIPT_FILENAME))
                    .map_err(|rollback| format!("recording audio adoption failed: {error}; removing new receipt failed: {rollback}"))?;
            }
            return Err(format!("recording audio adoption failed: {error}"));
        }
        Ok(row)
    }

    /// Per-record discovery keeps a damaged or older install from hiding usable profiles.
    /// Legacy rows are candidates for explicit adoption, not trusted runtime installs.
    pub fn installed_report(&self, store: &Store) -> AudioInstalledReport {
        let mut report = AudioInstalledReport::default();
        let rows = store.audio_records();
        let mut counts = std::collections::BTreeMap::new();
        for row in &rows {
            *counts.entry(row.alias.as_str()).or_insert(0usize) += 1;
        }
        for mut row in rows.iter().cloned() {
            row.status = unqualified_status(&row.status);
            let profile = self.get(&row.alias);
            let identity = profile.map(|profile| profile.identity.clone());
            let checked = (|| {
                if counts[row.alias.as_str()] != 1 {
                    return Err("ambiguous duplicate Audio store records".into());
                }
                if let Some(profile) = profile {
                    let canonical = store.audio_install_path(&row.alias);
                    if !audio_receipt_missing(&canonical)? {
                        // A native receipt proves ownership after a store-root move.
                        // Legacy records still require their exact original managed path.
                        read_receipt(profile, &canonical)?;
                        row.path = canonical;
                    }
                    validate_profile_record(profile, store, &row)?;
                    if audio_receipt_missing(&row.path)? {
                        validate_legacy_graph(profile, &row.path)?;
                        verify_asset_bytes(profile, &row.path, false, None)?;
                        return Ok(true);
                    }
                    self.resolve(store, &profile.identity)?;
                } else {
                    validate_managed_audio_record(store, &row)?;
                }
                Ok(false)
            })();
            match checked {
                Ok(needs_adoption) => match identity {
                    Some(identity) if needs_adoption => {
                        report.needs_adoption.push(AudioLegacyInstall {
                            record: row,
                            identity,
                        })
                    }
                    Some(_) => report.installed.push(row),
                    None => report.other_audio.push(row),
                },
                Err(error) => report.incompatible.push(AudioInstalledRecordError {
                    record: row,
                    identity,
                    error,
                }),
            }
        }
        report
    }

    pub fn installed(&self, store: &Store) -> Result<Vec<InstalledModel>, String> {
        Ok(self.installed_report(store).installed)
    }

    /// Remove an owned legacy install even when its payload cannot pass adoption.
    /// Receipt-backed installs use `delete`; this path never promotes damaged bytes.
    pub fn delete_legacy(
        &self,
        store: &Store,
        identity: &AudioProfileIdentity,
    ) -> Result<(), String> {
        let profile = self.profile(identity)?;
        let mut matching: Vec<_> = store
            .audio_records()
            .into_iter()
            .filter(|row| row.alias == identity.alias)
            .collect();
        if matching.len() != 1 {
            return Err("legacy deletion requires one unambiguous Audio store record".into());
        }
        let row = matching.pop().expect("one matching record");
        validate_profile_record(profile, store, &row)?;
        if !audio_receipt_missing(&row.path)? {
            return Err("receipt-backed Audio installs require receipt-checked deletion".into());
        }
        // Ownership permits missing or corrupt assets, but no redirected subtree.
        regular_paths(&row.path)?;
        delete_managed_audio_path(store, identity, Some(row))
    }

    /// Callers must close idle native handles before deleting. Corrupt payloads remain deletable.
    pub fn delete(&self, store: &Store, identity: &AudioProfileIdentity) -> Result<(), String> {
        let profile = self.profile(identity)?;
        let path = store.audio_install_path(&identity.alias);
        require_directory(&store.models_root())?;
        require_directory(&store.audio_models_dir())?;
        read_receipt(profile, &path)?;
        let row = store.installed_audio().remove(&identity.alias);
        delete_managed_audio_path(store, identity, row)
    }
}

fn delete_managed_audio_path(
    store: &Store,
    identity: &AudioProfileIdentity,
    row: Option<InstalledModel>,
) -> Result<(), String> {
    let path = store.audio_install_path(&identity.alias);
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tombstone = store.audio_models_dir().join(format!(
        ".delete-{}-{}-{}",
        identity.alias,
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::rename(&path, &tombstone).map_err(|e| format!("preparing audio deletion: {e}"))?;
    if let Err(e) = store.forget_audio(&identity.alias) {
        std::fs::rename(&tombstone, &path).map_err(|rollback| {
            format!("forgetting audio install failed: {e}; restoring install failed: {rollback}")
        })?;
        return Err(e);
    }
    if let Err(e) = std::fs::remove_dir_all(&tombstone) {
        let _ = std::fs::rename(&tombstone, &path);
        if let Some(row) = row {
            let _ = store.record(&row);
        }
        return Err(format!("removing audio install: {e}"));
    }
    Ok(())
}

fn unqualified_status(status: &str) -> String {
    match status {
        "unverified"
        | "unlisted"
        | "experimental"
        | "unsupported"
        | "runtime_pending"
        | "implemented_unqualified" => status.into(),
        _ => "unverified".into(),
    }
}

fn validate_managed_audio_record(store: &Store, row: &InstalledModel) -> Result<(), String> {
    if row.modality != crate::ModelModality::Audio
        || row.effective_modality() != crate::ModelModality::Audio
    {
        return Err("legacy record is not an Audio model".into());
    }
    if row.alias.is_empty()
        || !row
            .alias
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
    {
        return Err("legacy Audio alias is not safe".into());
    }
    if row.path != store.audio_install_path(&row.alias) {
        return Err("legacy Audio record does not name this store's managed path".into());
    }
    // A managed namespace cannot redirect adoption into another tree through a symlink.
    require_directory(&store.models_root())?;
    require_directory(&store.audio_models_dir())?;
    require_directory(&row.path)
}

fn validate_profile_record(
    profile: &AudioProfile,
    store: &Store,
    row: &InstalledModel,
) -> Result<(), String> {
    validate_managed_audio_record(store, row)?;
    if row.alias != profile.identity.alias
        || row.repo != profile.identity.repository
        || row.revision != profile.identity.revision
        || row.family != profile.family
    {
        return Err("legacy Audio record does not match the requested alias, repository, revision, and family".into());
    }
    let task_matches = match profile.identity.task {
        AudioTask::SpeechToText => {
            row.kind.as_deref().is_none_or(|kind| kind == "speech")
                && row
                    .variant
                    .as_deref()
                    .is_none_or(|variant| variant == "hf-safetensors")
        }
        AudioTask::TextToSpeech => {
            row.kind.as_deref().is_none_or(|kind| kind == "tts")
                && row
                    .variant
                    .as_deref()
                    .is_none_or(|variant| variant == "bf16")
        }
        AudioTask::Music => {
            row.kind.is_none()
                && row
                    .variant
                    .as_deref()
                    .is_none_or(|variant| variant == "affine-4bit")
        }
    };
    if !task_matches {
        return Err(
            "legacy Audio record task or variant metadata does not match the profile".into(),
        );
    }
    Ok(())
}

fn audio_receipt_missing(root: &Path) -> Result<bool, String> {
    match std::fs::symlink_metadata(root.join(RECEIPT_FILENAME)) {
        Ok(_) => Ok(false),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(true),
        Err(error) => Err(format!("checking Audio receipt: {error}")),
    }
}

fn validate_legacy_graph(profile: &AudioProfile, root: &Path) -> Result<(), String> {
    let mut actual = regular_paths(root)?;
    actual.remove(RECEIPT_FILENAME);
    if profile.identity.task == AudioTask::SpeechToText {
        actual.remove(model_io::speech_receipt::SPEECH_RECEIPT_FILENAME);
    }
    let expected = profile
        .assets
        .iter()
        .map(|asset| asset.path.clone())
        .collect();
    if actual != expected {
        return Err("legacy Audio install has an incomplete or unexpected asset graph".into());
    }
    Ok(())
}

fn new_audio_receipt(profile: &AudioProfile) -> AudioInstallReceipt {
    AudioInstallReceipt {
        schema_version: 1,
        identity: profile.identity.clone(),
        family: profile.family.clone(),
        files: profile
            .assets
            .iter()
            .map(|asset| {
                (
                    asset.path.clone(),
                    model_io::speech_receipt::FileEntry {
                        size: asset.size,
                        sha256: asset.sha256.clone(),
                    },
                )
            })
            .collect(),
    }
}

fn write_adoption_receipt(
    profile: &AudioProfile,
    root: &Path,
    cancel: &CancelFlag,
) -> Result<(), String> {
    use std::io::Write;
    struct TemporaryReceipt(PathBuf);
    impl Drop for TemporaryReceipt {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let staged = TemporaryReceipt(root.join(format!(
        ".audio-adoption-{}-{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&staged.0)
        .map_err(|error| format!("staging Audio adoption receipt: {error}"))?;
    let bytes = serde_json::to_vec_pretty(&new_audio_receipt(profile))
        .map_err(|error| error.to_string())?;
    file.write_all(&bytes)
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("writing Audio adoption receipt: {error}"))?;
    cancel.checkpoint().map_err(|error| error.to_string())?;
    // A hard link atomically creates the receipt without overwriting a concurrent owner's receipt.
    std::fs::hard_link(&staged.0, root.join(RECEIPT_FILENAME))
        .map_err(|error| format!("publishing Audio adoption receipt without overwrite: {error}"))
}

fn immutable_revision(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|b| b.is_ascii_hexdigit())
}
fn digest_shape(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn safe_asset_path(value: &str) -> bool {
    !value.is_empty()
        && value.is_ascii()
        && !value.contains('\\')
        && value
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
        && !Path::new(value).is_absolute()
}
fn validate_assets(assets: &[AudioAsset]) -> Result<(), String> {
    if assets.is_empty() {
        return Err("audio profile has no assets".into());
    }
    let mut paths = std::collections::HashSet::new();
    for asset in assets {
        if !safe_asset_path(&asset.path)
            || asset.size == 0
            || !digest_shape(&asset.sha256)
            || !paths.insert(&asset.path)
        {
            return Err(format!("invalid or duplicate audio asset {:?}", asset.path));
        }
    }
    Ok(())
}
fn validate_frontend(frontend: &AudioFrontendProvenance) -> Result<(), String> {
    use sha2::{Digest, Sha256};
    // Bundled pins remain separate from upstream byte hashes after ASCII JSON
    // transcoding. Both participate in the persisted catalog identity.
    let provenance: serde_json::Value = serde_json::from_str(include_str!(
        "../../audio/src/tts/kokoro/frontend/resources/provenance.json"
    ))
    .map_err(|e| e.to_string())?;
    let assets = provenance["assets"]
        .as_array()
        .ok_or("missing bundled frontend provenance")?;
    let expected: Vec<AudioFrontendResource> = assets[..6]
        .iter()
        .cloned()
        .map(serde_json::from_value)
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    let license: AudioFrontendResource =
        serde_json::from_value(assets[6].clone()).map_err(|e| e.to_string())?;
    if frontend.repository != license.repository
        || frontend.revision != license.revision
        || frontend.license != license.license
        || frontend.license_path != license.path
        || frontend.license_size != license.size
        || frontend.license_sha256 != license.sha256
        || frontend.bundled_license_path != license.bundled_path
        || frontend.bundled_license_size != license.bundled_size
        || frontend.bundled_license_sha256 != license.bundled_sha256
        || frontend.resources.len() != 6
    {
        return Err("invalid Kokoro frontend source/license provenance".into());
    }
    let bundled: [&[u8]; 6] = [
        include_bytes!("../../audio/src/tts/kokoro/frontend/resources/us_gold.json"),
        include_bytes!("../../audio/src/tts/kokoro/frontend/resources/us_silver.json"),
        include_bytes!("../../audio/src/tts/kokoro/frontend/licenses/MISAKI-APACHE-2.0.txt"),
        include_bytes!("../../audio/src/tts/kokoro/frontend/resources/pos_weights.json"),
        include_bytes!("../../audio/src/tts/kokoro/frontend/resources/pos_tags.json"),
        include_bytes!("../../audio/src/tts/kokoro/frontend/resources/pos_classes.txt"),
    ];
    let mut paths = std::collections::HashSet::new();
    for resource in &frontend.resources {
        let index = expected.iter().position(|pin| pin == resource).ok_or(
            "Kokoro frontend resource differs from its pinned upstream/bundled provenance",
        )?;
        if !paths.insert(&resource.path)
            || resource.bundled_size != bundled[index].len() as u64
            || resource.bundled_sha256 != format!("{:x}", Sha256::digest(bundled[index]))
        {
            return Err(
                "Kokoro frontend bundled resource is missing, changed or duplicated".into(),
            );
        }
    }
    let license_bytes =
        include_bytes!("../../audio/src/tts/kokoro/frontend/licenses/MISAKI-RS-MIT.txt");
    if frontend.bundled_license_size != license_bytes.len() as u64
        || frontend.bundled_license_sha256 != format!("{:x}", Sha256::digest(license_bytes))
    {
        return Err("Kokoro bundled frontend license differs from its pin".into());
    }
    Ok(())
}
fn asset_fingerprint(
    assets: &[AudioAsset],
    frontend: &Option<AudioFrontendProvenance>,
) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let mut ordered = assets.to_vec();
    ordered.sort_by(|a, b| a.path.cmp(&b.path));
    let bytes = serde_json::to_vec(&(ordered, frontend)).map_err(|e| e.to_string())?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
fn require_directory(path: &Path) -> Result<(), String> {
    let meta = std::fs::symlink_metadata(path)
        .map_err(|e| format!("reading audio directory {}: {e}", path.display()))?;
    if !meta.is_dir() {
        return Err(format!(
            "audio directory {} must not be a symlink",
            path.display()
        ));
    }
    Ok(())
}
fn regular_paths(root: &Path) -> Result<std::collections::BTreeSet<String>, String> {
    require_directory(root)?;
    fn visit(
        root: &Path,
        path: &Path,
        files: &mut std::collections::BTreeSet<String>,
    ) -> Result<(), String> {
        for entry in std::fs::read_dir(path).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let kind = entry.file_type().map_err(|e| e.to_string())?;
            let path = entry.path();
            if kind.is_dir() {
                visit(root, &path, files)?;
            } else if kind.is_file() {
                files.insert(
                    path.strip_prefix(root)
                        .map_err(|e| e.to_string())?
                        .to_str()
                        .ok_or("non-UTF8 audio path")?
                        .replace(std::path::MAIN_SEPARATOR, "/"),
                );
            } else {
                return Err(
                    "audio assets must be regular files, not symlinks or special files".into(),
                );
            }
        }
        Ok(())
    }
    let mut files = std::collections::BTreeSet::new();
    visit(root, root, &mut files)?;
    Ok(files)
}
fn verify_asset_bytes(
    profile: &AudioProfile,
    root: &Path,
    hash: bool,
    cancel: Option<&CancelFlag>,
) -> Result<(), String> {
    require_directory(root)?;
    for asset in &profile.assets {
        if let Some(cancel) = cancel {
            cancel.checkpoint().map_err(|e| e.to_string())?;
        }
        let path = root.join(&asset.path);
        // Check every ancestor so a nested asset cannot escape through a symlink.
        let mut parent = path.parent();
        while let Some(dir) = parent {
            if dir == root {
                break;
            }
            require_directory(dir)?;
            parent = dir.parent();
        }
        let meta = std::fs::symlink_metadata(&path)
            .map_err(|e| format!("reading audio asset {}: {e}", asset.path))?;
        if !meta.is_file() || meta.len() != asset.size {
            return Err(format!(
                "audio asset {} is absent, unsafe, or has changed size",
                asset.path
            ));
        }
        if hash
            && model_io::hash_file(&path, 1024 * 1024).map_err(|e| e.to_string())? != asset.sha256
        {
            return Err(format!(
                "audio asset {} SHA-256 does not match its immutable pin",
                asset.path
            ));
        }
    }
    Ok(())
}
fn read_receipt(profile: &AudioProfile, root: &Path) -> Result<AudioInstallReceipt, String> {
    require_directory(root)?;
    let path = root.join(RECEIPT_FILENAME);
    let meta =
        std::fs::symlink_metadata(&path).map_err(|e| format!("reading audio receipt: {e}"))?;
    if !meta.is_file() || meta.len() > 256 * 1024 {
        return Err("audio receipt must be a bounded regular file".into());
    }
    let receipt: AudioInstallReceipt =
        serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| format!("invalid audio receipt: {e}"))?;
    if receipt.schema_version != 1
        || receipt.identity != profile.identity
        || receipt.family != profile.family
        || receipt.files.len() != profile.assets.len()
        || profile.assets.iter().any(|a| {
            receipt
                .files
                .get(&a.path)
                .is_none_or(|f| f.size != a.size || f.sha256 != a.sha256)
        })
    {
        return Err(
            "audio receipt identity or asset graph does not match requested profile".into(),
        );
    }
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ModelModality, RepoRef, VerifiedSourceFile, VerifiedSourceGroup};
    use sha2::{Digest, Sha256};
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "catalog-audio-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&root).unwrap();
            Self(root)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn fixture(root: &Path) -> (AudioCatalog, AudioProfileIdentity, DownloadReceipt) {
        let mut catalog = AudioCatalog::embedded().unwrap();
        let profile = catalog
            .entries
            .iter_mut()
            .find(|p| p.identity.task == AudioTask::TextToSpeech)
            .unwrap();
        let staging_root = root.join("transfer");
        let source_root = staging_root.join("source-0000");
        let mut files = Vec::new();
        for asset in &mut profile.assets {
            let bytes = format!("fixture {}", asset.path).into_bytes();
            asset.size = bytes.len() as u64;
            asset.sha256 = format!("{:x}", Sha256::digest(&bytes));
            let staged_path = source_root.join(&asset.path);
            fs::create_dir_all(staged_path.parent().unwrap()).unwrap();
            fs::write(&staged_path, bytes).unwrap();
            files.push(VerifiedSourceFile {
                path: asset.path.clone(),
                staged_path,
                size: asset.size,
                sha256: asset.sha256.clone(),
            });
        }
        profile.identity.asset_fingerprint =
            asset_fingerprint(&profile.assets, &profile.frontend).unwrap();
        let identity = profile.identity.clone();
        let plan = catalog.pinned_plan(&identity).unwrap();
        let receipt = DownloadReceipt {
            owner_id: plan.owner_id,
            staging_root,
            sources: vec![VerifiedSourceGroup {
                role: plan.sources[0].role.clone(),
                repo: RepoRef::new(&identity.repository, &identity.revision),
                files,
            }],
        };
        (catalog, identity, receipt)
    }

    fn legacy_fixture(
        root: &Path,
        task: AudioTask,
    ) -> (AudioCatalog, AudioProfileIdentity, Store, InstalledModel) {
        let mut catalog = AudioCatalog::embedded().unwrap();
        let profile = catalog
            .entries
            .iter_mut()
            .find(|p| p.identity.task == task)
            .unwrap();
        let store = Store::new(root.join("store"));
        let path = store.audio_install_path(&profile.identity.alias);
        for asset in &mut profile.assets {
            let bytes = format!("legacy fixture {}", asset.path).into_bytes();
            asset.size = bytes.len() as u64;
            asset.sha256 = format!("{:x}", Sha256::digest(&bytes));
            let target = path.join(&asset.path);
            fs::create_dir_all(target.parent().unwrap()).unwrap();
            fs::write(target, bytes).unwrap();
        }
        profile.identity.asset_fingerprint =
            asset_fingerprint(&profile.assets, &profile.frontend).unwrap();
        let identity = profile.identity.clone();
        let row = InstalledModel {
            alias: identity.alias.clone(),
            repo: identity.repository.clone(),
            revision: identity.revision.clone(),
            path,
            family: profile.family.clone(),
            install_bytes: profile.download_bytes,
            installed_on: "2026-10-05".into(),
            status: "verified".into(),
            kind: None,
            modality: ModelModality::Audio,
            variant: (task == AudioTask::Music).then_some("affine-4bit".into()),
        };
        store.record(&row).unwrap();
        (catalog, identity, store, row)
    }

    #[test]
    fn legacy_music_adoption_verifies_the_pin_preserves_payloads_and_supports_management() {
        let root = Scratch::new();
        let (catalog, identity, store, row) = legacy_fixture(&root.0, AudioTask::Music);
        let profile = catalog.get(&identity.alias).unwrap();
        let payloads: Vec<_> = profile
            .assets
            .iter()
            .map(|a| (a.path.clone(), fs::read(row.path.join(&a.path)).unwrap()))
            .collect();
        assert!(!row.path.join(RECEIPT_FILENAME).exists());
        assert!(catalog.installed(&store).unwrap().is_empty());
        let pending = catalog.installed_report(&store);
        assert_eq!(pending.needs_adoption.len(), 1);
        assert_eq!(pending.needs_adoption[0].record.status, "unverified");
        assert!(pending.incompatible.is_empty());
        let adopted = catalog
            .adopt_legacy(&store, &identity, &CancelFlag::new())
            .unwrap();
        assert_eq!(adopted.status, "unverified");
        assert_eq!(adopted.path, row.path);
        assert_eq!(
            store.installed_audio()[&identity.alias].status,
            "unverified"
        );
        for (name, before) in payloads {
            assert_eq!(fs::read(row.path.join(name)).unwrap(), before);
        }
        assert_eq!(catalog.resolve(&store, &identity).unwrap(), row.path);
        assert_eq!(catalog.verify(&store, &identity).unwrap(), row.path);
        assert_eq!(catalog.installed(&store).unwrap(), vec![adopted]);
        let relocated = Store::new(root.0.join("relocated"));
        fs::rename(store.root(), relocated.root()).unwrap();
        assert_eq!(
            catalog.resolve(&relocated, &identity).unwrap(),
            relocated.audio_install_path(&identity.alias)
        );
        let relocated_report = catalog.installed_report(&relocated);
        assert!(relocated_report.incompatible.is_empty());
        assert_eq!(relocated_report.installed.len(), 1);
        assert_eq!(
            relocated_report.installed[0].path,
            relocated.audio_install_path(&identity.alias)
        );
        catalog.delete(&relocated, &identity).unwrap();
        assert!(!relocated.audio_install_path(&identity.alias).exists());
    }

    #[test]
    fn legacy_adoption_rejects_wrong_ownership_corruption_incomplete_and_extra_assets() {
        for mutation in 0..11 {
            let root = Scratch::new();
            let (catalog, identity, store, mut row) = legacy_fixture(&root.0, AudioTask::Music);
            match mutation {
                0 => row.repo = "foreign/model".into(),
                1 => row.revision = "a".repeat(40),
                2 => row.family = "whisper".into(),
                3 => row.modality = ModelModality::Text,
                4 => row.kind = Some("speech".into()),
                5 => row.variant = Some("mxfp8".into()),
                6 => row.alias = "other-alias".into(),
                7 => {
                    let path = &catalog.get(&identity.alias).unwrap().assets[0].path;
                    let bytes = fs::read(row.path.join(path)).unwrap();
                    fs::write(row.path.join(path), vec![b'x'; bytes.len()]).unwrap();
                }
                8 => {
                    fs::remove_file(
                        row.path
                            .join(&catalog.get(&identity.alias).unwrap().assets[0].path),
                    )
                    .unwrap();
                }
                9 => {
                    fs::write(row.path.join("unexpected.json"), b"foreign").unwrap();
                }
                _ => {
                    row.path = root.0.join("outside");
                    fs::create_dir_all(&row.path).unwrap();
                }
            }
            // Replace the raw registry row rather than silently rebasing its path.
            fs::write(
                store.root().join("installed.json"),
                serde_json::to_vec(&vec![row]).unwrap(),
            )
            .unwrap();
            assert!(
                catalog
                    .adopt_legacy(&store, &identity, &CancelFlag::new())
                    .is_err(),
                "mutation {mutation}"
            );
            let managed = store.audio_install_path(&identity.alias);
            assert!(managed.is_dir());
            assert!(
                !managed.join(RECEIPT_FILENAME).exists(),
                "mutation {mutation}"
            );
            assert!(catalog.installed(&store).unwrap().is_empty());
            assert!(catalog.delete(&store, &identity).is_err());
        }
    }

    #[test]
    fn legacy_listing_isolates_incompatible_records_and_preserves_other_music_variants() {
        let root = Scratch::new();
        let (catalog, identity, receipt) = fixture(&root.0);
        let store = Store::new(root.0.join("store"));
        let good = catalog
            .publish_verified(&store, &identity, &receipt, &CancelFlag::new())
            .unwrap();
        let mut legacy = good.clone();
        let music = catalog.get("minimax-music3-4bit").unwrap();
        legacy.alias = music.identity.alias.clone();
        legacy.repo = music.identity.repository.clone();
        legacy.revision = "a".repeat(40);
        legacy.family = music.family.clone();
        legacy.status = "verified".into();
        legacy.variant = Some("affine-4bit".into());
        legacy.path = store.audio_install_path(&legacy.alias);
        fs::create_dir_all(&legacy.path).unwrap();
        store.record(&legacy).unwrap();
        let mut other = legacy.clone();
        let other_pin = crate::embedded_music_entry("minimax-music3-mxfp8").unwrap();
        other.alias = other_pin.alias;
        other.repo = other_pin.model_id;
        other.revision = other_pin.revision;
        other.status = "unlisted".into();
        other.variant = Some("mxfp8".into());
        other.path = store.audio_install_path(&other.alias);
        fs::create_dir_all(&other.path).unwrap();
        store.record(&other).unwrap();
        assert_eq!(catalog.installed(&store).unwrap(), vec![good.clone()]);
        let report = catalog.installed_report(&store);
        assert_eq!(report.installed, vec![good]);
        assert!(report.needs_adoption.is_empty());
        assert_eq!(report.incompatible.len(), 1);
        assert_eq!(report.incompatible[0].record.alias, legacy.alias);
        assert_eq!(report.incompatible[0].record.status, "unverified");
        assert_eq!(report.other_audio, vec![other.clone()]);
        assert_eq!(store.resolve_audio(&other.alias).unwrap(), other.path);
        assert_eq!(
            crate::embedded_music_entry(&other.alias).unwrap().revision,
            other.revision
        );
    }

    #[test]
    fn legacy_receipt_requires_explicit_ownership_before_corrupt_payload_deletion() {
        let root = Scratch::new();
        let (catalog, identity, store, row) = legacy_fixture(&root.0, AudioTask::Music);
        assert!(catalog.delete(&store, &identity).is_err());
        catalog
            .adopt_legacy(&store, &identity, &CancelFlag::new())
            .unwrap();
        let asset = &catalog.get(&identity.alias).unwrap().assets[0];
        fs::write(row.path.join(&asset.path), vec![b'x'; asset.size as usize]).unwrap();
        assert!(catalog.verify(&store, &identity).is_err());
        catalog.delete(&store, &identity).unwrap();
        assert!(!row.path.exists());
    }

    #[test]
    fn legacy_delete_removes_owned_corruption_and_preserves_other_modalities() {
        for incomplete in [false, true] {
            let root = Scratch::new();
            let (catalog, identity, store, row) = legacy_fixture(&root.0, AudioTask::Music);
            let asset = &catalog.get(&identity.alias).unwrap().assets[0];
            if incomplete {
                fs::remove_file(row.path.join(&asset.path)).unwrap();
            } else {
                fs::write(row.path.join(&asset.path), vec![b'x'; asset.size as usize]).unwrap();
            }
            let mut text = row.clone();
            text.modality = ModelModality::Text;
            text.path = store.install_path(&identity.alias);
            fs::create_dir_all(&text.path).unwrap();
            fs::write(text.path.join("keep"), b"text").unwrap();
            store.record(&text).unwrap();
            let mut image = row.clone();
            image.modality = ModelModality::Image;
            image.path = store.image_install_path(&identity.alias);
            fs::create_dir_all(&image.path).unwrap();
            fs::write(image.path.join("keep"), b"image").unwrap();
            store.record(&image).unwrap();
            assert!(catalog
                .adopt_legacy(&store, &identity, &CancelFlag::new())
                .is_err());
            assert!(catalog.delete(&store, &identity).is_err());
            catalog.delete_legacy(&store, &identity).unwrap();
            assert!(!row.path.exists());
            assert!(store.audio_records().is_empty());
            assert_eq!(fs::read(text.path.join("keep")).unwrap(), b"text");
            assert_eq!(fs::read(image.path.join("keep")).unwrap(), b"image");
            assert!(store.installed().contains_key(&identity.alias));
            assert_eq!(store.resolve_image(&identity.alias).unwrap(), image.path);
        }
    }

    #[test]
    fn legacy_delete_refuses_mismatched_metadata_and_receipt_backed_installs() {
        for mutation in 0..8 {
            let root = Scratch::new();
            let (catalog, identity, store, mut row) = legacy_fixture(&root.0, AudioTask::Music);
            match mutation {
                0 => row.repo = "foreign/model".into(),
                1 => row.revision = "a".repeat(40),
                2 => row.family = "whisper".into(),
                3 => row.modality = ModelModality::Text,
                4 => row.kind = Some("speech".into()),
                5 => row.variant = Some("mxfp8".into()),
                6 => row.path = root.0.join("outside"),
                _ => {
                    catalog
                        .adopt_legacy(&store, &identity, &CancelFlag::new())
                        .unwrap();
                }
            }
            if mutation == 6 {
                fs::create_dir_all(&row.path).unwrap();
            }
            if mutation != 7 {
                fs::write(
                    store.root().join("installed.json"),
                    serde_json::to_vec(&vec![row]).unwrap(),
                )
                .unwrap();
            }
            assert!(
                catalog.delete_legacy(&store, &identity).is_err(),
                "mutation {mutation}"
            );
            assert!(store.audio_install_path(&identity.alias).is_dir());
        }
    }

    #[cfg(unix)]
    #[test]
    fn legacy_delete_refuses_file_directory_and_namespace_symlinks() {
        for mutation in 0..3 {
            let root = Scratch::new();
            let (catalog, identity, store, row) = legacy_fixture(&root.0, AudioTask::Music);
            let victim = match mutation {
                0 => row
                    .path
                    .join(&catalog.get(&identity.alias).unwrap().assets[0].path),
                1 => row.path.clone(),
                _ => store.audio_models_dir(),
            };
            let outside = root.0.join("outside");
            fs::rename(&victim, &outside).unwrap();
            std::os::unix::fs::symlink(&outside, &victim).unwrap();
            assert!(catalog.delete_legacy(&store, &identity).is_err());
            assert!(outside.exists());
            assert!(victim.exists());
        }
    }

    #[test]
    fn legacy_adoption_preserves_existing_unqualified_status() {
        let root = Scratch::new();
        let (catalog, identity, store, mut row) = legacy_fixture(&root.0, AudioTask::Music);
        row.status = "unlisted".into();
        store.record(&row).unwrap();
        let adopted = catalog
            .adopt_legacy(&store, &identity, &CancelFlag::new())
            .unwrap();
        assert_eq!(adopted.status, "unlisted");
        assert_eq!(store.installed_audio()[&identity.alias].status, "unlisted");
    }

    #[test]
    fn legacy_adoption_is_cancelled_before_writing_a_receipt() {
        let root = Scratch::new();
        let (catalog, identity, store, row) = legacy_fixture(&root.0, AudioTask::Music);
        let cancel = CancelFlag::new();
        cancel.cancel();
        assert!(catalog
            .adopt_legacy(&store, &identity, &cancel)
            .unwrap_err()
            .contains("cancelled"));
        assert!(!row.path.join(RECEIPT_FILENAME).exists());
    }

    #[test]
    fn initial_audio_profiles_have_exact_complete_pins_and_wire_metadata() {
        let catalog = AudioCatalog::embedded().unwrap();
        assert_eq!(catalog.entries().count(), 3);
        for (task, alias, repo, revision, rate, channels) in [
            (
                AudioTask::SpeechToText,
                "whisper-base",
                "openai/whisper-base",
                "e37978b90ca9030d5170a5c07aadb050351a65bb",
                16000,
                1,
            ),
            (
                AudioTask::TextToSpeech,
                "kokoro-82m-bf16",
                "mlx-community/Kokoro-82M-bf16",
                "a71e4d38b236d968966a2002c4c895dbd12b1c3c",
                24000,
                1,
            ),
            (
                AudioTask::Music,
                "minimax-music3-4bit",
                "mlx-community/MiniMax-Music3-4bit",
                "c7ea32923b245fe5afc22d740a1936ad2ac590f3",
                44100,
                2,
            ),
        ] {
            let profile = catalog.get(alias).unwrap();
            assert_eq!(profile.identity.task, task);
            assert_eq!(profile.identity.repository, repo);
            assert_eq!(profile.identity.revision, revision);
            assert_eq!(
                profile.pcm_format,
                AudioPcmFormat {
                    sample_rate: rate,
                    channels,
                    interleaved: true
                }
            );
            let plan = catalog.pinned_plan(&profile.identity).unwrap();
            assert_eq!(plan.sources.len(), 1);
            assert_eq!(plan.sources[0].files.len(), profile.assets.len());
            assert!(plan.sources[0]
                .files
                .iter()
                .all(|f| f.expected_size.is_some()
                    && f.expected_sha256.as_ref().is_some_and(|s| s.len() == 64)));
            assert_eq!(
                profile.download_bytes,
                profile.assets.iter().map(|a| a.size).sum::<u64>()
            );
            assert!(profile.resident_memory_bytes > 0);
            assert!(profile.resident_memory_evidence.contains("estimate"));
            assert!(!profile.evidence.is_empty());
        }
        let whisper = catalog.get("whisper-base").unwrap();
        for file in crate::embedded_speech_entry("whisper-base")
            .unwrap()
            .required_files
        {
            assert!(whisper.assets.iter().any(|a| a.path == file));
        }
        let music = catalog.get("minimax-music3-4bit").unwrap();
        for file in crate::embedded_music_entry("minimax-music3-4bit")
            .unwrap()
            .required_files
        {
            assert!(music.assets.iter().any(|a| a.path == file));
        }
        let kokoro = catalog.get("kokoro-82m-bf16").unwrap();
        for file in [
            "config.json",
            "kokoro-v1_0.safetensors",
            "voices/af_heart.safetensors",
        ] {
            assert!(kokoro.assets.iter().any(|a| a.path == file));
        }
        let frontend = kokoro.frontend.as_ref().unwrap();
        assert_eq!(frontend.repository, "MicheleYin/misaki-rs");
        assert_eq!(
            frontend.revision,
            "38bf1a534cce5f6864fb2df595447ca6f6891e74"
        );
        assert_eq!(frontend.license, "MIT");
        assert_eq!(frontend.license_sha256.len(), 64);
        let json = serde_json::to_value(kokoro).unwrap();
        assert_eq!(json["identity"]["task"], "text_to_speech");
        assert!(json.get("download_bytes").is_some());
        assert!(json.get("downloadBytes").is_none());
        assert_eq!(
            serde_json::from_value::<AudioProfile>(json).unwrap(),
            *kokoro
        );
    }

    #[test]
    fn frontend_pos_assets_are_complete_and_fingerprinted() {
        let catalog = AudioCatalog::embedded().unwrap();
        let profile = catalog.get("kokoro-82m-bf16").unwrap();
        let frontend = profile.frontend.as_ref().unwrap();
        assert_eq!(frontend.resources.len(), 6);
        for path in [
            "src/resources/tagger/weights.json",
            "src/resources/tagger/tags.json",
            "src/resources/tagger/classes.txt",
        ] {
            let resource = frontend
                .resources
                .iter()
                .find(|r| r.path == path)
                .expect("POS asset pin");
            assert_eq!(resource.license, "MIT");
            let mut changed = profile.frontend.clone();
            changed
                .as_mut()
                .unwrap()
                .resources
                .iter_mut()
                .find(|r| r.path == path)
                .unwrap()
                .sha256 = "0".repeat(64);
            assert_ne!(
                asset_fingerprint(&profile.assets, &changed).unwrap(),
                profile.identity.asset_fingerprint
            );
            assert!(validate_frontend(changed.as_ref().unwrap()).is_err());
        }
    }

    #[test]
    fn malformed_manifest_cannot_drop_runtime_assets_or_pin_unsafe_files() {
        for mutation in 0..8 {
            let mut wire: serde_json::Value = serde_json::from_str(ASSETS).unwrap();
            match mutation {
                0 => {
                    wire["profiles"][1]["assets"].as_array_mut().unwrap().pop();
                }
                1 => wire["profiles"][0]["assets"][0]["path"] = "../escape".into(),
                2 => wire["profiles"][0]["assets"][0]["sha256"] = "changed".into(),
                3 => wire["profiles"][0]["assets"][0]["size"] = 0.into(),
                4 => wire["profiles"][1]["source"]["revision"] = "main".into(),
                5 => wire["schema_version"] = 2.into(),
                6 => {
                    wire["profiles"][1]["frontend"]["resources"][0]["path"] = "unknown.json".into()
                }
                _ => wire["profiles"][1]["frontend"]["license"] = "unknown".into(),
            }
            assert!(
                AudioCatalog::from_manifest(&wire.to_string()).is_err(),
                "mutation {mutation}"
            );
        }
    }

    #[test]
    fn full_identity_rejects_each_stale_field_and_changed_asset_plan() {
        let catalog = AudioCatalog::embedded().unwrap();
        let identity = catalog.get("whisper-base").unwrap().identity.clone();
        for field in 0..5 {
            let mut stale = identity.clone();
            match field {
                0 => stale.task = AudioTask::Music,
                1 => stale.alias = "unknown".into(),
                2 => stale.repository = "other/model".into(),
                3 => stale.revision = "a".repeat(40),
                _ => stale.asset_fingerprint = "b".repeat(64),
            };
            assert!(catalog.profile(&stale).is_err());
            assert!(catalog.pinned_plan(&stale).is_err());
        }
        let mut assets = catalog.get("whisper-base").unwrap().assets.clone();
        assets[0].sha256 = "c".repeat(64);
        assert_ne!(
            asset_fingerprint(&assets, &None).unwrap(),
            identity.asset_fingerprint
        );
    }

    #[test]
    fn publication_rejects_incomplete_changed_duplicate_and_corrupt_receipts() {
        for mutation in 0..8 {
            let root = Scratch::new();
            let store = Store::new(root.0.join("store"));
            let (catalog, identity, mut receipt) = fixture(&root.0);
            match mutation {
                0 => {
                    receipt.sources[0].files.pop();
                }
                1 => receipt.sources[0].repo.revision = "a".repeat(40),
                2 => receipt.owner_id = "foreign".into(),
                3 => receipt.sources[0].files[0].sha256 = "b".repeat(64),
                4 => {
                    let p = &receipt.sources[0].files[0].staged_path;
                    let bytes = fs::read(p).unwrap();
                    fs::write(p, vec![b'x'; bytes.len()]).unwrap();
                }
                5 => receipt.sources[0].files[1] = receipt.sources[0].files[0].clone(),
                6 => receipt.sources[0].files[0].staged_path = root.0.join("outside"),
                _ => {
                    fs::write(
                        receipt.staging_root.join("source-0000/extra.json"),
                        b"foreign",
                    )
                    .unwrap();
                }
            }
            assert!(
                catalog
                    .publish_verified(&store, &identity, &receipt, &CancelFlag::new())
                    .is_err(),
                "mutation {mutation}"
            );
            assert!(!store.audio_install_path(&identity.alias).exists());
            assert!(!store.root().join("installed.json").exists());
        }
    }

    #[test]
    fn valid_publication_persists_pin_receipt_then_resolves_relocates_and_deletes() {
        let root = Scratch::new();
        let store = Store::new(root.0.join("store"));
        let (catalog, identity, receipt) = fixture(&root.0);
        let installed = catalog
            .publish_verified(&store, &identity, &receipt, &CancelFlag::new())
            .unwrap();
        assert_eq!(installed.modality, ModelModality::Audio);
        assert_eq!(installed.status, "unverified");
        assert_eq!(installed.repo, identity.repository);
        assert_eq!(installed.revision, identity.revision);
        assert!(installed.path.join("audio-receipt.json").is_file());
        assert_eq!(catalog.resolve(&store, &identity).unwrap(), installed.path);
        assert_eq!(catalog.installed(&store).unwrap(), vec![installed.clone()]);
        let relocated = Store::new(root.0.join("relocated"));
        fs::rename(store.root(), relocated.root()).unwrap();
        let destination = catalog.resolve(&relocated, &identity).unwrap();
        assert_eq!(destination, relocated.audio_install_path(&identity.alias));
        let mut text = installed;
        text.modality = ModelModality::Text;
        text.path = relocated.install_path(&identity.alias);
        fs::create_dir_all(&text.path).unwrap();
        relocated.record(&text).unwrap();
        catalog.delete(&relocated, &identity).unwrap();
        assert!(!destination.exists());
        assert!(text.path.exists());
        assert!(relocated.installed().contains_key(&identity.alias));
        assert!(catalog.installed(&relocated).unwrap().is_empty());
        assert!(catalog.resolve(&relocated, &identity).is_err());
    }

    #[test]
    fn failed_republication_and_cancel_preserve_the_previous_install() {
        let root = Scratch::new();
        let store = Store::new(root.0.join("store"));
        let (catalog, identity, receipt) = fixture(&root.0);
        let installed = catalog
            .publish_verified(&store, &identity, &receipt, &CancelFlag::new())
            .unwrap();
        assert!(catalog
            .publish_verified(&store, &identity, &receipt, &CancelFlag::new())
            .is_err());
        assert_eq!(catalog.resolve(&store, &identity).unwrap(), installed.path);
        let root = Scratch::new();
        let store = Store::new(root.0.join("store"));
        let (catalog, identity, receipt) = fixture(&root.0);
        let cancel = CancelFlag::new();
        cancel.cancel();
        assert!(catalog
            .publish_verified(&store, &identity, &receipt, &cancel)
            .unwrap_err()
            .contains("cancelled"));
        assert!(!store.audio_install_path(&identity.alias).exists());
    }

    #[test]
    fn resolution_and_deletion_refuse_changed_receipt_identity_and_corrupt_bytes() {
        let root = Scratch::new();
        let store = Store::new(root.0.join("store"));
        let (catalog, identity, receipt) = fixture(&root.0);
        let installed = catalog
            .publish_verified(&store, &identity, &receipt, &CancelFlag::new())
            .unwrap();
        let receipt_path = installed.path.join("audio-receipt.json");
        let bytes = fs::read(&receipt_path).unwrap();
        let mut wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        wire["identity"]["revision"] = "a".repeat(40).into();
        fs::write(&receipt_path, serde_json::to_vec(&wire).unwrap()).unwrap();
        assert!(catalog.resolve(&store, &identity).is_err());
        assert!(catalog.delete(&store, &identity).is_err());
        assert!(installed.path.exists());
        fs::write(receipt_path, bytes).unwrap();
        let weight = &catalog.get(&identity.alias).unwrap().assets[0];
        fs::write(
            installed.path.join(&weight.path),
            vec![b'x'; weight.size as usize],
        )
        .unwrap();
        assert!(catalog.resolve(&store, &identity).is_ok());
        assert!(catalog.verify(&store, &identity).is_err());
    }

    #[test]
    fn record_failure_rolls_back_publication_and_leaves_staging_owned_by_caller() {
        let root = Scratch::new();
        let store = Store::new(root.0.join("store"));
        fs::create_dir_all(store.root().join("installed.json")).unwrap();
        let (catalog, identity, receipt) = fixture(&root.0);
        assert!(catalog
            .publish_verified(&store, &identity, &receipt, &CancelFlag::new())
            .is_err());
        assert!(!store.audio_install_path(&identity.alias).exists());
        assert!(receipt.staging_root.join("source-0000").is_dir());
    }

    #[cfg(unix)]
    #[test]
    fn publication_rejects_staged_symlinks() {
        let root = Scratch::new();
        let store = Store::new(root.0.join("store"));
        let (catalog, identity, receipt) = fixture(&root.0);
        let staged = &receipt.sources[0].files[0].staged_path;
        let outside = root.0.join("outside");
        fs::rename(staged, &outside).unwrap();
        std::os::unix::fs::symlink(outside, staged).unwrap();
        assert!(catalog
            .publish_verified(&store, &identity, &receipt, &CancelFlag::new())
            .is_err());
    }
}
