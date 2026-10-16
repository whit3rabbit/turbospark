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
    Enhancement,
    Separation,
    Alignment,
    Diarization,
    SpeechDetection,
    Codec,
    LanguageIdentification,
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

/// Terms an installer must accept before a profile downloads, such as the
/// unverified provenance of a community-sourced weight. Clients show
/// `summary` (or a localized text for a known `id`) and pass the accepted ids
/// to [`AudioCatalog::install_acknowledged`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioAcknowledgment {
    pub id: String,
    pub summary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioCapabilities {
    pub languages: Vec<String>,
    pub voices: Vec<String>,
    pub max_input_seconds: Option<u32>,
    pub supports_lyrics: bool,
    #[serde(default)]
    pub operations: Vec<String>,
    #[serde(default)]
    pub timing: Option<String>,
    #[serde(default)]
    pub backend: String,
    #[serde(default)]
    pub cancellation: String,
    #[serde(default)]
    pub can_run: bool,
    #[serde(default)]
    pub unavailable_reason: Option<String>,
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
    /// Accepted before download; empty for most profiles.
    #[serde(default)]
    pub acknowledgments: Vec<AudioAcknowledgment>,
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
    metadata: Option<ProfileMetadata>,
    frontend: Option<AudioFrontendProvenance>,
}
/// Explicit profile metadata lets pinned manifests grow without a profile-count switch.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileMetadata {
    task: AudioTask,
    family: String,
    display_name: String,
    required_files: Vec<String>,
    sample_rate: u32,
    channels: u32,
    languages: Vec<String>,
    voices: Vec<String>,
    max_input_seconds: Option<u32>,
    #[serde(default)]
    acknowledgments: Vec<AudioAcknowledgment>,
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
            Self::Enhancement => "enhancement",
            Self::Separation => "separation",
            Self::Alignment => "alignment",
            Self::Diarization => "diarization",
            Self::SpeechDetection => "speech_detection",
            Self::Codec => "codec",
            Self::LanguageIdentification => "language_identification",
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
        if manifest.schema_version != 1 || manifest.profiles.is_empty() {
            return Err("audio manifest must have schema 1 and at least one profile".into());
        }
        let mut entries = Vec::new();
        let mut aliases = std::collections::HashSet::new();
        for row in manifest.profiles {
            if row.alias.is_empty()
                || !row
                    .alias
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
            {
                return Err("audio profile alias must be a plain lowercase catalog name".into());
            }
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
                acknowledgments,
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
                        speech.format.speech_family().as_str().to_string(),
                        "Whisper Base".to_string(),
                        speech.required_files,
                        16000,
                        1,
                        AudioCapabilities {
                            languages: vec!["auto".into(), "multilingual".into()],
                            voices: vec![],
                            max_input_seconds: Some(1800),
                            supports_lyrics: false,
                            ..profile_capabilities("whisper")
                        },
                        AudioReadiness::ImplementedUnqualified,
                        vec![],
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
                        "kokoro".to_string(),
                        "Kokoro 82M BF16".to_string(),
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
                            ..profile_capabilities("kokoro")
                        },
                        AudioReadiness::RuntimePending,
                        vec![],
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
                        "minimax_music3".to_string(),
                        "MiniMax Music 3 4-bit".to_string(),
                        music.required_files,
                        44100,
                        2,
                        AudioCapabilities {
                            languages: vec![],
                            voices: vec![],
                            max_input_seconds: Some(360),
                            supports_lyrics: true,
                            ..profile_capabilities("minimax_music3")
                        },
                        AudioReadiness::ImplementedUnqualified,
                        vec![],
                    )
                }
                alias => {
                    let metadata = row.metadata.ok_or_else(|| {
                        format!("audio profile {alias:?} needs explicit metadata")
                    })?;
                    let source = row
                        .source
                        .ok_or("audio profile needs an immutable source")?;
                    if metadata.family.is_empty()
                        || metadata.display_name.is_empty()
                        || metadata.sample_rate == 0
                        || !(1..=2).contains(&metadata.channels)
                        || metadata.required_files.is_empty()
                    {
                        return Err("invalid audio profile metadata".into());
                    }
                    for ack in &metadata.acknowledgments {
                        let plain = |v: &str| {
                            !v.is_empty()
                                && v.bytes().all(|b| {
                                    b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'
                                })
                        };
                        if !plain(&ack.id) || ack.summary.trim().is_empty() {
                            return Err(format!(
                                "{alias} has an acknowledgment without a plain id and a summary"
                            ));
                        }
                    }
                    let mut capabilities = profile_capabilities(&metadata.family);
                    capabilities.languages = metadata.languages;
                    capabilities.voices = metadata.voices;
                    capabilities.max_input_seconds = metadata.max_input_seconds;
                    let reviewed = REVIEWED_ADAPTERS.contains(&metadata.family.as_str());
                    // New profiles stay closed until a specific runtime adapter is reviewed.
                    if !reviewed {
                        capabilities.can_run = false;
                        if metadata.family != "qwen3_tts" {
                            capabilities.unavailable_reason = Some("This pinned profile has not been integrated and qualified for native execution".into());
                        }
                    }
                    (
                        metadata.task,
                        source.repository,
                        source.revision,
                        metadata.family,
                        metadata.display_name,
                        metadata.required_files,
                        metadata.sample_rate,
                        metadata.channels,
                        capabilities,
                        if reviewed {
                            AudioReadiness::ImplementedUnqualified
                        } else {
                            AudioReadiness::RuntimePending
                        },
                        metadata.acknowledgments,
                    )
                }
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
            let g2p_bytes = row
                .assets
                .iter()
                .filter(|a| a.path.starts_with("g2p/") && a.path.ends_with(".safetensors"))
                .try_fold(0u64, |n, a| n.checked_add(a.size))
                .ok_or("audio weight size overflow")?;
            let (resident_memory_bytes,resident_memory_evidence)=match task {
                // f32 weights, the bf16 G2P widened, and a reserve covering the
                // measured peak footprint of a 28 s reference.
                AudioTask::TextToSpeech if family == "megatts3" => (weight_bytes.saturating_add(g2p_bytes).saturating_add(19<<29),"measured peak footprint on Metal 8.8 GiB (3 s reference) to 14.3 GiB (27.5 s reference, 13 s of speech), 1.4 to 2.9 GiB above the CPU path; estimate: weights, widened G2P and 9.5 GiB working reserve"),
                AudioTask::TextToSpeech if family == "omnivoice" => (weight_bytes.saturating_mul(3).saturating_add(2<<30),"measured 3.2 GiB process footprint on one 3 s utterance (file-backed checkpoint pages excluded, 8-bit profile); estimate: three times checkpoint bytes for the f32-widened backbone plus a 2 GiB working reserve"),
                AudioTask::SpeechToText if family == "parakeet_tdt" => {
                    let compiled = row
                        .assets
                        .iter()
                        .filter(|a| a.path.ends_with("/weights/weight.bin"))
                        .try_fold(0u64, |n, a| n.checked_add(a.size))
                        .ok_or("audio weight size overflow")?;
                    (compiled.saturating_mul(3).saturating_add(16_000*1800*4).saturating_add(256<<20),"measured 488 MB peak footprint on one 2.8 s clip (GPU encoder, M4 Max); estimate: three times compiled weights, 30-minute PCM, and 256 MiB working reserve")
                }
                AudioTask::SpeechToText => (weight_bytes.saturating_mul(2).saturating_add(16_000*1800*4).saturating_add(256<<20),"estimate: twice checkpoint weights, 30-minute PCM, and 256 MiB working reserve"),
                AudioTask::TextToSpeech => (weight_bytes.saturating_mul(2).saturating_add(256<<20),"estimate: twice checkpoint weights and 256 MiB voice/working reserve"),
                AudioTask::Music => (weight_bytes.saturating_add(2<<30),"estimate: packed checkpoint weights and 2 GiB working reserve; full duration peak unqualified"),
                _ => (weight_bytes.saturating_mul(4),"estimate: runtime admission pending"),
            };
            let identity = AudioProfileIdentity {
                task,
                alias: row.alias,
                repository,
                revision,
                asset_fingerprint: asset_fingerprint(&row.assets, &row.frontend)?,
            };
            entries.push(AudioProfile {
                identity, family, display_name, capabilities,
                pcm_format: AudioPcmFormat { sample_rate: rate, channels, interleaved: true },
                download_bytes, resident_memory_bytes,
                resident_memory_evidence: resident_memory_evidence.into(), readiness,
                evidence: vec!["Immutable source asset plan; runtime/reference and measured resident gates remain unqualified".into()],
                assets: row.assets, frontend: row.frontend, acknowledgments,
            });
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
    /// Profiles with acknowledgments refuse here; use [`Self::install_acknowledged`].
    pub fn install(
        &self,
        client: &Client,
        catalog: &Catalog,
        store: &Store,
        identity: &AudioProfileIdentity,
        progress: &mut dyn FnMut(HubDownloadProgress),
        cancel: &CancelFlag,
    ) -> Result<InstalledModel, String> {
        self.install_acknowledged(client, catalog, store, identity, &[], progress, cancel)
    }

    /// [`Self::install`] for a caller that has shown the profile's
    /// acknowledgments and collected the accepted ids. Every acknowledgment
    /// must be accepted; the check runs before any network or store work.
    #[allow(clippy::too_many_arguments)]
    pub fn install_acknowledged(
        &self,
        client: &Client,
        catalog: &Catalog,
        store: &Store,
        identity: &AudioProfileIdentity,
        acknowledged: &[String],
        progress: &mut dyn FnMut(HubDownloadProgress),
        cancel: &CancelFlag,
    ) -> Result<InstalledModel, String> {
        check_acknowledged(self.profile(identity)?, acknowledged)?;
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
        if published.is_ok() {
            // Best effort: the published install holds these bytes now, and
            // a leftover cache costs disk, not correctness.
            let _ = receipt.discard_download_cache();
        }
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
        // The hub returns the repository id lowercased (Hub ids are
        // case-insensitive), so compare it the way `matches_exact_source` does.
        if group.role != expected.role
            || !group.repo.repo.eq_ignore_ascii_case(&expected.repo.repo)
            || !group
                .repo
                .revision
                .eq_ignore_ascii_case(&expected.repo.revision)
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

    /// [`Self::verify`] that can be stopped and can report progress.
    ///
    /// `cancelled` is polled once per hashed chunk (1 MiB), so even one
    /// multi-gigabyte weight file stops within a chunk of the request rather
    /// than only between files. `on_progress(asset, completed, total)` reports
    /// bytes hashed across ALL of the profile's assets, throttled to about
    /// every 16 MiB and at each file's end. A stopped verification returns
    /// [`VERIFY_CANCELLED`] as its error. The pinned hashes are compared
    /// exactly as in [`Self::verify`]: this adds observation, not a weaker
    /// check, and a cancelled run never reports success.
    pub fn verify_observed(
        &self,
        store: &Store,
        identity: &AudioProfileIdentity,
        cancelled: &dyn Fn() -> bool,
        on_progress: &mut dyn FnMut(&str, u64, u64),
    ) -> Result<PathBuf, String> {
        let path = self.resolve(store, identity)?;
        let mut observer = VerifyObserver {
            cancelled,
            on_progress,
        };
        verify_asset_bytes_observed(
            self.profile(identity)?,
            &path,
            true,
            None,
            Some(&mut observer),
        )?;
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
        if !profile.acknowledgments.is_empty() {
            return Err(format!(
                "{} needs its terms accepted at install; delete the folder and install it from the catalog",
                identity.alias
            ));
        }
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
            row.modality = crate::ModelModality::Audio;
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
        let mut known_aliases: std::collections::BTreeSet<String> =
            rows.iter().map(|r| r.alias.clone()).collect();
        for inst in &report.installed {
            known_aliases.insert(inst.alias.clone());
        }
        if let Ok(entries) = std::fs::read_dir(store.audio_models_dir()) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() && path.extension().and_then(|e| e.to_str()) == Some("gturbo") {
                    if let Some(alias) = path.file_stem().and_then(|s| s.to_str()) {
                        if !known_aliases.contains(alias)
                            && !audio_receipt_missing(&path).unwrap_or(true)
                        {
                            if let Some(profile) = self.get(alias) {
                                if read_receipt(profile, &path).is_ok()
                                    && verify_asset_bytes(profile, &path, false, None).is_ok()
                                {
                                    let installed = InstalledModel {
                                        alias: profile.identity.alias.clone(),
                                        repo: profile.identity.repository.clone(),
                                        revision: profile.identity.revision.clone(),
                                        path: path.clone(),
                                        family: profile.family.clone(),
                                        install_bytes: crate::directory_bytes(&path),
                                        installed_on: crate::install::today(),
                                        status: "unverified".into(),
                                        kind: if profile.identity.task == AudioTask::SpeechToText {
                                            Some("speech".into())
                                        } else {
                                            None
                                        },
                                        modality: crate::ModelModality::Audio,
                                        variant: if profile.identity.task == AudioTask::Music {
                                            Some("affine-4bit".into())
                                        } else {
                                            None
                                        },
                                    };
                                    let _ = store.record(&installed);
                                    report.installed.push(installed);
                                    known_aliases.insert(alias.to_string());
                                }
                            }
                        }
                    }
                }
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

/// Families whose generic catalog rows may run: a runtime adapter for them
/// exists and was reviewed. Every other new row stays closed.
const REVIEWED_ADAPTERS: &[&str] = &["megatts3", "omnivoice", "parakeet_tdt"];

fn check_acknowledged(profile: &AudioProfile, acknowledged: &[String]) -> Result<(), String> {
    let missing: Vec<&str> = profile
        .acknowledgments
        .iter()
        .filter(|a| !acknowledged.contains(&a.id))
        .map(|a| a.id.as_str())
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "installing {} requires accepting: {}",
            profile.identity.alias,
            missing.join(", ")
        ))
    }
}

fn profile_capabilities(family: &str) -> AudioCapabilities {
    if family == "omnivoice" {
        // The runtime engine is portable f32 CPU only; like Kokoro, the
        // CPU path needs the explicit portable opt-in until a Metal
        // offload exists. Cloning is capability-driven (voice_clone) and
        // requires consent at the runtime boundary.
        return AudioCapabilities {
            languages: vec![
                "en", "zh", "ja", "ko", "de", "fr", "es", "it", "pt", "ru", "ar", "hi", "nl",
                "pl", "tr", "vi", "th", "id", "uk", "sv", "fa", "cs", "el", "he",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            voices: vec![],
            max_input_seconds: None,
            supports_lyrics: false,
            operations: vec!["synthesize".into(), "voice_clone".into()],
            timing: None,
            backend: "portable_cpu".into(),
            cancellation: "diffusion_step".into(),
            can_run: false,
            unavailable_reason: Some(
                "OmniVoice has no Metal runtime; portable CPU execution requires the experimental opt-in"
                    .into(),
            ),
        };
    }
    if family == "megatts3" {
        return AudioCapabilities {
            languages: vec![],
            voices: vec![],
            max_input_seconds: None,
            supports_lyrics: false,
            operations: vec!["synthesize".into(), "voice_clone".into()],
            timing: None,
            // Metal by default; the CPU reference is the portable fallback.
            backend: "metal".into(),
            cancellation: "diffusion_step".into(),
            can_run: true,
            unavailable_reason: None,
        };
    }
    if family == "parakeet_tdt" {
        // Core ML programs targeting iOS 18 / macOS 15. On an older OS the
        // runtime refuses to open them, so the profile says so up front.
        let os_ok = turbospark_coreml::os_version().0 >= 15;
        return AudioCapabilities {
            languages: vec![],
            voices: vec![],
            max_input_seconds: None,
            supports_lyrics: false,
            operations: vec!["transcribe".into()],
            // Decoder-frame alignment on an 80 ms grid.
            timing: Some("token".into()),
            backend: "coreml".into(),
            cancellation: "window".into(),
            can_run: os_ok,
            unavailable_reason: (!os_ok)
                .then(|| "Parakeet on Core ML requires macOS 15 or newer".into()),
        };
    }
    let (operation,backend,cancellation,can_run,reason,timing) = match family {
        "whisper" => ("transcribe","metal","decode_step",true,None,Some("window")),
        "qwen3_tts" => ("synthesize","swift_mlx","decode_step",false,Some("Qwen3-TTS runs through the native Swift MLX provider; the Rust session API does not execute this family"),None),
        "kokoro" => ("synthesize","portable_cpu","sentence",false,Some("Kokoro Metal runtime is pending; CPU reference is available only with explicit portable opt-in"),None),
        "minimax_music3" => ("generate_music","metal","compute_operation",true,None,None),
        _ => ("inspect","unavailable","unavailable",false,Some("No native workspace runtime adapter is available"),None),
    };
    AudioCapabilities {
        languages: vec![],
        voices: vec![],
        max_input_seconds: None,
        supports_lyrics: false,
        operations: vec![operation.into()],
        timing: timing.map(str::to_string),
        backend: backend.into(),
        cancellation: cancellation.into(),
        can_run,
        unavailable_reason: reason.map(str::to_string),
    }
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
    if row.effective_modality() != crate::ModelModality::Audio {
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
        _ => false,
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
    use audio::tts::kokoro::frontend::bundled as files;
    use sha2::{Digest, Sha256};
    // Bundled pins remain separate from upstream byte hashes after ASCII JSON
    // transcoding. Both participate in the persisted catalog identity.
    let provenance: serde_json::Value =
        serde_json::from_str(files::PROVENANCE_JSON).map_err(|e| e.to_string())?;
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
        files::US_GOLD_JSON,
        files::US_SILVER_JSON,
        files::MISAKI_APACHE_LICENSE,
        files::POS_WEIGHTS_JSON,
        files::POS_TAGS_JSON,
        files::POS_CLASSES_TXT,
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
    let license_bytes = files::MISAKI_RS_MIT_LICENSE;
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
/// The error text a stopped [`AudioCatalog::verify_observed`] returns. A
/// constant so a caller can recognise its own cancel without parsing prose.
pub const VERIFY_CANCELLED: &str = "audio verification cancelled";

/// Cancellation and progress for one verification.
struct VerifyObserver<'a> {
    cancelled: &'a dyn Fn() -> bool,
    on_progress: &'a mut dyn FnMut(&str, u64, u64),
}

/// How many bytes pass between progress reports. The hash reads 1 MiB chunks;
/// reporting each one would call a host callback thousands of times for a
/// large model.
const VERIFY_PROGRESS_STRIDE: u64 = 16 * 1024 * 1024;

fn verify_asset_bytes(
    profile: &AudioProfile,
    root: &Path,
    hash: bool,
    cancel: Option<&CancelFlag>,
) -> Result<(), String> {
    verify_asset_bytes_observed(profile, root, hash, cancel, None)
}

fn verify_asset_bytes_observed(
    profile: &AudioProfile,
    root: &Path,
    hash: bool,
    cancel: Option<&CancelFlag>,
    mut observer: Option<&mut VerifyObserver<'_>>,
) -> Result<(), String> {
    require_directory(root)?;
    let total: u64 = profile.assets.iter().map(|a| a.size).sum();
    let mut finished_bytes = 0u64;
    for asset in &profile.assets {
        if let Some(cancel) = cancel {
            cancel.checkpoint().map_err(|e| e.to_string())?;
        }
        if let Some(observer) = observer.as_deref() {
            if (observer.cancelled)() {
                return Err(VERIFY_CANCELLED.to_string());
            }
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
        if hash {
            let digest = match observer.as_deref_mut() {
                None => Some(model_io::hash_file(&path, 1024 * 1024).map_err(|e| e.to_string())?),
                Some(observer) => {
                    let mut reported = 0u64;
                    let base = finished_bytes;
                    let name = asset.path.as_str();
                    let digest = model_io::hash_file_observed(&path, 1024 * 1024, |done| {
                        if (observer.cancelled)() {
                            return false;
                        }
                        if done - reported >= VERIFY_PROGRESS_STRIDE {
                            reported = done;
                            (observer.on_progress)(name, base + done, total);
                        }
                        true
                    })
                    .map_err(|e| e.to_string())?;
                    if digest.is_some() {
                        (observer.on_progress)(name, base + asset.size, total);
                    }
                    digest
                }
            };
            match digest {
                None => return Err(VERIFY_CANCELLED.to_string()),
                Some(digest) if digest != asset.sha256 => {
                    return Err(format!(
                        "audio asset {} SHA-256 does not match its immutable pin",
                        asset.path
                    ));
                }
                Some(_) => {}
            }
        }
        finished_bytes += asset.size;
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
            download_cache: Vec::new(),
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

    /// The observed verification must reach the SAME verdict as the plain one
    /// (it adds observation, not a weaker check), report progress up to the
    /// total, and stop on request without ever reporting success.
    #[test]
    fn observed_verification_agrees_with_the_plain_check_and_can_be_stopped() {
        let root = Scratch::new();
        let (catalog, identity, _store, row) = legacy_fixture(&root.0, AudioTask::TextToSpeech);
        let profile = catalog.get(&identity.alias).unwrap();

        // Untouched: passes, and the last report is "all bytes of all assets".
        let mut reports: Vec<(String, u64, u64)> = Vec::new();
        let never = || false;
        let mut sink =
            |asset: &str, done: u64, total: u64| reports.push((asset.to_string(), done, total));
        let mut observer = VerifyObserver {
            cancelled: &never,
            on_progress: &mut sink,
        };
        verify_asset_bytes_observed(profile, &row.path, true, None, Some(&mut observer)).unwrap();
        let total: u64 = profile.assets.iter().map(|a| a.size).sum();
        let last = reports.last().expect("at least one progress report");
        assert_eq!((last.1, last.2), (total, total), "{reports:?}");
        assert!(reports.windows(2).all(|w| w[0].1 <= w[1].1), "monotonic");

        // Stopped before it starts: the cancel text, not success.
        let always = || true;
        let mut ignore = |_: &str, _: u64, _: u64| {};
        let mut observer = VerifyObserver {
            cancelled: &always,
            on_progress: &mut ignore,
        };
        let err = verify_asset_bytes_observed(profile, &row.path, true, None, Some(&mut observer))
            .unwrap_err();
        assert_eq!(err, VERIFY_CANCELLED);

        // A flipped byte is still refused through the observed path.
        let victim = row.path.join(&profile.assets[0].path);
        let mut bytes = fs::read(&victim).unwrap();
        bytes[0] ^= 0xff;
        fs::write(&victim, bytes).unwrap();
        let mut observer = VerifyObserver {
            cancelled: &never,
            on_progress: &mut ignore,
        };
        let err = verify_asset_bytes_observed(profile, &row.path, true, None, Some(&mut observer))
            .unwrap_err();
        assert!(err.contains("SHA-256 does not match"), "{err}");
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
        assert_eq!(catalog.entries().count(), 12);
        for (task, alias, repo, revision, rate, channels) in [
            (
                AudioTask::TextToSpeech,
                "omnivoice-8bit",
                "mlx-community/OmniVoice-8bit",
                "b222e4d82e7bc20323be736ef05fcfc50defb71b",
                24000,
                1,
            ),
            (
                AudioTask::TextToSpeech,
                "omnivoice-bf16",
                "mlx-community/OmniVoice-bfloat16",
                "6119f7072580cc30c3cbdf496269216b03ad9742",
                24000,
                1,
            ),
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
            (
                AudioTask::TextToSpeech,
                "megatts3-f32",
                "cowWhySo/MegaTTS3-TurboSpark",
                "a55e74f50cbb88ca72658b4999a0ba85a567e466",
                24000,
                1,
            ),
            (
                AudioTask::SpeechToText,
                "parakeet-redux-coreml",
                "FluidInference/parakeet-redux-coreml",
                "8c5ef97a29cd120dc76b354b3f22b7fec3b486f9",
                16000,
                1,
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
        // The Core ML profile runs through the reviewed native adapter, and
        // its memory estimate counts compiled weights, not safetensors.
        let parakeet = catalog.get("parakeet-redux-coreml").unwrap();
        assert_eq!(parakeet.family, "parakeet_tdt");
        assert_eq!(parakeet.capabilities.backend, "coreml");
        assert_eq!(parakeet.readiness, AudioReadiness::ImplementedUnqualified);
        assert!(parakeet.resident_memory_bytes > 3 * 200_000_000);
        assert_eq!(
            parakeet.capabilities.can_run,
            turbospark_coreml::os_version().0 >= 15
        );
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
    fn megatts3_profile_pins_every_file_the_runtime_opens_and_its_terms() {
        let catalog = AudioCatalog::embedded().unwrap();
        let profile = catalog.get("megatts3-f32").unwrap();
        assert_eq!(profile.family, "megatts3");
        // What `MegaTts3::open` reads, plus the license and provenance notice.
        let mut paths: Vec<_> = profile.assets.iter().map(|a| a.path.as_str()).collect();
        paths.sort_unstable();
        assert_eq!(
            paths,
            [
                "LICENSE",
                "NOTICE.md",
                "aligner_lm/model.safetensors",
                "config.json",
                "diffusion_transformer/model.safetensors",
                "duration_lm/model.safetensors",
                "g2p/config.json",
                "g2p/model.safetensors",
                "g2p/tokenizer.json",
                "wavvae/model.safetensors",
            ]
        );
        assert!(profile.capabilities.can_run);
        assert!(profile
            .capabilities
            .operations
            .iter()
            .any(|o| o == "voice_clone"));
        assert_eq!(profile.capabilities.languages, ["en"]);
        assert_eq!(profile.capabilities.max_input_seconds, Some(28));
        assert_eq!(profile.readiness, AudioReadiness::ImplementedUnqualified);
        // Covers the largest measured peak: Metal, 27.5 s reference, 13 s of
        // speech (README memory gate).
        assert!(profile.resident_memory_bytes >= 15_363_862_144);
        let terms: Vec<_> = profile
            .acknowledgments
            .iter()
            .map(|a| a.id.as_str())
            .collect();
        assert_eq!(terms, ["megatts3_community_encoder"]);
        // The G2P pin is the upstream file, byte for byte.
        let g2p = profile
            .assets
            .iter()
            .find(|a| a.path == "g2p/model.safetensors")
            .unwrap();
        assert_eq!(
            g2p.sha256,
            "0f9d70d454ee35d023a9a54552716a8ccf2411c967abc6a857160527046f62a2"
        );
    }

    #[test]
    fn qwen3_tts_profiles_pin_all_native_swift_components() {
        let catalog = AudioCatalog::embedded().unwrap();
        let profiles: Vec<_> = catalog
            .entries()
            .filter(|p| p.family == "qwen3_tts")
            .collect();
        assert_eq!(profiles.len(), 5);
        for profile in profiles {
            assert_eq!(profile.identity.task, AudioTask::TextToSpeech);
            assert_eq!(profile.capabilities.backend, "swift_mlx");
            assert!(
                !profile.capabilities.can_run,
                "Rust sessions must not claim Swift inference"
            );
            assert_eq!(profile.pcm_format.sample_rate, 24_000);
            assert_eq!(profile.assets.len(), 8);
            assert!(profile
                .assets
                .iter()
                .any(|f| f.path == "speech_tokenizer/model.safetensors"));
            assert!(profile
                .assets
                .iter()
                .any(|f| f.path == "speech_tokenizer/config.json"));
            assert!(catalog.pinned_plan(&profile.identity).is_ok());
            assert_eq!(profile.capabilities.languages.len(), 11);
            assert_eq!(
                profile.capabilities.voices.is_empty(),
                !profile.identity.repository.contains("CustomVoice")
            );
        }
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
        for mutation in 0..9 {
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
                7 => receipt.sources[0].repo.repo = "other-owner/Kokoro-82M-bf16".into(),
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

    /// A real transfer receipt carries the repository id the hub resolved,
    /// which is lowercased. Every pinned audio repository is mixed case, so
    /// an exact compare refused every real install after the download.
    #[test]
    fn publication_accepts_the_lowercased_repository_id_the_hub_resolves() {
        let root = Scratch::new();
        let store = Store::new(root.0.join("store"));
        let (catalog, identity, mut receipt) = fixture(&root.0);
        assert_ne!(
            identity.repository,
            identity.repository.to_ascii_lowercase()
        );
        receipt.sources[0].repo = RepoRef::new(
            identity.repository.to_ascii_lowercase(),
            identity.revision.clone(),
        );
        let installed = catalog
            .publish_verified(&store, &identity, &receipt, &CancelFlag::new())
            .unwrap();
        assert_eq!(installed.repo, identity.repository);
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
    fn megatts3_manifest(family: &str, acknowledgments: serde_json::Value) -> String {
        let files = [
            "config.json",
            "g2p/model.safetensors",
            "diffusion_transformer/model.safetensors",
        ];
        let assets: Vec<_> = files
            .iter()
            .enumerate()
            .map(|(i, path)| serde_json::json!({"path": path, "size": 10, "sha256": format!("{:064x}", i + 1)}))
            .collect();
        serde_json::json!({
            "schema_version": 1,
            "profiles": [{
                "alias": "megatts3-test",
                "assets": assets,
                "source": {"repository": "someone/MegaTTS3-TurboSpark", "revision": "0".repeat(40)},
                "metadata": {
                    "task": "text_to_speech", "family": family, "display_name": "MegaTTS3",
                    "required_files": files, "sample_rate": 24000, "channels": 1,
                    "languages": ["en"], "voices": [], "max_input_seconds": 28,
                    "acknowledgments": acknowledgments
                }
            }]
        })
        .to_string()
    }
    fn encoder_ack() -> serde_json::Value {
        serde_json::json!([{"id": "megatts3_community_encoder", "summary": "The voice-cloning encoder is a community re-upload of unverified origin and license."}])
    }

    #[test]
    fn reviewed_megatts3_row_runs_clones_and_carries_its_acknowledgments() {
        let catalog =
            AudioCatalog::from_manifest(&megatts3_manifest("megatts3", encoder_ack())).unwrap();
        let p = catalog.get("megatts3-test").unwrap();
        assert!(p.capabilities.can_run);
        assert_eq!(p.capabilities.unavailable_reason, None);
        assert_eq!(p.capabilities.operations, vec!["synthesize", "voice_clone"]);
        assert_eq!(p.capabilities.backend, "metal");
        assert_eq!(p.capabilities.max_input_seconds, Some(28));
        assert_eq!(p.readiness, AudioReadiness::ImplementedUnqualified);
        assert_eq!(p.acknowledgments[0].id, "megatts3_community_encoder");
        // f32 weights (20 bytes), the widened G2P (10 more) and the 9.5 GiB reserve.
        assert_eq!(p.resident_memory_bytes, 20 + 10 + (19 << 29));

        // The same row under an unreviewed family stays closed with a reason.
        let closed =
            AudioCatalog::from_manifest(&megatts3_manifest("spark_tts", serde_json::json!([])))
                .unwrap();
        let p = closed.get("megatts3-test").unwrap();
        assert!(!p.capabilities.can_run);
        assert!(p.capabilities.unavailable_reason.is_some());
        assert_eq!(p.readiness, AudioReadiness::RuntimePending);

        for bad in [
            serde_json::json!([{"id": "", "summary": "x"}]),
            serde_json::json!([{"id": "Has Space", "summary": "x"}]),
            serde_json::json!([{"id": "ok", "summary": "  "}]),
        ] {
            assert!(AudioCatalog::from_manifest(&megatts3_manifest("megatts3", bad)).is_err());
        }
    }

    #[test]
    fn install_refuses_unacknowledged_terms_before_any_work() {
        let catalog =
            AudioCatalog::from_manifest(&megatts3_manifest("megatts3", encoder_ack())).unwrap();
        let identity = catalog.get("megatts3-test").unwrap().identity.clone();
        let dir = std::env::temp_dir().join(format!("ts-audio-ack-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = Store::new(dir.clone());
        let run = |acknowledged: &[String], cancel: &CancelFlag| {
            catalog
                .install_acknowledged(
                    &Client::new(),
                    &Catalog::embedded().unwrap(),
                    &store,
                    &identity,
                    acknowledged,
                    &mut |_| {},
                    cancel,
                )
                .unwrap_err()
        };
        // Not cancelled: only the acknowledgment check can stop it here.
        let err = run(&[], &CancelFlag::new());
        assert!(
            err.contains("requires accepting: megatts3_community_encoder"),
            "{err}"
        );
        let err = run(&["something_else".into()], &CancelFlag::new());
        assert!(err.contains("requires accepting"), "{err}");
        assert!(!dir.exists(), "a refused install must not touch the store");
        // Accepted: the check passes and the pre-cancelled flag stops it next.
        let cancelled = CancelFlag::new();
        cancelled.cancel();
        let err = run(&["megatts3_community_encoder".into()], &cancelled);
        assert!(!err.contains("requires accepting"), "{err}");
        assert!(!dir.exists());
        // The plain installer passes no acknowledgments.
        let err = catalog
            .install(
                &Client::new(),
                &Catalog::embedded().unwrap(),
                &store,
                &identity,
                &mut |_| {},
                &CancelFlag::new(),
            )
            .unwrap_err();
        assert!(err.contains("requires accepting"), "{err}");
        // Adoption cannot skip the terms either.
        let err = catalog
            .adopt_legacy(&store, &identity, &CancelFlag::new())
            .unwrap_err();
        assert!(err.contains("terms"), "{err}");
    }

    #[test]
    fn partial_catalog_does_not_require_exactly_three_profiles() {
        let mut json: serde_json::Value = serde_json::from_str(ASSETS).unwrap();
        json["profiles"].as_array_mut().unwrap().truncate(1);
        assert_eq!(
            AudioCatalog::from_manifest(&json.to_string())
                .unwrap()
                .entries()
                .count(),
            1
        );
    }
    #[test]
    fn discovered_components_do_not_become_runnable_profiles() {
        let rows = family_capabilities().unwrap();
        let unique: std::collections::HashSet<_> = rows.iter().map(|row| &row.family).collect();
        assert_eq!(unique.len(), rows.len());
        assert!(rows
            .iter()
            .filter(|row| row.component)
            .all(|row| !row.can_run));
        assert!(rows.iter().filter(|row| !row.can_run).all(|row| row
            .unavailable_reason
            .as_ref()
            .is_some_and(|v| !v.is_empty())));
        for empty in ["stt/cohere_asr", "stt/nemo"] {
            assert!(rows
                .iter()
                .find(|row| row.family == empty)
                .unwrap()
                .operations
                .is_empty());
        }
    }
}

/// Family discovery is separate from curated downloadable profiles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFamilyCapability {
    pub family: String,
    pub task: AudioTask,
    pub operations: Vec<String>,
    pub backend: String,
    pub can_run: bool,
    pub component: bool,
    pub unavailable_reason: Option<String>,
    pub source: String,
    pub evidence: String,
}
pub fn family_capabilities() -> Result<Vec<AudioFamilyCapability>, String> {
    serde_json::from_str(include_str!("audio_families.json")).map_err(|e| e.to_string())
}
