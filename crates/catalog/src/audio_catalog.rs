//! Task-aware pinned audio profiles and verified managed installs.

use crate::{
    drama3_install, CancelFlag, Catalog, Client, DownloadReceipt, HubDownloadProgress,
    InstalledModel, PinnedArtifactPlan, Store,
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

/// A profile whose install is made by converting its pinned upstream files
/// instead of copying them. `assets` pins the upstream files kept as they are;
/// the converter's outputs exist nowhere upstream, so the receipt records
/// them. Clients show `working_bytes` as the space an install needs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioDerived {
    pub tool: String,
    pub tool_version: u32,
    /// Free space the install needs while it runs.
    pub working_bytes: u64,
    /// What the finished install occupies.
    pub install_bytes: u64,
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
    /// Where Core ML places the run's dominant compute at the pinned
    /// revision, measured with `coreml_plan`: `ane`, `gpu`, `cpu`, or `mixed`
    /// when no one device carries it. `None` off Core ML. The compute units a
    /// loader requests are not its placement (docs/FLUIDAUDIO_PHASE0.md).
    #[serde(default)]
    pub compute: Option<String>,
    /// False when no HTTP audio server can attach the profile because only
    /// the TurboSpark app executes it. The server refuses these aliases.
    #[serde(default = "attachable_by_default")]
    pub http_api: bool,
}

fn attachable_by_default() -> bool {
    true
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
    /// Relative folder the assets live under inside the install, for loaders
    /// that only read a fixed layout (FluidAudio's Kokoro and PocketTTS).
    /// Asset paths stay the upstream paths. The receipt sits at the install
    /// root. Part of the asset fingerprint when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub install_subdir: Option<String>,
    /// Set when the install converts upstream files (Drama 3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derived: Option<AudioDerived>,
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
    #[serde(default)]
    install_subdir: Option<String>,
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
    /// The converter's outputs, for a derived profile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    derived: Option<DerivedReceipt>,
}

/// What a derived install holds that no upstream pin covers: each output's
/// size and sha256 as the install measured them, and the conversion that
/// made them.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DerivedReceipt {
    tool: String,
    tool_version: u32,
    outputs: std::collections::BTreeMap<String, model_io::speech_receipt::FileEntry>,
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
            let install_subdir = row.metadata.as_ref().and_then(|m| m.install_subdir.clone());
            if let Some(subdir) = &install_subdir {
                if !safe_install_subdir(subdir) {
                    return Err(format!(
                        "{} has an unsafe install_subdir {subdir:?}",
                        row.alias
                    ));
                }
            }
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
                        // Swift-hosted families keep their own reason: the
                        // app runs them even though a Rust session cannot.
                        if !capabilities.backend.starts_with("swift_") {
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
            // A derived profile reads the upstream shards through its converter, so
            // they are part of what an install downloads though no asset pins them.
            let derived = (family == "fish_drama3").then(|| AudioDerived {
                tool: drama3_install::TOOL.to_string(),
                tool_version: drama3_install::TOOL_VERSION,
                working_bytes: drama3_install::WORKING_BYTES,
                install_bytes: drama3_install::RESULT_BYTES,
            });
            if derived.is_some() {
                drama3_install::Pins::from_assets(&row.assets)?;
            }
            let download_bytes = row
                .assets
                .iter()
                .try_fold(0u64, |n, a| n.checked_add(a.size))
                .ok_or("audio download size overflow")?
                .checked_add(if derived.is_some() {
                    drama3_install::shard_total_bytes()
                } else {
                    0
                })
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
            let compiled_bytes = row
                .assets
                .iter()
                .filter(|a| a.path.ends_with("/weights/weight.bin"))
                .try_fold(0u64, |n, a| n.checked_add(a.size))
                .ok_or("audio weight size overflow")?;
            // Mono f32 at 16 kHz for the longest input a row admits.
            let pcm_bytes = 16_000u64 * 4 * u64::from(capabilities.max_input_seconds.unwrap_or(0));
            let (resident_memory_bytes,resident_memory_evidence)=match task {
                _ if capabilities.backend == "swift_coreml" => (compiled_bytes.saturating_mul(3).saturating_add(pcm_bytes).saturating_add(256<<20),"estimate: three times compiled Core ML weights, the row's longest PCM input, and 256 MiB working reserve; not measured in the app"),
                // f32 weights, the bf16 G2P widened, and a reserve covering the
                // measured peak footprint of a 28 s reference.
                AudioTask::TextToSpeech if family == "megatts3" => (weight_bytes.saturating_add(g2p_bytes).saturating_add(19<<29),"measured peak footprint on Metal 8.8 GiB (3 s reference) to 14.3 GiB (27.5 s reference, 13 s of speech), 1.4 to 2.9 GiB above the CPU path; estimate: weights, widened G2P and 9.5 GiB working reserve"),
                AudioTask::TextToSpeech if family == "fish_drama3" => (8_845_165_754u64,"runtime estimate for the 4-bit install (16 streamed expert slots, two 64 MiB snapshots); measured session peak 6,106 to 6,296 MiB over three readings on an M4 Max, docs/BENCHMARKS.md"),
                AudioTask::TextToSpeech if family == "omnivoice" => (weight_bytes.saturating_mul(3).saturating_add(2<<30),"measured 3.2 GiB process footprint on one 3 s utterance (file-backed checkpoint pages excluded, 8-bit profile); estimate: three times checkpoint bytes for the f32-widened backbone plus a 2 GiB working reserve"),
                AudioTask::SpeechToText if family == "parakeet_tdt" => {
                    let compiled = row
                        .assets
                        .iter()
                        .filter(|a| a.path.ends_with("/weights/weight.bin"))
                        .try_fold(0u64, |n, a| n.checked_add(a.size))
                        .ok_or("audio weight size overflow")?;
                    (compiled.saturating_mul(3).saturating_add(16_000*1800*4).saturating_add(256<<20),"estimate: three times compiled weights, 30-minute PCM, and 256 MiB working reserve; Redux measured 488 MB peak footprint on one 2.8 s clip (GPU encoder, M4 Max)")
                }
                AudioTask::SpeechToText if family == "parakeet_unified" => {
                    let compiled = row
                        .assets
                        .iter()
                        .filter(|a| a.path.ends_with("/weights/weight.bin"))
                        .try_fold(0u64, |n, a| n.checked_add(a.size))
                        .ok_or("audio weight size overflow")?;
                    (compiled.saturating_mul(3).saturating_add(256<<20),"estimate: three times compiled weights and 256 MiB working reserve; a live stream keeps about one 7 s window of audio")
                }
                AudioTask::Diarization if family == "pyannote_community1" => {
                    let compiled = row
                        .assets
                        .iter()
                        .filter(|a| a.path.ends_with("/weights/weight.bin"))
                        .try_fold(0u64, |n, a| n.checked_add(a.size))
                        .ok_or("audio weight size overflow")?;
                    (compiled.saturating_mul(3).saturating_add(16_000*1800*4).saturating_add(768<<20),"measured 858 MB peak footprint on a 30-minute input (M4 Max, Neural Engine embedding, compile cached); estimate: three times compiled weights, 30-minute PCM, and 768 MiB working reserve")
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
                asset_fingerprint: asset_fingerprint(
                    &row.assets,
                    &row.frontend,
                    install_subdir.as_deref(),
                )?,
            };
            entries.push(AudioProfile {
                identity, family, display_name, capabilities,
                pcm_format: AudioPcmFormat { sample_rate: rate, channels, interleaved: true },
                download_bytes, resident_memory_bytes,
                resident_memory_evidence: resident_memory_evidence.into(), readiness,
                evidence: vec!["Immutable source asset plan; runtime/reference and measured resident gates remain unqualified".into()],
                assets: row.assets, frontend: row.frontend, acknowledgments, install_subdir, derived,
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
        if let Some(derived) = &self.profile(identity)?.derived {
            return self.install_derived(client, store, identity, derived, progress, cancel);
        }
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

    /// A derived install: the five root files download whole, the shards stream
    /// through the converter, and every output is hashed into the receipt
    /// before the staging directory is renamed into place. Free space is
    /// checked before any network work, and `cancel` stops the walk at its
    /// next chunk read.
    fn install_derived(
        &self,
        client: &Client,
        store: &Store,
        identity: &AudioProfileIdentity,
        derived: &AudioDerived,
        progress: &mut dyn FnMut(HubDownloadProgress),
        cancel: &CancelFlag,
    ) -> Result<InstalledModel, String> {
        let profile = self.profile(identity)?;
        let pins = drama3_install::Pins::from_assets(&profile.assets)?;
        let available = model_io::available_bytes(&store.audio_install_path(&identity.alias))?;
        let source = drama3_install::HubSource::new(client.token(), cancel);
        self.install_derived_from(
            &source, &pins, available, store, identity, derived, progress, cancel,
        )
    }

    /// [`Self::install_derived`] with the bytes' source, the pins and the free
    /// space passed in, so the walk runs without a network.
    #[allow(clippy::too_many_arguments)]
    fn install_derived_from(
        &self,
        source: &dyn drama3_install::Source,
        pins: &drama3_install::Pins,
        available: u64,
        store: &Store,
        identity: &AudioProfileIdentity,
        derived: &AudioDerived,
        progress: &mut dyn FnMut(HubDownloadProgress),
        cancel: &CancelFlag,
    ) -> Result<InstalledModel, String> {
        use drama3_install::Progress;
        if derived.tool != drama3_install::TOOL
            || derived.tool_version != drama3_install::TOOL_VERSION
        {
            return Err(format!(
                "this build cannot make {:?} version {} installs",
                derived.tool, derived.tool_version
            ));
        }
        let profile = self.profile(identity)?;
        let owner_id = self.pinned_plan(identity)?.owner_id;
        let destination = store.audio_install_path(&identity.alias);
        std::fs::create_dir_all(store.audio_models_dir())
            .map_err(|e| format!("creating audio namespace: {e}"))?;
        // One bar for the whole install: the upstream bytes first, then the
        // hash of everything the converter wrote.
        let walk_total = pins.total_bytes();
        let overall_total = walk_total + derived.install_bytes;
        let mut report = |completed: u64, stage: &str| {
            progress(HubDownloadProgress {
                owner_id: owner_id.clone(),
                completed_files: 0,
                total_files: 0,
                current_path: Some(stage.to_string()),
                completed_bytes: completed.min(overall_total),
                total_bytes: overall_total,
            });
        };
        let pinned: std::collections::BTreeSet<String> =
            profile.assets.iter().map(|a| a.path.clone()).collect();
        drama3_install::install_with(
            &destination,
            pins,
            source,
            available,
            cancel,
            &mut |p: &Progress<'_>| report(p.completed_bytes, p.stage),
            |staging, sink| {
                let outputs = drama3_install::hash_outputs(
                    staging,
                    &pinned,
                    RECEIPT_FILENAME,
                    cancel,
                    &mut |name, done, _| {
                        sink(&Progress {
                            stage: &format!("verifying {name}"),
                            completed_bytes: walk_total + done.min(derived.install_bytes),
                            total_bytes: overall_total,
                        })
                    },
                )?;
                write_derived_receipt(profile, derived, staging, outputs)
            },
        )?;
        let installed = InstalledModel {
            alias: identity.alias.clone(),
            repo: identity.repository.clone(),
            revision: identity.revision.clone(),
            path: destination.clone(),
            family: profile.family.clone(),
            install_bytes: crate::directory_bytes(&destination),
            installed_on: crate::install::today(),
            status: "unverified".into(),
            kind: None,
            modality: crate::ModelModality::Audio,
            variant: None,
        };
        // A failure here leaves the finished install in place: it is 22 GB
        // and a later listing picks up an unrecorded install that has a
        // valid receipt.
        store.record(&installed).map_err(|e| {
            format!(
                "recording the audio install failed: {e}; the install is at {} and a listing will find it",
                destination.display()
            )
        })?;
        report(overall_total, "installed");
        Ok(installed)
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
        // The speech probe knows the Whisper and Parakeet layouts a Rust
        // session opens. Swift-hosted rows never reach one, and their Core ML
        // layouts would fail it, so they get no speech receipt.
        if identity.task == AudioTask::SpeechToText
            && !profile.capabilities.backend.starts_with("swift_")
        {
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
        let mut layout = match &profile.install_subdir {
            Some(subdir) => Some(StagedLayout::arrange(
                &receipt.staging_root,
                &source_root,
                subdir,
            )?),
            None => None,
        };
        let publish_root = layout
            .as_ref()
            .map_or_else(|| source_root.clone(), |l| l.root.clone());
        let native_receipt = new_audio_receipt(profile);
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(publish_root.join(RECEIPT_FILENAME))
            .map_err(|e| format!("creating audio receipt: {e}"))?;
        use std::io::Write;
        file.write_all(&serde_json::to_vec_pretty(&native_receipt).map_err(|e| e.to_string())?)
            .and_then(|_| file.sync_all())
            .map_err(|e| format!("writing audio receipt: {e}"))?;
        cancel.checkpoint().map_err(|e| e.to_string())?;
        std::fs::create_dir_all(store.audio_models_dir())
            .map_err(|e| format!("creating audio namespace: {e}"))?;
        std::fs::rename(&publish_root, &destination)
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
            std::fs::rename(&destination, &publish_root).map_err(|rollback| {
                format!("recording audio install failed: {e}; restoring staging failed: {rollback}")
            })?;
            return Err(format!("recording audio install failed: {e}"));
        }
        if let Some(layout) = layout.as_mut() {
            layout.published = true;
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
        let receipt = read_receipt(profile, &path)?;
        verify_install(profile, &path, &receipt, false, None, None)?;
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
        let profile = self.profile(identity)?;
        let receipt = read_receipt(profile, &path)?;
        verify_install(profile, &path, &receipt, true, None, Some(&mut observer))?;
        Ok(path)
    }

    /// Full byte verification for the runtime-open boundary, separate from picker resolution.
    pub fn verify(
        &self,
        store: &Store,
        identity: &AudioProfileIdentity,
    ) -> Result<PathBuf, String> {
        let path = self.resolve(store, identity)?;
        let profile = self.profile(identity)?;
        let receipt = read_receipt(profile, &path)?;
        verify_install(profile, &path, &receipt, true, None, None)?;
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
                                if read_receipt(profile, &path)
                                    .and_then(|r| {
                                        verify_install(profile, &path, &r, false, None, None)
                                    })
                                    .is_ok()
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

/// A verified staging tree moved under a row's `install_subdir` inside
/// `<staging>/publish`, so publication stays one rename. Dropped before
/// `published` is set, it moves the tree back to `source-0000`: the caller
/// owns a complete staging tree after any failure, layout or not.
struct StagedLayout {
    root: PathBuf,
    assets: PathBuf,
    source: PathBuf,
    published: bool,
}

impl StagedLayout {
    fn arrange(staging_root: &Path, source: &Path, subdir: &str) -> Result<Self, String> {
        let root = staging_root.join("publish");
        std::fs::create_dir(&root).map_err(|e| format!("staging audio layout: {e}"))?;
        let assets = root.join(subdir);
        if let Some(parent) = assets.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("staging audio layout: {e}"))?;
        }
        std::fs::rename(source, &assets).map_err(|e| format!("staging audio layout: {e}"))?;
        Ok(Self {
            root,
            assets,
            source: source.to_path_buf(),
            published: false,
        })
    }
}

impl Drop for StagedLayout {
    fn drop(&mut self) {
        if !self.published {
            let _ = std::fs::rename(&self.assets, &self.source);
        }
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
const REVIEWED_ADAPTERS: &[&str] = &[
    "fish_drama3",
    "megatts3",
    "omnivoice",
    "parakeet_tdt",
    "parakeet_unified",
    "pyannote_community1",
];

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
                "en", "zh", "ja", "ko", "de", "fr", "es", "it", "pt", "ru", "ar", "hi", "nl", "pl",
                "tr", "vi", "th", "id", "uk", "sv", "fa", "cs", "el", "he",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            voices: vec![],
            max_input_seconds: None,
            supports_lyrics: false,
            operations: vec!["synthesize".into(), "voice_clone".into()],
            timing: None,
            backend: "metal".into(),
            cancellation: "diffusion_step".into(),
            can_run: true,
            unavailable_reason: None,
            compute: None,
            http_api: true,
        };
    }
    if family == "fish_drama3" {
        // Metal only: the headless trunk, the tower and the codec all run on
        // the GPU, and there is no portable fallback. It speaks without a
        // reference too; a reference makes it a clone, which needs consent
        // and the reference's transcript at the runtime boundary.
        return AudioCapabilities {
            languages: vec![],
            voices: vec![],
            max_input_seconds: None,
            supports_lyrics: false,
            operations: vec!["synthesize".into(), "voice_clone".into()],
            timing: None,
            backend: "metal".into(),
            cancellation: "frame".into(),
            can_run: true,
            unavailable_reason: None,
            compute: None,
            http_api: true,
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
            compute: None,
            http_api: true,
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
            // The runtime loads the encoder, which carries the run, with
            // CPU+GPU units (`ParakeetCoreMlOptions::default`). A server test,
            // the first crate that sees both, holds the two together.
            compute: Some("gpu".into()),
            http_api: true,
        };
    }
    // Swift-hosted Core ML: FluidAudio 0.15.5 inside the app. Rust installs
    // and verifies these rows, the app runs them, and neither the Rust session
    // API nor an HTTP server can. `compute` is the placement measured with
    // `coreml_plan` at each pinned revision (docs/FLUIDAUDIO_PHASE0.md),
    // not the units FluidAudio requests: Silero VAD asks for the Neural
    // Engine and runs on the CPU. Re-measure on a pin bump.
    let fluid = match family {
        "fluidaudio_silero_vad" => Some(("detect_speech", "cpu", "chunk", None)),
        "fluidaudio_diarizer" => Some(("diarize", "mixed", "job", None)),
        "fluidaudio_kokoro_ane" => Some(("synthesize", "mixed", "sentence", None)),
        "fluidaudio_pocket_tts" => Some(("synthesize", "gpu", "chunk", None)),
        "fluidaudio_sensevoice" | "fluidaudio_paraformer_zh" => {
            Some(("transcribe", "ane", "window", Some("window")))
        }
        "fluidaudio_parakeet_ja" => Some(("transcribe", "ane", "window", Some("token"))),
        _ => None,
    };
    if let Some((operation, compute, cancellation, timing)) = fluid {
        return AudioCapabilities {
            languages: vec![],
            voices: vec![],
            max_input_seconds: None,
            supports_lyrics: false,
            operations: vec![operation.into()],
            timing: timing.map(str::to_string),
            backend: "swift_coreml".into(),
            cancellation: cancellation.into(),
            can_run: false,
            unavailable_reason: Some(
                "Runs only in the TurboSpark app through FluidAudio (Core ML). The Rust session API and the HTTP server do not execute this family."
                    .into(),
            ),
            compute: Some(compute.into()),
            http_api: false,
        };
    }
    if family == "parakeet_unified" {
        // Live transcription only (ts_audio_stream_*): batch opens refuse
        // it, so "transcribe" is deliberately absent. English only.
        let os_ok = turbospark_coreml::os_version().0 >= 15;
        return AudioCapabilities {
            languages: vec![],
            voices: vec![],
            max_input_seconds: None,
            supports_lyrics: false,
            operations: vec!["stream_transcribe".into()],
            timing: Some("token".into()),
            backend: "coreml".into(),
            cancellation: "chunk".into(),
            can_run: os_ok,
            unavailable_reason: (!os_ok)
                .then(|| "Parakeet on Core ML requires macOS 15 or newer".into()),
            // The int8 encoder carries the run and loads with CPU+Neural
            // Engine units (`UnifiedOptions::default`), held by a server test.
            compute: Some("ane".into()),
            http_api: true,
        };
    }
    if family == "pyannote_community1" {
        // Offline speaker diarization (who spoke when). The programs declare
        // macOS 14, but the runtime refuses it: FluidAudio reports Core ML's
        // CPU path crashing in BNNS during this pipeline there.
        let os_ok = turbospark_coreml::os_version().0 >= 15;
        return AudioCapabilities {
            languages: vec![],
            voices: vec![],
            max_input_seconds: None,
            supports_lyrics: false,
            operations: vec!["diarize".into()],
            // Turn boundaries on the segmentation frame grid (16.875 ms).
            timing: Some("segment".into()),
            backend: "coreml".into(),
            cancellation: "window".into(),
            can_run: os_ok,
            unavailable_reason: (!os_ok)
                .then(|| "Speaker diarization on Core ML requires macOS 15 or newer".into()),
            compute: Some("mixed".into()),
            http_api: true,
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
        compute: None,
        http_api: true,
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
        // Installed only by this catalog, which records neither field.
        AudioTask::Diarization | AudioTask::SpeechDetection => {
            row.kind.is_none() && row.variant.is_none()
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
    // Rows with a fixed layout postdate receipts: no receiptless install of
    // one was ever published, so nothing of that shape can be adopted.
    if profile.install_subdir.is_some() {
        return Err(format!(
            "{} installs only through the catalog; delete the folder and download it again",
            profile.identity.alias
        ));
    }
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
        derived: None,
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
    install_subdir: Option<&str>,
) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let mut ordered = assets.to_vec();
    ordered.sort_by(|a, b| a.path.cmp(&b.path));
    // The layout joins the hash only when set, so every row without one
    // keeps the fingerprint saved selections and receipts already carry.
    let bytes = match install_subdir {
        None => serde_json::to_vec(&(ordered, frontend)),
        Some(subdir) => serde_json::to_vec(&(ordered, frontend, subdir)),
    }
    .map_err(|e| e.to_string())?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

/// One to three plain relative folder names, so a row can lay out a fixed
/// loader directory but never escape the install.
fn safe_install_subdir(value: &str) -> bool {
    let parts: Vec<&str> = value.split('/').collect();
    (1..=3).contains(&parts.len())
        && parts.iter().all(|part| {
            !part.is_empty()
                && *part != "."
                && *part != ".."
                && part
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-')
        })
}

/// Where a profile's assets live inside an install: the install root, or
/// `install_subdir` under it. Every folder on the way must be a real
/// directory, so the subdirectory cannot redirect reads through a symlink.
fn asset_root(profile: &AudioProfile, root: &Path) -> Result<PathBuf, String> {
    let mut dir = root.to_path_buf();
    if let Some(subdir) = &profile.install_subdir {
        require_directory(&dir)?;
        for part in subdir.split('/') {
            dir.push(part);
            require_directory(&dir)?;
        }
    }
    Ok(dir)
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
    verify_asset_bytes_observed(profile, root, hash, cancel, None, 0)
}

/// `extra_total` is bytes a caller verifies after the pinned assets (a derived
/// install's outputs), so progress counts both as one total.
fn verify_asset_bytes_observed(
    profile: &AudioProfile,
    root: &Path,
    hash: bool,
    cancel: Option<&CancelFlag>,
    mut observer: Option<&mut VerifyObserver<'_>>,
    extra_total: u64,
) -> Result<(), String> {
    require_directory(root)?;
    let total: u64 = profile.assets.iter().map(|a| a.size).sum::<u64>() + extra_total;
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
        verify_one_file(
            root,
            &asset.path,
            asset.size,
            &asset.sha256,
            "its immutable pin",
            hash,
            observer.as_deref_mut(),
            finished_bytes,
            total,
        )?;
        finished_bytes += asset.size;
    }
    Ok(())
}

/// One file under `root`: a regular file of exactly `size` bytes, and with
/// `hash` also of sha256 `expected`. `source` names what the digest is
/// compared with, for the error.
#[allow(clippy::too_many_arguments)]
fn verify_one_file(
    root: &Path,
    name: &str,
    size: u64,
    expected: &str,
    source: &str,
    hash: bool,
    mut observer: Option<&mut VerifyObserver<'_>>,
    finished_bytes: u64,
    total: u64,
) -> Result<(), String> {
    let path = root.join(name);
    // Check every ancestor so a nested asset cannot escape through a symlink.
    let mut parent = path.parent();
    while let Some(dir) = parent {
        if dir == root {
            break;
        }
        require_directory(dir)?;
        parent = dir.parent();
    }
    let meta =
        std::fs::symlink_metadata(&path).map_err(|e| format!("reading audio asset {name}: {e}"))?;
    if !meta.is_file() || meta.len() != size {
        return Err(format!(
            "audio asset {name} is absent, unsafe, or has changed size"
        ));
    }
    if hash {
        let digest = match observer.as_deref_mut() {
            None => Some(model_io::hash_file(&path, 1024 * 1024).map_err(|e| e.to_string())?),
            Some(observer) => {
                let mut reported = 0u64;
                let base = finished_bytes;
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
                    (observer.on_progress)(name, base + size, total);
                }
                digest
            }
        };
        match digest {
            None => return Err(VERIFY_CANCELLED.to_string()),
            Some(digest) if digest != expected => {
                return Err(format!(
                    "audio asset {name} SHA-256 does not match {source}"
                ));
            }
            Some(_) => {}
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
    match (&profile.derived, &receipt.derived) {
        (None, None) => {}
        (Some(spec), Some(made)) => {
            if made.tool != spec.tool || made.tool_version != spec.tool_version {
                return Err(format!(
                    "audio install was made by {} version {}, this build expects version {}; delete it and install again",
                    made.tool, made.tool_version, spec.tool_version
                ));
            }
            for required in drama3_install::REQUIRED_OUTPUTS {
                if !made.outputs.contains_key(required) {
                    return Err(format!("audio receipt does not record {required}"));
                }
            }
            for (name, entry) in &made.outputs {
                if !safe_asset_path(name)
                    || profile.assets.iter().any(|a| &a.path == name)
                    || name == RECEIPT_FILENAME
                    || !digest_shape(&entry.sha256)
                {
                    return Err(format!("audio receipt has an unsafe output entry {name:?}"));
                }
            }
        }
        _ => {
            return Err(
                "audio receipt and profile disagree about whether the install is derived".into(),
            )
        }
    }
    Ok(receipt)
}

/// The pinned assets, then a derived profile's recorded outputs. `hash`
/// compares sha256 as well as size. Progress covers both parts as one count.
fn verify_install(
    profile: &AudioProfile,
    root: &Path,
    receipt: &AudioInstallReceipt,
    hash: bool,
    cancel: Option<&CancelFlag>,
    mut observer: Option<&mut VerifyObserver<'_>>,
) -> Result<(), String> {
    let pinned_total: u64 = profile.assets.iter().map(|a| a.size).sum();
    let derived_total: u64 = receipt
        .derived
        .as_ref()
        .map_or(0, |d| d.outputs.values().map(|e| e.size).sum());
    let assets = asset_root(profile, root)?;
    verify_asset_bytes_observed(
        profile,
        &assets,
        hash,
        cancel,
        observer.as_deref_mut(),
        derived_total,
    )?;
    if let Some(made) = &receipt.derived {
        let mut finished = pinned_total;
        for (name, entry) in &made.outputs {
            if let Some(cancel) = cancel {
                cancel.checkpoint().map_err(|e| e.to_string())?;
            }
            if let Some(observer) = observer.as_deref() {
                if (observer.cancelled)() {
                    return Err(VERIFY_CANCELLED.to_string());
                }
            }
            verify_one_file(
                root,
                name,
                entry.size,
                &entry.sha256,
                "the install's receipt",
                hash,
                observer.as_deref_mut(),
                finished,
                pinned_total + derived_total,
            )?;
            finished += entry.size;
        }
    }
    Ok(())
}

fn write_derived_receipt(
    profile: &AudioProfile,
    derived: &AudioDerived,
    staging: &Path,
    outputs: std::collections::BTreeMap<String, model_io::speech_receipt::FileEntry>,
) -> Result<(), String> {
    use std::io::Write;
    let receipt = AudioInstallReceipt {
        derived: Some(DerivedReceipt {
            tool: derived.tool.clone(),
            tool_version: derived.tool_version,
            outputs,
        }),
        ..new_audio_receipt(profile)
    };
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(staging.join(RECEIPT_FILENAME))
        .map_err(|e| format!("creating audio receipt: {e}"))?;
    file.write_all(&serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?)
        .and_then(|_| file.sync_all())
        .map_err(|e| format!("writing audio receipt: {e}"))
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
        let catalog = AudioCatalog::embedded().unwrap();
        let alias = catalog
            .entries
            .iter()
            .find(|p| p.identity.task == AudioTask::TextToSpeech)
            .unwrap()
            .identity
            .alias
            .clone();
        staged(catalog, &alias, root)
    }

    /// A finished transfer for any row: real bytes for every asset, with the
    /// row's sizes, digests and fingerprint rewritten to match them.
    fn staged(
        mut catalog: AudioCatalog,
        alias: &str,
        root: &Path,
    ) -> (AudioCatalog, AudioProfileIdentity, DownloadReceipt) {
        let profile = catalog
            .entries
            .iter_mut()
            .find(|p| p.identity.alias == alias)
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
        profile.identity.asset_fingerprint = asset_fingerprint(
            &profile.assets,
            &profile.frontend,
            profile.install_subdir.as_deref(),
        )
        .unwrap();
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
        profile.identity.asset_fingerprint = asset_fingerprint(
            &profile.assets,
            &profile.frontend,
            profile.install_subdir.as_deref(),
        )
        .unwrap();
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
        verify_asset_bytes_observed(profile, &row.path, true, None, Some(&mut observer), 0)
            .unwrap();
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
        let err =
            verify_asset_bytes_observed(profile, &row.path, true, None, Some(&mut observer), 0)
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
        let err =
            verify_asset_bytes_observed(profile, &row.path, true, None, Some(&mut observer), 0)
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
        assert_eq!(catalog.entries().count(), 18);
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
            (
                AudioTask::SpeechToText,
                "parakeet-ultra-coreml",
                "FluidInference/parakeet-ultra-coreml",
                "95eaa59a39d4394f047a4dc5cce480388a60d1b6",
                16000,
                1,
            ),
            (
                AudioTask::SpeechToText,
                "parakeet-unified-en-coreml",
                "FluidInference/parakeet-unified-en-0.6b-coreml",
                "d32e972dd4315f1dc3f6be28fb2aab0ab3e80358",
                16000,
                1,
            ),
            (
                AudioTask::Diarization,
                "speaker-diarization-coreml",
                "FluidInference/speaker-diarization-coreml",
                "7eab74f3463c736418d0eb10296261ae581146c9",
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
        // Ultra shares Redux's component contract, so it is the same family
        // and decode flow; only its int8 encoder is larger.
        for (alias, encoder_bytes) in [
            ("parakeet-redux-coreml", 182_389_312u64),
            ("parakeet-ultra-coreml", 594_211_328),
        ] {
            let parakeet = catalog.get(alias).unwrap();
            assert_eq!(parakeet.family, "parakeet_tdt");
            assert_eq!(parakeet.capabilities.backend, "coreml");
            assert_eq!(parakeet.readiness, AudioReadiness::ImplementedUnqualified);
            assert!(parakeet.resident_memory_bytes > 3 * encoder_bytes);
            assert_eq!(
                parakeet.capabilities.can_run,
                turbospark_coreml::os_version().0 >= 15
            );
            assert!(parakeet
                .capabilities
                .operations
                .contains(&"transcribe".to_string()));
        }
        // The live model pins one streaming encoder, not the 11 GB repo, and
        // is never offered as a batch transcriber.
        let unified = catalog.get("parakeet-unified-en-coreml").unwrap();
        assert_eq!(unified.family, "parakeet_unified");
        assert_eq!(unified.capabilities.operations, ["stream_transcribe"]);
        assert_eq!(unified.capabilities.languages, ["en"]);
        assert!(unified.download_bytes < 700_000_000);
        assert_eq!(
            unified
                .assets
                .iter()
                .filter(|a| a.path.ends_with(".mlmodelc/model.mil"))
                .count(),
            3
        );
        assert!(unified.resident_memory_bytes > 3 * 589_979_840);
        assert_eq!(
            unified.capabilities.can_run,
            turbospark_coreml::os_version().0 >= 15
        );
        // Only the CC-BY-4.0 community-1 programs, the PLDA psi and the
        // attribution files; the legacy wespeaker and pyannote_segmentation
        // programs in the same repository are outside that license scope.
        let diarization = catalog.get("speaker-diarization-coreml").unwrap();
        assert_eq!(diarization.family, "pyannote_community1");
        assert_eq!(diarization.capabilities.operations, ["diarize"]);
        assert_eq!(diarization.capabilities.backend, "coreml");
        assert_eq!(diarization.capabilities.max_input_seconds, Some(1800));
        assert_eq!(
            diarization.readiness,
            AudioReadiness::ImplementedUnqualified
        );
        assert_eq!(diarization.assets.len(), 23);
        assert_eq!(diarization.download_bytes, 21_623_423);
        // Covers the measured 30-minute peak footprint.
        assert!(diarization.resident_memory_bytes > 858_195_576);
        for program in ["Segmentation", "FBank", "Embedding", "PldaRho"] {
            assert!(diarization
                .assets
                .iter()
                .any(|a| a.path == format!("{program}.mlmodelc/weights/weight.bin")));
        }
        for file in ["plda-parameters.json", "NOTICE.md", "LICENSE"] {
            assert!(diarization.assets.iter().any(|a| a.path == file));
        }
        assert!(diarization
            .assets
            .iter()
            .all(|a| !a.path.contains("wespeaker") && !a.path.starts_with("pyannote_")));
        assert_eq!(
            diarization.capabilities.can_run,
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

    /// The FluidAudio speech rows: pinned like any row, app-only, with the
    /// measured placement, and the FunASR terms where they apply.
    #[test]
    fn fluidaudio_speech_rows_are_pinned_app_only_and_carry_their_terms() {
        let catalog = AudioCatalog::embedded().unwrap();
        for (alias, family, repo, revision, files, ack) in [
            (
                "parakeet-ja-coreml",
                "fluidaudio_parakeet_ja",
                "FluidInference/parakeet-0.6b-ja-coreml",
                "2952296ff1da4a6d6a7aec545e226367db80c612",
                21,
                None,
            ),
            (
                "paraformer-zh-coreml",
                "fluidaudio_paraformer_zh",
                "FluidInference/paraformer-large-zh-coreml",
                "9b4faaa06e476f38ab4a6e011a4467629df09059",
                17,
                Some("funasr_model_license"),
            ),
        ] {
            let p = catalog.get(alias).unwrap();
            assert_eq!(p.identity.task, AudioTask::SpeechToText, "{alias}");
            assert_eq!(p.family, family);
            assert_eq!(p.identity.repository, repo);
            assert_eq!(p.identity.revision, revision);
            assert_eq!(p.assets.len(), files, "{alias}");
            assert_eq!(p.capabilities.backend, "swift_coreml");
            assert_eq!(p.capabilities.compute.as_deref(), Some("ane"));
            assert!(
                !p.capabilities.can_run && !p.capabilities.http_api,
                "{alias}"
            );
            assert!(p
                .capabilities
                .unavailable_reason
                .as_deref()
                .unwrap()
                .contains("FluidAudio"));
            assert_eq!(p.readiness, AudioReadiness::RuntimePending);
            assert_eq!(p.install_subdir, None);
            assert!(p.assets.iter().any(|a| a.path == "vocab.json"), "{alias}");
            // Compiled programs only: Core ML loads the .mlmodelc beside a
            // package source, so a source would be dead weight in the install.
            assert!(p.assets.iter().all(|a| !a.path.contains(".mlpackage/")));
            let ids: Vec<_> = p.acknowledgments.iter().map(|a| a.id.as_str()).collect();
            assert_eq!(ids, ack.into_iter().collect::<Vec<_>>(), "{alias}");
            let weights: u64 = p
                .assets
                .iter()
                .filter(|a| a.path.ends_with("/weights/weight.bin"))
                .map(|a| a.size)
                .sum();
            assert!(weights > 0, "{alias}");
            assert_eq!(
                p.resident_memory_bytes,
                weights * 3 + 16_000 * 4 * 1800 + (256 << 20),
                "{alias}"
            );
        }
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
                asset_fingerprint(&profile.assets, &changed, None).unwrap(),
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
            asset_fingerprint(&assets, &None, None).unwrap(),
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

    /// The embedded Drama 3 profile with its pins swapped for the fixture's
    /// bytes, so a derived install runs end to end without a network.
    fn derived_fixture(
        root: &Path,
    ) -> (
        AudioCatalog,
        AudioProfileIdentity,
        Store,
        drama3_install::test_support::Memory,
        drama3_install::Pins,
    ) {
        let (source, pins) = drama3_install::test_support::fixture();
        let mut catalog = AudioCatalog::embedded().unwrap();
        let profile = catalog
            .entries
            .iter_mut()
            .find(|p| p.identity.alias == drama3_install::ALIAS)
            .unwrap();
        for asset in &mut profile.assets {
            let pin = std::iter::once(&pins.config)
                .chain(&pins.small)
                .chain([&pins.codec])
                .find(|p| p.name == asset.path)
                .unwrap();
            asset.size = pin.bytes;
            asset.sha256 = pin.sha256.clone();
        }
        profile.identity.asset_fingerprint = asset_fingerprint(
            &profile.assets,
            &profile.frontend,
            profile.install_subdir.as_deref(),
        )
        .unwrap();
        let identity = profile.identity.clone();
        (
            catalog,
            identity,
            Store::new(root.join("store")),
            source,
            pins,
        )
    }

    fn derived_install(
        catalog: &AudioCatalog,
        identity: &AudioProfileIdentity,
        store: &Store,
        source: &drama3_install::test_support::Memory,
        pins: &drama3_install::Pins,
        progress: &mut dyn FnMut(HubDownloadProgress),
        cancel: &CancelFlag,
    ) -> Result<InstalledModel, String> {
        let derived = catalog.profile(identity)?.derived.clone().unwrap();
        catalog.install_derived_from(
            source,
            pins,
            u64::MAX,
            store,
            identity,
            &derived,
            progress,
            cancel,
        )
    }

    #[test]
    fn the_drama3_profile_is_a_derived_metal_clone_profile() {
        let catalog = AudioCatalog::embedded().unwrap();
        let profile = catalog.get(drama3_install::ALIAS).unwrap();
        assert_eq!(profile.family, "fish_drama3");
        assert_eq!(profile.identity.task, AudioTask::TextToSpeech);
        assert_eq!(profile.identity.repository, drama3_install::REPO);
        assert_eq!(profile.identity.revision, drama3_install::REVISION);
        assert_eq!(profile.pcm_format.sample_rate, 44_100);
        assert_eq!(profile.pcm_format.channels, 1);
        let derived = profile.derived.as_ref().expect("a derived profile");
        assert_eq!(derived.tool, drama3_install::TOOL);
        assert_eq!(derived.working_bytes, drama3_install::WORKING_BYTES);
        // Only the five root files are pinned; the shards are read by the
        // converter and counted in the download size.
        let names: Vec<_> = profile.assets.iter().map(|a| a.path.as_str()).collect();
        assert_eq!(
            names,
            [
                "config.json",
                "LICENSE",
                "NOTICE",
                "tokenizer.json",
                "codec.safetensors"
            ]
        );
        let pinned: u64 = profile.assets.iter().map(|a| a.size).sum();
        assert_eq!(
            profile.download_bytes,
            pinned + drama3_install::shard_total_bytes()
        );
        assert_eq!(
            profile.capabilities.operations,
            ["synthesize", "voice_clone"]
        );
        assert_eq!(profile.capabilities.backend, "metal");
        assert!(profile.capabilities.can_run);
        assert_eq!(profile.readiness, AudioReadiness::ImplementedUnqualified);
        assert_eq!(profile.resident_memory_bytes, 8_845_165_754);
        // Other profiles are not derived, and say nothing about it on the wire.
        let other = catalog.get("megatts3-f32").unwrap();
        assert!(other.derived.is_none());
        assert!(!serde_json::to_string(other).unwrap().contains("derived"));
        assert!(serde_json::to_string(profile)
            .unwrap()
            .contains("\"derived\""));
        // The license terms are accepted before any download.
        let [ack] = profile.acknowledgments.as_slice() else {
            panic!("expected one acknowledgment");
        };
        assert_eq!(ack.id, "fish_audio_community_license");
        for term in [
            "USD 500,000",
            "hosted service",
            "LICENSE and NOTICE",
            "Built with Fish Audio",
            "non-commercial",
        ] {
            assert!(ack.summary.contains(term), "the terms omit {term:?}");
        }
        assert!(ack.summary.is_ascii());
        let families = family_capabilities().unwrap();
        let row = families
            .iter()
            .find(|f| f.family == "tts/fish_drama3")
            .unwrap();
        assert_eq!(row.operations, ["synthesize", "voice_clone"]);
        assert!(row.can_run);
    }

    #[test]
    fn a_derived_install_is_refused_without_the_license_before_any_work() {
        let root = Scratch::new();
        let (catalog, identity, store, _, _) = derived_fixture(&root.0);
        let err = catalog
            .install_acknowledged(
                &Client::new(),
                &Catalog::embedded().unwrap(),
                &store,
                &identity,
                &[],
                &mut |_| {},
                &CancelFlag::new(),
            )
            .unwrap_err();
        assert!(err.contains("fish_audio_community_license"), "{err}");
        assert!(!store.audio_install_path(&identity.alias).exists());
        assert!(!store.root().exists(), "nothing was created");
    }

    #[test]
    fn a_derived_install_publishes_a_receipt_that_resolves_and_verifies() {
        let root = Scratch::new();
        let (catalog, identity, store, source, pins) = derived_fixture(&root.0);
        let mut reports = Vec::new();
        let installed = derived_install(
            &catalog,
            &identity,
            &store,
            &source,
            &pins,
            &mut |p| reports.push(p),
            &CancelFlag::new(),
        )
        .unwrap();
        assert_eq!(installed.path, store.audio_install_path(&identity.alias));
        assert_eq!(installed.family, "fish_drama3");
        assert_eq!(installed.status, "unverified");
        // One monotone bar over a fixed total, ending full.
        let total = reports[0].total_bytes;
        let mut last = 0;
        for r in &reports {
            assert_eq!(r.total_bytes, total);
            assert!(r.completed_bytes >= last, "{:?}", r.current_path);
            last = r.completed_bytes;
        }
        assert_eq!(last, total);
        assert!(reports.iter().any(|r| r
            .current_path
            .as_deref()
            .is_some_and(|p| p.starts_with("verifying "))));

        // The receipt records the pinned files and the converter's outputs.
        let wire: serde_json::Value =
            serde_json::from_slice(&fs::read(installed.path.join(RECEIPT_FILENAME)).unwrap())
                .unwrap();
        assert_eq!(wire["derived"]["tool"], drama3_install::TOOL);
        for output in drama3_install::REQUIRED_OUTPUTS {
            assert!(
                wire["derived"]["outputs"][output]["sha256"].is_string(),
                "{output}"
            );
        }
        assert!(wire["derived"]["outputs"]["config.json"].is_null());

        assert_eq!(catalog.resolve(&store, &identity).unwrap(), installed.path);
        assert_eq!(catalog.verify(&store, &identity).unwrap(), installed.path);
        // Verification counts the outputs in its progress total.
        let mut verify_total = 0;
        let mut verify_last = 0;
        catalog
            .verify_observed(&store, &identity, &|| false, &mut |_, done, all| {
                verify_total = all;
                verify_last = done;
            })
            .unwrap();
        let outputs: u64 = wire["derived"]["outputs"]
            .as_object()
            .unwrap()
            .values()
            .map(|e| e["size"].as_u64().unwrap())
            .sum();
        let pinned: u64 = catalog
            .get(&identity.alias)
            .unwrap()
            .assets
            .iter()
            .map(|a| a.size)
            .sum();
        assert_eq!(verify_total, pinned + outputs);
        assert_eq!(verify_last, verify_total);
        assert_eq!(catalog.installed(&store).unwrap(), vec![installed.clone()]);
        // The staging directory is gone, and the install can be deleted.
        assert!(!drama3_install::staging_dir(&installed.path)
            .unwrap()
            .exists());
        catalog.delete(&store, &identity).unwrap();
        assert!(!installed.path.exists());
    }

    #[test]
    fn a_derived_install_notices_damage_to_what_upstream_does_not_pin() {
        let root = Scratch::new();
        let (catalog, identity, store, source, pins) = derived_fixture(&root.0);
        let installed = derived_install(
            &catalog,
            &identity,
            &store,
            &source,
            &pins,
            &mut |_| {},
            &CancelFlag::new(),
        )
        .unwrap();
        let side = installed.path.join("tts_side.safetensors");
        let original = fs::read(&side).unwrap();

        // Same size, different bytes: the cheap check passes, the full one fails.
        let mut flipped = original.clone();
        let last = flipped.len() - 1;
        flipped[last] ^= 0xff;
        fs::write(&side, &flipped).unwrap();
        assert!(catalog.resolve(&store, &identity).is_ok());
        let err = catalog.verify(&store, &identity).unwrap_err();
        assert!(
            err.contains("tts_side.safetensors") && err.contains("receipt"),
            "{err}"
        );

        // A truncated output fails even the cheap check.
        fs::write(&side, &original[..original.len() - 1]).unwrap();
        let err = catalog.resolve(&store, &identity).unwrap_err();
        assert!(err.contains("changed size"), "{err}");

        // A missing one too, and the damaged install is still deletable.
        fs::remove_file(&side).unwrap();
        assert!(catalog.resolve(&store, &identity).is_err());
        assert!(catalog.installed_report(&store).installed.is_empty());
        catalog.delete(&store, &identity).unwrap();
        assert!(!installed.path.exists());
    }

    #[test]
    fn a_derived_receipt_must_match_its_profile() {
        let root = Scratch::new();
        let (catalog, identity, store, source, pins) = derived_fixture(&root.0);
        let installed = derived_install(
            &catalog,
            &identity,
            &store,
            &source,
            &pins,
            &mut |_| {},
            &CancelFlag::new(),
        )
        .unwrap();
        let path = installed.path.join(RECEIPT_FILENAME);
        let good = fs::read(&path).unwrap();
        let edit = |change: &dyn Fn(&mut serde_json::Value)| {
            let mut wire: serde_json::Value = serde_json::from_slice(&good).unwrap();
            change(&mut wire);
            fs::write(&path, serde_json::to_vec(&wire).unwrap()).unwrap();
            catalog.resolve(&store, &identity).unwrap_err()
        };
        let err = edit(&|w| {
            w["derived"]["outputs"]
                .as_object_mut()
                .unwrap()
                .remove("tts_side.safetensors");
        });
        assert!(
            err.contains("does not record tts_side.safetensors"),
            "{err}"
        );
        let err = edit(&|w| w["derived"]["tool_version"] = 99.into());
        assert!(
            err.contains("version 99") && err.contains("install again"),
            "{err}"
        );
        let err = edit(&|w| w["derived"]["tool"] = "something_else".into());
        assert!(err.contains("something_else"), "{err}");
        // An output named like a pinned file, or outside the install, is unsafe.
        let entry = serde_json::json!({"size": 1, "sha256": "a".repeat(64)});
        let err = edit(&|w| w["derived"]["outputs"]["config.json"] = entry.clone());
        assert!(err.contains("unsafe output entry"), "{err}");
        let err = edit(&|w| w["derived"]["outputs"]["../escape"] = entry.clone());
        assert!(err.contains("unsafe output entry"), "{err}");
        // A derived profile with no derived section is not an intact install.
        let err = edit(&|w| {
            w.as_object_mut().unwrap().remove("derived");
        });
        assert!(err.contains("disagree"), "{err}");
        fs::write(&path, &good).unwrap();
        assert!(catalog.resolve(&store, &identity).is_ok());
    }

    #[test]
    fn a_pinned_profile_cannot_carry_a_derived_section() {
        let root = Scratch::new();
        let store = Store::new(root.0.join("store"));
        let (catalog, identity, receipt) = fixture(&root.0);
        let installed = catalog
            .publish_verified(&store, &identity, &receipt, &CancelFlag::new())
            .unwrap();
        let path = installed.path.join(RECEIPT_FILENAME);
        let mut wire: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        wire["derived"] = serde_json::json!({
            "tool": "fish_drama3_intake",
            "tool_version": 1,
            "outputs": {},
        });
        fs::write(&path, serde_json::to_vec(&wire).unwrap()).unwrap();
        let err = catalog.resolve(&store, &identity).unwrap_err();
        assert!(err.contains("disagree"), "{err}");
    }

    #[test]
    fn a_cancelled_derived_install_leaves_nothing_in_the_store() {
        let root = Scratch::new();
        let (catalog, identity, store, mut source, pins) = derived_fixture(&root.0);
        let cancel = CancelFlag::new();
        source.cancel_on_root = Some((2, cancel.clone()));
        let err = derived_install(
            &catalog,
            &identity,
            &store,
            &source,
            &pins,
            &mut |_| {},
            &cancel,
        )
        .unwrap_err();
        assert!(err.contains("cancelled"), "{err}");
        let destination = store.audio_install_path(&identity.alias);
        assert!(!destination.exists());
        assert!(!drama3_install::staging_dir(&destination).unwrap().exists());
        assert!(catalog.installed(&store).unwrap().is_empty());
    }

    #[test]
    fn a_derived_install_does_not_replace_an_existing_one() {
        let root = Scratch::new();
        let (catalog, identity, store, source, pins) = derived_fixture(&root.0);
        let installed = derived_install(
            &catalog,
            &identity,
            &store,
            &source,
            &pins,
            &mut |_| {},
            &CancelFlag::new(),
        )
        .unwrap();
        let calls = source.root_calls.load(std::sync::atomic::Ordering::SeqCst);
        let err = catalog
            .install_acknowledged(
                &Client::new(),
                &Catalog::embedded().unwrap(),
                &store,
                &identity,
                &["fish_audio_community_license".to_string()],
                &mut |_| {},
                &CancelFlag::new(),
            )
            .unwrap_err();
        assert!(err.contains("already exists"), "{err}");
        assert_eq!(
            source.root_calls.load(std::sync::atomic::Ordering::SeqCst),
            calls
        );
        assert_eq!(catalog.resolve(&store, &identity).unwrap(), installed.path);
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
    const FLUID_FAMILIES: [&str; 7] = [
        "fluidaudio_silero_vad",
        "fluidaudio_diarizer",
        "fluidaudio_kokoro_ane",
        "fluidaudio_pocket_tts",
        "fluidaudio_sensevoice",
        "fluidaudio_paraformer_zh",
        "fluidaudio_parakeet_ja",
    ];

    /// One generic row for a Swift-hosted family, with a compiled weight for
    /// the memory estimate to count.
    fn fluid_manifest(
        family: &str,
        task: &str,
        install_subdir: Option<&str>,
        max_input_seconds: Option<u32>,
    ) -> String {
        let files = [
            "Model.mlmodelc/coremldata.bin",
            "Model.mlmodelc/weights/weight.bin",
            "vocab.json",
        ];
        let assets: Vec<_> = files
            .iter()
            .enumerate()
            .map(|(i, path)| serde_json::json!({"path": path, "size": 1000 * (i + 1), "sha256": format!("{:064x}", i + 1)}))
            .collect();
        let mut metadata = serde_json::json!({
            "task": task, "family": family, "display_name": "Fluid test",
            "required_files": files, "sample_rate": 16000, "channels": 1,
            "languages": ["auto"], "voices": [], "max_input_seconds": max_input_seconds
        });
        if let Some(subdir) = install_subdir {
            metadata["install_subdir"] = subdir.into();
        }
        serde_json::json!({
            "schema_version": 1,
            "profiles": [{
                "alias": "fluid-test",
                "assets": assets,
                "source": {"repository": "FluidInference/test-coreml", "revision": "0".repeat(40)},
                "metadata": metadata
            }]
        })
        .to_string()
    }

    fn fluid_catalog(family: &str, task: &str, install_subdir: Option<&str>) -> AudioCatalog {
        AudioCatalog::from_manifest(&fluid_manifest(family, task, install_subdir, Some(1800)))
            .unwrap()
    }

    #[test]
    fn swift_hosted_families_are_app_only_with_a_measured_placement() {
        for family in FLUID_FAMILIES {
            let c = profile_capabilities(family);
            assert_eq!(c.backend, "swift_coreml", "{family}");
            assert!(!c.can_run && !c.http_api, "{family}");
            assert!(
                c.unavailable_reason
                    .as_deref()
                    .is_some_and(|r| r.contains("FluidAudio")),
                "{family}"
            );
            assert!(
                matches!(c.compute.as_deref(), Some("ane" | "gpu" | "cpu" | "mixed")),
                "{family}"
            );
            assert_eq!(c.operations.len(), 1, "{family}");
        }
        // Measured, not requested: Silero VAD asks for the Neural Engine and
        // Core ML plans every op on the CPU.
        assert_eq!(
            profile_capabilities("fluidaudio_silero_vad")
                .compute
                .as_deref(),
            Some("cpu")
        );
        // Families without an adapter stay attachable, as they were.
        assert!(profile_capabilities("spark_tts").http_api);
        let catalog = AudioCatalog::embedded().unwrap();
        assert!(catalog
            .entries()
            .any(|p| p.capabilities.backend == "swift_coreml"));
        for p in catalog.entries() {
            // App-only exactly when the app is the only runtime.
            assert_eq!(
                p.capabilities.http_api,
                p.capabilities.backend != "swift_coreml",
                "{}",
                p.identity.alias
            );
            // Every Core ML row, Rust or Swift hosted, names its placement.
            assert_eq!(
                p.capabilities.compute.is_some(),
                p.capabilities.backend.ends_with("coreml"),
                "{}",
                p.identity.alias
            );
        }
    }

    #[test]
    fn swift_hosted_rows_keep_their_reason_and_estimate_compiled_weights() {
        let catalog = AudioCatalog::from_manifest(&fluid_manifest(
            "fluidaudio_parakeet_ja",
            "speech_to_text",
            None,
            Some(1800),
        ))
        .unwrap();
        let p = catalog.get("fluid-test").unwrap();
        assert_eq!(p.readiness, AudioReadiness::RuntimePending);
        assert!(!p.capabilities.can_run);
        assert!(p
            .capabilities
            .unavailable_reason
            .as_deref()
            .unwrap()
            .contains("FluidAudio"));
        assert_eq!(p.capabilities.languages, ["auto"]);
        assert_eq!(p.capabilities.timing.as_deref(), Some("token"));
        // Three times the 2000-byte weight.bin, 30 minutes of 16 kHz f32, and
        // the reserve. Safetensors-based arms would count zero here.
        assert_eq!(
            p.resident_memory_bytes,
            3 * 2000 + 16_000 * 4 * 1800 + (256 << 20)
        );
        assert!(p.resident_memory_evidence.contains("compiled Core ML"));
        let vad = AudioCatalog::from_manifest(&fluid_manifest(
            "fluidaudio_silero_vad",
            "speech_detection",
            None,
            None,
        ))
        .unwrap();
        assert_eq!(
            vad.get("fluid-test").unwrap().resident_memory_bytes,
            3 * 2000 + (256 << 20)
        );
    }

    #[test]
    fn install_subdir_is_validated_and_joins_the_fingerprint_only_when_set() {
        for bad in [
            "",
            ".",
            "..",
            "a/../b",
            "/abs",
            "a//b",
            "a/",
            "a/b/c/d",
            "has space",
            "a\\b",
        ] {
            assert!(
                AudioCatalog::from_manifest(&fluid_manifest(
                    "fluidaudio_pocket_tts",
                    "text_to_speech",
                    Some(bad),
                    None
                ))
                .is_err(),
                "{bad:?}"
            );
        }
        let flat = fluid_catalog("fluidaudio_pocket_tts", "text_to_speech", None);
        let nested = fluid_catalog(
            "fluidaudio_pocket_tts",
            "text_to_speech",
            Some("Models/pocket-tts"),
        );
        let (flat, nested) = (
            flat.get("fluid-test").unwrap(),
            nested.get("fluid-test").unwrap(),
        );
        assert_eq!(nested.install_subdir.as_deref(), Some("Models/pocket-tts"));
        assert_ne!(
            flat.identity.asset_fingerprint,
            nested.identity.asset_fingerprint
        );
        assert_eq!(
            serde_json::to_value(nested).unwrap()["install_subdir"],
            "Models/pocket-tts"
        );
        assert!(serde_json::to_value(flat)
            .unwrap()
            .get("install_subdir")
            .is_none());
    }

    /// Saved selections and every install receipt carry these fingerprints.
    /// The expected values were computed outside Rust from audio_assets.json
    /// (compact JSON of the path-sorted assets and the frontend, SHA-256), so
    /// a change to the hashing of rows without a layout fails here.
    #[test]
    fn rows_without_a_layout_keep_their_fingerprints() {
        let catalog = AudioCatalog::embedded().unwrap();
        for (alias, fingerprint) in [
            (
                "whisper-base",
                "4422351069947e3a169e63f459b45f0c11603d074a27ca54c27afb0df67307cb",
            ),
            (
                "parakeet-redux-coreml",
                "ef38a6a0ac299b11ede6fefbf3fb2a0b9c121c99bcbe93c973839475d15d8935",
            ),
        ] {
            assert_eq!(
                catalog.get(alias).unwrap().identity.asset_fingerprint,
                fingerprint,
                "{alias}"
            );
        }
    }

    #[test]
    fn layout_rows_publish_under_their_subdir_then_resolve_verify_list_and_delete() {
        let root = Scratch::new();
        let store = Store::new(root.0.join("store"));
        let catalog = fluid_catalog(
            "fluidaudio_pocket_tts",
            "text_to_speech",
            Some("Models/pocket-tts"),
        );
        let (catalog, identity, receipt) = staged(catalog, "fluid-test", &root.0);
        let installed = catalog
            .publish_verified(&store, &identity, &receipt, &CancelFlag::new())
            .unwrap();
        let profile = catalog.get(&identity.alias).unwrap();
        let assets = installed.path.join("Models/pocket-tts");
        assert!(installed.path.join(RECEIPT_FILENAME).is_file());
        for asset in &profile.assets {
            assert!(assets.join(&asset.path).is_file(), "{}", asset.path);
            assert!(!installed.path.join(&asset.path).exists(), "{}", asset.path);
        }
        assert_eq!(catalog.resolve(&store, &identity).unwrap(), installed.path);
        assert_eq!(catalog.verify(&store, &identity).unwrap(), installed.path);
        assert_eq!(catalog.installed(&store).unwrap(), vec![installed.clone()]);
        // A flipped byte under the layout passes the size check only.
        let victim = assets.join(&profile.assets[0].path);
        let mut bytes = fs::read(&victim).unwrap();
        bytes[0] ^= 0xff;
        fs::write(&victim, &bytes).unwrap();
        assert!(catalog.resolve(&store, &identity).is_ok());
        assert!(catalog.verify(&store, &identity).is_err());
        bytes[0] ^= 0xff;
        fs::write(&victim, &bytes).unwrap();
        // No receiptless install of a layout row ever existed to adopt.
        assert!(catalog
            .adopt_legacy(&store, &identity, &CancelFlag::new())
            .is_err());
        catalog.delete(&store, &identity).unwrap();
        assert!(!installed.path.exists());
        assert!(catalog.installed(&store).unwrap().is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn layout_rows_refuse_a_symlinked_layout_folder() {
        let root = Scratch::new();
        let store = Store::new(root.0.join("store"));
        let catalog = fluid_catalog(
            "fluidaudio_pocket_tts",
            "text_to_speech",
            Some("Models/pocket-tts"),
        );
        let (catalog, identity, receipt) = staged(catalog, "fluid-test", &root.0);
        let installed = catalog
            .publish_verified(&store, &identity, &receipt, &CancelFlag::new())
            .unwrap();
        // The folder above the asset root, which the per-asset ancestor walk
        // does not reach.
        let inner = installed.path.join("Models");
        let outside = root.0.join("outside");
        fs::rename(&inner, &outside).unwrap();
        std::os::unix::fs::symlink(&outside, &inner).unwrap();
        assert!(catalog.resolve(&store, &identity).is_err());
        assert!(catalog.verify(&store, &identity).is_err());
    }

    #[test]
    fn layout_publication_failure_returns_the_complete_staging_tree() {
        let root = Scratch::new();
        let store = Store::new(root.0.join("store"));
        fs::create_dir_all(store.root().join("installed.json")).unwrap();
        let catalog = fluid_catalog(
            "fluidaudio_pocket_tts",
            "text_to_speech",
            Some("Models/pocket-tts"),
        );
        let (catalog, identity, receipt) = staged(catalog, "fluid-test", &root.0);
        assert!(catalog
            .publish_verified(&store, &identity, &receipt, &CancelFlag::new())
            .is_err());
        assert!(!store.audio_install_path(&identity.alias).exists());
        let source = receipt.staging_root.join("source-0000");
        for asset in &catalog.get(&identity.alias).unwrap().assets {
            assert!(source.join(&asset.path).is_file(), "{}", asset.path);
        }
    }

    /// Swift-hosted speech rows skip the Rust speech probe, which refuses
    /// their Core ML layout. The same row under a Rust family fails it.
    #[test]
    fn swift_hosted_speech_rows_publish_without_the_rust_speech_probe() {
        let root = Scratch::new();
        let store = Store::new(root.0.join("store"));
        let catalog = fluid_catalog("fluidaudio_parakeet_ja", "speech_to_text", None);
        let (catalog, identity, receipt) = staged(catalog, "fluid-test", &root.0);
        let installed = catalog
            .publish_verified(&store, &identity, &receipt, &CancelFlag::new())
            .unwrap();
        assert!(!installed
            .path
            .join(model_io::speech_receipt::SPEECH_RECEIPT_FILENAME)
            .exists());
        assert!(catalog.verify(&store, &identity).is_ok());

        let root = Scratch::new();
        let store = Store::new(root.0.join("store"));
        let catalog = fluid_catalog("spark_asr", "speech_to_text", None);
        let (catalog, identity, receipt) = staged(catalog, "fluid-test", &root.0);
        assert!(catalog
            .publish_verified(&store, &identity, &receipt, &CancelFlag::new())
            .is_err());
    }

    #[test]
    fn diarization_and_speech_detection_installs_are_listed_as_installed() {
        for (family, task) in [
            ("fluidaudio_diarizer", "diarization"),
            ("fluidaudio_silero_vad", "speech_detection"),
        ] {
            let root = Scratch::new();
            let store = Store::new(root.0.join("store"));
            let (catalog, identity, receipt) =
                staged(fluid_catalog(family, task, None), "fluid-test", &root.0);
            let installed = catalog
                .publish_verified(&store, &identity, &receipt, &CancelFlag::new())
                .unwrap();
            let report = catalog.installed_report(&store);
            assert!(
                report.incompatible.is_empty(),
                "{task}: {:?}",
                report.incompatible
            );
            assert_eq!(report.installed, vec![installed], "{task}");
        }
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
