//! Pinned embedding-model profiles and their install path.
//!
//! An embedding install is a directory of files under
//! `models/embeddings/<alias>` with an `embedding-receipt.json`. It is not
//! a registry row: embedding models are opened by path or alias from the
//! embedding fields (server, memory), and `resolve_model_arg` falls back to
//! this namespace. Keeping it out of the text registry keeps it out of the
//! chat model pickers, which would try to load it as a generator.
//!
//! Every asset is pinned by path, size and SHA-256 at an immutable
//! revision; the transfer verifies each one before the install is
//! published with a single rename.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{
    CancelFlag, Catalog, Client, DownloadReceipt, HubClient, HubDownloadProgress, Machine,
    PinnedArtifactPlan, PinnedSourceFile, PinnedSourceGroup, RepoRef, Store,
};

const EMBEDDED: &str = include_str!("embedding_models.json");
const SOURCE_ROLE: &str = "embedding";
pub const RECEIPT_FILENAME: &str = "embedding-receipt.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingAsset {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingProfile {
    pub alias: String,
    pub display_name: String,
    /// Runtime family, as `runtime::coreml::CoreMlLayout::family` names it.
    pub family: String,
    /// `coreml` for the Core ML exports.
    pub backend: String,
    pub repository: String,
    pub revision: String,
    pub license: String,
    /// Input kinds the model embeds into one space (`text`, `image`, `audio`).
    pub modalities: Vec<String>,
    pub dimensions: usize,
    pub matryoshka_dims: Vec<usize>,
    pub assets: Vec<EmbeddingAsset>,
}

impl EmbeddingProfile {
    pub fn download_bytes(&self) -> u64 {
        self.assets.iter().map(|a| a.size).sum()
    }

    /// Whether this machine can run the profile, and why not. Core ML
    /// exports need macOS 15 (multifunction packages); off macOS the
    /// runner is a stub and reports version 0.
    pub fn availability(&self) -> (bool, Option<String>) {
        match self.backend.as_str() {
            "coreml" if turbospark_coreml::os_version().0 >= 15 => (true, None),
            "coreml" => (
                false,
                Some(format!("{} requires macOS 15 or newer", self.display_name)),
            ),
            other => (false, Some(format!("unknown embedding backend {other:?}"))),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    profiles: Vec<EmbeddingProfile>,
}

/// Written into the install after every file is verified.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingReceipt {
    pub schema_version: u32,
    pub alias: String,
    pub repository: String,
    pub revision: String,
    pub family: String,
    pub files: BTreeMap<String, model_io::speech_receipt::FileEntry>,
}

fn safe_relative(path: &str) -> bool {
    !path.is_empty()
        && Path::new(path)
            .components()
            .all(|c| matches!(c, std::path::Component::Normal(_)))
}

fn hex64(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

#[derive(Debug, Clone)]
pub struct EmbeddingCatalog {
    profiles: Vec<EmbeddingProfile>,
}

impl EmbeddingCatalog {
    pub fn embedded() -> Result<Self, String> {
        Self::from_json(EMBEDDED)
    }

    fn from_json(json: &str) -> Result<Self, String> {
        let m: Manifest =
            serde_json::from_str(json).map_err(|e| format!("invalid embedding catalog: {e}"))?;
        if m.schema_version != 1 {
            return Err("embedding catalog must be schema 1".into());
        }
        let mut aliases = HashSet::new();
        for p in &m.profiles {
            if p.alias.is_empty()
                || !p
                    .alias
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
                || !aliases.insert(p.alias.clone())
            {
                return Err(format!(
                    "invalid or duplicate embedding alias {:?}",
                    p.alias
                ));
            }
            if p.revision.len() != 40 || !p.revision.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(format!("{} does not pin an immutable revision", p.alias));
            }
            if p.assets.is_empty() {
                return Err(format!("{} has no assets", p.alias));
            }
            let mut paths = HashSet::new();
            for a in &p.assets {
                if !safe_relative(&a.path)
                    || a.size == 0
                    || !hex64(&a.sha256)
                    || !paths.insert(&a.path)
                {
                    return Err(format!("{} has an invalid asset {:?}", p.alias, a.path));
                }
                if a.path == RECEIPT_FILENAME {
                    return Err(format!("{} pins the receipt name as an asset", p.alias));
                }
            }
            if p.matryoshka_dims.first() != Some(&p.dimensions)
                || p.matryoshka_dims
                    .iter()
                    .any(|&d| d == 0 || d > p.dimensions)
            {
                return Err(format!("{} has inconsistent dimensions", p.alias));
            }
        }
        Ok(Self {
            profiles: m.profiles,
        })
    }

    pub fn profiles(&self) -> &[EmbeddingProfile] {
        &self.profiles
    }

    pub fn get(&self, alias: &str) -> Option<&EmbeddingProfile> {
        self.profiles.iter().find(|p| p.alias == alias)
    }

    pub fn pinned_plan(&self, alias: &str) -> Result<PinnedArtifactPlan, String> {
        let p = self
            .get(alias)
            .ok_or_else(|| format!("no embedding profile named {alias:?}"))?;
        Ok(PinnedArtifactPlan {
            owner_id: format!("embedding:{alias}"),
            sources: vec![PinnedSourceGroup {
                role: SOURCE_ROLE.to_string(),
                repo: RepoRef::new(&p.repository, &p.revision),
                files: p
                    .assets
                    .iter()
                    .map(|a| PinnedSourceFile {
                        path: a.path.clone(),
                        expected_size: Some(a.size),
                        expected_sha256: Some(a.sha256.clone()),
                    })
                    .collect(),
            }],
        })
    }

    /// The install directory when it holds a receipt for this exact pin and
    /// every pinned file at its recorded size. A size check, not a re-hash:
    /// the transfer hashed every byte before publishing.
    pub fn installed_path(&self, store: &Store, alias: &str) -> Option<PathBuf> {
        let p = self.get(alias)?;
        let dir = store.embedding_install_path(alias);
        verify_install(p, &dir).ok().map(|_| dir)
    }

    /// Downloads, verifies and publishes one profile. Refuses when the
    /// install already exists.
    pub fn install(
        &self,
        client: &Client,
        catalog: &Catalog,
        store: &Store,
        alias: &str,
        progress: &mut dyn FnMut(HubDownloadProgress),
        cancel: &CancelFlag,
    ) -> Result<PathBuf, String> {
        let profile = self
            .get(alias)
            .ok_or_else(|| format!("no embedding profile named {alias:?}"))?;
        let plan = self.pinned_plan(alias)?;
        let destination = store.embedding_install_path(alias);
        if destination.exists() {
            return Err(format!(
                "{} already exists; remove it before pulling again",
                destination.display()
            ));
        }
        std::fs::create_dir_all(store.embedding_models_dir())
            .map_err(|e| format!("creating embedding store: {e}"))?;
        let hub = HubClient::new(
            client,
            catalog.clone(),
            Machine::default(),
            4096,
            model_io::ExpertCacheSlots::Auto,
        )
        .with_store(store.clone());
        let receipt = hub
            .download_pinned_artifacts(&plan, store.root(), progress, cancel)
            .map_err(|e| e.to_string())?;
        let result = (|| {
            cancel.checkpoint().map_err(|e| e.to_string())?;
            let staged = check_receipt(&receipt, profile)?;
            write_receipt(profile, &staged)?;
            std::fs::rename(&staged, &destination)
                .map_err(|e| format!("publishing {}: {e}", destination.display()))
        })();
        let _ = std::fs::remove_dir_all(&receipt.staging_root);
        result?;
        // Best effort: the published install holds these bytes now, and a
        // leftover cache costs disk, not correctness.
        let _ = receipt.discard_download_cache();
        Ok(destination)
    }
}

/// Checks the transfer returned exactly this profile's pinned files.
fn check_receipt(receipt: &DownloadReceipt, p: &EmbeddingProfile) -> Result<PathBuf, String> {
    if receipt.owner_id != format!("embedding:{}", p.alias) || receipt.sources.len() != 1 {
        return Err("embedding transfer returned an unexpected owner or source count".into());
    }
    let group = &receipt.sources[0];
    if group.role != SOURCE_ROLE
        || !group.repo.repo.eq_ignore_ascii_case(&p.repository)
        || group.repo.revision != p.revision
    {
        return Err("embedding transfer does not match the pinned repository".into());
    }
    let root = receipt.staging_root.join("source-0000");
    let mut remaining: HashSet<&str> = p.assets.iter().map(|a| a.path.as_str()).collect();
    for f in &group.files {
        let asset =
            p.assets.iter().find(|a| a.path == f.path).ok_or_else(|| {
                format!("embedding transfer returned a foreign file {:?}", f.path)
            })?;
        if !remaining.remove(f.path.as_str())
            || f.staged_path != root.join(&f.path)
            || f.size != asset.size
            || f.sha256 != asset.sha256
        {
            return Err(format!(
                "embedding transfer metadata does not match {}",
                f.path
            ));
        }
    }
    if !remaining.is_empty() {
        return Err("embedding transfer omitted pinned files".into());
    }
    Ok(root)
}

fn write_receipt(p: &EmbeddingProfile, dir: &Path) -> Result<(), String> {
    let receipt = EmbeddingReceipt {
        schema_version: 1,
        alias: p.alias.clone(),
        repository: p.repository.clone(),
        revision: p.revision.clone(),
        family: p.family.clone(),
        files: p
            .assets
            .iter()
            .map(|a| {
                (
                    a.path.clone(),
                    model_io::speech_receipt::FileEntry {
                        size: a.size,
                        sha256: a.sha256.clone(),
                    },
                )
            })
            .collect(),
    };
    let json = serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?;
    std::fs::write(dir.join(RECEIPT_FILENAME), json).map_err(|e| format!("writing receipt: {e}"))
}

/// The install holds a receipt for this exact pin, and every pinned file is
/// a regular file (no symlinks) at its recorded size. Extra files, such as
/// Core ML's `.compiled` cache, are allowed.
pub fn verify_install(p: &EmbeddingProfile, dir: &Path) -> Result<(), String> {
    let bytes = std::fs::read(dir.join(RECEIPT_FILENAME))
        .map_err(|e| format!("{}: no embedding receipt ({e})", dir.display()))?;
    let r: EmbeddingReceipt =
        serde_json::from_slice(&bytes).map_err(|e| format!("embedding receipt: {e}"))?;
    if r.alias != p.alias
        || r.revision != p.revision
        || !r.repository.eq_ignore_ascii_case(&p.repository)
    {
        return Err("embedding receipt names a different pin".into());
    }
    for a in &p.assets {
        let meta =
            std::fs::symlink_metadata(dir.join(&a.path)).map_err(|e| format!("{}: {e}", a.path))?;
        if !meta.is_file() || meta.len() != a.size {
            return Err(format!("{} is missing or the wrong size", a.path));
        }
        if r.files.get(&a.path).map(|f| &f.sha256) != Some(&a.sha256) {
            return Err(format!("receipt does not record {}", a.path));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_profile_is_pinned_and_consistent() {
        let c = EmbeddingCatalog::embedded().unwrap();
        let p = c.get("embeddinggemma-2-coreml").unwrap();
        assert_eq!(p.repository, "FluidInference/embeddinggemma-2-coreml");
        assert_eq!(p.revision, "3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4");
        assert_eq!(p.family, "embeddinggemma2");
        // Every file the runtime opens is pinned.
        for f in [
            "EmbeddingGemma2Text.mlpackage/Manifest.json",
            "embeddings.bf16",
            "tokenizer.json",
            "config.json",
            "EmbeddingGemma2Audio.mlpackage/Manifest.json",
            "EmbeddingGemma2Vision.mlpackage/Manifest.json",
            "position_embeddings.f16",
        ] {
            assert!(p.assets.iter().any(|a| a.path == f), "{f}");
        }
        let plan = c.pinned_plan(&p.alias).unwrap();
        assert_eq!(plan.sources[0].files.len(), p.assets.len());
        assert!(plan.sources[0]
            .files
            .iter()
            .all(|f| f.expected_size.is_some() && f.expected_sha256.is_some()));
        assert!(c.pinned_plan("owner/other").is_err());
    }

    #[test]
    fn manifest_refuses_unsafe_or_unpinned_rows() {
        let row = |alias: &str, rev: &str, path: &str| {
            format!(
                r#"{{"schema_version":1,"profiles":[{{"alias":"{alias}","display_name":"x","family":"f",
                "backend":"coreml","repository":"o/r","revision":"{rev}","license":"MIT",
                "modalities":["text"],"dimensions":8,"matryoshka_dims":[8,4],
                "assets":[{{"path":"{path}","size":1,"sha256":"{}"}}]}}]}}"#,
                "a".repeat(64)
            )
        };
        let rev = "0".repeat(40);
        assert!(EmbeddingCatalog::from_json(&row("ok-1", &rev, "a/b.bin")).is_ok());
        assert!(EmbeddingCatalog::from_json(&row("Bad", &rev, "a")).is_err());
        assert!(EmbeddingCatalog::from_json(&row("ok", "main", "a")).is_err());
        assert!(EmbeddingCatalog::from_json(&row("ok", &rev, "../a")).is_err());
        assert!(EmbeddingCatalog::from_json(&row("ok", &rev, "/a")).is_err());
        assert!(EmbeddingCatalog::from_json(&row("ok", &rev, RECEIPT_FILENAME)).is_err());
    }

    #[test]
    fn verify_install_checks_receipt_sizes_and_symlinks() {
        let dir = std::env::temp_dir().join(format!(
            "ts-embedding-install-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(dir.join("pkg")).unwrap();
        let p = EmbeddingProfile {
            alias: "t".into(),
            display_name: "t".into(),
            family: "f".into(),
            backend: "coreml".into(),
            repository: "o/r".into(),
            revision: "0".repeat(40),
            license: "MIT".into(),
            modalities: vec!["text".into()],
            dimensions: 8,
            matryoshka_dims: vec![8],
            assets: vec![EmbeddingAsset {
                path: "pkg/w.bin".into(),
                size: 3,
                sha256: "b".repeat(64),
            }],
        };
        assert!(verify_install(&p, &dir).is_err(), "no receipt");
        std::fs::write(dir.join("pkg/w.bin"), b"abc").unwrap();
        write_receipt(&p, &dir).unwrap();
        assert!(verify_install(&p, &dir).is_ok());
        // A compile cache next to the files is fine.
        std::fs::create_dir_all(dir.join(".compiled/x.mlmodelc")).unwrap();
        assert!(verify_install(&p, &dir).is_ok());
        std::fs::write(dir.join("pkg/w.bin"), b"abcd").unwrap();
        assert!(verify_install(&p, &dir).is_err(), "size changed");
        std::fs::remove_file(dir.join("pkg/w.bin")).unwrap();
        std::os::unix::fs::symlink("/etc/hosts", dir.join("pkg/w.bin")).unwrap();
        assert!(verify_install(&p, &dir).is_err(), "symlink");
        let mut other = p.clone();
        other.revision = "1".repeat(40);
        std::fs::remove_file(dir.join("pkg/w.bin")).unwrap();
        std::fs::write(dir.join("pkg/w.bin"), b"abc").unwrap();
        assert!(verify_install(&other, &dir).is_err(), "different pin");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
