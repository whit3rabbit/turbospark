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

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::pinned_dir::{PinnedDir, PinnedDirReceipt};
use crate::{CancelFlag, Catalog, Client, HubDownloadProgress, PinnedArtifactPlan, Status, Store};

pub use crate::pinned_dir::PinnedAsset;

const EMBEDDED: &str = include_str!("embedding_models.json");
const SOURCE_ROLE: &str = "embedding";
pub const RECEIPT_FILENAME: &str = "embedding-receipt.json";

pub type EmbeddingAsset = PinnedAsset;
/// Written into the install after every file is verified.
pub type EmbeddingReceipt = PinnedDirReceipt;

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
    /// Evidence, not intent. For an embedding profile, [`Status::Verified`]
    /// means every test named in [`Self::gates`] ran a real model installed
    /// from exactly this pin. The app highlights only verified profiles, so
    /// the loader refuses a verified row without gates.
    pub status: Status,
    /// Test targets (`crates/*/tests/<gate>.rs`) behind [`Self::status`].
    #[serde(default)]
    pub gates: Vec<String>,
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

    pub(crate) fn pinned(&self) -> PinnedDir<'_> {
        PinnedDir {
            kind: SOURCE_ROLE,
            receipt_name: RECEIPT_FILENAME,
            alias: &self.alias,
            repository: &self.repository,
            revision: &self.revision,
            family: &self.family,
            assets: &self.assets,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    profiles: Vec<EmbeddingProfile>,
}

#[derive(Debug, Clone)]
pub struct EmbeddingCatalog {
    profiles: Vec<EmbeddingProfile>,
}

impl EmbeddingCatalog {
    pub fn embedded() -> Result<Self, String> {
        Self::from_json(EMBEDDED)
    }

    pub(crate) fn from_json(json: &str) -> Result<Self, String> {
        let m: Manifest =
            serde_json::from_str(json).map_err(|e| format!("invalid embedding catalog: {e}"))?;
        if m.schema_version != 1 {
            return Err("embedding catalog must be schema 1".into());
        }
        let mut aliases = HashSet::new();
        for p in &m.profiles {
            if !aliases.insert(p.alias.clone()) {
                return Err(format!("duplicate embedding alias {:?}", p.alias));
            }
            p.pinned().validate()?;
            if p.matryoshka_dims.first() != Some(&p.dimensions)
                || p.matryoshka_dims
                    .iter()
                    .any(|&d| d == 0 || d > p.dimensions)
            {
                return Err(format!("{} has inconsistent dimensions", p.alias));
            }
            if p.status == Status::Verified && p.gates.is_empty() {
                return Err(format!("{} is marked verified but names no gate", p.alias));
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
        Ok(p.pinned().plan())
    }

    /// The install directory when it holds a receipt for this exact pin and
    /// every pinned file at its recorded size. A size check, not a re-hash:
    /// the transfer hashed every byte before publishing.
    pub fn installed_path(&self, store: &Store, alias: &str) -> Option<PathBuf> {
        let p = self.get(alias)?;
        let dir = store.embedding_install_path(alias);
        verify_install(p, &dir).ok().map(|_| dir)
    }

    /// What sits at `alias`'s install path: `current` (verifies against
    /// this pin), `stale` (an older pin or a damaged install of this alias,
    /// which a download replaces), `foreign` (not provably ours; a download
    /// refuses it), or `absent`. `None` for an unknown alias.
    pub fn install_state(&self, store: &Store, alias: &str) -> Option<&'static str> {
        let p = self.get(alias)?;
        Some(
            match p.pinned().existing(&store.embedding_install_path(alias)) {
                crate::pinned_dir::Existing::Current => "current",
                crate::pinned_dir::Existing::Absent => "absent",
                crate::pinned_dir::Existing::Replaceable(_) => "stale",
                crate::pinned_dir::Existing::Foreign(_) => "foreign",
            },
        )
    }

    /// Why `alias` has no verified install: an older pin, a damaged
    /// install, or nothing there. `None` when the install verifies.
    pub fn install_problem(&self, store: &Store, alias: &str) -> Option<String> {
        let p = self.get(alias)?;
        let dir = store.embedding_install_path(alias);
        match p.pinned().existing(&dir) {
            crate::pinned_dir::Existing::Current => None,
            crate::pinned_dir::Existing::Absent => Some(format!(
                "{} is not installed; download it first",
                p.display_name
            )),
            crate::pinned_dir::Existing::Replaceable(_) => Some(format!(
                "the installed {} is from an older revision or damaged; download it again",
                p.display_name
            )),
            crate::pinned_dir::Existing::Foreign(why) => Some(format!(
                "{} is not a TurboSpark install of {} ({why})",
                dir.display(),
                p.alias
            )),
        }
    }

    /// Downloads, verifies and publishes one profile at `revision`, which
    /// must be the catalog's pin (the caller states which pin it showed the
    /// user). Refuses a current install and a folder TurboSpark did not
    /// write; replaces an older pin of the same alias.
    #[allow(clippy::too_many_arguments)]
    pub fn install(
        &self,
        client: &Client,
        catalog: &Catalog,
        store: &Store,
        alias: &str,
        revision: &str,
        progress: &mut dyn FnMut(HubDownloadProgress),
        cancel: &CancelFlag,
    ) -> Result<PathBuf, String> {
        let profile = self
            .get(alias)
            .ok_or_else(|| format!("no embedding profile named {alias:?}"))?;
        profile.pinned().install(
            client,
            catalog,
            store,
            &store.embedding_install_path(alias),
            revision,
            progress,
            cancel,
        )
    }
}

/// The install holds a receipt for this exact pin, and every pinned file is
/// a regular file (no symlinks) at its recorded size. Extra files, such as
/// Core ML's `.compiled` cache, are allowed.
pub fn verify_install(p: &EmbeddingProfile, dir: &Path) -> Result<(), String> {
    p.pinned().verify(dir)
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
                "modalities":["text"],"dimensions":8,"matryoshka_dims":[8,4],"status":"runs",
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
    fn verified_needs_named_gates_that_exist() {
        let c = EmbeddingCatalog::embedded().unwrap();
        let p = c.get("embeddinggemma-2-coreml").unwrap();
        assert_eq!(p.status, Status::Verified);
        assert!(!p.gates.is_empty());
        // Every gate is a real test target somewhere under crates/.
        let crates = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        for profile in c.profiles() {
            for gate in &profile.gates {
                let found = std::fs::read_dir(crates)
                    .unwrap()
                    .flatten()
                    .any(|d| d.path().join("tests").join(format!("{gate}.rs")).is_file());
                assert!(found, "{}: gate {gate} names no test file", profile.alias);
            }
        }

        let row = |status: &str, gates: &str| {
            format!(
                r#"{{"schema_version":1,"profiles":[{{"alias":"m","display_name":"x","family":"f",
                "backend":"coreml","repository":"o/r","revision":"{}","license":"MIT",
                "modalities":["text"],"dimensions":8,"matryoshka_dims":[8],
                "status":"{status}"{gates},
                "assets":[{{"path":"a.bin","size":1,"sha256":"{}"}}]}}]}}"#,
                "0".repeat(40),
                "a".repeat(64)
            )
        };
        assert!(EmbeddingCatalog::from_json(&row("verified", "")).is_err());
        assert!(EmbeddingCatalog::from_json(&row("verified", r#","gates":[]"#)).is_err());
        assert!(EmbeddingCatalog::from_json(&row("verified", r#","gates":["g"]"#)).is_ok());
        assert!(EmbeddingCatalog::from_json(&row("runs", "")).is_ok());
        assert!(EmbeddingCatalog::from_json(&row("bogus", "")).is_err());
    }

    #[test]
    fn resolve_embedding_requires_current_pin() {
        let root = std::env::temp_dir().join(format!(
            "ts-embedding-resolve-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let store = Store::new(&root);
        let json = |rev: &str| {
            format!(
                r#"{{"schema_version":1,"profiles":[{{"alias":"m","display_name":"x","family":"f",
                "backend":"coreml","repository":"o/r","revision":"{rev}","license":"MIT",
                "modalities":["text"],"dimensions":8,"matryoshka_dims":[8],"status":"runs",
                "assets":[{{"path":"w.bin","size":3,"sha256":"{}"}}]}}]}}"#,
                "b".repeat(64)
            )
        };
        let old = EmbeddingCatalog::from_json(&json(&"0".repeat(40))).unwrap();
        let new = EmbeddingCatalog::from_json(&json(&"1".repeat(40))).unwrap();
        let dir = store.embedding_install_path("m");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("w.bin"), b"abc").unwrap();
        old.get("m").unwrap().pinned().write_receipt(&dir).unwrap();

        assert_eq!(store.resolve_embedding_in(&old, "m"), Some(dir.clone()));
        // The catalog moved on: the old install no longer answers to "m".
        assert_eq!(store.resolve_embedding_in(&new, "m"), None);
        let why = new.install_problem(&store, "m").unwrap();
        assert!(why.contains("older revision"), "{why}");
        assert_eq!(new.install_state(&store, "m"), Some("stale"));
        assert!(old.install_problem(&store, "m").is_none());
        assert_eq!(old.install_state(&store, "m"), Some("current"));
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(old.install_state(&store, "m"), Some("absent"));
        assert_eq!(store.resolve_embedding_in(&old, "../m"), None);
        let _ = std::fs::remove_dir_all(&root);
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
            status: Status::Runs,
            gates: vec![],
            assets: vec![EmbeddingAsset {
                path: "pkg/w.bin".into(),
                size: 3,
                sha256: "b".repeat(64),
            }],
        };
        assert!(verify_install(&p, &dir).is_err(), "no receipt");
        std::fs::write(dir.join("pkg/w.bin"), b"abc").unwrap();
        p.pinned().write_receipt(&dir).unwrap();
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
