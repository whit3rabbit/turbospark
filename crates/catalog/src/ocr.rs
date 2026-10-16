//! Pinned OCR-model profiles and their install path.
//!
//! An OCR install is a directory under `models/ocr/<alias>` with an
//! `ocr-receipt.json`, installed and verified exactly like a pinned
//! embedding install (`crate::embedding`). It is not a registry row, and
//! `ts_ocr_open` resolves it through [`crate::resolve_ocr_arg`] only, so an
//! OCR alias never answers for a text or embedding model.
//!
//! The one profile is the Core ML half of FluidInference
//! `unlimited-ocr-mlx-coreml`: both `.mlpackage` trees, the f16 token table,
//! the tokenizer and `config.json`. The `mlx/` folders are not pinned.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::pinned_dir::{PinnedAsset, PinnedDir};
use crate::{CancelFlag, Catalog, Client, HubDownloadProgress, PinnedArtifactPlan, Store};

const EMBEDDED: &str = include_str!("ocr_models.json");
const SOURCE_ROLE: &str = "ocr";
pub const RECEIPT_FILENAME: &str = "ocr-receipt.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OcrProfile {
    pub alias: String,
    pub display_name: String,
    /// Runtime family (`unlimited_ocr`).
    pub family: String,
    /// `coreml` for the Core ML exports.
    pub backend: String,
    pub repository: String,
    pub revision: String,
    pub license: String,
    pub assets: Vec<PinnedAsset>,
}

impl OcrProfile {
    pub fn download_bytes(&self) -> u64 {
        self.assets.iter().map(|a| a.size).sum()
    }

    /// Whether this machine can run the profile, and why not. The OCR
    /// decoder is a multifunction stateful package: macOS 15.
    pub fn availability(&self) -> (bool, Option<String>) {
        match self.backend.as_str() {
            "coreml" if turbospark_coreml::os_version().0 >= 15 => (true, None),
            "coreml" => (
                false,
                Some(format!("{} requires macOS 15 or newer", self.display_name)),
            ),
            other => (false, Some(format!("unknown OCR backend {other:?}"))),
        }
    }

    fn pinned(&self) -> PinnedDir<'_> {
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
    profiles: Vec<OcrProfile>,
}

#[derive(Debug, Clone)]
pub struct OcrCatalog {
    profiles: Vec<OcrProfile>,
}

impl OcrCatalog {
    pub fn embedded() -> Result<Self, String> {
        Self::from_json(EMBEDDED)
    }

    fn from_json(json: &str) -> Result<Self, String> {
        let m: Manifest =
            serde_json::from_str(json).map_err(|e| format!("invalid OCR catalog: {e}"))?;
        if m.schema_version != 1 {
            return Err("OCR catalog must be schema 1".into());
        }
        let mut aliases = HashSet::new();
        for p in &m.profiles {
            if !aliases.insert(p.alias.clone()) {
                return Err(format!("duplicate OCR alias {:?}", p.alias));
            }
            p.pinned().validate()?;
        }
        Ok(Self {
            profiles: m.profiles,
        })
    }

    pub fn profiles(&self) -> &[OcrProfile] {
        &self.profiles
    }

    pub fn get(&self, alias: &str) -> Option<&OcrProfile> {
        self.profiles.iter().find(|p| p.alias == alias)
    }

    pub fn pinned_plan(&self, alias: &str) -> Result<PinnedArtifactPlan, String> {
        let p = self
            .get(alias)
            .ok_or_else(|| format!("no OCR profile named {alias:?}"))?;
        Ok(p.pinned().plan())
    }

    /// The install directory when it verifies against the current pin.
    pub fn installed_path(&self, store: &Store, alias: &str) -> Option<PathBuf> {
        let p = self.get(alias)?;
        let dir = store.ocr_install_path(alias);
        p.pinned().verify(&dir).ok().map(|_| dir)
    }

    /// Downloads, verifies and publishes one profile at `revision`, which
    /// must be the catalog's pin. Same refusal and replacement rules as
    /// [`crate::EmbeddingCatalog::install`].
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
            .ok_or_else(|| format!("no OCR profile named {alias:?}"))?;
        profile.pinned().install(
            client,
            catalog,
            store,
            &store.ocr_install_path(alias),
            revision,
            progress,
            cancel,
        )
    }
}

/// See [`crate::embedding::verify_install`].
pub fn verify_install(p: &OcrProfile, dir: &Path) -> Result<(), String> {
    p.pinned().verify(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_profile_pins_the_core_ml_parts() {
        let c = OcrCatalog::embedded().unwrap();
        let p = c.get("unlimited-ocr-coreml").unwrap();
        assert_eq!(p.repository, "FluidInference/unlimited-ocr-mlx-coreml");
        assert_eq!(p.revision, "4568571eacd864e0a253c609b3c9d0e00450cebc");
        assert_eq!(p.family, "unlimited_ocr");
        assert_eq!(p.download_bytes(), 4_093_760_320);
        // Every file the runtime opens (`UnlimitedOcrCoreMl::open` reads the
        // package names from the pinned config.json), and nothing from mlx/.
        let mut paths: Vec<&str> = p.assets.iter().map(|a| a.path.as_str()).collect();
        paths.sort_unstable();
        assert_eq!(
            paths,
            [
                "UnlimitedOcrDecoder.mlpackage/Data/com.apple.CoreML/model.mlmodel",
                "UnlimitedOcrDecoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
                "UnlimitedOcrDecoder.mlpackage/Manifest.json",
                "UnlimitedOcrPageEncoder.mlpackage/Data/com.apple.CoreML/model.mlmodel",
                "UnlimitedOcrPageEncoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
                "UnlimitedOcrPageEncoder.mlpackage/Manifest.json",
                "config.json",
                "embeddings.f16",
                "tokenizer.json",
            ]
        );
        let plan = c.pinned_plan(&p.alias).unwrap();
        assert_eq!(plan.owner_id, "ocr:unlimited-ocr-coreml");
        assert_eq!(plan.sources[0].files.len(), p.assets.len());
    }

    #[test]
    fn resolve_ocr_needs_a_verified_install() {
        let root = std::env::temp_dir().join(format!(
            "ts-ocr-resolve-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let store = Store::new(&root);
        let dir = store.ocr_install_path("unlimited-ocr-coreml");
        std::fs::create_dir_all(&dir).unwrap();
        // A receipt alone is not an install: the pinned files are missing.
        let p = OcrCatalog::embedded().unwrap();
        let p = p.get("unlimited-ocr-coreml").unwrap();
        p.pinned().write_receipt(&dir).unwrap();
        assert!(store.resolve_ocr("unlimited-ocr-coreml").is_none());
        assert!(store.resolve_ocr("../x").is_none());
        let _ = std::fs::remove_dir_all(&root);
    }
}
