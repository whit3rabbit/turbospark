//! Typed speech install receipt: `speech-receipt.json` inside a speech
//! install directory.
//!
//! Separate from the decoder `.gturbo` receipt: speech distributions are
//! safetensors installs (config.json + model.safetensors + tokenizer.json)
//! recorded with an `install_kind` of `speech`, the speech family wire
//! string, the audio modality, the mel width, and the config-derived
//! memory ceiling the fit gate compares against. The file hashes ride the
//! same [`FileEntry`] shape the decoder receipt uses so the catalog's
//! existing rot-guard machinery reads them unchanged.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::ModelError;
pub use crate::install_receipt::FileEntry;

/// Receipt schema version.
pub const SPEECH_RECEIPT_SCHEMA_VERSION: i64 = 1;

/// The `install_kind` string speech installs record.
pub const SPEECH_INSTALL_KIND: &str = "speech";

/// The modality string speech installs record.
pub const AUDIO_MODALITY: &str = "audio";

/// Default filename for the speech receipt inside an install directory.
pub const SPEECH_RECEIPT_FILENAME: &str = "speech-receipt.json";

/// Typed speech install receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechInstallReceipt {
    pub schema_version: i64,
    /// Always [`SPEECH_INSTALL_KIND`]; the catalog's serde-defaulted
    /// `install_kind` field reads this.
    pub install_kind: String,
    /// The [`crate::speech_family::SpeechFamily`] wire string, e.g.
    /// `whisper`.
    pub speech_family: String,
    /// Always [`AUDIO_MODALITY`].
    pub modality: String,
    /// Mel filterbank width the model consumes (80 or 128).
    pub n_mels: usize,
    /// Config-derived working-set estimate; labeled estimated until a
    /// qualifying hardware gate measures it.
    pub memory_ceiling_bytes: u64,
    #[serde(default)]
    pub source_repo_id: Option<String>,
    #[serde(default)]
    pub source_revision: Option<String>,
    /// Per-file size and SHA-256 for every file the install requires.
    pub files: BTreeMap<String, FileEntry>,
}

impl SpeechInstallReceipt {
    /// Builds a receipt for a validated speech install.
    pub fn new(
        speech_family: &str,
        n_mels: usize,
        memory_ceiling_bytes: u64,
        source_repo_id: Option<String>,
        source_revision: Option<String>,
        files: BTreeMap<String, FileEntry>,
    ) -> Self {
        Self {
            schema_version: SPEECH_RECEIPT_SCHEMA_VERSION,
            install_kind: SPEECH_INSTALL_KIND.to_string(),
            speech_family: speech_family.to_string(),
            modality: AUDIO_MODALITY.to_string(),
            n_mels,
            memory_ceiling_bytes,
            source_repo_id,
            source_revision,
            files,
        }
    }

    /// Serializes the receipt to pretty JSON.
    pub fn to_json_pretty(&self) -> Result<String, ModelError> {
        serde_json::to_string_pretty(self).map_err(|e| ModelError::IoFailed {
            call: "serialize speech receipt".to_string(),
            detail: e.to_string(),
        })
    }

    /// Writes the receipt into an install directory as
    /// [`SPEECH_RECEIPT_FILENAME`].
    pub fn write_to_dir(&self, dir: &Path) -> Result<(), ModelError> {
        let path = dir.join(SPEECH_RECEIPT_FILENAME);
        let json = self.to_json_pretty()?;
        std::fs::write(&path, json).map_err(|e| ModelError::IoFailed {
            call: format!("write {}", path.display()),
            detail: e.to_string(),
        })
    }

    /// Reads and parses the receipt from an install directory.
    pub fn read_from_dir(dir: &Path) -> Result<Self, ModelError> {
        let path = dir.join(SPEECH_RECEIPT_FILENAME);
        let json = std::fs::read_to_string(&path).map_err(|e| ModelError::IoFailed {
            call: format!("read {}", path.display()),
            detail: e.to_string(),
        })?;
        let receipt: Self =
            serde_json::from_str(&json).map_err(|e| ModelError::TrustedReceiptInvalid {
                detail: format!("{} is not a speech receipt: {e}", path.display()),
            })?;
        receipt.validate()?;
        Ok(receipt)
    }

    /// Structural validation: the install kind, modality, and family wire
    /// string must all be the recognized ones.
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.schema_version != SPEECH_RECEIPT_SCHEMA_VERSION {
            return Err(ModelError::TrustedReceiptInvalid {
                detail: format!(
                    "unsupported speech receipt schema version {}",
                    self.schema_version
                ),
            });
        }
        if !matches!(self.n_mels, 80 | 128) {
            return Err(ModelError::TrustedReceiptInvalid {
                detail: format!("unsupported Whisper mel width {}", self.n_mels),
            });
        }
        if self.memory_ceiling_bytes == 0 {
            return Err(ModelError::TrustedReceiptInvalid {
                detail: "speech receipt memory estimate must be positive".to_string(),
            });
        }
        if self.install_kind != SPEECH_INSTALL_KIND {
            return Err(ModelError::TrustedReceiptInvalid {
                detail: format!(
                    "speech receipt carries install_kind {:?}, expected {:?}",
                    self.install_kind, SPEECH_INSTALL_KIND
                ),
            });
        }
        if self.modality != AUDIO_MODALITY {
            return Err(ModelError::TrustedReceiptInvalid {
                detail: format!(
                    "speech receipt carries modality {:?}, expected {:?}",
                    self.modality, AUDIO_MODALITY
                ),
            });
        }
        if crate::speech_family::SpeechFamily::parse(&self.speech_family).is_none() {
            return Err(ModelError::TrustedReceiptInvalid {
                detail: format!("unknown speech family {:?}", self.speech_family),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> SpeechInstallReceipt {
        let mut files = BTreeMap::new();
        files.insert(
            "config.json".to_string(),
            FileEntry {
                size: 100,
                sha256: "abc".to_string(),
            },
        );
        SpeechInstallReceipt::new("whisper", 80, 134_557_632, None, None, files)
    }

    #[test]
    fn receipt_round_trips_through_json() {
        let receipt = sample();
        let json = receipt.to_json_pretty().unwrap();
        let dir = std::env::temp_dir().join(format!("ts-speech-receipt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        receipt.write_to_dir(&dir).unwrap();
        let back = SpeechInstallReceipt::read_from_dir(&dir).unwrap();
        assert_eq!(back, receipt);
        std::fs::remove_dir_all(&dir).ok();
        assert!(json.contains("\"installKind\": \"speech\""));
    }

    #[test]
    fn receipt_rejects_wrong_kind_and_unknown_family() {
        let mut receipt = sample();
        receipt.install_kind = "decoder".to_string();
        assert!(receipt.validate().is_err());
        let mut receipt = sample();
        receipt.speech_family = "whisperx".to_string();
        assert!(receipt.validate().is_err());
        let mut receipt = sample();
        receipt.modality = "text".to_string();
        assert!(receipt.validate().is_err());
    }
}

#[cfg(test)]
mod structural_regression {
    use super::*;

    #[test]
    fn rejects_unknown_version_and_invalid_resource_metadata() {
        let receipt = SpeechInstallReceipt::new("whisper", 80, 1, None, None, BTreeMap::new());
        let mut invalid = receipt.clone();
        invalid.schema_version += 1;
        assert!(invalid
            .validate()
            .unwrap_err()
            .to_string()
            .contains("schema"));
        let mut invalid = receipt.clone();
        invalid.n_mels = 0;
        assert!(invalid.validate().unwrap_err().to_string().contains("mel"));
        let mut invalid = receipt;
        invalid.memory_ceiling_bytes = 0;
        assert!(invalid
            .validate()
            .unwrap_err()
            .to_string()
            .contains("memory"));
    }
}
