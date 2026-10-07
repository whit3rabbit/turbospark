//! Cohere Transcribe detection and refusal.
//!
//! Reference: `mlx_audio/stt/models/cohere_asr/` at mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`. No runnable checkpoint
//! profile exists for this family: the official CohereLabs distribution is
//! access-gated (Hub 403 in the verification environment) and the public
//! MLX mirror decodes incoherent text through the unmodified mlx-audio 0.5.7
//! reference itself. This module detects the family's distribution shape and
//! refuses to load it with that documented reason, preserving the negative
//! finding instead of silently mis-decoding.

use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::{Result, SpeechError};

/// The upstream model_type string for this family.
pub const COHERE_ASR_MODEL_TYPE: &str = "cohere_asr";

/// Official distribution (access-gated, revision recorded from the pinned
/// inventory investigation).
pub const OFFICIAL_REPOSITORY: &str = "CohereLabs/cohere-transcribe-03-2026";
pub const OFFICIAL_REVISION: &str = "b1eacc2686a3d08ceaae5f24a88b1d519620bc09";

/// Public MLX mirror that loads through `mlx-int8/` but produced incoherent
/// text through the unmodified reference on the smoke clip.
pub const MIRROR_REPOSITORY: &str = "mlx-community/cohere-transcribe-03-2026-mlx-8bit";
pub const MIRROR_REVISION: &str = "a0acb7f93cd32d82c4fbf801b6d8fb39d20c509f";

/// Reads a candidate model directory's `config.json` and reports whether it
/// is a Cohere Transcribe distribution.
pub fn detect(model_dir: &Path) -> Result<bool> {
    let path = model_dir.join("config.json");
    let json = fs::read_to_string(&path).map_err(|error| SpeechError::Input {
        why: format!("cannot read {}: {error}", path.display()),
    })?;
    let root: Value = serde_json::from_str(&json).map_err(|error| SpeechError::BadConfig {
        field: "config.json".into(),
        why: error.to_string(),
    })?;
    Ok(root.get("model_type").and_then(Value::as_str) == Some(COHERE_ASR_MODEL_TYPE))
}

/// Loads a Cohere Transcribe distribution; always refused.
///
/// The family architecture is not implemented because no checkpoint passed
/// reference verification. See the family README for the investigation
/// record and what a future implementation must re-verify first.
pub fn load(model_dir: &Path) -> Result<()> {
    if !detect(model_dir)? {
        return Err(SpeechError::BadConfig {
            field: "model_type".into(),
            why: format!(
                "not a {COHERE_ASR_MODEL_TYPE} distribution; this loader only reports the family refusal"
            ),
        });
    }
    Err(SpeechError::Unsupported {
        why: format!(
            "cohere_asr is not implemented: the official pin {OFFICIAL_REPOSITORY} @ \
             {OFFICIAL_REVISION} is access-gated (Hub 403) and the public mirror \
             {MIRROR_REPOSITORY} @ {MIRROR_REVISION} produced incoherent text through \
             unmodified mlx-audio 0.5.7 on the reference clip, so no runnable profile \
             is verified; see crates/audio/src/stt/cohere_asr/README.md"
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::{detect, load, MIRROR_REPOSITORY, OFFICIAL_REPOSITORY};

    #[test]
    fn pins_record_both_investigated_revisions() {
        assert_eq!(OFFICIAL_REPOSITORY, "CohereLabs/cohere-transcribe-03-2026");
        assert_eq!(
            MIRROR_REPOSITORY,
            "mlx-community/cohere-transcribe-03-2026-mlx-8bit"
        );
    }

    #[test]
    fn detects_the_family_config_and_refuses_to_load() {
        let root = std::env::temp_dir().join(format!("turbospark-cohere-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("config.json"),
            r#"{"model_type": "cohere_asr", "architectures": ["CohereAsrForCTC"]}"#,
        )
        .unwrap();
        assert!(detect(&root).unwrap());
        let error = load(&root).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("not implemented"), "{message}");
        assert!(message.contains("403"), "{message}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn foreign_configs_are_not_detected() {
        let root = std::env::temp_dir().join(format!("turbospark-cohere-f{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("config.json"), r#"{"model_type": "whisper"}"#).unwrap();
        assert!(!detect(&root).unwrap());
        std::fs::remove_dir_all(&root).unwrap();
    }
}
