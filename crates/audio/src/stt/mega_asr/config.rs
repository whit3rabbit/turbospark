//! Strict configuration parsing for the Mega-ASR family.
//!
//! Reference: `mlx_audio/stt/models/mega_asr/config.py` at mlx-audio 0.5.7,
//! commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`. The backbone half of the
//! config (audio tower, text decoder, quantization, token ids, languages) is
//! the Qwen3-ASR config and is validated by the shared
//! `crate::stt::qwen3_asr::config::Qwen3Config` parser; this module adds the
//! router and LoRA file settings plus the family profile rules.
//!
//! Two distribution profiles exist:
//!
//! - [`ProfileKind::Dynamic`]: `model_type "mega_asr"`. The checkpoint is the
//!   unmerged Qwen3-ASR base plus `router_weights` and `lora_weights` files
//!   (upstream defaults `extras/router.safetensors` and
//!   `extras/lora.safetensors`), and the router decides per clip whether the
//!   LoRA path runs.
//! - [`ProfileKind::PreMergedRobust`]: the always-on-robust quantization of
//!   the dynamic model. The robustness LoRA was folded into the Qwen3-ASR
//!   weights before quantization (documented by `merge_metadata.json` in the
//!   distribution, the profile discriminator), so there is no router and no
//!   LoRA file and every clip runs the robust path. The config still says
//!   `model_type "qwen3_asr"` because the wrapper was merged away; this family
//!   refuses such configs when `merge_metadata.json` is absent, which sends
//!   plain Qwen3-ASR checkpoints back to the `qwen3_asr` family.

use std::path::Path;

use serde_json::Value;

use crate::stt::qwen3_asr::config::Qwen3Config;
use crate::{Result, SpeechError};

pub const DEFAULT_ROUTER_WEIGHTS: &str = "extras/router.safetensors";
pub const DEFAULT_LORA_WEIGHTS: &str = "extras/lora.safetensors";

/// Which distribution shape a config describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileKind {
    /// `model_type "mega_asr"`: router plus unmerged base plus LoRA files.
    Dynamic,
    /// Pre-merged always-on-robust quantization identified by
    /// `merge_metadata.json` next to `config.json`.
    PreMergedRobust,
}

/// Router geometry declared by `router_config`. Values the pinned reference
/// omits fall back to the upstream `Model` constructor defaults. The weight
/// loader independently infers the geometry from tensor shapes (upstream
/// `AudioQualityRouter::from_converted`) and refuses a mismatch.
#[derive(Debug, Clone, PartialEq)]
pub struct RouterSettings {
    pub d_model: usize,
    pub nhead: usize,
    pub dim_feedforward: usize,
    pub num_layers: usize,
    pub n_mels: usize,
    pub frontend_hidden_dim: usize,
    pub classifier_hidden_dim: usize,
    pub max_len: usize,
}

impl Default for RouterSettings {
    fn default() -> Self {
        Self {
            d_model: 256,
            nhead: 4,
            dim_feedforward: 1024,
            num_layers: 1,
            n_mels: 80,
            frontend_hidden_dim: 128,
            classifier_hidden_dim: 128,
            max_len: 850,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MegaConfig {
    pub backbone: Qwen3Config,
    pub profile: ProfileKind,
    /// Present only for the dynamic profile; the pre-merged robust profile
    /// carries no router.
    pub router: Option<RouterSettings>,
    pub router_weights: String,
    pub lora_weights: String,
}

fn bad(field: &str, why: impl Into<String>) -> SpeechError {
    SpeechError::BadConfig {
        field: field.to_owned(),
        why: why.into(),
    }
}

fn positive(value: &Value, field: &str) -> Result<usize> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| bad(field, "must be a positive integer fitting usize"))
}

impl RouterSettings {
    /// Parses `router_config`, applying the upstream defaults for keys the
    /// reference leaves unset. `nhead` must stay 4: the reference weight
    /// conversion hardcodes it, so no other head count is verified.
    pub fn from_json(value: &Value) -> Result<Self> {
        let defaults = Self::default();
        let settings = Self {
            d_model: positive(value, "d_model")?,
            nhead: positive(value, "nhead")?,
            dim_feedforward: positive(value, "dim_feedforward")?,
            num_layers: positive(value, "num_layers")?,
            n_mels: positive(value, "n_mels")?,
            frontend_hidden_dim: value
                .get("frontend_hidden_dim")
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0)
                .unwrap_or(defaults.frontend_hidden_dim),
            classifier_hidden_dim: value
                .get("classifier_hidden_dim")
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0)
                .unwrap_or(defaults.classifier_hidden_dim),
            max_len: positive(value, "max_len")?,
        };
        if settings.nhead != 4 {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "Mega-ASR router supports the verified 4-head attention, found {}",
                    settings.nhead
                ),
            });
        }
        if settings.n_mels != 80 {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "Mega-ASR router supports the verified 80-band log-mel frontend, found {}",
                    settings.n_mels
                ),
            });
        }
        if settings.d_model % settings.nhead != 0 {
            return Err(bad(
                "router_config.d_model",
                "router d_model must divide across the 4 attention heads",
            ));
        }
        Ok(settings)
    }
}

impl MegaConfig {
    /// Picks the profile from `config.json` plus the distribution files. A
    /// `qwen3_asr` config is only a Mega-ASR artifact when
    /// `merge_metadata.json` documents the in-place LoRA fold; without it the
    /// checkpoint is a plain Qwen3-ASR model and belongs to that family.
    pub fn detect_profile(model_dir: &Path, root: &Value) -> Result<ProfileKind> {
        match root.get("model_type").and_then(Value::as_str) {
            Some("mega_asr") => Ok(ProfileKind::Dynamic),
            Some("qwen3_asr") => {
                if model_dir.join("merge_metadata.json").is_file() {
                    Ok(ProfileKind::PreMergedRobust)
                } else {
                    Err(SpeechError::Unsupported {
                        why: "model_type qwen3_asr without merge_metadata.json is a plain \
                              Qwen3-ASR checkpoint, not a Mega-ASR distribution"
                            .into(),
                    })
                }
            }
            other => Err(bad(
                "model_type",
                format!("expected mega_asr or a pre-merged qwen3_asr, found {other:?}"),
            )),
        }
    }

    pub fn from_json(root: &Value, profile: ProfileKind) -> Result<Self> {
        match (profile, root.get("model_type").and_then(Value::as_str)) {
            (ProfileKind::Dynamic, Some("mega_asr")) => {}
            (ProfileKind::PreMergedRobust, Some("qwen3_asr")) => {}
            (profile, other) => {
                return Err(bad(
                    "model_type",
                    format!("config model_type {other:?} does not match the {profile:?} profile"),
                ))
            }
        }
        if profile == ProfileKind::PreMergedRobust
            && (root.get("router_config").is_some()
                || root.get("router_weights").is_some()
                || root.get("lora_weights").is_some())
        {
            return Err(bad(
                "router_config",
                "the pre-merged robust profile must not declare router or LoRA configuration",
            ));
        }

        // The backbone is a Qwen3-ASR model; reuse the shared validated parser
        // by presenting the model_type it expects.
        let mut backbone_root = root.clone();
        backbone_root["model_type"] = Value::String("qwen3_asr".into());
        let backbone = Qwen3Config::from_json(&backbone_root)?;
        if !backbone.text.tie_word_embeddings {
            return Err(SpeechError::Unsupported {
                why: "Mega-ASR requires tied token embeddings".into(),
            });
        }

        let router = match profile {
            ProfileKind::Dynamic => {
                let value = root
                    .get("router_config")
                    .ok_or_else(|| bad("router_config", "missing for the dynamic profile"))?;
                Some(RouterSettings::from_json(value)?)
            }
            ProfileKind::PreMergedRobust => None,
        };
        let string_setting = |key: &str, default: &str| -> String {
            root.get(key)
                .and_then(Value::as_str)
                .unwrap_or(default)
                .to_owned()
        };
        Ok(Self {
            backbone,
            profile,
            router,
            router_weights: string_setting("router_weights", DEFAULT_ROUTER_WEIGHTS),
            lora_weights: string_setting("lora_weights", DEFAULT_LORA_WEIGHTS),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn pinned_config() -> Value {
        // The pinned Mega-ASR-8bit config.json, verbatim from the fixture.
        let fixture: Value =
            serde_json::from_str(include_str!("../../../testdata/mega_asr_reference.json"))
                .unwrap();
        fixture["config_json"].clone()
    }

    fn dynamic_config() -> Value {
        let mut root = pinned_config();
        root["model_type"] = json!("mega_asr");
        root["router_config"] = json!({
            "d_model": 256, "nhead": 4, "dim_feedforward": 1024,
            "num_layers": 1, "n_mels": 80, "max_len": 850
        });
        root["router_weights"] = json!("extras/router.safetensors");
        root["lora_weights"] = json!("extras/lora.safetensors");
        root
    }

    #[test]
    fn parses_the_pinned_pre_merged_profile() {
        let root = pinned_config();
        let config = MegaConfig::from_json(&root, ProfileKind::PreMergedRobust).unwrap();
        assert_eq!(config.profile, ProfileKind::PreMergedRobust);
        assert!(config.router.is_none());
        assert_eq!(config.backbone.text.hidden_size, 2048);
        assert_eq!(config.backbone.text.num_hidden_layers, 28);
        assert_eq!(config.backbone.audio.d_model, 1024);
        assert_eq!(config.backbone.quant_bits, 8);
        assert_eq!(config.backbone.quant_group_size, 64);
        assert_eq!(config.router_weights, DEFAULT_ROUTER_WEIGHTS);
        assert_eq!(config.lora_weights, DEFAULT_LORA_WEIGHTS);
    }

    #[test]
    fn parses_the_dynamic_profile_with_defaults() {
        let mut root = dynamic_config();
        // The pinned dynamic distribution omits the two hidden dims; the
        // reference Model constructor defaults them to 128.
        let config = MegaConfig::from_json(&root, ProfileKind::Dynamic).unwrap();
        let router = config.router.expect("dynamic profile carries a router");
        assert_eq!(router, RouterSettings::default());
        assert_eq!(config.router_weights, "extras/router.safetensors");

        root["router_config"]["frontend_hidden_dim"] = json!(96);
        let config = MegaConfig::from_json(&root, ProfileKind::Dynamic).unwrap();
        assert_eq!(config.router.unwrap().frontend_hidden_dim, 96);
    }

    #[test]
    fn detect_profile_requires_merge_metadata_for_qwen3_type() {
        let root = pinned_config();
        let dir = std::env::temp_dir().join(format!("mega-config-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(matches!(
            MegaConfig::detect_profile(&dir, &root),
            Err(SpeechError::Unsupported { .. })
        ));
        std::fs::write(dir.join("merge_metadata.json"), b"{}").unwrap();
        assert_eq!(
            MegaConfig::detect_profile(&dir, &root).unwrap(),
            ProfileKind::PreMergedRobust
        );
        assert_eq!(
            MegaConfig::detect_profile(&dir, &dynamic_config()).unwrap(),
            ProfileKind::Dynamic
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn refuses_cross_profile_model_types_and_foreign_geometry() {
        let pinned = pinned_config();
        // Dynamic profile demands model_type mega_asr.
        assert!(matches!(
            MegaConfig::from_json(&pinned, ProfileKind::Dynamic),
            Err(SpeechError::BadConfig { .. })
        ));
        // Pre-merged profile refuses router declarations.
        let mut with_router = pinned.clone();
        with_router["router_config"] = json!({"d_model": 256, "nhead": 4});
        assert!(matches!(
            MegaConfig::from_json(&with_router, ProfileKind::PreMergedRobust),
            Err(SpeechError::BadConfig { .. })
        ));
        // Unknown model type.
        let mut foreign = pinned;
        foreign["model_type"] = json!("whisper");
        assert!(matches!(
            MegaConfig::detect_profile(&std::env::temp_dir(), &foreign),
            Err(SpeechError::BadConfig { .. })
        ));
        // The reference weight conversion hardcodes 4 heads.
        let mut heads = dynamic_config();
        heads["router_config"]["nhead"] = json!(8);
        assert!(matches!(
            MegaConfig::from_json(&heads, ProfileKind::Dynamic),
            Err(SpeechError::Unsupported { .. })
        ));
        // The verified log-mel frontend is 80-band.
        let mut mels = dynamic_config();
        mels["router_config"]["n_mels"] = json!(128);
        assert!(matches!(
            MegaConfig::from_json(&mels, ProfileKind::Dynamic),
            Err(SpeechError::Unsupported { .. })
        ));
        // A dynamic config without router_config is refused.
        let mut bare = dynamic_config();
        bare.as_object_mut()
            .unwrap()
            .remove("router_config")
            .unwrap();
        assert!(matches!(
            MegaConfig::from_json(&bare, ProfileKind::Dynamic),
            Err(SpeechError::BadConfig { .. })
        ));
    }
}
