//! Strict configuration and packed-manifest parsing for the Phonon-1 family.
//!
//! Reference: `mlx_audio/stt/models/phonon/` (phonon.py, config.py,
//! transport.py) at mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`. The released distribution is a
//! checksummed byte-plane archive; upstream materializes it to a local
//! directory (transport.py `prepare_model_path`) before loading. This port
//! consumes an already-materialized directory and refuses the un-materialized
//! archive with instructions, because the tar+zstd byte-plane transport is
//! deliberately out of scope.
//!
//! The materialized `config.json` is a plain Qwen3-ASR config without a
//! `quantization_config`: the quantization facts live in
//! `packed_manifest.json`, which declares the packed decoder modules, the
//! slim decoder metadata, the quint5 code format, and the hybrid
//! MLX-native-affine embedding and audio tower. The backbone half of the
//! config is validated by the shared `qwen3_asr` parsers
//! (`AudioEncoderConfig::from_root`, `TextConfig::from_root`).

use std::collections::BTreeSet;
use std::path::Path;

use serde_json::Value;

use crate::quant::read_json;
use crate::stt::qwen3_asr::config::{AudioEncoderConfig, Qwen3Config, TextConfig};
use crate::{Result, SpeechError};

use super::packed::GROUP_SIZE;

/// `config_schema` of the un-materialized release repository.
pub const CONFIG_SCHEMA: &str = "fermion.phonon/1";

/// Accepted `packed_manifest.json` format prefix (transport.py
/// `is_phonon_model`).
pub const MANIFEST_FORMAT_PREFIX: &str = "sttg1a-";

/// Required manifest status marker.
pub const MANIFEST_STATUS: &str = "PASS";

/// Quint5 code format: ten base-5 symbols per 24 bits.
pub const DECODER_CODES_FORMAT: &str = "ten-base5-per-24bit-v1";

/// Slim decoder metadata: per-row base scale plus one residual scale.
pub const SLIM_METADATA_FORMAT: &str = "broadcast-scales-v1";

/// Hybrid quantization record for the embedding and audio tower.
pub const HYBRID_FORMAT: &str = "mlx-native-affine-v1";

/// Bit width of the two affine planes the quint5 codes unpack to.
pub const PLANE_BITS: u32 = 2;

/// The seven decoder linears packed per layer, matching the reference
/// `_decoder_linear_names`.
const DECODER_LINEAR_SUFFIXES: [&str; 7] = [
    "mlp.down_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "self_attn.k_proj",
    "self_attn.o_proj",
    "self_attn.q_proj",
    "self_attn.v_proj",
];

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

/// The affine bit widths this crate's groupwise dequantizer implements.
/// Any other scheme is refused rather than mis-decoded.
fn supported_bits(value: &Value, field: &str) -> Result<u32> {
    let bits = value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
        .filter(|bits| matches!(bits, 2 | 3 | 4 | 5 | 6 | 8))
        .ok_or_else(|| bad(field, "unsupported affine bit width"))?;
    Ok(bits)
}

fn require_affine(value: &Value, field: &str) -> Result<()> {
    if value.get("mode").and_then(Value::as_str) != Some("affine") {
        return Err(SpeechError::Unsupported {
            why: format!("{field} supports MLX affine groupwise quantization only"),
        });
    }
    Ok(())
}

/// One packed decoder linear declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct ManifestModule {
    pub name: String,
    pub in_features: usize,
    pub out_features: usize,
}

/// One hybrid-quantized audio tower linear declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioLinear {
    pub name: String,
    pub in_features: usize,
    pub out_features: usize,
    pub bias: bool,
    pub bits: u32,
    pub group_size: usize,
}

/// The hybrid-quantized tied token embedding declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingSpec {
    pub num_embeddings: usize,
    pub dims: usize,
    pub bits: u32,
    pub group_size: usize,
}

/// One weight shard of the materialized checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct ShardSpec {
    pub name: String,
    pub bytes: u64,
}

/// The parsed `packed_manifest.json`. The port verifies the slim quint5
/// layout (`broadcast-scales-v1` metadata, `ten-base5-per-24bit-v1` codes);
/// the reference's explicit-scale and pre-unpacked-plane variants are
/// refused at parse time.
#[derive(Debug, Clone, PartialEq)]
pub struct PhononManifest {
    pub format: String,
    pub modules: Vec<ManifestModule>,
    pub embedding: Option<EmbeddingSpec>,
    pub audio_linears: Vec<AudioLinear>,
    pub shards: Vec<ShardSpec>,
}

impl PhononManifest {
    pub fn from_json(root: &Value) -> Result<Self> {
        let format = root
            .get("format")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("packed_manifest.format", "missing"))?;
        if !format.starts_with(MANIFEST_FORMAT_PREFIX) {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "unsupported Phonon packed manifest format {format:?} (expected the \
                     {MANIFEST_FORMAT_PREFIX:?} family)"
                ),
            });
        }
        if root.get("status").and_then(Value::as_str) != Some(MANIFEST_STATUS) {
            return Err(bad(
                "packed_manifest.status",
                "the packed manifest is not marked PASS",
            ));
        }
        let group_size = root
            .get("group_size")
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| bad("packed_manifest.group_size", "missing"))?;
        let bits = root
            .get("bits")
            .and_then(Value::as_u64)
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| bad("packed_manifest.bits", "missing"))?;
        if group_size != GROUP_SIZE || bits != PLANE_BITS {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "unsupported Phonon packed decoder layout (group size {group_size}, \
                     {bits} bits; verified layout is groups of {GROUP_SIZE} with {PLANE_BITS} bits)"
                ),
            });
        }

        let mut modules = Vec::new();
        let mut seen = BTreeSet::new();
        for row in root
            .get("modules")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("packed_manifest.modules", "missing"))?
        {
            let name = row
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
                .ok_or_else(|| bad("packed_manifest.modules.name", "missing"))?
                .to_owned();
            if !seen.insert(name.clone()) {
                return Err(bad(
                    "packed_manifest.modules",
                    format!("duplicate module {name}"),
                ));
            }
            modules.push(ManifestModule {
                name,
                in_features: positive(row, "in_features")?,
                out_features: positive(row, "out_features")?,
            });
        }
        if modules.is_empty() {
            return Err(bad(
                "packed_manifest.modules",
                "must list the packed linears",
            ));
        }

        // The port verifies the slim quint5 layout the release ships; the
        // reference PackedTritLinear also supports explicit per-group scales
        // and pre-unpacked planes, which this port refuses until verified
        // against evidence.
        match root.get("decoder_metadata") {
            Some(value) => {
                if value.get("format").and_then(Value::as_str) != Some(SLIM_METADATA_FORMAT) {
                    return Err(SpeechError::Unsupported {
                        why: "unsupported Phonon decoder metadata format".into(),
                    });
                }
            }
            None => {
                return Err(SpeechError::Unsupported {
                    why: "this port verifies the slim Phonon decoder metadata \
                          (broadcast-scales-v1) only; the manifest carries explicit per-group \
                          scales instead"
                        .into(),
                })
            }
        }
        match root.get("decoder_codes") {
            Some(value) => {
                if value.get("format").and_then(Value::as_str) != Some(DECODER_CODES_FORMAT) {
                    return Err(SpeechError::Unsupported {
                        why: "unsupported Phonon decoder code format".into(),
                    });
                }
            }
            None => {
                return Err(SpeechError::Unsupported {
                    why: "this port verifies the quint5 Phonon decoder codes \
                          (ten-base5-per-24bit-v1) only; the manifest carries pre-unpacked \
                          plane words instead"
                        .into(),
                })
            }
        }

        let hybrid = root.get("hybrid_quantization");
        if hybrid
            .is_some_and(|value| value.get("format").and_then(Value::as_str) != Some(HYBRID_FORMAT))
        {
            return Err(SpeechError::Unsupported {
                why: "unsupported Phonon hybrid quantization format".into(),
            });
        }
        let embedding = match hybrid.and_then(|value| value.get("embedding")) {
            Some(value) if !value.is_null() => {
                if value.get("name").and_then(Value::as_str) != Some("model.embed_tokens") {
                    return Err(SpeechError::Unsupported {
                        why: "unsupported Phonon embedding target".into(),
                    });
                }
                require_affine(value, "packed_manifest.hybrid_quantization.embedding")?;
                let group_size = positive(value, "group_size")?;
                let dims = positive(value, "dims")?;
                if dims % group_size != 0 {
                    return Err(bad(
                        "packed_manifest.hybrid_quantization.embedding.dims",
                        "must be divisible by the affine group size",
                    ));
                }
                Some(EmbeddingSpec {
                    num_embeddings: positive(value, "num_embeddings")?,
                    dims,
                    bits: supported_bits(value, "bits")?,
                    group_size,
                })
            }
            _ => None,
        };
        let mut audio_linears = Vec::new();
        if let Some(rows) = hybrid.and_then(|value| value.get("audio_linears")) {
            let mut seen_audio = BTreeSet::new();
            for row in rows.as_array().ok_or_else(|| {
                bad(
                    "packed_manifest.hybrid_quantization.audio_linears",
                    "must be a list",
                )
            })? {
                let name = row
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        bad(
                            "packed_manifest.hybrid_quantization.audio_linears.name",
                            "missing",
                        )
                    })?
                    .to_owned();
                if !name.starts_with("audio_tower.") || !seen_audio.insert(name.clone()) {
                    return Err(bad(
                        "packed_manifest.hybrid_quantization.audio_linears",
                        format!("invalid audio linear target {name:?}"),
                    ));
                }
                require_affine(
                    row,
                    "packed_manifest.hybrid_quantization.audio_linears.mode",
                )?;
                let group_size = positive(row, "group_size")?;
                let in_features = positive(row, "in_features")?;
                if in_features % group_size != 0 {
                    return Err(bad(
                        "packed_manifest.hybrid_quantization.audio_linears.in_features",
                        format!("{name} input width is not divisible by the group size"),
                    ));
                }
                audio_linears.push(AudioLinear {
                    name,
                    in_features,
                    out_features: positive(row, "out_features")?,
                    bias: row.get("bias").and_then(Value::as_bool).ok_or_else(|| {
                        bad(
                            "packed_manifest.hybrid_quantization.audio_linears.bias",
                            "must be a boolean",
                        )
                    })?,
                    bits: supported_bits(row, "bits")?,
                    group_size,
                });
            }
        }

        let shards = root
            .get("shards")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("packed_manifest.shards", "missing"))?
            .iter()
            .map(|row| {
                Ok(ShardSpec {
                    name: row
                        .get("name")
                        .and_then(Value::as_str)
                        .ok_or_else(|| bad("packed_manifest.shards.name", "missing"))?
                        .to_owned(),
                    bytes: row
                        .get("bytes")
                        .and_then(Value::as_u64)
                        .ok_or_else(|| bad("packed_manifest.shards.bytes", "missing"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if shards.is_empty() {
            return Err(bad("packed_manifest.shards", "must list the weight shards"));
        }

        Ok(Self {
            format: format.to_owned(),
            modules,
            embedding,
            audio_linears,
            shards,
        })
    }

    /// The decoder linear names the reference constructs for a given layer
    /// count; the manifest must cover them exactly (upstream refuses any
    /// other set).
    pub fn expected_decoder_linears(num_hidden_layers: usize) -> BTreeSet<String> {
        let mut names = BTreeSet::new();
        for index in 0..num_hidden_layers {
            for suffix in DECODER_LINEAR_SUFFIXES {
                names.insert(format!("model.layers.{index}.{suffix}"));
            }
        }
        names
    }

    /// Validates that the manifest covers the decoder linears of a decoder
    /// with `num_hidden_layers` layers exactly.
    pub fn validate_module_coverage(&self, num_hidden_layers: usize) -> Result<()> {
        let expected = Self::expected_decoder_linears(num_hidden_layers);
        let declared: BTreeSet<String> = self.modules.iter().map(|m| m.name.clone()).collect();
        if declared != expected {
            let missing: Vec<&String> = expected.difference(&declared).collect();
            let extra: Vec<&String> = declared.difference(&expected).collect();
            return Err(bad(
                "packed_manifest.modules",
                format!(
                    "manifest does not cover the decoder linears exactly (missing {missing:?}, \
                     unexpected {extra:?})"
                ),
            ));
        }
        Ok(())
    }

    /// Validates the hybrid embedding declaration against the decoder
    /// geometry (upstream compares against the constructed dense weight).
    pub fn validate_embedding(&self, text: &TextConfig) -> Result<()> {
        let Some(embedding) = &self.embedding else {
            return Ok(());
        };
        if embedding.num_embeddings != text.vocab_size || embedding.dims != text.hidden_size {
            return Err(bad(
                "packed_manifest.hybrid_quantization.embedding",
                format!(
                    "shape mismatch: manifest [{}, {}] vs decoder [{}, {}]",
                    embedding.num_embeddings, embedding.dims, text.vocab_size, text.hidden_size
                ),
            ));
        }
        Ok(())
    }
}

/// Parsed Phonon checkpoint configuration: the Qwen3-ASR backbone geometry
/// plus the packed manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct PhononConfig {
    pub backbone: Qwen3Config,
    pub manifest: PhononManifest,
}

impl PhononConfig {
    /// Loads and validates `config.json` plus `packed_manifest.json` from a
    /// materialized checkpoint directory. The un-materialized release
    /// archive is refused with materialization instructions.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let root: Value =
            read_json(&config_path).map_err(|error| bad("config.json", error.to_string()))?;
        let has_manifest = model_dir.join("packed_manifest.json").is_file();
        if !has_manifest {
            if root.get("config_schema").and_then(Value::as_str) == Some(CONFIG_SCHEMA) {
                return Err(SpeechError::Unsupported {
                    why: "this Phonon release directory is an un-materialized transport \
                          archive; materialize it with the mlx-audio 0.5.7 reference first, \
                          e.g. python -c \"from pathlib import Path; from mlx_audio.stt.models.\
                          phonon.transport import prepare_model_path; \
                          print(prepare_model_path(Path('DIR')))\". The Rust port consumes the \
                          materialized safetensors directory and deliberately does not \
                          implement the tar+zstd byte-plane transport"
                        .into(),
                });
            }
            return Err(bad(
                "packed_manifest.json",
                "missing: this directory is not a materialized Phonon checkpoint (plain \
                 qwen3_asr checkpoints belong to the qwen3_asr family)",
            ));
        }
        if root.get("model_type").and_then(Value::as_str) != Some("qwen3_asr") {
            return Err(bad("model_type", "expected qwen3_asr"));
        }

        let manifest =
            PhononManifest::from_json(&read_json(&model_dir.join("packed_manifest.json"))?)?;
        let audio = AudioEncoderConfig::from_root(&root)?;
        let text = TextConfig::from_root(&root)?;
        manifest.validate_module_coverage(text.num_hidden_layers)?;
        manifest.validate_embedding(&text)?;

        let thinker = root.get("thinker_config").unwrap_or(&root);
        let token_id = |key: &str| -> Result<i32> {
            thinker
                .get(key)
                .or_else(|| root.get(key))
                .and_then(Value::as_i64)
                .and_then(|n| i32::try_from(n).ok())
                .ok_or_else(|| bad(key, "must be a signed 32-bit token id"))
        };
        let audio_token_id = token_id("audio_token_id")?;
        let audio_start_token_id = token_id("audio_start_token_id")?;
        let audio_end_token_id = token_id("audio_end_token_id")?;
        if audio.output_dim != text.hidden_size
            || audio_token_id < 0
            || audio_start_token_id < 0
            || audio_end_token_id < 0
        {
            return Err(bad(
                "audio_config.output_dim",
                "audio output width must equal decoder hidden size and token ids must be valid",
            ));
        }
        if !text.tie_word_embeddings {
            return Err(SpeechError::Unsupported {
                why: "Phonon requires tied token embeddings".into(),
            });
        }

        // The released Phonon-1 checkpoints are English-only even though the
        // embedded teacher config retains Qwen3-ASR's multilingual list; the
        // reference ModelConfig.from_dict overrides it.
        let supported_languages = vec!["English".to_owned()];
        // The shared struct records one affine scheme for the tied embedding
        // (the scheme fields only matter where an actual `.scales` tensor is
        // read); decoder linears carry their own manifest layout and the
        // audio tower its per-linear rows.
        let (quant_bits, quant_group_size) = manifest
            .embedding
            .as_ref()
            .map(|embedding| (embedding.bits, embedding.group_size))
            .unwrap_or((8, 64));
        Ok(Self {
            backbone: Qwen3Config {
                audio,
                text,
                audio_token_id,
                audio_start_token_id,
                audio_end_token_id,
                supported_languages,
                quant_bits,
                quant_group_size,
            },
            manifest,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{PhononConfig, PhononManifest, CONFIG_SCHEMA};
    use crate::SpeechError;
    use serde_json::{json, Value};

    fn fixture() -> Value {
        serde_json::from_str(include_str!("../../../testdata/phonon_reference.json")).unwrap()
    }

    fn manifest() -> PhononManifest {
        PhononManifest::from_json(&fixture()["packed_manifest"]).unwrap()
    }

    fn unmaterialized_dir() -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("phonon-unmaterialized-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&json!({
                "config_schema": CONFIG_SCHEMA,
                "model_id": "FermionResearch/Phonon-1",
                "artifact": {"filename": "phonon-audio6.bps.tar.zst"}
            }))
            .unwrap(),
        )
        .unwrap();
        dir
    }

    #[test]
    fn parses_the_pinned_manifest_from_the_fixture() {
        let manifest = manifest();
        assert!(manifest.format.starts_with("sttg1a-"));
        assert_eq!(manifest.modules.len(), 196);
        assert_eq!(manifest.audio_linears.len(), 111);
        assert!(manifest
            .audio_linears
            .iter()
            .all(|linear| linear.bits == 6 && linear.group_size == 128));
        assert!(!manifest.audio_linears[0].bias, "conv_out carries no bias");
        let embedding = manifest.embedding.unwrap();
        assert_eq!(embedding.num_embeddings, 151_936);
        assert_eq!(embedding.dims, 1_024);
        assert_eq!(embedding.bits, 8);
        assert_eq!(embedding.group_size, 64);
        assert_eq!(manifest.shards.len(), 1);
        assert_eq!(manifest.shards[0].name, "model-00001.safetensors");
    }

    #[test]
    fn refuses_layout_variants_the_port_has_not_verified() {
        let base = fixture()["packed_manifest"].clone();
        for field in ["decoder_metadata", "decoder_codes"] {
            let mut root = base.clone();
            root.as_object_mut().unwrap().remove(field).unwrap();
            let error = PhononManifest::from_json(&root).unwrap_err();
            assert!(
                matches!(error, SpeechError::Unsupported { .. }),
                "removing {field} must refuse"
            );
        }
    }

    #[test]
    fn manifest_covers_the_pinned_decoder_exactly() {
        let manifest = manifest();
        manifest.validate_module_coverage(28).unwrap();
        assert!(manifest.validate_module_coverage(27).is_err());
        assert!(manifest.validate_module_coverage(29).is_err());
    }

    #[test]
    fn parses_the_pinned_backbone_from_the_fixture() {
        let dir =
            std::env::temp_dir().join(format!("phonon-config-fixture-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&fixture()["config_json"]).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.join("packed_manifest.json"),
            serde_json::to_vec(&fixture()["packed_manifest"]).unwrap(),
        )
        .unwrap();
        let config = PhononConfig::load(&dir).unwrap();
        assert_eq!(config.backbone.text.hidden_size, 1024);
        assert_eq!(config.backbone.text.num_hidden_layers, 28);
        assert_eq!(config.backbone.text.intermediate_size, 3072);
        assert_eq!(config.backbone.audio.d_model, 896);
        assert_eq!(config.backbone.audio.encoder_layers, 18);
        assert_eq!(config.backbone.audio_token_id, 151_676);
        assert_eq!(config.backbone.supported_languages, ["English"]);
        assert_eq!(config.backbone.quant_bits, 8);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn refuses_the_un_materialized_transport_archive() {
        let dir = unmaterialized_dir();
        let error = PhononConfig::load(&dir).unwrap_err();
        match &error {
            SpeechError::Unsupported { why } => {
                assert!(why.contains("un-materialized"), "{why}");
                assert!(why.contains("prepare_model_path"), "{why}");
            }
            other => panic!("expected the materialization refusal, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn refuses_a_directory_without_a_packed_manifest() {
        let dir = std::env::temp_dir().join(format!("phonon-no-manifest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&fixture()["config_json"]).unwrap(),
        )
        .unwrap();
        let error = PhononConfig::load(&dir).unwrap_err();
        assert!(matches!(error, SpeechError::BadConfig { .. }), "{error:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn refuses_foreign_manifest_fields() {
        let base = fixture()["packed_manifest"].clone();
        let mutate = |name: &str, change: &dyn Fn(&mut Value)| {
            let mut root = base.clone();
            change(&mut root);
            let error = PhononManifest::from_json(&root).unwrap_err();
            assert!(!format!("{error:?}").is_empty(), "{name} produced no error");
            error
        };

        let error = mutate("format", &|root| {
            root["format"] = json!("fermion-other-v1");
        });
        assert!(matches!(error, SpeechError::Unsupported { .. }));
        let error = mutate("status", &|root| {
            root["status"] = json!("FAIL");
        });
        assert!(matches!(error, SpeechError::BadConfig { .. }));
        let error = mutate("bits", &|root| {
            root["bits"] = json!(4);
        });
        assert!(matches!(error, SpeechError::Unsupported { .. }));
        let error = mutate("group", &|root| {
            root["group_size"] = json!(64);
        });
        assert!(matches!(error, SpeechError::Unsupported { .. }));
        let error = mutate("codes", &|root| {
            root["decoder_codes"]["format"] = json!("eight-base5-per-16bit-v1");
        });
        assert!(matches!(error, SpeechError::Unsupported { .. }));
        let error = mutate("metadata", &|root| {
            root["decoder_metadata"]["format"] = json!("dense-scales-v1");
        });
        assert!(matches!(error, SpeechError::Unsupported { .. }));
        let error = mutate("hybrid", &|root| {
            root["hybrid_quantization"]["format"] = json!("packed-affine-v2");
        });
        assert!(matches!(error, SpeechError::Unsupported { .. }));
        let error = mutate("embedding-name", &|root| {
            root["hybrid_quantization"]["embedding"]["name"] = json!("model.embed");
        });
        assert!(matches!(error, SpeechError::Unsupported { .. }));
        let error = mutate("audio-name", &|root| {
            root["hybrid_quantization"]["audio_linears"][0]["name"] = json!("tower.conv_out");
        });
        assert!(matches!(error, SpeechError::BadConfig { .. }));
        let error = mutate("audio-bits", &|root| {
            root["hybrid_quantization"]["audio_linears"][0]["bits"] = json!(7);
        });
        assert!(matches!(error, SpeechError::BadConfig { .. }));
        let error = mutate("audio-group", &|root| {
            // 100 does not divide the 7680-wide conv_out input.
            root["hybrid_quantization"]["audio_linears"][0]["group_size"] = json!(100);
        });
        assert!(matches!(error, SpeechError::BadConfig { .. }));
        let error = mutate("audio-mode", &|root| {
            root["hybrid_quantization"]["audio_linears"][0]["mode"] = json!("affine-calc");
        });
        assert!(matches!(error, SpeechError::Unsupported { .. }));
        let error = mutate("dup-module", &|root| {
            let first = root["modules"][0].clone();
            root["modules"].as_array_mut().unwrap().push(first);
        });
        assert!(matches!(error, SpeechError::BadConfig { .. }));

        // A dropped module parses but fails the exact decoder coverage check.
        let mut root = base.clone();
        root["modules"].as_array_mut().unwrap().remove(0);
        let manifest = PhononManifest::from_json(&root).unwrap();
        assert!(manifest.validate_module_coverage(28).is_err());
    }

    #[test]
    fn refuses_an_embedding_that_contradicts_the_decoder() {
        let manifest = manifest();
        let text = crate::stt::qwen3_asr::config::TextConfig {
            vocab_size: 32_000,
            hidden_size: 1_024,
            intermediate_size: 3_072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            rotary_dim: 128,
            rms_norm_eps: 1.0e-6,
            rope_theta: 1_000_000.0,
            qk_norm: true,
            tie_word_embeddings: true,
        };
        let error = manifest.validate_embedding(&text).unwrap_err();
        assert!(matches!(error, SpeechError::BadConfig { .. }), "{error:?}");
    }
}
