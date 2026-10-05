//! The speech half of the catalog: bundled whisper entries, the local
//! install probe, and the typed speech receipt.
//!
//! Mirrors `image.rs`'s embedded-entries pattern: the bundled rows pin an
//! immutable revision and an exact file list, so the download engine and
//! the rot guard can validate metadata without touching weights. The probe
//! validates a staged directory BEFORE any receipt is written, refusing
//! foreign or hostile layouts with the reason; the receipt install hashes
//! every required file and records the config-derived memory ceiling,
//! which stays labeled estimated until a qualifying hardware gate.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use model_io::safetensors::SafetensorsFile;
use model_io::speech_family::SpeechFamily;
use model_io::speech_receipt::{FileEntry, SpeechInstallReceipt};
use model_io::whisper_config::{WhisperConfig, WhisperSpecialTokens};
use model_io::{hash_file, ModelError};

/// The weight format an entry ships, which decides the loader path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechFormat {
    /// HF exports: `d_model` config fields, `*_proj` tensor names, F16 /
    /// BF16 / F32 tensors.
    HfSafetensors,
    /// mlx-community groupwise 8-bit conversions: openai `n_*` config
    /// fields, openai module-tree tensor names, U32-packed weights with
    /// per-group F16 scales/biases (group 64), no tokenizer.json in the
    /// repo (paired from `tokenizer_source` at install time).
    Mlx8Bit,
    /// The 4-bit siblings: same layout, same group of 64, low nibble of
    /// each byte preceding the high nibble. Verified element-wise against
    /// the fp32 checkpoint (2.4e-3 mean, quantization noise) and
    /// end-to-end on whisper-tiny.en-4bit.
    Mlx4Bit,
}

impl SpeechFormat {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "hf-safetensors" => Some(Self::HfSafetensors),
            "mlx-8bit" => Some(Self::Mlx8Bit),
            "mlx-4bit" => Some(Self::Mlx4Bit),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::HfSafetensors => "hf-safetensors",
            Self::Mlx8Bit => "mlx-8bit",
            Self::Mlx4Bit => "mlx-4bit",
        }
    }
}

/// Where a distribution's `tokenizer.json` comes from when the model repo
/// does not ship one (the MLX conversions pair the openai tokenizer).
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct TokenizerSource {
    pub repo: String,
    pub file: String,
}

/// One bundled speech model row.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct SpeechCatalogEntry {
    /// Local alias, e.g. `whisper-tiny-en`.
    pub alias: String,
    /// Hugging Face repo id, e.g. `openai/whisper-tiny.en`.
    pub model_id: String,
    /// Pinned immutable revision.
    pub revision: String,
    /// Exact file set the install requires.
    pub required_files: Vec<String>,
    /// Weight format; rows written before the field existed are HF.
    #[serde(default = "default_format")]
    pub format: SpeechFormat,
    /// Tokenizer pairing for repos that omit tokenizer.json.
    #[serde(default)]
    pub tokenizer_source: Option<TokenizerSource>,
}

fn default_format() -> SpeechFormat {
    SpeechFormat::HfSafetensors
}

impl<'de> serde::Deserialize<'de> for SpeechFormat {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Self::parse(&s)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown speech format {s:?}")))
    }
}

const EMBEDDED: &str = include_str!("speech_models.json");

/// The bundled speech entries, parsed once per call (the file is small).
pub fn embedded_entries() -> Result<Vec<SpeechCatalogEntry>, String> {
    serde_json::from_str(EMBEDDED).map_err(|e| format!("embedded speech catalog is invalid: {e}"))
}

/// Looks up a bundled entry by alias.
pub fn embedded_entry(alias: &str) -> Result<SpeechCatalogEntry, String> {
    embedded_entries()?
        .into_iter()
        .find(|e| e.alias == alias)
        .ok_or_else(|| format!("no speech entry named {alias:?}"))
}

/// What the probe validated about a staged speech install.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechProbeReport {
    pub speech_family: SpeechFamily,
    pub config: WhisperConfig,
    pub tokens: WhisperSpecialTokens,
    /// Config-derived working-set estimate; never a measured number.
    pub memory_ceiling_bytes: u64,
}

const RUNTIME_FILES: [&str; 3] = ["config.json", "model.safetensors", "tokenizer.json"];

fn install_files(required_files: &[String]) -> Result<BTreeSet<String>, String> {
    let mut files: BTreeSet<String> = RUNTIME_FILES.iter().map(|name| name.to_string()).collect();
    for file in required_files {
        if file.is_empty()
            || !Path::new(file)
                .components()
                .all(|part| matches!(part, std::path::Component::Normal(_)))
        {
            return Err(format!("invalid speech install file path {file:?}"));
        }
        files.insert(file.clone());
    }
    Ok(files)
}

/// Validates a staged directory as a whisper speech install.
///
/// Checks, in order, each with its own refusal: the required files exist
/// and respect the size ceilings, `config.json` parses and validates as a
/// whisper config this runtime implements, `tokenizer.json` yields the
/// special tokens, and `model.safetensors` carries every tensor the
/// kernels consume. GGML/GGUF speech distributions are refused by the
/// file-set check before anything is parsed.
pub fn probe_speech_dir(
    dir: &Path,
    required_files: &[String],
) -> Result<SpeechProbeReport, String> {
    for file in &install_files(required_files)? {
        let path = dir.join(file);
        if !path.is_file() {
            return Err(format!("speech install is missing {file}"));
        }
        // Per-file ceiling: config/tokenizer JSON files are tens of KiB;
        // weights ride the disk, so only the small files are capped here.
        if file.ends_with(".json") {
            let len = std::fs::metadata(&path)
                .map_err(|e| format!("stat {file}: {e}"))?
                .len();
            if len > 64 * 1024 * 1024 {
                return Err(format!(
                    "{file} is {len} bytes; a config or tokenizer file is not"
                ));
            }
        }
        if file.ends_with(".gguf") || file.ends_with(".bin") {
            return Err(format!(
                "{file} looks like a GGML/GGUF distribution; speech installs are safetensors only"
            ));
        }
    }

    let config_text = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| format!("read config.json: {e}"))?;
    let config = WhisperConfig::from_json_str(&config_text)
        .map_err(|e| format!("config.json is not a supported whisper config: {e}"))?;

    let tokenizer = tokenizer::Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| format!("tokenizer.json does not load: {e}"))?;
    // The tokenizer may remap added-token IDs during loading. Probe the IDs
    // the runtime actually uses, rather than trusting the serialized hints.
    let tokenizer_json = tokenizer
        .to_string(false)
        .map_err(|e| format!("tokenizer.json does not serialize: {e}"))?;
    let tokens = WhisperSpecialTokens::from_tokenizer_json(&tokenizer_json)
        .map_err(|e| format!("tokenizer.json is not a whisper tokenizer: {e}"))?;
    tokens
        .validate_config(&config)
        .map_err(|e| format!("tokenizer.json does not match config.json: {e}"))?;
    if tokenizer
        .get_vocab(true)
        .values()
        .any(|&id| id as usize >= config.vocab_size)
    {
        return Err("tokenizer.json carries IDs outside config.json vocab_size".to_string());
    }

    let weights = SafetensorsFile::open(&dir.join("model.safetensors"))
        .map_err(|e| format!("model.safetensors does not open: {e}"))?;
    config
        .validate_weights(&weights)
        .map_err(|e| format!("model.safetensors is not a supported whisper checkpoint: {e}"))?;

    Ok(SpeechProbeReport {
        speech_family: SpeechFamily::Whisper,
        memory_ceiling_bytes: config.memory_ceiling_bytes(),
        config,
        tokens,
    })
}

/// Hashes every required file and writes the typed speech receipt into the
/// install directory. The directory must already pass
/// [`probe_speech_dir`]; this function re-probes before hashing.
pub fn install_speech_receipt(
    dir: &Path,
    required_files: &[String],
    source_repo_id: Option<String>,
    source_revision: Option<String>,
) -> Result<SpeechInstallReceipt, String> {
    let probe = probe_speech_dir(dir, required_files)?;
    let mut files = BTreeMap::new();
    for file in &install_files(required_files)? {
        let path = dir.join(file);
        let meta = std::fs::metadata(&path).map_err(|e| format!("stat {file}: {e}"))?;
        let sha = hash_file(&path, 1024 * 1024).map_err(|e| format!("hash {file}: {e}"))?;
        files.insert(
            file.to_string(),
            FileEntry {
                size: meta.len(),
                sha256: sha,
            },
        );
    }
    let receipt = SpeechInstallReceipt::new(
        probe.speech_family.as_str(),
        probe.config.n_mels,
        probe.memory_ceiling_bytes,
        source_repo_id,
        source_revision,
        files,
    );
    receipt
        .write_to_dir(dir)
        .map_err(|e| format!("write speech receipt: {e}"))?;
    Ok(receipt)
}

/// Verifies an installed speech directory against its receipt: every
/// required file present with the recorded size. (Full SHA-256
/// re-verification rides the shared install-receipt policy; this is the
/// fast open-path check.)
pub fn verify_speech_install(dir: &Path) -> Result<SpeechInstallReceipt, ModelError> {
    let receipt = SpeechInstallReceipt::read_from_dir(dir)?;
    for name in RUNTIME_FILES {
        if !receipt.files.contains_key(name) {
            return Err(ModelError::TrustedReceiptInvalid {
                detail: format!("speech receipt is missing {name}"),
            });
        }
    }
    install_files(&receipt.files.keys().cloned().collect::<Vec<_>>())
        .map_err(|detail| ModelError::TrustedReceiptInvalid { detail })?;
    for (file, entry) in &receipt.files {
        let path = dir.join(file);
        let len = std::fs::metadata(&path)
            .map_err(|_| ModelError::MissingFile { name: file.clone() })?
            .len();
        if len != entry.size {
            return Err(ModelError::TrustedReceiptInvalid {
                detail: format!("{file} is {len} bytes, receipt records {}", entry.size),
            });
        }
    }
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hf_tensor_shapes(config: &WhisperConfig) -> BTreeMap<String, Vec<usize>> {
        let d = config.d_model;
        let mut tensors: BTreeMap<String, Vec<usize>> = [
            ("encoder.conv1.weight", vec![d, config.n_mels, 3]),
            ("encoder.conv1.bias", vec![d]),
            ("encoder.conv2.weight", vec![d, d, 3]),
            ("encoder.conv2.bias", vec![d]),
            (
                "encoder.embed_positions.weight",
                vec![config.max_source_positions, d],
            ),
            (
                "decoder.embed_positions.weight",
                vec![config.max_target_positions, d],
            ),
            ("decoder.embed_tokens.weight", vec![config.vocab_size, d]),
        ]
        .into_iter()
        .map(|(name, shape)| (name.to_string(), shape))
        .collect();
        for (stack, count, ffn) in [
            ("encoder", config.encoder_layers(), config.encoder_ffn_dim),
            ("decoder", config.decoder_layers(), config.decoder_ffn_dim()),
        ] {
            for suffix in ["weight", "bias"] {
                tensors.insert(format!("{stack}.layer_norm.{suffix}"), vec![d]);
            }
            for layer in 0..count {
                let prefix = format!("{stack}.layers.{layer}");
                let attention: &[&str] = if stack == "encoder" {
                    &["self_attn"]
                } else {
                    &["self_attn", "encoder_attn"]
                };
                for attn in attention {
                    for projection in ["q", "k", "v", "out"] {
                        tensors.insert(
                            format!("{prefix}.{attn}.{projection}_proj.weight"),
                            vec![d, d],
                        );
                        if projection != "k" {
                            tensors
                                .insert(format!("{prefix}.{attn}.{projection}_proj.bias"), vec![d]);
                        }
                    }
                    for suffix in ["weight", "bias"] {
                        tensors.insert(format!("{prefix}.{attn}_layer_norm.{suffix}"), vec![d]);
                    }
                }
                for suffix in ["weight", "bias"] {
                    tensors.insert(format!("{prefix}.final_layer_norm.{suffix}"), vec![d]);
                }
                tensors.insert(format!("{prefix}.fc1.weight"), vec![ffn, d]);
                tensors.insert(format!("{prefix}.fc1.bias"), vec![ffn]);
                tensors.insert(format!("{prefix}.fc2.weight"), vec![d, ffn]);
                tensors.insert(format!("{prefix}.fc2.bias"), vec![d]);
            }
        }
        tensors
    }

    fn write_weights(dir: &Path, tensors: &BTreeMap<String, Vec<usize>>) {
        let mut offset = 0;
        let mut header = serde_json::Map::new();
        for (name, shape) in tensors {
            let bytes = shape.iter().product::<usize>() * 4;
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": "F32", "shape": shape, "data_offsets": [offset, offset + bytes]
                }),
            );
            offset += bytes;
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.resize(bytes.len() + offset, 0);
        std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
    }

    fn loadable_tokenizer_json() -> String {
        let mut added = serde_json::json!([
            {"id":50256,"content":"<|endoftext|>"},
            {"id":50257,"content":"<|startoftranscript|>"},
            {"id":50258,"content":"<|en|>"},
            {"id":50358,"content":"<|transcribe|>"},
            {"id":50361,"content":"<|notimestamps|>"},
            {"id":50362,"content":"<|0.00|>"}
        ]);
        let mut vocab: serde_json::Map<String, serde_json::Value> = (0..51864)
            .map(|id| (format!("fixture_{id}"), serde_json::json!(id)))
            .collect();
        for token in added.as_array_mut().unwrap() {
            let id = token["id"].as_u64().unwrap();
            vocab.remove(&format!("fixture_{id}"));
            vocab.insert(
                token["content"].as_str().unwrap().to_string(),
                serde_json::json!(id),
            );
            for key in ["single_word", "lstrip", "rstrip", "normalized"] {
                token[key] = false.into();
            }
            token["special"] = true.into();
        }
        serde_json::json!({
            "version": "1.0", "added_tokens": added,
            "model": {"type":"WordLevel", "vocab":vocab, "unk_token":"fixture_0"}
        })
        .to_string()
    }

    #[test]
    fn receipt_file_paths_stay_inside_the_install() {
        assert!(install_files(&["../outside".into()]).is_err());
        assert!(install_files(&["/outside".into()]).is_err());
        assert!(install_files(&["".into()]).is_err());
        assert_eq!(install_files(&[]).unwrap().len(), 3);
    }

    #[test]
    fn embedded_entries_parse_and_are_pinned() {
        let entries = embedded_entries().unwrap();
        assert!(
            entries.len() >= 4,
            "expected at least four bundled speech entries"
        );
        for entry in &entries {
            assert!(!entry.alias.is_empty());
            // 40-char hex revision: a real pin, not a placeholder.
            assert_eq!(entry.revision.len(), 40, "alias {}", entry.alias);
            assert!(entry.revision.bytes().all(|b| b.is_ascii_hexdigit()));
            assert!(entry.required_files.contains(&"config.json".to_string()));
            assert!(entry
                .required_files
                .contains(&"model.safetensors".to_string()));
            match entry.format {
                SpeechFormat::HfSafetensors => {
                    assert!(entry.model_id.starts_with("openai/whisper"));
                    assert!(entry.required_files.contains(&"tokenizer.json".to_string()));
                    assert!(entry.tokenizer_source.is_none());
                }
                SpeechFormat::Mlx8Bit | SpeechFormat::Mlx4Bit => {
                    // MLX repos do not ship the tokenizer; the pairing is
                    // mandatory so an install can fetch it.
                    assert!(entry.model_id.starts_with("mlx-community/whisper"));
                    let source = entry
                        .tokenizer_source
                        .as_ref()
                        .expect("mlx entries must pair a tokenizer source");
                    assert!(source.repo.starts_with("openai/whisper"));
                    assert_eq!(source.file, "tokenizer.json");
                }
            }
        }
        let witness = embedded_entry("whisper-tiny-en").unwrap();
        assert_eq!(witness.model_id, "openai/whisper-tiny.en");
    }

    #[test]
    fn probe_refuses_missing_and_foreign_layouts() {
        let dir = std::env::temp_dir().join(format!("ts-speech-probe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let required: Vec<String> = ["config.json", "model.safetensors", "tokenizer.json"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        // Empty directory: missing files.
        let err = probe_speech_dir(&dir, &required).unwrap_err();
        assert!(err.contains("missing config.json"), "{err}");

        // GGUF distribution: refused by name before parsing.
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        std::fs::write(dir.join("model.gguf"), b"GGUF").unwrap();
        let err = probe_speech_dir(&dir, &["config.json".to_string(), "model.gguf".to_string()])
            .unwrap_err();
        assert!(err.contains("GGML/GGUF"), "{err}");

        // Config present but a decoder config: refused on model_type.
        std::fs::write(dir.join("config.json"), r#"{"model_type": "llama"}"#).unwrap();
        std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(dir.join("model.safetensors"), b"").unwrap();
        let err = probe_speech_dir(&dir, &required).unwrap_err();
        assert!(err.contains("whisper"), "{err}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn probe_and_receipt_round_trip_on_a_real_layout() {
        // A synthetic whisper directory: tiny random tensors in real
        // safetensors framing, real config and tokenizer JSON. No
        // checkpoint download in tests.
        let dir = std::env::temp_dir().join(format!("ts-speech-install-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let required: Vec<String> = ["config.json", "model.safetensors", "tokenizer.json"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        const TINY_CONFIG: &str = r#"{"model_type":"whisper","d_model":16,"encoder_layers":2,"decoder_layers":2,
                "encoder_attention_heads":4,"decoder_attention_heads":4,
                "encoder_ffn_dim":32,"decoder_ffn_dim":32,"vocab_size":51864,
                "num_mel_bins":80,"max_source_positions":1500,"max_target_positions":448,
                "activation_function":"gelu","scale_embedding":false}"#;
        std::fs::write(dir.join("config.json"), TINY_CONFIG).unwrap();
        let tokenizer_json = loadable_tokenizer_json();
        std::fs::write(dir.join("tokenizer.json"), &tokenizer_json).unwrap();
        let config = WhisperConfig::from_json_str(TINY_CONFIG).unwrap();
        let tensors = hf_tensor_shapes(&config);
        write_weights(&dir, &tensors);

        let probe = probe_speech_dir(&dir, &required).unwrap();
        assert_eq!(probe.speech_family, SpeechFamily::Whisper);
        assert!(probe.memory_ceiling_bytes > 0);

        let receipt = install_speech_receipt(
            &dir,
            &required[..2],
            Some("openai/whisper-tiny.en".into()),
            None,
        )
        .unwrap();
        assert_eq!(receipt.install_kind, "speech");
        assert_eq!(receipt.speech_family, "whisper");
        assert_eq!(receipt.modality, "audio");
        assert_eq!(receipt.files.len(), 3);

        let verified = verify_speech_install(&dir).unwrap();
        assert_eq!(verified, receipt);

        let mut incomplete_tokenizer: serde_json::Value =
            serde_json::from_str(&tokenizer_json).unwrap();
        incomplete_tokenizer
            .as_object_mut()
            .unwrap()
            .remove("model");
        std::fs::write(dir.join("tokenizer.json"), incomplete_tokenizer.to_string()).unwrap();
        let error = probe_speech_dir(&dir, &required).unwrap_err();
        assert!(error.contains("tokenizer.json does not load"), "{error}");
        std::fs::write(dir.join("tokenizer.json"), &tokenizer_json).unwrap();

        let mut missing = tensors.clone();
        missing.remove("decoder.layers.0.encoder_attn.q_proj.bias");
        write_weights(&dir, &missing);
        let error = probe_speech_dir(&dir, &required).unwrap_err();
        assert!(error.contains("encoder_attn.q_proj.bias"), "{error}");
        let mut malformed = tensors.clone();
        malformed.insert("encoder.layers.0.fc1.weight".into(), vec![16, 32]);
        write_weights(&dir, &malformed);
        let error = probe_speech_dir(&dir, &required).unwrap_err();
        assert!(error.contains("encoder.layers.0.fc1.weight"), "{error}");
        write_weights(&dir, &tensors);

        let mut incomplete = receipt.clone();
        incomplete.files.remove("tokenizer.json");
        incomplete.write_to_dir(&dir).unwrap();
        assert!(verify_speech_install(&dir).is_err());
        receipt.write_to_dir(&dir).unwrap();

        // Tampering with a file size flips verification.
        let mut cfg_back = std::fs::read_to_string(dir.join("config.json")).unwrap();
        cfg_back.push(' ');
        std::fs::write(dir.join("config.json"), cfg_back).unwrap();
        assert!(verify_speech_install(&dir).is_err());

        std::fs::remove_dir_all(&dir).ok();
    }
}
