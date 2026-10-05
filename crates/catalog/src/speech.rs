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
    /// mlx-community Qwen3-ASR conversions: `qwen3_asr` config, U32-packed
    /// decoder weights with per-group BF16 scales/biases (group 64), and
    /// the tokenizer shipped as the original `vocab.json` / `merges.txt` /
    /// `tokenizer_config.json` trio instead of `tokenizer.json`.
    Mlx8BitBf16,
}

impl SpeechFormat {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "hf-safetensors" => Some(Self::HfSafetensors),
            "mlx-8bit" => Some(Self::Mlx8Bit),
            "mlx-4bit" => Some(Self::Mlx4Bit),
            "mlx-8bit-bf16" => Some(Self::Mlx8BitBf16),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::HfSafetensors => "hf-safetensors",
            Self::Mlx8Bit => "mlx-8bit",
            Self::Mlx4Bit => "mlx-4bit",
            Self::Mlx8BitBf16 => "mlx-8bit-bf16",
        }
    }

    /// The speech family every distribution of this format belongs to.
    /// The probe never trusts this; it sniffs `config.json` on disk.
    pub fn speech_family(&self) -> SpeechFamily {
        match self {
            Self::HfSafetensors | Self::Mlx8Bit | Self::Mlx4Bit => SpeechFamily::Whisper,
            Self::Mlx8BitBf16 => SpeechFamily::Qwen3Asr,
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

/// What the probe validated about a staged speech install, split by family
/// because the two runtimes consume different config parsers.
#[derive(Debug, Clone, PartialEq)]
pub enum SpeechProbeKind {
    Whisper {
        config: WhisperConfig,
        tokens: WhisperSpecialTokens,
    },
    Qwen3Asr {
        config: audio::stt::qwen3_asr::Qwen3Config,
    },
}

/// What the probe validated about a staged speech install.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechProbeReport {
    pub speech_family: SpeechFamily,
    /// Mel filterbank width the model consumes (80 or 128).
    pub n_mels: usize,
    /// Config-derived working-set estimate; never a measured number.
    pub memory_ceiling_bytes: u64,
    pub kind: SpeechProbeKind,
}

const WHISPER_RUNTIME_FILES: [&str; 3] = ["config.json", "model.safetensors", "tokenizer.json"];
const QWEN3_ASR_RUNTIME_FILES: [&str; 5] = [
    "config.json",
    "model.safetensors",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
];

/// The file set a family's runtime requires, before any per-entry extras.
fn runtime_files(family: SpeechFamily) -> &'static [&'static str] {
    match family {
        SpeechFamily::Whisper => &WHISPER_RUNTIME_FILES,
        SpeechFamily::Qwen3Asr => &QWEN3_ASR_RUNTIME_FILES,
    }
}

fn install_files(
    family: SpeechFamily,
    required_files: &[String],
) -> Result<BTreeSet<String>, String> {
    let mut files: BTreeSet<String> = runtime_files(family)
        .iter()
        .map(|name| name.to_string())
        .collect();
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

/// Sniffs the speech family from a staged `config.json` before the file
/// checks run, so the required file set matches the family. Anything that
/// does not declare `qwen3_asr` stays on the whisper path, whose parser
/// refuses foreign model types with its own reason.
fn sniff_speech_family(dir: &Path) -> SpeechFamily {
    let parsed: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap_or_default())
            .unwrap_or(serde_json::Value::Null);
    match parsed.get("model_type").and_then(serde_json::Value::as_str) {
        Some("qwen3_asr") => SpeechFamily::Qwen3Asr,
        _ => SpeechFamily::Whisper,
    }
}

/// Validates a staged directory as a speech install.
///
/// The family is sniffed from `config.json`'s `model_type`. Checks, in
/// order, each with its own refusal: the required files exist and respect
/// the size ceilings, `config.json` parses and validates as a config this
/// runtime implements, the tokenizer resolves the tokens the runtime
/// prompts with, and `model.safetensors` carries every tensor the loaders
/// consume. GGML/GGUF speech distributions are refused by the file-set
/// check before anything is parsed.
pub fn probe_speech_dir(
    dir: &Path,
    required_files: &[String],
) -> Result<SpeechProbeReport, String> {
    let family = sniff_speech_family(dir);
    for file in &install_files(family, required_files)? {
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

    match family {
        SpeechFamily::Whisper => {
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
            config.validate_weights(&weights).map_err(|e| {
                format!("model.safetensors is not a supported whisper checkpoint: {e}")
            })?;

            Ok(SpeechProbeReport {
                speech_family: SpeechFamily::Whisper,
                n_mels: config.n_mels,
                memory_ceiling_bytes: config.memory_ceiling_bytes(),
                kind: SpeechProbeKind::Whisper { config, tokens },
            })
        }
        SpeechFamily::Qwen3Asr => {
            let config_json: serde_json::Value = serde_json::from_slice(
                &std::fs::read(dir.join("config.json"))
                    .map_err(|e| format!("read config.json: {e}"))?,
            )
            .map_err(|e| format!("config.json is not valid JSON: {e}"))?;
            let config = audio::stt::qwen3_asr::Qwen3Config::from_json(&config_json)
                .map_err(|e| format!("config.json is not a supported qwen3_asr config: {e}"))?;
            let tokenizer = audio::stt::qwen3_asr::load_tokenizer(dir)
                .map_err(|e| format!("qwen3 tokenizer files do not load: {e}"))?;
            // The prompt builder injects these placeholder tokens by id, so
            // probe the resolved table the same way the runner validates it.
            for (token, expected) in [
                ("<|audio_pad|>", config.audio_token_id),
                ("<|audio_start|>", config.audio_start_token_id),
                ("<|audio_end|>", config.audio_end_token_id),
            ] {
                if tokenizer.token_to_id(token).map(|id| id as i32) != Some(expected) {
                    return Err(format!(
                        "qwen3 tokenizer resolves {token} away from its config.json id {expected}"
                    ));
                }
            }
            let weights = SafetensorsFile::open(&dir.join("model.safetensors"))
                .map_err(|e| format!("model.safetensors does not open: {e}"))?;
            audio::stt::qwen3_asr::checkpoint::validate_checkpoint(&weights, &config).map_err(
                |e| format!("model.safetensors is not a supported qwen3_asr checkpoint: {e}"),
            )?;

            Ok(SpeechProbeReport {
                speech_family: SpeechFamily::Qwen3Asr,
                n_mels: config.audio.num_mel_bins,
                memory_ceiling_bytes: config.estimated_resident_bytes(),
                kind: SpeechProbeKind::Qwen3Asr { config },
            })
        }
    }
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
    for file in &install_files(probe.speech_family, required_files)? {
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
        probe.n_mels,
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
    let family = SpeechFamily::parse(&receipt.speech_family).ok_or_else(|| {
        ModelError::TrustedReceiptInvalid {
            detail: format!("unknown speech family {:?}", receipt.speech_family),
        }
    })?;
    for name in runtime_files(family) {
        if !receipt.files.contains_key(*name) {
            return Err(ModelError::TrustedReceiptInvalid {
                detail: format!("speech receipt is missing {name}"),
            });
        }
    }
    install_files(family, &receipt.files.keys().cloned().collect::<Vec<_>>())
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
        assert!(install_files(SpeechFamily::Whisper, &["../outside".into()]).is_err());
        assert!(install_files(SpeechFamily::Whisper, &["/outside".into()]).is_err());
        assert!(install_files(SpeechFamily::Whisper, &["".into()]).is_err());
        assert_eq!(install_files(SpeechFamily::Whisper, &[]).unwrap().len(), 3);
        assert_eq!(install_files(SpeechFamily::Qwen3Asr, &[]).unwrap().len(), 5);
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
                SpeechFormat::Mlx8BitBf16 => {
                    // Qwen3-ASR ships its own BPE assets; no pairing, and
                    // the runtime tokenizer files must be required.
                    assert!(entry.model_id.starts_with("mlx-community/"));
                    assert!(entry.tokenizer_source.is_none());
                    for name in ["vocab.json", "merges.txt", "tokenizer_config.json"] {
                        assert!(
                            entry.required_files.contains(&name.to_string()),
                            "alias {} missing {name}",
                            entry.alias
                        );
                    }
                }
            }
        }
        let witness = embedded_entry("whisper-tiny-en").unwrap();
        assert_eq!(witness.model_id, "openai/whisper-tiny.en");
        let qwen3 = embedded_entry("qwen3-asr-06b-8bit").unwrap();
        assert_eq!(qwen3.model_id, "mlx-community/Qwen3-ASR-0.6B-8bit");
        assert_eq!(qwen3.format.speech_family(), SpeechFamily::Qwen3Asr);
        assert_eq!(qwen3.format.as_str(), "mlx-8bit-bf16");
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

    /// Header-only safetensors framing: the probe validates descriptors
    /// without loading values, so zero data with correct dtypes and shapes
    /// suffices. Dtypes follow the pinned layout: F32 audio tower, U32
    /// packed decoder weights, BF16 companions and RMS norms.
    fn write_qwen3_weights(dir: &Path, tensors: &BTreeMap<String, Vec<usize>>) {
        let dtype = |name: &str| -> &'static str {
            if name.starts_with("audio_tower") {
                "F32"
            } else if name.ends_with(".scales") || name.ends_with(".biases") {
                "BF16"
            } else if name.contains("embed_tokens") || name.ends_with("_proj.weight") {
                "U32"
            } else {
                "BF16"
            }
        };
        let mut header = serde_json::Map::new();
        let mut blob = 0usize;
        for (name, shape) in tensors {
            let elements: usize = shape.iter().product();
            let bytes = match dtype(name) {
                "BF16" => elements * 2,
                _ => elements * 4,
            };
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": dtype(name), "shape": shape, "data_offsets": [blob, blob + bytes]
                }),
            );
            blob += bytes;
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.resize(bytes.len() + blob, 0);
        std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
    }

    #[test]
    fn qwen3_probe_and_receipt_round_trip_on_a_real_layout() {
        let dir = std::env::temp_dir().join(format!("ts-speech-qwen3-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let config_json: serde_json::Value = serde_json::from_str(
            r#"{
            "model_type": "qwen3_asr",
            "audio_token_id": 3, "audio_start_token_id": 2, "audio_end_token_id": 4,
            "audio_config": {"num_mel_bins":128,"encoder_layers":1,"encoder_attention_heads":4,
                "encoder_ffn_dim":64,"d_model":64,"max_source_positions":64,"n_window":2,
                "n_window_infer":4,"downsample_hidden_size":8,"output_dim":64,
                "scale_embedding":false,"activation_function":"gelu"},
            "text_config": {"vocab_size":261,"hidden_size":64,"intermediate_size":128,
                "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
                "head_dim":64,"rms_norm_eps":1e-5,"rope_theta":10000.0,
                "tie_word_embeddings":true,"attention_bias":false,"hidden_act":"silu"},
            "quantization_config": {"bits":8,"group_size":64,"mode":"affine"}
        }"#,
        )
        .unwrap();
        std::fs::write(dir.join("config.json"), config_json.to_string()).unwrap();
        // A minimal Qwen2-style tokenizer: base vocab {"a","b"}, then the
        // three audio placeholders as the contiguous added tokens 2..4.
        std::fs::write(dir.join("vocab.json"), r#"{"a":0,"b":1}"#).unwrap();
        std::fs::write(dir.join("merges.txt"), "").unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"tokenizer_class":"Qwen2Tokenizer","add_prefix_space":false,
                "added_tokens_decoder":{
                    "2":{"content":"<|audio_start|>","special":true},
                    "3":{"content":"<|audio_pad|>","special":true},
                    "4":{"content":"<|audio_end|>","special":true}}}"#,
        )
        .unwrap();

        let config = audio::stt::qwen3_asr::Qwen3Config::from_json(&config_json).unwrap();
        let mut tensors = BTreeMap::new();
        // One representative tensor per required group is not enough: the
        // probe must refuse partial layouts, so list them all.
        let width = config.audio.downsample_hidden_size;
        let frequency = 128usize.div_ceil(2).div_ceil(2).div_ceil(2);
        tensors.insert(
            "audio_tower.conv2d1.weight".to_owned(),
            vec![width, 1, 3, 3],
        );
        tensors.insert("audio_tower.conv2d1.bias".to_owned(), vec![width]);
        tensors.insert(
            "audio_tower.conv2d2.weight".to_owned(),
            vec![width, width, 3, 3],
        );
        tensors.insert("audio_tower.conv2d2.bias".to_owned(), vec![width]);
        tensors.insert(
            "audio_tower.conv2d3.weight".to_owned(),
            vec![width, width, 3, 3],
        );
        tensors.insert("audio_tower.conv2d3.bias".to_owned(), vec![width]);
        tensors.insert(
            "audio_tower.conv_out.weight".to_owned(),
            vec![config.audio.d_model, width * frequency],
        );
        for layer in 0..config.audio.encoder_layers {
            let prefix = format!("audio_tower.layers.{layer}");
            for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                tensors.insert(
                    format!("{prefix}.self_attn.{projection}.weight"),
                    vec![config.audio.d_model, config.audio.d_model],
                );
                tensors.insert(
                    format!("{prefix}.self_attn.{projection}.bias"),
                    vec![config.audio.d_model],
                );
            }
            for norm in ["self_attn_layer_norm", "final_layer_norm"] {
                tensors.insert(
                    format!("{prefix}.{norm}.weight"),
                    vec![config.audio.d_model],
                );
                tensors.insert(format!("{prefix}.{norm}.bias"), vec![config.audio.d_model]);
            }
            tensors.insert(
                format!("{prefix}.fc1.weight"),
                vec![config.audio.encoder_ffn_dim, config.audio.d_model],
            );
            tensors.insert(
                format!("{prefix}.fc1.bias"),
                vec![config.audio.encoder_ffn_dim],
            );
            tensors.insert(
                format!("{prefix}.fc2.weight"),
                vec![config.audio.d_model, config.audio.encoder_ffn_dim],
            );
            tensors.insert(format!("{prefix}.fc2.bias"), vec![config.audio.d_model]);
        }
        tensors.insert(
            "audio_tower.ln_post.weight".to_owned(),
            vec![config.audio.d_model],
        );
        tensors.insert(
            "audio_tower.ln_post.bias".to_owned(),
            vec![config.audio.d_model],
        );
        tensors.insert(
            "audio_tower.proj1.weight".to_owned(),
            vec![config.audio.d_model, config.audio.d_model],
        );
        tensors.insert(
            "audio_tower.proj1.bias".to_owned(),
            vec![config.audio.d_model],
        );
        tensors.insert(
            "audio_tower.proj2.weight".to_owned(),
            vec![config.audio.output_dim, config.audio.d_model],
        );
        tensors.insert(
            "audio_tower.proj2.bias".to_owned(),
            vec![config.audio.output_dim],
        );
        // The descriptor check runs against real dtypes and shapes; the
        // packed decoder names carry their U32/BF16 framing.
        let text = &config.text;
        let q_width = text.num_attention_heads * text.head_dim;
        let kv_width = text.num_key_value_heads * text.head_dim;
        let group = config.quant_group_size;
        let packed =
            |tensors: &mut BTreeMap<String, Vec<usize>>, name: String, rows: usize, cols: usize| {
                tensors.insert(format!("{name}.weight"), vec![rows, cols / 4]);
                tensors.insert(format!("{name}.scales"), vec![rows, cols / group]);
                tensors.insert(format!("{name}.biases"), vec![rows, cols / group]);
            };
        packed(
            &mut tensors,
            "model.embed_tokens".to_owned(),
            text.vocab_size,
            text.hidden_size,
        );
        for layer in 0..text.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            let attn = format!("{prefix}.self_attn");
            let mlp = format!("{prefix}.mlp");
            tensors.insert(
                format!("{prefix}.input_layernorm.weight"),
                vec![text.hidden_size],
            );
            tensors.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![text.hidden_size],
            );
            tensors.insert(format!("{attn}.q_norm.weight"), vec![text.head_dim]);
            tensors.insert(format!("{attn}.k_norm.weight"), vec![text.head_dim]);
            packed(
                &mut tensors,
                format!("{attn}.q_proj"),
                q_width,
                text.hidden_size,
            );
            packed(
                &mut tensors,
                format!("{attn}.k_proj"),
                kv_width,
                text.hidden_size,
            );
            packed(
                &mut tensors,
                format!("{attn}.v_proj"),
                kv_width,
                text.hidden_size,
            );
            packed(
                &mut tensors,
                format!("{attn}.o_proj"),
                text.hidden_size,
                q_width,
            );
            packed(
                &mut tensors,
                format!("{mlp}.gate_proj"),
                text.intermediate_size,
                text.hidden_size,
            );
            packed(
                &mut tensors,
                format!("{mlp}.up_proj"),
                text.intermediate_size,
                text.hidden_size,
            );
            packed(
                &mut tensors,
                format!("{mlp}.down_proj"),
                text.hidden_size,
                text.intermediate_size,
            );
        }
        tensors.insert("model.norm.weight".to_owned(), vec![text.hidden_size]);
        write_qwen3_weights(&dir, &tensors);

        let required: Vec<String> = [
            "config.json",
            "model.safetensors",
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        // A foreign layout with a qwen3 config must still refuse: drop a
        // required tokenizer file first.
        std::fs::remove_file(dir.join("merges.txt")).unwrap();
        let error = probe_speech_dir(&dir, &required).unwrap_err();
        assert!(error.contains("missing merges.txt"), "{error}");
        std::fs::write(dir.join("merges.txt"), "").unwrap();

        let probe = probe_speech_dir(&dir, &required).unwrap();
        assert_eq!(probe.speech_family, SpeechFamily::Qwen3Asr);
        assert_eq!(probe.n_mels, 128);
        assert!(probe.memory_ceiling_bytes > 0);
        match &probe.kind {
            SpeechProbeKind::Qwen3Asr { config: parsed } => {
                assert_eq!(parsed.text.vocab_size, 261);
            }
            other => panic!("expected a qwen3_asr probe kind, got {other:?}"),
        }

        let receipt = install_speech_receipt(
            &dir,
            &required[..1],
            Some("mlx-community/Qwen3-ASR-0.6B-8bit".into()),
            None,
        )
        .unwrap();
        assert_eq!(receipt.speech_family, "qwen3_asr");
        assert_eq!(receipt.n_mels, 128);
        assert_eq!(receipt.files.len(), 5);
        let verified = verify_speech_install(&dir).unwrap();
        assert_eq!(verified, receipt);

        // Removing a runtime-required file flips verification even when the
        // optional extras stay intact.
        std::fs::remove_file(dir.join("merges.txt")).unwrap();
        assert!(verify_speech_install(&dir).is_err());

        std::fs::remove_dir_all(&dir).ok();
    }
}
