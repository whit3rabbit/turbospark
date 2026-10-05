//! Whisper `config.json` parse, SOT token constants, language table
//! extraction, and the derived memory ceiling.
//!
//! Deserializes the Hugging Face whisper `config.json` directly (openai/
//! whisper and compatible distributions). Validation is structural and
//! refuses, with the reason, every shape the reference kernels in
//! `compute::whisper` do not implement -- a wrong-but-fluent forward is
//! worse than a refused open.
//!
//! The special-token constants below are format constants of the whisper
//! tokenizer family (multilingual vocab 51865, English-only 51864); the
//! language table itself is extracted from the distribution's
//! `tokenizer.json` added tokens rather than hardcoded, so both the
//! multilingual and the `.en` vocabularies resolve without a second table
//! to drift.

use serde::Deserialize;

/// SOT-grammar token ids resolved from the loaded whisper tokenizer.
/// Token order and the English/multilingual text-vocabulary boundary are
/// part of the checkpoint contract, so extraction must be validated
/// against the configuration before decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WhisperSpecialTokens {
    /// `<|endoftext|>`: end of transcript and the greedy decode stop token.
    pub eot: u32,
    /// `<|startoftranscript|>`: the decode prompt's first token.
    pub sot: u32,
    /// First language token id (`<|en|>` on standard vocabularies).
    pub first_language: u32,
    /// Last language token id (inclusive).
    pub last_language: u32,
    /// `<|translate|>`.
    pub translate: Option<u32>,
    /// `<|transcribe|>`.
    pub transcribe: u32,
    /// `<|startoflm|>`.
    pub startoflm: Option<u32>,
    /// `<|nospeech|>` (also named `<|nocaptions|>` by HF exports).
    pub no_speech: Option<u32>,
    /// `<|notimestamps|>`: forced in no-timestamp decode mode.
    pub no_timestamps: u32,
    /// First `<|0.00|>` timestamp token.
    pub first_timestamp: u32,
    /// True for the multilingual text vocabulary. Both tokenizer variants
    /// carry language/task specials; English decoding omits that prompt
    /// block because its checkpoint was trained for English only.
    pub multilingual: bool,
}

impl WhisperSpecialTokens {
    /// The multilingual defaults, used as a cross-check baseline and as the
    /// fallback when a tokenizer.json is absent.
    pub const MULTILINGUAL: Self = Self {
        eot: 50257,
        sot: 50258,
        first_language: 50259,
        last_language: 50357,
        translate: Some(50358),
        transcribe: 50359,
        startoflm: Some(50360),
        no_speech: Some(50362),
        no_timestamps: 50363,
        first_timestamp: 50364,
        multilingual: true,
    };

    /// Extracts the special tokens from a distribution's `tokenizer.json`.
    ///
    /// Language tokens are the added tokens whose content is
    /// `<|code|>` with an all-ASCII-letter code (which excludes timestamps
    /// like `<|0.00|>` and the fixed specials by shape alone).
    pub fn from_tokenizer_json(json: &str) -> Result<Self, crate::error::ModelError> {
        #[derive(Deserialize)]
        struct AddedToken {
            id: u32,
            content: String,
        }
        #[derive(Deserialize)]
        struct TokenizerJson {
            added_tokens: Vec<AddedToken>,
        }
        let parsed: TokenizerJson =
            serde_json::from_str(json).map_err(|e| crate::error::ModelError::BadConfig {
                field: "tokenizer.json".to_string(),
                why: format!("not a tokenizers JSON document: {e}"),
            })?;
        let find = |content: &str| -> Option<u32> {
            parsed
                .added_tokens
                .iter()
                .find(|t| t.content == content)
                .map(|t| t.id)
        };
        let miss = |what: &str| crate::error::ModelError::BadConfig {
            field: "tokenizer.json".to_string(),
            why: format!("missing the {what} special token"),
        };
        let eot = find("<|endoftext|>").ok_or_else(|| miss("<|endoftext|>"))?;
        let sot = find("<|startoftranscript|>").ok_or_else(|| miss("<|startoftranscript|>"))?;
        let transcribe = find("<|transcribe|>").ok_or_else(|| miss("<|transcribe|>"))?;
        let no_timestamps = find("<|notimestamps|>").ok_or_else(|| miss("<|notimestamps|>"))?;
        // HF exports historically spell the silence token nocaptions.
        // Its presence does not identify an English/multilingual model.
        let translate = find("<|translate|>");
        let startoflm = find("<|startoflm|>");
        let no_speech = find("<|nospeech|>").or_else(|| find("<|nocaptions|>"));

        let mut languages: Vec<(u32, String)> = parsed
            .added_tokens
            .iter()
            .filter_map(|t| {
                let inner = t.content.strip_prefix("<|")?.strip_suffix("|>")?;
                if !inner.is_empty()
                    && inner.len() <= 3
                    && inner.bytes().all(|b| b.is_ascii_lowercase())
                {
                    Some((t.id, inner.to_string()))
                } else {
                    None
                }
            })
            .collect();
        languages.sort_unstable();
        if languages.is_empty() {
            return Err(miss("language tokens"));
        }

        if languages
            .windows(2)
            .any(|pair| pair[0].0.checked_add(1) != Some(pair[1].0))
        {
            return Err(crate::error::ModelError::BadConfig {
                field: "tokenizer.json".to_string(),
                why: "language tokens must occupy a contiguous ID range".to_string(),
            });
        }
        let first_timestamp = find("<|0.00|>").ok_or_else(|| miss("<|0.00|>"))?;
        // OpenAI's English encoding has 50256 text entries; multilingual
        // has 50257. Silence-token names differ across supported exports.
        let multilingual = match eot {
            50256 => false,
            50257 => true,
            _ => {
                return Err(crate::error::ModelError::BadConfig {
                    field: "tokenizer.json".to_string(),
                    why: "endoftext ID is not a supported whisper vocabulary boundary".to_string(),
                })
            }
        };
        Ok(Self {
            eot,
            sot,
            first_language: languages[0].0,
            last_language: languages[languages.len() - 1].0,
            translate,
            transcribe,
            startoflm,
            no_speech,
            no_timestamps,
            first_timestamp,
            multilingual,
        })
    }

    /// Checks that loaded tokenizer IDs match the checkpoint vocabulary
    /// and the ordered special-token region used by the suppression mask.
    pub fn validate_config(&self, config: &WhisperConfig) -> Result<(), crate::error::ModelError> {
        let bad = |why: String| crate::error::ModelError::BadConfig {
            field: "tokenizer.json".to_string(),
            why,
        };
        let ids = [
            Some(self.eot),
            Some(self.sot),
            Some(self.first_language),
            Some(self.last_language),
            self.translate,
            Some(self.transcribe),
            self.startoflm,
            self.no_speech,
            Some(self.no_timestamps),
            Some(self.first_timestamp),
        ];
        if ids
            .into_iter()
            .flatten()
            .any(|id| id as usize >= config.vocab_size)
        {
            return Err(bad("special token ID exceeds config vocab_size".to_string()));
        }
        if self.multilingual != (config.vocab_size >= 51865) {
            return Err(bad(
                "English/multilingual tokenizer does not match config vocab_size".to_string(),
            ));
        }
        if self.sot != self.eot + 1
            || self.first_language != self.sot + 1
            || self.last_language < self.first_language
            || self.transcribe <= self.last_language
            || self.no_timestamps <= self.transcribe
            || self.first_timestamp != self.no_timestamps + 1
            || [self.translate, self.startoflm, self.no_speech]
                .into_iter()
                .flatten()
                .any(|id| id <= self.last_language || id >= self.first_timestamp)
        {
            return Err(bad(
                "special token ordering does not match whisper decoding".to_string(),
            ));
        }
        for (field, configured, resolved) in [
            (
                "decoder_start_token_id",
                config.decoder_start_token_id,
                self.sot,
            ),
            ("eos_token_id", config.eos_token_id, self.eot),
        ] {
            if configured.is_some_and(|id| id != resolved) {
                return Err(bad(format!("{field} disagrees with the loaded tokenizer")));
            }
        }
        Ok(())
    }

    /// True when `id` is one of the language tokens.
    pub fn is_language(&self, id: u32) -> bool {
        (self.first_language..=self.last_language).contains(&id)
    }

    /// True when `id` is a timestamp token: every id from the first
    /// `<|0.00|>` to the end of the vocabulary (the eot token sits below
    /// the language range, so it cannot bound this from above).
    pub fn is_timestamp(&self, id: u32) -> bool {
        id >= self.first_timestamp
    }
}

/// Parsed architecture configuration for a whisper model.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct WhisperConfig {
    /// HF `model_type`; must be `whisper`.
    pub model_type: String,
    /// `d_model`: width of the encoder and decoder streams. HF exports
    /// name it `d_model`; MLX openai-layout exports `n_audio_state`
    /// (encoder) and `n_text_state` (decoder) with the same value.
    /// Resolved in `from_json_str` (which also accepts the openai names),
    /// so the field itself defaults for serde.
    #[serde(default)]
    pub d_model: usize,
    /// HF whisper exports carry BOTH `encoder_layers` and the aliased
    /// `num_hidden_layers` with the same value, so neither can be a serde
    /// alias of the other; [`Self::encoder_layers`] resolves them.
    #[serde(default)]
    pub num_hidden_layers: Option<usize>,
    #[serde(default)]
    pub encoder_layers: Option<usize>,
    #[serde(default, alias = "n_text_layer")]
    pub decoder_layers: usize,
    #[serde(default, alias = "encoder_attention_heads", alias = "n_audio_head")]
    pub num_attention_heads: usize,
    #[serde(default, alias = "decoder_attention_heads", alias = "n_text_head")]
    pub decoder_attention_heads: usize,
    #[serde(default)]
    pub encoder_ffn_dim: usize,
    #[serde(default)]
    pub decoder_ffn_dim: usize,
    #[serde(default, alias = "n_vocab")]
    pub vocab_size: usize,
    /// Mel filterbank bands; the kernels implement 80 and 128. HF exports
    /// name this `num_mel_bins`.
    /// Mel filterbank bands; the kernels implement 80 and 128. HF exports
    /// name this `num_mel_bins`. Resolved in `from_json_str`.
    #[serde(default, alias = "num_mel_bins")]
    pub n_mels: usize,
    #[serde(default, alias = "n_audio_ctx")]
    pub max_source_positions: usize,
    #[serde(default, alias = "n_text_ctx")]
    pub max_target_positions: usize,
    /// HF `activation_function`; the kernels implement exact-erf `gelu`.
    #[serde(default = "default_activation")]
    pub activation_function: String,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    /// When true, embeddings scale by `sqrt(d_model)` before the stack.
    #[serde(default)]
    pub scale_embedding: bool,
    /// `<|startoftranscript|>`; the decode prompt's first token.
    #[serde(default)]
    pub decoder_start_token_id: Option<u32>,
    /// `<|endoftext|>`.
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    /// HF forced decoder ids, e.g. `[[1, 50259], [2, 50359]]`. Present in
    /// some distributions; the runner builds its own SOT grammar and uses
    /// this only as a cross-check.
    #[serde(default)]
    pub forced_decoder_ids: Option<Vec<Vec<u32>>>,
    /// MLX conversions carry a top-level `{"group_size": .., "bits": ..}`
    /// block naming the groupwise affine quantization applied to the
    /// linear weights. Absent (or `None`) means unquantized tensors.
    /// MLX openai-layout configs also carry `n_audio_ctx`,
    /// `alignment_heads`, and other fields this crate ignores; serde
    /// denies nothing by default, so no allowlist is needed.
    #[serde(default)]
    pub quantization: Option<WhisperQuantization>,
}

/// The groupwise affine quantization an MLX conversion ships, from the
/// config's top-level `quantization` block. The kernels implement 8-bit
/// and 4-bit groups of 64 (the `*-8bit` and `*-4bit` mlx-community
/// conversions); both dequant formulas were verified element-wise against
/// the fp32 checkpoint before implementation (8-bit: 1.5e-4, 4-bit:
/// 2.4e-3, which is 4-bit quantization noise).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct WhisperQuantization {
    pub group_size: usize,
    pub bits: usize,
}

fn default_activation() -> String {
    "gelu".to_string()
}

fn default_layer_norm_eps() -> f32 {
    1e-5
}

/// Fields the kernels do not implement, or a shape that cannot run.
impl WhisperConfig {
    /// Parses and validates a whisper `config.json` string.
    ///
    /// Two layouts reach here: HF exports (`d_model`,
    /// `encoder_attention_heads`, `num_hidden_layers`/`encoder_layers`,
    /// `vocab_size`, `num_mel_bins`) and MLX openai-layout exports
    /// (`n_audio_state`/`n_text_state`, `n_audio_head`/`n_text_head`,
    /// `n_audio_layer`/`n_text_layer`, `n_vocab`, `n_mels`). Both name the
    /// same shapes, so the parser resolves over the raw JSON with explicit
    /// precedence instead of serde alias collisions: the openai exports
    /// carry `encoder_layers` AND a generic layer count together, which
    /// serde's alias mechanism cannot express on one struct field.
    pub fn from_json_str(json: &str) -> Result<Self, crate::error::ModelError> {
        let value: serde_json::Value =
            serde_json::from_str(json).map_err(|e| crate::error::ModelError::BadConfig {
                field: "config.json".to_string(),
                why: format!("not a whisper config: {e}"),
            })?;
        let pick_usize = |names: &[&str]| -> Result<usize, crate::error::ModelError> {
            for name in names {
                if let Some(v) = value.get(*name).and_then(|v| v.as_u64()) {
                    return Ok(v as usize);
                }
            }
            Err(crate::error::ModelError::BadConfig {
                field: "config.json".to_string(),
                why: format!("none of the shape fields {names:?} are present"),
            })
        };
        let d_model = pick_usize(&["d_model", "n_audio_state"])?;
        if value
            .get("n_text_state")
            .and_then(|v| v.as_u64())
            .is_some_and(|width| width != d_model as u64)
        {
            return Err(crate::error::ModelError::BadConfig {
                field: "n_text_state".to_string(),
                why: "encoder and decoder widths must match".to_string(),
            });
        }
        let num_attention_heads = pick_usize(&["encoder_attention_heads", "n_audio_head"])?;
        let vocab_size = pick_usize(&["vocab_size", "n_vocab"])?;
        let n_mels = pick_usize(&["num_mel_bins", "n_mels"])?;
        let num_hidden_layers = value.get("num_hidden_layers").and_then(|v| v.as_u64());
        let encoder_layers = value.get("encoder_layers").and_then(|v| v.as_u64());
        let n_audio_layer = value.get("n_audio_layer").and_then(|v| v.as_u64());
        let encoder_layers = encoder_layers
            .or(num_hidden_layers)
            .or(n_audio_layer)
            .ok_or_else(|| crate::error::ModelError::BadConfig {
                field: "config.json".to_string(),
                why: "none of encoder_layers, num_hidden_layers, n_audio_layer present".to_string(),
            })? as usize;
        let config: Self =
            serde_json::from_str(json).map_err(|e| crate::error::ModelError::BadConfig {
                field: "config.json".to_string(),
                why: format!("parsed shapes but the struct rejected the layout: {e}"),
            })?;
        // The resolved fields win over whatever serde defaulted: openai
        // layouts leave the HF-named fields at serde defaults.
        let mut config = config;
        config.d_model = d_model;
        config.num_attention_heads = num_attention_heads;
        config.vocab_size = vocab_size;
        config.n_mels = n_mels;
        config.num_hidden_layers = Some(encoder_layers);
        config.encoder_layers = Some(encoder_layers);
        // Openai-layout configs omit the FFN widths; the reference derives
        // them as four times the stream width (tiny: 384 -> 1536).
        if config.encoder_ffn_dim == 0 {
            config.encoder_ffn_dim =
                d_model
                    .checked_mul(4)
                    .ok_or_else(|| crate::error::ModelError::BadConfig {
                        field: "d_model".to_string(),
                        why: "FFN width overflows".to_string(),
                    })?;
        }
        if config.decoder_ffn_dim == 0 {
            config.decoder_ffn_dim =
                d_model
                    .checked_mul(4)
                    .ok_or_else(|| crate::error::ModelError::BadConfig {
                        field: "d_model".to_string(),
                        why: "FFN width overflows".to_string(),
                    })?;
        }
        config.validate()?;
        Ok(config)
    }

    /// Structural validation: every field the reference kernels key on.
    pub fn validate(&self) -> Result<(), crate::error::ModelError> {
        let reject = |field: &str, why: String| crate::error::ModelError::BadConfig {
            field: field.to_string(),
            why,
        };
        if self.model_type != "whisper" {
            return Err(reject(
                "model_type",
                format!("expected \"whisper\", got {:?}", self.model_type),
            ));
        }
        if self.activation_function != "gelu" {
            return Err(reject(
                "activation_function",
                format!(
                    "the kernels implement exact-erf gelu, not {:?}",
                    self.activation_function
                ),
            ));
        }
        if self.n_mels != 80 && self.n_mels != 128 {
            return Err(reject(
                "n_mels",
                format!(
                    "the kernels implement 80 or 128 mel bands, got {}",
                    self.n_mels
                ),
            ));
        }
        if self.d_model == 0 || self.num_attention_heads == 0 {
            return Err(reject(
                "d_model / num_attention_heads",
                "both must be positive".to_string(),
            ));
        }
        if self.d_model % self.num_attention_heads != 0 {
            return Err(reject(
                "d_model",
                format!(
                    "{} is not divisible by {} encoder attention heads",
                    self.d_model, self.num_attention_heads
                ),
            ));
        }
        if self.d_model < 4 || self.d_model % 2 != 0 {
            return Err(reject(
                "d_model",
                "sinusoidal positions require an even width of at least 4".to_string(),
            ));
        }
        if !self.layer_norm_eps.is_finite() || self.layer_norm_eps <= 0.0 {
            return Err(reject(
                "layer_norm_eps",
                "must be finite and positive".to_string(),
            ));
        }
        if self.d_model.checked_mul(4).is_none()
            || self
                .d_model
                .checked_mul(self.d_model)
                .and_then(|n| n.checked_mul(12))
                .is_none()
            || self.d_model.checked_mul(self.encoder_ffn_dim).is_none()
            || self.d_model.checked_mul(self.decoder_ffn_dim()).is_none()
            || self.d_model.checked_mul(self.vocab_size).is_none()
            || self
                .d_model
                .checked_mul(self.max_source_positions)
                .is_none()
            || self
                .d_model
                .checked_mul(self.max_target_positions)
                .is_none()
        {
            return Err(reject(
                "shape",
                "tensor element count overflows".to_string(),
            ));
        }
        let decoder_layers = self.decoder_layers();
        let decoder_heads = self.decoder_attention_heads();
        if decoder_heads == 0 || self.d_model % decoder_heads != 0 {
            return Err(reject(
                "decoder_attention_heads",
                format!(
                    "d_model {} must be positive and divisible by {} decoder heads",
                    self.d_model, decoder_heads
                ),
            ));
        }
        if self.encoder_layers() == 0 || decoder_layers == 0 {
            return Err(reject(
                "encoder_layers / decoder_layers",
                "both must be positive".to_string(),
            ));
        }
        if self.encoder_ffn_dim == 0 || self.decoder_ffn_dim == 0 {
            return Err(reject(
                "encoder_ffn_dim / decoder_ffn_dim",
                "both must be positive".to_string(),
            ));
        }
        if self.vocab_size == 0 {
            return Err(reject("vocab_size", "must be positive".to_string()));
        }
        if self.max_source_positions == 0 || self.max_target_positions == 0 {
            return Err(reject(
                "max_source_positions / max_target_positions",
                "both must be positive".to_string(),
            ));
        }
        Ok(())
    }

    /// Validates every tensor consumed by the runner without dequantizing it.
    /// Catalog probing and runtime loading share this refusal boundary.
    pub fn validate_weights(
        &self,
        file: &crate::safetensors::SafetensorsFile,
    ) -> Result<(), crate::error::ModelError> {
        use crate::error::ModelError;
        self.validate()?;
        let tensor = |name: &str, shape: &[usize], packed: bool| -> Result<(), ModelError> {
            let desc = file
                .descriptor(name)
                .ok_or_else(|| ModelError::TensorNotFound {
                    name: name.to_string(),
                })?;
            if desc.shape != shape {
                return Err(ModelError::IndexCorrupt {
                    detail: format!("tensor {name}: shape {:?}, expected {shape:?}", desc.shape),
                });
            }
            let dtype = desc.dtype.to_uppercase();
            let width = match (packed, dtype.as_str()) {
                (true, "U32") | (false, "F32") => 4usize,
                (false, "F16" | "BF16") => 2,
                _ => {
                    return Err(ModelError::IndexCorrupt {
                        detail: format!("tensor {name}: unsupported dtype {dtype}"),
                    })
                }
            };
            let expected = shape
                .iter()
                .try_fold(width, |n, &dim| n.checked_mul(dim))
                .ok_or_else(|| ModelError::IndexCorrupt {
                    detail: format!("tensor {name}: byte size overflows"),
                })?;
            let actual = file.raw_bytes(name)?.len();
            if actual != expected {
                return Err(ModelError::TensorSizeMismatch {
                    name: name.to_string(),
                    expected: expected as u64,
                    actual: actual as u64,
                });
            }
            Ok(())
        };
        let d = self.d_model;
        let mlx =
            self.quantization.is_some() || file.contains_tensor("decoder.token_embedding.weight");
        if mlx {
            if self.encoder_ffn_dim != 4 * d || self.decoder_ffn_dim() != 4 * d {
                return Err(ModelError::BadConfig {
                    field: "ffn_dim".to_string(),
                    why: "MLX Whisper blocks require FFN width 4 * d_model".to_string(),
                });
            }
            let linear = |base: &str,
                          out: usize,
                          input: usize,
                          bias_required: bool|
             -> Result<(), ModelError> {
                if file.contains_tensor(&format!("{base}.scales")) {
                    let quant = self.quantization.ok_or_else(|| ModelError::BadConfig {
                        field: "quantization".to_string(),
                        why: "packed weights require a quantization block".to_string(),
                    })?;
                    if !matches!(quant.bits, 4 | 8)
                        || quant.group_size != 64
                        || input % quant.group_size != 0
                    {
                        return Err(ModelError::BadConfig {
                            field: "quantization".to_string(),
                            why: format!("{base} requires 4-bit or 8-bit groups of 64 and a divisible input width"),
                        });
                    }
                    tensor(
                        &format!("{base}.weight"),
                        &[out, input / (32 / quant.bits)],
                        true,
                    )?;
                    tensor(
                        &format!("{base}.scales"),
                        &[out, input / quant.group_size],
                        false,
                    )?;
                    tensor(
                        &format!("{base}.biases"),
                        &[out, input / quant.group_size],
                        false,
                    )?;
                } else {
                    tensor(&format!("{base}.weight"), &[out, input], false)?;
                }
                let bias = format!("{base}.bias");
                if bias_required || file.contains_tensor(&bias) {
                    tensor(&bias, &[out], false)?;
                }
                Ok(())
            };
            tensor("encoder.conv1.weight", &[d, 3, self.n_mels], false)?;
            tensor("encoder.conv1.bias", &[d], false)?;
            tensor("encoder.conv2.weight", &[d, 3, d], false)?;
            tensor("encoder.conv2.bias", &[d], false)?;
            if file.contains_tensor("encoder.positional_embedding") {
                tensor(
                    "encoder.positional_embedding",
                    &[self.max_source_positions, d],
                    false,
                )?;
            }
            for norm in ["encoder.ln_post", "decoder.ln"] {
                tensor(&format!("{norm}.weight"), &[d], false)?;
                tensor(&format!("{norm}.bias"), &[d], false)?;
            }
            linear("decoder.token_embedding", self.vocab_size, d, false)?;
            tensor(
                "decoder.positional_embedding",
                &[self.max_target_positions, d],
                false,
            )?;
            for (stack, count) in [
                ("encoder", self.encoder_layers()),
                ("decoder", self.decoder_layers()),
            ] {
                for layer in 0..count {
                    let prefix = format!("{stack}.blocks.{layer}");
                    let attention: &[&str] = if stack == "encoder" {
                        &["attn"]
                    } else {
                        &["attn", "cross_attn"]
                    };
                    for attn in attention {
                        for projection in ["query", "key", "value", "out"] {
                            linear(
                                &format!("{prefix}.{attn}.{projection}"),
                                d,
                                d,
                                projection != "key",
                            )?;
                        }
                        tensor(&format!("{prefix}.{attn}_ln.weight"), &[d], false)?;
                        tensor(&format!("{prefix}.{attn}_ln.bias"), &[d], false)?;
                    }
                    linear(&format!("{prefix}.mlp1"), 4 * d, d, true)?;
                    linear(&format!("{prefix}.mlp2"), d, 4 * d, true)?;
                    tensor(&format!("{prefix}.mlp_ln.weight"), &[d], false)?;
                    tensor(&format!("{prefix}.mlp_ln.bias"), &[d], false)?;
                }
            }
        } else {
            let hf = |name: &str, shape: &[usize]| -> Result<(), ModelError> {
                let name = if file.contains_tensor(name) {
                    name.to_string()
                } else {
                    format!("model.{name}")
                };
                tensor(&name, shape, false)
            };
            hf("encoder.conv1.weight", &[d, self.n_mels, 3])?;
            hf("encoder.conv1.bias", &[d])?;
            hf("encoder.conv2.weight", &[d, d, 3])?;
            hf("encoder.conv2.bias", &[d])?;
            hf(
                "encoder.embed_positions.weight",
                &[self.max_source_positions, d],
            )?;
            hf(
                "decoder.embed_positions.weight",
                &[self.max_target_positions, d],
            )?;
            hf("decoder.embed_tokens.weight", &[self.vocab_size, d])?;
            for stack in ["encoder", "decoder"] {
                hf(&format!("{stack}.layer_norm.weight"), &[d])?;
                hf(&format!("{stack}.layer_norm.bias"), &[d])?;
            }
            for (stack, count, ffn) in [
                ("encoder", self.encoder_layers(), self.encoder_ffn_dim),
                ("decoder", self.decoder_layers(), self.decoder_ffn_dim()),
            ] {
                for layer in 0..count {
                    let prefix = format!("{stack}.layers.{layer}");
                    let attention: &[&str] = if stack == "encoder" {
                        &["self_attn"]
                    } else {
                        &["self_attn", "encoder_attn"]
                    };
                    for attn in attention {
                        for projection in ["q", "k", "v", "out"] {
                            hf(
                                &format!("{prefix}.{attn}.{projection}_proj.weight"),
                                &[d, d],
                            )?;
                            if projection != "k" {
                                hf(&format!("{prefix}.{attn}.{projection}_proj.bias"), &[d])?;
                            }
                        }
                        hf(&format!("{prefix}.{attn}_layer_norm.weight"), &[d])?;
                        hf(&format!("{prefix}.{attn}_layer_norm.bias"), &[d])?;
                    }
                    hf(&format!("{prefix}.fc1.weight"), &[ffn, d])?;
                    hf(&format!("{prefix}.fc1.bias"), &[ffn])?;
                    hf(&format!("{prefix}.fc2.weight"), &[d, ffn])?;
                    hf(&format!("{prefix}.fc2.bias"), &[d])?;
                    hf(&format!("{prefix}.final_layer_norm.weight"), &[d])?;
                    hf(&format!("{prefix}.final_layer_norm.bias"), &[d])?;
                }
            }
        }
        Ok(())
    }

    /// Encoder layer count: `encoder_layers`, falling back to the HF
    /// generic `num_hidden_layers`.
    pub fn encoder_layers(&self) -> usize {
        self.encoder_layers.or(self.num_hidden_layers).unwrap_or(0)
    }

    /// Decoder layer count, defaulting to the encoder count when the
    /// config omits it (English-only exports sometimes do).
    pub fn decoder_layers(&self) -> usize {
        if self.decoder_layers == 0 {
            self.encoder_layers()
        } else {
            self.decoder_layers
        }
    }

    /// Decoder head count, same defaulting rule.
    pub fn decoder_attention_heads(&self) -> usize {
        if self.decoder_attention_heads == 0 {
            self.num_attention_heads
        } else {
            self.decoder_attention_heads
        }
    }

    /// Decoder FFN width, same defaulting rule.
    pub fn decoder_ffn_dim(&self) -> usize {
        if self.decoder_ffn_dim == 0 {
            self.encoder_ffn_dim
        } else {
            self.decoder_ffn_dim
        }
    }

    /// Embedding scale: `sqrt(d_model)` when `scale_embedding` is set,
    /// 1.0 otherwise (the whisper reference behavior).
    pub fn embed_scale(&self) -> f32 {
        if self.scale_embedding {
            (self.d_model as f32).sqrt()
        } else {
            1.0
        }
    }

    /// Estimated allocations for the CPU runner: all weights are f32,
    /// plus decoder caches, encoder intermediates, frontend buffers, and
    /// a dequantization/loading margin. This is not a measured RSS ceiling:
    /// mapped checkpoint pages and allocator overhead remain host-dependent.
    pub fn memory_ceiling_bytes(&self) -> u64 {
        let d = self.d_model as u64;
        let ffn_e = self.encoder_ffn_dim as u64;
        let ffn_d = self.decoder_ffn_dim() as u64;
        let vocab = self.vocab_size as u64;
        let src = self.max_source_positions as u64;
        let tgt = self.max_target_positions as u64;
        let enc_layers = self.encoder_layers() as u64;
        let dec_layers = self.decoder_layers() as u64;

        // Encoder attention biases and norms contribute 7*d, the FFN
        // contributes ffn_e + d biases, and the final norm contributes 2*d.
        let conv1 = d
            .saturating_mul(self.n_mels as u64)
            .saturating_mul(3)
            .saturating_add(d);
        let conv2 = d.saturating_mul(d).saturating_mul(3).saturating_add(d);
        let enc_per_layer = d
            .saturating_mul(d)
            .saturating_mul(4)
            .saturating_add(d.saturating_mul(ffn_e).saturating_mul(2))
            .saturating_add(d.saturating_mul(8))
            .saturating_add(ffn_e);
        let encoder_elems = conv1
            .saturating_add(conv2)
            .saturating_add(src.saturating_mul(d))
            .saturating_add(enc_layers.saturating_mul(enc_per_layer))
            .saturating_add(d.saturating_mul(2));

        // Decoder self/cross attention contributes 6*d biases, its three
        // norms 6*d, and the FFN ffn_d + d, plus the final norm.
        let dec_per_layer = d
            .saturating_mul(d)
            .saturating_mul(8)
            .saturating_add(d.saturating_mul(ffn_d).saturating_mul(2))
            .saturating_add(d.saturating_mul(13))
            .saturating_add(ffn_d);
        let decoder_elems = vocab
            .saturating_mul(d)
            .saturating_add(tgt.saturating_mul(d))
            .saturating_add(dec_layers.saturating_mul(dec_per_layer))
            .saturating_add(d.saturating_mul(2));

        let weights = encoder_elems
            .saturating_add(decoder_elems)
            .saturating_mul(4);
        // KV Vec capacity may exceed logical length while appending.
        let caches = dec_layers
            .saturating_mul(d)
            .saturating_mul(4)
            .saturating_mul(src.saturating_mul(2).saturating_add(tgt.saturating_mul(4)));
        let encoder_scratch = src
            .saturating_mul(4)
            .saturating_mul(d.saturating_mul(12).saturating_add(ffn_e.saturating_mul(2)));
        // 30s PCM, complex STFT rows, mel rows, and their transient copies.
        let frontend = 480_000 * 4 + 3001 * (201 * 8 + self.n_mels as u64 * 8);
        // Largest embedding dequantization keeps packed words, unpacked
        // values, and the output alive; full f32 conversion also pages the
        // source. Account for two additional embedding-sized planes.
        let loading_margin = vocab.saturating_mul(d).saturating_mul(8);
        weights
            .saturating_add(caches)
            .saturating_add(encoder_scratch)
            .saturating_add(frontend)
            .saturating_add(loading_margin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// openai/whisper-tiny.en's config.json, reduced to the fields this
    /// crate reads.
    const TINY_EN: &str = r#"{
        "model_type": "whisper",
        "architectures": ["WhisperForConditionalGeneration"],
        "d_model": 384,
        "encoder_layers": 4,
        "decoder_layers": 4,
        "encoder_attention_heads": 6,
        "decoder_attention_heads": 6,
        "encoder_ffn_dim": 1536,
        "decoder_ffn_dim": 1536,
        "vocab_size": 51864,
        "n_mels": 80,
        "max_source_positions": 1500,
        "max_target_positions": 448,
        "activation_function": "gelu",
        "layer_norm_eps": 1e-05,
        "scale_embedding": false,
        "decoder_start_token_id": 50257,
        "eos_token_id": 50256,
        "pad_token_id": 50256
    }"#;

    #[test]
    fn accepts_the_tiny_en_shape() {
        let config = WhisperConfig::from_json_str(TINY_EN).unwrap();
        assert_eq!(config.d_model, 384);
        assert_eq!(config.n_mels, 80);
        assert_eq!(config.decoder_layers(), 4);
        assert!((config.embed_scale() - 1.0).abs() < 1e-7);
    }

    #[test]
    fn rejects_unsupported_shapes_with_reasons() {
        let cases: Vec<(String, String, &str)> = vec![
            (
                TINY_EN.replace("\"n_mels\": 80", "\"n_mels\": 64"),
                "n_mels".to_string(),
                "80 or 128",
            ),
            (
                TINY_EN.replace("\"gelu\"", "\"swish\""),
                "activation_function".to_string(),
                "gelu",
            ),
            (
                TINY_EN.replace(
                    "\"model_type\": \"whisper\"",
                    "\"model_type\": \"whisperx\"",
                ),
                "model_type".to_string(),
                "whisper",
            ),
            (
                TINY_EN.replace("\"d_model\": 384", "\"d_model\": 385"),
                "d_model".to_string(),
                "divisible",
            ),
        ];
        for (json, field, needle) in cases {
            let err = WhisperConfig::from_json_str(&json).unwrap_err();
            assert!(
                err.to_string().contains(&field) && err.to_string().contains(needle),
                "expected {field} rejection, got: {err}"
            );
        }
    }

    #[test]
    fn ceiling_is_derived_and_stable_for_tiny_en() {
        let config = WhisperConfig::from_json_str(TINY_EN).unwrap();
        // All resident weights must fit at f32, including encoder weights.
        let weight_bytes = 4 * (8_208_384 + 29_551_872);
        assert!(config.memory_ceiling_bytes() > weight_bytes);
        assert!(config.memory_ceiling_bytes() >= weight_bytes + 51864 * 384 * 8);
    }

    #[test]
    fn accepts_the_mlx_openai_layout() {
        // The exact config.json mlx-community/whisper-tiny.en-8bit ships.
        let mlx = r#"{
            "n_mels": 80,
            "n_audio_ctx": 1500,
            "n_audio_state": 384,
            "n_audio_head": 6,
            "n_audio_layer": 4,
            "n_vocab": 51864,
            "n_text_ctx": 448,
            "n_text_state": 384,
            "n_text_head": 6,
            "n_text_layer": 4,
            "quantization": {"group_size": 64, "bits": 8},
            "model_type": "whisper"
        }"#;
        let config = WhisperConfig::from_json_str(mlx).unwrap();
        assert_eq!(config.d_model, 384);
        assert_eq!(config.num_attention_heads, 6);
        assert_eq!(config.vocab_size, 51864);
        assert_eq!(config.n_mels, 80);
        assert_eq!(config.encoder_layers(), 4);
        assert_eq!(config.decoder_layers(), 4);
        assert_eq!(config.max_source_positions, 1500);
        assert_eq!(config.max_target_positions, 448);
        assert_eq!(
            config.quantization,
            Some(WhisperQuantization {
                group_size: 64,
                bits: 8
            })
        );
    }

    #[test]
    fn multilingual_token_constants_round_trip_through_a_tokenizer_fixture() {
        // The shape of a whisper tokenizer.json added_tokens array.
        let json = r#"{
            "added_tokens": [
                {"id": 50257, "content": "<|endoftext|>"},
                {"id": 50258, "content": "<|startoftranscript|>"},
                {"id": 50259, "content": "<|zh|>"},
                {"id": 50260, "content": "<|en|>"},
                {"id": 50261, "content": "<|de|>"},
                {"id": 50359, "content": "<|transcribe|>"},
                {"id": 50358, "content": "<|translate|>"},
                {"id": 50360, "content": "<|startoflm|>"},
                {"id": 50361, "content": "<|nospeech|>"},
                {"id": 50362, "content": "<|notimestamps|>"},
                {"id": 50363, "content": "<|0.00|>"},
                {"id": 50364, "content": "<|1.00|>"},
                {"id": 50365, "content": "<|2.00|>"}
            ]
        }"#;
        let tokens = WhisperSpecialTokens::from_tokenizer_json(json).unwrap();
        assert_eq!(tokens.eot, 50257);
        assert_eq!(tokens.sot, 50258);
        assert_eq!(tokens.first_language, 50259);
        assert_eq!(tokens.last_language, 50261);
        assert_eq!(tokens.transcribe, 50359);
        assert_eq!(tokens.no_timestamps, 50362);
        assert_eq!(tokens.first_timestamp, 50363);
        assert!(tokens.is_language(50260));
        assert!(!tokens.is_language(50363));
        assert!(tokens.is_timestamp(50364));
        assert!(!tokens.is_timestamp(50260));
    }

    #[test]
    fn tokenizer_variant_uses_text_boundary_and_exact_zero_timestamp() {
        let fixture = |eot: u32, silence: &str| {
            serde_json::json!({"added_tokens": [
                {"id":eot,"content":"<|endoftext|>"},
                {"id":eot+1,"content":"<|startoftranscript|>"},
                {"id":eot+2,"content":"<|en|>"},
                {"id":eot+102,"content":"<|transcribe|>"},
                {"id":eot+105,"content":silence},
                {"id":eot+106,"content":"<|notimestamps|>"},
                {"id":eot+108,"content":"<|1.00|>"},
                {"id":eot+107,"content":"<|0.00|>"}
            ]})
            .to_string()
        };
        let english =
            WhisperSpecialTokens::from_tokenizer_json(&fixture(50256, "<|nospeech|>")).unwrap();
        assert!(!english.multilingual);
        assert_eq!(english.first_timestamp, 50363);
        let multilingual =
            WhisperSpecialTokens::from_tokenizer_json(&fixture(50257, "<|nocaptions|>")).unwrap();
        assert!(multilingual.multilingual);
        assert_eq!(multilingual.no_speech, Some(50362));
        assert_eq!(multilingual.first_timestamp, 50364);
        let mut config = WhisperConfig::from_json_str(TINY_EN).unwrap();
        english.validate_config(&config).unwrap();
        assert!(multilingual.validate_config(&config).is_err());
        config.vocab_size = 51865;
        config.decoder_start_token_id = Some(multilingual.sot);
        config.eos_token_id = Some(multilingual.eot);
        multilingual.validate_config(&config).unwrap();
        let mut invalid = multilingual;
        invalid.first_timestamp = 51865;
        assert!(invalid.validate_config(&config).is_err());
    }

    #[test]
    fn tokenizer_json_without_specials_is_refused() {
        let err = WhisperSpecialTokens::from_tokenizer_json("{\"added_tokens\": []}").unwrap_err();
        assert!(err.to_string().contains("endoftext"), "{err}");
    }
}
