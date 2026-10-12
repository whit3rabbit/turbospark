//! Zonos v2 family (Zyphra ZONOS2).
//!
//! Port of `mlx_audio/tts/models/zonos2/` (model.py, config.py,
//! generation.py, prompt.py) at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/zonos2):
//! a custom backbone (summed per-column multi-embedding, gated attention
//! with weightless q/k RMSNorm and interleaved rope, fused-residual RMS
//! blocks, dense or router-MoE SwitchGLU feed-forward with the router
//! hidden-state thread) over a 9+1 column delayed-audio frame layout,
//! with the waveform decode through the Descript Audio Codec 44 kHz
//! port. See [README.md](README.md) for the verification evidence and
//! open items.
//!
//! Scope of this port, matching the reference single-segment path:
//! batch size 1 and non-streaming. Text runs through the reference
//! byte-level id encoding (BOS, utf-8 bytes + 192, EOS) exactly like
//! the reference prompt builder; the optional `TTSTextNormalizer` is a
//! caller concern. Speaker conditioning takes a caller-supplied
//! embedding vector; extracting it from reference audio needs the
//! separate downloadable speaker encoder and stays open.

use std::collections::HashMap;

use serde_json::Value;

use crate::codec::descript::Dac;
use crate::lm;
use crate::lm::switch_layers::SwitchGlu;
use crate::music::minimax_music3::rng;
use crate::nn::{Linear, RmsNorm};
use crate::tts::sampling;
use crate::{ops, quant, Result, SpeechError};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// `prompt.py` byte-vocabulary constants.
pub const PAD_ID: i32 = 0;
/// `prompt.py` byte-vocabulary constants.
pub const UNK_ID: i32 = 1;
/// `prompt.py` byte-vocabulary constants.
pub const BOS_ID: i32 = 2;
/// `prompt.py` byte-vocabulary constants.
pub const EOS_ID: i32 = 3;
/// Legacy symbol vocabulary below the byte ids.
pub const LEGACY_SYMBOL_VOCAB_SIZE: i32 = 192;
/// `LEGACY_SYMBOL_VOCAB_SIZE + 256`; `text_vocab` must cover it.
pub const BYTE_TEXT_VOCAB_SIZE: usize = 448;

/// The `_SILENCE_TOKENS_0_2S` table (17 frames of sheared silence).
pub const SILENCE_TOKENS_0_2S: [[i32; 9]; 17] = [
    [568, 778, 338, 524, 967, 360, 728, 550, 90],
    [568, 778, 10, 674, 364, 981, 741, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 778, 721, 842, 264, 974, 989, 507, 308],
];

/// `text_to_byte_ids`: BOS, the utf-8 bytes shifted above the legacy
/// symbols, EOS.
pub fn text_to_byte_ids(text: &str) -> Vec<i32> {
    let mut ids = vec![BOS_ID];
    ids.extend(
        text.as_bytes()
            .iter()
            .map(|&b| b as i32 + LEGACY_SYMBOL_VOCAB_SIZE),
    );
    ids.push(EOS_ID);
    ids
}

/// The conditioning-id base (`_conditioning_base_text_vocab`).
fn conditioning_base(
    text_vocab: usize,
    speaking_rate_num_buckets: usize,
    quality_bucket_counts: &[usize],
    speaker_background_num_buckets: usize,
    accurate_mode_num_buckets: usize,
    context: &str,
) -> Result<usize> {
    let base = text_vocab
        .checked_sub(speaking_rate_num_buckets)
        .and_then(|v| v.checked_sub(quality_bucket_counts.iter().sum::<usize>()))
        .and_then(|v| v.checked_sub(speaker_background_num_buckets))
        .and_then(|v| v.checked_sub(accurate_mode_num_buckets));
    base.ok_or_else(|| SpeechError::BadConfig {
        field: "text_vocab".into(),
        why: format!("text_vocab is too small for {context}"),
    })
}

/// `speaking_rate_token_id`.
pub fn speaking_rate_token_id(
    text_vocab: usize,
    speaking_rate_num_buckets: usize,
    speaking_rate_bucket: usize,
    quality_bucket_counts: &[usize],
    speaker_background_num_buckets: usize,
    accurate_mode_num_buckets: usize,
) -> Result<i32> {
    if speaking_rate_bucket >= speaking_rate_num_buckets {
        return Err(SpeechError::Input {
            why: format!(
                "speaking_rate_bucket must be in [0, {}]",
                speaking_rate_num_buckets.saturating_sub(1)
            ),
        });
    }
    let base = conditioning_base(
        text_vocab,
        speaking_rate_num_buckets,
        quality_bucket_counts,
        speaker_background_num_buckets,
        accurate_mode_num_buckets,
        "speaking-rate conditioning",
    )?;
    Ok((base + speaking_rate_bucket) as i32)
}

/// `quality_token_id`.
pub fn quality_token_id(
    text_vocab: usize,
    speaking_rate_num_buckets: usize,
    quality_bucket_counts: &[usize],
    feature_idx: usize,
    quality_bucket: usize,
    speaker_background_num_buckets: usize,
    accurate_mode_num_buckets: usize,
) -> Result<i32> {
    if feature_idx >= quality_bucket_counts.len() {
        return Err(SpeechError::Input {
            why: format!("quality feature index out of range: {feature_idx}"),
        });
    }
    if quality_bucket >= quality_bucket_counts[feature_idx] {
        return Err(SpeechError::Input {
            why: format!(
                "quality bucket for feature {feature_idx} must be in [0, {}]",
                quality_bucket_counts[feature_idx].saturating_sub(1)
            ),
        });
    }
    let base = conditioning_base(
        text_vocab,
        speaking_rate_num_buckets,
        quality_bucket_counts,
        speaker_background_num_buckets,
        accurate_mode_num_buckets,
        "quality conditioning",
    )?;
    let offset = speaking_rate_num_buckets
        + quality_bucket_counts[..feature_idx].iter().sum::<usize>()
        + quality_bucket;
    Ok((base + offset) as i32)
}

/// `speaker_background_token_id`.
pub fn speaker_background_token_id(
    text_vocab: usize,
    speaking_rate_num_buckets: usize,
    quality_bucket_counts: &[usize],
    clean: bool,
    speaker_background_num_buckets: usize,
    accurate_mode_num_buckets: usize,
) -> Result<i32> {
    if speaker_background_num_buckets < 2 {
        return Err(SpeechError::Input {
            why: "speaker_background_num_buckets must be at least 2".into(),
        });
    }
    let base = conditioning_base(
        text_vocab,
        speaking_rate_num_buckets,
        quality_bucket_counts,
        speaker_background_num_buckets,
        accurate_mode_num_buckets,
        "speaker-background conditioning",
    )?;
    let offset = speaking_rate_num_buckets
        + quality_bucket_counts.iter().sum::<usize>()
        + usize::from(!clean);
    Ok((base + offset) as i32)
}

/// `accurate_mode_token_id`.
pub fn accurate_mode_token_id(
    text_vocab: usize,
    speaking_rate_num_buckets: usize,
    quality_bucket_counts: &[usize],
    speaker_background_num_buckets: usize,
    accurate_mode_num_buckets: usize,
) -> Result<i32> {
    if accurate_mode_num_buckets == 0 {
        return Err(SpeechError::Input {
            why: "accurate_mode_num_buckets must be positive".into(),
        });
    }
    if speaker_background_num_buckets < 2 {
        return Err(SpeechError::Input {
            why: "speaker_background_num_buckets must be at least 2".into(),
        });
    }
    let base = conditioning_base(
        text_vocab,
        speaking_rate_num_buckets,
        quality_bucket_counts,
        speaker_background_num_buckets,
        accurate_mode_num_buckets,
        "accurate-mode conditioning",
    )?;
    let offset = speaking_rate_num_buckets
        + quality_bucket_counts.iter().sum::<usize>()
        + speaker_background_num_buckets;
    Ok((base + offset) as i32)
}

/// `shear`: the fixed prefix-delayed representation of a
/// `[time, codebooks]` table.
pub fn shear(x: &[Vec<i32>], pad: i32) -> Result<Vec<Vec<i32>>> {
    let t = x.len();
    let c = x
        .first()
        .map(|r| r.len())
        .ok_or_else(|| SpeechError::Input {
            why: "shear expects a non-empty table".to_string(),
        })?;
    if x.iter().any(|r| r.len() != c) {
        return Err(SpeechError::Input {
            why: "shear expects a rectangular table".to_string(),
        });
    }
    let mut out = vec![vec![pad; c]; t];
    for i in 0..t {
        for j in 0..c {
            // padded[(c - 1) + i][j] = x[i][j]; the gather reads
            // padded[(c - 1 + i) - j][j], which is x[i - j][j] once the
            // c - 1 pad rows drop off.
            if i >= j {
                out[i][j] = x[i - j][j];
            }
        }
    }
    Ok(out)
}

/// `shear_up`: undo the delay for a `[time, codebooks]` table.
pub fn shear_up(x: &[Vec<i32>], pad: i32) -> Result<Vec<Vec<i32>>> {
    let h = x.len();
    let w = x
        .first()
        .map(|r| r.len())
        .ok_or_else(|| SpeechError::Input {
            why: "shear_up expects a non-empty table".to_string(),
        })?;
    if x.iter().any(|r| r.len() != w) {
        return Err(SpeechError::Input {
            why: "shear_up expects a rectangular table".to_string(),
        });
    }
    let mut out = vec![vec![pad; w]; h];
    for (t, out_row) in out.iter_mut().enumerate() {
        for (c, slot) in out_row.iter_mut().enumerate() {
            let row = t + c;
            *slot = if row < h { x[row][c] } else { pad };
        }
    }
    Ok(out)
}

/// `silence_prompt_tokens`: the fixed table sliced to `n_codebooks`,
/// sheared, with the text column filled.
pub fn silence_prompt_tokens(
    n_codebooks: usize,
    audio_pad_id: i32,
    text_vocab: i32,
) -> Vec<Vec<i32>> {
    let sliced: Vec<Vec<i32>> = SILENCE_TOKENS_0_2S
        .iter()
        .map(|row| row[..n_codebooks].to_vec())
        .collect();
    let sheared = shear(&sliced, audio_pad_id).unwrap_or_default();
    sheared
        .into_iter()
        .map(|mut row| {
            row.push(text_vocab);
            row
        })
        .collect()
}

/// Per-step sampling seed. `seed + step` made seed `s` at step `k` equal
/// seed `s + k` at step 0, so nearby requests shared draws. Two rounds of
/// splitmix64 (Steele et al.) break that linear relation.
fn step_seed(seed: u64, step: usize) -> u64 {
    fn splitmix64(mut z: u64) -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    splitmix64(splitmix64(seed).wrapping_add(step as u64))
}

/// `Zonos2Config` (config.py), fully resolved.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub n_layers: usize,
    pub dim: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub intermediate_size: usize,
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub max_seqlen: usize,
    pub n_codebooks: usize,
    pub codebook_size: usize,
    pub eoa_id: i32,
    pub audio_pad_id: i32,
    pub text_vocab: usize,
    pub loss_softcap: f32,
    pub sample_rate: u32,
    pub speaker_enabled: bool,
    pub speaker_embedding_dim: usize,
    pub speaker_lda_dim: Option<usize>,
    pub speaking_rate_num_buckets: usize,
    pub quality_bucket_counts: Vec<usize>,
    pub speaker_background_num_buckets: usize,
    pub accurate_mode_num_buckets: usize,
    pub moe_n_experts: usize,
    pub moe_router_topk: usize,
    /// Per-layer top-k overrides (`special_topk_layers`).
    pub special_topk_layers: HashMap<usize, usize>,
    pub moe_router_dim: usize,
    pub moe_start_from_layer: usize,
    pub moe_end_from_layer: usize,
    pub norm_topk_prob: bool,
    pub moe_balancing_strategy: String,
}

impl ModelConfig {
    /// `Zonos2Config.from_dict` over a checkpoint `config.json` (the
    /// nested `model` dict flattens; unknown keys drop).
    pub fn from_json(value: &Value) -> Result<Self> {
        let data = match value.get("model") {
            Some(model) if model.is_object() => {
                let mut merged = model.as_object().unwrap().clone();
                if let Some(top) = value.as_object() {
                    for (k, v) in top {
                        merged.entry(k.clone()).or_insert_with(|| v.clone());
                    }
                }
                Value::Object(merged)
            }
            _ => value.clone(),
        };
        let uint = |key: &str, default: u64| -> usize {
            data.get(key).and_then(Value::as_u64).unwrap_or(default) as usize
        };
        let f64v = |key: &str, default: f64| -> f64 {
            data.get(key).and_then(Value::as_f64).unwrap_or(default)
        };
        let dim = uint("dim", 2048);
        let head_dim = uint("head_dim", 128);
        // Checked before `dim / head_dim` below: the n_heads default is
        // computed eagerly, so a zero head_dim would panic there first.
        if head_dim == 0 {
            return Err(SpeechError::BadConfig {
                field: "head_dim".into(),
                why: "head_dim must be > 0".to_string(),
            });
        }
        let n_heads = data
            .get("n_heads")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or(dim / head_dim);
        let n_kv_heads = data
            .get("n_kv_heads")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or(4);
        let ffn_dim_multiplier = f64v("ffn_dim_multiplier", 1.5);
        let multiple_of = uint("multiple_of", 256);
        if multiple_of == 0 {
            return Err(SpeechError::BadConfig {
                field: "multiple_of".into(),
                why: "multiple_of must be > 0".to_string(),
            });
        }
        let raw = (ffn_dim_multiplier * dim as f64) as usize;
        let intermediate_size = multiple_of * raw.div_ceil(multiple_of);
        let text_vocab = data
            .get("text_vocab")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .ok_or_else(|| SpeechError::BadConfig {
                field: "text_vocab".into(),
                why: "ZONOS2 requires text_vocab".to_string(),
            })?;
        let mut special_topk_layers = HashMap::new();
        if let Some(map) = data.get("special_topk_layers").and_then(Value::as_object) {
            for (k, v) in map {
                let layer: usize = k.parse().map_err(|_| SpeechError::BadConfig {
                    field: "special_topk_layers".into(),
                    why: format!("layer key {k:?} is not an integer"),
                })?;
                // A missing or non-integer value used to become top-1
                // silently; that changes routing, so refuse it.
                let topk = v.as_u64().ok_or_else(|| SpeechError::BadConfig {
                    field: "special_topk_layers".into(),
                    why: format!("top-k for layer {k:?} is not a non-negative integer"),
                })?;
                special_topk_layers.insert(layer, topk as usize);
            }
        }
        // The explicit counts win; otherwise the labeled
        // `quality_buckets` mapping derives them (the reference
        // `__post_init__`).
        let quality_bucket_counts: Vec<usize> = match data.get("quality_bucket_counts") {
            Some(entries) if entries.as_array().map(|a| !a.is_empty()).unwrap_or(false) => entries
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default(),
            _ => {
                let buckets = data.get("quality_buckets").and_then(Value::as_object);
                match data.get("quality_features").and_then(Value::as_array) {
                    // Each feature maps to its bucket count by position, so a
                    // non-string or unknown name must fail rather than be
                    // skipped (which shifts later indices) or counted as 0.
                    Some(features) if !features.is_empty() => features
                        .iter()
                        .map(|f| {
                            let name = f.as_str().ok_or_else(|| SpeechError::BadConfig {
                                field: "quality_features".into(),
                                why: format!("feature {f} is not a string"),
                            })?;
                            buckets
                                .and_then(|b| b.get(name))
                                .and_then(Value::as_array)
                                .map(|a| a.len())
                                .ok_or_else(|| SpeechError::BadConfig {
                                    field: "quality_features".into(),
                                    why: format!("feature {name:?} has no list in quality_buckets"),
                                })
                        })
                        .collect::<Result<Vec<usize>>>()?,
                    _ => buckets
                        .map(|b| {
                            b.values()
                                .map(|v| v.as_array().map(|a| a.len()).unwrap_or(0))
                        })
                        .map(|counts| counts.collect())
                        .unwrap_or_default(),
                }
            }
        };
        let speaker_enabled = data
            .get("speaker_enabled")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let speaker_lda_dim = data
            .get("speaker_lda_dim")
            .and_then(Value::as_u64)
            .map(|v| v as usize);
        // The router only tests for "quantile" and treats everything else
        // as legacy, so a typo would silently change routing. Accept the
        // two known names only; an absent or null field keeps the legacy
        // default.
        let balancing = match data.get("moe_balancing_strategy").filter(|v| !v.is_null()) {
            None => "legacy".to_string(),
            Some(value) => {
                let raw = value.as_str().ok_or_else(|| SpeechError::BadConfig {
                    field: "moe_balancing_strategy".into(),
                    why: format!("expected a string, got {value}"),
                })?;
                let normalized = raw.trim().to_lowercase().replace('-', "_");
                if normalized != "legacy" && normalized != "quantile" {
                    return Err(SpeechError::BadConfig {
                        field: "moe_balancing_strategy".into(),
                        why: format!(
                            "unknown strategy {raw:?}; expected \"legacy\" or \"quantile\""
                        ),
                    });
                }
                normalized
            }
        };
        let config = Self {
            n_layers: uint("n_layers", 28),
            dim,
            head_dim,
            n_heads,
            n_kv_heads,
            intermediate_size,
            norm_eps: f64v("norm_eps", 1e-5) as f32,
            rope_theta: f64v("rope_theta", 10000.0) as f32,
            max_seqlen: uint("max_seqlen", 6144),
            n_codebooks: uint("n_codebooks", 9),
            codebook_size: uint("codebook_size", 1024),
            eoa_id: uint("eoa_id", 1024) as i32,
            audio_pad_id: uint("audio_pad_id", 1025) as i32,
            text_vocab,
            loss_softcap: f64v("loss_softcap", 15.0) as f32,
            sample_rate: uint("sample_rate", 44100) as u32,
            speaker_enabled,
            speaker_embedding_dim: uint("speaker_embedding_dim", 2048),
            speaker_lda_dim,
            speaking_rate_num_buckets: uint("speaking_rate_num_buckets", 8),
            quality_bucket_counts,
            speaker_background_num_buckets: usize::from(
                data.get("speaker_background_token_enabled")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
            ) * 2,
            accurate_mode_num_buckets: usize::from(
                data.get("accurate_mode_token_enabled")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
            ),
            moe_n_experts: uint("moe_n_experts", 16),
            moe_router_topk: uint("moe_router_topk", 1),
            special_topk_layers,
            moe_router_dim: uint("moe_router_dim", 128),
            moe_start_from_layer: uint("moe_start_from_layer", 3),
            moe_end_from_layer: uint("moe_end_from_layer", 1),
            norm_topk_prob: data
                .get("norm_topk_prob")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            moe_balancing_strategy: balancing,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.text_vocab < BYTE_TEXT_VOCAB_SIZE {
            return Err(SpeechError::BadConfig {
                field: "text_vocab".into(),
                why: format!("text_vocab must include byte IDs (>= {BYTE_TEXT_VOCAB_SIZE})"),
            });
        }
        conditioning_base(
            self.text_vocab,
            self.speaking_rate_num_buckets,
            &self.quality_bucket_counts,
            self.speaker_background_num_buckets,
            self.accurate_mode_num_buckets,
            "prompt configuration",
        )?;
        if self.n_heads == 0 || self.dim % self.head_dim != 0 {
            return Err(SpeechError::BadConfig {
                field: "head_dim".into(),
                why: format!("dim {} must be a multiple of head_dim", self.dim),
            });
        }
        // Attention maps query head h to KV head h / (n_heads / n_kv_heads),
        // so the ratio must be an exact positive integer.
        if self.n_kv_heads == 0 || self.n_heads % self.n_kv_heads != 0 {
            return Err(SpeechError::BadConfig {
                field: "n_kv_heads".into(),
                why: format!(
                    "n_heads {} must be a positive multiple of n_kv_heads {}",
                    self.n_heads, self.n_kv_heads
                ),
            });
        }
        // The silence prompt table has one column per codebook; wider
        // configs would slice past the row, and 0 codebooks emit nothing.
        let silence_width = SILENCE_TOKENS_0_2S[0].len();
        if self.n_codebooks == 0 || self.n_codebooks > silence_width {
            return Err(SpeechError::BadConfig {
                field: "n_codebooks".into(),
                why: format!(
                    "n_codebooks must be in 1..={silence_width}, got {}",
                    self.n_codebooks
                ),
            });
        }
        // The DAC code clamp is `0..=codebook_size - 1`.
        if self.codebook_size == 0 {
            return Err(SpeechError::BadConfig {
                field: "codebook_size".into(),
                why: "codebook_size must be > 0".to_string(),
            });
        }
        // Top-k selects that many experts per token, so it must be in
        // 1..=moe_n_experts. Only checked when MoE layers exist: with
        // one expert or fewer, the router is never built.
        if self.moe_n_experts > 1 {
            let topks = std::iter::once(self.moe_router_topk)
                .chain(self.special_topk_layers.values().copied());
            for topk in topks {
                if topk == 0 || topk > self.moe_n_experts {
                    return Err(SpeechError::BadConfig {
                        field: "moe_router_topk".into(),
                        why: format!(
                            "top-k {topk} must be in 1..={} (moe_n_experts)",
                            self.moe_n_experts
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    /// `is_moe_layer`.
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        if self.moe_n_experts <= 1 {
            return false;
        }
        if layer_idx < self.moe_start_from_layer {
            return false;
        }
        if self.n_layers.saturating_sub(layer_idx) <= self.moe_end_from_layer {
            return false;
        }
        true
    }

    /// `num_experts_per_tok`.
    pub fn num_experts_per_tok(&self, layer_idx: usize) -> usize {
        self.special_topk_layers
            .get(&layer_idx)
            .copied()
            .unwrap_or(self.moe_router_topk)
    }

    /// The generated-frame width: `frame_width`.
    pub fn frame_width(&self) -> usize {
        self.n_codebooks + 1
    }

    /// `audio_vocab_size`.
    pub fn audio_vocab_size(&self) -> usize {
        self.codebook_size + 2
    }
}

/// The fused RMS norm (`Zonos2FusedRMSNorm`): fold the residual, then
/// normalize it, returning both the normed value and the new residual.
/// The affine weight is a plain multiply; the weightless variant (the
/// embedding norm) normalizes only, exactly like `mx.fast.rms_norm`
/// with a null weight (the ones weight is bit-identical).
#[derive(Debug, Clone)]
struct FusedRmsNorm {
    norm: RmsNorm,
}

impl FusedRmsNorm {
    fn new(weight: Option<Vec<f32>>, eps: f32, dim: usize) -> Self {
        Self {
            norm: RmsNorm::new(weight.unwrap_or_else(|| vec![1.0; dim]), eps),
        }
    }

    fn forward(&self, x: &[f32], len: usize, residual: Option<&[f32]>) -> (Vec<f32>, Vec<f32>) {
        let residual = match residual {
            Some(r) => {
                let mut r = r.to_vec();
                for (dst, src) in r.iter_mut().zip(x) {
                    *dst += src;
                }
                r
            }
            None => x.to_vec(),
        };
        let mut normed = residual.clone();
        self.norm.apply(&mut normed, len);
        (normed, residual)
    }
}

/// One decoder block (`Zonos2Block`).
struct Block {
    attention: Zonos2Attention,
    attention_norm: FusedRmsNorm,
    ffn_norm: FusedRmsNorm,
    feed_forward: FeedForward,
}

/// The gated attention (`Zonos2Attention`): weightless q/k RMSNorm, the
/// learned per-head `temp` applied AFTER the q norm (the norm is
/// scale-invariant, so folding `temp` into `wq` would be normalized
/// away), interleaved rope, GQA attention, and the per-head sigmoid gate
/// between the attention cores and `wo`.
struct Zonos2Attention {
    wq: Linear,
    k_proj: Linear,
    v_proj: Linear,
    wo: Linear,
    gater: Linear,
    /// Absolute per-head q scales (`|temp|`), length `n_heads`.
    temp: Vec<f32>,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    rope: lm::rope::Rope,
}

impl Zonos2Attention {
    /// Runs the gated attention over `x [rows, dim]` at rope `offset`,
    /// updating `cache` when present. Returns `[rows, dim]`.
    fn forward(
        &self,
        x: &[f32],
        rows: usize,
        offset: usize,
        mut cache: Option<&mut dyn lm::cache::LayerKv>,
    ) -> Result<Vec<f32>> {
        let dim = self.n_heads * self.head_dim;
        // gate = sigmoid(gater(x)) on the normed input.
        let gates: Vec<f32> = self
            .gater
            .forward(x, rows)
            .iter()
            .map(|g| 1.0 / (1.0 + (-g).exp()))
            .collect();
        let q = self.wq.forward(x, rows);
        let k = self.k_proj.forward(x, rows);
        let v = self.v_proj.forward(x, rows);
        // [heads, rows, head_dim] head-major views for the norms, the
        // temp scale, rope, and the attention loop.
        let mut qh = head_major(&q, rows, self.n_heads, self.head_dim);
        let mut kh = head_major(&k, rows, self.n_kv_heads, self.head_dim);
        let vh = head_major(&v, rows, self.n_kv_heads, self.head_dim);
        self.q_norm.apply(&mut qh, self.n_heads * rows);
        self.k_norm.apply(&mut kh, self.n_kv_heads * rows);
        // The reference multiplies q by |temp| after the norm, before
        // rope (linear, so the placement relative to rope is free; the
        // reference order is kept).
        for (h, factor) in self.temp.iter().enumerate() {
            for t in 0..rows {
                let row =
                    &mut qh[(h * rows + t) * self.head_dim..(h * rows + t + 1) * self.head_dim];
                for value in row {
                    *value *= factor;
                }
            }
        }
        self.rope.apply(&mut qh, self.n_heads, rows, offset);
        self.rope.apply(&mut kh, self.n_kv_heads, rows, offset);
        let (keys, values) = match &mut cache {
            Some(cache) => cache.update_and_fetch(&kh, &vh, self.n_kv_heads, rows, self.head_dim),
            None => (kh.as_slice(), vh.as_slice()),
        };
        let total = keys.len() / (self.n_kv_heads * self.head_dim);

        // Masked softmax attention per query head and position; the
        // causal bound is `j <= offset + i` (the reference "causal" mask;
        // single-position decode steps see the whole buffer).
        let rep = self.n_heads / self.n_kv_heads;
        let mut out = vec![0.0f32; self.n_heads * rows * self.head_dim];
        let mut scores = vec![0.0f32; total];
        for head in 0..self.n_heads {
            let kv_head = head / rep;
            let q_head = &qh[head * rows * self.head_dim..(head + 1) * rows * self.head_dim];
            let k_head =
                &keys[kv_head * total * self.head_dim..(kv_head + 1) * total * self.head_dim];
            let v_head =
                &values[kv_head * total * self.head_dim..(kv_head + 1) * total * self.head_dim];
            for i in 0..rows {
                let g = offset + i;
                let (low, high) = if rows == 1 {
                    (0, total - 1)
                } else {
                    (0, g.min(total - 1))
                };
                let q_row = &q_head[i * self.head_dim..(i + 1) * self.head_dim];
                let mut max = f32::NEG_INFINITY;
                for j in low..=high {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot += q_row[d] * k_head[j * self.head_dim + d];
                    }
                    let score = dot * self.scale;
                    scores[j] = score;
                    if score > max {
                        max = score;
                    }
                }
                let mut sum = 0.0f32;
                for score in &mut scores[low..=high] {
                    let weight = (*score - max).exp();
                    *score = weight;
                    sum += weight;
                }
                let inv = 1.0 / sum;
                let out_row = &mut out
                    [(head * rows + i) * self.head_dim..(head * rows + i + 1) * self.head_dim];
                for j in low..=high {
                    let weight = scores[j] * inv;
                    for d in 0..self.head_dim {
                        out_row[d] += weight * v_head[j * self.head_dim + d];
                    }
                }
            }
        }

        // Back to token-major [rows, heads * head_dim], then the per-head
        // gate after the attention cores and before wo.
        let mut joined = vec![0.0f32; rows * dim];
        for t in 0..rows {
            for head in 0..self.n_heads {
                let src = (head * rows + t) * self.head_dim;
                let dst = t * dim + head * self.head_dim;
                joined[dst..dst + self.head_dim].copy_from_slice(&out[src..src + self.head_dim]);
            }
        }
        for t in 0..rows {
            for head in 0..self.n_heads {
                let g = gates[t * self.n_heads + head];
                for d in 0..self.head_dim {
                    joined[t * dim + head * self.head_dim + d] *= g;
                }
            }
        }
        Ok(self.wo.forward(&joined, rows))
    }
}

/// Reinterprets `[rows, heads * dim]` as `[heads, rows, dim]`.
fn head_major(x: &[f32], rows: usize, heads: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; heads * rows * dim];
    for t in 0..rows {
        for h in 0..heads {
            let src = t * heads * dim + h * dim;
            let dst = (h * rows + t) * dim;
            out[dst..dst + dim].copy_from_slice(&x[src..src + dim]);
        }
    }
    out
}

enum FeedForward {
    Dense {
        /// Fused `w_in`: rows `[0..inter]` are the value half, rows
        /// `[inter..2 * inter]` the gate half.
        w_in: Linear,
        w_out: Linear,
        intermediate: usize,
    },
    MoE {
        router: Router,
        experts: SwitchGlu,
        norm_topk_prob: bool,
    },
}

impl FeedForward {
    fn forward(
        &self,
        x: &[f32],
        len: usize,
        router_states: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        match self {
            FeedForward::Dense {
                w_in,
                w_out,
                intermediate,
            } => {
                // The fused w_in: rows [0..inter] are the value half,
                // rows [inter..2 * inter] the gate half.
                let fused = w_in.forward(x, len);
                let mut gated = vec![0.0f32; intermediate * len];
                for t in 0..len {
                    let row = &fused[t * 2 * intermediate..(t + 1) * 2 * intermediate];
                    for i in 0..*intermediate {
                        let h = row[i];
                        let gate = row[intermediate + i];
                        gated[t * intermediate + i] = h * (gate / (1.0 + (-gate).exp()));
                    }
                }
                Ok((w_out.forward(&gated, len), None))
            }
            FeedForward::MoE {
                router,
                experts,
                norm_topk_prob,
            } => {
                let (topk_weights, topk_ids, next_states) =
                    router.forward(x, len, router_states)?;
                let slots = topk_ids.len() / len;
                let mut weights = topk_weights;
                if *norm_topk_prob {
                    for t in 0..len {
                        let row = &mut weights[t * slots..(t + 1) * slots];
                        let sum: f32 = row.iter().sum::<f32>() + 1e-8;
                        for w in row.iter_mut() {
                            *w /= sum;
                        }
                    }
                }
                let expert_outputs = experts.forward(x, len, &topk_ids)?;
                let out_dim = experts.output_dims();
                let mut out = vec![0.0f32; len * out_dim];
                for t in 0..len {
                    for slot in 0..slots {
                        let w = weights[t * slots + slot];
                        let src = &expert_outputs
                            [(t * slots + slot) * out_dim..(t * slots + slot + 1) * out_dim];
                        let dst = &mut out[t * out_dim..(t + 1) * out_dim];
                        for (d, value) in dst.iter_mut().enumerate() {
                            *value += w * src[d];
                        }
                    }
                }
                Ok((out, Some(next_states)))
            }
        }
    }
}

/// The `Zonos2Router`: the projected router state thread, the
/// two-gelu MLP, softmax probabilities, and top-k selection on the
/// balanced scores `probs + balancing_biases` (legacy) or
/// `probs - balancing_biases` (quantile), with the routing WEIGHTS taken
/// from the raw probabilities. Top-1 is an argmax; larger k sorts
/// descending with ties to the lower expert.
struct Router {
    down_proj: Linear,
    l0: Linear,
    l2: Linear,
    l4: Linear,
    rmsnorm_eda: RmsNorm,
    router_states_scale: Option<Vec<f32>>,
    /// Per-expert balancing biases (`balancing_biases`); the fixture
    /// generator randomizes them, so they are load-bearing for parity.
    balancing_biases: Vec<f32>,
    top_k: usize,
    use_legacy_balancing: bool,
    num_experts: usize,
}

impl Router {
    fn forward(
        &self,
        x: &[f32],
        len: usize,
        router_states: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Vec<i32>, Vec<f32>)> {
        let mut hidden = self.down_proj.forward(x, len);
        if let (Some(states), Some(scale)) = (router_states, &self.router_states_scale) {
            // router_states_scale broadcasts over the router dim per
            // token: hidden + states * scale.
            for (h, (s, scale)) in hidden
                .iter_mut()
                .zip(states.iter().zip(scale.iter().cycle()))
            {
                *h += *s * scale;
            }
        }
        let next_states = hidden.clone();
        self.rmsnorm_eda.apply(&mut hidden, len);
        let mut h = self.l0.forward(&hidden, len);
        for v in &mut h {
            ops::gelu_erf(std::slice::from_mut(v));
        }
        let mut h = self.l2.forward(&h, len);
        for v in &mut h {
            ops::gelu_erf(std::slice::from_mut(v));
        }
        let logits = self.l4.forward(&h, len);
        // Softmax per token over the experts.
        let mut probs = vec![0.0f32; len * self.num_experts];
        for t in 0..len {
            let row = &logits[t * self.num_experts..(t + 1) * self.num_experts];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            let prob = &mut probs[t * self.num_experts..(t + 1) * self.num_experts];
            for (p, &l) in prob.iter_mut().zip(row) {
                *p = (l - max).exp();
                sum += *p;
            }
            for p in prob.iter_mut() {
                *p /= sum;
            }
        }
        // Selection runs on the balanced scores; the weights stay the raw
        // probabilities of the picked experts.
        let sign = if self.use_legacy_balancing { 1.0 } else { -1.0 };
        let mut topk_ids = vec![0i32; len * self.top_k];
        let mut topk_weights = vec![0.0f32; len * self.top_k];
        for t in 0..len {
            let row = &probs[t * self.num_experts..(t + 1) * self.num_experts];
            let mut order: Vec<usize> = (0..self.num_experts).collect();
            order.sort_by(|&a, &b| {
                let sa = row[a] + sign * self.balancing_biases[a];
                let sb = row[b] + sign * self.balancing_biases[b];
                sb.partial_cmp(&sa)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            for slot in 0..self.top_k {
                let expert = order[slot];
                topk_ids[t * self.top_k + slot] = expert as i32;
                topk_weights[t * self.top_k + slot] = row[expert];
            }
        }
        Ok((topk_weights, topk_ids, next_states))
    }
}

/// `TTSSamplingParams` (generation.py).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TtsSamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub max_tokens: usize,
    pub ignore_eos: bool,
    pub repetition_window: usize,
    pub repetition_penalty: f32,
    pub repetition_codebooks: i32,
}

impl Default for TtsSamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.15,
            top_k: 106,
            top_p: 0.0,
            min_p: 0.18,
            max_tokens: 1024,
            ignore_eos: false,
            repetition_window: 50,
            repetition_penalty: 1.2,
            repetition_codebooks: 8,
        }
    }
}

/// `Zonos2GenerationState`: the generated delayed frames plus the eos
/// countdown.
pub struct GenerationState {
    pub n_codebooks: usize,
    pub eoa_id: i32,
    pub text_vocab: i32,
    pub eos_frame: Option<usize>,
    eos_countdown: usize,
    pub generated: Vec<Vec<i32>>,
}

impl GenerationState {
    pub fn new(n_codebooks: usize, eoa_id: i32, text_vocab: i32) -> Self {
        Self {
            n_codebooks,
            eoa_id,
            text_vocab,
            eos_frame: None,
            eos_countdown: 0,
            generated: Vec::new(),
        }
    }

    /// `finished`.
    pub fn finished(&self) -> bool {
        self.eos_frame.is_some() && self.eos_countdown == 0
    }

    /// `append`.
    pub fn append(&mut self, frame: &[i32], ignore_eos: bool) {
        self.generated
            .push(frame[..self.n_codebooks.min(frame.len())].to_vec());
        if ignore_eos {
            return;
        }
        if self.eos_frame.is_none() {
            let mut max_eos_cb: Option<usize> = None;
            for (i, &value) in frame.iter().enumerate().take(self.n_codebooks) {
                if value == self.eoa_id {
                    max_eos_cb = Some(i);
                }
            }
            if let Some(max_eos_cb) = max_eos_cb {
                let step = self.generated.len() - 1;
                self.eos_frame = Some(step.saturating_sub(max_eos_cb));
                self.eos_countdown = self.n_codebooks + 1;
            }
        }
        if self.eos_frame.is_some() && self.eos_countdown > 0 {
            self.eos_countdown -= 1;
        }
    }
}

/// `sample_frame` for one sequence: per-codebook repetition penalty,
/// temperature scaling, top-k on the scaled logits, softmax, top-p and
/// min-p on the probabilities with renormalization, and the
/// valid-check greedy fallback. The draw uses the crate-local threefry
/// RNG (one key per step, rows drawn in codebook order; the reference
/// draws through `mx.random` with one key for the whole frame).
pub fn sample_frame(
    logits: &[f32],
    n_codebooks: usize,
    codebook_vocab: usize,
    state: &GenerationState,
    params: &TtsSamplingParams,
    keys: &mut rng::KeySequence,
) -> Vec<i32> {
    let codebooks = n_codebooks;
    let vocab = codebook_vocab;
    assert_eq!(logits.len(), codebooks * vocab);
    let mut penalized = logits.to_vec();
    // _apply_repetition_penalty.
    if params.repetition_penalty > 1.0
        && params.repetition_window > 0
        && !state.generated.is_empty()
    {
        let limit = if params.repetition_codebooks < 0 {
            codebooks
        } else {
            codebooks.min(params.repetition_codebooks as usize)
        };
        let start = state
            .generated
            .len()
            .saturating_sub(params.repetition_window);
        for cb in 0..limit {
            let mut seen: Vec<usize> = Vec::new();
            for row in &state.generated[start..] {
                if cb < row.len() {
                    let value = row[cb];
                    if let Ok(index) = usize::try_from(value) {
                        if index < vocab && !seen.contains(&index) {
                            seen.push(index);
                        }
                    }
                }
            }
            for &index in &seen {
                let slot = cb * vocab + index;
                let value = penalized[slot];
                penalized[slot] = if value > 0.0 {
                    value / params.repetition_penalty
                } else {
                    value * params.repetition_penalty
                };
            }
        }
    }
    let mut ids = vec![0i32; codebooks];
    if params.temperature <= 1e-8 {
        for cb in 0..codebooks {
            ids[cb] = sampling::argmax(&penalized[cb * vocab..(cb + 1) * vocab]) as i32;
        }
    } else {
        let scale = 1.0 / params.temperature;
        let mut filtered = vec![0.0f32; penalized.len()];
        for (dst, src) in filtered.iter_mut().zip(&penalized) {
            *dst = src * scale;
        }
        // _apply_top_k on the scaled logits: keep lanes >= the kth
        // value (ties survive, exactly like the reference `where(<)`).
        if params.top_k > 0 && params.top_k < vocab {
            let mut kth = vec![0.0f32; codebooks];
            for cb in 0..codebooks {
                let row = &filtered[cb * vocab..(cb + 1) * vocab];
                let mut sorted = row.to_vec();
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                kth[cb] = sorted[params.top_k - 1];
            }
            for cb in 0..codebooks {
                for lane in &mut filtered[cb * vocab..(cb + 1) * vocab] {
                    if *lane < kth[cb] {
                        *lane = f32::NEG_INFINITY;
                    }
                }
            }
        }
        // Softmax to probabilities.
        let mut probs = vec![0.0f32; filtered.len()];
        for cb in 0..codebooks {
            let row = &filtered[cb * vocab..(cb + 1) * vocab];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            let prob = &mut probs[cb * vocab..(cb + 1) * vocab];
            for (p, &l) in prob.iter_mut().zip(row) {
                *p = (l - max).exp();
                sum += *p;
            }
            for p in prob.iter_mut() {
                *p /= sum;
            }
        }
        // _apply_top_p on the probabilities.
        if params.top_p > 0.0 && params.top_p < 1.0 {
            apply_top_p_probs(&mut probs, codebooks, vocab, params.top_p);
        }
        // _apply_min_p on the probabilities.
        if params.min_p > 0.0 {
            for cb in 0..codebooks {
                let row = &mut probs[cb * vocab..(cb + 1) * vocab];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let threshold = max * params.min_p;
                for p in row.iter_mut() {
                    if *p < threshold {
                        *p = 0.0;
                    }
                }
                normalize(row);
            }
        }
        // The valid check: finite lanes and a positive sum, else the
        // greedy argmax of the filtered logits.
        for cb in 0..codebooks {
            let prob = &probs[cb * vocab..(cb + 1) * vocab];
            let all_finite = prob.iter().all(|p| p.is_finite());
            let positive = prob.iter().sum::<f32>() > 0.0;
            let valid = all_finite && positive;
            if valid {
                let sample_logits: Vec<f32> = prob.iter().map(|&p| p.max(1e-20).ln()).collect();
                ids[cb] = rng::categorical(&sample_logits, keys.next()) as i32;
            } else {
                ids[cb] = sampling::argmax(&filtered[cb * vocab..(cb + 1) * vocab]) as i32;
            }
        }
    }
    ids.push(state.text_vocab);
    ids
}

fn normalize(row: &mut [f32]) {
    let sum: f32 = row.iter().sum();
    let denom = if sum > 1e-12 { sum } else { 1e-12 };
    for p in row.iter_mut() {
        *p /= denom;
    }
}

/// `_apply_top_p` over probability rows: zero the tail past the
/// cumulative `top_p` mass (the current entry excluded) and
/// renormalize.
fn apply_top_p_probs(probs: &mut [f32], codebooks: usize, vocab: usize, top_p: f32) {
    for cb in 0..codebooks {
        let row = &mut probs[cb * vocab..(cb + 1) * vocab];
        let mut order: Vec<usize> = (0..vocab).collect();
        order.sort_by(|&a, &b| {
            row[b]
                .partial_cmp(&row[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let mut cumulative = 0.0f32;
        let mut keep = vec![false; vocab];
        for &index in &order {
            if cumulative - row[index] > top_p {
                continue;
            }
            keep[index] = true;
            cumulative += row[index];
        }
        for (index, p) in row.iter_mut().enumerate() {
            if !keep[index] {
                *p = 0.0;
            }
        }
        normalize(row);
    }
}

/// The loaded family model (`Model`).
pub struct Model {
    pub config: ModelConfig,
    /// `n_codebooks` audio embedders plus the text embedder.
    embedders: Vec<Vec<f32>>,
    emb_norm: FusedRmsNorm,
    speaker_lda_projection: Option<Linear>,
    speaker_projection: Option<Linear>,
    layers: Vec<Block>,
    out_norm: FusedRmsNorm,
    multi_output: Linear,
}

impl Model {
    /// Opens a checkpoint directory (`config.json` + safetensors).
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        let weights = lm::Weights::open(dir)?;
        Self::load(&weights, &config_value)
    }

    /// Loads from opened weights and the raw config dict.
    pub fn load(weights: &lm::Weights, config: &Value) -> Result<Self> {
        let config = ModelConfig::from_json(config)?;
        Self::load_config(weights, config)
    }

    /// Loads from a resolved config.
    pub fn load_config(weights: &lm::Weights, config: ModelConfig) -> Result<Self> {
        let dim = config.dim;
        let audio_vocab = config.audio_vocab_size();
        let mut embedders = Vec::with_capacity(config.frame_width());
        for i in 0..config.n_codebooks {
            embedders.push(weights.embedding(
                &format!("multi_embedder.embedders.{i}"),
                audio_vocab,
                dim,
            )?);
        }
        embedders.push(weights.embedding(
            &format!("multi_embedder.embedders.{}", config.n_codebooks),
            config.text_vocab + 1,
            dim,
        )?);
        let (lda_projection, speaker_projection) = if config.speaker_enabled {
            match config.speaker_lda_dim {
                Some(lda) => (
                    Some(weights.linear_absolute(
                        "speaker_lda_projection",
                        config.speaker_embedding_dim,
                        lda,
                        lm::Bias::Required,
                    )?),
                    Some(weights.linear_absolute(
                        "speaker_projection",
                        lda,
                        dim,
                        lm::Bias::Required,
                    )?),
                ),
                None => (
                    None,
                    Some(weights.linear_absolute(
                        "speaker_projection",
                        config.speaker_embedding_dim,
                        dim,
                        lm::Bias::Required,
                    )?),
                ),
            }
        } else {
            (None, None)
        };
        let rope =
            lm::rope::Rope::initialize(config.head_dim, config.rope_theta, true, None, None)?;
        let kv_width = config.n_kv_heads * config.head_dim;
        let q_width = config.n_heads * config.head_dim;
        let mut layers = Vec::with_capacity(config.n_layers);
        for index in 0..config.n_layers {
            let prefix = format!("layers.{index}");
            // The fused wkv stores [2, kv_width, dim] chunked; the flat
            // order is [k; v] over the output axis, split into the two
            // projections.
            let wq = weights
                .tensor_absolute(&format!("{prefix}.attention.wq.weight"), &[q_width, dim])?;
            // The ChunkedLinear storage is [divisor, out / divisor, in];
            // the flat order is already the [out, in] linear.
            let kv_raw = weights.tensor_absolute(
                &format!("{prefix}.attention.wkv.weight"),
                &[2, kv_width, dim],
            )?;
            // The learned per-head temp scales q AFTER the q norm (the
            // reference `rms_norm(q) * |temp|`); it must not fold into wq,
            // which the scale-invariant norm would erase.
            let temp = weights
                .tensor_absolute(&format!("{prefix}.attention.temp"), &[1, config.n_heads, 1])?;
            let temp: Vec<f32> = (0..config.n_heads).map(|h| temp[h].abs()).collect();
            let mut k_proj = vec![0.0f32; kv_width * dim];
            k_proj.copy_from_slice(&kv_raw[..kv_width * dim]);
            let mut v_proj = vec![0.0f32; kv_width * dim];
            v_proj.copy_from_slice(&kv_raw[kv_width * dim..]);
            let attention = Zonos2Attention {
                wq: Linear::new(wq, None, dim, q_width),
                k_proj: Linear::new(k_proj, None, dim, kv_width),
                v_proj: Linear::new(v_proj, None, dim, kv_width),
                wo: weights.linear_absolute(
                    &format!("{prefix}.attention.wo"),
                    q_width,
                    dim,
                    lm::Bias::None,
                )?,
                gater: weights.linear_absolute(
                    &format!("{prefix}.attention.gater"),
                    dim,
                    config.n_heads,
                    lm::Bias::None,
                )?,
                temp,
                q_norm: RmsNorm::new(vec![1.0; config.head_dim], 1e-6),
                k_norm: RmsNorm::new(vec![1.0; config.head_dim], 1e-6),
                n_heads: config.n_heads,
                n_kv_heads: config.n_kv_heads,
                head_dim: config.head_dim,
                scale: (config.head_dim as f64).powf(-0.5) as f32,
                rope: rope.clone(),
            };
            let norm_weight = |name: &str| -> Result<FusedRmsNorm> {
                Ok(FusedRmsNorm::new(
                    Some(weights.tensor_absolute(name, &[dim])?),
                    config.norm_eps,
                    dim,
                ))
            };
            let attention_norm = norm_weight(&format!("{prefix}.attention_norm.weight"))?;
            let ffn_norm = norm_weight(&format!("{prefix}.ffn_norm.weight"))?;
            let feed_forward = if config.is_moe_layer(index) {
                let router_prefix = format!("{prefix}.feed_forward.router");
                let use_eda = index != config.moe_start_from_layer;
                let router = Router {
                    down_proj: weights.linear_absolute(
                        &format!("{router_prefix}.down_proj"),
                        dim,
                        config.moe_router_dim,
                        lm::Bias::Required,
                    )?,
                    l0: weights.linear_absolute(
                        &format!("{router_prefix}.router_mlp.l0"),
                        config.moe_router_dim,
                        config.moe_router_dim,
                        lm::Bias::Required,
                    )?,
                    l2: weights.linear_absolute(
                        &format!("{router_prefix}.router_mlp.l2"),
                        config.moe_router_dim,
                        config.moe_router_dim,
                        lm::Bias::Required,
                    )?,
                    l4: weights.linear_absolute(
                        &format!("{router_prefix}.router_mlp.l4"),
                        config.moe_router_dim,
                        config.moe_n_experts,
                        lm::Bias::None,
                    )?,
                    rmsnorm_eda: RmsNorm::new(
                        weights.tensor_absolute(
                            &format!("{router_prefix}.rmsnorm_eda.weight"),
                            &[config.moe_router_dim],
                        )?,
                        config.norm_eps,
                    ),
                    balancing_biases: weights.tensor_absolute(
                        &format!("{router_prefix}.balancing_biases"),
                        &[config.moe_n_experts],
                    )?,
                    router_states_scale: if use_eda {
                        Some(weights.tensor_absolute(
                            &format!("{router_prefix}.router_states_scale"),
                            &[config.moe_router_dim],
                        )?)
                    } else {
                        None
                    },
                    top_k: config.num_experts_per_tok(index),
                    use_legacy_balancing: config.moe_balancing_strategy != "quantile",
                    num_experts: config.moe_n_experts,
                };
                let load_expert = |name: &str, out: usize, input: usize| -> Result<Vec<f32>> {
                    weights.tensor_absolute(
                        &format!("{prefix}.feed_forward.experts.{name}"),
                        &[config.moe_n_experts, out, input],
                    )
                };
                let experts = SwitchGlu::new(
                    load_expert("gate_proj.weight", config.intermediate_size, dim)?,
                    load_expert("up_proj.weight", config.intermediate_size, dim)?,
                    load_expert("down_proj.weight", dim, config.intermediate_size)?,
                    None,
                    None,
                    None,
                    config.moe_n_experts,
                    dim,
                    config.intermediate_size,
                );
                FeedForward::MoE {
                    router,
                    experts,
                    norm_topk_prob: config.norm_topk_prob,
                }
            } else {
                let w_in = Linear::new(
                    weights.tensor_absolute(
                        &format!("{prefix}.feed_forward.w_in.weight"),
                        &[2, config.intermediate_size, dim],
                    )?,
                    None,
                    dim,
                    2 * config.intermediate_size,
                );
                FeedForward::Dense {
                    w_in,
                    w_out: weights.linear_absolute(
                        &format!("{prefix}.feed_forward.w_out"),
                        config.intermediate_size,
                        dim,
                        lm::Bias::None,
                    )?,
                    intermediate: config.intermediate_size,
                }
            };
            layers.push(Block {
                attention,
                attention_norm,
                ffn_norm,
                feed_forward,
            });
        }
        Ok(Self {
            emb_norm: FusedRmsNorm::new(None, config.norm_eps, dim),
            out_norm: FusedRmsNorm::new(
                Some(weights.tensor_absolute("out_norm.weight", &[dim])?),
                config.norm_eps,
                dim,
            ),
            multi_output: Linear::new(
                weights.tensor_absolute(
                    "multi_output.weight",
                    &[config.n_codebooks * config.audio_vocab_size(), dim],
                )?,
                None,
                dim,
                config.n_codebooks * config.audio_vocab_size(),
            ),
            embedders,
            speaker_lda_projection: lda_projection,
            speaker_projection,
            layers,
            config,
        })
    }

    /// `_forward_hidden` over `[rows, frame_width]` row-major frames.
    fn forward_hidden(
        &self,
        frames: &[i32],
        mut caches: Option<&mut lm::LayerCaches>,
        speaker_embedding: Option<&[f32]>,
        speaker_token_position: Option<usize>,
    ) -> Result<Vec<f32>> {
        let config = &self.config;
        let rows = frames.len() / config.frame_width();
        if frames.len() % config.frame_width() != 0 {
            return Err(SpeechError::Input {
                why: format!(
                    "frames must be a multiple of the {}-wide layout",
                    config.frame_width()
                ),
            });
        }
        let dim = config.dim;
        let mut h = vec![0.0f32; rows * dim];
        for row in 0..rows {
            for (column, embedder) in self.embedders.iter().enumerate() {
                let id =
                    usize::try_from(frames[row * config.frame_width() + column]).map_err(|_| {
                        SpeechError::Input {
                            why: "negative frame id".to_string(),
                        }
                    })?;
                let source = &embedder[id * dim..(id + 1) * dim];
                for (d, value) in source.iter().enumerate() {
                    h[row * dim + d] += *value;
                }
            }
        }
        // _inject_speaker at an integer position.
        if let (Some(embedding), Some(pos)) = (speaker_embedding, speaker_token_position) {
            if pos < rows {
                let projected = self.project_speaker(embedding)?;
                h[pos * dim..(pos + 1) * dim].copy_from_slice(&projected);
            }
        }
        let (normed, _) = self.emb_norm.forward(&h, rows, None);
        h = normed;
        let offset = caches
            .as_ref()
            .and_then(|c| c.first())
            .map(|c| c.offset)
            .unwrap_or(0);
        let mut residual: Option<Vec<f32>> = None;
        let mut router_states: Option<Vec<f32>> = None;
        for (index, layer) in self.layers.iter().enumerate() {
            // The first block folds no residual (the reference
            // initializes it None before the layer loop).
            let (normed, res) = layer.attention_norm.forward(&h, rows, residual.as_deref());
            residual = Some(res);
            let cache: Option<&mut dyn lm::cache::LayerKv> = caches
                .as_deref_mut()
                .map(|c| &mut c[index] as &mut dyn lm::cache::LayerKv);
            h = layer.attention.forward(&normed, rows, offset, cache)?;
            let (normed, res) = layer.ffn_norm.forward(&h, rows, residual.as_deref());
            residual = Some(res);
            let (fed, next_states) =
                layer
                    .feed_forward
                    .forward(&normed, rows, router_states.as_deref())?;
            router_states = next_states;
            h = fed;
        }
        let (normed, _) = self.out_norm.forward(&h, rows, residual.as_deref());
        Ok(normed)
    }

    /// `compute_logits` over the hidden states: the head, the
    /// codebook/vocab reshape, and the softcap.
    pub fn compute_logits(&self, hidden: &[f32], rows: usize) -> Vec<f32> {
        let config = &self.config;
        let raw = self.multi_output.forward(hidden, rows);
        let mut logits = raw;
        if config.loss_softcap > 0.0 {
            let cap = config.loss_softcap;
            for value in &mut logits {
                *value = cap * (*value / cap).tanh();
            }
        }
        logits
    }

    /// The speaker path: the optional LDA projection then the
    /// dimension projection.
    fn project_speaker(&self, embedding: &[f32]) -> Result<Vec<f32>> {
        let mut emb = embedding.to_vec();
        if let Some(lda) = &self.speaker_lda_projection {
            emb = lda.forward(&emb, 1);
        }
        match &self.speaker_projection {
            Some(projection) => Ok(projection.forward(&emb, 1)),
            None => Ok(emb),
        }
    }

    /// `_build_prompt_rows`: the optional speaker prefix, the
    /// conditioning markers, the byte-level text rows, and the silence
    /// tail.
    pub fn build_prompt_rows(
        &self,
        text: &str,
        speaking_rate_bucket: Option<usize>,
        quality_buckets: &[Option<usize>],
        speaker_conditioned: bool,
        clean_speaker_background: bool,
        accurate_mode: bool,
    ) -> Result<(Vec<Vec<i32>>, Option<usize>)> {
        let config = &self.config;
        let marker = |token: i32| -> Vec<i32> {
            let mut row = vec![config.audio_pad_id; config.n_codebooks];
            row.push(token);
            row
        };
        let mut rows: Vec<Vec<i32>> = Vec::new();
        if let Some(bucket) = speaking_rate_bucket {
            rows.push(marker(speaking_rate_token_id(
                config.text_vocab,
                config.speaking_rate_num_buckets,
                bucket,
                &config.quality_bucket_counts,
                config.speaker_background_num_buckets,
                config.accurate_mode_num_buckets,
            )?));
        }
        for (feature_idx, bucket) in quality_buckets.iter().enumerate() {
            if let Some(bucket) = bucket {
                rows.push(marker(quality_token_id(
                    config.text_vocab,
                    config.speaking_rate_num_buckets,
                    &config.quality_bucket_counts,
                    feature_idx,
                    *bucket,
                    config.speaker_background_num_buckets,
                    config.accurate_mode_num_buckets,
                )?));
            }
        }
        for token in text_to_byte_ids(text) {
            rows.push(marker(token));
        }
        for row in silence_prompt_tokens(
            config.n_codebooks,
            config.audio_pad_id,
            config.text_vocab as i32,
        ) {
            rows.push(row);
        }
        let mut speaker_token_position = None;
        if speaker_conditioned {
            // make_speaker_slot: the pad columns carry the text vocab
            // in the text column.
            let mut prefix = vec![marker(config.text_vocab as i32)];
            if config.speaker_background_num_buckets > 0 {
                prefix.push(marker(speaker_background_token_id(
                    config.text_vocab,
                    config.speaking_rate_num_buckets,
                    &config.quality_bucket_counts,
                    clean_speaker_background,
                    config.speaker_background_num_buckets,
                    config.accurate_mode_num_buckets,
                )?));
                if accurate_mode && config.accurate_mode_num_buckets > 0 {
                    prefix.push(marker(accurate_mode_token_id(
                        config.text_vocab,
                        config.speaking_rate_num_buckets,
                        &config.quality_bucket_counts,
                        config.speaker_background_num_buckets,
                        config.accurate_mode_num_buckets,
                    )?));
                }
            }
            rows.splice(..0, prefix);
            speaker_token_position = Some(0);
        }
        Ok((rows, speaker_token_position))
    }

    /// Runs one segment end to end (`generate`, single non-streaming
    /// segment): prompt prefill, the per-frame sample loop, and the
    /// returned delayed frames with the eos frame. The waveform decode
    /// is [`Model::decode_audio`].
    #[allow(
        clippy::too_many_arguments,
        reason = "the reference generate signature carries the conditioning knobs explicitly"
    )]
    pub fn generate(
        &self,
        text: &str,
        params: &TtsSamplingParams,
        speaker_embedding: Option<&[f32]>,
        clean_speaker_background: bool,
        accurate_mode: bool,
        speaking_rate_bucket: Option<usize>,
        quality_buckets: &[Option<usize>],
        seed: u64,
    ) -> Result<(Vec<Vec<i32>>, Option<usize>)> {
        let config = &self.config;
        let (rows, speaker_pos) = self.build_prompt_rows(
            text,
            speaking_rate_bucket,
            quality_buckets,
            speaker_embedding.is_some(),
            clean_speaker_background,
            accurate_mode,
        )?;
        let width = config.frame_width();
        let mut flat = Vec::with_capacity(rows.len() * width);
        for row in &rows {
            flat.extend_from_slice(row);
        }
        let mut caches = lm::make_prompt_cache(config.n_layers);
        let hidden =
            self.forward_hidden(&flat, Some(&mut caches), speaker_embedding, speaker_pos)?;
        let mut last_logits = self.compute_logits(&hidden[hidden.len() - config.dim..], 1);
        let mut state =
            GenerationState::new(config.n_codebooks, config.eoa_id, config.text_vocab as i32);
        for step in 0..params.max_tokens {
            let mut keys = rng::KeySequence::new(step_seed(seed, step));
            let frame = sample_frame(
                &last_logits,
                config.n_codebooks,
                config.audio_vocab_size(),
                &state,
                params,
                &mut keys,
            );
            state.append(&frame, params.ignore_eos);
            if state.finished() {
                break;
            }
            let hidden = self.forward_hidden(&frame, Some(&mut caches), None, None)?;
            last_logits = self.compute_logits(&hidden[hidden.len() - config.dim..], 1);
        }
        Ok((state.generated, state.eos_frame))
    }

    /// `_decode_audio`: shear up the delayed rows, trim to the eos
    /// frame, clip, and decode through the DAC. The trim length uses
    /// the DAC hop length (512 at 44 kHz, the reference constant).
    pub fn decode_audio(
        &self,
        dac: &Dac,
        delayed_rows: &[Vec<i32>],
        eos_frame: Option<usize>,
    ) -> Result<Vec<f32>> {
        if delayed_rows.is_empty() {
            return Ok(Vec::new());
        }
        let config = &self.config;
        let mut codes = shear_up(delayed_rows, config.audio_pad_id)?;
        if let Some(eos_frame) = eos_frame {
            codes.truncate(eos_frame);
        }
        if codes.is_empty() || codes[0].is_empty() {
            return Ok(Vec::new());
        }
        let frames = codes.len();
        for row in &mut codes {
            for value in row.iter_mut() {
                *value = (*value).clamp(0, config.codebook_size as i32 - 1);
            }
        }
        let columns = codes[0].len();
        let level_rows: Vec<Vec<i32>> = (0..columns)
            .map(|c| codes.iter().map(|row| row[c]).collect())
            .collect();
        let mut audio = dac.decode_codes(&level_rows)?;
        audio.truncate(frames * dac.config.hop_length());
        Ok(audio)
    }
}

#[cfg(test)]
mod tests;
