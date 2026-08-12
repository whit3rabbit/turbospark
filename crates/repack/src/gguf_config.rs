//! Rebuilds an [`ArchConfig`] from a GGUF file's metadata, the way
//! [`crate::peek_manifest_arch`] rebuilds one from an installed
//! `manifest.json`. Sibling of [`crate::qwen36_config`], which does the same
//! job for an HF `config.json`.
//!
//! **Start from the family baseline, override only what GGUF actually
//! determines.** `ArchConfig` carries two kinds of field. Shape fields
//! (layer counts, widths, expert counts) describe this checkpoint and are
//! read from the file. Behavioral fields (`attn_output_gate`,
//! `ffn_sandwich_norms`, `embedding_scaled_by_sqrt_hidden`,
//! `attention_scale`, `hidden_activation`, ...) describe the ARCHITECTURE
//! and are simply absent from GGUF metadata, because llama.cpp hardcodes
//! them in its graph builder instead. Inventing values for those would be
//! guessing; taking them from `known_architecture(family)` is the same
//! fallback `arch_validation` already applies to omitted manifest fields
//! (AGENTS.md Gotcha 24), so a GGUF install validates against exactly the
//! same numbers an MLX install does.
//!
//! `partial_rotary_factor` is a worked example of why the rule matters.
//! GGUF publishes `rope.dimension_count`, which for Gemma 4 equals the head
//! dim exactly, and reading a rotary fraction off it would yield 1.0 where
//! the true value is 0.25. The key does not mean what its name suggests, so
//! this module does not touch that field.

use model_io::{ArchConfig, LinearAttentionConfig, ModelFamily};

use crate::gguf_header::{GgufHeader, GgufValue};
use crate::gguf_names::family_for_architecture;

#[derive(Debug, Clone, PartialEq)]
pub enum GgufConfigError {
    MissingArchitecture,
    UnsupportedArchitecture { architecture: String },
    MissingKey { key: String },
    BadValue { key: String, detail: String },
    MissingTensor { name: String },
}

impl std::fmt::Display for GgufConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GgufConfigError::MissingArchitecture => {
                write!(f, "GGUF metadata has no general.architecture")
            }
            // The registry, not a bare string: a recognized-but-unported
            // architecture gets told what it would need and where the
            // checklist is (ROADMAP Phase M Stage 1).
            GgufConfigError::UnsupportedArchitecture { architecture } => {
                write!(
                    f,
                    "{}",
                    crate::arch_registry::describe_gguf_architecture(architecture)
                )
            }
            GgufConfigError::MissingKey { key } => write!(f, "GGUF metadata has no {key}"),
            GgufConfigError::BadValue { key, detail } => write!(f, "GGUF {key}: {detail}"),
            GgufConfigError::MissingTensor { name } => {
                write!(f, "GGUF file has no tensor {name}")
            }
        }
    }
}

impl std::error::Error for GgufConfigError {}

struct Meta<'a> {
    header: &'a GgufHeader,
    prefix: &'a str,
}

impl Meta<'_> {
    fn key(&self, suffix: &str) -> String {
        format!("{}.{suffix}", self.prefix)
    }

    fn opt(&self, suffix: &str) -> Option<&GgufValue> {
        self.header.metadata.get(&self.key(suffix))
    }

    fn u64(&self, suffix: &str) -> Result<u64, GgufConfigError> {
        let key = self.key(suffix);
        self.opt(suffix)
            .ok_or(GgufConfigError::MissingKey { key: key.clone() })?
            .as_u64()
            .ok_or(GgufConfigError::BadValue {
                key,
                detail: "not an unsigned integer".to_string(),
            })
    }

    fn i64(&self, suffix: &str) -> Result<i64, GgufConfigError> {
        Ok(self.u64(suffix)? as i64)
    }

    fn opt_i64(&self, suffix: &str) -> Option<i64> {
        self.opt(suffix)
            .and_then(GgufValue::as_u64)
            .map(|v| v as i64)
    }

    fn opt_f64(&self, suffix: &str) -> Option<f64> {
        self.opt(suffix).and_then(GgufValue::as_f64)
    }
}

/// `gpt-oss`'s alternating window (ROADMAP M5), derived rather than read.
///
/// This is the third spelling of "which layers slide" in this file and the
/// only one the file does not state directly. Gemma ships a per-layer BOOL
/// ARRAY; Qwen 3.6 a period; `gpt-oss` ships `attention.sliding_window` and
/// NOTHING ELSE, and llama.cpp's loader supplies the rest: an absent
/// `attention.sliding_window_pattern` means period 2, and
/// `llama_hparams::set_swa_pattern(2, dense_first = false)` computes
/// `is_swa[il] = (il % 2) < 1`. So EVEN layers slide.
///
/// AGENTS.md Gotcha 39's rule applies twice here. The default belongs to the
/// FORMAT (llama.cpp's 2), not to a neighbouring family, and a file that DOES
/// publish a period must be believed over it -- which is why the key is read
/// rather than assumed even though the shipped checkpoints omit it. Getting
/// the phase inverted yields a model wrong only past 128 tokens of context.
fn gpt_oss_layer_mask(m: &Meta<'_>, num_layers: usize) -> Result<Vec<u8>, GgufConfigError> {
    let window = m.i64("attention.sliding_window")?;
    if window <= 0 {
        return Err(GgufConfigError::BadValue {
            key: m.key("attention.sliding_window"),
            detail: format!("{window} is not a usable sliding window"),
        });
    }
    let period = m.opt_i64("attention.sliding_window_pattern").unwrap_or(2);
    if period < 1 {
        return Err(GgufConfigError::BadValue {
            key: m.key("attention.sliding_window_pattern"),
            detail: format!("period {period} is not usable"),
        });
    }
    Ok((0..num_layers)
        .map(|i| u8::from(i as i64 % period >= period - 1))
        .collect())
}

/// Layer kinds, matching `ArchConfig::full_attention_layer_mask`:
/// 0 = sliding-window, 1 = full attention, 2 = gated-DeltaNet linear.
fn gemma4_layer_mask(m: &Meta<'_>, num_layers: usize) -> Result<Vec<u8>, GgufConfigError> {
    let key = m.key("attention.sliding_window_pattern");
    let pattern = m
        .opt("attention.sliding_window_pattern")
        .and_then(GgufValue::as_array)
        .ok_or(GgufConfigError::MissingKey { key: key.clone() })?;
    if pattern.len() != num_layers {
        return Err(GgufConfigError::BadValue {
            key,
            detail: format!("{} entries for {num_layers} layers", pattern.len()),
        });
    }
    pattern
        .iter()
        .map(|v| {
            // true means "this layer slides", i.e. mask 0. The inversion is
            // the whole content of this function and is easy to get exactly
            // backwards, which would silently swap every layer's attention
            // kind and its rope base.
            v.as_bool()
                .map(|sliding| u8::from(!sliding))
                .ok_or_else(|| GgufConfigError::BadValue {
                    key: key.clone(),
                    detail: "entries are not booleans".to_string(),
                })
        })
        .collect()
}

fn qwen36_layer_mask(m: &Meta<'_>, num_layers: usize) -> Result<Vec<u8>, GgufConfigError> {
    let interval = m.u64("full_attention_interval")? as usize;
    if interval == 0 {
        return Err(GgufConfigError::BadValue {
            key: m.key("full_attention_interval"),
            detail: "must be non-zero".to_string(),
        });
    }
    // Every `interval`-th layer counting from one: 3, 7, 11, ... at 4.
    Ok((0..num_layers)
        .map(|i| if (i + 1) % interval == 0 { 1u8 } else { 2u8 })
        .collect())
}

/// Gemma 4 publishes `head_count_kv` as a PER-LAYER ARRAY, because its
/// sliding-window and global layers differ (8 vs 2). Qwen publishes a
/// scalar. Returns `(sliding, full)`.
fn kv_heads(m: &Meta<'_>, mask: &[u8]) -> Result<(i64, i64), GgufConfigError> {
    let key = m.key("attention.head_count_kv");
    let value = m
        .opt("attention.head_count_kv")
        .ok_or(GgufConfigError::MissingKey { key: key.clone() })?;

    if let Some(scalar) = value.as_u64() {
        return Ok((scalar as i64, scalar as i64));
    }
    let per_layer = value.as_array().ok_or_else(|| GgufConfigError::BadValue {
        key: key.clone(),
        detail: "neither an integer nor an array".to_string(),
    })?;
    if per_layer.len() != mask.len() {
        return Err(GgufConfigError::BadValue {
            key,
            detail: format!("{} entries for {} layers", per_layer.len(), mask.len()),
        });
    }
    let at = |want: u8| -> Option<i64> {
        mask.iter()
            .position(|k| *k == want)
            .and_then(|i| per_layer[i].as_u64())
            .map(|v| v as i64)
    };
    // A model with only one layer kind reuses that kind's count for both,
    // which is what the Qwen scalar path does too.
    let sliding = at(0);
    let full = at(1);
    match (sliding, full) {
        (Some(s), Some(f)) => Ok((s, f)),
        (Some(s), None) => Ok((s, s)),
        (None, Some(f)) => Ok((f, f)),
        (None, None) => Err(GgufConfigError::BadValue {
            key,
            detail: "no layer kind matched the per-layer array".to_string(),
        }),
    }
}

/// Qwen's gated-DeltaNet dimensions, which GGUF publishes under the `ssm.`
/// keys it borrows the tensor slots from.
fn linear_attention(m: &Meta<'_>) -> Result<LinearAttentionConfig, GgufConfigError> {
    let inner = m.i64("ssm.inner_size")?;
    let num_v_heads = m.i64("ssm.time_step_rank")?;
    if num_v_heads == 0 {
        return Err(GgufConfigError::BadValue {
            key: m.key("ssm.time_step_rank"),
            detail: "must be non-zero".to_string(),
        });
    }
    Ok(LinearAttentionConfig {
        num_k_heads: m.i64("ssm.group_count")?,
        num_v_heads,
        key_head_dim: m.i64("ssm.state_size")?,
        // `inner_size` is the whole value stream, so the per-head width is
        // implied rather than published.
        value_head_dim: inner / num_v_heads,
        conv_kernel_size: m.i64("ssm.conv_kernel")?,
    })
}

/// Vocabulary comes from the embedding tensor, not from metadata: the
/// tokenizer's token array is the authority for a tokenizer, but the
/// embedding's row count is the authority for the MODEL, and a checkpoint
/// whose embedding is padded past the token list would otherwise produce an
/// `ArchConfig` that disagrees with its own weights.
fn vocab_size(header: &GgufHeader) -> Result<i64, GgufConfigError> {
    let name = "token_embd.weight";
    let info = header
        .tensors
        .get(name)
        .ok_or(GgufConfigError::MissingTensor {
            name: name.to_string(),
        })?;
    // Dims are stored fastest-varying first, so this is [hidden, vocab].
    info.dims
        .get(1)
        .copied()
        .map(|v| v as i64)
        .ok_or(GgufConfigError::BadValue {
            key: name.to_string(),
            detail: format!("expected rank 2, got {:?}", info.dims),
        })
}

/// Rebuild an [`ArchConfig`] from GGUF metadata.
pub fn arch_from_gguf(header: &GgufHeader) -> Result<ArchConfig, GgufConfigError> {
    let architecture = header
        .architecture()
        .ok_or(GgufConfigError::MissingArchitecture)?;
    let family = family_for_architecture(architecture).ok_or_else(|| {
        GgufConfigError::UnsupportedArchitecture {
            architecture: architecture.to_string(),
        }
    })?;
    let m = Meta {
        header,
        prefix: architecture,
    };

    let mut arch = model_io::known_architecture(family);
    let num_layers = m.i64("block_count")?;
    arch.num_layers = num_layers;
    arch.hidden_size = m.i64("embedding_length")?;
    arch.num_heads = m.i64("attention.head_count")?;
    // A DENSE checkpoint publishes none of the three MoE keys, and the
    // `llama` architecture covers both halves (Mixtral has them, Llama 3.1
    // does not), so they are optional and default to the dense answer.
    arch.num_experts = m.opt_i64("expert_count").unwrap_or(0);
    arch.top_k_experts = m.opt_i64("expert_used_count").unwrap_or(0);
    // Gemma and Qwen publish a separate expert width; the `llama`
    // architecture does not, and its experts are `feed_forward_length` wide.
    // Measured absent on both Mixtral conversions, and asserted so in
    // `gguf_checkpoint_network.rs::scopes_phase_m2_from_the_mixtral_header`.
    arch.moe_intermediate_size = match m.opt_i64("expert_feed_forward_length") {
        Some(width) => width,
        None if arch.num_experts > 0 => m.i64("feed_forward_length")?,
        None => 0,
    };
    arch.vocab_size = vocab_size(header)?;
    // Untied only when the checkpoint actually ships a separate head.
    arch.tie_word_embeddings = !header.tensors.contains_key("output.weight");

    arch.full_attention_layer_mask = match family {
        ModelFamily::Gemma4 => gemma4_layer_mask(&m, num_layers as usize)?,
        ModelFamily::Qwen36 => qwen36_layer_mask(&m, num_layers as usize)?,
        // Every layer is full attention: neither a dense Llama nor a Mixtral
        // publishes `attention.sliding_window`, and Mistral 7B's window is a
        // property of that model rather than of the architecture.
        // Qwen3-MoE is the same story: no `attention.sliding_window` key on
        // the published file, every layer full attention.
        ModelFamily::Llama | ModelFamily::Qwen3Moe => vec![1u8; num_layers as usize],
        ModelFamily::GptOss => gpt_oss_layer_mask(&m, num_layers as usize)?,
        ModelFamily::DeepseekV4Flash => {
            return Err(GgufConfigError::UnsupportedArchitecture {
                architecture: architecture.to_string(),
            })
        }
    };
    let (kv_sliding, kv_full) = kv_heads(&m, &arch.full_attention_layer_mask)?;
    arch.num_kv_heads = kv_sliding;
    arch.num_full_kv_heads = kv_full;

    // The shared/dense FFN width. Qwen publishes it under its own key and
    // reuses `feed_forward_length` for something else, so prefer the
    // specific one and fall back.
    arch.intermediate_size = m
        .opt_i64("expert_shared_feed_forward_length")
        .or_else(|| m.opt_i64("feed_forward_length"))
        .unwrap_or(arch.intermediate_size);

    // A model with no sliding-window layers publishes no window.
    if let Some(window) = m.opt_i64("attention.sliding_window") {
        arch.sliding_window = window;
    }
    if let Some(cap) = m.opt_f64("final_logit_softcapping") {
        arch.final_logit_softcap = cap;
    }
    // `freq_base` is the GLOBAL layers' base and `freq_base_swa` the
    // sliding ones'. This port names them the other way round
    // (`full_rope_theta` / `rope_theta`), so a straight-through assignment
    // by similar name would swap them.
    if let Some(theta) = m.opt_f64("rope.freq_base") {
        arch.full_rope_theta = theta;
        // Single-base models publish only this one, and both fields take it.
        if m.opt_f64("rope.freq_base_swa").is_none() {
            arch.rope_theta = theta;
        }
    }
    if let Some(theta) = m.opt_f64("rope.freq_base_swa") {
        arch.rope_theta = theta;
    }

    // YaRN (ROADMAP M5). Read only when the file says the scaling TYPE is
    // yarn: `rope.scaling.factor` alone is ambiguous, since the same key
    // carries linear scaling's factor and applying YaRN's ramp to it would
    // be wrong in a way that is finite and fluent. Absent type means no
    // scaling, which is `RopeScalingConfig::NONE`, which is what the
    // baseline already holds for every other family.
    if m.opt("rope.scaling.type").and_then(GgufValue::as_str) == Some("yarn") {
        let key = m.key("rope.scaling.factor");
        let factor = m
            .opt_f64("rope.scaling.factor")
            .ok_or(GgufConfigError::MissingKey { key: key.clone() })?;
        if factor <= 0.0 {
            return Err(GgufConfigError::BadValue {
                key,
                detail: format!("yarn factor {factor} is not usable"),
            });
        }
        arch.rope_scaling = model_io::RopeScalingConfig {
            factor,
            original_context: m.opt_i64("rope.scaling.original_context_length").ok_or(
                GgufConfigError::MissingKey {
                    key: m.key("rope.scaling.original_context_length"),
                },
            )?,
            // The two betas DO have llama.cpp defaults (32 and 1), so an
            // absent one is not an error. gpt-oss publishes both.
            beta_fast: m.opt_f64("rope.scaling.yarn_beta_fast").unwrap_or(32.0),
            beta_slow: m.opt_f64("rope.scaling.yarn_beta_slow").unwrap_or(1.0),
        };
    }

    if family == ModelFamily::Qwen36 {
        arch.linear_attention = linear_attention(&m)?;
        // Qwen's key/value length are per-head and equal on both paths.
        if let Some(k) = m.opt_i64("attention.key_length") {
            arch.head_dim = k;
            arch.full_head_dim = k;
        }
    } else {
        match m.opt_i64("attention.key_length") {
            Some(k) => {
                arch.full_head_dim = k;
                // A model with one attention kind publishes no `_swa` width,
                // and its two head-dim fields must not be allowed to
                // disagree: the baseline's `head_dim` would otherwise survive
                // a checkpoint that moved `full_head_dim`, and nothing reads
                // them together.
                if m.opt("attention.key_length_swa").is_none() {
                    arch.head_dim = k;
                }
            }
            // `attention.key_length` IS OPTIONAL, AND LEAVING THE BASELINE'S
            // VALUE IN PLACE IS WRONG RATHER THAN CONSERVATIVE (ROADMAP M4).
            // llama.cpp defaults it to `embedding_length / head_count`, so
            // that is what a file omitting it means -- and the 2023-era
            // conversions omit it routinely.
            //
            // Nothing caught this for two families because they agree by
            // coincidence: Mixtral 8x7B, Mistral 7B and Llama 3.1 8B are all
            // 32 heads of 128, which is the baseline's own value. TinyLlama
            // 1.1B is 32 heads of 64, and it fails at the FIRST q_proj with
            // a packed-size mismatch rather than anywhere near the config.
            // Same shape as AGENTS.md Gotcha 37: a per-MODEL property left
            // standing on a per-BASELINE constant, invisible for exactly as
            // long as the two happen to be equal.
            None => {
                let heads = arch.num_heads;
                if heads <= 0 || arch.hidden_size % heads != 0 {
                    return Err(GgufConfigError::MissingKey {
                        key: format!(
                            "{architecture}.attention.key_length (absent, and \
                             embedding_length {} is not divisible by head_count {heads})",
                            arch.hidden_size
                        ),
                    });
                }
                arch.head_dim = arch.hidden_size / heads;
                arch.full_head_dim = arch.head_dim;
            }
        }
        if let Some(k) = m.opt_i64("attention.key_length_swa") {
            arch.head_dim = k;
        }
    }

    Ok(arch)
}
