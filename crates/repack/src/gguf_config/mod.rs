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

mod attention;
mod masks;
mod meta;

use model_io::{ArchConfig, ModelFamily};

use self::attention::{kv_heads, linear_attention};
use self::masks::{gemma4_layer_mask, gpt_oss_layer_mask, qwen_gdn_moe_layer_mask};
use self::meta::Meta;
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
    // `block_count` COUNTS THE MULTI-TOKEN-PREDICTION BLOCK, and this port's
    // `num_layers` is the trunk alone. llama.cpp writes the head as one more
    // `blk.<n>.` block and declares how many of the trailing ones are hers in
    // `nextn_predict_layers`; `Ornith-1.5-35B-A3B` reads 41 and 1 for a
    // 40-layer model. Taking `block_count` verbatim there derives a 41-layer
    // config whose mask marks index 40 LINEAR -- the head is full attention --
    // which validates structurally and runs the wrong block.
    //
    // Absent means 0 (AGENTS.md Gotcha 39: the default belongs to the FORMAT).
    // Every GGUF this port installed before Ornith omits the key, so their
    // derivations are unchanged.
    let mtp_blocks = m.opt_i64("nextn_predict_layers")?.unwrap_or(0);
    let block_count = m.i64("block_count")?;
    // Guarded only when the key is PRESENT and positive. A file that declares
    // no head is left exactly as it was before this subtraction existed,
    // including the degenerate zero-block fixtures whose block count this
    // function has never had an opinion about.
    if mtp_blocks < 0 || (mtp_blocks > 0 && mtp_blocks >= block_count) {
        return Err(GgufConfigError::BadValue {
            key: m.key("nextn_predict_layers"),
            detail: format!("{mtp_blocks} of {block_count} blocks leaves no trunk"),
        });
    }
    let num_layers = block_count - mtp_blocks;
    arch.num_layers = num_layers;
    arch.hidden_size = m.i64("embedding_length")?;
    arch.num_heads = m.i64("attention.head_count")?;
    // A DENSE checkpoint publishes none of the three MoE keys, and the
    // `llama` architecture covers both halves (Mixtral has them, Llama 3.1
    // does not), so they are optional and default to the dense answer.
    arch.num_experts = m.opt_i64("expert_count")?.unwrap_or(0);
    arch.top_k_experts = m.opt_i64("expert_used_count")?.unwrap_or(0);
    // Gemma and Qwen publish a separate expert width; the `llama`
    // architecture does not, and its experts are `feed_forward_length` wide.
    // Measured absent on both Mixtral conversions, and asserted so in
    // `gguf_checkpoint_network.rs::scopes_phase_m2_from_the_mixtral_header`.
    arch.moe_intermediate_size = match m.opt_i64("expert_feed_forward_length")? {
        Some(width) => width,
        None if arch.num_experts > 0 => m.i64("feed_forward_length")?,
        None => 0,
    };
    arch.vocab_size = vocab_size(header)?;
    // Untied only when the checkpoint actually ships a separate head.
    arch.tie_word_embeddings = !header.tensors.contains_key("output.weight");

    arch.full_attention_layer_mask = match family {
        ModelFamily::Gemma4 => gemma4_layer_mask(&m, num_layers as usize)?,
        // ONE builder for both halves, because both publish the same
        // `full_attention_interval` key and the same every-fourth-layer rule.
        // The dense half was refused here until `ornith-ai/Ornith-1.5-9B-GGUF`
        // became the first published `qwen35` file; before it the mask would
        // have been invented, which is why the refusal was right at the time.
        ModelFamily::QwenGdnMoe | ModelFamily::QwenGdnDense => {
            qwen_gdn_moe_layer_mask(&m, num_layers as usize)?
        }
        // Every layer is full attention: neither a dense Llama nor a Mixtral
        // publishes `attention.sliding_window`, and Mistral 7B's window is a
        // property of that model rather than of the architecture.
        // Qwen3-MoE is the same story: no `attention.sliding_window` key on
        // the published file, every layer full attention.
        ModelFamily::Llama | ModelFamily::Qwen3Moe => vec![1u8; num_layers as usize],
        ModelFamily::GptOss => gpt_oss_layer_mask(&m, num_layers as usize)?,
        // Refused rather than defaulted: no GGUF exists for either, so any
        // mask here would be invented. `muse_glimmer`'s is doubly so -- its
        // `[0,0,0,1]` window comes from a `layer_types` ARRAY that no GGUF
        // metadata key expresses.
        // `qwen4_exp` is refused for a DIFFERENT reason than these two and it
        // matters: GGUFs of it DO exist, so this is not "no file to read".
        // This port ingests its MLX safetensors, and nothing here has parsed
        // what a GGUF converter names its n-gram shards, its hyper-connection
        // tensors or its indexer -- so a mask derived here would be the one
        // part of an install that looked right while the rest went missing.
        // Refuse until the GGUF walk has an arm that reads the whole file.
        ModelFamily::DeepseekV4Flash | ModelFamily::MuseGlimmer | ModelFamily::Qwen4Exp => {
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
        .opt_i64("expert_shared_feed_forward_length")?
        .or(m.opt_i64("feed_forward_length")?)
        .unwrap_or(arch.intermediate_size);

    // A model with no sliding-window layers publishes no window.
    if let Some(window) = m.opt_i64("attention.sliding_window")? {
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
            original_context: m.opt_i64("rope.scaling.original_context_length")?.ok_or(
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

    // BOTH Qwen halves, because both run the gated-DeltaNet block and both
    // publish the same `ssm.*` keys.
    //
    // **THE DENSE HALF WAS MISSING HERE AND THE BUG WAS INVISIBLE ON THE MoE
    // ONE**, which is AGENTS.md Gotcha 37's shape: a fallback is correct for
    // exactly as long as one checkpoint exercises it. `qwen_gdn_moe_35b_a3b()`
    // declares `num_v_heads: 32` and the real file says 32, so the MoE half
    // read the right answer from the BASELINE whether or not this line ran.
    // `qwen_gdn_dense_27b()` declares 48 (Bonsai-27B's), so the first dense
    // GGUF derived `qkv_dim` = 2*2048 + 48*128 = 10240 against a real 8192 and
    // failed at the first linear layer's GEMV -- loudly, but four layers from
    // the cause, and only after a 5-minute stream. `ornith_gguf_network.rs`
    // asserts the linear block field by field now, which is a header read.
    if matches!(family, ModelFamily::QwenGdnMoe | ModelFamily::QwenGdnDense) {
        arch.linear_attention = linear_attention(&m)?;
        // Qwen's key/value length are per-head and equal on both paths.
        if let Some(k) = m.opt_i64("attention.key_length")? {
            arch.head_dim = k;
            arch.full_head_dim = k;
        }
    } else {
        match m.opt_i64("attention.key_length")? {
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
        if let Some(k) = m.opt_i64("attention.key_length_swa")? {
            arch.head_dim = k;
        }
    }

    Ok(arch)
}
