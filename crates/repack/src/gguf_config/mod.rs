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

// Model depth is attacker-controlled GGUF metadata. Keep it well below any
// size that could make the per-layer masks an allocation sink while leaving
// ample headroom above every architecture this port supports.
const MAX_MODEL_LAYERS: i64 = 4096;

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
fn vocab_size(header: &GgufHeader, hidden_size: i64) -> Result<i64, GgufConfigError> {
    let name = "token_embd.weight";
    let info = header
        .tensors
        .get(name)
        .ok_or(GgufConfigError::MissingTensor {
            name: name.to_string(),
        })?;
    // Dims are stored fastest-varying first, so this is [hidden, vocab].
    // The first dimension is also the independently encoded row. Checking
    // only the product would allow quantization blocks to straddle rows.
    if info.dims.len() != 2 || info.dims[0] != hidden_size as u64 {
        return Err(GgufConfigError::BadValue {
            key: name.to_string(),
            detail: format!("expected [{hidden_size}, vocab], got {:?}", info.dims),
        });
    }
    let (block_elems, _) =
        crate::gguf_header::ggml_type_block(info.ggml_type).ok_or_else(|| {
            GgufConfigError::BadValue {
                key: name.to_string(),
                detail: format!("unsupported ggml type {}", info.ggml_type),
            }
        })?;
    if info.dims[0] % block_elems != 0 {
        return Err(GgufConfigError::BadValue {
            key: name.to_string(),
            detail: format!(
                "row width {} is not divisible by the type-{} block size {block_elems}",
                info.dims[0], info.ggml_type
            ),
        });
    }
    Ok(info.dims[1] as i64)
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
    if family == ModelFamily::MiniMaxM2 {
        // Missing weights must not silently change this untied architecture.
        if !header.tensors.contains_key("output.weight") {
            return Err(GgufConfigError::MissingTensor {
                name: "output.weight".into(),
            });
        }
        for (key, expected) in [
            ("attention.key_length", 128),
            ("attention.value_length", 128),
            ("rope.dimension_count", 64),
            ("expert_gating_func", 2),
        ] {
            if m.i64(key)? != expected {
                return Err(GgufConfigError::BadValue {
                    key: m.key(key),
                    detail: format!("MiniMax-M2 requires {expected}"),
                });
            }
        }
        if m.opt_f64("attention.layer_norm_rms_epsilon")
            .map(|v| v as f32)
            != Some(1e-6f32)
        {
            return Err(GgufConfigError::BadValue {
                key: m.key("attention.layer_norm_rms_epsilon"),
                detail: "MiniMax-M2 requires RMS epsilon 1e-6".into(),
            });
        }
    }
    if family == ModelFamily::Qwen2Dense
        && m.opt_f64("attention.layer_norm_rms_epsilon")
            .map(|v| v as f32)
            != Some(1e-6f32)
    {
        return Err(GgufConfigError::BadValue {
            key: m.key("attention.layer_norm_rms_epsilon"),
            detail: "Qwen2 requires RMS epsilon 1e-6".into(),
        });
    }

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
    if !(1..=MAX_MODEL_LAYERS).contains(&block_count) {
        return Err(GgufConfigError::BadValue {
            key: m.key("block_count"),
            detail: format!("must be between 1 and {MAX_MODEL_LAYERS}"),
        });
    }
    // Guard the head subtraction separately so its error names the metadata
    // responsible; the resulting trunk depth is bounded below.
    if mtp_blocks < 0 || (mtp_blocks > 0 && mtp_blocks >= block_count) {
        return Err(GgufConfigError::BadValue {
            key: m.key("nextn_predict_layers"),
            detail: format!("{mtp_blocks} of {block_count} blocks leaves no trunk"),
        });
    }
    let num_layers = block_count - mtp_blocks;
    if !(1..=MAX_MODEL_LAYERS).contains(&num_layers) {
        return Err(GgufConfigError::BadValue {
            key: m.key("block_count"),
            detail: format!("{num_layers} trunk layers is outside 1..={MAX_MODEL_LAYERS}"),
        });
    }
    let num_layers_usize = usize::try_from(num_layers).map_err(|_| GgufConfigError::BadValue {
        key: m.key("block_count"),
        detail: format!("{num_layers} trunk layers cannot be represented on this platform"),
    })?;
    arch.num_layers = num_layers;
    arch.hidden_size = m.i64("embedding_length")?;
    arch.num_heads = m.i64("attention.head_count")?;
    // A DENSE checkpoint publishes none of the three MoE keys, and the
    // `llama` architecture covers both halves (Mixtral has them, Llama 3.1
    // does not), so they are optional and default to the dense answer.
    arch.num_experts = m.opt_i64("expert_count")?.unwrap_or(0);
    arch.top_k_experts = m.opt_i64("expert_used_count")?.unwrap_or(0);
    if family == ModelFamily::MiniMaxM2 && arch.num_experts <= 0 {
        return Err(GgufConfigError::BadValue {
            key: m.key("expert_count"),
            detail: "MiniMax expert count must be positive".into(),
        });
    }
    // Gemma and Qwen publish a separate expert width; the `llama`
    // architecture does not, and its experts are `feed_forward_length` wide.
    // Measured absent on both Mixtral conversions, and asserted so in
    // `gguf_checkpoint_network.rs::scopes_phase_m2_from_the_mixtral_header`.
    arch.moe_intermediate_size = match m.opt_i64("expert_feed_forward_length")? {
        Some(width) => width,
        None if arch.num_experts > 0 => m.i64("feed_forward_length")?,
        None => 0,
    };
    arch.vocab_size = vocab_size(header, arch.hidden_size)?;
    // Untied only when the checkpoint actually ships a separate head.
    arch.tie_word_embeddings = !header.tensors.contains_key("output.weight");

    arch.full_attention_layer_mask = match family {
        ModelFamily::Gemma4 => gemma4_layer_mask(&m, num_layers_usize)?,
        // Spark-X2.5 publishes the SAME bool-array key with the SAME
        // convention (true = this layer slides), so the Gemma builder IS
        // this family's builder; only the rope divergence below is its own.
        ModelFamily::Spark25 => gemma4_layer_mask(&m, num_layers_usize)?,
        // ONE builder for both halves, because both publish the same
        // `full_attention_interval` key and the same every-fourth-layer rule.
        // The dense half was refused here until `ornith-ai/Ornith-1.5-9B-GGUF`
        // became the first published `qwen35` file; before it the mask would
        // have been invented, which is why the refusal was right at the time.
        ModelFamily::QwenGdnMoe | ModelFamily::QwenGdnDense => {
            qwen_gdn_moe_layer_mask(&m, num_layers_usize)?
        }
        ModelFamily::Qwen4Exp => qwen4_exp_layer_mask(&m, num_layers_usize)?,
        // Every layer is full attention: neither a dense Llama nor a Mixtral
        // publishes `attention.sliding_window`, and Mistral 7B's window is a
        // property of that model rather than of the architecture.
        // Qwen3-MoE is the same story: no `attention.sliding_window` key on
        // the published file, every layer full attention. Dense `qwen3`
        // agrees: no sliding window at any published size (`docs/QWEN3_PHASE0.md`).
        ModelFamily::Llama
        | ModelFamily::Qwen3Moe
        | ModelFamily::Qwen3Dense
        | ModelFamily::Qwen2Dense
        | ModelFamily::MiniMaxM2 => {
            vec![1u8; num_layers_usize]
        }
        ModelFamily::GptOss => gpt_oss_layer_mask(&m, num_layers_usize)?,
        // Every `deepseek2` layer carries MLA attention (mask 5); the dense
        // LEAD is an FFN difference only, not an attention one, and this
        // architecture publishes no sliding-window key.
        ModelFamily::Deepseek2 => vec![5u8; num_layers_usize],
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
        // `qwen3_vl` refuses for `qwen4_exp`'s reason: GGUFs exist, the
        // converter's tower naming is unparsed, and the MLX path is the
        // intake this port built.
        ModelFamily::DeepseekV4Flash | ModelFamily::MuseGlimmer | ModelFamily::Qwen3Vl => {
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
            // The mscale family parameter arrives as llama.cpp's
            // `yarn_log_multiplier`, which is `0.1 * mscale`. Absent means
            // the plain-YaRN parameter 1.0, which `RopeScalingConfig`'s
            // zero-fallback already means (the multiplier derives as
            // `1 + 0.1 * mscale_param * ln(factor)` at the use site).
            // DeepSeek's is 0.0707; gpt-oss publishes none and keeps the
            // plain value, which is what preserves every pre-existing
            // install byte for byte.
            mscale: m
                .opt_f64("rope.scaling.yarn_log_multiplier")
                .map(|v| v * 10.0)
                .unwrap_or(0.0),
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
    if matches!(
        family,
        ModelFamily::QwenGdnMoe | ModelFamily::QwenGdnDense | ModelFamily::Qwen4Exp
    ) {
        arch.linear_attention = linear_attention(&m)?;
        // Qwen's key/value length are per-head and equal on both paths.
        if let Some(k) = m.opt_i64("attention.key_length")? {
            arch.head_dim = k;
            arch.full_head_dim = k;
        }
        if family == ModelFamily::Qwen4Exp {
            arch.linear_attention.output_gate_sigmoid = true;
            derive_qwen4_exp_extensions(&m, &mut arch)?;
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

    // DeepSeek V2 MLA: the head geometry lives in `mla`, not in the
    // `head_dim` pair, and the cache is ONE shared row per layer. The
    // checkpoint's `attention.head_count_kv = 16` describes its EXPANDED
    // form (what llama.cpp caches when it runs this file non-absorbed);
    // this port's absorbed form reads `[latent ; rope key]` once per token,
    // so `num_kv_heads` is 1 here whatever the file says
    // (`docs/DEEPSEEK2_PHASE0.md`). The generic `attention.key_length`
    // handling above already set both head-dim fields to 192; the baseline
    // semantics here are `head_dim` = the q/k head width and
    // `full_head_dim` = the per-head value width, and the mla block is the
    // authority the runtime reads.
    if family == ModelFamily::Deepseek2 {
        let kv_lora = m.i64("attention.kv_lora_rank")?;
        let rope_dim = m
            .opt_i64("rope.dimension_count")?
            .ok_or(GgufConfigError::MissingKey {
                key: m.key("rope.dimension_count"),
            })?;
        let key_length = m.i64("attention.key_length")?;
        if rope_dim <= 0 || rope_dim % 2 != 0 || rope_dim >= key_length {
            return Err(GgufConfigError::BadValue {
                key: m.key("rope.dimension_count"),
                detail: format!(
                    "{rope_dim} must be a positive even count below key_length {key_length}"
                ),
            });
        }
        let nope = key_length - rope_dim;
        if kv_lora <= 0 || nope <= 0 {
            return Err(GgufConfigError::BadValue {
                key: m.key("attention.kv_lora_rank"),
                detail: format!("latent {kv_lora} / nope {nope} must both be positive"),
            });
        }
        arch.mla = model_io::MlaConfig {
            kv_lora_rank: kv_lora,
            // The lite variants carry no q low-rank; the full ones publish
            // `attention.q_lora_rank`, which this port does not execute yet
            // and refuses below rather than silently dropping the branch.
            q_lora_rank: m.opt_i64("attention.q_lora_rank")?.unwrap_or(0),
            nope_head_dim: nope,
            rope_head_dim: rope_dim,
            v_head_dim: m.opt_i64("attention.value_length")?.unwrap_or(nope),
        };
        if arch.mla.q_lora_rank > 0 {
            return Err(GgufConfigError::BadValue {
                key: m.key("attention.q_lora_rank"),
                detail: format!(
                    "q low-rank {} is the full V2/V3 branch, which this port does not \
                     execute yet; the lite checkpoints carry none",
                    arch.mla.q_lora_rank
                ),
            });
        }
        arch.num_kv_heads = 1;
        arch.num_full_kv_heads = 1;
        arch.head_dim = arch.mla.key_head_dim();
        arch.full_head_dim = arch.mla.v_head_dim;
        // The shared expert width: the file publishes only the per-expert
        // width and the shared COUNT, and the shexp tensors are the two
        // fused into one SwiGLU. The dense lead's width is
        // `feed_forward_length`, which the generic fallback above would
        // have put in `intermediate_size` -- the two are different numbers
        // here (10944 against 2816) and each names a different projection.
        if let Some(shared_count) = m.opt_i64("expert_shared_count")? {
            if shared_count > 0 {
                arch.intermediate_size = arch.moe_intermediate_size * shared_count;
            }
        }
        if let Some(dense_ff) = m.opt_i64("feed_forward_length")? {
            arch.dense_lead_intermediate_size = dense_ff;
        }
        // llama.cpp's `leading_dense_block_count` / HF's
        // `first_k_dense_replace`. Also a POSITION: the packed-expert blob
        // files are numbered from the first routed layer, so the runtime
        // maps checkpoint layer L to blob position L minus this count.
        arch.num_dense_leading_layers = m.opt_i64("leading_dense_block_count")?.unwrap_or(0).max(0);
        if arch.num_dense_leading_layers >= arch.num_layers {
            return Err(GgufConfigError::BadValue {
                key: m.key("leading_dense_block_count"),
                detail: format!(
                    "{} dense lead layers leaves no MoE stack of {}",
                    arch.num_dense_leading_layers, arch.num_layers
                ),
            });
        }
    }

    // Spark-X2.5: the converter encodes the per-class PARTIAL FACTORS as
    // rotary dimensions (`int(head_dim * factor)`), which this port derives
    // from `partial_rotary_factor` instead. When the file publishes them,
    // cross-check the two derivations: a baseline whose factor disagreed
    // with the file would otherwise rotate a different width than the
    // conversion was built for, which is fluent rather than loud. Absent
    // keys skip the check -- the install's behavior comes from this port's
    // own baseline fields either way, and a file without the keys predates
    // the conventions this family mirrors rather than contradicting them.
    if family == ModelFamily::Spark25 {
        let full_head = arch.full_head_dim as f64;
        let sliding_head = arch.head_dim as f64;
        let expect_full = (full_head * arch.partial_rotary_factor).round() as i64;
        if let Some(dims) = m.opt_i64("rope.dimension_count")? {
            if dims != expect_full {
                return Err(GgufConfigError::BadValue {
                    key: m.key("rope.dimension_count"),
                    detail: format!(
                        "file rotates {dims} dims on the full layers but this port derives \
                         {expect_full} (head_dim {full_head} x partial_rotary_factor {})",
                        arch.partial_rotary_factor
                    ),
                });
            }
        }
        if let Some(dims) = m.opt_i64("rope.dimension_count_swa")? {
            // The SWA arm rotates the whole head (factor 1.0, the gemma4
            // pattern the flow hard-codes); the file must agree.
            if dims != sliding_head as i64 {
                return Err(GgufConfigError::BadValue {
                    key: m.key("rope.dimension_count_swa"),
                    detail: format!(
                        "file rotates {dims} dims on the sliding layers but this port rotates \
                         the whole head ({sliding_head})"
                    ),
                });
            }
        }
    }

    Ok(arch)
}

/// Qwen4Exp publishes its hybrid mask directly: `true` means recurrent GDN,
/// while the runtime's mask uses 2 for GDN and 1 for full attention.
fn qwen4_exp_layer_mask(m: &Meta<'_>, num_layers: usize) -> Result<Vec<u8>, GgufConfigError> {
    let key = m.key("attention.recurrent_layers");
    let values = m
        .opt("attention.recurrent_layers")
        .and_then(GgufValue::as_array)
        .ok_or_else(|| GgufConfigError::BadValue {
            key: key.clone(),
            detail: "not a boolean array".to_string(),
        })?;
    if values.len() != num_layers {
        return Err(GgufConfigError::BadValue {
            key,
            detail: format!("{} entries for {num_layers} layers", values.len()),
        });
    }
    values
        .iter()
        .map(|value| {
            value
                .as_bool()
                .map(|recurrent| if recurrent { 2 } else { 1 })
                .ok_or_else(|| GgufConfigError::BadValue {
                    key: m.key("attention.recurrent_layers"),
                    detail: "contains a non-boolean layer entry".to_string(),
                })
        })
        .collect()
}

/// Fields that llama.cpp keeps in Qwen4Exp-specific GGUF metadata rather
/// than in the common attention/SSM keys.
fn derive_qwen4_exp_extensions(m: &Meta<'_>, arch: &mut ArchConfig) -> Result<(), GgufConfigError> {
    let bad = |suffix: &str, detail: String| GgufConfigError::BadValue {
        key: m.key(suffix),
        detail,
    };

    let dimension = m.i64("rope.dimension_count")?;
    if dimension <= 0 || arch.full_head_dim <= 0 || dimension > arch.full_head_dim {
        return Err(bad(
            "rope.dimension_count",
            format!(
                "rotary width {dimension} must be positive and no greater than full head width {}",
                arch.full_head_dim
            ),
        ));
    }
    arch.partial_rotary_factor = dimension as f64 / arch.full_head_dim as f64;

    let hc_count = m.i64("hyper_connection.count")?;
    let hc_lowrank = m.i64("hyper_connection.low_rank")?;
    if hc_count <= 1 || hc_lowrank <= 0 {
        return Err(bad(
            "hyper_connection.count",
            format!("hyper-connection count {hc_count} and low rank {hc_lowrank} are invalid"),
        ));
    }
    arch.hyper_connections.mult = hc_count;
    arch.hyper_connections.lowrank = hc_lowrank;

    let compress_key = "attention.compress_ratios";
    let compress_values = integer_array(m, compress_key)?;
    if compress_values.len() != arch.full_attention_layer_mask.len() {
        return Err(bad(
            compress_key,
            format!(
                "{} entries for {} layers",
                compress_values.len(),
                arch.full_attention_layer_mask.len()
            ),
        ));
    }
    let mut compress_rate = None;
    for (layer, (&kind, &ratio)) in arch
        .full_attention_layer_mask
        .iter()
        .zip(&compress_values)
        .enumerate()
    {
        match kind {
            1 if ratio > 0 => {
                match compress_rate {
                    Some(expected) if expected != ratio => {
                        return Err(bad(
                        compress_key,
                        format!("full-attention layer {layer} uses ratio {ratio}, expected {expected}"),
                    ));
                    }
                    None => compress_rate = Some(ratio),
                    _ => {}
                }
            }
            2 if ratio == 0 => {}
            _ => {
                return Err(bad(
                    compress_key,
                    format!("layer {layer} mask {kind} disagrees with compression ratio {ratio}"),
                ));
            }
        }
    }
    let compress_rate = compress_rate.ok_or_else(|| {
        bad(
            compress_key,
            "no full-attention layer has a compression ratio".to_string(),
        )
    })?;
    let index_budget = m.i64("attention.indexer.top_k")?;
    if index_budget <= 0 || index_budget % compress_rate != 0 {
        return Err(bad(
            "attention.indexer.top_k",
            format!("index budget {index_budget} is not divisible by ratio {compress_rate}"),
        ));
    }
    arch.compressed_attention.index_n_heads = m.i64("attention.indexer.head_count")?;
    arch.compressed_attention.index_head_dim = m.i64("attention.indexer.key_length")?;
    arch.compressed_attention.index_budget = index_budget;
    arch.compressed_attention.csa_compress_rate = compress_rate;
    arch.compressed_attention.index_top_k = index_budget / compress_rate;

    let ngram_size = m.i64("ple.ngram_size")?;
    let heads_per_ngram = m.i64("ple.heads_per_ngram")?;
    let heads = ngram_size
        .checked_sub(1)
        .and_then(|orders| orders.checked_mul(heads_per_ngram))
        .filter(|&count| count > 0)
        .ok_or_else(|| bad("ple.ngram_size", "invalid PLE head count".to_string()))?;
    let per_layer_head_dim = m.i64("embedding_length_per_layer_input")?;
    if per_layer_head_dim <= 0 {
        return Err(bad(
            "embedding_length_per_layer_input",
            "must be positive".to_string(),
        ));
    }
    let ple_embed_dim = per_layer_head_dim.checked_mul(heads).ok_or_else(|| {
        bad(
            "ple.ngram_size",
            "PLE embedding width overflows".to_string(),
        )
    })?;
    // The HF config spells PLE layers one-based, but llama.cpp stores the
    // corresponding zero-based block indices in GGUF. ArchConfig keeps the
    // original one-based contract used by the safetensors path and runtime.
    let layer_indices = integer_array(m, "ple.layers")?;
    if layer_indices.is_empty()
        || layer_indices
            .iter()
            .any(|&layer| layer < 0 || layer >= arch.num_layers)
    {
        return Err(bad(
            "ple.layers",
            format!(
                "zero-based GGUF layer indices {layer_indices:?} are outside 0..{}",
                arch.num_layers
            ),
        ));
    }
    let layer_ids = layer_indices.into_iter().map(|layer| layer + 1).collect();
    arch.ple.ngram_size = ngram_size;
    arch.ple.heads_per_ngram = heads_per_ngram;
    arch.ple.ple_embed_dim = ple_embed_dim;
    arch.ple.conv_kernel_size = m.i64("ple.conv_kernel")?;
    arch.ple.layer_ids = layer_ids;
    arch.ple.eos_token_id = m.i64("ple.eos_token_id")?;

    Ok(())
}

fn integer_array(m: &Meta<'_>, suffix: &str) -> Result<Vec<i64>, GgufConfigError> {
    let key = m.key(suffix);
    let values =
        m.opt(suffix)
            .and_then(GgufValue::as_array)
            .ok_or_else(|| GgufConfigError::BadValue {
                key: key.clone(),
                detail: "not an unsigned integer array".to_string(),
            })?;
    values
        .iter()
        .map(|value| {
            let raw = value.as_u64().ok_or_else(|| GgufConfigError::BadValue {
                key: key.clone(),
                detail: "contains a non-unsigned integer".to_string(),
            })?;
            i64::try_from(raw).map_err(|_| GgufConfigError::BadValue {
                key: key.clone(),
                detail: "contains a value above i64::MAX".to_string(),
            })
        })
        .collect()
}

#[cfg(test)]
mod qwen4_exp_tests {
    use std::collections::BTreeMap;

    use super::*;

    fn uint_array(values: &[u64]) -> GgufValue {
        GgufValue::Array(values.iter().copied().map(GgufValue::U64).collect())
    }

    #[test]
    fn qwen4_exp_gguf_extensions_follow_the_published_metadata() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "qwen4exp.attention.recurrent_layers".into(),
            GgufValue::Array(vec![
                GgufValue::Bool(true),
                GgufValue::Bool(true),
                GgufValue::Bool(true),
                GgufValue::Bool(false),
            ]),
        );
        metadata.insert(
            "qwen4exp.attention.compress_ratios".into(),
            uint_array(&[0, 0, 0, 4]),
        );
        metadata.insert("qwen4exp.rope.dimension_count".into(), GgufValue::U64(64));
        metadata.insert("qwen4exp.hyper_connection.count".into(), GgufValue::U64(4));
        metadata.insert(
            "qwen4exp.hyper_connection.low_rank".into(),
            GgufValue::U64(320),
        );
        metadata.insert(
            "qwen4exp.attention.indexer.top_k".into(),
            GgufValue::U64(2048),
        );
        metadata.insert(
            "qwen4exp.attention.indexer.head_count".into(),
            GgufValue::U64(4),
        );
        metadata.insert(
            "qwen4exp.attention.indexer.key_length".into(),
            GgufValue::U64(128),
        );
        metadata.insert("qwen4exp.ple.ngram_size".into(), GgufValue::U64(3));
        metadata.insert("qwen4exp.ple.heads_per_ngram".into(), GgufValue::U64(8));
        metadata.insert(
            "qwen4exp.embedding_length_per_layer_input".into(),
            GgufValue::U64(160),
        );
        metadata.insert("qwen4exp.ple.layers".into(), uint_array(&[1]));
        metadata.insert("qwen4exp.ple.conv_kernel".into(), GgufValue::U64(4));
        metadata.insert("qwen4exp.ple.eos_token_id".into(), GgufValue::U64(248_044));
        let header = GgufHeader {
            version: 3,
            metadata,
            tensors: Default::default(),
            alignment: 32,
            data_region_start: 0,
        };
        let m = Meta {
            header: &header,
            prefix: "qwen4exp",
        };
        let mut arch = model_io::known_architecture(ModelFamily::Qwen4Exp);
        arch.num_layers = 4;
        arch.full_attention_layer_mask = qwen4_exp_layer_mask(&m, 4).expect("hybrid mask");
        derive_qwen4_exp_extensions(&m, &mut arch).expect("Qwen4Exp metadata");

        assert_eq!(arch.full_attention_layer_mask, vec![2, 2, 2, 1]);
        assert_eq!(arch.partial_rotary_factor, 0.25);
        assert_eq!(arch.hyper_connections.mult, 4);
        assert_eq!(arch.hyper_connections.lowrank, 320);
        assert_eq!(arch.compressed_attention.index_n_heads, 4);
        assert_eq!(arch.compressed_attention.index_kv_heads, 1);
        assert_eq!(arch.compressed_attention.index_head_dim, 128);
        assert_eq!(arch.compressed_attention.index_budget, 2048);
        assert_eq!(arch.compressed_attention.index_top_k, 512);
        assert_eq!(arch.compressed_attention.csa_compress_rate, 4);
        assert_eq!(arch.ple.ngram_size, 3);
        assert_eq!(arch.ple.heads_per_ngram, 8);
        assert_eq!(arch.ple.ple_embed_dim, 2560);
        assert_eq!(arch.ple.layer_ids, vec![2]);
        assert_eq!(arch.ple.layer_indices(), vec![1]);
        assert_eq!(arch.ple.conv_kernel_size, 4);
        assert_eq!(arch.ple.eos_token_id, 248_044);
    }
}
