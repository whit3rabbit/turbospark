//! Maps GGUF tensor names onto the canonical HF-style names the rest of the
//! repack pipeline already speaks, so [`crate::classify_for_family`], the
//! routed marker, the resident ordering, and `RealForwardRunner`'s lookups
//! all stay shared between the safetensors and GGUF intakes.
//!
//! EVERY ROW HERE WAS READ OFF A REAL FILE, not inferred from the format
//! docs. The left column comes from the ranged header fetch in
//! `tests/gguf_checkpoint_network.rs` against `ggml-org`'s conversions of
//! both supported architectures; the right column comes from the resident
//! index of the corresponding `.gturbo` install this port already runs.
//! Where the two disagreed, the install won, because it is what the runtime
//! actually looks up.
//!
//! Three findings from that comparison are load-bearing:
//!
//! 1. **Gemma 4 fuses gate and up into one routed tensor**
//!    (`ffn_gate_up_exps`, `[in, 2 * ffn, experts]`) where the MLX
//!    checkpoint keeps them separate. Qwen 3.6 does NOT fuse. So the split
//!    is one family's problem, not a general one.
//! 2. **Gemma 4's global-attention layers have no V projection at all.**
//!    Layers 5, 11, 17, 23, 29 carry `attn_k` at `[2816, 1024]` and simply
//!    omit `attn_v`. This is a property of the model, not of GGUF: the MLX
//!    install is missing `v_proj` on exactly those five layers too. A
//!    mapping that treated a missing per-layer tensor as an error would
//!    reject a perfectly good checkpoint.
//! 3. **Gemma 4 ties its embeddings and Qwen 3.6 does not.** Gemma's GGUF
//!    has no `output.weight` and its install has no `lm_head.weight`; Qwen
//!    has both. Again the two intakes agree, so nothing special is needed
//!    beyond not requiring the tensor.
//!
//! AGENTS.md Gotcha 26 is respected here rather than rediscovered: Qwen's
//! `linear_attn.A_log` and `linear_attn.dt_bias` take NO `.weight` suffix,
//! and the rows below spell them out.

mod deepseek2;
mod gemma4;
mod gpt_oss;
mod llama;
mod minimax;
mod qwen;
mod spark;

use model_io::ModelFamily;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufNameError {
    /// A name this port has no row for. Never downgraded to a skip: an
    /// unrecognized routed marker is the silent failure Gotcha 26 records,
    /// where every expert quietly becomes a resident tensor and the install
    /// runs correctly at many times the intended footprint.
    Unmapped { name: String, family: &'static str },
    /// `blk.<n>.` did not parse, or named a layer past `block_count`.
    BadLayerIndex { name: String },
    /// A pre-merge Mixtral conversion: one tensor per expert
    /// (`blk.0.ffn_down.3.weight`) where current llama.cpp writes one 3-D
    /// `ffn_down_exps`. Refused BY NAME rather than skipped, because the two
    /// layouts hold the same weights in the same order and misreading one as
    /// the other is silent (ROADMAP Phase M2).
    PreMergeExperts { name: String },
    /// `rope_freqs.weight`: a LEARNED per-dimension RoPE frequency scaling
    /// vector (Llama 3.1's, `[64]` F32), which this port's two rope kernels
    /// cannot express -- they take a scalar theta (ROADMAP M4).
    ///
    /// Refused BY NAME rather than ignored, and the distinction is the whole
    /// point. It WAS ignored, on the reading that RoPE frequencies are
    /// derived from `rope_theta` at runtime, which is true of every file
    /// that omits this tensor and false of every file that ships it. Dropping
    /// it produces an install that loads, decodes, and is wrong only at long
    /// context -- the failure mode with no symptom at the length anyone
    /// smoke-tests. Harmless until ROADMAP M4 made dense `llama` installs
    /// runnable, at which point a Llama 3.1 checkpoint became reachable.
    UnsupportedRopeScaling { name: String },
}

impl std::fmt::Display for GgufNameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GgufNameError::Unmapped { name, family } => {
                write!(f, "GGUF tensor {name} has no {family} mapping")
            }
            GgufNameError::BadLayerIndex { name } => {
                write!(f, "GGUF tensor {name} has an unparseable layer index")
            }
            GgufNameError::PreMergeExperts { name } => write!(
                f,
                "GGUF tensor {name} is a pre-merge per-expert tensor: this \
                 conversion predates llama.cpp merging experts into one 3-D \
                 ffn_*_exps, and this port reads only the merged layout. \
                 Re-convert the checkpoint, or fetch a build published after \
                 the merge"
            ),
            GgufNameError::UnsupportedRopeScaling { name } => write!(
                f,
                "GGUF tensor {name} is a learned per-dimension RoPE frequency \
                 scaling vector (Llama 3.1's), and this port's rope kernels take \
                 a scalar theta. Carrying the checkpoint without it would be \
                 wrong only at long context, so it is refused instead. Use a \
                 checkpoint that does not ship rope_freqs.weight (Mistral 7B, \
                 Llama 2, TinyLlama), or land scaled RoPE first"
            ),
        }
    }
}

impl std::error::Error for GgufNameError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufMapping {
    /// A resident tensor, carried into `model_weights.bin` verbatim under
    /// this canonical name.
    Resident(String),
    /// A routed-expert tensor for one layer and one role, sliced per expert.
    Routed { layer: usize, role: &'static str },
    /// Gemma 4's fused routed tensor: gate and up concatenated along the
    /// output dimension. Split into two roles by the walk.
    RoutedFusedGateUp { layer: usize },
    /// Recognized and deliberately not carried. `reason` is reported so a
    /// dropped tensor is a visible decision rather than a silent omission.
    Ignored { reason: &'static str },
}

/// Canonical prefix for per-layer tensors.
pub(crate) fn layer_prefix(layer: usize) -> String {
    format!("language_model.model.layers.{layer}.")
}

/// Rows shared by both families.
fn map_top_level(name: &str) -> Option<GgufMapping> {
    Some(match name {
        "token_embd.weight" => {
            GgufMapping::Resident("language_model.model.embed_tokens.weight".to_string())
        }
        "output_norm.weight" => {
            GgufMapping::Resident("language_model.model.norm.weight".to_string())
        }
        "output.weight" => GgufMapping::Resident("language_model.lm_head.weight".to_string()),
        _ => return None,
    })
}

/// Map one GGUF tensor name for `family`.
pub fn map_gguf_name(name: &str, family: ModelFamily) -> Result<GgufMapping, GgufNameError> {
    let unmapped = || GgufNameError::Unmapped {
        name: name.to_string(),
        family: family.as_str(),
    };

    if let Some(rest) = name.strip_prefix("blk.") {
        let (idx, suffix) = rest
            .split_once('.')
            .ok_or_else(|| GgufNameError::BadLayerIndex {
                name: name.to_string(),
            })?;
        let layer: usize = idx.parse().map_err(|_| GgufNameError::BadLayerIndex {
            name: name.to_string(),
        })?;
        if family == ModelFamily::Llama && llama::is_pre_merge_expert(suffix) {
            return Err(GgufNameError::PreMergeExperts {
                name: name.to_string(),
            });
        }
        return match family {
            ModelFamily::Gemma4 => gemma4::map_gemma4_layer(suffix, layer),
            ModelFamily::QwenGdnMoe => qwen::map_qwen_gdn_moe_layer(suffix, layer),
            ModelFamily::Llama => llama::map_llama_layer(suffix, layer),
            ModelFamily::Qwen2Dense => llama::map_qwen2_layer(suffix, layer),
            ModelFamily::Qwen3Moe => qwen::map_qwen3moe_layer(suffix, layer),
            ModelFamily::GptOss => gpt_oss::map_gpt_oss_layer(suffix, layer),
            ModelFamily::QwenGdnDense => qwen::map_qwen_gdn_dense_layer(suffix, layer),
            ModelFamily::Spark25 => spark::map_spark_layer(suffix, layer),
            ModelFamily::Qwen3Dense => qwen::map_qwen3_dense_layer(suffix, layer),
            ModelFamily::MiniMaxM2 => minimax::map_layer(suffix, layer),
            ModelFamily::Deepseek2 => deepseek2::map_deepseek2_layer(suffix, layer),
            // Neither of the first two is published as a GGUF; an unmapped
            // name is the right answer rather than a neighbour's table,
            // which would map names these families do not have.
            //
            // `qwen4_exp` IS published as a GGUF and is unmapped anyway. Its
            // trunk names would largely fall out of Qwen 3.6's table, and
            // that is exactly the trap: borrowing it would map the shared
            // two thirds and silently drop the n-gram shards, the
            // hyper-connections and the indexer, which is a partial install
            // that opens. Unmapped until the whole file has a table.
            ModelFamily::DeepseekV4Flash | ModelFamily::MuseGlimmer | ModelFamily::Qwen4Exp => None,
        }
        .ok_or_else(unmapped);
    }

    // Refused for every family, before the table, so no family can grow a
    // row for it by accident. See `UnsupportedRopeScaling`: this used to be
    // an `Ignored` row, which is the one disposition that produces a wrong
    // model rather than an error.
    if name == "rope_freqs.weight" {
        return Err(GgufNameError::UnsupportedRopeScaling {
            name: name.to_string(),
        });
    }
    map_top_level(name).ok_or_else(unmapped)
}

/// The GGUF `general.architecture` string for a family.
///
/// NOT the family's own `as_str()`. llama.cpp named Qwen 3.6's converter
/// after the 3.5 series it shares a graph with, so a real Qwen 3.6 GGUF
/// reports `qwen35moe`. Deriving this from `ModelFamily::as_str` would
/// silently fail to recognize every Qwen GGUF in the wild.
pub fn gguf_architecture(family: ModelFamily) -> Option<&'static str> {
    match family {
        ModelFamily::Gemma4 => Some("gemma4"),
        ModelFamily::QwenGdnMoe => Some("qwen35moe"),
        ModelFamily::Llama => Some("llama"),
        ModelFamily::Qwen3Moe => Some("qwen3moe"),
        ModelFamily::GptOss => Some("gpt-oss"),
        // The dense half drops the `moe` suffix rather than sharing the
        // string, which is what keeps `family_for_architecture` injective.
        // Read off `ornith-ai/Ornith-1.5-9B-GGUF`.
        ModelFamily::QwenGdnDense => Some("qwen35"),
        // Read off `XHToken/Spark-X2.5-4B-GGUF`'s Q4_K_M: the GGUF
        // architecture string and the HF `model_type` agree here, and the
        // conventions are an upstream llama.cpp merge rather than a fork.
        ModelFamily::Spark25 => Some("spark2_5"),
        // Read off llama.cpp's own `gguf-py/gguf/constants.py`
        // (`MODEL_ARCH.QWEN3 => "qwen3"`) and confirmed against the real
        // `Qwen/Qwen3-4B-GGUF` header, distinct from `qwen3moe` above.
        ModelFamily::Qwen3Dense => Some("qwen3"),
        ModelFamily::MiniMaxM2 => Some("minimax-m2"),
        ModelFamily::Qwen2Dense => Some("qwen2"),
        // Read off the V2-Lite header itself (`general.architecture`), and
        // the same spelling the HF `model_type` takes modulo the underscore.
        ModelFamily::Deepseek2 => Some("deepseek2"),
        // Neither of the first two is published as a GGUF. `None` is the
        // honest answer: inventing a string here would make
        // `family_for_architecture` claim to recognize a file that does not
        // exist.
        //
        // `qwen4_exp` is the opposite case and lands on the same answer.
        // Real GGUFs exist (unsloth's, bartowski's), so a string COULD be
        // read off one -- but this function feeds `family_for_architecture`,
        // whose whole contract is the architectures that RUN, and this port
        // refuses the GGUF at `gguf_config`. `Some` here would let a caller
        // mistake recognition for support, which is the distinction this
        // table exists to keep.
        ModelFamily::DeepseekV4Flash | ModelFamily::MuseGlimmer | ModelFamily::Qwen4Exp => None,
    }
}

/// Resolve a GGUF `general.architecture` string to a family, for the
/// architectures that RUN.
///
/// The table lives in [`crate::arch_registry`], which also carries the
/// recognized-but-unported strings; those return `None` here on purpose, so
/// no caller can mistake recognition for support (ROADMAP Phase M Stage 1).
pub fn family_for_architecture(arch: &str) -> Option<ModelFamily> {
    match crate::arch_registry::gguf_arch_support(arch) {
        Some(crate::arch_registry::ArchSupport::Supported(family)) => Some(family),
        _ => None,
    }
}
