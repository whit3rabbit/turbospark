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
fn layer_prefix(layer: usize) -> String {
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
        // Precomputed RoPE frequencies. This port derives its own from
        // `rope_theta` in the manifest, so carrying them would be dead
        // weight in the install.
        "rope_freqs.weight" => GgufMapping::Ignored {
            reason: "RoPE frequencies are derived from rope_theta at runtime",
        },
        _ => return None,
    })
}

/// Gemma 4's per-layer suffixes, verified against
/// `gemma-4-26B-A4B-it-Q8_0.gguf` and `~/models/gemma4.gturbo`.
fn map_gemma4_layer(suffix: &str, layer: usize) -> Option<GgufMapping> {
    let p = layer_prefix(layer);
    let resident = |tail: &str| Some(GgufMapping::Resident(format!("{p}{tail}")));
    match suffix {
        "attn_q.weight" => resident("self_attn.q_proj.weight"),
        "attn_k.weight" => resident("self_attn.k_proj.weight"),
        "attn_v.weight" => resident("self_attn.v_proj.weight"),
        "attn_output.weight" => resident("self_attn.o_proj.weight"),
        "attn_q_norm.weight" => resident("self_attn.q_norm.weight"),
        "attn_k_norm.weight" => resident("self_attn.k_norm.weight"),
        "attn_norm.weight" => resident("input_layernorm.weight"),
        "post_attention_norm.weight" => resident("post_attention_layernorm.weight"),
        // GGUF calls the pre-FFN norm `ffn_norm` and only numbers the
        // SECOND one; the install spells both out.
        "ffn_norm.weight" => resident("pre_feedforward_layernorm.weight"),
        "pre_ffw_norm_2.weight" => resident("pre_feedforward_layernorm_2.weight"),
        "post_ffw_norm.weight" => resident("post_feedforward_layernorm.weight"),
        "post_ffw_norm_1.weight" => resident("post_feedforward_layernorm_1.weight"),
        "post_ffw_norm_2.weight" => resident("post_feedforward_layernorm_2.weight"),
        // The dense/shared-expert FFN.
        "ffn_gate.weight" => resident("mlp.gate_proj.weight"),
        "ffn_up.weight" => resident("mlp.up_proj.weight"),
        "ffn_down.weight" => resident("mlp.down_proj.weight"),
        // Router. The two `.scale` tensors are matched by shape against the
        // install: `ffn_gate_inp.scale` is [hidden] like `router.scale`, and
        // `ffn_down_exps.scale` is [experts] like `router.per_expert_scale`
        // -- note that the latter is a ROUTER tensor despite its name
        // living under the down-projection's prefix.
        "ffn_gate_inp.weight" => resident("router.proj.weight"),
        "ffn_gate_inp.scale" => resident("router.scale"),
        "ffn_down_exps.scale" => resident("router.per_expert_scale"),
        "layer_output_scale.weight" => resident("layer_scalar"),
        "ffn_gate_up_exps.weight" => Some(GgufMapping::RoutedFusedGateUp { layer }),
        "ffn_down_exps.weight" => Some(GgufMapping::Routed {
            layer,
            role: "down",
        }),
        _ => None,
    }
}

/// Qwen 3.6's per-layer suffixes, verified against
/// `Qwen3.6-35B-A3B-Q4_K_M.gguf` and `~/models/qwen36.gturbo`. Note the
/// hybrid layer split: the 10 full-attention layers carry `attn_q/k/v`, the
/// 30 linear-attention layers carry `attn_qkv` plus the `ssm_*` family, and
/// no layer carries both.
fn map_qwen36_layer(suffix: &str, layer: usize) -> Option<GgufMapping> {
    let p = layer_prefix(layer);
    let resident = |tail: &str| Some(GgufMapping::Resident(format!("{p}{tail}")));
    match suffix {
        "attn_q.weight" => resident("self_attn.q_proj.weight"),
        "attn_k.weight" => resident("self_attn.k_proj.weight"),
        "attn_v.weight" => resident("self_attn.v_proj.weight"),
        "attn_output.weight" => resident("self_attn.o_proj.weight"),
        "attn_q_norm.weight" => resident("self_attn.q_norm.weight"),
        "attn_k_norm.weight" => resident("self_attn.k_norm.weight"),
        "attn_norm.weight" => resident("input_layernorm.weight"),
        "post_attention_norm.weight" => resident("post_attention_layernorm.weight"),
        // Gated DeltaNet. GGUF names these after the SSM family it borrows
        // its tensor slots from; the install names them after what they do.
        "attn_qkv.weight" => resident("linear_attn.in_proj_qkv.weight"),
        "attn_gate.weight" => resident("linear_attn.in_proj_z.weight"),
        "ssm_alpha.weight" => resident("linear_attn.in_proj_a.weight"),
        "ssm_beta.weight" => resident("linear_attn.in_proj_b.weight"),
        "ssm_out.weight" => resident("linear_attn.out_proj.weight"),
        "ssm_conv1d.weight" => resident("linear_attn.conv1d.weight"),
        "ssm_norm.weight" => resident("linear_attn.norm.weight"),
        // Gotcha 26: no `.weight` suffix on either of these, on either side.
        "ssm_a" => resident("linear_attn.A_log"),
        "ssm_dt.bias" => resident("linear_attn.dt_bias"),
        "ffn_gate_inp.weight" => resident("mlp.gate.weight"),
        "ffn_gate_inp_shexp.weight" => resident("mlp.shared_expert_gate.weight"),
        "ffn_gate_shexp.weight" => resident("mlp.shared_expert.gate_proj.weight"),
        "ffn_up_shexp.weight" => resident("mlp.shared_expert.up_proj.weight"),
        "ffn_down_shexp.weight" => resident("mlp.shared_expert.down_proj.weight"),
        "ffn_gate_exps.weight" => Some(GgufMapping::Routed {
            layer,
            role: "gate",
        }),
        "ffn_up_exps.weight" => Some(GgufMapping::Routed { layer, role: "up" }),
        "ffn_down_exps.weight" => Some(GgufMapping::Routed {
            layer,
            role: "down",
        }),
        _ => None,
    }
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
        return match family {
            ModelFamily::Gemma4 => map_gemma4_layer(suffix, layer),
            ModelFamily::Qwen36 => map_qwen36_layer(suffix, layer),
            ModelFamily::DeepseekV4Flash => None,
        }
        .ok_or_else(unmapped);
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
        ModelFamily::Qwen36 => Some("qwen35moe"),
        ModelFamily::DeepseekV4Flash => None,
    }
}

/// Resolve a GGUF `general.architecture` string to a family.
pub fn family_for_architecture(arch: &str) -> Option<ModelFamily> {
    match arch {
        "gemma4" => Some(ModelFamily::Gemma4),
        "qwen35moe" => Some(ModelFamily::Qwen36),
        _ => None,
    }
}
