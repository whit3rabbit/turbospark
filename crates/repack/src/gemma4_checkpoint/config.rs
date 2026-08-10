//! Gemma 4 and Qwen 3.6 config and quantization spec parsing.

use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily,
};

use crate::ranged_download::DownloadError;

#[derive(Debug)]
pub enum Gemma4Error {
    Config(String),
    MissingTensor(String),
    MissingCompanion(String),
    UnsupportedDtype { tensor: String, dtype: String },
    ShapeMismatch { tensor: String, detail: String },
    UnknownTensor(String),
    Download(DownloadError),
}

impl std::fmt::Display for Gemma4Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Gemma4Error::Config(detail) => write!(f, "config.json invalid: {detail}"),
            Gemma4Error::MissingTensor(name) => write!(f, "missing tensor: {name}"),
            Gemma4Error::MissingCompanion(name) => {
                write!(f, "missing .scales/.biases companion for: {name}")
            }
            Gemma4Error::UnsupportedDtype { tensor, dtype } => {
                write!(f, "tensor {tensor} has unsupported dtype {dtype}")
            }
            Gemma4Error::ShapeMismatch { tensor, detail } => {
                write!(f, "tensor {tensor}: {detail}")
            }
            Gemma4Error::UnknownTensor(name) => write!(f, "unclassifiable tensor: {name}"),
            Gemma4Error::Download(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Gemma4Error {}

impl From<DownloadError> for Gemma4Error {
    fn from(e: DownloadError) -> Self {
        Gemma4Error::Download(e)
    }
}

// ---------------------------------------------------------------------------
// config.json -> ArchConfig
// ---------------------------------------------------------------------------

/// Parses a Gemma 4 `config.json` (the `text_config` wrapper form the
/// multimodal checkpoints ship) into an [`ArchConfig`], mirroring the Swift
/// `ArchInfo.loadGemma4`: `layer_types` maps to the layer mask
/// (`full_attention` = 1, anything else = 0), the two `rope_parameters`
/// sub-objects supply the full/SWA thetas and the partial rotary factor,
/// and every family-extension field takes Gemma 4's fixed value.
pub fn parse_gemma4_config(json: &str) -> Result<ArchConfig, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    crate::arch_registry::refuse_foreign_config(&root, ModelFamily::Gemma4)
        .map_err(Gemma4Error::Config)?;
    let tc = root
        .get("text_config")
        .ok_or_else(|| Gemma4Error::Config("no text_config".to_string()))?;

    let i = |k: &str| -> Result<i64, Gemma4Error> {
        tc.get(k)
            .and_then(|v| v.as_i64())
            .ok_or_else(|| Gemma4Error::Config(format!("missing {k}")))
    };
    let d = |k: &str| -> Result<f64, Gemma4Error> {
        tc.get(k)
            .and_then(|v| v.as_f64())
            .ok_or_else(|| Gemma4Error::Config(format!("missing {k}")))
    };

    let mask: Vec<u8> = tc
        .get("layer_types")
        .and_then(|v| v.as_array())
        .map(|types| {
            types
                .iter()
                .map(|t| (t.as_str() == Some("full_attention")) as u8)
                .collect()
        })
        .unwrap_or_default();

    let rope = tc.get("rope_parameters");
    let rope_sub = |kind: &str, key: &str, fallback: f64| -> f64 {
        rope.and_then(|r| r.get(kind))
            .and_then(|s| s.get(key))
            .and_then(|v| v.as_f64())
            .unwrap_or(fallback)
    };
    let prf = rope_sub("full_attention", "partial_rotary_factor", 0.25);
    let full_theta = rope_sub("full_attention", "rope_theta", 1_000_000.0);
    let swa_theta = rope_sub("sliding_attention", "rope_theta", 10_000.0);

    let b = |k: &str| tc.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
    let act = tc
        .get("hidden_activation")
        .and_then(|v| v.as_str())
        .unwrap_or("gelu_pytorch_tanh")
        .to_string();

    Ok(ArchConfig {
        hidden_size: i("hidden_size")?,
        intermediate_size: i("intermediate_size")?,
        moe_intermediate_size: i("moe_intermediate_size")?,
        num_heads: i("num_attention_heads")?,
        num_kv_heads: i("num_key_value_heads")?,
        num_full_kv_heads: i("num_global_key_value_heads")?,
        head_dim: i("head_dim")?,
        full_head_dim: i("global_head_dim")?,
        vocab_size: i("vocab_size")?,
        sliding_window: i("sliding_window")?,
        final_logit_softcap: d("final_logit_softcapping")?,
        rope_theta: swa_theta,
        full_rope_theta: full_theta,
        partial_rotary_factor: prf,
        num_layers: i("num_hidden_layers")?,
        num_experts: i("num_experts")?,
        top_k_experts: i("top_k_experts")?,
        tie_word_embeddings: b("tie_word_embeddings"),
        attention_k_eq_v: b("attention_k_eq_v"),
        full_attention_layer_mask: mask,
        hidden_activation: act,
        family: ModelFamily::Gemma4,
        attn_output_gate: false,
        attention_scale: 1.0,
        embedding_scaled_by_sqrt_hidden: true,
        router_scaled: true,
        ffn_sandwich_norms: true,
        shared_expert_gated: false,
        rope_neox_subdim: false,
        linear_attention: LinearAttentionConfig::NONE,
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
    })
}

/// Per-tensor quantization widths, parsed from the checkpoint's own
/// `config.json -> quantization` object (MLX convention: a global
/// `group_size`/`bits` pair plus per-tensor override entries keyed by the
/// tensor base path -- an object with its own `bits`, or `false` for
/// tensors left unquantized, which pass-through handles by dtype anyway).
#[derive(Debug, Clone)]
pub struct Gemma4Quant {
    pub default_bits: u32,
    pub group_size: u32,
    pub bits_overrides: std::collections::HashMap<String, u32>,
}

impl Default for Gemma4Quant {
    /// 4-bit, 64-element groups, no overrides: the base mode this port's
    /// GEMV kernels assume.
    fn default() -> Self {
        Gemma4Quant {
            default_bits: 4,
            group_size: 64,
            bits_overrides: std::collections::HashMap::new(),
        }
    }
}

impl Gemma4Quant {
    pub fn bits_for(&self, base_name: &str) -> u32 {
        // Overrides are keyed by the source path without the `.weight`
        // suffix; try both to be safe.
        self.bits_overrides
            .get(base_name)
            .copied()
            .unwrap_or(self.default_bits)
    }
}

/// Parses `config.json -> quantization` into a [`Gemma4Quant`]. A missing
/// `quantization` object yields the 4-bit/group-64 default.
pub fn parse_gemma4_quantization(json: &str) -> Result<Gemma4Quant, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    let Some(q) = root.get("quantization").and_then(|v| v.as_object()) else {
        return Ok(Gemma4Quant::default());
    };
    let default_bits = q.get("bits").and_then(|v| v.as_u64()).unwrap_or(4) as u32;
    let group_size = q.get("group_size").and_then(|v| v.as_u64()).unwrap_or(64) as u32;
    let mut bits_overrides = std::collections::HashMap::new();
    for (key, value) in q {
        if key == "bits" || key == "group_size" || key == "mode" {
            continue;
        }
        if let Some(obj) = value.as_object() {
            if let Some(bits) = obj.get("bits").and_then(|v| v.as_u64()) {
                bits_overrides.insert(key.clone(), bits as u32);
            }
            if let Some(gs) = obj.get("group_size").and_then(|v| v.as_u64()) {
                if gs as u32 != group_size {
                    return Err(Gemma4Error::Config(format!(
                        "per-tensor group_size {gs} for {key} differs from the \
                         global {group_size}; this port's kernels assume one group size"
                    )));
                }
            }
        }
    }
    if group_size != 64 {
        return Err(Gemma4Error::Config(format!(
            "group_size {group_size} unsupported: this port's GEMV kernels assume 64"
        )));
    }
    Ok(Gemma4Quant {
        default_bits,
        group_size,
        bits_overrides,
    })
}
