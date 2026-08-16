use std::collections::BTreeMap;

use serde::Deserialize;

/// File size and SHA-256 entry in manifest.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ManifestFileEntry {
    /// Expected file size in bytes.
    pub size: u64,
    /// Expected SHA-256 digest string.
    pub sha256: String,
}

/// Serialized architecture specification in `manifest.json`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestArch {
    /// Hidden dimension size.
    pub hidden_size: i64,
    /// Shared expert intermediate size.
    pub ffn_intermediate: i64,
    /// Routed expert intermediate size.
    pub moe_intermediate_size: i64,
    /// Number of query attention heads.
    pub num_heads: i64,
    /// Number of key/value heads for SWA layers.
    #[serde(rename = "numKVHeads")]
    pub num_kv_heads: i64,
    /// Number of key/value heads for full attention layers.
    #[serde(rename = "numFullKVHeads")]
    pub num_full_kv_heads: i64,
    /// Head dimension size for SWA layers.
    pub head_dim: i64,
    /// Head dimension size for full attention layers.
    pub full_head_dim: i64,
    /// Vocabulary size.
    pub vocab_size: i64,
    /// Sliding window context limit.
    pub sliding_window: i64,
    /// Final logit soft-capping value.
    pub final_logit_softcap: f64,
    /// RoPE base theta value.
    pub rope_theta: f64,
    /// Full-attention RoPE base theta value.
    pub full_rope_theta: f64,
    /// Partial rotary dimension factor.
    pub partial_rotary_factor: f64,
    /// Total layer count.
    pub num_layers: i64,
    /// Total routed experts count per MoE layer.
    pub num_experts: i64,
    /// Selected expert count per token.
    pub top_k_experts: i64,
    /// True if word embeddings are tied to output lm_head.
    pub tie_word_embeddings: bool,
    /// True if key and value attention projections share memory structures.
    pub attention_k_eq_v: bool,
    /// Activation function name for MLP layers.
    pub hidden_activation: String,
    /// Per-layer attention type mask.
    pub full_attention_layer_mask: Vec<i64>,

    /// Model family identifier string.
    #[serde(default)]
    pub family: Option<String>,
    /// True if attention output is gated.
    #[serde(default)]
    pub attn_output_gate: Option<bool>,
    /// Attention scale factor.
    #[serde(default)]
    pub attention_scale: Option<f64>,
    /// True if embedding is scaled by sqrt(hidden_size).
    #[serde(default)]
    pub embedding_scaled_by_sqrt_hidden: Option<bool>,
    /// True if router weights are scaled.
    #[serde(default)]
    pub router_scaled: Option<bool>,
    /// True if sandwich RMSNorms are used in FFN.
    #[serde(default)]
    pub ffn_sandwich_norms: Option<bool>,
    /// True if shared expert output is scalar-gated.
    #[serde(default)]
    pub shared_expert_gated: Option<bool>,
    /// True if RoPE subdim uses NeoX ordering.
    #[serde(default)]
    pub rope_neox_subdim: Option<bool>,
    /// Linear attention key heads count.
    #[serde(default)]
    pub linear_num_k_heads: Option<i64>,
    /// Linear attention value heads count.
    #[serde(default)]
    pub linear_num_v_heads: Option<i64>,
    /// Linear attention key head dimension.
    #[serde(default)]
    pub linear_key_head_dim: Option<i64>,
    /// Linear attention value head dimension.
    #[serde(default)]
    pub linear_value_head_dim: Option<i64>,
    /// Linear attention depthwise conv kernel size.
    #[serde(default)]
    pub linear_conv_kernel_size: Option<i64>,

    /// Compressed attention Q LoRA rank.
    #[serde(default)]
    pub ca_q_lora_rank: Option<i64>,
    /// Compressed attention Out LoRA rank.
    #[serde(default)]
    pub ca_o_lora_rank: Option<i64>,
    /// Compressed attention output groups.
    #[serde(default)]
    pub ca_o_groups: Option<i64>,
    /// Compressed attention RoPE head dim.
    #[serde(default)]
    pub ca_rope_head_dim: Option<i64>,
    /// Compressed attention index heads count.
    #[serde(default)]
    pub ca_index_n_heads: Option<i64>,
    /// Compressed attention index head dim.
    #[serde(default)]
    pub ca_index_head_dim: Option<i64>,
    /// Compressed attention index top-k selection.
    #[serde(default)]
    pub ca_index_top_k: Option<i64>,
    /// Compressed attention CSA compress rate.
    #[serde(default, rename = "caCSACompressRate")]
    pub ca_csa_compress_rate: Option<i64>,
    /// Compressed attention HCA compress rate.
    #[serde(default, rename = "caHCACompressRate")]
    pub ca_hca_compress_rate: Option<i64>,
    /// Compressed attention RoPE theta.
    #[serde(default)]
    pub ca_compress_rope_theta: Option<f64>,
    /// Compressed attention RoPE scaling factor.
    #[serde(default)]
    pub ca_rope_scaling_factor: Option<f64>,
    /// Compressed attention RoPE scaling original max.
    #[serde(default)]
    pub ca_rope_scaling_original_max: Option<i64>,
    /// Compressed attention RoPE scaling beta fast.
    #[serde(default)]
    pub ca_rope_scaling_beta_fast: Option<f64>,
    /// Compressed attention RoPE scaling beta slow.
    #[serde(default)]
    pub ca_rope_scaling_beta_slow: Option<f64>,
    /// Hyper-connection multiplier.
    #[serde(default)]
    pub hc_mult: Option<i64>,
    /// Hyper-connection Sinkhorn iterations.
    #[serde(default)]
    pub hc_sinkhorn_iters: Option<i64>,
    /// Hyper-connection epsilon.
    #[serde(default)]
    pub hc_eps: Option<f64>,
    /// Hash-routed leading MoE layer count.
    #[serde(default)]
    pub num_hash_routed_layers: Option<i64>,
    /// Router scoring function name.
    #[serde(default)]
    pub router_scoring_func: Option<String>,
    /// Routed expert output scaling factor.
    #[serde(default)]
    pub routed_scaling_factor: Option<f64>,
    /// SwiGLU activation clamp limit.
    #[serde(default)]
    pub swiglu_limit: Option<f64>,
    /// YaRN rope scaling (ROADMAP M5, `gpt-oss`). Absent means no scaling,
    /// which is what `RopeScalingConfig::NONE` says and what every family
    /// before this one declares.
    #[serde(default)]
    pub rope_scaling_factor: Option<f64>,
    #[serde(default)]
    pub rope_scaling_original_context: Option<i64>,
    #[serde(default)]
    pub rope_scaling_beta_fast: Option<f64>,
    #[serde(default)]
    pub rope_scaling_beta_slow: Option<f64>,
}

/// Quantization parameters for a model component slot in `manifest.json`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestQuantSlot {
    /// Weight quantization bit width (e.g. 4 or 8).
    pub weight_bits: i64,
    /// Quantization scheme identifier (e.g. "affine" or "gguf").
    pub scheme: String,
    /// Scale data type name (e.g. "BF16").
    pub scale_type: String,
    /// Bias data type name (e.g. "BF16").
    pub bias_type: String,
    /// Quantization group element count.
    pub group_size: i64,
    /// The ggml block type, present only on `scheme: "gguf"` slots (ROADMAP
    /// Phase G). Optional because every affine install predates it and none
    /// writes it.
    #[serde(default)]
    pub ggml_type: Option<String>,
    /// Every ggml block type this slot carries, when it carries more than one
    /// (ROADMAP Phase S). Optional, and absent means "exactly `ggml_type`".
    ///
    /// A mixed sub-4-bit checkpoint needs this: the Phase S candidate's routed
    /// experts are IQ3_XXS gate/up over IQ4_NL down, with IQ4_XS and Q8_0 on
    /// layer 29, so no single type describes the slot. `ggml_type` stays as
    /// the DOMINANT one, which keeps a hand-read of the manifest informative;
    /// this is what the gate actually checks, member by member.
    #[serde(default)]
    pub ggml_types: Option<Vec<String>>,
}

impl ManifestQuantSlot {
    /// The block types this slot claims, dominant one first.
    pub(crate) fn declared_types(&self) -> Vec<&str> {
        match (&self.ggml_types, &self.ggml_type) {
            (Some(all), _) if !all.is_empty() => all.iter().map(String::as_str).collect(),
            (_, Some(one)) => vec![one.as_str()],
            _ => Vec::new(),
        }
    }
}

/// Grouped quantization configuration across model component slots.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestQuant {
    /// Embedding layer quantization slot.
    pub embedding: ManifestQuantSlot,
    /// Attention projections quantization slot.
    pub attention: ManifestQuantSlot,
    /// Router projection quantization slot.
    pub router: ManifestQuantSlot,
    /// Shared expert quantization slot.
    pub shared_expert: ManifestQuantSlot,
    /// Routed expert quantization slot.
    pub routed_expert: ManifestQuantSlot,
}

/// Top-level model manifest deserialized from `manifest.json`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Manifest {
    /// Format magic string ("GTURBO").
    pub magic: String,
    /// Major format version number.
    pub version_major: i64,
    /// Minor format version number.
    pub version_minor: i64,
    /// Feature flags map.
    pub flags: BTreeMap<String, bool>,
    /// Model identifier string.
    #[serde(rename = "modelID")]
    pub model_id: String,
    /// Source HF repository snapshot hash if recorded.
    #[serde(default)]
    pub source_snapshot_hash: Option<String>,
    /// Architecture configuration table.
    pub arch: ManifestArch,
    /// Quantization configuration table if present.
    #[serde(default)]
    pub quant: Option<ManifestQuant>,
    /// Installed files map matching filename to size and hash.
    pub files: BTreeMap<String, ManifestFileEntry>,
    /// Number of experts per layer.
    pub experts_per_layer: i64,
    /// Number of layers.
    pub num_layers: i64,
    /// Stride in bytes per expert in stream storage.
    pub expert_stride: u64,
}
