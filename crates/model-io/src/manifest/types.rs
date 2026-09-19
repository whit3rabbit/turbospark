use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::arch_config::VisionConfig;

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
    /// The gated DeltaNet output norm's activation: sigmoid when true, silu
    /// when false or absent. `qwen4_exp`'s `output_gate_type`.
    ///
    /// Absent means SILU, which is what every family before `qwen4_exp`
    /// declares and what `gdn_gated_norm` did unconditionally -- so every
    /// install already on disk keeps the behaviour it was written with
    /// (`crates/gpu` Gotcha 12).
    #[serde(default)]
    pub linear_output_gate_sigmoid: Option<bool>,

    /// Multi-head latent attention (`deepseek2`). All five are `Option` with
    /// zero fallback: absent means `MlaConfig::NONE`, never another family's
    /// value (AGENTS.md Gotcha 39).
    #[serde(default)]
    pub mla_kv_lora_rank: Option<i64>,
    /// The query low-rank; absent (or zero) is the lite variants' answer.
    #[serde(default)]
    pub mla_q_lora_rank: Option<i64>,
    #[serde(default)]
    pub mla_nope_head_dim: Option<i64>,
    #[serde(default)]
    pub mla_rope_head_dim: Option<i64>,
    #[serde(default)]
    pub mla_v_head_dim: Option<i64>,
    /// FFN width of the leading dense layers (`first_k_dense_replace`);
    /// absent means none.
    #[serde(default)]
    pub dense_lead_intermediate_size: Option<i64>,
    /// How many leading layers are dense; the routed blob files start after
    /// them. Absent means none.
    #[serde(default)]
    pub num_dense_leading_layers: Option<i64>,

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
    /// Indexer key/value head count (`qwen4_exp`'s `indexer_kv_heads`).
    #[serde(default)]
    pub ca_index_kv_heads: Option<i64>,
    /// Context length below which the indexer selects nothing
    /// (`qwen4_exp`'s `indexer_budget`).
    #[serde(default)]
    pub ca_index_budget: Option<i64>,
    /// Hyper-connection multiplier: the number of residual streams.
    #[serde(default)]
    pub hc_mult: Option<i64>,
    /// Hyper-connection Sinkhorn iterations. DeepSeek mHC only.
    #[serde(default)]
    pub hc_sinkhorn_iters: Option<i64>,
    /// Hyper-connection epsilon. DeepSeek mHC only.
    #[serde(default)]
    pub hc_eps: Option<f64>,
    /// Rank of the stream-mixing bottleneck. `qwen4_exp` only.
    #[serde(default)]
    pub hc_lowrank: Option<i64>,
    /// The hashed n-gram PLE table (`qwen4_exp`). Absent means no table,
    /// which is `PleConfig::NONE` and what every other family declares.
    #[serde(default)]
    pub ple_ngram_size: Option<i64>,
    #[serde(default)]
    pub ple_heads_per_ngram: Option<i64>,
    #[serde(default)]
    pub ple_ngram_vocab_size_base: Option<i64>,
    #[serde(default)]
    pub ple_make_divisible_by: Option<i64>,
    #[serde(default)]
    pub ple_split_ngram_parts: Option<i64>,
    #[serde(default)]
    pub ple_embed_dim: Option<i64>,
    #[serde(default)]
    pub ple_conv_kernel_size: Option<i64>,
    /// ONE-BASED layer ids, as the checkpoint spells them.
    #[serde(default)]
    pub ple_layer_ids: Option<Vec<i64>>,
    #[serde(default)]
    pub ple_seed: Option<i64>,
    #[serde(default)]
    pub ple_eos_token_id: Option<i64>,
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
    /// YaRN's `mscale` family parameter (`deepseek2`; `0.1 * mscale` is what
    /// the GGUF spelling calls `yarn_log_multiplier`). Absent means the
    /// plain-YaRN parameter 1.0, which preserves every pre-`deepseek2`
    /// manifest; the multiplier itself derives at the use site.
    #[serde(default)]
    pub rope_scaling_mscale: Option<f64>,
    /// The vision tower (ROADMAP M-V3). Absent means no tower, which is what
    /// `VisionConfig::NONE` says and what every install written before M-V3
    /// declares -- which is why all fifteen are `Option` and default rather
    /// than being required alongside the shape fields.
    #[serde(default)]
    pub vision_depth: Option<i64>,
    #[serde(default)]
    pub vision_hidden_size: Option<i64>,
    #[serde(default)]
    pub vision_intermediate_size: Option<i64>,
    #[serde(default)]
    pub vision_num_heads: Option<i64>,
    #[serde(default)]
    pub vision_patch_size: Option<i64>,
    #[serde(default)]
    pub vision_temporal_patch_size: Option<i64>,
    #[serde(default)]
    pub vision_in_channels: Option<i64>,
    #[serde(default)]
    pub vision_spatial_merge_size: Option<i64>,
    #[serde(default)]
    pub vision_num_position_embeddings: Option<i64>,
    #[serde(default)]
    pub vision_out_hidden_size: Option<i64>,
    /// mRoPE's `(t, h, w)` channel split. A fixed-size array rather than a
    /// `Vec`, so a file declaring the wrong number of sections is refused by
    /// serde at decode rather than indexed out of bounds at a dispatch.
    #[serde(default)]
    pub vision_mrope_section: Option<[i64; 3]>,
    #[serde(default)]
    pub vision_start_token_id: Option<i64>,
    #[serde(default)]
    pub vision_end_token_id: Option<i64>,
    #[serde(default)]
    pub vision_image_token_id: Option<i64>,
    #[serde(default)]
    pub vision_video_token_id: Option<i64>,
    /// The deepstack merger block indices. Absent on every install written
    /// before the field existed, which is exactly what an EMPTY list means
    /// (`VisionConfig`'s own default), so `Option` + `unwrap_or_default`
    /// resolves silence and explicit-`[]` to the same value.
    #[serde(default)]
    pub vision_deepstack_visual_indexes: Option<Vec<i64>>,
}

impl ManifestArch {
    /// Resolves the fifteen `vision*` fields into a [`VisionConfig`], the
    /// same way `arch_validation::validate_arch` resolves each of them:
    /// `.unwrap_or(0)` (or the equivalent empty triple), never a baseline's
    /// value. An absent field means the install declares no tower, and
    /// another family's answer about ITS tower is not evidence (AGENTS.md
    /// Gotcha 24) -- which is why this does NOT take a baseline to fall
    /// back to, unlike every other family-extension field on this struct.
    ///
    /// The one place both the manifest LOADER (`arch_validation`, inline)
    /// and a READER of an already-loaded manifest (`crates/repack`'s
    /// `manifest_peek`, and `model_io::vision_sidecar`) resolve this, so the
    /// two cannot drift the way they did once already (`crates/repack`
    /// Gotcha 18).
    pub fn vision_config(&self) -> VisionConfig {
        VisionConfig {
            depth: self.vision_depth.unwrap_or(0),
            hidden_size: self.vision_hidden_size.unwrap_or(0),
            intermediate_size: self.vision_intermediate_size.unwrap_or(0),
            num_heads: self.vision_num_heads.unwrap_or(0),
            patch_size: self.vision_patch_size.unwrap_or(0),
            temporal_patch_size: self.vision_temporal_patch_size.unwrap_or(0),
            in_channels: self.vision_in_channels.unwrap_or(0),
            spatial_merge_size: self.vision_spatial_merge_size.unwrap_or(0),
            num_position_embeddings: self.vision_num_position_embeddings.unwrap_or(0),
            out_hidden_size: self.vision_out_hidden_size.unwrap_or(0),
            mrope_section: self.vision_mrope_section.unwrap_or([0, 0, 0]),
            vision_start_token_id: self.vision_start_token_id.unwrap_or(0),
            vision_end_token_id: self.vision_end_token_id.unwrap_or(0),
            image_token_id: self.vision_image_token_id.unwrap_or(0),
            video_token_id: self.vision_video_token_id.unwrap_or(0),
            deepstack_visual_indexes: self
                .vision_deepstack_visual_indexes
                .clone()
                .unwrap_or_default(),
        }
    }
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
    /// Prism Hadamard-folded weight contract (the Bonsai-2 line), present only
    /// on installs whose quantized weights live in a signed block-Hadamard
    /// rotated basis. Absent on every install written before it existed, which
    /// is the correct default: no transform, weights consumed as stored.
    #[serde(default)]
    pub hadamard: Option<ManifestHadamard>,
}

/// One sign vector of the Hadamard contract: a `width`-long run of +/-1 F32
/// values inside the install's sibling `hadamard.bin` file.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestHadamardSigns {
    /// Activation width this vector applies to. Every folded entry whose
    /// input width is this value shares the vector (verified equal across
    /// modules on the real checkpoint).
    pub width: i64,
    /// Byte offset of the run inside `hadamard.bin`.
    pub offset: u64,
    /// Byte length of the run; `width * 4` (F32).
    pub bytes: u64,
}

/// The Hadamard-folded weight contract: quantized matrices stored as
/// `W' = W * diag(signs) * H_block` (Hadamard butterflies over `block`-sized
/// segments of the input axis), so the RUNTIME transforms activations, not
/// weights -- the forward transform on every folded entry's input, and the
/// inverse transform on the embedding's dequantized rows (the one entry whose
/// OUTPUT is the rotated activation). `hadamard.bin` carries the sign vectors
/// beside `model_weights.bin`, keyed from this section.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestHadamard {
    /// Butterfly block width in elements. Every folded input width must be a
    /// multiple of this.
    pub block: i64,
    /// One sign vector per distinct folded input width.
    pub signs: Vec<ManifestHadamardSigns>,
    /// Resident entry names whose INPUT activations carry the forward
    /// transform before the matmul.
    pub folded: Vec<String>,
    /// Resident entry names whose OUTPUT rows carry the inverse transform
    /// after dequantize. The embedding, on the one real checkpoint.
    pub inverse: Vec<String>,
}
