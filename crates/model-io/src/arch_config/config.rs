use super::family::ModelFamily;
use super::sub_configs::{
    CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig, MlaConfig, PleConfig,
    RopeScalingConfig, VisionConfig,
};

/// `full_attention_layer_mask` values: 0 = sliding-window attention, 1 =
/// full attention, 2 = gated-DeltaNet linear attention, 3 = compressed
/// sparse attention (CSA), 4 = heavily compressed attention (HCA), 5 =
/// multi-head latent attention (MLA, `deepseek2`).
#[derive(Debug, Clone, PartialEq)]
pub struct ArchConfig {
    /// Model hidden dimension size.
    pub hidden_size: i64,
    /// Shared-expert FFN width (== `ffn_intermediate` in the manifest).
    pub intermediate_size: i64,
    /// Per-expert FFN width.
    pub moe_intermediate_size: i64,
    /// Number of query attention heads.
    pub num_heads: i64,
    /// Number of key/value attention heads for SWA.
    pub num_kv_heads: i64,
    /// Number of key/value attention heads for full attention.
    pub num_full_kv_heads: i64,
    /// Head dimension size for SWA.
    pub head_dim: i64,
    /// Head dimension size for full attention.
    pub full_head_dim: i64,
    /// Vocabulary size.
    pub vocab_size: i64,
    /// Sliding window context size.
    pub sliding_window: i64,
    /// Final logit soft-capping value.
    pub final_logit_softcap: f64,
    /// RoPE base theta value.
    pub rope_theta: f64,
    /// Full-attention RoPE base theta value.
    pub full_rope_theta: f64,
    /// Partial rotary dimension scaling factor.
    pub partial_rotary_factor: f64,
    /// Number of layers in model architecture.
    pub num_layers: i64,
    /// How many leading layers are DENSE (`first_k_dense_replace`): the
    /// routed-expert blob files start after them, which is a POSITION in
    /// `packed_experts/`, not just a width. Zero means no dense lead, which
    /// is every family here except `deepseek2`.
    pub num_dense_leading_layers: i64,
    /// The dense lead layers' FFN width.
    ///
    /// A separate field from [`Self::intermediate_size`] because [`Self::intermediate_size`] is the SHARED
    /// expert's width on this architecture (2816) and the dense lead's is
    /// a different number (10944); one scalar cannot carry both, and a
    /// baseline that overloaded either would produce a layer that runs at
    /// the wrong width with no error. Read off the GGUF pair
    /// `feed_forward_length` (dense lead) against the shexp tensors'
    /// second dim (shared), not inferred.
    pub dense_lead_intermediate_size: i64,
    /// Number of routed experts in MoE layers.
    pub num_experts: i64,
    /// Number of selected experts per token.
    pub top_k_experts: i64,
    /// True if input word embeddings are tied with final lm_head.
    pub tie_word_embeddings: bool,
    /// True if K and V projections share memory structures.
    pub attention_k_eq_v: bool,
    /// Per-layer attention variant mask.
    pub full_attention_layer_mask: Vec<u8>,
    /// Activation function name for hidden MLP layers.
    pub hidden_activation: String,
    /// Model architecture family discriminator.
    pub family: ModelFamily,
    /// Full-attention q_proj emits `2 * num_heads * full_head_dim` rows:
    /// per-head [query; gate] halves.
    pub attn_output_gate: bool,
    /// Attention scaling factor (e.g. 1 / sqrt(head_dim)).
    pub attention_scale: f64,
    /// True if input embedding is scaled by sqrt(hidden_size).
    pub embedding_scaled_by_sqrt_hidden: bool,
    /// True if router weights are scaled.
    pub router_scaled: bool,
    /// True if FFN layers use sandwich RMSNorm.
    pub ffn_sandwich_norms: bool,
    /// True if shared expert output is scalar-gated.
    pub shared_expert_gated: bool,
    /// True if RoPE subdim uses NeoX permutation style.
    pub rope_neox_subdim: bool,
    /// Gated-DeltaNet linear attention configuration.
    pub linear_attention: LinearAttentionConfig,
    /// Multi-head latent attention configuration (`deepseek2`), or
    /// [`MlaConfig::NONE`].
    ///
    /// Resolved explicitly to `NONE` when the manifest omits it, never to a
    /// Gemma value, for the reason [`Self::vision`] states: an omitted
    /// family-extension field is otherwise validated against GEMMA's value
    /// whatever family the manifest claims (AGENTS.md Gotcha 24).
    /// `build_manifest_json` writes it unconditionally.
    pub mla: MlaConfig,
    /// Compressed attention configuration.
    pub compressed_attention: CompressedAttentionConfig,
    /// Hyper-connection configuration.
    pub hyper_connections: HyperConnectionConfig,
    /// Leading MoE layers whose expert selection is a frozen token-id
    /// lookup instead of a learned argmax. 0 for Gemma/Qwen.
    pub num_hash_routed_layers: i64,
    /// "softmax" (Gemma/Qwen) or "sqrtsoftplus" (DeepSeek V4).
    pub router_scoring_func: String,
    /// Scaling factor for routed expert outputs.
    pub routed_scaling_factor: f64,
    /// Clamp for expert gate/up pre-activations. 0 = no clamp.
    pub swiglu_limit: f64,
    /// YaRN rope scaling, or [`RopeScalingConfig::NONE`].
    pub rope_scaling: RopeScalingConfig,
    /// The vision tower, or [`VisionConfig::NONE`] (ROADMAP M-V3).
    ///
    /// Optional in the manifest and defaulted to `NONE`, which is what makes
    /// every install written before M-V3 keep opening. Read AGENTS.md Gotcha
    /// 24 before touching that default: an omitted family-extension field is
    /// resolved against the GEMMA baseline whatever family the manifest
    /// claims, so this one is resolved explicitly to `NONE` rather than to
    /// `gemma_defaults.vision`, and `build_manifest_json` writes it
    /// unconditionally.
    pub vision: VisionConfig,
    /// The hashed n-gram PLE table, or [`PleConfig::NONE`] (`qwen4_exp`).
    ///
    /// Resolved explicitly to `NONE` when the manifest omits it, never to
    /// `gemma_defaults.ple`, for the reason [`Self::vision`] states: an omitted
    /// family-extension field is otherwise validated against GEMMA's value
    /// whatever family the manifest claims (AGENTS.md Gotcha 24).
    /// `build_manifest_json` writes it unconditionally.
    pub ple: PleConfig,
}

impl ArchConfig {
    /// Resident INT4 GEMV shapes this architecture issues during decode:
    /// `(m, n)` pairs.
    pub fn decode_int4_gemv_shapes(&self) -> Vec<(i64, i64)> {
        let mut shapes = Vec::new();
        if self.has_compressed_attention_layers() {
            let ca = &self.compressed_attention;
            shapes.push((ca.q_lora_rank, self.hidden_size));
            shapes.push((self.num_heads * self.full_head_dim, ca.q_lora_rank));
            shapes.push((self.full_head_dim, self.hidden_size));
            shapes.push((
                ca.o_lora_rank,
                self.num_heads * self.full_head_dim / ca.o_groups,
            ));
            shapes.push((self.hidden_size, ca.o_groups * ca.o_lora_rank));
            shapes.push((2 * self.full_head_dim, self.hidden_size));
            shapes.push((self.full_head_dim, self.hidden_size));
            shapes.push((2 * ca.index_head_dim, self.hidden_size));
            shapes.push((ca.index_n_heads * ca.index_head_dim, ca.q_lora_rank));
        } else if self.attn_output_gate {
            shapes.push((2 * self.num_heads * self.full_head_dim, self.hidden_size));
        } else {
            shapes.push((self.num_heads * self.full_head_dim, self.hidden_size));
        }
        if !self.has_compressed_attention_layers() {
            shapes.push((
                self.num_full_kv_heads * self.full_head_dim,
                self.hidden_size,
            ));
            shapes.push((self.hidden_size, self.num_heads * self.full_head_dim));
        }
        if self.has_linear_attention_layers() {
            let la = &self.linear_attention;
            shapes.push((la.qkv_dim(), self.hidden_size));
            shapes.push((la.value_dim(), self.hidden_size));
            shapes.push((self.hidden_size, la.value_dim()));
        }
        shapes.push((self.intermediate_size, self.hidden_size));
        shapes.push((self.hidden_size, self.intermediate_size));
        shapes
    }

    /// Resident INT8 GEMV shapes issued during decode (router and,
    /// when present, the shared-expert scalar gate).
    pub fn decode_int8_gemv_shapes(&self) -> Vec<(i64, i64)> {
        let mut shapes = vec![(self.num_experts, self.hidden_size)];
        if self.shared_expert_gated {
            shapes.push((1, self.hidden_size));
        }
        shapes
    }

    /// Returns true if layer at index runs full attention.
    pub fn layer_is_full(&self, layer: usize) -> bool {
        self.full_attention_layer_mask[layer] == 1
    }
    /// Returns true if layer at index runs linear attention.
    pub fn layer_is_linear(&self, layer: usize) -> bool {
        self.full_attention_layer_mask[layer] == 2
    }
    /// Returns true if layer at index runs compressed sparse attention (CSA).
    pub fn layer_is_csa(&self, layer: usize) -> bool {
        self.full_attention_layer_mask[layer] == 3
    }
    /// Returns true if layer at index runs multi-head latent attention (MLA).
    pub fn layer_is_mla(&self, layer: usize) -> bool {
        self.full_attention_layer_mask[layer] == 5
    }
    /// Returns true if layer at index runs heavily compressed attention (HCA).
    pub fn layer_is_hca(&self, layer: usize) -> bool {
        self.full_attention_layer_mask[layer] == 4
    }
    /// True for any layer carrying a compressed long-range KV branch.
    pub fn layer_is_compressed(&self, layer: usize) -> bool {
        matches!(self.full_attention_layer_mask[layer], 3 | 4)
    }
    /// Returns true if model contains linear attention layers.
    pub fn has_linear_attention_layers(&self) -> bool {
        self.full_attention_layer_mask.contains(&2)
    }
    /// Returns true if model contains compressed attention layers.
    pub fn has_compressed_attention_layers(&self) -> bool {
        self.full_attention_layer_mask
            .iter()
            .any(|&v| v == 3 || v == 4)
    }
    /// Returns true if model contains MLA layers.
    pub fn has_mla_layers(&self) -> bool {
        self.full_attention_layer_mask.contains(&5)
    }
    /// Hash-routed MoE layer: expert selection is `tid2eid[token]`.
    pub fn layer_is_hash_routed(&self, layer: usize) -> bool {
        (layer as i64) < self.num_hash_routed_layers
    }
}
