//! Compile-time architecture baselines and the family-dependent config
//! types they're built from. Ported from
//! `Infrastructure/ModelIO/ModelTypes.swift`.
//!
//! `manifest.json -> arch` must match the resolved [`ArchConfig`]
//! field-by-field at load time (see `manifest::validate_arch`); mismatches
//! produce [`crate::ModelError::ArchMismatch`].

/// Model family discriminator. Selects the tensor-name contract, the layer
/// graph shape, and family-specific kernel behavior. Stored in
/// `manifest.json -> arch.family`; absent means Gemma 4 (the format's
/// original architecture).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFamily {
    Gemma4,
    QwenGdnMoe,
    DeepseekV4Flash,
    /// The `llama` GGUF architecture, which covers dense Llama 2/3.x and
    /// Mistral AND the Mixtral MoEs -- one string, distinguished only by
    /// `expert_count` (ROADMAP Phase M2). The baseline is Mixtral's because
    /// every behavioural field is shared and only shape fields differ.
    Llama,
    /// The `qwen3moe` GGUF architecture (Qwen3-30B-A3B and siblings), the
    /// first FINE-GRAINED MoE brought up after Mixtral showed that being MoE
    /// is not enough for this engine's memory result (AGENTS.md Gotcha 36).
    ///
    /// It runs through the SAME decode flow as [`ModelFamily::Llama`]
    /// (`crates/runtime/src/families/llama/`), because the layer graph is
    /// identical: plain GQA, raw residual add, one post-attention norm
    /// feeding router and routed experts, no shared expert, no softcap,
    /// full-head NeoX RoPE. It differs in exactly two places, both carried
    /// by `RealLlamaState`: it norms q and k PER HEAD before RoPE, and its
    /// RMS epsilon is 1e-6 where the `llama` architecture's is 1e-5.
    Qwen3Moe,
    /// The `gpt-oss` GGUF architecture (ROADMAP M5), the third fine-grained
    /// MoE and the FIFTH real family. Chosen by the M5 Phase 0 survey on
    /// AGENTS.md Gotcha 36's axis: 12.6 MiB per expert against Llama 4
    /// Scout's 77.8 and DeepSeek-V3's 34.5, both of which want tens of GiB
    /// of slot cache and cannot stream here.
    ///
    /// Unlike [`ModelFamily::Qwen3Moe`] this is NOT another family on an
    /// existing flow. Its layer differs from every flow here in four ways,
    /// each a first for this port: per-projection BIASES on q/k/v/output and
    /// on the router and every routed expert, ATTENTION SINKS (one learned
    /// logit per q head, added to the softmax denominator only), YARN rope
    /// scaling, and a CLAMPED SwiGLU whose activation is a swish with alpha
    /// times `(up + 1)` rather than silu times `up`. Its alternating
    /// 128-token sliding window is the one part that is free, because
    /// Gemma's SWA ring already exists.
    GptOss,
    /// The `qwen3_5` HF architecture: the gated-DeltaNet hybrid in its DENSE
    /// form. Two published checkpoints, `prism-ml/Bonsai-27B-mlx-1bit`
    /// (ROADMAP's 1-bit entry) and `Qwen/Qwen3.8-27B`, which share one
    /// `ArchConfig` exactly and differ only in quantization.
    ///
    /// **NAMED FOR THE ARCHITECTURE, NOT A VERSION, AND ITS WIRE STRING IS
    /// FROZEN AT THE OLD SPELLING.** [`ModelFamily::as_str`] still returns
    /// `"qwen35"` and [`ModelFamily::parse`] still reads it, because that
    /// string is written into every install's `manifest.json` and read back
    /// at load: renaming it would invalidate every `.gturbo` directory ever
    /// built. So the on-disk identifier is a FORMAT CONSTANT, historical and
    /// deliberately not descriptive, while this Rust name is free to say
    /// what the family is. Do not "fix" the string to match the variant.
    ///
    /// The variant was called `Qwen35` until 2026-08-15 and the rename is
    /// what stopped the drift: upstream keeps `model_type: qwen3_5` stable
    /// across checkpoints named 3.5, 3.6 and 3.8, so a version-shaped name
    /// reads as "the 3.5 one" when it means "the gated-DeltaNet dense one".
    /// Its sibling was worse -- `Qwen36` was named after the Qwen 3.6
    /// checkpoint while matching `model_type: qwen3_5_moe`.
    ///
    /// It runs through the SAME decode flow as [`ModelFamily::QwenGdnMoe`]
    /// (`crates/runtime/src/families/qwen/`), because every BEHAVIOURAL
    /// field is shared -- gated DeltaNet on the linear layers, gated full
    /// attention on every fourth, `attn_output_gate`, `head_dim` 256,
    /// `partial_rotary_factor` 0.25 at theta 1e7, no sandwich norms, no
    /// softcap, silu -- and only SHAPE fields differ (hidden 5120 against
    /// 2048, 64 layers against 40, 24 q heads over 4 kv). Read off the
    /// checkpoint's own `config.json`, not assumed.
    ///
    /// It differs in exactly two ways, and the first is why it needs a
    /// branch rather than just a baseline: it is DENSE, one
    /// `mlp.{gate,up,down}_proj` per layer where Qwen 3.6 has a router, a
    /// shared expert and 256 routed ones. The second is mrope
    /// (`mrope_section [11, 11, 10]`), which on TEXT positions reduces to
    /// the `rope_neox_subdim` already here -- a claim to verify against
    /// the reference, not to assume.
    ///
    /// **A SEPARATE VARIANT DESPITE SHARING A FLOW, and the precedent is
    /// [`ModelFamily::Qwen3Moe`] rather than [`ModelFamily::Llama`].**
    /// `qwen3moe` shares `families/llama/`'s flow ENTIRELY and is still
    /// its own variant, because its architecture string differs. `llama`
    /// covers a dense and an MoE half under one variant only because
    /// Mixtral and Mistral report the SAME string. Strings decide the
    /// variant; flows are shared separately.
    QwenGdnDense,
}

impl ModelFamily {
    /// Returns static string identifier for the model family.
    ///
    /// **THESE STRINGS ARE AN ON-DISK FORMAT AND TWO OF THEM NO LONGER MATCH
    /// THEIR VARIANT'S NAME. That is deliberate.** Every `.gturbo` install
    /// records this value in `manifest.json`, and [`ModelFamily::parse`]
    /// reads it back at load, so a string here is a compatibility promise to
    /// artifacts already on disk -- not a label to keep tidy.
    ///
    /// `QwenGdnMoe` therefore still writes `"qwen36"` and `QwenGdnDense`
    /// still writes `"qwen35"`, the version-shaped names both variants were
    /// called before 2026-08-15. Changing either would make every existing
    /// install of those families unloadable (`parse` returns `None`, and the
    /// open path reports an unknown family) for a cosmetic gain. The Rust
    /// names carry the meaning; these carry the history.
    ///
    /// A NEW family is free to pick a matching string, because nothing has
    /// been written with it yet.
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelFamily::Gemma4 => "gemma4",
            ModelFamily::QwenGdnMoe => "qwen36",
            ModelFamily::DeepseekV4Flash => "deepseekV4Flash",
            ModelFamily::Llama => "llama",
            ModelFamily::Qwen3Moe => "qwen3moe",
            ModelFamily::GptOss => "gptOss",
            ModelFamily::QwenGdnDense => "qwen35",
        }
    }

    /// Parses string identifier into model family.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "gemma4" => Some(ModelFamily::Gemma4),
            "qwen36" => Some(ModelFamily::QwenGdnMoe),
            "deepseekV4Flash" => Some(ModelFamily::DeepseekV4Flash),
            "llama" => Some(ModelFamily::Llama),
            "qwen3moe" => Some(ModelFamily::Qwen3Moe),
            "gptOss" => Some(ModelFamily::GptOss),
            "qwen35" => Some(ModelFamily::QwenGdnDense),
            _ => None,
        }
    }
}

/// YaRN rope scaling, as `gpt-oss` declares it (ROADMAP M5). Zeroed for
/// architectures that scale nothing, which is every other family here.
///
/// A grouped struct rather than four flat fields for the reason
/// [`LinearAttentionConfig`] is one: the four values are meaningless apart,
/// and one `NONE` in fifteen `ArchConfig` literals is less to get wrong than
/// four zeros in each.
///
/// THE VALUES ARE READ FROM THE FILE, NOT ASSUMED. `gpt-oss` publishes
/// `rope.scaling.{factor, original_context_length, yarn_beta_fast,
/// yarn_beta_slow}`, so this is a metadata path rather than a baseline
/// constant -- unlike the clamped SwiGLU's `alpha`, which llama.cpp
/// hardcodes and which therefore lives in the baseline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeScalingConfig {
    /// YaRN's `factor`. The interpolation scale is its reciprocal. ZERO
    /// means no scaling at all, which is what makes this struct's `NONE`
    /// unambiguous -- 1.0 would read as "declared, and the identity".
    pub factor: f64,
    /// The context length the checkpoint was trained at, which is what the
    /// correction dims are computed against.
    pub original_context: i64,
    /// YaRN's `beta_fast`, the high-frequency end of the correction ramp.
    pub beta_fast: f64,
    /// YaRN's `beta_slow`, the low-frequency end.
    pub beta_slow: f64,
}

impl RopeScalingConfig {
    /// No rope scaling, for the four families that declare none.
    pub const NONE: RopeScalingConfig = RopeScalingConfig {
        factor: 0.0,
        original_context: 0,
        beta_fast: 0.0,
        beta_slow: 0.0,
    };

    /// Whether YaRN applies at all.
    pub fn is_active(&self) -> bool {
        self.factor > 0.0
    }
}

/// Gated-DeltaNet (linear attention) dimensions. Zeroed for architectures
/// without linear-attention layers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearAttentionConfig {
    /// Number of key heads in linear attention.
    pub num_k_heads: i64,
    /// Number of value heads in linear attention.
    pub num_v_heads: i64,
    /// Key head dimension.
    pub key_head_dim: i64,
    /// Value head dimension.
    pub value_head_dim: i64,
    /// Depthwise convolution kernel size.
    pub conv_kernel_size: i64,
}

impl LinearAttentionConfig {
    /// Empty linear attention configuration for non-linear architectures.
    pub const NONE: LinearAttentionConfig = LinearAttentionConfig {
        num_k_heads: 0,
        num_v_heads: 0,
        key_head_dim: 0,
        value_head_dim: 0,
        conv_kernel_size: 0,
    };

    /// Fused qkv projection rows: 2 * K-dim + V-dim. Also the depthwise conv
    /// channel count.
    pub fn qkv_dim(&self) -> i64 {
        2 * self.num_k_heads * self.key_head_dim + self.num_v_heads * self.value_head_dim
    }

    /// Value dim, also the z-gate projection rows and out_proj columns.
    pub fn value_dim(&self) -> i64 {
        self.num_v_heads * self.value_head_dim
    }
}

/// DeepSeek-V4 compressed-attention dimensions. Zeroed for architectures
/// without CSA/HCA layers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressedAttentionConfig {
    /// Rank for Q LoRA projection.
    pub q_lora_rank: i64,
    /// Rank for Out LoRA projection.
    pub o_lora_rank: i64,
    /// Output projection groups.
    pub o_groups: i64,
    /// RoPE head dimension.
    pub rope_head_dim: i64,
    /// Index attention head count.
    pub index_n_heads: i64,
    /// Index head dimension.
    pub index_head_dim: i64,
    /// Index top-k selection count.
    pub index_top_k: i64,
    /// Compression rate for Compressed Sparse Attention (CSA).
    pub csa_compress_rate: i64,
    /// Compression rate for Heavily Compressed Attention (HCA).
    pub hca_compress_rate: i64,
    /// RoPE theta scaling base for compressed layers.
    pub compress_rope_theta: f64,
    /// RoPE scaling factor.
    pub rope_scaling_factor: f64,
    /// Original maximum sequence length for RoPE scaling.
    pub rope_scaling_original_max: i64,
    /// Beta fast parameter for YaRN RoPE scaling.
    pub rope_scaling_beta_fast: f64,
    /// Beta slow parameter for YaRN RoPE scaling.
    pub rope_scaling_beta_slow: f64,
}

impl CompressedAttentionConfig {
    /// Empty compressed attention configuration.
    pub const NONE: CompressedAttentionConfig = CompressedAttentionConfig {
        q_lora_rank: 0,
        o_lora_rank: 0,
        o_groups: 0,
        rope_head_dim: 0,
        index_n_heads: 0,
        index_head_dim: 0,
        index_top_k: 0,
        csa_compress_rate: 0,
        hca_compress_rate: 0,
        compress_rope_theta: 0.0,
        rope_scaling_factor: 0.0,
        rope_scaling_original_max: 0,
        rope_scaling_beta_fast: 0.0,
        rope_scaling_beta_slow: 0.0,
    };
}

/// Manifold-Constrained Hyper-Connection (mHC) residual dimensions. Zeroed
/// for architectures with a plain single-stream residual.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperConnectionConfig {
    /// Multiplier dimension.
    pub mult: i64,
    /// Number of Sinkhorn iterations.
    pub sinkhorn_iters: i64,
    /// Epsilon value for Sinkhorn normalization.
    pub eps: f64,
}

impl HyperConnectionConfig {
    /// Empty hyper connection configuration.
    pub const NONE: HyperConnectionConfig = HyperConnectionConfig {
        mult: 0,
        sinkhorn_iters: 0,
        eps: 0.0,
    };
}

/// `full_attention_layer_mask` values: 0 = sliding-window attention, 1 =
/// full attention, 2 = gated-DeltaNet linear attention, 3 = compressed
/// sparse attention (CSA), 4 = heavily compressed attention (HCA).
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
    /// Hash-routed MoE layer: expert selection is `tid2eid[token]`.
    pub fn layer_is_hash_routed(&self, layer: usize) -> bool {
        (layer as i64) < self.num_hash_routed_layers
    }
}
