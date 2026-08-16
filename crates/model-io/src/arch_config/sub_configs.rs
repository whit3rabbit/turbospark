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
