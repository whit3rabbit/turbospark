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
    /// The activation the gated output norm applies to `z`: sigmoid when true,
    /// silu when false. The checkpoint's `output_gate_type`.
    ///
    /// **NOT COSMETIC, AND FALSE IS THE PRE-`qwen4_exp` BEHAVIOUR.** Every
    /// family that reached `gdn_gated_norm` before `qwen4_exp` declares silu,
    /// which is why the kernel hardcoded it (`crates/gpu` Gotcha 12). Reading
    /// the wrong one is a one-character difference that produces fluent WRONG
    /// output rather than an error, so it is a field rather than a family
    /// constant: AGENTS.md Gotcha 61's shape, which is exactly a per-family
    /// property left standing in shared code.
    pub output_gate_sigmoid: bool,
}

impl LinearAttentionConfig {
    /// Empty linear attention configuration for non-linear architectures.
    pub const NONE: LinearAttentionConfig = LinearAttentionConfig {
        num_k_heads: 0,
        num_v_heads: 0,
        key_head_dim: 0,
        value_head_dim: 0,
        conv_kernel_size: 0,
        output_gate_sigmoid: false,
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

/// The `qwen3_5` vision tower's dimensions (ROADMAP M-V3). Zeroed for every
/// architecture without one, which today is every architecture this port
/// DECODES -- the tower is ingested before any flow can run it, so `NONE` is
/// what all fifteen existing baselines carry and the feature is off unless a
/// checkpoint's `vision_config` put something here.
///
/// A grouped struct with a `NONE`, for [`LinearAttentionConfig`]'s reason: the
/// values are meaningless apart, and one `NONE` per `ArchConfig` literal is
/// less to get wrong than a dozen zeros in each.
///
/// **EVERY FIELD IS READ OFF THE CHECKPOINT'S `vision_config`, none is a
/// baseline constant.** Both published `qwen3_5` checkpoints
/// (`Qwen/Qwen3.8-27B`, `prism-ml/Bonsai-27B-mlx-1bit`) declare an identical
/// block, so a value here that disagrees with the file is a parse bug rather
/// than a variant. `docs/VISION_PHASE0.md` item 1 has the inventory this was
/// read from; note `intermediate_size` is **4304** and not the 4608 the
/// planning table guessed by analogy with the merger's hidden width.
///
/// There are NO float fields on purpose. `arch_validation` compares manifest
/// floats with `!=` against a serde_json parser accurate to ~1 ULP, so a
/// non-binary-fraction float here could not round-trip (AGENTS.md Gotcha 24).
/// Everything the tower needs is an integer count or a token id.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisionConfig {
    /// Transformer blocks in the tower. ZERO is the inactive sentinel and is
    /// unambiguous: a real tower cannot have none.
    pub depth: i64,
    /// The tower's own residual width, unrelated to the text trunk's.
    pub hidden_size: i64,
    /// Per-block MLP width (`mlp.linear_fc1` rows).
    pub intermediate_size: i64,
    /// Attention heads. `head_dim` is `hidden_size / num_heads` and is
    /// derived rather than stored, because the checkpoint states neither and
    /// no consumer may guess a third value (AGENTS.md Gotcha 39).
    pub num_heads: i64,
    /// Spatial patch edge in pixels.
    pub patch_size: i64,
    /// Frames per patch. 2 on this family, which is why a still image is
    /// fed as a duplicated pair.
    pub temporal_patch_size: i64,
    /// Input image channels.
    pub in_channels: i64,
    /// Patch merge edge. `spatial_merge_size ^ 2` patches become one text
    /// token, so 2 means 4 patches per token.
    pub spatial_merge_size: i64,
    /// Rows in the learned position-embedding table, a square grid
    /// (2304 = 48x48) bilinearly interpolated to the page's own grid.
    pub num_position_embeddings: i64,
    /// The merger's output width, which equals the TEXT trunk's
    /// `hidden_size` -- the merger writes straight into the residual stream.
    pub out_hidden_size: i64,
    /// mRoPE's `(t, h, w)` channel split, summing to `rotary_dim / 2`.
    /// Read from `text_config.rope_parameters.mrope_section` rather than
    /// from the vision block; it describes how the TRUNK consumes image
    /// positions, and it is here because nothing else carries it.
    pub mrope_section: [i64; 3],
    /// `<|vision_start|>`.
    pub vision_start_token_id: i64,
    /// `<|vision_end|>`.
    pub vision_end_token_id: i64,
    /// `<|image_pad|>`, the token whose positions the merger's rows replace.
    pub image_token_id: i64,
    /// `<|video_pad|>`. Carried for completeness; no video path exists here.
    pub video_token_id: i64,
}

impl VisionConfig {
    /// No vision tower, for every architecture that declares none.
    pub const NONE: VisionConfig = VisionConfig {
        depth: 0,
        hidden_size: 0,
        intermediate_size: 0,
        num_heads: 0,
        patch_size: 0,
        temporal_patch_size: 0,
        in_channels: 0,
        spatial_merge_size: 0,
        num_position_embeddings: 0,
        out_hidden_size: 0,
        mrope_section: [0, 0, 0],
        vision_start_token_id: 0,
        vision_end_token_id: 0,
        image_token_id: 0,
        video_token_id: 0,
    };

    /// Whether this install carries a tower at all. Every vision-conditional
    /// path keys on this rather than on the family, because the family says
    /// which ARCHITECTURE a checkpoint is and this says whether the publisher
    /// shipped the tower with it -- `mlx-community`'s conversions have
    /// dropped a component before (`mtp.*`), so the two questions are
    /// genuinely different.
    pub fn is_active(&self) -> bool {
        self.depth > 0
    }

    /// Per-head attention width. Derived, never stored: the checkpoint's
    /// `vision_config` declares no `head_dim`, so storing one would be this
    /// port inventing a third source for a value with two.
    pub fn head_dim(&self) -> i64 {
        if self.num_heads == 0 {
            0
        } else {
            self.hidden_size / self.num_heads
        }
    }

    /// Patches per merged output token (`spatial_merge_size ^ 2`).
    pub fn patches_per_token(&self) -> i64 {
        self.spatial_merge_size * self.spatial_merge_size
    }

    /// The merger's input width: `patches_per_token` patch rows concatenated.
    pub fn merger_input_dim(&self) -> i64 {
        self.patches_per_token() * self.hidden_size
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
    /// Index key/value head count. `qwen4_exp`'s `indexer_kv_heads`; 0 for
    /// DeepSeek-V4-Flash, whose indexer declares no separate KV head count.
    pub index_kv_heads: i64,
    /// Index head dimension.
    pub index_head_dim: i64,
    /// Index top-k selection count, in the unit the architecture selects in:
    /// TOKENS for DeepSeek-V4-Flash, BLOCKS for `qwen4_exp` (which derives it
    /// as `index_budget / csa_compress_rate`).
    pub index_top_k: i64,
    /// Context length below which the indexer selects nothing and attention is
    /// exactly causal. `qwen4_exp`'s `indexer_budget`; 0 where no such
    /// threshold exists. Read it through [`Self::sparse_below`].
    pub index_budget: i64,
    /// Compression rate for Compressed Sparse Attention (CSA). Also
    /// `qwen4_exp`'s `indexer_compress_ratio`, which is the same quantity:
    /// how many tokens pool into one selectable block.
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
        index_kv_heads: 0,
        index_head_dim: 0,
        index_top_k: 0,
        index_budget: 0,
        csa_compress_rate: 0,
        hca_compress_rate: 0,
        compress_rope_theta: 0.0,
        rope_scaling_factor: 0.0,
        rope_scaling_original_max: 0,
        rope_scaling_beta_fast: 0.0,
        rope_scaling_beta_slow: 0.0,
    };

    /// The context length below which a query-sparse indexer selects nothing,
    /// so attention is exactly the plain causal case. 0 when the architecture
    /// declares no indexer.
    ///
    /// `qwen4_exp`'s indexer returns early at `kv_len <= indexer_budget`, which
    /// is what lets an engine without the selector serve short contexts
    /// EXACTLY rather than approximately, and refuse longer ones.
    pub fn sparse_below(&self) -> i64 {
        self.index_budget
    }
}

/// Wide-residual dimensions. Zeroed for architectures with a plain
/// single-stream residual, which is every family here except DeepSeek-V4-Flash
/// and `qwen4_exp`.
///
/// **TWO MECHANISMS SHARE THIS STRUCT AND ONLY `mult` MEANS THE SAME THING IN
/// BOTH.** `mult` is the number of residual streams the hidden state is tiled
/// into, so the stream is `mult * hidden_size` wide for the whole stack. How
/// those streams are MIXED differs: DeepSeek's mHC is Sinkhorn-normalised
/// (`sinkhorn_iters`, `eps`), while `qwen4_exp`'s gated residual is a low-rank
/// silu/sigmoid mix through `lowrank`. A field one mechanism does not use is
/// zero there.
///
/// Which mixing math runs is decided by [`super::family::ModelFamily`] and
/// never by sniffing which fields are non-zero, following `crates/runtime`
/// Gotcha 3. A discriminant field here would be a second, staler copy of a
/// question the family already answers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperConnectionConfig {
    /// Number of residual streams. The stream is `mult * hidden_size` wide.
    pub mult: i64,
    /// Number of Sinkhorn iterations. DeepSeek mHC only; 0 for `qwen4_exp`.
    pub sinkhorn_iters: i64,
    /// Epsilon value for Sinkhorn normalization. DeepSeek mHC only.
    pub eps: f64,
    /// Rank of the stream-mixing bottleneck (`hc_lowrank`). `qwen4_exp` only;
    /// 0 for DeepSeek mHC, whose mix carries no bottleneck.
    pub lowrank: i64,
}

impl HyperConnectionConfig {
    /// Empty hyper connection configuration.
    pub const NONE: HyperConnectionConfig = HyperConnectionConfig {
        mult: 0,
        sinkhorn_iters: 0,
        eps: 0.0,
        lowrank: 0,
    };

    /// True when the residual is more than one stream wide.
    pub fn is_active(&self) -> bool {
        self.mult > 1
    }
}

/// The hashed n-gram per-layer-embedding (PLE) table, or [`PleConfig::NONE`].
///
/// A grouped struct with a `NONE`, for [`LinearAttentionConfig`]'s reason: the
/// values are meaningless apart, and one `NONE` per `ArchConfig` literal is
/// less to get wrong than nine zeros in each.
///
/// **EVERY FIELD IS A SHAPE SCALAR, AND THE TABLE'S OWN CONTENTS ARE NOT HERE.**
/// The per-head prime vocabulary sizes, their offsets and the hash multipliers
/// are int64 BUFFERS in the checkpoint (`layer_multipliers`,
/// `ngram_heads_vocab_sizes`, `ngram_heads_offsets`), so a flow reads them from
/// the resident index rather than recomputing them. `seed` and
/// `ngram_vocab_size_base` are kept because they are what the reference
/// recomputes those buffers FROM when a checkpoint omits them, which makes them
/// a cross-check on the buffers rather than a second source of truth.
///
/// Integers only, deliberately: `arch_validation` compares manifest floats with
/// `!=` against a parser accurate to ~1 ULP, so a non-binary-fraction float here
/// could not round-trip (AGENTS.md Gotcha 24).
#[derive(Debug, Clone, PartialEq)]
pub struct PleConfig {
    /// Longest n-gram hashed. 3 means bigram and trigram heads.
    pub ngram_size: i64,
    /// Hash heads per n-gram order. Total heads is
    /// `(ngram_size - 1) * heads_per_ngram`.
    pub heads_per_ngram: i64,
    /// Each head's vocabulary is the nth prime after this value, so the heads
    /// are near this size and no two are equal.
    pub ngram_vocab_size_base: i64,
    /// The concatenated row count is padded up to a multiple of this.
    pub make_divisible_by: i64,
    /// How many shards the padded table is split into.
    pub split_ngram_parts: i64,
    /// Width of one embedding row set, split evenly across the hash heads.
    pub ple_embed_dim: i64,
    /// Depthwise convolution kernel size over the gated output.
    pub conv_kernel_size: i64,
    /// ONE-BASED layer ids carrying a PLE block, exactly as the checkpoint
    /// spells them. `[2]` means layer INDEX 1. Kept in the checkpoint's own
    /// convention so a reader comparing against `config.json` sees the same
    /// number, with the conversion done once at the call site.
    pub layer_ids: Vec<i64>,
    /// Seed the hash multipliers derive from when the checkpoint omits them.
    pub seed: i64,
    /// PLE's n-gram context resets at EOS boundaries and pads the start of
    /// a sequence with EOS (`docs/QWEN4_PHASE0.md` item 4). This is
    /// `text_config.eos_token_id`, a SCALAR distinct from
    /// `generation_config.json`'s two-entry stop list -- a decode flow
    /// needs this at `open()`, before any tokenizer is in scope (`crates/runtime`
    /// does not depend on `crates/tokenizer`), so it is read once from the
    /// checkpoint's own config at parse time rather than threaded in from a
    /// caller. The reference's own `validate_architecture` refuses PLE with
    /// no EOS set, which is why this has no meaningful default: `0` here
    /// means "not read from the checkpoint" and a PLE-active config must
    /// override it.
    pub eos_token_id: i64,
}

impl PleConfig {
    /// No PLE table, which is every architecture here except `qwen4_exp`.
    pub const NONE: PleConfig = PleConfig {
        ngram_size: 0,
        heads_per_ngram: 0,
        ngram_vocab_size_base: 0,
        make_divisible_by: 0,
        split_ngram_parts: 0,
        ple_embed_dim: 0,
        conv_kernel_size: 0,
        layer_ids: Vec::new(),
        seed: 0,
        eos_token_id: 0,
    };

    /// True when this install carries an n-gram table.
    pub fn is_active(&self) -> bool {
        self.ngram_size > 0 && !self.layer_ids.is_empty()
    }

    /// Total hash heads: one table per head, each its own prime.
    pub fn ngram_heads(&self) -> i64 {
        (self.ngram_size - 1).max(0) * self.heads_per_ngram
    }

    /// Width of one head's row. The reference divides `ple_embed_dim` by the
    /// head count, so a config whose heads do not divide it evenly is refused
    /// at parse rather than silently truncated here.
    pub fn head_dim(&self) -> i64 {
        let heads = self.ngram_heads();
        if heads == 0 {
            return 0;
        }
        self.ple_embed_dim / heads
    }

    /// ZERO-BASED layer indices, which is what a decode flow loops over.
    /// `layer_ids` is one-based because the checkpoint spells it that way.
    pub fn layer_indices(&self) -> Vec<i64> {
        self.layer_ids.iter().map(|id| id - 1).collect()
    }
}
