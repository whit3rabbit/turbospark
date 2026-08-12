//! The canonical architecture baselines, checked against an installed
//! model's manifest at load time. Ported from the `ArchConfig` static
//! members in `Infrastructure/ModelIO/ModelTypes.swift`.

use crate::arch_config::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily, RopeScalingConfig,
};

fn gemma4_layer_mask() -> Vec<u8> {
    let mut mask = vec![0u8; 30];
    let mut i = 5;
    while i < 30 {
        mask[i] = 1;
        i += 6;
    }
    mask
}

/// Canonical Gemma 4 26B-A4B baseline.
/// `intermediate_size = 2112` is the shared-expert FFN width (3x moe).
pub fn gemma4_26b_a4b() -> ArchConfig {
    ArchConfig {
        hidden_size: 2816,
        intermediate_size: 2112,
        moe_intermediate_size: 704,
        num_heads: 16,
        num_kv_heads: 8,
        num_full_kv_heads: 2,
        head_dim: 256,
        full_head_dim: 512,
        vocab_size: 262_144,
        sliding_window: 1024,
        final_logit_softcap: 30.0,
        rope_theta: 10_000.0,
        full_rope_theta: 1_000_000.0,
        partial_rotary_factor: 0.25,
        num_layers: 30,
        num_experts: 128,
        top_k_experts: 8,
        tie_word_embeddings: true,
        attention_k_eq_v: true,
        full_attention_layer_mask: gemma4_layer_mask(),
        hidden_activation: "gelu_pytorch_tanh".to_string(),
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
        rope_scaling: RopeScalingConfig::NONE,
    }
}

fn qwen36_layer_mask() -> Vec<u8> {
    // Layer kinds: 2 = gated-DeltaNet linear, 1 = full attention on every
    // 4th layer ((i + 1) % 4 == 0).
    let mut mask = vec![2u8; 40];
    let mut i = 3;
    while i < 40 {
        mask[i] = 1;
        i += 4;
    }
    mask
}

/// Canonical Qwen3.6-35B-A3B baseline: a 40-layer hybrid of 30
/// gated-DeltaNet linear-attention layers and 10 full-attention layers
/// (every 4th layer), 256 routed experts (top-8) plus a sigmoid-gated
/// shared expert, SwiGLU activations, untied lm_head, no logit softcap.
pub fn qwen36_35b_a3b() -> ArchConfig {
    ArchConfig {
        hidden_size: 2048,
        intermediate_size: 512,
        moe_intermediate_size: 512,
        num_heads: 16,
        num_kv_heads: 2,
        num_full_kv_heads: 2,
        head_dim: 256,
        full_head_dim: 256,
        vocab_size: 248_320,
        sliding_window: 0,
        final_logit_softcap: 0.0,
        rope_theta: 10_000_000.0,
        full_rope_theta: 10_000_000.0,
        partial_rotary_factor: 0.25,
        num_layers: 40,
        num_experts: 256,
        top_k_experts: 8,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: qwen36_layer_mask(),
        hidden_activation: "silu".to_string(),
        family: ModelFamily::Qwen36,
        attn_output_gate: true,
        attention_scale: 0.0625, // 256^-0.5
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: true,
        rope_neox_subdim: true,
        linear_attention: LinearAttentionConfig {
            num_k_heads: 16,
            num_v_heads: 32,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel_size: 4,
        },
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: RopeScalingConfig::NONE,
    }
}

/// Canonical Mixtral-8x7B-Instruct baseline (ROADMAP Phase M2): 32 dense
/// full-attention layers with GQA (32 query heads over 8 KV heads), 8 routed
/// experts at top-2 and no shared expert, SwiGLU, untied lm_head, no logit
/// softcap and no sliding window.
///
/// **This baseline is Mixtral's, and the family covers dense Llama too.**
/// `general.architecture = "llama"` is what both report -- a dense Llama 3.1
/// differs from this only in shape fields (`num_experts = 0`, its own vocab
/// and thetas), which `arch_from_gguf` reads off the file. Every BEHAVIOURAL
/// field below is shared, which is what makes one baseline honest for both;
/// see `gguf_config.rs`'s module header for why that split matters.
///
/// `intermediate_size` equals `moe_intermediate_size` because the file
/// publishes one `feed_forward_length` and has no shared expert to size
/// separately.
pub fn mixtral_8x7b() -> ArchConfig {
    ArchConfig {
        hidden_size: 4096,
        intermediate_size: 14_336,
        moe_intermediate_size: 14_336,
        num_heads: 32,
        num_kv_heads: 8,
        num_full_kv_heads: 8,
        head_dim: 128,
        full_head_dim: 128,
        vocab_size: 32_000,
        sliding_window: 0,
        final_logit_softcap: 0.0,
        rope_theta: 1_000_000.0,
        full_rope_theta: 1_000_000.0,
        // Full rotary: `rope.dimension_count` is 128, the whole head.
        partial_rotary_factor: 1.0,
        num_layers: 32,
        num_experts: 8,
        top_k_experts: 2,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: vec![1u8; 32],
        hidden_activation: "silu".to_string(),
        family: ModelFamily::Llama,
        attn_output_gate: false,
        // 128^-0.5 = 2^-3.5. NOT a binary fraction, unlike Gemma's 1.0 and
        // Qwen's 0.0625, so AGENTS.md Gotcha 24's round-trip warning applies
        // to this field and `crates/model-io/tests/arch_config.rs` pins it.
        attention_scale: 0.088_388_347_648_318_45,
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: false,
        rope_neox_subdim: false,
        linear_attention: LinearAttentionConfig::NONE,
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: RopeScalingConfig::NONE,
    }
}

/// Canonical Qwen3-30B-A3B baseline (ROADMAP Phase M, the fine-grained MoE
/// follow-on): 48 full-attention layers, 32 query heads over 4 KV heads at
/// head_dim 128, 128 routed experts at top-8 with NO shared expert, SwiGLU,
/// untied lm_head, no logit softcap, no sliding window.
///
/// **Every number here was read off the published
/// `Qwen/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf` header**, not from a
/// model card (`gguf_checkpoint_network.rs::scopes_qwen3moe_*`).
///
/// `head_dim` is 128 while `hidden_size / num_heads` is 64, so the query
/// projection emits 4096 rows against a 2048-wide stream. That is the usual
/// case rather than the exception (`docs/NEW_MODEL.md` Phase 0 says so), and
/// deriving the head dim from the hidden size would halve every GEMV here.
///
/// `intermediate_size` is the published `feed_forward_length`, which this
/// checkpoint carries and never uses: every layer is MoE and there is no
/// dense or shared FFN tensor in the file.
///
/// THE FIELD THAT MADE THIS FAMILY WORTH BRINGING UP is
/// `moe_intermediate_size = 768`: one expert is `3 x 768 x 2048` at Q4_K,
/// i.e. 2.5 MiB, so 16 slots over 48 layers pin 1.90 GiB. Mixtral's same
/// arithmetic reads 108.9 MiB and 54.5 GiB (AGENTS.md Gotcha 36).
pub fn qwen3_30b_a3b() -> ArchConfig {
    ArchConfig {
        hidden_size: 2048,
        intermediate_size: 6144,
        moe_intermediate_size: 768,
        num_heads: 32,
        num_kv_heads: 4,
        num_full_kv_heads: 4,
        head_dim: 128,
        full_head_dim: 128,
        vocab_size: 151_936,
        sliding_window: 0,
        final_logit_softcap: 0.0,
        rope_theta: 1_000_000.0,
        full_rope_theta: 1_000_000.0,
        // Full rotary: the whole head is rotated, as on the `llama` side.
        partial_rotary_factor: 1.0,
        num_layers: 48,
        num_experts: 128,
        top_k_experts: 8,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: vec![1u8; 48],
        hidden_activation: "silu".to_string(),
        family: ModelFamily::Qwen3Moe,
        attn_output_gate: false,
        // 128^-0.5 = 2^-3.5, the same non-binary-fraction value Mixtral's
        // head dim produces; AGENTS.md Gotcha 24's round-trip warning
        // applies and `crates/model-io/tests/arch_config.rs` pins it.
        attention_scale: 0.088_388_347_648_318_45,
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: false,
        rope_neox_subdim: false,
        linear_attention: LinearAttentionConfig::NONE,
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: RopeScalingConfig::NONE,
    }
}

/// `gpt-oss`'s alternating window, in this port's polarity (0 = sliding,
/// 1 = full attention).
///
/// EVEN LAYERS SLIDE, and that is read rather than guessed: the file
/// declares `attention.sliding_window = 128` and NO
/// `sliding_window_pattern`, so llama.cpp's default period of 2 applies and
/// `llama_hparams::set_swa_pattern` with `dense_first = false` computes
/// `is_swa[il] = (il % 2) < 1`. Getting the phase backwards gives a model
/// that is wrong only past 128 tokens of context, which no short smoke
/// reaches.
fn gpt_oss_layer_mask(layers: usize) -> Vec<u8> {
    (0..layers).map(|i| u8::from(i % 2 != 0)).collect()
}

/// Canonical `gpt-oss-20b` baseline (ROADMAP M5). 24 layers, hidden 2880,
/// 64 query heads over 8 KV heads at head_dim 64, 32 routed experts at top-4
/// with no shared expert, untied lm_head, no logit softcap, and an
/// alternating 128-token sliding window.
///
/// **Every shape here was read off the published
/// `ggml-org/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf` header**
/// (`gguf_checkpoint_network.rs::scopes_phase_m5_gpt_oss_layer`), and every
/// BEHAVIOURAL field off llama.cpp's `src/models/openai-moe.cpp`, which is
/// what wrote those bytes.
///
/// TWO FIELDS ARE NOT METADATA AND THAT ASYMMETRY IS THE POINT.
/// `swiglu_limit` is 7.0 and its companion alpha is 1.702, both HARDCODED in
/// llama.cpp's graph builder with its own "TODO: move to hparams?" beside
/// them, so they are baseline constants here and a file cannot override
/// them. The YaRN scalars in `rope_scaling` ARE published, so those are read.
/// Storing either the other way round would be a guess wearing the other's
/// clothes.
pub fn gpt_oss_20b() -> ArchConfig {
    ArchConfig {
        hidden_size: 2880,
        // One `feed_forward_length`, and `expert_feed_forward_length` equals
        // it. There is no shared expert to size separately.
        intermediate_size: 2880,
        moe_intermediate_size: 2880,
        num_heads: 64,
        num_kv_heads: 8,
        num_full_kv_heads: 8,
        head_dim: 64,
        full_head_dim: 64,
        vocab_size: 201_088,
        sliding_window: 128,
        final_logit_softcap: 0.0,
        // One base for both halves of the window: the file publishes
        // `rope.freq_base` and NO `rope.freq_base_swa`, unlike Gemma, which
        // is why these two agree.
        rope_theta: 150_000.0,
        full_rope_theta: 150_000.0,
        partial_rotary_factor: 1.0,
        num_layers: 24,
        num_experts: 32,
        top_k_experts: 4,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: gpt_oss_layer_mask(24),
        hidden_activation: "silu".to_string(),
        family: ModelFamily::GptOss,
        attn_output_gate: false,
        // `1/sqrt(64)` = 0.125 exactly, from llama.cpp's own
        // `1.0f/sqrtf(float(n_rot))`. A binary fraction, so unlike Mixtral's
        // and Qwen3-MoE's it survives AGENTS.md Gotcha 24's round-trip
        // without care.
        attention_scale: 0.125,
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: false,
        rope_neox_subdim: false,
        linear_attention: LinearAttentionConfig::NONE,
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        // llama.cpp's `SOFTMAX_WEIGHT`: top-k on the RAW logits, then a
        // softmax over the selected k alone and no renormalization. That is
        // `router_topk_gemma4`'s existing contract minus the per-expert
        // scale, so the spelling here is the same "softmax" the others use.
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 7.0,
        rope_scaling: RopeScalingConfig {
            factor: 32.0,
            original_context: 4096,
            beta_fast: 32.0,
            beta_slow: 1.0,
        },
    }
}

fn deepseek_v4_flash_layer_mask() -> Vec<u8> {
    // Layer kinds: 0 = sliding-window only (layers 0-1), then 3 = CSA on
    // even layers and 4 = HCA on odd layers.
    let mut mask = vec![0u8; 43];
    for (i, slot) in mask.iter_mut().enumerate().take(43).skip(2) {
        *slot = if i % 2 == 0 { 3 } else { 4 };
    }
    mask
}

/// Canonical DeepSeek-V4-Flash 284B-A13B baseline: 43 all-MoE layers,
/// shared-KV MQA attention, sliding window 128 on every layer, and
/// compressed long-range KV (CSA/HCA). The residual is 4 mHC streams.
/// Untied lm_head, no logit softcap.
pub fn deepseek_v4_flash_284b_a13b() -> ArchConfig {
    ArchConfig {
        hidden_size: 4096,
        intermediate_size: 2048,
        moe_intermediate_size: 2048,
        num_heads: 64,
        num_kv_heads: 1,
        num_full_kv_heads: 1,
        head_dim: 512,
        full_head_dim: 512,
        vocab_size: 129_280,
        sliding_window: 128,
        final_logit_softcap: 0.0,
        rope_theta: 10_000.0,
        full_rope_theta: 10_000.0,
        partial_rotary_factor: 0.125,
        num_layers: 43,
        num_experts: 256,
        top_k_experts: 6,
        tie_word_embeddings: false,
        attention_k_eq_v: true,
        full_attention_layer_mask: deepseek_v4_flash_layer_mask(),
        hidden_activation: "silu".to_string(),
        family: ModelFamily::DeepseekV4Flash,
        attn_output_gate: false,
        attention_scale: 0.044_194_173_824_159_216, // 512^-0.5
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: false,
        rope_neox_subdim: false,
        linear_attention: LinearAttentionConfig::NONE,
        compressed_attention: CompressedAttentionConfig {
            q_lora_rank: 1024,
            o_lora_rank: 1024,
            o_groups: 8,
            rope_head_dim: 64,
            index_n_heads: 64,
            index_head_dim: 128,
            index_top_k: 512,
            csa_compress_rate: 4,
            hca_compress_rate: 128,
            compress_rope_theta: 160_000.0,
            rope_scaling_factor: 16.0,
            rope_scaling_original_max: 65_536,
            rope_scaling_beta_fast: 32.0,
            rope_scaling_beta_slow: 1.0,
        },
        hyper_connections: HyperConnectionConfig {
            mult: 4,
            sinkhorn_iters: 20,
            eps: 1.0e-6,
        },
        num_hash_routed_layers: 3,
        router_scoring_func: "sqrtsoftplus".to_string(),
        routed_scaling_factor: 1.5,
        swiglu_limit: 10.0,
        rope_scaling: RopeScalingConfig::NONE,
    }
}

/// Registry keyed by `manifest.arch.family` for auto-detection at load.
pub fn known_architecture(family: ModelFamily) -> ArchConfig {
    match family {
        ModelFamily::Gemma4 => gemma4_26b_a4b(),
        ModelFamily::Qwen36 => qwen36_35b_a3b(),
        ModelFamily::DeepseekV4Flash => deepseek_v4_flash_284b_a13b(),
        ModelFamily::Llama => mixtral_8x7b(),
        ModelFamily::Qwen3Moe => qwen3_30b_a3b(),
        ModelFamily::GptOss => gpt_oss_20b(),
    }
}

pub fn all_known_architectures() -> Vec<ArchConfig> {
    vec![
        gemma4_26b_a4b(),
        qwen36_35b_a3b(),
        deepseek_v4_flash_284b_a13b(),
        mixtral_8x7b(),
        qwen3_30b_a3b(),
        gpt_oss_20b(),
    ]
}
