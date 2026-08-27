use crate::arch_config::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily, RopeScalingConfig, VisionConfig,
};

fn qwen_gdn_moe_layer_mask() -> Vec<u8> {
    qwen_hybrid_layer_mask(40)
}

/// The Qwen hybrid layer mask: gated-DeltaNet linear everywhere except
/// every 4th layer, which is full attention.
///
/// Layer kinds: 2 = gated-DeltaNet linear, 1 = full attention on every 4th
/// layer (`(i + 1) % 4 == 0`). Shared by Qwen 3.6 at 40 layers and
/// `qwen3_5` at 64, which both declare `full_attention_interval: 4` and
/// whose `layer_types` lists reproduce exactly this pattern.
fn qwen_hybrid_layer_mask(layers: usize) -> Vec<u8> {
    let mut mask = vec![2u8; layers];
    let mut i = 3;
    while i < layers {
        mask[i] = 1;
        i += 4;
    }
    mask
}

/// Canonical `qwen3_5` baseline: a 64-layer hybrid of 48 gated-DeltaNet
/// linear-attention layers and 16 full-attention layers (every 4th), a DENSE
/// SwiGLU FFN, untied lm_head, no logit softcap and no sliding window.
///
/// **It is named for the ARCHITECTURE rather than for a checkpoint because it
/// serves two published ones**, which is what the rename off `bonsai_27b`
/// records. `prism-ml/Bonsai-27B-mlx-1bit` (ROADMAP's 1-bit entry) came
/// first; `Qwen/Qwen3.8-27B` arrived 2026-08-14 and its `text_config` agrees
/// with Bonsai's on 33 of 35 fields -- everything below is identical, and the
/// only differences are `eos_token_id` (248044 against 248046, a tokenizer
/// concern that reaches no field here) and the quantization block, which is
/// not part of an `ArchConfig` either. The two therefore share this baseline
/// exactly, and the QUANTIZATION is what a reader should expect to differ
/// between installs of them: 1-bit at group 128 for Bonsai, INT4 at group 64
/// for the mlx-community Qwen3.8 artifact.
///
/// **Every behavioural field here is Qwen 3.6's and every shape field
/// differs**, which is what makes this the same relationship dense Mistral
/// has to Mixtral. All of it is read off the checkpoint's own `config.json`
/// (`model_type: qwen3_5`, `text_config.model_type: qwen3_5_text`) rather
/// than inferred from the sibling: `head_dim` 256, `vocab_size` 248320,
/// `rope_theta` 1e7, `partial_rotary_factor` 0.25, `attn_output_gate`, the
/// linear key/value head dims and the conv kernel all coincide with Qwen
/// 3.6 as published, while hidden is 5120 against 2048, layers 64 against
/// 40, and there are no experts at all.
///
/// Two fields are worth reading twice. `intermediate_size` is the DENSE FFN
/// width (17408) where Qwen 3.6's is its shared expert's 512, so the same
/// field name means a different thing in the two baselines -- the dense
/// `llama` half has exactly this collision. And `rope_scaling` is `NONE`
/// even though the checkpoint declares `mrope_section [11, 11, 10]`:
/// [`RopeScalingConfig`] carries YARN's four scalars, and mrope is not
/// YaRN. On TEXT positions mrope's three sections are equal and it reduces
/// to the `rope_neox_subdim` this baseline already sets, which is a claim
/// the cross-engine check has to verify rather than one to assume.
pub fn qwen_gdn_dense_27b() -> ArchConfig {
    ArchConfig {
        hidden_size: 5120,
        // The dense FFN, not a shared expert. See the doc above.
        intermediate_size: 17408,
        // No routed experts, so no routed width.
        moe_intermediate_size: 0,
        num_heads: 24,
        num_kv_heads: 4,
        num_full_kv_heads: 4,
        head_dim: 256,
        full_head_dim: 256,
        vocab_size: 248_320,
        sliding_window: 0,
        final_logit_softcap: 0.0,
        rope_theta: 10_000_000.0,
        full_rope_theta: 10_000_000.0,
        partial_rotary_factor: 0.25,
        num_layers: 64,
        num_experts: 0,
        top_k_experts: 0,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: qwen_hybrid_layer_mask(64),
        hidden_activation: "silu".to_string(),
        family: ModelFamily::QwenGdnDense,
        attn_output_gate: true,
        attention_scale: 0.0625, // 256^-0.5, a binary fraction (Gotcha 24)
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        // No shared expert to gate: the FFN is dense.
        shared_expert_gated: false,
        rope_neox_subdim: true,
        linear_attention: LinearAttentionConfig {
            num_k_heads: 16,
            // 48, against Qwen 3.6's 32: three V heads per K head rather
            // than two.
            num_v_heads: 48,
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
        vision: VisionConfig::NONE,
    }
}

/// Canonical Qwen3.6-35B-A3B baseline: a 40-layer hybrid of 30
/// gated-DeltaNet linear-attention layers and 10 full-attention layers
/// (every 4th layer), 256 routed experts (top-8) plus a sigmoid-gated
/// shared expert, SwiGLU activations, untied lm_head, no logit softcap.
pub fn qwen_gdn_moe_35b_a3b() -> ArchConfig {
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
        full_attention_layer_mask: qwen_gdn_moe_layer_mask(),
        hidden_activation: "silu".to_string(),
        family: ModelFamily::QwenGdnMoe,
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
        vision: VisionConfig::NONE,
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
        vision: VisionConfig::NONE,
    }
}
