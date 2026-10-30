use crate::arch_config::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig, MlaConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

/// Canonical `LFM3-1.7B` baseline: 24 DENSE layers of three kinds -- 13
/// gated short-convolution layers (mask 6), 6 full-attention and 5
/// sliding-window (256) layers, GQA at 16 query heads over 4 KV heads with
/// head_dim 128 -- a SwiGLU FFN on every layer, sandwich norms, and tied
/// embeddings.
///
/// Read off `LiquidAI/LFM3-1.7B`'s `config.json` and `modeling_lfm3.py`
/// (revision `40ebe0aa`), cross-checked against the official GGUF header
/// (`LiquidAI/LFM3-1.7B-GGUF`), never inferred. Facts live in
/// `docs/LFM3_PHASE0.md`.
///
/// **`partial_rotary_factor` (0.5) MEANS THE FULL-ATTENTION LAYERS ONLY**,
/// with the SWA arm at factor 1.0 hard-coded in the flow -- the gemma4 and
/// spark pattern. Unlike spark, the rotation is `rope_proportional_neox`'s:
/// HF zero-tails the inverse frequencies past `0.5 * head_dim / 2` and keeps
/// the head_dim divisor and the `(i, head_dim/2 + i)` pairing, so
/// `rope_neox_subdim` is FALSE. The GGUF says the same thing a different
/// way: `rope.dimension_count` is the whole head (128) and the partial
/// factor lives only in `rope_freqs.weight` (32 ones, then 32 x 1e30).
///
/// **`attn_output_gate` is FALSE even though every attention layer HAS an
/// output gate**, on the spark precedent: `self_attn.gate_proj` is its own
/// `[num_heads, hidden]` tensor, one sigmoid scalar per head, not a half of
/// `q_proj`. The flow applies it unconditionally.
///
/// `ffn_sandwich_norms` is TRUE: `post_operator_norm` normalizes the mixer
/// output (conv or attention) and `post_ffn_norm` the FFN output before each
/// residual add (`use_sandwich_norm`). The final norm is named
/// `embedding_norm` in HF (`output_norm` in GGUF) and runs AFTER the last
/// layer; the embedding itself is used unscaled and unnormed.
///
/// The RMS epsilon (1e-5 for every norm, q/k per-head norms included) rides
/// as a flow constant, following the spark / llama precedent.
pub fn lfm3_1_7b() -> ArchConfig {
    ArchConfig {
        hidden_size: 2048,
        // DENSE width (`w1`/`w3` rows). No experts.
        intermediate_size: 7680,
        moe_intermediate_size: 0,
        num_heads: 16,
        num_kv_heads: 4,
        num_full_kv_heads: 4,
        head_dim: 128,
        full_head_dim: 128,
        vocab_size: 128_000,
        sliding_window: 256,
        final_logit_softcap: 0.0,
        // SWA class: theta 1e4 over the whole head.
        rope_theta: 10_000.0,
        // Full class: theta 1e6 over the leading half of the pairs.
        full_rope_theta: 1_000_000.0,
        partial_rotary_factor: 0.5,
        num_layers: 24,
        dense_lead_intermediate_size: 0,
        num_dense_leading_layers: 0,
        num_experts: 0,
        top_k_experts: 0,
        tie_word_embeddings: true,
        attention_k_eq_v: false,
        full_attention_layer_mask: lfm3_layer_mask(),
        hidden_activation: "silu".to_string(),
        family: ModelFamily::Lfm3,
        attn_output_gate: false,
        // 128^-0.5 = 2^-3.5, the value the other head_dim-128 baselines
        // carry; AGENTS.md Gotcha 24's round-trip warning applies (it holds).
        attention_scale: 0.088_388_347_648_318_45,
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: true,
        shared_expert_gated: false,
        rope_neox_subdim: false,
        linear_attention: LinearAttentionConfig::NONE,
        mla: MlaConfig::NONE,
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: RopeScalingConfig::NONE,
        vision: VisionConfig::NONE,
        // `conv_L_cache` / GGUF `shortconv.l_cache`.
        short_conv_kernel_size: 7,
        ple: PleConfig::NONE,
    }
}

/// `config.layer_types` verbatim, as mask codes: 6 = `conv`, 1 =
/// `full_attention`, 0 = `sliding_attention`. Written out rather than
/// generated because the pattern is not periodic from layer 0 (two leading
/// conv layers, a trailing conv layer), and a formula that reproduced it
/// would be harder to check against the config than the list is.
pub fn lfm3_layer_mask() -> Vec<u8> {
    vec![
        6, 6, 1, 6, 0, 6, 1, 6, 0, 6, 1, 6, 0, 6, 1, 6, 0, 6, 1, 6, 0, 6, 1, 6,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the mask's population and positions against the published
    /// `layer_types`: 13 conv, 6 full at 2,6,...,22, 5 sliding at 4,...,20.
    /// The GGUF's `head_count_kv` (0 on conv layers) and
    /// `sliding_window_pattern` arrays say the same thing.
    #[test]
    fn the_layer_mask_matches_layer_types() {
        let mask = lfm3_layer_mask();
        assert_eq!(mask.len(), 24);
        assert_eq!(mask.iter().filter(|&&m| m == 6).count(), 13);
        let full: Vec<usize> = (0..24).filter(|&i| mask[i] == 1).collect();
        let swa: Vec<usize> = (0..24).filter(|&i| mask[i] == 0).collect();
        assert_eq!(full, vec![2, 6, 10, 14, 18, 22]);
        assert_eq!(swa, vec![4, 8, 12, 16, 20]);
    }

    /// The rotary sub-dimension the full layers rotate must be a positive
    /// even integer (NEW_MODEL.md Phase 1): 0.5 * 128 = 64 dims, 32 pairs.
    #[test]
    fn the_full_layer_rotary_dim_is_even() {
        let arch = lfm3_1_7b();
        let rotary = arch.partial_rotary_factor * arch.full_head_dim as f64;
        assert_eq!(rotary, 64.0);
        assert!(arch.has_short_conv_layers());
        assert!(arch.layer_is_short_conv(0));
        assert!(!arch.layer_is_short_conv(2));
    }
}
