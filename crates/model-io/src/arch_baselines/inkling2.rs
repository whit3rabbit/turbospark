//! Inkling 2 baselines. Facts: `docs/INKLING2_PHASE0.md`.

use crate::arch_config::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig, MlaConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

/// Inkling 2's attention schedule: full attention (1) on the last layer of
/// every 6-layer residual block, sliding window (0) elsewhere. The checkpoint
/// states it as `local_layer_ids` (every layer but 5, 11, 17, 23); the parser
/// checks the two agree rather than trusting this formula.
pub fn inkling2_layer_mask(num_layers: usize) -> Vec<u8> {
    (0..num_layers)
        .map(|i| u8::from((i + 1) % 6 == 0))
        .collect()
}

/// `thinkingmachines/Inkling-2-Mini` at `ed40e7da`, `config.json ->
/// text_config`. `num_layers` is the 24 PHYSICAL layers; the loop (layers
/// 6-17, 3 passes) and the other Inkling-only values are in the
/// `inkling2.json` sidecar. No RoPE, so both thetas are 0 and the rotary
/// factor is 0. `final_logit_softcap` is 30; `short_conv_kernel_size` is the
/// sconv taps (4); `routed_scaling_factor` is `route_scale` (8).
pub fn inkling2_mini() -> ArchConfig {
    ArchConfig {
        hidden_size: 2304,
        // No shared expert and no dense FFN: every layer is routed.
        intermediate_size: 0,
        moe_intermediate_size: 1152,
        num_heads: 24,
        num_kv_heads: 8,
        num_full_kv_heads: 8,
        head_dim: 128,
        full_head_dim: 128,
        // Padded table rows; the logits are cut to the 200,058-token
        // unpadded vocabulary (sidecar).
        vocab_size: 201_024,
        sliding_window: 1024,
        final_logit_softcap: 30.0,
        rope_theta: 0.0,
        full_rope_theta: 0.0,
        partial_rotary_factor: 0.0,
        num_layers: 24,
        dense_lead_intermediate_size: 0,
        num_dense_leading_layers: 0,
        num_experts: 128,
        top_k_experts: 8,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: inkling2_layer_mask(24),
        // Softcapped SwiGLU: softcap(g, 4) * sigmoid(g) * softcap(u, 25).
        hidden_activation: "situ".to_string(),
        family: ModelFamily::Inkling2Loop,
        // The sigmoid output gate is per q CHANNEL from its own projection
        // (`gate_attn`), not packed into q_proj.
        attn_output_gate: false,
        attention_scale: 0.088_388_347_648_318_45,
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: false,
        rope_neox_subdim: false,
        linear_attention: LinearAttentionConfig::NONE,
        mla: MlaConfig::NONE,
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "sigmoid".to_string(),
        routed_scaling_factor: 8.0,
        swiglu_limit: 0.0,
        rope_scaling: RopeScalingConfig::NONE,
        vision: VisionConfig::NONE,
        short_conv_kernel_size: 4,
        rope_freq_factors: Vec::new(),
        ple: PleConfig::NONE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_matches_the_checkpoints_local_layer_ids() {
        // `local_layer_ids` from config.json, verbatim.
        let local = [
            0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22,
        ];
        let mask = inkling2_layer_mask(24);
        for (i, &m) in mask.iter().enumerate() {
            assert_eq!(m == 0, local.contains(&i), "layer {i}");
        }
    }
}
