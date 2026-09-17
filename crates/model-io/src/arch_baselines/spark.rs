use crate::arch_config::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig, MlaConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

/// Canonical `Spark-X2.5-4B` baseline: 36 DENSE layers, GQA at 16 query
/// heads over 4 KV heads, the muse-shaped `[swa, swa, swa, full]` window at
/// 512, per-class RoPE (full: theta 5e6 with a 0.25 partial factor; SWA:
/// theta 1e4 full-head), a headwise scalar sigmoid output gate, exact-erf
/// GELU, and tied embeddings.
///
/// Read off `XHToken/Spark-X2.5-4B`'s `config.json` and `modeling_spark.py`,
/// cross-checked against the official GGUF's header
/// (`XHToken/Spark-X2.5-4B-GGUF`), never inferred. Facts live in
/// `docs/SPARK_PHASE0.md`.
///
/// **THE GGUF STORES THE THETAS INVERTED RELATIVE TO HF.** llama.cpp writes
/// the FULL-attention theta to plain `rope.freq_base` and the SWA theta to
/// `rope.freq_base_swa` even though 27 of 36 layers are SWA. `gguf_config`
/// maps by class, never by "which key looks like the default".
///
/// **`partial_rotary_factor` MEANS THE FULL-ATTENTION LAYERS ONLY**, with the
/// SWA arm at factor 1.0 hard-coded in the flow -- the gemma4 pattern
/// (`families/gemma4/attn.rs`), not a new ArchConfig field. The divisor for
/// the full layers' rotation is the ROTARY dim (64), which is
/// `rope_neox_subdim`'s convention (HF partial rotary pairs within the
/// sub-dim), NOT `rope_proportional_neox`'s head_dim divisor.
///
/// **`attn_output_gate` is FALSE even though this architecture HAS an
/// attention output gate**, with exactly the muse precedent: the field asks
/// whether `q_proj` emits `2 * num_heads * head_dim` rows as `[query; gate]`
/// pairs. Spark's gate is its own tensor, `self_attn.g_proj` at
/// `[num_heads, hidden]` -- one SCALAR per head, a shape no other family
/// here has (qwen packs full-width, muse is full-width). The flow applies
/// it unconditionally.
///
/// **`full_rope_theta` IS NONZERO AND THAT IS LEGAL HERE**, unlike
/// `muse_glimmer_30b` where 0 means NoPE: Spark rotates on every layer, and
/// the per-class thetas genuinely diverge (5e6 against 1e4). A flow that
/// treated a nonzero `full_rope_theta` as an error would refuse a correct
/// checkpoint; the museglimmer refusal axis is muse-specific.
///
/// `hidden_activation` is `"gelu"` and means EXACT ERF (the reference
/// implementation raises on anything else), not the tanh approximation the
/// string `"gelu_pytorch_tanh"` names for Gemma. The dispatch must match the
/// exact value and route to the erf kernel, never to the
/// `contains("silu")`-else-tanh fallback.
///
/// The RMS epsilon (1e-6, from `config.json` and the GGUF's
/// `attention.layer_norm_rms_epsilon`) rides as a flow constant in
/// `crates/runtime`'s `families/spark/state.rs`, following the llama /
/// qwen3moe precedent rather than spending an ArchConfig field on it.
pub fn spark_x25_4b() -> ArchConfig {
    ArchConfig {
        hidden_size: 2560,
        // DENSE width. `moe_intermediate_size` stays 0: there are no experts.
        intermediate_size: 10_240,
        moe_intermediate_size: 0,
        num_heads: 16,
        num_kv_heads: 4,
        num_full_kv_heads: 4,
        // One head dim for both layer kinds and one kv-head count for both:
        // `attention.key_length` == `attention.value_length` == 256, 4 kv
        // heads on the full layers too.
        head_dim: 256,
        full_head_dim: 256,
        vocab_size: 131_072,
        sliding_window: 512,
        final_logit_softcap: 0.0,
        rope_theta: 10_000.0,
        // The full-attention class rotates at 5e6 over its leading 64 dims.
        full_rope_theta: 5_000_000.0,
        // 0.25 on the FULL layers; the SWA layers rotate the full head.
        partial_rotary_factor: 0.25,
        num_layers: 36,
        dense_lead_intermediate_size: 0,
        num_dense_leading_layers: 0,
        num_experts: 0,
        top_k_experts: 0,
        // The GGUF carries no `output.weight`; the head reuses the embedding.
        tie_word_embeddings: true,
        attention_k_eq_v: false,
        // `[swa, swa, swa, full] x 9`, verbatim from `config.layer_types`
        // (GGUF `sliding_window_pattern` is the inverse: 1 = sliding).
        full_attention_layer_mask: spark_layer_mask(36),
        hidden_activation: "gelu".to_string(),
        family: ModelFamily::Spark25,
        attn_output_gate: false,
        // 256^-0.5 = 2^-4: an exact binary fraction, so the manifest f64
        // round trip is exact (AGENTS.md Gotcha 24 does not bite here).
        attention_scale: 0.0625,
        // The embedding is used directly; no sqrt(hidden), no norm.
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
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: RopeScalingConfig::NONE,
        vision: VisionConfig::NONE,
        ple: PleConfig::NONE,
    }
}

/// The `[0, 0, 0, 1]` window pattern, repeated to `num_layers`: the SAME
/// period and phase as `muse_glimmer_layer_mask` (full at `i % 4 == 3`),
/// 9 full layers here against muse's 13. A function for the same reason:
/// 36 typed entries are 36 chances to put a `1` in the wrong place, and the
/// GGUF intake has to build the inverse of `sliding_window_pattern` and
/// compare against it.
pub fn spark_layer_mask(num_layers: i64) -> Vec<u8> {
    (0..num_layers)
        .map(|i| if i % 4 == 3 { 1u8 } else { 0u8 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the mask's period, phase and population against the published
    /// `layer_types` array: full exactly at i % 4 == 3, 9 full of 36, and
    /// every non-full layer sliding. The GGUF's `sliding_window_pattern`
    /// is this INVERTED (1 = sliding), which the gguf intake relies on.
    #[test]
    fn the_layer_mask_is_three_sliding_then_one_full() {
        let mask = spark_layer_mask(36);
        assert_eq!(mask.len(), 36);
        assert_eq!(mask.iter().filter(|&&m| m == 1).count(), 9);
        for (i, &m) in mask.iter().enumerate() {
            assert_eq!(m, if i % 4 == 3 { 1 } else { 0 }, "layer {i}");
        }
        // First window reads [swa, swa, swa, full], verbatim from
        // config.json's layer_types head.
        assert_eq!(&mask[..4], &[0, 0, 0, 1]);
    }

    /// The baseline is exactly the published checkpoint config, and the two
    /// facts the flow branches on are pinned beside the fields they drive.
    #[test]
    fn the_baseline_matches_the_published_config() {
        let arch = spark_x25_4b();
        assert_eq!(arch.family, ModelFamily::Spark25);
        assert_eq!(arch.num_layers, 36);
        assert_eq!(arch.hidden_size, 2560);
        assert_eq!(arch.intermediate_size, 10_240);
        assert_eq!(arch.num_heads, 16);
        assert_eq!(arch.num_kv_heads, 4);
        assert_eq!(arch.head_dim, 256);
        assert_eq!(arch.vocab_size, 131_072);
        assert_eq!(arch.sliding_window, 512);
        // Per-class rope: full 5e6 over the leading 64 dims, SWA 1e4 over
        // the whole head.
        assert_eq!(arch.full_rope_theta, 5_000_000.0);
        assert_eq!(arch.rope_theta, 10_000.0);
        assert_eq!(arch.partial_rotary_factor, 0.25);
        assert!((arch.attention_scale - 0.0625).abs() < f64::EPSILON);
        assert_eq!(arch.final_logit_softcap, 0.0);
        assert!(arch.tie_word_embeddings);
        assert!(!arch.attn_output_gate);
        assert_eq!(arch.hidden_activation, "gelu");
        assert_eq!(arch.num_experts, 0);
    }
}
