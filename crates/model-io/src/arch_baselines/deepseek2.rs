use crate::arch_config::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig, MlaConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

/// Canonical `deepseek2` baseline: DeepSeek-V2-Lite-Chat, the pinned
/// witness (`mradermacher/DeepSeek-V2-Lite-Chat-GGUF` Q8_0, sidecars from
/// `deepseek-ai/DeepSeek-V2-Lite-Chat`). Every value below is read off the
/// checkpoint's own header and `config.json` (`docs/DEEPSEEK2_PHASE0.md`),
/// cross-checked against llama.cpp's `deepseek2.cpp` load_arch_hparams at
/// the commit the parity matrix cites.
///
/// **The baseline is named for the ARCHITECTURE, not the checkpoint**, on
/// `qwen_gdn_dense_27b`'s precedent: the `deepseek2` string covers DeepSeek
/// V2/V3, Kimi K2.5/K2.6, GLM-4.7-Flash and Mistral-Large-3, and a
/// checkpoint-shaped name would mislead every later reader of a shared
/// baseline. What those descendants change is mostly SHAPE (depth, expert
/// counts, `q_lora_rank`, vocab); every BEHAVIOURAL field here is the
/// architecture's and travels with the string.
///
/// Three fields deserve a second read. `intermediate_size` is the SHARED
/// expert width, 2816 = `n_shared_experts (2) x moe_intermediate (1408)`
/// fused into one SwiGLU exactly as llama.cpp's `build_ffn` runs it; the
/// DENSE lead layer's FFN is a different width (10944) and travels in
/// [`ArchConfig::dense_lead_intermediate_size`], because one scalar cannot
/// carry both and guessing either is fluent wrong output. `mla` carries
/// the latent-attention shapes whose absorbed form the runtime implements;
/// its `q_lora_rank` is 0 on the lite variants, and a full V2/V3 baseline
/// will set it nonzero. `rope_scaling.mscale` is 0.707 (from
/// `yarn_log_multiplier` 0.0707), the one number that separates this
/// architecture's YaRN from gpt-oss's.
///
/// `attention_scale` is derived, not assumed:
/// `(1 + 0.1 * 0.707 * ln 40)^2 / sqrt(192)` per the HF reference's
/// `softmax_scale * yarn_get_mscale(factor, mscale_all_dim) ** 2` and
/// llama.cpp's identical `kq_scale`. It is NOT a binary fraction; its
/// manifest round-trip is pinned by a test beside this baseline
/// (`arch_config.rs::deepseek2_attention_scale_round_trips`).
pub fn deepseek_v2_lite_16b() -> ArchConfig {
    ArchConfig {
        hidden_size: 2048,
        // The fused shared expert: 2 experts of 1408 run as ONE SwiGLU.
        intermediate_size: 2816,
        moe_intermediate_size: 1408,
        num_heads: 16,
        // The MLA cache is ONE shared row per layer (MQA after absorption),
        // not 16: the checkpoint's `head_count_kv = 16` describes its
        // EXPANDED form, which this port does not run. The runtime derives
        // the cache geometry from `mla`, so this field is vestigial here.
        num_kv_heads: 1,
        num_full_kv_heads: 1,
        // Not the per-head width (192): that lives in `mla`. Every MLA
        // layer is mask 5 and no SWA/full geometry is consulted, but the
        // fields exist so the struct stays uniform; they hold the q head
        // width and the v width respectively.
        head_dim: 192,
        full_head_dim: 128,
        vocab_size: 102_400,
        sliding_window: 0,
        final_logit_softcap: 0.0,
        rope_theta: 10_000.0,
        full_rope_theta: 10_000.0,
        // MLA rope is not partial by factor: the rotated window is the
        // trailing `mla.rope_head_dim` of each head. Zero reads as "the
        // window comes from mla", and no code may multiply this factor.
        partial_rotary_factor: 0.0,
        num_layers: 27,
        num_experts: 64,
        top_k_experts: 6,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        // 5 = MLA on every layer; the dense LEAD is an FFN difference only.
        full_attention_layer_mask: vec![5u8; 27],
        hidden_activation: "silu".to_string(),
        family: ModelFamily::Deepseek2,
        attn_output_gate: false,
        // (1 + 0.1 * 0.707 * ln 40)^2 / sqrt(192). See the doc above.
        attention_scale: 0.114_721_386_792_926_12,
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        // The shared expert is a plain SwiGLU with no scalar gate.
        shared_expert_gated: false,
        rope_neox_subdim: true,
        linear_attention: LinearAttentionConfig::NONE,
        mla: MlaConfig {
            kv_lora_rank: 512,
            q_lora_rank: 0,
            nope_head_dim: 128,
            rope_head_dim: 64,
            v_head_dim: 128,
        },
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        // Router logits feed a plain softmax over ALL 64 experts; top-6
        // weights are those full-softmax values with no renormalization
        // (`norm_topk_prob: false`) and `routed_scaling_factor` is 1.0.
        // The runtime's llama-flow top-k (softmax over the SELECTED) is a
        // DIFFERENT function and must not run here.
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: RopeScalingConfig {
            factor: 40.0,
            original_context: 4096,
            beta_fast: 32.0,
            beta_slow: 1.0,
            // The checkpoint's own value (yarn_log_multiplier 0.0707).
            mscale: 0.707,
        },
        vision: VisionConfig::NONE,
        ple: PleConfig::NONE,
        // Layer 0's dense FFN: `intermediate_size` at 10944 against the
        // shared expert's 2816 above. Zero means "no dense lead".
        dense_lead_intermediate_size: 10944,
        num_dense_leading_layers: 1,
    }
}
