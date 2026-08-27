use crate::arch_config::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily, RopeScalingConfig, VisionConfig,
};

/// Canonical `Muse-Glimmer-30B` baseline: 52 DENSE layers, GQA at 32 query
/// heads over 2 KV heads, an alternating three-sliding/one-full window, a
/// separate attention output gate, sandwich norms, and a logit softcap.
///
/// Read off `mlx-community/Muse-Glimmer-30B-4bit`'s `config.json` and
/// `mlx_vlm/models/muse_glimmer/language.py`, never inferred.
///
/// **`full_rope_theta` IS ZERO AND THAT IS THE FILE'S OWN VALUE, not an
/// "absent" sentinel.** `config.json` publishes a per-layer `layer_rope_theta`
/// array reading 500000 on every sliding layer and literally `0` on every
/// full one, and the reference gates its RoPE on `bool(layer_rope_theta[i])`.
/// So the thirteen full-attention layers are NoPE: no rotation at all, not a
/// default theta. The flow reads this field per layer and skips the rotation
/// when it is zero. Applying RoPE there instead is a different function that
/// still produces fluent text (AGENTS.md Gotcha 33's failure mode), which is
/// why it is stated here rather than left to the mask.
///
/// **`attn_output_gate` is FALSE even though this architecture HAS an
/// attention output gate.** That field asks a narrower question than its name
/// suggests: whether `q_proj` emits `2 * num_heads * head_dim` rows as
/// per-head `[query; gate]` pairs, which is Qwen's packing and needs
/// `split_q_gate_fp16`. Muse Glimmer's gate is its own tensor,
/// `self_attn.gate_proj` at `[num_heads * head_dim, hidden]`, so `q_proj` is
/// ordinary and the gate is applied by the flow. Setting this true would size
/// the q GEMV at double its real row count.
///
/// Three more fields whose values are not the obvious ones:
/// `embedding_scaled_by_sqrt_hidden` is false because this family norms the
/// embedding row (a no-scale RMS) where Gemma scales it; `num_experts` is 0
/// because the model is dense, so `moe_intermediate_size` encodes nothing and
/// `intermediate_size` is the dense FFN width (the same collision the dense
/// `llama` and `qwen3_5` halves have); and `attention_k_eq_v` is false
/// because there is a real, separate `v_proj`.
///
/// The two published scalars this struct deliberately does NOT carry --
/// `qk_scale_factor` (3.87) and `output_multiplier` (26^-0.5) -- plus the two
/// RMS epsilons live as constants in `crates/runtime`'s
/// `families/museglimmer/state.rs`, following the `rms_eps` precedent
/// (`llama`'s 1e-5 against `qwen3moe`'s 1e-6 are constants too, not fields).
/// None of the four is a binary fraction, and `arch_validation` compares
/// manifest floats with `!=` on `f64` where serde_json parses to ~1 ULP, so
/// carrying them through `manifest.json` would walk straight into AGENTS.md
/// Gotcha 24. `crates/repack`'s parser reads all four off `config.json` and
/// asserts them against those constants, so a future checkpoint that moves
/// one reddens a millisecond test rather than decoding subtly wrong.
pub fn muse_glimmer_30b() -> ArchConfig {
    ArchConfig {
        hidden_size: 6656,
        // DENSE width. `moe_intermediate_size` stays 0: there are no experts.
        intermediate_size: 19_968,
        moe_intermediate_size: 0,
        num_heads: 32,
        num_kv_heads: 2,
        num_full_kv_heads: 2,
        // One head dim for both layer kinds, unlike Gemma's 256/512 split.
        head_dim: 128,
        full_head_dim: 128,
        vocab_size: 202_048,
        sliding_window: 2048,
        final_logit_softcap: 20.0,
        rope_theta: 500_000.0,
        // NoPE on the full-attention layers; see the header.
        full_rope_theta: 0.0,
        // Full rotary over the head on the layers that rotate at all: the
        // reference builds its rope at `head_dim` with no partial factor.
        partial_rotary_factor: 1.0,
        num_layers: 52,
        num_experts: 0,
        top_k_experts: 0,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        // `[sliding, sliding, sliding, full] x 13`, verbatim from
        // `text_config.layer_types`.
        full_attention_layer_mask: muse_glimmer_layer_mask(52),
        hidden_activation: "silu".to_string(),
        family: ModelFamily::MuseGlimmer,
        attn_output_gate: false,
        // 128^-0.5 = 2^-3.5, the same value Mixtral carries and the same
        // Gotcha 24 caveat: not a binary fraction, so the round trip is
        // pinned by `crates/model-io/tests/arch_config.rs`. The reference's
        // `qk_scale_factor` is NOT folded in here -- it multiplies Q after
        // its per-head norm, and folding both into one constant would give
        // an arbitrary decimal on the `!=` path AND reassociate the FP order
        // the reference fixes.
        attention_scale: 0.088_388_347_648_318_45,
        // The embedding is NORMED, not scaled; see the header.
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
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
        vision: VisionConfig::NONE,
    }
}

/// The `[0, 0, 0, 1]` window pattern, repeated to `num_layers`.
///
/// A function rather than a literal because 52 entries typed out is 52
/// chances to put a `1` in the wrong place, and the repack parser has to
/// build the same mask from `text_config.layer_types` and compare against it.
/// The period is 4 with the FULL layer last, so layer index `i` is full when
/// `i % 4 == 3` -- layers 3, 7, 11, ..., 51, thirteen of them.
pub fn muse_glimmer_layer_mask(num_layers: i64) -> Vec<u8> {
    (0..num_layers)
        .map(|i| if i % 4 == 3 { 1u8 } else { 0u8 })
        .collect()
}
