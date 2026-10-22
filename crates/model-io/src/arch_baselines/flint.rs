use crate::arch_config::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig, MlaConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

/// Canonical `flint` baseline: Recursive Flint, pinned to the witness
/// `RecursiveSI/flint-int4` (MLX affine INT4 g64). Read off its
/// `config.json` and safetensors headers (`docs/FLINT_PHASE0.md`) and
/// cross-checked against the open llama.cpp port (ggml-org/llama.cpp#30928,
/// `src/models/flint.cpp`); neither is a vendor modeling file, because the
/// checkpoint ships none.
///
/// Layer flow, per layer `i` of 49:
///
/// ```text
/// h   = rms(x, input_layernorm)
/// q,k,v = q_proj h, k_proj h, v_proj h      # 32 / 4 / 4 heads of 128
/// q,k = per-head rms(q, q_norm), rms(k, k_norm)   # plain weight, dim 128
/// q,k = rope_neox(q,k, theta 10000)  if mask[i] == 0 (sliding)
///       unchanged                    if mask[i] == 1 (full = NoPE)
/// a   = attention(q,k,v, scale 128^-0.5, learned sink per q head,
///                 window 513 on sliding layers) ; x += o_proj a
/// h   = rms(x, post_attention_layernorm)
/// i < 2 : x += down(silu(gate h) * up h)                 # dense, 6144
/// i >= 2: p = sigmoid(h @ router^T)          # BF16 router, no bias
///         top6 by p; w = p / sum(p over the 6); scale 1.0
///         x += sum_j w_j * expert_j(h) + shared(h)       # shared UNGATED,
///                                                        # 1024, summed raw
/// ```
///
/// then `final rms -> lm_head` (INT4, untied) returns logits.
///
/// **The two dense lead layers are ALSO full-attention NoPE layers**
/// (`mask[0] == mask[1] == 1`), so "dense lead" and "NoPE" are independent
/// axes here and a flow must not tie them together.
///
/// **`full_rope_theta` is zero for the NoPE reason `muse_glimmer` documents**:
/// `no_rope_layer_indices` equals the full-attention set exactly, so the
/// flow skips the rotation whenever the mask is 1, and zero states that in
/// the manifest rather than leaving it to the mask alone.
///
/// `intermediate_size` is the SHARED expert's width (1024,
/// `shared_expert_intermediate_size`). The dense lead's 6144 (`config.json`
/// `intermediate_size`) is [`ArchConfig::dense_lead_intermediate_size`], for
/// the reason `deepseek_v2_lite_16b` states: one scalar cannot carry both.
///
/// `attention_scale` is `128^-0.5`, the value `llama`/`muse_glimmer` carry,
/// and is not a binary fraction; its manifest round trip is pinned by
/// `crates/model-io/tests/arch_config.rs`.
pub fn flint_40b_a2_5b() -> ArchConfig {
    ArchConfig {
        hidden_size: 2048,
        // The shared expert's width. See the header.
        intermediate_size: 1024,
        moe_intermediate_size: 512,
        num_heads: 32,
        num_kv_heads: 4,
        num_full_kv_heads: 4,
        head_dim: 128,
        full_head_dim: 128,
        vocab_size: 150_144,
        sliding_window: 513,
        final_logit_softcap: 0.0,
        rope_theta: 10_000.0,
        // NoPE on the full-attention layers; see the header.
        full_rope_theta: 0.0,
        partial_rotary_factor: 1.0,
        num_layers: 49,
        num_dense_leading_layers: 2,
        dense_lead_intermediate_size: 6144,
        num_experts: 256,
        top_k_experts: 6,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: flint_layer_mask(49),
        hidden_activation: "silu".to_string(),
        family: ModelFamily::Flint,
        // `q_proj` is 4096 = 32 * 128 rows: no packed output gate.
        attn_output_gate: false,
        attention_scale: 0.088_388_347_648_318_45,
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        // No `shared_expert_gate` tensor exists in any shard.
        shared_expert_gated: false,
        // Full rotary over the head, so NeoX sub-dimension and plain NeoX
        // pair the same elements.
        rope_neox_subdim: true,
        linear_attention: LinearAttentionConfig::NONE,
        mla: MlaConfig::NONE,
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        // `scoring_func: sigmoid` with `norm_topk_prob: true`: the top-6
        // sigmoid scores renormalised over the selected, scale 1.0. That is
        // neither the llama-flow softmax-over-selected nor MiniMax's
        // (biased) sigmoid, so the runtime router must be written for it.
        router_scoring_func: "sigmoid".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: RopeScalingConfig::NONE,
        vision: VisionConfig::NONE,
        ple: PleConfig::NONE,
    }
}

/// Flint's attention mask: `1` (full, NoPE) at layers 0, 1, then every
/// fourth layer from 5 (5, 9, ..., 45), then the last layer 48; `0`
/// (sliding, window 513) everywhere else. That is 14 full and 35 sliding
/// layers for `num_layers = 49`, verbatim from `config.json`'s `layer_types`.
///
/// A function so the repack parser can compare its `layer_types`-derived mask
/// against the same source, as `muse_glimmer_layer_mask` does. The pattern
/// is only defined for the 49-layer published shape: the last layer is full
/// by position, not by period, so it is stated rather than extrapolated.
pub fn flint_layer_mask(num_layers: i64) -> Vec<u8> {
    (0..num_layers)
        .map(|i| {
            let periodic = i >= 5 && (i - 5) % 4 == 0;
            if i == 0 || i == 1 || periodic || i == num_layers - 1 {
                1u8
            } else {
                0u8
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_matches_published_full_attention_layers() {
        let full: Vec<usize> = flint_layer_mask(49)
            .iter()
            .enumerate()
            .filter(|(_, &m)| m == 1)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            full,
            vec![0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48]
        );
    }
}
