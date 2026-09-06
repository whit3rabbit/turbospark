//! Architectural constants and config builders for synthetic `qwen4_exp` decode fixtures.

use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily, PleConfig,
};

use super::ngram::{NGRAM_EOS_TOKEN_ID, NGRAM_HEADS, NGRAM_HEAD_DIM, NGRAM_VOCAB_BASE, PLE_LAYER};

pub const HIDDEN: usize = 128;
pub const HC_COUNT: usize = 2;
pub const HC_LOWRANK: usize = 64;
pub const NUM_HEADS: usize = 4;
pub const HEAD_DIM: usize = 32;
pub const NUM_KV_HEADS: usize = 2;

pub(super) const LA_K_HEADS: usize = 2;
pub(super) const LA_V_HEADS: usize = 4;
pub(super) const LA_KEY_DIM: usize = 32;
pub(super) const LA_VALUE_DIM: usize = 32;
pub(super) const LA_CONV_K: usize = 4;

pub const NUM_EXPERTS: usize = 4;
/// The QSA indexer's shape: `IDX_HEADS` query heads over ONE shared key
/// head (`index_kv_heads == 1`, as on the real checkpoint), each
/// `IDX_HEAD_DIM` wide, pooling `IDX_COMPRESS` tokens per block.
/// `(IDX_HEADS + IDX_KV_HEADS) * IDX_HEAD_DIM` is 64 rows, the same row
/// count as this fixture's `k_proj`, so the INT4 packing path is one it
/// already exercises. `IDX_HEAD_DIM` must be at least `rotary_dim`
/// (`HEAD_DIM / 4 == 8`), which `validate_architecture` asserts.
pub const IDX_HEADS: usize = 3;
pub const IDX_KV_HEADS: usize = 1;
pub const IDX_HEAD_DIM: usize = 16;
pub const IDX_COMPRESS: usize = 4;
/// The default fixture keeps the REAL checkpoint's budget, so every test
/// written against it stays below budget and byte-identical to before the
/// indexer was wired (`the_synthetic_flows_arithmetic_is_frozen`);
/// [`build_synthetic_qwen4_exp_decode_install_with_indexer_budget`] is how
/// a test crosses it in a handful of tokens.
pub const IDX_BUDGET: i64 = 2048;
pub const TOP_K: usize = 2;
pub(super) const MOE_INTER: usize = 64;
/// The shared expert's width (`ArchConfig::intermediate_size` on this
/// family, matching `families/qwen/moe.rs`'s own convention).
pub(super) const SHARED_INTER: usize = 64;

/// `num_layers` this fixture always builds: enough to cover GDN, the PLE
/// layer (also GDN, matching the real checkpoint's own layer 1), and one
/// QSA-as-dense-attention layer, with MoE on every one of them.
pub const NUM_LAYERS: usize = 4;

/// `[GDN, GDN+PLE, attention, GDN]` -- `mask[l] == 1` is full attention,
/// `2` is linear (GDN).
pub(super) fn layer_mask() -> Vec<u8> {
    vec![2, 2, 1, 2]
}

pub(super) fn wide_dim() -> usize {
    HIDDEN * HC_COUNT
}

fn linear_attention() -> LinearAttentionConfig {
    LinearAttentionConfig {
        num_k_heads: LA_K_HEADS as i64,
        num_v_heads: LA_V_HEADS as i64,
        key_head_dim: LA_KEY_DIM as i64,
        value_head_dim: LA_VALUE_DIM as i64,
        conv_kernel_size: LA_CONV_K as i64,
        // The SIGMOID variant -- this fixture's whole reason for not being
        // `synthetic_qwen::dense`'s GDN shape reused verbatim.
        output_gate_sigmoid: true,
    }
}

/// A tiny but fully decode-shaped `qwen4_exp` architecture, at the real
/// checkpoint's indexer budget.
pub fn tiny_qwen4_exp_decode_arch(vocab_size: i64) -> ArchConfig {
    tiny_qwen4_exp_decode_arch_with_indexer_budget(vocab_size, IDX_BUDGET)
}

/// [`tiny_qwen4_exp_decode_arch`] with the QSA `index_budget` chosen by the
/// caller (a whole number of `IDX_COMPRESS`-token blocks; `index_top_k` is
/// derived as `budget / IDX_COMPRESS`, the reference's own definition).
pub fn tiny_qwen4_exp_decode_arch_with_indexer_budget(
    vocab_size: i64,
    indexer_budget: i64,
) -> ArchConfig {
    assert!(
        indexer_budget > 0 && indexer_budget % IDX_COMPRESS as i64 == 0,
        "indexer_budget must be a positive whole number of blocks"
    );
    let num_layers = NUM_LAYERS as i64;
    ArchConfig {
        hidden_size: HIDDEN as i64,
        // The shared expert's width on this family (`families/qwen/moe.rs`'s
        // convention, matching Qwen 3.6's).
        intermediate_size: SHARED_INTER as i64,
        moe_intermediate_size: MOE_INTER as i64,
        num_heads: NUM_HEADS as i64,
        num_kv_heads: NUM_KV_HEADS as i64,
        num_full_kv_heads: NUM_KV_HEADS as i64,
        head_dim: HEAD_DIM as i64,
        full_head_dim: HEAD_DIM as i64,
        vocab_size,
        sliding_window: 0,
        final_logit_softcap: 0.0,
        rope_theta: 10_000_000.0,
        full_rope_theta: 10_000_000.0,
        partial_rotary_factor: 0.25,
        num_layers,
        num_experts: NUM_EXPERTS as i64,
        top_k_experts: TOP_K as i64,
        // The real checkpoint's own value: `lm_head` is its own tensor.
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: layer_mask(),
        hidden_activation: "silu".to_string(),
        family: ModelFamily::Qwen4Exp,
        attn_output_gate: true,
        // A binary fraction, not `HEAD_DIM^-0.5` (irrational at this width):
        // `arch_validation` compares manifest floats with `!=` against a
        // ~1-ULP parser (AGENTS.md Gotcha 24), matching
        // `synthetic_qwen::dense_arch`'s own workaround.
        attention_scale: 0.125,
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: true,
        rope_neox_subdim: true,
        linear_attention: linear_attention(),
        // The real checkpoint's indexer SHAPE, scaled down (`IDX_*`), and
        // the caller's budget. Every one of these is dispatched on since
        // the indexer was wired (`families/qwen4/attn.rs`): heads and dim
        // size `index_qk_proj` and the per-head norms, compress and top_k
        // decide when block selection starts dropping blocks.
        compressed_attention: CompressedAttentionConfig {
            index_n_heads: IDX_HEADS as i64,
            index_kv_heads: IDX_KV_HEADS as i64,
            index_head_dim: IDX_HEAD_DIM as i64,
            index_top_k: indexer_budget / IDX_COMPRESS as i64,
            index_budget: indexer_budget,
            csa_compress_rate: IDX_COMPRESS as i64,
            q_lora_rank: 0,
            o_lora_rank: 0,
            o_groups: 0,
            rope_head_dim: 0,
            hca_compress_rate: 0,
            compress_rope_theta: 0.0,
            rope_scaling_factor: 0.0,
            rope_scaling_original_max: 0,
            rope_scaling_beta_fast: 0.0,
            rope_scaling_beta_slow: 0.0,
        },
        hyper_connections: HyperConnectionConfig {
            mult: HC_COUNT as i64,
            lowrank: HC_LOWRANK as i64,
            sinkhorn_iters: 0,
            eps: 0.0,
        },
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: model_io::RopeScalingConfig::NONE,
        vision: model_io::VisionConfig::NONE,
        ple: PleConfig {
            ngram_size: 3,
            heads_per_ngram: 2,
            ngram_vocab_size_base: NGRAM_VOCAB_BASE,
            make_divisible_by: 1,
            split_ngram_parts: 1,
            ple_embed_dim: (NGRAM_HEADS * NGRAM_HEAD_DIM) as i64,
            conv_kernel_size: 4,
            layer_ids: vec![PLE_LAYER as i64 + 1],
            seed: 1234,
            eos_token_id: NGRAM_EOS_TOKEN_ID,
        },
    }
}
