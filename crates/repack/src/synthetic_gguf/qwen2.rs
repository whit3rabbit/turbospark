//! A small Qwen2 GGUF covering the dense Llama-shaped intake path.
//!
//! The fixture keeps the Qwen2-specific parts that a generic Llama fixture
//! cannot exercise: the `qwen2` architecture tag, its explicit RMS epsilon,
//! and the three input-projection bias tensors.

use super::builder::{GgufBuilder, GgufFileAndRanges};

/// A complete, tiny Qwen2 GGUF with the tensor inventory needed by the
/// single-file repack path. Dimensions are deliberately small but tile the
/// Q4_K superblock and preserve grouped-query attention.
pub fn build_synthetic_qwen2_gguf() -> GgufFileAndRanges {
    const HIDDEN: u64 = 256;
    const HEADS: u64 = 4;
    const KV_HEADS: u64 = 2;
    const HEAD_DIM: u64 = 64;
    const Q_DIM: u64 = HEADS * HEAD_DIM;
    const KV_DIM: u64 = KV_HEADS * HEAD_DIM;
    const FFN: u64 = 256;
    const VOCAB: u64 = 512;
    const LAYERS: usize = 2;

    let mut b = GgufBuilder::new()
        .metadata_str("general.architecture", "qwen2")
        .metadata_u32("qwen2.block_count", LAYERS as u32)
        .metadata_u32("qwen2.embedding_length", HIDDEN as u32)
        .metadata_u32("qwen2.feed_forward_length", FFN as u32)
        .metadata_u32("qwen2.attention.head_count", HEADS as u32)
        .metadata_u32("qwen2.attention.head_count_kv", KV_HEADS as u32)
        .metadata_u32("qwen2.attention.key_length", HEAD_DIM as u32)
        .metadata_u32("qwen2.attention.value_length", HEAD_DIM as u32)
        .metadata_u32("qwen2.rope.dimension_count", HEAD_DIM as u32)
        .metadata_f32("qwen2.rope.freq_base", 1_000_000.0)
        .metadata_f32("qwen2.attention.layer_norm_rms_epsilon", 1e-6)
        .q4_k_tensor("token_embd.weight", &[HIDDEN, VOCAB], 1)
        .q4_k_tensor("output.weight", &[HIDDEN, VOCAB], 2)
        .f32_upcast_bf16_tensor("output_norm.weight", &[HIDDEN], 3);

    for layer in 0..LAYERS {
        let p = format!("blk.{layer}.");
        let seed = layer as u8 * 19 + 10;
        b = b
            .f32_upcast_bf16_tensor(&format!("{p}attn_norm.weight"), &[HIDDEN], seed)
            .f32_upcast_bf16_tensor(&format!("{p}ffn_norm.weight"), &[HIDDEN], seed + 1)
            .q4_k_tensor(&format!("{p}attn_q.weight"), &[HIDDEN, Q_DIM], seed + 2)
            .q4_k_tensor(&format!("{p}attn_k.weight"), &[HIDDEN, KV_DIM], seed + 3)
            .q4_k_tensor(&format!("{p}attn_v.weight"), &[HIDDEN, KV_DIM], seed + 4)
            .q4_k_tensor(
                &format!("{p}attn_output.weight"),
                &[Q_DIM, HIDDEN],
                seed + 5,
            )
            // Qwen2 biases arrive as F32 in common GGUF conversions and are
            // narrowed to the BF16 resident format by the repack walk.
            .f32_upcast_bf16_tensor(&format!("{p}attn_q.bias"), &[Q_DIM], seed + 6)
            .f32_upcast_bf16_tensor(&format!("{p}attn_k.bias"), &[KV_DIM], seed + 7)
            .f32_upcast_bf16_tensor(&format!("{p}attn_v.bias"), &[KV_DIM], seed + 8)
            .q4_k_tensor(&format!("{p}ffn_gate.weight"), &[HIDDEN, FFN], seed + 9)
            .q4_k_tensor(&format!("{p}ffn_up.weight"), &[HIDDEN, FFN], seed + 10)
            .q4_k_tensor(&format!("{p}ffn_down.weight"), &[FFN, HIDDEN], seed + 11);
    }

    b.build()
}
