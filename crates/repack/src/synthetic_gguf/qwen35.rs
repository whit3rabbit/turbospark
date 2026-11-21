//! A small dense `qwen35` GGUF with a K-quant trunk and, optionally, the
//! multi-token-prediction head llama.cpp appends as one more `blk.` block.
//!
//! It exists to drive a K-quant speculative verify end to end without the
//! 17 GB pi checkpoint: the walk ingests the head (`plan::classify`), the
//! runtime's gate admits Q4_K / Q6_K / Q8_0, and the batched verify runs the
//! K-quant GEMMs over GDN and full-attention layers alike.
//!
//! The block-type mix is the one a Q4_K_M file has: Q4_K matrices, Q6_K on
//! `output.weight` and every `ffn_down`, Q8_0 on `ssm_out` (the pi install's
//! re-encode) and on the head's `eh_proj`, and F32 `ssm_alpha`/`ssm_beta`
//! (BF16 once installed). Norms are BF16-exact F32, as llama.cpp writes them.
//!
//! Every row tiles a 256-element superblock, which is why the widths are
//! twice the MLX fixture's: hidden 256, four heads of 64, a 256-wide FFN and
//! a 256-wide GDN value stream.

use super::builder::{GgufBuilder, GgufFileAndRanges};

pub const QWEN35_GGUF_HIDDEN: u64 = 256;
pub const QWEN35_GGUF_VOCAB: u64 = 512;
/// Trunk depth. Layers 0..2 are GDN and layer 3 is full attention, at
/// `full_attention_interval` 4, so both verify paths run.
pub const QWEN35_GGUF_LAYERS: usize = 4;

const HEADS: u64 = 4;
const KV_HEADS: u64 = 2;
const HEAD_DIM: u64 = 64;
const Q_DIM: u64 = HEADS * HEAD_DIM;
const KV_DIM: u64 = KV_HEADS * HEAD_DIM;
const FFN: u64 = 256;
const LA_K_HEADS: u64 = 2;
const LA_V_HEADS: u64 = 4;
const LA_KEY_DIM: u64 = 64;
const LA_VALUE_DIM: u64 = 64;
const LA_CONV_K: u64 = 4;
const LA_VALUE: u64 = LA_V_HEADS * LA_VALUE_DIM;
const LA_QKV: u64 = 2 * LA_K_HEADS * LA_KEY_DIM + LA_VALUE;

/// The fixture, with or without the head block.
pub fn build_synthetic_qwen35_dense_gguf(with_head: bool) -> GgufFileAndRanges {
    const H: u64 = QWEN35_GGUF_HIDDEN;
    let blocks = QWEN35_GGUF_LAYERS as u32 + u32::from(with_head);
    let mut b = GgufBuilder::new()
        .metadata_str("general.architecture", "qwen35")
        .metadata_u32("qwen35.block_count", blocks)
        .metadata_u32("qwen35.embedding_length", H as u32)
        .metadata_u32("qwen35.feed_forward_length", FFN as u32)
        .metadata_u32("qwen35.attention.head_count", HEADS as u32)
        .metadata_u32("qwen35.attention.head_count_kv", KV_HEADS as u32)
        .metadata_u32("qwen35.attention.key_length", HEAD_DIM as u32)
        .metadata_u32("qwen35.attention.value_length", HEAD_DIM as u32)
        .metadata_u32("qwen35.rope.dimension_count", (HEAD_DIM / 4) as u32)
        .metadata_f32("qwen35.rope.freq_base", 10_000_000.0)
        .metadata_f32("qwen35.attention.layer_norm_rms_epsilon", 1e-6)
        .metadata_u32("qwen35.full_attention_interval", 4)
        .metadata_u32("qwen35.ssm.group_count", LA_K_HEADS as u32)
        .metadata_u32("qwen35.ssm.time_step_rank", LA_V_HEADS as u32)
        .metadata_u32("qwen35.ssm.state_size", LA_KEY_DIM as u32)
        .metadata_u32("qwen35.ssm.inner_size", LA_VALUE as u32)
        .metadata_u32("qwen35.ssm.conv_kernel", LA_CONV_K as u32);
    if with_head {
        b = b.metadata_u32("qwen35.nextn_predict_layers", 1);
    }
    b = b
        .q4_k_tensor("token_embd.weight", &[H, QWEN35_GGUF_VOCAB], 1)
        .q6_k_tensor("output.weight", &[H, QWEN35_GGUF_VOCAB], 2)
        .f32_upcast_bf16_tensor("output_norm.weight", &[H], 3);

    for layer in 0..QWEN35_GGUF_LAYERS {
        let p = format!("blk.{layer}.");
        let seed = layer as u8 * 23 + 10;
        b = b
            .f32_upcast_bf16_tensor(&format!("{p}attn_norm.weight"), &[H], seed)
            .f32_upcast_bf16_tensor(&format!("{p}post_attention_norm.weight"), &[H], seed + 1);
        if (layer + 1) % 4 == 0 {
            b = full_attention_block(b, &p, seed);
        } else {
            b = b
                .q4_k_tensor(&format!("{p}attn_qkv.weight"), &[H, LA_QKV], seed + 2)
                .q4_k_tensor(&format!("{p}attn_gate.weight"), &[H, LA_VALUE], seed + 3)
                // F32, as Q4_K_M and Q6_K files ship them; the walk narrows
                // them to BF16, so the verify runs the batched BF16 GEMM.
                .f32_tensor(&format!("{p}ssm_alpha.weight"), &[H, LA_V_HEADS], seed + 4)
                .f32_tensor(&format!("{p}ssm_beta.weight"), &[H, LA_V_HEADS], seed + 5)
                .q8_0_tensor(&format!("{p}ssm_out.weight"), &[LA_VALUE, H], seed + 6)
                .f32_tensor(
                    &format!("{p}ssm_conv1d.weight"),
                    &[LA_CONV_K, LA_QKV],
                    seed + 7,
                )
                // `-exp(A_log)`: the walk refuses a non-negative value.
                .tensor(&format!("{p}ssm_a"), 0, &[LA_V_HEADS], ssm_a(seed))
                .f32_tensor(&format!("{p}ssm_dt.bias"), &[LA_V_HEADS], seed + 9)
                .f32_upcast_bf16_tensor(&format!("{p}ssm_norm.weight"), &[LA_VALUE_DIM], seed + 10);
        }
        b = dense_ffn(b, &p, seed + 11);
    }

    if with_head {
        let p = format!("blk.{QWEN35_GGUF_LAYERS}.");
        let seed = 200;
        b = b
            .q8_0_tensor(&format!("{p}nextn.eh_proj.weight"), &[2 * H, H], seed)
            .f32_upcast_bf16_tensor(&format!("{p}nextn.enorm.weight"), &[H], seed + 1)
            .f32_upcast_bf16_tensor(&format!("{p}nextn.hnorm.weight"), &[H], seed + 2)
            .f32_upcast_bf16_tensor(&format!("{p}nextn.shared_head_norm.weight"), &[H], seed + 3)
            .f32_upcast_bf16_tensor(&format!("{p}attn_norm.weight"), &[H], seed + 4)
            .f32_upcast_bf16_tensor(&format!("{p}post_attention_norm.weight"), &[H], seed + 5);
        b = full_attention_block(b, &p, seed + 6);
        b = dense_ffn(b, &p, seed + 20);
    }

    b.build()
}

/// The six full-attention tensors. `attn_q` is twice `Q_DIM` wide because
/// the output gate is interleaved into it (`attn_output_gate`).
fn full_attention_block(b: GgufBuilder, p: &str, seed: u8) -> GgufBuilder {
    const H: u64 = QWEN35_GGUF_HIDDEN;
    b.q4_k_tensor(&format!("{p}attn_q.weight"), &[H, 2 * Q_DIM], seed + 2)
        .q4_k_tensor(&format!("{p}attn_k.weight"), &[H, KV_DIM], seed + 3)
        .q4_k_tensor(&format!("{p}attn_v.weight"), &[H, KV_DIM], seed + 4)
        .q4_k_tensor(&format!("{p}attn_output.weight"), &[Q_DIM, H], seed + 5)
        .f32_upcast_bf16_tensor(&format!("{p}attn_q_norm.weight"), &[HEAD_DIM], seed + 6)
        .f32_upcast_bf16_tensor(&format!("{p}attn_k_norm.weight"), &[HEAD_DIM], seed + 7)
}

/// `ssm_a` as llama.cpp stores it, `-exp(A_log)`: finite and negative, one
/// distinct value per V head so a head-order bug is visible.
fn ssm_a(seed: u8) -> Vec<u8> {
    (0..LA_V_HEADS)
        .flat_map(|h| (-(0.5 + 0.25 * h as f32 + seed as f32 * 1e-3)).to_le_bytes())
        .collect()
}

/// The dense FFN, with Q6_K on `ffn_down` as Q4_K_M puts it.
fn dense_ffn(b: GgufBuilder, p: &str, seed: u8) -> GgufBuilder {
    const H: u64 = QWEN35_GGUF_HIDDEN;
    b.q4_k_tensor(&format!("{p}ffn_gate.weight"), &[H, FFN], seed)
        .q4_k_tensor(&format!("{p}ffn_up.weight"), &[H, FFN], seed + 1)
        .q6_k_tensor(&format!("{p}ffn_down.weight"), &[FFN, H], seed + 2)
}
