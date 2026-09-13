//! Tests for `QsaIndexerCacheManager` against real Metal buffer
//! allocations: layer classification, sizing, shared-position writes, the
//! incremental pooled-block cursor, and zero/reset behavior.
#![cfg(target_os = "macos")]

use turbospark_gpu::{MetalContext, QsaIndexerCacheManager};

/// A small `qwen4_exp`-shaped `ArchConfig`: 3 layers, one QSA (`mask == 1`),
/// two GDN (`mask == 2`) -- QSA layers are a minority of the stack on the
/// real checkpoint (12 of 48) and this mirrors that shape rather than
/// making every layer a QSA one, which could not catch a wrong
/// `is_qsa_layer` classification.
fn qwen4_style_arch() -> model_io::ArchConfig {
    model_io::ArchConfig {
        hidden_size: 64,
        intermediate_size: 128,
        moe_intermediate_size: 64,
        num_heads: 2,
        num_kv_heads: 2,
        num_full_kv_heads: 2,
        head_dim: 8,
        full_head_dim: 8,
        vocab_size: 100,
        sliding_window: 0,
        final_logit_softcap: 0.0,
        rope_theta: 10_000_000.0,
        full_rope_theta: 10_000_000.0,
        partial_rotary_factor: 0.25,
        num_layers: 3,
        num_experts: 1,
        top_k_experts: 1,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: vec![2, 2, 1], // two linear (GDN), one QSA
        hidden_activation: "silu".to_string(),
        family: model_io::ModelFamily::Qwen4Exp,
        attn_output_gate: true,
        attention_scale: 0.0625,
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: true,
        rope_neox_subdim: true,
        linear_attention: model_io::LinearAttentionConfig {
            num_k_heads: 4,
            num_v_heads: 4,
            key_head_dim: 8,
            value_head_dim: 8,
            conv_kernel_size: 4,
            output_gate_sigmoid: true,
        },
        compressed_attention: model_io::CompressedAttentionConfig {
            index_n_heads: 4,
            index_kv_heads: 1,
            index_head_dim: 8,
            index_top_k: 4,
            index_budget: 16,
            csa_compress_rate: 4,
            q_lora_rank: 0,
            o_lora_rank: 0,
            o_groups: 0,
            rope_head_dim: 0,
            compress_rope_theta: 0.0,
            rope_scaling_factor: 0.0,
            rope_scaling_original_max: 0,
            rope_scaling_beta_fast: 0.0,
            rope_scaling_beta_slow: 0.0,
            hca_compress_rate: 0,
        },
        hyper_connections: model_io::HyperConnectionConfig {
            mult: 4,
            sinkhorn_iters: 0,
            eps: 0.0,
            lowrank: 320,
        },
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: model_io::RopeScalingConfig::NONE,
        vision: model_io::VisionConfig::NONE,
        ple: model_io::PleConfig::NONE,
    }
}

#[test]
fn classifies_qsa_layers_and_sizes_state() {
    let context = MetalContext::new().unwrap();
    let arch = qwen4_style_arch();
    let max_context = 32usize;
    let manager = QsaIndexerCacheManager::new(context.device(), &arch, max_context);

    assert!(!manager.is_qsa_layer(0));
    assert!(!manager.is_qsa_layer(1));
    assert!(manager.is_qsa_layer(2));

    // index_kv_heads(1) * index_head_dim(8) * 2 bytes (FP16).
    assert_eq!(manager.raw_stride(2), 8 * 2);
    assert_eq!(manager.raw_stride(0), 0, "non-QSA layer has no stride");
    // index_head_dim(8) * 2 bytes.
    assert_eq!(manager.pooled_stride(), 8 * 2);
    assert_eq!(manager.compress_ratio(), 4);
}

#[test]
#[should_panic(expected = "index_budget > 0")]
fn refuses_an_architecture_with_no_indexer() {
    let context = MetalContext::new().unwrap();
    let mut arch = qwen4_style_arch();
    arch.compressed_attention = model_io::CompressedAttentionConfig::NONE;
    let _ = QsaIndexerCacheManager::new(context.device(), &arch, 32);
}

/// AGENTS.md/CLAUDE.md S11: `pooled_blocks` is sized at exactly one row
/// per block; a checkpoint declaring `index_kv_heads != 1` must be refused
/// at construction, not silently normalized to 1 by a `.max(1)`.
#[test]
#[should_panic(expected = "index_kv_heads == 1")]
fn refuses_an_architecture_with_index_kv_heads_above_one() {
    let context = MetalContext::new().unwrap();
    let mut arch = qwen4_style_arch();
    arch.compressed_attention.index_kv_heads = 2;
    let _ = QsaIndexerCacheManager::new(context.device(), &arch, 32);
}

#[test]
#[should_panic(expected = "index_kv_heads == 1")]
fn refuses_an_architecture_with_index_kv_heads_zero() {
    let context = MetalContext::new().unwrap();
    let mut arch = qwen4_style_arch();
    arch.compressed_attention.index_kv_heads = 0;
    let _ = QsaIndexerCacheManager::new(context.device(), &arch, 32);
}

/// A model manifest can carry dimensions wider than the u32 kernel ABI.
/// This exact shape used to wrap `max_context * raw_stride` to a small
/// allocation while retaining a huge position-1 write offset.
#[test]
#[should_panic(expected = "QSA raw-key allocation overflows usize")]
fn refuses_wrapping_raw_key_allocation() {
    let context = MetalContext::new().unwrap();
    let mut arch = qwen4_style_arch();
    arch.compressed_attention.index_n_heads = 3;
    arch.compressed_attention.index_head_dim = (1i64 << 62) + 128;
    let _ = QsaIndexerCacheManager::new(context.device(), &arch, 4096);
}

#[test]
#[should_panic(expected = "not a QSA layer")]
fn raw_keys_view_panics_for_a_non_qsa_layer() {
    let context = MetalContext::new().unwrap();
    let arch = qwen4_style_arch();
    let manager = QsaIndexerCacheManager::new(context.device(), &arch, 32);
    let _ = manager.raw_keys_view(0);
}

/// The raw key buffer takes a caller-supplied `position` on every write
/// rather than owning its own cursor -- this is what "shares a position
/// counter with the KV cache" means in practice. Writing at position 5 and
/// reading the buffer back must land the bytes at byte offset
/// `5 * raw_stride`, not at the front of the buffer.
#[test]
fn write_raw_key_lands_at_the_given_positions_byte_offset() {
    let context = MetalContext::new().unwrap();
    let arch = qwen4_style_arch();
    let max_context = 32usize;
    let manager = QsaIndexerCacheManager::new(context.device(), &arch, max_context);

    let stride = manager.raw_stride(2);
    let row: Vec<u8> = (0..stride as u8).collect();
    let position = 5usize;
    manager.write_raw_key(2, position, &row);

    let buffer = manager.raw_keys_view(2);
    let ptr = buffer.contents() as *const u8;
    let all = unsafe { std::slice::from_raw_parts(ptr, buffer.length() as usize) };

    let expected_offset = position * stride;
    assert_eq!(&all[expected_offset..expected_offset + stride], &row[..]);
    // Nothing else in the buffer was touched.
    assert!(all[..expected_offset].iter().all(|&b| b == 0));
    assert!(all[expected_offset + stride..].iter().all(|&b| b == 0));
}

/// The pooled-block cursor only ever moves FORWARD, and
/// `pooled_block_write_offset` tracks it -- this is the whole of the
/// "a pooled block is never recomputed" incremental design.
#[test]
fn pooled_block_cursor_advances_and_reports_the_write_offset() {
    let context = MetalContext::new().unwrap();
    let arch = qwen4_style_arch();
    let mut manager = QsaIndexerCacheManager::new(context.device(), &arch, 32);

    assert_eq!(manager.pooled_block_count(2), 0);
    assert_eq!(manager.pooled_block_write_offset(2), 0);

    manager.advance_pooled_blocks(2, 3);
    assert_eq!(manager.pooled_block_count(2), 3);
    assert_eq!(
        manager.pooled_block_write_offset(2),
        3 * manager.pooled_stride()
    );

    manager.advance_pooled_blocks(2, 5);
    assert_eq!(manager.pooled_block_count(2), 5);
}

#[test]
#[should_panic(expected = "must not move the cursor backward")]
fn pooled_block_cursor_refuses_to_move_backward() {
    let context = MetalContext::new().unwrap();
    let arch = qwen4_style_arch();
    let mut manager = QsaIndexerCacheManager::new(context.device(), &arch, 32);
    manager.advance_pooled_blocks(2, 3);
    manager.advance_pooled_blocks(2, 1);
}

#[test]
#[should_panic(expected = "exceeds this layer's block capacity")]
fn pooled_block_cursor_refuses_to_exceed_capacity() {
    let context = MetalContext::new().unwrap();
    let arch = qwen4_style_arch();
    let max_context = 32usize; // 32 / compress_ratio(4) = 8 complete blocks max
    let mut manager = QsaIndexerCacheManager::new(context.device(), &arch, max_context);
    manager.advance_pooled_blocks(2, 9);
}

#[test]
fn reset_zeroes_the_pooled_block_cursor_but_not_the_ring_of_a_position_field() {
    let context = MetalContext::new().unwrap();
    let arch = qwen4_style_arch();
    let mut manager = QsaIndexerCacheManager::new(context.device(), &arch, 32);

    manager.advance_pooled_blocks(2, 4);
    assert_eq!(manager.pooled_block_count(2), 4);

    manager.reset();
    assert_eq!(
        manager.pooled_block_count(2),
        0,
        "reset must zero the pooled-block cursor for a fresh generation"
    );
}
