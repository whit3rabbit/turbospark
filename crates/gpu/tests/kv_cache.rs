//! Tests for `KvCacheManager` against real Metal buffer allocations: layer
//! kind classification, stride/capacity sizing, position advancement, ring
//! wraparound for sliding-window layers, and reset.
#![cfg(target_os = "macos")]

use mrefrust_gpu::{KvCacheManager, LayerKind, MetalContext};

fn toy_arch(mask: Vec<u8>) -> model_io::ArchConfig {
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
        sliding_window: 4,
        final_logit_softcap: 0.0,
        rope_theta: 10000.0,
        full_rope_theta: 10000.0,
        partial_rotary_factor: 1.0,
        num_layers: mask.len() as i64,
        num_experts: 1,
        top_k_experts: 1,
        tie_word_embeddings: true,
        attention_k_eq_v: true,
        full_attention_layer_mask: mask,
        hidden_activation: "gelu_pytorch_tanh".to_string(),
        family: model_io::ModelFamily::Gemma4,
        attn_output_gate: false,
        attention_scale: 1.0,
        embedding_scaled_by_sqrt_hidden: true,
        router_scaled: true,
        ffn_sandwich_norms: true,
        shared_expert_gated: false,
        rope_neox_subdim: false,
        linear_attention: model_io::LinearAttentionConfig::NONE,
        compressed_attention: model_io::CompressedAttentionConfig::NONE,
        hyper_connections: model_io::HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
    }
}

#[test]
fn classifies_layer_kinds_from_the_mask() {
    let context = MetalContext::new().unwrap();
    let arch = toy_arch(vec![0, 1]); // swa, full
    let cache = KvCacheManager::new(context.device(), &arch, 32, false, None, 8, None).unwrap();
    assert_eq!(cache.layer_kind(0), LayerKind::Swa);
    assert_eq!(cache.layer_kind(1), LayerKind::Full);
}

#[test]
fn stride_matches_kv_heads_times_head_dim_times_two() {
    let context = MetalContext::new().unwrap();
    let arch = toy_arch(vec![0]);
    let cache = KvCacheManager::new(context.device(), &arch, 32, false, None, 8, None).unwrap();
    assert_eq!(cache.stride(0), 2 * 8 * 2); // num_kv_heads * head_dim * fp16
}

#[test]
fn advance_moves_the_position_cursor() {
    let context = MetalContext::new().unwrap();
    let arch = toy_arch(vec![1]);
    let mut cache = KvCacheManager::new(context.device(), &arch, 32, false, None, 8, None).unwrap();
    assert_eq!(cache.position(), 0);
    cache.advance();
    cache.advance_by(3);
    assert_eq!(cache.position(), 4);
}

#[test]
#[should_panic(expected = "exceed max_context")]
fn advance_past_max_context_panics() {
    let context = MetalContext::new().unwrap();
    let arch = toy_arch(vec![1]);
    let mut cache = KvCacheManager::new(context.device(), &arch, 4, false, None, 8, None).unwrap();
    cache.advance_by(5);
}

#[test]
fn ring_enabled_swa_layer_caps_capacity_below_max_context() {
    let context = MetalContext::new().unwrap();
    let arch = toy_arch(vec![0, 1]);
    let cache = KvCacheManager::new(context.device(), &arch, 1000, true, Some(4), 8, None).unwrap();
    // swa capacity = sliding_window(4) + max_prefill_chunk_tokens(8) = 12.
    assert_eq!(cache.capacity(0), 12);
    assert_eq!(cache.ring_capacity(0), 12);
    // Full layers stay linear regardless of ring mode.
    assert_eq!(cache.capacity(1), 1000);
    assert_eq!(cache.ring_capacity(1), 0);
}

#[test]
fn k_and_v_slots_are_distinct_buffers_at_the_same_offset() {
    let context = MetalContext::new().unwrap();
    let arch = toy_arch(vec![1]);
    let cache = KvCacheManager::new(context.device(), &arch, 32, false, None, 8, None).unwrap();
    let (k_buf, k_off) = cache.k_slot(0, 5);
    let (v_buf, v_off) = cache.v_slot(0, 5);
    assert_eq!(k_off, v_off);
    assert!(!std::ptr::eq(k_buf, v_buf));
}

#[test]
fn linear_layer_has_no_kv_slots() {
    let context = MetalContext::new().unwrap();
    let mut arch = toy_arch(vec![2]);
    arch.linear_attention = model_io::LinearAttentionConfig {
        num_k_heads: 1,
        num_v_heads: 1,
        key_head_dim: 8,
        value_head_dim: 8,
        conv_kernel_size: 4,
    };
    let cache = KvCacheManager::new(context.device(), &arch, 32, false, None, 8, None).unwrap();
    assert_eq!(cache.layer_kind(0), LayerKind::Linear);
    assert_eq!(cache.capacity(0), 0);
}

#[test]
fn reset_zeroes_the_position_cursor() {
    let context = MetalContext::new().unwrap();
    let arch = toy_arch(vec![1]);
    let mut cache = KvCacheManager::new(context.device(), &arch, 32, false, None, 8, None).unwrap();
    cache.advance_by(10);
    assert_eq!(cache.position(), 10);
    cache.reset();
    assert_eq!(cache.position(), 0);
}

#[test]
fn key_view_reports_valid_token_count_and_ring_start_slot() {
    let context = MetalContext::new().unwrap();
    let arch = toy_arch(vec![0]);
    let mut cache =
        KvCacheManager::new(context.device(), &arch, 1000, true, Some(4), 4, None).unwrap();
    // ring capacity = 4 + 4 = 8.
    cache.advance_by(10);
    let view = cache.key_view(0);
    assert_eq!(view.valid_token_count, 10);
    assert_eq!(view.start_slot, 10 % 8);
}

/// Static KV accounting for the real Gemma 4 shape at the CLI's 4K
/// default: the fp16 ring caps the 25 SWA layers at 1024 + 128 = 1152
/// rows while the 5 full layers stay linear at 4096. Total KV bytes are
/// 319,815,680 (~305 MiB), versus 922,746,880 (~880 MiB) if every layer
/// were linear -- the ~575 MiB the ring reclaims, matching the Swift
/// runner's budget (docs/SYSTEM_DESIGN.md's "FP16 KV at 4K" line).
#[test]
fn real_gemma4_shape_kv_bytes_match_swift_budget() {
    let context = MetalContext::new().unwrap();
    let arch = model_io::gemma4_26b_a4b();
    let cache = KvCacheManager::new(context.device(), &arch, 4096, true, None, 128, None).unwrap();

    let mut total = 0usize;
    for layer in 0..30 {
        if arch.full_attention_layer_mask[layer] == 0 {
            assert_eq!(cache.ring_capacity(layer), 1152);
            assert_eq!(cache.stride(layer), 8 * 256 * 2);
        } else {
            assert_eq!(cache.ring_capacity(layer), 0);
            assert_eq!(cache.capacity(layer), 4096);
            assert_eq!(cache.stride(layer), 2 * 512 * 2);
        }
        // K and V buffers per layer.
        total += 2 * cache.buffer_length(layer);
    }
    assert_eq!(total, 319_815_680);
}
