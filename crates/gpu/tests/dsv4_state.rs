//! Tests for `Dsv4StateManager` against real Metal buffer allocations:
//! CSA/HCA/full-layer classification, window-ring position math, and
//! counters reset.
#![cfg(target_os = "macos")]

use turbospark_gpu::{Dsv4StateManager, MetalContext};

fn toy_dsv4_arch() -> model_io::ArchConfig {
    model_io::ArchConfig {
        hidden_size: 64,
        intermediate_size: 64,
        moe_intermediate_size: 64,
        num_heads: 2,
        num_kv_heads: 2,
        num_full_kv_heads: 1,
        head_dim: 8,
        full_head_dim: 8,
        vocab_size: 100,
        sliding_window: 4,
        final_logit_softcap: 0.0,
        rope_theta: 10000.0,
        full_rope_theta: 10000.0,
        partial_rotary_factor: 1.0,
        num_layers: 3,
        num_experts: 1,
        top_k_experts: 1,
        tie_word_embeddings: false,
        attention_k_eq_v: true,
        full_attention_layer_mask: vec![3, 4, 1], // CSA, HCA, full
        hidden_activation: "silu".to_string(),
        family: model_io::ModelFamily::DeepseekV4Flash,
        attn_output_gate: false,
        attention_scale: 1.0,
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: false,
        rope_neox_subdim: false,
        linear_attention: model_io::LinearAttentionConfig::NONE,
        compressed_attention: model_io::CompressedAttentionConfig {
            q_lora_rank: 16,
            o_lora_rank: 16,
            o_groups: 2,
            rope_head_dim: 8,
            index_n_heads: 2,
            index_head_dim: 4,
            index_top_k: 8,
            csa_compress_rate: 2,
            hca_compress_rate: 2,
            compress_rope_theta: 10000.0,
            rope_scaling_factor: 1.0,
            rope_scaling_original_max: 100,
            rope_scaling_beta_fast: 1.0,
            rope_scaling_beta_slow: 1.0,
        },
        hyper_connections: model_io::HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "sqrtsoftplus".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: model_io::RopeScalingConfig::NONE,
    }
}

#[test]
fn classifies_csa_hca_and_full_layers() {
    let context = MetalContext::new().unwrap();
    let arch = toy_dsv4_arch();
    let manager = Dsv4StateManager::new(context.device(), &arch, 64);

    assert!(manager.is_csa_or_hca(0)); // CSA
    assert!(manager.is_csa_or_hca(1)); // HCA
    assert!(!manager.is_csa_or_hca(2)); // full attention
}

#[test]
#[should_panic(expected = "not CSA/HCA")]
fn compressed_buffer_panics_for_a_full_attention_layer() {
    let context = MetalContext::new().unwrap();
    let arch = toy_dsv4_arch();
    let manager = Dsv4StateManager::new(context.device(), &arch, 64);
    let _ = manager.compressed_buffer(2);
}

#[test]
#[should_panic(expected = "not CSA")]
fn indexer_keys_panics_for_an_hca_layer() {
    let context = MetalContext::new().unwrap();
    let arch = toy_dsv4_arch();
    let manager = Dsv4StateManager::new(context.device(), &arch, 64);
    let _ = manager.indexer_keys_buffer(1);
}

#[test]
#[should_panic(expected = "at least one CSA/HCA layer")]
fn rejects_architectures_without_compressed_attention_layers() {
    let context = MetalContext::new().unwrap();
    let mut arch = toy_dsv4_arch();
    arch.full_attention_layer_mask = vec![1, 1, 1];
    let _ = Dsv4StateManager::new(context.device(), &arch, 64);
}

#[test]
fn window_ring_position_math() {
    let context = MetalContext::new().unwrap();
    let arch = toy_dsv4_arch();
    let manager = Dsv4StateManager::new(context.device(), &arch, 64);

    // ring_capacity == sliding_window == 4.
    assert_eq!(manager.window_slot(0), 0);
    assert_eq!(manager.window_slot(5), 1);
    assert_eq!(manager.window_count(0), 1);
    assert_eq!(manager.window_count(10), 4);
    assert_eq!(manager.window_start_position(0), 0);
    assert_eq!(manager.window_start_position(10), 7);
}

#[test]
fn reset_clears_counters() {
    let context = MetalContext::new().unwrap();
    let arch = toy_dsv4_arch();
    let mut manager = Dsv4StateManager::new(context.device(), &arch, 64);
    manager.counters[0].tokens = 5;
    manager.counters[0].has_prior = true;
    manager.reset();
    assert_eq!(manager.counters[0].tokens, 0);
    assert!(!manager.counters[0].has_prior);
}
