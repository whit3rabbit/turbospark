//! Tests for `GdnStateManager` against real Metal buffer allocations: state
//! sizing, per-layer linear-vs-not classification, and zero-reset.
#![cfg(target_os = "macos")]

use mrefrust_gpu::{GdnStateManager, MetalContext};

fn qwen_style_arch() -> model_io::ArchConfig {
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
        rope_theta: 10000.0,
        full_rope_theta: 10000.0,
        partial_rotary_factor: 1.0,
        num_layers: 3,
        num_experts: 1,
        top_k_experts: 1,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: vec![2, 2, 1], // two linear, one full
        hidden_activation: "silu".to_string(),
        family: model_io::ModelFamily::Qwen36,
        attn_output_gate: true,
        attention_scale: 1.0,
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
        },
        compressed_attention: model_io::CompressedAttentionConfig::NONE,
        hyper_connections: model_io::HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
    }
}

#[test]
fn classifies_linear_layers_and_sizes_state() {
    let context = MetalContext::new().unwrap();
    let arch = qwen_style_arch();
    let manager = GdnStateManager::new(context.device(), &arch);

    assert!(manager.is_linear(0));
    assert!(manager.is_linear(1));
    assert!(!manager.is_linear(2));

    // num_v_heads(4) * value_head_dim(8) * key_head_dim(8) * 4 bytes (FP32).
    assert_eq!(manager.state_bytes_per_layer, 4 * 8 * 8 * 4);
    // (conv_kernel_size - 1)(3) * qkv_dim * 2 bytes (FP16).
    let qkv_dim = 2 * 4 * 8 + 4 * 8; // 2*numKHeads*keyHeadDim + numVHeads*valueHeadDim
    assert_eq!(manager.conv_tail_bytes_per_layer, 3 * qkv_dim * 2);
}

#[test]
#[should_panic(expected = "not a linear-attention layer")]
fn state_buffer_panics_for_a_non_linear_layer() {
    let context = MetalContext::new().unwrap();
    let arch = qwen_style_arch();
    let manager = GdnStateManager::new(context.device(), &arch);
    let _ = manager.state_buffer(2);
}

#[test]
fn state_and_conv_tail_start_and_reset_to_zero() {
    let context = MetalContext::new().unwrap();
    let arch = qwen_style_arch();
    let mut manager = GdnStateManager::new(context.device(), &arch);

    let state = manager.state_buffer(0);
    let bytes = unsafe_read_all(state);
    assert!(bytes.iter().all(|&b| b == 0));

    // Mutate, then confirm reset() zeroes it again.
    unsafe_write_first_byte(state, 0xFF);
    manager.reset();
    let bytes = unsafe_read_all(manager.state_buffer(0));
    assert!(bytes.iter().all(|&b| b == 0));
}

fn unsafe_read_all(buffer: &metal::Buffer) -> Vec<u8> {
    let len = buffer.length() as usize;
    let ptr = buffer.contents() as *const u8;
    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
}

fn unsafe_write_first_byte(buffer: &metal::Buffer, value: u8) {
    let ptr = buffer.contents() as *mut u8;
    unsafe { *ptr = value };
}
