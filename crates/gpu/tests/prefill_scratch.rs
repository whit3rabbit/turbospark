//! Tests for `PrefillChunkScratchLayout`'s pure sizing arithmetic and
//! `PrefillChunkScratchBuffers`'s real Metal buffer allocation.
#![cfg(target_os = "macos")]

use turbospark_gpu::{MetalContext, PrefillChunkScratchBuffers, PrefillChunkScratchLayout};

fn dense_arch() -> model_io::ArchConfig {
    model_io::ArchConfig {
        hidden_size: 64,
        intermediate_size: 64,
        moe_intermediate_size: 0,
        num_heads: 2,
        num_kv_heads: 2,
        num_full_kv_heads: 2,
        head_dim: 32,
        full_head_dim: 32,
        vocab_size: 100,
        sliding_window: 0,
        final_logit_softcap: 30.0,
        rope_theta: 10000.0,
        full_rope_theta: 10000.0,
        partial_rotary_factor: 1.0,
        num_layers: 2,
        num_experts: 0,
        top_k_experts: 0,
        tie_word_embeddings: true,
        attention_k_eq_v: true,
        full_attention_layer_mask: vec![1, 1],
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
        rope_scaling: model_io::RopeScalingConfig::NONE,
    }
}

#[test]
fn layout_sizes_scale_with_chunk_tokens() {
    let arch = dense_arch();
    let layout = PrefillChunkScratchLayout::new(&arch, 8, 32);
    assert_eq!(layout.chunk_tokens, 8);
    assert_eq!(layout.hidden_elements(), 8 * 64);
    assert_eq!(layout.q_elements(), 8 * layout.q_proj_elements_per_token);
    // Dense (no MoE): routed_intermediate is 0, so route_partial_elements
    // and the routed gate/up/down scratch collapse to zero.
    assert_eq!(layout.route_partial_elements(), 0);
    assert_eq!(layout.routed_gate_up_act_elements(), 0);
}

#[test]
fn chunk_tokens_is_clamped_to_the_valid_range() {
    let arch = dense_arch();
    let too_large = PrefillChunkScratchLayout::new(&arch, 999_999, 32);
    assert_eq!(
        too_large.chunk_tokens,
        PrefillChunkScratchLayout::MAX_CHUNK_TOKENS
    );
    let zero = PrefillChunkScratchLayout::new(&arch, 0, 32);
    assert_eq!(zero.chunk_tokens, 1);
}

#[test]
fn total_persistent_bytes_is_the_sum_of_the_two_regions() {
    let arch = dense_arch();
    let layout = PrefillChunkScratchLayout::new(&arch, 16, 32);
    assert_eq!(
        layout.total_persistent_bytes(),
        layout.device_private_bytes() + layout.shared_metadata_bytes()
    );
}

#[test]
fn allocates_one_real_buffer_per_scratch_field() {
    let context = MetalContext::new().unwrap();
    let arch = dense_arch();
    let layout = PrefillChunkScratchLayout::new(&arch, 8, 32);
    let buffers = PrefillChunkScratchBuffers::allocate(context.device(), layout);

    assert_eq!(
        buffers.hidden.length() as usize,
        layout.hidden_elements() * 2
    );
    assert_eq!(buffers.q.length() as usize, layout.q_elements() * 2);
    // Placeholder-sized (floored at one element) fields for this dense,
    // non-gated, non-linear architecture.
    assert_eq!(buffers.attn_q.length(), 2);
    assert_eq!(buffers.gdn_conv_out.length(), 2);
}
