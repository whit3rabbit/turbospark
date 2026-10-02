//! Tests for `PrefillChunkScratchLayout`'s pure sizing arithmetic and
//! `PrefillChunkScratchBuffers`'s real Metal buffer allocation.
#![cfg(target_os = "macos")]

use turbospark_gpu::{
    BatchAttentionScratchLayout, BatchAttentionScratchLayoutError, MetalContext,
    PrefillChunkScratchBuffers, PrefillChunkScratchLayout, MAX_BATCH_ROWS,
};

#[test]
fn batch_attention_layout_accounts_for_every_capacity_buffer() {
    const NUM_Q_HEADS: usize = 8;
    for head_dim in [32, 64, 128, 256, 512] {
        let layout = BatchAttentionScratchLayout::new(NUM_Q_HEADS, head_dim).unwrap();
        let capacity = MAX_BATCH_ROWS;
        let q_elements = capacity * NUM_Q_HEADS * head_dim;
        let row_plan_bytes = capacity * 4 * std::mem::size_of::<u32>();
        let partial_bytes =
            capacity * NUM_Q_HEADS * 16 * (2 + head_dim) * std::mem::size_of::<f32>();
        let expected_total = capacity * NUM_Q_HEADS * head_dim * 4
            + capacity * 16
            + capacity * NUM_Q_HEADS * 16 * (2 + head_dim) * 4;

        assert_eq!(layout.capacity(), capacity);
        assert_eq!(layout.num_q_heads(), NUM_Q_HEADS);
        assert_eq!(layout.head_dim(), head_dim);
        assert_eq!(layout.max_chunks(), 16);
        assert_eq!(
            layout.q_buffer_bytes(),
            q_elements * std::mem::size_of::<u16>()
        );
        assert_eq!(
            layout.output_buffer_bytes(),
            q_elements * std::mem::size_of::<u16>()
        );
        assert_eq!(layout.row_plan_bytes(), row_plan_bytes);
        assert_eq!(layout.partial_state_bytes(), partial_bytes);
        assert_eq!(
            q_elements * 2 * std::mem::size_of::<u16>() + row_plan_bytes + partial_bytes,
            expected_total
        );
        assert_eq!(layout.total_bytes(), expected_total);
    }
}

#[test]
fn batch_attention_layout_rejects_invalid_head_dimensions() {
    assert_eq!(
        BatchAttentionScratchLayout::new(0, 32),
        Err(BatchAttentionScratchLayoutError::ZeroQueryHeads)
    );
    assert_eq!(
        BatchAttentionScratchLayout::new(1, 0),
        Err(BatchAttentionScratchLayoutError::InvalidHeadDimension)
    );
    assert_eq!(
        BatchAttentionScratchLayout::new(1, 513),
        Err(BatchAttentionScratchLayoutError::InvalidHeadDimension)
    );
}

#[test]
fn batch_attention_layout_refuses_size_overflow() {
    assert_eq!(
        BatchAttentionScratchLayout::new(usize::MAX, 512),
        Err(BatchAttentionScratchLayoutError::SizeOverflow)
    );
}

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
        dense_lead_intermediate_size: 0,
        num_dense_leading_layers: 0,
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
        mla: model_io::MlaConfig::NONE,
        compressed_attention: model_io::CompressedAttentionConfig::NONE,
        hyper_connections: model_io::HyperConnectionConfig::NONE,
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
