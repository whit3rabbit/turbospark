use turbospark_compute::encoder::{
    cls_pool_and_normalize, cosine_similarity, encoder_block_forward, encoder_embeddings_lookup,
    EncoderLayerWeights, EncoderReferenceConfig,
};

#[test]
fn test_cls_pool_and_normalize() {
    let hidden_size = 4;
    let hidden_states = vec![
        3.0, 4.0, 0.0, 0.0, // token 0 (norm = 5.0)
        1.0, 1.0, 1.0, 1.0, // token 1
    ];
    let pooled = cls_pool_and_normalize(&hidden_states, hidden_size);
    assert_eq!(pooled.len(), 4);
    assert!((pooled[0] - 0.6).abs() < 1e-6);
    assert!((pooled[1] - 0.8).abs() < 1e-6);
    assert!((pooled[2] - 0.0).abs() < 1e-6);
    assert!((pooled[3] - 0.0).abs() < 1e-6);

    // L2 norm of output must be 1.0
    let norm_sq: f32 = pooled.iter().map(|&v| v * v).sum();
    assert!((norm_sq - 1.0).abs() < 1e-6);
}

#[test]
fn test_cosine_similarity() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let c = vec![0.0, 1.0, 0.0];
    let d = vec![-1.0, 0.0, 0.0];

    assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    assert!((cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);
    assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 1e-6);

    // The function normalizes its inputs, so scale does not move a cosine
    // and an arbitrary vector gets an honest angle rather than a raw dot.
    let scaled = vec![7.0, 0.0, 0.0];
    let diagonal = vec![2.0, 2.0, 0.0];
    assert!((cosine_similarity(&scaled, &a) - 1.0).abs() < 1e-6);
    assert!((cosine_similarity(&scaled, &c) - 0.0).abs() < 1e-6);
    assert!((cosine_similarity(&diagonal, &a) - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);

    // A zero vector has no direction.
    assert_eq!(cosine_similarity(&a, &[0.0, 0.0, 0.0]), 0.0);
}

#[test]
fn test_encoder_embeddings_and_block_forward() {
    let config = EncoderReferenceConfig {
        hidden_size: 4,
        num_attention_heads: 2,
        intermediate_size: 8,
        layer_norm_eps: 1e-5,
        position_offset: 0,
        use_tanh_gelu: false,
    };
    assert_eq!(config.head_dim(), 2);

    let input_ids = vec![0, 1];
    let word_emb = vec![
        0.1, 0.2, 0.3, 0.4, // token 0
        0.5, 0.6, 0.7, 0.8, // token 1
    ];
    let pos_emb = vec![
        0.01, 0.02, 0.03, 0.04, // pos 0
        0.05, 0.06, 0.07, 0.08, // pos 1
    ];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];

    let emb = encoder_embeddings_lookup(
        &input_ids, None, &word_emb, &pos_emb, None, &gamma, &beta, &config,
    );
    assert_eq!(emb.len(), 8);

    // Identity-like layer weights
    let eye4 = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let zero4 = vec![0.0; 4];
    let w_inter = vec![0.1; 4 * 8];
    let b_inter = vec![0.0; 8];
    let w_out = vec![0.1; 8 * 4];
    let b_out = vec![0.0; 4];

    let weights = EncoderLayerWeights {
        q_weight: &eye4,
        q_bias: &zero4,
        k_weight: &eye4,
        k_bias: &zero4,
        v_weight: &eye4,
        v_bias: &zero4,
        out_weight: &eye4,
        out_bias: &zero4,
        attn_ln_weight: &gamma,
        attn_ln_bias: &beta,
        intermediate_weight: &w_inter,
        intermediate_bias: &b_inter,
        mlp_out_weight: &w_out,
        mlp_out_bias: &b_out,
        mlp_ln_weight: &gamma,
        mlp_ln_bias: &beta,
    };

    let out = encoder_block_forward(&emb, 2, &weights, &config);
    assert_eq!(out.len(), 8);

    // Each row output must be finite and normalized by LayerNorm (mean ~ 0)
    for row in 0..2 {
        let row_slice = &out[row * 4..(row + 1) * 4];
        let mean: f32 = row_slice.iter().sum::<f32>() / 4.0;
        assert!(
            mean.abs() < 1e-5,
            "LayerNorm output mean should be ~0, got {mean}"
        );
    }

    let pooled = cls_pool_and_normalize(&out, 4);
    assert_eq!(pooled.len(), 4);
    let norm_sq: f32 = pooled.iter().map(|&v| v * v).sum();
    assert!((norm_sq - 1.0).abs() < 1e-5);
}
