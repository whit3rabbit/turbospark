#![cfg(target_os = "macos")]

use model_io::encoder_config::EncoderConfig;
use turbospark_runtime::encoder::weights::{EncoderLayerWeightsOwned, EncoderWeights};
use turbospark_runtime::{cosine_similarity, EncoderRunner};

#[test]
fn test_encoder_runner_synthetic() {
    let hidden_size = 16;
    let num_heads = 4;
    let intermediate_size = 32;
    let vocab_size = 100;
    let max_positions = 64;

    let config = EncoderConfig {
        model_type: "bert".to_string(),
        architectures: vec!["BertModel".to_string()],
        hidden_size,
        num_hidden_layers: 2,
        num_attention_heads: num_heads,
        intermediate_size,
        max_position_embeddings: max_positions,
        vocab_size,
        type_vocab_size: 2,
        pad_token_id: 0,
        layer_norm_eps: 1e-5,
        quantization: None,
    };

    // Synthetic embedding tables with varying values across hidden dim
    let mut word_embeddings = Vec::with_capacity(vocab_size * hidden_size);
    for v in 0..vocab_size {
        for h in 0..hidden_size {
            word_embeddings.push(((v * 17 + h * 31) % 100) as f32 * 0.01);
        }
    }
    let mut position_embeddings = Vec::with_capacity(max_positions * hidden_size);
    for p in 0..max_positions {
        for h in 0..hidden_size {
            position_embeddings.push(((p * 13 + h * 7) % 50) as f32 * 0.005);
        }
    }
    let token_type_embeddings = Some(vec![0.0f32; 2 * hidden_size]);
    let emb_ln_weight = vec![1.0f32; hidden_size];
    let emb_ln_bias = vec![0.0f32; hidden_size];

    let mut layers = Vec::new();
    for _ in 0..2 {
        layers.push(EncoderLayerWeightsOwned {
            q_weight: vec![0.05f32; hidden_size * hidden_size],
            q_bias: vec![0.0f32; hidden_size],
            k_weight: vec![0.05f32; hidden_size * hidden_size],
            k_bias: vec![0.0f32; hidden_size],
            v_weight: vec![0.05f32; hidden_size * hidden_size],
            v_bias: vec![0.0f32; hidden_size],
            out_weight: vec![0.05f32; hidden_size * hidden_size],
            out_bias: vec![0.0f32; hidden_size],
            attn_ln_weight: vec![1.0f32; hidden_size],
            attn_ln_bias: vec![0.0f32; hidden_size],
            intermediate_weight: vec![0.05f32; intermediate_size * hidden_size],
            intermediate_bias: vec![0.0f32; intermediate_size],
            mlp_out_weight: vec![0.05f32; hidden_size * intermediate_size],
            mlp_out_bias: vec![0.0f32; hidden_size],
            mlp_ln_weight: vec![1.0f32; hidden_size],
            mlp_ln_bias: vec![0.0f32; hidden_size],
        });
    }

    let weights = EncoderWeights {
        word_embeddings,
        position_embeddings,
        token_type_embeddings,
        emb_ln_weight,
        emb_ln_bias,
        layers,
    };

    let runner = EncoderRunner::from_parts(config, weights);

    // Encode single sequence
    let tokens_a = vec![1, 10, 20, 30, 2]; // [CLS], text..., [SEP]
    let emb_a = runner
        .encode_tokens(&tokens_a, None)
        .expect("encode tokens_a");
    assert_eq!(emb_a.len(), hidden_size);

    // L2 norm must be 1.0
    let norm_sq: f32 = emb_a.iter().map(|&v| v * v).sum();
    assert!(
        (norm_sq - 1.0).abs() < 1e-5,
        "embedding L2 norm must be 1.0, got {norm_sq}"
    );

    // Cosine similarity to self is 1.0
    let sim_self = cosine_similarity(&emb_a, &emb_a);
    assert!((sim_self - 1.0).abs() < 1e-5, "self similarity must be 1.0");

    // Encode different sequence
    let tokens_b = vec![1, 55, 2];
    let emb_b = runner
        .encode_tokens(&tokens_b, None)
        .expect("encode tokens_b");
    assert_eq!(emb_b.len(), hidden_size);

    // Batch encode matches individual encode
    let batch = vec![tokens_a.clone(), tokens_b.clone()];
    let batch_res = runner.encode_batch_tokens(&batch).expect("batch encode");
    assert_eq!(batch_res.len(), 2);
    assert_eq!(batch_res[0], emb_a);
    assert_eq!(batch_res[1], emb_b);
}
