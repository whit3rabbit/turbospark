//! Tests for full-SHA256 install verification, using a synthetic toy
//! manifest (same shape as `mrefrust-model-io`'s own manifest tests) plus
//! real on-disk files so the hasher has actual bytes to read.

use std::sync::atomic::{AtomicU64, Ordering};

use mrefrust_repack::verify_install_full_sha256;

fn toy_arch() -> model_io::ArchConfig {
    model_io::ArchConfig {
        hidden_size: 64,
        intermediate_size: 128,
        moe_intermediate_size: 64,
        num_heads: 2,
        num_kv_heads: 1,
        num_full_kv_heads: 1,
        head_dim: 32,
        full_head_dim: 32,
        vocab_size: 100,
        sliding_window: 16,
        final_logit_softcap: 0.0,
        rope_theta: 10000.0,
        full_rope_theta: 10000.0,
        partial_rotary_factor: 1.0,
        num_layers: 1,
        num_experts: 4,
        top_k_experts: 2,
        tie_word_embeddings: true,
        attention_k_eq_v: true,
        full_attention_layer_mask: vec![1],
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

fn tempdir() -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "mrefrust-repack-verify-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn write_install(dir: &std::path::Path, weights_content: &[u8]) {
    std::fs::create_dir_all(dir.join("packed_experts")).unwrap();
    std::fs::write(dir.join("model_weights.bin"), weights_content).unwrap();
    std::fs::write(dir.join("packed_experts/layout.json"), b"{}").unwrap();
    std::fs::write(dir.join("packed_experts/layer_00.bin"), b"expert-bytes").unwrap();

    let weights_hash = sha256_hex(weights_content);
    let layout_hash = sha256_hex(b"{}");
    let layer_hash = sha256_hex(b"expert-bytes");

    let manifest_json = format!(
        r#"{{
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": {{}},
            "modelID": "toy",
            "sourceSnapshotHash": null,
            "arch": {{
                "hiddenSize": 64, "ffnIntermediate": 128, "moeIntermediateSize": 64,
                "numHeads": 2, "numKVHeads": 1, "numFullKVHeads": 1,
                "headDim": 32, "fullHeadDim": 32, "vocabSize": 100,
                "slidingWindow": 16, "finalLogitSoftcap": 0.0,
                "ropeTheta": 10000.0, "fullRopeTheta": 10000.0,
                "partialRotaryFactor": 1.0, "numLayers": 1, "numExperts": 4,
                "topKExperts": 2, "tieWordEmbeddings": true, "attentionKEqV": true,
                "hiddenActivation": "gelu_pytorch_tanh", "fullAttentionLayerMask": [1]
            }},
            "quant": null,
            "files": {{
                "model_weights.bin": {{"size": {}, "sha256": "{weights_hash}"}},
                "packed_experts/layout.json": {{"size": 2, "sha256": "{layout_hash}"}},
                "packed_experts/layer_00.bin": {{"size": 12, "sha256": "{layer_hash}"}}
            }},
            "expertsPerLayer": 4,
            "numLayers": 1,
            "expertStride": 4096
        }}"#,
        weights_content.len()
    );
    std::fs::write(dir.join("manifest.json"), manifest_json).unwrap();
}

fn sha256_hex(data: &[u8]) -> String {
    model_io::hash_data(data)
}

#[test]
fn verify_install_full_sha256_passes_for_a_matching_install() {
    let dir = tempdir();
    write_install(&dir, b"model weight bytes");
    verify_install_full_sha256(&dir, &toy_arch()).unwrap();
}

#[test]
fn verify_install_full_sha256_fails_when_a_file_is_tampered() {
    let dir = tempdir();
    write_install(&dir, b"model weight bytes");
    std::fs::write(dir.join("model_weights.bin"), b"tampered!!!").unwrap();
    let err = verify_install_full_sha256(&dir, &toy_arch()).unwrap_err();
    assert!(matches!(err, model_io::ModelError::ChecksumMismatch { .. }));
}
