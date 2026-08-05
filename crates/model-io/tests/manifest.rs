//! Tests for manifest.json decode + validation against a resolved
//! `ArchConfig`, using a synthetic (non-Gemma-sized) toy architecture so no
//! `quant` block is required.

use std::io::Write;

use mrefrust_model_io::{
    load_manifest, peek_family, LinearAttentionConfig, ModelError, ModelFamily,
};

fn toy_arch() -> mrefrust_model_io::ArchConfig {
    mrefrust_model_io::ArchConfig {
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
        num_layers: 2,
        num_experts: 4,
        top_k_experts: 2,
        tie_word_embeddings: true,
        attention_k_eq_v: true,
        full_attention_layer_mask: vec![1, 1],
        hidden_activation: "gelu_pytorch_tanh".to_string(),
        family: ModelFamily::Gemma4,
        attn_output_gate: false,
        attention_scale: 1.0,
        embedding_scaled_by_sqrt_hidden: true,
        router_scaled: true,
        ffn_sandwich_norms: true,
        shared_expert_gated: false,
        rope_neox_subdim: false,
        linear_attention: LinearAttentionConfig::NONE,
        compressed_attention: mrefrust_model_io::CompressedAttentionConfig::NONE,
        hyper_connections: mrefrust_model_io::HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
    }
}

fn write_manifest(dir: &std::path::Path, json: &str) {
    let mut f = std::fs::File::create(dir.join("manifest.json")).unwrap();
    f.write_all(json.as_bytes()).unwrap();
}

fn toy_manifest_json() -> String {
    r#"{
        "magic": "GTURBO",
        "versionMajor": 1,
        "versionMinor": 0,
        "flags": {},
        "modelID": "toy",
        "sourceSnapshotHash": null,
        "arch": {
            "hiddenSize": 64,
            "ffnIntermediate": 128,
            "moeIntermediateSize": 64,
            "numHeads": 2,
            "numKVHeads": 1,
            "numFullKVHeads": 1,
            "headDim": 32,
            "fullHeadDim": 32,
            "vocabSize": 100,
            "slidingWindow": 16,
            "finalLogitSoftcap": 0.0,
            "ropeTheta": 10000.0,
            "fullRopeTheta": 10000.0,
            "partialRotaryFactor": 1.0,
            "numLayers": 2,
            "numExperts": 4,
            "topKExperts": 2,
            "tieWordEmbeddings": true,
            "attentionKEqV": true,
            "hiddenActivation": "gelu_pytorch_tanh",
            "fullAttentionLayerMask": [1, 1]
        },
        "quant": null,
        "files": {
            "model_weights.bin": {"size": 1, "sha256": "a"},
            "packed_experts/layout.json": {"size": 1, "sha256": "b"},
            "packed_experts/layer_00.bin": {"size": 1, "sha256": "c"},
            "packed_experts/layer_01.bin": {"size": 1, "sha256": "d"}
        },
        "expertsPerLayer": 4,
        "numLayers": 2,
        "expertStride": 4096
    }"#
    .to_string()
}

#[test]
fn load_succeeds_for_a_matching_toy_manifest() {
    let dir = tempfile_dir();
    write_manifest(dir.path(), &toy_manifest_json());
    let manifest = load_manifest(dir.path(), &toy_arch(), 4 * 1024 * 1024).unwrap();
    assert_eq!(manifest.magic, "GTURBO");
    assert_eq!(manifest.num_layers, 2);
}

#[test]
fn load_rejects_hidden_size_mismatch() {
    let dir = tempfile_dir();
    write_manifest(dir.path(), &toy_manifest_json());
    let mut arch = toy_arch();
    arch.hidden_size = 999;
    let err = load_manifest(dir.path(), &arch, 4 * 1024 * 1024).unwrap_err();
    match err {
        ModelError::ArchMismatch { field, .. } => assert_eq!(field, "hiddenSize"),
        other => panic!("expected ArchMismatch, got {other:?}"),
    }
}

#[test]
fn load_rejects_missing_manifest_file() {
    let dir = tempfile_dir();
    let err = load_manifest(dir.path(), &toy_arch(), 4 * 1024 * 1024).unwrap_err();
    assert!(matches!(err, ModelError::PartialInstall { .. }));
}

#[test]
fn load_rejects_unknown_flag() {
    let dir = tempfile_dir();
    let json = toy_manifest_json().replacen("\"flags\": {}", "\"flags\": {\"bogus\": true}", 1);
    write_manifest(dir.path(), &json);
    let err = load_manifest(dir.path(), &toy_arch(), 4 * 1024 * 1024).unwrap_err();
    assert!(matches!(err, ModelError::UnknownFlag { .. }));
}

#[test]
fn load_rejects_missing_layer_file_entry() {
    let dir = tempfile_dir();
    let json = toy_manifest_json().replace(
        "\"packed_experts/layer_01.bin\": {\"size\": 1, \"sha256\": \"d\"}",
        "\"unused\": {\"size\": 1, \"sha256\": \"d\"}",
    );
    write_manifest(dir.path(), &json);
    let err = load_manifest(dir.path(), &toy_arch(), 4 * 1024 * 1024).unwrap_err();
    assert!(matches!(err, ModelError::MissingFile { .. }));
}

#[test]
fn peek_family_defaults_to_gemma_when_family_field_is_absent() {
    let dir = tempfile_dir();
    write_manifest(dir.path(), &toy_manifest_json());
    let family = peek_family(dir.path(), 4 * 1024 * 1024).unwrap();
    assert_eq!(family, ModelFamily::Gemma4);
}

fn tempfile_dir() -> TempDir {
    TempDir::new()
}

/// Minimal owned-directory helper so this test crate doesn't need an extra
/// `tempfile` dependency for a handful of throwaway directories.
struct TempDir {
    path: std::path::PathBuf,
}

impl TempDir {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let unique_counter = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        let unique = format!(
            "mrefrust-model-io-test-{}-{}-{unique_counter}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        path.push(unique);
        std::fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
