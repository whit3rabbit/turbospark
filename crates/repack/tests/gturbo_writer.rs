//! Round-trip test: assemble a synthetic `.gturbo` install with
//! `write_gturbo_install`, then read it back through every
//! `mrefrust_model_io` loader (manifest, packed-expert layout, resident
//! index) and the full-SHA256 verifier, confirming the writer's output
//! matches what those readers expect byte-for-byte.

use std::sync::atomic::{AtomicU64, Ordering};

use mrefrust_repack::{verify_install_full_sha256, ExpertBlob, LayerBlobs, SubTensor};

fn tempdir() -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "mrefrust-gturbo-writer-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}

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
        num_layers: 2,
        num_experts: 2,
        top_k_experts: 1,
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
    }
}

const EXPERT_STRIDE: u64 = 4096;
const EXPERTS_PER_LAYER: usize = 2;
const NUM_LAYERS: usize = 2;

fn synthetic_layers() -> Vec<LayerBlobs> {
    (0..NUM_LAYERS)
        .map(|layer| LayerBlobs {
            layer,
            experts: (0..EXPERTS_PER_LAYER)
                .map(|expert| ExpertBlob {
                    expert,
                    sub_tensors: vec![
                        SubTensor {
                            role: "gate".to_string(),
                            bytes: vec![(layer * 10 + expert) as u8; 64],
                            dtype: "int4".to_string(),
                            shape: vec![64, 64],
                        },
                        SubTensor {
                            role: "gate_scales".to_string(),
                            bytes: vec![0xAB, 0xCD],
                            dtype: "bf16".to_string(),
                            shape: vec![1],
                        },
                    ],
                })
                .collect(),
        })
        .collect()
}

#[test]
fn writer_output_round_trips_through_every_model_io_loader() {
    let dir = tempdir();
    let arch = toy_arch();
    mrefrust_repack::write_gturbo_install(
        &dir,
        &arch,
        "toy-model",
        EXPERT_STRIDE,
        EXPERTS_PER_LAYER,
        &synthetic_layers(),
        &[1, 2, 3, 4],
    )
    .unwrap();

    let manifest = model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES).unwrap();
    assert_eq!(manifest.magic, "GTURBO");
    assert_eq!(manifest.num_layers, NUM_LAYERS as i64);
    assert_eq!(manifest.expert_stride, EXPERT_STRIDE);

    let layout = model_io::load_packed_experts_layout(
        &dir,
        model_io::PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES,
    )
    .unwrap();
    assert_eq!(layout.expert_stride, EXPERT_STRIDE);
    let entry = layout.expert(1, 0);
    assert_eq!(entry.offset, 0);
    assert_eq!(entry.sub_tensors["gate"].size, 64);

    let weights_path = dir.join("model_weights.bin");
    let index = model_io::load_resident_index(&weights_path).unwrap();
    assert_eq!(index.header.entry_count, 0);
    assert_eq!(index.header.resident_size, 4);

    verify_install_full_sha256(&dir, &arch).unwrap();
}

#[test]
fn tampered_layer_file_fails_full_sha256_verification() {
    let dir = tempdir();
    let arch = toy_arch();
    mrefrust_repack::write_gturbo_install(
        &dir,
        &arch,
        "toy-model",
        EXPERT_STRIDE,
        EXPERTS_PER_LAYER,
        &synthetic_layers(),
        &[],
    )
    .unwrap();

    let layer_path = dir.join("packed_experts/layer_00.bin");
    let mut bytes = std::fs::read(&layer_path).unwrap();
    bytes[0] ^= 0xFF;
    std::fs::write(&layer_path, bytes).unwrap();

    let err = verify_install_full_sha256(&dir, &arch).unwrap_err();
    assert!(matches!(err, model_io::ModelError::ChecksumMismatch { .. }));
}

#[test]
fn rejects_expert_blob_that_overflows_the_stride() {
    let dir = tempdir();
    let arch = toy_arch();
    let mut layers = synthetic_layers();
    layers[0].experts[0].sub_tensors.push(SubTensor {
        role: "overflow".to_string(),
        bytes: vec![0u8; EXPERT_STRIDE as usize], // pushes this expert past the stride
        dtype: "int4".to_string(),
        shape: vec![1],
    });
    let err = mrefrust_repack::write_gturbo_install(
        &dir,
        &arch,
        "toy-model",
        EXPERT_STRIDE,
        EXPERTS_PER_LAYER,
        &layers,
        &[],
    )
    .unwrap_err();
    assert!(matches!(
        err,
        mrefrust_repack::WriterError::ExpertOversized { .. }
    ));
}
