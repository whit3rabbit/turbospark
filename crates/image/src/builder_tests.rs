use super::*;
use crate::install::{ImageManifest, IMAGE_RECEIPT_NAME};
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn owners_follow_the_frozen_stage_boundary() {
    assert_eq!(
        super::builder_files::owner_for_path("components/text_encoder/index.json").unwrap(),
        "text_encoder_stage"
    );
    assert_eq!(
        super::builder_files::owner_for_path("components/transformer/tensors.bin").unwrap(),
        "transformer_stage"
    );
    assert_eq!(
        super::builder_files::owner_for_path("components/scheduler/config.json").unwrap(),
        "pipeline_control"
    );
    assert_eq!(
        super::builder_files::owner_for_path("components/vae_decoder/tensors.bin").unwrap(),
        "vae_stage"
    );
}

#[test]
fn files_outside_components_are_rejected() {
    let error = super::builder_files::owner_for_path("manifest.json")
        .expect_err("root file must not be a component payload");
    assert!(error.contains("outside components"));
}

#[test]
fn builds_and_reopens_a_complete_synthetic_install() {
    let root = temporary_directory();
    let source = root.join("source");
    let output = root.join("image.gturbo");
    fs::create_dir_all(source.join("tokenizer")).expect("create tokenizer source");
    fs::create_dir_all(source.join("scheduler")).expect("create scheduler source");
    fs::write(source.join("tokenizer/tokenizer.json"), b"tokenizer").expect("write tokenizer");
    fs::write(
        source.join("tokenizer/tokenizer_config.json"),
        b"{\"chat_template\": \"test\"}",
    )
    .expect("write tokenizer config");
    fs::write(
        source.join("scheduler/scheduler_config.json"),
        b"{\"shift\": 3.0}",
    )
    .expect("write scheduler");
    write_safetensors_source(&source.join("text_encoder"), TEXT_ENCODER_INDEX);
    write_safetensors_source(&source.join("transformer"), TRANSFORMER_INDEX);
    write_safetensors_source(&source.join("vae"), VAE_INDEX);
    fs::remove_file(source.join("vae").join(VAE_INDEX)).expect("remove optional VAE index");

    let report = build_image_install(&ImageInstallSpec {
        source_root: source,
        output_root: output.clone(),
        model_id: "Tongyi-MAI/Z-Image-Turbo".to_string(),
        model_revision: "test-revision".to_string(),
    })
    .expect("build image install");
    assert_eq!(report.output_root, output);
    assert_eq!(report.file_count, 9);
    assert!(report.total_bytes > 0);

    let manifest = ImageManifest::load(&output).expect("load image manifest");
    manifest
        .verify_files(&output)
        .expect("verify image install");
    assert_eq!(manifest.source["model_revision"], "test-revision");
    assert!(output.join("components/transformer/index.json").is_file());
    assert!(output.join("components/vae_decoder/tensors.bin").is_file());

    let receipt_path = output.join(IMAGE_RECEIPT_NAME);
    let receipt_bytes = fs::read(&receipt_path).expect("read image receipt");
    let mut receipt: serde_json::Value =
        serde_json::from_slice(&receipt_bytes).expect("parse image receipt");
    receipt["manifestSha256"] = serde_json::json!("00".repeat(32));
    fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&receipt).expect("serialize tampered receipt"),
    )
    .expect("write tampered receipt");
    let error = manifest
        .verify_files(&output)
        .expect_err("tampered receipt must be rejected");
    assert!(error.contains("manifest SHA mismatch"));
    fs::write(&receipt_path, receipt_bytes).expect("restore image receipt");
    manifest
        .verify_files(&output)
        .expect("restored image receipt");

    let error = build_image_install(&ImageInstallSpec {
        source_root: root.join("source"),
        output_root: output,
        model_id: "Tongyi-MAI/Z-Image-Turbo".to_string(),
        model_revision: "test-revision".to_string(),
    })
    .expect_err("existing install must not be overwritten");
    assert!(error.contains("refusing to overwrite"));
    fs::remove_dir_all(root).expect("remove test directory");
}

fn write_safetensors_source(directory: &Path, index_name: &str) {
    fs::create_dir_all(directory).expect("create source component");
    let source_file = index_name
        .strip_suffix(".index.json")
        .expect("indexed source file name");
    let linear = vec![0.25f32; 64];
    let norm = vec![1.0f32; 64];
    let mut payload = Vec::with_capacity((linear.len() + norm.len()) * 4);
    for value in linear.iter().chain(&norm) {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    let header = serde_json::json!({
        "linear.weight": {
            "dtype": "F32",
            "shape": [1, 64],
            "data_offsets": [0, 256]
        },
        "norm.weight": {
            "dtype": "F32",
            "shape": [64],
            "data_offsets": [256, 512]
        }
    });
    let header_bytes = serde_json::to_vec(&header).expect("serialize source header");
    let mut shard = Vec::with_capacity(8 + header_bytes.len() + payload.len());
    shard.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    shard.extend_from_slice(&header_bytes);
    shard.extend_from_slice(&payload);
    fs::write(directory.join(source_file), shard).expect("write source shard");
    fs::write(
        directory.join(index_name),
        serde_json::to_vec(&serde_json::json!({
            "weight_map": {
                "linear.weight": source_file,
                "norm.weight": source_file
            }
        }))
        .expect("serialize source index"),
    )
    .expect("write source index");
}

fn temporary_directory() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "turbospark-image-builder-{}-{nanos}",
        std::process::id()
    ));
    fs::create_dir_all(&root).expect("create test directory");
    root
}
