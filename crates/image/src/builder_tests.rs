use super::*;
use crate::install::{ImageManifest, IMAGE_RECEIPT_NAME};
use crate::packed::{PackedTensorStore, MLX_AFFINE_BITS};
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

#[test]
fn builds_all_mlx_widths_through_complete_install() {
    for bits in MLX_AFFINE_BITS {
        let root = temporary_directory();
        let source = root.join("source");
        let output = root.join("image.gturbo");
        write_common_source_tree(&source);
        write_mlx_transformer_source(&source.join("transformer"), TRANSFORMER_INDEX, bits);
        write_safetensors_source(&source.join("text_encoder"), TEXT_ENCODER_INDEX);
        write_safetensors_source(&source.join("vae"), VAE_INDEX);
        fs::remove_file(source.join("vae").join(VAE_INDEX)).expect("remove optional VAE index");

        build_image_install(&ImageInstallSpec {
            source_root: source,
            output_root: output.clone(),
            model_id: "andrevp/Z-Image-Turbo-MLX-test".to_string(),
            model_revision: format!("synthetic-{bits}bit"),
        })
        .unwrap_or_else(|error| panic!("build MLX {bits}-bit image install: {error}"));

        let transformer =
            PackedTensorStore::open(&output.join("components/transformer")).expect("open MLX");
        let tensor = transformer
            .tensor("linear.weight")
            .expect("MLX tensor in packed transformer");
        assert_eq!(tensor.storage_dtype, "MLX_AFFINE");
        assert_eq!(tensor.shape, [1, 64]);
        assert_eq!(
            tensor.quantization.as_ref().expect("MLX metadata").bits,
            bits
        );
        let manifest = ImageManifest::load(&output).expect("load MLX manifest");
        assert_eq!(
            manifest.components["transformer"].metadata["observed_affine_bit_widths"],
            serde_json::json!([bits])
        );
        assert_eq!(
            manifest.components["transformer"].metadata["quantization_label"],
            serde_json::json!(format!("mlx-affine-linear-weights-group-64-bits-{bits}"))
        );
        assert_eq!(
            transformer
                .load_row("linear.weight", 0)
                .expect("decode MLX row")
                .len(),
            64
        );
        manifest
            .verify_files(&output)
            .expect("verify MLX image install");
        fs::remove_dir_all(root).expect("remove test directory");
    }
}

#[test]
fn builds_fp16_source_through_complete_install() {
    let root = temporary_directory();
    let source = root.join("source");
    let output = root.join("image.gturbo");
    write_common_source_tree(&source);
    write_fp16_transformer_source(&source.join("transformer"), TRANSFORMER_INDEX);
    write_safetensors_source(&source.join("text_encoder"), TEXT_ENCODER_INDEX);
    write_safetensors_source(&source.join("vae"), VAE_INDEX);
    fs::remove_file(source.join("vae").join(VAE_INDEX)).expect("remove optional VAE index");

    build_image_install(&ImageInstallSpec {
        source_root: source,
        output_root: output.clone(),
        model_id: "andrevp/Z-Image-Turbo-MLX".to_string(),
        model_revision: "synthetic-fp16".to_string(),
    })
    .expect("build F16 image install");

    let transformer =
        PackedTensorStore::open(&output.join("components/transformer")).expect("open F16");
    assert_eq!(
        transformer
            .tensor("linear.weight")
            .expect("F16 tensor in packed transformer")
            .storage_dtype,
        "F32"
    );
    let manifest = ImageManifest::load(&output).expect("load F16 manifest");
    assert_eq!(
        manifest.components["transformer"].metadata["observed_affine_bit_widths"],
        serde_json::json!([])
    );
    assert_eq!(
        manifest.components["transformer"].metadata["quantization_scheme"],
        serde_json::json!("unquantized")
    );
    assert_eq!(
        manifest.components["transformer"].metadata["quantization_label"],
        serde_json::json!("unquantized")
    );
    assert!(transformer
        .load_row("linear.weight", 0)
        .expect("decode F16 source row")
        .iter()
        .all(|value| (*value - 1.5).abs() < 0.001));
    manifest
        .verify_files(&output)
        .expect("verify F16 image install");
    fs::remove_dir_all(root).expect("remove test directory");
}

fn write_common_source_tree(source: &Path) {
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
}

fn write_mlx_transformer_source(directory: &Path, index_name: &str, bits: u8) {
    let packed_cols = 64 * bits as usize / 32;
    let mut packed = vec![0u8; packed_cols * 4];
    for col in 0..64 {
        let value = (col as u16) & ((1 << bits) - 1);
        let bit_offset = col * bits as usize;
        for bit in 0..bits as usize {
            if value & (1 << bit) != 0 {
                packed[(bit_offset + bit) / 8] |= 1 << ((bit_offset + bit) % 8);
            }
        }
    }
    let scale = 0x3a00u16.to_le_bytes();
    let bias = 0x3c00u16.to_le_bytes();
    write_transformer_shard(
        directory,
        index_name,
        vec![
            ("linear.weight", "U32", vec![1, packed_cols], packed),
            ("linear.scales", "F16", vec![1, 1], scale.to_vec()),
            ("linear.biases", "F16", vec![1, 1], bias.to_vec()),
        ],
    );
}

fn write_fp16_transformer_source(directory: &Path, index_name: &str) {
    let mut raw = Vec::with_capacity(64 * 2);
    for _ in 0..64 {
        raw.extend_from_slice(&0x3e00u16.to_le_bytes());
    }
    write_transformer_shard(
        directory,
        index_name,
        vec![("linear.weight", "F16", vec![1, 64], raw)],
    );
}

fn write_transformer_shard(
    directory: &Path,
    index_name: &str,
    tensors: Vec<(&str, &str, Vec<usize>, Vec<u8>)>,
) {
    fs::create_dir_all(directory).expect("create transformer source");
    let source_file = index_name
        .strip_suffix(".index.json")
        .expect("indexed source file name");
    let mut payload = Vec::new();
    let mut header = serde_json::Map::new();
    let mut weight_map = serde_json::Map::new();
    for (name, dtype, shape, bytes) in tensors {
        let start = payload.len();
        payload.extend_from_slice(&bytes);
        let end = payload.len();
        header.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, end],
            }),
        );
        weight_map.insert(
            name.to_string(),
            serde_json::Value::String(source_file.to_string()),
        );
    }
    let header_bytes =
        serde_json::to_vec(&serde_json::Value::Object(header)).expect("serialize MLX header");
    let mut shard = Vec::with_capacity(8 + header_bytes.len() + payload.len());
    shard.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    shard.extend_from_slice(&header_bytes);
    shard.extend_from_slice(&payload);
    fs::write(directory.join(source_file), shard).expect("write MLX source shard");
    fs::write(
        directory.join(index_name),
        serde_json::to_vec(&serde_json::json!({
            "weight_map": weight_map,
        }))
        .expect("serialize MLX source index"),
    )
    .expect("write MLX source index");
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
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    let sequence = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!(
        "turbospark-image-builder-{}-{nanos}-{sequence}",
        std::process::id()
    ));
    fs::create_dir_all(&root).expect("create test directory");
    root
}
