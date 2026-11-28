//! `mage_vl` intake refusals and the tower walk, on a synthetic checkpoint
//! under the MLX names (`docs/MAGE_VL_VISION_PHASE0.md`).

use turbospark_repack::{
    build_synthetic_mage_vl_install, parse_mage_vl_vision_config, parse_qwen3_config,
    synthetic_mage_vl_checkpoint,
};

fn config() -> serde_json::Value {
    let fixture = synthetic_mage_vl_checkpoint(128, [100, 101, 102, 103]);
    serde_json::from_str(&fixture.config_json).unwrap()
}

#[test]
fn the_trunk_parses_from_text_config_with_either_rope_spelling() {
    let mut root = config();
    let nested = parse_qwen3_config(&root.to_string()).expect("rope_parameters");
    assert_eq!(nested.rope_theta, 5_000_000.0);
    // The netprojects conversions write the flat key and no block.
    let text = root["text_config"].as_object_mut().unwrap();
    text.remove("rope_parameters");
    text.insert("rope_theta".into(), 5_000_000.into());
    let flat = parse_qwen3_config(&root.to_string()).expect("rope_theta");
    assert_eq!(nested, flat);

    root["text_config"]["rope_parameters"] =
        serde_json::json!({"rope_theta": 5_000_000, "rope_type": "yarn"});
    let err = parse_qwen3_config(&root.to_string()).unwrap_err();
    assert!(err.to_string().contains("yarn"), "{err}");
}

#[test]
fn the_unquantized_and_cowork_source_layouts_are_refused_by_name() {
    let mut root = config();
    root.as_object_mut().unwrap().remove("quantization");
    let err = parse_qwen3_config(&root.to_string()).unwrap_err();
    assert!(
        err.to_string().contains("mage_vl installs from an MLX"),
        "{err}"
    );

    let mut root = config();
    root["model_type"] = "llava_onevision2".into();
    let err = parse_qwen3_config(&root.to_string()).unwrap_err();
    assert!(err.to_string().contains("llava_onevision2"), "{err}");
}

#[test]
fn every_hardcoded_tower_value_is_checked() {
    assert!(parse_mage_vl_vision_config(&config().to_string()).is_ok());
    for (key, value) in [
        ("hidden_act", serde_json::json!("gelu_pytorch_tanh")),
        ("layer_norm_type", serde_json::json!("rms_norm")),
        ("layer_norm_eps", serde_json::json!(1e-5)),
        ("rope_theta", serde_json::json!(100.0)),
        ("patch_size", serde_json::json!(14)),
        ("temporal_patch_size", serde_json::json!(2)),
        ("spatial_merge_size", serde_json::json!(3)),
        ("use_head", serde_json::json!(true)),
        ("use_patch_position_encoding", serde_json::json!(true)),
        // Head width 16: the 4:6:6 rope split does not exist.
        ("num_attention_heads", serde_json::json!(4)),
    ] {
        let mut root = config();
        root["vision_config"][key] = value;
        assert!(
            parse_mage_vl_vision_config(&root.to_string()).is_err(),
            "{key} was accepted"
        );
    }
}

/// The walk writes the manifest kind, and the PyTorch conv layout (the OptiQ
/// sidecar's, same byte count) is refused by shape.
#[test]
fn the_walk_writes_the_tower_and_refuses_a_channels_first_patch_embedding() {
    use turbospark_repack::{
        parse_gemma4_quantization, parse_header, write_gemma4_install, MemoryRangeSource,
        DEFAULT_MAX_HEADER_BYTES,
    };
    let dir = std::env::temp_dir().join(format!("tsp-mage-walk-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    build_synthetic_mage_vl_install(&dir, None).expect("installs");
    let manifest = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
    assert!(manifest.contains("\"mage_vit\""), "{manifest}");

    let fixture = synthetic_mage_vl_checkpoint(128, [100, 101, 102, 103]);
    let find = |hay: &[u8], needle: &[u8]| hay.windows(needle.len()).position(|w| w == needle);
    let mut blob = fixture.safetensors;
    let name = find(&blob, b"patch_embedding.weight").unwrap();
    let at = name + find(&blob[name..], b"[64,16,16,3]").expect("shape in the header");
    blob[at..at + 12].copy_from_slice(b"[64,3,16,16]");

    let mut arch = parse_qwen3_config(&fixture.config_json).unwrap();
    arch.vision = parse_mage_vl_vision_config(&fixture.config_json).unwrap();
    let header = parse_header(&blob, DEFAULT_MAX_HEADER_BYTES).unwrap();
    let err = write_gemma4_install(
        &dir.join("refused"),
        &arch,
        "tiny",
        &header,
        &MemoryRangeSource::new(&blob),
        &parse_gemma4_quantization(&fixture.config_json).unwrap(),
    )
    .err()
    .expect("refused");
    assert!(err.to_string().contains("patch_embedding"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}
