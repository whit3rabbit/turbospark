use std::path::{Path, PathBuf};
use turbospark_audio::sts::mel_roformer::{MelRoFormer, MelRoFormerConfig};
use turbospark_model_io::safetensors::SafetensorsFile;
fn testdata(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/mel_roformer")
        .join(name)
}
fn read_json(name: &str) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(testdata(name)).unwrap()).unwrap()
}
fn load_model() -> MelRoFormer {
    let config = MelRoFormerConfig::from_json(&read_json("config.json")).unwrap();
    let file = SafetensorsFile::open(&testdata("tiny_weights.safetensors")).unwrap();
    MelRoFormer::load(config, &file).unwrap()
}

#[test]
fn malformed_config_and_audio_are_rejected_before_compute() {
    for (key, value) in [
        ("hop_length", serde_json::json!(0)),
        ("dim", serde_json::json!(0)),
        ("heads", serde_json::json!(0)),
        ("n_fft", serde_json::json!(0)),
        ("num_bands", serde_json::json!(0)),
        ("dim_head", serde_json::json!(3)),
        ("hop_length", serde_json::json!(-1)),
        ("num_stems", serde_json::json!(2)),
    ] {
        let mut config = read_json("config.json");
        config[key] = value;
        assert!(
            MelRoFormerConfig::from_json(&config).is_err(),
            "accepted {key}"
        );
    }
    let model = load_model();
    for invalid in [
        vec![],
        vec![0.0; 3],
        vec![0.0, f32::NAN],
        vec![f32::INFINITY, 0.0],
    ] {
        assert!(model.forward(&invalid).is_err());
    }
}
#[test]
fn checkpoint_geometry_must_match_config_before_inference() {
    let mut config = MelRoFormerConfig::from_json(&read_json("config.json")).unwrap();
    config.heads += 1;
    let file = SafetensorsFile::open(&testdata("tiny_weights.safetensors")).unwrap();
    assert!(MelRoFormer::load(config, &file).is_err());
}
