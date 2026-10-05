#![cfg(target_os = "macos")]

use audio::music::minimax_music3::{GenerateRequest, Model, TextGenerateRequest};
use std::path::PathBuf;
use turbospark_runtime::Music3Runner;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../audio/testdata/minimax_music3")
        .join(name)
}

#[track_caller]
fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    let mut worst = 0.0f32;
    for (&a, &e) in actual.iter().zip(expected) {
        assert!(a.is_finite() && e.is_finite());
        worst = worst.max((a - e).abs());
    }
    assert!(worst <= tolerance, "maximum error {worst} > {tolerance}");
}

#[test]
fn music3_metal_matches_cpu_ar_and_waveform_and_resets() {
    let trace: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture("ar_trace_short.json")).unwrap()).unwrap();
    let ids: Vec<i32> = trace["text_ids"][0]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    for tree in ["converted_plain", "converted_q8"] {
        let cpu = Model::load_converted(&fixture(tree)).unwrap();
        let runner = Music3Runner::open_with_precision(
            &fixture(tree),
            audio::music::minimax_music3::Music3Precision::Float32,
        )
        .unwrap();
        let (expected, codes) = cpu.generate_frame_hiddens(&ids, 3, 7).unwrap();
        let mut decisions = 0;
        let (actual, actual_codes) = runner
            .generate_frame_hiddens_traced(&ids, 3, 7, |trace| {
                assert_eq!(trace.hiddens.len(), 2 * runner.config().hidden_size);
                assert!(trace.guided_logits.iter().all(|v| v.is_finite()));
                decisions += 1;
            })
            .unwrap();
        assert_eq!(decisions, 4 * runner.config().num_codebooks);
        assert_eq!(actual_codes, codes);
        close(&actual, &expected, 2e-4);
        let request = GenerateRequest {
            text_ids: ids.clone(),
            frames: 3,
            steps: 2,
            seed: 7,
        };
        let expected = cpu.generate(&request).unwrap();
        let actual = runner.generate(&request).unwrap();
        assert_eq!(
            (actual.frames, actual.samples, actual.sample_rate),
            (expected.frames, expected.samples, expected.sample_rate)
        );
        close(&actual.waveform, &expected.waveform, 2e-4);
        let again = runner.generate(&request).unwrap();
        assert_eq!(again.waveform, actual.waveform);
        assert!(runner.resident_weight_bytes() > 0);
    }
}

#[test]
fn music3_metal_validates_text_and_request() {
    let runner = Music3Runner::open(&fixture("converted_plain")).unwrap();
    let mut request = TextGenerateRequest::new("piano", "[instrumental]");
    request.duration_seconds = Some(0.04);
    request.steps = Some(1);
    let generated = runner.generate_text(&request).unwrap();
    assert_eq!(generated.frames, 1);
    assert!(generated.samples > 0);
    request.steps = Some(0);
    assert!(runner.generate_text(&request).is_err());
    assert!(runner
        .generate(&GenerateRequest {
            text_ids: vec![-1],
            frames: 1,
            steps: 1,
            seed: 0,
        })
        .is_err());
}

#[test]
fn music3_metal_preserves_two_chunk_overlap_and_stereo_crop() {
    let cpu = Model::load_converted(&fixture("converted_plain")).unwrap();
    let runner = Music3Runner::open_with_precision(
        &fixture("converted_plain"),
        audio::music::minimax_music3::Music3Precision::Float32,
    )
    .unwrap();
    let data = std::fs::read(fixture("long_flow_hiddens.npy")).unwrap();
    let header = u16::from_le_bytes([data[8], data[9]]) as usize;
    let hiddens: Vec<f32> = data[10 + header..]
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let frames = hiddens.len() / (cpu.config().num_codebooks * cpu.config().hidden_size);
    assert!(frames > 200);
    let expected = cpu.run_flow(&hiddens, frames, 1, 0).unwrap();
    let actual = runner.run_flow(&hiddens, frames, 1, 0).unwrap();
    close(&actual, &expected, 2e-4);
}

#[test]
#[ignore = "requires a local pinned full-size Music 3 checkpoint and a Metal device"]
fn music3_real_checkpoint_generates_stereo_waveform() {
    let path = std::env::var_os("TURBOSPARK_MUSIC3_INSTALL_DIR")
        .expect("set TURBOSPARK_MUSIC3_INSTALL_DIR to the pinned checkpoint");
    let runner = Music3Runner::open(&PathBuf::from(path)).unwrap();
    assert_eq!(runner.config().hidden_size, 4096);
    assert_eq!(runner.config().num_hidden_layers, 36);
    assert!(runner.resident_weight_bytes() > 1_000_000_000);
    let mut request =
        TextGenerateRequest::new("Soft piano, warm melody, instrumental", "[instrumental]");
    request.duration_seconds = Some(1.0);
    request.steps = Some(2);
    request.seed = Some(7);
    let generated = runner.generate_text(&request).unwrap();
    assert!(generated.frames > 0 && generated.frames <= 25);
    assert!(generated.samples > 0);
    assert_eq!(generated.waveform.len(), generated.samples * 2);
    assert_eq!(generated.sample_rate, 44100);
    assert!(generated
        .waveform
        .iter()
        .all(|v| v.is_finite() && v.abs() <= 1.));
    assert!(generated.waveform.iter().any(|v| v.abs() > 1e-5));
    eprintln!(
        "Music 3 checkpoint: {} frames, {} samples, {} resident bytes",
        generated.frames,
        generated.samples,
        runner.resident_weight_bytes()
    );
}

#[test]
#[ignore = "requires a pinned checkpoint and MLX 0.32.3 float32 reference probe dumps"]
fn music3_real_checkpoint_matches_controlled_float32_reference() {
    let model = std::env::var_os("TURBOSPARK_MUSIC3_INSTALL_DIR").expect("checkpoint directory");
    let reference = PathBuf::from(
        std::env::var_os("TURBOSPARK_MUSIC3_REFERENCE_DIR").expect("reference dump directory"),
    );
    let request: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference.join("request.json")).unwrap()).unwrap();
    assert_eq!(
        request["reference_pin"],
        "feb25a37b07923bae556e59111995071d66afa0d"
    );
    assert_eq!(request["mlx_version"], "0.32.3");
    assert_eq!(request["precision"], "float32");
    let ids: Vec<i32> = serde_json::from_value(request["ids"].clone()).unwrap();
    let expected_codes: Vec<Vec<i32>> =
        serde_json::from_slice(&std::fs::read(reference.join("codes.json")).unwrap()).unwrap();
    let read = |name: &str| {
        let data = std::fs::read(reference.join(name)).unwrap();
        assert_eq!(data.len() % 4, 0);
        data.chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    let runner = Music3Runner::open_with_precision(
        &PathBuf::from(model),
        audio::music::minimax_music3::Music3Precision::Float32,
    )
    .unwrap();
    let frames = request["frames"].as_u64().unwrap() as usize;
    let steps = request["steps"].as_u64().unwrap() as usize;
    let seed = request["seed"].as_u64().unwrap();
    let (hiddens, codes) = runner.generate_frame_hiddens(&ids, frames, seed).unwrap();
    assert_eq!(codes, expected_codes);
    close(&hiddens, &read("hiddens.f32"), 2e-4);
    let wave = runner.run_flow(&hiddens, codes.len(), steps, seed).unwrap();
    close(&wave, &read("wave.f32"), 2e-4);
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}
fn native_gate(model: &std::path::Path, reference: &std::path::Path, repeat: bool) {
    use audio::music::minimax_music3::Music3Precision;
    let request: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference.join("request.json")).unwrap()).unwrap();
    assert_eq!(request["precision"], "checkpoint");
    assert_eq!(request["mlx_version"], "0.32.3");
    assert_eq!(
        request["reference_pin"],
        "feb25a37b07923bae556e59111995071d66afa0d"
    );
    let ids = serde_json::from_value::<Vec<i32>>(request["ids"].clone()).unwrap();
    let expected_codes: Vec<Vec<i32>> =
        serde_json::from_slice(&std::fs::read(reference.join("codes.json")).unwrap()).unwrap();
    let expected_warmup: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference.join("warmup.json")).unwrap()).unwrap();
    let runner = Music3Runner::open_with_precision(model, Music3Precision::Checkpoint).unwrap();
    for _ in 0..if repeat { 2 } else { 1 } {
        let mut warmup = vec![];
        let (hidden,codes)=runner.generate_frame_hiddens_traced(&ids,request["frames"].as_u64().unwrap() as usize,request["seed"].as_u64().unwrap(),|t|{
            if t.frame==0 {
                close(t.hiddens,&read_f32(&reference.join(format!("warmup_{}_hidden.f32",t.codebook))),2e-4);
                close(t.guided_logits,&read_f32(&reference.join(format!("warmup_{}_logits.f32",t.codebook))),2e-4);
                warmup.push(serde_json::json!({"codebook":t.codebook,"key":[t.key.0,t.key.1],"sampled":t.sampled}));
            }
        }).unwrap();
        let expected_warmup:Vec<_>=expected_warmup.as_array().unwrap().iter().map(|v|serde_json::json!({"codebook":v["codebook"],"key":v["key"],"sampled":v["sampled"]})).collect();
        assert_eq!(warmup, expected_warmup);
        assert_eq!(codes, expected_codes);
        close(&hidden, &read_f32(&reference.join("hiddens.f32")), 2e-4);
        let wave = runner
            .run_flow(
                &hidden,
                codes.len(),
                request["steps"].as_u64().unwrap() as usize,
                request["seed"].as_u64().unwrap(),
            )
            .unwrap();
        close(&wave, &read_f32(&reference.join("wave.f32")), 2e-4);
    }
}

#[test]
fn seven_native_tiny_profiles_match_pinned_mlx_and_reset() {
    for encoding in [
        "bf16", "affine8", "affine6", "affine4", "mxfp8", "mxfp4", "nvfp4",
    ] {
        eprintln!("native profile: {encoding}");
        let path = fixture("precision").join(encoding);
        native_gate(&path, &path, true);
    }
}
#[test]
fn native_201_frame_overlap_and_cropping_match_mlx() {
    let path = fixture("precision/long201");
    let request: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("request.json")).unwrap()).unwrap();
    let runner = Music3Runner::open(&path).unwrap();
    let wave = runner
        .run_flow(
            &read_f32(&path.join("hiddens.f32")),
            201,
            request["steps"].as_u64().unwrap() as usize,
            request["seed"].as_u64().unwrap(),
        )
        .unwrap();
    close(&wave, &read_f32(&path.join("wave.f32")), 5e-4);
}
#[test]
#[ignore = "requires pinned MXFP8 install and an independently generated native MLX reference"]
fn real_music3_native_checkpoint_matches_mlx() {
    let model =
        PathBuf::from(std::env::var_os("TURBOSPARK_MUSIC3_INSTALL_DIR").expect("install dir"));
    let reference =
        PathBuf::from(std::env::var_os("TURBOSPARK_MUSIC3_REFERENCE_DIR").expect("reference dir"));
    let request: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference.join("request.json")).unwrap()).unwrap();
    assert_eq!(
        request["checkpoint_revision"],
        "d00a12c3c7f80eb66379dd02dd0f30ed0ce2d96e"
    );
    assert_eq!(
        request["reference_source_sha256"],
        "a886c16bcb9322986a3a9adac0383ee95e666dbdbd2cf5af8a8c262010ab591a"
    );
    assert_eq!(request["hiddens_dtype"], "bfloat16");
    assert_eq!(request["wave_dtype"], "bfloat16");
    native_gate(&model, &reference, false);
}
