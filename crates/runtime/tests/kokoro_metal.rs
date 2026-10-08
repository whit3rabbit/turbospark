#![cfg(target_os = "macos")]
//! Exact checkpoint gate. Requires existing immutable checkpoint/reference paths.
//! TURBOSPARK_KOKORO_DIR=<pin> TURBOSPARK_KOKORO_REFERENCE=<reference.json>
//! cargo test -p turbospark-runtime --test kokoro_metal -- --ignored --nocapture
use audio::tts::kokoro::SynthesisRequest;
use serde_json::Value;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use turbospark_runtime::{
    native_audio::{AudioCancel, AudioEvent, AudioRequest, AudioSession, AudioTask},
    KokoroRunner,
};
fn checkpoint() -> PathBuf {
    std::env::var_os("TURBOSPARK_KOKORO_DIR")
        .expect("set existing exact Kokoro pin")
        .into()
}
fn pcm(model: &KokoroRunner, request: &SynthesisRequest, seed: u64) -> Vec<f32> {
    let mut output = Vec::new();
    model
        .synthesize_controlled(
            request,
            seed,
            |_, _| Ok(()),
            |chunk| {
                output.extend(chunk);
                true
            },
        )
        .unwrap();
    output
}
#[test]
#[ignore = "requires exact installed Kokoro checkpoint and independent MLX reference"]
fn kokoro_seeded_metal_synthesis_matches_reference_and_resets() {
    let reference: Value = serde_json::from_slice(
        &std::fs::read(
            std::env::var_os("TURBOSPARK_KOKORO_REFERENCE")
                .expect("set independently generated reference.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        reference["reference_revision"],
        "e1b19b9054bf163f5d812221a54fcc346f1890e9"
    );
    assert_eq!(reference["mlx_version"], "0.31.2");
    assert_eq!(
        reference["checkpoint_revision"],
        "a71e4d38b236d968966a2002c4c895dbd12b1c3c"
    );
    let model = KokoroRunner::open(&checkpoint()).unwrap();
    assert!(model.using_metal());
    let traces = Rc::new(RefCell::new(HashMap::<String, Vec<f32>>::new()));
    let capture = traces.clone();
    model.set_trace_observer(move |name, values, _| {
        capture.borrow_mut().insert(name.into(), values.to_vec());
    });
    let request = SynthesisRequest::new(reference["text"].as_str().unwrap());
    let first = pcm(&model, &request, 0);
    model.clear_trace_observer();
    if let Some(path) = std::env::var_os("TURBOSPARK_KOKORO_TRACE") {
        let mut output = traces.borrow().clone();
        output.insert("pcm".into(), first.clone());
        std::fs::write(path, serde_json::to_vec(&output).unwrap()).unwrap();
    }
    assert!(first.iter().all(|v| v.is_finite()));
    assert!(!first.is_empty());
    let mut failures = Vec::new();
    for (stage, expected) in reference["traces"].as_object().unwrap() {
        let expected: Vec<f32> = expected
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let captured = traces.borrow();
        let actual = if stage == "pcm" {
            &first
        } else {
            captured.get(stage).expect("stage emitted")
        };
        assert_eq!(actual.len(), expected.len(), "{stage} shape");
        let maximum = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let rms = (actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            / actual.len() as f64)
            .sqrt();
        let scale = expected.iter().map(|v| v.abs()).fold(1.0f32, f32::max);
        println!(
            "{stage}: {} values, maximum {maximum}, rms {rms}, scale {scale}",
            actual.len()
        );
        let passed = if stage == "durations" {
            actual == &expected
        } else if stage == "pcm" {
            maximum <= 2e-3 && rms <= 2e-4
        } else {
            maximum <= scale * 1e-4
        };
        println!("{stage} gate: {}", if passed { "PASS" } else { "FAIL" });
        if !passed {
            failures.push(stage.clone());
        }
    }
    let second = pcm(&model, &request, 0);
    assert_eq!(first, second, "request seed resets every run");
    let third = pcm(&model, &request, 7);
    assert_ne!(first, third, "seed must affect harmonic source");
    let counts = model.dispatch_counts();
    for op in ["embedding", "linear", "attention", "convolution", "lstm"] {
        assert!(
            counts.get(op).copied().unwrap_or(0) > 0,
            "real {op} dispatch"
        );
    }
    println!(
        "Metal dispatches {counts:?}, resident weight bytes {}, PCM samples {}",
        model.resident_weight_bytes(),
        first.len()
    );
    if let Some(path) = std::env::var_os("TURBOSPARK_KOKORO_OUTPUT") {
        std::fs::write(
            path,
            audio::write_wav_f32(&audio::Waveform::new(24_000, 1, first.clone()).unwrap()),
        )
        .unwrap();
    }
    assert!(
        failures.is_empty(),
        "frozen checkpoint gates failed: {failures:?}"
    );
}
#[test]
#[ignore = "requires exact installed Kokoro checkpoint"]
fn kokoro_cancellation_stops_before_next_segment() {
    let model = KokoroRunner::open(&checkpoint()).unwrap();
    let cancel = AudioCancel::default();
    let request = SynthesisRequest::new("Hello, world! Hello, world!");
    let mut emitted = 0;
    let mut checked = Vec::new();
    let result = model.synthesize_controlled(
        &request,
        0,
        |index, total| {
            checked.push((index, total));
            cancel
                .checkpoint()
                .map_err(|why| audio::SpeechError::Input { why })
        },
        |_| {
            emitted += 1;
            cancel.cancel();
            true
        },
    );
    assert!(result.unwrap_err().to_string().contains("cancelled"));
    assert_eq!(emitted, 1);
    assert_eq!(checked, vec![(0, 2), (1, 2)]);
}
#[test]
#[ignore = "requires exact installed Kokoro checkpoint"]
fn kokoro_worker_uses_metal_serializes_and_returns_typed_pcm() {
    let session = AudioSession::open(checkpoint(), AudioTask::TextToSpeech, false).unwrap();
    assert_eq!(session.family(), "kokoro");
    let request = AudioRequest {
        task: AudioTask::TextToSpeech,
        // More than two queued events remain at the first PCM callback, so the
        // bounded channel keeps execution active while the permit is checked.
        text: "Hello, world! Hello, world! Hello, world!".into(),
        seed: Some(7),
        ..Default::default()
    };
    let mut count = 0;
    let mut progress = Vec::new();
    let mut busy_checked = false;
    let mut permit_checked = false;
    let result = session
        .execute(
            request.clone(),
            Vec::new(),
            AudioCancel::default(),
            |event| match event {
                AudioEvent::Pcm(chunk) => {
                    if !permit_checked {
                        permit_checked = true;
                        assert!(
                            turbospark_runtime::heavy::HeavyWorkGuard::try_acquire().is_none(),
                            "the audio worker must retain its permit while execution is active"
                        );
                    }
                    assert!(chunk.len() <= turbospark_runtime::native_audio::MAX_PCM_CHUNK);
                    assert!(chunk.iter().all(|v| v.is_finite()));
                    count += chunk.len();
                }
                AudioEvent::Progress(p) => {
                    progress.push(p);
                    if !busy_checked {
                        busy_checked = true;
                        assert!(session
                            .execute(request.clone(), Vec::new(), AudioCancel::default(), |_| {})
                            .unwrap_err()
                            .contains("busy"));
                    }
                }
            },
        )
        .unwrap();
    assert!(permit_checked, "the active-worker permit check must run");
    assert_eq!(result.seed, Some(7));
    assert_eq!(result.sample_count, count);
    let format = result.pcm_format.unwrap();
    assert_eq!(format.sample_rate, 24_000);
    assert_eq!(format.channels, 1);
    assert!(format.interleaved);
    assert!(progress
        .iter()
        .any(|p| p.stage == "synthesis" && p.completed == 0 && p.total == Some(3)));
    let cancel = AudioCancel::default();
    let request = AudioRequest {
        task: AudioTask::TextToSpeech,
        text: "Hello, world! Hello, world!".into(),
        ..Default::default()
    };
    assert!(session
        .execute(request, Vec::new(), cancel.clone(), |event| {
            if matches!(event, AudioEvent::Pcm(_)) {
                cancel.cancel()
            }
        })
        .unwrap_err()
        .contains("cancelled"));
    drop(session);
}

#[test]
#[ignore = "requires exact installed checkpoint and independent vocoder tensors"]
fn kokoro_vocoder_replay_separates_identical_f0_and_source_inputs() {
    let reference: Value = serde_json::from_slice(
        &std::fs::read(std::env::var_os("TURBOSPARK_KOKORO_REFERENCE").unwrap()).unwrap(),
    )
    .unwrap();
    let values = |v: &Value| -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    };
    let features = values(&reference["traces"]["decoder_block3"]);
    let f0 = values(&reference["traces"]["f0"]);
    let source = values(&reference["traces"]["harmonic_source"]);
    let expected_pcm = values(&reference["traces"]["pcm"]);
    let style = values(&reference["style"]);
    let model = KokoroRunner::open(&checkpoint()).unwrap();
    let traces = Rc::new(RefCell::new(HashMap::<String, Vec<f32>>::new()));
    let capture = traces.clone();
    model.set_trace_observer(move |name, values, _| {
        capture.borrow_mut().insert(name.into(), values.to_vec());
    });
    let (native_source, from_f0) = model
        .diagnostic_vocoder(&features, &style, &f0, 0, None)
        .unwrap();
    let (_, from_source) = model
        .diagnostic_vocoder(&features, &style, &f0, 0, Some(&source))
        .unwrap();
    if let Some(path) = std::env::var_os("TURBOSPARK_KOKORO_REPLAY") {
        std::fs::write(path,serde_json::to_vec(&serde_json::json!({"source":native_source,"pcm_from_f0":from_f0,"pcm_from_source":from_source,"traces":traces.borrow().clone()})).unwrap()).unwrap();
    }
    let mut metrics = Vec::new();
    for (name, actual, expected) in [
        ("identical_f0_source", &native_source, &source),
        ("identical_f0_pcm", &from_f0, &expected_pcm),
        ("identical_source_pcm", &from_source, &expected_pcm),
    ] {
        assert_eq!(actual.len(), expected.len());
        let maximum = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let rms = (actual
            .iter()
            .zip(expected)
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            / actual.len() as f64)
            .sqrt();
        println!("{name}: maximum {maximum}, rms {rms}");
        metrics.push((name, maximum, rms));
    }
    assert!(metrics[0].1 <= 1e-4, "identical F0 harmonic source parity");
    for (_, maximum, rms) in &metrics[1..] {
        assert!(*maximum <= 2e-3);
        assert!(*rms <= 2e-4);
    }
}

#[test]
#[ignore = "requires installed checkpoint for actual loaded configuration"]
fn kokoro_memory_admission_scales_with_segment_duration_and_speed() {
    let model = KokoroRunner::open(&checkpoint()).unwrap();
    let mut request = SynthesisRequest::new("Hello, world!");
    let short = model.estimate_run_bytes(&request).unwrap();
    request.speed = 0.5;
    let slow = model.estimate_run_bytes(&request).unwrap();
    assert!(
        slow > short,
        "slower synthesis requires a larger activation reserve"
    );
    request = SynthesisRequest::new("The quick brown fox jumps over the lazy dog ".repeat(10));
    let long = model.estimate_run_bytes(&request).unwrap();
    assert!(
        long > slow,
        "longer checked segments require a larger activation reserve"
    );
    request = SynthesisRequest::new("Hello, world! Hello, world!");
    assert_eq!(
        model.estimate_run_bytes(&request).unwrap(),
        short,
        "segments run serially, reserve their peak rather than their sum"
    );
}

#[test]
#[ignore = "requires exact installed checkpoint for request seed-reset guard"]
fn kokoro_seed_reset_is_observable_between_requests() {
    let model = KokoroRunner::open(&checkpoint()).unwrap();
    let request = SynthesisRequest::new("Hello, world!");
    let first = pcm(&model, &request, 0);
    let second = pcm(&model, &request, 7);
    let third = pcm(&model, &request, 0);
    assert_ne!(
        first, second,
        "requested seed must change the native source"
    );
    assert_eq!(
        first, third,
        "request seed reset restores the original waveform"
    );
}
