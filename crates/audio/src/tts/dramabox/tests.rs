//! Parity and regression tests for the DramaBox family.
//!
//! The fixture-backed cases land with `tools/gen_dramabox_fixtures.py`;
//! see the family README for the protocol and measured tolerances.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::{
    aligned_frame_count, decode_waveform_refusal, estimate_speech_duration,
    resolve_generation_duration, DiffusionNoise, GenerateParams, Model,
};

fn testdata(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/tts_dramabox")
        .join(name)
}

fn read_json(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).expect("fixture json");
    serde_json::from_str(&text).expect("fixture json")
}

fn f32_vec(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .expect("values")
        .iter()
        .map(|v| v.as_f64().expect("value") as f32)
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn assert_relative(got: &[f32], want: &[f32], relative: f32, what: &str) {
    let magnitude = want.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    let gate = relative * magnitude;
    let diff = max_abs_diff(got, want);
    if std::env::var("DRAMABOX_MEASURE").is_ok() {
        eprintln!(
            "{what}: diff {diff:e} magnitude {magnitude:e} relative {:.3e}",
            diff / magnitude
        );
    }
    assert!(diff <= gate, "{what} diff {diff} exceeds {gate}");
}

/// Replays the recorded reference `mx.random.normal` draws in order.
struct RecordedNoise {
    draws: Vec<Vec<f32>>,
    cursor: usize,
}

impl RecordedNoise {
    fn new(draws: &[Value]) -> Self {
        Self {
            draws: draws.iter().map(|draw| f32_vec(&draw["values"])).collect(),
            cursor: 0,
        }
    }
}

impl DiffusionNoise for RecordedNoise {
    fn normal(&mut self, rows: usize, cols: usize) -> Vec<f32> {
        let draw = self
            .draws
            .get(self.cursor)
            .unwrap_or_else(|| panic!("draw {} missing", self.cursor))
            .clone();
        self.cursor += 1;
        assert_eq!(draw.len(), rows * cols, "draw {} shape", self.cursor - 1);
        draw
    }
}

/// Builds the cross-attention contexts through the ported conditioner
/// over the engineered Gemma hidden states the traces carry (the text
/// encoder is a separate download; see the README).
fn contexts_of(m: &Model, traces: &Value) -> (Vec<f32>, Vec<f32>, usize) {
    let context_tokens = traces["context_tokens"].as_u64().expect("ctx") as usize;
    let hidden = f32_vec(&traces["conditioner_inputs"]);
    let negative_hidden = f32_vec(&traces["negative_conditioner_inputs"]);
    let mask: Vec<i32> = traces["attention_mask"]
        .as_array()
        .expect("mask")
        .iter()
        .map(|v| v.as_i64().expect("mask entry") as i32)
        .collect();
    let context = m
        .encode_context(&hidden, context_tokens, &mask)
        .expect("context encodes");
    let negative = m
        .encode_context(&negative_hidden, context_tokens, &mask)
        .expect("negative context encodes");
    (context, negative, context_tokens)
}

/// D1: the full guided loop (cfg 2.0, stg 1.0, auto rescale) matches
/// the reference denoised latents end to end, conditioner included.
#[test]
fn guided_generation_matches_the_reference_latents() {
    let traces = read_json(&testdata("generate/traces.json"));
    let mut m = Model::open(&testdata("generate")).expect("dramabox model loads");
    let tokens = traces["tokens"].as_u64().expect("tokens") as usize;
    let (context, negative, context_tokens) = contexts_of(&m, &traces);
    let params = GenerateParams {
        cfg_scale: traces["cfg_scale"].as_f64().expect("cfg") as f32,
        stg_scale: traces["stg_scale"].as_f64().expect("stg") as f32,
        stg_block: traces["stg_block"].as_u64().expect("block") as usize,
        rescale_scale_auto: true,
        rescale_scale: 0.0,
        modality_scale: 1.0,
        steps: traces["steps"].as_u64().expect("steps") as usize,
    };
    let mut noise = RecordedNoise::new(traces["draws"].as_array().expect("draws"));
    let got = m
        .generate(
            tokens,
            &context,
            context_tokens,
            Some(&negative),
            context_tokens,
            &params,
            &mut noise,
        )
        .expect("generate runs");
    assert_relative(&got.latents, &f32_vec(&traces["latents"]), 2e-4, "latents");
}

/// D2: the plain-path twin (cfg 1.0, stg 0.0) matches without any
/// guidance passes.
#[test]
fn plain_generation_matches_the_reference_latents() {
    let traces = read_json(&testdata("plain/traces.json"));
    let mut m = Model::open(&testdata("plain")).expect("dramabox model loads");
    let tokens = traces["tokens"].as_u64().expect("tokens") as usize;
    let (context, _negative, context_tokens) = contexts_of(&m, &traces);
    let params = GenerateParams {
        cfg_scale: 1.0,
        stg_scale: 0.0,
        stg_block: 0,
        rescale_scale_auto: false,
        rescale_scale: 0.0,
        modality_scale: 1.0,
        steps: traces["steps"].as_u64().expect("steps") as usize,
    };
    let mut noise = RecordedNoise::new(traces["draws"].as_array().expect("draws"));
    let got = m
        .generate(
            tokens,
            &context,
            context_tokens,
            None,
            0,
            &params,
            &mut noise,
        )
        .expect("generate runs");
    assert_relative(&got.latents, &f32_vec(&traces["latents"]), 2e-4, "latents");
}

/// D3: the waveform decode is refused with the AudioVAE explanation.
#[test]
fn waveform_decode_is_refused() {
    let error = decode_waveform_refusal();
    assert!(matches!(error, crate::SpeechError::Unsupported { .. }));
    let text = format!("{error}");
    assert!(text.contains("AudioVAE"), "{text}");
}

/// D4: the duration estimator matches the reference numbers on the
/// documented cases.
#[test]
fn duration_estimation_matches_the_reference_rules() {
    // Short spoken text (< 40 chars): 0.6 * 14 cps, plus 2 seconds,
    // rounded to one decimal: 18 / 8.4 + 2 = 4.1.
    assert!((estimate_speech_duration("hello there friend", 1.0) - 4.1).abs() < 1e-4);
    // A quoted sentence over the 3-word single-quote rule.
    let quoted = r#"he said "hello there my dear friend, sit down""#;
    let duration = estimate_speech_duration(quoted, 1.0);
    assert!(duration > 4.0, "{duration}");
    // Speed scales the spoken term: 18 / 16.8 + 2 = 3.1.
    let fast = estimate_speech_duration("hello there friend", 2.0);
    assert!((fast - 3.1).abs() < 1e-4);
    // Nonverbal terms add on top.
    let sighing =
        estimate_speech_duration(r#"he sighs, "hello there my dear friend, sit down""#, 1.0);
    assert!(sighing > duration, "{sighing} vs {duration}");
    // The floor and the generation resolution: the floored 3.0 scales
    // to 3.3 before its own floor.
    assert!((resolve_generation_duration("hi", 1.0, 1.1, 0.0) - 3.3).abs() < 1e-4);
    assert!((resolve_generation_duration("hi", 1.0, 1.1, 5.5) - 5.5).abs() < 1e-4);
}

/// D5: the 25 fps + 8-token frame alignment.
#[test]
fn frame_count_alignment_matches_the_reference() {
    // 1.0 s -> 26 frames -> round to ((25 + 4) / 8) * 8 + 1 = 33.
    assert_eq!(aligned_frame_count(1.0, 25.0), 25);
    // 0.0 s -> 1 -> 1.
    assert_eq!(aligned_frame_count(0.0, 25.0), 1);
    // 2.0 s -> 51 frames -> ((50 + 4) / 8) * 8 + 1 = 49.
    assert_eq!(aligned_frame_count(2.0, 25.0), 49);
}

/// D6: the config parser resolves the documented defaults.
#[test]
fn config_defaults_match_the_reference_dataclasses() {
    let value: Value =
        serde_json::from_str(&std::fs::read_to_string(testdata("generate/config.json")).unwrap())
            .unwrap();
    let config = super::ModelConfig::from_json(&value).expect("config parses");
    assert_eq!(config.audio.sample_rate, 8000);
    assert_eq!(config.audio.latent_sample_rate, 16000);
    assert_eq!(config.audio.vae_channels, 4);
    assert_eq!(config.audio.mel_bins, 4);
    // The fixture config pins the guidance defaults it records with.
    assert_eq!(config.inference_defaults.cfg_scale, 2.0);
    assert_eq!(config.inference_defaults.stg_scale, 1.0);
    assert_eq!(config.inference_defaults.steps, 30);
    assert_eq!(config.transformer.num_layers, 2);
    assert!(config.transformer.rope_split);
    assert!(config.transformer.frequencies_precision_float64);
    assert_eq!(config.text_encoder_hidden_size, 8);
}
