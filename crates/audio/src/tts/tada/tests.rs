//! Parity and regression tests for the TADA family.
//!
//! The fixture-backed cases land with `tools/gen_tada_fixtures.py`;
//! see the family README for the protocol and measured tolerances.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::{build_prompt, DiffusionNoise, GenerateParams, Model, SpecialIds};

fn testdata(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/tts_tada")
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
    if std::env::var("TADA_MEASURE").is_ok() {
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

/// The tiny fixture tokenizer's special ids (shared with the Python
/// generator).
const SPECIAL: SpecialIds = SpecialIds {
    bos: 3,
    eot: 4,
    start_header: 5,
    end_header: 6,
};
const PAD: i32 = 7;

fn params_of(traces: &Value) -> GenerateParams {
    let p = &traces["params"];
    GenerateParams {
        acoustic_cfg_scale: p["acoustic_cfg_scale"].as_f64().unwrap_or(1.6) as f32,
        duration_cfg_scale: p["duration_cfg_scale"].as_f64().unwrap_or(1.0) as f32,
        cfg_schedule: p["cfg_schedule"].as_str().unwrap_or("cosine").to_owned(),
        noise_temperature: p["noise_temperature"].as_f64().unwrap_or(0.9) as f32,
        num_flow_matching_steps: p["num_flow_matching_steps"].as_u64().unwrap_or(4) as usize,
        time_schedule: p["time_schedule"].as_str().unwrap_or("logsnr").to_owned(),
        num_transition_steps: p["num_transition_steps"].as_u64().unwrap_or(5) as usize,
        max_tokens: p["max_tokens"].as_u64().unwrap_or(64) as usize,
        num_extra_steps: p["num_extra_steps"].as_u64().unwrap_or(0) as usize,
        temperature: 0.6,
        top_k: 0,
        top_p: 0.9,
        repetition_penalty: 1.1,
        structural: SPECIAL,
        pad_token_id: PAD,
        eos_ids: vec![SPECIAL.eot],
    }
}

fn prompt_of(traces: &Value) -> super::Prompt {
    super::Prompt {
        ids: f32_vec(&traces["prompt_ids"])
            .iter()
            .map(|v| *v as i32)
            .collect(),
        prefix_len: traces["prefix_len"].as_u64().expect("prefix") as usize,
    }
}

/// T1: the full zero-shot run with CFG matches the reference features
/// and durations.
#[test]
fn zero_shot_generation_with_cfg_matches_the_reference() {
    let traces = read_json(&testdata("generate/traces.json"));
    let prompt = prompt_of(&traces);
    let mut m = Model::open(&testdata("generate")).expect("tada model loads");
    let params = params_of(&traces);
    let mut noise = RecordedNoise::new(traces["draws"].as_array().expect("draws"));
    let got = m
        .generate(&prompt, &params, &mut noise)
        .expect("generate runs");
    assert_eq!(
        got.feature_frames,
        traces["feature_frames"].as_u64().expect("frames") as usize
    );
    let want_time: Vec<i32> = f32_vec(&traces["time_before"])
        .iter()
        .map(|v| *v as i32)
        .collect();
    assert_eq!(got.time_before, want_time, "time_before");
    assert_relative(
        &got.acoustic_features,
        &f32_vec(&traces["features"]),
        2e-4,
        "features",
    );
}

/// T2: the cfg-1.0 twin (single-row forwards, no guidance) matches.
#[test]
fn zero_shot_generation_without_cfg_matches_the_reference() {
    let traces = read_json(&testdata("nocfg/traces.json"));
    let prompt = prompt_of(&traces);
    let mut m = Model::open(&testdata("nocfg")).expect("tada model loads");
    let params = params_of(&traces);
    let mut noise = RecordedNoise::new(traces["draws"].as_array().expect("draws"));
    let got = m
        .generate(&prompt, &params, &mut noise)
        .expect("generate runs");
    assert_eq!(
        got.feature_frames,
        traces["feature_frames"].as_u64().expect("frames") as usize
    );
    assert_relative(
        &got.acoustic_features,
        &f32_vec(&traces["features"]),
        2e-4,
        "features",
    );
}

/// T3: the prompt builder assembles the reference chat prefix.
#[test]
fn prompt_builder_matches_the_reference_layout() {
    let prompt = build_prompt(&SPECIAL, &[10, 11], &[12, 13, 14], &[20, 21], 2);
    assert_eq!(
        prompt.ids,
        vec![3, 5, 10, 11, 6, 4, 5, 12, 13, 14, 6, 20, 21, 4, 4]
    );
    // Prefix without the BOS.
    assert_eq!(prompt.prefix_len, 10);
}

/// T4: the waveform decode is refused with the tada-codec explanation.
#[test]
fn waveform_decode_is_refused() {
    let error = super::decode_waveform_refusal();
    assert!(matches!(error, crate::SpeechError::Unsupported { .. }));
    let text = format!("{error}");
    assert!(text.contains("tada-codec"), "{text}");
}

/// T5: the config parser resolves the documented defaults.
#[test]
fn config_defaults_match_the_reference_dataclass() {
    let value: Value =
        serde_json::from_str(&std::fs::read_to_string(testdata("generate/config.json")).unwrap())
            .unwrap();
    let config = super::ModelConfig::from_json(&value).expect("config parses");
    assert_eq!(config.acoustic_dim, 8);
    assert_eq!(config.num_time_classes, 16);
    assert_eq!(config.shift_acoustic, 2);
    assert_eq!(config.head_layers, 2);
    assert_eq!(config.acoustic_std, 1.5);
    assert_eq!(config.sample_rate, 24000);
    assert_eq!(config.bottleneck_dim, None);
    // num_time_bits = ceil(log2(16)) = 4.
    assert_eq!(config.decoder.num_hidden_layers, 2);
    assert_eq!(config.decoder.head_dim, 8);
}
