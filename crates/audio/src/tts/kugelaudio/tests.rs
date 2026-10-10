//! Parity and regression tests for the KugelAudio family.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::{DiffusionNoise, GenerateParams, Model, SPEECH_START_ID};

fn testdata(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/tts_kugelaudio")
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

/// The latents are the raw diffusion samples divided by the
/// checkpoint's speech scaling factor, which drives their amplitude to
/// the hundreds on the tiny fixture; the gate is relative to the
/// reference magnitude. Measured relative difference when the gate was
/// frozen: 3.7e-6 (f32 accumulation order), so the gate sits well above
/// it.
const LATENTS_RELATIVE_TOLERANCE: f32 = 1e-4;

/// Measured maximum difference against the fixture waveforms when the
/// gate was frozen; the assertion uses this value, not a re-measured
/// one.
const WAVEFORM_TOLERANCE: f32 = 2e-4;

fn assert_relative(got: &[f32], want: &[f32], relative: f32, what: &str) {
    let magnitude = want.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    let gate = relative * magnitude;
    let diff = max_abs_diff(got, want);
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

fn prompt_of(traces: &Value) -> Vec<i32> {
    f32_vec(&traces["prompt_ids"])
        .iter()
        .map(|v| *v as i32)
        .collect()
}

fn model(case: &str) -> Model {
    Model::open(&testdata(case)).expect("kugelaudio model loads")
}

#[test]
fn greedy_generation_matches_the_reference_latents_and_waveform() {
    let traces = read_json(&testdata("generate/traces.json"));
    let prompt = prompt_of(&traces);
    let mut m = model("generate");
    let params = GenerateParams {
        cfg_scale: traces["cfg_scale"].as_f64().expect("cfg") as f32,
        max_tokens: traces["max_tokens"].as_u64().expect("max") as usize,
        ddpm_steps: None,
    };
    let mut noise = RecordedNoise::new(traces["draws"].as_array().expect("draws"));
    let got = m
        .generate(&prompt, &params, &mut noise)
        .expect("generate runs");
    assert_eq!(
        got.latent_frames,
        traces["latent_frames"].as_u64().expect("frames") as usize
    );
    assert_eq!(
        got.token_count,
        traces["token_count"].as_u64().expect("tokens") as usize
    );
    assert_relative(
        &got.latents,
        &f32_vec(&traces["latents"]),
        LATENTS_RELATIVE_TOLERANCE,
        "latents",
    );
    assert_relative(
        &got.samples,
        &f32_vec(&traces["waveform"]),
        WAVEFORM_TOLERANCE,
        "waveform",
    );
}

#[test]
fn engineered_stop_ends_without_latents() {
    let traces = read_json(&testdata("stop/traces.json"));
    let prompt = prompt_of(&traces);
    let mut m = model("stop");
    let params = GenerateParams {
        cfg_scale: traces["cfg_scale"].as_f64().expect("cfg") as f32,
        max_tokens: traces["max_tokens"].as_u64().expect("max") as usize,
        ddpm_steps: None,
    };
    let mut noise = RecordedNoise::new(traces["draws"].as_array().expect("draws"));
    let got = m
        .generate(&prompt, &params, &mut noise)
        .expect("generate runs");
    assert_eq!(got.latent_frames, 0);
    assert!(got.samples.is_empty());
    assert!(got.latents.is_empty());
    assert_eq!(
        got.token_count,
        traces["token_count"].as_u64().expect("tokens") as usize
    );
}

#[test]
fn prompt_text_matches_the_reference_builder() {
    assert_eq!(
        super::format_speaker_text("hello world"),
        "Speaker 0: hello world"
    );
    assert_eq!(
        super::format_speaker_text("  Speaker 1: hi  "),
        "Speaker 1: hi"
    );
    assert_eq!(
        super::text_section("Speaker 0: hi"),
        " Text input:\n Speaker 0: hi\n Speech output:\n"
    );
    let full = super::prompt_text("hello");
    assert!(full.starts_with(" Transform the text provided"));
    assert!(full.ends_with(" Text input:\n Speaker 0: hello\n Speech output:\n"));
    // The recorded fixture prompt ends with the speech start id.
    let traces = read_json(&testdata("generate/traces.json"));
    let prompt = f32_vec(&traces["prompt_ids"])
        .iter()
        .map(|v| *v as i32)
        .collect::<Vec<_>>();
    assert_eq!(*prompt.last().expect("id"), SPEECH_START_ID);
}
