//! Parity and regression tests for the VibeVoice family.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::{DiffusionNoise, GenerateParams, Model};
use crate::SpeechError;

fn testdata(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/tts_vibevoice")
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

/// Both gates are relative to the reference magnitude: the latents are
/// diffusion outputs divided by the seeded scaling factor and the
/// waveform peaks above unit amplitude on the tiny fixture. Measured
/// when the gates were frozen: latents 1.1e-6 relative worst case,
/// waveform 1.5e-6 (f32 accumulation order over the f16-mantissa
/// fixture weights), so the 2e-4 gates sit above with margin.
/// `VIBEVOICE_MEASURE=1` re-prints the measured diffs.
const LATENTS_RELATIVE_TOLERANCE: f32 = 2e-4;
const WAVEFORM_TOLERANCE: f32 = 2e-4;

fn assert_relative(got: &[f32], want: &[f32], relative: f32, what: &str) {
    let magnitude = want.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    let gate = relative * magnitude;
    let diff = max_abs_diff(got, want);
    if std::env::var("VIBEVOICE_MEASURE").is_ok() {
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

fn prompt_of(traces: &Value) -> Vec<i32> {
    f32_vec(&traces["prompt_ids"])
        .iter()
        .map(|v| *v as i32)
        .collect()
}

fn params_of(traces: &Value) -> GenerateParams {
    GenerateParams {
        cfg_scale: traces["cfg_scale"].as_f64().expect("cfg") as f32,
        max_tokens: traces["max_tokens"].as_u64().expect("max") as usize,
        ddpm_steps: None,
    }
}

/// V1: zero DDPM steps would return the initial Gaussian as the latent.
#[test]
fn zero_ddpm_steps_is_rejected() {
    let traces = read_json(&testdata("generate/traces.json"));
    let prompt = prompt_of(&traces);
    let mut m = Model::open(&testdata("generate")).expect("vibevoice model loads");
    let params = GenerateParams {
        ddpm_steps: Some(0),
        ..GenerateParams::default()
    };
    let mut noise = RecordedNoise::new(&[]);
    let error = m.generate(&prompt, &params, &mut noise).unwrap_err();
    assert!(matches!(error, SpeechError::Input { .. }), "{error}");
}

/// V2: the full-budget greedy run matches the reference latents and
/// waveform end to end (three text windows, eight diffusion samples,
/// the classifier never firing).
#[test]
fn greedy_generation_matches_the_reference_latents_and_waveform() {
    let traces = read_json(&testdata("generate/traces.json"));
    let prompt = prompt_of(&traces);
    let mut m = Model::open(&testdata("generate")).expect("vibevoice model loads");
    let params = params_of(&traces);
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

/// V3: the engineered classifier ends the run after the first latent.
#[test]
fn engineered_eos_ends_after_the_first_latent() {
    let traces = read_json(&testdata("stop/traces.json"));
    let prompt = prompt_of(&traces);
    let mut m = Model::open(&testdata("stop")).expect("vibevoice model loads");
    let params = params_of(&traces);
    let mut noise = RecordedNoise::new(traces["draws"].as_array().expect("draws"));
    let got = m
        .generate(&prompt, &params, &mut noise)
        .expect("generate runs");
    assert_eq!(
        got.latent_frames,
        traces["latent_frames"].as_u64().expect("frames") as usize
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

/// V4: generation from a loaded voice cache matches the reference run
/// that starts from the same caches (both 4D and 3D storage paths ride
/// the fixture; see the voice_kv unit test for the 3D normalizer).
#[test]
fn voice_conditioned_generation_matches_the_reference() {
    let traces = read_json(&testdata("voice/traces.json"));
    let prompt = prompt_of(&traces);
    let mut m = Model::open(&testdata("voice")).expect("vibevoice model loads");
    m.load_voice(&testdata("voice/voices/myvoice.safetensors"))
        .expect("voice cache loads");
    let params = params_of(&traces);
    let mut noise = RecordedNoise::new(traces["draws"].as_array().expect("draws"));
    let got = m
        .generate(&prompt, &params, &mut noise)
        .expect("generate runs");
    assert_eq!(
        got.latent_frames,
        traces["latent_frames"].as_u64().expect("frames") as usize
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

/// V5: the config parser resolves the documented defaults.
#[test]
fn config_defaults_match_the_reference_dataclasses() {
    let value: Value =
        serde_json::from_str(&std::fs::read_to_string(testdata("generate/config.json")).unwrap())
            .unwrap();
    let config = super::ModelConfig::from_json(&value).expect("config parses");
    assert_eq!(config.sample_rate, 8000);
    assert_eq!(config.acoustic_vae_dim, 8);
    assert_eq!(config.tts_backbone_num_hidden_layers, 1);
    // decoder_ratios/depths absent: the encoder values fall back, the
    // depths reversed.
    assert_eq!(config.acoustic_tokenizer.ratios, vec![2, 2]);
    assert_eq!(config.acoustic_tokenizer.depths, vec![2, 2]);
    assert_eq!(config.acoustic_tokenizer.channels, 1);
    assert!(config.acoustic_tokenizer.conv_bias);
    assert_eq!(config.diffusion_head.prediction_type, "v_prediction");
    assert_eq!(config.diffusion_head.ddpm_num_steps, 20);
    assert_eq!(config.diffusion_head.ddpm_num_inference_steps, 4);
    assert_eq!(config.diffusion_head.ddpm_beta_schedule, "cosine");
    // The split: 2 total layers, 1 upper.
    assert_eq!(config.decoder.num_hidden_layers, 2);
    assert_eq!(config.decoder.head_dim, 4);
}

/// V6: the documented empty-text refusal path (no tts state builds, so
/// the reference breaks out of the loop with no audio).
#[test]
fn empty_prompt_produces_no_audio() {
    let mut m = Model::open(&testdata("generate")).expect("vibevoice model loads");
    let params = GenerateParams::default();
    let mut noise = RecordedNoise::new(&[]);
    let got = m.generate(&[], &params, &mut noise).expect("generate runs");
    assert_eq!(got.latent_frames, 0);
    assert!(got.samples.is_empty());
}

/// V7: the batch-free 3D (rows, kv, d) voice-key normalizer rotates to
/// head-major; the fixture's 4D files cover the identity branch.
#[test]
fn voice_kv_normalizes_both_storage_orders() {
    use super::voice_kv;
    // 3D (rows=2, kv=2, dim=1): row-major [r0k0, r0k1, r1k0, r1k1].
    let got = voice_kv("k", &[2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]).expect("3d");
    assert_eq!(got, vec![1.0, 3.0, 2.0, 4.0]);
    // 4D Swift-style (1, kv=2, rows=2, dim=1) is already head-major.
    let got = voice_kv("k", &[1, 2, 2, 1], vec![1.0, 3.0, 2.0, 4.0]).expect("4d");
    assert_eq!(got, vec![1.0, 3.0, 2.0, 4.0]);
    assert!(voice_kv("k", &[2, 2, 2, 1], vec![0.0; 8]).is_err());
}
