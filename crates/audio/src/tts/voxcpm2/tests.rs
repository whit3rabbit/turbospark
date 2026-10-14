//! Parity and regression tests for the VoxCPM2 family.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::{ContinuationPrompt, FlowNoise, GenerateParams, Model, VoicePrompt};
use crate::tts::voxcpm::SeededFlowNoise;
use crate::SpeechError;

fn testdata(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/tts_voxcpm2")
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

fn i32_vec(value: &Value) -> Vec<i32> {
    value
        .as_array()
        .expect("values")
        .iter()
        .map(|v| v.as_i64().expect("value") as i32)
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Stage gates are relative to the reference magnitude. The prompt and
/// backbone stages hold 2e-4 (measured at or below 2e-6). The
/// generate loop carries the same transcendental floor as voxcpm
/// (sinusoidal time embeddings at angles up to 1000 radians, see the
/// voxcpm README), amplified by the per-step feedback, so collected
/// latents and waveforms are gated by [`FEEDBACK_TOLERANCE`], frozen
/// at the measured worst cases with margin; `VOXCPM2_MEASURE=1`
/// re-prints.
const RELATIVE_TOLERANCE: f32 = 2e-4;
const FEEDBACK_TOLERANCE: f32 = 2e-2;

fn assert_relative(got: &[f32], want: &[f32], what: &str) {
    let magnitude = want.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    let gate = RELATIVE_TOLERANCE * magnitude;
    let diff = max_abs_diff(got, want);
    if std::env::var("VOXCPM2_MEASURE").is_ok() {
        eprintln!("{what}: diff {diff:e} relative {:.3e}", diff / magnitude);
    }
    assert!(
        diff <= gate,
        "{what} diff {diff} exceeds {gate} (relative {:.3e})",
        diff / magnitude
    );
}

fn assert_feedback_relative(got: &[f32], want: &[f32], what: &str) {
    let magnitude = want.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    let gate = FEEDBACK_TOLERANCE * magnitude;
    let diff = max_abs_diff(got, want);
    if std::env::var("VOXCPM2_MEASURE").is_ok() {
        eprintln!(
            "{what}: diff {diff:e} relative {:.3e} [feedback]",
            diff / magnitude
        );
    }
    assert!(
        diff <= gate,
        "{what} diff {diff} exceeds {gate} (relative {:.3e})",
        diff / magnitude
    );
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

impl FlowNoise for RecordedNoise {
    fn normal(&mut self, n: usize) -> Vec<f32> {
        let draw = self
            .draws
            .get(self.cursor)
            .unwrap_or_else(|| panic!("draw {} missing", self.cursor))
            .clone();
        self.cursor += 1;
        assert_eq!(draw.len(), n, "draw {} shape", self.cursor - 1);
        draw
    }
}

fn params_of(traces: &Value) -> GenerateParams {
    GenerateParams {
        max_tokens: traces["generate"]["max_tokens"].as_u64().expect("max") as usize,
        warmup_patches: traces["generate"]["warmup_patches"]
            .as_u64()
            .expect("warmup") as usize,
        min_tokens: traces["generate"]["min_tokens"].as_u64().expect("min") as usize,
        inference_timesteps: traces["generate"]["inference_timesteps"]
            .as_u64()
            .expect("steps") as usize,
        cfg_value: traces["generate"]["cfg_value"].as_f64().expect("cfg") as f32,
        streaming_prefix_len: traces["generate"]["streaming_prefix_len"]
            .as_u64()
            .expect("prefix") as usize,
    }
}

/// The combined caller ids (before the start token / pad zeros). In
/// reference mode the recorded ids start with the
/// `[103 | zeros | 104]` prefix the model builds itself, so strip it.
fn ids_of(traces: &Value) -> Vec<i32> {
    let ids = i32_vec(&traces["prompt"]["input_ids"]);
    let start = if traces["mode"] == "ref_clone" {
        ids.iter()
            .position(|&id| id == super::REF_AUDIO_END_TOKEN)
            .map(|p| p + 1)
            .unwrap_or(0)
    } else {
        0
    };
    let ids = &ids[start..];
    let end = ids
        .iter()
        .position(|&id| id == super::AUDIO_START_TOKEN)
        .unwrap_or(ids.len());
    ids[..end].to_vec()
}

fn prompts_of(traces: &Value) -> (Option<VoicePrompt>, Option<ContinuationPrompt>) {
    let voice = traces["ref"].as_object().map(|r| VoicePrompt {
        ref_audio: f32_vec(&r["ref_audio"]),
    });
    let continuation = traces["continuation"]
        .as_object()
        .map(|c| ContinuationPrompt {
            prompt_audio: f32_vec(&c["prompt_audio"]),
        });
    (voice, continuation)
}

/// V1: the config parser resolves the documented defaults.
#[test]
fn config_defaults_match_the_reference_dataclasses() {
    let value: Value =
        serde_json::from_str(&std::fs::read_to_string(testdata("zero_shot/config.json")).unwrap())
            .unwrap();
    let config = super::ModelConfig::from_json(&value).expect("config parses");
    assert_eq!(config.lm_config.hidden_size, 32);
    assert!(config.lm_config.use_mup, "voxcpm2 defaults to mup");
    assert_eq!(config.lm_config.scale_emb, 12);
    assert_eq!(config.residual_lm_num_layers, 2);
    assert!(!config.residual_lm_no_rope);
    assert_eq!(config.dit_config.hidden_dim, 32);
    assert!(!config.dit_config.dit_mean_mode);
    assert_eq!(config.patch_size, 2);
    assert_eq!(config.feat_dim, 8);
    assert_eq!(config.scalar_quantization_latent_dim, 16);
    assert_eq!(config.audio_vae_config.sample_rate, 16000);
    assert_eq!(config.audio_vae_config.out_sample_rate, 48000);
    assert_eq!(
        config.audio_vae_config.sr_bin_boundaries,
        Some(vec![20000, 30000, 40000])
    );
    assert_eq!(config.audio_vae_config.cond_type, "scale_bias");
    assert_eq!(config.audio_vae_config.chunk_size(), 2);
    assert_eq!(config.audio_vae_config.decode_chunk_size(), 4);

    // The reference dataclass defaults on a bare object.
    let bare = super::ModelConfig::from_json(&serde_json::json!({})).expect("bare");
    assert_eq!(bare.lm_config.hidden_size, 1024);
    assert!(bare.lm_config.use_mup);
    assert_eq!(bare.dit_config.num_layers, 4);
    assert_eq!(bare.patch_size, 4);
    assert_eq!(bare.residual_lm_num_layers, 8);
    assert_eq!(bare.audio_vae_config.out_sample_rate, 48000);
    // mean_mode normalizes to dit_mean_mode.
    let named = super::ModelConfig::from_json(&serde_json::json!({
        "dit_config": {"mean_mode": true}
    }))
    .expect("named");
    assert!(named.dit_config.dit_mean_mode);
}

/// V2: the zero-shot prompt and first backbone passes match.
#[test]
fn zero_shot_prompt_stages_match_the_reference() {
    let traces = read_json(&testdata("zero_shot/traces.json"));
    let mut m = Model::open(&testdata("zero_shot")).expect("voxcpm2 model loads");
    let scale_emb = if m.config.lm_config.use_mup {
        m.config.lm_config.scale_emb as f32
    } else {
        1.0
    };
    let bundle = m
        .build_prompt(&ids_of(&traces), None, None, scale_emb, 4)
        .expect("prompt builds");
    assert_relative(
        &bundle.combined,
        &f32_vec(&traces["prompt"]["combined_embed"]),
        "combined_embed",
    );
    let hidden = m.config.lm_config.hidden_size;
    let total = bundle.combined.len() / hidden;
    let enc = m
        .base_lm
        .forward(
            &bundle.combined,
            total,
            Some(&mut m.lm_cache),
            super::super::voxcpm::minicpm::Mask::Causal,
        )
        .expect("prefill");
    let fsq = m.fsq.forward(&enc, total);
    let mut mixed = enc.clone();
    for row in 0..total {
        if bundle.audio_mask[row] == 0.0 {
            continue;
        }
        for col in 0..hidden {
            mixed[row * hidden + col] = fsq[row * hidden + col];
        }
    }
    let last = &mixed[(total - 1) * hidden..];
    assert_relative(last, &f32_vec(&traces["prompt"]["lm_hidden"]), "lm_hidden");
    // V2 fusion residual input.
    let mut fusion_in = vec![0.0f32; total * hidden * 2];
    for row in 0..total {
        for col in 0..hidden {
            fusion_in[row * hidden * 2 + col] = mixed[row * hidden + col];
            fusion_in[row * hidden * 2 + hidden + col] =
                bundle.audio_mask[row] * bundle.feat_embed[row * hidden + col];
        }
    }
    let residual_input = m.fusion_concat_proj.forward(&fusion_in, total);
    let res = m
        .residual_lm
        .forward(
            &residual_input,
            total,
            Some(&mut m.res_cache),
            super::super::voxcpm::minicpm::Mask::Causal,
        )
        .expect("residual prefill");
    let res_hidden = &res[(total - 1) * hidden..];
    assert_relative(
        res_hidden,
        &f32_vec(&traces["prompt"]["residual_hidden"]),
        "residual_hidden",
    );
}

/// V3: the reference prefix ([103 | patches | 104]) masks and features
/// match, and the condition starts at the last patch.
#[test]
fn ref_clone_prompt_stages_match_the_reference() {
    let traces = read_json(&testdata("ref_clone/traces.json"));
    let mut m = Model::open(&testdata("ref_clone")).expect("voxcpm2 model loads");
    let scale_emb = if m.config.lm_config.use_mup {
        m.config.lm_config.scale_emb as f32
    } else {
        1.0
    };
    let (voice, _) = prompts_of(&traces);
    let bundle = m
        .build_prompt(&ids_of(&traces), voice.as_ref(), None, scale_emb, 4)
        .expect("prompt builds");
    assert_relative(
        &bundle.audio_mask,
        &f32_vec(&traces["prompt"]["audio_mask"]),
        "audio_mask",
    );
    assert_relative(
        &bundle.combined,
        &f32_vec(&traces["prompt"]["combined_embed"]),
        "combined_embed",
    );
    assert_relative(
        &bundle.prefix_feat_cond,
        &f32_vec(&traces["prompt"]["prefix_feat_cond"]),
        "prefix_feat_cond",
    );
}

/// V3b: the scipy Fourier resample and the VAE patch encode of the
/// reference audio match stage by stage.
#[test]
fn ref_encode_stages_match_the_reference() {
    let traces = read_json(&testdata("ref_clone/traces.json"));
    let ref_audio = f32_vec(&traces["ref"]["ref_audio"]);
    let resampled = super::audio_vae::fourier_resample(
        &ref_audio,
        traces["prompt"]["ref_resampled"].as_array().unwrap().len(),
    )
    .expect("resample runs");
    eprintln!("got resampled {resampled:.6?}");
    eprintln!(
        "want resampled {:.6?}",
        f32_vec(&traces["prompt"]["ref_resampled"])
    );
    assert_relative(
        &resampled,
        &f32_vec(&traces["prompt"]["ref_resampled"]),
        "resample",
    );
    let mut m = Model::open(&testdata("ref_clone")).expect("voxcpm2 model loads");
    let (rows, patches) = m.encode_wav(&ref_audio, "right").expect("encode runs");
    assert_eq!(
        patches.len(),
        f32_vec(&traces["prompt"]["ref_patches"]).len()
    );
    assert_relative(
        &patches,
        &f32_vec(&traces["prompt"]["ref_patches"]),
        "patches",
    );
    let _ = rows;
}

/// V4: the full zero-shot run matches end to end; the stop head ends
/// the run at `real_steps == min_tokens + 1`.
#[test]
fn zero_shot_generation_matches_the_reference_end_to_end() {
    let traces = read_json(&testdata("zero_shot/traces.json"));
    let mut m = Model::open(&testdata("zero_shot")).expect("voxcpm2 model loads");
    let params = params_of(&traces);
    let ids = ids_of(&traces);
    let mut noise = RecordedNoise::new(traces["noise_draws"].as_array().expect("draws"));
    let got = m
        .generate(&ids, &params, &mut noise, None, None)
        .expect("generate runs");
    assert_eq!(got.steps, traces["steps"].as_array().unwrap().len());
    assert!(got.stopped);
    assert_eq!(
        got.token_count,
        traces["prompt"]["token_count"].as_u64().unwrap() as usize
    );
    assert_feedback_relative(&got.latents, &f32_vec(&traces["latents"]), "latents");
    assert_feedback_relative(&got.samples, &f32_vec(&traces["waveform"]), "waveform");
    assert_eq!(got.sample_rate, 48000);
}

/// V5: the never-stop run collects exactly max_tokens patches.
#[test]
fn full_budget_generation_runs_to_max_tokens() {
    let traces = read_json(&testdata("length/traces.json"));
    let mut m = Model::open(&testdata("length")).expect("voxcpm2 model loads");
    let params = params_of(&traces);
    let ids = ids_of(&traces);
    let mut noise = RecordedNoise::new(traces["noise_draws"].as_array().expect("draws"));
    let got = m
        .generate(&ids, &params, &mut noise, None, None)
        .expect("generate runs");
    assert_eq!(got.steps, params.max_tokens);
    assert!(!got.stopped);
    assert_feedback_relative(&got.latents, &f32_vec(&traces["latents"]), "latents");
    assert_feedback_relative(&got.samples, &f32_vec(&traces["waveform"]), "waveform");
}

/// V6: the reference-prefix run matches end to end.
#[test]
fn ref_clone_generation_matches_the_reference_end_to_end() {
    let traces = read_json(&testdata("ref_clone/traces.json"));
    let mut m = Model::open(&testdata("ref_clone")).expect("voxcpm2 model loads");
    let params = params_of(&traces);
    let ids = ids_of(&traces);
    let (voice, _) = prompts_of(&traces);
    let mut noise = RecordedNoise::new(traces["noise_draws"].as_array().expect("draws"));
    let got = m
        .generate(&ids, &params, &mut noise, voice.as_ref(), None)
        .expect("generate runs");
    assert!(got.stopped);
    assert_feedback_relative(&got.latents, &f32_vec(&traces["latents"]), "latents");
    assert_feedback_relative(&got.samples, &f32_vec(&traces["waveform"]), "waveform");
}

/// V7: the continuation run keeps the prompt prefix patches, trims the
/// decode front, and matches end to end.
#[test]
fn continuation_generation_matches_the_reference_end_to_end() {
    let traces = read_json(&testdata("continuation/traces.json"));
    let mut m = Model::open(&testdata("continuation")).expect("voxcpm2 model loads");
    let params = params_of(&traces);
    let ids = ids_of(&traces);
    let (_, continuation) = prompts_of(&traces);
    let mut noise = RecordedNoise::new(traces["noise_draws"].as_array().expect("draws"));
    let got = m
        .generate(&ids, &params, &mut noise, None, continuation.as_ref())
        .expect("generate runs");
    assert!(got.continued);
    assert!(!got.stopped);
    let want_latents = f32_vec(&traces["latents"]);
    assert_eq!(got.latents.len(), want_latents.len(), "prefix + generated");
    assert_feedback_relative(&got.latents, &want_latents, "latents");
    assert_feedback_relative(&got.samples, &f32_vec(&traces["waveform"]), "waveform");
}

/// V8: the sway t-span matches the recorded schedule exactly.
#[test]
fn euler_t_span_matches_the_recorded_sway_schedule() {
    let traces = read_json(&testdata("zero_shot/traces.json"));
    let steps = traces["generate"]["inference_timesteps"].as_u64().unwrap() as usize;
    let span = super::dit::UnifiedCfm::t_span(steps);
    assert_eq!(
        span.len(),
        traces["steps"][0]["t_span"].as_array().unwrap().len()
    );
    assert!(
        max_abs_diff(&span, &f32_vec(&traces["steps"][0]["t_span"])) < 1e-5,
        "t_span exact"
    );
}

/// V9: the crate-local threefry RNG reproduces the recorded MLX
/// normal stream from the same seed.
#[test]
fn seeded_threefry_reproduces_the_recorded_normal_draws() {
    let traces = read_json(&testdata("zero_shot/traces.json"));
    let draws = traces["noise_draws"].as_array().expect("draws");
    let mut noise = SeededFlowNoise::new(101);
    for draw in draws {
        let values = f32_vec(&draw["values"]);
        let got = noise.normal(values.len());
        assert!(max_abs_diff(&got, &values) < 1e-6, "seeded draw");
    }
}

/// V10: degenerate parameters are refused.
#[test]
fn degenerate_generation_parameters_are_refused() {
    let traces = read_json(&testdata("zero_shot/traces.json"));
    let mut m = Model::open(&testdata("zero_shot")).expect("voxcpm2 model loads");
    let ids = ids_of(&traces);
    let mut noise = RecordedNoise::new(&[]);
    let params = GenerateParams {
        max_tokens: 0,
        ..params_of(&traces)
    };
    let error = m
        .generate(&ids, &params, &mut noise, None, None)
        .unwrap_err();
    assert!(matches!(error, SpeechError::Input { .. }), "{error}");
    let params = GenerateParams {
        inference_timesteps: 0,
        ..params_of(&traces)
    };
    let error = m
        .generate(&ids, &params, &mut noise, None, None)
        .unwrap_err();
    assert!(matches!(error, SpeechError::Input { .. }), "{error}");
}
