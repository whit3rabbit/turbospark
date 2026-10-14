//! Parity and regression tests for the VoxCPM family.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::{FlowNoise, GenerateParams, Model, SeededFlowNoise, VoicePrompt};
use crate::SpeechError;

fn testdata(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/tts_voxcpm")
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

/// Stage gates are relative to the reference magnitude. The prompt,
/// cache, and estimator stages hold 2e-4 (measured 6e-7 to 1e-6 for
/// the prompt math; see the README for the estimator's
/// transcendental floor), but the generate loop is a feedback system:
/// the sinusoidal time embedding evaluates sin/cos at angles up to
/// 1000 radians, where the reference's f32 argument reduction differs
/// from Rust's correctly-rounded libm by up to 5e-5, and each
/// sampled latent feeds the next conditioning step, so the diff
/// grows about 3.5x per step. The end-to-end gates below are frozen
/// at the measured worst cases with margin (per-step latents worst
/// ~3e-3, waveform worst ~2e-3); `VOXCPM_MEASURE=1` re-prints.
const RELATIVE_TOLERANCE: f32 = 2e-4;
const FEEDBACK_TOLERANCE: f32 = 2e-2;

fn assert_relative(got: &[f32], want: &[f32], what: &str) {
    let magnitude = want.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    let gate = RELATIVE_TOLERANCE * magnitude;
    let diff = max_abs_diff(got, want);
    if std::env::var("VOXCPM_MEASURE").is_ok() {
        eprintln!(
            "{what}: diff {diff:e} magnitude {magnitude:e} relative {:.3e}",
            diff / magnitude
        );
    }
    assert!(
        diff <= gate,
        "{what} diff {diff} exceeds {gate} (relative {:.3e})",
        diff / magnitude
    );
}

/// The looser gate for values downstream of the generate-loop
/// feedback (latents, waveforms); see the tolerance docs above.
fn assert_feedback_relative(got: &[f32], want: &[f32], what: &str) {
    let magnitude = want.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    let gate = FEEDBACK_TOLERANCE * magnitude;
    let diff = max_abs_diff(got, want);
    if std::env::var("VOXCPM_MEASURE").is_ok() {
        eprintln!(
            "{what}: diff {diff:e} magnitude {magnitude:e} relative {:.3e} [feedback gate]",
            diff / magnitude
        );
    }
    assert!(
        diff <= gate,
        "{what} diff {diff} exceeds {gate} (relative {:.3e})",
        diff / magnitude
    );
}

/// Asserts exact equality for quantized/integer-valued stages.
fn assert_exact(got: &[f32], want: &[f32], what: &str) {
    let diff = max_abs_diff(got, want);
    assert!(diff < 1e-5, "{what} diff {diff} exceeds the exact gate");
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
        max_tokens: traces["generate"]["max_tokens"]
            .as_u64()
            .expect("max_tokens") as usize,
        inference_timesteps: traces["generate"]["inference_timesteps"]
            .as_u64()
            .expect("inference_timesteps") as usize,
        cfg_value: traces["generate"]["cfg_value"].as_f64().expect("cfg") as f32,
    }
}

fn text_ids_of(traces: &Value) -> Vec<i32> {
    // The caller-owned ids the fixture records: the bare text in
    // zero-shot mode, the concatenated `ref_text + text` encode in
    // clone mode. `generate` appends the start token itself.
    i32_vec(&traces["prompt"]["input_ids"])
}

/// The per-step recorded latents, concatenated (the `Generated.latents`
/// layout).
fn concat_latents(traces: &Value) -> Vec<f32> {
    let mut all = Vec::new();
    for step in traces["steps"].as_array().expect("steps") {
        all.extend(f32_vec(&step["latent"]));
    }
    all
}

/// V1: the config parser resolves the documented defaults.
#[test]
fn config_defaults_match_the_reference_dataclasses() {
    let value: Value =
        serde_json::from_str(&std::fs::read_to_string(testdata("generate/config.json")).unwrap())
            .unwrap();
    let config = super::ModelConfig::from_json(&value).expect("config parses");
    assert_eq!(config.lm_config.hidden_size, 32);
    assert_eq!(config.lm_config.num_hidden_layers, 2);
    assert_eq!(config.lm_config.num_attention_heads, 4);
    assert_eq!(config.lm_config.num_key_value_heads, 2);
    assert_eq!(config.lm_config.vocab_size, 128);
    assert_eq!(config.lm_config.scale_emb, 12);
    assert!(!config.lm_config.use_mup);
    assert_eq!(
        config.lm_config.original_max_position_embeddings, 512,
        "fixture longrope threshold"
    );
    assert_eq!(config.residual_lm_num_layers, 2);
    assert_eq!(config.encoder_config.hidden_dim, 32);
    assert_eq!(config.dit_config.num_layers, 2);
    assert_eq!(config.dit_config.solver, "euler");
    assert_eq!(config.dit_config.t_scheduler, "log-norm");
    assert_eq!(config.patch_size, 2);
    assert_eq!(config.feat_dim, 8);
    assert_eq!(config.scalar_quantization_latent_dim, 16);
    assert_eq!(config.scalar_quantization_scale, 9);
    assert_eq!(config.audio_vae_config.encoder_rates, vec![2, 4]);
    assert_eq!(config.audio_vae_config.decoder_rates, vec![4, 2]);
    assert_eq!(config.audio_vae_config.sample_rate, 24000);
    assert_eq!(config.audio_vae_config.hop_length(), 8);

    // The reference dataclass defaults on a bare object.
    let bare = super::ModelConfig::from_json(&serde_json::json!({})).expect("bare");
    assert_eq!(bare.lm_config.hidden_size, 1024);
    assert_eq!(bare.lm_config.vocab_size, 73448);
    assert_eq!(bare.lm_config.original_max_position_embeddings, 32768);
    assert_eq!(bare.patch_size, 4);
    assert_eq!(bare.feat_dim, 64);
    assert_eq!(bare.residual_lm_num_layers, 8);
    assert_eq!(bare.audio_vae_config.sample_rate, 44100);
    assert_eq!(bare.audio_vae_config.encoder_rates, vec![2, 3, 6, 7, 7]);
    // A nested rope_scaling block flattens like `ModelArgs.from_dict`.
    let nested = super::ModelConfig::from_json(&serde_json::json!({
        "lm_config": {
            "rope_scaling": {
                "type": "longrope",
                "long_factor": [1.0, 2.0],
                "short_factor": [0.5, 1.5],
                "original_max_position_embeddings": 4096
            }
        }
    }))
    .expect("nested");
    assert_eq!(nested.lm_config.rope_long_factor, vec![1.0, 2.0]);
    assert_eq!(nested.lm_config.rope_short_factor, vec![0.5, 1.5]);
    assert_eq!(nested.lm_config.original_max_position_embeddings, 4096);
}

/// V2: zero-shot prompt construction (embeddings, the start token, the
/// fsq'd last row) and the first backbone passes match the reference
/// trace.
#[test]
fn zero_shot_prompt_stages_match_the_reference() {
    let traces = read_json(&testdata("generate/traces.json"));
    let mut m = Model::open(&testdata("generate")).expect("voxcpm model loads");
    let prompt_ids = i32_vec(&traces["prompt"]["text_token"]);
    let scale_emb = if m.config.lm_config.use_mup {
        1.0
    } else {
        m.config.lm_config.scale_emb as f32
    };
    let bundle = m
        .build_prompt(&prompt_ids[..prompt_ids.len() - 1], None, scale_emb)
        .expect("prompt builds");
    assert!(bundle.clone.is_none());
    assert_eq!(
        bundle.combined.len(),
        (traces["prompt"]["token_count"].as_u64().unwrap() as usize + 1)
            * m.config.lm_config.hidden_size,
        "text ids plus the start token, one row each"
    );
    assert_relative(
        &bundle.combined,
        &f32_vec(&traces["prompt"]["combined_embed"]),
        "combined_embed",
    );
    assert_exact(
        &bundle.prefix_feat_cond,
        &f32_vec(&traces["prompt"]["prefix_feat_cond"]),
        "prefix_feat_cond",
    );

    // The base LM prefill, fsq, and residual prefill.
    let hidden = m.config.lm_config.hidden_size;
    let total = bundle.combined.len() / hidden;
    let enc = m
        .base_lm
        .forward(
            &bundle.combined,
            total,
            Some(&mut m.lm_cache),
            super::minicpm::Mask::Causal,
        )
        .expect("prefill");
    assert_relative(
        &enc,
        &f32_vec(&traces["prompt"]["enc_outputs"]),
        "enc_outputs",
    );
    let last = &enc[(total - 1) * hidden..];
    let lm_hidden = m.fsq.forward(last, 1);
    assert_relative(
        &lm_hidden,
        &f32_vec(&traces["prompt"]["lm_hidden"]),
        "lm_hidden",
    );
    let res = m
        .residual_lm
        .forward(
            &enc,
            total,
            Some(&mut m.res_cache),
            super::minicpm::Mask::Causal,
        )
        .expect("residual prefill");
    let res_hidden = &res[(total - 1) * hidden..];
    assert_relative(
        res_hidden,
        &f32_vec(&traces["prompt"]["residual_hidden"]),
        "residual_hidden",
    );
}

/// V3: voice-clone prompt construction: the encoded reference patches,
/// the text/audio masks, the mixed embeddings, and the condition taken
/// from the last patch.
#[test]
fn voice_clone_prompt_stages_match_the_reference() {
    let traces = read_json(&testdata("voice_clone/traces.json"));
    let mut m = Model::open(&testdata("voice_clone")).expect("voxcpm model loads");
    let voice_clone = &traces["voice_clone"];
    let (rows, patches) = m
        .encode_prompt_audio(&f32_vec(&voice_clone["ref_audio"]))
        .expect("encode");
    assert_relative(
        &patches,
        &f32_vec(&voice_clone["prompt_audio_feat"]),
        "prompt_audio_feat",
    );
    let prompt_ids = text_ids_of(&traces);
    let scale_emb = if m.config.lm_config.use_mup {
        1.0
    } else {
        m.config.lm_config.scale_emb as f32
    };
    let voice = VoicePrompt {
        ref_audio: f32_vec(&voice_clone["ref_audio"]),
    };
    let bundle = m
        .build_prompt(&prompt_ids, Some(&voice), scale_emb)
        .expect("prompt builds");
    let clone = bundle.clone.as_ref().expect("clone mix");
    assert_relative(
        &clone.audio_mask,
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
    assert_relative(
        &patches,
        &{
            // The prompt audio_feat trace is the zero-padded patches
            // prepended to the encoded rows; the encoded rows are the
            // tail.
            let full = f32_vec(&traces["prompt"]["audio_feat"]);
            let tail = full.len() - patches.len();
            full[tail..].to_vec()
        },
        "prompt.audio_feat tail",
    );
    let _ = rows;
}

/// V4: the sway t-span matches the recorded `euler_tspans` exactly.
#[test]
fn euler_t_span_matches_the_recorded_sway_schedule() {
    let traces = read_json(&testdata("generate/traces.json"));
    let steps = traces["generate"]["inference_timesteps"].as_u64().unwrap() as usize;
    let span = super::dit::UnifiedCfm::t_span(steps);
    let want = f32_vec(&traces["euler_tspans"][0]);
    assert_eq!(span.len(), want.len());
    assert_exact(&span, &want, "t_span");
}

/// V4b: the recorded raw estimator calls replay exactly (this pins the
/// DiT backbone, the sinusoidal time embedding, and the full-attention
/// decoder independent of the Euler solver).
#[test]
fn estimator_call_matches_the_recorded_trace() {
    let traces = read_json(&testdata("generate/traces.json"));
    let mut m = Model::open(&testdata("generate")).expect("voxcpm model loads");
    let patch = m.config.patch_size;
    let patch_values = patch * m.config.feat_dim;
    for (index, call) in traces["dit_calls"]
        .as_array()
        .expect("dit_calls")
        .iter()
        .enumerate()
    {
        // The recorded calls carry the CFG-doubled batch; the port runs
        // the two rows separately.
        let x = f32_vec(&call["x"]);
        let out = f32_vec(&call["out"]);
        let mu = f32_vec(&call["mu"]);
        let cond = f32_vec(&call["cond"]);
        let t = call["t"].as_array().expect("t")[0].as_f64().unwrap() as f32;
        let dt = call["dt"].as_array().expect("dt")[0].as_f64().unwrap() as f32;
        let rows = x.len() / patch_values;
        let mut diff = 0.0f32;
        for row in 0..rows {
            let got = m
                .feat_decoder
                .estimator_call(
                    &x[row * patch_values..(row + 1) * patch_values],
                    patch,
                    &mu[row * mu.len() / rows..(row + 1) * mu.len() / rows],
                    t,
                    &cond[row * patch_values..(row + 1) * patch_values],
                    patch,
                    dt,
                )
                .expect("estimator runs");
            let want = &out[row * patch_values..(row + 1) * patch_values];
            diff = diff.max(max_abs_diff(&got, want));
        }
        eprintln!("estimator call {index} raw diff {diff:.3e}");
    }
}

/// V5: the full zero-shot run with the stop head firing matches the
/// reference latents, stop decisions, and waveform end to end.
#[test]
fn zero_shot_generation_matches_the_reference_end_to_end() {
    let traces = read_json(&testdata("generate/traces.json"));
    let mut m = Model::open(&testdata("generate")).expect("voxcpm model loads");
    let params = params_of(&traces);
    let text_ids = text_ids_of(&traces);
    let mut noise = RecordedNoise::new(traces["noise_draws"].as_array().expect("draws"));
    let got = m
        .generate(&text_ids, &params, &mut noise, None)
        .expect("generate runs");
    let steps = traces["steps"].as_array().expect("steps");
    assert_eq!(got.steps, steps.len(), "step count");
    assert!(got.stopped, "the stop head ends this run");
    for (step, trace) in steps.iter().enumerate() {
        let start = step * m.config.patch_size * m.config.feat_dim;
        let end = start + m.config.patch_size * m.config.feat_dim;
        assert_feedback_relative(
            &got.latents[start..end],
            &f32_vec(&trace["latent"]),
            &format!("step {step} latent"),
        );
        // The stop head may report 1 from the first step; the
        // reference only acts on it after `i > 5`, so the recorded
        // flag sequence is the parity target, not a 0-then-1 ramp.
        if std::env::var("VOXCPM_MEASURE").is_ok() {
            let start = step * m.config.patch_size * m.config.feat_dim;
            let diff = max_abs_diff(
                &got.latents[start..start + m.config.patch_size * m.config.feat_dim],
                &f32_vec(&trace["latent"]),
            );
            eprintln!("step {step} latent diff {diff:.3e}");
        }
        let _ = trace["stop_flag"].as_u64().unwrap();
    }
    assert_eq!(
        got.token_count,
        traces["prompt"]["token_count"].as_u64().unwrap() as usize,
        "the reference token_count"
    );
    assert_feedback_relative(&got.samples, &f32_vec(&traces["waveform"]), "waveform");
}

/// V6: the never-stop run goes to `max_tokens`.
#[test]
fn full_budget_generation_runs_to_max_tokens() {
    let traces = read_json(&testdata("length/traces.json"));
    let mut m = Model::open(&testdata("length")).expect("voxcpm model loads");
    let params = params_of(&traces);
    let text_ids = text_ids_of(&traces);
    let mut noise = RecordedNoise::new(traces["noise_draws"].as_array().expect("draws"));
    let got = m
        .generate(&text_ids, &params, &mut noise, None)
        .expect("generate runs");
    assert_eq!(got.steps, params.max_tokens);
    assert!(!got.stopped);
    assert_feedback_relative(&got.latents, &concat_latents(&traces), "latents");
    assert_feedback_relative(&got.samples, &f32_vec(&traces["waveform"]), "waveform");
}

/// V7: the voice-clone run matches end to end, including the prompt
/// patch condition.
#[test]
fn voice_clone_generation_matches_the_reference_end_to_end() {
    let traces = read_json(&testdata("voice_clone/traces.json"));
    let mut m = Model::open(&testdata("voice_clone")).expect("voxcpm model loads");
    let params = params_of(&traces);
    let text_ids = text_ids_of(&traces);
    let voice = VoicePrompt {
        ref_audio: f32_vec(&traces["voice_clone"]["ref_audio"]),
    };
    let mut noise = RecordedNoise::new(traces["noise_draws"].as_array().expect("draws"));
    let got = m
        .generate(&text_ids, &params, &mut noise, Some(&voice))
        .expect("generate runs");
    let steps = traces["steps"].as_array().expect("steps");
    assert_eq!(got.steps, steps.len());
    assert_feedback_relative(&got.samples, &f32_vec(&traces["waveform"]), "waveform");
    assert_feedback_relative(&got.latents, &concat_latents(&traces), "latents");
}

/// V8: the LongRoPE case crosses the factor-table threshold: the
/// 21-position prefill and every decode step past position 15 use the
/// long factors.
#[test]
fn longrope_factor_flip_matches_the_reference() {
    let traces = read_json(&testdata("longrope/traces.json"));
    let mut m = Model::open(&testdata("longrope")).expect("voxcpm model loads");
    let params = params_of(&traces);
    let text_ids = text_ids_of(&traces);
    assert!(text_ids.len() + 1 > m.config.lm_config.original_max_position_embeddings);
    let mut noise = RecordedNoise::new(traces["noise_draws"].as_array().expect("draws"));
    let got = m
        .generate(&text_ids, &params, &mut noise, None)
        .expect("generate runs");
    assert_eq!(got.steps, params.max_tokens);
    assert_feedback_relative(&got.samples, &f32_vec(&traces["waveform"]), "waveform");
    assert_feedback_relative(&got.latents, &concat_latents(&traces), "latents");
}

/// V9: the crate-local threefry RNG reproduces the recorded MLX normal
/// stream from the same seed, so standalone runs need no replay.
#[test]
fn seeded_threefry_reproduces_the_recorded_normal_draws() {
    let traces = read_json(&testdata("generate/traces.json"));
    let draws = traces["noise_draws"].as_array().expect("draws");
    let mut noise = SeededFlowNoise::new(101);
    for draw in draws {
        let values = f32_vec(&draw["values"]);
        let got = noise.normal(values.len());
        let diff = max_abs_diff(&got, &values);
        assert!(diff < 1e-6, "seeded draw diff {diff:e}");
    }
}

/// V10: zero budget and zero solver steps are refused, not silently
/// empty (the reference would fail on an empty concatenate).
#[test]
fn degenerate_generation_parameters_are_refused() {
    let traces = read_json(&testdata("generate/traces.json"));
    let mut m = Model::open(&testdata("generate")).expect("voxcpm model loads");
    let text_ids = text_ids_of(&traces);
    let mut noise = RecordedNoise::new(&[]);
    let params = GenerateParams {
        max_tokens: 0,
        ..params_of(&traces)
    };
    let error = m
        .generate(&text_ids, &params, &mut noise, None)
        .unwrap_err();
    assert!(matches!(error, SpeechError::Input { .. }), "{error}");
    let params = GenerateParams {
        inference_timesteps: 0,
        ..params_of(&traces)
    };
    let error = m
        .generate(&text_ids, &params, &mut noise, None)
        .unwrap_err();
    assert!(matches!(error, SpeechError::Input { .. }), "{error}");
}
