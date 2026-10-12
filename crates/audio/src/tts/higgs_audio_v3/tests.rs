//! Parity and regression tests for the Higgs Audio v3 TTS family.
//!
//! Pinned fixtures reside in `testdata/tts_higgs_audio_v3/`, generated
//! by `tools/gen_higgs_audio_v3_fixtures.py` against mlx-audio 0.5.7
//! commit `e1b19b9054bf163f5d812221a54fcc346f1890e9` (mlx 0.32.3). The
//! generator runs the true reference modules (the v3 `Model` over a
//! seeded tiny qwen3 backbone, the true prompt builder and `step`
//! sampler, a tiny seeded Higgs codec with the real decode stacks), so
//! the traces carry the reference delay-pattern math, prompt
//! embeddings, forward audio logits, greedy frame loops, and codec
//! waveforms.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::{
    apply_delay_pattern, apply_masked_filter, greedy_sample_audio, reverse_delay_pattern,
    GenerateParams, Model, ModelConfig, PromptBuilder, PromptParts, ReferenceCodes, SamplerState,
    StopReason, AUDIO_PLACEHOLDER_ID,
};
use crate::codec::higgs_audio::HiggsAudioTokenizer;
use crate::lm;

fn testdata(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/tts_higgs_audio_v3")
        .join(name)
}

fn read_json(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).expect("fixture json");
    serde_json::from_str(&text).expect("fixture json")
}

fn i32_vec(value: &Value) -> Vec<i32> {
    value
        .as_array()
        .expect("ids")
        .iter()
        .map(|v| v.as_i64().expect("id") as i32)
        .collect()
}

fn i32_rows(value: &Value) -> Vec<Vec<i32>> {
    value
        .as_array()
        .expect("rows")
        .iter()
        .map(i32_vec)
        .collect()
}

fn f32_vec(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .expect("values")
        .iter()
        .map(|v| v.as_f64().expect("value") as f32)
        .collect()
}

/// Filtered logit entries use null for the reference's negative
/// infinities (strict JSON cannot carry them).
fn f32_vec_masked(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .expect("values")
        .iter()
        .map(|v| {
            if v.is_null() {
                f32::NEG_INFINITY
            } else {
                v.as_f64().expect("value") as f32
            }
        })
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

/// Measured maximum differences against the fixture traces when the
/// gates were frozen; assertions use these values, not re-measured ones.
const LOGIT_TOLERANCE: f32 = 2e-5;
const EMBED_TOLERANCE: f32 = 1e-6;
const WAVEFORM_TOLERANCE: f32 = 1e-6;

fn generate_model() -> Model {
    Model::open(&testdata("generate")).expect("family model loads")
}

fn generate_codec() -> HiggsAudioTokenizer {
    HiggsAudioTokenizer::open(&testdata("codec")).expect("tiny codec loads")
}

fn fixture_builder() -> PromptBuilder {
    let traces = read_json(&testdata("generate/traces.json"));
    let prompt = &traces["prompt"];
    PromptBuilder {
        tts_id: prompt["tts_id"].as_i64().unwrap() as i32,
        ref_audio_id: prompt["ref_audio_id"].as_i64().unwrap() as i32,
        text_id: prompt["text_id"].as_i64().unwrap() as i32,
        audio_id: prompt["audio_id"].as_i64().unwrap() as i32,
        ref_text_id: prompt["ref_text_id"].as_i64().map(|v| v as i32),
    }
}

#[test]
fn delay_pattern_math_matches_the_reference() {
    let traces = read_json(&testdata("delay/traces.json"));
    let applied = apply_delay_pattern(
        &i32_rows(&traces["apply_input"]),
        traces["boc_id"].as_i64().unwrap() as i32,
        traces["eoc_id"].as_i64().unwrap() as i32,
    )
    .expect("applies");
    assert_eq!(applied, i32_rows(&traces["apply_output"]));
    let reversed = reverse_delay_pattern(&i32_rows(&traces["reverse_input"])).expect("reverses");
    assert_eq!(reversed, i32_rows(&traces["reverse_output"]));
}

#[test]
fn reverse_refuses_too_short_timelines() {
    let error = reverse_delay_pattern(&[vec![1, 2, 3]]).unwrap_err();
    assert!(error.to_string().contains("need L >= K"), "{error}");
}

#[test]
fn prompt_builder_matches_the_reference_layout() {
    let traces = read_json(&testdata("generate/traces.json"));
    let prompt = &traces["prompt"];
    let builder = fixture_builder();
    let references = vec![ReferenceCodes {
        text_ids: prompt
            .get("ref_text_ids")
            .and_then(Value::as_array)
            .map(|entries| {
                entries
                    .iter()
                    .map(|v| v.as_i64().expect("id") as i32)
                    .collect()
            }),
        delayed_codes: i32_rows(&prompt["ref_delayed"]),
    }];
    let parts = builder
        .build_prompt(&i32_vec(&prompt["text_ids"]), &references)
        .expect("builds");
    assert_eq!(parts.token_ids, i32_vec(&traces["prompt_token_ids"]));
    assert_eq!(
        parts.audio_segments.first().map(|(start, _)| *start),
        Some(prompt["segment_start"].as_u64().unwrap() as usize)
    );
    assert_eq!(
        parts.audio_segments.first().map(|(_, codes)| codes.len()),
        Some(i32_rows(&prompt["ref_delayed"]).len())
    );
    // Placeholders ride the token ids only inside the segment window.
    let start = parts.audio_segments[0].0;
    let width = parts.audio_segments[0].1.len();
    assert!(parts.token_ids[..start]
        .iter()
        .all(|&t| t != AUDIO_PLACEHOLDER_ID));
    assert_eq!(
        parts.token_ids[start..start + width],
        vec![AUDIO_PLACEHOLDER_ID; width]
    );
}

#[test]
fn prompt_embeddings_match_the_reference_rows() {
    let traces = read_json(&testdata("generate/traces.json"));
    let model = generate_model();
    let prompt = &traces["prompt"];
    let builder = fixture_builder();
    let references = vec![ReferenceCodes {
        text_ids: prompt
            .get("ref_text_ids")
            .and_then(Value::as_array)
            .map(|entries| {
                entries
                    .iter()
                    .map(|v| v.as_i64().expect("id") as i32)
                    .collect()
            }),
        delayed_codes: i32_rows(&prompt["ref_delayed"]),
    }];
    let parts = builder
        .build_prompt(&i32_vec(&prompt["text_ids"]), &references)
        .expect("builds");
    let embeds = model.build_prompt_embeddings(&parts).expect("embeds");
    let diff = max_abs_diff(&embeds, &f32_vec(&traces["prompt_embeds"]));
    assert!(diff <= EMBED_TOLERANCE, "prompt embeds diff {diff}");
    // The single-frame embed matches the row-sum view of the fused table.
    let frame = i32_vec(&traces["probe_frame"]);
    let one = model.embed_audio_codes(&frame).expect("frame embed");
    let second = i32_vec(&traces["probe_frame2"]);
    let rows = model
        .embed_audio_codes_rows(&[frame.clone(), second.clone()])
        .expect("rows embed");
    assert_eq!(rows.len(), one.len() * 2);
    let diff = max_abs_diff(&rows[..one.len()], &one);
    assert!(diff < f32::EPSILON, "row embed diff {diff}");
    let direct = model.embed_audio_codes(&second).expect("second embed");
    let diff = max_abs_diff(&rows[one.len()..], &direct);
    assert!(diff < f32::EPSILON, "second row embed diff {diff}");
}

#[test]
fn forward_audio_logits_match_the_reference() {
    let traces = read_json(&testdata("generate/traces.json"));
    let mut model = generate_model();
    let embeds = f32_vec(&traces["prompt_embeds"]);
    let mut caches = lm::make_prompt_cache(model.config.text_config.num_hidden_layers);
    let hidden = model
        .backbone
        .forward_hidden(lm::engine::Input::Embeddings(&embeds), Some(&mut caches))
        .expect("forward");
    let hidden_size = model.config.text_config.hidden_size;
    let last = &hidden[hidden.len() - hidden_size..];
    let diff = max_abs_diff(last, &f32_vec(&traces["forward_last_hidden"]));
    assert!(diff <= LOGIT_TOLERANCE, "last hidden diff {diff}");
    let logits = model.audio_logits(last);
    let diff = max_abs_diff(&logits, &f32_vec(&traces["forward_audio_logits"]));
    assert!(diff <= LOGIT_TOLERANCE, "audio logits diff {diff}");
    assert_eq!(
        logits.len(),
        model.config.audio_num_codebooks * model.config.codebook_stride()
    );
}

fn run_scenario(scenario: &Value) {
    let mut model = generate_model();
    let codec = generate_codec();
    let embeds = f32_vec(&read_json(&testdata("generate/traces.json"))["prompt_embeds"]);
    let params = GenerateParams {
        max_new_frames: scenario["max_new_frames"].as_u64().unwrap() as usize,
        temperature: 0.0,
        top_p: None,
        top_k: None,
        fade_in_ms: 30.0,
        fade_out_ms: 15.0,
    };
    let got = model
        .generate(&codec, &embeds, &params, 0)
        .unwrap_or_else(|error| panic!("scenario {} generate failed: {error}", scenario["name"]));
    assert_eq!(
        got.frames,
        i32_rows(&scenario["frames"]),
        "scenario {} raw frames differ",
        scenario["name"]
    );
    assert_eq!(
        got.codes,
        i32_rows(&scenario["codes"]),
        "scenario {} aligned codes differ",
        scenario["name"]
    );
    let want_stop = scenario["stop_reason"].as_str().expect("stop reason");
    if want_stop == "eoc-ramp" {
        assert_eq!(got.stopped_by, StopReason::EocRamp);
    } else {
        assert_eq!(got.stopped_by, StopReason::MaxSteps, "want {want_stop}");
    }
    assert_eq!(
        got.num_frames_raw,
        scenario["num_frames_raw"].as_u64().unwrap() as usize
    );
    assert_eq!(
        got.num_frames_aligned,
        scenario["num_frames_aligned"].as_u64().unwrap() as usize
    );
    let diff = max_abs_diff(&got.samples, &f32_vec(&scenario["waveform_faded"]));
    assert!(diff <= WAVEFORM_TOLERANCE, "faded waveform diff {diff}");
}

#[test]
fn greedy_generation_stops_at_the_eoc_ramp_with_reference_frames() {
    let traces = read_json(&testdata("generate/traces.json"));
    run_scenario(&traces["scenarios"][0]);
}

#[test]
fn greedy_generation_stops_at_max_steps_with_reference_frames() {
    let traces = read_json(&testdata("generate/traces.json"));
    run_scenario(&traces["scenarios"][1]);
}

#[test]
fn sampler_state_ramp_and_countdown_follow_the_reference_step() {
    // A synthetic run: force the logits so codebook 0 hits EOC on the
    // first post-ramp step, then watch the countdown drain.
    let k = 4;
    let v = 6;
    let boc = 4;
    let eoc = 5;
    let mut state = SamplerState::new(k);
    let params = GenerateParams {
        temperature: 0.0,
        ..GenerateParams::default()
    };
    // Ramp-in: rows 1..4 tail-forced to BOC, one codebook per step.
    for step_index in 0..k {
        let mut logits = vec![0.0f32; k * v];
        for codebook in 0..k {
            logits[codebook * v + 1] = 3.0;
        }
        let codes = super::step(
            &logits,
            &mut state,
            &params,
            boc,
            eoc,
            crate::music::minimax_music3::rng::Key::new(1),
        )
        .expect("steps");
        let expected_tail = if step_index + 1 < k {
            step_index + 1
        } else {
            0
        };
        for (codebook, &token) in codes.iter().enumerate() {
            if codebook >= expected_tail && expected_tail > 0 {
                assert_eq!(token, boc, "ramp step {step_index} codebook {codebook}");
            } else {
                assert_eq!(token, 1, "ramp step {step_index} codebook {codebook}");
            }
        }
    }
    // Force EOC on codebook 0: countdown becomes K - 2.
    let mut logits = vec![0.0f32; k * v];
    for codebook in 0..k {
        let winner = if codebook == 0 { eoc } else { 2 };
        logits[codebook * v + winner as usize] = 3.0;
    }
    let codes = super::step(
        &logits,
        &mut state,
        &params,
        boc,
        eoc,
        crate::music::minimax_music3::rng::Key::new(2),
    )
    .expect("steps");
    assert_eq!(codes[0], eoc);
    assert_eq!(state.eoc_countdown, Some(k as i32 - 2));
    assert!(!state.generation_done);
    // Two drain steps (countdown K-2 -> 1 -> 0); the last sets
    // generation_done and keeps drawing the EOC row.
    for expected in [Some(1), Some(0)] {
        let codes = super::step(
            &logits,
            &mut state,
            &params,
            boc,
            eoc,
            crate::music::minimax_music3::rng::Key::new(3),
        )
        .expect("steps");
        assert_eq!(state.eoc_countdown, expected);
        assert_eq!(codes[0], eoc);
    }
    // After done, step returns the STOP_CODE row without drawing.
    assert_eq!(
        super::step(
            &logits,
            &mut state,
            &params,
            boc,
            eoc,
            crate::music::minimax_music3::rng::Key::new(4)
        )
        .expect("steps"),
        vec![-1; k]
    );
}

#[test]
fn sampling_filters_match_the_reference_on_fixed_logits() {
    let traces = read_json(&testdata("sampling/traces.json"));
    for case in traces["cases"].as_array().expect("cases") {
        let logits = f32_vec(&case["logits"]);
        let codebooks = case["codebooks"].as_u64().unwrap() as usize;
        let width = logits.len() / codebooks;
        let greedy = greedy_sample_audio(&logits, codebooks, width);
        assert_eq!(greedy, i32_vec(&case["greedy"]), "case {}", case["name"]);
        if let Some(masked) = case.get("top_k_masked") {
            let top_k = case["top_k"].as_u64().unwrap() as usize;
            let mut scaled = logits.clone();
            for codebook in 0..codebooks {
                let row = &mut scaled[codebook * width..(codebook + 1) * width];
                apply_masked_filter(row, &|logprobs| {
                    crate::tts::sampling::apply_top_k(logprobs, top_k)
                })
                .expect("filters");
            }
            let diff = max_abs_diff(&scaled, &f32_vec_masked(masked));
            assert!(diff <= 1e-6, "case {} top_k mask diff {diff}", case["name"]);
        }
        if let Some(masked) = case.get("top_p_masked") {
            let top_p = case["top_p"].as_f64().unwrap() as f32;
            let mut scaled = logits.clone();
            for codebook in 0..codebooks {
                let row = &mut scaled[codebook * width..(codebook + 1) * width];
                apply_masked_filter(row, &|logprobs| {
                    crate::tts::sampling::apply_top_p(logprobs, top_p);
                    Ok(())
                })
                .expect("filters");
            }
            let diff = max_abs_diff(&scaled, &f32_vec_masked(masked));
            assert!(diff <= 1e-6, "case {} top_p mask diff {diff}", case["name"]);
        }
        // A sampled draw always lands on a surviving class and the
        // greedy shortcuts hold.
        let params = GenerateParams {
            temperature: 0.8,
            top_p: Some(0.5),
            top_k: Some(4),
            ..GenerateParams::default()
        };
        let sampled = super::sample_independent(
            &logits,
            codebooks,
            width,
            &params,
            crate::music::minimax_music3::rng::Key::new(9),
        )
        .expect("samples");
        assert_eq!(sampled.len(), codebooks);
        for token in &sampled {
            assert!(
                *token >= 0 && (*token as usize) < width,
                "sample {token} out of range"
            );
        }
        let greedy_params = GenerateParams {
            temperature: 0.0,
            ..GenerateParams::default()
        };
        assert_eq!(
            super::sample_independent(
                &logits,
                codebooks,
                width,
                &greedy_params,
                crate::music::minimax_music3::rng::Key::new(10),
            )
            .expect("greedy"),
            greedy
        );
        let topk_one = GenerateParams {
            temperature: 0.8,
            top_k: Some(1),
            ..GenerateParams::default()
        };
        assert_eq!(
            super::sample_independent(
                &logits,
                codebooks,
                width,
                &topk_one,
                crate::music::minimax_music3::rng::Key::new(11),
            )
            .expect("greedy"),
            greedy
        );
    }
}

#[test]
fn family_config_defaults_and_encoder_fallbacks_match_the_reference() {
    let defaults = ModelConfig::from_json(&serde_json::json!({})).unwrap();
    assert_eq!(defaults.audio_num_codebooks, 8);
    assert_eq!(defaults.audio_codebook_size, 1026);
    assert_eq!(defaults.audio_boc_token_id, 1024);
    assert_eq!(defaults.audio_eoc_token_id, 1025);
    assert_eq!(defaults.audio_token_id, -100);
    assert!(defaults.use_delay_pattern);
    assert_eq!(defaults.sample_rate, 24000);
    assert_eq!(defaults.text_config.hidden_size, 2560);
    assert_eq!(defaults.text_config.num_hidden_layers, 36);
    assert_eq!(defaults.text_config.head_dim, 128);
    assert_eq!(defaults.text_config.rope_theta, 1_000_000.0);
    assert_eq!(defaults.text_config.rms_norm_eps, 1e-6);
    assert_eq!(defaults.text_config.vocab_size, 151936);
    assert!(defaults.text_config.rope_scaling.is_none());
    assert_eq!(defaults.codebook_stride(), 1026);

    // The audio_encoder_config fallbacks feed the missing keys.
    let from_encoder = ModelConfig::from_json(&serde_json::json!({
        "audio_encoder_config": {
            "vocab_size": 10,
            "num_codebooks": 4,
            "use_delay_pattern": false
        }
    }))
    .unwrap();
    assert_eq!(from_encoder.audio_codebook_size, 10);
    assert_eq!(from_encoder.audio_num_codebooks, 4);
    assert_eq!(from_encoder.audio_boc_token_id, 8);
    assert_eq!(from_encoder.audio_eoc_token_id, 9);
    assert!(!from_encoder.use_delay_pattern);

    // The rope_parameters alias feeds rope_theta.
    let aliased = ModelConfig::from_json(&serde_json::json!({
        "text_config": {"rope_parameters": {"rope_theta": 50000.0}}
    }))
    .unwrap();
    assert_eq!(aliased.text_config.rope_theta, 50000.0);

    // The explicit top-level keys win over the encoder block.
    let explicit = ModelConfig::from_json(&serde_json::json!({
        "audio_codebook_size": 12,
        "audio_boc_token_id": 3,
        "audio_eoc_token_id": 4,
        "audio_encoder_config": {"vocab_size": 10}
    }))
    .unwrap();
    assert_eq!(explicit.audio_codebook_size, 12);
    assert_eq!(explicit.audio_boc_token_id, 3);
    assert_eq!(explicit.audio_eoc_token_id, 4);

    let refused = ModelConfig::from_json(&serde_json::json!({
        "audio_num_codebooks": 0
    }))
    .unwrap_err();
    assert!(refused.to_string().contains("positive"), "{refused}");
}

#[test]
fn embed_refuses_a_code_past_its_codebook_rows() {
    // A code equal to the stride would read the next codebook's rows.
    let model = generate_model();
    let k = model.config.audio_num_codebooks;
    let stride = model.config.codebook_stride() as i32;
    let mut frame = vec![0; k];
    frame[0] = stride;
    let err = model.embed_audio_codes(&frame).unwrap_err();
    assert!(err.to_string().contains("codebook 0"), "{err}");
    model
        .embed_audio_codes(&vec![0; k])
        .expect("zero codes are in range");
}

#[test]
fn empty_transcript_adds_no_ref_text_marker() {
    // The reference tests `if reference.text`: an empty transcript must not
    // emit a <|ref_text|> marker with no text after it.
    const REF_TEXT: i32 = 999;
    let builder = PromptBuilder {
        tts_id: 1,
        ref_audio_id: 2,
        text_id: 3,
        audio_id: 4,
        ref_text_id: Some(REF_TEXT),
    };
    let reference = |text_ids: Option<Vec<i32>>| ReferenceCodes {
        text_ids,
        delayed_codes: vec![vec![0]],
    };
    let empty = builder
        .build_prompt(&[5], &[reference(Some(Vec::new()))])
        .expect("builds");
    assert!(
        !empty.token_ids.contains(&REF_TEXT),
        "{:?}",
        empty.token_ids
    );
    let spoken = builder
        .build_prompt(&[5], &[reference(Some(vec![7]))])
        .expect("builds");
    assert!(spoken.token_ids.contains(&REF_TEXT));
}

#[test]
fn prompt_embeddings_refuse_out_of_range_segments() {
    // Pub PromptParts fields are caller-controlled: a segment past the end,
    // or one that starts before the previous segment ends, must be an error.
    let model = generate_model();
    let k = model.config.audio_num_codebooks;
    let row = vec![0; k];
    let past_end = PromptParts {
        token_ids: vec![1, 2, 3],
        audio_segments: vec![(2, vec![row.clone(); 3])],
    };
    assert!(model.build_prompt_embeddings(&past_end).is_err());
    let overlapping = PromptParts {
        token_ids: vec![1, 2, 3, 4],
        audio_segments: vec![(2, vec![row.clone()]), (1, vec![row.clone()])],
    };
    assert!(model.build_prompt_embeddings(&overlapping).is_err());
}

#[test]
fn generate_refuses_an_empty_prompt() {
    let mut model = generate_model();
    let codec = generate_codec();
    let error = model
        .generate(&codec, &[], &GenerateParams::default(), 0)
        .unwrap_err();
    assert!(error.to_string().contains("empty forward input"), "{error}");
}
