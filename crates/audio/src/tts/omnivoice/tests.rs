//! Parity tests for the OmniVoice family against the fixtures recorded
//! by `tools/gen_omnivoice_fixtures.py` (tiny seeded bidirectional
//! qwen3 backbone, 8 audio codebooks of vocab 17 with the mask at 16,
//! the greedy CFG token draw at temperature 0, recorded position-Gumbel
//! uniforms, and the tiny HiggsAudioV2 codec decode).

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::Model;
use crate::codec::higgs_audio::HiggsAudioTokenizer;

fn testdata(segments: &[&str]) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("testdata/tts_omnivoice");
    for segment in segments {
        path.push(segment);
    }
    path
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
        .expect("ids")
        .iter()
        .map(|v| v.as_i64().expect("id") as i32)
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Frozen gate over the tiny-codec waveform; measured 5.6e-9 (the
/// decode is the shared codec port, so the pin travels through it).
const WAVEFORM_TOLERANCE: f32 = 1e-5;

#[test]
fn config_defaults_match_the_reference() {
    let config = super::OmniVoiceConfig::default();
    assert_eq!(config.model_type, "omnivoice");
    assert_eq!(config.audio_vocab_size, 1025);
    assert_eq!(config.audio_mask_id, 1024);
    assert_eq!(config.num_audio_codebook, 8);
    assert_eq!(config.sample_rate, 24000);
}

#[test]
fn combine_text_follows_the_reference_rules() {
    // Strip, join, newline removal, space-run collapse.
    assert_eq!(
        super::combine_text("  hello   world  ", Some(" ref\ttext ")),
        "ref text hello world"
    );
    assert_eq!(super::combine_text("a\r\nb", None), "ab");
    // Whitespace touching a CJK character is removed on both sides.
    assert_eq!(super::combine_text("中文 test", None), "中文test");
    assert_eq!(super::combine_text("test 中文", None), "test中文");
    assert_eq!(super::combine_text("a 中文 b", None), "a中文b");
    assert_eq!(
        super::combine_text("日本語と日本語", None),
        "日本語と日本語"
    );
}

#[test]
fn style_and_text_strings_match_the_reference_rules() {
    assert_eq!(
        super::build_style_text("en", "None", false),
        "<|lang_start|>en<|lang_end|><|instruct_start|>None<|instruct_end|>"
    );
    assert_eq!(
        super::build_style_text("zh", "excited", true),
        "<|denoise|><|lang_start|>zh<|lang_end|><|instruct_start|>excited<|instruct_end|>"
    );
    assert_eq!(
        super::build_wrapped_text(" hi ", None),
        "<|text_start|>hi<|text_end|>"
    );
}

#[test]
fn generation_matches_the_reference() {
    let traces = read_json(&testdata(&["traces.json"]));
    let codebooks = 8usize;
    assert_eq!(
        traces["cases"].as_array().unwrap()[0]["cond_input_ids"]
            .as_array()
            .unwrap()
            .len()
            % codebooks,
        0
    );

    for case in traces["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let style_ids = i32_vec(&case["style_ids"]);
        let text_ids = i32_vec(&case["text_ids"]);
        let target_len = case["target_len"].as_u64().unwrap() as usize;
        let n_text = style_ids.len() + text_ids.len();
        let ref_tokens: Option<Vec<Vec<i32>>> = if case["denoise"].as_bool().unwrap() {
            // The voice-clone case carries a 4-frame reference block
            // between the text and the target.
            let rows = case["cond_input_ids"].as_array().unwrap()
                [(n_text) * codebooks..(n_text + 4) * codebooks]
                .chunks(codebooks)
                .map(|row| row.iter().map(|v| v.as_i64().unwrap() as i32).collect())
                .collect::<Vec<Vec<i32>>>();
            assert_eq!(rows.len(), 4);
            Some(rows)
        } else {
            None
        };

        let model = Model::open(&testdata(&["generate"])).expect("model loads");
        // The assembled inputs.
        let (input_ids, audio_mask) = model
            .prepare_inference_inputs(&style_ids, &text_ids, target_len, ref_tokens.as_deref())
            .expect("inputs assemble");
        let want_ids = i32_vec(&case["cond_input_ids"]);
        assert_eq!(input_ids, want_ids, "{name} input ids");
        let want_mask = case["cond_audio_mask"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() == 1)
            .collect::<Vec<bool>>();
        assert_eq!(audio_mask, want_mask, "{name} audio mask");

        // The iterative unmask over the recorded position-Gumbel draws.
        let draws: Vec<Vec<f32>> = case["gumbel_draws"]
            .as_array()
            .expect("draws")
            .iter()
            .map(f32_vec)
            .collect();
        let mut runner = model;
        let tokens = runner
            .iterative_unmask(
                &input_ids,
                &audio_mask,
                target_len,
                traces["num_steps"].as_u64().unwrap() as usize,
                traces["guidance_scale"].as_f64().unwrap() as f32,
                traces["class_temperature"].as_f64().unwrap() as f32,
                traces["position_temperature"].as_f64().unwrap() as f32,
                traces["layer_penalty_factor"].as_f64().unwrap() as f32,
                traces["t_shift"].as_f64().unwrap() as f32,
                Some(&draws),
                0,
            )
            .expect("unmask");
        let want_tokens = i32_vec(&case["tokens"]);
        assert_eq!(tokens, want_tokens, "{name} tokens");

        // The codec decode: tokens [T, C] -> per-book rows -> samples.
        let codec = HiggsAudioTokenizer::open(&testdata(&["codec"])).expect("codec loads");
        let frames = tokens.len() / codebooks;
        let mut codes: Vec<Vec<i32>> = Vec::with_capacity(codebooks);
        for _ in 0..codebooks {
            codes.push(Vec::with_capacity(frames));
        }
        for frame in tokens.chunks(codebooks) {
            for (c, slot) in codes.iter_mut().enumerate() {
                slot.push(frame[c]);
            }
        }
        let audio = codec.decode_tokens(&codes).expect("decode");
        let want = f32_vec(&case["waveform"]);
        assert_eq!(audio.len(), want.len(), "{name} waveform length");
        let diff = max_abs_diff(&audio, &want);
        assert!(diff <= WAVEFORM_TOLERANCE, "{name} waveform diff {diff}");
    }
}

// The guards below reject bad inputs before any backbone pass, so they
// run on the checked-in tiny fixture without a long forward.

#[test]
fn reference_frames_must_have_one_code_per_codebook() {
    let model = Model::open(&testdata(&["generate"])).expect("model loads");
    let codebooks = model.config.num_audio_codebook;
    let style = [1i32];
    let text = [2i32];

    // A 7-code frame used to be zero-filled into the last codebook column.
    let short = vec![vec![3i32; codebooks - 1]];
    assert!(model
        .prepare_inference_inputs(&style, &text, 2, Some(&short))
        .is_err());
    let long = vec![vec![3i32; codebooks + 1]];
    assert!(model
        .prepare_inference_inputs(&style, &text, 2, Some(&long))
        .is_err());
    let exact = vec![vec![3i32; codebooks]];
    assert!(model
        .prepare_inference_inputs(&style, &text, 2, Some(&exact))
        .is_ok());
}

#[test]
fn zero_steps_and_oversized_targets_are_rejected() {
    let mut model = Model::open(&testdata(&["generate"])).expect("model loads");
    let (ids, mask) = model
        .prepare_inference_inputs(&[1i32], &[2i32], 2, None)
        .expect("inputs assemble");

    // Zero steps used to return the all-mask target as zeros.
    let zero_steps = model.iterative_unmask(&ids, &mask, 2, 0, 0.0, 0.0, 0.0, 0.0, 1.0, None, 0);
    assert!(zero_steps.is_err());

    // A target longer than the sequence used to underflow `c_len - target_len`.
    let oversized = model.iterative_unmask(
        &ids,
        &mask,
        mask.len() + 1,
        4,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        None,
        0,
    );
    assert!(oversized.is_err());
}
