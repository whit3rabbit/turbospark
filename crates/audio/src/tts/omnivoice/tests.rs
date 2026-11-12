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
                None,
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
    let zero_steps =
        model.iterative_unmask(&ids, &mask, 2, 0, 0.0, 0.0, 0.0, 0.0, 1.0, None, 0, None);
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
        None,
    );
    assert!(oversized.is_err());
}

// The released checkpoints keep the raw k2-fsa layout: `llm.` backbone
// names, the fused audio tables, and the rowwise int8 scheme recorded in
// `mlx_manifest.json`. These tests repackage the tiny fixture into that
// layout and require the loaded model to be numerically identical to the
// converted-layout model over the same dequantized weights.

/// Minimal safetensors writer: F32/I8/F16 tensors keyed by name.
fn write_safetensors(
    path: &Path,
    tensors: &[(impl AsRef<str>, impl AsRef<str>, Vec<usize>, Vec<u8>)],
) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let start = data.len();
        data.extend_from_slice(bytes);
        header.insert(
            name.as_ref().to_string(),
            serde_json::json!({
                "dtype": dtype.as_ref(),
                "shape": shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let header_json = serde_json::to_string(&serde_json::Value::Object(header)).unwrap();
    let mut out = Vec::new();
    out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    out.extend_from_slice(header_json.as_bytes());
    out.extend_from_slice(&data);
    std::fs::write(path, out).unwrap();
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Rowwise-symmetric int8 quantization mirroring the conversion script:
/// per-group `scale = max(|group|) / 127` (floored at 1e-8) and
/// `q = round(v / scale)` clamped to `[-127, 127]`. Returns the packed
/// values and the scales.
fn quantize_rowwise(values: &[f32], group_size: usize) -> (Vec<i8>, Vec<f32>) {
    let groups = values.len().div_ceil(group_size);
    let mut packed = Vec::with_capacity(values.len());
    let mut scales = Vec::with_capacity(groups);
    for group in 0..groups {
        let slice = &values[group * group_size..(group + 1) * group_size];
        let scale = (slice.iter().fold(0.0f32, |max, v| max.max(v.abs())) / 127.0).max(1e-8);
        scales.push(scale);
        for value in slice {
            let q = (value / scale).round().clamp(-127.0, 127.0);
            packed.push(q as i8);
        }
    }
    (packed, scales)
}

/// Repackages the tiny fixture into the raw release layout
/// (`llm.` names, fused audio tables, rowwise int8 + manifest) and
/// requires the loaded model to match the converted-layout model loaded
/// from the same dequantized weights, forward for forward.
#[test]
fn raw_release_layout_intake_matches_the_converted_layout() {
    use turbospark_model_io::safetensors::SafetensorsFile;

    let fixture = SafetensorsFile::open(&testdata(&["generate", "model.safetensors"]))
        .expect("fixture model opens");
    let config = std::fs::read_to_string(testdata(&["generate", "config.json"])).expect("config");

    let dir = std::env::temp_dir().join(format!("omnivoice-intake-{}", std::process::id()));
    let raw_dir = dir.join("raw");
    let plain_dir = dir.join("plain");
    std::fs::create_dir_all(&raw_dir).unwrap();
    std::fs::create_dir_all(&plain_dir).unwrap();
    std::fs::write(raw_dir.join("config.json"), &config).unwrap();
    std::fs::write(plain_dir.join("config.json"), &config).unwrap();

    const GROUP_SIZE: usize = 8;
    let mut raw_tensors: Vec<(String, String, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut manifest_entries = serde_json::Map::new();
    let mut plain_tensors: Vec<(String, String, Vec<usize>, Vec<u8>)> = Vec::new();

    // Fused audio tables: concatenate the per-codebook fixture tables,
    // quantize the fusion, and split the dequantized rows back out for
    // the converted-layout comparison model.
    for role in ["audio_embeddings", "audio_heads"] {
        let mut values = Vec::new();
        let mut shape = Vec::new();
        for codebook in 0..8 {
            let name = format!("{role}.{codebook}.weight");
            let descriptor = fixture.descriptor(&name).expect("fixture table");
            shape = descriptor.shape.clone();
            values.extend(fixture.load_as_f32(&name).expect("fixture table"));
        }
        let (rows, hidden) = (shape[0] * 8, shape[1]);
        let groups = hidden / GROUP_SIZE;
        let (packed, scales) = quantize_rowwise(&values, GROUP_SIZE);
        let fused_name = format!("{role}.weight");
        raw_tensors.push((
            fused_name.clone(),
            "I8".into(),
            vec![rows, groups, GROUP_SIZE],
            packed.iter().map(|&b| b as u8).collect(),
        ));
        raw_tensors.push((
            format!("{fused_name}.scales"),
            "F32".into(),
            vec![rows, groups],
            f32_bytes(&scales),
        ));
        manifest_entries.insert(
            fused_name,
            serde_json::json!({"bits": 8, "group_size": GROUP_SIZE, "shape": [rows, hidden]}),
        );
        // The dequantized fused table, split back per codebook.
        let mut dequantized = Vec::with_capacity(values.len());
        for (group, scale) in scales.iter().enumerate() {
            for value in &packed[group * GROUP_SIZE..(group + 1) * GROUP_SIZE] {
                dequantized.push(*value as f32 * scale);
            }
        }
        let table_len = shape[0] * hidden;
        for codebook in 0..8 {
            let rows = &dequantized[codebook * table_len..(codebook + 1) * table_len];
            plain_tensors.push((
                format!("{role}.{codebook}.weight"),
                "F32".into(),
                shape.clone(),
                f32_bytes(rows),
            ));
        }
    }

    for name in fixture.tensor_names() {
        let Some(base_name) = name.strip_suffix(".weight") else {
            continue;
        };
        // The fused audio tables were handled above; the fixture's
        // per-codebook tables must not be copied again.
        if base_name.starts_with("audio_embeddings.") || base_name.starts_with("audio_heads.") {
            continue;
        }
        let descriptor = fixture.descriptor(name).expect("fixture tensor");
        let values = fixture.load_as_f32(name).expect("fixture tensor");
        let raw_name = name.replacen("backbone.", "llm.", 1);
        let plain_values = if descriptor.shape.len() == 2 {
            let (packed, scales) = quantize_rowwise(&values, GROUP_SIZE);
            let rows = descriptor.shape[0];
            let groups = descriptor.shape[1] / GROUP_SIZE;
            raw_tensors.push((
                raw_name.clone(),
                "I8".into(),
                vec![rows, groups, GROUP_SIZE],
                packed.iter().map(|&b| b as u8).collect(),
            ));
            raw_tensors.push((
                format!("{raw_name}.scales"),
                "F32".into(),
                vec![rows, groups],
                f32_bytes(&scales),
            ));
            manifest_entries.insert(
                raw_name,
                serde_json::json!({"bits": 8, "group_size": GROUP_SIZE, "shape": descriptor.shape}),
            );
            // The comparison model must hold the dequantized values, not
            // the pre-quantization originals.
            packed
                .iter()
                .zip(
                    scales
                        .iter()
                        .flat_map(|scale| std::iter::repeat_n(*scale, GROUP_SIZE)),
                )
                .map(|(value, scale)| *value as f32 * scale)
                .collect::<Vec<f32>>()
        } else {
            raw_tensors.push((
                raw_name,
                "F32".into(),
                descriptor.shape.clone(),
                f32_bytes(&values),
            ));
            values
        };
        plain_tensors.push((
            name.to_string(),
            "F32".into(),
            descriptor.shape.clone(),
            f32_bytes(&plain_values),
        ));
    }

    let manifest = serde_json::json!({
        "format": "mlx-compatible-safetensors",
        "quantization": {
            "format": "omnivoice-rowwise",
            "bits": 8,
            "group_size": GROUP_SIZE,
            "tensors": serde_json::Value::Object(manifest_entries.clone()),
        },
    });
    std::fs::write(
        raw_dir.join("mlx_manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
    write_safetensors(&raw_dir.join("model.safetensors"), &raw_tensors);
    write_safetensors(&plain_dir.join("model.safetensors"), &plain_tensors);

    let raw = Model::open(&raw_dir).expect("raw-layout model loads");
    let plain = Model::open(&plain_dir).expect("converted-layout model loads");
    let (raw_ids, raw_mask) = {
        let (ids, mask) = raw
            .prepare_inference_inputs(&[3i32, 1, 4], &[5, 2, 6, 7], 2, None)
            .expect("inputs assemble");
        (ids, mask)
    };
    let (plain_ids, plain_mask) = {
        let (ids, mask) = plain
            .prepare_inference_inputs(&[3i32, 1, 4], &[5, 2, 6, 7], 2, None)
            .expect("inputs assemble");
        (ids, mask)
    };
    assert_eq!(raw_ids, plain_ids);
    assert_eq!(raw_mask, plain_mask);

    let mut raw_runner = raw;
    let mut plain_runner = plain;
    let raw_logits = raw_runner
        .forward(&raw_ids, &raw_mask)
        .expect("raw forward");
    let plain_logits = plain_runner
        .forward(&plain_ids, &plain_mask)
        .expect("plain forward");
    assert_eq!(raw_logits.len(), plain_logits.len());
    for (index, (raw, plain)) in raw_logits.iter().zip(&plain_logits).enumerate() {
        assert_eq!(
            raw.to_bits(),
            plain.to_bits(),
            "logit [{index}] diverges between the raw and converted layouts"
        );
    }

    // A manifest whose per-tensor scheme disagrees with its own top level
    // must refuse, never guess.
    std::fs::write(
        raw_dir.join("mlx_manifest.json"),
        serde_json::to_string(&serde_json::json!({
            "quantization": {
                "format": "omnivoice-rowwise",
                "bits": 8,
                "group_size": GROUP_SIZE,
                "tensors": {
                    "llm.embed_tokens.weight": {
                        "bits": 4,
                        "group_size": GROUP_SIZE,
                        "shape": [128, 32],
                    }
                },
            },
        }))
        .unwrap(),
    )
    .unwrap();
    assert!(
        Model::open(&raw_dir).is_err(),
        "a 4-bit manifest entry must refuse"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
