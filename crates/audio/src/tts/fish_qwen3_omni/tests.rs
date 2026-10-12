//! Parity and regression tests for the fish_qwen3_omni family.
//!
//! Pinned fixtures reside in `testdata/tts_fish_qwen3_omni/`, generated
//! by `tools/gen_fish_qwen3_omni_fixtures.py` against mlx-audio 0.5.7
//! commit `e1b19b9054bf163f5d812221a54fcc346f1890e9` (mlx 0.32.3). The
//! generator runs the true reference module with `mx.compile` neutered,
//! a stub tokenizer, and the tiny seeded Fish S1 DAC from
//! `testdata/fish_s1_dac` (reloaded from the same saved weights the
//! Rust codec tests use). Sampling draws replay the recorded uniform
//! arrays one call at a time; greedy calls draw nothing.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::{
    adjust_speed, decode_codes, encode_for_inference, sample_logits, GenerateParams, Generated,
    Model, PromptMessage, PromptPart, RAS_WIN_SIZE,
};
use crate::codec::fish_s1_dac::FishS1Dac;
use crate::tts::sampling;

fn testdata(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/tts_fish_qwen3_omni")
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

/// Measured maximum difference against the fixture traces when the
/// gates were frozen; assertions use this value, not a re-measured one.
/// Slow-stack logits measured 4.2e-5, fast-stack logits 1.5e-6, and
/// waveforms 1.8e-7 (f32 accumulation order against MLX), so the gate
/// sits well above every measurement.
const WAVEFORM_TOLERANCE: f32 = 2e-4;

/// Replays the recorded uniform arrays call by call; panics when the
/// Rust control flow consumes a different call width.
struct UniformReplay {
    calls: Vec<Vec<f32>>,
    index: usize,
}

impl UniformReplay {
    fn new(value: &Value) -> Self {
        Self {
            calls: value
                .as_array()
                .expect("uniform calls")
                .iter()
                .map(|call| {
                    call.as_array()
                        .expect("uniform draw")
                        .iter()
                        .map(|v| v.as_f64().expect("uniform") as f32)
                        .collect()
                })
                .collect(),
            index: 0,
        }
    }

    fn draw(&mut self, width: usize) -> Vec<f32> {
        let call = self
            .calls
            .get(self.index)
            .unwrap_or_else(|| panic!("uniform call {} missing", self.index));
        self.index += 1;
        assert_eq!(
            call.len(),
            width,
            "uniform call {} width {} != {width}",
            self.index - 1,
            call.len()
        );
        call.clone()
    }

    fn consumed(&self) -> usize {
        self.index
    }

    fn recorded(&self) -> usize {
        self.calls.len()
    }
}

fn fish_codec() -> FishS1Dac {
    // The tiny codec fixture stores `tiny_config.json` +
    // `tiny_weights.safetensors` rather than the from_pretrained layout.
    use crate::codec::fish_s1_dac::FishS1DacConfig;
    use turbospark_model_io::safetensors::SafetensorsFile;
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/fish_s1_dac");
    let config_text = std::fs::read_to_string(dir.join("tiny_config.json")).expect("config");
    let value: serde_json::Value = serde_json::from_str(&config_text).expect("config json");
    let config = FishS1DacConfig::from_json(&value).expect("codec config");
    let file = SafetensorsFile::open(&dir.join("tiny_weights.safetensors")).expect("weights");
    FishS1Dac::load(config, &file).expect("tiny codec loads")
}

fn generate_model() -> Model {
    Model::open(&testdata("generate")).expect("family model loads")
}

fn build_bias(model: &mut Model, traces: &Value) {
    let bias = &traces["semantic_logit_bias"];
    model.semantic_logit_bias = Some(model.build_semantic_logit_bias(
        bias["vocab_size"].as_u64().expect("vocab") as usize,
        bias["im_end_id"].as_i64().expect("im_end") as i32,
    ));
}

/// The system + user turns for batch 0, ids straight from the traces.
fn system_and_user(traces: &Value, batch_index: usize) -> Vec<PromptMessage> {
    let headers = &traces["headers"];
    let im_end = i32_vec(&headers["im_end_ids"]);
    vec![
        PromptMessage {
            header_ids: i32_vec(&headers["system_header_ids"]),
            parts: vec![PromptPart::Text(vec![stub_id(
                "convert the provided text to speech",
            )])],
            im_end_ids: im_end.clone(),
        },
        PromptMessage {
            header_ids: i32_vec(&headers["user_header_ids"]),
            parts: vec![PromptPart::Text(i32_vec(
                &traces["batch_text_ids"][batch_index],
            ))],
            im_end_ids: im_end,
        },
    ]
}

fn special_ids(traces: &Value) -> super::SpecialIds {
    let headers = &traces["headers"];
    super::SpecialIds {
        user_header_ids: i32_vec(&headers["user_header_ids"]),
        assistant_voice_header_ids: i32_vec(&headers["assistant_voice_header_ids"]),
        im_end_ids: i32_vec(&headers["im_end_ids"]),
        im_end_token_id: headers["im_end_token_id"].as_i64().expect("im_end") as i32,
    }
}

#[test]
fn forward_rows_match_the_reference() {
    let traces = read_json(&testdata("generate/traces.json"));
    let forward = &traces["forward"];
    let mut model = generate_model();
    let books = model.config.num_codebooks;
    let tc = model.config.text_config.vocab_size;
    let ac_dim = model.config.audio_decoder_config.dim;
    let mut caches = model.model.make_cache();
    // Prefill over the book-major prompt planes.
    let inp = i32_vec(&forward["input"]);
    let len = inp.len() / (books + 1);
    let (logits, hidden) = model
        .model
        .forward(&inp, len, Some(&mut caches))
        .expect("prefill");
    let logits = last_row_of(&logits, tc);
    let hidden = last_row_of(&hidden, ac_dim);
    let diff = max_abs_diff(&logits, &f32_vec(&forward["prompt_logits"]));
    assert!(diff <= WAVEFORM_TOLERANCE, "prefill logits diff {diff}");
    let diff = max_abs_diff(&hidden, &f32_vec(&forward["prompt_hidden"]));
    assert!(diff <= WAVEFORM_TOLERANCE, "prefill hidden diff {diff}");
    // The fast stack over the PREFILL hidden (the first generate step's
    // actual fast input).
    {
        let mut fast_cache = model.model.make_fast_cache();
        let fast_prefill = model
            .model
            .fast_forward_cached(&hidden, &mut fast_cache)
            .expect("fast prefill");
        let diff = max_abs_diff(&fast_prefill, &f32_vec(&forward["prefill_fast_logits"]));
        assert!(diff <= WAVEFORM_TOLERANCE, "prefill fast diff {diff}");
        let code0 = forward["prefill_semantic_code"].as_i64().expect("code") as i32;
        let fast_hidden = super::embedding_row_for_tests(&model, code0).expect("embed");
        let logits = model
            .model
            .fast_forward_cached(&fast_hidden, &mut fast_cache)
            .expect("prefill residual");
        let diff = max_abs_diff(&logits, &f32_vec(&forward["prefill_residual0_logits"]));
        assert!(diff <= WAVEFORM_TOLERANCE, "prefill residual diff {diff}");
    }
    // One decode step on the first eos row (the cache continues).
    let step = i32_vec(&forward["step_input"]);
    let (logits, hidden) = model
        .model
        .forward(&step, 1, Some(&mut caches))
        .expect("step");
    let logits = last_row_of(&logits, tc);
    let hidden = last_row_of(&hidden, ac_dim);
    let diff = max_abs_diff(&logits, &f32_vec(&forward["step_logits"]));
    assert!(diff <= WAVEFORM_TOLERANCE, "step logits diff {diff}");
    let diff = max_abs_diff(&hidden, &f32_vec(&forward["step_hidden"]));
    assert!(diff <= WAVEFORM_TOLERANCE, "step hidden diff {diff}");
    // The fast stack: prefill over the projected hidden, then the
    // residual fills feeding the fixture codes.
    let mut fast_cache = model.model.make_fast_cache();
    let fast_prefill = model
        .model
        .fast_forward_cached(&hidden, &mut fast_cache)
        .expect("fast prefill");
    let diff = max_abs_diff(
        &fast_prefill,
        &f32_vec(&forward["step_fast_prefill_logits"]),
    );
    assert!(diff <= WAVEFORM_TOLERANCE, "step fast prefill diff {diff}");
    let semantic_code = forward["semantic_code"].as_i64().expect("code") as i32;
    let mut fast_hidden = super::embedding_row_for_tests(&model, semantic_code).expect("embed");
    for (book, want) in forward["residual_logits"]
        .as_array()
        .expect("rows")
        .iter()
        .enumerate()
    {
        let logits = model
            .model
            .fast_forward_cached(&fast_hidden, &mut fast_cache)
            .expect("residual");
        let diff = max_abs_diff(&logits, &f32_vec(want));
        assert!(diff <= WAVEFORM_TOLERANCE, "residual {book} diff {diff}");
        let next_code = forward["step_input"][(book + 1) + 1]
            .as_i64()
            .expect("code") as i32;
        fast_hidden = super::embedding_row_for_tests(&model, next_code).expect("embed");
    }
}

fn last_row_of(rows: &[f32], width: usize) -> Vec<f32> {
    rows[rows.len() - width..].to_vec()
}

#[test]
fn family_config_parses_the_reference_fields() {
    let model = generate_model();
    let config = &model.config;
    assert_eq!(config.semantic_start_token_id, 100);
    assert_eq!(config.semantic_end_token_id, 115);
    assert_eq!(config.sample_rate, 44100);
    assert_eq!(config.num_codebooks, 4);
    assert_eq!(config.text_config.vocab_size, 300);
    assert_eq!(config.text_config.dim, 32);
    assert!(config.text_config.attention_qk_norm);
    assert_eq!(config.audio_decoder_config.vocab_size, 12);
    assert_eq!(config.audio_decoder_config.dim, 24);
    assert!(!config.audio_decoder_config.attention_qk_norm);
}

#[test]
fn prompt_encoding_matches_the_reference_matrices() {
    let traces = read_json(&testdata("generate/traces.json"));
    let headers = &traces["headers"];
    let im_end = i32_vec(&headers["im_end_ids"]);
    let books = 4;
    let semantic_begin = 100;

    // First prompt: system + user turn.
    let first =
        encode_for_inference(&system_and_user(&traces, 0), books, semantic_begin).expect("encodes");
    let want = &traces["first_prompt_matrix"];
    assert_eq!(first.tokens, i32_vec(&want["tokens"]));
    assert_eq!(first.codebooks, i32_rows(&want["codebooks"]));

    // Assistant header prompt: the prefill input for the first batch.
    let mut with_header = system_and_user(&traces, 0);
    with_header.push(PromptMessage {
        header_ids: i32_vec(&headers["assistant_voice_header_ids"]),
        parts: Vec::new(),
        im_end_ids: Vec::new(),
    });
    let assistant = encode_for_inference(&with_header, books, semantic_begin).expect("encodes");
    let want = &traces["assistant_prompt_matrix"];
    assert_eq!(assistant.tokens, i32_vec(&want["tokens"]));
    assert_eq!(assistant.codebooks, i32_rows(&want["codebooks"]));

    // Second prompt: carries the first assistant VQ reply; the reply's
    // semantic book rides the text row offset by the semantic begin id.
    let eos_codes = i32_rows(&traces["scenarios"]["eos"]["codes"]);
    let mut conversation = system_and_user(&traces, 0);
    conversation.push(PromptMessage {
        header_ids: i32_vec(&headers["assistant_voice_header_ids"]),
        parts: vec![PromptPart::Vq(eos_codes)],
        im_end_ids: im_end.clone(),
    });
    conversation.push(PromptMessage {
        header_ids: i32_vec(&headers["user_header_ids"]),
        parts: vec![PromptPart::Text(i32_vec(&traces["batch_text_ids"][1]))],
        im_end_ids: im_end,
    });
    let second = encode_for_inference(&conversation, books, semantic_begin).expect("encodes");
    let want = &traces["second_prompt_matrix"];
    assert_eq!(second.tokens, i32_vec(&want["tokens"]));
    assert_eq!(second.codebooks, i32_rows(&want["codebooks"]));

    // The multi-part prompt case with VQ parts mid-conversation.
    let prompt_traces = read_json(&testdata("prompt/traces.json"));
    let matrix_want = &prompt_traces["matrix"];
    let vq = i32_rows(&prompt_traces["vq_codes"]);
    let im_end = i32_vec(&headers["im_end_ids"]);
    let messages = vec![
        PromptMessage {
            header_ids: i32_vec(&headers["system_header_ids"]),
            parts: vec![
                PromptPart::Text(vec![stub_id("convert the provided text to speech")]),
                PromptPart::Text(vec![stub_id("<|speaker:0|>hello")]),
                PromptPart::Text(vec![stub_id("\n\nSpeech:\n")]),
                PromptPart::Vq(vq),
            ],
            im_end_ids: im_end.clone(),
        },
        PromptMessage {
            header_ids: i32_vec(&headers["user_header_ids"]),
            parts: vec![PromptPart::Text(vec![stub_id("hi there")])],
            im_end_ids: im_end,
        },
        PromptMessage {
            header_ids: i32_vec(&headers["assistant_voice_header_ids"]),
            parts: Vec::new(),
            im_end_ids: Vec::new(),
        },
    ];
    let got = encode_for_inference(&messages, books, semantic_begin).expect("encodes");
    assert_eq!(got.tokens, i32_vec(&matrix_want["tokens"]));
    assert_eq!(got.codebooks, i32_rows(&matrix_want["codebooks"]));
}

/// The generator's stub tokenizer maps every string to
/// `3 + ascii_sum % 90`; the tests reuse it for prompt-case text ids.
fn stub_id(text: &str) -> i32 {
    3 + (text.bytes().map(|b| b as usize).sum::<usize>() % 90) as i32
}

#[test]
fn vq_prompt_part_validates_book_shape() {
    let error = encode_for_inference(
        &[PromptMessage {
            header_ids: vec![1],
            parts: vec![PromptPart::Vq(vec![vec![1, 2], vec![3]])],
            im_end_ids: Vec::new(),
        }],
        4,
        100,
    )
    .unwrap_err();
    assert!(error.to_string().contains("books"), "{error}");
}

#[test]
fn sample_logits_matches_the_reference_on_fixed_logits() {
    let traces = read_json(&testdata("sampling/traces.json"));
    for case in traces["cases"].as_array().expect("cases") {
        let logits = f32_vec(&case["logits"]);
        let mut replay = UniformReplay::new(&case["uniform_calls"]);
        let token = sample_logits(
            &logits,
            case["temperature"].as_f64().expect("temp") as f32,
            case["top_p"].as_f64().expect("top_p") as f32,
            case["top_k"].as_i64().expect("top_k") as i32,
            &mut |width| replay.draw(width),
        )
        .expect("sample");
        assert_eq!(
            token,
            case["token"].as_u64().expect("token") as usize,
            "case {}",
            case["name"]
        );
        assert_eq!(
            replay.consumed(),
            replay.recorded(),
            "case {}",
            case["name"]
        );
    }
}

#[test]
fn ras_decisions_match_the_reference() {
    let traces = read_json(&testdata("sampling/traces.json"));
    let bias_info = read_json(&testdata("generate/traces.json"))["semantic_logit_bias"].clone();
    let start = bias_info["semantic_start"].as_i64().expect("start") as i32;
    let end = bias_info["semantic_end"].as_i64().expect("end") as i32;
    for case in traces["ras_cases"].as_array().expect("cases") {
        let logits = f32_vec(&case["logits"]);
        // The additive semantic mask the model builds.
        let mut biased = logits.clone();
        for (index, value) in biased.iter_mut().enumerate() {
            let id = index as i32;
            if !((start..=end).contains(&id) || id == 299) {
                *value += -1e9;
            }
        }
        let previous = i32_vec(&case["previous"]);
        let mut replay = UniformReplay::new(&case["uniform_calls"]);
        let normal = sampling::argmax(&biased);
        let in_range = start <= normal as i32 && normal as i32 <= end;
        let should_use_high = in_range && previous.contains(&(normal as i32));
        assert_eq!(
            should_use_high,
            case["should_use_high"].as_bool().expect("flag"),
            "case {}",
            case["name"]
        );
        let final_token = if should_use_high {
            sample_logits(&biased, 1.0, 0.9, 30, &mut |width| replay.draw(width)).expect("high")
        } else {
            normal
        };
        assert_eq!(
            final_token,
            case["final_token"].as_u64().expect("final") as usize,
            "case {}",
            case["name"]
        );
        assert_eq!(
            replay.consumed(),
            replay.recorded(),
            "case {} draws",
            case["name"]
        );
    }
}

fn run_generate_scenario(traces: &Value, name: &str) -> Generated {
    let scenario = &traces["scenarios"][name];
    let mut model = generate_model();
    build_bias(&mut model, traces);
    let params = GenerateParams {
        max_tokens: scenario["max_tokens"].as_u64().expect("max") as usize,
        temperature: scenario["temperature"].as_f64().expect("temp") as f32,
        top_p: 0.7,
        top_k: 30,
        speed: 1.0,
    };
    let mut replay = UniformReplay::new(&scenario["uniform_calls"]);
    let codes = model
        .generate_codes_for_batch(
            &system_and_user(traces, 0),
            &special_ids(traces),
            traces["batch_token_counts"][0].as_u64().expect("count") as usize,
            &params,
            &mut |width| replay.draw(width),
        )
        .unwrap_or_else(|error| panic!("scenario {name} generate failed: {error}"));
    assert_eq!(
        replay.consumed(),
        replay.recorded(),
        "scenario {name} consumed the full uniform stream"
    );
    let codec = fish_codec();
    let samples = decode_codes(&codec, &codes, 1.0).expect("decodes");
    Generated { codes, samples }
}

#[test]
fn greedy_generation_stops_at_im_end_with_reference_codes() {
    let traces = read_json(&testdata("generate/traces.json"));
    let got = run_generate_scenario(&traces, "eos");
    let want = &traces["scenarios"]["eos"];
    assert_eq!(got.codes, i32_rows(&want["codes"]));
    assert_eq!(
        got.codes[0].len(),
        want["steps"].as_u64().expect("steps") as usize
    );
    let diff = max_abs_diff(&got.samples, &f32_vec(&want["waveform"]));
    assert!(diff <= WAVEFORM_TOLERANCE, "eos waveform diff {diff}");
}

#[test]
fn sampled_generation_replays_the_uniform_stream() {
    let traces = read_json(&testdata("generate/traces.json"));
    let got = run_generate_scenario(&traces, "sampled");
    let want = &traces["scenarios"]["sampled"];
    assert_eq!(got.codes, i32_rows(&want["codes"]));
    let diff = max_abs_diff(&got.samples, &f32_vec(&want["waveform"]));
    assert!(diff <= WAVEFORM_TOLERANCE, "sampled waveform diff {diff}");
}

#[test]
fn budget_scenario_stops_at_max_tokens() {
    let traces = read_json(&testdata("generate/traces.json"));
    let got = run_generate_scenario(&traces, "budget");
    let want = &traces["scenarios"]["budget"];
    assert_eq!(got.codes, i32_rows(&want["codes"]));
    let diff = max_abs_diff(&got.samples, &f32_vec(&want["waveform"]));
    assert!(diff <= WAVEFORM_TOLERANCE, "budget waveform diff {diff}");
}

#[test]
fn semantic_logit_bias_masks_outside_the_range() {
    let model = generate_model();
    let bias = model.build_semantic_logit_bias(300, 299);
    assert_eq!(bias.len(), 300);
    for (index, value) in bias.iter().enumerate() {
        let expected = if (100..=115).contains(&index) || index == 299 {
            0.0
        } else {
            -1e9
        };
        assert_eq!(*value, expected, "bias entry {index}");
    }
}

#[test]
fn generate_refuses_without_semantic_bias() {
    let traces = read_json(&testdata("generate/traces.json"));
    let mut model = generate_model();
    let error = model
        .generate(
            &fish_codec(),
            &system_and_user(&traces, 0),
            &[i32_vec(&traces["batch_text_ids"][0])],
            &[1],
            &special_ids(&traces),
            &GenerateParams::default(),
            0,
        )
        .unwrap_err();
    assert!(error.to_string().contains("semantic logit bias"), "{error}");
}

#[test]
fn decode_budget_must_fit_max_seq_len() {
    // Prompt plus decode steps must fit the max_seq_len RoPE table and
    // cache. Overflow in the sum must refuse too, not wrap.
    assert!(super::check_decode_fits(10, 5, 15).is_ok());
    let error = super::check_decode_fits(10, 6, 15).expect_err("one step too many");
    assert!(matches!(error, crate::SpeechError::Input { .. }), "{error}");
    assert!(super::check_decode_fits(usize::MAX, 1, 100).is_err());
}

#[test]
fn zero_num_codebooks_is_refused_at_load() {
    // The residual loop runs `num_codebooks - 1` times; zero used to
    // underflow there. It is now a BadConfig at config parse.
    let mut value = read_json(&testdata("generate/config.json"));
    value["audio_decoder_config"]["num_codebooks"] = serde_json::json!(0);
    let error = super::ModelConfig::from_json(&value).expect_err("zero books refused");
    assert!(
        matches!(&error, crate::SpeechError::BadConfig { field, .. } if field == "num_codebooks"),
        "{error}"
    );
}

#[test]
fn sample_logits_refuses_a_short_uniform_draw() {
    // A draw vector of the wrong width used to fall back to argmax
    // silently. It is now an Input error.
    let logits = [0.1f32, 0.2, 0.3];
    let error = sample_logits(&logits, 1.0, 1.0, 0, &mut |_| vec![0.5]).expect_err("short draw");
    assert!(matches!(error, crate::SpeechError::Input { .. }), "{error}");
}

#[test]
fn adjust_speed_matches_the_reference() {
    let traces = read_json(&testdata("speed/traces.json"));
    for case in traces["cases"].as_array().expect("cases") {
        let input = f32_vec(&case["input"]);
        let got =
            adjust_speed(input, case["speed"].as_f64().expect("speed") as f32).expect("resamples");
        // The reference builds positions with mx.linspace; the two
        // l computations differ by a few ulps.
        let diff = max_abs_diff(&got, &f32_vec(&case["output"]));
        assert!(diff <= 1e-5, "speed {} diff {diff}", case["speed"]);
    }
    assert!(adjust_speed(Vec::new(), 0.0).is_err());
}

#[test]
fn ras_window_constant_matches_the_reference() {
    assert_eq!(RAS_WIN_SIZE, 10);
    assert_eq!(super::RAS_HIGH_TEMP, 1.0);
    assert_eq!(super::RAS_HIGH_TOP_P, 0.9);
    assert_eq!(stub_id("convert the provided text to speech"), 6);
}
