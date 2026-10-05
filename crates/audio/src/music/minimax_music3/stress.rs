//! Stress and robustness cases beyond the fixture parity suite.
//!
//! Covers the valid request boundaries (sampling steps 1 and 30, a
//! prompt of exactly MAX_PROMPT_TOKENS), the chunk-window schedule at
//! and past the two-chunk boundary, the MAX_AUDIO_FRAMES ceiling, and
//! the end-token-on-the-first-frame zero-frames error path. The end
//! token is forced by patching the end-token row of `lm_head` in a
//! throwaway copy of the converted tree: the patched weight makes the
//! end-token logit dominate the top-k window on the first frame, so
//! the sampler takes it before any audio frame is emitted.
//!
//! The long ceiling runs are `#[ignore]`d because they cost seconds in
//! release and minutes in a debug build; run them with:
//!
//! ```sh
//! cargo test -p turbospark-audio --release minimax_music3::stress \
//!   -- --ignored --nocapture --test-threads=1
//! ```

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

fn testdata(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/minimax_music3")
        .join(name)
}

fn bench_tree(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/minimax_music3_bench")
        .join(name)
}

fn read_json(name: &str) -> serde_json::Value {
    let text = std::fs::read_to_string(testdata(name)).expect(name);
    serde_json::from_str(&text).expect(name)
}

fn floats(value: &serde_json::Value) -> Vec<f32> {
    value
        .as_array()
        .expect("float array")
        .iter()
        .map(|v| v.as_f64().expect("f64") as f32)
        .collect()
}

fn ints(value: &serde_json::Value) -> Vec<i64> {
    value
        .as_array()
        .expect("int array")
        .iter()
        .map(|v| v.as_i64().expect("i64"))
        .collect()
}

fn temp_tree_dir(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir =
        std::env::temp_dir().join(format!("music3-stress-{label}-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

/// Copy the converted plain tree, patching the `lm_head` row of the
/// end token to `1e9 * h` against the recorded frame-0 hidden state.
/// The conditional logit is then `1e9 * |h|^2`, positive regardless of
/// the hidden state's signs, and the test asserts the CFG-guided score
/// clears every other entry by orders of magnitude before relying on
/// it, so the forced end token is proven rather than hoped for.
fn tree_with_forced_end_token() -> PathBuf {
    let source = testdata("converted_plain/model.safetensors");
    let bytes = std::fs::read(&source).expect("model.safetensors");
    let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("header len")) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&bytes[8..8 + header_len]).expect("safetensors header");
    let entry = &header["language_model.lm_head.weight"];
    // Safetensors data offsets are relative to the data section, which
    // starts after the 8-byte length and the header itself.
    let section_base = 8 + header_len;
    let row_start = section_base + entry["data_offsets"][0].as_u64().expect("offset") as usize;

    let config = read_json("tiny_config.json");
    let end_token = config["audio_end_token_id"].as_u64().expect("end id") as usize;
    let hidden = config["hidden_size"].as_u64().expect("hidden") as usize;

    // The trace records the last hidden state of frame 0 for both CFG
    // rows; row 0 is the conditional half.
    let trace = read_json("ar_trace_short.json");
    let last_hidden = floats(&trace["frames"][0]["last_hidden"]);
    assert_eq!(last_hidden.len(), 2 * hidden, "both CFG rows recorded");
    let (conditional, unconditional) = last_hidden.split_at(hidden);

    let row = row_start + end_token * hidden * 4;
    let mut patched = bytes.clone();
    for (d, value) in conditional.iter().enumerate() {
        let offset = row + d * 4;
        let forced = 1.0e9f32 * value;
        patched[offset..offset + 4].copy_from_slice(&forced.to_le_bytes());
    }

    // guided = 1.5 * cond - 0.5 * uncond, with AR_CFG_SCALE = 1.5, and
    // the patched row contributes 1e9 * <h, cond> to both logits.
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let guided =
        1.0e9 * (1.5 * dot(conditional, conditional) - 0.5 * dot(unconditional, conditional));
    let scale = conditional.iter().map(|v| v.abs()).sum::<f32>();
    assert!(
        guided > 1.0e8 * scale.max(1.0),
        "patched end-token margin too small: {guided} vs scale {scale}"
    );

    let dir = temp_tree_dir("end-token");
    std::fs::write(dir.join("model.safetensors"), patched).expect("patched tree");
    std::fs::copy(
        testdata("converted_plain/config.json"),
        dir.join("config.json"),
    )
    .expect("config copy");
    dir
}

#[test]
fn step_boundaries_one_and_thirty_generate() {
    let model = Model::load_converted(&testdata("converted_plain")).expect("plain loads");
    let text_ids = vec![1i32, 5, 6, 2];
    for steps in [1usize, 30] {
        let generation = model
            .generate(&GenerateRequest {
                text_ids: text_ids.clone(),
                frames: 2,
                steps,
                seed: 7,
            })
            .unwrap_or_else(|err| panic!("steps {steps} refused: {err}"));
        assert!(generation.waveform.iter().all(|v| v.is_finite()));
        assert_eq!(generation.sample_rate, SAMPLING_RATE);
    }
}

#[test]
fn empty_prompt_is_refused() {
    let model = Model::load_converted(&testdata("converted_plain")).expect("plain loads");
    let err = model
        .generate_frame_hiddens(&[], 4, 7)
        .expect_err("empty prompt");
    assert!(err.to_string().contains("empty prompt"), "{err}");
}

/// The end token can stop the AR stage before the requested frame
/// count; generate() must hand the flow stage the emitted hiddens, not
/// the request. With the fixture weights the short prompt stops at 27
/// frames, so a 50-frame request exercises the early stop end to end.
#[test]
fn generate_completes_when_the_end_token_stops_early() {
    let model = Model::load_converted(&testdata("converted_plain")).expect("plain loads");
    let trace = read_json("ar_trace_short.json");
    let text_ids: Vec<i32> = ints(&trace["text_ids"][0])
        .iter()
        .map(|v| *v as i32)
        .collect();
    let (_, codes) = model
        .generate_frame_hiddens(&text_ids, 50, 7)
        .expect("AR stage");
    assert!(
        codes.len() < 50,
        "the fixture tree must stop early on the end token"
    );
    let generation = model
        .generate(&GenerateRequest {
            text_ids,
            frames: 50,
            steps: 2,
            seed: 7,
        })
        .expect("early end token must not fail generate()");
    assert_eq!(generation.frames, codes.len(), "Generation.frames");
    assert!(generation.waveform.iter().all(|v| v.is_finite()));
}

#[test]
fn chunk_windows_at_and_past_the_two_chunk_boundary() {
    assert_eq!(chunk_starts(1), vec![0]);
    assert_eq!(chunk_starts(CHUNK_FRAMES), vec![0]);
    assert_eq!(chunk_starts(CHUNK_FRAMES + 1), vec![0, CHUNK_HOP]);
    // 300 frames still fit two overlapping windows; 301 spills into a
    // third short tail window.
    assert_eq!(chunk_starts(300), vec![0, 100]);
    assert_eq!(chunk_starts(301), vec![0, 100, 200]);
    let starts = chunk_starts(MAX_AUDIO_FRAMES);
    assert_eq!(starts.len(), (MAX_AUDIO_FRAMES - CHUNK_HOP) / CHUNK_HOP);
    assert_eq!(
        *starts.last().expect("nonempty"),
        MAX_AUDIO_FRAMES - 2 * CHUNK_HOP
    );
}

/// Tiled hidden states drive multi-chunk flow schedules without the AR
/// stage, mirroring the fixture generator's two-chunk scenario.
fn tiled_hiddens(model: &Model, frames: usize) -> Vec<f32> {
    let fused = model.config().num_codebooks * model.config().hidden_size;
    let trace = read_json("ar_trace_short.json");
    let short = floats(&trace["frame_hiddens_stacked"]);
    assert_eq!(short.len(), 2 * fused, "short trace frame hiddens");
    let mut hiddens = Vec::with_capacity(frames * fused);
    for frame in 0..frames {
        let src = (frame % 2) * fused;
        hiddens.extend_from_slice(&short[src..src + fused]);
    }
    hiddens
}

#[test]
fn flow_schedules_at_200_201_300_and_301_frames() {
    let model = Model::load_converted(&testdata("converted_plain")).expect("plain loads");
    let mut lengths = Vec::new();
    for frames in [200usize, 201, 300, 301] {
        let hiddens = tiled_hiddens(&model, frames);
        let first = model
            .run_flow(&hiddens, frames, 2, 7)
            .unwrap_or_else(|err| panic!("flow at {frames} frames failed: {err}"));
        let second = model.run_flow(&hiddens, frames, 2, 7).expect("repeat flow");
        assert_eq!(
            first, second,
            "flow at {frames} frames must be deterministic"
        );
        assert_eq!(first.len() % 2, 0, "planar stereo");
        assert!(first.iter().all(|v| v.is_finite()), "frames {frames}");
        lengths.push(first.len() / 2);
    }
    // Sample counts must grow with the schedule (extra chunks add the
    // overlap regions back at tiny dims, where the crop constants
    // exceed the chunk and the short-chunk guard keeps chunks whole).
    assert!(
        lengths[0] < lengths[1] && lengths[1] < lengths[2] && lengths[2] < lengths[3],
        "stitched samples must grow with frames: {lengths:?}"
    );
}

#[test]
fn end_token_on_the_first_frame_yields_the_zero_frames_error() {
    let dir = tree_with_forced_end_token();
    let model = Model::load_converted(&dir).expect("patched tree loads");
    // The patched row is fitted to the recorded frame-0 hidden state,
    // so the run must replay that exact prompt.
    let trace = read_json("ar_trace_short.json");
    let text_ids: Vec<i32> = ints(&trace["text_ids"][0])
        .iter()
        .map(|v| *v as i32)
        .collect();
    let err = model
        .generate(&GenerateRequest {
            text_ids: text_ids.clone(),
            frames: 8,
            steps: 2,
            seed: 7,
        })
        .expect_err("the forced end token must stop before any audio frame");
    assert!(
        err.to_string().contains("zero audio frames"),
        "unexpected error: {err}"
    );
    // The AR stage reports the same failure on its own.
    let err = model
        .generate_frame_hiddens(&text_ids, 8, 7)
        .expect_err("zero frames");
    assert!(err.to_string().contains("zero audio frames"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The AR stage runs to MAX_AUDIO_FRAMES and returns exactly that many
/// frames when the end token never wins. Needs the long-AR bench tree
/// (the fixture tree stops on the end token after 27 frames).
#[test]
#[ignore = "runs 9000 AR frames; seconds in release, minutes in debug"]
fn ar_runs_to_the_frame_ceiling() {
    let dir = bench_tree("tiny_long");
    if !dir.join("config.json").is_file() {
        panic!(
            "missing {}; generate it with tools/gen_minimax_music3_bench_tree.py --variant tiny_long",
            dir.display()
        );
    }
    let model = Model::load_converted(&dir).expect("tiny_long loads");
    let text_ids = vec![1i32, 5, 6, 2];
    let start = std::time::Instant::now();
    let (hiddens, codes) = model
        .generate_frame_hiddens(&text_ids, MAX_AUDIO_FRAMES, 7)
        .expect("ceiling run");
    let wall = start.elapsed();
    let config = model.config();
    let fused = config.num_codebooks * config.hidden_size;
    assert_eq!(hiddens.len(), MAX_AUDIO_FRAMES * fused, "frame hiddens");
    assert_eq!(codes.len(), MAX_AUDIO_FRAMES, "frame codes");
    assert!(codes.iter().all(|code| {
        code.len() == config.num_codebooks
            && (code[0] as usize) < config.semantic_vocab_size
            && code[1..]
                .iter()
                .all(|c| (*c as usize) < config.audio_vocab_size)
    }));
    println!(
        "ar ceiling: {MAX_AUDIO_FRAMES} frames in {:.2?} ({:.3} ms/frame), tiny dims",
        wall,
        wall.as_secs_f64() * 1e3 / MAX_AUDIO_FRAMES as f64
    );
}

/// A prompt of exactly MAX_PROMPT_TOKENS is accepted and generates.
#[test]
#[ignore = "5000-token causal prefill; slow in a debug build"]
fn prompt_at_exact_max_tokens_generates() {
    let model = Model::load_converted(&testdata("converted_plain")).expect("plain loads");
    let vocab = model.config().vocab_size;
    let text_ids: Vec<i32> = (0..MAX_PROMPT_TOKENS)
        .map(|i| ((i * 7 + 3) % vocab) as i32)
        .collect();
    let generation = model
        .generate(&GenerateRequest {
            text_ids,
            frames: 4,
            steps: 2,
            seed: 7,
        })
        .expect("prompt at the token ceiling");
    assert!(generation.waveform.iter().all(|v| v.is_finite()));
}
