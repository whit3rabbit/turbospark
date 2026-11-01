//! Gemma 4 image injection on the synthetic trunk
//! (`families/gemma4/vision_prefill.rs`).
//!
//! The synthetic installs declare no tower, so the rows are made up here and
//! set through the test-only setter; the placeholder id is the arch's
//! `image_token_id`, 0 on a tower-less install. Six layers, so layer 5 is a
//! full-attention layer and the rest slide with a window of 8.
//!
//! What is held:
//! - the chunk boundary never moves the logits, including chunks that end
//!   inside a span and chunks the span already ran past;
//! - the rows reach the model;
//! - an image row attends AHEAD on the sliding layers: changing only later
//!   rows of its span moves its logits, which causal attention could not do;
//! - the sequential path refuses an image position instead of guessing.
#![cfg(target_os = "macos")]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use turbospark_repack::{
    build_synthetic_gemma4_dense_real_install, build_synthetic_gemma4_real_install,
};
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};
use turbospark_vision_io::ImageSpan;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-gemma4-vision-trunk-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const VOCAB: i64 = 128;
const LAYERS: i64 = 6;
const WINDOW: i64 = 8;
const IMAGE: i32 = 0;
/// Text, a five-token image span at 3..8, text.
const PROMPT: [i32; 11] = [5, 9, 2, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, 7, 1, 3];
const SPAN: ImageSpan = ImageSpan { start: 3, len: 5 };

fn open(moe: bool) -> (RealForwardRunner, usize) {
    let dir = temp_dir();
    let arch = if moe {
        build_synthetic_gemma4_real_install(
            &dir,
            VOCAB,
            LAYERS,
            16,
            4,
            WINDOW,
            "tiny-g4-vision-moe",
        )
    } else {
        build_synthetic_gemma4_dense_real_install(
            &dir,
            VOCAB,
            LAYERS,
            WINDOW,
            "tiny-g4-vision-dense",
        )
    }
    .expect("synthetic install builds");
    assert_eq!(arch.vision.image_token_id as i32, IMAGE);
    assert!(arch.full_attention_layer_mask[5] == 1 && arch.full_attention_layer_mask[0] == 0);
    let hidden = arch.hidden_size as usize;
    (
        RealForwardRunner::open_with_options(&dir, arch, 4096, 16).expect("install opens"),
        hidden,
    )
}

/// Deterministic, non-flat FP16 rows, `seed` picks the pattern.
fn rows(len: usize, hidden: usize, seed: f32) -> Vec<u16> {
    (0..len * hidden)
        .map(|i| f16::from_f32(0.8 * ((i as f32) * 0.37 + seed).sin()).to_bits())
        .collect()
}

fn chunked(
    runner: &mut RealForwardRunner,
    tokens: &[i32],
    image: &[u16],
    chunk: usize,
) -> Vec<f32> {
    runner.reset();
    runner
        .set_prompt_vision_gemma4_rows_for_tests(&[image.to_vec()], &[SPAN], tokens.len())
        .expect("rows set");
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let mut offset = 0;
    while offset < tokens.len() {
        let take = (tokens.len() - offset).min(chunk);
        runner
            .prefill_chunk_with_status(
                &tokens[offset..offset + take],
                offset,
                &mut logits,
                offset + take == tokens.len(),
            )
            .expect("chunk succeeds");
        offset += take;
    }
    logits.iter().map(|v| v.to_f32()).collect()
}

fn check_chunk_invariance(moe: bool) {
    let (mut runner, h) = open(moe);
    let image = rows(SPAN.len, h, 0.5);
    let expected = chunked(&mut runner, &PROMPT, &image, PROMPT.len());
    assert!(expected.iter().all(|v| v.is_finite()));
    assert!(
        expected.iter().any(|&v| v != expected[0]),
        "degenerate logits"
    );
    // 1 and 2 end chunks inside the span and leave whole chunks the span
    // already ran; 4 ends one exactly at the span's first row.
    for chunk in [1usize, 2, 3, 4, 5, 7, 8] {
        let actual = chunked(&mut runner, &PROMPT, &image, chunk);
        assert_eq!(
            actual, expected,
            "chunk {chunk} moved the logits (moe {moe})"
        );
    }
}

#[test]
fn the_chunk_boundary_does_not_move_an_image_prompt_dense() {
    check_chunk_invariance(false);
}

#[test]
fn the_chunk_boundary_does_not_move_an_image_prompt_moe() {
    check_chunk_invariance(true);
}

fn check_rows_and_look_ahead(moe: bool) {
    let (mut runner, h) = open(moe);
    let a = rows(SPAN.len, h, 0.5);

    // The rows reach the model.
    let base = chunked(&mut runner, &PROMPT, &a, 4);
    let mut flipped = a.clone();
    flipped[2 * h + 3] = f16::from_f32(3.0).to_bits();
    let moved = chunked(&mut runner, &PROMPT, &flipped, 4);
    assert_ne!(
        base, moved,
        "an image row change did not reach the logits (moe {moe})"
    );

    // Look-ahead: a chunk ending on the span's FIRST row returns that row's
    // logits. Changing only rows 1..5 must move them, through the sliding
    // layers. Causal attention would leave them identical.
    // The prompt is cut after the span's first row, so the span runs past
    // the chunk and the map has to describe the whole prompt: rebuild it
    // with the full length.
    let run = |runner: &mut RealForwardRunner, image: &[u16]| {
        runner.reset();
        runner
            .set_prompt_vision_gemma4_rows_for_tests(&[image.to_vec()], &[SPAN], PROMPT.len())
            .expect("rows set");
        let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
        runner
            .prefill_chunk_with_status(&PROMPT[..SPAN.start + 1], 0, &mut logits, false)
            .expect("chunk succeeds");
        logits.iter().map(|v| v.to_f32()).collect::<Vec<f32>>()
    };
    let first_a = run(&mut runner, &a);
    let mut later = a.clone();
    for v in &mut later[h..] {
        *v = f16::from_f32(-f16::from_bits(*v).to_f32()).to_bits();
    }
    let first_b = run(&mut runner, &later);
    assert_eq!(&a[..h], &later[..h], "row 0 itself must be unchanged");
    assert_ne!(
        first_a, first_b,
        "the span's first row ignored the rows after it (moe {moe}); the sliding layers are \
         not attending ahead"
    );
}

#[test]
fn image_rows_reach_the_model_and_attend_ahead_dense() {
    check_rows_and_look_ahead(false);
}

#[test]
fn image_rows_reach_the_model_and_attend_ahead_moe() {
    check_rows_and_look_ahead(true);
}

#[test]
fn the_sequential_path_refuses_an_image_position() {
    let (mut runner, h) = open(false);
    runner.reset();
    runner
        .set_prompt_vision_gemma4_rows_for_tests(&[rows(SPAN.len, h, 0.5)], &[SPAN], PROMPT.len())
        .expect("rows set");
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    for (p, &t) in PROMPT[..SPAN.start].iter().enumerate() {
        runner
            .produce_prefill(t, p, &mut logits)
            .expect("text decodes");
    }
    let err = runner
        .produce_prefill(IMAGE, SPAN.start, &mut logits)
        .expect_err("an image position must be refused on the sequential path");
    assert!(err.contains("image placeholder"), "{err}");
}

#[test]
fn a_span_wider_than_the_ring_is_refused_up_front() {
    let (mut runner, h) = open(false);
    // The ring is window + 128 on a tower-less install.
    let len = 200;
    let err = runner
        .set_prompt_vision_gemma4_rows_for_tests(
            &[rows(len, h, 0.5)],
            &[ImageSpan { start: 1, len }],
            len + 2,
        )
        .expect_err("a span the ring cannot hold must be refused");
    assert!(err.to_string().contains("ring"), "{err}");
}
