//! The `unlimited_ocr2` flow on a synthetic install. Untrained weights make
//! "it decoded" worthless (`crates/runtime` Gotcha 11), so each test moves one
//! input and requires the logits to move, or not to.
//!
//! **Blind spots, stated:** nothing here compares against a reference, and
//! nothing can show V stays unrotated. Those are the real-model gates
//! (`unlimited_ocr2_real.rs`, `docs/UNLIMITED_OCR2_PHASE0.md`).
#![cfg(target_os = "macos")]

use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use turbospark_repack::{
    build_synthetic_unlimited_ocr2_install, SYNTHETIC_UOCR2_IMAGE_TOKEN as IMAGE,
    SYNTHETIC_UOCR2_PAGE_TOKEN as PAGE,
};
use turbospark_runtime::{LogitProducer, RealForwardRunner};

const VOCAB: i64 = 512;
const HIDDEN: usize = 128;

fn tempdir() -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("uocr2-synth-{}-{unique}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn open() -> RealForwardRunner {
    let dir = tempdir();
    let arch = build_synthetic_unlimited_ocr2_install(&dir, VOCAB).unwrap();
    RealForwardRunner::open_with_max_context(&dir, arch, 1024).unwrap()
}

/// `<bos> <PAGE> images... text...`, past the point routing can reach them.
fn prompt(images: usize, text: usize) -> Vec<i32> {
    let mut ids = vec![0, PAGE];
    ids.extend(std::iter::repeat_n(IMAGE, images));
    ids.extend((0..text).map(|t| 200 + (t % 100) as i32));
    ids
}

fn rows(images: usize, seed: u32) -> Vec<f16> {
    let mut x = seed.wrapping_mul(2_654_435_761) | 1;
    (0..images * HIDDEN)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            f16::from_f32((x as f32 / u32::MAX as f32) * 2.0 - 1.0)
        })
        .collect()
}

fn run(runner: &mut RealForwardRunner, ids: &[i32], image_rows: Vec<f16>) -> Vec<u16> {
    runner.set_unlimited_ocr2_prompt(ids, image_rows).unwrap();
    LogitProducer::reset(runner);
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    for (position, &token) in ids.iter().enumerate() {
        runner.produce(token, position, &mut logits).unwrap();
    }
    logits.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn a_used_runner_replays_a_fresh_ones_generation() {
    let ids = prompt(64, 200);
    let fresh = run(&mut open(), &ids, rows(64, 1));
    assert!(fresh.iter().any(|b| *b != 0), "all-zero logits");
    // Against a FRESH runner, and with other rows first. Stale summaries keep
    // the low block indices, so a used runner replaying its own first rows would
    // look up exactly the values it wrote and pass with no reset at all.
    let mut used = open();
    let other = run(&mut used, &ids, rows(64, 2));
    assert_ne!(fresh, other, "image rows never reached the logits");
    assert_eq!(
        fresh,
        run(&mut used, &ids, rows(64, 1)),
        "reset left router or KV state behind"
    );
}

#[test]
fn a_query_routes_to_image_blocks_past_its_local_window() {
    let mut runner = open();
    let ids = prompt(64, 200);
    run(&mut runner, &ids, rows(64, 1));
    let (positions, bias, _) = runner.unlimited_ocr2_last_route().unwrap();
    let pos = ids.len() as u32 - 1;
    let remote: Vec<usize> = (0..positions.len())
        .filter(|&i| positions[i] + 128 <= pos)
        .collect();
    assert!(!remote.is_empty(), "no key past the local window");
    // Trailing text is on no page yet (upstream's response page -1), so it may
    // route to any completed block. `candidates_restrict_a_query_to_its_page`
    // covers the restriction.
    // The whole local window is attended, unbiased, on top.
    let local: Vec<u32> = positions
        .iter()
        .copied()
        .filter(|p| p + 128 > pos)
        .collect();
    assert_eq!(local, ((pos - 127)..=pos).collect::<Vec<_>>());
    assert!(positions.len() <= 512);
    // Three blocks route, so their centred log-priors cannot all be zero unless
    // the prior was skipped. (Bias is what makes this DASH and not a plain cut.)
    assert!(
        remote.iter().any(|&i| bias[i] != 0.0),
        "the routed blocks carry no bias"
    );
    assert!(
        (0..positions.len())
            .filter(|&i| positions[i] + 128 > pos)
            .all(|i| bias[i] == 0.0),
        "the local window must be unbiased"
    );
}

#[test]
fn candidates_restrict_a_query_to_its_page() {
    let mut runner = open();
    // Page 0 is positions 2..=201, page 1 is 203..=402. Block 6 (192..=223)
    // holds both, so it is a candidate for either.
    let mut ids = vec![0, PAGE];
    ids.extend(std::iter::repeat_n(IMAGE, 200));
    ids.push(PAGE);
    ids.extend(std::iter::repeat_n(IMAGE, 200));
    let remote = |runner: &RealForwardRunner, pos: u32| -> Vec<u32> {
        let (positions, _, _) = runner.unlimited_ocr2_last_route().unwrap();
        positions.into_iter().filter(|p| p + 128 <= pos).collect()
    };
    run(&mut runner, &ids, rows(400, 1));
    let pos = ids.len() as u32 - 1;
    let page_one = remote(&runner, pos);
    assert!(!page_one.is_empty(), "no routed keys");
    assert!(
        page_one.iter().all(|&p| p >= 192),
        "a page 1 query read page 0: {page_one:?}"
    );
    // A generated `<PAGE>` starts response page 0, which reads page 0's blocks.
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    runner.produce(PAGE, ids.len(), &mut logits).unwrap();
    let pos = ids.len() as u32;
    let response = remote(&runner, pos);
    assert!(!response.is_empty(), "no routed keys after the marker");
    assert!(
        response.iter().all(|&p| p < 224),
        "response page 0 read page 1: {response:?}"
    );
}

#[test]
fn a_position_rotates_by_its_page_coordinates() {
    let mut runner = open();
    // Two pages. Page coordinates count the markers, and the last pair's angle
    // is the page modulo 2 times an inverse frequency of 1.
    let mut ids = vec![0, PAGE];
    ids.extend(std::iter::repeat_n(IMAGE, 8));
    ids.push(PAGE);
    ids.extend(std::iter::repeat_n(IMAGE, 8));
    ids.extend([200, 201]);
    runner.set_unlimited_ocr2_prompt(&ids, rows(16, 1)).unwrap();
    LogitProducer::reset(&mut runner);
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    for (position, &token) in ids.iter().enumerate() {
        runner.produce(token, position, &mut logits).unwrap();
        let (_, _, angles) = runner.unlimited_ocr2_last_route().unwrap();
        let page_angle = *angles.last().unwrap();
        let (page_mod_2, tok) = match position {
            0 => (0.0, 0.0),
            1..=9 => (1.0, position as f32 - 1.0),
            10..=18 => (0.0, position as f32 - 10.0),
            // Past the last image the pages restart at 0, and these tokens
            // continue page 0's count from BOS.
            _ => (0.0, position as f32 - 18.0),
        };
        assert_eq!(page_angle, page_mod_2, "page angle at {position}");
        assert_eq!(angles[0], tok, "token angle at {position}");
    }
}

#[test]
fn image_rows_reach_later_queries() {
    let mut runner = open();
    // Through routing or through the chain of local windows: this cannot tell
    // which, and `a_query_routes_to_image_blocks_past_its_local_window` does.
    let ids = prompt(64, 200);
    let a = run(&mut runner, &ids, rows(64, 1));
    let b = run(&mut runner, &ids, rows(64, 2));
    assert_ne!(a, b, "the image rows were not injected");
}

#[test]
fn feeding_other_ids_than_the_installed_prompt_is_refused() {
    let mut runner = open();
    let ids = prompt(4, 4);
    runner.set_unlimited_ocr2_prompt(&ids, rows(4, 1)).unwrap();
    LogitProducer::reset(&mut runner);
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let err = runner.produce(5, 0, &mut logits).unwrap_err();
    assert!(err.to_string().contains("installed prompt"), "{err}");
}

#[test]
fn a_prompt_needs_one_row_per_image_token() {
    let mut runner = open();
    let err = runner
        .set_unlimited_ocr2_prompt(&prompt(4, 4), rows(3, 1))
        .unwrap_err();
    assert!(err.to_string().contains("embedding rows"), "{err}");
}

#[test]
fn decode_allocates_no_metal_buffers() {
    let mut runner = open();
    let ids = prompt(8, 8);
    run(&mut runner, &ids, rows(8, 1));
    let before = runner.gpu_buffer_allocations();
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    for t in 0..40 {
        runner
            .produce(300 + t as i32, ids.len() + t, &mut logits)
            .unwrap();
    }
    assert_eq!(runner.gpu_buffer_allocations(), before);
}

#[test]
fn a_decoded_page_marker_advances_without_error() {
    let mut runner = open();
    let ids = prompt(8, 8);
    run(&mut runner, &ids, rows(8, 1));
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    // Past the last prompt page, where `advance` clamps.
    for (t, token) in [PAGE, 300, PAGE, PAGE, 301].into_iter().enumerate() {
        runner.produce(token, ids.len() + t, &mut logits).unwrap();
    }
    assert!(logits.iter().all(|v| v.to_f32().is_finite()));
}
