//! Image injection in the `lfm3` flow, on a synthetic install
//! (`docs/LFM3_VL_VISION_PHASE0.md`).
//!
//! # What it can and cannot see
//!
//! It CAN see what the lfm3 flow wires: that a tower row replaces the
//! embedding lookup at an image-pad position and nothing else, at BOTH embed
//! sites (the sequential `produce` and the chunked prefill driver), and that
//! the rows reach the logits. The trunk's positions stay plain sequential, so
//! there is no angle half to this seam, unlike the qwen flows.
//!
//! It CANNOT see whether the injected rows are the RIGHT rows: the fixture's
//! weights are untrained and the rows are arbitrary. The real-install arm
//! (`lfm3_vl_vision_real.rs`, plus the page transcription) measures the
//! function.
//!
//! The special ids are in-vocab stand-ins (the fixture vocabulary is 64):
//! they are a PARAMETER of `mrope_position_triples`, and on a real install
//! the placeholder is an ordinary vocabulary member.
#![cfg(target_os = "macos")]

use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_repack::build_synthetic_lfm3_mlx_install;
use turbospark_runtime::vision::VisionEmbedding;
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};
use turbospark_vision_io::{mrope_position_triples, GridThw, MropePositions, VisionSpecialIds};

const VOCAB: i64 = 64;
const LAYERS: i64 = 8;
const PROMPT: usize = 200;
/// In-vocab stand-ins, kept out of the filler tokens below.
const VISION_START: i32 = 60;
const IMAGE_PAD: i32 = 61;
/// Grid 1x4x4 at merge 2 is a 2x2 block of merged tokens.
const GRID: GridThw = GridThw { t: 1, h: 4, w: 4 };
const MERGED: usize = 4;
/// Where the marker sits: the four placeholders follow it. Chosen so a split
/// at 100 lands mid-block.
const MARKER_AT: usize = 99;

fn tempdir() -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "turbospark-lfm3-inject-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn install() -> (std::path::PathBuf, model_io::ArchConfig) {
    let dir = tempdir();
    let arch = build_synthetic_lfm3_mlx_install(&dir, VOCAB, LAYERS, "lfm3-toy-vision", 64)
        .expect("writes");
    (dir, arch)
}

/// Filler below the special ids, the marker, then four placeholders.
fn prompt_ids() -> Vec<i32> {
    let mut ids: Vec<i32> = (0..PROMPT).map(|i| ((i * 7 + 3) % 59) as i32).collect();
    ids[MARKER_AT] = VISION_START;
    for slot in ids.iter_mut().skip(MARKER_AT + 1).take(MERGED) {
        *slot = IMAGE_PAD;
    }
    ids
}

fn positions(ids: &[i32]) -> MropePositions {
    mrope_position_triples(
        ids,
        &[GRID],
        VisionSpecialIds {
            vision_start: VISION_START,
            image_pad: IMAGE_PAD,
        },
        2,
    )
    .expect("the walk places the one image")
}

/// `MERGED` rows of `hidden` FP16 values, distinct per row and per `seed`.
fn embedding(hidden: usize, seed: f32) -> VisionEmbedding {
    let rows: Vec<u16> = (0..MERGED * hidden)
        .map(|i| half::f16::from_f32(((i as f32) * 0.173 + seed).sin() * 0.8).to_bits())
        .collect();
    VisionEmbedding {
        rows,
        merged_tokens: MERGED,
        out_hidden: hidden,
        grid: GRID,
        deepstack: Vec::new(),
    }
}

fn open(dir: &std::path::Path, arch: &model_io::ArchConfig) -> RealForwardRunner {
    RealForwardRunner::open(dir, arch.clone()).expect("the lfm3 install opens")
}

/// The synthetic arch declares no tower, so the trunk's own width is the row
/// width (`set_prompt_vision` validates against it).
fn with_image(runner: &mut RealForwardRunner, arch: &model_io::ArchConfig, ids: &[i32], seed: f32) {
    runner
        .set_prompt_vision(
            &[embedding(arch.hidden_size as usize, seed)],
            &positions(ids),
            ids.len(),
        )
        .expect("the map validates");
}

/// `produce` over `ids`, then three decode steps; the logits after each.
fn sequential(runner: &mut RealForwardRunner, ids: &[i32]) -> Vec<Vec<u16>> {
    let mut logits = vec![half::f16::ZERO; VOCAB as usize];
    for (i, &t) in ids.iter().enumerate() {
        runner.produce(t, i, &mut logits).expect("produce");
    }
    let mut out = vec![logits.iter().map(|v| v.to_bits()).collect::<Vec<u16>>()];
    for step in 0..3 {
        runner
            .produce((step % 5) as i32, ids.len() + step, &mut logits)
            .expect("produce");
        out.push(logits.iter().map(|v| v.to_bits()).collect());
    }
    out
}

/// The same through `prefill_chunk` calls of `split` tokens. The exact arm
/// (scalar GEMM, one-row attention) is bit-identical to sequential prefill.
fn chunked(runner: &mut RealForwardRunner, ids: &[i32], split: &[usize]) -> Vec<Vec<u16>> {
    assert_eq!(split.iter().sum::<usize>(), ids.len());
    assert!(runner.supports_chunked_prefill());
    runner.set_prefill_steel(false);
    runner.set_prefill_rows_attention(false);
    let mut logits = vec![half::f16::ZERO; VOCAB as usize];
    let mut at = 0usize;
    for &n in split {
        runner
            .prefill_chunk(&ids[at..at + n], at, &mut logits)
            .expect("prefill_chunk");
        at += n;
    }
    let mut out = vec![logits.iter().map(|v| v.to_bits()).collect::<Vec<u16>>()];
    for step in 0..3 {
        runner
            .produce((step % 5) as i32, ids.len() + step, &mut logits)
            .expect("produce");
        out.push(logits.iter().map(|v| v.to_bits()).collect());
    }
    out
}

/// BOTH embed sites blit the same rows: the chunked driver, under splits
/// that put the image block inside one chunk and straddling a boundary, is
/// bit-identical to the per-token walk, and so is every decode step after.
#[test]
fn both_embed_sites_inject_the_same_rows() {
    let (dir, arch) = install();
    let ids = prompt_ids();

    let mut seq = open(&dir, &arch);
    with_image(&mut seq, &arch, &ids, 0.0);
    let want = sequential(&mut seq, &ids);

    // One call; a cut at 100 (mid-block: the marker is at 99, the
    // placeholders at 100..104); and one that ends a chunk between two
    // placeholders.
    for split in [
        vec![PROMPT],
        vec![100, 2, 98],
        vec![MARKER_AT + 3, 1, PROMPT - MARKER_AT - 4],
    ] {
        let mut runner = open(&dir, &arch);
        with_image(&mut runner, &arch, &ids, 0.0);
        let got = chunked(&mut runner, &ids, &split);
        assert_eq!(got, want, "split {split:?}");
    }
}

/// The blit REPLACES the lookup rather than riding beside it: the id spelled
/// at an image position moves nothing, and an id at a text position moves
/// something. The first half alone passes against a flow that ignores token
/// ids, the second alone against one that never blits.
#[test]
fn the_blit_replaces_the_lookup_and_text_positions_still_look_up() {
    let (dir, arch) = install();
    let ids = prompt_ids();

    let mut base = open(&dir, &arch);
    with_image(&mut base, &arch, &ids, 0.0);
    let want = sequential(&mut base, &ids);

    // A different (in-vocab, non-special) id under every placeholder. The map
    // is built from the ORIGINAL ids: it is the walk's output, not a function
    // of what `produce` is later handed.
    let mut renamed = ids.clone();
    for slot in renamed.iter_mut().skip(MARKER_AT + 1).take(MERGED) {
        *slot = 5;
    }
    let mut runner = open(&dir, &arch);
    with_image(&mut runner, &arch, &ids, 0.0);
    assert_eq!(sequential(&mut runner, &renamed), want);

    let mut edited = ids.clone();
    edited[50] = (edited[50] + 1) % 59;
    let mut runner = open(&dir, &arch);
    with_image(&mut runner, &arch, &ids, 0.0);
    assert_ne!(sequential(&mut runner, &edited), want);
}

/// The rows reach the logits, and an image run is not a no-image run.
#[test]
fn the_injected_rows_move_the_logits() {
    let (dir, arch) = install();
    let ids = prompt_ids();

    let mut a = open(&dir, &arch);
    with_image(&mut a, &arch, &ids, 0.0);
    let with_a = sequential(&mut a, &ids);

    let mut b = open(&dir, &arch);
    with_image(&mut b, &arch, &ids, 1.7);
    let with_b = sequential(&mut b, &ids);
    assert_ne!(with_a, with_b, "different rows must give different logits");

    let mut none = open(&dir, &arch);
    let without = sequential(&mut none, &ids);
    assert_ne!(
        with_a, without,
        "an injected prompt must differ from the lookup"
    );
}
