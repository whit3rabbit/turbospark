//! Image injection on a synthetic `qwen4_exp` install (stage 3 of
//! `docs/VISION.md`'s four gates, for this family).
//!
//! The fixture is the decode-capable Qwen4Exp install with the toy tower
//! attached at the trunk's width and its matrices quantized to MLX affine
//! 4 bits, the way the real conversion ships them
//! (`docs/QWEN4EXP_VISION_PHASE0.md` item 1). So every case here also runs
//! the repack's dequantizing tower reader.
//!
//! It measures WIRING: that the tower row reaches the residual at an image
//! position in both drivers, that the mRoPE angle reaches the QSA layers,
//! that the two drivers agree byte for byte, and that the refusals fire. The
//! weights are untrained, so nothing here says the injected rows are the
//! RIGHT rows. That is stage 2 and stage 4, which need a real install.
//!
//! The special ids are in-vocab stand-ins for the real 248053/248056/248054,
//! as in `vision_inject_synthetic.rs`: the ids are a parameter of
//! `mrope_position_triples`, and a real placeholder is an ordinary vocab
//! member.
#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use turbospark_repack::{
    build_synthetic_qwen4_exp_decode_install, build_synthetic_qwen4_exp_decode_install_with_vision,
    tiny_vision_config,
};
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};
use turbospark_vision_io::{
    mrope_position_triples, GridThw, ImageSpan, MropePositions, PreprocessParams,
    PreprocessedImage, VisionSpecialIds,
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Must exceed the fixture's n-gram EOS fill (999), as in
/// `real_forward_qwen4_chunked.rs`.
const VOCAB: i64 = 2048;
const MAX_CONTEXT: usize = 512;
/// The real checkpoint's QSA budget. At `MAX_CONTEXT` 512 selection can never
/// run (128 blocks against `index_top_k` 512), so image prompts are admitted.
const REAL_BUDGET: i64 = 2048;
/// A budget `MAX_CONTEXT` crosses, for the refusal.
const TINY_BUDGET: i64 = 16;
const MODEL_ID: &str = "qwen4-vision";

const VISION_START: i32 = 21;
const IMAGE_PAD: i32 = 22;
const VISION_END: i32 = 23;

/// Chunk spans for the driver comparison: one token, the micro-batch edges,
/// and a whole prompt.
const SPANS: [usize; 6] = [1, 2, 3, 5, 16, 64];

fn specials() -> VisionSpecialIds {
    VisionSpecialIds {
        vision_start: VISION_START,
        image_pad: IMAGE_PAD,
    }
}

/// Three text tokens, a 2x2 merged image, three more. The 2x2 block varies
/// both h and w.
fn prompt_inside() -> Vec<i32> {
    let mut ids = vec![301, 457, 613, VISION_START];
    ids.extend([IMAGE_PAD; 4]);
    ids.extend([VISION_END, 719, 829, 941]);
    ids
}

/// The image block starts at 14, so its four rows straddle the 16-row
/// micro-batch boundary.
fn prompt_straddling() -> Vec<i32> {
    let mut ids: Vec<i32> = (0..13).map(|p| 100 + 37 * p).collect();
    ids.push(VISION_START);
    ids.extend([IMAGE_PAD; 4]);
    ids.push(VISION_END);
    ids.extend([1201, 1303, 1409]);
    ids
}

fn temp_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-qwen4-vision-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn build(tag: &str, budget: i64, streamed: bool) -> PathBuf {
    let dir = temp_dir(tag);
    build_synthetic_qwen4_exp_decode_install_with_vision(&dir, VOCAB, MODEL_ID, budget, streamed)
        .expect("qwen4_exp install with a quantized tower");
    dir
}

fn open(dir: &Path) -> RealForwardRunner {
    let arch = turbospark_repack::peek_manifest_arch(dir).expect("peeks");
    RealForwardRunner::open_with_max_context(dir, arch, MAX_CONTEXT).expect("opens")
}

/// Removes an install directory on drop. The tower opens LAZILY from the
/// install on the first image, so the directory must outlive the runner's
/// first `encode_image`, not just its open.
struct Cleanup(PathBuf);

impl Drop for Cleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn open_vision(tag: &str) -> (RealForwardRunner, Cleanup) {
    let dir = build(tag, REAL_BUDGET, true);
    let runner = open(&dir);
    (runner, Cleanup(dir))
}

fn params() -> PreprocessParams {
    let v = tiny_vision_config();
    PreprocessParams {
        patch_size: v.patch_size as usize,
        temporal_patch_size: v.temporal_patch_size as usize,
        merge_size: v.spatial_merge_size as usize,
        in_channels: v.in_channels as usize,
        min_pixels: 1,
        max_pixels: 1 << 20,
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
        rescale_factor: 1.0 / 255.0,
    }
}

fn image_on(params: &PreprocessParams, grid: GridThw) -> PreprocessedImage {
    let dim = params.patch_dim();
    let patch_rows: Vec<f32> = (0..grid.patches() * dim)
        .map(|i| ((i as f32) * 0.0173).sin() * 0.8)
        .collect();
    PreprocessedImage {
        merged_tokens: grid.merged_tokens(params.merge_size),
        patch_rows,
        grid,
        resized: (grid.h * params.patch_size, grid.w * params.patch_size),
    }
}

fn grid() -> GridThw {
    GridThw::new(1, 4, 4)
}

fn positions_for(ids: &[i32]) -> MropePositions {
    mrope_position_triples(ids, &[grid()], specials(), params().merge_size)
        .expect("the walk places the one image")
}

fn embed(runner: &mut RealForwardRunner) -> turbospark_runtime::vision::VisionEmbedding {
    let params = params();
    runner
        .encode_image(&image_on(&params, grid()), &params)
        .expect("the tower runs")
}

fn inject(runner: &mut RealForwardRunner, ids: &[i32]) {
    let embedding = embed(runner);
    runner
        .set_prompt_vision(&[embedding], &positions_for(ids), ids.len())
        .expect("the map validates");
}

/// Every prompt token through `produce_prefill` but the last, which goes
/// through `produce`, as `run_raw_completion` does.
fn sequential(runner: &mut RealForwardRunner, ids: &[i32]) -> Vec<u16> {
    runner.reset();
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let last = ids.len() - 1;
    for (position, &token) in ids.iter().enumerate() {
        if position == last {
            runner.produce(token, position, &mut logits)
        } else {
            runner.produce_prefill(token, position, &mut logits)
        }
        .expect("sequential prefill succeeds");
    }
    logits.iter().map(|v| v.to_bits()).collect()
}

fn chunked(runner: &mut RealForwardRunner, ids: &[i32], chunk: usize) -> Vec<u16> {
    runner.reset();
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let mut offset = 0usize;
    while offset < ids.len() {
        let take = (ids.len() - offset).min(chunk);
        runner
            .prefill_chunk(&ids[offset..offset + take], offset, &mut logits)
            .expect("chunked prefill succeeds");
        offset += take;
    }
    logits.iter().map(|v| v.to_bits()).collect()
}

// ---------------------------------------------------------------------------
// Ingest
// ---------------------------------------------------------------------------

/// Both writers carry the quantized tower, declare it, and write the same
/// packed blocks. An arm in one writer only is the shape of the MTP head's
/// shipped HEADLESS install.
#[test]
fn both_writers_carry_the_quantized_tower() {
    let streamed = build("writer-streamed", REAL_BUDGET, true);
    let plain = build("writer-plain", REAL_BUDGET, false);
    for dir in [&streamed, &plain] {
        let arch = turbospark_repack::peek_manifest_arch(dir).expect("peeks");
        assert!(arch.vision.is_active(), "{} lost its tower", dir.display());
        assert_eq!(arch.vision.out_hidden_size, arch.hidden_size);
    }
    let blobs = |d: &Path| std::fs::read(d.join("packed_vision/blobs.bin")).expect("blobs");
    let same = blobs(&streamed) == blobs(&plain);
    let _ = std::fs::remove_dir_all(&streamed);
    let _ = std::fs::remove_dir_all(&plain);
    assert!(same, "the two writers packed different tower blocks");
}

/// The tower opens on the dequantized install and returns one trunk-width
/// row per merged token.
#[test]
fn the_tower_returns_trunk_width_rows() {
    let (mut runner, _dir) = open_vision("tower-rows");
    let embedding = embed(&mut runner);
    let hidden = turbospark_repack::QWEN4_DECODE_HIDDEN;
    assert_eq!(embedding.out_hidden, hidden);
    assert_eq!(embedding.rows.len(), 4 * hidden);
    assert!(
        embedding
            .rows
            .iter()
            .all(|&b| f16::from_bits(b).is_finite()),
        "the dequantized tower produced a non-finite row"
    );
}

// ---------------------------------------------------------------------------
// The text path is untouched
// ---------------------------------------------------------------------------

/// A vision install handed no image decodes exactly as a separately built
/// install with no tower: the tower's presence moves no trunk byte.
#[test]
fn a_vision_install_with_no_image_decodes_as_one_without_a_tower() {
    let ids = prompt_inside();
    let (mut with_tower, _dir) = open_vision("no-image");
    let without_dir = temp_dir("no-tower");
    let arch = build_synthetic_qwen4_exp_decode_install(&without_dir, VOCAB, MODEL_ID)
        .expect("text-only install");
    let mut without =
        RealForwardRunner::open_with_max_context(&without_dir, arch, MAX_CONTEXT).expect("opens");
    let _ = std::fs::remove_dir_all(&without_dir);
    assert_eq!(
        sequential(&mut with_tower, &ids),
        sequential(&mut without, &ids),
        "the tower's presence moved the text path"
    );
}

/// A degenerate table (every triple `t == h == w`) is byte-identical to no
/// table: the dispatch condition keeps text on the scalar rope kernel.
#[test]
fn a_degenerate_position_table_is_byte_identical_to_no_table() {
    let ids = prompt_inside();
    let (mut runner, _dir) = open_vision("degenerate");
    let baseline = sequential(&mut runner, &ids);
    let degenerate = MropePositions {
        triples: (0..ids.len() as i32).map(|p| (p, p, p)).collect(),
        rope_delta: 0,
        spans: Vec::new(),
    };
    runner
        .set_prompt_vision(&[], &degenerate, ids.len())
        .expect("an image-free map validates");
    assert_eq!(baseline, sequential(&mut runner, &ids));
}

/// Proves the case above can fail: one diverged triple moves the logits, so
/// the mRoPE arm is reachable through the QSA layers.
///
/// MUTATION: pass `RopePosition::Sequential` at the produce call site ->
/// reddens.
#[test]
fn one_diverged_triple_reaches_the_qsa_rope() {
    let ids = prompt_inside();
    let (mut runner, _dir) = open_vision("diverged");
    let baseline = sequential(&mut runner, &ids);
    let mut triples: Vec<(i32, i32, i32)> = (0..ids.len() as i32).map(|p| (p, p, p)).collect();
    triples[6] = (6, 5, 4);
    let diverged = MropePositions {
        triples,
        rope_delta: 0,
        spans: Vec::new(),
    };
    runner
        .set_prompt_vision(&[], &diverged, ids.len())
        .expect("validates");
    assert_ne!(
        baseline,
        sequential(&mut runner, &ids),
        "a diverged triple changed nothing; the mRoPE arm is unreachable"
    );
}

// ---------------------------------------------------------------------------
// The blit
// ---------------------------------------------------------------------------

/// The tower rows reach the logits: perturbing ONE injected row moves them.
/// One row rather than all, so a flow that blits row 0 everywhere still
/// fails.
///
/// MUTATION: delete the `Some(row)` arm in `produce.rs` -> reddens.
#[test]
fn perturbing_one_injected_row_moves_the_logits() {
    let ids = prompt_inside();
    let (mut runner, _dir) = open_vision("rows-reach");
    let embedding = embed(&mut runner);
    let positions = positions_for(&ids);
    runner
        .set_prompt_vision(std::slice::from_ref(&embedding), &positions, ids.len())
        .expect("validates");
    let before = sequential(&mut runner, &ids);

    let mut perturbed = embedding.clone();
    let row = 2 * perturbed.out_hidden;
    for v in &mut perturbed.rows[row..row + perturbed.out_hidden] {
        *v = f16::from_f32(f16::from_bits(*v).to_f32() + 0.25).to_bits();
    }
    runner
        .set_prompt_vision(&[perturbed], &positions, ids.len())
        .expect("validates");
    assert_ne!(
        before,
        sequential(&mut runner, &ids),
        "the injected rows do not reach the logits"
    );
}

/// PLE still hashes the token id at an image position
/// (`docs/QWEN4EXP_VISION_PHASE0.md` item 4): both references feed the raw
/// placeholder ids to the n-gram head. So unlike the Qwen3.8 flow, changing
/// the id under an image row MUST move the logits here. A change that fed PLE
/// something else at image positions, or skipped it there, would hold them
/// still.
#[test]
fn ple_still_reads_the_placeholder_id_under_an_image_row() {
    let ids = prompt_inside();
    let (mut runner, _dir) = open_vision("ple-ids");
    inject(&mut runner, &ids);
    let baseline = sequential(&mut runner, &ids);

    // Position 5 is inside the placeholder run. The map keys on position, so
    // the same row is still blitted there.
    let mut swapped = ids.clone();
    swapped[5] = 99;
    assert_ne!(
        baseline,
        sequential(&mut runner, &swapped),
        "the n-gram head ignored the id under an image row"
    );
}

// ---------------------------------------------------------------------------
// The two drivers agree
// ---------------------------------------------------------------------------

/// An image prompt chunks byte-identically to the sequential path, across
/// spans that put boundaries before, inside and after the image.
///
/// MUTATION: delete the `Some(row)` arm in `prefill.rs` -> reddens. Pass
/// `RopePosition::Sequential` at the chunked attention call -> reddens.
#[test]
fn an_image_prompt_chunks_byte_identically_to_sequential() {
    for ids in [prompt_inside(), prompt_straddling()] {
        let (mut runner, _dir) = open_vision("chunked");
        inject(&mut runner, &ids);
        let reference = sequential(&mut runner, &ids);
        for chunk in SPANS {
            assert_eq!(
                chunked(&mut runner, &ids, chunk),
                reference,
                "chunk {chunk} moved an image prompt's logits ({} tokens)",
                ids.len()
            );
        }
    }
}

/// The straddling prompt places its image where that case assumes.
#[test]
fn the_straddling_prompt_crosses_the_micro_batch_boundary() {
    let p = positions_for(&prompt_straddling());
    assert_eq!(p.spans, vec![ImageSpan { start: 14, len: 4 }]);
}

/// THE ANGLE, ISOLATED. A spans-free table shifted by a constant: no row can
/// blit, so only the rotation can differ between drivers.
#[test]
fn a_shifted_table_with_no_image_chunks_byte_identically() {
    let ids = prompt_inside();
    let (mut runner, _dir) = open_vision("shifted");
    let shifted = MropePositions {
        triples: (0..ids.len() as i32)
            .map(|p| (p + 5, p + 5, p + 5))
            .collect(),
        rope_delta: 5,
        spans: Vec::new(),
    };
    runner
        .set_prompt_vision(&[], &shifted, ids.len())
        .expect("validates");
    let reference = sequential(&mut runner, &ids);
    for chunk in SPANS {
        assert_eq!(
            chunked(&mut runner, &ids, chunk),
            reference,
            "chunk {chunk}"
        );
    }
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

/// Where QSA selection could run, an image prompt is refused at
/// `set_prompt_vision` and leaves no map behind; the dense diagnostic admits
/// it; and turning selection back on after the map is set is refused at the
/// next pass.
#[test]
fn an_image_prompt_past_the_qsa_budget_is_refused() {
    let dir = build("past-budget", TINY_BUDGET, true);
    let mut runner = open(&dir);
    let _dir = Cleanup(dir);
    let ids = prompt_inside();
    let embedding = embed(&mut runner);
    let positions = positions_for(&ids);

    let err = runner
        .set_prompt_vision(std::slice::from_ref(&embedding), &positions, ids.len())
        .expect_err("a context past the budget must refuse an image");
    assert!(err.to_string().contains("QSA budget"), "{err}");
    assert!(runner.prompt_vision().is_none(), "a refused map stayed set");

    runner.set_qsa_force_dense(true);
    runner
        .set_prompt_vision(std::slice::from_ref(&embedding), &positions, ids.len())
        .expect("the dense diagnostic never selects, so the image is exact");
    runner.set_qsa_force_dense(false);
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let err = runner
        .produce(ids[0], 0, &mut logits)
        .expect_err("selection turned back on must refuse at the pass");
    assert!(err.to_string().contains("QSA budget"), "{err}");
    let err = runner
        .prefill_chunk(&ids, 0, &mut logits)
        .expect_err("and at the chunked driver");
    assert!(err.contains("QSA budget"), "{err}");
}

/// The batched-GEMV seam takes no per-row rope, so it refuses an image.
#[test]
fn the_batched_gemv_arm_refuses_an_image_prompt() {
    let (mut runner, _dir) = open_vision("batched");
    let ids = prompt_inside();
    inject(&mut runner, &ids);
    runner.set_batched_gemv_prefill(true);
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let err = runner
        .prefill_chunk(&ids, 0, &mut logits)
        .expect_err("the batched arm must refuse an image prompt");
    assert!(err.contains("TURBOSPARK_BATCHED_GEMV"), "{err}");
}
