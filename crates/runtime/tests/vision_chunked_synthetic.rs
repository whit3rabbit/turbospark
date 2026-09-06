#![cfg(target_os = "macos")]
//! An IMAGE prompt through the dense qwen chunked prefill driver
//! (`families/qwen/prefill.rs`), against the sequential path.
//!
//! # Why this is a separate file from `real_forward_qwen35_chunked.rs`
//!
//! That file's prompt is 11 tokens and `MAX_PREFILL_BATCH` is 16, so no span
//! in its sweep can cross a micro-batch boundary -- the straddle hazard would
//! go untested there by construction. It also builds a tower-less install.
//! Both prompts here are 24 tokens, and one of them puts an image span across
//! position 16 on purpose.
//!
//! # The bar, and the two halves it has to separate
//!
//! Byte-identity against the SEQUENTIAL path, never against another chunked
//! arm, exactly as every other chunked test in this repo establishes.
//!
//! The change under test has two independent halves -- the BLIT (a tower row
//! replaces the table lookup at an image-pad position) and the ANGLE
//! (`RopePosition::Triple` from the position table) -- and breaking either one
//! alone still produces finite, plausible logits.
//!
//! **MEASURED: an ordinary image prompt cannot separate them.** On P_A,
//! deleting the blit and forcing `Sequential` redden exactly the same set of
//! cases, so a red run says "vision is broken" and not which half. The two
//! ISOLATING cases below exist for that reason and are the only ones that
//! attribute a failure: P_C's one-merged-token image has a degenerate table
//! (the angle mutation cannot reach it), and the shifted spans-free table has
//! no image row at all (the blit mutation cannot reach it). Each case's doc
//! comment records what was measured rather than what was predicted.
//!
//! # What it cannot see
//!
//! Whether the injected rows are the RIGHT rows. The fixture's weights are
//! untrained, so this is a WIRING proof -- right row, right offset, right
//! angle, right ordering -- and nothing more. `vision_tower_parity.rs` and the
//! real-install A/B are what measure the function.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use foundation::LogitValue as F16;
use turbospark_repack::{
    build_synthetic_qwen_gdn_dense_install_with_vision_streamed, tiny_vision_config,
};
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};
use turbospark_vision_io::{
    mrope_position_triples, GridThw, ImageSpan, MropePositions, PreprocessParams,
    PreprocessedImage, VisionSpecialIds,
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

const VOCAB: i64 = 256;
const LAYERS: i64 = 4;
const BITS: u32 = 1;
const INT4: u32 = 4;
const GRID_H: usize = 4;
const GRID_W: usize = 4;

/// Mirrors `real_forward_qwen35_chunked.rs`'s own constant so the two files
/// cannot drift on what a micro-batch is. Not importable: it is
/// `pub(crate)` in `real_forward_types.rs`.
const MICRO_BATCH: usize = 16;

/// In-vocab stand-ins, for `vision_inject_synthetic.rs`'s reason: this
/// fixture's vocabulary is 256 and the real placeholder id is 248,056, which
/// the driver refuses before the blit is reached. The ids are a PARAMETER of
/// `mrope_position_triples`, so picking in-vocab ones misrepresents nothing.
const VISION_START: i32 = 21;
const IMAGE_PAD: i32 = 22;
const VISION_END: i32 = 23;

/// Grid 1x4x4 at merge 2 gives `llm_h = llm_w = 2`, hence 4 merged tokens.
/// The block is 2x2 rather than 1xN so h and w BOTH vary inside it -- a 1xN
/// block leaves one axis constant and could not tell a selector that ignores
/// it from one that does not.
const MERGED: usize = 4;

/// P_A: the image sits at absolute `[4, 8)`, wholly inside micro-batch 0.
fn prompt_inside() -> Vec<i32> {
    let mut ids = vec![3, 7, 11, VISION_START];
    ids.extend(std::iter::repeat_n(IMAGE_PAD, MERGED));
    ids.push(VISION_END);
    ids.extend([13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]);
    assert_eq!(ids.len(), 24);
    ids
}

/// P_B: the image sits at absolute `[15, 19)` and STRADDLES the micro-batch
/// boundary at 16. Three of its four rows land in micro-batch 1, so a driver
/// that resolved rows or angles micro-batch-relatively rather than absolutely
/// diverges here and nowhere else.
fn prompt_straddling() -> Vec<i32> {
    let mut ids = vec![3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
    ids.push(VISION_START);
    ids.extend(std::iter::repeat_n(IMAGE_PAD, MERGED));
    ids.push(VISION_END);
    ids.extend([59, 61, 67, 71]);
    assert_eq!(ids.len(), 24);
    assert_eq!(ids[15], IMAGE_PAD, "the span must open just below 16");
    ids
}

/// P_C: a 2x2 grid at merge 2 gives exactly ONE merged token, so the walk's
/// triple for it is `(start, start, start)` -- degenerate -- and the block
/// advances the clock by exactly the one position it occupies, leaving
/// `rope_delta` at 0. Every angle in this prompt therefore equals what
/// `Sequential` would produce, while the blit still fires on its one image
/// row. That asymmetry is what isolates the two halves.
fn prompt_one_merged_token() -> Vec<i32> {
    let mut ids = vec![3, 7, 11, VISION_START, IMAGE_PAD, VISION_END];
    ids.extend([
        13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
    ]);
    assert_eq!(ids.len(), 24);
    ids
}

fn specials() -> VisionSpecialIds {
    VisionSpecialIds {
        vision_start: VISION_START,
        image_pad: IMAGE_PAD,
    }
}

fn temp_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-vision-chunked-{tag}-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn open_vision(tag: &str, bits: u32) -> RealForwardRunner {
    let dir = temp_dir(tag);
    build_synthetic_qwen_gdn_dense_install_with_vision_streamed(
        &dir,
        VOCAB,
        LAYERS,
        "qwen35-vision-chunked",
        bits,
    )
    .expect("streamed dense install with a vision tower");
    open(&dir)
}

fn open(dir: &Path) -> RealForwardRunner {
    let arch = turbospark_repack::peek_manifest_arch(dir).expect("peeks");
    RealForwardRunner::open(dir, arch).expect("opens")
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
        resized: (grid.h * params.patch_size, grid.w * params.patch_size),
        grid,
    }
}

fn positions_on(ids: &[i32], grid: GridThw) -> MropePositions {
    let p = params();
    mrope_position_triples(ids, &[grid], specials(), p.merge_size)
        .expect("the walk places the one image")
}

fn positions_for(ids: &[i32]) -> MropePositions {
    positions_on(ids, GridThw::new(1, GRID_H, GRID_W))
}

/// Encode the fixture image and hand the runner the map for `ids`.
fn inject(runner: &mut RealForwardRunner, ids: &[i32]) {
    inject_on(runner, ids, GridThw::new(1, GRID_H, GRID_W));
}

fn inject_on(runner: &mut RealForwardRunner, ids: &[i32], grid: GridThw) {
    let p = params();
    let embedding = runner
        .encode_image(&image_on(&p, grid), &p)
        .expect("the tower runs");
    runner
        .set_prompt_vision(&[embedding], &positions_on(ids, grid), ids.len())
        .expect("the map validates");
}

/// The reference: every prompt token through `produce_prefill` but the last,
/// which goes through `produce`, exactly as `run_raw_completion` does.
fn sequential_prefill(runner: &mut RealForwardRunner, tokens: &[i32]) -> Vec<u16> {
    runner.reset();
    let mut logits = vec![F16::from_f32(0.0); VOCAB as usize];
    let last = tokens.len() - 1;
    for (position, &token) in tokens.iter().enumerate() {
        if position == last {
            runner.produce(token, position, &mut logits)
        } else {
            runner.produce_prefill(token, position, &mut logits)
        }
        .expect("sequential prefill succeeds");
    }
    logits.into_iter().map(|v| v.to_bits()).collect()
}

/// The same prompt through `prefill_chunk`, split into spans of `chunk`.
fn chunked_prefill(runner: &mut RealForwardRunner, tokens: &[i32], chunk: usize) -> Vec<u16> {
    runner.reset();
    let mut logits = vec![F16::from_f32(0.0); VOCAB as usize];
    let mut offset = 0usize;
    while offset < tokens.len() {
        let take = (tokens.len() - offset).min(chunk);
        runner
            .prefill_chunk(&tokens[offset..offset + take], offset, &mut logits)
            .expect("chunked prefill succeeds");
        offset += take;
    }
    logits.into_iter().map(|v| v.to_bits()).collect()
}

/// Every span worth walking. `16` and `17` are the ones that put a
/// micro-batch split inside a chunk, `24` is one chunk holding two
/// micro-batches, and the small ones cut an image run at a chunk boundary.
const SPANS: [usize; 9] = [1, 2, 3, 4, 7, 11, 16, 17, 24];

// ---------------------------------------------------------------------------
// The fixture's own shape, pinned before anything reasons from it
// ---------------------------------------------------------------------------

/// A failure here is a change in the WALK, not in the driver, and reading it
/// as the latter costs a session. It also states the property the straddle
/// cases depend on: P_B's span really does cross 16.
#[test]
fn the_two_prompts_place_their_spans_where_the_cases_assume() {
    let inside = positions_for(&prompt_inside());
    assert_eq!(inside.spans, vec![ImageSpan { start: 4, len: 4 }]);
    assert!(
        inside.spans[0].start + inside.spans[0].len <= MICRO_BATCH,
        "P_A's image must sit wholly inside micro-batch 0"
    );

    let straddling = positions_for(&prompt_straddling());
    assert_eq!(straddling.spans, vec![ImageSpan { start: 15, len: 4 }]);
    let s = straddling.spans[0];
    assert!(
        s.start < MICRO_BATCH && s.start + s.len > MICRO_BATCH,
        "P_B's image must straddle the micro-batch boundary at {MICRO_BATCH}"
    );

    // Non-zero on both, which is what makes the ANGLE case discriminating: a
    // `rope_delta` of 0 would let `Sequential` and `Triple` agree at every
    // decode position and the mutation would survive.
    assert_eq!(inside.rope_delta, -2);
    assert_eq!(straddling.rope_delta, -2);
}

/// A vision install reports chunked support now. Until 2026-09-06 a live
/// `prompt_vision` map suppressed it (`real_forward_api.rs`).
///
/// MUTATION: restore the `&& self.prompt_vision.is_none()` conjunct. Reddens
/// this alone -- the driver's own guard is gone, so every prefill case below
/// still passes.
#[test]
fn a_vision_install_reports_chunked_prefill_support() {
    let mut runner = open_vision("supports", BITS);
    let ids = prompt_inside();
    inject(&mut runner, &ids);
    assert!(
        runner.supports_chunked_prefill(),
        "an image prompt must be servable by the chunked driver"
    );
}

// ---------------------------------------------------------------------------
// The two halves, separated
// ---------------------------------------------------------------------------

/// **THE BLIT, ISOLATED FROM THE ANGLE.** P_C's single merged token gives a
/// degenerate triple and a `rope_delta` of 0, so every angle in the prompt is
/// what `Sequential` would have produced -- and the mRoPE mutation therefore
/// cannot reach this case at all, while the blit still fires on the one image
/// row.
///
/// The two halves need a case each because the obvious end-to-end comparison
/// cannot separate them: on P_A, DELETING THE BLIT and FORCING `Sequential`
/// redden exactly the same four cases, so neither says which half broke.
/// Measured, not assumed.
///
/// MUTATION: delete the blit arm -> reddens. Force `RopePosition::Sequential`
/// -> stays GREEN, which is the whole point of the prompt's shape.
#[test]
fn a_single_merged_token_image_isolates_the_blit_from_the_angle() {
    let mut runner = open_vision("blit-isolated", BITS);
    let ids = prompt_one_merged_token();
    let grid = GridThw::new(1, 2, 2);

    // The premise the isolation rests on, asserted rather than assumed: if the
    // walk ever stopped producing a degenerate table here, this case would
    // quietly go back to testing both halves at once.
    let p = positions_on(&ids, grid);
    assert_eq!(p.spans, vec![ImageSpan { start: 4, len: 1 }]);
    assert_eq!(p.rope_delta, 0);
    assert!(
        p.triples.iter().all(|&(t, h, w)| t == h && h == w),
        "P_C's table must be degenerate everywhere, or this case tests both halves"
    );

    inject_on(&mut runner, &ids, grid);
    let reference = sequential_prefill(&mut runner, &ids);
    for chunk in SPANS {
        let got = chunked_prefill(&mut runner, &ids, chunk);
        assert_eq!(
            got, reference,
            "chunk {chunk} moved a one-token image's logits"
        );
    }
}

/// **THE ANGLE, ISOLATED FROM THE BLIT.** A hand-built map with NO spans and
/// no embeddings, whose triples are every position SHIFTED by a constant. The
/// blit arm is unreachable (`row_for` returns `None` everywhere, there being
/// no span to fall in), so the only thing that can differ from the sequential
/// path is the angle each position is rotated by.
///
/// A shift is a legal thing to hand `set_prompt_vision`: `position` keeps its
/// other two jobs either way (the KV slot index and the `position + 1`
/// attention span), so only the ANGLE moves. That is exactly the property
/// M-V5 relies on for a real image prompt's trailing text.
///
/// MUTATION: force `RopePosition::Sequential` -> reddens. Delete the blit arm
/// -> stays GREEN, since nothing here ever takes it.
#[test]
fn a_shifted_position_table_with_no_image_isolates_the_angle_from_the_blit() {
    let mut runner = open_vision("angle-isolated", BITS);
    let ids = prompt_inside();
    const SHIFT: i32 = 5;

    let shifted = MropePositions {
        triples: (0..ids.len() as i32)
            .map(|p| (p + SHIFT, p + SHIFT, p + SHIFT))
            .collect(),
        rope_delta: SHIFT,
        spans: Vec::new(),
    };
    runner
        .set_prompt_vision(&[], &shifted, ids.len())
        .expect("a spans-free shifted map validates");

    let reference = sequential_prefill(&mut runner, &ids);
    for chunk in SPANS {
        let got = chunked_prefill(&mut runner, &ids, chunk);
        assert_eq!(
            got, reference,
            "chunk {chunk} moved a shifted table's logits"
        );
    }
}

/// The BLIT half. One chunk, so no boundary is involved at all and the only
/// thing that can differ from sequential is whether the tower row replaced
/// the table lookup.
///
/// MUTATION: delete the `Some(row) => write_buffer_bytes` arm in `prefill.rs`
/// so every row takes `encode_embed_any`. MEASURED: reddens this and every
/// other image case in this file, and leaves all 11 cases in
/// `real_forward_qwen35_chunked.rs` green -- including that file's
/// `an_image_free_map_reproduces_the_sequential_logits`, which has no spans
/// and so never takes the blit arm.
///
/// It does not isolate the blit half (the angle mutation reddens the same
/// set); `a_single_merged_token_image_isolates_the_blit_from_the_angle` is
/// what does. This one's job is the simplest possible statement of the claim:
/// one chunk, no boundary, image in, same bytes out.
#[test]
fn an_image_prompt_chunks_byte_identically_to_the_sequential_path() {
    let mut runner = open_vision("blit", BITS);
    let ids = prompt_inside();
    inject(&mut runner, &ids);

    let reference = sequential_prefill(&mut runner, &ids);
    let got = chunked_prefill(&mut runner, &ids, ids.len());
    assert_eq!(
        got, reference,
        "one chunk over an image prompt moved the logits"
    );
}

/// The ANGLE half, and the ONLY case that isolates it. Runs the full sweep so
/// a chunk boundary cannot be what carries the difference.
///
/// MUTATION: `rope[t]` back to `RopePosition::Sequential` in
/// `prefill_layers.rs`. MEASURED: reddens every case in this file whose table
/// is non-degenerate -- this one, both boundary cases, the blit case above and
/// the shifted-table case -- and leaves green every text-only case in
/// `real_forward_qwen35_chunked.rs` (they are `Sequential` anyway) plus that
/// file's `an_image_free_map_reproduces_the_sequential_logits`, whose
/// degenerate table at `rope_delta` 0 makes `Triple(p, p, p)` and `Sequential`
/// the same function.
///
/// So this case does NOT isolate the angle on its own; the shifted-table case
/// below is what does. This one's job is the SWEEP -- that no chunk size
/// changes the angle a position gets.
#[test]
fn the_rope_angle_follows_the_position_table_through_the_chunked_driver() {
    let mut runner = open_vision("angle", BITS);
    let ids = prompt_inside();
    inject(&mut runner, &ids);

    let reference = sequential_prefill(&mut runner, &ids);
    for chunk in SPANS {
        let got = chunked_prefill(&mut runner, &ids, chunk);
        assert_eq!(
            got, reference,
            "chunk {chunk} moved an image prompt's logits"
        );
    }
}

// ---------------------------------------------------------------------------
// Boundaries: chunk, and micro-batch
// ---------------------------------------------------------------------------

/// Chunk-span invariance on the prompt whose image sits inside micro-batch 0.
///
/// MUTATION: `pv.row_for(start_position + t)` to `pv.row_for(t)`. Reddens
/// this, and also every other image case in this file including the
/// single-chunk one.
///
/// **THAT LAST CLAUSE REFUTES THE OBVIOUS PREDICTION AND IS THE USEFUL PART.**
/// "One chunk starting at 0, so absolute and relative agree" is wrong: a chunk
/// is split into micro-batches of `MICRO_BATCH` INSIDE the driver, so
/// `start_position` advances to 16 for the second micro-batch of a 24-token
/// prompt even when the caller passed one chunk. Under the mutation, rows
/// 16..24 then resolve against positions 0..8, which is where P_A's image
/// lives -- so the tail text gets blitted with tower rows. A test that only
/// varied the CHUNK size would be blind to this; what reaches it is the prompt
/// being longer than a micro-batch.
///
/// The micro-batch-relative variant (`start_position % MICRO_BATCH + t`)
/// reddens the same set, for the same reason. No mutation found so far
/// isolates a boundary case.
#[test]
fn the_chunk_boundary_does_not_move_an_image_prompts_logits() {
    let mut runner = open_vision("chunk-boundary", BITS);
    let ids = prompt_inside();
    inject(&mut runner, &ids);

    let reference = sequential_prefill(&mut runner, &ids);
    for chunk in SPANS {
        let got = chunked_prefill(&mut runner, &ids, chunk);
        assert_eq!(got, reference, "chunk {chunk} moved the logits");
    }
}

/// The MICRO-BATCH boundary, which no chunk size can avoid: at chunk 24 the
/// driver still splits into micro-batches of 16, and P_B's image span crosses
/// that split with three of its four rows on the far side.
///
/// **NO MUTATION FOUND SO FAR REDDENS THIS CASE ALONE**, and that is recorded
/// rather than tuned away. Every position-plumbing mutation tried (relative,
/// micro-batch-relative, on the row or on the angle) reddens P_A's cases too,
/// because both prompts are 24 tokens and therefore both have a second
/// micro-batch where the relative/absolute distinction already bites -- see
/// the case above for the mechanism.
///
/// It is kept because it is the only case here whose image span actually
/// CROSSES the boundary, so it covers a geometry nothing else does: three of
/// four tower rows blitted from one micro-batch's loop, one from the previous.
/// A future change that special-cased the first row of a micro-batch would be
/// caught here and nowhere else.
#[test]
fn an_image_span_that_straddles_a_micro_batch_boundary_reproduces_sequential() {
    let mut runner = open_vision("straddle", BITS);
    let ids = prompt_straddling();
    inject(&mut runner, &ids);

    let reference = sequential_prefill(&mut runner, &ids);
    for chunk in SPANS {
        let got = chunked_prefill(&mut runner, &ids, chunk);
        assert_eq!(
            got, reference,
            "chunk {chunk} moved a straddling image prompt's logits"
        );
    }
}

// ---------------------------------------------------------------------------
// The refusal that stays, and the lifetime rule
// ---------------------------------------------------------------------------

/// `TURBOSPARK_BATCHED_GEMV` plus an image prompt is refused by name. The
/// refusal is about the ANGLE: `encode_full_attention_block_batched` rotates
/// at the raw position and takes no `RopePosition`, so the blit would land
/// correctly and every image position would then be rotated by its index.
///
/// Needs INT4, because that arm's `encode_gemm_any` has no kernel at one bit
/// and would refuse on the WIDTH first, testing nothing about vision.
///
/// MUTATION: delete the refusal. PREDICTED result, recorded so nobody reads a
/// green run as evidence the arm works: this case reddens and nothing else
/// does, because no other case sets the seam. The arm would then RUN, the
/// blit would land, and the logits would diverge from sequential at every
/// position whose rope triple is not its index.
#[test]
fn the_batched_gemv_arm_refuses_an_image_prompt_by_name() {
    let mut runner = open_vision("batched-refusal", INT4);
    let ids = prompt_inside();
    inject(&mut runner, &ids);
    runner.set_batched_gemv_prefill(true);

    let mut logits = vec![F16::from_f32(0.0); VOCAB as usize];
    let Err(err) = runner.prefill_chunk(&ids, 0, &mut logits) else {
        panic!("the batched arm must refuse an image prompt");
    };
    let text = err.to_string();
    assert!(
        text.contains("TURBOSPARK_BATCHED_GEMV") && text.contains("angle"),
        "the refusal must name the seam and the reason; got {text}"
    );
}

/// The injection map survives the CHUNKED generation loop's own `reset()`,
/// which is the path a front end takes on this family now.
///
/// `vision_inject_synthetic.rs` pins the same contract through
/// `run_raw_completion`. It needs a chunked sibling for exactly the reason
/// M-V5's bug shipped (Gotcha 29): the nine cases that preceded that guard
/// all drove `produce` directly, and the loop that calls `reset()` was the
/// one nobody tested. `run_raw_completion_chunked` is now a second such loop.
///
/// MUTATION: restore `prompt_vision = None` inside `reset()`. Reddens this
/// and the sequential guard next door, which is expected and correct -- the
/// point is that this path is covered at all.
#[test]
fn an_injected_map_survives_the_chunked_prefill_drivers_reset() {
    let mut runner = open_vision("survives-reset", BITS);
    let ids = prompt_inside();
    inject(&mut runner, &ids);

    // `chunked_prefill` calls `reset()` first, exactly as the generation loop
    // does before its first chunk. If the map were cleared there, this would
    // reproduce a run with no injection at all.
    let with_map = chunked_prefill(&mut runner, &ids, 8);

    runner.clear_prompt_vision();
    let without_map = chunked_prefill(&mut runner, &ids, 8);

    assert_ne!(
        with_map, without_map,
        "the injection must survive reset(): a cleared map would make these equal, \
         which is M-V5's shipped bug reached through the chunked loop"
    );
}
