//! Image injection and the trunk's mRoPE dispatch on a synthetic install
//! (ROADMAP M-V5, stage 1).
//!
//! # What it can and cannot see
//!
//! It CAN see the two things M-V5 actually wires: that a tower row replaces
//! the embedding lookup at an image-pad position and nothing else, and that
//! the rope dispatch condition holds -- in particular that a degenerate
//! position table changes NOTHING, which is the invariant the real quality
//! gate then confirms at scale.
//!
//! It CANNOT see whether the injected rows are the RIGHT rows. The fixture's
//! weights are untrained, so this file measures wiring; the cross-engine arm
//! against mlx-vlm is what measures the function.
//!
//! # Why the special token ids here are not the checkpoint's
//!
//! `tiny_vision_config` carries the real `image_token_id` (248,056) because it
//! is metadata rather than a shape. This fixture's vocabulary is 256, and
//! `produce` refuses a token id outside the vocab BEFORE it reaches the blit,
//! so a prompt spelled with the real ids could not run here. The ids are a
//! PARAMETER of `mrope_position_triples`, which is what lets the test pick
//! in-vocab ones without misrepresenting anything: on a real install the
//! placeholder is an ordinary vocabulary member and the check passes.
#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use foundation::LogitValue as F16;
use turbospark_repack::{
    build_synthetic_qwen_gdn_dense_install_at_bits,
    build_synthetic_qwen_gdn_dense_install_with_vision_streamed, tiny_vision_config,
};
use turbospark_runtime::{LogitProducer, RealForwardRunner};
use turbospark_vision_io::{
    mrope_position_triples, GridThw, ImageSpan, MropePositions, PreprocessParams,
    PreprocessedImage, VisionSpecialIds,
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

const VOCAB: i64 = 256;
const LAYERS: i64 = 4;
const BITS: u32 = 1;
const GRID_H: usize = 4;
const GRID_W: usize = 4;

/// In-vocab stand-ins; see the module header.
const VISION_START: i32 = 21;
const IMAGE_PAD: i32 = 22;
const VISION_END: i32 = 23;

/// The prompt this file walks: three text tokens, the marker, four
/// placeholders, the closing marker, three more text tokens.
///
/// Four placeholders because grid 1x4x4 at merge 2 gives `llm_h = llm_w = 2`,
/// hence `1 * 2 * 2` merged tokens. The block is 2x2 rather than 1xN so that
/// h and w BOTH vary inside it -- a 1xN block leaves one axis constant and
/// could not tell a selector that ignores it from one that does not.
fn prompt_ids() -> Vec<i32> {
    vec![
        3,
        7,
        11,
        VISION_START,
        IMAGE_PAD,
        IMAGE_PAD,
        IMAGE_PAD,
        IMAGE_PAD,
        VISION_END,
        13,
        17,
        19,
    ]
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
        "turbospark-vision-mv5-{tag}-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn build_vision(tag: &str) -> PathBuf {
    let dir = temp_dir(tag);
    build_synthetic_qwen_gdn_dense_install_with_vision_streamed(
        &dir,
        VOCAB,
        LAYERS,
        "qwen35-vision-mv5",
        BITS,
    )
    .expect("streamed dense install with a vision tower");
    dir
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

fn image(params: &PreprocessParams) -> PreprocessedImage {
    let grid = GridThw::new(1, GRID_H, GRID_W);
    let dim = params.patch_dim();
    let patch_rows: Vec<f32> = (0..grid.patches() * dim)
        .map(|i| ((i as f32) * 0.0173).sin() * 0.8)
        .collect();
    PreprocessedImage {
        merged_tokens: grid.merged_tokens(params.merge_size),
        patch_rows,
        grid,
        resized: (GRID_H * params.patch_size, GRID_W * params.patch_size),
    }
}

fn positions_for(ids: &[i32]) -> MropePositions {
    let params = params();
    mrope_position_triples(
        ids,
        &[GridThw::new(1, GRID_H, GRID_W)],
        specials(),
        params.merge_size,
    )
    .expect("the walk places the one image")
}

/// Walk a whole prompt and return the LAST position's logits.
///
/// The last position, because everything before it is what the attention span
/// and the position table have to have got right for it to be meaningful.
fn walk(runner: &mut RealForwardRunner, ids: &[i32]) -> Vec<u16> {
    let mut head = vec![F16::from_f32(0.0); VOCAB as usize];
    for (position, &token) in ids.iter().enumerate() {
        runner
            .produce(token, position, &mut head)
            .expect("produces");
    }
    head.into_iter().map(|v| v.to_bits()).collect()
}

fn digest(logits: &[u16]) -> String {
    let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
    model_io::hash_data(&bytes)[..8].to_string()
}

/// Encode the fixture image and hand the runner the map for `ids`.
fn inject(runner: &mut RealForwardRunner, ids: &[i32]) -> MropePositions {
    let params = params();
    let embedding = runner
        .encode_image(&image(&params), &params)
        .expect("the tower runs");
    let positions = positions_for(ids);
    runner
        .set_prompt_vision(&[embedding], &positions, ids.len())
        .expect("the map validates");
    positions
}

// ---------------------------------------------------------------------------
// The walk agrees with what M-V5 assumes about it
// ---------------------------------------------------------------------------

/// Pins the shape the rest of the file reasons from, so a change in the walk
/// shows up here rather than as a confusing failure three cases down.
///
/// Note position 4 -- the block's FIRST placeholder -- is `(4, 4, 4)`:
/// degenerate, and therefore routed to the pre-existing kernel even though it
/// is an image token. That is the dispatch rule being keyed on the DATA rather
/// than on a token classification, visible in the table itself.
#[test]
fn the_position_table_compresses_the_image_and_shifts_the_text_after_it() {
    let ids = prompt_ids();
    let p = positions_for(&ids);

    assert_eq!(p.spans, vec![ImageSpan { start: 4, len: 4 }]);
    assert_eq!(
        p.triples,
        vec![
            (0, 0, 0),
            (1, 1, 1),
            (2, 2, 2),
            (3, 3, 3),
            // The 2x2 merged grid, t then h then w.
            (4, 4, 4),
            (4, 4, 5),
            (4, 5, 4),
            (4, 5, 5),
            // The block advanced the clock by max(t, h, w) = 2, not by 4.
            (6, 6, 6),
            (7, 7, 7),
            (8, 8, 8),
            (9, 9, 9),
        ]
    );
    // Twelve tokens spending ten positions.
    assert_eq!(p.rope_delta, -2);
}

// ---------------------------------------------------------------------------
// The degenerate-equivalence invariant
// ---------------------------------------------------------------------------

/// **The invariant M-V5 is judged on.** A vision install that is never handed
/// an image must decode exactly as an install with no tower at all.
///
/// Compared against a SEPARATELY BUILT non-vision install rather than against
/// a frozen constant, because the two builders seed from the same `model_id`
/// and a constant could be re-frozen. This says the tower's presence changes
/// no trunk byte, which is the claim.
#[test]
fn a_vision_install_with_no_image_decodes_as_one_without_a_tower() {
    let with_tower = build_vision("no-image");
    let without = temp_dir("no-tower");
    build_synthetic_qwen_gdn_dense_install_at_bits(
        &without,
        VOCAB,
        LAYERS,
        "qwen35-vision-mv5",
        BITS,
    )
    .expect("dense install with no tower");

    let ids = prompt_ids();
    let a = walk(&mut open(&with_tower), &ids);
    let b = walk(&mut open(&without), &ids);

    let _ = std::fs::remove_dir_all(&with_tower);
    let _ = std::fs::remove_dir_all(&without);
    assert_eq!(a, b, "the tower's presence moved the text path");
}

/// The same invariant one level in: a position table whose every triple is
/// DEGENERATE must produce the identical bytes, even though it takes the
/// `RopePosition::Triple` arm rather than `Sequential`.
///
/// This is what pins the dispatch condition itself. `Sequential` and
/// `Triple(p, p, p)` at the same `p` reach different match arms and must reach
/// the same kernel with the same argument; anything that made the mRoPE kernel
/// fire on a degenerate triple would redden here and nowhere else in this
/// file, because every other case has a real image in it.
#[test]
fn a_degenerate_position_table_is_byte_identical_to_no_table_at_all() {
    let dir = build_vision("degenerate");
    let ids = prompt_ids();

    let baseline = walk(&mut open(&dir), &ids);

    let mut runner = open(&dir);
    let degenerate = MropePositions {
        triples: (0..ids.len() as i32).map(|p| (p, p, p)).collect(),
        rope_delta: 0,
        spans: Vec::new(),
    };
    runner
        .set_prompt_vision(&[], &degenerate, ids.len())
        .expect("an image-free map validates");
    let injected = walk(&mut runner, &ids);

    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(
        baseline, injected,
        "a degenerate triple table changed the trunk's output"
    );
}

/// Proves the case above can fail, so its equality is evidence rather than a
/// property of a flow that ignores its rope positions entirely.
#[test]
fn the_degenerate_table_case_discriminates() {
    let dir = build_vision("degenerate-discriminates");
    let ids = prompt_ids();

    let baseline = walk(&mut open(&dir), &ids);

    let mut runner = open(&dir);
    // One position's h and w pulled off t. Nothing else differs.
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
    let injected = walk(&mut runner, &ids);

    let _ = std::fs::remove_dir_all(&dir);
    assert_ne!(
        baseline, injected,
        "diverging one position's triple changed nothing; the mRoPE arm is unreachable"
    );
}

// ---------------------------------------------------------------------------
// The blit
// ---------------------------------------------------------------------------

/// The tower's rows have to REACH the logits, or every other case here is
/// measuring an injection that does not happen.
#[test]
fn perturbing_the_injected_rows_moves_the_logits() {
    let dir = build_vision("rows-reach");
    let ids = prompt_ids();
    let params = params();

    let mut runner = open(&dir);
    let embedding = runner
        .encode_image(&image(&params), &params)
        .expect("the tower runs");
    let positions = positions_for(&ids);

    runner
        .set_prompt_vision(std::slice::from_ref(&embedding), &positions, ids.len())
        .expect("validates");
    let before = walk(&mut runner, &ids);

    // A single row, not the whole tensor: the whole-tensor version would also
    // pass against an implementation that blits row 0 everywhere.
    let mut perturbed = embedding.clone();
    let row = 2 * perturbed.out_hidden;
    for v in &mut perturbed.rows[row..row + perturbed.out_hidden] {
        *v = F16::from_f32(F16::from_bits(*v).to_f32() + 0.25).to_bits();
    }
    let mut runner = open(&dir);
    runner
        .set_prompt_vision(&[perturbed], &positions, ids.len())
        .expect("validates");
    let after = walk(&mut runner, &ids);

    let _ = std::fs::remove_dir_all(&dir);
    assert_ne!(before, after, "the injected rows do not reach the logits");
}

/// The blit REPLACES the lookup rather than riding beside it. Change the token
/// id sitting at an image position and nothing may move; change one at a text
/// position and something must.
///
/// The pair is the point. The first half alone passes against a flow that
/// ignores token ids entirely, and the second alone passes against one that
/// never blits.
#[test]
fn a_span_position_ignores_its_token_id_and_a_text_position_does_not() {
    let dir = build_vision("replaces-lookup");
    let ids = prompt_ids();

    let mut runner = open(&dir);
    inject(&mut runner, &ids);
    let baseline = walk(&mut runner, &ids);

    // Position 5 is inside the placeholder run.
    let mut swapped_image = ids.clone();
    swapped_image[5] = 99;
    let mut runner = open(&dir);
    inject(&mut runner, &ids);
    let image_swapped = walk(&mut runner, &swapped_image);

    // Position 2 is ordinary text.
    let mut swapped_text = ids.clone();
    swapped_text[2] = 99;
    let mut runner = open(&dir);
    inject(&mut runner, &ids);
    let text_swapped = walk(&mut runner, &swapped_text);

    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(
        baseline, image_swapped,
        "the token id at an image position reached the residual stream; the blit is \
         not replacing the lookup"
    );
    assert_ne!(
        baseline, text_swapped,
        "a text position stopped reading the embedding table"
    );
}

// ---------------------------------------------------------------------------
// Lifetime
// ---------------------------------------------------------------------------

/// `reset()` drops the map, which is what makes a bulk-OCR loop safe: page
/// N+1's prefill must not inherit page N's spans.
#[test]
fn reset_clears_the_injection_map_and_rollback_does_not() {
    let dir = build_vision("lifetime");
    let ids = prompt_ids();

    let mut runner = open(&dir);
    inject(&mut runner, &ids);
    assert!(runner.prompt_vision().is_some());

    let point = runner.checkpoint();
    runner.rollback(&point);
    assert!(
        runner.prompt_vision().is_some(),
        "a speculative rewind stays inside the prompt and must keep its map"
    );

    runner.reset();
    assert!(
        runner.prompt_vision().is_none(),
        "a new generation inherited the previous prompt's spans"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Refusals. Each is a wrong image reaching the model fluently.
// ---------------------------------------------------------------------------

#[test]
fn a_malformed_map_is_refused_by_name() {
    let dir = build_vision("refusals");
    let ids = prompt_ids();
    let params = params();

    let mut runner = open(&dir);
    let embedding = runner
        .encode_image(&image(&params), &params)
        .expect("the tower runs");
    let positions = positions_for(&ids);

    let refusal = |runner: &mut RealForwardRunner,
                   e: &[turbospark_runtime::vision::VisionEmbedding],
                   p: &MropePositions,
                   len: usize| {
        match runner.set_prompt_vision(e, p, len) {
            Ok(()) => panic!("accepted a malformed map"),
            Err(err) => err.to_string(),
        }
    };

    // The triples cover a different prompt than the one being prefilled.
    let msg = refusal(
        &mut runner,
        std::slice::from_ref(&embedding),
        &positions,
        ids.len() + 1,
    );
    assert!(msg.contains("mrope triples cover"), "{msg}");

    // Images and placeholder runs disagree in COUNT.
    let msg = refusal(&mut runner, &[], &positions, ids.len());
    assert!(msg.contains("placeholder span"), "{msg}");

    // A span longer than the tower's own merged-token count: the second half
    // of it would be blitted from the next image's rows, or off the end.
    let mut long_span = positions.clone();
    long_span.spans[0].len += 1;
    let msg = refusal(
        &mut runner,
        std::slice::from_ref(&embedding),
        &long_span,
        ids.len(),
    );
    assert!(msg.contains("merged token"), "{msg}");

    // A span running past the prompt.
    let mut past_end = positions.clone();
    past_end.spans[0].start = ids.len() - 1;
    let msg = refusal(
        &mut runner,
        std::slice::from_ref(&embedding),
        &past_end,
        ids.len(),
    );
    assert!(msg.contains("runs past"), "{msg}");

    // A delta that would put the first decode position before zero.
    let mut bad_delta = positions.clone();
    bad_delta.rope_delta = -(ids.len() as i32) - 1;
    let msg = refusal(
        &mut runner,
        std::slice::from_ref(&embedding),
        &bad_delta,
        ids.len(),
    );
    assert!(msg.contains("rope_delta"), "{msg}");

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// The change detector
// ---------------------------------------------------------------------------

/// A frozen digest over an injected run.
///
/// Every case above is SELF-RELATIVE -- each builds its own baseline inside
/// the same binary, so a change to the math that applies to both arms leaves
/// all of them green (AGENTS.md Gotcha 51). This constant is the only
/// assertion here compared against something computed before the change.
///
/// It is a change DETECTOR and not a correctness claim: the fixture's weights
/// and the fixture's image are both untrained noise, so it says the injected
/// pipeline moved and never that it is right. Re-freezing needs a stated
/// reason.
const FROZEN_INJECTED_DIGEST: &str = "5f0967ae";

#[test]
fn an_injected_run_has_a_frozen_digest() {
    let dir = build_vision("digest");
    let ids = prompt_ids();
    let mut runner = open(&dir);
    inject(&mut runner, &ids);
    let got = digest(&walk(&mut runner, &ids));
    println!("injected digest = {got}");
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(
        got, FROZEN_INJECTED_DIGEST,
        "the injected pipeline moved; see this constant's doc comment before re-freezing"
    );
}
