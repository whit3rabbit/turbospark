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

/// The map survives BOTH `rollback` and `reset`, and only an explicit
/// `clear_prompt_vision` drops it.
///
/// **M-V5 had `reset` clearing it and that was wrong**, for a reason no test
/// here could see: `run_raw_completion` calls `reset` at ENTRY, so the clear
/// landed on the map for the very prompt about to be prefilled. See
/// `an_injected_map_survives_the_generation_loops_own_reset`, which is the
/// case that reaches it.
///
/// What keeps a bulk-OCR loop safe instead is the caller CONSUMING the map
/// per page. A caller who forgets gets no injection rather than the previous
/// page's -- vague answers instead of confident wrong ones.
#[test]
fn only_an_explicit_clear_drops_the_injection_map() {
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
        runner.prompt_vision().is_some(),
        "reset runs at the START of the generation the map belongs to; clearing there \
         destroys it before prefill"
    );

    runner.clear_prompt_vision();
    assert!(
        runner.prompt_vision().is_none(),
        "an explicit clear is what ends the map's life"
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
///
/// **DEVICE-BRANCHED, since 2026-09-04**; see `real_forward_muse.rs`'s
/// identical fix for the full account of why a bit-exact hash of
/// GPU-computed FP16 values does not reproduce on CI's virtualized
/// `macos-latest` runner, and why the fix is to run the ORIGINAL exact
/// digest comparison on real Apple Silicon (unchanged from before this
/// fix) and fall back to a tolerant comparison against `FROZEN_LOGITS`
/// below only on a virtualized device. `FROZEN_LOGITS` is recovered from
/// the same real-hardware run this digest reproduces on.
const FROZEN_INJECTED_DIGEST: &str = "5f0967ae";

/// The full frozen logit array `FROZEN_INJECTED_DIGEST` was taken over.
/// Only consulted on a virtualized device; the real-hardware branch
/// compares the digest directly.
#[rustfmt::skip]
const FROZEN_LOGITS: [f32; VOCAB as usize] = [
    -2.5136719, 0.4638672, -6.1796875, 6.0, -4.4570313, -2.3222656, 4.8359375, 6.7109375,
    4.3007813, 13.1953125, -2.3144531, 7.0507813, 7.5546875, 5.1289063, 1.8242188, -1.2402344,
    -1.8417969, 14.5078125, -0.016479492, -8.7421875, 2.6679688, 2.9550781, 3.1542969, -0.6894531,
    -0.4597168, 7.296875, 2.9707031, -1.0185547, -14.421875, -5.453125, -5.4609375, -1.3642578,
    -13.0, -0.56933594, 2.4902344, -0.008026123, 1.3251953, 1.1142578, 2.3789063, -1.8193359,
    3.0449219, -6.8515625, -0.6923828, 6.1210938, -3.5742188, 5.7070313, 5.4648438, 5.0898438,
    0.5151367, 4.0546875, 6.4492188, -3.296875, -0.60058594, 1.8515625, -3.8417969, 12.7265625,
    -6.4609375, -1.9892578, -7.515625, -6.2890625, -2.9667969, 1.7060547, -5.3398438, 3.6699219,
    9.3046875, -1.4677734, 2.765625, 0.8125, 9.0703125, 2.875, -4.7460938, -1.0068359,
    0.92529297, 1.9970703, -3.1308594, 7.4257813, -5.7148438, -1.7373047, -0.5283203, -5.1757813,
    -1.8027344, -4.7578125, 11.28125, 10.5234375, -2.0292969, -5.7421875, 1.4716797, -1.3769531,
    5.421875, 1.5957031, 6.609375, -0.69873047, -8.359375, 10.3203125, 4.46875, -2.1914063,
    1.9345703, 10.9375, -6.1132813, 8.171875, -2.6582031, -1.578125, -9.203125, 6.828125,
    9.6328125, 2.0097656, -3.6777344, -6.2304688, 4.1953125, 2.6855469, -0.39526367, 1.3818359,
    -0.86279297, 9.921875, 2.5507813, -4.3359375, -4.265625, 7.3476563, 4.7929688, -1.046875,
    -7.421875, -2.7949219, -5.9609375, 1.4052734, 13.7734375, 4.6992188, -1.0136719, 6.1367188,
    7.6171875, 4.9726563, -2.3066406, -3.8984375, 2.9648438, 0.40283203, -1.9042969, 0.4765625,
    -4.7539063, -0.5751953, -7.7304688, 0.28271484, -0.10656738, -4.9570313, -9.375, -5.375,
    -8.0859375, -0.48535156, 1.9238281, -2.2324219, 6.1132813, -13.71875, 1.1035156, 4.5390625,
    -0.7739258, -11.8046875, -4.1914063, 1.7011719, -1.8398438, -7.5742188, -10.8515625, 9.96875,
    6.4609375, 4.28125, -0.45703125, -9.4296875, -2.5761719, 5.4492188, 6.2773438, 5.8164063,
    -4.765625, -5.3164063, -0.61328125, -5.8242188, -3.5605469, -1.7988281, -2.46875, 1.3144531,
    -3.5625, 7.9882813, 2.6445313, -7.9296875, -10.28125, 8.65625, 7.0546875, 2.359375,
    5.7265625, 7.8125, -1.0429688, -3.7363281, 9.015625, -6.421875, -0.8833008, 1.7177734,
    9.9375, -2.8847656, 5.5546875, 5.71875, -4.1757813, -8.921875, 9.9453125, -3.3261719,
    2.6289063, -4.3242188, 4.1640625, -0.089782715, -6.8671875, -7.0625, 6.90625, -1.4931641,
    -0.58447266, 10.2421875, 1.0625, 1.5839844, -6.15625, -2.8417969, -0.7861328, 2.8203125,
    5.203125, -3.5820313, -9.734375, -0.6142578, -7.4609375, -2.4238281, -10.171875, -2.9707031,
    -3.0234375, -2.1894531, -7.5859375, 4.6171875, -3.2382813, 0.96240234, 1.2529297, 2.671875,
    2.6074219, 12.2578125, 3.0429688, -7.5117188, -6.2382813, 1.8916016, 4.9257813, 1.4101563,
    11.1953125, 5.6796875, 7.3007813, -0.4921875, 3.6113281, -2.3300781, 5.6054688, 1.7978516,
    1.0527344, 8.421875, 5.1367188, 10.25, 3.2695313, 2.2617188, -1.3115234, -3.7851563,
];

#[test]
fn an_injected_run_has_a_frozen_digest() {
    let dir = build_vision("digest");
    let ids = prompt_ids();
    let mut runner = open(&dir);
    inject(&mut runner, &ids);
    let bits = walk(&mut runner, &ids);
    let got = digest(&bits);
    println!("injected digest = {got}");
    let _ = std::fs::remove_dir_all(&dir);
    let context = gpu::MetalContext::new().expect("Metal device");
    let device_name = context.device().name().to_string();
    drop(context);
    if device_name.contains("Paravirtual") {
        println!(
            "device {device_name:?} is virtualized, not the real Apple Silicon this digest \
             was taken on; comparing against the frozen reference with a tolerance instead"
        );
        let floats: Vec<f32> = bits
            .iter()
            .map(|&b| half::f16::from_bits(b).to_f32())
            .collect();
        assert_eq!(floats.len(), FROZEN_LOGITS.len());
        for (i, (&got, &want)) in floats.iter().zip(FROZEN_LOGITS.iter()).enumerate() {
            let diff = (got - want).abs();
            let tol = 0.02_f32.max(want.abs() * 0.02);
            assert!(
                diff <= tol,
                "logit {i}: the injected pipeline moved: got {got}, want {want} \
                 (diff {diff}, tolerance {tol}); see FROZEN_INJECTED_DIGEST's doc before \
                 re-freezing"
            );
        }
        return;
    }
    assert_eq!(
        got, FROZEN_INJECTED_DIGEST,
        "the injected pipeline moved; see this constant's doc comment before re-freezing"
    );
}

// ---------------------------------------------------------------------------
// The generation LOOP, not `produce` (ROADMAP M-V7)
// ---------------------------------------------------------------------------

/// **THE ONE CASE THAT WOULD HAVE CAUGHT M-V5's REAL BUG.**
///
/// `run_raw_completion` calls `producer.reset()` at ENTRY, and M-V5 wired
/// `reset` to clear `prompt_vision` -- so the injection map was destroyed
/// before a single token was prefilled. Every `--image` run prefilled
/// placeholder embeddings and hallucinated, with the right prompt length and
/// no error anywhere.
///
/// Nothing above catches it: every other case in this file drives `produce`
/// directly, and so does `crates/bench`'s cross-engine dump. A caller setting
/// a map and then running the ORDINARY generation loop was untested, which is
/// exactly the path every front end takes.
///
/// The assertion is that the injected rows still REACH the output through the
/// loop -- perturb them and the generated tokens must move.
#[test]
fn an_injected_map_survives_the_generation_loops_own_reset() {
    let dir = build_vision("loop-reset");
    let ids = prompt_ids();
    let params = params();

    let generate = |perturb: bool| -> Vec<i32> {
        let mut runner = open(&dir);
        let mut embedding = runner
            .encode_image(&image(&params), &params)
            .expect("the tower runs");
        if perturb {
            for v in &mut embedding.rows {
                *v = F16::from_f32(F16::from_bits(*v).to_f32() + 0.25).to_bits();
            }
        }
        let positions = positions_for(&ids);
        runner
            .set_prompt_vision(std::slice::from_ref(&embedding), &positions, ids.len())
            .expect("validates");

        let config = turbospark_runtime::GenerationConfig {
            shaping: selection::ShapingConfig::new(0.0, 0, None, 1.0, None).unwrap(),
            max_new_tokens: 4,
            stop_strings: Vec::new(),
            extra_stop_tokens: Vec::new(),
            rate: Default::default(),
        };
        let tokenizer = load_fixture_tokenizer();
        let mut out = Vec::new();
        turbospark_runtime::run_raw_completion(
            &mut runner,
            &tokenizer,
            &ids,
            &config,
            4096,
            VOCAB as usize,
            |e| {
                if let turbospark_runtime::RawDecodeProgress::Token { id, .. } = e {
                    out.push(id);
                }
            },
        )
        .expect("generates");
        out
    };

    let plain = generate(false);
    let perturbed = generate(true);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!plain.is_empty(), "the loop generated nothing");
    assert_ne!(
        plain, perturbed,
        "the injected rows did not reach the generation loop; `reset` is eating the map"
    );
}

/// The fixture tokenizer, for the loop's stop matcher and detokenizer alone.
///
/// The synthetic install ships no sidecars, and the loop needs SOME tokenizer;
/// its vocabulary is irrelevant here because the ids are supplied directly and
/// the assertion is about whether the rows reach the output.
fn load_fixture_tokenizer() -> tokenizer::MfTokenizer {
    let dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    tokenizer::MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer loads")
}
