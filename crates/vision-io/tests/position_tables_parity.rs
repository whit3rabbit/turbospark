//! The three position tables against the reference: the interpolated
//! position embedding, the vision rope frequency rows, and the mRoPE triples.

mod pos_embed_oracle {
    include!("generated/pos_embed_oracle.rs");
}
mod rope_oracle {
    include!("generated/rope_freqs_oracle.rs");
}
mod mrope_oracle {
    include!("generated/mrope_oracle.rs");
}

use turbospark_vision_io::{
    mrope::VisionSpecialIds, mrope_position_triples, pos_embed_weights,
    rope::vision_rope_freq_rows, splice_and_walk, splice_image_placeholders, GridThw, ImageSpan,
    PreprocessParams,
};

fn params(merge_size: usize) -> PreprocessParams {
    PreprocessParams {
        patch_size: 16,
        temporal_patch_size: 2,
        merge_size,
        in_channels: 3,
        min_pixels: 1,
        max_pixels: 1 << 40,
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
        rescale_factor: 1.0 / 255.0,
    }
}

// ------------------------------------------------------------- pos embedding

/// Apply the index/weight table to the seeded reference table, producing the
/// rows the tower would produce.
fn gather(table: &turbospark_vision_io::PosEmbedTable, hidden: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; table.len() * hidden];
    for k in 0..4 {
        for (row, (&index, &weight)) in table.indices[k].iter().zip(&table.weights[k]).enumerate() {
            let src = &pos_embed_oracle::TABLE[index * hidden..(index + 1) * hidden];
            for (slot, &value) in out[row * hidden..(row + 1) * hidden].iter_mut().zip(src) {
                *slot += value * weight;
            }
        }
    }
    out
}

/// Four weighted table rows summed in f32; the reference sums the same four in
/// the same order, so the only slack is the last bit or two of the accumulator.
const TOLERANCE: f32 = 1e-6;

macro_rules! pos_embed_case {
    ($test:ident, $grid:ident, $out:ident) => {
        #[test]
        fn $test() {
            let (t, h, w) = pos_embed_oracle::$grid;
            let table = pos_embed_weights(
                GridThw::new(t, h, w),
                pos_embed_oracle::SIDE,
                &params(pos_embed_oracle::MERGE_SIZE),
            )
            .expect("interpolates");
            assert_eq!(table.len(), t * h * w);
            let got = gather(&table, pos_embed_oracle::HIDDEN);
            let want = pos_embed_oracle::$out;
            assert_eq!(got.len(), want.len());
            for (i, (&g, &wv)) in got.iter().zip(want).enumerate() {
                assert!(
                    (g - wv).abs() <= TOLERANCE,
                    "grid {t}x{h}x{w}, element {i}: got {g}, want {wv}"
                );
            }
        }
    };
}

pos_embed_case!(pos_embed_2x2, GRID_2X2, OUT_2X2);
pos_embed_case!(pos_embed_4x6, GRID_4X6, OUT_4X6);
// Exactly the table's own side: every patch lands on a table row and the
// fractional weights are all zero, so this is the case a broken interpolation
// can still pass. Kept because it is the case a broken INDEX cannot.
pos_embed_case!(pos_embed_8x8, GRID_8X8, OUT_8X8);
// Larger than the table's side on both axes: upsampling past it, which is the
// ordinary case for a real page.
pos_embed_case!(pos_embed_12x10, GRID_12X10, OUT_12X10);
pos_embed_case!(pos_embed_2x16, GRID_2X16, OUT_2X16);
pos_embed_case!(pos_embed_6x2, GRID_6X2, OUT_6X2);

#[test]
fn the_four_interpolation_weights_sum_to_one() {
    // An invariant no golden states, and the one that catches a weight
    // expression that is individually plausible: `dh * dw` in the wrong corner
    // still lands in `[0, 1]` and still varies with the grid.
    for (h, w) in [(2usize, 2usize), (4, 6), (8, 8), (12, 10), (1, 1), (16, 3)] {
        let table = pos_embed_weights(GridThw::new(1, h, w), pos_embed_oracle::SIDE, &params(1))
            .expect("interpolates");
        for row in 0..table.len() {
            let sum: f32 = (0..4).map(|k| table.weights[k][row]).sum();
            assert!((sum - 1.0).abs() < 1e-6, "grid {h}x{w} row {row}: {sum}");
        }
    }
}

#[test]
fn a_single_patch_axis_reads_the_tables_first_row_not_its_centre() {
    // `linspace(0, n-1, 1)` yields `[0]`, so a one-patch axis maps to index 0
    // with weight 1. Reproduced from the reference rather than "fixed" to the
    // centre, which is what an independent derivation would pick.
    let table = pos_embed_weights(GridThw::new(1, 1, 1), pos_embed_oracle::SIDE, &params(1))
        .expect("interpolates");
    assert_eq!(table.len(), 1);
    assert_eq!(
        [
            table.indices[0][0],
            table.indices[1][0],
            table.indices[2][0],
            table.indices[3][0]
        ],
        [0, 1, pos_embed_oracle::SIDE, pos_embed_oracle::SIDE + 1]
    );
    assert_eq!(table.weights[0][0], 1.0);
    for k in 1..4 {
        assert_eq!(table.weights[k][0], 0.0);
    }
}

#[test]
fn the_mapping_is_endpoint_preserving_rather_than_half_pixel() {
    // The last patch of an axis must land exactly on the LAST table row with
    // no fractional weight. Under the half-pixel convention it lands short of
    // it and blends, which is a smooth shift rather than an error, and is the
    // single most likely way to get this function subtly wrong.
    for count in [2usize, 3, 5, 9, 17] {
        let table = pos_embed_weights(
            GridThw::new(1, count, 1),
            pos_embed_oracle::SIDE,
            &params(1),
        )
        .expect("interpolates");
        let last = table.len() - 1;
        assert_eq!(
            table.indices[0][last] / pos_embed_oracle::SIDE,
            pos_embed_oracle::SIDE - 1,
            "count {count}: last patch does not reach the last table row"
        );
        assert_eq!(
            table.weights[0][last] + table.weights[1][last],
            1.0,
            "count {count}"
        );
        assert_eq!(table.indices[0][0], 0, "count {count}: first patch");
    }
}

// ---------------------------------------------------------------- vision rope

macro_rules! rope_case {
    ($test:ident, $grid:ident, $rows:ident) => {
        #[test]
        fn $test() {
            let (t, h, w) = rope_oracle::$grid;
            let got = vision_rope_freq_rows(
                GridThw::new(t, h, w),
                rope_oracle::HEAD_DIM,
                rope_oracle::THETA,
                &params(rope_oracle::MERGE_SIZE),
            )
            .expect("builds rows");
            let want = rope_oracle::$rows;
            assert_eq!(got.len(), want.len(), "grid {t}x{h}x{w}: row count");
            assert_eq!(got.len(), t * h * w * rope_oracle::HEAD_DIM / 2);
            for (i, (&g, &wv)) in got.iter().zip(want).enumerate() {
                // FOUR ULP, and the slack is the REFERENCE's. Every other
                // fixture in this crate is held to an absolute 1e-6 or to
                // exact equality; this one cannot be, because mlx's f32 `**`
                // is not correctly rounded. Measured on `inv_freq[2]` at
                // theta 10,000: mlx returns 0.3593813478946686 where the
                // correctly-rounded f32 is 0.35938137769699097, about 2.5 ULP
                // low, and this port lands on the latter. So the disagreement
                // is the reference being slightly less accurate, not this
                // being slightly wrong -- which is why the bar is stated in
                // ULP rather than nudged until it passes.
                let bar = 4.0 * f32::EPSILON * g.abs().max(1.0);
                assert!(
                    (g - wv).abs() <= bar,
                    "grid {t}x{h}x{w}, element {i}: got {g}, want {wv}"
                );
            }
        }
    };
}

rope_case!(rope_1_2x2, GRID_1_2X2, ROWS_1_2X2);
rope_case!(rope_1_4x6, GRID_1_4X6, ROWS_1_4X6);
rope_case!(rope_1_2x10, GRID_1_2X10, ROWS_1_2X10);
// `t > 1`, where the reference TILES the coordinate list rather than
// extending it: the second temporal group repeats the first's coordinates.
rope_case!(rope_2_4x4, GRID_2_4X4, ROWS_2_4X4);

#[test]
fn the_height_half_comes_first_and_the_fixture_can_tell() {
    // Swapping the two halves keeps the row width and the value range, so the
    // only thing that catches it is a grid whose h and w differ -- assert the
    // fixture contains one before trusting the cases above.
    let asymmetric = rope_oracle::GRID_1_4X6;
    assert_ne!(asymmetric.1, asymmetric.2);
    let (t, h, w) = asymmetric;
    let rows = vision_rope_freq_rows(
        GridThw::new(t, h, w),
        rope_oracle::HEAD_DIM,
        rope_oracle::THETA,
        &params(rope_oracle::MERGE_SIZE),
    )
    .expect("builds rows");
    let half = rope_oracle::HEAD_DIM / 4;
    // Patch 0 of window 0 is at (row 0, col 0), so both halves are zero; the
    // patch at local (0, 1) is (row 0, col 1), so its height half is zero and
    // its width half is not. That asymmetry is the discriminator.
    let row1 = &rows[rope_oracle::HEAD_DIM / 2..rope_oracle::HEAD_DIM];
    assert!(
        row1[..half].iter().all(|&v| v == 0.0),
        "height half should be row 0"
    );
    assert!(
        row1[half..].iter().any(|&v| v != 0.0),
        "width half should be column 1"
    );
}

#[test]
fn the_vision_theta_is_not_the_trunks() {
    // The trunk declares 1e7 and the tower takes the library's 10,000. A test
    // rather than a comment because the two are one constant apart and the
    // wrong one produces finite, plausible, spatially-flat rotations.
    assert_eq!(turbospark_vision_io::rope::VISION_ROPE_THETA, 10_000.0);
    assert_eq!(
        rope_oracle::THETA,
        turbospark_vision_io::rope::VISION_ROPE_THETA
    );
}

// ----------------------------------------------------------------- mrope

macro_rules! mrope_case {
    ($test:ident, $ids:ident, $grids:ident, $triples:ident, $delta:ident) => {
        #[test]
        fn $test() {
            let grids: Vec<GridThw> = mrope_oracle::$grids
                .iter()
                .map(|&(t, h, w)| GridThw::new(t, h, w))
                .collect();
            let got = mrope_position_triples(
                mrope_oracle::$ids,
                &grids,
                VisionSpecialIds {
                    vision_start: mrope_oracle::VISION_START_ID,
                    image_pad: mrope_oracle::IMAGE_PAD_ID,
                },
                mrope_oracle::MERGE_SIZE,
            )
            .expect("walks");
            assert_eq!(got.triples, mrope_oracle::$triples.to_vec(), "triples");
            assert_eq!(got.rope_delta, mrope_oracle::$delta, "rope_delta");
            assert_eq!(got.spans.len(), grids.len(), "span count");
            for (span, grid) in got.spans.iter().zip(&grids) {
                assert_eq!(
                    span.len,
                    grid.merged_tokens(mrope_oracle::MERGE_SIZE),
                    "span length"
                );
                for i in span.start..span.start + span.len {
                    assert_eq!(
                        mrope_oracle::$ids[i],
                        mrope_oracle::IMAGE_PAD_ID,
                        "span covers a non-placeholder at {i}"
                    );
                }
            }
        }
    };
}

mrope_case!(
    mrope_text_only,
    TEXT_ONLY_IDS,
    TEXT_ONLY_GRIDS,
    TEXT_ONLY_TRIPLES,
    TEXT_ONLY_ROPE_DELTA
);
mrope_case!(
    mrope_image_mid,
    IMAGE_MID_IDS,
    IMAGE_MID_GRIDS,
    IMAGE_MID_TRIPLES,
    IMAGE_MID_ROPE_DELTA
);
mrope_case!(
    mrope_image_at_start,
    IMAGE_AT_START_IDS,
    IMAGE_AT_START_GRIDS,
    IMAGE_AT_START_TRIPLES,
    IMAGE_AT_START_ROPE_DELTA
);
mrope_case!(
    mrope_image_at_end,
    IMAGE_AT_END_IDS,
    IMAGE_AT_END_GRIDS,
    IMAGE_AT_END_TRIPLES,
    IMAGE_AT_END_ROPE_DELTA
);
mrope_case!(
    mrope_two_images,
    TWO_IMAGES_IDS,
    TWO_IMAGES_GRIDS,
    TWO_IMAGES_TRIPLES,
    TWO_IMAGES_ROPE_DELTA
);
mrope_case!(
    mrope_adjacent_images,
    ADJACENT_IMAGES_IDS,
    ADJACENT_IMAGES_GRIDS,
    ADJACENT_IMAGES_TRIPLES,
    ADJACENT_IMAGES_ROPE_DELTA
);
mrope_case!(
    mrope_square_image,
    SQUARE_IMAGE_IDS,
    SQUARE_IMAGE_GRIDS,
    SQUARE_IMAGE_TRIPLES,
    SQUARE_IMAGE_ROPE_DELTA
);

#[test]
fn a_text_run_gets_the_same_position_in_all_three_slots() {
    // What makes text behave as ordinary 1-D positions, and therefore what
    // lets M-V5 keep every text token on the existing rope path. Asserted over
    // the fixture rather than one case: a walk that diverged the three slots
    // on, say, a run after an image would still pass a text-only case.
    for (ids, grids) in [
        (mrope_oracle::IMAGE_MID_IDS, mrope_oracle::IMAGE_MID_GRIDS),
        (mrope_oracle::TWO_IMAGES_IDS, mrope_oracle::TWO_IMAGES_GRIDS),
    ] {
        let grids: Vec<GridThw> = grids
            .iter()
            .map(|&(t, h, w)| GridThw::new(t, h, w))
            .collect();
        let got = mrope_position_triples(
            ids,
            &grids,
            VisionSpecialIds {
                vision_start: mrope_oracle::VISION_START_ID,
                image_pad: mrope_oracle::IMAGE_PAD_ID,
            },
            mrope_oracle::MERGE_SIZE,
        )
        .expect("walks");
        let in_image: Vec<bool> = {
            let mut flags = vec![false; ids.len()];
            for span in &got.spans {
                flags[span.start..span.start + span.len].fill(true);
            }
            flags
        };
        for (i, &(t, h, w)) in got.triples.iter().enumerate() {
            if !in_image[i] {
                assert_eq!((t, h), (h, w), "text token {i} has split positions");
            }
        }
    }
}

#[test]
fn every_generated_mrope_case_has_a_test() {
    // The fixture's case list is regenerated from the script; the `mrope_case!`
    // invocations are typed. Without this, adding a case to the script would
    // silently leave it untested.
    let tested = [
        "text_only",
        "image_mid",
        "image_at_start",
        "image_at_end",
        "two_images",
        "adjacent_images",
        "square_image",
    ];
    assert_eq!(tested.to_vec(), mrope_oracle::CASE_NAMES.to_vec());
}

#[test]
fn a_grid_count_mismatch_is_refused() {
    // The reference indexes its grid list positionally and would read past
    // the end or leave one unused. Both directions are checked.
    let one = [GridThw::new(1, 4, 6)];
    let special = VisionSpecialIds {
        vision_start: mrope_oracle::VISION_START_ID,
        image_pad: mrope_oracle::IMAGE_PAD_ID,
    };
    assert!(mrope_position_triples(mrope_oracle::IMAGE_MID_IDS, &[], special, 2).is_err());
    assert!(mrope_position_triples(mrope_oracle::TEXT_ONLY_IDS, &one, special, 2).is_err());
}

#[test]
fn a_placeholder_with_no_vision_start_is_treated_as_text() {
    // The reference SIZES its loop from `vision_start` markers and PLACES each
    // block from the placeholder token. On a malformed prompt the two
    // disagree: an unmarked placeholder is not counted, so the loop does not
    // run and it falls through as ordinary text. Reproduced rather than
    // corrected -- agreeing with the trunk on every prompt it was trained on
    // is worth more than being right about one it was not.
    let ids = [1000, mrope_oracle::IMAGE_PAD_ID, 1001];
    let got = mrope_position_triples(
        &ids,
        &[],
        VisionSpecialIds {
            vision_start: mrope_oracle::VISION_START_ID,
            image_pad: mrope_oracle::IMAGE_PAD_ID,
        },
        2,
    )
    .expect("walks");
    assert_eq!(got.triples, vec![(0, 0, 0), (1, 1, 1), (2, 2, 2)]);
    assert_eq!(got.rope_delta, 0);
    assert!(got.spans.is_empty());
}

#[test]
fn an_image_costs_fewer_positions_than_tokens() {
    // The property the whole scheme exists for. A square image spends
    // `max(t, h/merge, w/merge)` positions for `h*w/merge^2` tokens, so its
    // rope_delta is strongly negative -- and a walk that advanced by the token
    // count instead would still produce monotone, plausible triples.
    let grids = [GridThw::new(1, 8, 8)];
    let got = mrope_position_triples(
        mrope_oracle::SQUARE_IMAGE_IDS,
        &grids,
        VisionSpecialIds {
            vision_start: mrope_oracle::VISION_START_ID,
            image_pad: mrope_oracle::IMAGE_PAD_ID,
        },
        2,
    )
    .expect("walks");
    // 16 placeholder tokens, 4 positions.
    assert_eq!(got.spans[0].len, 16);
    assert!(got.rope_delta < 0, "rope_delta {}", got.rope_delta);
    assert_eq!(got.rope_delta, mrope_oracle::SQUARE_IMAGE_ROPE_DELTA);
}

// ---------------------------------------------------------------------------
// The splice (ROADMAP M-V6)
// ---------------------------------------------------------------------------

/// The splice and the walk are two halves of one pipeline, so the case that
/// matters is the COMPOSITION: a template's one-placeholder-per-image output,
/// expanded, must produce the spans the walk then finds.
#[test]
fn the_splice_feeds_the_walk_the_spans_it_expects() {
    let special = VisionSpecialIds {
        vision_start: mrope_oracle::VISION_START_ID,
        image_pad: mrope_oracle::IMAGE_PAD_ID,
    };
    // What a template renders: ONE placeholder, whatever the image's size.
    let rendered = [
        1000,
        mrope_oracle::VISION_START_ID,
        mrope_oracle::IMAGE_PAD_ID,
        1001,
        1002,
    ];
    // Grid 1x4x6 at merge 2 is 2x3 = 6 merged tokens.
    let grid = GridThw::new(1, 4, 6);
    let merged = grid.merged_tokens(2);
    assert_eq!(merged, 6);

    let spliced = splice_image_placeholders(&rendered, mrope_oracle::IMAGE_PAD_ID, &[merged])
        .expect("splices");
    assert_eq!(spliced.len(), rendered.len() - 1 + merged);
    let walked = mrope_position_triples(&spliced, &[grid], special, 2).expect("walks");
    assert_eq!(walked.spans, vec![ImageSpan { start: 2, len: 6 }]);
    // The block spent max(1, 2, 3) = 3 positions for its 6 tokens, so the two
    // text tokens after it resume at 5 rather than at 8.
    assert_eq!(walked.triples[8..], [(5, 5, 5), (6, 6, 6)]);
}

/// Order is preserved and each image gets its OWN count. A splice that used
/// one count for every image passes the single-image case above.
#[test]
fn two_images_expand_to_their_own_counts_in_order() {
    let pad = mrope_oracle::IMAGE_PAD_ID;
    let ids = [1000, pad, 1001, pad, 1002];
    let got = splice_image_placeholders(&ids, pad, &[2, 5]).expect("splices");
    let mut want = vec![1000];
    want.extend(std::iter::repeat_n(pad, 2));
    want.push(1001);
    want.extend(std::iter::repeat_n(pad, 5));
    want.push(1002);
    assert_eq!(got, want);
}

/// A count of one is the identity, which is worth pinning because it is the
/// shape a caller reaches for when an image is tiny and it must not become a
/// special case.
#[test]
fn a_single_merged_token_leaves_the_sequence_unchanged() {
    let pad = mrope_oracle::IMAGE_PAD_ID;
    let ids = [1000, pad, 1001];
    assert_eq!(
        splice_image_placeholders(&ids, pad, &[1]).expect("splices"),
        ids.to_vec()
    );
}

#[test]
fn a_placeholder_count_mismatch_is_refused_in_both_directions() {
    let pad = mrope_oracle::IMAGE_PAD_ID;
    // Two placeholders, one count.
    assert!(splice_image_placeholders(&[pad, 1000, pad], pad, &[4]).is_err());
    // One placeholder, two counts.
    assert!(splice_image_placeholders(&[pad, 1000], pad, &[4, 4]).is_err());
    // And a zero count, which would DROP the placeholder and renumber every
    // later span while leaving the prompt fluent.
    assert!(splice_image_placeholders(&[pad], pad, &[0]).is_err());
}

/// A prompt with no images is untouched, which is the path every text-only
/// caller takes and the one that must cost nothing.
#[test]
fn a_text_only_sequence_passes_through() {
    let pad = mrope_oracle::IMAGE_PAD_ID;
    let ids = [1000, 1001, 1002];
    assert_eq!(
        splice_image_placeholders(&ids, pad, &[]).expect("splices"),
        ids.to_vec()
    );
}

/// The composed helper must agree with calling the two halves by hand, and
/// must derive its counts from the GRIDS rather than accepting them.
#[test]
fn splice_and_walk_derives_its_counts_from_the_grids() {
    let special = VisionSpecialIds {
        vision_start: mrope_oracle::VISION_START_ID,
        image_pad: mrope_oracle::IMAGE_PAD_ID,
    };
    let rendered = [
        1000,
        mrope_oracle::VISION_START_ID,
        mrope_oracle::IMAGE_PAD_ID,
        1001,
    ];
    let grids = [GridThw::new(1, 4, 6)];

    let composed = splice_and_walk(&rendered, &grids, special, 2).expect("composes");
    let by_hand = splice_image_placeholders(
        &rendered,
        mrope_oracle::IMAGE_PAD_ID,
        &[grids[0].merged_tokens(2)],
    )
    .expect("splices");
    assert_eq!(composed.ids, by_hand);
    assert_eq!(
        composed.positions,
        mrope_position_triples(&by_hand, &grids, special, 2).expect("walks")
    );
    // And the count really came from the grid: 1x4x6 at merge 2 is 6.
    assert_eq!(composed.positions.spans[0].len, 6);
}
