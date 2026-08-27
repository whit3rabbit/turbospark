//! `smart_resize` against the reference processor.
//!
//! Every case is an EXACT equality: the output is a pair of integers and the
//! computation that produces it is integer-valued, so there is nothing here a
//! tolerance could legitimately absorb. A near miss is a whole patch row.

mod oracle {
    include!("generated/smart_resize_oracle.rs");
}

use turbospark_vision_io::{resized_dims, PreprocessParams, VisionIoError};

fn params(min_pixels: usize, max_pixels: usize) -> PreprocessParams {
    PreprocessParams {
        patch_size: 16,
        temporal_patch_size: 2,
        merge_size: 2,
        in_channels: 3,
        min_pixels,
        max_pixels,
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
        rescale_factor: 1.0 / 255.0,
    }
}

#[test]
fn every_oracle_case_matches_exactly() {
    assert_eq!(params(0, 0).spatial_factor(), oracle::FACTOR);
    let mut checked = 0usize;
    for &(orig_h, orig_w, min_px, max_px, want_h, want_w) in oracle::CASES {
        let got = resized_dims(orig_h, orig_w, &params(min_px, max_px))
            .unwrap_or_else(|e| panic!("{orig_h}x{orig_w} in {min_px}..{max_px}: {e}"));
        assert_eq!(
            got,
            (want_h, want_w),
            "{orig_h}x{orig_w} in {min_px}..{max_px}"
        );
        checked += 1;
    }
    // The fixture is not empty and has not silently shrunk to a handful of
    // rows: a parity test over zero cases passes.
    assert!(checked > 200, "only {checked} oracle cases");
}

#[test]
fn the_oracle_sweep_exercises_both_budget_branches_and_neither() {
    // A sweep that never leaves the untouched branch would agree with a
    // `resized_dims` that ignores the budget entirely, so assert the fixture
    // reaches all three outcomes before believing the case above.
    let (mut downscaled, mut upscaled, mut untouched) = (0, 0, 0);
    for &(orig_h, orig_w, min_px, max_px, want_h, want_w) in oracle::CASES {
        let rounded = want_h * want_w;
        let orig = orig_h * orig_w;
        if rounded > orig && rounded >= min_px && orig < min_px {
            upscaled += 1;
        } else if rounded < orig && orig > max_px {
            downscaled += 1;
        } else {
            untouched += 1;
        }
    }
    assert!(upscaled > 0 && downscaled > 0 && untouched > 0);
}

#[test]
fn an_extreme_aspect_ratio_is_refused_before_any_rounding() {
    for &(orig_h, orig_w) in oracle::REFUSED {
        let err = resized_dims(
            orig_h,
            orig_w,
            &params(oracle::REAL_MIN_PIXELS, oracle::REAL_MAX_PIXELS),
        )
        .expect_err("{orig_h}x{orig_w} should be refused");
        assert!(
            matches!(err, VisionIoError::AspectRatioTooExtreme { .. }),
            "{orig_h}x{orig_w} refused for the wrong reason: {err}"
        );
    }
    // 200.0 exactly is accepted: the reference tests `> 200`, and a `>=`
    // here would refuse a legal image with a message about an illegal one.
    assert!(resized_dims(
        200,
        1,
        &params(oracle::REAL_MIN_PIXELS, oracle::REAL_MAX_PIXELS)
    )
    .is_ok());
}

#[test]
fn a_zero_edge_is_refused_rather_than_clamped_to_one_patch() {
    // With no lower budget to rescue it, a source under half a factor rounds
    // to zero on both axes. The reference returns that zero; this refuses it,
    // because a zero-patch grid is not an image and a silent clamp to one
    // patch would invent pixels.
    let err = resized_dims(3, 3, &params(0, 1 << 40)).expect_err("3x3 at min_pixels 0");
    assert!(
        matches!(err, VisionIoError::InvalidDimensions { .. }),
        "{err}"
    );
}

#[test]
fn a_zero_source_edge_is_refused() {
    for (h, w) in [(0, 10), (10, 0), (0, 0)] {
        assert!(matches!(
            resized_dims(h, w, &params(1024, 4096)),
            Err(VisionIoError::InvalidDimensions { .. })
        ));
    }
}
