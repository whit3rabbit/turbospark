//! The full preprocessing pipeline against the reference processor.
//!
//! # The one thing this test does that a parity test normally must not
//!
//! It PERMUTES before comparing. The reference's `pixel_values` carry
//! `(C, T, P_h, P_w)` inside each patch row and this crate emits
//! `(T, P_h, P_w, C)`, deliberately, so that the tower's
//! `patch_embed.proj.weight` can be copied verbatim at repack
//! (`docs/VISION_PHASE0.md` item 4).
//!
//! A permutation inside a parity test is exactly the move that lets a wrong
//! answer pass, so it is guarded two ways. It is written out as index
//! arithmetic in [`oracle_index`] rather than hidden in a reshape, and
//! `the_permutation_is_not_the_identity` asserts the two orders genuinely
//! differ on this fixture -- on a case with `C == 1` or `T == 1 && P == 1`
//! they would coincide and the whole comparison would prove nothing.

mod oracle {
    include!("generated/preprocess_oracle.rs");
}

use turbospark_vision_io::{preprocess, PreprocessParams, Rgb8Image};

/// Where the element this crate puts at `(t, py, px, c)` sits in the
/// reference's row.
///
/// Ours: `t*(P*P*C) + py*(P*C) + px*C + c`, i.e. `(T, P_h, P_w, C)`.
/// Theirs: `c*(T*P*P) + t*(P*P) + py*P + px`, i.e. `(C, T, P_h, P_w)`.
fn oracle_index(mine: usize, patch: usize, tps: usize, channels: usize) -> usize {
    let (pp, ppc) = (patch * patch, patch * patch * channels);
    let t = mine / ppc;
    let rest = mine % ppc;
    let py = rest / (patch * channels);
    let rest = rest % (patch * channels);
    let px = rest / channels;
    let c = rest % channels;
    c * (tps * pp) + t * pp + py * patch + px
}

fn params(geometry: (usize, usize, usize), budget: (usize, usize)) -> PreprocessParams {
    let (patch_size, merge_size, temporal_patch_size) = geometry;
    PreprocessParams {
        patch_size,
        temporal_patch_size,
        merge_size,
        in_channels: 3,
        min_pixels: budget.0,
        max_pixels: budget.1,
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
        rescale_factor: 1.0 / 255.0,
    }
}

/// A normalized sample lives in `[-1, 1]`, so an absolute bound IS a relative
/// one here and is the honest form: a near-zero sample has no meaningful
/// relative error.
///
/// 1e-6 rather than anything looser. The resize is fixed-point exact and the
/// normalize is one multiply and one subtract, so the only slack that can
/// legitimately exist is f32 rounding on values of order one. A tolerance set
/// just above a pixel level (2/255 after this normalization) would absorb an
/// entire wrong sample, which is the failure sconce recorded: a bar chosen for
/// comfort hid a 65,000x regression.
const TOLERANCE: f32 = 1e-6;

struct Case {
    name: &'static str,
    src: &'static [u8],
    src_hw: (usize, usize),
    geometry: (usize, usize, usize),
    budget: (usize, usize),
    grid: (usize, usize, usize),
    rows_total: usize,
    patch_dim: usize,
    stride: usize,
    pixel_values: &'static [f32],
}

/// The generated fixture is flat consts rather than a struct, so the cases are
/// gathered by hand here. `CASE_NAMES` in the fixture is the cross-check that
/// none has been dropped.
fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "tiny_a",
            src: oracle::TINY_A_SRC,
            src_hw: oracle::TINY_A_SRC_HW,
            geometry: oracle::TINY_A_GEOMETRY,
            budget: oracle::TINY_A_BUDGET,
            grid: oracle::TINY_A_GRID,
            rows_total: oracle::TINY_A_PATCH_ROWS_TOTAL,
            patch_dim: oracle::TINY_A_PATCH_DIM,
            stride: oracle::TINY_A_SAMPLE_STRIDE,
            pixel_values: oracle::TINY_A_PIXEL_VALUES,
        },
        Case {
            name: "tiny_identity",
            src: oracle::TINY_IDENTITY_SRC,
            src_hw: oracle::TINY_IDENTITY_SRC_HW,
            geometry: oracle::TINY_IDENTITY_GEOMETRY,
            budget: oracle::TINY_IDENTITY_BUDGET,
            grid: oracle::TINY_IDENTITY_GRID,
            rows_total: oracle::TINY_IDENTITY_PATCH_ROWS_TOTAL,
            patch_dim: oracle::TINY_IDENTITY_PATCH_DIM,
            stride: oracle::TINY_IDENTITY_SAMPLE_STRIDE,
            pixel_values: oracle::TINY_IDENTITY_PIXEL_VALUES,
        },
        Case {
            name: "tiny_upscaled",
            src: oracle::TINY_UPSCALED_SRC,
            src_hw: oracle::TINY_UPSCALED_SRC_HW,
            geometry: oracle::TINY_UPSCALED_GEOMETRY,
            budget: oracle::TINY_UPSCALED_BUDGET,
            grid: oracle::TINY_UPSCALED_GRID,
            rows_total: oracle::TINY_UPSCALED_PATCH_ROWS_TOTAL,
            patch_dim: oracle::TINY_UPSCALED_PATCH_DIM,
            stride: oracle::TINY_UPSCALED_SAMPLE_STRIDE,
            pixel_values: oracle::TINY_UPSCALED_PIXEL_VALUES,
        },
        Case {
            name: "tiny_downscaled",
            src: oracle::TINY_DOWNSCALED_SRC,
            src_hw: oracle::TINY_DOWNSCALED_SRC_HW,
            geometry: oracle::TINY_DOWNSCALED_GEOMETRY,
            budget: oracle::TINY_DOWNSCALED_BUDGET,
            grid: oracle::TINY_DOWNSCALED_GRID,
            rows_total: oracle::TINY_DOWNSCALED_PATCH_ROWS_TOTAL,
            patch_dim: oracle::TINY_DOWNSCALED_PATCH_DIM,
            stride: oracle::TINY_DOWNSCALED_SAMPLE_STRIDE,
            pixel_values: oracle::TINY_DOWNSCALED_PIXEL_VALUES,
        },
        Case {
            name: "real_geometry",
            src: oracle::REAL_GEOMETRY_SRC,
            src_hw: oracle::REAL_GEOMETRY_SRC_HW,
            geometry: oracle::REAL_GEOMETRY_GEOMETRY,
            budget: oracle::REAL_GEOMETRY_BUDGET,
            grid: oracle::REAL_GEOMETRY_GRID,
            rows_total: oracle::REAL_GEOMETRY_PATCH_ROWS_TOTAL,
            patch_dim: oracle::REAL_GEOMETRY_PATCH_DIM,
            stride: oracle::REAL_GEOMETRY_SAMPLE_STRIDE,
            pixel_values: oracle::REAL_GEOMETRY_PIXEL_VALUES,
        },
    ]
}

/// Run one case and return `(ours, case)`.
fn run(case: &Case) -> Vec<f32> {
    let params = params(case.geometry, case.budget);
    let image = Rgb8Image::new(case.src_hw.1, case.src_hw.0, case.src.to_vec())
        .unwrap_or_else(|e| panic!("{}: {e}", case.name));
    let out = preprocess(&image, &params).unwrap_or_else(|e| panic!("{}: {e}", case.name));
    assert_eq!(
        (out.grid.t, out.grid.h, out.grid.w),
        case.grid,
        "{}: grid",
        case.name
    );
    assert_eq!(
        out.grid.patches(),
        case.rows_total,
        "{}: patch row count",
        case.name
    );
    assert_eq!(
        out.patch_rows.len(),
        case.rows_total * case.patch_dim,
        "{}: patch matrix size",
        case.name
    );
    assert_eq!(
        out.merged_tokens,
        case.rows_total / (case.geometry.1 * case.geometry.1),
        "{}: merged token count",
        case.name
    );
    out.patch_rows
}

#[test]
fn every_case_matches_the_reference_under_the_documented_permutation() {
    for case in cases() {
        let ours = run(&case);
        let (patch, _merge, tps) = case.geometry;
        let mut compared = 0usize;
        for (k, &want) in case.pixel_values.iter().enumerate() {
            // The fixture stores every `stride`-th element of the reference's
            // FLATTENED pixel_values, so recover the row and the offset first,
            // then permute the offset alone.
            let flat = k * case.stride;
            let row = flat / case.patch_dim;
            let within_theirs = flat % case.patch_dim;
            // Invert `oracle_index` by scanning the row: cheap at these sizes
            // and it exercises the same mapping the forward direction uses,
            // so an error in it cannot cancel itself out.
            let within_ours = (0..case.patch_dim)
                .find(|&m| oracle_index(m, patch, tps, 3) == within_theirs)
                .unwrap_or_else(|| panic!("{}: no source for offset {within_theirs}", case.name));
            let got = ours[row * case.patch_dim + within_ours];
            assert!(
                (got - want).abs() <= TOLERANCE,
                "{}: row {row}, their offset {within_theirs} / our offset {within_ours}: \
                 got {got}, want {want}",
                case.name
            );
            compared += 1;
        }
        assert!(compared > 0, "{}: nothing compared", case.name);
    }
}

#[test]
fn the_permutation_is_not_the_identity() {
    // If it were, this test file would be comparing our order against itself
    // and could not see an axis-order bug at all. Assert the fixture
    // discriminates before believing the case above.
    for case in cases() {
        let (patch, _merge, tps) = case.geometry;
        let moved = (0..case.patch_dim)
            .filter(|&m| oracle_index(m, patch, tps, 3) != m)
            .count();
        assert!(
            moved * 2 > case.patch_dim,
            "{}: only {moved} of {} offsets move under the permutation",
            case.name,
            case.patch_dim
        );
    }
}

#[test]
fn the_permutation_is_a_bijection() {
    for case in cases() {
        let (patch, _merge, tps) = case.geometry;
        let mut seen = vec![false; case.patch_dim];
        for m in 0..case.patch_dim {
            let o = oracle_index(m, patch, tps, 3);
            assert!(
                o < case.patch_dim,
                "{}: offset {m} maps out of range",
                case.name
            );
            assert!(!seen[o], "{}: offset {o} claimed twice", case.name);
            seen[o] = true;
        }
    }
}

#[test]
fn no_generated_case_is_missing_from_the_hand_written_list() {
    // The fixture's cases are regenerated from the script's own table; this
    // file's list is typed. Without this check, adding a case to the script
    // would silently leave it untested.
    let listed: Vec<&str> = cases().iter().map(|c| c.name).collect();
    assert_eq!(listed, oracle::CASE_NAMES.to_vec());
}

#[test]
fn the_fixture_covers_all_three_resize_branches() {
    // Without this the whole file would pass against a pipeline that ignored
    // the pixel budget, since three of four cases would be untouched.
    let (mut up, mut down, mut identity) = (0, 0, 0);
    for case in cases() {
        let params = params(case.geometry, case.budget);
        let (rh, rw) = turbospark_vision_io::resized_dims(case.src_hw.0, case.src_hw.1, &params)
            .expect("resizes");
        if (rh, rw) == case.src_hw {
            identity += 1;
        } else if rh * rw > case.src_hw.0 * case.src_hw.1 {
            up += 1;
        } else {
            down += 1;
        }
    }
    // The identity arm is the one that is easy to leave uncovered, and it is
    // the one that proves the resampler's early-out matters: a bicubic
    // resample onto the source's own size is a mild blur, not a no-op.
    assert!(
        up > 0 && down > 0 && identity > 0,
        "up {up}, down {down}, identity {identity}"
    );
}
