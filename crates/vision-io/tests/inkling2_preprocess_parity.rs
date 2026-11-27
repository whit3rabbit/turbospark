//! Inkling 2 preprocessing against the numpy transcription of transformers'
//! `Inkling2ImageProcessor` (`scripts/inkling2_preprocess_oracle.py`).
//!
//! The reference emits `[N, 40, 40, 3]` tiles in raster order and leaves the
//! 2x2 patch split to the model. This crate emits merge-block patch rows. The
//! comparison maps each reference sample to its patch row by index arithmetic
//! in [`patch_index`], never by reshaping, and the fixtures are non-square and
//! not multiples of 40, so a swapped axis or a padding error cannot pass.

mod oracle {
    include!("generated/inkling2_preprocess_oracle.rs");
}

use turbospark_vision_io::{
    inkling2_composite_rgba, inkling2_preprocess, Rgb8Image, INKLING2_MAX_TOKENS, INKLING2_PATCH,
    INKLING2_PATCH_DIM, INKLING2_TILE,
};

/// The closed-form source pattern, identical to the generator's.
fn pattern(h: usize, w: usize, ch: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(h * w * ch);
    for y in 0..h {
        for x in 0..w {
            for c in 0..ch {
                out.push(((x * 7 + y * 13 + c * 61 + (x * y) % 17) % 256) as u8);
            }
        }
    }
    out
}

/// Where the reference's tile sample `(tile, y, x, c)` (flat index `i`) sits
/// in the merge-block patch rows.
fn patch_index(i: usize) -> usize {
    let per_tile = INKLING2_TILE * INKLING2_TILE * 3;
    let tile = i / per_tile;
    let rem = i % per_tile;
    let (y, x, c) = (
        rem / (INKLING2_TILE * 3),
        (rem / 3) % INKLING2_TILE,
        rem % 3,
    );
    let patch = tile * 4 + (y / INKLING2_PATCH) * 2 + x / INKLING2_PATCH;
    let (py, px) = (y % INKLING2_PATCH, x % INKLING2_PATCH);
    patch * INKLING2_PATCH_DIM + (py * INKLING2_PATCH + px) * 3 + c
}

fn check(name: &str, src: (usize, usize, usize), grid: (usize, usize), samples: &[f32], sum: f64) {
    let (h, w, ch) = src;
    let raw = pattern(h, w, ch);
    let image = match ch {
        3 => Rgb8Image::new(w, h, raw).unwrap(),
        4 => inkling2_composite_rgba(w, h, &raw).unwrap(),
        _ => unreachable!(),
    };
    let p = inkling2_preprocess(&image, INKLING2_MAX_TOKENS).unwrap();
    assert_eq!((p.rows, p.cols), grid, "{name}: grid");
    let total = grid.0 * grid.1 * INKLING2_TILE * INKLING2_TILE * 3;
    assert_eq!(p.pixels.len(), total, "{name}: sample count");
    for (k, &want) in samples.iter().enumerate() {
        let i = k * oracle::STRIDE;
        let got = p.pixels[patch_index(i)];
        // Same f32 expression on both sides: exact.
        assert_eq!(got, want, "{name}: tile sample {i}");
    }
    let got_sum: f64 = p.pixels.iter().map(|&v| f64::from(v)).sum();
    assert!(
        (got_sum - sum).abs() < 1e-6 * total as f64,
        "{name}: sum {got_sum} vs {sum}"
    );
}

#[test]
fn an_rgb_image_with_padding_on_both_edges_matches() {
    check(
        "rgb 83x121",
        oracle::RGB_83X121_SRC,
        oracle::RGB_83X121_GRID,
        oracle::RGB_83X121_SAMPLES,
        oracle::RGB_83X121_SUM,
    );
}

#[test]
fn an_rgba_image_composites_onto_the_checkerboard_like_the_reference() {
    check(
        "rgba 45x63",
        oracle::RGBA_45X63_SRC,
        oracle::RGBA_45X63_GRID,
        oracle::RGBA_45X63_SAMPLES,
        oracle::RGBA_45X63_SUM,
    );
}

#[test]
fn a_single_exact_tile_has_no_padding() {
    check(
        "rgb 40x40",
        oracle::RGB_40X40_SRC,
        oracle::RGB_40X40_GRID,
        oracle::RGB_40X40_SAMPLES,
        oracle::RGB_40X40_SUM,
    );
}
