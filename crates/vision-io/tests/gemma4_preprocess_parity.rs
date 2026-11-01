//! Gemma 4 preprocessing against mlx-vlm's `Gemma4ImageProcessor`.
//!
//! The reference emits `(C, H, W)` pixels of the resized image and leaves
//! patchify to the model. This crate emits patch rows. The comparison maps
//! each reference sample to its patch row by index arithmetic in
//! [`patch_index`], never by reshaping, and the fixture is non-square so a
//! swapped x and y cannot pass.

mod oracle {
    include!("generated/gemma4_preprocess_oracle.rs");
}

use turbospark_vision_io::{
    gemma4_preprocess, gemma4_resize_dims, Gemma4Patches, Rgb8Image, GEMMA4_PATCH,
    GEMMA4_PATCH_DIM, GEMMA4_POOL,
};

/// Where the reference's `(c, y, x)` sample sits in [`Gemma4Patches::pixels`].
fn patch_index(chw: usize, h: usize, w: usize) -> usize {
    let c = chw / (h * w);
    let y = (chw % (h * w)) / w;
    let x = chw % w;
    let grid_w = w / GEMMA4_PATCH;
    let patch = (y / GEMMA4_PATCH) * grid_w + x / GEMMA4_PATCH;
    let (r, col) = (y % GEMMA4_PATCH, x % GEMMA4_PATCH);
    patch * GEMMA4_PATCH_DIM + (r * GEMMA4_PATCH + col) * 3 + c
}

#[test]
fn resize_dims_match_the_reference() {
    for &(h, w, budget, oh, ow) in oracle::RESIZE_DIMS {
        let got = gemma4_resize_dims(h, w, budget * GEMMA4_POOL * GEMMA4_POOL).unwrap();
        assert_eq!(got, (oh, ow), "{h}x{w} at {budget} soft tokens");
    }
}

/// One reference image and what the reference made of it.
struct Case {
    name: &'static str,
    src_hw: (usize, usize),
    src: &'static [u8],
    budget: usize,
    out_hw: (usize, usize),
    soft: usize,
    stride: usize,
    samples: &'static [f32],
}

fn check(c: Case) {
    let name = c.name;
    let img = Rgb8Image::new(c.src_hw.1, c.src_hw.0, c.src.to_vec()).unwrap();
    let p: Gemma4Patches = gemma4_preprocess(&img, c.budget).unwrap();
    let (h, w) = c.out_hw;
    assert_eq!(
        (p.grid_h * GEMMA4_PATCH, p.grid_w * GEMMA4_PATCH),
        (h, w),
        "{name}"
    );
    assert_eq!(p.soft_tokens(), c.soft, "{name}");
    assert_eq!(c.samples.len(), (3 * h * w).div_ceil(c.stride), "{name}");
    let mut worst = 0.0f32;
    for (i, &want) in c.samples.iter().enumerate() {
        let got = p.pixels[patch_index(i * c.stride, h, w)];
        assert!(got.is_finite(), "{name}: non-finite sample {i}");
        worst = worst.max((got - want).abs());
    }
    // Both sides resample to the same u8 and divide by 255: the only slack is
    // the f32 division itself.
    assert!(worst <= 1e-6, "{name}: worst abs diff {worst}");
}

#[test]
fn a_wide_image_matches_the_reference() {
    check(Case {
        name: "wide",
        src_hw: oracle::WIDE_SRC_HW,
        src: oracle::WIDE_SRC,
        budget: oracle::WIDE_BUDGET,
        out_hw: oracle::WIDE_OUT_HW,
        soft: oracle::WIDE_SOFT_TOKENS,
        stride: oracle::WIDE_STRIDE,
        samples: oracle::WIDE_CHW_SAMPLES,
    });
}

#[test]
fn a_tall_image_matches_the_reference() {
    check(Case {
        name: "tall",
        src_hw: oracle::TALL_SRC_HW,
        src: oracle::TALL_SRC,
        budget: oracle::TALL_BUDGET,
        out_hw: oracle::TALL_OUT_HW,
        soft: oracle::TALL_SOFT_TOKENS,
        stride: oracle::TALL_STRIDE,
        samples: oracle::TALL_CHW_SAMPLES,
    });
}

/// The mapping must not collapse to the identity on this fixture, or the
/// comparisons above would prove nothing about patch order.
#[test]
fn the_patch_mapping_is_not_the_identity() {
    let (h, w) = oracle::WIDE_OUT_HW;
    let moved = (0..3 * h * w)
        .filter(|&i| patch_index(i, h, w) != i)
        .count();
    assert!(moved > (3 * h * w) / 2);
}
