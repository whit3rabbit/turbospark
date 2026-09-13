//! Unit coverage for the pipeline steps the oracle fixture structurally
//! cannot discriminate, plus decode.
//!
//! Everything the reference CAN see is checked in `preprocess_parity.rs`
//! against real reference output. This file exists for the properties that
//! parity is blind to, and each test says which blindness it covers -- a test
//! here that duplicates a parity assertion is dead weight.

use turbospark_vision_io::{
    decode_image_bytes, normalize::rescale_and_normalize, params::PreprocessParams,
    resize::resample_pil, resize::resize_bicubic_pil, round_half_to_even, VisionIoError,
    MAX_IMAGE_DIM,
};

fn params_with(mean: [f32; 3], std: [f32; 3]) -> PreprocessParams {
    PreprocessParams {
        patch_size: 2,
        temporal_patch_size: 2,
        merge_size: 2,
        in_channels: 3,
        min_pixels: 16,
        max_pixels: 4096,
        image_mean: mean,
        image_std: std,
        rescale_factor: 1.0 / 255.0,
    }
}

// ---------------------------------------------------------------- normalize

#[test]
fn normalize_applies_mean_before_std_and_per_channel() {
    // COVERS A PARITY BLIND SPOT. Both real checkpoints declare
    // `image_mean == image_std == [0.5; 3]`, so in the oracle fixture the two
    // are interchangeable: swapping them, or applying the divide before the
    // subtract, is invisible there. Distinct values per channel make the
    // formula's shape observable.
    let params = params_with([0.1, 0.2, 0.3], [2.0, 4.0, 8.0]);
    let got = rescale_and_normalize(&[255, 255, 255, 0, 0, 0], &params);
    let want = [
        (1.0 - 0.1) / 2.0,
        (1.0 - 0.2) / 4.0,
        (1.0 - 0.3) / 8.0,
        (0.0 - 0.1) / 2.0,
        (0.0 - 0.2) / 4.0,
        (0.0 - 0.3) / 8.0,
    ];
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < 1e-7, "sample {i}: got {g}, want {w}");
    }
}

#[test]
fn the_shipped_mean_and_std_make_the_swap_invisible() {
    // The invariance itself, stated as a test so it is not rediscovered as a
    // finding: with mean == std the two are exchangeable and no fixture drawn
    // from these checkpoints can catch a swap.
    let params = params_with([0.5; 3], [0.5; 3]);
    let swapped = params_with(params.image_std, params.image_mean);
    let src: Vec<u8> = (0u8..=60).collect();
    assert_eq!(
        rescale_and_normalize(&src, &params),
        rescale_and_normalize(&src, &swapped)
    );
}

#[test]
fn the_shipped_normalization_is_the_documented_two_x_minus_one() {
    let params = params_with([0.5; 3], [0.5; 3]);
    let got = rescale_and_normalize(&[0, 128, 255], &params);
    for (i, (&g, px)) in got.iter().zip([0u8, 128, 255]).enumerate() {
        let want = 2.0 * (px as f32 / 255.0) - 1.0;
        assert!((g - want).abs() < 1e-7, "channel {i}: got {g}, want {want}");
    }
}

// ------------------------------------------------------------------- resize

/// A deterministic, non-flat test image, interleaved `(y, x, c)`.
fn ramp(h: usize, w: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(h * w * 3);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                out.push(((y * 29 + x * 53 + c * 97) % 251) as u8);
            }
        }
    }
    out
}

#[test]
fn an_identity_resize_is_the_identity_either_way() {
    // COVERS A PARITY BLIND SPOT, and settles what the early-out is for.
    // At `scale == 1` the half-pixel centre lands every sample on an integer
    // offset and the Keys kernel is zero at every nonzero integer, so the
    // weights collapse to `[0, 1, 0, 0]` and the filter path reproduces its
    // input exactly. The early-out is therefore a SPEED path, not a
    // correctness requirement -- which is why deleting it survives the parity
    // suite, and why that survivor is expected rather than a gap.
    for (h, w) in [(1, 1), (5, 7), (16, 16), (3, 40)] {
        let src = ramp(h, w);
        assert_eq!(
            resize_bicubic_pil(&src, w, h, w, h),
            src,
            "early-out {h}x{w}"
        );
        assert_eq!(resample_pil(&src, w, h, w, h), src, "filter path {h}x{w}");
    }
}

#[test]
fn a_flat_image_survives_every_resize_unchanged() {
    // The filter weights sum to one after PIL's clamp-then-renormalize edge
    // handling, so a constant image must come back constant -- INCLUDING at
    // the borders, which is precisely where renormalizing before the clamp
    // (or zero-padding the input) darkens the result. A border-only error is
    // a small fraction of the samples and easy to miss in a bulk comparison.
    let (h, w) = (9, 11);
    let src = vec![137u8; h * w * 3];
    for (nh, nw) in [(4, 4), (32, 8), (3, 27), (18, 22)] {
        let out = resize_bicubic_pil(&src, w, h, nw, nh);
        assert_eq!(out.len(), nh * nw * 3);
        assert!(
            out.iter().all(|&v| v == 137),
            "{nh}x{nw} moved a flat image: {:?}",
            &out[..out.len().min(12)]
        );
    }
}

#[test]
fn a_monotone_ramp_stays_monotone_across_a_resize() {
    // A per-row invariant that no golden fixture states: the filter is a
    // weighted local average, so a non-decreasing row must resample to a
    // non-decreasing row. A sign error in the kernel's negative lobes, or a
    // window read at the wrong offset, breaks this while leaving every value
    // in range and the output looking plausible.
    let (h, w) = (4, 32);
    let mut src = Vec::with_capacity(h * w * 3);
    for _y in 0..h {
        for x in 0..w {
            let v = (x * 8) as u8;
            src.extend_from_slice(&[v, v, v]);
        }
    }
    for nw in [8usize, 16, 64, 97] {
        let out = resize_bicubic_pil(&src, w, h, nw, h);
        for y in 0..h {
            for x in 1..nw {
                let (prev, cur) = (out[(y * nw + x - 1) * 3], out[(y * nw + x) * 3]);
                assert!(cur >= prev, "width {nw}, row {y}, x {x}: {prev} -> {cur}");
            }
        }
    }
}

// ----------------------------------------------------------------- rounding

#[test]
fn banker_rounding_differs_from_f64_round_where_it_must() {
    // The exact-.5 cases, called out individually because this is the one
    // helper whose whole reason to exist is disagreeing with the obvious
    // spelling.
    for (input, banker, away) in [
        (0.5, 0.0, 1.0),
        (1.5, 2.0, 2.0),
        (2.5, 2.0, 3.0),
        (3.5, 4.0, 4.0),
        (-2.5, -2.0, -3.0),
        (-0.5, -0.0, -1.0),
    ] {
        assert_eq!(
            round_half_to_even(input),
            banker,
            "round_half_to_even({input})"
        );
        assert_eq!(f64::round(input), away, "f64::round({input})");
    }
    // And it is a no-op away from .5, which is what keeps it from being a
    // second rounding rule.
    for input in [0.1, 2.4, 2.6, -7.3, 1e9 + 0.25] {
        assert_eq!(round_half_to_even(input), input.round(), "{input}");
    }
}

// ------------------------------------------------------------------- decode

/// A minimal in-memory PNG, so the test needs no fixture file.
fn png_bytes(width: u32, height: u32, color: image::Rgb<u8>) -> Vec<u8> {
    let buffer = image::RgbImage::from_pixel(width, height, color);
    let mut out = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(buffer)
        .write_to(&mut out, image::ImageFormat::Png)
        .expect("encodes");
    out.into_inner()
}

#[test]
fn a_png_round_trips_to_interleaved_rgb8() {
    let decoded = decode_image_bytes(&png_bytes(4, 3, image::Rgb([10, 20, 30]))).expect("decodes");
    assert_eq!((decoded.width, decoded.height), (4, 3));
    assert_eq!(decoded.data.len(), 4 * 3 * 3);
    assert_eq!(decoded.sample(2, 3, 1), 20);
    assert!(decoded.data.chunks_exact(3).all(|p| p == [10, 20, 30]));
}

#[test]
fn a_grayscale_image_is_broadcast_to_three_channels() {
    // The reference's `convert("RGB")`. A one-channel decode reaching the
    // patchifier would produce a third of the expected row width and fail
    // far from the cause.
    let gray = image::GrayImage::from_pixel(2, 2, image::Luma([77]));
    let mut encoded = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageLuma8(gray)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .expect("encodes");
    let decoded = decode_image_bytes(&encoded.into_inner()).expect("decodes");
    assert_eq!(decoded.data, vec![77u8; 2 * 2 * 3]);
}

#[test]
fn garbage_is_refused_with_a_decode_error() {
    let err = decode_image_bytes(b"not an image at all").expect_err("refused");
    assert!(matches!(err, VisionIoError::Decode { .. }), "{err}");
}

#[test]
fn the_dimension_cap_is_stated_in_the_error() {
    // The cap itself cannot be exercised through an ImageBuffer without a
    // large allocation,
    // so this pins the message shape and the constant instead: a reader who
    // hits it needs the limit in the text, not just the fact of refusal.
    let err = VisionIoError::ImageTooLarge {
        width: MAX_IMAGE_DIM + 1,
        height: 8,
        max_side: MAX_IMAGE_DIM,
    };
    let text = err.to_string();
    assert!(text.contains("4097") && text.contains("4096"), "{text}");
}

#[test]
fn dimensions_are_read_without_decoding_pixels() {
    let bytes = png_bytes(4, 3, image::Rgb([10, 20, 30]));
    assert_eq!(
        turbospark_vision_io::image_dimensions(&bytes).unwrap(),
        (4, 3)
    );
}

#[test]
fn an_oversized_png_is_refused_from_its_header() {
    let mut bytes = png_bytes(1, 1, image::Rgb([0, 0, 0]));
    bytes[16..20].copy_from_slice(&4097u32.to_be_bytes());
    let mut crc = 0xffff_ffffu32;
    for &byte in &bytes[12..29] {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & 0u32.wrapping_sub(crc & 1));
        }
    }
    bytes[29..33].copy_from_slice(&(!crc).to_be_bytes());

    let err = turbospark_vision_io::image_dimensions(&bytes).unwrap_err();
    assert!(matches!(err, VisionIoError::Decode { .. }), "{err}");
}

/// **`rescale_factor` IS OPTIONAL AND THE REAL CHECKPOINT OMITS IT.**
///
/// `mlx-community/Qwen3.8-27B-4bit`'s `preprocessor_config.json` carries
/// `size`, `patch_size`, `merge_size`, `temporal_patch_size`, `image_mean`
/// and `image_std` -- and no `rescale_factor`. This crate REQUIRED it until
/// the first real `--image` run refused a valid install by name.
///
/// The default is the reference processor's own signature default
/// (`processing_qwen3_vl.py:162`), not a guess: it is the universal 0-255 to
/// 0-1 conversion. Contrast the pixel budget, which stays required because
/// the generic library default is wrong for this family by a factor of 16
/// (crate Gotcha 6).
#[test]
fn an_absent_rescale_factor_takes_the_references_own_default() {
    let json = r#"{
        "size": {"shortest_edge": 65536, "longest_edge": 16777216},
        "patch_size": 16, "merge_size": 2, "temporal_patch_size": 2,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]
    }"#;
    let params = PreprocessParams::from_preprocessor_config_json(json).expect("parses");
    assert_eq!(
        params.rescale_factor,
        turbospark_vision_io::DEFAULT_RESCALE_FACTOR as f32
    );
    assert_eq!(params.rescale_factor, 1.0 / 255.0);
}

/// A DECLARED value still wins, so the default cannot mask a checkpoint that
/// means something else.
#[test]
fn a_declared_rescale_factor_beats_the_default() {
    let json = r#"{
        "size": {"shortest_edge": 65536, "longest_edge": 16777216},
        "patch_size": 16, "merge_size": 2, "temporal_patch_size": 2,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        "rescale_factor": 0.5
    }"#;
    let params = PreprocessParams::from_preprocessor_config_json(json).expect("parses");
    assert_eq!(params.rescale_factor, 0.5);
}

/// The pixel budget is still REFUSED when absent, which is the asymmetry the
/// default above must not erode.
#[test]
fn an_absent_pixel_budget_is_still_refused() {
    let json = r#"{
        "patch_size": 16, "merge_size": 2, "temporal_patch_size": 2,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]
    }"#;
    assert!(PreprocessParams::from_preprocessor_config_json(json).is_err());
}
