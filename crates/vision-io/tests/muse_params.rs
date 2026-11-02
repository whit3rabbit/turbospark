//! The flat spelling of Muse-Glimmer's image-processor config selects the
//! token-cap rule, and `preprocess` follows it end to end.

use turbospark_vision_io::{preprocess, PreprocessParams, ResizeRule, Rgb8Image};

const MUSE: &str = r#"{
  "image_mean": [0.5, 0.5, 0.5],
  "image_processor_type": "MuseGlimmerImageProcessor",
  "image_std": [0.5, 0.5, 0.5],
  "max_image_tokens": 4096,
  "merge_size": 2,
  "patch_size": 14,
  "rescale_factor": 0.00392156862745098,
  "temporal_patch_size": 2
}"#;

#[test]
fn max_image_tokens_selects_the_muse_rule_and_derives_the_pixel_pair() {
    let p = PreprocessParams::from_preprocessor_config_json(MUSE).expect("parses");
    assert_eq!(p.resize, ResizeRule::MuseTokens { max_tokens: 4096 });
    assert_eq!(
        (p.patch_size, p.merge_size, p.temporal_patch_size),
        (14, 2, 2)
    );
    assert_eq!(p.max_pixels, 4096 * 28 * 28);
    assert_eq!(p.image_mean, [0.5; 3]);
}

#[test]
fn a_qwen_config_keeps_the_qwen_rule() {
    let json = r#"{"patch_size":16,"temporal_patch_size":2,"merge_size":2,
        "size":{"shortest_edge":65536,"longest_edge":16777216},
        "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]}"#;
    let p = PreprocessParams::from_preprocessor_config_json(json).expect("parses");
    assert_eq!(p.resize, ResizeRule::QwenPixels);
}

#[test]
fn a_zero_token_cap_is_refused() {
    let json = MUSE.replace("4096", "0");
    assert!(PreprocessParams::from_preprocessor_config_json(&json).is_err());
}

#[test]
fn preprocess_follows_the_rule_and_the_memory_clamp_lowers_the_cap() {
    let mut p = PreprocessParams::from_preprocessor_config_json(MUSE).unwrap();
    // 560x700 is already a multiple of 28: no resize, a 50x40 patch grid.
    let data = vec![128u8; 560 * 700 * 3];
    let image = Rgb8Image::new(560, 700, data).unwrap();
    let out = preprocess(&image, &p).unwrap();
    assert_eq!((out.grid.t, out.grid.h, out.grid.w), (1, 50, 40));
    assert_eq!(out.merged_tokens, 500);
    assert_eq!(out.patch_rows.len(), 2000 * 1176);
    // A flat mid-grey normalizes to ~0 with mean and std 0.5.
    assert!(out.patch_rows.iter().all(|v| v.abs() < 0.01));

    // The session's budget clamp lowers max_pixels; the cap follows it.
    p.max_pixels = 100 * 28 * 28;
    let small = preprocess(&image, &p).unwrap();
    assert!(small.merged_tokens <= 100, "{}", small.merged_tokens);
}
