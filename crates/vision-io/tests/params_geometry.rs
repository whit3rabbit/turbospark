//! `PreprocessParams` fields are pub, so a caller can build one without the
//! config parser's positivity checks. Every pub entry point that divides by
//! `patch_size`, `merge_size` or their product must refuse a zero with an
//! error rather than panic on a division or remainder by zero.

use std::panic::{catch_unwind, AssertUnwindSafe};

use turbospark_vision_io::{
    patch_rows, pos_embed_weights, preprocess, resized_dims, vision_rope_freq_rows, GridThw,
    PreprocessParams, Rgb8Image,
};

fn params() -> PreprocessParams {
    let json = r#"{"patch_size":16,"temporal_patch_size":2,"merge_size":2,
        "size":{"shortest_edge":65536,"longest_edge":16777216},
        "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]}"#;
    PreprocessParams::from_preprocessor_config_json(json).expect("parses")
}

/// Each zeroed variant the parser would refuse.
fn zeroed() -> Vec<(&'static str, PreprocessParams)> {
    let mut patch = params();
    patch.patch_size = 0;
    let mut merge = params();
    merge.merge_size = 0;
    let mut temporal = params();
    temporal.temporal_patch_size = 0;
    vec![
        ("patch_size", patch),
        ("merge_size", merge),
        ("temporal_patch_size", temporal),
    ]
}

/// Ok(true) for an Err result, Ok(false) for an Ok result, Err on a panic.
fn refused<T>(
    f: impl FnOnce() -> Result<T, turbospark_vision_io::VisionIoError>,
) -> Result<bool, ()> {
    catch_unwind(AssertUnwindSafe(f))
        .map(|r| r.is_err())
        .map_err(|_| ())
}

#[test]
fn the_valid_params_pass_every_entry_point() {
    let p = params();
    assert!(p.check_geometry().is_ok());
    let pixels = vec![0.0f32; 32 * 32 * 3];
    assert!(patch_rows(&pixels, 32, 32, &p).is_ok());
    let grid = GridThw::new(1, 2, 2);
    assert!(vision_rope_freq_rows(grid, 64, 10_000.0, &p).is_ok());
    assert!(pos_embed_weights(grid, 4, &p).is_ok());
}

#[test]
fn a_zero_geometry_field_is_an_error_not_a_panic() {
    let image = Rgb8Image {
        width: 256,
        height: 256,
        data: vec![128; 256 * 256 * 3],
    };
    let pixels = vec![0.0f32; 32 * 32 * 3];
    let grid = GridThw::new(1, 2, 2);
    for (field, p) in zeroed() {
        assert!(p.check_geometry().is_err(), "{field}: check_geometry");
        assert_eq!(
            refused(|| patch_rows(&pixels, 32, 32, &p)),
            Ok(true),
            "{field}: patch_rows"
        );
        assert_eq!(
            refused(|| resized_dims(256, 256, &p)),
            Ok(true),
            "{field}: resized_dims"
        );
        assert_eq!(
            refused(|| vision_rope_freq_rows(grid, 64, 10_000.0, &p)),
            Ok(true),
            "{field}: vision_rope_freq_rows"
        );
        assert_eq!(
            refused(|| pos_embed_weights(grid, 4, &p)),
            Ok(true),
            "{field}: pos_embed_weights"
        );
        assert_eq!(
            refused(|| preprocess(&image, &p)),
            Ok(true),
            "{field}: preprocess"
        );
    }
}
