//! Both rope row builders are pub and take `theta` as a plain f32. A zero,
//! negative or non-finite theta turns the frequency ladder into inf or NaN
//! rows that the tower would rotate by silently, so both must refuse it.

use turbospark_vision_io::{
    pixtral_rope_freq_rows, vision_rope_freq_rows, GridThw, PreprocessParams,
};

fn params() -> PreprocessParams {
    let json = r#"{"patch_size":16,"temporal_patch_size":2,"merge_size":2,
        "size":{"shortest_edge":65536,"longest_edge":16777216},
        "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]}"#;
    PreprocessParams::from_preprocessor_config_json(json).expect("parses")
}

const BAD: [f32; 5] = [0.0, -1.0, -10_000.0, f32::NAN, f32::INFINITY];

#[test]
fn a_valid_theta_gives_finite_rows() {
    let rows = vision_rope_freq_rows(GridThw::new(1, 2, 2), 64, 10_000.0, &params()).unwrap();
    assert!(rows.iter().all(|v| v.is_finite()));
    let rows = pixtral_rope_freq_rows(2, 2, 64, 10_000.0).unwrap();
    assert!(rows.iter().all(|v| v.is_finite()));
}

#[test]
fn qwen_rope_refuses_a_bad_theta() {
    for theta in BAD {
        assert!(
            vision_rope_freq_rows(GridThw::new(1, 2, 2), 64, theta, &params()).is_err(),
            "theta {theta}"
        );
    }
}

#[test]
fn pixtral_rope_refuses_a_bad_theta() {
    for theta in BAD {
        assert!(
            pixtral_rope_freq_rows(2, 2, 64, theta).is_err(),
            "theta {theta}"
        );
    }
}
