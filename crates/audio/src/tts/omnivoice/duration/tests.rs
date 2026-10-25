//! Duration-estimator parity: spot checks for the weight rules plus the
//! recorded fixture from `tools/gen_omnivoice_duration_fixture.py`
//! (the reference `RuleDurationEstimator` at the pinned mlx-audio
//! commit, over mixed-script texts and boundary codepoints).

use super::{char_weight, estimate_duration, total_weight};

fn fixture() -> serde_json::Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/tts_omnivoice/duration_fixture.json");
    let text = std::fs::read_to_string(&path).expect("duration fixture");
    serde_json::from_str(&text).expect("duration fixture json")
}

#[test]
fn spot_weights_follow_the_reference_table() {
    assert_eq!(char_weight('a'), 1.0);
    assert_eq!(char_weight('Z'), 1.0);
    assert_eq!(char_weight(' '), 0.2);
    assert_eq!(char_weight('.'), 0.5);
    assert_eq!(char_weight('5'), 3.5);
    assert_eq!(char_weight('中'), 3.0);
    assert_eq!(char_weight('あ'), 2.2);
    assert_eq!(char_weight('한'), 2.5);
    assert_eq!(char_weight('\u{0640}'), 0.0); // tatweel
    assert_eq!(char_weight('\u{0301}'), 0.0); // combining acute
    assert_eq!(char_weight('\u{05d0}'), 1.5); // Hebrew alef
    assert_eq!(char_weight('\u{0627}'), 1.5); // Arabic alef
    assert_eq!(char_weight('\u{0e01}'), 1.5); // Thai ko kai
    assert_eq!(char_weight('\u{0966}'), 3.5); // Devanagari zero (Nd)
    assert_eq!(char_weight('\u{ff10}'), 3.5); // fullwidth zero (Nd)
    assert_eq!(char_weight('\u{ff01}'), 0.5); // fullwidth bang (Po)
    assert_eq!(char_weight('\u{2af}'), 1.0); // last Latin range row
    assert_eq!(char_weight('\u{2b0}'), 1.0); // modifier letters (Lm)
    assert_eq!(char_weight('\u{a000}'), 3.0); // Yi syllable
                                              // U+20000 itself is not `> 0x20000` in the reference; only the
                                              // codepoints above it take the CJK weight.
    assert_eq!(char_weight('\u{20000}'), 1.0);
    assert_eq!(char_weight('\u{20001}'), 3.0); // CJK Ext B
    assert_eq!(char_weight('\u{1d400}'), 1.0); // mathematical bold A
}

#[test]
fn total_weight_sums_in_reference_order() {
    // "Nice to meet you." is the reference fallback calibration text:
    // 14 letters + 3 spaces + 1 period.
    let weight = total_weight("Nice to meet you.");
    assert!((weight - 14.1).abs() < 1e-12, "weight {weight}");
}

#[test]
fn estimate_matches_the_recorded_reference() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().expect("fixture cases");
    assert!(cases.len() >= 12, "fixture must cover a spread of texts");
    for case in cases {
        let target = case["target"].as_str().expect("target");
        let reference = case["ref"].as_str().expect("ref");
        let duration = case["ref_duration"].as_f64().expect("ref duration");
        let low = case["low_threshold"].as_f64();
        let boost = case["boost_strength"].as_f64().unwrap_or(3.0);
        let expected = case["estimate_bits"].as_u64().expect("estimate bits");
        let got = estimate_duration(target, reference, duration, low, boost);
        assert_eq!(
            got.to_bits(),
            expected,
            "case {:?} estimate bits {:#x} != reference {:#x}",
            target,
            got.to_bits(),
            expected
        );
        let want_weight = case["target_weight_bits"].as_u64().expect("weight bits");
        assert_eq!(total_weight(target).to_bits(), want_weight);
        // The readable float is present for review only.
        assert!(case["estimate"].as_f64().is_some());
    }
}

#[test]
fn empty_reference_and_bad_duration_return_zero() {
    assert_eq!(estimate_duration("hello", "", 25.0, None, 3.0), 0.0);
    assert_eq!(estimate_duration("hello", "abc", 0.0, None, 3.0), 0.0);
    assert_eq!(estimate_duration("hello", "abc", -1.0, None, 3.0), 0.0);
}
