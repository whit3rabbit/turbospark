//! Domain and error-guard scenarios.
//!
//! Corresponds to behavior-spec test-011 (obl-select-011) and the
//! destination-decided contracts for non-finite score entries, an empty
//! candidate domain, and extreme penalty-factor saturation
//! (obl-select-edge-001, obl-select-edge-002).

use foundation::{LogitValue, LogitsView, TokenId};
use turbospark_selection::{select, shaped_distribution, ShapingConfig};

fn logits(values: &[f32]) -> Vec<LogitValue> {
    values.iter().map(|&v| LogitValue::from_f32(v)).collect()
}

#[test]
fn a_rank_based_working_set_larger_than_the_candidate_domain_still_returns_a_valid_choice() {
    let scores = logits(&[1.0, 1.1, 0.9]);
    let view = LogitsView::new(&scores);
    let config = ShapingConfig::new(0.8, 250, Some(0.99), 1.0, Some(4)).unwrap();
    let history: Vec<TokenId> = vec![];

    for position in 0..20 {
        let chosen = select(view, &config, &history, position).unwrap();
        assert!((0..3).contains(&chosen));
    }
}

#[test]
fn a_non_finite_score_entry_is_rejected() {
    let scores = logits(&[1.0, f32::NAN, 0.5]);
    let view = LogitsView::new(&scores);
    let config = ShapingConfig::new(0.8, 0, None, 1.0, Some(1)).unwrap();
    let result = select(view, &config, &[], 0);
    assert!(result.is_err());
}

#[test]
fn an_empty_candidate_domain_is_rejected() {
    let scores: Vec<LogitValue> = vec![];
    let view = LogitsView::new(&scores);
    let config = ShapingConfig::new(0.8, 0, None, 1.0, Some(1)).unwrap();
    let result = select(view, &config, &[], 0);
    assert!(result.is_err());
}

#[test]
fn an_extreme_penalty_factor_saturates_without_producing_a_non_finite_score() {
    let scores = logits(&[10.0, -10.0, 0.0]);
    let view = LogitsView::new(&scores);
    let config = ShapingConfig::new(0.8, 0, None, 1_000_000.0, Some(1)).unwrap();
    let history: Vec<TokenId> = vec![0, 1];

    for position in 0..20 {
        let chosen = select(view, &config, &history, position).unwrap();
        assert!((0..3).contains(&chosen));
    }
}

/// Every probability over the shaped support must be finite and the whole
/// distribution must sum to one, or the sampler is drawing from garbage.
fn assert_finite_normalized(values: &[f32], penalty: f64, history: &[TokenId]) {
    let scores = logits(values);
    let view = LogitsView::new(&scores);
    let config = ShapingConfig::new(0.8, 0, None, penalty, Some(1)).unwrap();
    let dist = shaped_distribution(view, &config, history, 0).unwrap();
    let mut total = 0.0f64;
    for &token in dist.support() {
        let p = dist.probability(token);
        assert!(
            p.is_finite(),
            "penalty {penalty}: token {token} has p = {p}"
        );
        total += p;
    }
    assert!(
        (total - 1.0).abs() < 1e-9,
        "penalty {penalty}: mass sums to {total}"
    );
}

// A finite f64 penalty past f32::MAX used to cast to +inf: a zero score then
// became 0 * inf = NaN and poisoned the softmax sum. The value arrives
// unbounded from an HTTP `repetition_penalty` field.
#[test]
fn a_penalty_beyond_the_f32_range_saturates_without_producing_a_non_finite_score() {
    assert_finite_normalized(&[10.0, -10.0, 0.0], 1e300, &[0, 1, 2]);
}

// A tiny finite f64 penalty used to cast to 0.0 (or divide a positive score
// to +inf), which made the softmax max infinite and every exp NaN.
#[test]
fn a_penalty_below_the_f32_range_saturates_without_producing_a_non_finite_score() {
    assert_finite_normalized(&[10.0, -10.0, 0.0], 1e-300, &[0, 1, 2]);
    assert_finite_normalized(&[10.0, -10.0, 0.0], 1e-40, &[0, 1, 2]);
}

// Every candidate penalized negative past f32::MIN used to become -inf, so
// the softmax max was -inf and (-inf) - (-inf) produced NaN for all of them.
#[test]
fn a_penalty_that_overflows_every_score_negative_still_yields_a_distribution() {
    assert_finite_normalized(&[-10.0, -20.0], 1e38, &[0, 1]);
}
