//! Domain and error-guard scenarios.
//!
//! Corresponds to behavior-spec test-011 (obl-select-011) and the
//! destination-decided contracts for non-finite score entries, an empty
//! candidate domain, and extreme penalty-factor saturation
//! (obl-select-edge-001, obl-select-edge-002).

use foundation::{LogitValue, LogitsView, TokenId};
use mrefrust_selection::{select, ShapingConfig};

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
