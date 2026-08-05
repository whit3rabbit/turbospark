//! Deterministic-selection and temperature-spread scenarios.
//!
//! Corresponds to behavior-spec test-001 (obl-select-001) and test-009
//! (obl-select-009).

use std::collections::HashSet;

use foundation::{LogitValue, LogitsView, TokenId};
use mrefrust_selection::{select, ShapingConfig};

fn logits(values: &[f32]) -> Vec<LogitValue> {
    values.iter().map(|&v| LogitValue::from_f32(v)).collect()
}

#[test]
fn zero_temperature_always_returns_the_highest_scoring_candidate() {
    let scores = logits(&[0.1, 0.2, 5.0, 0.3, 0.15]);
    let view = LogitsView::new(&scores);
    // Truncation and seed are configured but must be ignored on this path.
    let config = ShapingConfig::new(0.0, 3, Some(0.5), 1.0, Some(123)).unwrap();
    let history: Vec<TokenId> = vec![];

    for position in 0..10 {
        let chosen = select(view, &config, &history, position).unwrap();
        assert_eq!(chosen, 2);
    }
}

#[test]
fn raised_temperature_spreads_selection_across_multiple_candidates() {
    let scores = logits(&[0.1, 0.2, 5.0, 0.3, 0.15]);
    let view = LogitsView::new(&scores);
    let config = ShapingConfig::new(2.0, 0, None, 1.0, Some(9)).unwrap();
    let history: Vec<TokenId> = vec![];

    let mut dominant_hits = 0u32;
    let mut distinct: HashSet<TokenId> = HashSet::new();
    let trials = 300;
    for position in 0..trials {
        let chosen = select(view, &config, &history, position).unwrap();
        distinct.insert(chosen);
        if chosen == 2 {
            dominant_hits += 1;
        }
    }

    assert!(
        distinct.len() > 1,
        "expected temperature to produce more than one distinct candidate"
    );
    assert!(
        dominant_hits < trials as u32,
        "dominant candidate should not win every single trial under a raised temperature"
    );
}
