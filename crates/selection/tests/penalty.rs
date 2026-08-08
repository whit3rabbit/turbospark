//! Repetition-penalty effect scenarios, observed through the composed
//! selection pipeline.
//!
//! Corresponds to behavior-spec test-006 (obl-select-006) and test-007
//! (obl-select-007).

use std::collections::HashMap;

use foundation::{LogitValue, LogitsView, TokenId};
use turbospark_selection::{select, ShapingConfig};

fn logits(values: &[f32]) -> Vec<LogitValue> {
    values.iter().map(|&v| LogitValue::from_f32(v)).collect()
}

fn frequency(config: &ShapingConfig, history: &[TokenId], trials: u64) -> HashMap<TokenId, u32> {
    let scores = logits(&[2.0, 2.0, 2.0, 2.0]);
    let view = LogitsView::new(&scores);
    let mut counts = HashMap::new();
    for position in 0..trials {
        let chosen = select(view, config, history, position).unwrap();
        *counts.entry(chosen).or_insert(0) += 1;
    }
    counts
}

#[test]
fn penalized_history_candidate_is_selected_materially_less_often() {
    let history = vec![0i32];
    let trials = 400;

    let unpenalized = ShapingConfig::new(1.0, 0, None, 1.0, Some(1)).unwrap();
    let penalized = ShapingConfig::new(1.0, 0, None, 4.0, Some(1)).unwrap();

    let baseline = frequency(&unpenalized, &history, trials);
    let attenuated = frequency(&penalized, &history, trials);

    let baseline_share = *baseline.get(&0).unwrap_or(&0);
    let attenuated_share = *attenuated.get(&0).unwrap_or(&0);
    assert!(
        attenuated_share < baseline_share,
        "expected penalty to reduce candidate 0's share: baseline {baseline_share}, attenuated {attenuated_share}"
    );
}

#[test]
fn identity_penalty_and_empty_history_are_both_no_ops() {
    let history: Vec<TokenId> = vec![0];
    let no_history: Vec<TokenId> = vec![];

    let identity = ShapingConfig::new(1.0, 0, None, 1.0, Some(1)).unwrap();
    let with_penalty_no_history = ShapingConfig::new(1.0, 0, None, 4.0, Some(1)).unwrap();

    let baseline = frequency(&identity, &no_history, 200);
    let with_identity_and_history = frequency(&identity, &history, 200);
    let with_penalty_but_no_history = frequency(&with_penalty_no_history, &no_history, 200);

    assert_eq!(with_identity_and_history, baseline);
    assert_eq!(with_penalty_but_no_history, baseline);
}
