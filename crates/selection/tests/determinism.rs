//! Seeded reproducibility scenarios.
//!
//! Corresponds to behavior-spec test-002 (obl-select-002) and test-003
//! (obl-select-003).

use foundation::{LogitValue, LogitsView, TokenId};
use turbospark_selection::{select, ShapingConfig};

fn logits(values: &[f32]) -> Vec<LogitValue> {
    values.iter().map(|&v| LogitValue::from_f32(v)).collect()
}

#[test]
fn identical_seed_score_history_and_position_agree() {
    let scores = logits(&[1.0, 1.05, 0.9, 1.1, 0.95, 1.0, 0.8, 1.2]);
    let view = LogitsView::new(&scores);
    let config = ShapingConfig::new(0.8, 0, None, 1.0, Some(7)).unwrap();
    let history: Vec<TokenId> = vec![];

    let a = select(view, &config, &history, 3).unwrap();
    let b = select(view, &config, &history, 3).unwrap();
    assert_eq!(a, b);
}

#[test]
fn replaying_an_increasing_sequence_of_positions_matches_element_by_element() {
    let per_position_scores: Vec<Vec<LogitValue>> = (0..6)
        .map(|i| logits(&[1.0, 1.05 + i as f32 * 0.01, 0.9, 1.1, 0.95, 1.0]))
        .collect();
    let config = ShapingConfig::new(0.8, 0, None, 1.0, Some(99)).unwrap();
    let history: Vec<TokenId> = vec![];

    let run = |series: &[Vec<LogitValue>]| -> Vec<TokenId> {
        series
            .iter()
            .enumerate()
            .map(|(pos, scores)| {
                let view = LogitsView::new(scores);
                select(view, &config, &history, pos as u64).unwrap()
            })
            .collect()
    };

    let first_run = run(&per_position_scores);
    let second_run = run(&per_position_scores);
    assert_eq!(first_run, second_run);
}
