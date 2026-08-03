//! Repetition-penalty attenuation of the raw score vector.
//!
//! Runs before the score-to-probability conversion and before either
//! truncation mode or the temperature reweighting.

use foundation::TokenId;
use std::collections::HashSet;

/// Attenuate the score of every distinct candidate identifier present in
/// `history` exactly once, regardless of how often it repeats in history.
///
/// A documented no-op when `penalty` is the identity value or `history` is
/// empty: the score vector is left unmodified in that case.
pub fn apply_repetition_penalty(scores: &mut [f32], history: &[TokenId], penalty: f64) {
    if penalty == 1.0 || history.is_empty() {
        return;
    }
    let seen: HashSet<TokenId> = history.iter().copied().collect();
    let penalty = penalty as f32;
    for &id in &seen {
        let Ok(idx) = usize::try_from(id) else {
            continue;
        };
        if idx >= scores.len() {
            continue;
        }
        let s = scores[idx];
        scores[idx] = if s > 0.0 { s / penalty } else { s * penalty };
    }
}

#[cfg(test)]
mod tests {
    //! Light inline checks; frequency-based penalty-effect coverage lives in
    //! the integration test suite.
    use super::*;

    #[test]
    fn identity_penalty_is_a_no_op() {
        let mut scores = [1.0f32, -1.0, 2.0];
        let original = scores;
        apply_repetition_penalty(&mut scores, &[0, 2], 1.0);
        assert_eq!(scores, original);
    }

    #[test]
    fn empty_history_is_a_no_op() {
        let mut scores = [1.0f32, -1.0, 2.0];
        let original = scores;
        apply_repetition_penalty(&mut scores, &[], 4.0);
        assert_eq!(scores, original);
    }

    #[test]
    fn a_repeated_history_entry_is_attenuated_only_once() {
        let mut scores = [4.0f32];
        apply_repetition_penalty(&mut scores, &[0, 0, 0], 2.0);
        assert_eq!(scores[0], 2.0);
    }

    #[test]
    fn positive_scores_divide_and_negative_scores_multiply_by_the_penalty() {
        let mut scores = [4.0f32, -4.0];
        apply_repetition_penalty(&mut scores, &[0, 1], 2.0);
        assert_eq!(scores[0], 2.0);
        assert_eq!(scores[1], -8.0);
    }
}
