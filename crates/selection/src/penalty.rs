//! Repetition, presence, and frequency penalty attenuation of the raw score
//! vector.
//!
//! Runs before the score-to-probability conversion and before either
//! truncation mode or the temperature reweighting.

use foundation::TokenId;
use std::collections::HashMap;

/// Attenuate the score of every distinct candidate identifier present in
/// `history` exactly once, regardless of how often it repeats in history.
///
/// A documented no-op when `penalty` is the identity value or `history` is
/// empty: the score vector is left unmodified in that case.
pub fn apply_repetition_penalty(scores: &mut [f32], history: &[TokenId], penalty: f64) {
    if penalty == 1.0 || history.is_empty() {
        return;
    }
    // One bit per candidate, not a hash set of the history: this runs once per
    // generated token over the whole context, and hashing a 128K history cost
    // about 1.7 ms per token (2026-11, M4 Max). Out-of-range ids set no bit.
    let mut seen = vec![0u64; scores.len().div_ceil(64)];
    for &id in history {
        match usize::try_from(id) {
            Ok(idx) if idx < scores.len() => seen[idx / 64] |= 1 << (idx % 64),
            _ => {}
        }
    }
    // `ShapingConfig` accepts any finite positive f64, and a request can send
    // one outside the f32 range. Unclamped, the cast gives inf (0 * inf is
    // NaN) or 0 (s / 0 is inf), and both poison the softmax. Clamping is a
    // no-op, bit for bit, for every penalty already representable as a
    // normal f32.
    let penalty = (penalty as f32).clamp(f32::MIN_POSITIVE, f32::MAX);
    for (word, &bits) in seen.iter().enumerate() {
        let mut bits = bits;
        while bits != 0 {
            let idx = word * 64 + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let s = scores[idx];
            let attenuated = if s > 0.0 { s / penalty } else { s * penalty };
            // Saturate rather than overflow: a score of -inf on every candidate
            // makes the softmax max -inf and every exp NaN.
            scores[idx] = attenuated.clamp(f32::MIN, f32::MAX);
        }
    }
}

/// Apply OpenAI-style presence and frequency penalties over the GENERATED
/// suffix of history only -- `generated_suffix`, which a caller derives as
/// `&history[history.len() - generated..]` -- never the prompt tokens ahead
/// of it.
///
/// **THIS IS llama.cpp'S CONVENTION, NOT OPENAI'S OWN**, and the difference
/// is deliberate rather than an oversight: OpenAI penalizes over the whole
/// context INCLUDING the prompt, which would let a caller's own input text
/// suppress tokens the model needs to quote or continue verbatim (a system
/// prompt containing the word the caller most wants repeated, say). Counting
/// only what THIS turn already generated is the same reasoning
/// `apply_repetition_penalty` was never extended to touch the prompt with.
/// See `DEVIATIONS.md` for the full rationale.
///
/// `logit -= presence_penalty * 1[count > 0] + frequency_penalty * count`
///
/// A documented no-op when both penalties are their identity value (`0.0`)
/// or the generated suffix is empty.
pub fn apply_presence_and_frequency_penalty(
    scores: &mut [f32],
    generated_suffix: &[TokenId],
    presence_penalty: f64,
    frequency_penalty: f64,
) {
    if (presence_penalty == 0.0 && frequency_penalty == 0.0) || generated_suffix.is_empty() {
        return;
    }
    let mut counts: HashMap<TokenId, u32> = HashMap::new();
    for &id in generated_suffix {
        *counts.entry(id).or_insert(0) += 1;
    }
    for (id, count) in counts {
        let Ok(idx) = usize::try_from(id) else {
            continue;
        };
        if idx >= scores.len() {
            continue;
        }
        let penalty = presence_penalty + frequency_penalty * count as f64;
        scores[idx] -= penalty as f32;
    }
}

#[cfg(test)]
mod tests {
    //! Light inline checks; frequency-based penalty-effect coverage lives in
    //! the integration test suite.
    use super::*;

    /// The bitset walk must touch exactly the set the history names: bit
    /// positions across several words, repeats, ids past the vocabulary and
    /// negative ids.
    #[test]
    fn bitset_matches_a_per_candidate_membership_scan() {
        let len = 200usize;
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for round in 0..50 {
            let history: Vec<TokenId> = (0..round * 3)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state % 260) as TokenId - 20
                })
                .chain([0, 63, 64, 127, 128, 199, 200])
                .collect();
            let mut scores: Vec<f32> = (0..len).map(|i| i as f32 - 100.0).collect();
            let want: Vec<f32> = scores
                .iter()
                .enumerate()
                .map(|(i, &s)| match history.contains(&(i as TokenId)) {
                    true if s > 0.0 => s / 1.5,
                    true => s * 1.5,
                    false => s,
                })
                .collect();
            apply_repetition_penalty(&mut scores, &history, 1.5);
            assert_eq!(scores, want, "round {round}");
        }
    }

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

    #[test]
    fn zero_presence_and_frequency_penalty_is_a_no_op() {
        let mut scores = [1.0f32, -1.0, 2.0];
        let original = scores;
        apply_presence_and_frequency_penalty(&mut scores, &[0, 0, 2], 0.0, 0.0);
        assert_eq!(scores, original);
    }

    #[test]
    fn an_empty_generated_suffix_is_a_no_op() {
        let mut scores = [1.0f32, -1.0, 2.0];
        let original = scores;
        apply_presence_and_frequency_penalty(&mut scores, &[], 0.5, 0.5);
        assert_eq!(scores, original);
    }

    /// **THE DISCRIMINATING TEST FOR PRESENCE VS FREQUENCY.** A token that
    /// appears three times is penalized ONCE by presence (flat, regardless
    /// of count) and THREE TIMES by frequency (scaled by count) -- the two
    /// knobs would be indistinguishable on a fixture that never repeats a
    /// token.
    #[test]
    fn presence_penalty_applies_once_regardless_of_repeat_count() {
        let mut scores = [10.0f32];
        apply_presence_and_frequency_penalty(&mut scores, &[0, 0, 0], 1.5, 0.0);
        assert_eq!(scores[0], 8.5);
    }

    #[test]
    fn frequency_penalty_scales_with_repeat_count() {
        let mut scores = [10.0f32];
        apply_presence_and_frequency_penalty(&mut scores, &[0, 0, 0], 0.0, 1.5);
        assert_eq!(scores[0], 10.0 - 1.5 * 3.0);
    }

    #[test]
    fn both_penalties_compose_additively() {
        let mut scores = [10.0f32];
        apply_presence_and_frequency_penalty(&mut scores, &[0, 0], 1.0, 0.5);
        // presence: -1.0 (once) + frequency: -0.5 * 2 (twice) = -2.0
        assert_eq!(scores[0], 8.0);
    }

    #[test]
    fn a_token_that_never_appears_is_untouched() {
        let mut scores = [10.0f32, 10.0];
        apply_presence_and_frequency_penalty(&mut scores, &[0, 0], 1.0, 1.0);
        assert_eq!(scores[1], 10.0);
    }

    #[test]
    fn a_negative_penalty_boosts_rather_than_suppresses() {
        let mut scores = [10.0f32];
        apply_presence_and_frequency_penalty(&mut scores, &[0], -0.5, 0.0);
        assert_eq!(scores[0], 10.5);
    }
}
