//! Drama 3's sampler: top-k then top-p as the reference samples, and its
//! repetition-aware sampling (RAS) of the main token.
//!
//! Port of `sample_top_k_top_p` and `Model._step`'s RAS in mlx-audio
//! `fish_drama3/drama3.py` at [`super::REFERENCE_COMMIT`]. The filtered
//! distribution is the reference's; the draw is not. MLX's global random
//! state cannot be reproduced from Rust, so draws use the crate-local
//! MLX-compatible threefry keys (`crate::music::minimax_music3::rng`), the same
//! divergence `crate::tts::sampling` documents. A fixed seed fixes the Rust
//! sequence. Sampling owns normalization: callers pass raw logits.

use crate::music::minimax_music3::rng;
use crate::{Result, SpeechError};

/// The reference recipe's defaults.
pub const DEFAULT_TEMPERATURE: f32 = 0.7;
pub const DEFAULT_TOP_K: usize = 30;
pub const DEFAULT_TOP_P: f32 = 0.7;
/// Frames one chunk may generate.
pub const DEFAULT_MAX_FRAMES: usize = 1024;

/// A semantic main token that repeats one of the last `RAS_WINDOW` main
/// tokens is replaced by a second draw with these settings.
pub const RAS_WINDOW: usize = 16;
pub const RAS: SamplingParams = SamplingParams {
    temperature: 1.0,
    top_k: 30,
    top_p: 0.95,
};

/// Temperature is clamped up to this before dividing.
const MIN_TEMPERATURE: f32 = 1e-5;

/// One sampler's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl SamplingParams {
    /// The reference recipe (temperature 0.7, top-k 30, top-p 0.7).
    pub const REFERENCE: SamplingParams = SamplingParams {
        temperature: DEFAULT_TEMPERATURE,
        top_k: DEFAULT_TOP_K,
        top_p: DEFAULT_TOP_P,
    };

    /// Refuses settings the reference would fail on or that select nothing.
    pub fn validate(&self) -> Result<()> {
        let bad = |why: String| Err(SpeechError::Input { why });
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return bad(format!(
                "temperature {} must be finite and non-negative",
                self.temperature
            ));
        }
        if self.top_k == 0 {
            return bad("top_k must be at least 1".to_string());
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return bad(format!("top_p {} must be in (0, 1]", self.top_p));
        }
        Ok(())
    }
}

/// The candidates `sample_top_k_top_p` would draw from for one row, as
/// `(index, scaled logit)` in descending order.
///
/// The temperature applies before top-p. The cumulative probability runs
/// over the full vocabulary (the log-sum-exp is taken over every logit, not
/// only the top k), and the candidate whose running sum crosses `top_p` is
/// dropped unless it ranks first. Ties keep the lower index first.
pub fn candidates(logits: &[f32], params: SamplingParams) -> Vec<(usize, f32)> {
    if logits.is_empty() {
        return Vec::new();
    }
    let scale = 1.0 / params.temperature.max(MIN_TEMPERATURE);
    let scaled: Vec<f32> = logits.iter().map(|&v| v * scale).collect();
    // At low temperature the logits share a huge offset that would swamp
    // the log-sum-exp in f32 and push every probability to 1.
    let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let x: Vec<f32> = scaled.iter().map(|&v| v - max).collect();
    let lse = x.iter().map(|&v| v.exp()).sum::<f32>().ln();

    let k = params.top_k.min(x.len());
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by(|&a, &b| x[b].total_cmp(&x[a]).then(a.cmp(&b)));
    order.truncate(k);

    let mut cumulative = 0.0f32;
    let mut kept = Vec::with_capacity(k);
    for (rank, &index) in order.iter().enumerate() {
        cumulative += (x[index] - lse).exp();
        if rank > 0 && cumulative > params.top_p {
            // Every later rank's running sum is at least this one's.
            break;
        }
        kept.push((index, x[index]));
    }
    kept
}

/// One draw from [`candidates`]: an index into `logits`.
pub fn sample(logits: &[f32], params: SamplingParams, key: rng::Key) -> usize {
    let kept = candidates(logits, params);
    let values: Vec<f32> = kept.iter().map(|&(_, v)| v).collect();
    kept[rng::categorical(&values, key)].0
}

/// The frame's main token from the semantic head's logits, with RAS.
///
/// `logits` index `candidate_ids` (the semantic band, then EOS). Both draws
/// are always taken, as in the reference, so the key sequence does not
/// depend on whether the retry was used. `history` is the last
/// [`RAS_WINDOW`] main tokens (prompt rows' token column at the start).
pub fn main_token(
    logits: &[f32],
    candidate_ids: &[i32],
    params: SamplingParams,
    history: &[i32],
    semantic: std::ops::RangeInclusive<i32>,
    first_key: rng::Key,
    retry_key: rng::Key,
) -> i32 {
    let first = candidate_ids[sample(logits, params, first_key)];
    let retry = candidate_ids[sample(logits, RAS, retry_key)];
    if semantic.contains(&first) && history.contains(&first) {
        retry
    } else {
        first
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference's `TestSampler::test_low_temperature_keeps_tied_candidates`:
    /// four tied logits hold 0.25 each, so top_p 0.95 keeps the three ranks
    /// whose running sum stays below it.
    #[test]
    fn low_temperature_keeps_tied_candidates() {
        let logits = [1000.0f32, 1000.0, 1000.0, 1000.0, -50.0];
        let params = SamplingParams {
            temperature: 0.0,
            top_k: 30,
            top_p: 0.95,
        };
        let mut keys = rng::KeySequence::new(0);
        let draws: std::collections::BTreeSet<usize> = (0..200)
            .map(|_| sample(&logits, params, keys.next()))
            .collect();
        assert_eq!(draws.len(), 3);
        assert!(draws.iter().all(|&d| d < 4));
    }

    #[test]
    fn the_crossing_candidate_is_dropped_unless_it_ranks_first() {
        // Probabilities about 0.665, 0.245, 0.090.
        let logits = [2.0f32, 1.0, 0.0];
        let p = |top_p| SamplingParams {
            temperature: 1.0,
            top_k: 30,
            top_p,
        };
        let ids =
            |top_p| -> Vec<usize> { candidates(&logits, p(top_p)).iter().map(|c| c.0).collect() };
        assert_eq!(ids(0.5), [0], "rank 0 is kept although it crosses");
        assert_eq!(ids(0.7), [0], "rank 1 crosses 0.7 and is dropped");
        assert_eq!(ids(0.95), [0, 1]);
        assert_eq!(ids(1.0), [0, 1, 2]);
    }

    #[test]
    fn mass_runs_over_the_whole_vocabulary_not_the_top_k() {
        // Two strong logits and many weak ones: within the top 2 alone the
        // first would hold ~0.73, over the full row far less, so top_p 0.5
        // keeps the second only if the mass is taken over the full row.
        let mut logits = vec![0.0f32; 200];
        logits[0] = 3.0;
        logits[1] = 2.0;
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 2,
            top_p: 0.5,
        };
        let ids: Vec<usize> = candidates(&logits, params).iter().map(|c| c.0).collect();
        assert_eq!(ids, [0, 1]);
    }

    #[test]
    fn top_k_bounds_the_candidates() {
        let logits: Vec<f32> = (0..50).map(|i| i as f32 * 0.001).collect();
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 5,
            top_p: 1.0,
        };
        let ids: Vec<usize> = candidates(&logits, params).iter().map(|c| c.0).collect();
        assert_eq!(ids, [49, 48, 47, 46, 45]);
    }

    #[test]
    fn ras_replaces_only_a_repeated_semantic_token() {
        // Candidate 0 is semantic id 100, candidate 1 is EOS 7. Temperature
        // 0 makes the first draw deterministic.
        let ids = [100, 7];
        let greedy = SamplingParams {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
        };
        let mut keys = rng::KeySequence::new(3);
        let favour_semantic = [5.0f32, 0.0];
        let semantic = 100..=200;
        let fresh = main_token(
            &favour_semantic,
            &ids,
            greedy,
            &[1, 2],
            semantic.clone(),
            keys.next(),
            keys.next(),
        );
        assert_eq!(fresh, 100, "not in the history: kept");
        // Repeated: the retry draw (temperature 1, top-p 0.95) decides; with
        // these logits it can be either candidate, so assert it was drawn
        // from them rather than equal to `first` by construction.
        let favour_eos = [0.0f32, 5.0];
        let eos = main_token(
            &favour_eos,
            &ids,
            greedy,
            &[7, 7],
            semantic.clone(),
            keys.next(),
            keys.next(),
        );
        assert_eq!(eos, 7, "EOS is never replaced, even when repeated");
        // A repeat is redrawn under RAS's own top-p 0.95. With probabilities
        // about 0.39, 0.35 and 0.24 the first two stay under it and the third
        // crosses, so the retry lands on 100 or 7, never on 101. (With two
        // candidates the second would always cross 0.95 and be dropped.)
        let three = [100, 7, 101];
        let mut retried = std::collections::BTreeSet::new();
        for _ in 0..200 {
            retried.insert(main_token(
                &[1.0, 0.9, 0.5],
                &three,
                greedy,
                &[100],
                semantic.clone(),
                keys.next(),
                keys.next(),
            ));
        }
        assert_eq!(
            retried,
            [7, 100].into_iter().collect(),
            "a repeat is redrawn"
        );
    }

    #[test]
    fn validation_refuses_degenerate_settings() {
        assert!(SamplingParams::REFERENCE.validate().is_ok());
        for bad in [
            SamplingParams {
                top_k: 0,
                ..SamplingParams::REFERENCE
            },
            SamplingParams {
                top_p: 0.0,
                ..SamplingParams::REFERENCE
            },
            SamplingParams {
                top_p: 1.5,
                ..SamplingParams::REFERENCE
            },
            SamplingParams {
                temperature: f32::NAN,
                ..SamplingParams::REFERENCE
            },
            SamplingParams {
                temperature: -1.0,
                ..SamplingParams::REFERENCE
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }
}
