//! Paired A/B statistics and the verdict rule.
//!
//! The unit is a PAIR: one A run and one B run taken back to back in rotated
//! order, so slow thermal and page-cache drift cancels within the pair. Each
//! pair gives one relative improvement of B over A on the primary metric,
//! signed so positive always means B is better whatever the metric's
//! direction.
//!
//! The rule is fixed in code and copied into every verdict record. An agent
//! iterating against this evaluator cannot loosen it from a spec; a spec may
//! only raise `min_effect`.

use serde::{Deserialize, Serialize};

/// Whether larger values of the metric are better (tok/s) or worse (seconds,
/// ms per token, bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    HigherIsBetter,
    LowerIsBetter,
}

/// The relative improvement of `b` over `a`, positive when `b` is better.
/// `None` when `a` is not a positive finite number.
pub fn improvement(a: f64, b: f64, direction: Direction) -> Option<f64> {
    if !(a.is_finite() && b.is_finite() && a > 0.0) {
        return None;
    }
    Some(match direction {
        Direction::HigherIsBetter => (b - a) / a,
        Direction::LowerIsBetter => (a - b) / a,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// B is better by at least `min_effect`, and the evidence agrees.
    Win,
    /// B is worse by at least `min_effect`, and the evidence agrees.
    Loss,
    /// Enough clean pairs, no effect at `min_effect`.
    Noise,
    /// Too few clean pairs to say anything.
    Inconclusive,
}

/// Minimum clean pairs for any WIN or LOSS.
pub const MIN_PAIRS: usize = 3;
/// Below this many pairs, every pair must agree in sign; at or above it, the
/// one-sided exact sign test must reach [`SIGN_TEST_ALPHA`].
pub const UNANIMITY_BELOW: usize = 6;
pub const SIGN_TEST_ALPHA: f64 = 0.05;
/// The floor under any spec's `min_effect`. Run-to-run spread on this host is
/// about 1.5 tok/s on decode (`docs/BENCHMARKING.md`), so a smaller claimed
/// effect would be a claim about noise.
pub const MIN_EFFECT_FLOOR: f64 = 0.02;
pub const BOOTSTRAP_RESAMPLES: usize = 10_000;
pub const BOOTSTRAP_SEED: u64 = 0x5eed_7a5b_2026_1022;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerdictReport {
    pub verdict: Verdict,
    pub pairs: usize,
    pub improvements: Vec<f64>,
    pub median: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub positives: usize,
    pub negatives: usize,
    /// One-sided exact sign-test p for "B better" and for "B worse".
    pub p_better: f64,
    pub p_worse: f64,
    pub min_effect: f64,
    /// Why the verdict is what it is, in words. Copied into the ledger.
    pub reasons: Vec<String>,
}

pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Median absolute deviation (unscaled).
pub fn mad(values: &[f64]) -> f64 {
    let m = median(values);
    let dev: Vec<f64> = values.iter().map(|v| (v - m).abs()).collect();
    median(&dev)
}

/// xorshift64*: deterministic, so a verdict recomputed from the ledger
/// reproduces its own CI exactly.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

/// Percentile bootstrap 95% interval of the median.
pub fn bootstrap_median_ci(values: &[f64]) -> (f64, f64) {
    let n = values.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mut rng = Rng(BOOTSTRAP_SEED);
    let mut medians = Vec::with_capacity(BOOTSTRAP_RESAMPLES);
    let mut sample = vec![0.0; n];
    for _ in 0..BOOTSTRAP_RESAMPLES {
        for slot in sample.iter_mut() {
            *slot = values[(rng.next() % n as u64) as usize];
        }
        medians.push(median(&sample));
    }
    medians.sort_by(|a, b| a.total_cmp(b));
    let at = |q: f64| medians[((q * (medians.len() - 1) as f64).round()) as usize];
    (at(0.025), at(0.975))
}

/// One-sided exact sign test: P(X >= k) for X ~ Binomial(n, 1/2).
pub fn sign_test_p(k: usize, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    // Sum C(n, i) / 2^n for i >= k, in log space to stay exact enough for
    // any n this harness will see.
    let ln_choose = |n: usize, i: usize| -> f64 {
        let lgamma = |x: usize| (1..=x).map(|j| (j as f64).ln()).sum::<f64>();
        lgamma(n) - lgamma(i) - lgamma(n - i)
    };
    (k..=n)
        .map(|i| (ln_choose(n, i) - n as f64 * std::f64::consts::LN_2).exp())
        .sum::<f64>()
        .min(1.0)
}

/// The verdict for a set of clean pair improvements. A short count is
/// INCONCLUSIVE; the orchestrator keeps adding pairs up to `max_pairs`
/// before asking.
pub fn decide(improvements: &[f64], min_effect: f64) -> VerdictReport {
    let min_effect = min_effect.max(MIN_EFFECT_FLOOR);
    let n = improvements.len();
    let med = median(improvements);
    let (ci_low, ci_high) = bootstrap_median_ci(improvements);
    let positives = improvements.iter().filter(|v| **v > 0.0).count();
    let negatives = improvements.iter().filter(|v| **v < 0.0).count();
    let nonzero = positives + negatives;
    let p_better = sign_test_p(positives, nonzero);
    let p_worse = sign_test_p(negatives, nonzero);
    let mut reasons = Vec::new();

    let verdict = if n < MIN_PAIRS {
        reasons.push(format!("{n} clean pairs, need {MIN_PAIRS}"));
        Verdict::Inconclusive
    } else {
        let agrees = |count: usize, p: f64| {
            if n < UNANIMITY_BELOW {
                count == n
            } else {
                p <= SIGN_TEST_ALPHA
            }
        };
        let win = med >= min_effect && ci_low > 0.0 && agrees(positives, p_better);
        let loss = med <= -min_effect && ci_high < 0.0 && agrees(negatives, p_worse);
        if win {
            Verdict::Win
        } else if loss {
            Verdict::Loss
        } else {
            if med.abs() < min_effect {
                reasons.push(format!(
                    "median {:+.2}% is inside the {:.2}% effect floor",
                    med * 100.0,
                    min_effect * 100.0
                ));
            }
            if ci_low <= 0.0 && ci_high >= 0.0 {
                reasons.push(format!(
                    "95% CI [{:+.2}%, {:+.2}%] spans zero",
                    ci_low * 100.0,
                    ci_high * 100.0
                ));
            }
            if n < UNANIMITY_BELOW && positives != n && negatives != n {
                reasons.push(format!(
                    "pairs disagree in sign ({positives} better, {negatives} worse) with fewer than {UNANIMITY_BELOW} pairs"
                ));
            }
            Verdict::Noise
        }
    };
    VerdictReport {
        verdict,
        pairs: n,
        improvements: improvements.to_vec(),
        median: med,
        ci_low,
        ci_high,
        positives,
        negatives,
        p_better,
        p_worse,
        min_effect,
        reasons,
    }
}

/// The effect floor a host's own A-against-A spread justifies: twice the MAD
/// of a null run's pair improvements. Combined with the spec's request and
/// [`MIN_EFFECT_FLOOR`] by taking the largest.
pub fn null_floor(null_improvements: &[f64]) -> f64 {
    if null_improvements.is_empty() {
        0.0
    } else {
        2.0 * mad(null_improvements)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn improvement_sign_follows_direction() {
        assert!((improvement(10.0, 11.0, Direction::HigherIsBetter).unwrap() - 0.1).abs() < 1e-12);
        assert!((improvement(10.0, 9.0, Direction::LowerIsBetter).unwrap() - 0.1).abs() < 1e-12);
        assert!(improvement(0.0, 1.0, Direction::HigherIsBetter).is_none());
        assert!(improvement(f64::NAN, 1.0, Direction::HigherIsBetter).is_none());
    }

    #[test]
    fn sign_test_matches_binomial_tail() {
        // P(X >= 6 | n = 6) = 1/64.
        assert!((sign_test_p(6, 6) - 1.0 / 64.0).abs() < 1e-12);
        // P(X >= 5 | n = 6) = 7/64.
        assert!((sign_test_p(5, 6) - 7.0 / 64.0).abs() < 1e-12);
        assert!((sign_test_p(0, 6) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn clear_win_and_clear_loss() {
        let win = decide(&[0.10, 0.12, 0.09, 0.11], 0.02);
        assert_eq!(win.verdict, Verdict::Win, "{:?}", win.reasons);
        let loss = decide(&[-0.10, -0.12, -0.09, -0.11], 0.02);
        assert_eq!(loss.verdict, Verdict::Loss, "{:?}", loss.reasons);
    }

    #[test]
    fn too_few_pairs_is_inconclusive_however_large() {
        assert_eq!(decide(&[0.5, 0.6], 0.02).verdict, Verdict::Inconclusive);
    }

    #[test]
    fn small_or_mixed_effects_are_noise() {
        // Under the floor even though every pair agrees.
        assert_eq!(
            decide(&[0.01, 0.012, 0.011, 0.009], 0.0).verdict,
            Verdict::Noise
        );
        // Large median but one pair disagrees with fewer than six pairs.
        assert_eq!(
            decide(&[0.10, 0.12, -0.05, 0.11], 0.02).verdict,
            Verdict::Noise
        );
    }

    #[test]
    fn spec_can_only_raise_the_floor() {
        let r = decide(&[0.03, 0.031, 0.029, 0.03], 0.0);
        assert_eq!(r.min_effect, MIN_EFFECT_FLOOR);
        assert_eq!(r.verdict, Verdict::Win);
        assert_eq!(
            decide(&[0.03, 0.031, 0.029, 0.03], 0.05).verdict,
            Verdict::Noise
        );
    }

    #[test]
    fn many_pairs_use_the_sign_test_not_unanimity() {
        // 8 of 9 better, large median: p = 10/512 < 0.05.
        let mut v = vec![0.08; 8];
        v.push(-0.01);
        assert_eq!(decide(&v, 0.02).verdict, Verdict::Win);
    }

    #[test]
    fn bootstrap_is_deterministic() {
        let v = [0.01, 0.05, -0.02, 0.07, 0.03];
        assert_eq!(bootstrap_median_ci(&v), bootstrap_median_ci(&v));
    }
}
