//! The materialized shaped distribution and the residual sampler
//! speculative decoding needs at a positive temperature
//! (ROADMAP P1 item 4).
//!
//! Leviathan et al. (arXiv 2211.17192) and Chen et al. (arXiv 2302.01318)
//! make a speculative round exact at any temperature by REPLACING
//! argmax-equality with a rejection step over the two full distributions:
//! a proposal x drawn from the drafter's shaped q is accepted with
//! probability p(x)/q(x), and on rejection the corrected token is drawn
//! from the normalized residual max(p - q, 0). Both p and q here are the
//! SHAPED distributions -- the same pipeline [`crate::select`] samples
//! from, penalties and truncation included -- which is what makes the
//! composite match a sequential sampled decode in distribution rather
//! than merely in argmax.

use std::collections::HashMap;

use crate::derive::{derive_step_value, to_unit_interval};

/// A normalized distribution over an explicit support, parallel arrays in
/// NO particular order (the support comes out of the truncation pipeline,
/// which is ranked-descending, but nothing here relies on that).
#[derive(Debug, Clone, PartialEq)]
pub struct ShapedDistribution {
    support: Vec<u32>,
    probs: Vec<f64>,
}

impl ShapedDistribution {
    /// From the truncation pipeline's surviving set and its reweighted
    /// probabilities. The constructor TRUSTS normalization to within
    /// f64 rounding; [`Self::probability`] does not renormalize.
    pub fn new(support: Vec<u32>, probs: Vec<f64>) -> Self {
        assert_eq!(
            support.len(),
            probs.len(),
            "support and probabilities must be parallel"
        );
        // `draw` falls back to the last entry on rounding, so an empty
        // support would panic there instead of at the faulty constructor.
        assert!(!support.is_empty(), "support must not be empty");
        Self { support, probs }
    }

    /// The deterministic arm's distribution: one candidate, all mass.
    pub fn point_mass(token: u32) -> Self {
        Self {
            support: vec![token],
            probs: vec![1.0],
        }
    }

    pub fn support(&self) -> &[u32] {
        &self.support
    }

    /// The probability of `token`, 0.0 outside the support: a truncated
    /// distribution assigns no mass where it was truncated, which is
    /// exactly the "certain rejection" the acceptance step needs rather
    /// than an error.
    pub fn probability(&self, token: u32) -> f64 {
        match self.support.iter().position(|&t| t == token) {
            Some(i) => self.probs[i],
            None => 0.0,
        }
    }

    /// One draw, over the SAME seeded-uniform derivation [`crate::select`]
    /// uses (`derive_step_value(seed, position)`), so a seeded run's draws
    /// stay reproducible position by position.
    pub fn draw(&self, seed: Option<u64>, position: u64) -> u32 {
        let target = to_unit_interval(derive_step_value(seed, position));
        let mut cumulative = 0.0;
        for (i, &w) in self.probs.iter().enumerate() {
            cumulative += w;
            if target < cumulative {
                return self.support[i];
            }
        }
        // Numerical rounding at the tail: the last support entry, matching
        // `select`'s own draw fallback.
        *self.support.last().expect("support is never empty")
    }
}

/// The normalized residual `max(p - q, 0)` over the union of the two
/// supports: the distribution a REJECTED proposal's corrected token is
/// drawn from.
///
/// `None` when the residual's total mass underflows the epsilon below --
/// mathematically reachable only when p and q are the same distribution,
/// where every proposal is accepted and no correction draw can happen;
/// numerically it can also appear when q covers p almost everywhere. The
/// caller falls back to a plain `p` draw there, which is exact in the
/// limit and biased only by the epsilon.
pub fn residual(p: &ShapedDistribution, q: &ShapedDistribution) -> Option<ShapedDistribution> {
    /// Below this much residual mass, treat p as covered by q.
    const RESIDUAL_EPSILON: f64 = 1e-12;

    // A hash join, because `top_k: 0` from a request makes both supports the
    // whole vocabulary and `q.probability` per entry is then ~3.4e10 steps per
    // rejected proposal. Lookup only, so hash order cannot reach the output;
    // `or_insert` keeps the first occurrence, as `probability` does.
    let mut q_mass: HashMap<u32, f64> = HashMap::with_capacity(q.support.len());
    for (&token, &prob) in q.support.iter().zip(&q.probs) {
        q_mass.entry(token).or_insert(prob);
    }

    let mut support: Vec<u32> = Vec::with_capacity(p.support.len());
    let mut probs: Vec<f64> = Vec::with_capacity(p.support.len());
    for (i, &token) in p.support.iter().enumerate() {
        let mass = (p.probs[i] - q_mass.get(&token).copied().unwrap_or(0.0)).max(0.0);
        if mass > 0.0 {
            support.push(token);
            probs.push(mass);
        }
    }
    let total: f64 = probs.iter().sum();
    if total < RESIDUAL_EPSILON || support.is_empty() {
        return None;
    }
    for prob in &mut probs {
        *prob /= total;
    }
    Some(ShapedDistribution { support, probs })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist(support: &[u32], probs: &[f64]) -> ShapedDistribution {
        ShapedDistribution::new(support.to_vec(), probs.to_vec())
    }

    /// An empty support has no token to draw, so `draw` would panic on its
    /// tail fallback far from the bad construction. Refuse it at `new`.
    #[test]
    #[should_panic(expected = "support must not be empty")]
    fn new_refuses_an_empty_support() {
        let _ = ShapedDistribution::new(Vec::new(), Vec::new());
    }

    #[test]
    fn probability_is_zero_outside_the_support() {
        let d = dist(&[3, 7], &[0.25, 0.75]);
        assert_eq!(d.probability(3), 0.25);
        assert_eq!(d.probability(7), 0.75);
        assert_eq!(d.probability(4), 0.0);
    }

    /// The Leviathan worked example shape: p uniform over {0,1}, q all mass
    /// on 0. The residual is p - q on 1 only, so a rejected proposal's
    /// correction is ALWAYS token 1 -- and acceptance of a 0-proposal is
    /// certain (p(0)/q(0) = 1).
    #[test]
    fn residual_takes_the_positive_part_over_the_union() {
        let p = dist(&[0, 1], &[0.5, 0.5]);
        let q = dist(&[0], &[1.0]);
        let r = residual(&p, &q).expect("half the mass survives");
        assert_eq!(r.support(), &[1]);
        assert_eq!(r.probability(1), 1.0);
        assert_eq!(r.probability(0), 0.0);
    }

    /// q wider than p: the tokens only q carries contribute nothing (p - q
    /// clamps at 0), and the residual is p renormalized over where p
    /// exceeds q.
    #[test]
    fn residual_ignores_tokens_only_q_carries() {
        let p = dist(&[0, 1], &[0.6, 0.4]);
        let q = dist(&[0, 1, 2], &[0.2, 0.1, 0.7]);
        let r = residual(&p, &q).expect("0.4 + 0.3 survives");
        assert_eq!(r.probability(2), 0.0);
        assert!((r.probability(0) - 0.4 / 0.7).abs() < 1e-12);
        assert!((r.probability(1) - 0.3 / 0.7).abs() < 1e-12);
    }

    /// Identical distributions have no residual: every proposal is accepted
    /// with probability 1, so the None contract is the exact case.
    #[test]
    fn identical_distributions_have_no_residual() {
        let p = dist(&[0, 1], &[0.5, 0.5]);
        assert_eq!(residual(&p, &p), None);
    }

    /// A point-mass target the drafter cannot see at all: the residual is
    /// the whole target, which is the "useless drafter still exact" case.
    #[test]
    fn a_covered_target_residuals_to_itself_renormalized() {
        let p = dist(&[5], &[1.0]);
        let q = dist(&[9], &[1.0]);
        let r = residual(&p, &q).expect("all mass survives");
        assert_eq!(r.support(), &[5]);
        assert_eq!(r.probability(5), 1.0);
    }

    /// `top_k: 0` over HTTP makes both supports the whole vocabulary, in
    /// different rank orders, and a rejected proposal runs this join with no
    /// cancellation point inside it. A scan per entry is ~3.4e10 steps here,
    /// about 11 s per rejection even optimized. The join runs on a worker so
    /// the red case fails at the budget instead of holding the suite; the
    /// budget is over 25x what the hash join takes in a debug build.
    #[test]
    fn residual_at_full_vocabulary_is_not_a_quadratic_join() {
        const GEMMA_VOCAB: u32 = 262_144;
        const BUDGET: std::time::Duration = std::time::Duration::from_secs(5);

        let n = GEMMA_VOCAB as f64;
        let weight_sum = n * (n + 1.0) / 2.0;
        let support: Vec<u32> = (0..GEMMA_VOCAB).collect();
        let probs: Vec<f64> = (0..GEMMA_VOCAB)
            .map(|t| f64::from(t + 1) / weight_sum)
            .collect();
        let p = ShapedDistribution::new(support.clone(), probs);
        // Reversed, so no entry of p sits near its own position in q.
        let q = ShapedDistribution::new(
            support.into_iter().rev().collect(),
            vec![1.0 / n; GEMMA_VOCAB as usize],
        );

        let (done, finished) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = done.send(residual(&p, &q));
        });
        let r = finished
            .recv_timeout(BUDGET)
            .expect("residual over a full vocabulary must not take seconds")
            .expect("p is not q, so mass survives");
        // p exceeds uniform q on the upper half of the tokens, in p's order.
        let upper_half: Vec<u32> = (GEMMA_VOCAB / 2..GEMMA_VOCAB).collect();
        assert_eq!(r.support(), upper_half);
    }

    /// The draw walks the residual's support in order, so the join must keep
    /// p's order and the exact float values or a seeded run's corrections
    /// change. `probability` answers with the FIRST occurrence of a duplicate,
    /// and `ShapedDistribution::new` admits duplicates, so q has one here.
    #[test]
    fn residual_keeps_p_order_and_matches_probability_bit_for_bit() {
        let p = dist(&[9, 2, 40, 7, 1000], &[0.3, 0.25, 0.2, 0.15, 0.1]);
        let q = dist(&[7, 40, 40, 2, 11], &[0.05, 0.1, 0.6, 0.3, 0.05]);

        let mass: Vec<f64> = p
            .support()
            .iter()
            .zip(&p.probs)
            .map(|(&t, &w)| (w - q.probability(t)).max(0.0))
            .collect();
        let total: f64 = mass.iter().filter(|&&m| m > 0.0).sum();
        let (want_support, want_probs): (Vec<u32>, Vec<f64>) = p
            .support()
            .iter()
            .zip(&mass)
            .filter(|(_, &m)| m > 0.0)
            .map(|(&t, &m)| (t, m / total))
            .unzip();

        // 40 is the discriminator: first occurrence leaves it 0.1, last would
        // clamp it out.
        assert_eq!(want_support, [9, 40, 7, 1000]);
        let r = residual(&p, &q).expect("mass survives");
        assert_eq!(r.support(), want_support);
        let got_bits: Vec<u64> = r.probs.iter().map(|w| w.to_bits()).collect();
        let want_bits: Vec<u64> = want_probs.iter().map(|w| w.to_bits()).collect();
        assert_eq!(got_bits, want_bits);
    }

    /// A seeded draw is reproducible and stays inside the support, for the
    /// same `(seed, position)` pair.
    #[test]
    fn seeded_draws_are_reproducible_and_in_support() {
        let d = dist(&[11, 12, 13], &[0.2, 0.3, 0.5]);
        for position in [0u64, 1, 7, 12345] {
            let a = d.draw(Some(42), position);
            let b = d.draw(Some(42), position);
            assert_eq!(a, b, "position {position}");
            assert!(d.support().contains(&a));
        }
        // Different positions do not all collapse to one entry on a
        // three-way support (a draw that ignored the uniform would).
        let distinct: std::collections::HashSet<u32> =
            (0..64).map(|p| d.draw(Some(42), p)).collect();
        assert!(distinct.len() >= 2, "draw never varied: {distinct:?}");
    }
}
