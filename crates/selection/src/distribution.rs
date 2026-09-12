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

    let mut support: Vec<u32> = Vec::with_capacity(p.support.len());
    let mut probs: Vec<f64> = Vec::with_capacity(p.support.len());
    for (i, &token) in p.support.iter().enumerate() {
        let mass = (p.probs[i] - q.probability(token)).max(0.0);
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
