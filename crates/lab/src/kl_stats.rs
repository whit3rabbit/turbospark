//! Per-token quality statistics over teacher-forced logits (LAB-02).
//!
//! A mean KL over a handful of final positions hides the few catastrophic
//! tokens and cannot say whether a drift is quality loss or a coin flip. Over
//! a forced trace (both arms see the reference's token ids) this reports, per
//! position: KL, whether the top-1 flipped, how decided the reference was at
//! that position (its top-2 margin), and the log-probability each arm gave
//! the token the reference actually took, with a paired standard error.
//!
//! Statistics widen evidence, never tolerances: the bounds stay in
//! `gates.rs`. A few hundred tokens give a standard error around 0.005 nats,
//! so they cannot certify a smaller change.

use serde::{Deserialize, Serialize};

// The bench crate's math on macOS; the same two functions elsewhere, where
// the lab does not depend on it but `gates.rs` still builds.
#[cfg(not(target_os = "macos"))]
use portable::{log_softmax, top_k};
#[cfg(target_os = "macos")]
use turbospark_bench::logit_compare::{log_softmax, top_k};

#[cfg(not(target_os = "macos"))]
mod portable {
    pub fn log_softmax(logits: &[f64]) -> Vec<f64> {
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = logits.iter().map(|v| (v - max).exp()).sum();
        let log_z = max + sum.ln();
        logits.iter().map(|v| v - log_z).collect()
    }

    pub fn top_k(logits: &[f64], k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|&a, &b| {
            let (x, y) = (logits[b], logits[a]);
            x.partial_cmp(&y)
                .unwrap_or_else(|| x.is_nan().cmp(&y.is_nan()))
        });
        idx.truncate(k);
        idx
    }
}

/// A flip whose reference margin is under this many logits is a near tie.
pub const SMALL_MARGIN: f64 = 0.5;
/// A flip whose reference margin is at least this many logits: the reference
/// was sure and the arm chose otherwise.
pub const LARGE_MARGIN: f64 = 2.0;

/// One position, reference against arm.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub kl: f64,
    pub max_abs_dlogit: f64,
    pub flipped: bool,
    /// Reference top-1 minus top-2 logit.
    pub ref_margin: f64,
    /// Log-probability of the forced next token, per arm.
    pub ref_logp: f64,
    pub arm_logp: f64,
    pub bits_equal: bool,
}

/// Compare one position. `target` is the token the reference took here.
pub fn position(reference: &[f64], arm: &[f64], target: usize) -> Result<Position, String> {
    if reference.len() != arm.len() || reference.len() < 2 {
        return Err(format!(
            "position: {} reference logits against {}",
            reference.len(),
            arm.len()
        ));
    }
    if target >= reference.len() {
        return Err(format!("position: target {target} outside the vocabulary"));
    }
    let (lr, la) = (log_softmax(reference), log_softmax(arm));
    let kl = lr.iter().zip(&la).map(|(a, b)| a.exp() * (a - b)).sum();
    let top = top_k(reference, 2);
    // f64::max drops NaN; carry it so a broken arm is not reported finite.
    let max_abs_dlogit = reference.iter().zip(arm).fold(0.0f64, |m, (a, b)| {
        let d = (a - b).abs();
        if m.is_nan() || d.is_nan() {
            f64::NAN
        } else {
            m.max(d)
        }
    });
    Ok(Position {
        kl,
        max_abs_dlogit,
        // A NaN anywhere in the arm is a broken position, never agreement.
        flipped: top[0] != top_k(arm, 1)[0] || arm.iter().any(|v| v.is_nan()),
        ref_margin: reference[top[0]] - reference[top[1]],
        ref_logp: lr[target],
        arm_logp: la[target],
        bits_equal: reference
            .iter()
            .zip(arm)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub tokens: usize,
    pub mean_kl: f64,
    pub median_kl: f64,
    pub p99_kl: f64,
    /// Standard error of the mean KL.
    pub se_kl: f64,
    pub flips: usize,
    pub flip_rate: f64,
    /// Share of flips whose reference margin was under [`SMALL_MARGIN`].
    pub small_margin_flip_share: Option<f64>,
    /// Flips whose reference margin was at least [`LARGE_MARGIN`].
    pub large_margin_flips: usize,
    pub ref_ppl: f64,
    pub arm_ppl: f64,
    /// Mean of (reference log p - arm log p) of the forced token, nats;
    /// positive means the arm predicts the reference text worse.
    pub nll_delta: f64,
    /// Paired standard error of `nll_delta`.
    pub nll_delta_se: f64,
    pub all_bits_equal: bool,
}

fn mean_se(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    if v.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, (var / n).sqrt())
}

/// `q` in [0, 1] over a copy, nearest rank. NaN sorts last, so a broken
/// position shows in the p99.
fn quantile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s.get(((q * (s.len().max(1) - 1) as f64).round()) as usize)
        .copied()
        .unwrap_or(f64::NAN)
}

pub fn summarize(positions: &[Position]) -> Summary {
    let n = positions.len();
    let kl: Vec<f64> = positions.iter().map(|p| p.kl).collect();
    let delta: Vec<f64> = positions.iter().map(|p| p.ref_logp - p.arm_logp).collect();
    let flips: Vec<&Position> = positions.iter().filter(|p| p.flipped).collect();
    let (mean_kl, se_kl) = mean_se(&kl);
    let (nll_delta, nll_delta_se) = mean_se(&delta);
    let ppl = |f: fn(&Position) -> f64| (-positions.iter().map(f).sum::<f64>() / n as f64).exp();
    Summary {
        tokens: n,
        mean_kl,
        median_kl: quantile(&kl, 0.5),
        p99_kl: quantile(&kl, 0.99),
        se_kl,
        flips: flips.len(),
        flip_rate: flips.len() as f64 / n.max(1) as f64,
        small_margin_flip_share: (!flips.is_empty()).then(|| {
            flips.iter().filter(|p| p.ref_margin < SMALL_MARGIN).count() as f64 / flips.len() as f64
        }),
        large_margin_flips: flips
            .iter()
            .filter(|p| p.ref_margin >= LARGE_MARGIN)
            .count(),
        ref_ppl: ppl(|p| p.ref_logp),
        arm_ppl: ppl(|p| p.arm_logp),
        nll_delta,
        nll_delta_se,
        all_bits_equal: positions.iter().all(|p| p.bits_equal),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_rows_are_exact_and_flip_nothing() {
        let l = [0.5, 3.0, -1.0, 2.0];
        let p = position(&l, &l, 1).unwrap();
        assert_eq!(p.kl, 0.0);
        assert!(p.bits_equal && !p.flipped);
        assert_eq!(p.ref_margin, 1.0);
        let s = summarize(&[p, p]);
        assert_eq!(s.flips, 0);
        assert_eq!(s.small_margin_flip_share, None);
        assert_eq!(s.nll_delta, 0.0);
        assert!((s.ref_ppl - s.arm_ppl).abs() < 1e-12);
        assert!(s.all_bits_equal);
    }

    #[test]
    fn flips_are_split_by_how_sure_the_reference_was() {
        // A near tie (margin 0.1) that flips, and a sure one (margin 5).
        let tie = position(&[1.1, 1.0, -9.0], &[1.0, 1.1, -9.0], 0).unwrap();
        let sure = position(&[6.0, 1.0, -9.0], &[1.0, 6.0, -9.0], 0).unwrap();
        let same = position(&[2.0, 1.0, -9.0], &[2.0, 1.0, -9.0], 0).unwrap();
        let s = summarize(&[tie, sure, same]);
        assert_eq!(s.flips, 2);
        assert_eq!(s.small_margin_flip_share, Some(0.5));
        assert_eq!(s.large_margin_flips, 1);
        // The arm predicts the reference text worse.
        assert!(s.nll_delta > 0.0 && s.arm_ppl > s.ref_ppl);
        assert!(!s.all_bits_equal);
    }

    #[test]
    fn nan_logits_do_not_read_as_agreement() {
        let p = position(&[1.0, 2.0, 0.0], &[1.0, f64::NAN, 0.0], 0).unwrap();
        assert!(p.kl.is_nan() && p.max_abs_dlogit.is_nan() && p.flipped);
        assert!(summarize(&[p]).mean_kl.is_nan());
    }

    #[test]
    fn mismatched_rows_and_targets_refuse() {
        assert!(position(&[1.0, 2.0], &[1.0], 0).is_err());
        assert!(position(&[1.0, 2.0], &[1.0, 2.0], 2).is_err());
    }
}
