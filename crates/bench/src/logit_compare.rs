//! Comparing two next-token distributions, shared by the prefill arm probes
//! and `tslab`'s quality gate.
//!
//! This is the math the arm probes carried as private copies
//! (`qwen38_prefill_arm_probe.rs`, `qwen36_prefill_arm_probe.rs`), moved
//! here unchanged so a bound measured by one is the bound measured by the
//! other.

/// Log-softmax in f64 with the max subtracted first.
pub fn log_softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = logits.iter().map(|v| (v - max).exp()).sum();
    let log_z = max + sum.ln();
    logits.iter().map(|v| v - log_z).collect()
}

/// Indices of the `k` largest logits, largest first.
pub fn top_k(logits: &[f64], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.truncate(k);
    idx
}

/// One reference/arm comparison at one position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Comparison {
    /// KL(reference || arm) in nats over the full vocabulary.
    pub kl: f64,
    /// Largest absolute logit difference.
    pub max_abs_dlogit: f64,
    pub top1_agree: bool,
    /// How many of the reference's top 5 are in the arm's top 5.
    pub top5_overlap: usize,
}

pub fn compare(reference: &[f64], arm: &[f64]) -> Comparison {
    let (lr, ls) = (log_softmax(reference), log_softmax(arm));
    let kl: f64 = lr.iter().zip(&ls).map(|(a, b)| a.exp() * (a - b)).sum();
    let max_abs_dlogit = reference
        .iter()
        .zip(arm)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let (t5r, t5s) = (top_k(reference, 5), top_k(arm, 5));
    Comparison {
        kl,
        max_abs_dlogit,
        top1_agree: t5r.first() == t5s.first(),
        top5_overlap: t5r.iter().filter(|i| t5s.contains(i)).count(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_distributions_have_zero_kl() {
        let l = [1.0, 2.0, 3.0, -1.0, 0.5, 0.0];
        let c = compare(&l, &l);
        assert_eq!(c.kl, 0.0);
        assert!(c.top1_agree);
        assert_eq!(c.top5_overlap, 5);
    }

    #[test]
    fn kl_is_positive_and_shift_invariant() {
        let r = [1.0, 2.0, 3.0];
        let a = [1.5, 2.0, 2.5];
        let shifted: Vec<f64> = a.iter().map(|v| v + 100.0).collect();
        let c = compare(&r, &a);
        assert!(c.kl > 0.0);
        assert!((compare(&r, &shifted).kl - c.kl).abs() < 1e-12);
    }
}
