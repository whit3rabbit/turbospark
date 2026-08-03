//! Composition of probability-mass truncation and rank-based truncation
//! over a full-distribution probability vector.
//!
//! Probability-mass truncation is always evaluated against the same
//! `probs` slice produced once by [`softmax`] over the full candidate
//! domain; it is never renormalized to a rank-truncated subset. Rank-based
//! truncation is applied afterward, capping whatever survived the
//! probability-mass step.

/// Convert raw scores into a full-vocabulary-normalized probability
/// distribution using a numerically stable softmax. Monotonic in the input
/// scores, so ranking is preserved.
pub fn softmax(scores: &[f32]) -> Vec<f64> {
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let exps: Vec<f64> = scores.iter().map(|&s| (s as f64 - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Rank candidate indices by descending probability. Ties break by
/// ascending index for a fixed, consistent order.
pub fn rank_indices(probs: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx
}

/// Keep the smallest ranked prefix whose cumulative probability mass first
/// reaches or exceeds `threshold`, evaluated against the full-distribution
/// probabilities in `probs`. `None` means probability-mass truncation is
/// disabled and every ranked candidate survives.
pub fn truncate_by_probability_mass(
    ranked: &[usize],
    probs: &[f64],
    threshold: Option<f64>,
) -> Vec<usize> {
    let Some(threshold) = threshold else {
        return ranked.to_vec();
    };
    let mut kept = Vec::new();
    let mut cumulative = 0.0;
    for &idx in ranked {
        kept.push(idx);
        cumulative += probs[idx];
        if cumulative >= threshold {
            break;
        }
    }
    if kept.is_empty() && !ranked.is_empty() {
        kept.push(ranked[0]);
    }
    kept
}

/// Cap the surviving set to its top `top_k` entries, preserving rank order.
/// `0` means rank-based truncation is disabled and the full surviving set
/// is kept. Never underflows to zero members while `surviving` is
/// non-empty.
pub fn truncate_by_rank(surviving: &[usize], top_k: u32) -> Vec<usize> {
    if top_k == 0 || surviving.is_empty() {
        return surviving.to_vec();
    }
    let n = (top_k as usize).min(surviving.len());
    surviving[..n].to_vec()
}
