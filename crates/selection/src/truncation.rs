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
    idx.sort_by(rank_order(probs));
    idx
}

/// The first `k` indices [`rank_indices`] would produce, without ordering
/// the rest of the candidate domain.
///
/// Both truncation steps below keep only a PREFIX of the full ranked
/// order, so whenever rank-based truncation is enabled the tail is
/// unobservable and sorting it is pure waste. At the real Gemma 4
/// vocabulary (262144) the full sort this replaces cost ~17 ms per
/// token -- more than the entire GPU forward pass, and the whole of this
/// port's measured decode gap against the Swift original, which samples
/// on the GPU instead (docs/BENCHMARKS.md).
pub fn rank_top_k(probs: &[f64], k: usize) -> Vec<usize> {
    let k = k.min(probs.len());
    if k == 0 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    if k < idx.len() {
        // Partitions in O(n) so that idx[..k] is exactly the top-k set;
        // it is not yet ordered within itself, hence the sort after.
        idx.select_nth_unstable_by(k - 1, rank_order(probs));
        idx.truncate(k);
    }
    idx.sort_unstable_by(rank_order(probs));
    idx
}

/// Descending probability, ties by ascending index. Shared so the partial
/// and full ranking cannot drift apart.
fn rank_order(probs: &[f64]) -> impl Fn(&usize, &usize) -> std::cmp::Ordering + '_ {
    |&a: &usize, &b: &usize| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    }
}

/// [`rank_indices`] into a reused `u32` scratch buffer: same order, same
/// tie-break, no fresh multi-megabyte allocation per decoded token. `keys`
/// may be any monotone image of the probabilities (the hot path passes
/// unnormalized `exp(s - max)` values; division by the positive normalizer
/// cannot reorder them).
pub fn rank_indices_u32_into(keys: &[f64], out: &mut Vec<u32>) {
    fill_identity(out, keys.len());
    out.sort_unstable_by(rank_order_u32(keys));
}

/// [`rank_top_k`] into a reused `u32` scratch buffer. See
/// [`rank_indices_u32_into`] for what `keys` may be.
pub fn rank_top_k_u32_into(keys: &[f64], k: usize, out: &mut Vec<u32>) {
    let k = k.min(keys.len());
    if k == 0 {
        out.clear();
        return;
    }
    fill_identity(out, keys.len());
    if k < out.len() {
        // Partitions in O(n) so that out[..k] is exactly the top-k set;
        // it is not yet ordered within itself, hence the sort after.
        out.select_nth_unstable_by(k - 1, rank_order_u32(keys));
        out.truncate(k);
    }
    out.sort_unstable_by(rank_order_u32(keys));
}

/// The identity permutation `0..len` in `out`, reusing its capacity.
fn fill_identity(out: &mut Vec<u32>, len: usize) {
    assert!(
        u32::try_from(len).is_ok(),
        "candidate domain exceeds u32 index range"
    );
    out.clear();
    out.extend(0..len as u32);
}

/// [`rank_order`] over `u32` indices. The comparator never returns Equal
/// for distinct indices, so stable and unstable sorts agree.
fn rank_order_u32(keys: &[f64]) -> impl Fn(&u32, &u32) -> std::cmp::Ordering + '_ {
    |&a: &u32, &b: &u32| {
        keys[b as usize]
            .partial_cmp(&keys[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    }
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
