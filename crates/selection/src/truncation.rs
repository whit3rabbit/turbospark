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

/// Descending key, NaN last, ties by ascending index.
///
/// **This must be a TOTAL order and for a long time it was not.** Both
/// comparators used to read `partial_cmp(..).unwrap_or(Equal).then(index)`,
/// which makes a NaN compare Equal to every real key and is therefore
/// intransitive: on `[1.0, NaN, 5.0]` it reports `0 < 1` and `1 < 2` and
/// `0 > 2`. Rust's sorts detect that and PANIC with "user-provided
/// comparison function does not correctly implement a total order", so
/// `rank_indices*` and `rank_top_k*` could abort the process on some
/// NaN-bearing inputs -- not all, which is why it went unnoticed.
///
/// Sorting NaN LAST rather than adopting `f64::total_cmp` is deliberate,
/// and the reason is behavioural rather than aesthetic. `total_cmp` is the
/// IEEE totalOrder predicate, under which a positive NaN sits ABOVE
/// infinity, so a NaN score would become the top-ranked candidate and get
/// itself selected -- worse than the panic it replaced. It would also
/// separate `-0.0` from `0.0`, which the old comparator called equal, and
/// that is a live difference for a public function documented to take "any
/// monotone image of the probabilities".
///
/// So on NaN-free input this is EXACTLY the old comparator (`partial_cmp`
/// always resolves, `-0.0` and `0.0` stay equal, ties fall to the index),
/// and nothing any caller in this workspace can reach changes. `select`
/// rejects a non-finite score vector before ranking, so the hot path was
/// never exposed either way.
fn rank_cmp(key_a: f64, key_b: f64, a: usize, b: usize) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (key_a.is_nan(), key_b.is_nan()) {
        (false, false) => key_b
            .partial_cmp(&key_a)
            .expect("neither key is NaN in this arm")
            .then(a.cmp(&b)),
        // Two NaNs are indistinguishable as keys, so the index alone
        // orders them -- the same rule ties get everywhere else here.
        (true, true) => a.cmp(&b),
        // A NaN ranks after every real key, whichever side it is on.
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
    }
}

/// Descending probability, ties by ascending index. Shared so the partial
/// and full ranking cannot drift apart.
fn rank_order(probs: &[f64]) -> impl Fn(&usize, &usize) -> std::cmp::Ordering + '_ {
    |&a: &usize, &b: &usize| rank_cmp(probs[a], probs[b], a, b)
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
///
/// **This is 70% of the host sampler's cost and the reason is the ACCESS
/// PATTERN, not the complexity** (measured 2026-08-15, `tests/
/// host_sampler_cost.rs`: 2.061 ms of `select`'s 2.940 at V=262144). The
/// obvious implementation -- `select_nth_unstable_by` over the identity
/// permutation -- is already O(n), but it partitions `u32` INDICES under a
/// comparator that dereferences `keys[a]` and `keys[b]`, so on a 2 MiB key
/// array almost every one of the ~n comparisons is a cache miss, and it
/// writes a 1 MiB identity permutation first.
///
/// So this finds the cut with one SEQUENTIAL pass instead, then ranks only
/// the indices that reach it. The result is identical rather than merely
/// equivalent: with no NaN present `rank_order_u32` is a total order
/// (`partial_cmp` always resolves and the index tie-break settles the
/// rest), the collected set is a superset of the top-`k` because it admits
/// every tie at the cut, and reducing a superset to its top-`k` then
/// ordering it yields the same prefix as sorting the whole domain.
/// `tests/rank_top_k.rs` checks that against the full-sort reference, ties
/// included.
///
/// **THE ADMITTED SET IS BOUNDED BY NOTHING, AND WHAT KEEPS THAT CHEAP IS
/// THE ORDER IT IS COLLECTED IN.** How many indices reach the cut is a
/// property of the input: on a real distribution it is `k` or a few more,
/// and on a largely constant key array (a uniform-ish distribution, or an
/// `exp` pass that underflowed most of the vocabulary to one value) it is
/// the whole domain. Sorting the whole domain is exactly the cost this
/// function exists to avoid, so the second case looks like a hole.
///
/// It is not one, and the reason is a coincidence worth pinning rather
/// than relying on silently. Everything tied at the cut has the SAME key,
/// so `rank_order_u32` falls to its ascending-index tie-break; the collect
/// loop pushes indices in ascending order; so the admitted set arrives
/// ALREADY SORTED but for the at most `k - 1` entries above the cut, and
/// `sort_unstable_by` is pdqsort, which takes that in linear time. A fully
/// tied cut costs 1.3-1.6x a well-separated one, not the ~50x a real full
/// sort would (`tests/rank_top_k.rs`, the tied-cut timing case).
///
/// **Partitioning the admitted set first was tried and is SLOWER**, by a
/// measured 1.4x on that shape: `select_nth_unstable_by` cannot exploit an
/// already-sorted input and pays random access into the 2 MiB key array to
/// find a cut the sort gets for free. Do not "fix" this by adding one.
///
/// What WOULD reopen the hole is breaking the collection order -- a
/// parallel or chunked collect loop, or a tie-break that is not ascending
/// index. Measured on the same shape, shuffling the admitted set takes the
/// sort from 0.304 to 5.450 ms. That is what the test above guards.
///
/// A NaN key falls back to the old path deliberately. `partial_cmp(..)
/// .unwrap_or(Equal)` makes NaN compare equal to everything, which is NOT a
/// total order, so the two routes are entitled to disagree there -- and
/// `select` cannot reach it (it rejects a non-finite score vector before
/// ranking), so the fallback is for this function's own public contract.
pub fn rank_top_k_u32_into(keys: &[f64], k: usize, out: &mut Vec<u32>) {
    let k = k.min(keys.len());
    if k == 0 {
        out.clear();
        return;
    }
    if k < keys.len() {
        if let Some(cut) = kth_largest_value(keys, k) {
            out.clear();
            for (i, &v) in keys.iter().enumerate() {
                if v >= cut {
                    out.push(i as u32);
                }
            }
            // PUSHED IN ASCENDING INDEX ORDER, and that is load-bearing
            // rather than incidental -- see this function's doc comment.
            out.sort_unstable_by(rank_order_u32(keys));
            out.truncate(k);
            return;
        }
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

/// The `k`th largest VALUE in `keys`, counting duplicates with
/// multiplicity, or `None` if any key is NaN.
///
/// One sequential pass holding the `k` largest values seen so far in an
/// unordered array with its minimum tracked. The minimum is recomputed on
/// each displacement, which is `O(k)` and looks wasteful until it is
/// counted: a displacement needs a value beating the running `k`th best, so
/// over a domain of `n` in arbitrary order it happens about `k ln(n / k)`
/// times -- ~533 times at `n = 262144, k = 64`, against 262144 sequential
/// comparisons that almost all fail. A heap would improve the term that is
/// already negligible.
///
/// Values, not indices: the caller re-derives the tie-break by sorting the
/// admitted set with the real comparator, so this deliberately knows
/// nothing about ordering beyond `>`.
fn kth_largest_value(keys: &[f64], k: usize) -> Option<f64> {
    debug_assert!(k > 0 && k <= keys.len());
    let mut top: Vec<f64> = Vec::with_capacity(k);
    let mut min = f64::INFINITY;
    let mut min_at = 0usize;
    for &v in keys {
        if v.is_nan() {
            return None;
        }
        if top.len() < k {
            top.push(v);
            if top.len() == k {
                (min_at, min) = argmin(&top);
            }
        } else if v > min {
            // Strictly greater: an equal value leaves `k` values at or
            // above `min`, so the cut has not moved.
            top[min_at] = v;
            (min_at, min) = argmin(&top);
        }
    }
    Some(min)
}

/// Position and value of the smallest entry. `values` is non-empty and
/// NaN-free by construction at every call site.
fn argmin(values: &[f64]) -> (usize, f64) {
    let mut at = 0usize;
    for (i, &v) in values.iter().enumerate().skip(1) {
        if v < values[at] {
            at = i;
        }
    }
    (at, values[at])
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
    |&a: &u32, &b: &u32| rank_cmp(keys[a as usize], keys[b as usize], a as usize, b as usize)
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

/// Keep only the entries of `surviving` whose score is at least `min_p`
/// times `ranked`'s own top score -- `ranked[0]`, since `ranked` is the
/// FULL descending order this composes against, never just whatever
/// remained after an earlier truncation step. `scores` may be normalized
/// probabilities OR any positive monotone image of them (the hot path in
/// `choose.rs` passes unnormalized `exp(s - max)` values): the comparison
/// is a RATIO against `scores[top]`, and dividing both sides by the same
/// missing normalizer cancels out, so the two are interchangeable here.
///
/// `None` means min-p truncation is disabled and `surviving` passes through
/// unfiltered. Never underflows to zero members while `surviving` is
/// non-empty: the top entry of `ranked` always clears its own threshold, so
/// if it is a member of `surviving` the result cannot be empty, and if it
/// is NOT (top-k or top-p already excluded it) an empty result falls back
/// to `surviving`'s own first entry rather than nothing.
pub fn truncate_by_min_p(
    surviving: &[usize],
    ranked: &[usize],
    scores: &[f64],
    min_p: Option<f64>,
) -> Vec<usize> {
    let Some(min_p) = min_p else {
        return surviving.to_vec();
    };
    let Some(&top) = ranked.first() else {
        return surviving.to_vec();
    };
    let threshold = min_p * scores[top];
    let kept: Vec<usize> = surviving
        .iter()
        .copied()
        .filter(|&i| scores[i] >= threshold)
        .collect();
    if kept.is_empty() && !surviving.is_empty() {
        return vec![surviving[0]];
    }
    kept
}
