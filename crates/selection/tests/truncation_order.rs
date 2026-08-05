//! Truncation composition-order scenarios.
//!
//! Corresponds to behavior-spec test-004 (obl-select-004), test-005
//! (obl-select-005), and test-010 (obl-select-010): the composed order is
//! cumulative-probability truncation first, over the full distribution,
//! then the rank-based cap over that surviving set.

use mrefrust_selection::truncation::{
    rank_indices, softmax, truncate_by_probability_mass, truncate_by_rank,
};

#[test]
fn rank_based_truncation_keeps_only_the_top_n_candidates() {
    let scores = vec![5.0f32, 4.0, 3.0, 2.0, 1.0];
    let probs = softmax(&scores);
    let ranked = rank_indices(&probs);
    let capped = truncate_by_rank(&ranked, 2);
    assert_eq!(capped, vec![0, 1]);
}

#[test]
fn probability_mass_truncation_keeps_the_smallest_sufficient_prefix() {
    // Engineered so the top two candidates hold nearly all the probability
    // mass at a 0.9 threshold.
    let scores = vec![10.0f32, 9.5, 0.0, -1.0, -2.0];
    let probs = softmax(&scores);
    let ranked = rank_indices(&probs);
    let kept = truncate_by_probability_mass(&ranked, &probs, Some(0.9));
    assert!(kept.len() <= 3);
    assert!(kept.contains(&0));
}

#[test]
fn composed_order_is_mass_truncation_first_then_rank_cap() {
    // Engineered so reaching 0.9 of the *full-distribution* probability
    // mass needs the first three ranked candidates, while a rank-based cap
    // of two then trims that surviving set down further. If rank-based
    // truncation instead ran first and the mass check were evaluated
    // against a subset renormalized to just the rank-capped candidates, the
    // threshold would already be satisfied by the rank cap alone (any
    // subset renormalizes to a total mass of 1.0), incorrectly treating the
    // third-ranked candidate as unnecessary before it is ever considered.
    let scores = vec![2.0f32, 1.8, 1.0, -1.0, -2.0];
    let probs = softmax(&scores);
    let ranked = rank_indices(&probs);

    let mass_kept = truncate_by_probability_mass(&ranked, &probs, Some(0.9));
    assert!(
        mass_kept.len() >= 3,
        "expected mass truncation against the full distribution to need at least 3 candidates, got {mass_kept:?}"
    );

    let composed = truncate_by_rank(&mass_kept, 2);
    assert_eq!(
        composed,
        vec![mass_kept[0], mass_kept[1]],
        "rank-based truncation must cap the mass-truncated set, not replace it"
    );
    assert!(composed.len() < mass_kept.len());
}
