//! Min-p truncation: threshold edges against the reference
//! `truncate_by_min_p`, composition order against an earlier top-k cut, and
//! an end-to-end effect scenario through the composed selection pipeline.

use std::collections::HashMap;

use foundation::{LogitValue, LogitsView, TokenId};
use turbospark_selection::truncation::{truncate_by_min_p, truncate_by_rank};
use turbospark_selection::{select, ShapingConfig};

#[test]
fn a_score_exactly_at_the_threshold_is_kept() {
    let ranked = vec![0, 1, 2];
    let scores = vec![10.0, 5.0, 1.0];
    // threshold = 0.5 * 10.0 = 5.0; index 1's score sits exactly on it.
    let kept = truncate_by_min_p(&ranked, &ranked, &scores, Some(0.5));
    assert_eq!(kept, vec![0, 1]);
}

#[test]
fn a_score_just_below_the_threshold_is_excluded() {
    let ranked = vec![0, 1, 2];
    let scores = vec![10.0, 4.999, 1.0];
    let kept = truncate_by_min_p(&ranked, &ranked, &scores, Some(0.5));
    assert_eq!(kept, vec![0]);
}

/// **THE COMPOSITION-ORDER TEST.** Min-p alone, over the full domain, would
/// admit indices 0, 1, AND 2 (`4.0` and `3.0` both clear `0.3 * 8.0 = 2.4`).
/// Composed AFTER a top-k cut of 1, only index 0 survived that cut in the
/// first place, so min-p can only narrow it further -- never reintroduce a
/// candidate an earlier stage already excluded.
#[test]
fn min_p_composes_after_an_earlier_top_k_cut_and_cannot_reintroduce_excluded_candidates() {
    let ranked = vec![0, 1, 2, 3];
    let scores = vec![8.0, 4.0, 3.0, 0.1];

    let post_top_k = truncate_by_rank(&ranked, 1);
    assert_eq!(post_top_k, vec![0]);

    let kept = truncate_by_min_p(&post_top_k, &ranked, &scores, Some(0.3));
    assert_eq!(
        kept,
        vec![0],
        "min-p must not reintroduce candidates 1 or 2, which a prior top-k=1 already excluded"
    );
}

#[test]
fn none_disables_min_p_and_returns_surviving_unfiltered() {
    let surviving = vec![0, 1, 2];
    let ranked = vec![0, 1, 2];
    let scores = vec![10.0, 0.001, 0.0001];
    let kept = truncate_by_min_p(&surviving, &ranked, &scores, None);
    assert_eq!(kept, surviving);
}

#[test]
fn an_empty_surviving_set_stays_empty() {
    let kept = truncate_by_min_p(&[], &[0, 1], &[10.0, 1.0], Some(0.5));
    assert!(kept.is_empty());
}

fn logits(values: &[f32]) -> Vec<LogitValue> {
    values.iter().map(|&v| LogitValue::from_f32(v)).collect()
}

/// End to end through `select`: a high min_p suppresses the low-probability
/// tail entirely.
///
/// **CANDIDATE 3's BASELINE SHARE HAS TO BE LARGE ENOUGH TO OBSERVE, OR THE
/// TEST PROVES NOTHING.** A first version of this test used logits far
/// enough apart (`[5.0, 4.5, 4.0, -5.0]`) that candidate 3's UNPENALIZED
/// share was already ~0 in 500 trials -- disabling the min_p wiring
/// entirely left the test passing, since "never drawn" was already true
/// beforehand. `[2.0, 1.0, 0.0, -1.0]` at temperature 1.0 gives candidate 3
/// an unpenalized share around 3% (expected ~16 of 500 trials, so its
/// absence is not survivorship noise), while `min_p: 0.5`'s threshold
/// (`0.5 * exp(2.0 - 2.0) = 0.5`) excludes every candidate but 0
/// (`exp(1.0-2.0) ~= 0.37 < 0.5`) -- checked with mutation: deleting the
/// min_p wiring in `choose.rs` left THIS version of the test failing.
#[test]
fn a_high_min_p_suppresses_low_probability_candidates_from_ever_being_drawn() {
    let scores = logits(&[2.0, 1.0, 0.0, -1.0]);
    let view = LogitsView::new(&scores);
    let history: Vec<TokenId> = Vec::new();

    let without_min_p = ShapingConfig::new(1.0, 0, None, 1.0, None).unwrap();
    let with_min_p = ShapingConfig::new(1.0, 0, None, 1.0, None)
        .unwrap()
        .with_min_p(0.5)
        .unwrap();

    let mut baseline = HashMap::new();
    let mut suppressed = HashMap::new();
    for position in 0..500u64 {
        *baseline
            .entry(select(view, &without_min_p, &history, position).unwrap())
            .or_insert(0u32) += 1;
        *suppressed
            .entry(select(view, &with_min_p, &history, position).unwrap())
            .or_insert(0u32) += 1;
    }
    assert!(
        baseline.get(&3).is_some_and(|&n| n > 0),
        "candidate 3 should appear without min_p, or this test cannot distinguish disabled \
         from enabled: {baseline:?}"
    );
    assert_eq!(
        suppressed.get(&3),
        None,
        "candidate 3 should never survive a 0.5 min_p threshold against candidate 0's peak: {suppressed:?}"
    );
}

#[test]
fn zero_min_p_is_bit_identical_to_disabled() {
    let scores = logits(&[1.0, 2.0, 0.5, 3.0]);
    let view = LogitsView::new(&scores);
    let history: Vec<TokenId> = Vec::new();

    let disabled = ShapingConfig::new(0.8, 4, Some(0.95), 1.0, Some(3)).unwrap();
    let explicit_zero = ShapingConfig::new(0.8, 4, Some(0.95), 1.0, Some(3))
        .unwrap()
        .with_min_p(0.0)
        .unwrap();

    for position in 0..50u64 {
        assert_eq!(
            select(view, &disabled, &history, position).unwrap(),
            select(view, &explicit_zero, &history, position).unwrap(),
        );
    }
}
