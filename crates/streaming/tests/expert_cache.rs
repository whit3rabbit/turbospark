//! Scripted access-trace tests for the LFU/LRU expert-slot cache policy.

use std::collections::HashSet;

use mrefrust_streaming::{coalesced_adjacent_advice_ranges, ExpertCache, ExpertCachePolicy};

#[test]
fn cold_cache_all_misses() {
    let mut cache = ExpertCache::new(4, ExpertCachePolicy::Lfu, 16);
    let plan = cache.plan(&[1, 2, 3], &HashSet::new());
    assert_eq!(plan.hits, 0);
    assert_eq!(plan.misses, vec![0, 1, 2]);
    assert_eq!(plan.assigned_slots.len(), 3);
}

#[test]
fn repeated_request_is_a_hit_after_commit() {
    let mut cache = ExpertCache::new(4, ExpertCachePolicy::Lfu, 16);
    let plan = cache.plan(&[1, 2], &HashSet::new());
    cache.commit_plan(&plan);

    let plan2 = cache.plan(&[1], &HashSet::new());
    assert_eq!(plan2.hits, 1);
    assert_eq!(plan2.misses, Vec::<usize>::new());
    assert_eq!(plan2.assigned_slots[0], plan.assigned_slots[0]);
}

#[test]
fn lfu_evicts_the_least_frequently_used_expert() {
    let mut cache = ExpertCache::new(2, ExpertCachePolicy::Lfu, 16);
    // Fill both slots.
    let p1 = cache.plan(&[1, 2], &HashSet::new());
    cache.commit_plan(&p1);
    // Touch expert 1 again so its use count is higher than expert 2's.
    let p2 = cache.plan(&[1], &HashSet::new());
    cache.commit_plan(&p2);
    assert_eq!(p2.hits, 1);

    // A new expert must evict the least-used resident (expert 2), not
    // expert 1.
    let p3 = cache.plan(&[3], &HashSet::new());
    cache.commit_plan(&p3);
    let resident = cache.resident_experts_snapshot();
    assert!(resident.contains(&Some(1)), "resident = {resident:?}");
    assert!(resident.contains(&Some(3)), "resident = {resident:?}");
    assert!(!resident.contains(&Some(2)), "resident = {resident:?}");
}

#[test]
fn lru_evicts_the_least_recently_used_slot() {
    let mut cache = ExpertCache::new(2, ExpertCachePolicy::Lru, 16);
    let p1 = cache.plan(&[1, 2], &HashSet::new());
    cache.commit_plan(&p1);
    // Touch expert 1 again, making expert 2 the least-recently-used slot
    // even though both were only requested once each.
    let p2 = cache.plan(&[1], &HashSet::new());
    cache.commit_plan(&p2);

    let p3 = cache.plan(&[3], &HashSet::new());
    cache.commit_plan(&p3);
    let resident = cache.resident_experts_snapshot();
    assert!(resident.contains(&Some(1)));
    assert!(resident.contains(&Some(3)));
    assert!(!resident.contains(&Some(2)));
}

#[test]
fn plan_returns_none_when_misses_exceed_evictable_slots() {
    let mut cache = ExpertCache::new(2, ExpertCachePolicy::Lfu, 16);
    // Avoid both slots, so nothing is evictable, but request 1 miss.
    let avoiding: HashSet<usize> = [0, 1].into_iter().collect();
    let plan = cache.plan_if_possible(&[7], &avoiding);
    assert!(plan.is_none());
}

#[test]
fn non_resident_experts_deduplicates_and_preserves_request_order() {
    let mut cache = ExpertCache::new(4, ExpertCachePolicy::Lfu, 16);
    let plan = cache.plan(&[1], &HashSet::new());
    cache.commit_plan(&plan);

    let missing = cache.non_resident_experts(&[1, 2, 3, 2, 1]);
    assert_eq!(missing, vec![2, 3]);
}

#[test]
fn speculative_reservation_does_not_shift_lfu_bookkeeping_until_published() {
    let mut cache = ExpertCache::new(2, ExpertCachePolicy::Lfu, 16);
    let p1 = cache.plan(&[1, 2], &HashSet::new());
    cache.commit_plan(&p1);

    let reservation = cache.reserve_speculative_slots(&[3], 0);
    assert_eq!(reservation.len(), 1);
    // The reserved slot is marked empty while the (simulated) read is in
    // flight, so it does not look like a hit yet.
    let resident_mid_flight = cache.resident_experts_snapshot();
    assert!(resident_mid_flight.iter().filter(|e| e.is_some()).count() == 1);

    cache.publish_speculative_reservation(&reservation, &[0]);
    let resident = cache.resident_experts_snapshot();
    assert!(resident.contains(&Some(3)));
}

#[test]
fn speculative_reservation_failed_read_leaves_slot_empty() {
    let mut cache = ExpertCache::new(2, ExpertCachePolicy::Lfu, 16);
    let p1 = cache.plan(&[1, 2], &HashSet::new());
    cache.commit_plan(&p1);

    let reservation = cache.reserve_speculative_slots(&[3], 0);
    // Publish with no loaded indices, simulating a failed read.
    cache.publish_speculative_reservation(&reservation, &[]);
    let resident = cache.resident_experts_snapshot();
    assert!(!resident.contains(&Some(3)));
}

#[test]
fn coalesced_adjacent_advice_ranges_merges_touching_and_overlapping_ranges() {
    let ranges = [(0, 10), (10, 5), (100, 20), (105, 5)];
    let merged = coalesced_adjacent_advice_ranges(&ranges);
    // (105, 5) ends at 110, fully inside (100, 20) which ends at 120, so it
    // contributes nothing beyond the existing range.
    assert_eq!(merged, vec![(0, 15), (100, 20)]);
}

#[test]
fn coalesced_adjacent_advice_ranges_drops_zero_length_ranges() {
    let ranges = [(0, 0), (5, 10)];
    let merged = coalesced_adjacent_advice_ranges(&ranges);
    assert_eq!(merged, vec![(5, 10)]);
}
