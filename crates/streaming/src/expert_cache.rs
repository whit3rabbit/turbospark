//! Pure LFU/LRU expert-slot cache policy, split out of
//! `Infrastructure/Streaming/PreadExpertStreamer.swift` so it can be tested
//! against scripted access traces independent of any actual file I/O or GPU
//! buffer.

use std::collections::HashSet;

/// Cache eviction policy for routed experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertCachePolicy {
    /// Least Recently Used eviction.
    Lru,
    /// Least Frequently Used eviction.
    Lfu,
}

impl ExpertCachePolicy {
    /// Default cache policy (LFU).
    pub const DEFAULT: ExpertCachePolicy = ExpertCachePolicy::Lfu;
}

/// Execution plan generated for a set of requested expert indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertCachePlan {
    /// Requested expert indices.
    pub experts: Vec<usize>,
    /// Assigned slot index per requested expert.
    pub assigned_slots: Vec<usize>,
    /// Indices into `experts` that missed cache and need loading.
    pub misses: Vec<usize>,
    /// Total count of cache hits.
    pub hits: usize,
}

/// Results of OS kernel I/O advice operations (`madvise`/`fadvise`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExpertIoAdviceResult {
    /// Number of expert ranges requested for I/O advice.
    pub requested: usize,
    /// Number of advice calls that failed.
    pub failed: usize,
    /// Number of coalesced system calls made.
    pub calls: usize,
    /// Total bytes included in I/O advice.
    pub bytes: u64,
    /// Number of requested ranges skipped.
    pub skipped: usize,
    /// Maximum latency of an advice call in nanoseconds.
    pub max_call_nanos: u64,
}

impl ExpertIoAdviceResult {
    /// Constructs a skipped advice result.
    pub fn skipped(requested: usize, bytes: u64) -> Self {
        Self {
            requested,
            bytes,
            skipped: requested,
            ..Default::default()
        }
    }
}

/// Fixed per-layer slot cache: which expert (if any) each slot currently
/// holds, and the bookkeeping (last-use clock, per-expert hit counts,
/// in-flight speculative reads) the eviction policy reads.
pub struct ExpertCache {
    slot_count: usize,
    cache_policy: ExpertCachePolicy,
    slot_expert: Vec<Option<usize>>,
    slot_last_use: Vec<u64>,
    expert_use_count: Vec<u64>,
    speculative_in_flight: Vec<bool>,
    use_clock: u64,
}

impl ExpertCache {
    /// Constructs an expert slot cache.
    pub fn new(
        slot_count: usize,
        cache_policy: ExpertCachePolicy,
        experts_per_layer: usize,
    ) -> Self {
        assert!(slot_count > 0, "slot_count must be positive");
        Self {
            slot_count,
            cache_policy,
            slot_expert: vec![None; slot_count],
            slot_last_use: vec![0; slot_count],
            expert_use_count: vec![0; experts_per_layer.max(1)],
            speculative_in_flight: vec![false; slot_count],
            use_clock: 0,
        }
    }

    /// Returns total number of expert slots.
    pub fn slot_count(&self) -> usize {
        self.slot_count
    }

    /// Read-only residency probe: which of `experts` are *not* currently in
    /// a slot, deduplicated and in request order. Touches neither the LFU
    /// counters nor the use clock nor slot assignment.
    pub fn non_resident_experts(&self, experts: &[usize]) -> Vec<usize> {
        let resident: HashSet<usize> = self.slot_expert.iter().filter_map(|&e| e).collect();
        let mut seen = HashSet::new();
        experts
            .iter()
            .copied()
            .filter(|e| !resident.contains(e) && seen.insert(*e))
            .collect()
    }

    /// Builds a plan placing every expert in `experts` into a slot, or
    /// `None` if there are not enough evictable slots for the misses.
    pub fn plan_if_possible(
        &mut self,
        experts: &[usize],
        avoiding_slots: &HashSet<usize>,
    ) -> Option<ExpertCachePlan> {
        assert!(
            experts.len() <= self.slot_count,
            "expert cache needs at least {} slots",
            experts.len()
        );
        let avoiding_slots: HashSet<usize> = avoiding_slots
            .iter()
            .copied()
            .filter(|&s| s < self.slot_count)
            .collect();

        let clock = self.use_clock + 1;
        let mut assigned_slots = vec![usize::MAX; experts.len()];
        let mut reserved = vec![false; self.slot_count];

        for (index, &expert) in experts.iter().enumerate() {
            if let Some(slot) = (0..self.slot_count)
                .find(|&slot| !reserved[slot] && self.slot_expert[slot] == Some(expert))
            {
                assigned_slots[index] = slot;
                reserved[slot] = true;
            }
        }
        for &slot in &avoiding_slots {
            reserved[slot] = true;
        }
        // A slot a speculative read is still filling must not be handed to
        // the real plan: its buffer is being written from another task.
        for (slot, in_flight) in self.speculative_in_flight.iter().enumerate() {
            if *in_flight {
                reserved[slot] = true;
            }
        }

        let misses: Vec<usize> = (0..experts.len())
            .filter(|&i| assigned_slots[i] == usize::MAX)
            .collect();
        let mut evictable: Vec<usize> = (0..self.slot_count).filter(|&s| !reserved[s]).collect();
        evictable.sort_by(|&a, &b| self.eviction_order(a, b));
        if misses.len() > evictable.len() {
            return None;
        }

        self.use_clock = clock;
        for &expert in experts {
            if expert < self.expert_use_count.len() {
                self.expert_use_count[expert] = self.expert_use_count[expert].saturating_add(1);
            }
        }
        for &slot in &assigned_slots {
            if slot != usize::MAX {
                self.slot_last_use[slot] = clock;
            }
        }
        for (offset, &index) in misses.iter().enumerate() {
            let slot = evictable[offset];
            assigned_slots[index] = slot;
            self.slot_expert[slot] = None;
            self.slot_last_use[slot] = clock;
        }

        let hits = experts.len() - misses.len();
        Some(ExpertCachePlan {
            experts: experts.to_vec(),
            assigned_slots,
            misses,
            hits,
        })
    }

    /// Same as [`ExpertCache::plan_if_possible`], but panics (matching the
    /// Swift `preconditionFailure`) when the cache cannot place the
    /// requested misses.
    pub fn plan(&mut self, experts: &[usize], avoiding_slots: &HashSet<usize>) -> ExpertCachePlan {
        self.plan_if_possible(experts, avoiding_slots)
            .expect("expert cache cannot place requested misses")
    }

    /// Marks a plan's miss slots as now holding their assigned expert.
    /// Call after the corresponding reads have completed successfully.
    pub fn commit_plan(&mut self, plan: &ExpertCachePlan) {
        for &index in &plan.misses {
            self.slot_expert[plan.assigned_slots[index]] = Some(plan.experts[index]);
        }
    }

    /// Reserves slots for a speculative read of `experts` (already filtered
    /// through `non_resident_experts`). Victims are picked with the normal
    /// eviction order but no LFU/clock bookkeeping is written, since an
    /// unconfirmed guess must not shift the policy. `keep_evictable` slots
    /// are left untouched so the plan that follows always has room.
    pub fn reserve_speculative_slots(
        &mut self,
        experts: &[usize],
        keep_evictable: usize,
    ) -> Vec<(usize, usize)> {
        if experts.is_empty() {
            return Vec::new();
        }
        let wanted: HashSet<usize> = experts.iter().copied().collect();
        let mut reserved = vec![false; self.slot_count];
        for (slot, (in_flight, expert)) in self
            .speculative_in_flight
            .iter()
            .zip(self.slot_expert.iter())
            .enumerate()
        {
            let holds_wanted = expert.is_some_and(|e| wanted.contains(&e));
            if *in_flight || holds_wanted {
                reserved[slot] = true;
            }
        }
        let mut evictable: Vec<usize> = (0..self.slot_count).filter(|&s| !reserved[s]).collect();
        evictable.sort_by(|&a, &b| self.eviction_order(a, b));
        let budget = experts
            .len()
            .min(evictable.len().saturating_sub(keep_evictable));
        if budget == 0 {
            return Vec::new();
        }

        let mut reservation = Vec::with_capacity(budget);
        for (expert, &slot) in experts.iter().zip(evictable.iter()).take(budget) {
            self.speculative_in_flight[slot] = true;
            self.slot_expert[slot] = None;
            reservation.push((*expert, slot));
        }
        reservation
    }

    /// Publishes a speculative reservation's outcome: `loaded` names the
    /// indices into `reservation` whose read succeeded. Slots whose read
    /// failed stay empty; the in-flight mark is always cleared.
    pub fn publish_speculative_reservation(
        &mut self,
        reservation: &[(usize, usize)],
        loaded: &[usize],
    ) {
        for &index in loaded {
            let (expert, slot) = reservation[index];
            self.slot_expert[slot] = Some(expert);
        }
        for &(_, slot) in reservation {
            self.speculative_in_flight[slot] = false;
        }
    }

    /// Test/diagnostic view of slot residency.
    pub fn resident_experts_snapshot(&self) -> Vec<Option<usize>> {
        self.slot_expert.clone()
    }

    fn eviction_order(&self, lhs: usize, rhs: usize) -> std::cmp::Ordering {
        if self.cache_policy == ExpertCachePolicy::Lru {
            return self.slot_last_use[lhs].cmp(&self.slot_last_use[rhs]);
        }
        let lhs_expert = self.slot_expert[lhs];
        let rhs_expert = self.slot_expert[rhs];
        match (lhs_expert, rhs_expert) {
            (None, None) => std::cmp::Ordering::Equal,
            (None, Some(_)) => std::cmp::Ordering::Less,
            (Some(_), None) => std::cmp::Ordering::Greater,
            (Some(le), Some(re)) => {
                let lc = self.expert_use_count.get(le).copied().unwrap_or(0);
                let rc = self.expert_use_count.get(re).copied().unwrap_or(0);
                lc.cmp(&rc)
                    .then_with(|| self.slot_last_use[lhs].cmp(&self.slot_last_use[rhs]))
            }
        }
    }
}

/// Merges overlapping/adjacent `(offset, count)` byte ranges into the
/// minimal set of non-overlapping ranges covering the same bytes.
pub fn coalesced_adjacent_advice_ranges(ranges: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut sorted: Vec<(u64, u64)> = ranges.iter().copied().filter(|&(_, c)| c > 0).collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut result: Vec<(u64, u64)> = Vec::new();
    for range in sorted {
        if let Some(last) = result.last_mut() {
            let last_end = last.0 + last.1;
            let range_end = range.0 + range.1;
            if range.0 <= last_end {
                last.1 = last_end.max(range_end) - last.0;
                continue;
            }
        }
        result.push(range);
    }
    result
}
