//! A bounded pool of PARKED per-session KV/recurrent state, so one runner
//! can serve several distinct conversations without each turn discarding
//! the others' reusable state.
//!
//! `crate::kv_prefix::KvPrefix::common_prefix` already answers "how much of
//! ONE recorded state does this prompt continue"; this pool generalizes
//! that to N tracked candidates. `RealForwardRunner` keeps its existing
//! single `kv`/`real_qwen.gdn`/`kv_prefix` fields as the "live" session, and
//! this pool holds `session_slots - 1` others, pre-allocated at open. A
//! swap moves a `SessionSlot`'s `kv`/`gdn`/`kv_prefix` onto the live fields
//! via `std::mem::replace` (an O(1) struct move, never a memcpy) and parks
//! whatever was live in its place. See `RealForwardRunner::select_session`
//! and `reset()` in `real_forward_traits.rs` for the two call sites.
//!
//! Capacity is fixed at open and never changes: every `take`/`take_lru` call
//! is paired with exactly one `park` call by its caller, so the pool's
//! `Vec` length never moves after construction.
//!
//! LRU eviction uses a monotonic TICK rather than a wall-clock timestamp,
//! mirroring `crates/streaming`'s `ExpertCache` (`slot_last_use`/
//! `use_clock`): eviction only needs relative order, and a counter is
//! deterministic to unit-test where two `Instant::now()` calls in a tight
//! loop are not.

use foundation::TokenId;

use crate::kv_prefix::KvPrefix;

/// One session's KV and (on a GDN family) recurrent state, independently
/// allocated so it can be swapped onto `RealForwardRunner`'s own fields.
pub(crate) struct SessionSlot {
    pub(crate) kv: gpu::KvCacheManager,
    /// `None` on every family but the two qwen halves: `GdnStateManager` is
    /// only ever built for `RealQwenState` (`crates/gpu/CLAUDE.md`'s
    /// `gdn_state.rs` entry). A GDN-family pool allocates one per slot; a
    /// non-GDN family's pool allocates none.
    pub(crate) gdn: Option<gpu::GdnStateManager>,
    pub(crate) kv_prefix: KvPrefix,
    /// See the module doc: a monotonic tick, stamped by
    /// [`SessionPool::park`], never read or set anywhere else.
    pub(crate) last_used: u64,
}

/// A fixed-size pool of parked slots. See the module doc for why the "live"
/// session is NOT a member of this pool: it stays on `RealForwardRunner`'s
/// own fields, which is what keeps every family's dispatch code
/// (`families/*/attn.rs`, `moe.rs`, `prefill.rs`) unaware this exists.
pub(crate) struct SessionPool {
    slots: Vec<SessionSlot>,
    clock: u64,
}

impl SessionPool {
    pub(crate) fn new(slots: Vec<SessionSlot>) -> Self {
        Self { slots, clock: 0 }
    }

    /// `--session-slots 1` (the default) or `0`: no parked slots, and every
    /// method below is then unreachable, per `RealForwardRunner::select_session`
    /// and `reset()`'s own `capacity() > 0` guards. This is the whole
    /// byte-identity guarantee for the default case.
    pub(crate) fn empty() -> Self {
        Self::new(Vec::new())
    }

    pub(crate) fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// The best-scoring parked slot for `prompt_ids` and its score, or
    /// `None` when the pool is empty or every parked slot scores zero.
    /// Zero is filtered out rather than offered as a "match": promoting one
    /// unrelated parked slot over another buys nothing and still pays the
    /// swap. The caller decides whether the returned score beats the LIVE
    /// session's own (`KvPrefix::common_prefix` against
    /// `RealForwardRunner::kv_prefix`) -- this method never sees that value,
    /// only the parked candidates.
    pub(crate) fn best_match(&self, prompt_ids: &[TokenId]) -> Option<(usize, usize)> {
        best_scoring_index(
            self.slots
                .iter()
                .map(|slot| slot.kv_prefix.common_prefix(prompt_ids)),
        )
    }

    /// Removes and returns slot `idx`. Always paired with exactly one
    /// `park` call from the same caller (the outgoing live slot), so the
    /// pool's size never moves across a swap.
    pub(crate) fn take(&mut self, idx: usize) -> SessionSlot {
        self.slots.remove(idx)
    }

    /// Removes and returns the least-recently-used slot. Only ever called
    /// on a nonempty pool, by `reset()`'s park-and-promote path, which is
    /// itself gated on `capacity() > 0`.
    pub(crate) fn take_lru(&mut self) -> SessionSlot {
        let idx = lru_index(self.slots.iter().map(|slot| slot.last_used))
            .expect("take_lru called on an empty pool");
        self.slots.remove(idx)
    }

    /// Inserts `slot`, stamping it with the current tick so it reads as the
    /// most-recently-used slot in the pool until something else is parked
    /// after it.
    pub(crate) fn park(&mut self, mut slot: SessionSlot) {
        self.clock += 1;
        slot.last_used = self.clock;
        self.slots.push(slot);
    }
}

/// The index (and value) of the highest score, or `None` on an empty or
/// all-zero iterator. A pure function so the scoring rule is unit-testable
/// with no `KvCacheManager`/`GdnStateManager` in the loop -- both need a
/// real Metal device to construct, which `SessionSlot` otherwise requires.
fn best_scoring_index(scores: impl Iterator<Item = usize>) -> Option<(usize, usize)> {
    scores
        .enumerate()
        .max_by_key(|&(_, score)| score)
        .filter(|&(_, score)| score > 0)
}

/// The index of the smallest tick, or `None` on an empty iterator.
fn lru_index(ticks: impl Iterator<Item = u64>) -> Option<usize> {
    ticks
        .enumerate()
        .min_by_key(|&(_, tick)| tick)
        .map(|(idx, _)| idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_highest_score_wins() {
        assert_eq!(best_scoring_index([3, 5, 2].into_iter()), Some((1, 5)));
    }

    #[test]
    fn an_empty_iterator_never_scores() {
        assert_eq!(best_scoring_index(std::iter::empty()), None);
    }

    /// An all-zero pool is not offered as a match at all: every candidate
    /// shares nothing with the prompt, so there is nothing worth swapping
    /// to. `RealForwardRunner::select_session` relies on this to skip the
    /// swap entirely rather than promoting an arbitrary unrelated slot.
    #[test]
    fn an_all_zero_pool_offers_no_match() {
        assert_eq!(best_scoring_index([0, 0, 0].into_iter()), None);
    }

    /// A single nonzero score among zeros still wins, and at its own value
    /// rather than a rounded-up one -- the caller compares this exact
    /// number against the live session's own score.
    #[test]
    fn a_lone_nonzero_score_wins_at_its_own_value() {
        assert_eq!(best_scoring_index([0, 7, 0].into_iter()), Some((1, 7)));
    }

    #[test]
    fn the_oldest_tick_is_evicted() {
        assert_eq!(lru_index([5, 1, 3].into_iter()), Some(1));
    }

    #[test]
    fn an_empty_pool_has_no_lru() {
        assert_eq!(lru_index(std::iter::empty()), None);
    }
}
