//! `KvCacheManager`'s rewind and reset methods, split out of `kv_cache.rs`
//! per that file's own size (Gotcha: "keep code files under 400 lines").
//! A child module of `kv_cache`, following `attention_decode_tests.rs`'s
//! sibling-file precedent, so this `impl` block can reach the struct's
//! private fields directly.

use super::KvCacheManager;
use crate::kv_cache_mem::{advise_dontneed, page_size_bytes};

impl KvCacheManager {
    /// How many tokens [`Self::rewind_by`] can drop and still leave every
    /// layer's readable window intact. `usize::MAX` when nothing rings.
    ///
    /// A full-attention layer stores position `p` at slot `p` and is read
    /// over `[0, position)`, so a rewind only shrinks the range and any
    /// count is safe. A ring layer stores at `p % capacity` and is read
    /// over the last `swa_window` positions, so writing `k` tokens past a
    /// point overwrites the `k` slots holding positions `[p-k-capacity,
    /// p-capacity)`. Rewinding to `p-k` then needs `[p-k-window, p-k)`, and
    /// those survive exactly when `capacity >= window + k`.
    ///
    /// The slack is real rather than lucky: `new` sizes a ring at
    /// `sliding_window + max_prefill_chunk_tokens`, so the chunk budget is
    /// also the rewind budget. A ring that was never allowed to wrap
    /// (`capacity == max_context`) has no constraint at all.
    pub fn max_safe_rewind(&self) -> usize {
        let mut budget = usize::MAX;
        for layer in 0..self.num_layers {
            if self.ring_capacity(layer) == 0 || self.capacity_tokens[layer] >= self.max_context {
                continue;
            }
            budget = budget.min(self.capacity_tokens[layer].saturating_sub(self.swa_window));
        }
        budget
    }

    /// Moves the cursor back `count` tokens, discarding the rows written at
    /// `[position - count, position)`. Rows are addressed by ABSOLUTE
    /// position and views are cut at `valid_token_count`, so nothing has to
    /// be erased: the next writes overwrite the same slots.
    ///
    /// This is the attention half of a speculative-decoding rollback (the
    /// recurrent half is `GdnStateManager::restore`). Panics rather than
    /// silently corrupting when a ring layer's window would lose rows; see
    /// [`Self::max_safe_rewind`].
    pub fn rewind_by(&mut self, count: usize) {
        assert!(count <= self.position, "rewind below position 0");
        let budget = self.max_safe_rewind();
        assert!(
            count <= budget,
            "rewind of {count} exceeds the ring slack of {budget}: a sliding-window \
             layer would read rows this generation has already overwritten"
        );
        self.position -= count;
    }

    /// Drops all cached positions and returns physical pages to the OS via
    /// `MADV_DONTNEED`, so a finished generation does not keep its KV
    /// resident into the next turn.
    pub fn reset(&mut self) {
        self.position = 0;
        let page_size = page_size_bytes();
        let mut advised: Vec<*const std::ffi::c_void> = Vec::new();
        for layer in 0..self.num_layers {
            advise_dontneed(&self.k_buffers[layer], page_size, &mut advised);
            advise_dontneed(&self.v_buffers[layer], page_size, &mut advised);
        }
    }
}
