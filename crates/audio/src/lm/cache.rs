//! KV caches for the shared LM backbones.
//!
//! Single-batch port of `mlx_audio/lm/models/cache.py` `KVCache` and
//! `RotatingKVCache` (vendored from mlx-lm) at the pinned mlx-audio commit.
//! `update_and_fetch` appends the new positions and returns the full
//! `[heads, total, head_dim]` buffers; `offset` counts every position ever
//! cached, which is also the rope offset the attention layers read before
//! the update. `RotatingKvCache` keeps at most `max_size` rows in a
//! physically rotated buffer, exactly mirroring the reference `_trim`,
//! `_temporal_order`, `_update_concat`, and `_update_in_place` paths; the
//! sliding-window attention that consumes it applies the reference mask
//! semantics (see `engine::Attention`). The batched `BatchKVCache`,
//! `ArraysCache`, and `CacheList` variants stay unported until a family
//! needs them.

/// The cache operations the shared attention reads. Implemented by
/// [`KvCache`], [`RotatingKvCache`], and [`PromptCache`].
pub trait LayerKv {
    /// Positions already cached (the pre-update rope offset).
    fn offset(&self) -> usize;

    /// Appends `keys`/`values` `[n_kv_heads, new_positions, head_dim]` and
    /// returns the readable buffer pair.
    fn update_and_fetch(
        &mut self,
        keys: &[f32],
        values: &[f32],
        n_kv_heads: usize,
        new_positions: usize,
        head_dim: usize,
    ) -> (&[f32], &[f32]);

    /// Drops every cached position.
    fn reset(&mut self);
}

/// Appending KV cache for one layer, batch 1. Keys and values are
/// `[n_kv_heads, total_positions, head_dim]`, row-major.
#[derive(Debug, Clone, Default)]
pub struct KvCache {
    /// Positions already cached; also the rope offset for the next update.
    pub offset: usize,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    n_kv_heads: usize,
    head_dim: usize,
}

impl KvCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Appends `keys`/`values` `[n_kv_heads, new_positions, head_dim]` and
    /// returns the concatenated `[n_kv_heads, offset + new, head_dim]` pair
    /// so the attention reads one contiguous buffer. The insert is per kv
    /// head: a flat extend would land the new rows inside the last head's
    /// region and corrupt every slot after the first append.
    pub fn update_and_fetch(
        &mut self,
        keys: &[f32],
        values: &[f32],
        n_kv_heads: usize,
        new_positions: usize,
        head_dim: usize,
    ) -> (&[f32], &[f32]) {
        if self.offset == 0 {
            self.n_kv_heads = n_kv_heads;
            self.head_dim = head_dim;
        } else {
            debug_assert_eq!(self.n_kv_heads, n_kv_heads);
            debug_assert_eq!(self.head_dim, head_dim);
        }
        let span = new_positions * head_dim;
        for head in 0..n_kv_heads {
            // New rows go after this head's old rows; every earlier head
            // already grew by `span`, which shifts the insert point.
            let insert_at = (head + 1) * self.offset * head_dim + head * span;
            self.keys.splice(
                insert_at..insert_at,
                keys[head * span..(head + 1) * span].iter().copied(),
            );
            self.values.splice(
                insert_at..insert_at,
                values[head * span..(head + 1) * span].iter().copied(),
            );
        }
        self.offset += new_positions;
        (&self.keys, &self.values)
    }

    /// Clears every cached position (the reference `@property offset`
    /// reset path families use between segments).
    pub fn reset(&mut self) {
        self.offset = 0;
        self.keys.clear();
        self.values.clear();
    }
}

impl LayerKv for KvCache {
    fn offset(&self) -> usize {
        self.offset
    }

    fn update_and_fetch(
        &mut self,
        keys: &[f32],
        values: &[f32],
        n_kv_heads: usize,
        new_positions: usize,
        head_dim: usize,
    ) -> (&[f32], &[f32]) {
        KvCache::update_and_fetch(self, keys, values, n_kv_heads, new_positions, head_dim)
    }

    fn reset(&mut self) {
        KvCache::reset(self)
    }
}

/// Sliding-window cache: a `[heads, max_size, head_dim]` ring buffer with
/// the reference `RotatingKVCache` bookkeeping. Single batch.
///
/// Buffer states, mirroring the Python exactly:
///
/// - Multi-position updates append whole blocks after a `_temporal_order`
///   restore and trim to `max_size - 1 + new_positions`, so the returned
///   buffer is always in temporal order (slot 0 is the oldest kept
///   position). The reference never trims on the first call, so a prompt
///   longer than the window stays untrimmed until the next update.
/// - Single-position updates write in place at `idx`, wrapping to `keep`
///   once the buffer is full; the returned buffer is then physically
///   rotated, and the newest position sits at slot `(idx - 1) % max_size`.
///   Every returned row is always one of the last `min(offset, max_size)`
///   positions, so the window attention may treat the buffer as "the last
///   positions" and needs no per-slot position map.
///
/// The reference grows the buffer in `step = 256` row chunks and trims the
/// zero padding away on return; this port grows by exactly the new rows,
/// which returns byte-identical buffers (the padding rows are never
/// observable).
#[derive(Debug, Clone, Default)]
pub struct RotatingKvCache {
    /// Every position ever written; also the rope offset for the next
    /// update (the ring never resets it).
    pub offset: usize,
    /// Window size: the maximum rows the buffer retains once full.
    pub max_size: usize,
    /// Rows kept at the front through every trim (cohere2 uses 0).
    pub keep: usize,
    /// Next write slot, mirroring the reference `_idx`.
    idx: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
    n_kv_heads: usize,
    head_dim: usize,
}

impl RotatingKvCache {
    pub fn new(max_size: usize, keep: usize) -> Self {
        Self {
            max_size,
            keep,
            ..Self::default()
        }
    }

    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// `update_and_fetch`: the reference branches on the incoming sequence
    /// length, not on cache state.
    pub fn update_and_fetch(
        &mut self,
        keys: &[f32],
        values: &[f32],
        n_kv_heads: usize,
        new_positions: usize,
        head_dim: usize,
    ) -> (&[f32], &[f32]) {
        if self.offset == 0 {
            self.n_kv_heads = n_kv_heads;
            self.head_dim = head_dim;
        } else {
            debug_assert_eq!(self.n_kv_heads, n_kv_heads);
            debug_assert_eq!(self.head_dim, head_dim);
        }
        if new_positions == 1 {
            self.update_in_place(keys, values, n_kv_heads, head_dim);
        } else {
            self.update_concat(keys, values, n_kv_heads, new_positions, head_dim);
        }
        (&self.keys, &self.values)
    }

    /// Port of `_update_in_place`: grow while below the window, trim the
    /// oldest rows past it, wrap the write slot, then write the single new
    /// row per head.
    fn update_in_place(&mut self, keys: &[f32], values: &[f32], heads: usize, dim: usize) {
        let len = self.keys.len() / (heads * dim);
        let prev = self.offset;
        if self.keys.is_empty() || (prev >= len && len < self.max_size) {
            // The reference allocates min(step, max_size - prev) rows; the
            // padding beyond the new row is never returned, so this port
            // appends exactly the new rows.
            self.append_rows(heads, dim, 1);
            self.idx = prev;
        }
        let len = self.keys.len() / (heads * dim);
        let trim_size = len as isize - self.max_size as isize;
        if trim_size > 0 {
            // _trim: keep [0..keep) and [trim + keep..len), dropping the
            // (len - max_size) rows in between.
            let trim = trim_size as usize;
            self.retain_row_ranges(heads, dim, &[(0, self.keep), (trim + self.keep, len)]);
            self.idx = self.max_size;
        }
        if self.idx == self.max_size {
            self.idx = self.keep;
        }
        self.write_rows(keys, values, heads, dim, self.idx, 1);
        self.offset += 1;
        self.idx += 1;
    }

    /// Port of `_update_concat`: restore temporal order, trim to
    /// `max_size - 1 + new_positions` keeping `keep` front rows, append.
    fn update_concat(
        &mut self,
        keys: &[f32],
        values: &[f32],
        heads: usize,
        new_positions: usize,
        dim: usize,
    ) {
        if self.keys.is_empty() {
            self.keys = keys.to_vec();
            self.values = values.to_vec();
        } else {
            self.temporal_order(heads, dim);
            let len = self.keys.len() / (heads * dim);
            self.idx = len;
            let trim_size = self.idx as isize - self.max_size as isize + 1;
            if trim_size > 0 {
                let trim = trim_size as usize;
                let kept = len - trim;
                self.retain_row_ranges(heads, dim, &[(0, self.keep), (trim + self.keep, len)]);
                self.append_rows(heads, dim, new_positions);
                // New rows follow the kept rows inside every head's region.
                self.write_rows(keys, values, heads, dim, kept, new_positions);
            } else {
                self.append_rows(heads, dim, new_positions);
                self.write_rows(keys, values, heads, dim, len, new_positions);
            }
        }
        self.offset += new_positions;
        self.idx = self.keys.len() / (heads * dim);
    }

    /// Port of `_temporal_order`: rotate a wrapped buffer back into
    /// temporal order as `[0..keep) ++ [idx..len) ++ [keep..idx)`. The
    /// slice-else branch (`idx >= offset` past the identity case) mirrors
    /// the reference's leading-slice clamp.
    fn temporal_order(&mut self, heads: usize, dim: usize) {
        let len = self.keys.len() / (heads * dim);
        if self.idx == len {
            return;
        }
        let ranges: Vec<(usize, usize)> = if self.idx < self.offset {
            let mut ranges = Vec::with_capacity(3);
            if self.keep > 0 {
                ranges.push((0, self.keep));
            }
            ranges.push((self.idx, len));
            if self.idx > self.keep {
                ranges.push((self.keep, self.idx));
            }
            ranges
        } else {
            vec![(0, self.idx.min(len))]
        };
        self.retain_row_ranges(heads, dim, &ranges);
    }

    /// Keeps, per head, the concatenation of `ranges` (absolute row spans
    /// in the current buffer). This is `_trim` generalized over its
    /// concatenate list.
    fn retain_row_ranges(&mut self, heads: usize, dim: usize, ranges: &[(usize, usize)]) {
        let rows = self.keys.len() / (heads * dim);
        let mut keys = Vec::with_capacity(self.keys.len());
        let mut values = Vec::with_capacity(self.values.len());
        for head in 0..heads {
            let base = head * rows * dim;
            for &(start, end) in ranges {
                let (start, end) = (start.min(rows), end.min(rows));
                if end > start {
                    keys.extend_from_slice(&self.keys[base + start * dim..base + end * dim]);
                    values.extend_from_slice(&self.values[base + start * dim..base + end * dim]);
                }
            }
        }
        self.keys = keys;
        self.values = values;
    }

    /// Appends `rows` zero rows at every head's tail (the reference's zero
    /// grow chunk). The insert points derive from the pre-append length,
    /// captured once: each earlier head's growth shifts the later heads'
    /// tails, and re-reading `keys.len()` inside the loop would walk the
    /// insert point past the buffer.
    fn append_rows(&mut self, heads: usize, dim: usize, rows: usize) {
        let rows = rows * dim;
        let len = self.keys.len();
        for head in 0..heads {
            let at = (head + 1) * len / heads + head * rows;
            self.keys.splice(at..at, std::iter::repeat_n(0.0, rows));
            self.values.splice(at..at, std::iter::repeat_n(0.0, rows));
        }
    }

    /// Writes `rows` rows starting at absolute row `at`, per head.
    fn write_rows(
        &mut self,
        keys: &[f32],
        values: &[f32],
        heads: usize,
        dim: usize,
        at: usize,
        rows: usize,
    ) {
        let buffer_rows = self.keys.len() / (heads * dim);
        let span = rows * dim;
        for head in 0..heads {
            let dst = head * buffer_rows * dim + at * dim;
            self.keys[dst..dst + span].copy_from_slice(&keys[head * span..(head + 1) * span]);
            self.values[dst..dst + span].copy_from_slice(&values[head * span..(head + 1) * span]);
        }
    }

    /// Clears every cached position.
    pub fn reset(&mut self) {
        self.offset = 0;
        self.idx = 0;
        self.keys.clear();
        self.values.clear();
    }
}

impl LayerKv for RotatingKvCache {
    fn offset(&self) -> usize {
        self.offset
    }

    fn update_and_fetch(
        &mut self,
        keys: &[f32],
        values: &[f32],
        n_kv_heads: usize,
        new_positions: usize,
        head_dim: usize,
    ) -> (&[f32], &[f32]) {
        RotatingKvCache::update_and_fetch(self, keys, values, n_kv_heads, new_positions, head_dim)
    }

    fn reset(&mut self) {
        RotatingKvCache::reset(self)
    }
}

/// One layer's prompt cache: the dense appending cache or the rotating
/// sliding-window cache, matching the reference `make_cache` mixes
/// (cohere2 alternates them by `sliding_window_pattern`).
#[derive(Debug, Clone)]
pub enum PromptCache {
    /// Full-attention layer (`KVCache` in the reference).
    Dense(KvCache),
    /// Sliding-window layer (`RotatingKVCache` in the reference).
    Rotating(RotatingKvCache),
}

impl Default for PromptCache {
    fn default() -> Self {
        Self::Dense(KvCache::new())
    }
}

impl PromptCache {
    /// Builds the layer cache for a sliding-window layer.
    pub fn rotating(max_size: usize, keep: usize) -> Self {
        Self::Rotating(RotatingKvCache::new(max_size, keep))
    }

    /// Builds the layer cache for a full-attention layer.
    pub fn dense() -> Self {
        Self::Dense(KvCache::new())
    }
}

impl LayerKv for PromptCache {
    fn offset(&self) -> usize {
        match self {
            PromptCache::Dense(cache) => cache.offset(),
            PromptCache::Rotating(cache) => cache.offset(),
        }
    }

    fn update_and_fetch(
        &mut self,
        keys: &[f32],
        values: &[f32],
        n_kv_heads: usize,
        new_positions: usize,
        head_dim: usize,
    ) -> (&[f32], &[f32]) {
        match self {
            PromptCache::Dense(cache) => {
                cache.update_and_fetch(keys, values, n_kv_heads, new_positions, head_dim)
            }
            PromptCache::Rotating(cache) => {
                cache.update_and_fetch(keys, values, n_kv_heads, new_positions, head_dim)
            }
        }
    }

    fn reset(&mut self) {
        match self {
            PromptCache::Dense(cache) => cache.reset(),
            PromptCache::Rotating(cache) => cache.reset(),
        }
    }
}

/// One cache per layer, in layer order.
pub type LayerCaches = Vec<KvCache>;

/// `make_prompt_cache` for the dense backbones: a fresh empty cache per
/// layer.
pub fn make_prompt_cache(num_layers: usize) -> LayerCaches {
    (0..num_layers).map(|_| KvCache::new()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression guard for the flat-extend bug: appending a
    /// `[heads, new, dim]` block to a `[heads, total, dim]` buffer must grow
    /// every head's row range, not land inside the last head's region. The
    /// decoder fixtures caught this end to end; this pins it locally.
    #[test]
    fn per_head_append_keeps_the_head_major_layout() {
        // Two heads, head_dim 2. Head h row t holds (100 * t + 10 * h, ...).
        let mut cache = KvCache::new();
        let prefill = vec![
            0.0, 1.0, // head 0 row 0
            0.2, 1.2, // head 0 row 1
            10.0, 11.0, // head 1 row 0
            10.2, 11.2, // head 1 row 1
        ];
        cache.update_and_fetch(&prefill, &prefill, 2, 2, 2);
        let decode = vec![0.4, 1.4, 10.4, 11.4];
        let (keys, _) = cache.update_and_fetch(&decode, &decode, 2, 1, 2);
        let keys = keys.to_vec();
        assert_eq!(cache.offset, 3);
        assert_eq!(
            keys,
            &[
                0.0, 1.0, 0.2, 1.2, 0.4, 1.4, // head 0 rows 0..3
                10.0, 11.0, 10.2, 11.2, 10.4, 11.4, // head 1 rows 0..3
            ]
        );
        // A second decode keeps the invariant: new rows still land at each
        // head's tail after the earlier heads grew.
        let decode2 = vec![0.6, 1.6, 10.6, 11.6];
        let (keys, values) = cache.update_and_fetch(&decode2, &decode2, 2, 1, 2);
        assert_eq!(keys.len(), 2 * 4 * 2);
        assert_eq!(
            keys,
            &[
                0.0, 1.0, 0.2, 1.2, 0.4, 1.4, 0.6, 1.6, //
                10.0, 11.0, 10.2, 11.2, 10.4, 11.4, 10.6, 11.6,
            ]
        );
        assert_eq!(&values[6..8], &[0.6, 1.6]);
        assert_eq!(&values[14..16], &[10.6, 11.6]);
        cache.reset();
        assert_eq!(cache.offset, 0);
        assert!(cache.keys.is_empty());
    }

    /// A window-4 ring fed a 7-row prefill then single decode steps must
    /// reproduce the reference trim-then-rotate buffer states: prefill
    /// stays untrimmed, the first decode trims to the window, and later
    /// decodes overwrite the ring in place.
    #[test]
    fn rotating_cache_matches_the_reference_ring_states() {
        let mut cache = RotatingKvCache::new(4, 0);
        // Prefill 7 positions, one row per head (2 heads, dim 1): row t of
        // head h holds the position tag (10 * h + t).
        let mut prefill = Vec::new();
        for head in 0..2 {
            for t in 0..7 {
                prefill.push((10 * head + t) as f32);
            }
        }
        let (keys, _) = cache.update_and_fetch(&prefill, &prefill, 2, 7, 1);
        // The first prefill is never trimmed (reference `_update_concat`
        // direct-assign path).
        let keys = keys.to_vec();
        assert_eq!(cache.offset, 7);
        assert_eq!(
            keys,
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
        );

        // Decode 7: trim to rows 3..6, write at the wrapped slot 0.
        let (keys, _) = cache.update_and_fetch(&[7.0, 17.0], &[7.0, 17.0], 2, 1, 1);
        let keys = keys.to_vec();
        assert_eq!(cache.offset, 8);
        assert_eq!(keys, &[7.0, 4.0, 5.0, 6.0, 17.0, 14.0, 15.0, 16.0]);

        // Decode 8 and 9: in-place overwrite at slots 1 and 2.
        let (keys, _) = cache.update_and_fetch(&[8.0, 18.0], &[8.0, 18.0], 2, 1, 1);
        assert_eq!(keys, &[7.0, 8.0, 5.0, 6.0, 17.0, 18.0, 15.0, 16.0]);
        let (keys, _) = cache.update_and_fetch(&[9.0, 19.0], &[9.0, 19.0], 2, 1, 1);
        assert_eq!(keys, &[7.0, 8.0, 9.0, 6.0, 17.0, 18.0, 19.0, 16.0]);
        assert_eq!(cache.idx, 3);
    }

    /// A short prefill followed by a decode must grow then wrap without
    /// losing rows: prefill 3 into a window of 4, decode once (buffer
    /// full), decode again (rotation overwrites the oldest row).
    #[test]
    fn rotating_cache_grows_before_wrapping() {
        let mut cache = RotatingKvCache::new(4, 0);
        let prefill = [0.0, 1.0, 2.0];
        let (keys, _) = cache.update_and_fetch(&prefill, &prefill, 1, 3, 1);
        assert_eq!(keys, &[0.0, 1.0, 2.0]);
        let (keys, _) = cache.update_and_fetch(&[3.0], &[3.0], 1, 1, 1);
        let keys = keys.to_vec();
        assert_eq!(cache.offset, 4);
        assert_eq!(keys, &[0.0, 1.0, 2.0, 3.0]);
        let (keys, _) = cache.update_and_fetch(&[4.0], &[4.0], 1, 1, 1);
        assert_eq!(keys, &[4.0, 1.0, 2.0, 3.0]);
    }

    /// A second multi-position update after a short prefill must trim the
    /// oldest rows and stay in temporal order (reference `_update_concat`
    /// trim path, the `max_size - 1 + S` invariant).
    #[test]
    fn rotating_streaming_prefill_trims_to_window_minus_one_plus_chunk() {
        let mut cache = RotatingKvCache::new(4, 0);
        let (keys, _) = cache.update_and_fetch(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0], 1, 3, 1);
        assert_eq!(keys, &[0.0, 1.0, 2.0]);
        let (keys, _) = cache.update_and_fetch(&[3.0, 4.0], &[3.0, 4.0], 1, 2, 1);
        // trim_size = 3 - 4 + 1 = 0, so the buffer just grows to 5 rows.
        assert_eq!(keys, &[0.0, 1.0, 2.0, 3.0, 4.0]);
        let (keys, _) = cache.update_and_fetch(&[5.0, 6.0], &[5.0, 6.0], 1, 2, 1);
        // trim_size = 5 - 4 + 1 = 2: keep rows 2..5, append -> last
        // window + chunk - 1 positions in order.
        assert_eq!(keys, &[2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(cache.offset, 7);
        assert_eq!(cache.idx, 5);

        // A decode after the streaming prefill trims to exactly the window
        // and rotates.
        let (keys, _) = cache.update_and_fetch(&[7.0], &[7.0], 1, 1, 1);
        assert_eq!(keys, &[7.0, 4.0, 5.0, 6.0]);
    }

    /// The `keep` rows survive every trim and the write slot wraps to them
    /// (the reference `_idx = keep` reset).
    #[test]
    fn rotating_cache_keeps_front_rows_through_trims() {
        let mut cache = RotatingKvCache::new(4, 1);
        let prefill = [0.0, 1.0, 2.0, 3.0, 4.0];
        let (keys, _) = cache.update_and_fetch(&prefill, &prefill, 1, 5, 1);
        // Direct assign: nothing trimmed on the first call.
        assert_eq!(keys, &[0.0, 1.0, 2.0, 3.0, 4.0]);
        let (keys, _) = cache.update_and_fetch(&[5.0], &[5.0], 1, 1, 1);
        // trim_size = 5 - 4 = 1: keep rows [0] and [2..5) (four rows),
        // write 5 at slot 1 (idx reset from max_size to keep).
        assert_eq!(keys, &[0.0, 5.0, 3.0, 4.0]);
        let (keys, _) = cache.update_and_fetch(&[6.0], &[6.0], 1, 1, 1);
        // No growth (buffer full), no trim, idx = 2.
        assert_eq!(keys, &[0.0, 5.0, 6.0, 4.0]);
    }

    /// The enum dispatches to the right cache.
    #[test]
    fn prompt_cache_dispatches() {
        let mut dense = PromptCache::dense();
        let (keys, _) = dense.update_and_fetch(&[1.0, 2.0], &[1.0, 2.0], 1, 2, 1);
        assert_eq!(keys, &[1.0, 2.0]);
        assert_eq!(LayerKv::offset(&dense), 2);

        let mut rotating = PromptCache::rotating(2, 0);
        let (keys, _) = rotating.update_and_fetch(&[1.0, 2.0], &[1.0, 2.0], 1, 2, 1);
        assert_eq!(keys, &[1.0, 2.0]);
        let (keys, _) = rotating.update_and_fetch(&[3.0], &[3.0], 1, 1, 1);
        assert_eq!(keys, &[3.0, 2.0]);
        assert_eq!(LayerKv::offset(&rotating), 3);

        dense.reset();
        assert_eq!(LayerKv::offset(&dense), 0);
        rotating.reset();
        assert_eq!(LayerKv::offset(&rotating), 0);
    }
}
