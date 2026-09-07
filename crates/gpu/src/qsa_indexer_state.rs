//! Persistent per-layer state for `qwen4_exp`'s QSA (query-sparse
//! attention) indexer (`docs/QWEN4_PHASE0.md` section 5): the RAW
//! (un-normed, un-roped) key history every QSA layer's indexer reads, and
//! the incrementally-pooled block cache built from it. Owned by
//! `RealQwen4State::qsa` and driven from `families/qwen4/attn.rs` since
//! 2026-09-05 (raw key written every token, blocks advanced as they complete).
//!
//! **NOT `Dsv4StateManager`.** That is DeepSeek-V4-Flash's CSA/HCA state, a
//! different mechanism (LoRA ranks, a two-rate compress split, no
//! `index_budget`) with its own unwired buffers; the two share nothing but
//! both being attention-adjacent per-layer state.

use metal::{Device, MTLResourceOptions};
use model_io::ArchConfig;

use crate::kv_cache_mem::{advise_dontneed, page_size_bytes, write_into};

const FP16_SIZE: usize = 2;

/// Per-layer QSA indexer state: the raw key history the indexer pools from,
/// and the pooled-block cache built from it incrementally.
///
/// **The raw key buffer shares its POSITION with the main `KvCacheManager`
/// deliberately rather than tracking its own** (`docs/QWEN4_PHASE0.md`
/// section 5: "it must share a position counter with the KV cache"). Every
/// write here therefore takes `position` as a caller-supplied argument, the
/// same shape [`crate::kv_cache::KvCacheManager::write_k`] uses, rather than
/// this struct owning an `advance()` of its own -- two independent cursors
/// advancing over the same token stream is exactly the "upstream
/// restore-misalignment bug" that section also names as the reason the
/// `QsaCacheManager` design keeps one cursor rather than two.
pub struct QsaIndexerCacheManager {
    /// `Some` only at QSA layers: this family's `mask == 1` (full
    /// attention) layers, AND ONLY when the architecture actually declares
    /// an indexer (`compressed_attention.index_budget > 0`) -- `mask == 1`
    /// alone is not enough, since every OTHER family's dense-attention
    /// layers use the identical mask value and have no indexer at all.
    raw_keys: Vec<Option<metal::Buffer>>,
    /// `Some` at the same layers as `raw_keys`: pooled, per-head-normed,
    /// RoPE'd block rows, `index_kv_heads == 1` so one row per block.
    /// Filled incrementally -- see [`Self::pooled_block_count`].
    pooled_blocks: Vec<Option<metal::Buffer>>,
    /// How many of a layer's COMPLETE blocks are already pooled, normed and
    /// roped, i.e. how far into `pooled_blocks` real data has been written.
    /// mlx-vlm's own "first_new_block" cursor
    /// (`Qwen4ExpQSAIndexer.select_from_projected`): a block, once pooled,
    /// never needs recomputing, so the wiring that eventually drives this
    /// only has to process `[pooled_block_count(layer), new_complete_count)`
    /// each time a new block completes rather than the whole history.
    pooled_block_count: Vec<usize>,
    index_head_dim: usize,
    index_kv_heads: usize,
    compress_ratio: usize,
    max_context: usize,
}

impl QsaIndexerCacheManager {
    /// `config` must declare an indexer (a positive
    /// `compressed_attention.index_budget`) and carry at least one
    /// `mask == 1` layer -- matches
    /// [`crate::dsv4_state::Dsv4StateManager::new`]'s own precondition
    /// assert for the sibling mechanism.
    pub fn new(device: &Device, config: &ArchConfig, max_context: usize) -> Self {
        let ca = &config.compressed_attention;
        assert!(
            ca.index_budget > 0,
            "QsaIndexerCacheManager requires an architecture with index_budget > 0 \
             (mask == 1 alone does not imply an indexer -- every dense-attention \
             family shares that mask value)"
        );
        assert!(max_context > 0, "max_context must be positive");

        let index_head_dim = ca.index_head_dim as usize;
        // AGENTS.md/CLAUDE.md S11: `pooled_blocks` is sized at exactly ONE
        // row per block (`pooled_stride` below has no `index_kv_heads`
        // factor, matching `qsa_indexer.rs:156`'s own raw-key addressing at
        // `head_dim * 2`), and `families/qwen4/state.rs` already refuses
        // any other value at open. `.max(1)` used to silently accept 0 (and
        // any value above 1) here instead of refusing by name, one
        // constructor away from that assumption.
        assert_eq!(
            ca.index_kv_heads, 1,
            "QsaIndexerCacheManager requires index_kv_heads == 1, got {}",
            ca.index_kv_heads
        );
        let index_kv_heads = ca.index_kv_heads as usize;
        let compress_ratio = ca.csa_compress_rate as usize;
        assert!(index_head_dim > 0, "index_head_dim must be positive");
        assert!(compress_ratio > 0, "csa_compress_rate must be positive");

        let raw_stride = index_kv_heads * index_head_dim * FP16_SIZE;
        let raw_len = (max_context * raw_stride) as u64;
        let max_complete_blocks = max_context / compress_ratio;
        let pooled_stride = index_head_dim * FP16_SIZE;
        let pooled_len = (max_complete_blocks * pooled_stride).max(1) as u64;

        let num_layers = config.num_layers as usize;
        let mut raw_keys = Vec::with_capacity(num_layers);
        let mut pooled_blocks = Vec::with_capacity(num_layers);
        let mut pooled_block_count = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            if !config.layer_is_full(layer) {
                raw_keys.push(None);
                pooled_blocks.push(None);
                pooled_block_count.push(0);
                continue;
            }
            raw_keys.push(Some(
                device.new_buffer(raw_len.max(1), MTLResourceOptions::StorageModeShared),
            ));
            pooled_blocks.push(Some(
                device.new_buffer(pooled_len, MTLResourceOptions::StorageModeShared),
            ));
            pooled_block_count.push(0);
        }

        Self {
            raw_keys,
            pooled_blocks,
            pooled_block_count,
            index_head_dim,
            index_kv_heads,
            compress_ratio,
            max_context,
        }
    }

    /// True when `layer` carries indexer state (this family's QSA layers).
    pub fn is_qsa_layer(&self, layer: usize) -> bool {
        self.raw_keys[layer].is_some()
    }

    /// Bytes per token in the raw key buffer (`index_kv_heads * index_head_dim * 2`).
    pub fn raw_stride(&self, layer: usize) -> usize {
        if self.is_qsa_layer(layer) {
            self.index_kv_heads * self.index_head_dim * FP16_SIZE
        } else {
            0
        }
    }

    /// Bytes per row in the pooled-block buffer (`index_head_dim * 2`).
    pub fn pooled_stride(&self) -> usize {
        self.index_head_dim * FP16_SIZE
    }

    /// The `compress_ratio` (`csa_compress_rate`) blocks are pooled over.
    pub fn compress_ratio(&self) -> usize {
        self.compress_ratio
    }

    fn raw_keys_buffer(&self, layer: usize) -> &metal::Buffer {
        self.raw_keys[layer]
            .as_ref()
            .expect("layer is not a QSA layer")
    }

    /// Write target for `layer`'s RAW (un-normed, un-roped) key at
    /// `position` -- the position `KvCacheManager` is ALSO writing this
    /// token's real K/V at, per this struct's own shared-cursor contract.
    pub fn raw_key_slot(&self, layer: usize, position: usize) -> (&metal::Buffer, usize) {
        assert!(position < self.max_context, "position exceeds max_context");
        (
            self.raw_keys_buffer(layer),
            position * self.raw_stride(layer),
        )
    }

    /// Host-writes one token's raw key row (`raw_stride(layer)` bytes).
    /// Mirrors [`crate::kv_cache::KvCacheManager::write_k`]'s plain-memcpy
    /// shape -- shared storage makes this O(1) per token, no upload.
    pub fn write_raw_key(&self, layer: usize, position: usize, bytes: &[u8]) {
        let (buffer, offset) = self.raw_key_slot(layer, position);
        write_into(buffer, offset, bytes);
    }

    /// The full raw key buffer for `layer`, for a pooling dispatch to read
    /// (e.g. [`crate::qsa_indexer::encode_qsa_pool_blocks_mean`]) over
    /// whichever prefix of it is currently valid.
    pub fn raw_keys_view(&self, layer: usize) -> &metal::Buffer {
        self.raw_keys_buffer(layer)
    }

    /// The pooled-block buffer for `layer`, for a scoring dispatch to read
    /// (e.g. [`crate::qsa_indexer::encode_qsa_score_blocks`]) and for a
    /// pooling dispatch to WRITE into starting at
    /// `pooled_block_count(layer) * pooled_stride()`.
    pub fn pooled_blocks_buffer(&self, layer: usize) -> &metal::Buffer {
        self.pooled_blocks[layer]
            .as_ref()
            .expect("layer is not a QSA layer")
    }

    /// How many of `layer`'s complete blocks are already pooled, normed and
    /// roped. A caller pools/norms/ropes exactly the NEW complete blocks
    /// (`[pooled_block_count(layer), new_complete_count)`) into
    /// `pooled_blocks_buffer(layer)` starting at this cursor's byte offset,
    /// then calls [`Self::advance_pooled_blocks`] to record it -- a
    /// pooled block is never recomputed once written.
    pub fn pooled_block_count(&self, layer: usize) -> usize {
        self.pooled_block_count[layer]
    }

    /// Byte offset into `pooled_blocks_buffer(layer)` where the NEXT
    /// not-yet-pooled block belongs.
    pub fn pooled_block_write_offset(&self, layer: usize) -> usize {
        self.pooled_block_count[layer] * self.pooled_stride()
    }

    /// Records that blocks up to (exclusive) `new_count` have been pooled,
    /// normed and roped into `pooled_blocks_buffer(layer)` by the caller.
    /// Refuses to move the cursor BACKWARD or skip ahead of what the raw
    /// key history could have produced, since either would silently claim
    /// blocks were computed that were not.
    pub fn advance_pooled_blocks(&mut self, layer: usize, new_count: usize) {
        assert!(self.is_qsa_layer(layer), "layer is not a QSA layer");
        assert!(
            new_count >= self.pooled_block_count[layer],
            "advance_pooled_blocks must not move the cursor backward"
        );
        let max_complete_blocks = self.max_context / self.compress_ratio;
        assert!(
            new_count <= max_complete_blocks,
            "advance_pooled_blocks exceeds this layer's block capacity"
        );
        self.pooled_block_count[layer] = new_count;
    }

    /// Drops all cached raw keys and pooled blocks and returns their pages
    /// to the OS via `MADV_DONTNEED`/zeroing, matching
    /// [`crate::kv_cache::KvCacheManager::reset`]'s and
    /// [`crate::gdn_state::GdnStateManager::reset`]'s own per-generation
    /// contract. Does NOT touch any position counter -- there is none here
    /// to reset, by this struct's own shared-cursor design; the caller
    /// resets `KvCacheManager`'s position as it always did.
    pub fn reset(&mut self) {
        self.pooled_block_count.fill(0);
        let page_size = page_size_bytes();
        let mut advised: Vec<*const std::ffi::c_void> = Vec::new();
        for buffer in self.raw_keys.iter().flatten() {
            advise_dontneed(buffer, page_size, &mut advised);
        }
        for buffer in self.pooled_blocks.iter().flatten() {
            advise_dontneed(buffer, page_size, &mut advised);
        }
    }
}
