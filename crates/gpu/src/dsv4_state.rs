//! Per-layer attention state for DeepSeek-V4's compressed-attention (CSA)
//! and heavily-compressed-attention (HCA) layers. Ported from
//! `Runtime/KVCache/DSV4StateManager.swift`.
//!
//! Every layer keeps a sliding-window K=V ring (`sliding_window` rows of
//! `full_head_dim` FP16; K and V are the same storage by construction).
//! CSA/HCA layers additionally keep the compressed-entry cache, pending
//! source rows awaiting a full compression window, and (CSA only) the
//! prior window's `Ca` slices plus the indexer's parallel set. All buffers
//! are allocated once at construction; `reset()` discards the large ring/
//! compressed/indexer buffers (`POSIX_MADV_DONTNEED`, like
//! `KvCacheManager::reset`) and zeroes the small pending/prior buffers so a
//! fresh sequence never reads a stale row.

use metal::{Device, MTLResourceOptions};
use model_io::ArchConfig;

const FP16_SIZE: usize = 2;

/// Per-layer bookkeeping the decode loop updates as compressor/indexer
/// windows commit. Pure counters, not GPU state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LayerCounters {
    pub tokens: usize,
    pub compressed_entries: usize,
    pub pending_rows: usize,
    pub indexer_entries: usize,
    pub indexer_pending_rows: usize,
    pub has_prior: bool,
    pub indexer_has_prior: bool,
}

pub struct Dsv4StateManager {
    window_kv: Vec<metal::Buffer>,
    compressed_kv: Vec<Option<metal::Buffer>>,
    pending_kv: Vec<Option<metal::Buffer>>,
    pending_gate: Vec<Option<metal::Buffer>>,
    prior_ca_kv: Vec<Option<metal::Buffer>>,
    prior_ca_gate: Vec<Option<metal::Buffer>>,
    indexer_keys: Vec<Option<metal::Buffer>>,
    indexer_pending_kv: Vec<Option<metal::Buffer>>,
    indexer_pending_gate: Vec<Option<metal::Buffer>>,
    indexer_prior_ca_kv: Vec<Option<metal::Buffer>>,
    indexer_prior_ca_gate: Vec<Option<metal::Buffer>>,
    pub counters: Vec<LayerCounters>,
    ring_capacity: usize,
}

impl Dsv4StateManager {
    /// `config` must have at least one CSA/HCA layer (matches the Swift
    /// original's `precondition(config.hasCompressedAttentionLayers)`).
    pub fn new(device: &Device, config: &ArchConfig, max_context: usize) -> Self {
        assert!(
            config.has_compressed_attention_layers(),
            "Dsv4StateManager requires at least one CSA/HCA layer"
        );
        let ca = &config.compressed_attention;
        let head_dim = config.full_head_dim as usize;
        let ring_capacity = config.sliding_window as usize;
        let num_layers = config.num_layers as usize;

        let make = |elements: usize| -> metal::Buffer {
            device.new_buffer(
                (elements * FP16_SIZE).max(16) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };

        let mut window_kv = Vec::with_capacity(num_layers);
        let mut compressed_kv = Vec::with_capacity(num_layers);
        let mut pending_kv = Vec::with_capacity(num_layers);
        let mut pending_gate = Vec::with_capacity(num_layers);
        let mut prior_ca_kv = Vec::with_capacity(num_layers);
        let mut prior_ca_gate = Vec::with_capacity(num_layers);
        let mut indexer_keys = Vec::with_capacity(num_layers);
        let mut indexer_pending_kv = Vec::with_capacity(num_layers);
        let mut indexer_pending_gate = Vec::with_capacity(num_layers);
        let mut indexer_prior_ca_kv = Vec::with_capacity(num_layers);
        let mut indexer_prior_ca_gate = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            window_kv.push(make(ring_capacity * head_dim));
            let is_csa = config.layer_is_csa(layer);
            let is_hca = config.layer_is_hca(layer);
            if !(is_csa || is_hca) {
                compressed_kv.push(None);
                pending_kv.push(None);
                pending_gate.push(None);
                prior_ca_kv.push(None);
                prior_ca_gate.push(None);
                indexer_keys.push(None);
                indexer_pending_kv.push(None);
                indexer_pending_gate.push(None);
                indexer_prior_ca_kv.push(None);
                indexer_prior_ca_gate.push(None);
                continue;
            }

            let rate = (if is_csa {
                ca.csa_compress_rate
            } else {
                ca.hca_compress_rate
            }) as usize;
            let row_width = if is_csa { 2 * head_dim } else { head_dim };
            let max_entries = max_context.div_ceil(rate);
            compressed_kv.push(Some(make(max_entries * head_dim)));
            pending_kv.push(Some(make(rate * row_width)));
            pending_gate.push(Some(make(rate * row_width)));

            if is_csa {
                prior_ca_kv.push(Some(make(rate * head_dim)));
                prior_ca_gate.push(Some(make(rate * head_dim)));
                let idx_rate = ca.csa_compress_rate as usize;
                let idx_head_dim = ca.index_head_dim as usize;
                let idx_entries = max_context.div_ceil(idx_rate);
                indexer_keys.push(Some(make(idx_entries * idx_head_dim)));
                indexer_pending_kv.push(Some(make(idx_rate * 2 * idx_head_dim)));
                indexer_pending_gate.push(Some(make(idx_rate * 2 * idx_head_dim)));
                indexer_prior_ca_kv.push(Some(make(idx_rate * idx_head_dim)));
                indexer_prior_ca_gate.push(Some(make(idx_rate * idx_head_dim)));
            } else {
                prior_ca_kv.push(None);
                prior_ca_gate.push(None);
                indexer_keys.push(None);
                indexer_pending_kv.push(None);
                indexer_pending_gate.push(None);
                indexer_prior_ca_kv.push(None);
                indexer_prior_ca_gate.push(None);
            }
        }

        let mut manager = Self {
            window_kv,
            compressed_kv,
            pending_kv,
            pending_gate,
            prior_ca_kv,
            prior_ca_gate,
            indexer_keys,
            indexer_pending_kv,
            indexer_pending_gate,
            indexer_prior_ca_kv,
            indexer_prior_ca_gate,
            counters: vec![LayerCounters::default(); num_layers],
            ring_capacity,
        };
        manager.reset();
        manager
    }

    pub fn is_csa_or_hca(&self, layer: usize) -> bool {
        self.compressed_kv[layer].is_some()
    }

    pub fn window_buffer(&self, layer: usize) -> &metal::Buffer {
        &self.window_kv[layer]
    }

    pub fn compressed_buffer(&self, layer: usize) -> &metal::Buffer {
        self.compressed_kv[layer]
            .as_ref()
            .expect("layer is not CSA/HCA")
    }

    pub fn indexer_keys_buffer(&self, layer: usize) -> &metal::Buffer {
        self.indexer_keys[layer].as_ref().expect("layer is not CSA")
    }

    /// Ring slot receiving the row for absolute `position`.
    pub fn window_slot(&self, position: usize) -> usize {
        position % self.ring_capacity
    }

    /// Rows currently valid in the window ring after `position` was written.
    pub fn window_count(&self, position: usize) -> usize {
        (position + 1).min(self.ring_capacity)
    }

    /// Absolute position of the oldest valid window row.
    pub fn window_start_position(&self, position: usize) -> usize {
        (position + 1).saturating_sub(self.ring_capacity)
    }

    pub fn reset(&mut self) {
        self.counters.fill(LayerCounters::default());
        let page_size = page_size_bytes();
        let mut advised: Vec<*const std::ffi::c_void> = Vec::new();
        for buffer in &self.window_kv {
            advise_dontneed(buffer, page_size, &mut advised);
        }
        for buffer in self.compressed_kv.iter().flatten() {
            advise_dontneed(buffer, page_size, &mut advised);
        }
        for buffer in self.indexer_keys.iter().flatten() {
            advise_dontneed(buffer, page_size, &mut advised);
        }
        for buffer in self.pending_kv.iter().flatten() {
            zero_buffer(buffer);
        }
        for buffer in self.pending_gate.iter().flatten() {
            zero_buffer(buffer);
        }
        for buffer in self.prior_ca_kv.iter().flatten() {
            zero_buffer(buffer);
        }
        for buffer in self.prior_ca_gate.iter().flatten() {
            zero_buffer(buffer);
        }
        for buffer in self.indexer_pending_kv.iter().flatten() {
            zero_buffer(buffer);
        }
        for buffer in self.indexer_pending_gate.iter().flatten() {
            zero_buffer(buffer);
        }
        for buffer in self.indexer_prior_ca_kv.iter().flatten() {
            zero_buffer(buffer);
        }
        for buffer in self.indexer_prior_ca_gate.iter().flatten() {
            zero_buffer(buffer);
        }
    }
}

fn page_size_bytes() -> usize {
    4096
}

fn advise_dontneed(
    buffer: &metal::Buffer,
    page_size: usize,
    seen: &mut Vec<*const std::ffi::c_void>,
) {
    let ptr = buffer.contents() as *const std::ffi::c_void;
    if seen.contains(&ptr) {
        return;
    }
    seen.push(ptr);
    let len = (buffer.length() as usize / page_size) * page_size;
    if len > 0 {
        // SAFETY: `ptr` is the base address of a live `MTLBuffer` allocated
        // with shared storage mode (CPU-and-GPU-visible, page-aligned by
        // Metal's allocator), and `len` is rounded down to a whole number
        // of pages so this never advises past the buffer's own allocation.
        #[allow(unsafe_code)]
        unsafe {
            libc::posix_madvise(ptr as *mut std::ffi::c_void, len, libc::POSIX_MADV_DONTNEED);
        }
    }
}

fn zero_buffer(buffer: &metal::Buffer) {
    let len = buffer.length() as usize;
    if len == 0 {
        return;
    }
    // SAFETY: `buffer` is a live, CPU-visible (`storageModeShared`) `MTLBuffer`
    // allocated by this module with exactly `len` bytes; writing zeros over
    // its full extent cannot go out of bounds.
    #[allow(unsafe_code)]
    unsafe {
        std::ptr::write_bytes(buffer.contents() as *mut u8, 0, len);
    }
}
