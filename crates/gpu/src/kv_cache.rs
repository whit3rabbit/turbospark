//! Per-layer FP16 K/V storage for the decode loop. Ported from
//! `Runtime/KVCache/KVCacheManager.swift`.
//!
//! One K buffer and one V buffer per layer, allocated once at construction
//! (the decode hot path never allocates). Linear-attention and compressed-
//! attention layers (Qwen 3.6 gated-DeltaNet, DeepSeek-V4 CSA/HCA) carry no
//! per-token K/V rows here — they share one page-sized placeholder buffer so
//! the parallel per-layer arrays stay non-optional, matching the Swift
//! original. Their real state lives in [`crate::gdn_state::GdnStateManager`]
//! (linear layers) or a future DSV4 state manager (compressed layers).

use metal::{Device, MTLResourceOptions};
use model_io::ArchConfig;

use crate::context::GpuError;

/// Which attention variant a layer runs, sourced from
/// `ArchConfig.full_attention_layer_mask` (0 = swa, 1 = full, 2 = linear,
/// 3/4 = DeepSeek V4 CSA/HCA, both `Compressed` here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Swa,
    Full,
    Linear,
    Compressed,
}

/// A read view the attention kernels bind.
pub struct KvView<'a> {
    pub buffer: &'a metal::Buffer,
    /// Byte offset of logical position 0. Always 0 under linear storage.
    pub offset: usize,
    /// Bytes per token (`num_kv_heads * head_dim * size_of::<f16>()`).
    pub stride: usize,
    /// Number of valid positions written so far.
    pub valid_token_count: usize,
    /// Ring start slot. 0 under linear storage.
    pub start_slot: usize,
}

const FP16_SIZE: usize = 2;

pub struct KvCacheManager {
    max_context: usize,
    fp16_ring_enabled: bool,
    num_layers: usize,
    k_buffers: Vec<metal::Buffer>,
    v_buffers: Vec<metal::Buffer>,
    strides: Vec<usize>,
    kinds: Vec<LayerKind>,
    capacity_tokens: Vec<usize>,
    position: usize,
}

#[allow(clippy::too_many_arguments)]
impl KvCacheManager {
    pub fn new(
        device: &Device,
        config: &ArchConfig,
        max_context: usize,
        fp16_ring_enabled: bool,
        sliding_window: Option<usize>,
        max_prefill_chunk_tokens: usize,
        fp16_ring_capacity_override: Option<usize>,
    ) -> Result<Self, GpuError> {
        assert!(max_context > 0, "max_context must be positive");
        assert!(
            max_prefill_chunk_tokens > 0,
            "max_prefill_chunk_tokens must be positive"
        );

        let swa_stride = config.num_kv_heads as usize * config.head_dim as usize * FP16_SIZE;
        let full_stride =
            config.num_full_kv_heads as usize * config.full_head_dim as usize * FP16_SIZE;
        let swa_capacity = max_context.min(
            fp16_ring_capacity_override
                .unwrap_or(
                    sliding_window.unwrap_or(config.sliding_window as usize)
                        + max_prefill_chunk_tokens,
                )
                .max(1),
        );

        let num_layers = config.num_layers as usize;
        let mut k_buffers = Vec::with_capacity(num_layers);
        let mut v_buffers = Vec::with_capacity(num_layers);
        let mut strides = Vec::with_capacity(num_layers);
        let mut kinds = Vec::with_capacity(num_layers);
        let mut capacity_tokens = Vec::with_capacity(num_layers);

        let all_placeholder = config.has_compressed_attention_layers();
        let mut linear_placeholder: Option<metal::Buffer> = None;

        for layer in 0..num_layers {
            let mask_value = config.full_attention_layer_mask[layer];
            if mask_value == 2 || all_placeholder {
                let placeholder = match &linear_placeholder {
                    Some(existing) => existing.clone(),
                    None => {
                        let made = device.new_buffer(
                            page_size_bytes() as u64,
                            MTLResourceOptions::StorageModeShared,
                        );
                        linear_placeholder = Some(made.clone());
                        made
                    }
                };
                k_buffers.push(placeholder.clone());
                v_buffers.push(placeholder);
                strides.push(0);
                kinds.push(if mask_value == 2 {
                    LayerKind::Linear
                } else {
                    LayerKind::Compressed
                });
                capacity_tokens.push(0);
                continue;
            }
            let is_full = mask_value != 0;
            let stride = if is_full { full_stride } else { swa_stride };
            let capacity = if fp16_ring_enabled && !is_full {
                swa_capacity
            } else {
                max_context
            };
            let length = (capacity * stride) as u64;

            k_buffers.push(device.new_buffer(length.max(1), MTLResourceOptions::StorageModeShared));
            v_buffers.push(device.new_buffer(length.max(1), MTLResourceOptions::StorageModeShared));
            strides.push(stride);
            kinds.push(if is_full {
                LayerKind::Full
            } else {
                LayerKind::Swa
            });
            capacity_tokens.push(capacity);
        }

        Ok(Self {
            max_context,
            fp16_ring_enabled,
            num_layers,
            k_buffers,
            v_buffers,
            strides,
            kinds,
            capacity_tokens,
            position: 0,
        })
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn layer_kind(&self, layer: usize) -> LayerKind {
        self.kinds[layer]
    }

    /// Bytes per token for `layer` (K and V share the same stride).
    pub fn stride(&self, layer: usize) -> usize {
        self.strides[layer]
    }

    /// Physical token capacity for `layer`.
    pub fn capacity(&self, layer: usize) -> usize {
        self.capacity_tokens[layer]
    }

    pub fn ring_capacity(&self, layer: usize) -> usize {
        if self.fp16_ring_enabled && self.kinds[layer] == LayerKind::Swa {
            self.capacity_tokens[layer]
        } else {
            0
        }
    }

    pub fn buffer_length(&self, layer: usize) -> usize {
        self.capacity_tokens[layer] * self.strides[layer]
    }

    /// Write target for this layer's K projection at `position`.
    pub fn k_slot(&self, layer: usize, position: usize) -> (&metal::Buffer, usize) {
        assert!(
            !matches!(self.kinds[layer], LayerKind::Linear | LayerKind::Compressed),
            "layer kind has no KV slots"
        );
        self.validate_range(position, 1);
        (
            &self.k_buffers[layer],
            self.physical_slot(layer, position) * self.strides[layer],
        )
    }

    /// Write target for this layer's V projection at `position`. Always
    /// distinct from `k_slot` — full layers do not alias K and V.
    pub fn v_slot(&self, layer: usize, position: usize) -> (&metal::Buffer, usize) {
        assert!(
            !matches!(self.kinds[layer], LayerKind::Linear | LayerKind::Compressed),
            "layer kind has no KV slots"
        );
        self.validate_range(position, 1);
        (
            &self.v_buffers[layer],
            self.physical_slot(layer, position) * self.strides[layer],
        )
    }

    pub fn key_view(&self, layer: usize) -> KvView<'_> {
        self.key_view_at(layer, self.position)
    }

    pub fn key_view_at(&self, layer: usize, valid_token_count: usize) -> KvView<'_> {
        self.validate_valid_token_count(valid_token_count);
        KvView {
            buffer: &self.k_buffers[layer],
            offset: 0,
            stride: self.strides[layer],
            valid_token_count,
            start_slot: self.ring_start_slot(layer, valid_token_count),
        }
    }

    pub fn value_view(&self, layer: usize) -> KvView<'_> {
        self.value_view_at(layer, self.position)
    }

    pub fn value_view_at(&self, layer: usize, valid_token_count: usize) -> KvView<'_> {
        self.validate_valid_token_count(valid_token_count);
        KvView {
            buffer: &self.v_buffers[layer],
            offset: 0,
            stride: self.strides[layer],
            valid_token_count,
            start_slot: self.ring_start_slot(layer, valid_token_count),
        }
    }

    /// Advance the position cursor once the current token's K/V are written
    /// across all layers.
    pub fn advance(&mut self) {
        self.advance_by(1);
    }

    pub fn advance_by(&mut self, count: usize) {
        assert!(
            self.position + count <= self.max_context,
            "advance would exceed max_context"
        );
        self.position += count;
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

    fn validate_range(&self, start: usize, count: usize) {
        assert!(
            start + count <= self.max_context,
            "range exceeds max_context"
        );
    }

    fn validate_valid_token_count(&self, count: usize) {
        assert!(
            count <= self.max_context,
            "valid_token_count exceeds max_context"
        );
    }

    fn physical_slot(&self, layer: usize, position: usize) -> usize {
        let capacity = self.capacity_tokens[layer];
        assert!(capacity > 0, "layer has no KV storage");
        position % capacity
    }

    fn ring_start_slot(&self, layer: usize, valid_token_count: usize) -> usize {
        if !self.fp16_ring_enabled || self.kinds[layer] != LayerKind::Swa {
            return 0;
        }
        let capacity = self.capacity_tokens[layer];
        if valid_token_count <= capacity {
            0
        } else {
            valid_token_count % capacity
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
        // `MADV_DONTNEED` only affects physical residency, never the
        // buffer's validity as an object.
        #[allow(unsafe_code)]
        unsafe {
            libc::posix_madvise(ptr as *mut std::ffi::c_void, len, libc::POSIX_MADV_DONTNEED);
        }
    }
}
