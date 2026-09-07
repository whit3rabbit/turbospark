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
//!
//! **TurboQuant KV-cache quantization** (`--kv-bits`) is the one thing
//! that makes K and V strides genuinely DIFFERENT on some layers (K3/V4
//! packs to different word counts than a shared FP16 stride ever could),
//! which is why this manager tracks `k_strides`/`v_strides` as two arrays
//! rather than one shared `strides`. [`Self::new`] is the `KvQuant::Off`
//! wrapper around [`Self::new_with_kv_quant`], so every pre-existing
//! caller and every frozen digest goes through the identical FP16-only
//! path unchanged. See `docs/TRUBOQUANT.md`.

use metal::{Device, MTLResourceOptions};
use model_io::{ArchConfig, KvQuant};

use crate::context::GpuError;
use crate::kv_cache_mem::{page_size_bytes, write_into};
use crate::kv_quant_tables::KvQuantTables;

/// Which attention variant a layer runs, sourced from
/// `ArchConfig.full_attention_layer_mask` (0 = swa, 1 = full, 2 = linear,
/// 3/4 = DeepSeek V4 CSA/HCA, both `Compressed` here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Sliding-window attention layer using ring buffer KV storage.
    Swa,
    /// Full attention layer using full context length KV storage.
    Full,
    /// Linear-attention layer (e.g., Qwen gated-DeltaNet) carrying no per-token KV storage.
    Linear,
    /// Compressed attention layer carrying no per-token KV storage.
    Compressed,
}

/// KV cache manager orchestrating per-layer Metal buffers for token generation.
pub struct KvCacheManager {
    max_context: usize,
    fp16_ring_enabled: bool,
    num_layers: usize,
    k_buffers: Vec<metal::Buffer>,
    v_buffers: Vec<metal::Buffer>,
    k_strides: Vec<usize>,
    v_strides: Vec<usize>,
    kinds: Vec<LayerKind>,
    capacity_tokens: Vec<usize>,
    position: usize,
    /// The sliding window the SWA ring layers are read over, kept only so
    /// [`Self::max_safe_rewind`] can derive the ring's slack. Zero when the
    /// model has no SWA layers.
    swa_window: usize,
    /// `Some((k_bits, v_bits))` for a layer TurboQuant quantizes
    /// ([`model_io::layer_is_quantized`]), `None` otherwise -- including
    /// every layer when `kv_quant` was [`KvQuant::Off`].
    quant: Vec<Option<(u8, u8)>>,
    /// The codebooks, midpoints and sign vectors every quantized layer
    /// shares. `None` when no layer quantizes.
    tables: Option<KvQuantTables>,
}

#[allow(clippy::too_many_arguments)]
impl KvCacheManager {
    /// Allocates KV cache buffers for all model layers based on architecture and context limits.
    /// The [`KvQuant::Off`] wrapper around [`Self::new_with_kv_quant`].
    pub fn new(
        device: &Device,
        config: &ArchConfig,
        max_context: usize,
        fp16_ring_enabled: bool,
        sliding_window: Option<usize>,
        max_prefill_chunk_tokens: usize,
        fp16_ring_capacity_override: Option<usize>,
    ) -> Result<Self, GpuError> {
        Self::new_with_kv_quant(
            device,
            config,
            max_context,
            fp16_ring_enabled,
            sliding_window,
            max_prefill_chunk_tokens,
            fp16_ring_capacity_override,
            KvQuant::Off,
        )
    }

    /// [`Self::new`], honoring `kv_quant`. Every quantized layer's K and V
    /// buffers are sized from [`model_io::kv_layer_strides`] -- the same
    /// per-layer arithmetic `context_policy`'s `kv_bytes_for_context_with`
    /// uses to ESTIMATE this manager's footprint before it ever opens, so
    /// the two cannot silently disagree about what a session will commit.
    ///
    /// Panics if `kv_quant` is on and `config.full_head_dim` fails
    /// [`model_io::rht_supported`] -- callers must refuse `--kv-bits` by
    /// name before reaching here (see `real_forward_init`'s
    /// `kv_quant_unsupported_reason`), the same contract
    /// [`KvQuantTables::new`] states.
    pub fn new_with_kv_quant(
        device: &Device,
        config: &ArchConfig,
        max_context: usize,
        fp16_ring_enabled: bool,
        sliding_window: Option<usize>,
        max_prefill_chunk_tokens: usize,
        fp16_ring_capacity_override: Option<usize>,
        kv_quant: KvQuant,
    ) -> Result<Self, GpuError> {
        assert!(max_context > 0, "max_context must be positive");
        assert!(
            max_prefill_chunk_tokens > 0,
            "max_prefill_chunk_tokens must be positive"
        );

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
        let mut k_strides = Vec::with_capacity(num_layers);
        let mut v_strides = Vec::with_capacity(num_layers);
        let mut kinds = Vec::with_capacity(num_layers);
        let mut capacity_tokens = Vec::with_capacity(num_layers);
        let mut quant = Vec::with_capacity(num_layers);

        let mut linear_placeholder: Option<metal::Buffer> = None;

        let tables = KvQuantTables::new(device, config.full_head_dim as usize, kv_quant);

        for layer in 0..num_layers {
            let mask_value = config.full_attention_layer_mask[layer];
            // AGENTS.md/CLAUDE.md S1: keyed on THIS layer's own mask value
            // alone. A model with one mask-3/4 (compressed) layer used to
            // give EVERY layer a placeholder here, including mask-0/1 full-
            // attention ones -- which then read `LayerKind::Compressed`
            // (the `else` arm below, since `mask_value == 2` is false for
            // them) and panic on the first `k_slot`/`v_slot` call, because
            // that layer has real per-token KV rows and no placeholder can
            // stand in for them.
            if mask_value == 2 || mask_value == 3 || mask_value == 4 {
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
                k_strides.push(0);
                v_strides.push(0);
                kinds.push(if mask_value == 2 {
                    LayerKind::Linear
                } else {
                    LayerKind::Compressed
                });
                capacity_tokens.push(0);
                quant.push(None);
                continue;
            }
            let is_full = mask_value != 0;
            let (k_stride, v_stride) = model_io::kv_layer_strides(config, layer, kv_quant);
            let layer_quant =
                if model_io::layer_is_quantized(kv_quant, mask_value, layer, num_layers) {
                    match kv_quant {
                        KvQuant::TurboQuant { k_bits, v_bits } => Some((k_bits, v_bits)),
                        KvQuant::Off => {
                            unreachable!("layer_is_quantized is false whenever kv_quant is Off")
                        }
                    }
                } else {
                    None
                };
            let capacity = if fp16_ring_enabled && !is_full {
                swa_capacity
            } else {
                max_context
            };
            let k_length = capacity as u64 * k_stride;
            let v_length = capacity as u64 * v_stride;

            k_buffers
                .push(device.new_buffer(k_length.max(1), MTLResourceOptions::StorageModeShared));
            v_buffers
                .push(device.new_buffer(v_length.max(1), MTLResourceOptions::StorageModeShared));
            k_strides.push(k_stride as usize);
            v_strides.push(v_stride as usize);
            kinds.push(if is_full {
                LayerKind::Full
            } else {
                LayerKind::Swa
            });
            capacity_tokens.push(capacity);
            quant.push(layer_quant);
        }

        Ok(Self {
            max_context,
            fp16_ring_enabled,
            num_layers,
            k_buffers,
            v_buffers,
            k_strides,
            v_strides,
            kinds,
            capacity_tokens,
            position: 0,
            swa_window: sliding_window.unwrap_or(config.sliding_window as usize),
            quant,
            tables,
        })
    }

    /// Returns current position cursor index.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Returns attention layer kind for `layer`.
    pub fn layer_kind(&self, layer: usize) -> LayerKind {
        self.kinds[layer]
    }

    /// Bytes per token for `layer`'s K side. Equal to [`Self::v_stride`] on
    /// every FP16 layer; the two can differ on a TurboQuant-quantized one
    /// (K and V bit widths need not match, e.g. K3/V4).
    pub fn k_stride(&self, layer: usize) -> usize {
        self.k_strides[layer]
    }

    /// Bytes per token for `layer`'s V side.
    pub fn v_stride(&self, layer: usize) -> usize {
        self.v_strides[layer]
    }

    /// `Some((k_bits, v_bits))` when `layer` is TurboQuant-quantized,
    /// `None` otherwise (every layer, when `kv_quant` was
    /// [`KvQuant::Off`]).
    pub fn layer_quant(&self, layer: usize) -> Option<(u8, u8)> {
        self.quant[layer]
    }

    /// The shared codebook/sign tables, when any layer quantizes.
    pub fn quant_tables(&self) -> Option<&KvQuantTables> {
        self.tables.as_ref()
    }

    /// Physical token capacity for `layer`.
    pub fn capacity(&self, layer: usize) -> usize {
        self.capacity_tokens[layer]
    }

    /// Returns ring buffer token capacity for SWA layer, or 0 for non-ring layers.
    pub fn ring_capacity(&self, layer: usize) -> usize {
        if self.fp16_ring_enabled && self.kinds[layer] == LayerKind::Swa {
            self.capacity_tokens[layer]
        } else {
            0
        }
    }

    /// Returns total K buffer byte length at `layer`.
    pub fn buffer_length(&self, layer: usize) -> usize {
        self.capacity_tokens[layer] * self.k_strides[layer]
    }

    /// Returns total V buffer byte length at `layer`.
    pub fn v_buffer_length(&self, layer: usize) -> usize {
        self.capacity_tokens[layer] * self.v_strides[layer]
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
            self.physical_slot(layer, position) * self.k_strides[layer],
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
            self.physical_slot(layer, position) * self.v_strides[layer],
        )
    }

    /// Host-writes one token's K row (`k_stride(layer)` bytes) into the
    /// layer's persistent K buffer at `position`'s physical slot. Shared
    /// storage mode makes this a plain memcpy into unified memory — O(1)
    /// per token, replacing any full-history re-upload. (The deeper form,
    /// a kernel writing its projection output straight into the slot,
    /// comes with encoder-level dispatch batching.)
    ///
    /// Refuses a TurboQuant-quantized layer by name: its row is never an
    /// in-place raw-byte target, only `encode_kv_quantize_tq` writes it
    /// (through `crate::kv_quantize::encode_kv_quantize_tq`), after the
    /// projection has been normed and RoPE'd on a staging row. A caller
    /// reaching this method for such a layer skipped that commit path.
    pub fn write_k(&self, layer: usize, position: usize, bytes: &[u8]) {
        assert!(
            self.quant[layer].is_none(),
            "write_k: layer {layer} is TurboQuant-quantized; write through \
             kv_quantize::encode_kv_quantize_tq instead"
        );
        let (buffer, offset) = self.k_slot(layer, position);
        write_into(buffer, offset, bytes);
    }

    /// Host-writes one token's V row; see [`KvCacheManager::write_k`].
    pub fn write_v(&self, layer: usize, position: usize, bytes: &[u8]) {
        assert!(
            self.quant[layer].is_none(),
            "write_v: layer {layer} is TurboQuant-quantized; write through \
             kv_quantize::encode_kv_quantize_tq instead"
        );
        let (buffer, offset) = self.v_slot(layer, position);
        write_into(buffer, offset, bytes);
    }

    /// Advance the position cursor once the current token's K/V are written
    /// across all layers.
    pub fn advance(&mut self) {
        self.advance_by(1);
    }

    /// Advances sequence position cursor by `count` tokens.
    pub fn advance_by(&mut self, count: usize) {
        assert!(
            self.position + count <= self.max_context,
            "advance would exceed max_context"
        );
        self.position += count;
    }

    fn validate_range(&self, start: usize, count: usize) {
        assert!(
            start + count <= self.max_context,
            "range exceeds max_context"
        );
    }

    fn physical_slot(&self, layer: usize, position: usize) -> usize {
        let capacity = self.capacity_tokens[layer];
        assert!(capacity > 0, "layer has no KV storage");
        position % capacity
    }
}

#[path = "kv_cache_rewind.rs"]
mod kv_cache_rewind;
