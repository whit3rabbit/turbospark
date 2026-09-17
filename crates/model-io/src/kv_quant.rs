//! TurboQuant KV-cache quantization policy: the `--kv-bits` value type,
//! which layers are eligible, and how many bytes a quantized row costs.
//!
//! Deliberately portable, following [`crate::context_policy`]: no `gpu`
//! dependency and no OS probe. The codec's actual numerics (the rotation,
//! the Lloyd-Max codebook, the packing) live in
//! `turbospark_compute::kv_quant`, ported from mlx-vlm's
//! `_TurboQuantMSECodec` -- see `docs/TRUBOQUANT.md`. This module only
//! answers the questions [`crate::context_policy`] and `crates/gpu`'s
//! `KvCacheManager` need before any row is ever quantized: is this layer
//! eligible, and how many bytes does its row cost.
//!
//! **Layer policy is mlx-vlm's own `should_quantize_kv_layer`**: quantize
//! every layer when the stack has two layers or fewer, otherwise quantize
//! every FULL-attention layer except the last one (measured sensitive to
//! quantization on Gemma-class models). Sliding-window and linear
//! (gated-DeltaNet) layers are never quantized regardless of this rule --
//! [`layer_is_quantized`] takes the mask value as a precondition, not an
//! afterthought.

use crate::ArchConfig;

/// `--kv-bits` selection. `Off` (the default) leaves every layer FP16, so
/// no memory-oracle or quality-gate row moves unless a caller opts in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvQuant {
    /// FP16 everywhere, the port's behavior before this feature existed.
    #[default]
    Off,
    /// TurboQuant at `k_bits`/`v_bits`. Only reached through [`KvQuant::parse`]
    /// or a literal matching one of the four widths that function accepts;
    /// nothing downstream validates an arbitrary pair.
    TurboQuant {
        /// Bits per key-row coordinate.
        k_bits: u8,
        /// Bits per value-row coordinate.
        v_bits: u8,
    },
}

impl KvQuant {
    /// Parses `"off"`, `"2"`, `"3"`, `"3.5"`, or `"4"`. `"3.5"` splits into
    /// K3/V4 -- floor for keys, ceil for values -- mlx-vlm's own split for
    /// its one fractional width (keys tolerate coarser quantization better
    /// than values in their own measurements; this port does not re-derive
    /// that, only carries the convention).
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "off" => Ok(Self::Off),
            "2" => Ok(Self::TurboQuant {
                k_bits: 2,
                v_bits: 2,
            }),
            "3" => Ok(Self::TurboQuant {
                k_bits: 3,
                v_bits: 3,
            }),
            "3.5" => Ok(Self::TurboQuant {
                k_bits: 3,
                v_bits: 4,
            }),
            "4" => Ok(Self::TurboQuant {
                k_bits: 4,
                v_bits: 4,
            }),
            other => Err(format!(
                "--kv-bits must be one of off|2|3|3.5|4, got {other:?}"
            )),
        }
    }

    /// True when this is a TurboQuant selection rather than [`KvQuant::Off`].
    pub fn is_on(self) -> bool {
        !matches!(self, Self::Off)
    }

    /// A short label for a startup line: `"off"`, `"2"`, `"3"`, `"4"`, or
    /// `"3.5 (K3/V4)"` when the two widths differ.
    pub fn label(self) -> String {
        match self {
            Self::Off => "off".to_string(),
            Self::TurboQuant { k_bits, v_bits } if k_bits == v_bits => k_bits.to_string(),
            Self::TurboQuant { k_bits, v_bits } => {
                let avg = (k_bits as f32 + v_bits as f32) / 2.0;
                format!("{avg} (K{k_bits}/V{v_bits})")
            }
        }
    }
}

/// Whether layer `layer` of a `num_layers`-deep stack should quantize its
/// KV cache, GIVEN that `mask_value` is the layer's own
/// `full_attention_layer_mask` entry.
///
/// Mirrors mlx-vlm's `should_quantize_kv_layer` exactly: for stacks of two
/// layers or fewer, every layer quantizes; otherwise every layer but the
/// last does. This function does not itself check whether the layer is a
/// full-attention one -- callers gate on `mask_value == 1` first (as
/// [`kv_layer_strides`] does), because a sliding-window or linear layer is
/// never a candidate regardless of its position in the stack.
pub fn layer_is_quantized(quant: KvQuant, mask_value: u8, layer: usize, num_layers: usize) -> bool {
    if !quant.is_on() || mask_value != 1 {
        return false;
    }
    num_layers <= 2 || layer + 1 < num_layers
}

/// A full head_dim TurboQuant can rotate: a power of two in `32..=512`,
/// every full head_dim this port's families actually use. mlx-vlm's own
/// dense-rotation-matrix fallback for other widths is not ported (see
/// `docs/TRUBOQUANT.md`), so an install failing this check refuses
/// `--kv-bits` by name at open rather than silently falling back to a
/// slower path nothing here implements.
pub fn rht_supported(head_dim: i64) -> bool {
    if !(32..=512).contains(&head_dim) {
        return false;
    }
    let d = head_dim as u64;
    d & (d - 1) == 0
}

/// Packed `u32` word count for one row of `head_dim` `bits`-wide codebook
/// indices, LSB-first: `ceil(head_dim * bits / 32)`.
///
/// Duplicated from `turbospark_compute::kv_quant::packed_words` rather than
/// imported -- this crate does not depend on `compute`, and the formula is
/// one line; the two are pinned against each other by
/// `crates/gpu/tests/kv_cache_quant.rs`.
pub fn tq_packed_words(head_dim: i64, bits: u8) -> u64 {
    if head_dim <= 0 || bits == 0 {
        return 0;
    }
    (head_dim as u64 * bits as u64).div_ceil(32)
}

/// FP16 size in bytes, duplicated per `context_policy.rs`'s own precedent
/// (`FP32_SIZE` beside `FP16_SIZE`) rather than exposing that module's
/// private constant.
const FP16_BYTES: u64 = 2;

/// Bytes one quantized K or V row costs, across every KV head: one `f32`
/// norm plus the packed words, per head.
pub fn tq_row_bytes(kv_heads: i64, head_dim: i64, bits: u8) -> u64 {
    let words = tq_packed_words(head_dim, bits);
    kv_heads.max(0) as u64 * (4 + 4 * words)
}

/// Per-token K and V byte cost for one layer, honoring `quant` when that
/// layer is eligible ([`layer_is_quantized`]). FP16 on every ineligible
/// layer (sliding-window, linear, or the one full layer the last-layer rule
/// excludes), and on every layer when `quant` is [`KvQuant::Off`].
///
/// K and V strides can DIFFER under TurboQuant (12 vs 16 packed words at
/// K3/V4, `head_dim` 128), which is why this returns a pair rather than the
/// single stride [`crate::context_policy::kv_bytes_for_context`]'s FP16-only
/// arithmetic could get away with.
pub fn kv_layer_strides(arch: &ArchConfig, layer: usize, quant: KvQuant) -> (u64, u64) {
    let mask = arch
        .full_attention_layer_mask
        .get(layer)
        .copied()
        .unwrap_or(1);
    if mask == 2 {
        return (0, 0);
    }
    // Mask 5 (MLA, `deepseek2`) reads its geometry from the `mla` block,
    // not from the head-dim pair: the cache row is the compressed
    // `[latent ; rope key]` ONE row per token (`num_kv_heads` is already 1
    // on this architecture, but the DERIVATION belongs here rather than in
    // a caller's assumption). K carries the whole row; V is read as the
    // row's first `kv_lora` halves, so its stride keeps the row width and
    // the buffer is never written by the MLA flow.
    if mask == 5 {
        let row = arch.mla.cache_row_dim().max(0) as u64 * FP16_BYTES;
        let v = arch.mla.kv_lora_rank.max(0) as u64 * FP16_BYTES;
        if !layer_is_quantized(quant, mask, layer, arch.num_layers as usize) {
            return (row, v);
        }
        // TurboQuant against the compressed latent cache is not a thing:
        // the quantization gates refuse mask 5 with --kv-bits, and falling
        // through would silently size the buffers as if it had applied.
        return (row, v);
    }
    let (kv_heads, head_dim) = if mask == 0 {
        (arch.num_kv_heads, arch.head_dim)
    } else {
        (arch.num_full_kv_heads, arch.full_head_dim)
    };

    if !layer_is_quantized(quant, mask, layer, arch.num_layers as usize) {
        let fp16 = kv_heads.max(0) as u64 * head_dim.max(0) as u64 * FP16_BYTES;
        return (fp16, fp16);
    }

    match quant {
        KvQuant::TurboQuant { k_bits, v_bits } => (
            tq_row_bytes(kv_heads, head_dim, k_bits),
            tq_row_bytes(kv_heads, head_dim, v_bits),
        ),
        KvQuant::Off => unreachable!("layer_is_quantized is false whenever quant is Off"),
    }
}
