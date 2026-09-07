//! GPU-resident tables for a TurboQuant-quantized layer: the sign vector,
//! the Lloyd-Max codebook, and its midpoints, for both the K and V sides.
//! Built ONCE at open from `turbospark_compute::kv_quant` (the codec's
//! numerical ground truth) and reused by every `kv_quantize_tq` and
//! `attention_*_tq` dispatch for the life of the session -- the codebook
//! fit is a 32768-point, 100-iteration Lloyd-Max loop and is not something
//! to rebuild per token.
//!
//! Sized for the FULL head_dim only: TurboQuant never quantizes a
//! sliding-window layer ([`model_io::layer_is_quantized`] requires
//! `mask_value == 1`), so a model with both `head_dim` and `full_head_dim`
//! needs no second table.

use metal::Device;
use model_io::KvQuant;
use turbospark_compute::kv_quant::{codebook, midpoints, sign_vector, KEY_SEED, VALUE_SEED};

/// One side's (K or V) codebook, midpoints and sign vector, each as a
/// Metal buffer of `f32`.
pub struct TqSideTables {
    pub signs: metal::Buffer,
    pub codebook: metal::Buffer,
    pub midpoints: metal::Buffer,
    pub bits: u8,
    pub levels: u32,
    /// The `head_dim` these tables were built for (AGENTS.md/CLAUDE.md B3):
    /// `kv_quantize.rs::encode_kv_quantize_tq` takes a `TqSideTables` alone,
    /// not the parent `KvQuantTables`, so it needs its own copy of the
    /// dimension to assert against rather than trusting the caller's
    /// `head_dim` argument agrees -- the same check
    /// `encode_attention_decode_tq` already makes against
    /// `KvQuantTables::full_head_dim` on its side.
    pub dim: usize,
}

impl TqSideTables {
    fn build(device: &Device, dim: usize, bits: u8, seed: u64) -> Self {
        let signs_data = sign_vector(dim, seed);
        let codebook_data = codebook(dim, bits);
        let midpoints_data = midpoints(&codebook_data);
        Self {
            signs: device.new_buffer_with_data(
                signs_data.as_ptr().cast(),
                (signs_data.len() * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ),
            codebook: device.new_buffer_with_data(
                codebook_data.as_ptr().cast(),
                (codebook_data.len() * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ),
            midpoints: device.new_buffer_with_data(
                midpoints_data.as_ptr().cast(),
                (midpoints_data.len() * 4).max(4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ),
            bits,
            levels: 1u32 << bits,
            dim,
        }
    }
}

/// Both sides' tables for one quantized layer stack (one K side, one V
/// side; every quantized layer in a model shares ONE pair, since the codec
/// is keyed on `(head_dim, bits, seed)` and every quantized layer in a
/// single-head_dim model shares all three).
pub struct KvQuantTables {
    pub k: TqSideTables,
    pub v: TqSideTables,
    pub full_head_dim: usize,
}

impl KvQuantTables {
    /// Builds both sides' tables, or returns `None` for [`KvQuant::Off`].
    /// Panics if `quant` is on and `full_head_dim` fails
    /// [`model_io::rht_supported`] -- callers must check that (and refuse
    /// `--kv-bits` by name) before reaching here; see
    /// `real_forward_init::kv_quant_unsupported_reason`.
    pub fn new(device: &Device, full_head_dim: usize, quant: KvQuant) -> Option<Self> {
        match quant {
            KvQuant::Off => None,
            KvQuant::TurboQuant { k_bits, v_bits } => {
                assert!(
                    model_io::rht_supported(full_head_dim as i64),
                    "KvQuantTables::new: head_dim {full_head_dim} does not support the RHT; \
                     the caller must refuse --kv-bits before opening the cache"
                );
                Some(Self {
                    k: TqSideTables::build(device, full_head_dim, k_bits, KEY_SEED),
                    v: TqSideTables::build(device, full_head_dim, v_bits, VALUE_SEED),
                    full_head_dim,
                })
            }
        }
    }
}
