//! The DFlash2 block-diffusion drafter's ingest (`docs/DFLASH2.md`).
//!
//! The second drafter this walk knows, after `mtp.rs`, and it splits FOUR
//! ways where the head splits two. The MTP head's rule was "rank 2
//! quantizes, rank 1 narrows", and the drafter is almost that plus two
//! exceptions the head has no analogue for:
//!
//! - `*.base_kernel` is RANK 3 (`[2 sides, 2 taps, hidden]`, 20 KB), the
//!   drafter's static conv taps. Narrowed raw to BF16; there is nothing to
//!   quantize at that size and the conv kernel reads it as plain BF16.
//! - `candidate_selector.*_codebook` is rank 2 but is NOT a projection:
//!   nothing ever multiplies against it on the GPU, the selector GATHERS
//!   rows of it on the host, so quantizing it would buy 190 MB of resident
//!   space at the cost of a dequantizing gather. It stays BF16 raw.
//!
//! Everything else rank-2 (`fc`, the selector's `hidden_projection`, the
//! ten `kernel_projection`s, the attention and MLP projections) quantizes
//! to INT4 affine for exactly the reason `mtp.rs`'s header gives: this
//! engine dispatches no unquantized GEMV, so a BF16 matrix in an install
//! is a drafter that fails at its first dispatch. Quantizing a drafter
//! remains a throughput axis and never a correctness one.
//!
//! Names arrive `dflash.`-prefixed (see `classify::DFLASH_PREFIX`): the
//! published repository spells them bare and the CALLER renames the header
//! before the walk, so no name this module sees can collide with a trunk
//! tensor.

use super::config::Gemma4Error;
use super::mtp::decode_bf16;
use super::narrow::narrow_raw_to_bf16;
use super::shards::{shape4, Gemma4Shards};
use crate::repack::quantize_matrix_int4;
use crate::resident_writer::{RawTensorSpec, ResidentEntrySpec};

/// The drafter's resident entries plus what narrowing its raw tensors cost.
pub struct DflashRead {
    pub entries: Vec<ResidentEntrySpec>,
    /// One `(tensor, values that lost bits)` row per lossily-narrowed
    /// tensor. EMPTY for the real drafter, which is BF16 throughout.
    pub lossy_narrowing: Vec<(String, usize)>,
}

/// Whether a rank-2 drafter tensor is a selector codebook, i.e. one the
/// runtime gathers on the host rather than multiplying on the GPU.
///
/// Keyed on the tensor's own suffix rather than a count, so a selector that
/// grew a third table is ingested the same way without this function
/// moving.
fn is_codebook(name: &str) -> bool {
    name.ends_with("_codebook")
}

/// Reads the drafter: projections to INT4 affine, codebooks and conv base
/// kernels raw BF16, norms narrowed to BF16.
pub fn read_dflash_entries(
    shards: &Gemma4Shards<'_>,
    dflash_bases: &[&str],
) -> Result<DflashRead, Gemma4Error> {
    let mut entries = Vec::with_capacity(dflash_bases.len());
    let mut lossy_narrowing = Vec::new();

    for &name in dflash_bases {
        let t = shards.info(name)?;
        if !matches!(t.shape.len(), 1..=3) {
            return Err(Gemma4Error::ShapeMismatch {
                tensor: name.to_string(),
                detail: format!(
                    "a DFlash2 tensor of rank {}, expected 1, 2 or 3",
                    t.shape.len()
                ),
            });
        }
    }

    let mut cursor = 0usize;
    while cursor < dflash_bases.len() {
        let wave_start = cursor;
        let mut wave_bytes = 0u64;
        while cursor < dflash_bases.len() {
            let item_bytes = shards
                .location_of(dflash_bases[cursor])
                .map(|(_, s, e)| e.saturating_sub(s))
                .unwrap_or(0);
            if cursor > wave_start
                && wave_bytes.saturating_add(item_bytes) > super::shards::PREFETCH_WINDOW_BYTES
            {
                break;
            }
            wave_bytes = wave_bytes.saturating_add(item_bytes);
            cursor += 1;
        }
        let wave = &dflash_bases[wave_start..cursor];
        let _guard = shards.prefetch_guard(wave)?;
        for &name in wave {
            let t = shards.info(name)?;
            match t.shape.len() {
                2 if !is_codebook(name) => {
                    let rows = t.shape[0] as usize;
                    let cols = t.shape[1] as usize;
                    let data = decode_bf16(name, &t.dtype, shards.read(name)?)?;
                    if data.len() != rows * cols {
                        return Err(Gemma4Error::ShapeMismatch {
                            tensor: name.to_string(),
                            detail: format!(
                                "{} values does not match the declared {rows}x{cols}",
                                data.len()
                            ),
                        });
                    }
                    let quantized = quantize_matrix_int4(&data, rows, cols).map_err(|e| {
                        Gemma4Error::ShapeMismatch {
                            tensor: name.to_string(),
                            detail: e.to_string(),
                        }
                    })?;
                    entries.push(ResidentEntrySpec::Int4(
                        crate::repack::resident_spec_from_int4_rows(name, &quantized, cols),
                    ));
                }
                1..=3 => {
                    let narrowed = narrow_raw_to_bf16(name, &t.dtype, shards.read(name)?)?;
                    if narrowed.lossy > 0 {
                        lossy_narrowing.push((name.to_string(), narrowed.lossy));
                    }
                    entries.push(ResidentEntrySpec::Raw(RawTensorSpec {
                        name: name.to_string(),
                        dtype: narrowed.dtype,
                        bytes: narrowed.bytes,
                        shape: shape4(&t.shape),
                    }));
                }
                _ => unreachable!("rank checked above"),
            }
        }
    }

    Ok(DflashRead {
        entries,
        lossy_narrowing,
    })
}
