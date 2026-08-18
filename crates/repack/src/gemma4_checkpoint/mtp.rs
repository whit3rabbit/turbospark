//! The multi-token-prediction head's ingest (`docs/MTP_SPECULATIVE.md`,
//! step 1).
//!
//! **This is the walk's first QUANTIZING arm.** Every other resident tensor
//! reaches an install one of two ways: already packed by the publisher and
//! passed through (`pass_through_packed`), or unquantized and narrowed to
//! BF16 (`narrow_raw_to_bf16`). The head takes neither, and the reason is
//! that it comes from a DIFFERENT REPOSITORY than the trunk around it.
//!
//! `mlx-community/Qwen3.8-27B-4bit` -- the artifact every `qwen3_5` install
//! here is streamed from -- drops `mtp.*` in conversion
//! (`tests/qwen38_checkpoint_network.rs` says so in its header, and
//! `tests/mtp_head_network.rs` reads the head off the official
//! `Qwen/Qwen3.8-27B` instead). So the head arrives BF16, at full width,
//! where its trunk arrives pre-packed. Narrowing it would leave 849 MB of
//! BF16 matrices in an install whose engine dispatches no unquantized GEMV;
//! the first draft step would fail at `encode_gemv_any` with "no dispatched
//! GEMV kernel", four layers of code away from the cause.
//!
//! **Quantizing the drafter is not a numerics compromise**, which is what
//! licenses doing it here rather than growing a BF16 GEMV. A drafter's output
//! is verified by the target model and either accepted or discarded, so its
//! quality is a THROUGHPUT axis: a worse drafter proposes worse tokens, more
//! get rejected, and the speedup falls. It can never change what the engine
//! emits. `docs/MTP_SPECULATIVE.md` states the same thing from the other end
//! -- the whole appeal of MTP over a separate drafter is that the head is
//! 1.5% of a forward pass, and it is that only once it is INT4.
//!
//! The MATRIX/NORM split is by RANK, not by name. Every rank-2 tensor in the
//! head is a projection and every rank-1 one is an RMS norm; that holds for
//! all fifteen and is checked rather than assumed, because a name list would
//! have to be revised for a head with a different inventory while a rank
//! check would not. Norms stay BF16 for the reason `hf_checkpoint.rs`'s own
//! header gives for the opposite choice: quantizing a norm scale to INT4 is
//! not how a production repacker treats them.

use super::config::Gemma4Error;
use super::narrow::narrow_raw_to_bf16;
use super::shards::{shape4, Gemma4Shards};
use crate::repack::quantize_matrix_int4;
use crate::resident_writer::{RawTensorSpec, ResidentEntrySpec, ResidentTensorSpec};

/// The head's resident entries plus what narrowing its norms cost.
pub struct MtpRead {
    pub entries: Vec<ResidentEntrySpec>,
    /// One `(tensor, values that lost bits)` row per lossily-narrowed norm.
    /// EMPTY for the real head, whose every tensor is BF16 already -- a
    /// nonzero row here means the publisher changed the head's dtype, which
    /// is worth noticing rather than absorbing.
    pub lossy_narrowing: Vec<(String, usize)>,
}

/// Reads the head, quantizing its projections to INT4 affine and narrowing
/// its norms to BF16.
///
/// The INT4 group size is `compute::quant::GROUP_SIZE` (64), which is also
/// what the trunk's own INT4 install uses -- not a coincidence worth relying
/// on, but not a conflict either: the resident index records each tensor's
/// companion planes, so the runtime derives a group size per tensor rather
/// than reading one model-wide number.
pub fn read_mtp_entries(
    shards: &Gemma4Shards<'_>,
    mtp_bases: &[&str],
) -> Result<MtpRead, Gemma4Error> {
    let mut entries = Vec::with_capacity(mtp_bases.len());
    let mut lossy_narrowing = Vec::new();

    for &name in mtp_bases {
        let t = shards.info(name)?;
        match t.shape.len() {
            // A PROJECTION. Quantized, because the alternative is an install
            // the engine cannot dispatch.
            2 => {
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
                let mut packed = Vec::with_capacity(rows * cols / 2);
                let mut scales = Vec::new();
                let mut biases = Vec::new();
                for row in &quantized {
                    packed.extend_from_slice(&row.packed);
                    scales.extend_from_slice(&row.scales);
                    biases.extend_from_slice(&row.biases);
                }
                entries.push(ResidentEntrySpec::Int4(ResidentTensorSpec {
                    name: name.to_string(),
                    packed,
                    scales,
                    biases,
                    rows: rows as u32,
                    cols: cols as u32,
                }));
            }
            // AN RMS NORM. Narrowed, exactly as the trunk's norms are.
            1 => {
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
            // Refused rather than guessed at. The published head is fifteen
            // tensors of rank 1 and 2; a rank-3 one would mean a head with
            // more than one block bundled per tensor, which is a different
            // ingest and not a wider version of this one.
            other => {
                return Err(Gemma4Error::ShapeMismatch {
                    tensor: name.to_string(),
                    detail: format!("an MTP head tensor of rank {other}, expected 1 or 2"),
                })
            }
        }
    }

    Ok(MtpRead {
        entries,
        lossy_narrowing,
    })
}

/// Decodes an unquantized tensor to `f32` for quantization.
///
/// BF16 and F32 only, and F16 is REFUSED rather than decoded. The published
/// head is BF16 throughout and asserted to be; accepting F16 here would mean
/// guessing which of two same-width encodings a future publisher chose, which
/// is `crates/repack` Gotcha 9's silent failure exactly.
fn decode_bf16(tensor: &str, dtype: &str, bytes: Vec<u8>) -> Result<Vec<f32>, Gemma4Error> {
    match dtype {
        "BF16" => Ok(bytes
            .chunks_exact(2)
            .map(|c| compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        "F32" => Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        other => Err(Gemma4Error::UnsupportedDtype {
            tensor: tensor.to_string(),
            dtype: format!("{other} in an MTP head projection (expected BF16)"),
        }),
    }
}
