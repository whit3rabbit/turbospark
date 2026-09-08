//! Reads an EXISTING install's `model_weights.bin` back into
//! [`crate::resident_writer::ResidentEntrySpec`]s -- the exact inverse of
//! `resident_writer::build_resident_weights_bin_mixed`.
//!
//! What licenses this: that function does not care where a spec's bytes came
//! from, so an entry read back byte for byte off disk and one freshly
//! quantized from a network shard are indistinguishable to it. The one
//! caller today is grafting a multi-token-prediction head onto an install
//! already on disk without re-streaming its trunk over the network
//! (`docs/MTP_SPECULATIVE.md`'s graft step;
//! `gemma4_checkpoint::graft_qwen_gdn_dense_mtp_head`).

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::resident_writer::{
    RawTensorSpec, ResidentEntrySpec, ResidentTensorSpec, DTYPE_BF16, DTYPE_FP16, DTYPE_FP32,
    DTYPE_INT1_AFFINE, DTYPE_INT2_AFFINE, DTYPE_INT4_AFFINE, DTYPE_INT8_AFFINE,
};

fn le_u16s(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Reads every resident entry out of `dir/model_weights.bin`, in on-disk
/// (file-offset) order, as the specs that would rebuild it byte for byte
/// through [`crate::resident_writer::build_resident_weights_bin_mixed`].
///
/// Refuses a GGUF-block dtype by name: those have no [`ResidentEntrySpec`]
/// variant to round-trip through. An INT4/INT8/INT1/INT2-affine or raw
/// BF16/FP16/FP32 tensor is the whole set this reads, which is exactly the
/// set an MLX-sourced install (`SourceKind::Mlx`) ever writes.
pub fn read_resident_entries(dir: &Path) -> Result<Vec<ResidentEntrySpec>, String> {
    let weights_path = dir.join("model_weights.bin");
    let index = model_io::load_resident_index(&weights_path).map_err(|e| e.to_string())?;
    let mut file = std::fs::File::open(&weights_path)
        .map_err(|e| format!("opening {}: {e}", weights_path.display()))?;

    let mut entries: Vec<_> = index.entries.into_values().collect();
    // File-offset order, so a read-then-rebuild round trip through this
    // reader reproduces the order the original writer placed these tensors
    // in, apart from whatever the caller appends afterward.
    entries.sort_by_key(|e| e.file_offset);

    let mut read_at = |offset: u64, size: u64| -> Result<Vec<u8>, String> {
        let mut buf = vec![0u8; size as usize];
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| format!("seeking {}: {e}", weights_path.display()))?;
        file.read_exact(&mut buf)
            .map_err(|e| format!("reading {}: {e}", weights_path.display()))?;
        Ok(buf)
    };

    let mut specs = Vec::with_capacity(entries.len());
    for e in entries {
        let bytes = read_at(e.file_offset, e.size_bytes)?;
        let spec = match e.dtype {
            DTYPE_BF16 | DTYPE_FP16 | DTYPE_FP32 => ResidentEntrySpec::Raw(RawTensorSpec {
                name: e.name,
                dtype: e.dtype,
                bytes,
                shape: e.shape,
            }),
            tag @ (DTYPE_INT4_AFFINE | DTYPE_INT8_AFFINE | DTYPE_INT1_AFFINE
            | DTYPE_INT2_AFFINE) => {
                let scales = le_u16s(&read_at(e.scale_offset, e.scale_size)?);
                let biases = le_u16s(&read_at(e.bias_offset, e.bias_size)?);
                let t = ResidentTensorSpec {
                    name: e.name,
                    packed: bytes,
                    scales,
                    biases,
                    rows: e.shape.0,
                    cols: e.shape.1,
                };
                match tag {
                    DTYPE_INT4_AFFINE => ResidentEntrySpec::Int4(t),
                    DTYPE_INT8_AFFINE => ResidentEntrySpec::Int8(t),
                    DTYPE_INT1_AFFINE => ResidentEntrySpec::Int1(t),
                    _ => ResidentEntrySpec::Int2(t),
                }
            }
            other => {
                return Err(format!(
                    "{}: dtype {other} has no ResidentEntrySpec to round-trip through \
                     (a GGUF block type?); this install cannot be re-emitted from its \
                     resident index alone",
                    e.name
                ))
            }
        };
        specs.push(spec);
    }
    Ok(specs)
}
