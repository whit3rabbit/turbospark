//! Unlimited-OCR-2's tower file: SAM, CLIP, the projector and the two learned
//! separator rows, as raw FP16 in `uocr2_tower.bin`, indexed by
//! `uocr2_tower.json`.
//!
//! A separate file rather than resident tensors, for two reasons. The tower's
//! kernels (`crates/gpu/src/vision.rs`) read FP16 weights where the resident
//! walk narrows to BF16, and a text prompt should pay nothing for a tower it
//! never runs: the runtime reads this file on the first image.
//!
//! The projector is 4-bit affine in the checkpoint, the one quantized tower
//! tensor, and is dequantized here because the tower runs no quantized GEMV.

use std::io::Write;
use std::path::Path;

use crate::gemma4_checkpoint::{convert_raw_to_fp16, ConvertedFp16, Gemma4Error, Gemma4Shards};
use crate::safetensors_header::SafetensorsHeader;
use crate::unlimited_ocr2_config::UOCR2_TOWER_PREFIX;

pub const UOCR2_TOWER_BIN: &str = "uocr2_tower.bin";
pub const UOCR2_TOWER_INDEX: &str = "uocr2_tower.json";

fn bf16_bits(bytes: &[u8]) -> impl Iterator<Item = u16> + '_ {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
}

/// `y = W x + b`'s weight, as `[rows, cols]` FP16, from the packed INT4 plane.
fn dequantize_projector(
    shards: &Gemma4Shards<'_>,
    base: &str,
) -> Result<(ConvertedFp16, Vec<u64>), Gemma4Error> {
    let weight = format!("{base}.weight");
    let info = shards.info(&weight)?;
    let (rows, words) = (info.shape[0] as usize, info.shape[1] as usize);
    let cols = words * 8;
    let packed = shards.read(&weight)?;
    let scales: Vec<u16> = bf16_bits(&shards.read(&format!("{base}.scales"))?).collect();
    let biases: Vec<u16> = bf16_bits(&shards.read(&format!("{base}.biases"))?).collect();
    let groups = cols / 64;
    let mut values = Vec::with_capacity(rows * cols * 4);
    for r in 0..rows {
        let row = compute::Int4AffineRow {
            packed: packed[r * words * 4..(r + 1) * words * 4].to_vec(),
            scales: scales[r * groups..(r + 1) * groups].to_vec(),
            biases: biases[r * groups..(r + 1) * groups].to_vec(),
        };
        for v in compute::dequantize_int4_affine(&row, cols) {
            values.extend_from_slice(&v.to_le_bytes());
        }
    }
    Ok((
        convert_raw_to_fp16(&weight, "F32", values)?,
        vec![rows as u64, cols as u64],
    ))
}

/// Write the tower beside an install. `headers` are the canonicalized ones.
pub fn write_unlimited_ocr2_tower(
    dir: &Path,
    headers: &[SafetensorsHeader],
    shards: &Gemma4Shards<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let projector_base = format!("{UOCR2_TOWER_PREFIX}projector");
    let companions = [
        format!("{projector_base}.scales"),
        format!("{projector_base}.biases"),
    ];
    let mut file = std::io::BufWriter::new(std::fs::File::create(dir.join(UOCR2_TOWER_BIN))?);
    let mut index = serde_json::Map::new();
    let mut offset = 0u64;
    for (name, info) in headers
        .iter()
        .flat_map(|h| h.tensors.iter())
        .filter(|(n, _)| n.starts_with(UOCR2_TOWER_PREFIX))
    {
        if companions.contains(name) {
            continue;
        }
        let (converted, shape) = if *name == format!("{projector_base}.weight") {
            dequantize_projector(shards, &projector_base)?
        } else {
            // Refuses an overflow by name, and counts what underflowed.
            (
                convert_raw_to_fp16(name, &info.dtype, shards.read(name)?)?,
                info.shape.clone(),
            )
        };
        let bytes = converted.bytes;
        file.write_all(&bytes)?;
        let key = name.strip_prefix(UOCR2_TOWER_PREFIX).unwrap_or(name);
        index.insert(
            key.to_string(),
            serde_json::json!({ "offset": offset, "shape": shape, "lossy": converted.lossy }),
        );
        offset += bytes.len() as u64;
    }
    file.flush()?;
    std::fs::write(
        dir.join(UOCR2_TOWER_INDEX),
        serde_json::to_vec_pretty(&serde_json::Value::Object(index))?,
    )?;
    Ok(())
}
