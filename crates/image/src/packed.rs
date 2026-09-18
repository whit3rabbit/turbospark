//! Packed image-component storage for the IG2 candidate quantization.
//!
//! This is a component format, not a replacement for the text `.gturbo`
//! format. Each tensor has a checked byte span in one payload file. Eligible
//! two-dimensional linear weights use the repository's affine INT4 group-64
//! layout; embeddings, norms, modulation tensors, and non-matrix tensors stay
//! unquantized, with F32 source tensors narrowed to BF16 to match the
//! reference execution dtype.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use compute::{dequantize_int4_affine, f32_to_bf16, quantize_int4_affine, Int4AffineRow};
use serde::{Deserialize, Serialize};

use crate::text_encoder::ShardedSafetensors;

pub const PACKED_INDEX_NAME: &str = "index.json";
pub const PACKED_DATA_NAME: &str = "tensors.bin";
pub const PACKED_MAGIC: &str = "turbospark.image.packed.v1";
pub const PACKED_VERSION: u32 = 1;
pub const PACKED_GROUP_SIZE: usize = 64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedIndex {
    pub magic: String,
    pub version: u32,
    pub data_file: String,
    pub data_sha256: String,
    pub tensor_inventory_sha256: String,
    pub tensors: BTreeMap<String, PackedTensor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedTensor {
    pub shape: Vec<usize>,
    pub source_dtype: String,
    pub storage_dtype: String,
    pub offset: u64,
    pub length: u64,
    pub quantization: Option<PackedQuantization>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedQuantization {
    pub scheme: String,
    pub group_size: usize,
    pub nibble_order: String,
    pub scale_and_bias_convention: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedTensorReport {
    pub tensor_count: usize,
    pub quantized_tensor_count: usize,
    pub payload_bytes: u64,
    pub data_sha256: String,
    pub tensor_inventory_sha256: String,
}

#[derive(Debug, Clone)]
pub struct PackedTensorStore {
    index: PackedIndex,
    data_path: PathBuf,
}

impl PackedTensorStore {
    pub fn open(root: &Path) -> Result<Self, String> {
        let index_path = root.join(PACKED_INDEX_NAME);
        let bytes = fs::read(&index_path).map_err(|e| {
            format!(
                "failed to read packed image index {}: {e}",
                index_path.display()
            )
        })?;
        let index: PackedIndex = serde_json::from_slice(&bytes).map_err(|e| {
            format!(
                "failed to parse packed image index {}: {e}",
                index_path.display()
            )
        })?;
        if index.magic != PACKED_MAGIC {
            return Err(format!(
                "packed image index magic {:?} is not {:?}",
                index.magic, PACKED_MAGIC
            ));
        }
        if index.version != PACKED_VERSION {
            return Err(format!(
                "unsupported packed image index version {}, expected {}",
                index.version, PACKED_VERSION
            ));
        }
        if index.data_file.is_empty() || Path::new(&index.data_file).is_absolute() {
            return Err("packed image index data_file must be a relative path".to_string());
        }
        if Path::new(&index.data_file)
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return Err("packed image index data_file escapes its component directory".to_string());
        }
        let data_path = root.join(&index.data_file);
        let data_len = fs::metadata(&data_path)
            .map_err(|e| {
                format!(
                    "failed to stat packed image payload {}: {e}",
                    data_path.display()
                )
            })?
            .len();
        if index.data_sha256.len() != 64
            || !index
                .data_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("packed image index has an invalid data_sha256".to_string());
        }
        let actual_data_sha256 = model_io::hash_file(&data_path, 1 << 20)
            .map_err(|e| format!("failed to hash packed image payload: {e}"))?;
        if actual_data_sha256 != index.data_sha256.to_ascii_lowercase() {
            return Err(format!(
                "packed image payload SHA-256 {}, expected {}",
                actual_data_sha256, index.data_sha256
            ));
        }
        if index.tensor_inventory_sha256.len() != 64
            || !index
                .tensor_inventory_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("packed image index has an invalid tensor_inventory_sha256".to_string());
        }
        let actual_inventory_sha256 = compute_tensor_inventory_sha256(&index.tensors);
        if actual_inventory_sha256 != index.tensor_inventory_sha256.to_ascii_lowercase() {
            return Err(format!(
                "packed image tensor inventory SHA-256 {}, expected {}",
                actual_inventory_sha256, index.tensor_inventory_sha256
            ));
        }

        let mut ranges = Vec::with_capacity(index.tensors.len());
        for (name, tensor) in &index.tensors {
            validate_tensor(name, tensor)?;
            let end = tensor
                .offset
                .checked_add(tensor.length)
                .ok_or_else(|| format!("packed tensor {name} byte range overflows"))?;
            if end > data_len {
                return Err(format!(
                    "packed tensor {name} byte range {}..{} exceeds payload size {}",
                    tensor.offset, end, data_len
                ));
            }
            ranges.push((tensor.offset, end, name.as_str()));
        }
        ranges.sort_unstable_by_key(|(start, _, _)| *start);
        for pair in ranges.windows(2) {
            if pair[1].0 < pair[0].1 {
                return Err(format!(
                    "packed tensor ranges overlap: {} and {}",
                    pair[0].2, pair[1].2
                ));
            }
        }

        Ok(Self { index, data_path })
    }

    pub fn contains_tensor(&self, name: &str) -> bool {
        self.index.tensors.contains_key(name)
    }

    /// Return the checked descriptor for a tensor without reading its
    /// payload. Native backends use this to bind a tensor's byte span
    /// directly from the mapped component buffer.
    pub fn tensor(&self, name: &str) -> Option<&PackedTensor> {
        self.index.tensors.get(name)
    }

    /// The immutable payload path named by the checked packed index.
    pub fn payload_path(&self) -> &Path {
        &self.data_path
    }

    /// Return the verified payload size without mapping or reading it.
    pub fn payload_bytes(&self) -> Result<u64, String> {
        fs::metadata(&self.data_path)
            .map(|metadata| metadata.len())
            .map_err(|e| format!("failed to stat packed image payload: {e}"))
    }

    /// Return the inclusive byte span containing a component block's tensors.
    /// The span includes padding and any interleaved tensors because the
    /// synchronous streamer reads one contiguous range per block.
    pub fn prefix_span(&self, prefix: &str) -> Option<(u64, u64)> {
        self.index
            .tensors
            .iter()
            .filter(|(name, _)| name.as_str() == prefix || name.starts_with(&format!("{prefix}.")))
            .fold(None, |range, (_, tensor)| {
                let end = tensor.offset.checked_add(tensor.length)?;
                Some(match range {
                    Some((start, current_end)) => (start.min(tensor.offset), current_end.max(end)),
                    None => (tensor.offset, end),
                })
            })
    }

    /// Read a checked byte range from the packed payload without mapping the
    /// whole component. Streaming image blocks use this after `open` has
    /// validated every tensor span and the payload hash.
    pub(crate) fn read_payload_range(&self, offset: u64, length: usize) -> Result<Vec<u8>, String> {
        let end = offset
            .checked_add(length as u64)
            .ok_or_else(|| "packed image payload range overflows".to_string())?;
        let payload_len = fs::metadata(&self.data_path)
            .map_err(|e| format!("failed to stat packed image payload: {e}"))?
            .len();
        if end > payload_len {
            return Err(format!(
                "packed image payload range {offset}..{end} exceeds {payload_len}"
            ));
        }
        self.read_range(offset, length)
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.index
            .tensors
            .get(name)
            .map(|tensor| tensor.shape.as_slice())
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.index.tensors.keys().map(String::as_str)
    }

    pub fn load_tensor(&self, name: &str) -> Result<Vec<f32>, String> {
        let tensor = self
            .index
            .tensors
            .get(name)
            .ok_or_else(|| format!("tensor {name} not found in packed index"))?;
        let bytes = self.read_tensor_range(tensor, 0, tensor.length)?;
        let elements = element_count(&tensor.shape)?;
        match tensor.storage_dtype.as_str() {
            "F32" => decode_f32(name, &bytes, elements),
            "BF16" => decode_bf16(name, &bytes, elements),
            "INT4_AFFINE" => decode_int4(name, tensor, &bytes, elements),
            other => Err(format!(
                "tensor {name} has unsupported storage dtype {other}"
            )),
        }
    }

    pub fn load_row(&self, name: &str, row: usize) -> Result<Vec<f32>, String> {
        let tensor = self
            .index
            .tensors
            .get(name)
            .ok_or_else(|| format!("tensor {name} not found in packed index"))?;
        if tensor.shape.len() != 2 || row >= tensor.shape[0] {
            return Err(format!("row {row} is out of bounds for tensor {name}"));
        }
        let cols = tensor.shape[1];
        match tensor.storage_dtype.as_str() {
            "F32" => {
                let row_bytes = cols
                    .checked_mul(4)
                    .ok_or_else(|| format!("tensor {name} row size overflows"))?;
                let offset = row
                    .checked_mul(row_bytes)
                    .ok_or_else(|| format!("tensor {name} row offset overflows"))?;
                let bytes = self.read_tensor_range(tensor, offset, row_bytes as u64)?;
                decode_f32(name, &bytes, cols)
            }
            "BF16" => {
                let row_bytes = cols
                    .checked_mul(2)
                    .ok_or_else(|| format!("tensor {name} row size overflows"))?;
                let offset = row
                    .checked_mul(row_bytes)
                    .ok_or_else(|| format!("tensor {name} row offset overflows"))?;
                let bytes = self.read_tensor_range(tensor, offset, row_bytes as u64)?;
                decode_bf16(name, &bytes, cols)
            }
            "INT4_AFFINE" => {
                let groups = cols / PACKED_GROUP_SIZE;
                let row_bytes = cols / 2 + groups * 4;
                let start = row
                    .checked_mul(row_bytes)
                    .ok_or_else(|| format!("tensor {name} row offset overflows"))?;
                let bytes = self.read_tensor_range(tensor, start, row_bytes as u64)?;
                decode_int4_row(name, &bytes, cols)
            }
            other => Err(format!(
                "tensor {name} has unsupported storage dtype {other}"
            )),
        }
    }

    fn read_tensor_range(
        &self,
        tensor: &PackedTensor,
        relative_offset: usize,
        length: u64,
    ) -> Result<Vec<u8>, String> {
        let relative_end = (relative_offset as u64)
            .checked_add(length)
            .ok_or_else(|| "packed tensor byte range overflows".to_string())?;
        if relative_end > tensor.length {
            return Err("packed tensor byte range exceeds its declared span".to_string());
        }
        let offset = tensor
            .offset
            .checked_add(relative_offset as u64)
            .ok_or_else(|| "packed tensor file offset overflows".to_string())?;
        let length = usize::try_from(length)
            .map_err(|_| "packed tensor byte length does not fit in usize".to_string())?;
        self.read_range(offset, length)
    }

    fn read_range(&self, offset: u64, length: usize) -> Result<Vec<u8>, String> {
        let mut file = File::open(&self.data_path)
            .map_err(|e| format!("failed to open packed image payload: {e}"))?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| format!("failed to seek packed image payload: {e}"))?;
        let mut bytes = vec![0u8; length];
        file.read_exact(&mut bytes)
            .map_err(|e| format!("failed to read packed image tensor: {e}"))?;
        Ok(bytes)
    }
}

/// Convert a safetensors component to the checked packed image format.
pub fn pack_component(
    source_dir: &Path,
    source_index_name: &str,
    output_dir: &Path,
) -> Result<PackedTensorReport, String> {
    if output_dir.exists() {
        return Err(format!(
            "refusing to overwrite existing packed component {}",
            output_dir.display()
        ));
    }
    let source = ShardedSafetensors::open_indexed(source_dir, source_index_name)?;
    fs::create_dir_all(output_dir).map_err(|e| {
        format!(
            "failed to create packed component {}: {e}",
            output_dir.display()
        )
    })?;
    let data_path = output_dir.join(PACKED_DATA_NAME);
    let index_path = output_dir.join(PACKED_INDEX_NAME);
    let data_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&data_path)
        .map_err(|e| format!("failed to create packed image payload: {e}"))?;
    let mut writer = BufWriter::new(data_file);
    let mut tensors = BTreeMap::new();
    let mut quantized_tensor_count = 0;

    for (name, shard_name) in source.source_tensors() {
        let shard = source
            .shards
            .get(shard_name)
            .ok_or_else(|| format!("source shard {shard_name} not loaded"))?;
        let descriptor = shard
            .descriptor(name)
            .ok_or_else(|| format!("source tensor {name} missing from shard {shard_name}"))?;
        let offset = writer
            .stream_position()
            .map_err(|e| format!("failed to query packed payload offset: {e}"))?;
        let quantized = can_quantize(name, &descriptor.shape);
        let (storage_dtype, quantization) = if quantized {
            let values = shard
                .load_as_f32(name)
                .map_err(|e| format!("failed to load source tensor {name}: {e}"))?;
            let cols = descriptor.shape[1];
            for row in values.chunks_exact(cols) {
                let packed = quantize_int4_affine(row);
                write_int4_row(&mut writer, &packed)?;
            }
            quantized_tensor_count += 1;
            (
                "INT4_AFFINE".to_string(),
                Some(PackedQuantization {
                    scheme: "four-bit-linear-weights-group-64".to_string(),
                    group_size: PACKED_GROUP_SIZE,
                    nibble_order: "low_nibble_even_high_nibble_odd".to_string(),
                    scale_and_bias_convention: "value = nibble * bf16_scale + bf16_bias"
                        .to_string(),
                }),
            )
        } else {
            let source_dtype = descriptor.dtype.to_ascii_uppercase();
            match source_dtype.as_str() {
                "F32" => {
                    let values = shard
                        .load_as_f32(name)
                        .map_err(|e| format!("failed to load source tensor {name}: {e}"))?;
                    for value in values {
                        writer
                            .write_all(&f32_to_bf16(value).to_le_bytes())
                            .map_err(|e| format!("failed to write narrowed tensor {name}: {e}"))?;
                    }
                    ("BF16".to_string(), None)
                }
                "BF16" => {
                    let raw = shard
                        .raw_bytes(name)
                        .map_err(|e| format!("failed to read source tensor {name}: {e}"))?;
                    let element_bytes = 2;
                    let expected = element_count(&descriptor.shape)?
                        .checked_mul(element_bytes)
                        .ok_or_else(|| format!("source tensor {name} byte length overflows"))?;
                    if raw.len() != expected {
                        return Err(format!(
                            "source tensor {name} has {} bytes, expected {}",
                            raw.len(),
                            expected
                        ));
                    }
                    writer
                        .write_all(raw)
                        .map_err(|e| format!("failed to write source tensor {name}: {e}"))?;
                    (source_dtype, None)
                }
                _ => {
                    let values = shard
                        .load_as_f32(name)
                        .map_err(|e| format!("failed to load source tensor {name}: {e}"))?;
                    for value in &values {
                        writer
                            .write_all(&value.to_le_bytes())
                            .map_err(|e| format!("failed to write source tensor {name}: {e}"))?;
                    }
                    ("F32".to_string(), None)
                }
            }
        };
        let end = writer
            .stream_position()
            .map_err(|e| format!("failed to query packed payload end: {e}"))?;
        let entry = PackedTensor {
            shape: descriptor.shape.clone(),
            source_dtype: descriptor.dtype.clone(),
            storage_dtype: storage_dtype.clone(),
            offset,
            length: end - offset,
            quantization,
        };
        tensors.insert(name.to_string(), entry);
    }
    writer
        .flush()
        .map_err(|e| format!("failed to flush packed image payload: {e}"))?;
    drop(writer);

    let data_sha256 = model_io::hash_file(&data_path, 1 << 20)
        .map_err(|e| format!("failed to hash packed image payload: {e}"))?;
    let tensor_inventory_sha256 = compute_tensor_inventory_sha256(&tensors);
    let payload_bytes = fs::metadata(&data_path)
        .map_err(|e| format!("failed to stat packed image payload: {e}"))?
        .len();
    let index = PackedIndex {
        magic: PACKED_MAGIC.to_string(),
        version: PACKED_VERSION,
        data_file: PACKED_DATA_NAME.to_string(),
        data_sha256: data_sha256.clone(),
        tensor_inventory_sha256: tensor_inventory_sha256.clone(),
        tensors,
    };
    let index_bytes = serde_json::to_vec_pretty(&index)
        .map_err(|e| format!("failed to serialize packed image index: {e}"))?;
    let mut index_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&index_path)
        .map_err(|e| format!("failed to create packed image index: {e}"))?;
    index_file
        .write_all(&index_bytes)
        .map_err(|e| format!("failed to write packed image index: {e}"))?;

    Ok(PackedTensorReport {
        tensor_count: index.tensors.len(),
        quantized_tensor_count,
        payload_bytes,
        data_sha256,
        tensor_inventory_sha256,
    })
}

fn validate_tensor(name: &str, tensor: &PackedTensor) -> Result<(), String> {
    let elements = element_count(&tensor.shape)?;
    if tensor.length == 0 {
        return Err(format!("packed tensor {name} has an empty byte span"));
    }
    match tensor.storage_dtype.as_str() {
        "F32" => {
            let expected = elements
                .checked_mul(4)
                .ok_or_else(|| format!("packed tensor {name} F32 byte length overflows"))?;
            if tensor.length != expected as u64 {
                return Err(format!(
                    "packed tensor {name} F32 length {} does not match {} elements",
                    tensor.length, elements
                ));
            }
            if tensor.quantization.is_some() {
                return Err(format!(
                    "packed tensor {name} has quantization metadata but is F32"
                ));
            }
        }
        "BF16" => {
            let expected = elements
                .checked_mul(2)
                .ok_or_else(|| format!("packed tensor {name} BF16 byte length overflows"))?;
            if tensor.length != expected as u64 {
                return Err(format!(
                    "packed tensor {name} BF16 length {} does not match {} elements",
                    tensor.length, elements
                ));
            }
            if tensor.quantization.is_some() {
                return Err(format!(
                    "packed tensor {name} has quantization metadata but is BF16"
                ));
            }
        }
        "INT4_AFFINE" => {
            if tensor.shape.len() != 2 || tensor.shape[1] % PACKED_GROUP_SIZE != 0 {
                return Err(format!(
                    "packed INT4 tensor {name} has invalid shape {:?}",
                    tensor.shape
                ));
            }
            let quantization = tensor.quantization.as_ref().ok_or_else(|| {
                format!("packed INT4 tensor {name} is missing quantization metadata")
            })?;
            if quantization.group_size != PACKED_GROUP_SIZE {
                return Err(format!("packed tensor {name} has unsupported group size"));
            }
            let rows = tensor.shape[0];
            let cols = tensor.shape[1];
            let expected = rows * (cols / 2 + cols / PACKED_GROUP_SIZE * 4);
            if tensor.length != expected as u64 {
                return Err(format!(
                    "packed tensor {name} INT4 length {} does not match expected {}",
                    tensor.length, expected
                ));
            }
        }
        other => {
            return Err(format!(
                "packed tensor {name} has unsupported storage dtype {other}"
            ))
        }
    }
    Ok(())
}

fn element_count(shape: &[usize]) -> Result<usize, String> {
    if shape.is_empty() {
        return Err("packed tensor shape cannot be empty".to_string());
    }
    shape.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| "packed tensor element count overflows".to_string())
    })
}

pub(crate) fn compute_tensor_inventory_sha256(tensors: &BTreeMap<String, PackedTensor>) -> String {
    let mut inventory = String::new();
    for (name, tensor) in tensors {
        inventory.push_str(name);
        inventory.push('|');
        inventory.push_str(&format!(
            "{:?}|{}|{}|{}|{}\n",
            tensor.shape, tensor.source_dtype, tensor.storage_dtype, tensor.offset, tensor.length
        ));
    }
    model_io::hash_data(inventory.as_bytes())
}

fn decode_f32(name: &str, bytes: &[u8], elements: usize) -> Result<Vec<f32>, String> {
    let expected_bytes = elements
        .checked_mul(4)
        .ok_or_else(|| format!("packed tensor {name} F32 byte length overflows"))?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "packed tensor {name} has an invalid F32 byte length"
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn decode_bf16(name: &str, bytes: &[u8], elements: usize) -> Result<Vec<f32>, String> {
    let expected_bytes = elements
        .checked_mul(2)
        .ok_or_else(|| format!("packed tensor {name} BF16 byte length overflows"))?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "packed tensor {name} has an invalid BF16 byte length"
        ));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| f32::from_bits((u16::from_le_bytes(chunk.try_into().unwrap()) as u32) << 16))
        .collect())
}

fn decode_int4(
    name: &str,
    tensor: &PackedTensor,
    bytes: &[u8],
    elements: usize,
) -> Result<Vec<f32>, String> {
    let rows = tensor.shape[0];
    let cols = tensor.shape[1];
    let row_bytes = cols / 2 + cols / PACKED_GROUP_SIZE * 4;
    if bytes.len() != rows * row_bytes || elements != rows * cols {
        return Err(format!(
            "packed tensor {name} has an invalid INT4 byte length"
        ));
    }
    let mut output = Vec::with_capacity(elements);
    for row in bytes.chunks_exact(row_bytes) {
        output.extend(decode_int4_row(name, row, cols)?);
    }
    Ok(output)
}

fn decode_int4_row(name: &str, row: &[u8], cols: usize) -> Result<Vec<f32>, String> {
    let packed_len = cols / 2;
    let scale_len = cols / PACKED_GROUP_SIZE * 2;
    let expected_len = packed_len + scale_len * 2;
    if row.len() != expected_len {
        return Err(format!(
            "packed tensor {name} has an invalid INT4 byte length"
        ));
    }
    let packed = row[..packed_len].to_vec();
    let scales = read_u16_vec(&row[packed_len..packed_len + scale_len]);
    let biases = read_u16_vec(&row[packed_len + scale_len..]);
    Ok(dequantize_int4_affine(
        &Int4AffineRow {
            packed,
            scales,
            biases,
        },
        cols,
    ))
}

fn read_u16_vec(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn write_int4_row(writer: &mut BufWriter<File>, row: &Int4AffineRow) -> Result<(), String> {
    writer
        .write_all(&row.packed)
        .map_err(|e| format!("failed to write packed INT4 values: {e}"))?;
    for value in row.scales.iter().chain(&row.biases) {
        writer
            .write_all(&value.to_le_bytes())
            .map_err(|e| format!("failed to write packed INT4 affine values: {e}"))?;
    }
    Ok(())
}

fn can_quantize(name: &str, shape: &[usize]) -> bool {
    shape.len() == 2
        && shape[0] > 0
        && shape[1] % PACKED_GROUP_SIZE == 0
        && name.ends_with(".weight")
        // Keep the packer aligned with the frozen IG0 emulation. Input and
        // output projections, caption/image embedders, modulation, and other
        // control tensors remain higher precision even when their shape fits
        // the generic group-64 storage format.
        && is_quality_quantized_projection(name)
        && !is_protected_tensor(name)
}

fn is_quality_quantized_projection(name: &str) -> bool {
    name.contains(".self_attn.")
        || name.contains(".mlp.")
        || name.contains(".attention.to_")
        || name.contains(".feed_forward.w")
}

fn is_protected_tensor(name: &str) -> bool {
    [
        "embed_tokens",
        "norm",
        "pad_token",
        "modulation",
        "position",
        "pos_embed",
        "rope",
        "t_embedder",
    ]
    .iter()
    .any(|part| name.contains(part))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn packs_linear_weights_with_the_shared_int4_layout() {
        let root = temporary_directory("pack");
        let source = root.join("source");
        let output = root.join("packed");
        fs::create_dir_all(&source).expect("create source");

        let mut payload = Vec::new();
        let linear: Vec<f32> = (0..128).map(|value| value as f32 / 10.0 - 4.0).collect();
        let norm: Vec<f32> = (0..64).map(|value| value as f32).collect();
        for value in linear.iter().chain(&norm) {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let header = serde_json::json!({
            "layers.0.attention.to_q.weight": {
                "dtype": "F32",
                "shape": [2, 64],
                "data_offsets": [0, 512]
            },
            "norm.weight": {
                "dtype": "F32",
                "shape": [64],
                "data_offsets": [512, 768]
            }
        });
        let header_bytes = serde_json::to_vec(&header).expect("serialize safetensors header");
        let mut shard = Vec::with_capacity(8 + header_bytes.len() + payload.len());
        shard.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        shard.extend_from_slice(&header_bytes);
        shard.extend_from_slice(&payload);
        fs::write(source.join("shard.safetensors"), shard).expect("write source shard");
        fs::write(
            source.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "weight_map": {
                    "layers.0.attention.to_q.weight": "shard.safetensors",
                    "norm.weight": "shard.safetensors"
                }
            }))
            .expect("serialize source index"),
        )
        .expect("write source index");

        let report = pack_component(&source, "model.safetensors.index.json", &output)
            .expect("pack source component");
        assert_eq!(report.tensor_count, 2);
        assert_eq!(report.quantized_tensor_count, 1);

        let store = PackedTensorStore::open(&output).expect("open packed component");
        let quantized = store
            .load_tensor("layers.0.attention.to_q.weight")
            .expect("load int4 tensor");
        assert_eq!(quantized.len(), linear.len());
        assert!(quantized
            .iter()
            .zip(&linear)
            .all(|(actual, expected)| (actual - expected).abs() < 0.3));
        let expected_norm: Vec<f32> = norm
            .iter()
            .map(|value| compute::bf16_to_f32(compute::f32_to_bf16(*value)))
            .collect();
        assert_eq!(
            store
                .index
                .tensors
                .get("norm.weight")
                .expect("norm index entry")
                .storage_dtype,
            "BF16"
        );
        assert_eq!(
            store.load_tensor("norm.weight").expect("load BF16 tensor"),
            expected_norm
        );
        assert_eq!(
            store
                .load_row("layers.0.attention.to_q.weight", 1)
                .expect("load packed row"),
            quantized[64..].to_vec()
        );

        fs::remove_dir_all(root).expect("remove test directory");
    }

    #[test]
    fn quantizes_only_the_frozen_quality_projection_families() {
        let shape = [3840, 3840];
        for name in [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "layers.0.attention.to_q.weight",
            "layers.0.feed_forward.w1.weight",
        ] {
            assert!(can_quantize(name, &shape), "expected {name} to quantize");
        }
        for name in [
            "all_x_embedder.2-1.weight",
            "cap_embedder.1.weight",
            "all_final_layer.2-1.linear.weight",
            "layers.0.attention_norm1.weight",
            "t_embedder.mlp.0.weight",
        ] {
            assert!(
                !can_quantize(name, &shape),
                "expected {name} to stay precise"
            );
        }
    }

    #[test]
    fn refuses_a_modified_packed_payload() {
        let root = temporary_directory("hash");
        let data_path = root.join(PACKED_DATA_NAME);
        fs::create_dir_all(&root).expect("create test directory");
        fs::write(&data_path, [1u8, 2, 3, 4]).expect("write payload");
        let index = PackedIndex {
            magic: PACKED_MAGIC.to_string(),
            version: PACKED_VERSION,
            data_file: PACKED_DATA_NAME.to_string(),
            data_sha256: "00".repeat(32),
            tensor_inventory_sha256: "11".repeat(32),
            tensors: BTreeMap::from([(
                "x".to_string(),
                PackedTensor {
                    shape: vec![1],
                    source_dtype: "F32".to_string(),
                    storage_dtype: "F32".to_string(),
                    offset: 0,
                    length: 4,
                    quantization: None,
                },
            )]),
        };
        fs::write(
            root.join(PACKED_INDEX_NAME),
            serde_json::to_vec(&index).expect("serialize packed index"),
        )
        .expect("write packed index");
        let error = PackedTensorStore::open(&root).expect_err("modified payload must fail");
        assert!(error.contains("SHA-256"));
        fs::remove_dir_all(root).expect("remove test directory");
    }

    fn temporary_directory(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "turbospark-image-{label}-{}-{nanos}",
            std::process::id()
        ))
    }
}
