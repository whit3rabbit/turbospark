//! Streams Qwen4Exp's inline IQ4_NL PLE table into its own row store.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use model_io::{ArchConfig, ModelFamily, NgramTableLayout};

use crate::gguf_checkpoint::types::GgufRepackError;
use crate::gguf_header::GgufHeader;
use crate::ranged_download::RangeSource;

const TENSOR_NAME: &str = "per_layer_token_embd.weight";
const IQ4_NL_TYPE: u32 = 20;
const IQ4_NL_BLOCK_ELEMENTS: u64 = 32;
const IQ4_NL_BLOCK_BYTES: u64 = 18;
const READ_CHUNK_ROWS: u64 = 65_536;
const PROGRESS_BYTES: u64 = 1 << 30;

/// Writes the large PLE tensor as contiguous IQ4_NL rows without ever
/// materializing more than one bounded range read.
pub(super) fn write_gguf_ngram_table(
    dir: &Path,
    header: &GgufHeader,
    source: &dyn RangeSource,
    arch: &ArchConfig,
    tensor_name: &str,
    mut progress: impl FnMut(&str),
) -> Result<(), GgufRepackError> {
    if arch.family != ModelFamily::Qwen4Exp || tensor_name != TENSOR_NAME {
        return Err(GgufRepackError::ShapeMismatch {
            tensor: tensor_name.to_string(),
            detail: "separate IQ4_NL PLE storage is only defined for Qwen4Exp".to_string(),
        });
    }

    let layer_indices = arch.ple.layer_indices();
    let [layer_index] = layer_indices[..] else {
        return Err(GgufRepackError::ShapeMismatch {
            tensor: tensor_name.to_string(),
            detail: format!(
                "Qwen4Exp PLE declares {} layers; the table store requires exactly one",
                layer_indices.len()
            ),
        });
    };

    let order_count = arch.ple.ngram_size.checked_sub(1).filter(|&n| n > 0);
    let head_count = order_count
        .and_then(|n| n.checked_mul(arch.ple.heads_per_ngram))
        .filter(|&n| n > 0)
        .ok_or_else(|| bad_shape(tensor_name, "invalid PLE n-gram head count"))?;
    let embed_dim = u64::try_from(arch.ple.ple_embed_dim)
        .map_err(|_| bad_shape(tensor_name, "PLE embedding width is negative"))?;
    let head_count = u64::try_from(head_count)
        .map_err(|_| bad_shape(tensor_name, "PLE head count is negative"))?;
    if embed_dim % head_count != 0 {
        return Err(bad_shape(
            tensor_name,
            "PLE embedding width does not divide evenly across hash heads",
        ));
    }
    let head_dim = embed_dim / head_count;

    let info = header
        .tensors
        .get(tensor_name)
        .ok_or_else(|| GgufRepackError::MissingTensor {
            name: tensor_name.to_string(),
        })?;
    if info.ggml_type != IQ4_NL_TYPE {
        return Err(bad_shape(
            tensor_name,
            format!("expected IQ4_NL type {IQ4_NL_TYPE}, got {}", info.ggml_type),
        ));
    }
    let Some((block_elements, block_bytes)) = crate::gguf_header::ggml_type_block(info.ggml_type)
    else {
        return Err(bad_shape(
            tensor_name,
            "IQ4_NL has no registered block size",
        ));
    };
    if block_elements != IQ4_NL_BLOCK_ELEMENTS || block_bytes != IQ4_NL_BLOCK_BYTES {
        return Err(bad_shape(
            tensor_name,
            format!("unexpected IQ4_NL block {block_elements} elements / {block_bytes} bytes"),
        ));
    }
    if info.dims.len() != 2 || info.dims[0] != head_dim {
        return Err(bad_shape(
            tensor_name,
            format!(
                "expected GGUF dimensions [{head_dim}, rows], got {:?}",
                info.dims
            ),
        ));
    }
    if head_dim % IQ4_NL_BLOCK_ELEMENTS != 0 {
        return Err(bad_shape(
            tensor_name,
            format!("row width {head_dim} is not a whole number of IQ4_NL blocks"),
        ));
    }
    let rows = info.dims[1];
    let shards = u64::try_from(arch.ple.split_ngram_parts)
        .ok()
        .filter(|&n| n > 0)
        .ok_or_else(|| bad_shape(tensor_name, "PLE shard count must be positive"))?;
    if rows == 0 || rows % shards != 0 {
        return Err(bad_shape(
            tensor_name,
            format!("{rows} rows do not divide evenly across {shards} PLE shards"),
        ));
    }

    let weight_bytes = (head_dim / IQ4_NL_BLOCK_ELEMENTS)
        .checked_mul(IQ4_NL_BLOCK_BYTES)
        .ok_or_else(|| bad_shape(tensor_name, "IQ4_NL row width overflows"))?;
    let record_bytes = weight_bytes;
    let multipliers = metadata_i64_array(header, "ple.layer_multipliers")?;
    let head_vocab_sizes = metadata_i64_array(header, "ple.head_vocab_sizes")?;
    let head_offsets = metadata_i64_array(header, "ple.head_offsets")?;
    let expected_multiplier_count = usize::try_from(arch.ple.ngram_size)
        .map_err(|_| bad_shape(tensor_name, "PLE multiplier count is not representable"))?;
    let expected_head_count = usize::try_from(head_count)
        .map_err(|_| bad_shape(tensor_name, "PLE head count is not representable"))?;
    if multipliers.len() != expected_multiplier_count
        || head_vocab_sizes.len() != expected_head_count
        || head_offsets.len() != expected_head_count
    {
        return Err(bad_shape(
            tensor_name,
            format!(
                "hash arrays have lengths multipliers={}, vocab={}, offsets={}; expected \
                 {expected_multiplier_count}, {expected_head_count}, {expected_head_count}",
                multipliers.len(),
                head_vocab_sizes.len(),
                head_offsets.len()
            ),
        ));
    }
    let layout = NgramTableLayout {
        version: NgramTableLayout::VERSION,
        rows,
        rows_per_shard: rows / shards,
        shards,
        head_dim,
        group_size: IQ4_NL_BLOCK_ELEMENTS,
        bits: 4,
        record_bytes,
        weight_bytes,
        scale_bytes: 0,
        bias_bytes: 0,
        companion_dtype: "inline".to_string(),
        ggml_type: Some("iq4_nl".to_string()),
        layer_index: u64::try_from(layer_index)
            .map_err(|_| bad_shape(tensor_name, "PLE layer index is negative"))?,
        multipliers,
        head_vocab_sizes,
        head_offsets,
    };
    layout
        .validate()
        .map_err(|e| bad_shape(tensor_name, format!("invalid n-gram table metadata: {e:?}")))?;

    let (start, end) =
        header
            .absolute_range(tensor_name)
            .ok_or_else(|| GgufRepackError::MissingTensor {
                name: tensor_name.to_string(),
            })??;
    let expected_bytes = layout
        .blob_bytes()
        .ok_or_else(|| bad_shape(tensor_name, "table byte length overflows"))?;
    if end.checked_sub(start) != Some(expected_bytes) {
        return Err(bad_shape(
            tensor_name,
            format!(
                "tensor range is {} bytes but {} IQ4_NL rows require {expected_bytes}",
                end.saturating_sub(start),
                rows
            ),
        ));
    }

    let table_dir = dir.join(model_io::NGRAM_TABLE_DIR);
    std::fs::create_dir_all(&table_dir).map_err(|e| io_error(&table_dir, e))?;
    let header_path = table_dir.join(model_io::NGRAM_TABLE_HEADER);
    // A failed rewrite must not leave a previous header pointing at a partial
    // rows.bin file.
    if header_path.exists() {
        std::fs::remove_file(&header_path).map_err(|e| io_error(&header_path, e))?;
    }
    let blob_path = table_dir.join(model_io::NGRAM_TABLE_BLOB);
    let blob = File::create(&blob_path).map_err(|e| io_error(&blob_path, e))?;
    let mut blob = BufWriter::with_capacity(1 << 20, blob);
    progress(&format!(
        "streaming Qwen4Exp PLE table: {rows} IQ4_NL rows, {expected_bytes} bytes"
    ));
    copy_tensor_rows(
        source,
        start..end,
        rows,
        record_bytes,
        READ_CHUNK_ROWS,
        &mut blob,
        &mut progress,
    )?;
    blob.flush().map_err(|e| io_error(&blob_path, e))?;
    drop(blob);
    let actual_bytes = std::fs::metadata(&blob_path)
        .map_err(|e| io_error(&blob_path, e))?
        .len();
    if actual_bytes != expected_bytes {
        return Err(bad_shape(
            tensor_name,
            format!("wrote {actual_bytes} bytes, expected {expected_bytes}"),
        ));
    }

    let header_bytes = serde_json::to_vec_pretty(&layout)
        .map_err(|e| bad_shape(tensor_name, format!("serializing table header: {e}")))?;
    if header_bytes.len() as u64 > model_io::NGRAM_HEADER_MAX_BYTES {
        return Err(bad_shape(
            tensor_name,
            "table header exceeds its reader cap",
        ));
    }
    let temp_header_path = table_dir.join(format!("{}.tmp", model_io::NGRAM_TABLE_HEADER));
    let mut temp_header =
        File::create(&temp_header_path).map_err(|e| io_error(&temp_header_path, e))?;
    temp_header
        .write_all(&header_bytes)
        .and_then(|()| temp_header.sync_all())
        .map_err(|e| io_error(&temp_header_path, e))?;
    drop(temp_header);
    std::fs::rename(&temp_header_path, &header_path).map_err(|e| io_error(&header_path, e))?;

    model_io::load_ngram_table_layout(dir)
        .map_err(|e| bad_shape(tensor_name, format!("reader rejected table header: {e:?}")))?
        .ok_or_else(|| bad_shape(tensor_name, "table header did not read back"))?;
    progress(&format!("Qwen4Exp PLE table written ({rows} rows)"));
    Ok(())
}

fn metadata_i64_array(header: &GgufHeader, suffix: &str) -> Result<Vec<i64>, GgufRepackError> {
    let key = format!("qwen4exp.{suffix}");
    let value = header
        .metadata
        .get(&key)
        .ok_or_else(|| GgufRepackError::ShapeMismatch {
            tensor: TENSOR_NAME.to_string(),
            detail: format!("missing GGUF metadata {key}"),
        })?;
    let values = value
        .as_array()
        .ok_or_else(|| GgufRepackError::ShapeMismatch {
            tensor: TENSOR_NAME.to_string(),
            detail: format!("GGUF metadata {key} is not an integer array"),
        })?;
    values
        .iter()
        .map(|v| {
            let raw = v.as_u64().ok_or_else(|| GgufRepackError::ShapeMismatch {
                tensor: TENSOR_NAME.to_string(),
                detail: format!("GGUF metadata {key} contains a non-unsigned integer"),
            })?;
            i64::try_from(raw).map_err(|_| GgufRepackError::ShapeMismatch {
                tensor: TENSOR_NAME.to_string(),
                detail: format!("GGUF metadata {key} contains a value above i64::MAX"),
            })
        })
        .collect()
}

fn copy_tensor_rows(
    source: &dyn RangeSource,
    source_range: std::ops::Range<u64>,
    rows: u64,
    row_bytes: u64,
    chunk_rows: u64,
    sink: &mut dyn Write,
    progress: &mut dyn FnMut(&str),
) -> Result<(), GgufRepackError> {
    let start = source_range.start;
    let end = source_range.end;
    let expected = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| bad_shape(TENSOR_NAME, "table byte length overflows"))?;
    if row_bytes == 0 || chunk_rows == 0 || end.checked_sub(start) != Some(expected) {
        return Err(bad_shape(
            TENSOR_NAME,
            "invalid row range, row width, or chunk size",
        ));
    }

    let mut row = 0u64;
    let mut last_progress_gib = 0u64;
    while row < rows {
        let rows_this_chunk = chunk_rows.min(rows - row);
        let chunk_bytes = rows_this_chunk
            .checked_mul(row_bytes)
            .ok_or_else(|| bad_shape(TENSOR_NAME, "chunk byte length overflows"))?;
        let chunk_start = start
            .checked_add(
                row.checked_mul(row_bytes)
                    .ok_or_else(|| bad_shape(TENSOR_NAME, "source row offset overflows"))?,
            )
            .ok_or_else(|| bad_shape(TENSOR_NAME, "source range offset overflows"))?;
        let chunk_end = chunk_start
            .checked_add(chunk_bytes)
            .ok_or_else(|| bad_shape(TENSOR_NAME, "source range end overflows"))?;
        if chunk_end > end {
            return Err(bad_shape(
                TENSOR_NAME,
                "chunk exceeds the declared tensor range",
            ));
        }
        let bytes = source.read_range(chunk_start, chunk_end)?;
        if bytes.len() as u64 != chunk_bytes {
            return Err(crate::ranged_download::DownloadError::ShortRead {
                expected: chunk_bytes,
                actual: bytes.len() as u64,
            }
            .into());
        }
        sink.write_all(&bytes)
            .map_err(|e| io_error(Path::new(model_io::NGRAM_TABLE_BLOB), e))?;
        row += rows_this_chunk;

        let complete_bytes = row * row_bytes;
        let current_progress_gib = complete_bytes / PROGRESS_BYTES;
        if current_progress_gib > last_progress_gib {
            last_progress_gib = current_progress_gib;
            progress(&format!(
                "Qwen4Exp PLE table streamed: {complete_bytes} of {expected} bytes"
            ));
        }
    }
    Ok(())
}

fn bad_shape(tensor: &str, detail: impl Into<String>) -> GgufRepackError {
    GgufRepackError::ShapeMismatch {
        tensor: tensor.to_string(),
        detail: detail.into(),
    }
}

fn io_error(path: &Path, error: std::io::Error) -> GgufRepackError {
    GgufRepackError::Io {
        path: path.display().to_string(),
        detail: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    use super::*;
    use crate::gguf_header::{GgufTensorInfo, GgufValue};
    use crate::ranged_download::DownloadError;

    struct MemorySource {
        bytes: Vec<u8>,
        ranges: Mutex<Vec<(u64, u64)>>,
    }

    impl RangeSource for MemorySource {
        fn read_range(&self, start: u64, end: u64) -> Result<Vec<u8>, DownloadError> {
            self.ranges.lock().unwrap().push((start, end));
            let source_start = start;
            let source_end = end;
            let start = usize::try_from(source_start).map_err(|_| DownloadError::InvalidRange {
                start: source_start,
                end_exclusive: source_end,
            })?;
            let end = usize::try_from(source_end).map_err(|_| DownloadError::InvalidRange {
                start: source_start,
                end_exclusive: source_end,
            })?;
            self.bytes
                .get(start..end)
                .map(ToOwned::to_owned)
                .ok_or(DownloadError::ShortRead {
                    expected: (end - start) as u64,
                    actual: self.bytes.len().saturating_sub(start) as u64,
                })
        }
    }

    fn temp_install_dir() -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "turbospark-gguf-ple-{}-{nonce}",
            std::process::id()
        ))
    }

    fn metadata_array(values: &[u64]) -> GgufValue {
        GgufValue::Array(values.iter().copied().map(GgufValue::U64).collect())
    }

    fn fixture() -> (GgufHeader, ArchConfig, MemorySource) {
        let bytes = (0..72).map(|n| n as u8).collect::<Vec<_>>();
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "qwen4exp.ple.layer_multipliers".into(),
            metadata_array(&[17, 19]),
        );
        metadata.insert("qwen4exp.ple.head_vocab_sizes".into(), metadata_array(&[2]));
        metadata.insert("qwen4exp.ple.head_offsets".into(), metadata_array(&[0]));
        let mut tensors = BTreeMap::new();
        tensors.insert(
            TENSOR_NAME.into(),
            GgufTensorInfo {
                ggml_type: IQ4_NL_TYPE,
                dims: vec![32, 4],
                offset: 0,
            },
        );
        let header = GgufHeader {
            version: 3,
            metadata,
            tensors,
            alignment: 32,
            data_region_start: 0,
        };
        let mut arch = model_io::known_architecture(ModelFamily::Qwen4Exp);
        arch.ple.ngram_size = 2;
        arch.ple.heads_per_ngram = 1;
        arch.ple.ple_embed_dim = 32;
        arch.ple.split_ngram_parts = 2;
        arch.ple.layer_ids = vec![1];
        let source = MemorySource {
            bytes,
            ranges: Mutex::new(Vec::new()),
        };
        (header, arch, source)
    }

    #[test]
    fn writer_streams_iq4_nl_rows_and_emits_a_reader_validated_layout() {
        let (header, arch, source) = fixture();
        let dir = temp_install_dir();
        write_gguf_ngram_table(&dir, &header, &source, &arch, TENSOR_NAME, |_| {})
            .expect("write IQ4_NL PLE rows");

        let layout = model_io::load_ngram_table_layout(&dir)
            .expect("read layout")
            .expect("layout present");
        assert_eq!(layout.ggml_type.as_deref(), Some("iq4_nl"));
        assert_eq!(layout.rows, 4);
        assert_eq!(layout.rows_per_shard, 2);
        assert_eq!(layout.shards, 2);
        assert_eq!(layout.head_dim, 32);
        assert_eq!(layout.layer_index, 0);
        assert_eq!(layout.multipliers, vec![17, 19]);
        assert_eq!(layout.head_vocab_sizes, vec![2]);
        assert_eq!(layout.head_offsets, vec![0]);
        assert_eq!(
            std::fs::read(
                dir.join(model_io::NGRAM_TABLE_DIR)
                    .join(model_io::NGRAM_TABLE_BLOB)
            )
            .expect("read table blob"),
            source.bytes
        );
        assert_eq!(source.ranges.lock().unwrap().as_slice(), &[(0, 72)]);
        std::fs::remove_dir_all(dir).expect("remove temp install");
    }

    #[test]
    fn copy_reads_only_bounded_whole_row_ranges() {
        let source = MemorySource {
            bytes: (0..72).map(|n| n as u8).collect(),
            ranges: Mutex::new(Vec::new()),
        };
        let mut output = Vec::new();
        copy_tensor_rows(&source, 0..72, 4, 18, 2, &mut output, &mut |_| {})
            .expect("copy bounded rows");
        assert_eq!(output, source.bytes);
        assert_eq!(
            source.ranges.lock().unwrap().as_slice(),
            &[(0, 36), (36, 72)]
        );
    }
}
