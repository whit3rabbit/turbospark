//! `model_weights.bin`'s leading index region: a fixed header followed by a
//! fixed-width entry table, a string table, and (on installs written since
//! manifest 1.1) a format trailer in the page padding. Ported from
//! `Infrastructure/ModelIO/ResidentIndex.swift`.
//!
//! The trailer is the last [`TRAILER_BYTES`] of the index region: the
//! [`INDEX_MAGIC`] bytes, a `u32` format version, and a reserved `u32`. It
//! sits in padding that the Swift reader and older turbospark builds skip,
//! so they keep opening new installs, while this reader refuses a format
//! version it does not know by name instead of misreading the bytes. A file
//! with no trailer is the legacy layout and is accepted as it always was.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use crate::error::ModelError;

/// Byte size of the resident index fixed header (24 bytes).
pub const HEADER_BYTES: usize = 24;
/// Byte size of each resident index table entry (72 bytes).
pub const ENTRY_BYTES: usize = 72;
/// Byte size of the format trailer at the end of the index region.
pub const TRAILER_BYTES: usize = 16;
/// Magic bytes opening the format trailer.
pub const INDEX_MAGIC: [u8; 8] = *b"GTURBIDX";
/// The index layout this reader understands. The writer stamps it into the
/// trailer and a different value is refused with
/// [`ModelError::UnsupportedIndexFormat`].
pub const INDEX_FORMAT_VERSION: u32 = 1;

/// `indexSize` is the full byte size of the leading index region: it
/// INCLUDES the header itself, the entry table, the string table, and the
/// writer's page padding. The resident tensor region starts at file byte
/// `index_size`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentIndexHeader {
    /// Full byte size of the index region.
    pub index_size: u64,
    /// Total bytes in the resident weights region.
    pub resident_size: u64,
    /// Number of tensor entries in the index table.
    pub entry_count: u64,
}

/// One named resident tensor entry in the index table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentIndexEntry {
    /// Tensor lookup name.
    pub name: String,
    /// Data type tag byte.
    pub dtype: u8,
    /// Absolute file offset of the packed weight bytes (>= `index_size`).
    pub file_offset: u64,
    /// Weight payload size in bytes.
    pub size_bytes: u64,
    /// Logical tensor shape (rank padded to 4).
    pub shape: (u32, u32, u32, u32),
    /// File offset for scale factors.
    pub scale_offset: u64,
    /// Byte size of scale factors.
    pub scale_size: u64,
    /// File offset for bias values.
    pub bias_offset: u64,
    /// Byte size of bias values.
    pub bias_size: u64,
}

/// Complete resident index including header and named tensor entries map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentIndex {
    /// Index header.
    pub header: ResidentIndexHeader,
    /// Map of tensor names to index entries.
    pub entries: HashMap<String, ResidentIndexEntry>,
}

fn corrupt(detail: impl Into<String>) -> ModelError {
    ModelError::IndexCorrupt {
        detail: detail.into(),
    }
}

fn validate_fixed_affine_entry(entry: &ResidentIndexEntry) -> Result<(), ModelError> {
    let bits = match entry.dtype {
        4 => 4u64,
        5 => 8u64,
        _ => return Ok(()),
    };
    let rows = u64::from(entry.shape.0);
    let cols = u64::from(entry.shape.1);
    if rows == 0 || cols == 0 || cols % 64 != 0 {
        return Err(corrupt(format!(
            "tensor {} has invalid affine shape {}x{}",
            entry.name, rows, cols
        )));
    }
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| corrupt(format!("tensor {} shape overflows", entry.name)))?;
    let packed = elements
        .checked_mul(bits)
        .and_then(|value| value.checked_div(8))
        .ok_or_else(|| corrupt(format!("tensor {} packed size overflows", entry.name)))?;
    let companions = rows
        .checked_mul(cols / 64)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| corrupt(format!("tensor {} companion size overflows", entry.name)))?;
    if entry.size_bytes != packed || entry.scale_size != companions || entry.bias_size != companions
    {
        return Err(corrupt(format!(
            "tensor {} affine planes are {}, {}, and {} bytes; shape {}x{} requires {}, {}, and {}",
            entry.name,
            entry.size_bytes,
            entry.scale_size,
            entry.bias_size,
            rows,
            cols,
            packed,
            companions,
            companions
        )));
    }
    if entry.scale_offset % 2 != 0 || entry.bias_offset % 2 != 0 {
        return Err(corrupt(format!(
            "tensor {} affine companion offsets are not 2-byte aligned",
            entry.name
        )));
    }
    Ok(())
}

/// Read the header + index region out of `model_weights.bin`. The tensor
/// data region (starting at byte `header.index_size`) is not read here.
pub fn load(file_path: &Path) -> Result<ResidentIndex, ModelError> {
    let mut file = std::fs::File::open(file_path).map_err(|e| ModelError::IoFailed {
        call: "open".to_string(),
        detail: e.to_string(),
    })?;
    let file_len = file
        .metadata()
        .map_err(|e| ModelError::IoFailed {
            call: "stat".to_string(),
            detail: e.to_string(),
        })?
        .len();

    let mut header_buf = [0u8; HEADER_BYTES];
    file.read_exact(&mut header_buf)
        .map_err(|_| corrupt("short read for IndexHeader"))?;
    let header = ResidentIndexHeader {
        index_size: u64::from_le_bytes(header_buf[0..8].try_into().unwrap()),
        resident_size: u64::from_le_bytes(header_buf[8..16].try_into().unwrap()),
        entry_count: u64::from_le_bytes(header_buf[16..24].try_into().unwrap()),
    };

    if header.index_size < HEADER_BYTES as u64 {
        return Err(corrupt(format!(
            "indexSize {} < header size {HEADER_BYTES}",
            header.index_size
        )));
    }
    let entry_table_bytes = header
        .entry_count
        .checked_mul(ENTRY_BYTES as u64)
        .ok_or_else(|| {
            corrupt(format!(
                "entryCount {} overflows the entry table",
                header.entry_count
            ))
        })?;
    let expected_entry_table_end = (HEADER_BYTES as u64)
        .checked_add(entry_table_bytes)
        .ok_or_else(|| corrupt("header + entry table size overflows"))?;
    if expected_entry_table_end > header.index_size {
        return Err(corrupt(format!(
            "header+entries ({expected_entry_table_end}) > indexSize {}",
            header.index_size
        )));
    }
    // Bounds every offset an entry can declare, and (since `index_size` is
    // what sizes the allocation below) caps that allocation at the file's own
    // length rather than at whatever a corrupt header claims.
    let region_end = header
        .index_size
        .checked_add(header.resident_size)
        .ok_or_else(|| corrupt("indexSize + residentSize overflows"))?;
    if region_end > file_len {
        return Err(corrupt(format!(
            "indexSize {} + residentSize {} = {region_end} exceeds file length {file_len}",
            header.index_size, header.resident_size
        )));
    }

    // Checked so a 32-bit target refuses a >4 GiB index instead of truncating
    // it. entry_count is bounded by index_size above, so its casts below fit.
    let region_len = usize::try_from(header.index_size)
        .map_err(|_| corrupt("indexSize exceeds the address space"))?;
    let mut index_buf = vec![0u8; region_len];
    {
        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(0))
            .map_err(|e| ModelError::IoFailed {
                call: "seek".to_string(),
                detail: e.to_string(),
            })?;
    }
    file.read_exact(&mut index_buf)
        .map_err(|_| corrupt("short read for index region"))?;

    let mut entries = HashMap::with_capacity(header.entry_count as usize);
    // The trailer is honored only past the string table, so a legacy name
    // that happens to end in the magic bytes is not mistaken for one.
    let mut string_table_end = HEADER_BYTES + header.entry_count as usize * ENTRY_BYTES;
    for i in 0..header.entry_count as usize {
        let base = HEADER_BYTES + i * ENTRY_BYTES;
        let p = &index_buf[base..base + ENTRY_BYTES];
        let name_offset = u32::from_le_bytes(p[0..4].try_into().unwrap()) as usize;
        let name_length = u16::from_le_bytes(p[4..6].try_into().unwrap()) as usize;
        let dtype = p[6];
        // byte 7 reserved
        let file_offset = u64::from_le_bytes(p[8..16].try_into().unwrap());
        let size_bytes = u64::from_le_bytes(p[16..24].try_into().unwrap());
        let s0 = u32::from_le_bytes(p[24..28].try_into().unwrap());
        let s1 = u32::from_le_bytes(p[28..32].try_into().unwrap());
        let s2 = u32::from_le_bytes(p[32..36].try_into().unwrap());
        let s3 = u32::from_le_bytes(p[36..40].try_into().unwrap());
        let scale_offset = u64::from_le_bytes(p[40..48].try_into().unwrap());
        let scale_size = u64::from_le_bytes(p[48..56].try_into().unwrap());
        let bias_offset = u64::from_le_bytes(p[56..64].try_into().unwrap());
        let bias_size = u64::from_le_bytes(p[64..72].try_into().unwrap());

        if name_offset < HEADER_BYTES || name_offset + name_length > region_len {
            return Err(corrupt(format!(
                "entry {i} name range [{name_offset}, {}) out of index region [{HEADER_BYTES}, {region_len})",
                name_offset + name_length
            )));
        }
        let name = String::from_utf8_lossy(&index_buf[name_offset..name_offset + name_length])
            .into_owned();
        string_table_end = string_table_end.max(name_offset + name_length);

        // Every offset/size pair below is untrusted input: a corrupt or
        // hostile index can name any `u64`, and every consumer (host slices,
        // `ResidentBuffer::map`, the GPU's `gpu_offset`) subtracts
        // `index_size` and slices with no bound of its own -- this is the
        // one place that CAN check, and `runtime::real_forward_utils`'s
        // `resident_matrix` doc already claims it does.
        let in_region = |offset: u64, size: u64| -> bool {
            size == 0
                || (offset >= header.index_size
                    && offset
                        .checked_add(size)
                        .is_some_and(|end| end <= region_end))
        };
        if !in_region(file_offset, size_bytes) {
            return Err(corrupt(format!(
                "entry {i} ({name}) payload [{file_offset}, +{size_bytes}) outside the resident \
                 region [{}, {region_end})",
                header.index_size
            )));
        }
        if !in_region(scale_offset, scale_size) {
            return Err(corrupt(format!(
                "entry {i} ({name}) scale plane [{scale_offset}, +{scale_size}) outside the \
                 resident region [{}, {region_end})",
                header.index_size
            )));
        }
        if !in_region(bias_offset, bias_size) {
            return Err(corrupt(format!(
                "entry {i} ({name}) bias plane [{bias_offset}, +{bias_size}) outside the \
                 resident region [{}, {region_end})",
                header.index_size
            )));
        }

        let entry = ResidentIndexEntry {
            name: name.clone(),
            dtype,
            file_offset,
            size_bytes,
            shape: (s0, s1, s2, s3),
            scale_offset,
            scale_size,
            bias_offset,
            bias_size,
        };
        validate_fixed_affine_entry(&entry)?;
        if entries.contains_key(&name) {
            return Err(corrupt(format!("duplicate tensor name {name}")));
        }
        entries.insert(name, entry);
    }

    check_trailer(&index_buf, string_table_end)?;

    Ok(ResidentIndex { header, entries })
}

/// Refuse an index region whose trailer names a format this reader does not
/// implement. A region with no trailer past its string table is the legacy
/// layout and passes.
fn check_trailer(index_buf: &[u8], string_table_end: usize) -> Result<(), ModelError> {
    let Some(trailer_start) = index_buf.len().checked_sub(TRAILER_BYTES) else {
        return Ok(());
    };
    if trailer_start < string_table_end {
        return Ok(());
    }
    let trailer = &index_buf[trailer_start..];
    if trailer[..8] != INDEX_MAGIC {
        return Ok(());
    }
    let found = u32::from_le_bytes(trailer[8..12].try_into().unwrap());
    if found != INDEX_FORMAT_VERSION {
        return Err(ModelError::UnsupportedIndexFormat {
            found,
            supported: INDEX_FORMAT_VERSION,
        });
    }
    Ok(())
}
