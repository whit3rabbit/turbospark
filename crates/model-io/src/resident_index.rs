//! `model_weights.bin`'s leading index region: a fixed header followed by a
//! fixed-width entry table and a string table. Ported from
//! `Infrastructure/ModelIO/ResidentIndex.swift`.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use crate::error::ModelError;

/// Byte size of the resident index fixed header (24 bytes).
pub const HEADER_BYTES: usize = 24;
/// Byte size of each resident index table entry (72 bytes).
pub const ENTRY_BYTES: usize = 72;

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

    let region_len = header.index_size as usize;
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
        if entries.contains_key(&name) {
            return Err(corrupt(format!("duplicate tensor name {name}")));
        }
        entries.insert(name, entry);
    }

    Ok(ResidentIndex { header, entries })
}
