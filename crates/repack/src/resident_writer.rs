//! Builds a `model_weights.bin` resident-tensor index with real, named
//! entries, matching the on-disk format `mrefrust_model_io::resident_index`
//! reads (24-byte header, 72-byte fixed entry table, then a string table,
//! then the raw tensor data region). `gturbo_writer::write_gturbo_install`
//! only ever writes an *empty* index (`entry_count == 0`); this module is
//! for callers that need real, addressable tensors — e.g. a small synthetic
//! model whose weights a real forward pass reads back by name.

const HEADER_BYTES: usize = 24;
const ENTRY_BYTES: usize = 72;

/// One named resident tensor: INT4-affine packed weight rows (see
/// `mrefrust_compute::quant::Int4AffineRow`), concatenated across `rows`
/// rows of `cols` elements each.
#[derive(Debug, Clone)]
pub struct ResidentTensorSpec {
    pub name: String,
    pub packed: Vec<u8>,
    pub scales: Vec<u16>,
    pub biases: Vec<u16>,
    pub rows: u32,
    pub cols: u32,
}

/// INT4-affine dtype tag stored in each entry's `dtype` byte. Not validated
/// by the reader; this port's only consumer of the tag is its own writer
/// and `crates/runtime`'s `RealForwardRunner`.
const DTYPE_INT4_AFFINE: u8 = 4;

fn u16_slice_to_le_bytes(values: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Assembles a complete `model_weights.bin`: header, entry table, string
/// table, then the raw tensor data region (packed bytes, then scale bytes,
/// then bias bytes, back to back per tensor, in `specs` order).
pub fn build_resident_weights_bin(specs: &[ResidentTensorSpec]) -> Vec<u8> {
    let entry_count = specs.len();
    let entry_table_bytes = entry_count * ENTRY_BYTES;
    let string_table_start = HEADER_BYTES + entry_table_bytes;

    let mut name_ranges = Vec::with_capacity(entry_count);
    let mut names_len = 0usize;
    for spec in specs {
        name_ranges.push((string_table_start + names_len, spec.name.len()));
        names_len += spec.name.len();
    }
    let string_table_end = string_table_start + names_len;
    // Page-align the index region (not required by the reader, but matches
    // the page-aligned convention the rest of this format uses).
    let index_size = string_table_end.div_ceil(4096) * 4096;

    let mut data = Vec::new();
    struct Placed {
        packed_offset: u64,
        packed_size: u64,
        scale_offset: u64,
        scale_size: u64,
        bias_offset: u64,
        bias_size: u64,
    }
    let mut placed = Vec::with_capacity(entry_count);
    for spec in specs {
        let packed_offset = index_size as u64 + data.len() as u64;
        data.extend_from_slice(&spec.packed);
        let scale_bytes = u16_slice_to_le_bytes(&spec.scales);
        let scale_offset = index_size as u64 + data.len() as u64;
        data.extend_from_slice(&scale_bytes);
        let bias_bytes = u16_slice_to_le_bytes(&spec.biases);
        let bias_offset = index_size as u64 + data.len() as u64;
        data.extend_from_slice(&bias_bytes);
        placed.push(Placed {
            packed_offset,
            packed_size: spec.packed.len() as u64,
            scale_offset,
            scale_size: scale_bytes.len() as u64,
            bias_offset,
            bias_size: bias_bytes.len() as u64,
        });
    }

    let mut out = vec![0u8; index_size];
    out[0..8].copy_from_slice(&(index_size as u64).to_le_bytes());
    out[8..16].copy_from_slice(&(data.len() as u64).to_le_bytes());
    out[16..24].copy_from_slice(&(entry_count as u64).to_le_bytes());

    for (i, spec) in specs.iter().enumerate() {
        let base = HEADER_BYTES + i * ENTRY_BYTES;
        let (name_offset, name_len) = name_ranges[i];
        let p = &placed[i];
        out[base..base + 4].copy_from_slice(&(name_offset as u32).to_le_bytes());
        out[base + 4..base + 6].copy_from_slice(&(name_len as u16).to_le_bytes());
        out[base + 6] = DTYPE_INT4_AFFINE;
        out[base + 7] = 0;
        out[base + 8..base + 16].copy_from_slice(&p.packed_offset.to_le_bytes());
        out[base + 16..base + 24].copy_from_slice(&p.packed_size.to_le_bytes());
        out[base + 24..base + 28].copy_from_slice(&spec.rows.to_le_bytes());
        out[base + 28..base + 32].copy_from_slice(&spec.cols.to_le_bytes());
        out[base + 32..base + 36].copy_from_slice(&0u32.to_le_bytes());
        out[base + 36..base + 40].copy_from_slice(&0u32.to_le_bytes());
        out[base + 40..base + 48].copy_from_slice(&p.scale_offset.to_le_bytes());
        out[base + 48..base + 56].copy_from_slice(&p.scale_size.to_le_bytes());
        out[base + 56..base + 64].copy_from_slice(&p.bias_offset.to_le_bytes());
        out[base + 64..base + 72].copy_from_slice(&p.bias_size.to_le_bytes());
    }
    for (i, spec) in specs.iter().enumerate() {
        let (name_offset, name_len) = name_ranges[i];
        out[name_offset..name_offset + name_len].copy_from_slice(spec.name.as_bytes());
    }

    out.extend_from_slice(&data);
    out
}
