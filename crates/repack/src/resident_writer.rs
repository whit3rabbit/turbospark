//! Builds a `model_weights.bin` resident-tensor index with real, named
//! entries, matching the on-disk format `turbospark_model_io::resident_index`
//! reads (24-byte header, 72-byte fixed entry table, then a string table,
//! then the raw tensor data region). `gturbo_writer::write_gturbo_install`
//! only ever writes an *empty* index (`entry_count == 0`); this module is
//! for callers that need real, addressable tensors — e.g. a small synthetic
//! model whose weights a real forward pass reads back by name.

const HEADER_BYTES: usize = 24;
const ENTRY_BYTES: usize = 72;

/// One named resident tensor: INT4-affine packed weight rows (see
/// `turbospark_compute::quant::Int4AffineRow`), concatenated across `rows`
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

/// INT8-affine dtype tag: same packed+scales+biases entry shape as INT4,
/// one byte per element instead of one nibble.
pub const DTYPE_INT8_AFFINE: u8 = 5;

/// Raw (unquantized, companion-less) dtype tags, matching the Swift
/// repacker's `IndexEntry` convention: 1 = BF16, 2 = FP16, 3 = FP32.
pub const DTYPE_BF16: u8 = 1;
pub const DTYPE_FP16: u8 = 2;
pub const DTYPE_FP32: u8 = 3;

/// GGUF block-quantized dtype tags, added by ROADMAP Phase G.
///
/// These get their own numbers rather than reusing [`DTYPE_INT8_AFFINE`]
/// and friends because the layouts are not interchangeable: an affine
/// tensor is three separate regions (packed weights, BF16 scales, BF16
/// biases) at group 64, while a GGUF block is self-contained, carrying its
/// own scale inline ahead of its weights. A reader that mistook a Q8_0
/// block for an INT8-affine row would decode the f16 scale as two weights
/// and be silently, plausibly wrong.
///
/// Entries carrying these tags have NO scale/bias companions: their
/// `scale_offset`/`scale_size`/`bias_offset`/`bias_size` are all zero.
///
/// A TAG HERE IS NOT PERMISSION TO RUN. The walk writes an install for every
/// block type it can parse, and whether that install opens is decided
/// separately, per type, by `model_io::EXECUTABLE_GGUF_TYPES` and by
/// `RealForwardRunner::open`'s copy of the same set. Q4_0 has a tag and no
/// kernel, and is refused by name.
pub const DTYPE_GGUF_Q8_0: u8 = 6;
pub const DTYPE_GGUF_Q4_K: u8 = 7;
pub const DTYPE_GGUF_Q6_K: u8 = 8;
pub const DTYPE_GGUF_Q4_0: u8 = 9;
/// The IQ-codebook tags (ROADMAP Phase S). Distinct from the K-quant ones
/// above for a reason beyond bookkeeping: these decode through a table that
/// ships with the format rather than through arithmetic, so a reader that
/// guessed at one of them cannot approximate it.
pub const DTYPE_GGUF_IQ3_XXS: u8 = 10;
pub const DTYPE_GGUF_IQ4_NL: u8 = 11;
pub const DTYPE_GGUF_IQ4_XS: u8 = 12;

/// Every GGUF block dtype tag, for consumers that need to reject the whole
/// family in one check rather than enumerate it and drift.
pub const GGUF_BLOCK_DTYPES: [u8; 7] = [
    DTYPE_GGUF_Q8_0,
    DTYPE_GGUF_Q4_K,
    DTYPE_GGUF_Q6_K,
    DTYPE_GGUF_Q4_0,
    DTYPE_GGUF_IQ3_XXS,
    DTYPE_GGUF_IQ4_NL,
    DTYPE_GGUF_IQ4_XS,
];

/// One named raw tensor (a norm vector, a scalar like `router.scale`):
/// bytes stored verbatim, no scale/bias companions, `dtype` one of
/// [`DTYPE_BF16`]/[`DTYPE_FP16`]/[`DTYPE_FP32`].
#[derive(Debug, Clone)]
pub struct RawTensorSpec {
    pub name: String,
    pub dtype: u8,
    pub bytes: Vec<u8>,
    /// Logical shape, rank padded to 4 with trailing zeros.
    pub shape: (u32, u32, u32, u32),
}

/// A resident entry: this port's INT4- or INT8-affine packed layout
/// (weight bytes + BF16 scales + BF16 biases) or a raw tensor.
#[derive(Debug, Clone)]
pub enum ResidentEntrySpec {
    Int4(ResidentTensorSpec),
    Int8(ResidentTensorSpec),
    Raw(RawTensorSpec),
}

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
    let views: Vec<EntryView<'_>> = specs
        .iter()
        .map(|t| EntryView::Packed(t, DTYPE_INT4_AFFINE))
        .collect();
    build_from_views(&views)
}

/// [`build_resident_weights_bin`] over a mix of INT4-affine and raw
/// (BF16/FP16/FP32) entries — what a real checkpoint repack produces:
/// pass-through quantized projections plus unquantized norms/scalars.
pub fn build_resident_weights_bin_mixed(specs: &[ResidentEntrySpec]) -> Vec<u8> {
    let views: Vec<EntryView<'_>> = specs
        .iter()
        .map(|s| match s {
            ResidentEntrySpec::Int4(t) => EntryView::Packed(t, DTYPE_INT4_AFFINE),
            ResidentEntrySpec::Int8(t) => EntryView::Packed(t, DTYPE_INT8_AFFINE),
            ResidentEntrySpec::Raw(r) => EntryView::Raw(r),
        })
        .collect();
    build_from_views(&views)
}

enum EntryView<'a> {
    Packed(&'a ResidentTensorSpec, u8),
    Raw(&'a RawTensorSpec),
}

impl EntryView<'_> {
    fn name(&self) -> &str {
        match self {
            EntryView::Packed(t, _) => &t.name,
            EntryView::Raw(r) => &r.name,
        }
    }
}

fn build_from_views(specs: &[EntryView<'_>]) -> Vec<u8> {
    let entry_count = specs.len();
    let entry_table_bytes = entry_count * ENTRY_BYTES;
    let string_table_start = HEADER_BYTES + entry_table_bytes;

    let mut name_ranges = Vec::with_capacity(entry_count);
    let mut names_len = 0usize;
    for spec in specs {
        name_ranges.push((string_table_start + names_len, spec.name().len()));
        names_len += spec.name().len();
    }
    let string_table_end = string_table_start + names_len;
    // Align the index region to the Swift repacker's Layout.pageBytes
    // (16 KiB, the Apple Silicon page size). This makes the resident data
    // region start page-aligned, which is what lets the runtime mmap it
    // and hand the mapping straight to Metal via newBufferWithBytesNoCopy
    // (page-aligned base required) with a zero slice shift.
    const PAGE_BYTES: usize = 16_384;
    let index_size = string_table_end.div_ceil(PAGE_BYTES) * PAGE_BYTES;

    let mut data = Vec::new();
    struct Placed {
        dtype: u8,
        weight_offset: u64,
        weight_size: u64,
        shape: (u32, u32, u32, u32),
        scale_offset: u64,
        scale_size: u64,
        bias_offset: u64,
        bias_size: u64,
    }
    let mut placed = Vec::with_capacity(entry_count);
    for spec in specs {
        // 4-byte-align every entry's start: packed u32 weights are read
        // with 4-byte loads by some kernels, and BF16 entries can leave
        // the cursor 2 mod 4.
        while data.len() % 4 != 0 {
            data.push(0);
        }
        match spec {
            EntryView::Packed(t, dtype) => {
                let weight_offset = index_size as u64 + data.len() as u64;
                data.extend_from_slice(&t.packed);
                let scale_bytes = u16_slice_to_le_bytes(&t.scales);
                let scale_offset = index_size as u64 + data.len() as u64;
                data.extend_from_slice(&scale_bytes);
                let bias_bytes = u16_slice_to_le_bytes(&t.biases);
                let bias_offset = index_size as u64 + data.len() as u64;
                data.extend_from_slice(&bias_bytes);
                placed.push(Placed {
                    dtype: *dtype,
                    weight_offset,
                    weight_size: t.packed.len() as u64,
                    shape: (t.rows, t.cols, 0, 0),
                    scale_offset,
                    scale_size: scale_bytes.len() as u64,
                    bias_offset,
                    bias_size: bias_bytes.len() as u64,
                });
            }
            EntryView::Raw(r) => {
                let weight_offset = index_size as u64 + data.len() as u64;
                data.extend_from_slice(&r.bytes);
                placed.push(Placed {
                    dtype: r.dtype,
                    weight_offset,
                    weight_size: r.bytes.len() as u64,
                    shape: r.shape,
                    scale_offset: 0,
                    scale_size: 0,
                    bias_offset: 0,
                    bias_size: 0,
                });
            }
        }
    }

    let mut out = vec![0u8; index_size];
    out[0..8].copy_from_slice(&(index_size as u64).to_le_bytes());
    out[8..16].copy_from_slice(&(data.len() as u64).to_le_bytes());
    out[16..24].copy_from_slice(&(entry_count as u64).to_le_bytes());

    for (i, p) in placed.iter().enumerate() {
        let base = HEADER_BYTES + i * ENTRY_BYTES;
        let (name_offset, name_len) = name_ranges[i];
        out[base..base + 4].copy_from_slice(&(name_offset as u32).to_le_bytes());
        out[base + 4..base + 6].copy_from_slice(&(name_len as u16).to_le_bytes());
        out[base + 6] = p.dtype;
        out[base + 7] = 0;
        out[base + 8..base + 16].copy_from_slice(&p.weight_offset.to_le_bytes());
        out[base + 16..base + 24].copy_from_slice(&p.weight_size.to_le_bytes());
        out[base + 24..base + 28].copy_from_slice(&p.shape.0.to_le_bytes());
        out[base + 28..base + 32].copy_from_slice(&p.shape.1.to_le_bytes());
        out[base + 32..base + 36].copy_from_slice(&p.shape.2.to_le_bytes());
        out[base + 36..base + 40].copy_from_slice(&p.shape.3.to_le_bytes());
        out[base + 40..base + 48].copy_from_slice(&p.scale_offset.to_le_bytes());
        out[base + 48..base + 56].copy_from_slice(&p.scale_size.to_le_bytes());
        out[base + 56..base + 64].copy_from_slice(&p.bias_offset.to_le_bytes());
        out[base + 64..base + 72].copy_from_slice(&p.bias_size.to_le_bytes());
    }
    for (i, spec) in specs.iter().enumerate() {
        let (name_offset, name_len) = name_ranges[i];
        out[name_offset..name_offset + name_len].copy_from_slice(spec.name().as_bytes());
    }

    out.extend_from_slice(&data);
    out
}
