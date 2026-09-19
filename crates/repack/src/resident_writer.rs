//! Builds a `model_weights.bin` resident-tensor index with real, named
//! entries, matching the on-disk format `turbospark_model_io::resident_index`
//! reads (24-byte header, 72-byte fixed entry table, then a string table,
//! then the raw tensor data region). `gturbo_writer::write_gturbo_install`
//! only ever writes an *empty* index (`entry_count == 0`); this module is
//! for callers that need real, addressable tensors — e.g. a small synthetic
//! model whose weights a real forward pass reads back by name.

use std::io::{self, Write};

const HEADER_BYTES: usize = 24;
const ENTRY_BYTES: usize = 72;

/// One named resident tensor: INT4-affine packed weight rows (see
/// `turbospark_compute::quant::Int4AffineRow`), concatenated across `rows`
/// rows of `cols` elements each.
#[derive(Debug, Clone)]
pub struct ResidentTensorSpec {
    /// Tensor lookup name string.
    pub name: String,
    /// Raw packed weight byte payload.
    pub packed: Vec<u8>,
    /// Scale values per group.
    pub scales: Vec<u16>,
    /// Bias values per group.
    pub biases: Vec<u16>,
    /// Matrix row count.
    pub rows: u32,
    /// Matrix column count.
    pub cols: u32,
}

/// INT4-affine dtype tag stored in each entry's `dtype` byte. Not validated
/// by `crates/model-io`'s reader; this port's consumers of the tag are this
/// module's own writer, `resident_reader.rs` (reading an install back for
/// the MTP graft), and `crates/runtime`'s `RealForwardRunner`. `pub` for
/// that second reason: the value is part of the ON-DISK FORMAT, and a
/// private restatement of it in the reader is the one copy that can drift.
pub const DTYPE_INT4_AFFINE: u8 = 4;

/// INT8-affine dtype tag: same packed+scales+biases entry shape as INT4,
/// one byte per element instead of one nibble.
pub const DTYPE_INT8_AFFINE: u8 = 5;

/// 1-bit-affine dtype tag (ROADMAP's 1-bit entry): the same three-region
/// entry shape again, one BIT per element.
///
/// **The number is 15 rather than something next to its two affine siblings,
/// and the reason is worth stating: 6..=14 were taken by the GGUF block tags
/// before this existed.** It is NOT a GGUF block dtype and must never join
/// [`GGUF_BLOCK_DTYPES`] -- a reader that treated it as one would look for a
/// scale inside the weight run, where a 1-bit affine tensor keeps its scales
/// in a separate plane like INT4 does.
///
/// Two things differ from [`DTYPE_INT4_AFFINE`] beyond the width, and both
/// are invisible to any length check, which is why the tag has to be
/// distinct rather than INT4 with a bit count beside it: the companions are
/// FP16 rather than BF16, and the group is 128 rather than 64.
pub const DTYPE_INT1_AFFINE: u8 = 15;

/// 2-bit-affine dtype tag (ROADMAP's ternary entry): the same three-region
/// entry shape again, two BITS per element.
///
/// 16 for [`DTYPE_INT1_AFFINE`]'s reason -- it is simply the next free number,
/// 6..=14 being the GGUF block tags -- and it is NOT a GGUF block dtype, so it
/// must never join [`GGUF_BLOCK_DTYPES`] either.
///
/// A distinct tag rather than INT1 with a width beside it, because the
/// resident index records no width: the packed run's LENGTH is what says
/// whether a row is one bit or two, and reading a 2-bit tensor as 1-bit finds
/// a row of exactly half the columns rather than an error. The companion
/// dtype (FP16) and group size (128) happen to agree with the 1-bit tag's,
/// which is a fact about one publisher's two checkpoints rather than about the
/// widths.
pub const DTYPE_INT2_AFFINE: u8 = 16;

/// Raw (unquantized, companion-less) dtype tags, matching the Swift
/// repacker's `IndexEntry` convention: 1 = BF16, 2 = FP16, 3 = FP32.
pub const DTYPE_BF16: u8 = 1;
/// FP16 raw unquantized dtype tag.
pub const DTYPE_FP16: u8 = 2;
/// FP32 raw unquantized dtype tag.
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
/// GGUF Q4_K block dtype tag.
pub const DTYPE_GGUF_Q4_K: u8 = 7;
/// GGUF Q6_K block dtype tag.
pub const DTYPE_GGUF_Q6_K: u8 = 8;
/// GGUF Q4_0 block dtype tag.
pub const DTYPE_GGUF_Q4_0: u8 = 9;
/// GGUF IQ3_XXS block dtype tag (ROADMAP Phase S). Distinct from the K-quant ones
/// above for a reason beyond bookkeeping: these decode through a table that
/// ships with the format rather than through arithmetic, so a reader that
/// guessed at one of them cannot approximate it.
pub const DTYPE_GGUF_IQ3_XXS: u8 = 10;
/// GGUF IQ4_NL block dtype tag.
pub const DTYPE_GGUF_IQ4_NL: u8 = 11;
/// GGUF IQ4_XS block dtype tag.
pub const DTYPE_GGUF_IQ4_XS: u8 = 12;
/// GGUF Q5_K block dtype tag (ROADMAP Phase M2, for Mixtral's `attn_output`).
/// Tag 13 rather than the next free number on purpose: it matches ggml's own
/// type id for Q5_K, which the three K-quant tags above predate and do not.
pub const DTYPE_GGUF_Q5_K: u8 = 13;
/// GGUF Q2_K block dtype tag.
pub const DTYPE_GGUF_Q2_K: u8 = 17;
/// GGUF IQ2_XXS block dtype tag.
pub const DTYPE_GGUF_IQ2_XXS: u8 = 18;
/// GGUF IQ2_XS block dtype tag.
pub const DTYPE_GGUF_IQ2_XS: u8 = 19;
/// GGUF IQ1_S block dtype tag.
pub const DTYPE_GGUF_IQ1_S: u8 = 20;
/// GGUF IQ3_S block dtype tag.
pub const DTYPE_GGUF_IQ3_S: u8 = 21;
/// GGUF IQ2_S block dtype tag.
pub const DTYPE_GGUF_IQ2_S: u8 = 22;
/// GGUF IQ1_M block dtype tag.
pub const DTYPE_GGUF_IQ1_M: u8 = 23;
/// GGUF Q3_K block dtype tag (Dense Qwen2 roadmap item). Tag 24 rather than
/// ggml's own type id 11 -- unlike Q5_K's deliberate match -- because 11 was
/// taken by IQ4_NL before this existed; the K-quant tags that match their
/// ggml ids predate the IQ ones that displaced them. Kernels: a resident
/// GEMV and an embedding lookup, no routed pair (the pinned Qwen2.5 Q3_K_M
/// keeps its experts -- there are none, it is dense -- and every routed slot
/// would be a new file's problem, on Q6_K's footing).
pub const DTYPE_GGUF_Q3_K: u8 = 24;

/// Every GGUF block dtype tag, for consumers that need to reject the whole
/// family in one check rather than enumerate it and drift.
pub const GGUF_BLOCK_DTYPES: [u8; 16] = [
    DTYPE_GGUF_Q8_0,
    DTYPE_GGUF_Q4_K,
    DTYPE_GGUF_Q6_K,
    DTYPE_GGUF_Q4_0,
    DTYPE_GGUF_IQ3_XXS,
    DTYPE_GGUF_IQ4_NL,
    DTYPE_GGUF_IQ4_XS,
    DTYPE_GGUF_Q5_K,
    DTYPE_GGUF_Q2_K,
    DTYPE_GGUF_IQ2_XXS,
    DTYPE_GGUF_IQ2_XS,
    DTYPE_GGUF_IQ1_S,
    DTYPE_GGUF_IQ3_S,
    DTYPE_GGUF_IQ2_S,
    DTYPE_GGUF_IQ1_M,
    DTYPE_GGUF_Q3_K,
];

/// One named raw tensor (a norm vector, a scalar like `router.scale`):
/// bytes stored verbatim, no scale/bias companions, `dtype` one of
/// [`DTYPE_BF16`]/[`DTYPE_FP16`]/[`DTYPE_FP32`].
#[derive(Debug, Clone)]
pub struct RawTensorSpec {
    /// Tensor lookup name string.
    pub name: String,
    /// Raw data type byte tag (`DTYPE_BF16`, `DTYPE_FP16`, `DTYPE_FP32`).
    pub dtype: u8,
    /// Unquantized raw byte payload.
    pub bytes: Vec<u8>,
    /// Logical shape, rank padded to 4 with trailing zeros.
    pub shape: (u32, u32, u32, u32),
}

/// A resident entry: this port's INT4- or INT8-affine packed layout
/// (weight bytes + BF16 scales + BF16 biases) or a raw tensor.
#[derive(Debug, Clone)]
pub enum ResidentEntrySpec {
    /// INT4-affine quantized tensor spec.
    Int4(ResidentTensorSpec),
    /// INT8-affine quantized tensor spec.
    Int8(ResidentTensorSpec),
    /// 1-bit-affine quantized tensor spec (ROADMAP's 1-bit entry). Same three
    /// regions as the two above; its companions are FP16 and its group is
    /// 128, neither of which this type carries -- see [`DTYPE_INT1_AFFINE`].
    Int1(ResidentTensorSpec),
    /// 2-bit-affine quantized tensor spec (ROADMAP's ternary entry). Same
    /// three regions again; like [`Self::Int1`] its companions are FP16 and
    /// its group is 128 -- see [`DTYPE_INT2_AFFINE`].
    Int2(ResidentTensorSpec),
    /// Unquantized raw tensor spec.
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
    let mut out = Vec::new();
    write_from_views(&views, &mut out).expect("writing into a Vec<u8> cannot fail");
    out
}

/// [`build_resident_weights_bin`] over a mix of INT4-affine and raw
/// (BF16/FP16/FP32) entries — what a real checkpoint repack produces:
/// pass-through quantized projections plus unquantized norms/scalars.
pub fn build_resident_weights_bin_mixed(specs: &[ResidentEntrySpec]) -> Vec<u8> {
    let mut out = Vec::new();
    write_resident_weights_bin_mixed(specs, &mut out).expect("writing into a Vec<u8> cannot fail");
    out
}

/// Streaming counterpart to [`build_resident_weights_bin_mixed`]: writes the
/// SAME bytes directly to `out` rather than assembling the whole resident
/// region in memory first.
///
/// `build_resident_weights_bin_mixed` is now a thin wrapper over this
/// (writing into a `Vec<u8>`), so the two are byte-identical by
/// CONSTRUCTION rather than by a separate test asserting they agree: there
/// is only one implementation. On a real streamed install this is what lets
/// `StreamingGturboWriter::finish_streaming` write the resident region
/// straight to `model_weights.bin` without ever holding a second, then a
/// third, full copy of it in memory (the caller's `specs` -- built while
/// reading and quantizing the checkpoint -- is the one copy this function
/// does not eliminate).
pub fn write_resident_weights_bin_mixed(
    specs: &[ResidentEntrySpec],
    out: &mut dyn Write,
) -> io::Result<()> {
    let views: Vec<EntryView<'_>> = specs
        .iter()
        .map(|s| match s {
            ResidentEntrySpec::Int4(t) => EntryView::Packed(t, DTYPE_INT4_AFFINE),
            ResidentEntrySpec::Int8(t) => EntryView::Packed(t, DTYPE_INT8_AFFINE),
            ResidentEntrySpec::Int1(t) => EntryView::Packed(t, DTYPE_INT1_AFFINE),
            ResidentEntrySpec::Int2(t) => EntryView::Packed(t, DTYPE_INT2_AFFINE),
            ResidentEntrySpec::Raw(r) => EntryView::Raw(r),
        })
        .collect();
    write_from_views(&views, out)
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

/// Writes `n` zero bytes to `out` without allocating an `n`-byte buffer.
fn write_zeros(out: &mut dyn Write, n: usize) -> io::Result<()> {
    const ZEROS: [u8; 64] = [0u8; 64];
    let mut remaining = n;
    while remaining > 0 {
        let chunk = remaining.min(ZEROS.len());
        out.write_all(&ZEROS[..chunk])?;
        remaining -= chunk;
    }
    Ok(())
}

/// One entry's placement in the data region, computed from LENGTHS alone
/// (never from the bytes themselves) so the layout is known before a single
/// byte of tensor data is written.
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

/// Writes header, entry table, string table, then the data region, to
/// `out`, in two passes over `specs`.
///
/// **PASS 1 computes every entry's placement from `.len()` calls alone --
/// never copying or even reading a tensor's bytes -- because the header and
/// entry table have to be written before the data region they describe, and
/// this port's on-disk format is not self-describing enough to write them
/// any other way.** That is the one unavoidable pre-pass; nothing here
/// holds a second copy of the DATA to produce it.
///
/// **PASS 2 writes the real bytes straight to `out`, padding to each
/// entry's OWN `weight_offset` from pass 1 rather than recomputing the
/// alignment rule a second time** -- so the two passes cannot silently
/// disagree about where a tensor lands; there is one source of truth for
/// the layout and pass 2 is only ever catching up to it.
///
/// This is what lets [`build_resident_weights_bin_mixed`] and a real
/// streamed install ([`StreamingGturboWriter::finish_streaming`], in
/// `gturbo_writer/streaming.rs`) share one implementation: the in-memory
/// caller passes a `Vec<u8>` as `out` and the streamed caller passes an open
/// file, and neither this function nor its caller ever builds a second,
/// full-sized copy of the resident region to get there.
fn write_from_views(specs: &[EntryView<'_>], out: &mut dyn Write) -> io::Result<()> {
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

    // PASS 1: placement, from lengths alone.
    let mut placed = Vec::with_capacity(entry_count);
    let mut cursor = 0u64;
    for spec in specs {
        // 4-byte-align every entry's start: packed u32 weights are read
        // with 4-byte loads by some kernels, and BF16 entries can leave
        // the cursor 2 mod 4.
        cursor += (4 - (cursor % 4)) % 4;
        match spec {
            EntryView::Packed(t, dtype) => {
                let weight_offset = index_size as u64 + cursor;
                cursor += t.packed.len() as u64;
                let scale_size = (t.scales.len() * 2) as u64;
                let scale_offset = index_size as u64 + cursor;
                cursor += scale_size;
                let bias_size = (t.biases.len() * 2) as u64;
                let bias_offset = index_size as u64 + cursor;
                cursor += bias_size;
                placed.push(Placed {
                    dtype: *dtype,
                    weight_offset,
                    weight_size: t.packed.len() as u64,
                    shape: (t.rows, t.cols, 0, 0),
                    scale_offset,
                    scale_size,
                    bias_offset,
                    bias_size,
                });
            }
            EntryView::Raw(r) => {
                let weight_offset = index_size as u64 + cursor;
                cursor += r.bytes.len() as u64;
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
    let data_len = cursor;

    // Header.
    let mut header = [0u8; HEADER_BYTES];
    header[0..8].copy_from_slice(&(index_size as u64).to_le_bytes());
    header[8..16].copy_from_slice(&data_len.to_le_bytes());
    header[16..24].copy_from_slice(&(entry_count as u64).to_le_bytes());
    out.write_all(&header)?;

    // Entry table.
    for (i, p) in placed.iter().enumerate() {
        let (name_offset, name_len) = name_ranges[i];
        // Writer invariants, not hostile-input hazards -- but a silent
        // truncation here writes a plausible, wrong index with no error at
        // any later read, so these are asserted rather than cast blindly.
        // A `debug_assert!` would not do: release is what writes real
        // installs.
        assert!(
            name_offset <= u32::MAX as usize,
            "resident index string table offset {name_offset} exceeds the 32-bit field \
             (string table over 4 GiB)"
        );
        assert!(
            name_len <= u16::MAX as usize,
            "tensor name {:?} is {name_len} bytes, exceeding the 16-bit name-length field",
            specs[i].name()
        );
        let mut entry = [0u8; ENTRY_BYTES];
        entry[0..4].copy_from_slice(&(name_offset as u32).to_le_bytes());
        entry[4..6].copy_from_slice(&(name_len as u16).to_le_bytes());
        entry[6] = p.dtype;
        entry[7] = 0;
        entry[8..16].copy_from_slice(&p.weight_offset.to_le_bytes());
        entry[16..24].copy_from_slice(&p.weight_size.to_le_bytes());
        entry[24..28].copy_from_slice(&p.shape.0.to_le_bytes());
        entry[28..32].copy_from_slice(&p.shape.1.to_le_bytes());
        entry[32..36].copy_from_slice(&p.shape.2.to_le_bytes());
        entry[36..40].copy_from_slice(&p.shape.3.to_le_bytes());
        entry[40..48].copy_from_slice(&p.scale_offset.to_le_bytes());
        entry[48..56].copy_from_slice(&p.scale_size.to_le_bytes());
        entry[56..64].copy_from_slice(&p.bias_offset.to_le_bytes());
        entry[64..72].copy_from_slice(&p.bias_size.to_le_bytes());
        out.write_all(&entry)?;
    }

    // String table, then pad the index region up to `index_size`.
    for spec in specs {
        out.write_all(spec.name().as_bytes())?;
    }
    write_zeros(out, index_size - string_table_end)?;

    // PASS 2: the data region, streamed straight from each spec's own
    // bytes. Padding is taken from pass 1's `placed` entries rather than
    // recomputed, so this loop can never disagree with the layout the
    // entry table just declared.
    let mut cursor = 0u64;
    for (i, spec) in specs.iter().enumerate() {
        let want = placed[i].weight_offset - index_size as u64;
        write_zeros(out, (want - cursor) as usize)?;
        cursor = want;
        match spec {
            EntryView::Packed(t, _) => {
                out.write_all(&t.packed)?;
                cursor += t.packed.len() as u64;
                let scale_bytes = u16_slice_to_le_bytes(&t.scales);
                out.write_all(&scale_bytes)?;
                cursor += scale_bytes.len() as u64;
                let bias_bytes = u16_slice_to_le_bytes(&t.biases);
                out.write_all(&bias_bytes)?;
                cursor += bias_bytes.len() as u64;
            }
            EntryView::Raw(r) => {
                out.write_all(&r.bytes)?;
                cursor += r.bytes.len() as u64;
            }
        }
    }
    debug_assert_eq!(
        cursor, data_len,
        "pass 2 must consume exactly what pass 1 sized"
    );

    Ok(())
}
