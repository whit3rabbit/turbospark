//! Host-side dispatch for `dequant_q4_k_gemv_simd` in
//! `shaders/dequant_q4_k.metal` (ROADMAP Phase G Stage 2).
//!
//! PORT-LOCAL rather than vendored, like its Q8_0 sibling: the Swift engine
//! has no GGUF intake, so there is no upstream kernel this mirrors. Its only
//! contract is `turbospark_compute::dequant_q4_k_gemv`, which is what
//! `crates/gpu/tests/dequant_q4_k_gemv_parity.rs` holds it to.
//!
//! Like Q8_0 and unlike the affine siblings, a Q4_K row is ONE byte run: both
//! super-scales and all sixteen 6-bit sub-scales and sub-mins live inside the
//! 144-byte superblock, so there are no planes to bind and no group size to
//! agree on. The dispatch shape is the same as Q8_0's for a different reason:
//! there, 32 lanes over a 32-element block is one weight per lane; here it is
//! 32 lanes over a 32-BYTE nibble group, so a lane owns one byte and hence
//! two elements 32 apart in two different sub-blocks.

use half::f16;
use metal::FunctionConstantValues;

use crate::bytes::{half_slice_to_le_bytes, read_half_buffer, u32_bytes};
use crate::context::{
    dispatch_one_threadgroup_per_row, dispatch_one_threadgroup_per_row_offsets, GpuError,
    MetalContext, PassEncoder,
};

const SOURCE: &str = include_str!("shaders/dequant_q4_k.metal");
const THREADS_PER_GROUP: u64 = 256; // 8 rows/threadgroup * 32 lanes/SIMD group.
const ROWS_PER_THREADGROUP: u64 = 8;

/// Elements per Q4_K superblock. Mirrors `turbospark_compute::Q4_K_BLOCK_ELEMS`;
/// the two are held equal by `crates/repack`'s block-table test.
pub const Q4_K_BLOCK_ELEMS: usize = 256;
/// Bytes per Q4_K superblock: two f16 super-scales, 12 packed bytes of 6-bit
/// sub-scales and sub-mins, then 128 nibble-packed quants.
pub const Q4_K_BLOCK_BYTES: usize = 144;

/// The kernel declares no function constants. Same deliberate choice as the
/// Q8_0 dispatch: every other quant kernel here specializes M/N, and this
/// shape is not on a hot path yet.
fn no_function_constants() -> FunctionConstantValues {
    FunctionConstantValues::new()
}

/// Bytes a Q4_K row of `n` elements occupies.
pub fn q4_k_row_bytes(n: usize) -> usize {
    assert_eq!(
        n % Q4_K_BLOCK_ELEMS,
        0,
        "N ({n}) is not a whole number of {Q4_K_BLOCK_ELEMS}-element superblocks"
    );
    n / Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES
}

/// `y[m] = sum_n W[m, n] * x[n]` over Q4_K rows. Each row is the raw GGUF
/// byte run for that output channel; every row must share the same `n`.
pub fn dequant_q4_k_gemv(
    context: &mut MetalContext,
    weight_rows: &[&[u8]],
    x: &[f16],
    n: usize,
) -> Result<Vec<f16>, GpuError> {
    assert!(!weight_rows.is_empty());
    assert_eq!(x.len(), n);
    let row_bytes = q4_k_row_bytes(n);

    let m = weight_rows.len();
    let mut w_bytes = Vec::with_capacity(m * row_bytes);
    for row in weight_rows {
        assert_eq!(row.len(), row_bytes, "every row must be {row_bytes} bytes");
        w_bytes.extend_from_slice(row);
    }

    let w_buffer = context.new_buffer_with_data(&w_bytes);
    let x_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(x));
    let y_buffer = context.new_output_buffer((m * std::mem::size_of::<u16>()) as u64);

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let pipeline = context.pipeline(
        SOURCE,
        "dequant_q4_k_gemv_simd",
        &no_function_constants(),
        b"",
    )?;
    let threadgroups = m.div_ceil(ROWS_PER_THREADGROUP as usize) as u64;
    dispatch_one_threadgroup_per_row(
        context,
        &pipeline,
        &[(&w_buffer, 0), (&x_buffer, 1), (&y_buffer, 2)],
        &[(u32_bytes(&m_u32), 3), (u32_bytes(&n_u32), 4)],
        threadgroups,
        THREADS_PER_GROUP,
    );

    Ok(read_half_buffer(&y_buffer, m))
}

/// A whole Q4_K weight matrix addressed IN PLACE inside one shared
/// `MTLBuffer` (normally `ResidentGpuWeights::buffer`): `rows` consecutive
/// byte runs of `q4_k_row_bytes(cols)` each, starting at `weights_offset`.
pub struct Q4KResidentMatrix<'a> {
    pub buffer: &'a metal::Buffer,
    pub weights_offset: u64,
    pub rows: usize,
    pub cols: usize,
}

/// Encoder-level offset-bound Q4_K GEMV: same kernel and math as
/// [`dequant_q4_k_gemv`], weights bound as an offset into `w.buffer`.
pub fn encode_dequant_q4_k_gemv_resident(
    context: &mut MetalContext,
    pass: &PassEncoder,
    w: &Q4KResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
) -> Result<(), GpuError> {
    assert!(w.rows > 0);
    let _ = q4_k_row_bytes(w.cols);
    let m_u32 = w.rows as u32;
    let n_u32 = w.cols as u32;
    let pipeline = context.pipeline(
        SOURCE,
        "dequant_q4_k_gemv_simd",
        &no_function_constants(),
        b"",
    )?;
    let threadgroups = w.rows.div_ceil(ROWS_PER_THREADGROUP as usize) as u64;
    pass.encode_threadgroups(
        &pipeline,
        &[
            (w.buffer, 0, w.weights_offset),
            (x.0, 1, x.1),
            (y.0, 2, y.1),
        ],
        &[(u32_bytes(&m_u32), 3), (u32_bytes(&n_u32), 4)],
        threadgroups,
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// Encoder-level `embed_lookup_q4_k`: dequantizes one row of a Q4_K
/// embedding table (bound in place, normally an offset into the resident
/// buffer) into `out` (`d` halfs), scaled by `out_scale`.
///
/// Qwen 3.6's Q4_K_M keeps `token_embd.weight` at Q4_K, which is why this
/// exists; the Q8_0 sibling covers Gemma's file. One binding rather than the
/// affine lookup's three, for the usual block-quant reason.
pub fn encode_embed_lookup_q4_k(
    context: &mut MetalContext,
    pass: &PassEncoder,
    table: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    token_id: u32,
    d: u32,
    out_scale: f32,
) -> Result<(), GpuError> {
    assert_eq!(d as usize % Q4_K_BLOCK_ELEMS, 0);
    let pipeline = context.pipeline(SOURCE, "embed_lookup_q4_k", &no_function_constants(), b"")?;
    pass.encode_threads_3d(
        &pipeline,
        &[(table.0, 0, table.1), (out.0, 1, out.1)],
        &[
            (u32_bytes(&token_id), 2),
            (u32_bytes(&d), 3),
            (crate::bytes::f32_bytes(&out_scale), 4),
        ],
        (d as u64, 1, 1),
        (64, 1, 1),
    );
    Ok(())
}

/// One-shot [`encode_dequant_q4_k_gemv_resident`] for the parity tests.
pub fn dequant_q4_k_gemv_resident(
    context: &mut MetalContext,
    w: &Q4KResidentMatrix<'_>,
    x: &[f16],
) -> Result<Vec<f16>, GpuError> {
    assert_eq!(x.len(), w.cols);
    let x_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(x));
    let y_buffer = context.new_output_buffer((w.rows * std::mem::size_of::<u16>()) as u64);

    let m_u32 = w.rows as u32;
    let n_u32 = w.cols as u32;
    let pipeline = context.pipeline(
        SOURCE,
        "dequant_q4_k_gemv_simd",
        &no_function_constants(),
        b"",
    )?;
    let threadgroups = w.rows.div_ceil(ROWS_PER_THREADGROUP as usize) as u64;
    dispatch_one_threadgroup_per_row_offsets(
        context,
        &pipeline,
        &[
            (w.buffer, 0, w.weights_offset),
            (&x_buffer, 1, 0),
            (&y_buffer, 2, 0),
        ],
        &[(u32_bytes(&m_u32), 3), (u32_bytes(&n_u32), 4)],
        threadgroups,
        THREADS_PER_GROUP,
    );

    Ok(read_half_buffer(&y_buffer, w.rows))
}
