//! Host-side dispatch for `dequant_q5_k_gemv_simd` in
//! `shaders/dequant_q5_k.metal` (ROADMAP Phase M2).
//!
//! PORT-LOCAL rather than vendored, like its Q8_0, Q4_K and Q6_K siblings: the
//! Swift engine has no GGUF intake, so there is no upstream kernel this
//! mirrors. Its only contract is `turbospark_compute::dequant_q5_k_gemv`,
//! which is what `crates/gpu/tests/dequant_q5_k_gemv_parity.rs` holds it to.
//!
//! Q5_K exists here for one tensor shape: Mixtral 8x7B's Q4_K_M carries
//! `attn_output` in Q5_K while its experts are Q4_K. So this dispatch has no
//! embedding-lookup and no MoE sibling, the same judgement `dequant_q6_k_gemv`
//! records, and gains one only when a real file asks.
//!
//! **The shader source is a CONCATENATION** (crate Gotcha 4): the Q5_K kernel
//! reuses `q4_k_scale_min` out of `dequant_q4_k.metal`, because the 6-bit
//! scale-and-min packing is Q4_K's byte for byte and a second copy of that
//! split is exactly what would let the two drift. `concat!` of two
//! `include_str!`s is one `&'static str` with one stable address, which is
//! what the address-keyed pipeline cache needs (Gotcha 1).

use half::f16;
use metal::FunctionConstantValues;

use crate::bytes::{half_slice_to_le_bytes, read_half_buffer, u32_bytes};
use crate::context::{
    dispatch_one_threadgroup_per_row, dispatch_one_threadgroup_per_row_offsets, GpuError,
    MetalContext, PassEncoder,
};

const SOURCE: &str = concat!(
    include_str!("shaders/dequant_q4_k.metal"),
    include_str!("shaders/dequant_q5_k.metal")
);
const THREADS_PER_GROUP: u64 = 256; // 8 rows/threadgroup * 32 lanes/SIMD group.
const ROWS_PER_THREADGROUP: u64 = 8;

/// Elements per Q5_K superblock. Mirrors `turbospark_compute::Q5_K_BLOCK_ELEMS`;
/// the two are held equal by `crates/repack`'s block-table test.
pub const Q5_K_BLOCK_ELEMS: usize = 256;
/// Bytes per Q5_K superblock: f16 `d`, f16 `dmin`, 12 packed scale bytes, 32
/// fifth-bit bytes, then 128 nibble bytes.
pub const Q5_K_BLOCK_BYTES: usize = 176;

/// The kernel declares no function constants, the same deliberate choice as
/// every other GGUF dispatch here: the affine kernels specialize M/N, and
/// these shapes are not on a hot path yet.
fn no_function_constants() -> FunctionConstantValues {
    FunctionConstantValues::new()
}

/// Bytes a Q5_K row of `n` elements occupies.
pub fn q5_k_row_bytes(n: usize) -> usize {
    assert_eq!(
        n % Q5_K_BLOCK_ELEMS,
        0,
        "N ({n}) is not a whole number of {Q5_K_BLOCK_ELEMS}-element superblocks"
    );
    n / Q5_K_BLOCK_ELEMS * Q5_K_BLOCK_BYTES
}

/// `y[m] = sum_n W[m, n] * x[n]` over Q5_K rows. Each row is the raw GGUF byte
/// run for that output channel; every row must share the same `n`.
pub fn dequant_q5_k_gemv(
    context: &mut MetalContext,
    weight_rows: &[&[u8]],
    x: &[f16],
    n: usize,
) -> Result<Vec<f16>, GpuError> {
    assert!(!weight_rows.is_empty());
    assert_eq!(x.len(), n);
    let row_bytes = q5_k_row_bytes(n);

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
        "dequant_q5_k_gemv_simd",
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

/// A whole Q5_K weight matrix addressed IN PLACE inside one shared
/// `MTLBuffer` (normally `ResidentGpuWeights::buffer`): `rows` consecutive
/// byte runs of `q5_k_row_bytes(cols)` each, starting at `weights_offset`.
pub struct Q5KResidentMatrix<'a> {
    /// The shared buffer the matrix lives in.
    pub buffer: &'a metal::Buffer,
    /// Byte offset of row 0 inside that buffer.
    pub weights_offset: u64,
    /// Output rows.
    pub rows: usize,
    /// Elements per row.
    pub cols: usize,
}

/// Encoder-level offset-bound Q5_K GEMV: same kernel and math as
/// [`dequant_q5_k_gemv`], weights bound as an offset into `w.buffer`.
pub fn encode_dequant_q5_k_gemv_resident(
    context: &mut MetalContext,
    pass: &PassEncoder,
    w: &Q5KResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
) -> Result<(), GpuError> {
    assert!(w.rows > 0);
    let _ = q5_k_row_bytes(w.cols);
    let m_u32 = w.rows as u32;
    let n_u32 = w.cols as u32;
    let pipeline = context.pipeline(
        SOURCE,
        "dequant_q5_k_gemv_simd",
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

/// One-shot [`encode_dequant_q5_k_gemv_resident`] for the parity tests.
pub fn dequant_q5_k_gemv_resident(
    context: &mut MetalContext,
    w: &Q5KResidentMatrix<'_>,
    x: &[f16],
) -> Result<Vec<f16>, GpuError> {
    assert_eq!(x.len(), w.cols);
    let x_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(x));
    let y_buffer = context.new_output_buffer((w.rows * std::mem::size_of::<u16>()) as u64);

    let m_u32 = w.rows as u32;
    let n_u32 = w.cols as u32;
    let pipeline = context.pipeline(
        SOURCE,
        "dequant_q5_k_gemv_simd",
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
