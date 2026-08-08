//! Host-side dispatch for `dequant_q8_0_gemv_simd` in
//! `shaders/dequant_q8_0.metal` (ROADMAP Phase G Stage 2).
//!
//! Unlike its INT4/INT8 siblings this shader is PORT-LOCAL rather than
//! vendored: the Swift engine has no GGUF intake, so there is no upstream
//! kernel it mirrors. Its only contract is
//! `mrefrust_compute::dequant_q8_0_gemv`, which is what
//! `crates/gpu/tests/dequant_q8_0_gemv_parity.rs` holds it to.
//!
//! A Q8_0 row is ONE byte run, not three planes: the scale lives inside each
//! 34-byte block. So there is no `scales`/`biases` pair to bind and no group
//! size to agree on, which is why this dispatch is shorter than the INT8 one
//! rather than a copy of it.

use half::f16;
use metal::FunctionConstantValues;

use crate::bytes::{half_slice_to_le_bytes, read_half_buffer, u32_bytes};
use crate::context::{
    dispatch_one_threadgroup_per_row, dispatch_one_threadgroup_per_row_offsets, GpuError,
    MetalContext, PassEncoder,
};

const SOURCE: &str = include_str!("shaders/dequant_q8_0.metal");
const THREADS_PER_GROUP: u64 = 256; // 8 rows/threadgroup * 32 lanes/SIMD group.
const ROWS_PER_THREADGROUP: u64 = 8;

/// Elements per Q8_0 block. Mirrors `mrefrust_compute::Q8_0_BLOCK_ELEMS`;
/// the two are held equal by `crates/repack`'s block-table test.
pub const Q8_0_BLOCK_ELEMS: usize = 32;
/// Bytes per Q8_0 block: an f16 scale then 32 signed weights.
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// The kernel declares no function constants, so nothing is specialized into
/// it. Kept as a named function rather than inlined so the reason is
/// recorded: every other quant dispatch here has an M/N specialization and
/// this one deliberately does not, since the shape is not on a hot path yet.
fn no_function_constants() -> FunctionConstantValues {
    FunctionConstantValues::new()
}

/// Bytes a Q8_0 row of `n` elements occupies.
pub fn q8_0_row_bytes(n: usize) -> usize {
    assert_eq!(
        n % Q8_0_BLOCK_ELEMS,
        0,
        "N ({n}) is not a whole number of {Q8_0_BLOCK_ELEMS}-element blocks"
    );
    n / Q8_0_BLOCK_ELEMS * Q8_0_BLOCK_BYTES
}

/// `y[m] = sum_n W[m, n] * x[n]` over Q8_0 rows. Each row is the raw GGUF
/// byte run for that output channel; every row must share the same `n`.
pub fn dequant_q8_0_gemv(
    context: &mut MetalContext,
    weight_rows: &[&[u8]],
    x: &[f16],
    n: usize,
) -> Result<Vec<f16>, GpuError> {
    assert!(!weight_rows.is_empty());
    assert_eq!(x.len(), n);
    let row_bytes = q8_0_row_bytes(n);

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
        "dequant_q8_0_gemv_simd",
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

/// A whole Q8_0 weight matrix addressed IN PLACE inside one shared
/// `MTLBuffer` (normally `ResidentGpuWeights::buffer`): `rows` consecutive
/// byte runs of `q8_0_row_bytes(cols)` each, starting at `weights_offset`.
/// The Q8_0 sibling of `Int8ResidentMatrix`, and shorter for the same reason
/// the dispatch above is: there are no scale or bias planes to locate.
pub struct Q8_0ResidentMatrix<'a> {
    pub buffer: &'a metal::Buffer,
    pub weights_offset: u64,
    pub rows: usize,
    pub cols: usize,
}

/// Encoder-level offset-bound Q8_0 GEMV: same kernel and math as
/// [`dequant_q8_0_gemv`], weights bound as an offset into `w.buffer`.
pub fn encode_dequant_q8_0_gemv_resident(
    context: &mut MetalContext,
    pass: &PassEncoder,
    w: &Q8_0ResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
) -> Result<(), GpuError> {
    assert!(w.rows > 0);
    let _ = q8_0_row_bytes(w.cols);
    let m_u32 = w.rows as u32;
    let n_u32 = w.cols as u32;
    let pipeline = context.pipeline(
        SOURCE,
        "dequant_q8_0_gemv_simd",
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

/// One-shot [`encode_dequant_q8_0_gemv_resident`] for the parity tests.
pub fn dequant_q8_0_gemv_resident(
    context: &mut MetalContext,
    w: &Q8_0ResidentMatrix<'_>,
    x: &[f16],
) -> Result<Vec<f16>, GpuError> {
    assert_eq!(x.len(), w.cols);
    let x_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(x));
    let y_buffer = context.new_output_buffer((w.rows * std::mem::size_of::<u16>()) as u64);

    let m_u32 = w.rows as u32;
    let n_u32 = w.cols as u32;
    let pipeline = context.pipeline(
        SOURCE,
        "dequant_q8_0_gemv_simd",
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

/// Encoder-level `embed_lookup_q8_0`: dequantizes one row of a Q8_0
/// embedding table (bound in place, normally an offset into the resident
/// buffer) into `out` (`d` halfs), scaled by `out_scale`.
///
/// The Q8_0 sibling of [`crate::encode_embed_lookup_int4`], with one binding
/// instead of three: a Q8_0 row carries its scales inline.
pub fn encode_embed_lookup_q8_0(
    context: &mut MetalContext,
    pass: &PassEncoder,
    table: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    token_id: u32,
    d: u32,
    out_scale: f32,
) -> Result<(), GpuError> {
    assert_eq!(d as usize % Q8_0_BLOCK_ELEMS, 0);
    let pipeline = context.pipeline(SOURCE, "embed_lookup_q8_0", &no_function_constants(), b"")?;
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
