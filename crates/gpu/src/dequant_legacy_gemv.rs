//! Host-side dispatch for the Q4_0, Q4_1, Q5_0 and Q5_1 kernels in
//! `shaders/dequant_legacy.metal`: a resident GEMV and an embedding lookup per
//! type. No routed-expert pair; no real file asks for one, and an install
//! carrying legacy experts fails at the routed dispatch, by name, like Q6_K.
//!
//! PORT-LOCAL, like the Q8_0 and K-quant siblings. The contract is the CPU
//! decoders `turbospark_compute::dequantize_q{4,5}_{0,1}`, which
//! `crates/gpu/tests/dequant_legacy_gemv_parity.rs` holds the kernels to and
//! which `crates/repack`'s tests hold to ggml's own `to_float`.
//!
//! One module for four types because they share one block geometry (32
//! elements) and so one dispatch shape; only the block bytes and the kernel
//! name differ, and that difference is a [`LegacyBlockType`] value rather than
//! four copies of this file.

use half::f16;
use metal::FunctionConstantValues;

use crate::bytes::{half_slice_to_le_bytes, read_half_buffer, u32_bytes};
use crate::context::{
    dispatch_one_threadgroup_per_row_offsets, GpuError, MetalContext, PassEncoder,
};

const SOURCE: &str = include_str!("shaders/dequant_legacy.metal");
const THREADS_PER_GROUP: u64 = 256; // 8 rows/threadgroup * 32 lanes/SIMD group.
const ROWS_PER_THREADGROUP: u64 = 8;

/// Elements per block, shared by all four types.
pub const LEGACY_BLOCK_ELEMS: usize = 32;
/// Bytes per Q4_0 block: f16 `d`, 16 nibble bytes.
pub const Q4_0_BLOCK_BYTES: usize = 18;
/// Bytes per Q4_1 block: f16 `d`, f16 `m`, 16 nibble bytes.
pub const Q4_1_BLOCK_BYTES: usize = 20;
/// Bytes per Q5_0 block: f16 `d`, u32 fifth bits, 16 nibble bytes.
pub const Q5_0_BLOCK_BYTES: usize = 22;
/// Bytes per Q5_1 block: f16 `d`, f16 `m`, u32 fifth bits, 16 nibble bytes.
pub const Q5_1_BLOCK_BYTES: usize = 24;

/// Which legacy block layout a table or matrix holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegacyBlockType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
}

impl LegacyBlockType {
    /// Every legacy type, for tests that sweep them.
    pub const ALL: [LegacyBlockType; 4] = [
        LegacyBlockType::Q4_0,
        LegacyBlockType::Q4_1,
        LegacyBlockType::Q5_0,
        LegacyBlockType::Q5_1,
    ];

    /// Bytes one 32-element block occupies.
    pub fn block_bytes(self) -> usize {
        match self {
            LegacyBlockType::Q4_0 => Q4_0_BLOCK_BYTES,
            LegacyBlockType::Q4_1 => Q4_1_BLOCK_BYTES,
            LegacyBlockType::Q5_0 => Q5_0_BLOCK_BYTES,
            LegacyBlockType::Q5_1 => Q5_1_BLOCK_BYTES,
        }
    }

    /// Bytes a row of `n` elements occupies; panics on a ragged `n`, because a
    /// partial block is a shape error and not something to round.
    pub fn row_bytes(self, n: usize) -> usize {
        assert_eq!(
            n % LEGACY_BLOCK_ELEMS,
            0,
            "N ({n}) is not a whole number of {LEGACY_BLOCK_ELEMS}-element blocks"
        );
        n / LEGACY_BLOCK_ELEMS * self.block_bytes()
    }

    fn gemv_kernel(self) -> &'static str {
        match self {
            LegacyBlockType::Q4_0 => "dequant_q4_0_gemv_simd",
            LegacyBlockType::Q4_1 => "dequant_q4_1_gemv_simd",
            LegacyBlockType::Q5_0 => "dequant_q5_0_gemv_simd",
            LegacyBlockType::Q5_1 => "dequant_q5_1_gemv_simd",
        }
    }

    fn embed_kernel(self) -> &'static str {
        match self {
            LegacyBlockType::Q4_0 => "embed_lookup_q4_0",
            LegacyBlockType::Q4_1 => "embed_lookup_q4_1",
            LegacyBlockType::Q5_0 => "embed_lookup_q5_0",
            LegacyBlockType::Q5_1 => "embed_lookup_q5_1",
        }
    }
}

/// The kernels declare no function constants, the same deliberate choice as
/// the other GGUF dispatches here. The pipeline cache key is the kernel name
/// plus the shader source, and the kernel name already encodes the type.
fn no_function_constants() -> FunctionConstantValues {
    FunctionConstantValues::new()
}

/// A whole legacy-quant weight matrix addressed IN PLACE inside one shared
/// `MTLBuffer` (normally `ResidentGpuWeights::buffer`): `rows` consecutive
/// byte runs of `kind.row_bytes(cols)` each, starting at `weights_offset`.
pub struct LegacyResidentMatrix<'a> {
    pub buffer: &'a metal::Buffer,
    pub weights_offset: u64,
    pub rows: usize,
    pub cols: usize,
    pub kind: LegacyBlockType,
}

/// Encoder-level offset-bound GEMV over legacy blocks:
/// `y[m] = sum_n W[m, n] * x[n]`.
pub fn encode_dequant_legacy_gemv_resident(
    context: &mut MetalContext,
    pass: &PassEncoder,
    w: &LegacyResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
) -> Result<(), GpuError> {
    assert!(w.rows > 0);
    let _ = w.kind.row_bytes(w.cols);
    let m_u32 = w.rows as u32;
    let n_u32 = w.cols as u32;
    let pipeline = context.pipeline(SOURCE, w.kind.gemv_kernel(), &no_function_constants(), b"")?;
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

/// One-shot [`encode_dequant_legacy_gemv_resident`] for the parity tests.
pub fn dequant_legacy_gemv_resident(
    context: &mut MetalContext,
    w: &LegacyResidentMatrix<'_>,
    x: &[f16],
) -> Result<Vec<f16>, GpuError> {
    assert_eq!(x.len(), w.cols);
    let x_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(x));
    let y_buffer = context.new_output_buffer((w.rows * std::mem::size_of::<u16>()) as u64);

    let m_u32 = w.rows as u32;
    let n_u32 = w.cols as u32;
    let pipeline = context.pipeline(SOURCE, w.kind.gemv_kernel(), &no_function_constants(), b"")?;
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

/// Encoder-level embedding lookup: dequantizes one row of a legacy-quant table
/// (bound in place, normally an offset into the resident buffer) into `out`
/// (`d` halfs), scaled by `out_scale`.
#[allow(clippy::too_many_arguments)]
pub fn encode_embed_lookup_legacy(
    context: &mut MetalContext,
    pass: &PassEncoder,
    kind: LegacyBlockType,
    table: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    token_id: u32,
    d: u32,
    out_scale: f32,
) -> Result<(), GpuError> {
    assert_eq!(d as usize % LEGACY_BLOCK_ELEMS, 0);
    let pipeline = context.pipeline(SOURCE, kind.embed_kernel(), &no_function_constants(), b"")?;
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
