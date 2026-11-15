//! Host-side dispatch for the kernels in `shaders/dequant_ext.metal`: a
//! resident GEMV and an embedding lookup per GGUF block type that ggml added
//! after the K and IQ families: Q1_0, TQ2_0, TQ1_0, NVFP4, Q2_0 and MXFP4.
//! No routed-expert pair. PORT-LOCAL, like the legacy sibling this mirrors;
//! the contract is the CPU decoders in `turbospark_compute`, which
//! `crates/gpu/tests/dequant_ext_gemv_parity.rs` holds the kernels to.

use half::f16;
use metal::FunctionConstantValues;

use crate::bytes::{half_slice_to_le_bytes, read_half_buffer, u32_bytes};
use crate::context::{
    dispatch_one_threadgroup_per_row_offsets, GpuError, MetalContext, PassEncoder,
};

const SOURCE: &str = concat!(
    include_str!("shaders/dequant_iq1_narrow_tables.metal"),
    include_str!("shaders/dequant_ext.metal")
);
const THREADS_PER_GROUP: u64 = 256; // 8 rows/threadgroup * 32 lanes/SIMD group.
const ROWS_PER_THREADGROUP: u64 = 8;

/// Elements per Q1_0 block.
pub const Q1_0_BLOCK_ELEMS: usize = 128;
/// Bytes per Q1_0 block: f16 `d`, 16 sign bytes.
pub const Q1_0_BLOCK_BYTES: usize = 18;

/// Elements per TQ2_0 block.
pub const TQ2_0_BLOCK_ELEMS: usize = 256;
/// Bytes per TQ2_0 block: 64 code bytes, f16 `d` last.
pub const TQ2_0_BLOCK_BYTES: usize = 66;
/// Elements per TQ1_0 block.
pub const TQ1_0_BLOCK_ELEMS: usize = 256;
/// Bytes per TQ1_0 block: 48 `qs`, 4 `qh`, f16 `d` last.
pub const TQ1_0_BLOCK_BYTES: usize = 54;
/// Elements per NVFP4 block.
pub const NVFP4_BLOCK_ELEMS: usize = 64;
/// Bytes per NVFP4 block: 4 E4M3 scale bytes first, then 32 code bytes.
pub const NVFP4_BLOCK_BYTES: usize = 36;

/// Elements per Q2_0 block.
pub const Q2_0_BLOCK_ELEMS: usize = 64;
/// Bytes per Q2_0 block: f16 `d`, 16 code bytes.
pub const Q2_0_BLOCK_BYTES: usize = 18;
// MXFP4's block size is `crate::MXFP4_BLOCK_ELEMS`/`_BYTES` (the routed kernels
// export it); this file reads those rather than redefining them.

/// Elements per block of the Unsloth fork IQ1_XS, IQ1_XXS and IQ1_XXXS.
pub const IQ1_NARROW_BLOCK_ELEMS: usize = 256;
/// IQ1_XS block bytes: f16 `d`, 32 `qs`, 8 `qh`, 4 `sc`.
pub const IQ1_XS_BLOCK_BYTES: usize = 46;
/// IQ1_XXS block bytes: f16 `d`, 32 `qs`, 8 `qh`.
pub const IQ1_XXS_BLOCK_BYTES: usize = 42;
/// IQ1_XXXS block bytes: f16 `d`, 32 `qs`, 4 `sc`.
pub const IQ1_XXXS_BLOCK_BYTES: usize = 38;

/// Which block layout a table or matrix holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtBlockType {
    Q1_0,
    Tq2_0,
    Tq1_0,
    Nvfp4,
    Q2_0,
    Mxfp4,
    Iq1Xs,
    Iq1Xxs,
    Iq1Xxxs,
}

impl ExtBlockType {
    /// Every type, for tests that sweep them.
    pub const ALL: [ExtBlockType; 9] = [
        ExtBlockType::Q1_0,
        ExtBlockType::Tq2_0,
        ExtBlockType::Tq1_0,
        ExtBlockType::Nvfp4,
        ExtBlockType::Q2_0,
        ExtBlockType::Mxfp4,
        ExtBlockType::Iq1Xs,
        ExtBlockType::Iq1Xxs,
        ExtBlockType::Iq1Xxxs,
    ];

    /// Elements one block holds.
    pub fn block_elems(self) -> usize {
        match self {
            ExtBlockType::Q1_0 => Q1_0_BLOCK_ELEMS,
            ExtBlockType::Tq2_0 => TQ2_0_BLOCK_ELEMS,
            ExtBlockType::Tq1_0 => TQ1_0_BLOCK_ELEMS,
            ExtBlockType::Nvfp4 => NVFP4_BLOCK_ELEMS,
            ExtBlockType::Q2_0 => Q2_0_BLOCK_ELEMS,
            ExtBlockType::Mxfp4 => crate::MXFP4_BLOCK_ELEMS,
            ExtBlockType::Iq1Xs | ExtBlockType::Iq1Xxs | ExtBlockType::Iq1Xxxs => {
                IQ1_NARROW_BLOCK_ELEMS
            }
        }
    }

    /// Bytes one block occupies.
    pub fn block_bytes(self) -> usize {
        match self {
            ExtBlockType::Q1_0 => Q1_0_BLOCK_BYTES,
            ExtBlockType::Tq2_0 => TQ2_0_BLOCK_BYTES,
            ExtBlockType::Tq1_0 => TQ1_0_BLOCK_BYTES,
            ExtBlockType::Nvfp4 => NVFP4_BLOCK_BYTES,
            ExtBlockType::Q2_0 => Q2_0_BLOCK_BYTES,
            ExtBlockType::Mxfp4 => crate::MXFP4_BLOCK_BYTES,
            ExtBlockType::Iq1Xs => IQ1_XS_BLOCK_BYTES,
            ExtBlockType::Iq1Xxs => IQ1_XXS_BLOCK_BYTES,
            ExtBlockType::Iq1Xxxs => IQ1_XXXS_BLOCK_BYTES,
        }
    }

    /// Bytes a row of `n` elements occupies; panics on a ragged `n`, because a
    /// partial block is a shape error and not something to round.
    pub fn row_bytes(self, n: usize) -> usize {
        assert_eq!(
            n % self.block_elems(),
            0,
            "N ({n}) is not a whole number of {}-element blocks",
            self.block_elems()
        );
        n / self.block_elems() * self.block_bytes()
    }

    fn gemv_kernel(self) -> &'static str {
        match self {
            ExtBlockType::Q1_0 => "dequant_q1_0_gemv_simd",
            ExtBlockType::Tq2_0 => "dequant_tq2_0_gemv_simd",
            ExtBlockType::Tq1_0 => "dequant_tq1_0_gemv_simd",
            ExtBlockType::Nvfp4 => "dequant_nvfp4_gemv_simd",
            ExtBlockType::Q2_0 => "dequant_q2_0_gemv_simd",
            ExtBlockType::Mxfp4 => "dequant_mxfp4_gemv_simd",
            ExtBlockType::Iq1Xs => "dequant_iq1_xs_gemv_simd",
            ExtBlockType::Iq1Xxs => "dequant_iq1_xxs_gemv_simd",
            ExtBlockType::Iq1Xxxs => "dequant_iq1_xxxs_gemv_simd",
        }
    }

    fn embed_kernel(self) -> &'static str {
        match self {
            ExtBlockType::Q1_0 => "embed_lookup_q1_0",
            ExtBlockType::Tq2_0 => "embed_lookup_tq2_0",
            ExtBlockType::Tq1_0 => "embed_lookup_tq1_0",
            ExtBlockType::Nvfp4 => "embed_lookup_nvfp4",
            ExtBlockType::Q2_0 => "embed_lookup_q2_0",
            ExtBlockType::Mxfp4 => "embed_lookup_mxfp4",
            ExtBlockType::Iq1Xs => "embed_lookup_iq1_xs",
            ExtBlockType::Iq1Xxs => "embed_lookup_iq1_xxs",
            ExtBlockType::Iq1Xxxs => "embed_lookup_iq1_xxxs",
        }
    }
}

/// The kernels declare no function constants, the same deliberate choice as
/// the other GGUF dispatches here. The pipeline cache key is the kernel name
/// plus the shader source, and the kernel name already encodes the type.
fn no_function_constants() -> FunctionConstantValues {
    FunctionConstantValues::new()
}

/// A whole ext-quant weight matrix addressed IN PLACE inside one shared
/// `MTLBuffer` (normally `ResidentGpuWeights::buffer`): `rows` consecutive
/// byte runs of `kind.row_bytes(cols)` each, starting at `weights_offset`.
pub struct ExtResidentMatrix<'a> {
    pub buffer: &'a metal::Buffer,
    pub weights_offset: u64,
    pub rows: usize,
    pub cols: usize,
    pub kind: ExtBlockType,
}

/// Encoder-level offset-bound GEMV over ext blocks:
/// `y[m] = sum_n W[m, n] * x[n]`.
pub fn encode_dequant_ext_gemv_resident(
    context: &mut MetalContext,
    pass: &PassEncoder,
    w: &ExtResidentMatrix<'_>,
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

/// One-shot [`encode_dequant_ext_gemv_resident`] for the parity tests.
pub fn dequant_ext_gemv_resident(
    context: &mut MetalContext,
    w: &ExtResidentMatrix<'_>,
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

/// Encoder-level embedding lookup: dequantizes one row of a ext-quant table
/// (bound in place, normally an offset into the resident buffer) into `out`
/// (`d` halfs), scaled by `out_scale`.
#[allow(clippy::too_many_arguments)]
pub fn encode_embed_lookup_ext(
    context: &mut MetalContext,
    pass: &PassEncoder,
    kind: ExtBlockType,
    table: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    token_id: u32,
    d: u32,
    out_scale: f32,
) -> Result<(), GpuError> {
    assert_eq!(d as usize % kind.block_elems(), 0);
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
