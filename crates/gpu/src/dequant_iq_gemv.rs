//! Host-side dispatch for the three IQ GEMVs in `shaders/dequant_iq.metal`
//! (ROADMAP Phase S): `dequant_iq4_nl_gemv_simd`, `dequant_iq4_xs_gemv_simd`
//! and `dequant_iq3_xxs_gemv_simd`.
//!
//! PORT-LOCAL rather than vendored, like every other GGUF kernel here: the
//! Swift engine has no GGUF intake. The contract is
//! `turbospark_compute::quant_gguf_iq`, which
//! `crates/gpu/tests/dequant_iq_gemv_parity.rs` holds these to.
//!
//! ONE MODULE FOR THREE TYPES, where Q8_0, Q4_K and Q6_K each got their own.
//! They differ from the K-quants in the way that matters for a dispatch --
//! they decode through a codebook rather than an arithmetic reconstruction --
//! and they are identical to each other in every way a HOST can see: the same
//! 32-lane one-element-per-lane shape, the same eight rows per threadgroup,
//! the same bindings, no function constants. Splitting them three ways would
//! be three copies of `dequant_iq_gemv` differing in a string.
//!
//! The row-bytes helpers are the one place the types diverge here, and they
//! are the reason a caller cannot pass the wrong block size silently: each
//! asserts its own element count.
//!
//! These GEMVs exist for PARITY, not because a resident tensor needs them
//! today. The Phase S candidate carries its IQ types only in routed experts
//! (`moe_gguf.rs`), and its resident core is Q8_0 and Q6_K. They are cheap,
//! they are what a parity test can drive directly, and they are what a future
//! checkpoint putting IQ4_NL in an attention projection would need.

use half::f16;
use metal::FunctionConstantValues;

use crate::bytes::{half_slice_to_le_bytes, read_half_buffer, u32_bytes};
use crate::context::{dispatch_one_threadgroup_per_row, GpuError, MetalContext};

pub(crate) const SOURCE: &str = include_str!("shaders/dequant_iq.metal");
const THREADS_PER_GROUP: u64 = 256; // 8 rows/threadgroup * 32 lanes/SIMD group.
const ROWS_PER_THREADGROUP: u64 = 8;

/// IQ4_NL elements per block (32).
pub const IQ4_NL_BLOCK_ELEMS: usize = 32;
/// IQ4_NL bytes per block (18).
pub const IQ4_NL_BLOCK_BYTES: usize = 18;
/// IQ4_XS elements per block (256).
pub const IQ4_XS_BLOCK_ELEMS: usize = 256;
/// IQ4_XS bytes per block (136).
pub const IQ4_XS_BLOCK_BYTES: usize = 136;
/// IQ3_XXS elements per block (256).
pub const IQ3_XXS_BLOCK_ELEMS: usize = 256;
/// IQ3_XXS bytes per block (98).
pub const IQ3_XXS_BLOCK_BYTES: usize = 98;

/// Which of the three layouts a byte run is in.
///
/// A host-side enum rather than a Metal function constant because the kernels
/// differ in their inner loop, not in a parameter: MSL cannot select between
/// three different strides without branching per element.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IqBlockType {
    /// IQ4_NL block layout.
    Iq4Nl,
    /// IQ4_XS block layout.
    Iq4Xs,
    /// IQ3_XXS block layout.
    Iq3Xxs,
}

impl IqBlockType {
    pub(crate) fn kernel(self) -> &'static str {
        match self {
            IqBlockType::Iq4Nl => "dequant_iq4_nl_gemv_simd",
            IqBlockType::Iq4Xs => "dequant_iq4_xs_gemv_simd",
            IqBlockType::Iq3Xxs => "dequant_iq3_xxs_gemv_simd",
        }
    }

    pub(crate) fn block(self) -> (usize, usize) {
        match self {
            IqBlockType::Iq4Nl => (IQ4_NL_BLOCK_ELEMS, IQ4_NL_BLOCK_BYTES),
            IqBlockType::Iq4Xs => (IQ4_XS_BLOCK_ELEMS, IQ4_XS_BLOCK_BYTES),
            IqBlockType::Iq3Xxs => (IQ3_XXS_BLOCK_ELEMS, IQ3_XXS_BLOCK_BYTES),
        }
    }

    /// Bytes a row of `n` elements occupies in this layout.
    pub fn row_bytes(self, n: usize) -> usize {
        let (elems, bytes) = self.block();
        assert_eq!(
            n % elems,
            0,
            "N ({n}) is not a whole number of {elems}-element {self:?} blocks"
        );
        n / elems * bytes
    }
}

/// The kernels declare no function constants, the same deliberate choice the
/// Q8_0/Q4_K/Q6_K dispatches made: every affine quant kernel here specializes
/// M/N, and these shapes are not on a hot path (the routed pair in
/// `moe_gguf.rs` is).
fn no_function_constants() -> FunctionConstantValues {
    FunctionConstantValues::new()
}

/// `y[m] = sum_n W[m, n] * x[n]` over IQ rows. Each row is the raw GGUF byte
/// run for that output channel; every row must share the same `n`.
pub fn dequant_iq_gemv(
    context: &mut MetalContext,
    kind: IqBlockType,
    weight_rows: &[&[u8]],
    x: &[f16],
    n: usize,
) -> Result<Vec<f16>, GpuError> {
    assert!(!weight_rows.is_empty());
    assert_eq!(x.len(), n);
    let row_bytes = kind.row_bytes(n);

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
    let pipeline = context.pipeline(SOURCE, kind.kernel(), &no_function_constants(), b"")?;
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
