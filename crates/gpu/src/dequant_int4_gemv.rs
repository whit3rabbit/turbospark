//! Host-side dispatch for the `dequant_int4_gemv_simd` kernel in
//! `shaders/dequant_int4.metal` (vendored verbatim from
//! `Metal/Quant/dequant_int4.metal`). Matches
//! `mrefrust_compute::dequant_int4_gemv` exactly: `y[m] = sum_n W[m,n] *
//! x[n]` over affine-INT4-packed rows with a group size of 64.
//!
//! `dequant_int4.metal` also ships `embed_lookup_int4` and
//! `dequant_int4_qkv_gemv_simd`; neither is dispatched here.

use half::f16;
use metal::{FunctionConstantValues, MTLDataType};

use crate::bytes::{half_slice_to_le_bytes, read_half_buffer, u16_slice_to_le_bytes, u32_bytes};
use crate::context::{dispatch_one_threadgroup_per_row, GpuError, MetalContext};

const SOURCE: &str = include_str!("shaders/dequant_int4.metal");
const THREADS_PER_GROUP: u64 = 256; // 8 rows/threadgroup * 32 lanes/SIMD group.
const ROWS_PER_THREADGROUP: u64 = 8;

/// `dequant_int4_gemv_simd` declares function constants `FC_INT4_M` (20,
/// uint), `FC_INT4_N` (21, uint), and `FC_INT4_USE_FC` (22, bool); setting
/// `FC_INT4_USE_FC = false` keeps the kernel on its runtime M/N arguments.
/// The sibling `dequant_int4_qkv_gemv_simd` kernel's own constants (23-26)
/// are not referenced by this one and need no specialization here.
fn unused_function_constants() -> FunctionConstantValues {
    let values = FunctionConstantValues::new();
    let zero: u32 = 0;
    let use_fc = false;
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 20);
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 21);
    values.set_constant_value_at_index((&use_fc as *const bool).cast(), MTLDataType::Bool, 22);
    values
}

/// One row of affine-INT4-packed weights, laid out exactly as
/// `mrefrust_compute::Int4AffineRow`: `N/2` packed-nibble bytes, `N/64` BF16
/// scale bit patterns, `N/64` BF16 bias bit patterns.
pub struct Int4AffineRowGpu<'a> {
    pub packed: &'a [u8],
    pub scales: &'a [u16],
    pub biases: &'a [u16],
}

/// `y[m] = sum_n W[m, n] * x[n]`, dispatched on the GPU via
/// `dequant_int4_gemv_simd`. Every row must share the same `n` (a multiple
/// of 64).
pub fn dequant_int4_gemv(
    context: &mut MetalContext,
    weight_rows: &[Int4AffineRowGpu<'_>],
    x: &[f16],
    n: usize,
) -> Result<Vec<f16>, GpuError> {
    assert!(!weight_rows.is_empty());
    assert_eq!(x.len(), n);
    assert_eq!(n % 64, 0, "N must be a multiple of 64");

    let m = weight_rows.len();
    let mut w_bytes = Vec::with_capacity(m * n / 2);
    let mut scale_bits = Vec::with_capacity(m * n / 64);
    let mut bias_bits = Vec::with_capacity(m * n / 64);
    for row in weight_rows {
        assert_eq!(row.packed.len(), n / 2);
        assert_eq!(row.scales.len(), n / 64);
        assert_eq!(row.biases.len(), n / 64);
        w_bytes.extend_from_slice(row.packed);
        scale_bits.extend_from_slice(row.scales);
        bias_bits.extend_from_slice(row.biases);
    }

    let w_buffer = context.new_buffer_with_data(&w_bytes);
    let scales_buffer = context.new_buffer_with_data(&u16_slice_to_le_bytes(&scale_bits));
    let biases_buffer = context.new_buffer_with_data(&u16_slice_to_le_bytes(&bias_bits));
    let x_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(x));
    let y_buffer = context.new_output_buffer((m * std::mem::size_of::<u16>()) as u64);

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let pipeline = context.pipeline(
        SOURCE,
        "dequant_int4_gemv_simd",
        &unused_function_constants(),
    )?;
    let threadgroups = m.div_ceil(ROWS_PER_THREADGROUP as usize) as u64;
    dispatch_one_threadgroup_per_row(
        context,
        &pipeline,
        &[
            (&w_buffer, 0),
            (&scales_buffer, 1),
            (&biases_buffer, 2),
            (&x_buffer, 3),
            (&y_buffer, 4),
        ],
        &[(u32_bytes(&m_u32), 5), (u32_bytes(&n_u32), 6)],
        threadgroups,
        THREADS_PER_GROUP,
    );

    Ok(read_half_buffer(&y_buffer, m))
}
