//! Host-side dispatch for `shaders/dequant_int4_mma8.metal`: the small-M
//! INT4 GEMM on `simdgroup_matrix` with register operands
//! (`docs/INT4_SMALL_M_MMA.md`). One call serves 1..=8 token rows at about
//! the cost of one; the shader header carries the exactness contract.
//!
//! Two things are baked per shape and never chosen per call, because each
//! changes the summation order and a row's bits must not depend on R: the
//! K split ([`int4_mma8_k_split`]) and the code mode (subnormal only). Both
//! of the GEMM's function constants and the split are in the pipeline key.

use metal::{FunctionConstantValues, MTLDataType};

use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::dequant_int4_gemv::Int4ResidentMatrix;

static SOURCE: &str = include_str!("shaders/dequant_int4_mma8.metal");

/// Token rows one call serves: one 8x8 tile's columns.
pub const INT4_MMA8_MAX_ROWS: usize = 8;

/// The K split for an `n x k` matrix: simdgroups per 8 output rows, each
/// over one slice of the 64-groups. A function of the shape ALONE, so
/// decode and verify on the same matrix sum in the same order.
///
/// From `int4_mma8_reg_bench.rs`'s `mma8_k_split_sweep` (M4 Max, AC): 4 is
/// best or within 2% at 1024+ row blocks (gate/up, packed q, 32768 rows), 8
/// at 640 (down, o_proj, MTP fc) and 128 (k/v), 16 below that (in_proj_a/b
/// at 48 rows). Capped at the group count so no slice is empty.
pub fn int4_mma8_k_split(n: usize, k: usize) -> usize {
    let blocks = n.div_ceil(8);
    let ks = if blocks >= 1024 {
        4
    } else if blocks >= 64 {
        8
    } else {
        16
    };
    ks.min(k / 64).max(1)
}

/// `y[t, m] = sum_n W[m, n] * x[t, n]` for `t` in `0..rows`, `rows` in
/// `1..=8`, with x and y TOKEN-MAJOR as in
/// [`crate::encode_dequant_int4_gemm_resident`]. Not bit-identical to that
/// kernel or to the GEMV; bit-identical to itself at any `rows`.
pub fn encode_dequant_int4_mma8_resident(
    context: &mut MetalContext,
    pass: &PassEncoder,
    w: &Int4ResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    rows: usize,
) -> Result<(), GpuError> {
    let (n, k) = (w.rows, w.cols);
    if !(1..=INT4_MMA8_MAX_ROWS).contains(&rows) {
        return Err(GpuError::PipelineCreate(format!(
            "int4 mma8: {rows} rows outside 1..={INT4_MMA8_MAX_ROWS}"
        )));
    }
    if k % 64 != 0 || n == 0 {
        return Err(GpuError::PipelineCreate(format!(
            "int4 mma8: {n}x{k} needs K a multiple of 64 and N > 0"
        )));
    }
    for (what, offset) in [
        ("weight", w.weights_offset),
        ("scale", w.scales_offset),
        ("bias", w.biases_offset),
        ("x", x.1),
        ("y", y.1),
    ] {
        if offset % 2 != 0 {
            return Err(GpuError::PipelineCreate(format!(
                "int4 mma8 needs 2-aligned {what} offsets, got {offset}"
            )));
        }
    }

    let groups = k / 64;
    let need = groups;
    let grow = context
        .int4_mma8_scratch
        .as_ref()
        .is_none_or(|(_, _, have)| *have < need);
    if grow {
        let table = context.new_output_buffer((groups * 32 * 16 * 2) as u64);
        let sums = context.new_output_buffer((groups * 8 * 4) as u64);
        context.int4_mma8_scratch = Some((table, sums, need));
    }
    let (table, sums, _) = context
        .int4_mma8_scratch
        .clone()
        .expect("scratch was just ensured");

    let (k32, r32, n32) = (k as u32, rows as u32, n as u32);
    let prep = context.pipeline(
        SOURCE,
        "dequant_int4_mma8_table",
        &FunctionConstantValues::new(),
        b"",
    )?;
    pass.encode_threadgroups(
        &prep,
        &[(x.0, 0, x.1), (&table, 1, 0), (&sums, 2, 0)],
        &[(u32_bytes(&k32), 3), (u32_bytes(&r32), 4)],
        groups as u64,
        32,
    );

    let ks = int4_mma8_k_split(n, k) as u32;
    let values = FunctionConstantValues::new();
    values.set_constant_value_at_index((&n32 as *const u32).cast(), MTLDataType::UInt, 40);
    values.set_constant_value_at_index((&k32 as *const u32).cast(), MTLDataType::UInt, 41);
    values.set_constant_value_at_index((&ks as *const u32).cast(), MTLDataType::UInt, 42);
    let mut key = Vec::with_capacity(12);
    key.extend_from_slice(&n32.to_le_bytes());
    key.extend_from_slice(&k32.to_le_bytes());
    key.extend_from_slice(&ks.to_le_bytes());
    let gemm = context.pipeline(SOURCE, "dequant_int4_mma8_gemm", &values, &key)?;
    pass.encode_threadgroups(
        &gemm,
        &[
            (w.buffer, 0, w.weights_offset),
            (w.buffer, 1, w.scales_offset),
            (w.buffer, 2, w.biases_offset),
            (&table, 3, 0),
            (&sums, 4, 0),
            (y.0, 5, y.1),
        ],
        &[(u32_bytes(&r32), 6)],
        n.div_ceil(8) as u64,
        (ks * 32) as u64,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::int4_mma8_k_split;

    #[test]
    fn the_k_split_is_a_function_of_the_shape_and_never_leaves_a_slice_empty() {
        assert_eq!(int4_mma8_k_split(17408, 5120), 4);
        assert_eq!(int4_mma8_k_split(5120, 17408), 8);
        assert_eq!(int4_mma8_k_split(1024, 5120), 8);
        assert_eq!(int4_mma8_k_split(48, 5120), 16);
        // 2 groups: never more slices than groups.
        assert_eq!(int4_mma8_k_split(48, 128), 2);
        assert_eq!(int4_mma8_k_split(1, 64), 1);
    }
}
