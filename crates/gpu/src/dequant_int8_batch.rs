//! Host-side dispatch for `dequant_int8_gemm_simd`, the INT8-affine sibling
//! of `dequant_int4_batch.rs`'s batched kernel (see that file's module doc
//! for why one dispatch must hold B accumulators, and the shader's header
//! for the two lines that actually differ).
//!
//! PORT-LOCAL: the Swift engine decodes one token per forward pass, so
//! there is no upstream kernel and no vendoring question. Its only contract
//! is the INT8 GEMV's own -- B rows of this must equal B separate
//! `dequant_int8_gemv_resident` calls, which is what the parity test
//! asserts, and through that `turbospark_compute::dequant_int8_gemv`.
//!
//! The row-block table is the INT4 sibling's, borrowed rather than
//! re-measured: `FC_GEMM_R` is proven not to move a bit at any width (that
//! proof lives on the sibling and its argument carries over -- only which
//! SIMD group owns a row changes), so borrowing the table cannot move a
//! byte. Whether it is the right THROUGHPUT table for INT8 is a named
//! open question (see the shader header); nothing here claims it.

use metal::{FunctionConstantValues, MTLDataType};

use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext};
use crate::dequant_int8_gemv::Int8ResidentMatrix;

/// Gotcha 4: one `&'static str` with one stable address, because the
/// pipeline cache keys on the address.
const SOURCE: &str = concat!(
    include_str!("shaders/dequant_int8.metal"),
    include_str!("shaders/dequant_int8_batch.metal")
);

/// The INT4 sibling's dispatch shape, carried over unchanged.
pub const INT8_GEMM_THREADS_PER_GROUP: u64 = 256;
const SIMDGROUPS_PER_THREADGROUP: usize = 8;

/// The accumulator array is a fixed-size register array, so this is a hard
/// precondition, not a clamp. Same value and reason as the INT4 kernel.
pub const INT8_MAX_BATCH_ROWS: usize = 16;

/// Output rows one SIMD group may own (`FC_INT8_GEMM_R`), mirroring the
/// shader's `kInt8MaxRowBlock`. Bit-exact at every width; see the sibling.
pub const INT8_MAX_GEMM_ROW_BLOCK: usize = 4;

/// The INT8 sibling's [`best_row_block`]: the same steps the INT4
/// measurement produced. NOT a measurement of this kernel; see the module
/// doc and the shader header.
pub fn best_row_block_int8(batch: usize) -> usize {
    match batch {
        0 | 1 => 1,
        2..=4 => 2,
        _ => 4,
    }
}

/// `dequant_int8_gemm_simd` declares `FC_INT8_GEMM_M` (100), `FC_INT8_GEMM_N`
/// (101), `FC_INT8_GEMM_B` (102), `FC_INT8_GEMM_USE_FC` (103) and
/// `FC_INT8_GEMM_R` (104). Indices 70-74 belong to `dequant_int8.metal`,
/// which this source is concatenated onto, and are set to their
/// unspecialized values because the shared file declares them.
///
/// THE KEY MUST CARRY THE BAKED VALUES, for the reason both siblings
/// record: `MetalContext::pipeline` caches on (source address, name, key),
/// so a shared key silently serves the first-compiled shape's pipeline to
/// every later one, and a wrong baked N reads the wrong fraction of every
/// row while producing finite, plausible output (crate Gotcha 1).
fn specialized_constants_row_blocked(
    m: u32,
    n: u32,
    b: u32,
    r: u32,
) -> (FunctionConstantValues, [u8; 16]) {
    let values = FunctionConstantValues::new();
    let zero: u32 = 0;
    let off = false;
    let on = true;
    // The concatenated GEMV file's own constants, left unspecialized.
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 70);
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 71);
    values.set_constant_value_at_index((&off as *const bool).cast(), MTLDataType::Bool, 72);
    // This kernel's.
    values.set_constant_value_at_index((&m as *const u32).cast(), MTLDataType::UInt, 100);
    values.set_constant_value_at_index((&n as *const u32).cast(), MTLDataType::UInt, 101);
    values.set_constant_value_at_index((&b as *const u32).cast(), MTLDataType::UInt, 102);
    values.set_constant_value_at_index((&on as *const bool).cast(), MTLDataType::Bool, 103);
    values.set_constant_value_at_index((&r as *const u32).cast(), MTLDataType::UInt, 104);
    let mut key = [0u8; 16];
    key[..4].copy_from_slice(&m.to_le_bytes());
    key[4..8].copy_from_slice(&n.to_le_bytes());
    key[8..12].copy_from_slice(&b.to_le_bytes());
    key[12..].copy_from_slice(&r.to_le_bytes());
    (values, key)
}

/// `y[b, m] = sum_n W[m, n] * x[b, n]` for `b` in `0..batch`, INT8-affine
/// weights. `x` holds `batch * w.cols` halfs and `y` `batch * w.rows`,
/// both TOKEN-MAJOR.
pub fn encode_dequant_int8_gemm_resident(
    context: &mut MetalContext,
    pass: &crate::context::PassEncoder,
    w: &Int8ResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    batch: usize,
) -> Result<(), GpuError> {
    encode_dequant_int8_gemm_resident_blocked(
        context,
        pass,
        w,
        x,
        y,
        batch,
        best_row_block_int8(batch),
    )
}

/// [`encode_dequant_int8_gemm_resident`] with `row_block` output rows per
/// SIMD group ([`INT8_MAX_GEMM_ROW_BLOCK`]). Not a second numerics path:
/// every width is bit-exact against the GEMV, which the parity test
/// asserts across the whole grid rather than only the chosen width.
#[allow(clippy::too_many_arguments)]
pub fn encode_dequant_int8_gemm_resident_blocked(
    context: &mut MetalContext,
    pass: &crate::context::PassEncoder,
    w: &Int8ResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    batch: usize,
    row_block: usize,
) -> Result<(), GpuError> {
    assert_eq!(w.cols % 64, 0, "N must be a multiple of 64");
    assert!(w.rows > 0);
    assert!(
        (1..=INT8_MAX_BATCH_ROWS).contains(&batch),
        "batch {batch} outside 1..={INT8_MAX_BATCH_ROWS}"
    );
    assert!(
        (1..=INT8_MAX_GEMM_ROW_BLOCK).contains(&row_block),
        "row_block {row_block} outside 1..={INT8_MAX_GEMM_ROW_BLOCK}"
    );
    let (m, n, b, r) = (w.rows as u32, w.cols as u32, batch as u32, row_block as u32);
    let (constants, key) = specialized_constants_row_blocked(m, n, b, r);
    let pipeline = context.pipeline(SOURCE, "dequant_int8_gemm_simd", &constants, &key)?;
    pass.encode_threadgroups(
        &pipeline,
        &[
            (w.buffer, 0, w.weights_offset),
            (w.buffer, 1, w.scales_offset),
            (w.buffer, 2, w.biases_offset),
            (x.0, 3, x.1),
            (y.0, 4, y.1),
        ],
        &[(u32_bytes(&m), 5), (u32_bytes(&n), 6), (u32_bytes(&b), 7)],
        gemm_threadgroups_int8(w.rows, row_block),
        INT8_GEMM_THREADS_PER_GROUP,
    );
    Ok(())
}

/// Threadgroups for `rows` output rows at `row_block` rows per SIMD group.
///
/// A NAMED FUNCTION FOR THE REASON THE SIBLING'S `gemm_threadgroups` RECORDS:
/// dropping the `* row_block` over-dispatches, the surplus threadgroups
/// return at the kernel's first branch, and the output stays bit-correct
/// while the COST moves by `row_block` -- which would read as "row blocking
/// does not help". Asserted as arithmetic in this module's own tests.
fn gemm_threadgroups_int8(rows: usize, row_block: usize) -> u64 {
    rows.div_ceil(SIMDGROUPS_PER_THREADGROUP * row_block) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row is covered, and NO threadgroup is launched that has no row
    /// to do. The sibling's own test carries the full mutation account.
    #[test]
    fn a_dispatch_covers_every_row_and_launches_no_idle_threadgroup() {
        for row_block in 1..=INT8_MAX_GEMM_ROW_BLOCK {
            let per_group = SIMDGROUPS_PER_THREADGROUP * row_block;
            for rows in [
                1,
                7,
                per_group - 1,
                per_group,
                per_group + 1,
                72,
                1024,
                15360,
                3840,
            ] {
                let groups = gemm_threadgroups_int8(rows, row_block) as usize;
                assert!(
                    groups * per_group >= rows,
                    "rows {rows} row_block {row_block}: {groups} groups cover only \
                     {} rows, so {} rows are never written",
                    groups * per_group,
                    rows - groups * per_group
                );
                assert!(
                    groups.saturating_sub(1) * per_group < rows,
                    "rows {rows} row_block {row_block}: {groups} groups, but {} would \
                     already cover them. The surplus returns at the kernel's first \
                     branch, so the output is correct and the dispatch costs {}x too \
                     much",
                    groups - 1,
                    groups as f64 / (groups - 1).max(1) as f64
                );
            }
        }
    }

    /// The host caps must agree with the shader's `constexpr` array bounds.
    #[test]
    fn the_host_caps_match_the_shaders_array_bounds() {
        assert!(SOURCE.contains(&format!(
            "constant constexpr uint kInt8MaxBatchRows = {INT8_MAX_BATCH_ROWS};"
        )));
        assert!(SOURCE.contains(&format!(
            "constant constexpr uint kInt8MaxRowBlock = {INT8_MAX_GEMM_ROW_BLOCK};"
        )));
    }

    /// The borrowed width table stays dispatchable at every reachable batch.
    #[test]
    fn every_reachable_batch_chooses_a_dispatchable_width() {
        for batch in 1..=INT8_MAX_BATCH_ROWS {
            let r = best_row_block_int8(batch);
            assert!(
                (1..=INT8_MAX_GEMM_ROW_BLOCK).contains(&r),
                "batch {batch} chose row_block {r}, outside 1..={INT8_MAX_GEMM_ROW_BLOCK}"
            );
        }
    }
}
