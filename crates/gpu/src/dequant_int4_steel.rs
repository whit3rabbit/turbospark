//! Matrix-unit INT4 GEMM for prefill: Apple MLX's Steel `qmm_t_impl`,
//! vendored from MLX v0.31.1 (MIT) and adapted to this engine's resident
//! INT4 layout by `scripts/vendor_mlx_steel_qmm.py`, which also writes
//! `shaders/dequant_int4_steel.metal` and records every upstream file's
//! SHA-256 in its banner. Regenerate with that script rather than editing
//! the shader.
//!
//! **NOT BIT-EXACT AGAINST `dequant_int4_gemm_simd`.** Each weight is
//! dequantized in FP32 from its BF16 scale and bias, rounded once to an FP16
//! matrix operand, and reduced in the matrix units' order with FP32
//! accumulation. The scalar kernels keep the weight in FP32 and factor the
//! affine term out of the dot product. So this is prefill-only, reached
//! from a caller that has accepted the trade and is gated on model quality
//! (`docs/BATCHED_PREFILL.md`), never from M=1 decode or speculative verify.
//!
//! Why vendor rather than tile our own: the repository's hand-written MMA
//! kernel lost at every width (`ROADMAP.md` dead ends 13, 14 and 16), while
//! this one measured 1.4x the scalar batched kernel at 16 rows and 2.9x at
//! 32 in the isolated probe (`docs/MLX_KERNELS.md`).

use metal::FunctionConstantValues;

use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::dequant_int4_gemv::Int4ResidentMatrix;
use crate::dequant_int4_mma8::{encode_dequant_int4_mma8_resident, INT4_MMA8_MAX_ROWS};

pub(crate) static SOURCE: &str = include_str!("shaders/dequant_int4_steel.metal");

/// Threads per threadgroup: `qmm_t_impl`'s 2 x 2 simdgroup layout.
const THREADS_PER_GROUP: u64 = 128;
/// The quantization group the vendored loader is instantiated for.
const GROUP_SIZE: usize = 64;

/// Which instantiated tile (`BM` token rows, `BK` reduction depth, `BN`
/// output rows per threadgroup) to dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SteelTile {
    /// 32 x 32 x 32, the probe's best tile at 16 and 32 rows.
    M32K32N32,
    /// 64 x 32 x 64, for wider token batches.
    M64K32N64,
}

impl SteelTile {
    fn dims(self) -> (usize, usize, usize) {
        match self {
            Self::M32K32N32 => (32, 32, 32),
            Self::M64K32N64 => (64, 32, 64),
        }
    }

    fn function(self) -> &'static str {
        match self {
            Self::M32K32N32 => "dequant_int4_steel_32_32_32",
            Self::M64K32N64 => "dequant_int4_steel_64_32_64",
        }
    }
}

/// Whether `w` fits the vendored kernel at `tile`: the output rows must be a
/// whole number of `BN` tiles (the kernel is instantiated `aligned_N`), and
/// the input width a whole number of quantization groups and `BK` steps.
/// Callers fall back to `dequant_int4_gemm_simd` otherwise (Qwen3.8's
/// 48-row GDN `a`/`b` projections are the case in practice).
pub fn steel_supports(w: &Int4ResidentMatrix<'_>, tile: SteelTile) -> bool {
    let (_, bk, bn) = tile.dims();
    w.rows > 0
        && w.cols > 0
        && w.rows % bn == 0
        && w.cols % GROUP_SIZE == 0
        && w.cols % bk == 0
        && i32::try_from(w.rows).is_ok()
        && i32::try_from(w.cols).is_ok()
}

/// `y[batch, rows] = x[batch, cols] * dequant(w)^T` on the matrix units.
/// `x` and `y` are row-major FP16 `(buffer, byte offset)` views, `batch` is
/// any positive count (rows past it are masked inside the kernel).
pub fn encode_dequant_int4_gemm_steel(
    context: &mut MetalContext,
    pass: &PassEncoder,
    w: &Int4ResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    batch: usize,
    tile: SteelTile,
) -> Result<(), GpuError> {
    assert!(
        steel_supports(w, tile),
        "{} x {} does not fit the Steel tile {tile:?}",
        w.rows,
        w.cols
    );
    encode_steel(context, pass, w, x, y, batch, tile, tile.function())
}

/// Whether `w` fits the 32-row EDGE entry: the same checks as
/// [`steel_supports`] minus the whole-tile row count, so a matrix whose last
/// 32-row output tile is partial (Qwen3.8's 48-row GDN `a`/`b`) qualifies.
/// A separate predicate rather than a looser [`steel_supports`], because
/// callers of that one (the ANE split, the lab bench) rely on it meaning
/// "aligned".
pub fn steel_edge_supports(w: &Int4ResidentMatrix<'_>) -> bool {
    let (_, bk, _) = SteelTile::M32K32N32.dims();
    w.rows > 0
        && w.cols > 0
        && w.cols % GROUP_SIZE == 0
        && w.cols % bk == 0
        && i32::try_from(w.rows).is_ok()
        && i32::try_from(w.cols).is_ok()
}

/// [`encode_dequant_int4_gemm_steel`] at the 32 x 32 x 32 tile for a matrix
/// whose row count need not be a multiple of 32 (`aligned_N = false`: the
/// partial last tile loads its weights through `load_safe` and stores
/// masked). Full tiles run the aligned entry's arithmetic, so a row's bits
/// are the aligned kernel's whenever both apply.
pub fn encode_dequant_int4_gemm_steel_edge(
    context: &mut MetalContext,
    pass: &PassEncoder,
    w: &Int4ResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    batch: usize,
) -> Result<(), GpuError> {
    assert!(
        steel_edge_supports(w),
        "{} x {} does not fit the Steel edge tile",
        w.rows,
        w.cols
    );
    encode_steel(
        context,
        pass,
        w,
        x,
        y,
        batch,
        SteelTile::M32K32N32,
        "dequant_int4_steel_32_32_32_edge",
    )
}

/// Most output rows a matrix can have and still take the small-N route.
/// Qwen3.8's GDN `in_proj_a`/`in_proj_b` are 48; the next shapes up (the
/// 1024-row K/V projections) are whole Steel tiles and never reach it.
pub const SMALL_N_MAX_ROWS: usize = 64;

/// The widest batch the 8-row matrix-unit tile serves before the Steel edge
/// tile is cheaper. At 48 x 5120 the 8-row tile costs about 6.5 us per 8 rows
/// and the edge tile a flat ~99 us whatever the width, so they cross near 96
/// rows (`tests/steel_edge_n48_bench.rs`, M4 Max).
pub const SMALL_N_MMA8_MAX_BATCH: usize = 96;

/// An INT4 matrix [`encode_dequant_int4_gemm_small_n`] can serve: few output
/// rows, not a whole Steel tile (those take the aligned entry), and the layout
/// both kernels need (a whole number of 64-wide quantization groups, 2-aligned
/// plane offsets).
pub fn small_n_supports(w: &Int4ResidentMatrix<'_>) -> bool {
    w.rows <= SMALL_N_MAX_ROWS
        && w.rows % 32 != 0
        && steel_edge_supports(w)
        && [w.weights_offset, w.scales_offset, w.biases_offset]
            .iter()
            .all(|o| o % 2 == 0)
}

/// `y[batch, rows] = x[batch, cols] * w^T` for a [`small_n_supports`] matrix,
/// on the prefill path that has accepted Steel's rounding. Up to
/// [`SMALL_N_MMA8_MAX_BATCH`] rows run the 8-row matrix-unit tile in calls of
/// [`INT4_MMA8_MAX_ROWS`] (its K split is tuned for this shape); wider batches
/// run one Steel edge dispatch. Neither is bit-identical to the scalar GEMM.
pub fn encode_dequant_int4_gemm_small_n(
    context: &mut MetalContext,
    pass: &PassEncoder,
    w: &Int4ResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    batch: usize,
) -> Result<(), GpuError> {
    assert!(
        small_n_supports(w),
        "{} x {} does not fit the small-N route",
        w.rows,
        w.cols
    );
    assert!(batch > 0, "an empty batch has nothing to multiply");
    if batch > SMALL_N_MMA8_MAX_BATCH {
        return encode_dequant_int4_gemm_steel_edge(context, pass, w, x, y, batch);
    }
    let mut done = 0usize;
    while done < batch {
        let take = (batch - done).min(INT4_MMA8_MAX_ROWS);
        encode_dequant_int4_mma8_resident(
            context,
            pass,
            w,
            (x.0, x.1 + (done * w.cols * 2) as u64),
            (y.0, y.1 + (done * w.rows * 2) as u64),
            take,
        )?;
        done += take;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_steel(
    context: &mut MetalContext,
    pass: &PassEncoder,
    w: &Int4ResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    batch: usize,
    tile: SteelTile,
    function: &'static str,
) -> Result<(), GpuError> {
    assert!(batch > 0, "an empty batch has nothing to multiply");
    let batch_i32 = i32::try_from(batch).expect("batch fits i32");
    assert!(
        x.0.length() >= x.1 + (batch * w.cols * 2) as u64,
        "x holds fewer than {batch} rows of {} halfs",
        w.cols
    );
    assert!(
        y.0.length() >= y.1 + (batch * w.rows * 2) as u64,
        "y holds fewer than {batch} rows of {} halfs",
        w.rows
    );
    let (bm, _, bn) = tile.dims();
    let (k, n) = (w.cols as u32, w.rows as u32);
    let m = batch_i32 as u32;
    // No function constants: shape is a runtime argument, so one pipeline
    // per tile serves every projection and batch width.
    let pipeline = context.pipeline(SOURCE, function, &FunctionConstantValues::new(), &[])?;
    pass.encode_threadgroups_3d(
        &pipeline,
        &[
            (w.buffer, 0, w.weights_offset),
            (w.buffer, 1, w.scales_offset),
            (w.buffer, 2, w.biases_offset),
            (x.0, 3, x.1),
            (y.0, 4, y.1),
        ],
        &[(u32_bytes(&k), 5), (u32_bytes(&n), 6), (u32_bytes(&m), 7)],
        (w.rows.div_ceil(bn) as u64, batch.div_ceil(bm) as u64, 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}
