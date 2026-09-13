//! Host-side dispatch for `shaders/dflash_conv.metal`: DFlash2's grouped
//! dynamic depthwise convolution (`docs/DFLASH2.md`).
//!
//! One kernel serves all four call sites in a drafter layer (around
//! attention and around the MLP, `prepare` and `finish`), because the two
//! axes that distinguish them are arguments rather than code: `side`
//! selects the base-kernel half, and `delta_col_base` selects the half of
//! the kernel_projection GEMM's output that carries the matching dynamic
//! coefficients. `prepare` and `finish` are therefore the same dispatch
//! with `side = 0` / `1` and `delta_col_base = 0` / `taps * groups`.
//!
//! THE DELTA CONTRACT deserves spelling out because it is the one thing a
//! caller can get wrong silently. The reference computes
//! `kernel_projection(x).reshape(rows, 2, taps, groups)` and hands side 0's
//! half to `prepare` and side 1's to `finish`; the reshape means side s's
//! half is COLUMNS `[s * taps * groups, (s + 1) * taps * groups)` of each
//! row, so the GEMM output is bound whole and sliced by stride/base here.
//! A caller that swaps the halves produces a plausible drafter with
//! crossed coefficients, not an error.

use metal::FunctionConstantValues;

use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext, PassEncoder};

const SOURCE: &str = include_str!("shaders/dflash_conv.metal");
const THREADS_PER_GROUP: u64 = 256;

/// The drafter's conv tap count, read off the published checkpoint
/// (`conv_kernel_size`). A constant rather than an argument because no
/// other value exists to pass: the kernel loops `taps` generically, the
/// host always hands it 2.
pub const DFLASH_TAPS: u32 = 2;

/// The drafter's conv channel-group size (`conv_group_size`).
pub const DFLASH_GROUP_SIZE: u32 = 16;

/// Encodes one conv dispatch.
///
/// * `x`, `out`: `[rows * hidden]` FP16, row-major.
/// * `delta`: the kernel_projection GEMM's whole output; `delta_row_stride`
///   is its row width (`2 * taps * groups`) and `delta_col_base` the side's
///   column offset within each row (`side * taps * groups`).
/// * `base`: the `[2, taps, hidden]` BF16 `base_kernel`, bound by offset.
/// * `side`: 0 for `prepare`'s convolution, 1 for `finish`'s.
/// * `out_scale`: multiplies the result. `1.0` reproduces the reference
///   exactly; a `finish` call passes `1 / DFLASH_RESIDUAL_SCALE` so the
///   residual stream it feeds fits FP16 (see the shader's header).
#[allow(clippy::too_many_arguments)]
pub fn encode_dflash_grouped_conv(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: (&metal::Buffer, u64),
    delta: (&metal::Buffer, u64),
    base: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    rows: u32,
    hidden: u32,
    side: u32,
    out_scale: f32,
) -> Result<(), GpuError> {
    // AGENTS.md/CLAUDE.md S12: `hidden / DFLASH_GROUP_SIZE` truncates on a
    // non-multiple, and the top channels would then read the next tap's
    // columns -- a plausible, wrong drafter with no error anywhere.
    if hidden % DFLASH_GROUP_SIZE != 0 {
        return Err(GpuError::PipelineCreate(format!(
            "dflash conv hidden {hidden} must be a multiple of DFLASH_GROUP_SIZE ({DFLASH_GROUP_SIZE})"
        )));
    }
    let groups = hidden / DFLASH_GROUP_SIZE;
    let delta_row_stride = 2 * DFLASH_TAPS * groups;
    let delta_col_base = side * DFLASH_TAPS * groups;
    let count = rows * hidden;
    let pipeline = context.pipeline(
        SOURCE,
        "dflash_grouped_conv_fp16",
        &FunctionConstantValues::new(),
        b"",
    )?;
    // AGENTS.md/CLAUDE.md S12: floored at 1, matching rope.rs's own
    // convention, rather than dispatching a zero grid at count == 0.
    let grid = (count as u64).div_ceil(THREADS_PER_GROUP).max(1) * THREADS_PER_GROUP;
    pass.encode_threads_3d(
        &pipeline,
        &[
            (x.0, 0, x.1),
            (delta.0, 1, delta.1),
            (base.0, 2, base.1),
            (out.0, 3, out.1),
        ],
        &[
            (u32_bytes(&rows), 4),
            (u32_bytes(&hidden), 5),
            (u32_bytes(&DFLASH_TAPS), 6),
            (u32_bytes(&DFLASH_GROUP_SIZE), 7),
            (u32_bytes(&delta_row_stride), 8),
            (u32_bytes(&delta_col_base), 9),
            (u32_bytes(&side), 10),
            (crate::bytes::f32_bytes(&out_scale), 11),
        ],
        (grid, 1, 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}

/// Encodes the aux-state CAPTURE: `dst[r * dst_stride + c] = src[r *
/// hidden + c]` for `r < rows`, `c < hidden`.
///
/// The source is always a row-major `[rows][hidden]` residual buffer, so
/// its stride is `hidden` by construction and not an argument; only the
/// destination strides (the capture buffer's `[row][aux][hidden]` layout).
/// After a tapped trunk layer completes, the residual rows land in the fc
/// projection's own input layout, so nothing rearranges them again.
pub fn encode_dflash_copy_rows(
    context: &mut MetalContext,
    pass: &PassEncoder,
    src: (&metal::Buffer, u64),
    dst: (&metal::Buffer, u64),
    rows: u32,
    hidden: u32,
    dst_stride: u32,
) -> Result<(), GpuError> {
    let count = rows
        .checked_mul(hidden)
        .expect("copy row element count overflows u32");
    let view_bytes = |buffer: &metal::Buffer, offset: u64, stride: u32| {
        let elements = if rows == 0 {
            0
        } else {
            u64::from(rows - 1)
                .checked_mul(u64::from(stride))
                .and_then(|n| n.checked_add(u64::from(hidden)))
                .expect("copy row view length overflows u64")
        };
        let end = elements
            .checked_mul(2)
            .and_then(|bytes| offset.checked_add(bytes))
            .expect("copy row byte range overflows u64");
        assert!(end <= buffer.length(), "copy row view exceeds buffer");
    };
    view_bytes(src.0, src.1, hidden);
    view_bytes(dst.0, dst.1, dst_stride);
    let pipeline = context.pipeline(
        SOURCE,
        "copy_strided_rows_fp16",
        &FunctionConstantValues::new(),
        b"",
    )?;
    // AGENTS.md/CLAUDE.md S12: floored at 1, matching rope.rs's own
    // convention, rather than dispatching a zero grid at count == 0.
    let grid = (count as u64).div_ceil(THREADS_PER_GROUP).max(1) * THREADS_PER_GROUP;
    let src_stride = hidden;
    pass.encode_threads_3d(
        &pipeline,
        &[(src.0, 0, src.1), (dst.0, 1, dst.1)],
        &[
            (u32_bytes(&rows), 2),
            (u32_bytes(&hidden), 3),
            (u32_bytes(&src_stride), 4),
            (u32_bytes(&dst_stride), 5),
        ],
        (grid, 1, 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}
