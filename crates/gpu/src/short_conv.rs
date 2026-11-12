//! Host-side dispatch for `shaders/short_conv.metal`: LFM3's gated short
//! convolution (`Lfm3ShortConv`). PORT-LOCAL, no Swift original; the
//! contract is `turbospark_compute::short_conv_reference`.
//!
//! Per conv layer the flow is `in_proj` GEMV (`[3H]`, `B | C | xx`), then
//! one of these encoders (writes `C * conv(B * xx)` and advances the tail),
//! then the `out_proj` GEMV. The tail is `[K - 1, H]` FP16 holding the last
//! `K - 1` rows of `B * xx`, owned by [`crate::ShortConvStateManager`].
//!
//! Weights are BF16 `[H, K]` channel-major (`w[ch * K + j]`), the
//! checkpoint's dtype; the HF tensor is `[H, 1, K]`, which is the same bytes.

use metal::FunctionConstantValues;

use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext, PassEncoder};

const SOURCE: &str = include_str!("shaders/short_conv.metal");
const THREADS: u64 = 256;

const FP16_SIZE: u64 = 2;
const BF16_SIZE: u64 = 2;

fn pipeline(
    context: &mut MetalContext,
    name: &'static str,
) -> Result<metal::ComputePipelineState, GpuError> {
    // No function constants: `taps` and `hidden` are runtime arguments, so
    // there is nothing to put in the cache key.
    context.pipeline(SOURCE, name, &FunctionConstantValues::new(), b"")
}

/// Refuses a zero shape and any view that would run past its buffer. A
/// short in_proj or tail binding would otherwise read a neighbour's bytes
/// and produce plausible output.
fn validate(
    hidden: u32,
    taps: u32,
    rows: u32,
    views: &[(&str, (&metal::Buffer, u64), u64)],
) -> Result<(), GpuError> {
    if hidden == 0 || taps == 0 || rows == 0 {
        return Err(GpuError::PipelineCreate(format!(
            "short conv needs hidden, taps and rows >= 1; got hidden {hidden}, taps {taps}, rows {rows}"
        )));
    }
    for (name, (buffer, offset), bytes) in views {
        if offset
            .checked_add(*bytes)
            .is_none_or(|end| end > buffer.length())
        {
            return Err(GpuError::PipelineCreate(format!(
                "short conv {name} view ({bytes} bytes at {offset}) exceeds its buffer ({})",
                buffer.length()
            )));
        }
    }
    Ok(())
}

/// One decode row.
///
/// * `tail`: `[taps - 1, hidden]` FP16, shifted in place to include this
///   row's `B * xx`.
/// * `in_proj`: this row's `[3 * hidden]` FP16 in_proj output, `B | C | xx`.
/// * `weight`: `[hidden, taps]` BF16, channel-major.
/// * `out`: `[hidden]` FP16, `C * y`, the out_proj GEMV's input.
#[allow(clippy::too_many_arguments)]
pub fn encode_short_conv_decode(
    context: &mut MetalContext,
    pass: &PassEncoder,
    tail: (&metal::Buffer, u64),
    in_proj: (&metal::Buffer, u64),
    weight: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    hidden: u32,
    taps: u32,
) -> Result<(), GpuError> {
    let h = u64::from(hidden);
    let k = u64::from(taps);
    validate(
        hidden,
        taps,
        1,
        &[
            ("tail", tail, k.saturating_sub(1) * h * FP16_SIZE),
            ("in_proj", in_proj, 3 * h * FP16_SIZE),
            ("weight", weight, h * k * BF16_SIZE),
            ("out", out, h * FP16_SIZE),
        ],
    )?;
    let p = pipeline(context, "short_conv_decode")?;
    pass.encode_threads_3d(
        &p,
        &[
            (tail.0, 0, tail.1),
            (in_proj.0, 1, in_proj.1),
            (weight.0, 2, weight.1),
            (out.0, 3, out.1),
        ],
        &[(u32_bytes(&hidden), 4), (u32_bytes(&taps), 5)],
        (h, 1, 1),
        (THREADS, 1, 1),
    );
    Ok(())
}

/// `rows` consecutive rows, then the tail update, in two dispatches on
/// `pass`. Bit-identical to `rows` calls of [`encode_short_conv_decode`]:
/// output and resulting tail both.
///
/// * `in_proj`: `[rows, 3 * hidden]` FP16, row-major.
/// * `out`: `[rows, hidden]` FP16.
///
/// The second dispatch reads the tail the first one read, so `pass` must be
/// a serial pass (`MetalContext::begin_pass`), whose dispatches order on the
/// buffers they share.
#[allow(clippy::too_many_arguments)]
pub fn encode_short_conv_prefill(
    context: &mut MetalContext,
    pass: &PassEncoder,
    tail: (&metal::Buffer, u64),
    in_proj: (&metal::Buffer, u64),
    weight: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    hidden: u32,
    taps: u32,
    rows: u32,
) -> Result<(), GpuError> {
    let h = u64::from(hidden);
    let k = u64::from(taps);
    let t = u64::from(rows);
    validate(
        hidden,
        taps,
        rows,
        &[
            ("tail", tail, k.saturating_sub(1) * h * FP16_SIZE),
            ("in_proj", in_proj, t * 3 * h * FP16_SIZE),
            ("weight", weight, h * k * BF16_SIZE),
            ("out", out, t * h * FP16_SIZE),
        ],
    )?;
    let p = pipeline(context, "short_conv_prefill")?;
    pass.encode_threads_3d(
        &p,
        &[
            (tail.0, 0, tail.1),
            (in_proj.0, 1, in_proj.1),
            (weight.0, 2, weight.1),
            (out.0, 3, out.1),
        ],
        &[
            (u32_bytes(&hidden), 4),
            (u32_bytes(&taps), 5),
            (u32_bytes(&rows), 6),
        ],
        (h, t, 1),
        (THREADS, 1, 1),
    );
    if taps > 1 {
        let p = pipeline(context, "short_conv_tail_update")?;
        pass.encode_threads_3d(
            &p,
            &[(tail.0, 0, tail.1), (in_proj.0, 1, in_proj.1)],
            &[
                (u32_bytes(&hidden), 2),
                (u32_bytes(&taps), 3),
                (u32_bytes(&rows), 4),
            ],
            (h, 1, 1),
            (THREADS, 1, 1),
        );
    }
    Ok(())
}
