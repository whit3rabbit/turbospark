//! Host dispatch for `shaders/nemotron_diar.metal`: the one kernel the
//! Nemotron-3 diarization encoder needs beyond the whisper f32 set
//! (`whisper_encoder`). The runtime owns the buffers and sequences it with
//! those kernels in one pass. `tests/nemotron_diar_parity.rs` holds it to a
//! CPU reference of `ops::rope_neox`.

use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::rms_norm::unused_function_constants;
use crate::whisper_encoder::F32View;

const SOURCE: &str = include_str!("shaders/nemotron_diar.metal");
const THREADS: u64 = 256;

/// NeoX rotary of the first `rotary` lanes of every head in `rows` rows of
/// `heads * head_dim` values, from `[rows, rotary / 2]` cos/sin tables.
/// `x` rows are `x_stride` elements apart and `y` rows `y_stride`, so a
/// view into a fused QKV projection can be rotated into its own buffer.
/// `x` and `y` must not overlap.
#[allow(clippy::too_many_arguments)]
pub fn encode_rope_neox_rows(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: F32View,
    cos: F32View,
    sin: F32View,
    y: F32View,
    rows: u32,
    heads: u32,
    head_dim: u32,
    rotary: u32,
    x_stride: u32,
    y_stride: u32,
) -> Result<(), GpuError> {
    let width = heads
        .checked_mul(head_dim)
        .ok_or_else(|| GpuError::InvalidInput("rope width overflows".into()))?;
    if rows == 0 || width == 0 || rotary % 2 != 0 || rotary > head_dim {
        return Err(GpuError::InvalidInput(format!(
            "rope needs positive rows and width and an even rotary <= head_dim \
             (rows {rows}, heads {heads}, head_dim {head_dim}, rotary {rotary})"
        )));
    }
    if x_stride < width || y_stride < width {
        return Err(GpuError::InvalidInput(format!(
            "rope strides must cover the row (width {width}, x {x_stride}, y {y_stride})"
        )));
    }
    let total = rows
        .checked_mul(width)
        .ok_or_else(|| GpuError::InvalidInput("rope element count overflows".into()))?;
    let pipeline = context.pipeline(
        SOURCE,
        "nemotron_rope_neox_f32",
        &unused_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[x.binding(0), cos.binding(1), sin.binding(2), y.binding(3)],
        &[
            (u32_bytes(&rows), 4),
            (u32_bytes(&heads), 5),
            (u32_bytes(&head_dim), 6),
            (u32_bytes(&rotary), 7),
            (u32_bytes(&x_stride), 8),
            (u32_bytes(&y_stride), 9),
        ],
        u64::from(total).div_ceil(THREADS),
        THREADS,
    );
    Ok(())
}
