//! Host dispatch for `shaders/conformer_stream.metal`: the cache-aware
//! streaming Conformer kernels the Multitalker Parakeet encoder needs on
//! Metal beyond the whisper f32 set (GEMM, layer norm, add). Everything is
//! buffer-level and batched over speakers; the runtime owns the buffers and
//! each speaker's caches. `tests/conformer_stream_parity.rs` holds every
//! kernel to a CPU reference of `crates/audio`'s streaming encoder.

use crate::bytes::{f32_bytes, u32_bytes};
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::rms_norm::unused_function_constants;
use crate::whisper_encoder::F32View;

const SOURCE: &str = include_str!("shaders/conformer_stream.metal");
const THREADS: u64 = 256;
/// Threads per attention threadgroup (one (speaker, head, query) each).
const ATTN_THREADS: u64 = 128;
/// The attention kernel's threadgroup score and query capacity.
pub const MAX_STREAM_KEYS: usize = 256;
pub const MAX_STREAM_HEAD_DIM: usize = 256;

fn groups(count: u32) -> u64 {
    u64::from(count).div_ceil(THREADS)
}

fn product(parts: &[u32]) -> Result<u32, GpuError> {
    parts
        .iter()
        .try_fold(1u32, |acc, &p| acc.checked_mul(p))
        .ok_or_else(|| GpuError::InvalidInput("conformer stream shape overflows".into()))
}

/// `y = x * sigmoid(x)` over `count` floats (in place is fine).
pub fn encode_silu(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: F32View,
    y: F32View,
    count: u32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "conformer_silu_f32",
        &unused_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[x.binding(0), y.binding(1)],
        &[(u32_bytes(&count), 2)],
        groups(count),
        THREADS,
    );
    Ok(())
}

/// `dst = src` over `count` floats. The views must not overlap.
pub fn encode_copy(
    context: &mut MetalContext,
    pass: &PassEncoder,
    src: F32View,
    dst: F32View,
    count: u32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "conformer_copy_f32",
        &unused_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[src.binding(0), dst.binding(1)],
        &[(u32_bytes(&count), 2)],
        groups(count),
        THREADS,
    );
    Ok(())
}

/// GLU of `pw` (`[speakers * rows, 2 * channels]`) into rows `cache_rows..`
/// of each speaker's `[cache_rows + rows, channels]` block of `ext`; rows
/// at or past `valid` become zero.
#[allow(clippy::too_many_arguments)]
pub fn encode_glu_extend(
    context: &mut MetalContext,
    pass: &PassEncoder,
    pw: F32View,
    ext: F32View,
    speakers: u32,
    rows: u32,
    valid: u32,
    channels: u32,
    cache_rows: u32,
) -> Result<(), GpuError> {
    if valid > rows {
        return Err(GpuError::InvalidInput(format!(
            "GLU valid rows {valid} exceed the chunk's {rows}"
        )));
    }
    let total = product(&[speakers, rows, channels])?;
    let pipeline = context.pipeline(
        SOURCE,
        "conformer_glu_extend_f32",
        &unused_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[pw.binding(0), ext.binding(1)],
        &[
            (u32_bytes(&speakers), 2),
            (u32_bytes(&rows), 3),
            (u32_bytes(&valid), 4),
            (u32_bytes(&channels), 5),
            (u32_bytes(&cache_rows), 6),
        ],
        groups(total),
        THREADS,
    );
    Ok(())
}

/// Causal depthwise conv of each speaker's `[kernel - 1 + rows, channels]`
/// block of `ext` into `[speakers * rows, channels]` rows of `out`, with
/// `w` `[channels, kernel]` and an optional `[channels]` bias. `bias` is
/// bound either way; pass any valid view without one.
#[allow(clippy::too_many_arguments)]
pub fn encode_depthwise_causal(
    context: &mut MetalContext,
    pass: &PassEncoder,
    ext: F32View,
    w: F32View,
    bias: Option<F32View>,
    out: F32View,
    speakers: u32,
    rows: u32,
    channels: u32,
    kernel: u32,
) -> Result<(), GpuError> {
    if kernel == 0 {
        return Err(GpuError::InvalidInput(
            "depthwise kernel must be positive".into(),
        ));
    }
    let total = product(&[speakers, rows, channels])?;
    let pipeline = context.pipeline(
        SOURCE,
        "conformer_depthwise_causal_f32",
        &unused_function_constants(),
        b"",
    )?;
    let has_bias = u32::from(bias.is_some());
    pass.encode_threadgroups(
        &pipeline,
        &[
            ext.binding(0),
            w.binding(1),
            bias.unwrap_or(w).binding(2),
            out.binding(3),
        ],
        &[
            (u32_bytes(&speakers), 4),
            (u32_bytes(&rows), 5),
            (u32_bytes(&channels), 6),
            (u32_bytes(&kernel), 7),
            (u32_bytes(&has_bias), 8),
        ],
        groups(total),
        THREADS,
    );
    Ok(())
}

/// Shape of one [`encode_relpos_attention`] call.
#[derive(Clone, Copy, Debug)]
pub struct RelPosAttention {
    pub speakers: u32,
    /// Query rows per speaker (the chunk).
    pub rows: u32,
    /// Real query rows; later queries give zero rows.
    pub valid: u32,
    pub heads: u32,
    pub head_dim: u32,
    /// Attention cache rows before the chunk in each speaker's K/V block.
    pub attn_frames: u32,
    /// Rows the position table was built for (`attn_frames + chunk`).
    pub window: u32,
}

/// Cached relative-position attention for every (speaker, head, valid
/// query). `firsts` is a `[speakers]` u32 buffer of each speaker's first
/// valid key; keys end at `attn_frames + valid`.
#[allow(clippy::too_many_arguments)]
pub fn encode_relpos_attention(
    context: &mut MetalContext,
    pass: &PassEncoder,
    q: F32View,
    k: F32View,
    v: F32View,
    pos: F32View,
    bias_u: F32View,
    bias_v: F32View,
    firsts: &metal::Buffer,
    out: F32View,
    s: RelPosAttention,
) -> Result<(), GpuError> {
    if s.valid > s.rows || s.head_dim as usize > MAX_STREAM_HEAD_DIM || s.head_dim == 0 {
        return Err(GpuError::InvalidInput(format!(
            "relpos attention shape {s:?} exceeds the kernel"
        )));
    }
    // The kernel reads position row `window - 1 + key - attn_frames - query`
    // in u32 arithmetic. The smallest (key 0, last valid query) is
    // `window - attn_frames - valid`, so a window under `attn_frames + valid`
    // underflows into a read far outside the table.
    let keys = s.attn_frames.checked_add(s.valid);
    if keys.is_none_or(|keys| keys as usize > MAX_STREAM_KEYS || s.window < keys)
        || s.window < s.rows
    {
        return Err(GpuError::InvalidInput(format!(
            "relpos attention keys or window out of range: {s:?}"
        )));
    }
    let groups = product(&[s.speakers, s.heads, s.rows])?;
    let pipeline = context.pipeline(
        SOURCE,
        "conformer_relpos_attn_f32",
        &unused_function_constants(),
        b"",
    )?;
    let sqrt_head_dim = (s.head_dim as f32).sqrt();
    pass.encode_threadgroups(
        &pipeline,
        &[
            q.binding(0),
            k.binding(1),
            v.binding(2),
            pos.binding(3),
            bias_u.binding(4),
            bias_v.binding(5),
            (firsts, 6, 0),
            out.binding(7),
        ],
        &[
            (u32_bytes(&s.rows), 8),
            (u32_bytes(&s.valid), 9),
            (u32_bytes(&s.heads), 10),
            (u32_bytes(&s.head_dim), 11),
            (u32_bytes(&s.attn_frames), 12),
            (u32_bytes(&s.window), 13),
            (f32_bytes(&sqrt_head_dim), 14),
        ],
        u64::from(groups),
        ATTN_THREADS,
    );
    Ok(())
}
