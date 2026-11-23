//! Host-side dispatch for `shaders/inkling2_batch.metal`: the multi-row
//! (speculative verify) forms of the Inkling 2 decode kernels.
//!
//! PORT-LOCAL. Row `r` of each kernel computes the bits the single-row kernel
//! it mirrors computes for that row (`tests/inkling2_batch_parity.rs`).
//! Strides are in ELEMENTS. Every entry point checks its shape and refuses
//! rather than dispatching out of contract.

use crate::bytes::u32_bytes;
use crate::{GpuError, MetalContext, PassEncoder};
use metal::FunctionConstantValues;

const SOURCE: &str = concat!(
    include_str!("shaders/moe.metal"),
    include_str!("shaders/inkling2_batch.metal")
);

/// Rows one batched dispatch serves: a chunked prefill panel. Speculative
/// verify uses at most 8 (the DSpark block, anchor + 7); past 8 the GEMV
/// takes its 32-accumulator twin.
pub const INKLING2_BATCH_MAX_ROWS: usize = 32;
/// Rows the 8-accumulator GEMV kernel serves (the verify block).
const GEMV_ROWS_SMALL: u32 = 8;

type Buf<'a> = (&'a metal::Buffer, u64);

fn invalid(msg: String) -> GpuError {
    GpuError::InvalidInput(msg)
}

fn rows_ok(n_rows: u32) -> Result<(), GpuError> {
    if n_rows == 0 || n_rows as usize > INKLING2_BATCH_MAX_ROWS {
        return Err(invalid(format!(
            "inkling2 batch: {n_rows} rows is outside 1..={INKLING2_BATCH_MAX_ROWS}"
        )));
    }
    Ok(())
}

fn pipeline(
    context: &mut MetalContext,
    kernel: &'static str,
) -> Result<metal::ComputePipelineState, GpuError> {
    context.pipeline(SOURCE, kernel, &FunctionConstantValues::new(), b"")
}

/// A row-addressed buffer: row `r` at `base + row(r) * stride` elements, where
/// `row(r) = (ring_base + r) % ring` when `ring > 0` (a KV ring slot) and `r`
/// otherwise.
#[derive(Clone, Copy)]
pub struct Inkling2Rows<'a> {
    pub buf: Buf<'a>,
    pub stride: u32,
    pub ring_base: u32,
    pub ring: u32,
}

impl<'a> Inkling2Rows<'a> {
    pub fn plain(buf: Buf<'a>, stride: u32) -> Self {
        Self {
            buf,
            stride,
            ring_base: 0,
            ring: 0,
        }
    }
}

/// `dst[r][0..n] = src[r][0..n]` for `n_rows` rows.
pub fn encode_inkling2_copy_rows(
    context: &mut MetalContext,
    pass: &PassEncoder,
    src: Buf<'_>,
    src_stride: u32,
    dst: Buf<'_>,
    dst_stride: u32,
    n: u32,
    n_rows: u32,
) -> Result<(), GpuError> {
    rows_ok(n_rows)?;
    if n == 0 {
        return Ok(());
    }
    let p = pipeline(context, "inkling2_copy_rows")?;
    pass.encode_threads_3d(
        &p,
        &[(src.0, 0, src.1), (dst.0, 1, dst.1)],
        &[
            (u32_bytes(&n), 2),
            (u32_bytes(&src_stride), 3),
            (u32_bytes(&dst_stride), 4),
        ],
        (u64::from(n), u64::from(n_rows), 1),
        (256, 1, 1),
    );
    Ok(())
}

/// RMS norm of `n_vec` vectors of `d` per row (`rmsnorm_bf16w` with a
/// weight, `rmsnorm_no_scale` / `_perhead` without), in and out row-addressed
/// (`out` may alias `x`). `min(256, d)` threads, the single-row width.
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_rms_rows(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: Inkling2Rows<'_>,
    weight: Option<Buf<'_>>,
    out: Inkling2Rows<'_>,
    d: u32,
    n_vec: u32,
    eps: f32,
    n_rows: u32,
) -> Result<(), GpuError> {
    rows_ok(n_rows)?;
    if d == 0 || n_vec == 0 || x.ring != out.ring || x.ring_base != out.ring_base {
        return Err(invalid(format!(
            "inkling2 rms rows: d {d}, n_vec {n_vec}, in/out ring addressing must agree"
        )));
    }
    let (w, has) = match weight {
        Some(w) => (w, 1u32),
        None => (x.buf, 0u32),
    };
    let p = pipeline(context, "inkling2_rms_rows")?;
    pass.encode_threadgroups_3d(
        &p,
        &[
            (x.buf.0, 0, x.buf.1),
            (w.0, 1, w.1),
            (out.buf.0, 2, out.buf.1),
        ],
        &[
            (u32_bytes(&d), 3),
            (&eps.to_le_bytes(), 4),
            (u32_bytes(&x.stride), 5),
            (u32_bytes(&out.stride), 6),
            (u32_bytes(&has), 7),
            (u32_bytes(&x.ring_base), 8),
            (u32_bytes(&x.ring), 9),
        ],
        (u64::from(n_vec), u64::from(n_rows), 1),
        (256u64.min(u64::from(d)), 1, 1),
    );
    Ok(())
}

/// `encode_inkling2_depth_mix` per row: row `r`'s states at
/// `states + r * states_stride`, its `n_q` outputs at `out + r * out_stride`.
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_depth_mix_rows(
    context: &mut MetalContext,
    pass: &PassEncoder,
    states: Buf<'_>,
    states_stride: u32,
    queries: Buf<'_>,
    out: Buf<'_>,
    out_stride: u32,
    n_states: u32,
    n_q: u32,
    h: u32,
    n_rows: u32,
) -> Result<(), GpuError> {
    rows_ok(n_rows)?;
    let n = n_states as usize;
    if n == 0 || n > crate::INKLING2_DEPTH_MAX_STATES || n_q == 0 || h == 0 {
        return Err(invalid(format!(
            "depth mix rows: {n} states, {n_q} queries, h {h}"
        )));
    }
    let p = pipeline(context, "inkling2_depth_mix_rows")?;
    pass.encode_threadgroups_3d(
        &p,
        &[
            (states.0, 0, states.1),
            (queries.0, 1, queries.1),
            (out.0, 2, out.1),
        ],
        &[
            (u32_bytes(&n_states), 3),
            (u32_bytes(&h), 4),
            (u32_bytes(&states_stride), 5),
            (u32_bytes(&out_stride), 6),
        ],
        (u64::from(n_q), u64::from(n_rows), 1),
        (256, 1, 1),
    );
    Ok(())
}

/// `encode_inkling2_control_combine` per row. `strides` are cur, msg,
/// gain/forget and out.
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_control_combine_rows(
    context: &mut MetalContext,
    pass: &PassEncoder,
    cur: Buf<'_>,
    msg: Buf<'_>,
    gain: Buf<'_>,
    forget: Buf<'_>,
    out: Buf<'_>,
    n: u32,
    strides: [u32; 4],
    n_rows: u32,
) -> Result<(), GpuError> {
    rows_ok(n_rows)?;
    if n == 0 {
        return Err(invalid("control combine rows needs n > 0".into()));
    }
    let mut sb = [0u8; 16];
    for (i, s) in strides.iter().enumerate() {
        sb[i * 4..i * 4 + 4].copy_from_slice(&s.to_le_bytes());
    }
    let p = pipeline(context, "inkling2_control_combine_rows")?;
    pass.encode_threads_3d(
        &p,
        &[
            (cur.0, 0, cur.1),
            (msg.0, 1, msg.1),
            (gain.0, 2, gain.1),
            (forget.0, 3, forget.1),
            (out.0, 4, out.1),
        ],
        &[(u32_bytes(&n), 5), (&sb, 6)],
        (u64::from(n), u64::from(n_rows), 1),
        (256, 1, 1),
    );
    Ok(())
}

/// One matrix of [`encode_inkling2_bf16_gemv_rows`]: as
/// [`crate::Inkling2GemvSegment`], with the output row stride.
#[derive(Debug, Clone, Copy)]
pub struct Inkling2GemvRowsSegment<'a> {
    pub w_offset: u64,
    pub rows: u32,
    pub cols: u32,
    pub x: usize,
    pub y: (&'a metal::Buffer, u64),
    pub y_stride: u32,
}

/// [`crate::encode_inkling2_bf16_gemv_multi`] over `n_rows` input rows, each
/// weight row read once: input `i`'s row `m` at `xs[i].0 + m * xs[i].1`
/// halfs, segment `s`'s output row `m` at `y + m * y_stride`.
pub fn encode_inkling2_bf16_gemv_rows(
    context: &mut MetalContext,
    pass: &PassEncoder,
    weights: &metal::Buffer,
    xs: &[(Buf<'_>, u32)],
    segments: &[Inkling2GemvRowsSegment<'_>],
    n_rows: u32,
) -> Result<(), GpuError> {
    rows_ok(n_rows)?;
    if segments.is_empty() || segments.len() > 8 {
        return Err(invalid(format!(
            "bf16 gemv rows takes 1 to 8 segments, got {}",
            segments.len()
        )));
    }
    if xs.is_empty() || xs.len() > 3 || xs.iter().any(|(x, s)| x.1 % 8 != 0 || s % 4 != 0) {
        return Err(invalid(
            "bf16 gemv rows takes 1 to 3 inputs at 8-byte-aligned offsets and strides".into(),
        ));
    }
    let small = n_rows <= GEMV_ROWS_SMALL;
    let (mut w_off, mut rows, mut cols, mut x_idx) = ([0u64; 8], [0u32; 8], [0u32; 8], [0u32; 8]);
    let mut y_stride = [0u32; 8];
    let mut tg_start = [0u32; 9];
    let mut groups = 0u32;
    for (i, s) in segments.iter().enumerate() {
        if s.rows == 0
            || s.cols == 0
            || s.cols % 4 != 0
            || s.w_offset % 8 != 0
            || s.y.1 % 2 != 0
            || s.x >= xs.len()
            || (n_rows > 1 && s.y_stride < s.rows)
            || xs[s.x].1 < s.cols && n_rows > 1
        {
            return Err(invalid(format!(
                "bf16 gemv rows segment {i}: rows {} cols {} y stride {} input {} of {}",
                s.rows,
                s.cols,
                s.y_stride,
                s.x,
                xs.len()
            )));
        }
        w_off[i] = s.w_offset;
        rows[i] = s.rows;
        cols[i] = s.cols;
        x_idx[i] = s.x as u32;
        y_stride[i] = s.y_stride;
        tg_start[i] = groups;
        groups += s.rows.div_ceil(8);
    }
    for g in &mut tg_start[segments.len()..] {
        *g = groups;
    }
    let mut x_stride = [0u32; 3];
    for (i, (_, s)) in xs.iter().enumerate() {
        x_stride[i] = *s;
    }
    // The shader's `Inkling2GemvRowsTable`: the 200-byte single-row table,
    // then n_rows, x_stride[3], y_stride[8].
    let mut bytes = Vec::with_capacity(248);
    for v in w_off {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in rows.iter().chain(&cols).chain(&x_idx).chain(&tg_start) {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes.extend_from_slice(&(segments.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&n_rows.to_le_bytes());
    for v in x_stride.iter().chain(&y_stride) {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    // Struct alignment is 8 (the ulongs): pad to a multiple of 8.
    while bytes.len() % 8 != 0 {
        bytes.push(0);
    }
    let kernel = if small {
        "inkling2_bf16_gemv_rows"
    } else {
        "inkling2_bf16_gemv_rows32"
    };
    let p = pipeline(context, kernel)?;
    let mut buffers: Vec<(&metal::Buffer, u64, u64)> = vec![(weights, 0, 0)];
    for i in 0..3 {
        let (x, _) = xs[i.min(xs.len() - 1)];
        buffers.push((x.0, 2 + i as u64, x.1));
    }
    for i in 0..8 {
        let y = segments[i.min(segments.len() - 1)].y;
        buffers.push((y.0, 5 + i as u64, y.1));
    }
    pass.encode_threadgroups(&p, &buffers, &[(&bytes, 1)], u64::from(groups), 256);
    Ok(())
}

/// `encode_inkling2_short_conv_decode` over `n_rows` consecutive rows, the
/// tail advancing between rows. `snap` receives `n_rows + 1` tails of
/// `(taps - 1) * ch` halfs: entry 0 the block-start tail, entry `r + 1` the
/// tail after row `r` (the rollback snapshots). `dynamic` is
/// `(dyn rows, row stride, heads)` for the full-attention K/V form. `out` is
/// row-addressed (a KV ring slot).
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_short_conv_rows(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: Buf<'_>,
    x_stride: u32,
    tail: Buf<'_>,
    weight: Buf<'_>,
    dynamic: Option<(Buf<'_>, u32, u32)>,
    out: Inkling2Rows<'_>,
    snap: Buf<'_>,
    ch: u32,
    taps: u32,
    n_rows: u32,
) -> Result<(), GpuError> {
    rows_ok(n_rows)?;
    if ch == 0 || !(2..=8).contains(&taps) {
        return Err(invalid(format!(
            "short conv rows needs ch > 0 and taps in 2..=8, got ch {ch} taps {taps}"
        )));
    }
    let (dyn_buf, dyn_stride, heads) = match dynamic {
        Some((b, stride, heads)) => {
            if heads == 0 || ch % heads != 0 {
                return Err(invalid(format!(
                    "short conv rows heads ({heads}) must divide ch ({ch})"
                )));
            }
            (b, stride, heads)
        }
        None => (weight, 0, 0),
    };
    let p = pipeline(context, "inkling2_short_conv_rows")?;
    pass.encode_threadgroups(
        &p,
        &[
            (x.0, 0, x.1),
            (tail.0, 1, tail.1),
            (weight.0, 2, weight.1),
            (dyn_buf.0, 3, dyn_buf.1),
            (out.buf.0, 4, out.buf.1),
            (snap.0, 5, snap.1),
        ],
        &[
            (u32_bytes(&ch), 6),
            (u32_bytes(&taps), 7),
            (u32_bytes(&heads), 8),
            (u32_bytes(&n_rows), 9),
            (u32_bytes(&x_stride), 10),
            (u32_bytes(&dyn_stride), 11),
            (u32_bytes(&out.ring_base), 12),
            (u32_bytes(&out.ring), 13),
            (u32_bytes(&out.stride), 14),
        ],
        u64::from(ch).div_ceil(256),
        256,
    );
    Ok(())
}

/// `encode_inkling2_l2_clip` over `n_vec` vectors in each of `post.len()`
/// rows, row `r` with post scale `post[r]`; `x` is row-addressed.
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_l2_clip_rows(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: Inkling2Rows<'_>,
    scale: Option<(Buf<'_>, u32)>,
    n_vec: u32,
    dim: u32,
    threshold: f32,
    post: &[f32],
) -> Result<(), GpuError> {
    let n_rows = post.len() as u32;
    rows_ok(n_rows)?;
    if n_vec == 0 || dim == 0 {
        return Err(invalid("L2 clip rows needs n_vec and dim > 0".into()));
    }
    let (scale_buf, stride, has) = match scale {
        Some((buf, stride)) => {
            if stride != 0 && stride != dim {
                return Err(invalid(format!(
                    "L2 clip rows scale stride must be 0 or dim ({dim}), got {stride}"
                )));
            }
            (buf, stride, 1u32)
        }
        None => (x.buf, 0, 0u32),
    };
    let mut pb = [0u8; 4 * INKLING2_BATCH_MAX_ROWS];
    for (i, v) in post.iter().enumerate() {
        pb[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    let p = pipeline(context, "inkling2_l2_clip_rows")?;
    pass.encode_threadgroups_3d(
        &p,
        &[(x.buf.0, 0, x.buf.1), (scale_buf.0, 1, scale_buf.1)],
        &[
            (u32_bytes(&n_vec), 2),
            (u32_bytes(&dim), 3),
            (u32_bytes(&stride), 4),
            (u32_bytes(&has), 5),
            (&threshold.to_le_bytes(), 6),
            (&pb, 7),
            (u32_bytes(&x.ring_base), 8),
            (u32_bytes(&x.ring), 9),
            (u32_bytes(&x.stride), 10),
        ],
        (u64::from(n_vec).div_ceil(8), u64::from(n_rows), 1),
        (256, 1, 1),
    );
    Ok(())
}

/// `encode_inkling2_rel_table` per row: `r + row * r_stride` into
/// `table + row * heads * extent` (F32).
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_rel_table_rows(
    context: &mut MetalContext,
    pass: &PassEncoder,
    r: Buf<'_>,
    r_stride: u32,
    proj: Buf<'_>,
    table: Buf<'_>,
    heads: u32,
    d_rel: u32,
    extent: u32,
    n_rows: u32,
) -> Result<(), GpuError> {
    rows_ok(n_rows)?;
    if heads == 0 || d_rel == 0 || extent == 0 {
        return Err(invalid(
            "rel table rows needs heads, d_rel, extent > 0".into(),
        ));
    }
    let p = pipeline(context, "inkling2_rel_table_rows")?;
    pass.encode_threads_3d(
        &p,
        &[(r.0, 0, r.1), (proj.0, 1, proj.1), (table.0, 2, table.1)],
        &[
            (u32_bytes(&heads), 3),
            (u32_bytes(&d_rel), 4),
            (u32_bytes(&extent), 5),
            (u32_bytes(&r_stride), 6),
        ],
        (u64::from(heads) * u64::from(extent), u64::from(n_rows), 1),
        (256, 1, 1),
    );
    Ok(())
}

/// `encode_inkling2_attention_decode` for rows at positions `pos0 ..
/// pos0 + n_rows`, each over its own range (`window` 0: from position 0, else
/// the last `window` positions). `q` and `out` rows are `heads * head_dim`
/// apart, `sinks` rows `heads`, `table` rows `heads * extent` (F32). Every
/// row's K/V must already be in the ring, and the ring must hold the oldest
/// position row 0 reads and the newest row: `ring >= span`, where span is
/// `window + n_rows - 1` on a window slot and `pos0 + n_rows` on a full one.
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_attention_rows(
    context: &mut MetalContext,
    pass: &PassEncoder,
    q: Buf<'_>,
    k: Buf<'_>,
    v: Buf<'_>,
    table: Buf<'_>,
    sinks: Buf<'_>,
    out: Buf<'_>,
    window: u32,
    pos0: u32,
    shape: crate::Inkling2AttnShape,
    n_rows: u32,
) -> Result<(), GpuError> {
    rows_ok(n_rows)?;
    let s = shape;
    let last = pos0 + n_rows - 1;
    let first0 = if window == 0 || pos0 < window {
        0
    } else {
        pos0 + 1 - window
    };
    if s.kv_heads == 0
        || s.heads % s.kv_heads != 0
        || s.head_dim == 0
        || s.head_dim % 32 != 0
        || s.head_dim > crate::INKLING2_ATTN_MAX_HEAD_DIM
        || s.extent == 0
        || s.ring == 0
        || last - first0 >= s.ring
    {
        return Err(invalid(format!(
            "Inkling 2 attention rows: shape {s:?}, positions {first0}..={last} do not fit the ring"
        )));
    }
    let group = s.heads / s.kv_heads;
    let q_stride = s.heads * s.head_dim;
    let p = pipeline(context, "inkling2_attention_rows")?;
    pass.encode_threadgroups_3d(
        &p,
        &[
            (q.0, 0, q.1),
            (k.0, 1, k.1),
            (v.0, 2, v.1),
            (table.0, 3, table.1),
            (sinks.0, 4, sinks.1),
            (out.0, 5, out.1),
        ],
        &[
            (u32_bytes(&window), 6),
            (u32_bytes(&pos0), 7),
            (u32_bytes(&s.kv_heads), 8),
            (u32_bytes(&group), 9),
            (u32_bytes(&s.head_dim), 10),
            (u32_bytes(&s.extent), 11),
            (u32_bytes(&s.ring), 12),
            (u32_bytes(&q_stride), 13),
            (u32_bytes(&s.heads), 14),
        ],
        (u64::from(s.heads), u64::from(n_rows), 1),
        (256, 1, 1),
    );
    Ok(())
}

/// `encode_minimax_router` per row: F32 `logits[r][experts]` from
/// `x + r * x_stride`.
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_router_rows(
    context: &mut MetalContext,
    pass: &PassEncoder,
    weights: Buf<'_>,
    x: Buf<'_>,
    x_stride: u32,
    out: Buf<'_>,
    experts: u32,
    hidden: u32,
    n_rows: u32,
) -> Result<(), GpuError> {
    rows_ok(n_rows)?;
    let p = pipeline(context, "inkling2_router_rows")?;
    pass.encode_threadgroups_3d(
        &p,
        &[(weights.0, 0, weights.1), (x.0, 1, x.1), (out.0, 2, out.1)],
        &[
            (u32_bytes(&hidden), 3),
            (u32_bytes(&x_stride), 4),
            (u32_bytes(&experts), 5),
        ],
        (u64::from(experts), u64::from(n_rows), 1),
        (32, 1, 1),
    );
    Ok(())
}

/// `encode_inkling2_router_select` per row: logits `[rows][num_experts]`,
/// outputs `[rows][8]`.
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_router_select_rows(
    context: &mut MetalContext,
    pass: &PassEncoder,
    logits: Buf<'_>,
    correction_bias: Buf<'_>,
    out_indices: Buf<'_>,
    out_weights: Buf<'_>,
    num_experts: u32,
    route_scale: f32,
    n_rows: u32,
) -> Result<(), GpuError> {
    rows_ok(n_rows)?;
    let ne = num_experts as usize;
    if !(crate::INKLING2_ROUTER_TOP_K..=crate::INKLING2_ROUTER_MAX_EXPERTS).contains(&ne) {
        return Err(invalid(format!("router select rows: {ne} experts")));
    }
    let p = pipeline(context, "inkling2_router_select_rows")?;
    pass.encode_threadgroups(
        &p,
        &[
            (logits.0, 0, logits.1),
            (correction_bias.0, 1, correction_bias.1),
            (out_indices.0, 2, out_indices.1),
            (out_weights.0, 3, out_weights.1),
        ],
        &[
            (u32_bytes(&num_experts), 4),
            (&route_scale.to_le_bytes(), 5),
        ],
        u64::from(n_rows),
        32,
    );
    Ok(())
}
