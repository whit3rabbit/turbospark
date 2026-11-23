//! Host-side dispatch for `shaders/inkling2_ops.metal`: the Inkling 2 router
//! selection, L2 clip, short convolution (decode) and depth mix.
//!
//! PORT-LOCAL. The math and layouts are in the shader header; the CPU contract
//! is `turbospark_compute::inkling2`, held by `tests/inkling2_ops_parity.rs`.
//! Every entry point checks its shape against the kernel's fixed limits and
//! refuses rather than dispatching out of contract, because each of these
//! kernels would otherwise read or write out of bounds silently.

use crate::bytes::u32_bytes;
use crate::{GpuError, MetalContext, PassEncoder};
use metal::FunctionConstantValues;

const SOURCE: &str = concat!(
    include_str!("shaders/moe.metal"),
    include_str!("shaders/inkling2_ops.metal")
);

/// Experts the router kernel can rank (32 lanes x 8 per lane).
pub const INKLING2_ROUTER_MAX_EXPERTS: usize = 256;
/// Experts the router selects; the Mini's `num_experts_per_tok`.
pub const INKLING2_ROUTER_TOP_K: usize = 8;
/// States the depth mix accepts (block outputs plus the current state).
pub const INKLING2_DEPTH_MAX_STATES: usize = 16;

fn invalid(msg: String) -> GpuError {
    GpuError::InvalidInput(msg)
}

fn f32_bytes(v: &f32) -> [u8; 4] {
    v.to_le_bytes()
}

fn pipeline(
    context: &mut MetalContext,
    kernel: &'static str,
) -> Result<metal::ComputePipelineState, GpuError> {
    context.pipeline(SOURCE, kernel, &FunctionConstantValues::new(), b"")
}

/// Sigmoid router selection: `out_indices[0..8]` / `out_weights[0..8]` in rank
/// order from FP32 `logits[num_experts]` and FP32 `correction_bias`.
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_router_select(
    context: &mut MetalContext,
    pass: &PassEncoder,
    logits: (&metal::Buffer, u64),
    correction_bias: (&metal::Buffer, u64),
    out_indices: (&metal::Buffer, u64),
    out_weights: (&metal::Buffer, u64),
    num_experts: u32,
    route_scale: f32,
) -> Result<(), GpuError> {
    let ne = num_experts as usize;
    if !(INKLING2_ROUTER_TOP_K..=INKLING2_ROUTER_MAX_EXPERTS).contains(&ne) {
        return Err(invalid(format!(
            "Inkling 2 router needs {INKLING2_ROUTER_TOP_K}..={INKLING2_ROUTER_MAX_EXPERTS} experts, got {ne}"
        )));
    }
    let p = pipeline(context, "inkling2_router_select_k8")?;
    pass.encode_threadgroups(
        &p,
        &[
            (logits.0, 0, logits.1),
            (correction_bias.0, 1, correction_bias.1),
            (out_indices.0, 2, out_indices.1),
            (out_weights.0, 3, out_weights.1),
        ],
        &[(u32_bytes(&num_experts), 4), (&f32_bytes(&route_scale), 5)],
        1,
        32,
    );
    Ok(())
}

/// Per-vector L2 clip in place over FP16 `x[n_vec, dim]`. `scale` is an
/// optional BF16 per-channel scale and its stride (0 = one `[dim]` scale for
/// every vector, `dim` = one per vector). `post_scale` multiplies after the
/// clip (the full-attention `tau(pos)`; pass 1.0 otherwise).
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_l2_clip(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: (&metal::Buffer, u64),
    scale: Option<((&metal::Buffer, u64), u32)>,
    n_vec: u32,
    dim: u32,
    threshold: f32,
    post_scale: f32,
) -> Result<(), GpuError> {
    if n_vec == 0 || dim == 0 {
        return Err(invalid("L2 clip needs n_vec and dim > 0".into()));
    }
    let (scale_buf, stride, has) = match scale {
        Some((buf, stride)) => {
            if stride != 0 && stride != dim {
                return Err(invalid(format!(
                    "L2 clip scale stride must be 0 or dim ({dim}), got {stride}"
                )));
            }
            (buf, stride, 1u32)
        }
        // Unread when has_scale is 0; bound so the argument table is complete.
        None => (x, 0, 0u32),
    };
    let p = pipeline(context, "inkling2_l2_clip")?;
    pass.encode_threadgroups(
        &p,
        &[(x.0, 0, x.1), (scale_buf.0, 1, scale_buf.1)],
        &[
            (u32_bytes(&n_vec), 2),
            (u32_bytes(&dim), 3),
            (u32_bytes(&stride), 4),
            (u32_bytes(&has), 5),
            (&f32_bytes(&threshold), 6),
            (&f32_bytes(&post_scale), 7),
        ],
        u64::from(n_vec).div_ceil(8),
        256,
    );
    Ok(())
}

/// One decode row of the causal short convolution with residual. `tail`
/// (FP16 `[taps - 1, ch]`, oldest first) is advanced in place. `dynamic` is
/// the full-attention K/V variant's per-head taps (FP16 `[heads, taps]`, lag
/// 0 = current row) and its head count; `None` is the static conv.
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_short_conv_decode(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: (&metal::Buffer, u64),
    tail: (&metal::Buffer, u64),
    weight: (&metal::Buffer, u64),
    dynamic: Option<((&metal::Buffer, u64), u32)>,
    out: (&metal::Buffer, u64),
    ch: u32,
    taps: u32,
) -> Result<(), GpuError> {
    if ch == 0 || taps < 2 {
        return Err(invalid(format!(
            "short conv needs ch > 0 and taps >= 2, got ch {ch} taps {taps}"
        )));
    }
    let (dyn_buf, heads) = match dynamic {
        Some((buf, heads)) => {
            if heads == 0 || ch % heads != 0 {
                return Err(invalid(format!(
                    "short conv heads ({heads}) must divide ch ({ch})"
                )));
            }
            (buf, heads)
        }
        // Unread when heads is 0.
        None => (weight, 0),
    };
    let p = pipeline(context, "inkling2_short_conv_decode")?;
    pass.encode_threadgroups(
        &p,
        &[
            (x.0, 0, x.1),
            (tail.0, 1, tail.1),
            (weight.0, 2, weight.1),
            (dyn_buf.0, 3, dyn_buf.1),
            (out.0, 4, out.1),
        ],
        &[
            (u32_bytes(&ch), 5),
            (u32_bytes(&taps), 6),
            (u32_bytes(&heads), 7),
        ],
        u64::from(ch).div_ceil(256),
        256,
    );
    Ok(())
}

/// What [`encode_inkling2_short_conv_residual`] does with the conv output.
#[derive(Clone, Copy, Debug)]
pub enum Inkling2ConvResidual<'a> {
    /// `cur += conv` (the `residual_add_fp16` arithmetic).
    Add,
    /// `save = cur; cur = conv`: the first layer of a residual block pushes
    /// the finished stream into the state array and starts a new one.
    StartBlock { save: (&'a metal::Buffer, u64) },
}

/// A static-tap decode short conv fused with the residual step after it.
/// Bit-identical to [`encode_inkling2_short_conv_decode`] then
/// `encode_residual_add` (or then the two copies), which
/// `tests/inkling2_fused_parity.rs` holds.
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_short_conv_residual(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: (&metal::Buffer, u64),
    tail: (&metal::Buffer, u64),
    weight: (&metal::Buffer, u64),
    cur: (&metal::Buffer, u64),
    step: Inkling2ConvResidual<'_>,
    ch: u32,
    taps: u32,
) -> Result<(), GpuError> {
    if ch == 0 || taps < 2 {
        return Err(invalid(format!(
            "short conv needs ch > 0 and taps >= 2, got ch {ch} taps {taps}"
        )));
    }
    let (save, mode) = match step {
        Inkling2ConvResidual::Add => (cur, 0u32),
        Inkling2ConvResidual::StartBlock { save } => (save, 1u32),
    };
    let p = pipeline(context, "inkling2_short_conv_residual")?;
    pass.encode_threadgroups(
        &p,
        &[
            (x.0, 0, x.1),
            (tail.0, 1, tail.1),
            (weight.0, 2, weight.1),
            (cur.0, 3, cur.1),
            (save.0, 4, save.1),
        ],
        &[
            (u32_bytes(&ch), 5),
            (u32_bytes(&taps), 6),
            (u32_bytes(&mode), 7),
        ],
        u64::from(ch).div_ceil(256),
        256,
    );
    Ok(())
}

/// Shapes and scalars for [`encode_inkling2_qkv_post`]. `dyn_heads` is
/// `kv_heads` on a full-attention layer (dynamic conv taps) and 0 on SWA.
#[derive(Clone, Copy, Debug)]
pub struct Inkling2QkvPostShape {
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub d_rel: u32,
    pub taps: u32,
    pub dyn_heads: u32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub tau: f32,
    pub rms_eps: f32,
}

/// Buffers for [`encode_inkling2_qkv_post`]. `k_out` / `v_out` are the slot's
/// KV row; `dyn_k` / `dyn_v` are read only when `dyn_heads > 0`.
pub struct Inkling2QkvPostBuffers<'a> {
    pub q: (&'a metal::Buffer, u64),
    pub q_scale: (&'a metal::Buffer, u64),
    pub r: (&'a metal::Buffer, u64),
    pub r_scale: (&'a metal::Buffer, u64),
    pub k_pre: (&'a metal::Buffer, u64),
    pub v_pre: (&'a metal::Buffer, u64),
    pub tail_k: (&'a metal::Buffer, u64),
    pub tail_v: (&'a metal::Buffer, u64),
    pub k_w: (&'a metal::Buffer, u64),
    pub v_w: (&'a metal::Buffer, u64),
    pub dyn_k: (&'a metal::Buffer, u64),
    pub dyn_v: (&'a metal::Buffer, u64),
    pub k_out: (&'a metal::Buffer, u64),
    pub v_out: (&'a metal::Buffer, u64),
}

/// One dispatch for the attention post-projection of a decode row: K and V
/// short convs into the KV row, K L2 clip, V per-head unit RMS, and the q and
/// r L2 clips with `tau`. Bit-identical to the chain of
/// [`encode_inkling2_short_conv_decode`] x2, [`encode_inkling2_l2_clip`] x3
/// and `encode_rms_norm_no_scale_perhead` (same helpers, same reduction
/// tree), which `tests/inkling2_ops_parity.rs` holds.
pub fn encode_inkling2_qkv_post(
    context: &mut MetalContext,
    pass: &PassEncoder,
    b: &Inkling2QkvPostBuffers<'_>,
    s: Inkling2QkvPostShape,
) -> Result<(), GpuError> {
    // The V norm keeps `rmsnorm_no_scale_perhead`'s tree only for hd threads
    // in whole SIMD groups of one element each (hd <= 256, the group cap).
    if s.head_dim == 0 || s.head_dim % 32 != 0 || s.head_dim > 256 {
        return Err(invalid(format!(
            "qkv_post needs head_dim a multiple of 32 up to 256, got {}",
            s.head_dim
        )));
    }
    if s.heads == 0 || s.kv_heads == 0 || s.d_rel == 0 || s.d_rel > 32 * 8 || s.taps < 2 {
        return Err(invalid(format!(
            "qkv_post shape out of contract: heads {} kv_heads {} d_rel {} taps {}",
            s.heads, s.kv_heads, s.d_rel, s.taps
        )));
    }
    if s.dyn_heads != 0 && s.dyn_heads != s.kv_heads {
        return Err(invalid(format!(
            "qkv_post dyn_heads must be 0 or kv_heads ({}), got {}",
            s.kv_heads, s.dyn_heads
        )));
    }
    let mut params = Vec::with_capacity(40);
    for v in [
        s.heads,
        s.kv_heads,
        s.head_dim,
        s.d_rel,
        s.taps,
        s.dyn_heads,
    ] {
        params.extend_from_slice(&v.to_le_bytes());
    }
    for v in [s.qk_clip, s.rel_clip, s.tau, s.rms_eps] {
        params.extend_from_slice(&v.to_le_bytes());
    }
    let groups = 2 * u64::from(s.heads) + 2 * u64::from(s.kv_heads);
    let p = pipeline(context, "inkling2_qkv_post")?;
    pass.encode_threadgroups(
        &p,
        &[
            (b.q.0, 0, b.q.1),
            (b.q_scale.0, 1, b.q_scale.1),
            (b.r.0, 2, b.r.1),
            (b.r_scale.0, 3, b.r_scale.1),
            (b.k_pre.0, 4, b.k_pre.1),
            (b.v_pre.0, 5, b.v_pre.1),
            (b.tail_k.0, 6, b.tail_k.1),
            (b.tail_v.0, 7, b.tail_v.1),
            (b.k_w.0, 8, b.k_w.1),
            (b.v_w.0, 9, b.v_w.1),
            (b.dyn_k.0, 10, b.dyn_k.1),
            (b.dyn_v.0, 11, b.dyn_v.1),
            (b.k_out.0, 12, b.k_out.1),
            (b.v_out.0, 13, b.v_out.1),
        ],
        &[(&params, 14)],
        groups.div_ceil(8),
        256,
    );
    Ok(())
}

/// Depth mix: for each of `n_q` F32 queries `[n_q, h]`, the softmax-weighted
/// sum of the RMS-normed FP16 `states[n_states, h]` (current state last).
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_depth_mix(
    context: &mut MetalContext,
    pass: &PassEncoder,
    states: (&metal::Buffer, u64),
    queries: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    n_states: u32,
    n_q: u32,
    h: u32,
) -> Result<(), GpuError> {
    let n = n_states as usize;
    if n == 0 || n > INKLING2_DEPTH_MAX_STATES || n_q == 0 || h == 0 {
        return Err(invalid(format!(
            "depth mix needs 1..={INKLING2_DEPTH_MAX_STATES} states, n_q > 0, h > 0; got {n}, {n_q}, {h}"
        )));
    }
    let p = pipeline(context, "inkling2_depth_mix")?;
    pass.encode_threadgroups(
        &p,
        &[
            (states.0, 0, states.1),
            (queries.0, 1, queries.1),
            (out.0, 2, out.1),
        ],
        &[(u32_bytes(&n_states), 3), (u32_bytes(&h), 4)],
        u64::from(n_q),
        256,
    );
    Ok(())
}

/// Elements per row the fused mix+norm holds per thread; rows up to
/// `256 * 16` wide.
pub const INKLING2_MIX_NORM_MAX_H: u32 = 256 * 16;

/// Depth mix fused with the BF16 RMS norm after it: one dispatch for
/// [`encode_inkling2_depth_mix`] (with `cur` read in place as the last state,
/// so no copy into `states`) followed by one `encode_rms_norm_bf16w` per
/// query, query `j` normed with `norm_weights[j]`. `norm_out` is `[n_q, h]`;
/// the un-normed mix goes to `mix_out` only when one is given. Bit-identical
/// to the separate kernels (`tests/inkling2_fused_parity.rs`); `h` must be at
/// least 256 so the norm's 256-thread tree is the one `rmsnorm_bf16w` uses.
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_depth_mix_norm(
    context: &mut MetalContext,
    pass: &PassEncoder,
    states: (&metal::Buffer, u64),
    cur: (&metal::Buffer, u64),
    queries: (&metal::Buffer, u64),
    mix_out: Option<(&metal::Buffer, u64)>,
    norm_weights: &[(&metal::Buffer, u64)],
    norm_out: (&metal::Buffer, u64),
    n_states: u32,
    h: u32,
    eps: f32,
) -> Result<(), GpuError> {
    let n = n_states as usize;
    let n_q = norm_weights.len();
    if n == 0 || n > INKLING2_DEPTH_MAX_STATES || !(1..=3).contains(&n_q) {
        return Err(invalid(format!(
            "depth mix+norm needs 1..={INKLING2_DEPTH_MAX_STATES} states and 1..=3 norms; got {n}, {n_q}"
        )));
    }
    if !(256..=INKLING2_MIX_NORM_MAX_H).contains(&h) {
        return Err(invalid(format!(
            "depth mix+norm needs 256 <= h <= {INKLING2_MIX_NORM_MAX_H}, got {h}"
        )));
    }
    let w = |j: usize| norm_weights[j.min(n_q - 1)];
    let (w0, w1, w2) = (w(0), w(1), w(2));
    let (mo, write) = match mix_out {
        Some(m) => (m, 1u32),
        // Unwritten when write_mix is 0; bound so the table is complete.
        None => (norm_out, 0u32),
    };
    let p = pipeline(context, "inkling2_depth_mix_norm")?;
    pass.encode_threadgroups(
        &p,
        &[
            (states.0, 0, states.1),
            (cur.0, 1, cur.1),
            (queries.0, 2, queries.1),
            (mo.0, 3, mo.1),
            (w0.0, 4, w0.1),
            (w1.0, 5, w1.1),
            (w2.0, 6, w2.1),
            (norm_out.0, 7, norm_out.1),
        ],
        &[
            (u32_bytes(&n_states), 8),
            (u32_bytes(&h), 9),
            (&f32_bytes(&eps), 10),
            (u32_bytes(&write), 11),
        ],
        n_q as u64,
        256,
    );
    Ok(())
}

/// Relative-bias table: FP32 `table[heads, extent]` from FP16 `r[heads, d_rel]`
/// and BF16 `proj[d_rel, extent]`.
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_rel_table(
    context: &mut MetalContext,
    pass: &PassEncoder,
    r: (&metal::Buffer, u64),
    proj: (&metal::Buffer, u64),
    table: (&metal::Buffer, u64),
    heads: u32,
    d_rel: u32,
    extent: u32,
) -> Result<(), GpuError> {
    if heads == 0 || d_rel == 0 || extent == 0 {
        return Err(invalid(
            "rel table needs heads, d_rel and extent > 0".into(),
        ));
    }
    let p = pipeline(context, "inkling2_rel_table")?;
    pass.encode_threadgroups(
        &p,
        &[(r.0, 0, r.1), (proj.0, 1, proj.1), (table.0, 2, table.1)],
        &[
            (u32_bytes(&heads), 3),
            (u32_bytes(&d_rel), 4),
            (u32_bytes(&extent), 5),
        ],
        (u64::from(heads) * u64::from(extent)).div_ceil(256),
        256,
    );
    Ok(())
}

/// Shape of one Inkling decode-attention dispatch.
#[derive(Debug, Clone, Copy)]
pub struct Inkling2AttnShape {
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub extent: u32,
    /// Rows in the K/V buffers: position `kp` lives at row `kp % ring`. A
    /// sliding-window slot passes its window, a full slot its capacity.
    pub ring: u32,
}

/// The largest head the decode attention serves (4 channels per lane).
pub const INKLING2_ATTN_MAX_HEAD_DIM: u32 = 128;

/// Decode attention for the query at `pos` over cached positions
/// `first..=pos`, with the relative-bias `table` (from
/// [`encode_inkling2_rel_table`]) and FP16 per-head `sinks`. `q` FP16
/// `[heads, head_dim]`; `k`, `v` FP16 `[shape.ring, kv_heads, head_dim]`,
/// position `kp` at row `kp % ring`, so `pos - first < ring`.
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_attention_decode(
    context: &mut MetalContext,
    pass: &PassEncoder,
    q: (&metal::Buffer, u64),
    k: (&metal::Buffer, u64),
    v: (&metal::Buffer, u64),
    table: (&metal::Buffer, u64),
    sinks: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    first: u32,
    pos: u32,
    shape: Inkling2AttnShape,
) -> Result<(), GpuError> {
    let s = shape;
    if s.kv_heads == 0
        || s.heads % s.kv_heads != 0
        || s.head_dim == 0
        || s.head_dim % 32 != 0
        || s.head_dim > INKLING2_ATTN_MAX_HEAD_DIM
        || s.extent == 0
        || first > pos
        || s.ring == 0
        || pos - first >= s.ring
    {
        return Err(invalid(format!(
            "Inkling 2 attention: unsupported shape {s:?} or range {first}..={pos}"
        )));
    }
    let group = s.heads / s.kv_heads;
    let p = pipeline(context, "inkling2_attention_decode")?;
    pass.encode_threadgroups(
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
            (u32_bytes(&first), 6),
            (u32_bytes(&pos), 7),
            (u32_bytes(&s.kv_heads), 8),
            (u32_bytes(&group), 9),
            (u32_bytes(&s.head_dim), 10),
            (u32_bytes(&s.extent), 11),
            (u32_bytes(&s.ring), 12),
        ],
        u64::from(s.heads),
        256,
    );
    Ok(())
}

/// Loop controller combine: `out = cur - tanh(forget) * cur + tanh(gain) *
/// msg` over `n` FP16 elements; `gain` and `forget` are pre-tanh.
#[allow(clippy::too_many_arguments)]
pub fn encode_inkling2_control_combine(
    context: &mut MetalContext,
    pass: &PassEncoder,
    cur: (&metal::Buffer, u64),
    msg: (&metal::Buffer, u64),
    gain: (&metal::Buffer, u64),
    forget: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    n: u32,
) -> Result<(), GpuError> {
    if n == 0 {
        return Err(invalid("control combine needs n > 0".into()));
    }
    let p = pipeline(context, "inkling2_control_combine")?;
    pass.encode_threadgroups(
        &p,
        &[
            (cur.0, 0, cur.1),
            (msg.0, 1, msg.1),
            (gain.0, 2, gain.1),
            (forget.0, 3, forget.1),
            (out.0, 4, out.1),
        ],
        &[(u32_bytes(&n), 5)],
        u64::from(n).div_ceil(256),
        256,
    );
    Ok(())
}

/// `dst[0..n] = src[0..n]` over FP16. The ranges must not overlap.
pub fn encode_inkling2_copy_half(
    context: &mut MetalContext,
    pass: &PassEncoder,
    src: (&metal::Buffer, u64),
    dst: (&metal::Buffer, u64),
    n: u32,
) -> Result<(), GpuError> {
    if n == 0 {
        return Ok(());
    }
    let p = pipeline(context, "inkling2_copy_half")?;
    pass.encode_threadgroups(
        &p,
        &[(src.0, 0, src.1), (dst.0, 1, dst.1)],
        &[(u32_bytes(&n), 2)],
        u64::from(n).div_ceil(256),
        256,
    );
    Ok(())
}

/// Segments one [`encode_inkling2_bf16_gemv_multi`] dispatch can carry.
pub const INKLING2_GEMV_MAX_SEGMENTS: usize = 8;

/// One matrix of a multi-segment BF16 GEMV: `rows x cols` BF16 at byte
/// `w_offset` of the weights buffer, reading input `x` (an index into the
/// dispatch's inputs) and writing `rows` FP16 values to `y`.
#[derive(Debug, Clone, Copy)]
pub struct Inkling2GemvSegment<'a> {
    pub w_offset: u64,
    pub rows: u32,
    pub cols: u32,
    pub x: usize,
    pub y: (&'a metal::Buffer, u64),
}

/// Up to eight resident BF16 GEMVs in one dispatch, all matrices in place in
/// `weights` (bound at offset 0). `xs` holds one to three FP16 inputs. Refuses
/// anything the kernel's vector loads cannot address: columns not a multiple
/// of 4, a matrix, input or output that is not 8-byte aligned, or a bad input
/// index.
pub fn encode_inkling2_bf16_gemv_multi(
    context: &mut MetalContext,
    pass: &PassEncoder,
    weights: &metal::Buffer,
    xs: &[(&metal::Buffer, u64)],
    segments: &[Inkling2GemvSegment<'_>],
) -> Result<(), GpuError> {
    gemv_multi_impl(
        context,
        pass,
        weights,
        xs,
        segments,
        "inkling2_bf16_gemv_multi",
        8,
    )
}

/// [`encode_inkling2_bf16_gemv_multi`] with two weight rows per simdgroup
/// (`inkling2_bf16_gemv_multi2`): the same bits per row, twice the loads in
/// flight.
pub fn encode_inkling2_bf16_gemv_multi2(
    context: &mut MetalContext,
    pass: &PassEncoder,
    weights: &metal::Buffer,
    xs: &[(&metal::Buffer, u64)],
    segments: &[Inkling2GemvSegment<'_>],
) -> Result<(), GpuError> {
    gemv_multi_impl(
        context,
        pass,
        weights,
        xs,
        segments,
        "inkling2_bf16_gemv_multi2",
        16,
    )
}

fn gemv_multi_impl(
    context: &mut MetalContext,
    pass: &PassEncoder,
    weights: &metal::Buffer,
    xs: &[(&metal::Buffer, u64)],
    segments: &[Inkling2GemvSegment<'_>],
    kernel: &'static str,
    rows_per_tg: u32,
) -> Result<(), GpuError> {
    if segments.is_empty() || segments.len() > INKLING2_GEMV_MAX_SEGMENTS {
        return Err(invalid(format!(
            "bf16 gemv multi takes 1 to {INKLING2_GEMV_MAX_SEGMENTS} segments, got {}",
            segments.len()
        )));
    }
    if xs.is_empty() || xs.len() > 3 || xs.iter().any(|x| x.1 % 8 != 0) {
        return Err(invalid(
            "bf16 gemv multi takes 1 to 3 inputs at 8-byte-aligned offsets".into(),
        ));
    }
    // The shader's `Inkling2GemvTable`: w_off[8] (u64), rows[8], cols[8],
    // x_idx[8], tg_start[9], n_seg (u32), 200 bytes, little-endian.
    let (mut w_off, mut rows, mut cols, mut x_idx) = ([0u64; 8], [0u32; 8], [0u32; 8], [0u32; 8]);
    let mut tg_start = [0u32; 9];
    let mut groups = 0u32;
    for (i, s) in segments.iter().enumerate() {
        if s.rows == 0
            || s.cols == 0
            || s.cols % 4 != 0
            || s.w_offset % 8 != 0
            || s.y.1 % 2 != 0
            || s.x >= xs.len()
        {
            return Err(invalid(format!(
                "bf16 gemv multi segment {i}: rows {} cols {} (multiple of 4), \
                 w offset {} (8-aligned), input {} of {}",
                s.rows,
                s.cols,
                s.w_offset,
                s.x,
                xs.len()
            )));
        }
        w_off[i] = s.w_offset;
        rows[i] = s.rows;
        cols[i] = s.cols;
        x_idx[i] = s.x as u32;
        tg_start[i] = groups;
        groups += s.rows.div_ceil(rows_per_tg);
    }
    for g in &mut tg_start[segments.len()..] {
        *g = groups;
    }
    let mut bytes = Vec::with_capacity(200);
    for v in w_off {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in rows.iter().chain(&cols).chain(&x_idx).chain(&tg_start) {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes.extend_from_slice(&(segments.len() as u32).to_le_bytes());
    debug_assert_eq!(bytes.len(), 200);
    let p = pipeline(context, kernel)?;
    let mut buffers: Vec<(&metal::Buffer, u64, u64)> = vec![(weights, 0, 0)];
    for i in 0..3 {
        let x = xs[i.min(xs.len() - 1)];
        buffers.push((x.0, 2 + i as u64, x.1));
    }
    // Unused output slots repeat the last segment's output; the kernel never
    // writes them.
    for i in 0..INKLING2_GEMV_MAX_SEGMENTS {
        let y = segments[i.min(segments.len() - 1)].y;
        buffers.push((y.0, 5 + i as u64, y.1));
    }
    pass.encode_threadgroups(&p, &buffers, &[(&bytes, 1)], u64::from(groups), 256);
    Ok(())
}
