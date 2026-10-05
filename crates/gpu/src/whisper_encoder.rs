//! Host-side dispatch for `shaders/whisper_encoder.metal`, the f32 kernels
//! behind the whisper Metal execution path (runtime whisper module).
//!
//! The CPU reference is `compute::whisper` plus the `compute::vision` helper
//! kernels it borrows (layer_norm, matmul_bias, gelu_erf); the parity tests
//! in `tests/whisper_encoder_parity.rs` hold every kernel here to those
//! references on real hardware. Everything is buffer-level: the runtime owns
//! resident weight and activation buffers and sequences many dispatches per
//! pass, so unlike `whisper_conv` there are no per-call host copies here.
//!
//! Layout contract (matches the HF checkpoint and the CPU reference):
//! linear weights are `[out, in]` row-major, activations stream as
//! `[seq, d_model]` rows, attention runs per head over `[len, d_model]`
//! caches with `head_dim`-wide slices. All math is f32 with f32
//! accumulation, the reference's own precision.

use crate::bytes::{f32_bytes, u32_bytes};
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::rms_norm::unused_function_constants;

pub(crate) static SOURCE: &str = include_str!("shaders/whisper_encoder.metal");

/// Threads per threadgroup for the one-thread-per-row kernels (softmax,
/// layer norm). The shaders' shared-memory tree reductions require a
/// power of two.
pub const ROW_THREADS: u64 = 256;

/// The matmul tile's threads per threadgroup (32 x 32 outputs).
const GEMM_THREADS: (u64, u64, u64) = (32, 32, 1);

/// The largest key range `whisper_attn_step_f32` can attend over. Whisper
/// self-attention reaches `max_target_positions` (448) and cross-attention
/// `max_source_positions` (1500); the headroom is deliberate slack, not a
/// tuning knob, because the scores live in threadgroup memory.
pub const MAX_ATTN_STEP: usize = 1536;

fn groups_for(count: u32) -> u64 {
    (count as u64).div_ceil(ROW_THREADS)
}

/// A buffer view: base buffer plus the element offset of the view's first
/// element. Head slices and packed projections are views into larger
/// tensors; the byte offset rides the encoder's per-binding offset.
#[derive(Clone, Copy)]
pub struct F32View<'a> {
    pub buffer: &'a metal::Buffer,
    /// Element offset from the buffer start.
    pub offset_elements: u64,
}

impl F32View<'_> {
    /// The view's start as a byte offset, for dispatch helpers that take
    /// raw `(buffer, index, offset)` triples.
    pub(crate) fn byte_offset(&self) -> u64 {
        self.offset_elements * std::mem::size_of::<f32>() as u64
    }
}

impl<'a> F32View<'a> {
    pub fn new(buffer: &'a metal::Buffer) -> Self {
        Self {
            buffer,
            offset_elements: 0,
        }
    }

    pub fn at(buffer: &'a metal::Buffer, offset_elements: u64) -> Self {
        Self {
            buffer,
            offset_elements,
        }
    }

    /// The `(buffer, index, byte offset)` triple for one declared shader
    /// argument slot. The index is a REQUIRED parameter on purpose: the
    /// first cut reported slot 0 unconditionally and every view in a
    /// multi-argument dispatch landed on the same binding -- the conv
    /// wrote its output through the input's slot (all-zero front end) and
    /// the GEMV read five ways out of bounds (a 33-minute GPU stall).
    /// Slots must be stated where the buffers are listed.
    pub(crate) fn binding(&self, index: u64) -> (&'a metal::Buffer, u64, u64) {
        (
            self.buffer,
            index,
            self.offset_elements * std::mem::size_of::<f32>() as u64,
        )
    }
}

/// `out[m, n] = out_scale * sum_k a[m, k] * w[n, k] + bias[n]`, f32 end to
/// end.
///
/// `w` is ROW-MAJOR BY OUTPUT (`[n, k]`), the `nn.Linear` layout. The
/// strides are the row strides of each operand in elements; a projection
/// over contiguous tensors passes `k`, `k`, `n`. A head slice of a
/// `[seq, d_model]` stream passes the stream's row stride for the operand
/// stride with the view offset pointing at the head's first element.
///
/// `w_transposed` reads the weight as `w[k, n]` instead (address
/// `k * w_stride + n`), which is the attention value mix: the score
/// matrix multiplies V, and V's rows are keys, not outputs. The scores
/// GEMM (`Q x K^T`) does NOT set it -- there `w[n, k] = K[n, k]` is the
/// plain nn.Linear orientation over the key stream.
///
/// `out_scale` multiplies the reduction before the bias. The attention
/// score projections pass head_dim^-0.5 (the reference scales the dot
/// products; scaling the Q rows instead would need an extra dispatch per
/// layer); every other call site passes 1.0.
///
/// `bias` is bound whether or not it is used (an unbound declared argument
/// is undefined behaviour); `has_bias` selects, and a caller with none may
/// pass any valid view, including `a`.
#[allow(clippy::too_many_arguments)]
pub fn encode_matmul_bias(
    context: &mut MetalContext,
    pass: &PassEncoder,
    a: F32View,
    w: F32View,
    bias: Option<F32View>,
    out: F32View,
    m: u32,
    k: u32,
    n: u32,
    a_stride: u32,
    w_stride: u32,
    out_stride: u32,
    w_transposed: bool,
    out_scale: f32,
) -> Result<(), GpuError> {
    if m == 0 || k == 0 || n == 0 {
        return Err(GpuError::InvalidInput(
            "whisper matmul dims must be positive".to_string(),
        ));
    }
    if a_stride < k || w_stride < k.min(n) || out_stride < n {
        return Err(GpuError::InvalidInput(format!(
            "whisper matmul strides must cover k/n (m {m}, k {k}, n {n}, a_stride {a_stride}, w_stride {w_stride}, out_stride {out_stride})"
        )));
    }
    let pipeline = context.pipeline(
        SOURCE,
        "whisper_matmul_bias_f32",
        &unused_function_constants(),
        b"",
    )?;
    let has_bias: u32 = u32::from(bias.is_some());
    let w_transposed_u32: u32 = u32::from(w_transposed);
    let bias_binding = bias.unwrap_or(a);
    let (a_buf, _a_index, a_off) = a.binding(0);
    let (w_buf, _w_index, w_off) = w.binding(0);
    let (b_buf, _b_index, b_off) = bias_binding.binding(0);
    let (o_buf, _o_index, o_off) = out.binding(0);
    pass.encode_threadgroups_3d(
        &pipeline,
        &[
            (a_buf, 0, a_off),
            (w_buf, 1, w_off),
            (b_buf, 2, b_off),
            (o_buf, 3, o_off),
        ],
        &[
            (u32_bytes(&m), 4),
            (u32_bytes(&k), 5),
            (u32_bytes(&n), 6),
            (u32_bytes(&a_stride), 7),
            (u32_bytes(&w_stride), 8),
            (u32_bytes(&out_stride), 9),
            (u32_bytes(&has_bias), 10),
            (u32_bytes(&w_transposed_u32), 11),
            (f32_bytes(&out_scale), 12),
        ],
        (u64::from(n).div_ceil(32), u64::from(m).div_ceil(32), 1),
        GEMM_THREADS,
    );
    Ok(())
}

/// Row softmax over a `[rows, n]` f32 buffer. In place is fine: each
/// threadgroup owns one row exclusively and reads precede writes within it.
pub fn encode_softmax_rows(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: F32View,
    y: F32View,
    rows: u32,
    n: u32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "whisper_softmax_rows_f32",
        &unused_function_constants(),
        b"",
    )?;
    let (x_buf, _xi, x_off) = x.binding(0);
    let (y_buf, _yi, y_off) = y.binding(0);
    pass.encode_threadgroups(
        &pipeline,
        &[(x_buf, 0, x_off), (y_buf, 1, y_off)],
        &[(u32_bytes(&n), 2)],
        u64::from(rows),
        ROW_THREADS,
    );
    Ok(())
}

/// Exact-erf GELU over `count` f32 values. In place is fine: one read, one
/// write, same element.
pub fn encode_gelu_erf(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: F32View,
    y: F32View,
    count: u32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "whisper_gelu_erf_f32",
        &unused_function_constants(),
        b"",
    )?;
    let (x_buf, _xi, x_off) = x.binding(0);
    let (y_buf, _yi, y_off) = y.binding(0);
    pass.encode_threadgroups(
        &pipeline,
        &[(x_buf, 0, x_off), (y_buf, 1, y_off)],
        &[(u32_bytes(&count), 2)],
        groups_for(count),
        ROW_THREADS,
    );
    Ok(())
}

/// `y[i] = a[i] + b[i]` over `count` f32 values.
pub fn encode_add(
    context: &mut MetalContext,
    pass: &PassEncoder,
    a: F32View,
    b: F32View,
    y: F32View,
    count: u32,
) -> Result<(), GpuError> {
    let pipeline =
        context.pipeline(SOURCE, "whisper_add_f32", &unused_function_constants(), b"")?;
    let (a_buf, _ai, a_off) = a.binding(0);
    let (b_buf, _bi, b_off) = b.binding(0);
    let (y_buf, _yi, y_off) = y.binding(0);
    pass.encode_threadgroups(
        &pipeline,
        &[(a_buf, 0, a_off), (b_buf, 1, b_off), (y_buf, 2, y_off)],
        &[(u32_bytes(&count), 3)],
        groups_for(count),
        ROW_THREADS,
    );
    Ok(())
}

/// Transposes the conv front end's `[d, seq]` band-major output into
/// `[seq, d]` rows while adding the positional row per position.
#[allow(clippy::too_many_arguments)]
pub fn encode_transpose_pos(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: F32View,
    pos: F32View,
    y: F32View,
    seq: u32,
    d: u32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "whisper_transpose_pos_f32",
        &unused_function_constants(),
        b"",
    )?;
    let (x_buf, _xi, x_off) = x.binding(0);
    let (p_buf, _pi, p_off) = pos.binding(0);
    let (y_buf, _yi, y_off) = y.binding(0);
    let total = u32::try_from(seq as u64 * d as u64)
        .map_err(|_| GpuError::InvalidInput("transpose_pos overflow".to_string()))?;
    pass.encode_threadgroups(
        &pipeline,
        &[(x_buf, 0, x_off), (p_buf, 1, p_off), (y_buf, 2, y_off)],
        &[(u32_bytes(&seq), 3), (u32_bytes(&d), 4)],
        groups_for(total),
        ROW_THREADS,
    );
    Ok(())
}

/// LayerNorm per row of a `[rows, d]` f32 buffer. Out may alias `x` only
/// per whole-buffer when rows == 1.
#[allow(clippy::too_many_arguments)]
pub fn encode_layer_norm(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: F32View,
    weight: F32View,
    bias: F32View,
    y: F32View,
    rows: u32,
    d: u32,
    eps: f32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "whisper_layer_norm_f32",
        &unused_function_constants(),
        b"",
    )?;
    let (x_buf, _xi, x_off) = x.binding(0);
    let (w_buf, _wi, w_off) = weight.binding(0);
    let (b_buf, _bi, b_off) = bias.binding(0);
    let (y_buf, _yi, y_off) = y.binding(0);
    pass.encode_threadgroups(
        &pipeline,
        &[
            (x_buf, 0, x_off),
            (w_buf, 1, w_off),
            (b_buf, 2, b_off),
            (y_buf, 3, y_off),
        ],
        &[(u32_bytes(&d), 4), (f32_bytes(&eps), 5)],
        u64::from(rows),
        ROW_THREADS,
    );
    Ok(())
}

/// Single-query attention step over a `[len_max, d_model]` K/V cache pair.
///
/// `append` selects the self-attention contract: `k_new`/`v_new` (`[d]`
/// views, the layer's fresh projections) are written into cache row `pos`
/// before the query at `pos` attends over rows `0..=pos`. With `append`
/// off the caches are read-only over `0..attend_len` (cross attention over
/// the encoded window) and `k_new`/`v_new` are unread -- pass `q` for
/// them, matching the shader's "any valid buffer" rule.
#[allow(clippy::too_many_arguments)]
pub fn encode_attn_step(
    context: &mut MetalContext,
    pass: &PassEncoder,
    q: F32View,
    k_new: F32View,
    v_new: F32View,
    k_cache: F32View,
    v_cache: F32View,
    out: F32View,
    pos: u32,
    attend_len: u32,
    d_model: u32,
    head_dim: u32,
    append: bool,
    scale: f32,
    v_transposed: bool,
) -> Result<(), GpuError> {
    if head_dim == 0 || d_model % head_dim != 0 {
        return Err(GpuError::InvalidInput(
            "whisper attn head_dim must divide d_model".to_string(),
        ));
    }
    if attend_len as usize > MAX_ATTN_STEP {
        return Err(GpuError::InvalidInput(format!(
            "whisper attn step attends {attend_len} keys, the kernel caps at {MAX_ATTN_STEP}"
        )));
    }
    if append && pos >= attend_len {
        return Err(GpuError::InvalidInput(
            "whisper attn append row must be inside the attended range".to_string(),
        ));
    }
    let heads = d_model / head_dim;
    let pipeline = context.pipeline(
        SOURCE,
        "whisper_attn_step_f32",
        &unused_function_constants(),
        b"",
    )?;
    let (q_buf, _qi, q_off) = q.binding(0);
    let (kn_buf, _ki, kn_off) = k_new.binding(0);
    let (vn_buf, _vi, vn_off) = v_new.binding(0);
    let (kc_buf, _kci, kc_off) = k_cache.binding(0);
    let (vc_buf, _vci, vc_off) = v_cache.binding(0);
    let (o_buf, _oi, o_off) = out.binding(0);
    let append_u32: u32 = u32::from(append);
    let v_transposed_u32: u32 = u32::from(v_transposed);
    pass.encode_threadgroups(
        &pipeline,
        &[
            (q_buf, 0, q_off),
            (kn_buf, 1, kn_off),
            (vn_buf, 2, vn_off),
            (kc_buf, 3, kc_off),
            (vc_buf, 4, vc_off),
            (o_buf, 5, o_off),
        ],
        &[
            (u32_bytes(&pos), 6),
            (u32_bytes(&attend_len), 7),
            (u32_bytes(&d_model), 8),
            (u32_bytes(&head_dim), 9),
            (u32_bytes(&append_u32), 10),
            (f32_bytes(&scale), 11),
            (u32_bytes(&v_transposed_u32), 12),
        ],
        u64::from(heads),
        ROW_THREADS,
    );
    Ok(())
}

/// Transposes a `[rows, cols]` f32 buffer into `[cols, rows]`, one thread
/// per element. Serves the cross-attention value caches (see the attn
/// step's `v_transposed` contract).
pub fn encode_transpose(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: F32View,
    y: F32View,
    rows: u32,
    cols: u32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "whisper_transpose_f32",
        &unused_function_constants(),
        b"",
    )?;
    let total = u32::try_from(rows as u64 * cols as u64)
        .map_err(|_| GpuError::InvalidInput("transpose overflow".to_string()))?;
    pass.encode_threadgroups(
        &pipeline,
        &[x.binding(0), y.binding(1)],
        &[(u32_bytes(&rows), 2), (u32_bytes(&cols), 3)],
        groups_for(total),
        ROW_THREADS,
    );
    Ok(())
}

/// Single-row GEMV (`m == 1` matmul): one simd-group per output row, lanes
/// striding the reduction, no staging or threadgroup barriers. This is
/// the decode loop's projection form; the tiled kernel's chunk barriers
/// dominate its runtime at `m == 1`. `out_scale` rides the reduction
/// (head_dim^-0.5 for attention score rows); `bias` and `residual` are
/// each optional -- a fused residual write replaces a separate add
/// dispatch, which matters at sixty dispatches per token. A caller with
/// none of one binds any valid view, `a` by convention.
#[allow(clippy::too_many_arguments)]
pub fn encode_gemv(
    context: &mut MetalContext,
    pass: &PassEncoder,
    a: F32View,
    w: F32View,
    bias: Option<F32View>,
    residual: Option<F32View>,
    out: F32View,
    k: u32,
    n: u32,
    w_stride: u32,
    w_transposed: bool,
    out_scale: f32,
) -> Result<(), GpuError> {
    if k == 0 || n == 0 {
        return Err(GpuError::InvalidInput(
            "whisper gemv dims must be positive".to_string(),
        ));
    }
    if w_stride < k.min(n) {
        return Err(GpuError::InvalidInput(format!(
            "whisper gemv w_stride must cover k/n (k {k}, n {n}, w_stride {w_stride})"
        )));
    }
    let pipeline = context.pipeline(
        SOURCE,
        "whisper_gemv_f32",
        &unused_function_constants(),
        b"",
    )?;
    let has_bias: u32 = u32::from(bias.is_some());
    let has_residual: u32 = u32::from(residual.is_some());
    let w_transposed_u32: u32 = u32::from(w_transposed);
    let bias_binding = bias.unwrap_or(a);
    let residual_binding = residual.unwrap_or(a);
    pass.encode_threadgroups(
        &pipeline,
        &[
            a.binding(0),
            w.binding(1),
            bias_binding.binding(2),
            residual_binding.binding(3),
            out.binding(4),
        ],
        &[
            (u32_bytes(&k), 5),
            (u32_bytes(&n), 6),
            (u32_bytes(&w_stride), 7),
            (u32_bytes(&has_bias), 8),
            (u32_bytes(&has_residual), 9),
            (u32_bytes(&w_transposed_u32), 10),
            (f32_bytes(&out_scale), 11),
        ],
        u64::from(n),
        32,
    );
    Ok(())
}
