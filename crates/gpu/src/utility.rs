//! Host-side dispatch for `shaders/utility.metal` (vendored from
//! `Metal/Primitives/utility.metal`, plus the `gelu_pytorch_tanh` helper it
//! borrows from `moe.metal` — see the shader's own header). These are the
//! small elementwise kernels that keep activations on the GPU between the
//! big kernels: gated-FFN activation multiplies and the residual add.
//!
//! `sigmoid_gate_mul_fp16`, `sigmoid_scalar_mul_fp16`, and
//! `split_q_gate_fp16` (all Qwen 3.6-specific) are dispatched from
//! `families/qwen/` (three or more production call sites each).

use foundation::SteeringMode;
use metal::FunctionConstantValues;

use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext, PassEncoder};

const SOURCE: &str = include_str!("shaders/utility.metal");
const THREADS_PER_GROUP: u64 = 256;

fn grid_for(count: u32) -> u64 {
    // AGENTS.md/CLAUDE.md S12: floored at 1, matching rope.rs's own
    // convention -- a `count == 0` caller still dispatches one
    // (harmless, bounds-checked-empty) threadgroup rather than zero.
    (count as u64).div_ceil(THREADS_PER_GROUP).max(1) * THREADS_PER_GROUP
}

fn encode_elementwise(
    context: &mut MetalContext,
    pass: &PassEncoder,
    function_name: &'static str,
    buffers: &[(&metal::Buffer, u64, u64)],
    count: u32,
    count_index: u64,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(SOURCE, function_name, &FunctionConstantValues::new(), b"")?;
    pass.encode_threads_3d(
        &pipeline,
        buffers,
        &[(u32_bytes(&count), count_index)],
        (grid_for(count), 1, 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}

/// `out[i] = gelu_pytorch_tanh(gate[i]) * up[i]` (Gemma's GeGLU).
pub fn encode_gelu_mul(
    context: &mut MetalContext,
    pass: &PassEncoder,
    gate: (&metal::Buffer, u64),
    up: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    count: u32,
) -> Result<(), GpuError> {
    encode_elementwise(
        context,
        pass,
        "gelu_mul_fp16",
        &[(gate.0, 0, gate.1), (up.0, 1, up.1), (out.0, 2, out.1)],
        count,
        3,
    )
}

/// `out[i] = silu(gate[i]) * up[i]` (Qwen's SwiGLU).
pub fn encode_silu_mul(
    context: &mut MetalContext,
    pass: &PassEncoder,
    gate: (&metal::Buffer, u64),
    up: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    count: u32,
) -> Result<(), GpuError> {
    encode_elementwise(
        context,
        pass,
        "silu_mul_fp16",
        &[(gate.0, 0, gate.1), (up.0, 1, up.1), (out.0, 2, out.1)],
        count,
        3,
    )
}

/// `y[i] += bf16(bias[i])`, in place — ROADMAP M5's per-projection bias.
///
/// `bias` is BF16 because that is what `transcode_f32` narrows gpt-oss's F32
/// biases to at repack, the same width and the same reader as a norm weight.
/// See the shader for why this is a separate pass rather than an argument on
/// every GEMV.
pub fn encode_bias_add(
    context: &mut MetalContext,
    pass: &PassEncoder,
    y: (&metal::Buffer, u64),
    bias: (&metal::Buffer, u64),
    count: u32,
) -> Result<(), GpuError> {
    encode_elementwise(
        context,
        pass,
        "bias_add_bf16_fp16",
        &[(y.0, 0, y.1), (bias.0, 1, bias.1)],
        count,
        2,
    )
}

/// `hidden[i] += delta[i]`, in place, in FP16 — the residual stream stays
/// on the GPU exactly as the Swift original keeps it.
pub fn encode_residual_add(
    context: &mut MetalContext,
    pass: &PassEncoder,
    hidden: (&metal::Buffer, u64),
    delta: (&metal::Buffer, u64),
    count: u32,
) -> Result<(), GpuError> {
    encode_elementwise(
        context,
        pass,
        "residual_add_fp16",
        &[(hidden.0, 0, hidden.1), (delta.0, 1, delta.1)],
        count,
        2,
    )
}

/// `x[i] *= half(scalar)`, in place — Gemma 4's per-layer `layer_scalar`
/// multiply on the residual stream (see the shader's port-local note).
pub fn encode_scalar_mul(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: (&metal::Buffer, u64),
    scalar: f32,
    count: u32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "scalar_mul_fp16",
        &FunctionConstantValues::new(),
        b"",
    )?;
    pass.encode_threads_3d(
        &pipeline,
        &[(x.0, 0, x.1)],
        &[
            (crate::bytes::f32_bytes(&scalar), 1),
            (u32_bytes(&count), 2),
        ],
        (grid_for(count), 1, 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}

/// `x[i] = silu(x[i])`, in place -- a plain UNARY silu, port-local
/// (`qwen4_exp`'s hyper-connection mix). Every other silu call site in
/// this port is `silu_mul_fp16`'s gated pair; this is the one place the
/// activation applies to nothing but itself.
pub fn encode_silu(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: (&metal::Buffer, u64),
    count: u32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(SOURCE, "silu_fp16", &FunctionConstantValues::new(), b"")?;
    pass.encode_threads_3d(
        &pipeline,
        &[(x.0, 0, x.1)],
        &[(u32_bytes(&count), 1)],
        (grid_for(count), 1, 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}

/// `x[i] = sigmoid(x[i])`, in place -- the unary sibling of
/// [`encode_silu`], for the same reason.
pub fn encode_sigmoid(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: (&metal::Buffer, u64),
    count: u32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(SOURCE, "sigmoid_fp16", &FunctionConstantValues::new(), b"")?;
    pass.encode_threads_3d(
        &pipeline,
        &[(x.0, 0, x.1)],
        &[(u32_bytes(&count), 1)],
        (grid_for(count), 1, 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}

/// `out[i] *= sigmoid(gate[i])`, in place -- Qwen 3.6's full-attention
/// output gate (the second half of its packed `q_proj`).
pub fn encode_sigmoid_gate_mul(
    context: &mut MetalContext,
    pass: &PassEncoder,
    out: (&metal::Buffer, u64),
    gate: (&metal::Buffer, u64),
    count: u32,
) -> Result<(), GpuError> {
    encode_elementwise(
        context,
        pass,
        "sigmoid_gate_mul_fp16",
        &[(out.0, 0, out.1), (gate.0, 1, gate.1)],
        count,
        2,
    )
}

/// `out[i] = gelu_erf(gate[i]) * up[i]` -- Spark-X2.5's MLP activation, the
/// EXACT-erf GELU where [`encode_gelu_mul`] applies Gemma's tanh form. The
/// two agree to about 5e-4 absolute, which is why this is a separate kernel
/// and not a mode byte (see the shader). Port-local; the contract is
/// `turbospark_compute::gating::gelu_erf_mul`.
pub fn encode_gelu_erf_mul(
    context: &mut MetalContext,
    pass: &PassEncoder,
    gate: (&metal::Buffer, u64),
    up: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    count: u32,
) -> Result<(), GpuError> {
    encode_elementwise(
        context,
        pass,
        "gelu_erf_mul_fp16",
        &[(gate.0, 0, gate.1), (up.0, 1, up.1), (out.0, 2, out.1)],
        count,
        3,
    )
}

/// `y[i] *= sigmoid(gate[0])`, in place -- Qwen 3.6's shared-expert scalar
/// gate. `gate` is a one-element buffer, the `shared_expert_gate` GEMV's
/// output; the kernel reads element 0 for every `i`.
pub fn encode_sigmoid_scalar_mul(
    context: &mut MetalContext,
    pass: &PassEncoder,
    y: (&metal::Buffer, u64),
    gate: (&metal::Buffer, u64),
    count: u32,
) -> Result<(), GpuError> {
    encode_elementwise(
        context,
        pass,
        "sigmoid_scalar_mul_fp16",
        &[(y.0, 0, y.1), (gate.0, 1, gate.1)],
        count,
        2,
    )
}

/// `out[i] *= sigmoid(gate[i / head_dim])`, in place -- Spark-X2.5's
/// per-head attention output gate. `gate` holds ONE logit per head
/// (`total / head_dim` elements), each scaling its own head's whole slice of
/// `out`, where [`encode_sigmoid_gate_mul`]'s gate is as long as the row.
/// Port-local; the contract is
/// `turbospark_compute::gating::sigmoid_head_gate_mul`.
pub fn encode_sigmoid_head_gate_mul(
    context: &mut MetalContext,
    pass: &PassEncoder,
    out: (&metal::Buffer, u64),
    gate: (&metal::Buffer, u64),
    head_dim: u32,
    total: u32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "sigmoid_head_gate_mul_fp16",
        &FunctionConstantValues::new(),
        b"",
    )?;
    pass.encode_threads_3d(
        &pipeline,
        &[(out.0, 0, out.1), (gate.0, 1, gate.1)],
        &[(u32_bytes(&head_dim), 2), (u32_bytes(&total), 3)],
        (grid_for(total), 1, 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}

/// Splits a `[heads, 2 * dim]` packed projection into contiguous
/// `[heads, dim]` `q` and `gate` buffers. Qwen 3.6's `q_proj` emits per-head
/// `[query; gate]` pairs, which the per-head norm, RoPE, and attention
/// kernels cannot consume interleaved.
pub fn encode_split_q_gate(
    context: &mut MetalContext,
    pass: &PassEncoder,
    packed: (&metal::Buffer, u64),
    q: (&metal::Buffer, u64),
    gate: (&metal::Buffer, u64),
    heads: u32,
    dim: u32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "split_q_gate_fp16",
        &FunctionConstantValues::new(),
        b"",
    )?;
    let count = heads * dim;
    pass.encode_threads_3d(
        &pipeline,
        &[(packed.0, 0, packed.1), (q.0, 1, q.1), (gate.0, 2, gate.1)],
        &[(u32_bytes(&heads), 3), (u32_bytes(&dim), 4)],
        (grid_for(count), 1, 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}

/// Splits a fused `[q (q_elems) | k (kv_elems) | v (kv_elems)]` projection
/// output into its three destination buffers. Spark's `q_k_v_proj` is one
/// weight, so what three separate GEMVs used to produce arrives as one row
/// and the split moves to the activation side. A pure copy (three range
/// copies, not a permute), so the tests assert exact bits. Port-local; the
/// contract is `turbospark_compute::gating::split_qkv`.
#[allow(clippy::too_many_arguments)]
pub fn encode_split_qkv(
    context: &mut MetalContext,
    pass: &PassEncoder,
    src: (&metal::Buffer, u64),
    q: (&metal::Buffer, u64),
    k: (&metal::Buffer, u64),
    v: (&metal::Buffer, u64),
    q_elems: u32,
    kv_elems: u32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "split_qkv_fp16",
        &FunctionConstantValues::new(),
        b"",
    )?;
    let count = q_elems + 2 * kv_elems;
    pass.encode_threads_3d(
        &pipeline,
        &[
            (src.0, 0, src.1),
            (q.0, 1, q.1),
            (k.0, 2, k.1),
            (v.0, 3, v.1),
        ],
        &[(u32_bytes(&q_elems), 4), (u32_bytes(&kv_elems), 5)],
        (grid_for(count), 1, 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}

/// The scalar operands of one [`encode_steer_direction`] dispatch.
///
/// Grouped into a struct rather than passed as eight positional arguments
/// because six of them are `f32` or `u32` and a transposed pair would be a
/// silent wrong edit rather than a type error -- `alpha` and `target` in
/// particular are both plain floats whose swap produces a plausible result.
#[derive(Debug, Clone, Copy)]
pub struct SteerParams {
    /// Length of the direction, and of the row window it edits.
    pub d_len: u32,
    /// How many consecutive rows of `x` to edit. 1 during decode, `M` for a
    /// batched verify or a prefill chunk.
    pub rows: u32,
    /// Distance between rows of `x`, in ELEMENTS and never bytes. Taking it
    /// in elements is deliberate: the residual stream's own row offsets are
    /// computed in bytes at every call site (`m * hidden * 2`), so a stride
    /// that accepted bytes would read correctly at `rows == 1` and silently
    /// edit every other row at `rows > 1`.
    pub row_stride: u32,
    /// Which of the four edits to apply.
    pub mode: SteeringMode,
    /// Strength. `0.0` is the exact identity in every mode, which is the null
    /// control the steering probe leans on.
    pub alpha: f32,
    /// `1 / ||d||`, precomputed by the loader. See
    /// `turbospark_compute::steering::inv_norm`, including why a zero
    /// direction yields `0.0` here rather than an infinity.
    pub inv_norm: f32,
    /// The coefficient [`SteeringMode::Clamp`] pins the stream to. Ignored by
    /// the other two modes.
    pub target: f32,
    /// Coefficient magnitude below which the edit does not fire. Non-positive
    /// fires always. Evaluated inside the kernel, never by a host reading the
    /// coefficient back -- that would cost a synchronization per layer per
    /// token.
    pub gate_threshold: f32,
}

/// Applies one directional-steering edit to `params.rows` rows of a residual
/// stream, in place, and writes each row's pre-edit unit coefficient to
/// `coeff`.
///
/// See the shader for the math, for why the mode is a uniform rather than a
/// function constant, and for the FP16 overflow hazard that
/// [`SteeringMode::Add`] and [`SteeringMode::Clamp`] carry and
/// [`SteeringMode::Ablate`] does not. The contract is
/// `turbospark_compute::steering::steer_in_place`.
///
/// `coeff` must hold at least `params.rows` FP32 elements and must be bound
/// even when the caller ignores it: an unbound Metal buffer argument is
/// undefined behaviour, not an empty one.
pub fn encode_steer_direction(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: (&metal::Buffer, u64),
    direction: (&metal::Buffer, u64),
    coeff: (&metal::Buffer, u64),
    params: &SteerParams,
) -> Result<(), GpuError> {
    // AGENTS.md/CLAUDE.md S8: `GpuError::PipelineCreate` for a shape
    // violation is this crate's existing spelling (`vision.rs`,
    // `gdn_shape.rs::validate`) -- a bad steering direction should refuse
    // to dispatch, not abort the process.
    if params.d_len == 0 {
        return Err(GpuError::PipelineCreate(
            "steering direction is empty".to_string(),
        ));
    }
    if params.rows == 0 {
        return Err(GpuError::PipelineCreate(
            "steering dispatch has no rows".to_string(),
        ));
    }
    if params.row_stride < params.d_len {
        return Err(GpuError::PipelineCreate(format!(
            "steering row stride {} is shorter than the direction ({}), so rows would overlap",
            params.row_stride, params.d_len
        )));
    }

    let pipeline = context.pipeline(
        SOURCE,
        "steer_direction_fp16",
        &FunctionConstantValues::new(),
        b"",
    )?;
    let mode = params.mode.as_u32();
    pass.encode_threads_3d(
        &pipeline,
        &[
            (x.0, 0, x.1),
            (direction.0, 1, direction.1),
            (coeff.0, 2, coeff.1),
        ],
        &[
            (u32_bytes(&params.d_len), 3),
            (u32_bytes(&params.row_stride), 4),
            (u32_bytes(&mode), 5),
            (crate::bytes::f32_bytes(&params.alpha), 6),
            (crate::bytes::f32_bytes(&params.inv_norm), 7),
            (crate::bytes::f32_bytes(&params.target), 8),
            (crate::bytes::f32_bytes(&params.gate_threshold), 9),
        ],
        // One threadgroup per row: the kernel reduces across the whole row,
        // so a row is the unit of work and `threadgroup_position_in_grid`
        // is the row index. The threadgroup width must stay at
        // THREADS_PER_GROUP -- the shader sizes its partial-sum array for
        // 256 threads (8 SIMD groups) and widening it here would overrun.
        (params.rows as u64 * THREADS_PER_GROUP, 1, 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}

/// `logits[i] = softcap * tanh(logits[i] / softcap)`, in place -- the output
/// head's final step for architectures with a logit softcap (see the
/// shader's port-local note on why the cap is dispatched without the
/// softmax `logit_softcap_softmax` fuses onto it).
pub fn encode_logit_softcap(
    context: &mut MetalContext,
    pass: &PassEncoder,
    logits: (&metal::Buffer, u64),
    softcap: f32,
    count: u32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "logit_softcap_fp16",
        &FunctionConstantValues::new(),
        b"",
    )?;
    pass.encode_threads_3d(
        &pipeline,
        &[(logits.0, 0, logits.1)],
        &[
            (crate::bytes::f32_bytes(&softcap), 1),
            (u32_bytes(&count), 2),
        ],
        (grid_for(count), 1, 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}
