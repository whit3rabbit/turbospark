//! Host-side dispatch for `shaders/utility.metal` (vendored from
//! `Metal/Primitives/utility.metal`, plus the `gelu_pytorch_tanh` helper it
//! borrows from `moe.metal` — see the shader's own header). These are the
//! small elementwise kernels that keep activations on the GPU between the
//! big kernels: gated-FFN activation multiplies and the residual add.
//!
//! `sigmoid_gate_mul_fp16`, `sigmoid_scalar_mul_fp16`, and
//! `split_q_gate_fp16` (all Qwen 3.6-specific) are vendored but not yet
//! dispatched — no Qwen path exists in this port yet.

use metal::FunctionConstantValues;

use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext, PassEncoder};

const SOURCE: &str = include_str!("shaders/utility.metal");
const THREADS_PER_GROUP: u64 = 256;

fn grid_for(count: u32) -> u64 {
    (count as u64).div_ceil(THREADS_PER_GROUP) * THREADS_PER_GROUP
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
