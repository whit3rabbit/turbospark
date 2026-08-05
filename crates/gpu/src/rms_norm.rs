//! Host-side dispatch for the `rmsnorm_no_scale` kernel in
//! `shaders/rmsnorm.metal` (vendored verbatim from
//! `Metal/Primitives/rmsnorm.metal` — Mference already compiles its MSL
//! shaders from source at startup, so the shader text itself needs no
//! translation, only a Rust-side dispatch).
//!
//! This is the first of the fourteen `.metal` shader files to get a wired
//! dispatch path; it is deliberately the simplest kernel (single input
//! buffer, no learned weight, no quantization) so it can serve as an
//! end-to-end proof that shader compile -> buffer upload -> dispatch ->
//! readback -> CPU-reference parity works on real hardware before the
//! remaining, much larger kernels (attention, MoE, quantized GEMV, the
//! chunked prefill pipeline) are ported.

use half::f16;
use metal::{FunctionConstantValues, MTLDataType};

use crate::bytes::{f32_bytes, half_slice_to_le_bytes, read_half_buffer, u32_bytes};
use crate::context::{dispatch_one_threadgroup_per_row, GpuError, MetalContext, PassEncoder};

const SOURCE: &str = include_str!("shaders/rmsnorm.metal");
const THREADS_PER_GROUP: u64 = 256;

/// `rmsnorm.metal`'s kernels declare (but conditionally read) function
/// constants `FC_RMS_D` (index 30, uint) and `FC_RMS_USE_FC` (index 31,
/// bool). Setting `FC_RMS_USE_FC = false` keeps every kernel on its normal
/// runtime-`D` path (see `MetalContext::pipeline`'s docs for why both must
/// be specialized regardless).
fn unused_function_constants() -> FunctionConstantValues {
    let values = FunctionConstantValues::new();
    let d: u32 = 0;
    let use_fc = false;
    values.set_constant_value_at_index((&d as *const u32).cast(), MTLDataType::UInt, 30);
    values.set_constant_value_at_index((&use_fc as *const bool).cast(), MTLDataType::Bool, 31);
    values
}

/// Encoder-level variant of [`rms_norm_no_scale`]: reads `D` halfs at
/// `x`, writes `D` halfs at `out`, both inside caller-owned buffers, as
/// one dispatch appended to `pass`.
pub fn encode_rms_norm_no_scale(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    d: u32,
    eps: f32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "rmsnorm_no_scale",
        &unused_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[(x.0, 0, x.1), (out.0, 1, out.1)],
        &[(u32_bytes(&d), 2), (f32_bytes(&eps), 3)],
        1,
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// Encoder-level scaled RMSNorm (`rmsnorm_bf16w`): `out[i] = x[i] *
/// rsqrt(mean(x^2) + eps) * weight[i]`, with `weight` a BF16 vector bound
/// in place (normally an offset into the resident weights buffer) — the
/// learned-norm form real Gemma 4 / Llama checkpoints need.
#[allow(clippy::too_many_arguments)]
pub fn encode_rms_norm_bf16w(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: (&metal::Buffer, u64),
    weight: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    d: u32,
    eps: f32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(SOURCE, "rmsnorm_bf16w", &unused_function_constants(), b"")?;
    pass.encode_threadgroups(
        &pipeline,
        &[(x.0, 0, x.1), (weight.0, 1, weight.1), (out.0, 2, out.1)],
        &[(u32_bytes(&d), 3), (f32_bytes(&eps), 4)],
        1,
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// `y[i] = x[i] * rsqrt(mean(x^2) + eps)`, dispatched on the GPU via
/// `rmsnorm_no_scale`. `x.len()` is the row width `D`.
pub fn rms_norm_no_scale(
    context: &mut MetalContext,
    x: &[f16],
    eps: f32,
) -> Result<Vec<f16>, GpuError> {
    let d = x.len() as u32;
    let x_bytes = half_slice_to_le_bytes(x);
    let x_buffer = context.new_buffer_with_data(&x_bytes);
    let out_buffer = context.new_output_buffer((x.len() * std::mem::size_of::<u16>()) as u64);

    let pipeline = context.pipeline(
        SOURCE,
        "rmsnorm_no_scale",
        &unused_function_constants(),
        b"",
    )?;
    dispatch_one_threadgroup_per_row(
        context,
        &pipeline,
        &[(&x_buffer, 0), (&out_buffer, 1)],
        &[(u32_bytes(&d), 2), (f32_bytes(&eps), 3)],
        1,
        THREADS_PER_GROUP.min(d.max(1) as u64),
    );

    Ok(read_half_buffer(&out_buffer, x.len()))
}
