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
        // AGENTS.md/CLAUDE.md S12: matches the per-head/grouped siblings'
        // `.min(dim)` dispatch width. A no-op for every real install (`d`
        // is always the model's hidden size, >= 1152 here); only a
        // synthetic fixture at `d < 256` could ever see this move.
        THREADS_PER_GROUP.min(d.max(1) as u64),
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
        // AGENTS.md/CLAUDE.md S12: see `encode_rms_norm_no_scale`'s
        // identical note.
        THREADS_PER_GROUP.min(d.max(1) as u64),
    );
    Ok(())
}

/// Encoder-level CENTERED scaled RMSNorm (`rmsnorm_bf16w_centered`):
/// `out[i] = x[i] * rsqrt(mean(x^2) + eps) * (1 + weight[i])`.
///
/// The `muse_glimmer` family's four per-layer norms
/// (`input_layernorm`, `post_attention_layernorm`,
/// `pre_feedforward_layernorm`, `post_feedforward_layernorm`), whose stored
/// weights are OFFSETS FROM UNITY. Its contract is
/// `turbospark_compute::rms_norm_centered`; PORT-LOCAL, since Swift reads no
/// architecture with this convention.
///
/// **Do not reach for this by family.** The same model's FINAL norm
/// (`model.norm.weight`) is a plain `nn.RMSNorm` and takes
/// [`encode_rms_norm_bf16w`] instead. Picking one per family rather than per
/// TENSOR gives a model that decodes and is wrong.
#[allow(clippy::too_many_arguments)]
pub fn encode_rms_norm_bf16w_centered(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: (&metal::Buffer, u64),
    weight: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    d: u32,
    eps: f32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "rmsnorm_bf16w_centered",
        &unused_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[(x.0, 0, x.1), (weight.0, 1, weight.1), (out.0, 2, out.1)],
        &[(u32_bytes(&d), 3), (f32_bytes(&eps), 4)],
        1,
        // AGENTS.md/CLAUDE.md S12: see `encode_rms_norm_no_scale`'s
        // identical note.
        THREADS_PER_GROUP.min(d.max(1) as u64),
    );
    Ok(())
}

/// Encoder-level per-head scaled RMSNorm (`rmsnorm_bf16w_perhead`): `x`
/// holds `num_heads * head_dim` halfs; each head is normalized
/// independently and multiplied by the shared `[head_dim]` BF16 weight
/// (Gemma 4's q_norm/k_norm). One threadgroup per head.
#[allow(clippy::too_many_arguments)]
pub fn encode_rms_norm_bf16w_perhead(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: (&metal::Buffer, u64),
    weight: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    num_heads: u32,
    head_dim: u32,
    eps: f32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "rmsnorm_bf16w_perhead",
        &unused_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[(x.0, 0, x.1), (weight.0, 1, weight.1), (out.0, 2, out.1)],
        &[(u32_bytes(&head_dim), 3), (f32_bytes(&eps), 4)],
        num_heads as u64,
        THREADS_PER_GROUP.min(head_dim.max(1) as u64),
    );
    Ok(())
}

/// Encoder-level CENTERED per-head scaled RMSNorm
/// (`rmsnorm_bf16w_perhead_centered`): [`encode_rms_norm_bf16w_perhead`] with
/// the effective scale `(1 + weight[i])`, for a per-head norm whose stored
/// weight is an OFFSET FROM UNITY.
///
/// The `qwen3_5` MTP head's `q_norm`/`k_norm`. **Do not reach for this by
/// family**: the TRUNK's per-head norms of the same names, in the same model,
/// are plain and take [`encode_rms_norm_bf16w_perhead`]. That is the same
/// per-TENSOR rule the whole-vector pair carries (AGENTS.md Gotcha 50), and
/// picking one per family gives a model that decodes and is wrong.
#[allow(clippy::too_many_arguments)]
pub fn encode_rms_norm_bf16w_perhead_centered(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: (&metal::Buffer, u64),
    weight: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    num_heads: u32,
    head_dim: u32,
    eps: f32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "rmsnorm_bf16w_perhead_centered",
        &unused_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[(x.0, 0, x.1), (weight.0, 1, weight.1), (out.0, 2, out.1)],
        &[(u32_bytes(&head_dim), 3), (f32_bytes(&eps), 4)],
        num_heads as u64,
        THREADS_PER_GROUP.min(head_dim.max(1) as u64),
    );
    Ok(())
}

/// Encoder-level grouped CENTERED RMSNorm (`rmsnorm_bf16w_grouped_centered`):
/// `x` holds `groups * group_dim` halfs; each `group_dim`-wide slice is
/// normalized independently, and the result is multiplied by a SINGLE
/// `groups * group_dim`-wide weight read at each element's own global index
/// (not shared across groups the way [`encode_rms_norm_bf16w_perhead_centered`]'s
/// per-head weight is). `qwen4_exp`'s `hc_norm` and PLE's
/// `norm_key`/`norm_query`/`norm_conv` (`docs/QWEN4_PHASE0.md` item 9, norm
/// taxonomy row 1). One threadgroup per group.
///
/// **Do not reach for this at `groups = 1` in place of
/// [`encode_rms_norm_bf16w_centered`].** The two are mathematically identical
/// there, which is exactly why a caller must not be trusted to set `groups`
/// per tensor: `hc_norm` and `pre_fc_norm_hidden` are two DIFFERENT tensors on
/// two different call sites, and the plain one dispatches the other function
/// by name.
#[allow(clippy::too_many_arguments)]
pub fn encode_rms_norm_bf16w_grouped_centered(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: (&metal::Buffer, u64),
    weight: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    groups: u32,
    group_dim: u32,
    eps: f32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "rmsnorm_bf16w_grouped_centered",
        &unused_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[(x.0, 0, x.1), (weight.0, 1, weight.1), (out.0, 2, out.1)],
        &[(u32_bytes(&group_dim), 3), (f32_bytes(&eps), 4)],
        groups as u64,
        THREADS_PER_GROUP.min(group_dim.max(1) as u64),
    );
    Ok(())
}

/// One-shot [`encode_rms_norm_bf16w_grouped_centered`] over host slices, for
/// the parity tests: `x` and `weight_bits` (BF16 bit patterns) are both
/// `[groups * group_dim]`, `group_dim = x.len() / groups`.
pub fn rms_norm_bf16w_grouped_centered(
    context: &mut MetalContext,
    x: &[f16],
    weight_bits: &[u16],
    groups: u32,
    eps: f32,
) -> Result<Vec<f16>, GpuError> {
    assert_eq!(x.len(), weight_bits.len());
    assert_eq!(
        x.len() % groups as usize,
        0,
        "x.len() must be a whole number of groups"
    );
    let group_dim = (x.len() / groups as usize) as u32;
    let x_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(x));
    let w_buffer = context.new_buffer_with_data(&crate::bytes::u16_slice_to_le_bytes(weight_bits));
    let out_buffer = context.new_output_buffer((x.len() * 2) as u64);

    let pass = context.begin_pass();
    encode_rms_norm_bf16w_grouped_centered(
        context,
        &pass,
        (&x_buffer, 0),
        (&w_buffer, 0),
        (&out_buffer, 0),
        groups,
        group_dim,
        eps,
    )?;
    pass.commit_and_wait();
    Ok(read_half_buffer(&out_buffer, x.len()))
}

/// Encoder-level per-head no-scale RMSNorm (`rmsnorm_no_scale_perhead`):
/// Gemma 4's v_norm. One threadgroup per head.
pub fn encode_rms_norm_no_scale_perhead(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    num_heads: u32,
    head_dim: u32,
    eps: f32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "rmsnorm_no_scale_perhead",
        &unused_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[(x.0, 0, x.1), (out.0, 1, out.1)],
        &[(u32_bytes(&head_dim), 2), (f32_bytes(&eps), 3)],
        num_heads as u64,
        THREADS_PER_GROUP.min(head_dim.max(1) as u64),
    );
    Ok(())
}

/// One-shot [`encode_rms_norm_bf16w_perhead`] over host slices, for the
/// parity tests: `x` is `[num_heads * head_dim]`, `weight_bits` is the
/// shared `[head_dim]` BF16 weight as raw bit patterns.
pub fn rms_norm_bf16w_perhead(
    context: &mut MetalContext,
    x: &[f16],
    weight_bits: &[u16],
    num_heads: u32,
    eps: f32,
) -> Result<Vec<f16>, GpuError> {
    let head_dim = weight_bits.len() as u32;
    assert_eq!(x.len(), (num_heads * head_dim) as usize);
    let x_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(x));
    let w_buffer = context.new_buffer_with_data(&crate::bytes::u16_slice_to_le_bytes(weight_bits));
    let out_buffer = context.new_output_buffer((x.len() * 2) as u64);

    let pass = context.begin_pass();
    encode_rms_norm_bf16w_perhead(
        context,
        &pass,
        (&x_buffer, 0),
        (&w_buffer, 0),
        (&out_buffer, 0),
        num_heads,
        head_dim,
        eps,
    )?;
    pass.commit_and_wait();
    Ok(read_half_buffer(&out_buffer, x.len()))
}

/// One-shot [`encode_rms_norm_no_scale_perhead`] over host slices.
pub fn rms_norm_no_scale_perhead(
    context: &mut MetalContext,
    x: &[f16],
    num_heads: u32,
    head_dim: u32,
    eps: f32,
) -> Result<Vec<f16>, GpuError> {
    assert_eq!(x.len(), (num_heads * head_dim) as usize);
    let x_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(x));
    let out_buffer = context.new_output_buffer((x.len() * 2) as u64);

    let pass = context.begin_pass();
    encode_rms_norm_no_scale_perhead(
        context,
        &pass,
        (&x_buffer, 0),
        (&out_buffer, 0),
        num_heads,
        head_dim,
        eps,
    )?;
    pass.commit_and_wait();
    Ok(read_half_buffer(&out_buffer, x.len()))
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
