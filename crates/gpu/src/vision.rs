//! Host-side dispatch for `shaders/vision.metal`, the `qwen3_5` vision tower
//! (ROADMAP M-V2).
//!
//! PORT-LOCAL: the Swift engine has no vision tower, so nothing here is
//! vendored and there is no upstream kernel to diff against.
//! `turbospark_compute::vision` is the only definition of what these compute
//! and `tests/vision_parity.rs` is what holds them to it.
//!
//! # Every weight is FP16
//!
//! Every other learned weight bound in this crate is BF16, because every
//! other family's resident tensors are stored that way. The vision tower is
//! signed off at FP16 end to end (`docs/VISION_PHASE0.md`): its checkpoints
//! ship F16, the extreme-page probe puts peak activations at 13.8% of FP16's
//! ceiling with a factor of 7.3 in hand, and INT4 was measured and rejected
//! on OCR quality. The two types are the same WIDTH, so binding a BF16
//! tensor at one of these buffers passes every length check and reads the
//! bytes as a different number.

use metal::FunctionConstantValues;

use crate::bytes::{f32_bytes, u32_bytes};
use crate::context::{GpuError, MetalContext, PassEncoder};

const SOURCE: &str = include_str!("shaders/vision.metal");
const THREADS_PER_GROUP: u64 = 256;

/// The head-dimension ceiling `vision_attention_bidir_fp16` can address.
///
/// Its register tile gives each lane `head_dim / 32` accumulators and the
/// shader sizes that at 4. The tower runs at 72 (hidden 1152 over 16 heads),
/// so this is comfortable; it is a hard limit rather than a tuning knob,
/// because exceeding it truncates the head silently.
pub const MAX_ATTENTION_HEAD_DIM: u32 = 128;

/// No kernel in this file declares a function constant, but
/// `MetalContext::pipeline` takes a set regardless. An empty one, named so
/// the call sites read as a deliberate choice rather than an omission.
fn no_function_constants() -> FunctionConstantValues {
    FunctionConstantValues::new()
}

/// The number of threadgroups needed to cover `count` elements at
/// [`THREADS_PER_GROUP`].
fn groups_for(count: u32) -> u64 {
    (count as u64).div_ceil(THREADS_PER_GROUP)
}

/// LayerNorm over `rows` rows of `d` elements: `(x - mean) / sqrt(var + eps)
/// * weight + bias`, FP16 in and out with an FP32 accumulator.
///
/// NOT RMSNorm. The mean is subtracted and a bias is added, which is what
/// `nn.LayerNorm` does and what every other norm in this crate does not.
#[allow(clippy::too_many_arguments)]
pub fn encode_vision_layer_norm(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: (&metal::Buffer, u64),
    weight: (&metal::Buffer, u64),
    bias: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    rows: u32,
    d: u32,
    eps: f32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "vision_layer_norm_fp16",
        &no_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[
            (x.0, 0, x.1),
            (weight.0, 1, weight.1),
            (bias.0, 2, bias.1),
            (out.0, 3, out.1),
        ],
        &[(u32_bytes(&d), 4), (f32_bytes(&eps), 5)],
        rows as u64,
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// Which GELU. The tower uses BOTH -- tanh in each block's MLP, the exact erf
/// form in the merger -- and they agree to about 3e-4, so this is a choice a
/// caller has to make per call site rather than a detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeluKind {
    /// `nn.GELU(approx="tanh")`, the per-block MLP's activation.
    Tanh,
    /// A bare `nn.GELU()`, i.e. the exact erf form. The merger's activation.
    Erf,
}

impl GeluKind {
    fn kernel(self) -> &'static str {
        match self {
            // SEPARATE KERNELS rather than one with a mode uniform. A distinct
            // name is a distinct pipeline by construction, where a mode byte
            // missing from `pipeline`'s constants key would silently reuse
            // whichever compiled first (Gotcha 1) -- and here that would make
            // the tower's two activations one function.
            GeluKind::Tanh => "vision_gelu_tanh_fp16",
            GeluKind::Erf => "vision_gelu_erf_fp16",
        }
    }
}

/// Apply a GELU elementwise, in place, over `count` FP16 values.
pub fn encode_vision_gelu(
    context: &mut MetalContext,
    pass: &PassEncoder,
    y: (&metal::Buffer, u64),
    count: u32,
    kind: GeluKind,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(SOURCE, kind.kernel(), &no_function_constants(), b"")?;
    pass.encode_threadgroups(
        &pipeline,
        &[(y.0, 0, y.1)],
        &[(u32_bytes(&count), 1)],
        groups_for(count),
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// Apply the tower's 2-D rotary embedding in place to a `[seq, heads,
/// head_dim]` FP16 buffer.
///
/// `freqs` is `[seq, head_dim / 2]`, one row per token shared by every head,
/// as `turbospark_vision_io::vision_rope_freq_rows` emits it. The pairing is
/// the NeoX half-split: element `i` rotates with `i + head_dim / 2`.
#[allow(clippy::too_many_arguments)]
pub fn encode_vision_rope_2d(
    context: &mut MetalContext,
    pass: &PassEncoder,
    qkv: (&metal::Buffer, u64),
    freqs: (&metal::Buffer, u64),
    seq: u32,
    heads: u32,
    head_dim: u32,
) -> Result<(), GpuError> {
    if head_dim % 2 != 0 {
        // `GpuError::PipelineCreate` for a shape violation is this crate's
        // existing spelling (`gdn_shape.rs::validate`), not a mis-selected
        // variant: the enum carries no shape arm and adding one would widen a
        // type every module here matches on.
        return Err(GpuError::PipelineCreate(format!(
            "vision rope head_dim {head_dim} must be even"
        )));
    }
    let pipeline =
        context.pipeline(SOURCE, "vision_rope_2d_fp16", &no_function_constants(), b"")?;
    let half = u64::from(head_dim / 2);
    pass.encode_threads_3d(
        &pipeline,
        &[(qkv.0, 0, qkv.1), (freqs.0, 1, freqs.1)],
        &[
            (u32_bytes(&seq), 2),
            (u32_bytes(&heads), 3),
            (u32_bytes(&head_dim), 4),
        ],
        (half, u64::from(heads), u64::from(seq)),
        (half.min(THREADS_PER_GROUP), 1, 1),
    );
    Ok(())
}

/// Bidirectional multi-head attention over one image's patches.
///
/// `q`, `k`, `v` and `out` are each `[seq, heads, head_dim]` FP16. No mask
/// and no KV cache: every patch attends to every patch, which is the whole
/// difference from `attention_decode`.
#[allow(clippy::too_many_arguments)]
pub fn encode_vision_attention(
    context: &mut MetalContext,
    pass: &PassEncoder,
    q: (&metal::Buffer, u64),
    k: (&metal::Buffer, u64),
    v: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    seq: u32,
    heads: u32,
    head_dim: u32,
    scale: f32,
) -> Result<(), GpuError> {
    if head_dim == 0 || head_dim > MAX_ATTENTION_HEAD_DIM {
        // Refused rather than clamped: the shader's register tile would
        // truncate the head and return a plausible, wrong vector.
        return Err(GpuError::PipelineCreate(format!(
            "vision attention head_dim {head_dim} must be in 1..={MAX_ATTENTION_HEAD_DIM}"
        )));
    }
    let pipeline = context.pipeline(
        SOURCE,
        "vision_attention_bidir_fp16",
        &no_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups_3d(
        &pipeline,
        &[
            (q.0, 0, q.1),
            (k.0, 1, k.1),
            (v.0, 2, v.1),
            (out.0, 3, out.1),
        ],
        &[
            (u32_bytes(&seq), 4),
            (u32_bytes(&heads), 5),
            (u32_bytes(&head_dim), 6),
            (f32_bytes(&scale), 7),
        ],
        (u64::from(seq), u64::from(heads), 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}

/// `out[m][n] = sum_k a[m][k] * b[n][k] + bias[n]`, all FP16 with an FP32
/// accumulator.
///
/// `b` is ROW-MAJOR BY OUTPUT (`[n, k]`), which is how an `nn.Linear` weight
/// ships. Serves every projection in the tower.
///
/// `bias` is bound whether or not it is used, because an unbound buffer at a
/// declared argument index is undefined behaviour; `has_bias` selects. A
/// caller with no bias may pass any valid buffer, including one of the
/// others.
#[allow(clippy::too_many_arguments)]
pub fn encode_vision_matmul(
    context: &mut MetalContext,
    pass: &PassEncoder,
    a: (&metal::Buffer, u64),
    b: (&metal::Buffer, u64),
    bias: Option<(&metal::Buffer, u64)>,
    out: (&metal::Buffer, u64),
    m: u32,
    k: u32,
    n: u32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(SOURCE, "vision_matmul_fp16", &no_function_constants(), b"")?;
    let has_bias: u32 = u32::from(bias.is_some());
    let bias_binding = bias.unwrap_or((a.0, a.1));
    pass.encode_threadgroups_3d(
        &pipeline,
        &[
            (a.0, 0, a.1),
            (b.0, 1, b.1),
            (bias_binding.0, 2, bias_binding.1),
            (out.0, 3, out.1),
        ],
        &[
            (u32_bytes(&k), 4),
            (u32_bytes(&n), 5),
            (u32_bytes(&has_bias), 6),
        ],
        (u64::from(m), u64::from(n), 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}

/// `y += x` over `count` FP16 values, with an FP32 intermediate.
pub fn encode_vision_residual_add(
    context: &mut MetalContext,
    pass: &PassEncoder,
    y: (&metal::Buffer, u64),
    x: (&metal::Buffer, u64),
    count: u32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "vision_residual_add_fp16",
        &no_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[(y.0, 0, y.1), (x.0, 1, x.1)],
        &[(u32_bytes(&count), 2)],
        groups_for(count),
        THREADS_PER_GROUP,
    );
    Ok(())
}
