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
/// because exceeding it truncates the head silently. The matrix-unit kernel
/// sizes its fragments for the same 128 (`kAttnMaxTiles`).
pub const MAX_ATTENTION_HEAD_DIM: u32 = 128;

/// Only the matrix-unit kernels declare function constants ([`gemm_pipeline`],
/// [`attention_pipeline`]), but `MetalContext::pipeline` takes a set
/// regardless. An empty one, named so the call sites read as a deliberate
/// choice rather than an omission.
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
    /// CLIP's `quick_gelu`, `x * sigmoid(1.702 x)`.
    Quick,
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
            GeluKind::Quick => "vision_gelu_quick_fp16",
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

/// Apply SiLU (`x * sigmoid(x)`) elementwise, in place, over `count` FP16
/// values: the Qwen2.5-VL tower's gated MLP activation, run on the gate
/// projection before it is multiplied by the up projection.
pub fn encode_vision_silu(
    context: &mut MetalContext,
    pass: &PassEncoder,
    y: (&metal::Buffer, u64),
    count: u32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(SOURCE, "vision_silu_fp16", &no_function_constants(), b"")?;
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
/// difference from `attention_decode`. Runs on the kernel
/// [`VisionAttentionKernel::from_env`] picks.
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
    encode_vision_attention_with(
        VisionAttentionKernel::from_env(),
        context,
        pass,
        q,
        k,
        v,
        out,
        seq,
        heads,
        head_dim,
        scale,
    )
}

/// Which kernel serves [`encode_vision_attention`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionAttentionKernel {
    /// `vision_attention_bidir_fp16_mma`, flash style on the matrix units
    /// (32 query rows per threadgroup). The default.
    Mma,
    /// `vision_attention_bidir_fp16`, one threadgroup per (query, head).
    /// Kept for A/B runs (`TURBOSPARK_VISION_ATTN_MMA=0`) and as the parity
    /// cross-check.
    Scalar,
}

impl VisionAttentionKernel {
    /// The process default: [`VisionAttentionKernel::Mma`] unless
    /// `TURBOSPARK_VISION_ATTN_MMA=0`. Read once per process.
    pub fn from_env() -> Self {
        static KERNEL: std::sync::OnceLock<VisionAttentionKernel> = std::sync::OnceLock::new();
        *KERNEL.get_or_init(|| {
            if std::env::var("TURBOSPARK_VISION_ATTN_MMA").as_deref() == Ok("0") {
                VisionAttentionKernel::Scalar
            } else {
                VisionAttentionKernel::Mma
            }
        })
    }
}

/// The matrix-unit attention's query rows per threadgroup (`kAttnBQ`) and
/// threadgroup size.
const ATTN_BQ: u32 = 32;
const ATTN_THREADS: u64 = 128;

/// [`encode_vision_attention`] on an explicit kernel, for parity tests and
/// benches that need both arms in one process.
///
/// A `head_dim` that is not a multiple of 4 runs on the scalar kernel even
/// under [`VisionAttentionKernel::Mma`]: the matrix-unit kernel stages rows
/// as half4. The tower cannot reach it (its 2-D rope already requires a
/// multiple of 4), and no `vision_attn_mma` line is printed for it.
#[allow(clippy::too_many_arguments)]
pub fn encode_vision_attention_with(
    kernel: VisionAttentionKernel,
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
    let buffers = [
        (q.0, 0, q.1),
        (k.0, 1, k.1),
        (v.0, 2, v.1),
        (out.0, 3, out.1),
    ];
    let constants = [
        (u32_bytes(&seq), 4),
        (u32_bytes(&heads), 5),
        (u32_bytes(&head_dim), 6),
        (f32_bytes(&scale), 7),
    ];
    if kernel == VisionAttentionKernel::Mma && head_dim % 4 == 0 {
        // Proof for the lab that the matrix-unit attention ran (the knob row
        // expects this line absent under `TURBOSPARK_VISION_ATTN_MMA=0`).
        foundation::engaged!("vision_attn_mma");
        let pipeline = attention_pipeline(context, head_dim)?;
        pass.encode_threadgroups_3d(
            &pipeline,
            &buffers,
            &constants,
            (u64::from(seq.div_ceil(ATTN_BQ)), u64::from(heads), 1),
            (ATTN_THREADS, 1, 1),
        );
    } else {
        let pipeline = context.pipeline(
            SOURCE,
            "vision_attention_bidir_fp16",
            &no_function_constants(),
            b"",
        )?;
        pass.encode_threadgroups_3d(
            &pipeline,
            &buffers,
            &constants,
            (u64::from(seq), u64::from(heads), 1),
            (THREADS_PER_GROUP, 1, 1),
        );
    }
    Ok(())
}

/// `vision_attention_bidir_fp16_mma` specialized on `FC_VISION_ATTN_HEAD_DIM`
/// (function constant 111). The head dimension is the whole cache key:
/// without it the first tower's head (72 on `qwen3_5`) would serve another's
/// (64 on `qwen3_vl`), reading every row at the wrong stride.
fn attention_pipeline(
    context: &mut MetalContext,
    head_dim: u32,
) -> Result<metal::ComputePipelineState, GpuError> {
    let values = FunctionConstantValues::new();
    values.set_constant_value_at_index(
        (&head_dim as *const u32).cast(),
        metal::MTLDataType::UInt,
        111,
    );
    context.pipeline(
        SOURCE,
        "vision_attention_bidir_fp16_mma",
        &values,
        &head_dim.to_le_bytes(),
    )
}

/// Which kernel serves [`encode_vision_matmul`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionMatmulKernel {
    /// `vision_matmul_fp16_mma`, the matrix-unit GEMM (64 x 64 tiles). The
    /// default.
    Mma,
    /// `vision_matmul_fp16`, one threadgroup per output element. Kept for
    /// A/B runs (`TURBOSPARK_VISION_MMA=0`) and as the parity cross-check.
    PerOutput,
}

impl VisionMatmulKernel {
    /// The process default: [`VisionMatmulKernel::Mma`] unless
    /// `TURBOSPARK_VISION_MMA=0`. Read once per process.
    pub fn from_env() -> Self {
        static KERNEL: std::sync::OnceLock<VisionMatmulKernel> = std::sync::OnceLock::new();
        *KERNEL.get_or_init(|| {
            if std::env::var("TURBOSPARK_VISION_MMA").as_deref() == Ok("0") {
                VisionMatmulKernel::PerOutput
            } else {
                VisionMatmulKernel::Mma
            }
        })
    }
}

/// The matrix-unit GEMM's output tile (`kGemmBM`, `kGemmBN`) and K step
/// (`kGemmBK`) in `vision.metal`, and its threadgroup size.
const GEMM_TILE: u32 = 64;
const GEMM_BK: u32 = 32;
const GEMM_THREADS: u64 = 128;

/// `out[m][n] = sum_k a[m][k] * b[n][k] + bias[n]`, all FP16 with an FP32
/// accumulator.
///
/// `b` is ROW-MAJOR BY OUTPUT (`[n, k]`), which is how an `nn.Linear` weight
/// ships. Serves every projection in the tower, on the kernel
/// [`VisionMatmulKernel::from_env`] picks.
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
    encode_vision_matmul_with(
        VisionMatmulKernel::from_env(),
        context,
        pass,
        a,
        b,
        bias,
        out,
        m,
        k,
        n,
    )
}

/// [`encode_vision_matmul`] on an explicit kernel, for parity tests and
/// benches that need both arms in one process.
#[allow(clippy::too_many_arguments)]
pub fn encode_vision_matmul_with(
    kernel: VisionMatmulKernel,
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
    let has_bias: u32 = u32::from(bias.is_some());
    let bias_binding = bias.unwrap_or((a.0, a.1));
    let buffers = [
        (a.0, 0, a.1),
        (b.0, 1, b.1),
        (bias_binding.0, 2, bias_binding.1),
        (out.0, 3, out.1),
    ];
    match kernel {
        VisionMatmulKernel::PerOutput => {
            let pipeline =
                context.pipeline(SOURCE, "vision_matmul_fp16", &no_function_constants(), b"")?;
            pass.encode_threadgroups_3d(
                &pipeline,
                &buffers,
                &[
                    (u32_bytes(&k), 4),
                    (u32_bytes(&n), 5),
                    (u32_bytes(&has_bias), 6),
                ],
                (u64::from(m), u64::from(n), 1),
                (THREADS_PER_GROUP, 1, 1),
            );
        }
        VisionMatmulKernel::Mma => {
            // Proof for the lab that the matrix-unit GEMM ran (the knob row
            // expects this line absent under `TURBOSPARK_VISION_MMA=0`).
            foundation::engaged!("vision_mma");
            let pipeline = gemm_pipeline(context, k % GEMM_BK == 0)?;
            pass.encode_threadgroups_3d(
                &pipeline,
                &buffers,
                &[
                    (u32_bytes(&k), 4),
                    (u32_bytes(&n), 5),
                    (u32_bytes(&has_bias), 6),
                    (u32_bytes(&m), 7),
                ],
                (
                    u64::from(n.div_ceil(GEMM_TILE)),
                    u64::from(m.div_ceil(GEMM_TILE)),
                    1,
                ),
                (GEMM_THREADS, 1, 1),
            );
        }
    }
    Ok(())
}

/// `vision_matmul_fp16_mma` specialized on `FC_VISION_GEMM_K_ALIGNED`
/// (function constant 110). The flag is the whole cache key: without it the
/// first K seen would serve every K, and an aligned build reading a K tail
/// runs past the end of each row.
fn gemm_pipeline(
    context: &mut MetalContext,
    k_aligned: bool,
) -> Result<metal::ComputePipelineState, GpuError> {
    let values = FunctionConstantValues::new();
    values.set_constant_value_at_index(
        (&k_aligned as *const bool).cast(),
        metal::MTLDataType::Bool,
        110,
    );
    context.pipeline(
        SOURCE,
        "vision_matmul_fp16_mma",
        &values,
        &[u8::from(k_aligned)],
    )
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

/// RMSNorm over `rows` rows of `d` FP16 elements, the Gemma 4 tower's norm:
/// `x * rsqrt(mean(x^2) + eps) * weight`, the weight applied directly, or
/// no weight at all when `weight` is `None` (the v norm and the projector's
/// pre-norm). A per-head norm is `rows = seq * heads, d = head_dim`.
///
/// NOT [`encode_vision_layer_norm`], which subtracts the mean and adds a
/// bias. Safe in place (`out == x`).
#[allow(clippy::too_many_arguments)]
pub fn encode_vision_rms_norm(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: (&metal::Buffer, u64),
    weight: Option<(&metal::Buffer, u64)>,
    out: (&metal::Buffer, u64),
    rows: u32,
    d: u32,
    eps: f32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "vision_rms_norm_fp16",
        &no_function_constants(),
        b"",
    )?;
    let has_weight: u32 = u32::from(weight.is_some());
    // Bound either way, for `encode_vision_matmul`'s bias reason.
    let weight_binding = weight.unwrap_or((x.0, x.1));
    pass.encode_threadgroups(
        &pipeline,
        &[
            (x.0, 0, x.1),
            (weight_binding.0, 1, weight_binding.1),
            (out.0, 2, out.1),
        ],
        &[
            (u32_bytes(&d), 3),
            (f32_bytes(&eps), 4),
            (u32_bytes(&has_weight), 5),
        ],
        rows as u64,
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// The Gemma 4 tower's axial rope, in place on a `[seq, heads, head_dim]`
/// FP16 buffer.
///
/// `freqs` is `[seq, head_dim / 2]` F32: the x half's `head_dim / 4` angles,
/// then the y half's. Each half of the head rotates within itself (element
/// `i` with `i + head_dim / 4`), which is NOT [`encode_vision_rope_2d`]'s
/// pairing across the whole head.
#[allow(clippy::too_many_arguments)]
pub fn encode_vision_rope_axial(
    context: &mut MetalContext,
    pass: &PassEncoder,
    qk: (&metal::Buffer, u64),
    freqs: (&metal::Buffer, u64),
    seq: u32,
    heads: u32,
    head_dim: u32,
) -> Result<(), GpuError> {
    if head_dim == 0 || head_dim % 4 != 0 {
        return Err(GpuError::PipelineCreate(format!(
            "axial vision rope head_dim {head_dim} must be a nonzero multiple of 4"
        )));
    }
    let pipeline = context.pipeline(
        SOURCE,
        "vision_rope_axial_fp16",
        &no_function_constants(),
        b"",
    )?;
    let pairs = u64::from(head_dim / 2);
    pass.encode_threads_3d(
        &pipeline,
        &[(qk.0, 0, qk.1), (freqs.0, 1, freqs.1)],
        &[
            (u32_bytes(&seq), 2),
            (u32_bytes(&heads), 3),
            (u32_bytes(&head_dim), 4),
        ],
        (pairs, u64::from(heads), u64::from(seq)),
        (pairs.min(THREADS_PER_GROUP), 1, 1),
    );
    Ok(())
}

/// `y[g, c] += mean_{k<4} x[4g + k, c]` over `groups` rows of `width` FP16
/// values: the MiniCPM-V window merger's residual. `x` holds `4 * groups` rows,
/// each window's four tokens consecutive.
pub fn encode_vision_window_mean4_add(
    context: &mut MetalContext,
    pass: &PassEncoder,
    y: (&metal::Buffer, u64),
    x: (&metal::Buffer, u64),
    groups: u32,
    width: u32,
) -> Result<(), GpuError> {
    let count = groups.checked_mul(width).ok_or_else(|| {
        GpuError::PipelineCreate(format!("window mean {groups} x {width} overflows u32"))
    })?;
    let pipeline = context.pipeline(
        SOURCE,
        "vision_window_mean4_add_fp16",
        &no_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[(y.0, 0, y.1), (x.0, 1, x.1)],
        &[(u32_bytes(&width), 2), (u32_bytes(&count), 3)],
        groups_for(count),
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// `y *= x` over `count` FP16 values, with an FP32 intermediate: the gated
/// MLP's `gelu(gate) * up`.
pub fn encode_vision_mul(
    context: &mut MetalContext,
    pass: &PassEncoder,
    y: (&metal::Buffer, u64),
    x: (&metal::Buffer, u64),
    count: u32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(SOURCE, "vision_mul_fp16", &no_function_constants(), b"")?;
    pass.encode_threadgroups(
        &pipeline,
        &[(y.0, 0, y.1), (x.0, 1, x.1)],
        &[(u32_bytes(&count), 2)],
        groups_for(count),
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// The LFM3-VL conv block's core: `out = c * depthwise_conv3x3(b * v)` where
/// `proj` is the `[seq, 3 * channels]` output of the block's `in_proj` split
/// as `b | c | v` along the row.
///
/// `seq = height * width`, tokens in merge-window order (see the kernel),
/// `w` is the `[channels, 3, 3]` FP16 depthwise weight. `height` and `width`
/// must each be a multiple of `merge`; the convolution is zero padded.
#[allow(clippy::too_many_arguments)]
pub fn encode_vision_gated_dwconv3x3(
    context: &mut MetalContext,
    pass: &PassEncoder,
    out: (&metal::Buffer, u64),
    proj: (&metal::Buffer, u64),
    w: (&metal::Buffer, u64),
    height: u32,
    width: u32,
    merge: u32,
    channels: u32,
) -> Result<(), GpuError> {
    if merge == 0 || height % merge != 0 || width % merge != 0 {
        return Err(GpuError::PipelineCreate(format!(
            "gated dwconv grid {height}x{width} is not whole {merge}x{merge} merge windows"
        )));
    }
    let count = u64::from(height) * u64::from(width) * u64::from(channels);
    let count = u32::try_from(count).map_err(|_| {
        GpuError::PipelineCreate(format!(
            "gated dwconv {height}x{width}x{channels} exceeds a 32-bit thread index"
        ))
    })?;
    let pipeline = context.pipeline(
        SOURCE,
        "vision_gated_dwconv3x3_fp16",
        &no_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[(out.0, 0, out.1), (proj.0, 1, proj.1), (w.0, 2, w.1)],
        &[
            (u32_bytes(&height), 3),
            (u32_bytes(&width), 4),
            (u32_bytes(&merge), 5),
            (u32_bytes(&channels), 6),
        ],
        groups_for(count),
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// The index value that makes [`encode_vision_gather_rows`] write a zero row.
pub const VISION_GATHER_ZERO: u32 = u32::MAX;

/// `out[r] = in[index[r]]` for `rows` rows of `cols` FP16 values, or an all
/// zero row where `index[r]` is [`VISION_GATHER_ZERO`]. `index` is a `u32`
/// buffer of `rows` entries; `out` must not alias `input`.
///
/// DeepSeek-OCR's SAM window partition (zero padded) and its inverse are this
/// gather with two index lists.
pub fn encode_vision_gather_rows(
    context: &mut MetalContext,
    pass: &PassEncoder,
    input: (&metal::Buffer, u64),
    index: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    rows: u32,
    cols: u32,
) -> Result<(), GpuError> {
    let count = u32::try_from(u64::from(rows) * u64::from(cols)).map_err(|_| {
        GpuError::PipelineCreate(format!(
            "vision gather {rows}x{cols} exceeds a 32-bit thread index"
        ))
    })?;
    let pipeline = context.pipeline(
        SOURCE,
        "vision_gather_rows_fp16",
        &no_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[
            (input.0, 0, input.1),
            (index.0, 1, index.1),
            (out.0, 2, out.1),
        ],
        &[(u32_bytes(&rows), 3), (u32_bytes(&cols), 4)],
        groups_for(count),
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// im2col over one NHWC `[height, width, chans]` FP16 image with a SQUARE
/// `ksize` kernel: `[out_h * out_w, ksize * ksize * chans]` rows whose columns
/// are `(tap_row, tap_col, channel)`, channel fastest, zero outside the image.
///
/// That column order is a `[O, kh, kw, I]` conv weight flattened row-major, so
/// the convolution is `encode_vision_matmul` of these rows against the weight
/// as it ships. `out_h` and `out_w` are the caller's own
/// `(size + 2 * pad - ksize) / stride + 1`, checked here against the buffers
/// they imply only by the 32-bit index bound.
#[allow(clippy::too_many_arguments)]
pub fn encode_vision_im2col(
    context: &mut MetalContext,
    pass: &PassEncoder,
    input: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    height: u32,
    width: u32,
    chans: u32,
    ksize: u32,
    stride: u32,
    pad: u32,
) -> Result<(u32, u32), GpuError> {
    if ksize == 0 || stride == 0 || height + 2 * pad < ksize || width + 2 * pad < ksize {
        return Err(GpuError::PipelineCreate(format!(
            "im2col kernel {ksize} stride {stride} pad {pad} does not fit a {height}x{width} image"
        )));
    }
    let out_h = (height + 2 * pad - ksize) / stride + 1;
    let out_w = (width + 2 * pad - ksize) / stride + 1;
    let row_width = u64::from(ksize) * u64::from(ksize) * u64::from(chans);
    let count = u32::try_from(u64::from(out_h) * u64::from(out_w) * row_width).map_err(|_| {
        GpuError::PipelineCreate(format!(
            "im2col output {out_h}x{out_w}x{row_width} exceeds a 32-bit thread index"
        ))
    })?;
    let pipeline = context.pipeline(SOURCE, "vision_im2col_fp16", &no_function_constants(), b"")?;
    pass.encode_threadgroups(
        &pipeline,
        &[(input.0, 0, input.1), (out.0, 1, out.1)],
        &[
            (u32_bytes(&height), 2),
            (u32_bytes(&width), 3),
            (u32_bytes(&chans), 4),
            (u32_bytes(&ksize), 5),
            (u32_bytes(&stride), 6),
            (u32_bytes(&pad), 7),
            (u32_bytes(&out_h), 8),
            (u32_bytes(&out_w), 9),
        ],
        groups_for(count),
        THREADS_PER_GROUP,
    );
    Ok((out_h, out_w))
}

/// The most keys (`gh * gw`) [`encode_vision_sam_attention`] holds a score row
/// for, and the head dimension it is written for.
pub const SAM_ATTENTION_MAX_KEYS: u32 = 4096;
pub const SAM_ATTENTION_HEAD_DIM: u32 = 64;

/// SAM attention with the decomposed relative-position bias computed in the
/// kernel (see `vision_sam_attention_fp16`).
///
/// `q`, `k`, `v` and `out` are `[windows * gh * gw, heads * 64]` FP16, a row
/// holding every head head-major. `rel_h` is `[2 * gh - 1, 64]` and `rel_w`
/// `[2 * gw - 1, 64]`. `windows` independent `gh x gw` grids run in one
/// dispatch (the partitioned window blocks); a global block is `windows = 1`
/// over the whole grid. Refused, never truncated: a grid side above 64, more
/// keys than the score row holds, or a head dimension other than 64.
#[allow(clippy::too_many_arguments)]
pub fn encode_vision_sam_attention(
    context: &mut MetalContext,
    pass: &PassEncoder,
    q: (&metal::Buffer, u64),
    k: (&metal::Buffer, u64),
    v: (&metal::Buffer, u64),
    rel_h: (&metal::Buffer, u64),
    rel_w: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    gh: u32,
    gw: u32,
    windows: u32,
    heads: u32,
    scale: f32,
) -> Result<(), GpuError> {
    if gh == 0 || gw == 0 || gh > 64 || gw > 64 || gh * gw > SAM_ATTENTION_MAX_KEYS {
        return Err(GpuError::PipelineCreate(format!(
            "SAM attention grid {gh}x{gw} must have sides in 1..=64 and at most \
             {SAM_ATTENTION_MAX_KEYS} keys"
        )));
    }
    if heads == 0 || windows == 0 {
        return Err(GpuError::PipelineCreate(
            "SAM attention needs at least one head and one window".to_string(),
        ));
    }
    let pipeline = context.pipeline(
        SOURCE,
        "vision_sam_attention_fp16",
        &no_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups_3d(
        &pipeline,
        &[
            (q.0, 0, q.1),
            (k.0, 1, k.1),
            (v.0, 2, v.1),
            (rel_h.0, 3, rel_h.1),
            (rel_w.0, 4, rel_w.1),
            (out.0, 5, out.1),
        ],
        &[
            (u32_bytes(&gh), 6),
            (u32_bytes(&gw), 7),
            (u32_bytes(&heads), 8),
            (f32_bytes(&scale), 9),
        ],
        (u64::from(gh * gw), u64::from(heads), u64::from(windows)),
        (128, 1, 1),
    );
    Ok(())
}

/// CLIP's activation, in place, over `count` FP16 values
/// (`compute::uocr2_tower::quick_gelu`).
pub fn encode_vision_quick_gelu(
    context: &mut MetalContext,
    pass: &PassEncoder,
    y: (&metal::Buffer, u64),
    count: u32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "vision_quick_gelu_fp16",
        &no_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[(y.0, 0, y.1)],
        &[(u32_bytes(&count), 1)],
        groups_for(count),
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// One convolution's shape, MLX's layouts: `x` `[h, w, c_in]`, `weight`
/// `[c_out, kh, kw, c_in]`, zero padding.
#[derive(Debug, Clone, Copy)]
pub struct VisionConvShape {
    pub h: u32,
    pub w: u32,
    pub c_in: u32,
    pub c_out: u32,
    pub kh: u32,
    pub kw: u32,
    pub stride: u32,
    pub pad: u32,
}

impl VisionConvShape {
    /// Output `(h, w)`.
    pub fn output(&self) -> (u32, u32) {
        (
            (self.h + 2 * self.pad - self.kh) / self.stride + 1,
            (self.w + 2 * self.pad - self.kw) / self.stride + 1,
        )
    }
}

/// `compute::uocr2_tower::conv2d`. `bias` is bound either way, as in
/// [`encode_vision_matmul`].
pub fn encode_vision_conv2d(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: (&metal::Buffer, u64),
    weight: (&metal::Buffer, u64),
    bias: Option<(&metal::Buffer, u64)>,
    out: (&metal::Buffer, u64),
    shape: VisionConvShape,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(SOURCE, "vision_conv2d_fp16", &no_function_constants(), b"")?;
    let (oh, ow) = shape.output();
    let has_bias = u32::from(bias.is_some());
    let bind = bias.unwrap_or(weight);
    let words = |v: [u32; 4]| -> Vec<u8> { v.iter().flat_map(|w| w.to_le_bytes()).collect() };
    let in_hwc = words([shape.h, shape.w, shape.c_in, has_bias]);
    let k_hw = words([shape.kh, shape.kw, shape.stride, shape.pad]);
    let out_hwc = words([oh, ow, shape.c_out, 0]);
    pass.encode_threads_3d(
        &pipeline,
        &[
            (x.0, 0, x.1),
            (weight.0, 1, weight.1),
            (bind.0, 2, bind.1),
            (out.0, 3, out.1),
        ],
        &[(&in_hwc, 4), (&k_hw, 5), (&out_hwc, 6)],
        (u64::from(ow), u64::from(oh), u64::from(shape.c_out)),
        (u64::from(ow).min(THREADS_PER_GROUP), 1, 1),
    );
    Ok(())
}

/// SAM's windowed attention with its decomposed relative-position bias
/// (`compute::uocr2_tower::relpos_attention`). `seq` is a whole number of
/// `side * side` windows, and `rel_h`/`rel_w` are `[side, side, head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn encode_vision_attention_relpos(
    context: &mut MetalContext,
    pass: &PassEncoder,
    q: (&metal::Buffer, u64),
    k: (&metal::Buffer, u64),
    v: (&metal::Buffer, u64),
    rel_h: (&metal::Buffer, u64),
    rel_w: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    seq: u32,
    heads: u32,
    head_dim: u32,
    side: u32,
    scale: f32,
) -> Result<(), GpuError> {
    if head_dim == 0 || head_dim > MAX_ATTENTION_HEAD_DIM || seq % (side * side) != 0 {
        return Err(GpuError::PipelineCreate(format!(
            "relpos attention: head_dim {head_dim} must be in 1..={MAX_ATTENTION_HEAD_DIM} and \
             seq {seq} a multiple of {side}^2"
        )));
    }
    let pipeline = context.pipeline(
        SOURCE,
        "vision_attention_relpos_fp16",
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
            (rel_h.0, 8, rel_h.1),
            (rel_w.0, 9, rel_w.1),
        ],
        &[
            (u32_bytes(&seq), 4),
            (u32_bytes(&heads), 5),
            (u32_bytes(&head_dim), 6),
            (f32_bytes(&scale), 7),
            (u32_bytes(&side), 10),
        ],
        (u64::from(seq), u64::from(heads), 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}
