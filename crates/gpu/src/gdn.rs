//! Host-side dispatch for `shaders/gdn.metal` (vendored verbatim from
//! `Metal/GDN/gdn.metal`): the gated-DeltaNet linear-attention chain Qwen
//! 3.6's mask-2 layers run. Eight kernels -- the fused four-way input
//! projection, causal depthwise conv (decode / prefill / tail update), the
//! per-head q/k norm, the gated delta recurrence (decode / prefill), and
//! the gated output norm.
//!
//! **This module's shader source is a CONCATENATION.** `gdn.metal`'s fused
//! input projection calls `dequant_int4_gemv_simd_body`, a `static inline`
//! defined in `dequant_int4.metal`: the Swift build concatenates every
//! shader module into one library, so the call resolves there. This port
//! compiles one library per file, so [`SOURCE`] glues those two files
//! together at compile time. `concat!` of two `include_str!`s is a single
//! `&'static str` constant with one stable address, which is exactly what
//! `MetalContext`'s address-keyed pipeline cache needs (crate Gotcha 1) --
//! but it does mean the INT4 kernels get compiled a second time in this
//! library, and that a caller must never pass `dequant_int4.metal`'s own
//! constant here (it would miss the cache, not misbehave).
//!
//! Buffer layouts follow the projections directly:
//! - `mixed_qkv` / `conv_out` rows: `[q: Hk*Dk][k: Hk*Dk][v: Hv*Dv]`
//! - recurrent state: FP32 `[Hv, Dv, Dk]` (owned by [`crate::GdnStateManager`])
//! - conv tail: FP16 `[conv_kernel_size - 1, qkv_dim]`, RAW (pre-conv) rows
//!
//! Function constants 90-94 (the input projection's constant-folded row
//! counts) are declared but left unspecialized: every dispatch here takes
//! its shape at runtime, so `FC_GDN_IN_USE_FC` is false and the constants
//! key is empty.

use metal::{FunctionConstantValues, MTLDataType};

use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::dequant_int4_gemv::Int4ResidentMatrix;

/// See the module doc: `dequant_int4.metal` FIRST, because `gdn.metal`
/// calls into it.
const SOURCE: &str = concat!(
    include_str!("shaders/dequant_int4.metal"),
    "\n",
    include_str!("shaders/gdn.metal")
);

/// `gdn_qk_norm` and `gdn_gated_norm` reduce four SIMD partials with a
/// hardcoded `for (i = 0; i < 4; ++i)` loop over `partial[]`. They are
/// correct at 128 threads per threadgroup and ONLY at 128: fewer leaves
/// uninitialized partials in the sum, more drops the extra SIMDs' work.
const NORM_THREADS: u64 = 128;
/// One SIMD group (32 lanes) covers a row of `S`; four `dv` rows share a
/// threadgroup.
const DELTA_THREADS: (u64, u64, u64) = (32, 4, 1);
const ROWS_PER_IN_PROJ_TG: u64 = 8;
const IN_PROJ_THREADS: u64 = 256;
const CONV_THREADS: u64 = 256;

/// GDN head geometry, the dispatch-side mirror of
/// `model_io::LinearAttentionConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnShape {
    pub num_k_heads: u32,
    pub num_v_heads: u32,
    pub key_head_dim: u32,
    pub value_head_dim: u32,
    pub conv_kernel_size: u32,
}

impl GdnShape {
    /// Conv channels, and the row width of `mixed_qkv` / `conv_out`.
    pub fn qkv_dim(&self) -> u32 {
        2 * self.num_k_heads * self.key_head_dim + self.num_v_heads * self.value_head_dim
    }

    /// `Hv * Dv`: z-gate width, `y` width, out_proj columns.
    pub fn value_dim(&self) -> u32 {
        self.num_v_heads * self.value_head_dim
    }

    /// The kernels' structural preconditions, duplicated from the Swift
    /// `GDN.init`. Each one is a silent-wrong-answer trap, not a crash:
    /// the delta kernel tiles `Dk` into 32 lanes of `float s[8]`, and its
    /// `dv` grid steps by 4.
    pub fn validate(&self) -> Result<(), GpuError> {
        let bad = |detail: String| Err(GpuError::PipelineCreate(detail));
        if self.key_head_dim == 0 || self.key_head_dim % 32 != 0 {
            return bad(format!(
                "key_head_dim {} must be a positive multiple of 32",
                self.key_head_dim
            ));
        }
        if self.key_head_dim / 32 > 8 {
            return bad(format!(
                "key_head_dim {} exceeds the kernel's 8-register lane tile",
                self.key_head_dim
            ));
        }
        if self.value_head_dim == 0 || self.value_head_dim % 4 != 0 {
            return bad(format!(
                "value_head_dim {} must be a positive multiple of 4",
                self.value_head_dim
            ));
        }
        if self.num_k_heads == 0 || self.num_v_heads % self.num_k_heads != 0 {
            return bad(format!(
                "num_v_heads {} must be a multiple of num_k_heads {}",
                self.num_v_heads, self.num_k_heads
            ));
        }
        if self.conv_kernel_size < 2 {
            return bad(format!(
                "conv_kernel_size {} must be at least 2",
                self.conv_kernel_size
            ));
        }
        Ok(())
    }

    fn head_dim_bytes(&self) -> [([u8; 4], u64); 4] {
        [
            (self.num_k_heads.to_le_bytes(), 7),
            (self.num_v_heads.to_le_bytes(), 8),
            (self.key_head_dim.to_le_bytes(), 9),
            (self.value_head_dim.to_le_bytes(), 10),
        ]
    }
}

/// Every function constant either shader in [`SOURCE`] declares: INT4's
/// 20-26 and GDN's 90-94. Metal requires all constants a function reads to
/// be set before a pipeline can be built, and setting the ones it does not
/// read is harmless -- the same defensive shape the other kernel modules
/// use. `USE_FC = false` keeps every kernel on its runtime-argument path.
fn gdn_function_constants() -> FunctionConstantValues {
    let values = FunctionConstantValues::new();
    let zero: u32 = 0;
    let use_fc = false;
    for index in [20, 21, 23, 24, 25, 90, 91, 92, 93] {
        values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, index);
    }
    for index in [22, 26, 94] {
        values.set_constant_value_at_index(
            (&use_fc as *const bool).cast(),
            MTLDataType::Bool,
            index,
        );
    }
    values
}

fn pipeline(
    context: &mut MetalContext,
    name: &'static str,
) -> Result<metal::ComputePipelineState, GpuError> {
    context.pipeline(SOURCE, name, &gdn_function_constants(), b"")
}

/// Fused `in_proj_qkv` / `_z` / `_a` / `_b` INT4 GEMV: one dispatch over
/// the concatenated row space instead of four, two of which
/// (`a` and `b`, `num_v_heads` rows each) would be near-empty launches.
/// Bit-identical to the four separate GEMVs -- the per-row body and its
/// operand order are unchanged.
#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_in_proj(
    context: &mut MetalContext,
    pass: &PassEncoder,
    qkv: &Int4ResidentMatrix<'_>,
    z: &Int4ResidentMatrix<'_>,
    a: &Int4ResidentMatrix<'_>,
    b: &Int4ResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    qkv_out: (&metal::Buffer, u64),
    z_out: (&metal::Buffer, u64),
    a_out: (&metal::Buffer, u64),
    b_out: (&metal::Buffer, u64),
) -> Result<(), GpuError> {
    let n = qkv.cols;
    assert_eq!(n % 64, 0, "hidden size must be a multiple of 64");
    for m in [z, a, b] {
        assert_eq!(m.cols, n, "all four input projections share one x");
    }
    assert_eq!(
        a.rows, b.rows,
        "a and b both project one row per value head"
    );
    // The row body reads packed weights through a `ushort*`: the repacker
    // guarantees 2-byte sub-tensor alignment but not 4-byte.
    for m in [qkv, z, a, b] {
        assert_eq!(
            m.weights_offset % 2,
            0,
            "gdn_in_proj_gemv_simd needs 2-aligned weight offsets"
        );
    }

    let p = pipeline(context, "gdn_in_proj_gemv_simd")?;
    let (qkv_rows, z_rows, ab_rows, n32) =
        (qkv.rows as u32, z.rows as u32, a.rows as u32, n as u32);
    let total_rows = (qkv_rows + z_rows + 2 * ab_rows) as u64;
    let mut buffers: Vec<(&metal::Buffer, u64, u64)> = Vec::with_capacity(17);
    for (slot, m) in [qkv, z, a, b].into_iter().enumerate() {
        let base = slot as u64 * 3;
        buffers.push((m.buffer, base, m.weights_offset));
        buffers.push((m.buffer, base + 1, m.scales_offset));
        buffers.push((m.buffer, base + 2, m.biases_offset));
    }
    buffers.push((x.0, 12, x.1));
    buffers.push((qkv_out.0, 13, qkv_out.1));
    buffers.push((z_out.0, 14, z_out.1));
    buffers.push((a_out.0, 15, a_out.1));
    buffers.push((b_out.0, 16, b_out.1));
    pass.encode_threadgroups(
        &p,
        &buffers,
        &[
            (u32_bytes(&qkv_rows), 17),
            (u32_bytes(&z_rows), 18),
            (u32_bytes(&ab_rows), 19),
            (u32_bytes(&n32), 20),
        ],
        total_rows.div_ceil(ROWS_PER_IN_PROJ_TG),
        IN_PROJ_THREADS,
    );
    Ok(())
}

/// Decode: `out = silu(causal_conv([tail | qkv]))`, shifting `tail` in
/// place afterwards. `tail` holds the RAW rows, not the activated ones.
#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_conv_decode(
    context: &mut MetalContext,
    pass: &PassEncoder,
    shape: GdnShape,
    tail: (&metal::Buffer, u64),
    qkv: (&metal::Buffer, u64),
    conv_weight: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
) -> Result<(), GpuError> {
    shape.validate()?;
    let p = pipeline(context, "gdn_conv_mix_decode")?;
    let (channels, taps) = (shape.qkv_dim(), shape.conv_kernel_size);
    pass.encode_threads_3d(
        &p,
        &[
            (tail.0, 0, tail.1),
            (qkv.0, 1, qkv.1),
            (conv_weight.0, 2, conv_weight.1),
            (out.0, 3, out.1),
        ],
        &[(u32_bytes(&channels), 4), (u32_bytes(&taps), 5)],
        (channels as u64, 1, 1),
        (CONV_THREADS, 1, 1),
    );
    Ok(())
}

/// Prefill: the same conv over `rows` chunk rows at once. `tail` is
/// READ-ONLY here -- [`encode_gdn_conv_tail_update`] refreshes it after.
#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_conv_prefill(
    context: &mut MetalContext,
    pass: &PassEncoder,
    shape: GdnShape,
    tail: (&metal::Buffer, u64),
    qkv_rows: (&metal::Buffer, u64),
    conv_weight: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    rows: u32,
) -> Result<(), GpuError> {
    shape.validate()?;
    let p = pipeline(context, "gdn_conv_mix_prefill")?;
    let (channels, taps) = (shape.qkv_dim(), shape.conv_kernel_size);
    pass.encode_threads_3d(
        &p,
        &[
            (tail.0, 0, tail.1),
            (qkv_rows.0, 1, qkv_rows.1),
            (conv_weight.0, 2, conv_weight.1),
            (out.0, 3, out.1),
        ],
        &[
            (u32_bytes(&channels), 4),
            (u32_bytes(&taps), 5),
            (u32_bytes(&rows), 6),
        ],
        (channels as u64, rows.max(1) as u64, 1),
        (CONV_THREADS, 1, 1),
    );
    Ok(())
}

/// After a prefill chunk: `tail := last (K - 1) raw rows of [tail | chunk]`.
pub fn encode_gdn_conv_tail_update(
    context: &mut MetalContext,
    pass: &PassEncoder,
    shape: GdnShape,
    tail: (&metal::Buffer, u64),
    qkv_rows: (&metal::Buffer, u64),
    rows: u32,
) -> Result<(), GpuError> {
    shape.validate()?;
    let p = pipeline(context, "gdn_conv_tail_update")?;
    let (channels, taps) = (shape.qkv_dim(), shape.conv_kernel_size);
    pass.encode_threads_3d(
        &p,
        &[(tail.0, 0, tail.1), (qkv_rows.0, 1, qkv_rows.1)],
        &[
            (u32_bytes(&channels), 2),
            (u32_bytes(&taps), 3),
            (u32_bytes(&rows), 4),
        ],
        (channels as u64, (taps - 1) as u64, 1),
        (CONV_THREADS, 1, 1),
    );
    Ok(())
}

/// Per-head no-weight RMS norm over the q and k slices of `conv_out`, IN
/// PLACE, with the delta-rule scales folded in (`q *= 1/Dk`,
/// `k *= 1/sqrt(Dk)`). The v slice is left alone.
pub fn encode_gdn_qk_norm(
    context: &mut MetalContext,
    pass: &PassEncoder,
    shape: GdnShape,
    conv_out: (&metal::Buffer, u64),
    rows: u32,
) -> Result<(), GpuError> {
    shape.validate()?;
    let p = pipeline(context, "gdn_qk_norm")?;
    let (k_heads, key_dim, row_stride) = (shape.num_k_heads, shape.key_head_dim, shape.qkv_dim());
    pass.encode_threadgroups_3d(
        &p,
        &[(conv_out.0, 0, conv_out.1)],
        &[
            (u32_bytes(&k_heads), 1),
            (u32_bytes(&key_dim), 2),
            (u32_bytes(&row_stride), 3),
        ],
        (2 * k_heads as u64, rows.max(1) as u64, 1),
        (NORM_THREADS, 1, 1),
    );
    Ok(())
}

/// One gated delta-rule step. `state` is FP32 `[Hv, Dv, Dk]`, updated in
/// place; `y` receives `[Hv * Dv]` halfs.
#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_delta_decode(
    context: &mut MetalContext,
    pass: &PassEncoder,
    shape: GdnShape,
    conv_out: (&metal::Buffer, u64),
    a_proj: (&metal::Buffer, u64),
    b_proj: (&metal::Buffer, u64),
    a_log: (&metal::Buffer, u64),
    dt_bias: (&metal::Buffer, u64),
    state: &metal::Buffer,
    y: (&metal::Buffer, u64),
) -> Result<(), GpuError> {
    shape.validate()?;
    let p = pipeline(context, "gdn_delta_step_decode")?;
    let dims = shape.head_dim_bytes();
    pass.encode_threadgroups_3d(
        &p,
        &[
            (conv_out.0, 0, conv_out.1),
            (a_proj.0, 1, a_proj.1),
            (b_proj.0, 2, b_proj.1),
            (a_log.0, 3, a_log.1),
            (dt_bias.0, 4, dt_bias.1),
            (state, 5, 0),
            (y.0, 6, y.1),
        ],
        &dims.iter().map(|(v, i)| (&v[..], *i)).collect::<Vec<_>>(),
        (
            shape.num_v_heads as u64,
            (shape.value_head_dim / 4) as u64,
            1,
        ),
        DELTA_THREADS,
    );
    Ok(())
}

/// The same recurrence with the token loop inside the kernel: state stays
/// in registers across the whole chunk and is written back once.
#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_delta_prefill(
    context: &mut MetalContext,
    pass: &PassEncoder,
    shape: GdnShape,
    conv_out: (&metal::Buffer, u64),
    a_proj: (&metal::Buffer, u64),
    b_proj: (&metal::Buffer, u64),
    a_log: (&metal::Buffer, u64),
    dt_bias: (&metal::Buffer, u64),
    state: &metal::Buffer,
    y: (&metal::Buffer, u64),
    rows: u32,
) -> Result<(), GpuError> {
    shape.validate()?;
    let p = pipeline(context, "gdn_delta_step_prefill")?;
    let dims = shape.head_dim_bytes();
    let row_stride = shape.qkv_dim();
    let mut bytes: Vec<(&[u8], u64)> = dims.iter().map(|(v, i)| (&v[..], *i)).collect();
    bytes.push((u32_bytes(&rows), 11));
    bytes.push((u32_bytes(&row_stride), 12));
    pass.encode_threadgroups_3d(
        &p,
        &[
            (conv_out.0, 0, conv_out.1),
            (a_proj.0, 1, a_proj.1),
            (b_proj.0, 2, b_proj.1),
            (a_log.0, 3, a_log.1),
            (dt_bias.0, 4, dt_bias.1),
            (state, 5, 0),
            (y.0, 6, y.1),
        ],
        &bytes,
        (
            shape.num_v_heads as u64,
            (shape.value_head_dim / 4) as u64,
            1,
        ),
        DELTA_THREADS,
    );
    Ok(())
}

/// `out = rmsnorm(y; weight) * silu(z)`, per value head, over `rows` rows.
#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_gated_norm(
    context: &mut MetalContext,
    pass: &PassEncoder,
    shape: GdnShape,
    y: (&metal::Buffer, u64),
    z: (&metal::Buffer, u64),
    weight: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    rows: u32,
) -> Result<(), GpuError> {
    shape.validate()?;
    let p = pipeline(context, "gdn_gated_norm")?;
    let (v_heads, value_dim) = (shape.num_v_heads, shape.value_head_dim);
    pass.encode_threadgroups_3d(
        &p,
        &[
            (y.0, 0, y.1),
            (z.0, 1, z.1),
            (weight.0, 2, weight.1),
            (out.0, 3, out.1),
        ],
        &[(u32_bytes(&v_heads), 4), (u32_bytes(&value_dim), 5)],
        (v_heads as u64, rows.max(1) as u64, 1),
        (NORM_THREADS, 1, 1),
    );
    Ok(())
}
