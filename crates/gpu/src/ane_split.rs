//! The Metal half of the opt-in ANE+Steel prefill split (feature
//! `ane-prefill`, off by default): dense qwen projections split by output
//! rows between the Neural Engine and Steel (`docs/SPEEDUP_PLAYBOOK.md`).
//!
//! The Neural Engine runs a Core ML program on F16 IOSurface planes that
//! Metal wraps without copying. This module owns the GPU work around it:
//!
//! - [`encode_ane_dequant_rows`]: INT4 weight rows to an F16 plane, so the
//!   program never sees the resident encoding (`main-npu`'s conversion).
//! - [`encode_ane_scale_rows`]: each token scaled by the power of two that
//!   puts its peak magnitude in [32, 64). The ANE flushes FP16 subnormals.
//! - [`encode_ane_convert_out`]: the program's `W * A^T` transposed into the
//!   engine's row-major output at its column offset, unscaled, raising a
//!   flag on any non-finite value so the caller can redo the micro-batch on
//!   Steel.
//! - [`encode_dequant_int4_gemm_steel_rows`]: Steel over a slice of output
//!   rows, written straight into the shared output.
//! - [`HostEvent`] and [`begin_pass_after`]: the GPU waits for the host's
//!   signal that the ANE rows are in place, so a segment can be encoded and
//!   committed before the prediction it waits on has run.
//!
//! The kernels are re-derived from the Phase 0 probe's (`probe.swift`), with
//! exact power-of-two scales (bit-built, not `exp2`) so the unscale is exact.

use metal::FunctionConstantValues;

use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::dequant_int4_gemv::Int4ResidentMatrix;

/// The command-queue capacity the feature build asks for. A split
/// micro-batch holds `2 * points + 1` uncommitted buffers at once, and
/// every class split on qwen38 is 4 points per layer, 513 buffers.
pub const MAX_COMMAND_BUFFERS: usize = 1024;

static SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float bf16_bits_to_float(ushort u) { return as_type<float>(uint(u) << 16); }

// Rows [0, rows) of an INT4 matrix (cols / 2 bytes per row, low nibble
// first; BF16 scale and bias per 64-wide group) -> F16 rows `stride` halfs
// apart. One thread expands one 32-bit word, 8 weights inside one group.
// Each weight is `q * scale + bias` in FP32 rounded once to F16, the
// operand Steel builds.
kernel void ane_dequant_rows_f16(
    device const uint* w [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const ushort* biases [[buffer(2)]],
    device half* dst [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& stride [[buffer(6)]],
    uint2 g [[thread_position_in_grid]]) {
    uint words = cols / 8;
    if (g.x >= words || g.y >= rows) return;
    uint packed = w[g.y * words + g.x];
    uint gi = g.y * (cols / 64) + (g.x * 8) / 64;
    float s = bf16_bits_to_float(scales[gi]);
    float b = bf16_bits_to_float(biases[gi]);
    half4 lo, hi;
    for (uint i = 0; i < 4; i++) {
        lo[i] = half(float((packed >> (4 * i)) & 0xFu) * s + b);
        hi[i] = half(float((packed >> (4 * (i + 4))) & 0xFu) * s + b);
    }
    device half4* out = (device half4*)(dst + g.y * stride + g.x * 8);
    out[0] = lo;
    out[1] = hi;
}

// One 256-thread threadgroup per token row of x (cols halfs per row):
// a[t, :] = x[t, :] * 2^e with e chosen from the row peak so the peak lands
// in [32, 64), and inv[t] = 2^-e. Both are built from exponent bits so the
// later unscale is exact.
kernel void ane_scale_rows(
    device const half* x [[buffer(0)]],
    device half* a [[buffer(1)]],
    device float* inv [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& stride [[buffer(4)]],
    uint t [[thread_index_in_threadgroup]],
    uint g [[threadgroup_position_in_grid]]) {
    threadgroup float peaks[8];
    threadgroup float scale_tg;
    device const half* row = x + ulong(g) * cols;
    float peak = 0.0f;
    for (uint i = t; i < cols; i += 256) peak = max(peak, abs(float(row[i])));
    peak = simd_max(peak);
    if (t % 32 == 0) peaks[t / 32] = peak;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float p = 0.0f;
        for (uint i = 0; i < 8; i++) p = max(p, peaks[i]);
        int e = 0;
        if (p > 0.0f && isfinite(p)) {
            int ex;
            frexp(p, ex);  // p = m * 2^ex, m in [0.5, 1): floor(log2 p) = ex - 1
            e = clamp(5 - (ex - 1), -126, 126);
        }
        scale_tg = as_type<float>(uint(127 + e) << 23);
        inv[g] = as_type<float>(uint(127 - e) << 23);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = scale_tg;
    device half* out = a + ulong(g) * stride;
    for (uint i = t; i < cols; i += 256) out[i] = half(float(row[i]) * s);
}

// o[rows, n] (`ostride` halfs per row) -> y[t * ldy + r] for t < n, times
// the token's exact inverse scale. 16 x 16 tiles through threadgroup memory
// so the read and the write are both coalesced. Any non-finite result sets
// `nonfinite`: the ANE can overflow F16 where Steel would not, and the host
// then redoes the micro-batch on Steel.
kernel void ane_convert_out(
    device const half* o [[buffer(0)]],
    device half* y [[buffer(1)]],
    device const float* inv [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& ostride [[buffer(5)]],
    constant uint& ldy [[buffer(6)]],
    device atomic_uint* nonfinite [[buffer(7)]],
    uint2 t [[thread_position_in_threadgroup]],
    uint2 g [[threadgroup_position_in_grid]]) {
    threadgroup half tile[16][17];
    uint r = g.y * 16 + t.y, c = g.x * 16 + t.x;
    tile[t.y][t.x] = (r < rows && c < n) ? o[ulong(r) * ostride + c] : half(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint tok = g.x * 16 + t.y, row = g.y * 16 + t.x;
    if (tok < n && row < rows) {
        half v = half(float(tile[t.x][t.y]) * inv[tok]);
        if (!isfinite(v)) {
            atomic_store_explicit(nonfinite, 1u, memory_order_relaxed);
        }
        y[ulong(tok) * ldy + row] = v;
    }
}
"#;

fn pipeline(
    context: &mut MetalContext,
    name: &'static str,
) -> Result<metal::ComputePipelineState, GpuError> {
    context.pipeline(SOURCE, name, &FunctionConstantValues::new(), &[])
}

fn u32_of(v: usize, what: &str) -> Result<u32, GpuError> {
    u32::try_from(v).map_err(|_| GpuError::InvalidInput(format!("{what} {v} exceeds u32")))
}

/// Dequantizes rows `[0, rows)` of `w` into the F16 plane `dst`, `stride`
/// halfs per row.
pub fn encode_ane_dequant_rows(
    context: &mut MetalContext,
    pass: &PassEncoder,
    w: &Int4ResidentMatrix<'_>,
    rows: usize,
    dst: (&metal::Buffer, u64),
    stride: usize,
) -> Result<(), GpuError> {
    if rows == 0 || rows > w.rows || w.cols % 64 != 0 || stride < w.cols || stride % 4 != 0 {
        return Err(GpuError::InvalidInput(format!(
            "ane dequant: {rows} of {} rows, {} cols, stride {stride}",
            w.rows, w.cols
        )));
    }
    if w.weights_offset % 4 != 0 || w.scales_offset % 2 != 0 || w.biases_offset % 2 != 0 {
        return Err(GpuError::InvalidInput(
            "ane dequant: weight, scale or bias offset is misaligned".into(),
        ));
    }
    if dst.0.length() < dst.1 + (((rows - 1) * stride + w.cols) * 2) as u64 {
        return Err(GpuError::InvalidInput(
            "ane dequant: plane too small".into(),
        ));
    }
    let pipe = pipeline(context, "ane_dequant_rows_f16")?;
    let (cols, rows32, stride32) = (
        u32_of(w.cols, "cols")?,
        u32_of(rows, "rows")?,
        u32_of(stride, "stride")?,
    );
    pass.encode_threads_3d(
        &pipe,
        &[
            (w.buffer, 0, w.weights_offset),
            (w.buffer, 1, w.scales_offset),
            (w.buffer, 2, w.biases_offset),
            (dst.0, 3, dst.1),
        ],
        &[
            (u32_bytes(&cols), 4),
            (u32_bytes(&rows32), 5),
            (u32_bytes(&stride32), 6),
        ],
        ((w.cols / 8) as u64, rows as u64, 1),
        (256, 1, 1),
    );
    Ok(())
}

/// Scales the first `m` token rows of `x` (`cols` halfs each) into the F16
/// plane `dst` (`stride` halfs per row) and writes each row's exact inverse
/// scale to `inv` (`m` f32s).
#[allow(clippy::too_many_arguments)]
pub fn encode_ane_scale_rows(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: (&metal::Buffer, u64),
    cols: usize,
    m: usize,
    dst: (&metal::Buffer, u64),
    stride: usize,
    inv: &metal::Buffer,
) -> Result<(), GpuError> {
    if m == 0 || stride < cols || inv.length() < (m * 4) as u64 {
        return Err(GpuError::InvalidInput(format!(
            "ane scale: {m} rows, {cols} cols, stride {stride}"
        )));
    }
    if x.0.length() < x.1 + (m * cols * 2) as u64
        || dst.0.length() < dst.1 + (((m - 1) * stride + cols) * 2) as u64
    {
        return Err(GpuError::InvalidInput("ane scale: buffer too small".into()));
    }
    let pipe = pipeline(context, "ane_scale_rows")?;
    let (cols32, stride32) = (u32_of(cols, "cols")?, u32_of(stride, "stride")?);
    pass.encode_threadgroups(
        &pipe,
        &[(x.0, 0, x.1), (dst.0, 1, dst.1), (inv, 2, 0)],
        &[(u32_bytes(&cols32), 3), (u32_bytes(&stride32), 4)],
        m as u64,
        256,
    );
    Ok(())
}

/// Writes `o[r, t]` (`rows` rows of the plane, `o_stride` halfs apart) to
/// `y[t * ldy + r]` for tokens `t < m`, times `inv[t]`, and stores 1 in the
/// `u32` at `nonfinite` if any written value is not finite. The caller
/// zeroes the flag.
#[allow(clippy::too_many_arguments)]
pub fn encode_ane_convert_out(
    context: &mut MetalContext,
    pass: &PassEncoder,
    o: (&metal::Buffer, u64),
    rows: usize,
    m: usize,
    o_stride: usize,
    inv: &metal::Buffer,
    y: (&metal::Buffer, u64),
    ldy: usize,
    nonfinite: &metal::Buffer,
) -> Result<(), GpuError> {
    if rows == 0
        || m == 0
        || o_stride < m
        || ldy < rows
        || inv.length() < (m * 4) as u64
        || nonfinite.length() < 4
    {
        return Err(GpuError::InvalidInput(format!(
            "ane convert: {rows} rows, {m} tokens, stride {o_stride}, ldy {ldy}"
        )));
    }
    if o.0.length() < o.1 + (((rows - 1) * o_stride + m) * 2) as u64
        || y.0.length() < y.1 + (((m - 1) * ldy + rows) * 2) as u64
    {
        return Err(GpuError::InvalidInput(
            "ane convert: buffer too small".into(),
        ));
    }
    let pipe = pipeline(context, "ane_convert_out")?;
    let (rows32, m32, os32, ldy32) = (
        u32_of(rows, "rows")?,
        u32_of(m, "tokens")?,
        u32_of(o_stride, "stride")?,
        u32_of(ldy, "ldy")?,
    );
    pass.encode_threadgroups_3d(
        &pipe,
        &[(o.0, 0, o.1), (y.0, 1, y.1), (inv, 2, 0), (nonfinite, 7, 0)],
        &[
            (u32_bytes(&rows32), 3),
            (u32_bytes(&m32), 4),
            (u32_bytes(&os32), 5),
            (u32_bytes(&ldy32), 6),
        ],
        (m.div_ceil(16) as u64, rows.div_ceil(16) as u64, 1),
        (16, 16, 1),
    );
    Ok(())
}

/// Steel over output rows `[row0, w.rows)` of `w`, written to
/// `y[t * ldy + row0 + r]`. The vendored kernel is instantiated
/// `aligned_N`, so its `N` argument is only the output row stride and the
/// grid bounds the rows; passing `ldy` there leaves the shader untouched.
#[allow(clippy::too_many_arguments)]
pub fn encode_dequant_int4_gemm_steel_rows(
    context: &mut MetalContext,
    pass: &PassEncoder,
    w: &Int4ResidentMatrix<'_>,
    row0: usize,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    ldy: usize,
    batch: usize,
) -> Result<(), GpuError> {
    const BM: usize = 32;
    const BN: usize = 32;
    let rows = w.rows.saturating_sub(row0);
    let group_bytes = (w.cols / 64 * 2) as u64;
    let view = Int4ResidentMatrix {
        buffer: w.buffer,
        weights_offset: w.weights_offset + (row0 * w.cols / 2) as u64,
        scales_offset: w.scales_offset + row0 as u64 * group_bytes,
        biases_offset: w.biases_offset + row0 as u64 * group_bytes,
        rows,
        cols: w.cols,
    };
    if batch == 0
        || ldy < w.rows
        || row0 % BN != 0
        || !crate::steel_supports(&view, crate::SteelTile::M32K32N32)
    {
        return Err(GpuError::InvalidInput(format!(
            "steel rows: rows [{row0}, {}) of {} x {}, ldy {ldy}, batch {batch}",
            w.rows, w.rows, w.cols
        )));
    }
    if x.0.length() < x.1 + (batch * w.cols * 2) as u64
        || y.0.length() < y.1 + (((batch - 1) * ldy + w.rows) * 2) as u64
    {
        return Err(GpuError::InvalidInput(
            "steel rows: buffer too small".into(),
        ));
    }
    let pipe = context.pipeline(
        crate::dequant_int4_steel::SOURCE,
        "dequant_int4_steel_32_32_32",
        &FunctionConstantValues::new(),
        &[],
    )?;
    let (k, n, m) = (
        u32_of(w.cols, "cols")?,
        u32_of(ldy, "ldy")?,
        u32_of(batch, "batch")?,
    );
    pass.encode_threadgroups_3d(
        &pipe,
        &[
            (view.buffer, 0, view.weights_offset),
            (view.buffer, 1, view.scales_offset),
            (view.buffer, 2, view.biases_offset),
            (x.0, 3, x.1),
            (y.0, 4, y.1 + (row0 * 2) as u64),
        ],
        &[(u32_bytes(&k), 5), (u32_bytes(&n), 6), (u32_bytes(&m), 7)],
        ((rows / BN) as u64, batch.div_ceil(BM) as u64, 1),
        (128, 1, 1),
    );
    Ok(())
}

/// An `MTLSharedEvent` the host signals. GPU work encoded with
/// [`begin_pass_after`] waits until the value reaches its threshold.
pub struct HostEvent {
    event: metal::SharedEvent,
}

impl HostEvent {
    pub fn new(context: &MetalContext) -> Self {
        Self {
            event: context.device().new_shared_event(),
        }
    }

    pub fn value(&self) -> u64 {
        self.event.signaled_value()
    }

    /// Releases every pass waiting on a value up to `value`. Values only
    /// grow; a smaller one would leave nothing released.
    pub fn signal(&self, value: u64) {
        self.event.set_signaled_value(value);
    }
}

/// [`MetalContext::begin_pass_labeled`] whose command buffer first waits for
/// `event` to reach `value`. The wait is encoded before the compute encoder
/// opens, as Metal requires. No dispatch profiling on these passes.
pub fn begin_pass_after(
    context: &MetalContext,
    label: &'static str,
    event: &HostEvent,
    value: u64,
) -> PassEncoder {
    let command_buffer = context.queue().new_command_buffer().to_owned();
    command_buffer.set_label(label);
    command_buffer.encode_wait_for_event(&event.event, value);
    let encoder = command_buffer.new_compute_command_encoder().to_owned();
    PassEncoder::new(command_buffer, std::cell::RefCell::new(encoder), None)
}

#[link(name = "QuartzCore", kind = "framework")]
extern "C" {
    fn CACurrentMediaTime() -> f64;
}

/// Host time in seconds, the timebase of `GPUStartTime` and `GPUEndTime`.
pub fn host_seconds() -> f64 {
    // SAFETY: CACurrentMediaTime takes no arguments and reads the host
    // clock.
    #[allow(unsafe_code)]
    unsafe {
        CACurrentMediaTime()
    }
}
