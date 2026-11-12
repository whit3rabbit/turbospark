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

use crate::bytes::{f32_bytes, u32_bytes};
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::dequant_int4_gemv::Int4ResidentMatrix;

/// See the module doc: `dequant_int4.metal` FIRST, because `gdn.metal`
/// calls into it.
static SOURCE: &str = concat!(
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

pub use crate::gdn_shape::GdnShape;

/// Every function constant either shader in [`SOURCE`] declares: INT4's
/// 20-26 and GDN's 90-98. Metal requires all constants a function reads to
/// be set before a pipeline can be built, and setting the ones it does not
/// read is harmless -- the same defensive shape the other kernel modules
/// use. `USE_FC = false` keeps a kernel on its runtime-argument path;
/// `gdn_in_proj_gemv_simd` alone overrides 90-94 via [`in_proj_pipeline`],
/// and `gdn_gated_norm` alone overrides 95 via [`gated_norm_sigmoid_pipeline`].
/// Every OTHER pipeline built through the plain [`pipeline`] helper --
/// including `gdn_gated_norm`'s own default build -- gets `95 = false`
/// here, which is what keeps this declaration from moving the two families
/// that already dispatch that kernel: `FC_GDN_GATE_SIGMOID` reads `false`
/// for them exactly as it always has. `gdn_delta_step_prefill` alone reads
/// 96/97, whose defaults here (RECORD false, WRITE_STATE true) are the
/// eager behavior every existing caller keeps; the lazy verify build sets
/// them through [`delta_prefill_lazy_pipeline`]. 98 (`FC_GDN_DK`) defaults
/// to 0, the runtime key dim; the delta kernels bake it through
/// [`delta_pipeline`].
fn gdn_function_constants() -> FunctionConstantValues {
    let values = FunctionConstantValues::new();
    let zero: u32 = 0;
    let use_fc = false;
    for index in [20, 21, 23, 24, 25, 90, 91, 92, 93, 98] {
        values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, index);
    }
    for index in [22, 26, 94, 95, 96, 99] {
        values.set_constant_value_at_index(
            (&use_fc as *const bool).cast(),
            MTLDataType::Bool,
            index,
        );
    }
    let write_state = true;
    values.set_constant_value_at_index((&write_state as *const bool).cast(), MTLDataType::Bool, 97);
    values
}

pub(crate) fn pipeline(
    context: &mut MetalContext,
    name: &'static str,
) -> Result<metal::ComputePipelineState, GpuError> {
    context.pipeline(SOURCE, name, &gdn_function_constants(), b"")
}

/// `gdn_in_proj_gemv_simd` with its row counts and N BAKED (constants
/// 90-94): +3-4% cold weight-read rate at these shapes, same finding and
/// same date as the plain GEMV's `specialized_constants` (see
/// `dequant_int4_gemv.rs`). The INT4 constants 20-26 stay unspecialized --
/// this kernel reads its own 90-93 through `gdn_in_*`, not `int4_fc_*`.
/// The KEY carries all four baked values, for the reason documented there:
/// a shared key silently reuses the first-compiled shape's pipeline.
pub(crate) fn in_proj_pipeline(
    context: &mut MetalContext,
    qkv_rows: u32,
    z_rows: u32,
    ab_rows: u32,
    n: u32,
) -> Result<metal::ComputePipelineState, GpuError> {
    let values = gdn_function_constants();
    let use_fc = true;
    values.set_constant_value_at_index((&qkv_rows as *const u32).cast(), MTLDataType::UInt, 90);
    values.set_constant_value_at_index((&z_rows as *const u32).cast(), MTLDataType::UInt, 91);
    values.set_constant_value_at_index((&ab_rows as *const u32).cast(), MTLDataType::UInt, 92);
    values.set_constant_value_at_index((&n as *const u32).cast(), MTLDataType::UInt, 93);
    values.set_constant_value_at_index((&use_fc as *const bool).cast(), MTLDataType::Bool, 94);
    let mut key = [0u8; 16];
    key[..4].copy_from_slice(&qkv_rows.to_le_bytes());
    key[4..8].copy_from_slice(&z_rows.to_le_bytes());
    key[8..12].copy_from_slice(&ab_rows.to_le_bytes());
    key[12..].copy_from_slice(&n.to_le_bytes());
    context.pipeline(SOURCE, "gdn_in_proj_gemv_simd", &values, &key)
}

/// `gdn_gated_norm` with `FC_GDN_GATE_SIGMOID` BAKED true (constant 95):
/// `qwen4_exp`'s `output_gate_type: sigmoid` variant of the gate `gdn_silu`
/// otherwise applies. The KEY carries the one byte that decides it, for
/// [`in_proj_pipeline`]'s reason -- a shared key would silently reuse
/// whichever gate compiled first, making the two conventions one function
/// (crate Gotcha 1, AGENTS.md Gotcha 50).
fn gated_norm_sigmoid_pipeline(
    context: &mut MetalContext,
) -> Result<metal::ComputePipelineState, GpuError> {
    let values = gdn_function_constants();
    let sigmoid = true;
    values.set_constant_value_at_index((&sigmoid as *const bool).cast(), MTLDataType::Bool, 95);
    context.pipeline(SOURCE, "gdn_gated_norm", &values, &[sigmoid as u8])
}

/// The one key dim the delta kernels bake (`FC_GDN_DK`): every shipped
/// qwen GDN checkpoint. Baking it unrolls the per-lane state loops, 2.78x
/// on the lazy verify kernel at qwen38 shapes (docs/KAPITSA_KERNELS.md).
/// Any other `key_head_dim` keeps the runtime-loop build.
const BAKED_KEY_DIM: u32 = 128;

/// `TURBOSPARK_GDN_DK128=0` keeps every delta kernel on the runtime-`Dk`
/// build, for A/B runs. Read once per process.
fn key_dim_bake_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("TURBOSPARK_GDN_DK128").as_deref() != Ok("0"))
}

/// The `FC_GDN_DK` value a dispatch at `key_head_dim` bakes:
/// [`BAKED_KEY_DIM`] or 0 (runtime `keyDim`).
fn baked_key_dim(key_head_dim: u32) -> u32 {
    #[cfg(test)]
    if tests::FORCE_RUNTIME_KEY_DIM.with(|f| f.get()) {
        return 0;
    }
    if key_head_dim == BAKED_KEY_DIM && key_dim_bake_enabled() {
        // Proof for the lab that the baked build ran (the knob row lists this
        // line as expected unless `TURBOSPARK_GDN_DK128=0`).
        foundation::engaged!("gdn_dk128");
        BAKED_KEY_DIM
    } else {
        0
    }
}

/// A delta kernel (`gdn_delta_step_decode`, `gdn_delta_step_prefill`,
/// `gdn_state_fold`) with (RECORD, WRITE_STATE) and `FC_GDN_DK` baked. The
/// KEY carries all three values: the lazy build (true, false) must never
/// reach an eager caller, and a baked `Dk` must never reach a dispatch at
/// another `key_head_dim` (crate Gotcha 1). Kernels that do not read 96/97
/// get the eager defaults, which they ignore.
pub(crate) fn delta_pipeline(
    context: &mut MetalContext,
    name: &'static str,
    record: bool,
    write_state: bool,
    key_dim: u32,
) -> Result<metal::ComputePipelineState, GpuError> {
    delta_pipeline_tree(context, name, record, write_state, key_dim, false)
}

/// [`delta_pipeline`] with `FC_GDN_TREE` (99) as well, the key's 7th byte.
/// Every delta pipeline sets it, false unless a tree verify asked.
fn delta_pipeline_tree(
    context: &mut MetalContext,
    name: &'static str,
    record: bool,
    write_state: bool,
    key_dim: u32,
    tree: bool,
) -> Result<metal::ComputePipelineState, GpuError> {
    let values = gdn_function_constants();
    values.set_constant_value_at_index((&record as *const bool).cast(), MTLDataType::Bool, 96);
    values.set_constant_value_at_index((&write_state as *const bool).cast(), MTLDataType::Bool, 97);
    values.set_constant_value_at_index((&key_dim as *const u32).cast(), MTLDataType::UInt, 98);
    values.set_constant_value_at_index((&tree as *const bool).cast(), MTLDataType::Bool, 99);
    let mut key = [0u8; 7];
    key[0] = record as u8;
    key[1] = write_state as u8;
    key[2..6].copy_from_slice(&key_dim.to_le_bytes());
    key[6] = tree as u8;
    context.pipeline(SOURCE, name, &values, &key)
}

/// `gdn_delta_step_decode` as [`encode_gdn_delta_decode`] builds it for a
/// checkpoint with `key_dim`, for kernel warmup.
pub(crate) fn delta_decode_pipeline(
    context: &mut MetalContext,
    key_dim: u32,
) -> Result<metal::ComputePipelineState, GpuError> {
    delta_pipeline(
        context,
        "gdn_delta_step_decode",
        false,
        true,
        baked_key_dim(key_dim),
    )
}

/// `gdn_delta_step_prefill` with (RECORD, WRITE_STATE) baked (true, false):
/// the lazy verify's build. It leaves the live state at the block start and
/// writes each row's `delta` to `delta_tape`, for [`encode_gdn_state_fold`]
/// to fold in at the round end.
fn delta_prefill_lazy_pipeline(
    context: &mut MetalContext,
    key_dim: u32,
) -> Result<metal::ComputePipelineState, GpuError> {
    delta_pipeline(context, "gdn_delta_step_prefill", true, false, key_dim)
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
    // AGENTS.md/CLAUDE.md S8: `GpuError::PipelineCreate` for a shape
    // violation is this crate's existing spelling (`vision.rs`,
    // `gdn_shape.rs::validate`) -- a bad checkpoint should refuse to
    // dispatch, not abort the process.
    let n = qkv.cols;
    if n % 64 != 0 {
        return Err(GpuError::PipelineCreate(format!(
            "gdn_in_proj hidden size {n} must be a multiple of 64"
        )));
    }
    for m in [z, a, b] {
        if m.cols != n {
            return Err(GpuError::PipelineCreate(format!(
                "gdn_in_proj: all four input projections must share one x, got {} against {n}",
                m.cols
            )));
        }
    }
    if a.rows != b.rows {
        return Err(GpuError::PipelineCreate(format!(
            "gdn_in_proj: a and b both project one row per value head, got {} and {}",
            a.rows, b.rows
        )));
    }
    // The row body reads packed weights through a `ushort*`: the repacker
    // guarantees 2-byte sub-tensor alignment but not 4-byte.
    for m in [qkv, z, a, b] {
        if m.weights_offset % 2 != 0 {
            return Err(GpuError::PipelineCreate(format!(
                "gdn_in_proj_gemv_simd needs 2-aligned weight offsets, got {}",
                m.weights_offset
            )));
        }
    }

    let (qkv_rows, z_rows, ab_rows, n32) =
        (qkv.rows as u32, z.rows as u32, a.rows as u32, n as u32);
    let p = in_proj_pipeline(context, qkv_rows, z_rows, ab_rows, n32)?;
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
///
/// `dilation` generalizes this for `qwen4_exp`'s PLE conv
/// (`kernel_size=4`, `dilation=3`); every existing family passes `1`, which
/// reproduces the pre-dilation kernel exactly (see the shader's own doc).
/// `tail` must hold `(taps - 1) * dilation` rows, not `taps - 1`.
#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_conv_decode(
    context: &mut MetalContext,
    pass: &PassEncoder,
    shape: GdnShape,
    tail: (&metal::Buffer, u64),
    qkv: (&metal::Buffer, u64),
    conv_weight: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    dilation: u32,
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
        &[
            (u32_bytes(&channels), 4),
            (u32_bytes(&taps), 5),
            (u32_bytes(&dilation), 6),
        ],
        (channels as u64, 1, 1),
        (CONV_THREADS, 1, 1),
    );
    Ok(())
}

/// Prefill: the same conv over `rows` chunk rows at once. `tail` is
/// READ-ONLY here -- [`encode_gdn_conv_tail_update`] refreshes it after.
/// See [`encode_gdn_conv_decode`] for `dilation`.
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
    dilation: u32,
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
            (u32_bytes(&dilation), 7),
        ],
        (channels as u64, rows.max(1) as u64, 1),
        (CONV_THREADS, 1, 1),
    );
    Ok(())
}

/// The tree verify's conv source table: for row `t` and tap `j` (oldest
/// first), the row `K-1-j` steps up `t`'s parent chain, or `depth(t) - s`
/// (negative, a tail row) when the path from the root is shorter than the
/// step. A chain of parents gives `t + j - (K-1)`, the chain kernel's own
/// index, so the arithmetic is unchanged.
pub(crate) fn conv_tree_sources(parent: &[i32], taps: usize) -> Vec<i32> {
    let depth = |mut t: i32| {
        let mut d = 0;
        while parent[t as usize] >= 0 {
            t = parent[t as usize];
            d += 1;
        }
        d
    };
    let mut out = Vec::with_capacity(parent.len() * (taps - 1));
    for t in 0..parent.len() as i32 {
        let d = depth(t);
        for j in 0..taps as i32 - 1 {
            let steps = taps as i32 - 1 - j;
            if steps <= d {
                let mut at = t;
                for _ in 0..steps {
                    at = parent[at as usize];
                }
                out.push(at);
            } else {
                out.push(d - steps);
            }
        }
    }
    out
}

/// [`encode_gdn_conv_prefill`] over a tree of rows (`parent[t] < t`, -1 at
/// a root): each row's taps read its own root path, then the tail. Undilated
/// only. Bit-identical per node to the chain kernel over that node's path.
#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_conv_prefill_tree(
    context: &mut MetalContext,
    pass: &PassEncoder,
    shape: GdnShape,
    tail: (&metal::Buffer, u64),
    qkv_rows: (&metal::Buffer, u64),
    conv_weight: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    parent: &[i32],
) -> Result<(), GpuError> {
    shape.validate()?;
    let rows = parent.len() as u32;
    if rows == 0 || rows > GDN_TREE_MAX_ROWS || (0..parent.len()).any(|t| parent[t] >= t as i32) {
        return Err(GpuError::PipelineCreate(format!(
            "gdn tree conv needs 1..={GDN_TREE_MAX_ROWS} rows with parents before children; \
             got {parent:?}"
        )));
    }
    let (channels, taps) = (shape.qkv_dim(), shape.conv_kernel_size);
    let sources = conv_tree_sources(parent, taps as usize);
    let source_bytes: Vec<u8> = sources.iter().flat_map(|s| s.to_le_bytes()).collect();
    let values = gdn_function_constants();
    let tree = true;
    values.set_constant_value_at_index((&tree as *const bool).cast(), MTLDataType::Bool, 99);
    let p = context.pipeline(SOURCE, "gdn_conv_mix_prefill", &values, b"tree")?;
    let dilation = 1u32;
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
            (u32_bytes(&dilation), 7),
            (&source_bytes[..], 8),
        ],
        (channels as u64, rows as u64, 1),
        (CONV_THREADS, 1, 1),
    );
    Ok(())
}

/// After a prefill chunk: `tail := last (K - 1) * dilation raw rows of
/// [tail | chunk]`. See [`encode_gdn_conv_decode`] for `dilation`.
pub fn encode_gdn_conv_tail_update(
    context: &mut MetalContext,
    pass: &PassEncoder,
    shape: GdnShape,
    tail: (&metal::Buffer, u64),
    qkv_rows: (&metal::Buffer, u64),
    rows: u32,
    dilation: u32,
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
            (u32_bytes(&dilation), 5),
        ],
        (channels as u64, (taps - 1) as u64 * dilation as u64, 1),
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
    encode_gdn_qk_norm_with_rms_epsilon(context, pass, shape, conv_out, rows, 1e-6)
}

/// Qwen4's published reference normalizes with `x / sqrt(sum(x*x) + eps)`.
/// This kernel expresses the same operation as RMS in mean space, so callers
/// that need that convention pass `eps / key_head_dim`. The older GDN families
/// keep using [`encode_gdn_qk_norm`] and its historical mean-space epsilon.
pub fn encode_gdn_qk_norm_with_rms_epsilon(
    context: &mut MetalContext,
    pass: &PassEncoder,
    shape: GdnShape,
    conv_out: (&metal::Buffer, u64),
    rows: u32,
    rms_epsilon: f32,
) -> Result<(), GpuError> {
    shape.validate()?;
    if !rms_epsilon.is_finite() || rms_epsilon < 0.0 {
        return Err(GpuError::PipelineCreate(format!(
            "gdn_qk_norm RMS epsilon must be finite and non-negative, got {rms_epsilon}"
        )));
    }
    let p = pipeline(context, "gdn_qk_norm")?;
    let (k_heads, key_dim, row_stride) = (shape.num_k_heads, shape.key_head_dim, shape.qkv_dim());
    pass.encode_threadgroups_3d(
        &p,
        &[(conv_out.0, 0, conv_out.1)],
        &[
            (u32_bytes(&k_heads), 1),
            (u32_bytes(&key_dim), 2),
            (u32_bytes(&row_stride), 3),
            (f32_bytes(&rms_epsilon), 4),
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
    let p = delta_pipeline(
        context,
        "gdn_delta_step_decode",
        false,
        true,
        baked_key_dim(shape.key_head_dim),
    )?;
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
    let p = delta_pipeline(
        context,
        "gdn_delta_step_prefill",
        false,
        true,
        baked_key_dim(shape.key_head_dim),
    )?;
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

/// Asserts a tape view of `rows` rows of `row_elems` elements of
/// `elem_bytes` each fits in `buffer` from `offset`. A slot stride that
/// drifts from the recording pass's batch is silent recurrent-state
/// corruption, so the lazy tape's views are checked the way
/// `encode_copy_rows_fp16_strided` checks its own.
fn assert_tape_view(
    what: &str,
    buffer: &metal::Buffer,
    offset: u64,
    rows: u32,
    row_elems: u32,
    elem_bytes: u64,
) {
    let end = u64::from(rows)
        .checked_mul(u64::from(row_elems))
        .and_then(|n| n.checked_mul(elem_bytes))
        .and_then(|n| n.checked_add(offset))
        .expect("tape view byte range overflows u64");
    assert!(
        end <= buffer.length(),
        "{what} tape view exceeds its buffer"
    );
}

/// The prefill recurrence with (RECORD, WRITE_STATE) baked (true, false):
/// identical y, no state write-back, per-row `delta` recorded to
/// `delta_tape`. See [`delta_prefill_lazy_pipeline`] and
/// [`encode_gdn_state_fold`].
#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_delta_prefill_lazy(
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
    delta_tape: (&metal::Buffer, u64),
    rows: u32,
) -> Result<(), GpuError> {
    shape.validate()?;
    assert_tape_view(
        "delta",
        delta_tape.0,
        delta_tape.1,
        rows,
        shape.value_dim(),
        4,
    );
    let p = delta_prefill_lazy_pipeline(context, baked_key_dim(shape.key_head_dim))?;
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
            (delta_tape.0, 13, delta_tape.1),
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

/// Rows one tree verify takes: the anchor and seven nodes, every node's
/// state in registers (`FC_GDN_TREE` in `gdn.metal`).
pub const GDN_TREE_MAX_ROWS: u32 = 8;

/// Whether [`encode_gdn_delta_prefill_lazy_tree`] serves this shape: the key
/// head dim must be the baked 128 (`TURBOSPARK_GDN_DK128` not `0`), because
/// every node's state lives in registers.
pub fn gdn_tree_supported(shape: GdnShape) -> bool {
    baked_key_dim(shape.key_head_dim) == BAKED_KEY_DIM
}

/// [`encode_gdn_delta_prefill_lazy`] over a TREE of rows: row `t` starts
/// from the state row `parent[t]` left, or from the block-start state when
/// `parent[t] < 0`. Parents come before children (`parent[t] < t`). Each
/// node's `y` and recorded `delta` are what the chain kernel gives over that
/// node's root path, so after the accepted path is compacted to rows
/// `0..keep` the round-end fold is the chain's. Refused unless the key head
/// dim is baked at 128 and `rows <= 8`: every node's state lives in
/// registers.
#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_delta_prefill_lazy_tree(
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
    delta_tape: (&metal::Buffer, u64),
    parent: &[i32],
) -> Result<(), GpuError> {
    shape.validate()?;
    let rows = parent.len() as u32;
    let key_dim = baked_key_dim(shape.key_head_dim);
    if key_dim != BAKED_KEY_DIM || rows == 0 || rows > GDN_TREE_MAX_ROWS {
        return Err(GpuError::PipelineCreate(format!(
            "gdn tree verify needs the key head dim baked at {BAKED_KEY_DIM} and 1..={} rows; \
             got key head dim {} (baked {key_dim}) and {rows} rows",
            GDN_TREE_MAX_ROWS, shape.key_head_dim
        )));
    }
    if let Some(t) = (0..parent.len()).find(|&t| parent[t] >= t as i32) {
        return Err(GpuError::PipelineCreate(format!(
            "gdn tree verify: row {t}'s parent {} does not precede it",
            parent[t]
        )));
    }
    assert_tape_view(
        "delta",
        delta_tape.0,
        delta_tape.1,
        rows,
        shape.value_dim(),
        4,
    );
    let p = delta_pipeline_tree(
        context,
        "gdn_delta_step_prefill",
        true,
        false,
        key_dim,
        true,
    )?;
    let dims = shape.head_dim_bytes();
    let row_stride = shape.qkv_dim();
    let parent_bytes: Vec<u8> = parent.iter().flat_map(|p| p.to_le_bytes()).collect();
    let mut bytes: Vec<(&[u8], u64)> = dims.iter().map(|(v, i)| (&v[..], *i)).collect();
    bytes.push((u32_bytes(&rows), 11));
    bytes.push((u32_bytes(&row_stride), 12));
    bytes.push((&parent_bytes[..], 14));
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
            (delta_tape.0, 13, delta_tape.1),
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

/// Applies `rows` recorded rank-1 updates to the live state in place: the
/// round-end commit of the lazy verify tape, bit-identical to the eager
/// kernel's register state after the same rows. Same grid as the delta
/// kernels; the strides are per-row ELEMENT counts of each tape.
#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_state_fold(
    context: &mut MetalContext,
    pass: &PassEncoder,
    shape: GdnShape,
    k_tape: (&metal::Buffer, u64),
    delta_tape: (&metal::Buffer, u64),
    a_tape: (&metal::Buffer, u64),
    a_log: (&metal::Buffer, u64),
    dt_bias: (&metal::Buffer, u64),
    state: &metal::Buffer,
    rows: u32,
) -> Result<(), GpuError> {
    shape.validate()?;
    let p = delta_pipeline(
        context,
        "gdn_state_fold",
        false,
        true,
        baked_key_dim(shape.key_head_dim),
    )?;
    let k_stride = shape.num_k_heads * shape.key_head_dim;
    let a_stride = shape.num_v_heads;
    let delta_stride = shape.num_v_heads * shape.value_head_dim;
    assert_tape_view("fold k", k_tape.0, k_tape.1, rows, k_stride, 2);
    assert_tape_view(
        "fold delta",
        delta_tape.0,
        delta_tape.1,
        rows,
        delta_stride,
        4,
    );
    assert_tape_view("fold a", a_tape.0, a_tape.1, rows, a_stride, 2);
    let (hkb, hvb, dkb, dvb, rowsb, ksb, asb, dsb) = (
        shape.num_k_heads.to_le_bytes(),
        shape.num_v_heads.to_le_bytes(),
        shape.key_head_dim.to_le_bytes(),
        shape.value_head_dim.to_le_bytes(),
        rows.to_le_bytes(),
        k_stride.to_le_bytes(),
        a_stride.to_le_bytes(),
        delta_stride.to_le_bytes(),
    );
    let bytes: [(&[u8], u64); 8] = [
        (&hkb[..], 6),
        (&hvb[..], 7),
        (&dkb[..], 8),
        (&dvb[..], 9),
        (&rowsb[..], 10),
        (&ksb[..], 11),
        (&asb[..], 12),
        (&dsb[..], 13),
    ];
    pass.encode_threadgroups_3d(
        &p,
        &[
            (k_tape.0, 0, k_tape.1),
            (delta_tape.0, 1, delta_tape.1),
            (a_tape.0, 2, a_tape.1),
            (a_log.0, 3, a_log.1),
            (dt_bias.0, 4, dt_bias.1),
            (state, 5, 0),
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

/// `out = rmsnorm(y; weight) * sigmoid(z)`, per value head, over `rows`
/// rows -- `qwen4_exp`'s `output_gate_type: sigmoid` variant of
/// [`encode_gdn_gated_norm`]. **Do not reach for this by family**: the
/// gate is a CHECKPOINT property, and `qwen3_5`/`qwen3_6` take the plain
/// (silu) function unchanged, through their existing call sites, which this
/// function does not touch. The contract is
/// `turbospark_compute::gated_norm_sigmoid`.
#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_gated_norm_sigmoid(
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
    let p = gated_norm_sigmoid_pipeline(context)?;
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

#[cfg(test)]
#[path = "gdn_tests.rs"]
mod tests;
