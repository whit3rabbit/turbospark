//! Host-side dispatch for the two kernels in `shaders/dequant_1bit.metal`
//! (ROADMAP's 1-bit entry, step 2): MLX `affine` at `bits: 1`, the
//! `prism-ml/Bonsai-27B-mlx-1bit` layout.
//!
//! PORT-LOCAL rather than vendored, like the GGUF set: the Swift engine
//! reads no 1-bit checkpoint. The contract is
//! `turbospark_compute::quant_1bit`'s two GEMVs, which is what
//! `crates/gpu/tests/dequant_1bit_gemv_parity.rs` holds these to.
//!
//! Three things differ from the INT4/INT8 affine siblings, and each one is a
//! silently wrong answer rather than a failure if carried over. The
//! companions are FP16, so they bind as `device const half*` and NOT as the
//! siblings' `device const bfloat*` (same width, so nothing checks it). The
//! group size is the checkpoint's 128 rather than the siblings' compile-time
//! 64, and it travels as a runtime uniform for the reason the shader header
//! gives. And a byte holds 8 elements, so the LSB-first bit order inside it
//! is load-bearing and was measured against `mx.dequantize` rather than
//! assumed.
//!
//! **The symmetric fast path has no resident form yet, and that is a
//! decision.** The affine kernel decodes any 1-bit install; the `+/-1` one
//! is selected per tensor at repack time, and WHICH offsets it is handed
//! depends on whether step 3 writes a bias plane for a symmetric tensor at
//! all. Binding it now would be guessing at a layout that does not exist.

use half::f16;
use metal::FunctionConstantValues;

use crate::bytes::{half_slice_to_le_bytes, read_half_buffer, u16_slice_to_le_bytes, u32_bytes};
use crate::context::{
    dispatch_one_threadgroup_per_row, dispatch_one_threadgroup_per_row_offsets, GpuError,
    MetalContext, PassEncoder,
};

const SOURCE: &str = include_str!("shaders/dequant_1bit.metal");
const THREADS_PER_GROUP: u64 = 256; // 8 rows/threadgroup * 32 lanes/SIMD group.
const ROWS_PER_THREADGROUP: u64 = 8;

// There is deliberately no `BONSAI_GROUP_SIZE` here.
// `turbospark_compute::quant_1bit` already owns that constant, every entry
// point below takes the group size as an argument, and a second copy of a
// per-checkpoint number in a crate that cannot see the first is the drift
// hazard AGENTS.md Gotchas 37 and 38 are both instances of.

/// Neither kernel declares a function constant, so nothing is specialized
/// into either. Named rather than inlined so the reason is on the record:
/// the one axis that would want specializing is the group size, and it is a
/// uniform on purpose (a specialization axis that misses `pipeline`'s
/// constants key silently reuses the wrong pipeline, crate Gotcha 1).
fn no_function_constants() -> FunctionConstantValues {
    FunctionConstantValues::new()
}

/// Bytes a 1-bit row of `n` elements occupies.
pub fn int1_row_bytes(n: usize) -> usize {
    assert_eq!(n % 8, 0, "N ({n}) is not a whole number of bytes");
    n / 8
}

fn check_shape(n: usize, group_size: usize) {
    assert!(group_size > 0, "group size must be positive");
    assert_eq!(
        group_size % 8,
        0,
        "group size {group_size} is not a whole number of bytes"
    );
    assert_eq!(
        n % group_size,
        0,
        "N ({n}) is not a multiple of the group size {group_size}"
    );
}

/// One row of 1-bit-affine packed weights, laid out exactly as
/// `turbospark_compute::quant_1bit::Int1AffineRow`: `n / 8` bytes,
/// `n / group_size` FP16 scale bit patterns, and the same shape of biases.
pub struct Int1AffineRowGpu<'a> {
    pub packed: &'a [u8],
    pub scales: &'a [u16],
    pub biases: &'a [u16],
}

/// One row for the `+/-1` fast path. It carries NO bias plane, which is the
/// point: the caller has to have established `bias == -scale/2` (see
/// `turbospark_compute::quant_1bit::is_symmetric`) before it can build one
/// of these, and there is no field through which an unchecked bias could
/// reach the kernel.
pub struct Int1SymmetricRowGpu<'a> {
    pub packed: &'a [u8],
    pub scales: &'a [u16],
}

/// `y[m] = sum_n W[m, n] * x[n]` over 1-bit affine rows, the GENERAL form:
/// every group decodes through its own `(scale, bias)` pair.
pub fn dequant_int1_gemv(
    context: &mut MetalContext,
    weight_rows: &[Int1AffineRowGpu<'_>],
    x: &[f16],
    n: usize,
    group_size: usize,
) -> Result<Vec<f16>, GpuError> {
    assert!(!weight_rows.is_empty());
    assert_eq!(x.len(), n);
    check_shape(n, group_size);
    let row_bytes = int1_row_bytes(n);
    let n_groups = n / group_size;

    let m = weight_rows.len();
    let mut w_bytes = Vec::with_capacity(m * row_bytes);
    let mut scale_bits = Vec::with_capacity(m * n_groups);
    let mut bias_bits = Vec::with_capacity(m * n_groups);
    for row in weight_rows {
        assert_eq!(
            row.packed.len(),
            row_bytes,
            "every row is {row_bytes} bytes"
        );
        assert_eq!(row.scales.len(), n_groups);
        assert_eq!(row.biases.len(), n_groups);
        w_bytes.extend_from_slice(row.packed);
        scale_bits.extend_from_slice(row.scales);
        bias_bits.extend_from_slice(row.biases);
    }

    let w_buffer = context.new_buffer_with_data(&w_bytes);
    let scales_buffer = context.new_buffer_with_data(&u16_slice_to_le_bytes(&scale_bits));
    let biases_buffer = context.new_buffer_with_data(&u16_slice_to_le_bytes(&bias_bits));
    let x_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(x));
    let y_buffer = context.new_output_buffer((m * std::mem::size_of::<u16>()) as u64);

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let g_u32 = group_size as u32;
    let pipeline = context.pipeline(
        SOURCE,
        "dequant_int1_gemv_simd",
        &no_function_constants(),
        b"",
    )?;
    let threadgroups = m.div_ceil(ROWS_PER_THREADGROUP as usize) as u64;
    dispatch_one_threadgroup_per_row(
        context,
        &pipeline,
        &[
            (&w_buffer, 0),
            (&scales_buffer, 1),
            (&biases_buffer, 2),
            (&x_buffer, 3),
            (&y_buffer, 4),
        ],
        &[
            (u32_bytes(&m_u32), 5),
            (u32_bytes(&n_u32), 6),
            (u32_bytes(&g_u32), 7),
        ],
        threadgroups,
        THREADS_PER_GROUP,
    );

    Ok(read_half_buffer(&y_buffer, m))
}

/// The `+/-1` form of [`dequant_int1_gemv`], for rows every group of which
/// satisfies `bias == -scale/2`.
///
/// Not bit-identical to the general form and not meant to be: factoring the
/// scale out of the group reassociates the sum, which is why this is a
/// separate kernel rather than a fast path inside the other. Mirrors
/// `turbospark_compute::dequant_int1_gemv_symmetric`.
pub fn dequant_int1_gemv_symmetric(
    context: &mut MetalContext,
    weight_rows: &[Int1SymmetricRowGpu<'_>],
    x: &[f16],
    n: usize,
    group_size: usize,
) -> Result<Vec<f16>, GpuError> {
    assert!(!weight_rows.is_empty());
    assert_eq!(x.len(), n);
    check_shape(n, group_size);
    let row_bytes = int1_row_bytes(n);
    let n_groups = n / group_size;

    let m = weight_rows.len();
    let mut w_bytes = Vec::with_capacity(m * row_bytes);
    let mut scale_bits = Vec::with_capacity(m * n_groups);
    for row in weight_rows {
        assert_eq!(
            row.packed.len(),
            row_bytes,
            "every row is {row_bytes} bytes"
        );
        assert_eq!(row.scales.len(), n_groups);
        w_bytes.extend_from_slice(row.packed);
        scale_bits.extend_from_slice(row.scales);
    }

    let w_buffer = context.new_buffer_with_data(&w_bytes);
    let scales_buffer = context.new_buffer_with_data(&u16_slice_to_le_bytes(&scale_bits));
    let x_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(x));
    let y_buffer = context.new_output_buffer((m * std::mem::size_of::<u16>()) as u64);

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let g_u32 = group_size as u32;
    let pipeline = context.pipeline(
        SOURCE,
        "dequant_int1_gemv_symmetric_simd",
        &no_function_constants(),
        b"",
    )?;
    let threadgroups = m.div_ceil(ROWS_PER_THREADGROUP as usize) as u64;
    dispatch_one_threadgroup_per_row(
        context,
        &pipeline,
        &[
            (&w_buffer, 0),
            (&scales_buffer, 1),
            (&x_buffer, 2),
            (&y_buffer, 3),
        ],
        &[
            (u32_bytes(&m_u32), 4),
            (u32_bytes(&n_u32), 5),
            (u32_bytes(&g_u32), 6),
        ],
        threadgroups,
        THREADS_PER_GROUP,
    );

    Ok(read_half_buffer(&y_buffer, m))
}

/// Encoder-level `embed_lookup_int1`: dequantizes one row of a 1-bit affine
/// embedding table (bound in place, normally offsets into the resident
/// buffer) into `out` (`d` halfs), scaled by `out_scale`.
///
/// The 1-bit sibling of [`crate::encode_embed_lookup_int4`]. It exists
/// because the real checkpoint quantizes `embed_tokens` at one bit like
/// everything else, which was read off the safetensors header rather than
/// assumed -- so this type's footing is a GEMV plus a lookup and no
/// routed-expert pair (AGENTS.md Gotcha 29's per-type rule).
///
/// One argument wider than the INT4 sibling, and it is the group size: see
/// the module header for why that travels rather than being a constant.
#[allow(clippy::too_many_arguments)]
pub fn encode_embed_lookup_int1(
    context: &mut MetalContext,
    pass: &PassEncoder,
    table: (&metal::Buffer, u64),
    scales: (&metal::Buffer, u64),
    biases: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    token_id: u32,
    d: u32,
    group_size: u32,
    out_scale: f32,
) -> Result<(), GpuError> {
    check_shape(d as usize, group_size as usize);
    let pipeline = context.pipeline(SOURCE, "embed_lookup_int1", &no_function_constants(), b"")?;
    pass.encode_threads_3d(
        &pipeline,
        &[
            (table.0, 0, table.1),
            (scales.0, 1, scales.1),
            (biases.0, 2, biases.1),
            (out.0, 3, out.1),
        ],
        &[
            (u32_bytes(&token_id), 4),
            (u32_bytes(&d), 5),
            (u32_bytes(&group_size), 6),
            (crate::bytes::f32_bytes(&out_scale), 7),
        ],
        (d as u64, 1, 1),
        (64, 1, 1),
    );
    Ok(())
}

/// A whole 1-bit affine weight matrix addressed IN PLACE inside one shared
/// `MTLBuffer` (normally `ResidentGpuWeights::buffer`): `rows * cols / 8`
/// weight bytes at `weights_offset`, `rows * cols / group_size` FP16 scale
/// bit patterns at `scales_offset`, same-shaped biases at `biases_offset`.
///
/// `group_size` is a field rather than a constant because it is the
/// checkpoint's property, not the container's.
pub struct Int1ResidentMatrix<'a> {
    pub buffer: &'a metal::Buffer,
    pub weights_offset: u64,
    pub scales_offset: u64,
    pub biases_offset: u64,
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
}

/// Encoder-level offset-bound 1-bit GEMV: same kernel and math as
/// [`dequant_int1_gemv`], weights bound as offsets into `w.buffer`.
pub fn encode_dequant_int1_gemv_resident(
    context: &mut MetalContext,
    pass: &PassEncoder,
    w: &Int1ResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
) -> Result<(), GpuError> {
    assert!(w.rows > 0);
    check_shape(w.cols, w.group_size);
    let m_u32 = w.rows as u32;
    let n_u32 = w.cols as u32;
    let g_u32 = w.group_size as u32;
    let pipeline = context.pipeline(
        SOURCE,
        "dequant_int1_gemv_simd",
        &no_function_constants(),
        b"",
    )?;
    let threadgroups = w.rows.div_ceil(ROWS_PER_THREADGROUP as usize) as u64;
    pass.encode_threadgroups(
        &pipeline,
        &[
            (w.buffer, 0, w.weights_offset),
            (w.buffer, 1, w.scales_offset),
            (w.buffer, 2, w.biases_offset),
            (x.0, 3, x.1),
            (y.0, 4, y.1),
        ],
        &[
            (u32_bytes(&m_u32), 5),
            (u32_bytes(&n_u32), 6),
            (u32_bytes(&g_u32), 7),
        ],
        threadgroups,
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// One-shot [`encode_dequant_int1_gemv_resident`] for the parity tests.
pub fn dequant_int1_gemv_resident(
    context: &mut MetalContext,
    w: &Int1ResidentMatrix<'_>,
    x: &[f16],
) -> Result<Vec<f16>, GpuError> {
    assert_eq!(x.len(), w.cols);
    check_shape(w.cols, w.group_size);
    let x_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(x));
    let y_buffer = context.new_output_buffer((w.rows * std::mem::size_of::<u16>()) as u64);

    let m_u32 = w.rows as u32;
    let n_u32 = w.cols as u32;
    let g_u32 = w.group_size as u32;
    let pipeline = context.pipeline(
        SOURCE,
        "dequant_int1_gemv_simd",
        &no_function_constants(),
        b"",
    )?;
    let threadgroups = w.rows.div_ceil(ROWS_PER_THREADGROUP as usize) as u64;
    dispatch_one_threadgroup_per_row_offsets(
        context,
        &pipeline,
        &[
            (w.buffer, 0, w.weights_offset),
            (w.buffer, 1, w.scales_offset),
            (w.buffer, 2, w.biases_offset),
            (&x_buffer, 3, 0),
            (&y_buffer, 4, 0),
        ],
        &[
            (u32_bytes(&m_u32), 5),
            (u32_bytes(&n_u32), 6),
            (u32_bytes(&g_u32), 7),
        ],
        threadgroups,
        THREADS_PER_GROUP,
    );

    Ok(read_half_buffer(&y_buffer, w.rows))
}
