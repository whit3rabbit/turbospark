//! Host-side dispatch for `shaders/dequant_int4_fused.metal`: the bit-exact
//! fusions of the dense-FFN and shared-expert dispatch chains
//! (docs/KAPITSA_KERNELS.md items 4 and 5). The shader header carries the
//! bit-exactness contract; this module's job is the pipeline cache keys.
//!
//! Every constant each kernel reads is ALWAYS set and ALWAYS in the key, the
//! `dequant_int4_gemv.rs` convention: (M, N) are baked per shape like the
//! plain GEMV's since 2026-08-16, and the selector constants (30 = gelu vs
//! silu, 31/32 = the down kernel's epilogue) decide which compiled body a
//! lookup must match -- a byte that missed the key would silently reuse
//! whichever compiled first (crate Gotcha 1).

use metal::{FunctionConstantValues, MTLDataType};

use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::dequant_int4_gemv::Int4ResidentMatrix;

static SOURCE: &str = include_str!("shaders/dequant_int4_fused.metal");

const THREADS_PER_GROUP: u64 = 256; // 8 rows/threadgroup * 32 lanes/SIMD group.
const ROWS_PER_THREADGROUP: u64 = 8;

fn fused_constants(m: u32, n: u32, selector: [bool; 3]) -> (FunctionConstantValues, Vec<u8>) {
    let values = FunctionConstantValues::new();
    let use_fc = true;
    values.set_constant_value_at_index((&m as *const u32).cast(), MTLDataType::UInt, 20);
    values.set_constant_value_at_index((&n as *const u32).cast(), MTLDataType::UInt, 21);
    values.set_constant_value_at_index((&use_fc as *const bool).cast(), MTLDataType::Bool, 22);
    for (i, &v) in selector.iter().enumerate() {
        values.set_constant_value_at_index(
            (&v as *const bool).cast(),
            MTLDataType::Bool,
            30 + i as u64,
        );
    }
    let mut key = Vec::with_capacity(11);
    key.extend_from_slice(&m.to_le_bytes());
    key.extend_from_slice(&n.to_le_bytes());
    for v in selector {
        key.push(v as u8);
    }
    (values, key)
}

/// Fused gate+up+activation GEMV: one dispatch replacing two GEMV calls and
/// the elementwise activation, at bit-identical outputs (the shader header
/// carries the rounding contract). `gate` and `up` must share the shape
/// (both are `[inter, hidden]`).
#[allow(clippy::too_many_arguments)]
pub fn encode_gate_up_act_gemv(
    context: &mut MetalContext,
    pass: &PassEncoder,
    gate: &Int4ResidentMatrix<'_>,
    up: &Int4ResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    use_gelu: bool,
) -> Result<(), GpuError> {
    let (m, n) = (gate.rows, gate.cols);
    if up.rows != m || up.cols != n {
        return Err(GpuError::PipelineCreate(format!(
            "gate_up_act: gate is {m}x{n} but up is {}x{}",
            up.rows, up.cols
        )));
    }
    if n % 64 != 0 {
        return Err(GpuError::PipelineCreate(format!(
            "gate_up_act: hidden size {n} must be a multiple of 64"
        )));
    }
    for mat in [gate, up] {
        if mat.weights_offset % 2 != 0 {
            return Err(GpuError::PipelineCreate(format!(
                "gate_up_act needs 2-aligned weight offsets, got {}",
                mat.weights_offset
            )));
        }
    }
    if x.1 % 8 != 0 {
        return Err(GpuError::PipelineCreate(format!(
            "gate_up_act: x offset {} must be 8-byte aligned (half4 loads)",
            x.1
        )));
    }
    let (m32, n32) = (m as u32, n as u32);
    let (constants, key) = fused_constants(m32, n32, [use_gelu, false, false]);
    let p = context.pipeline(
        SOURCE,
        "dequant_int4_gate_up_act_gemv_simd",
        &constants,
        &key,
    )?;
    let threadgroups = m.div_ceil(ROWS_PER_THREADGROUP as usize) as u64;
    pass.encode_threadgroups(
        &p,
        &[
            (gate.buffer, 0, gate.weights_offset),
            (gate.buffer, 1, gate.scales_offset),
            (gate.buffer, 2, gate.biases_offset),
            (up.buffer, 3, up.weights_offset),
            (up.buffer, 4, up.scales_offset),
            (up.buffer, 5, up.biases_offset),
            (x.0, 6, x.1),
            (out.0, 7, out.1),
        ],
        &[(u32_bytes(&m32), 8), (u32_bytes(&n32), 9)],
        threadgroups,
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// The down kernel's store epilogue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DownEpilogue {
    /// `y[row] = half(acc)` -- the plain GEMV store.
    Plain,
    /// `y[row] = half(float(y[row]) + float(half(acc)))` -- the dense tail's
    /// residual add, in place.
    Residual,
    /// `y[row] = half(float(half(acc)) / (1 + exp(-aux[0])))` -- the shared
    /// expert's scalar gate, division form, reading the logit half from
    /// `aux[0]`.
    SigmoidGate,
}

/// Down projection with the FFN tail's LAST elementwise dispatch folded into
/// the store. Bit-identical to the unfused dispatch pair; see the shader
/// header.
#[allow(clippy::too_many_arguments)]
pub fn encode_down_epi_gemv(
    context: &mut MetalContext,
    pass: &PassEncoder,
    down: &Int4ResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    aux: (&metal::Buffer, u64),
    epilogue: DownEpilogue,
) -> Result<(), GpuError> {
    let (m, n) = (down.rows, down.cols);
    if n % 64 != 0 {
        return Err(GpuError::PipelineCreate(format!(
            "down_epi: input width {n} must be a multiple of 64"
        )));
    }
    if down.weights_offset % 2 != 0 {
        return Err(GpuError::PipelineCreate(format!(
            "down_epi needs 2-aligned weight offsets, got {}",
            down.weights_offset
        )));
    }
    if x.1 % 8 != 0 {
        return Err(GpuError::PipelineCreate(format!(
            "down_epi: x offset {} must be 8-byte aligned (half4 loads)",
            x.1
        )));
    }
    let (residual, sigmoid_gate) = match epilogue {
        DownEpilogue::Plain => (false, false),
        DownEpilogue::Residual => (true, false),
        DownEpilogue::SigmoidGate => (false, true),
    };
    let (m32, n32) = (m as u32, n as u32);
    let (constants, key) = fused_constants(m32, n32, [false, residual, sigmoid_gate]);
    let p = context.pipeline(SOURCE, "dequant_int4_down_epi_gemv_simd", &constants, &key)?;
    let threadgroups = m.div_ceil(ROWS_PER_THREADGROUP as usize) as u64;
    pass.encode_threadgroups(
        &p,
        &[
            (down.buffer, 0, down.weights_offset),
            (down.buffer, 1, down.scales_offset),
            (down.buffer, 2, down.biases_offset),
            (x.0, 3, x.1),
            (y.0, 4, y.1),
            (aux.0, 5, aux.1),
        ],
        &[(u32_bytes(&m32), 6), (u32_bytes(&n32), 7)],
        threadgroups,
        THREADS_PER_GROUP,
    );
    Ok(())
}
