//! Host-side dispatch for `shaders/moe.metal`'s decode kernels (vendored
//! verbatim from `Metal/MoE/moe.metal`): `moe_phase1_gate_up_act_u16load`
//! (per selected expert, the gated INT4 gate/up GEMV plus hidden
//! activation into a per-slot activation row) and
//! `moe_phase2_down_reduce_k8` (per output dim, each slot's INT4 down
//! GEMV, weighted by its routing weight, summed on top of a residual).
//! Expert weights are read IN PLACE from up to `kMaxStreamedExperts == 8`
//! caller-owned blob buffers (streamer slots or any other page of memory)
//! through a Metal argument buffer — the `RoutedBlobs` pointer array —
//! with one uniform [`MoeExpertOffsets`] describing where each
//! sub-tensor lives inside a blob. Nothing is copied.
//!
//! The `.metal` file's routers (`router_gemv_gemma4_r4` is INT8-weight),
//! the parallel top-k selectors, and the whole DeepSeek-V4 INT2 family
//! are vendored but not dispatched yet; the runtime keeps its host top-k
//! (see `crates/runtime`'s module docs and DEVIATIONS.md).

use metal::{FunctionConstantValues, MTLDataType, MTLResourceOptions};

use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext, PassEncoder};

const SOURCE: &str = include_str!("shaders/moe.metal");
const ROWS_PER_THREADGROUP: u64 = 8;
const THREADS_PER_GROUP: u64 = 256;

/// `kMaxStreamedExperts` in the shader: the fixed slot count both decode
/// kernels are written against. Phase 2 reduces all eight partials
/// unconditionally, so unused slots must carry a zero routing weight and
/// a valid (any) blob pointer.
pub const MAX_STREAMED_EXPERTS: usize = 8;

/// Blob-relative byte offsets of the nine expert sub-tensors — the
/// shader's `ExpertOffsets` struct, uniform across all bound blobs.
/// Weight offsets need 2-byte alignment (the kernels assemble 4-byte
/// chunks from u16 loads); scale/bias offsets are BF16 so 2-byte too —
/// except `down_W_off`, which the phase-2 row helper reads with 4-byte
/// `uint` loads, so it must be 4-byte aligned.
#[derive(Debug, Clone, Copy)]
pub struct MoeExpertOffsets {
    pub gate_w: u32,
    pub gate_s: u32,
    pub gate_b: u32,
    pub up_w: u32,
    pub up_s: u32,
    pub up_b: u32,
    pub down_w: u32,
    pub down_s: u32,
    pub down_b: u32,
}

impl MoeExpertOffsets {
    fn bytes(&self) -> [u8; 36] {
        let mut out = [0u8; 36];
        for (i, v) in [
            self.gate_w,
            self.gate_s,
            self.gate_b,
            self.up_w,
            self.up_s,
            self.up_b,
            self.down_w,
            self.down_s,
            self.down_b,
        ]
        .into_iter()
        .enumerate()
        {
            out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        out
    }
}

/// The shader's function constants: `FC_MOE_D`(0)/`F`(1)/`TOP_K`(2)/
/// `USE_FC`(3, false = runtime arguments)/`ACT_SILU`(4)/`SWIGLU_LIMIT`(5)
/// plus the router constants (40-43) other kernels in the file declare.
fn moe_function_constants(use_silu: bool) -> FunctionConstantValues {
    let values = FunctionConstantValues::new();
    let zero: u32 = 0;
    let use_fc = false;
    let limit: f32 = 0.0;
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 0);
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 1);
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 2);
    values.set_constant_value_at_index((&use_fc as *const bool).cast(), MTLDataType::Bool, 3);
    values.set_constant_value_at_index((&use_silu as *const bool).cast(), MTLDataType::Bool, 4);
    values.set_constant_value_at_index((&limit as *const f32).cast(), MTLDataType::Float, 5);
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 40);
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 41);
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 42);
    values.set_constant_value_at_index((&use_fc as *const bool).cast(), MTLDataType::Bool, 43);
    values
}

fn constants_key(use_silu: bool) -> [u8; 1] {
    [use_silu as u8]
}

/// One reusable `RoutedBlobs` argument buffer (the Swift original's
/// `reusableRoutedArgBuffer`): allocated once, re-encoded with the
/// current token's expert blob pointers before each MoE dispatch pair.
pub struct RoutedBlobsBuffer {
    buffer: metal::Buffer,
}

impl RoutedBlobsBuffer {
    pub fn new(context: &mut MetalContext, use_silu: bool) -> Result<Self, GpuError> {
        let encoder = context.argument_encoder(
            SOURCE,
            "moe_phase1_gate_up_act_u16load",
            &moe_function_constants(use_silu),
            &constants_key(use_silu),
            0,
        )?;
        let buffer = context.device().new_buffer(
            encoder.encoded_length(),
            MTLResourceOptions::StorageModeShared,
        );
        Ok(Self { buffer })
    }

    /// Points the argument buffer's `blob[i]` entries at `blobs[i]`
    /// (buffer + byte offset). Entries past `blobs.len()` are bound to
    /// `blobs[0]` so phase 2's unconditional eight-slot reduce reads
    /// valid memory (their routing weights must be zero).
    pub fn bind(
        &self,
        context: &mut MetalContext,
        use_silu: bool,
        blobs: &[(&metal::Buffer, u64)],
    ) -> Result<(), GpuError> {
        assert!(!blobs.is_empty() && blobs.len() <= MAX_STREAMED_EXPERTS);
        let encoder = context.argument_encoder(
            SOURCE,
            "moe_phase1_gate_up_act_u16load",
            &moe_function_constants(use_silu),
            &constants_key(use_silu),
            0,
        )?;
        encoder.set_argument_buffer(&self.buffer, 0);
        for i in 0..MAX_STREAMED_EXPERTS {
            let (buffer, offset) = blobs[i.min(blobs.len() - 1)];
            encoder.set_buffer(i as u64, buffer, offset);
        }
        Ok(())
    }

    pub fn buffer(&self) -> &metal::Buffer {
        &self.buffer
    }
}

/// Phase 1: for each of `top_k` slots, `acts[slot*f_dim + f] =
/// activation(gate_f(x)) * up_f(x)` over the slot's blob. `x` holds
/// `d_dim` halfs; `acts` holds `top_k * f_dim` halfs.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_phase1(
    context: &mut MetalContext,
    pass: &PassEncoder,
    routed: &RoutedBlobsBuffer,
    offsets: &MoeExpertOffsets,
    x: (&metal::Buffer, u64),
    acts: (&metal::Buffer, u64),
    d_dim: u32,
    f_dim: u32,
    top_k: u32,
    use_silu: bool,
) -> Result<(), GpuError> {
    assert_eq!(d_dim % 64, 0);
    assert!(top_k as usize <= MAX_STREAMED_EXPERTS);
    let pipeline = context.pipeline(
        SOURCE,
        "moe_phase1_gate_up_act_u16load",
        &moe_function_constants(use_silu),
        &constants_key(use_silu),
    )?;
    let rows = (top_k * f_dim) as u64;
    pass.encode_threadgroups(
        &pipeline,
        &[(routed.buffer(), 0, 0), (x.0, 2, x.1), (acts.0, 3, acts.1)],
        &[
            (&offsets.bytes(), 1),
            (u32_bytes(&d_dim), 4),
            (u32_bytes(&f_dim), 5),
            (u32_bytes(&top_k), 6),
        ],
        rows.div_ceil(ROWS_PER_THREADGROUP),
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// Phase 2: `y[d] = residual[d] + sum_slot routing_w[slot] *
/// down_d(acts[slot])`. `routing_w` must hold exactly
/// [`MAX_STREAMED_EXPERTS`] halfs (zero-padded past `top_k`); `residual`
/// and `y` hold `d_dim` halfs.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_phase2(
    context: &mut MetalContext,
    pass: &PassEncoder,
    routed: &RoutedBlobsBuffer,
    offsets: &MoeExpertOffsets,
    acts: (&metal::Buffer, u64),
    routing_w: (&metal::Buffer, u64),
    residual: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    d_dim: u32,
    f_dim: u32,
    use_silu: bool,
) -> Result<(), GpuError> {
    assert_eq!(f_dim % 64, 0);
    let pipeline = context.pipeline(
        SOURCE,
        "moe_phase2_down_reduce_k8",
        &moe_function_constants(use_silu),
        &constants_key(use_silu),
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[
            (routed.buffer(), 0, 0),
            (acts.0, 2, acts.1),
            (routing_w.0, 3, routing_w.1),
            (residual.0, 4, residual.1),
            (y.0, 5, y.1),
        ],
        &[
            (&offsets.bytes(), 1),
            (u32_bytes(&d_dim), 6),
            (u32_bytes(&f_dim), 7),
        ],
        d_dim as u64,
        THREADS_PER_GROUP,
    );
    Ok(())
}
