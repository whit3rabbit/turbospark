//! Host-side dispatch for `shaders/moe.metal`'s decode kernels (vendored
//! verbatim from `Metal/MoE/moe.metal`): `moe_phase1_gate_up_act_u16load`
//! (per selected expert, the gated INT4 gate/up GEMV plus hidden
//! activation into a per-slot activation row) and
//! `moe_phase2_down_reduce_k8` (per output dim, each slot's INT4 down
//! GEMV, weighted by its routing weight, summed on top of a residual).
//! Expert weights are read IN PLACE from up to
//! `kMaxStreamedExperts == `[`MAX_STREAMED_EXPERTS`] caller-owned blob
//! buffers (streamer slots or any other page of memory)
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

/// This module's compiled library, for a caller that needs to reflect one of
/// ITS functions' argument-buffer layout rather than assume it agrees with
/// another library's (S7: [`RoutedBlobsBuffer::new_for`]/[`bind_for`]).
pub fn moe_decode_source() -> &'static str {
    SOURCE
}

/// `kMaxStreamedExperts` in the shader: the CEILING both decode kernels'
/// fixed-size buffers are sized against (16, widened from 8 for
/// `qwen4_exp`'s top_k=10). Phase 1's dispatch width is proportional to
/// `top_k * f_dim` regardless of this ceiling; phase 2's dispatch width and
/// its reduce loop are both `top_k` slots exactly (`encode_moe_phase2`), so
/// a caller below the ceiling pays no padding-slot compute. `RoutedBlobs`'
/// pointer array and the routing-weight buffer are still sized at the full
/// ceiling and zero/pointer-padded past `top_k` (`RoutedBlobsBuffer::bind`),
/// because a caller with a smaller top_k than another install open in the
/// same process must not under-allocate a buffer a wider install also uses.
pub const MAX_STREAMED_EXPERTS: usize = 16;

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
    pub(crate) fn bytes(&self) -> [u8; 36] {
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
pub(crate) fn moe_function_constants(use_silu: bool) -> FunctionConstantValues {
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

pub(crate) fn constants_key(use_silu: bool) -> [u8; 1] {
    [use_silu as u8]
}

/// One reusable `RoutedBlobs` argument buffer (the Swift original's
/// `reusableRoutedArgBuffer`): allocated once, re-encoded with the
/// current token's expert blob pointers before each MoE dispatch pair.
pub struct RoutedBlobsBuffer {
    buffer: metal::Buffer,
}

impl RoutedBlobsBuffer {
    /// Reflects `moe_decode::SOURCE`'s own `moe_phase1_gate_up_act_u16load`
    /// (the vendored INT4-affine pair). Every `moe_gguf` block type shares a
    /// DIFFERENT library and needs [`Self::new_for`] instead (S7).
    pub fn new(context: &mut MetalContext, use_silu: bool) -> Result<Self, GpuError> {
        Self::new_for(context, SOURCE, "moe_phase1_gate_up_act_u16load", use_silu)
    }

    /// The same, reflecting a caller-chosen library and function's
    /// `RoutedBlobs` argument-buffer layout instead of this module's own.
    ///
    /// `RoutedBlobsBuffer` is bound at dispatch time to whichever per-layer
    /// kernel a `RoutedLayerLayout` resolves (`moe_gguf`'s Q8_0/Q4_K/IQ*/
    /// MXFP4 pairs, or this module's vendored pair), and reflecting the
    /// wrong library's layout to build the encoder is an unenforced
    /// assumption (S7) -- the same reasoning [`crate::RoutedBlobsWideBuffer::new_for`]
    /// states for its own two known alternatives. `source` must be the same
    /// `&'static str` constant the matching dispatch passes: the pipeline and
    /// argument-encoder caches key on its ADDRESS, not its text (Gotcha 1).
    pub fn new_for(
        context: &mut MetalContext,
        source: &'static str,
        function: &'static str,
        use_silu: bool,
    ) -> Result<Self, GpuError> {
        let encoder = context.argument_encoder(
            source,
            function,
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
    /// `blobs[0]` so a WIDER install sharing this argument-buffer layout
    /// still reads valid memory at any slot index up to
    /// [`MAX_STREAMED_EXPERTS`]; phase 2 itself only reduces the first
    /// `top_k` of them (`encode_moe_phase2`), so padding past `blobs.len()`
    /// is defensive rather than load-bearing for a single install.
    ///
    /// Reflects the same `moe_phase1_gate_up_act_u16load` layout [`Self::new`]
    /// built the buffer against. Use [`Self::bind_for`] for a buffer created
    /// with [`Self::new_for`].
    pub fn bind(
        &self,
        context: &mut MetalContext,
        use_silu: bool,
        blobs: &[(&metal::Buffer, u64)],
    ) -> Result<(), GpuError> {
        self.bind_for(
            context,
            SOURCE,
            "moe_phase1_gate_up_act_u16load",
            use_silu,
            blobs,
        )
    }

    /// [`Self::bind`] against a caller-chosen library and function's layout.
    /// `source`/`function` must be the pair the buffer was created with
    /// (see [`Self::new_for`]).
    pub fn bind_for(
        &self,
        context: &mut MetalContext,
        source: &'static str,
        function: &'static str,
        use_silu: bool,
        blobs: &[(&metal::Buffer, u64)],
    ) -> Result<(), GpuError> {
        assert!(!blobs.is_empty() && blobs.len() <= MAX_STREAMED_EXPERTS);
        let encoder = context.argument_encoder(
            source,
            function,
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
/// down_d(acts[slot])`, summed over exactly the first `top_k` slots (slot
/// order is summation order -- AGENTS.md Gotcha 8). `routing_w` must hold
/// at least `top_k` halfs; `residual` and `y` hold `d_dim` halfs.
///
/// The dispatch width is `top_k` simdgroups (one per slot, matching the
/// kernel's `sg_idx`-indexed reduce), never the [`MAX_STREAMED_EXPERTS`]
/// ceiling: a caller at the ceiling's own top_k (`qwen4_exp`) reduces every
/// dispatched slot with no padding, and a caller below it (every
/// pre-widening family) dispatches exactly as many threads as it always
/// did, so this widening costs no phase-2 throughput on an unaffected
/// install.
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
    top_k: u32,
    use_silu: bool,
) -> Result<(), GpuError> {
    assert_eq!(f_dim % 64, 0);
    assert!((1..=MAX_STREAMED_EXPERTS as u32).contains(&top_k));
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
            (u32_bytes(&top_k), 8),
        ],
        d_dim as u64,
        top_k as u64 * 32,
    );
    Ok(())
}

/// Encoder-level Gemma 4 INT8 router GEMV (`router_gemv_gemma4_r4`):
/// `logits[e] = sum_n dequant_int8(W[e, n]) * x[n] * effective_scale[n]`,
/// with `effective_scale` a BF16 `[D]` vector (the checkpoint's
/// `router.scale` with `1/sqrt(D)` pre-folded in — see the Swift
/// `RealForwardRunner`'s effective-scale buffers). Weights, scales, and
/// biases are `(buffer, byte offset)` views, normally into the resident
/// buffer; `out_logits` receives `num_experts` FP32 values.
#[allow(clippy::too_many_arguments)]
pub fn encode_router_gemv_gemma4(
    context: &mut MetalContext,
    pass: &PassEncoder,
    weights: (&metal::Buffer, u64),
    scales: (&metal::Buffer, u64),
    biases: (&metal::Buffer, u64),
    hidden: (&metal::Buffer, u64),
    effective_scale: (&metal::Buffer, u64),
    out_logits: (&metal::Buffer, u64),
    num_experts: u32,
    d_dim: u32,
) -> Result<(), GpuError> {
    assert_eq!(d_dim % 64, 0);
    let pipeline = context.pipeline(
        SOURCE,
        "router_gemv_gemma4_r4",
        &moe_function_constants(false),
        &constants_key(false),
    )?;
    // 4 expert rows per threadgroup, one SIMD group (32 lanes) per row.
    let rows_per_tg = 4u64;
    pass.encode_threadgroups(
        &pipeline,
        &[
            (weights.0, 0, weights.1),
            (scales.0, 1, scales.1),
            (biases.0, 2, biases.1),
            (hidden.0, 3, hidden.1),
            (effective_scale.0, 4, effective_scale.1),
            (out_logits.0, 5, out_logits.1),
        ],
        &[(u32_bytes(&num_experts), 6), (u32_bytes(&d_dim), 7)],
        (num_experts as u64).div_ceil(rows_per_tg),
        rows_per_tg * 32,
    );
    Ok(())
}

/// One-shot [`encode_router_gemv_gemma4`] over host slices, for the parity
/// tests: `w_bytes` is `[num_experts, d]` INT8 weight bytes,
/// `scale_bits`/`bias_bits` are `[num_experts, d/64]` BF16 bit patterns,
/// `eff_bits` is the `[d]` BF16 effective scale.
pub fn router_gemv_gemma4(
    context: &mut MetalContext,
    w_bytes: &[u8],
    scale_bits: &[u16],
    bias_bits: &[u16],
    x: &[half::f16],
    eff_bits: &[u16],
    num_experts: u32,
) -> Result<Vec<f32>, GpuError> {
    let d = x.len() as u32;
    assert_eq!(w_bytes.len(), (num_experts * d) as usize);
    assert_eq!(eff_bits.len(), d as usize);
    let w_buffer = context.new_buffer_with_data(w_bytes);
    let s_buffer = context.new_buffer_with_data(&crate::bytes::u16_slice_to_le_bytes(scale_bits));
    let b_buffer = context.new_buffer_with_data(&crate::bytes::u16_slice_to_le_bytes(bias_bits));
    let x_buffer = context.new_buffer_with_data(&crate::bytes::half_slice_to_le_bytes(x));
    let e_buffer = context.new_buffer_with_data(&crate::bytes::u16_slice_to_le_bytes(eff_bits));
    let out_buffer = context.new_output_buffer(num_experts as u64 * 4);

    let pass = context.begin_pass();
    encode_router_gemv_gemma4(
        context,
        &pass,
        (&w_buffer, 0),
        (&s_buffer, 0),
        (&b_buffer, 0),
        (&x_buffer, 0),
        (&e_buffer, 0),
        (&out_buffer, 0),
        num_experts,
        d,
    )?;
    pass.commit_and_wait();
    Ok(crate::bytes::read_f32_buffer(
        &out_buffer,
        num_experts as usize,
    ))
}
