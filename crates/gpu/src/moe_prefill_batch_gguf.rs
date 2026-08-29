//! Host-side dispatch for `shaders/moe_prefill_batch_gguf.metal`: the
//! batched routed-expert pair for chunked prefill over MXFP4 expert blobs
//! (`docs/BATCHED_PREFILL.md` step 5). PORT-LOCAL, like everything else in
//! this port's GGUF set -- the Swift engine has no GGUF intake.
//!
//! The contract is BIT-IDENTITY with the MXFP4 DECODE pair
//! (`moe_phase1_gate_up_act_mxfp4` + `moe_phase2_down_reduce_k8_mxfp4`)
//! run M times, held by
//! `crates/gpu/tests/moe_prefill_batch_gguf_parity.rs`. Both call the same
//! `dequant_mxfp4_row_simd` on the same bytes with the same
//! [`Mxfp4Activation`] parameters, and the fused phase 2 accumulates each
//! token's slots in router-rank order -- the same summation order the
//! decode kernel takes over dispatch slots, a correctness constraint rather
//! than a style choice (AGENTS.md Gotcha 27).
//!
//! **Why MXFP4 is step 5's first arm and Q4_K/Q6_K is not**, measured
//! 2026-08-27 before either was written: `gpt-oss`'s routed pair is 61.4%
//! of its prefill GPU device time where Gemma's is 38.2%, its un-batchable
//! expert `pread` bucket is 8.2% where the two 128-expert families run
//! 25-37%, and its union stays under the slot count at every M -- so it is
//! the only family that reaches M=16, against M=8 for the others. It also
//! needs ONE block type for both phases, where `qwen3moe` puts Q4_K on
//! gate/up and Q6_K on down. The full table is in
//! `docs/BATCHED_PREFILL.md`, "Step 5's two arms, measured before building
//! either".
//!
//! The shader source is the `moe_gguf` chain, then
//! `moe_prefill_batch.metal`, then `moe_prefill_batch_gguf.metal`. The last
//! needs BOTH halves: `RoutedBlobsWide` and `MoePrefillRoute` from the
//! affine batched file, and `dequant_mxfp4_row_simd` /
//! `moe_activate_mxfp4` / `mxfp4_row_bytes` from `moe_gguf.metal`. Same two
//! caveats as every other concatenation here (Gotcha 4): it must be ONE
//! `&'static str` (the caches key on its address, not its text), and the
//! borrowed files' own kernels are compiled again in this library.

use crate::bytes::{f32_bytes, u32_bytes};
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::moe_decode::{constants_key, moe_function_constants, MoeExpertOffsets};
use crate::moe_gguf::{Mxfp4Activation, MXFP4_BLOCK_ELEMS};
use crate::moe_prefill_batch::RoutedBlobsWideBuffer;

/// The concatenation this module's kernels compile from. Public because
/// [`RoutedBlobsWideBuffer::new_for`] must be handed the SAME `&'static str`
/// constant the dispatches below pass (Gotcha 1: the caches key on address).
pub const SOURCE: &str = concat!(
    include_str!("shaders/moe.metal"),
    include_str!("shaders/dequant_q4_k.metal"),
    include_str!("shaders/dequant_q6_k.metal"),
    include_str!("shaders/dequant_iq.metal"),
    include_str!("shaders/moe_gguf.metal"),
    include_str!("shaders/moe_prefill_batch.metal"),
    include_str!("shaders/moe_prefill_batch_gguf.metal")
);

/// Phase 1's function name, the one an argument encoder for the wide
/// routed-blobs buffer must be taken from.
pub const PHASE1_MXFP4: &str = "moe_prefill_phase1_routes_mxfp4";
/// Phase 2's function name.
pub const PHASE2_MXFP4: &str = "moe_prefill_phase2_fused_mxfp4";

const ROWS_PER_THREADGROUP: u64 = 8;
const THREADS_PER_GROUP: u64 = 256;

/// The silu function constant this module's pipelines specialize on, fixed
/// rather than taken as a parameter. `FC_MOE_ACT_SILU` reaches these kernels
/// only through `moe_activate_mxfp4`'s PLAIN arm (`alpha <= 0`); the family
/// that owns this pair runs `Mxfp4Activation::GPT_OSS`, where the flag is
/// dead. What the value DOES decide is which pipeline specialization
/// compiles: the argument encoder and every dispatch must agree on
/// `constants_key`, and a caller free to pass either value could silently
/// compile a second copy of the seven-file concatenation mid-prefill
/// (Gotcha 1's cost). `true` matches the runtime's choice for this family
/// (`families/gptoss/moe.rs`: silu is the nearer wrong answer if a layout
/// ever resolved to a non-MXFP4 kernel).
const USE_SILU: bool = true;

/// Allocates the wide routed-blobs argument buffer this pair reads,
/// encoded by [`PHASE1_MXFP4`] out of [`SOURCE`].
///
/// A convenience over [`RoutedBlobsWideBuffer::new_for`] so no caller has
/// to remember which of the two libraries and which function name go
/// together; pairing them wrongly is the failure that would silently bind
/// a differently-encoded pointer array.
pub fn new_routed_blobs_wide(
    context: &mut MetalContext,
) -> Result<RoutedBlobsWideBuffer, GpuError> {
    RoutedBlobsWideBuffer::new_for(context, SOURCE, PHASE1_MXFP4, USE_SILU)
}

/// Points this pair's argument buffer at `blobs`, the sub-batch's resident
/// expert blobs in cache-slot order. See
/// [`RoutedBlobsWideBuffer::bind_for`] for the padding rule.
pub fn bind_routed_blobs_wide(
    buffer: &RoutedBlobsWideBuffer,
    context: &mut MetalContext,
    blobs: &[(&metal::Buffer, u64)],
) -> Result<(), GpuError> {
    buffer.bind_for(context, SOURCE, PHASE1_MXFP4, USE_SILU, blobs)
}

/// Phase 1 over a route list, MXFP4 blobs: for each route
/// `acts[(token * top_k + rank) * F + f] = swiglu_oai(gate_f(x_token) + gb,
/// up_f(x_token) + ub)`. `x` holds `tokens * D` halfs, `acts` holds
/// `tokens * top_k * F` halfs, `routes` holds `route_count` encoded
/// [`crate::MoePrefillRoute`] values.
///
/// The activation parameters ride as UNIFORMS, never function constants,
/// for the reason the decode pair records: `constants_key` is one byte wide
/// and a specialization axis that misses it silently reuses the wrong
/// pipeline.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_prefill_phase1_mxfp4(
    context: &mut MetalContext,
    pass: &PassEncoder,
    routed: &RoutedBlobsWideBuffer,
    offsets: &MoeExpertOffsets,
    x: (&metal::Buffer, u64),
    acts: (&metal::Buffer, u64),
    routes: (&metal::Buffer, u64),
    d_dim: u32,
    f_dim: u32,
    top_k: u32,
    route_count: u32,
    act: Mxfp4Activation,
) -> Result<(), GpuError> {
    assert_eq!(d_dim as usize % MXFP4_BLOCK_ELEMS, 0);
    assert_eq!(f_dim as usize % MXFP4_BLOCK_ELEMS, 0);
    assert!(top_k as usize <= crate::moe_decode::MAX_STREAMED_EXPERTS);
    let pipeline = context.pipeline(
        SOURCE,
        PHASE1_MXFP4,
        &moe_function_constants(USE_SILU),
        &constants_key(USE_SILU),
    )?;
    let rows = (route_count * f_dim) as u64;
    let has_bias = u32::from(act.has_bias);
    pass.encode_threadgroups(
        &pipeline,
        &[
            (routed.buffer(), 0, 0),
            (x.0, 2, x.1),
            (acts.0, 3, acts.1),
            (routes.0, 7, routes.1),
        ],
        &[
            (&offsets.bytes(), 1),
            (u32_bytes(&d_dim), 4),
            (u32_bytes(&f_dim), 5),
            (u32_bytes(&top_k), 6),
            (u32_bytes(&route_count), 8),
            (u32_bytes(&has_bias), 9),
            (f32_bytes(&act.alpha), 10),
            (f32_bytes(&act.limit), 11),
        ],
        rows.div_ceil(ROWS_PER_THREADGROUP),
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// Fused phase 2 over the same routes, MXFP4 blobs: `y[t * D + d]` is
/// `residual[t * D + d]` plus `sum_rank routing_w[t * top_k + rank] *
/// (down_d(acts[t, rank]) + db[d])`, reduced in rank order per token from
/// the residual seed -- the decode kernel's arithmetic with a token axis.
///
/// The per-expert down bias is added INSIDE the routing weight, per slot,
/// because it is that expert's own bias on that expert's own output.
/// `gpt-oss` has no shared expert, so `residual` is a zeroed buffer and the
/// seeded reduce reproduces decode's `0.0f + p` exactly.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_prefill_phase2_fused_mxfp4(
    context: &mut MetalContext,
    pass: &PassEncoder,
    routed: &RoutedBlobsWideBuffer,
    offsets: &MoeExpertOffsets,
    acts: (&metal::Buffer, u64),
    routing_w: (&metal::Buffer, u64),
    routes: (&metal::Buffer, u64),
    residual: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    d_dim: u32,
    f_dim: u32,
    top_k: u32,
    tokens: u32,
    has_bias: bool,
) -> Result<(), GpuError> {
    assert_eq!(f_dim as usize % MXFP4_BLOCK_ELEMS, 0);
    assert!(top_k as usize <= crate::moe_decode::MAX_STREAMED_EXPERTS);
    let pipeline = context.pipeline(
        SOURCE,
        PHASE2_MXFP4,
        &moe_function_constants(USE_SILU),
        &constants_key(USE_SILU),
    )?;
    let has_bias = u32::from(has_bias);
    // One threadgroup per (d, token) output, 256 threads = 8 SIMD groups,
    // one per routing rank.
    pass.encode_threadgroups_3d(
        &pipeline,
        &[
            (routed.buffer(), 0, 0),
            (acts.0, 2, acts.1),
            (routing_w.0, 3, routing_w.1),
            (routes.0, 4, routes.1),
            (y.0, 5, y.1),
            (residual.0, 10, residual.1),
        ],
        &[
            (&offsets.bytes(), 1),
            (u32_bytes(&d_dim), 6),
            (u32_bytes(&f_dim), 7),
            (u32_bytes(&top_k), 8),
            (u32_bytes(&tokens), 9),
            (u32_bytes(&has_bias), 11),
        ],
        (d_dim as u64, tokens as u64, 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}
