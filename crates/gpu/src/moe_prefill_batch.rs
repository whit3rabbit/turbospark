//! Host-side dispatch for `shaders/moe_prefill_batch.metal`: the batched
//! routed-expert pair for chunked prefill (`docs/BATCHED_PREFILL.md`
//! steps 2 and 3), PORT-LOCAL like the GGUF pairs -- Swift's engine has
//! no wide-binding form (its `dsv4_prefill_moe_*` kernels tile eight
//! experts at a time).
//!
//! The contract is BIT-IDENTITY with the vendored decode pair
//! (`moe_phase1_gate_up_act_u16load` + `moe_phase2_down_reduce_k8`) run
//! M times, held by `crates/gpu/tests/moe_prefill_batch_parity.rs`. Both
//! kernels call the same INT4 row helpers on the same bytes, and the
//! fused phase 2 accumulates each token's slots in router-rank order --
//! the same summation order the decode kernel takes over dispatch slots,
//! which is a correctness constraint rather than a style choice (AGENTS.md
//! Gotcha 27).
//!
//! The shader source is `moe.metal` then `moe_prefill_batch.metal`, in
//! that order: the second uses the first's `ExpertOffsets`, the INT4 row
//! helpers and `moe_fc_*` declarations. Same caveats as the `moe_gguf`
//! arrangement: the concatenation must be ONE `&'static str` (the
//! pipeline cache keys on its address, not its text) and the borrowed
//! file's own kernels are compiled twice in this process.

use metal::MTLResourceOptions;

use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::moe_decode::{constants_key, moe_function_constants, MoeExpertOffsets};

const SOURCE: &str = concat!(
    include_str!("shaders/moe.metal"),
    include_str!("shaders/moe_prefill_batch.metal")
);
const ROWS_PER_THREADGROUP: u64 = 8;
const THREADS_PER_GROUP: u64 = 256;

/// `kMaxPrefillExpertBindings` in the shader: the width of the wide
/// argument buffer. A routed sub-batch must keep every expert of its
/// UNION resident at once, which is the caller's constraint
/// (`union <= slot_count`); this width is the ceiling on `slot_count`
/// the kernels can name, matching the largest value in the CLI's
/// allowed set.
pub const MAX_PREFILL_EXPERT_BINDINGS: usize = 32;

/// The shader's `MoePrefillRoute`: which token's row, which routing rank
/// (the router's own ranking -- load-bearing for the phase-2 reduce
/// order), and which cache slot the expert's blob lives in.
#[derive(Debug, Clone, Copy)]
pub struct MoePrefillRoute {
    pub token: u32,
    pub rank: u32,
    pub slot: u32,
}

impl MoePrefillRoute {
    /// Serializes a slice of prefill routes into little-endian byte representation.
    pub fn bytes(routes: &[MoePrefillRoute]) -> Vec<u8> {
        let mut out = Vec::with_capacity(routes.len() * 16);
        for r in routes {
            out.extend_from_slice(&r.token.to_le_bytes());
            out.extend_from_slice(&r.rank.to_le_bytes());
            out.extend_from_slice(&r.slot.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
        }
        out
    }
}

/// One reusable wide `RoutedBlobsWide` argument buffer: allocated once,
/// re-encoded with the sub-batch's expert blob pointers before each
/// batched dispatch pair. Unlike [`crate::RoutedBlobsBuffer`] it is
/// HOST-WRITTEN ONCE per layer, not per token: the chunk driver retires
/// the previous layer's routed command buffer before this layer's routed
/// half begins (see `families/gemma4/mod.rs`), so one bank is enough.
/// `Clone` is a refcount bump on the underlying `MTLBuffer` -- callers
/// hold a second HANDLE to the same argument buffer, never a copy.
#[derive(Clone)]
pub struct RoutedBlobsWideBuffer {
    buffer: metal::Buffer,
}

impl RoutedBlobsWideBuffer {
    /// Creates a new wide routed-blobs argument buffer for the INT4-affine
    /// batched pair.
    pub fn new(context: &mut MetalContext, use_silu: bool) -> Result<Self, GpuError> {
        Self::new_for(context, SOURCE, "moe_prefill_phase1_routes_int4", use_silu)
    }

    /// The same, for a batched pair living in a DIFFERENT shader library --
    /// `moe_prefill_batch_gguf.rs`'s MXFP4 pair is the second caller
    /// (`docs/BATCHED_PREFILL.md` step 5).
    ///
    /// The two libraries declare `RoutedBlobsWide` identically, so it is
    /// tempting to encode both with one function and share the buffer. That
    /// is an assumption about `encoded_length` across two separate
    /// compilations, and it is the kind this repo pays for: the encoder is
    /// taken from the function that will READ the buffer, so a layout that
    /// ever diverged would be a compile-time mismatch rather than a silently
    /// misread pointer array.
    ///
    /// `source` must be the same `&'static str` constant the matching
    /// dispatch passes -- the pipeline and argument-encoder caches key on its
    /// ADDRESS, not its text (Gotcha 1).
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

    /// Points the argument buffer's `blob[i]` entries at `blobs[i]`.
    /// Entries past `blobs.len()` repeat the LAST blob; padded ranks never
    /// read them (the kernel guards `sg_idx < top_k`), but a valid pointer
    /// is bound anyway so an argument-buffer validation pass has nothing to
    /// reject. Which blob the padding repeats is arbitrary and differs from
    /// [`crate::RoutedBlobsBuffer::bind`], which repeats `blobs[0]`.
    pub fn bind(
        &self,
        context: &mut MetalContext,
        use_silu: bool,
        blobs: &[(&metal::Buffer, u64)],
    ) -> Result<(), GpuError> {
        self.bind_for(
            context,
            SOURCE,
            "moe_prefill_phase1_routes_int4",
            use_silu,
            blobs,
        )
    }

    /// [`Self::bind`] for a pair in a different shader library. See
    /// [`Self::new_for`]; `source` and `function` must be the pair the
    /// buffer was created for.
    pub fn bind_for(
        &self,
        context: &mut MetalContext,
        source: &'static str,
        function: &'static str,
        use_silu: bool,
        blobs: &[(&metal::Buffer, u64)],
    ) -> Result<(), GpuError> {
        assert!(!blobs.is_empty() && blobs.len() <= MAX_PREFILL_EXPERT_BINDINGS);
        let encoder = context.argument_encoder(
            source,
            function,
            &moe_function_constants(use_silu),
            &constants_key(use_silu),
            0,
        )?;
        encoder.set_argument_buffer(&self.buffer, 0);
        for i in 0..MAX_PREFILL_EXPERT_BINDINGS {
            let (buffer, offset) = blobs[i.min(blobs.len() - 1)];
            encoder.set_buffer(i as u64, buffer, offset);
        }
        Ok(())
    }

    /// Returns a reference to the underlying Metal argument buffer.
    pub fn buffer(&self) -> &metal::Buffer {
        &self.buffer
    }
}

/// Phase 1 over a route list: for each route
/// `acts[(token * top_k + rank) * F + f] = act(gate_f(x_token)) *
/// up_f(x_token)`. `x` holds `tokens * D` halfs, `acts` holds
/// `tokens * top_k * F` halfs, `routes` holds `route_count` encoded
/// [`MoePrefillRoute`] values.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_prefill_phase1(
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
    use_silu: bool,
) -> Result<(), GpuError> {
    assert_eq!(d_dim % 64, 0);
    assert_eq!(f_dim % 64, 0);
    let pipeline = context.pipeline(
        SOURCE,
        "moe_prefill_phase1_routes_int4",
        &moe_function_constants(use_silu),
        &constants_key(use_silu),
    )?;
    let rows = (route_count * f_dim) as u64;
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
        ],
        rows.div_ceil(ROWS_PER_THREADGROUP),
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// Fused phase 2 over the same routes, computing `y[t * D + d]` as
/// `residual[t * D + d]` plus `sum_rank routing_w[t * top_k + rank] *
/// down_d(acts[t, rank])`, reduced in rank order per token from the
/// RESIDUAL seed -- the decode kernel's arithmetic with a token axis.
/// `routing_w` holds `tokens * top_k` halfs; `residual` and `y` hold
/// `tokens * D` halfs and may be the same buffer.
///
/// `residual` mirrors `moe_phase2_down_reduce_k8`'s argument of that name
/// and exists for one family. Gemma, `llama`, `gpt-oss` and the synthetic
/// flow all pass `zero_hidden` on the decode path, which is why this
/// kernel hardcoded a `0.0f` seed and was right to; `qwen3_5` passes its
/// GATED SHARED-EXPERT output, and FP addition is not associative, so
/// `(h1 + p0 + ... + p7)` is not `(p0 + ... + p7) + h1` and the shared
/// expert cannot be folded in by a later residual add. Callers with no
/// shared expert bind a zeroed buffer and get the old arithmetic exactly:
/// `0.0f + p` is the same operation the previous seed performed.
///
/// A UNIFORM `has_residual` flag was the alternative and is not taken: a
/// flag is a second thing to get wrong (pass it wrong and the seed is
/// silently zeros or garbage), where a zeroed buffer is inert, and it
/// would put a branch in the reduce. A function constant is refused for
/// Gotcha 1's reason -- a specialization axis missing from
/// `constants_key` silently reuses whichever pipeline compiled first.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_prefill_phase2_fused(
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
    use_silu: bool,
) -> Result<(), GpuError> {
    assert_eq!(f_dim % 64, 0);
    assert!(top_k as usize <= crate::moe_decode::MAX_STREAMED_EXPERTS);
    let pipeline = context.pipeline(
        SOURCE,
        "moe_prefill_phase2_fused_int4",
        &moe_function_constants(use_silu),
        &constants_key(use_silu),
    )?;
    // One threadgroup per (d, token) output, 256 threads = 8 SIMD
    // groups, one per routing rank.
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
        ],
        (d_dim as u64, tokens as u64, 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}
