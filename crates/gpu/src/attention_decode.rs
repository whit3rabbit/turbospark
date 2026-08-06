//! Host-side dispatch for single-token decode attention, from
//! `shaders/attention.metal` (vendored verbatim from
//! `Metal/Attention/attention.metal`). Dispatches the two-pass
//! split-KV (Flash-Decoding) kernels `attention_decode_partial` +
//! `attention_decode_combine` with `num_chunks == 1`: one pass over the
//! whole `[kv_start, seq_len)` range per Q head, no chunk-combine rescale
//! needed (a single chunk's `m_glob` equals its own `m`, so the combine
//! pass reduces to `out = o / d`, byte-identical to a fused single-pass
//! kernel — the doc comment in the vendored shader spells this out).
//! Matches `mrefrust_compute::causal_attention`'s layout and (`window:
//! None`) semantics exactly.
//!
//! `attention.metal` also ships `attention_decode_gqa_swa_partial` (a
//! performance variant of `attention_decode_partial` that shares K/V reads
//! across the Q heads mapped to one KV head — not required for
//! correctness, since `attention_decode_partial` already handles GQA by
//! recomputing the shared KV head per Q head) and the whole MPP prefill
//! path (`attention_prefill_causal_tiled`,
//! `attention_prefill_full_tensorops_2d_validity_v2`); neither is
//! dispatched here. The
//! ring-buffer KV addressing (`FC_ATTN_RING_CAP`) IS wired:
//! [`encode_attention_decode`] takes a `ring_capacity` (0 = linear layout)
//! that `crates/runtime`'s `RealForwardRunner` activates for
//! sliding-window layers once `seq_len` exceeds the ring, matching the
//! Swift runner's rule (identity slot mapping below capacity, so the
//! non-ring pipeline stays byte-identical until the first wrap).

use half::f16;
use metal::{FunctionConstantValues, MTLDataType};

use crate::bytes::{f32_bytes, half_slice_to_le_bytes, read_half_buffer, u32_bytes};
use crate::context::{dispatch_one_threadgroup_per_row, GpuError, MetalContext, PassEncoder};

const SOURCE: &str = include_str!("shaders/attention.metal");
const THREADS_PER_GROUP: u64 = 256; // kAttnThreads.

/// Upper bound on the split-KV chunk count, and therefore on how many
/// partial slots [`AttentionScratch`] has to hold. Swift splits 16 ways
/// too. `crates/gpu/tests/attention_chunk_bench.rs` is where the number
/// comes from: at the real Gemma 4 shapes, 16 is at or within a few
/// percent of the best chunk count at every context from 256 to 4096,
/// while 32 starts regressing (more empty chunks and a wider combine for
/// no extra parallelism).
const MAX_CHUNKS: u32 = 16;

/// How many KV positions one threadgroup must own before splitting again.
/// Chunks shorter than this are not worth their share of the combine
/// pass, and a chunk count above the range length would dispatch
/// threadgroups whose loop never executes.
const MIN_POSITIONS_PER_CHUNK: u32 = 16;

/// Splits `[kv_start, seq_len)` across threadgroups.
///
/// `attention_decode_partial` dispatches exactly `num_q_heads *
/// num_chunks` threadgroups, so at one chunk a Gemma 4 decode attention
/// occupies 16 of them and each walks the whole KV range serially, two
/// threadgroup-wide reductions per position. That is occupancy-bound, not
/// bandwidth-bound: chunking is worth 6x at 256 positions and 12x at 4096
/// on an M4 Max (see [`MAX_CHUNKS`]).
///
/// Returning 1 for short ranges is not just an optimization: it keeps
/// short-context decode bit-identical to the single-chunk path, since
/// `num_chunks > 1` reassociates the online-softmax partial sums.
///
/// Rounded DOWN to a power of two so the result only ever takes five
/// values. The chunk count is specialized into the pipeline, so every
/// distinct value is a separate MSL compile of `attention.metal`, and
/// prefill walks the whole range one token at a time — an unrounded rule
/// would compile 16 pipelines per (scale, ring) pair on the way up. The
/// bench has 8 and 16 within a few percent of each other, so the rounding
/// costs nothing measurable.
fn chunks_for(range: u32) -> u32 {
    let chunks = (range / MIN_POSITIONS_PER_CHUNK).clamp(1, MAX_CHUNKS);
    1 << chunks.ilog2()
}

/// `attention_decode_partial` references function constants
/// `FC_ATTN_HEAD_DIM`(60)/`FC_ATTN_NUM_Q_HEADS`(61)/`FC_ATTN_NUM_KV_HEADS`(62)/
/// `FC_ATTN_USE_FC`(63)/`FC_ATTN_SCALE`(64)/`FC_ATTN_NUM_CHUNKS`(65)/
/// `FC_ATTN_RING_CAP`(69) (transitively, via its `attn_fc_*`/`attn_ring_slot`
/// helper calls); `attention_decode_combine` only references a subset
/// (60/63/65). Specializing the full set for both pipelines is simplest and
/// harmless — with one exception: `FC_ATTN_SCALE` and `FC_ATTN_NUM_CHUNKS`
/// are checked *unconditionally* in the shader (`is_function_constant_defined`
/// with no `FC_ATTN_USE_FC` gate, unlike head_dim/num_q_heads/num_kv_heads),
/// so specializing them at all makes the shader use the specialized value
/// instead of the runtime buffer argument — a dummy `0` for
/// `FC_ATTN_NUM_CHUNKS` previously caused a `% 0` inside the kernel. All
/// three must be set to this dispatch's real values: `scale` and
/// `num_chunks` are whatever the caller resolved ([`chunks_for`]), and
/// `FC_ATTN_RING_CAP` is the caller's `ring_capacity` (0 = linear
/// layout; nonzero makes every K/V row read go through
/// `attn_ring_slot(p) = p % cap`). All three variable constants' bytes go
/// into the pipeline-cache constants key ([`attention_constants_key`]) so
/// each (scale, ring, chunks) triple caches as a distinct pipeline —
/// omitting the ring from the key would silently reuse the linear
/// pipeline for ring dispatches, and omitting the chunk count would reuse
/// a pipeline whose specialized `NC` disagrees with the grid, which
/// scrambles the chunk-to-threadgroup mapping instead of failing.
fn attention_function_constants(
    scale: f32,
    ring_capacity: u32,
    num_chunks: u32,
) -> FunctionConstantValues {
    let values = FunctionConstantValues::new();
    let zero_u32: u32 = 0;
    let use_fc = false;
    values.set_constant_value_at_index((&zero_u32 as *const u32).cast(), MTLDataType::UInt, 60);
    values.set_constant_value_at_index((&zero_u32 as *const u32).cast(), MTLDataType::UInt, 61);
    values.set_constant_value_at_index((&zero_u32 as *const u32).cast(), MTLDataType::UInt, 62);
    values.set_constant_value_at_index((&use_fc as *const bool).cast(), MTLDataType::Bool, 63);
    values.set_constant_value_at_index((&scale as *const f32).cast(), MTLDataType::Float, 64);
    values.set_constant_value_at_index((&num_chunks as *const u32).cast(), MTLDataType::UInt, 65);
    values.set_constant_value_at_index(
        (&ring_capacity as *const u32).cast(),
        MTLDataType::UInt,
        69,
    );
    values
}

fn attention_constants_key(scale: f32, ring_capacity: u32, num_chunks: u32) -> [u8; 12] {
    let mut key = [0u8; 12];
    key[..4].copy_from_slice(&scale.to_le_bytes());
    key[4..8].copy_from_slice(&ring_capacity.to_le_bytes());
    key[8..].copy_from_slice(&num_chunks.to_le_bytes());
    key
}

/// `Q: [num_q_heads, head_dim]`, `K`/`V: [seq_len, num_kv_heads, head_dim]`
/// (same layout `mrefrust_compute::causal_attention` uses). Returns
/// `[num_q_heads, head_dim]`. `num_q_heads` must be a multiple of
/// `num_kv_heads` (GQA). No sliding-window support (always attends over
/// the full `[0, seq_len)` range) — see module docs.
#[allow(clippy::too_many_arguments)]
pub fn attention_decode(
    context: &mut MetalContext,
    q: &[f16],
    k: &[f16],
    v: &[f16],
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    seq_len: u32,
    scale: f32,
) -> Result<Vec<f16>, GpuError> {
    assert_eq!(q.len(), (num_q_heads * head_dim) as usize);
    assert_eq!(k.len(), (seq_len * num_kv_heads * head_dim) as usize);
    assert_eq!(v.len(), (seq_len * num_kv_heads * head_dim) as usize);
    assert_eq!(num_q_heads % num_kv_heads, 0);

    let k_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(k));
    let v_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(v));
    attention_decode_buffers(
        context,
        q,
        &k_buffer,
        &v_buffer,
        head_dim,
        num_q_heads,
        num_kv_heads,
        seq_len,
        scale,
    )
}

/// Caller-owned scratch for the two-pass decode attention: the partial
/// max/denominator/output accumulators the combine pass folds. Sized once
/// (per runner) for `num_q_heads`/`head_dim`; reused every token.
///
/// Always sized for [`MAX_CHUNKS`] partials per Q head, not for the chunk
/// count of any one dispatch, because the count varies with context
/// length ([`chunks_for`]) and the buffers are allocated once at startup.
/// The waste is bounded and small: 16 Q heads x 16 chunks x 512 head_dim
/// x 4 bytes is 512 KiB against a multi-GiB session.
pub struct AttentionScratch {
    pub m: metal::Buffer,
    pub d: metal::Buffer,
    pub o: metal::Buffer,
}

impl AttentionScratch {
    pub fn new(context: &MetalContext, num_q_heads: u32, head_dim: u32) -> Self {
        let slots = (num_q_heads * MAX_CHUNKS) as u64;
        Self {
            m: context.new_output_buffer(slots * 4),
            d: context.new_output_buffer(slots * 4),
            o: context.new_output_buffer(slots * head_dim as u64 * 4),
        }
    }
}

/// Encoder-level variant of [`attention_decode_buffers`]: Q at a
/// `(buffer, byte offset)` view, K/V read in place from persistent cache
/// buffers, output written to `out`, both passes appended to `pass`.
///
/// `kv_start` restricts attention to `[kv_start, seq_len)` — pass
/// `seq_len - window` (clamped at 0) for a sliding-window layer, `0` for
/// full attention; the kernel's chunk arithmetic handles both identically.
///
/// `ring_capacity` selects the KV addressing: `0` = linear layout (K/V row
/// for logical position `p` lives at row `p`); nonzero = ring layout (row
/// `p % ring_capacity`). The read range stays logical (`[kv_start,
/// seq_len)`), so callers using the ring must guarantee
/// `seq_len - kv_start <= ring_capacity` or older rows alias newer ones.
#[allow(clippy::too_many_arguments)]
pub fn encode_attention_decode(
    context: &mut MetalContext,
    pass: &PassEncoder,
    q: (&metal::Buffer, u64),
    k_buffer: &metal::Buffer,
    v_buffer: &metal::Buffer,
    scratch: &AttentionScratch,
    out: (&metal::Buffer, u64),
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    seq_len: u32,
    kv_start: u32,
    ring_capacity: u32,
    scale: f32,
) -> Result<(), GpuError> {
    assert_eq!(num_q_heads % num_kv_heads, 0);
    assert!(kv_start < seq_len);
    assert!(
        ring_capacity == 0 || seq_len - kv_start <= ring_capacity,
        "attention window larger than ring capacity: rows would alias"
    );
    let stored_tokens = if ring_capacity > 0 {
        ring_capacity
    } else {
        seq_len
    };
    let kv_bytes = (stored_tokens * num_kv_heads * head_dim) as u64 * 2;
    assert!(k_buffer.length() >= kv_bytes, "K buffer too small");
    assert!(v_buffer.length() >= kv_bytes, "V buffer too small");

    let range = seq_len - kv_start;
    let num_chunks = chunks_for(range);
    // Ceiling division, so the last chunk is the short one and every
    // earlier chunk is full. Flooring would leave a tail of positions
    // past the final chunk's end, silently dropping them from the softmax.
    let chunk_len = range.div_ceil(num_chunks);

    let constants = attention_function_constants(scale, ring_capacity, num_chunks);
    let constants_key = attention_constants_key(scale, ring_capacity, num_chunks);
    let partial_pipeline = context.pipeline(
        SOURCE,
        "attention_decode_partial",
        &constants,
        &constants_key,
    )?;
    pass.encode_threadgroups(
        &partial_pipeline,
        &[
            (q.0, 0, q.1),
            (k_buffer, 1, 0),
            (v_buffer, 2, 0),
            (&scratch.m, 3, 0),
            (&scratch.d, 4, 0),
            (&scratch.o, 5, 0),
        ],
        &[
            (u32_bytes(&head_dim), 6),
            (u32_bytes(&num_q_heads), 7),
            (u32_bytes(&num_kv_heads), 8),
            (u32_bytes(&seq_len), 9),
            (u32_bytes(&kv_start), 10),
            (u32_bytes(&chunk_len), 11),
            (u32_bytes(&num_chunks), 12),
            (f32_bytes(&scale), 13),
        ],
        (num_q_heads * num_chunks) as u64,
        THREADS_PER_GROUP,
    );

    let combine_pipeline = context.pipeline(
        SOURCE,
        "attention_decode_combine",
        &constants,
        &constants_key,
    )?;
    pass.encode_threadgroups(
        &combine_pipeline,
        &[
            (&scratch.m, 0, 0),
            (&scratch.d, 1, 0),
            (&scratch.o, 2, 0),
            (out.0, 3, out.1),
        ],
        &[(u32_bytes(&head_dim), 4), (u32_bytes(&num_chunks), 5)],
        num_q_heads as u64,
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// Same two-pass decode attention, but K and V are read straight out of
/// caller-owned persistent buffers (`KvCacheManager`'s per-layer K/V, in
/// linear `[seq_len, num_kv_heads, head_dim]` layout starting at offset 0)
/// instead of being re-uploaded per token. This is the shape the runner's
/// decode loop uses; `attention_decode` (slice K/V) remains for parity
/// tests.
#[allow(clippy::too_many_arguments)]
pub fn attention_decode_buffers(
    context: &mut MetalContext,
    q: &[f16],
    k_buffer: &metal::Buffer,
    v_buffer: &metal::Buffer,
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    seq_len: u32,
    scale: f32,
) -> Result<Vec<f16>, GpuError> {
    assert_eq!(q.len(), (num_q_heads * head_dim) as usize);
    assert_eq!(num_q_heads % num_kv_heads, 0);
    let kv_bytes = (seq_len * num_kv_heads * head_dim) as u64 * 2;
    assert!(k_buffer.length() >= kv_bytes, "K buffer too small");
    assert!(v_buffer.length() >= kv_bytes, "V buffer too small");

    let q_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(q));
    let m_buffer = context.new_output_buffer((num_q_heads as usize * 4) as u64);
    let d_buffer = context.new_output_buffer((num_q_heads as usize * 4) as u64);
    let o_buffer = context.new_output_buffer((num_q_heads * head_dim) as u64 * 4);

    let kv_start: u32 = 0;
    let chunk_len = seq_len;
    // Deliberately unsplit. This is the parity-test entry point, and the
    // single-chunk accumulation is the exact reference the chunked path
    // is allowed to differ from only by FP reassociation.
    let num_chunks: u32 = 1;

    let constants = attention_function_constants(scale, 0, num_chunks);
    let constants_key = attention_constants_key(scale, 0, num_chunks);
    let partial_pipeline = context.pipeline(
        SOURCE,
        "attention_decode_partial",
        &constants,
        &constants_key,
    )?;
    dispatch_one_threadgroup_per_row(
        context,
        &partial_pipeline,
        &[
            (&q_buffer, 0),
            (k_buffer, 1),
            (v_buffer, 2),
            (&m_buffer, 3),
            (&d_buffer, 4),
            (&o_buffer, 5),
        ],
        &[
            (u32_bytes(&head_dim), 6),
            (u32_bytes(&num_q_heads), 7),
            (u32_bytes(&num_kv_heads), 8),
            (u32_bytes(&seq_len), 9),
            (u32_bytes(&kv_start), 10),
            (u32_bytes(&chunk_len), 11),
            (u32_bytes(&num_chunks), 12),
            (f32_bytes(&scale), 13),
        ],
        num_q_heads as u64,
        THREADS_PER_GROUP,
    );

    let out_buffer = context.new_output_buffer((num_q_heads * head_dim) as u64 * 2);
    let combine_pipeline = context.pipeline(
        SOURCE,
        "attention_decode_combine",
        &constants,
        &constants_key,
    )?;
    dispatch_one_threadgroup_per_row(
        context,
        &combine_pipeline,
        &[
            (&m_buffer, 0),
            (&d_buffer, 1),
            (&o_buffer, 2),
            (&out_buffer, 3),
        ],
        &[(u32_bytes(&head_dim), 4), (u32_bytes(&num_chunks), 5)],
        num_q_heads as u64,
        THREADS_PER_GROUP,
    );

    Ok(read_half_buffer(
        &out_buffer,
        (num_q_heads * head_dim) as usize,
    ))
}

#[cfg(test)]
mod tests {
    use super::{chunks_for, MAX_CHUNKS, MIN_POSITIONS_PER_CHUNK};

    /// The parity tests exercise the chunked path but cannot pin this
    /// mapping: a `chunks_for` that always returned 1 would still match
    /// the CPU reference, just slowly. These are the boundaries that
    /// matter -- never 0 (the kernel takes `% num_chunks`), never above
    /// the scratch's capacity, and 1 while the range is short enough that
    /// splitting would cost more than it buys.
    #[test]
    fn chunk_count_stays_within_its_bounds() {
        for range in [1u32, 2, 15, 16, 31, 32, 255, 256, 1024, 4096, u32::MAX] {
            let chunks = chunks_for(range);
            assert!(chunks >= 1, "range {range}: {chunks}");
            assert!(chunks <= MAX_CHUNKS, "range {range}: {chunks}");
            assert!(chunks.is_power_of_two(), "range {range}: {chunks}");
            assert!(
                chunks == 1 || chunks * MIN_POSITIONS_PER_CHUNK <= range,
                "range {range} split {chunks} ways leaves chunks under {MIN_POSITIONS_PER_CHUNK}"
            );
        }
    }

    #[test]
    fn short_ranges_stay_unsplit_and_long_ones_saturate() {
        assert_eq!(chunks_for(0), 1);
        assert_eq!(chunks_for(MIN_POSITIONS_PER_CHUNK * 2 - 1), 1);
        assert_eq!(chunks_for(MIN_POSITIONS_PER_CHUNK * 2), 2);
        assert_eq!(chunks_for(MIN_POSITIONS_PER_CHUNK * MAX_CHUNKS), MAX_CHUNKS);
        assert_eq!(chunks_for(u32::MAX), MAX_CHUNKS);
    }
}
