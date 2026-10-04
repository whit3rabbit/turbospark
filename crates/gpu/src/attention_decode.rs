//! Host-side dispatch for single-token decode attention, from
//! `shaders/attention.metal` (vendored verbatim from
//! `Metal/Attention/attention.metal`). Dispatches the two-pass
//! split-KV (Flash-Decoding) kernels `attention_decode_partial` +
//! `attention_decode_combine` with `num_chunks == 1`: one pass over the
//! whole `[kv_start, seq_len)` range per Q head, no chunk-combine rescale
//! needed (a single chunk's `m_glob` equals its own `m`, so the combine
//! pass reduces to `out = o / d`, byte-identical to a fused single-pass
//! kernel — the doc comment in the vendored shader spells this out).
//! Matches `turbospark_compute::causal_attention`'s layout and (`window:
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

use std::mem::size_of;

use half::f16;
use metal::{FunctionConstantValues, MTLDataType};

use crate::bytes::{f32_bytes, half_slice_to_le_bytes, read_half_buffer, u32_bytes};
use crate::context::{dispatch_one_threadgroup_per_row, GpuError, MetalContext, PassEncoder};
use crate::prefill_scratch::BatchAttentionScratchLayout;

pub(crate) static SOURCE: &str = include_str!("shaders/attention.metal");
const THREADS_PER_GROUP: u64 = 256; // kAttnThreads.

/// Upper bound on the split-KV chunk count, and therefore on how many
/// partial slots [`AttentionScratch`] has to hold. Swift splits 16 ways
/// too. `crates/gpu/tests/attention_chunk_bench.rs` is where the number
/// comes from: at the real Gemma 4 shapes, 16 is at or within a few
/// percent of the best chunk count at every context from 256 to 4096,
/// while 32 starts regressing (more empty chunks and a wider combine for
/// no extra parallelism).
pub(crate) const MAX_CHUNKS: u32 = 16;

/// How many KV positions one threadgroup must own before splitting again.
/// Chunks shorter than this are not worth their share of the combine
/// pass, and a chunk count above the range length would dispatch
/// threadgroups whose loop never executes.
const MIN_POSITIONS_PER_CHUNK: u32 = 16;

/// The decode-attention shader family's own per-lane/threadgroup array
/// ceiling: `attention.metal`'s `kAttnMaxHeadDim`, `attention_indexed.metal`'s
/// `kIdxAttnMaxHeadDim`, `attention_tq.metal`'s `kTqAttnMaxHeadDim` and
/// `kv_quantize_tq.metal`'s `kTqMaxHeadDim` are all 512, and none of this
/// module's six dispatch entry points checked it (AGENTS.md/CLAUDE.md B3): a
/// `head_dim` above 512 overruns those fixed-size arrays silently rather
/// than being refused. Named distinctly from `vision::MAX_ATTENTION_HEAD_DIM`
/// (128, that tower's own register-tile ceiling) so the two cannot be
/// confused at a call site or collide as crate-root re-exports.
pub const MAX_DECODE_ATTENTION_HEAD_DIM: u32 = 512;

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
pub(crate) fn chunks_for(range: u32) -> u32 {
    let chunks = (range / MIN_POSITIONS_PER_CHUNK).clamp(1, MAX_CHUNKS);
    1 << chunks.ilog2()
}

/// Per-live-row causal bounds and split-KV geometry for batched attention.
///
/// The batch API uses the fixed linear FP16 K/V layout. Plans always begin at
/// position zero; ring addressing, sinks, and quantized K/V are not part of
/// this contract.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RowChunkPlan {
    pub(crate) key_start: u32,
    pub(crate) seq_len: u32,
    pub(crate) chunk_len: u32,
    pub(crate) num_chunks: u32,
}
const _: [(); 16] = [(); size_of::<RowChunkPlan>()];

/// K/V addressing mode reported by the caller for the buffers being passed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchAttentionKvLayout {
    /// Contiguous `[position, kv_head, head_dim]` rows starting at position 0.
    Linear,
    /// Any other non-linear addressing mode is unsupported.
    NonLinear,
}

/// Element format reported by the caller for the K/V buffers being passed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchAttentionKvFormat {
    /// Native FP16 K/V rows.
    Fp16,
    /// Any packed or otherwise quantized K/V representation.
    Quantized,
    /// Any non-FP16, non-quantized representation.
    Other,
}

/// Actual K/V and attention modes backing a batch-attention request.
///
/// Callers must derive this descriptor from the actual cache and model
/// attention configuration that own the supplied buffers. In particular,
/// `kv_layout`, `ring_capacity`, and `kv_format` come from the cache, while
/// `kv_start` and `sink_count` come from the active attention configuration.
/// Do not set these values from the desired dispatch mode alone. The
/// validator requires zero-start linear FP16 K/V, with no ring addressing
/// and no attention sinks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchAttentionInputContract {
    /// Cache addressing mode for the supplied K/V buffers.
    pub kv_layout: BatchAttentionKvLayout,
    /// Element format for the supplied K/V buffers.
    pub kv_format: BatchAttentionKvFormat,
    /// First key position addressable by this request.
    pub kv_start: u32,
    /// Physical token capacity when these K/V buffers use ring addressing.
    /// This must be zero for the supported linear layout.
    pub ring_capacity: usize,
    /// Number of attention sink logits supplied by the model configuration.
    pub sink_count: usize,
}

impl BatchAttentionInputContract {
    fn validate(self) -> Result<(), GpuError> {
        match self.kv_layout {
            BatchAttentionKvLayout::Linear => {}
            BatchAttentionKvLayout::NonLinear => {
                return Err(invalid_batch_input(
                    "non-linear K/V addressing is unsupported",
                ))
            }
        }
        match self.kv_format {
            BatchAttentionKvFormat::Fp16 => {}
            BatchAttentionKvFormat::Quantized => {
                return Err(invalid_batch_input("quantized K/V buffers are unsupported"))
            }
            BatchAttentionKvFormat::Other => {
                return Err(invalid_batch_input("batch K/V buffers must use FP16"))
            }
        }
        if self.ring_capacity != 0 {
            return Err(invalid_batch_input(format!(
                "ring K/V addressing with capacity {} is unsupported",
                self.ring_capacity
            )));
        }
        if self.kv_start != 0 {
            return Err(invalid_batch_input(format!(
                "batch attention requires kv_start 0, got {}",
                self.kv_start
            )));
        }
        if self.sink_count != 0 {
            return Err(invalid_batch_input(
                "attention sinks are unsupported by batch attention",
            ));
        }
        Ok(())
    }
}

/// Minimum buffer lengths for one validated batched-attention request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BatchAttentionBufferLengths {
    pub(crate) q_bytes: u64,
    pub(crate) k_bytes: u64,
    pub(crate) v_bytes: u64,
    pub(crate) output_bytes: u64,
    pub(crate) row_plan_bytes: u64,
    pub(crate) partial_bytes: u64,
}

/// GPU-private validation result consumed by the batch encoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BatchAttentionPlan {
    pub(crate) live_rows: u32,
    pub(crate) capacity: usize,
    pub(crate) num_q_heads: u32,
    pub(crate) num_kv_heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) row_plans: Vec<RowChunkPlan>,
    pub(crate) max_chunks: u32,
    pub(crate) partial_threadgroups: u64,
    pub(crate) combine_threadgroups: u64,
    pub(crate) required_buffers: BatchAttentionBufferLengths,
}

/// Construct and validate the complete row plan for a contiguous batch.
///
/// Query positions and dimensions use the same `u32` domain as the Metal
/// argument ABI. `capacity` is the row capacity represented by the caller's
/// scratch buffers. All planned ranges, byte counts, and dispatch counts are
/// checked before allocating the row-plan vector.
pub(crate) fn build_batch_attention_plan(
    input_contract: BatchAttentionInputContract,
    live_rows: usize,
    capacity: usize,
    first_query_position: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
) -> Result<BatchAttentionPlan, GpuError> {
    input_contract.validate()?;
    if live_rows == 0 || live_rows > capacity {
        return Err(invalid_batch_input(format!(
            "live row count {live_rows} must be in 1..={capacity}"
        )));
    }
    if num_q_heads == 0 || num_kv_heads == 0 {
        return Err(invalid_batch_input(
            "query and KV head counts must both be positive",
        ));
    }
    if num_q_heads % num_kv_heads != 0 {
        return Err(invalid_batch_input(format!(
            "query head count {num_q_heads} must be divisible by KV head count {num_kv_heads}"
        )));
    }
    if head_dim == 0 || head_dim > MAX_DECODE_ATTENTION_HEAD_DIM {
        return Err(invalid_batch_input(format!(
            "head dimension {head_dim} must be in 1..={MAX_DECODE_ATTENTION_HEAD_DIM}"
        )));
    }

    let last_row = u32::try_from(live_rows - 1)
        .map_err(|_| invalid_batch_input("live row count exceeds the Metal row-index range"))?;
    let last_seq_len = first_query_position
        .checked_add(last_row)
        .and_then(|position| position.checked_add(1))
        .ok_or_else(|| invalid_batch_input("query position range overflows u32"))?;
    let max_chunks = chunks_for(last_seq_len);
    if !(1..=MAX_CHUNKS).contains(&max_chunks) {
        return Err(invalid_batch_input(format!(
            "chunk count {max_chunks} exceeds 1..={MAX_CHUNKS}"
        )));
    }

    let live_rows_u32 = u32::try_from(live_rows)
        .map_err(|_| invalid_batch_input("live row count exceeds the Metal row-index range"))?;
    let q_heads = usize::try_from(num_q_heads)
        .map_err(|_| invalid_batch_input("query head count does not fit host indexing"))?;
    let kv_heads = usize::try_from(num_kv_heads)
        .map_err(|_| invalid_batch_input("KV head count does not fit host indexing"))?;
    let dim = usize::try_from(head_dim)
        .map_err(|_| invalid_batch_input("head dimension does not fit host indexing"))?;
    let max_chunks_usize = usize::try_from(max_chunks)
        .map_err(|_| invalid_batch_input("chunk count does not fit host indexing"))?;
    let max_seq_len = usize::try_from(last_seq_len)
        .map_err(|_| invalid_batch_input("sequence length does not fit host indexing"))?;

    let q_bytes = checked_batch_byte_count(&[live_rows, q_heads, dim], size_of::<f16>())?;
    let output_bytes = q_bytes;
    let k_bytes = checked_batch_byte_count(&[max_seq_len, kv_heads, dim], size_of::<f16>())?;
    let v_bytes = k_bytes;
    let row_plan_bytes = checked_batch_byte_count(&[live_rows], size_of::<RowChunkPlan>())?;
    let partial_width = dim
        .checked_add(2)
        .ok_or_else(|| invalid_batch_input("partial-state width overflows host indexing"))?;
    let partial_bytes = checked_batch_byte_count(
        &[live_rows, q_heads, max_chunks_usize, partial_width],
        size_of::<f32>(),
    )?;

    let row_tiles = checked_ceil_div_usize(live_rows, 4)?;
    let partial_threadgroups = checked_batch_product(&[row_tiles, q_heads, max_chunks_usize])?;
    let combine_threadgroups = checked_batch_product(&[live_rows, q_heads])?;

    let mut row_plans = Vec::new();
    row_plans.try_reserve_exact(live_rows).map_err(|error| {
        invalid_batch_input(format!("cannot allocate {live_rows} row plans: {error}"))
    })?;
    for row in 0..live_rows {
        let row_offset = u32::try_from(row)
            .map_err(|_| invalid_batch_input("row index exceeds the Metal row-index range"))?;
        let seq_len = first_query_position
            .checked_add(row_offset)
            .and_then(|position| position.checked_add(1))
            .ok_or_else(|| invalid_batch_input("query position range overflows u32"))?;
        let key_start = 0;
        let range = seq_len
            .checked_sub(key_start)
            .ok_or_else(|| invalid_batch_input("row plan has an inverted key range"))?;
        if range == 0 {
            return Err(invalid_batch_input(
                "row plan must include at least one key",
            ));
        }
        let num_chunks = chunks_for(range);
        if !(1..=MAX_CHUNKS).contains(&num_chunks) {
            return Err(invalid_batch_input(format!(
                "row chunk count {num_chunks} exceeds 1..={MAX_CHUNKS}"
            )));
        }
        let chunk_len = checked_ceil_div_u32(range, num_chunks)?;
        row_plans.push(RowChunkPlan {
            key_start,
            seq_len,
            chunk_len,
            num_chunks,
        });
    }
    let validated_max_chunks = validate_batch_row_plans(&row_plans)?;
    if validated_max_chunks != max_chunks {
        return Err(invalid_batch_input(
            "batch chunk count does not match the generated row plans",
        ));
    }

    Ok(BatchAttentionPlan {
        live_rows: live_rows_u32,
        capacity,
        num_q_heads,
        num_kv_heads,
        head_dim,
        row_plans,
        max_chunks,
        partial_threadgroups: u64::try_from(partial_threadgroups).map_err(|_| {
            invalid_batch_input("partial dispatch count exceeds Metal's index range")
        })?,
        combine_threadgroups: u64::try_from(combine_threadgroups).map_err(|_| {
            invalid_batch_input("combine dispatch count exceeds Metal's index range")
        })?,
        required_buffers: BatchAttentionBufferLengths {
            q_bytes,
            k_bytes,
            v_bytes,
            output_bytes,
            row_plan_bytes,
            partial_bytes,
        },
    })
}

/// Validate the private row plans before deriving dispatch bounds from them.
pub(crate) fn validate_batch_row_plans(plans: &[RowChunkPlan]) -> Result<u32, GpuError> {
    let first = plans
        .first()
        .ok_or_else(|| invalid_batch_input("batch row plan must contain at least one row"))?;
    let first_position = first
        .seq_len
        .checked_sub(1)
        .ok_or_else(|| invalid_batch_input("row plan sequence length must be positive"))?;
    let mut max_chunks = 0;

    for (row, plan) in plans.iter().enumerate() {
        let row_offset = u32::try_from(row)
            .map_err(|_| invalid_batch_input("row index exceeds the Metal row-index range"))?;
        let expected_seq_len = first_position
            .checked_add(row_offset)
            .and_then(|position| position.checked_add(1))
            .ok_or_else(|| invalid_batch_input("row plan position range overflows u32"))?;
        if plan.key_start != 0 || plan.seq_len != expected_seq_len {
            return Err(invalid_batch_input(format!(
                "row {row} must have key_start 0 and contiguous seq_len {expected_seq_len}"
            )));
        }
        let range = plan
            .seq_len
            .checked_sub(plan.key_start)
            .ok_or_else(|| invalid_batch_input(format!("row {row} has an inverted key range")))?;
        if range == 0 {
            return Err(invalid_batch_input(format!(
                "row {row} must include at least one key"
            )));
        }
        let expected_chunks = chunks_for(range);
        if !(1..=MAX_CHUNKS).contains(&plan.num_chunks) || plan.num_chunks != expected_chunks {
            return Err(invalid_batch_input(format!(
                "row {row} chunk count {} does not match expected {expected_chunks}",
                plan.num_chunks
            )));
        }
        let expected_chunk_len = checked_ceil_div_u32(range, plan.num_chunks)?;
        if plan.chunk_len != expected_chunk_len {
            return Err(invalid_batch_input(format!(
                "row {row} chunk length {} does not match expected {expected_chunk_len}",
                plan.chunk_len
            )));
        }
        max_chunks = max_chunks.max(plan.num_chunks);
    }

    Ok(max_chunks)
}

impl BatchAttentionPlan {
    /// Refuse short Metal buffers before the encoder can create either pass.
    pub(crate) fn validate_buffer_lengths(
        &self,
        actual: BatchAttentionBufferLengths,
    ) -> Result<(), GpuError> {
        for (name, actual, required) in [
            ("Q", actual.q_bytes, self.required_buffers.q_bytes),
            ("K", actual.k_bytes, self.required_buffers.k_bytes),
            ("V", actual.v_bytes, self.required_buffers.v_bytes),
            (
                "output",
                actual.output_bytes,
                self.required_buffers.output_bytes,
            ),
            (
                "row-plan",
                actual.row_plan_bytes,
                self.required_buffers.row_plan_bytes,
            ),
            (
                "partial-state",
                actual.partial_bytes,
                self.required_buffers.partial_bytes,
            ),
        ] {
            if actual < required {
                return Err(invalid_batch_input(format!(
                    "{name} buffer has {actual} bytes, requires at least {required}"
                )));
            }
        }
        Ok(())
    }
}

fn checked_batch_product(values: &[usize]) -> Result<usize, GpuError> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| invalid_batch_input("batch size or dispatch arithmetic overflows"))
    })
}

fn checked_batch_byte_count(values: &[usize], element_size: usize) -> Result<u64, GpuError> {
    let elements = checked_batch_product(values)?;
    let bytes = elements
        .checked_mul(element_size)
        .ok_or_else(|| invalid_batch_input("batch buffer byte count overflows"))?;
    u64::try_from(bytes)
        .map_err(|_| invalid_batch_input("batch buffer byte count exceeds Metal's range"))
}

fn checked_ceil_div_usize(value: usize, divisor: usize) -> Result<usize, GpuError> {
    if divisor == 0 {
        return Err(invalid_batch_input(
            "batch dispatch tile size must be positive",
        ));
    }
    let quotient = value / divisor;
    quotient
        .checked_add(usize::from(value % divisor != 0))
        .ok_or_else(|| invalid_batch_input("batch dispatch tile count overflows"))
}

fn checked_ceil_div_u32(value: u32, divisor: u32) -> Result<u32, GpuError> {
    if divisor == 0 {
        return Err(invalid_batch_input("row-plan chunk count must be positive"));
    }
    let quotient = value / divisor;
    quotient
        .checked_add(u32::from(value % divisor != 0))
        .ok_or_else(|| invalid_batch_input("row-plan chunk length overflows"))
}

fn invalid_batch_input(detail: impl Into<String>) -> GpuError {
    GpuError::InvalidInput(detail.into())
}

/// Caller-owned row-plan and FP32 partial buffers for batched attention.
///
/// The layout is retained with the buffers so the encoder uses the capacity
/// that sized the allocation rather than a caller-supplied dispatch value.
pub struct BatchAttentionScratch {
    pub row_plan: metal::Buffer,
    pub m: metal::Buffer,
    pub d: metal::Buffer,
    pub o: metal::Buffer,
    layout: BatchAttentionScratchLayout,
}

impl BatchAttentionScratch {
    /// Allocate row-plan and partial buffers from the supplied capacity layout.
    pub fn new(
        context: &MetalContext,
        layout: BatchAttentionScratchLayout,
    ) -> Result<Self, GpuError> {
        let capacity = layout.capacity();
        let num_q_heads = layout.num_q_heads();
        let head_dim = layout.head_dim();
        let max_chunks = layout.max_chunks();
        let slots = checked_batch_product(&[capacity, num_q_heads, max_chunks])?;
        let scalar_bytes = checked_batch_byte_count(&[slots], size_of::<f32>())?;
        let output_bytes = checked_batch_byte_count(&[slots, head_dim], size_of::<f32>())?;
        let partial_bytes = scalar_bytes
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(output_bytes))
            .ok_or_else(|| invalid_batch_input("batch partial-state allocation overflows"))?;
        if partial_bytes
            != u64::try_from(layout.partial_state_bytes()).map_err(|_| {
                invalid_batch_input("batch partial-state allocation exceeds Metal's range")
            })?
        {
            return Err(invalid_batch_input(
                "batch scratch layout does not match its partial-state allocation",
            ));
        }
        let row_plan_bytes = u64::try_from(layout.row_plan_bytes())
            .map_err(|_| invalid_batch_input("row-plan allocation exceeds Metal's range"))?;

        Ok(Self {
            row_plan: context.new_output_buffer(row_plan_bytes),
            m: context.new_output_buffer(scalar_bytes),
            d: context.new_output_buffer(scalar_bytes),
            o: context.new_output_buffer(output_bytes),
            layout,
        })
    }

    /// Capacity and dimensions used to allocate these buffers.
    pub fn layout(&self) -> BatchAttentionScratchLayout {
        self.layout
    }
}

fn available_buffer_bytes(buffer: &metal::Buffer, offset: u64) -> Result<u64, GpuError> {
    buffer
        .length()
        .checked_sub(offset)
        .ok_or_else(|| invalid_batch_input("buffer byte offset is past the end of the buffer"))
}

fn checked_buffer_sum(values: &[u64]) -> Result<u64, GpuError> {
    values.iter().try_fold(0u64, |sum, value| {
        sum.checked_add(*value)
            .ok_or_else(|| invalid_batch_input("batch buffer length sum overflows"))
    })
}

fn require_buffer_length(name: &str, actual: u64, required: u64) -> Result<(), GpuError> {
    if actual < required {
        return Err(invalid_batch_input(format!(
            "{name} buffer has {actual} bytes, requires at least {required}"
        )));
    }
    Ok(())
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
pub(crate) fn attention_function_constants(
    scale: f32,
    ring_capacity: u32,
    num_chunks: u32,
    has_sinks: bool,
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
    // ROADMAP M5. `false` makes the combine kernel's `sinks` argument not
    // exist, so the four sink-free families bind exactly what they always
    // did and compile to the same code.
    values.set_constant_value_at_index((&has_sinks as *const bool).cast(), MTLDataType::Bool, 70);
    values
}

pub(crate) fn attention_constants_key(
    scale: f32,
    ring_capacity: u32,
    num_chunks: u32,
    has_sinks: bool,
) -> [u8; 13] {
    let mut key = [0u8; 13];
    key[..4].copy_from_slice(&scale.to_le_bytes());
    key[4..8].copy_from_slice(&ring_capacity.to_le_bytes());
    key[8..12].copy_from_slice(&num_chunks.to_le_bytes());
    // The fourth axis, and it has to be here for the reason the doc above
    // gives about the other three: without it a sink dispatch and a
    // sink-free one at the same (scale, ring, chunks) share a cache entry,
    // and whichever compiled first wins. That failure is silent both ways --
    // sinks ignored, or a kernel reading an argument nobody bound.
    key[12] = u8::from(has_sinks);
    key
}

/// `Q: [num_q_heads, head_dim]`, `K`/`V: [seq_len, num_kv_heads, head_dim]`
/// (same layout `turbospark_compute::causal_attention` uses). Returns
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
    assert!(head_dim <= MAX_DECODE_ATTENTION_HEAD_DIM);

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

/// Encode both batch-only split-KV attention passes for row-major Q/output.
///
/// `input_contract` must describe the actual cache/configuration that owns
/// K/V. The GPU crate builds contiguous row plans from `first_query_position`
/// and `live_rows`, validates every buffer before compiling or encoding a
/// pass, and specializes both batch pipelines by scale and maximum row chunk
/// count. Existing one-row callers continue to use
/// [`encode_attention_decode`].
#[allow(clippy::too_many_arguments)]
pub fn encode_attention_decode_batch(
    context: &mut MetalContext,
    pass: &PassEncoder,
    q: (&metal::Buffer, u64),
    k_buffer: &metal::Buffer,
    v_buffer: &metal::Buffer,
    scratch: &BatchAttentionScratch,
    out: (&metal::Buffer, u64),
    input_contract: BatchAttentionInputContract,
    first_query_position: u32,
    live_rows: usize,
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    scale: f32,
) -> Result<(), GpuError> {
    let layout = scratch.layout;
    if layout.num_q_heads() != num_q_heads as usize || layout.head_dim() != head_dim as usize {
        return Err(invalid_batch_input(
            "batch scratch layout dimensions do not match attention dimensions",
        ));
    }

    let plan = build_batch_attention_plan(
        input_contract,
        live_rows,
        layout.capacity(),
        first_query_position,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )?;

    let q_bytes = available_buffer_bytes(q.0, q.1)?;
    let output_bytes = available_buffer_bytes(out.0, out.1)?;
    let partial_bytes =
        checked_buffer_sum(&[scratch.m.length(), scratch.d.length(), scratch.o.length()])?;
    plan.validate_buffer_lengths(BatchAttentionBufferLengths {
        q_bytes,
        k_bytes: k_buffer.length(),
        v_bytes: v_buffer.length(),
        output_bytes,
        row_plan_bytes: scratch.row_plan.length(),
        partial_bytes,
    })?;

    // Enforce the full layout allocation, not just the live batch prefix.
    // The runtime and memory oracle both size these buffers from this layout.
    require_buffer_length(
        "Q",
        q_bytes,
        u64::try_from(layout.q_buffer_bytes())
            .map_err(|_| invalid_batch_input("Q buffer requirement exceeds Metal's range"))?,
    )?;
    require_buffer_length(
        "output",
        output_bytes,
        u64::try_from(layout.output_buffer_bytes())
            .map_err(|_| invalid_batch_input("output buffer requirement exceeds Metal's range"))?,
    )?;
    require_buffer_length(
        "row-plan",
        scratch.row_plan.length(),
        u64::try_from(layout.row_plan_bytes())
            .map_err(|_| invalid_batch_input("row-plan requirement exceeds Metal's range"))?,
    )?;
    let capacity_slots =
        checked_batch_product(&[layout.capacity(), layout.num_q_heads(), layout.max_chunks()])?;
    let scalar_plane_bytes = checked_batch_byte_count(&[capacity_slots], size_of::<f32>())?;
    let output_plane_bytes =
        checked_batch_byte_count(&[capacity_slots, layout.head_dim()], size_of::<f32>())?;
    require_buffer_length("partial max", scratch.m.length(), scalar_plane_bytes)?;
    require_buffer_length(
        "partial denominator",
        scratch.d.length(),
        scalar_plane_bytes,
    )?;
    require_buffer_length("partial output", scratch.o.length(), output_plane_bytes)?;
    if scratch.row_plan.contents().is_null() {
        return Err(invalid_batch_input(
            "row-plan buffer must be CPU-writable shared memory",
        ));
    }

    let constants = attention_function_constants(scale, 0, plan.max_chunks, false);
    let constants_key = attention_constants_key(scale, 0, plan.max_chunks, false);
    // Compile both functions before the first dispatch so a missing combine
    // entry point cannot leave a partial-only pass in the caller's encoder.
    let partial_pipeline = context.pipeline(
        SOURCE,
        "attention_decode_batch_partial",
        &constants,
        &constants_key,
    )?;
    let combine_pipeline = context.pipeline(
        SOURCE,
        "attention_decode_batch_combine",
        &constants,
        &constants_key,
    )?;

    // RowChunkPlan is repr(C), exactly four u32 fields, and Metal buffers are
    // shared CPU/GPU storage. The capacity tail is intentionally untouched.
    unsafe {
        std::ptr::copy_nonoverlapping(
            plan.row_plans.as_ptr(),
            scratch.row_plan.contents().cast::<RowChunkPlan>(),
            plan.row_plans.len(),
        );
    }

    pass.encode_threadgroups(
        &partial_pipeline,
        &[
            (q.0, 0, q.1),
            (k_buffer, 1, 0),
            (v_buffer, 2, 0),
            (&scratch.m, 3, 0),
            (&scratch.d, 4, 0),
            (&scratch.o, 5, 0),
            (&scratch.row_plan, 9, 0),
        ],
        &[
            (u32_bytes(&head_dim), 6),
            (u32_bytes(&num_q_heads), 7),
            (u32_bytes(&num_kv_heads), 8),
            (u32_bytes(&plan.live_rows), 10),
            (f32_bytes(&scale), 11),
        ],
        plan.partial_threadgroups,
        THREADS_PER_GROUP,
    );

    pass.encode_threadgroups(
        &combine_pipeline,
        &[
            (&scratch.m, 0, 0),
            (&scratch.d, 1, 0),
            (&scratch.o, 2, 0),
            (out.0, 3, out.1),
        ],
        &[
            (u32_bytes(&head_dim), 4),
            (u32_bytes(&num_q_heads), 5),
            (u32_bytes(&plan.max_chunks), 6),
            (u32_bytes(&plan.live_rows), 7),
        ],
        plan.combine_threadgroups,
        THREADS_PER_GROUP,
    );
    Ok(())
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
    // ROADMAP M5's attention sinks: one BF16 logit per q head, added to the
    // softmax denominator. `None` for every family but `gpt-oss`, and the
    // combine kernel then has no such argument at all.
    sinks: Option<(&metal::Buffer, u64)>,
) -> Result<(), GpuError> {
    assert_eq!(num_q_heads % num_kv_heads, 0);
    assert!(head_dim <= MAX_DECODE_ATTENTION_HEAD_DIM);
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
    let kv_bytes = stored_tokens as u64 * num_kv_heads as u64 * head_dim as u64 * 2;
    assert!(k_buffer.length() >= kv_bytes, "K buffer too small");
    assert!(v_buffer.length() >= kv_bytes, "V buffer too small");

    let range = seq_len - kv_start;
    let num_chunks = chunks_for(range);
    // Ceiling division, so the last chunk is the short one and every
    // earlier chunk is full. Flooring would leave a tail of positions
    // past the final chunk's end, silently dropping them from the softmax.
    let chunk_len = range.div_ceil(num_chunks);

    let has_sinks = sinks.is_some();
    let constants = attention_function_constants(scale, ring_capacity, num_chunks, has_sinks);
    let constants_key = attention_constants_key(scale, ring_capacity, num_chunks, has_sinks);
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
    let mut combine_buffers: Vec<(&metal::Buffer, u64, u64)> = vec![
        (&scratch.m, 0, 0),
        (&scratch.d, 1, 0),
        (&scratch.o, 2, 0),
        (out.0, 3, out.1),
    ];
    if let Some((buf, off)) = sinks {
        combine_buffers.push((buf, 6, off));
    }
    pass.encode_threadgroups(
        &combine_pipeline,
        &combine_buffers,
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
pub(crate) fn attention_decode_buffers(
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
    let kv_bytes = seq_len as u64 * num_kv_heads as u64 * head_dim as u64 * 2;
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

    let constants = attention_function_constants(scale, 0, num_chunks, false);
    let constants_key = attention_constants_key(scale, 0, num_chunks, false);
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
#[path = "attention_decode_tests.rs"]
mod tests;
