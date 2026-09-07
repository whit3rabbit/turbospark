//! Host-side dispatch for `shaders/attention_tq.metal`: decode attention
//! over TurboQuant-quantized K/V rows. Mirrors `attention_decode.rs`'s
//! two-pass split-KV structure (and `attention_indexed.rs`'s sparse
//! variant) exactly, substituting the packed-row kernels and adding the
//! [`crate::KvQuantTables`] buffers every dispatch needs.
//!
//! Ring addressing is never wired here: TurboQuant only quantizes
//! full-attention layers, so callers must not reach these entry points for
//! a sliding-window layer (`crate::kv_cache::KvCacheManager::layer_quant`
//! is `None` there by construction).

use metal::{FunctionConstantValues, MTLDataType};

use crate::bytes::{f32_bytes, u32_bytes};
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::kv_quant_tables::KvQuantTables;

use crate::attention_decode::chunks_for;

const SOURCE: &str = include_str!("shaders/attention_tq.metal");
const THREADS_PER_GROUP: u64 = 256;

fn combine_function_constants(has_sinks: bool) -> FunctionConstantValues {
    let values = FunctionConstantValues::new();
    values.set_constant_value_at_index((&has_sinks as *const bool).cast(), MTLDataType::Bool, 80);
    values
}

fn combine_constants_key(has_sinks: bool) -> [u8; 1] {
    [u8::from(has_sinks)]
}

/// Caller-owned scratch for the two-pass TQ decode attention. Distinct
/// from [`crate::AttentionScratch`] only in name -- the layout (partials
/// per Q head per chunk) is identical, but a quantized layer's dispatch
/// never mixes buffers with a dense one, so keeping the types separate
/// stops a caller from accidentally handing one path's scratch to the
/// other's pipeline.
pub struct TqAttentionScratch {
    pub m: metal::Buffer,
    pub d: metal::Buffer,
    pub o: metal::Buffer,
}

impl TqAttentionScratch {
    pub fn new(context: &MetalContext, num_q_heads: u32, head_dim: u32) -> Self {
        const MAX_CHUNKS: u32 = 16;
        let slots = (num_q_heads * MAX_CHUNKS) as u64;
        Self {
            m: context.new_output_buffer(slots * 4),
            d: context.new_output_buffer(slots * 4),
            o: context.new_output_buffer(slots * head_dim as u64 * 4),
        }
    }
}

/// Dense two-pass decode attention over TurboQuant-quantized K/V.
///
/// `k_buffer`/`v_buffer` hold packed rows (`crate::kv_quantize::encode_kv_quantize_tq`'s
/// output layout: `[stored_tokens, num_kv_heads, 1 + packed_words]` `u32`,
/// LINEAR addressing only -- no ring). `tables` supplies the codebooks,
/// midpoints-free at decode time, and both sign vectors.
#[allow(clippy::too_many_arguments)]
pub fn encode_attention_decode_tq(
    context: &mut MetalContext,
    pass: &PassEncoder,
    q: (&metal::Buffer, u64),
    k_buffer: &metal::Buffer,
    v_buffer: &metal::Buffer,
    scratch: &TqAttentionScratch,
    out: (&metal::Buffer, u64),
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    seq_len: u32,
    kv_start: u32,
    scale: f32,
    tables: &KvQuantTables,
    sinks: Option<(&metal::Buffer, u64)>,
) -> Result<(), GpuError> {
    assert_eq!(num_q_heads % num_kv_heads, 0);
    assert!(kv_start < seq_len);
    assert_eq!(tables.full_head_dim as u32, head_dim);

    let range = seq_len - kv_start;
    let num_chunks = chunks_for(range);
    let chunk_len = range.div_ceil(num_chunks);

    let k_bits = tables.k.bits as u32;
    let v_bits = tables.v.bits as u32;
    let k_packed_words = model_io::tq_packed_words(head_dim as i64, tables.k.bits) as u32;
    let v_packed_words = model_io::tq_packed_words(head_dim as i64, tables.v.bits) as u32;

    let has_sinks = sinks.is_some();
    let partial_pipeline = context.pipeline(
        SOURCE,
        "attention_decode_partial_tq",
        &FunctionConstantValues::new(),
        &[],
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
            (&tables.k.signs, 16, 0),
            (&tables.k.codebook, 17, 0),
            (&tables.v.codebook, 18, 0),
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
            (u32_bytes(&k_bits), 14),
            (u32_bytes(&v_bits), 15),
            (u32_bytes(&k_packed_words), 19),
            (u32_bytes(&v_packed_words), 20),
        ],
        (num_q_heads * num_chunks) as u64,
        THREADS_PER_GROUP,
    );

    encode_combine_tq(
        context,
        pass,
        scratch,
        out,
        head_dim,
        num_q_heads,
        num_chunks,
        tables,
        sinks,
        has_sinks,
    )
}

/// Sparse pass 1 (`qwen4_exp` QSA) over an explicit position list, plus the
/// shared combine pass. Mirrors [`crate::encode_attention_decode_indexed`]'s
/// relationship to [`encode_attention_decode_tq`] exactly.
#[allow(clippy::too_many_arguments)]
pub fn encode_attention_decode_indexed_tq(
    context: &mut MetalContext,
    pass: &PassEncoder,
    q: (&metal::Buffer, u64),
    k_buffer: &metal::Buffer,
    v_buffer: &metal::Buffer,
    positions: (&metal::Buffer, u64),
    n_sel: u32,
    scratch: &TqAttentionScratch,
    out: (&metal::Buffer, u64),
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    scale: f32,
    tables: &KvQuantTables,
) -> Result<(), GpuError> {
    assert_eq!(num_q_heads % num_kv_heads, 0);
    assert_eq!(tables.full_head_dim as u32, head_dim);

    let num_chunks = chunks_for(n_sel.max(1));
    let chunk_len = n_sel.max(1).div_ceil(num_chunks);

    let k_bits = tables.k.bits as u32;
    let v_bits = tables.v.bits as u32;
    let k_packed_words = model_io::tq_packed_words(head_dim as i64, tables.k.bits) as u32;
    let v_packed_words = model_io::tq_packed_words(head_dim as i64, tables.v.bits) as u32;

    let partial_pipeline = context.pipeline(
        SOURCE,
        "attention_decode_indexed_partial_tq",
        &FunctionConstantValues::new(),
        &[],
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
            (positions.0, 9, positions.1),
            (&tables.k.signs, 16, 0),
            (&tables.k.codebook, 17, 0),
            (&tables.v.codebook, 18, 0),
        ],
        &[
            (u32_bytes(&head_dim), 6),
            (u32_bytes(&num_q_heads), 7),
            (u32_bytes(&num_kv_heads), 8),
            (u32_bytes(&n_sel), 10),
            (u32_bytes(&chunk_len), 11),
            (u32_bytes(&num_chunks), 12),
            (f32_bytes(&scale), 13),
            (u32_bytes(&k_bits), 14),
            (u32_bytes(&v_bits), 15),
            (u32_bytes(&k_packed_words), 19),
            (u32_bytes(&v_packed_words), 20),
        ],
        (num_q_heads * num_chunks) as u64,
        THREADS_PER_GROUP,
    );

    encode_combine_tq(
        context,
        pass,
        scratch,
        out,
        head_dim,
        num_q_heads,
        num_chunks,
        tables,
        None,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_combine_tq(
    context: &mut MetalContext,
    pass: &PassEncoder,
    scratch: &TqAttentionScratch,
    out: (&metal::Buffer, u64),
    head_dim: u32,
    num_q_heads: u32,
    num_chunks: u32,
    tables: &KvQuantTables,
    sinks: Option<(&metal::Buffer, u64)>,
    has_sinks: bool,
) -> Result<(), GpuError> {
    let constants = combine_function_constants(has_sinks);
    let constants_key = combine_constants_key(has_sinks);
    let combine_pipeline = context.pipeline(
        SOURCE,
        "attention_decode_combine_tq",
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
    combine_buffers.push((&tables.v.signs, 7, 0));
    pass.encode_threadgroups(
        &combine_pipeline,
        &combine_buffers,
        &[(u32_bytes(&head_dim), 4), (u32_bytes(&num_chunks), 5)],
        num_q_heads as u64,
        THREADS_PER_GROUP,
    );
    Ok(())
}
