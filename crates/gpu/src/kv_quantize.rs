//! Host-side dispatch for `shaders/kv_quantize_tq.metal`: quantizes M rows
//! of FP16 K or V into TurboQuant's packed format. One dispatch handles
//! both the K and V side of a commit -- callers make two calls, one per
//! side, each with that side's own [`crate::TqSideTables`].

use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::kv_quant_tables::TqSideTables;

const SOURCE: &str = include_str!("shaders/kv_quantize_tq.metal");
const THREADS_PER_GROUP: u64 = 256;

/// Quantizes `rows` rows of `num_kv_heads` heads each from `src` (FP16,
/// `head_dim` elements per head, `source_row_stride_elems` elements
/// between one row's start and the next -- i.e. `num_kv_heads * head_dim`
/// for a tightly packed staging buffer) into `dst`, one packed row per
/// (row, kv_head): `[1 + packed_words]` `u32` words, word 0 the row's
/// `f32` norm bit pattern.
///
/// `dst` must be laid out `[rows, num_kv_heads, 1 + packed_words]` and the
/// caller writes it at the RIGHT physical offset for the cache's own
/// addressing (a full-attention layer's capacity equals `max_context`, so
/// `validate_range` on the caller's `KvCacheManager` already forbids a
/// wraparound write here — this dispatch itself has no notion of a ring).
#[allow(clippy::too_many_arguments)]
pub fn encode_kv_quantize_tq(
    context: &mut MetalContext,
    pass: &PassEncoder,
    src: (&metal::Buffer, u64),
    source_row_stride_elems: u32,
    dst: (&metal::Buffer, u64),
    tables: &TqSideTables,
    head_dim: u32,
    num_kv_heads: u32,
    rows: u32,
) -> Result<(), GpuError> {
    let packed_words = model_io::tq_packed_words(head_dim as i64, tables.bits) as u32;
    let levels_minus_one = tables.levels.saturating_sub(1);
    let bits = tables.bits as u32;
    let pipeline = context.pipeline(
        SOURCE,
        "kv_quantize_tq",
        &metal::FunctionConstantValues::new(),
        &[],
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[
            (src.0, 0, src.1),
            (dst.0, 1, dst.1),
            (&tables.signs, 2, 0),
            (&tables.midpoints, 3, 0),
        ],
        &[
            (u32_bytes(&head_dim), 4),
            (u32_bytes(&num_kv_heads), 5),
            (u32_bytes(&bits), 6),
            (u32_bytes(&packed_words), 7),
            (u32_bytes(&source_row_stride_elems), 8),
            (u32_bytes(&levels_minus_one), 9),
        ],
        (rows * num_kv_heads) as u64,
        THREADS_PER_GROUP,
    );
    Ok(())
}
