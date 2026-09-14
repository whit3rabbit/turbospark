//! Host-side dispatch for `shaders/qsa_indexer.metal` (`qwen4_exp`'s QSA
//! block indexer: pooling and scoring; PORT-LOCAL, `docs/QWEN4_PHASE0.md`
//! section 5). Matched to `turbospark_compute::qsa_indexer`'s CPU
//! reference. Block SELECTION (top-k) stays host-side, matching this
//! port's existing MoE router precedent -- no kernel for it here.

use metal::FunctionConstantValues;

use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext, PassEncoder};

const SOURCE: &str = include_str!("shaders/qsa_indexer.metal");
const THREADS_PER_GROUP: u64 = 256;

fn grid_for(count: u32) -> u64 {
    // AGENTS.md/CLAUDE.md S12: floored at 1, matching rope.rs's own
    // convention -- a `count == 0` caller still dispatches one
    // (harmless, bounds-checked-empty) threadgroup rather than zero.
    (count as u64).div_ceil(THREADS_PER_GROUP).max(1) * THREADS_PER_GROUP
}

/// `pooled[b*D+d] = mean over t in [0, compress_ratio) of keys[(b*compress_ratio+t)*D+d]`.
///
/// `keys` is `[visible * head_dim]` RAW (un-normed, un-roped), oldest token
/// first; `pooled` is `[num_blocks * head_dim]`. Only complete blocks are
/// read -- `num_blocks` must be `visible / compress_ratio` rounded down,
/// and the caller owns the ragged tail (never pooled, always selected).
pub fn encode_qsa_pool_blocks_mean(
    context: &mut MetalContext,
    pass: &PassEncoder,
    keys: (&metal::Buffer, u64),
    pooled: (&metal::Buffer, u64),
    compress_ratio: u32,
    head_dim: u32,
    num_blocks: u32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "qsa_pool_blocks_mean_fp16",
        &FunctionConstantValues::new(),
        b"",
    )?;
    let total = num_blocks * head_dim;
    pass.encode_threads_3d(
        &pipeline,
        &[(keys.0, 0, keys.1), (pooled.0, 1, pooled.1)],
        &[
            (u32_bytes(&compress_ratio), 2),
            (u32_bytes(&head_dim), 3),
            (u32_bytes(&num_blocks), 4),
        ],
        (grid_for(total), 1, 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}

/// `scores[b] = sum_h(relu(sum_d(q[h*D+d] * pooled[b*D+d]))) / sqrt(D)`.
///
/// `q` is `[num_heads * head_dim]`, ALREADY normed and roped at the
/// query's own current position; `pooled` is `[num_blocks * head_dim]`,
/// ALREADY normed and roped at each block's own first-token position
/// (`rope_neox_subdim`, one call per block since blocks do not share a
/// position -- see the CPU reference's module doc). `scores` is
/// `[num_blocks]` FP32, one threadgroup per block.
#[allow(clippy::too_many_arguments)]
pub fn encode_qsa_score_blocks(
    context: &mut MetalContext,
    pass: &PassEncoder,
    q: (&metal::Buffer, u64),
    pooled: (&metal::Buffer, u64),
    scores: (&metal::Buffer, u64),
    num_heads: u32,
    head_dim: u32,
    num_blocks: u32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "qsa_score_blocks_fp16",
        &FunctionConstantValues::new(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[
            (q.0, 0, q.1),
            (pooled.0, 1, pooled.1),
            (scores.0, 2, scores.1),
        ],
        &[(u32_bytes(&num_heads), 3), (u32_bytes(&head_dim), 4)],
        num_blocks as u64,
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// Pools, norms and RoPE's the run of newly-completed blocks
/// `[first_new_block, first_new_block + new_block_count)`, writing them
/// into `pooled` at those blocks' own rows -- the whole of "keep a QSA
/// layer's pooled-block cache up to date" as ONE composed call, matched to
/// `turbospark_compute::qsa_indexer`'s `the_full_indexer_chain_composes`
/// reference chain and this crate's own module doc on the resolved
/// norm/RoPE convention (`rms_norm_centered` per block, `rope_neox_subdim`
/// at `rotary_dim`/`theta`).
///
/// Three dispatches, not one, and the split is not incidental:
/// 1. [`encode_qsa_pool_blocks_mean`] once for every new block (pooling has
///    no position dependence, so it batches freely).
/// 2. The existing per-head CENTERED norm
///    (`crate::rms_norm::encode_rms_norm_bf16w_perhead_centered`) once for
///    every new block, treating each pooled block as one "head" -- the
///    kernel's own indexing (`x + head * headDim`) is exactly "N
///    independent `head_dim`-wide reductions sharing one weight," which is
///    what pooled-block norming is, whatever the kernel's name says.
///    `k_norm_weight` must be the checkpoint's `k_layernorm` weight, BF16
///    (this port's usual unquantized-text-tensor convention -- NOT FP16
///    the way the pooled rows themselves are).
/// 3. [`crate::rope::encode_rope_neox_subdim`], but this one CANNOT batch:
///    its `position` argument is one scalar for the whole dispatch, and
///    each new block's own first-token absolute position differs. One
///    dispatch per block is the real, measured shape this composition
///    costs -- see this crate's own `CLAUDE.md` for why that cannot be
///    designed around without a kernel able to read a PER-ROW position
///    table, which does not exist for this convention anywhere in this
///    port yet.
///
/// `raw_keys` is the layer's full raw key history
/// (`QsaIndexerCacheManager::raw_keys_view`); only the token range the new
/// blocks cover is read. `pooled` is the layer's pooled-block buffer
/// (`QsaIndexerCacheManager::pooled_blocks_buffer`); rows before
/// `first_new_block` are untouched, matching the incremental design's
/// promise that a pooled block is never recomputed. `key_start_position` is
/// the ABSOLUTE position of `raw_keys`' row 0 (0 unless a future caller
/// trims the front of the raw history), so block `b`'s own RoPE position is
/// `key_start_position + b * compress_ratio`. The caller still owns
/// advancing `QsaIndexerCacheManager`'s own cursor
/// (`advance_pooled_blocks`) once this dispatch's command buffer has been
/// committed -- this function only encodes work, it does not touch host
/// state.
#[allow(clippy::too_many_arguments)]
pub fn encode_qsa_advance_blocks(
    context: &mut MetalContext,
    pass: &PassEncoder,
    raw_keys: (&metal::Buffer, u64),
    pooled: (&metal::Buffer, u64),
    k_norm_weight: (&metal::Buffer, u64),
    compress_ratio: u32,
    head_dim: u32,
    rotary_dim: u32,
    theta: f32,
    eps: f32,
    first_new_block: u32,
    new_block_count: u32,
    key_start_position: u32,
) -> Result<(), GpuError> {
    if new_block_count == 0 {
        return Ok(());
    }
    let row_bytes = head_dim as u64 * 2; // FP16
    let raw_offset = raw_keys.1 + (first_new_block * compress_ratio) as u64 * row_bytes;
    let pooled_offset = pooled.1 + first_new_block as u64 * row_bytes;

    encode_qsa_pool_blocks_mean(
        context,
        pass,
        (raw_keys.0, raw_offset),
        (pooled.0, pooled_offset),
        compress_ratio,
        head_dim,
        new_block_count,
    )?;

    crate::rms_norm::encode_rms_norm_bf16w_perhead_centered(
        context,
        pass,
        (pooled.0, pooled_offset),
        k_norm_weight,
        (pooled.0, pooled_offset),
        new_block_count,
        head_dim,
        eps,
    )?;

    for i in 0..new_block_count {
        let block_index = first_new_block + i;
        let block_position = key_start_position + block_index * compress_ratio;
        let block_offset = pooled.1 + block_index as u64 * row_bytes;
        crate::rope::encode_rope_neox_subdim(
            context,
            pass,
            (pooled.0, block_offset),
            block_position,
            1,
            head_dim,
            rotary_dim,
            theta,
        )?;
    }
    Ok(())
}
