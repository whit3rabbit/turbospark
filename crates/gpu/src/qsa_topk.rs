//! Host dispatch for the GPU QSA block selector.
//!
//! Selection is kept on the GPU after block scoring. The rank kernel matches
//! `compute::select_blocks`: descending score, then ascending block index for
//! ties. The position kernel compacts selected blocks in ascending order and
//! appends the unpooled ragged tail.

use metal::FunctionConstantValues;

use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext, PassEncoder};

const SOURCE: &str = include_str!("shaders/qsa_topk.metal");
const THREADS_PER_GROUP: u64 = 256;

/// Scores complete blocks and writes the ascending selected token positions.
///
/// `scores` contains one FP32 score per complete block. `ranks` is scratch
/// with one `u32` per complete block. `positions` receives the selected
/// complete blocks followed by the always-selected ragged tail. `count` is
/// written by the GPU because the following attention dispatch consumes it
/// without reading it back to the host. `status` is sticky until the caller
/// clears it and becomes non-zero if any score is NaN. Infinite scores are
/// valid and follow the CPU oracle's ordering.
#[allow(clippy::too_many_arguments)]
pub fn encode_qsa_topk_positions(
    context: &mut MetalContext,
    pass: &PassEncoder,
    scores: &metal::Buffer,
    ranks: &metal::Buffer,
    positions: &metal::Buffer,
    count: &metal::Buffer,
    status: &metal::Buffer,
    num_blocks: u32,
    visible: u32,
    block_topk: u32,
    compress_ratio: u32,
) -> Result<(), GpuError> {
    assert!(compress_ratio > 0, "QSA compress_ratio must be positive");
    assert_eq!(
        num_blocks,
        visible / compress_ratio,
        "QSA complete block count must match visible / compress_ratio"
    );
    assert!(
        scores.length() >= num_blocks as u64 * 4,
        "QSA score buffer too small"
    );
    assert!(
        ranks.length() >= num_blocks as u64 * 4,
        "QSA rank buffer too small"
    );
    assert!(count.length() >= 4, "QSA count buffer too small");
    assert!(status.length() >= 4, "QSA status buffer too small");
    let max_selected = block_topk.min(num_blocks);
    let max_positions = max_selected
        .checked_mul(compress_ratio)
        .and_then(|n| n.checked_add(visible % compress_ratio))
        .expect("QSA selected-position count overflows u32");
    assert!(
        positions.length() >= max_positions as u64 * 4,
        "QSA selected-position buffer too small"
    );

    if num_blocks > 0 {
        // Score generation is the preceding dispatch in this same pass.
        pass.memory_barrier_with_buffers(&[scores]);
        let rank_pipeline = context.pipeline(
            SOURCE,
            "qsa_topk_block_ranks_fp32",
            &FunctionConstantValues::new(),
            b"",
        )?;
        pass.encode_threadgroups(
            &rank_pipeline,
            &[(scores, 0, 0), (ranks, 1, 0), (status, 2, 0)],
            &[(u32_bytes(&num_blocks), 3)],
            num_blocks as u64,
            THREADS_PER_GROUP,
        );
        pass.memory_barrier_with_buffers(&[ranks]);
    }

    let positions_pipeline = context.pipeline(
        SOURCE,
        "qsa_topk_write_positions",
        &FunctionConstantValues::new(),
        b"",
    )?;
    pass.encode_threadgroups(
        &positions_pipeline,
        &[(ranks, 0, 0), (positions, 1, 0), (count, 2, 0)],
        &[
            (u32_bytes(&num_blocks), 3),
            (u32_bytes(&visible), 4),
            (u32_bytes(&block_topk), 5),
            (u32_bytes(&compress_ratio), 6),
        ],
        1,
        1,
    );
    pass.memory_barrier_with_buffers(&[positions, count]);
    Ok(())
}
