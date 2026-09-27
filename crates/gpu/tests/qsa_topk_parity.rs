#![cfg(target_os = "macos")]
//! GPU QSA block selection must match the CPU oracle's ordering and mask,
//! including ties, the ragged tail, infinity ordering, and NaN refusal.

use turbospark_gpu::{
    encode_qsa_topk_positions, read_buffer_bytes, write_buffer_bytes, MetalContext,
};

fn read_u32s(buffer: &metal::Buffer, count: usize) -> Vec<u32> {
    read_buffer_bytes(buffer, 0, count * 4)
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn run_selector(
    context: &mut MetalContext,
    scores: &[f32],
    visible: usize,
    block_topk: usize,
    compress_ratio: usize,
) -> (Vec<u32>, u32) {
    let num_blocks = visible / compress_ratio;
    assert_eq!(scores.len(), num_blocks);
    let tail = visible % compress_ratio;
    let max_positions = block_topk.min(num_blocks) * compress_ratio + tail;
    let scores_buffer = context.new_buffer_with_data(scores);
    let ranks_buffer = context.new_output_buffer((num_blocks.max(1) * 4) as u64);
    let positions_buffer = context.new_output_buffer((max_positions.max(1) * 4) as u64);
    let count_buffer = context.new_output_buffer(4);
    let status_buffer = context.new_output_buffer(4);
    write_buffer_bytes(&status_buffer, 0, &[0; 4]);

    let pass = context.begin_pass();
    encode_qsa_topk_positions(
        context,
        &pass,
        &scores_buffer,
        &ranks_buffer,
        &positions_buffer,
        &count_buffer,
        &status_buffer,
        num_blocks as u32,
        visible as u32,
        block_topk as u32,
        compress_ratio as u32,
    )
    .expect("QSA selector dispatch");
    pass.commit_and_wait();

    let count = read_u32s(&count_buffer, 1)[0];
    assert_eq!(read_u32s(&status_buffer, 1)[0], 0);
    let positions = read_u32s(&positions_buffer, count as usize);
    (positions, count)
}

fn cpu_positions(scores: &[f32], visible: usize, block_topk: usize, compress: usize) -> Vec<u32> {
    turbospark_compute::select_blocks(scores, visible, compress, block_topk)
        .into_iter()
        .enumerate()
        .filter_map(|(position, selected)| selected.then_some(position as u32))
        .collect()
}

#[test]
fn gpu_selector_matches_cpu_order_ties_and_ragged_tail() {
    let mut context = MetalContext::new().expect("Metal device");

    // Blocks 1, 2 and 4 tie at the top-k boundary. Lower indices win, then
    // the selected token positions must be emitted in ascending order.
    let scores = [1.0f32, 5.0, 5.0, 0.0, 5.0, 3.0];
    let expected = cpu_positions(&scores, 26, 2, 4);
    let (actual, count) = run_selector(&mut context, &scores, 26, 2, 4);
    assert_eq!(count as usize, expected.len());
    assert_eq!(actual, expected);
    assert_eq!(actual, [4, 5, 6, 7, 8, 9, 10, 11, 24, 25]);

    // Every score ties, so the first three blocks must win.
    let scores = [1.0f32; 9];
    let expected = cpu_positions(&scores, 36, 3, 4);
    let (actual, count) = run_selector(&mut context, &scores, 36, 3, 4);
    assert_eq!(count as usize, expected.len());
    assert_eq!(actual, expected);
    assert_eq!(actual, (0..12).collect::<Vec<_>>());

    // Selecting at least every complete block preserves the identity list
    // and still appends the two-token tail.
    let scores = [0.0f32, 2.0, 1.0, 3.0];
    let expected = cpu_positions(&scores, 18, 8, 4);
    let (actual, count) = run_selector(&mut context, &scores, 18, 8, 4);
    assert_eq!(count as usize, expected.len());
    assert_eq!(actual, expected);
    assert_eq!(actual, (0..18).collect::<Vec<_>>());

    // Infinities are ordered scores, not invalid inputs to the CPU oracle.
    let scores = [f32::NEG_INFINITY, f32::INFINITY, 7.0];
    let expected = cpu_positions(&scores, 12, 1, 4);
    let (actual, count) = run_selector(&mut context, &scores, 12, 1, 4);
    assert_eq!(count as usize, expected.len());
    assert_eq!(actual, expected);
    assert_eq!(actual, (4..8).collect::<Vec<_>>());
}

#[test]
fn nan_score_sets_status_and_keeps_attention_count_safe() {
    let mut context = MetalContext::new().expect("Metal device");
    let scores = [2.0f32, f32::NAN, 3.0];
    let num_blocks = scores.len();
    let visible = 12usize;
    let scores_buffer = context.new_buffer_with_data(&scores);
    let ranks_buffer = context.new_output_buffer((num_blocks * 4) as u64);
    let positions_buffer = context.new_output_buffer(8 * 4);
    let count_buffer = context.new_output_buffer(4);
    let status_buffer = context.new_output_buffer(4);
    write_buffer_bytes(&status_buffer, 0, &[0; 4]);

    let pass = context.begin_pass();
    encode_qsa_topk_positions(
        &mut context,
        &pass,
        &scores_buffer,
        &ranks_buffer,
        &positions_buffer,
        &count_buffer,
        &status_buffer,
        num_blocks as u32,
        visible as u32,
        1,
        4,
    )
    .expect("QSA selector dispatch");
    pass.commit_and_wait();

    assert_eq!(read_u32s(&status_buffer, 1), [1]);
    assert_eq!(read_u32s(&count_buffer, 1), [1]);
    assert!(read_u32s(&positions_buffer, 1)[0] < visible as u32);
}
