//! Exercises the public batch-attention encoder through both Metal passes.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{
    autorelease_pool, encode_attention_decode_batch, BatchAttentionInputContract,
    BatchAttentionKvFormat, BatchAttentionKvLayout, BatchAttentionScratch,
    BatchAttentionScratchLayout, MetalContext,
};

fn fp16_buffer(context: &MetalContext, values: &[f32]) -> metal::Buffer {
    let values: Vec<f16> = values.iter().copied().map(f16::from_f32).collect();
    context.new_buffer_with_data(&values)
}

fn read_fp16(buffer: &metal::Buffer, count: usize) -> Vec<f32> {
    let values = unsafe { std::slice::from_raw_parts(buffer.contents() as *const u16, count) };
    values
        .iter()
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect()
}

fn expected_attention(q: &[f32], k: &[f32], v: &[f32], seq_len: usize, scale: f32) -> Vec<f32> {
    let head_dim = 4;
    let scores: Vec<f32> = (0..seq_len)
        .map(|position| {
            let mut score = 0.0;
            for i in 0..head_dim {
                score += q[i] * k[position * head_dim + i];
            }
            score * scale
        })
        .collect();
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = scores
        .iter()
        .map(|score| (score - max_score).exp())
        .collect();
    let denominator: f32 = weights.iter().sum();
    (0..head_dim)
        .map(|i| {
            weights
                .iter()
                .enumerate()
                .map(|(position, weight)| weight * v[position * head_dim + i])
                .sum::<f32>()
                / denominator
        })
        .collect()
}

fn linear_fp16_contract() -> BatchAttentionInputContract {
    BatchAttentionInputContract {
        kv_layout: BatchAttentionKvLayout::Linear,
        kv_format: BatchAttentionKvFormat::Fp16,
        kv_start: 0,
        ring_capacity: 0,
        sink_count: 0,
    }
}

#[test]
fn public_encoder_runs_both_passes_and_keeps_scale_and_chunk_cache_axes() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let head_dim = 4u32;
    let num_q_heads = 1u32;
    let num_kv_heads = 1u32;
    let live_rows = 2usize;
    let layout = BatchAttentionScratchLayout::new(num_q_heads as usize, head_dim as usize)
        .expect("batch layout");
    let scratch = BatchAttentionScratch::new(&context, layout).expect("batch scratch");
    let row_plan_canary = 0xfeed_beefu32;
    unsafe {
        std::slice::from_raw_parts_mut(
            scratch.row_plan.contents().cast::<u32>(),
            layout.capacity() * 4,
        )
        .fill(row_plan_canary);
    }

    let q_values: Vec<f32> = (0..layout.capacity() * head_dim as usize)
        .map(|i| 0.125 + (i % 9) as f32 * 0.0625)
        .collect();
    let k_values: Vec<f32> = (0..40 * head_dim as usize)
        .map(|i| -0.25 + (i % 13) as f32 * 0.0625)
        .collect();
    let v_values: Vec<f32> = (0..40 * head_dim as usize)
        .map(|i| 0.5 + (i % 7) as f32 * 0.125)
        .collect();
    let q_buffer = fp16_buffer(&context, &q_values);
    let k_buffer = fp16_buffer(&context, &k_values);
    let v_buffer = fp16_buffer(&context, &v_values);

    // The first request selects NC=1. The next request on the same context
    // selects NC=2, then changes scale without changing NC. Correct outputs
    // prove both specialization values remain distinct in the pipeline cache.
    for (first_query_position, scale) in [(14u32, 0.5f32), (31, 0.5), (31, 0.25)] {
        autorelease_pool(|| {
            let output_buffer = context.new_output_buffer(layout.output_buffer_bytes() as u64);
            let pass = context.begin_pass();
            encode_attention_decode_batch(
                &mut context,
                &pass,
                (&q_buffer, 0),
                &k_buffer,
                &v_buffer,
                &scratch,
                (&output_buffer, 0),
                linear_fp16_contract(),
                first_query_position,
                live_rows,
                head_dim,
                num_q_heads,
                num_kv_heads,
                scale,
            )
            .expect("encode batch partial and combine passes");
            pass.commit_and_wait();

            let expected_chunk_count = if first_query_position >= 16 { 2 } else { 1 };
            let row_plans = unsafe {
                std::slice::from_raw_parts(
                    scratch.row_plan.contents().cast::<u32>(),
                    layout.capacity() * 4,
                )
            };
            for row in 0..live_rows {
                let seq_len = first_query_position as usize + row + 1;
                let chunk_len = seq_len.div_ceil(expected_chunk_count as usize);
                assert!(
                    row_plans[row * 4..row * 4 + 4]
                        == [0, seq_len as u32, chunk_len as u32, expected_chunk_count],
                    "derived row plan {row}: {:?}",
                    &row_plans[row * 4..row * 4 + 4]
                );
            }
            assert!(row_plans[live_rows * 4..]
                .iter()
                .all(|&value| value == row_plan_canary));

            let actual = read_fp16(&output_buffer, live_rows * head_dim as usize);
            for row in 0..live_rows {
                let seq_len = first_query_position as usize + row + 1;
                let expected = expected_attention(
                    &q_values[row * head_dim as usize..(row + 1) * head_dim as usize],
                    &k_values,
                    &v_values,
                    seq_len,
                    scale,
                );
                for (i, want) in expected.into_iter().enumerate() {
                    let got = actual[row * head_dim as usize + i];
                    assert!(
                        (got - want).abs() < 0.003,
                        "row {row}, element {i}, first={first_query_position}, scale={scale}: got {got}, want {want}"
                    );
                }
            }
        });
    }
}
