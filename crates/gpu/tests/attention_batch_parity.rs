//! Compares the public multi-row attention encoder with the established
//! one-row encoder and the independent CPU causal-attention reference.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{
    attention_decode, autorelease_pool, encode_attention_decode_batch, BatchAttentionInputContract,
    BatchAttentionKvFormat, BatchAttentionKvLayout, BatchAttentionScratch,
    BatchAttentionScratchLayout, MetalContext,
};

const HEAD_DIM: usize = 8;
const NUM_Q_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 2;
const FIRST_QUERY_POSITION: u32 = 47;
const MAX_LIVE_ROWS: usize = 16;
const PARITY_TOLERANCE: f32 = 0.02;

fn to_f16(values: &[f32]) -> Vec<f16> {
    values.iter().map(|&value| f16::from_f32(value)).collect()
}

fn read_fp16(buffer: &metal::Buffer, count: usize) -> Vec<f32> {
    let values = unsafe { std::slice::from_raw_parts(buffer.contents() as *const u16, count) };
    values
        .iter()
        .map(|&bits| f16::from_bits(bits).to_f32())
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

fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    turbospark_compute::max_abs_diff(left, right)
}

#[test]
fn batch_outputs_match_serial_gpu_and_cpu_for_supported_batch_sizes_with_gqa() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let head_width = NUM_Q_HEADS * HEAD_DIM;
    let max_seq_len = FIRST_QUERY_POSITION as usize + MAX_LIVE_ROWS;
    let kv_width = NUM_KV_HEADS * HEAD_DIM;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let q_f32: Vec<f32> = (0..MAX_LIVE_ROWS * head_width)
        .map(|index| ((index as f32 * 0.071).sin()) * 0.7)
        .collect();
    let k_f32: Vec<f32> = (0..max_seq_len * kv_width)
        .map(|index| ((index as f32 * 0.113).sin()) * 0.4)
        .collect();
    // Distinct positive values for each KV head make a wrong GQA mapping or
    // an unwritten output visible in every live result.
    let v_f32: Vec<f32> = (0..max_seq_len * kv_width)
        .map(|index| {
            let kv_head = (index / HEAD_DIM) % NUM_KV_HEADS;
            1.2 + kv_head as f32 * 0.7 + ((index as f32 * 0.17).cos()) * 0.2
        })
        .collect();
    let q_f16 = to_f16(&q_f32);
    let k_f16 = to_f16(&k_f32);
    let v_f16 = to_f16(&v_f32);
    let layout = BatchAttentionScratchLayout::new(NUM_Q_HEADS, HEAD_DIM)
        .expect("valid batch-attention scratch layout");
    let q_buffer = context.new_buffer_with_data(&q_f16);
    let k_buffer = context.new_buffer_with_data(&k_f16);
    let v_buffer = context.new_buffer_with_data(&v_f16);

    for live_rows in [1usize, 2, 4, MAX_LIVE_ROWS] {
        let scratch = BatchAttentionScratch::new(&context, layout).expect("batch scratch");
        autorelease_pool(|| {
            let output = context.new_output_buffer(layout.output_buffer_bytes() as u64);
            let pass = context.begin_pass();
            encode_attention_decode_batch(
                &mut context,
                &pass,
                (&q_buffer, 0),
                &k_buffer,
                &v_buffer,
                &scratch,
                (&output, 0),
                linear_fp16_contract(),
                FIRST_QUERY_POSITION,
                live_rows,
                HEAD_DIM as u32,
                NUM_Q_HEADS as u32,
                NUM_KV_HEADS as u32,
                scale,
            )
            .expect("batch encoder accepts the linear FP16 GQA fixture");
            pass.commit_and_wait();

            let actual = read_fp16(&output, live_rows * head_width);
            for row in 0..live_rows {
                let seq_len = FIRST_QUERY_POSITION as usize + row + 1;
                let row_q_start = row * head_width;
                let row_q_end = row_q_start + head_width;
                let serial_gpu = attention_decode(
                    &mut context,
                    &q_f16[row_q_start..row_q_end],
                    &k_f16[..seq_len * kv_width],
                    &v_f16[..seq_len * kv_width],
                    HEAD_DIM as u32,
                    NUM_Q_HEADS as u32,
                    NUM_KV_HEADS as u32,
                    seq_len as u32,
                    scale,
                )
                .expect("existing one-row GPU encoder succeeds");
                let serial_gpu: Vec<f32> = serial_gpu.iter().map(|value| value.to_f32()).collect();
                let batch_row = &actual[row * head_width..(row + 1) * head_width];
                let gpu_error = max_abs_diff(batch_row, &serial_gpu);
                assert!(
                    gpu_error < PARITY_TOLERANCE,
                    "M={live_rows}, row={row}: batch versus serial GPU error {gpu_error}"
                );

                let cpu = turbospark_compute::causal_attention(
                    &q_f32[row_q_start..row_q_end],
                    &k_f32[..seq_len * kv_width],
                    &v_f32[..seq_len * kv_width],
                    HEAD_DIM,
                    NUM_Q_HEADS,
                    NUM_KV_HEADS,
                    seq_len,
                    None,
                    Some(scale),
                );
                let cpu_error = max_abs_diff(batch_row, &cpu);
                assert!(
                    cpu_error < PARITY_TOLERANCE,
                    "M={live_rows}, row={row}: batch versus CPU causal reference error {cpu_error}"
                );
                assert!(
                    batch_row.iter().all(|value| value.is_finite() && *value > 0.5),
                    "M={live_rows}, row={row}: expected finite, nonzero live output, got {batch_row:?}"
                );
            }
        });
    }
}
