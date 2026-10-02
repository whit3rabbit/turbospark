//! Covers batch attention rows with different causal context and chunk plans.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{
    autorelease_pool, encode_attention_decode_batch, BatchAttentionInputContract,
    BatchAttentionKvFormat, BatchAttentionKvLayout, BatchAttentionScratch,
    BatchAttentionScratchLayout, MetalContext,
};

const HEAD_DIM: usize = 8;
const NUM_Q_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 2;
const CAPACITY: usize = 16;
const MAX_CHUNKS: usize = 16;
const PARITY_TOLERANCE: f32 = 0.02;

struct BatchRun {
    output: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    scratch: BatchAttentionScratch,
}

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

fn read_fp32(buffer: &metal::Buffer, count: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, count) }.to_vec()
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

fn run_batch(context: &mut MetalContext, first_query_position: u32, live_rows: usize) -> BatchRun {
    let head_width = NUM_Q_HEADS * HEAD_DIM;
    let max_seq_len = first_query_position as usize + live_rows;
    let kv_width = NUM_KV_HEADS * HEAD_DIM;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let layout = BatchAttentionScratchLayout::new(NUM_Q_HEADS, HEAD_DIM).expect("valid layout");
    assert_eq!(layout.capacity(), CAPACITY);

    // The public batch API requires capacity-sized Q and output buffers even
    // when this test dispatches only a short live prefix.
    let q: Vec<f32> = (0..CAPACITY * head_width)
        .map(|index| (index as f32 * 0.071).sin() * 0.7)
        .collect();
    let k: Vec<f32> = (0..max_seq_len * kv_width)
        .map(|index| (index as f32 * 0.113).sin() * 0.4)
        .collect();
    let v: Vec<f32> = (0..max_seq_len * kv_width)
        .map(|index| {
            let kv_head = (index / HEAD_DIM) % NUM_KV_HEADS;
            1.2 + kv_head as f32 * 0.7 + (index as f32 * 0.17).cos() * 0.2
        })
        .collect();
    let q_buffer = context.new_buffer_with_data(&to_f16(&q));
    let k_buffer = context.new_buffer_with_data(&to_f16(&k));
    let v_buffer = context.new_buffer_with_data(&to_f16(&v));
    let scratch = BatchAttentionScratch::new(context, layout).expect("batch scratch");
    let output = context.new_output_buffer(layout.output_buffer_bytes() as u64);

    autorelease_pool(|| {
        let pass = context.begin_pass();
        encode_attention_decode_batch(
            context,
            &pass,
            (&q_buffer, 0),
            &k_buffer,
            &v_buffer,
            &scratch,
            (&output, 0),
            linear_fp16_contract(),
            first_query_position,
            live_rows,
            HEAD_DIM as u32,
            NUM_Q_HEADS as u32,
            NUM_KV_HEADS as u32,
            scale,
        )
        .expect("batch encoder accepts the linear FP16 GQA fixture");
        pass.commit_and_wait();
    });

    BatchRun {
        output: read_fp16(&output, live_rows * head_width),
        q,
        k,
        v,
        scratch,
    }
}

fn assert_cpu_causal_parity(run: &BatchRun, first_query_position: u32, live_rows: usize) {
    let head_width = NUM_Q_HEADS * HEAD_DIM;
    let kv_width = NUM_KV_HEADS * HEAD_DIM;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    for row in 0..live_rows {
        let seq_len = first_query_position as usize + row + 1;
        let q_start = row * head_width;
        let expected = turbospark_compute::causal_attention(
            &run.q[q_start..q_start + head_width],
            &run.k[..seq_len * kv_width],
            &run.v[..seq_len * kv_width],
            HEAD_DIM,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            seq_len,
            None,
            Some(scale),
        );
        let actual = &run.output[q_start..q_start + head_width];
        let error = turbospark_compute::max_abs_diff(actual, &expected);
        assert!(
            error < PARITY_TOLERANCE,
            "row {row}, seq_len={seq_len}: batch versus CPU causal reference error {error}"
        );
        assert!(actual.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn mixed_context_rows_match_their_causal_references_and_write_neutral_slots() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    // These rows have sequence lengths 31, 32, 33, and 34. The first row
    // uses one chunk, while later rows use two; row 2 has a short final chunk.
    let first_query_position = 30;
    let live_rows = 4;
    let run = run_batch(&mut context, first_query_position, live_rows);

    assert_cpu_causal_parity(&run, first_query_position, live_rows);

    let plans = unsafe {
        std::slice::from_raw_parts(run.scratch.row_plan.contents().cast::<u32>(), CAPACITY * 4)
    };
    let expected_plans = [
        [0, 31, 31, 1],
        [0, 32, 16, 2],
        [0, 33, 17, 2],
        [0, 34, 17, 2],
    ];
    for (row, expected) in expected_plans.iter().enumerate() {
        assert_eq!(&plans[row * 4..row * 4 + 4], expected, "row {row} plan");
    }

    // Row zero has only one planned chunk although NC is two for this batch.
    // Its second partial slot must be neutral before combine, including O.
    let dispatch_chunks = 2;
    let scalar_slots = CAPACITY * NUM_Q_HEADS * MAX_CHUNKS;
    let m = read_fp32(&run.scratch.m, scalar_slots);
    let d = read_fp32(&run.scratch.d, scalar_slots);
    let o = read_fp32(&run.scratch.o, scalar_slots * HEAD_DIM);
    for q_head in 0..NUM_Q_HEADS {
        let slot = q_head * dispatch_chunks + 1;
        assert_eq!(m[slot], f32::NEG_INFINITY, "neutral max for head {q_head}");
        assert_eq!(d[slot], 0.0, "neutral denominator for head {q_head}");
        let o_start = slot * HEAD_DIM;
        assert!(
            o[o_start..o_start + HEAD_DIM]
                .iter()
                .all(|&value| value == 0.0),
            "neutral numerator for head {q_head}"
        );
    }
    assert_eq!(
        dispatch_chunks,
        expected_plans.iter().map(|plan| plan[3]).max().unwrap() as usize
    );
}

#[test]
fn batch_api_at_position_zero_matches_the_single_token_causal_reference() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let run = run_batch(&mut context, 0, 1);
    assert_cpu_causal_parity(&run, 0, 1);

    let plans = unsafe {
        std::slice::from_raw_parts(run.scratch.row_plan.contents().cast::<u32>(), CAPACITY * 4)
    };
    assert_eq!(&plans[..4], &[0, 1, 1, 1]);
}
