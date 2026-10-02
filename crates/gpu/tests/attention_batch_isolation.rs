//! Verifies causal isolation and inactive-row guards through the public encoder.
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
const LIVE_ROWS: usize = 3;
const FIRST_QUERY_POSITION: u32 = 3;
const KV_POSITIONS: usize = 8;
const PARITY_TOLERANCE: f32 = 0.02;
const OUTPUT_CANARY: f32 = 0.125;
const ROW_PLAN_CANARY: u32 = 0xdead_beef;
const PARTIAL_M_CANARY: f32 = 12_345.0;
const PARTIAL_D_CANARY: f32 = 23_456.0;
const PARTIAL_O_CANARY: f32 = 34_567.0;

struct BatchRun {
    output: Vec<f32>,
    row_plan: Vec<u32>,
}

fn to_f16(values: &[f32]) -> Vec<f16> {
    values.iter().map(|&value| f16::from_f32(value)).collect()
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

fn write_f32(buffer: &metal::Buffer, value: f32) {
    let count = buffer.length() as usize / std::mem::size_of::<f32>();
    unsafe { std::slice::from_raw_parts_mut(buffer.contents().cast::<f32>(), count) }.fill(value);
}

fn read_f32(buffer: &metal::Buffer) -> Vec<f32> {
    let count = buffer.length() as usize / std::mem::size_of::<f32>();
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<f32>(), count) }.to_vec()
}

fn run_batch(context: &mut MetalContext, perturb_future_kv: bool) -> BatchRun {
    let row_width = NUM_Q_HEADS * HEAD_DIM;
    let kv_width = NUM_KV_HEADS * HEAD_DIM;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let layout = BatchAttentionScratchLayout::new(NUM_Q_HEADS, HEAD_DIM)
        .expect("valid batch-attention layout");
    assert_eq!(layout.capacity(), CAPACITY);

    let mut q = vec![f32::NAN; CAPACITY * row_width];
    for row in 0..LIVE_ROWS {
        q[row * row_width..(row + 1) * row_width].fill(0.25);
    }

    let mut k = vec![0.0; KV_POSITIONS * kv_width];
    let mut v = vec![0.0; KV_POSITIONS * kv_width];
    for position in 0..KV_POSITIONS {
        for kv_head in 0..NUM_KV_HEADS {
            for dim in 0..HEAD_DIM {
                let index = position * kv_width + kv_head * HEAD_DIM + dim;
                if position < FIRST_QUERY_POSITION as usize + 1 {
                    // Keep the prefix identical across both runs.
                    v[index] = -0.5;
                } else if perturb_future_kv {
                    k[index] = 0.5;
                    v[index] = 3.5;
                } else {
                    v[index] = 0.25;
                }
            }
        }
    }

    let q_buffer = context.new_buffer_with_data(&to_f16(&q));
    let k_buffer = context.new_buffer_with_data(&to_f16(&k));
    let v_buffer = context.new_buffer_with_data(&to_f16(&v));
    let scratch = BatchAttentionScratch::new(context, layout).expect("batch scratch");
    unsafe {
        std::slice::from_raw_parts_mut(
            scratch.row_plan.contents().cast::<u32>(),
            layout.capacity() * 4,
        )
        .fill(ROW_PLAN_CANARY);
    }
    write_f32(&scratch.m, PARTIAL_M_CANARY);
    write_f32(&scratch.d, PARTIAL_D_CANARY);
    write_f32(&scratch.o, PARTIAL_O_CANARY);

    let output = context.new_output_buffer(layout.output_buffer_bytes() as u64);
    let output_canary_bits = f16::from_f32(OUTPUT_CANARY).to_bits();
    unsafe {
        std::slice::from_raw_parts_mut(
            output.contents().cast::<u16>(),
            layout.output_buffer_bytes() / std::mem::size_of::<u16>(),
        )
        .fill(output_canary_bits);
    }

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
            FIRST_QUERY_POSITION,
            LIVE_ROWS,
            HEAD_DIM as u32,
            NUM_Q_HEADS as u32,
            NUM_KV_HEADS as u32,
            scale,
        )
        .expect("public batch encoder accepts the linear FP16 GQA fixture");
        pass.commit_and_wait();
    });

    let output_bits = unsafe {
        std::slice::from_raw_parts(
            output.contents().cast::<u16>(),
            layout.output_buffer_bytes() / std::mem::size_of::<u16>(),
        )
    };
    let row_plan = unsafe {
        std::slice::from_raw_parts(
            scratch.row_plan.contents().cast::<u32>(),
            layout.capacity() * 4,
        )
        .to_vec()
    };
    let m = read_f32(&scratch.m);
    let d = read_f32(&scratch.d);
    let o = read_f32(&scratch.o);

    assert!(row_plan[LIVE_ROWS * 4..]
        .iter()
        .all(|&value| value == ROW_PLAN_CANARY));
    assert!(output_bits[LIVE_ROWS * row_width..]
        .iter()
        .all(|&bits| bits == output_canary_bits));

    // All rows use one chunk, so the compact live prefix ends at M * heads.
    // The remaining capacity and chunk slots must not be touched by either pass.
    let live_slots = LIVE_ROWS * NUM_Q_HEADS;
    assert!(m[live_slots..]
        .iter()
        .all(|&value| value == PARTIAL_M_CANARY));
    assert!(d[live_slots..]
        .iter()
        .all(|&value| value == PARTIAL_D_CANARY));
    assert!(o[live_slots * HEAD_DIM..]
        .iter()
        .all(|&value| value == PARTIAL_O_CANARY));

    let output = output_bits[..LIVE_ROWS * row_width]
        .iter()
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    assert!(output.iter().all(|value| value.is_finite()));

    BatchRun { output, row_plan }
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

#[test]
fn future_kv_and_inactive_rows_cannot_change_live_causal_outputs() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let baseline = run_batch(&mut context, false);
    let perturbed = run_batch(&mut context, true);
    let row_width = NUM_Q_HEADS * HEAD_DIM;

    let expected_plans = [[0, 4, 4, 1], [0, 5, 5, 1], [0, 6, 6, 1]];
    for (row, expected) in expected_plans.iter().enumerate() {
        assert_eq!(
            &baseline.row_plan[row * 4..row * 4 + 4],
            expected,
            "baseline row {row} plan"
        );
        assert_eq!(
            &perturbed.row_plan[row * 4..row * 4 + 4],
            expected,
            "perturbed row {row} plan"
        );
    }

    let row_zero_start = 0;
    let row_zero_end = row_width;
    let early_row_diff = max_abs_diff(
        &baseline.output[row_zero_start..row_zero_end],
        &perturbed.output[row_zero_start..row_zero_end],
    );
    assert!(
        early_row_diff <= PARITY_TOLERANCE,
        "future K/V changed row 0 by {early_row_diff}, tolerance {PARITY_TOLERANCE}"
    );

    let later_rows_start = row_width;
    let later_rows_end = LIVE_ROWS * row_width;
    let later_row_diff = max_abs_diff(
        &baseline.output[later_rows_start..later_rows_end],
        &perturbed.output[later_rows_start..later_rows_end],
    );
    assert!(
        later_row_diff > PARITY_TOLERANCE,
        "future K/V perturbation did not affect a later row: max diff {later_row_diff}"
    );

    // The helper verifies output, plan, and compact partial scratch canaries
    // for both runs, including the NaN Q rows beyond the three-row live prefix.
}
