//! Directly dispatches the batch-only partial attention shader and inspects
//! row/head/chunk states across four-query tiles and disjoint row plans.
#![cfg(target_os = "macos")]

use half::f16;
use metal::{FunctionConstantValues, MTLDataType};
use turbospark_gpu::MetalContext;

const SOURCE: &str = include_str!("../src/shaders/attention.metal");
const THREADS_PER_GROUP: u64 = 256;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&value| f16::from_f32(value).to_bits().to_le_bytes())
        .collect()
}

fn constants(scale: f32, num_chunks: u32) -> FunctionConstantValues {
    let values = FunctionConstantValues::new();
    let zero = 0u32;
    let use_fc = false;
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 60);
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 61);
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 62);
    values.set_constant_value_at_index((&use_fc as *const bool).cast(), MTLDataType::Bool, 63);
    values.set_constant_value_at_index((&scale as *const f32).cast(), MTLDataType::Float, 64);
    values.set_constant_value_at_index((&num_chunks as *const u32).cast(), MTLDataType::UInt, 65);
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 69);
    values
}

fn read_f32(buffer: &metal::Buffer, count: usize) -> Vec<f32> {
    let values = unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, count) };
    values.to_vec()
}

fn read_f16(buffer: &metal::Buffer, count: usize) -> Vec<f32> {
    let values = unsafe { std::slice::from_raw_parts(buffer.contents() as *const u16, count) };
    values
        .iter()
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect()
}

fn expected_partial(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    q_head: usize,
    seq_len: usize,
    key_start: usize,
    chunk_len: usize,
    chunk: usize,
    scale: f32,
) -> (f32, f32, Vec<f32>) {
    let start = key_start + chunk * chunk_len;
    let end = (start + chunk_len).min(seq_len);
    let kv_head = q_head / (num_q_heads / num_kv_heads);
    let mut m = f32::NEG_INFINITY;
    let mut d = 0.0f32;
    let mut o = vec![0.0f32; head_dim];

    for position in start..end {
        let key_base = (position * num_kv_heads + kv_head) * head_dim;
        let q_base = q_head * head_dim;
        let mut score = 0.0f32;
        for i in 0..head_dim {
            score += q[q_base + i] * k[key_base + i];
        }
        let score = score * scale;
        let next_m = m.max(score);
        let alpha = (m - next_m).exp();
        let weight = (score - next_m).exp();
        d = d * alpha + weight;
        for i in 0..head_dim {
            o[i] = o[i] * alpha + weight * v[key_base + i];
        }
        m = next_m;
    }

    (m, d, o)
}

#[test]
fn reuses_kv_across_overlapping_and_disjoint_four_query_tiles() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let head_dim = 4u32;
    let num_q_heads = 2u32;
    let num_kv_heads = 1u32;
    let capacity = 8usize;
    let live_rows = 5u32;
    let num_chunks = 3u32;
    let seq_len = 16usize;
    let scale = 0.5f32;

    let mut q: Vec<f32> = (0..capacity * num_q_heads as usize * head_dim as usize)
        .map(|i| -0.5 + (i % 11) as f32 * 0.125)
        .collect();
    let inactive_q_start = live_rows as usize * num_q_heads as usize * head_dim as usize;
    q[inactive_q_start..].fill(4.0);
    let k: Vec<f32> = (0..seq_len * head_dim as usize)
        .map(|i| 0.125 + (i % 7) as f32 * 0.0625)
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim as usize)
        .map(|i| 0.75 + (i % 5) as f32 * 0.125)
        .collect();
    // The first tile mixes overlapping and disjoint intervals for the same
    // chunk. Row 4 is the only live query in the tail tile. Rows 5-7 are
    // inactive and carry invalid-plan canaries.
    let row_plans = [
        [1, 6, 2, 3],
        [2, 8, 2, 3],
        [9, 12, 2, 2],
        [14, 16, 1, 2],
        [5, 9, 3, 2],
    ];
    let mut plans = Vec::with_capacity(capacity * 4);
    for plan in row_plans {
        plans.extend(plan);
    }
    for _ in live_rows as usize..capacity {
        plans.extend([u32::MAX; 4]);
    }

    let q_buffer = context.new_buffer_with_data(&f16_bytes(&q));
    let k_buffer = context.new_buffer_with_data(&f16_bytes(&k));
    let v_buffer = context.new_buffer_with_data(&f16_bytes(&v));
    let plan_buffer = context.new_buffer_with_data(&u32_bytes(&plans));

    let partial_slots = capacity * num_q_heads as usize * num_chunks as usize;
    let canary = 777.0f32;
    let m_buffer = context.new_buffer_with_data(&f32_bytes(&vec![canary; partial_slots]));
    let d_buffer = context.new_buffer_with_data(&f32_bytes(&vec![canary; partial_slots]));
    let o_buffer =
        context.new_buffer_with_data(&f32_bytes(&vec![canary; partial_slots * head_dim as usize]));

    let pipeline = context
        .pipeline(
            SOURCE,
            "attention_decode_batch_partial",
            &constants(scale, num_chunks),
            &[scale.to_le_bytes(), num_chunks.to_le_bytes()].concat(),
        )
        .expect("batch partial pipeline");

    let pass = context.begin_pass();
    pass.encode_threadgroups(
        &pipeline,
        &[
            (&q_buffer, 0, 0),
            (&k_buffer, 1, 0),
            (&v_buffer, 2, 0),
            (&m_buffer, 3, 0),
            (&d_buffer, 4, 0),
            (&o_buffer, 5, 0),
            (&plan_buffer, 9, 0),
        ],
        &[
            (&head_dim.to_le_bytes(), 6),
            (&num_q_heads.to_le_bytes(), 7),
            (&num_kv_heads.to_le_bytes(), 8),
            (&live_rows.to_le_bytes(), 10),
            (&scale.to_le_bytes(), 11),
        ],
        ((live_rows as u64 + 3) / 4) * num_q_heads as u64 * num_chunks as u64,
        THREADS_PER_GROUP,
    );
    pass.commit_and_wait();

    let got_m = read_f32(&m_buffer, partial_slots);
    let got_d = read_f32(&d_buffer, partial_slots);
    let got_o = read_f32(&o_buffer, partial_slots * head_dim as usize);
    for row in 0..live_rows as usize {
        let [key_start, row_seq_len, chunk_len, row_chunks] = row_plans[row];
        let (key_start, row_seq_len, chunk_len, row_chunks) = (
            key_start as usize,
            row_seq_len as usize,
            chunk_len as usize,
            row_chunks as usize,
        );
        for q_head in 0..num_q_heads as usize {
            for chunk in 0..num_chunks as usize {
                let slot = (row * num_q_heads as usize + q_head) * num_chunks as usize + chunk;
                if chunk >= row_chunks {
                    assert_eq!(
                        got_m[slot],
                        f32::NEG_INFINITY,
                        "neutral m at {row}/{q_head}/{chunk}"
                    );
                    assert_eq!(got_d[slot], 0.0, "neutral d at {row}/{q_head}/{chunk}");
                    assert!(
                        got_o[slot * head_dim as usize..(slot + 1) * head_dim as usize]
                            .iter()
                            .all(|&x| x == 0.0),
                        "neutral o at {row}/{q_head}/{chunk}"
                    );
                    continue;
                }

                let (want_m, want_d, want_o) = expected_partial(
                    &q[row * num_q_heads as usize * head_dim as usize..],
                    &k,
                    &v,
                    head_dim as usize,
                    num_q_heads as usize,
                    num_kv_heads as usize,
                    q_head,
                    row_seq_len,
                    key_start,
                    chunk_len,
                    chunk,
                    scale,
                );
                assert!(
                    (got_m[slot] - want_m).abs() < 2e-5,
                    "m at {row}/{q_head}/{chunk}: got {}, want {want_m}",
                    got_m[slot]
                );
                assert!(
                    (got_d[slot] - want_d).abs() < 2e-5,
                    "d at {row}/{q_head}/{chunk}: got {}, want {want_d}",
                    got_d[slot]
                );
                for (i, (&got, want)) in got_o
                    [slot * head_dim as usize..(slot + 1) * head_dim as usize]
                    .iter()
                    .zip(want_o)
                    .enumerate()
                {
                    assert!(
                        (got - want).abs() < 2e-5,
                        "o at {row}/{q_head}/{chunk}/{i}: got {got}, want {want}"
                    );
                }
            }
        }
    }

    let inactive_start = live_rows as usize * num_q_heads as usize * num_chunks as usize;
    assert!(got_m[inactive_start..].iter().all(|&x| x == canary));
    assert!(got_d[inactive_start..].iter().all(|&x| x == canary));
    assert!(got_o[inactive_start * head_dim as usize..]
        .iter()
        .all(|&x| x == canary));
}

#[test]
fn combines_partial_states_per_live_row_and_head_without_touching_absent_rows() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let head_dim = 4u32;
    let num_q_heads = 2u32;
    let max_chunks = 4u32;
    let live_rows = 3u32;
    let capacity = 5usize;
    let partial_slots = capacity * num_q_heads as usize * max_chunks as usize;
    let active_chunks = [[2usize, 4], [1, 3], [3, 2]];

    let mut m = vec![f32::NEG_INFINITY; partial_slots];
    let mut d = vec![0.0f32; partial_slots];
    let mut o = vec![0.0f32; partial_slots * head_dim as usize];
    for row in 0..live_rows as usize {
        for q_head in 0..num_q_heads as usize {
            for chunk in 0..active_chunks[row][q_head] {
                let slot = (row * num_q_heads as usize + q_head) * max_chunks as usize + chunk;
                m[slot] = -0.5 + row as f32 * 0.23 + q_head as f32 * 0.11 + chunk as f32 * 0.17;
                d[slot] = 0.75 + ((row + q_head + chunk) % 5) as f32 * 0.2;
                for i in 0..head_dim as usize {
                    o[slot * head_dim as usize + i] =
                        -0.4 + (row * 7 + q_head * 3 + chunk + i) as f32 * 0.09;
                }
            }
        }
    }

    // Inactive input rows are deliberately invalid. A live-row guard must
    // return before the combine kernel reads any of these partial states.
    let inactive_start = live_rows as usize * num_q_heads as usize * max_chunks as usize;
    m[inactive_start..].fill(f32::NAN);
    d[inactive_start..].fill(f32::NAN);
    o[inactive_start * head_dim as usize..].fill(f32::NAN);

    let m_buffer = context.new_buffer_with_data(&f32_bytes(&m));
    let d_buffer = context.new_buffer_with_data(&f32_bytes(&d));
    let o_buffer = context.new_buffer_with_data(&f32_bytes(&o));
    let canary = 321.0f32;
    let output_count = capacity * num_q_heads as usize * head_dim as usize;
    let output_buffer = context.new_buffer_with_data(&f16_bytes(&vec![canary; output_count]));

    let pipeline = context
        .pipeline(
            SOURCE,
            "attention_decode_batch_combine",
            &FunctionConstantValues::new(),
            &[],
        )
        .expect("batch combine pipeline");

    let pass = context.begin_pass();
    pass.encode_threadgroups(
        &pipeline,
        &[
            (&m_buffer, 0, 0),
            (&d_buffer, 1, 0),
            (&o_buffer, 2, 0),
            (&output_buffer, 3, 0),
        ],
        &[
            (&head_dim.to_le_bytes(), 4),
            (&num_q_heads.to_le_bytes(), 5),
            (&max_chunks.to_le_bytes(), 6),
            (&live_rows.to_le_bytes(), 7),
        ],
        capacity as u64 * num_q_heads as u64,
        THREADS_PER_GROUP,
    );
    pass.commit_and_wait();

    let got = read_f16(&output_buffer, output_count);
    for row in 0..live_rows as usize {
        for q_head in 0..num_q_heads as usize {
            let count = active_chunks[row][q_head];
            let first_slot = (row * num_q_heads as usize + q_head) * max_chunks as usize;
            let max_m = (0..count)
                .map(|chunk| m[first_slot + chunk])
                .fold(f32::NEG_INFINITY, f32::max);
            let denominator: f32 = (0..count)
                .map(|chunk| d[first_slot + chunk] * (m[first_slot + chunk] - max_m).exp())
                .sum();

            for i in 0..head_dim as usize {
                let numerator: f32 = (0..count)
                    .map(|chunk| {
                        o[(first_slot + chunk) * head_dim as usize + i]
                            * (m[first_slot + chunk] - max_m).exp()
                    })
                    .sum();
                let want = numerator / denominator;
                let index = (row * num_q_heads as usize + q_head) * head_dim as usize + i;
                assert!(
                    (got[index] - want).abs() < 1e-3,
                    "output at {row}/{q_head}/{i}: got {}, want {want}",
                    got[index]
                );
            }
        }
    }

    let inactive_output_start = live_rows as usize * num_q_heads as usize * head_dim as usize;
    assert!(got[inactive_output_start..]
        .iter()
        .all(|&value| value == canary));
}
