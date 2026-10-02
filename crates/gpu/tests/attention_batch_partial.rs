//! Directly dispatches the batch-only partial attention shader and inspects
//! row/head/chunk states, including neutral chunks and an inactive row.
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

fn expected_partial(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
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
    let kv_head = q_head / 2;
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
fn writes_each_live_row_head_chunk_and_preserves_inactive_row() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");

    let head_dim = 4u32;
    let num_q_heads = 2u32;
    let num_kv_heads = 1u32;
    let capacity = 3usize;
    let live_rows = 2u32;
    let num_chunks = 3u32;
    let seq_len = 5usize;
    let scale = 0.5f32;

    let q = vec![
        0.25, -0.5, 0.75, 0.125, -0.25, 0.5, 0.375, -0.625, // row 0
        0.5, 0.25, -0.375, 0.75, -0.5, -0.125, 0.625, 0.25, // row 1
        4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, // inactive row canary
    ];
    let k: Vec<f32> = (0..seq_len * head_dim as usize)
        .map(|i| 0.125 + (i % 7) as f32 * 0.0625)
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim as usize)
        .map(|i| 0.75 + (i % 5) as f32 * 0.125)
        .collect();
    // Distinct valid plans: row 0 partitions [0, 4) as 3 + 1, row 1
    // partitions [0, 5) as 2 + 2 + 1. The final row is outside live_rows.
    let plans = [
        0,
        4,
        3,
        2,
        0,
        5,
        2,
        3,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        u32::MAX,
    ];

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
        (capacity as u64) * num_q_heads as u64 * num_chunks as u64,
        THREADS_PER_GROUP,
    );
    pass.commit_and_wait();

    let got_m = read_f32(&m_buffer, partial_slots);
    let got_d = read_f32(&d_buffer, partial_slots);
    let got_o = read_f32(&o_buffer, partial_slots * head_dim as usize);
    for row in 0..live_rows as usize {
        let (key_start, row_seq_len, chunk_len, row_chunks) = if row == 0 {
            (0usize, 4usize, 3usize, 2usize)
        } else {
            (0usize, 5usize, 2usize, 3usize)
        };
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
