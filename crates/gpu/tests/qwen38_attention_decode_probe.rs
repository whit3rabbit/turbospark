//! Synthetic GPU timing of one-row decode attention at the Qwen3.8-27B
//! gated-attention shape (head_dim 256, 24 Q heads over 4 KV heads) across
//! context lengths: the barrier kernel behind `encode_attention_decode`
//! against the GQA-shared partial at several (simdgroups, chunks) settings.
//!
//! Each sample is one command buffer holding 16 attention dispatches over 16
//! distinct layer K/V buffers, the per-token shape of the model's 16
//! gated-attention layers, so no layer's K/V is still cache-resident from the
//! previous one. Arms are interleaved per repetition and the first round is
//! discarded. Diagnostic only: synthetic K/V, no model, no quality claim, and
//! not a frozen benchmark row. Run explicitly:
//!
//! ```sh
//! cargo test -p turbospark-gpu --release --test qwen38_attention_decode_probe -- --ignored --nocapture
//! ```
//!
//! The second probe times one prefill micro-batch's attention the same way:
//! the one-row kernel once per row, the dense Llama batch kernel, and the
//! multi-row GQA partial at several launches.

#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{
    autorelease_pool, encode_attention_decode, encode_attention_decode_batch,
    encode_attention_decode_gqa, encode_attention_decode_gqa_rows,
    encode_attention_decode_gqa_rows_with, AttentionScratch, BatchAttentionInputContract,
    BatchAttentionKvFormat, BatchAttentionKvLayout, BatchAttentionScratch,
    BatchAttentionScratchLayout, DecodeAttentionKernel, GqaRowsScratch, MetalContext,
    GQA_ROWS_SCRATCH_CHUNKS,
};

const HEAD_DIM: u32 = 256;
const NUM_Q_HEADS: u32 = 24;
const NUM_KV_HEADS: u32 = 4;
const LAYERS: usize = 16;
const CONTEXTS: [u32; 6] = [512, 2048, 4096, 8192, 16384, 32768];
const REPS: usize = 7;

fn le(values: impl Iterator<Item = f16>) -> Vec<u8> {
    values.flat_map(|v| v.to_bits().to_le_bytes()).collect()
}

#[derive(Clone, Copy, Debug)]
enum Arm {
    Barrier,
    /// (simdgroups, chunks, unroll).
    Gqa(u32, u32, u32),
}

#[test]
#[ignore = "performance probe on a real Metal device; run explicitly"]
fn qwen38_decode_attention_time_by_context() {
    let mut context = MetalContext::new().expect("Metal device");
    let max_seq = *CONTEXTS.last().unwrap() as usize;
    let kv_width = (NUM_KV_HEADS * HEAD_DIM) as usize;
    let q_width = (NUM_Q_HEADS * HEAD_DIM) as usize;
    let q = context.new_buffer_with_data(&le(
        (0..q_width).map(|i| f16::from_f32(((i as f32) * 0.019).sin() * 0.5))
    ));
    let kv: Vec<_> = (0..LAYERS)
        .map(|layer| {
            let phase = layer as f32 * 0.37;
            let k = context.new_buffer_with_data(&le((0..max_seq * kv_width)
                .map(|i| f16::from_f32(((i as f32) * 0.0007 + phase).cos() * 0.3))));
            let v = context.new_buffer_with_data(&le((0..max_seq * kv_width)
                .map(|i| f16::from_f32(0.7 + ((i as f32) * 0.0011 + phase).sin() * 0.3))));
            (k, v)
        })
        .collect();
    let out = context.new_output_buffer((q_width * 2) as u64);
    let scratch = AttentionScratch::new(&context, NUM_Q_HEADS, HEAD_DIM);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let arms = [
        Arm::Barrier,
        Arm::Gqa(8, 16, 1),
        Arm::Gqa(8, 16, 2),
        Arm::Gqa(8, 16, 4),
        Arm::Gqa(4, 16, 4),
        Arm::Gqa(8, 8, 4),
    ];
    for &seq_len in &CONTEXTS {
        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        for rep in 0..=REPS {
            for (a, &arm) in arms.iter().enumerate() {
                let t = autorelease_pool(|| {
                    let pass = context.begin_pass();
                    for (k, v) in &kv {
                        match arm {
                            Arm::Barrier => encode_attention_decode(
                                &mut context,
                                &pass,
                                (&q, 0),
                                k,
                                v,
                                &scratch,
                                (&out, 0),
                                HEAD_DIM,
                                NUM_Q_HEADS,
                                NUM_KV_HEADS,
                                seq_len,
                                0,
                                0,
                                scale,
                                None,
                            ),
                            Arm::Gqa(sg, chunks, unroll) => encode_attention_decode_gqa(
                                &mut context,
                                &pass,
                                (&q, 0),
                                k,
                                v,
                                &scratch,
                                (&out, 0),
                                HEAD_DIM,
                                NUM_Q_HEADS,
                                NUM_KV_HEADS,
                                seq_len,
                                0,
                                0,
                                scale,
                                None,
                                sg,
                                chunks.min(seq_len.next_power_of_two()),
                                unroll,
                            ),
                        }
                        .expect("encode");
                    }
                    pass.commit_and_wait_with_gpu_time()
                });
                if rep > 0 {
                    samples[a].push(t * 1e3);
                }
            }
        }
        let kv_bytes = LAYERS as f64 * seq_len as f64 * kv_width as f64 * 2.0 * 2.0;
        let median = |s: &mut Vec<f64>| {
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s[s.len() / 2]
        };
        let base = median(&mut samples[0]);
        println!(
            "seq_len {seq_len}: unique K/V {:.0} MB per token",
            kv_bytes / 1e6
        );
        for (a, arm) in arms.iter().enumerate() {
            let m = median(&mut samples[a]);
            println!(
                "  {:<14} {:>8.3} ms/token  {:>6.0} GB/s unique  {:>5.2}x vs barrier",
                format!("{arm:?}"),
                m,
                kv_bytes / (m * 1e-3) / 1e9,
                base / m
            );
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum RowsArm {
    /// The one-row GQA kernel once per row: the prefill micro-batch today.
    PerRow,
    /// `attention_decode_batch_partial` in groups of up to 16 rows.
    Batch16,
    /// (rows per threadgroup, heads per simdgroup, chunks, unroll, tile).
    Rows(u32, u32, u32, u32, u32),
    /// `encode_attention_decode_gqa_rows`, the production launch.
    RowsProd,
}

/// Attention for one prefill micro-batch of `rows` query rows ending at
/// `seq_len` (row r at position `seq_len - rows + r`), over the 16 layers'
/// distinct K/V. The per-row loop is what `encode_full_attention_block_batched`
/// runs today; the batch kernel is the dense Llama prefill kernel; the rows
/// kernel stages each K/V tile once per group of rows.
#[test]
#[ignore = "performance probe on a real Metal device; run explicitly"]
fn qwen38_prefill_rows_attention_time_by_context() {
    let mut context = MetalContext::new().expect("Metal device");
    let max_rows = 64usize;
    let max_seq = *CONTEXTS.last().unwrap() as usize;
    let kv_width = (NUM_KV_HEADS * HEAD_DIM) as usize;
    let q_width = (NUM_Q_HEADS * HEAD_DIM) as usize;
    let q = context.new_buffer_with_data(&le(
        (0..max_rows * q_width).map(|i| f16::from_f32(((i as f32) * 0.019).sin() * 0.5))
    ));
    let kv: Vec<_> = (0..LAYERS)
        .map(|layer| {
            let phase = layer as f32 * 0.37;
            let k = context.new_buffer_with_data(&le((0..max_seq * kv_width)
                .map(|i| f16::from_f32(((i as f32) * 0.0007 + phase).cos() * 0.3))));
            let v = context.new_buffer_with_data(&le((0..max_seq * kv_width)
                .map(|i| f16::from_f32(0.7 + ((i as f32) * 0.0011 + phase).sin() * 0.3))));
            (k, v)
        })
        .collect();
    let out = context.new_output_buffer((max_rows * q_width * 2) as u64);
    let row_scratch = AttentionScratch::new(&context, NUM_Q_HEADS, HEAD_DIM)
        .with_decode_kernel(DecodeAttentionKernel::GqaShared);
    let layout = BatchAttentionScratchLayout::new(NUM_Q_HEADS as usize, HEAD_DIM as usize)
        .expect("batch scratch layout");
    let batch_capacity = layout.capacity();
    let batch_scratch = BatchAttentionScratch::new(&context, layout).expect("batch scratch");
    let rows_scratch = GqaRowsScratch::new(&context, max_rows as u32, NUM_Q_HEADS, HEAD_DIM, 16);
    // Sized as the runtime sizes it, so the production launch picks the
    // chunk counts it picks there.
    let prod_scratch = GqaRowsScratch::new(
        &context,
        max_rows as u32,
        NUM_Q_HEADS,
        HEAD_DIM,
        GQA_ROWS_SCRATCH_CHUNKS,
    );
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let contract = BatchAttentionInputContract {
        kv_layout: BatchAttentionKvLayout::Linear,
        kv_format: BatchAttentionKvFormat::Fp16,
        kv_start: 0,
        ring_capacity: 0,
        sink_count: 0,
    };
    // Production is Rows(2, 1, 2, 3, 12) (`gqa_rows_launch`). Rows(8, 6, ..)
    // is the first design, a row's whole group per simdgroup, kept for the
    // record with the dense Llama batch kernel (Batch16), which ran 4-12x
    // slower than the per-row loop at this shape.
    let arms = [
        RowsArm::PerRow,
        RowsArm::Batch16,
        RowsArm::Rows(2, 1, 2, 3, 12),
        RowsArm::Rows(2, 1, 2, 4, 16),
        RowsArm::Rows(2, 1, 8, 4, 16),
        RowsArm::Rows(4, 1, 4, 4, 16),
        RowsArm::Rows(8, 6, 4, 4, 16),
        RowsArm::RowsProd,
    ];
    // 64 and 16 are the Steel and scalar prefill micro-batches; 3 and 6 are
    // block-2 and block-5 verifies.
    for rows in [64usize, 16, 7, 6, 3, 1] {
        for &seq_len in &CONTEXTS {
            let first = seq_len - rows as u32;
            let mut samples: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
            for rep in 0..=REPS {
                for (a, &arm) in arms.iter().enumerate() {
                    let t = autorelease_pool(|| {
                        let pass = context.begin_pass();
                        for (k, v) in &kv {
                            match arm {
                                RowsArm::PerRow => {
                                    for r in 0..rows {
                                        let off = (r * q_width * 2) as u64;
                                        encode_attention_decode(
                                            &mut context,
                                            &pass,
                                            (&q, off),
                                            k,
                                            v,
                                            &row_scratch,
                                            (&out, off),
                                            HEAD_DIM,
                                            NUM_Q_HEADS,
                                            NUM_KV_HEADS,
                                            first + r as u32 + 1,
                                            0,
                                            0,
                                            scale,
                                            None,
                                        )
                                        .expect("encode");
                                    }
                                }
                                RowsArm::Batch16 => {
                                    for g in (0..rows).step_by(batch_capacity) {
                                        let off = (g * q_width * 2) as u64;
                                        encode_attention_decode_batch(
                                            &mut context,
                                            &pass,
                                            (&q, off),
                                            k,
                                            v,
                                            &batch_scratch,
                                            (&out, off),
                                            contract,
                                            first + g as u32,
                                            (rows - g).min(batch_capacity),
                                            HEAD_DIM,
                                            NUM_Q_HEADS,
                                            NUM_KV_HEADS,
                                            scale,
                                        )
                                        .expect("encode");
                                    }
                                }
                                RowsArm::RowsProd => {
                                    encode_attention_decode_gqa_rows(
                                        &mut context,
                                        &pass,
                                        (&q, 0),
                                        k,
                                        v,
                                        &prod_scratch,
                                        (&out, 0),
                                        HEAD_DIM,
                                        NUM_Q_HEADS,
                                        NUM_KV_HEADS,
                                        first,
                                        rows as u32,
                                        0,
                                        0,
                                        scale,
                                    )
                                    .expect("encode");
                                }
                                RowsArm::Rows(group_rows, heads, chunks, unroll, tile) => {
                                    encode_attention_decode_gqa_rows_with(
                                        &mut context,
                                        &pass,
                                        (&q, 0),
                                        k,
                                        v,
                                        &rows_scratch,
                                        (&out, 0),
                                        HEAD_DIM,
                                        NUM_Q_HEADS,
                                        NUM_KV_HEADS,
                                        first,
                                        rows as u32,
                                        0,
                                        0,
                                        scale,
                                        group_rows,
                                        heads,
                                        chunks,
                                        unroll,
                                        tile,
                                    )
                                    .expect("encode");
                                }
                            }
                        }
                        pass.commit_and_wait_with_gpu_time()
                    });
                    if rep > 0 {
                        samples[a].push(t * 1e3);
                    }
                }
            }
            let median = |s: &mut Vec<f64>| {
                s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                s[s.len() / 2]
            };
            let base = median(&mut samples[0]);
            println!("rows {rows}, seq_len {seq_len}: 16 layers per micro-batch");
            for (a, arm) in arms.iter().enumerate() {
                let m = median(&mut samples[a]);
                println!(
                    "  {:<22} {:>9.3} ms/batch  {:>7.3} ms/token  {:>5.2}x vs per-row",
                    format!("{arm:?}"),
                    m,
                    m / rows as f64,
                    base / m
                );
            }
        }
    }
}
