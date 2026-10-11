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

#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{
    autorelease_pool, encode_attention_decode, encode_attention_decode_gqa, AttentionScratch,
    MetalContext,
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
