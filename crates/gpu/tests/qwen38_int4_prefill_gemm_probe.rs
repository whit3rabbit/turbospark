//! Synthetic GPU timing of the INT4 prefill GEMM at Qwen3.8-27B's projection
//! shapes: the scalar `dequant_int4_gemm_simd` (16 rows per dispatch, so M
//! tokens take ceil(M / 16) dispatches, each re-reading the weights) against
//! the vendored Steel kernel at both tiles (all M rows in one dispatch).
//!
//! Each sample is one command buffer that runs the projection over two
//! distinct weight copies, so a matrix is never still cache-resident from
//! the previous dispatch. Arms are interleaved per repetition and the first
//! round is discarded. Diagnostic only: synthetic weights, no model, not a
//! frozen benchmark row. Run explicitly:
//!
//! ```sh
//! cargo test -p turbospark-gpu --release --test qwen38_int4_prefill_gemm_probe -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use half::{bf16, f16};
use turbospark_gpu::{
    autorelease_pool, encode_dequant_int4_gemm_resident, encode_dequant_int4_gemm_steel,
    Int4ResidentMatrix, MetalContext, SteelTile, MAX_BATCH_ROWS,
};

/// (name, output rows, input width) of Qwen3.8-27B's INT4 projections.
const SHAPES: [(&str, usize, usize); 6] = [
    ("ffn gate/up", 17408, 5120),
    ("ffn down", 5120, 17408),
    ("attn q+gate", 12288, 5120),
    ("gdn qkv", 10240, 5120),
    ("o / out_proj", 5120, 6144),
    ("attn k/v", 1024, 5120),
];
const BATCHES: [usize; 4] = [16, 32, 64, 128];
const COPIES: usize = 2;
const REPS: usize = 5;

#[derive(Clone, Copy, Debug)]
enum Arm {
    Scalar,
    Steel(SteelTile),
}

fn blob(rows: usize, cols: usize, seed: u64) -> (Vec<u8>, u64, u64) {
    let mut state = seed | 1;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as u32
    };
    let groups = rows * cols / 64;
    let mut out: Vec<u8> = (0..rows * cols / 2).map(|_| next() as u8).collect();
    let scales = out.len() as u64;
    for _ in 0..groups {
        out.extend_from_slice(&bf16::from_f32(0.01).to_bits().to_le_bytes());
    }
    let biases = out.len() as u64;
    for _ in 0..groups {
        out.extend_from_slice(&bf16::from_f32(-0.05).to_bits().to_le_bytes());
    }
    (out, scales, biases)
}

#[test]
#[ignore = "performance probe on a real Metal device; run explicitly"]
fn qwen38_int4_prefill_gemm_time_by_batch() {
    let mut context = MetalContext::new().expect("Metal device");
    let max_batch = *BATCHES.iter().max().unwrap();
    for (name, rows, cols) in SHAPES {
        let copies: Vec<_> = (0..COPIES)
            .map(|i| {
                let (data, scales, biases) = blob(rows, cols, 0xC0FFEE + i as u64);
                (context.new_buffer_with_data(&data), scales, biases)
            })
            .collect();
        let x_values: Vec<u8> = (0..max_batch * cols)
            .flat_map(|i| {
                f16::from_f32(((i % 13) as f32 - 6.0) / 64.0)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        let x = context.new_buffer_with_data(&x_values);
        let y = context.new_output_buffer((max_batch * rows * 2) as u64);
        let weight_bytes = (rows * cols / 2 + rows * cols / 64 * 4) as f64;
        println!(
            "{name} ({rows} x {cols}), {:.1} MB of weights",
            weight_bytes / 1e6
        );
        for batch in BATCHES {
            let arms = [
                Arm::Scalar,
                Arm::Steel(SteelTile::M32K32N32),
                Arm::Steel(SteelTile::M64K32N64),
            ];
            let mut samples = vec![Vec::new(); arms.len()];
            for rep in 0..=REPS {
                for (a, &arm) in arms.iter().enumerate() {
                    let t = autorelease_pool(|| {
                        let pass = context.begin_pass();
                        for (buffer, scales, biases) in &copies {
                            let w = Int4ResidentMatrix {
                                buffer,
                                weights_offset: 0,
                                scales_offset: *scales,
                                biases_offset: *biases,
                                rows,
                                cols,
                            };
                            match arm {
                                Arm::Scalar => {
                                    let mut done = 0;
                                    while done < batch {
                                        let take = (batch - done).min(MAX_BATCH_ROWS);
                                        encode_dequant_int4_gemm_resident(
                                            &mut context,
                                            &pass,
                                            &w,
                                            (&x, (done * cols * 2) as u64),
                                            (&y, (done * rows * 2) as u64),
                                            take,
                                        )
                                        .expect("scalar encodes");
                                        done += take;
                                    }
                                }
                                Arm::Steel(tile) => encode_dequant_int4_gemm_steel(
                                    &mut context,
                                    &pass,
                                    &w,
                                    (&x, 0),
                                    (&y, 0),
                                    batch,
                                    tile,
                                )
                                .expect("steel encodes"),
                            }
                        }
                        pass.commit_and_wait_with_gpu_time()
                    });
                    if rep > 0 {
                        samples[a].push(t * 1e3 / COPIES as f64);
                    }
                }
            }
            let median = |s: &mut Vec<f64>| {
                s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                s[s.len() / 2]
            };
            let base = median(&mut samples[0]);
            let flops = 2.0 * batch as f64 * rows as f64 * cols as f64;
            let line: Vec<String> = arms
                .iter()
                .zip(samples.iter_mut())
                .map(|(arm, s)| {
                    let m = median(s);
                    format!(
                        "{arm:?} {m:.3} ms ({:.1} TFLOPS, {:.2}x)",
                        flops / (m * 1e-3) / 1e12,
                        base / m
                    )
                })
                .collect();
            println!("  M={batch:>3}: {}", line.join(" | "));
        }
    }
}
