#![cfg(target_os = "macos")]
//! What do Qwen3.8's 48-row GDN `in_proj_a`/`in_proj_b` cost per prefill
//! micro-batch on the scalar GEMM (16-row slices, what `encode_gemm_rows`
//! runs today) against the Steel EDGE entry (`aligned_N = false`)?
//!
//! ```sh
//! cargo test -p turbospark-gpu --test steel_edge_n48_bench --release -- --ignored --nocapture
//! ```
//!
//! A pool of distinct weight copies keeps every call cold. The host is
//! shared, so read the ratio, not the microseconds.

use turbospark_gpu::{
    autorelease_pool, encode_dequant_int4_gemm_resident, encode_dequant_int4_gemm_steel_edge,
    encode_dequant_int4_mma8_resident, Int4ResidentMatrix, MetalContext, INT4_MMA8_MAX_ROWS,
    MAX_BATCH_ROWS,
};

const ROWS: usize = 48;
const COLS: usize = 5120;
const POOL: usize = 64;
const ROUNDS: usize = 7;

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    ScalarSlices,
    SteelEdge,
    Mma8Calls,
}

fn time_arm(
    context: &mut MetalContext,
    weights: &metal::Buffer,
    stride: u64,
    x: &metal::Buffer,
    y: &metal::Buffer,
    batch: usize,
    arm: Arm,
) -> f64 {
    let packed = (ROWS * COLS / 2) as u64;
    let sb = (ROWS * COLS / 64 * 2) as u64;
    let align = |v: u64| v.div_ceil(256) * 256;
    let scales = align(packed);
    let biases = align(scales + sb);
    let calls = POOL * 4;
    let seconds = autorelease_pool(|| {
        let pass = context.begin_pass();
        for i in 0..calls {
            let base = stride * (i % POOL) as u64;
            let w = Int4ResidentMatrix {
                buffer: weights,
                weights_offset: base,
                scales_offset: base + scales,
                biases_offset: base + biases,
                rows: ROWS,
                cols: COLS,
            };
            match arm {
                Arm::ScalarSlices => {
                    let mut done = 0;
                    while done < batch {
                        let take = (batch - done).min(MAX_BATCH_ROWS);
                        encode_dequant_int4_gemm_resident(
                            context,
                            &pass,
                            &w,
                            (x, (done * COLS * 2) as u64),
                            (y, (done * ROWS * 2) as u64),
                            take,
                        )
                        .unwrap();
                        done += take;
                    }
                }
                Arm::Mma8Calls => {
                    let mut done = 0;
                    while done < batch {
                        let take = (batch - done).min(INT4_MMA8_MAX_ROWS);
                        encode_dequant_int4_mma8_resident(
                            context,
                            &pass,
                            &w,
                            (x, (done * COLS * 2) as u64),
                            (y, (done * ROWS * 2) as u64),
                            take,
                        )
                        .unwrap();
                        done += take;
                    }
                }
                Arm::SteelEdge => {
                    encode_dequant_int4_gemm_steel_edge(context, &pass, &w, (x, 0), (y, 0), batch)
                        .unwrap();
                }
            }
        }
        pass.commit_and_wait_with_gpu_time()
    });
    seconds * 1e6 / calls as f64
}

#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn n48_scalar_slices_against_steel_edge() {
    let mut context = MetalContext::new().expect("Metal device");
    let packed = (ROWS * COLS / 2) as u64;
    let sb = (ROWS * COLS / 64 * 2) as u64;
    let align = |v: u64| v.div_ceil(256) * 256;
    let stride = align(align(align(packed) + sb) + sb);
    let weights = context.new_output_buffer(stride * POOL as u64);
    let widths = [32usize, 64, 128, 256, 512];
    let maxw = *widths.iter().max().unwrap();
    let x = context.new_output_buffer((maxw * COLS * 2) as u64);
    let y = context.new_output_buffer((maxw * ROWS * 2) as u64);
    let arms = [Arm::ScalarSlices, Arm::SteelEdge, Arm::Mma8Calls];

    for &b in &widths {
        for arm in arms {
            time_arm(&mut context, &weights, stride, &x, &y, b, arm);
        }
    }
    println!("\nmicroseconds of GPU time per 48x5120 projection, median of {ROUNDS}");
    for &b in &widths {
        let mut s = vec![Vec::new(); arms.len()];
        for round in 0..ROUNDS {
            for k in 0..arms.len() {
                let a = (k + round) % arms.len();
                s[a].push(time_arm(&mut context, &weights, stride, &x, &y, b, arms[a]));
            }
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let m: Vec<f64> = s.iter_mut().map(med).collect();
        println!(
            "  batch {b:>4}: scalar slices {:>8.1} us   steel edge {:>8.1} us ({:.2}x)   mma8 x{} calls {:>8.1} us ({:.2}x)",
            m[0],
            m[1],
            m[0] / m[1],
            b.div_ceil(INT4_MMA8_MAX_ROWS),
            m[2],
            m[0] / m[2]
        );
    }
}
