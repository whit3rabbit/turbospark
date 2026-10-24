//! MEASUREMENT ONLY, NOT FOR COMMIT. Model-free cross-check for the qwen38
//! prefill width sweep: GPU seconds of every INT4 projection one prompt of
//! `PROBE_TOKENS` tokens issues at micro-batch width W, at Qwen3.8-27B's real
//! shapes and layer counts, using the same Steel-or-scalar rule as
//! `real_forward_dispatch::encode_gemm_rows`. Each sample runs the matrix over
//! two distinct weight copies so no copy is cache-resident from the previous
//! dispatch; arms interleave per rep and the first rep is discarded.
//!
//! ```sh
//! PROBE_TOKENS=2727 cargo test -p turbospark-gpu --release \
//!   --test qwen38_prefill_width_proj_sum -- --ignored --nocapture
//! ```
#![cfg(target_os = "macos")]

use half::{bf16, f16};
use turbospark_gpu::{
    autorelease_pool, encode_dequant_int4_gemm_resident, encode_dequant_int4_gemm_steel,
    steel_supports, Int4ResidentMatrix, MetalContext, SteelTile, MAX_BATCH_ROWS,
};

/// (name, output rows, input width, matrices per micro-batch). 64 layers:
/// 48 GDN, 16 full attention. Gate and up are separate GEMMs.
const SHAPES: [(&str, usize, usize, usize); 9] = [
    ("mlp gate+up", 17408, 5120, 128),
    ("mlp down", 5120, 17408, 64),
    ("attn q+gate", 12288, 5120, 16),
    ("attn k+v", 1024, 5120, 32),
    ("attn o", 5120, 6144, 16),
    ("gdn qkv", 10240, 5120, 48),
    ("gdn z", 6144, 5120, 48),
    ("gdn a+b", 48, 5120, 96),
    ("gdn out", 5120, 6144, 48),
];
const WIDTHS: [usize; 6] = [64, 128, 256, 512, 1024, 2048];
const STEEL_MIN_ROWS: usize = 32;
const COPIES: usize = 2;
const REPS: usize = 5;

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
fn qwen38_prefill_projection_seconds_by_width() {
    let tokens: usize = std::env::var("PROBE_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2727);
    let mut context = MetalContext::new().expect("Metal device");
    // Every micro-batch size a prompt of `tokens` produces at each width.
    let mut sizes: Vec<usize> = WIDTHS
        .iter()
        .flat_map(|&w| [w, tokens % w])
        .filter(|&m| m > 0)
        .collect();
    sizes.sort_unstable();
    sizes.dedup();
    let max_m = *sizes.last().unwrap();
    // per_matrix[shape][size] = median GPU seconds of ONE matrix at m rows.
    let mut per_matrix = vec![vec![0.0f64; sizes.len()]; SHAPES.len()];
    for (s, &(name, rows, cols, _)) in SHAPES.iter().enumerate() {
        let copies: Vec<_> = (0..COPIES)
            .map(|i| {
                let (data, scales, biases) = blob(rows, cols, 0xC0FFEE + i as u64);
                (context.new_buffer_with_data(&data), scales, biases)
            })
            .collect();
        let x_values: Vec<u8> = (0..max_m * cols)
            .flat_map(|i| {
                f16::from_f32(((i % 13) as f32 - 6.0) / 64.0)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        let x = context.new_buffer_with_data(&x_values);
        let y = context.new_output_buffer((max_m * rows * 2) as u64);
        let mut samples = vec![Vec::new(); sizes.len()];
        for rep in 0..=REPS {
            for (k, &m) in sizes.iter().enumerate() {
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
                        if m >= STEEL_MIN_ROWS && steel_supports(&w, SteelTile::M32K32N32) {
                            encode_dequant_int4_gemm_steel(
                                &mut context,
                                &pass,
                                &w,
                                (&x, 0),
                                (&y, 0),
                                m,
                                SteelTile::M32K32N32,
                            )
                            .expect("steel encodes");
                        } else {
                            let mut done = 0;
                            while done < m {
                                let take = (m - done).min(MAX_BATCH_ROWS);
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
                    }
                    pass.commit_and_wait_with_gpu_time()
                });
                if rep > 0 {
                    samples[k].push(t / COPIES as f64);
                }
            }
        }
        for (k, v) in samples.iter_mut().enumerate() {
            v.sort_by(f64::total_cmp);
            per_matrix[s][k] = v[v.len() / 2];
        }
        let row: Vec<String> = sizes
            .iter()
            .zip(&per_matrix[s])
            .map(|(m, t)| format!("m={m}:{:.3}ms", t * 1e3))
            .collect();
        println!("{name:<12} {rows}x{cols} per matrix: {}", row.join(" "));
    }
    let at = |m: usize| sizes.iter().position(|&s| s == m).unwrap();
    println!("\nprompt of {tokens} tokens, projection GPU seconds by width:");
    for &w in &WIDTHS {
        let full = tokens / w;
        let tail = tokens % w;
        let mut total = 0.0;
        let mut parts = Vec::new();
        for (s, &(name, _, _, count)) in SHAPES.iter().enumerate() {
            let mut sec = full as f64 * per_matrix[s][at(w)] * count as f64;
            if tail > 0 {
                sec += per_matrix[s][at(tail)] * count as f64;
            }
            total += sec;
            parts.push(format!("{name}={sec:.3}"));
        }
        println!(
            "W={w:>3}: {total:.3} s ({} full + tail {tail})  {}",
            full,
            parts.join(" ")
        );
    }
}
