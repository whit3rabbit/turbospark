#![cfg(target_os = "macos")]
//! `attention_decode_indexed_biased_partial`, DASH's attention, against the
//! kernel it is copied from and against a CPU softmax with the bias applied.
//!
//! The bitwise case is what lets the indexed kernel's real-model record stand
//! for this one. The CPU case is what shows the bias does something, and that
//! it lands after the scale: upstream adds its mask to the scaled scores.

use half::f16;
use turbospark_gpu::{attention_decode_indexed, attention_decode_indexed_biased, MetalContext};

const HEAD_DIM: usize = 128;
const HEADS: usize = 10;
const SEQ: usize = 600;
const SCALE: f32 = 0.088_388_35;

/// Deterministic, roughly unit-scale values.
fn noise(n: usize, seed: u32) -> Vec<f16> {
    let mut x = seed.wrapping_mul(2_654_435_761) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            f16::from_f32((x as f32 / u32::MAX as f32) * 2.0 - 1.0)
        })
        .collect()
}

fn fixture() -> (Vec<f16>, Vec<f16>, Vec<f16>, Vec<u32>) {
    let q = noise(HEADS * HEAD_DIM, 1);
    let k = noise(SEQ * HEADS * HEAD_DIM, 2);
    let v = noise(SEQ * HEADS * HEAD_DIM, 3);
    // The real shape: 12 blocks of 32 plus a 128-token local window.
    let mut positions: Vec<u32> = (0..12u32).flat_map(|b| b * 32..(b + 1) * 32).collect();
    positions.extend(472u32..600);
    assert_eq!(positions.len(), 512);
    (q, k, v, positions)
}

fn cpu_reference(q: &[f16], k: &[f16], v: &[f16], positions: &[u32], bias: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; HEADS * HEAD_DIM];
    for head in 0..HEADS {
        let q_row = &q[head * HEAD_DIM..(head + 1) * HEAD_DIM];
        let scores: Vec<f32> = positions
            .iter()
            .zip(bias)
            .map(|(&p, b)| {
                let base = (p as usize * HEADS + head) * HEAD_DIM;
                let dot: f32 = q_row
                    .iter()
                    .zip(&k[base..base + HEAD_DIM])
                    .map(|(a, c)| a.to_f32() * c.to_f32())
                    .sum();
                dot * SCALE + b
            })
            .collect();
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let total: f32 = weights.iter().sum();
        for (&p, w) in positions.iter().zip(&weights) {
            let base = (p as usize * HEADS + head) * HEAD_DIM;
            for e in 0..HEAD_DIM {
                out[head * HEAD_DIM + e] += (w / total) * v[base + e].to_f32();
            }
        }
    }
    out
}

#[test]
fn zero_bias_is_bit_identical_to_the_indexed_kernel() {
    let mut ctx = MetalContext::new().unwrap();
    let (q, k, v, positions) = fixture();
    let bias = vec![0.0f32; positions.len()];
    let (h, n) = (HEAD_DIM as u32, HEADS as u32);
    let plain = attention_decode_indexed(&mut ctx, &q, &k, &v, &positions, h, n, n, SCALE).unwrap();
    let biased =
        attention_decode_indexed_biased(&mut ctx, &q, &k, &v, &positions, &bias, h, n, n, SCALE)
            .unwrap();
    let bits = |r: &[f16]| r.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&plain), bits(&biased));
}

#[test]
fn bias_matches_the_cpu_softmax_and_moves_the_output() {
    let mut ctx = MetalContext::new().unwrap();
    let (q, k, v, positions) = fixture();
    // Per block, as DASH's prior is, and zero on the local window.
    let bias: Vec<f32> = (0..positions.len())
        .map(|i| {
            if i < 384 {
                ((i / 32) as f32 - 5.5) * 0.7
            } else {
                0.0
            }
        })
        .collect();
    let (h, n) = (HEAD_DIM as u32, HEADS as u32);
    let got =
        attention_decode_indexed_biased(&mut ctx, &q, &k, &v, &positions, &bias, h, n, n, SCALE)
            .unwrap();
    let want = cpu_reference(&q, &k, &v, &positions, &bias);
    let unbiased = cpu_reference(&q, &k, &v, &positions, &vec![0.0; positions.len()]);
    let (mut worst, mut moved) = (0.0f32, 0.0f32);
    for ((g, w), u) in got.iter().zip(&want).zip(&unbiased) {
        assert!(g.to_f32().is_finite(), "non-finite attention output");
        worst = worst.max((g.to_f32() - w).abs());
        moved = moved.max((w - u).abs());
    }
    // FP16 output, so the tolerance is a few ULPs at this magnitude.
    assert!(
        worst < 2e-3,
        "GPU differs from the CPU reference by {worst}"
    );
    // A kernel ignoring the bias would pass the bound above if the bias were
    // too small to matter, so show that this one is not.
    assert!(moved > 20.0 * worst, "bias moved output only {moved}");
}

#[test]
fn only_listed_rows_are_read() {
    let mut ctx = MetalContext::new().unwrap();
    let (q, mut k, mut v, positions) = fixture();
    let row = HEADS * HEAD_DIM;
    for p in 0..SEQ {
        if !positions.contains(&(p as u32)) {
            k[p * row..(p + 1) * row].fill(f16::NAN);
            v[p * row..(p + 1) * row].fill(f16::NAN);
        }
    }
    let bias = vec![0.25f32; positions.len()];
    let (h, n) = (HEAD_DIM as u32, HEADS as u32);
    let got =
        attention_decode_indexed_biased(&mut ctx, &q, &k, &v, &positions, &bias, h, n, n, SCALE)
            .unwrap();
    assert!(got.iter().all(|x| x.to_f32().is_finite()));
}
