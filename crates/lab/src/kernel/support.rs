//! Shared pieces for kernel benches: synthetic weights in the resident
//! layouts, and the sample discipline every bench here follows.
//!
//! The discipline is the in-tree probes' (`crates/gpu/tests/*_probe.rs`):
//! distinct weight copies per sample so nothing is still cache-resident from
//! the previous dispatch, arms alternated within each repetition, the first
//! repetition discarded, GPU time from the command buffer's own timestamps.

use half::{bf16, f16};

/// An INT4 affine blob in the resident layout: packed codes, then BF16
/// scales, then BF16 biases, one per 64-element group. Returns the bytes and
/// the scales and biases offsets.
pub fn int4_blob(rows: usize, cols: usize, seed: u64) -> (Vec<u8>, u64, u64) {
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

/// Little-endian f16 bytes of `n` deterministic activations.
pub fn f16_activations(n: usize, seed: u32) -> Vec<u8> {
    (0..n)
        .flat_map(|i| {
            f16::from_f32((((i as u32).wrapping_add(seed) % 13) as f32 - 6.0) / 64.0)
                .to_bits()
                .to_le_bytes()
        })
        .collect()
}

pub fn f16_bytes(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values
        .flat_map(|v| f16::from_f32(v).to_bits().to_le_bytes())
        .collect()
}

/// How many distinct weight copies to rotate through so a sample's weights
/// exceed what the system-level cache can hold: at least `min_bytes` across
/// copies, between 2 and 8 copies.
pub fn copies_for(bytes_per_copy: usize, min_bytes: usize) -> usize {
    min_bytes.div_ceil(bytes_per_copy.max(1)).clamp(2, 8)
}

pub fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2]
}
