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

/// An INT8 affine blob in the resident layout: `rows * cols` code bytes,
/// then BF16 scales, then BF16 biases, one per 64-element group.
pub fn int8_blob(rows: usize, cols: usize, seed: u64) -> (Vec<u8>, u64, u64) {
    let mut state = seed | 1;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as u32
    };
    let groups = rows * cols / 64;
    let mut out: Vec<u8> = (0..rows * cols).map(|_| next() as u8).collect();
    let scales = out.len() as u64;
    for _ in 0..groups {
        out.extend_from_slice(&bf16::from_f32(0.0008).to_bits().to_le_bytes());
    }
    let biases = out.len() as u64;
    for _ in 0..groups {
        out.extend_from_slice(&bf16::from_f32(-0.1).to_bits().to_le_bytes());
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

/// Untimed GPU work committed right before a timed pass, so the pass starts
/// on a device that is up to speed.
///
/// Why: a pass's per-dispatch time is set by the state the device is in when
/// the pass starts, and from idle that state is slow: a 64-dispatch pass of
/// the 48-row GEMV read 18.5 us per dispatch from a cold device against 3.2
/// us on a busy one, and a 16384-dispatch pass started cold stayed slow for
/// its whole 280 ms. Only tens of milliseconds of continuous work queued
/// ahead of the pass bring it up: primers of 1024 and 4096 tiny dispatches
/// did not move the number, 16384 did (`tests/int4_one_row_floor_probe.rs`,
/// `cold_device_primer_and_pass_length`). A bench that times tiny passes with
/// host work between them (building the next pass, reading a hash) measures
/// that state, not the kernel; M = 1 of `int4_matmul` was exactly that, and
/// the 8-row pass that followed it read warm.
///
/// The primer runs one-row GEMVs over its own small matrix, never the
/// measured weights, so it warms the device and not the cache the copies are
/// there to defeat.
#[cfg(target_os = "macos")]
pub struct Primer {
    buffer: gpu::MetalBuffer,
    scales: u64,
    biases: u64,
    x: gpu::MetalBuffer,
    y: gpu::MetalBuffer,
}

/// One-row GEMV dispatches per primer: about 50 ms of GPU work on the M4
/// Max, the shortest length that brought a cold device to speed.
#[cfg(target_os = "macos")]
pub const PRIMER_DISPATCHES: usize = 16384;
#[cfg(target_os = "macos")]
const PRIMER_ROWS: usize = 64;
#[cfg(target_os = "macos")]
const PRIMER_COLS: usize = 5120;

#[cfg(target_os = "macos")]
impl Primer {
    pub fn new(context: &gpu::MetalContext) -> Self {
        let (data, scales, biases) = int4_blob(PRIMER_ROWS, PRIMER_COLS, 0x9A1E_0001);
        Self {
            buffer: context.new_buffer_with_data(&data),
            scales,
            biases,
            x: context.new_buffer_with_data(&f16_activations(PRIMER_COLS, 3)),
            y: context.new_output_buffer((PRIMER_ROWS * 2) as u64),
        }
    }

    /// Encode and commit the primer without waiting. The caller commits its
    /// timed pass next on the same queue, so the pass starts as the primer
    /// ends, and waits on the returned handle afterwards to read the
    /// primer's own GPU time.
    pub fn commit(&self, context: &mut gpu::MetalContext) -> Result<gpu::CommittedPass, String> {
        let pass = context.begin_pass_labeled("primer");
        let w = gpu::Int4ResidentMatrix {
            buffer: &self.buffer,
            weights_offset: 0,
            scales_offset: self.scales,
            biases_offset: self.biases,
            rows: PRIMER_ROWS,
            cols: PRIMER_COLS,
        };
        for _ in 0..PRIMER_DISPATCHES {
            gpu::encode_dequant_int4_gemv_resident(context, &pass, &w, (&self.x, 0), (&self.y, 0))
                .map_err(|e| e.to_string())?;
        }
        Ok(pass.commit())
    }

    /// What the record says about the primer.
    pub fn describe(primer_ms: &mut [f64]) -> serde_json::Value {
        serde_json::json!({
            "dispatches": PRIMER_DISPATCHES,
            "shape": format!("{PRIMER_ROWS}x{PRIMER_COLS}"),
            "gpu_ms_median": if primer_ms.is_empty() { f64::NAN } else { median(primer_ms) },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_rotate_between_two_and_eight() {
        assert_eq!(copies_for(1, 256 << 20), 8);
        assert_eq!(copies_for(1 << 30, 256 << 20), 2);
        assert_eq!(copies_for(64 << 20, 256 << 20), 4);
    }
}
