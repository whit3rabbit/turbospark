//! Small POD-to-bytes helpers shared by every kernel dispatch module, for
//! staging scalar `constant` buffer arguments and half-precision data
//! buffers.

use half::f16;

pub fn u32_bytes(v: &u32) -> &[u8] {
    // SAFETY: `u32` has no padding and any bit pattern is valid; the
    // returned slice borrows `v` for the caller's stack frame only.
    #[allow(unsafe_code)]
    unsafe {
        std::slice::from_raw_parts((v as *const u32).cast(), 4)
    }
}

pub fn f32_bytes(v: &f32) -> &[u8] {
    // SAFETY: same as `u32_bytes`, for `f32`.
    #[allow(unsafe_code)]
    unsafe {
        std::slice::from_raw_parts((v as *const f32).cast(), 4)
    }
}

pub fn half_slice_to_le_bytes(values: &[f16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for v in values {
        bytes.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    bytes
}

/// Byte-encodes a slice of raw 16-bit values (e.g. BF16 bit patterns, which
/// have no dedicated Rust storage type in this workspace — see
/// `mrefrust_compute::quant`) as little-endian bytes for a Metal buffer.
pub fn u16_slice_to_le_bytes(values: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

pub fn read_half_buffer(buffer: &metal::Buffer, len: usize) -> Vec<f16> {
    let ptr = buffer.contents() as *const u16;
    // SAFETY: `buffer` was allocated with `len * size_of::<u16>()` bytes in
    // shared storage mode by the caller, and the GPU command buffer that
    // wrote it has already been waited on (`wait_until_completed`), so the
    // CPU-visible contents are complete and valid for `len` `u16` reads.
    #[allow(unsafe_code)]
    let bits = unsafe { std::slice::from_raw_parts(ptr, len) };
    bits.iter().map(|&b| f16::from_bits(b)).collect()
}

pub fn read_f32_buffer(buffer: &metal::Buffer, len: usize) -> Vec<f32> {
    let ptr = buffer.contents() as *const f32;
    // SAFETY: same contract as `read_half_buffer`, with 4-byte elements.
    #[allow(unsafe_code)]
    let values = unsafe { std::slice::from_raw_parts(ptr, len) };
    values.to_vec()
}
