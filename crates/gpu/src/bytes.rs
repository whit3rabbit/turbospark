//! Small POD-to-bytes helpers shared by every kernel dispatch module, for
//! staging scalar `constant` buffer arguments and half-precision data
//! buffers.

use half::f16;

/// Reinterprets a reference to a `u32` scalar as a 4-byte slice.
pub fn u32_bytes(v: &u32) -> &[u8] {
    // SAFETY: `u32` has no padding and any bit pattern is valid; the
    // returned slice borrows `v` for the caller's stack frame only.
    #[allow(unsafe_code)]
    unsafe {
        std::slice::from_raw_parts((v as *const u32).cast(), 4)
    }
}

/// Reinterprets a reference to an `f32` scalar as a 4-byte slice.
pub fn f32_bytes(v: &f32) -> &[u8] {
    // SAFETY: same as `u32_bytes`, for `f32`.
    #[allow(unsafe_code)]
    unsafe {
        std::slice::from_raw_parts((v as *const f32).cast(), 4)
    }
}

/// Encodes a slice of `f16` values as a vector of little-endian bytes.
pub fn half_slice_to_le_bytes(values: &[f16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for v in values {
        bytes.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    bytes
}

/// Byte-encodes a slice of raw 16-bit values (e.g. BF16 bit patterns, which
/// have no dedicated Rust storage type in this workspace — see
/// `turbospark_compute::quant`) as little-endian bytes for a Metal buffer.
pub fn u16_slice_to_le_bytes(values: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Reads `len` `f16` elements back from a completed Metal CPU/GPU shared buffer.
pub fn read_half_buffer(buffer: &metal::Buffer, len: usize) -> Vec<f16> {
    let ptr = buffer.contents() as *const u16;
    // SAFETY: `buffer` was allocated with `len * size_of::<u16>()` bytes in
    // shared storage mode by the caller, and the GPU command buffer that
    // wrote it has already been waited on (`wait_until_completed`), so the
    // CPU-visible contents are complete and valid for `len` `u16` reads.
    // `f16` is `#[repr(transparent)]` over `u16`, so the element-wise
    // `from_bits` map this used to do was a memcpy written out longhand.
    #[allow(unsafe_code)]
    let values = unsafe { std::slice::from_raw_parts(ptr.cast::<f16>(), len) };
    values.to_vec()
}

/// Reads `len` `f32` elements back from a completed Metal CPU/GPU shared buffer.
pub fn read_f32_buffer(buffer: &metal::Buffer, len: usize) -> Vec<f32> {
    read_f32_buffer_at(buffer, 0, len)
}

/// [`read_f32_buffer`] starting `first` ELEMENTS in, for a buffer holding
/// one row per token of a prefill chunk. The offset is in elements rather
/// than bytes so a caller cannot pass a byte offset by mistake and read a
/// misaligned window that still returns finite numbers.
pub fn read_f32_buffer_at(buffer: &metal::Buffer, first: usize, len: usize) -> Vec<f32> {
    assert!(
        (first + len) * std::mem::size_of::<f32>() <= buffer.length() as usize,
        "read_f32_buffer_at reads past the buffer"
    );
    let ptr = buffer.contents() as *const f32;
    // SAFETY: same contract as `read_half_buffer`, with 4-byte elements, and
    // the range is asserted to be inside the allocation above.
    #[allow(unsafe_code)]
    let values = unsafe { std::slice::from_raw_parts(ptr.add(first), len) };
    values.to_vec()
}
