//! Metal buffer memory read and write helpers (`write_buffer_bytes`, `read_buffer_f16`).

/// Host-writes `bytes` into a shared-storage buffer at `offset` (a plain
/// memcpy into unified memory). Callers sequence this against GPU work
/// themselves: write before committing the pass that reads it.
pub fn write_buffer_bytes(buffer: &metal::Buffer, offset: usize, bytes: &[u8]) {
    assert!(offset + bytes.len() <= buffer.length() as usize);
    // SAFETY: bounds asserted above against a live shared-storage
    // `MTLBuffer`'s allocation; the source and destination never overlap
    // (one is a Rust slice, the other a Metal allocation).
    #[allow(unsafe_code)]
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            (buffer.contents() as *mut u8).add(offset),
            bytes.len(),
        );
    }
}

/// Host-reads `len` raw bytes from a shared-storage buffer at `offset`.
/// Dtype-blind on purpose: the recurrent-state snapshot behind a
/// speculative rollback copies FP32 state and FP16 conv rows through the
/// same path and never interprets either.
pub fn read_buffer_bytes(buffer: &metal::Buffer, offset: usize, len: usize) -> Vec<u8> {
    assert!(offset + len <= buffer.length() as usize);
    // SAFETY: bounds asserted above against a live shared-storage
    // `MTLBuffer`'s allocation; `u8` has no alignment requirement and no
    // invalid bit patterns.
    #[allow(unsafe_code)]
    let bytes =
        unsafe { std::slice::from_raw_parts((buffer.contents() as *const u8).add(offset), len) };
    bytes.to_vec()
}

/// Host-reads `count` halfs from a shared-storage buffer starting at
/// `byte_offset`. Only valid after the pass that wrote them completed.
pub fn read_buffer_f16(buffer: &metal::Buffer, byte_offset: usize, count: usize) -> Vec<half::f16> {
    assert!(byte_offset + count * 2 <= buffer.length() as usize);
    // SAFETY: bounds asserted above; u16 has no invalid bit patterns and
    // the 2-byte alignment holds for any offset this crate binds halfs at.
    #[allow(unsafe_code)]
    let bits = unsafe {
        std::slice::from_raw_parts(
            (buffer.contents() as *const u8).add(byte_offset) as *const u16,
            count,
        )
    };
    bits.iter().map(|&b| half::f16::from_bits(b)).collect()
}
