//! Metal buffer memory read and write helpers (`write_buffer_bytes`, `read_buffer_f16`).

/// Host-writes `bytes` into a shared-storage buffer at `offset` (a plain
/// memcpy into unified memory). Callers sequence this against GPU work
/// themselves: write before committing the pass that reads it.
pub fn write_buffer_bytes(buffer: &metal::Buffer, offset: usize, bytes: &[u8]) {
    // AGENTS.md/CLAUDE.md B5: checked, not `offset + bytes.len()` -- this
    // crate's own root `Cargo.toml` deliberately carries no
    // `[profile.release]`, so `overflow-checks` is off in release and a
    // wrapped sum would pass a bounds assert ahead of the unsafe copy below.
    assert!(offset.checked_add(bytes.len()).unwrap() <= buffer.length() as usize);
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
    assert!(offset.checked_add(len).unwrap() <= buffer.length() as usize);
    // SAFETY: bounds asserted above against a live shared-storage
    // `MTLBuffer`'s allocation; `u8` has no alignment requirement and no
    // invalid bit patterns.
    #[allow(unsafe_code)]
    let bytes =
        unsafe { std::slice::from_raw_parts((buffer.contents() as *const u8).add(offset), len) };
    bytes.to_vec()
}

/// Borrows `count` halfs of a shared-storage buffer starting at `byte_offset`
/// as a slice, with no copy at all. Only valid after the pass that wrote them
/// completed, and only for as long as `buffer` is alive and no GPU work is
/// writing the range -- which is why this is private and the two public
/// wrappers below copy.
///
/// `half::f16` is `#[repr(transparent)]` over `u16`, so this is the same
/// reinterpretation `read_buffer_bytes` does one function up, at 2-byte
/// elements instead of 1.
#[allow(unsafe_code)]
unsafe fn f16_slice(buffer: &metal::Buffer, byte_offset: usize, count: usize) -> &[half::f16] {
    // SAFETY (caller-checked bounds): `u16` has no invalid bit patterns, and
    // the 2-byte alignment holds for any offset this crate binds halfs at.
    std::slice::from_raw_parts(
        (buffer.contents() as *const u8).add(byte_offset) as *const half::f16,
        count,
    )
}

/// Host-reads halfs from a shared-storage buffer starting at `byte_offset`
/// INTO a caller-owned slice. Only valid after the pass that wrote them
/// completed.
///
/// This is the form the decode path wants and `read_buffer_f16` is not. Every
/// family's per-token head read is `logits.copy_from_slice(&read_buffer_f16(..))`
/// at the full vocabulary, so the owned-`Vec` form costs a 512 KiB allocation
/// and a second 512 KiB copy per decoded token, for a destination that already
/// exists and is already the right length.
pub fn read_buffer_f16_into(buffer: &metal::Buffer, byte_offset: usize, dst: &mut [half::f16]) {
    let dst_bytes = dst.len().checked_mul(2).unwrap();
    assert!(byte_offset.checked_add(dst_bytes).unwrap() <= buffer.length() as usize);
    // SAFETY: bounds asserted above.
    #[allow(unsafe_code)]
    let src = unsafe { f16_slice(buffer, byte_offset, dst.len()) };
    dst.copy_from_slice(src);
}

/// Host-reads `count` halfs from a shared-storage buffer starting at
/// `byte_offset`. Only valid after the pass that wrote them completed.
///
/// Prefer `read_buffer_f16_into` on any path that runs per token; this form
/// allocates. It remains the convenient one for tests and for the parity
/// dispatches, which read a fresh output buffer once per call.
pub fn read_buffer_f16(buffer: &metal::Buffer, byte_offset: usize, count: usize) -> Vec<half::f16> {
    let count_bytes = count.checked_mul(2).unwrap();
    assert!(byte_offset.checked_add(count_bytes).unwrap() <= buffer.length() as usize);
    // SAFETY: bounds asserted above.
    #[allow(unsafe_code)]
    let src = unsafe { f16_slice(buffer, byte_offset, count) };
    src.to_vec()
}
