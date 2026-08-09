//! Readahead advice for a `pread`-backed file descriptor. Ported from
//! `Infrastructure/Streaming/RDAdvice.swift`.
//!
//! macOS's `F_RDADVISE` `fcntl` is the only unsafe, platform-specific call
//! in this crate; everywhere else this is a documented no-op, matching the
//! workspace's cross-cutting rule that `streaming` is one of the few crates
//! allowed unsafe code and platform `cfg`s.

use std::time::Instant;

/// Result of an `F_RDADVISE` readahead call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RdAdviceCallResult {
    /// Total bytes requested for readahead (clipped to `i32::MAX`).
    pub requested_bytes: u64,
    /// Whether the underlying `fcntl` call succeeded.
    pub succeeded: bool,
    /// Time spent in the `fcntl` call in nanoseconds.
    pub elapsed_nanos: u64,
}

/// Clips requested byte count to `i32::MAX` for kernel call limits.
pub fn clipped_byte_count(byte_count: u64) -> u64 {
    byte_count.min(i32::MAX as u64)
}

/// Issues an `F_RDADVISE` readahead hint for a file descriptor range on macOS (no-op elsewhere).
#[cfg(target_os = "macos")]
pub fn call(fd: std::os::unix::io::RawFd, offset: u64, byte_count: u64) -> RdAdviceCallResult {
    let clipped_count = clipped_byte_count(byte_count);
    let start = Instant::now();

    #[repr(C)]
    struct Radvisory {
        ra_offset: libc::off_t,
        ra_count: libc::c_int,
    }
    const F_RDADVISE: libc::c_int = 44;

    let mut advice = Radvisory {
        ra_offset: offset as libc::off_t,
        ra_count: clipped_count as libc::c_int,
    };
    // SAFETY: `fd` is a valid, caller-owned file descriptor for the
    // lifetime of this call; `advice` is a plain-old-data struct matching
    // Darwin's `radvisory` layout, and `F_RDADVISE` reads it without
    // retaining the pointer past the call.
    #[allow(unsafe_code)]
    let succeeded = unsafe { libc::fcntl(fd, F_RDADVISE, &mut advice as *mut Radvisory) == 0 };

    RdAdviceCallResult {
        requested_bytes: clipped_count,
        succeeded,
        elapsed_nanos: start.elapsed().as_nanos() as u64,
    }
}

/// Issues an `F_RDADVISE` readahead hint for a file descriptor range on macOS (no-op elsewhere).
#[cfg(not(target_os = "macos"))]
pub fn call(_fd: i32, _offset: u64, byte_count: u64) -> RdAdviceCallResult {
    let clipped_count = clipped_byte_count(byte_count);
    RdAdviceCallResult {
        requested_bytes: clipped_count,
        succeeded: false,
        elapsed_nanos: 0,
    }
}
