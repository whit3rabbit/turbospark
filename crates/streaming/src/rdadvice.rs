//! Readahead advice for a `pread`-backed file descriptor. Ported from
//! `Infrastructure/Streaming/RDAdvice.swift`.
//!
//! macOS's `F_RDADVISE` `fcntl` is the only unsafe, platform-specific call
//! in this crate; everywhere else this is a documented no-op, matching the
//! workspace's cross-cutting rule that `streaming` is one of the few crates
//! allowed unsafe code and platform `cfg`s.

// Only the macOS arm times anything; the no-op arm reports zero. Gated so
// a non-macOS build of this crate is warning-clean, which it could not be
// checked for until `libc` stopped being a macOS-only dependency.
#[cfg(target_os = "macos")]
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

/// Issues a `posix_fadvise(WILLNEED)` readahead hint on Linux, the
/// platform's analogue of the Darwin arm above (ROADMAP P3.5). WILLNEED is
/// a non-blocking hint, which is what the caller wants -- the read itself
/// happens on the chunk plan's schedule, and DONTNEED would fight the
/// mapped-residency mode for the page cache.
#[cfg(target_os = "linux")]
pub fn call(fd: i32, offset: u64, byte_count: u64) -> RdAdviceCallResult {
    let clipped_count = clipped_byte_count(byte_count);
    // SAFETY: `fd` is a valid, caller-owned descriptor; the remaining
    // arguments are plain integers and the call retains nothing.
    let succeeded = unsafe {
        libc::posix_fadvise(
            fd,
            offset as libc::off_t,
            clipped_count as libc::off_t,
            libc::POSIX_FADV_WILLNEED,
        ) == 0
    };
    RdAdviceCallResult {
        requested_bytes: clipped_count,
        succeeded,
        // fadvise has no timed sibling here; the macOS arm's timing exists
        // because F_RDADVISE is on the hot miss path and its cost was
        // questioned. Measure before adding one on this side.
        elapsed_nanos: 0,
    }
}

/// No-op off macOS AND off Linux (Windows and friends): documented rather
/// than silent, matching the crate header's rule.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn call(_fd: i32, _offset: u64, byte_count: u64) -> RdAdviceCallResult {
    let clipped_count = clipped_byte_count(byte_count);
    RdAdviceCallResult {
        requested_bytes: clipped_count,
        succeeded: false,
        elapsed_nanos: 0,
    }
}
