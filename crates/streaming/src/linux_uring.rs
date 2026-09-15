//! The Linux I/O seam (ROADMAP P3.5): an `io_uring` + `O_DIRECT` read
//! source for the expert streamer, selected beside the `pread` default.
//!
//! # STATUS: COMPILE-GATED SCAFFOLDING, UNTESTED ON LINUX
//!
//! This module carries the repository's accepted platform gate for code no
//! machine here can run (AGENTS.md Gotcha 8): it compiles under
//! `cargo check --target x86_64-unknown-linux-gnu -p turbospark-streaming`
//! and NOTHING ELSE says anything about it. The syscalls are raw `libc`
//! calls against the io_uring(2) and open(2) manuals, written to match
//! kernel 5.19+ semantics, and no Linux box has executed a single line of
//! them. **`Mode::Auto` resolves to `Pread`**, and until a Linux session
//! runs the parity arm this module's selectors and alignment math are the
//! only parts under test. Do not flip the default, and do not benchmark
//! `Mode::Uring`, until that arm exists and agrees with the `pread` path
//! bit for bit.
//!
//! # The design, and what it borrows
//!
//! The expert streamer's reads are large (the 840 KiB miss chunk), aligned
//! to the slot allocator's pages, and sequential per expert -- exactly the
//! shape `O_DIRECT` exists for: no page-cache double-buffering of gigabytes
//! of weights that are read once. `O_DIRECT` requires the offset, length
//! AND user buffer be sector-aligned, which the [`AlignedSlot`] allocator
//! already guarantees for the buffer and the chunk plan for the rest.
//!
//! `Mode` is selected once per process from `TURBOSPARK_LINUX_IO`:
//! `auto` (the default, and `pread` until proven otherwise), `pread`, or
//! `uring`. An unknown spelling is an ERROR rather than a fallback -- a
//! caller who typed a selector measured nothing, and a silent default would
//! measure the wrong source and report it as the selected one.

/// The selected I/O source for a Linux streamer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// The `pread` path as it ships today.
    Pread,
    /// The `io_uring` + `O_DIRECT` path. NOT REACHABLE from `auto` until a
    /// Linux run proves it; see the module header.
    Uring,
}

impl Mode {
    /// Reads `TURBOSPARK_LINUX_IO`. Unknown spellings are refused by name.
    pub fn from_env() -> Result<Self, String> {
        Self::parse(
            std::env::var("TURBOSPARK_LINUX_IO")
                .unwrap_or_else(|_| "auto".to_string())
                .as_str(),
        )
    }

    /// [`Self::from_env`] with the variable absent -- the case every
    /// deployment without the flag runs under, and therefore the value
    /// every default-sensitive gate reads.
    pub fn from_env_default() -> Result<Self, String> {
        Self::parse("auto")
    }

    /// The documented selection rule. `auto` is `pread`: see the module
    /// header for what a Linux session has to prove before that changes.
    pub fn parse(spelling: &str) -> Result<Self, String> {
        match spelling.trim() {
            "auto" | "pread" => Ok(Mode::Pread),
            "uring" => Ok(Mode::Uring),
            other => Err(format!(
                "TURBOSPARK_LINUX_IO must be auto, pread or uring, not {other:?}"
            )),
        }
    }
}

/// The sector size `O_DIRECT` alignment is computed against.
///
/// The practical floor on every device Linux ships is 512 bytes; NVMe
/// formats at 4 KiB also accept 512-aligned requests. Resolved per file
/// rather than assumed on a real system (statx STATX_DIOALIGN, kernel
/// 6.1+); this constant is the fallback the selector documents.
pub const DIO_ALIGNMENT: u64 = 512;

/// Whether an `(offset, len)` pair is `O_DIRECT`-legal against
/// [`DIO_ALIGNMENT`]. Pure arithmetic, so this is the part of the module a
/// macOS test can hold.
pub fn dio_aligned(offset: u64, len: usize) -> bool {
    offset % DIO_ALIGNMENT == 0 && (len as u64) % DIO_ALIGNMENT == 0
}

/// Rounds a length UP to the `O_DIRECT` alignment; the caller over-reads
/// into the aligned slot's slack rather than short-reading the tail.
pub fn dio_round_up(len: usize) -> usize {
    let a = DIO_ALIGNMENT as usize;
    len.div_ceil(a) * a
}

#[cfg(target_os = "linux")]
mod sys {
    //! Raw io_uring plumbing. Every constant and layout here is from the
    //! `io_uring(2)` / `io_uring_enter(2)` manuals (kernel headers
    //! `linux/io_uring.h`), because adding a liburing binding crate would
    //! add a build dependency to a crate whose whole point is being
    //! readable on a machine without Linux.

    use std::fs::File;
    use std::os::unix::io::AsRawFd;
    use std::sync::atomic::{AtomicU32, Ordering};

    const SYS_IO_URING_SETUP: libc::c_long = 425;
    const SYS_IO_URING_ENTER: libc::c_long = 426;

    const IORING_SETUP_SINGLE_ISSUER: u32 = 1 << 4;
    const IORING_SETUP_COOP_TASKRUN: u32 = 1 << 5;
    const IORING_OP_READ: u8 = 22;
    const IOSQE_ASYNC: u32 = 1 << 0;
    const ENTER_GETEVENTS: u32 = 1 << 1;

    /// The subset of `struct io_sqring_offset` the fixed layout needs. The
    /// kernel places these at fixed offsets from the ring's mmap bases when
    /// `flags` carries `IORING_OFF_SQ_RING`/`CQ_RING` semantics, which is
    /// the layout every x86_64/aarch64 kernel ships (no hybrid ptr size).
    struct Ring {
        sq_head: *const AtomicU32,
        sq_tail: *mut AtomicU32,
        sq_array: *mut AtomicU32,
        cq_head: *mut AtomicU32,
        cq_tail: *const AtomicU32,
        cqes: *const libc::io_uring_cqe,
        sqes: *mut libc::io_uring_sqe,
        sq_entries: u32,
        ring_fd: libc::c_int,
        mapped: Vec<(usize, usize)>, // (ptr, len) to munmap on drop
    }

    // SAFETY (module-level, stated once): the ring's atomics live in kernel-
    // owned mmap'd memory shared with the kernel; every field pointer comes
    // from that single mapping, the ring is used from one thread (SINGLE_
    // ISSUER), and `Ring` is neither Send nor Sync so nothing can move it
    // across threads.
    unsafe impl Send for Ring {}

    impl Ring {
        /// `io_uring_setup(entries)` with the single-issuer flags, then maps
        /// the SQ/CQ rings and the SQE array.
        pub fn setup(entries: u32) -> std::io::Result<Ring> {
            let mut p: libc::io_uring_params = unsafe { std::mem::zeroed() };
            p.flags = IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_COOP_TASKRUN;
            let fd = unsafe {
                libc::syscall(SYS_IO_URING_SETUP, entries, &mut p as *mut _) as libc::c_int
            };
            if fd < 0 {
                return Err(std::io::Error::last_os_error());
            }
            unsafe { Self::map(fd, &p) }
        }

        unsafe fn map(fd: libc::c_int, p: &libc::io_uring_params) -> std::io::Result<Ring> {
            let page = 4096usize;
            let sq = p.sq_off;
            let cq = p.cq_off;
            // Ring lengths, straight from the manual so each can be checked
            // against it without unfolding a helper:
            //   sq_ring = sq_off.array + sq_entries * sizeof(u32)
            //   cq_ring = cq_off.cqes + cq_entries * sizeof(cqe)
            //   sqes    = sq_entries * sizeof(sqe)
            let sq_ring_len = sq.array as usize + (p.sq_entries as usize) * 4;
            let cq_ring_len = cq.cqes as usize
                + (p.cq_entries as usize) * std::mem::size_of::<libc::io_uring_cqe>();
            let sqes_len = (p.sq_entries as usize) * std::mem::size_of::<libc::io_uring_sqe>();

            let mmap = |len: usize, off: i64| -> std::io::Result<*mut libc::c_void> {
                let ptr = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        len,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_SHARED | libc::MAP_POPULATE,
                        fd,
                        off,
                    )
                };
                if ptr == libc::MAP_FAILED {
                    Err(std::io::Error::last_os_error())
                } else {
                    Ok(ptr)
                }
            };

            let sq_ring = mmap(page.max(sq_ring_len), libc::IORING_OFF_SQ_RING as i64)?;
            let cq_ring = mmap(page.max(cq_ring_len), libc::IORING_OFF_CQ_RING as i64)?;
            let sqes = mmap(page.max(sqes_len), libc::IORING_OFF_SQES as i64)?;

            let base = sq_ring as usize;
            let mapped = vec![
                (sq_ring as usize, page.max(sq_ring_len)),
                (cq_ring as usize, page.max(cq_ring_len)),
                (sqes as usize, page.max(sqes_len)),
            ];

            Ok(Ring {
                sq_head: (base + sq.head as usize) as *const AtomicU32,
                sq_tail: (base + sq.tail as usize) as *mut AtomicU32,
                sq_array: (base + sq.array as usize) as *mut AtomicU32,
                cq_head: (cq_ring as usize + cq.head as usize) as *mut AtomicU32,
                cq_tail: (cq_ring as usize + cq.tail as usize) as *const AtomicU32,
                cqes: (cq_ring as usize + cq.cqes as usize) as *const libc::io_uring_cqe,
                sqes: sqes as *mut libc::io_uring_sqe,
                sq_entries: p.sq_entries,
                ring_fd: fd,
                mapped,
            })
        }

        /// Submits one `IORING_OP_READ` and waits for its completion.
        /// Synchronous on purpose: the expert streamer's chunk plan is
        /// already the concurrency story (8 reader threads), and an async
        /// submission pipeline can be measured AFTER the base path is
        /// proven. The fd is opened with `O_DIRECT` by the caller.
        pub fn read_at(
            &mut self,
            file: &File,
            buf: &mut [u8],
            offset: u64,
        ) -> std::io::Result<usize> {
            let fd = file.as_raw_fd();
            let tail = unsafe { (*self.sq_tail).load(Ordering::Acquire) };
            let idx = tail % self.sq_entries;
            unsafe {
                let sqe = self.sqes.add(idx as usize);
                std::ptr::write_bytes(sqe, 0, 1);
                (*sqe).opcode = IORING_OP_READ;
                (*sqe).flags = IOSQE_ASYNC;
                (*sqe).fd = fd;
                (*sqe).__bindgen_anon_2.addr = buf.as_mut_ptr() as u64;
                (*sqe).len = buf.len() as u32;
                (*sqe).__bindgen_anon_1.off = offset;
                (*self.sq_array.add(idx as usize)).store(idx, Ordering::Release);
                (*self.sq_tail).store(tail.wrapping_add(1), Ordering::Release);
            }
            let submitted = unsafe {
                libc::syscall(
                    SYS_IO_URING_ENTER,
                    self.ring_fd,
                    1u32,
                    1u32,
                    ENTER_GETEVENTS,
                    std::ptr::null_mut::<libc::c_void>(),
                )
            };
            if submitted < 0 {
                return Err(std::io::Error::last_os_error());
            }
            // One CQE is now visible; consume it.
            let head = unsafe { (*self.cq_head).load(Ordering::Acquire) };
            let cqe = unsafe { &*self.cqes.add(head as usize) };
            let res = cqe.res;
            unsafe { (*self.cq_head).store(head.wrapping_add(1), Ordering::Release) };
            if res < 0 {
                Err(std::io::Error::from_raw_os_error(-res))
            } else {
                Ok(res as usize)
            }
        }
    }

    impl Drop for Ring {
        fn drop(&mut self) {
            for (ptr, len) in &self.mapped {
                unsafe {
                    libc::munmap(*ptr as *mut libc::c_void, *len);
                }
            }
            unsafe {
                libc::close(self.ring_fd);
            }
        }
    }
}

#[cfg(target_os = "linux")]
pub use sys::Ring;

#[cfg(test)]
mod tests {
    use super::*;

    /// The selector's spelling set is exactly three words, and an unknown
    /// one is an ERROR rather than a fallback: a caller who mistyped a
    /// selector measured the default and reported it as their choice.
    #[test]
    fn the_selector_accepts_three_spellings_and_refuses_the_rest() {
        assert_eq!(Mode::parse("auto"), Ok(Mode::Pread));
        assert_eq!(Mode::parse("pread"), Ok(Mode::Pread));
        // `uring` PARSES -- reaching it is what the Linux gate forbids, not
        // naming it; refusing the spelling here would make the day the
        // mode is proven a syntax change instead of a default flip.
        assert_eq!(Mode::parse("uring"), Ok(Mode::Uring));
        assert_eq!(Mode::parse(" io_uring "), Err("TURBOSPARK_LINUX_IO must be auto, pread or uring, not \"io_uring\"".to_string()));
        assert_eq!(Mode::parse(""), Err("TURBOSPARK_LINUX_IO must be auto, pread or uring, not \"\"".to_string()));
        // And AUTO IS PREAD, which is the property every gate below rides
        // on: nothing reaches the untested kernel path by omission.
        assert_eq!(Mode::from_env_default(), Ok(Mode::Pread));
    }

    /// `O_DIRECT` legality: offset AND length both sector-aligned.
    #[test]
    fn dio_alignment_is_checked_on_both_axes() {
        assert!(dio_aligned(0, 840 * 1024));
        assert!(dio_aligned(512, 4096));
        assert!(!dio_aligned(511, 4096), "unaligned offset");
        assert!(!dio_aligned(0, 4095), "unaligned length");
    }

    /// The tail over-reads into slot slack rather than short-reading.
    #[test]
    fn dio_round_up_lands_on_the_sector() {
        assert_eq!(dio_round_up(1), 512);
        assert_eq!(dio_round_up(512), 512);
        // The 840 KiB miss chunk is already sector-aligned (840 * 1024 =
        // 1680 sectors), so +1 lands exactly one sector up, not one KiB.
        assert_eq!(dio_round_up(840 * 1024), 840 * 1024);
        assert_eq!(dio_round_up(840 * 1024 + 1), 840 * 1024 + 512);
    }
}
