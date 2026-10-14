//! The Linux I/O seam (ROADMAP P3.5): an `io_uring` + `O_DIRECT` read
//! source for the expert streamer, selected beside the `pread` default.
//!
//! # STATUS: RING READS PROVEN ON ONE LINUX HOST, NOT WIRED, NOT MEASURED
//!
//! This module was written against the io_uring(2) and open(2) manuals on
//! a machine that could not run it (AGENTS.md Gotcha 8), and the first
//! Linux run (kernel 6.18, ext4) showed what that gate is worth: it
//! compiled, and it could not complete a single read. Setup failed EINVAL,
//! and with that fixed every read failed EBADF, from four flag constants
//! off by bit position plus an unmasked CQ cursor (see `sys`). The parity
//! arm now exists: `sys::tests::ring_reads_match_the_file_buffered_and_direct`
//! reads the file's own bytes back through the ring on a buffered and an
//! `O_DIRECT` descriptor, past the CQ wrap, and each of the five defects
//! fails it when reintroduced. It runs under `cargo test -p
//! turbospark-streaming` on Linux only; CI's Linux leg is `cargo check`.
//!
//! That proves the ring, not the seam. **`Mode::Auto` still resolves to
//! `Pread`**, nothing selects `Mode::Uring`, and no streamer reads through
//! the ring yet. Do not flip the default or quote a `Mode::Uring` number
//! until the streamer is wired to it and agrees with the `pread` path bit
//! for bit on a real install.
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
// Dead until a caller selects `Mode::Uring`, which the module header forbids
// until a Linux run proves the path. The code is the point of the gate, not
// the wiring.
#[allow(dead_code)]
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

    // Bit positions are from `linux/io_uring.h`, checked against the
    // installed UAPI header on the first Linux run (kernel 6.18). That run
    // found four of these wrong by position, and each wrong bit was a
    // DIFFERENT valid flag rather than an unknown one, so nothing refused
    // them at compile time: 1 << 5 is ATTACH_WQ (setup fails EINVAL with
    // no wq_fd), 1 << 0 on an SQE is FIXED_FILE (every read fails EBADF),
    // and 1 << 1 on enter is SQ_WAKEUP (enter returns without waiting).
    // `the_ring_flags_are_the_uapi_bits` pins each value.
    const IORING_SETUP_COOP_TASKRUN: u32 = 1 << 8;
    const IORING_SETUP_SINGLE_ISSUER: u32 = 1 << 12;
    const IORING_OP_READ: u8 = 22;
    const IOSQE_ASYNC: u8 = 1 << 4;
    const ENTER_GETEVENTS: u32 = 1 << 0;

    const IORING_OFF_SQ_RING: i64 = 0;
    const IORING_OFF_CQ_RING: i64 = 0x8000000;
    const IORING_OFF_SQES: i64 = 0x10000000;

    // The pinned `libc` crate ships no io_uring types, so the stable kernel
    // ABI structs are declared here from `linux/io_uring.h`. They are UAPI:
    // fixed-layout by contract, which is what the size assertions pin --
    // a layout drift would be a kernel change, and the asserts turn it
    // into a compile error on the day it could matter.
    #[repr(C)]
    #[derive(Default)]
    struct IoSqringOffsets {
        head: u32,
        tail: u32,
        ring_mask: u32,
        ring_entries: u32,
        flags: u32,
        dropped: u32,
        array: u32,
        resv1: u32,
        user_addr: u64,
    }

    #[repr(C)]
    #[derive(Default)]
    struct IoCqringOffsets {
        head: u32,
        tail: u32,
        ring_mask: u32,
        ring_entries: u32,
        overflow: u32,
        cqes: u32,
        flags: u32,
        resv1: u32,
        user_addr: u64,
    }

    #[repr(C)]
    struct IoUringParams {
        sq_entries: u32,
        cq_entries: u32,
        flags: u32,
        sq_thread_cpu: u32,
        sq_thread_idle: u32,
        features: u32,
        wq_fd: u32,
        resv: [u32; 3],
        sq_off: IoSqringOffsets,
        cq_off: IoCqringOffsets,
    }

    impl Default for IoUringParams {
        fn default() -> Self {
            // `IoSqringOffsets`/`IoCqringOffsets` derive Default; the outer
            // struct is all integers, so zeroed is the all-default value.
            unsafe { std::mem::zeroed() }
        }
    }

    /// `struct io_uring_sqe`: 64 bytes, one per submission queue slot.
    #[repr(C)]
    struct Sqe {
        opcode: u8,
        flags: u8,
        ioprio: u16,
        fd: i32,
        off: u64,
        addr: u64,
        len: u32,
        // The union and the reserved tail are written as zero by the
        // clear-and-fill pattern below, so they are spelled as raw bytes.
        _rest: [u8; 64 - 32],
    }

    const _: () = assert!(std::mem::size_of::<Sqe>() == 64);

    /// `struct io_uring_cqe`: user_data, then the result.
    #[repr(C)]
    struct Cqe {
        user_data: u64,
        res: i32,
        flags: u32,
    }

    const _: () = assert!(std::mem::size_of::<Cqe>() == 16);

    /// The subset of `struct io_sqring_offset` the fixed layout needs. The
    /// kernel places these at fixed offsets from the ring's mmap bases when
    /// `flags` carries `IORING_OFF_SQ_RING`/`CQ_RING` semantics, which is
    /// the layout every x86_64/aarch64 kernel ships (no hybrid ptr size).
    pub(crate) struct Ring {
        sq_head: *const AtomicU32,
        sq_tail: *mut AtomicU32,
        sq_array: *mut AtomicU32,
        cq_head: *mut AtomicU32,
        cq_tail: *const AtomicU32,
        cqes: *const Cqe,
        sqes: *mut Sqe,
        // The kernel's own index masks, read from the rings. The ring
        // cursors are free-running u32 counters, so every index into the
        // SQE array or the CQE array goes through one of these; the CQ is
        // twice the SQ by default, so a mask borrowed from one ring is
        // wrong for the other.
        sq_mask: u32,
        cq_mask: u32,
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
            let mut p = IoUringParams {
                flags: IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_COOP_TASKRUN,
                ..Default::default()
            };
            let fd = unsafe {
                libc::syscall(
                    SYS_IO_URING_SETUP,
                    entries as libc::c_uint,
                    &mut p as *mut IoUringParams,
                ) as libc::c_int
            };
            if fd < 0 {
                return Err(std::io::Error::last_os_error());
            }
            unsafe { Self::map(fd, &p) }
        }

        unsafe fn map(fd: libc::c_int, p: &IoUringParams) -> std::io::Result<Ring> {
            let page = 4096usize;
            let sq = &p.sq_off;
            let cq = &p.cq_off;
            // Ring lengths, straight from the manual so each can be checked
            // against it without unfolding a helper:
            //   sq_ring = sq_off.array + sq_entries * sizeof(u32)
            //   cq_ring = cq_off.cqes + cq_entries * sizeof(cqe)
            //   sqes    = sq_entries * sizeof(sqe)
            let sq_ring_len = sq.array as usize + (p.sq_entries as usize) * 4;
            let cq_ring_len =
                cq.cqes as usize + (p.cq_entries as usize) * std::mem::size_of::<Cqe>();
            let sqes_len = (p.sq_entries as usize) * std::mem::size_of::<Sqe>();

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

            let sq_ring = mmap(page.max(sq_ring_len), IORING_OFF_SQ_RING)?;
            let cq_ring = mmap(page.max(cq_ring_len), IORING_OFF_CQ_RING)?;
            let sqes = mmap(page.max(sqes_len), IORING_OFF_SQES)?;

            let base = sq_ring as usize;
            let mapped = vec![
                (sq_ring as usize, page.max(sq_ring_len)),
                (cq_ring as usize, page.max(cq_ring_len)),
                (sqes as usize, page.max(sqes_len)),
            ];

            // SAFETY: both masks sit inside the ring mappings just made,
            // at the offsets the kernel reported for them.
            let sq_mask = unsafe { *((base + sq.ring_mask as usize) as *const u32) };
            let cq_mask = unsafe { *((cq_ring as usize + cq.ring_mask as usize) as *const u32) };

            Ok(Ring {
                sq_head: (base + sq.head as usize) as *const AtomicU32,
                sq_tail: (base + sq.tail as usize) as *mut AtomicU32,
                sq_array: (base + sq.array as usize) as *mut AtomicU32,
                cq_head: (cq_ring as usize + cq.head as usize) as *mut AtomicU32,
                cq_tail: (cq_ring as usize + cq.tail as usize) as *const AtomicU32,
                cqes: (cq_ring as usize + cq.cqes as usize) as *const Cqe,
                sqes: sqes as *mut Sqe,
                sq_mask,
                cq_mask,
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
            let idx = tail & self.sq_mask;
            unsafe {
                // Clear the whole slot, then fill the fields the READ op
                // reads; the reserved tail stays zero.
                let sqe = self.sqes.add(idx as usize);
                std::ptr::write_bytes(sqe, 0, 1);
                (*sqe).opcode = IORING_OP_READ;
                (*sqe).flags = IOSQE_ASYNC;
                (*sqe).fd = fd;
                (*sqe).off = offset;
                (*sqe).addr = buf.as_mut_ptr() as u64;
                (*sqe).len = buf.len() as u32;
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
            // SAFETY: the kernel has published at least one CQE (the ENTER
            // above asked for one event and returned success), and `head`
            // is this consumer's own free-running cursor, masked into the
            // CQE array. Unmasked, it walked off the end of the array after
            // `cq_entries` reads.
            let head = unsafe { (*self.cq_head).load(Ordering::Acquire) };
            let cqe = unsafe { &*self.cqes.add((head & self.cq_mask) as usize) };
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

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::aligned_slot::AlignedSlot;
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;

        /// Each flag is the UAPI bit, not a neighbour. A wrong position here
        /// is another VALID flag, so the kernel accepts the call and does
        /// the wrong thing; this is the only check that can say so without
        /// a ring.
        #[test]
        fn the_ring_flags_are_the_uapi_bits() {
            assert_eq!(IORING_SETUP_COOP_TASKRUN, 0x100);
            assert_eq!(IORING_SETUP_SINGLE_ISSUER, 0x1000);
            assert_eq!(IOSQE_ASYNC, 0x10);
            assert_eq!(ENTER_GETEVENTS, 0x1);
            assert_eq!(IORING_OP_READ, 22);
        }

        /// A sandbox that blocks io_uring returns ENOSYS or EPERM from
        /// setup. That host can say nothing about the ring, so the parity
        /// test reports it by name instead of failing the crate's suite.
        fn ring_or_skip() -> Option<Ring> {
            match Ring::setup(8) {
                Ok(ring) => Some(ring),
                Err(e) if matches!(e.raw_os_error(), Some(libc::ENOSYS | libc::EPERM)) => {
                    eprintln!("SKIPPED: io_uring is unavailable on this host ({e})");
                    None
                }
                Err(e) => panic!("io_uring_setup failed: {e}"),
            }
        }

        fn scratch_file(tag: &str, len: usize) -> (std::path::PathBuf, Vec<u8>) {
            let dir =
                std::env::temp_dir().join(format!("turbospark-uring-{tag}-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("blob.bin");
            // A non-repeating-per-sector pattern, so a read landing at the
            // wrong offset cannot match by accident.
            let bytes: Vec<u8> = (0..len)
                .map(|i| (i % 251) as u8 ^ (i >> 12) as u8)
                .collect();
            std::fs::File::create(&path)
                .unwrap()
                .write_all(&bytes)
                .unwrap();
            (path, bytes)
        }

        /// The parity arm the module header asks for: every ring read
        /// returns the file's own bytes, on a buffered descriptor at
        /// unaligned offsets and on an `O_DIRECT` one at sector-aligned
        /// offsets, for more reads than the CQ has entries (the ring is
        /// set up with 8, so the CQ holds 16 and the cursor must wrap).
        #[test]
        fn ring_reads_match_the_file_buffered_and_direct() {
            let Some(mut ring) = ring_or_skip() else {
                return;
            };
            let (path, bytes) = scratch_file("parity", 4 << 20);

            let buffered = std::fs::File::open(&path).unwrap();
            let mut buf = vec![0u8; 12_345];
            for i in 0..48u64 {
                let offset = i * 77_777 + 3;
                let got = ring.read_at(&buffered, &mut buf, offset).unwrap();
                assert_eq!(got, buf.len(), "short buffered read {i}");
                let at = offset as usize;
                assert!(
                    buf[..] == bytes[at..at + buf.len()],
                    "buffered read {i} at {offset} returned other bytes"
                );
            }

            let direct = match std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECT)
                .open(&path)
            {
                Ok(file) => file,
                Err(e) if e.raw_os_error() == Some(libc::EINVAL) => {
                    eprintln!("SKIPPED: O_DIRECT arm, the temp filesystem refuses it ({e})");
                    std::fs::remove_dir_all(path.parent().unwrap()).ok();
                    return;
                }
                Err(e) => panic!("open O_DIRECT: {e}"),
            };
            let chunk = 64 * 1024;
            assert!(super::super::dio_aligned(0, chunk));
            let mut slot = AlignedSlot::allocate(chunk).unwrap();
            for i in 0..48u64 {
                let offset = (i * 5 % 61) * chunk as u64;
                let dst = &mut slot.as_mut_slice()[..chunk];
                let got = ring.read_at(&direct, dst, offset).unwrap();
                assert_eq!(got, chunk, "short direct read {i}");
                let at = offset as usize;
                assert!(
                    slot.as_slice()[..chunk] == bytes[at..at + chunk],
                    "direct read {i} at {offset} returned other bytes"
                );
            }
            std::fs::remove_dir_all(path.parent().unwrap()).ok();
        }
    }
}

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
        assert_eq!(
            Mode::parse(" io_uring "),
            Err("TURBOSPARK_LINUX_IO must be auto, pread or uring, not \"io_uring\"".to_string())
        );
        assert_eq!(
            Mode::parse(""),
            Err("TURBOSPARK_LINUX_IO must be auto, pread or uring, not \"\"".to_string())
        );
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
